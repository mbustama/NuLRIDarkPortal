"""
compare_cosmo_transport.py -- the dark portal at eps' = 0 around 1 keV, where the solver switches the cosmological
phase from the dark-portal treatment (resonance only; below SI_COSMOLOGY_MIN_M_EV) to dp_si's full transport (above):
the flux at Earth (summed over flavors, over the free-streaming one, and the flavor fractions at 100 TeV) with each
treatment forced (dark_portal.cosmo_transport 'dark_portal' or 'si') at the same masses and couplings, to see whether
the step the joined plane shows at 1 keV (2026-10-22) is physics or the seam between the two. Writes
output/analysis/cosmo_transport_seam.npz and prints a table.
  python tools/compare_cosmo_transport.py [--config <dp ul config>] [--workers N]
"""
import argparse
import copy
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MASSES = (1e2, 3e2, 1e3, 3e3, 1e4)
COUPLINGS = (10 ** -2.4, 10 ** -2.0, 10 ** -1.7)
MODES = ("dark_portal", "si")
OUT = os.path.join(REPO, "output", "analysis", "cosmo_transport_seam.npz")


def flux(args):
    path, mode, M, g = args
    os.environ["NUMBA_NUM_THREADS"] = "2"
    os.environ.pop("NULRI_FLUX_CACHE", None)
    sys.path[:0] = [os.path.join(REPO, "src"), REPO]
    import dark_portal_solver as dps
    from analysis import config as config_mod
    raw = copy.deepcopy(json.load(open(path)))
    if mode != "free":
        raw["solver"]["dark_portal"]["cosmo_transport"] = mode
    raw.pop("exact_profile", None)
    raw["parameters"].pop("m_lightest_eV", None)
    cfg = config_mod.from_dict(raw, path)
    m = cfg.model()
    v = {p.name: p.start for p in cfg.parameters}
    pl = raw["planes"][0]
    v.update({pl["x"]: M, pl["y"]: 1e-30 if mode == "free" else g})
    try:
        E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
        F = np.asarray(dps.propagate_weighted(E, params, sources=[np.array([1.0, 2.0, 0.0]) / 3.0])[0])
        return mode, M, g, np.asarray(E).tolist(), F.reshape(len(E), -1).tolist(), None
    except Exception as e:
        return mode, M, g, None, None, repr(e)[:300]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=os.path.join(REPO, "analysis/configs/campaign_ml/now_ic26/dp_e_ul.json"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    tasks = [(a.config, mode, M, g) for mode in MODES for M in MASSES for g in COUPLINGS]
    tasks += [(a.config, "free", M, 0.0) for M in MASSES]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(flux, tasks, chunksize=1)
    got = {(mode, M, g): (E, F, err) for mode, M, g, E, F, err in res}
    np.savez(OUT, **{f"{mode}_{M:g}_{g:.4g}": np.array(F) for (mode, M, g), (E, F, err) in got.items() if F},
             E=np.array(next(E for E, F, err in got.values() if E)))
    print(f"{'M (eV)':>8} {'log g':>6} | {'dark_portal: total/free, f_e,f_mu at 100 TeV':>44} | {'si: same':>36}")
    for M in MASSES:
        E, F0, _ = got[("free", M, 0.0)]
        if F0 is None:
            print(f"{M:8g}: free run failed")
            continue
        E = np.array(E)
        F0 = np.array(F0)
        k = int(np.argmin(abs(E - 1e14)))
        band = (E > 1e13) & (E < 1e16)
        for g in COUPLINGS:
            row = []
            for mode in MODES:
                _, F, err = got[(mode, M, g)]
                if F is None:
                    row.append(f"FAILED {err[:30]}")
                    continue
                F = np.array(F)
                tot = F.sum(axis=1) / F0.sum(axis=1)
                f = F[k] / F[k].sum() if F.ndim == 2 and F.shape[1] >= 3 else [np.nan] * 3
                row.append(f"min {tot[band].min():.3f} mean {tot[band].mean():.3f}, f {f[0]:.3f} {f[1]:.3f}")
            print(f"{M:8g} {np.log10(g):6.2f} | {row[0]:>44} | {row[1]:>36}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
