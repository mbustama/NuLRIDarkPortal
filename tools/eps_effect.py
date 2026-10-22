"""
eps_effect.py -- what the dark portal's kinetic mixing eps' does to the neutrinos arriving at Earth (2026-10-22): at an
ultralight mediator mass and a fixed g_D sum|U|^2, the flavor fractions at 100 TeV (and over 10 TeV - 1 PeV) as eps'
grows, for the three textures, from the potential planes' configurations (now_ic26, pion-decay sources; the potential's
size goes with g_D eps'). Writes output/analysis/eps_effect.npz; --plot draws a quick look.
  python tools/eps_effect.py [--mass 1e-20] [--split -16] [--workers N]
"""
import argparse
import copy
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXTURES = ("e", "mu", "tau")
LOG_EPS = np.arange(-40.0, -19.9, 1.0)
OUT = os.path.join(REPO, "output", "analysis", "eps_effect.npz")


def flux(args):
    path, log_p = args
    os.environ["NUMBA_NUM_THREADS"] = "2"
    os.environ.pop("NULRI_FLUX_CACHE", None)
    sys.path[:0] = [os.path.join(REPO, "src"), REPO]
    import dark_portal_solver as dps
    from analysis import config as config_mod
    raw = copy.deepcopy(json.load(open(path)))
    raw.pop("exact_profile", None)
    raw["parameters"].pop("m_lightest_eV", None)
    cfg = config_mod.from_dict(raw, path)
    m = cfg.model()
    v = {p.name: p.start for p in cfg.parameters}
    pl = raw["planes"][0]
    v.update({pl["x"]: MASS, pl["y"]: 10.0 ** log_p})
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    return path, log_p, np.asarray(E).tolist(), np.asarray(dps.propagate_weighted(E, params)).tolist()


MASS = 1e-20


def main(argv=None):
    global MASS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mass", type=float, default=1e-20)
    ap.add_argument("--split", type=int, default=-16, help="log10 g_D sum|U|^2: one of -24, -16, -8")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    MASS = a.mass
    tag = {-24: "x1m24", -16: "x1m16", -8: "x1m08"}[a.split]
    cfgs = {t: os.path.join(REPO, f"analysis/configs/campaign_ml/now_ic26/dp_{t}_pot_{tag}_pos.json") for t in TEXTURES}
    tasks = [(cfgs[t], le + a.split) for t in TEXTURES for le in LOG_EPS] + [(cfgs["e"], -60.0)]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1, initializer=_set_mass, initargs=(MASS,)) as pool:
        res = pool.map(flux, tasks, chunksize=1)
    E = np.array(res[0][2])
    out = {"E": E, "log_eps": LOG_EPS, "mass": MASS, "split": a.split}
    for t in TEXTURES:
        out[t] = np.array([next(F for p, lp, _, F in res if p == cfgs[t] and abs(lp - (le + a.split)) < 1e-9)
                           for le in LOG_EPS])
    out["none"] = np.array(next(F for p, lp, _, F in res if lp == -60.0))
    np.savez(OUT, **out)
    k = int(np.argmin(abs(E - 1e14)))
    f0 = out["none"][k] / out["none"][k].sum()
    print(f"mass {MASS:g} eV, g_D sum|U|^2 = 1e{a.split}; no effect: f = {f0.round(3)}")
    for t in TEXTURES:
        f = out[t][:, k, :] / out[t][:, k, :].sum(axis=1, keepdims=True)
        print(t, " ".join(f"{le:.0f}:{fe:.2f}/{fm:.2f}" for le, (fe, fm, ft) in zip(LOG_EPS, f)))
    return 0


def _set_mass(m):
    global MASS
    MASS = m


if __name__ == "__main__":
    sys.exit(main())
