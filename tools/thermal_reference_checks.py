"""
thermal_reference_checks.py -- the thermal reference (tools/thermal_reference.py) on the production path, before the
effect-size runs (2026-10-18). Fluxes at Earth of an MeV-plane configuration, each run in a fresh process:
  1. kernel 'rest' against production without its thermal resonance terms: the plumbing (to rounding);
  2. slow relics (lightest mass 0.1 eV): the reference against production;
  3. convergence: 24 nodes per half-panel and 2 sub-steps against 48 and 4;
and the time of each reference run.

  python tools/thermal_reference_checks.py <config.json> [--workers N]
"""
import argparse
import copy
import json
import os
import sys
import time
from multiprocessing import get_context

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTS = [{"m_Zd_eV": 3.16e4, "g_D_trace": 2e-3}, {"m_Zd_eV": 4.22e6, "g_D_trace": 1.35e-3}]


def flux(args):
    """(label, point, m_lightest, seconds, E, flux summed over flavors) for one run."""
    path, mode, pt, m_l, opts = args
    sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
    os.environ.pop("NULRI_FLUX_CACHE", None)
    import numpy as np
    import dp_si
    if mode == "prod_rest":
        orig = dp_si.propagate_si

        def wrapped(*a, **k):
            k["thermal"] = False
            return orig(*a, **k)
        dp_si.propagate_si = wrapped
    elif mode.startswith("ref"):
        import thermal_reference
        thermal_reference.install(**opts)
    import dark_portal_solver as dps
    from analysis import config as config_mod
    raw = copy.deepcopy(json.load(open(path)))
    raw["solver"].setdefault("cosmology", {})["m_lightest_eV"] = m_l
    raw["parameters"].pop("m_lightest_eV", None)
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, os.path.abspath(path))
    m = cfg.model()
    v = {p.name: p.start for p in cfg.parameters}
    v.update(pt)
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    t = time.time()
    F = np.asarray(dps.propagate_weighted(E, params))
    dt = time.time() - t
    F = F.sum(axis=tuple(range(1, F.ndim))) if F.ndim > 1 else F
    return mode + str(sorted(opts.items())), json.dumps(pt), m_l, dt, np.asarray(E).tolist(), F.tolist()


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    path = os.path.abspath(a.config)
    fine = dict(kernel="thermal", ng=48, n_sub=4)
    base = dict(kernel="thermal", ng=24, n_sub=2)
    rest = dict(kernel="rest", ng=24, n_sub=1)
    tasks = []
    for pt in POINTS:
        tasks.append((path, "ref", pt, 1e-6, base))                  # first: builds the extended tables
    for pt in POINTS:
        for m_l in (1e-6, 1e-3, 0.03):
            tasks += [(path, "prod_rest", pt, m_l, {}), (path, "ref_rest", pt, m_l, rest)]
        tasks += [(path, "prod", pt, 0.1, {}), (path, "ref", pt, 0.1, base), (path, "ref_fine", pt, 1e-6, fine)]
    ctx = get_context("spawn")
    with ctx.Pool(len(POINTS), maxtasksperchild=1) as pool:          # the tables, once per mediator mass
        res = pool.map(flux, tasks[:len(POINTS)], chunksize=1)
    print("PROGRESS 10% | extended tables built", flush=True)
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        res += pool.map(flux, tasks[len(POINTS):], chunksize=1)
    print("PROGRESS 100% | runs done", flush=True)
    got = {(lab.split("[")[0], p, m): (dt, np.array(E), np.array(F)) for lab, p, m, dt, E, F in res}

    def diff(k1, k2):
        dt1, E, F1 = got[k1]
        F2 = got[k2][2]
        band = (E > 1e13) & (E < 1e16)
        return float(np.max(np.abs(F1[band] - F2[band])) / np.max(np.abs(F2[band])))

    for pt in POINTS:
        p = json.dumps(pt)
        print(f"== {p}")
        for m_l in (1e-6, 1e-3, 0.03):
            print(f"  plumbing, m_l {m_l:g}: max |ref(rest) - prod(no thermal)| / max = "
                  f"{diff(('ref_rest', p, m_l), ('prod_rest', p, m_l)):.2e}")
        print(f"  slow relics (0.1 eV): max |ref - prod| / max = {diff(('ref', p, 0.1), ('prod', p, 0.1)):.2e}")
        print(f"  convergence at 1e-6 eV: max |ref(24, 2) - ref(48, 4)| / max = "
              f"{diff(('ref', p, 1e-6), ('ref_fine', p, 1e-6)):.2e}")
        print(f"  time per transport: ref {got[('ref', p, 1e-6)][0]:.0f} s, ref_fine {got[('ref_fine', p, 1e-6)][0]:.0f} s, "
              f"prod {got[('prod', p, 0.1)][0]:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
