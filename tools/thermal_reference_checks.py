"""
thermal_reference_checks.py -- dp_si's thermal relics in every collision term (dark_portal.thermal_products,
2026-10-18) against the reference (tools/thermal_reference.py) at the analysis grid: the fluxes at Earth of an
MeV-plane configuration at two points and three lightest masses, each run in a fresh process -- production with the
switch on ('full'), production without it ('old': the resonance alone), the reference ('ref'); then the time of a
second transport (a process's tables built) with the switch on and off.

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
MLS = (1e-6, 1e-3, 0.03)


def flux(args):
    """(mode, point, m_lightest, seconds, E, flux summed over flavors); modes ending in '_time' run a second
    transport at 1.1 times the coupling and time that one."""
    path, mode, pt, m_l, threads = args
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    os.environ.pop("NULRI_FLUX_CACHE", None)
    sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
    import numpy as np
    if mode == "ref":
        import thermal_reference
        thermal_reference.install("thermal", 1, 8)
    import dark_portal_solver as dps
    from analysis import config as config_mod

    def run(point):
        raw = copy.deepcopy(json.load(open(path)))
        raw["solver"].setdefault("cosmology", {})["m_lightest_eV"] = m_l
        raw["solver"]["dark_portal"]["thermal_products"] = mode.startswith("full")
        raw["parameters"].pop("m_lightest_eV", None)
        raw.pop("exact_profile", None)
        cfg = config_mod.from_dict(raw, os.path.abspath(path))
        m = cfg.model()
        v = {p.name: p.start for p in cfg.parameters}
        v.update(point)
        E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
        t = time.time()
        F = np.asarray(dps.propagate_weighted(E, params))
        return time.time() - t, E, F

    dt, E, F = run(pt)
    if mode.endswith("_time"):
        dt, E, F = run(dict(pt, g_D_trace=1.1 * pt["g_D_trace"]))
    F = F.sum(axis=tuple(range(1, F.ndim))) if F.ndim > 1 else F
    return mode, json.dumps(pt), m_l, dt, np.asarray(E).tolist(), F.tolist()


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    path = os.path.abspath(a.config)
    tasks = [(path, mode, pt, m_l, 2) for mode in ("ref", "full", "old") for pt in POINTS for m_l in MLS]
    ctx = get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(flux, tasks, chunksize=1)
    print("PROGRESS 80% | fluxes done", flush=True)
    timing = [(path, mode, POINTS[1], m_l, 4) for mode in ("full_time", "old_time") for m_l in (1e-6, 0.03)]
    with ctx.Pool(4, maxtasksperchild=1) as pool:
        res += pool.map(flux, timing, chunksize=1)
    print("PROGRESS 100% | timing done", flush=True)
    got = {(mode, p, m): (dt, np.array(E), np.array(F)) for mode, p, m, dt, E, F in res}

    def rel(k1, k2):
        _, E, F1 = got[k1]
        F2 = got[k2][2]
        band = (E > 1e13) & (E < 1e16) & (F2 > 0)
        return float(np.max(np.abs(F1[band] / F2[band] - 1.0)))

    for pt in POINTS:
        p = json.dumps(pt)
        print(f"== {p}")
        for m_l in MLS:
            print(f"  m_l {m_l:g}: max |full / ref - 1| = {rel(('full', p, m_l), ('ref', p, m_l)):.2e}, "
                  f"max |old / ref - 1| = {rel(('old', p, m_l), ('ref', p, m_l)):.2e}; "
                  f"ref {got[('ref', p, m_l)][0]:.0f} s")
    p = json.dumps(POINTS[1])
    for m_l in (1e-6, 0.03):
        print(f"time per transport (4 threads), m_l {m_l:g}: full {got[('full_time', p, m_l)][0]:.1f} s, "
              f"old {got[('old_time', p, m_l)][0]:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
