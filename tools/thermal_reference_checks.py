"""
thermal_reference_checks.py -- dp_si's thermal relics in every collision term (dark_portal.thermal_products,
2026-10-18; secret_interactions.thermal_products, 2026-10-21) against the reference (tools/thermal_reference.py) at the
analysis grid: the fluxes at Earth of an MeV-plane configuration (the dark portal's, or a secret interaction's) at two
points and three lightest masses, each run in a fresh process -- production with the switch on ('full'), production
without it ('old': the dark portal's resonance alone, the secret interactions' relics at rest), the reference
('ref'); then the time of a second transport (a process's tables built) with the switch on and off. Writes
output/analysis/thermal_checks[_<config name>].npz.

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
MG = [(3.16e4, 2e-3), (4.22e6, 1.35e-3)]       # (mediator mass, coupling) of the two points
POINTS = []                                   # set in main from the plane's axes
MLS = (1e-6, 1e-3, 0.03)


def flux(args):
    """(mode, point, m_lightest, seconds, E, flux summed over flavors); modes ending in '_time' run a second
    transport at 1.1 times the coupling and time that one."""
    path, mode, pt, m_l, threads = args
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    os.environ.pop("NULRI_FLUX_CACHE", None)
    sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
    import numpy as np
    if mode.startswith("ref"):
        import thermal_reference
        thermal_reference.install("thermal", *((2, 16) if mode == "ref_fine" else (1, 8)))
    import dark_portal_solver as dps
    from analysis import config as config_mod

    def run(point):
        raw = copy.deepcopy(json.load(open(path)))
        raw["solver"].setdefault("cosmology", {})["m_lightest_eV"] = m_l
        si = raw["solver"].get("secret_interactions", {})
        (si if si.get("enabled") else raw["solver"]["dark_portal"])["thermal_products"] = mode.startswith("full")
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
        y = json.load(open(path))["planes"][0]["y"]
        dt, E, F = run(dict(pt, **{y: 1.1 * pt[y]}))
    F = F.sum(axis=tuple(range(1, F.ndim))) if F.ndim > 1 else F
    return mode, json.dumps(pt), m_l, dt, np.asarray(E).tolist(), F.tolist()


def main(argv=None):
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    path = os.path.abspath(a.config)
    global POINTS
    pl = json.load(open(path))["planes"][0]
    POINTS = [{pl["x"]: m, pl["y"]: g} for m, g in MG]
    tasks = [(path, mode, pt, m_l, 2) for mode in ("ref", "full", "old") for pt in POINTS for m_l in MLS]
    tasks.insert(0, (path, "ref_fine", POINTS[0], MLS[0], 2))     # the reference's convergence (2 sub-steps, 16 nodes)
    ctx = get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(flux, tasks, chunksize=1)
    print("PROGRESS 80% | fluxes done", flush=True)
    timing = [(path, mode, POINTS[1], m_l, 4) for mode in ("full_time", "old_time") for m_l in (1e-6, 0.03)]
    with ctx.Pool(4, maxtasksperchild=1) as pool:
        res += pool.map(flux, timing, chunksize=1)
    print("PROGRESS 100% | timing done", flush=True)
    got = {(mode, p, m): (dt, np.array(E), np.array(F)) for mode, p, m, dt, E, F in res}
    idx = {json.dumps(pt): i for i, pt in enumerate(POINTS)}
    name = json.load(open(path))["name"]
    np.savez(os.path.join(REPO, "output", "analysis", "thermal_checks.npz" if name.startswith("dp_")
                          else f"thermal_checks_{name}.npz"),
             **{f"{mode}_{idx[p]}_{m:g}": np.vstack([E, F]) for (mode, p, m), (dt, E, F) in got.items()})

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
    p = json.dumps(POINTS[0])
    print(f"reference convergence at {p}, m_l {MLS[0]:g}: max |ref / ref_fine - 1| = "
          f"{rel(('ref', p, MLS[0]), ('ref_fine', p, MLS[0])):.2e}, max |full / ref_fine - 1| = "
          f"{rel(('full', p, MLS[0]), ('ref_fine', p, MLS[0])):.2e}")
    p = json.dumps(POINTS[1])
    for m_l in (1e-6, 0.03):
        print(f"time per transport (4 threads), m_l {m_l:g}: full {got[('full_time', p, m_l)][0]:.1f} s, "
              f"old {got[('old_time', p, m_l)][0]:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
