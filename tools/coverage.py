"""
tools/coverage.py -- coverage of the limits with pseudo-experiments (Mauricio's go, 2026-11-25: minimal tests, a pilot
first to measure the cost).

For one plane configuration of today's first data set (IceCube's released MESE flavor likelihood and its spectrum
segments; no event samples), one mass M of the plane's grid and true couplings g_t:

  1. Lightest-mass grid. At every coupling g of the plane's column at M and every lightest mass m of ML_GRID, chi^2
     is fitted over the cheap nuisances (normalizations, indices, compositions: no new solver run), m held. Each
     (g, m) needs the solver once (every index node); the runs go to the flux cache (NULRI_FLUX_CACHE) and every
     pseudo-experiment reuses them. chi^2(g) = min over the grid. The exact profile of the campaign (exact_profile.py)
     moves m continuously, which would need new solver runs for every pseudo-experiment.
  2. Check of that grid on the real data: chi^2(g) from the grid against the stored exact profile of the plane
     (--plane-json, the exported plane); the result reports the largest difference. The pseudo-experiments are only
     as good as this check.
  3. Truth at (M, g_t): the real data's fit there (grid minimum), its nuisances theta_t; the true composition f_t
     (Objective.fractions) and the true segments C_t (Objective.spectrum_segments).
  4. Pseudo-experiments, as in NuFlavorRatiosSteriles (its appendix "Pseudo-experiments"): the measured composition
     f_obs = f_t + (f~ - f_c), f~ drawn from the released TS grid (probability exp(-TS/2) on a fine mesh of the flavor
     triangle), f_c its best node, and the grid moved rigidly so that its best node sits at f_obs (surfaces.recentred).
     Each segment D_k = C_t,k + z_k s_k, z_k standard normal, s_k the measurement's lower error for z_k > 0 and its
     upper one otherwise (the errors that chi^2 then applies; spectra.chi2), floored at 0, with the measured error
     widths around D_k.
  5. Test statistic of the upper limit at g_t: q = chi^2(g_t) - min_g chi^2(g) if the best coupling of the column lies
     below g_t, else 0 (the paper reads upper limits only). The limit covers g_t when q < 3.84; coverage = the
     fraction of pseudo-experiments that do, with its binomial error.

Workers are spawned processes (the solver is not fork-safe); each builds the objective once.

Usage (on Tycho, NULRI_FLUX_CACHE set to the job's own folder):
  python tools/coverage.py CONFIG --mass M_eV --true-log10g -3.7 [-3.4 ...] --toys N --workers W
                           --plane-json PATH --out OUT.json [--seed S] [--ml-grid m1,m2,...]
"""
import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from analysis import config as acfg            # noqa: E402
from analysis import run                       # noqa: E402
from analysis import surfaces                  # noqa: E402
from analysis.fitting import fit              # noqa: E402

ML = "m_lightest_eV"
#: The lightest masses (eV) at which chi^2 is fitted; the profile over m is the minimum over them.
ML_GRID = (1e-6, 1e-3, 3e-3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
THRESHOLD = 3.84
_W = {}


# ----------------------------------------------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------------------------------------------
def _init(path, n_points):
    global ML_GRID
    if os.environ.get("COVERAGE_ML_GRID"):
        ML_GRID = tuple(float(x) for x in os.environ["COVERAGE_ML_GRID"].split(","))
    cfg = acfg.load(path)
    # the forecasts have no data: the Asimov data set, as the campaign's runs made it (analysis/run.py)
    dc = run.asimov_centre(cfg) if run.asimov_only(cfg) else None
    obj = cfg.objective(model=cfg.model(cache_size=n_points + 16), data_centre=dc)
    _W.update(cfg=cfg, obj=obj, pset=cfg.parameter_set(),
              settings=dataclasses.replace(cfg.fit, fd_central=True, n_random=0, polish=False),
              spectrum=dict(obj.spectrum) if obj.spectrum is not None else None, plane=cfg.planes[0])


def _fit_point(held, start):
    r = fit(_W["obj"], _W["pset"].holding(held), starts=[start] if start else None, settings=_W["settings"])
    return float(r.chi2), dict(r.values)


def _real_task(task):
    """The real data's fit at one (g, m): fills the flux cache."""
    held, start = task
    t0 = time.perf_counter()
    chi2, values = _fit_point(held, start)
    return dict(held=held, chi2=chi2, values=values, seconds=time.perf_counter() - t0)


def _set_data(f_obs, seg_values):
    """The pseudo-experiment's data in this worker's objective."""
    obj = _W["obj"]
    obj.data = surfaces.recentred(obj.surface, *f_obs)
    obj.data_centre = tuple(f_obs)
    spec = dict(_W["spectrum"])
    D0 = spec["value"]
    s_dn, s_up = D0 - spec["lo"], spec["hi"] - D0
    spec.update(value=seg_values, lo=np.maximum(seg_values - s_dn, 0.0), hi=seg_values + s_up)
    obj.spectrum = spec


def _toy_task(task):
    """One pseudo-experiment: chi^2 per coupling (minimum over the lightest-mass grid), q at the true coupling."""
    k, f_obs, seg_values, couplings, M, g_true, starts = task
    t0 = time.perf_counter()
    _set_data(f_obs, seg_values)
    plane = _W["plane"]
    chi = []
    for g in couplings:
        best = min(_fit_point({plane.x: M, plane.y: g, ML: m}, starts.get((g, m)))[0] for m in ML_GRID)
        chi.append(best)
    chi = np.array(chi)
    i_true = int(np.argmin(np.abs(np.log10(couplings) - np.log10(g_true))))
    i_min = int(np.argmin(chi))
    q = float(chi[i_true] - chi[i_min]) if couplings[i_min] <= couplings[i_true] else 0.0
    return dict(toy=k, q=q, g_hat=float(couplings[i_min]), chi2=chi.tolist(), f_obs=list(map(float, f_obs)),
                seconds=time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------------------------------
# Pseudo-data
# ----------------------------------------------------------------------------------------------------------------
def flavor_draws(surface_name, n, rng):
    """n draws of f~ - f_c (offsets in (f_e, f_mu)) from the released TS grid: probability exp(-TS/2) on a fine mesh
    of the triangle (spacing 0.005), each draw jittered uniformly within its cell."""
    p = surfaces.surface(surface_name)
    step = 0.005
    fe, fmu = np.meshgrid(np.arange(0.0, 1.0 + 1e-9, step), np.arange(0.0, 1.0 + 1e-9, step), indexing="ij")
    ok = fe + fmu <= 1.0 + 1e-12
    fe, fmu = fe[ok], fmu[ok]
    TS = np.array([surfaces.chi2(a, b, p) for a, b in zip(fe, fmu)])
    w = np.exp(-0.5 * (TS - np.nanmin(TS)))
    w[~np.isfinite(w)] = 0.0
    idx = rng.choice(len(fe), size=n, p=w / w.sum())
    jitter = rng.uniform(-0.5 * step, 0.5 * step, size=(n, 2))
    return np.column_stack([fe[idx], fmu[idx]]) + jitter - np.array([p["true_mu_x"], p["true_mu_y"]])


def segment_draws(C_true, spec, n, rng):
    """n draws of the segments around the true ones (module docstring, step 4)."""
    D0 = spec["value"]
    s_dn, s_up = D0 - spec["lo"], spec["hi"] - D0
    z = rng.standard_normal((n, len(C_true)))
    return np.maximum(C_true[None, :] + z * np.where(z > 0, s_dn, s_up), 0.0)


# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--mass", type=float, required=True, help="M (eV), snapped to the plane's grid")
    ap.add_argument("--true-log10g", type=float, nargs="+", required=True)
    ap.add_argument("--toys", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--plane-json", default=None, help="the exported plane (data/plane_*.json) for the grid check")
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--ml-grid", default=None, help="comma-separated lightest masses (eV); default ML_GRID")
    ap.add_argument("--ml-log", type=float, nargs=3, default=None, metavar=("LO", "HI", "N"),
                    help="the lightest masses log-spaced from LO to HI eV (N values), plus 1e-6 eV")
    ap.add_argument("--log10g-range", type=float, nargs=2, default=None,
                    help="only the column's couplings in this range (plus its smallest, the no-effect reference)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not os.environ.get("NULRI_FLUX_CACHE"):
        raise SystemExit("NULRI_FLUX_CACHE must name the job's flux-cache folder: the workers share the solver runs")
    global ML_GRID
    if a.ml_grid:
        ML_GRID = tuple(float(x) for x in a.ml_grid.split(","))
    if a.ml_log:
        ML_GRID = (1e-6,) + tuple(float(x) for x in np.geomspace(a.ml_log[0], a.ml_log[1], int(a.ml_log[2])))

    cfg = acfg.load(a.config)
    plane = cfg.planes[0]
    M = float(plane.x_grid[np.argmin(np.abs(np.log10(plane.x_grid) - np.log10(a.mass)))])
    couplings = np.asarray(plane.y_grid, dtype=float)
    if a.log10g_range:
        lg = np.log10(couplings)
        keep = (lg >= a.log10g_range[0] - 1e-9) & (lg <= a.log10g_range[1] + 1e-9)
        keep[0] = True
        couplings = couplings[keep]
    n_points = len(couplings) * len(ML_GRID)
    out = dict(config=a.config, mass_eV=M, couplings=couplings.tolist(), ml_grid=list(ML_GRID), seed=a.seed,
               workers=a.workers, threshold=THRESHOLD, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    ctx = mp.get_context("spawn")
    # the workers read the lightest-mass grid from the module: pass it through the environment of the spawned ones
    os.environ["COVERAGE_ML_GRID"] = ",".join(repr(m) for m in ML_GRID)
    # build this process' objective first: a failing initializer fails here, at once, instead of in workers
    # that the pool respawns forever
    _init(a.config, n_points)
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.config, n_points)) as pool:
        # 1. the real data on the grid (fills the flux cache)
        t0 = time.perf_counter()
        tasks = [({plane.x: M, plane.y: float(g), ML: m}, None) for g in couplings for m in ML_GRID]
        real = pool.map(_real_task, tasks, chunksize=1)
        out["real_seconds"] = time.perf_counter() - t0
        starts = {(r["held"][plane.y], r["held"][ML]): r["values"] for r in real}
        chi_real = np.array([min(r["chi2"] for r in real if r["held"][plane.y] == g) for g in couplings])
        out["real_chi2"] = chi_real.tolist()
        # 2. the grid against the stored exact profile
        if a.plane_json:
            d = json.load(open(a.plane_json))
            gx = np.asarray(d["grids"][0], float)
            col = np.asarray(d["chi2"], float)[int(np.argmin(np.abs(np.log10(gx) - np.log10(M))))]
            yg = np.asarray(d["grids"][1], float)
            col = np.array([col[int(np.argmin(np.abs(np.log10(yg) - np.log10(g))))] for g in couplings])
            out["exact_chi2"] = col.tolist()
            out["grid_minus_exact_max"] = float(np.nanmax(chi_real - col))
            out["grid_minus_exact_min"] = float(np.nanmin(chi_real - col))
        _write(a.out, out)
        # 3-5. per true coupling: truth, pseudo-data, fits (--toys 0: the real data and the check only)
        out["truths"] = []
        for lg in (a.true_log10g if a.toys > 0 else []):
            g_t = float(couplings[np.argmin(np.abs(np.log10(couplings) - lg))])
            best = min((r for r in real if r["held"][plane.y] == g_t), key=lambda r: r["chi2"])
            theta = best["values"]
            obj = _W["obj"]
            f_t = np.asarray(obj.fractions(theta), float)[:2]
            C_t = np.asarray(obj.spectrum_segments(theta), float)
            rng = np.random.default_rng([a.seed, int(round(1000 * (lg + 20.0)))])     # entropy must be >= 0
            df = flavor_draws(obj.surface, a.toys, rng)
            segs = segment_draws(C_t, _W["spectrum"], a.toys, rng)
            t1 = time.perf_counter()
            toys = pool.map(_toy_task, [(k, f_t + df[k], segs[k], couplings, M, g_t, starts) for k in range(a.toys)],
                            chunksize=1)
            q = np.array([t["q"] for t in toys])
            cov = float(np.mean(q < THRESHOLD))
            out["truths"].append(dict(log10g_true=float(np.log10(g_t)), theta=theta, f_true=f_t.tolist(),
                                      C_true=C_t.tolist(), coverage=cov,
                                      coverage_err=float(np.sqrt(cov * (1 - cov) / len(q))),
                                      toys_seconds=time.perf_counter() - t1, toys=toys))
            _write(a.out, out)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(a.out, out)


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
