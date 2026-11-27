"""
tools/coverage_limits.py -- coverage of the 95% C.L. limits with pseudo-experiments, through the limits' own code
(Mauricio's decisions, 2026-11-25; paper_planning.md): for one plane configuration of today's first data set
(IceCube's released MESE flavor likelihood and its spectrum segments), one mass M and one true coupling g_t (the
limit at M):

  1. Real data. The exact profile (ExactProfiler.profile_global: the configuration's lightest-mass scan, then fits
     from its local minima -- what the plane runs do at each point) at the window's couplings (the plane's grid from
     1 decade below g_t to 0.4 above), at the plane's smallest coupling (no effect) and at g_t. This fills the job's
     flux cache, which every pseudo-experiment reuses. The truth is the profile at (M, g_t): its values theta_t, the
     true composition f_t (Objective.fractions) and the true segments C_t (Objective.spectrum_segments).
  2. Pseudo-data, as in NuFlavorRatiosSteriles (tools/coverage.py: flavor_draws, segment_draws): the composition from
     IceCube's released likelihood moved to f_t; each spectrum segment around C_t with the chi^2's asymmetric errors.
  3. Each pseudo-experiment profiled at the same couplings, from theta_t.
  4. Window check: where a pseudo-experiment's lowest chi^2 (the smallest coupling aside) sits on the window's lowest
     or highest coupling, the next grid coupling beyond it is fitted too, until it does not (or the grid ends).
  5. Reading: each pseudo-experiment's column as a one-column plane summary (chi2_sm = its chi^2 at the smallest
     coupling, as tools/export_limits.py sets it), read by analysis.exclusion.field, the limits' own reading (per-mass
     Delta chi^2 >= 3.84, PCHIP in log10 g, specks out, upper limits only). g_t is covered when the field is not
     positive there. Coverage = the fraction covered, with its binomial error.

Usage (on Tycho, NULRI_FLUX_CACHE the job's own folder; the configuration with the 62-mass scan, e.g.
analysis/configs/campaign_ml62/now_icecube/le_ltau_mev_fx__b8.json):
  python tools/coverage_limits.py CONFIG --mass M_eV --true-log10g LG --toys 100 --workers W --out OUT.json
  python tools/coverage_limits.py --reread OUT.json      (step 5 again, with this checkout's analysis.exclusion)
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "tools")]

from analysis import config as acfg            # noqa: E402
from analysis import exclusion as ex           # noqa: E402
from analysis import surfaces                  # noqa: E402
from coverage import flavor_draws, segment_draws  # noqa: E402

WINDOW = (-1.0, 0.4)        # decades around the true coupling (his choice)
MAX_ROUNDS = 8
_W = {}


# ----------------------------------------------------------------------------------------------------------------
# Pure parts (tests/test_coverage_limits.py)
# ----------------------------------------------------------------------------------------------------------------
def window(y_grid, lg_true, lo=WINDOW[0], hi=WINDOW[1], lowest=None):
    """The plane's couplings from lg_true + lo to lg_true + hi (decades), the smallest coupling apart; with `lowest`
    (log10 g), from there if it lies lower (2026-11-26: the columns are read from their minimum, which lay near
    10^-4.3 in the first pseudo-experiments, 2.5 decades below the refit's limits)."""
    lg = np.log10(np.asarray(y_grid, float))
    low = lg_true + lo if lowest is None else min(lg_true + lo, float(lowest))
    return [float(g) for g, l in zip(y_grid[1:], lg[1:]) if low - 1e-9 <= l <= lg_true + hi + 1e-9]


def widen(y_grid, chi):
    """The couplings to add to one pseudo-experiment's column {g: chi2}: the grid's next coupling beyond the window's
    lowest or highest one where the column's lowest chi^2 (the grid's smallest coupling aside) sits there."""
    y = [float(g) for g in y_grid]
    inner = sorted(g for g in chi if g != y[0])
    if not inner:
        return []
    g_min = min(inner, key=lambda g: chi[g])
    add = []
    if g_min == inner[0]:
        i = y.index(inner[0])
        if i - 1 >= 1:
            add.append(y[i - 1])
    if g_min == inner[-1]:
        i = y.index(inner[-1])
        if i + 1 < len(y):
            add.append(y[i + 1])
    return add


def read(names, M, chi, g_true):
    """The limits' reading of one column {g: chi2} (its smallest coupling the no-effect reference): (g_true covered,
    the lowest excluded log10 g or None), through analysis.exclusion.field on a one-column plane summary."""
    gs = sorted(chi)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, f"plane_{names[0]}_{names[1]}.json")
        json.dump(dict(kind="plane", names=list(names), grids=[[float(M)], gs], log=[True, True],
                       chi2=[[float(chi[g]) for g in gs]], chi2_sm=float(chi[gs[0]])), open(path, "w"))
        _, G, F, _, _ = ex.field(path)
    f = np.asarray(F[:, 0].filled(-1.0), float)
    excluded = f > 0
    covered = not bool(np.interp(np.log10(g_true), G, f) > 0)
    return covered, (float(G[int(np.argmax(excluded))]) if excluded.any() else None)


# ----------------------------------------------------------------------------------------------------------------
# Workers (spawned: the solver is not fork-safe)
# ----------------------------------------------------------------------------------------------------------------
def _init(path, n_random=0):
    import dataclasses
    from analysis.exact_profile import ExactProfiler, ExactSettings
    cfg = acfg.load(path)
    obj = cfg.objective()
    exact = ExactProfiler(cfg, objective=obj, settings=ExactSettings.from_dict(cfg.exact))
    if n_random:
        # 2026-11-26: every fit of the cheap parameters also from n_random random starts (fitting.FitSettings). The
        # single-start fit sticks in local minima on pseudo-data (chi^2 7 and 12 too high at two points refitted); on
        # the real data the two agree to 0.2 at every coupling of both columns (Tycho jobs 867700, 868318, 868320)
        exact.fit_settings = dataclasses.replace(exact.fit_settings, n_random=int(n_random))
    _W.update(cfg=cfg, obj=obj, exact=exact, real=(obj.data, obj.data_centre, dict(obj.spectrum)), toy=None)


def _set_data(k, f_obs, seg_values):
    """Pseudo-experiment k's data in this worker's objective (k None: the real data)."""
    if _W["toy"] == k:
        return
    obj = _W["obj"]
    data, centre, spec0 = _W["real"]
    if k is None:
        obj.data, obj.data_centre, obj.spectrum = data, centre, dict(spec0)
    else:
        obj.data = surfaces.recentred(obj.surface, *f_obs)
        obj.data_centre = tuple(f_obs)
        D0 = spec0["value"]
        s_dn, s_up = D0 - spec0["lo"], spec0["hi"] - D0
        obj.spectrum = dict(spec0, value=seg_values, lo=np.maximum(seg_values - s_dn, 0.0), hi=seg_values + s_up)
    _W["toy"] = k


def _task(task):
    """One exact profile: (toy k or None, f_obs, segments, held, start) -> record."""
    k, f_obs, seg, held, start = task
    _set_data(k, None if f_obs is None else np.asarray(f_obs), None if seg is None else np.asarray(seg))
    t0 = time.perf_counter()
    r = _profile(held, start)
    if r is None:
        return dict(toy=k, held=held, chi2=float("nan"), values={}, message="not allowed", seconds=0.0)
    return dict(toy=k, held=held, chi2=float(r.chi2), values=dict(r.values), message=r.message,
                seconds=time.perf_counter() - t0)


def _profile(held, start):
    """The exact profile at one point as the plane runs make it (analysis.scans._run_chain): the lightest mass capped
    where the model stops being perturbative (ExactProfiler.lightest_cap; None where every mass is allowed), then
    profile_global."""
    ex = _W["exact"]
    cap = ex.lightest_cap(held)
    if cap is not None and cap <= 0.0:
        return None                      # not perturbative at any lightest mass: no fit, as the plane runs
    ex.set_lightest_cap(cap)
    return ex.profile_global(held, {n: v for n, v in start.items() if n not in held})


def _truth(task):
    """The truth at (M, g_t) on the real data: its profile, f_t and C_t."""
    held, start = task
    _set_data(None, None, None)
    r = _profile(held, start)
    obj = _W["obj"]
    theta = {**dict(r.values), **held}
    return dict(chi2=float(r.chi2), values=theta, f_true=np.asarray(obj.fractions(theta), float)[:2].tolist(),
                C_true=np.asarray(obj.spectrum_segments(theta), float).tolist())


# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    if argv is None and len(sys.argv) == 3 and sys.argv[1] == "--reread":
        o = reread(sys.argv[2])
        print(f"coverage {o['coverage']:.3f} +- {o['coverage_err']:.3f} ({len(o['results'])} pseudo-experiments)")
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--mass", type=float, required=True, help="M (eV), snapped to the plane's grid")
    ap.add_argument("--true-log10g", type=float, required=True, help="the true coupling: the limit at M")
    ap.add_argument("--toys", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--lowest-log10g", type=float, default=None,
                    help="the window's lowest coupling where it lies below 1 decade under the true one")
    ap.add_argument("--window-hi", type=float, default=WINDOW[1], help="decades above the true coupling")
    ap.add_argument("--n-random", type=int, default=0,
                    help="random starts added to every fit of the cheap parameters (0: the plane runs' single start)")
    ap.add_argument("--only", default=None, metavar="LO:HI",
                    help="fit only the pseudo-experiments LO..HI-1 of the --toys drawn (the draws stay those of the seed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lo_k, hi_k = (0, a.toys) if a.only is None else (int(v) for v in a.only.split(":"))
    if a.toys > 0 and not 0 <= lo_k < hi_k <= a.toys:      # --toys 0: the real column and the truth only
        raise SystemExit("--only LO:HI needs 0 <= LO < HI <= --toys")
    ks = list(range(lo_k, hi_k))
    if not os.environ.get("NULRI_FLUX_CACHE"):
        raise SystemExit("NULRI_FLUX_CACHE must name the job's flux-cache folder: the workers share the solver runs")
    _init(a.config, a.n_random)           # here first: a failing configuration fails at once
    cfg = _W["cfg"]
    pl = cfg.planes[0]
    names = (pl.x, pl.y)
    y = [float(g) for g in pl.y_grid]
    M = float(pl.x_grid[int(np.argmin(np.abs(np.log10(pl.x_grid) - np.log10(a.mass))))])
    g_t = 10.0 ** a.true_log10g
    start0 = {p.name: p.start for p in cfg.parameters}
    gs = [y[0]] + window(y, a.true_log10g, hi=a.window_hi, lowest=a.lowest_log10g)
    on_grid = [g for g in gs if abs(np.log10(g) - a.true_log10g) < 1e-9]
    out = dict(config=a.config, mass_eV=M, log10g_true=a.true_log10g, window=[WINDOW[0], a.window_hi],
               lowest_log10g=a.lowest_log10g, couplings=gs,
               toys=a.toys, only=[lo_k, hi_k], n_random=a.n_random, seed=a.seed, workers=a.workers, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.config, a.n_random)) as pool:
        # 1. the real data's column and the truth (fills the flux cache)
        t0 = time.perf_counter()
        truth = pool.apply_async(_truth, (({pl.x: M, pl.y: g_t}, start0),))
        real = pool.map(_task, [(None, None, None, {pl.x: M, pl.y: g}, start0) for g in gs], chunksize=1)
        truth = truth.get()
        out.update(names=list(names), real={repr(r["held"][pl.y]): r["chi2"] for r in real}, truth=truth,
                   real_seconds=time.perf_counter() - t0)
        _write(a.out, out)
        if a.toys <= 0:
            return
        # 2. pseudo-data
        rng = np.random.default_rng([a.seed, int(round(1000 * (a.true_log10g + 40.0)))])
        f_t, C_t = np.asarray(truth["f_true"]), np.asarray(truth["C_true"])
        f_obs = f_t[None, :] + flavor_draws(_W["obj"].surface, a.toys, rng)
        segs = segment_draws(C_t, _W["real"][2], a.toys, rng)
        th = {n: v for n, v in truth["values"].items() if n not in names}
        # 3-4. the fits, widened where needed
        # the true coupling is fitted too (2026-11-26): its chi^2 kept apart (chi2_true: the test statistic at the
        # truth, for a corrected threshold), never in the column, which is read on the plane's grid as the limits are
        chi = {k: {} for k in ks}
        chi_true, vals = {}, {k: {} for k in ks}

        def partial():
            return dict(chi2={str(k): {repr(g): c for g, c in sorted(chi[k].items())} for k in chi},
                        chi2_true={str(k): c for k, c in chi_true.items()},
                        f_obs=f_obs.tolist(), segments=segs.tolist())
        todo = {k: list(gs) + ([] if on_grid else [g_t]) for k in ks}
        t1 = time.perf_counter()
        for rnd in range(MAX_ROUNDS):
            tasks = [(k, f_obs[k].tolist(), segs[k].tolist(), {pl.x: M, pl.y: g}, th) for k in sorted(todo)
                     for g in todo[k]]
            if not tasks:
                break
            for n, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=1), 1):
                if r["held"][pl.y] == g_t and not on_grid:
                    chi_true[r["toy"]] = r["chi2"]
                else:
                    chi[r["toy"]][r["held"][pl.y]] = r["chi2"]
                # each fit's solution (the parameters that are not held), for later checks of the fits
                vals[r["toy"]][repr(r["held"][pl.y])] = {n: v for n, v in r["values"].items() if n not in names}
                if n % 50 == 0:               # kept as the round goes: a task stopped inside it loses at most 50 fits
                    out["partial"] = partial()
                    _write(a.out, out)
                if n % 10 == 0 or n == len(tasks):
                    print(f"round {rnd + 1}: {n} of {len(tasks)} profiles, {time.perf_counter() - t1:.0f} s since the "
                          f"pseudo-experiments began; the last took {r['seconds']:.0f} s", flush=True)
            todo = {k: [g for g in widen(y, chi[k]) if g not in chi[k]] for k in chi}
            todo = {k: v for k, v in todo.items() if v}
            out["rounds"] = rnd + 1
            # kept after every round: a task stopped later loses no finished round
            out["partial"] = partial()
            _write(a.out, out)
        out["toys_seconds"] = time.perf_counter() - t1
        out["window_unresolved"] = sorted(todo)            # still on an edge after MAX_ROUNDS (or none)
        # 5. the reading (redone on the laptop with the repository's analysis.exclusion: --reread)
        out.pop("partial", None)
        out["results"] = [dict(toy=k, f_obs=f_obs[k].tolist(), segments=segs[k].tolist(),
                               chi2={repr(g): c for g, c in sorted(chi[k].items())},
                               chi2_true=chi[k][on_grid[0]] if on_grid else chi_true[k], values=vals[k]) for k in ks]
        out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write(a.out, out)
        try:
            reread(a.out)
        except Exception as err:                         # an older analysis.exclusion on Tycho: read it here later
            print("reading left for --reread:", repr(err))


def reread(path):
    """The reading (step 5) of a finished run's columns, with this checkout's analysis.exclusion; writes coverage,
    its binomial error and each pseudo-experiment's result into the file."""
    out = json.load(open(path))
    names = tuple(out.get("names") or _names(out["config"]))
    M, g_t = out["mass_eV"], 10.0 ** out["log10g_true"]
    for r in out["results"]:
        r["covered"], r["lowest_excluded_log10g"] = read(names, M, {float(g): c for g, c in r["chi2"].items()}, g_t)
        if r.get("chi2_true") is not None:     # the test statistic at the truth: from the column's lowest chi^2
            r["dchi2_true"] = float(r["chi2_true"] - min(min(r["chi2"].values()), r["chi2_true"]))
    c = np.array([r["covered"] for r in out["results"]], float)
    out.update(coverage=float(c.mean()), coverage_err=float(np.sqrt(c.mean() * (1 - c.mean()) / len(c))),
               real_read=dict(zip(("covered", "lowest_excluded_log10g"),
                                  read(names, M, {float(g): v for g, v in out["real"].items()}, g_t))))
    _write(path, out)
    return out


def _names(config):
    pl = json.load(open(config))["planes"][0]
    return pl["x"], pl["y"]


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
