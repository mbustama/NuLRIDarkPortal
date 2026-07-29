"""
convergence_study.py

Finds the coarsest safe values of the solver's resolution parameters:

    n_z_bins        redshift steps of the cosmological integration
    n_E_bins        energy bins (also the cascading integral's resolution)
    N_steps_MW      path-length steps per Milky Way trajectory
    N_steps_earth   path-length steps per Earth chord (NOTE: under the default
                    earth_step_scheme='avg' this axis is certified step-count-
                    independent -- N = 8..1000 agree to <= 5e-9 at the standard
                    validation points -- so ladders on it mainly diagnose the
                    legacy 'left' scheme; the chord-count axis still matters)
    n_mw_paths      Milky Way trajectories in the solid-angle average
    n_earth_chords  Earth chords in the zenith average

For each axis it runs the FULL production pipeline (run_solver: real Milky Way state,
both neutrinos and antineutrinos, all config plumbing) over a ladder of values, holding
every other parameter at its production value, and compares each rung against the finest
rung of the ladder (the reference). The comparison metric is the MAXIMUM relative
difference across the full output spectrum (every energy bin x every flavor), plus the
energy-averaged flavor ratios as a secondary integral metric. This follows the
validation discipline established by the N_steps_earth near-miss documented in
dark_portal_solver.py: convergence verdicts from synthetic states, single energy bins,
or median errors are not trustworthy -- an energy-localized resonance error hides from
all three.

IMPORTANT: the safe values DEPEND ON THE DARK-PORTAL PARAMETERS. The cosmological
absorption resonance sits at E ~ m_Zd^2 / (2 m_nu) and moves through the energy band as
m_Zd changes; the Earth's MSW-like level crossing depends on the potential magnitude
g_D * eps' / m_Zd^2. A step count that converges beautifully at one parameter point can
be badly under-resolved at another where a resonance lands in-band. This tool therefore
evaluates every axis at several parameter points (defaults chosen to place the
resonance below, inside, and above the energy band) and recommends the WORST CASE
across them. Re-run it whenever the parameter region a fit will explore changes.

Usage examples:

    # Quick preview (coarse ladders, reduced baseline -- for smoke-testing the tool):
    python3 convergence_study.py --profile quick

    # Production study of the two cosmological-grid axes:
    python3 convergence_study.py --axes n_z_bins n_E_bins --n_cores 8

    # Everything, at production baselines (slow -- hours; run it overnight):
    python3 convergence_study.py --n_cores 8

    # Custom parameter points (JSON list of dark_portal override dicts):
    python3 convergence_study.py --param_points my_points.json

Results are written to --output_dir (default ../output/convergence/) as a JSON record
and a human-readable text report, with a per-axis recommendation: the smallest tested
value whose max relative difference against the reference stays below --tolerance.
"""

import os
import sys
import json
import argparse
import time

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_TOOL_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import copy

import numpy as np
import dark_portal_solver as dps
from flux_postprocessing import compute_energy_averaged_ratios

# This tool runs the solver dozens of times back to back; per-run progress bars would
# drown the study's own log (and tqdm writes to stderr, escaping stdout suppression).
import dp_core
dp_core.HAS_TQDM = False  # flag lives in dp_core since the module split

_CONFIG_TEMPLATE = None

# Ladder of tested values per axis, and the finer reference each is compared against.
# The reference deliberately sits well above the largest tested value so that "the
# finest rung agrees with the reference" is itself evidence of convergence rather than
# an assumption. All overridable via --profile quick or a custom config.
DEFAULT_LADDERS = {
    "n_z_bins":       {"values": [100, 200, 400, 700, 1000], "reference": 1500},
    "n_E_bins":       {"values": [100, 200, 400, 700, 1000], "reference": 1500},
    "N_steps_MW":     {"values": [25, 50, 100, 200],          "reference": 400},
    "N_steps_earth":  {"values": [125, 250, 500, 1000],       "reference": 2000},
    "n_mw_paths":     {"values": [6, 12, 25, 50],             "reference": 100},
    "n_earth_chords": {"values": [3, 6, 12, 25, 50],          "reference": 100},
}

QUICK_LADDERS = {
    "n_z_bins":       {"values": [50, 100, 200], "reference": 400},
    "n_E_bins":       {"values": [50, 100, 200], "reference": 400},
    "N_steps_MW":     {"values": [10, 25, 50],   "reference": 100},
    "N_steps_earth":  {"values": [50, 125, 250], "reference": 500},
    "n_mw_paths":     {"values": [2, 6, 12],     "reference": 25},
    "n_earth_chords": {"values": [1, 3, 6],      "reference": 12},
}

# Baseline (held-fixed) resolution while one axis is scanned. The production baseline is
# the solver's own default grid; the quick baseline is only for smoke-testing the tool.
# (n_earth_chords=12 / N_steps_MW=25 track the solver defaults adopted from the
# 2026-07-28 production study in output/convergence/; N_steps_earth=25 tracks the
# 2026-07-29 default under the certified step-count-independent 'avg' scheme --
# holding it higher only slows every rung of the other axes with no accuracy gain.)
PRODUCTION_BASELINE = {"n_E_bins": 1000, "n_z_bins": 1000, "n_mw_paths": 6,
                       "n_earth_chords": 12, "N_steps_MW": 25, "N_steps_earth": 25}
QUICK_BASELINE = {"n_E_bins": 100, "n_z_bins": 100, "n_mw_paths": 6,
                  "n_earth_chords": 3, "N_steps_MW": 25, "N_steps_earth": 25}

# Default dark-portal parameter points: the cosmological absorption resonance sits at
# E_res ~ m_Zd^2 / (2 m_nu). With m_nu ~ 0.01-0.05 eV and the default energy band
# 1e13-3e16 eV, these three m_Zd choices place E_res below the band, squarely inside it,
# and near its upper edge; the last point also strengthens the potential (larger g_D)
# to stress the Earth-crossing MSW-like resonance.
DEFAULT_PARAM_POINTS = [
    {"label": "resonance_below_band", "m_Zd_eV": 3.0e5, "g_D": 1e-4, "epsilon_prime": 0.1},
    {"label": "resonance_in_band",    "m_Zd_eV": 2.0e6, "g_D": 1e-4, "epsilon_prime": 0.1},
    {"label": "resonance_high_strong_coupling", "m_Zd_eV": 2.0e7, "g_D": 1e-2, "epsilon_prime": 0.1},
]


def make_config(baseline, param_point, n_cores, scratch_dir):
    """Builds a full solver config at the given baseline resolutions and dark-portal point."""
    global _CONFIG_TEMPLATE
    if _CONFIG_TEMPLATE is None:
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _CONFIG_TEMPLATE = dps.create_default_config(
                os.path.join(scratch_dir, "convergence_base_config.json"))
        finally:
            sys.stdout = old_stdout
    cfg = copy.deepcopy(_CONFIG_TEMPLATE)
    for k, v in baseline.items():
        cfg["grid"][k] = int(v)
    for k, v in param_point.items():
        if k == "label":
            continue
        cfg["dark_portal"][k] = v
    cfg["execution"]["n_cores"] = int(n_cores)
    cfg["execution"]["verbose"] = 0
    return cfg


def run_once(cfg):
    """Runs the full pipeline silently and returns (E_bins, Flux, ratios)."""
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        E, F = dps.run_solver(cfg)
    finally:
        sys.stdout = old_stdout
    ratios = compute_energy_averaged_ratios(E, np.ascontiguousarray(F))
    return E, F, ratios


def compare_same_grid(F_test, F_ref):
    """Max relative difference across the full spectrum (identical energy grids)."""
    return float(np.max(np.abs(F_test - F_ref) / np.maximum(np.abs(F_ref), 1e-300)))


def compare_across_grids(E_test, F_test, E_ref, F_ref):
    """Max relative difference when the energy grids differ (the n_E_bins axis):
    log-log-interpolates the reference spectrum onto the test grid. The result folds in
    interpolation error on top of genuine resolution error, so it is reported alongside
    the grid-independent flavor-ratio metric rather than trusted alone."""
    rel = 0.0
    logE_ref = np.log(E_ref)
    for a in range(3):
        ref_interp = np.exp(np.interp(np.log(E_test), logE_ref,
                                      np.log(np.maximum(F_ref[:, a], 1e-300))))
        rel = max(rel, float(np.max(np.abs(F_test[:, a] - ref_interp) /
                                    np.maximum(np.abs(ref_interp), 1e-300))))
    return rel


def study_axis(axis, ladder, baseline, param_point, n_cores, scratch_dir, log):
    """Runs the ladder + reference for one axis at one parameter point."""
    results = {"axis": axis, "param_point": param_point, "reference": ladder["reference"],
               "rungs": []}

    cfg_ref = make_config(baseline, param_point, n_cores, scratch_dir)
    cfg_ref["grid"][axis] = int(ladder["reference"])
    t0 = time.perf_counter()
    E_ref, F_ref, ratios_ref = run_once(cfg_ref)
    log(f"    reference {axis}={ladder['reference']}: {time.perf_counter()-t0:.1f}s")

    for val in ladder["values"]:
        cfg = make_config(baseline, param_point, n_cores, scratch_dir)
        cfg["grid"][axis] = int(val)
        t0 = time.perf_counter()
        E, F, ratios = run_once(cfg)
        dt = time.perf_counter() - t0

        if axis == "n_E_bins":
            max_rel_flux = compare_across_grids(E, F, E_ref, F_ref)
        else:
            max_rel_flux = compare_same_grid(F, F_ref)
        max_abs_ratio = float(np.max(np.abs(ratios - ratios_ref)))

        results["rungs"].append({"value": int(val), "max_rel_flux": max_rel_flux,
                                 "max_abs_ratio_diff": max_abs_ratio, "runtime_s": dt})
        log(f"    {axis}={val}: max_rel_flux={max_rel_flux:.3e}, "
            f"max_abs_ratio_diff={max_abs_ratio:.3e} ({dt:.1f}s)")
    return results


def _rung_passes(rung, criterion, tolerance, ratio_tolerance):
    """Whether one ladder rung meets the active convergence criterion.

    criterion:
      "ratio" (default) -- energy-averaged flavor-ratio agreement only. This is the
          observable the likelihood actually consumes, and the one that CAN converge:
          the pointwise-flux metric is structurally unmeetable in the deep absorption
          dip, where exponential attenuation amplifies any resolved-rate difference
          into an O(1) relative difference of a minuscule flux.
      "flux"  -- pointwise max relative flux only (diagnostic; expect strict verdicts).
      "both"  -- the original strict AND of the two gates.
    """
    flux_ok = rung["max_rel_flux"] < tolerance
    ratio_ok = rung["max_abs_ratio_diff"] < ratio_tolerance
    if criterion == "ratio":
        return ratio_ok
    if criterion == "flux":
        return flux_ok
    return flux_ok and ratio_ok


def recommend(axis_results_across_points, tolerance, ratio_tolerance, criterion="ratio"):
    """Smallest tested value from which EVERY tested value upward also satisfies the
    criterion at every parameter point.

    The stronger "upward-closed" condition matters because several axes converge
    non-monotonically (fringe aliasing makes intermediate resolutions test worse than
    coarser ones -- e.g. the production study's n_z_bins ratio errors go 9.1e-3 at 100,
    1.1e-2 at 200, 1.5e-2 at 400, then back under 1e-2 only at 1000). Under a naive
    "smallest passing value" rule, 100 would be recommended even though refining it can
    make the answer WORSE; a value is only trustworthy if refinement never degrades it
    within the tested ladder."""
    by_value = {}
    for res in axis_results_across_points:
        for rung in res["rungs"]:
            v = rung["value"]
            by_value.setdefault(v, []).append(rung)
    values = sorted(by_value)
    passes = {v: all(_rung_passes(r, criterion, tolerance, ratio_tolerance)
                     for r in by_value[v]) for v in values}
    recommended = None
    for v in reversed(values):
        if passes[v]:
            recommended = v
        else:
            break
    return recommended


def reanalyze(saved_json_path, tolerance, ratio_tolerance, criterion, log):
    """Recomputes per-axis recommendations from a previously saved study JSON under new
    tolerances / criterion -- no solver runs. A full production study takes the better
    part of an hour; the per-rung metrics it records are criterion-agnostic, so
    re-deriving verdicts is free."""
    with open(saved_json_path) as f:
        saved = json.load(f)
    log(f"Re-analysis of {saved_json_path}")
    log(f"Criterion: {criterion} (flux tol {tolerance:g}, ratio tol {ratio_tolerance:g})")
    log(f"Original study profile: {saved.get('profile')}, baseline {saved.get('baseline')}")
    for axis, entry in saved.get("axes", {}).items():
        rec = recommend(entry["results"], tolerance, ratio_tolerance, criterion)
        worst = {}
        for res in entry["results"]:
            for rung in res["rungs"]:
                v = rung["value"]
                w = worst.setdefault(v, {"max_rel_flux": 0.0, "max_abs_ratio_diff": 0.0})
                w["max_rel_flux"] = max(w["max_rel_flux"], rung["max_rel_flux"])
                w["max_abs_ratio_diff"] = max(w["max_abs_ratio_diff"], rung["max_abs_ratio_diff"])
        log(f"\n  Axis {axis} (worst case across parameter points):")
        for v in sorted(worst):
            log(f"    {axis}={v}: max_rel_flux={worst[v]['max_rel_flux']:.3e}, "
                f"max_abs_ratio_diff={worst[v]['max_abs_ratio_diff']:.3e}")
        if rec is not None:
            log(f"    -> RECOMMENDED {axis} >= {rec} under criterion '{criterion}'")
        else:
            log(f"    -> no tested value of {axis} met criterion '{criterion}'; "
                f"use the study reference or extend the ladder")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axes", nargs="*", default=list(DEFAULT_LADDERS.keys()),
                        help="Which resolution axes to study (default: all six).")
    parser.add_argument("--profile", choices=["production", "quick"], default="production",
                        help="'quick' uses coarse ladders and a reduced baseline -- for "
                             "smoke-testing the tool only, NOT for real recommendations.")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Max relative flux difference accepted vs the reference (default 1e-3).")
    parser.add_argument("--ratio_tolerance", type=float, default=1e-2,
                        help="Max absolute flavor-ratio difference accepted (default 1e-2 -- "
                             "percent-level flavor ratios; tighten to 1e-3 for sub-percent work).")
    parser.add_argument("--criterion", choices=["ratio", "flux", "both"], default="ratio",
                        help="Which metric gates the recommendation (default 'ratio': the "
                             "energy-averaged flavor ratios the likelihood consumes -- the "
                             "pointwise-flux gate is structurally unmeetable in the deep "
                             "absorption dip; 'both' restores the original strict AND).")
    parser.add_argument("--reanalyze", type=str, default=None, metavar="SAVED_JSON",
                        help="Recompute recommendations from a previously saved study JSON "
                             "under the given tolerances/criterion, without running the solver.")
    parser.add_argument("--param_points", type=str, default=None,
                        help="JSON file with a list of dark_portal override dicts; defaults "
                             "to three points placing the resonance below/inside/above band.")
    parser.add_argument("--n_cores", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--output_dir", type=str, default=os.path.join(_TOOL_DIR, "..", "output", "convergence"))
    args = parser.parse_args()

    if args.reanalyze:
        lines = []
        def relog(msg):
            print(msg, flush=True)
            lines.append(msg)
        reanalyze(args.reanalyze, args.tolerance, args.ratio_tolerance, args.criterion, relog)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, f"reanalysis_{args.criterion}_{stamp}.txt")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nSaved: {out}")
        return

    ladders = QUICK_LADDERS if args.profile == "quick" else DEFAULT_LADDERS
    baseline = QUICK_BASELINE if args.profile == "quick" else PRODUCTION_BASELINE

    if args.param_points:
        with open(args.param_points) as f:
            param_points = json.load(f)
    else:
        param_points = DEFAULT_PARAM_POINTS

    os.makedirs(args.output_dir, exist_ok=True)
    scratch_dir = args.output_dir

    lines = []
    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"Convergence study: profile={args.profile}, axes={args.axes}, "
        f"tolerance={args.tolerance:g} (flux) / {args.ratio_tolerance:g} (ratios), "
        f"n_cores={args.n_cores}")
    log(f"Baseline resolutions while scanning each axis: {baseline}")
    log(f"Parameter points: {[p.get('label', str(p)) for p in param_points]}")

    all_results = {"profile": args.profile, "baseline": baseline,
                   "tolerance": args.tolerance, "ratio_tolerance": args.ratio_tolerance,
                   "criterion": args.criterion,
                   "param_points": param_points, "axes": {}}

    for axis in args.axes:
        if axis not in ladders:
            log(f"[skip] unknown axis: {axis}")
            continue
        log(f"\n=== Axis: {axis} (reference {ladders[axis]['reference']}) ===")
        per_point = []
        for point in param_points:
            log(f"  Parameter point: {point.get('label', point)}")
            per_point.append(study_axis(axis, ladders[axis], baseline, point,
                                        args.n_cores, scratch_dir, log))
        rec = recommend(per_point, args.tolerance, args.ratio_tolerance, args.criterion)
        all_results["axes"][axis] = {"results": per_point, "recommended": rec}
        if rec is not None:
            log(f"  -> RECOMMENDED {axis} >= {rec} (worst case across parameter points, "
                f"criterion '{args.criterion}': flux tol {args.tolerance:g} / "
                f"ratio tol {args.ratio_tolerance:g})")
        else:
            log(f"  -> NO tested value of {axis} met criterion '{args.criterion}' at every "
                f"parameter point; use the reference value {ladders[axis]['reference']} or extend the ladder.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"convergence_{args.profile}_{stamp}.json")
    txt_path = os.path.join(args.output_dir, f"convergence_{args.profile}_{stamp}.txt")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nSaved: {json_path}\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
