"""
run_limits.py

Computes profiled exclusion limits on the dark-portal couplings as functions of the
mediator mass: for each of the planes

    (m_Zd_eV, u_e), (m_Zd_eV, u_mu), (m_Zd_eV, u_tau),
    (m_Zd_eV, g_D), (m_Zd_eV, epsilon_prime)

it evaluates the profile likelihood on a (log, log) grid -- at every grid point the pair
is fixed and ALL other floating parameters are minimized over with Minuit -- and then
extracts, per m_Zd column, the coupling values where the profiled delta-chi2 crosses the
configured confidence thresholds (default: 90% and 95% CL for one parameter of
interest, delta-chi2 = 2.71 / 3.84). The result is the familiar exclusion curve
"coupling vs mediator mass, everything else profiled".

Reuses the existing frequentist machinery end to end: GenericLikelihoodWrapper for the
objective, a global simplex+migrad fit for the reference minimum (which, running in the
main process, automatically uses the multi-core spatial handover), scan_pair_parallel
for the distributed grid, and repair_2d_scan_outliers for stuck-minimizer cleanup.

Pairs whose members are locked in the base config's fixed_params (e.g. u_tau,
epsilon_prime in the shipped configs) are automatically unlocked for their own plane --
each plane gets its own likelihood wrapper built from a pair-specific copy of the
config.

Usage:
    python3 -m src.run_limits --config_file ../config/pion_2050_case_5_fixed_nuisance.json

Optional config block (all keys have defaults):

    "limits_settings": {
        "pairs": [["m_Zd_eV", "u_e"], ["m_Zd_eV", "u_mu"], ["m_Zd_eV", "u_tau"],
                  ["m_Zd_eV", "g_D"], ["m_Zd_eV", "epsilon_prime"]],
        "n_steps": 20,
        "x_range": [1e3, 1e9],
        "y_ranges": {"u_e": [1e-3, 1.0]},
        "thresholds": {"90": 2.71, "95": 3.84},
        "use_checkpointing": true
    }

Outputs (under <output_dir>/limits/):
    limits_2d_<X>_vs_m_Zd_eV.json   full grid + delta-chi2 surface + limit curves
    limits_<X>_vs_m_Zd_eV.txt       per-column limit table (one row per m_Zd)
    limits_summary.pdf              log-log overview plot of every computed plane
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import copy
import argparse
import multiprocessing
from multiprocessing import Pool, cpu_count

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from iminuit import Minuit

from src.core.config import load_config
from src.core.likelihood import GenericLikelihoodWrapper
from src.engines.optimizers import estimate_param_names, estimate_start_values
from src.engines.scanner import scan_pair_parallel
from src.topology.repair import repair_2d_scan_outliers

DEFAULT_PAIRS = [["m_Zd_eV", "u_e"], ["m_Zd_eV", "u_mu"], ["m_Zd_eV", "u_tau"],
                 ["m_Zd_eV", "g_D"], ["m_Zd_eV", "epsilon_prime"]]
DEFAULT_THRESHOLDS = {"90": 2.71, "95": 3.84}
DEFAULT_X_RANGE = [1e3, 1e9]


def build_pair_config(base_config, pair):
    """Returns (pair_config, param_names) where both members of the pair float even if
    the base config pins them in fixed_params, and every other floating parameter of the
    base config keeps floating (those are the ones profiled over)."""
    cfg = copy.deepcopy(base_config)
    for p in pair:
        cfg.get("fixed_params", {}).pop(p, None)

    param_names = estimate_param_names(cfg)
    for p in pair:
        if p not in param_names:
            param_names.append(p)
    return cfg, param_names


def safe_start_values(param_names, cfg):
    """Start values clamped inside the configured parameter limits (some shipped start
    values, e.g. u_tau = 0.0, sit below their own lower limit)."""
    starts = estimate_start_values(param_names, cfg)
    limits_cfg = cfg.get("parameter_limits", {})
    for p, v in starts.items():
        lim = limits_cfg.get(p)
        if lim and len(lim) == 2:
            low, high = lim
            if low is not None and v < low:
                starts[p] = low * 1.001 if low > 0 else low + 1e-9
            if high is not None and v > high:
                starts[p] = high * 0.999
    return starts


def extract_column_limits(y_grid, dchi2_column, threshold):
    """All y values where the profiled delta-chi2 crosses `threshold` within one m_Zd
    column, found by linear interpolation in log10(y). Returns a (possibly empty) list;
    two entries bracket an excluded (or allowed) band, one entry is a one-sided limit."""
    crossings = []
    logy = np.log10(y_grid)
    d = dchi2_column - threshold
    for i in range(len(y_grid) - 1):
        if not (np.isfinite(d[i]) and np.isfinite(d[i + 1])):
            continue
        # Failed grid points carry a 1e9 sentinel; a crossing against a sentinel is an
        # artifact of the failure, not a limit. Only bracket between evaluated points.
        if dchi2_column[i] > 1e5 or dchi2_column[i + 1] > 1e5:
            continue
        if d[i] * d[i + 1] < 0:
            frac = d[i] / (d[i] - d[i + 1])
            crossings.append(float(10 ** (logy[i] + frac * (logy[i + 1] - logy[i]))))
        elif d[i] == 0.0:
            crossings.append(float(y_grid[i]))
    return crossings


def run_pair(base_config, pair, ls, output_dir, pool, num_cores, repair_cfg):
    """Runs one (m_Zd, X) plane end to end; returns the payload dict that is also
    written to disk."""
    param_x, param_y = pair          # x = m_Zd_eV, y = the coupling
    cfg, param_names = build_pair_config(base_config, pair)
    start_vals = safe_start_values(param_names, cfg)
    limits = {p: tuple(cfg.get("parameter_limits", {}).get(p, (None, None))) for p in param_names}

    print(f"\n=== Plane: {param_y} vs {param_x} ===")
    print(f"  Floating (profiled) parameters: {[p for p in param_names if p not in pair]}")

    func = GenericLikelihoodWrapper(cfg, param_names)

    m = Minuit(func, name=param_names, **start_vals)
    m.errordef = Minuit.LEAST_SQUARES
    for p, lim in limits.items():
        m.limits[p] = lim
    if cfg.get("scan_settings", {}).get("use_explicit_step_sizes", False):
        for p in param_names:
            if p in cfg.get("step_sizes", {}):
                m.errors[p] = cfg["step_sizes"][p]

    try:
        m.simplex()
        m.migrad()
        global_min_chi2 = m.fval
        print(f"  Global best fit chi2 = {global_min_chi2:.4f}")
    except Exception as e:
        print(f"  [WARNING] Global fit failed ({e}); the grid minimum will set the baseline.")
        global_min_chi2 = np.inf

    # Grid windows: x from limits_settings.x_range; y from limits_settings.y_ranges[param_y]
    # falling back to the physical parameter_limits. Only the pair's entries of this dict
    # feed the grid construction -- the profiled parameters keep their physical limits.
    x_range = ls.get("x_range", DEFAULT_X_RANGE)
    y_range = ls.get("y_ranges", {}).get(param_y, cfg.get("parameter_limits", {}).get(param_y, [1e-6, 1.0]))
    scan_limits = dict(limits)
    scan_limits[param_x] = tuple(x_range)
    scan_limits[param_y] = tuple(y_range)

    n_steps = int(ls.get("n_steps", 20))
    _strategy = base_config.get("scan_settings", {}).get("minuit_strategy", 0)
    _rseed = base_config.get("scan_settings", {}).get("retry_seed", -1)
    # Per-segment resume file: a crashed plane previously lost every completed point
    # (the plane-level checkpoint in main() only kicks in once the whole plane is done).
    seg_ckpt_path = os.path.join(output_dir, f"limits_2d_{param_y}_vs_{param_x}.segments.jsonl")
    x_grid, y_grid, Z_chi2, param_grid = scan_pair_parallel(
        m, param_x, param_y, scan_limits, func, param_names,
        n_steps=n_steps, minuit_retries=repair_cfg.get("minuit_retries", 3),
        force_rigid_bounds=True, pool=pool, num_cores=num_cores,
        use_log_x=True, use_log_y=True,
        use_warm_start=base_config.get("scan_settings", {}).get("use_warm_start", True),
        minuit_strategy=_strategy, retry_seed=_rseed,
        checkpoint_path=seg_ckpt_path)

    if not repair_cfg.get("disable_outlier_repair", False):
        x_grid, y_grid, Z_chi2 = repair_2d_scan_outliers(
            x_grid, y_grid, Z_chi2, param_grid, (param_x, param_y), func, scan_limits,
            param_names, dict(zip(m.parameters, m.errors)),
            use_simplex=repair_cfg.get("use_simplex_polish", True),
            threshold=repair_cfg.get("spike_threshold", 0.2),
            max_passes=repair_cfg.get("max_repair_passes", 3),
            minuit_retries=repair_cfg.get("minuit_retries", 3),
            pool=pool, num_cores=num_cores,
            minuit_strategy=_strategy, retry_seed=_rseed)

    baseline = min(global_min_chi2, float(np.nanmin(Z_chi2)))
    dchi2 = np.nan_to_num(Z_chi2 - baseline, nan=1e9)
    dchi2[dchi2 < 0] = 0.0

    thresholds = ls.get("thresholds", DEFAULT_THRESHOLDS)
    limit_curves = {}
    for label, thr in thresholds.items():
        per_column = []
        for j in range(len(x_grid)):
            # Z rows index y, columns index x (scan_pair_parallel task order)
            per_column.append(extract_column_limits(y_grid, dchi2[:, j], float(thr)))
        limit_curves[label] = per_column

    payload = {
        "pair": pair, "x_grid": x_grid.tolist(), "y_grid": y_grid.tolist(),
        "delta_chi2": dchi2.tolist(), "baseline_chi2": baseline,
        "thresholds": thresholds, "limit_curves": limit_curves,
        "profiled_over": [p for p in param_names if p not in pair],
    }

    json_path = os.path.join(output_dir, f"limits_2d_{param_y}_vs_{param_x}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    try:
        os.remove(seg_ckpt_path)
    except OSError:
        pass

    txt_path = os.path.join(output_dir, f"limits_{param_y}_vs_{param_x}.txt")
    with open(txt_path, "w") as f:
        f.write(f"# Profiled limits on {param_y} vs {param_x}"
                f" (profiled over: {', '.join(payload['profiled_over'])})\n")
        f.write(f"# {param_x}    " + "    ".join(f"{param_y}_crossings_{lbl}CL" for lbl in thresholds) + "\n")
        for j, xv in enumerate(x_grid):
            cols = []
            for lbl in thresholds:
                c = limit_curves[lbl][j]
                cols.append(",".join(f"{v:.6e}" for v in c) if c else "none")
            f.write(f"{xv:.6e}    " + "    ".join(cols) + "\n")
    print(f"  Saved: {json_path}\n  Saved: {txt_path}")
    return payload


def plot_summary(payloads, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping the summary plot.")
        return
    n = len(payloads)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    for k, payload in enumerate(payloads):
        ax = axes[k // ncols][k % ncols]
        x = np.array(payload["x_grid"])
        y = np.array(payload["y_grid"])
        Z = np.array(payload["delta_chi2"])
        X, Y = np.meshgrid(x, y)
        pcm = ax.pcolormesh(X, Y, np.minimum(Z, 25.0), shading="auto", cmap="viridis_r")
        fig.colorbar(pcm, ax=ax, label=r"$\Delta\chi^2$ (capped at 25)")
        styles = {"90": "-", "95": "--"}
        for lbl, curves in payload["limit_curves"].items():
            xs, ys = [], []
            for j, crossings in enumerate(curves):
                for c in crossings:
                    xs.append(x[j]); ys.append(c)
            if xs:
                ax.plot(xs, ys, "r" + styles.get(lbl, ":"), lw=1.4, marker=".", ms=3,
                        label=f"{lbl}% CL")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(payload["pair"][0]); ax.set_ylabel(payload["pair"][1])
        ax.legend(loc="best", fontsize=8)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    out = os.path.join(output_dir, "limits_summary.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Profiled coupling-vs-m_Zd exclusion limits.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to <config output_dir>/limits/.")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Subset of couplings to run (e.g. --pairs u_e g_D).")
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--num_cores", type=int, default=None)
    parser.add_argument("--no_checkpointing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config_file)
    ls = config.get("limits_settings", {})
    if args.n_steps is not None:
        ls["n_steps"] = args.n_steps

    pairs = ls.get("pairs", DEFAULT_PAIRS)
    if args.pairs:
        pairs = [p for p in pairs if p[1] in args.pairs]

    output_dir = args.output_dir or os.path.join(config.get("output_dir", "./results"), "limits")
    os.makedirs(output_dir, exist_ok=True)

    req_cores = args.num_cores or config.get("scan_settings", {}).get("num_cores", cpu_count())
    actual = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else cpu_count()
    num_cores = max(1, min(int(req_cores), actual))
    print(f"Limits scan: {len(pairs)} plane(s), {num_cores} core(s), output -> {output_dir}")

    repair_cfg = config.get("repair_settings", {})
    use_checkpointing = (not args.no_checkpointing) and ls.get("use_checkpointing", True)

    # Same worker JIT prewarm as main.py's scan pool (see src.physics.kernel.prewarm_worker).
    from src.physics.kernel import prewarm_worker

    payloads = []
    with Pool(processes=num_cores, initializer=prewarm_worker) as pool:
        for pair in pairs:
            json_path = os.path.join(output_dir, f"limits_2d_{pair[1]}_vs_{pair[0]}.json")
            if use_checkpointing and os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        payloads.append(json.load(f))
                    print(f"[Checkpoint] Loaded existing plane {pair[1]} vs {pair[0]}.")
                    continue
                except Exception:
                    pass
            payloads.append(run_pair(config, pair, ls, output_dir, pool, num_cores, repair_cfg))

    plot_summary(payloads, output_dir)

    # Reproducibility manifest: exact config + code state that produced these limits.
    try:
        from dark_portal_solver import write_run_manifest
        write_run_manifest(os.path.join(output_dir, "limits_manifest.json"), config)
    except Exception as e:
        print(f"[Warning] Could not write run manifest: {e}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
