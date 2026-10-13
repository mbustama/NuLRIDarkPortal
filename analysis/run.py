"""
run.py

The command line of the analysis:

  python -m analysis CONFIG.json [--fit] [--profiles [NAME ...]] [--planes] [--asimov]
                                 [--workers N] [--repair-passes K] [--summarize]

Without a task flag it runs everything the configuration lists. Results go to the
configuration's output_dir, under data/ (the published or projected measurement) or asimov/
(the surface re-centred on the model's prediction at the configuration's 'asimov' values):

  fit.json, fit.jsonl                   the global fit and its checkpoint
  profile_<name>.json / .jsonl / .pdf   a profile with its intervals, checkpoint and figure
  plane_<x>_<y>.json / .jsonl / .pdf    a plane (Delta chi^2 map), checkpoint and figure
  manifest.json                         provenance of every run: code version, configuration,
                                        machine, settings, timing

Rerunning resumes from the checkpoints (delete a checkpoint to recompute that scan). Every run
ends by re-summarizing all scans present against the lowest chi^2 found anywhere; --summarize
does only that.
"""
import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

from . import REPO_DIR, config as config_mod, intervals, plots, scans


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", REPO_DIR, *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def provenance(cfg) -> dict:
    """Code version, configuration digest, machine and library versions."""
    import numba
    import scipy
    return dict(time=datetime.datetime.now().isoformat(timespec="seconds"),
                git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain", "--", "src", "analysis")),
                config=cfg.path, config_digest=cfg.digest(), host=socket.gethostname(),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                numba=numba.__version__, argv=sys.argv)


def asimov_centre(cfg) -> tuple:
    """(f_e, f_mu) predicted at the configuration's 'asimov' values (free parameters not named
    there take their starts)."""
    values = {p.name: p.start for p in cfg.parameters}
    values.update(cfg.asimov)
    f = cfg.model(cache_size=1).fractions(values)
    return float(f[0]), float(f[1])


def _write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _load_scan(path: str, names, grids, identity):
    recs = scans._best_by_index(scans._read_jsonl(path, identity))
    shape = tuple(len(g) for g in grids)
    chi2 = np.full(shape, np.inf)
    values = {}
    for ix, r in recs.items():
        if len(ix) == len(shape) and all(0 <= i < n for i, n in zip(ix, shape)):
            chi2[ix] = r["chi2"]
            values[ix] = r["values"]
    return chi2, values


def summarize(cfg, out_dir: str, data_centre=None, log=print) -> dict:
    """Re-reads the fit and every scan checkpoint in out_dir, takes the lowest chi^2 found
    anywhere as the reference, and writes each scan's summary (Delta chi^2, intervals) and
    figure. Returns the reference minimum and where it was found."""
    fit_path = os.path.join(out_dir, "fit.json")
    best = json.load(open(fit_path)) if os.path.exists(fit_path) else None
    if best is not None and best.get("id") != cfg.identity(data_centre):
        log("  fit.json was computed under another configuration: ignored")
        best = None
    ref = dict(chi2=best["chi2"] if best else np.inf, values=best["values"] if best else None, where="global fit")
    loaded = []
    for s in cfg.profiles:
        p = os.path.join(out_dir, f"profile_{s.name}.jsonl")
        if os.path.exists(p):
            chi2, vals = _load_scan(p, [s.name], [s.grid], scans.scan_identity(cfg, [s.name], [s.grid], data_centre))
            loaded.append(("profile", [s.name], [s.grid], chi2, vals))
    rasters = {}
    for s in cfg.planes:
        p = os.path.join(out_dir, f"plane_{s.x}_{s.y}.jsonl")
        if os.path.exists(p):
            chi2, vals = _load_scan(p, [s.x, s.y], [s.x_grid, s.y_grid],
                                    scans.scan_identity(cfg, [s.x, s.y], [s.x_grid, s.y_grid], data_centre))
            loaded.append(("plane", [s.x, s.y], [s.x_grid, s.y_grid], chi2, vals))
            rasters[(s.x, s.y)] = s.raster
    for kind, names, grids, chi2, vals in loaded:
        if np.isfinite(chi2).any() and chi2.min() < ref["chi2"] - scans.REPAIR_TOL:
            ix = np.unravel_index(np.argmin(chi2), chi2.shape)
            ref = dict(chi2=float(chi2.min()), values=vals[tuple(int(i) for i in ix)], where=f"{kind} {'/'.join(names)}")
    if ref["where"] != "global fit" and best is not None:
        log(f"  note: the {ref['where']} found chi2 = {ref['chi2']:.6g} below the global fit's {best['chi2']:.6g}; "
            f"Delta chi^2 is measured from it")
    for kind, names, grids, chi2, vals in loaded:
        logs = [cfg.parameter(n).space == "log10" for n in names]
        d = chi2 - ref["chi2"]
        summary = dict(kind=kind, names=names, grids=[list(map(float, g)) for g in grids], log=logs if kind == "plane" else logs[0],
                       chi2=np.where(np.isfinite(chi2), chi2, None).tolist(), dchi2=np.where(np.isfinite(d), d, None).tolist(),
                       chi2_min=ref["chi2"], chi2_min_from=ref["where"], best=ref["values"], title=f"{cfg.name} ({os.path.basename(out_dir)})",
                       complete=bool(np.isfinite(chi2).all()))
        if kind == "profile":
            thr = cfg.thresholds["1d"]
            summary["thresholds"] = thr
            summary["intervals"] = intervals.summarize_profile(grids[0], chi2, ref["chi2"], thr, logs[0])["intervals"]
            stem = os.path.join(out_dir, f"profile_{names[0]}")
            summary["dchi2"] = [None if v is None else float(v) for v in summary["dchi2"]]
            _write_json(stem + ".json", summary)
            plots.plot_profile({**summary, "dchi2": [np.inf if v is None else v for v in summary["dchi2"]]}, stem + ".pdf")
            for lab, ivs in summary["intervals"].items():
                log(f"  {names[0]} at {lab}: " + ("; ".join(intervals.describe(iv) for iv in ivs) or "no interval"))
        else:
            stem = os.path.join(out_dir, f"plane_{names[0]}_{names[1]}")
            raster = rasters.get(tuple(names))
            if raster is None:
                thr = cfg.thresholds["2d"]
                summary["thresholds"] = thr
                _write_json(stem + ".json", summary)
                plots.plot_plane({**summary, "dchi2": np.where(np.isfinite(d), d, np.inf).tolist()}, stem + ".pdf")
                continue
            # read per value of the raster parameter (decided 2026-10-13), at the 1D thresholds
            thr = cfg.thresholds["1d"]
            ax = names.index(raster)
            other = names[1 - ax]
            c = chi2 if ax == 0 else chi2.T
            rs = intervals.summarize_raster(grids[ax], grids[1 - ax], c, thr, ref["chi2"],
                                            cfg.parameter(other).space == "log10")
            dcol = rs["dchi2"] if ax == 0 else rs["dchi2"].T
            summary.update(raster=raster, thresholds=thr, columns=rs["columns"], elevated=rs["elevated"],
                           dchi2_global=summary["dchi2"],
                           dchi2=np.where(np.isfinite(dcol), dcol, None).tolist())
            _write_json(stem + ".json", summary)
            plots.plot_plane({**summary, "dchi2": np.where(np.isfinite(dcol), dcol, np.inf).tolist()}, stem + ".pdf")
            if rs["elevated"]:
                log(f"  {len(rs['elevated'])} of {len(grids[ax])} {raster} columns never reach Delta chi^2 < "
                    f"{max(thr.values()):g} of the best fit within the scanned {other} "
                    f"({min(rs['elevated']):.3g} to {max(rs['elevated']):.3g}): extend the {other} range there")
    return ref


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="analysis configuration (JSON)")
    ap.add_argument("--fit", action="store_true", help="run the global fit")
    ap.add_argument("--profiles", nargs="*", default=None, help="run these profiles (all if none named)")
    ap.add_argument("--planes", action="store_true", help="run the planes")
    ap.add_argument("--asimov", action="store_true", help="fit the Asimov data set of the configuration")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: the cores this job has, minus one locally)")
    ap.add_argument("--repair-passes", type=int, default=3, help="maximum repair passes per scan")
    ap.add_argument("--summarize", action="store_true", help="only re-summarize existing results")
    args = ap.parse_args(argv)

    import dp_core
    cfg = config_mod.load(args.config)
    out_dir = cfg.output_path("asimov" if args.asimov else "data")
    os.makedirs(out_dir, exist_ok=True)
    workers = args.workers or dp_core.default_n_cores()
    if args.summarize:
        centre = asimov_centre(cfg) if args.asimov else None
        summarize(cfg, out_dir, centre)
        return 0
    everything = not (args.fit or args.profiles is not None or args.planes)
    run = dict(provenance=provenance(cfg), workers=workers, tasks=[])
    t_start = time.perf_counter()
    data_centre = None
    if args.asimov:
        data_centre = asimov_centre(cfg)
        print(f"Asimov data: surface re-centred at (f_e, f_mu) = ({data_centre[0]:.6f}, {data_centre[1]:.6f})")
    run["data_centre"] = data_centre

    fit_path = os.path.join(out_dir, "fit.json")
    stored = json.load(open(fit_path)) if os.path.exists(fit_path) else None
    if stored is not None and stored.get("id") != cfg.identity(data_centre):
        print("fit.json was computed under another configuration: refitting")
        stored = None
    if everything or args.fit or stored is None:
        print(f"Global fit of {cfg.name} ({len(cfg.parameters)} free parameters, {1 + cfg.fit.n_random} starts, {workers} workers)")
        res = scans.global_fit(cfg, data_centre, workers, checkpoint=os.path.join(out_dir, "fit.jsonl"))
        _write_json(fit_path, dict(values=res["values"], chi2=res["chi2"], nfev=res["nfev"], seconds=res["seconds"],
                                   starts=res["starts"], data_centre=data_centre, id=res["id"]))
        print(f"  best fit chi2 = {res['chi2']:.6g} at " + ", ".join(f"{k} = {v:.6g}" for k, v in res["values"].items()))
        run["tasks"].append(dict(task="fit", seconds=res["seconds"], nfev=res["nfev"]))
    best = json.load(open(fit_path))["values"]

    wanted = None if (everything or args.profiles == []) else set(args.profiles or [])
    if everything or args.profiles is not None:
        for s in cfg.profiles:
            if wanted is not None and s.name not in wanted:
                continue
            print(f"Profile of {s.name}: {len(s.grid)} points from {s.grid[0]:.4g} to {s.grid[-1]:.4g}")
            r = scans.profile(cfg, s.name, s.grid, best, data_centre, workers,
                              checkpoint=os.path.join(out_dir, f"profile_{s.name}.jsonl"), max_passes=args.repair_passes)
            run["tasks"].append(dict(task=f"profile {s.name}", seconds=r.seconds, nfev=r.nfev, passes=r.passes))
    if everything or args.planes:
        for s in cfg.planes:
            print(f"Plane {s.x} x {s.y}: {len(s.x_grid)} x {len(s.y_grid)} points")
            r = scans.plane(cfg, s.x, s.y, s.x_grid, s.y_grid, best, data_centre, workers,
                            checkpoint=os.path.join(out_dir, f"plane_{s.x}_{s.y}.jsonl"), max_passes=args.repair_passes)
            run["tasks"].append(dict(task=f"plane {s.x}/{s.y}", seconds=r.seconds, nfev=r.nfev, passes=r.passes))
    print("Summary")
    ref = summarize(cfg, out_dir, data_centre)
    run["chi2_min"] = ref["chi2"]
    run["seconds"] = time.perf_counter() - t_start
    man_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else dict(runs=[])
    manifest["config"] = cfg.raw
    manifest["runs"].append(run)
    _write_json(man_path, manifest)
    print(f"Done in {run['seconds']:.1f} s; results in {out_dir}")
    return 0
