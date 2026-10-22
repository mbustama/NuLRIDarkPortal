"""
rerun_relic_scattering.py -- the campaign's points that the relic-scattering code changes (tools/
validate_relic_scattering.py flags; 2026-10-22), refitted with it. Each refit replaces the stored solution at its point
(analysis/exclusion.corrected reads rerun_rs.*.jsonl after the thermal-grid corrections, and takes them over whatever
their value: new physics, not a better minimum). Left out, measured unchanged (validation, 2026-10-22): the points
with the same transport and no soft scatterings (Delta chi^2 exactly 0), and the g_D sum|U|^2 = 1e-16 potential
planes' points with the soft scatterings only (|Delta chi^2| <= 0.07, median 1e-5).

  plan <flags.jsonl> <manifest.json> --tasks N [--seconds-json S] [--only family,...] [--per-plane K]
      the points to refit, balanced in run time over N tasks (seconds per fit by data set and family: SECONDS, or a
      pilot's measurement); --per-plane K takes at most K per plane and kind (a pilot)
  run <manifest.json> <task> --data-root R [--workers 32]
      each point fitted from its stored solution, mass and coupling held, with the code this file runs from; records
      rerun_rs.<task>.jsonl beside the plane's summary in R's output/ (resumes: points already there are skipped)
  times <data-root> [--tasks a-b]
      per data set and family, the seconds per fit measured in the records (a pilot's)
"""
import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
from validate_relic_scattering import _family, _out_dir, solutions  # noqa: E402

SEED = 20261022
# seconds per fit on one core, a first guess (refine_grid.SECONDS' 'dpul' for the new transport, 'ul' else); replaced
# by the pilot's measurement (--seconds-json)
SECONDS = {"now_ic26": 90, "now_events": 540, "proj_future": 145}


def _skip(r):
    """Measured unchanged: no new transport and no scatterings, or the 1e-16 planes' scatterings only."""
    if "error" in r or "skip" in r or not r["flagged"]:
        return True
    return r["new"] == r["old"] and "_pot_x1m16_" in r["config"]


def plan(flags_path, manifest, n_tasks, seconds_json=None, only=None, per_plane=None):
    secs = dict(SECONDS)
    fam_secs = {}
    if seconds_json:
        fam_secs = {tuple(k.split("|")): v for k, v in json.load(open(seconds_json)).items()}
    rng = np.random.default_rng(SEED)
    pts = []
    for line in open(flags_path):
        r = json.loads(line)
        if _skip(r):
            continue
        ds, fam = _family(r["config"])
        if only and fam not in only:
            continue
        kind = "transport" if r["new"] != r["old"] else "jumps"
        pts.append(dict(config=r["config"], source=r["source"], held=r["held"], kind=kind,
                        seconds=float(fam_secs.get((ds, fam, kind), fam_secs.get((ds, fam), secs[ds])))))
    if per_plane:
        by = {}
        for p in pts:
            by.setdefault((p["config"], p["kind"]), []).append(p)
        pts = []
        for k in sorted(by):
            rows = by[k]
            pts += [rows[i] for i in sorted(rng.choice(len(rows), size=min(per_plane, len(rows)), replace=False))]
    # balance: longest first into the least loaded task, grouped by plane within a task (one objective per plane)
    pts.sort(key=lambda p: -p["seconds"])
    loads = np.zeros(n_tasks)
    tasks = [[] for _ in range(n_tasks)]
    for p in pts:
        k = int(np.argmin(loads))
        tasks[k].append(p)
        loads[k] += p["seconds"]
    for t in tasks:
        t.sort(key=lambda p: (p["config"], -p["seconds"]))
    total = float(loads.sum())
    json.dump(dict(tasks=tasks, points=len(pts), core_seconds=total), open(manifest, "w"))
    print(f"{len(pts)} points, ~{total / 3600:.0f} core-hours, {n_tasks} tasks of ~{loads.max() / 3600 / 32:.2f} h "
          f"on 32 cores -> {manifest}")
    by = {}
    for p in pts:
        k = _family(p["config"]) + (p["kind"],)
        by[k] = by.get(k, 0) + 1
    for k in sorted(by):
        print(f"  {k[0]:12s} {k[1]:10s} {k[2]:10s} {by[k]:7d}")
    return 0


_W = {}


def _fit(task):
    """One refit, in a worker that keeps one plane's objective at a time (the tasks come grouped by plane)."""
    from analysis import scans
    cfgp, raw, dc, held, start, settings, src = task
    t0 = time.time()
    try:
        if _W.get("cfgp") != cfgp:
            scans._init_worker(raw, cfgp, dc)
            _W["cfgp"] = cfgp
        r = scans._fit_held(held, [start], settings)
        return cfgp, dict(source=src, held=held, chi2=float(r.chi2), values=r.values, seconds=time.time() - t0,
                          error=None)
    except Exception as e:
        return cfgp, dict(source=src, held=held, chi2=None, values=None, seconds=time.time() - t0,
                          error=repr(e)[:300])


def run(manifest, task, data_root, workers):
    """A task's points, every plane's in one pool (2026-10-22: a pool per plane left most workers idle where a plane
    had few points)."""
    from analysis.run import asimov_only
    from refine_grid import Progress
    share = json.load(open(manifest))["tasks"][task]
    jobs, outs = [], {}
    for cfg in sorted({p["config"] for p in share}):
        cfgp = os.path.join(REPO, cfg)
        raw = json.load(open(cfgp))
        out_dir, c = _out_dir(cfgp, data_root)
        dc = None
        if asimov_only(c):
            runs = json.load(open(os.path.join(out_dir, "manifest.json")))["runs"]
            dc = tuple(next(r["data_centre"] for r in reversed(runs) if r.get("data_centre")))
        done = set()
        for f in glob.glob(os.path.join(out_dir, "rerun_rs.*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("chi2") is not None:
                    done.add((r["source"], json.dumps(r["held"], sort_keys=True)))
        sols = {(s_, json.dumps(h, sort_keys=True)): v for s_, h, v in solutions(cfgp, data_root)}
        settings = {**asdict(c.fit), "n_random": 0}
        n0 = len(jobs)
        for p in share:
            if p["config"] != cfg:
                continue
            key = (p["source"], json.dumps(p["held"], sort_keys=True))
            if key in done or key not in sols:
                continue
            start = {k: v for k, v in sols[key].items() if k not in p["held"]}
            jobs.append((cfgp, raw, dc, p["held"], start, settings, p["source"]))
        outs[cfgp] = out_dir
        print(f"{cfg}: {len(jobs) - n0} fits", flush=True)
    print(f"{len(jobs)} fits in all", flush=True)
    if not jobs:
        return 0
    progress = Progress(len(jobs), "fits")
    files = {}
    try:
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for k, (cfgp, r) in enumerate(pool.imap_unordered(_fit, jobs, chunksize=1)):
                if cfgp not in files:
                    files[cfgp] = open(os.path.join(outs[cfgp], f"rerun_rs.{task}.jsonl"), "a")
                files[cfgp].write(json.dumps(r) + "\n")
                files[cfgp].flush()
                progress(k + 1)
    finally:
        for fh in files.values():
            fh.close()
    print("done", flush=True)
    return 0


def times(data_root, lo=None, hi=None):
    from validate_relic_scattering import _planes
    acc = {}
    for cfgp in _planes():
        out_dir, _ = _out_dir(cfgp, data_root)
        for f in glob.glob(os.path.join(out_dir, "rerun_rs.*.jsonl")):
            t = int(f.rsplit(".", 2)[-2])
            if lo is not None and not lo <= t <= hi:
                continue
            for line in open(f):
                r = json.loads(line)
                k = _family(os.path.relpath(cfgp, REPO))
                acc.setdefault(k, []).append(r["seconds"])
    out = {}
    for k in sorted(acc):
        s = np.array(acc[k])
        out["|".join(k)] = float(np.mean(s))
        print(f"{k[0]:12s} {k[1]:10s} {len(s):5d} fits: mean {s.mean():7.1f} s, median {np.median(s):7.1f}, max "
              f"{s.max():7.1f}")
    json.dump(out, open(os.path.join(REPO, "output", "analysis", "rerun_rs_seconds.json"), "w"), indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("flags")
    a.add_argument("manifest")
    a.add_argument("--tasks", type=int, required=True)
    a.add_argument("--seconds-json", default=None)
    a.add_argument("--only", default=None)
    a.add_argument("--per-plane", type=int, default=None)
    a = sub.add_parser("run")
    a.add_argument("manifest")
    a.add_argument("task", type=int)
    a.add_argument("--data-root", default=REPO)
    a.add_argument("--workers", type=int, default=32)
    a = sub.add_parser("times")
    a.add_argument("data_root")
    a.add_argument("--tasks", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return plan(a.flags, a.manifest, a.tasks, a.seconds_json, a.only.split(",") if a.only else None, a.per_plane)
    if a.cmd == "run":
        return run(a.manifest, a.task, a.data_root, a.workers)
    lo = hi = None
    if a.tasks:
        lo, hi = (int(v) for v in a.tasks.split("-"))
    return times(a.data_root, lo, hi)


if __name__ == "__main__":
    sys.exit(main())
