"""
exact_progress.py -- PROGRESS lines for exact-profile plane runs (pilots and the campaign), with the measured cost per
point projected onto the whole exact campaign (2026-10-18).

Reads every plane checkpoint (plane_*.jsonl) and global fit (fit.jsonl) under a tree of planes laid out as
<tree>/<data set>/<plane>/..., and prints

    PROGRESS <p>% | <points done> of <total> | <rate> points/min | projected full campaign <hours> h on <cores> cores

The projection: per data set and plane kind (balanced potential, coupling-split potential, ultralight gauge, MeV), the
mean seconds per point (each point's fit in one single-threaded worker, its repairs included), times that group's
points in the whole campaign (analysis/configs/campaign_exact: every plane of every data set, the _fx twins included),
plus each plane's global fit, over the cores. An MeV point's scan of the lightest mass is shared through the disk
cache by the configurations with the same energy grid (now_ic26, proj_2050 and their twins: 4; now_events and
proj_future with theirs: 2): its share counts 1/n of its seconds. Groups not measured yet take the mean of their kind
(marked 'partial').

  python tools/exact_progress.py <tree> [--total N] [--cores 176] [--watch [--interval 5] [--heartbeat 110]]
  python tools/exact_progress.py <tree> --table          # per-group costs
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETS = ("now_ic26", "now_events", "proj_2050", "proj_future")
KINDS = ("bal", "split", "ul", "mev")
SHARE = {"now_ic26": 4, "proj_2050": 4, "now_events": 2, "proj_future": 2}


def kind_of(name: str) -> str:
    if "_mev" in name:
        return "mev"
    if "_ul" in name:
        return "ul"
    return "bal" if "_bal" in name else "split"


def campaign_counts(root: str = None):
    """{(data set, kind): (points, planes)} of the whole exact campaign."""
    root = root or os.path.join(REPO, "analysis", "configs", "campaign_exact")
    out = defaultdict(lambda: [0, 0])
    for ds in SETS:
        for f in glob.glob(os.path.join(root, ds, "*.json")):
            if os.path.basename(f) == "campaign.json":
                continue
            c = json.load(open(f))
            for p in c.get("planes", []):
                g = out[(ds, kind_of(os.path.basename(f)))]
                g[0] += int(p["nx"]) * int(p["ny"])
                g[1] += 1
    return {k: tuple(v) for k, v in out.items()}


def read_tree(tree: str):
    """(points: {(ds, kind): [cost per point]}, fits: {(ds, kind): [global-fit seconds]}, done, times)."""
    points, fits, times = defaultdict(list), defaultdict(list), []
    done = 0
    for f in glob.glob(os.path.join(tree, "**", "plane_*.jsonl"), recursive=True):
        rel = os.path.relpath(f, tree).split(os.sep)
        ds = next((x for x in rel if x in SETS), None)
        name = rel[rel.index(ds) + 1] if ds and rel.index(ds) + 1 < len(rel) else os.path.basename(os.path.dirname(f))
        cost, seen = defaultdict(float), set()
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "chi2" not in r or "index" not in r:
                continue
            ix = tuple(r["index"])
            sec = float(r.get("seconds", 0.0))
            if kind_of(name) == "mev" and ds in SHARE:
                sec -= float(r.get("scan_seconds", 0.0)) * (1.0 - 1.0 / SHARE[ds])
            cost[ix] += sec
            if ix not in seen and r.get("pass", 0) == 0:
                seen.add(ix)
                if "t" in r:
                    times.append(float(r["t"]))
        done += len(seen)
        points[(ds, kind_of(name))].extend(cost[ix] for ix in seen)
    for f in glob.glob(os.path.join(tree, "**", "fit.jsonl"), recursive=True):
        rel = os.path.relpath(f, tree).split(os.sep)
        ds = next((x for x in rel if x in SETS), None)
        name = os.path.basename(os.path.dirname(f))
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("final") and "seconds" in r:
                fits[(ds, kind_of(name))].append(float(r["seconds"]))
    return points, fits, done, sorted(times)


def projection(points, fits, cores: int):
    """(hours, partial, per-group table rows)."""
    counts = campaign_counts()
    mean = {k: sum(v) / len(v) for k, v in points.items() if v}
    fmean = {k: sum(v) / len(v) for k, v in fits.items() if v}
    by_kind = defaultdict(list)
    for (ds, kd), m in mean.items():
        by_kind[kd].append(m)
    overall = (sum(sum(v) for v in points.values()) / max(1, sum(len(v) for v in points.values()))) if mean else None
    total, partial, rows = 0.0, False, []
    for (ds, kd), (n_pts, n_planes) in sorted(counts.items()):
        m = mean.get((ds, kd))
        if m is None:
            partial = True
            m = (sum(by_kind[kd]) / len(by_kind[kd])) if by_kind[kd] else overall
        if m is None:
            return None, True, rows
        g = fmean.get((ds, kd), sum(fmean.values()) / len(fmean) if fmean else 0.0)
        sub = n_pts * m + n_planes * g
        total += sub
        rows.append((ds, kd, len(points.get((ds, kd), [])), m, g, n_pts, sub / 3600.0 / cores))
    return total / 3600.0 / cores, partial, rows


def line(tree: str, total: int, cores: int, t_start: float = None) -> str:
    points, fits, done, times = read_tree(tree)
    p = 100.0 * done / total if total else 0.0
    rate = 0.0
    if times:
        t0 = times[0] if t_start is None else min(t_start, times[0])
        span = max(time.time() - t0, 60.0)
        rate = 60.0 * len(times) / span
    h, partial, _ = projection(points, fits, cores)
    proj = "n/a" if h is None else f"{h:.1f}" + (" (partial)" if partial else "")
    return (f"PROGRESS {min(p, 100.0):.1f}% | {done} of {total} | {rate:.1f} points/min | projected full campaign "
            f"{proj} h on {cores} cores")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tree")
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--cores", type=int, default=176)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--heartbeat", type=float, default=110.0)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)
    if a.table:
        points, fits, done, _ = read_tree(a.tree)
        h, partial, rows = projection(points, fits, a.cores)
        print(f"{'set':12s} {'kind':6s} {'measured':>8s} {'s/point':>8s} {'s/fit':>7s} {'points':>7s} {'h':>6s}")
        for ds, kd, n, m, g, n_pts, hh in rows:
            print(f"{ds:12s} {kd:6s} {n:8d} {m:8.1f} {g:7.0f} {n_pts:7d} {hh:6.2f}")
        print(f"campaign: {h if h is None else round(h, 1)} h on {a.cores} cores{' (partial)' if partial else ''}; "
              f"{done} points measured")
        return 0
    if not a.watch:
        print(line(a.tree, a.total, a.cores), flush=True)
        return 0
    t_start, last_sig, last_print = time.time(), None, 0.0
    while True:
        files = glob.glob(os.path.join(a.tree, "**", "*.jsonl"), recursive=True)
        sig = tuple(sorted((f, os.path.getsize(f)) for f in files if os.path.exists(f)))
        now = time.time()
        if sig != last_sig or now - last_print >= a.heartbeat:
            try:
                print(line(a.tree, a.total, a.cores), flush=True)
            except Exception as e:                    # a line cut mid-write, a file moved: next time
                print(f"PROGRESS (unreadable now: {e})", flush=True)
            last_sig, last_print = sig, now
        if os.path.exists(os.path.join(a.tree, "STOP_PROGRESS")):
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
