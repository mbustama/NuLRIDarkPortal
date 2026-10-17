"""
job_progress.py -- one progress line for the Tycho monitor (2026-10-17), printed to a job's output file:

  PROGRESS 37.2% | 3 of 12 planes finished | ETA Sun 18 Oct 18:40

The monitor shows a job's progress only when its output has such a line (the last one counts): "PROGRESS", the
percentage, then free text. slurm/common.sh runs this every PROGRESS_EVERY seconds (default 300) in campaign jobs
(PACKS set: an array task runs line SLURM_ARRAY_TASK_ID + 1 of the pack list) and clean-up jobs (TREE set: the planes
unfinished when the job started, as slurm/campaign_sweep.sh lists them).

A plane counts as finished when its plane_*.json exists, otherwise as the share of its grid points in its
checkpoint (plane_*.jsonl, first-pass points; at most 99%: the repair passes come after). Planes are weighted by
their estimated cost (tools/campaign_packs.py), so a slow plane counts for more than a fast one. The ETA extrapolates
the rate since the job's first progress line (kept in logs/progress_<job id>.json).
"""
import glob
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))


def plane_fraction(cfg_path):
    """(fraction done, finished?, grid points) of one configuration's plane."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    out = cfg["output_dir"]
    pl = cfg["planes"][0]
    total = int(pl["nx"]) * int(pl["ny"])
    if any(glob.glob(os.path.join(out, sub, "plane_*.json")) for sub in ("data", "asimov")):
        return 1.0, True, total
    done = 0
    for f in glob.glob(os.path.join(out, "*", "plane_*.jsonl")):
        seen = set()
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "index" in r:
                    seen.add(tuple(r["index"]))
        done = max(done, len(seen))
    return min(done / max(total, 1), 0.99), False, total


def weight(cfg_path, points):
    try:
        from campaign_packs import plane_cost
        dataset = os.path.basename(os.path.dirname(cfg_path))
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        return plane_cost(dataset, name, points, bpl="campaign_bpl" in cfg_path)[0]
    except Exception:
        return float(points)


def unfinished(tree):
    todo = []
    for path in sorted(glob.glob(os.path.join(tree, "*", "*.json"))):
        if os.path.basename(path) == "campaign.json":
            continue
        with open(path) as f:
            out = json.load(f)["output_dir"]
        if not any(glob.glob(os.path.join(out, sub, "plane_*.json")) for sub in ("data", "asimov")):
            todo.append(path)
    return todo


def main():
    os.chdir(REPO)
    job = os.environ.get("SLURM_JOB_ID", "local")
    state_path = os.path.join("logs", f"progress_{job}.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    if os.environ.get("PACKS"):
        k = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        with open(os.environ["PACKS"]) as f:
            lines = f.read().splitlines()
        configs = lines[k].split() if k < len(lines) else []
    elif os.environ.get("TREE"):
        configs = state.get("configs") or unfinished(os.environ["TREE"])
    else:
        return
    if not configs:
        print("PROGRESS 100.0% | nothing to do", flush=True)
        return
    num = den = 0.0
    finished = 0
    for c in configs:
        frac, fin, points = plane_fraction(c)
        w = weight(c, points)
        num += w * frac
        den += w
        finished += fin
    p = num / den if den > 0 else 0.0
    now = time.time()
    if "t0" not in state:
        state.update(t0=now, p0=p, configs=configs)
        os.makedirs("logs", exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f)
    rate = (p - state["p0"]) / (now - state["t0"]) if now > state["t0"] else 0.0
    eta = time.strftime("%a %d %b %H:%M", time.localtime(now + (1.0 - p) / rate)) if rate > 0 else "unknown"
    print(f"PROGRESS {100.0 * p:.1f}% | {finished} of {len(configs)} planes finished | ETA {eta}", flush=True)


if __name__ == "__main__":
    main()
