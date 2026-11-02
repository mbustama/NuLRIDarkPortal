"""
reeval_events.py -- the HESE + DR2 chi^2 re-evaluated with its nuisance profile from fixed starts (docs item 21,
2026-10-24): the profile's warm start made a chi^2 depend on what the process had evaluated before (up to 1.5 at 0.6%
of points; the lower value is the right one). At every now_events plane's reference points (each column's minimum,
and the grid's smallest coupling: the no-effect chi^2) and at its points within WINDOW of the threshold (Delta chi^2
from the column's minimum, every correction applied), the chi^2 at the point's current solution with the profile run
from the priors' centres and from STARTS seeded random starts (tools/events_profile_check.py), the lowest kept:
reeval_events.<task>.jsonl beside the plane, which analysis/exclusion.corrected applies last, taking over where lower.

  plan <manifest.json> --tasks N
  run <manifest.json> <task> [--workers 32]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
CONFIGS = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_events")
WINDOW = 2.0
STARTS = 16
_W = {}


def _points(cfgp):
    """[(source, held, values, chi2)] of a plane at its current state: the summary's grid and the refinement records,
    the corrections applied (exclusion.corrected), the reruns' solutions where they replaced a point."""
    from analysis import exclusion as ex
    from check_edge_islands import load
    from analysis.config import from_dict
    cfg = from_dict(json.load(open(cfgp)), cfgp)
    out_dir = cfg.output_path("data")
    if not glob.glob(os.path.join(out_dir, "plane_*.json")):
        return []
    s, x, y, chi, best = load(out_dir)
    xn, yn = s["names"]
    corr = ex.corrected(out_dir, xn, yn)
    vals = {}
    for f in glob.glob(os.path.join(out_dir, "rerun_rs.[0-9]*.jsonl")):
        for r in ex._records([f]):
            if r.get("values"):
                vals[(r["source"], ex._key(r["held"], xn, yn))] = r["values"]
    pts = []
    for (i, j), b in best.items():
        k = ("grid", ex._key({xn: x[i], yn: y[j]}, xn, yn))
        c = corr.get(k, chi[i, j])
        v = vals.get(k, b.get("values"))
        if v and c == c:
            pts.append(("grid", {xn: float(x[i]), yn: float(y[j])}, v, float(c)))
    for f in glob.glob(os.path.join(out_dir, "refine_*.jsonl")):
        if os.path.basename(f).startswith("refine_recheck"):
            continue
        src = os.path.basename(f)
        for r in ex._records([f], corr, xn, yn):
            k = (src, ex._key(r["held"], xn, yn))
            v = vals.get(k, r.get("values"))
            if v and r.get("chi2") is not None:
                pts.append((src, r["held"], v, float(r["chi2"])))
    return pts, xn, yn, float(np.log10(y[0]))


def plan(manifest, n_tasks):
    from analysis import exclusion as ex
    jobs = []
    for cfgp in sorted(glob.glob(os.path.join(CONFIGS, "*.json"))):
        raw = json.load(open(cfgp))
        if not (isinstance(raw, dict) and "planes" in raw and "solver" in raw):
            continue
        res = _points(cfgp)
        if not res:
            continue
        pts, xn, yn, lg0 = res
        cols = {}
        for p in pts:
            cols.setdefault(round(float(np.log10(p[1][xn])), 6), []).append(p)
        take = []
        for lm, cp in cols.items():
            cmin = min(c for *_, c in cp)
            for p in cp:
                d = p[3] - cmin
                if d == 0.0 or abs(d - ex.THRESHOLD) <= WINDOW or abs(np.log10(p[1][yn]) - lg0) < 1e-6:
                    take.append(dict(config=os.path.relpath(cfgp, REPO), source=p[0], held=p[1], values=p[2],
                                     stored=p[3]))
        jobs += take
        print(f"{os.path.basename(cfgp)}: {len(take)} of {len(pts)}", flush=True)
    tasks = [jobs[k::n_tasks] for k in range(n_tasks)]
    for t in tasks:
        t.sort(key=lambda p: p["config"])
    json.dump(dict(tasks=tasks, points=len(jobs)), open(manifest, "w"))
    print(f"{len(jobs)} points -> {manifest}")
    return 0


def _one(p):
    cfgp = os.path.join(REPO, p["config"])
    try:
        if _W.get("path") != cfgp:
            from analysis.config import from_dict
            _W.clear()
            obj = from_dict(json.load(open(cfgp)), cfgp).objective()
            ev = obj.events
            terms = getattr(ev, "terms", [ev])
            orig = {id(t): type(t).astro for t in terms}
            for t in terms:
                def astro(values, t=t):
                    t._A_last = orig[id(t)](t, values)
                    return t._A_last
                t.astro = astro
            _W.update(path=cfgp, obj=obj, terms=terms)
        obj, terms = _W["obj"], _W["terms"]
        v = dict(p["values"])
        v.update(p["held"])
        for t in terms:
            t._z0 = np.zeros(len(t.names))
        total = float(obj.chi2(v))                   # the profile from the priors' centres
        rest = total
        best = 0.0
        key = json.dumps([p["config"], p["source"], p["held"]], sort_keys=True)
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
        for t in terms:
            A = t._A_last
            lo = np.array([-np.inf if b[0] is None else (b[0] - c) / s for b, c, s in zip(t.bounds, t.centre, t.sigma)])
            hi = np.array([np.inf if b[1] is None else (b[1] - c) / s for b, c, s in zip(t.bounds, t.centre, t.sigma)])
            vals = []
            for k in range(STARTS + 1):
                t._z0 = np.zeros(len(t.names)) if k == 0 else np.clip(rng.standard_normal(len(t.names)), lo, hi)
                x, _ = t.profile(A)
                vals.append(float(t.exact(x, A)["total"]))
            rest -= vals[0]
            best += min(vals)
        return dict(source=p["source"], held=p["held"], chi2=rest + best, chi2_centre=total, stored=p["stored"],
                    error=None), cfgp
    except Exception as e:
        return dict(source=p["source"], held=p["held"], chi2=None, stored=p["stored"], error=repr(e)[:300]), cfgp


def run(manifest, task, workers):
    from analysis.config import from_dict
    from refine_grid import Progress
    os.environ.pop("NULRI_FLUX_CACHE", None)
    share = json.load(open(manifest))["tasks"][task]
    progress = Progress(len(share), "points")
    files = {}
    with get_context("spawn").Pool(min(workers, max(1, len(share)))) as pool:
        for k, (r, cfgp) in enumerate(pool.imap_unordered(_one, share, chunksize=1)):
            if cfgp not in files:
                out_dir = from_dict(json.load(open(cfgp)), cfgp).output_path("data")
                files[cfgp] = open(os.path.join(out_dir, f"reeval_events.{task}.jsonl"), "a")
            files[cfgp].write(json.dumps(r) + "\n")
            files[cfgp].flush()
            progress(k + 1)
    for fh in files.values():
        fh.close()
    print("done", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("manifest")
    a.add_argument("--tasks", type=int, required=True)
    a = sub.add_parser("run")
    a.add_argument("manifest")
    a.add_argument("task", type=int)
    a.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return plan(a.manifest, a.tasks)
    return run(a.manifest, a.task, a.workers)


if __name__ == "__main__":
    sys.exit(main())
