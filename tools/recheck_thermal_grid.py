"""
recheck_thermal_grid.py -- the campaign's points that went through dp_si's transport (the dark portal's MeV regime:
mediators of 1 keV and up with a negligible potential, relics thermal in every collision), re-evaluated after the
resonant kernels' grid fix (dp_si._thin_grid, 2026-10-21: nearly coincident nodes made the interpolation of the
thermal kernels overshoot -- a crash, or isolated energies absorbed outright). At every stored solution (the plane
summaries' grid points and every refine_*.jsonl record) the chi^2 with the grid as it was (THERMAL_GRID_MIN_GAP = 0,
which must give back the stored value: the control) and with the fix, each grid in its own fresh processes (no cache
shared between them; NULRI_FLUX_CACHE is ignored). Results go next to each plane's summary, recheck.<task>.jsonl.

  python tools/recheck_thermal_grid.py plan <manifest.json> --tasks N      every affected plane's points, balanced
  python tools/recheck_thermal_grid.py run <manifest.json> <task> [--workers 32]
  python tools/recheck_thermal_grid.py report                              per plane: the control and the change
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
CAMPAIGN = os.path.join(REPO, "analysis", "configs", "campaign_ml")
M_MIN = 1e3                                   # dark_portal_solver.SI_COSMOLOGY_MIN_M_EV: dp_si below it never
GAPS = (0.5, 0.0)                             # the fix, then the grid as it was
# seconds per chi^2 evaluation on one core (a guess for the plan; the cost is one model evaluation)
SECONDS = {"now_ic26": 20, "now_events": 90, "proj_future": 60}
_W = {}


def solutions(cfgp):
    """[(source, held, values, stored chi2)] of a plane at masses >= M_MIN: the summary's grid points and the
    refine_*.jsonl records. Empty if the plane has no results."""
    from analysis.config import from_dict
    from analysis.run import asimov_only
    from check_edge_islands import load
    raw = json.load(open(cfgp))
    if not isinstance(raw, dict) or "planes" not in raw:          # inventories and other files beside the configs
        return []
    cfg = from_dict(raw, cfgp)
    out_dir = cfg.output_path("asimov" if asimov_only(cfg) else "data")
    if not glob.glob(os.path.join(out_dir, "plane_*.json")):
        return []
    s, x, y, chi, best = load(out_dir)
    xn, yn = s["names"]
    out = []
    for (i, j), b in best.items():
        if x[i] >= M_MIN and b.get("values"):
            out.append(("grid", {xn: float(x[i]), yn: float(y[j])}, b["values"], float(chi[i, j])))
    for f in sorted(glob.glob(os.path.join(out_dir, "refine_*.jsonl"))):
        if os.path.basename(f).startswith("refine_recheck"):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("held", {}).get(xn, 0.0) >= M_MIN and r.get("values"):
                out.append((os.path.basename(f), r["held"], r["values"], float(r["chi2"])))
    return out


def plan(manifest, tasks):
    configs = sorted(glob.glob(os.path.join(CAMPAIGN, "*", "*.json")))
    items = []
    for c in configs:
        pts = solutions(c)
        if pts:
            ds = os.path.basename(os.path.dirname(c))
            items.append(dict(config=os.path.relpath(c, REPO), n=len(pts), seconds=2 * SECONDS[ds]))
    total = sum(i["n"] * i["seconds"] for i in items)
    per = total / tasks
    shares, cur, load_ = [], [], 0.0
    for it in sorted(items, key=lambda i: -i["seconds"]):
        lo = 0
        while lo < it["n"]:
            room = max(1, int((per - load_) / it["seconds"]))
            hi = min(it["n"], lo + room)
            cur.append(dict(config=it["config"], lo=lo, hi=hi))
            load_ += (hi - lo) * it["seconds"]
            lo = hi
            if load_ >= per * 0.999 and len(shares) < tasks - 1:
                shares.append(cur)
                cur, load_ = [], 0.0
    if cur:
        shares.append(cur)
    json.dump(dict(tasks=shares, planes=len(items), points=sum(i["n"] for i in items), core_seconds=total),
              open(manifest, "w"))
    print(f"{len(items)} planes, {sum(i['n'] for i in items)} points, ~{total / 3600:.0f} core-hours (both grids), "
          f"{len(shares)} tasks -> {manifest}")


def _init(raw, cfgp, dc, gap):
    os.environ.pop("NULRI_FLUX_CACHE", None)
    import dp_si
    from analysis.config import from_dict
    dp_si.THERMAL_GRID_MIN_GAP = gap
    cfg = from_dict(raw, cfgp)
    _W.update(cfg=cfg, objective=cfg.objective(data_centre=dc), model=cfg.model())


def _one(task):
    """(source, held, chi2 or None if the point does not use dp_si, error)."""
    import dark_portal_solver as dps
    source, held, values = task
    v = dict(values)
    v.update(held)
    try:
        m = _W["model"]
        E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
        if params.get("si") is not None or not dps._use_si_cosmology(params, E):
            return source, held, None, None              # secret interactions, or the dark portal's own phase
        return source, held, float(_W["objective"].chi2(v)), None
    except Exception as e:                                   # a failing point is recorded, not fatal
        return source, held, None, repr(e)[:300]


def run(manifest, task, workers):
    from analysis.config import from_dict
    from analysis.run import asimov_only
    share = json.load(open(manifest))["tasks"][task]
    for item in share:
        cfgp = os.path.join(REPO, item["config"])
        raw = json.load(open(cfgp))
        cfg = from_dict(raw, cfgp)
        kind = "asimov" if asimov_only(cfg) else "data"
        out_dir = cfg.output_path(kind)
        dc = None
        if kind == "asimov":
            runs = json.load(open(os.path.join(out_dir, "manifest.json")))["runs"]
            dc = tuple(next(r["data_centre"] for r in reversed(runs) if r.get("data_centre")))
        pts = solutions(cfgp)[item["lo"]:item["hi"]]
        stored = {(s, json.dumps(h, sort_keys=True)): c for s, h, _, c in pts}
        path = os.path.join(out_dir, f"recheck.{task}.jsonl")
        done = set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["source"], json.dumps(r["held"], sort_keys=True), r["gap"]))
        with open(path, "a") as fh:
            for gap in GAPS:
                todo = [(s, h, v) for s, h, v, _ in pts if (s, json.dumps(h, sort_keys=True), gap) not in done]
                print(f"{item['config']} gap {gap}: {len(todo)} points ({len(pts) - len(todo)} done)", flush=True)
                if not todo:
                    continue
                with get_context("spawn").Pool(min(workers, len(todo)), initializer=_init,
                                               initargs=(raw, cfgp, dc, gap)) as pool:
                    for k, (s, h, chi2, err) in enumerate(pool.imap_unordered(_one, todo, chunksize=1)):
                        fh.write(json.dumps(dict(source=s, held=h, gap=gap, chi2=chi2, error=err,
                                                 stored=stored[(s, json.dumps(h, sort_keys=True))])) + "\n")
                        fh.flush()
                        if (k + 1) % 50 == 0:
                            print(f"PROGRESS {100 * (k + 1) / len(todo):.0f}% | {k + 1} of {len(todo)}", flush=True)
    print("done", flush=True)


def report():
    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "output", "analysis", "campaign_ml", "*", "*", "*"))):
        recs = {}
        for f in glob.glob(os.path.join(d, "recheck.*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs.setdefault((r["source"], json.dumps(r["held"], sort_keys=True)), {})[r["gap"]] = r
        if not recs:
            continue
        si = [g for g in recs.values() if 0.0 in g and 0.5 in g and g[0.0]["chi2"] is not None]
        err = sum(1 for g in recs.values() for r in g.values() if r.get("error"))
        ctrl = [abs(g[0.0]["chi2"] - g[0.0]["stored"]) for g in si]
        dlt = [g[0.5]["chi2"] - g[0.0]["chi2"] for g in si if g[0.5]["chi2"] is not None]
        name = os.path.relpath(d, os.path.join(REPO, "output", "analysis", "campaign_ml"))
        rows.append(name)
        print(f"{name:45s} {len(recs):5d} pts, {len(si):5d} via dp_si, errors {err}; control max |old - stored| "
              f"{max(ctrl) if ctrl else 0:.2e}; |new - old| > 0.05: {sum(abs(x) > 0.05 for x in dlt)}, > 1: "
              f"{sum(abs(x) > 1 for x in dlt)}, max {max(map(abs, dlt)) if dlt else 0:.2f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("manifest")
    p.add_argument("--tasks", type=int, required=True)
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("task", type=int)
    r.add_argument("--workers", type=int, default=32)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a.manifest, a.tasks)
    elif a.cmd == "run":
        run(a.manifest, a.task, a.workers)
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
