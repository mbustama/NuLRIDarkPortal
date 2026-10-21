"""
refine_grid.py -- the campaign's planes sampled finer and higher (Mauricio, 2026-10-21): couplings up to 1 at every
mass of a plane, and new masses 1/32 decade apart (three between the MeV planes' grid masses) between 10^2.5 and
10^7.5 eV, each with couplings 1e-5 - 1 every 0.2 decade. Each point is fitted (the campaign's own code, mass and
coupling held) from the stored solution of the nearest grid point. Results go next to the plane's summary, in
refine_grid.<task>.jsonl (at grid masses) and refine_grid_between.<task>.jsonl (new masses), which
analysis/exclusion.py merges into the plane's excluded region; the plane's own checkpoint is not touched.

Two steps:
  python tools/refine_grid.py plan <manifest.json> --tasks N [--configs <config> ...]
      every target plane's points, split into N tasks of about equal run time (seconds per fit by data set and plane
      kind, measured on the campaign: SECONDS), written to the manifest;
  python tools/refine_grid.py run <manifest.json> <task> [--workers 32]
      that task's points, plane by plane; points already in the plane's refine_grid*.jsonl are skipped, so a rerun
      redoes only what is missing. Asimov planes take the data centre from their manifest.json.
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
from check_edge_islands import load  # noqa: E402

TOP = 0.0                              # log10 of the largest coupling
STEP_G = 0.2                           # decades between new couplings
MASS_RANGE = (2.5, 7.5)                # log10 m of the new masses, every 1/32 decade
G_NEW = (-5.0, TOP)                    # log10 g at the new masses
# seconds per fit, measured on the campaign (median, one core): (data set, 'mev' | 'ul')
SECONDS = {("now_ic26", "mev"): 88, ("now_events", "mev"): 300, ("proj_future", "mev"): 153,
           ("now_ic26", "ul"): 4, ("now_events", "ul"): 22, ("proj_future", "ul"): 12}


def points(cfgp):
    """[(log10 m, log10 g, between)] for the plane of configuration cfgp."""
    pl = json.load(open(cfgp))["planes"][0]
    lx = np.linspace(*np.log10(pl["x_range"]), pl["nx"])
    ly = np.linspace(*np.log10(pl["y_range"]), pl["ny"])
    step_y = ly[1] - ly[0]
    out = []
    if ly[-1] < TOP - 1e-9:                                     # higher couplings at every grid mass
        g_up = np.arange(ly[-1] + step_y, TOP + 1e-9, step_y)
        out += [(float(m), float(g), False) for m in lx for g in g_up]
    js = np.arange(int(np.ceil(MASS_RANGE[0] * 32 - 1e-9)), int(np.floor(MASS_RANGE[1] * 32 + 1e-9)) + 1)
    new_m = [j / 32 for j in js if lx[0] < j / 32 < lx[-1] and np.min(abs(lx - j / 32)) > 1e-6]
    g_new = np.arange(G_NEW[0], G_NEW[1] + 1e-9, STEP_G)
    out += [(float(m), float(g), True) for m in new_m for g in g_new]
    return out


def kind_of(cfgp):
    ds = os.path.basename(os.path.dirname(cfgp))
    return ds, ("ul" if "_ul" in os.path.basename(cfgp) or "_pot_" in os.path.basename(cfgp) else "mev")


def plan(manifest, tasks, configs):
    items = []
    for c in configs:
        pts = points(c)
        if pts:
            items.append(dict(config=os.path.relpath(c, REPO), points=pts, seconds=SECONDS[kind_of(c)]))
    total = sum(len(i["points"]) * i["seconds"] for i in items)
    per = total / tasks
    shares, cur, load_ = [], [], 0.0
    for it in sorted(items, key=lambda i: -i["seconds"]):       # cut each plane's points into shares of ~equal time
        pts = list(it["points"])
        while pts:
            room = max(1, int((per - load_) / it["seconds"]))
            take, pts = pts[:room], pts[room:]
            cur.append(dict(config=it["config"], points=take))
            load_ += len(take) * it["seconds"]
            if load_ >= per * 0.999 and len(shares) < tasks - 1:
                shares.append(cur)
                cur, load_ = [], 0.0
    if cur:
        shares.append(cur)
    json.dump(dict(tasks=shares, total_fits=sum(len(i["points"]) for i in items), core_seconds=total),
              open(manifest, "w"))
    print(f"{len(items)} planes, {sum(len(i['points']) for i in items)} fits, {total / 3600:.0f} core-hours, "
          f"{len(shares)} tasks of ~{per / 3600:.1f} core-hours each -> {manifest}")


def run_one(task):
    from analysis import scans
    held, start, settings, between = task
    r = scans._fit_held(held, [start], settings)
    return held, float(r.chi2), r.values, between


def run(manifest, task, workers):
    from analysis import scans
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
        s, x, y, chi, best = load(out_dir)
        xn, yn = s["names"]
        lx, ly = np.log10(x), np.log10(y)
        done = set()
        for f in glob.glob(os.path.join(out_dir, "refine_grid*.jsonl")):
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((round(np.log10(r["held"][xn]), 6), round(np.log10(r["held"][yn]), 6)))
        settings = {**asdict(cfg.fit), "n_random": 0}
        tasks = []
        for m, g, between in item["points"]:
            if (round(m, 6), round(g, 6)) in done:
                continue
            i, j = int(np.argmin(abs(lx - m))), int(np.argmin(abs(ly - g)))
            tasks.append(({xn: float(10 ** m), yn: float(10 ** g)}, best[(i, j)]["values"], settings, between))
        print(f"{item['config']}: {len(tasks)} fits ({len(item['points']) - len(tasks)} already done) on "
              f"{min(workers, max(1, len(tasks)))} workers", flush=True)
        if not tasks:
            continue
        files = {b: open(os.path.join(out_dir, f"refine_grid{'_between' if b else ''}.{task}.jsonl"), "a")
                 for b in (False, True)}
        with get_context("spawn").Pool(min(workers, len(tasks)), initializer=scans._init_worker,
                                       initargs=(raw, cfgp, dc)) as pool:
            for k, (held, chi2, values, b) in enumerate(pool.imap_unordered(run_one, tasks)):
                files[b].write(json.dumps(dict(held=held, chi2=chi2, values=values)) + "\n")
                files[b].flush()
                if (k + 1) % 50 == 0:
                    print(f"PROGRESS {100 * (k + 1) / len(tasks):.0f}% | {k + 1} of {len(tasks)} fits", flush=True)
        for f in files.values():
            f.close()
    print("done", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("manifest")
    p.add_argument("--tasks", type=int, required=True)
    p.add_argument("--configs", nargs="+", default=None)
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("task", type=int)
    r.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        configs = a.configs or sorted(
            c for ds in ("now_ic26", "now_events", "proj_future")
            for c in glob.glob(os.path.join(REPO, "analysis/configs/campaign_ml", ds, "*.json"))
            if os.path.basename(c)[:-5].replace("_fx", "") in
            {"le_lmu_ul", "le_ltau_ul", "le_lmu_mev", "le_ltau_mev", "dp_e_mev", "dp_mu_mev", "dp_tau_mev"})
        plan(a.manifest, a.tasks, configs)
    else:
        run(a.manifest, a.task, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
