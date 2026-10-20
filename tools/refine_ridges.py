"""
refine_ridges.py -- finer sampling across the thin ridges of the ultralight planes (campaign_ml, 2026-10-20): rows of
small excluded islands or allowed holes at one coupling-to-mass ratio g/m (one strength of the cosmological potential),
thinner than the grid, which a coarse grid catches only at some points. For each ridge of
tools/ridges.json and each mass of the plane, 11 couplings across +-0.5 decade around
g = m 10^ratio, each fitted (the campaign's own fitting code, mass and coupling held) from the stored solutions of the
two bracketing grid points; written to <plane output>/<data|asimov>/refine_ridges.<part>.jsonl.
  python tools/refine_ridges.py <plane number in ridges.json> <part> <parts> [--workers 16]
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
sys.path[:0] = [os.path.join(REPO, "src"), REPO]


def run(task):
    from analysis import scans
    held, starts, settings = task
    r = scans._fit_held(held, starts, settings)
    return held, float(r.chi2), r.values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plane", type=int)
    ap.add_argument("part", type=int)
    ap.add_argument("parts", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--between", type=int, default=0,
                    help="also this many masses between neighbouring grid masses (log-spaced), so that a ridge is "
                         "sampled densely along its length; written to refine_ridges_between.<part>.jsonl")
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    t = json.load(open(os.path.join(REPO, "tools", "ridges.json")))[a.plane]
    cfgp = os.path.join(REPO, f"analysis/configs/campaign_ml/{t['ds']}/{t['plane']}.json")
    raw = json.load(open(cfgp))
    cfg = from_dict(raw, cfgp)
    dc = asimov_centre(cfg) if t["kind"] == "asimov" else None
    out_dir = os.path.join(REPO, raw["output_dir"], t["kind"])
    s = json.load(open(glob.glob(f"{out_dir}/plane_*.json")[0]))
    x, y = np.array(s["grids"][0]), np.array(s["grids"][1])
    xn, yn = s["names"]
    best = {}
    for f in glob.glob(f"{out_dir}/plane_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "chi2" in r and (tuple(r["index"]) not in best or r["chi2"] < best[tuple(r["index"])]["chi2"]):
                best[tuple(r["index"])] = r
    settings = {**asdict(cfg.fit), "n_random": 0}
    tasks = []
    if a.between:
        masses = [(i, float(x[i] * (x[i + 1] / x[i]) ** (k / (a.between + 1))))
                  for i in range(len(x) - 1) for k in range(1, a.between + 1)]
    else:
        masses = [(i, float(m)) for i, m in enumerate(x)]
    for ratio in t["ratios"]:
        for i, m in masses:
            for delta in np.linspace(-0.5, 0.5, 11):
                g = float(m * 10 ** (ratio + delta))
                if not (y[0] < g < y[-1]):
                    continue
                jl = int(np.searchsorted(y, g) - 1)
                cols = (i, i + 1) if a.between else (i,)
                starts = [best.get((c, jj), {}).get("values") for c in cols for jj in (jl, jl + 1)]
                tasks.append(({xn: m, yn: g}, [st for st in starts if st], settings))
    tasks = tasks[a.part::a.parts]
    print(f"{t['ds']}/{t['plane']}: part {a.part} of {a.parts}, {len(tasks)} fits on {a.workers} workers", flush=True)
    path = f"{out_dir}/refine_ridges{'_between' if a.between else ''}.{a.part}.jsonl"
    with get_context("spawn").Pool(a.workers, initializer=scans._init_worker, initargs=(raw, cfgp, dc)) as pool, \
            open(path, "w") as f:
        for k, (held, chi2, values) in enumerate(pool.imap_unordered(run, tasks)):
            f.write(json.dumps(dict(held=held, chi2=chi2, values=values)) + "\n")
            f.flush()
            if (k + 1) % 20 == 0:
                print(f"PROGRESS {100 * (k + 1) / len(tasks):.0f}% | {k + 1} of {len(tasks)} fits", flush=True)
    print("done", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
