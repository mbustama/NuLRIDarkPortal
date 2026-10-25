"""
pilot_fits.py -- the time a plane's fits take, before a campaign (2026-10-21): n points spread over the plane's grid
(every k-th mass and coupling), each fitted (the campaign's own code, mass and coupling held) from the configuration's
starts, on a pool; prints each fit's chi^2 and seconds, and their median and mean. Asimov planes take the data centre
from --centre-from (another plane of the same data set and composition, its manifest.json: tools/resummarize.py) so
that the pilot does not spend a solver run on it.
  python tools/pilot_fits.py <config> [--n 16] [--workers 16] [--centre-from <output dir of another plane>]
                             [--points x,y ...]
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]


def run(task):
    from analysis import scans
    held, start, settings = task
    t0 = time.perf_counter()
    r = scans._fit_held(held, [start], settings)
    return held, float(r.chi2), time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--centre-from", default=None)
    ap.add_argument("--points", nargs="+", default=None,
                    help="explicit points 'x,y' in the plane's axes (2026-10-25) instead of the spread over its grid")
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_only
    raw = json.load(open(a.config))
    cfg = from_dict(raw, a.config)
    dc = None
    if asimov_only(cfg):
        runs = json.load(open(os.path.join(a.centre_from, "manifest.json")))["runs"]
        dc = tuple(next(r["data_centre"] for r in reversed(runs) if r.get("data_centre")))
    pl = raw["planes"][0]
    xs = np.logspace(*np.log10(pl["x_range"]), pl["nx"])
    ys = np.logspace(*np.log10(pl["y_range"]), pl["ny"])
    k = int(np.ceil(np.sqrt(a.n)))
    pts = [(float(x), float(y)) for x in xs[np.linspace(0, len(xs) - 1, k).astype(int)]
           for y in ys[np.linspace(0, len(ys) - 1, k).astype(int)]][:a.n]
    if a.points:
        pts = [tuple(float(v) for v in p.split(",")) for p in a.points]
    start = {p.name: p.start for p in cfg.parameters}
    settings = {**asdict(cfg.fit), "n_random": 0}
    tasks = [({pl["x"]: x, pl["y"]: y}, {kk: v for kk, v in start.items() if kk not in (pl["x"], pl["y"])}, settings)
             for x, y in pts]
    print(f"{raw['name']}: {len(tasks)} fits on {a.workers} workers", flush=True)
    secs = []
    t0 = time.perf_counter()
    with get_context("spawn").Pool(min(a.workers, len(tasks)), initializer=scans._init_worker,
                                   initargs=(raw, os.path.abspath(a.config), dc)) as pool:
        for held, chi2, s in pool.imap_unordered(run, tasks):
            secs.append(s)
            print(f"  m {held[pl['x']]:.1e} {pl['y']} {held[pl['y']]:.1e}: chi2 {chi2:.3f}, {s:.0f} s", flush=True)
    print(f"PILOT {raw['name']}: median {np.median(secs):.0f} s, mean {np.mean(secs):.0f} s, max {max(secs):.0f} s "
          f"per fit; wall {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
