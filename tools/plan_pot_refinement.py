"""
plan_pot_refinement.py -- the mass-eps' refinement (Mauricio, 2026-10-21: after the mass-g_D scans): the potential
planes (dp_<flavor>_pot_x*, masses and g_D eps' a decade apart) sampled finer along their 95% edges, as a manifest
that tools/refine_grid.py runs (its records merge into the planes' reading: analysis/exclusion.field):
  at every grid mass, couplings every STEP decade within +-HALF decade of each crossing of the 95% edge (the column's
  reading, analysis/exclusion.field); between neighbouring grid masses, NEW masses 1/(NEW+1) decade apart, with
  couplings every STEP decade within +-HALF decade of either neighbour's crossings. Grid points left out.
Task indices start at OFFSET (record files refine_grid[_between].<task>.jsonl beside the grid extension's 0-199).
  python tools/plan_pot_refinement.py <manifest.json> --tasks N [--splits x1m24 x1m16] [--count]
(--splits: only the planes at those g_D sum|U|^2; 2026-10-22, the 1e-8 planes held until the relic-scattering
treatment below 1 keV is settled.)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
STEP, HALF, NEW, OFFSET = 0.1, 0.6, 3, 2000


def crossings(lx, G, F):
    """{log10 m: [log10 g of each 95% edge in the column]} from the reading's signed-distance field."""
    out = {}
    for k, m in enumerate(lx):
        col = np.asarray(F[:, k].filled(np.nan))
        s = np.sign(col)
        idx = np.flatnonzero((s[1:] * s[:-1]) < 0)
        out[round(float(m), 6)] = [float(0.5 * (G[i] + G[i + 1])) for i in idx]
    return out


def points(cfgp, out_dir):
    from analysis import exclusion as ex
    path = glob.glob(os.path.join(out_dir, "plane_*.json"))[0]
    s = json.load(open(path))
    lx_grid = np.log10(s["grids"][0])
    ly_grid = np.log10(s["grids"][1])
    lx, G, F, _, _ = ex.field(path)
    cr = crossings(lx, G, F)
    on_grid = lambda g: np.min(abs(ly_grid - g)) < 1e-6
    inside = lambda g: ly_grid[0] - 1e-9 <= g <= ly_grid[-1] + 1e-9
    around = lambda cs: sorted({round(float(c0 + d), 6) for c0 in cs for d in np.arange(-HALF, HALF + 1e-9, STEP)})
    pts = []
    for m in lx_grid:
        cs = cr.get(round(float(m), 6), [])
        pts += [(float(m), g, False) for g in around(cs) if inside(g) and not on_grid(g)]
    for a, b in zip(lx_grid[:-1], lx_grid[1:]):
        cs = cr.get(round(float(a), 6), []) + cr.get(round(float(b), 6), [])
        for k in range(1, NEW + 1):
            m = round(float(a + (b - a) * k / (NEW + 1)), 6)
            pts += [(m, g, True) for g in around(cs) if inside(g)]
    return pts


def main(argv=None):
    from analysis.config import from_dict
    from analysis.run import asimov_only
    from refine_grid import SECONDS, kind_of
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--tasks", type=int, required=True)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--splits", nargs="+", default=None, help="tags of the planes to take, e.g. x1m24 x1m16")
    a = ap.parse_args(argv)
    items = []
    for c in sorted(glob.glob(os.path.join(REPO, "analysis/configs/campaign_ml/*/dp_*_pot_*.json"))):
        if a.splits and not any(f"_pot_{t}_" in os.path.basename(c) for t in a.splits):
            continue
        cfg = from_dict(json.load(open(c)), c)
        out_dir = cfg.output_path("asimov" if asimov_only(cfg) else "data")
        if not glob.glob(os.path.join(out_dir, "plane_*.json")):
            continue
        pts = points(c, out_dir)
        if pts:
            items.append(dict(config=os.path.relpath(c, REPO), points=pts, seconds=SECONDS[kind_of(c)]))
    total = sum(len(i["points"]) * i["seconds"] for i in items)
    print(f"{len(items)} planes, {sum(len(i['points']) for i in items)} fits, ~{total / 3600:.0f} core-hours")
    if a.count:
        return 0
    per = total / a.tasks
    shares, cur, load = [], [], 0.0
    for it in sorted(items, key=lambda i: -i["seconds"]):
        pts = list(it["points"])
        while pts:
            room = max(1, int((per - load) / it["seconds"]))
            take, pts = pts[:room], pts[room:]
            cur.append(dict(config=it["config"], points=take))
            load += len(take) * it["seconds"]
            if load >= per * 0.999 and len(shares) < a.tasks - 1:
                shares.append(cur)
                cur, load = [], 0.0
    if cur:
        shares.append(cur)
    json.dump(dict(tasks=[[] for _ in range(OFFSET)] + shares, total_fits=sum(len(i["points"]) for i in items),
                   core_seconds=total), open(a.manifest, "w"))
    print(f"{len(shares)} tasks from {OFFSET} -> {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
