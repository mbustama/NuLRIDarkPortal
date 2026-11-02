"""
check_thin_band.py -- is the thin excluded band of the future detectors' L_e-L_mu ultralight plane with fitted source
compositions (campaign_ml, 2026-10-20: one grid point, g = 3e-28 at m = 1e-20 eV, at Delta chi^2 = 4.35 between
neighbours at 0.86 and 2.06) a real feature or a fit that missed its minimum? With the campaign's own fitting code:
  (a) that point refitted from several starts -- its own solution, its four neighbours', the pion-source plane's
      (with the electron fractions at 1/3), the global best fit's, and random starts;
  (b) a finer coupling grid, 10 values between 1e-28 and 1e-27, at the five masses nearest 1e-22..1e-18 eV, each
      from the stored solutions of the bracketing grid points and of the pion-source plane.
Delta chi^2 is per mass, against the plane's column minimum.
  python tools/check_thin_band.py [--workers 16]
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
CFG = os.path.join(REPO, "analysis/configs/campaign_ml/proj_future/le_lmu_ul_fx.json")
OUT = os.path.join(REPO, "output/analysis/campaign_ml/proj_future")


def load_plane(name):
    s = json.load(open(glob.glob(f"{OUT}/{name}/asimov/plane_*.json")[0]))
    best = {}
    for f in glob.glob(f"{OUT}/{name}/asimov/plane_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "chi2" in r and (tuple(r["index"]) not in best or r["chi2"] < best[tuple(r["index"])]["chi2"]):
                best[tuple(r["index"])] = r
    chi = np.array([[np.nan if v is None else v for v in row] for row in s["chi2"]])
    return np.array(s["grids"][0]), np.array(s["grids"][1]), chi, best, s["names"]


def run(task):
    from analysis import scans
    label, held, starts, settings = task
    r = scans._fit_held(held, starts, settings)
    return label, held, float(r.chi2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    raw = json.load(open(CFG))
    cfg = from_dict(raw, CFG)
    dc = asimov_centre(cfg)
    x, y, chi, best, names = load_plane("le_lmu_ul_fx")
    _, _, chi_pion, best_pion, _ = load_plane("le_lmu_ul")
    xn, yn = names
    base = {**asdict(cfg.fit), "n_random": 0}
    pion = lambda ix: {**best_pion[ix]["values"], "f_e_S": 1 / 3, "f_e_S_gal": 1 / 3} if ix in best_pion else None
    tasks = []
    # (a) the point from several starts, one fit each
    i = int(np.argmin(abs(np.log(x / 1e-20))))
    j = int(np.argmin(abs(np.log(y / 3e-28))))
    held = {xn: float(x[i]), yn: float(y[j])}
    gfit = json.load(open(glob.glob(f"{OUT}/le_lmu_ul_fx/asimov/fit.json")[0]))
    starts = {"own": best[(i, j)]["values"], "global best": gfit.get("values", gfit.get("best")),
              "pion plane": pion((i, j))}
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        starts[f"neighbour {di:+d},{dj:+d}"] = best.get((i + di, j + dj), {}).get("values")
    for k, (lab, s) in enumerate(starts.items()):
        if s:
            tasks.append((f"a|{lab}", held, [s], base))
    tasks.append(("a|8 random starts", held, [starts["own"]], {**base, "n_random": 8}))
    # (b) a finer coupling grid at five masses
    for m in (1e-22, 1e-21, 1e-20, 1e-19, 1e-18):
        ii = int(np.argmin(abs(np.log(x / m))))
        for g in np.logspace(-28, -27, 10):
            jl = int(np.searchsorted(y, g) - 1)
            sts = [best.get((ii, jl), {}).get("values"), best.get((ii, jl + 1), {}).get("values"), pion((ii, jl)),
                   pion((ii, jl + 1))]
            tasks.append((f"b|{x[ii]:.1e}", {xn: float(x[ii]), yn: float(g)}, [s for s in sts if s], base))
    print(f"{len(tasks)} fits on {a.workers} workers", flush=True)
    with get_context("spawn").Pool(a.workers, initializer=scans._init_worker,
                                   initargs=(raw, os.path.abspath(CFG), dc)) as pool:
        res = pool.map(run, tasks, chunksize=1)
    colmin = {ii: np.nanmin(chi[ii]) for ii in range(len(x))}
    print(f"\n(a) m = {x[i]:.1e} eV, g = {y[j]:.1e}: stored chi2 {chi[i, j]:.3f}, dchi2 {chi[i, j] - colmin[i]:.2f} "
          f"(pion-source plane there: {chi_pion[i, j]:.3f})")
    for lab, h, c in res:
        if lab.startswith("a|"):
            print(f"   from {lab[2:]:18s}: chi2 {c:.3f}  dchi2 {c - colmin[i]:6.2f}")
    print("\n(b) dchi2 per mass on the finer coupling grid (the stored grid's values in brackets)")
    rows = {}
    for lab, h, c in res:
        if lab.startswith("b|"):
            rows.setdefault(h[xn], []).append((h[yn], c))
    for m, pts in rows.items():
        ii = int(np.argmin(abs(x - m)))
        stored = " ".join(f"[{y[k]:.0e}:{chi[ii, k] - colmin[ii]:.2f}]" for k in range(len(y)) if 3e-29 <= y[k] <= 3e-27)
        print(f"   m={m:.1e}: " + " ".join(f"{g:.1e}:{c - colmin[ii]:.2f}" for g, c in pts) + f"\n            stored: {stored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
