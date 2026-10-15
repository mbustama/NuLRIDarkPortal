"""
combine_split_planes.py -- the second curve of a potential plane: the bound profiled over the split of the coupling
product (decided 2026-10-14, docs/dev/SCOPE_WIDENING_PLAN.md decision 4).

The potential fixes only P = g_D eps' |U|^2, but the relic neutrinos' screening grows with each coupling separately.
A plane run with dark_portal.coupling_split = 'balanced' (x = g_D |U|^2 = eps' = sqrt(P): screening negligible) is
the first curve; runs with the neutrinos' side fixed at x (coupling_split = x, eps' = P / x) are the variants. At each
(m_Zd, P) the profiled chi^2 is the smallest over the balanced run and every variant that covers that point, read per
mediator mass at the 1D thresholds as the planes are (a point is excluded only if every split is). A variant need
only cover the masses where its screening differs from the balanced split's (m_Zd below ~10 m_D(x)): elsewhere both
give the same chi^2, so the balanced run stands for it.

  python tools/combine_split_planes.py --balanced <run dir> --variants <run dir> ... --out <dir>

Each run directory is an analysis output (its data/plane_<x>_<y>.json and data/manifest.json); the variants' grids
must be subsets of the balanced run's (matched by value).
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

from analysis import intervals, plots  # noqa: E402
from analysis.model import parse_coupling_split  # noqa: E402


def load_run(run_dir: str):
    """(summary, coupling_split) of an analysis run's plane."""
    data = run_dir
    for sub in ("data", "asimov"):                  # the future detectors' projections run on Asimov data
        if glob.glob(os.path.join(run_dir, sub, "plane_*.json")):
            data = os.path.join(run_dir, sub)
            break
    planes = sorted(glob.glob(os.path.join(data, "plane_*.json")))
    if len(planes) != 1:
        raise ValueError(f"{run_dir}: expected one plane_*.json, found {planes}")
    with open(planes[0]) as f:
        summary = json.load(f)
    with open(os.path.join(data, "manifest.json")) as f:
        cfg = json.load(f)["config"]
    split = parse_coupling_split(cfg.get("solver", {}).get("dark_portal", {}).get("coupling_split", "fixed"))
    return summary, split


def _index(values, grid, what):
    grid = np.asarray(grid, dtype=float)
    out = []
    for v in values:
        k = int(np.argmin(np.abs(np.log(grid) - np.log(v))))
        if abs(grid[k] / v - 1.0) > 1e-9:
            raise ValueError(f"{what} {v:g} is not on the balanced run's grid")
        out.append(k)
    return out


def combine(balanced, variants):
    """The profiled summary from the balanced run's summary and [(summary, x), ...] of the variants."""
    names = balanced["names"]
    m_grid, P_grid = (np.asarray(g, dtype=float) for g in balanced["grids"])
    chi2 = np.asarray(balanced["chi2"], dtype=float).copy()      # [i_m, i_P]
    chi2_min = float(balanced["chi2_min"])
    used = np.zeros(chi2.shape, dtype=int)                       # 0: balanced; k: variant k (1-based)
    for n, (s, x) in enumerate(variants, start=1):
        if s["names"] != names:
            raise ValueError(f"variant x = {x:g} scans {s['names']}, the balanced run {names}")
        if not isinstance(x, float):
            raise ValueError(f"a variant needs a numeric coupling_split, got {x!r}")
        im = _index(s["grids"][0], m_grid, "m_Zd")
        iP = _index(s["grids"][1], P_grid, "P")
        c = np.asarray(s["chi2"], dtype=float)
        for a, k in enumerate(im):
            for b, j in enumerate(iP):
                if P_grid[j] <= x and np.isfinite(c[a, b]) and c[a, b] < chi2[k, j]:
                    chi2[k, j] = c[a, b]
                    used[k, j] = n
        chi2_min = min(chi2_min, float(s["chi2_min"]))
    chi2_min = min(chi2_min, float(np.min(chi2[np.isfinite(chi2)])))
    thr = balanced["thresholds"]
    rs = intervals.summarize_raster(m_grid, P_grid, chi2, thr, chi2_min, True)
    out = dict(kind="plane", names=names, grids=balanced["grids"], log=balanced.get("log", [True, True]),
               chi2=chi2.tolist(), chi2_min=chi2_min, chi2_min_from="the lowest of the runs' global fits",
               title=balanced.get("title", "") + " -- profiled over the coupling split",
               complete=bool(np.isfinite(chi2).all()), raster=names[0], thresholds=thr, columns=rs["columns"],
               elevated=rs["elevated"], dchi2=np.where(np.isfinite(rs["dchi2"]), rs["dchi2"], None).tolist(),
               splits=["balanced"] + [x for _, x in variants], split_used=used.tolist())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--balanced", required=True, help="the balanced run's output directory")
    ap.add_argument("--variants", nargs="+", required=True, help="the variant runs' output directories")
    ap.add_argument("--out", required=True, help="directory for the profiled plane (json, pdf)")
    a = ap.parse_args(argv)
    bal, split = load_run(a.balanced)
    if split != "balanced":
        raise ValueError(f"{a.balanced}: coupling_split is {split!r}, not 'balanced'")
    variants = [load_run(d) for d in a.variants]
    out = combine(bal, variants)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"plane_{out['names'][0]}_{out['names'][1]}_profiled_split")
    with open(stem + ".json", "w") as f:
        json.dump(out, f)
    plots.plot_plane({**out, "dchi2": np.where(np.asarray(out["dchi2"], dtype=object) == None, np.inf,  # noqa: E711
                                               out["dchi2"]).astype(float).tolist()}, stem + ".pdf")
    print(f"wrote {stem}.json, .pdf")


if __name__ == "__main__":
    main()
