"""
check_band_numerics.py -- is the thin excluded band of the fitted-composition ultralight planes (campaign_ml,
2026-10-21) a numerical artifact of the flavor calculation? At 7 points on the band (its strongest coupling at that
mass, from tools/refine_edges.py and refine_ridges.py), the couplings 0.3 decade below and above it, and the column's
smallest coupling (no effect, the reference) are refitted (the campaign's own fitting code, mass and coupling held)
with the solver's grids as in the campaign ('standard') or all refined twice over ('doubled': twice the energy and
redshift bins, Sun paths, Earth chords and steps inside the Earth, half the transit angle step); the Asimov data are
recomputed with the same grids. A real band keeps its delta chi^2 when the grids are doubled.
Written to <plane output>/<data|asimov>/band_numerics.<grid>.jsonl.
  python tools/check_band_numerics.py <future_lmu | future_ltau | events_lmu> <standard | doubled> [--workers N]
"""
import argparse
import copy
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
from check_edge_islands import PLANES, load  # noqa: E402

POINTS = [("future_ltau", -22, -27.30), ("future_ltau", -16, -24.90), ("future_ltau", -14, -20.70),
          ("future_ltau", -8, -9.75), ("future_lmu", -22, -27.50), ("events_lmu", -22, -27.60),
          ("events_lmu", -24, -24.40)]   # (plane, log10 m, log10 g of the band's strongest coupling)


def doubled(raw):
    raw = copy.deepcopy(raw)
    g = raw["solver"]["grid"]
    for k in ("n_E_bins", "n_z_bins", "n_sun_paths", "n_earth_chords", "N_steps_earth"):
        g[k] = 2 * int(g[k])
    g["transit_dtheta"] = 0.5 * float(g["transit_dtheta"])
    return raw


def run(task):
    from analysis import scans
    label, held, start, settings = task
    r = scans._fit_held(held, [start], settings)
    return label, held, float(r.chi2), r.values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plane", choices=sorted(PLANES))
    ap.add_argument("grid", choices=("standard", "doubled"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    ds, name, kind = PLANES[a.plane]
    cfgp = os.path.join(REPO, f"analysis/configs/campaign_ml/{ds}/{name}.json")
    raw0 = json.load(open(cfgp))
    raw = doubled(raw0) if a.grid == "doubled" else raw0
    cfg = from_dict(raw, cfgp)
    dc = asimov_centre(cfg) if kind == "asimov" else None
    out_dir = os.path.join(REPO, raw0["output_dir"], kind)
    s, x, y, chi, best = load(out_dir)
    xn, yn = s["names"]
    lx, ly = np.log10(x), np.log10(y)
    settings = {**asdict(cfg.fit), "n_random": 0}
    tasks = []
    for plane, m, g in POINTS:
        if plane != a.plane:
            continue
        i = int(np.argmin(abs(lx - m)))
        for label, lg in (("below", g - 0.3), ("band", g), ("above", g + 0.3), ("reference", ly[0])):
            j = int(np.argmin(abs(ly - lg)))
            tasks.append((f"{m}|{label}", {xn: float(x[i]), yn: float(10 ** lg)}, best[(i, j)]["values"], settings))
    print(f"{ds}/{name}, {a.grid} grids: {len(tasks)} fits on {min(a.workers, len(tasks))} workers", flush=True)
    path = f"{out_dir}/band_numerics.{a.grid}.jsonl"
    with get_context("spawn").Pool(min(a.workers, len(tasks)), initializer=scans._init_worker,
                                   initargs=(raw, cfgp, dc)) as pool, open(path, "w") as f:
        for label, held, chi2, values in pool.imap_unordered(run, tasks):
            f.write(json.dumps(dict(label=label, held=held, chi2=chi2, values=values)) + "\n")
            f.flush()
            print(label, round(chi2, 3), flush=True)
    print("done", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
