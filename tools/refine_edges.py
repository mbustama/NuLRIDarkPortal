"""
refine_edges.py -- finer sampling along the thin excluded band of the fitted-composition ultralight planes (campaign_ml,
2026-10-20). With the source compositions fitted, a strong effect is absorbed by a change of composition; only where
it switches on, along the edge of the pion-decay plane's excluded region, does it change with energy and stay
excluded -- a band ~0.5 decade wide, flat near 3e-28, climbing 3.5-6 decades per decade of mass between 1e-15 and
1e-13 eV, where the grid catches it only at isolated points (tools/check_edge_islands.py: those points are real, every
start agrees). Followed here:
  (a) along the pion-decay plane's edge (its lowest excluded coupling per mass), 1e-30 to 1e-13 eV: at every grid
      mass, couplings 0.1 decade apart across +-0.6 decade around it; between neighbouring grid masses, enough masses
      that the band moves at most 0.25 decade from one to the next (at least 3), couplings across both neighbours'
      edges +-0.6 decade;
  (b) the IceCube events' row along the upper edge of their pion-decay region, g = m 10^-0.5, 1e-28 to 1e-20 eV:
      3 masses between grid masses, +-0.6 decade.
Each fit from the stored solution of the nearest grid point. Written to <plane output>/<data|asimov>/
refine_edges.<part>.jsonl (grid masses) and refine_edges_between.<part>.jsonl (between them).
  python tools/refine_edges.py <future_lmu | future_ltau | events_lmu> <part> <parts> [--workers 8] [--count]
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
from check_edge_islands import PLANES, load  # noqa: E402

DIAGONAL = {"events_lmu": (-0.5, -28, -20)}   # log10(g/m), log10 m range


def run(task):
    from analysis import scans
    held, start, settings, between = task
    r = scans._fit_held(held, [start], settings)
    return held, float(r.chi2), r.values, between


def points(x, y, chi_pion, plane):
    """[(m, g, between)] -- (a) and (b) of the module docstring"""
    lx, ly = np.log10(x), np.log10(y)
    d = chi_pion - np.nanmin(chi_pion, axis=1, keepdims=True)
    edge = {}
    for i in range(len(x)):
        ex = np.flatnonzero(d[i] >= 3.84)
        if len(ex) and -30 <= lx[i] <= -13:
            edge[i] = ly[ex[0]]
    out = []
    win = lambda lo, hi: np.arange(lo, hi + 1e-9, 0.1)
    for i, e in edge.items():
        out += [(lx[i], g, False) for g in win(e - 0.6, e + 0.6)]
        if i + 1 in edge:
            e1 = edge[i + 1]
            n = max(4, int(np.ceil(abs(e1 - e) / 0.25)))
            for k in range(1, n):
                out += [(lx[i] + k / n * (lx[i + 1] - lx[i]), g, True) for g in win(min(e, e1) - 0.6, max(e, e1) + 0.6)]
    if plane in DIAGONAL:
        r, m0, m1 = DIAGONAL[plane]
        for i in range(len(x)):
            if m0 <= lx[i] <= m1:
                for k in range(4 if lx[i] < m1 else 1):
                    m = lx[i] + k / 4 * (lx[min(i + 1, len(x) - 1)] - lx[i])
                    out += [(m, g, k > 0) for g in win(m + r - 0.6, m + r + 0.6)]
    return [(m, g, b) for m, g, b in out if ly[0] < g < ly[-1]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plane", choices=sorted(PLANES))
    ap.add_argument("part", type=int)
    ap.add_argument("parts", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--count", action="store_true", help="only print the number of fits")
    a = ap.parse_args(argv)
    ds, name, kind = PLANES[a.plane]
    cfgp = os.path.join(REPO, f"analysis/configs/campaign_ml/{ds}/{name}.json")
    raw = json.load(open(cfgp))
    out_dir = os.path.join(REPO, raw["output_dir"], kind)
    s, x, y, chi, best = load(out_dir)
    _, _, _, chi_pion, _ = load(out_dir.replace(f"/{name}/", f"/{name[:-3]}/"))
    pts = points(x, y, chi_pion, a.plane)
    if a.count:
        print(a.plane, len(pts), "fits,", sum(b for *_, b in pts), "between grid masses")
        return 0
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    cfg = from_dict(raw, cfgp)
    dc = asimov_centre(cfg) if kind == "asimov" else None
    xn, yn = s["names"]
    settings = {**asdict(cfg.fit), "n_random": 0}
    lx, ly = np.log10(x), np.log10(y)
    tasks = []
    for m, g, b in pts[a.part::a.parts]:
        i, j = int(np.argmin(abs(lx - m))), int(np.argmin(abs(ly - g)))
        near = sorted(best, key=lambda ij: (abs(ij[0] - i) + abs(ij[1] - j), abs(ij[1] - j)))[0] if (i, j) not in best \
            else (i, j)
        tasks.append(({xn: float(10 ** m), yn: float(10 ** g)}, best[near]["values"], settings, b))
    print(f"{ds}/{name}: part {a.part} of {a.parts}, {len(tasks)} fits on {a.workers} workers", flush=True)
    files = {b: open(f"{out_dir}/refine_edges{'_between' if b else ''}.{a.part}.jsonl", "w") for b in (False, True)}
    with get_context("spawn").Pool(a.workers, initializer=scans._init_worker, initargs=(raw, cfgp, dc)) as pool:
        for k, (held, chi2, values, b) in enumerate(pool.imap_unordered(run, tasks)):
            files[b].write(json.dumps(dict(held=held, chi2=chi2, values=values)) + "\n")
            files[b].flush()
            if (k + 1) % 20 == 0:
                print(f"PROGRESS {100 * (k + 1) / len(tasks):.0f}% | {k + 1} of {len(tasks)} fits", flush=True)
    for f in files.values():
        f.close()
    print("done", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
