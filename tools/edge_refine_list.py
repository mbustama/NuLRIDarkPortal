"""
edge_refine_list.py -- review item 8, second half (his choice 2026-11-11: "curve now, refine where needed"): the mass
columns of a plane where the 95% edge read by straight lines in log10 g and by the monotone cubic (analysis/exclusion.py
curve) differ by more than 2% in the coupling. There the coupling grid is too coarse for the reading: new couplings
between the two samples that bracket the edge are needed (new solver runs).

  python tools/edge_refine_list.py <plane_*.json> [...]     prints, per flagged column: log10 m, both edges, the bracket
"""
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from analysis import exclusion as ex  # noqa: E402

TOL = np.log10(1.02)        # 2% in the coupling


def edges(path, kind):
    old = ex.INTERP
    ex.INTERP = kind
    try:
        lx, G, F, yn, C = ex.field(path)
    finally:
        ex.INTERP = old
    out = {}
    for j, m in enumerate(lx):
        col = np.asarray(F[:, j].filled(-1.0))
        flips = np.flatnonzero(np.sign(col[1:]) != np.sign(col[:-1]))
        out[round(float(m), 6)] = 0.5 * (G[flips] + G[flips + 1])        # every crossing in the column
    return out


def main(paths):
    for path in paths:
        lin, cub = edges(path, "linear"), edges(path, "pchip")
        flagged = []
        for m in lin:
            a, b = lin[m], cub.get(m, np.array([]))
            if len(a) != len(b):
                flagged.append((m, a, b, "different number of crossings"))
                continue
            d = np.abs(a - b)
            if np.any(d > TOL):
                flagged.append((m, a, b, f"largest difference {100 * (10 ** d.max() - 1):.1f}%"))
        print(f"{path}: {len(flagged)} of {len(lin)} columns need couplings near their edge")
        for m, a, b, why in flagged:
            print(f"  log10 m = {m:+.3f}: straight {np.round(a, 3).tolist()}, cubic {np.round(b, 3).tolist()} ({why})")


if __name__ == "__main__":
    main(sys.argv[1:])
