"""
merge_ml62.py -- folds the set-1 MeV refit (62-mass lightest-mass scan, chains of 8; decided 2026-11-25) into the
stored planes, so that tools/export_limits.py keeps, at each (mass, coupling), the lower chi^2 of the stored profile
and the refit's (his rule: "keep the lowest chi^2 among these and the stored one").

export_limits joins the blocks of a plane (b<k>/data directories beside or below its data/) point by point, keeping
the smallest chi^2 at equal (mass, coupling) floats. The refit's blocks build their grids from their own ranges
(np.logspace of the block's ends), which differ from the stored grid by a few ulp, so this tool copies each refit block
into the stored plane's directory as b<1000+k>/data with its masses and couplings snapped to the stored grid's values
(refusing anything farther than 1e-9 in log10).

  python tools/merge_ml62.py <refit plane dir> <stored plane dir>
     e.g. output/analysis/campaign_ml62/now_icecube/le_ltau_mev_fx output/analysis/campaign_ml/now_icecube/le_ltau_mev_fx
"""
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

TOL = 1e-9          # in log10
OFFSET = 1000       # the refit's blocks: b1000, b1001, ... (export_limits joins any b<k>)


def stored_grids(stored):
    """(names, x grid, y grid) of the stored plane: its summary, or the union of its blocks' summaries."""
    xs, ys, names = set(), set(), None
    for f in glob.glob(os.path.join(stored, "data", "plane_*.json")) + glob.glob(os.path.join(stored, "b*", "data",
                                                                                               "plane_*.json")):
        m = re.match(r"^b(\d+)$", os.path.basename(os.path.dirname(os.path.dirname(f))))
        if m and int(m.group(1)) >= OFFSET:
            continue                       # an earlier merge's own blocks
        s = json.load(open(f))
        names = names or s["names"]
        xs.update(map(float, s["grids"][0]))
        ys.update(map(float, s["grids"][1]))
    if names is None:
        raise FileNotFoundError(f"no stored plane summary under {stored}")
    return names, np.array(sorted(xs)), np.array(sorted(ys))


def snap(v, grid):
    i = int(np.argmin(np.abs(np.log10(grid) - np.log10(v))))
    if abs(np.log10(grid[i]) - np.log10(v)) > TOL:
        raise ValueError(f"{v!r} is not on the stored grid (nearest {grid[i]!r})")
    return float(grid[i])


def merge(refit, stored):
    names, gx, gy = stored_grids(stored)
    n = 0
    for d in sorted(glob.glob(os.path.join(refit, "b*", "data"))):
        k = int(os.path.basename(os.path.dirname(d))[1:])
        out = os.path.join(stored, f"b{OFFSET + k}", "data")
        os.makedirs(out, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*")):
            base = os.path.basename(f)
            if base.startswith("plane_") and base.endswith(".json"):
                s = json.load(open(f))
                assert s["names"] == names, (s["names"], names)
                s["grids"] = [[snap(v, gx) for v in s["grids"][0]], [snap(v, gy) for v in s["grids"][1]]]
                json.dump(s, open(os.path.join(out, base), "w"))
            elif base.startswith("plane_") and base.endswith(".jsonl"):
                with open(os.path.join(out, base), "w") as w:
                    for line in open(f):
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "held" in r:
                            r["held"] = {names[0]: snap(r["held"][names[0]], gx),
                                         names[1]: snap(r["held"][names[1]], gy)}
                        w.write(json.dumps(r) + "\n")
            elif base in ("fit.json", "fit.jsonl", "manifest.json"):
                shutil.copy(f, os.path.join(out, base))
        n += 1
    return n


if __name__ == "__main__":
    print(merge(sys.argv[1], sys.argv[2]), "blocks merged")
