"""
split_planes.py -- the final ultralight run's planes cut into blocks of mass columns (2026-10-25): at 30-45 min per fit
(every process on) one plane on one node takes days, so each plane's raster columns are split into blocks of --cols
columns, each its own configuration (the same grid points: x_range from the block's first to last grid mass, nx its
column count; output in its own folder), run as the campaign runs planes (slurm/campaign_array.sh, one block per task).
The pack list puts the fixed-composition planes first, so that their solver runs fill the flux cache
(NULRI_FLUX_CACHE) before their fitted-composition twins (_fx) need them.

  python tools/split_planes.py split <out_dir> <packs.txt> --cols 4 <config> [<config> ...]
  python tools/split_planes.py merge <config> <out_dir>      (later: the blocks' columns as one plane)
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _grid(lo, hi, n):
    return np.logspace(np.log10(lo), np.log10(hi), int(n))


def split(out_dir, packs, cols, configs):
    os.makedirs(out_dir, exist_ok=True)
    nominal, twins = [], []
    for path in configs:
        raw = json.load(open(path))
        pl = raw["planes"][0]
        if pl["raster"] != pl["x"]:
            raise ValueError(f"{path}: the raster is not the x axis")
        xs = _grid(*pl["x_range"], pl["nx"])
        dset = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)[:-5]
        for b, k in enumerate(range(0, len(xs), cols)):
            blk = xs[k:k + cols]
            c = copy.deepcopy(raw)
            p = c["planes"][0]
            p["x_range"] = [float(blk[0]), float(blk[-1])]
            p["nx"] = int(len(blk))
            # the block's grid masses are the plane's (to rounding: checked)
            sub = _grid(*p["x_range"], p["nx"])
            if not np.allclose(sub, blk, rtol=1e-12, atol=0.0):
                raise ValueError(f"{path} block {b}: grid differs")
            c["name"] = f"{raw['name']}__b{b}"
            c["output_dir"] = f"output/analysis/campaign_final/{dset}/{name}/b{b}"
            c["_comment"] = (f"Final ultralight run (2026-10-25): block {b} (columns {k}-{k + len(blk) - 1}) of "
                             f"{os.path.relpath(path, REPO)} (tools/split_planes.py).")
            d = os.path.join(out_dir, dset)
            os.makedirs(d, exist_ok=True)
            f = os.path.join(d, f"{name}__b{b}.json")
            json.dump(c, open(f, "w"), indent=1)
            (twins if name.endswith("_fx") else nominal).append(os.path.relpath(f, REPO))
    with open(packs, "w") as fh:
        for f in nominal + twins:
            fh.write(f + "\n")
    print(f"{len(nominal)} fixed-composition and {len(twins)} fitted-composition blocks -> {packs}")
    return 0


def merge(config, out_dir):
    raise NotImplementedError("merge: to be written before the final plots (the blocks' columns as one plane)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("split")
    a.add_argument("out_dir")
    a.add_argument("packs")
    a.add_argument("configs", nargs="+")
    a.add_argument("--cols", type=int, default=4)
    a = sp.add_parser("merge")
    a.add_argument("config")
    a.add_argument("out_dir")
    args = ap.parse_args(argv)
    if args.cmd == "split":
        return split(args.out_dir, args.packs, args.cols, args.configs)
    return merge(args.config, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
