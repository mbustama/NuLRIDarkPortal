"""
split_planes.py -- the final ultralight run's planes cut into blocks of mass columns (2026-10-25): at 30-45 min per fit
(every process on) one plane on one node takes days, so each plane's raster columns are split into blocks of --cols
columns, each its own configuration (the same grid points: x_range from the block's first to last grid mass, nx its
column count; output in its own folder), run as the campaign runs planes (slurm/campaign_array.sh, one block per task).
The pack list puts the fixed-composition planes first, so that their solver runs fill the flux cache
(NULRI_FLUX_CACHE) before their fitted-composition twins (_fx) need them.

  python tools/split_planes.py split <out_dir> <packs.txt> --cols 4 <config> [<config> ...]
  python tools/split_planes.py merge <config> [<root>]       the blocks' columns as one plane, the standard
      model's fit as each column's reference where lower (analysis.exclusion.field reads chi2_sm)
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


def _sm_chi2(root, dset, name):
    """The standard-model fit's chi^2 for plane `name` of data set `dset`: the lepton-number plane of the same data set
    and composition (its range reaches the no-effect limit, g = 1e-32; its global fit sits there), from every block's
    fit.json -- which agree; the dark-portal planes share its data, likelihood and standard-model flux."""
    ref = "le_lmu_ul_fx" if name.endswith("_fx") else "le_lmu_ul"
    vals = []
    for f in sorted(glob.glob(os.path.join(root, dset, ref, "b*", "data", "fit.json"))):
        r = json.load(open(f))
        g = r["values"].get("g_D")
        if g is not None and g < 1e-30:
            vals.append(float(r["chi2"]))
    if not vals:
        raise ValueError(f"no standard-model fit for {dset}/{name} under {root}/{dset}/{ref}")
    if max(vals) - min(vals) > 1e-6:
        raise ValueError(f"{dset}/{ref}: the blocks' standard-model fits differ ({min(vals)}, {max(vals)})")
    return min(vals)


def merge(config, root):
    """The blocks of plane `config` (its original configuration) under `root` (output/analysis/campaign_final) as one
    plane summary in the usual layout, <root>/<set>/<name>/data/plane_<x>_<y>.json (+ .jsonl, the blocks' records with
    their indices in the whole grid): the grids and chi^2 joined, chi2_sm the standard-model fit's chi^2 (each column read
    from it where it lies below the column's own minimum: analysis.exclusion.field), the columns' intervals read so."""
    sys.path[:0] = [os.path.join(REPO, "src"), REPO]
    from analysis import config as config_mod, intervals
    raw = json.load(open(config))
    cfg = config_mod.from_dict(raw, config)
    pl = raw["planes"][0]
    xs = _grid(*pl["x_range"], pl["nx"])
    dset = os.path.basename(os.path.dirname(config))
    name = os.path.basename(config)[:-5]
    here = os.path.join(root, dset, name)
    blocks = sorted(glob.glob(os.path.join(here, "b*", "data", "plane_*.json")),
                    key=lambda f: int(f.split(os.sep)[-3][1:]))
    nblk = len(range(0, len(xs), 4))
    if len(blocks) != nblk:
        raise ValueError(f"{here}: {len(blocks)} block summaries, {nblk} expected")
    summ = [json.load(open(f)) for f in blocks]
    y = np.array(summ[0]["grids"][1])
    x = np.concatenate([np.array(b["grids"][0]) for b in summ])
    if len(x) != len(xs) or not np.allclose(x, xs, rtol=1e-12, atol=0.0):
        raise ValueError(f"{here}: the blocks' masses are not the plane's")
    if any(not np.allclose(b["grids"][1], y, rtol=1e-12, atol=0.0) for b in summ):
        raise ValueError(f"{here}: the blocks' couplings differ")
    chi = np.array([[np.nan if v is None else v for v in row] for b in summ for row in b["chi2"]], dtype=float)
    sm = _sm_chi2(root, dset, name)
    ref = float(np.nanmin(np.append(chi.ravel(), sm)))
    thr = cfg.thresholds["1d"]
    log_y = cfg.parameter(pl["y"]).space == "log10"
    # the columns read with the no-effect limit in each column's range: one more coupling below the grid at the
    # standard-model chi^2 (the intervals may start there; the grid stored is the plane's own)
    y0 = y[0] / 10.0 if log_y else y[0] - (y[1] - y[0])
    chi_ext = np.hstack([np.full((len(x), 1), sm), chi])
    rs = intervals.summarize_raster(x, np.concatenate([[y0], y]), chi_ext, thr, ref, log_y)
    d = chi - ref
    dcol = rs["dchi2"][:, 1:]
    out = os.path.join(here, "data")
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"plane_{pl['x']}_{pl['y']}")
    summary = dict(kind="plane", names=[pl["x"], pl["y"]], grids=[x.tolist(), y.tolist()], log=summ[0]["log"],
                   chi2=np.where(np.isfinite(chi), chi, None).tolist(),
                   dchi2=np.where(np.isfinite(dcol), dcol, None).tolist(),
                   dchi2_global=np.where(np.isfinite(d), d, None).tolist(),
                   chi2_min=ref, chi2_min_from="standard-model fit" if ref == sm else "plane", chi2_sm=sm,
                   best=summ[0]["best"], title=f"{cfg.name} (data, merged)", complete=bool(np.isfinite(chi).all()),
                   raster=pl["raster"], thresholds=thr, columns=rs["columns"], elevated=rs["elevated"],
                   merged_from=[os.path.relpath(f, root) for f in blocks])
    json.dump(summary, open(stem + ".json", "w"))
    off = np.concatenate([[0], np.cumsum([len(b["grids"][0]) for b in summ])])
    with open(stem + ".jsonl", "w") as fh:
        for k, f in enumerate(blocks):
            for line in open(f[:-5] + ".jsonl"):
                r = json.loads(line)
                if "index" in r and isinstance(r["index"], list) and r["index"]:
                    r["index"] = [int(r["index"][0]) + int(off[k])] + list(r["index"][1:])
                fh.write(json.dumps(r) + "\n")
    print(f"{dset}/{name}: {len(blocks)} blocks, {len(x)} masses; standard model {sm:.4f}, reference {ref:.4f} "
          f"({summary['chi2_min_from']}); {len(rs['elevated'])} columns elevated -> {stem}.json")
    return 0


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
    a.add_argument("root", nargs="?", default=os.path.join(REPO, "output", "analysis", "campaign_final"))
    args = ap.parse_args(argv)
    if args.cmd == "split":
        return split(args.out_dir, args.packs, args.cols, args.configs)
    return merge(args.config, args.root)


if __name__ == "__main__":
    sys.exit(main())
