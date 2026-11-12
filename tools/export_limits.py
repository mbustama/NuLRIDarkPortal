"""
export_limits.py -- the 95% C.L. excluded couplings of every plane under a results tree, as JSON (2026-11-12, his
request: "Create limit curves and save them on JSON files from everything you have already. For the partial cases,
create what you have").

A plane is every directory holding data/ or asimov/ with plane_<x>_<y>.json (a finished run's summary) or only
plane_<x>_<y>.jsonl (the records of a run still going or stopped); blocks of one plane (b<k>/, m<mass>/ directories)
are joined: each (mass, coupling) keeps its smallest chi^2. The no-effect chi^2 (chi2_sm) is the smallest chi^2 at the
plane's smallest coupling over all masses (the heavy columns, where that coupling has no effect); the columns are read
by analysis.exclusion.field, as the figures read them (per-mass Delta chi^2 >= 3.84, PCHIP in log10 g, specks out).

Written per plane, under --out with the plane's relative path:
  data/plane_<x>_<y>.json  the joined summary (the format analysis.exclusion reads; chi2_sm, merged_from)
  limits.json              per mass column: log10 m, the excluded intervals of log10 g, the lowest excluded coupling,
                           whether every point of the column was computed, and the not-modelled boundary of the
                           plane's configuration where analysis.exclusion has one as a line (log10 g >= log10 m + c);
                           plus the blocks read (finished or partial, with their point counts)
  python tools/export_limits.py <results root> --out <dir>
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
from analysis import exclusion as ex  # noqa: E402

BLOCK = re.compile(r"^(b\d+|m[0-9eE.+-]+)$")


def planes(root):
    """{plane key (relative path): [data directories of its blocks]}"""
    out = defaultdict(list)
    for d in sorted(set(os.path.dirname(p) for p in glob.glob(f"{root}/**/plane_*.json*", recursive=True))):
        if os.path.basename(d) not in ("data", "asimov"):
            continue
        parent = os.path.dirname(d)
        key = os.path.dirname(parent) if BLOCK.match(os.path.basename(parent)) else parent
        out[os.path.relpath(key, root)].append(d)
    return out


def block_points(d):
    """(names, {(m, g): chi2}, finished, config) of one block's data directory."""
    pts, names, finished, cfg = {}, None, False, None
    summ = [f for f in glob.glob(f"{d}/plane_*.json")]
    if summ:
        s = json.load(open(summ[0]))
        names, finished = s["names"], True
        for i, m in enumerate(s["grids"][0]):
            for j, g in enumerate(s["grids"][1]):
                v = s["chi2"][i][j]
                if v is not None and np.isfinite(v):
                    pts[(float(m), float(g))] = float(v)
    for f in glob.glob(f"{d}/plane_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("chi2") is None or not np.isfinite(r["chi2"]) or "held" not in r:
                continue
            k = tuple(float(v) for v in r["held"].values())
            if names is None:
                names = list(r["held"])
            if k not in pts or r["chi2"] < pts[k]:
                pts[k] = float(r["chi2"])
    if os.path.exists(f"{d}/manifest.json"):
        cfg = json.load(open(f"{d}/manifest.json")).get("config")
    return names, pts, finished, cfg


def no_effect(dirs, yn, gmin):
    vals = set()
    for d in dirs:
        f = f"{d}/fit.json"
        if os.path.exists(f):
            fj = json.load(open(f))
            if fj.get("values", {}).get(yn, np.inf) <= gmin * (1 + 1e-9) and fj.get("chi2") is not None:
                vals.add(round(float(fj["chi2"]), 6))
    return min(vals) if vals else None, sorted(vals)


def boundary(cfg_raw):
    """(slope, intercept, kind) of the not-modelled line, or None (scalar relic-field regions are not a line)."""
    if not cfg_raw:
        return None
    try:
        from analysis.config import from_dict
        cfg = from_dict(cfg_raw, os.path.join(REPO, "analysis", "configs", "manifest.json"))
        b = ex.not_allowed_boundary(cfg)
        c = ex.longitudinal_corner(cfg)
        out = {}
        if b is not None:
            out["not_perturbative"] = {"slope": float(b[0]), "intercept": float(b[1])}
        if c is not None:
            out["longitudinal_left_out"] = {"slope": 1.0, "intercept": float(c)}
        if ex.not_modelled_measure(cfg) is not None:
            out["relic_field_region"] = "analysis.exclusion.not_modelled_mask(cfg) (not a line)"
        return out or None
    except Exception as e:                                  # an old configuration the current code does not read
        return {"error": f"{type(e).__name__}: {e}"}


def export(key, dirs, out_root):
    names, pts, blocks, cfg = None, {}, [], None
    for d in dirs:
        n, p, fin, c = block_points(d)
        names = names or n
        cfg = cfg or c
        blocks.append({"dir": d, "finished": fin, "points": len(p)})
        for k, v in p.items():
            if k not in pts or v < pts[k]:
                pts[k] = v
    if not pts:
        return None
    xs = sorted({k[0] for k in pts})
    ys = sorted({k[1] for k in pts})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: j for j, v in enumerate(ys)}
    chi = np.full((len(xs), len(ys)), np.nan)
    for (m, g), v in pts.items():
        chi[xi[m], yi[g]] = v
    # the no-effect chi^2: the smallest chi^2 at the plane's smallest coupling over all masses (heavy columns there
    # have no effect; a block's fit.json is not it where light columns act already at the smallest coupling, as the
    # dark portal's decay does: its fit.json at 1e-13 eV, g = 1e-8 is 164.9 against 9.88 without an effect)
    sm = float(np.nanmin(chi[:, 0])) if np.isfinite(chi[:, 0]).any() else None
    _, sms = no_effect(dirs, names[1], ys[0])
    odir = os.path.join(out_root, key, "data")
    os.makedirs(odir, exist_ok=True)
    path = os.path.join(odir, f"plane_{names[0]}_{names[1]}.json")
    json.dump(dict(kind="plane", names=names, grids=[xs, ys], log=[True, True],
                   chi2=np.where(np.isfinite(chi), chi, None).tolist(), chi2_sm=sm, merged_from=len(dirs),
                   title=key), open(path, "w"))
    lx, G, F, _, _ = ex.field(path)
    full = {round(float(np.log10(m)), 6): bool(np.isfinite(chi[i]).all()) for i, m in enumerate(xs)}
    cols = []
    for k, l in enumerate(lx):
        f = F[:, k]
        if np.ma.is_masked(f) and f.mask.all():
            cols.append({"log10_m": float(l), "read": False})
            continue
        exc = np.asarray(f.filled(-1.0)) > 0
        runs, start = [], None
        for j, e in enumerate(exc):
            if e and start is None:
                start = j
            if (not e or j == len(exc) - 1) and start is not None:
                end = j if e else j - 1
                runs.append([round(float(G[start]), 4), round(float(G[end]), 4)])
                start = None
        cols.append({"log10_m": round(float(l), 6), "read": True, "excluded_log10_g": runs,
                     "lowest_excluded_log10_g": runs[0][0] if runs else None,
                     # excluded down to the smallest coupling computed: the limit lies below the grid
                     "limit_below_grid": bool(runs) and runs[0][0] <= round(float(np.log10(ys[0])), 4) + 1e-6,
                     "all_couplings_computed": full.get(round(float(l), 6))})
    res = {"plane": key, "x": names[0], "y": names[1], "confidence": "95% C.L., per-mass Delta chi^2 >= 3.84",
           "reading": "analysis.exclusion.field (PCHIP in log10 g, specks out, upper limits only)",
           "chi2_no_effect": sm, "chi2_fit_json_at_smallest_coupling": sms,
           "complete": all(b["finished"] for b in blocks), "blocks": blocks,
           "not_modelled": boundary(cfg), "columns": cols}
    json.dump(res, open(os.path.join(out_root, key, "limits.json"), "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    found = planes(a.root)
    index = []
    for key, dirs in found.items():
        try:
            r = export(key, dirs, a.out)
        except Exception as e:
            print(f"FAILED {key}: {type(e).__name__}: {e}", flush=True)
            index.append({"plane": key, "error": f"{type(e).__name__}: {e}"})
            continue
        if r is None:
            continue
        nb = len(r["blocks"])
        nf = sum(b["finished"] for b in r["blocks"])
        nx = sum(1 for c in r["columns"] if c.get("excluded_log10_g"))
        index.append({"plane": key, "complete": r["complete"], "blocks_finished": nf, "blocks": nb,
                      "columns": len(r["columns"]), "columns_with_exclusion": nx})
        print(f"{key}: {nf}/{nb} blocks finished, {nx}/{len(r['columns'])} columns exclude", flush=True)
    json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
