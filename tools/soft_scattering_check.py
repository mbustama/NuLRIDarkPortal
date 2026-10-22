"""
soft_scattering_check.py -- how much the flavor equalization by relic scattering below 1 keV weighs in the fits
(2026-10-22, the relic-scattering question): the dark portal at eps' = 0 with dp_si's full transport forced
(dark_portal.cosmo_transport = 'si') at a sub-keV mass, fitted (the plane's own fit, mass and coupling held) at
couplings from where the relics' scatterings are too rare to matter to where they equalize the flavors (rate >> H)
but cost too little energy to dent the spectrum, and above. The chi^2 against the smallest coupling's tells whether
the equalization alone (a shift of ~0.01 in the flavor fractions at Earth for pion-decay sources) moves a fit.
Prints a table and writes output/analysis/soft_scattering_check.json.
  python tools/soft_scattering_check.py [--mass 100] [--log-g -7 -5 -4.5 -4 -3.5 -3] [--data-root <repo with outputs>]
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
OUT = os.path.join(REPO, "output", "analysis", "soft_scattering_check.json")
DATASETS = ("now_ic26", "now_events", "proj_future")
TEXTURES = ("e", "mu", "tau")


def _config(ds, t, data_root):
    """(raw config with the full transport forced, its path, the Asimov data centre or None)."""
    from analysis.config import from_dict
    from analysis.run import asimov_only
    cfgp = os.path.join(REPO, "analysis", "configs", "campaign_ml", ds, f"dp_{t}_ul.json")
    raw = copy.deepcopy(json.load(open(cfgp)))
    raw["solver"]["dark_portal"]["cosmo_transport"] = "si"
    cfg = from_dict(raw, cfgp)
    dc = None
    if asimov_only(cfg):
        out_dir = os.path.relpath(cfg.output_path("asimov"), REPO)
        runs = json.load(open(os.path.join(data_root, out_dir, "manifest.json")))["runs"]
        dc = tuple(next(r["data_centre"] for r in reversed(runs) if r.get("data_centre")))
    return raw, cfgp, dc


def one(task):
    ds, t, M, lg, data_root = task
    os.environ.pop("NULRI_FLUX_CACHE", None)
    from analysis import scans
    from analysis.config import from_dict
    raw, cfgp, dc = _config(ds, t, data_root)
    scans._init_worker(raw, cfgp, dc)
    cfg = scans._W["cfg"]
    pl = raw["planes"][0]
    held = {pl["x"]: M, pl["y"]: 10.0 ** lg}
    start = {p.name: p.start for p in cfg.parameters if p.name not in held}
    settings = {**asdict(cfg.fit), "n_random": 0}
    try:
        r = scans._fit_held(held, [start], settings)
        return ds, t, lg, float(r.chi2), None
    except Exception as e:
        return ds, t, lg, None, repr(e)[:300]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mass", type=float, default=100.0)
    ap.add_argument("--log-g", type=float, nargs="+", default=[-7.0, -5.0, -4.5, -4.0, -3.5, -3.0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--data-root", default=REPO, help="the repository whose output/ holds the planes' manifests")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    tasks = [(ds, t, a.mass, lg, a.data_root) for ds in a.datasets for t in TEXTURES for lg in a.log_g]
    with get_context("spawn").Pool(min(a.workers, len(tasks)), maxtasksperchild=1) as pool:
        res = pool.map(one, tasks, chunksize=1)
    out = {}
    print(f"M = {a.mass:g} eV, full transport forced; chi^2 - chi^2(log g = {a.log_g[0]:g}), fitted")
    print(f"{'data set':12s} {'texture':7s} " + " ".join(f"{lg:>7g}" for lg in a.log_g))
    for ds in a.datasets:
        for t in TEXTURES:
            row = {lg: (c, e) for d, tt, lg, c, e in res if d == ds and tt == t}
            c0 = row[a.log_g[0]][0]
            out[f"{ds}/{t}"] = {f"{lg:g}": row[lg][0] for lg in a.log_g}
            cells = []
            for lg in a.log_g:
                c, e = row[lg]
                cells.append(f"{'ERR':>7s}" if c is None or c0 is None else f"{c - c0:7.2f}")
            print(f"{ds:12s} {t:7s} " + " ".join(cells))
            for lg in a.log_g:
                if row[lg][1]:
                    print(f"  {ds}/{t} log g {lg:g}: {row[lg][1]}")
    json.dump(dict(mass=a.mass, log_g=a.log_g, chi2=out), open(OUT, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
