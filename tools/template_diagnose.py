"""
template_diagnose.py -- which part of the mixing templates' error dominates at a point (2026-10-18): the chi^2 of
the model with templates against exact runs, moving one mixing parameter at a time to +-3 sigma (Table II's for
projections, NuFIT 6.1's for current data), with the spectral indices at the central nodes and away from them, and
moving all of them together. One parameter's error at the central nodes is its interpolation; growing away from them,
the index transfer; all together beyond the sum of the single ones, the cross terms.

  python tools/template_diagnose.py <config.json> --point '{"m_Zd_eV": 1e-30, "g_D": 3.3e-30}' [--out result.json]
"""
import argparse
import copy
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--point", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.pop("NULRI_FLUX_CACHE", None)
    from analysis import config as config_mod
    from analysis.run import asimov_centre, asimov_only
    from make_campaign_configs import MIX_KIND, TABLE2
    from template_validation import _ranges
    raw = json.load(open(a.config))
    dataset = os.path.basename(os.path.dirname(os.path.abspath(a.config)))
    exact_raw = copy.deepcopy(raw)
    for spec in exact_raw["parameters"].values():
        spec.pop("template", None)
    ct = config_mod.from_dict(raw, os.path.abspath(a.config))
    ce = config_mod.from_dict(exact_raw, os.path.abspath(a.config))
    centre = asimov_centre(ct) if asimov_only(ct) else None
    ot, oe = ct.objective(data_centre=centre), ce.objective(data_centre=centre)
    rg = _ranges(MIX_KIND[dataset])
    base = {p.name: p.start for p in ct.parameters if p.name not in TABLE2}
    base.update(json.loads(a.point))
    mt = ot.model
    central = {}
    if mt.p_nodes is not None:
        central["p_index"] = float(mt.p_nodes[len(mt.p_nodes) // 2])
    if mt.p1_nodes is not None:
        central["p_index_low"] = float(mt.p1_nodes[len(mt.p1_nodes) // 2])
    off = {}
    for n in ("p_index", "p_index_low"):
        if n in ct.free_names:
            lo, hi = ct.parameter(n).bounds
            off[n] = lo + 0.85 * (hi - lo)
    rows = []

    def err(v):
        return float(ot.chi2(v) - oe.chi2(v))

    for label, idx in (("central nodes", central), ("off the nodes", off)):
        for n, (c, _) in TABLE2.items():
            for sgn in (-1.0, 1.0):
                v = dict(base, **idx, **{n: c + sgn * rg[n]})
                rows.append(dict(index=label, moved=n, shift=sgn * rg[n], dchi2=err(v)))
                print(f"{label:14s} {n:9s} {sgn * rg[n]:+.4g}: chi2(templates) - chi2(exact) = {rows[-1]['dchi2']:+.4f}",
                      flush=True)
        for sgn in (-1.0, 1.0):
            v = dict(base, **idx, **{n: c + sgn * rg[n] for n, (c, _) in TABLE2.items()})
            rows.append(dict(index=label, moved="all", shift=sgn, dchi2=err(v)))
            print(f"{label:14s} all at {'+' if sgn > 0 else '-'}3 sigma: {rows[-1]['dchi2']:+.4f}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(config=a.config, point=json.loads(a.point), rows=rows), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
