"""
template_validation.py -- the mixing templates against exact solver runs at the campaign's resolution (2026-10-18;
docs/dev/MIXING_VARIATION.md).

For a campaign configuration with templates and points of its plane: at random values of the six mixing parameters
within the data kind's ranges (current data: NuFIT 6.1's 3 sigma; projections: Table II's 3 sigma) and random spectral
indices within their bounds, the chi^2 and the fluxes of the model with templates against the same configuration
without them (every mixing value then a solver run). Prints, per point, the largest |chi^2 difference| and the
largest flux error relative to each bin's largest flavor flux, and the time of the point's runs and of one template
run; writes them as JSON.

  python tools/template_validation.py <config.json> --points '[{"m_Zd_eV": 1e-13, "g_D_eps_trace": 1e-40}]'
                                      [--samples 6] [--seed 1] [--out results.json]
"""
import argparse
import copy
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402

THREE_SIGMA = {"now": dict(s12sq=0.036, s13sq=0.0017, s23sq=0.12, d_cp_deg=180.0, dm21_sq=0.6e-5, dm31_sq=0.07e-3)}


def _ranges(kind):
    from make_campaign_configs import TABLE2
    if kind == "now":
        return THREE_SIGMA["now"]
    return {n: 3.0 * w for n, (_, w) in TABLE2.items()}


def _flux_error(a, b):
    sc = np.max(np.abs(b), axis=-1, keepdims=True)
    return float(np.max(np.abs(a - b) / np.maximum(sc, 1e-300)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--points", required=True, help="JSON list of {plane x: value, plane y: value}")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.pop("NULRI_FLUX_CACHE", None)
    from analysis import config as config_mod
    from analysis.run import asimov_centre, asimov_only
    from make_campaign_configs import MIX_KIND, TABLE2
    raw = json.load(open(a.config))
    dataset = os.path.basename(os.path.dirname(os.path.abspath(a.config)))
    kind = MIX_KIND[dataset]
    exact_raw = copy.deepcopy(raw)
    for spec in exact_raw["parameters"].values():
        spec.pop("template", None)
    ct = config_mod.from_dict(raw, os.path.abspath(a.config))
    ce = config_mod.from_dict(exact_raw, os.path.abspath(a.config))
    centre = asimov_centre(ct) if asimov_only(ct) else None
    ot = ct.objective(data_centre=centre)
    oe = ce.objective(data_centre=centre)
    mt, me = ot.model, oe.model
    rng = np.random.default_rng(a.seed)
    rg = _ranges(kind)
    rows = []
    for pt in json.loads(a.points):
        base = {p.name: p.start for p in ct.parameters if p.name not in TABLE2}
        base.update(pt)
        t0 = time.perf_counter()
        ot.chi2(base)
        t_point = time.perf_counter() - t0
        n_tmpl = sum(len(t.points) for t in mt.templates.values())
        dchi2, ferr = 0.0, 0.0
        for _ in range(a.samples):
            v = dict(base)
            for n, (c, _) in TABLE2.items():
                v[n] = c + rg[n] * rng.uniform(-1.0, 1.0)
            for n in ("p_index", "p_index_low"):
                p = ct.parameter(n) if n in ct.free_names else None
                if p is not None:
                    v[n] = float(rng.uniform(p.bounds[0], p.bounds[1]))
            dchi2 = max(dchi2, abs(ot.chi2(v) - oe.chi2(v)))
            if mt.sky:
                for lat in [-89.99] + list(mt.site_lats):
                    gt, ge = mt.site_band_fluxes(v, lat), me.site_band_fluxes(v, lat)
                    ferr = max(ferr, _flux_error(gt[1], ge[1]), _flux_error(gt[2], ge[2]))
            else:
                ferr = max(ferr, _flux_error(mt.flux(v)[1], me.flux(v)[1]))
        row = dict(config=raw["name"], point=pt, samples=a.samples, max_abs_dchi2=dchi2, max_flux_error=ferr,
                   seconds_point_with_templates=t_point, template_runs_per_piece=n_tmpl)
        rows.append(row)
        print(f"{raw['name']} {pt}: |chi2(templates) - chi2(exact)| <= {dchi2:.3g}, flux error <= {ferr:.3g} "
              f"over {a.samples} values; the point with its {n_tmpl} template runs per piece took {t_point:.0f} s",
              flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
