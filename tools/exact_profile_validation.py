"""
exact_profile_validation.py -- the exact profile over the mixing and the lightest mass (analysis/exact_profile.py)
against brute-force fits with the solver at every value, at points of a campaign plane (2026-10-18).

For a campaign configuration (its mixing templates dropped, m_lightest freed: exact_profile.exact_config) and points of
its plane: the exact profile from the priors' centre (and from the other octant of theta23 under NuFIT, and from extra
lightest masses with --ml-starts), then, with --bruteforce, L-BFGS-B over every parameter with the solver at every
value, started at the profile's solution and at the priors' centre. Prints per point the chi^2 of each, the
certificate, the centres, the solver runs and the times; writes them as JSON.

  python tools/exact_profile_validation.py <config.json> --points '[{"m_Zd_eV": 1e-13, "g_D_eps_trace": 1e-40}]'
                                           [--bruteforce] [--ml-starts 1e-3 0.01] [--out results.json]
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

UPPER_OCTANT = 0.55          # NuFIT 6.1's second minimum in sin^2 theta23 (normal ordering), 0.76 above the first


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--points", required=True, help="JSON list of {plane x: value, plane y: value}")
    ap.add_argument("--bruteforce", action="store_true")
    ap.add_argument("--bf-evals", type=int, default=1500)
    ap.add_argument("--ml-starts", type=float, nargs="*", default=[])
    ap.add_argument("--settings", default="{}", help="JSON ExactSettings overrides")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.pop("NULRI_FLUX_CACHE", None)
    from analysis import config as config_mod
    from analysis.exact_profile import ExactProfiler, ExactSettings, exact_config
    from analysis.fitting import FitSettings
    from analysis.nufit import NuFitPairwisePrior
    from analysis.run import asimov_centre, asimov_only
    raw = exact_config(json.load(open(a.config)))
    cfg = config_mod.from_dict(raw, os.path.abspath(a.config))
    t0 = time.perf_counter()
    centre = asimov_centre(cfg) if asimov_only(cfg) else None
    prof = ExactProfiler(cfg, data_centre=centre, settings=ExactSettings.from_dict(json.loads(a.settings)))
    print(f"{raw['name']}: objective built in {time.perf_counter() - t0:.1f} s; eta = {prof.eta}", flush=True)
    nufit = any(isinstance(p, NuFitPairwisePrior) for p in cfg.priors)
    rows = []
    for pt in json.loads(a.points):
        start = {p.name: p.start for p in cfg.parameters if p.name not in pt}
        starts = [("prior centre", start)]
        if nufit:
            starts.append(("upper octant", dict(start, s23sq=UPPER_OCTANT)))
        for m in a.ml_starts:
            starts.append((f"m_lightest {m:g}", dict(start, m_lightest_eV=float(m))))
        row = dict(config=raw["name"], point=pt, starts=[])
        best = None
        for label, st in starts:
            r = prof.profile(pt, st, log=lambda s: print(s, flush=True))
            print(f"{raw['name']} {pt} from {label}: chi2 = {r.chi2:.6f}, {r.message}; {len(r.centres)} centres, "
                  f"{r.solver_runs} solver runs, {r.seconds:.0f} s, {r.nfev} chi2 evaluations", flush=True)
            row["starts"].append(dict(label=label, **r.as_dict()))
            if best is None or r.chi2 < best.chi2:
                best = r
        row["profile_chi2"] = best.chi2
        row["profile_values"] = best.values
        if a.bruteforce:
            t = time.perf_counter()
            n0 = prof.exact_model.n_solver_calls
            bf = prof.bruteforce(pt, [best.values, start],
                                 settings=FitSettings(**{**vars(prof.fit_settings), "max_evals": a.bf_evals}))
            row["bruteforce"] = dict(chi2=bf.chi2, values=bf.values, nfev=bf.nfev, trace=bf.trace,
                                     solver_runs=prof.exact_model.n_solver_calls - n0, seconds=time.perf_counter() - t)
            print(f"{raw['name']} {pt}: brute force chi2 = {bf.chi2:.6f} (from the profile's solution and the priors' "
                  f"centre: {bf.trace}), {bf.nfev} evaluations, {row['bruteforce']['solver_runs']} solver runs, "
                  f"{row['bruteforce']['seconds']:.0f} s; profile - brute force = {best.chi2 - bf.chi2:+.2e}",
                  flush=True)
        rows.append(row)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
