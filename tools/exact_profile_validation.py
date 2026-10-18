"""
exact_profile_validation.py -- the exact profile over the mixing and the lightest mass (analysis/exact_profile.py)
against brute-force fits with the solver at every value, at points of a campaign plane (2026-10-18).

For a campaign configuration (its mixing templates dropped, m_lightest freed: exact_profile.exact_config) and points of
its plane: the exact profile (profile_global: with --ml-scan the lightest mass's scan first, with --octants both
octants of theta23), then, with --bruteforce, L-BFGS-B over every parameter with the solver at every value, started at
the profile's solution, at the priors' centre and at --bf-ml-starts lightest masses. Prints per point the chi^2 of
each, the certificate, the starts, the solver runs and the times; writes them as JSON.

  python tools/exact_profile_validation.py <config.json> --points '[{"m_Zd_eV": 1e-13, "g_D_eps_trace": 1e-40}]'
         [--ml-scan] [--octants] [--bruteforce] [--bf-ml-starts 1e-3 0.01] [--out results.json]
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--points", required=True, help="JSON list of {plane x: value, plane y: value}")
    ap.add_argument("--ml-scan", action="store_true", help="the lightest mass's scan (ML_SCAN) before the profile")
    ap.add_argument("--octants", action="store_true", help="also from the mirrored octant of theta23")
    ap.add_argument("--bruteforce", action="store_true")
    ap.add_argument("--bf-evals", type=int, default=80, help="brute force: L-BFGS-B iterations (central gradients)")
    ap.add_argument("--bf-ml-starts", type=float, nargs="*", default=[])
    ap.add_argument("--bf-profile-only", action="store_true",
                    help="brute force from the profile's solution (and --bf-ml-starts) only, not the priors' centre")
    ap.add_argument("--settings", default="{}", help="JSON ExactSettings overrides")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.pop("NULRI_FLUX_CACHE", None)
    from analysis import config as config_mod
    from analysis.exact_profile import ML_SCAN, ExactProfiler, ExactSettings, exact_config
    from analysis.fitting import FitSettings
    from analysis.run import asimov_centre, asimov_only
    raw = exact_config(json.load(open(a.config)))
    cfg = config_mod.from_dict(raw, os.path.abspath(a.config))
    t0 = time.perf_counter()
    centre = asimov_centre(cfg) if asimov_only(cfg) else None
    st = dict(json.loads(a.settings))
    if a.ml_scan:
        st["ml_scan"] = list(ML_SCAN)
    if a.octants:
        st["octants"] = True
    prof = ExactProfiler(cfg, data_centre=centre, settings=ExactSettings.from_dict(st))
    print(f"{raw['name']}: objective built in {time.perf_counter() - t0:.1f} s; eta = {prof.eta}", flush=True)
    rows = []
    for pt in json.loads(a.points):
        start = {p.name: p.start for p in cfg.parameters if p.name not in pt}
        r = prof.profile_global(pt, start, log=lambda s: print(s, flush=True))
        print(f"{raw['name']} {pt}: profile chi2 = {r.chi2:.6f}, {r.message}; {r.solver_runs} solver runs, "
              f"{r.seconds:.0f} s, {r.nfev} chi2 evaluations", flush=True)
        for x in r.starts:
            if "start" not in x:                     # the other octant's probe (profile_global)
                print(f"    octant probe {x.get('octant_probe')}: chi2 {x['chi2']:.6f}", flush=True)
                continue
            print(f"    start {x['start']}: chi2 {x['chi2']:.6f}, {'certified' if x['certified'] else 'NOT certified'}, "
                  f"{x['centres']} centres, {x['solver_runs']} runs", flush=True)
        row = dict(config=raw["name"], point=pt, **r.as_dict())
        if a.bruteforce:
            t = time.perf_counter()
            n0 = prof.exact_model.n_solver_calls
            bstarts = [r.values] + ([] if a.bf_profile_only else [start]) \
                + [dict(start, m_lightest_eV=float(m)) for m in a.bf_ml_starts]
            nfree = len([p for p in cfg.parameters if p.name not in pt])
            bf = prof.bruteforce(pt, bstarts, settings=FitSettings(**{**vars(prof.fit_settings),
                                                                      "max_evals": a.bf_evals * (2 * nfree + 1)}))
            row["bruteforce"] = dict(chi2=bf.chi2, values=bf.values, nfev=bf.nfev, trace=bf.trace,
                                     solver_runs=prof.exact_model.n_solver_calls - n0, seconds=time.perf_counter() - t)
            print(f"{raw['name']} {pt}: brute force chi2 = {bf.chi2:.6f} (per start: {bf.trace}), {bf.nfev} "
                  f"evaluations, {row['bruteforce']['solver_runs']} solver runs, {row['bruteforce']['seconds']:.0f} s; "
                  f"profile - brute force = {r.chi2 - bf.chi2:+.2e}", flush=True)
        rows.append(row)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
