"""
exact_campaign_cost.py -- the exact-profile campaign's cost from a pilot (2026-10-18; docs/dev/EXACT_PROFILE.md).

Reads the pilot's plane checkpoints (output/analysis/pilot_exact/<set>/<plane>/{data,asimov}/plane_*.jsonl: per point
the fit's seconds on one core, its solver runs, and whether it inherited its chain's certificate), the mean cost of an
inherited and of a fully profiled point per data set and plane kind (potential balanced, potential split, ultralight,
MeV), and extrapolates to the campaign (analysis/configs/campaign_exact/<set>/campaign.json) with the fraction of
inherited points either the pilot's own or, with --negligible, the fraction of points whose chi^2 the earlier
campaign found within 1e-4 of their column's smallest coupling (tools: np_fraction in the validation notes).

  python tools/exact_campaign_cost.py [--pilot output/analysis/pilot_exact] [--cores 176]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# the earlier campaign's fraction of points within 1e-4 in chi^2 of their column's smallest coupling (fixed mixing;
# nulri_bpl, 2026-10-18), per data set and kind; proj_2050 taken as now_ic26's
NEGLIGIBLE = {"now_ic26": dict(bal=0.36, split=0.26, ul=0.47, mev=0.57),
              "now_events": dict(bal=0.35, split=0.20, ul=0.44, mev=0.76),
              "proj_future": dict(bal=0.40, split=0.09, ul=0.46, mev=0.78)}
NEGLIGIBLE["proj_2050"] = NEGLIGIBLE["now_ic26"]


def kind_of(name: str) -> str:
    if name.endswith("_mev") or "_mev_" in name:
        return "mev"
    if name.endswith("_ul") or "_ul_" in name:
        return "ul"
    return "bal" if "_bal_" in name else "split"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", default=os.path.join(REPO, "output", "analysis", "pilot_exact"))
    ap.add_argument("--cores", type=int, default=176)
    ap.add_argument("--negligible", action="store_true", help="the earlier campaign's negligible fractions")
    a = ap.parse_args(argv)
    stats = defaultdict(lambda: dict(inh=[], full=[], runs=[]))
    for f in sorted(glob.glob(os.path.join(a.pilot, "*", "*", "*", "plane_*.jsonl"))):
        ds, name = f.split(os.sep)[-4], f.split(os.sep)[-3]
        best = {}
        for line in open(f):
            r = json.loads(line) if line.strip() else {}
            if "chi2" in r and "index" in r:
                best.setdefault(tuple(r["index"]), []).append(r)
        st = stats[(ds, kind_of(name))]
        for recs in best.values():
            sec = sum(r.get("seconds", 0.0) for r in recs)              # a point's fits, repairs included
            (st["inh"] if recs[0].get("message", "").startswith("inherited") else st["full"]).append(sec)
            st["runs"].append(sum(r.get("solver_runs", 0) for r in recs))
    print(f"{'set':12s} {'kind':6s} {'points':>6s} {'inherited':>9s} {'s/inh':>7s} {'s/full':>7s} {'runs':>6s}")
    mean = {}
    for (ds, k), st in sorted(stats.items()):
        n_i, n_f = len(st["inh"]), len(st["full"])
        si = sum(st["inh"]) / n_i if n_i else 0.0
        sf = sum(st["full"]) / n_f if n_f else 0.0
        mean[(ds, k)] = (si, sf, n_i / max(n_i + n_f, 1))
        print(f"{ds:12s} {k:6s} {n_i + n_f:6d} {n_i / max(n_i + n_f, 1):9.2f} {si:7.1f} {sf:7.1f} "
              f"{sum(st['runs']) / max(n_i + n_f, 1):6.0f}")
    total = 0.0
    for ds in ("now_ic26", "now_events", "proj_2050", "proj_future"):
        path = os.path.join(REPO, "analysis", "configs", "campaign_exact", ds, "campaign.json")
        if not os.path.exists(path):
            continue
        camp = json.load(open(path))
        per = defaultdict(int)
        for p in camp["planes"]:
            per[kind_of(p["name"])] += p["points"]
        sub = 0.0
        for k, n in per.items():
            key = (ds, k) if (ds, k) in mean else next(((d, k) for d in ("now_ic26", "proj_2050", "proj_future",
                                                                       "now_events") if (d, k) in mean), None)
            if key is None:
                print(f"  {ds} {k}: no pilot", file=sys.stderr)
                continue
            si, sf, fi = mean[key]
            if a.negligible:
                fi = NEGLIGIBLE[ds][k]
            sub += n * (fi * si + (1.0 - fi) * sf)
        total += sub
        print(f"{ds}: {sub / 3600:.0f} core-hours")
    print(f"campaign: {total / 3600:.0f} core-hours = {total / 3600 / a.cores:.0f} h on {a.cores} cores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
