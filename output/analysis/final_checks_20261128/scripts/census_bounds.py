"""census_bounds.py -- for every plane of today's data with the source composition fitted (the planes of the paper's
limit figures), at each mediator mass: the fitted parameters at the last allowed coupling below the limit (the point
that sets it), and which of them sit on an edge of their allowed range. Read from the point records on Tycho
(2026-11-27, to define the final run). Prints a table per family of planes.
  python3 census_bounds.py"""
import collections
import glob
import json
import math
import os

X = "/lustre/astro/mbustama/external"
BOUNDS = {"p_index": (2.0, 3.0), "p_index_low": (0.5, 2.5), "norm_astro": (0.01, 1000.0), "norm_gal": (0.0, 3.0),
          "f_e_S": (0.0, 1.0), "f_e_S_gal": (0.0, 1.0)}
#: family -> list of (glob of plane directories, the same planes' refit directories or None)
FAMILIES = {
    "L_e-L_alpha, ultralight": (X + "/nulri_set1_sm/output/analysis/campaign_final_sm/now_icecube/le_*_ul_fx", None),
    "dark portal, ultralight": (X + "/nulri_set1_final/output/analysis/campaign_final/now_icecube/dp_*_ul_fx", None),
    "secret interactions, below 1 keV": (X + "/nulri_refit_set/output/analysis/campaign_si/now_icecube/si_*_ul_fx", None),
    "L_e-L_alpha and dark portal, MeV": (X + "/nulri_set1_ml/output/analysis/campaign_ml/now_icecube/*_mev_fx",
                                         X + "/nulri_set1_ml/output/analysis/campaign_ml62/now_icecube"),
    "secret interactions, 1 keV up": (X + "/nulri_refit_set/output/analysis/campaign_si/now_icecube/si_*_fx",
                                      X + "/nulri_refit_set/output/analysis/campaign_si_ml62/now_icecube"),
}


def load(dirs):
    """{(log10 mass, log10 coupling): the record of lowest chi^2 with fitted values}"""
    recs = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
            if "/b1" in f and any(p.startswith("b1") and len(p) == 5 for p in f.split("/")):
                continue                                   # merged copies of the refit (b1000+)
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h, v = r.get("held"), r.get("values")
                if not h or not v or r.get("chi2") is None or len(h) != 2:
                    continue
                a, b = list(h.values())
                if a <= 0 or b <= 0:
                    continue
                k = (round(math.log10(a), 3), round(math.log10(b), 3))
                if k not in recs or r["chi2"] < recs[k]["chi2"]:
                    recs[k] = r
    return recs


for fam, (pat, refit) in FAMILIES.items():
    planes = sorted(p for p in glob.glob(pat) if ("_ul" in os.path.basename(p)) == ("_ul" in pat))
    n_cols = 0
    hits = collections.Counter()
    ex = collections.defaultdict(list)
    by_plane = {}
    for p in planes:
        dirs = [p] + ([os.path.join(refit, os.path.basename(p))] if refit and os.path.isdir(os.path.join(refit, os.path.basename(p))) else [])
        R = load(dirs)
        if not R:
            continue
        gmin = min(g for _, g in R)
        sm = min(r["chi2"] for (m, g), r in R.items() if g == gmin)
        cols = collections.defaultdict(list)
        for (m, g), r in R.items():
            cols[m].append((g, r))
        n_here = h_here = 0
        for m, pts in cols.items():
            pts.sort(key=lambda t: t[0])
            cmin = min(min(r["chi2"] for _, r in pts), sm)
            kbest = min(range(len(pts)), key=lambda i: pts[i][1]["chi2"])
            first = next((i for i in range(kbest, len(pts)) if pts[i][1]["chi2"] - cmin >= 3.84), None)
            if first is None or first == 0:
                continue                                   # no limit in this column, or excluded from the bottom
            v = pts[first - 1][1]["values"]                # the last allowed coupling below the limit
            n_cols += 1
            n_here += 1
            any_hit = False
            for name, (lo, hi) in BOUNDS.items():
                if name not in v:
                    continue
                span = hi - lo
                for edge, at in (("low", abs(v[name] - lo) < 0.005 * span), ("high", abs(v[name] - hi) < 0.005 * span)):
                    if at:
                        hits[f"{name} at its {edge} edge ({lo if edge == 'low' else hi:g})"] += 1
                        any_hit = True
                        if len(ex[name + edge]) < 2:
                            ex[name + edge].append(f"{os.path.basename(p)} 10^{m:g} eV")
            h_here += any_hit
            hits["any parameter on an edge"] += any_hit
        by_plane[os.path.basename(p)] = (h_here, n_here)
    print(f"\n== {fam}: {len(by_plane)} planes, {n_cols} mass columns with a limit")
    for k, c in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"   {c:5d} ({100.0 * c / max(n_cols, 1):3.0f}%)  {k}")
    worst = sorted(by_plane.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))[:3]
    print("   most affected planes:", ", ".join(f"{k} {a}/{b}" for k, (a, b) in worst))
