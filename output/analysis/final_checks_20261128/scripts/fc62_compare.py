"""Forecast refit so far against the 11-mass forecast, point by point (run on Tycho): where does chi^2 drop?"""
import json, glob, math, collections, os
X = "/lustre/astro/mbustama/external"
def load(pat):
    recs = {}
    for f in glob.glob(pat, recursive=True):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if "held" in r and "chi2" in r:
                h = list(r["held"].values()); k = (round(math.log10(h[0]), 3), round(math.log10(h[1]), 3))
                if k not in recs or r["chi2"] < recs[k]["chi2"]: recs[k] = r
    return recs
rows = collections.defaultdict(list)
for kind, new_root, old_root in (("si", X + "/nulri_forecast/output/analysis/campaign_si_fc62/proj_future_planA", X + "/nulri_forecast/output/analysis/campaign_si/proj_future_planA"),
                                 ("ml", X + "/nulri_set1_ml/output/analysis/campaign_ml_fc62/proj_future_planA", X + "/nulri_set1_ml/output/analysis/campaign_ml/proj_future_planA")):
    for pd in sorted(glob.glob(new_root + "/*")):
        plane = os.path.basename(pd)
        new = load(pd + "/b*/asimov/plane_*.jsonl")
        old = load(old_root + f"/{plane}/**/asimov/plane_*.jsonl") if kind == "si" else load(old_root + f"/{plane}/**/plane_*.jsonl")
        c0 = min(v["chi2"] for v in old.values())
        for k, r in new.items():
            if k in old:
                rows[k[0]].append((plane, k[1], old[k]["chi2"] - c0, r["chi2"] - c0, r["values"].get("m_lightest_eV"), old[k]["values"].get("m_lightest_eV")))
print("per mediator mass (log10 M/eV): points compared; of those with old Delta chi^2 > 1: count, and how many dropped by more than 20% / below 3.84 from above")
for lm in sorted(rows):
    v = rows[lm]; eff = [x for x in v if x[2] > 1.0]
    drop = [x for x in eff if x[3] < 0.8 * x[2]]; lost = [x for x in eff if x[2] >= 3.84 and x[3] < 3.84]
    print(f"  {lm:.3f}: {len(v)} points; with an effect {len(eff)}; dropped >20%: {len(drop)}; excluded -> allowed: {len(lost)}")
    for x in sorted(drop, key=lambda x: x[3] / x[2])[:4]:
        print(f"      {x[0]} g=10^{x[1]:.3f}: Delta chi^2 {x[2]:.2f} -> {x[3]:.2f}; lightest mass {x[5]:.3g} -> {x[4]:.3g} eV")

import numpy as np
print("\npoints with an effect: new minus old Delta chi^2")
alld = []
for lm in sorted(rows):
    eff = [x for x in rows[lm] if x[2] > 1.0]
    if not eff: continue
    d = np.array([x[3] - x[2] for x in eff]); rel = np.array([(x[3] - x[2]) / x[2] for x in eff]); alld += list(d)
    near = [x for x in eff if 2.0 <= x[2] <= 20.0]
    print(f"  {lm:.3f}: n {len(eff)}; change min {d.min():+.3f} median {np.median(d):+.3f} max {d.max():+.3f}; largest relative drop {100 * rel.min():+.1f}%; near the threshold (old 2-20): {len(near)} points in {len({x[0] for x in near})} cases, their largest drop {min([x[3] - x[2] for x in near] or [0]):+.3f}; couplings 10^{min(x[1] for x in eff):.2f}..10^{max(x[1] for x in eff):.2f}")
    for x in sorted(near, key=lambda x: x[3] - x[2])[:3]:
        print(f"      {x[0]} g=10^{x[1]:.3f}: {x[2]:.3f} -> {x[3]:.3f}; lightest mass {x[5]:.3g} -> {x[4]:.3g} eV")
print("cases with points near the threshold:", sorted({x[0] for lm in rows for x in rows[lm] if 2.0 <= x[2] <= 20.0}))
