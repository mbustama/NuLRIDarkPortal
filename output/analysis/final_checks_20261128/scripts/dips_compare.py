"""The dips refit with 16 extra starts (output/final_cols/dip_*.json) against the refit's single-start records at
the same couplings (run on Tycho)."""
import glob, json, math
X = "/lustre/astro/mbustama/external"
SETS = [(X + "/nulri_refit_set/output/final_cols/dip_si_*.json", X + "/nulri_refit_set/output/analysis/campaign_si_ml62/now_icecube/si_scalar_majorana_tau_fx", X + "/nulri_refit_set/output/analysis/campaign_si/now_icecube/si_scalar_majorana_tau_fx"),
        (X + "/nulri_set1_ml/output/final_cols/dip_le_*.json", X + "/nulri_set1_ml/output/analysis/campaign_ml62/now_icecube/le_ltau_mev_fx", X + "/nulri_set1_ml/output/analysis/campaign_ml/now_icecube/le_ltau_mev_fx")]
def load(d, lm):
    recs = {}
    for f in glob.glob(d + "/**/plane_*.jsonl", recursive=True):
        if any(p.startswith("b1") and len(p) == 5 for p in f.split("/")): continue
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            h = r.get("held")
            if h and r.get("chi2") is not None:
                a, b = list(h.values())
                if abs(math.log10(a) - lm) < 1e-4:
                    k = round(math.log10(b), 3)
                    if k not in recs or r["chi2"] < recs[k]: recs[k] = r["chi2"]
    return recs
for pat, refit, stored in SETS:
    for f in sorted(glob.glob(pat)):
        o = json.load(open(f)); lm = math.log10(o["mass_eV"])
        old = load(refit, lm); st = load(stored, lm)
        best = {k: min(v for v in (old.get(k), st.get(k)) if v is not None) for k in set(old) | set(st)}
        real = {round(math.log10(float(g)), 3): c for g, c in o["real"].items()}
        ref = min(real, key=float)          # the smallest coupling: no effect
        d = [(g, real[g], best.get(g)) for g in sorted(real)]
        diffs = [a - b for _, a, b in d if b is not None]
        print(f"{f.split('/')[-1]}: mass 10^{lm:.3f} eV, limit read before at 10^{o['log10g_true']}; {len(d)} couplings, {round(o.get('real_seconds', 0))} s")
        print("   multi-start minus single-start chi2: min %+.3f, max %+.3f" % (min(diffs), max(diffs)))
        for g, a, b in d:
            if b is not None and abs(a - b) > 0.05 or abs(g - o["log10g_true"]) < 0.13:
                print(f"     log g {g:7.3f}: {a:.3f} (16 extra starts) | {b if b is None else round(b, 3)} (limits) | Delta chi2 from no effect: {a - real[ref]:+.3f} | {None if b is None else round(b - real[ref], 3):+}")
