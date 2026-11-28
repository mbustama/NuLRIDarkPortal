"""Limits with the wide spectral-index ranges (gamma 2-4, gamma_low 0.1-2.5) against the limits of limits_20261127
(gamma 2-3, gamma_low 0.5-2.5), per mediator mass; both read by tools/export_limits.py. Run from the repo root:
  python idx_compare.py <scratch>/idx"""
import glob, json, os, shutil, subprocess, sys
import numpy as np
D = sys.argv[1]
shutil.rmtree(D + "/lim", ignore_errors=True)
r = subprocess.run([sys.executable, "tools/export_limits.py", D + "/raw", "--out", D + "/lim"], capture_output=True, text=True)
print(r.stdout.strip()[-400:]); assert r.returncode == 0, r.stderr[-600:]
REF = {"si_scalar_majorana_tau_fx": "output/analysis/limits_20261127/nulri_refit_set/campaign_si/now_icecube/si_scalar_majorana_tau_fx",
       "le_ltau_mev_fx": "output/analysis/limits_20261127/nulri_set1_ml/campaign_ml/now_icecube/le_ltau_mev_fx"}
for plane, ref in REF.items():
    new = json.load(open(f"{D}/lim/now_icecube/{plane}/limits.json")); old = json.load(open(ref + "/limits.json"))
    O = {round(c["log10_m"], 3): c for c in old["columns"]}
    print(f"\n{plane}: no-effect chi2 {old['chi2_no_effect']:.4f} (standard ranges) -> {new['chi2_no_effect']:.4f} (wide)")
    print("  log10 M/eV | limit, standard ranges -> wide ranges (change in log10 g) | column complete")
    d = []
    for c in new["columns"]:
        m = round(c["log10_m"], 3); a = O[m]["lowest_excluded_log10_g"]; b = c["lowest_excluded_log10_g"]
        ch = None if a is None or b is None else b - a
        if ch is not None and c["all_couplings_computed"]: d.append(ch)
        print(f"   {m:6.3f} | {a} -> {b} ({'' if ch is None else format(ch, '+.3f')}) | {c['all_couplings_computed']}")
    if d:
        d = np.array(d); print(f"  complete columns: {len(d)}; change of the limit: median {np.median(d):+.3f}, largest weakening {d.max():+.3f}, largest strengthening {d.min():+.3f}; weaker by more than 0.1 in {int((d > 0.1).sum())}, by more than 0.3 in {int((d > 0.3).sum())}")
