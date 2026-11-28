"""The limit at each dip with the fits started from 16 more points: the merged column of limits_20261127 with the
new chi^2 where it is lower, read by the limits' own rule (tools/coverage_limits.read). Run from the repo root."""
import glob, json, math, sys
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import coverage_limits as cl
PL = {"si": "output/analysis/limits_20261127/nulri_refit_set/campaign_si/now_icecube/si_scalar_majorana_tau_fx",
      "le": "output/analysis/limits_20261127/nulri_set1_ml/campaign_ml/now_icecube/le_ltau_mev_fx"}
for f in sorted(glob.glob(sys.argv[1] + "/dip_*.json")):
    o = json.load(open(f)); kind = f.split("dip_")[1][:2]
    s = json.load(open(glob.glob(PL[kind] + "/data/plane_*.json")[0]))
    X = np.asarray(s["grids"][0], float); Y = np.asarray(s["grids"][1], float)
    i = int(np.argmin(abs(np.log10(X) - math.log10(o["mass_eV"]))))
    col = {float(y): c for y, c in zip(Y, s["chi2"][i]) if c is not None}
    new = dict(col); n_low = 0; big = 0.0
    for g, c in o["real"].items():
        g = float(g); k = min(col, key=lambda y: abs(math.log10(y) - math.log10(g)))
        assert abs(math.log10(k) - math.log10(g)) < 1e-6
        if c < new[k] - 1e-9: big = max(big, new[k] - c); new[k] = c; n_low += 1
    g_t = 10.0 ** o["log10g_true"]
    a = cl.read(tuple(s["names"]), X[i], col, g_t)[1]; b = cl.read(tuple(s["names"]), X[i], new, g_t)[1]
    print(f"{f.split('/')[-1]}: 10^{math.log10(X[i]):.3f} eV; {len(o['real'])} couplings refit, {n_low} lower (largest drop {big:.3f}); limit log10 g {a:.3f} -> {b:.3f}")
