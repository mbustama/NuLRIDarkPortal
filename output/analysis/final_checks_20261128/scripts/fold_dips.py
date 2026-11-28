"""Fold the six dip columns refit with 16 extra starts (output/final_cols/dip_*.json of 2026-11-28) into the joined
planes of limits_20261127: at each coupling the lower chi^2 is kept (his rule; Mauricio, 2026-11-28: "Yes"). The
changes are listed in the plane summary under "multistart_20261128". Run from the repo root: python fold_dips.py <dir>"""
import glob, json, math, sys
import numpy as np
PL = {"si": "output/analysis/limits_20261127/nulri_refit_set/campaign_si/now_icecube/si_scalar_majorana_tau_fx",
      "le": "output/analysis/limits_20261127/nulri_set1_ml/campaign_ml/now_icecube/le_ltau_mev_fx"}
for kind, d in PL.items():
    f = glob.glob(d + "/data/plane_*.json")[0]; s = json.load(open(f))
    X = np.asarray(s["grids"][0], float); Y = np.asarray(s["grids"][1], float)
    log = [e for e in s.get("multistart_20261128", [])]
    done = {(e["log10_m"], e["log10_g"]) for e in log}
    for dip in sorted(glob.glob(sys.argv[1] + f"/dip_{kind}_*.json")):
        o = json.load(open(dip)); i = int(np.argmin(abs(np.log10(X) - math.log10(o["mass_eV"]))))
        for g, c in o["real"].items():
            j = int(np.argmin(abs(np.log10(Y) - math.log10(float(g)))))
            assert abs(math.log10(Y[j]) - math.log10(float(g))) < 1e-6
            old = s["chi2"][i][j]; key = (round(math.log10(X[i]), 3), round(math.log10(Y[j]), 3))
            if old is not None and c < old - 1e-9 and key not in done:
                s["chi2"][i][j] = float(c); log.append(dict(log10_m=key[0], log10_g=key[1], before=old, after=float(c), source=dip.split("/")[-1]))
    s["multistart_20261128"] = log
    json.dump(s, open(f, "w"))
    print(kind, f.split("limits_20261127/")[1], ":", len(log), "points lowered; largest drop %.3f" % max(e["before"] - e["after"] for e in log))
