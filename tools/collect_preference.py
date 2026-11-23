"""collect_preference.py -- IceCube's preference for an MeV mediator, per mass, from the final planes (2026-11-22, for
the paper's Sec. V.5 and Fig. 12; his go "Yes, draft both").

For each present-day data set (now_icecube: MESE flavor + spectrum; now_mese_dr2: MESE events + DR2 tracks) and each
MeV plane with the source composition fitted (*_fx; le_lmu_mev, le_ltau_mev, dp_*_mev, and the 16 secret-interaction
planes), the preference at each mass is chi^2 at the column's smallest coupling (no new interaction) minus the lowest
chi^2 over the coupling, as in the limits (per-mass Delta chi^2). Also stored: the coupling of that lowest chi^2.

The planes are the copies made by the limits collection of 2026-11-21 (the checkouts' plane_*.json, one per block),
passed as the root folder:
  python tools/collect_preference.py ROOT   -> output/analysis/preference_20261122.json
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "output", "analysis", "preference_20261122.json")
DATASETS = ("now_icecube", "now_mese_dr2")


def collect(root):
    out = {}
    for ds in DATASETS:
        for f in sorted(glob.glob(os.path.join(root, "*", "*", ds, "*_fx", "**", "plane_*.json"), recursive=True)):
            plane = f.split(f"/{ds}/")[1].split("/")[0]
            if "_ul" in plane or not ("_mev" in plane or plane.startswith("si_")):
                continue
            d = json.load(open(f))
            X = np.log10(np.asarray(d["grids"][0], float))
            Y = np.log10(np.asarray(d["grids"][1], float))
            chi = np.array([[np.nan if v is None else v for v in row] for row in d["chi2"]], float)
            rec = out.setdefault(ds, {}).setdefault(plane, {"source": [], "columns": {}})
            rec["source"].append(os.path.relpath(f, root))
            for i, x in enumerate(X):
                if np.isnan(chi[i, 0]) or np.all(np.isnan(chi[i])):
                    continue
                j = int(np.nanargmin(chi[i]))
                rec["columns"][f"{x:.4f}"] = {"preference": float(chi[i, 0] - chi[i, j]), "log10_g_best": float(Y[j])}
    return out


if __name__ == "__main__":
    res = collect(sys.argv[1])
    res["_comment"] = ("per-mass preference = chi^2 at the smallest coupling minus the lowest chi^2 over the coupling "
                       "(tools/collect_preference.py, 2026-11-22)")
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    for ds in DATASETS:
        for plane, rec in sorted(res[ds].items()):
            c = rec["columns"]
            k = max(c, key=lambda x: c[x]["preference"])
            print(f"{ds:13s} {plane:32s} max {c[k]['preference']:5.2f} at log10 M = {float(k):.3f}, "
                  f"log10 g = {c[k]['log10_g_best']:.2f}")
