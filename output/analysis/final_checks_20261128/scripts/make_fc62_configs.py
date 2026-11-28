"""Configs of the forecast refit with a 62-mass lightest-mass scan (Mauricio, 2026-11-27: "Launch the forecast refit
tonight"): copies of the forecast configs (proj_future_planA, fitted composition) with exact_profile.ml_scan = [1e-6] +
61 masses from 1e-4 to 0.1 eV, as in the refit of the present-day limits (make_ml62_configs.py), and chains of 2
(a chain keeps the scans of all its points in memory: with chains of 8 the pilot passed 51 GB after 86 min), for
the mediator masses 10^5.5 .. 10^6.375 eV (8 columns), in new output directories.
  si: run in nulri_forecast  -> the campaign's blocks b10..b13 of the 16 MeV secret-interaction planes (64 configs)
  ml: run in nulri_set1_ml   -> le_lmu, le_ltau, dp_e, dp_mu, dp_tau _mev_fx cut into 2-column blocks b6..b9 (20)
Usage (from the checkout): python ../make_fc62_configs.py si|ml PACKS.txt"""
import glob
import json
import os
import sys

import numpy as np

ML_SCAN = [1e-6] + [float(m) for m in np.geomspace(1e-4, 0.1, 61)]
LO, HI = 5.5, 6.375


def edit(d, out_dir):
    d["exact_profile"]["ml_scan"] = ML_SCAN
    d["exact_profile"]["chain_length"] = 2
    d["output_dir"] = out_dir
    return d


def main(kind, packs):
    lines = []
    if kind == "si":
        for f0 in sorted(glob.glob("analysis/configs/campaign_si_b/proj_future_planA/si_*_fx__b*.json")):
            base = os.path.basename(f0)
            if "_ul" in base:
                continue
            d = json.load(open(f0))
            p0 = d["planes"][0]
            lx = np.log10(np.geomspace(p0["x_range"][0], p0["x_range"][1], p0["nx"]))
            if not (lx.min() > LO - 1e-6 and lx.max() < HI + 1e-6):
                continue
            name, b = base[:-5].split("__")
            edit(d, f"output/analysis/campaign_si_fc62/proj_future_planA/{name}/{b}")
            f = f"analysis/configs/campaign_si_fc62/proj_future_planA/{base}"
            os.makedirs(os.path.dirname(f), exist_ok=True)
            json.dump(d, open(f, "w"), indent=1)
            lines.append(f)
    else:
        for name in ("le_lmu_mev_fx", "le_ltau_mev_fx", "dp_e_mev_fx", "dp_mu_mev_fx", "dp_tau_mev_fx"):
            d0 = json.load(open(f"analysis/configs/campaign_ml/proj_future_planA/{name}.json"))
            p0 = d0["planes"][0]
            x = np.geomspace(p0["x_range"][0], p0["x_range"][1], p0["nx"])
            cuts = [list(range(k, k + 2)) for k in range(0, len(x) - 3, 2)]
            cuts.append(list(range(cuts[-1][-1] + 1, len(x))))
            for b, c in enumerate(cuts):
                lx = np.log10(x[c])
                if not (lx.min() > LO - 1e-6 and lx.max() < HI + 1e-6):
                    continue
                d = json.loads(json.dumps(d0))
                d["planes"][0].update(x_range=[float(x[c[0]]), float(x[c[-1]])], nx=len(c))
                edit(d, f"output/analysis/campaign_ml_fc62/proj_future_planA/{name}/b{b}")
                f = f"analysis/configs/campaign_ml_fc62/proj_future_planA/{name}__b{b}.json"
                os.makedirs(os.path.dirname(f), exist_ok=True)
                json.dump(d, open(f, "w"), indent=1)
                lines.append(f)
    open(packs, "w").write("\n".join(lines) + "\n")
    print(kind, len(lines), "configs ->", packs)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
