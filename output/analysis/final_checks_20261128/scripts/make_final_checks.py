"""make_final_checks.py -- configurations of the last checks (2026-11-27, Mauricio: "if re-runs are to happen, they
should happen now"; "I need to do a final run"). Run from a checkout: python3 ../make_final_checks.py si|ml

idx  How much the MeV limits of today's data depend on the allowed ranges of the spectral indices: the refit's blocks
     (62 lightest masses, chains of 8) of one case per checkout, with gamma in 2-4 (13 nodes, the present spacing) and
     gamma_low in 0.1-2.5 (4 nodes; the solver needs a positive index) instead of 2-3 (7) and 0.5-2.5 (3); mediator masses 10^4.75 .. 10^7.125 eV.
       si: si_scalar_majorana_tau_fx blocks b7..b16      ml: le_ltau_mev_fx blocks b3..b12
     -> analysis/configs/final_idx/, output/analysis/final_idx/<plane>/b<k>, packs_final_idx.txt
ul   One column of an ultralight plane with a 62-mass scan of the lightest mass (these planes fit it from its start,
     without a scan): a copy of the block's config with exact_profile.ml_scan, for tools/coverage_limits.py --toys 0.
       si: si_scalar_majorana_tau_ul_fx (block b10)      ml: le_ltau_ul_fx (block b3, config from nulri_set1_sm)
     -> analysis/configs/final_ul/<plane>.json"""
import glob
import json
import os
import sys

import numpy as np

ML_SCAN = [1e-6] + [float(m) for m in np.geomspace(1e-4, 0.1, 61)]
X = "/lustre/astro/mbustama/external"


def main(kind):
    if kind == "si":
        src, plane, blocks = "analysis/configs/campaign_si_ml62/now_icecube", "si_scalar_majorana_tau_fx", range(7, 17)
        ul_src = "analysis/configs/campaign_si_b/now_icecube/si_scalar_majorana_tau_ul_fx__b10.json"
        ul_name = "si_scalar_majorana_tau_ul_fx"
    else:
        src, plane, blocks = "analysis/configs/campaign_ml62/now_icecube", "le_ltau_mev_fx", range(3, 13)
        ul_src = X + "/nulri_set1_sm/analysis/configs/campaign_final_sm/now_icecube/le_ltau_ul_fx__b3.json"
        ul_name = "le_ltau_ul_fx"
    lines = []
    os.makedirs("analysis/configs/final_idx", exist_ok=True)
    for b in blocks:
        d = json.load(open(f"{src}/{plane}__b{b}.json"))
        assert d["parameters"]["p_index"]["bounds"] == [2.0, 3.0] and d["parameters"]["p_index_low"]["bounds"] == [0.5, 2.5]
        d["parameters"]["p_index"].update(bounds=[2.0, 4.0], nodes=13)
        d["parameters"]["p_index_low"].update(bounds=[0.1, 2.5], nodes=4)   # the solver needs a positive index
        d["output_dir"] = f"output/analysis/final_idx/{plane}/b{b}"
        f = f"analysis/configs/final_idx/{plane}__b{b}.json"
        json.dump(d, open(f, "w"), indent=1)
        lines.append(f)
    open("packs_final_idx.txt", "w").write("\n".join(lines) + "\n")
    os.makedirs("analysis/configs/final_ul", exist_ok=True)
    d = json.load(open(ul_src))
    d.setdefault("exact_profile", {})["ml_scan"] = ML_SCAN
    d["output_dir"] = f"output/analysis/final_ul/{ul_name}"
    json.dump(d, open(f"analysis/configs/final_ul/{ul_name}.json", "w"), indent=1)
    p = d["planes"][0]
    print(kind, len(lines), "index-range configs -> packs_final_idx.txt;", f"final_ul/{ul_name}.json: masses",
          np.round(np.log10(np.geomspace(p["x_range"][0], p["x_range"][1], p["nx"])), 3).tolist(), "couplings",
          np.round(np.log10(p["y_range"]), 2).tolist(), p.get("ny"))


if __name__ == "__main__":
    main(sys.argv[1])
