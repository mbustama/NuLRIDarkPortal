"""
make_galactic_configs.py -- the analysis configurations with the Galactic component (Mauricio, 2026-10-14: "Do 2"
-- the Galactic share in the flavor fits; decisions: the extragalactic flux anchored to IceCube's 2026
astrophysical flux at 10-21.5 TeV with a nuisance norm_astro, Gaussian 1 +- 0.26; norm_gal Gaussian 0.91 +- 0.18;
x_gal = f_e_S_gal flat in [0, 1]; dark-portal emission points in 3-D).

For every configuration in analysis/configs with planes, writes <name>_gal.json next to it: the same analysis
with galactic.enabled (CRINGE, emission '3d'), sky.nside 8 (the all-sky flux needs no finer pixels: N_side 4 and
8 agree to 1e-3, tests/test_galactic.py), sky.extragalactic_norm at dp_galactic.ICECUBE_2026_ANCHOR, the three
nuisances free with their priors, and output in output/analysis/<name>_gal. The originals stay as they are.

  python tools/make_galactic_configs.py
"""
import copy
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_galactic  # noqa: E402

NUISANCES = {"norm_gal": {"start": 0.91, "bounds": [0.0, 3.0], "space": "linear"},
             "f_e_S_gal": {"start": 0.333333, "bounds": [0.0, 1.0], "space": "linear"},
             "norm_astro": {"start": 1.0, "bounds": [0.1, 3.0], "space": "linear"}}
PRIORS = [{"name": "norm_gal", "type": "gaussian", "centre": 0.91, "width": 0.18},
          {"name": "norm_astro", "type": "gaussian", "centre": 1.0, "width": 0.26}]


def main():
    norm, E_ref = dp_galactic.icecube_2026_anchor()
    written = []
    for path in sorted(glob.glob(os.path.join(REPO, "analysis", "configs", "*.json"))):
        name = os.path.basename(path)[:-5]
        if name.endswith("_gal"):
            continue
        with open(path) as f:
            cfg = json.load(f)
        if not cfg.get("planes"):
            continue
        new = copy.deepcopy(cfg)
        new["name"] = name + "_gal"
        new["_comment_galactic"] = (
            "With the Galactic component (tools/make_galactic_configs.py, 2026-10-14): CRINGE x norm_gal (Gaussian "
            "0.91 +- 0.18, IceCube 2026), sources (x, 1 - x, 0) with x = f_e_S_gal flat in [0, 1], dark-portal "
            "emission points in 3-D; the extragalactic flux in physical units anchored to IceCube's 2026 "
            "astrophysical flux at 10-21.5 TeV (E^2 Phi = 3.623e-8 per flavor at 14.7 TeV) times norm_astro "
            "(Gaussian 1 +- 0.26, the bin's errors). The nuisances are linear: no extra solver calls.")
        solver = new["solver"]
        solver.setdefault("galactic", {}).update(enabled=True, template="cringe", emission="3d")
        solver.setdefault("sky", {}).update(nside=8, extragalactic_norm=norm, extragalactic_norm_E_GeV=E_ref)
        for k, v in NUISANCES.items():
            new["parameters"].setdefault(k, v)
        names = {p["name"] for p in new.get("priors", [])}
        new["priors"] = list(new.get("priors", [])) + [p for p in PRIORS if p["name"] not in names]
        new["output_dir"] = os.path.join("output", "analysis", name + "_gal")
        out = os.path.join(REPO, "analysis", "configs", name + "_gal.json")
        with open(out, "w") as f:
            json.dump(new, f, indent=1)
            f.write("\n")
        written.append(name + "_gal")
    print(f"wrote {len(written)} configurations:", " ".join(written))


if __name__ == "__main__":
    main()
