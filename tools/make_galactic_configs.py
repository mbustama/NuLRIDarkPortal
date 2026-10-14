"""
make_galactic_configs.py -- the analysis configurations with the Galactic component (Mauricio's task 2, redesigned
2026-10-14 after the first planes lost their limits: with the extragalactic flux anchored only at emission, strong
absorption left a Galactic-dominated flux that looked like pion decay).

Decisions (Mauricio, 2026-10-14): "The flux normalization and shape are floating parameters of the statistical
analysis" (the 15 TeV match only for illustration); the flavor fits take IceCube's 2026 measured astrophysical
spectrum as data (analysis/spectra.py, 'ic26_astrophysical'); "Both Galactic and extragalactic flavor composition
should float as (x, 1-x, 0) ... can be different from one another. For our nominal results, where we do not float
the flavor composition, assume that both ... is from full pion decay."

For every configuration in analysis/configs with planes, writes next to it
  <name>_gal.json   nominal: Galactic and extragalactic sources at pion decay (1/3, 2/3, 0), fixed;
  <name>_galx.json  both compositions free, independently: f_e_S (extragalactic; f_mu_S the remainder, f_tau_S 0)
                    and f_e_S_gal (Galactic), each flat in [0, 1];
both with galactic.enabled (CRINGE, emission points in 3-D), sky.nside 8 (N_side 4 and 8 agree to 1e-3 for the
all-sky flux), data.spectrum 'ic26_astrophysical', and free: the extragalactic normalization norm_astro (flat in
log10 over [1e-2, 1e3], no prior; 1 = without new physics IceCube 2026's flux at 14.7 TeV, the unit set by
sky.extragalactic_norm), the spectral index p_index (flat over [2, 3]; analysis.model interpolates between 7 solver
runs), and norm_gal (Gaussian 0.91 +- 0.18, IceCube 2026's fit of the same template). Output in
output/analysis/<name>_gal and <name>_galx. The originals stay as they are.

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

COMMON = {"norm_gal": {"start": 0.91, "bounds": [0.0, 3.0], "space": "linear"},
          "norm_astro": {"start": 1.0, "bounds": [0.01, 1000.0], "space": "log10"},
          "p_index": {"start": 2.5, "bounds": [2.0, 3.0], "space": "linear"}}
COMPOSITIONS = {"f_e_S": {"start": 0.333333, "bounds": [0.0, 1.0], "space": "linear"},
                "f_e_S_gal": {"start": 0.333333, "bounds": [0.0, 1.0], "space": "linear"}}
PRIORS = [{"name": "norm_gal", "type": "gaussian", "centre": 0.91, "width": 0.18}]
PION = dict(f_e_S=1.0 / 3.0, f_mu_S=2.0 / 3.0, f_tau_S=0.0)


def variant(cfg, name, free_compositions):
    norm, E_ref = dp_galactic.icecube_2026_anchor()
    suffix = "_galx" if free_compositions else "_gal"
    new = copy.deepcopy(cfg)
    new["name"] = name + suffix
    new["_comment_galactic"] = (
        "With the Galactic component (tools/make_galactic_configs.py; Mauricio, 2026-10-14): CRINGE x norm_gal "
        "(Gaussian 0.91 +- 0.18, IceCube 2026), dark-portal emission points in 3-D; the data are the flavor surface "
        "and IceCube 2026's astrophysical spectrum (10 TeV - 1 PeV) against the extragalactic flux at the Earth; "
        "the extragalactic normalization (norm_astro, no prior) and spectral index (p_index) float. "
        + ("Source compositions free: extragalactic (f_e_S, 1 - f_e_S, 0) and Galactic (f_e_S_gal, 1 - f_e_S_gal, "
           "0), independently." if free_compositions else
           "Source compositions fixed at pion decay (1/3, 2/3, 0), Galactic and extragalactic (nominal)."))
    new["data"] = dict(new["data"], spectrum="ic26_astrophysical")
    solver = new["solver"]
    solver.setdefault("galactic", {}).update(enabled=True, template="cringe", emission="3d", f_e_S_gal=1.0 / 3.0)
    solver.setdefault("sky", {}).update(nside=8, extragalactic_norm=norm, extragalactic_norm_E_GeV=E_ref)
    solver.setdefault("source", {}).update(PION)
    for k, v in COMMON.items():
        new["parameters"][k] = v
    if free_compositions:
        for k, v in COMPOSITIONS.items():
            new["parameters"][k] = v
    names = {p["name"] for p in new.get("priors", [])}
    new["priors"] = [p for p in new.get("priors", []) if p["name"] != "norm_astro"] + \
        [p for p in PRIORS if p["name"] not in names]
    new["output_dir"] = os.path.join("output", "analysis", name + suffix)
    return new


def main():
    written = []
    for path in sorted(glob.glob(os.path.join(REPO, "analysis", "configs", "*.json"))):
        name = os.path.basename(path)[:-5]
        if name.endswith(("_gal", "_galx")):
            continue
        with open(path) as f:
            cfg = json.load(f)
        if not cfg.get("planes"):
            continue
        for free in (False, True):
            new = variant(cfg, name, free)
            out = os.path.join(REPO, "analysis", "configs", new["name"] + ".json")
            with open(out, "w") as f:
                json.dump(new, f, indent=1)
                f.write("\n")
            written.append(new["name"])
    print(f"wrote {len(written)} configurations:", " ".join(written))


if __name__ == "__main__":
    main()
