"""
hese_toolkit_weights.py -- the HESE 7.5-yr release's own astrophysical weights per MC event, computed with the
release's code (data_loader, binning, weighter, det_sys_weights; github.com/icecube/HESE-7-year-data-release),
for the event-by-event comparison in tools/hese_check.py (plan step 8). Runs in the release's own environment
(Tycho: conda env hese7np -- Python 3.10, NumPy 1.23 (the release builds ragged arrays, which NumPy >= 1.24
rejects), photospline for the detector-systematics splines), not the solver's; it imports
nothing from this repository. Approved by Mauricio, 2026-10-14 ("both checks").

At the paper's best fit (astro_norm 6.37, astro_gamma 2.87) with the atmospheric and muon components off
(conv_norm = prompt_norm = muon_norm = 0) and the detector parameters at the fit script's nominal values
(epsilon_dom 0.99, epsilon_head_on 0, anisotropy_scale 1; cr_delta_gamma -0.05, nunubar_ratio 1, kpi_ratio 1):
  astro_weight_bare  livetime x weightOverFluxOverLivetime x the release's flux per type (weighter.flux_spl);
  astro_weight_full  the release's Weighter.get_weights (with its detector corrections).

  /lustre/astro/mbustama/conda_envs/hese7np/bin/python tools/hese_toolkit_weights.py --hese RELEASE_DIR --out FILE.npz
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hese", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)
    os.chdir(a.hese)                      # the release loads its files by relative paths
    sys.path.insert(0, a.hese)
    import data_loader
    import binning
    import weighter
    livetime = 227708167.68
    mc = data_loader.load_mc(["./resources/data/HESE_mc_observable.json", "./resources/data/HESE_mc_flux.json",
                              "./resources/data/HESE_mc_truth.json"])
    sorted_mc, _ = binning.bin_data(mc)
    W = weighter.Weighter(sorted_mc)
    names = ["cr_delta_gamma", "nunubar_ratio", "anisotropy_scale", "astro_gamma", "astro_norm", "conv_norm",
             "epsilon_dom", "epsilon_head_on", "muon_norm", "kpi_ratio", "prompt_norm"]
    params = np.array([-0.05, 1.0, 1.0, 2.87, 6.37, 0.0, 0.99, 0.0, 0.0, 1.0, 0.0])
    full = np.asarray(W.get_weights(livetime, names, params)[0], dtype=np.float64)
    g = np.zeros(len(names))
    flux = W.flux_spl(sorted_mc, astro_norm=[6.37, g.copy()], astro_gamma=[2.87, g.copy()])
    bare = np.asarray(flux[0], dtype=np.float64) * sorted_mc["weightOverFluxOverLivetime"] * livetime
    np.savez_compressed(out_path, primaryEnergy=sorted_mc["primaryEnergy"], primaryZenith=sorted_mc["primaryZenith"],
                        recoDepositedEnergy=sorted_mc["recoDepositedEnergy"], recoMorphology=sorted_mc["recoMorphology"],
                        primaryType=sorted_mc["primaryType"], astro_weight_bare=bare, astro_weight_full=full,
                        note=np.array("release's code at the best fit (6.37, 2.87), atmospheric and muons off, detector "
                                      "parameters nominal (0.99, 0, 1)"))
    print(f"{len(bare)} events; astrophysical events: bare {bare.sum():.4f}, with detector corrections {full.sum():.4f}; "
          f"wrote {out_path}")


if __name__ == "__main__":
    main()
