"""
hese_detector_systematics.py -- the HESE release's detector systematics per event, exported as arrays for the
starting-event rates (docs/dev/HESE_RATES_PLAN.md step 3; Mauricio, 2026-10-14: the release's detector systematics as
linear nuisances). Runs the release's own code (det_sys_weights.SysWeighter, photospline splines; PRD 104, 022002,
Eqs. E13-E14) in its own environment (Tycho: conda_envs/hese7np), so our environment never loads it.

The release multiplies each event's weight by 10^(spline(x, epsilon) - spline(x, epsilon_0)) per flux component
(astrophysical, conventional, prompt) and systematic: DOM efficiency (prior 0.99 +- 0.1, bounds 0.8-1.25), head-on
(hole-ice forward) efficiency (0 +- 0.5) and ice-anisotropy scale (1 +- 0.2). Exported per event (the release's cuts
and order, which tools/hese_reduce.py shares; the deposited energies are stored to check the alignment): log10 of the
exact weights on a grid of each systematic's values spanning about +-2 sigma (GRIDS), float32, interpolated by the
consumer (IceCubeRates.starting_events.hese: cubic spline per event). A single slope at the nominal value, tried
first, missed the weights at 1 sigma by 30-50% (head-on) and by factors (DOM efficiency) for some events.

  /lustre/astro/mbustama/conda_envs/hese7np/bin/python tools/hese_detector_systematics.py --hese RELEASE --out FILE
"""
import argparse
import os
import sys
import time

import numpy as np

SYSTEMATICS = (("dom_eff", "get_dom_eff_weights", 0.99, 0.1), ("head_on", "get_hole_ice_weights", 0.0, 0.5),
               ("anisotropy", "get_anisotropy_weights", 1.0, 0.2))
GRIDS = {"dom_eff": [0.80, 0.85, 0.90, 0.95, 0.99, 1.04, 1.09, 1.14, 1.19],
         "head_on": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
         "anisotropy": [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]}
COMPONENTS = ("Astro", "Conv", "Prompt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hese", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, a.hese)
    os.chdir(a.hese)                                    # the release finds its splines relative to its folder
    from data_loader import load_mc
    from det_sys_weights import SysWeighter
    files = [os.path.join("resources", "data", f) for f in ("HESE_mc_observable.json", "HESE_mc_flux.json",
                                                             "HESE_mc_truth.json")]
    mc = load_mc(files)
    sw = SysWeighter(mc)
    out = dict(E_dep_GeV=np.asarray(mc["recoDepositedEnergy"]), nominal=np.array([s[2] for s in SYSTEMATICS]),
               sigma=np.array([s[3] for s in SYSTEMATICS]), names=np.array([s[0] for s in SYSTEMATICS]))
    nu = np.abs(np.asarray(mc["primaryType"])) != 13
    for name, getter, eps0, sig in SYSTEMATICS:
        f = getattr(sw, getter)
        grid = np.array(GRIDS[name])
        out[f"grid_{name}"] = grid
        for comp in COMPONENTS:
            logw = np.empty((len(grid), len(mc)), dtype=np.float32)
            for i, eps in enumerate(grid):
                w, _ = f(comp, (float(eps), np.array([1.0])))
                logw[i] = np.log10(np.maximum(w, 1e-30))
            out[f"log10w_{name}_{comp}"] = logw
            i0 = int(np.argmin(np.abs(grid - eps0)))
            spread = np.percentile(np.abs(logw[:, nu] - logw[i0, nu]), 99, axis=1)
            print(f"{name:10s} {comp:6s}: at nominal |log10 w| max {np.abs(logw[i0, nu]).max():.2e}; 99th percentile of "
                  f"|log10 w| over the grid {np.round(spread, 3).tolist()}", flush=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, {len(mc)} events, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
