"""
hese_reduce.py -- the HESE 7.5-year release's Monte Carlo and data reduced to arrays for the starting-event rates
(docs/dev/HESE_RATES_PLAN.md, Mauricio's task 1). Reads the release's JSON files (data only; the release on Tycho in
external/hese_7yr, commit 2d1ed03), applies its cuts (data_loader.py: deposited energy 60 TeV-10 PeV, double cascades
10-1000 m long) and writes one npz (~70 MB, kept outside the repository; IceCubeRates.starting_events.hese reads it):

  mc_type (PDG: +-12, +-14, +-16), mc_E_GeV (true energy), mc_cos_zenith (true), mc_interaction (1 CC, 2 NC, 3 GR),
  mc_weight (weightOverFluxOverLivetime, GeV sr cm^2), mc_E_dep_GeV, mc_zenith_reco (rad), mc_morphology (0 cascade,
  1 track, 2 double cascade), mc_length_m, mc_pion_flux, mc_kaon_flux, mc_prompt_flux (the release's atmospheric
  fluxes), mc_conv_veto, mc_prompt_veto (its self-veto corrections), mc_muon_weight (muonWeightOverLivetime, s^-1);
  data_* (the 102 events: E_dep, zenith, morphology, length); livetime_s; the release's bins (binning.py defaults).

  python -I tools/hese_reduce.py --hese .../HESE-7-year-data-release --out FILE.npz
"""
import argparse
import json
import os
import time

import numpy as np

LIVETIME_S = 227708167.68
CUTS = dict(emin=60.0e3, emax=1.0e7, lmin=10.0, lmax=1000.0)


def release_bins(emin=60.0e3, emax=1.0e7, ewidth=0.111, eedge=60.0e3, lmin=10.0, lmax=1000.0, lwidth=0.1,
                 ledge=10.0, nzenith=10):
    """The release's binning.get_bins, rewritten (same arithmetic): energy, zenith (rad, increasing), length."""
    n, lo = 1, eedge
    while lo > emin:
        n += 1
        lo /= 10.0 ** ewidth
    hi = eedge
    while hi < emax:
        n += 1
        hi *= 10.0 ** ewidth
    energy = np.logspace(np.log10(lo), np.log10(hi), n)
    zenith = np.arccos(np.linspace(-1, 1, nzenith + 1))[::-1]
    n, lo = 1, ledge
    while lo > lmin:
        n += 1
        lo /= 10.0 ** lwidth
    hi = ledge
    while hi < lmax:
        n += 1
        hi *= 10.0 ** lwidth
    length = np.logspace(np.log10(lo), np.log10(hi), n)
    return energy, zenith, length


def keep(E, morph, L):
    with np.errstate(invalid="ignore"):
        return ((E >= CUTS["emin"]) & (E <= CUTS["emax"])
                & ((morph != 2) | ((L >= CUTS["lmin"]) & (L <= CUTS["lmax"]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hese", required=True, help="the release's folder (with resources/data)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    d = {}
    for f in ("HESE_mc_observable.json", "HESE_mc_flux.json", "HESE_mc_truth.json"):
        with open(os.path.join(a.hese, "resources", "data", f)) as fh:
            d.update(json.load(fh))
    d = {k: np.asarray(v) for k, v in d.items()}
    k = keep(d["recoDepositedEnergy"], d["recoMorphology"], d["recoLength"])
    with open(os.path.join(a.hese, "resources", "data", "HESE_data.json")) as fh:
        data = {kk: np.asarray(v) for kk, v in json.load(fh).items()}
    kd = keep(data["recoDepositedEnergy"], data["recoMorphology"], data["recoLength"])
    eb, zb, lb = release_bins()
    out = dict(mc_type=d["primaryType"][k].astype(np.int16), mc_E_GeV=d["primaryEnergy"][k],
               mc_cos_zenith=np.cos(d["primaryZenith"][k]), mc_interaction=d["interactionType"][k].astype(np.int8),
               mc_weight=d["weightOverFluxOverLivetime"][k], mc_E_dep_GeV=d["recoDepositedEnergy"][k],
               mc_zenith_reco=d["recoZenith"][k], mc_morphology=d["recoMorphology"][k].astype(np.int8),
               mc_length_m=d["recoLength"][k], mc_pion_flux=d["pionFlux"][k], mc_kaon_flux=d["kaonFlux"][k],
               mc_prompt_flux=d["promptFlux"][k], mc_conv_veto=d["conventionalSelfVetoCorrection"][k],
               mc_prompt_veto=d["promptSelfVetoCorrection"][k], mc_muon_weight=d["muonWeightOverLivetime"][k],
               data_E_dep_GeV=data["recoDepositedEnergy"][kd], data_zenith_reco=data["recoZenith"][kd],
               data_morphology=data["recoMorphology"][kd].astype(np.int8), data_length_m=data["recoLength"][kd],
               livetime_s=LIVETIME_S, energy_bins_GeV=eb, zenith_bins_rad=zb, length_bins_m=lb,
               cuts=np.array(json.dumps(CUTS)), n_mc_all=len(k), source=np.array(
                   "IceCube HESE 7.5-yr release (github.com/icecube/HESE-7-year-data-release; PRD 104, 022002), "
                   "reduced by tools/hese_reduce.py with the release's cuts"))
    np.savez_compressed(a.out, **out)
    nu = np.abs(out["mc_type"]) != 13
    print(f"MC: {k.sum()} of {len(k)} events pass the cuts ({nu.sum()} neutrino events); data: {kd.sum()} of "
          f"{len(kd)}; bins: {len(eb) - 1} energy ({eb[0]:.4g}-{eb[-1]:.4g} GeV), {len(zb) - 1} zenith, "
          f"{len(lb) - 1} length; types {sorted(set(out['mc_type'].tolist()))}; wrote {a.out} "
          f"({os.path.getsize(a.out) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
