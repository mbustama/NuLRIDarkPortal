# Plan: starting events with the HESE 7.5-year release (Mauricio's task 1)

Status: **agreed 2026-10-14** (Mauricio: MCEq + Barr with the release's self-veto; the release's detector
systematics as linear nuisances; rates and figures first, then the likelihood). Builds on the per-direction output
(docs/dev/DIRECTIONS_PLAN.md, step 8: our weighting of the release's MC equals the release's code to 2e-14), the
event-rate module (IceCubeRates/, task 3: docs/dev/DR2_TRACKS_PLAN.md), the atmospheric background as in
NuFlavorSingleSource (MCEq + Barr; Mauricio 2026-10-14), and "the flux normalization and shape are floating
parameters of the statistical analysis" (Mauricio 2026-10-14).

## 1. Goal

Expected HESE events (cascades, tracks, double cascades) in reconstructed deposited energy and direction for any
flux the solver computes (dark portal, secret interactions, standard), extragalactic and Galactic, with the
atmospheric backgrounds; then the binned Poisson likelihood against the 102 observed events, with the release's
nuisances and ours.

## 2. Inputs (on Tycho, verified 2026-10-13/14)

| Item | Content | Use |
|---|---|---|
| HESE 7.5-yr release (github.com/icecube/HESE-7-year-data-release, 2d1ed03; PRD 104, 022002) | 821 764 MC events: true type (PDG), energy, zenith, interaction (CC/NC/GR), weightOverFluxOverLivetime; reconstructed deposited energy, zenith, morphology, length; per event pionFlux, kaonFlux, promptFlux, conventional and prompt self-veto corrections, muonWeightOverLivetime. 102 data events. Livetime 227 708 167.68 s | the response (exact per-event weighting), the self-veto, the muon background, the data |
| The release's toolkit (its own environment conda_envs/hese7np: NumPy 1.23, photospline) | the fit (HESE_fit.py: 11 nuisances), detector-systematic splines (det_sys_weights.py: DOM efficiency, head-on efficiency, ice anisotropy) | cross-checks; per-event detector derivatives exported once as data (npz), so our environment never loads its code or pickles |
| IceCubeRates/data/atmospheric_mceq_southpole.npz (task 3) | MCEq SIBYLL-2.3d x H3a at the South Pole: nominal conventional + prompt for all six species, derivatives in 12 Barr parameters and the cosmic-ray index | the atmospheric neutrinos, as NuFlavorSingleSource |
| propagate_sky at N_side 8 (Mauricio: HESE can use coarser pixels) | per-declination-band fluxes, extragalactic and Galactic | the signal; the MC has no azimuth, so the Galactic flux enters averaged over right ascension per band |

The release's own fit (HESE_fit.py priors): cr_delta_gamma -0.05 +- 0.05, nunubar_ratio 1 +- 0.1, anisotropy_scale
1 +- 0.2, conv_norm 1 +- 0.4, epsilon_dom 0.99 +- 0.1, epsilon_head_on 0 +- 0.5, muon_norm 1 +- 0.5, kpi_ratio
1 +- 0.1; astro_norm, astro_gamma and prompt_norm free.

## 3. Steps

1. **Weighting module** (`IceCubeRates/starting_events/hese.py`): the MC reduced once to npz (cuts of the release:
   deposited energy 60 TeV-10 PeV, double cascades 10-1000 m), N = T sum_i w_i Phi_type(E_i, dec_i) with
   sin dec = -cos zenith (step 8's code, moved into the package); histograms in deposited energy x morphology
   (x reconstructed zenith band), the release's binning.
2. **Backgrounds**: atmospheric neutrinos from our MCEq table per event (type, energy, zenith; up-going take the
   mirror production zenith, as for tracks), times the release's conventional self-veto correction (Barr
   derivatives included: they change pi/K production) and the prompt one for the prompt part; the release's muon
   MC with a normalization nuisance. Check: with the release's own fluxes in place of ours, the release's
   best-fit expectation per bin (HESE_fit.py at its best fit) to its precision; ours against it (MCEq vs the
   release's Honda-based conventional and BERSS prompt: differences expected, shown).
3. **Detector systematics** (if decided): per-event derivatives of the weights with respect to DOM efficiency,
   head-on efficiency and anisotropy scale at their nominal values, computed with the release's det_sys_weights.py
   in its environment and exported as arrays; linear nuisances in our fit with the release's priors.
4. **Validation**: (a) the release's best fit (astro 6.37, gamma 2.87, its nuisances) reproduces its expected
   events per bin; (b) our standard-physics expectation vs the 102 events (per morphology, energy, zenith);
   (c) the Galactic component's events (step 8: 2.8 above 60 TeV).
5. **Physics cases and figures**: standard, a dark-portal point and an SI point, with and without the Galactic
   component; per morphology and energy; then the likelihood (binned Poisson, nuisances profiled) for planes.

Note for step 2: the conventional and prompt parts take different self-veto corrections, and the cosmic-ray tilt
moves both, so the table needs the prompt part's own cosmic-ray derivative (the task-3 table has the total's): one
extra MCEq task per month (tools/atmospheric_background.py, prompt derivative of cr_dgamma). The Barr parameters
move pi/K production only (conventional).

## 4. Decisions (Mauricio, 2026-10-14: the first option of each)

- **Atmospheric neutrinos**: our MCEq + Barr (as NuFlavorSingleSource and the tracks) with the release's per-event
  self-veto corrections (Recommended), or the release's own model (Honda-based conventional, BERSS prompt, with
  conv_norm, kpi_ratio, cr_delta_gamma).
- **Detector systematics**: the release's three (DOM efficiency, head-on efficiency, ice anisotropy) as linear
  nuisances from per-event derivatives exported once (Recommended), or none for now.
- **First deliverable**: rates and figures (as for tracks), then the likelihood (Recommended); or straight to the
  likelihood.
