# Plan: the event likelihoods (HESE and DR2 tracks) in the statistical analysis

Status: agreed 2026-10-14 (Mauricio's two decisions below; the rest are technical defaults, mine to decide and
report). Builds on the first deliverables of tasks 1 and 3 (docs/dev/HESE_RATES_PLAN.md, docs/dev/DR2_TRACKS_PLAN.md)
and on the flavor analysis with the Galactic component (task 2).

## Decisions (Mauricio, 2026-10-14)

1. **Events replace IC26.** IceCube's 2026 spectrum (arXiv:2607.25966) is fitted to cascades (DNNC) and starting
   tracks (ESTES), May 2011-Nov 2023, and northern tracks (NT), June 2010-Nov 2023: largely the events of HESE (2010-
   2017, above 60 TeV) and of DR2's IC79 + IC86 (2010-2022). A fit with HESE and/or DR2 events therefore drops the IC26
   spectrum term. HESE and DR2 combine with each other (they share at most a few starting tracks) and with the 2050
   flavor projection. The task-2 fits keep IC26.
2. **DR2 tracks: IC79 + IC86, reconstructed energy above 3 TeV, a free normalization per configuration.** 85% of the
   livetime (312.8 + 3920.6 of 4963.4 days). IC40 (extra muons within 10 deg of the horizon) and IC59 (its responses
   and events disagree above 3 TeV) are left out. Each configuration's atmospheric expectation has its own
   normalization (10% prior), on top of the shared atmospheric nuisances.

## Design (technical defaults)

**Fluxes per direction.** The likelihoods need the flux per declination band, flavor and kind (nu, nubar), not the
sky average the flavor fits use. A new solver entry, `propagate_sky_basis`, returns in one call the extragalactic
bands for each pure-flavor source (the composition is linear: (x, 1 - x, 0) from the nu_e and nu_mu responses) and the
Galactic bands for each pure-flavor Galactic source (the band responses computed once, as `galactic_allsky` does).
The analysis model gains a sky mode: one such call per propagation point (and per spectral-index node), cached in
memory, interpolated in p_index as now; the flavor fractions come from the same run's sky average (equal to
`propagate_weighted`'s to ~1e-5), so both data sets see one calculation.

**Energy grid.** DR2's reconstructed energies above 3 TeV come from neutrinos down to ~1 TeV, below the flavor fits'
grid (10 TeV-30 PeV, 400 nodes). Event fits extend it downwards with the same spacing: 515 nodes from 0.995 TeV, the
last 400 identical to the old grid. The flavor fractions stay averaged over 10 TeV-30 PeV (`data.flavor_band_eV`):
each node's weight depends only on its energy, so the average is unchanged to rounding.

**Responses, precomputed once.** Both folds are linear in the flux, so each becomes a fixed linear map from the
per-band fluxes to expected counts per analysis bin. Between solver nodes the flux is interpolated linearly in log E
after flattening by E^2.5 (linear in the flux; within 1e-4 of the log-log interpolation for spectral indices 2-3):
- DR2: the fold of IceCubeRates.northern_muons.dr2, factored as (per-band flux -> the effective area's bins)
  then (those bins -> reconstructed energy x declination, with the smearing and the PSF kernel); validated against
  `DR2Tracks.expected`.
- HESE: each Monte Carlo event's weight T w_i spread onto the two nearest energy nodes of its band, flavor and kind,
  summed per analysis bin into a sparse matrix, one per node of each detector systematic.

**Nuisances, profiled inside the event term.** For given physics and astrophysical parameters the event term
minimizes over its own nuisances (warm-started, analytic gradients), so the outer fits see the same parameters as
now. Expected counts per bin:
- HESE: astrophysical (with the release's 'Astro' detector weights) + (1 + conv_norm) [conventional + sum_p theta_p
  d_p] ('Conv' weights) + (1 + theta_prompt) prompt ('Prompt' weights) + muon_norm x muons.
- DR2, per configuration c: astrophysical + (1 + n_c) [nominal + sum_p theta_p d_p + theta_prompt prompt].
- Shared between HESE and DR2: the 12 Barr parameters, cr_dgamma (0.05) and theta_prompt (0.5), the priors of
  NuFlavorSingleSource. HESE only: conv_norm (0.4), muon_norm (0.5), and the detector systematics dom_eff (0.99 +-
  0.1), head_on (0 +- 0.5), anisotropy (1 +- 0.2) -- the release's priors, bounded by the exported grids (2 sigma).
  DR2 only: n_IC79, n_IC86 (0.1).
- Detector systematics: the profile uses a per-bin model -- the release's exact per-event weights at each grid
  node, summed per bin and template; between nodes, a cubic through the four nearest nodes in log (as
  `detector_factor` does per event), multiplied across the three systematics (exact on the grid when one systematic
  moves). It is not exact elsewhere (up to 6% in sparse bins, 0.004 Poisson sigma; -2 ln L off by up to 0.02 at
  fixed nuisances), so the reported -2 ln L is evaluated with the exact per-event product at the profiled nuisances
  (the user's rule: an approximation only where it is nearly exact). The model only steers the profile.
- -2 ln L = 2 sum [mu - n + n ln(n / mu)] + the priors' chi^2; real n allowed (Asimov data).

**Configurations.** `data.events = ["hese", "dr2"]` (either or both) with `data.spectrum` absent (rule 1, enforced);
the HESE files from `NULRI_HESE_MC` and `NULRI_HESE_DETSYS`; DR2's responses and counts from IceCubeRates/data.
`tools/make_event_configs.py` writes the event variants of the task-2 configurations.

## Steps

1. `propagate_sky_basis` in the solver, tested: the bases recombine into `propagate_sky` (extragalactic and Galactic)
   to rounding, and the Galactic sky average equals `galactic_allsky`.
2. DR2 counts (`tools/dr2_counts.py`, on Tycho where the event lists are): every configuration on the full grids, to
   `IceCubeRates/data/dr2_counts.npz`. The DR2 linear response, validated against the fold.
3. HESE: the sparse responses per detector node; the per-bin detector reweighting validated against the per-event
   product; the atmospheric and muon templates.
4. `analysis/events.py`: the event term with its profiled nuisances; the model's sky mode; configuration keys.
5. Validation: (a) Asimov closure (data = the model at a point; the fit returns it, -2 ln L = 0); (b) standard physics
   against the real events: astrophysical normalization and index against IceCube's own fits, nuisance pulls; (c)
   timing of an objective call; (d) the flavor fractions on the extended grid against the old grid.
6. Event planes on Tycho, after task 2 (cores as the standing default).

## Later: future detectors (Mauricio, 2026-10-14)

"Implement future detectors like we have done for NuFlavorSingleSource", with two decisions: (1) in projections the
future detectors' event forecasts (tracks, showers and double cascades per detector) **replace** the projected 2050
flavor surface, which summarizes the same future events; present-day fits keep the events above (or IC26) with the
surface. (2) NuFlavorSingleSource's set: IceCube, IceCube-Gen2 (IceCube's strings and the new ones), KM3NeT/ARCA,
Baikal-GVD, P-ONE, TRIDENT, NEON and HUNT, each at full size for 10 years. Its responses are analytic effective areas
calibrated to published numbers (`code/detectors.py`, `calibrate()`), read at each site's latitude
(`forecast.sky_segments`), with one MCEq table and per-block background normalizations; they become further event
blocks on this machinery once the IceCube likelihoods are validated, checked by reproducing that project's rates for a
reference flux.

## Progress

- Step 1 done: `propagate_sky_basis` (src/dark_portal_solver.py); the bases recombine into `propagate_sky` to 1e-14
  (extragalactic) and 1e-15 (Galactic), and the Galactic sky average weighted by the bands' pixel shares is
  `galactic_allsky` (tests/test_event_likelihood.py).
- Step 2 (counts) done: `tools/dr2_counts.py` on Tycho -> `IceCubeRates/data/dr2_counts.npz` (IC40 36 900, IC59
  107 011, IC79 101 956, IC86 1 397 488 events; northern sky above 3.16 TeV: IC79 2 277, IC86 30 008).
- Step 2b and 3 done (`tools/event_likelihood_checks.py`, output/hese/event_likelihood_checks.json): the DR2 linear
  fold (`northern_muons.dr2.LinearFold`) equals `DR2Tracks.expected` to 2e-15 for E^-2.5 and to 6e-8 for solver skies
  (log-log against the flattened interpolation); 1.5-2.8 ms per configuration. The HESE sparse maps
  (`starting_events.hese.LinearResponse`, 1.2M nonzeros each) equal the per-event weighting to 6e-15 for E^-2.5 and
  2e-7 for a dark-portal sky. The per-bin detector model against the exact per-event product, all three systematics
  within 2 sigma (40 draws): worst 6.2% (Astro), 5.1% (Conv), 2.5% (Prompt) in sparse bins, i.e. 0.004, 0.002 and
  0.0009 Poisson sigma; one systematic between grid values: 0.5%, 5.1%, 1.9%. Against the 60 events at fixed
  nuisances, -2 ln L differs by up to 0.019 (median 0.003): hence the exact evaluation above.
- Step 4 done: `analysis/events.py` (EventLikelihood: HESE and DR2 terms, 21 nuisances profiled with analytic
  gradients and warm starts; the deviance continued smoothly below a floor so trial steps into unphysical nuisances
  stay finite; the Barr parameters and cr_dgamma unbounded, held by their priors -- an L-BFGS-B problem bounded on every
  side takes a full first step, which from a warm start landed 9.6 sigma away and stalled; convergence judged by the
  projected gradient), the model's sky mode (`FlavorModel(sky=True, flavor_band_eV=...)`, `band_fluxes`), the
  configuration keys `data.events` and `data.flavor_band_eV` (exclusive with `data.spectrum`), Asimov event data from
  the configuration's asimov values. Laptop timing: a sky run 3-7 s per spectral-index node; 0.1-0.2 s per objective
  call once cached. Asimov closure exact (-2 ln L = 0 at the truth).

