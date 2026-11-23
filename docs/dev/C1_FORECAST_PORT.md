# C1: the forecasts on NuFlavorSingleSource's Plan A detector models (2026-11-11)

Audit finding C1 (docs/dev/AUDIT_2026-11-08_resolution.md): the forecast used the detector models that
NuFlavorSingleSource withdrew on 2026-10-15. They are replaced by its Plan A cards, run unchanged from its final code
(Tycho: /lustre/astro/mbustama/plan_a_final_blocks, commit 56e3215).

**His choice (2026-11-11):** tracks binned in measured energy and declination band; showers and double cascades in
measured energy only, as set 2's present data.

## Steps

1. **Cards per declination** (tools/build_forecast_planA.py, Tycho, 1 core per task): each card's fine output for a
   point source at dec, reduced to K summed over the rings (all events from that direction), the whole-sky background
   and the background within 5 deg of dec (its density there). The site, the Earth's absorption by local zenith and the
   rotation through the day are the card's own. Grid: 3 points per solver declination band (20 bands, equal in sin dec).
   Deposit variants (showers and double cascades) rebuilt as Plan A does (code/make_blocks.py variant_env).
2. **Blocks** (Tycho, Plan A's code): per card, topology and band, a Fine of one ring (K weighted by each point's solid
   angle: per unit flux per sr), assembled with Plan A's blocks.assemble on common measured-energy bins: the response
   per flavor and every nuisance template of Plan A's tables. Backgrounds: tracks, the density near the band times its
   solid angle; showers and double cascades, the whole-sky background split over the bands by solid angle (the
   analysis sums the bands, so the split drops out). Glashow kept as its own piece so it follows the nubar_e flux.
3. **Likelihood** (analysis/future_events.py): events = sum over bands of the response times the model's declination-
   band flux (FlavorModel.band_fluxes; the same at every site, the card handling the site), 10 live years per
   detector, Plan A's priors; profiled as before.
4. **Checks:** a band's response against the card's own point-source events at that declination; the diffuse
   astrophysical events against Plan A's own diffuse background (its 'astro_*' components) per card and topology; the
   expected numbers of events per telescope.

## Simplifications to state

- The cards average nu and nubar (MESE's and DR2's convention); only the Glashow piece follows the nubar_e flux.
- Off the South Pole, the solver's crossing of the Earth is the South Pole's at the same declination (C2). It changes
  nothing for the dark portal and secret interactions (no matter effect); for ultralight L_e - L_alpha it is the open
  part of C2.
- A track's measured direction is taken in the band of its true direction (pointing below 1 deg; bands 6-26 deg wide).

## Runs

- Pilot (job 76732, 8 x 1 core, decs -40 and +40): every card but Gen2 ran (Gen2 needs PyTables: venv_tables). Per
  declination: 24-200 s, GVD 460-911 s. Channels kept per card (Glashow 'nue_gr' apart, except GVD's own shower model).
- Full build (job 76802, 2026-11-11): 1,000 tasks of 1 core, at most 300 at a time; nominal 8 cards x 20 bands (3
  declinations each) and the 6 deposit variants for the 7 cards other than IceCube (whose starting events have no energy
  model). Estimated from the pilot: ~25 core-h nominal, ~100 core-h variants.

## Checks (2026-11-11, Tycho devel; tools/check_*.py in the Tycho work area)

- Whole-sky background of showers and double cascades: a card counts a background while its source is in view, so the
  full-day one is that of an always-in-view declination (the maximum over declinations; IceCube's and ARCA's showers
  are the same at every declination). The sum over the bands of the background densities agrees to 0.90-1.00.
- Our diffuse signal (the per-band responses times Plan A's nominal diffuse flux) against Plan A's own diffuse events
  per flavor: tracks' nu_mu 1.000-1.007; the rest of our tracks is nu_tau -> tau -> mu, which Plan A's background path
  leaves out (our signal counts it). Water detectors' showers and double cascades 0.99-1.00. IceCube's and Gen2's
  starting events 0.93-0.955 for every flavor: a difference between Plan A's own point-source and background paths
  for those cards, flavor-blind, so the free astrophysical normalization absorbs it.

## Code review (2026-11-11, one read-only agent; his request) and fixes

- Fixed: the deposit variants' showers and double cascades took the nominal whole-sky background, so their background
  templates were zero; each variant now takes its own, at the nominal's always-in-view declination (HUNT, Gen2 and
  the others now non-zero; ARCA's stays zero because its published background replaces our model's, as Plan A does).
- Fixed: the always-in-view declination is chosen on the physical backgrounds only, not the derivative templates.
- Blocks reassembled (job c1-assemble, 2026-11-11 ~15:00); previous ones kept in blocks_v1/; MD5s updated in
  IceCubeRates/data/future_planA_MD5SUMS.txt. The running forecast pilot read the previous blocks.
- Noted, not changed: (i) a free nubar share in sky mode with a flavor-surface or spectrum term would evaluate those
  terms at the configured share (no configuration does this; a guard to add); (ii) E11 compares the whole set of
  src/data .npz files, so an extra unrelated file changes every key (conservative); (iii) MESE with double cascades
  alone has a degenerate muon nuisance; (iv) E1 raises at a share of 0.
- Confirmed by the review: units and normalization, the Glashow piece, band and flavor order, the deviance's edge cases,
  the profile and the exact gradient, E1's mixing and derivative, MESE histograms.

## Pilot (jobs 79802, 79902, 93811)

- Majorana block (si_scalar_majorana_mu b17): 6h43 on 32 cores; best-fit chi2 1.64 (the mass prior, below); 95% limits 0.0041 at
  17.8 MeV and 0.058 at 23.7 MeV (set 1: 0.44, 0.58).
- Dirac block (si_scalar_dirac_tau b19): hit the 12 h limit at 86.7% (last points ~35 min per 1%); resumed 2026-11-12 as
  93811 on astro3_long, its stale RUNNING.lock removed.
- Both pilots read the blocks before the code-review fix (blocks_v1); the resume is pinned to blocks_v1
  (NULRI_FORECAST_BLOCKS) so the block stays consistent. **Before the production forecasts, move both pilot outputs
  (campaign_si/proj_future_planA/si_scalar_{majorana_mu/b17,dirac_tau/b19}) aside**, or production resumes from them.
- **Best-fit chi^2 1.64, explained (2026-11-12):** the cosmological mass prior, a half-Gaussian in the sum of masses
  from 0 with 95% at 0.09 eV (sigma = 0.0459 eV). At the smallest lightest mass (1e-6 eV, normal ordering) the sum is
  0.0588 eV, so (0.0588/0.0459)^2 = 1.639, against 1.6393 in fit.json. Every fit carries it (present data too); it
  cancels in Delta chi^2. The remainder (below 0.001) bounds any mismatch between the Asimov events and the model.
- Dirac block resumed (93811): finished in 6h33 more, 18.5 h in total on 32 cores.
- **Forecast configurations for the other families (2026-11-12, tools/make_set_configs.py proj_future_planA from each
  campaign's proj_future):** nulri_set1_final campaign_final (90; the dark portal's 54 are final, its L_e - L_alpha
  ones are superseded by campaign_final_sm), nulri_set1_b campaign_final_b (10; dark portal 6 final), nulri_set1_sm
  campaign_final_sm (36), nulri_set1_ml campaign_ml (56). These copies hold no Plan A blocks: launch with
  NULRI_FORECAST_BLOCKS=/lustre/astro/mbustama/external/nulri_forecast_planA/blocks (MD5s in
  IceCubeRates/data/future_planA_MD5SUMS.txt). Open before the L_e - L_alpha ultralight forecasts: review item 2 (C2).
- Numerics check (review item 9): tools/check_forecast_numerics.py, job 109270 (standard vs doubled grids, edges of
  the Majorana pilot's 17.78 and 23.71 MeV columns, current blocks).
