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
