# Physics review, 2026-11-11 (one agent, read-only; his request)

Scope: which simplifications could move a final limit by more than 0.1 in Delta chi^2 or 2% in the coupling. Read
only docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md, docs/dev/AUDIT_2026-11-08_resolution.md, docs/dev/C1_FORECAST_PORT.md and
paper/main.tex's data section. Sizes are the reviewer's estimates unless marked measured; nothing here is verified by
me yet.

Ranked items (reviewer's words condensed):
1. Set 2's MESE energy-only term is biased in flavor, not only weaker: weaker on e-tau, stronger (not conservative)
   against tau depletion (TS at f_tau = 0: 5.93 vs 5.02), best fit tau-rich (0.19, 0.27, 0.54); cascades ~15% high
   against the release. Up to ~0.9 in Delta chi^2 in flavor-driven planes; small for MeV dips. Check: TS at the
   standard composition from mese_gate.json; one set 2 plane re-evaluated with the cascade response scaled 692.6/795.
2. Forecasts, ultralight L_e - L_alpha: the South Pole's Earth crossing at every site (C2): order 1 in Delta chi^2
   where the Earth's potential sets the flavor; zero for dark portal and secret interactions. Check: swap dec -> -dec
   bands for the off-pole cards at three edge points.
3. Set 1: one event-weighted composition against a likelihood for a composition constant in energy (and the grid's
   dependence on the assumed spectrum, measured: f_tau = 0 at 91.9% BPL vs 99.7% SPL). Above threshold in ultralight
   planes; small in MeV planes. Check: morphology-specific weights and the SPL grid at a few edge points.
4. Antineutrino share: linearity exact physically; numerical test only at shares 0.5, 0.75 (models not recorded).
   The 1:2 bound vs true p-gamma (no nubar_e): ~0.5-1 per half Glashow event near 6.3 PeV. Detector models averaging
   nu/nubar: ~0.4-1 at an edge for ultralight L_e - L_alpha only.
5. Set 1 spectrum segments taken as independent (likely anti-correlated neighbours); break fixed at 24.5 TeV, lower
   index floor 0.5. Order 1 in chi^2; may feed the 0.5-7.5 MeV preference. Check: break +-1 sigma, segment
   correlations -0.2 to -0.4, chi^2 only.
6. m_Zd ~ 1e-3 to 1e-1 eV (mediator decay back): transport switch at g = 1e-4 (6.4% response difference, measured);
   item 24a (vector longitudinal scatterings) still due; relic-population bracket not redone with the new likelihoods.
7. nu_tau -> tau -> mu tracks left out of DR2 (B3): ~0.3-0.5 at DR2-driven edges; forecasts count them.
8. Edge read by linear interpolation in log10 g: ~2% at 0.08 dex spacing, ~14% at 0.25 dex (limits slightly strong).
9. Numerical tolerances judged against present data, not forecast statistics (1e4-1e5 events). Check: three edge
   points of one forecast column with doubled steps, bins, nodes, chords.
10. Set 2 pilot: the plane's minimum 0.49 below the certified global fit. Check: compare the two solutions.

Stated assumptions exceeding the threshold by the docs' numbers (not reopened): mixing fixed (up to 0.18, ~3% in g);
source evolution and normal ordering; new interaction not applied to atmospheric neutrinos.
Below threshold (reviewer): cross sections fixed; SM potential off for DP/SI; redshift-grid normalization; Zeno
factor; Galactic path terms; lightest-mass floor; Dirac helicity transport; IceCube/Gen2 starting-event 0.93-0.955.
Not in the list, unverified: relic annihilation into the mediator for M < m_nu at g > ~1e-5.

## Checks run (2026-11-11, his go: items 8, 3, 5)

- **Item 8, measured (scratch_checks/item8_edges.py on Tycho, every crossing of 3.84 in the finished planes):** the
  edge from linear interpolation in log10 g against a quadratic through the three nearest points. Median |shift| of
  the coupling: SI MeV (0.125 dex grid) 1.5-1.8% (25-44% of crossings over 2%); LRI/DP MeV (0.2 dex) 2.6-2.9%; DP
  ultralight (0.25 dex) 3.2-3.5%; LRI and SI ultralight (0.5 dex) 7-9% (95th percentile ~14%); DP potential planes
  (1 dex) 12%. Above the 2% threshold except in part for SI MeV.
- **Items 3 and 5, submitted:** one block with limit crossings per family, set 1 and three variants: v3a composition
  flux-weighted instead of MESE-event-weighted; v3b IceCube's single-power-law flavor grid; v5 spectrum segments with
  neighbour correlation -0.3 (new optional data.spectrum_correlation, tests/test_spectra.py). Blocks:
  si_scalar_dirac_mu_fx b13 (MeV), si_vector_majorana_universal_ul_fx b16, dp_e_ul b6 (nulri_set1_final),
  le_ltau_ul b7 (nulri_set1_sm). Jobs 82347 (nulri_checks_si), 82348, 82350; set1-mev throttle 55 meanwhile.
- Set 1 pilots: nulri_set1_final and nulri_set1_ml lacked src/data/icecube_cf_2025_table10.json (older code than the
  file); copied (JSON: not in the cache key), resubmitted (82349, 82351).
- **Item 8 fix (his choice: curve now, refine where needed):** analysis/exclusion.py reads Delta chi^2 between
  samples with a monotone cubic (PCHIP) in log10 g (INTERP; 'linear' kept); tests/test_exclusion.py: the logic tests
  pinned to straight lines, a new test that the cubic is closer on a convex column (10 pass). tools/edge_refine_list.py
  lists the columns where the two readings differ by more than 2%: on an old ultralight plane
  (campaign_ml/now_events/le_lmu_ul) 66 of 82 columns, by 19-30%. So the ultralight planes will need couplings added
  near most edges (new solver runs; cost to measure on a pilot once their set 1 planes are final).
- **Items 3 and 5, first block (si_vector_majorana_universal_ul_fx b16, set 1, two masses):** 95% upper limits
  3.415e-3 / 3.496e-3; flux-weighted composition -0.4% / 0.0%; single-power-law grid +0.04% / -0.1%; segments
  correlated at -0.3: -2.3% / -2.9%. Item 3 below threshold here; item 5 just above.
- Set 1 pilots: DP ultralight (nulri_set1_final) finished from the cache in 27 s (its fit at the start point is
  chi^2 176.5; the old campaign's was 175.7: a strongly excluded point, not a code mismatch). The MeV families'
  copy (nulri_set1_ml) failed: analysis/exclusion.py imported src/dp_unitarity.py, which its older src lacks; the
  import is now optional (no boundary, as that code had none); pilot resubmitted (82849).
- **Item 10, settled (read, no runs):** the set 2 pilot's plane minimum (947.706) is a real point, m = 17.8 MeV,
  g = 0.042 (certified fit), where the data mildly prefer an effect; the "global fit" (948.201) is the no-effect start
  point. Delta chi^2 is measured from the plane minimum, as it should be. Not a convergence problem.
- **Item 6(b), check 24a (scratch_checks/check24a.py):** on the old data sets, points with g / M >= 1e-2 eV^-1 not
  excluded: SI vector ultralight 896 of 3,964 (now_ic26), 892 (now_events); DP ultralight 190 of 1,292 (now_ic26), 80
  of 792 (now_events). Many may lie above the unitarity line (excused): the lines are being computed per plane
  (scratch_checks/unitarity_lines.py) before any conclusion. Direction if real: the left-out longitudinal scatterings
  remove flux, so leaving them out makes those limits weaker (conservative), but the accepted approximation's
  condition fails.
- **Item 4(a), settled:** E1's linearity tested with new physics in every family (tests/test_event_likelihood.py
  test_e1_free_share_reproduces_fixed_runs: none, dark_portal and si_scalar_dirac with the Galactic part on,
  si_vector_majorana): the free-share model reproduces runs configured at shares 0.5 and 0.75 to 1e-6 (single
  precision); at the configured share exact without new physics, to 1e-6 with it (the neutrino-only run has
  antineutrino secondaries). 5 passed on Tycho.
- **Item 6(b), check 24a FAILS (below each plane's unitarity line, scratch_checks/check24a_line.py; lines from
  analysis.exclusion.not_allowed_boundary, log10 g = log10 M + c, c = 1.55-2.53):** every tested point with g/M >= 1e-2
  eV^-1 lies below its line, and on the old data sets 896 of 3,964 SI vector ultralight points and 190 of 1,292 DP
  ultralight points there are allowed (e.g. si_vector_majorana_universal_ul b2: M = 1e-26 eV, g = 3.2e-25, Delta chi^2
  0; dp_e_ul b5: M = 1e-10 eV, g = 1.8e-8). The accepted approximation (vector longitudinal scatterings left out,
  item 24(a) of the list) is therefore not justified there: its own condition says to add them (dp_si's scalar
  channels with the Goldstone couplings). Direction: they remove flux (38-73% at g/M = 0.1, all from g/M = 1), so
  today's planes are too permissive near the unitarity line. A hard fix (new transport for those regions).
- **Check 24a, his choice: hatch as not modelled.** analysis/exclusion.py longitudinal_corner (log10 g >= log10 M - 2
  for the vector secret interactions and the dark portal; not the scalar, not L_e - L_alpha, which include the
  Goldstone channels) joins not_modelled_mask; tests/test_exclusion.py (11 pass). **Paper text item (his request,
  2026-11-11): explain in the paper why the band is hatched** (the longitudinal scatterings left out; check 24a; no
  claim there).
