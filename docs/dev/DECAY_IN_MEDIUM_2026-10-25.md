# Neutrino decay in a medium, 2026-10-25

Scope: replace the model in which an eigenstate of the potential decayed "as a whole" by the decay between the
propagation eigenstates, a(p) -> b(z p) + Z_D, everywhere the transport uses it. Derivation: notes/notes.tex,
paragraph "Decay in a medium (2026-10-25 ...)" (after the Galactic component's paragraph; the section "The transport
equation in full" is not touched). Nothing is committed.

## What was wrong

Until today, in a potential, each eigenstate a decayed at the mean of its mass content's vacuum rates,
sum_i T_ia Gamma_i, with vacuum kinematics, and each daughter mass state was split over the eigenstates by its
content (dp_decay.decay_step with use_T; the solver's T_bins and Wd). Three things are wrong with that:

1. The amplitude is a coherent sum over the mass components of the parent and of the daughter, not an incoherent mix
   of mass-state decays. The mix lets the most stable eigenstate (mostly nu_1, a small heavier admixture eps^2) decay
   into itself at eps^2 Gamma_3, which energy conservation forbids. Over a Hubble time Gamma_3 t_H reaches 1e15, so
   even eps^2 ~ 1e-10 slid the flux down in energy.
2. The window is set by the effective masses mu (the eigenvalues of M^2 + sigma S), not the vacuum masses: thresholds
   move, and a potential opens flips nu_i -> nubar_i of the same mass state, closed in vacuum.
3. The daughter is an eigenstate at its own, lower momentum (sigma' = z sigma), with its own eigenvector.

The drift substeps of the dark-portal phase (_drift_pieces, DRIFT_CFL, DRIFT_SUB_MAX, _mean_log_loss) followed the
slide of item 1. They are removed (below).

## The physics, as implemented (for the transport equation's decay term)

Kinds s = nu, nubar. Mass basis: D = diag(m_i) (the Lagrangian masses, lightest included); S' = U^dag S U the potential's
texture; G' = U^dag G U the mediator's coupling (the whole of it, identity part included: the diagonal entries drive
the flips). sigma = 2 E (1+z) x (the potential per unit texture), E the observed energy, E (1+z) the physical one.

- Parent: eigenstate a of kind s at physical energy E_p = E (1+z): mu_a(sigma) and c_a(sigma) the eigenvalues
  (ascending) and eigenvectors of D^2 + sigma S' (nu) or D^2 - sigma S'^* (nubar).
- Daughter: eigenstate b of kind s' at z E_p: mu'_b(z sigma), c'_b(z sigma), of its own kind's matrix at sigma' = z sigma.
  Kept: s' = s (same helicity). Flipped: s' = the other kind (the dark portal is Majorana, dark_portal_solver; dp_medium
  has no Dirac switch).
- Window (light-front energy conservation): k_T^2 = (1 - z) [z mu_a(sigma) - mu'_b(z sigma)] - z M^2 >= 0, 0 < z < 1.
- Rate: dGamma_{a->b}/dz = |A|^2 / (16 pi E_p), with, for G_s = G' (s = nu) or G'^* (s = nubar):
  - kept: |A|^2 = 2 k_T^2 (z + 1/z) / (1-z)^2 |alpha|^2 + |(z M^2 - k_T^2)/(1-z) alpha + (1-z) beta|^2 / (M^2 z),
    alpha = c'^dag G_s c, beta = c'^dag D G_s^* D c;
  - flipped: |A|^2 = (2/z) |alpha + z beta|^2 + k_T^2 |alpha + beta|^2 / (M^2 z),
    alpha = c'^dag D G_s c, beta = c'^dag G_s^* D c.
  The gain in the daughter's energy E' = z E_p is dGamma/dE' = (1/E_p) dGamma/dz. E_p Gamma depends on (E, z) only
  through sigma.
- Closed: a kept a -> a, and any kept a -> b with b above a (Weyl: mu'_b(z sigma)/z >= mu'_b(sigma)). The lowest
  state of a kind decays only by flips.
- No condition on the coupling against the texture: summed over the mediator's polarizations with the in-medium q,
  the formulas above are exact for any S (the review's spinor computation; tests/test_medium.py
  test_rates_are_the_spinor_amplitude, 1e-9, a coupling that does not commute with S included). The first version of
  this note required [G', S'] = 0 and the Ladder refused otherwise; that guard is gone. The potential is not a mass in
  the Goldstone coupling: it enters through the eigenvectors and the kinematics only.
- Valid where zeta_a = 2 E_p Gamma_a / min_b |mu_a - mu_b| << 1 (the width small against the splittings of its kind;
  else H - i Gamma/2 mixes the eigenstates). zeta depends on sigma alone; each run returns its largest value
  (meta['zeta'] of propagate_sky and propagate_sky_basis: cosmological, galactic). Vacuum: zeta ~ 1e-4 (g/M)^2 (eV
  units); potential-dominated: zeta ~ (G' m / M)^2 / 2 pi (the first version said / 8 pi: its flip rate was a factor
  4 low). Measured over the planes: below (the zeta map).
- In vacuum (c = 1) the rates are dp_decay.rest_rates and daughter_spectra exactly (totals to 5e-8, every bin's share
  to 3e-6, the old fits' own accuracy).

## Code

- src/dp_medium.py (new): vertices, the |A|^2 coefficients, the ladder of sigma values (the grid's bin ratio) with
  each rung's eigensystems, the decay plans per rung and channel (totals, own bin, cut bins as exact integrals over the
  parent's spread across its bin, runs of whole bins by moments), and the kernels: bin_rates, deposit, gains,
  decay_sweep (the Galactic segments), widths (the jumps' Zeno damping). Within a daughter bin the daughter's
  eigenvector is held at the bin's centre and its mu' follows the potential's slope across the bin.
- src/dp_sweep.py (rewritten): step_full, the dark-portal phase's step: each bin solved exactly with the jumps (Zeno
  damped by the in-medium widths), the absorption, the injection and the decays; the daughters below deposited after.
- src/dark_portal_solver.py: the decay path through dp_medium.Ladder (cosmology and the Galactic paths); removed
  decay_width_unit, T_table / T_bins / Wd, _drift_pieces, DRIFT_CFL, DRIFT_SUB_MAX, _mean_log_loss.
- src/dp_galactic_path.py: segments' decays by dp_medium.decay_sweep (nearest rung, see below); one sweep per segment
  when no jumps act; DECAY_SKIP = 1e-12 (a segment where nothing decays by more than that skips the decays).
- src/dp_decay.py: docstring; decay_step's use_T removed (ValueError). src/dp_cosmo.py: _build_sigma_mass_table removed.
- Tests: tests/test_medium.py (new), tests/test_dp_sweep.py (new signature), tests/test_decay.py,
  tests/test_galactic_path.py.

## Drift substeps: removed

They followed the slide of the old model's eigenstate self-decay; with the decay between eigenstates nothing slides.
The physical slides left are the jump-decay cycles (a stable state converted into a decaying one, again and again),
which step_full solves per bin exactly over a step; their remaining step dependence is first order in the redshift
step (the sources from above constant over a step) and does not improve with more decay pieces per step (tried: 2
pieces, same result), only with more redshift steps. The phase's cost went from 600-750 s per point (old code, the
same points on Tycho earlier today) to 4-8 s.

## The other processes

- Soft-scattering jumps: they use <b|G|a> at the beam's own sigma. Right: the momentum transfer is soft and the relic's
  recoil (s ~ 2 E m_r >> Delta mu) absorbs the energy the change of eigenstate needs. Only their Zeno damping used the
  old widths; it now uses the in-medium ones.
- Relic mass-change loss (dp_relic_loss): the beam state enters only through (G G^dag)_aa; right as it is.
- Relic composition: unaffected. The relics are non-relativistic, their splittings >> V, and a vector potential on a
  Majorana neutrino at rest is velocity-suppressed. Where the potential could matter (g/M >~ 1e-3 eV^-1) their vacuum
  decays are complete by z = 1 (exponent >~ 1e20); the flips do not change the occupation per mass state.

## Validation

Unit tests (tests/test_medium.py and the updated ones): the vacuum limit is dp_decay's rates and spectra; a -> a and
upward kept decays closed with effective-mass thresholds; the plans match the exact rates (eigenvectors at every
z sigma, fine grid) at three potential-dominated points; the kinds are conjugate (nu <-> nubar with S -> -S^*); the runs
match every bin's exact integral; each decay keeps one neutrino (cosmology and Galaxy, with and without potential); a
small step in vacuum equals dp_decay.decay_step; the Galactic sweep equals the cosmological step at the same sigma.
Tycho: the whole suite, slow tests included, 534 passed, 1 skipped (job 66912640, 32 workers, 29 min). The decay
test files alone, serial, CI's selection: 35 passed in 65 s, none above 34 s (that one,
test_windowed_kernel_matches_brute_force, unchanged). tests/test_speedups.py::test_galactic_emission_regrouped_equals_direct
took 28 min with the Galactic in-medium decays: marked slow (CI was at 20-26 of its 30 minutes). CI itself not run (nothing
pushed).

Tycho, the cosmological phase alone (populations at z = 0; the Sun, Earth and Galaxy left out). Shape = the ratio to the
reference with its median divided out; flavor = the largest flavor-fraction difference; both over 1e13-1e16 eV where the
flux is >= 1e-3 of the standard one. Reference: 3184 redshift steps. Grid = twice the energy bins (at 796 steps).
Times: one thread, 200 steps, nodes shared by 24-32 such runs.

| Point (M, g in eV; m_l) | flux/std at 1e14 eV | zeta | 200 steps shape / flavor | 400 | 800 | grid x2 shape | vs dp_si | time |
|---|---|---|---|---|---|---|---|---|
| le_lmu_ul (1e-20, 1e-20; 0.1) | 0.945 | 1.6e-3 | 3.2e-6 / 1.2e-7 | - | 1.3e-6 / 2.5e-8 | 5.2e-4 | - | 6.0 s |
| le_lmu_ul (1e-16, 3e-17; 0.01) | 0.639 | 1.3e-5 | 2.1e-5 / 1.1e-7 | - | 1.4e-6 / 2.1e-8 | 5.4e-4 | - | 4.3 s |
| le_lmu_ul (1e-12, 1e-11; 0.01) | 0.396 | 1.5e-2 | 7.0e-5 / 2.5e-11 | 2.8e-5 | 2.8e-5 / 3.6e-12 | 5.2e-4 | - | 6.9 s |
| le_lmu_ul (1e-4, 1e-6; 1e-3) | 1.3e-4 (drained) | 2.8e-9 | no flux >= 1e-3 std | | | | | 6.5 s |
| dp_mu_ul (1e-4, 1e-6; 1e-3) | 3.0e-3 | 2.4e-9 | 2.0e-3 / 3.3e-3 | 8.2e-4 / 1.3e-3 | 3.0e-4 / 5.0e-4 | - | 8.2e-4 / 1.3e-3 | 5.3 s |
| dp_mu_ul (1e-6, 1e-6; 1e-6) | 3.0e-7 (drained) | 2.4e-5 | no flux >= 1e-3 std | | | | | 7.2 s |
| dp_e_ul (1e-8, 1e-8; 0.1) | 1.67e-2 | 4.4e-7 | 3.7e-3 / 3.6e-3 | 1.6e-3 / 1.7e-3 | 5.4e-4 / 7.1e-4 | 6.9e-4 | 1.7e-3 / 1.7e-3 | 4.2 s |
| le_lmu_ul (1e-8, 1e-8; 0.1) | 8.1e-3 | 1.6e-3 | 3.1e-3 / 1.4e-3 | 1.4e-3 / 6.6e-4 | 5.3e-4 / 2.8e-4 | 2.6e-3 | - | 7.8 s |
| le_ltau_ul (1e-8, 1e-8; 0.1) | 1.4e-3 | 1.6e-3 | 6.1e-3 / 1.9e-3 | not run | 1.2e-3 / 3.8e-4 | - | - | 6.7 s |
| le_lmu_ul (1e-7, 1e-8; 0.1) | 0.41 | 3.0e-6 | 1.0e-4 / 2.9e-3 | - | 1.6e-5 / 3.0e-4 | - | - | 4.2 s |
| le_lmu_ul (1e-9, 1e-9; 0.03) | 0.095 | 3.0e-4 | 1.3e-3 / 1.9e-6 | - | 8.3e-5 / 3.7e-7 | - | - | 6.7 s |
| le_lmu_ul (1e-14, 1e-14; 1e-6) | 0.29 | 1.2e-4 | 1.3e-4 / 6.2e-9 | - | 8.4e-6 / 1.3e-9 | - | - | 5.2 s |
| le_lmu_ul (1e-18, 1e-18; 0.05) | 0.86 | 5.5e-4 | 1.1e-5 / 3.2e-8 | - | 1.1e-6 / 6.6e-9 | - | - | 7.8 s |
| le_ltau_ul (1e-16, 1e-16; 0.01) | 0.65 | 1.7e-4 | 5.0e-5 / 5.6e-9 | - | 3.4e-6 / 1.2e-9 | - | - | 5.2 s |
| le_lmu_ul (1e-10, 1e-11; 0.1) | 0.94 | 3.0e-6 | 1.2e-7 / 2.8e-5 | - | 1.2e-8 / 2.3e-6 | - | - | 6.2 s |
| le_lmu_ul (1e-8, 1e-7; 0.01) | 3.4e-8 (drained) | 1.5e-2 | no flux >= 1e-3 std | | | | | 7.7 s |

"vs dp_si": dp_si at 200 steps, where the potential is negligible (the dp points; production uses dp_si there). The le
(1e-8, 1e-8) grid entry compares 3184 steps at x1 and x2 bins (2.4e-3 there).

Reading: wherever the potential sets the eigenstates the decays converge at 200 steps to <= 1.3e-4 in shape and far
better in flavor; the grid dependence is the grid's own (5e-4, the same without decays). Where decays and jumps cycle in
nearly vacuum eigenstates and drain the flux to 1e-3-1e-2 of the standard one, 200 steps leave 3-6e-3 in shape, 400
steps <= 1.7e-3 (dp_e, le_lmu), 800 steps <= 1.2e-3.

### The Galactic component

Tycho, galactic_allsky alone (N_side 16 working map, 3072 lines of sight x 97 segments, 439 energy bins), one thread.
Same metrics as above, over the sky-summed Galactic flux; the composition here is over the states at the Sun (what
galactic_allsky returns), not flavors.

| Point | Galactic flux / std at 1e14 eV, states (new) | old code | time new | time old | nearest rung vs two rungs | n_split 2 vs 1 |
|---|---|---|---|---|---|---|
| le_lmu_ul (1e-16, 3e-17; 0.01) | 0.571, (0.51, 0.49, 0) | 0.033, (0.99, 0.007, 0.005) | 1411 s | 784 s | 3.7e-5 shape, 3.0e-5 composition (two rungs: 2447 s) | 1.7e-8 shape, 6.8e-9 composition (3268 s) |
| le_lmu_ul (1e-12, 1e-11; 0.01) | 0.580, (0.50, 0.50, 0) | 1.2e-4, (0.96, 0.01, 0.03) | 1773 s | 746 s | 7.7e-7 shape, 4.2e-7 composition | 2.8e-11 shape, 1.4e-11 composition (2288 s) |
| le_lmu_ul (1e-20, 1e-20; 0.1) | 0.584, (0.5, 0.5, 0) | - | 1773 s | - | 1.0e-7 shape, 3.6e-7 composition | - |
| le_lmu_ul (1e-8, 1e-8; 0.1), jumps and decays cycling | 0.584, (0.5, 0.5, 0) (n_split 2) | pending (job 66913465) | pending | pending | - | pending (n_split 2: 5975 s) |
| le_lmu_ul (1e-14, 1e-14; 1e-6) | 0.560, (0.49, 0.51, 0) | - | 1920 s | - | - | - |
| dp_mu_ul (1e-4, 1e-6; 1e-3) | 0.844 (vacuum path) | - | 82.5 s | - | - | 1.9e-6 shape, 7.8e-7 composition |

The new times ran six to a 32-core node, the old two on two cores. The Galactic decays cost about twice the old code's
(the old one was wrong: it drained the Galactic flux of these points to 1e-4-3e-2 of the standard one, nearly all in
one state, by the same self-decay as in the cosmology; the new one keeps the standard run's state composition here,
(0.5, 0.5, 0)). Where the time
goes (micro-benchmark, 575 bins, one segment 4.4 ms): the runs' deposits 55%, the gains 30%, the rest 15%. Interpolating
between two rungs instead of the nearest one costs 1.7x and changes the flux by 4e-5, so the Galaxy keeps the nearest.

## Remaining approximations, with sizes

1. The daughter's eigenvector held at its bin's centre (eigenvalue linear across the bin): second order in the bin
   width. Against the exact formula (eigenvectors at every z sigma) the widths agree to 2e-4 and every bin's share to
   1e-8 at the validation points (tests/test_medium.py, test_plans_match_the_exact_rates).
2. The ladder: each step interpolates linearly in log sigma between the two rungs around it (cosmology); the Galaxy
   uses the nearest rung (sigma within the grid's half-bin ratio, 1% at 115 bins per decade). Both are inside the
   "grid x2" column for the cosmology (5e-4); the nearest rung against two rungs in the Galaxy: 4e-5.
3. The parent spread uniformly over its bin (as dp_decay.window_plan): inside the "grid x2" column.
4. Fermi's golden rule between eigenstates, zeta << 1: the largest at the validation points 1.5e-2; corrections of
   order zeta. zeta is recorded per run; it exceeds 0.1 only for g/M >~ 30 eV^-1 in vacuum, and where the potential
   dominates for Goldstone Yukawas G' m / M >~ 1.6, near the perturbative-unitarity boundary.
5. [G, V] = 0: exact for every model here below 1 keV (checked in the code).
6. The redshift step (cycling points): above.

## Decays where the transport has none

Measured with the exact widths (scratchpad env_scan.py; le_lmu_ul, E = 3e16 eV, g/M <= 60 eV^-1, m_l 1e-6, 0.01,
0.1 eV, one point per decade in M and g). theta = sigma / Delta m21^2 of the same potential (>~ 1e-3: its coherent
effects already change the flux strongly). Decay depth tau > 1e-3 counts as "matters".

- Cosmology above the vacuum band (M > m_3 - m_1, where the code has no decays): only at (M, g) = (0.1, 0.1) eV,
  theta = 843, never for the lowest states.
- The extragalactic beam crossing the Galaxy (10 kpc; dp_transit treats the crossing as coherent): some state for
  g/M >= 1e-3 (theta >= 5e-6); the lowest states (what an in-band beam arrives as, since its vacuum decays are complete)
  only for theta >= 3.7 (g/M >= 0.1). Out of the band: theta >= 2.6e3.
- The Sun (1 AU at 1 AU): some state only for theta >= 840 (M <= 1e-17 eV); the lowest states never.
- The Earth (its radius at its surface): theta >= 3.4e5 (M <= 1e-12 eV); the lowest states never.

So decays from the Sun on matter only where the same potential's coherent effects are already overwhelming. The Galaxy
crossing of the extragalactic beam does matter (tools/galaxy_crossing.py, le_lmu and le_ltau, m_l = 0.01 eV, four
lines of sight, at the detector over 1e13-1e16 eV): g/M = 0.1 eV^-1 moves the flux by 13-41% at the top, its shape by
11-34% and a flavor fraction by up to 0.5; g/M = 1 by 1-17%, 1-16%, 0.15-0.19; g/M = 10 by 1e-8. It is in the
transport since 2026-10-25: dark_portal_solver._flux_runs carries the z = 0 populations of all the runs of a point (the
columns of one sweep) along every line of sight with the Galactic component's processes (dp_galactic_path.
crossing_populations), averages them per declination band and hands each band to the Sun/Earth handover through its
responses to the three unit inputs; the whole sky is the bands weighted by their cos(zenith) widths.

## Found elsewhere

- The solver's M_rot holds the mass-squared differences (U diag(0, dm21, dm31) U^dag), not the masses squared; mu is
  then offset by m_1^2 from <D^2>. Ladder takes the offset explicitly (moff). Any other code that reads absolute
  eigenvalues from those tables would be offset the same way (none found).
- 'combined' mode (below 1 keV, g >= 1e-4, a potential): the flux is rescaled by dp_si's, whose decays are vacuum
  decays between mass states; the dark-portal phase's are in-medium. Inconsistent, but at such couplings the flux is
  drained (le (1e-4, 1e-6): 1.3e-4 of the standard one).
- The extragalactic beam's decays during the Galaxy crossing are not modelled (above).

## Decisions for the owner

1. Redshift steps at the cycling points (decays and jumps cycling in nearly vacuum eigenstates, flux 1e-3-1e-2 of the
   standard one): 200 steps leave up to 6.1e-3 in shape (target 3e-3). Options: n_z_bins 400 for the decay band (cost
   x2 on those points, measured <= 1.7e-3 where run), or accept. Recommendation: 400 on the points where decays act.
2. The Galactic component's cost with the in-medium decays: 1400-1900 s per point on one thread where the Galaxy's
   potential matters (about twice the old code's 750-780 s, which was wrong), 83 s on the vacuum path. The sweep is
   parallel over lines of sight. Options: accept; or a coarser working map for the potential path (N_side 8: 4x
   fewer lines of sight, accuracy not measured); or optimize the runs' deposits further (55% of the
   time; gain not measured). Recommendation: accept.


## Follow-up after the review (2026-10-25, afternoon)

What changed (each with its check):
- Ladder room (review finding 1): the plans' room per channel grows on demand (dp_medium.Ladder; NEXP/NRUN are only the
  starting sizes). The review's failing ladder (le_ltau_ul 0.0316 eV, 1e-5, m_l 1e-3, Galactic) builds with room 48
  (26 needed); tests/test_medium.py test_ladder_room_grows checks its widths against the spinor reference (1e-3).
  tools/preflight_ladders.py builds every ladder of the final grids as production does: 15,501 of its 58,225 units
  (random order, all five planes, 11 lightest masses) built with 0 failures; rooms grew to 48 and 96 in places.
- zeta (finding 2): returned per run in meta['zeta'] (cosmological, galactic) of propagate_sky(_basis); the
  pure-vacuum value where dp_si takes the decays. The coordinator's pilot: zeta ~ 1e-4 at the limits.
- The Galactic segments (finding 4): every energy bin exact over each distance bin with the jumps, the decays and the
  emission as a source (dp_sweep.step_columns: step_full at z = 0 for several columns sharing each bin's
  exponentials); the emission spread over its distance bin. Tests: the vacuum path equals the sweep (1e-9), each
  column equals a sweep of its own (1e-12), the jumps keep the number.
- The extragalactic beam's crossing (finding 3): measured (tools/galaxy_crossing.py; above, plus g/M = 1e-4: 1e-6,
  1e-3: <= 1.7e-4, 1e-2: <= 1.3e-2 toward the centre), then put in the transport: dark_portal_solver._flux_runs carries
  the z = 0 populations of all the runs of a point (one sweep, the runs as columns) along every line of sight with the
  Galactic component's processes, averages them per declination band, and hands each band to the Sun/Earth handover
  through its responses to the three unit inputs; the whole sky is the bands weighted by their cos(zenith) widths.
  Tests: an identity crossing reproduces the handover's bands to 1e-10; the decays alone keep the number (1e-10).
  Why the effect is largest near g/M ~ 0.1: the cosmology leaves each kind in its lowest state (nu mu-like, nubar
  e-like on le_lmu, split by m2_mumu - m2_ee ~ 1e-3 eV^2); the Galaxy's potential (theta ~ 2e3 at g/M = 0.1) opens
  the flip between them, too slow over 30 kpc at small g/M (rate ~ g^2), already done in the cosmology at g/M >~ 10.
- The paths' sky (cost): the Galactic component's paths and the crossing follow N_side 4 lines of sight
  (GALACTIC_PATH_NSIDE, CROSSING_PATH_NSIDE; each carries its N_side 16 pixels' emission at their mean density). Against
  N_side 8: the crossing 3.3e-4 in total and 1e-5 in composition, the Galactic component 5.5% (inside its
  normalization's 20% prior; Mauricio's decision). The processes as a factor on the fine sky (GALACTIC_PROCESS_FACTOR,
  kept off) was worse: 12-14% on the sky mean, 22-28% on the bands.
- Grid extension (finding 5): the Galactic part is extended above the grid also at points routed to 'si'.
- Injection (finding 8): the step's exact emission integral for every dark-portal run below 1 keV (MeV unchanged).
- Absolute checks (finding 6): tools/validate_transport.py check M (le planes): the run's own ladder and steps, decays
  alone: the number kept on a grid extended 12 decades down (1e-13 to 9e-11 at three points) and the top bin's highest
  state's exponent against independently integrated in-medium widths (<= 2.5e-4).
- Findings 9-11: no [G, S] condition (the spinor test, a non-commuting coupling included; the guard removed); the
  potential-dominated flip rate is E Gamma ~ |G'_ii m_i / M|^2 sigma |S'_ii| / 4 pi (4x the first note); ladder_x0
  checks the ladder's top; window tests back to exact references (1e-10, 1e-12); a step_full test with jumps, decays and
  Zeno widths against expm of the 6 x 6 system; 'combined' with decays (slow test).
- The 6 x 6 exponentials: dp_sweep._phi_rate (Taylor at |X| <= 1/2 with an adaptive order, doublings with the rate
  blocks kept) agrees with mpmath to 2e-12; dp_si._phi_functions (unchanged, used by dp_si) to 1e-7 on the same
  matrices.

Cost (one model.fluxes at le_lmu_ul (1e-12, 1e-13), one thread, Galaxy off): 60.6 s, 12 cosmological runs, 4776
step_full calls (29.9 s of a 33.9 s profiled call); without the exponentials 35.7 s, without the deposits 27.4 s
(timing switches only). The Galactic component with N_side 4 lines of sight: 134-167 s with 8 threads including
compilation (galactic_allsky; the factor variant 174-365 s on one thread; plain N_side 4 on one thread not measured;
cached per point by the analysis). The crossing at N_side 4 adds to every extragalactic call (propagate_weighted with
the crossing, 8 threads: 262 s at N_side 4 including compilation, 23.5 s at N_side 2 compiled; N_side 4 compiled on
one thread not measured). The cost work continues in
the main session.
