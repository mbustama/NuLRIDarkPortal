# Simplifications and assumptions

Started 2026-10-17 at Mauricio's request. One entry per simplification or assumption in the analysis: what it is,
why it is acceptable (or not yet), how large its effect is where known, and its status. Keep it current: add an entry
whenever a new one is made, and update the status when one is quantified or removed.

Status words: **accepted** (deliberate, effect understood to be small or conservative), **to quantify** (believed
small, not yet measured), **open** (decided to remove or change, not yet done).

## Propagation

1. **Absorption in the Earth applied after the flavor conversion.** *Bounded* (2026-11-08; his call "Bound is enough"):
   it matters only where flavor changes inside the Earth (the L_e - L_alpha potential) and only for the flavor-dependent
   parts of absorption, which concern at most 3e-4 (up-going nubar_e at 4-10 PeV) and 10% (up-going nu_tau above
   100 TeV) of HESE's astrophysical events (HESE Monte Carlo at the broken power law). (2026-10-17)
   The solver propagates only flavor through the Earth (the new force's potential), never removing neutrinos; the
   standard absorption comes from the detector responses (HESE's Monte Carlo propagates each event through the
   Earth; DR2's effective areas include absorption; the future detectors multiply by NuFlavorSingleSource's survival
   factor). So the flavors are converted over the whole path, then absorbed with the flavor mix that arrives. This is
   exact where absorption is the same for every flavor, which holds for CC + NC absorption over most of the range;
   it is approximate where absorption depends on flavor: anti-electron neutrinos near the Glashow resonance (6.3 PeV)
   and tau-neutrino regeneration, where the flavor mix along the path matters. A proper treatment propagates flavor
   and absorption together.

2. **No cross-section variation.** *Accepted.* (Mauricio, 2026-10-12 and 2026-10-15)
   Cross sections are fixed at the values in the public responses. Varying them properly needs the Earth propagation
   with regeneration and effective areas rebuilt with the varied cross sections, which we do not do.

3. **Standard matter effects off in the Earth** (`use_SM_potential: false`). *Wrong for L_e - L_alpha; fixed in the
   code 2026-11-08* (Mauricio: "It is incorrect"). The justification written here on 2026-10-17 (mine, never put to
   him) held only without a new potential in the Earth: on an arriving mixture of mass states the SM potential changes
   the flavor by <= 1.4e-3 at 100 GeV and ~1e-5 above 1 TeV (core-crossing, scratchpad sm_earth_check.py). With the
   L_e - L_alpha potential the arriving states are not mass states and both potentials act together in the Earth.
   Both are flavor-diagonal, so within a chord only the vacuum term changes flavor, by at most ~Delta m^2_31 L / E
   (0.016 at 10 TeV, 0.16 at 1 TeV, order 1 at 100 GeV through the core). The code put one constant density on every
   chord when asked; since 2026-11-08 standard_mixing.SM_potential_profile = "prem" (default) takes PREM's electron
   density step by step (dp_spatial._sm_electron_seq; tests/test_sm_earth_potential.py: an independent evolution to
   1e-9), "constant" the code before. Every run so far has it off: the dark portal, the secret interactions and every
   MeV plane are unaffected (no new potential in the Earth); the L_e - L_alpha planes below 1 keV are to be rerun with
   it on.

4. **New physics acts on the astrophysical flux only, not on atmospheric neutrinos.** *Accepted (conservative).*
   The atmospheric neutrinos crossing the Earth would also feel the new potential; leaving that out can only weaken
   the bounds.

5. **No t-channel scattering on the relic neutrinos below 1 keV.** *Fixed (2026-10-22, merged 895e760); the
   affected points rerun 2026-10-23.* Validation (tools/validate_relic_scattering.py): every stored solution flagged by
   what the new code changes there; 4,080 sampled old against new -- unflagged points Delta chi^2 exactly 0 (all MeV
   planes), the 1e-16 potential planes' soft-scattering-only points <= 0.07 (not rerun). Rerun
   (tools/rerun_relic_scattering.py, rerun_rs.*.jsonl, which replace the stored values): 151,546 refits (dark portal
   ultralight below 1 keV, the 1e-8 potential planes' flagged points, the top rows of every potential plane below
   1 keV up to eps' = 1e-2, the L_e-L_mu/L_e-L_tau ultralight planes' flagged points), all successful; above
   eps' (or g) = 1e-2 the 82,352 transport points were marked excluded where their column's refitted 1e-2 point is
   excluded (the energy losses only grow with the coupling), and the 273 elsewhere refitted. Then the refinement
   along the new edges (refine_grid tasks 3000-3699). Until then the dark-portal phase (below 1 keV) had the resonance only,
   and the split variants stopped at g_D sum|U|^2 = 1e-8 on the grounds that the t channel matters above ~1e-7; that
   reckoning left out the Z mixing's coupling to every neutrino, kappa = eps' g / c_W = 0.74 eps', which in the
   potential planes reaches 0.74. Now (notes, 'Soft scatterings on the relics'):
   - the soft scatterings (the t channel's forward peak, sigma ~ |G|^4 / 4 pi M_t^2) move neutrinos between
     eigenstates at their own energy: jumps in the dark-portal phase (dp_cosmo._soft_jumps), Zeno-suppressed where
     the coherences are damped faster than they turn. Fast, they equalize the flavors (1/3 each); in the 1e-8
     potential planes below ~1e-6 eV they erase the cosmological potential's effect (toy and solver).
   - M_t^2 = M^2 + m_D^2, m_D the relics' Debye mass (Majorana: their charge is v-suppressed). The spin (magnetic)
     part of the exchange is not statically screened; its cutoff is dynamical, at a comparable or smaller scale. The
     results do not depend on it where it matters: there the rates exceed the expansion rate by ~1e10.
   - the hard scatterings (energy losses, mass-independent below 1 keV, 1.5% of the flux at couplings 1e-3, 40% at
     3e-3) go through dp_si's transport, below 1 keV only for couplings above SI_HARD_MIN_G = 1e-4 (~1e-6 losses
     there); with a potential too ('combined'), the dark-portal phase's composition times dp_si's flux per energy.
     Exact when the hard scatterings do not depend on the flavor (kappa I, which dominates them in the potential
     planes); against a reference transport with every scattered neutrino projected exactly on the new eigenbasis
     (scratch ref_combined.py): totals to 3 digits, flavors to 1e-3 near the edges (6e-3 where 2% of the flux
     survives).
   - left out: the scatterings during the crossing of the Galaxy (where they could act, M_t < ~3e-11 eV, the
     cosmological phase has already equalized, or the potential suppresses them); the u-channel interference
     (~ln(1/mu) against 1/mu); the hard scatterings of a non-universal coupling where the potential matters (only
     where the potential alone already excludes).
   Effect: the top edge of every potential plane below 1 keV, from eps' ~5e-3 to ~2e-3; the 1e-8 potential planes'
   diagonal edges below ~1e-6 eV; the dark portal's ultralight planes (eps' = 0) below 1 keV; possibly the L_e-L_mu,
   L_e-L_tau ultralight planes near 1 eV-1 keV (to be measured).
   The projections' Asimov data are the Standard Model's (Mauricio, 2026-10-22, option (a)): generated with the
   scatterings left out (dark_portal.relic_scattering false; analysis/config.objective), since at a split such as
   g_D sum|U|^2 = 1e-8 they act even at eps' -> 0 (each mass column is read against its own best fit). The two
   accuracy fixes of the thermal kernels (8-point angular quadrature far below the resonance, the sigma = m split)
   stay with Dirac relics only: for Majorana ones they moved the projected MeV planes' chi^2 by ~0.1 (median), within
   the 2e-4 agreement with the reference either way.

6. **Galactic component propagated adiabatically** from its emission points to the Sun (Landau-Zener parameter
   >= 1.5e5 wherever checked; `tools/galactic_adiabaticity.py`). *Accepted.*

6a. **MeV transport: the produced neutrinos' spectra, the profiles' totals, and the 1/s and pair factors with
   the relics at rest** (only the resonance factors averaged over the relics' thermal momenta since 2026-10-14).
   *Fixed behind a switch* (2026-10-18; notes Sec. "Relic motion in the collision transport"):
   `dark_portal.thermal_products: true` (dp_si `thermal="full"`) averages every collision term -- rates, in-bin
   terms, produced spectra -- over the relics' thermal momenta. Off by default; the campaign configs do not set it
   yet. Against the reference (`tools/thermal_reference.py`, converged to 2e-5) at the analysis grid, two MeV
   points and lightest masses 1e-6, 1e-3, 0.03 eV: within 2e-4 in the flux, where the resonance-only treatment is
   1-17% off (a factor ~100 inside one absorption dip). A transport costs 1.1-2 times one without it.

6b. **MeV transport: the resonance's thermal averaging switches on and off abruptly** (thermal width against two
   redshift steps; a bin within the broadened resonance's reach), so chi^2 jumps as m_lightest varies, and its two
   s-t interference profiles keep their at-rest totals; its momentum quadrature is also up to 1.3e-2 off where the
   resonance sits at sigma ~ m for a relativistic relic (m/T ~ 0.03; 2e-3 at 1e-6 eV). *Fixed with 6a's switch*
   (2026-10-18): no switches, the interference's profiles averaged too, and momentum panels at the relic's mass
   (`relic_state_rates_analytic(..., fine_m=True)`, within 3e-9 of direct quadrature). *Open* without the switch.

## Sources and spectrum

7. **The extragalactic spectrum is IceCube's 2026 broken power law with the break fixed** at 10^4.39 GeV, both
   indices free (the lower one in [0.5, 2.5], while IceCube's 1-sigma range reaches 0.01). *Accepted.* (2026-10-15;
   `docs/dev/BROKEN_POWER_LAW.md`) The two pieces are split at the energy today, each a population with its own
   redshift weighting.

8. **Sources up to z = 4, their density tracking the star-formation rate** (dp_cosmo.calc_R_source: (1+z)^3.4 to z = 1,
   (1+z)^-0.3 beyond; no evolution added to it, rho_evol = 0; wording corrected 2026-11-08). *Accepted (fixed choice).*

9. **Source flavor composition** at pion decay (1/3, 2/3, 0) in the nominal planes; fitted as (x, 1 - x, 0) in the
   `_fx` twins, extragalactic and Galactic independently. No tau neutrinos at the sources. *Accepted.*
   The composition is the same at every energy. Consequence (2026-10-21): with it fitted, the ultralight planes
   exclude a thin band (~0.5 decade) along the edge of the pion-decay region, where the flavor composition at Earth
   steps with energy and no energy-independent composition can follow; sources whose composition changes with energy
   (muon damping) could absorb it. Left out of the exclusion figures, described in the notes ('Thin excluded bands
   with fitted source compositions').

10. **Neutrinos and antineutrinos 1:1 at the sources.** *Accepted.*

## Standard neutrino physics

11. **Mixing parameters fixed in the likelihood**, at Table II's central values (arXiv:2606.17132), normal
    ordering, for every data set. *Accepted.* (Mauricio, 2026-10-18: "Fix mixing parameters. Make a note that we will
    simply estimate the effect of them differently, but not vary them in the likelihood.") Their effect on the limits
    is estimated separately, outside the likelihood; the method is not yet chosen (a first look: item 11g). History:
    2026-10-12 decided to vary them; 2026-10-18 the fixed-mixing campaign was cancelled to vary them (present-day data
    under NuFIT 6.1's pairwise chi^2, item 11a; projections under Table II's Gaussians), templates (11d) and then an
    exact profile at every point (branch agent/exact-profile-v2) were built and piloted, and dropped the same day: too
    costly for an effect expected to move the contours by a few percent in coupling.
    The lightest neutrino mass stays varied: a half-Gaussian penalty on the mass sum bounded at zero, Table II's
    0.12 eV now and 0.09 eV for projections at 95% C.L. as there and in its sources: sigma = limit / 1.960 (Mauricio,
    2026-10-18).

11a. **NuFIT 6.1's chi^2 as the sum of three two-dimensional projections** (sin^2 th13 x sin^2 th12, sin^2 th23 x
    delta_CP, Delta m^2_21 x Delta m^2_31), each profiled by NuFIT over the other parameters. *Not used since
    2026-10-18* (the mixing is fixed, item 11); kept in `analysis/nufit.py`. Was accepted (Mauricio, 2026-10-18). Summing projections drops the correlations between parameters in different
    tables; NuFIT publishes no full six-dimensional table. Interpolated by bicubic splines, shifted up by their
    undershoot below zero (at most 0.05 in chi^2, a constant).

11b. **Normal ordering only.** *Accepted.* (Mauricio, 2026-10-18) NuFIT 6.1 disfavors the inverted ordering by
    Delta chi^2 = 5.9, and the mass-sum limits disfavor it further.

11c. **Majorana phases not varied.** *Accepted.* (Mauricio, 2026-10-18: they don't matter to our observables.)

11d. **The mixing parameters varied through templates, not a solver run at every value.** *Not used since
    2026-10-18* (the mixing is fixed, item 11; the templates' chi^2 errors reached ~2 for projections).
    (`analysis/templates.py`, docs/dev/MIXING_VARIATION.md) At each point, runs with one parameter moved
    to a few values, at the central spectral-index node; the flux in between is interpolated parameter by parameter
    (no cross terms) and the relative correction is taken to be the same at every index. Measured on the test
    configuration: the index transfer below 1e-6 of each bin's flux, s12 linear 1e-5, delta_CP over a whole turn 1e-4
    with 7 runs; the validation at the campaign's resolution (chi^2 against exact runs) is running.

11e. **The exact profile over m_lightest** (`analysis/exact_profile.py`, docs/dev/EXACT_PROFILE.md; 2026-10-18).
    *Accepted, with two stated limits.* Built for the mixing and m_lightest together; since the mixing is fixed
    (item 11) it runs over m_lightest alone (its parameters are those of the seven left free), so NuFIT's octant
    probe below no longer applies. Every reported chi^2 is the solver's at the reported values, with
    a per-point certificate of local optimality (`stop` = 1e-3 since 2026-10-18, 1e-4 before: Tycho 66462115 moved
    the final chi^2 by at most 5.9e-4 between the two). Its global part rests on two choices: NuFIT's other octant of
    theta23 is probed by one exact evaluation and profiled when within 1.0 of the best; at the MeV planes m_lightest
    is scanned on 1e-6 eV and 1e-4 ... 0.1 eV (three per decade since 2026-10-18, five before) before the local
    profiles (a basin narrower than ~0.3 decade could be missed; none is expected: an absorption feature spans a
    factor 5 in energy). m_lightest >= 1e-6 eV (the MeV transport treats an exactly massless relic as inert; the
    mass sum changes by 1e-6 eV). Since 2026-10-18 (v2): (i) a point whose solver runs at its chain's anchor equal the
    anchor's to 1e-6 per element takes the anchor's certificate after one evaluation (the error is second order in
    that difference, below 1e-6 in chi^2; checked by `inherit_check`); at the MeV planes only once a scan of
    m_lightest at a coupling at least the point's equals the no-new-physics scan to 1e-6 at every mass (absorption
    grows with the coupling: assumed monotonic); (ii) a certificate may use forward differences (step 0.05 widths)
    when the curvature measured at the chain's last central certificate, doubled, bounds their truncation (both
    surrogates predict no decrease beyond `stop`, and differ by at most 1e-4): it assumes the curvature of log F
    changes by less than a factor 2 between neighbouring points within 1 width.

6c. **Secret interactions: relics at rest; a Dirac relic's helicity.** The secret-interaction planes so far take
   the relics at rest (as 2107.13568), and a Dirac relic as unpolarized (half of it couples to the vector; the
   scalar sees both halves). *Fixed behind a switch* (`secret_interactions.thermal_products: true`, 2026-10-21;
   notes Sec. "Dirac relics in motion: their helicity"). Majorana relics: as 6a (scalar within 1.3e-4 of the
   reference at the analysis grid). Dirac relics keep their helicity, so the beam meets their left- and
   right-chiral components in shares (1 +- pi)/2, pi = (E_p sigma - m^2) / (p sigma); carried exactly (the
   brute force: the polarized relic is that mixture of massless chiral relics, up to m^2/s). The vector couples to
   the left-chiral share only; the scalar's s channel needs the right-chiral one (tiny on a relativistic relic),
   and the polarized background couples the visible and sterile neutrinos, which are then transported together.
   At the analysis grid (two MeV points, lightest masses 1e-6, 1e-3, 0.03 eV) within 3e-4 of the free flux of
   the reference; the helicity changes the fluxes by up to 0.9 of the free flux (scalar, 31.6 keV, 1e-6 eV) against
   relics in motion taken as unpolarized. A Dirac scalar transport costs 3.4-3.9 times one at rest, a vector
   1.6-2 times. Off by default; the SI campaign is to switch it on.

   A numerical note: far below the resonance (s ~ 1e-4 M^2) the semi-analytic kernel's closed-form angular integral
   cancels to 1e-3-1e-2 relative. For the Dirac shares it is taken by quadrature there (2026-10-21); for rho
   (Majorana, and the dark portal's thermal products) since 2026-11-08, with rho's lattice weights split at
   sigma = m as the shares' are. *Fixed* (2026-11-08; tests/test_thermal_products.py: pol 0 against direct 2D
   quadrature to 1e-6 at s ~ 1e-4 M^2, the shares' weights adding up to rho's to 1e-9). The dark portal's MeV
   planes computed before it carry the closed form (its effect on the fluxes within 6a's 2e-4).

11f. **Relativistic relics** (m_lightest below ~1e-3 eV, where T_nu (1 + z) is comparable): the resonance-only
    thermal treatment there (thermal=True, dp_si._thermal_resonance_terms) is unverified. *Not used* (checked
    2026-11-08): every MeV and secret-interaction configuration of campaign_ml, campaign_final and campaign_si sets
    thermal_products (thermal="full", 6a, 6c), which takes the relics in motion in every collision term and is
    checked against the reference at relativistic relics (lightest mass 1e-6 eV: tests/test_thermal_products.py,
    both mediators, Majorana and Dirac, within 2e-4 of the free flux; 1e-3 eV: 6a's tool checks). *Open* only for
    runs without the switch.

11g. **The size of the mixing's effect: a first look.** *To quantify* (method open, item 11). From the exact
    profile's validation runs of 2026-10-18 (branch agent/exact-profile-v2; logs and fits on Tycho under
    nulri_v2/output/validation, cancelled after ~20 min): points on the earlier campaign's limit contours, chi^2 with
    the mixing at Table II's centre against chi^2 with the six mixing parameters and m_lightest profiled. At the 18
    points with a saved fit where m_lightest stayed at its start (so the difference is the mixing's alone), the
    profile lowered chi^2 by at most 0.18 at 17 (every parameter within 0.3 sigma of Table II), and by 8.8 at one
    point deep inside the excluded region (proj_2050 dp_mu_pot_x1m12_pos, chi^2 ~ 5,600). At the one point with two
    couplings, the contour moves by ~3% in coupling. Coverage is thin for present-day data (2 such points, both 0.0).
    The lightest mass, by contrast, moved chi^2 by 7 to 1,000 at the other 7 points: it stays varied (item 11).

## Data and detectors

12. **Flavor-composition data (IceCube's MESE 2025 measurement, the 2050 projection) compared with flux-weighted
    flavor fractions** averaged over 10 TeV-30 PeV, not folded through the detectors' responses. *Accepted for now.*
    A detector measures flavor with each flavor's own effective area and cross section; folding the flux through each
    measurement's response would fix it (the event data sets do).

13. **IceCube's spectrum segments taken as independent** (their correlations are not published). *Accepted.*
    Since 2026-11-07 the segments are the published table (IceCube:2025ewu, arXiv:2507.22234, Table 10, CF column;
    `icecube_cf_2025`: 10 segments, 4.6 TeV-10 PeV), no longer those read off arXiv:2607.25966's Fig. 4
    (`ic26_astrophysical`, 7 segments to 1 PeV, three error bars off by up to a factor 2); a segment measured at 0
    keeps its 68% upper limit as the upper error and a zero lower one. (Mauricio, 2026-11-07)

14. **HESE's detector systematics: the profile uses a per-bin model, and the reported chi^2 is the exact per-event
    evaluation at the profiled nuisances.** The release's four-point interpolation of the systematics has kinks at
    its grid values, where the profile's minimum can sit. *Accepted.*

15. **DR2: IC79 + IC86, reconstructed energy above 10^3.5 GeV, declination above -5 deg,** a normalization per
    configuration (10% prior). *Accepted.*

16. **Future detectors: NuFlavorSingleSource's detector models** (effective areas parametrized and calibrated to
    published numbers, not the detectors' own tables), its systematics, no self-veto, each detector at full size for
    10 years. Projections use Asimov data. *Accepted.* (2026-10-14/15)

16a. **The 2050 flavor projection dropped from the campaign** (data set proj_2050: the projected 2050 flavor
    composition with IceCube's 2026 spectrum). *Removed* (Mauricio, 2026-10-18: "Remove 2050, document in case we want
    to revive later"). The future detectors' event forecasts (item 16) replaced it as the projection on 2026-10-14;
    it had stayed in the campaign by inertia (232 of its 928 planes). To revive it: regenerate its configurations
    with `python tools/make_campaign_configs.py --mixing lightest --datasets proj_2050` (the generator still defines
    it), and add it to the packs (`tools/campaign_packs.py --tree campaign_ml --sets ... proj_2050`). Its planes
    reuse IceCube 2026's solver runs (the same physics), so its added cost is the fits (~700 core-hours in the
    pilot's pricing).

16b. **The campaign restricted to L_e-L_mu, L_e-L_tau and the dark portal's MeV regime.** *Removed* (Mauricio,
    2026-10-18, in steps: "For long-range potentials, keep only Le-Lm and Le-Lt"; the B-3L models' long-range planes
    too; the dark portal's textures "only e, mu, tau, and all three equally", then for its MeV planes "only e, m, tau";
    the MeV gauge models "Only Le-Lm and Le-Lt"). Kept: L_e-L_mu and L_e-L_tau over ultralight masses (1e-30-1e4 eV: their long-range potentials) and over
    MeV masses (1e4-1e9 eV), and the dark portal's MeV planes for textures e, mu and tau: 7 per data set and source
    composition, x 2 compositions x 3 data sets = 42 planes, 76,050 points. Not generated any more: the dark
    portal's potential planes (7 textures x 2 signs x balanced + 5 coupling splits) and its e-mu, e-tau, mu-tau and
    e-mu-tau MeV planes; L_mu-L_tau (potential and MeV); B-3L_e, B-3L_mu, B-3L_tau, B-(3/2)(L_mu+L_tau) (ultralight and MeV). At
    MeV masses B-3L_mu and B-3L_tau couple to the neutrinos as the dark portal's pure mu and tau textures do (their
    planes the same up to the coupling's scale, below ~200 MeV where B-3L_mu's mediator gains mu+ mu- decays). The
    coupling-split planes' missing lightest-mass scan (the local fit missed a minimum lower by 1.96 in chi^2) no longer
    applies. To revive any of them: widen tools/make_campaign_configs.ML_KEEP and regenerate (`--mixing lightest
    --datasets ...`); split planes then need exact_profile's ml_scan, as the MeV planes have.

## Numerics

17. **The solver's redshift grid** (200 steps in the campaign): the flux normalization is accurate to a few 1e-3
    (first-order error). *Accepted.* Measured 2026-10-22: 200 steps put 0.5% more flux than 800 at every energy
    alike, and dp_si's transport (steps uniform in ln(1+z)) 0.17% less than 800 -- the "0.7%" between the two
    transports. The flux normalization floats in every fit, so no chi^2 moves; where a plane switches transport
    (below 1 keV, at SI_HARD_MIN_G) the step is absorbed the same way.

18. **Interpolation in the spectral indices** between solver runs at 3 nodes (7 for the MeV planes' upper index):
    at worst 0.35% in flux and 9e-4 in flavor fractions for the lower index (strong MeV absorption); ~1e-5 for the
    potential. *Accepted.* (2026-10-15)

19. **Sky resolution:** HEALPix N_side 8, the Earth crossed on 12 chords per declination band (grid.n_earth_chords),
    the Sun on 8 paths at Gauss-Legendre cosines of the angle to it (grid.n_sun_paths; the sky isotropic around it).
    *Accepted.* (Corrected 2026-11-08: it read 2 chords and an isotropic Sun average; the campaigns run 12 and 8.)
    The accuracy of 12 chords per band is not measured.

20. **The thermal kernels' grid (a bug, fixed 2026-10-21).** In every MeV transport with thermal products (the
    campaign's L_e-L_mu, L_e-L_tau and dark-portal MeV planes, and the 1-10 keV end of the ultralight and potential
    planes), the grid on which the resonance kernels' exact thermal averages are interpolated merged a regular grid
    with finer ones around slow relics' resonances, which put nodes arbitrarily close together; deep in the thermal
    tail the kernels' quadrature is good to ~1%, and the 4-point interpolation of their logarithm through two nearly
    coincident nodes overshot without bound: infinite kernels (a crash; the failed fits of the potential planes at
    1-10 keV) or isolated energies absorbed outright (in 48 random MeV points: two off by 0.66-0.76 of the free flux,
    once inside the data range). *Fixed* (dp_si._thin_grid: neighbours at least half the finest spacing apart; within
    3.4e-5 of the reference where the old grid was 0.76 off). The campaign's results were re-evaluated with the
    fixed code at every stored solution (125,458 evaluations; tools/recheck_thermal_grid.py; the old grid's control
    gives back the stored chi^2 exactly for MESE and the projection) and refitted where chi^2 moved by more than 0.05
    (~600 MeV points) or the fit had failed (7,478 potential-plane points), and the planes' reading takes the
    corrected values (analysis/exclusion.corrected). The fix moved chi^2 by more than 0.05 at 0.5-0.8% of points
    (at most ~1.5); no robust edge moved; one artifact went (a thin excluded sliver at 1 MeV, 1e-4.96-1e-4.6, in the
    MESE L_e-L_tau planes, from one point the bug had pulled below 3.84); edges moved in a few columns where Delta
    chi^2 hovers within ~0.07 of 3.84 over a wide range (HESE+DR2, dark portal tau and e near 3-5 MeV); the
    potential planes gained their 1-10 keV columns. HESE+DR2's chi^2 itself is not exactly reproducible on
    re-evaluation at 0.6% of points (up to ~1.5, its inner nuisance fit): see 21.

21. **HESE+DR2's nuisance profile starts from the previous evaluation's nuisances** (events.EventLikelihood.profile,
    warm start), and where the profile has more than one minimum -- the HESE detector systematics' interpolation has
    kinks at its nodes -- the chi^2 depends on what the process evaluated before. Measured 2026-10-22
    (tools/events_profile_check.py, 240 points): restarted from 17 starting points, chi^2 moves at 85% of the
    mismatched points (by up to 1.48) and at none of the matched ones; the lowest start is never more than 0.01 below
    the lower of the stored and the re-evaluated values, so the lower one is right and the other a fit stopped early
    (L-BFGS-B fallbacks at 73% of the mismatched points' profiles, 30% of the matched). No 95% call among the
    mismatched controls flips. *Open, planned:* one re-evaluation pass at the end of the campaign, from fixed starts,
    keeping the lowest chi^2, at the planes' reference points and within ~2 of the threshold; the fitting code is
    left as it is (changing it would slow every HESE+DR2 fit).

22. **Ultralight mediators: relic mass changes, neutrino decay, the models' range of validity, the potential planes
    (2026-10-24; notes Sec. 12.8).** (a) A soft scattering that leaves the relic in a heavier mass state costs the beam
    Delta m^2/(2 p^-) per event; at 1e-9 eV and g_D = 1e-4 there are ~1e19 such events. Implemented 2026-10-24
    (src/dp_relic_loss.py, both transports): **validated** before the final run (4f09ab4, 2026-10-25: convergence audits, the validation map, an independent review). (b) For m_Zd < m_i - m_j the heavier neutrinos decay into the lightest and the mediator
    (Majorana rate twice the Dirac one); for g_D >~ 1e-7 the beam arrives as nu_1 and the relics are nu_1 only.
    Implemented 2026-10-24 (src/dp_decay.py, both transports; relics' composition after decays by z = 1, a hotter
    Fermi-Dirac nu_1 gas standing for the daughters; the mediator invisible; daughter kernels from massless
    kinematics; where the potential dominates, the incoherent split over eigenstates -- wrong, replaced 2026-10-25 by
    (e)): **validated** before the final run (4f09ab4, 2026-10-25; 400 redshift steps where decays act).
    Bounded so far (2026-10-24): the time the relics' composition is taken at is immaterial -- it enters only through the soft
    scatterings and the energy loss, and where those act at all the relics have long decayed: their decay exponent
    grows as g^2 (m/m_Zd)^2, the scatterings as g^4 / m_Zd^2 (at g = 1e-8 the exponent is already >= 1e7 wherever the
    decay is open, m_Zd up to 0.999 of the threshold, from z = 1 or from today); the daughters' kinematics are exact since 2026-10-24 evening
    (dp_decay.daughter_spectra: the energy window set by the mediator's and the daughter's masses, the kept and flipped
    helicities from light-front traces; the earlier massless kernels, claimed here to be within (m_j/m_i)^2 = 3%, were
    wrong near the thresholds and for nearly degenerate masses, where the daughter keeps 80-100% of the energy and the
    kept share is 1/3-0.6, not 1/2 -- the review's finding 1); an antineutrino's flipped daughter
    that lands in its parent's own energy bin can decay only from the next step on (dark-portal phase, 200 steps;
    the Galaxy, one sub-cycle). Open: the relic
    population where the decay is open is modelled as all nu_1 with three times the number, a Fermi-Dirac gas at
    3^(1/3) T; the true one depends on the dark sector's history (the mediator decays back into nu_1 pairs when
    m_Zd > 2 m_1, the daughters are colder than that gas). The jumps scale with the relics' number, the energy loss
    with number x <1/E>: **bracketed** 2026-11-08 (decision C; number x1 to x3, <1/E> x1/2 to x2): only the number moves
    limits, at m_Zd = 1e-2 eV, 10 points in the IceCube-spectrum planes, where x3 gives the weaker limit. (c) Where the couplings of the mediator's longitudinal mode (its
    Goldstone, Y_ij = i (G'_ij m_i + G'_ji m_j)/m_Zd) break perturbative unitarity (src/dp_unitarity.py, against
    Allwicher et al., JHEP 10 (2021) 129), the planes keep their ranges and the region is marked as not allowed
    (Mauricio, 2026-10-24). The boundary falls with the lightest mass (c from 2.46 to 1.25 for the e texture between
    1e-6 and 0.1 eV; 1e-6 eV gives the narrowest region, not the widest as first stated): the fits use only the
    lightest masses where the model holds at each point (exact_profile.lightest_cap; Mauricio, 2026-10-24), and the
    points where none does (the 1e-6 eV line) are shaded and not fitted: **accepted**. (d) The 18 potential planes per
    data set are dropped (Mauricio, 2026-10-24): in the model followed (Jana & Swain, arXiv:2607.04441) the Z mixing
    obeys eps' <= (v_phi/v_H) m_Zd/m_Z (tree level, their Eq. 6; the loop, their Eq. 20, is smaller), and every
    stored ultralight exclusion needs eps' 8-30 orders of magnitude above it (tools/check_zd_mixing_bound.py):
    **accepted**.
    (e) **Decay in a medium (2026-10-25; notes 'Decay in a medium', docs/dev/DECAY_IN_MEDIUM_2026-10-25.md).** In a
    potential the decay acts between the propagation eigenstates, a(p) -> b(z p) + Z_D, each with its own eigenvector
    (the daughter's at z sigma) and the window set by the effective masses; the old model (each eigenstate decaying at
    its mass content's mean rate, vacuum kinematics) let the most stable eigenstate decay into itself. Implemented in
    src/dp_medium.py (both transports; the dark-portal phase's drift pieces removed). Assumptions, with sizes: (i)
    [G, V] = 0 (the potential is the mediator's own field; exact for every model below 1 keV, checked in the code); (ii)
    Fermi's golden rule between eigenstates, zeta = 2 E Gamma / splitting << 1 (<= 1.5e-2 at the validation points;
    recorded per run; > 0.1 only for g/M >~ 30 eV^-1 in vacuum or Goldstone Yukawas >~ 1.6 near the unitarity
    boundary); (iii) the daughter's eigenvector held at its bin's centre (second order in the bin width: widths to 2e-4,
    bins to 1e-8 against the exact formula); (iv) the sigma ladder, interpolated in log sigma between rungs in the
    cosmology, the nearest rung in the Galaxy; (v) no decays from the Sun on: they matter only where the same
    potential's coherent effects are overwhelming (Sun theta >= 840, Earth theta >= 3.4e5). The extragalactic beam's
    crossing of the Galaxy, measured to move the detected flux by 13-41% (shape up to 34%, a flavor fraction up to
    0.5) at g/M = 0.1 eV^-1, is followed since 2026-10-25 (dark_portal_solver._flux_runs: every line of sight with the
    Galactic component's processes, per declination band into the Sun/Earth handover). For the cost, the Galactic
    component's paths and the crossing follow N_side 4 lines of sight (each carrying its N_side 16 pixels' emission at
    their mean density): against N_side 8, 5.5% on the Galactic component (inside its normalization's 20% prior) and
    3.3e-4 (composition 1e-5) on the crossing: **accepted** (Mauricio, 2026-10-25; the limits to be checked in the
    everything-on pilot). Validated on Tycho (cosmological
    phase): where the potential sets the eigenstates, 200 steps within 1.3e-4 in shape; where decays and jumps cycle
    and drain the flux to 1e-3-1e-2 of the standard one, 3-6e-3 at 200 steps, <= 1.7e-3 at 400: **decided** (Mauricio,
    2026-10-25): 400 steps wherever the decays act in the dark-portal phase (DECAY_Z_REFINE = 2). Galaxy: halved segments within 1.7e-8, the nearest rung within 4e-5 of
    the interpolation (three le_lmu points).
    (f) **The mediator's decay back (2026-11-07/08, decision B; src/dp_mediator.py).** Where m_Zd > m_k + m_l the
    mediator the decays make (nu_i -> nu_j + Z_D) decays back into neutrino pairs before the beam reaches the Earth,
    returning its energy as neutrinos and antineutrinos; until 2026-11-07 it was invisible. Now followed, behind
    dark_portal.mediator_decay_back, **on by default** (one default, dp_config's; propagate_spectrum reads it when not
    given one); False is the code before, bit for bit. It runs in every transport of the dark-portal and lepton-number
    planes: the dark-portal phase (dp_sweep.step_full_med), dp_si's transport where that takes the cosmology (the 'si'
    mode: no potential and g >= SI_HARD_MIN_G = 1e-4, i.e. the dark portal's band above 1e-4; there the one flux holding
    both kinds), and both dp_si transports of the lepton-number planes' Goldstone factors below 1 keV
    (dp_si_light.goldstone_factors; item 24 (a)). Since 2026-11-08 the secret interactions below 1 keV read it too ((g)
    below). The dark portal's ladder tables (dp_mediator.Mediator) have no Dirac neutrinos -- the dark-portal transport
    has none: dp_mediator.check_supported raises DecayBackUnsupported (a RuntimeError, which the objective does not turn
    into its wall) at configuration if ever asked; nothing from the module can end inside the objective as a wall. It
    acts where a pair is open (m_Zd > 2
    m_lightest) and the mediator's decay exponent over the path at the grid's lowest energy reaches 1e-8; elsewhere the
    run is the code before, bit for bit. What is modelled: the width per pair from the whole coupling G' = U^dag G U
    (its identity part decays the mediator into nu_k nu_k), Majorana, 1/2 for identical daughters, massive daughters'
    phase space (2 |G'|^2 [2 M^2 - m_k^2 - m_l^2 - (m_k^2 - m_l^2)^2 / M^2] - 12 m_k m_l Re G'^2, times S p / 24 pi M^2;
    massless: Tr(G G^dag) M / 24 pi; k = l: G'^2 M beta^3 / 24 pi); the mediator as three populations by light-front
    helicity (0, +1, -1), each decaying at Gamma_Z m_Zd / E (each helicity's rate is a third of the sum: checked against
    the traces to 1e-10); the daughters' light-front helicities and shares from the traces -- massless daughters give
    one neutrino and one antineutrino, the neutrino's share 6 x (1-x), 3 (1-x)^2, 3 x^2 for helicity 0, +1, -1 (the
    d^1_{lam,-1} distributions; checked), while massive daughters also give two neutrinos or two antineutrinos (m_Zd =
    0.01, m_2 = 8.7e-3 eV: a helicity +1 mediator gives two antineutrinos 49% of the time); the mediator's light-front
    helicity at production from the same traces, which depends on the daughter's share (the rest-frame f_L of rest_rates
    is not the share that sets the decay in flight: 0.93 at m_3 = 0.05, m_Zd = 0.01 eV, against 0.47 longitudinal for
    the kept decays nu_3 -> nu_1 on average; a neutrino parent emits helicity -1 rather than +1, an antineutrino the
    mirror -- kept, it moves energy between the neutrinos and the antineutrinos). The tables (per parent bin, the shares
    of the mediators and of their daughters in each bin below, the parent spread over its bin): the light-front
    densities are polynomials in the decay angle (degree <= 5), taken exactly (a degree-8 Chebyshev series at 33 points)
    and integrated in the logit of the share with 8-point Gauss-Legendre panels, valid up to the kinematic thresholds
    (tested at both couplings' thresholds); number exact by telescoping (what falls below the grid in the last slot);
    energy exact at the bins' centres per table block -- a tilt where it moves no entry by more than half, else a share
    moved into the adjacent bin, else a cloud-in-cell split. dp_si's tables take the mediator's own mass m_Zd, not the
    collision mass (Debye screened). Approximations, with their exactness conditions: (i) the mediator's production and
    its daughters use vacuum kinematics and labels (eigenstate a = the a-th lightest mass state; the daughters' mass
    states projected on the eigenstates of their bin): exact where the potential is below 1e-6 of the smallest splitting
    at the bins where mediators are made, and good to O(sigma |s| / splitting) above it -- the dark portal's band has no
    potential (eps' = 0); for L_e - L_mu at m_Zd = 1e-2 eV the largest sigma |s| / Delta m^2_21 where mediators are made
    is 9e-6, 8e-5, 8e-4 at g' = 1e-6, 3e-6, 1e-5, and the share made above the threshold 5e-10, 2e-7, 2e-5 (recorded per
    run: dark_portal_solver._MEDIATOR_LAST medium_measure, dp_mediator diag[10]); channels a potential alone opens get a
    longitudinal mediator uniform in its share (number kept, energy approximate); (ii) within a step each bin's
    mediators are in the bin's own linear system with its neutrinos (dp_sweep 9 x 9, dp_si n + 3), solved exactly (e^M,
    phi_1, phi_2): made by the bin's decays into the bin itself, decaying, their neutrinos landing in the bin; what the
    bin sends below (mediators, their neutrinos) is a constant source there over the step (first order in the step, like
    the beam's own daughters). The mediators' decay exponent is capped at 1e9 per step (they drain within the step all
    the same). The Goldstone factors' scalar transport keeps its step-average scheme, the mediators as three more
    populations: against the exact solution of the same steps (decays alone, 100 decay lengths per step) within 2.7e-3
    of the step's change in the beam and 1.2e-2 in the mediators, halving with the step (tests/test_mediator.py). The
    first version (2026-11-07) added the mediators' neutrinos in their own bin at the step's end, which left an
    unrelaxed population ~ the step: dp_si's flux went as 1 / z_substeps, and the on / off of 330 reported then at g =
    1e-4 was that artifact (now 1.79); (iii) mediators alive at z = 0 are invisible (their decay exponent across the
    Galaxy is ~1e-5 of the cosmological one). Left out (would reuse dp_mediator's kernels with their own tables): the
    mediator's scattering on the relics; its production and decay on the Galactic lines of sight (dp_galactic_path;
    ~1e-5 of the path). The secret interactions below 1 keV: (g).
    Validation (2026-11-08; tests/test_mediator.py, 20 tests; Tycho, the final run's grid, 515 bins, 400 steps,
    m_lightest 1e-6 eV): off is the code before -- a frozen reference of 917dbff at four points (the dark-portal phase
    with decays, dp_si's transport, a potential, the Goldstone factors; bit for bit on the laptop, the test at 1e-9
    across machines), and on Tycho the arrival populations bit for bit at six points, the detector fluxes bit for bit at
    two and, at four, within 1.8e-15 of their maximum -- the same in two runs on one node, bit for bit on the laptop
    (compiled-code rounding on Tycho's CPU, not traced further); with it on, every one of the four transports runs (a
    test). The mediators' bookkeeping closes to 3e-13 (two neutrinos per decayed mediator, their energy the mediator's);
    decays alone, on returns the injected energy to 6.5e-7 (beam, live mediators, below the grid; off loses 1.15% of it)
    and the beam's number gain is twice the mediators' daughters on the grid to 1.3e-6. dp_si against dp_sweep, one step
    of decays alone: within 2e-6 of the change. Detector flux on / off, largest over 10 TeV-30 PeV (it falls to 1 at the
    grid's top), at m_Zd = 1e-2 eV: dark portal g 1e-6: 1.016; 1e-5: 1.99 (1.7-1.76 at 10 TeV, 1.83-1.90 at 100 TeV);
    1e-4 (dp_si's transport): 1.79; L_e - L_mu g' 1e-6: 1.058; 3e-6: 1.77; 1e-5: 1.85; m_Zd = 3e-2 eV, dark portal 1e-5:
    1.89; the flavor fractions at 100 TeV move by <= 0.007 (0.014 at m_Zd = 3e-2 eV). Outside the band, m_Zd = 1e-3 eV,
    g = 1e-6: 1.010 (dark portal), 1.11 (L_e - L_mu). The two transports' on / off at the same point: 4.5e-6 apart at g
    = 1e-6, 6.4% at 1e-5; at 1e-4 0.2-7% from 10 TeV to 1 PeV, apart near the grid's top (these runs skip the grid's
    extension above it, and dp_sweep has no hard scatterings). Medium measure (largest sigma |s| / Delta m^2_21 where
    mediators are made), L_e - L_mu at 1e-2 eV: 8.6e-6, 7.6e-5, 8.5e-4 at g' = 1e-6, 3e-6, 1e-5. The Goldstone factors
    (L_e - L_mu, 1e-2 eV): with the decay back in both of their transports against without, R moves by <= 2.5e-4 over 10
    TeV-30 PeV at g' = 3e-4 and by <= 2.3% at 1e-3 (where R itself spans 0.69-1.47); dp_si's 4 substeps against 2: R_tot
    within 4e-7 and 8e-6 (per state 6e-4); the detector flux on / off there 1.49-1.69 and 1.48-1.71.
    Convergence, on (off) and the on / off ratio: dark portal g = 1e-5, 800 steps against 400: 3.6e-3 (2.7e-3), ratio
    1.7e-3; 1029 bins against 515: 4.7e-3 (4.7e-3), ratio 4.2e-4; g = 1e-4, dp_si's 4 substeps against 2: 6.5e-3
    (6.6e-3), ratio 2.3e-4; its 1029 bins against 515: 7.2e-3 (6.7e-3), ratio 4.4e-4; the dark-portal phase at 1e-4, 800
    steps against 400: ratio 7e-5; L_e - L_mu g' = 3e-6: 1.5e-4 (steps), 4.8e-3 (bins, as off), ratio <= 1.8e-4. Cost on
    the final grid not re-measured since the coupled solve. **Implemented; the tests (laptop and Tycho) and the Tycho
    checks above pass; default on; the band to rerun (decision B).**
    (g) **The secret interactions' decay back below 1 keV (2026-11-08; dp_mediator.SIMediator, dp_si's sweep,
    dp_si_light.subkev_inputs).** The mediator X (scalar phi or vector V) that dp_si_light's decays nu_i -> nu_j + X make
    decays back, X -> nu_k nu_l (Dirac: nu_k nubar_l) where M > m_k + m_l, at Gamma_X M / E in the lab, its daughters
    returned to dp_si's populations; until 2026-11-08 it was invisible. *The setting:* dark_portal.mediator_decay_back,
    the same one, on by default (decided 2026-11-08: one switch and one default for "the decays' mediator decays back"
    in every model; the secret interactions already read dark_portal.neutrino_decay and relic_scattering; a second
    switch could leave the two families with different physics by accident); False is the code before, bit for bit.
    It acts where decays act (dp_si_light's decay not None), a pair is open (M > 2 m_lightest) and the mediator's decay
    exponent over the path at the grid's lowest energy reaches DECAY_BACK_MIN = 1e-8 (meta mediator_path_exponent);
    elsewhere the run is the code before, bit for bit. *What is modelled* (dp_mediator's construction of (f), for the
    four vertices of dp_si_light; paper Eq. lag_si): scalar, Majorana lambda_kl P_L + lambda*_kl P_R (lambda = U^T G U,
    the whole coupling); scalar, Dirac h_kl (h = U^dag G U); vector, Majorana gamma^mu (G'_kl P_L - G'*_kl P_R) (the dark
    portal's vertex: the tables are dp_mediator.Mediator.fold()'s, bit for bit, tested); vector, Dirac gamma^mu G'_kl
    P_L; the identity part of G rotated exactly (it decays X into nu_k nu_k). Widths with massive daughters: the
    scalar's sum|M|^2 = 2 |l|^2 (M^2 - m_k^2 - m_l^2) - 4 m_k m_l Re(l^2) (Majorana, S = 1/2 for k = l; massless
    Tr(l l^dag) M / 16 pi) and 2 |h|^2 (M^2 - (m_k + m_l)^2) (Dirac, every ordered pair; massless Tr(h h^dag) M / 8 pi),
    times p / (8 pi M^2); the vector's as (f). The scalar is one population (no polarization), the vector three (LF
    helicity); the daughters' LF helicities and shares from the exact light-front traces (Density): summed over them
    the scalar decays isotropically (each share uniform on its window, checked to 1e-10), massless daughters in
    same-helicity pairs (J = 0). Kinds: Majorana, one flux holding both (as dp_si's); Dirac, dp_si's six states --
    nu_k (LF helicity -1) and nubar_l (+1) visible, the others sterile (states 3-5, which decay again and are invisible
    at the detector): the massless scalar gives one visible and one sterile daughter per decay, the vector two visible;
    the production by the visible and the sterile parents (a -> c as dp_si_light.transport_arrays), the daughter kept
    or flipped. The kinds are summed exactly: the CP mirror of every decay has the same rates (they depend on |g|^2 and
    Re(g^2)), shares and visibilities. Tables per block exact in number and energy at the bins' centres (as (f)), also
    near the kinematic thresholds (tested). *Approximations, with their exactness conditions:* (i) vacuum kinematics --
    exact (no potential below 1 keV for these models); the mediator's vacuum mass M (not the collision mass, which for
    the vector is Debye screened): exact while the relics' Debye mass^2 is << M^2 (recorded, meta debye_sq_over_M2);
    (ii) per bin the mediators are in the bin's system with its neutrinos (dp_si n + 3: the vector's exact bin solution;
    the scalar's step-average scheme, first order in the step, as (f) (ii)), what a bin sends below a constant source
    over the step; (iii) mediators alive at z = 0 invisible; (iv) not on the Galactic lines of sight (dp_si_light's
    Galactic paths, ~1e-5 of the path; their beam's decays' mediators stay invisible). Validation (2026-11-08;
    tests/test_si_mediator.py, 30 tests; Tycho: the whole suite passes, 640 tests; tools/validate_si_decay_back.py on
    the SI campaign's grid -- 515 bins, 2 substeps, thermal products -- texture mu, m_lightest 1e-6 eV, pion source,
    E^-2.735, no Galactic component): off is the code before -- a frozen reference of five SI points (the four models at
    1e-2 eV, g = 1e-5, and the scalar at 1e-3 eV; tools/make_decay_back_reference.py --si), bit for bit on the laptop,
    the test at 1e-9 on Tycho; the widths against the traces to 1e-12 (massless: dp_si's widths; the scalar's P wave
    at k = l); the production densities add up to dp_si_light.daughter_spectra (to 1e-6, the Dirac vector's fit); the
    tables block by block exact (also within 1e-6-1e-10 of the thresholds); the sweep, decays alone: two neutrinos per
    decayed mediator and their energy the mediator's to rounding, both schemes; the Dirac daughters' visible share
    as the tables' (massless: the scalar 1, the vector 2 per decay); the mediators decay at Gamma M / E to 1e-11.
    Decays alone on the production grid with its extension (400 steps over a Hubble time, a beam above 1 PeV): on, the
    energy of the beam, the live mediators and what fell below the grid closes to <= 3e-15 with massive daughters
    (m_lightest 0.01 eV, M = 0.03 eV) and to 0.5-3.2e-9 at 1e-6 eV (the beam's own daughters below the grid, not
    counted), the neutrinos' number gain two per decayed mediator to rounding with massive daughters (0.998-0.99999 at
    1e-6 eV: the same below-grid daughters, in number); off
    loses the mediators' 8-17% of the energy. Flux at Earth on / off, all flavors, over 10 TeV-30 PeV (largest; at 10
    TeV, 100 TeV, 1 PeV), m_phi = 1e-2 eV: scalar Majorana g = 1e-6: 1.049 (1.049, 1.006, 1.000); 3e-6: 1.103; 1e-5:
    1.109 (1.109, 1.104, 1.049); scalar Dirac 1e-6: 1.038; 3e-6: 1.055; 1e-5: 1.056 (1.056, 1.055, 1.038) -- one visible
    daughter per decay, about half the Majorana's; vector Majorana 1e-6: 1.049; 3e-6: 1.81; 1e-5: 2.36 (2.10, 2.34,
    1.70); vector Dirac 1e-6: 1.022; 3e-6: 1.35; 1e-5: 1.72 (1.54, 1.68, 1.41); m_phi = 1e-3 eV, g = 1e-6: <= 1.003, g =
    1e-5: scalar Majorana 1.14, Dirac 1.13, vector Majorana 1.56, Dirac 1.32 (at 10 TeV; <= 1.06 at 100 TeV); never
    below 1; the flavor fractions at 100 TeV move by <= 0.014. The scalar's step-average scheme against the exact bin
    solution of the same steps, with the decay back (1e-2 eV, g = 1e-6, 1e-5, both kinds): <= 3.5e-9 in the flux
    (<= 7e-8 of the decay back's effect); in the sweep test (one step of 100 decay lengths) 2-3e-3 of the beam's change,
    the mediators 4.5-5.5%, falling with the step. Convergence (largest over 10 TeV-30 PeV), on / off: the redshift
    steps doubled (4 substeps against 2): <= 1.9e-6 for the scalars (g = 1e-6, 1e-5), <= 6e-7 for the vectors at 1e-6,
    2.1e-3 (vector Majorana) and 1.0e-3 (vector Dirac) at 1e-5, where off itself moves 2.6e-3 and 2.0e-3; the bins
    doubled (1029 against 515), g = 1e-5: 1.3e-6 (scalars), 2.0e-3 and 9.7e-4 (vectors; off 2.5e-3 and 1.9e-3).
    Cost (Tycho, 4 threads, one transport after the point's step terms, 1e-2 eV, g = 1e-5): on / off 1.06 (scalar
    Majorana, 14.4 s), 1.02 (scalar Dirac, 41.4 s), 1.13 (vector Majorana, 21.5 s), 1.14 (vector Dirac, 34.3 s).
    **Implemented and validated (tests on the laptop and Tycho, the checks above); default on; the SI planes' sub-keV
    band where decays act (a pair open, M > 2 m_lightest) to rerun.**

23. **Secret interactions below 1 keV, scalar mediator: the relics' own background field is not modelled; the region
    where it matters is shaded** (decided, Mauricio, 2026-10-25). *Why it exists:* a scalar couples to the neutrinos'
    scalar density, to which relic neutrinos and antineutrinos contribute with the same sign, so the relic background
    sources a steady scalar field that shifts every neutrino mass by ~ g^2 n / M^2, growing as the mediator gets lighter.
    A vector couples to the number current (neutrinos minus antineutrinos), which cancels for a background without a
    lepton asymmetry (the second-order remainder is <= 1e-3 g^2 of Delta m^2_21, negligible everywhere); the dark-portal
    and lepton-number vectors get their potential from ordinary matter, already transported. *Where it matters:* masses
    and mixing shift by more than 1e-3 of Delta m^2_21 (at z = 4) for g / M >~ 30-1e4 eV^-1: log10(g/M) = 1.5-2.1 at
    m_lightest = 0.1 eV, 1.8-2.7 for relics as produced, up to 3.8-4.0 where the relics have decayed into a light nu_1
    (which carries almost no scalar density) -- computed per texture, kind and lightest mass by
    dp_si_light.scalar_field_measure (notes, "Secret interactions below 1 keV", table); above g / M ~ 0.9-7e3 eV^-1
    (the non-relativistic relic states, z <= 4) the scalar's attraction between relics beats their pressure, far
    stronger than gravity, and the uniform relic gas is unstable (it clumps). At the scalar's limits this is every
    mediator mass below ~1e-11-1e-9 eV (decay limit, g ~ 5e-8-2e-7) and below ~1e-6-1e-4 eV (hard-scattering limit,
    g ~ 6e-3) (2026-10-25, computed). *Why shaded rather than modelled:* in the clumping regime every rate the
    transport uses (absorption, scattering, decays, screening) assumes a uniform relic gas of known density,
    temperature and masses; with clumps these are unknown without a structure-formation calculation, so neither an
    exclusion nor an allowed region computed there would be valid -- it must be shaded in any case. Modelling the smooth
    mass shift below the clumping onset (a self-consistent field per redshift, fed into the transport; estimated ~1-1.5
    days, not measured) would recover only the band between the onset and clumping, ~1-2 decades of mediator mass
    (~1e-10 to 3e-9 eV) near the decay limit. *How:* a computed boundary per texture and lightest mass, recorded and
    shaded on the plots, with the fits skipped there and capped to the lightest masses where the model holds (as the
    unitarity region, decision A): analysis.exclusion.not_modelled_measure / not_modelled_mask / draw_not_modelled,
    exact_profile.ExactProfiler._measure_cap (per point: the boundary is not one line, it moves where the relics
    decay). *To revive:* mass-varying neutrinos (homogeneous part); the clumping regime belongs to
    a separate cosmological analysis.

24. **Secret interactions below 1 keV: what the transport adds and leaves out** (2026-10-25; notes, "Secret
    interactions below 1 keV"; src/dp_si_light.py). *Added:* the decays into the mediator for the four models (exact
    light-front helicity spectra; Dirac neutrinos as six states, visible and sterile), the relics after their own
    decays, the vector's Debye-screened mass and relic mass-change loss, the Galactic lines of sight (decays,
    conversions, loss), the vector's unitarity line for Dirac neutrinos; since 2026-11-08 the decays' mediators decaying
    back into neutrino pairs in flight (item 22 (g); until then invisible). *Accepted approximations:* (a) the vector's
    longitudinal mode's own scatterings (Goldstone exchange and pair production, Yukawa ~ g m / M) are left out: run
    as dp_si's scalar channels with that Yukawa they leave the flux at 10 TeV unchanged at g / M = 0.01 eV^-1, at 0.62
    (0.27 for m_lightest = 0.1 eV) at 0.1, and remove it from g / M = 1 on -- inside the region the decays exclude
    (from g / M ~ 1e-4 the pilot's fits give Delta chi^2 ~ 10, now_ic26, pion); *check due on the campaign's results*:
    every vector point with g / M between 1e-2 eV^-1 and the unitarity line excluded in every data set and composition
    (else add them: dp_si's scalar channels with the Goldstone couplings beside the vector's); the dark-portal
    ultralight planes leave them out too, with the same check due. *On the lepton-number planes (L_e-L_mu, L_e-L_tau)
    they are included* (Mauricio, 2026-10-25: they move that limit): Majorana Goldstone channels in dp_si as a factor
    per mass state on the dark-portal phase's populations (dp_si_light.goldstone_factors; exact where the potential
    leaves the eigenstates mass states where they absorb, the bound recorded per run as meta['zeta']['goldstone']);
    not on the Galactic paths, not for Dirac neutrinos; (b) dp_si's quasi-elastic conversions have no Zeno factor from the decay widths: at most 2%
    on the conversions, within a decade of the unitarity line; (c) the scalar's collision terms take max(M,
    m_heaviest): exact but for the pair channel's collinear logarithm, which the neutrino masses cut (here at the
    heaviest mass; acts only four or more decades in g inside the decay-excluded region); the vector's pair channel
    keeps ln(s/M^2) below the neutrino masses, where its hard scatterings act only beyond the unitarity line; (d) Dirac
    relics: the relic mass-change loss with their left-chiral share (1 + v)/2 averaged before the 1/E weight (exact at
    rest and relativistic), and a relic's sterile decay daughters counted as relics of their mass state; (e) no relic
    lepton asymmetry (with one as large as the baryon asymmetry, a Dirac vector's first-order potential would matter
    only for g / M >~ 0.1-1 eV^-1, inside the decay-excluded region); (f) the hard scatterings on the Galactic paths
    left out, as for the secret interactions above 1 keV (the Galaxy's relic column is ~1e-5 of the cosmological one);
    (g) the extragalactic beam's crossing of the Galaxy is not followed for the secret interactions (no potential: it
    continues the cosmological evolution at z = 0 for ~1e-5 of a Hubble time). *Unchanged:* everything at 1 keV and
    above, bit for bit (tools/validate_si_light.py unchanged). *Numerics (2026-11-08):* the vector's profile tables
    (dp_si.build_vector_table_tri) crashed at the two lightest mediators, 1e-30 and 1e-29 eV (the SI pilot, both
    vector models: mu = M^2 / s ~ 1e-78, x_- = (1 - beta) / 2 rounded to zero and (a + delta)^2 underflowed); now the
    exact forms x_- = 2 mu / (1 + beta) and the ratios a / (a + delta), b / (b + delta): the tables build down to
    mu = 1e-150 (exact while mu^2 is representable) and vary smoothly in ln mu; the planes at 1e-30 eV run, within
    2e-2 of 1e-29 eV, itself within 1e-3 of 1e-28 eV (g = 1e-36, 40 bins; tests/test_si_light.py). The exact forms
    hold at every mu, so the tables move everywhere by the old forms' cancellation: <= 4e-14 of the scalar's pair
    table, <= 3e-16 of the vector's (mu = 1e-30-0.5, against 917dbff) -- at 1 keV and above no longer bit for bit
    with the code before 2026-11-08.

25. **The present-data sets and the forecasts after the audit of 2026-11-08** (2026-11-11; docs/dev/AUDIT_2026-11-08_
    resolution.md, docs/dev/C1_FORECAST_PORT.md, docs/dev/REVIEW_2026-11-11_physics.md). *Accepted, to state:*
    (a) set 1 compares IceCube's flavor likelihood (released grid, fitted under IceCube's broken power law) with our
    flux's composition averaged over 5 TeV-10 PeV and weighted by MESE's expected events, as if constant in energy;
    measured at one ultralight block: weighting by flux instead moves the limit by <= 0.4%, the single-power-law grid
    by <= 0.1% (review item 3; the other families pending);
    (b) set 1's spectrum segments are taken as independent (their correlations are not published); with a neighbour
    correlation of -0.3 the limit at one ultralight block moves by 2.3-2.9% (review item 5; others pending); the
    spectrum's break is our source model's, fixed at 24.5 TeV;
    (c) set 2's MESE term uses reconstructed energy only (the release's 1-D projections) and fails its gate against
    IceCube's flavor result (TS at f_e = 0: 1.42 vs 8.65), his choice to use it and state it; DR2's tracks near the
    horizon at 4-10 TeV are 11% below our atmospheric model (B7: our single-power-law index 2.62 vs IceCube's 2.37);
    nu_tau -> tau -> mu tracks are left out of DR2 (B3; 6.2% of IceCube's astrophysical tracks at 1:1:1, 16.7 of 270 per year, from the
    Plan A IceCube card's channels, review check 2026-11-11);
    (d) the antineutrino share at the sources is free between 1:2 and 1:1 (E1), built linearly from a neutrino-only
    run and the standard run; the Galactic flux keeps 1:1;
    (e) forecasts: Plan A's detector models average neutrinos and antineutrinos except the Glashow piece; tracks are
    binned in declination band and energy, showers and double cascades in energy only; the solver's crossing of the
    Earth is the South Pole's at every site (exact without a matter effect; the open part of C2 for ultralight
    L_e - L_alpha); IceCube's and Gen2's starting events 5-7% below Plan A's own diffuse counts (flavor-blind);
    (f) the 95% edge is read with a monotone cubic in log10 g between samples (review item 8); where it and straight
    lines differ by more than 2%, couplings are to be added near the edge (tools/edge_refine_list.py).
