# Simplifications and assumptions

Started 2026-10-17 at Mauricio's request. One entry per simplification or assumption in the analysis: what it is,
why it is acceptable (or not yet), how large its effect is where known, and its status. Keep it current: add an entry
whenever a new one is made, and update the status when one is quantified or removed.

Status words: **accepted** (deliberate, effect understood to be small or conservative), **to quantify** (believed
small, not yet measured), **open** (decided to remove or change, not yet done).

## Propagation

1. **Absorption in the Earth applied after the flavor conversion.** *To quantify.* (2026-10-17)
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

3. **Standard matter effects off in the Earth** (`use_SM_potential: false`). *Accepted.*
   At TeV-PeV the standard matter potential barely changes the flavor of neutrinos arriving as decohered states.

4. **New physics acts on the astrophysical flux only, not on atmospheric neutrinos.** *Accepted (conservative).*
   The atmospheric neutrinos crossing the Earth would also feel the new potential; leaving that out can only weaken
   the bounds.

5. **No t-channel scattering on the relic neutrinos below 1 keV.** *Fixed in code (branch relic-scattering,
   2026-10-22); the affected planes to rerun.* Until then the dark-portal phase (below 1 keV) had the resonance only,
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

8. **Sources without redshift evolution, up to z = 4.** *Accepted (fixed choice).*

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
   cancels to 1e-3-1e-2 relative. For the Dirac shares it is taken by quadrature there; for rho (Majorana, and
   the dark portal's thermal products) the closed form stays while the running campaign uses it. *To fix after
   the campaign* (its effect on the fluxes is within 6a's 2e-4).

11f. **Relativistic relics** (m_lightest below ~1e-3 eV, where T_nu (1 + z) is comparable): the transport's thermal
    treatment there (dp_si._thermal_resonance_terms) is unverified. *Open.* The MeV planes' profile can land there.

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

13. **IceCube's 2026 spectrum segments taken as independent** (their correlations are not published). *Accepted.*

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

19. **Sky resolution:** HEALPix N_side 8, the Earth crossed on 2 chords per direction band, the Sun averaged
    isotropically. *Accepted.*

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
