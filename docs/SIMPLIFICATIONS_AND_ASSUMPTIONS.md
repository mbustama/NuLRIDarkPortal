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

5. **No t-channel scattering on the relic neutrinos.** *Accepted, with a range restriction.*
   Not modeled; above couplings of ~1e-7 it matters, so the coupling-split variants stop at x = 1e-8.

6. **Galactic component propagated adiabatically** from its emission points to the Sun (Landau-Zener parameter
   >= 1.5e5 wherever checked; `tools/galactic_adiabaticity.py`). *Accepted.*

## Sources and spectrum

7. **The extragalactic spectrum is IceCube's 2026 broken power law with the break fixed** at 10^4.39 GeV, both
   indices free (the lower one in [0.5, 2.5], while IceCube's 1-sigma range reaches 0.01). *Accepted.* (2026-10-15;
   `docs/dev/BROKEN_POWER_LAW.md`) The two pieces are split at the energy today, each a population with its own
   redshift weighting.

8. **Sources without redshift evolution, up to z = 4.** *Accepted (fixed choice).*

9. **Source flavor composition** at pion decay (1/3, 2/3, 0) in the nominal planes; fitted as (x, 1 - x, 0) in the
   `_fx` twins, extragalactic and Galactic independently. No tau neutrinos at the sources. *Accepted.*

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

## Numerics

17. **The solver's redshift grid** (200 steps in the campaign): the flux normalization is accurate to a few 1e-3
    (first-order error). *Accepted.*

18. **Interpolation in the spectral indices** between solver runs at 3 nodes (7 for the MeV planes' upper index):
    at worst 0.35% in flux and 9e-4 in flavor fractions for the lower index (strong MeV absorption); ~1e-5 for the
    potential. *Accepted.* (2026-10-15)

19. **Sky resolution:** HEALPix N_side 8, the Earth crossed on 2 chords per direction band, the Sun averaged
    isotropically. *Accepted.*
