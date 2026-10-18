# The exact profile over the mixing and the lightest mass (2026-10-18)

Mauricio's decisions (binding): in every fit at every point the six oscillation parameters (sin^2 th12, sin^2 th13,
sin^2 th23, delta_CP, Delta m^2_21, Delta m^2_31) and the lightest neutrino mass are profiled; present-day data under
NuFIT 6.1's pairwise chi^2, projections under Table II's Gaussians; the mass sum under a half-Gaussian (0.12 eV now,
0.09 eV for projections, 95%); normal ordering; no Majorana phases. The profile must be exact: the true minimum over
every nuisance at every point, to the fits' tolerance. The template scheme (docs/dev/MIXING_VARIATION.md) missed it by
0.1-2 in chi^2 (projections, cross terms) and is not used.

## Why it is hard

Every new value of the seven parameters eta needs new solver runs at every spectral-index node of both pieces of the
broken power law: ~3 s on Tycho for a potential point, 17-28 s for an MeV point (5-8 s after the batched transport
below). A fit with finite differences tries hundreds of values. And the lightest mass is not a gentle nuisance: at the
MeV planes it moves the absorption resonances (E = M^2 / 2 m_j) and chi^2 by up to ~80, with several minima (e.g.
b_3le_mev at 1e4 eV: chi^2 75.5 at m_lightest -> 0, 12.4 at 0.016 eV).

## The method (analysis/exact_profile.py)

**Exact centres, local surrogates.** A trust-region Gauss-Newton iteration in eta whose iterates (centres) are
evaluated exactly. At a centre c the model is a templates.py model centred there: the exact runs at c (every index
node) and, per parameter of eta, runs at c + h (h a quarter of the prior's width; each piece's central index node),
combined element by element as F(eta) = F(c) exp[sum_i (eta_i - c_i)/h_i log(F_i/F_c)]. At eta = c it is the exact
model for every nu (the template runs are skipped there: the factor is exactly 1); its gradient in eta is the
differences'. The surrogate chi^2 is this flux in the analysis' own likelihood with the priors exact -- no solver run.

**Steps.** The surrogate is minimized over (eta, nu), eta in a box of Delta prior widths; its predicted decrease is
checked by evaluating the step's end exactly (a new centre): accepted if chi^2 fell, the radius adapted from the
ratio of actual to predicted decrease (up to 64 widths, for points deep in an excluded region whose mixing is pulled
far), rejected otherwise.

**Certificate.** A centre is reported when the surrogate built with central differences (runs at c - h too: the
parabola in log F through the three; one-sided three-point at a bound) predicts no decrease larger than `stop` (1e-4)
within at least 0.5 widths. Then (i) the reported chi^2 is the exact model's at the reported values; (ii) the
surrogate has the exact gradient in eta at c up to h^2/6 times the third derivative (~1% of the slope), and the
exact gradient in nu; so the exact chi^2's minimum near c lies below the reported value by at most stop + e^2/(2
lambda), e that gradient error and lambda the curvature (at least the prior's) -- ~1e-6 beyond stop.

**Separate basins** (profile_global):
- *Octants* (present-day data): NuFIT 6.1 has a second minimum of sin^2 th23 at 0.55 (0.756 above the first, 0.47).
  The best solution is moved into the other octant by the shift between the two minima of NuFIT's (s23, delta_CP)
  table, (+0.08, -25 deg), evaluated exactly (one evaluation), and profiled from there when within 1.0 of the best.
- *The lightest mass* (MeV planes): the exact chi^2 on a grid of m_lightest (1e-6 eV and 1e-4 ... 0.1 eV, five per
  decade) at Table II's mixing (the same for every data set: the runs are shared through the disk cache), then the
  exact profile from every local minimum of the scan within 3 of its lowest. The grid's spacing (0.2 decade) is finer
  than any structure chi^2 can have in m_lightest: an absorption feature spans E_res/5 to E_res (the sources' redshifts)
  and the data's energy bins are 0.1-0.33 decade wide, so a basin is at least ~0.3 decade wide.

**m_lightest's bound.** 1e-6 eV, not 0: the MeV transport (dp_si) treats an exactly massless relic as not interacting
(s = 2 E m = 0), while any m > 0 gets the thermal treatment, continuous as m -> 0. The mass-sum penalty changes by 3e-5.

**Planes** (analysis/scans.py, with "exact_profile" in the configuration): chains along the coupling at each mass, from
the smallest coupling up; the global fit is the exact profile at the smallest coupling (no new physics). A point whose
solver runs at its chain's last certified solution equal that solution's runs (to 1e-10 per element relative to the
energy's largest; 2.4e-7 -- two float32 units -- in sky mode, whose bands are single precision) takes its certificate
after one evaluation (ExactProfiler.inherit: chi^2 is then the anchor's function of every parameter up to that
perturbation, whose effect on the minimum is first order at the anchor's solution -- where it is evaluated exactly --
and second order elsewhere). Never with the lightest-mass scan (other masses are not compared). Repairs skip a
neighbour's solution inside the point's certificate box.

**Fits.** The fits inside (centres' nu, surrogate steps) use forward differences of step 1e-5 internal units: the
campaign's 1e-3 decade in norm_astro is biased by ~1.25 chi^2 per decade by the curvature IceCube's 2026 spectrum
puts there, more than the slope near the minimum, so L-BFGS-B's line search failed and every fit stopped 1e-4 to
1e-3 (once 0.03) above the minimum (found 2026-10-18; fitting.FitSettings.fd_central / fd_step).

## Speed-ups that change no number

- **MeV transport for many sources at once** (dp_si._si_sweep_cols): a point's 20 transports per mixing value (2
  basis sources x 10 index nodes) share their collision operator; one sweep now carries all columns (the unit
  mass-state sources of every index node and window, kept per physics), the secondaries pushed through a
  source-independent kernel. Agrees with the one-source sweep to 1e-11 (tests/test_si_batch.py); a column's result
  does not depend on its batch.
- **No template run at a centre** (FlavorModel._at_theta0): the model at its centre is the exact model, bit for bit.

## Fixed on the way

- The 35 wall points (chi^2 = 1e9) of the October campaign: the x = 1e-20 planes' top coupling, which Tycho's numpy
  computes as logspace(-60, -20, 41)[-1] = 1.0000000000000001e-20, one rounding above the split x: the model refused
  P > x. Within 1e-12 it is now x (epsilon' = 1).
- The fits' forward-difference step (above).

## v2: the cost cuts (2026-10-18, branch agent/exact-profile-v2)

- **Inheritance at 1e-6** (`inherit_eps`, sky mode too): the old 1e-10 lay below the solver's own differences
  between couplings and never fired. With the point's runs at the anchor's eta within eps of the anchor's, chi^2_P =
  chi^2_A + delta with delta = O(eps x chi^2's sensitivity to the flux); the reported value is exact at the anchor's
  eta (nu refitted) and the minimum lies below it by ~|grad_eta delta|^2 / (2 lambda) -- second order. Measured by
  `inherit_check` (the central certificate's predicted decrease at inherited points minus the anchor's).
- **Forward certificates**: differences of step 0.05 widths (the solver's slopes from +-0.05 and +-0.25 widths agree
  to ~1e-6 per width; forward ones differ from central ones by exactly (h/2) x the curvature of log F). A forward
  model certifies when, with the curvature K measured at the chain's last central certificate (within 1 width),
  the surrogate minimized with every slope shifted by twice the truncation (stacked log ratios minus K; the
  model's `stack_shift`) still predicts no decrease beyond `stop` and at most 1e-4 more than unshifted (the
  predicted decrease is convex in the slopes: every truncation between 0 and twice the measured is covered). Else
  central differences, which refresh K.
- **The Earth/zenith geometry** depends on the mixing only through the screening mass: its pieces (zenith weights,
  columns) were already kept per exact argument set; the Earth transfer's products are now batched matmuls.
- **m_lightest scan at three per decade** (11 masses): shared through the disk cache by the configurations with the
  same energy grid (now_ic26, proj_2050 and twins; now_events and twin; proj_future and twin) -- the scan's mixing
  is Table II's for all.
- **MeV inheritance** (Mauricio's option B): along a mass column the points that pass the inheritance test against
  the global fit (no new physics) are held back; the scan of m_lightest is probed at the highest of them, going
  down: if its runs equal the global fit's at every scanned mass (1e-6), new physics is below the solver's reach at
  every mass there and at every smaller coupling (absorption grows with the coupling), and those points inherit;
  a point whose scan shows new physics is profiled in full (its scan's runs reused). Above the first point failing
  the test, every point is profiled in full.
- **Per-point checkpoints**: each point's record is written by its worker when done (with its time `t`);
  `tools/exact_progress.py --watch` prints the PROGRESS line (rate, campaign projection) on every new point and
  every 110 s.
- **stop 1e-3** (Tycho 66462115: the final chi^2 moved by at most 5.9e-4 between 1e-3 and 1e-4 at eight points).
- **Speed-ups equal to rounding** (tests/test_speedups.py): the Galactic emission's mixing-independent sums kept
  per density pattern (dp_galactic.emission_structure: half of the emission's time), the zenith Filon weights
  compiled (40x), the MeV profile tables kept per relic mass (a Delta m^2 difference rebuilds one relic's).

## Validation (tools/exact_profile_validation.py; Tycho)

To be filled with round 3 (output/validation/exact_66461885) and the pilot (output/analysis/pilot_exact).

## Cost

To be filled from the pilot.

## How to run

    python tools/make_campaign_configs.py --spectrum bpl --mixing exact      # analysis/configs/campaign_exact/
    python -m analysis analysis/configs/campaign_exact/<set>/<plane>.json --workers 16

with NULRI_FLUX_CACHE set (the MeV scans share runs). Validation at chosen points, against brute-force fits:

    python tools/exact_profile_validation.py <config> --points '[{...}]' [--ml-scan] [--octants] [--bruteforce]

## Still open

- Relativistic relics (m_lightest below ~1e-3 eV): the thermal treatment there is unverified; the MeV planes'
  profile can land there.
