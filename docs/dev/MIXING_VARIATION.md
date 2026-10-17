# Varying the mixing and the lightest mass (2026-10-18)

Mauricio, 2026-10-17/18: "Jobs on current data should be run varying NuFIT 6.1 and the lightest neutrino mass
according to current upper limits on sum of neutrino masses. Jobs on projections should use tighter mixing-parameter
constraints and tighter limits on the sum of masses. Same as Table II in arXiv:2606.17132." -- "For the current
runs, vary using the pairwise NuFIT 6.1 chi2, not 1D chi2 gaussian distributions. Only use gaussian for
projections." -- normal ordering only; Majorana phases not varied ("they don't matter to us"); the projections'
splitting widths as Table II's numbers (0.5%).

## Penalties

- **Present-day data** (`now_ic26`, `now_events`): NuFIT 6.1's three pairwise tables summed (`analysis/nufit.py`,
  prior type `nufit_6.1_pairwise`): sin^2 th13 x sin^2 th12, sin^2 th23 x delta_CP, Delta m^2_21 x Delta m^2_31.
  Bicubic splines shifted by their undershoot (<= 0.05), delta_CP periodic, each table from its own ordering's
  minimum.
- **Projections** (`proj_2050`, `proj_future`): Table II's Gaussians (normal ordering): sin^2 th12 0.3088 +-
  0.0016416, sin^2 th13 0.02248 +- 0.00059, sin^2 th23 0.470 +- 0.00418, delta_CP 212 +- 6.687 deg, Delta m^2_21
  7.537e-5 +- 3.7685e-7 eV^2, Delta m^2_31 2.511e-3 +- 1.2555e-5 eV^2.
- **Mass sum** (Mauricio, 2026-10-18: "Use Table II's limits: 0.12 eV now, 0.09 eV for projections"): the paper's
  "half-Gaussian bounded at Sum_nu = 0", chi^2 += (Sum_nu / sigma)^2. Table II quotes both limits at 95% C.L., as do
  its sources: [16] PDG 2024 (Planck 2018 + BAO, 0.12 eV), [150] Palanque-Delabrouille et al. 2020 (Ly-alpha +
  Planck + lensing + BAO, 0.09 eV) and [151] Di Valentino, Gariazzo & Mena 2021 (0.09 eV); the 'projection' is a
  2020-21 bound. The paper gives no sigma. Correct translation (Mauricio: "If the paper says 95% C.L., inspect
  what's correct"): the limit holds 95% of the half-Gaussian, sigma = limit / Phi^-1(0.975) = limit / 1.960 --
  what a posterior bounded at zero with that 95% quantile is, and the same threshold (Delta chi^2 = 3.84 at the
  limit) as the planes are read at: 0.0612 eV now, 0.0459 eV for projections (`HalfGaussianPrior`, quantile
  'half', cl 0.95). The 'one_sided' convention, limit / 1.645 (GeoneutrinoDecay's), leaves 10% of the
  half-Gaussian above a 95% limit: it is a 90% limit's translation, 20% wider. Consequences (normal ordering,
  Delta chi^2 = 3.84 above m_lightest = 0): m_lightest up to 0.036 eV now and 0.025 eV for projections (0.044 and
  0.031 with limit / 1.645).

## Templates (`analysis/templates.py`)

A solver run at every value of six more parameters inside every fit is out of reach. A templated parameter is never
passed to the solver except in its template runs: at each point, runs with one parameter moved to each of its
template points, at the central index node of each piece; the flux at any value is the theta0 flux times
1 + sum_r w_r (F_r / F_0 - 1), element by element (Lagrange weights; trigonometric for delta_CP over a whole turn).
The Galactic part, independent of the index, is interpolated directly. The model also gives the derivatives in the
templated parameters in closed form, so the future detectors' exact gradient covers them (and the mass-sum
penalty's dependence on the splittings).

Measured on the test configuration (potential point where new physics changes the flux by up to 13%), largest error
relative to each bin's largest flavor flux:

| | error |
|---|---|
| s12sq linear, half a step away | 1e-5 |
| the correction taken at the central index node, at the extreme nodes | below 1e-6 |
| delta_CP over a whole turn, 3 runs | 3e-2 |
| delta_CP, 5 runs | 2e-3 |
| delta_CP, 7 runs | 1e-4 |

## Cost

A template run repeats what a point computes once for all its index nodes and sources: the Earth crossing, the
Sun's near field, the Galactic emission. Measured at the campaign's resolution (future detectors' grid, laptop):
a potential point 11 s for its six index runs, one template run 8-11 s; a MeV point 55 s, one template run 12-14 s.
Reductions (no change to any number): the Earth crossing as each chord's evolution operator, applied to the three
unit inputs (`dp_spatial._earth_transfer_columns`); the zenith-averaging weights and the Galactic maps kept between
runs.

## Open

- Template schemes per data kind (accuracy scan over the 3-sigma ranges at campaign resolution).
- The lightest mass: where it matters (MeV absorption, the coupling-split potential planes' screening) and how
  finely to profile it (`tools/mlightest_study.py` on the earlier campaign's limit contours).
