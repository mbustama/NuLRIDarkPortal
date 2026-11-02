# Convergence audit of the transport, 2026-10-25

Scope: the numerical convergence of the transport before the final run, starting from the grid-wide validation
(tools/validate_transport.py, 6465 points of the ultralight planes). I changed no file under src/, analysis/, tests/,
notes/ or docs/ (except this one). The findings of TRANSPORT_REVIEW_2026-10-24.md and TRANSPORT_REVIEW_2_2026-10-24.md
are not repeated.

How I checked. All experiments run the cosmological phase alone: `propagate_earth_spatial` is replaced by the identity,
so the numbers are eigenstate populations at z = 0. The Sun, the Earth and the Galaxy are left out, and full-point
times were not measured. The two candidate fixes were tested as monkeypatches; src is untouched:

- `decay_step_whole`: an eigenstate decays as a whole.
- `decay_step_cic`: the parent is spread over its bin (cloud in cell).

Scripts are in the session scratchpad `conv/`: cosmo_only.py, patch_decay.py, toy.py, spline_check.py, job1.sh, job2.sh.
Tycho jobs 66910963 and 66910989 ran in the clone /lustre/astro/mbustama/external/nulri_conv_audit, at dev a0e876d,
each with 32 cores as 8 workers x 4 numba threads. Both were cancelled after about 25 min, once only uninformative
variants were left. The times quoted are wall times of the cosmological phase with 4 threads, and include JIT
compilation where a variant ran first.

## Verdict

The cycling-regime non-convergence has four separate causes. Three are numerical, and one is a decay formula that is
right only when the rate x step is small.

| # | What fails | Where it matters | Fix | Cost |
|---|---|---|---|---|
| R1 | dp_decay.decay_step: an eigenstate's survival over a step is 1 - sum_i T_ia (1 - e^-lam_i), so it **saturates** at 1 - (its decaying mass content) however fast the decay | le planes (potential-mixed eigenstates): g/M >~ 0.1 in the cosmology, >~ 0.01 in the Galaxy | keep = exp(-sum_i T_ia lam_i), decays split by T_ia lam_i | none |
| R2 | the injection is one pulse per z-step, placed between two half-steps of a stiff cycle drain: steady state x/(2 sinh(x/2)) of the true one, x = lam_eff dt | dp and le points where the conversion-decay cycle (or, after R1, the eigenstate decay) drains the beam | exponential integrator per bin, injection and inflow as sources (dp_si's column sweep) | dp: route to dp_si (measured: same time); le: port the sweep |
| R3 | Strang alternation of the jumps and a decay that is stiff within the sub-step: first order, drain rate low by R h / 2 (13% at CYCLE_STEP 0.25) | same as R2 (opposite sign: partial cancellation) | same as R2 | same |
| R4 | window_plan puts the parent at its bin centre: a daughter that keeps z > 1/h never leaves the bin, so narrow windows lose no energy | quasi-degenerate pairs (2 -> 1 at m_l >~ 0.03 eV): energy-grid dependence up to 46% (dp_si) and 39% (dark-portal phase) | cloud-in-cell weights at the window edges | none |

Converged references agree. At the dp_mu cycling point, the dark-portal phase at 16x steps and CYCLE_STEP 0.0125 matches
dp_si to <= 1.8% below 5e15 eV (<= 0.8% below 1e14 eV). At the quasi-degenerate dp_e point, the dark-portal phase with
nE x2 (or cloud in cell at nE x1), 4x steps and CYCLE_STEP 0.05 matches dp_si at nE x4 to ~1.5-3% below 1e15 eV. dp_si
itself converges at z_substeps = 2 (<= 0.16%). Its default of 1 does not (0.5-1.1%).

Safe as they are: the spectral-index spline, the shortcut thresholds, dp_si's caps and conversion rescaling, the lower
grid edge, the vacuum SVD, and the first-order injection (a uniform normalization).

## 1. Root cause in the dark-portal phase's cycling regime

Points studied (from the validation's flagged set):

- dp_mu_ul (m_Zd, g) = (1e-4, 1e-6), m_l = 1e-3: no potential (eps' = 0), conversion-decay cycle.
- dp_e_ul (1e-8, 1e-8), m_l = 0.1: quasi-degenerate.
- le_lmu_ul (1e-16, 1e-15), m_l = 1e-6: B/A = 0.25.
- le_lmu_ul (1e-16, 3e-17), m_l = 0.01, and (1e-20, 1e-20), m_l = 0.1: moderate points.

### R1. Eigenstate decay saturates (le planes; decisive)

Code: dp_decay.decay_step, the `keep` loop. For an eigenstate a with mass content T_ia, each mass component decays over
the step on its own: keep = 1 - sum_i T_ia (1 - e^-lam_i), lam_i = Gamma_i m_i dt / E. When lam_i >> 1, keep tends to
1 - T_ia whatever the rate. A potential-mixed eigenstate with a 1e-6 nu_3 admixture therefore loses at most 1e-6 per
sweep, even where its true rate is 1e9 per step.

Physics. Where Gamma_lab is far below the eigenvalue splittings, the eigenstate is the stationary state and decays as a
whole: H_eff = H - i Gamma / 2, so to first order Im E_a = sum_i T_ia Gamma_i / 2 and keep = exp(-sum_i T_ia lam_i). The
daughters, mass states, then project onto the eigenstates again (T[b, j, :], as now) and keep decaying. In vacuum the
condition Gamma_lab << splitting is energy-independent: 2 Gamma_rest m / dm^2 ~ 1e-4 (g/M)^2 for L_e - L_mu. That is
<= 0.01 for g/M <= 10 eV^-1. It reaches 0.1-0.5 only near the unitarity edge (Goldstone Yukawa ~ g m_3 / M ~ 3), where
the flux is already drained by orders of magnitude. Beyond it the decays become Zeno-frozen and neither formula applies;
an assert per point is cheap.

Evidence:

- **Step dependence.** At le_lmu (1e-16, 1e-15, 1e-6) the flux scales as dt: 2x steps give 0.50, 4x give 0.25 (the
  validation's B), at high energy. Decays alone reproduce it exactly (relic_scattering off: ratio 1.0000), and jumps play
  no part.
- **Mass basis.** The patched step (dec_i = Na (1 - e^-L) T_ia lam_i / L, L = sum_i T_ia lam_i) equals the original to
  rounding (max difference 0). dp_si and the potential-free dp planes are unchanged.
- **Effect.** At the same point the patched flux is 1e-4 to 1e-6 of the current one: fully drained, beyond the
  normalization's reach.
- **Scan in g/M at M = 1e-16 eV.** Physics depends on g/M alone above the Hubble scale. The table gives the ratio to the
  no-decay flux over 1e13-1e16 eV:

| g/M | m_l | current | fixed |
|---|---|---|---|
| <= 0.03 | 1e-6 | 0.49 (complete decay, flat) | same |
| 0.1 | 1e-6 | 0.42-0.49 | 0.21-0.49 |
| 0.3 | 0.01 | 0.34-0.64 | 0.009-0.46 |
| 1 | 0.01 | 0.056-0.64 | 2e-5-0.006 |
| 0.3 | 0.1 | 0.95 (flat) | 0.38-0.94 |
| 1 | 0.1 | 0.87-0.95 | 0.012-0.36 |

At (1e-16, 3e-17, 0.01), per energy: current 0.64 (4.6e12 eV) to 0.18 (3e16 eV); fixed 0.52 to 7e-4.

- **Validation.** In the le decay band (M < 0.05 eV), the B failures (shape or flavor > 1e-2) are 0 for g/M < 1e-4,
  20/547 for 1e-4 <= g/M < 0.1, and 441/484 for g/M >= 0.1. GB (Galaxy) failures start at g/M = 1e-2: the Galaxy's
  electron density is 1e5-1e7 times the cosmic mean, so its eigenstates are mixed at lower couplings.

Fix (dp_decay.decay_step; dp_galactic_path uses the same function):

```
L = sum_i T_ia lam_i
p = -expm1(-L)
keep = 1 - p
dec_i = Na p T_ia lam_i / L      (then the existing daughter windows)
```

Two side effects, both checked:

- **Vacuum mass content.** The decays now see tiny admixtures. In the vacuum limit (the dp planes, c_z = 0) T is the
  identity up to rounding (~1e-32). Inside the allowed region Gamma_lab t_H <~ 1e17, so the spurious drain is <~ 1e-15.
  Setting T = delta below the table's vacuum floor is optional.
- **Galaxy's vacuum fast path.** GALACTIC_VACUUM_ROTATION (1e-6) bounds the rotation theta, not the decay through it.
  But V and Gamma both scale as (g/M)^2, so theta <= 1e-6 at E_max forces (g/M)^2 <~ 2e-7 and theta^2 Gamma_lab L_gal
  <~ 1e-8. In the dp planes theta = 0. Safe.

Impact on the limits. R1 decides the le planes inside the decay band where g/M >~ 0.1 (cosmology) and >~ 0.01 (the
Galactic component). If the data exclude a completely decayed flux (all nu_1 at Earth, flavor ~(0.68, 0.22, 0.10)), the
95% contour lies at g/M ~ 1e-4 - 1e-3 and R1 only deepens the excluded region. If they allow it, R1 moves the contour.
Not measured: refit one le column (e.g. M = 1e-16 eV, profiled m_l) after the fix.

'combined' mode (le points with g >= 1e-4) takes its total flux from dp_si, which has no admixture decays. After R1 the
rescaling would cancel that drain. Route the le decay-band points with g/M >= 0.01 to the dark-portal phase, or flag them.

### R2 and R3. Pulse injection and splitting order reduction (dp planes; le after R1)

The step is [D J]_1/2, then C (injection and absorption), then [D J]_1/2. Inside each half, n_sub Strang alternations
of the jumps J and the decays D. Two errors of opposite sign:

- **R2, pulse injection.** Where the cycle drains the stable state at lam_eff with x = lam_eff dt ~ 1 (a stiff sink),
  the injected pulse decays over half a step. Steady state: S dt / (2 sinh(x/2)) instead of S / lam_eff. The ratio
  x/(2 sinh(x/2)) is 0.96 at x = 1, 0.85 at 2 and 0.41 at 5. Too low at coarse steps, second order but with
  coefficient lam_eff^2.
- **R3, splitting.** With the decay stiff inside a sub-step (Gamma h >> 1), Strang's survival per sub-step is
  (1 + e^-2Rh)/2 instead of e^-Rh. The drain rate comes out low by R h / 2, first order (order reduction). At
  CYCLE_STEP 0.25 that is up to 13%. The cap CYCLE_MAX and the midpoint rates do not enter, which is why changing
  them changed nothing.

Toy (toy.py: one bin, the code's exact operator sequence, against the ODE):

- As n_sub grows, code/exact tends to the pulse prediction: 0.9635 against 0.9595 at x = 1, 0.8596 against 0.8509 at
  x = 2.
- At R h = 0.25, x = 0.5, it is 1.13. At R h = 0.25, x = 2, it is 1.007: the two errors cancel by accident.

Real code, dp_mu (1e-4, 1e-6, 1e-3). Flux / dp_si (z_substeps 8), 4.6e12 to 5e15 eV:

| variant | ratio | time |
|---|---|---|
| default (nz 199, CYCLE_STEP 0.25) | 0.83-0.91 | 7 s |
| nz 796 | 1.01-1.08 | 8 s |
| nz 3184 | 1.02-1.09 | 17 s |
| nz 199, CYCLE_STEP 0.05 | 0.78-0.80 | 55 s |
| nz 796, 0.05 | 0.97-0.99 | 38 s |
| nz 3184, 0.0125 | 0.983-1.000 | 178 s |
| same, no decay width in the jumps' damping (F4) | identical | |
| dp_si z_substeps 1 / 2 / 4 | 0.994-0.998 / 0.998-0.9997 / 0.9997-1.0001 | -, 7 s, 18 s |
| dp_si nE x2 | 0.9996-1.0000 | 58 s |
| dp_si E_extension 100 | 0.9984-0.9998 | 10 s |

Above 1e16 eV the dark-portal phase falls to 0.59-0.91: the missing inflow from above (finding H).

Fix:

- **dp ul planes** (eps' = 0, no potential): route every point where the decays and the soft scatterings are both on to
  dp_si with z_substeps = 2. Same physics (mass states = eigenstates), golden-rule conversions (F4's correct physics),
  ν + ν̄ folded (exact at 1:1, F10). Hard scatterings are included, <= 1e-6 below g = 1e-4. The E_extension is built in.
  Measured: 7-11 s against 7-32 s for the dark-portal default at the two dp points.
- **le planes** (potential; dp_si cannot carry the eigenstates): port dp_si's column sweep to the dark-portal phase.
  Per bin and step:
  - the matrix is the jump rates (as _soft_jumps) minus the whole-eigenstate decay rates (R1), plus the own-bin
    daughters;
  - injection plus inflow from above enter as a constant source;
  - F(dt) = e^M F + phi1 S dt, and the decayed number comes from <F> = phi1 F + phi2 S dt;
  - daughters go below through window_plan's running sums.

  This removes R2 and R3 and the in-sweep chain overcounting. dp_si shows such a sweep converges at ~160 steps.
  Interim, measured without jumps: R1 + CIC + nz 3184 (~10 s, against ~1 s now). At (1e-16, 3e-17, 0.01) that is within
  0.5% of nz 12736 below 5e15 eV, and 0.8% at 1.3e16 eV. nz 796 is within 1% only up to 2e15 eV. Where the drain is deep
  in the top decade (m_l = 0.1, g/M = 1), nz 3184 is still 4-23% off and nE x2 moves it by up to 50%. That case needs the
  sweep.

### R4. Bin-centre daughters for narrow windows (energy-grid dependence C)

window_plan treats the parent at its bin centre. The own bin receives only z in [1/h, 1], so a daughter keeping more than
1/h = 0.990 of the energy stays put and the decay costs no energy. For 2 -> 1 at m_l = 0.1, z lies in [0.9926, 1]. Each
jump-decay cycle then loses 0.37% physically and 0 in the code, and hundreds of cycles accumulate.

Single decay of an E^-2.7 spectrum against the exact convolution: bin-centre +0.64% per decay at 439 bins, +0.08% at 877;
cloud in cell +0.01% at both.

Transport at dp_e (1e-8, 1e-8, 0.1), normalized to the dark-portal phase at nE x4:

| variant | ratio |
|---|---|
| dark-portal nE x1 / x2 / x4 | 1.39 / 1.003 / 1 |
| dark-portal + CIC nE x1 / x2 | 0.970 / 0.967 |
| dp_si nE x1 / x2 / x4 | 1.27 / 0.869 / 0.8675 |
| dark-portal nE x2 + nz 796 + CYCLE_STEP 0.05 | within 1.5% of dp_si x4 below 1e15 eV (the rest: steps, top edge) |

Fix: cloud in cell. The parent is uniform in ln E over its bin, so offset k receives int f(z) tri(ln z / L + k) dz,
L = 2 ln h. Sum_k tri = 1, so number is kept exactly. Only the offsets the window's edges touch change; the interior
bins' moments differ by O(L^2/24) = 2e-5 and stay as they are. Narrow windows need more than 6 slots (width + 2); dp_si's
`for x in range(6)` loops follow. No runtime cost; the patched variant is bit-identical where windows are wide (m_l 1e-3).

Relevance: only windows a few bins wide, i.e. 2 -> 1 at m_l >~ 0.03 eV. The m_l prior (sum of masses < 0.12 eV)
disfavors that range. The C failures at m_l = 1e-6 sit at the grid top (H), not here.

## 2. The whole transport, item by item

| Item | Safe? | Evidence / reason | Fix if not |
|---|---|---|---|
| Grid top, dark-portal phase (H) | No | Converged dark-portal phase vs dp_si with its extension: 2-5% at 5e15 eV, 9-19% at 1.3e16 eV, 41% at 3e16 eV (dp_mu and dp_e points) | Internal extension: x10 is enough below 1e16 eV. dp_si's E_extension 10 vs 100: <= 0.2% below 1e16 eV and <= 1.4% in the top bin, at the cycling and quasi-degenerate points. Use x30 if the top bin's 1% matters |
| Extension: Galactic template | Design gap | dp_galactic._intensity **raises ValueError** above 1e8 GeV, so an extended grid cannot be passed through as it is | Zero emission above the template end, or keep the Galactic path on the analysis grid. The in-Galaxy cascade from above 3e16 eV is <= (3e16/E)^-(p-1) x the Galactic share, <~ 1e-3 below 1e16 eV |
| Extension: Earth | Safe | dp_spatial has no energy-transfer term | Crop to the analysis grid before propagate_earth_spatial (cost) |
| Galactic path, decays | No (R1) | GB failures in le from g/M = 1e-2 | Automatic with R1, after recompile (decay_step is bound at import) |
| Galactic path, emission placement | No | Bin j's emission is added at its far end and evolved over the whole segment: e^-x instead of (1 - e^-x)/x where lam_eff L_seg >~ 1. n_split (GB) splits the evolution, not the emission. GB, shape or flavor > 1e-2: ~160 of ~400 points per le family (mostly R1), 18-30 of ~215 per dp family | Emission as a source over its segment (phi1, as R2), or spread over n_split pieces at their midpoints. Then GB at n_split 2 and 4 |
| Galactic vacuum SVD (tol 1e-10, per-energy relative) | Safe | Truncation relative at every energy | None |
| GALACTIC_VACUUM_ROTATION | Safe, also after R1 | theta <= 1e-6 forces theta^2 Gamma_lab L_gal <~ 1e-8 (R1) | None |
| SOFT_JUMP_MIN (1e-4 of the flux moved over the path) | Safe | A bound on the effect; uses the unsuppressed rates | None |
| Decay switch-offs (1e-8 H0 at E_min, cosmology; 1e-6 over 30 kpc, Galaxy) | Safe | Bounds on the decayed fraction from the largest mass-state rate; after R1, T <= 1 keeps them conservative | None |
| Relic-loss switch-off (1e-6 E_min) | Safe | Relative shift <= 1e-6 | None |
| SI_HARD_MIN_G (1e-4) | Safe | Hard scatterings cost 1.5% at 1e-3, scaling as G^4: 1.5e-6 here. Mass-independent at s >> M^2 | None |
| dp_si DECAY_STEP_CAP / RATIO | Safe | A capped state still drains within the step; its events go to the decay to 1e-7 | None |
| dp_si CONVERSION_* rescaling | Safe in the allowed region | Keeps the in-bin equilibrium; error ~ loss/conversions <= 1e-7. lose includes tdec, so with fast decays the cap rises and nothing is rescaled. Review-2 L1 applies only outside the allowed region | Optional: count the capped or rescaled bins per run |
| dp_si steps | Not at the default | z_substeps 1 vs 8: 0.5-1.1%; 2 vs 8: <= 0.16%; 4 vs 8: <= 0.03% | z_substeps = 2 |
| dp_si energy grid | Safe except R4 | nE x2: <= 0.04% (dp_mu); 46% quasi-degenerate (R4) | R4 |
| Spectral-index spline (7 nodes, LOG_SPLINE_BEND_MAX) | Safe | Direct runs at the 6 midpoints vs the spline, at three cycling points (dp_mu si2, dp_e si2, le g/M 0.3 fixed): total flux <= 4e-5, state fractions <= 4e-7; largest bend 0.02 (guard at 5) | None |
| Injection (T1, low-z end) | Safe | Uniform +0.5% (B = 0.995 everywhere), absorbed by the free normalization | Optional: Gauss-Legendre step integral, as decay_I |
| Lower grid edge (daughters and losses below 4.57e12 eV) | Safe | Lost below the band; nothing flows upward | None |
| Zeno factor with the decay widths (F4) | Immaterial where tested | Identical when converged at dp_mu (W/w ~ 1e-8 there). Moot for dp after routing | Keep F4's recommendation |
| 'combined' mode | Not after R1 (le, g >= 1e-4, g/M >= 0.01) | See R1 | Route or flag |

## 3. What the final run needs

Target: shape and flavor to a few 1e-3 wherever the flux is within ~1e3 of the no-effect flux.

1. **Code, before the run (no runtime cost).**
   - R1 in decay_step, with a flag where Gamma_lab / min splitting > 0.1 (only near the unitarity edge).
   - R4 (cloud in cell) in window_plan, with enough slots in decay_step and dp_si.
2. **Routing.** dark_portal_solver._cosmology_mode: sub-keV, potential negligible (the existing sigma_max test; always
   for eps' = 0), decays open and jump_unit not None, so 'si' with z_substeps = 2. Measured: same time or less.
3. **le planes.** Port the column sweep (section 1, R2/R3). Until then, n_z_bins = 3200 for le points where the decays
   are open. Measured ~10 s against ~1 s (cosmological phase, no jumps). It is not enough in the deepest top-decade
   drains, nor where jumps are on (dp_mu-like points needed nz 3184 and CYCLE_STEP 0.0125: 178 s). Do not use 'combined'
   for le decay-band points with g/M >= 0.01.
4. **Grid extension** x10, internal, cropped before the Earth; the Galactic emission zero above 1e8 GeV.
5. **Galactic emission** as a source over each segment.
6. **Revalidate** (validate_transport) with acceptance d_shape and d_flavor <= 3e-3 where F_A >= 1e-3 F_std:
   - B: nz x4 for the dark-portal phase, z_substeps 2 -> 4 for dp_si;
   - C: nE x2;
   - H: extension x10 against x100;
   - GB: n_split 2 and 4;
   - GD;
   - F with the local potentials' admixture included.

   Then refit one le column, M = 1e-16 eV with m_l profiled, to see whether R1 moves the contour.

Not measured here: full-point times (Earth, Sun, Galaxy), the Galactic path under R1, and the fits.
