# Transport-equation review, 2026-10-24 (independent, adversarial)

Scope: the working tree at 5a36f8b plus the uncommitted diff (dark_portal_solver.py, dp_cosmo.py, dp_decay.py,
dp_galactic_path.py, dp_relic_loss.py, dp_si.py, dp_absorption.py, dp_unitarity.py, analysis/exclusion.py's boundary),
against notes Sec. 12.7-12.9 and docs item 22. I changed no code. The checks below ran in seconds on the laptop; their
scripts are in the session scratchpad (lind.py, checks.py, unit.py, jumps.py). I did not run the solver.

Short verdict. The bookkeeping is sound: units, (1+z) powers, observed vs physical energy, the redshift measure,
number conservation, and the closed-form steps are all right. Two physics errors change results inside the allowed
region:

1. The decay daughters' kinematics ignore the mediator's mass.
2. The not-allowed boundary is computed at the lightest mass that gives the *narrowest* region, not the widest.

One error is latent: the relic mass-change loss leaves out the Z-mixing coupling. It is harmless only because the
potential planes are dropped. The Zeno factor in the jumps is wrong physics, but it has no effect wherever results
depend on it. Details follow.

---------------------------------------------------------------------------------------------------------------------

## 1. The transport equation as the code implements it

State. F^k_a(E, z) is the number per unit *observed* energy E, for kind k in {nu, nubar} (the two Majorana
helicities) and state a. In the dark-portal phase, a is an eigenstate of H = [M_rot + sigma S]/(2E(1+z)), with
sigma = E c_z(z) and c_z = 2(1+z)A(z) (solver.py:314-330), in ascending order and carried adiabatically. In dp_si,
a is a mass state with nu + nubar folded into one flux. In the Galaxy, a is an eigenstate of the local potential at
z = 0. The equation is d/dt at fixed E_obs, with dt = dz/((1+z)H).

| # | Term (physics) | Code | Discretization | Checked today |
|---|---|---|---|---|
| T1 | Injection: w_k Phi_0 (E(1+z))^-p R(z)(1+z) sum_f P_fa(sigma_inj) rho_f | dp_cosmo.py:526,544-545 | Per step at z_k (low-z end), dt = dz/(H(z_k)(1+z_k)) (solver.py:490): first order, +0.5% normalization (notes l.2117) | Normalization against free_streaming_flux (notes l.2117) |
| T2 | Resonant absorption -Gamma^abs_a F_a, with secondaries of both kinds (3E^2/E_j^3 same kind, 3(E_j-E)^2/E_j^3 other kind) x (W^dag G G^dag W)_aa / sum_G | dp_cosmo.py:445-591, 632-688; rates solver.py:521-534 | Exact per bin for any G dt; top-down sweep, 2x2 own-bin system | test_absorption.py; earlier audits |
| T3 | Soft jumps sum_b (R_ab F_b - R_ba F_a), R_ba = Gamma_u \|<b\|G_j\|a>\|^2 w^2/(w^2+D^2). Gamma_u = 2 n_hel (1+z)^3 sum_r eta'_r (G'G'^dag)_rr / (4 pi (m_Zd^2 + m_D^2(z))), with G' from the *full* G (kappa included). G_j = G - kappa I. w = (lam_a - lam_b)/(2E(1+z)). D = Gamma_u[(GG_a+GG_b)/2 - G_aa G_bb] + (W_a+W_b)/2 | solver.py:352-363, 771-799; dp_cosmo.py:179-217, 241-344 | Closed form on the complement of the uniform mode. Exact for any rate x dt: I verified it against expm to 3e-13 over 200 random rate sets (jumps.py) | test_relic_scattering.py (6 tests); test_decay.py::test_jumps_zeno_protected_by_decay_widths |
| T4 | Decay: -sum_i T_ia Gamma_i m_i/(E(1+z)) F_a. Daughters in mass state j at z = E/E_p in [0,1]: the kept half [f_T + (f_L - f_T)z] to the same kind, the flipped half [f_L + (f_T - f_L)z] to the other kind, split over eigenstates by T_jb at the daughter's bin | dp_decay.py:34-66 (rates), 84-168 (step); solver.py:386-411, 503-516, 549-554, 718-747 | Exact exponential per bin with optical depth I_k = int dz/((1+z)^2 H) (8-point GL). Strang halves around T1+T2; n_sub <= 64 alternations with T3 per half step. Top-down sweep, so chains run within a step | test_decay.py (9 tests); exact nu_3 survival 0.5257 vs 0.5266 (notes l.2280) |
| T5 | Relic mass-change loss: dE_obs/dz = C(z)(G G^dag)_aa/((1+z)^2 H). C = 2 n_hel (1+z)^3 sum_{r!=s} eta'_r \|G'_sr\|^2 (m_s^2 - m_r^2)/2 <1/E_r>_{T_r} / (4 pi M_t^2), with the *non-universal* G in both factors | dp_relic_loss.py:35-99; solver.py:365-384, 517-520, 555-556, 750-762 | Rigid shift of each bin, re-binned to conserve number. Two halves, each C(z_k) x I_k/2 | test_relic_loss.py (3 tests); 0.9458 vs 0.9455 (notes l.2285) |
| T6 | Relics after their own decays: eta' (heaviest first, by branching, time-dilated) | dp_decay.py:179-204; solver.py:260-268 (z_ref = 1); solver.py:1167-1174 (Galaxy, z_ref = 0) | 600-node trapezoid in ln(1+z) | test_decay.py::test_relic_composition, ::test_solver_uses_the_decayed_relics |
| T7 | Debye mass m_D^2(T_r) (Majorana, v-suppressed) | dp_absorption.py:113-152 | Used at T0(1+z) with eta' in T3 and T5. dp_si uses today's value at every z (solver.py:571-572). The Galactic emission uses the pre-decay eta (solver.py:1097) | test_absorption.py |
| T8 | dp_si, where routed (solver.py:802-847): s/t/u/pair collisions; quasi-elastic conversions at golden-rule rates (no Zeno); hard-loss drift; decays inside the column sweep (folded, flat in z in [0,1]); T5 in halves | dp_si.py:1789-2049, 2824-2835, 2980-3031; solver.py:560-599 | Exact per-bin matrix exponential, with in-bin conversions rescaled beyond 1e8 per step | test_si.py; test_relic_scattering.py::test_jumps_against_dp_si, ::test_stiff_bin_keeps_its_losses |
| T9 | Galaxy (z = 0, below 1 keV): T3-T5 along each line of sight. Per segment: L/2, [D/2, (J D)^n_sub], L/2 | dp_galactic_path.py:50-160, 224-305; solver.py:1142-1243 | Emission placed at bin centres. A vacuum fast path by SVD | test_galactic_path.py (vacuum = sweep; jumps conserve number) |

Order within one cosmological step k: [D J]_{1/2} L_{1/2} (T1+T2) [D J]_{1/2} L_{1/2} with decays on, or
L_{1/2} (T1+T2) J L_{1/2} with decays off. All rate tables are taken at z_k, the step's low-z end.

Verified correct, so not listed as findings:

- The Majorana rest rate, including the 12 Re(G'^2) m_i m_j cross term, re-derived from the trace.
- The kept/flipped angular mapping, the kinds of the flipped daughters, and the conjugations for nubar (U*, G*).
- The kappa-invariance of the dissipator D[G + kappa I] = D[G], so dropping kappa from G_j is exact.
- Gamma_u with the full G on the relic side.
- n_hel = 3 zeta(3) T^3/(4 pi^2) = 56 cm^-3, and 2 n_hel per Majorana mass state.
- (1+z)^3 n, T(1+z), w at E(1+z), the decay exponent Gamma m/E_obs x int dz/((1+z)^2 H), and the loss in E_obs with
  1/(1+z).
- KPC_EV_INV.
- Number conservation of T3, T4 and T5 except at the lower grid edge.
- No double counting: the loss adds only the energy cost of events already counted (at zero cost) in T3 and in
  dp_si profile 2/6, whose H_rr = (G G^dag)_rr sums over the relic's final states.
- The combined mode's rescaling cancels T3, T4 and T5 between the reference and dp_si.

## 2. Findings (most severe first)

### F1. BUG: the decay daughters' energy shares ignore the mediator's mass

Where:

- dp_decay.py:155-168: kernels on z in [0,1], own-bin share from z0 = E_lo/E_b.
- dp_si.py:1911-1918 and 1974-1981: flat on [0,1].
- docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md:295 ("at most (m_j/m_i)^2 = 3%").
- notes l.2189-2193 and 2318.

Reasoning. In the rest frame a massless daughter has E* = (m_i^2 - M^2)/(2 m_i). In flight,
z = (1 - M^2/m_i^2)(1 + cos theta)/2. In general z lies in [(E* - p*)/m_i, (E* + p*)/m_i], with
E* = (m_i^2 + m_j^2 - M^2)/(2 m_i). The angular distributions the notes checked are right, but they map onto
[0, z_max] with z_max = 1 - M^2/m_i^2, not onto [0, 1]. The rest of the energy goes to the mediator. The f_L used in
the code (1/(1 + 2M^2/m_i^2)) already assumes a massive M, so the two pieces are inconsistent. The documented bound
counts only m_j and misses M, which can approach m_i - m_j.

Size. The daughters' weight in an E^-p spectrum scales as z_max^(p-1) (checks.py):

| M^2/m_i^2 | 0.01 | 0.1 | 0.25 | 0.5 | 0.8 |
|---|---|---|---|---|---|
| true/code, p = 2.735 | 0.98 | 0.83 | 0.61 | 0.30 | 0.06 |
| true/code, p = 1.31 (below the break) | 1.00 | 0.97 | 0.92 | 0.81 | 0.61 |

This matters for M above about 0.3 m_i: m_Zd from about 0.015 to 0.05 eV for nu_3, and about 0.003 to 0.009 eV for
nu_2 at m_l = 1e-6. The thresholds move with the profiled m_lightest. The decays there are complete for g of about
1e-8 and above, so the beam arrives as nu_1 plus daughters, and the code overstates the daughters' flux by up to
2-3x. The flavor (nu_1) is unchanged and the normalization floats, so the effect on the chi^2 is mostly the spectral
shape across the break and the event-rate normalization. Not negligible: it is a decade-wide band in m_Zd.

Fix. Rescale z to [z_lo_ij, z_hi_ij] per (i, j) in decay_step: A and B become per-pair, the own-bin share runs from
max(z0, z_lo) to z_hi, and the running sums cover only bins below z_hi E_b. In dp_si, make the density flat on
[z_lo, z_hi]. Update docs item 22(b).

Check. decay_step with Gam[2,0] large and fL from rest_rates at M = 0.5 m_3: the code's mean daughter share (folded)
is 1/2, the true one is (1 - 0.25)/2 = 0.375. Then rerun the exact-survival comparison of notes l.2280 at
M = 0.03 eV.

### F2. BUG (analysis, dp_unitarity usage): the not-allowed boundary is drawn where it is narrowest, not widest

Where: analysis/exclusion.py:294-332 ("1e-6 eV: the widest region"), docs:304, notes l.2217.

Reasoning. Y_ij = i(G'_ij m_i + G'_ji m_j)/M grows with the masses, so a larger m_lightest gives a more restrictive
bound. With dp_unitarity.neutrino_unitarity and the code's own couplings (unit.py), the intercept c of
log10 g = log10 m_Zd + c is:

| m_l (eV) | 1e-6 | 1e-3 | 1e-2 | 0.03 | 0.1 |
|---|---|---|---|---|---|
| texture e | 2.459 | 2.438 | 2.160 | 1.760 | 1.250 |
| texture mu | 1.762 | 1.761 | 1.729 | 1.600 | 1.226 |

The fit profiles m_l over [1e-6, 0.1], with Sum m < 0.12 eV at 95%. A point just inside the drawn boundary has a
unitarity measure of 25 (e) or 2.1 (mu) at m_l = 0.03, and 262 (e) at m_l = 0.1, so the profile can sit on a
non-perturbative model. Marking at m_l = 1e-6 is "non-perturbative for every m_l"; that may be intended, but then the
text must say "narrowest", and the profile over m_l should be restricted to perturbative values.

Check. Compute c(m_l) per plane at the profiled best-fit m_l of each mass column. Mark the region per point, or
restrict the m_l profile.

### F3. BUG (latent; potential planes only): the relic mass-change loss leaves out kappa in the beam vertex

Where:

- dp_relic_loss.py:35-55 and its docstring.
- solver.py:371-384 (G_loss = dark_portal_coupling(g_D, u, 0)).
- solver.py:750-762 (GG table built from G_j = G - kappa I).
- solver.py:1150-1191 (Galaxy, G_nu without kappa).
- The notes' formula l.2147 is written for the full G' = g_D w w^dag + kappa I (l.2126-2127).

Reasoning. In beam a -> c with relic r -> s (s != r), the relic factor |G'_sr|^2 does not depend on kappa, but the
beam factor is sum_c |G_ca|^2 = ((G + kappa I)(G + kappa I)^dag)_aa. It includes kappa^2 + 2 kappa Re G_aa, because
the beam need not change state. Gamma_u of the jumps does include kappa, on its relic side, so the loss is
inconsistent with the jumps. dp_si's hard scatterings also use the full G (solver.py:567).

Size. At split 1e-8, texture e, m_Zd = 1e-12 eV, the loss over the path at z = 0 rates (checks.py) is:

| eps' | code | with kappa |
|---|---|---|
| 1e-4 | 1.3e8 eV | 1.1e16 eV |
| 1e-2 | 1.3e4 eV | 1.1e16 eV |
| 1 | 1.3 eV | 1.1e16 eV |

With kappa the beam is drained. Today this is irrelevant: every final plane below 1 keV has eps' = 0 (dp ul) or no
kappa (L_e - L_mu, L_e - L_tau use the full G_coupling), and the MeV planes have C of order 1/M^2, which is
negligible. It becomes fatal if the potential planes come back, and it touches the 'combined' mode's statement that
kappa dominates.

Fix. Use the full G in the beam factor: loss_beam and a GG table from the full G, or add kappa^2 + 2 kappa Gd.
Alternatively, raise an error for kappa != 0 below 1 keV.

### F4. SUSPECT (physics, no measurable effect in the allowed region): the Zeno form of the jump rates

Where: dp_cosmo.py:241-257 and 292-297; notes l.2076-2080 and 2268-2270; tests/test_decay.py:147-164.

Reasoning. Eliminating the coherences does not give R = Gamma J w^2/(w^2 + D^2) in general. It holds when the
pointer basis (the eigenbasis of G) is close to H's eigenbasis, and fails otherwise. I solved the exact Lindblad
equation (lind.py):

- **G = sigma_x.** The transfer a -> b stays at the golden-rule rate for any Gamma: 95 against 100 at Gamma = 100w,
  where the code gives 0.01. The H-basis populations equalize, and the coherence in G's basis freezes instead.
- **G nearly diagonal in H's basis.** The code's formula holds: 1.98e-4 against 1.96e-4.
- **A fast-decaying final state, weak scattering.** The exact loss rate of a is Gamma J for every W. The code
  suppresses it by (2w/W)^2 (W = 1000w: 4e-9 against 1e-3).

So "a state that decays fast is Zeno-protected from the conversions into it" is not physical for scattering-driven
transfers. A width damps the coherence feedback; it does not suppress the direct G rho G term. For the dark portal,
G' = g w w^dag mixes every mass state (the first case), and so does L_e - L_mu in the mass basis.

Why it does not move results:

- D >= w needs jump rates of at least about 1e-21 eV, which means more than 1e11 jumps per Hubble time, already
  saturated. Zeno freezing would need D/w > w t_H ~ 1e12-1e16, which is never reached in the planes.
- The width term needs Gamma_rest >~ m/2. At the drawn boundary, W_3/w_31 <= 0.18 (e) and 0.08 (mu), and
  W_2/w_21 <= 0.05, so the suppression is at most 1%.

Recommendation: drop Wd from D, or document the formula as valid only in the secular regime. The test
test_jumps_zeno_protected_by_decay_widths encodes the wrong physics.

### F5. SUSPECT: the mediator is taken as invisible, and the notes' reason for it is wrong

Where: notes l.2194 and 2432; docs item 22(b).

Reasoning. For m_Zd > 2 m_1 (most of the decay band at m_l = 1e-6), Z_D -> nu nu. Its lab decay rate
Gamma_Z m_Zd/E_Z exceeds H_0 for g of about 1e-6 at 1e-2 eV. The mediator carries on average half of the parent's
energy, not "the softest part". For p = 2.735 I estimate the weights per decay at:

- kept daughters: 0.275
- flipped daughters: 0.091
- the mediator's two neutrinos (x ~ 6x(1-x)): about 0.25

So including them would raise the regenerated flux by about 70%, mostly as nu_1 pairs: a normalization and shape
effect. This is already "to quantify" in stage 2; the justification in the notes should be corrected.

### F6. NOTE: inconsistent relic composition and Debye mass across parts (all immaterial in the allowed region)

- **Galactic emission.** It screens its potential with the pre-decay eta (solver.py:1097), while the path processes
  use eta' (1167-1175).
- **Galactic transfer.** _galactic_responses and the 'local' emission go through propagate_spectrum with
  arrival_flavor set. That skips the relic decays (solver.py:262), so the transit's Debye mass is pre-decay for the
  Galactic component and post-decay for the extragalactic one.
- **Absorption grid.** Gamma_grid_all comes from _flux_setup (solver.py:923) with the pre-decay eta, contrary to the
  comment at solver.py:260-261. The resonance lies far below the band where decays are open, so it has no effect.
- **dp_si's M_t.** dp_si takes M_t from today's Debye mass at every z (solver.py:571-572); the dark-portal phase uses
  m_D(z).
- **Why none of this matters.** In the allowed region m_D/m_Zd <~ 1e-2, because m_D ~ 2e-5 g and
  m_Zd > g 10^-c. Should the region change (F2), recheck.
- **z_ref.** relic_composition uses z_ref = 1 for the cosmology and 0 for the Galaxy. This is immaterial for a
  different reason than the docs give. The docs say "the planes' weakest coupling (1e-8)", but dp ul goes down to
  1e-10 and L_e - L_mu to 1e-34. The right reason: where T3 or T5 matter at all, the relic decay exponent D is
  necessarily >> 1. My estimate is D ~ 4e26 (g/M)^2, while a relic effect needs g^4/M^2 >~ 1e-19.

### F7. NOTE (numerics): shift_step's numerical diffusion

Where: dp_relic_loss.py:58-99.

On the campaign grid (439 bins, 400 shifts) against the exactly shifted power law (checks.py):

| Loss at 100 TeV | Flux at 100 TeV | Flux at 10 TeV |
|---|---|---|
| 5% | +0.14% | +1.1% (loss O(1) there) |
| 23% | +0.6% | +2.2% |

The scheme is first-order upwind with bin-average densities, and the bias is conservative (flux too high). Check:
n_E doubled at one edge point.

### F8. NOTE: no grid extension above E_max in the dark-portal phase and the Galaxy

dp_si has E_extension = 10. In the dark-portal phase and the Galaxy, the daughters (T4) and the loss inflow from
parents above 30 PeV are missing. The deficit is about (E/E_max)^p times the daughters' share: about 2% at 10 PeV,
and about 15% x share at E_max/2.

### F9. NOTE (splitting order)

- With decays on, the sequence [DJ]/2 L/2 C [DJ]/2 L/2 is not a palindrome, so the error is first order in [DJ, L].
- With decays off, the jumps run once, after C, which is first order. Switching decays on therefore also changes how
  the jumps are split.
- Gamma_u, C, W, T and c_z for the first half step come from z_k, not from the upper half.

The measured step convergence (0.9474/0.9471/0.9470 for 200/400/800 steps) shows the effect is small. Cheap fixes:
apply L/2 [DJ]/2 C [DJ]/2 L/2, and evaluate C(z) and Gamma_u at the half-step midpoints.

### F10. NOTE: the 'si' mode below 1 keV folds nu + nubar

In the dp ul planes, eps' = 0 and g_trace >= 1e-4 route to 'si'. There the flipped daughters' kind is lost: the
flux is split by injection weights (solver.py:593-599). This is exact for the sum, and for the per-kind fluxes only
at nubar/nu = 1, which is the configs' value.

With separate (1,0) and (0,1) runs (propagate_nu_nubar), the Earth transit and the event rates see nubar daughters
with nu cross sections. The mislabelling cancels only in the 1:1 sum.

### F11. NOTE: the jumps have no kinematic |t|_min cut

A transition a -> b needs |t| >= w_ab^2 (w ~ 1e-16 to 1e-21 eV), so sigma behaves as 1/(M_t^2 + w^2), not 1/M_t^2.
This matters only for M_t < w, which falls in the not-allowed region. In the allowed region there are fewer than
1e-2 jumps per path there.

### F12. NOTE: dead code

dp_si._decay_columns (dp_si.py:2809-2821) is no longer called.

## 3. Recommended checks before the production run, in order

1. F1: fix the kinematics. Check the mean daughter share and rerun one complete-decay point at m_Zd = 0.03 eV, plus
   the notes' exact-survival comparison.
2. F2: recompute the not-allowed boundary per mass column at the profiled m_l, and decide whether to restrict the
   m_l profile. Correct "widest".
3. F3: use the full G in the loss's beam factor, or guard against kappa != 0 below 1 keV.
4. F4: remove Wd from D, or leave it and record why it is immaterial. Fix the test's claim.
5. F7: run one edge point with n_E doubled to bound the loss diffusion.
6. F5: quantify the mediator's re-decay at one point in the decay band (stage 2).
