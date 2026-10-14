# Physics audit of the solver and analysis (2026-10-14)

Requested by Mauricio, 2026-10-14: "Before we make more runs, I need you to double-check every part of the physics
code for correctness" (no agents). No new runs until it is done; the task-2 reruns already on Tycho and the electron
emission-point table finish, their results on hold. Each item: what the code does, the derivation or check, the verdict.
Verdicts: OK; ERROR (wrong physics, to fix); APPROX (a stated or unstated approximation, to keep or improve); NOTE
(convention or assumption to state in the paper).

Order: 1 absorption and regeneration; 2 cosmological propagation; 3 Hamiltonian, potentials, conventions; 4 the local
transit (Galaxy, Sun, Earth's exterior and chords, zenith average); 5 matter models; 6 flux assembly, sky, Galactic
component; 7 secret interactions (dp_si); 8 event rates and likelihoods.

## 1. Resonant absorption on relic neutrinos and regeneration (src/dp_thermal.py, src/dp_cosmo.py)

| # | Item | Verdict |
|---|---|---|
| 1a | Relic-state counting: every relic mass state gets the incident flavor's full coupling (Sum G)_alpha (T7/T12) | ERROR, decided: weight by the state's own components |
| 1b | Width normalization: T11 Gamma = Sum G M/12pi is the vector-Dirac width; active neutrinos are left-handed, M/24pi with the T9 coupling G = g_D U U^dag | ERROR, decided: Lagrangian coupling throughout |
| 1c | Incident eigenstates and regenerated secondaries are projected on flavor incoherently (|W_fa|^2); the amplitudes are coherent | ERROR (new) |
| 1d | Only the s-channel resonance; t, u channels and Z_D Z_D production omitted | APPROX (new) |
| 1e | Flux factor (1 - cos theta) for every relic mass (T7's massless form); the Moller factor is 1 - v cos theta | APPROX (new), O(v) |
| 1f | T12's eps' term: cross term c * eps' g/c_W on every flavor pair, no eps'^2 term | ERROR, decided: all three pieces |
| 1g | 56 relics per helicity and mass state as targets: Majorana relics (Dirac halves the rates) | NOTE |
| 1h | Secondary spectra 3x^2 (same kind) and 3(1-x)^2 (other kind) | OK |
| 1i | No visible decay channels (e+e- through the Z mixing, or gauge couplings to electrons) in the width | ERROR (new), decided with 1f |
| 1j | Debye mass (T23): the relativistic thermal mass, flavor-diagonal |u_alpha|^4. Today's relics are slow, and the mediator couples to the left-handed current: a Majorana relic's charge is -h v (its vector current vanishes), so slow relics screen less, by ~3.3 T/m (a Dirac relic's charge is (1 + v)/2) | APPROX; fixed (see below) |
| 1k | Step bookkeeping: exact exponential absorption per step, number-conserving bin integrals, injection weighting (1+z) per unit observed energy, SFR of Yuksel et al. (2008) | OK |
| 1l | Kernel normalization: Gamma = int d^3p f sigma v / (2pi)^3, 56 cm^-3 per state; sigma_phys = sigma_raw Sum G / 6 pi^2 | OK |

**1a, 1b (and 1i, 1f).** For the Lagrangian coupling G (flavor basis, coefficient of nubar_alpha gamma^mu P_L nu_beta),
the amplitude of nu_alpha + nubar_i -> Z_D is (G^T U*)_alpha i, so |(G U)_alpha i|^2; summed over the three states it is
(G G^dag)_alpha alpha. The Breit-Wigner of a vector formed by nu_L + nubar_R integrates to int sigma ds = 48 pi^2
Gamma_in / M with Gamma_in = |.|^2 M / 24 pi, i.e. 2 pi |(G U)_alpha i|^2 per target -- the s channel of dp_si's vector
(brute-force amplitudes) and the Z-burst <sigma> = 2 sqrt2 pi G_F. The kernel gives 4 pi (Sum G)_alpha per target on
each of the three states: ~x6 too much absorption (x3 counting, x2 width). Checked numerically
(tools/absorption_relic_weights.py: a flavor-diagonal coupling absorbs nu_e on m1, m2, m3 in the ratio 1 : 1.14 :
1.01 instead of |U_ei|^2 = 0.68 : 0.30 : 0.02). My 2026-10-12 normalization (944360d) took T11 at face value; its
Z-burst claim holds only with the Z coupling divided by sqrt2.

**1c.** An incident eigenstate a has flavor amplitudes W_fa (W: eigenvectors of the Hamiltonian at (E, z)); its
amplitude on relic state i is (U^dag G W)_ia, which in vacuum (W = U) is the mass-basis coupling. The code uses
Sum_f |W_fa|^2 Gamma_f: wrong whenever the coupling is not diagonal in the eigenbasis -- e.g. a flavor-diagonal
coupling in vacuum absorbs mass state 1 only on relic 1, while the flavor projection spreads it over all three. The
regenerated neutrino likewise enters eigenstate a with probability (W^dag G G^dag W)_aa / Tr(G G^dag), not
Sum_f |W_fa|^2 (G G^dag)_ff / Tr. Both are functions of sigma = E c_z only (W depends on sigma), so they tabulate on the
sigma nodes like the present projections.

**1d.** dp_si's vector transport has every channel; for a vector the t channel is forward-peaked above the resonance,
sigma -> |G|^4 / 4 pi M^2, and gives the "quasi-elastic" limits of the vector SI planes below the window. Without a
potential (the MeV planes, eps' = 0; the gauge models at MeV masses, whose cosmological potential is ~1e-39 eV and whose
Earth potential is flavor-diagonal) the dark-portal and gauge-model absorption IS a vector secret interaction with
coupling G, which dp_si already computes (plus visible width).

**1e.** Direct quadrature of the kernel's closed form: I_ang = (1/2pi) int_{-1}^{1} sigma_raw(s) (1 - c) dc to 1e-15 for
m = 1e-5 ... 5e-2 eV. With the Moller factor (1 - v c), v = p / E_bg, the energy-integrated absorption changes at O(v)
(v ~ 0.05 for m = 0.01 eV); the dip wings by up to 4x for m ~ 1e-3 eV.

**1c, an absolute check.** A flavor-universal coupling (the Z's) is diagonal in the mass basis, so in vacuum mass state
j is absorbed only on relic state j (the relic-neutrino absorption dips of Eberle, Ringwald, Song & Weiler); the flavor
projection spreads it over all three. In the potential-dominated regime of a rank-one texture, the two eigenstates
orthogonal to u do not couple to Z_D at all ((U^dag G W)_ra = 0), yet the projection absorbs them.

## 2. Cosmological propagation (src/dp_cosmo.py, propagate_spectrum)

| # | Item | Verdict |
|---|---|---|
| 2a | H(z) = H0 sqrt(Omega_m (1+z)^3 + Omega_L); dt = dz / (H (1+z)) | OK |
| 2b | Source evolution: Yuksel et al. (2008) SFR, breaks at z = 1 and 4 (continuous), times (1+z)^rho_evol | OK |
| 2c | Injection per unit observed energy carries (1+z) (fixed 124fd03): flux today = int dz R Phi_0 (E(1+z))^-p / H | OK |
| 2d | Populations of the instantaneous eigenstates, coherences dropped; adiabatic in z (oscillation lengths ~1e10 m vs the Hubble length) | OK |
| 2e | Eigenvectors depend on sigma = 2E(1+z)A(z) only (scale invariance); tabulated, frame_at accurate at any sigma | OK |
| 2f | Injected flavors split over the eigenstates at the source's potential (host galaxy option) | OK |
| 2g | Uniform medium's potential cut at the Hubble radius: (1 - (1+x)e^-x)/m^2, x = m R_H (as ABDN23) | OK (a choice; stated) |

## 3. Hamiltonian, potentials and conventions (dp_core, dp_spatial.lri_coefficient, the cosmological coefficient)

| # | Item | Verdict |
|---|---|---|
| 3a | PMNS: PDG parametrization (R23 U13(delta) R12) | OK |
| 3b | Antineutrinos: U -> U*, V -> -V*, applied once, centrally | OK (tests) |
| 3c | T9 coefficient (4 c_W m_Z^2 / g)(G_F / sqrt2) = g/(2 c_W): 0.3713 with the code's c_W = 0.88, m_Z, g = 0.65, G_F (0.3693 from g/2c_W: the four constants are 0.5% mutually inconsistent) | NOTE |
| 3d | eps' convention: half Davoudiasl-Lee-Marciano's eps_Z (neutron coupling eps' g/2c_W, nu_L coupling eps' g/c_W) | NOTE (self-consistent) |
| 3e | Inverted ordering: masses read the configured splitting as Delta m^2_32, the vacuum Hamiltonian as Delta m^2_31 (differ by Delta m^2_21, 3%) | ERROR (minor; IO only) |
| 3f | Only P = g_D eps'|U|^2 > 0 is scanned for the dark portal (and g' eps' for L_mu - L_tau); the sign is free. Flipping it equals delta -> -delta for the nu + nubar sum, but not for HESE (Glashow: nubar_e) | APPROX (scope) |
| 3g | Standard matter potential (off by default; flavor-diagonal, negligible above 10 TeV) uses one constant n_e, not PREM | NOTE |

## 4. Local transit (src/dp_transit.py, src/dp_spatial.py)

| # | Item | Verdict |
|---|---|---|
| 4a | Galaxy adiabatic (potential varies over >= 10 pc, oscillation lengths <= 0.005 pc); local Yukawa-smoothed density | OK (test T9) |
| 4b | Sun: coherent transit along Gauss-Legendre angles, exact in the adiabatic and sudden limits | OK (brute-force tests) |
| 4c | Earth's exterior and chords: Yukawa integral of PREM, exact step averages, the 'phase' zenith rule | OK (tests) |
| 4d | End-to-end potential-dominated limit (new check, 2026-10-14): flavor at the Earth = sum_a |W_alpha a|^2 sum_f |W_fa|^2 f_f with W the texture's eigenvectors (null space split by the projected vacuum term); the solver at m_Zd = 1e-20 eV agrees to 7e-11 (e-mu), 3e-11 (mu-tau), 1e-11 (democratic), nu and nubar | OK |
| 4e | Detector depth 1.5 km (top of the array; its centre is ~1.95 km) | NOTE (negligible) |

## 5. Matter models (src/dp_spatial.py, dp_core, dp_matter)

Cosmic n_n0 = n_b Y_p/2 and n_e0 = n_b(1 - Y_p/2); Galaxy (McMillan 2011 bulge and discs, Miller & Bregman 2013 halo,
McMillan 2017 HI/H2) with neutron and electron fractions per component; Sun, Moon, Earth counts; PREM with Y_e by
layer. OK (tests: surface densities, counts against masses, layer fractions; step 1's linearity tests).

## 6. Flux assembly, sky and Galactic component (dark_portal_solver, dp_sky, dp_galactic, flux_postprocessing)

| # | Item | Verdict |
|---|---|---|
| 6a | nu / nubar injection weights, coupled evolution through the secondaries | OK |
| 6b | Bands in declination, solid-angle weights, absolute normalization (extragalactic_scale) | OK (HESE consumer check to 2e-14 against IceCube's code) |
| 6c | Flavor fractions number-weighted over 10 TeV-30 PeV; IceCube's flavor measurements are event-weighted under a power law | NOTE (the event likelihoods do not have it) |
| 6d | Galactic emission in the local eigenstates, adiabatic to the Sun | OK (test T9) |

## 7. Secret interactions (src/dp_si.py)

Mass-basis couplings, per relic state and helicity, every channel (s, t, u, VV), widths M sum|G|^2/24pi (vector) and
the scalar's, Majorana and Dirac. Brute-force amplitudes, nuSIprop, 2107.13568 Figs. 3 and 5. OK (not re-derived).
Its vector transport with G = g_D u u^dag is exactly the dark portal without potential (eps' = 0).

## 8. Event rates and likelihoods (IceCubeRates, analysis)

| # | Item | Verdict |
|---|---|---|
| 8a | HESE: IceCube's MC (all flavors, CC/NC/GR), our flux per flavor and kind | OK (2e-14 against IceCube's code) |
| 8b | DR2 tracks: the release's nu_mu + nubar_mu effective areas only; nu_tau -> tau -> mu (17% BR, softer muons, ~3-4% of the rate for equal fluxes) absent | APPROX (flavor-dependent, small) |
| 8c | Poisson likelihoods, nuisances, priors; standard fits reproduce IceCube (gamma 2.85 vs 2.87) | OK |

## Decisions (Mauricio, 2026-10-14) and fixes

| Item | Decision | Fix (code) | Checks |
|---|---|---|---|
| 1a | each relic mass state by its own flavor content | `dp_absorption`, `dp_cosmo._build_sigma_coupling_tables` | Z-burst normalization; universal coupling absorbs mass state j only on relic j |
| 1b | the left-handed width M/24pi with the Lagrangian coupling (T9's) | `dp_absorption.AbsorptionSetup` (sum_G = Tr(G G^dag)/2 + 12 pi Gamma_vis / M) | old form 6x the new per target, diagonal coupling |
| 1c | coherent amplitudes per eigenstate (the correct form of 1a) | weights (U^dag G W)_ra, (W^dag G G^dag W)_aa tabulated on the sigma nodes; `_cosmo_coherent_rates` | sum rules; eigenstates orthogonal to u untouched in the potential-dominated limit |
| 1f, 1i | all three pieces: G = g_D u u^dag + eps' g/c_W I; e+e- and mu+mu- in the width | `dark_portal_coupling`, `z_mixing_visible_width` | width formula (the Z's 83 MeV); Gamma_nu/Gamma fractions |
| 1e | the Moller flux factor 1 - v cos(theta) | `relic_state_rates_analytic(..., moller=True)` | against 50-digit / dense quadrature |
| 1j | the Debye mass of slow relics, mass-basis couplings, the Z-mixing piece | `dp_absorption.debye_mass_sq` (Majorana relics, as the absorption's targets; Dirac as an option): per state |G'_rr|^2 x 2 x (1/2 pi^2) int f(p/T) d[p E q^2]/dp dp with q = v (Majorana) or (1 + v)/2 (Dirac). The first version (f7f3523) gave every relic the full vector charge, overstating today's screening mass squared by 2-4 orders of magnitude; corrected the same day | relativistic T^2/6 (both); slow relics 9 zeta(3) T^3/(2 pi^2 m) (Majorana), m T ln2/(4 pi^2) (Dirac); direct quadrature at m = 3T |
| screening | only localized bodies (Earth, Moon, Sun, Galaxy, host galaxy); the uniform cosmological potential unscreened (no gradient, no force on the relics; no reservoir of dark charge) | cosmological coefficient with the bare mass; `transit_to_earth(m_bare=...)` and the emission points add the unscreened cosmic mean | test_screening_spares_the_uniform_cosmic_potential |
| 1d | the MeV regime through dp_si (every collision) | `_use_si_cosmology` ('auto': cosmological potential < 1e-4 of Delta m^2_21 at every E, z), `dp_si.propagate_si(thermal=True, Gamma_vis=...)`, `_thermal_resonance_terms` | two codes' resonance agree to 0.1% (median) at 3 MeV; relics at rest overstate the dip 2x |
| 3e | (technical) | inverted ordering: the vacuum term's m_3^2 - m_1^2 = -(|Delta m^2_32| - Delta m^2_21), as the masses read the splitting | -- |
| 3f | both signs of the potential (dark portal, L_mu - L_tau) | the sign of g_D eps' folded into the cosmological table | test_negative_potential_sign_is_the_flipped_texture |

Consequences: with eps' = 0.1 the Z mixing alone couples the mediator to every neutrino with 0.074, absorbing at MeV masses
and giving a screening mass ~2e-6 eV (a range of ~12 cm) that cuts off every ultralight local potential -- "u = 0" is no longer "no new
physics" unless eps' = 0 too (tests updated). The potential planes therefore profile eps' (decided), which needs
physical upper limits on eps' and g_D |U|^2 (fifth-force tests, neutrino self-interactions): to settle at the profile
setup. absorption_model 'theory_notes' keeps the old form for comparisons.
