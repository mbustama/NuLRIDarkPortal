"""
test_physics_regression.py

Regression tests for the dark-portal physics kernels (src/dp_*.py, re-exported
through the src/dark_portal_solver.py facade -- tests access everything as dps.<name>;
NOTE monkeypatches must target the OWNER module, e.g. dp_spatial, not the facade).

These pin down two things that are easy to silently break in a future refactor and hard
to notice from the final flux plots alone:

1. The closed-form (exact, analytic) angular integral used inside
   thermal_averaged_rates_flavor to evaluate the Breit-Wigner resonance without
   "pole-skipping" at small couplings. It is checked against brute-force numerical
   quadrature over cos(theta), both in isolation and end-to-end through the real,
   JIT-compiled thermal_averaged_rates_flavor function.

2. The neutrino/antineutrino conjugation relation of the dark-portal LRI potential
   (V_LRI_antineutrino = -conj(V_LRI_neutrino)), and the fact that the cosmological and
   spatial-handover code paths must agree with each other. This is the double-conjugation
   bug that was silently dormant only because the CLI/JSON interface restricts the sterile
   mixing elements (u_e, u_mu, u_tau) to real numbers; it corrupts antineutrino flavor
   phases the moment a complex (CP-violating) mixing vector is used.

Run with: python3 -m pytest tests/test_physics_regression.py -v
"""

import os
import sys
import json
import subprocess

import numpy as np
import pytest
from scipy import integrate

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_FREQ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FrequentistAnalysis"))
if _FREQ_DIR not in sys.path:
    sys.path.insert(0, _FREQ_DIR)

import dark_portal_solver as dps  # noqa: E402


# =============================================================================
# 1. Resonance angular integral vs. brute-force quadrature
# =============================================================================

def _closed_form_I_ang(A, B, M2, MG):
    """Verbatim re-implementation of the closed-form angular integral computed inside
    thermal_averaged_rates_flavor (src/dark_portal_solver.py)."""
    MG_sq = MG**2
    u_plus = A - M2 + B
    u_minus = A - M2 - B
    C = 1.0 - (A - M2) / B
    D = 1.0 - (A - 2.0 * M2) / B
    F = C * M2 - MG_sq / B
    return (
        (2.0 / B)
        + (D / (2.0 * B)) * np.log((u_plus**2 + MG_sq) / (u_minus**2 + MG_sq))
        + (F / (B * MG)) * (np.arctan(u_plus / MG) - np.arctan(u_minus / MG))
    )


def _quadrature_I_ang(A, B, M2, MG):
    """Brute-force integral, over cos(theta) in [-1, 1], of the exact integrand the
    closed form above is the antiderivative of: 1/B + (D*u + F)/(u^2 + MG^2), with
    u = A - B*cos(theta) - M2. (See the derivation note below for how this was
    reverse-engineered from the closed form.)

    This is NOT the same as the naive Breit-Wigner x (1 - cos(theta)) weight one might
    guess from the physics notes alone -- the extra structure comes from the full
    (mass-dependent) numerator of the resonant cross section. Reproducing that numerator
    here, rather than assuming a simplified form, is what makes this a meaningful check:
    it fails immediately if a future edit changes the closed-form coefficients (C, D, F)
    without correspondingly updating what physical integrand they represent.
    """
    MG_sq = MG**2
    C = 1.0 - (A - M2) / B
    D = 1.0 - (A - 2.0 * M2) / B
    F = C * M2 - MG_sq / B

    def integrand(c):
        u = A - B * c - M2
        return 1.0 / B + (D * u + F) / (u**2 + MG_sq)

    # For a narrow resonance (small MG) the integrand is a sharp spike at u=0, i.e. at
    # cos(theta) = (A - M2) / B; scipy's adaptive quadrature can miss it entirely unless
    # told where to look (this is precisely the "pole-skipping" the closed form in
    # src/dark_portal_solver.py exists to avoid -- see its module docstring/README).
    c_star = (A - M2) / B
    points = [c_star] if -1.0 < c_star < 1.0 else None
    val, _ = integrate.quad(integrand, -1.0, 1.0, points=points, limit=200)
    return val


@pytest.mark.parametrize("seed", range(8))
def test_angular_integral_closed_form_matches_quadrature(seed):
    """The closed-form antiderivative must exactly reproduce brute-force quadrature of
    its own implied integrand, across a spread of physically-plausible kinematic points."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(5.0, 50.0)
    B = rng.uniform(0.5, 10.0)
    M2 = rng.uniform(0.1, 30.0)
    MG = rng.uniform(1e-3, 5.0)

    analytic = _closed_form_I_ang(A, B, M2, MG)
    numeric = _quadrature_I_ang(A, B, M2, MG)

    assert np.isclose(analytic, numeric, rtol=1e-8, atol=1e-10)


def _reimplemented_thermal_rate(E0, z, sum_G, sum_G_alpha, m_Zd, m_nu_bg_arr, T_nu0, eta_mass, n_p=100):
    """Plain-Python (non-numba) re-implementation of thermal_averaged_rates_flavor, using
    the closed-form angular integral (validated independently above) with the exact same
    momentum grid, Fermi-Dirac weighting and flavor bookkeeping as
    src/dark_portal_solver.py. This is deliberately NOT built on adaptive quadrature: for a
    narrow resonance the per-momentum integrand is a sharp near-delta-function spike, which
    is exactly the "pole-skipping" failure mode the closed form exists to avoid (see the
    module docstring / README), so scipy.integrate.quad is not a reliable ground truth here
    -- a hand transcription of the same closed form is. What this test catches is a wiring
    bug (wrong sign, wrong prefactor, off-by-one in the momentum grid, wrong flavor
    indexing) introduced when editing the numba kernel, since the two implementations are
    written independently from the same formulas but never share code.
    """
    T_nu = T_nu0 * (1.0 + z)
    n_nu_base = 56.0 * (1.0 + z)**3 * dps.cm_inv3_to_eV3

    Gamma_width = (sum_G * m_Zd) / (12.0 * np.pi)
    MG = m_Zd * Gamma_width
    M2 = m_Zd**2

    rates = np.zeros(3, dtype=np.float64)
    if MG == 0.0:
        return rates

    prefactor = 1.0 / (2.0 * np.pi)

    for i in range(3):
        if eta_mass[i] <= 0:
            continue

        n_nu_i = n_nu_base * eta_mass[i]
        m_i_sq = m_nu_bg_arr[i]**2
        T_i = T_nu * (eta_mass[i]**(1.0 / 3.0))
        E_inc = E0 * (1.0 + z)

        p_max = 20.0 * T_i
        dp = p_max / n_p
        p = (np.arange(n_p) + 0.5) * dp
        E_bg = np.sqrt(m_i_sq + p**2)
        f_FD = 1.0 / (np.exp(p / T_i) + 1.0)
        p2_f_FD = (p**2) * f_FD

        A = m_i_sq + 2.0 * E_inc * E_bg
        B = 2.0 * E_inc * p

        I_ang = _closed_form_I_ang(A, B, M2, MG)

        term = prefactor * p2_f_FD * np.maximum(I_ang, 0.0) * dp
        integral_sum = np.sum(term)

        for alpha in range(3):
            rates[alpha] += integral_sum * sum_G_alpha[alpha] * (n_nu_i / n_nu_base)

    return rates


@pytest.mark.parametrize("E0_eV,z,m_Zd_eV", [
    (1e13, 0.5, 2.0e6),
    (5e14, 2.0, 5.0e5),
    (3e13, 3.0, 3.0e6),
])
def test_thermal_averaged_rates_matches_reimplementation(E0_eV, z, m_Zd_eV):
    """End-to-end check of the real, JIT-compiled thermal_averaged_rates_flavor against an
    independently hand-transcribed Python version built on the same closed-form angular
    integral. A future change that breaks the closed form's wiring into the momentum sum
    (wrong sign, wrong prefactor, mismatched u/M2 convention, ...) will show up here even
    if it wouldn't be caught by test_angular_integral_closed_form_matches_quadrature alone.
    """
    g_D, eps_prime = 1e-4, 0.1
    u_mag = np.array([0.5, 0.5, 0.0])
    sum_G = 0.0
    sum_G_alpha = np.zeros(3)
    for a in range(3):
        for g in range(3):
            coupling_prod = u_mag[a] * u_mag[g]
            val = g_D * coupling_prod * (g_D * coupling_prod + eps_prime * (0.65 / 0.88))
            sum_G += val
            sum_G_alpha[a] += val

    m_nu_bg_arr = np.array([0.0, 8.6e-3, 5.01e-2])
    T_nu0 = 1.95e-4
    eta_mass = np.array([1.0, 1.0, 1.0])

    rates_jit = dps.thermal_averaged_rates_flavor(E0_eV, z, sum_G, sum_G_alpha, m_Zd_eV, m_nu_bg_arr, T_nu0, eta_mass)
    rates_reimpl = _reimplemented_thermal_rate(E0_eV, z, sum_G, sum_G_alpha, m_Zd_eV, m_nu_bg_arr, T_nu0, eta_mass)

    # rtol history: 1e-9 while the JIT kernel's transcendentals were plain libm. With
    # an SVML-enabled llvmlite (2026-07 environment change), fastmath kernels emit
    # SVML's fast variants, which differ from this numpy reimplementation's libm at
    # up to ~5e-7 on this integrand -- three orders below the midpoint kernel's own
    # accuracy (~1e-3 significance-weighted vs dense truth; 6-9% at resonance cells).
    # 1e-5 still pins the closed form itself (historical bug classes were >= 1e-3).
    np.testing.assert_allclose(rates_jit, rates_reimpl, rtol=1e-5, atol=0.0)


# =============================================================================
# 2. Antineutrino LRI potential conjugation (regression test for the fixed bug)
# =============================================================================

def test_antineutrino_LRI_conjugation_cosmological_path():
    """V_LRI(antineutrino) must equal -conj(V_LRI(neutrino)) (theory notes, Eq. 9-10),
    for a *complex* (CP-violating) sterile mixing vector. This is exactly the case that
    was silently broken by a double-conjugation bug in get_adiabatic_mixing_cosmo: the
    bug only manifested for complex u_arr, which is why it was dormant for every shipped
    config (all of which use real u_e/u_mu/u_tau)."""
    u_base = np.array([0.5, 0.5 * np.exp(1j * 0.7), 0.1 * np.exp(-1j * 1.3)], dtype=np.complex128)
    m_Zd, g_D, eps_prime, n_n0, T_nu0 = 2.0e6, 1e-4, 0.1, 2.5e-13, 1.95e-4
    eta_bg = np.ones(3)
    z = 1.0

    S_nu = np.ascontiguousarray(np.outer(u_base, np.conjugate(u_base)))
    S_nubar = np.ascontiguousarray(np.outer(np.conjugate(u_base), u_base))
    V_nu = dps.calc_V_LRI_cosmo(z, u_base, S_nu, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
    V_nubar = dps.calc_V_LRI_cosmo(z, np.conjugate(u_base), S_nubar, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
    V_nubar = -V_nubar  # matches the (fixed) sign-flip-only convention in get_adiabatic_mixing_cosmo

    np.testing.assert_allclose(V_nubar, -np.conjugate(V_nu), rtol=1e-12, atol=1e-30)


def test_get_adiabatic_mixing_cosmo_diagonalizes_correct_antineutrino_hamiltonian():
    """get_adiabatic_mixing_cosmo (the cosmological-propagation path) must return
    eigenvectors of H_vac + V_LRI(antineutrino), where V_LRI(antineutrino) is the
    physically-correct -conj(V_LRI(neutrino)) -- not the double-conjugated potential the
    original bug silently substituted for real-only u_arr (where the two coincide, which
    is exactly why the bug shipped unnoticed)."""
    u_base = np.array([0.4 * np.exp(1j * 0.3), 0.6, 0.2 * np.exp(-1j * 0.9)], dtype=np.complex128)
    m_Zd, g_D, eps_prime, T_nu0 = 3.0e6, 2e-4, 0.05, 1.95e-4
    eta_bg = np.ones(3)
    n_n0 = 2.5e-13
    z = 0.5
    E0 = 1e13

    U_pmns = np.eye(3, dtype=np.complex128)
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = U_pmns @ M_sq @ np.conjugate(U_pmns).T
    H_vac = (1.0 / (2.0 * E0 * (1.0 + z))) * M_rot

    S_nu = np.ascontiguousarray(np.outer(u_base, np.conjugate(u_base)))
    V_nu = dps.calc_V_LRI_cosmo(z, u_base, S_nu, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
    V_nubar_correct = -np.conjugate(V_nu)
    H_tot_reference = H_vac + V_nubar_correct

    # Antineutrino call: the caller pre-conjugates u_arr AND the structure matrix
    # (see propagate_spectrum), as production code does.
    S_nubar = np.ascontiguousarray(np.conjugate(S_nu))
    evecs = dps.get_adiabatic_mixing_cosmo(E0, z, M_rot, np.conjugate(u_base), S_nubar, m_Zd, g_D, eps_prime,
                                           n_n0, T_nu0, True, True, eta_bg)

    # evecs must diagonalize H_tot_reference: evecs^dagger @ H_tot_reference @ evecs is diagonal.
    diagonalized = np.conjugate(evecs).T @ H_tot_reference @ evecs
    off_diag_mask = ~np.eye(3, dtype=bool)
    assert np.max(np.abs(diagonalized[off_diag_mask])) < 1e-10 * np.max(np.abs(np.diag(diagonalized)))


def test_cosmological_and_spatial_LRI_conjugation_agree():
    """The cosmological-phase potential (get_adiabatic_mixing_cosmo/calc_V_LRI_cosmo) and
    the spatial-handover potential (propagate_earth_spatial's Coef * U_LRI_struct) must
    transform identically between neutrinos and antineutrinos -- they are two independent
    code paths for the same physical potential, and previously disagreed for complex u_arr."""
    u_base = np.array([0.4 * np.exp(1j * 0.3), 0.6, 0.2 * np.exp(-1j * 0.9)], dtype=np.complex128)
    m_Zd, g_D, eps_prime, T_nu0 = 3.0e6, 2e-4, 0.05, 1.95e-4
    eta_bg = np.ones(3)
    n_n0 = 2.5e-13
    z = 0.5

    # Cosmological path: single sign flip applied to a potential built from pre-conjugated u_arr.
    S_nu = np.ascontiguousarray(np.outer(u_base, np.conjugate(u_base)))
    S_nubar = np.ascontiguousarray(np.conjugate(S_nu))
    V_cosmo_nu = dps.calc_V_LRI_cosmo(z, u_base, S_nu, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
    V_cosmo_nubar = -dps.calc_V_LRI_cosmo(z, np.conjugate(u_base), S_nubar, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)

    # Spatial-handover path: Coef flips sign only; U_LRI_struct is built from whatever u_arr
    # is passed in. Coef itself excludes the number density (propagate_earth_spatial applies
    # it separately as Coef * n_tot at each spatial step), so it must be applied here too --
    # calc_V_LRI_cosmo, in contrast, bundles n_n_z = n_n0*(1+z)^3 directly into its own A_LRI.
    c_W, m_Z, g_weak = 0.88, 91.18e9, 0.65
    m_D_sq = dps.calc_debye_mass_sq(g_D, u_base, T_nu0, eta_bg)
    Coef = (g_D * eps_prime) / (m_Zd**2 + m_D_sq) * (4.0 * c_W * m_Z**2 / g_weak) * (np.sqrt(2.0) / 2.0 * dps.G_F)
    n_n_z = n_n0 * (1.0 + z)**3

    def U_LRI_struct(u_arr):
        M = np.zeros((3, 3), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                M[i, j] = u_arr[i] * np.conjugate(u_arr[j])
        return M

    V_spatial_nu = (Coef * n_n_z) * U_LRI_struct(u_base)
    V_spatial_nubar = (-Coef * n_n_z) * U_LRI_struct(np.conjugate(u_base))

    np.testing.assert_allclose(V_cosmo_nu, V_spatial_nu, rtol=1e-10)
    np.testing.assert_allclose(V_cosmo_nubar, V_spatial_nubar, rtol=1e-10)


# =============================================================================
# 3. n_e_earth_cm3 wiring (regression test for the fixed dead-parameter bug)
# =============================================================================

def test_n_e_earth_cm3_affects_flux_when_SM_potential_enabled(tmp_path):
    """standard_mixing.n_e_earth_cm3 must actually influence V_SM once use_SM_potential
    is on; previously it was parsed and threaded as far as propagate_spectrum and then
    silently dropped, so changing it had zero effect on any output."""
    import copy

    cfg = dps.create_default_config(str(tmp_path / "test_config.json"))
    cfg["grid"]["n_E_bins"] = 20
    cfg["grid"]["n_z_bins"] = 20
    cfg["grid"]["n_mw_paths"] = 1
    cfg["grid"]["n_earth_chords"] = 1
    cfg["grid"]["N_steps_MW"] = 5
    cfg["execution"]["verbose"] = 0
    cfg["execution"]["n_cores"] = 1
    cfg["standard_mixing"]["use_SM_potential"] = True

    cfg_lo = copy.deepcopy(cfg)
    cfg_lo["standard_mixing"]["n_e_earth_cm3"] = 1.0e23
    cfg_hi = copy.deepcopy(cfg)
    cfg_hi["standard_mixing"]["n_e_earth_cm3"] = 5.0e24

    _, F_lo = dps.run_solver(cfg_lo)
    _, F_hi = dps.run_solver(cfg_hi)

    assert not np.allclose(F_lo, F_hi)


# =============================================================================
# 4. Earth-transit step count must not depend on chord length
# =============================================================================

def test_earth_transit_step_count_is_uniform_across_chords():
    """propagate_earth_spatial must use the same N_steps_earth for every chord regardless
    of its (very different) baseline length.

    A tempting-looking optimization -- give every chord the same *physical* km/step as the
    longest (core-crossing) chord, so much shorter chords (e.g. a grazing costh=0 chord can
    be ~90x shorter) get proportionally fewer steps -- was tried, and looked safe under a
    synthetic, purely-diagonal test state (converged to 1e-12 against a 2000-step
    reference). At production defaults, with the real off-diagonal-coherence state the
    Milky Way phase actually produces, it silently introduced an energy-localized ~10%
    error for the shortened chords, concentrated exactly at an MSW-like matter-resonance
    crossing inside the Earth -- near such a crossing, eigenvectors rotate rapidly with
    *position*, independent of total path length, so a shorter chord is not entitled to
    fewer steps. This test inspects the actual step counts propagate_earth_spatial hands to
    each chord and fails if they are ever allowed to differ by chord again without a
    resonance-aware justification.
    """
    # NOTE: since the 2026-07 module split, propagate_earth_spatial resolves
    # _earth_chord_worker from dp_spatial's namespace -- patching the facade's
    # re-export would silently spy on nothing.
    import dp_spatial
    recorded_n_steps = []
    orig_worker = dp_spatial._earth_chord_worker

    def spying_worker(args):
        recorded_n_steps.append(args[2])  # (costh, baseline_km, N_steps_earth, ...)
        return orig_worker(args)

    dp_spatial._earth_chord_worker = spying_worker
    try:
        E_bins = np.logspace(13, 16, 20)
        F_mass_arrival = np.full((20, 3), 1e20)
        U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
        M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
        M_rot = U_pmns @ M_sq @ np.conjugate(U_pmns).T
        u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)

        dps.propagate_earth_spatial(E_bins, F_mass_arrival, M_rot, u_arr,
                                    2.0e6, 1e-4, 0.1, 1.95e-4, True, np.ones(3),
                                    1.5, False, 0.0, False, 1, 6,
                                    N_steps_MW=2, r_start_kpc=100.0,
                                    show_progress=False, verbose=0, n_cores=1)
    finally:
        dp_spatial._earth_chord_worker = orig_worker

    assert len(set(recorded_n_steps)) == 1, f"Earth chords received different step counts: {recorded_n_steps}"


# =============================================================================
# 5. init_mw_density_matrix redundant-computation hoist (regression test for the fix)
# =============================================================================

def test_init_mw_density_matrix_matches_reimplementation():
    """init_mw_density_matrix's total ambient number density (n_tot_edge, from the Milky
    Way/Sun/Moon/Earth-point-source terms) depends only on r_start_kpc/theta/phi/m_eff --
    never on the energy bin -- so it was hoisted out of the per-energy-bin loop (it was
    previously being recomputed, identically, on every iteration; the sibling
    step_mw_approach/step_earth_transit functions already did this hoist correctly and this
    was the one place it was missed). An independent, hand-transcribed reimplementation
    (not sharing the hoisted code path) pins down both the hoist and the physics it must
    reproduce."""
    N_E = 12
    E_bins = np.logspace(13, 16, N_E)
    rng = np.random.default_rng(0)
    F_mass_arrival = rng.uniform(1e18, 1e22, size=(N_E, 3))

    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = np.ascontiguousarray(U_pmns @ M_sq @ np.conjugate(U_pmns).T)

    r_start_kpc, theta, phi = 100.0, np.pi / 3.0, 1.1
    m_eff, Coef = 5.0e5, 3.7e-19
    u_arr = np.array([0.5, 0.3 * np.exp(1j * 0.4), 0.2], dtype=np.complex128)
    U_LRI_struct = np.outer(u_arr, np.conjugate(u_arr))

    rho_f = dps.init_mw_density_matrix(r_start_kpc, theta, phi, E_bins, F_mass_arrival,
                                       M_rot, m_eff, Coef, U_LRI_struct)

    # Independent reimplementation: recompute n_tot_edge directly from the physical formula
    # (not via any dps internals beyond the plain helper calc_MW_neutron_density), then build
    # H_arr and rho_f per-bin from scratch using plain numpy.
    n_n_MW_edge = dps.calc_MW_neutron_density(r_start_kpc, theta, phi) * dps.cm_inv3_to_eV3
    r_eV = r_start_kpc * 3.08567758e16 * 1e3 * dps.m_to_eV_inv
    r_sun_eV = np.sqrt(r_eV**2 + dps.d_sun_eV_inv**2)
    r_moon_eV = np.sqrt(r_eV**2 + dps.d_moon_eV_inv**2)
    n_sun_eV3 = dps.N_n_sun * (m_eff**2) / (4.0 * np.pi * r_sun_eV) * np.exp(-m_eff * r_sun_eV)
    n_moon_eV3 = dps.N_n_moon * (m_eff**2) / (4.0 * np.pi * r_moon_eV) * np.exp(-m_eff * r_moon_eV)
    n_earth_eV3 = dps.N_n_earth * (m_eff**2) / (4.0 * np.pi * r_eV) * np.exp(-m_eff * r_eV)
    n_tot_edge = n_n_MW_edge + n_sun_eV3 + n_moon_eV3 + n_earth_eV3

    rho_f_expected = np.zeros((N_E, 3, 3), dtype=np.complex128)
    for i in range(N_E):
        H_vac = (1.0 / (2.0 * E_bins[i])) * M_rot
        H_arr = H_vac + (Coef * n_tot_edge) * U_LRI_struct
        evals, evecs = np.linalg.eigh(H_arr)
        rho_f_expected[i] = evecs @ np.diag(F_mass_arrival[i]) @ np.conjugate(evecs).T

    np.testing.assert_allclose(rho_f, rho_f_expected, rtol=1e-10, atol=1e-30)


# =============================================================================
# 6. Shared dE helper (regression test for the DRY refactor)
# =============================================================================

def test_compute_dE_geometric_matches_independent_formula():
    """compute_dE_geometric (the single implementation shared by propagate_spectrum here
    and compute_energy_averaged_ratios in dark_portal_scanner.py) must reproduce the
    geometric-midpoint bin width computed directly from the log-spaced grid."""
    E_bins = np.logspace(13, 16, 37)
    dE = dps.compute_dE_geometric(E_bins)

    half_ratio = np.sqrt(E_bins[1] / E_bins[0])
    expected = E_bins * half_ratio - E_bins / half_ratio

    np.testing.assert_allclose(dE, expected, rtol=1e-12)


def test_energy_averaged_ratios_uses_shared_dE_helper():
    """flux_postprocessing.compute_energy_averaged_ratios must be built on the same
    dark_portal_solver.compute_dE_geometric used by propagate_spectrum, not a separate
    copy-pasted formula -- regression test for the DRY refactor that replaced the
    previously duplicated dE computation with this shared, imported helper. Also checks
    the function's move out of dark_portal_scanner.py into flux_postprocessing.py: the
    scanner must still re-export it (backward compatibility), and both names must be the
    same object."""
    import flux_postprocessing as fpp
    import dark_portal_scanner as dpsc
    assert dpsc.compute_energy_averaged_ratios is fpp.compute_energy_averaged_ratios

    E_bins = np.logspace(13, 16, 25)
    dE = dps.compute_dE_geometric(E_bins)
    Flux_array = np.zeros((25, 3))
    Flux_array[:, 0] = 1.0 / dE
    Flux_array[:, 1] = 2.0 / dE
    ratios = fpp.compute_energy_averaged_ratios(E_bins, Flux_array)
    np.testing.assert_allclose(ratios, [1.0 / 3.0, 2.0 / 3.0, 0.0], rtol=1e-9)


# =============================================================================
# 7. FrequentistAnalysis static_data wiring (regression test for the fixed dead-parameter bugs)
# =============================================================================

def _make_frequentist_config(**physics_overrides):
    """Minimal physics_constants block sized for fast tests, with production-representative
    defaults for everything compute_user_model_single needs to run end-to-end."""
    physics_constants = {
        "H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_L": 0.685,
        "T_nu0_eV": 1.95e-4, "n_n0_cm3": 0.25e-6, "m_lightest_eV": 0.01,
        "mass_ordering": "normal", "eta_e": 1.0, "eta_mu": 1.0, "eta_tau": 1.0,
        "detector_depth_km": 1.5, "p_index": 2.5, "Phi_0": 1.0, "z_max": 4.0,
        "rho_evol": 0.0, "f_e_S": 0.333333, "f_mu_S": 0.666667, "f_tau_S": 0.0,
        "is_antineutrino": "sum", "E_min_eV": 1e13, "E_max_eV": 3e16,
        "n_E_bins": 15, "n_z_bins": 15, "n_mw_paths": 1, "n_earth_chords": 1,
        "N_steps_MW": 5, "N_steps_earth": 5,
    }
    physics_constants.update(physics_overrides)
    return {"experiment": "test", "physics_constants": physics_constants}


def _default_dark_portal_params():
    return {"m_Zd_eV": 2.0e6, "g_D": 1e-4, "epsilon_prime": 0.1, "u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0}


def test_frequentist_static_data_wires_use_SM_potential_and_n_e_earth():
    """FrequentistAnalysis's GenericLikelihoodWrapper must actually read use_SM_potential and
    n_e_earth_cm3 from physics_constants into static_data, and kernel.py's
    compute_user_model_single must actually use them -- previously both were hardcoded
    (use_SM_potential=False, n_e_earth=0.0) regardless of the config, silently dropping the
    same way n_e_earth_cm3 was previously dropped in dark_portal_solver.py's own run_solver
    (see test_n_e_earth_cm3_affects_flux_when_SM_potential_enabled above)."""
    from src.core.likelihood import GenericLikelihoodWrapper
    from src.physics.kernel import compute_user_model_single

    # n_e_earth_cm3 values chosen so the effect is comfortably above float noise: this
    # dark-portal configuration's LRI potential dominates over the standard MSW potential by
    # several orders of magnitude (as expected -- the whole point of the model is that the
    # new force dominates), so use_SM_potential's effect on the output, while real and
    # non-zero at any n_e_earth_cm3, only clears typical float-comparison tolerances once
    # n_e_earth_cm3 is large enough (empirically checked to give a ~1e-4-level relative
    # effect at 5e24 cm^-3, vs ~1e-16 machine noise when the parameter is dead/dropped).
    cfg_off = _make_frequentist_config(use_SM_potential=False, n_e_earth_cm3=5.0e24)
    cfg_lo = _make_frequentist_config(use_SM_potential=True, n_e_earth_cm3=1.0e23)
    cfg_hi = _make_frequentist_config(use_SM_potential=True, n_e_earth_cm3=5.0e24)

    func_off = GenericLikelihoodWrapper(cfg_off, [])
    func_lo = GenericLikelihoodWrapper(cfg_lo, [])
    func_hi = GenericLikelihoodWrapper(cfg_hi, [])

    assert len(func_off.static_data) == 36
    assert func_off.static_data[27] is False or func_off.static_data[27] == False  # noqa: E712
    assert func_lo.static_data[27] is True or func_lo.static_data[27] == True  # noqa: E712
    assert func_lo.static_data[28] != func_hi.static_data[28]

    params = _default_dark_portal_params()
    result_off = compute_user_model_single(params, func_off.static_data)
    result_lo = compute_user_model_single(params, func_lo.static_data)
    result_hi = compute_user_model_single(params, func_hi.static_data)

    # Turning on use_SM_potential (with n_e_earth actually wired through) must change the
    # output, and different n_e_earth_cm3 values must give different outputs from each other.
    assert not np.allclose(result_off, result_hi, rtol=1e-6, atol=0.0)
    assert not np.allclose(result_lo, result_hi, rtol=1e-6, atol=0.0)


def test_frequentist_static_data_wires_use_debye_screening():
    """use_debye_screening is present in every shipped physics_constants block's spirit
    (dark_portal_solver.py's own config exposes it under dark_portal.use_debye_screening)
    but was hardcoded to True in kernel.py regardless of config. Regression test that it is
    now read from static_data and actually changes the propagated flux when toggled."""
    from src.core.likelihood import GenericLikelihoodWrapper
    from src.physics.kernel import compute_user_model_single

    cfg_on = _make_frequentist_config(use_debye_screening=True)
    cfg_off = _make_frequentist_config(use_debye_screening=False)

    func_on = GenericLikelihoodWrapper(cfg_on, [])
    func_off = GenericLikelihoodWrapper(cfg_off, [])

    assert func_on.static_data[29] is True or func_on.static_data[29] == True  # noqa: E712
    assert func_off.static_data[29] is False or func_off.static_data[29] == False  # noqa: E712

    # A large g_D (strong dark coupling) makes the Debye screening mass non-negligible
    # relative to m_Zd, so toggling it should visibly change the propagated flux.
    params = dict(_default_dark_portal_params())
    params["m_Zd_eV"] = 1.0e3
    params["g_D"] = 1.0

    result_on = compute_user_model_single(params, func_on.static_data)
    result_off = compute_user_model_single(params, func_off.static_data)

    # Wiring check: a dropped flag would give BIT-IDENTICAL results. Under the 'left'
    # Earth scheme the toggle's small V-shift was amplified by discretization phase
    # noise into an O(1e-4) flux change; the phase-exact 'avg' scheme (dev-earth)
    # removes that amplification, leaving only the true physical effect (~1e-7 here)
    # -- so assert at a tolerance that detects wiring, not noise amplification.
    assert not np.allclose(result_on, result_off, rtol=1e-12, atol=1e-12)


def test_frequentist_static_data_backward_compatible_with_shorter_tuples():
    """compute_user_model_single must still work, with safe defaults, given an older
    26-element static_data tuple that predates N_steps_earth/use_SM_potential/n_e_earth/
    use_debye_screening -- these were all added as tuple extensions specifically to avoid
    breaking any caller still constructing the older, shorter form."""
    from src.physics.kernel import compute_user_model_single

    old_static_data = (
        0.33, 0.33, 0.05, 0.05,
        np.logspace(13, 16, 10), np.linspace(4.0, 0.0, 10),
        np.array([0.333333, 0.666667, 0.0]),
        67.4 * dps.km_per_s / dps.Mpc_to_eV_inv, 0.315, 0.685,
        np.eye(3, dtype=np.complex128),
        np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128),
        np.array([0.0, 8.6e-3, 5.01e-2]),
        0.25e-6 * dps.cm_inv3_to_eV3, 1.95e-4, np.ones(3, dtype=np.float64),
        1.5, "sum", 2.5, 0.0, 1.0,
        1.0, 1, 1, 5, 100.0,
    )
    assert len(old_static_data) == 26

    result = compute_user_model_single(_default_dark_portal_params(), old_static_data)
    assert len(result) == 3
    assert all(np.isfinite(result))


# =============================================================================
# 7b. Configurable potential structure matrices (U_LRI_struct / V_SM_struct)
# =============================================================================

def _propagate_small(u_arr, is_anti, U_LRI_struct=None, V_SM_struct=None, use_SM=False):
    """Minimal, fast propagate_spectrum call used by the structure-override tests.
    Mirrors propagation_worker's conventions: U_pmns/u_arr are conjugated here for
    antineutrinos, while structure overrides are passed in neutrino convention
    (propagate_spectrum conjugates those internally)."""
    E_bins = np.logspace(13, 16, 12)
    z_bins = np.linspace(4.0, 0.0, 12)
    rho_src = np.array([0.333333, 0.666667, 0.0])
    U_pmns = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58),
                                                     np.radians(49.1), np.radians(132.0)))
    M_sq = np.ascontiguousarray(np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128))
    H0 = 67.4 * dps.km_per_s / dps.Mpc_to_eV_inv
    m_nu_bg = np.array([0.01, np.sqrt(0.01**2 + 7.41e-5), np.sqrt(0.01**2 + 2.511e-3)])

    local_U_pmns = np.ascontiguousarray(np.conjugate(U_pmns)) if is_anti else U_pmns
    local_u = np.ascontiguousarray(np.conjugate(u_arr)) if is_anti else np.ascontiguousarray(u_arr)

    return dps.propagate_spectrum(
        E_bins, z_bins, rho_src, H0, 0.315, 0.685, local_U_pmns, M_sq,
        local_u, 2.0e6, 1e-4, 0.1, m_nu_bg, 0.25e-6 * dps.cm_inv3_to_eV3, 1.95e-4,
        True, 1.5, use_SM, 1.5e24 * dps.cm_inv3_to_eV3,
        is_anti, 2.5, 0.0, 1.0, np.ones(3), 1, 1,
        N_steps_MW=5, r_start_kpc=100.0, show_progress=False, verbose=0, n_cores=1,
        N_steps_earth=5, U_LRI_struct=U_LRI_struct, V_SM_struct=V_SM_struct)


def test_structure_override_default_equals_absent():
    """Passing the default textures explicitly (U_LRI_struct = outer(u, conj u),
    V_SM_struct = diag(1,0,0)) must reproduce the no-override path exactly, for both
    neutrinos and antineutrinos -- pins down that the override plumbing introduces no
    convention shift on the path every existing config exercises."""
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S_lri = np.outer(u, np.conjugate(u))
    S_sm = np.diag([1.0, 0.0, 0.0]).astype(np.complex128)
    for is_anti in (False, True):
        F_default = _propagate_small(u, is_anti, use_SM=True)
        F_explicit = _propagate_small(u, is_anti, U_LRI_struct=S_lri, V_SM_struct=S_sm, use_SM=True)
        np.testing.assert_array_equal(F_default, F_explicit)


def test_structure_override_changes_flux():
    """A genuinely different Hermitian texture must change the output flux: overriding
    U_LRI_struct always, and overriding V_SM_struct when use_SM_potential is on."""
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    F_default = _propagate_small(u, False, use_SM=True)

    S_lri_alt = np.array([[0.25, 0.0, 0.1], [0.0, 0.25, 0.0], [0.1, 0.0, 0.2]], dtype=np.complex128)
    F_lri = _propagate_small(u, False, U_LRI_struct=S_lri_alt, use_SM=True)
    assert not np.allclose(F_default, F_lri)

    S_sm_alt = np.diag([1.0, 0.5, 0.0]).astype(np.complex128)
    F_sm = _propagate_small(u, False, V_SM_struct=S_sm_alt, use_SM=True)
    assert not np.allclose(F_default, F_sm, rtol=1e-10, atol=0.0)


def test_structure_override_antineutrino_conjugation_matches_u_arr_path():
    """For a complex (CP-violating) sterile mixing vector, running with the explicit
    override S = outer(u, conj u) must reproduce the default u_arr-built path exactly for
    BOTH neutrinos and antineutrinos. This pins the override's antineutrino convention
    (conjugated centrally in propagate_spectrum) to the pre-conjugated-u_arr convention the
    double-conjugation fix established -- if a future edit conjugates the override in the
    wrong place (or twice), the antineutrino case here fails while the neutrino case
    passes."""
    u = np.array([0.4 * np.exp(1j * 0.3), 0.6, 0.2 * np.exp(-1j * 0.9)], dtype=np.complex128)
    S = np.outer(u, np.conjugate(u))
    for is_anti in (False, True):
        F_default = _propagate_small(u, is_anti)
        F_override = _propagate_small(u, is_anti, U_LRI_struct=S)
        np.testing.assert_array_equal(F_default, F_override)


def test_parse_structure_matrix_3x3_formats_and_validation():
    """parse_structure_matrix_3x3 must accept plain numbers and [re, im] pairs, and must
    reject wrong shapes and non-Hermitian matrices loudly (eigh reads only one triangle,
    so a non-Hermitian structure would be silently wrong, not visibly broken)."""
    parsed = dps.parse_structure_matrix_3x3(
        [[1.0, [0.0, 0.5], 0.0], [[0.0, -0.5], 2.0, 0.0], [0.0, 0.0, 3.0]], "test")
    expected = np.array([[1.0, 0.5j, 0.0], [-0.5j, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.complex128)
    np.testing.assert_array_equal(parsed, expected)

    with pytest.raises(ValueError):
        dps.parse_structure_matrix_3x3([[1.0, 0.0], [0.0, 1.0]], "test")  # wrong shape
    with pytest.raises(ValueError):
        dps.parse_structure_matrix_3x3(
            [[1.0, 0.7, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]], "test")  # not Hermitian


def test_run_solver_accepts_structure_overrides_in_config(tmp_path):
    """End-to-end: the config keys standard_mixing.V_SM_struct and
    dark_portal.U_LRI_struct must reach the physics through run_solver -- an override
    differing from the default texture must change the output flux."""
    import copy

    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["grid"].update(dict(n_E_bins=12, n_z_bins=12, n_mw_paths=1, n_earth_chords=1,
                            N_steps_MW=5, N_steps_earth=5))
    cfg["execution"].update(dict(n_cores=1, verbose=0))

    cfg_override = copy.deepcopy(cfg)
    cfg_override["dark_portal"]["U_LRI_struct"] = [[0.25, 0.0, 0.1], [0.0, 0.25, 0.0], [0.1, 0.0, 0.2]]

    _, F_base = dps.run_solver(cfg)
    _, F_override = dps.run_solver(cfg_override)
    assert not np.allclose(F_base, F_override)


# =============================================================================
# 7c. Closed-form spatial propagator (regression tests for the eigh replacement)
# =============================================================================

def _propagator_via_eigh(H, t):
    """Reference propagator exp(-i H t) via full eigendecomposition -- the construction
    propagator_3x3 replaced in the spatial hot loops."""
    evals, evecs = np.linalg.eigh(H)
    return (evecs * np.exp(-1j * evals * t)) @ np.conjugate(evecs).T


def test_propagator_3x3_matches_eigh_at_physics_scales():
    """propagator_3x3 (closed-form trigonometric eigenvalues + Lagrange spectral
    interpolation) must agree with the eigh construction to fp noise across random
    Hermitian Hamiltonians at the magnitudes and step lengths the spatial transit
    actually uses (|H| ~ 1e-20..1e-12 eV, dr ~ 1e8..1e13 eV^-1), and must stay unitary."""
    rng = np.random.default_rng(7)
    out = np.zeros((3, 3), dtype=np.complex128)
    worst = 0.0
    worst_unitarity = 0.0
    for _ in range(500):
        scale = 10.0 ** rng.uniform(-20, -12)
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        H = np.ascontiguousarray((A + A.conj().T) / 2 * scale)
        t = 10.0 ** rng.uniform(8, 13)
        dps.propagator_3x3(H, t, out)
        worst = max(worst, float(np.max(np.abs(out - _propagator_via_eigh(H, t)))))
        worst_unitarity = max(worst_unitarity, float(np.max(np.abs(out @ out.conj().T - np.eye(3)))))
    assert worst < 1e-12
    assert worst_unitarity < 1e-12


def test_propagator_3x3_near_degenerate_stability():
    """Near-degenerate spectra are exactly what an MSW-like matter resonance produces --
    and, with the rank-1 LRI structure (u_tau = 0), what the Earth-transit Hamiltonian
    looks like at essentially EVERY step wherever the matter potential rivals the vacuum
    term. The Newton divided-difference form (sinc first differences + Taylor branch for
    tiny phase spread) must therefore stay accurate across gaps from harmless (1e-4
    relative) down to essentially exact degeneracy (1e-14) with no eigendecomposition
    fallback at all."""
    rng = np.random.default_rng(11)
    out = np.zeros((3, 3), dtype=np.complex128)
    worst = 0.0
    for _ in range(500):
        gap = 10.0 ** rng.uniform(-14, -4)
        lam = np.array([1.0, 1.0 + gap, 2.0]) * 1e-16
        Q = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
        H = Q @ np.diag(lam) @ Q.conj().T
        H = np.ascontiguousarray((H + H.conj().T) / 2)
        t = 1e11
        dps.propagator_3x3(H, t, out)
        worst = max(worst, float(np.max(np.abs(out - _propagator_via_eigh(H, t)))))
    assert worst < 1e-9


def test_propagator_3x3_rank1_dominated_hamiltonian():
    """The production regime that previously forced a 100% eigh-fallback rate: a rank-1
    potential dominating a small vacuum term gives H two eigenvalues degenerate by
    construction. The divided-difference propagator must handle it at full accuracy and
    stay unitary, across the potential-to-vacuum ratio range the Earth transit spans."""
    rng = np.random.default_rng(21)
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = U_pmns @ M_sq @ np.conjugate(U_pmns).T
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S = np.outer(u, np.conjugate(u))
    out = np.zeros((3, 3), dtype=np.complex128)
    worst = 0.0
    worst_unit = 0.0
    for _ in range(400):
        E = 10.0 ** rng.uniform(13, 16.5)
        V_scale = 10.0 ** rng.uniform(-19, -15)     # spans vacuum-dominated to V-dominated
        H = np.ascontiguousarray(M_rot / (2.0 * E) + V_scale * S)
        t = 10.0 ** rng.uniform(9, 12)              # Earth-step dr range in eV^-1
        dps.propagator_3x3(H, t, out)
        worst = max(worst, float(np.max(np.abs(out - _propagator_via_eigh(H, t)))))
        worst_unit = max(worst_unit, float(np.max(np.abs(out @ out.conj().T - np.eye(3)))))
    assert worst < 1e-10
    assert worst_unit < 1e-11


def test_propagator_3x3_large_phase_node_exactness():
    """Milky-Way-regime regression: per-step phases lam*t reach ~1e10 radians (gap phases
    ~1e9), where double precision defines a phase only to ~1e-5 rad. The propagator must
    still interpolate the COMPUTED e^{-i lam t} exactly at the eigenvalue nodes -- i.e.
    remain unitary to fp precision -- because a coefficient form that only satisfies the
    interpolation conditions approximately (the pure sinc form did, at ~1e-5 per step)
    turns phase-representation noise into a cumulative AMPLITUDE error (measured 1.3e-4
    end-to-end before the node-exact branch existed). Checks eigenvalue-wise
    |U's spectrum| = 1 and agreement with the eigh construction."""
    rng = np.random.default_rng(33)
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = U_pmns @ M_sq @ np.conjugate(U_pmns).T
    out = np.zeros((3, 3), dtype=np.complex128)
    worst_unit = 0.0
    worst_vs_eigh = 0.0
    for _ in range(300):
        E = 10.0 ** rng.uniform(13, 16)
        H = np.ascontiguousarray(M_rot / (2.0 * E))       # vacuum-dominated, like the MW
        t = 10.0 ** rng.uniform(25, 27)                    # MW step lengths in eV^-1
        dps.propagator_3x3(H, t, out)
        worst_unit = max(worst_unit, float(np.max(np.abs(out @ out.conj().T - np.eye(3)))))
        # vs eigh construction: phases individually rounded, but both must be unitary and
        # agree on the |U|-structure; compare full matrices (phase decompositions differ
        # at ~1e-5 rad, i.e. ~1e-5 absolute on elements is the shared representation
        # limit -- the unitarity assertion above is the strict one)
        worst_vs_eigh = max(worst_vs_eigh, float(np.max(np.abs(np.abs(out) - np.abs(_propagator_via_eigh(H, t))))))
    assert worst_unit < 1e-12, f"non-unitary at large phases: {worst_unit:.3e}"
    assert worst_vs_eigh < 1e-4


def test_eigenvectors_3x3_matches_eigh():
    """eigenvectors_3x3 (closed-form projector-column method, replacing np.linalg.eigh in
    the cosmological mixing loop) must return true eigenvectors with columns in
    ASCENDING-eigenvalue order -- eigh's convention, which the adiabatic mass-eigenstate
    bookkeeping relies on. Column phases are arbitrary (as they are for eigh), so the
    comparison uses |U|^2, the only projection every downstream consumer uses."""
    rng = np.random.default_rng(3)
    worst_resid = 0.0
    worst_proj = 0.0
    for _ in range(500):
        scale = 10.0 ** rng.uniform(-20, -12)
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        H = np.ascontiguousarray((A + A.conj().T) / 2 * scale)
        evals_ref, evecs_ref = np.linalg.eigh(H)
        U = dps.eigenvectors_3x3(H)
        # Residual: each column must be an eigenvector of H with the ascending eigenvalue
        for k in range(3):
            resid = np.max(np.abs(H @ U[:, k] - evals_ref[k] * U[:, k])) / scale
            worst_resid = max(worst_resid, float(resid))
        worst_proj = max(worst_proj, float(np.max(np.abs(np.abs(U)**2 - np.abs(evecs_ref)**2))))
    assert worst_resid < 1e-10
    assert worst_proj < 1e-9

    # Near-degenerate spectra take the eigh fallback and must agree exactly with eigh
    for _ in range(100):
        gap = 10.0 ** rng.uniform(-14, -6)
        lam = np.array([1.0, 1.0 + gap, 2.0]) * 1e-16
        Q = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
        H = np.ascontiguousarray((Q @ np.diag(lam) @ Q.conj().T + (Q @ np.diag(lam) @ Q.conj().T).conj().T) / 2)
        U = dps.eigenvectors_3x3(H)
        evals_ref, evecs_ref = np.linalg.eigh(H)
        np.testing.assert_allclose(np.abs(U)**2, np.abs(evecs_ref)**2, atol=1e-12)


def test_cascade_suffix_sum_matches_bruteforce():
    """The O(N_E) suffix-sum form of the cascading integral in
    step_cosmological_propagation must reproduce the original O(N_E^2) double loop:
    dF_casc[i, a] = BR[a] * 3 E_i^2 * sum_{j >= i} (sum_b Gamma[j,b] F_flavor[j,b]) * W[j].
    The identity holds because the cascading kernel is separable (the whole i-dependence
    is the 3 E_i^2 prefactor); this test drives the real JIT kernel and checks its
    dF_casc buffer against an independent brute-force evaluation built from the same
    U_z projections."""
    N_E = 40
    E_bins = np.logspace(13, 16, N_E)
    rng = np.random.default_rng(5)
    F_mass = rng.uniform(1e18, 1e22, size=(N_E, 3))
    rho_src = np.array([0.333, 0.667, 0.0])
    U_pmns = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58),
                                                     np.radians(49.1), np.radians(132.0)))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = np.ascontiguousarray(U_pmns @ M_sq @ np.conjugate(U_pmns).T)
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S = np.ascontiguousarray(np.outer(u_arr, np.conjugate(u_arr)))
    eta_bg = np.ones(3)
    BR = np.array([0.4, 0.35, 0.25])
    dE = dps.compute_dE_geometric(E_bins)
    W_kernel = dE / E_bins**3
    Gamma_pre = rng.uniform(1e-20, 1e-18, size=(N_E, 3))

    F_flavor = np.zeros((N_E, 3)); Gamma_flavor = np.zeros((N_E, 3))
    dF_casc = np.zeros((N_E, 3)); P_all = np.zeros((N_E, 3, 3))
    rate_buf = np.zeros(N_E)
    # Mixing now arrives through the sigma-table (see _build_sigma_projection_table)
    c_z = 1e-30
    sigma_nodes = np.geomspace(E_bins[0] * c_z * 0.5, E_bins[-1] * c_z * 2.0, 512)
    P_table = dps._build_sigma_projection_table(M_rot, S, sigma_nodes)
    log_s0 = np.log(sigma_nodes[0])
    inv_dlogs = (len(sigma_nodes) - 1) / np.log(sigma_nodes[-1] / sigma_nodes[0])
    dps.step_cosmological_propagation(0.5, 1.0, 1.0, E_bins, F_mass.copy(), rho_src,
                                      P_table, log_s0, inv_dlogs, c_z,
                                      2.5, 1.0, BR, W_kernel,
                                      F_flavor, Gamma_flavor, dF_casc, P_all, rate_buf, Gamma_pre)

    # Brute-force reference from the buffers the kernel itself filled
    expected = np.zeros((N_E, 3))
    for i in range(N_E):
        integral = 0.0
        for j in range(i, N_E):
            integral += np.sum(Gamma_flavor[j] * F_flavor[j]) * W_kernel[j]
        expected[i, :] = BR * 3.0 * E_bins[i]**2 * integral
    np.testing.assert_allclose(dF_casc, expected, rtol=1e-12)


def test_build_thermal_grid_exact_and_interpolated():
    """build_thermal_grid must be bit-identical to precompute_all_thermal_rates when
    interpolation is disabled (n_z_thermal = 0, the default, or >= N_z), and when
    enabled must reproduce the exact rates at every fine-grid point that coincides with
    a coarse-grid node (uniform-to-uniform index mapping is exact, so coincident nodes
    interpolate with weight 0)."""
    E_bins = np.logspace(13, 16, 8)
    z_bins = np.linspace(4.0, 0.0, 21)
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    sum_G, sum_G_alpha, eta_mass = dps.compute_gamma_grid_inputs(U_pmns, u_arr, 1e-4, 0.1, np.ones(3))
    m_nu = np.array([0.01, 0.012, 0.06])

    # thermal_analytic=False pins the midpoint reference kernel: this test checks the
    # z-interpolation MECHANICS, independently of the (now default) semi-analytic
    # momentum kernel, whose own routing is covered by
    # test_build_thermal_grid_analytic_switch.
    exact = dps.precompute_all_thermal_rates(E_bins, z_bins, sum_G, sum_G_alpha, 2.0e6, m_nu, 1.95e-4, eta_mass)
    for n_sub in (0, 21, 100):
        got = dps.build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, 2.0e6, m_nu, 1.95e-4, eta_mass, n_z_thermal=n_sub,
                                     thermal_analytic=False)
        np.testing.assert_array_equal(got, exact)

    # n_sub = 11 on a 21-point fine grid: every even fine index coincides with a node
    interp = dps.build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, 2.0e6, m_nu, 1.95e-4, eta_mass, n_z_thermal=11,
                                    thermal_analytic=False)
    np.testing.assert_allclose(interp[::2], exact[::2], rtol=1e-12)
    # Odd indices are genuine midpoint interpolations of the coarse neighbors
    np.testing.assert_allclose(interp[1::2], 0.5 * (exact[::2][:-1] + exact[::2][1:]), rtol=1e-12)


def test_spatial_E_block_parallelism_is_exact():
    """The spatial handover splits (path/chord, E-block) tasks across workers for load
    balancing. Energy bins evolve independently, and per-unit results are reassembled to
    full arrays before the average over units, so the multi-worker blocked path must be
    EXACTLY equal (not merely close) to the single-worker sequential path."""
    E_bins = np.logspace(13, 16, 30)
    rng = np.random.default_rng(9)
    F_mass_arrival = rng.uniform(1e18, 1e22, size=(30, 3))
    U_pmns = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58),
                                                     np.radians(49.1), np.radians(132.0)))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = np.ascontiguousarray(U_pmns @ M_sq @ np.conjugate(U_pmns).T)
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)

    common = dict(N_steps_MW=4, r_start_kpc=100.0, show_progress=False, verbose=0,
                  N_steps_earth=6)
    F_seq = dps.propagate_earth_spatial(E_bins, F_mass_arrival, M_rot, u_arr,
                                        2.0e6, 1e-4, 0.1, 1.95e-4, True, np.ones(3),
                                        1.5, False, 0.0, False, 3, 5,
                                        n_cores=1, **common)
    F_par = dps.propagate_earth_spatial(E_bins, F_mass_arrival, M_rot, u_arr,
                                        2.0e6, 1e-4, 0.1, 1.95e-4, True, np.ones(3),
                                        1.5, False, 0.0, False, 3, 5,
                                        n_cores=3, **common)
    np.testing.assert_array_equal(F_seq, F_par)


def _spatial_args_small():
    E_bins = np.logspace(13, 16, 20)
    rng = np.random.default_rng(13)
    F_mass_arrival = rng.uniform(1e18, 1e22, size=(20, 3))
    U_pmns = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58),
                                                     np.radians(49.1), np.radians(132.0)))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = np.ascontiguousarray(U_pmns @ M_sq @ np.conjugate(U_pmns).T)
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    args = (E_bins, F_mass_arrival, M_rot, u_arr, 2.0e6, 1e-4, 0.1, 1.95e-4, True, np.ones(3),
            1.5, False, 0.0, False, 2, 3)
    kwargs = dict(N_steps_MW=4, r_start_kpc=100.0, show_progress=False, verbose=0,
                  n_cores=1, N_steps_earth=6)
    return args, kwargs


def test_use_gpu_falls_back_to_cpu_without_cuda():
    """With use_gpu=True but no cupy/CUDA available, propagate_earth_spatial must warn
    and produce EXACTLY the CPU result -- the flag must never change physics output when
    the GPU path cannot run. (On a machine with a working GPU this test still passes
    vacuously unless results diverge, which the dedicated GPU test below then catches.)"""
    import gpu_spatial
    args, kwargs = _spatial_args_small()
    F_cpu = dps.propagate_earth_spatial(*args, **kwargs)
    F_flagged = dps.propagate_earth_spatial(*args, use_gpu=True, **kwargs)
    if not gpu_spatial.gpu_available():
        np.testing.assert_array_equal(F_cpu, F_flagged)


def test_gpu_spatial_matches_cpu():
    """GPU spatial handover vs CPU path at identical inputs. SKIPPED without a CUDA
    device -- run this on the GPU machine before trusting --use_gpu results. The two
    paths share the density helpers and the closed-form propagator mathematics, so
    agreement should be at fp-noise level."""
    import gpu_spatial
    if not gpu_spatial.gpu_available():
        pytest.skip("no cupy/CUDA device available")
    args, kwargs = _spatial_args_small()
    F_cpu = dps.propagate_earth_spatial(*args, **kwargs)
    F_gpu = dps.propagate_earth_spatial(*args, use_gpu=True, **kwargs)
    rel = np.max(np.abs(F_cpu - F_gpu) / np.maximum(np.abs(F_cpu), 1e-300))
    assert rel < 1e-9, f"GPU vs CPU max relative difference {rel:.3e}"


def test_sigma_projection_table_matches_exact_eigenvectors():
    """The cosmological mixing runs through a one-parameter table: |U|^2 of
    H(sigma) = M_rot + sigma * S, with sigma = 2 E (1+z) A_LRI(z) (verified exact scale
    invariance). The table's linear-in-log-sigma interpolation must reproduce direct
    eigenvector evaluations across the level crossing -- crossing structure in log sigma
    is mixing-angle-broad, so unlike the thermal resonance there is no sub-grid feature
    to alias. End-to-end validation (4 parameter points incl. a crossing-in-range one at
    2.7e-14) lives in the commit record; this pins the table machinery itself."""
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = np.ascontiguousarray(U_pmns @ M_sq @ np.conjugate(U_pmns).T)
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S = np.ascontiguousarray(np.outer(u, np.conjugate(u)))

    scale = np.max(np.abs(M_rot))
    sig_min, sig_max = 1e-6 * scale, 1e2 * scale     # spans vacuum limit -> crossing -> potential-dominated
    nodes = np.geomspace(sig_min, sig_max, 4096)
    table = dps._build_sigma_projection_table(M_rot, S, nodes)
    log_s0 = np.log(nodes[0])
    inv_dlogs = (len(nodes) - 1) / np.log(sig_max / sig_min)

    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(300):
        sig = 10.0 ** rng.uniform(np.log10(sig_min * 1.01), np.log10(sig_max * 0.99))
        U = dps.eigenvectors_3x3(np.ascontiguousarray(M_rot + sig * S))
        P_exact = np.abs(U) ** 2
        f = (np.log(sig) - log_s0) * inv_dlogs
        k0 = min(max(int(f), 0), len(nodes) - 2)
        w = min(max(f - k0, 0.0), 1.0)
        P_interp = (1.0 - w) * table[k0] + w * table[k0 + 1]
        worst = max(worst, float(np.max(np.abs(P_interp - P_exact))))
    assert worst < 1e-4, f"sigma-table interpolation error {worst:.3e}"


def test_mw_density_finite_on_galactic_center_line():
    """calc_MW_neutron_density previously crashed (ZeroDivisionError) or returned NaN when
    evaluated on the exact line through the galactic center (r ~ R_0 = 8.33 kpc along
    theta = pi/2, phi = 0 -- the trajectory of one of the six default Milky Way paths):
    catastrophic cancellation in the law-of-cosines argument went a hair negative under
    fastmath, sqrt produced NaN, and the bulge term's power turned that into an
    exception. The argument is now clamped to zero (any negative value is pure
    rounding). The density there must be finite, positive, and continuous with the
    immediate neighborhood."""
    v_center = dps.calc_MW_neutron_density(8.33, np.pi / 2.0, 0.0)
    v_near = dps.calc_MW_neutron_density(8.33 + 1e-10, np.pi / 2.0, 0.0)
    v_off = dps.calc_MW_neutron_density(8.32, np.pi / 2.0, 0.0)
    assert np.isfinite(v_center) and v_center > 0
    assert np.isclose(v_center, v_near, rtol=1e-6)
    assert np.isfinite(v_off) and v_off > 0


# =============================================================================
# 8. MainProcess-adaptive n_cores (regression test for the fix)
# =============================================================================

def test_frequentist_n_cores_standalone_wired_from_scan_settings():
    """GenericLikelihoodWrapper must read scan_settings.num_cores (clamped to actually
    available cores) into static_data as n_cores_standalone, for kernel.py's
    run_propagation to use whenever it is *not* running inside main.py's scan/repair
    multiprocessing.Pool."""
    from src.core.likelihood import GenericLikelihoodWrapper

    cfg = _make_frequentist_config()
    cfg["scan_settings"] = {"num_cores": 4}
    func = GenericLikelihoodWrapper(cfg, [])
    assert len(func.static_data) == 36
    assert func.static_data[30] == 4

    # Clamped to what's actually available, never silently allowed above it.
    import multiprocessing as _mp
    huge_request = _mp.cpu_count() * 100
    cfg_huge = _make_frequentist_config()
    cfg_huge["scan_settings"] = {"num_cores": huge_request}
    func_huge = GenericLikelihoodWrapper(cfg_huge, [])
    assert 1 <= func_huge.static_data[30] <= _mp.cpu_count()


def _call_kernel_and_report_process(args):
    """Module-level (picklable) helper for
    test_frequentist_kernel_uses_n_cores_standalone_only_in_main_process -- must live at
    module scope so multiprocessing can pickle it for dispatch to a real Pool worker."""
    import multiprocessing as _mp
    from src.physics.kernel import compute_user_model_single
    static_data, params = args
    proc_name = _mp.current_process().name
    result = compute_user_model_single(params, static_data)
    return proc_name, proc_name == "MainProcess", result


def test_frequentist_kernel_uses_n_cores_standalone_only_in_main_process():
    """kernel.py's run_propagation must use n_cores_standalone when running in the main
    process (e.g. main.py's initial m.simplex()/m.migrad() global fit, which runs before
    any scan/repair Pool worker ever touches this function), but must force n_cores back
    to 1 whenever called from inside an actual multiprocessing.Pool worker -- otherwise a
    worker would itself spin up a nested pool and oversubscribe every core (the same
    hazard the BLAS/numba thread-pinning guards in dp_core.py exist to avoid).
    Both must produce the exact same physics result regardless of core count."""
    from src.core.likelihood import GenericLikelihoodWrapper
    import multiprocessing as _mp

    cfg = _make_frequentist_config()
    cfg["scan_settings"] = {"num_cores": max(2, min(4, _mp.cpu_count()))}
    func = GenericLikelihoodWrapper(cfg, [])
    params = _default_dark_portal_params()

    assert _mp.current_process().name == "MainProcess"
    result_main = _call_kernel_and_report_process((func.static_data, params))

    with _mp.Pool(processes=2) as pool:
        result_worker = pool.apply(_call_kernel_and_report_process, ((func.static_data, params),))

    proc_name_main, is_main_flag, flux_main = result_main
    proc_name_worker, is_main_worker_flag, flux_worker = result_worker

    assert is_main_flag is True
    assert is_main_worker_flag is False
    assert proc_name_worker != "MainProcess"
    np.testing.assert_allclose(flux_main, flux_worker, rtol=1e-9)


def test_frequentist_persistent_pool_reused_under_spawn_not_recreated_per_call():
    """Regression test for a near-miss caught during this fix's own development: the first
    version of the MainProcess-adaptive n_cores fix created a fresh multiprocessing.Pool
    inside propagate_earth_spatial on every single call. That is cheap under the 'fork' start
    method (the default on Linux, and what every earlier timing check in this fix's
    development happened to use) but *catastrophically* expensive under 'spawn' -- a full
    interpreter relaunch plus reimport per worker, per call. main.py explicitly forces
    'spawn' (multiprocessing.set_start_method('spawn', force=True)), and measured directly on
    the real production config, using more cores made every single Migrad evaluation ~10x
    *slower* without pool reuse (3.75s/eval vs 0.37s/eval). kernel.py's _get_persistent_pool
    must construct the Pool at most once and reuse it across many calls.

    This runs in an isolated subprocess: the multiprocessing start method is process-global
    and can only be forced to 'spawn' once, before any Pool has been created in that process
    -- doing so here would affect every other test in this file that uses multiprocessing.
    """
    script = f'''
import sys, multiprocessing, multiprocessing.pool
sys.path.insert(0, {_FREQ_DIR!r})
multiprocessing.set_start_method("spawn", force=True)

_pool_construct_count = [0]
_RealPool = multiprocessing.pool.Pool
class _CountingPool(_RealPool):
    def __init__(self, *a, **k):
        _pool_construct_count[0] += 1
        super().__init__(*a, **k)
multiprocessing.Pool = _CountingPool

from src.core.likelihood import GenericLikelihoodWrapper
from src.physics.kernel import compute_user_model_single

config = {{
    "experiment": "test",
    "physics_constants": {{
        "H0_km_s_Mpc": 67.4, "Omega_m": 0.315, "Omega_L": 0.685,
        "T_nu0_eV": 1.95e-4, "n_n0_cm3": 0.25e-6, "m_lightest_eV": 0.01,
        "mass_ordering": "normal", "eta_e": 1.0, "eta_mu": 1.0, "eta_tau": 1.0,
        "detector_depth_km": 1.5, "p_index": 2.5, "Phi_0": 1.0, "z_max": 4.0,
        "rho_evol": 0.0, "f_e_S": 0.333333, "f_mu_S": 0.666667, "f_tau_S": 0.0,
        "is_antineutrino": "sum", "E_min_eV": 1e13, "E_max_eV": 3e16,
        "n_E_bins": 12, "n_z_bins": 12, "n_mw_paths": 8, "n_earth_chords": 8,
        "N_steps_MW": 5, "N_steps_earth": 5,
    }},
    "scan_settings": {{"num_cores": 4}},
}}
params = {{"m_Zd_eV": 2.0e6, "g_D": 1e-4, "epsilon_prime": 0.1, "u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0}}

func = GenericLikelihoodWrapper(config, [])
results = [compute_user_model_single(params, func.static_data) for _ in range(3)]

assert all(r == results[0] for r in results), f"results differ across repeated calls: {{results}}"
print(f"POOL_CONSTRUCTIONS={{_pool_construct_count[0]}}")
print("OK")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=420)
    assert result.returncode == 0, f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"

    pool_count_lines = [l for l in result.stdout.splitlines() if l.startswith("POOL_CONSTRUCTIONS=")]
    assert pool_count_lines, f"missing pool count marker in output: {result.stdout}"
    pool_constructions = int(pool_count_lines[0].split("=")[1])
    assert pool_constructions == 1, f"expected exactly 1 Pool construction reused across 3 calls, got {pool_constructions}"


def test_threads_mode_prange_then_spawn_pool_same_process():
    """The TBB gate for the threads-based spatial mode: run the parallel prange spatial
    kernel (multi-threaded), then create AND use a spawn multiprocessing Pool in the
    same process, then run the parallel kernel again. With numba's default workqueue
    threading layer this combination deadlocked (420 s hang) and once segfaulted --
    the reason the first threads-mode attempt was reverted. Under the TBB layer it
    must complete. Skipped when the tbb runtime is unavailable (threads mode is then
    disabled and the hazard cannot arise). Isolated subprocess: the start method is
    process-global."""
    if not dps._TBB_LAYER:
        pytest.skip("tbb runtime not available; threads mode disabled")
    script = f'''
import sys, multiprocessing
sys.path.insert(0, {_SRC_DIR!r})
import numpy as np

def pool_task(seed):
    # each spawn worker runs a real (single-threaded) solver evaluation
    import dark_portal_solver as dps
    cfg = {{"grid": {{"n_E_bins": 10, "n_z_bins": 10, "n_mw_paths": 1, "n_earth_chords": 1,
                      "N_steps_MW": 2, "N_steps_earth": 2}},
            "execution": {{"n_cores": 1, "verbose": 0}}}}
    E, F = dps.run_solver(cfg)
    return float(F.sum())

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    import numba
    import dark_portal_solver as dps
    assert dps._TBB_LAYER, "tbb layer not active in subprocess"

    cfg = {{"grid": {{"n_E_bins": 40, "n_z_bins": 20, "n_mw_paths": 2, "n_earth_chords": 2,
                      "N_steps_MW": 5, "N_steps_earth": 20}},
            "execution": {{"n_cores": 4, "verbose": 0}}}}
    E, F1 = dps.run_solver(cfg)              # threads-mode parallel kernel
    assert numba.threading_layer() == "tbb", numba.threading_layer()

    with multiprocessing.Pool(processes=2) as pool:
        out = pool.map(pool_task, [0, 1])    # spawn pool AFTER the parallel kernel
    assert all(np.isfinite(v) for v in out)

    E, F2 = dps.run_solver(cfg)              # parallel kernel again after the pool
    assert np.array_equal(F1, F2), "threads-mode result changed across pool cycle"
    print("OK")
'''
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=420)
    finally:
        os.unlink(path)
    assert result.returncode == 0, f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"


# =============================================================================
# 9. Config normalization, ratio-skip, sentinel-safe intervals
# =============================================================================

def test_run_solver_rejects_unknown_config_keys(tmp_path):
    """merge_config_with_defaults must reject typo'd sections/keys loudly: a misspelled
    key previously fell through every .get() fallback and silently did nothing -- the
    parsed-but-dropped failure mode behind several past bugs (n_e_earth_cm3,
    use_SM_potential, execution.n_cores)."""
    import copy
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["grid"].update(dict(n_E_bins=10, n_z_bins=10, n_mw_paths=1, n_earth_chords=1,
                            N_steps_MW=3, N_steps_earth=3))
    cfg["execution"].update(dict(n_cores=1, verbose=0))

    bad = copy.deepcopy(cfg)
    bad["grid"]["n_eart_chords"] = 5  # typo
    with pytest.raises(ValueError, match="n_eart_chords"):
        dps.run_solver(bad)

    bad2 = copy.deepcopy(cfg)
    bad2["gird"] = {"n_E_bins": 10}  # typo'd section
    with pytest.raises(ValueError, match="gird"):
        dps.run_solver(bad2)

    # _comment keys are the accepted annotation convention and must pass through
    ok = copy.deepcopy(cfg)
    ok["grid"]["_comment_note"] = "annotation"
    E, F = dps.run_solver(ok)
    assert np.all(np.isfinite(F))


def test_nubar_ratio_zero_skips_antineutrino_propagation(tmp_path):
    """In 'sum' mode with nubar_to_nu_ratio = 0 the antineutrino propagation carries
    exactly zero weight and must be skipped entirely -- the result must equal the
    explicit neutrino-only mode bit for bit."""
    import copy
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["grid"].update(dict(n_E_bins=10, n_z_bins=10, n_mw_paths=1, n_earth_chords=1,
                            N_steps_MW=3, N_steps_earth=3))
    cfg["execution"].update(dict(n_cores=1, verbose=0))

    cfg_sum0 = copy.deepcopy(cfg)
    cfg_sum0["source"]["is_antineutrino"] = "sum"
    cfg_sum0["source"]["nubar_to_nu_ratio"] = 0.0
    cfg_nu = copy.deepcopy(cfg)
    cfg_nu["source"]["is_antineutrino"] = "false"

    _, F_sum0 = dps.run_solver(cfg_sum0)
    _, F_nu = dps.run_solver(cfg_nu)
    np.testing.assert_array_equal(F_sum0, F_nu)


def test_calculate_intervals_ignores_sentinel_boundaries():
    """A failed grid point (1e9 sentinel) adjacent to a valid sub-threshold point must
    not fabricate a threshold crossing at the failure boundary."""
    from src.analysis.statistics import calculate_intervals

    x = np.linspace(0.0, 1.0, 11)
    d = np.full(11, 0.5)          # everywhere comfortably below a 2.71 threshold
    d[5] = 1e9                    # one failed point mid-scan
    res = calculate_intervals(x, d, "p", {"90": 2.71}, merge_gaps=False)
    intervals = res["90"]
    assert intervals is not None
    # Every crossing-derived boundary except the scan edges would be spurious here:
    # the entire valid curve sits below threshold, so the allowed region must span the
    # full scan without interior breakpoints from the sentinel.
    assert len(intervals) == 1
    assert intervals[0][0] == pytest.approx(0.0) and intervals[0][1] == pytest.approx(1.0)


# =============================================================================
# Extreme-mediator-mass domain: phase-noise guard + loud NaN contract (task #3)
# =============================================================================

def test_propagator_3x3_phase_noise_regime_stays_unitary():
    """The all-NaN-flux bug at extreme mediator masses (m_Zd ~ 1e-6 eV): the
    1/m_Zd^2-enhanced Earth potential reaches ~1e17 eV, the rank-1 LRI texture makes the
    Hamiltonian doubly degenerate, and per-step phases spread*t reach ~1e27 rad -- far
    beyond the ~1/eps radians double precision can represent. The deflation branch's
    polynomial coefficients are then evaluated at cancellation-noise-limited nodes and
    the propagator came out non-unitary by ~11 orders of magnitude (measured singular
    values ~5e11); 500 such steps overflowed rho to Inf/NaN. The phase-noise guard must
    keep U exactly unitary across the whole potential ladder, at both Earth- and
    Milky-Way-scale step lengths, and must preserve the far eigenvector's ray."""
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = U_pmns @ M_sq @ np.conjugate(U_pmns).T
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S = np.outer(u, np.conjugate(u))
    v_far = (u / np.linalg.norm(u)).reshape(3)
    out = np.zeros((3, 3), dtype=np.complex128)

    worst_unit = 0.0
    worst_ray = 0.0
    # V ladders start where the phase-noise guard governs (eps * spread * t > 0.1);
    # below that the pre-existing (unguarded) deflation path applies, whose small
    # non-unitarity band is pinned as-is by the pipeline validation, not by this test.
    for t, logV_min in ((1.29e11, 6), (6.25e32, 2)):    # Earth step / Milky Way step
        for logV in range(logV_min, 23, 2):
            for E in (1e13, 3e16):
                V_scale = 10.0 ** logV
                H = np.ascontiguousarray(M_rot / (2.0 * E) + V_scale * S)
                dps.propagator_3x3(H, t, out)
                assert np.isfinite(out).all(), f"non-finite U at V={V_scale:.0e}, t={t:.2e}"
                worst_unit = max(worst_unit, float(np.max(np.abs(out @ out.conj().T - np.eye(3)))))
                # the far (rank-1) eigenvector must map to itself up to a phase
                w = out @ v_far
                worst_ray = max(worst_ray, abs(1.0 - abs(np.vdot(v_far, w))))
    assert worst_unit < 1e-12, f"non-unitary in phase-noise regime: {worst_unit:.3e}"
    assert worst_ray < 1e-10, f"far-eigenvector ray not preserved: {worst_ray:.3e}"


def test_extreme_mediator_mass_flux_is_finite(tmp_path):
    """End-to-end contract for the formerly all-NaN domain: m_Zd_eV = 1e-6 (g_D = 1e-4,
    epsilon_prime = 0.1) with the full 500 Earth steps (the NaN trigger -- repeated
    application of the non-unitary propagator needed a few hundred steps to overflow)
    must now return an all-finite, non-negative-total flux."""
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["dark_portal"].update(dict(m_Zd_eV=1e-6, g_D=1e-4, epsilon_prime=0.1))
    cfg["grid"].update(dict(n_E_bins=20, n_z_bins=20, n_mw_paths=2, n_earth_chords=2,
                            N_steps_MW=25, N_steps_earth=500))
    cfg["execution"].update(dict(n_cores=1, verbose=0))
    E, F = dps.run_solver(cfg)
    assert np.isfinite(F).all()
    assert float(np.sum(F)) > 0.0


def test_propagate_spectrum_raises_on_nonfinite_gamma_grid():
    """The loud-NaN contract: a non-finite thermal-rate grid must raise a ValueError
    naming the dark-portal parameters instead of silently propagating NaN into the
    returned spectrum (the historical failure mode)."""
    E_bins = np.logspace(13, 16, 8)
    z_bins = np.linspace(4.0, 0.0, 8)
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.ascontiguousarray(np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128))
    u_arr = np.ascontiguousarray(np.array([0.5, 0.5, 0.0], dtype=np.complex128))
    m_nu_bg = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    bad_grid = np.zeros((8, 8, 3))
    bad_grid[3, 4, 1] = np.nan

    with pytest.raises(ValueError, match="thermal-rate grid.*m_Zd_eV=2e"):
        dps.propagate_spectrum(
            E_bins, z_bins, np.array([0.33, 0.67, 0.0]),
            1.5e-33, 0.315, 0.685, np.ascontiguousarray(U_pmns), M_sq,
            u_arr, 2.0e6, 1e-4, 0.1, m_nu_bg, 2.5e-13, 1.95e-4,
            True, 1.5, False, 0.0, False, 2.5, 0.0, 1.0, np.ones(3),
            1, 1, N_steps_MW=3, show_progress=False, verbose=0, n_cores=1,
            precomputed_Gamma_grid_all=bad_grid, N_steps_earth=3)


def test_frequentist_wrapper_returns_sentinel_on_kernel_failure(monkeypatch, tmp_path):
    """GenericLikelihoodWrapper must convert BOTH kernel failure modes established by the
    task-#3 contract into the 1e9 sentinel: a raised ValueError (the new loud domain
    check in propagate_spectrum) and a NaN chi2 (e.g. NaN flavor fractions reaching the
    detector likelihood). Uses a tiny mock grid so no real physics runs."""
    import src.core.likelihood as lk

    config = {
        "experiment": "icecube_mese_2025",
        "physics_constants": {"n_E_bins": 8, "n_z_bins": 8, "n_mw_paths": 1,
                              "n_earth_chords": 1, "N_steps_MW": 2, "N_steps_earth": 2},
        "start_values": {"m_Zd_eV": 1e-6, "g_D": 1e-4, "epsilon_prime": 0.1},
        "scan_settings": {"num_cores": 1},
    }
    wrapper = lk.GenericLikelihoodWrapper(config, ["m_Zd_eV"])

    def raise_domain_error(params, static_data):
        raise ValueError("Non-finite thermal-rate grid (Gamma) at m_Zd_eV=1e-06 ...")

    monkeypatch.setattr(lk, "compute_user_model_single", raise_domain_error)
    assert wrapper(1e-6) == 1e9

    monkeypatch.setattr(lk, "compute_user_model_single",
                        lambda params, static_data: (np.nan, np.nan, np.nan))
    assert wrapper(2e-6) == 1e9


# =============================================================================
# Semi-analytic thermal momentum integration (task #1)
# =============================================================================

def _dense_thermal_truth(E0, z, sum_G, sum_G_alpha, m_Zd, m_nu, T_nu0, eta_mass):
    """Independent brute-force reimplementation of the thermal-rate momentum integral:
    dense midpoint panels refined geometrically around the (closed-form) resonance
    window edge, evaluating the same closed-form I_ang in plain numpy. This is the
    'quadrature told where the window is' referee for the semi-analytic kernel."""
    T_nu = T_nu0 * (1.0 + z)
    MG = m_Zd**2 * sum_G / (12.0 * np.pi)
    MG2 = MG * MG
    M2 = m_Zd**2
    x = E0 * (1.0 + z)
    rates = np.zeros(3)
    if MG == 0.0:
        return rates
    for i in range(3):
        if eta_mass[i] <= 0:
            continue
        m = m_nu[i]
        m2 = m * m
        T_i = T_nu * eta_mass[i] ** (1.0 / 3.0)
        pmax = 20.0 * T_i
        delta = (M2 - m2) / (2.0 * x)
        cuts = [0.0, pmax]
        if delta > 0:
            pe = (m2 - delta**2) / (2 * delta) if delta < m else (delta**2 - m2) / (2 * delta)
            if 0 < pe < pmax:
                Ebg_e = np.sqrt(m2 + pe * pe)
                wz = MG / (2.0 * x * delta / Ebg_e)
                for fac in (1e6, 1e4, 1e2, 3.0):
                    for s in (-1, 1):
                        c = pe + s * fac * wz
                        if 0 < c < pmax:
                            cuts.append(c)
                cuts.append(pe)
        cuts = np.array(sorted(set(cuts)))
        total = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            n = 6000
            p = a + (np.arange(n) + 0.5) * (b - a) / n
            E_bg = np.sqrt(m2 + p**2)
            f = 1.0 / (np.exp(p / T_i) + 1.0)
            A = m2 + 2 * x * E_bg
            B = 2 * x * p
            u_p = A - M2 + B
            u_m = A - M2 - B
            C = 1.0 - (A - M2) / B
            D = 1.0 - (A - 2.0 * M2) / B
            F = C * M2 - MG2 / B
            I = (2.0 / B + (D / (2 * B)) * np.log((u_p**2 + MG2) / (u_m**2 + MG2))
                 + (F / (B * MG)) * (np.arctan(u_p / MG) - np.arctan(u_m / MG)))
            total += np.sum(p * p * f * np.maximum(I, 0.0)) * (b - a) / n
        rates += total / (2 * np.pi) * sum_G_alpha * eta_mass[i]
    return rates


def test_thermal_analytic_matches_dense_quadrature():
    """The semi-analytic kernel must match the independent dense-quadrature referee
    wherever rates are significant, across a g_D ladder spanning narrow to wide widths
    and cells targeted ON the resonance (where the old midpoint kernel carries 6-9%
    edge error). Acceptance: significance-weighted error
    |new - truth| / max(truth, 1e-3 * Gamma_max) < 1e-5 (measured headroom: ~2.5e-7).
    Cells far below significance sit on the closed form's own pointwise cancellation
    noise floor and are excluded by construction of the metric."""
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    m_nu = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    T_nu0, eps_p = 1.95e-4, 0.1

    for m_Zd in (2.0e6, 2.0e7):
        for g_D in (1e-8, 1e-4, 1.0):
            sum_G, sum_G_alpha, eta_mass = dps.compute_gamma_grid_inputs(
                U_pmns, u_arr, g_D, eps_p, np.ones(3))
            cells = [(1e13, 0.5), (1e15, 1.0), (1e16, 3.0)]
            for i in (0, 2):
                for fac in (0.5, 1.0):
                    delta = fac * m_nu[i]
                    x = (m_Zd**2 - m_nu[i]**2) / (2.0 * delta)
                    cells.append((x / 2.0, 1.0))
            rows = []
            for (E0, z) in cells:
                truth = _dense_thermal_truth(E0, z, sum_G, sum_G_alpha, m_Zd, m_nu, T_nu0, eta_mass)
                new = dps.thermal_averaged_rates_flavor_analytic(
                    E0, z, sum_G, sum_G_alpha, m_Zd, m_nu, T_nu0, eta_mass,
                    dps._GL16_X, dps._GL16_W, dps._GL10_X, dps._GL10_W, dps._FD16)
                rows.append((truth[0], new[0]))
            gmax = max(r[0] for r in rows)
            assert gmax > 0
            worst = max(abs(r[1] - r[0]) / max(r[0], 1e-3 * gmax) for r in rows)
            assert worst < 1e-5, f"m_Zd={m_Zd:.1e}, g_D={g_D:.0e}: weighted err {worst:.3e}"


def test_thermal_analytic_linear_scaling_in_sum_G():
    """Coupling scans rely on Gamma scaling linearly with sum_G off resonance (task #7's
    rescaling gate): at a window-closed cell the semi-analytic kernel must track the
    sum_G ratio to ~1e-6 per decade over the g_D range where rates are significant.
    (At g_D <= ~1e-7 the rates fall to the kernel's noise floor and linearity degrades
    to ~1e-3 -- documented, and exactly what the #7 element-wise gate protects against.)"""
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    m_nu = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    prev = None
    for g_D in (1e-2, 1e-3, 1e-4):
        sum_G, sum_G_alpha, eta_mass = dps.compute_gamma_grid_inputs(
            U_pmns, u_arr, g_D, 0.1, np.ones(3))
        r = dps.thermal_averaged_rates_flavor_analytic(
            1e15, 1.0, sum_G, sum_G_alpha, 2e6, m_nu, T_nu0=1.95e-4, eta_mass=eta_mass,
            glx=dps._GL16_X, glw=dps._GL16_W, glx10=dps._GL10_X, glw10=dps._GL10_W,
            fd16=dps._FD16)
        if prev is not None:
            sg_prev, r_prev = prev
            ratio = (r[0] / r_prev) / (sum_G / sg_prev)
            assert abs(ratio - 1.0) < 1e-6, f"g_D={g_D:.0e}: linearity broken at {ratio}"
        prev = (sum_G, r[0])


def test_build_thermal_grid_analytic_switch():
    """grid.thermal_analytic=False must reproduce the midpoint reference kernel
    bit-for-bit through build_thermal_grid; True (the default) selects the semi-analytic
    kernel, which must agree with the reference at the percent level where rates are
    significant (their difference is dominated by the reference's own resonance-edge
    quadrature error)."""
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    m_nu = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    sum_G, sum_G_alpha, eta_mass = dps.compute_gamma_grid_inputs(
        U_pmns, u_arr, 1e-4, 0.1, np.ones(3))
    E_bins = np.logspace(13, np.log10(3e16), 40)
    z_bins = np.linspace(4.0, 0.0, 40)

    ref = dps.precompute_all_thermal_rates(E_bins, z_bins, sum_G, sum_G_alpha, 2e6,
                                           m_nu, 1.95e-4, eta_mass)
    off = dps.build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, 2e6, m_nu, 1.95e-4,
                                 eta_mass, thermal_analytic=False)
    np.testing.assert_array_equal(off, ref)

    on = dps.build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, 2e6, m_nu, 1.95e-4,
                                eta_mass, thermal_analytic=True)
    assert not np.array_equal(on, ref)
    gmax = ref.max()
    rel = np.abs(on - ref) / np.maximum(ref, 1e-3 * gmax)
    assert rel.max() < 0.2, f"analytic vs midpoint disagree beyond the reference's own error: {rel.max():.3e}"
    assert np.isfinite(on).all()


# =============================================================================
# Two-component thermal-grid cache
# =============================================================================

def _tc_setup():
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    m_nu = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    _, _, eta_mass = dps.compute_gamma_grid_inputs(U_pmns, u_arr, 1e-4, 0.1, np.ones(3))
    E_bins = np.logspace(13, np.log10(3e16), 30)
    z_bins = np.linspace(4.0, 0.0, 30)
    return m_nu, eta_mass, E_bins, z_bins


def test_thermal_grid_cached_exact_path_matches_plain_build():
    """A cold-cache call must reproduce build_thermal_grid up to the (ulp-level)
    factoring of the sum_G_alpha weights out of the state sum, and the reference
    paths (thermal_analytic=False, n_z_thermal) must bypass the cache bit-exactly."""
    m_nu, eta_mass, E_bins, z_bins = _tc_setup()
    dps._tc_cache.clear()
    sum_G, sum_G_alpha, _ = dps.compute_gamma_grid_inputs(
        dps.construct_PMNS(0.5831, 0.1498, 0.857, 2.304),
        np.array([0.5, 0.5, 0.0], dtype=np.complex128), 1e-4, 0.1, np.ones(3))
    plain = dps.build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, 2e6, m_nu, 1.95e-4, eta_mass)
    cached = dps.thermal_grid_cached(E_bins, z_bins, sum_G, sum_G_alpha, 2e6, m_nu, 1.95e-4, eta_mass)
    np.testing.assert_allclose(cached, plain, rtol=1e-12)

    ref = dps.build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, 2e6, m_nu, 1.95e-4,
                                 eta_mass, thermal_analytic=False)
    byp = dps.thermal_grid_cached(E_bins, z_bins, sum_G, sum_G_alpha, 2e6, m_nu, 1.95e-4,
                                  eta_mass, thermal_analytic=False)
    np.testing.assert_array_equal(byp, ref)


def test_thermal_grid_cached_surrogate_verified_and_bracketed():
    """After exact builds at two bracket points and one interior point (which doubles
    as the verification sample), the surrogate must activate, agree with a fresh
    exact build inside the bracket to the gate accuracy, and NEVER be used outside
    the verified bracket (out-of-range requests fall back to exact builds)."""
    m_nu, eta_mass, E_bins, z_bins = _tc_setup()
    dps._tc_cache.clear()
    ones = np.ones(3)
    m_Zd = 2e6
    # bracket 1e-6 .. 1e-5 (~1 decade), interior verification at 3e-6
    for s in (1e-6, 1e-5, 3.16e-6):
        dps.thermal_grid_cached(E_bins, z_bins, s, ones, m_Zd, m_nu, 1.95e-4, eta_mass)
    state = next(iter(dps._tc_cache.values()))
    assert state["fit"] is not None, "surrogate did not activate after bracketed builds"
    assert state["fit"]["s_lo"] == 1e-6 and state["fit"]["s_hi"] == 1e-5

    # interior request: surrogate vs an out-of-band exact build
    s_test = 5e-6
    sur = dps.thermal_grid_cached(E_bins, z_bins, s_test, ones, m_Zd, m_nu, 1.95e-4, eta_mass)
    exact = dps.build_thermal_grid(E_bins, z_bins, s_test, ones, m_Zd, m_nu, 1.95e-4, eta_mass)[:, :, 0]
    sig = exact > 1e-3 * exact.max()
    rel = np.max(np.abs(sur[:, :, 0][sig] - exact[sig]) / exact[sig])
    assert rel < 5e-7, f"surrogate off gate inside verified bracket: {rel:.3e}"

    # out-of-range request must NOT use the fit (exact rebuild, point stored)
    n_pts = len(state["points"])
    dps.thermal_grid_cached(E_bins, z_bins, 1e-3, ones, m_Zd, m_nu, 1.95e-4, eta_mass)
    assert len(state["points"]) == n_pts + 1, "out-of-bracket request did not take the exact path"


def test_thermal_grid_cached_gate_rejects_wide_bracket():
    """A bracket wider than the measured validity of the two-component model must be
    refused by construction (spread cap), so the surrogate never activates from
    points too far apart to be trustworthy."""
    m_nu, eta_mass, E_bins, z_bins = _tc_setup()
    dps._tc_cache.clear()
    ones = np.ones(3)
    for s in (1e-9, 1e-2, 1e-5):     # 7-decade bracket, interior third point
        dps.thermal_grid_cached(E_bins, z_bins, s, ones, 2e6, m_nu, 1.95e-4, eta_mass)
    state = next(iter(dps._tc_cache.values()))
    assert state["fit"] is None, "surrogate activated across a bracket beyond the model's validity"


def test_thermal_grid_cached_exact_point_reuse():
    """A repeat request at an already-built sum_G must be served from the stored
    exact point (no rebuild) and must return the identical grid -- Minuit re-evaluates
    its incumbent point several times per profiled fit, so this is a real scan path."""
    m_nu, eta_mass, E_bins, z_bins = _tc_setup()
    dps._tc_cache.clear()
    weights = np.array([0.5, 1.0, 0.25])
    first = dps.thermal_grid_cached(E_bins, z_bins, 1e-6, weights, 2e6, m_nu, 1.95e-4, eta_mass)
    state = next(iter(dps._tc_cache.values()))
    n_pts = len(state["points"])
    again = dps.thermal_grid_cached(E_bins, z_bins, 1e-6, weights, 2e6, m_nu, 1.95e-4, eta_mass)
    assert len(state["points"]) == n_pts, "exact repeat rebuilt instead of reusing the stored point"
    np.testing.assert_array_equal(again, first)


def test_thermal_grid_cached_narrow_bracket_activates_within_gate():
    """Minuit-inner-fit-sized coupling moves (well below the historical 0.3-dex floor)
    must be able to activate the surrogate, and its output inside the verified bracket
    must agree with a fresh exact build to the gate accuracy. This is the case the
    original floor silently excluded: during real profiled scans every evaluation paid
    the full exact build because inner-fit steps never spanned 0.3 dex."""
    m_nu, eta_mass, E_bins, z_bins = _tc_setup()
    dps._tc_cache.clear()
    ones = np.ones(3)
    m_Zd = 2e6
    # ~0.08-dex bracket with an interior verification point: typical inner-fit moves
    for s in (1.0e-6, 1.2e-6, 1.1e-6):
        dps.thermal_grid_cached(E_bins, z_bins, s, ones, m_Zd, m_nu, 1.95e-4, eta_mass)
    state = next(iter(dps._tc_cache.values()))
    assert state["fit"] is not None, "narrow bracket did not activate the surrogate"
    assert state["failures"] == 0

    s_test = 1.15e-6
    sur = dps.thermal_grid_cached(E_bins, z_bins, s_test, ones, m_Zd, m_nu, 1.95e-4, eta_mass)
    assert len(state["points"]) == 3, "in-bracket request took the exact path"
    exact = dps.build_thermal_grid(E_bins, z_bins, s_test, ones, m_Zd, m_nu, 1.95e-4, eta_mass)[:, :, 0]
    sig = exact > 1e-3 * exact.max()
    rel = np.max(np.abs(sur[:, :, 0][sig] - exact[sig]) / exact[sig])
    assert rel < 5e-7, f"narrow-bracket surrogate off gate: {rel:.3e}"


def test_thermal_grid_cached_narrow_failures_do_not_disable():
    """Gate failures at sub-0.3-dex spreads must NOT count toward disabling the key:
    the failure counter exists to detect two-component model breakdown (a wide-bracket
    phenomenon), and a numerically degenerate narrow fit must not permanently turn the
    cache off. Exercised directly against _tc_try_fit with a synthetic non-conforming
    point set."""
    from dp_thermal import _tc_try_fit, _TC_MAX_FAILURES
    shape = (4, 4)
    U1 = np.full(shape, 1.0)
    U2 = np.full(shape, 1.001)
    U_bad = np.full(shape, 5.0)      # wildly off any two-component prediction
    for _ in range(_TC_MAX_FAILURES + 1):
        state = {"points": [(1.0e-6, U1), (1.05e-6, U_bad), (1.1e-6, U2)],
                 "fit": None, "failures": 0, "disabled": False}
        _tc_try_fit(state, 1.05e-6)   # ~0.02-dex bracket: narrow regime
        assert state["fit"] is None, "non-conforming narrow fit must not activate"
        assert state["failures"] == 0, "narrow-regime gate failure was counted"
        assert not state["disabled"]

    # The wide regime still disables after repeated model-breakdown failures.
    state = {"points": [(1.0e-7, U1), (1.05e-6, U_bad), (1.1e-5, U2)],
             "fit": None, "failures": 0, "disabled": False}
    for _ in range(_TC_MAX_FAILURES):
        _tc_try_fit(state, 1.05e-6)   # 2-dex bracket: wide regime
    assert state["disabled"], "wide-regime model breakdown must still disable the key"


# =============================================================================
# Earth step-averaged potential scheme (dev-earth)
# =============================================================================

def test_earth_layer_crossings_match_bruteforce():
    """The closed-form PREM layer-crossing positions along a chord must match a dense
    sign-change scan of r(l) - r_b for every boundary, on core-crossing, mid, and
    grazing chords."""
    depth = 1.5
    for costh in (-1.0, -0.7, -0.3, -0.05):
        baseline = dps.get_baseline_earth(depth, costh)
        got = dps._earth_layer_crossings(baseline, costh, depth)
        # brute force: dense r(l), find crossings of each boundary
        l = np.linspace(0.0, baseline, 2_000_001)
        f1 = baseline - l
        f2 = 6371.0 - depth
        r = np.sqrt(f1**2 + f2**2 + 2.0 * f1 * f2 * costh)
        expected = []
        for r_b in dps._PREM_BOUNDARIES_KM:
            s = np.sign(r - r_b)
            # strict sign product < 0: a sample sitting EXACTLY on a boundary (the
            # costh = -1 chord enters at r = 6371.0 precisely) is not an interior
            # crossing
            idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
            for i in idx:
                expected.append(0.5 * (l[i] + l[i + 1]))
        # exclude near-endpoint artifacts: at costh = -1 the chord's entry point sits
        # EXACTLY on the r = 6371 surface boundary (r(0) = 6371.0), which the dense
        # sign scan reports as a crossing; the closed form correctly returns only
        # interior crossings.
        expected = np.array(sorted(e for e in expected if 1e-6 * baseline < e < baseline * (1 - 1e-6)))
        assert len(got) == len(expected), f"costh={costh}: {len(got)} vs {len(expected)} crossings"
        if len(got):
            np.testing.assert_allclose(got, expected, atol=baseline / 1_000_000)


def test_avg_earth_density_matches_dense_average():
    """The layer-split GL step averages must equal dense numerical averages of the
    (discontinuous, piecewise-polynomial) density to high accuracy, and their step sum
    must reproduce the total path integral exactly for different step counts -- the
    property that makes the 'avg' scheme's LRI phase step-count-independent."""
    depth, costh = 1.5, -0.83
    baseline = dps.get_baseline_earth(depth, costh)
    n_amb = 1.36e-14
    cross = dps._earth_layer_crossings(baseline, costh, depth)

    totals = {}
    for N in (37, 250):
        nbar = dps._avg_earth_density_seq(baseline, costh, N, depth, n_amb, cross,
                                          dps._GL10_X, dps._GL10_W)
        dr = baseline / N
        # dense referee for a few steps, including ones containing a layer crossing
        check = {0, N // 2, N - 1}
        for c in cross[:3]:
            check.add(min(N - 1, int(c / dr)))
        for s in check:
            a, b = s * dr, (s + 1) * dr
            # jump-aware referee: adaptive quadrature on the smooth subsegments between
            # layer crossings (any fixed-step rule ACROSS a density discontinuity, or
            # even dense trapezoids on the curved smooth parts, referee the wrong way)
            edges = [a] + [c for c in cross if a < c < b] + [b]
            ref_int = 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                val, _ = integrate.quad(
                    lambda x: dps.spatial_total_density_earth(x, baseline, costh, depth, n_amb),
                    lo, hi, limit=200)
                ref_int += val
            ref = ref_int / dr
            assert abs(nbar[s] - ref) / ref < 1e-12, f"N={N} step {s}: {nbar[s]:.8e} vs {ref:.8e}"
        totals[N] = float(np.sum(nbar) * dr)
    # the path integral must be identical across step counts (fp-level)
    assert abs(totals[37] - totals[250]) / totals[250] < 1e-12


def test_earth_step_scheme_default_and_switch(tmp_path):
    """dev-earth default is the phase-exact 'avg' scheme; the historical 'left' scheme
    must remain selectable (and differ -- 'avg' corrects the Riemann phase error);
    unknown scheme names must be rejected loudly."""
    import copy
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["grid"].update(dict(n_E_bins=16, n_z_bins=16, n_mw_paths=1, n_earth_chords=2,
                            N_steps_MW=3, N_steps_earth=40))
    cfg["execution"].update(dict(n_cores=1, verbose=0))
    assert cfg["grid"]["earth_step_scheme"] == "avg"

    cfg_left = copy.deepcopy(cfg)
    cfg_left["grid"]["earth_step_scheme"] = "left"
    _, F_avg = dps.run_solver(cfg)
    _, F_left = dps.run_solver(cfg_left)
    assert np.isfinite(F_avg).all()
    assert not np.array_equal(F_left, F_avg)

    bad = copy.deepcopy(cfg)
    bad["grid"]["earth_step_scheme"] = "midpointish"
    with pytest.raises(ValueError, match="earth_step_scheme"):
        dps.run_solver(bad)


def test_earth_avg_scheme_step_count_certificate(tmp_path):
    """The core property of the 'avg' scheme, at test scale: results are essentially
    step-count-INDEPENDENT (the accumulated LRI phase equals Coef * integral(n dl)
    exactly for any N), while the 'left' scheme's Riemann phase error makes its
    results scatter with N. This is the certificate that justified reducing
    N_steps_earth (production evidence: N = 8..1000 agree to <= 4.5e-9 at the
    below/inband/strong validation points; see the dev-earth study)."""
    import copy
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["grid"].update(dict(n_E_bins=24, n_z_bins=16, n_mw_paths=1, n_earth_chords=3,
                            N_steps_MW=3))
    cfg["execution"].update(dict(n_cores=1, verbose=0))

    flux = {}
    for scheme in ("avg", "left"):
        for n in (25, 200):
            c = copy.deepcopy(cfg)
            c["grid"]["earth_step_scheme"] = scheme
            c["grid"]["N_steps_earth"] = n
            _, flux[(scheme, n)] = dps.run_solver(c)

    scatter_avg = np.max(np.abs(flux[("avg", 25)] - flux[("avg", 200)])
                         / np.maximum(np.abs(flux[("avg", 200)]), 1e-300))
    scatter_left = np.max(np.abs(flux[("left", 25)] - flux[("left", 200)])
                          / np.maximum(np.abs(flux[("left", 200)]), 1e-300))
    assert scatter_avg < 1e-6, f"avg scheme not step-count-stable: {scatter_avg:.3e}"
    assert scatter_avg < 0.01 * scatter_left, (
        f"certificate inverted: avg {scatter_avg:.3e} vs left {scatter_left:.3e}")


# =============================================================================
# GPU spatial path: v3 propagator port, mathematics validated with numpy backend
# =============================================================================

def test_gpu_batched_propagator_matches_cpu_numpy_backend():
    """_batched_propagator_gpu takes its array backend as a parameter and uses only
    numpy-compatible operations, so the v3 port's MATHEMATICS is validated here on the
    CPU by running it with numpy as the backend against propagator_3x3, one batch
    mixing every branch regime (masks must coexist within a single call):
    generic well-separated, rank-1-dominated deflation, constructed near-degenerate,
    Milky-Way-scale phases (node-exactness), and the phase-noise projector regime.
    Only execution under cupy itself remains for the skipped-on-CPU device test."""
    from gpu_spatial import _batched_propagator_gpu

    rng = np.random.default_rng(17)
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = U_pmns @ M_sq @ np.conjugate(U_pmns).T
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S = np.outer(u, np.conjugate(u))

    Hs, ts, regimes = [], [], []
    for _ in range(60):                                   # generic Hermitian
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        Hs.append((A + A.conj().T) / 2 * 10.0 ** rng.uniform(-20, -12))
        ts.append(10.0 ** rng.uniform(8, 13))
        regimes.append("generic")
    for _ in range(60):                                   # rank-1-dominated (deflation)
        E = 10.0 ** rng.uniform(13, 16.5)
        Hs.append(M_rot / (2.0 * E) + 10.0 ** rng.uniform(-19, -15) * S)
        ts.append(10.0 ** rng.uniform(9, 12))
        regimes.append("generic")
    for _ in range(40):                                   # constructed near-degenerate
        gap = 10.0 ** rng.uniform(-14, -4)
        lam = np.array([1.0, 1.0 + gap, 2.0]) * 1e-16
        Q = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
        Hd = Q @ np.diag(lam) @ Q.conj().T
        Hs.append((Hd + Hd.conj().T) / 2)
        ts.append(1e11)
        regimes.append("generic")
    for _ in range(40):                                   # MW-scale phases (node-exact)
        E = 10.0 ** rng.uniform(13, 16)
        Hs.append(M_rot / (2.0 * E))
        ts.append(10.0 ** rng.uniform(25, 27))
        regimes.append("large_phase")
    for logV in range(6, 23, 2):                          # phase-noise projector regime
        Hs.append(M_rot / (2.0 * 1e13) + (10.0 ** logV) * S)
        ts.append(1.29e11)
        regimes.append("noise")
    Hs.append(np.eye(3, dtype=np.complex128) * 2.5e-16)   # pure-phase branch
    ts.append(1e11)
    regimes.append("generic")
    Hs.append(M_rot / (2.0 * 1e15))                       # Taylor branch (tiny spread*t)
    ts.append(1.0)
    regimes.append("generic")

    H_batch = np.ascontiguousarray(np.stack(Hs))
    t_batch = np.array(ts)
    U_batch = _batched_propagator_gpu(np, H_batch, t_batch)

    out = np.zeros((3, 3), dtype=np.complex128)
    v_far = (u / np.linalg.norm(u)).reshape(3)
    worst_direct = 0.0
    worst_unit = 0.0
    worst_abs = 0.0
    worst_ray = 0.0
    for k in range(len(Hs)):
        dps.propagator_3x3(np.ascontiguousarray(H_batch[k]), float(t_batch[k]), out)
        Ub = U_batch[k]
        worst_unit = max(worst_unit, float(np.max(np.abs(Ub @ Ub.conj().T - np.eye(3)))))
        if regimes[k] == "generic":
            # small total phases: scalar and batched arithmetic agree to fp noise
            worst_direct = max(worst_direct, float(np.max(np.abs(Ub - out))))
        elif regimes[k] == "large_phase":
            # large phases: ulp-level arithmetic differences (e.g. abs(z)**2 vs
            # re^2+im^2) rotate phases at the shared representation limit; the
            # meaningful invariants are unitarity and the |U| structure
            worst_abs = max(worst_abs, float(np.max(np.abs(np.abs(Ub) - np.abs(out)))))
        else:
            # phase-noise projector regime: even |U| depends on the (machine-noise)
            # relative phase between the far and pair eigenvalues, so scalar and
            # batched legitimately disagree there; the meaningful invariants are
            # unitarity (checked above) and preservation of the far eigenvector's ray
            worst_ray = max(worst_ray, abs(1.0 - abs(np.vdot(v_far, Ub @ v_far))))
    assert worst_direct < 1e-9, f"batched vs scalar propagator: {worst_direct:.3e}"
    assert worst_unit < 1e-11, f"batched propagator non-unitary: {worst_unit:.3e}"
    assert worst_abs < 1e-4, f"batched |U| structure off at large phases: {worst_abs:.3e}"
    assert worst_ray < 1e-10, f"batched projector far-ray not preserved: {worst_ray:.3e}"


# =============================================================================
# Scan-infrastructure regression tests (checkpointing + deterministic retry jitter)
# =============================================================================

def test_run_segments_checkpoint_resume_and_stale_discard(tmp_path):
    """_run_segments must (a) persist each completed segment, (b) resume by executing
    ONLY the segments missing from a matching checkpoint, (c) discard a checkpoint
    whose header (scan identity/grid/segmentation) does not match, and (d) return
    results identical to an uncheckpointed run in all cases."""
    from src.engines.scanner import _run_segments

    calls = []

    def worker(seg):
        calls.append(tuple(seg))
        return [(float(v), {"p": float(v) * 2}) for v in seg]

    segments = [[1.0, 2.0], [3.0], [4.0, 5.0]]
    ckpt = str(tmp_path / "seg.jsonl")

    reference = _run_segments(worker, segments, None, "test")

    calls.clear()
    full = _run_segments(worker, segments, None, "test", checkpoint_path=ckpt, grid_signature="g1")
    assert [tuple(f) for f in full] == [tuple(r) for r in reference]
    assert len(calls) == 3
    assert os.path.exists(ckpt)  # the file is left for the caller to clean up

    # Simulate a crash after two segments: keep header + first two records only
    lines = open(ckpt).read().splitlines()
    with open(ckpt, "w") as f:
        f.write("\n".join(lines[:3]) + "\n")
    calls.clear()
    resumed = _run_segments(worker, segments, None, "test", checkpoint_path=ckpt, grid_signature="g1")
    assert calls == [tuple(segments[2])], "resume re-ran already-completed segments"
    assert [tuple(r[1].items()) for r in resumed] == [tuple(r[1].items()) for r in reference]
    assert [r[0] for r in resumed] == [r[0] for r in reference]

    # A different grid signature must invalidate the checkpoint entirely
    calls.clear()
    _run_segments(worker, segments, None, "test", checkpoint_path=ckpt, grid_signature="DIFFERENT")
    assert len(calls) == 3, "stale checkpoint (different grid) was trusted"
    os.remove(ckpt)


def test_scanner_checkpoint_resume(tmp_path, monkeypatch):
    """dark_portal_scanner.run_parameter_scan must persist completed points and, on
    resume, recompute ONLY the missing ones; the final output must contain every grid
    point exactly once, sorted, and the partial file must be removed on success. A
    stale checkpoint (changed config) must be ignored."""
    import dark_portal_scanner as scanner

    grid = np.array([1.0, 2.0, 4.0])
    cfg = {"execution": {"n_cores": 1}}
    out = str(tmp_path / "scan_test.txt")

    calls = []

    def fake_worker(args):
        name, val, config = args
        calls.append(val)
        return val, np.array([0.1 * val, 0.2 * val, 0.7 * val])

    monkeypatch.setattr(scanner, "scan_worker", fake_worker)
    monkeypatch.setattr(scanner, "HAS_TQDM", False)

    scanner.run_parameter_scan("g_D", grid, cfg, out)
    assert len(calls) == 3
    assert not os.path.exists(out + ".partial.jsonl"), "partial file not cleaned up"
    data = np.loadtxt(out)
    np.testing.assert_array_equal(data[:, 0], grid)

    # Pre-seed a partial with the first two points done; only the third must run
    header = scanner._scan_checkpoint_header("g_D", cfg)
    with open(out + ".partial.jsonl", "w") as f:
        f.write(json.dumps(header) + "\n")
        for v in grid[:2]:
            f.write(json.dumps({"val_hex": float(v).hex(), "val": float(v),
                                "ratios": [0.1 * v, 0.2 * v, 0.7 * v]}) + "\n")
    calls.clear()
    scanner.run_parameter_scan("g_D", grid, cfg, out)
    assert calls == [4.0], f"resume recomputed completed points: {calls}"
    data = np.loadtxt(out)
    np.testing.assert_array_equal(data[:, 0], grid)
    np.testing.assert_allclose(data[:, 1], 0.1 * grid, rtol=1e-5)

    # Stale checkpoint: different config hash -> everything recomputed
    with open(out + ".partial.jsonl", "w") as f:
        f.write(json.dumps(scanner._scan_checkpoint_header("g_D", {"execution": {"n_cores": 2}})) + "\n")
        f.write(json.dumps({"val_hex": float(grid[0]).hex(), "val": float(grid[0]),
                            "ratios": [9.9, 9.9, 9.9]}) + "\n")
    calls.clear()
    scanner.run_parameter_scan("g_D", grid, cfg, out)
    assert len(calls) == 3, "stale (different-config) checkpoint was trusted"


def test_global_fit_checkpoint_identity_and_load(tmp_path):
    """The global-fit checkpoint must load ONLY when the identity hash (floating set,
    physics, priors, start conditions) matches AND the stored fit was valid; anything
    else forces a fresh fit. Guards against silently reusing a minimum computed under
    different physics."""
    from src.main import global_fit_identity, load_global_fit_checkpoint

    cfg = {"experiment": "x", "physics_constants": {"m_lightest_eV": 0.01},
           "start_values": {"g_D": 1e-4}, "fixed_params": {}, "parameter_limits": {},
           "step_sizes": {}}
    ident = global_fit_identity(cfg, ["g_D"], asimov=False)
    assert ident == global_fit_identity(dict(cfg), ["g_D"], asimov=False)
    assert ident != global_fit_identity(cfg, ["g_D", "u_e"], asimov=False)
    assert ident != global_fit_identity(cfg, ["g_D"], asimov=True)
    cfg2 = {**cfg, "physics_constants": {"m_lightest_eV": 0.02}}
    assert ident != global_fit_identity(cfg2, ["g_D"], asimov=False)

    path = str(tmp_path / "global_fit.json")
    payload = {"identity": ident, "fval": -15.0, "valid": True,
               "values": {"g_D": 1.3e-4}, "errors": {"g_D": 5e-6}}
    with open(path, "w") as f:
        json.dump(payload, f)
    loaded = load_global_fit_checkpoint(path, ident)
    assert loaded is not None and loaded["fval"] == -15.0

    assert load_global_fit_checkpoint(path, "different") is None
    payload["valid"] = False
    with open(path, "w") as f:
        json.dump(payload, f)
    assert load_global_fit_checkpoint(path, ident) is None, "invalid fit was reused"
    with open(path, "w") as f:
        f.write("{corrupt")
    assert load_global_fit_checkpoint(path, ident) is None


def test_split_segments_min_length_floor():
    """min_segment_length must floor the warm-start segment length: at high worker
    counts the ~3-per-core rule otherwise produces single-point segments and
    warm-starting silently dies. Default (1) must reproduce the historical split."""
    from src.engines.scanner import _split_segments

    items = list(range(100))
    default = _split_segments(items, 200)
    assert len(default) == 100, "historical high-core split changed"
    assert _split_segments(items, 200, min_len=1) == default

    floored = _split_segments(items, 200, min_len=4)
    assert len(floored) == 25
    assert all(len(s) >= 4 for s in floored)
    assert [x for s in floored for x in s] == items, "floored split lost or reordered points"

    # floor larger than the grid: one segment, never zero
    assert _split_segments(items, 200, min_len=1000) == [items]


def test_minuit_maxcall_caps_stalled_fits():
    """minuit_maxcall must bound the per-migrad evaluation count of a hard inner fit
    (measured on a real scan: one uncapped point burned 870 evaluations against a
    26-eval median) while still returning a finite result; 0 keeps Minuit's own
    internal limit (historical behavior)."""
    from src.engines.optimizers import minimize_point

    calls = {"n": 0}

    def rosenbrock(w, x, y, z):
        calls["n"] += 1
        return (100 * (y - x**2)**2 + (1 - x)**2
                + 100 * (z - y**2)**2 + (1 - y)**2 + (w - 1.0)**2)

    names = ["w", "x", "y", "z"]
    starts = {"w": 1.0, "x": -1.5, "y": 2.5, "z": -1.5}
    errors = {p: 0.1 for p in names}
    limits = {p: (-5.0, 5.0) for p in names}

    calls["n"] = 0
    fval_uncapped, _ = minimize_point(1.0, "w", starts, errors, rosenbrock, limits,
                                      names, minuit_retries=1, retry_seed=3)
    n_uncapped = calls["n"]

    calls["n"] = 0
    fval_capped, _ = minimize_point(1.0, "w", starts, errors, rosenbrock, limits,
                                    names, minuit_retries=1, retry_seed=3,
                                    minuit_maxcall=20)
    n_capped = calls["n"]

    assert np.isfinite(fval_capped)
    assert n_capped < n_uncapped, f"cap did not reduce evals ({n_capped} vs {n_uncapped})"
    # ncall is an approximate per-iteration check, so allow generous overhead
    assert n_capped < 60, f"capped fit still ran {n_capped} evaluations"


def test_default_relic_neutrino_temperature_is_physical():
    """The relic-neutrino temperature must be (4/11)^(1/3) T_CMB, converted to eV.

    Until 2026-10-12 the default was 1.95e-4 eV: "1.95 K" entered as eV (2.26 K, 16% too
    hot). The thermal kernel's relic density follows from the Fermi-Dirac integral, so the
    absorption rates scale ~T^3 and were ~57% too high. Pins the default AND the shipped
    analysis configs, which carry their own copy of the value."""
    T_expected = (4.0 / 11.0) ** (1.0 / 3.0) * 2.7255 * 8.617333262e-5   # eV
    assert abs(dps.DEFAULT_CONFIG["cosmology"]["T_nu0_eV"] / T_expected - 1.0) < 1e-5

    cfg_dir = os.path.join(_FREQ_DIR, "config")
    for name in ("pion_2050_case_5_fixed_nuisance.json", "pion_2050_case_5_fixed_nuisance_mZD.json",
                 os.path.join("tests", "dark_portal_frequentist_config.json")):
        with open(os.path.join(cfg_dir, name)) as f:
            T_cfg = json.load(f)["physics_constants"]["T_nu0_eV"]
        assert abs(T_cfg / T_expected - 1.0) < 1e-5, f"{name} carries T_nu0_eV = {T_cfg}"


def test_default_cosmological_neutron_density_is_physical():
    """The cosmological dark-portal potential is sourced by neutrons only, which sit almost
    entirely in helium-4: n_n0 = n_b * Y_p / 2. Until 2026-10-12 the default was the total
    baryon density (0.25e-6 cm^-3), 8.1x too large. Pins the default and the three analysis
    configurations, each of which carries its own copy."""
    n_b, Y_p = 2.515e-7, 0.2458             # PDG 2024 (CMB); LBT Y_p project (2026)
    expected = n_b * Y_p / 2.0
    assert abs(dps.DEFAULT_CONFIG["cosmology"]["n_n0_cm3"] / expected - 1.0) < 1e-3

    cfg_dir = os.path.join(_FREQ_DIR, "config")
    for name in ("pion_2050_case_5_fixed_nuisance.json", "pion_2050_case_5_fixed_nuisance_mZD.json",
                 os.path.join("tests", "dark_portal_frequentist_config.json")):
        with open(os.path.join(cfg_dir, name)) as f:
            n_cfg = json.load(f)["physics_constants"]["n_n0_cm3"]
        assert abs(n_cfg / expected - 1.0) < 1e-3, f"{name} carries n_n0_cm3 = {n_cfg}"


def test_milky_way_neutron_fractions():
    """The Galactic model must convert stellar mass at 0.20 neutrons per nucleon and the hot
    halo's electron density at 0.15 neutrons per electron (not the 0.5 of rock used until
    2026-10-12). Checked at the Sun's position against the model's own components."""
    import dp_spatial
    assert dp_spatial._MW_NEUTRONS_PER_NUCLEON_STARS == 0.20
    assert dp_spatial._MW_NEUTRONS_PER_ELECTRON_GAS == 0.15
    # Msun/pc^3 -> nucleons/cm^3
    msun_g, pc_cm, m_p_g = 1.98847e33, 3.08568e18, 1.67262e-24
    assert abs(dp_spatial._NUCLEONS_PER_CM3_PER_MSUN_PC3 / (msun_g / pc_cm**3 / m_p_g) - 1.0) < 1e-4

    # At the Sun (r = 0): R_gal = R_0 = 8.33 kpc, z = 0.
    R0 = 8.33
    rho_bulge = (94.1 / (1.0 + R0 / 0.075)**1.8) * np.exp(-(R0 / 2.1)**2)
    rho_thin = (753.0 / 600.0) * np.exp(-R0 / 3.0)
    rho_thick = (182.0 / 1800.0) * np.exp(-R0 / 3.5)
    n_e_gas = 0.46 * (1.0 + (R0 / 0.35)**2)**(-1.065)
    expected = (rho_bulge + rho_thin + rho_thick) * 40.464 * 0.20 + n_e_gas * 0.15
    got = dp_spatial.calc_MW_neutron_density(0.0, 0.0, 0.0)
    assert abs(got / expected - 1.0) < 1e-6, (got, expected)


def test_resonant_absorption_scales_with_relic_temperature():
    """The absorption lives in the resonant dip, and the relic density there comes from the
    kernel's Fermi-Dirac integral, so the ENERGY-INTEGRATED dip scales as T^3 while the
    relic momentum spread sets the dip's shape: depth ~T^2, width ~T. Measured 2026-10-12:
    the old 1.95e-4 eV default inflated the integrated absorption by 1.573 (T^3 = 1.574)
    and the peak by 1.353 (T^2 = 1.353). Window: +-10% around the at-rest resonance of the
    0.07 eV state, E_res = m_Zd^2 / (2 m), clear of the two lighter states' dips."""
    m_Zd = 3.0e7
    m_nu = np.array([0.05, 0.051, 0.07])
    E_bins = m_Zd**2 / (2 * 0.07) * np.linspace(0.90, 1.10, 2001)
    z_bins = np.array([0.0])
    sum_G, sum_G_alpha, eta_mass = 1e-8, np.array([0.5e-8, 0.3e-8, 0.2e-8]), np.ones(3)
    T_old, T_new = 1.95e-4, dps.DEFAULT_CONFIG["cosmology"]["T_nu0_eV"]
    G = {T: dps.build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, m_Zd, m_nu, T, eta_mass)[0, :, 0]
         for T in (T_old, T_new)}
    r = T_old / T_new
    integral_ratio = np.trapz(G[T_old], E_bins) / np.trapz(G[T_new], E_bins)
    assert abs(integral_ratio / r**3 - 1.0) < 2e-3, integral_ratio
    assert abs((G[T_old].max() / G[T_new].max()) / r**2 - 1.0) < 1e-2


def test_retry_seed_deterministic_and_default_preserved():
    """retry_seed >= 0 must make the retry-jitter RNG stream a pure function of
    (base seed, parameter, grid value) -- identical across calls and processes --
    while distinct grid values still get distinct streams. retry_seed < 0 keeps the
    historical per-call os.urandom seeding (streams differ call to call)."""
    from src.engines.optimizers import _seed_retry_rng

    _seed_retry_rng(7, "m_Zd_eV", float(1e6).hex())
    a = np.random.random(4)
    _seed_retry_rng(7, "m_Zd_eV", float(1e6).hex())
    b = np.random.random(4)
    np.testing.assert_array_equal(a, b)

    _seed_retry_rng(7, "m_Zd_eV", float(2e6).hex())
    c = np.random.random(4)
    assert not np.array_equal(a, c), "different grid values shared a jitter stream"

    _seed_retry_rng(8, "m_Zd_eV", float(1e6).hex())
    d = np.random.random(4)
    assert not np.array_equal(a, d), "different base seeds shared a jitter stream"

    _seed_retry_rng(-1, "m_Zd_eV", float(1e6).hex())
    e = np.random.random(4)
    _seed_retry_rng(-1, "m_Zd_eV", float(1e6).hex())
    f = np.random.random(4)
    assert not np.array_equal(e, f), "default (-1) unexpectedly deterministic"


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
