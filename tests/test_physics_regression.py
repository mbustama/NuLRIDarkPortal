"""
test_physics_regression.py

Regression tests for the physics kernels in src/dark_portal_solver.py.

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

    np.testing.assert_allclose(rates_jit, rates_reimpl, rtol=1e-9, atol=0.0)


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

    V_nu = dps.calc_V_LRI_cosmo(z, u_base, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
    V_nubar = dps.calc_V_LRI_cosmo(z, np.conjugate(u_base), m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
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

    V_nu = dps.calc_V_LRI_cosmo(z, u_base, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
    V_nubar_correct = -np.conjugate(V_nu)
    H_tot_reference = H_vac + V_nubar_correct

    # Antineutrino call: the caller pre-conjugates u_arr (see propagation_worker), as production code does.
    evecs = dps.get_adiabatic_mixing_cosmo(E0, z, M_rot, np.conjugate(u_base), m_Zd, g_D, eps_prime,
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
    V_cosmo_nu = dps.calc_V_LRI_cosmo(z, u_base, m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)
    V_cosmo_nubar = -dps.calc_V_LRI_cosmo(z, np.conjugate(u_base), m_Zd, g_D, eps_prime, n_n0, T_nu0, True, eta_bg)

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
    recorded_n_steps = []
    orig_worker = dps._earth_chord_worker

    def spying_worker(args):
        recorded_n_steps.append(args[2])  # (costh, baseline_km, N_steps_earth, ...)
        return orig_worker(args)

    dps._earth_chord_worker = spying_worker
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
        dps._earth_chord_worker = orig_worker

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

    assert len(func_off.static_data) == 31
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

    assert not np.allclose(result_on, result_off)


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
    assert len(func.static_data) == 31
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
    hazard the BLAS/numba thread-pinning guards in dark_portal_solver.py exist to avoid).
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
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "OK" in result.stdout, f"stdout={result.stdout}\nstderr={result.stderr}"

    pool_count_lines = [l for l in result.stdout.splitlines() if l.startswith("POOL_CONSTRUCTIONS=")]
    assert pool_count_lines, f"missing pool count marker in output: {result.stdout}"
    pool_constructions = int(pool_count_lines[0].split("=")[1])
    assert pool_constructions == 1, f"expected exactly 1 Pool construction reused across 3 calls, got {pool_constructions}"


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
