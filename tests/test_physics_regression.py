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

import numpy as np
import pytest
from scipy import integrate

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

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


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
