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

# np.trapz is gone from NumPy 2.x (np.trapezoid since 2.0); the laptop and Tycho pin NumPy 1.26,
# which has only np.trapz, and CI also runs the latest NumPy (.github/workflows/tests.yml)
_trapz = getattr(np, "trapezoid", None) or np.trapz
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
    cfg["grid"]["n_sun_paths"] = 1
    cfg["grid"]["n_earth_chords"] = 1
    cfg["execution"]["verbose"] = 0
    cfg["execution"]["n_cores"] = 1
    cfg["standard_mixing"]["use_SM_potential"] = True

    cfg_lo = copy.deepcopy(cfg)
    cfg_lo["standard_mixing"]["n_e_earth_cm3"] = 1.0e23
    cfg_hi = copy.deepcopy(cfg)
    cfg_hi["standard_mixing"]["n_e_earth_cm3"] = 5.0e24

    _, F_lo = dps.run_solver(cfg_lo)
    _, F_hi = dps.run_solver(cfg_hi)

    # A wiring check: the output must respond, well above rounding. (The response is small,
    # ~1e-5 relative: the SM potential is flavor-diagonal and MSW effects are suppressed at
    # these energies. Before 2026-10-12 the runaway cosmological cascade amplified it.)
    assert np.max(np.abs(F_lo - F_hi) / np.abs(F_hi)) > 1e-8


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
                                    show_progress=False, verbose=0, n_cores=1)
    finally:
        dp_spatial._earth_chord_worker = orig_worker

    assert len(set(recorded_n_steps)) == 1, f"Earth chords received different step counts: {recorded_n_steps}"


# =============================================================================
# 5. The adiabatic Galactic map (replaced init_mw_density_matrix on 2026-10-13)
# =============================================================================

def test_galactic_map_matches_reimplementation():
    """dp_transit.galactic_map gives each energy bin the cosmological populations on the local
    eigenstates, W diag(F) W^dag with W the eigenvectors of M_rot/2E + Coef n_bg S in ascending
    order. An independent numpy reimplementation (eigh, at a potential where it is accurate)
    pins the convention and the physics."""
    from dp_transit import galactic_map, texture_scales
    N_E = 12
    E_bins = np.logspace(13, 16, N_E)
    rng = np.random.default_rng(0)
    F_mass_arrival = rng.uniform(1e18, 1e22, size=(N_E, 3))
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = np.ascontiguousarray(U_pmns @ M_sq @ np.conjugate(U_pmns).T)
    u_arr = np.array([0.5, 0.3 * np.exp(1j * 0.4), 0.2], dtype=np.complex128)
    S = np.ascontiguousarray(np.outer(u_arr, np.conjugate(u_arr)))
    for sign in (1.0, -1.0):
        Coef, n_bg = sign * 3.7e-3, 1.3e-14          # V ~ 5e-17 eV: comparable to the vacuum term
        Q, s, Mp, floor, _, _ = texture_scales(M_rot, np.sign(Coef) * S)
        rho = np.empty((N_E, 3, 3), dtype=np.complex128)
        galactic_map(E_bins, F_mass_arrival, 2.0 * abs(Coef), n_bg, Q, Mp, s, floor, rho)
        for i in range(N_E):
            H = M_rot / (2.0 * E_bins[i]) + Coef * n_bg * S
            w, v = np.linalg.eigh(H)
            expected = v @ np.diag(F_mass_arrival[i]) @ np.conjugate(v).T
            np.testing.assert_allclose(rho[i], expected, rtol=1e-10, atol=1e-10 * F_mass_arrival[i].max())


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
# 7b. Configurable potential structure matrices (U_LRI_struct / V_SM_struct)
# =============================================================================

def _propagate_small(u_arr, is_anti, U_LRI_struct=None, V_SM_struct=None, use_SM=False):
    """Minimal, fast propagate_spectrum call used by the structure-override tests: neutrino
    (is_anti False) or antineutrino sources alone, everything in neutrino convention
    (propagate_spectrum applies the antineutrino transforms itself). Returns the flux at
    the detector, neutrinos plus antineutrinos."""
    E_bins = np.logspace(13, 16, 12)
    z_bins = np.linspace(4.0, 0.0, 12)
    rho_src = np.array([0.333333, 0.666667, 0.0])
    U_pmns = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58),
                                                     np.radians(49.1), np.radians(132.0)))
    M_sq = np.ascontiguousarray(np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128))
    H0 = 67.4 * dps.km_per_s / dps.Mpc_to_eV_inv
    m_nu_bg = np.array([0.01, np.sqrt(0.01**2 + 7.41e-5), np.sqrt(0.01**2 + 2.511e-3)])

    F_nu, F_nb = dps.propagate_spectrum(
        E_bins, z_bins, rho_src, H0, 0.315, 0.685, U_pmns, M_sq,
        np.ascontiguousarray(u_arr), 2.0e6, 1e-4, 0.1, m_nu_bg, 0.25e-6 * dps.cm_inv3_to_eV3, 1.95e-4,
        True, 1.5, use_SM, 1.5e24 * dps.cm_inv3_to_eV3,
        (0.0, 1.0) if is_anti else (1.0, 0.0), 2.5, 0.0, 1.0, np.ones(3), 1, 1,
        show_progress=False, verbose=0, n_cores=1,
        N_steps_earth=5, U_LRI_struct=U_LRI_struct, V_SM_struct=V_SM_struct)
    return F_nu + F_nb


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
    cfg["grid"].update(dict(n_E_bins=12, n_z_bins=12, n_sun_paths=1, n_earth_chords=1,
                            N_steps_earth=5))
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
    """propagator_3x3 (spectral sum over the deflated eigenbasis) must agree with the
    eigh construction to fp noise across random
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
    term. The deflated eigenbasis must therefore stay accurate across gaps from harmless
    (1e-4 relative) down to essentially exact degeneracy (1e-14), where the splitting is
    below double-precision resolution and the pair is propagated as degenerate."""
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
    construction. The propagator must handle it at full accuracy and stay unitary, across
    the potential-to-vacuum ratio range the Earth transit spans."""
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
    assert worst_unit < 1e-14


def test_propagator_3x3_unitary_at_milky_way_phases():
    """Milky-Way-regime regression: per-step phases lam*t reach ~1e10 radians (gap phases
    ~1e9), where double precision defines a phase only to ~1e-5 rad. U must stay unitary
    to rounding regardless -- any amplitude error compounds over the steps (an earlier
    polynomial form that only approximately interpolated the computed phases gave 1.3e-4
    end-to-end) -- and agree with the eigh construction on |U|."""
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
    assert worst_unit < 1e-14, f"non-unitary at large phases: {worst_unit:.3e}"
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

    # Near-degenerate spectra take the eigh fallback. Within the nearly degenerate pair the basis is
    # determined only to ~eps/gap (up to 1e-2 here), so two LAPACK builds -- numba's and numpy's, or
    # one build on two CPUs -- may pick different pairs (a GitHub runner did, 2026-10-13); what is
    # fixed is the projector onto the pair, the isolated column, and each column being an eigenvector.
    for _ in range(100):
        gap = 10.0 ** rng.uniform(-14, -6)
        lam = np.array([1.0, 1.0 + gap, 2.0]) * 1e-16
        Q = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]
        H = np.ascontiguousarray((Q @ np.diag(lam) @ Q.conj().T + (Q @ np.diag(lam) @ Q.conj().T).conj().T) / 2)
        U = dps.eigenvectors_3x3(H)
        evals_ref, evecs_ref = np.linalg.eigh(H)
        P_pair = U[:, :2] @ U[:, :2].conj().T
        P_ref = evecs_ref[:, :2] @ evecs_ref[:, :2].conj().T
        np.testing.assert_allclose(P_pair, P_ref, atol=1e-12)
        np.testing.assert_allclose(np.abs(U[:, 2]) ** 2, np.abs(evecs_ref[:, 2]) ** 2, atol=1e-12)
        for k in range(3):
            assert np.max(np.abs(H @ U[:, k] - evals_ref[k] * U[:, k])) < 1e-12 * 2e-16


def _cascade_step_setup(N_E=40, seed=5):
    """Synthetic inputs for one call of step_cosmological_pair: random mass-state fluxes of
    both kinds and the sigma-tables of neutrinos and antineutrinos."""
    E_bins = np.logspace(13, 16, N_E)
    rng = np.random.default_rng(seed)
    F_nu = rng.uniform(1e18, 1e22, size=(N_E, 3))
    F_nb = rng.uniform(1e18, 1e22, size=(N_E, 3))
    U_pmns = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58),
                                                     np.radians(49.1), np.radians(132.0)))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    u_arr = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    c_z = 1e-30
    sigma_nodes = np.geomspace(E_bins[0] * c_z * 0.5, E_bins[-1] * c_z * 2.0, 512)
    log_s0 = np.log(sigma_nodes[0])
    inv_dlogs = (len(sigma_nodes) - 1) / np.log(sigma_nodes[-1] / sigma_nodes[0])
    tables = []
    for anti in (False, True):
        U = np.conjugate(U_pmns) if anti else U_pmns
        u = np.conjugate(u_arr) if anti else u_arr
        M_rot = np.ascontiguousarray(U @ M_sq @ np.conjugate(U).T)
        S = np.outer(u, np.conjugate(u))
        Q_t, s_t, Mp_t, floor_t, _, _ = dps.texture_scales(M_rot, np.ascontiguousarray(-S if anti else S))
        P_table = dps._build_sigma_projection_table(Q_t, Mp_t, s_t, floor_t, sigma_nodes)
        tables.append((P_table, log_s0, inv_dlogs, -c_z if anti else c_z))
    return E_bins, F_nu, F_nb, rng, tables


def _one_pair_step(E_bins, F_nu, F_nb, Gamma, dt, Phi_0, tables, w=(0.5, 0.5),
                   BR=np.array([0.4, 0.35, 0.25])):
    W3, pref3, self_same = dps.cascade_kernel_weights(E_bins)
    W1, W2, c1, c2, self_cross = dps.cross_kernel_weights(E_bins)
    N = len(E_bins)
    P_nu, P_nb = np.zeros((N, 3, 3)), np.zeros((N, 3, 3))
    R_nu, R_nb = np.zeros(N), np.zeros(N)
    F_nu, F_nb = F_nu.copy(), F_nb.copy()
    dps.step_cosmological_pair(0.5, dt, 1.0, E_bins, F_nu, F_nb, w[0], w[1], np.array([0.333, 0.667, 0.0]),
                               *tables[0], *tables[1], 2.5, Phi_0, BR, W3, pref3, self_same,
                               W1, W2, c1, c2, self_cross, P_nu, P_nb, np.ascontiguousarray(Gamma), R_nu, R_nb,
                               tables[0][3], tables[1][3], np.zeros_like(P_nu), np.zeros_like(P_nb))
    return F_nu, F_nb, R_nu, R_nb, P_nu, P_nb


def test_cascade_kernel_weights_are_exact_bin_integrals():
    """The discrete cascade kernel must be the exact bin integral of 3 E^2 / E_j^3 on the
    geometric grid: its fractions add up to 1 - (E_lo,0 / E_j)^3, the secondaries landing
    above the grid's lower edge, and never more. The midpoint weights used until 2026-10-12
    gave each parent 1 + 1.5 dlnE secondaries (1.2% created per generation)."""
    from scipy.integrate import quad
    E = np.logspace(13, 16.5, 300)
    W, pref, selff = dps.cascade_kernel_weights(E)
    dE = dps.compute_dE_geometric(E)
    h = np.sqrt(E[1] / E[0])
    for j in (0, 1, 7, 150, 299):
        kept = selff[j] + sum(pref[i] * dE[i] * W[j] / dE[j] for i in range(j))
        assert abs(kept - (1.0 - (E[0] / h / E[j])**3)) < 1e-12, j
        if j > 0:
            i = j // 2
            ref = quad(lambda e: 3 * e**2 / E[j]**3, E[i] / h, E[i] * h)[0]
            assert abs(pref[i] * dE[i] * W[j] / dE[j] / ref - 1.0) < 1e-12
        ref_self = quad(lambda e: 3 * e**2 / E[j]**3, E[j] / h, E[j])[0]
        assert abs(selff[j] / ref_self - 1.0) < 1e-12


def test_cross_kernel_weights_are_exact_bin_integrals():
    """The secondary of the other kind carries 3 (E_j - E)^2 / E_j^3 (the remaining quarter
    of the energy): its discrete weights must be the exact bin integrals, adding up to
    (1 - E_lo,0 / E_j)^3, everything above the grid's lower edge."""
    from scipy.integrate import quad
    E = np.logspace(13, 16.5, 300)
    W3, pref3, _ = dps.cascade_kernel_weights(E)
    W1, W2, c1, c2, self_cross = dps.cross_kernel_weights(E)
    dE = dps.compute_dE_geometric(E)
    h = np.sqrt(E[1] / E[0])

    def frac(i, j):
        return (c1[i] * W1[j] - c2[i] * W2[j] + pref3[i] * W3[j]) * dE[i] / dE[j]
    for j in (0, 1, 7, 150, 299):
        kept = self_cross[j] + sum(frac(i, j) for i in range(j))
        assert abs(kept - (1.0 - E[0] / h / E[j])**3) < 1e-10, j
        for i in {0, j // 2, max(j - 1, 0)}:
            if i < j:
                ref = quad(lambda e: 3 * (E[j] - e)**2 / E[j]**3, E[i] / h, E[i] * h, epsrel=1e-13)[0]
                assert abs(frac(i, j) / ref - 1.0) < 1e-8, (i, j)
        ref_self = quad(lambda e: 3 * (E[j] - e)**2 / E[j]**3, E[j] / h, E[j], epsrel=1e-13)[0]
        assert abs(self_cross[j] / ref_self - 1.0) < 1e-10


def test_cascade_step_balances_number_at_any_optical_depth():
    """One cosmological step must balance the number of neutrinos plus antineutrinos
    exactly, whatever the optical depth per step: every absorption removes one particle and
    emits two (one of each kind), so after = before + injected + absorbed - (secondaries
    lost below the grid). Until 2026-10-12 the regenerated source was each bin's initial
    rate Gamma F held over the whole step (up to 1e70 too many particles); until 2026-10-13
    only the secondary of the absorbed particle's own kind was kept."""
    E_bins, F_nu0, F_nb0, rng, tables = _cascade_step_setup()
    dE = dps.compute_dE_geometric(E_bins)
    x0 = E_bins[0] / np.sqrt(E_bins[1] / E_bins[0]) / E_bins        # E_lo,0 / E_j
    lost_frac = x0**3 + (1.0 - (1.0 - x0)**3)                        # same-kind + other-kind losses
    dt = 1.0
    for gdt_scale in (1e-8, 1e-2, 1.0, 1e3, 1e10):
        Gamma = rng.uniform(0.1, 1.0, size=(len(E_bins), 3)) * gdt_scale / dt
        for Phi_0, w in ((0.0, (0.5, 0.5)), (1e50, (0.5, 0.5)), (1e50, (1.0, 0.0))):
            F_nb_in = F_nb0 if w[1] > 0 else np.zeros_like(F_nb0)
            F_nu1, F_nb1, R_nu, R_nb, P_nu, P_nb = _one_pair_step(E_bins, F_nu0, F_nb_in, Gamma, dt, Phi_0, tables, w)
            inj = Phi_0 * (E_bins * 1.5)**(-2.5) * 1.5 * dt           # z = 0.5, R_z = 1, dE_z/dE = 1.5; weights sum to 1
            absorbed = (R_nu + R_nb) * dt * dE
            before = np.sum((F_nu0 + F_nb_in).sum(axis=1) * dE) + np.sum(inj * dE)
            after = np.sum((F_nu1 + F_nb1).sum(axis=1) * dE)
            expected = before + np.sum(absorbed) - np.sum(absorbed * lost_frac)
            assert abs(after - expected) <= 1e-11 * max(before, expected), (gdt_scale, Phi_0, w, after, expected)
            assert np.all(F_nu1 >= 0.0) and np.all(F_nb1 >= 0.0)
            if w[1] == 0.0 and gdt_scale >= 1.0:
                assert np.sum(F_nb1) > 0.0                           # antineutrinos from neutrino absorptions


def test_cascade_step_limits():
    """Without absorption the step is pure injection, F + w S dt for each kind. For a small
    optical depth per step it agrees with the explicit first-order update (rates G F held
    over the step, each absorption feeding both kinds) to O(Gamma dt)."""
    E_bins, F_nu0, F_nb0, rng, tables = _cascade_step_setup()
    dt = 2.0
    w = (0.3, 0.7)
    F_nu1, F_nb1, R_nu, R_nb, P_nu, P_nb = _one_pair_step(E_bins, F_nu0, F_nb0, np.zeros((len(E_bins), 3)),
                                                          dt, 1e50, tables, w)
    inj_spec = 1e50 * (E_bins * 1.5)**(-2.5) * 1.5      # z = 0.5: per unit observed energy, dE_z/dE = 1.5
    rho = np.array([0.333, 0.667, 0.0])
    np.testing.assert_allclose(F_nu1, F_nu0 + w[0] * np.einsum("iak,a->ik", P_nu, rho) * inj_spec[:, None] * dt, rtol=1e-14)
    np.testing.assert_allclose(F_nb1, F_nb0 + w[1] * np.einsum("iak,a->ik", P_nb, rho) * inj_spec[:, None] * dt, rtol=1e-14)
    assert np.all(R_nu == 0.0) and np.all(R_nb == 0.0)
    # small optical depth: the explicit update
    BR = np.array([0.4, 0.35, 0.25])
    Gamma = rng.uniform(0.1, 1.0, size=(len(E_bins), 3)) * 1e-7 / dt
    F_nu1, F_nb1, R_nu, R_nb, P_nu, P_nb = _one_pair_step(E_bins, F_nu0, F_nb0, Gamma, dt, 0.0, tables, w, BR)
    W3, pref3, self_same = dps.cascade_kernel_weights(E_bins)
    W1, W2, c1, c2, self_cross = dps.cross_kernel_weights(E_bins)

    def above(x):
        return np.concatenate([np.cumsum(x[::-1])[::-1][1:], [0.0]])
    G_nu = np.einsum("iak,ia->ik", P_nu, Gamma)
    G_nb = np.einsum("iak,ia->ik", P_nb, Gamma)
    Rn = np.sum(G_nu * F_nu0, axis=1)
    Rb = np.sum(G_nb * F_nb0, axis=1)
    same_n = pref3 * above(Rn * W3) + self_same * Rn
    same_b = pref3 * above(Rb * W3) + self_same * Rb
    cross_n = c1 * above(Rn * W1) - c2 * above(Rn * W2) + pref3 * above(Rn * W3) + self_cross * Rn
    cross_b = c1 * above(Rb * W1) - c2 * above(Rb * W2) + pref3 * above(Rb * W3) + self_cross * Rb
    F_nu_expl = F_nu0 * (1 - G_nu * dt) + np.einsum("iak,a->ik", P_nu, BR) * (same_n + cross_b)[:, None] * dt
    F_nb_expl = F_nb0 * (1 - G_nb * dt) + np.einsum("iak,a->ik", P_nb, BR) * (same_b + cross_n)[:, None] * dt
    # the exact step differs from the explicit one at O(G dt) of the regenerated part, which
    # in the lowest bins is up to ~1e-3 of the flux (random spectra): ~1e-10 relative
    np.testing.assert_allclose(F_nu1, F_nu_expl, rtol=1e-10, atol=0)
    np.testing.assert_allclose(F_nb1, F_nb_expl, rtol=1e-10, atol=0)
    # the exact absorbed number also includes secondaries absorbed within the step (up to
    # ~1e-5 of the flux in the lowest bins, where the other kind's soft secondaries collect)
    np.testing.assert_allclose(R_nu, Rn, rtol=1e-5)
    np.testing.assert_allclose(R_nb, Rb, rtol=1e-5)


def test_neutrino_sources_produce_antineutrinos_only_through_absorption():
    """With neutrino sources alone, antineutrinos reach the detector only as secondaries of
    absorptions: none without dark-portal interactions, some with them. (Since 2026-10-14 the Z mixing
    alone couples the mediator to every neutrino, so 'without interactions' means u = 0 AND eps' = 0.)"""
    E_bins = np.logspace(13, 16, 30)
    z_bins = np.linspace(4.0, 0.0, 30)
    U_pmns = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58),
                                                     np.radians(49.1), np.radians(132.0)))
    M_sq = np.ascontiguousarray(np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128))
    H0 = 67.4 * dps.km_per_s / dps.Mpc_to_eV_inv
    m_nu_bg = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    out = []
    for u, eps in ((0.0, 0.0), (0.5, 1e-4)):
        u_arr = np.ascontiguousarray(np.array([u, u, 0.0], dtype=np.complex128))
        out.append(dps.propagate_spectrum(
            E_bins, z_bins, np.array([1 / 3, 2 / 3, 0.0]), H0, 0.315, 0.685, U_pmns, M_sq,
            u_arr, 2.0e6, 1e-4, eps, m_nu_bg, 3.091e-8 * dps.cm_inv3_to_eV3, 1.676389e-4,
            True, 1.5, False, 0.0, (1.0, 0.0), 2.5, 0.0, 1.0, np.ones(3), 1, 2,
            show_progress=False, verbose=0, n_cores=1, N_steps_earth=5,
            earth_step_scheme="avg", earth_zenith_average="phase"))
    (F_nu0, F_nb0), (F_nu1, F_nb1) = out
    assert np.all(F_nb0 == 0.0) and np.sum(F_nu0) > 0.0
    assert np.sum(F_nb1) > 1e-3 * np.sum(F_nu1)


def test_solver_never_returns_more_energy_than_injected(tmp_path):
    """End to end, at the three MeV benchmark couplings: every absorption hands its energy to
    its two secondaries (3/4 and 1/4 on average), so the energy flux at the detector equals
    the no-interaction one less what falls below the grid -- never more, up to the placement
    of secondaries at bin centres (2e-4 on this 60-bin grid) -- and the energy-averaged
    flavor fractions converge in the redshift step. Before 2026-10-12 the energy flux came
    out up to 1e67 times the injected one."""
    import copy
    base = dps.create_default_config(str(tmp_path / "cfg.json"))
    base["grid"].update(dict(n_E_bins=60, n_sun_paths=1, n_earth_chords=2, N_steps_earth=10))
    base["execution"].update(dict(n_cores=1, verbose=0))

    def run(nz, **dp):
        c = copy.deepcopy(base)
        c["grid"]["n_z_bins"] = nz
        c["dark_portal"].update(dp)
        return dps.run_solver(c)
    E, F0 = run(100, u_e=0.0, u_mu=0.0, epsilon_prime=0.0)     # no coupling at all (eps' also couples, 2026-10-14)
    dE = dps.compute_dE_geometric(E)
    e0 = np.sum(E * F0.sum(axis=1) * dE)
    from flux_postprocessing import compute_energy_averaged_ratios as ratios
    for dp in (dict(m_Zd_eV=3e5), dict(m_Zd_eV=2e6), dict(m_Zd_eV=2e7, g_D=1e-2)):
        E, F = run(100, **dp)
        assert np.sum(E * F.sum(axis=1) * dE) <= e0 * (1 + 1e-3), dp
        E, F4 = run(400, **dp)
        assert np.max(np.abs(ratios(E, F4) - ratios(E, F))) < 2e-3, dp


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
    # build_thermal_grid returns the raw kernel grid times the physical normalization
    exact = (dps.precompute_all_thermal_rates(E_bins, z_bins, sum_G, sum_G_alpha, 2.0e6, m_nu, 1.95e-4, eta_mass)
             * dps.absorption_normalization(sum_G))
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

    common = dict(show_progress=False, verbose=0,
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
    kwargs = dict(show_progress=False, verbose=0,
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
    Q_t, s_t, Mp_t, floor_t, _, _ = dps.texture_scales(M_rot, S)
    table = dps._build_sigma_projection_table(Q_t, Mp_t, s_t, floor_t, nodes)
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
# 8. TBB threading layer: a prange kernel, then a spawn pool, in one process
# =============================================================================

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
    cfg = {{"grid": {{"n_E_bins": 10, "n_z_bins": 10, "n_sun_paths": 1, "n_earth_chords": 1,
                      "N_steps_earth": 2}},
            "execution": {{"n_cores": 1, "verbose": 0}}}}
    E, F = dps.run_solver(cfg)
    return float(F.sum())

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    import numba
    import dark_portal_solver as dps
    assert dps._TBB_LAYER, "tbb layer not active in subprocess"

    cfg = {{"grid": {{"n_E_bins": 40, "n_z_bins": 20, "n_sun_paths": 2, "n_earth_chords": 2,
                      "N_steps_earth": 20}},
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
    cfg["grid"].update(dict(n_E_bins=10, n_z_bins=10, n_sun_paths=1, n_earth_chords=1,
                            N_steps_earth=3))
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
    """In 'sum' mode with nubar_to_nu_ratio = 0 only neutrinos are injected, exactly as in
    the explicit neutrino-only mode: the two must agree bit for bit (the antineutrinos at
    the detector are then only the secondaries of absorptions)."""
    import copy
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["grid"].update(dict(n_E_bins=10, n_z_bins=10, n_sun_paths=1, n_earth_chords=1,
                            N_steps_earth=3))
    cfg["execution"].update(dict(n_cores=1, verbose=0))

    cfg_sum0 = copy.deepcopy(cfg)
    cfg_sum0["source"]["is_antineutrino"] = "sum"
    cfg_sum0["source"]["nubar_to_nu_ratio"] = 0.0
    cfg_nu = copy.deepcopy(cfg)
    cfg_nu["source"]["is_antineutrino"] = "false"

    _, F_sum0 = dps.run_solver(cfg_sum0)
    _, F_nu = dps.run_solver(cfg_nu)
    np.testing.assert_array_equal(F_sum0, F_nu)


# =============================================================================
# Extreme potentials: exact unitarity at every potential + loud NaN contract
# =============================================================================

def test_propagator_3x3_unitary_at_every_potential():
    """U = exp(-i H t) must be unitary to rounding for every potential the transit can
    meet, from vacuum-dominated to the 1/m_Zd^2-enhanced Earth potential at m_Zd ~ 1e-6 eV
    (~1e17 eV, spread*t ~ 1e28 rad), at Earth- and Milky-Way-scale steps, for two- and
    three-flavor and single-flavor sterile textures. History: the polynomial form (v3,
    until 2026-10-12) was non-unitary by ~eps * spread * min(t, 2/gap) -- 5e-3 per Earth
    step at 70 eV (the m_Zd = 40 eV benchmark), and ~5e11 (overflowing rho to NaN in 500
    steps) at m_Zd = 1e-6 eV before a guard for that extreme; its complement basis divided
    by zero for u_mu = 0 or u_e = 0 above ~1e7 eV. The far eigenvector (the rank-1
    potential's direction) must map to itself up to a phase wherever the potential
    dominates."""
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_sq = np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)
    M_rot = U_pmns @ M_sq @ np.conjugate(U_pmns).T
    out = np.zeros((3, 3), dtype=np.complex128)
    worst_unit = 0.0
    worst_ray = 0.0
    for u in ([0.5, 0.5, 0.0], [0.3, 0.1, 0.2], [0.7, 0.0, 0.0], [0.0, 0.4, 0.0]):
        u = np.array(u, dtype=np.complex128)
        S = np.outer(u, np.conjugate(u))
        v_far = u / np.linalg.norm(u)
        for t in (1e9, 1.29e11, 1e12, 1e26, 6.25e32):           # Earth steps / Milky Way steps
            for logV in range(-24, 23):
                for E in (1e13, 3e16):
                    H = np.ascontiguousarray(M_rot / (2.0 * E) + 10.0 ** logV * S)
                    dps.propagator_3x3(H, t, out)
                    assert np.isfinite(out).all(), f"non-finite U at V=1e{logV}, t={t:.2e}, u={u.real}"
                    worst_unit = max(worst_unit, float(np.max(np.abs(out @ out.conj().T - np.eye(3)))))
                    if logV >= 0:
                        worst_ray = max(worst_ray, abs(1.0 - abs(np.vdot(v_far, out @ v_far))))
    assert worst_unit < 1e-14, f"non-unitary: {worst_unit:.3e}"
    assert worst_ray < 1e-10, f"far-eigenvector ray not preserved: {worst_ray:.3e}"


def test_propagator_3x3_unresolved_pair_is_not_rotated_by_rounding():
    """Under a rank-1 potential 1e18 times the vacuum term, the two states orthogonal to
    the potential's direction are split by the vacuum term alone, far below what H
    resolves in double precision (eps * potential). Their true rotation over an Earth
    step is ~1e-5 rad; propagating the rounding-level splitting instead would rotate them
    by tens of radians per step. The propagator must evolve the pair as degenerate."""
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_rot = U_pmns @ np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128) @ np.conjugate(U_pmns).T
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S = np.outer(u, np.conjugate(u))
    w1 = np.array([1.0, -1.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)   # orthogonal to u
    w2 = np.array([0.0, 0.0, 1.0], dtype=np.complex128)
    out = np.zeros((3, 3), dtype=np.complex128)
    worst = 0.0
    for logV in (2, 6, 10, 17):
        for E in (1e13, 3e16):
            H = np.ascontiguousarray(M_rot / (2.0 * E) + 10.0 ** logV * S)
            dps.propagator_3x3(H, 1.29e11, out)
            for w in (w1, w2, (w1 + 1j * w2) / np.sqrt(2.0)):
                worst = max(worst, 1.0 - abs(np.vdot(w, out @ w)))
    assert worst < 1e-9, f"pair rotated by rounding noise: 1 - |<w|U|w>| = {worst:.3e}"


def test_propagator_3x3_matches_extended_precision():
    """Against exp(-i H t) of the same floating-point H evaluated with 50 digits
    (mpmath), for random Hermitian H and production textures with |H| t up to 1e3 rad:
    the error must stay at the rounding level of the phases, ~eps |H| t, above the 1e-13
    truncation of the Taylor branch that serves spreads below 1.2e-4 / t."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_rot = U_pmns @ np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128) @ np.conjugate(U_pmns).T
    u = np.array([0.5, 0.5, 0.0], dtype=np.complex128)
    S = np.outer(u, np.conjugate(u))
    rng = np.random.default_rng(41)
    out = np.zeros((3, 3), dtype=np.complex128)
    worst = 0.0
    for k in range(120):
        if k % 2 == 0:
            A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            scale = 10.0 ** rng.uniform(-20, -12)
            H = np.ascontiguousarray((A + A.conj().T) / 2 * scale)
            t = 10.0 ** rng.uniform(-1, 3) / scale
        else:
            E = 10.0 ** rng.uniform(13, 16.5)
            H = np.ascontiguousarray(M_rot / (2.0 * E) + 10.0 ** rng.uniform(-19, -15) * S)
            t = 10.0 ** rng.uniform(9, 12)
        M = mp.matrix(3, 3)
        for i in range(3):
            M[i, i] = mp.mpf(H[i, i].real)
            for j in range(i + 1, 3):
                M[i, j] = mp.mpc(H[i, j].real, H[i, j].imag)
                M[j, i] = mp.conj(M[i, j])
        X = mp.expm(-1j * mp.mpf(t) * M)
        ref = np.array([[complex(X[i, j]) for j in range(3)] for i in range(3)])
        dps.propagator_3x3(H, t, out)
        phase = float(np.max(np.abs(np.linalg.eigvalsh(H)))) * t
        worst = max(worst, float(np.max(np.abs(out - ref))) / (1e-13 + 2e-15 * phase))
    assert worst < 1.0, f"error beyond the phases' rounding level: {worst:.2f} x tolerance"


def test_spatial_handover_conserves_every_total():
    """The Milky Way and Earth transits are unitary evolutions: every energy bin's total
    flux must leave the spatial handover unchanged, including where the potential
    dominates (m_Zd = 40 eV; until 2026-10-12 the totals came out 0.995-1.0066) and at
    m_Zd = 1e-6 eV with single-flavor mixing (previously a ZeroDivisionError)."""
    E_bins = np.geomspace(1e13, 3e16, 24)
    rng = np.random.default_rng(8)
    F_mass = rng.uniform(0.5, 2.0, size=(24, 3)) * 1e-8
    U_pmns = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_rot = U_pmns @ np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128) @ np.conjugate(U_pmns).T
    for m_Zd, g_D, eps_p, u in ((40.0, 1e-6, 0.1, (0.5, 0.5, 0.0)), (1e-6, 1e-4, 0.1, (0.5, 0.0, 0.0))):
        u_arr = np.array(u, dtype=np.complex128)
        for anti in (False, True):
            F = dps.propagate_earth_spatial(E_bins, F_mass, M_rot, u_arr, m_Zd, g_D, eps_p,
                                            1.676e-4, True, np.ones(3), 1.5, False, 0.0, anti, 2, 3,
                                            show_progress=False, verbose=0, n_cores=1,
                                            N_steps_earth=60, earth_step_scheme="avg",
                                            earth_zenith_average="phase")
            ratio = F.sum(axis=1) / F_mass.sum(axis=1)
            assert np.all(np.abs(ratio - 1.0) < 1e-12), (m_Zd, anti, float(np.max(np.abs(ratio - 1.0))))


def test_numba_cache_is_keyed_by_the_sources():
    """numba validates a cached function against its own file only, so a kernel that
    calls a jitted function from another module would keep the callee's old machine code
    after an edit (it happened: dp_spatial's kernels kept the replaced propagator). The
    cache directory must be keyed by a digest of every source file."""
    import dp_core
    import numba
    if os.environ.get("NUMBA_CACHE_DIR", "").rstrip("/").endswith(dp_core._source_digest()) is False:
        pytest.skip("NUMBA_CACHE_DIR set by the user")
    assert numba.config.CACHE_DIR == os.environ["NUMBA_CACHE_DIR"]
    assert dps.propagator_3x3._cache._cache_path.startswith(os.environ["NUMBA_CACHE_DIR"])


def test_extreme_mediator_mass_flux_is_finite(tmp_path):
    """End-to-end contract for the formerly all-NaN domain: m_Zd_eV = 1e-6 (g_D = 1e-4,
    epsilon_prime = 0.1) with the full 500 Earth steps (the NaN trigger -- repeated
    application of the non-unitary propagator needed a few hundred steps to overflow)
    must now return an all-finite, non-negative-total flux."""
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["dark_portal"].update(dict(m_Zd_eV=1e-6, g_D=1e-4, epsilon_prime=0.1))
    cfg["grid"].update(dict(n_E_bins=20, n_z_bins=20, n_sun_paths=2, n_earth_chords=2,
                            N_steps_earth=500))
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
            True, 1.5, False, 0.0, (1.0, 0.0), 2.5, 0.0, 1.0, np.ones(3),
            1, 1, show_progress=False, verbose=0, n_cores=1,
            precomputed_Gamma_grid_all=bad_grid, N_steps_earth=3)


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

    ref = (dps.precompute_all_thermal_rates(E_bins, z_bins, sum_G, sum_G_alpha, 2e6,
                                            m_nu, 1.95e-4, eta_mass) * dps.absorption_normalization(sum_G))
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
    exact point (no rebuild) and must return the identical grid -- a minimizer re-evaluates
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
    """Minimizer-step-sized coupling moves (well below the historical 0.3-dex floor)
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
    contact, heavy = dps.earth_yukawa_table(None), dps.earth_yukawa_table(2e6)
    for N in (37, 250):
        nbar = dps._avg_earth_density_seq(baseline, costh, N, depth, n_amb, cross,
                                          dps._GL10_X, dps._GL10_W, contact)
        # at MeV masses the Earth's Yukawa integral is its contact limit, bit for bit
        assert np.array_equal(nbar, dps._avg_earth_density_seq(baseline, costh, N, depth, n_amb,
                                                               cross, dps._GL10_X, dps._GL10_W, heavy))
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
    cfg["grid"].update(dict(n_E_bins=16, n_z_bins=16, n_sun_paths=1, n_earth_chords=2,
                            N_steps_earth=40))
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
    cfg["grid"].update(dict(n_E_bins=24, n_z_bins=16, n_sun_paths=1, n_earth_chords=3))
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
# GPU spatial path: propagator port, mathematics validated with numpy backend
# =============================================================================

def test_gpu_batched_propagator_matches_cpu_numpy_backend():
    """_batched_propagator_gpu takes its array backend as a parameter and uses only
    numpy-compatible operations, so the port's MATHEMATICS is validated here on the
    CPU by running it with numpy as the backend against propagator_3x3, one batch
    mixing every branch regime (masks must coexist within a single call):
    generic well-separated, rank-1-dominated, constructed near-degenerate, Milky-Way-scale
    phases, and pairs below double-precision resolution (including single-flavor
    textures). Only execution under cupy itself remains for the skipped-on-CPU device
    test."""
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
    for _ in range(40):                                   # MW-scale phases
        E = 10.0 ** rng.uniform(13, 16)
        Hs.append(M_rot / (2.0 * E))
        ts.append(10.0 ** rng.uniform(25, 27))
        regimes.append("large_phase")
    for logV in range(6, 23, 2):                          # unresolved pair
        Hs.append(M_rot / (2.0 * 1e13) + (10.0 ** logV) * S)
        ts.append(1.29e11)
        regimes.append("noise")
    u1 = np.array([0.7, 0.0, 0.0], dtype=np.complex128)   # single-flavor texture
    for logV in range(-2, 23, 4):
        Hs.append(M_rot / (2.0 * 1e14) + (10.0 ** logV) * np.outer(u1, u1))
        ts.append(1.29e11)
        regimes.append("noise1")
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
            # potential-dominated: even |U| depends on the (rounding-level) relative
            # phase between the far and pair eigenvalues, so scalar and batched
            # legitimately disagree there; the meaningful invariants are unitarity
            # (checked above) and preservation of the far eigenvector's ray
            vf = v_far if regimes[k] == "noise" else np.array([1.0, 0.0, 0.0], dtype=np.complex128)
            worst_ray = max(worst_ray, abs(1.0 - abs(np.vdot(vf, Ub @ vf))))
    assert worst_direct < 1e-9, f"batched vs scalar propagator: {worst_direct:.3e}"
    assert worst_unit < 1e-14, f"batched propagator non-unitary: {worst_unit:.3e}"
    assert worst_abs < 1e-4, f"batched |U| structure off at large phases: {worst_abs:.3e}"
    assert worst_ray < 1e-10, f"batched far-eigenvector ray not preserved: {worst_ray:.3e}"


# =============================================================================
# Scan-infrastructure regression tests (checkpointing + deterministic retry jitter)
# =============================================================================

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


def _analysis_config_files(cfg_dir):
    """Every analysis configuration under cfg_dir, the campaign's subdirectories included (not their plane lists)."""
    out = []
    for root, _, files in os.walk(cfg_dir):
        out += [os.path.join(root, f) for f in files if f.endswith(".json") and f != "campaign.json"]
    return sorted(out)


def test_default_relic_neutrino_temperature_is_physical():
    """The relic-neutrino temperature must be (4/11)^(1/3) T_CMB, converted to eV.

    Until 2026-10-12 the default was 1.95e-4 eV: "1.95 K" entered as eV (2.26 K, 16% too
    hot). The thermal kernel's relic density follows from the Fermi-Dirac integral, so the
    absorption rates scale ~T^3 and were ~57% too high. Pins the default AND the shipped
    analysis configs, which carry their own copy of the value."""
    T_expected = (4.0 / 11.0) ** (1.0 / 3.0) * 2.7255 * 8.617333262e-5   # eV
    assert abs(dps.DEFAULT_CONFIG["cosmology"]["T_nu0_eV"] / T_expected - 1.0) < 1e-5

    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs")
    for name in _analysis_config_files(cfg_dir):
        with open(name) as f:
            T_cfg = json.load(f)["solver"]["cosmology"]["T_nu0_eV"]
        assert abs(T_cfg / T_expected - 1.0) < 1e-5, f"{name} carries T_nu0_eV = {T_cfg}"


def test_default_cosmological_neutron_density_is_physical():
    """The cosmological dark-portal potential is sourced by neutrons only, which sit almost
    entirely in helium-4: n_n0 = n_b * Y_p / 2. Until 2026-10-12 the default was the total
    baryon density (0.25e-6 cm^-3), 8.1x too large. Pins the default and the analysis
    configurations, each of which carries its own copy."""
    n_b, Y_p = 2.515e-7, 0.2458             # PDG 2024 (CMB); LBT Y_p project (2026)
    expected = n_b * Y_p / 2.0
    assert abs(dps.DEFAULT_CONFIG["cosmology"]["n_n0_cm3"] / expected - 1.0) < 1e-3

    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs")
    for name in _analysis_config_files(cfg_dir):
        with open(name) as f:
            n_cfg = json.load(f)["solver"]["cosmology"]["n_n0_cm3"]
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
    # cold HI + H2 discs (McMillan 2017) at the mid-plane: sech^2(0) = 1
    rho_cold = (53.1 / (4 * 85.0) * np.exp(-4.0 / R0 - R0 / 7.0)
                + 2180.0 / (4 * 45.0) * np.exp(-12.0 / R0 - R0 / 1.5))
    expected = ((rho_bulge + rho_thin + rho_thick) * 40.464 * 0.20 + rho_cold * 40.464 * 0.142
                + n_e_gas * 0.15)
    got = dp_spatial.calc_MW_neutron_density(0.0, 0.0, 0.0)
    assert abs(got / expected - 1.0) < 1e-6, (got, expected)


def test_cold_gas_discs_reproduce_mcmillan_surface_densities():
    """Integrated over height, each McMillan (2017) gas disc must return its surface density
    Sigma_0 exp(-R_m/R - R/R_d): about 10 (HI) and 2 (H2) Msun/pc^2 at the Sun. The sech^2 form
    must stay finite far from the plane, and the density must vanish on the axis."""
    import dp_spatial
    z_kpc = np.linspace(-3.0, 3.0, 600001)
    for disc, sigma_sun in ((dp_spatial._HI_DISC, 10.0), (dp_spatial._H2_DISC, 2.0)):
        Sigma_0, R_d, R_m, z_d = disc
        rho = np.array([dp_spatial._gas_disc_density(8.33, z, Sigma_0, R_d, R_m, z_d) for z in z_kpc])
        sigma = _trapz(rho, z_kpc * 1e3)                      # Msun/pc^2
        assert abs(sigma / (Sigma_0 * np.exp(-R_m / 8.33 - 8.33 / R_d)) - 1.0) < 1e-6
        assert abs(sigma / sigma_sun - 1.0) < 0.07, (disc, sigma)
        assert dp_spatial._gas_disc_density(8.33, 100.0, Sigma_0, R_d, R_m, z_d) == 0.0
        assert dp_spatial._gas_disc_density(0.0, 0.0, Sigma_0, R_d, R_m, z_d) == 0.0


def test_point_source_neutron_numbers():
    """The Sun, Moon and Earth enter as point sources of the potential. Their neutron numbers
    must match their compositions: the Earth's is PREM integrated with the same per-layer
    fractions the chords use; the Sun's is 0.161 neutrons per nucleon (present bulk Y = 0.308,
    Z = 0.0142), the Moon's 0.505 (rock with a small iron core). Until 2026-10-12 the Sun and the
    Moon both used 0.248 per nucleon."""
    from scipy.integrate import quad
    import dp_core
    import dp_spatial
    m_u = 1.66054e-24
    edges = [0.0, 1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0, 6346.6, 6356.0, 6368.0, 6371.0]
    N_earth = sum(quad(lambda r: 4 * np.pi * r * r * dp_spatial.calc_earth_density(r)
                       * dp_spatial.earth_neutrons_per_gram(r), a, b, epsrel=1e-10)[0]
                  for a, b in zip(edges[:-1], edges[1:])) * 1e15
    assert abs(dp_core.N_n_earth / N_earth - 1.0) < 1e-3, (dp_core.N_n_earth, N_earth)
    assert abs(dp_core.N_n_sun * m_u / 1.98847e33 - 0.161) < 1e-3
    assert abs(dp_core.N_n_moon * m_u / 7.342e25 - 0.505) < 1e-3


def test_earth_neutrons_per_gram_by_layer():
    """Along Earth chords the neutron density is rho * (1 - Y_e) * N_A, with Y_e = 0.4656 in the
    core, 0.4957 in the mantle and crust, and water (8/18.015 per molecular mass) in PREM's top
    layer; until 2026-10-12 every layer used 0.5. The regional boundaries must coincide with
    PREM layer boundaries, or the 'avg' Earth scheme's layer-split integration would straddle a
    discontinuity and lose its exactness."""
    import dp_spatial
    NA = 6.02214076e23
    cases = {1000.0: 1 - 0.4656, 3000.0: 1 - 0.4656, 3480.0: 1 - 0.4656,
             3480.1: 1 - 0.4957, 5000.0: 1 - 0.4957, 6360.0: 1 - 0.4957, 6368.0: 1 - 0.4957,
             6370.0: 8.0 / 18.015}
    for r, f_n in cases.items():
        assert abs(dp_spatial.earth_neutrons_per_gram(r) / (f_n * NA) - 1.0) < 1e-12, r
    for boundary in (dp_spatial._EARTH_R_CORE_KM, dp_spatial._EARTH_R_OCEAN_FLOOR_KM):
        assert boundary in dp_spatial._PREM_BOUNDARIES_KM


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
    integral_ratio = _trapz(G[T_old], E_bins) / _trapz(G[T_new], E_bins)
    assert abs(integral_ratio / r**3 - 1.0) < 2e-3, integral_ratio
    assert abs((G[T_old].max() / G[T_new].max()) / r**2 - 1.0) < 1e-2


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))


# =============================================================================
# Earth chord geometry (2026-10-12)
# =============================================================================

def test_exact_chord_baseline_starts_on_the_surface():
    """get_baseline_earth must put every chord's entry point on the surface. Until 2026-10-12 it
    dropped the d^2 terms of the root, so chords started up to 8 m above the ground; where that
    vacuum lead-in fell inside _earth_layer_crossings' end margin, the 'avg' integration silently
    ran the top layer out to l = 0. Down-going chords (costh > 0) use the cancellation-free form."""
    depth = 1.5
    f2 = 6371.0 - depth
    for costh in np.linspace(-1.0, 1.0, 201):
        B = dps.get_baseline_earth(depth, costh)
        r_entry = np.sqrt(B**2 + f2**2 + 2.0 * B * f2 * costh)
        assert abs(r_entry - 6371.0) < 1e-11, (costh, r_entry)
    assert abs(dps.get_baseline_earth(depth, 1.0) - depth) < 1e-12
    assert abs(dps.get_baseline_earth(depth, -1.0) - (2 * 6371.0 - depth)) < 1e-9
    # the horizontal chord no longer meets the surface a second time inside the chord
    assert len(dps._earth_layer_crossings(dps.get_baseline_earth(depth, 0.0), 0.0, depth)) == 0


# =============================================================================
# The Earth's own potential: the spherical Yukawa integral (2026-10-12)
# =============================================================================

def _prem_neutrons_cm3(r):
    """PREM neutron density (cm^-3) the contact way: calc_earth_density x neutrons per gram."""
    import dp_spatial
    return dp_spatial.calc_earth_density(r) * dp_spatial.earth_neutrons_per_gram(r)


def test_prem_coefficient_table_matches_calc_earth_density():
    """The Yukawa integral reads PREM from _PREM_RHO_COEFFS / _PREM_F_COEFFS, the contact form from
    calc_earth_density: the two copies must be one profile, boundary convention included, or the
    heavy-mass limit would not reproduce the contact form."""
    import dp_spatial
    edges = dps._PREM_EDGES_KM
    assert np.array_equal(dps._PREM_BOUNDARIES_KM, edges[1:])
    rng = np.random.default_rng(7)
    radii = np.concatenate([rng.uniform(0.0, 6371.0, 4000), edges[1:], edges[1:-1] + 1e-7, [1e-3]])
    for r in radii:
        s = dp_spatial._yk_shell(r, edges)
        c = dps._PREM_RHO_COEFFS[s]
        x = r / 6371.0
        rho = c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3
        assert abs(rho - dps.calc_earth_density(r)) <= 1e-13 * rho, r
        f = sum(dps._PREM_F_COEFFS[s, k] * r**k for k in range(5))
        assert abs(f / r / _prem_neutrons_cm3(r) - 1.0) < 1e-13, r


def test_yukawa_uniform_ball_closed_form():
    """A uniform ball has a closed form at any mediator mass (NuOscProbExact notebook 20),
    n_eff = n [1 - (R + 1/m) e^{-mR} sinh(mr)/r] inside and n e^{-mr} [R cosh(mR) - sinh(mR)/m]/r
    outside. The 'direct' evaluation must reproduce it to fp RELATIVE accuracy (it has no
    cancellation, so the long-range limit n (mR)^2/2 is resolved too); the 'heavy' one to fp
    accuracy in units of n, from m R = 1 up to the MeV-like 1e8."""
    import mpmath
    import dp_spatial
    mpmath.mp.dps = 40
    R, n = 6371.0, 1e24
    F = np.array([[0.0, n, 0.0, 0.0, 0.0]])                    # f(r) = r n
    radii = np.concatenate([np.linspace(0.0, R, 41), [1e-6, R * (1 - 1e-9)],
                            R * np.array([1.0 + 1e-9, 1.001, 1.1, 1.5])])

    def exact(m, r):
        m, r, Rm = mpmath.mpf(m), mpmath.mpf(r), mpmath.mpf(R)
        if r == 0:
            return n * (1 - (Rm + 1 / m) * mpmath.exp(-m * Rm) * m)
        if r <= Rm:
            return n * (1 - (Rm + 1 / m) * mpmath.exp(-m * Rm) * mpmath.sinh(m * r) / r)
        return n * mpmath.exp(-m * r) * (Rm * mpmath.cosh(m * Rm) - mpmath.sinh(m * Rm) / m) / r

    for mR in (1e-7, 1e-3, 0.3, 1.0, 10.0, 49.0, 51.0, 300.0, 1e4, 1e8):
        m = mR / R
        modes = ([dp_spatial._YUKAWA_DIRECT] if mR <= 300.0 else []) + \
                ([dp_spatial._YUKAWA_HEAVY] if mR >= 1.0 else [])
        for mode in modes:
            yk = dps.yukawa_profile_table(np.array([0.0, R]), F, m, mode=mode)
            for r in radii:
                got, ref = dps.yukawa_profile_density(r, yk), float(exact(m, r))
                if mode == dp_spatial._YUKAWA_DIRECT:
                    assert abs(got - ref) <= 1e-12 * abs(ref), (mR, mode, r, got, ref)
                else:
                    assert abs(got - ref) <= 1e-13 * n + 1e-12 * abs(ref), (mR, mode, r, got, ref)
    # the default choice switches at m R = 50
    assert int(dps.yukawa_profile_table(np.array([0.0, R]), F, 49.0 / R)[0][0]) == dp_spatial._YUKAWA_DIRECT
    assert int(dps.yukawa_profile_table(np.array([0.0, R]), F, 51.0 / R)[0][0]) == dp_spatial._YUKAWA_HEAVY


def test_earth_yukawa_heavy_limit_is_contact_form_bit_for_bit():
    """At the masses of every current analysis point the mediator range is far below any PREM
    length, so the Yukawa integral must return the contact density EXACTLY, not to a tolerance:
    Earth phases are ~1e5-1e8 rad there, and the phase-noise regime (tens of eV) redraws its
    realization under any ulp change. Checked pointwise, along chords, and for the step averages
    of every chord of the 12-, 13- and 50-chord grids at the default 25 steps -- each against the
    contact table through the same compiled code (a separately compiled product can differ by an
    ulp through fused multiply-adds, which says nothing about the Yukawa term)."""
    import dp_spatial
    rng = np.random.default_rng(11)
    radii = rng.uniform(0.0, 6371.0, 3000)
    radii = radii[np.min(np.abs(radii[:, None] - dps._PREM_EDGES_KM[None, :]), axis=1) > 1e-6]
    contact = dps.earth_yukawa_table(None)
    for m_eV in (3e5, 2e6, 2e7, 40.0):
        yk = dps.earth_yukawa_table(m_eV)
        assert int(yk[0][0]) == dp_spatial._YUKAWA_HEAVY
        for r in radii:
            assert dps.earth_neutron_density_cm3(r, yk) == dps.earth_neutron_density_cm3(r, contact)
            assert abs(dps.earth_neutron_density_cm3(r, yk) / _prem_neutrons_cm3(r) - 1) < 1e-15
        for n_ch in (12, 13, 50):
            for costh in np.linspace(-1.0, 0.0, n_ch):
                B = dps.get_baseline_earth(1.5, costh)
                cross = dps._earth_layer_crossings(B, costh, 1.5)
                for l in rng.uniform(0.0, B, 10):
                    assert (dps.earth_total_density(l, B, costh, 1.5, 1.36e-14, yk)
                            == dps.earth_total_density(l, B, costh, 1.5, 1.36e-14, contact))
                args = (B, costh, 25, 1.5, 1.36e-14, cross, dps._GL10_X, dps._GL10_W)
                assert np.array_equal(dps._avg_earth_density_seq(*args, yk),
                                      dps._avg_earth_density_seq(*args, contact)), (m_eV, n_ch, costh)


def test_earth_yukawa_matches_bruteforce_shell_kernel():
    """n_eff from the tables against direct quadrature of the shell-averaged Yukawa kernel,
    n_eff(r) = (m/2r) int_0^R r' n(r') [e^{-m|r-r'|} - e^{-m(r+r')}] dr', with n taken from
    calc_earth_density (independent of the coefficient table), for ranges from beyond R_E to
    200 m -- both evaluations -- at radii on both sides of PREM boundaries and at the surface."""
    edges = [e for e in dps._PREM_EDGES_KM if 0.0 < e < 6371.0]
    for m_eV in (3e-14, 3e-13, 1e-11, 1e-9):
        yk = dps.earth_yukawa_table(m_eV)
        m = yk[0][1]
        for r in (0.5, 700.0, 1221.4, 1221.6, 3000.0, 3479.9, 3480.1, 5000.0,
                  6360.0, 6367.99, 6368.01, 6370.0, 6371.0):
            near = [x for x in r + np.array([0, -1, 1, -3, 3, -10, 10, -30, 30]) / m if 0 < x < 6371.0]
            pts = [0.0] + sorted(set(edges) | set(near)) + [6371.0]
            kern = lambda x: x * _prem_neutrons_cm3(x) * (np.exp(-m * abs(r - x)) - np.exp(-m * (r + x)))
            ref = sum(integrate.quad(kern, a, b, epsabs=0, epsrel=1e-12, limit=200)[0]
                      for a, b in zip(pts[:-1], pts[1:]) if b > a) * m / (2.0 * r)
            got = dps.earth_neutron_density_cm3(r, yk)
            assert abs(got / ref - 1.0) < 1e-9, (m_eV, r, got, ref)


def test_earth_yukawa_long_range_limit_and_exterior():
    """For a range far beyond the Earth (m R = 1e-12; the corrections are O(m R)) the potential
    takes the enclosed-charge form n_eff/m^2 = N(<r)/(4 pi r) + int_r^R r' n dr'. Outside, the
    Earth is a point source of
    N_eff = 4 pi int r'^2 n shc(m r') dr' neutrons, which at such a range is the Earth's neutron
    number -- so the Earth chords now meet the Milky Way phase's Earth point source (N_n_earth)."""
    import dp_core
    import dp_spatial
    m_km = 1e-12 / 6371.0
    yk = dps.earth_yukawa_table(m_km / dp_spatial._KM_IN_EV_INV)
    assert int(yk[0][0]) == dp_spatial._YUKAWA_DIRECT
    edges = [e for e in dps._PREM_EDGES_KM if 0.0 < e < 6371.0]

    def piecewise(fun, a, b):
        p = [a] + [e for e in edges if a < e < b] + [b]
        return sum(integrate.quad(fun, lo, hi, epsabs=0, epsrel=1e-13)[0] for lo, hi in zip(p[:-1], p[1:]))

    for r in (1.0, 1000.0, 1221.5, 3000.0, 3480.0, 5000.0, 6000.0, 6370.0, 6371.0):
        ref = m_km**2 * (piecewise(lambda x: x * x * _prem_neutrons_cm3(x), 0.0, r) / r
                         + piecewise(lambda x: x * _prem_neutrons_cm3(x), r, 6371.0))
        assert abs(dps.earth_neutron_density_cm3(r, yk) / ref - 1.0) < 1e-9, r
    for r in (6371.0 * (1 + 1e-9), 2 * 6371.0):
        N_eff = dps.earth_neutron_density_cm3(r, yk) * 4 * np.pi * r * np.exp(m_km * r) / m_km**2 * 1e15
        assert abs(N_eff / dp_core.N_n_earth - 1.0) < 1e-3, (r, N_eff)


def test_earth_yukawa_evaluations_agree_across_the_switch():
    """'heavy' (integration by parts) and 'direct' (tabulated P, Q) are both exact for PREM's
    polynomials, so forced to the same mass they must agree everywhere -- inside, at every
    boundary, outside -- across the m R = 50 switch."""
    import dp_spatial
    radii = np.concatenate([np.linspace(0.0, 6371.0, 997), dps._PREM_EDGES_KM, [6371.5, 7000.0]])
    for mR in (10.0, 50.0, 300.0):
        m_eV = mR / 6371.0 / dp_spatial._KM_IN_EV_INV
        hv = dps.earth_yukawa_table(m_eV, mode=dp_spatial._YUKAWA_HEAVY)
        di = dps.earth_yukawa_table(m_eV, mode=dp_spatial._YUKAWA_DIRECT)
        for r in radii:
            a, b = dps.earth_neutron_density_cm3(r, hv), dps.earth_neutron_density_cm3(r, di)
            assert abs(a - b) <= 1e-12 * abs(b) + 1e-14 * 4.2e24, (mR, r, a, b)


def test_avg_scheme_step_count_independent_with_yukawa():
    """With a mediator range anywhere from 200 m to beyond R_E the boundary layers along the
    chords (crossings, the surface at entry and above the detector) must be resolved, so the
    step-summed path integral stays independent of the step count -- the 'avg' scheme's
    certificate -- and equals an independent dense integral of n_eff (composite 20-point
    Gauss-Legendre graded toward every layer at ratio 1.1)."""
    depth = 1.5
    f2 = 6371.0 - depth
    x20, w20 = np.polynomial.legendre.leggauss(20)
    import dp_spatial
    for m_eV in (1e-9, 1e-11, 1e-13, 3e-14, 1e-16):
        yk = dps.earth_yukawa_table(m_eV)
        for costh in (-1.0, -0.83, -0.3, -0.02, 0.0):
            B = dps.get_baseline_earth(depth, costh)
            cross = dps._earth_layer_crossings(B, costh, depth)
            tot = {}
            for N in (8, 25, 1000):
                nbar = dps._avg_earth_density_seq(B, costh, N, depth, 0.0, cross,
                                                  dps._GL10_X, dps._GL10_W, yk)
                tot[N] = np.sum(nbar) * (B / N)
            assert abs(tot[8] / tot[1000] - 1) < 1e-12, (m_eV, costh, tot)
            assert abs(tot[25] / tot[1000] - 1) < 1e-12, (m_eV, costh, tot)
            if costh in (-0.83, 0.0):
                fl, fw = dp_spatial._earth_chord_features(B, costh, depth, cross, yk)
                E = {0.0, B} | set(cross)
                for lf, w in zip(fl, fw):
                    x = w / 64
                    while x < B:
                        E |= {e for e in (lf - x, lf + x) if 0.0 < e < B}
                        x *= 1.1
                E = np.array(sorted(E))
                ref = 0.0
                for lo, hi in zip(E[:-1], E[1:]):
                    l = 0.5 * (lo + hi) + 0.5 * (hi - lo) * x20
                    r = np.sqrt((B - l)**2 + f2**2 + 2 * (B - l) * f2 * costh)
                    ref += 0.5 * (hi - lo) * np.sum(w20 * np.array(
                        [dps.earth_neutron_density_cm3(x, yk) for x in r]))
                assert abs(tot[25] / (ref * dps.cm_inv3_to_eV3) - 1) < 1e-11, (m_eV, costh)


def test_earth_range_mediator_end_to_end(tmp_path):
    """A mediator whose range is the Earth's radius (direct mode), at couplings that keep the
    Earth transit coherent (phases of tens of rad): the solver must run, return finite
    non-negative fluxes, and keep the 'avg' step-count certificate end to end."""
    import copy
    cfg = dps.create_default_config(str(tmp_path / "cfg.json"))
    cfg["grid"].update(dict(n_E_bins=24, n_z_bins=16, n_sun_paths=1, n_earth_chords=3))
    cfg["execution"].update(dict(n_cores=1, verbose=0))
    cfg["dark_portal"].update(dict(m_Zd_eV=3.1e-14, g_D=1e-24, epsilon_prime=1e-25))
    flux = {}
    for n in (25, 200):
        c = copy.deepcopy(cfg)
        c["grid"]["N_steps_earth"] = n
        _, flux[n] = dps.run_solver(c)
    assert np.isfinite(flux[25]).all() and (flux[25] >= 0).all()
    assert np.max(np.abs(flux[200] - flux[25])) <= 1e-8 * np.max(np.abs(flux[200]))


# =============================================================================
# Zenith average of the Earth chords (2026-10-12)
# =============================================================================

def _synthetic_chords(nodes, alpha, rho_in, S):
    """Chord density matrices exp(-i Psi S) rho_in exp(+i Psi S) with Psi = alpha * column,
    the potential-dominated evolution the 'phase' rule assumes, plus the projectors of S."""
    import dp_spatial
    P, lam = dp_spatial._lri_texture_projectors(S)
    col = dp_spatial._earth_columns(nodes, 1.5, 0.0, dps.earth_yukawa_table(None))

    def evolve(psi):
        U = sum(np.exp(-1j * l * psi) * p for l, p in zip(lam, P))
        return U @ rho_in @ U.conj().T
    return np.array([evolve(alpha * c) for c in col])[:, None], evolve, P


def test_zenith_weights_reduce_to_trapezoid_and_stay_hermitian():
    """With no phase the 'phase' weights are the trapezoid rule over the chord nodes, and the
    averaged density matrix is Hermitian with unit trace for any texture, including a
    full-rank complex one with three distinct eigenvalues."""
    import dp_spatial
    nodes = np.linspace(-1.0, 0.0, 12)
    c_fine, idx = dp_spatial._earth_zenith_fine_grid(nodes, 1.5, 512)
    w = dp_spatial._zenith_filon_weights(c_fine, idx, np.zeros_like(c_fine))
    h = nodes[1] - nodes[0]
    trap = np.full(12, h)
    trap[[0, -1]] = h / 2
    np.testing.assert_allclose(w.real, trap, rtol=0, atol=1e-14)
    assert np.max(np.abs(w.imag)) < 1e-15
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    S = A + A.conj().T
    B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho_in = B @ B.conj().T
    rho_in /= np.trace(rho_in).real
    rho_k, _, P = _synthetic_chords(nodes, 3e-8, rho_in, S)
    assert len(P) == 3
    avg = dps.average_earth_chords(rho_k, nodes, 3e-8 / dp_spatial._KM_IN_EV_INV, S, 1.5, 0.0,
                                   1.0, dps.earth_yukawa_table(None), "phase")[0]
    assert np.max(np.abs(avg - avg.conj().T)) < 1e-14
    assert abs(np.trace(avg).real - 1.0) < 1e-13


def test_zenith_phase_average_matches_continuum():
    """The 'phase' average of 12 or 13 chords must equal the exact integral over costh of
    the potential-dominated evolution -- a dense trapezoid over 200,001 chords up to Earth
    phases of 1e3 rad, the decoherent block-diagonal limit sum_g P_g rho P_g at 1e8 rad --
    while the historical equal-weight mean misses by 1e-3 to 1e-1 (the chord-sampling floor)."""
    import dp_spatial
    S = np.outer([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]).astype(complex)
    rng = np.random.default_rng(3)
    B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho_in = B @ B.conj().T
    rho_in /= np.trace(rho_in).real
    c_dense = np.linspace(-1.0, 0.0, 200_001)
    col_dense = dp_spatial._earth_columns(c_dense, 1.5, 0.0, dps.earth_yukawa_table(None))
    for K in (12, 13):
        nodes = np.linspace(-1.0, 0.0, K)
        for psi_max in (0.1, 3.0, 52.0, 1e3, 1e8):
            alpha = psi_max / col_dense[0]
            rho_k, evolve, P = _synthetic_chords(nodes, alpha, rho_in, S)
            got = dps.average_earth_chords(rho_k, nodes, alpha / dp_spatial._KM_IN_EV_INV, S, 1.5,
                                           0.0, 1.0, dps.earth_yukawa_table(None), "phase")[0]
            if psi_max <= 1e3:
                ref = _trapz(np.array([evolve(alpha * c) for c in col_dense]), c_dense, axis=0)
            else:
                ref = sum(p @ rho_in @ p for p in P)
            assert np.max(np.abs(got - ref)) < 1e-5, (K, psi_max, np.max(np.abs(got - ref)))
            if psi_max >= 3.0:
                assert np.max(np.abs(rho_k[:, 0].mean(axis=0) - ref)) > 1e-3


def test_full_sky_average_matches_continuum_and_dilutes_the_earth():
    """With down-going directions (decided 2026-10-13) the 'phase' average runs over the whole
    sky, cos(zenith) from -1 to 1, where down-going chords cross only the ice above the
    detector (1.5 to ~140 km). It must equal the exact integral over the sky of the
    potential-dominated evolution (dense trapezoid, 400,001 directions); a potential that
    leaves the ice above unchanged must show half the up-going effect, and a very strong
    one the fully mixed limit."""
    import dp_spatial
    S = np.outer([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]).astype(complex)
    rng = np.random.default_rng(4)
    B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho_in = B @ B.conj().T
    rho_in /= np.trace(rho_in).real
    up = np.linspace(-1.0, 0.0, 12)
    full = np.concatenate([up, -up[-2::-1]])
    c_dense = np.linspace(-1.0, 1.0, 400_001)
    col_dense = dp_spatial._earth_columns(c_dense, 1.5, 0.0, dps.earth_yukawa_table(None))
    assert abs(dps.get_baseline_earth(1.5, 1.0) - 1.5) < 1e-12
    for psi_max in (0.1, 3.0, 52.0, 1e3):
        alpha = psi_max / col_dense[0]
        rho_k, evolve, P = _synthetic_chords(full, alpha, rho_in, S)
        got = dps.average_earth_chords(rho_k, full, alpha / dp_spatial._KM_IN_EV_INV, S, 1.5,
                                       0.0, 1.0, dps.earth_yukawa_table(None), "phase")[0]
        ref = _trapz(np.array([evolve(alpha * c) for c in col_dense]), c_dense, axis=0) / 2.0
        assert np.max(np.abs(got - ref)) < 1e-5, (psi_max, np.max(np.abs(got - ref)))
    # How much the sky dilutes the Earth's effect depends on the potential: while the ice
    # above adds little phase (core phase <~ 50 rad: <= 0.06 rad at the horizon) the down-going
    # half is untouched and the flavor shift halves; for strong potentials both halves mix
    # fully and the average reaches the block-diagonal limit sum_g P_g rho P_g.
    def averaged(nodes, psi_core):
        alpha = psi_core / col_dense[0]
        rho_k, _, P = _synthetic_chords(nodes, alpha, rho_in, S)
        return dps.average_earth_chords(rho_k, nodes, alpha / dp_spatial._KM_IN_EV_INV, S, 1.5, 0.0,
                                        1.0, dps.earth_yukawa_table(None), "phase")[0], P
    for psi_core in (3.0, 52.0):
        shift_up = np.real(np.diag(averaged(up, psi_core)[0] - rho_in))
        shift_full = np.real(np.diag(averaged(full, psi_core)[0] - rho_in))
        np.testing.assert_allclose(shift_full, 0.5 * shift_up, rtol=0.02, atol=1e-7)
    got, P = averaged(full, 1e9)
    assert np.max(np.abs(got - sum(p @ rho_in @ p for p in P))) < 1e-4


def test_zenith_average_options_and_numba_crossings():
    """'phase' needs the phase-exact 'avg' steps (direct callers asking otherwise get an
    error; run_solver maps 'left' to the historical 'plain'), unknown options are rejected,
    and the numba crossing finder used for the fine grid agrees with the chords' own."""
    import dp_spatial
    for costh in np.linspace(-1.0, 1.0, 81):              # up- and down-going
        B = dps.get_baseline_earth(1.5, costh)
        np.testing.assert_allclose(dp_spatial._earth_layer_crossings_nb(B, costh, 1.5),
                                   dps._earth_layer_crossings(B, costh, 1.5), rtol=0, atol=1e-9)
    E = np.geomspace(1e13, 1e14, 3)
    F = np.ones((3, 3)) / 3
    M_rot = np.zeros((3, 3), dtype=complex)
    u = np.array([0.5, 0.5, 0.0], dtype=complex)
    args = (E, F, M_rot, u, 2e6, 1e-4, 0.1, 1.676389e-4, True, np.ones(3), 1.5, False, 0.0,
            False, 1, 2)
    with pytest.raises(ValueError, match="phase"):
        dps.propagate_earth_spatial(*args, N_steps_earth=4, show_progress=False,
                                    verbose=0, earth_step_scheme="left", earth_zenith_average="phase")
    with pytest.raises(ValueError, match="earth_zenith_average"):
        dps.propagate_earth_spatial(*args, N_steps_earth=4, show_progress=False,
                                    verbose=0, earth_step_scheme="avg", earth_zenith_average="median")
    assert dps.DEFAULT_CONFIG["grid"]["earth_zenith_average"] == "phase"
    assert dps.DEFAULT_CONFIG["grid"]["earth_sky"] == "full"
    with pytest.raises(ValueError, match="earth_sky"):
        dps.propagate_earth_spatial(*args, N_steps_earth=4, show_progress=False,
                                    verbose=0, earth_step_scheme="avg", earth_sky="down")


def test_core_count_respects_slurm_allocation_and_affinity(monkeypatch):
    """Pools are sized from dp_core.available_cores(): the affinity mask, capped by
    SLURM_CPUS_PER_TASK. os.cpu_count() alone reports a whole Tycho node (128 cores) whatever
    the job was given. The default pool keeps one core free on a laptop and uses the whole
    allocation inside a SLURM job."""
    import dp_core
    n_aff = len(os.sched_getaffinity(0))
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_ID"):
        monkeypatch.delenv(var, raising=False)
    assert dp_core.available_cores() == n_aff
    assert dp_core.default_n_cores() == max(1, n_aff - 1)
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert dp_core.available_cores() == 1 == dp_core.default_n_cores()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(n_aff + 100))
    assert dp_core.available_cores() == n_aff == dp_core.default_n_cores()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    assert dp_core.available_cores() == n_aff


# =============================================================================
# Normalization of the resonant absorption (2026-10-12)
# =============================================================================

def _rate_inputs(u, g_D, eps=0.1):
    U = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    return dps.compute_gamma_grid_inputs(U, np.array([u, u, 0.0], dtype=np.complex128), g_D, eps, np.ones(3))


def test_absorption_strength_scales_with_the_couplings():
    """The resonant absorption must scale with the couplings: at fixed flavor ratios the
    integrated rate grows with Sum G (Breit-Wigner with Gamma_in Gamma_tot in the numerator)
    and vanishes with it. Until 2026-10-12 the integrated rate was identical from Sum G =
    7.6e-4 down to 3e-19: a vanishing coupling only narrowed the resonance."""
    E = np.geomspace(1e13, 3e16, 60)
    z = np.linspace(4.0, 0.0, 60)
    m_nu = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    totals = []
    for g_D in (1e-6, 1e-8, 1e-10):
        sG, sGa, em = _rate_inputs(0.5, g_D)
        G = dps.build_thermal_grid(E, z, sG, sGa, 2e6, m_nu, 1.676389e-4, em)
        totals.append((sG, G.sum()))
    (s1, t1), (s2, t2), (s3, t3) = totals
    assert abs((t1 / t2) / (s1 / s2) - 1.0) < 1e-3, (t1 / t2, s1 / s2)
    assert abs((t2 / t3) / (s2 / s3) - 1.0) < 1e-3
    sG, sGa, em = _rate_inputs(0.0, 1e-4)
    assert dps.build_thermal_grid(E, z, sG, sGa, 2e6, m_nu, 1.676389e-4, em).max() == 0.0


def test_absorption_narrow_width_limit():
    """At a narrow width the rate must equal the delta-function limit of
    sigma_alpha = (Sum G)_alpha (Sum G) s / (3 pi [(s - M^2)^2 + M^2 Gamma^2]), whose integral over
    s is 4 pi (Sum G)_alpha: Gamma_alpha = sum_i eta_i (Sum G)_alpha / pi * int dp p^2 f_i (1 - c*)/B,
    with c* = (A - M^2)/B the angle where s = M^2 (A = m_i^2 + 2 x E_bg, B = 2 x p). The kernel's
    (1 - cos theta) flux factor is kept, as in Eq. (7) of the theory notes."""
    from scipy.integrate import quad
    m_nu = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    sG, sGa, em = _rate_inputs(0.5, 1e-12)          # width ~1e-16 of M: deep in the narrow limit
    T0, M = 1.676389e-4, 2e6
    for E0, z in ((2.2e14, 0.3), (6.0e13, 1.1), (4.1e13, 0.0)):
        got = dps.build_thermal_grid(np.array([E0, E0 * 1.0001]), np.array([z, z]), sG, sGa, M, m_nu, T0, em)[0, 0]
        x = E0 * (1.0 + z)
        ref = np.zeros(3)
        for i in range(3):
            m, Ti = m_nu[i], T0 * (1.0 + z) * em[i] ** (1.0 / 3.0)

            def integrand(p):
                E_bg = np.sqrt(m * m + p * p)
                A, B = m * m + 2 * x * E_bg, 2 * x * p
                c = (A - M * M) / B
                return 0.0 if abs(c) > 1 else p * p / (np.exp(p / Ti) + 1.0) * (1.0 - c) / B
            # the integrand jumps where c* = +-1: E_bg -+ p = delta = (M^2 - m^2)/(2x)
            d = (M * M - m * m) / (2 * x)
            edges = [abs(m * m - d * d) / (2 * d)]
            pts = sorted({*np.linspace(0, 20 * Ti, 41)[1:-1], *[e for e in edges if 0 < e < 20 * Ti]})
            val = quad(integrand, 0.0, 20 * Ti, limit=800, points=pts, epsabs=0, epsrel=1e-10)[0]
            ref += em[i] * sGa / np.pi * val
        assert ref.max() > 0, (E0, z)
        np.testing.assert_allclose(got, ref, rtol=2e-3, err_msg=f"E0={E0}, z={z}")


def test_vanishing_sterile_mixing_recovers_standard_propagation(tmp_path):
    """As the sterile mixing goes to zero the dark sector must switch off continuously: the
    flux at u = 1e-10 equals the one at u = 0 (no interactions). Until 2026-10-12 any nonzero
    u, however small, kept the full resonant absorption and its runaway cascade."""
    import copy
    base = dps.create_default_config(str(tmp_path / "cfg.json"))
    base["grid"].update(dict(n_E_bins=60, n_z_bins=60, n_sun_paths=1, n_earth_chords=2,
                             N_steps_earth=10))
    base["execution"].update(dict(n_cores=1, verbose=0))
    out = []
    for u in (0.0, 1e-10):
        c = copy.deepcopy(base)
        c["dark_portal"].update(u_e=u, u_mu=u)
        out.append(dps.run_solver(c)[1])
    np.testing.assert_allclose(out[1], out[0], rtol=1e-8)


def test_source_redshift_weighting_main_solver():
    """The main solver's sources weigh in as int dz R(z) (1+z)^-p / H(z) -- every emitted neutrino
    arrives (the injection per unit observed energy carries dE_z/dE = 1+z). Without interactions the
    flux is a pure power law and the ratio of z_max = 4 to z_max = 1 is that of the closed form (to the
    z-grid's first-order error); the weighting R(z)/(1+z) used until 2026-10-13 gives 16% less."""
    from scipy.integrate import quad
    import dark_portal_solver as dps
    from dp_cosmo import calc_R_source
    fl = {}
    for zmax in [1.0, 4.0]:
        cfg = {"dark_portal": {"u_e": 0.0, "u_mu": 0.0, "u_tau": 0.0, "g_D": 1e-30, "epsilon_prime": 0.0},
               "source": {"z_max": zmax, "p_index": 2.5},
               "grid": {"n_E_bins": 20, "n_z_bins": 400, "n_earth_chords": 2, "n_sun_paths": 2},
               "execution": {"n_cores": 1, "verbose": 0}}
        E, params = dps.build_solver_params(cfg)
        fl[zmax] = dps.propagate_weighted(E, params).sum(1)
    Om, OL = params["Omega_m"], params["Omega_L"]
    I = lambda zmax: quad(lambda z: calc_R_source(z, 0.0) * (1 + z) ** -2.5 / np.sqrt(Om * (1 + z) ** 3 + OL),
                          0, zmax, points=[1.0], limit=200)[0]
    ratio = fl[4.0] / fl[1.0]
    assert np.ptp(ratio) < 1e-12 * ratio.mean()
    assert abs(ratio.mean() / (I(4.0) / I(1.0)) - 1) < 5e-3
