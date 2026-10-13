"""
Tests of the secret-interaction module (src/dp_si.py, begun 2026-10-13): the scalar cross sections
of arXiv:2107.13568 (Appendix A) against the paper's closed form and direct quadrature of its
differential cross sections, the stable kinematic integrals and regeneration profiles against
high-precision references, and the cosmological transport against its exact limits (no coupling,
number balance, the narrow-width limit). Small grids: a few seconds after compilation.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core as C  # noqa: E402
import dp_si as si  # noqa: E402

H0 = 67.4 * C.km_per_s / C.Mpc_to_eV_inv


def _mixing():
    th = np.radians([33.70916346871883, 8.559057632313843, 43.280093616242404, 212.0])
    return C.construct_PMNS(*th)


def _transport_kw(p_index=2.5):
    U = _mixing()
    return dict(z_max=4.0, H0=H0, Omega_m=0.315, Omega_L=0.685, U_pmns=U,
                masses=C.get_mass_eigenvalues(0.01, 7.49e-5, 2.513e-3, "normal"), T_nu0=1.676389e-4,
                eta_mass=np.ones(3), rho_src=np.array([1 / 3, 2 / 3, 0.0]), p_index=p_index, rho_evol=0.0,
                Phi_0=1.0)


def _random_couplings(seed=1, scale=0.05):
    g = np.random.default_rng(seed).normal(size=(3, 3))
    return scale * (g + g.T)


# =============================================================================
# Cross sections
# =============================================================================

@pytest.mark.parametrize("s", [0.6, 0.99, 1.0, 1.7, 3.9, 4.5, 30.0, 1e3])
def test_channels_match_printed_A11(s):
    """The channel decomposition (sums over the final states done with the coupling identities)
    equals Eq. (A11) as printed, for a generic real coupling matrix (paper's double-scalar
    coupling), at and around the resonance and the double-scalar threshold. Far above it the
    printed form loses digits itself (its log of (r - s + 2 M^2) cancels, ~eps s^2/M^4; ours is
    exact there, test_kinematic_integrals_high_precision)."""
    g = _random_couplings()
    tol = 1e-13 if s < 100 else 1e-11
    M = 1.0
    Gam = si.scalar_width(M, g)
    for i, j in [(0, 0), (0, 2), (1, 2)]:
        a = si.sigma_scalar_majorana_total(s, M, Gam, g, i, j, mode="paper")
        b = si.sigma_scalar_majorana_A11(s, M, Gam, g, i, j, mode="paper")
        assert abs(a / b - 1) < tol


@pytest.mark.parametrize("s", [0.6, 30.0])
def test_total_is_integral_of_differential(s):
    """Eq. (A11) is the integral of Eq. (A12) over t in [-s, 0], summed over the unordered final
    states with the 1/2 for k = l, plus Eq. (A13) over its kinematic range (phiphi_t_range): this
    fixes the reading of the appendix's prefactors and of the scalar's t limits."""
    from scipy import integrate
    g = _random_couplings()
    M = 1.0
    Gam = si.scalar_width(M, g)
    i, j = 0, 2
    q = 0.0
    for k in range(3):
        for l in range(k, 3):
            q += integrate.quad(lambda t: si.dsigma_dt_scalar_majorana(s, t, M, Gam, g, i, j, k, l),
                                -s, 0.0, epsabs=0, epsrel=1e-12, limit=200)[0]
    if s > 4.0:
        tm, tp = si.phiphi_t_range(s, M)
        q += integrate.quad(lambda t: si.dsigma_dt_scalar_majorana_phiphi(s, t, M, g, i, j),
                            tm, tp, epsabs=0, epsrel=1e-12, limit=400, points=[0.5 * (tm + tp)])[0]
    assert abs(si.sigma_scalar_majorana_total(s, M, Gam, g, i, j) / q - 1) < 1e-10


def test_kinematic_integrals_high_precision():
    """The double-scalar bracket and the profile norms agree with 50-digit arithmetic from
    threshold to s/M^2 = 1e14 and for r = s/M^2 from 1e-8 to 1e9, where the closed forms cancel."""
    import mpmath as mp
    mp.mp.dps = 50

    def Ipp(s):
        s = mp.mpf(s)
        r = mp.sqrt(s * (s - 4))
        return ((s * s - 4 * s + 6) / (s - 2) * mp.log(((r + s - 2) / (r - s + 2)) ** 2) - 6 * r) / s ** 2

    for s in [4.0000001, 4.01, 5.0, 20.0, 1e6, 1e14]:
        assert abs(si.phiphi_kinematic(s, 1.0) / float(Ipp(s)) - 1) < 1e-14
    for r in [1e-8, 1e-2, 0.49, 0.51, 3.0, 1e9]:
        rm = mp.mpf(r)
        L = mp.log1p(rm)
        ref = (1 + 1 / (1 + rm) - 2 * L / rm, (rm - L) / rm, 1 - 2 * (1 + rm) * L / (rm * (2 + rm)))
        got = si.profile_norms(r, si._GL24_X, si._GL24_W)
        for a, b in zip(got, ref):
            assert abs(a / float(b) - 1) < 1e-14


def test_narrow_width_limit():
    """Across a narrow resonance the s channel integrates to pi g_ar^2 (summed over the final
    states): the normalization behind 'half the number of targets' (module notes)."""
    from scipy import integrate
    g = 1e-3 * np.diag([1.0, 0.7, 0.4])
    M = 1.0
    Gam = si.scalar_width(M, g)
    b = M * Gam
    val = integrate.quad(lambda s: si.sigma_scalar_majorana_channels(s, M, Gam, g, 1, 1)[0], 1 - 1e4 * b, 1 + 1e4 * b,
                         points=[1.0], limit=400, epsrel=1e-12)[0]
    assert abs(val / (np.pi * g[1, 1] ** 2) - 1) < 2e-4      # the BW tails beyond 1e4 widths


def test_mass_basis_couplings():
    """Universal couplings stay diagonal, a single flavor gives g |U_fi| |U_fj| whatever the CP
    phase, and a generic texture with complex mass-basis couplings is refused."""
    U = _mixing()
    gu = si.mass_basis_couplings(si.flavor_couplings("universal", 0.3), U)
    assert np.allclose(gu, 0.3 * np.eye(3), atol=1e-15)
    gt = si.mass_basis_couplings(si.flavor_couplings("tau", 0.2), U)
    assert np.allclose(gt, 0.2 * np.outer(np.abs(U[2]), np.abs(U[2])), rtol=0, atol=1e-16)
    with pytest.raises(ValueError):
        si.mass_basis_couplings(si.flavor_couplings("matrix", 1.0, [[0, 1, 0], [1, 0, 0], [0, 0, 0]]), U)


def test_coupling_identities():
    """Every 2 -> 2 absorption yields two secondaries and every double-scalar one four: the sums
    over the secondary's mass state of the regeneration coefficients against the absorption's."""
    g = _random_couplings(seed=5)
    Cabs, Creg, G2 = si.coupling_tensors(g)
    S = Creg.sum(axis=3)
    assert np.allclose(S[0], 2 * Cabs[0], rtol=1e-13, atol=0)
    assert np.allclose(S[1] + S[2], 2 * Cabs[1], rtol=1e-13, atol=0)
    assert np.allclose(S[3] + S[5], 2 * Cabs[2], rtol=1e-12, atol=1e-30)
    assert np.allclose(S[4], 2 * Cabs[3], rtol=1e-13, atol=0)
    assert np.allclose(S[6], 4 * Cabs[4], rtol=1e-13, atol=0)


# =============================================================================
# Regeneration profiles
# =============================================================================

def test_profile_tables():
    """The bin integrals of the regeneration profiles add up to their closed-form norms and agree
    bin by bin with 30-digit quadrature, from mu = M^2/s = 1e-9 (far above the resonance, where
    the t and u channels peak within mu of the endpoints) to 1e8 (far below)."""
    import mpmath as mp
    mp.mp.dps = 30
    dlnE = 0.02
    mus = np.array([1e-9, 3e-3, 0.019, 0.2, 1.0, 1e3, 1e8])
    B = si.build_profile_table(mus, 2400, dlnE, 4096)
    h = np.exp(dlnE / 2)
    P = [None, lambda x, m: (1 - x) ** 2 / ((1 - x) + m) ** 2, lambda x, m: x ** 2 / (x + m) ** 2,
         lambda x, m: (1 - x) / ((1 - x) + m), lambda x, m: x * (1 - x) / ((x + m) * (1 - x + m)), lambda x, m: x / (x + m)]
    for n, mu in enumerate(mus):
        N2, N4, N5 = si.profile_norms(1.0 / mu, si._GL24_X, si._GL24_W)
        target = [1.0, N2, N2, N4, N5, N4, 1.0 if mu < 0.25 else 0.0]
        sums = B[n].sum(axis=0)
        for q in range(7):
            assert abs(sums[q] - target[q]) <= 1e-12 * max(target[q], 1.0)
        for d in [0, 1, 5, 9, 300]:
            x_hi = 1.0 if d == 0 else h ** (-(2 * d - 1))
            x_lo = h ** (-(2 * d + 1))
            for q in range(1, 6):
                ref = float(mp.quad(lambda x: P[q](x, mp.mpf(mu)), [mp.mpf(x_lo), mp.mpf(x_hi)]))
                assert abs(B[n, d, q] / ref - 1) < 1e-13


def test_double_scalar_neutrino_spectrum():
    """The neutrinos of double-scalar production (each scalar's energy from Eq. A13, decaying flat
    in [0, E_phi], Eq. A16) agree bin by bin with 30-digit quadrature."""
    import mpmath as mp
    mp.mp.dps = 30
    dlnE = 0.02
    h = np.exp(dlnE / 2)
    mu = 0.01
    B = si.build_profile_table(np.array([mu]), 1500, dlnE, 8192)[0, :, 6]
    m = mp.mpf(mu)
    beta = mp.sqrt(1 - 4 * m)
    xm, xp = (1 - beta) / 2, (1 + beta) / 2
    f = lambda x: -(2 * x - 1) ** 2 * (x * x - x + m) / ((x - 1 + m) ** 2 * (x - m) ** 2)  # noqa: E731
    Z = mp.quad(f, [xm, xm + m * m, xm + m, 0.5, xp - m, xp - m * m, xp])

    def H(y):
        y = mp.mpf(y)
        F0 = mp.quad(f, [xm, min(y, xp)]) if y > xm else 0
        lo = max(y, xm)
        F1 = mp.quad(lambda x: f(x) / x, [lo, xp]) if lo < xp else 0
        return (F0 + y * F1) / Z

    for d in [1, 3, 20, 115, 460]:
        ex = float(H(h ** (-(2 * d - 1))) - H(h ** (-(2 * d + 1))))
        assert abs(B[d] / ex - 1) < 1e-9


# =============================================================================
# Transport
# =============================================================================

def _grid(n=100):
    return np.logspace(13, np.log10(3e16), n)


def test_no_coupling_is_free_streaming():
    """With the coupling -> 0 the transport returns the free-streaming fluxes (exactly integrated
    source term)."""
    kw = _transport_kw()
    E = _grid()
    gm = si.mass_basis_couplings(si.flavor_couplings("tau", 1e-30), kw["U_pmns"])
    F = si.propagate_si(E, M=5e6, g_mass=gm, **kw)
    F0 = si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, kw["U_pmns"], kw["rho_src"], 2.5, 0.0, 1.0)
    assert np.max(np.abs(F / F0 - 1)) < 1e-12


@pytest.mark.parametrize("tex,M,g", [("tau", 5e6, 0.1), ("universal", 1e5, 0.03), ("e", 1e9, 1.0)])
def test_number_balance(tex, M, g):
    """Number in the grid at z = 0 = injected + secondaries in the grid - absorbed, to rounding;
    two secondaries per absorption without double-scalar production (M = 1 GeV: s < 4 M^2
    throughout), more with it."""
    kw = _transport_kw()
    E = _grid()
    gm = si.mass_basis_couplings(si.flavor_couplings(tex, g), kw["U_pmns"])
    F, info = si.propagate_si(E, M=M, g_mass=gm, return_diagnostics=True, **kw)
    bal = (info["final_number"] - info["injected"]) - (info["secondaries_in_grid"] - info["absorbed"])
    assert abs(bal) < 1e-12 * info["injected"]
    assert np.all(np.isfinite(F)) and np.all(F >= 0)
    ratio = info["secondaries_expected"] / info["absorbed"]
    if M == 1e9:
        assert abs(ratio - 2.0) < 1e-12
    elif M == 1e5:
        assert ratio > 2.01
    assert info["secondaries_in_grid"] <= info["secondaries_expected"] * (1 + 1e-12)


def test_absorption_only_is_below_free_streaming():
    """Without regeneration the fluxes only lose particles; with it the s channel at 5 MeV piles
    secondaries below the dips (2107.13568, Fig. 5)."""
    kw = _transport_kw(p_index=2.0)
    E = _grid(200)
    gm = si.mass_basis_couplings(si.flavor_couplings("tau", 0.1), kw["U_pmns"])
    F0 = si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, kw["U_pmns"], kw["rho_src"], 2.0, 0.0, 1.0)
    Fa = si.propagate_si(E, M=5e6, g_mass=gm, regeneration=False, **kw)
    Fr = si.propagate_si(E, M=5e6, g_mass=gm, **kw)
    assert np.all(Fa <= F0 * (1 + 1e-12))
    assert Fr.sum(1)[0] > 2.0 * F0.sum(1)[0]


def test_delta_limit_independent_of_coupling():
    """In the narrow-width ('delta') approximation every neutrino crossing the resonance is
    absorbed once g >~ 1e-3 (2107.13568, Eq. 4): g = 0.01 and 0.1 give the same spectrum."""
    kw = _transport_kw(p_index=2.0)
    E = _grid(150)
    F = []
    for g in [0.01, 0.1]:
        gm = si.mass_basis_couplings(si.flavor_couplings("tau", g), kw["U_pmns"])
        F.append(si.propagate_si(E, M=5e6, g_mass=gm, channels="delta", **kw))
    # relative to the free-streaming flux: e^-tau at the crossings differs (tau ~ g_ar^2, and
    # g |U_tau a| |U_tau r| is small for some pairs at g = 0.01): 5e-4 here
    F0 = si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, kw["U_pmns"], kw["rho_src"], 2.0, 0.0, 1.0)
    assert np.max(np.abs(F[0] - F[1]) / F0) < 2e-3


def test_fig5_regression():
    """The 2107.13568 Fig. 5 setup (nu_tau, M = 5 MeV, g = 0.1, E^-2, sum of masses 0.1 eV,
    normal ordering) on the analysis grid: all-flavor flux over free streaming at 1e4 and 1e7 GeV,
    as validated on 2026-10-13 (reproduces the paper's curves; 1600-bin reference within 4e-3)."""
    from scipy.optimize import brentq
    U = C.construct_PMNS(np.radians(33.44), np.radians(8.57), np.radians(49.2), np.radians(197.0))
    m1 = brentq(lambda m: C.get_mass_eigenvalues(m, 7.42e-5, 2.517e-3, "normal").sum() - 0.1, 1e-6, 0.1)
    kw = _transport_kw(p_index=2.0)
    kw.update(U_pmns=U, masses=C.get_mass_eigenvalues(m1, 7.42e-5, 2.517e-3, "normal"))
    E = np.logspace(13, np.log10(3e16), 400)
    gm = si.mass_basis_couplings(si.flavor_couplings("tau", 0.1), U)
    F = si.propagate_si(E, M=5e6, g_mass=gm, **kw)
    F0 = si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, U, kw["rho_src"], 2.0, 0.0, 1.0)
    r = F.sum(1) / F0.sum(1)
    assert abs(r[0] - 2.208) < 0.01
    assert abs(r[np.argmin(np.abs(E - 1e16))] - 0.581) < 0.01
