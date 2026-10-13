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
    """Majorana mass-basis couplings lambda = U^T G U: a single flavor gives |lambda_ij| =
    g |U_fi| |U_fj|; a flavor-universal coupling is g U^T U, off-diagonal at delta_CP = 212 deg
    (it would be g 1 only without CP violation, as 2107.13568 assumes)."""
    U = _mixing()
    gu = si.mass_basis_couplings(si.flavor_couplings("universal", 0.3), U)
    assert np.allclose(gu, 0.3 * (U.T @ U), rtol=0, atol=1e-15)
    assert 0.03 < np.max(np.abs(gu - 0.3 * np.eye(3))) < 0.05
    gt = si.mass_basis_couplings(si.flavor_couplings("tau", 0.2), U)
    assert np.allclose(np.abs(gt), 0.2 * np.outer(np.abs(U[2]), np.abs(U[2])), rtol=0, atol=1e-16)
    gm = si.mass_basis_couplings(si.flavor_couplings("matrix", 1.0, [[0, 1, 0], [1, 0, 0], [0, 0, 0]]), U)
    assert np.allclose(gm, gm.T) and np.max(np.abs(gm.imag)) > 1e-3


def test_phases_drop_out():
    """Every rate is invariant under the Majorana phases (U -> U diag(e^{i phi})), for any texture;
    for a single flavor the Dirac phase drops out too (the tensors equal those of the real
    couplings g |U_fi| |U_fj|)."""
    U = _mixing()
    G = np.random.default_rng(7).normal(size=(3, 3))
    G = 0.1 * (G + G.T)
    P = np.diag(np.exp(1j * np.array([0.0, 0.7, -1.9])))
    A1, R1, _ = si.coupling_tensors(si.mass_basis_couplings(G, U))
    A2, R2, _ = si.coupling_tensors(si.mass_basis_couplings(G, U @ P))
    assert np.allclose(A1, A2, rtol=1e-12, atol=1e-30) and np.allclose(R1, R2, rtol=1e-12, atol=1e-30)
    At, Rt, _ = si.coupling_tensors(si.mass_basis_couplings(si.flavor_couplings("tau", 0.2), U @ P))
    Ar, Rr, _ = si.coupling_tensors(0.2 * np.outer(np.abs(U[2]), np.abs(U[2])))
    assert np.allclose(At, Ar, rtol=1e-12, atol=1e-30) and np.allclose(Rt, Rr, rtol=1e-12, atol=1e-30)


@pytest.mark.parametrize("s", [0.6, 30.0])
def test_complex_couplings_total_is_integral_of_differential(s):
    """With complex couplings the closed-form channel sums (coupling_tensors' identities) equal the
    integral of the generalized differential cross section over t, summed over the final states,
    plus double-scalar production; the CP-odd part averages out between nu and nubar."""
    from scipy import integrate
    rng = np.random.default_rng(11)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    g = 0.05 * (A + A.T)
    M = 1.0
    Gam = si.scalar_width(M, g)
    for i, j in [(0, 0), (0, 2), (1, 2)]:
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
        cpm = [si.dsigma_dt_scalar_majorana(s, -0.3 * s, M, Gam, g, i, j, 0, 2, cp=c) for c in (-1, 0, 1)]
        assert abs(0.5 * (cpm[0] + cpm[2]) - cpm[1]) <= 1e-14 * abs(cpm[1])


@pytest.mark.parametrize("complex_couplings", [False, True])
def test_coupling_identities(complex_couplings):
    """Every 2 -> 2 absorption yields two secondaries and every double-scalar one four: the sums
    over the secondary's mass state of the regeneration coefficients against the absorption's
    (real and complex couplings)."""
    g = _random_couplings(seed=5)
    if complex_couplings:
        A = np.random.default_rng(6).normal(size=(3, 3))
        g = g + 0.05j * (A + A.T)
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


# =============================================================================
# The solver's dispatch and the analysis
# =============================================================================

def test_solver_dispatch_and_analysis_model():
    """With secret_interactions.enabled the solver's flux functions return dp_si's transport in
    flavor (the injection's nu/nubar split of the total), and the analysis floats m_phi_eV and
    g_phi like any solver key; unsupported settings fail at parsing."""
    import dark_portal_solver as dps
    cfg = {"secret_interactions": {"enabled": True, "texture": "tau", "m_phi_eV": 5e6, "g_phi": 0.05},
           "grid": {"n_E_bins": 100}, "execution": {"n_cores": 1, "verbose": 0}}
    E, params = dps.build_solver_params(cfg)
    F = dps.propagate_weighted(E, params)
    U = C.construct_PMNS(params["th12"], params["th13"], params["th23"], params["d_cp"])
    gm = si.mass_basis_couplings(si.flavor_couplings("tau", 0.05), U)
    Fm = si.propagate_si(E, params["z_max"], params["H0_eV"], params["Omega_m"], params["Omega_L"], U,
                         params["m_nu_bg_arr"], params["T_nu0_eV"], np.ones(3),
                         np.array([params["f_e_S"], params["f_mu_S"], params["f_tau_S"]]), params["p_index"],
                         params["rho_evol"], params["Phi_0"], 5e6, gm)
    assert np.allclose(F, Fm @ (np.abs(U) ** 2).T, rtol=1e-12, atol=0)
    Fa, Fb = dps.propagate_nu_nubar(E, params)
    w = dps.injection_weights(params)
    assert np.allclose(w[0] * Fa + w[1] * Fb, F, rtol=1e-12, atol=0)
    sys.path.insert(0, REPO)
    from analysis.model import FlavorModel
    model = FlavorModel(cfg, free_names=["m_phi_eV", "g_phi"])
    f1 = model.fractions({"m_phi_eV": 5e6, "g_phi": 0.05})
    f0 = model.fractions({"m_phi_eV": 5e6, "g_phi": 0.0})
    assert np.max(np.abs(f1 - f0)) > 1e-4 and abs(f1.sum() - 1) < 1e-12    # 9e-4 in f_e
    for bad in [{"mediator": "tensor"}, {"neutrino_type": "weyl"}, {"texture": "sterile"}, {"g_phi": 4.0},
                {"channels": "t"}, {"z_substeps": 0}, {"dirac_sterile": "keep"}]:
        with pytest.raises(ValueError):
            dps.build_solver_params({"secret_interactions": dict(enabled=True, **bad)})


# =============================================================================
# Brute-force amplitudes (tests/amplitudes_bruteforce.py): explicit Dirac matrices and spinors,
# Majorana fermions by the fermion-flow rules of Denner et al. -- independent of every closed form
# =============================================================================

sys.path.insert(0, os.path.dirname(__file__))


def test_bruteforce_reproduces_textbook_qed():
    """The brute-force engine itself: Bhabha (LR -> LR) and Moller (LL -> LL) by photon exchange
    give 4 e^4 u^2 (1/s + 1/t)^2 and 4 e^4 s^2 (1/t + 1/u)^2."""
    import amplitudes_bruteforce as B
    for s, c in [(7.0, 0.3), (2.0, -0.6)]:
        m2, t, u = B.vector_nunubar(s, c, 1e-9, 1.0, 1.0)
        assert abs(m2 / (4 * u * u * (1 / s + 1 / t) ** 2) - 1) < 1e-12
        m2, t, u = B.vector_nunu(s, c, 1e-9, 1.0, 1.0)
        assert abs(m2 / (4 * s * s * (1 / t + 1 / u) ** 2) - 1) < 1e-12


def test_bruteforce_vector_mediator():
    """A vector coupled to left-handed currents, any complex couplings: nu nubar -> nu nubar is
    4 u^2 |A_s/(s - M^2) + A_t/(t - M^2)|^2, nu nu -> nu nu is 4 s^2 |A_t/(t - M^2) + A_u/(u - M^2)|^2,
    and nu nubar -> V V is 4 [t/u + u/t + 4 M^2 s/(t u) - M^4 (1/t^2 + 1/u^2)]."""
    import amplitudes_bruteforce as B
    rng = np.random.default_rng(1)
    for s, c, M in [(7.0, 0.3, 1.3), (0.3, 0.1, 1.0), (40.0, 0.9, 2.0)]:
        As, At, Au = rng.normal(size=3) + 1j * rng.normal(size=3)
        m2, t, u = B.vector_nunubar(s, c, M, As, At)
        assert abs(m2 / (4 * u * u * abs(As / (s - M * M) + At / (t - M * M)) ** 2) - 1) < 1e-12
        m2, t, u = B.vector_nunu(s, c, M, At, Au)
        assert abs(m2 / (4 * s * s * abs(At / (t - M * M) + Au / (u - M * M)) ** 2) - 1) < 1e-12
    for s, c, M in [(4.5, -0.2, 1.0), (1e4, 0.95, 3.0)]:
        m2, t, u = B.vector_pair(s, c, M)
        br = t / u + u / t + 4 * M * M * s / (t * u) - M ** 4 * (1 / t ** 2 + 1 / u ** 2)
        assert abs(m2 / (4 * br) - 1) < 1e-12


def test_bruteforce_scalar_majorana():
    """Majorana neutrinos, scalar exchange: Eq. (A12) for real couplings; for complex ones the
    generalization of dsigma_dt_scalar_majorana including its CP-odd term (cp = -1 for an incident
    neutrino, +1 for an antineutrino); double-scalar production with the coherent factor."""
    import amplitudes_bruteforce as B
    rng = np.random.default_rng(5)
    M, Gam = 1.0, 0.05
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    for lam in (0.4 * (A.real + A.real.T), 0.4 * (A + A.T)):
        for h1, cp in [(-0.5, -1), (0.5, 1)]:
            for s, c in [(0.7, 0.2), (1.02, -0.5), (9.0, 0.8)]:
                for (i, j, k, l) in [(0, 0, 0, 0), (0, 1, 2, 2), (1, 2, 0, 1)]:
                    bf, t = B.majorana_dsigma_dt(s, c, M, Gam, lam, i, j, k, l, h1=h1)
                    an = si.dsigma_dt_scalar_majorana(s, t, M, Gam, lam, i, j, k, l, cp=cp)
                    assert abs(bf / an - 1) < 1e-12
    lam = 0.4 * (A + A.T)
    for s, c in [(5.0, 0.3), (40.0, -0.7)]:
        for (i, j) in [(0, 0), (0, 2)]:
            bf, t = B.majorana_phiphi(s, c, M, lam, i, j)
            assert abs(bf / si.dsigma_dt_scalar_majorana_phiphi(s, t, M, lam, i, j, mode="coherent") - 1) < 1e-12


def _dirac_secondaries_bruteforce(B, s, x, M, Gam, h, a, r, h1=-0.5):
    """Visible and invisible secondaries of each mass state at energy fraction x for an incident
    nu_a of helicity h1 on unpolarized relic nu_r and nubar_r (weight 1/2 each in units of the
    density per helicity), and the absorption density."""
    HS = (-0.5, 0.5)
    vis, inv, tot = np.zeros(3), np.zeros(3), 0.0
    for h2 in HS:
        for k in range(3):
            for l in range(3):
                for h3 in HS:
                    for h4 in HS:
                        d = abs(B.dirac_nunu(s, 2 * x - 1, M, h, a, r, k, l, h1, h2, h3, h4)[0]) ** 2 / (32 * np.pi * s)
                        tot += 0.5 * d
                        (vis if B.visible("nu", h3) else inv)[k] += d
                        d = abs(B.dirac_nunubar(s, 2 * x - 1, M, Gam, h, a, r, k, l, h1, h2, h3, h4)[0]) ** 2 / (32 * np.pi * s)
                        tot += d
                        (vis if B.visible("nu", h3) else inv)[k] += d
                        d = abs(B.dirac_nunubar(s, 1 - 2 * x, M, Gam, h, a, r, k, l, h1, h2, h3, h4)[0]) ** 2 / (32 * np.pi * s)
                        (vis if B.visible("nubar", h4) else inv)[l] += d
    return vis, inv, tot


def _dirac_secondaries_closed(s, x, M, Gam, h, a, r):
    """The derived Dirac coefficients (CP-even): number densities of visible and invisible nu_c at x."""
    H = h @ h.conj().T
    M2 = M * M
    mu = M2 / s
    bw = (s - M2) ** 2 + M2 * Gam ** 2
    P = [1.0, (1 - x) ** 2 / ((1 - x) + mu) ** 2, x ** 2 / (x + mu) ** 2, (1 - x) / ((1 - x) + mu),
         x * (1 - x) / ((x + mu) * (1 - x + mu)), x / (x + mu)]
    kin = [s / bw, 1 / s, 1 / s, (s - M2) / bw, 1 / s, (s - M2) / bw]
    p16 = 16 * np.pi
    vis, inv = np.zeros(3), np.zeros(3)
    for c in range(3):
        Cv = [abs(h[a, r]) ** 2 * H[c, c].real / p16, 0.0, H[a, a].real * abs(h[r, c]) ** 2 / p16, 0.0, 0.0,
              (h[r, a] * np.conj(h[r, c]) * H[a, c]).real / p16]
        Ci = [abs(h[a, r]) ** 2 * H[c, c].real / p16, abs(h[c, a]) ** 2 * H[r, r].real / (8 * np.pi),
              H[a, a].real * abs(h[r, c]) ** 2 / p16, (h[r, a] * np.conj(h[c, a]) * H[c, r]).real / p16,
              (h[c, a] * np.conj(h[c, r]) * H[a, r]).real / p16, 0.0]
        vis[c] = sum(Cv[q] * kin[q] * P[q] for q in range(6))
        inv[c] = sum(Ci[q] * kin[q] * P[q] for q in range(6))
    return vis, inv


def test_bruteforce_scalar_dirac():
    """Dirac neutrinos, scalar exchange (h = U^dag G U, Hermitian): the visible (nu_L, nubar_R) and
    invisible (nu_R, nubar_L) secondaries of each mass state, CP-averaged (the brute force with h and
    with h*), equal the derived coefficients; the total equals the appendix's Eq. (A8) for real
    couplings; an incident nu_R gives those of a nu_L with visibility swapped (parity)."""
    import amplitudes_bruteforce as B
    from scipy import integrate
    rng = np.random.default_rng(9)
    M = 1.0
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    for h in (0.3 * (A.real + A.real.T), 0.3 * (A + A.conj().T)):
        Gam = np.sum(abs(h) ** 2) * M / (8 * np.pi)
        for s in [0.5, 1.03, 6.0]:
            for x in [0.13, 0.91]:
                for (a, r) in [(0, 0), (1, 2)]:
                    v1, i1, _ = _dirac_secondaries_bruteforce(B, s, x, M, Gam, h, a, r)
                    v2, i2, _ = _dirac_secondaries_bruteforce(B, s, x, M, Gam, h.conj(), a, r)
                    vm, im = _dirac_secondaries_closed(s, x, M, Gam, h, a, r)
                    sc = max(abs(v1).max(), abs(i1).max())
                    assert np.max(np.abs(0.5 * (v1 + v2) - vm)) < 1e-13 * sc
                    assert np.max(np.abs(0.5 * (i1 + i2) - im)) < 1e-13 * sc
                    w1, j1, _ = _dirac_secondaries_bruteforce(B, s, x, M, Gam, h, a, r, h1=0.5)
                    w2, j2, _ = _dirac_secondaries_bruteforce(B, s, x, M, Gam, h.conj(), a, r, h1=0.5)
                    assert np.max(np.abs(0.5 * (w1 + w2) - im)) < 1e-13 * sc
                    assert np.max(np.abs(0.5 * (j1 + j2) - vm)) < 1e-13 * sc
    g = 0.3 * (A.real + A.real.T)
    Gam = np.sum(g ** 2) * M / (8 * np.pi)
    for s in [0.5, 6.0]:
        tot = integrate.quad(lambda x: _dirac_secondaries_bruteforce(B, s, x, M, Gam, g, 1, 2)[2], 0, 1, epsrel=1e-10)[0]
        M2 = M * M
        bw = (s - M2) ** 2 + M2 * Gam ** 2
        L = np.log(M2 / (M2 + s))
        A8 = sum(g[1, 2] ** 2 * g[k, l] ** 2 * s / bw
                 + (1.5 * g[1, k] ** 2 * g[2, l] ** 2 + 0.5 * g[1, l] ** 2 * g[2, k] ** 2) * ((s + 2 * M2) / (s * (s + M2)) + 2 * M2 / s ** 2 * L)
                 - g[1, 2] * g[k, l] * g[1, k] * g[2, l] * (M2 / s - 1) * (s + M2 * L) / bw
                 + 0.5 * g[1, k] * g[2, l] * g[1, l] * g[2, k] * (1 / s + 2 * M2 * (M2 + s) / (s ** 2 * (2 * M2 + s)) * L)
                 for k in range(3) for l in range(3)) / (16 * np.pi)
        assert abs(tot / A8 - 1) < 1e-9


# =============================================================================
# Dirac neutrinos and the vector mediator in the transport
# =============================================================================

def test_dirac_scalar_transport():
    """Scalar, Dirac: number balance of the visible+sterile run; tracking the sterile secondaries
    only adds visible neutrinos (they regenerate visible ones); without coupling, free streaming."""
    kw = _transport_kw(p_index=2.0)
    E = _grid(150)
    gm = si.mass_basis_couplings_model(si.flavor_couplings("tau", 0.1), kw["U_pmns"], "scalar", False)
    Ft, info = si.propagate_si(E, M=5e6, g_mass=gm, majorana=False, return_diagnostics=True, **kw)
    bal = (info["final_number"] - info["injected"]) - (info["secondaries_in_grid"] - info["absorbed"])
    assert abs(bal) < 1e-12 * info["injected"]
    Fd = si.propagate_si(E, M=5e6, g_mass=gm, majorana=False, dirac_sterile="drop", **kw)
    assert np.all(np.isfinite(Ft)) and np.all(Ft >= 0) and np.all(Fd >= 0)
    assert np.all(Ft >= Fd * (1 - 1e-12))
    assert np.max(Ft.sum(1) / Fd.sum(1)) > 1.05          # sterile secondaries matter below the dips
    g0 = si.mass_basis_couplings_model(si.flavor_couplings("tau", 1e-30), kw["U_pmns"], "scalar", False)
    F0 = si.propagate_si(E, M=5e6, g_mass=g0, majorana=False, **kw)
    assert np.max(np.abs(F0 / si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, kw["U_pmns"], kw["rho_src"], 2.0, 0.0, 1.0) - 1)) < 1e-12


@pytest.mark.parametrize("majorana", [True, False])
def test_vector_transport(majorana):
    """Vector coupled to left-handed currents: number balance, positivity, free streaming without
    coupling, and the forward-peaked regime (1 keV, where nearly every scattering is quasi-elastic)."""
    kw = _transport_kw()
    E = _grid(150)
    for tex, M, g in [("tau", 5e6, 0.1), ("universal", 1e3, 0.1), ("mu", 1e9, 1.0)]:
        gm = si.mass_basis_couplings_model(si.flavor_couplings(tex, g), kw["U_pmns"], "vector", majorana)
        F, info = si.propagate_si(E, M=M, g_mass=gm, mediator="vector", majorana=majorana, return_diagnostics=True, **kw)
        bal = (info["final_number"] - info["injected"]) - (info["secondaries_in_grid"] - info["absorbed"])
        assert abs(bal) < 1e-9 * info["injected"]
        assert np.all(np.isfinite(F)) and np.all(F >= -1e-12 * F.max())
    g0 = si.mass_basis_couplings_model(si.flavor_couplings("tau", 1e-30), kw["U_pmns"], "vector", majorana)
    F0 = si.propagate_si(E, M=5e6, g_mass=g0, mediator="vector", majorana=majorana, **kw)
    assert np.max(np.abs(F0 / si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, kw["U_pmns"], kw["rho_src"], 2.5, 0.0, 1.0) - 1)) < 1e-12


def test_vector_cross_sections():
    """The vector's closed forms: the tensors' channel sums against the integral of
    vector_dsigma_dt over the final states and both relic components; the narrow-width s channel
    integrates to 2 pi |G_ra|^2 per relic component (the Z-burst normalization); Dirac = Majorana/2."""
    from scipy import integrate
    rng = np.random.default_rng(4)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    G = 0.2 * (A + A.conj().T)
    M = 1.0
    Gam = si.mediator_width(M, G, "vector")
    Cm, _ = si.vector_tensors(G, True)
    Cd, _ = si.vector_tensors(G, False)
    assert np.allclose(Cd, 0.5 * Cm, rtol=0, atol=0)
    for s in [0.4, 3.0, 30.0]:
        mu = M * M / s
        Nq = np.empty(10)
        si.vector_norms(mu, si._GL24_X, si._GL24_W, Nq)
        bw = (s - M * M) ** 2 + M * M * Gam ** 2
        kin = [s / bw, (s - M * M) / bw, 1.0 / s, si.vv_kinematic(s, M * M)]
        for a, r in [(0, 0), (1, 2)]:
            tens = sum(si.VECTOR_W[q] * Cm[q, a, r].sum() * kin[si.VECTOR_JT[q]] * Nq[q] for q in range(9))
            # nu nubar: distinguishable final particles, every ordered (k, l); nu nu: identical-type
            # particles, the unordered final states (the 1/2 for k = l is in vector_dsigma_dt)
            quad = 0.0
            for k in range(3):
                for l in range(3):
                    quad += integrate.quad(lambda t: si.vector_dsigma_dt(s, t, M, Gam, G, a, r, k, l, relic="nubar"),
                                           -s, 0, epsrel=1e-12, limit=200)[0]
                    if l >= k:
                        quad += integrate.quad(lambda t: si.vector_dsigma_dt(s, t, M, Gam, G, a, r, k, l, relic="nu"),
                                               -s, 0, epsrel=1e-12, limit=200)[0]
            assert abs(tens / quad - 1) < 1e-10
    g1 = np.zeros((3, 3))
    g1[0, 0] = 1e-3
    Gam1 = si.mediator_width(1.0, g1, "vector")
    val = integrate.quad(lambda s: integrate.quad(lambda t: sum(si.vector_dsigma_dt(s, t, 1.0, Gam1, g1, 0, 0, k, l)
                                                               for k in range(3) for l in range(3)), -s, 0)[0],
                         1 - 1e4 * Gam1, 1 + 1e4 * Gam1, points=[1.0], limit=400, epsrel=1e-10)[0]
    assert abs(val / (2 * np.pi * 1e-6) - 1) < 2e-4


@pytest.mark.parametrize("mediator,nu_type", [("scalar", "dirac"), ("vector", "majorana"), ("vector", "dirac")])
def test_solver_dispatch_new_models(mediator, nu_type):
    """The solver's flux functions run the Dirac scalar and the vector through dp_si with the
    model's mass-basis couplings (U^dag G U)."""
    import dark_portal_solver as dps
    cfg = {"secret_interactions": {"enabled": True, "mediator": mediator, "neutrino_type": nu_type, "texture": "tau",
                                   "m_phi_eV": 5e6, "g_phi": 0.05},
           "grid": {"n_E_bins": 100}, "execution": {"n_cores": 1, "verbose": 0}}
    E, params = dps.build_solver_params(cfg)
    F = dps.propagate_weighted(E, params)
    U = C.construct_PMNS(params["th12"], params["th13"], params["th23"], params["d_cp"])
    gm = np.conjugate(U).T @ si.flavor_couplings("tau", 0.05) @ U
    Fm = si.propagate_si(E, params["z_max"], params["H0_eV"], params["Omega_m"], params["Omega_L"], U,
                         params["m_nu_bg_arr"], params["T_nu0_eV"], np.ones(3),
                         np.array([params["f_e_S"], params["f_mu_S"], params["f_tau_S"]]), params["p_index"],
                         params["rho_evol"], params["Phi_0"], 5e6, gm, majorana=(nu_type == "majorana"), mediator=mediator)
    assert np.allclose(F, Fm @ (np.abs(U) ** 2).T, rtol=1e-12, atol=0)
