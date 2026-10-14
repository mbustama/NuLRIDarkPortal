"""
Tests of the transit to the Earth (src/dp_transit.py and the Earth's exterior in
src/dp_spatial.py), added 2026-10-13 with the redesign of the spatial handover: the graded
eigensystems, the frame table and the adiabatic-frame integrator against brute force, the
Galactic potential's Yukawa integral, the Earth's exterior and the cosmological horizon factor.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import dp_core as C  # noqa: E402
from dp_core import njit  # noqa: E402
from dp_propagator import propagator_3x3  # noqa: E402
import dp_transit as T  # noqa: E402
import dp_spatial as SP  # noqa: E402


def _mixing():
    th = np.radians([33.70916346871883, 8.559057632313843, 43.280093616242404, 212.0])
    U = C.construct_PMNS(*th)
    M_rot = np.ascontiguousarray(U @ np.diag([0.0, 7.49e-5, 2.513e-3]).astype(complex) @ U.conj().T)
    u = np.array([0.0316227766, 0.0316227766, 0.0], dtype=complex)
    return M_rot, np.ascontiguousarray(np.outer(u, u.conj()))


# =============================================================================
# Eigensystems at any potential
# =============================================================================

@pytest.mark.parametrize("texture", ["rank1", "full", "degenerate"])
def test_graded_eigensystem_matches_high_precision(texture):
    """frame_at's eigenvalues and eigenvectors agree with 60-digit arithmetic to ~1e-14 for
    potentials up to 1e40 times the vacuum term, where LAPACK on the formed Hamiltonian loses
    the vacuum splitting of a texture's degenerate eigenvalues (rank one: u u^dag; degenerate:
    diag(1, 1, 0) in a rotated basis, whose pair is not at zero)."""
    import mpmath as mp
    mp.mp.dps = 60
    M_rot, S = _mixing()
    if texture == "full":
        S = np.ascontiguousarray(np.array([[1.0, 0.2j, 0.1], [-0.2j, -0.7, 0.3], [0.1, 0.3, 0.2]]))
    S_mp = None
    if texture == "degenerate":
        # exactly degenerate in 60 digits (its double rounding splits the pair by ~eps, which
        # texture_basis snaps; the reference must be the texture meant, not its rounding)
        rng = np.random.default_rng(3)
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        cols = []
        for k in range(3):
            v = mp.matrix([mp.mpc(complex(A[a, k])) for a in range(3)])
            for w in cols:
                v = v - (sum(mp.conj(w[a]) * v[a] for a in range(3))) * w
            v = v / mp.sqrt(sum(abs(v[a]) ** 2 for a in range(3)))
            cols.append(v)
        S_mp = mp.matrix(3, 3)
        for a in range(3):
            for b in range(3):
                S_mp[a, b] = sum(cols[k][a] * mp.conj(cols[k][b]) for k in range(2))
        S = np.ascontiguousarray(np.array([[complex(S_mp[a, b]) for b in range(3)] for a in range(3)]))
    Q, s, Mp, floor, _, _ = T.texture_scales(M_rot, S)
    for sig in (0.0, 1e-3, 1.0, 1e3, 1e10, 1e20, 1e30, 1e40):
        vals = np.empty(3)
        W = np.empty((3, 3), dtype=complex)
        T.frame_at(Q, Mp, s, floor, sig, vals, W)
        H = mp.matrix(3, 3)
        for a in range(3):
            for b in range(3):
                Sab = S_mp[a, b] if S_mp is not None else mp.mpc(complex(S[a, b]))
                H[a, b] = mp.mpc(complex(M_rot[a, b])) + mp.mpf(sig) * Sab
        E, V = mp.eighe(H)
        order = sorted(range(3), key=lambda i: E[i])
        for k in range(3):
            j = order[k]
            if texture == "degenerate":
                # texture_basis shifts every eigenvalue by sigma: compare the splittings
                ref = float(E[j] - E[order[0]])
                assert abs((vals[k] - vals[0]) - ref) <= 1e-13 * max(abs(ref), 1e-3), (sig, k)
            else:
                assert abs(vals[k] - float(E[j])) <= 1e-13 * max(abs(float(E[j])), 1e-3), (sig, k)
            ov = abs(sum(mp.conj(mp.mpc(complex(W[a, k]))) * V[a, j] for a in range(3)))
            assert abs(1.0 - float(ov)) < 1e-13, (sig, k)


def test_texture_basis_snaps_rank_one_zeros():
    _, S = _mixing()
    Q, s = T.texture_basis(S)
    assert s[0] == 0.0 and s[1] == 0.0 and s[2] > 0.0
    np.testing.assert_allclose(Q @ np.diag(s) @ Q.conj().T, S, atol=1e-17)


def test_log_unitary_inverts_the_exponential():
    import scipy.linalg as sl
    rng = np.random.default_rng(1)
    for ang in (1e-9, 1e-3, 0.1, 1.0, 2.5):
        A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        K0 = A - A.conj().T
        K0 *= ang / np.max(np.abs(np.linalg.eigvals(K0)))
        K = np.empty((3, 3), dtype=complex)
        T.log_unitary3(np.ascontiguousarray(sl.expm(K0)), K)
        assert np.max(np.abs(K - K0)) < 1e-15 + 1e-15 * ang


def test_frame_table_nodes_turn_at_most_dtheta_and_are_eigenbases():
    M_rot, S = _mixing()
    for sign in (1.0, -1.0):
        tab = T.build_frame_table(M_rot, sign * S, dtheta=0.01)
        assert len(tab["sig"]) > 100
        for i in range(len(tab["sig"]) - 1):
            assert T._rotation_proxy(tab["W"][i], tab["W"][i + 1]) <= 0.01 + 1e-12
        for i in range(0, len(tab["sig"]), 37):
            H = M_rot + tab["sig"][i] * sign * S
            W = tab["W"][i]
            D = W.conj().T @ H @ W
            off = D - np.diag(np.diag(D))
            assert np.max(np.abs(off)) <= 1e-12 * max(np.max(np.abs(np.diag(D))), 1e-3)
            # (H formed in double precision carries eps |H| of rounding)
            np.testing.assert_allclose(np.real(np.diag(D)), tab["nu"][i], rtol=0,
                                       atol=4e-15 * max(np.max(np.abs(tab["nu"][i])), 1e-3))
        # aligned phases: consecutive overlaps real and positive
        for i in range(len(tab["sig"]) - 1):
            ov = np.diag(tab["W"][i].conj().T @ tab["W"][i + 1])
            assert np.all(np.abs(np.imag(ov)) < 1e-12) and np.all(np.real(ov) > 0.9)


# =============================================================================
# The integrator against brute force
# =============================================================================

@njit(cache=False)
def _brute(E, kappa, n_bg, Cn, m, D, ca, t_max, rho0, M_rot, S, nsteps):
    """Fourth-order Magnus in the flavor basis, from t_max down to 0, with steps resolving
    every oscillation: the reference for march_path."""
    h = t_max / nsteps
    rho = rho0.copy()
    c1 = 0.5 - np.sqrt(3.0) / 6.0
    c2 = 0.5 + np.sqrt(3.0) / 6.0
    Hh = np.empty((3, 3), dtype=np.complex128)
    Uo = np.empty((3, 3), dtype=np.complex128)
    for i in range(nsteps):
        x0 = i * h
        s1 = kappa * (n_bg + T._path_n(t_max - (x0 + c1 * h), Cn, m, D, ca))
        s2 = kappa * (n_bg + T._path_n(t_max - (x0 + c2 * h), Cn, m, D, ca))
        H1 = (M_rot + s1 * S) / (2.0 * E)
        H2 = (M_rot + s2 * S) / (2.0 * E)
        Om = -1j * h * 0.5 * (H1 + H2) - (np.sqrt(3.0) / 12.0) * h * h * (H2 @ H1 - H1 @ H2)
        Hh[:, :] = 1j * Om
        for a in range(3):
            Hh[a, a] = complex(Hh[a, a].real, 0.0)
            for b in range(a + 1, 3):
                v = 0.5 * (Hh[a, b] + np.conj(Hh[b, a]))
                Hh[a, b] = v
                Hh[b, a] = np.conj(v)
        propagator_3x3(Hh, 1.0, Uo)
        rho = Uo @ rho @ Uo.conj().T
    return rho


@pytest.mark.parametrize("E, v_over_gap, ca", [(1e15, 3.0, 0.9), (1e15, 0.3, -0.6), (1e14, 3.0, 0.4)])
def test_march_path_matches_brute_force(E, v_over_gap, ca):
    """Through a 1/r potential that turns on over AU scales comparable to the oscillation
    length -- neither adiabatic nor sudden -- the adiabatic-frame integrator reproduces a
    brute-force integration that resolves every oscillation, to its first-order error in the
    table spacing (~1e-4 at dtheta = 0.01, ~1e-5 at 0.002)."""
    M_rot, S = _mixing()
    gap = 2.513e-3 / (2 * E)
    kappa = 2 * E
    Cn = v_over_gap * gap / 2e-3
    m, D = 1e-22, C.d_sun_eV_inv
    t_max = min(300 * D, 3e5 / gap / (1 + v_over_gap))
    tab = T.build_frame_table(M_rot, S, dtheta=0.002)
    vals = np.empty(3)
    W0 = np.empty((3, 3), dtype=complex)
    T.frame_at(tab["Q"], tab["Mp"], tab["s"], tab["floor"], kappa * T._path_n(t_max, Cn, m, D, ca), vals, W0)
    rho0 = np.ascontiguousarray(W0 @ np.diag([0.5, 0.3, 0.2]).astype(complex) @ W0.conj().T)
    out = np.empty((3, 3), dtype=complex)
    T.march_path(E, kappa, 0.0, Cn, m, D, ca, t_max, rho0, tab["sig"], tab["nu"], tab["W"], tab["K"],
                 tab["Q"], tab["Mp"], tab["s"], tab["floor"], T._GLX8, T._GLW8, out)
    ns = int(t_max * gap * (1 + v_over_gap) * 40) + 1000
    ref = _brute(E, kappa, 0.0, Cn, m, D, ca, t_max, rho0, M_rot, S, ns)
    assert np.max(np.abs(out - ref)) < 6e-5
    assert abs(np.trace(out).real - 1.0) < 1e-12


def test_march_path_adiabatic_limit_is_the_local_eigenbasis():
    """When the potential grows slowly against the oscillation length (V AU >> 1), the march
    ends in the local eigenbasis with the starting populations, whatever the path."""
    M_rot, S = _mixing()
    E = 1e13
    gap = 2.513e-3 / (2 * E)
    Cn = 50 * gap / 2e-3
    m, D, kappa = 1e-22, C.d_sun_eV_inv, 2 * E
    tab = T.build_frame_table(M_rot, S, dtheta=0.002)
    F = np.array([0.5, 0.3, 0.2])
    t_max = 1e6 * D
    vals = np.empty(3)
    W0 = np.empty((3, 3), dtype=complex)
    W1 = np.empty((3, 3), dtype=complex)
    T.frame_at(tab["Q"], tab["Mp"], tab["s"], tab["floor"], kappa * T._path_n(t_max, Cn, m, D, -0.5), vals, W0)
    T.frame_at(tab["Q"], tab["Mp"], tab["s"], tab["floor"], kappa * Cn, vals, W1)
    rho0 = np.ascontiguousarray(W0 @ np.diag(F).astype(complex) @ W0.conj().T)
    out = np.empty((3, 3), dtype=complex)
    T.march_path(E, kappa, 0.0, Cn, m, D, -0.5, t_max, rho0, tab["sig"], tab["nu"], tab["W"], tab["K"],
                 tab["Q"], tab["Mp"], tab["s"], tab["floor"], T._GLX8, T._GLW8, out)
    pops = np.real(np.diag(W1.conj().T @ out @ W1))
    np.testing.assert_allclose(pops, F, atol=2e-4)


# =============================================================================
# The Galaxy
# =============================================================================

def test_mw_sky_density_table_is_current():
    """src/data/mw_sky_density.npz matches fresh quadrature of the Galactic model (rerun
    tools/mw_sky_density.py after changing calc_MW_neutron_density)."""
    import mw_sky_density as M
    d = np.load(os.path.join(REPO, "src", "data", "mw_sky_density.npz"))
    assert str(d["model_digest"]) == M.model_digest()
    r, nb = d["r_kpc"], d["nbar_cm3"]
    for x in (1e-4, 0.1, 1.0, 8.33, 50.0):
        i = int(np.argmin(np.abs(r - x)))
        assert abs(M.sky_average(r[i]) / nb[i] - 1.0) < 1e-12


def test_galactic_effective_density_limits():
    """The local density for short ranges (exactly, above the contact switch, and continuous
    across it); for ranges far beyond the Galaxy (still inside the horizon), m^2 times the
    Coulomb integral int nbar r dr plus the cosmic mean's share."""
    from dp_spatial import calc_MW_neutron_density
    R_H = 1.0 / (67.4 * C.km_per_s / C.Mpc_to_eV_inv)
    n0 = 3.091e-8 * C.cm_inv3_to_eV3
    n_loc = calc_MW_neutron_density(0.0, 0.0, 0.0) * C.cm_inv3_to_eV3
    assert T.galactic_effective_density(1e-12, n0, R_H) == n_loc
    assert T.galactic_effective_density(2e6, n0, R_H) == n_loc
    below = T.galactic_effective_density(0.9999e-12, n0, R_H)
    assert abs(below / n_loc - 1.0) < 1e-13
    # m -> 0 inside the horizon: n_eff / m^2 -> int nbar(r) r dr over the Galaxy (+ cosmic part)
    r, nbar = T._mw_table()
    coul = float(np.sum(0.5 * (nbar[1:] * r[1:] ** 2 + nbar[:-1] * r[:-1] ** 2) * np.diff(np.log(r))))
    m = 1e-31
    gal_only = T.galactic_effective_density(m, 0.0, R_H)
    assert abs(gal_only / m**2 / coul - 1.0) < 2e-3


# =============================================================================
# The Earth's exterior and the cosmological horizon
# =============================================================================

def test_earth_exterior_is_the_point_source_far_away():
    """Outside the Earth n_eff = (m/r) e^{-m(r-R)} P(R): for m R << 1 the point source of the
    Earth's 1.849e51 neutrons, m^2 N / (4 pi r)."""
    m_eV = 1e-20
    yk = SP.earth_yukawa_table(m_eV)
    m_km = m_eV * SP._KM_IN_EV_INV
    for r in (6371.0 * 2, 6371.0 * 100):
        n = SP._ext_density_cm3(r, yk)
        n_point = C.N_n_earth * m_km**2 / (4 * np.pi * r) / 1e15     # km^-3 -> cm^-3
        assert abs(n / n_point - 1.0) < 2e-3


def test_earth_exterior_columns_against_direct_quadrature():
    """earth_exterior_chord's step columns and dcol against scipy quadrature along the line."""
    from scipy.integrate import quad
    yk = SP.earth_yukawa_table(1e-13)
    R = 6371.0
    for c in (-0.9, -0.05, 0.3, 0.99):
        L, col, dcol = SP.earth_exterior_chord(c, 1.5, yk, SP._GL10_X, SP._GL10_W)
        b = (R - 1.5) * np.sqrt(1 - c * c)
        f = lambda r: SP._ext_density_cm3(r, yk) * r / np.sqrt(r * r - b * b)
        ref = quad(f, R, 2 * R, limit=400, epsabs=0, epsrel=1e-12)[0]
        assert abs(np.sum(col) / ref - 1.0) < 1e-9
        g = lambda r: SP._ext_density_cm3(r, yk) * (r / np.sqrt(r * r - b * b) - 1.0)
        ref_d = quad(g, 2 * R, 2e3 * R, limit=400, epsabs=0, epsrel=1e-12)[0] + \
            quad(g, 2e3 * R, np.inf, limit=400, epsabs=0, epsrel=1e-10)[0]
        assert abs(dcol / ref_d - 1.0) < 1e-6
        assert abs(np.sum(L) - (np.sqrt(4 * R * R - b * b) - np.sqrt(R * R - b * b))) < 1e-9


def test_earth_exterior_only_for_light_mediators():
    assert SP.earth_exterior_active(1e-10)
    assert not SP.earth_exterior_active(1e-3)
    assert not SP.earth_exterior_active(2e6)


def test_horizon_factor_limits():
    """(1 - (1 + x) e^{-x}) / m^2: 1/m^2 inside the horizon, R_H^2 / 2 beyond, continuous."""
    from dp_cosmo import yukawa_horizon_inv_m2
    R_H = 7e32
    assert yukawa_horizon_inv_m2(1e-24, R_H) == pytest.approx(1e24, rel=1e-15)
    assert yukawa_horizon_inv_m2(0.0, R_H) == pytest.approx(0.5 * R_H**2, rel=1e-15)
    for x in (1e-6, 0.9999e-3, 1.0001e-3, 0.5, 5.0):
        m = x / R_H
        exact = float((1 - (1 + x) * np.exp(-x)) / m**2) if x > 1e-2 else \
            R_H**2 * (x**2 / 2 - x**3 / 3 + x**4 / 8 - x**5 / 30) / x**2
        assert yukawa_horizon_inv_m2(m * m, R_H) == pytest.approx(exact, rel=1e-12)
    assert yukawa_horizon_inv_m2(4.0, np.inf) == 0.25


# =============================================================================
# Sources in a host galaxy (source.host_galaxy_scale, 2026-10-13)
# =============================================================================

def _fractions(scale, m, P, eps=1e-20, grid=(60, 40)):
    import io
    import dark_portal_solver as dps
    from flux_postprocessing import compute_energy_averaged_ratios
    u = 0.0316227766
    g = P / (eps * u * u) if eps > 0 else P / (u * u)
    cfg = {"dark_portal": dict(m_Zd_eV=m, g_D=g, epsilon_prime=eps, u_e=u, u_mu=u, u_tau=0.0,
                               use_debye_screening=False),
           "source": {"host_galaxy_scale": scale},
           "execution": {"n_cores": 1, "verbose": 0}, "grid": dict(n_E_bins=grid[0], n_z_bins=grid[1])}
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        E, F = dps.run_solver(cfg)
    finally:
        sys.stdout = old
    return np.asarray(compute_energy_averaged_ratios(E, F)), F


def test_host_galaxy_off_is_the_default():
    """host_galaxy_scale = 0 is the cosmic mean: identical, bit for bit, to omitting it."""
    import io
    import dark_portal_solver as dps
    base = {"dark_portal": dict(m_Zd_eV=1e-21, g_D=1e-34, epsilon_prime=0.1, u_e=0.0316227766,
                                u_mu=0.0316227766, u_tau=0.0),
            "execution": {"n_cores": 1, "verbose": 0}, "grid": dict(n_E_bins=40, n_z_bins=30)}
    with_key = {**base, "source": {"host_galaxy_scale": 0.0}}
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        _, F0 = dps.run_solver(base)
        _, F1 = dps.run_solver(with_key)
    finally:
        sys.stdout = old
    assert np.array_equal(F0, F1)


def test_host_galaxy_moves_the_source_transition():
    """With the sources in a Milky-Way-like host, the injected flavors split over the
    potential's eigenstates at couplings 1e4 times lower than in the cosmic mean density; deep
    in that regime both give the same composition (the sources see the potential), and without
    a coherent potential (epsilon' = 0) the host changes nothing."""
    m = 1e-21
    deep_off, _ = _fractions(0.0, m, 1e8 * m * m)
    host_on, _ = _fractions(1.0, m, 1.0 * m * m)
    off_there, _ = _fractions(0.0, m, 1.0 * m * m)
    assert np.max(np.abs(host_on - deep_off)) < 2e-4
    assert np.max(np.abs(off_there - deep_off)) > 1e-2
    _, F_a = _fractions(0.0, 1e-3, 1e-12, eps=0.0)
    _, F_b = _fractions(5.0, 1e-3, 1e-12, eps=0.0)
    assert np.array_equal(F_a, F_b)
