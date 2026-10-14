"""
Tests of the resonant absorption from the mediator's Lagrangian coupling (src/dp_absorption.py, the per-state kernel
in src/dp_thermal.py, the coherent tables in src/dp_cosmo.py; docs/dev/PHYSICS_AUDIT.md items 1a-1c, 1e, 1f, 1i, 1j;
2026-10-14). Absolute checks: the Z-burst normalization, a flavor-universal coupling absorbing each mass state only
on its own relic, the eigenstates the coupling misses left alone, the sum rules, the visible width, the Moller kernel
against dense quadrature, the Debye mass's limits.
"""
import os
import sys

import numpy as np

# np.trapz is gone from NumPy 2.x (CI runs the latest); the laptop and Tycho pin 1.26, which lacks np.trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core as C  # noqa: E402
import dp_absorption as A  # noqa: E402
import dp_thermal as TH  # noqa: E402
import dp_cosmo as CO  # noqa: E402
import dp_transit as TR  # noqa: E402
import dark_portal_solver as dps  # noqa: E402

U = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
M_NU = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
T0 = 1.676389e-4
G_F = 1.1663787e-23


def _integrated_rate(M, sum_G, m_nu, r, n=60001):
    """int Gamma_r dE over relic state r's own resonance at z = 0, per unit weight (physical normalization)."""
    Er = (M * M - m_nu[r] ** 2) / (2.0 * m_nu[r])
    E = np.geomspace(0.6 * Er, 1.6 * Er, n)
    g = TH.relic_state_grid_cached(E, np.array([0.0]), sum_G, M, m_nu, T0, np.ones(3), moller=True)[0][:, r]
    return _trapz(g, E)


def test_z_burst_normalization():
    """For the Z's coupling to nu_L, g/(2 c_W) (I), a relic at rest absorbs nu_alpha with <sigma> = int sigma ds / M^2
    = 2 sqrt2 pi G_F per target, weighted by the relic state's nu_alpha content: int Gamma dE = n <sigma> M^2 /(2 m)
    |U_alpha r|^2 (n = 56 cm^-3 per state). Heavy relics (m >> T) and a narrow width."""
    M = 3e6
    scale = 1e-3          # a Z-like coupling 1000 times weaker: Gamma/M ~ 5e-9, no Breit-Wigner tails outside the window
    ab = A.AbsorptionSetup(scale * (0.65 / (2 * 0.88)) * np.eye(3), U, M)
    m_nu = np.array([0.05, 0.06, 0.07])                       # all non-relativistic
    n = 3 * 1.2020569031595942 * T0 ** 3 / (4 * np.pi ** 2)  # per helicity and state
    G2 = scale ** 2 * (0.65 / (2 * 0.88)) ** 2                # sqrt2 G_F M_Z^2 in the code's constants, scaled
    Qabs, _ = ab.vacuum_weights()
    for r in range(3):
        I = _integrated_rate(M, ab.sum_G, m_nu, r)
        for a in range(3):
            expect = n * 2 * np.pi * G2 * (1.0 if a == r else 0.0) / (2 * m_nu[r])
            got = I * Qabs[r, a]
            assert abs(got - expect) <= 3e-3 * n * 2 * np.pi * G2 / (2 * m_nu[r]), (r, a, got, expect)


def test_universal_coupling_absorbs_each_mass_state_on_its_own_relic():
    """A flavor-universal coupling is diagonal in the mass basis: in vacuum mass state a meets only relic state a
    (the classic relic-neutrino absorption dips); the theory notes' flavor form spread it over all three."""
    ab = A.AbsorptionSetup(1e-3 * np.eye(3), U, 3e6)
    Qabs, Qreg = ab.vacuum_weights()
    assert np.allclose(Qabs, 0.5e-6 * np.eye(3), rtol=0, atol=1e-18)
    assert np.allclose(Qreg, 0.5e-6, rtol=1e-12)


def _tables(G, M=3e6, anti=False, sigma=(1e-30, 1e-6, 1e-3, 1.0, 1e3, 1e9)):
    ab = A.AbsorptionSetup(G, U, M, anti=anti)
    Us = U.conj() if anti else U
    M_rot = np.ascontiguousarray(Us @ np.diag([0, 7.41e-5, 2.511e-3]).astype(complex) @ Us.conj().T)
    u = np.array([0.5, 0.5, 0.0])
    S = np.ascontiguousarray(np.outer(u, u).astype(complex))
    Q, s, Mp, floor, _, _ = TR.texture_scales(M_rot, S)
    P, Qa, Qr = CO._build_sigma_coupling_tables(Q, Mp, s, floor, np.array(sigma), ab.A, ab.B)
    return ab, P, Qa, Qr


@pytest.mark.parametrize("anti", [False, True])
def test_coupling_tables_sum_rules_and_limits(anti):
    """Sum rules: sum_r Qabs[r, a] = Qreg[a] (unitarity of U); sum_a Qreg[a] = Tr(G G^dag)/2. Vacuum limit:
    the mass-basis couplings. Potential-dominated limit of the rank-one texture u u^dag: the two eigenstates
    orthogonal to u do not couple to g u u^dag at all."""
    u = np.array([0.5, 0.5, 0.0])
    G = 2e-3 * np.outer(u, u)
    ab, P, Qa, Qr = _tables(G, anti=anti)
    scale = Qr.max()                       # entries ~1e-14 of the scale are rounding (columns the coupling misses)
    assert np.allclose(Qa.sum(axis=1), Qr, rtol=1e-12, atol=1e-12 * scale)
    assert np.allclose(Qr.sum(axis=1), 0.5 * np.trace(G @ G.T), rtol=1e-12)
    # vacuum (sigma = 1e-30): the mass eigenstates, |(U^dag G U)_ra|^2 / 2 up to each column's phase
    Us = U.conj() if anti else U
    Gs = G.conj() if anti else G
    assert np.allclose(Qa[0], 0.5 * np.abs(Us.conj().T @ Gs @ Us) ** 2, rtol=1e-6, atol=1e-16)
    # potential-dominated (sigma = 1e9): one eigenstate carries it all
    top = Qa[-1].sum(axis=0)
    assert np.sum(top > 1e-12 * top.max()) == 1
    assert abs(top.max() / (0.5 * (2e-3 * 0.5) ** 2) - 1.0) < 1e-6


def test_visible_width_and_neutrino_fraction():
    """The width adds the visible channels; the decays' neutrino fractions sum to Gamma_nu / Gamma; the Z mixing's
    e+e- width at 3 MeV with kappa = eps' g/c_W; no visible channel below 2 m_e."""
    eps = 1e-3
    k = A.kappa_z_mixing(eps)
    assert abs(k - eps * 0.65 / 0.88) < 1e-18
    Gv = A.z_mixing_visible_width(3e6, eps)
    r = (A.M_E / 3e6) ** 2
    expect = 3e6 / (12 * np.pi) * k * k * ((-0.5 + 2 * A.SIN2_W) ** 2 * (1 + 2 * r) + 0.25 * (1 - 4 * r)) * np.sqrt(1 - 4 * r)
    assert abs(Gv / expect - 1) < 1e-12
    assert A.z_mixing_visible_width(1.0e6, eps) == 0.0
    G = A.dark_portal_coupling(1e-3, [0.5, 0.5, 0.0], eps)
    ab = A.AbsorptionSetup(G, U, 3e6, Gv)
    _, Qreg = ab.vacuum_weights()
    assert abs(Qreg.sum() / ab.sum_G - ab.Gamma_nu / ab.Gamma) < 1e-14
    assert abs(ab.Gamma - (3e6 * np.real(np.trace(G @ G.conj().T)) / (24 * np.pi) + Gv)) < 1e-12 * ab.Gamma


def test_theory_notes_form_is_the_old_kernel():
    """absorption_model 'theory_notes' keeps the old numbers: the per-state kernel summed with the flavor weights is
    the flavor kernel; and the old form is 6x the new one per target for a diagonal coupling (x3 counting, x2 width)."""
    E = np.geomspace(1e13, 1e16, 40)
    z = np.linspace(4.0, 0.0, 9)
    st = TH.relic_state_grid_cached(E, z, 1e-6, 3e6, M_NU, T0, np.ones(3), moller=False)
    old = TH.build_thermal_grid(E, z, 1e-6, np.full(3, 1e-6), 3e6, M_NU, T0, np.ones(3))
    assert np.max(np.abs(st.sum(axis=2) * 1e-6 - old[:, :, 0])) <= 1e-13 * old.max()
    # per target: the old Gamma_in = (Sum G)_alpha M / 12 pi with (Sum G)_alpha = c^2 on EVERY state; the new one
    # |(c U)_alpha r|^2 / 2 on state r -- summed over the states, (3 c^2) vs (c^2 / 2)
    ab = A.AbsorptionSetup(1e-3 * np.eye(3), U, 3e6)
    Qabs, _ = ab.vacuum_weights()
    assert abs(3 * 1e-6 / Qabs.sum(axis=0)[0] - 6.0) < 1e-12


def _dense_moller(E0, z, sum_G, m_Zd, m_nu, T_nu0):
    """Dense brute-force p-integral of the Moller closed form (independent numpy), per state."""
    T_nu = T_nu0 * (1 + z)
    MG = m_Zd ** 2 * sum_G / (12 * np.pi)
    MG2, M2, x = MG * MG, m_Zd ** 2, E0 * (1 + z)
    out = np.zeros(3)
    for i in range(3):
        m = m_nu[i]
        m2 = m * m
        pmax = 20 * T_nu
        delta = (M2 - m2) / (2 * x)
        cuts = [0.0, pmax]
        if delta > 0:
            pe = (m2 - delta ** 2) / (2 * delta) if delta < m else (delta ** 2 - m2) / (2 * delta)
            if 0 < pe < pmax:
                wz = MG / (2 * x * delta / np.sqrt(m2 + pe * pe))
                for fac in (1e6, 1e4, 1e2, 3.0):
                    for sgn in (-1, 1):
                        c = pe + sgn * fac * wz
                        if 0 < c < pmax:
                            cuts.append(c)
                cuts.append(pe)
        cuts = np.array(sorted(set(cuts)))
        tot = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            n = 6000
            p = a + (np.arange(n) + 0.5) * (b - a) / n
            Eb = np.sqrt(m2 + p * p)
            v = p / Eb
            Aa = m2 + 2 * x * Eb
            B = 2 * x * p
            up, um = Aa - M2 + B, Aa - M2 - B
            K = B - v * (Aa - M2)
            I = (2 * v * B + 0.5 * (K + v * M2) * np.log((up ** 2 + MG2) / (um ** 2 + MG2))
                 + (K * M2 - v * MG2) / MG * (np.arctan(up / MG) - np.arctan(um / MG))) / B ** 2
            f = 1 / (np.exp(p / T_nu) + 1)
            tot += np.sum(p * p * f * np.maximum(I, 0.0)) * (b - a) / n
        out[i] = tot / (2 * np.pi)
    return out


def test_moller_kernel_matches_dense_quadrature():
    """The per-state kernel with the Moller flux factor against an independent dense quadrature, significance-
    weighted as the flavor kernel's test (cells on and around each relic resonance, narrow to wide widths, a
    semi-relativistic lightest state)."""
    m_nu = dps.get_mass_eigenvalues(0.002, 7.41e-5, 2.511e-3, "normal")
    for m_Zd in (2e6, 2e7):
        for sum_G in (1e-14, 1e-8, 1e-2):
            rows = []
            cells = [(1e13, 0.5), (1e15, 1.0), (1e16, 3.0)]
            for i in range(3):
                for fac in (0.5, 1.0):
                    delta = fac * m_nu[i]
                    cells.append(((m_Zd ** 2 - m_nu[i] ** 2) / (2 * delta) / 2.0, 1.0))
            for E0, z in cells:
                truth = _dense_moller(E0, z, sum_G, m_Zd, m_nu, T0)
                new = TH.relic_state_rates_analytic(E0, z, sum_G, m_Zd, m_nu, T0, np.ones(3), TH._GL16_X, TH._GL16_W,
                                                    TH._GL10_X, TH._GL10_W, TH._FD16, True)
                rows += list(zip(truth, new))
            gmax = max(t for t, _ in rows)
            worst = max(abs(nw - t) / max(t, 1e-3 * gmax) for t, nw in rows)
            assert worst < 1e-5, (m_Zd, sum_G, worst)


def test_debye_mass_limits():
    """Relativistic relics: |G'|^2 T^2 / 6 per state (the theory notes' T23), Majorana and Dirac. Slow relics: a
    Majorana neutrino's charge under the left-handed current is -h v, so 9 zeta(3) |G'|^2 T^3 / (2 pi^2 m); a Dirac
    one's is (1 + v)/2, so |G'|^2 m T ln2 / (4 pi^2). Against a direct quadrature of q(p)^2 (-df/dE) at m ~ T."""
    from scipy.integrate import quad
    from scipy.special import zeta
    one, z3 = np.array([1.0, 0, 0]), np.ones(3)
    for maj in (True, False):
        rel = A.debye_mass_sq(one, np.array([1e-12, 0, 0]), T0, z3, majorana=maj)
        assert abs(rel / (T0 * T0 / 6) - 1) < 1e-12
    m = 1.0
    nr = A.debye_mass_sq(one, np.array([m, 0, 0]), T0, z3)
    assert abs(nr / (9 * zeta(3) * T0 ** 3 / (2 * np.pi ** 2 * m)) - 1) < 1e-6
    nr = A.debye_mass_sq(one, np.array([m, 0, 0]), T0, z3, majorana=False)
    assert abs(nr / (m * T0 * np.log(2) / (4 * np.pi ** 2) + T0 * T0 / 12) - 1) < 1e-6  # + the O(v) term
    m = 3 * T0
    for maj in (True, False):
        def integrand(p):
            E = np.sqrt(p * p + m * m)
            v = p / E
            q2 = v * v if maj else 0.25 * (1 + v) ** 2
            ex = np.exp(-p / T0)
            return p * p * q2 * (E / p) * ex / (1 + ex) ** 2 / T0  # -df/dE = (E/p) f (1 - f) / T
        # the d^3p/(2 pi)^3 integral of q^2 (-df/dE), both helicities
        direct = 2 * quad(integrand, 0, 60 * T0, epsabs=0, epsrel=1e-12, limit=200)[0] / (2 * np.pi ** 2)
        got = A.debye_mass_sq(one, np.array([m, 0, 0]), T0, z3, majorana=maj)
        assert abs(got / direct - 1) < 1e-9, (maj, got, direct)


def test_lagrangian_absorption_end_to_end():
    """The solver with the corrected absorption: no coupling at all (u = 0, eps' = 0) is standard propagation;
    with only the Z mixing (u = 0, eps' > 0) the mediator still couples to every neutrino and absorbs them flavor-
    universally, near-unchanged flavor fractions; nu and nubar sources give the same energy budget bounds."""
    import dp_core
    dp_core.HAS_TQDM = False
    base = {"grid": {"n_E_bins": 160, "n_z_bins": 160}, "execution": {"n_cores": 1, "verbose": 0}}

    def run(dp):
        E, params = dps.build_solver_params({**base, "dark_portal": dp})
        return E, dps.propagate_weighted(E, params)
    E, F0 = run({"u_e": 0.0, "u_mu": 0.0, "epsilon_prime": 0.0})
    _, Fz = run({"u_e": 0.0, "u_mu": 0.0, "epsilon_prime": 1e-2, "m_Zd_eV": 3e6})
    _, Fl = run({"m_Zd_eV": 3e6, "g_D": 1e-3, "epsilon_prime": 0.0})
    dE = C.compute_dE_geometric(E)
    for F in (Fz, Fl):
        assert np.sum(F.sum(1) * E * dE) <= np.sum(F0.sum(1) * E * dE) * (1 + 1e-9)
    ratio = Fz.sum(1) / F0.sum(1)
    assert ratio.min() < 0.5                                     # the Z mixing alone absorbs
    frac = lambda F: (F * dE[:, None]).sum(0) / (F * dE[:, None]).sum()
    assert np.max(np.abs(frac(Fz) - frac(F0))) < 2e-2


def test_screening_spares_the_uniform_cosmic_potential():
    """The relics screen the potentials of localized bodies (range 1/m_eff, m_eff^2 = m^2 + m_D^2) but not the uniform
    cosmic mean (no gradient, no force on them; no reservoir of dark charge -- decided 2026-10-14): at the Sun, with a
    screening mass far above the mediator's, the Galactic integral's cosmic part is n0 (1 - (1+x)e^-x) with the bare
    x = m R_H, the Sun's and the Moon's point sources vanish, and without screening nothing changes."""
    import dp_spatial as SP
    import dp_transit as TRN
    R_H = 1.0 / (67.4 * C.km_per_s / C.Mpc_to_eV_inv)
    n0 = 3.091e-8 * C.cm_inv3_to_eV3
    m, m_eff = 1e-30, 1e-20
    U3 = U
    M_rot = np.ascontiguousarray(U3 @ np.diag([0, 7.41e-5, 2.511e-3]).astype(complex) @ U3.conj().T)
    S = np.ascontiguousarray(np.outer([0.5, 0.5, 0], [0.5, 0.5, 0]).astype(complex))
    E = np.geomspace(1e13, 1e15, 3)
    F = np.ones((3, 3))
    yk = SP.earth_yukawa_table(m_eff)
    _, n_amb, info = SP.transit_to_earth(E, F, M_rot, S, 1e-30, m_eff, n0, R_H, yk, n_sun_paths=1, m_bare=m)
    gal_screened = TRN.galactic_effective_density(m_eff, n0, R_H)
    from dp_cosmo import yukawa_horizon_inv_m2
    expect = gal_screened + n0 * m_eff ** 2 * (yukawa_horizon_inv_m2(m * m, R_H) - yukawa_horizon_inv_m2(m_eff ** 2, R_H))
    assert abs(info["n_gal_eV3"] / expect - 1) < 1e-12
    # the cosmic part dominates and is the bare one: Coef(m_eff) x this = the unscreened uniform potential
    unscreened = n0 * m_eff ** 2 * yukawa_horizon_inv_m2(m * m, R_H)
    assert abs(info["n_gal_eV3"] / unscreened - 1) < 1e-6
    _, _, info0 = SP.transit_to_earth(E, F, M_rot, S, 1e-30, m_eff, n0, R_H, yk, n_sun_paths=1)
    assert info0["n_gal_eV3"] == gal_screened                         # no m_bare: as before
    _, _, info1 = SP.transit_to_earth(E, F, M_rot, S, 1e-30, m_eff, n0, R_H, yk, n_sun_paths=1, m_bare=m_eff)
    assert info1["n_gal_eV3"] == gal_screened                         # no screening: nothing added


def test_negative_potential_sign_is_the_flipped_texture():
    """g_D eps' < 0 (the other sign of the dark portal's potential, 2026-10-14) equals the positive product with the
    texture's sign flipped, in the cosmological table as in the transit: the cosmological phase used |sigma| with
    the sign only from the kind and the matter charge until then."""
    E = np.geomspace(1e13, 1e16, 8)
    z = np.linspace(4.0, 0.0, 12)
    M_sq = np.ascontiguousarray(np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128))
    H0 = 67.4 * dps.km_per_s / dps.Mpc_to_eV_inv
    u = np.ascontiguousarray(np.array([0.5, 0.5, 0.0], dtype=np.complex128))
    S = np.outer(u, u.conj())

    def run(eps, Sx):
        return dps.propagate_spectrum(E, z, np.array([1 / 3, 2 / 3, 0.0]), H0, 0.315, 0.685, np.ascontiguousarray(U),
                                      M_sq, u, 1e-19, 1e-33, eps, M_NU, 3.091e-8 * C.cm_inv3_to_eV3, T0, False, 1.5,
                                      False, 0.0, (0.5, 0.5), 2.5, 0.0, 1.0, np.ones(3), 2, 4, show_progress=False,
                                      verbose=0, n_cores=1, N_steps_earth=25, U_LRI_struct=Sx, earth_step_scheme="avg",
                                      earth_zenith_average="phase", earth_sky="full")
    a = run(-1e-20, S)
    b = run(1e-20, -S)
    c = run(1e-20, S)
    for x, y in zip(a, b):
        assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(x))
    assert np.max(np.abs(c[0] - a[0])) > 1e-3 * np.max(np.abs(a[0]))
