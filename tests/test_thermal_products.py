"""
Thermal relics in every collision term (dp_si thermal="full", 2026-10-18): the lattice weights carry rho's
normalization and its moments in the node coordinate up to the third (cubic interpolation is exact for cubics), and
the transport agrees with the reference (tools/thermal_reference.py: rho-weighted mixtures of at-rest relics by direct
quadrature) on a small grid, where the resonance-only treatment (thermal=True) does not. Dirac relics (2026-10-21):
the same with the chiral shares rho_L, rho_R = rho (1 +- pi)/2, and the polarized transport (visible and sterile
together) against the parity runs when the shares are made equal.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import dp_si  # noqa: E402
import thermal_reference as tr  # noqa: E402

DELTA = 0.0228


@pytest.mark.parametrize("pol", [0, 1, -1])
@pytest.mark.parametrize("m, T", [(1e-6, 3.4e-4), (1e-3, 3.4e-4), (8.6e-3, 1e-3), (0.05, 1.7e-4), (1.0, 1e-4)])
def test_weights_carry_rho_moments(m, T, pol):
    """pol +-1: a Dirac relic's chiral shares, against the reference's rho (1 +- pi)/2 written out directly, to 1e-7
    of rho's own moments (the right-chiral share of a relativistic relic, ~m^2/p^2 of rho, lies partly below rho's
    support, at 6e-11 of rho for m = 1e-6); the shares' weights add up to rho's (to 1e-6: the shares' quadrature is
    split at sigma = m, rho's not)."""
    W = np.zeros(512)
    i0, s, nw = dp_si._thermal_weights(m, T, 0.37, DELTA, 8, 16.0, 1e-9, W, pol)
    assert nw > 0
    nu = i0 + s * np.arange(nw)
    glx, glw = tr._gl(0.0, 1.0, 24)
    sig = np.empty(4000)
    w = np.empty(4000)
    w0 = np.empty(4000)
    n = tr.sigma_nodes(m, T, np.empty(0), glx, glw, sig, w, pol)
    tr.sigma_nodes(m, T, np.empty(0), glx, glw, sig, w0, 0)
    nus = 0.37 + np.log(sig[:n]) / DELTA
    for k in range(4):
        ref = (w[:n] * nus ** k).sum()
        scale = max(1.0, abs(ref), abs((w0[:n] * nus ** k).sum()))
        assert abs((W[:nw] * nu ** k).sum() - ref) <= 1e-7 * scale, k
    if pol == 1:
        W0, WR = np.zeros(512), np.zeros(512)
        assert dp_si._thermal_weights(m, T, 0.37, DELTA, 8, 16.0, 1e-9, W0, 0) == (i0, s, nw)
        dp_si._thermal_weights(m, T, 0.37, DELTA, 8, 16.0, 1e-9, WR, -1)
        assert np.max(np.abs(W + WR - W0)) <= 1e-6 * np.max(np.abs(W0))


_TRANSPORT_CASES = [("vector", 3e6, 3e-2, True), ("vector", 3e5, 1e-2, True), ("scalar", 3e6, 3e-2, True),
                    ("scalar", 3e5, 1e-2, True), ("vector", 3e6, 3e-2, False), ("vector", 3e5, 1e-2, False),
                    ("scalar", 3e6, 3e-2, False), ("scalar", 3e5, 1e-2, False)]


def _transport_kw(mediator, M, majorana):
    return dict(z_max=1.0, H0=1.44e-33, Omega_m=0.315, Omega_L=0.685, U_pmns=np.eye(3), T_nu0=1.68e-4,
                eta_mass=np.ones(3), rho_src=np.full(3, 1 / 3), p_index=2.0, rho_evol=0.0, Phi_0=1.0,
                majorana=majorana, channels="full", mediator=mediator, E_extension=2.0,
                masses=np.array([1e-6, 8.6e-3, 0.05]), M=M)


@pytest.mark.parametrize("mediator, M, g, majorana", _TRANSPORT_CASES)
def test_transport_matches_reference(mediator, M, g, majorana, monkeypatch):
    """thermal="full" against the reference, for both mediators (the scalar since 2026-10-21) and both kinds of
    neutrinos (Dirac 2026-10-21: the relics' chiral shares, the scalar tracking its sterile secondaries)."""
    E = 1e13 * np.exp(DELTA * np.arange(100))
    kw = _transport_kw(mediator, M, majorana)
    gm = g * np.eye(3, dtype=np.complex128)
    F0 = dp_si.propagate_si(E, g_mass=0 * gm, **kw).sum(axis=1)
    full = dp_si.propagate_si(E, g_mass=gm, thermal="full", **kw).sum(axis=1)
    res = dp_si.propagate_si(E, g_mass=gm, thermal=True, **kw).sum(axis=1)
    monkeypatch.setattr(dp_si, "_transport_columns", tr.transport_columns_ref)
    monkeypatch.setitem(tr.OPTIONS, "kernel", "thermal")
    dp_si._COLUMNS.clear()
    ref = dp_si.propagate_si(E, g_mass=gm, **kw).sum(axis=1)
    dp_si._COLUMNS.clear()
    err_full, err_res = np.max(np.abs(full - ref) / F0), np.max(np.abs(res - ref) / F0)
    assert err_full < 2e-4
    # the test can tell: the resonance-only treatment misses by far more (vector 3e-3 and up; scalar 1.1e-3 at 3 MeV)
    assert err_res > max(5e-4, 10.0 * err_full)


@pytest.mark.parametrize("pol", [0, 1, -1])
@pytest.mark.parametrize("m", [1e-6, 1e-5, 1e-4])
def test_resonant_kernel_resolves_the_relic_mass(m, pol):
    """relic_state_rates_analytic with fine_m: the thermal s/BW with the resonance at sigma = m (a relativistic relic)
    against direct quadrature over rho (with one [0, p_max] momentum panel it was 1e-2 off at m = 1e-5); pol +-1
    over a Dirac relic's chiral shares (rho_R: the relic moving along the beam, sigma < m)."""
    from scipy import integrate
    import dp_thermal
    M, gam, T0, z = 3.16e4, 1e-7, 1.6765e-4, 1.0
    T = T0 * (1 + z)
    a = M * M / m
    st = dp_thermal.relic_state_rates_analytic(a / (2 * (1 + z)), z, 12 * np.pi * gam, M, np.array([m, 0.0, 0.0]), T0,
                                               np.array([1.0, 0.0, 0.0]), dp_si._TGL16_X, dp_si._TGL16_W,
                                               dp_si._TGL10_X, dp_si._TGL10_W, dp_si._TFD16, True, 0, True, pol)[0]
    n = dp_si.relic_density_per_helicity(T0) * (1 + z) ** 3

    def f(u):
        s = a * np.exp(u)
        return tr.rho(np.exp(u), m, T, pol) * np.exp(u) * s / ((s - M * M) ** 2 + (M * M * gam) ** 2)
    u0 = np.log(m)
    cuts = sorted({np.log(m * m / (80 * T)), np.log(80 * T)} | {u0 + sg * gam * 10.0 ** k for k in range(9)
                                                                   for sg in (-1, 1)})
    ref = sum(integrate.quad(f, p, q, limit=400, epsabs=0, epsrel=1e-10)[0] for p, q in zip(cuts[:-1], cuts[1:]))
    assert abs(st / (2 * np.pi * n) / ref - 1.0) < 1e-6


def test_si_config_switch():
    """secret_interactions.thermal_products (2026-10-21) turns thermal="full" on, for Majorana and Dirac neutrinos;
    it needs the full channels."""
    import dark_portal_solver as dps
    base = {"secret_interactions": {"enabled": True, "mediator": "scalar", "neutrino_type": "majorana",
                                    "m_phi_eV": 3e6, "g_phi": 1e-2, "texture": "e"}}
    _, params = dps.build_solver_params(base)
    assert params["si"]["thermal"] is False
    on = {"secret_interactions": dict(base["secret_interactions"], thermal_products=True)}
    _, params = dps.build_solver_params(on)
    assert params["si"]["thermal"] == "full"
    _, params = dps.build_solver_params({"secret_interactions": dict(on["secret_interactions"], neutrino_type="dirac")})
    assert params["si"]["thermal"] == "full" and params["si"]["majorana"] is False
    with pytest.raises(ValueError, match="full"):
        dps.build_solver_params({"secret_interactions": dict(on["secret_interactions"], channels="s")})


@pytest.mark.parametrize("mediator, sterile", [("scalar", "track"), ("scalar", "drop"), ("vector", "track")])
def test_polarized_transport_without_polarization(mediator, sterile, monkeypatch):
    """The Dirac transport on the relics' chiral shares (relic slots weighted by rho_L, rho_R; the scalar's visible and
    sterile neutrinos in one six-state sweep) with the shares forced to rho/2 each -- an unpolarized background --
    equals the parity runs (sum and difference of visible and sterile, kappa = 1/2) on the full rho, to rounding:
    the chiral tensors, the six-state sweep and the slot bookkeeping against the at-rest decomposition."""
    E = 1e13 * np.exp(DELTA * np.arange(60))
    kw = dict(_transport_kw(mediator, 3e5, False), dirac_sterile=sterile)
    gm = 1e-2 * np.eye(3, dtype=np.complex128)
    w_orig, r_orig, spec_orig = dp_si._thermal_weights_all, dp_si._resonance_grid, dp_si.model_spec

    def half_weights(*a):
        w_orig(*a[:12], 0)
        if a[12] != 0:
            a[8][...] *= 0.5

    def half_kernels(*a):
        r_orig(*a[:14], 0)
        if a[14] != 0:
            a[12][...] *= 0.5
            a[13][...] *= 0.5

    def fresh():
        dp_si._STEP_TERMS.clear()
        dp_si._COLUMNS.clear()
    fresh()
    monkeypatch.setattr(dp_si, "_thermal_weights_all", half_weights)
    monkeypatch.setattr(dp_si, "_resonance_grid", half_kernels)
    pol = dp_si.propagate_si(E, g_mass=gm, thermal="full", **kw)
    monkeypatch.setattr(dp_si, "_thermal_weights_all", w_orig)
    monkeypatch.setattr(dp_si, "_resonance_grid", r_orig)
    monkeypatch.setattr(dp_si, "model_spec", lambda *a, **k: spec_orig(*a, **dict(k, polarized=False)))
    fresh()
    runs = dp_si.propagate_si(E, g_mass=gm, thermal="full", **kw)
    fresh()
    F0 = dp_si.propagate_si(E, g_mass=0 * gm, **kw)
    assert np.max(np.abs(pol - runs).sum(axis=1) / F0.sum(axis=1)) < 1e-11
    assert np.max(np.abs(runs - F0).sum(axis=1) / F0.sum(axis=1)) > 1e-2           # the coupling does something


@pytest.mark.parametrize("mode", [0, 1])
def test_kernels_far_below_the_resonance(mode):
    """relic_state_rates_analytic with fine_m at s ~ 1e-4 M^2, where the closed form's cancellation loses 1e-3 to 1e-2:
    the chiral shares' angular quadrature against direct 2D quadrature (2026-10-21), to the momentum quadrature's own
    ~3e-7 (pol 0 keeps the closed form while the dark-portal campaign runs)."""
    from scipy import integrate
    import dp_thermal
    T0, z, M, gam = 1.6765e-4, 0.7, 4.2e6, 1e-3
    T, M2, MG = T0 * (1 + z), M * M, M * M * gam
    for E0, m in [(1e12, 1e-3), (1e12, 1e-6), (3.16e12, 0.05)]:
        x = E0 * (1 + z)
        for pol in (1, -1):
            def ang(p):
                Ep = np.hypot(p, m)
                v = p / Ep
                fl = {0: lambda c: 1 - v * c, 1: lambda c: (1 + v) * (1 - c) / 2, -1: lambda c: (1 - v) * (1 + c) / 2}[pol]
                ker = lambda c: ((m * m + 2 * x * (Ep - p * c)) - (M2 if mode else 0.0)) / (
                    (m * m + 2 * x * (Ep - p * c) - M2) ** 2 + MG ** 2)
                return integrate.quad(lambda c: fl(c) * ker(c), -1, 1, epsabs=0, epsrel=1e-13)[0]
            ref = integrate.quad(lambda p: p * p / (np.exp(p / T) + 1) * ang(p), 0, 20 * T, epsabs=0, epsrel=1e-12,
                                 limit=200, points=[v_ for v_ in (2 * m, 20 * m, 200 * m) if v_ < 20 * T])[0] / (2 * np.pi)
            st = dp_thermal.relic_state_rates_analytic(E0, z, 12 * np.pi * gam, M, np.array([m, 0.0, 0.0]), T0,
                                                       np.array([1.0, 0.0, 0.0]), dp_si._TGL16_X, dp_si._TGL16_W,
                                                       dp_si._TGL10_X, dp_si._TGL10_W, dp_si._TFD16, True, mode, True,
                                                       pol)[0]
            assert abs(st / ref - 1) < 1e-6, (E0, m, pol)


def test_kernel_grid_thinning():
    """The resonant kernels' grid (2026-10-21): merged grids put nodes arbitrarily close, and 4-point Lagrange of a
    noisy log-kernel through two nearly coincident nodes overshoots without bound (an infinite kernel at 453 keV, a
    crash; isolated bins absorbed outright otherwise). _thin_grid keeps neighbours at least the gap apart, the ends
    kept; the interpolation through the thinned grid stays within the noise."""
    rng = np.random.default_rng(3)
    x = np.unique(np.r_[np.arange(0.0, 1.0, 0.02), np.arange(0.3001, 0.6, 0.0083), 0.4 + 1e-9])
    t = dp_si._thin_grid(x, 0.5 * 0.0083)
    assert t[0] == x[0] and t[-1] == x[-1] and np.all(np.diff(t) >= 0.5 * 0.0083 - 1e-15)
    assert np.array_equal(dp_si._thin_grid(x, 0.0), x)
    f = lambda u: -50.0 + 2.0 * u                                   # a smooth log-kernel, with 1% noise
    for grid in (x, t):
        y = (f(grid) + 0.01 * rng.standard_normal(len(grid)))[:, None]
        q = np.linspace(0.01, 0.98, 2001)
        err = [abs(dp_si._interp4(grid, y, 0, u, min(max(np.searchsorted(grid, u) - 1, 0), len(grid) - 2)) - f(u))
               for u in q]
        if grid is x:
            assert max(err) > 100.0                                 # the overshoot through the coincident pair
        else:
            assert max(err) < 0.1
