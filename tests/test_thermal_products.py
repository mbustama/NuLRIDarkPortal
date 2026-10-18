"""
Thermal relics in every collision term (dp_si thermal="full", 2026-10-18): the lattice weights carry rho's
normalization and its moments in the node coordinate up to the third (cubic interpolation is exact for cubics), and
the transport agrees with the reference (tools/thermal_reference.py: rho-weighted mixtures of at-rest relics by direct
quadrature) on a small grid, where the resonance-only treatment (thermal=True) does not.
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


@pytest.mark.parametrize("m, T", [(1e-6, 3.4e-4), (1e-3, 3.4e-4), (8.6e-3, 1e-3), (0.05, 1.7e-4), (1.0, 1e-4)])
def test_weights_carry_rho_moments(m, T):
    W = np.zeros(512)
    i0, s, nw = dp_si._thermal_weights(m, T, 0.37, DELTA, 8, 16.0, 1e-9, W)
    assert nw > 0
    nu = i0 + s * np.arange(nw)
    glx, glw = tr._gl(0.0, 1.0, 24)
    sig = np.empty(4000)
    w = np.empty(4000)
    n = tr.sigma_nodes(m, T, np.empty(0), glx, glw, sig, w)
    nus = 0.37 + np.log(sig[:n]) / DELTA
    for k in range(4):
        ref = (w[:n] * nus ** k).sum()
        assert abs((W[:nw] * nu ** k).sum() - ref) <= 1e-7 * max(1.0, abs(ref)), k


@pytest.mark.parametrize("M, g", [(3e6, 3e-2), (3e5, 1e-2)])
def test_transport_matches_reference(M, g, monkeypatch):
    E = 1e13 * np.exp(DELTA * np.arange(100))
    kw = dict(z_max=1.0, H0=1.44e-33, Omega_m=0.315, Omega_L=0.685, U_pmns=np.eye(3), T_nu0=1.68e-4,
              eta_mass=np.ones(3), rho_src=np.full(3, 1 / 3), p_index=2.0, rho_evol=0.0, Phi_0=1.0, majorana=True,
              channels="full", mediator="vector", E_extension=2.0, masses=np.array([1e-6, 8.6e-3, 0.05]), M=M)
    gm = g * np.eye(3, dtype=np.complex128)
    F0 = dp_si.propagate_si(E, g_mass=0 * gm, **kw).sum(axis=1)
    full = dp_si.propagate_si(E, g_mass=gm, thermal="full", **kw).sum(axis=1)
    res = dp_si.propagate_si(E, g_mass=gm, thermal=True, **kw).sum(axis=1)
    monkeypatch.setattr(dp_si, "_transport_columns", tr.transport_columns_ref)
    monkeypatch.setitem(tr.OPTIONS, "kernel", "thermal")
    dp_si._COLUMNS.clear()
    ref = dp_si.propagate_si(E, g_mass=gm, **kw).sum(axis=1)
    dp_si._COLUMNS.clear()
    assert np.max(np.abs(full - ref) / F0) < 2e-4
    assert np.max(np.abs(res - ref) / F0) > 3e-3


@pytest.mark.parametrize("m", [1e-6, 1e-5, 1e-4])
def test_resonant_kernel_resolves_the_relic_mass(m):
    """relic_state_rates_analytic with fine_m: the thermal s/BW with the resonance at sigma = m (a relativistic relic)
    against direct quadrature over rho (with one [0, p_max] momentum panel it was 1e-2 off at m = 1e-5)."""
    from scipy import integrate
    import dp_thermal
    M, gam, T0, z = 3.16e4, 1e-7, 1.6765e-4, 1.0
    T = T0 * (1 + z)
    a = M * M / m
    st = dp_thermal.relic_state_rates_analytic(a / (2 * (1 + z)), z, 12 * np.pi * gam, M, np.array([m, 0.0, 0.0]), T0,
                                               np.array([1.0, 0.0, 0.0]), dp_si._TGL16_X, dp_si._TGL16_W,
                                               dp_si._TGL10_X, dp_si._TGL10_W, dp_si._TFD16, True, 0, True)[0]
    n = dp_si.relic_density_per_helicity(T0) * (1 + z) ** 3

    def f(u):
        s = a * np.exp(u)
        return tr.rho(np.exp(u), m, T) * np.exp(u) * s / ((s - M * M) ** 2 + (M * M * gam) ** 2)
    u0 = np.log(m)
    cuts = sorted({np.log(m * m / (80 * T)), np.log(80 * T)} | {u0 + sg * gam * 10.0 ** k for k in range(9)
                                                                   for sg in (-1, 1)})
    ref = sum(integrate.quad(f, p, q, limit=400, epsabs=0, epsrel=1e-10)[0] for p, q in zip(cuts[:-1], cuts[1:]))
    assert abs(st / (2 * np.pi * n) / ref - 1.0) < 1e-6
