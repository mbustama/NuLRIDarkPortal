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
    i0, s, nw = dp_si._thermal_weights(m, T, 0.37, DELTA, W)
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
