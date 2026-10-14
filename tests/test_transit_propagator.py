"""
The near-field transit's propagators (2026-10-14): a path's propagator P, computed once per energy, potential and
path, gives the march of any density matrix as P rho P^dag; source_transit reuses them for every arriving state of
one solver point. Checked against the density-matrix march, on the transit's own geometry.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core as C  # noqa: E402
import dp_transit as T  # noqa: E402


def _mixing():
    th = np.radians([33.70916346871883, 8.559057632313843, 43.280093616242404, 212.0])
    U = C.construct_PMNS(*th)
    M_rot = np.ascontiguousarray(U @ np.diag([0.0, 7.49e-5, 2.513e-3]).astype(complex) @ U.conj().T)
    u = np.array([0.0316227766, 0.0316227766, 0.0], dtype=complex)
    return M_rot, np.ascontiguousarray(np.outer(u, u.conj()))


def _random_rho(rng):
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = A @ A.conj().T
    return np.ascontiguousarray(rho / np.trace(rho).real)


@pytest.mark.parametrize("E, v_over_gap, ca", [(1e15, 3.0, 0.9), (1e15, 0.3, -0.6), (1e14, 30.0, 0.4)])
def test_path_propagator_is_the_march(E, v_over_gap, ca):
    """P rho P^dag equals march_path(rho) for random density matrices (to rounding), and P is unitary."""
    M_rot, S = _mixing()
    gap = 2.513e-3 / (2 * E)
    kappa = 2 * E
    Cn = v_over_gap * gap / 2e-3
    m, D = 1e-22, C.d_sun_eV_inv
    t_max = min(300 * D, 3e5 / gap / (1 + v_over_gap))
    tab = T.build_frame_table(M_rot, S, dtheta=0.002)
    args = (tab["sig"], tab["nu"], tab["W"], tab["K"], tab["Q"], tab["Mp"], tab["s"], tab["floor"], T._GLX8, T._GLW8)
    P = np.empty((3, 3), dtype=complex)
    T.march_path_propagator(E, kappa, 0.0, Cn, m, D, ca, t_max, *args, P)
    assert np.max(np.abs(P @ P.conj().T - np.eye(3))) < 1e-12
    rng = np.random.default_rng(7)
    for _ in range(5):
        rho = _random_rho(rng)
        out = np.empty((3, 3), dtype=complex)
        T.march_path(E, kappa, 0.0, Cn, m, D, ca, t_max, rho, *args, out)
        assert np.max(np.abs(P @ rho @ P.conj().T - out)) < 1e-12


def test_source_transit_reuses_propagators():
    """source_transit through the Sun: with the propagators (default) equal to the density-matrix march for the
    states the solver passes -- populations of the arriving eigenstates, here those of the vacuum basis (n_bg = 0) --
    to rounding (1e-12), for several of them, served from one cache entry; a table without a cache key (built
    directly) takes the march. States with coherences agree to 1e-7: over these 20-pc paths the dynamic phases reach
    ~1e8 rad at 10 TeV, which double precision holds to ~5e-8 rad in either method (the march rounds their
    differences, the propagator keeps each phase), far below anything averaged over an energy bin."""
    M_rot, S = _mixing()
    E = np.geomspace(1e13, 3e16, 40)
    tab = T.frame_table(M_rot, S, 0.002)
    C_sun = 1e-20
    coef_abs = 1e2              # V = |Coef| C = 1e-18 eV: comparable to the vacuum term at ~1 PeV
    ca, w = T.sun_paths(4)
    args = (tab, coef_abs, 0.0, C_sun, 1e-24, C.d_sun_eV_inv, ca, w)
    vals, U = np.linalg.eigh(M_rot)
    rng = np.random.default_rng(3)
    T._NEARFIELD_CACHE.clear()
    for k in range(4):
        if k < 3:
            pops = rng.dirichlet(np.ones(3))
            rho = np.ascontiguousarray(np.stack([U @ np.diag(pops).astype(complex) @ U.conj().T for _ in E]))
            tol = 1e-12
        else:
            rho = np.ascontiguousarray(np.stack([_random_rho(rng) for _ in E]))
            tol = 1e-7
        fast, n1 = T.source_transit(E, rho, *args)
        T.NEARFIELD_REUSE = False
        try:
            slow, n2 = T.source_transit(E, rho, *args)
        finally:
            T.NEARFIELD_REUSE = True
        assert n1 == n2 > 0
        assert np.max(np.abs(fast - slow)) < tol, (k, np.max(np.abs(fast - slow)))
        assert len(T._NEARFIELD_CACHE) == 1
    plain = T.build_frame_table(M_rot, S, dtheta=0.002)
    assert "_key" not in plain
    a, _ = T.source_transit(E, rho, plain, *args[1:])
    b, _ = T.source_transit(E, rho, *args)
    assert np.max(np.abs(a - b)) < 1e-7 and len(T._NEARFIELD_CACHE) == 1
