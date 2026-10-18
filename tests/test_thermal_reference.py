"""
The thermal reference (tools/thermal_reference.py, 2026-10-18): the collision-weighted distribution of
sigma = s / 2E' (normalization, <1/sigma> = <1/E_p>), its quadrature against a brute-force double integral over the
relic's momentum and angle (light mediator, the pair threshold, the resonance), and the extended table's interpolation
against the spectra computed at the same mu.
"""
import os
import sys

import numpy as np
import pytest
from scipy import integrate

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import dp_si  # noqa: E402
import thermal_reference as tr  # noqa: E402

ZETA3 = 1.2020569031595942
GLX, GLW = tr._gl(0.0, 1.0, 24)
H0, OM, OL = 1.5e-33, 0.3, 0.7


@pytest.mark.parametrize("m, T", [(1e-6, 1.7e-4), (1e-3, 3.4e-4), (0.05, 3.4e-4), (0.1, 1e-3)])
def test_distribution_normalized_with_the_inverse_energy_moment(m, T):
    sig = np.empty(1000)
    w = np.empty(1000)
    n = tr.sigma_nodes(m, T, np.empty(0), GLX, GLW, sig, w)
    assert abs(w[:n].sum() - 1.0) < 1e-6
    fd = lambda x, g: x * x / (np.exp(x) + 1.0) * g(x * T)
    inv = integrate.quad(fd, 0, 60, args=(lambda p: 1.0 / np.sqrt(p * p + m * m),), epsrel=1e-12, limit=200)[0]
    inv /= 1.5 * ZETA3
    assert abs((w[:n] / sig[:n]).sum() / inv - 1.0) < 1e-6


def _at_rest(E, zlo, zhi, zmid, sig, M, MG, q, prof=1):
    """J_jt[q] times the profile total of q, for one bin and a relic of mass sig at rest."""
    P = len(dp_si.VECTOR_JT)
    J = np.zeros((1, 3, 4))
    Nq, Lq, Dq = (np.zeros((1, 3, P)) for _ in range(3))
    dp_si._step_integrals(np.array([E]), zlo, zhi, zmid, np.array([sig, 0.0, 0.0]), np.array([56.0, 0.0, 0.0]),
                          M * M, MG, H0, OM, OL, False, prof, 1.05, J, Nq, Lq, Dq)
    return J[0, 0, dp_si.VECTOR_JT[q]] * Nq[0, 0, q]


@pytest.mark.slow
@pytest.mark.parametrize("E, m, M, q", [(1e14, 1e-6, 3.16e4, 6), (1e14, 1e-6, 3.16e4, 9), (3e13, 1e-4, 2e5, 9),
                                         (2e15, 1e-4, 4.22e6, 0), (2e15, 1e-4, 4.22e6, 4)])
def test_mixture_equals_brute_force(E, m, M, q):
    """One bin and step: the mixture's total for profile q against int dp dc of the same at-rest quantity at
    sigma = E_p (1 - v c), weighted by the Fermi-Dirac momenta and (1 - v c) / 2, split where sigma crosses the
    resonance and the pair threshold of the step."""
    zlo, zhi = 1.5, 1.52
    zmid = np.sqrt(zlo * zhi)
    T = 1.68e-4 * zmid
    MG = M * M * 1e-7
    P = len(dp_si.VECTOR_JT)
    oN, oL, oD = (np.zeros((1, P)) for _ in range(3))
    oR = np.zeros((1, P))
    B = np.zeros((4, 1, P))
    tr.mixture_step(np.array([E]), zlo, zhi, zmid, m, 56.0, T, M * M, MG, H0, OM, OL, False, 1, 1.05,
                    dp_si.VECTOR_JT, GLX, GLW, 0.0, 1.0, B, oN, oL, oD, oR, np.array([0, 1], dtype=np.int64))
    edges = [M * M / (2 * E * zhi), M * M / (2 * E * zlo), 4 * M * M / (2 * E * zhi), 4 * M * M / (2 * E * zlo)]

    def angular(x):
        p = x * T
        Ep = np.sqrt(p * p + m * m)
        v = p / Ep
        pts = sorted({min(max((1.0 - e / Ep) / v, -1.0), 1.0) for e in edges} - {-1.0, 1.0})
        f = lambda c: 0.5 * (1.0 - v * c) * _at_rest(E, zlo, zhi, zmid, Ep * (1.0 - v * c), M, MG, q)
        return integrate.quad(f, -1.0, 1.0, points=pts or None, epsrel=1e-10, limit=400)[0]

    num = integrate.quad(lambda x: x * x / (np.exp(x) + 1.0) * angular(x), 0.0, 44.0, epsrel=1e-9, limit=400)[0]
    ref = num / (1.5 * ZETA3)
    assert ref != 0.0
    # at the resonance (Gamma / M = 1e-7 here) both sides are converged to ~1e-5 only
    tol = 3e-5 if dp_si.VECTOR_JT[q] in (0, 1) else 1e-5
    assert abs(oN[0, q] / ref - 1.0) < tol, (oN[0, q], ref)


@pytest.mark.parametrize("mu", [1e-3, 0.05, 0.2, 2.0, 50.0])
def test_extended_table_interpolation(mu):
    """Rows interpolated between lattice nodes (4-point, half a production spacing) against the spectra computed at
    mu itself."""
    N, dlnE, n_u = 120, 0.0228, 4096
    dl = 0.5 * dlnE
    top = mu * np.exp(2.37 * dl)
    mus = top * np.exp(-np.arange(6) * dl)
    B, _ = dp_si.build_vector_table_tri(mus, np.full(6, N - 1, dtype=np.int64), dlnE, n_u)
    B = B.reshape(6, N, -1)
    ref, _ = dp_si.build_vector_table_tri(np.array([mu]), np.array([N - 1], dtype=np.int64), dlnE, n_u)
    out = np.empty((N, B.shape[2]))
    tr.interp_rows(np.log(mu), np.log(top), dl, B, N - 1, out)
    scale = np.max(np.abs(ref), axis=0)
    ok = scale > 0
    err = np.max(np.abs(out - ref)[:, ok] / scale[ok])
    # just below the pair threshold (mu -> 1/4) the pair spectrum varies as sqrt(1 - 4 mu): cubic in ln mu at half
    # the production spacing is good to ~1e-4 there, where the pair cross section itself vanishes
    assert err < (2e-4 if 0.15 < mu < 0.25 else 1e-5), err
