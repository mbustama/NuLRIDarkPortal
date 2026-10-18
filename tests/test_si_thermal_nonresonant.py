"""
The MeV transport's non-resonant terms over the relics' thermal momenta (src/dp_si.py, _thermal_nonresonant_terms,
2026-10-18): the 1/s term and the pair cross section, at rest until then (a lightest neutrino of 1e-6 eV made the 1/s
term ~400 times too large). They must reduce to the rest values for slow relics, reach the exact massless limit
<1/p> = (pi^2 / 12) / (3 zeta(3) / 2) / T, and equal a brute-force double quadrature over momentum and angle.
"""
import os
import sys

import numpy as np
from scipy import integrate

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_si  # noqa: E402

T0, H0, OM, OL = 1.68e-4, 1.5e-33, 0.3, 0.7
PX, PW = dp_si._GL12_X, dp_si._GL12_W
ZETA3 = 1.2020569031595942


def _terms(E, m, M, zp1, half=1e-6, prof=1, thermal=True):
    """J (N, 3, 4) for one step around zp1 (relic state 0 of mass m, the others off), and the step's weight
    dL n0 (1+z)^3 / H at its midpoint."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    L_lo, L_hi = np.log(zp1) - half, np.log(zp1) + half
    masses = np.array([m, 0.0, 0.0])
    n0 = np.array([56.0, 0.0, 0.0])
    J = np.zeros((len(E), 3, 4))
    P = 10
    Nq, Lq, Dq = np.zeros((len(E), 3, P)), np.zeros((len(E), 3, P)), np.zeros((len(E), 3, P))
    dp_si._step_integrals(E, np.exp(L_lo), np.exp(L_hi), zp1, masses, n0, M * M, 0.0, H0, OM, OL, False, prof,
                          1.01, J, Nq, Lq, Dq)
    if thermal:
        dp_si._thermal_nonresonant_terms(E, L_lo, L_hi, masses, T0, np.ones(3), n0, M * M, H0, OM, OL, prof, J,
                                         dp_si._FDX32, dp_si._FDW32, PX, PW)
    w = 2.0 * half * n0[0] * zp1 ** 3 / (H0 * np.sqrt(OM * zp1 ** 3 + OL))
    return J, w


def test_slow_relics_keep_the_rest_values():
    E = np.array([1e13, 1e14, 1e15])
    for prof in (0, 1):
        rest, _ = _terms(E, 0.2, 3e4, 1.5, prof=prof, thermal=False)
        th, _ = _terms(E, 0.2, 3e4, 1.5, prof=prof)
        for k in (2, 3):
            assert np.allclose(th[:, 0, k], rest[:, 0, k], rtol=2e-4, atol=0.0), (prof, k, th[:, 0, k], rest[:, 0, k])


def test_one_over_s_reaches_the_massless_limit():
    zp1, E = 2.0, 1e14
    th, w = _terms(E, 1e-9, 3e4, zp1)
    T = T0 * zp1
    exact = (np.pi ** 2 / 12.0) / (1.5 * ZETA3) / T / (2.0 * E * zp1)
    assert abs(th[0, 0, 2] / w / exact - 1.0) < 1e-6


def test_pair_term_equals_brute_force():
    """<K(s) (1 - v c)> over the Fermi-Dirac momenta and the angle, by nested adaptive quadrature split where the
    channel opens (unsplit, the reference itself was 0.5% off near the threshold)."""
    zp1 = 2.0
    T = T0 * zp1
    for prof in (0, 1):
        for E, m, M in ((1e14, 1e-5, 3e4), (1e14, 1e-5, 2e5), (3e13, 3e-3, 1e5)):
            th, w = _terms(E, m, M, zp1, prof=prof)
            Ez, M2 = E * zp1, M * M

            def angular(p):
                Ep = np.sqrt(p * p + m * m)
                v = p / Ep

                def f(c):
                    s = m * m + 2.0 * Ez * Ep * (1.0 - v * c)
                    return dp_si._pair_kinematic(prof, s, M2) * (1.0 - v * c) if s > 4.0 * M2 else 0.0
                c_th = (1.0 - (4.0 * M2 - m * m) / (2.0 * Ez * Ep)) / v       # s > 4 M^2 for c < c_th
                if c_th <= -1.0:
                    return 0.0
                return 0.5 * integrate.quad(f, -1.0, min(c_th, 1.0), epsrel=1e-12, limit=200)[0]
            X = (4.0 * M2 - m * m) / (2.0 * Ez)                 # the channel opens at E_p + p = X
            x_th = (X * X - m * m) / (2.0 * X) / T if X > m else 0.0
            g = lambda x: x * x / (np.exp(x) + 1.0) * angular(x * T)
            num = sum(integrate.quad(g, a, b, epsrel=1e-11, limit=400)[0]
                      for a, b in ((x_th, x_th + 0.5), (x_th + 0.5, x_th + 3.0), (x_th + 3.0, x_th + 45.0)))
            ref = num / (1.5 * ZETA3)
            assert ref > 0.0
            assert abs(th[0, 0, 3] / w / ref - 1.0) < 2e-4, (prof, E, m, M, th[0, 0, 3] / w, ref)
