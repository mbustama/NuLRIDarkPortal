"""The vector's longitudinal-mode scatterings (2026-10-25; notes, "Secret interactions below 1 keV"): the q q / M^2 part
of the massive propagator between currents the neutrino masses do not conserve, by explicit spinors
(tests/amplitudes_bruteforce.py's), and dp_si_light's Goldstone factors."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import amplitudes_bruteforce as B  # noqa: E402
import dp_decay  # noqa: E402
import dp_si_light as L  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

U = construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0))


def _cm(s, m1, m2, th, ph):
    rs = np.sqrt(s)
    k = np.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / (2 * rs)
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return np.concatenate([[np.hypot(k, m1)], k * n]), np.concatenate([[np.hypot(k, m2)], -k * n])


def _u(p, m, h):
    k = np.linalg.norm(p[1:])
    n = p[1:] / k
    return B.u_massive(p, m, B._xi(np.arccos(np.clip(n[2], -1, 1)), np.arctan2(n[1], n[0]), h))


def _cur(a, b):
    return np.array([B.bar(a) @ B.GAMMA[m] @ B.PL @ b for m in range(4)])


def test_longitudinal_exchange_is_a_yukawa_exchange_without_interference():
    """Dirac neutrinos, vertex G' nubar gamma P_L nu, t channel nu_a nu_r -> nu_k nu_l with every mass kept: q.J is
    the Yukawa bilinear m_a ubar P_R u - m_k ubar P_L u (Dirac equation); the q q / M^2 part adds to the -g part with
    no interference; the -g part with massive spinors is dp_si's massless 4 s^2 / (t - M^2)^2."""
    p1, p2 = _cm(1.0, 0.05, 0.0087, 0.0, 0.0)
    p3, p4 = _cm(1.0, 0.0087, 0.05, 1.1, 0.3)
    q = p1 - p3
    for h1 in (-1, 1):
        for h3 in (-1, 1):
            a, b = _u(p3, 0.0087, h3), _u(p1, 0.05, h1)
            lhs = (B.METRIC @ q) @ _cur(a, b)
            rhs = 0.05 * (B.bar(a) @ B.PR @ b) - 0.0087 * (B.bar(a) @ B.PL @ b)
            assert abs(lhs - rhs) < 1e-10 * abs(lhs)
    s, M = 1e12, 1e-6
    for ma, mr, mk, ml in ((0.05, 1e-6, 0.0087, 0.05), (0.0087, 0.05, 1e-6, 0.0087)):
        p1, p2 = _cm(s, ma, mr, 0.0, 0.0)
        for th in (0.3, 2.0):
            p3, p4 = _cm(s, mk, ml, th, 0.4)
            q = p1 - p3
            t = B.mdot(q, q)
            T2 = L2 = F2 = 0.0
            u1 = _u(p1, ma, -1)
            for h2 in (-1, 1):
                u2 = _u(p2, mr, h2)
                for h3 in (-1, 1):
                    u3 = _u(p3, mk, h3)
                    J1 = _cur(u3, u1)
                    y1 = ma * (B.bar(u3) @ B.PR @ u1) - mk * (B.bar(u3) @ B.PL @ u1)
                    for h4 in (-1, 1):
                        u4 = _u(p4, ml, h4)
                        J2 = _cur(u4, u2)
                        y2 = -(mr * (B.bar(u4) @ B.PR @ u2) - ml * (B.bar(u4) @ B.PL @ u2))
                        T = (J1 @ (B.METRIC @ J2)) / (t - M * M)
                        Lg = -y1 * y2 / (M * M * (t - M * M))
                        T2 += abs(T) ** 2
                        L2 += abs(Lg) ** 2
                        F2 += abs(T + Lg) ** 2
            assert abs(F2 - T2 - L2) < 1e-10 * F2
            assert abs(T2 / 2 / (4 * s * s / (t - M * M) ** 2 / 2) - 1.0) < 1e-6
            assert L2 > 1e4 * T2                                   # M << m: the longitudinal part dominates


def test_goldstone_normalization_is_the_decays_longitudinal_rate():
    """A scalar with the Majorana Goldstone couplings Y decays at |Y_ji|^2 m_i / 16 pi (M, m_j -> 0): the vector's
    Majorana rate there (dp_decay), its longitudinal part."""
    m = np.array([1e-9, 8.68e-3, 0.0501])
    G = 1e-12 * np.array([[0.0, 0, 0], [0, 1.0, 0], [0, 0, 0]])
    M = 1e-9
    Y = L.goldstone_yukawa(G, U, m, M)
    Gam, _ = dp_decay.rest_rates(G, U, m, M)
    for i, j in ((2, 0), (1, 0)):
        assert np.isclose(abs(Y[j, i]) ** 2 * m[i] / (16 * np.pi), Gam[i, j], rtol=1e-5)


def test_goldstone_factors_gate_and_bounds():
    """No factor below GOLDSTONE_Y_MIN; above, the flux with the Goldstone channels below the one without at the
    lowest energies (absorption), and the solver's switch only on the lepton-number planes below 1 keV."""
    import dp_si
    import dark_portal_solver as dps
    m = np.array([1e-6, np.sqrt(1e-12 + 7.537e-5), np.sqrt(1e-12 + 2.511e-3)])
    E = np.geomspace(4.57e12, 3e16, 60)
    H0 = 1.4377e-33
    G = 1e-13 * np.diag([1.0, -1.0, 0.0])
    assert L.goldstone_factors(E, G * 1e-3, U, m, 1e-12, 1.676e-4, np.ones(3), H0, 0.315, 0.685, 4.0, 2.5, 0.0,
                               None) is None
    Gam, _ = dp_decay.rest_rates(G, U, m, 1e-12)
    dec = (Gam, dp_decay.daughter_spectra(G, U, m, 1e-12))
    eta = dp_decay.relic_composition(Gam, m, (np.abs(U) ** 2).T @ np.ones(3), 1.676e-4, H0, 0.315, 0.685)
    R, Rt = L.goldstone_factors(E, G, U, m, 1e-12, 1.676e-4, eta, H0, 0.315, 0.685, 4.0, 2.5, 0.0, dec)
    assert R.shape == (60, 3) and np.all(np.isfinite(R)) and Rt[0] < 0.99
    assert dps._goldstone_on({"model": "L_e-L_mu", "m_Zd_eV": 1e-12})
    assert not dps._goldstone_on({"model": "dark_portal", "m_Zd_eV": 1e-12})
    assert not dps._goldstone_on({"model": "L_e-L_tau", "m_Zd_eV": 1e5})
    assert dp_si is not None
