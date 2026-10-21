"""
amplitudes_bruteforce.py -- tree-level amplitudes of neutrino-neutrino scattering evaluated by brute
force (explicit Dirac matrices, massless helicity spinors and, for vector bosons, explicit
polarization vectors), as an independent check of the closed forms in src/dp_si.py. Used by
tests/test_si.py; no physics code imports it.

Conventions: chiral (Weyl) representation, metric (+,-,-,-); the centre-of-momentum frame with
the incident neutrino along +z; Feynman rules with the fermion-permutation signs written out per
process. Each function returns |M|^2 summed over the final helicities (or polarizations) for
fixed initial helicities.
"""
import numpy as np

I2 = np.eye(2)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
Z2 = np.zeros((2, 2))
G0 = np.block([[Z2, I2], [I2, Z2]]).astype(complex)
GI = [np.block([[Z2, s], [-s, Z2]]) for s in (SX, SY, SZ)]
GAMMA = [G0] + GI
G5 = np.block([[-I2, Z2], [Z2, I2]]).astype(complex)
PL = 0.5 * (np.eye(4) - G5)
PR = 0.5 * (np.eye(4) + G5)
METRIC = np.diag([1.0, -1.0, -1.0, -1.0])


def mom(E, theta, phi=0.0, m=0.0):
    """Four-momentum of energy E along (theta, phi), mass m."""
    p = np.sqrt(max(E * E - m * m, 0.0))
    return np.array([E, p * np.sin(theta) * np.cos(phi), p * np.sin(theta) * np.sin(phi), p * np.cos(theta)])


def slash(p):
    return sum(METRIC[m, m] * p[m] * GAMMA[m] for m in range(4))


def _xi(theta, phi, h):
    """Two-component helicity spinor along (theta, phi), sigma.n xi = 2h xi."""
    if h > 0:
        return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    return np.array([-np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)])


def u(p, h):
    """Massless particle spinor of helicity h (+-1/2)."""
    E = p[0]
    th = np.arccos(np.clip(p[3] / E, -1, 1))
    ph = np.arctan2(p[2], p[1])
    x = _xi(th, ph, h)
    z = np.zeros(2)
    return np.sqrt(2 * E) * (np.concatenate([x, z]) if h < 0 else np.concatenate([z, x]))


def v(p, h):
    """Massless antiparticle spinor of helicity h: the opposite chirality slot."""
    E = p[0]
    th = np.arccos(np.clip(p[3] / E, -1, 1))
    ph = np.arctan2(p[2], p[1])
    x = _xi(th, ph, -h)
    z = np.zeros(2)
    return np.sqrt(2 * E) * (np.concatenate([x, z]) if h > 0 else np.concatenate([z, -x]))


def bar(sp):
    return sp.conj() @ G0


def pols(k, M):
    """The three polarization vectors of a massive vector of momentum k (upper index)."""
    E = k[0]
    kk = np.linalg.norm(k[1:])
    n = k[1:] / kk
    th = np.arccos(np.clip(n[2], -1, 1))
    ph = np.arctan2(n[1], n[0])
    e1 = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    e2 = np.array([-np.sin(ph), np.cos(ph), 0.0])
    out = [np.concatenate([[0.0], e1]).astype(complex), np.concatenate([[0.0], e2]).astype(complex),
           np.concatenate([[kk / M], E * n / M]).astype(complex)]
    return out


def cm_kinematics(s, cos3, m3=0.0, m4=0.0):
    """p1 (+z), p2 (-z) massless incoming; p3 at angle acos(cos3), p4 opposite; final masses m3, m4."""
    rs = np.sqrt(s)
    p1 = mom(rs / 2, 0.0)
    p2 = mom(rs / 2, np.pi)
    E3 = (s + m3 ** 2 - m4 ** 2) / (2 * rs)
    E4 = (s + m4 ** 2 - m3 ** 2) / (2 * rs)
    th = np.arccos(cos3)
    p3 = mom(E3, th, 0.3, m3)
    p4 = mom(E4, np.pi - th, 0.3 + np.pi, m4)
    return p1, p2, p3, p4


def mdot(a, b):
    return a[0] * b[0] - a[1:] @ b[1:]


# --- vector mediator, left-handed currents: vertex g_xy nubar_x gamma^mu P_L nu_y V_mu -------------

def _cur(a, b):
    """(bar(a) gamma^mu P_L b) with a lower Lorentz index."""
    return np.array([METRIC[m, m] * (bar(a) @ GAMMA[m] @ PL @ b) for m in range(4)])


def _cur_up(a, b):
    return np.array([bar(a) @ GAMMA[m] @ PL @ b for m in range(4)])


def vector_nunubar(s, cos3, M, As, At):
    """nu_L(p1) nubar_R(p2) -> nu_L(p3) nubar_R(p4) (s and t channels): |M|^2 for the
    helicities (-,+) -> (-,+) (the only ones a left-handed current connects), coupling products
    A_s (s channel) and A_t (t channel), no width."""
    p1, p2, p3, p4 = cm_kinematics(s, cos3)
    t = mdot(p1 - p3, p1 - p3)
    Ms = (_cur_up(u(p3, -0.5), v(p4, +0.5)) @ _cur(v(p2, +0.5), u(p1, -0.5))) * As / (s - M * M)
    Mt = (_cur_up(u(p3, -0.5), u(p1, -0.5)) @ _cur(v(p2, +0.5), v(p4, +0.5))) * At / (t - M * M)
    return abs(Ms - Mt) ** 2, t, -s - t


def vector_nunu(s, cos3, M, At, Au):
    """nu_L(p1) nu_L(p2) -> nu_L(p3) nu_L(p4) (t and u channels), |M|^2, couplings A_t, A_u."""
    p1, p2, p3, p4 = cm_kinematics(s, cos3)
    t = mdot(p1 - p3, p1 - p3)
    uu = mdot(p1 - p4, p1 - p4)
    Mt = (_cur_up(u(p3, -0.5), u(p1, -0.5)) @ _cur(u(p4, -0.5), u(p2, -0.5))) * At / (t - M * M)
    Mu = (_cur_up(u(p4, -0.5), u(p1, -0.5)) @ _cur(u(p3, -0.5), u(p2, -0.5))) * Au / (uu - M * M)
    return abs(Mt - Mu) ** 2, t, uu


def vector_pair(s, cos3, M, C=1.0):
    """nu_L(p1) nubar_R(p2) -> V(k1) V(k2) through neutrino exchange in the t and u channels,
    |M|^2 summed over the six polarization pairs; t = (p1 - k1)^2."""
    p1, p2, k1, k2 = cm_kinematics(s, cos3, M, M)
    t = mdot(p1 - k1, p1 - k1)
    uu = mdot(p1 - k2, p1 - k2)
    tot = 0.0
    ub = bar(v(p2, +0.5))
    u1 = u(p1, -0.5)
    for e1 in pols(k1, M):
        for e2 in pols(k2, M):
            s1 = slash(np.conj(e1))
            s2 = slash(np.conj(e2))
            amp = ub @ (s2 @ PL @ slash(p1 - k1) @ s1 / t + s1 @ PL @ slash(p1 - k2) @ s2 / uu) @ PL @ u1
            tot += abs(C * amp) ** 2
    return tot, t, uu


# --- scalar mediator --------------------------------------------------------------------------------
# Majorana: L = -1/2 phi (lambda_ij chi_i chi_j + h.c.) = -1/2 phi Psibar_i (lambda_ij P_L + lambda*_ij P_R) Psi_j
# for Majorana spinors; Feynman rules of Denner, Eck, Hahn & Kublbeck (fermion flow): vertex
# -i (lambda_ij P_L + lambda*_ij P_R) for either flow direction (lambda symmetric), v = C ubar^T, and
# the sign of each diagram from the permutation of the external spinors.
CCONJ = 1j * GAMMA[2] @ G0


def vM(p, h):
    """Antiparticle-flow spinor of a Majorana fermion of helicity h: C ubar^T."""
    return CCONJ @ bar(u(p, h))


def _gam(lam, i, j):
    return lam[i, j] * PL + np.conj(lam[i, j]) * PR


def majorana_22(s, cos3, M, Gam, lam, i, j, k, l, h1, h2, h3, h4):
    """nu_i(p1,h1) nu_j(p2,h2) -> nu_k(p3,h3) nu_l(p4,h4), Majorana neutrinos, scalar exchange in the
    s, t and u channels (s with the width): the complex amplitude."""
    p1, p2, p3, p4 = cm_kinematics(s, cos3)
    t = mdot(p1 - p3, p1 - p3)
    uu = mdot(p1 - p4, p1 - p4)
    U1, U2, U3, U4 = u(p1, h1), u(p2, h2), u(p3, h3), u(p4, h4)
    V2, V4 = vM(p2, h2), vM(p4, h4)
    Ms = (bar(V2) @ _gam(lam, j, i) @ U1) * (bar(U3) @ _gam(lam, k, l) @ V4) / (s - M * M + 1j * M * Gam)
    Mt = (bar(U3) @ _gam(lam, k, i) @ U1) * (bar(U4) @ _gam(lam, l, j) @ U2) / (t - M * M)
    Mu = (bar(U4) @ _gam(lam, l, i) @ U1) * (bar(U3) @ _gam(lam, k, j) @ U2) / (uu - M * M)
    # spinor orders (2,1,3,4), (3,1,4,2), (4,1,3,2): parities odd, odd, even
    return -Ms - Mt + Mu, t, uu


def majorana_dsigma_dt(s, cos3, M, Gam, lam, i, j, k, l, h1=-0.5):
    """dsigma/dt for an incident nu_i of helicity h1 on nu_j, summed over the relic's helicity and the
    final ones, 1/2 for k = l: the convention of 2107.13568's Eq. (A12)."""
    tot = 0.0
    for h2 in (-0.5, 0.5):
        for h3 in (-0.5, 0.5):
            for h4 in (-0.5, 0.5):
                amp, t, uu = majorana_22(s, cos3, M, Gam, lam, i, j, k, l, h1, h2, h3, h4)
                tot += abs(amp) ** 2
    sym = 0.5 if k == l else 1.0
    return sym * tot / (16 * np.pi * s * s), t


def majorana_phiphi(s, cos3, M, lam, i, j, h1=-0.5):
    """nu_i(p1,h1) nu_j(p2) -> phi(kA) phi(kB) through nu_k exchange (t and u), summed over k
    coherently, over the relic helicity; dsigma/dt with the 1/2 for identical scalars."""
    p1, p2, kA, kB = cm_kinematics(s, cos3, M, M)
    t = mdot(p1 - kA, p1 - kA)
    uu = mdot(p1 - kB, p1 - kB)
    tot = 0.0
    U1 = u(p1, h1)
    for h2 in (-0.5, 0.5):
        V2 = vM(p2, h2)
        amp = 0.0
        for kk in range(3):
            amp += bar(V2) @ (_gam(lam, j, kk) @ slash(p1 - kA) @ _gam(lam, kk, i) / t
                              + _gam(lam, j, kk) @ slash(p1 - kB) @ _gam(lam, kk, i) / uu) @ U1
        tot += abs(amp) ** 2
    return 0.5 * tot / (16 * np.pi * s * s), t


# Dirac: L = -phi h_ij nubar_i nu_j (h Hermitian, no projector); vertex -i h_xy for the outgoing (or
# annihilated-antiparticle) index x and the incoming (or created-antiparticle) index y.
def dirac_nunu(s, cos3, M, h, a, r, k, l, h1, h2, h3, h4):
    """nu_a(p1) nu_r(p2) -> nu_k(p3) nu_l(p4) (t and u channels), complex amplitude."""
    p1, p2, p3, p4 = cm_kinematics(s, cos3)
    t = mdot(p1 - p3, p1 - p3)
    uu = mdot(p1 - p4, p1 - p4)
    Mt = (bar(u(p3, h3)) @ u(p1, h1)) * h[k, a] * (bar(u(p4, h4)) @ u(p2, h2)) * h[l, r] / (t - M * M)
    Mu = (bar(u(p4, h4)) @ u(p1, h1)) * h[l, a] * (bar(u(p3, h3)) @ u(p2, h2)) * h[k, r] / (uu - M * M)
    return Mt - Mu, t, uu


def dirac_nunubar(s, cos3, M, Gam, h, a, r, k, l, h1, h2, h3, h4):
    """nu_a(p1) nubar_r(p2) -> nu_k(p3) nubar_l(p4) (s and t channels), complex amplitude."""
    p1, p2, p3, p4 = cm_kinematics(s, cos3)
    t = mdot(p1 - p3, p1 - p3)
    Ms = (bar(u(p3, h3)) @ v(p4, h4)) * h[k, l] * (bar(v(p2, h2)) @ u(p1, h1)) * h[r, a] / (s - M * M + 1j * M * Gam)
    Mt = (bar(u(p3, h3)) @ u(p1, h1)) * h[k, a] * (bar(v(p2, h2)) @ v(p4, h4)) * h[r, l] / (t - M * M)
    return Ms - Mt, t, -s - t


def visible(kind, hel):
    """Left-handed neutrinos and right-handed antineutrinos interact through the weak force."""
    return (kind == "nu" and hel < 0) or (kind == "nubar" and hel > 0)


def vector_pair_lab_longitudinal(s, cos3, M):
    """nu_L(p1) nubar_R(p2) -> V(k1) V(k2): |M|^2 with V1 longitudinal in the RELIC rest frame
    (reached from the CM frame by a boost along the collision axis), summed over V2's polarizations.
    For a V that is ultrarelativistic there, its longitudinal state is the light-cone polarization
    eps_0 = k/M - M q/(k.q), q along the relic's momentum p2 (q null: unchanged by that boost)."""
    p1, p2, k1, k2 = cm_kinematics(s, cos3, M, M)
    t = mdot(p1 - k1, p1 - k1)
    uu = mdot(p1 - k2, p1 - k2)
    e0 = (k1 / M - M * p2 / mdot(k1, p2)).astype(complex)
    ub = bar(v(p2, +0.5))
    u1 = u(p1, -0.5)
    tot = 0.0
    s1 = slash(np.conj(e0))
    for e2 in pols(k2, M):
        s2 = slash(np.conj(e2))
        amp = ub @ (s2 @ PL @ slash(p1 - k1) @ s1 / t + s1 @ PL @ slash(p1 - k2) @ s2 / uu) @ PL @ u1
        tot += abs(amp) ** 2
    return tot, t, uu, mdot(k1, p2)


# --- massive, polarized Dirac relics (2026-10-21) -----------------------------------------------------------------
# A moving relic of mass m keeps its helicity; the amplitudes see its left- and right-chiral components. These
# helpers evaluate the Dirac processes with an arbitrary relic spinor (massive, any spin) and arbitrary momenta, for
# tests/test_si.py's check that, averaged over the final azimuth about the collision axis, a relic of spin
# polarization P along the beam (rest frame) is exactly the mixture (1 + P)/2 left-chiral + (1 - P)/2 right-chiral
# of massless relics, up to m^2/s.
SIG4 = [I2.astype(complex), SX, SY, SZ]


def _sqrt_psig(p, m, bar_=False):
    """sqrt(p.sigma) (bar_: sqrt(p.sigmabar)) of a momentum p of mass m, by its eigenvalues E -+ |p| (the small one as
    m^2 / (E + |p|), without cancellation)."""
    E, k = p[0], np.linalg.norm(p[1:])
    m2 = m * m
    if k == 0.0:
        return np.sqrt(E) * I2.astype(complex)
    n = p[1:] / k
    proj = 0.5 * (I2 + (n[0] * SX + n[1] * SY + n[2] * SZ))         # onto sigma.n = +1
    hi, lo = np.sqrt(E + k), np.sqrt(m2 / (E + k))
    return (lo * proj + hi * (I2 - proj)) if not bar_ else (hi * proj + lo * (I2 - proj))


def u_massive(p, m, xi):
    """Massive particle spinor (sqrt(p.sigma) xi, sqrt(p.sigmabar) xi); xi its rest-frame spin state (reached by the
    pure boost from rest)."""
    return np.concatenate([_sqrt_psig(p, m) @ xi, _sqrt_psig(p, m, True) @ xi])


def v_massive(p, m, eta):
    """Massive antiparticle spinor (sqrt(p.sigma) eta, -sqrt(p.sigmabar) eta); eta is the rest-frame state opposite to
    the physical spin (as v(p, h) above)."""
    return np.concatenate([_sqrt_psig(p, m) @ eta, -(_sqrt_psig(p, m, True) @ eta)])


def relic_cm(kind, s, m, theta_s, phi_s):
    """The relic of mass m in the CM frame of a massless neutrino along +z: (p1, p2, spinor), spin along
    (theta_s, phi_s) in its rest frame (reached by a boost along z, so the beam is +z there too); kind 'nu' or
    'nubar'."""
    rs = np.sqrt(s)
    k = (s - m * m) / (2.0 * rs)
    p1 = np.array([k, 0.0, 0.0, k])
    p2 = np.array([(s + m * m) / (2.0 * rs), 0.0, 0.0, -k])
    if kind == "nu":
        return p1, p2, u_massive(p2, m, _xi(theta_s, phi_s, +1))
    return p1, p2, v_massive(p2, m, _xi(theta_s, phi_s, -1))


def finals_cm(p1, p2, cth, phi, mf=0.0):
    """Final momenta (p3, p4) at CM angle acos(cth) to p1 and azimuth phi, both of mass mf, in the frame of p1, p2
    (boosted from their CM)."""
    P = p1 + p2
    rs = np.sqrt(mdot(P, P))
    b = P[1:] / P[0]

    def boost(beta, x):
        b2 = beta @ beta
        if b2 == 0.0:
            return x.copy()
        g = 1.0 / np.sqrt(1.0 - b2)
        bp = beta @ x[1:]
        return np.concatenate([[g * (x[0] - bp)], x[1:] + ((g - 1.0) * bp / b2 - g * x[0]) * beta])
    q1 = boost(b, p1)
    n = q1[1:] / np.linalg.norm(q1[1:])
    e1 = np.cross(n, [0.3, 0.5, 0.8])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    kf = np.sqrt(rs * rs / 4.0 - mf * mf)
    d = cth * n + np.sqrt(1.0 - cth * cth) * (np.cos(phi) * e1 + np.sin(phi) * e2)
    p3 = boost(-b, np.concatenate([[rs / 2.0], kf * d]))
    return p3, P - p3


def dirac_amp2(proc, p1, p2, p3, p4, w2, c, M, Gam):
    """|M|^2 summed over the final helicities (polarizations) for an incident nu_L(p1) on a relic of spinor w2 (any
    mass and spin): proc 'V_nnbar' (vector, s + t), 'V_nn' (t + u), 'V_pair' (nu nubar -> V V; p3, p4 the vectors),
    'S_nn', 'S_nnbar' (scalar, Dirac), 'S_phiphi' (nu nubar -> phi phi); c: two complex coupling products."""
    u1 = u(p1, -0.5)
    s = mdot(p1 + p2, p1 + p2)
    t = mdot(p1 - p3, p1 - p3)
    uu = mdot(p1 - p4, p1 - p4)
    HS = (-0.5, 0.5)
    tot = 0.0
    if proc == "V_pair":
        ub = bar(w2)
        for e1 in pols(p3, M):
            for e2 in pols(p4, M):
                s1, s2 = slash(np.conj(e1)), slash(np.conj(e2))
                a = ub @ (s2 @ PL @ slash(p1 - p3) @ s1 / t + s1 @ PL @ slash(p1 - p4) @ s2 / uu) @ PL @ u1
                tot += abs(a) ** 2
        return tot
    if proc == "S_phiphi":
        return abs(bar(w2) @ (c[0] * slash(p1 - p3) / t + c[1] * slash(p1 - p4) / uu) @ u1) ** 2
    for h3 in HS:
        for h4 in HS:
            U3 = u(p3, h3)
            if proc == "V_nnbar":
                V4 = v(p4, h4)
                a = (_cur_up(U3, V4) @ _cur(w2, u1) * c[0] / (s - M * M + 1j * M * Gam)
                     - _cur_up(U3, u1) @ _cur(w2, V4) * c[1] / (t - M * M))
            elif proc == "V_nn":
                U4 = u(p4, h4)
                a = (_cur_up(U3, u1) @ _cur(U4, w2) * c[0] / (t - M * M)
                     - _cur_up(U4, u1) @ _cur(U3, w2) * c[1] / (uu - M * M))
            elif proc == "S_nn":
                U4 = u(p4, h4)
                a = ((bar(U3) @ u1) * (bar(U4) @ w2) * c[0] / (t - M * M)
                     - (bar(U4) @ u1) * (bar(U3) @ w2) * c[1] / (uu - M * M))
            else:                                                  # S_nnbar
                V4 = v(p4, h4)
                a = ((bar(U3) @ V4) * (bar(w2) @ u1) * c[0] / (s - M * M + 1j * M * Gam)
                     - (bar(U3) @ u1) * (bar(w2) @ V4) * c[1] / (t - M * M))
            tot += abs(a) ** 2
    return tot
