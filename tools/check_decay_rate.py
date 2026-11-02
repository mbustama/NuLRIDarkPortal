"""Item 2 check: nu_i -> nu_j + V (coupling g nu_j-bar gamma^mu P_L nu_i V_mu), rest-frame rate from the trace with the
massive polarization sum, against g^2 m^3/(32 pi M^2) (1 - x)^2 (1 + 2x) (m_j = 0); and the daughter's angular
distribution about the parent's spin, split by the daughter's chirality, transverse and longitudinal."""
import numpy as np
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]]); I2 = np.eye(2); Z2 = np.zeros((2, 2))
gam = [np.block([[Z2, I2], [I2, Z2]])] + [np.block([[Z2, s], [-s, Z2]]) for s in (sx, sy, sz)]
g5 = np.block([[-I2, Z2], [Z2, I2]]); PL = (np.eye(4) - g5) / 2; PR = (np.eye(4) + g5) / 2
eta = np.diag([1., -1, -1, -1]); sl = lambda v: sum(eta[m, m] * v[m] * gam[m] for m in range(4)); I4 = np.eye(4)
def amp2(pi, pj, q, mi, mj, M, spin=None, proj=I4, pol="all", V=None):
    """|M|^2 summed over the daughter's spin, with the parent's spin s (or averaged), the daughter projected on proj (P_L or P_R)."""
    Pi = (sl(pi) + mi * I4) @ ((I4 + g5 @ sl(spin)) / 2 if spin is not None else I4 / 2)
    V = PL if V is None else V          # the vertex's chirality: gamma^mu P_L (Dirac, left-handed) or the Majorana P_R piece
    T = np.array([[np.trace(proj @ (sl(pj) + mj * I4) @ gam[m] @ V @ Pi @ gam[n] @ V) for n in range(4)] for m in range(4)])
    qlo = eta @ q
    if pol == "all":   P = -eta + np.outer(qlo, qlo) / M**2
    elif pol == "long":
        # the longitudinal polarization along q in the rest frame: eps = (|q|, E_q qhat)/M
        e = np.array([np.linalg.norm(q[1:]), *(q[0] * q[1:] / np.linalg.norm(q[1:]))]) / M; el = eta @ e; P = np.outer(el, el)
    else:
        e = np.array([np.linalg.norm(q[1:]), *(q[0] * q[1:] / np.linalg.norm(q[1:]))]) / M; el = eta @ e
        P = -eta + np.outer(qlo, qlo) / M**2 - np.outer(el, el)
    return np.real(np.einsum("mn,mn->", T, P))
mi, mj = 0.05, 0.0
for M in (1e-2, 1e-3, 1e-4):
    x = M**2 / mi**2; pf = (mi**2 - M**2) / (2 * mi)
    pi = np.array([mi, 0, 0, 0]); pj = np.array([pf, 0, 0, pf]); q = pi - pj
    G = pf / (8 * np.pi * mi**2) * amp2(pi, pj, q, mi, mj, M)   # isotropic for an unpolarized parent
    print(f"M {M:.0e}: Gamma/[m^3/(32 pi M^2)(1-x)^2(1+2x)] = {G / (mi**3 / (32 * np.pi * M**2) * (1 - x)**2 * (1 + 2 * x)):.6f}")
# angular distribution about the parent's spin (s along +z): daughter at angle th; 1 + alpha cos th
M = 1e-4; pf = (mi**2 - M**2) / (2 * mi); pi = np.array([mi, 0, 0, 0]); s = np.array([0, 0, 0, 1.0])
for pol in ("trans", "long"):
    for name, proj in (("left-chiral daughter", PL), ("right-chiral daughter", PR)):
        v = []
        for c in (1.0, -1.0, 0.0):
            th = np.arccos(c); pj = np.array([pf, pf * np.sin(th), 0, pf * c]); q = pi - pj
            v.append(amp2(pi, pj, q, mi, mj, M, spin=s, proj=proj, pol=pol))
        tot = v[0] + v[1]
        print(f"{pol:5s} {name}: share {(v[0]+v[1])/2:.3e}, alpha = {(v[0]-v[1])/tot if tot else 0:+.3f}, linear check {v[2]/((v[0]+v[1])/2) if tot else 0:.3f}")

# the Majorana vertex's P_R piece (2026-10-24): its rate and its daughters' chirality and angular distribution
for M in (1e-2, 1e-4):
    x = M**2 / mi**2; pf = (mi**2 - M**2) / (2 * mi)
    pi = np.array([mi, 0, 0, 0]); pj = np.array([pf, 0, 0, pf]); q = pi - pj
    print(f"M {M:.0e}: P_R rate / P_L rate = {amp2(pi, pj, q, mi, mj, M, V=PR) / amp2(pi, pj, q, mi, mj, M):.6f}")
M = 1e-4; pf = (mi**2 - M**2) / (2 * mi); pi = np.array([mi, 0, 0, 0]); s = np.array([0, 0, 0, 1.0])
for pol in ("trans", "long"):
    for name, proj in (("left-chiral daughter", PL), ("right-chiral daughter", PR)):
        v = []
        for c in (1.0, -1.0, 0.0):
            th = np.arccos(c); pj = np.array([pf, pf * np.sin(th), 0, pf * c]); q = pi - pj
            v.append(amp2(pi, pj, q, mi, mj, M, spin=s, proj=proj, pol=pol, V=PR))
        tot = v[0] + v[1]
        print(f"P_R vertex, {pol:5s} {name}: share {(v[0]+v[1])/2:.3e}, alpha = {(v[0]-v[1])/tot if tot else 0:+.3f}, linear check {v[2]/((v[0]+v[1])/2) if tot else 0:.3f}")

# massive daughter, the full Majorana vertex gamma^mu (G P_L - G* P_R) (2026-10-24): the trace against
# Gamma = |p| / (16 pi m_i^2) [2|G|^2 (m_i^2 + m_j^2 - 2M^2 + (m_i^2 - m_j^2)^2/M^2) + 12 Re(G^2) m_i m_j]
for G in (1.0 + 0j, np.exp(0.6j), 1j):
    for mi, mj, M in ((0.05, 8.6e-3, 1e-3), (0.05, 0.03, 1e-4), (0.0087, 1e-3, 1e-5)):
        pf = np.sqrt((mi**2 - (mj + M)**2) * (mi**2 - (mj - M)**2)) / (2 * mi)
        Ej = np.sqrt(pf**2 + mj**2)
        pi = np.array([mi, 0, 0, 0]); pj = np.array([Ej, 0, 0, pf]); q = pi - pj
        Vtx = G * PL - np.conj(G) * PR
        T = np.array([[np.trace((sl(pj) + mj * I4) @ gam[m] @ Vtx @ ((sl(pi) + mi * I4) / 2) @ gam[0] @ (gam[n] @ Vtx).conj().T @ gam[0]) for n in range(4)] for m in range(4)])
        Pm = -eta + np.outer(eta @ q, eta @ q) / M**2
        Gam_tr = pf / (8 * np.pi * mi**2) * np.real(np.einsum("mn,mn->", T, Pm))
        Gam_f = pf / (16 * np.pi * mi**2) * (2 * abs(G)**2 * (mi**2 + mj**2 - 2 * M**2 + (mi**2 - mj**2)**2 / M**2) + 12 * np.real(G**2) * mi * mj)
        print(f"G {G:.2f} m_i {mi} m_j {mj} M {M:.0e}: trace / formula = {Gam_tr / Gam_f:.6f}")
