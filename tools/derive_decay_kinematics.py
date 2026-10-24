"""
derive_decay_kinematics.py -- the energy and helicity of the daughters of nu_i -> nu_j + Z_D in flight, with every mass
kept (2026-10-24; the review's finding 1: the massless-daughter kernels ignored the mediator's and the daughter's
masses).

A parent of energy E >> m_i has a definite helicity h (Majorana: h = -1 the 'neutrino', +1 the 'antineutrino'); so
does the daughter, h'. For a boost along +z to infinity, helicity is light-front helicity: the spin vector of a
momentum k is s = k/m - m n / (n.k), n = (1, 0, 0, -1) (n.k = k^+), the same in every frame boosted along z. The rate
for (h, h') is then a covariant trace, evaluated in the parent's rest frame (parent spin s = h z-hat):
    |M|^2_{h h'} = sum_pol Tr[(k' + m_j)(1 + h' g5 s'_LF)/2 Gamma^mu (p + m_i)(1 + h g5 s_LF)/2 Gammabar^nu] P_mu nu,
Gamma^mu = gamma^mu (G P_L - G* P_R) (the Majorana vertex, G = G'_ji), P = -g + q q / M^2; the daughter's energy share is
its light-front fraction z = k'^+ / p^+ = (E* + p* cos th*) / m_i, in [z-, z+] = [(E* - p*)/m_i, (E* + p*)/m_i], and
dGamma/dz = (p*/(16 pi m_i^2)) |M|^2 (2 / (z+ - z-)) (the angle's measure; dGamma/dcos = p*/(32 pi m_i^2) |M|^2 summed
over the vector's polarizations).

Checks: the four (h, h') rates add up to dp_decay.rest_rates; h' = h ('kept') and h' = -h ('flipped') reproduce the
massless kernels (dp_decay's docstring) for m_j, M -> 0; dGamma/dz is a polynomial of degree <= 2 in z on [z-, z+]
(fitted at many nodes, residual printed). Run: python tools/derive_decay_kinematics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)
Z2 = np.zeros((2, 2))
gam = [np.block([[Z2, I2], [I2, Z2]])] + [np.block([[Z2, s], [-s, Z2]]) for s in (sx, sy, sz)]
g5 = np.block([[-I2, Z2], [Z2, I2]])
PL = (np.eye(4) - g5) / 2
PR = (np.eye(4) + g5) / 2
eta = np.diag([1.0, -1, -1, -1])
I4 = np.eye(4)
NV = np.array([1.0, 0.0, 0.0, -1.0])


def sl(v):
    return sum(eta[m, m] * v[m] * gam[m] for m in range(4))


def dot(a, b):
    return a @ eta @ b


def s_lf(k, m):
    """Light-front helicity spin vector (helicity +1) of momentum k, mass m > 0."""
    return k / m - m * NV / dot(NV, k)


def rho(k, m, h):
    """u u-bar for helicity h (light-front), mass m (m = 0: the chiral projector, helicity = chirality)."""
    if m == 0.0:
        return (PR if h > 0 else PL) @ sl(k)
    return (sl(k) + m * I4) @ (I4 + h * g5 @ sl(s_lf(k, m))) / 2


def msq(p, k, mi, mj, M, h, hp, G):
    q = p - k
    V = G * PL - np.conj(G) * PR
    Gam = [gam[m] @ V for m in range(4)]
    Gbar = [gam[0] @ X.conj().T @ gam[0] for X in Gam]
    Rk = rho(k, mj, hp)
    Rp = rho(p, mi, h)
    T = np.array([[np.trace(Rk @ Gam[m] @ Rp @ Gbar[n]) for n in range(4)] for m in range(4)])
    P = -eta + np.outer(eta @ q, eta @ q) / M ** 2
    return float(np.real(np.einsum("mn,mn->", T, P)))


def spectra(mi, mj, M, G=1.0 + 0j, nodes=41):
    """(zm, zp, {(h, hp): (z, dGamma/dz)}) in the parent's rest frame; the parent along +z in the lab."""
    pf = np.sqrt(max((mi ** 2 - (mj + M) ** 2) * (mi ** 2 - (mj - M) ** 2), 0.0)) / (2 * mi)
    Ej = np.sqrt(pf ** 2 + mj ** 2)
    zm, zp = (Ej - pf) / mi, (Ej + pf) / mi
    p = np.array([mi, 0.0, 0.0, 0.0])
    out = {}
    c = np.cos(np.linspace(np.pi, 0.0, nodes))
    for h in (-1, 1):
        for hp in (-1, 1):
            w = []
            for ci in c:
                si = np.sqrt(max(1 - ci * ci, 0.0))
                k = np.array([Ej, pf * si, 0.0, pf * ci])
                w.append(msq(p, k, mi, mj, M, h, hp, G))
            z = (Ej + pf * c) / mi
            dGdz = pf / (16 * np.pi * mi ** 2) * np.array(w) * 2.0 / (zp - zm)
            out[(h, hp)] = (z, dGdz)
    return zm, zp, out


def main():
    import dp_decay
    np.set_printoptions(precision=5, suppress=False)
    cases = [(0.05, 0.0, 1e-4), (0.05, 1e-6, 1e-4), (0.05, 8.6e-3, 1e-3), (0.05, 8.6e-3, 0.03), (0.112, 0.1, 1e-3),
             (0.112, 0.1, 0.008), (0.1004, 0.1, 1e-4), (0.0087, 1e-6, 0.005)]
    for G in (1.0 + 0j, np.exp(0.7j)):
        for mi, mj, M in cases:
            zm, zp, sp = spectra(mi, mj, M, G)
            tot = {}
            line = f"G {G:.2f} m_i {mi} m_j {mj} M {M:g}: z in [{zm:.4f}, {zp:.4f}]"
            for (h, hp), (z, f) in sp.items():
                rate = np.trapz(f, z) if hasattr(np, "trapz") else np.trapezoid(f, z)
                cf = np.polyfit(z, f, 2)
                res = np.max(np.abs(np.polyval(cf, z) - f)) / max(np.max(np.abs(f)), 1e-300)
                tot[(h, hp)] = rate
                line += f"\n   h {h:+d} -> h' {hp:+d}: rate {rate:.6e}, quadratic fit residual {res:.1e}"
            # the formula, with G' a 3x3 holding G at (j, i)
            Gm = np.zeros((3, 3), complex)
            Gm[1, 0] = G
            Gm[0, 1] = np.conj(G)
            Gam_f, _ = dp_decay.rest_rates(Gm, np.eye(3), np.array([mj, mi, 10.0]), M)   # index 1 the parent, 0 the daughter
            Gf = Gam_f[1, 0]
            for h in (-1, 1):
                line += f"\n   parent h {h:+d}: total {tot[(h, -1)] + tot[(h, 1)]:.6e} / formula {Gf:.6e} = " \
                        f"{(tot[(h, -1)] + tot[(h, 1)]) / Gf:.6f}; kept share {tot[(h, h)] / (tot[(h, -1)] + tot[(h, 1)]):.4f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
