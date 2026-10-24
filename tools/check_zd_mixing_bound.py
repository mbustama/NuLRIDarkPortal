"""
check_zd_mixing_bound.py -- the Z - Z_D mass mixing of Jana & Swain (arXiv:2607.04441, Eq. 6) against the Z_D mass
(notes Sec. 'Relic mass changes and neutrino decay below 0.05 eV', 2026-10-24). Fields: H (the SM-like doublet, no dark
charge, vev v_H), phi (a doublet with dark charge Q, vev v_phi), a dark singlet (charge Q_S, vev w). The (Z0, Z_D0)
mass matrix from their kinetic terms; for random models with a dark boson below m_Z/1000, the dark-like eigenstate's
Z0 component (its coupling to the Z current over g/c_W, i.e. eps') against (v_phi/v_H) m_ZD/m_Z, in 50-digit arithmetic.
Result (2026-10-24, 20,000 models): max ratio 1.006 -- eps' <= (v_phi/v_H) m_ZD/m_Z.
"""
import numpy as np
import mpmath as mp

mp.mp.dps = 50
rng = np.random.default_rng(3)
gz = mp.mpf("0.74")
worst = 0.0
for _ in range(20000):
    v1 = mp.mpf(float(rng.uniform(1, 246)))
    v2 = mp.sqrt(mp.mpf(246) ** 2 - v1 ** 2)
    gD = mp.mpf(10) ** float(rng.uniform(-14, 0))
    Q = mp.mpf(float(rng.choice([1, 2, 0.5])))
    QS = mp.mpf(float(rng.choice([1, 2])))
    w = mp.mpf(10) ** float(rng.uniform(-14, 3))
    M = mp.matrix([[gz ** 2 * (v1 ** 2 + v2 ** 2) / 4, -gz * gD * Q * v2 ** 2 / 2],
                   [-gz * gD * Q * v2 ** 2 / 2, gD ** 2 * (Q ** 2 * v2 ** 2 + QS ** 2 * w ** 2)]])
    E, V = mp.eighe(M)
    k = 0 if abs(V[1, 0]) > abs(V[1, 1]) else 1
    md, mZ = mp.sqrt(E[k]), mp.sqrt(E[1 - k])
    if md / mZ > mp.mpf("1e-3"):
        continue
    worst = max(worst, float(abs(V[0, k]) * mZ / md / (v2 / v1)))
print(f"max of eps' m_Z/m_ZD / (v_phi/v_H) over models with m_ZD < m_Z/1000: {worst:.4f}")
