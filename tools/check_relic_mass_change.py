"""Item 1 check: exact 2->2 kinematics and Dirac traces for nu(k) + nu_r(p) -> nu(k') + nu_s(p') by t-channel vector
exchange, left-handed currents at both vertices."""
import numpy as np, mpmath as mp
mp.mp.dps = 60
# --- (a) kinematics: the beam's energy loss at small |t|, exact, against (m_s^2 - m_r^2 + |t|)/(2 p^-)
def omega_exact(E, p3, mr, ms, t, phi):
    """Solve p.(k - k') = (ms^2 - mr^2 - t)/2 with k' at angle th: t = -2 E E' (1 - cos th)."""
    E, mr, ms, t = mp.mpf(E), mp.mpf(mr), mp.mpf(ms), mp.mpf(t)
    px, py, pz = [mp.mpf(x) for x in p3]; Er = mp.sqrt(px**2 + py**2 + pz**2 + mr**2)
    rhs = (ms**2 - mr**2 - t) / 2
    def f(Ep):
        c = 1 + t / (2 * E * Ep); s = mp.sqrt(max(0, 1 - c * c))
        kp = (Ep, Ep * s * mp.cos(phi), Ep * s * mp.sin(phi), Ep * c)
        q = (E - kp[0], -kp[1], -kp[2], E - kp[3])
        return Er * q[0] - px * q[1] - py * q[2] - pz * q[3] - rhs
    return E - mp.findroot(f, E * (1 - mp.mpf(10) ** -12))
rng = np.random.default_rng(0)
E = 1e14; t = -1e-24
worst = 0
for mr, ms in ((8.6e-3, 0.05), (1e-6, 0.05), (0.05, 8.6e-3), (1e-6, 8.6e-3)):
    for _ in range(4):
        p3 = rng.normal(size=3) * 5e-4
        Er = np.sqrt(p3 @ p3 + mr * mr); pminus = Er - p3[2]
        w = omega_exact(E, p3, mr, ms, t, rng.uniform(0, 6.28))
        pred = (ms**2 - mr**2 - t) / (2 * pminus)
        worst = max(worst, abs(float(w) / pred - 1))
print(f"(a) omega exact vs (m_s^2-m_r^2+|t|)/(2p^-): worst relative difference {worst:.1e}")
# --- (b) rate-weighted cost over relic directions: <v_rel omega> = (m_s^2-m_r^2)/2 <1/E_r>, v_rel = p^-/E_r
mr, ms = 1e-6, 0.05
for p in (1e-5, 5e-4, 1e-2):
    c = np.linspace(-1, 1, 200001); Er = np.sqrt(p * p + mr * mr)
    pm = Er - p * c
    lhs = np.trapz((pm / Er) * (ms**2 - mr**2) / (2 * pm), c) / 2
    print(f"(b) |p| {p:.0e}: <v_rel omega> / [(m_s^2-m_r^2)/(2 E_r)] = {lhs / ((ms**2 - mr**2) / (2 * Er)):.6f}")
# --- (c) Dirac traces: sum |M|^2 for r -> s against r -> r at the same small t (massless beam, LL x LL)
g0 = np.diag([1, -1, -1, -1]).astype(complex)
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]]); I2 = np.eye(2); Z2 = np.zeros((2, 2))
gam = [np.block([[Z2, I2], [I2, Z2]])] + [np.block([[Z2, s], [-s, Z2]]) for s in (sx, sy, sz)]
g5 = np.block([[-I2, Z2], [Z2, I2]]); PL = (np.eye(4) - g5) / 2
eta = np.diag([1., -1, -1, -1])
sl = lambda v: sum(eta[m, m] * v[m] * gam[m] for m in range(4))
def msq(k, kp, p, pp, mr, ms):
    L = np.array([[np.trace(sl(kp) @ gam[m] @ PL @ sl(k) @ gam[n] @ PL) for n in range(4)] for m in range(4)])
    H = np.array([[np.trace((sl(pp) + ms * np.eye(4)) @ gam[m] @ PL @ (sl(p) + mr * np.eye(4)) @ gam[n] @ PL) for n in range(4)] for m in range(4)])
    return np.real(np.einsum("mn,mn,m,n->", L, H, np.diag(eta), np.diag(eta)))
def final(k, p, mr, ms, th):
    """k' at angle th in the x-z plane with p' on shell (massless beam, relic p): solve for E'."""
    from scipy.optimize import brentq
    def f(Ep):
        kp = np.array([Ep, Ep * np.sin(th), 0, Ep * np.cos(th)]); pp = k + p - kp
        return pp[0]**2 - pp[1:] @ pp[1:] - ms * ms
    Ep = brentq(f, 1e-9 * k[0], k[0] * (1 + 1e-3)); kp = np.array([Ep, Ep * np.sin(th), 0, Ep * np.cos(th)])
    return kp, k + p - kp
E = 1e3; k = np.array([E, 0, 0, E])
for mr, ms in ((8.6e-3, 0.05), (0.05, 8.6e-3)):
    p = np.array([mr, 0, 0, 0.0])
    for th in (1e-6, 1e-4):
        kp, pp = final(k, p, mr, ms, th); kq, pq = final(k, p, mr, mr, th)
        t1 = (k - kp) @ eta @ (k - kp); t0 = (k - kq) @ eta @ (k - kq)
        print(f"(c) m_r {mr} -> m_s {ms}, theta {th:.0e}: |M|^2(r->s)/|M|^2(r->r) = {msq(k, kp, p, pp, mr, ms) / msq(k, kq, p, pq, mr, mr):.6f}"
              f"  (t {t1:.2e} vs {t0:.2e}; trace vs 16 (k.p)(k'.p'): {msq(k, kp, p, pp, mr, ms) / (16 * (k @ eta @ p) * (kp @ eta @ pp)):.6f})")
