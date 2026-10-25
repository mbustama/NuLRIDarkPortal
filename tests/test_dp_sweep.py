"""dp_sweep: the dark-portal phase's step with the decays, every process of a bin solved together (2026-10-25); the
decays between the eigenstates of the medium (dp_medium)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_decay as dd  # noqa: E402
import dp_medium as dm  # noqa: E402
import dp_sweep  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402
from dp_cosmo import cascade_kernel_weights, cross_kernel_weights, step_cosmological_rates, _soft_jumps  # noqa: E402

U = np.asarray(construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0)))
S_EMU = np.diag([1.0, -1.0, 0.0]).astype(complex)


class Setup:
    def __init__(self, N=40, lo=1e13, hi=1e16, ml=0.1, M=0.008, g=1e-9, decay=True, c=0.0, seed=1):
        rng = np.random.default_rng(seed)
        self.E = np.geomspace(lo, hi, N)
        self.m = np.sqrt(np.array([0.0, 7.5e-5, 2.5e-3]) + ml * ml)
        self.G = g * S_EMU if decay else 0.0 * S_EMU
        self.Gam = dd.rest_rates(self.G, U, self.m, M)[0]
        M_rot = U @ np.diag(self.m ** 2 - self.m[0] ** 2) @ U.conj().T
        self.lad = dm.Ladder(self.E, self.G, U, self.m, M, [(M_rot, S_EMU), (np.conj(M_rot), -S_EMU)], c)
        self.c = c
        self.use_dec = decay
        self.F_nu = rng.uniform(0.5, 1.0, (N, 3)) * (self.E[:, None] / 1e14) ** -2
        self.F_nb = rng.uniform(0.5, 1.0, (N, 3)) * (self.E[:, None] / 1e14) ** -2
        self.G_nu = np.zeros((N, 3))
        self.G_nb = np.zeros((N, 3))
        self.brp_nu = np.full((N, 3), 0.5 / 3.0)
        self.brp_nb = np.full((N, 3), 0.5 / 3.0)
        self.Pi = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        self.Phi_0 = 0.0
        self.J = np.zeros((2, 3, 3))
        self.lam = np.zeros((2, 3))
        self.Gamma_unit = 0.0

    def run(self, dt, jump_dt=None, decay_dt=None, z=0.0, n=1, rho=None, cache=None):
        """n steps of dt (the same rates each), on copies; returns (F_nu, F_nb, R_nu, R_nb). rho: the sources' flavor
        composition (default equal); cache: (cache, have) for step_full's shared exponentials."""
        N = len(self.E)
        W3, pref3, ss = cascade_kernel_weights(self.E)
        W1, W2, c1, c2, sx = cross_kernel_weights(self.E)
        Fn, Fb = self.F_nu.copy(), self.F_nb.copy()
        Rn, Rb = np.zeros(N), np.zeros(N)
        z0 = np.zeros((2, 3))
        tab = (self.J, z0, z0, self.lam, 0.0, 1.0, 0.0)
        rho = np.full(3, 1.0 / 3.0) if rho is None else np.asarray(rho, dtype=np.float64)
        cc = (np.zeros((1, 4, 6, 6)), np.zeros(1, dtype=np.bool_), False) if cache is None else (*cache, True)
        for _ in range(n):
            dp_sweep.step_full((1.0 + z) ** -1.5, dt, 1.0 + z, dt if jump_dt is None else jump_dt,
                               dt if decay_dt is None else decay_dt, self.E, Fn, Fb, 0.5, 0.5, rho,
                               self.Pi, self.Pi, self.G_nu, self.G_nb, self.brp_nu, self.brp_nb, 2.5, self.Phi_0, W3,
                               pref3, ss, W1, W2, c1, c2, sx, Rn, Rb, np.ones(N), self.Gamma_unit > 0.0,
                               self.Gamma_unit, *tab, *tab, self.lad.arrays, self.lad.coord(self.c), self.use_dec, True,
                               self.lad.diag, *self.lad.buffers(), np.zeros((1, 2, 3)), False, *cc)
        return Fn, Fb, Rn, Rb

    def number(self, Fn, Fb):
        h = np.sqrt(self.E[1] / self.E[0])
        dE = self.E * (h - 1.0 / h)
        return float(((Fn + Fb).sum(axis=1) * dE).sum())


def test_without_decays_and_jumps_it_is_step_cosmological_rates():
    """Absorption, its cascade and the injection alone: the step before (whose own-bin secondaries are a constant
    source at the step's mean rate, exact to O(self x (G dt)^2))."""
    s = Setup(decay=False)
    rng = np.random.default_rng(2)
    dt = 1e30
    s.G_nu = rng.uniform(0.0, 0.02, (len(s.E), 3)) / dt
    s.G_nb = rng.uniform(0.0, 0.02, (len(s.E), 3)) / dt
    s.Phi_0 = 0.3 * 1e14 ** 2.5 / dt
    Fn, Fb, Rn, Rb = s.run(dt)
    W3, pref3, ss = cascade_kernel_weights(s.E)
    W1, W2, c1, c2, sx = cross_kernel_weights(s.E)
    an, ab = s.F_nu.copy(), s.F_nb.copy()
    rn, rb = np.zeros(len(s.E)), np.zeros(len(s.E))
    step_cosmological_rates(0.0, dt, 1.0, s.E, an, ab, 0.5, 0.5, np.full(3, 1.0 / 3.0), s.Pi, s.Pi, s.G_nu, s.G_nb,
                            s.brp_nu, s.brp_nb, 2.5, s.Phi_0, W3, pref3, ss, W1, W2, c1, c2, sx, rn, rb,
                            np.ones(len(s.E)))
    assert np.max(np.abs(Fn / an - 1.0)) < 1e-5 and np.max(np.abs(Fb / ab - 1.0)) < 1e-5
    assert np.max(np.abs(Rn / rn - 1.0)) < 1e-4 and np.max(np.abs(Rb / rb - 1.0)) < 1e-4


def test_jumps_alone_are_soft_jumps():
    """The soft scatterings alone: dp_cosmo._soft_jumps' exact step (moderate), and their equilibrium where the
    conversions per step are stiff (scaled down here, as dp_si). To dp_si._phi_functions' accuracy (5e-9 against
    scipy's expm at 0.7 per step)."""
    for rate_dt, tol in ((0.7, 3e-8), (3e12, 3e-8)):
        s = Setup(decay=False)
        s.J[:] = np.array([[0.0, 1.0, 0.4], [1.0, 0.0, 0.2], [0.4, 0.2, 0.0]])
        s.lam[:, 1] = 1.0
        s.lam[:, 2] = 3.0
        s.Gamma_unit = rate_dt / 1e30
        Fn, Fb, _, _ = s.run(1e30)
        for F0, F1 in ((s.F_nu, Fn), (s.F_nb, Fb)):
            ref = F0.copy()
            _soft_jumps(s.E, 1.0, s.J, np.zeros((2, 3)), np.zeros((2, 3)), s.lam, 0.0, 1.0, 0.0, s.Gamma_unit, 1e30,
                        ref, np.zeros((len(s.E), 3)), False, False)
            assert np.max(np.abs(F1 / ref - 1.0)) < tol


def test_decays_keep_the_number():
    """Each decay keeps one neutrino: with daughters that stay on the grid (nearly degenerate states, ml 0.1 eV), the
    total number is kept to rounding for any decay per step, without and with a potential."""
    for c in (0.0, 1e-17):
        for f in (1e-3, 1.0, 30.0, 1e12):
            s = Setup(c=c)
            s.F_nu[: len(s.E) // 2] = 0.0
            s.F_nb[: len(s.E) // 2] = 0.0
            lab = float(np.max(s.Gam.sum(axis=1) * s.m)) / s.E[-1]
            Fn, Fb, _, _ = s.run(f / lab)
            assert abs(s.number(Fn, Fb) / s.number(s.F_nu, s.F_nb) - 1.0) < 1e-10
            assert np.all(Fn >= 0.0) and np.all(Fb >= 0.0)


def test_cycling_is_independent_of_the_step():
    """Conversions and decays cycling (a stable state converted into a decaying one, again and again), with the
    injection, 10 decays and 5 conversions per step at the top, 1e4 decays at the bottom: the top bin (nothing from
    above) exactly the same in one step as in 16; from the populations' steady state (the cosmological flux's regime
    where the cycles are fast) every bin stays there whatever the step (the sources from above constant over a step,
    as dp_si, are then exact)."""
    s = Setup(ml=1e-6, M=1e-4)
    s.J[:] = np.array([[0.0, 1.0, 0.4], [1.0, 0.0, 0.2], [0.4, 0.2, 0.0]])
    s.lam[:, 1] = 1.0
    s.lam[:, 2] = 3.0
    lab = float(np.max(s.Gam.sum(axis=1) * s.m)) / s.E[-1]
    dt = 10.0 / lab
    s.Gamma_unit = 5.0 / dt
    s.Phi_0 = 0.3 * 1e14 ** 2.5 / dt
    one = s.run(dt)
    many = s.run(dt / 16, n=16)
    for a, b in zip(one[:2], many[:2]):
        assert np.max(np.abs(a[-1] / b[-1] - 1.0)) < 3e-8
    s.F_nu, s.F_nb = s.run(dt / 16, n=400)[:2]
    one = s.run(dt)
    many = s.run(dt / 16, n=16)
    for a, b in zip(one[:2], many[:2]):
        assert np.max(np.abs(a / b - 1.0)) < 1e-6


def test_top_bin_is_the_exact_six_state_solution():
    """The jumps (Zeno-damped by the in-medium decay widths), the decays and their own-bin daughters (into the other
    kind too) together, at z = 1 in a potential: the top bin (nothing from above) after one step against expm of its
    6 x 6 rate matrix assembled here independently (the rates per unit time: the jumps Gamma_unit J om^2 / (om^2 + D^2),
    D = Gamma_unit x the tables' width term + (w_a + w_b) / 2 with w the physical widths E Gamma / (E (1+z)); the
    decays from the ladder's plans at the bin's rungs)."""
    from scipy.linalg import expm
    s = Setup(ml=1e-6, M=1e-4, c=1e-17)
    s.J[:] = np.array([[0.0, 1.0, 0.4], [1.0, 0.0, 0.2], [0.4, 0.2, 0.0]])
    s.lam[:, 1] = 1.0
    s.lam[:, 2] = 3.0
    N = len(s.E)
    z = 1.0
    lad = s.lad
    Eb = s.E[-1]
    x0 = lad.coord(s.c)
    r0, w = dm.rung_of_bin(x0, N - 1)
    tot = (1 - w) * lad.arrays[4][r0] + w * lad.arrays[4][r0 + 1]
    own = (1 - w) * lad.arrays[5][r0] + w * lad.arrays[5][r0 + 1]
    pref_t = 1.0 / (16 * np.pi * lad.Ms * Eb)                  # per unit time over (1+z): the plans' E Gamma / E
    lam_t = tot.reshape(2, 3, 2, 3).sum(axis=(2, 3)) * pref_t   # E Gamma / E_obs
    width = lam_t / (1 + z)                                     # the physical width
    dt = 0.7 / float(lam_t.max())
    s.Gamma_unit = 2.0 / dt
    rng = np.random.default_rng(4)
    s.F_nu[:] = 0.0
    s.F_nb[:] = 0.0
    s.F_nu[-1] = rng.uniform(0.2, 1.0, 3)
    s.F_nb[-1] = rng.uniform(0.2, 1.0, 3)
    Fn, Fb, _, _ = s.run(dt, z=z)
    M6 = np.zeros((6, 6))
    for t in range(2):
        for a in range(3):
            for b in range(a + 1, 3):
                om = (s.lam[0, a] - s.lam[0, b]) / (2 * Eb * (1 + z))
                D = 0.5 * (width[t, a] + width[t, b])
                r = s.Gamma_unit * s.J[0, a, b] * om * om / (om * om + D * D)
                M6[3 * t + a, 3 * t + b] += r
                M6[3 * t + b, 3 * t + a] += r
                M6[3 * t + a, 3 * t + a] -= r
                M6[3 * t + b, 3 * t + b] -= r
        for a in range(3):
            M6[3 * t + a, 3 * t + a] -= lam_t[t, a]
            for td in range(2):
                for b in range(3):
                    ch = ((t * 3 + a) * 2 + (0 if td == t else 1)) * 3 + b
                    M6[3 * td + b, 3 * t + a] += own[ch] * pref_t
    x = expm(M6 * dt) @ np.concatenate([s.F_nu[-1], s.F_nb[-1]])
    got = np.concatenate([Fn[-1], Fb[-1]])
    assert np.any(own > 0.0) and np.max(np.abs(got - x)) < 1e-10 * np.max(np.abs(x))


def test_cached_exponentials_are_bit_for_bit():
    """step_full's per-bin exponentials shared by the runs of one point (2026-10-25): a run with other sources that
    reuses the first run's cache gives exactly what it gives without it; a bin whose matrix differs is recomputed."""
    s = Setup(ml=1e-6, M=1e-4)
    s.J[:] = np.array([[0.0, 1.0, 0.4], [1.0, 0.0, 0.2], [0.4, 0.2, 0.0]])
    s.lam[:, 1] = 1.0
    s.lam[:, 2] = 3.0
    dt = 1e30
    s.Gamma_unit = 3.0 / dt
    s.Phi_0 = 0.3 * 1e14 ** 2.5 / dt
    N = len(s.E)
    cache = (np.zeros((N, 4, 6, 6)), np.zeros(N, dtype=np.bool_))
    s.run(dt, rho=[1.0, 0.0, 0.0], cache=cache)
    assert cache[1].all()
    for rho in ([0.0, 1.0, 0.0], [0.2, 0.3, 0.5]):
        a = s.run(dt, rho=rho, cache=cache)
        b = s.run(dt, rho=rho)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
    s.Gamma_unit *= 1.5                               # other rates: every bin recomputed, the cache overwritten
    a = s.run(dt, rho=[0.0, 1.0, 0.0], cache=cache)
    b = s.run(dt, rho=[0.0, 1.0, 0.0])
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
