"""dp_medium: the decay into the mediator between the propagation eigenstates of a medium (2026-10-25;
docs/dev/DECAY_IN_MEDIUM_2026-10-25.md)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_decay as dd  # noqa: E402
import dp_medium as dm  # noqa: E402
import dp_sweep  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402
from dp_cosmo import cascade_kernel_weights, cross_kernel_weights  # noqa: E402

# NumPy 2 renamed trapz to trapezoid (CI runs NumPy 2, the pinned environment 1.26)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
U = np.asarray(construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0)))
S_EMU = np.diag([1.0, -1.0, 0.0]).astype(complex)


# explicit light-front spinors (the review's brute force, docs/dev/REVIEW_2026-10-25.md finding 9): Dirac
# representation-free chiral gammas, u(p) = (pslash + m) gamma^+ chi / norm, the mediator's polarizations summed as
# -J.J* + |q.J|^2 / M^2 with the in-medium q -- independent of the |A|^2 formulas the code uses
_I2, _Z2 = np.eye(2), np.zeros((2, 2))
_SX = np.array([[0, 1], [1, 0]], complex)
_SY = np.array([[0, -1j], [1j, 0]])
_SZ = np.diag([1.0, -1.0]).astype(complex)
_G0 = np.block([[_Z2, _I2], [_I2, _Z2]]).astype(complex)
_GS = [np.block([[_Z2, q], [-q, _Z2]]) for q in (_SX, _SY, _SZ)]
_G5 = np.block([[-_I2, _Z2], [_Z2, _I2]]).astype(complex)
_PL, _PR = (np.eye(4) - _G5) / 2, (np.eye(4) + _G5) / 2
_GAM = [_G0] + _GS
_METRIC = np.diag([1.0, -1.0, -1.0, -1.0])
_GPLUS = _G0 + _GS[2]
_CHI = {-1: np.array([0, 1, 0, 0], complex), +1: np.array([0, 0, 1, 0], complex)}


def _spinors(pp, kT, m, lam):
    """u (n, 4) for light-front momenta p+ = pp (n,), p_T = (kT, 0) (n,), mass m, helicity lam."""
    pm = (kT * kT + m * m) / pp
    p0, p3 = (pp + pm) / 2, (pp - pm) / 2
    sl = (_G0[None] * p0[:, None, None] - _GS[0][None] * kT[:, None, None] - _GS[2][None] * p3[:, None, None]
          + m * np.eye(4)[None])
    u = np.einsum('nij,jk,k->ni', sl, _GPLUS, _CHI[lam])
    nrm = np.real(np.einsum('ni,ij,jk,nk->n', u.conj(), _G0, _GPLUS, u)) / (2 * pp)
    return u / np.sqrt(nrm)[:, None]


def _spinor_amp2(c, cz, Gp, m, mu_a, mu_bz, z, M, lam_in, lam_out):
    """|A|^2 (n,) of a(p) -> b(z p) + Z_D at the daughter fractions z (n,): parent eigenvector c (3,) and effective
    mass squared mu_a, daughter eigenvectors cz (n, 3) and mu_bz (n,) at each z sigma, vertex
    gamma^mu (G'_ji P_L - G'*_ji P_R) between the mass states' spinors (p+ = 1)."""
    Ms = M * M
    kT2 = (1 - z) * (z * mu_a - mu_bz) - z * Ms
    ok = kT2 >= 0.0
    kT = np.sqrt(np.where(ok, kT2, 0.0))
    one = np.ones_like(z)
    P = np.array([(1 + mu_a) / 2, 0.0, 0.0, (1 - mu_a) / 2])
    pmd = (kT2 + mu_bz) / z
    Pd = np.stack([(z + pmd) / 2, kT, 0 * z, (z - pmd) / 2], axis=1)
    q = P[None] - Pd
    J = np.zeros((len(z), 4), complex)
    for i in range(3):
        ui = _spinors(one, 0 * z, m[i], lam_in)
        for j in range(3):
            uj = _spinors(z, kT, m[j], lam_out)
            V = Gp[j, i] * _PL - np.conj(Gp[j, i]) * _PR
            ub = np.einsum('ni,ij->nj', uj.conj(), _G0)
            Jij = np.stack([np.einsum('ni,ij,jk,nk->n', ub, g, V, ui) for g in _GAM], axis=1)
            J += (np.conj(cz[:, j]) * c[i])[:, None] * Jij
    JJ = np.real(np.einsum('ni,ij,nj->n', J.conj(), _METRIC, J))
    qJ = np.einsum('ni,ij,nj->n', q, _METRIC, J)
    return np.where(ok, -JJ + np.abs(qJ) ** 2 / Ms, 0.0)


def _formula_amp2(c, cz, Gs, m, mu_a, mu_bz, z, M, rel):
    """The module docstring's |A|^2 (the code's)."""
    D = np.diag(m).astype(complex)
    Ms = M * M
    kT2 = (1 - z) * (z * mu_a - mu_bz) - z * Ms
    if rel == 0:
        al = np.einsum('nj,jl,l->n', cz.conj(), Gs, c)
        be = np.einsum('nj,jl,l->n', cz.conj(), D @ Gs.conj() @ D, c)
        f = 2 * kT2 * (z + 1 / z) / (1 - z) ** 2 * np.abs(al) ** 2 + np.abs((z * Ms - kT2) / (1 - z) * al
                                                                         + (1 - z) * be) ** 2 / (Ms * z)
    else:
        al = np.einsum('nj,jl,l->n', cz.conj(), D @ Gs, c)
        be = np.einsum('nj,jl,l->n', cz.conj(), Gs.conj() @ D, c)
        f = 2 / z * np.abs(al + z * be) ** 2 + kT2 * np.abs(al + be) ** 2 / (Ms * z)
    return np.where(kT2 >= 0.0, f, 0.0)


# a coupling that does not commute with the texture S_EMU (the formulas hold for any S)
G_NC = np.array([[1.0, 0.5, 0.0], [0.5, -1.0, 0.0], [0.0, 0.0, 0.0]], complex)


def _masses(ml):
    return np.sqrt(np.array([0.0, 7.537e-5, 2.511e-3]) + ml * ml)


def _kinds(m, S=S_EMU, Um=U):
    """The two kinds' (M_rot, S_signed) as the solver builds them: M_rot from the differences of the masses squared
    (the ladder adds the lightest back)."""
    M_rot = Um @ np.diag(m ** 2 - np.min(m) ** 2) @ Um.conj().T
    return [(M_rot, S), (np.conj(M_rot), -np.conj(S))]


def _exact_widths(G, Um, m, M, S, sig, s, nz=3000):
    """E Gamma[a, rel, b] of kind s at sigma, independently of the code: the spinors' |A|^2 (_spinor_amp2) with the
    exact eigenvectors and effective masses at the parent's sigma and at each daughter's z sigma, on a fine z grid
    (log near 0, log in 1 - z near 1)."""
    Gp = dm.coupling_mass_basis(G, Um)
    Sp = Um.conj().T @ S @ Um
    H0 = np.diag(m * m).astype(complex)
    mu, c = np.linalg.eigh(H0 + (sig * Sp if s == 0 else -sig * Sp.conj()))
    zc = np.concatenate([np.geomspace(1e-12, 0.5, nz), 1.0 - np.geomspace(0.5, 1e-15, nz)[1:]])
    out = np.zeros((3, 2, 3))
    lam_in = -1 if s == 0 else +1

    def daughter(z, sd):
        return np.linalg.eigh(np.array([H0 + (zz * sig * Sp if sd == 0 else -zz * sig * Sp.conj()) for zz in z]))

    for rel in range(2):
        sd = s if rel == 0 else 1 - s
        musc, _ = daughter(zc, sd)
        for a in range(3):
            for b in range(3):
                kT2 = (1 - zc) * (zc * mu[a] - musc[:, b]) - zc * M * M
                inside = np.nonzero(kT2 >= 0.0)[0]
                if len(inside) == 0:
                    continue
                # the window bracketed on the coarse grid, then resolved: the coarse points inside and 4000 more
                lo, hi = zc[max(inside[0] - 1, 0)], zc[min(inside[-1] + 1, len(zc) - 1)]
                z = np.unique(np.concatenate([zc[(zc >= lo) & (zc <= hi)], np.linspace(lo, hi, 4000)]))
                mus, cs = daughter(z, sd)
                f = _spinor_amp2(c[:, a], cs[:, :, b], Gp, m, mu[a], mus[:, b], z, M, lam_in,
                                 lam_in if rel == 0 else -lam_in)
                out[a, rel, b] = _trapz(f, z) / (16 * np.pi)
    return out

def _plan_widths(lad, r):
    """E Gamma[s, a, rel, b] of the ladder's rung r."""
    tot = lad.arrays[4][r]
    return tot.reshape(2, 3, 2, 3) / (16 * np.pi * lad.Ms)


def test_vacuum_limit_is_the_vacuum_decay():
    """Without a potential the plans are dp_decay's: each channel's total the rest rate x mass (rest_rates) x its kept
    or flipped share, and its own-bin share window_plan's (daughter_spectra's fitted spectra: to their fit, 5e-8 and
    3e-6); no flip nu_i -> nubar_i (closed in vacuum)."""
    E = np.geomspace(1e12, 3e16, 300)
    h = np.sqrt(E[1] / E[0])
    G = 1e-6 * S_EMU
    for ml, M in ((0.1, 1e-3), (1e-3, 1e-4), (1e-6, 1e-8)):
        m = _masses(ml)
        lad = dm.Ladder(E, G, U, m, M, _kinds(m), 0.0)
        Gam, _ = dd.rest_rates(G, U, m, M)
        spec = dd.daughter_spectra(G, U, m, M)
        kf = np.zeros((2, 3, 3, 2, 2), np.int64)
        kx = np.zeros((2, 3, 3, 2, dd.NSLOT), np.int64)
        fx = np.zeros((2, 3, 3, 2, dd.NSLOT))
        dd.window_plan(spec, h, len(E), kf, kx, fx)
        r = 102
        W = _plan_widths(lad, r)
        own = lad.arrays[5][r].reshape(2, 3, 2, 3) / (16 * np.pi * lad.Ms)
        for s in range(2):
            for i in range(3):
                for rel in range(2):
                    for j in range(3):
                        if Gam[i, j] == 0.0:
                            assert W[s, i, rel, j] == 0.0
                            continue
                        zm, zp, a, b, c = spec[s, i, j, rel]
                        share = a * (zp - zm) + 0.5 * b * (zp ** 2 - zm ** 2) + (c * np.log(zp / zm) if c else 0.0)
                        assert np.isclose(W[s, i, rel, j], Gam[i, j] * m[i] * share, rtol=1e-7)
                        own_old = sum(fx[s, i, j, rel, x] for x in range(dd.NSLOT) if kx[s, i, j, rel, x] == 0)
                        assert np.isclose(own[s, i, rel, j], Gam[i, j] * m[i] * own_old, rtol=3e-6)


def test_lowest_state_cannot_decay_into_itself():
    """A kept decay a -> a is closed whatever the potential (mu'_b(z sigma) / z >= mu_b(sigma), Weyl): the exact
    kinematics leave no window for any state into itself, the lowest state has no kept channel, and the plans have
    none (where the model before had the lowest state decay into itself through its heavier content)."""
    m = _masses(0.1)
    E = np.geomspace(1e12, 3e16, 200)
    for c in (1e-20, 1e-18, 1e-16):
        lad = dm.Ladder(E, 1e-20 * S_EMU, U, m, 1e-20, _kinds(m), c)
        W = _plan_widths(lad, lad.nr // 2)
        for s in range(2):
            assert np.all(W[s, 0, 0] == 0.0)                       # the lowest: no kept channel at all
            for a in range(3):
                assert W[s, a, 0, a] == 0.0
        Sp = U.conj().T @ S_EMU @ U
        for sig in np.geomspace(1e-9, 1e-1, 9):
            for z in (0.999, 0.9, 0.5, 0.1):
                mu = np.linalg.eigvalsh(np.diag(m * m) + sig * Sp)
                muz = np.linalg.eigvalsh(np.diag(m * m) + z * sig * Sp)
                assert np.all(z * mu - muz <= 1e-15 * np.abs(mu).max())  # Q <= 0 for b = a


@pytest.mark.slow     # 15 s a case on a laptop (the spinor reference): Tycho's suite runs it
@pytest.mark.parametrize("ml, M, g, c, G0", [(0.1, 1e-20, 1e-20, 2.7e-19, S_EMU),
                                             (0.01, 1e-16, 3e-17, 2.5e-20, S_EMU),
                                             (0.01, 1e-12, 1e-11, 2.7e-17, S_EMU),
                                             (0.01, 1e-12, 1e-11, 2.7e-17, G_NC)])
def test_plans_match_the_exact_rates(ml, M, g, c, G0):
    """At the le_lmu_ul validation points (z = 2's coefficient), and with a coupling that does not commute with the
    texture: every state's total width from the plans against the explicit spinors' amplitude with the eigenvectors
    at the parent's and each daughter's own sigma (to 1e-3: the daughter bin's eigenvector held, second order in the
    bin width; the reference's z grid)."""
    m = _masses(ml)
    E = np.geomspace(994997970133.3439, 3e16, 515)
    lad = dm.Ladder(E, g * G0, U, m, M, _kinds(m), 1.5 * c)
    for Eo in (1e13, 1e15, 1e16):
        x = np.log(Eo * c / lad.sig0) / (2 * np.log(lad.h))
        r = int(round(x))
        Wl = _plan_widths(lad, r).sum(axis=(2, 3))
        for s in range(2):
            We = _exact_widths(g * G0, U, m, M, S_EMU, lad.sig[r], s).sum(axis=(1, 2))
            for a in range(3):
                if We[a] > 1e-6 * We.max():
                    assert abs(Wl[s, a] / We[a] - 1.0) < 1e-3, (Eo, s, a, Wl[s, a], We[a])


def test_kinds_are_conjugate():
    """Real mixing and coupling: the antineutrinos in the potential V decay as the neutrinos in -V (the CP
    conjugate), channel by channel; the flips nu_i -> nubar_i go one way only (the potential splits the two)."""
    Ur = np.asarray(construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), 0.0)).real.astype(complex)
    m = _masses(0.05)
    E = np.geomspace(1e12, 3e16, 200)
    c = 3e-18
    a = dm.Ladder(E, 1e-15 * S_EMU, Ur, m, 1e-16, _kinds(m, S_EMU, Ur), c)
    b = dm.Ladder(E, 1e-15 * S_EMU, Ur, m, 1e-16, _kinds(m, -S_EMU, Ur), c)
    for r in (a.nr // 3, a.nr // 2, a.nr - 10):
        Wa, Wb = _plan_widths(a, r), _plan_widths(b, r)
        assert np.allclose(Wa[1], Wb[0], rtol=1e-9, atol=1e-30 * Wa.max())
        assert np.allclose(Wa[0], Wb[1], rtol=1e-9, atol=1e-30 * Wa.max())
        for i in range(3):
            assert Wa[0, i, 1, i] == 0.0 or Wa[1, i, 1, i] == 0.0


def test_runs_match_every_bins_exact_integral():
    """The runs' moments (deposit's add / sub, gains' psi) against every daughter bin by its exact integral
    (_explicit), both rungs blended, vacuum and potentials, narrow and wide windows."""
    E = np.geomspace(1e12, 3e16, 300)
    h = np.sqrt(E[1] / E[0])
    for ml, M, c in ((1e-3, 1e-4, 0.0), (1e-3, 1e-4, 3e-18), (0.01, 1e-8, 1e-16), (1e-6, 1e-10, 1e-17)):
        m = _masses(ml)
        G = 1e-7 * S_EMU
        lad = dm.Ladder(E, G, U, m, M, _kinds(m), c)
        x0 = lad.coord(c)
        K = dm.vertex_matrices(G, U, m)
        for p in (299, 150):
            for s in range(2):
                for a in range(3):
                    pend, acc, add, sub, act = lad.buffers()
                    pref = 1.0 / (16 * np.pi * lad.Ms * E[p])
                    dm.deposit(lad.arrays, x0, p, s, a, pref, E[p], h, pend, add, sub, act)
                    got = np.zeros((p, 2, 3))
                    out = np.zeros((2, 3))
                    for e in range(p - 1, -1, -1):
                        dm.gains(lad.arrays, x0, e, E[e], h, acc, add, sub, pend, out, act)
                        got[e] = out
                    assert not np.any(add) and not np.any(sub) and not np.any(pend)
                    r0, w = dm.rung_of_bin(x0, p)
                    ref = np.zeros((p, 2, 3))
                    for st, om in ((0, 1 - w), (1, w)):
                        r = r0 + st
                        Pa = np.zeros((3, 3), complex)
                        dm._unpack(lad.P9[r, s, a], Pa)
                        for rel in range(2):
                            sd = s if rel == 0 else 1 - s
                            N9 = np.zeros((3, 9))
                            dm._parent_N(K, s, rel, Pa, N9, np.zeros((3, 3), complex), np.zeros((3, 3), complex),
                                         np.zeros(9))
                            for b in range(3):
                                if lad.arrays[4][r, ((s * 3 + a) * 2 + rel) * 3 + b] == 0.0:
                                    continue
                                for k in range(1, p + 1):
                                    rho = r - k
                                    Dl = (lad.m2[r, s, a] - lad.m2[rho, sd, b]
                                          + lad.sig[r] * (lad.dd[r, s, a] - lad.dd[rho, sd, b]))
                                    mb = lad.m2[rho, sd, b]
                                    ref[p - k, sd, b] += om * pref * dm._explicit(
                                        rel, N9, lad.P9[rho, sd, b] * dm.TRW, Dl + mb, Dl, mb, lad.Ms, h,
                                        h ** (-2 * k - 1), h ** (-2 * k + 1), np.zeros(3))
                    tot = np.abs(ref).sum()
                    if tot > 0.0:
                        assert np.abs(got - ref).sum() < 1e-8 * tot


class _Step:
    """dp_sweep.step_full with the decays alone (no injection, absorption or jumps)."""

    def __init__(self, E, lad, c):
        self.E, self.lad, self.x0 = E, lad, lad.coord(c)
        N = len(E)
        self.W3, self.pref3, self.ss = cascade_kernel_weights(E)
        self.W1, self.W2, self.c1, self.c2, self.sx = cross_kernel_weights(E)
        self.Z = np.zeros((N, 3))
        self.Pi = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        self.brp = np.full((N, 3), 0.5 / 3)

    def run(self, Fn, Fb, dt, n=1):
        Fn, Fb = Fn.copy(), Fb.copy()
        N = len(self.E)
        Rn, Rb = np.zeros(N), np.zeros(N)
        z0 = np.zeros((2, 3))
        tab = (np.zeros((2, 3, 3)), z0, z0, z0, 0.0, 1.0, 0.0)
        for _ in range(n):
            dp_sweep.step_full(0.0, dt / n, 1.0, dt / n, dt / n, self.E, Fn, Fb, 0.5, 0.5, np.full(3, 1 / 3), self.Pi,
                               self.Pi, self.Z, self.Z, self.brp, self.brp, 2.5, 0.0, self.W3, self.pref3, self.ss,
                               self.W1, self.W2, self.c1, self.c2, self.sx, Rn, Rb, np.ones(N), False, 0.0, *tab, *tab,
                               self.lad.arrays, self.x0, True, True, self.lad.diag, *self.lad.buffers(),
                               np.zeros((1, 2, 3)), False, np.zeros((1, 4, 6, 6)), np.zeros(1, dtype=np.bool_), False)
        return Fn, Fb


def _number(E, Fn, Fb):
    h = np.sqrt(E[1] / E[0])
    return float(((Fn + Fb).sum(axis=1) * E * (h - 1 / h)).sum())


def test_each_decay_keeps_one_neutrino():
    """Nearly degenerate masses (ml 0.1 eV: every daughter on the grid): the number is kept to rounding for any decay
    per step, with and without a potential (the eigenstates, the flips), and the populations stay positive."""
    E = np.geomspace(1e13, 1e16, 60)
    m = _masses(0.1)
    rng = np.random.default_rng(1)
    Fn = rng.uniform(0.5, 1, (60, 3)) * (E[:, None] / 1e14) ** -2
    Fb = rng.uniform(0.5, 1, (60, 3)) * (E[:, None] / 1e14) ** -2
    Fn[:20] = Fb[:20] = 0.0
    G = 1e-6 * S_EMU
    lab = float(np.max(dd.rest_rates(G, U, m, 1e-3)[0].sum(1) * m)) / E[-1]
    for c in (0.0, 1e-20, 1e-17):
        st = _Step(E, dm.Ladder(E, G, U, m, 1e-3, _kinds(m), c), c)
        for f in (1e-3, 1.0, 1e3):
            a, b = st.run(Fn, Fb, f / lab)
            assert abs(_number(E, a, b) / _number(E, Fn, Fb) - 1.0) < 1e-10
            assert np.all(a >= 0.0) and np.all(b >= 0.0)


def test_small_step_in_vacuum_is_decay_step():
    """Without a potential a small step (1e-7 decayed) moves the populations as dp_decay.decay_step (the vacuum
    model the in-medium one reduces to), to its second order and daughter_spectra's fit."""
    E = np.geomspace(1e13, 1e16, 60)
    rng = np.random.default_rng(2)
    for ml, M in ((1e-6, 1e-4), (0.1, 1e-3), (0.01, 1e-4)):
        m = _masses(ml)
        G = 1e-6 * S_EMU
        Gam, _ = dd.rest_rates(G, U, m, M)
        spec = dd.daughter_spectra(G, U, m, M)
        Fn = rng.uniform(0.5, 1, (60, 3)) * (E[:, None] / 1e14) ** -2
        Fb = rng.uniform(0.5, 1, (60, 3)) * (E[:, None] / 1e14) ** -2
        dt = 1e-7 / (float(np.max(Gam.sum(1) * m)) / E[0])
        a, b = _Step(E, dm.Ladder(E, G, U, m, M, _kinds(m), 0.0), 0.0).run(Fn, Fb, dt)
        an, ab = Fn.copy(), Fb.copy()
        dd.decay_step(E, 1.0, dt, Gam, spec, m, an, ab, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
        for F1, A1, F0 in ((a, an, Fn), (b, ab, Fb)):
            assert np.max(np.abs((F1 - F0) - (A1 - F0))) < 1e-6 * np.max(np.abs(A1 - F0))


def test_galactic_sweep_is_the_cosmological_step():
    """The Galactic segments' decay sweep (dp_medium.decay_sweep, own-bin chains in order of the effective masses)
    against the cosmological step (dp_sweep.step_full, the bin solved exactly) over the same path at z = 0, with a
    potential: the same to second order in the decays per step and in the number (some flips' daughters leave the
    grid's bottom); without a potential (every daughter on the grid) the sweep keeps the number."""
    E = np.geomspace(1e13, 1e16, 60)
    m = _masses(0.1)
    G = 1e-6 * S_EMU
    c = 1e-17
    lad = dm.Ladder(E, G, U, m, 1e-3, _kinds(m), c)
    rng = np.random.default_rng(3)
    Fn = rng.uniform(0.5, 1, (60, 3)) * (E[:, None] / 1e14) ** -2
    Fb = rng.uniform(0.5, 1, (60, 3)) * (E[:, None] / 1e14) ** -2
    lam = float(np.max(lad.width(E[0], c))) / E[0]
    for f in (1e-4, 1e-2):
        L = f / lam
        a, b = _Step(E, lad, c).run(Fn, Fb, L)
        gn, gb = Fn.copy(), Fb.copy()
        dm.decay_sweep(lad.arrays, lad.coord(c), E, gn, gb, L, *lad.buffers())
        d_ref = np.abs(np.concatenate([a - Fn, b - Fb])).max()
        assert np.abs(np.concatenate([gn - a, gb - b])).max() < 3.0 * f * d_ref
        lost_ref = _number(E, Fn, Fb) - _number(E, a, b)
        assert abs((_number(E, Fn, Fb) - _number(E, gn, gb)) - lost_ref) < 3.0 * f * abs(lost_ref) + 1e-13 * _number(
            E, Fn, Fb)
    vac = dm.Ladder(E, G, U, m, 1e-3, _kinds(m), 0.0)
    Fn[:20] = Fb[:20] = 0.0                              # the chains stay above the grid's bottom
    gn, gb = Fn.copy(), Fb.copy()
    dm.decay_sweep(vac.arrays, vac.coord(0.0), E, gn, gb, 1e3 / lam, *vac.buffers())
    assert abs(_number(E, gn, gb) / _number(E, Fn, Fb) - 1.0) < 1e-12


def test_rates_are_the_spinor_amplitude():
    """The kept and flipped |A|^2 the code uses (module docstring) equal the explicit light-front spinors' amplitude
    summed over the mediator's polarizations, to 1e-9: both kinds, three textures, Majorana phases, vacuum and
    potential, and a coupling that does not commute with the texture (no [G, S] = 0 needed)."""
    Um = U @ np.diag(np.exp(1j * np.array([0.0, 0.7, 2.1])))
    m = _masses(0.03)
    z = np.array([0.05, 0.3, 0.7, 0.95, 0.999])
    worst = 0.0
    for S, G in ((S_EMU, 3e-3 * S_EMU), (np.diag([1.0, 0, -1]).astype(complex), 2e-3 * np.diag([1.0, 0, -1])),
                 (S_EMU, 2e-3 * G_NC)):
        for UU in (U, Um):
            Gp = dm.coupling_mass_basis(np.asarray(G, complex), UU)
            Sp = UU.conj().T @ S @ UU
            for sig in (0.0, 3e-5, 1e-3, 3e-2):
                for M in (1e-4, 3e-3):
                    for s in (0, 1):
                        H0 = np.diag(m * m).astype(complex)
                        mu, c = np.linalg.eigh(H0 + (sig * Sp if s == 0 else -sig * Sp.conj()))
                        for rel in (0, 1):
                            sd = s if rel == 0 else 1 - s
                            lam = -1 if s == 0 else +1
                            mus, cs = np.linalg.eigh(np.array([H0 + (zz * sig * Sp if sd == 0 else
                                                                     -zz * sig * Sp.conj()) for zz in z]))
                            Gs = Gp if s == 0 else Gp.conj()
                            for a in range(3):
                                for b in range(3):
                                    bf = _spinor_amp2(c[:, a], cs[:, :, b], Gp, m, mu[a], mus[:, b], z, M, lam,
                                                      lam if rel == 0 else -lam)
                                    fo = _formula_amp2(c[:, a], cs[:, :, b], Gs, m, mu[a], mus[:, b], z, M, rel)
                                    sc = np.maximum(np.maximum(np.abs(bf), np.abs(fo)), 1e-300)
                                    worst = max(worst, float(np.max(np.abs(bf - fo) / sc)))
    assert worst < 1e-9, worst


def test_ladder_room_grows():
    """Near a threshold every bin of a wide window can be a cut bin (the review's finding 1: le_ltau_ul, M = 0.0316
    eV, g = 1e-5, m_l = 1e-3, the Galactic ladder; 26 cut bins in one channel): the ladder grows the plans' room
    instead of refusing, and the grown rung's widths are the exact ones."""
    th = np.radians([33.75878818411903, 8.623062290036087, 43.280093616242404, 212.0])
    Ut = np.asarray(construct_PMNS(*th))
    m = np.sqrt(np.array([0.0, 7.537e-05, 2.511e-3]) + 1e-6)
    S = np.diag([1.0, 0.0, -1.0]).astype(complex)
    E = np.geomspace(4572260913255.834, 3e16, 439)
    M = 10 ** -1.5
    lad = dm.Ladder(E, 1e-5 * S, Ut, m, M, _kinds(m, S, Ut), 2.4980427467047744e-19)
    assert lad.capacity[0] > dm.NEXP
    r = int(np.argmax(lad.arrays[6].max(axis=1)))
    assert lad.arrays[6][r].max() > dm.NEXP
    Wl = _plan_widths(lad, r).sum(axis=(2, 3))
    for s in range(2):
        We = _exact_widths(1e-5 * S, Ut, m, M, S, lad.sig[r], s).sum(axis=(1, 2))
        for a in range(3):
            if We[a] > 1e-6 * We.max():
                assert abs(Wl[s, a] / We[a] - 1.0) < 1e-3, (s, a, Wl[s, a], We[a])
