"""dp_mediator (2026-11-07, decision B): the mediator the beam's decays make, Z_D -> nu_k nu_l in flight -- the widths,
the light-front distributions and helicities, the tables' number and energy per block (also at the kinematic
thresholds), the step (dp_sweep.step_full_med): conservation, the mediator's lab rate, a potential's projections,
nothing changed where it cannot act; the code before (a frozen reference); dp_si's transport, against dp_sweep's; the
configuration's checks."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dp_decay as dd  # noqa: E402
import dp_medium as dm  # noqa: E402
import dp_mediator as dmed  # noqa: E402
import dp_sweep  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402
from dp_cosmo import cascade_kernel_weights, cross_kernel_weights  # noqa: E402

U = np.asarray(construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0)))
S_EMU = np.diag([1.0, -1.0, 0.0]).astype(complex)
G_TAU = np.zeros((3, 3), complex)
G_TAU[2, 2] = 1.0
_XG, _WG = np.polynomial.legendre.leggauss(40)
REPO = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(REPO, "analysis", "configs", "campaign_final", "now_events")


def _masses(ml, inverted=False):
    if inverted:      # m_3 the lightest
        return np.sqrt(np.array([2.511e-3 - 7.537e-5, 2.511e-3, 0.0]) + ml * ml)
    return np.sqrt(np.array([0.0, 7.537e-5, 2.511e-3]) + ml * ml)


def _kinds(m, S=S_EMU):
    M_rot = U @ np.diag(m ** 2 - np.min(m) ** 2) @ U.conj().T
    return [(M_rot, S), (np.conj(M_rot), -np.conj(S))]


def _rate_from_traces(M, mk, ml, g, majorana, lam):
    """The width of one LF helicity from the traces: S p / (16 pi M^2) int dcos sum_h |M|^2 (S = 1/2 for identical)."""
    p = dmed._p_star(M, mk, ml)
    tot = 0.0
    for h1 in (-1, 1):
        for h2 in (-1, 1):
            tot += sum(w * dmed.msq_decay(M, mk, ml, c, lam, h1, h2, g, majorana)[0] for c, w in zip(_XG, _WG))
    return tot * p / (16.0 * np.pi * M * M)


@pytest.mark.parametrize("majorana", (True, False))
def test_pair_widths_are_the_traces(majorana):
    """Each LF helicity decays at the width formula (rotational invariance), massive daughters, complex couplings,
    identical particles (Majorana k = l: S = 1/2); massless daughters: Tr(G G^dag) M / 24 pi; k = l: g^2 M beta^3 / 24
    pi (the axial coupling's P wave)."""
    G = 1e-6 * (G_TAU + 0.3 * S_EMU + 0.2j * (np.eye(3, k=1) - np.eye(3, k=-1)))
    for ml, M in ((1e-6, 1e-2), (0.01, 0.035), (0.02, 0.09)):
        m = _masses(ml)
        W = dmed.pair_widths(G, U, m, M, majorana)
        Gp = dmed.coupling_mass_basis(G, U)
        for k in range(3):
            for l in range(3):
                if W[k, l] == 0.0:
                    continue
                S = 0.5 if (majorana and k == l) else 1.0
                for lam in (0, 1, -1):
                    r = S * _rate_from_traces(M, m[k], m[l], Gp[k, l], majorana, lam)
                    assert abs(r / W[k, l] - 1.0) < 1e-10
    m0 = np.array([1e-12, 2e-12, 3e-12])
    W = dmed.pair_widths(G, U, m0, 1e-2)
    assert abs(W.sum() / (np.trace(G @ G.conj().T).real * 1e-2 / (24 * np.pi)) - 1.0) < 1e-9
    Gd = 1e-6 * np.eye(3, dtype=complex)
    m = _masses(0.02)
    W = dmed.pair_widths(Gd, np.eye(3), m, 0.05)
    beta = np.sqrt(1.0 - 4.0 * m[0] ** 2 / 0.05 ** 2)
    assert abs(W[0, 0] / (1e-12 * 0.05 * beta ** 3 / (24 * np.pi)) - 1.0) < 1e-12


def _moments(d, other=False):
    """(<t>, <t^2>) of a Density (other: of the partner's share 1 - t)."""
    if other:
        return d.moment(lambda t, y: y), d.moment(lambda t, y: y * y)
    return d.moment(lambda t, y: t), d.moment(lambda t, y: t * t)


def test_massless_daughters_are_the_d_functions():
    """Massless daughters: one neutrino (LF helicity -1) and one antineutrino per decay; the neutrino's share 6 x (1-x)
    (longitudinal), 3 (1-x)^2 (+1), 3 x^2 (-1): moments <x> = 1/2, 1/4, 3/4 and <x^2> = 3/10, 1/10, 3/5."""
    m = np.array([1e-9, 2e-9, 3e-9])
    dens, W, res = dmed.decay_densities(G_TAU * 1e-6, U, m, 1e-2)
    assert res < 1e-9
    ref = {0: (0.5, 0.3), 1: (0.25, 0.1), 2: (0.75, 0.6)}
    for k in range(3):
        for l in range(3):
            if W[k, l] <= 0.0:
                continue
            for li in range(3):
                tot = sum(dens[(k, l, li, hi)].total for hi in range(4))
                # same-helicity pairs: none
                assert dens[(k, l, li, 0)].total < 1e-8 * tot and dens[(k, l, li, 3)].total < 1e-8 * tot
                d1, d2 = dens[(k, l, li, 1)], dens[(k, l, li, 2)]     # (-1, +1): nu first; (+1, -1): nu second
                a1, b1 = _moments(d1)
                a2, b2 = _moments(d2, other=True)
                w1, w2 = d1.total / tot, d2.total / tot
                assert abs(w1 + w2 - 1.0) < 1e-8
                assert abs(w1 * a1 + w2 * a2 - ref[li][0]) < 1e-6 and abs(w1 * b1 + w2 * b2 - ref[li][1]) < 1e-6


def test_production_adds_up_to_daughter_spectra():
    """The three LF helicities' densities of the daughter's share add up to daughter_spectra's a + b z + c / z (each
    rel's share of the pair's rate)."""
    for ml, M in ((1e-6, 1e-2), (0.01, 0.03), (1e-6, 1e-4)):
        m = _masses(ml)
        G = 1e-6 * G_TAU
        spec = dd.daughter_spectra(G, U, m, M)
        dens, res = dmed.production_densities(G, U, m, M)
        assert res < 1e-9
        for s_ in range(2):
            for i in range(3):
                for j in range(3):
                    if (s_, i, j, 0, 0) not in dens:
                        continue
                    tot = sum(dens[(s_, i, j, rel, li)].total for rel in range(2) for li in range(3))
                    for rel in range(2):
                        zm, zp, a, b, c = spec[s_, i, j, rel]
                        z = np.linspace(zm, zp, 9)[1:-1]
                        got = sum(dens[(s_, i, j, rel, li)].pdf(z) for li in range(3)) / tot
                        ref = a + b * z + c / z
                        assert np.max(np.abs(got - ref)) < 1e-7 * np.max(np.abs(ref))


def test_helicities_at_the_collinear_ends():
    """At the collinear ends (k_T = 0) the angular momentum along the beam fixes the mediator's LF helicity: in
    nu_i -> nu_j + Z_D a kept daughter leaves lam = 0 and a flipped one lam = h (a neutrino parent: -1); in
    Z_D -> nu_k nu_l, lam = (h1 + h2) / 2. Every other helicity's |M|^2 vanishes there (massive and massless
    daughters, complex couplings)."""
    g = np.exp(0.7j)
    for mi, mj, M in ((0.05, 1e-6, 1e-2), (0.05, 8.7e-3, 1e-2), (0.112, 0.1, 1e-3), (0.05, 1e-6, 0.045)):
        for c in (-1.0, 1.0):
            v = {(h, hp, lam): dmed.msq_production(mi, mj, M, c, h, hp, lam, g)[0]
                 for h in (-1, 1) for hp in (-1, 1) for lam in (0, 1, -1)}
            big = max(abs(x) for x in v.values())
            assert big > 0.0
            for (h, hp, lam), x in v.items():
                if lam != (h - hp) // 2:
                    assert abs(x) < 1e-12 * big
    for M, mk, ml in ((1e-2, 1e-6, 8.7e-3), (0.03, 8.7e-3, 8.7e-3), (0.06, 8.7e-3, 0.05), (1e-2, 1e-6, 1e-6)):
        for c in (-1.0, 1.0):
            v = {(h1, h2, lam): dmed.msq_decay(M, mk, ml, c, lam, h1, h2, g)[0]
                 for h1 in (-1, 1) for h2 in (-1, 1) for lam in (0, 1, -1)}
            big = max(abs(x) for x in v.values())
            assert big > 0.0
            for (h1, h2, lam), x in v.items():
                if lam != (h1 + h2) // 2:
                    assert abs(x) < 1e-12 * big


def _check_tables(med, G, m, M, h, N):
    """Every block of the tables: its number its exact share, its energy at the bins' centres exact, no negative entry
    (the review's findings 1 and 4)."""
    K1 = med.prod.shape[-1]
    e = h ** (-2.0 * np.arange(K1))
    pdens, _ = dmed.production_densities(G, U, m, M)
    for ch in range(dmed.NCH):
        assert np.all(med.prod[ch] >= 0.0)
        if med.fallback[ch]:
            continue
        s_, a, rel, b = ch // 18, (ch // 6) % 3, (ch // 3) % 2, ch % 3
        i, j = med.order[a], med.order[b]
        ds = [pdens[(s_, i, j, rel, li)] for li in range(3)]
        tot = sum(d.total for d in ds if d.total > 0.0)
        assert abs(med.prod[ch].sum() - 1.0) < 1e-12
        dcdf = lambda z: sum(d.total * d.cdf(z) for d in ds if d.total > 0.0) / tot  # noqa: E731
        dtab = dmed._table(dcdf, ds[0].lo, ds[0].hi, h, N, K1)
        assert abs(np.sum(med.prod[ch] * e) + np.sum(dtab * e) - 1.0) < 1e-12     # the decay's energy, exactly
        for li, d in enumerate(ds):
            share = d.total / tot if d.total > 1e-14 * tot else 0.0
            assert abs(med.prod[ch, li].sum() - share) < 1e-12                   # each helicity's share, exactly
    ddens, W, _ = dmed.decay_densities(G, U, m, M)
    for li in range(3):
        assert np.all(med.dec[li] >= 0.0)
        assert abs(med.dec[li].sum() - 2.0 * med.br_nu) < 1e-12
        assert abs(np.sum(med.dec[li] * e) - med.br_nu) < 1e-12
        # each kind and mass state's share and energy, exactly
        num = np.zeros((2, 3))
        ene = np.zeros((2, 3))
        for k in range(3):
            for l in range(3):
                if W[k, l] <= 0.0:
                    continue
                tot = sum(ddens[(k, l, li, hi)].total for hi in range(4) if ddens[(k, l, li, hi)].total > 0.0)
                for hi in range(4):
                    d = ddens[(k, l, li, hi)]
                    if not d.total > 1e-14 * tot:
                        continue
                    w = W[k, l] / med.Gamma * d.total / tot
                    num[hi // 2, k] += w
                    num[hi % 2, l] += w
                    ene[hi // 2, k] += w * d.mean
                    ene[hi % 2, l] += w * (1.0 - d.mean)
        assert np.max(np.abs(med.dec[li].sum(axis=-1) - num)) < 1e-12
        assert np.max(np.abs(np.sum(med.dec[li] * e, axis=-1) - ene)) < 1e-12


def test_tables_keep_number_and_energy_per_block():
    """The tables per decay, block by block: a production channel's mediators of each LF helicity hold its exact share,
    and together the parent's energy (its bin's centre) less the daughter's as the vacuum decay places it; a decay's
    neutrinos of each kind and mass state hold their exact share and continuous energy (adding up to two neutrinos
    and the mediator's energy, times the neutrino branching)."""
    for n_E in (90, 515):
        E = np.geomspace(1e12, 3e16, n_E)
        for ml, M, G in ((1e-6, 1e-2, 2e-5 * G_TAU), (0.01, 0.035, 1e-6 * S_EMU), (1e-6, 3e-2, 1e-5 * G_TAU)):
            m = _masses(ml)
            med = dmed.Mediator(G, U, m, M, E)
            _check_tables(med, G, m, M, med.h, n_E)


def _thresholds():
    """The review's failing points (finding 1) and more: within 1e-4 .. 1e-12 of every kinematic threshold."""
    pts = []
    for ml in (1e-6, 1e-3, 1e-2, 0.05):
        m = _masses(ml)
        for rel in (1e-4, 1e-6, 1e-9, 1e-12):
            pts += [(ml, 2 * m[0] * (1 + rel)), (ml, 2 * m[1] * (1 + rel)), (ml, (m[0] + m[1]) * (1 + rel))]
            pts += [(ml, (m[1] - m[0]) * (1 - rel)), (ml, (m[2] - m[0]) * (1 - rel)), (ml, (m[2] - m[1]) * (1 - rel))]
    m = _masses(1e-6)
    pts += [(1e-6, (m[2] - m[0]) - 1e-9), (0.05, (_masses(0.05)[1] - _masses(0.05)[0]) * (1 - 1e-3))]
    return pts


@pytest.mark.parametrize("G", (2e-5 * G_TAU, 1e-6 * S_EMU), ids=("tau", "e-mu"))
def test_tables_at_the_thresholds(G):
    """Near every kinematic threshold (the review's finding 1: the closed-form fits and the energy tilt raised there,
    and the analysis' objective made such a point a wall) the tables build, every block exact."""
    E = np.geomspace(1e12, 3e16, 515)
    for ml, M in _thresholds():
        m = _masses(ml)
        med = dmed.Mediator(G, U, m, M, E)
        _check_tables(med, G, m, M, med.h, len(E))


class _Step:
    """dp_sweep.step_full_med with the decays alone (no injection, absorption or jumps)."""

    def __init__(self, E, G, m, M, c=0.0):
        self.E = E
        self.lad = dm.Ladder(E, G, U, m, M, _kinds(m), c)
        self.x0 = self.lad.coord(c)
        self.med = dmed.Mediator(G, U, m, M, E)
        N = len(E)
        self.W3, self.pref3, self.ss = cascade_kernel_weights(E)
        self.W1, self.W2, self.c1, self.c2, self.sx = cross_kernel_weights(E)
        self.Z0 = np.zeros((N, 3))
        self.Pi = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        self.brp = np.full((N, 3), 0.5 / 3)

    def run(self, Fn, Fb, Zm, dt, n=1, use_med=True, rate=None, decay_dts=None, use_dec=True):
        Fn, Fb, Zm = Fn.copy(), Fb.copy(), Zm.copy()
        N = len(self.E)
        Rn, Rb = np.zeros(N), np.zeros(N)
        z0 = np.zeros((2, 3))
        tab = (np.zeros((2, 3, 3)), z0, z0, z0, 0.0, 1.0, 0.0)
        mp, mnp, mps, mdec, mnd, mds, mr = self.med.arrays
        mr = mr if rate is None else rate
        pZ, pM, diag = np.zeros((N, 3)), np.zeros((N, 2, 3)), np.zeros(dmed.NDIAG)
        for i in range(n):
            ddt = dt / n if decay_dts is None else decay_dts[i]
            args = (0.0, dt / n, 1.0, dt / n, ddt, self.E, Fn, Fb, 0.5, 0.5, np.full(3, 1 / 3), self.Pi,
                    self.Pi, self.Z0, self.Z0, self.brp, self.brp, 2.5, 0.0, self.W3, self.pref3, self.ss,
                    self.W1, self.W2, self.c1, self.c2, self.sx, Rn, Rb, np.ones(N), False, 0.0, *tab, *tab,
                    self.lad.arrays, self.x0, use_dec, True, self.lad.diag, *self.lad.buffers(),
                    np.zeros((1, 2, 3)), False, np.zeros((1, 4, 6, 6)), np.zeros(1, dtype=np.bool_), False)
            if use_med:
                dp_sweep.step_full_med(*args, True, Zm, mp, mnp, mps, mdec, mnd, mds, mr, pZ, pM, diag)
            else:
                dp_sweep.step_full(*args)
        assert not np.any(pZ) and not np.any(pM)
        return Fn, Fb, Zm, diag


def _budget(E, Fn, Fb, Zm):
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    nb = (Fn + Fb).sum(axis=1) * dE
    nz = Zm.sum(axis=1) * dE
    return nb.sum(), (nb * E).sum(), nz.sum(), (nz * E).sum()


@pytest.mark.parametrize("inverted", (False, True), ids=("normal", "inverted"))
def test_step_keeps_number_and_energy(inverted):
    """Massive daughters (m_lightest 0.01 eV, M 0.03 eV: nu_3 -> nu_1,2 + Z_D, Z_D -> nu_1 nu_1, nu_1 nu_2, nu_2 nu_2;
    the inverted ordering: the ladder's channels mapped on other mass states), populations in the grid's top decade,
    any decay per step: the neutrinos gain two per decayed mediator, the mediators gain one per decay and lose the
    decayed, the energy of both together is kept to rounding (nothing leaves the grid; a wrong label would break it);
    the populations stay positive."""
    E = np.geomspace(1e13, 1e17, 90)
    m = _masses(0.01, inverted)
    G = 1e-6 * G_TAU
    M = 0.03 if not inverted else 0.025
    st = _Step(E, G, m, M)
    rng = np.random.default_rng(3)
    Fn = rng.uniform(0.5, 1, (90, 3)) * (E[:, None] / 1e16) ** -2
    Fb = rng.uniform(0.5, 1, (90, 3)) * (E[:, None] / 1e16) ** -2
    Fn[:68] = Fb[:68] = 0.0
    Zm = np.zeros((90, 3))
    Zm[70:] = rng.uniform(0.1, 0.5, (20, 3)) * (E[70:, None] / 1e16) ** -2
    lab = float(np.max(dd.rest_rates(G, U, m, M)[0].sum(1) * m)) / E[-1]
    for f in (1e-2, 1.0, 30.0):
        for n in (1, 3):
            a, b, z, diag = st.run(Fn, Fb, Zm, f / lab, n=n)
            n0, e0, z0, ez0 = _budget(E, Fn, Fb, Zm)
            n1, e1, z1, ez1 = _budget(E, a, b, z)
            assert abs(diag[dmed.D_PROD_LOW]) < 1e-12 * diag[dmed.D_PROD]          # rounding: nothing leaves
            assert abs(diag[dmed.D_DAU_LOW]) < 1e-12 * diag[dmed.D_DAU]
            assert diag[dmed.D_DEC] > 0.0
            assert abs((n1 - n0) / (2.0 * diag[dmed.D_DEC]) - 1.0) < 1e-9
            assert abs((z1 - z0) - (diag[dmed.D_PROD] - diag[dmed.D_DEC])) < 1e-9 * z0
            assert abs((e1 + ez1) / (e0 + ez0) - 1.0) < 1e-11
            assert np.all(a >= 0.0) and np.all(b >= 0.0) and np.all(z >= 0.0)


def test_in_a_potential():
    """In a potential (sigma up to ~10 Delta m^2_21 over the grid): the mediators' neutrinos land on the eigenstates of
    their bin as the ladder's projectors put each mass state there (P9, blended between the bin's rungs) -- the same
    mediators in vacuum, projected by hand; with the beam decaying too (its eigenstates' decays make mediators, and
    the mediators' neutrinos decay again) the number is kept to rounding (each projection sums to one), the mediators
    made off the vacuum are counted, the populations stay positive."""
    E = np.geomspace(1e9, 1e17, 140)
    m = _masses(0.01)
    G, M, c = 1e-6 * G_TAU, 0.03, 1e-19
    vac, pot = _Step(E, G, m, M, 0.0), _Step(E, G, m, M, c)
    assert pot.x0 >= 0.0 and np.max(pot.lad.sig) > 0.0
    Zm = np.zeros((140, 3))
    Zm[115:] = np.linspace(0.2, 0.6, 25)[:, None] * np.array([1.0, 0.5, 0.25])
    zero = np.zeros((140, 3))
    dt = 3.0 * E[120] / vac.med.rate
    # the beam's own decays off (use_dec False), so that the daughters, eigenstates of the potential, are not
    # decaying further within the step (the heaviest eigenstate holds some nu_3 there)
    a0, b0, z0, _ = vac.run(zero, zero, Zm, dt, use_dec=False)
    a1, b1, z1, _ = pot.run(zero, zero, Zm, dt, use_dec=False)
    assert np.array_equal(z0, z1) and np.max(a1) > 0.0
    P9 = pot.lad.P9
    for kind, (F0, F1) in enumerate(((a0, a1), (b0, b1))):
        for e in range(140):
            r0, w = dm.rung_of_bin(pot.x0, e)
            P = (1.0 - w) * P9[r0, kind, :, :3] + (w * P9[r0 + 1, kind, :, :3] if w > 0.0 else 0.0)  # P[a, k]
            mass = F0[e][np.argsort(vac.lad.mu[2, kind])]   # vacuum eigenstates = mass states (normal: in order)
            np.testing.assert_allclose(F1[e], P @ F0[e], rtol=1e-12, atol=1e-14 * np.max(F0))
            assert np.allclose(mass, F0[e])
    # the beam decaying too: the number kept to rounding, positivity, the off-vacuum production counted
    rng = np.random.default_rng(4)
    Fn = rng.uniform(0.5, 1, (140, 3)) * (E[:, None] / 1e16) ** -2
    Fb = rng.uniform(0.5, 1, (140, 3)) * (E[:, None] / 1e16) ** -2
    Fn[:120] = Fb[:120] = 0.0
    lab = float(np.max(dd.rest_rates(G, U, m, M)[0].sum(1) * m)) / E[-1]
    a, b, z, diag = pot.run(Fn, Fb, Zm, 1.0 / lab, n=2)
    n0, _, z0_, _ = _budget(E, Fn, Fb, Zm)
    n1, _, z1_, _ = _budget(E, a, b, z)
    # the neutrinos: two per decayed mediator, less those of them below the grid (the beam's own daughters all on it)
    assert abs((n1 - n0 + diag[dmed.D_DAU_LOW]) / (2.0 * diag[dmed.D_DEC]) - 1.0) < 1e-9
    assert diag[dmed.D_MEDIUM] > 0.0 and diag[dmed.D_SIGMA] > 0.0
    assert np.all(a >= 0.0) and np.all(b >= 0.0) and np.all(z >= 0.0)


def test_mediators_decay_at_the_lab_rate():
    """A population of mediators alone decays at Gamma_Z m_Zd / E per unit decay time (the redshift: the steps'
    int dz / ((1+z)^2 H), any sequence of them), its two neutrinos per decay; also with the bin's own decays off
    (use_dec False: the neutrinos still arrive; the review's finding 7)."""
    E = np.geomspace(1e12, 1e16, 60)
    m = _masses(1e-6)
    st = _Step(E, 1e-5 * G_TAU, m, 1e-2)
    Zm = np.zeros((60, 3))
    Zm[40:] = 1.0
    zero = np.zeros((60, 3))
    rate = st.med.rate
    dts = np.array([0.3, 1.0, 0.1, 2.0]) * E[45] / rate
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    for use_dec in (True, False):
        a, b, z, diag = st.run(zero, zero, Zm, 1.0, n=4, decay_dts=dts, use_dec=use_dec)
        surv = np.exp(-rate / E[40:] * dts.sum())
        assert np.max(np.abs(z[40:] / Zm[40:] - surv[:, None])) < 1e-12
        dec = ((Zm - z).sum(axis=1) * dE).sum()
        assert abs(diag[dmed.D_DEC] / dec - 1.0) < 1e-12
        nb = ((a + b).sum(axis=1) * dE).sum()
        assert abs((nb + diag[dmed.D_DAU_LOW]) / (2.0 * dec * st.med.br_nu) - 1.0) < 1e-9


def test_nothing_changes_where_the_mediator_cannot_act():
    """step_full is step_full_med without the mediator, bit for bit; with it, a mediator that never decays leaves the
    beam as it was, bit for bit; dp_mediator.setup gives None where no pair is open (m_Zd <= 2 m_1) or the mediator
    cannot decay over the path."""
    E = np.geomspace(1e13, 1e16, 40)
    m = _masses(1e-6)
    st = _Step(E, 1e-6 * G_TAU, m, 1e-2)
    rng = np.random.default_rng(5)
    Fn = rng.uniform(0.5, 1, (40, 3)) * (E[:, None] / 1e14) ** -2
    Fb = rng.uniform(0.5, 1, (40, 3)) * (E[:, None] / 1e14) ** -2
    lab = float(np.max(dd.rest_rates(1e-6 * G_TAU, U, m, 1e-2)[0].sum(1) * m)) / E[0]
    a0, b0, _, _ = st.run(Fn, Fb, np.zeros((40, 3)), 0.5 / lab, n=2, use_med=False)
    a1, b1, z1, diag = st.run(Fn, Fb, np.zeros((40, 3)), 0.5 / lab, n=2, use_med=True, rate=0.0)
    assert np.array_equal(a0, a1) and np.array_equal(b0, b1)
    assert diag[dmed.D_PROD] > 0.0 and diag[dmed.D_DEC] == 0.0 and np.any(z1 > 0.0)
    H0 = 1.4e-33
    assert dmed.setup(1e-6 * G_TAU, U, _masses(0.01), 0.019, E, H0, 0.315, 0.685, 4.0) is None       # 2 m_1 = 0.02
    assert dmed.setup(1e-30 * G_TAU, U, m, 1e-2, E, H0, 0.315, 0.685, 4.0) is None                    # never decays
    assert dmed.setup(1e-6 * G_TAU, U, m, 1e-2, E, H0, 0.315, 0.685, 4.0) is not None


def _si_args(G, M):
    import dp_si
    E = np.geomspace(1e13, 1e16, 40)
    m = _masses(1e-6)
    g_mass = dp_si.mass_basis_couplings_model(G, U, "vector", True)
    decay = (dd.rest_rates(G, U, m, M)[0], dd.daughter_spectra(G, U, m, M))
    kw = dict(majorana=True, channels="full", mediator="vector", z_substeps=1, E_extension=3.0, decay=decay)
    return (E, 1.0, 1.44e-33, 0.315, 0.685, U, m, 1.68e-4, np.ones(3), np.array([1 / 3, 2 / 3, 0.0]), 2.5, 0.0,
            1.0, M, g_mass), kw


def test_si_transport_bookkeeping_and_cache():
    """dp_si's transport with the decay back: two neutrinos per decayed mediator and their energy the decayed one's;
    on a second call (the column cache) the same record (the review's finding 5); the mediator's own mass, not the
    transport's collision mass (finding 3): a different M with the same m_Zd leaves the tables' width alone."""
    import dp_si
    G, M = 1e-4 * G_TAU, 1e-2
    args, kw = _si_args(G, M)
    off = dp_si.propagate_si(*args, **kw)
    assert not dp_si.MEDIATOR_LAST
    on = dp_si.propagate_si(*args, decay_back=(G, 0.0, M), **kw)
    d = np.array(dp_si.MEDIATOR_LAST["diag"][0][0])
    assert d[dmed.D_DEC] > 0.0
    assert abs((d[dmed.D_DAU] + d[dmed.D_DAU_LOW]) / (2.0 * d[dmed.D_DEC]) - 1.0) < 1e-9
    assert abs((d[dmed.D_DAU_E] + d[dmed.D_DAU_LOW_E]) / d[dmed.D_DEC_E] - 1.0) < 1e-9
    assert np.all(on.sum(axis=1) >= off.sum(axis=1) * (1.0 - 1e-12))
    assert np.any(on.sum(axis=1) > off.sum(axis=1) * 1.01)
    on2 = dp_si.propagate_si(*args, decay_back=(G, 0.0, M), **kw)                   # cached
    assert np.array_equal(on, on2)
    np.testing.assert_allclose(dp_si.MEDIATOR_LAST["diag"][0][0], d, rtol=1e-12)
    dp_si.propagate_si(*args, **kw)
    assert not dp_si.MEDIATOR_LAST


def test_fast_cycles_converge_in_the_steps():
    """Where the bin's conversions and decays cycle fast (the dark portal, m_Zd = 1e-2 eV, g = 1e-4), the flux with
    the decay back converges in the redshift steps as the flux without it does (2026-11-08: with the mediators' own-bin
    neutrinos deferred to the step's end, dp_si's flux went as 1 / z_substeps -- an unrelaxed population ~ dt -- and
    on / off reached 330 instead of ~1.8). dp_si (its substeps 2 against 4) and the dark-portal phase (10 against 20
    configured redshift steps), small grids."""
    import dp_si
    G, M = 1e-4 * G_TAU, 1e-2
    args, kw = _si_args(G, M)
    r = {}
    for q in (2, 4):
        kq = dict(kw, z_substeps=q)
        off = dp_si.propagate_si(*args, **kq).sum(axis=1)
        on = dp_si.propagate_si(*args, decay_back=(G, 0.0, M), **kq).sum(axis=1)
        r[q] = on / off
    assert np.max(r[2]) < 5.0                                   # no runaway: the energy returned, not created
    assert np.max(np.abs(r[4] / r[2] - 1.0)) < 0.03
    vals = dict(m_Zd_eV=1e-2, g_D_trace=1e-4, m_lightest_eV=1e-6)
    rd = {}
    for nz in (10, 20):
        tot = {}
        for on in (False, True):
            E, p, a = _cfg_run("dp_tau_ul", 7, vals, dict(mediator_decay_back=on, cosmo_transport="dark_portal"),
                               dict(n_E_bins=40, n_z_bins=nz))
            tot[on] = sum(x.sum(axis=1) for x in a["P"])
        rd[nz] = tot[True] / tot[False]
    band = E >= 1e13
    assert np.max(rd[10][band]) < 5.0
    assert np.max(np.abs(rd[20][band] / rd[10][band] - 1.0)) < 0.03


def _cfg_run(plane, block, vals, dp=None, grid=None, arrival=True):
    import dark_portal_solver as dps
    from analysis.config import from_dict
    raw = json.load(open(os.path.join(CFG, f"{plane}__b{block}.json")))
    raw["solver"]["grid"].update(grid or {})
    raw["solver"]["galactic"]["enabled"] = False
    raw["solver"]["dark_portal"].update(dp or {})
    model = from_dict(raw, "").model()
    E, p = dps.build_solver_params(model._run_cfg(model._overrides(dict(vals))))
    p["show_progress"] = False
    setup = dps._flux_setup(E, p)
    return E, p, dps._flux_from_setup(E, p, setup, (0.5, 0.5), None, arrival_only=arrival)


def test_dp_si_against_dp_sweep():
    """The decay back implemented twice -- dp_si's sweep (one flux holding both kinds, mass states, window_plan's
    daughters, Mediator.fold()'s tables) and the dark-portal phase's (both kinds, the ladder's channels and
    daughters) -- over one step of the decays alone (dp_si's collision tables zero; vacuum; equal kinds), a beam and
    mediators already there, massive daughters (m_lightest 0.01 eV) and nearly massless ones (1e-6 eV), short and
    long steps: the neutrinos (both kinds) and the mediators (longitudinal; transverse summed: a folded flux does not
    tell +1 from -1) agree to the daughters' placements (dp_decay.daughter_spectra's fit against the ladder's closed form)."""
    import dp_si_light
    N = 120
    E = np.geomspace(1e12, 3e16, N)
    rng = np.random.default_rng(6)
    for ml, M in ((0.01, 0.03), (1e-6, 1e-2)):
        m = _masses(ml)
        G = 1e-6 * G_TAU
        med = dmed.Mediator(G, U, m, M, E)
        Gam, _ = dd.rest_rates(G, U, m, M)
        spec = dd.daughter_spectra(G, U, m, M)
        arr = dp_si_light.transport_arrays(Gam, spec, m, E, True)
        lab = float(np.max(Gam.sum(1) * m)) / E[N // 2]
        for f in (0.1, 3.0):
            dt = f / lab
            F0 = (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
            F0[:30] = 0.0
            Z0 = np.zeros((N, 3))
            Z0[60:] = (E[60:] ** -2.0)[:, None] * rng.uniform(0.1, 0.5, (N - 60, 3))
            # dp_si
            beam_si, zs, dsi = _si_kernel(E, F0, Z0, dt, med, arr, True)
            # the dark-portal phase: half of each mass state in each kind
            st = _Step(E, G, m, M)
            a, b, zw, dsw = st.run(0.5 * F0, 0.5 * F0, Z0, dt)
            beam_sw = a + b
            scale = np.max(np.abs(beam_sw - F0))
            assert np.max(np.abs(beam_si - beam_sw)) < 2e-6 * scale, (ml, M, f)
            assert np.max(np.abs(zs[:, 0] - zw[:, 0])) < 2e-6 * np.max(zw), (ml, M, f)
            assert np.max(np.abs(zs[:, 1:].sum(1) - zw[:, 1:].sum(1))) < 2e-6 * np.max(zw), (ml, M, f)
            assert abs(dsi[dmed.D_DEC] / dsw[dmed.D_DEC] - 1.0) < 2e-6


def _si_kernel(E, F0, Z0, dt, med, arr, exact, parts=1):
    """dp_si's sweep (_si_sweep_cols_med) over dt in `parts` steps, the decays alone (collision tables zero), a beam
    F0 (N, 3) and mediators Z0 (N, 3) already there; exact: exact_bin (the vector's exact bin solution) or the
    scalar's step-average scheme. Returns (F, Z, mediator diag)."""
    import dp_si
    N = len(E)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    n = nr = 3
    P = 1
    F = np.ascontiguousarray(F0[:, :, None].copy())
    Zs = np.ascontiguousarray(Z0[:, :, None].copy())
    fp, fnp, fps, fdec, fnd, fds, frate = med.fold()
    z3 = np.zeros((N, nr, P))
    dsi = np.zeros((1, dmed.NDIAG))
    for _ in range(parts):
        dp_si._si_sweep_cols_med(
            F, E, dE, np.zeros((N, n, 1)), np.zeros((N, nr, 4)), z3, z3.copy(), z3.copy(),
            np.zeros(P, np.int64), np.zeros(P), np.zeros(P, np.int64), np.zeros((P, n, nr, n)),
            np.zeros((P, n, nr, n)), np.zeros((nr, 2, P)), np.zeros((nr, 2), np.int64),
            np.zeros((N, nr), np.int64), dt / parts, np.zeros((N, n, 1)), np.zeros((1, 6)), exact,
            arr["rate"], arr["br"], arr["own"], arr["kfull"], arr["kex"], arr["fex"], arr["mom"], 1.0, True,
            True, Zs, fp, fnp, fps, np.ascontiguousarray(fdec[:, None]), fnd, np.ascontiguousarray(fds[:, None]),
            frate, np.zeros((N, 3, 1)), np.zeros((N, 1, 1, 3)), dsi)
    return F[:, :, 0], Zs[:, :, 0], dsi[0]


def test_dp_si_scalar_scheme_with_the_mediators():
    """dp_si's scalar scheme (not exact_bin; the Goldstone factors run the decay back through it) with the bin's
    mediators as three more populations, what lands in the bin held at its step average: two neutrinos per decayed
    mediator, their energy the decayed one's; against the exact solution of the same steps (decays alone, as above)
    within its first-order error, which falls with the step (2026-11-08: before, the scalar with the decay back raised
    a ValueError, which the objective turns into its wall)."""
    import dp_si_light
    N = 120
    E = np.geomspace(1e12, 3e16, N)
    rng = np.random.default_rng(6)
    for ml, M in ((0.01, 0.03), (1e-6, 1e-2)):
        m = _masses(ml)
        G = 1e-6 * G_TAU
        med = dmed.Mediator(G, U, m, M, E)
        Gam, _ = dd.rest_rates(G, U, m, M)
        arr = dp_si_light.transport_arrays(Gam, dd.daughter_spectra(G, U, m, M), m, E, True)
        lab = float(np.max(Gam.sum(1) * m)) / E[N // 2]
        for f in (0.1, 100.0):
            dt = f / lab
            F0 = (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
            F0[:30] = 0.0
            Z0 = np.zeros((N, 3))
            Z0[60:] = (E[60:] ** -2.0)[:, None] * rng.uniform(0.1, 0.5, (N - 60, 3))
            gap = {}
            for parts in (1, 4):
                Fe, Ze, de = _si_kernel(E, F0, Z0, dt, med, arr, True, parts)
                Fs, Zs, ds = _si_kernel(E, F0, Z0, dt, med, arr, False, parts)
                assert np.all(np.isfinite(Fs)) and np.all(np.isfinite(Zs))
                assert ds[dmed.D_DEC] > 0.0
                assert abs((ds[dmed.D_DAU] + ds[dmed.D_DAU_LOW]) / (2.0 * ds[dmed.D_DEC]) - 1.0) < 1e-12
                assert abs((ds[dmed.D_DAU_E] + ds[dmed.D_DAU_LOW_E]) / ds[dmed.D_DEC_E] - 1.0) < 1e-12
                scale = np.max(np.abs(Fe - F0))
                gap[parts] = (np.max(np.abs(Fs - Fe)) / scale, np.max(np.abs(Zs - Ze)) / np.max(Ze),
                              abs(ds[dmed.D_DEC] / de[dmed.D_DEC] - 1.0))
            assert gap[1][0] < 5e-3 and gap[1][1] < 3e-2 and gap[1][2] < 3e-2, (ml, M, f, gap)
            for x in range(3):
                assert gap[4][x] < 0.6 * gap[1][x] + 1e-9, (ml, M, f, gap)


def test_every_transport_runs_with_the_decay_back(monkeypatch):
    """With the decay back on (the default), the transports of the frozen reference's points run and give finite
    fluxes -- the dark-portal phase with decays, dp_si's transport, a potential, the Goldstone factors (2026-11-08: the
    last raised inside the objective before) -- the Goldstone factors computed with the decay back."""
    import dp_si_light
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import make_decay_back_reference as ref
    seen = []
    orig = dp_si_light.goldstone_factors

    def record(*a, **kw):
        out = orig(*a, **kw)
        seen.append((kw.get("decay_back") is not None, out))
        return out

    monkeypatch.setattr(dp_si_light, "goldstone_factors", record)
    want = np.load(os.path.join(os.path.dirname(__file__), "data", "decay_back_off_reference.npz"))
    for i, (plane, block, vals) in enumerate(ref.POINTS):
        seen.clear()
        _, _, a = _cfg_run(plane, block, vals, dict(mediator_decay_back=True), ref.GRID)
        on = (a["P"][0] + a["P"][1]).sum(axis=1)
        off = (want[f"p{i}_nu"] + want[f"p{i}_nb"]).sum(axis=1)
        assert np.all(np.isfinite(on)), (plane, vals)
        if vals.get("g_D") == 3e-4:                                        # the Goldstone factors' point
            assert seen and all(db for db, _ in seen) and seen[0][1] is not None
            assert np.all(np.isfinite(seen[0][1][0])) and np.all(np.isfinite(seen[0][1][1]))
            assert np.max(on / off) > 1.3


def test_off_is_the_code_before():
    """With dark_portal.mediator_decay_back False the transports are the code before the decay back (917dbff): a
    frozen reference of small transports (tools/make_decay_back_reference.py; the dark-portal phase with decays, dp_si's
    transport, a potential, the Goldstone factors). Bit for bit on the machine that made it; 1e-9 elsewhere (the
    compiled code's rounding may differ across CPUs and NumPy versions)."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import make_decay_back_reference as ref
    want = np.load(os.path.join(os.path.dirname(__file__), "data", "decay_back_off_reference.npz"))
    for i, (plane, block, vals) in enumerate(ref.POINTS):
        _, _, a = _cfg_run(plane, block, vals, dict(mediator_decay_back=False), ref.GRID)
        for kind, key in enumerate(("nu", "nb")):
            np.testing.assert_allclose(a["P"][kind], want[f"p{i}_{key}"], rtol=1e-9, atol=0.0)


def test_unsupported_fails_at_configuration():
    """What the decay back does not cover (Dirac neutrinos) raises DecayBackUnsupported when the run is configured --
    not a ValueError, which the analysis' objective turns into its wall (chi^2 = 1e9); the dark portal's configurations
    pass, with the setting on by default."""
    import dark_portal_solver as dps
    assert not issubclass(dmed.DecayBackUnsupported, ValueError)
    with pytest.raises(dmed.DecayBackUnsupported, match="mediator_decay_back = False"):
        dmed.check_supported(majorana=False)
    with pytest.raises(dmed.DecayBackUnsupported):
        dmed.Mediator(1e-6 * G_TAU, U, _masses(1e-6), 1e-2, np.geomspace(1e12, 1e16, 20), majorana=False)
    assert dps.mediator_decay_back_default() is True
    raw = json.load(open(os.path.join(CFG, "dp_tau_ul__b7.json")))
    from analysis.config import from_dict
    model = from_dict(raw, "").model()
    _, p = dps.build_solver_params(model._run_cfg(model._overrides(dict(m_Zd_eV=1e-2, g_D_trace=1e-5))))
    assert p["mediator_decay_back"] is True
