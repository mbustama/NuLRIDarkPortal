"""The decay back of the secret interactions' mediator below 1 keV (2026-11-08; dp_mediator.SIMediator, dp_si's sweep,
dp_si_light.subkev_inputs): the four vertices' widths and light-front densities against the traces and the analytic
forms (the scalar's isotropic decay, its same-helicity pairs, the Dirac kinds), the tables' number and energy per block,
the vector with Majorana neutrinos the dark portal's tables bit for bit, dp_si's sweep (two neutrinos per decayed
mediator, the energy kept, the Dirac visible and sterile daughters, the lab rate, the scalar's scheme against the exact
one), the setting off the code before (a frozen reference), the transport's steps."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dp_mediator as dmed  # noqa: E402
import dp_si_light as dsl  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

U = np.asarray(construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0)))
G_MU = np.zeros((3, 3), complex)
G_MU[1, 1] = 1.0
G_ANY = np.diag([0.3, 1.0, 0.5]).astype(complex) + 0.2 * (np.eye(3, k=1) + np.eye(3, k=-1))
MODELS = [("scalar", True), ("scalar", False), ("vector", True), ("vector", False)]
IDS = ["scalar-majorana", "scalar-dirac", "vector-majorana", "vector-dirac"]
_XG, _WG = np.polynomial.legendre.leggauss(40)
REPO = os.path.join(os.path.dirname(__file__), "..")
H0 = 1.4377e-33


def _masses(ml):
    return np.sqrt(np.array([0.0, 7.537e-5, 2.511e-3]) + ml * ml)


def _lams(med):
    return (None,) if med == "scalar" else (0, 1, -1)


@pytest.mark.parametrize("med,maj", MODELS, ids=IDS)
def test_widths_are_the_traces(med, maj):
    """Every pair's width (each vector LF helicity alone: rotational invariance) is the traces' integral, massive
    daughters, complex couplings, identical Majorana daughters at S = 1/2; massless: Tr(l l^dag) M / 16 pi (scalar,
    Majorana), Tr(h h^dag) M / 8 pi (scalar, Dirac), Tr(G G^dag) M / 24 pi (vector) -- dp_si's widths; k = l with a real
    coupling: the scalar's P wave g^2 M beta^3 / (16 pi) (Majorana), / (8 pi) (Dirac)."""
    import dp_si
    G = 1e-6 * G_ANY
    for ml, M in ((1e-6, 1e-2), (0.01, 0.035), (0.02, 0.09)):
        m = _masses(ml)
        W = dmed.si_pair_widths(G, U, m, M, med, maj)
        cpl = dmed.si_coupling(G, U, med, maj)
        assert np.any(W > 0.0)
        for k in range(3):
            for l in range(3):
                if W[k, l] == 0.0:
                    continue
                S = 0.5 if (maj and k == l) else 1.0
                p = dmed._p_star(M, m[k], m[l])
                for lam in _lams(med):
                    tot = sum(w * dmed.msq_decay_si(M, m[k], m[l], c, lam, h1, h2, cpl[k, l], med, maj)[0]
                              for h1 in (-1, 1) for h2 in (-1, 1) for c, w in zip(_XG, _WG))
                    assert abs(S * tot * p / (16.0 * np.pi * M * M) / W[k, l] - 1.0) < 1e-12
    m0 = np.array([1e-12, 2e-12, 3e-12])
    W = dmed.si_pair_widths(G, U, m0, 1e-2, med, maj)
    gm = dp_si.mass_basis_couplings_model(G, U, med, maj)
    assert abs(W.sum() / dp_si.mediator_width(1e-2, gm, med, maj) - 1.0) < 1e-12
    if med == "scalar":
        m = _masses(0.02)
        W = dmed.si_pair_widths(1e-6 * np.eye(3, dtype=complex), np.eye(3), m, 0.05, med, maj)
        beta = np.sqrt(1.0 - 4.0 * m[0] ** 2 / 0.05 ** 2)
        assert abs(W[0, 0] / (1e-12 * 0.05 * beta ** 3 / ((16.0 if maj else 8.0) * np.pi)) - 1.0) < 1e-12


@pytest.mark.parametrize("maj", (True, False), ids=("majorana", "dirac"))
def test_scalar_decays_isotropically(maj):
    """The scalar has no direction: summed over the daughters' helicities its decay is isotropic, each daughter's share
    uniform on its window (massive daughters included). Massless daughters come in same-helicity pairs only (J = 0),
    each half the time -- Majorana: two neutrinos or two antineutrinos; Dirac: nu(-1) nubar(-1) or nu(+1) nubar(+1), one
    visible daughter per decay; massive daughters also in opposite-helicity pairs."""
    m0 = np.array([1e-12, 2e-12, 3e-12])
    dens, W, res = dmed.si_decay_densities(1e-6 * G_ANY, U, m0, 1e-2, "scalar", maj)
    assert res < 1e-12
    for (k, l) in zip(*np.nonzero(W)):
        tot = sum(dens[(k, l, 0, hi)].total for hi in range(4))
        sh = np.array([dens[(k, l, 0, hi)].total / tot for hi in range(4)])
        np.testing.assert_allclose(sh, [0.5, 0.0, 0.0, 0.5], atol=1e-12)
    m = _masses(1e-6)
    E = np.geomspace(1e12, 1e16, 60)
    for G in (1e-6 * G_MU, 1e-6 * G_ANY):
        dens, W, res = dmed.si_decay_densities(G, U, m, 1e-2, "scalar", maj)
        seen_opposite = False
        for (k, l) in zip(*np.nonzero(W)):
            ds = [dens[(k, l, 0, hi)] for hi in range(4)]
            tot = sum(d.total for d in ds)
            x = np.linspace(ds[0].lo, ds[0].hi, 11)[1:-1]
            pdf = sum(d.pdf(x) for d in ds if d.total > 0.0) / tot
            np.testing.assert_allclose(pdf * (ds[0].hi - ds[0].lo), 1.0, rtol=1e-10)
            seen_opposite |= (ds[1].total + ds[2].total) > 0.1 * tot
        assert seen_opposite                                # nu_1 nu_2 with m_2 = 8.7e-3 eV, M = 1e-2 eV
        # the tables: the daughters' kinds as the densities say (Dirac: visible and sterile)
        S = dmed.SIMediator(G, U, m, 1e-2, E, "scalar", maj)
        num = np.zeros((2, 3))
        for (k, l) in zip(*np.nonzero(W)):
            tot = sum(dens[(k, l, 0, hi)].total for hi in range(4))
            for hi in range(4):
                w = W[k, l] / S.Gamma * dens[(k, l, 0, hi)].total / tot
                if not dens[(k, l, 0, hi)].total > 1e-14 * tot:
                    continue
                num[hi // 2, k] += w
                num[(hi % 2) if maj else 1 - hi % 2, l] += w
        np.testing.assert_allclose(S.dec[0].sum(axis=-1), num, atol=1e-12)
        assert not np.any(S.dec[1:])                        # the scalar's one population


@pytest.mark.parametrize("med,maj", MODELS, ids=IDS)
def test_production_adds_up_to_daughter_spectra(med, maj):
    """The populations' densities of the daughter's share add up to dp_si_light.daughter_spectra's a + b z + c / z
    (each rel's share of the pair's rate; the Dirac vector's fit there is good to ~1e-7)."""
    for ml, M in ((1e-6, 1e-2), (0.01, 0.03), (1e-6, 1e-4)):
        m = _masses(ml)
        G = 1e-6 * G_MU
        spec = dsl.daughter_spectra(G, U, m, M, med, maj)
        dens, res = dmed.si_production_densities(G, U, m, M, med, maj)
        assert res < 1e-9
        npol = dmed.si_npol(med)
        seen = 0
        for (s_, i, j, rel, li) in dens:
            if rel or li:
                continue
            tot = sum(dens[(s_, i, j, r, x)].total for r in range(2) for x in range(npol))
            for r in range(2):
                zm, zp, a, b, c = spec[s_, i, j, r]
                z = np.linspace(zm, zp, 9)[1:-1]
                got = sum(dens[(s_, i, j, r, x)].pdf(z) for x in range(npol)) / tot
                ref = a + b * z + c / z
                assert np.max(np.abs(got - ref)) < 1e-6 * np.max(np.abs(ref)), (s_, i, j, r)
            seen += 1
        assert seen > 0


def _check_si_tables(S, G, m, M, med, maj, E):
    """Every block: a decay's neutrinos of each kind and mass state hold their exact share and continuous energy (two
    neutrinos and the mediator's energy in all); a production row's populations their exact shares, their energy the
    parent's less the daughter's box table; dp_si's (a, c) rows one mediator per decay wherever transport_arrays has
    that decay; no negative entry."""
    h, N = S.h, S.N
    K1 = N + 2
    e = h ** (-2.0 * np.arange(K1))
    ddens, W, _ = dmed.si_decay_densities(G, U, m, M, med, maj)
    for li in range(S.npol):
        assert np.all(S.dec[li] >= 0.0)
        assert abs(S.dec[li].sum() - 2.0) < 1e-12
        assert abs(np.sum(S.dec[li] * e) - 1.0) < 1e-12
        num, ene = np.zeros((2, 3)), np.zeros((2, 3))
        for (k, l) in zip(*np.nonzero(W)):
            tot = sum(ddens[(k, l, li, hi)].total for hi in range(4) if ddens[(k, l, li, hi)].total > 0.0)
            for hi in range(4):
                d = ddens[(k, l, li, hi)]
                if not d.total > 1e-14 * tot:
                    continue
                w = W[k, l] / S.Gamma * d.total / tot
                k2 = hi % 2 if maj else 1 - hi % 2
                num[hi // 2, k] += w
                num[k2, l] += w
                ene[hi // 2, k] += w * d.mean
                ene[k2, l] += w * (1.0 - d.mean)
        assert np.max(np.abs(S.dec[li].sum(axis=-1) - num)) < 1e-12
        assert np.max(np.abs(np.sum(S.dec[li] * e, axis=-1) - ene)) < 1e-12
    pdens, _ = dmed.si_production_densities(G, U, m, M, med, maj)
    for (s_, i, j, rel) in zip(*np.nonzero(S.have)):
        row = S.rows[s_, i, j, rel]
        assert np.all(row >= 0.0)
        assert abs(row.sum() - 1.0) < 1e-12
        ds = [pdens[(s_, i, j, rel, li)] for li in range(S.npol)]
        tot = sum(d.total for d in ds if d.total > 1e-14 * max(x.total for x in ds))
        dcdf = lambda z: sum(d.total * d.cdf(z) for d in ds if d.total > 1e-14 * tot) / tot  # noqa: E731
        dtab = dmed._table(dcdf, ds[0].lo, ds[0].hi, h, N, K1)
        assert abs(np.sum(row * e) + np.sum(dtab * e) - 1.0) < 1e-12
        for li, d in enumerate(ds):
            share = d.total / tot if d.total > 1e-14 * tot else 0.0
            assert abs(row[li].sum() - share) < 1e-12
    prod = S.arrays[0]
    Gam = dsl.rest_rates(G, U, m, M, med, maj)
    arr = dsl.transport_arrays(Gam, dsl.daughter_spectra(G, U, m, M, med, maj), m, E, maj)
    for a in range(S.n):
        for c in range(S.n):
            if arr["br"][a, c] > 1e-12:
                assert abs(prod[a, c].sum() - 1.0) < 1e-12, (a, c)


@pytest.mark.parametrize("med,maj", MODELS, ids=IDS)
def test_tables_keep_number_and_energy_per_block(med, maj):
    """The tables per block (_check_si_tables) at the target regime and near the kinematic thresholds."""
    E = np.geomspace(1e12, 3e16, 200)
    pts = [(1e-6, 1e-2, 1e-5 * G_MU), (0.01, 0.035, 1e-6 * G_ANY), (1e-6, 1e-3, 1e-6 * G_MU)]
    m = _masses(1e-6)
    for rel in (1e-6, 1e-10):
        pts += [(1e-6, (m[0] + m[1]) * (1 + rel), 1e-6 * G_MU), (1e-6, (m[2] - m[1]) * (1 - rel), 1e-6 * G_ANY)]
    m = _masses(0.01)
    pts += [(0.01, 2 * m[0] * (1 + 1e-9), 1e-6 * G_ANY)]
    for ml, M, G in pts:
        m = _masses(ml)
        S = dmed.SIMediator(G, U, m, M, E, med, maj)
        if S.Gamma <= 0.0:
            continue
        _check_si_tables(S, G, m, M, med, maj, E)


def test_vector_majorana_is_the_dark_portals():
    """The secret interactions' vector with Majorana neutrinos has the dark portal's vertex: its tables are
    Mediator.fold()'s, bit for bit."""
    E = np.geomspace(1e12, 3e16, 150)
    for ml, M, G in ((1e-6, 1e-2, 2e-5 * G_MU), (0.01, 0.035, 1e-6 * G_ANY)):
        m = _masses(ml)
        a = dmed.SIMediator(G, U, m, M, E, "vector", True).arrays
        b = dmed.Mediator(G, U, m, M, E).fold()
        for x, y in zip(a, b):
            if isinstance(x, np.ndarray):
                assert np.array_equal(np.squeeze(x), np.squeeze(y))
            else:
                assert x == y


def _kernel(E, F0, Z0, dt, tabs, arr, exact, n, parts=1):
    """dp_si's sweep (_si_sweep_cols_med) over dt in `parts` steps, the decays alone (no collisions), a beam F0 (N, n)
    and mediators Z0 (N, 3) already there; exact: the vector's exact bin solution, else the scalar's step-average
    scheme. Returns (F, Z, the mediators' diag)."""
    import dp_si
    N = len(E)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    nr, P = 3, 1
    F = np.ascontiguousarray(F0[:, :, None].copy())
    Zs = np.ascontiguousarray(Z0[:, :, None].copy())
    fp, fnp, fps, fdec, fnd, fds, frate = tabs
    z3 = np.zeros((N, nr, P))
    dsi = np.zeros((1, dmed.NDIAG))
    for _ in range(parts):
        dp_si._si_sweep_cols_med(
            F, E, dE, np.zeros((N, n, 1)), np.zeros((N, nr, 4)), z3, z3.copy(), z3.copy(),
            np.zeros(P, np.int64), np.zeros(P), np.zeros(P, np.int64), np.zeros((P, n, nr, n)),
            np.zeros((P, n, nr, n)), np.zeros((nr, 2, P)), np.zeros((nr, 2), np.int64),
            np.zeros((N, nr), np.int64), dt / parts, np.zeros((N, n, 1)), np.zeros((1, 6)), exact,
            arr["rate"], arr["br"], arr["own"], arr["kfull"], arr["kex"], arr["fex"], arr["mom"], 1.0, True,
            True, Zs, fp, fnp, fps, fdec, fnd, fds, frate, np.zeros((N, 3, 1)),
            np.zeros((N, 1, fdec.shape[1], 3)), dsi)
    return F[:, :, 0], Zs[:, :, 0], dsi[0]


def _setup_kernel(med, maj, ml, M, E, G=1e-6 * G_MU):
    m = _masses(ml)
    S = dmed.SIMediator(G, U, m, M, E, med, maj)
    Gam = dsl.rest_rates(G, U, m, M, med, maj)
    arr = dsl.transport_arrays(Gam, dsl.daughter_spectra(G, U, m, M, med, maj), m, E, maj)
    lab = float(np.max(Gam.sum(1) * m)) / E[len(E) // 2]
    return S, arr, lab


@pytest.mark.parametrize("med,maj", MODELS, ids=IDS)
def test_sweep_keeps_number_and_energy(med, maj):
    """dp_si's sweep with the decay back, the decays alone, short and long steps, both schemes: two neutrinos per
    decayed mediator and their energy the decayed one's (to rounding); massive daughters (m_lightest 0.01 eV: every
    daughter on the grid) -- the neutrinos gain two per decayed mediator, and the energy of the beam, the mediators and
    what falls below the grid is kept to rounding; nearly massless ones (1e-6 eV), to the beam's own daughters below
    the grid (<= 3e-8); no entry below -1e-13 of the largest (the beam's daughters' placement rounds)."""
    N = 120
    E = np.geomspace(1e12, 3e16, N)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    rng = np.random.default_rng(6)
    n = 3 if maj else 6
    for ml, M, tol in ((0.01, 0.03, 1e-13), (1e-6, 1e-2, 3e-8)):
        S, arr, lab = _setup_kernel(med, maj, ml, M, E)
        for f in (0.1, 30.0):
            F0 = np.zeros((N, n))
            F0[:, :3] = (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
            if not maj:
                F0[:, 3:] = 0.3 * (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
            F0[:70] = 0.0
            Z0 = np.zeros((N, 3))
            Z0[80:, :S.npol] = (E[80:] ** -2.0)[:, None] * rng.uniform(0.1, 0.5, (N - 80, S.npol))
            e0 = ((F0.sum(1) + Z0.sum(1)) * E * dE).sum()
            n0 = (F0.sum(1) * dE).sum()
            for exact in (True, False):
                F, Z, d = _kernel(E, F0, Z0, f / lab, S.arrays, arr, exact, n)
                assert d[dmed.D_DEC] > 0.0 and d[dmed.D_PROD] > 0.0
                assert abs((d[dmed.D_DAU] + d[dmed.D_DAU_LOW]) / (2.0 * d[dmed.D_DEC]) - 1.0) < 1e-12
                assert abs((d[dmed.D_DAU_E] + d[dmed.D_DAU_LOW_E]) / d[dmed.D_DEC_E] - 1.0) < 1e-12
                e1 = ((F.sum(1) + Z.sum(1)) * E * dE).sum() + d[dmed.D_PROD_LOW_E] + d[dmed.D_DAU_LOW_E]
                assert abs(e1 / e0 - 1.0) < tol, (ml, f, exact, e1 / e0 - 1.0)
                if ml == 0.01:
                    n1 = (F.sum(1) * dE).sum()
                    assert abs((n1 - n0 + d[dmed.D_DAU_LOW]) / (2.0 * d[dmed.D_DEC]) - 1.0) < 1e-10
                assert F.min() > -1e-13 * F.max() and Z.min() >= 0.0
                assert not np.any(Z[:, S.npol:])


@pytest.mark.parametrize("med", ("scalar", "vector"))
def test_dirac_visible_and_sterile_daughters(med):
    """Dirac neutrinos: the mediators' daughters land in dp_si's visible (states 0-2) and sterile (3-5) populations by
    the tables' kinds -- massless daughters (m_lightest 1e-9 eV, M = 5e-3 eV: nu_1 nubar_1 alone): the scalar one
    visible and one sterile per decay, the vector two visible (P_L); massive ones: the tables' shares (m_lightest 1e-6
    eV, M = 1e-2 eV: nu_1 nubar_2 with m_2 = 8.7e-3 eV), the two kinds adding up to two per decay."""
    N = 100
    E = np.geomspace(1e12, 3e16, N)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    for ml, M in ((1e-9, 5e-3), (1e-6, 1e-2)):
        S, arr, lab = _setup_kernel(med, False, ml, M, E)
        Z0 = np.zeros((N, 3))
        Z0[60:, :S.npol] = (E[60:] ** -2.0)[:, None]
        rate = S.rate
        F, Z, d = _kernel(E, np.zeros((N, 6)), Z0, 5.0 * E[80] / rate, S.arrays, arr, True, 6)
        vis = (F[:, :3].sum(1) * dE).sum()
        ste = (F[:, 3:].sum(1) * dE).sum()
        low = d[dmed.D_DAU_LOW]
        assert abs((vis + ste + low) / (2.0 * d[dmed.D_DEC]) - 1.0) < 1e-12
        # the visible share of the daughters (per decayed mediator, by population, on and below the grid)
        D = ((Z0 - Z) * dE[:, None]).sum(axis=0)                       # decayed per population
        want_vis = sum(D[li] * S.dec[li, 0].sum() for li in range(S.npol)) / D.sum()
        got_vis = vis / d[dmed.D_DEC]
        if ml == 1e-9:
            assert abs(want_vis - (1.0 if med == "scalar" else 2.0)) < 1e-6
        assert abs(got_vis - want_vis) < 1e-3 * want_vis + 2.0 * low / d[dmed.D_DEC]
        assert ste > 0.0 if (med == "scalar" or ml == 1e-6) else ste < 1e-6 * vis


@pytest.mark.parametrize("med,maj", MODELS, ids=IDS)
def test_mediators_decay_at_the_lab_rate(med, maj):
    """A population of mediators alone decays at Gamma M / E per unit decay time, in both schemes, its two neutrinos
    per decay."""
    N = 60
    E = np.geomspace(1e12, 1e16, N)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    S, arr, _ = _setup_kernel(med, maj, 1e-6, 1e-2, E, 1e-5 * G_MU)
    n = 3 if maj else 6
    Z0 = np.zeros((N, 3))
    Z0[40:, :S.npol] = 1.0
    dt = 0.7 * E[45] / S.rate
    surv = np.exp(-S.rate / E[40:] * dt)
    for exact in (True, False):
        F, Z, d = _kernel(E, np.zeros((N, n)), Z0, dt, S.arrays, arr, exact, n, parts=3)
        # (the exact scheme's matrix exponential rounds to ~1e-12 over three steps)
        assert np.max(np.abs(Z[40:, :S.npol] / Z0[40:, :S.npol] - surv[:, None])) < 1e-11
        dec = ((Z0 - Z).sum(axis=1) * dE).sum()
        assert abs(d[dmed.D_DEC] / dec - 1.0) < 1e-12
        nb = (F.sum(axis=1) * dE).sum()
        assert abs((nb + d[dmed.D_DAU_LOW]) / (2.0 * dec) - 1.0) < 1e-10


@pytest.mark.parametrize("maj", (True, False), ids=("majorana", "dirac"))
def test_scalar_scheme_against_the_exact_one(maj):
    """The scalar's transport (its step-average scheme, the mediator one more population) against the exact solution
    of the same steps, the decays alone: within its first-order error, which falls with the step (one step of 100
    decay lengths: the beam 2-3e-3 of its change, the mediators 4.5-5.5%, 1.7-2% in four steps)."""
    N = 120
    E = np.geomspace(1e12, 3e16, N)
    rng = np.random.default_rng(8)
    n = 3 if maj else 6
    for ml, M in ((0.01, 0.03), (1e-6, 1e-2)):
        S, arr, lab = _setup_kernel("scalar", maj, ml, M, E)
        for f in (0.1, 100.0):
            F0 = np.zeros((N, n))
            F0[:, :3] = (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
            F0[:30] = 0.0
            Z0 = np.zeros((N, 3))
            Z0[60:, 0] = (E[60:] ** -2.0) * rng.uniform(0.1, 0.5, N - 60)
            gap = {}
            for parts in (1, 4):
                Fe, Ze, de = _kernel(E, F0, Z0, f / lab, S.arrays, arr, True, n, parts)
                Fs, Zs, ds = _kernel(E, F0, Z0, f / lab, S.arrays, arr, False, n, parts)
                scale = np.max(np.abs(Fe - F0))
                gap[parts] = (np.max(np.abs(Fs - Fe)) / scale, np.max(np.abs(Zs - Ze)) / np.max(Ze),
                              abs(ds[dmed.D_DEC] / de[dmed.D_DEC] - 1.0))
            assert gap[1][0] < 5e-3 and gap[1][1] < 8e-2 and gap[1][2] < 3e-2, (ml, M, f, gap)
            for x in range(3):
                assert gap[4][x] < 0.6 * gap[1][x] + 1e-9, (ml, M, f, gap)


def test_subkev_inputs_gate_the_decay_back():
    """subkev_inputs gives dp_si its decay back where it acts -- decays, a pair open, the mediator's decay exponent over
    the path >= DECAY_BACK_MIN -- and None elsewhere (then the transport is the code before, bit for bit) and with the
    setting off; the default is dp_config's dark_portal.mediator_decay_back (on)."""
    import dp_config
    assert dp_config.DEFAULT_CONFIG["dark_portal"]["mediator_decay_back"] is True
    E0 = 1e12

    def sub(G, ml, M, med="scalar", maj=True, **kw):
        return dsl.subkev_inputs(G, U, _masses(ml), M, med, maj, 1.676e-4, np.ones(3), H0, 0.315, 0.685, 4.0, E0, **kw)

    for med, maj in MODELS:
        s = sub(1e-5 * G_MU, 1e-6, 1e-2, med, maj)
        assert s["decay"] is not None and s["decay_back"] is not None
        assert s["decay_back"][3] == med and s["decay_back"][4] is maj and s["decay_back"][1] == 0.0
        assert s["meta"]["mediator_path_exponent"] >= dsl.DECAY_BACK_MIN
        assert sub(1e-5 * G_MU, 1e-6, 1e-2, med, maj, decay_back=False)["decay_back"] is None
        assert sub(1e-5 * G_MU, 1e-6, 1e-2, med, maj, decay_on=False)["decay_back"] is None
    assert sub(1e-5 * G_MU, 0.01, 0.019)["decay_back"] is None              # 2 m_1 = 0.02 eV: no pair
    assert sub(1e-5 * G_MU, 1e-6, 2e-6 * 0.999)["decay_back"] is None       # below 2 m_1
    s = sub(1e-11 * G_MU, 1e-6, 1e-2)                                       # decays, the mediator ~never decays
    assert s["decay"] is not None and s["decay_back"] is None
    assert s["meta"]["mediator_path_exponent"] < dsl.DECAY_BACK_MIN
    assert sub(1e-14 * G_MU, 1e-6, 1e-2)["decay"] is None


def _si_cfg_flux(plane, vals, dp=None, grid=None, source=(1 / 3, 2 / 3, 0.0)):
    import dark_portal_solver as dps
    from analysis.config import from_dict
    raw = json.load(open(os.path.join(REPO, "analysis", "configs", "campaign_si", "now_events", f"{plane}.json")))
    raw["solver"]["grid"].update(grid or {})
    raw["solver"]["galactic"]["enabled"] = False
    raw["solver"].setdefault("dark_portal", {}).update(dp or {})
    model = from_dict(raw, "").model()
    E, p = dps.build_solver_params(model._run_cfg(model._overrides(dict(vals))))
    p["show_progress"] = False
    setup = dps._flux_setup(E, p)
    return E, dps._flux_from_setup(E, p, setup, (0.5, 0.5), np.asarray(source, dtype=np.float64))


def test_off_is_the_code_before():
    """With dark_portal.mediator_decay_back False the secret interactions below 1 keV are the code before the
    extension (2026-11-08): a frozen reference of five small transports (tools/make_decay_back_reference.py --si; the
    four models at m_phi = 1e-2 eV, g = 1e-5, and the scalar at 1e-3 eV), bit for bit on the machine that made it,
    1e-9 elsewhere. With the default (on) every one runs, finite, and the four at 1e-2 eV move by more than 3%."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import make_decay_back_reference as ref
    want = np.load(os.path.join(os.path.dirname(__file__), "data", "si_decay_back_off_reference.npz"))
    for i, (plane, vals) in enumerate(ref.SI_POINTS):
        _, off = _si_cfg_flux(plane, vals, dict(mediator_decay_back=False), ref.SI_GRID, ref.SI_SOURCE)
        for kind, key in enumerate(("nu", "nb")):
            np.testing.assert_allclose(off[kind], want[f"s{i}_{key}"], rtol=1e-9, atol=0.0)
        _, on = _si_cfg_flux(plane, vals, None, ref.SI_GRID, ref.SI_SOURCE)
        t_on, t_off = (on[0] + on[1]).sum(1), (want[f"s{i}_nu"] + want[f"s{i}_nb"]).sum(1)
        assert np.all(np.isfinite(t_on)) and np.all(t_on >= t_off * (1.0 - 1e-9))
        if vals["m_phi_eV"] == 1e-2:
            assert np.max(t_on / t_off) > 1.03, (plane, np.max(t_on / t_off))


def test_transport_converges_in_the_steps():
    """dp_si's transport with the decay back (m_phi = 1e-2 eV, g = 1e-6, the four models, a 60-bin grid): the on / off
    ratio with 4 redshift substeps per bin against 2 within 1e-3 (the full grid's convergence: docs item 22 (f))."""
    import dp_si
    m = _masses(1e-6)
    E = np.geomspace(1e12, 3e16, 60)
    for med, maj in MODELS:
        G = 1e-6 * G_MU
        s = dsl.subkev_inputs(G, U, m, 1e-2, med, maj, 1.676e-4, np.ones(3), H0, 0.315, 0.685, 4.0, E[0])
        gm = dp_si.mass_basis_couplings_model(G, U, med, maj)
        args = (E, 4.0, H0, 0.315, 0.685, U, m, 1.676e-4, s["eta"], np.array([1 / 3, 2 / 3, 0.0]), 2.735, 0.0, 1.0,
                s["M_coll"], gm)
        r = {}
        for q in (2, 4):
            kw = dict(majorana=maj, mediator=med, z_substeps=q, E_extension=10.0, decay=s["decay"],
                      relic_loss=s["relic_loss"], thermal="full")
            off = dp_si.propagate_si(*args, **kw).sum(1)
            on = dp_si.propagate_si(*args, decay_back=s["decay_back"], **kw).sum(1)
            assert np.all(on >= off * (1.0 - 1e-12))
            r[q] = on / off
        assert np.max(r[2]) > 1.01
        assert np.max(np.abs(r[4] / r[2] - 1.0)) < 1e-3, (med, maj)
