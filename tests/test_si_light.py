"""dp_si_light: the secret interactions below 1 keV (2026-10-25; notes, "Secret interactions below 1 keV")."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_decay as dd  # noqa: E402
import dp_si  # noqa: E402
import dp_si_light as L  # noqa: E402
import dp_unitarity  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

U = construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0))
G_RAND = 1e-8 * np.array([[0.3, 1, 0.5], [1, 0.2, 0.7], [0.5, 0.7, -0.4]])
MODELS = [("scalar", True), ("scalar", False), ("vector", True), ("vector", False)]
H0 = 1.4377e-33
T0 = 1.676e-4


def _m(ml):
    return np.sqrt(np.array([0.0, 7.5e-5, 2.5e-3]) + ml * ml)


@pytest.mark.parametrize("med,maj", MODELS)
def test_spectra_add_up_to_the_rest_rates(med, maj):
    """Each parent helicity's kept and flipped spectra integrate to the closed-form rest rate (the total does not depend
    on the spin), the fit a + b z + c/z is exact at other angles, for light and quasi-degenerate masses."""
    for ml, M in ((1e-6, 1e-4), (1e-6, 0.03), (0.1, 1e-3), (0.1, 0.008), (1e-10, 1e-12)):
        m = _m(ml)
        Gam = L.rest_rates(G_RAND, U, m, M, med, maj)
        sp, tot = L.daughter_spectra(G_RAND, U, m, M, med, maj, return_totals=True)
        lam = L.mass_couplings(G_RAND, U, med, maj)
        assert np.any(Gam > 0)
        for i in range(3):
            for j in range(3):
                if Gam[i, j] == 0.0:
                    assert not np.any(sp[:, i, j])
                    continue
                assert np.allclose(tot[:, i, j], Gam[i, j], rtol=1e-12)
                mi, mj = m[i], m[j]
                pf = L._p_star(mi, mj, M)
                Ej = np.sqrt(pf * pf + mj * mj)
                emp = mj * mj / (Ej + pf)
                V = L._vertex(lam[j, i], med, maj)
                for s_, h in ((0, -1.0), (1, 1.0)):
                    for rel, hp in ((0, h), (1, -h)):
                        zm, zp, a, b, c = sp[s_, i, j, rel]
                        for cc in (-0.77, 0.123, 0.9):
                            f = L._msq(np.array([mi, 0, 0, 0.0]), np.array([Ej, pf * np.sqrt(1 - cc * cc), 0, pf * cc]),
                                       mi, mj, M, h, hp, V, med, emp + pf * (1 + cc)) / (16 * np.pi * mi)
                            z = (emp + pf * (1 + cc)) / mi
                            fit = (a + b * z + (c / z if c else 0.0)) * tot[s_, i, j]
                            assert abs(fit - f) < 1e-10 * tot[s_, i, j] / (zp - zm)


def test_massless_limits():
    """Nearly massless daughter and mediator: the scalar keeps the helicity with 2z and flips it with 2(1-z) (both
    kinds, Majorana and Dirac); the Dirac vector's visible parent keeps it with 2z (the longitudinal part) and never
    flips, its sterile parent flips into a visible daughter with 2(1-z); the vector with Majorana neutrinos is
    dp_decay's."""
    m0 = np.array([1e-9, 8.68e-3, 0.0501])
    G1 = 1e-8 * np.diag([0.0, 1.0, 0.0]).astype(complex)
    for maj in (True, False):
        sp = L.daughter_spectra(G1, U, m0, 1e-9, "scalar", maj)
        for s_ in range(2):
            assert np.allclose(sp[s_, 2, 0, 0, 2:], [0.0, 1.0, 0.0], atol=1e-6)
            assert np.allclose(sp[s_, 2, 0, 1, 2:], [1.0, -1.0, 0.0], atol=1e-6)
    sp = L.daughter_spectra(G1, U, m0, 1e-9, "vector", False)
    assert np.allclose(sp[0, 2, 0, 0, 2:], [0.0, 2.0, 0.0], atol=1e-6)
    assert np.allclose(sp[0, 2, 0, 1, 2:], 0.0, atol=1e-6)
    assert np.allclose(sp[1, 2, 0, 1, 2:], [2.0, -2.0, 0.0], atol=1e-6)
    m = _m(1e-6)
    assert np.array_equal(L.daughter_spectra(G_RAND, U, m, 1e-4, "vector", True), dd.daughter_spectra(G_RAND, U, m, 1e-4))
    assert np.array_equal(L.rest_rates(G_RAND, U, m, 1e-4, "vector", True), dd.rest_rates(G_RAND, U, m, 1e-4)[0])


def test_majorana_arrays_are_dp_si_s_earlier_ones():
    """transport_arrays for Majorana neutrinos: bit for bit the arrays dp_si assembled before (the dark-portal model's
    runs through dp_si are unchanged)."""
    m = _m(1e-6)
    G = 1e-9 * np.array([[0.3, 1, 0.5], [1, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    Gam, _ = dd.rest_rates(G, U, m, 1e-4)
    spec = dd.daughter_spectra(G, U, m, 1e-4)
    E = 1e12 * np.exp(0.02 * np.arange(300))
    arr = L.transport_arrays(Gam, spec, m, E, True)
    tot = Gam.sum(1)
    spf = np.zeros_like(spec)
    spf[0, :, :, 0, :2] = spec[0, :, :, 0, :2]
    spf[0, :, :, 0, 2:] = spec[0, :, :, 0, 2:] + spec[0, :, :, 1, 2:]
    kf = np.zeros((2, 3, 3, 2, 2), np.int64)
    kx = np.zeros((2, 3, 3, 2, dd.NSLOT), np.int64)
    fx = np.zeros((2, 3, 3, 2, dd.NSLOT))
    dd.window_plan(spf, float(np.sqrt(E[1] / E[0])), len(E), kf, kx, fx)
    assert np.array_equal(arr["rate"], np.where(tot > 0, tot * m, 0.0))
    assert np.array_equal(arr["br"], np.divide(Gam, tot[:, None], out=np.zeros((3, 3)), where=tot[:, None] > 0))
    assert np.array_equal(arr["kfull"], kf[0, :, :, 0]) and np.array_equal(arr["kex"], kx[0, :, :, 0])
    assert np.array_equal(arr["fex"], fx[0, :, :, 0])
    assert np.array_equal(arr["own"], np.where(kx[0, :, :, 0] == 0, fx[0, :, :, 0], 0.0).sum(-1))
    assert np.array_equal(arr["mom"], np.stack([spf[0, :, :, 0, 4], spf[0, :, :, 0, 2], spf[0, :, :, 0, 3]], -1))


@pytest.mark.parametrize("med,maj", MODELS)
def test_segment_keeps_the_number(med, maj):
    """_segment: decay chains (nu_3 -> nu_2 -> nu_1, the Dirac sterile states) and conversions keep the number of
    neutrinos (none falls below the grid here), for decays slow, comparable and fast against the segment."""
    m = np.array([1e-6, 8.68e-3, 0.0501])
    E = np.geomspace(1e2, 1e17, 600)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    gf = dp_si.flavor_couplings("mu", 1e-4, None)
    Gam = L.rest_rates(gf, U, m, 1e-3, med, maj)
    arr = L.transport_arrays(Gam, L.daughter_spectra(gf, U, m, 1e-3, med, maj), m, E, maj)
    n = 3 if maj else 6
    rng = np.random.default_rng(5)
    for scale in (1e-2, 1.0, 1e6):
        lam = np.ascontiguousarray(scale * arr["rate"][None, :] / E[:, None] * E[500] / max(arr["rate"]))
        Rc = np.zeros((len(E), n, n))
        # conversions among the visible states (not with the fastest decays: there the conversion-decay cycles
        # drain the flux below the grid, as they should)
        Rc[:, 0, 1] = Rc[:, 1, 0] = 0.3 * scale if scale <= 1.0 else 0.0
        F = np.zeros((len(E), n, 1))
        S = np.zeros((len(E), n, 1))
        S[500:505, :3, 0] = rng.uniform(0.5, 1.0, (5, 3)) / dE[500:505, None]
        L._segment(F, E, dE, S, 1.0, lam, Rc, arr["br"], arr["own"], arr["kfull"], arr["kex"], arr["fex"],
                   arr["mom"], True)
        num = np.sum(F[:, :, 0] * dE[:, None])
        inj = np.sum(S[:, :, 0] * dE[:, None])
        assert abs(num / inj - 1.0) < 1e-7, (scale, num / inj)


def test_segment_matches_dp_si_s_sweep():
    """The Galactic segment and dp_si's sweep with every collision off solve the same decays (a constant source over
    a step, the daughters of the bins above held constant): the same populations (Majorana, one flux)."""
    m = _m(1e-6)
    G = 1e-9 * np.array([[0.3, 1, 0.5], [1, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    N = 120
    E = np.geomspace(1e12, 3e16, N)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    Gam, _ = dd.rest_rates(G, U, m, 1e-4)
    arr = L.transport_arrays(Gam, dd.daughter_spectra(G, U, m, 1e-4), m, E, True)
    arr["rate"] = arr["rate"] * 3.0 / np.max(arr["rate"] / E[0])
    rng = np.random.default_rng(1)
    S0 = (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
    F0 = (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
    Fs = np.ascontiguousarray(F0[:, :, None].copy())
    nr = P = 1
    z3 = np.zeros((N, nr, P))
    dp_si._si_sweep_cols(Fs, E, dE, np.ascontiguousarray(S0[:, :, None]), np.zeros((N, nr, 4)), z3, z3.copy(),
                         z3.copy(), np.zeros(P, np.int64), np.zeros(P), np.zeros(P, np.int64), np.zeros((P, 3, nr, 3)),
                         np.zeros((P, 3, nr, 3)), np.zeros((nr, 2, P)), np.zeros((nr, 2), np.int64),
                         np.zeros((N, nr), np.int64), 1.0, np.zeros((N, 3, 1)), np.zeros((1, 6)), True, arr["rate"],
                         arr["br"], arr["own"], arr["kfull"], arr["kex"], arr["fex"], arr["mom"], 1.0, True)
    Fg = np.ascontiguousarray(F0[:, :, None].copy())
    lam = np.ascontiguousarray(arr["rate"][None, :] / E[:, None])
    L._segment(Fg, E, dE, np.ascontiguousarray(S0[:, :, None]), 1.0, lam, np.zeros((N, 3, 3)), arr["br"], arr["own"],
               arr["kfull"], arr["kex"], arr["fex"], arr["mom"], True)
    assert np.max(np.abs(Fg - Fs)) < 1e-12 * np.max(np.abs(Fs))


def test_galactic_transfer_limits_and_brute_force():
    """No process: the averaged transfer in every pixel. With decays and conversions: the SVD path (one outward sweep
    over the emission's shapes) against each pixel's own line of sight swept inward with its actual emission per
    bin."""
    import dp_galactic
    dp_galactic.use_data_dir(None)
    E = np.geomspace(4.57e12, 1e17, 80)
    comps = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    out = L.galactic_transfer(E, U, None, "cringe", 8, comps)
    P = dp_galactic.averaged_transfer(U)
    assert np.allclose(out[1][5], P @ comps[1])
    m = np.array([1e-6, 8.68e-3, 0.0501])
    gf = dp_si.flavor_couplings("mu", 1e-5, None)
    r = L.galactic_rates(E, gf, U, m, 1e-3, "vector", True, T0, np.ones(3), H0, 0.315, 0.685)
    assert r is not None and r["dec"] is not None and np.any(r["Rc"] > 0)
    out = L.galactic_transfer(E, U, r, "cringe", 8, comps)
    X, Iout = L.emission_rows("cringe", E, 8)
    e = dp_galactic._data_file("galactic_emission.npz")["gas_s_edges_kpc"]
    Ls = np.diff(e) * L.KPC_EV_INV
    L0 = float(e[0]) * L.KPC_EV_INV
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    d = r["dec"]
    A = np.abs(U) ** 2
    for pix in (0, 300, 700):
        F = np.zeros((len(E), 3, 3))
        for k in range(len(Ls) - 1, -1, -1):
            S = np.zeros((len(E), 3, 3))
            for j in range(3):
                S[:, j, j] = X[pix, k] / Ls[k]
            L._segment(F, E, dE, S, Ls[k], r["lam"], r["Rc"], d["br"], d["own"], d["kfull"], d["kex"], d["fex"],
                       d["mom"], True)
        L._segment(F, E, dE, np.zeros_like(F), L0, r["lam"], r["Rc"], d["br"], d["own"], d["kfull"], d["kex"],
                   d["fex"], d["mom"], True)
        p = A.T @ comps[1]
        ref = (F @ p) / Iout[pix][:, None] @ A.T
        ok = Iout[pix] > 1e-6 * Iout[pix].max()
        # the two differ by how the daughters of the energy bins above are held over a segment (constant in both, over
        # different segments: first order in the segments, 1e-6 here); the SVD's truncation is 1e-10
        assert np.max(np.abs(out[1][pix][ok] - ref[ok])) < 1e-5 * np.max(np.abs(ref[ok])), pix


def test_dirac_unitarity_and_the_general_partial_wave():
    """The Dirac Goldstone measure: one Dirac pair reproduces 2 |y|^2 / (8 pi / 3) (Allwicher et al. Eq. 4.7); the
    n-fermion partial wave equals dp_unitarity's for three Majorana fermions; the measure scales as (g / M)^2."""
    for y in (0.3, 1.7):
        Yp = np.array([[0, y], [y, 0]], dtype=complex)
        assert np.isclose(2 * L._partial_wave_max_n(Yp), y * y / (8 * np.pi / 3), rtol=1e-12)
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Y = Y + Y.T
    assert np.isclose(L._partial_wave_max_n(Y), dp_unitarity.partial_wave_max(Y), rtol=1e-12)
    m = _m(1e-6)
    G = dp_si.flavor_couplings("mu", 1.0, None)
    for maj in (True, False):
        a, b = L.goldstone_measure(G * 1e-10, U, m, 1.0, maj), L.goldstone_measure(G * 1e-12, U, m, 1e-2, maj)
        assert np.isclose(a, b, rtol=1e-10)


def test_scalar_field_measure():
    """The relic field: zero without coupling; (g / M)^2 at a fixed relic composition (decays off); the relics after
    their decays (the light nu_1 alone carries almost no scalar density) move the line up."""
    m = _m(1e-6)
    G = dp_si.flavor_couplings("mu", 1.0, None)
    args = (T0, np.ones(3), H0, 0.315, 0.685, 4.0, 7.537e-5)
    assert L.scalar_field_measure(G * 0.0, U, m, 1e-9, True, *args) == 0.0
    for maj in (True, False):
        a = L.scalar_field_measure(G * 1e-10, U, m, 1e-9, maj, *args, decay_on=False)
        b = L.scalar_field_measure(G * 1e-12, U, m, 1e-11, maj, *args, decay_on=False)
        assert np.isclose(a, b, rtol=1e-6)
        # nu_1 at 1e-6 eV after the decays: smaller by orders of magnitude
        assert L.scalar_field_measure(G * 1e-8, U, m, 1e-9, maj, *args) < 1e-2 * L.scalar_field_measure(
            G * 1e-8, U, m, 1e-9, maj, *args, decay_on=False)


def test_subkev_inputs():
    """The inputs dp_si gets below 1 keV: decays only where they act, the relics after their decays, the scalar's
    collision mass floored at the heaviest neutrino, the vector's Debye-screened mass, and no loss for the scalar."""
    m = _m(1e-6)
    gv = dp_si.flavor_couplings("mu", 1e-9, None)
    inp = L.subkev_inputs(gv, U, m, 1e-6, "vector", True, T0, np.ones(3), H0, 0.315, 0.685, 4.0, 4.6e12)
    assert inp["decay"] is not None and inp["eta"][0] > 2.9 and inp["M_coll"] >= 1e-6
    inp = L.subkev_inputs(gv * 1e-10, U, m, 1e-6, "vector", True, T0, np.ones(3), H0, 0.315, 0.685, 4.0, 4.6e12)
    assert inp["decay"] is None and inp["relic_loss"] is None
    inp = L.subkev_inputs(gv, U, m, 1.0, "vector", False, T0, np.ones(3), H0, 0.315, 0.685, 4.0, 4.6e12)
    assert inp["decay"] is None                                 # above the mass splittings
    gs = dp_si.flavor_couplings("e", 1e-3, None)
    inp = L.subkev_inputs(gs, U, m, 1e-20, "scalar", False, T0, np.ones(3), H0, 0.315, 0.685, 4.0, 4.6e12)
    assert inp["M_coll"] == m[2] and inp["relic_loss"] is None and inp["decay"] is not None


def _si_cfg(name):
    import copy
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from analysis import config as config_mod
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_si", "now_ic26", name + ".json")
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    return p, config_mod.from_dict(raw, p)


def test_analysis_boundaries_for_the_secret_interactions():
    """analysis.exclusion for the SI planes (2026-10-25): the vector's line from its Goldstone couplings (Majorana and
    Dirac), the scalar's per-point measure (no line), and the fits' cap per point for the scalar
    (exact_profile._measure_cap): 0 deep in the region, None far below it, in between the mass where it crosses 1."""
    from types import SimpleNamespace
    from analysis import exclusion as ex
    from analysis.exact_profile import ExactProfiler
    for name, c_ref in (("si_vector_majorana_mu_ul", 1.76), ("si_vector_dirac_mu_ul", 1.93)):
        p, _ = _si_cfg(name)
        slope, c = ex.not_allowed_boundary(p, 1e-6)
        assert slope == 1.0 and abs(c - c_ref) < 0.01
        assert ex.not_modelled_measure(p) is None
    p, cfg = _si_cfg("si_scalar_majorana_mu_ul")
    assert ex.not_allowed_boundary(p, 1e-6) is None
    f = ex.not_modelled_measure(p)
    assert f(1e-6, 1e-1, 1e-6) > 1.0 and f(1e-6, 1e-6, 1e-6) < 1.0
    me = SimpleNamespace(_ml_bounds0=(1e-6, 0.1), eta=["m_lightest_eV"], cfg=cfg)
    me._measure_cap = lambda held: ExactProfiler._measure_cap(me, held)
    pl = cfg.planes[0]
    assert ExactProfiler.lightest_cap(me, {pl.x: 1e-6, pl.y: 1.0}) == 0.0
    assert ExactProfiler.lightest_cap(me, {pl.x: 1e-6, pl.y: 1e-9}) is None
    assert ExactProfiler.lightest_cap(me, {pl.x: 1e3, pl.y: 1.0}) is None          # the keV planes: never
    # between the lightest masses' lines (relics decayed: 10^4 at 1e-6 eV, 10^2 at 0.1 eV, mu texture)
    cap = ExactProfiler.lightest_cap(me, {pl.x: 1e-6, pl.y: 1e-3})
    assert 1e-6 < cap < 0.1
    assert f(1e-6, 1e-3, cap / 1.3) < 1.0 < f(1e-6, 1e-3, cap * 1.3)
