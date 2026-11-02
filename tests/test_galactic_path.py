"""dp_galactic_path: the incoherent processes along the Galactic lines of sight (2026-10-24)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_galactic_path as gp  # noqa: E402
import dp_medium  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

M_NU = np.array([0.0, 8.6e-3, 0.05])
F_S = np.array([1.0 / 3.0, 2.0 / 3.0, 0.0])


def _setup(seed=1, decay=True, jumps=True, loss=True):
    """Twelve pixels (N_side 1), twelve equal distance bins from 2.5 to 32.5 kpc (the nearest bin's edge as far from
    the Sun as a bin is long), no potential (coef2 0): random intensities and distance weights; rates set to O(1)
    effects over the path."""
    rng = np.random.default_rng(seed)
    U = construct_PMNS(0.59, 0.15, 0.84, 3.4)
    E = np.geomspace(1e3, 1e7, 24)
    npix, ns = 12, 12
    s = np.linspace(2.5, 32.5, ns + 1)                  # the bins' edges
    I = rng.uniform(0.2, 1.0, (npix, 1)) * (E / E[0]) ** -2.0
    w = rng.uniform(0.0, 1.0, (npix, len(E)))
    q_gas = rng.uniform(0.0, 1.0, (npix, ns))
    q_src = rng.uniform(0.0, 1.0, (npix, ns))
    G = 3e-13 * np.array([[0.0, 1.0, 0.5], [1.0, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    L = s[-1] * gp.KPC_EV_INV
    kinds, tables, P_vac = [], [], np.zeros((2, len(E), 3))
    for kk, Uk in enumerate((U, np.conj(U))):
        Gk = np.conj(G) if kk else G
        M_rot = Uk @ np.diag(M_NU ** 2) @ np.conj(Uk).T
        S = np.diag([1.0, -1.0, 0.0])
        t = gp.build_tables(M_rot, S, Uk, Gk, 0.0, 0.0, n_sigma=2)
        tables.append(t)
        P_vac[kk] = F_S @ np.abs(Uk) ** 2
        kinds.append(dict(P=np.stack([P_vac[kk], P_vac[kk]]), j=np.zeros((npix, ns), np.int64),
                          t=np.zeros((npix, ns)), n=np.ones((npix, ns)), coef2=0.0, tables=t))
    jump_unit = 1.0 / (L * max(float(tables[0]["J"].max()), 1e-300)) if jumps else 0.0
    loss_C = 0.3 * E[0] / (L * float(tables[0]["GG"].max())) if loss else 0.0
    ladder = None
    if decay:
        M_rot = U @ np.diag(M_NU ** 2) @ np.conj(U).T
        S = np.diag([1.0, -1.0, 0.0]).astype(complex)
        ladder = dp_medium.Ladder(E, G, U, M_NU, 1e-4, [(M_rot, S), (np.conj(M_rot), -S)], 0.0)
    rates = dict(jump_unit=jump_unit, ladder=ladder, loss_C=loss_C)
    return E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates


@pytest.mark.slow     # 30-90 s on CI's runners (2026-10-24): Tycho's suite runs it
def test_vacuum_path_matches_the_sweep():
    """Equal segments (the order of the path's pieces then immaterial): the spectral-basis path equals the
    per-pixel sweep to rounding, with all three processes on."""
    E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates = _setup()
    a = gp.sweep_populations(E, I, w, q_gas, q_src, s, kinds, rates)
    b = gp.vacuum_populations(E, I, w, q_gas, q_src, s, P_vac, tables, rates, 1, 1)
    scale = a.sum(axis=(0, 3))[None, :, :, None]
    assert np.max(np.abs(b - a) / scale) < 1e-9
    # the processes act: the populations moved from the emitted ones
    emitted = (w[:, :, None] * q_gas.sum(1)[:, None, None] + (1 - w[:, :, None]) * q_src.sum(1)[:, None, None])
    assert np.max(np.abs(a[0] - emitted * P_vac[0][None])) > 1e-2 * emitted.max()


def test_jumps_keep_the_number():
    """The soft scatterings alone move neutrinos between states, never in or out: per pixel and energy the
    populations at the Sun sum to the emitted number per unit intensity."""
    E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates = _setup(decay=False, loss=False)
    a = gp.sweep_populations(E, I, w, q_gas, q_src, s, kinds, rates)
    emitted = w * q_gas.sum(1)[:, None] + (1 - w) * q_src.sum(1)[:, None]
    for kk in range(2):
        assert np.allclose(a[kk].sum(-1), emitted, rtol=1e-12)


def test_columns_are_independent_runs():
    """The columns of one sweep (the compositions) share every bin's exponentials: each column is what a sweep with
    it alone gives, to rounding, with all three processes on."""
    E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates = _setup()
    rng = np.random.default_rng(7)
    P2 = [rng.uniform(0.0, 1.0, k["P"].shape) for k in kinds]
    both = [dict(k, P=np.stack([k["P"], p2])) for k, p2 in zip(kinds, P2)]
    ab = gp.sweep_populations(E, I, w, q_gas, q_src, s, both, rates)
    a = gp.sweep_populations(E, I, w, q_gas, q_src, s, kinds, rates)
    b = gp.sweep_populations(E, I, w, q_gas, q_src, s, [dict(k, P=p2) for k, p2 in zip(kinds, P2)], rates)
    for x, y in ((ab[0], a), (ab[1], b)):
        assert np.max(np.abs(x - y)) <= 1e-12 * np.max(np.abs(y))


def test_crossing_limits():
    """The extragalactic beam's crossing: without processes it is the identity (the adiabatic transit as before);
    with the decays alone (daughters on the grid: nearly degenerate masses) each line of sight keeps the number."""
    E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates = _setup(decay=False, jumps=False, loss=False)
    rng = np.random.default_rng(3)
    P = rng.uniform(0.1, 1.0, (2, len(E), 2, 3))
    groups = (np.arange(13, dtype=np.int64), np.arange(12, dtype=np.int64))
    Q = gp.crossing_populations(E, P, s, kinds, rates, groups)
    assert np.array_equal(Q, np.broadcast_to(P, Q.shape))
    U = construct_PMNS(0.59, 0.15, 0.84, 3.4)
    m = np.sqrt(np.array([0.0, 7.5e-5, 2.5e-3]) + 0.01)
    G = 3e-13 * np.array([[0.0, 1.0, 0.5], [1.0, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    S = np.diag([1.0, -1.0, 0.0]).astype(complex)
    M_rot = U @ np.diag(m ** 2 - m[0] ** 2) @ np.conj(U).T
    lad = dp_medium.Ladder(E, G, U, m, 1e-4, [(M_rot, S), (np.conj(M_rot), -S)], 0.0)
    P[:, : len(E) // 2] = 0.0
    Q = gp.crossing_populations(E, P, s, kinds, dict(rates, ladder=lad), groups)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    n0 = (P.sum(axis=(2, 3)) * dE).sum(axis=1)
    n1 = (Q.sum(axis=(3, 4)) * dE).sum(axis=2)
    assert np.max(np.abs(n1 / n0[None] - 1.0)) < 1e-10
    assert np.max(np.abs(Q - P[None])) > 1e-3 * np.max(P)            # the decays act


@pytest.mark.slow     # a dark-portal phase and the handover's band responses on the campaign grid
def test_crossing_without_processes_is_the_transit():
    """The extragalactic beam's crossing goes into the handover per declination band through its responses to the
    three unit inputs (dark_portal_solver._flux_runs): with an identity crossing the bands are the handover's own to
    rounding, and the whole sky the bands weighted by their cos(zenith) widths."""
    import copy
    import json
    import os
    import dark_portal_solver as dps
    from analysis import config as config_mod
    repo = os.path.join(os.path.dirname(__file__), "..")
    p = os.path.join(repo, "analysis", "configs", "campaign_ml", "now_ic26", "le_lmu_ul.json")
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    m = cfg.model()
    v = {q.name: q.start for q in cfg.parameters}
    v.update(m_Zd_eV=1e-12, g_D=1e-13)
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    params["show_progress"] = False
    E = np.asarray(E)
    setup = dps._flux_setup(E, params)
    import dp_sky
    grid = dp_sky.SkyGrid(8, None)
    runs = [(dps.injection_weights(params), None)]
    orig = dps._crossing_bands
    try:
        dps._crossing_bands = lambda E_run, params_, setup_, P, dec_edges: np.repeat(P[None], len(dec_edges) - 1, 0)
        got = dps._flux_runs(E, params, setup, runs, band_costh=grid.costh_edges)[0]
    finally:
        dps._crossing_bands = orig
    ref = dps._flux_from_setup(E, params, setup, runs[0][0], band_costh=grid.costh_edges)
    for a, b in zip(got, ref):
        assert np.max(np.abs(a - b)) <= 1e-10 * np.max(np.abs(b))


def _campaign_point(name, x, y, **over):
    import copy
    import json
    import os
    import dark_portal_solver as dps
    from analysis import config as config_mod
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_ml", "now_ic26", name)
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    m = cfg.model()
    v = {q.name: q.start for q in cfg.parameters}
    pl = raw["planes"][0]
    v.update({pl["x"]: x, pl["y"]: y})
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    params["show_progress"] = False
    params.update(over)
    return np.asarray(E), params


@pytest.mark.slow     # the crossing's sweeps on the campaign grid
def test_crossing_in_the_transport():
    """The crossing's paths in the transport (dark_portal_solver._flux_runs, _crossing_bands): where the Galaxy's
    potential sets the eigenstates (le_lmu_ul at g/M = 1 eV^-1: the beam's flip there moves the flux by up to ~17%)
    and where it is negligible (dp_mu_ul: one vacuum line of sight), finite and non-negative, the number of the beam
    not raised, and switched off by EXTRAGALACTIC_CROSSING; and where nothing acts it is the transit as before."""
    import dark_portal_solver as dps
    assert dps.EXTRAGALACTIC_CROSSING
    for name, x, y, moves in (("le_lmu_ul.json", 1e-12, 1e-12, True), ("dp_mu_ul.json", 1e-4, 1e-6, None),
                              ("le_lmu_ul.json", 1e-12, 1e-40, False)):
        E, p = _campaign_point(name, x, y)
        on = np.asarray(dps.propagate_weighted(E, p))
        try:
            dps.EXTRAGALACTIC_CROSSING = False
            off = np.asarray(dps.propagate_weighted(E, p))
        finally:
            dps.EXTRAGALACTIC_CROSSING = True
        assert np.all(np.isfinite(on)) and np.all(on >= 0.0)
        h = np.sqrt(E[1] / E[0])
        dE = E * (h - 1 / h)
        assert (on.sum(-1) * dE).sum() <= (off.sum(-1) * dE).sum() * (1 + 1e-3)
        d = np.max(np.abs(on - off)) / np.max(off)
        if moves is True:
            assert d > 1e-3, d
        elif moves is False:
            assert np.array_equal(on, off)
