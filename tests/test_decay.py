"""dp_decay: neutrino decay into the mediator (notes Sec. 12.8 item 2, 2026-10-24)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_decay as dd  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

U = construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0))
M_NU = np.array([1e-6, 8.68e-3, 0.0501])


def _grid(N=120, lo=1e12, hi=1e16):
    return np.geomspace(lo, hi, N)


def test_rates_match_the_formula_and_ignore_a_universal_coupling():
    G = 1e-8 * np.outer([0, 1, 0], [0, 1, 0]).astype(complex)
    Gam, fL = dd.rest_rates(G, U, M_NU, 1e-4)
    Gp = U.conj().T @ G @ U
    mi, mj, M = M_NU[2], M_NU[0], 1e-4
    p = np.sqrt((mi**2 - (mj + M) ** 2) * (mi**2 - (mj - M) ** 2)) / (2 * mi)
    ref = p / (16 * np.pi * mi**2) * (2 * abs(Gp[0, 2]) ** 2 * (mi**2 + mj**2 - 2 * M**2 + (mi**2 - mj**2) ** 2 / M**2)
                                      + 12 * (Gp[0, 2] ** 2).real * mi * mj)
    assert np.isclose(Gam[2, 0], ref, rtol=1e-12)
    assert Gam[0, 1] == Gam[0, 2] == Gam[1, 2] == 0.0          # only downward
    Gk, _ = dd.rest_rates(G + 0.74 * np.eye(3), U, M_NU, 1e-4)  # a universal part changes nothing (to its rounding)
    assert np.allclose(Gk, Gam, rtol=1e-6, atol=0)
    G0, _ = dd.rest_rates(0.74 * np.eye(3), U, M_NU, 1e-20)
    assert not np.any(G0)


def _number(E, F):
    h = np.sqrt(E[1] / E[0])
    return float(np.sum(F.sum(axis=1) * (E * h - E / h)))


def test_number_conserved_and_survival():
    E = _grid()
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1e-20
    fL = np.full((3, 3), 0.9)
    Fn = np.zeros((len(E), 3))
    Fb = np.zeros((len(E), 3))
    Fn[100, 2] = 1.0
    n0 = _number(E, Fn)
    zp1, dt = 1.0, 1e30
    dd.decay_step(E, zp1, dt, Gam, fL, M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
    lam = Gam[2, 0] * M_NU[2] / E[100] * dt
    assert np.isclose(Fn[100, 2], np.exp(-lam), rtol=1e-12)
    # kept, but for the daughters below the grid: z < E_lo,0 / E_100 (kernels <= 1 per unit z)
    lost = n0 - (_number(E, Fn) + _number(E, Fb))
    z_min = E[0] / np.sqrt(E[1] / E[0]) / E[100]
    assert 0.0 <= lost <= n0 * (-np.expm1(-lam)) * z_min * 1.0001


def test_daughter_spectra():
    E = _grid(N=400)
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1.0
    for fl in (1.0, 0.0, 0.3):
        fL = np.full((3, 3), fl)
        Fn = np.zeros((len(E), 3))
        Fb = np.zeros((len(E), 3))
        j = 399
        Fn[j, 2] = 1.0
        dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
        h = np.sqrt(E[1] / E[0])
        dEj = E[j] * h - E[j] / h
        # the daughters per bin against the kernels integrated over each bin (parent at E_j)
        lo, hi = np.minimum(E / h, E[j]), np.minimum(E * h, E[j])
        z_lo, z_hi = lo / E[j], hi / E[j]
        ft = 1 - fl
        same = ft * (z_hi - z_lo) + 0.5 * (fl - ft) * (z_hi**2 - z_lo**2)
        other = fl * (z_hi - z_lo) + 0.5 * (ft - fl) * (z_hi**2 - z_lo**2)
        got_same = Fn[:, 0] * (E * h - E / h) / dEj
        got_other = Fb[:, 0] * (E * h - E / h) / dEj
        assert np.allclose(got_same, same, atol=1e-12)
        assert np.allclose(got_other, other, atol=1e-12)


def test_chain_runs_within_a_step():
    E = _grid()
    Gam = np.zeros((3, 3))
    Gam[2, 1] = 1.0
    Gam[1, 0] = 1.0
    fL = np.full((3, 3), 1.0)
    Fn = np.zeros((len(E), 3))
    Fb = np.zeros((len(E), 3))
    Fn[110, 2] = 1.0
    n0 = _number(E, Fn)
    dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
    assert Fn[:, 1:].sum() + Fb[:, 1:].sum() < 1e-12 * n0
    lost = n0 - (_number(E, Fn) + _number(E, Fb))
    assert lost >= -1e-12 * n0


def test_fold_is_flat():
    E = _grid(N=300)
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1.0
    fL = np.full((3, 3), 0.8)
    F = np.zeros((len(E), 3))
    F[299, 2] = 1.0
    dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, F, F, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, True)
    h = np.sqrt(E[1] / E[0])
    dEj = E[299] * h - E[299] / h
    lo, hi = np.minimum(E / h, E[299]), np.minimum(E * h, E[299])
    flat = (hi - lo) / E[299]
    assert np.allclose(F[:, 0] * (E * h - E / h) / dEj, flat, atol=1e-12)


def test_eigenstate_content():
    """Populations as eigenstates with a mass content: a pure mass-3 eigenstate decays like mass state 3, and the
    daughters are split over the eigenstates by their mass content."""
    E = _grid(N=50)
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1.0
    fL = np.full((3, 3), 1.0)
    T = np.zeros((len(E), 3, 3))
    T[:, 0, 0] = T[:, 0, 1] = 0.5        # mass state 1 shared by eigenstates 0 and 1
    T[:, 1, 0] = T[:, 1, 1] = 0.5
    T[:, 2, 2] = 1.0
    Fn = np.zeros((len(E), 3))
    Fb = np.zeros((len(E), 3))
    Fn[49, 2] = 1.0
    dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, Fn, Fb, T, T, True, False)
    assert Fn[:, 2].sum() < 1e-12
    assert np.allclose(Fn[:, 0], Fn[:, 1])


def test_relic_composition():
    H0, Om, OL, T0 = 1.4377e-33, 0.315, 0.685, 1.676e-4
    eta = np.ones(3)
    assert np.allclose(dd.relic_composition(np.zeros((3, 3)), M_NU, eta, T0, H0, Om, OL), eta)
    Gam = np.zeros((3, 3))
    Gam[2, 1] = Gam[2, 0] = Gam[1, 0] = 1e-20           # ~1e13 H0 at rest: all decayed long before z = 1
    out = dd.relic_composition(Gam, M_NU, eta, T0, H0, Om, OL)
    assert np.allclose(out, [3.0, 0.0, 0.0], atol=1e-9)
    Gam_slow = Gam * 1e-40                              # nothing decays
    assert np.allclose(dd.relic_composition(Gam_slow, M_NU, eta, T0, H0, Om, OL), eta, atol=1e-9)


def test_jumps_zeno_protected_by_decay_widths():
    """dp_cosmo._soft_jumps with the decay widths (2026-10-24): a pair at frequency omega with a damping D converts at
    Gamma J omega^2 / (omega^2 + D^2); widths Wd add (Wd_a + Wd_b) / 2 to D."""
    from dp_cosmo import _soft_jumps
    E = np.array([1e14])
    J = np.zeros((2, 3, 3))
    J[:, 0, 1] = J[:, 1, 0] = 1.0
    lam = np.zeros((2, 3))
    lam[:, 1] = 2.0 * 1e14 * 1e-20             # omega_01 = 1e-20 eV at zp1 = 1
    zeros = np.zeros((2, 3))
    for wd in (0.0, 3e-20):
        F = np.array([[1.0, 0.0, 0.0]])
        Wd = np.array([[wd, wd, 0.0]])
        rate_dt = 1e-6
        _soft_jumps(E, 1.0, J, zeros, zeros, lam, 0.0, 1.0, 0.0, 1.0, rate_dt, F, Wd, True)
        om, D = 1e-20, wd
        expect = 0.5 * (1.0 - np.exp(-2.0 * rate_dt * om * om / (om * om + D * D))) if wd else 0.5 * (1.0 - np.exp(-2.0 * rate_dt))
        assert np.isclose(F[0, 1], expect, rtol=1e-6)


def test_truth_model_has_no_decay():
    """The projections' Asimov data come from the Standard Model: relic scatterings and decay both off."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import json
    from analysis import config as config_mod
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_ml", "proj_future", "dp_mu_ul.json")
    raw = json.load(open(p))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    truth = cfg._truth_model()
    E, params = __import__("dark_portal_solver").build_solver_params(truth._run_cfg(truth._overrides(
        {q.name: q.start for q in cfg.parameters})))
    assert params["neutrino_decay"] is False and params["relic_scattering"] is False


def test_solver_uses_the_decayed_relics():
    """propagate_spectrum: where the relics decay, every relic effect sees their composition after the decays (the
    Debye mass and the jumps' rates get eta' = (3, 0, 0)-like)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import json, copy
    import dark_portal_solver as dps
    import dp_absorption
    from analysis import config as config_mod
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_ml", "now_ic26", "dp_mu_ul.json")
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    m = cfg.model()
    v = {q.name: q.start for q in cfg.parameters}
    v.update({"m_Zd_eV": 1e-6, "g_D_trace": 1e-6})
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    params["show_progress"] = False
    params["cosmo_transport"] = "dark_portal"
    seen = []
    orig = dp_absorption.debye_mass_sq
    def spy(G2, m_nu, T, eta, *a, **kw):
        seen.append(np.array(eta, dtype=float))
        return orig(G2, m_nu, T, eta, *a, **kw)
    dp_absorption.debye_mass_sq = spy
    try:
        dps.propagate_weighted(np.asarray(E), params, sources=[np.array([1.0, 2.0, 0.0]) / 3.0])
    finally:
        dp_absorption.debye_mass_sq = orig
    assert seen and all(np.allclose(e, [3.0, 0.0, 0.0], atol=1e-6) for e in seen), seen[:3]
