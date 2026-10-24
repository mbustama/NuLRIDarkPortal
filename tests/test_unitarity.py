"""dp_unitarity against the bounds of Allwicher, Arnan, Barducci, Nardecchia, JHEP 10 (2021) 129."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_unitarity as du  # noqa: E402


def _threshold(Y_unit):
    """The scale s at which partial_wave_max(s Y_unit) = 1/2."""
    return np.sqrt(0.5 / du.partial_wave_max(Y_unit))


def test_one_majorana_flavor():
    Y = np.zeros((3, 3), complex)
    Y[0, 0] = 1.0
    assert np.isclose(_threshold(Y) ** 2, 4 * np.pi)              # their Eq. (6.2), J = 0


def test_dirac_pair():
    Y = np.zeros((3, 3), complex)
    Y[0, 1] = Y[1, 0] = 1.0
    assert np.isclose(_threshold(Y) ** 2, 8 * np.pi / 3)          # their Eq. (4.7), N = 1, real scalar


def test_half_integer_channel():
    Y = np.zeros((3, 3), complex)
    Y[0, 0] = 1.0
    j12 = 3.0 / (32 * np.pi)
    assert np.isclose(np.sqrt(0.5 / j12) ** 2, 16 * np.pi / 3)   # their Eq. (6.4), real scalar


def test_phases_do_not_matter_for_one_entry():
    Y = np.zeros((3, 3), complex)
    Y[1, 2] = Y[2, 1] = np.exp(0.7j)
    assert np.isclose(_threshold(Y) ** 2, 8 * np.pi / 3)


def test_goldstone_yukawa_matches_the_decay_rate():
    """|Y_31| = |G'_31| m_3 / M for m_1 = 0: the Majorana rate |Y|^2 m/(16 pi) is twice the Dirac LH vector rate."""
    Gp = np.zeros((3, 3), complex)
    Gp[0, 2] = Gp[2, 0] = 2e-9
    m = np.array([0.0, 8.6e-3, 0.05])
    Y = du.goldstone_yukawa(Gp, m, 1e-6)
    assert np.isclose(abs(Y[0, 2]), 2e-9 * 0.05 / 1e-6)


def test_plane_boundary_matches_a_direct_evaluation():
    """analysis.exclusion.not_allowed_boundary: the line log10 g = log10 m + c is where the measure crosses 1."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from analysis import exclusion as ex
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_ml", "now_ic26", "dp_mu_ul.json")
    slope, c = ex.not_allowed_boundary(p)
    assert slope == 1.0
    import copy, json
    import dark_portal_solver as dps
    import dp_absorption
    from dp_core import construct_PMNS
    from analysis import config as config_mod
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    m = cfg.model()
    for lm in (-12.0, -6.0):
        for dy, side in ((0.01, True), (-0.01, False)):
            v = {q.name: q.start for q in cfg.parameters}
            v.update({"m_Zd_eV": 10.0 ** lm, "g_D_trace": 10.0 ** (lm + c + dy)})
            _, pr = dps.build_solver_params(m._run_cfg(m._overrides(v)))
            G = dp_absorption.dark_portal_coupling(pr["g_D"], np.array([pr["u_e"], pr["u_mu"], pr["u_tau"]], complex), 0.0)
            U = construct_PMNS(pr["th12"], pr["th13"], pr["th23"], pr["d_cp"])
            Gp = U.conj().T @ G @ U
            r = du.neutrino_unitarity(Gp, np.asarray(pr["m_nu_bg_arr"]), 10.0 ** lm)
            assert (r >= 1.0) == side


def test_lightest_cap_follows_the_boundary():
    """exact_profile.ExactProfiler.lightest_cap (2026-10-24, the fits only where the model holds): 0 above the
    1e-6 eV line, None below the 0.1 eV one, and in between the mass where the line crosses the point."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import copy
    import json
    from types import SimpleNamespace
    from analysis import config as config_mod
    from analysis import exclusion as ex
    from analysis.exact_profile import ExactProfiler
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_ml", "now_ic26", "dp_mu_ul.json")
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    me = SimpleNamespace(_ml_bounds0=(1e-6, 0.1), eta=["m_lightest_eV"], cfg=cfg)
    pl = cfg.planes[0]
    c6, c1 = ex.not_allowed_boundary(p, 1e-6)[1], ex.not_allowed_boundary(p, 0.1)[1]
    m = 1e-4
    assert ExactProfiler.lightest_cap(me, {pl.x: m, pl.y: m * 10 ** (c6 + 0.01)}) == 0.0
    assert ExactProfiler.lightest_cap(me, {pl.x: m, pl.y: m * 10 ** (c1 - 0.01)}) is None
    t = 0.5 * (c6 + c1)
    cap = ExactProfiler.lightest_cap(me, {pl.x: m, pl.y: m * 10 ** t})
    assert 1e-6 < cap < 0.1
    assert ex.not_allowed_boundary(p, cap / 1.2)[1] > t > ex.not_allowed_boundary(p, cap * 1.2)[1]
