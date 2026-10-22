"""
Soft scatterings on the relics and the dark portal below 1 keV (2026-10-22; notes, 'Soft scatterings on the relics'):
the eigenstate jumps keep the number and drive the states they connect to equal shares when fast, and are
Zeno-suppressed when the coherences are damped faster than they turn; the routing of the cosmological phase; a run
whose scatterings add up to nothing is the same as before, bit for bit; full equalization where they are fast; and the
jumps' normalization against dp_si's transport, which carries the same scatterings in the mass basis, where both apply.
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

import dark_portal_solver as dps  # noqa: E402
from dp_cosmo import _soft_jumps  # noqa: E402

PION = [np.array([1.0, 2.0, 0.0]) / 3.0]


def _tables(J, lam, Gd=None, GG=None):
    """Two identical sigma nodes holding one set of jump tables (the interpolation then returns them)."""
    J2 = np.ascontiguousarray(np.stack([J, J]))
    lam2 = np.ascontiguousarray(np.stack([lam, lam]))
    Gd = np.zeros(3) if Gd is None else Gd
    GG = np.zeros(3) if GG is None else GG
    return J2, np.ascontiguousarray(np.stack([Gd, Gd])), np.ascontiguousarray(np.stack([GG, GG])), lam2


def test_jumps_conserve_number_and_equalize():
    """Rates far above 1/dt: the populations become equal shares of the total; slow: they barely move; always the
    same total."""
    rng = np.random.default_rng(3)
    A = rng.random((3, 3))
    J = A + A.T
    lam = np.array([0.0, 1e-4, 3e-3])
    J2, Gd2, GG2, lam2 = _tables(J, lam)
    E = np.array([1e14])
    for G_unit, expect_equal in ((1e-20, True), (1e-40, False)):
        F = np.array([[0.7, 0.2, 0.1]])
        _soft_jumps(E, 2.0, J2, Gd2, GG2, lam2, 0.0, 1.0, 1e-30, G_unit, 1e30, F)
        assert abs(F.sum() - 1.0) < 1e-12
        if expect_equal:
            assert np.allclose(F, 1.0 / 3.0, atol=1e-12)
        else:
            assert np.allclose(F, [[0.7, 0.2, 0.1]], atol=1e-8)


def test_jumps_zeno_suppressed():
    """Coherences damped much faster than they turn (D >> w): the transfer slows as w^2 / D^2 (here no motion), while
    the same rate with undamped coherences equalizes."""
    J = np.ones((3, 3))
    lam = np.array([0.0, 1e-4, 3e-3])            # w ~ 1e-4 / 2E(1+z) ~ 2.5e-19 eV at 1e14 eV, z = 1
    E = np.array([1e14])
    G_unit = 1e-10
    J2, Gd2, GG2, lam2 = _tables(J, lam, Gd=np.array([1.0, 0.0, 0.0]), GG=np.array([1.0, 0.0, 0.0]))
    F = np.array([[0.7, 0.2, 0.1]])
    _soft_jumps(E, 2.0, J2, Gd2, GG2, lam2, 0.0, 1.0, 1e-30, G_unit, 1e3, F)   # undamped pair 1-2: rate x dt 1e-7
    assert np.allclose(F, [[0.7, 0.2, 0.1]], atol=1e-7)
    J2, Gd2, GG2, lam2 = _tables(J, lam)
    F = np.array([[0.7, 0.2, 0.1]])
    _soft_jumps(E, 2.0, J2, Gd2, GG2, lam2, 0.0, 1.0, 1e-30, G_unit, 1e12, F)
    assert np.allclose(F, 1.0 / 3.0, atol=1e-9)


def _point(cfg_name, x, y, **over):
    """(E, params) of a campaign configuration at its plane's (x, y)."""
    from analysis import config as config_mod
    p = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26", cfg_name)
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    raw["parameters"].pop("m_lightest_eV", None)
    cfg = config_mod.from_dict(raw, p)
    m = cfg.model()
    v = {q.name: q.start for q in cfg.parameters}
    pl = raw["planes"][0]
    v.update({pl["x"]: x, pl["y"]: y})
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    params["show_progress"] = False
    params.update(over)
    return np.asarray(E), params


def test_cosmology_mode():
    """'si' at 1 keV and above where the potential is negligible; below 1 keV the dark-portal phase for couplings
    below SI_HARD_MIN_G, dp_si above it where the potential is negligible, both ('combined') where it is not."""
    E, p = _point("dp_e_ul.json", 1e4, 1e-3)
    assert dps._cosmology_mode(p, E) == "si"
    E, p = _point("dp_e_ul.json", 100.0, 1e-6)
    assert dps._cosmology_mode(p, E) == "dark_portal"
    E, p = _point("dp_e_ul.json", 100.0, 1e-3)
    assert dps._cosmology_mode(p, E) == "si"
    E, p = _point("dp_mu_pot_x1m08_pos.json", 1e-10, 1e-11)     # eps' = 1e-3: kappa = 7e-4, and a potential
    assert dps._cosmology_mode(p, E) == "combined"
    E, p = _point("dp_mu_pot_x1m16_pos.json", 1e-10, 1e-30)     # eps' = 1e-14: kappa below SI_HARD_MIN_G
    assert dps._cosmology_mode(p, E) == "dark_portal"


def _flux(E, p):
    F = np.asarray(dps.propagate_weighted(E, p, sources=PION)[0])
    k = int(np.argmin(abs(E - 1e14)))
    return F, F[k] / F[k].sum()


def test_negligible_scatterings_bit_for_bit():
    """A potential plane at g_D sum|U|^2 = 1e-16 (the scatterings far below the expansion rate): the same flux with
    the jumps' code as without it."""
    E, p = _point("dp_mu_pot_x1m16_pos.json", 1e-20, 1e-38)
    F1, _ = _flux(E, p)
    save = dps.SOFT_JUMP_MIN
    try:
        dps.SOFT_JUMP_MIN = np.inf
        F0, _ = _flux(E, p)
    finally:
        dps.SOFT_JUMP_MIN = save
    assert np.array_equal(F0, F1)


def test_fast_scatterings_equalize():
    """g_D sum|U|^2 = 1e-8 at 1e-20 eV (screened, rate ~1e10 H0): equal flavor fractions at 100 TeV whatever the
    potential."""
    E, p = _point("dp_mu_pot_x1m08_pos.json", 1e-20, 1e-34)
    _, f = _flux(E, p)
    assert np.allclose(f, 1.0 / 3.0, atol=2e-3)


@pytest.mark.parametrize("log_g", [-4.5, -4.25, -4.0])
def test_jumps_against_dp_si(log_g):
    """eps' = 0 at 100 eV, where the soft scatterings are partial and the hard ones negligible: the dark-portal phase's
    jumps (forced) and dp_si's transport (forced, mass basis) give the same flavor fractions at 100 TeV (to 2e-3: the
    two differ in their redshift steps and dp_si's interference terms)."""
    E, p = _point("dp_mu_ul.json", 100.0, 10.0 ** log_g)
    _, f_dp = _flux(E, dict(p, cosmo_transport="dark_portal"))
    _, f_si = _flux(E, dict(p, cosmo_transport="si"))
    assert np.max(np.abs(f_dp - f_si)) < 2e-3, (f_dp, f_si)
