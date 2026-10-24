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
        _soft_jumps(E, 2.0, J2, Gd2, GG2, lam2, 0.0, 1.0, 1e-30, G_unit, 1e30, F, np.zeros((1, 3)), False)
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
    _soft_jumps(E, 2.0, J2, Gd2, GG2, lam2, 0.0, 1.0, 1e-30, G_unit, 1e3, F, np.zeros((1, 3)), False)   # undamped pair 1-2: rate x dt 1e-7
    assert np.allclose(F, [[0.7, 0.2, 0.1]], atol=1e-7)
    J2, Gd2, GG2, lam2 = _tables(J, lam)
    F = np.array([[0.7, 0.2, 0.1]])
    _soft_jumps(E, 2.0, J2, Gd2, GG2, lam2, 0.0, 1.0, 1e-30, G_unit, 1e12, F, np.zeros((1, 3)), False)
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
    """A potential plane at g_D sum|U|^2 = 1e-16 and 1e-12 eV (the scatterings far below the expansion rate): the
    same flux with the jumps' code as without it."""
    E, p = _point("dp_mu_pot_x1m16_pos.json", 1e-12, 1e-38)
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
    potential. The decay into the mediator off (it acts there too, 2026-10-24; the point is beyond the models' range
    and in a dropped potential plane): this checks the jumps."""
    E, p = _point("dp_mu_pot_x1m08_pos.json", 1e-20, 1e-34, neutrino_decay=False)
    _, f = _flux(E, p)
    assert np.allclose(f, 1.0 / 3.0, atol=2e-3)


@pytest.mark.slow     # 30-90 s on CI's runners (2026-10-24): Tycho's suite runs it
@pytest.mark.parametrize("log_g", [-4.5, -4.25, -4.0])
def test_jumps_against_dp_si(log_g):
    """eps' = 0 at 100 eV, where the soft scatterings are partial and the hard ones negligible: the dark-portal phase's
    jumps (forced) and dp_si's transport (forced, mass basis) give the same flavor fractions at 100 TeV (to 2e-3: the
    two differ in their redshift steps and dp_si's interference terms)."""
    E, p = _point("dp_mu_ul.json", 100.0, 10.0 ** log_g)
    _, f_dp = _flux(E, dict(p, cosmo_transport="dark_portal"))
    _, f_si = _flux(E, dict(p, cosmo_transport="si"))
    assert np.max(np.abs(f_dp - f_si)) < 2e-3, (f_dp, f_si)


def test_switch_off_and_standard_model_asimov():
    """dark_portal.relic_scattering false: below 1 keV the dark-portal phase without the scatterings (the code before
    2026-10-22); the analysis' Asimov data come from a model with it off (the Standard Model at a split such as
    g_D sum|U|^2 = 1e-8, where they act even at eps' -> 0)."""
    from analysis.config import load
    E, p = _point("dp_mu_pot_x1m08_pos.json", 1e-10, 1e-11, relic_scattering=False)
    assert dps._cosmology_mode(p, E) == "dark_portal"
    cfg = load(os.path.join(REPO, "analysis", "configs", "campaign_ml", "proj_future", "dp_mu_pot_x1m08_pos.json"))
    truth = cfg._truth_model()
    assert truth.base["dark_portal"]["relic_scattering"] is False
    assert cfg.model().base["dark_portal"].get("relic_scattering", True) is True


def test_stiff_bin_keeps_its_losses():
    """dp_si._bin_matrix (2026-10-24): in-bin conversions at 1e8-1e30 per step against losses of order 1e-2 used to
    leave only their rounding in the bin's exponential (flux loss up to 100% at m_Zd < 1e-5 eV); rescaled, e^M is
    the fast-mixing limit pi e^{-pi.l} 1^T to ~1e-7 at every rate."""
    import dp_si
    from scipy.linalg import expm, null_space
    rng = np.random.default_rng(2)
    for big in (1e10, 1e16, 1e20, 1e30):
        for _ in range(10):
            tau = big * rng.uniform(0.05, 1.0, 3)
            lose = rng.uniform(1e-3, 0.5, 3)
            Qh = rng.uniform(0.0, 1.0, (3, 3))
            np.fill_diagonal(Qh, 0.0)
            Qh /= Qh.sum(axis=0)
            Mx = np.zeros((7, 7))
            dp_si._bin_matrix(tau, lose, Qh * (1.0 - lose / tau), np.zeros((3, 3)), Mx)
            A = np.ascontiguousarray(Mx[:3, :3])
            Em = dp_si._phi_eig(A)[0] if np.abs(A).max() > dp_si.STIFF_BIN else expm(A)
            C = Qh * tau
            np.fill_diagonal(C, -tau)
            pi = null_space(C / big)[:, 0]
            pi /= pi.sum()
            ref = np.outer(pi, np.ones(3)) * np.exp(-(pi * lose).sum())
            assert np.abs(Em - ref).max() < 1e-6


def test_jump_step_keeps_the_total_at_any_rate():
    """_soft_jumps' step (2026-10-24): through eigh the uniform mode's zero eigenvalue came out as ~1e-16 of the
    largest rate, which beyond rates x dt ~ 1e16 removed up to half of the flux (the Zeno-transition band). The exact
    3-state solution keeps the total to rounding at every rate and matches expm of the rate matrix."""
    from scipy.linalg import expm
    rng = np.random.default_rng(3)
    E = np.array([1e14])
    for scale in (1e-3, 1.0, 1e3, 1e10, 1e16, 1e20, 1e30):
        for _ in range(10):
            R = scale * rng.uniform(0.0, 1.0, 3) * (rng.uniform(size=3) > 0.2)
            J = np.zeros((2, 3, 3))
            J[:, 1, 0] = J[:, 0, 1] = R[0]
            J[:, 2, 0] = J[:, 0, 2] = R[1]
            J[:, 2, 1] = J[:, 1, 2] = R[2]
            F = rng.uniform(0.0, 1.0, (1, 3))
            F0 = F.copy()
            _soft_jumps(E, 1.0, J, np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), 0.0, 1.0, 1.0, 1.0, 1.0, F,
                        np.zeros((1, 3)), False)
            assert abs(F.sum() - F0.sum()) <= 1e-14 * F0.sum()
            A = np.array([[0, R[0], R[1]], [R[0], 0, R[2]], [R[1], R[2], 0]], dtype=float)
            A -= np.diag(A.sum(axis=0))
            if scale <= 1e3:
                assert np.allclose(F[0], expm(A) @ F0[0], rtol=1e-10, atol=1e-14)
