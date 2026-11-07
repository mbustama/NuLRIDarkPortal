"""
Tests of the beam along its trip (src/dp_trajectory.py; 2026-11-07, the paper's figure of the six species along the
trip). The cosmological phase's track (dark_portal_solver.propagate_spectrum's, dp_si's) ends on the solver's own
arrival -- on the dark portal's path with the longitudinal-mode factors applied along the way, which the solver
applies at z = 0 -- and is linear in its sources (two shells add up to their union). The local stages replay the
solver's kernels along one path: the Galaxy's line of sight ends on the crossing's populations for that line (checked
inside dp_trajectory.galaxy), the Sun's path on dp_spatial.transit_to_earth's for that path, and the Earth's chord on
dp_spatial._earth_chord_worker's for that chord.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core  # noqa: E402
import dark_portal_solver as dps  # noqa: E402
import dp_spatial  # noqa: E402
import dp_trajectory as tj  # noqa: E402

dp_core.HAS_TQDM = False

MIXING = dict(th12_deg=33.75878818411903, th13_deg=8.623062290036087, th23_deg=43.280093616242404, d_cp_deg=212.0,
              dm21_sq=7.537e-5, dm31_sq=2.511e-3)
GRID = {"E_min_eV": 1e13, "E_max_eV": 3e16, "n_E_bins": 16, "n_z_bins": 24}


def _gauge(model, m, g):
    config = {"model": {"name": model}, "cosmology": {"m_lightest_eV": 1e-6}, "standard_mixing": MIXING,
              "dark_portal": dict(m_Zd_eV=m, g_D=g, u_e=0.0, u_mu=0.0, u_tau=0.0, epsilon_prime=0.0),
              "source": {"p_index": 2.5}, "grid": GRID, "execution": {"n_cores": 1, "verbose": 0}}
    return dps.build_solver_params(config)


def _si(m, g):
    config = {"secret_interactions": {"enabled": True, "mediator": "scalar", "neutrino_type": "majorana",
                                      "m_phi_eV": m, "g_phi": g, "texture": "tau"},
              "cosmology": {"m_lightest_eV": 1e-6}, "standard_mixing": MIXING,
              "source": {"p_index": 2.5}, "grid": GRID, "execution": {"n_cores": 1, "verbose": 0}}
    return dps.build_solver_params(config)


def _tot(d, k):
    return d["F_nu"][k] + d["F_nb"][k]


@pytest.mark.parametrize("make", [lambda: _gauge("L_e-L_mu", 1e-26, 1e-27), lambda: _si(1e7, 1e-3),
                                  lambda: _gauge("L_e-L_mu", 1e7, 1e-3)],
                         ids=["le_lmu_ultralight_goldstone", "si_scalar_tau", "le_lmu_mev_dp_si"])
def test_track_ends_on_the_arrival_and_shells_add_up(make):
    E, p = make()
    u = tj.universe(E, p)
    assert u["z"][-1] == 0.0
    if u["si"]:
        ref = u["arrival"][0] + u["arrival"][1]
        np.testing.assert_allclose(_tot(u, -1), ref, rtol=1e-12, atol=1e-30 * ref.max())
    else:
        # the number of neutrinos in each state at z = 0 is the arrival's (the flavors are its decohered ones)
        tot = (u["P"][0] + u["P"][1]).sum(axis=1)
        np.testing.assert_allclose(_tot(u, -1).sum(axis=1), tot, rtol=1e-12, atol=1e-30 * tot.max())
    a = tj.universe(E, p, source_z=(2.0, 10.0))
    b = tj.universe(E, p, source_z=(-1.0, 2.0))
    s_all = u["F_nu"] + u["F_nb"]
    s_sum = a["F_nu"] + a["F_nb"] + b["F_nu"] + b["F_nb"]
    np.testing.assert_allclose(s_sum, s_all, rtol=1e-11, atol=1e-12 * s_all.max())


def test_local_stages_end_on_the_solver_for_their_path():
    """L_e - L_mu at 1e-20 eV, where the Sun's potential acts: the Sun's path ends on transit_to_earth's for that
    path; the chord's replay on _earth_chord_worker's from the same start; nothing changes the number of neutrinos
    outside the cosmological phase here (no Galactic processes in the CI's tests: the crossing is off)."""
    E, p = _gauge("L_e-L_mu", 1e-20, 1e-24)
    u = tj.universe(E, p, source_z=(1.8, 2.2))
    ga = tj.galaxy(E, p, u)
    np.testing.assert_allclose(_tot(ga, 0).sum(1), _tot(u, -1).sum(1), rtol=1e-12)
    su = tj.sun(E, u, ga, n_pos=12)
    for kk in range(2):
        np.testing.assert_allclose((su["F_nu"], su["F_nb"])[kk][-1], np.real(np.einsum('iaa->ia', su["rho"][kk])),
                                   rtol=1e-13)
    np.testing.assert_allclose(_tot(su, -1).sum(1), _tot(ga, -1).sum(1), rtol=1e-10)
    ea = tj.earth(E, u, su)
    np.testing.assert_allclose(_tot(ea, -1).sum(1), _tot(su, -1).sum(1), rtol=1e-10)
    np.testing.assert_allclose(ea["l_km"][-1], ea["L_km"], rtol=1e-12)


@pytest.mark.slow
def test_galaxy_replay_with_the_processes():
    """L_e - L_mu at 1e-26 eV, where the Milky Way's potential opens decays (Tycho's suite: the crossing on): the
    line's replay equals the crossing's populations for that line (dp_trajectory.galaxy raises otherwise), and the
    decays drain the flux along the line."""
    E, p = _gauge("L_e-L_mu", 1e-26, 1e-27)
    u = tj.universe(E, p, source_z=(1.8, 2.2))
    ga = tj.galaxy(E, p, u)
    assert ga["processes"]
    assert np.all(_tot(ga, -1).sum(1) <= _tot(ga, 0).sum(1) * (1 + 1e-12))
