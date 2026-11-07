"""
Tests of the simplified trips (dark_portal_solver.TRIPS; 2026-11-07, for the paper's Fig. 1): without a potential and
without collisions every trip is the full one; 'no_local' is flux_stops' arrival at the Milky Way; and where a
flavor-diagonal potential dominates at the detector, the earlier standard ('detector', the potential evaluated only
there) returns the source composition, while the full trip -- the neutrinos emitted far from the Sun, which then
turns on its L_e - L_mu potential adiabatically -- turns nu_2 into nu_tau.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core  # noqa: E402
import dark_portal_solver as dps  # noqa: E402
from flux_postprocessing import compute_energy_averaged_ratios as averaged  # noqa: E402

dp_core.HAS_TQDM = False

MIXING = dict(th12_deg=33.75878818411903, th13_deg=8.623062290036087, th23_deg=43.280093616242404, d_cp_deg=212.0,
              dm21_sq=7.537e-5, dm31_sq=2.511e-3)


def _params(m, g, **dp):
    config = {"model": {"name": "L_e-L_mu"}, "cosmology": {"m_lightest_eV": 1e-6}, "standard_mixing": MIXING,
              "dark_portal": dict(m_Zd_eV=m, g_D=g, u_e=0.0, u_mu=0.0, u_tau=0.0, epsilon_prime=0.0,
                                  neutrino_decay=False, relic_scattering=False, **dp),
              "grid": {"E_min_eV": 1e13, "E_max_eV": 3e16, "n_E_bins": 40, "n_z_bins": 30},
              "execution": {"n_cores": 1, "verbose": 0}}
    return dps.build_solver_params(config)


def _flux(E, p, trip):
    return dps.propagate_weighted(E, dict(p, trip=trip))


def test_config_key_and_unknown_trip():
    E, p = _params(1e-20, 1e-24, trip="detector")
    assert p["trip"] == "detector"
    with pytest.raises(ValueError):
        _params(1e-20, 1e-24, trip="elsewhere")


def test_without_potential_and_collisions_every_trip_is_the_full_one():
    E, p = _params(1e-20, 1e-24)
    p["potential_product"] = 0.0
    p["G_coupling"] = np.zeros((3, 3), dtype=np.complex128)
    ref = _flux(E, p, "full")
    for trip in ("no_local", "detector"):
        assert np.max(np.abs(_flux(E, p, trip) / ref - 1.0)) < 1e-9, trip


def test_no_local_is_the_arrival_at_the_milky_way():
    E, p = _params(1e-20, 1e-24)
    st = dps.flux_stops(E, p)
    assert np.max(np.abs(_flux(E, p, "no_local") / st["universe"] - 1.0)) < 1e-9


def test_detector_potential_returns_the_source_composition():
    """At 1e-20 eV the Sun's potential dominates at the Earth, ~1e8 times the vacuum term at 1 PeV: its eigenstates
    are the flavors, so the earlier standard keeps the source composition, while the full trip maps the mass states
    onto flavors (tests/test_flux_stops.py), nu_2 onto nu_tau."""
    E, p = _params(1e-20, 1e-24)
    src = np.array([p["f_e_S"], p["f_mu_S"], p["f_tau_S"]])
    F = _flux(E, p, "detector")
    assert np.max(np.abs(averaged(E, F) - src / src.sum())) < 1e-3
    assert np.max(np.abs(F.sum(axis=1) / _flux(E, dict(p, potential_product=0.0), "detector").sum(axis=1) - 1.0)) < 1e-9
    assert averaged(E, _flux(E, p, "full"))[2] > 0.3
