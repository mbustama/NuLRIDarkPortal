"""
Tests of dark_portal_solver.flux_stops (2026-11-02, for notebooks/fluxes_at_earth.ipynb): the flux at the stops of the
trip -- arriving at the Milky Way, after it, near the Earth, at the detector -- against exact results: without a
potential every stop is the arrival; the transit and the Earth conserve the number of neutrinos; and where the Sun's
L_e - L_mu potential dominates, the adiabatic transit turns each mass state into one flavor (nu_2 into nu_tau for
neutrinos and antineutrinos alike, nu_1 and nu_3 into nu_e and nu_mu in opposite orders).
"""
import os
import sys

import numpy as np

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


def _pmns():
    d = np.radians
    t12, t13, t23, dcp = (d(MIXING[k]) for k in ("th12_deg", "th13_deg", "th23_deg", "d_cp_deg"))
    s12, c12, s13, c13, s23, c23 = np.sin(t12), np.cos(t12), np.sin(t13), np.cos(t13), np.sin(t23), np.cos(t23)
    e = np.exp(1j * dcp)
    return np.array([[c12 * c13, s12 * c13, s13 / e],
                     [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
                     [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])


def test_no_potential_every_stop_is_the_arrival():
    """With the potential off, nothing after the Universe changes the decohered mass states: every stop agrees."""
    E, p = _params(1e-20, 1e-24)
    p["potential_product"] = 0.0
    st = dps.flux_stops(E, p)
    ref = st["universe"]
    for k in ("galaxy", "earth", "detector"):
        assert np.max(np.abs(st[k] / ref - 1.0)) < 1e-9, k


def test_number_conserved_through_the_stops():
    """The transit and the Earth only rotate the flavors: each energy's total is the same at every stop."""
    E, p = _params(1e-20, 1e-24)
    st = dps.flux_stops(E, p)
    tot = {k: v.sum(axis=1) for k, v in st.items()}
    for k in ("galaxy", "earth", "detector"):
        assert np.max(np.abs(tot[k] / tot["universe"] - 1.0)) < 1e-9, k


def test_sun_maps_mass_states_onto_flavors():
    """At 1e-20 eV the Sun's L_e - L_mu potential, ~1e8 times the vacuum term at 1 PeV, dominates near the Earth and
    turns on adiabatically (over ~100 AU): f_tau there is the nu_2 share of the arrival, f_e = f_mu the mean of the
    nu_1 and nu_3 shares. The Galaxy's potential is negligible at this mass."""
    E, p = _params(1e-20, 1e-24)
    st = dps.flux_stops(E, p)
    P = np.abs(_pmns()) ** 2
    shares = np.linalg.solve(P, averaged(E, st["universe"]))         # the arrival's mass-state shares
    f_earth = averaged(E, st["earth"])
    assert abs(f_earth[2] - shares[1]) < 2e-3
    assert abs(f_earth[0] - 0.5 * (shares[0] + shares[2])) < 2e-3
    assert abs(f_earth[0] - f_earth[1]) < 1e-3
    assert np.max(np.abs(averaged(E, st["galaxy"]) - averaged(E, st["universe"]))) < 1e-3
    assert np.max(np.abs(st["detector"] / st["earth"] - 1.0)) < 1e-2
