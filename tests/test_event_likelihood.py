"""
Tests of the event likelihoods (docs/dev/EVENT_LIKELIHOOD_PLAN.md).

Step 1: the solver's per-direction output for several source compositions at once (propagate_sky_basis) recombines
into propagate_sky's for any (x, 1 - x, 0), extragalactic and Galactic, for the dark portal (Galactic emission in 3-D
and at the local potential) and for secret interactions; its Galactic sky average, weighted by the bands' pixel
shares, is galactic_allsky's.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dark_portal_solver as dps  # noqa: E402

DP = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 300 * 1.66e-54 / 0.05, "epsilon_prime": 0.1}
SI = {"enabled": True, "mediator": "scalar", "texture": "tau", "m_phi_eV": 5e6, "g_phi": 0.05}


def _sky_cfg(x, x_gal, emission="3d", **over):
    cfg = {"grid": {"n_E_bins": 8, "n_z_bins": 40, "n_sun_paths": 2, "N_steps_earth": 8, "E_min_eV": 1e13,
                    "E_max_eV": 1e16},
           "source": {"f_e_S": x, "f_mu_S": 1.0 - x, "f_tau_S": 0.0},
           "execution": {"n_cores": 1, "verbose": 0},
           "sky": {"output": "bands", "nside": 8, "extragalactic_norm": 1e-18, "extragalactic_norm_E_GeV": 1e5},
           "galactic": {"enabled": True, "emission": emission, "f_e_S_gal": x_gal}}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


@pytest.mark.parametrize("case", ["dark portal, 3-D emission", "dark portal, local emission", "secret interactions"])
def test_sky_basis_recombines_into_propagate_sky(case):
    x, x_gal = 0.3, 0.7
    over = {"secret_interactions": SI} if case == "secret interactions" else {"dark_portal": DP}
    E, params = dps.build_solver_params(_sky_cfg(x, x_gal, "local" if "local" in case else "3d", **over))
    params["show_progress"] = False
    sky = dps.propagate_sky(E, params)
    basis = dps.propagate_sky_basis(E, params)
    nb = len(sky["dec_band_edges"]) - 1
    assert basis["bands_nu"].shape == (2, nb, len(E), 3) and basis["galactic_bands_nu"].shape == (2, nb, len(E), 3)
    assert np.array_equal(basis["dec_band_edges"], sky["dec_band_edges"])
    assert basis["meta"]["extragalactic_scale"] == sky["meta"]["extragalactic_scale"]
    for kind in ("nu", "nubar"):
        eg = x * basis["bands_" + kind][0] + (1.0 - x) * basis["bands_" + kind][1]
        assert np.allclose(eg, sky["bands_" + kind], rtol=1e-12, atol=0)
        ga = x_gal * basis["galactic_bands_" + kind][0] + (1.0 - x_gal) * basis["galactic_bands_" + kind][1]
        assert np.allclose(ga, sky["galactic_bands_" + kind], rtol=1e-12, atol=0)
        allsky = np.tensordot(basis["galactic_band_share"], ga, axes=(0, 0))
        assert np.allclose(allsky, sky["galactic_allsky_" + kind], rtol=1e-12, atol=0)
    both = basis["galactic_bands_nu"] + basis["galactic_bands_nubar"]
    G = dps.galactic_allsky(E, params)
    assert np.allclose(np.tensordot(basis["galactic_band_share"], both, axes=(0, 1)), G, rtol=1e-12, atol=0)
