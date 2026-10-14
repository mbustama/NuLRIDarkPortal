"""
Tests of the eight benchmark models (src/dp_models.py; docs/dev/SCOPE_WIDENING_PLAN.md, step 2, 2026-10-14): each
model's charges, texture, couplings and visible width, and the solver against the potential-dominated closed form
for every texture -- e.g. L_e - L_mu's diag(1, -1, 0) has distinct eigenvalues, so flavor states do not mix and the
sources' composition arrives unchanged.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core  # noqa: E402
import dp_models as MD  # noqa: E402
import dp_absorption as A  # noqa: E402
import dark_portal_solver as dps  # noqa: E402

dp_core.HAS_TQDM = False


def test_model_table():
    """Charges, textures, couplings and widths as the gauge structures give them."""
    assert MD.spec("l_e-l_mu") == MD.MODELS["L_e-L_mu"]
    with pytest.raises(ValueError):
        MD.spec("B-L")
    assert np.allclose(MD.texture("L_e-L_mu"), np.diag([1, -1, 0]))
    assert np.allclose(MD.texture("B-3L_e"), np.diag([-3, 0, 0]))
    assert np.allclose(MD.texture("dark_portal", [0.5, 0.5, 0]), np.outer([0.5, 0.5, 0], [0.5, 0.5, 0]))
    g, eps = 1e-3, 1e-4
    assert MD.potential_product("dark_portal", g, eps) == g * eps
    assert MD.potential_product("L_mu-L_tau", g, eps) == g * eps
    assert abs(MD.potential_product("L_e-L_mu", g, eps) * MD.K_Z / g ** 2 - 1) < 1e-15
    G = MD.neutrino_coupling("L_mu-L_tau", g, eps)
    assert np.allclose(G, g * np.diag([0, 1, -1]) + A.kappa_z_mixing(eps) * np.eye(3))
    assert np.allclose(MD.neutrino_coupling("B-3/2(L_mu+L_tau)", g, eps), g * np.diag([0, -1.5, -1.5]))
    # visible widths: L_e - L_mu's e+e- at 3 MeV (vector coupling g); below 2 m_e nothing; B - 3L_tau none
    M = 3e6
    r = (A.M_E / M) ** 2
    expect = g * g * M / (12 * np.pi) * (1 + 2 * r) * np.sqrt(1 - 4 * r)
    assert abs(MD.visible_width("L_e-L_mu", M, g, eps) / expect - 1) < 1e-12
    assert MD.visible_width("L_e-L_mu", 1e6, g, eps) == 0.0
    assert MD.visible_width("B-3L_tau", M, g, eps) == 0.0
    assert abs(MD.visible_width("B-3L_e", M, g, eps) / (9 * expect) - 1) < 1e-12


def test_config_sets_the_model():
    """The model section sets the matter charge, texture, coupling, potential product and width; it refuses a
    conflicting matter charge or texture."""
    _, p = dps.build_solver_params({"model": {"name": "B-3L_e"}, "dark_portal": {"g_D": 1e-3, "m_Zd_eV": 3e6}})
    assert p["matter_charge"].sign == -1.0 and (p["matter_charge"].q_n, p["matter_charge"].q_pe) == (-1.0, 2.0)
    assert np.allclose(p["U_LRI_struct"], np.diag([-3, 0, 0]))
    assert np.allclose(p["G_coupling"], 1e-3 * np.diag([-3, 0, 0]))
    assert abs(p["potential_product"] - 1e-6 / MD.K_Z) < 1e-30
    with pytest.raises(ValueError, match="matter charge"):
        dps.build_solver_params({"model": {"name": "L_e-L_mu"}, "matter_charge": {"q_e": 1.0}})
    _, p = dps.build_solver_params({})
    assert p["model"] == "dark_portal" and p.get("G_coupling") is None


def _closed_form(S, src, anti=False):
    """The potential-dominated composition at the Earth: eigenvectors of the texture S, each degenerate subspace
    split by the projected vacuum term."""
    U = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    Us = U.conj() if anti else U
    Ss = np.conj(S) if anti else S
    M_rot = Us @ np.diag([0, 7.41e-5, 2.511e-3]).astype(complex) @ Us.conj().T
    w, V = np.linalg.eigh(Ss)
    W = np.zeros((3, 3), dtype=complex)
    k = 0
    while k < 3:
        grp = [k]
        while k + len(grp) < 3 and abs(w[k + len(grp)] - w[k]) < 1e-12 * max(1.0, abs(w).max()):
            grp.append(k + len(grp))
        B = V[:, grp]
        if len(grp) > 1:
            ww, vv = np.linalg.eigh(B.conj().T @ M_rot @ B)
            B = B @ vv
        W[:, grp] = B
        k += len(grp)
    pops = (np.abs(W) ** 2).T @ src
    return (np.abs(W) ** 2) @ pops


@pytest.mark.parametrize("model", ["L_e-L_mu", "L_e-L_tau", "L_mu-L_tau", "B-3L_e", "B-3L_mu", "B-3/2(L_mu+L_tau)"])
def test_potential_dominated_limit_per_model(model):
    """Each gauge model's solver run in the potential-dominated regime (an ultralight mediator, the cosmic mean
    and the local bodies all far above the vacuum term, negligible absorption and screening) against the closed
    form; L_e - L_mu and L_e - L_tau keep the sources' flavors (distinct eigenvalues)."""
    src = np.array([1 / 3, 2 / 3, 0.0])
    # the cosmic mean's potential ~1e-11 eV >> the vacuum term: g'^2 n_e0 / m^2 (direct), g eps' K n_n0 / m^2 (mixing);
    # the relics then screen the local bodies (screening mass >> 1e-20 eV) but not the cosmic mean, which keeps the
    # whole path potential-dominated with one texture
    g = 1e-15 if MD.spec(model)[0] == "direct" else 1e-10
    eps = 1e-20 if MD.spec(model)[0] == "mixing" else 0.0
    cfg = {"model": {"name": model}, "dark_portal": {"g_D": g, "epsilon_prime": eps, "m_Zd_eV": 1e-20},
           "grid": {"n_E_bins": 6, "n_z_bins": 30, "n_sun_paths": 2, "n_earth_chords": 4, "N_steps_earth": 10},
           "execution": {"n_cores": 1, "verbose": 0}, "source": {"f_e_S": 1 / 3, "f_mu_S": 2 / 3, "f_tau_S": 0.0}}
    E, params = dps.build_solver_params(cfg)
    params["show_progress"] = False
    setup = dps._flux_setup(E, params)
    F_nu, F_nb = dps._flux_from_setup(E, params, setup, (0.5, 0.5))
    S = MD.texture(model)
    for F, anti in ((F_nu, False), (F_nb, True)):
        fr = F / F.sum(axis=1, keepdims=True)
        assert np.max(np.abs(fr - _closed_form(S, src, anti))) < 1e-8, (model, anti)
    if model in ("L_e-L_mu", "L_e-L_tau"):
        assert np.allclose(_closed_form(S, src), src, atol=1e-12)
