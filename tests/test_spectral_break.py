"""
The broken power law (Mauricio, 2026-10-15: IceCube's 2026 combined fit bends below ~25 TeV; analysis/model.py):
the sources' emission window in the solver (source.injection_E_min_GeV / injection_E_max_GeV), and the model's two
pieces, joined continuously at the break.
"""
import copy
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import dp_si as si  # noqa: E402
from test_si import H0, _grid, _transport_kw  # noqa: E402

E_B_GEV = 10 ** 4.39                     # IceCube's 2026 break (NuFlavorSingleSource's CF26)
N_REF, E_REF_GEV = 1.7e-16, 1.5e4


def _solver(eps=0.0, **source):
    """A small solver configuration; eps' > 0 for the potential's coupling product (with an ultralight mediator:
    eps' also couples the mediator to the Z, which absorbs strongly at the default 2 MeV)."""
    s = {"grid": {"n_E_bins": 60, "E_min_eV": 1e12, "E_max_eV": 3e16, "n_z_bins": 60, "n_sun_paths": 2,
                  "n_earth_chords": 2, "N_steps_earth": 8},
         "sky": {"nside": 4, "extragalactic_norm": N_REF, "extragalactic_norm_E_GeV": E_REF_GEV},
         "galactic": {"enabled": False},
         "dark_portal": {"g_D": 1e-60, "epsilon_prime": eps, "u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0}}
    if source:
        s["source"] = dict(source)
    return s


def test_si_windows_add_up_to_the_whole_power_law():
    """The MeV transport is linear in its sources: the two windows' fluxes add up to the whole power law's to
    rounding; nothing moves up in energy (the lower piece is zero above the break), while the upper piece's
    secondaries land below it. Without coupling the windows cut the free-streaming flux exactly."""
    kw = _transport_kw()
    E = _grid()
    Eb = E_B_GEV * 1e9
    gm = si.mass_basis_couplings(si.flavor_couplings("tau", 0.1), kw["U_pmns"])
    F = si.propagate_si(E, M=5e6, g_mass=gm, **kw)
    Fu = si.propagate_si(E, M=5e6, g_mass=gm, inj_window=(Eb, np.inf), **kw)
    Fl = si.propagate_si(E, M=5e6, g_mass=gm, inj_window=(0.0, Eb), **kw)
    assert np.max(np.abs(Fu + Fl - F)) < 1e-12 * np.max(F)
    assert np.all(Fl[E >= Eb] == 0.0) and np.all(Fl[E < Eb] > 0.0)
    assert np.all(Fu[E < Eb] > 0.0)                          # secondaries of absorptions above the break
    F0 = si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, kw["U_pmns"], kw["rho_src"], 2.5, 0.0, 1.0)
    F0u = si.free_streaming_flux(E, 4.0, H0, 0.315, 0.685, kw["U_pmns"], kw["rho_src"], 2.5, 0.0, 1.0,
                                 inj_window=(Eb, np.inf))
    assert np.array_equal(F0u[E >= Eb], F0[E >= Eb]) and np.all(F0u[E < Eb] == 0.0)
    gz = si.mass_basis_couplings(si.flavor_couplings("tau", 1e-30), kw["U_pmns"])
    assert np.max(np.abs(si.propagate_si(E, M=5e6, g_mass=gz, inj_window=(Eb, np.inf), **kw) - F0u)) < 1e-12 * F0u.max()


def test_dark_portal_windows_add_up_to_the_whole_power_law():
    """The dark portal's cosmological kernel (with a potential that converts flavors): the windows' fluxes add up
    to the whole power law's to rounding, and the lower one is zero above the break."""
    from analysis.model import FlavorModel
    m = FlavorModel(_solver(eps=0.1), free_names=("m_Zd_eV", "g_D_eps_trace"))
    v = {"m_Zd_eV": 1e-20, "g_D_eps_trace": 1e-36}
    E, F, _, _ = m._fluxes_exact(v)
    _, Fu, _, _ = m._fluxes_exact(dict(v, injection_E_min_GeV=E_B_GEV))
    _, Fl, _, _ = m._fluxes_exact(dict(v, injection_E_max_GeV=E_B_GEV))
    above = E >= E_B_GEV * 1e9
    assert np.max(np.abs(Fu + Fl - F)) < 1e-12 * np.max(F)
    assert np.all(Fl[above] == 0.0) and np.all(Fl[~above] > 0.0)
    _, F0, _, _ = m._fluxes_exact({"m_Zd_eV": 1e-20, "g_D_eps_trace": 1e-60})
    assert np.max(np.abs(F[:, 0] / F[:, 1] - F0[:, 0] / F0[:, 1])) > 1e-3     # the potential does act here


def _bpl(E_GeV, p2, p1):
    """The broken power law's shape (per flavor, nu + nubar, at norm_astro = 1)."""
    x = np.asarray(E_GeV) / E_REF_GEV
    return N_REF * np.where(E_GeV >= E_B_GEV, x ** -p2, (E_B_GEV / E_REF_GEV) ** (p1 - p2) * x ** -p1)


def test_model_without_new_physics_is_the_broken_power_law():
    """Without new physics the model's extragalactic flux is IceCube's broken power law: each piece's shape exact,
    the normalization the closed form's to the solver's redshift grid (~3e-3 here), continuous at the break; at the
    pieces' index nodes and between them."""
    from analysis.model import FlavorModel
    m = FlavorModel(_solver(E_break_GeV=E_B_GEV, p_index_low=1.31), free_names=("p_index", "p_index_low"),
                    p_nodes=[2.2, 2.5, 2.8], p1_nodes=[0.5, 1.5, 2.5])
    for p2, p1 in ((2.5, 1.5), (2.735, 1.31)):
        E, F = m.extragalactic({"p_index": p2, "p_index_low": p1})
        E_GeV = E / 1e9
        r = F.sum(axis=1) / 3.0 / _bpl(E_GeV, p2, p1)
        up, lo = E_GeV >= E_B_GEV, E_GeV < E_B_GEV
        assert np.ptp(r[up]) < 1e-9 and np.ptp(r[lo]) < 1e-9            # each piece's shape
        assert abs(r[up][0] - 1.0) < 1e-2 and abs(r[lo][0] - 1.0) < 1e-2  # its normalization
        assert abs(r[up][0] / r[lo][0] - 1.0) < 1e-2                      # joined at the break


def test_equal_indices_give_the_single_power_law():
    """With p_index_low = p_index the two pieces make up the single power law: at an index node exactly (fluxes;
    per-direction bands to their single-precision rounding), between nodes both interpolations within their error
    of the run at that index. The point has a strong cascade (eps' = 0.1 couples the ultralight mediator to the Z):
    the upper piece's secondaries carry 15-60% of the flux below the break."""
    from analysis.model import FlavorModel
    nodes = [2.2, 2.5, 2.8]
    for sky in (False, True):
        mb = FlavorModel(_solver(eps=0.1, E_break_GeV=E_B_GEV), free_names=("norm_astro", "p_index", "p_index_low"),
                         p_nodes=nodes, p1_nodes=nodes, sky=sky)
        mp = FlavorModel(_solver(eps=0.1), free_names=("norm_astro", "p_index"), p_nodes=nodes, sky=sky)
        for p, tol in ((2.5, 1e-12 if not sky else 1e-6), (2.63, 0.03)):
            v = {"norm_astro": 1.3, "p_index": p, "p_index_low": p, "m_Zd_eV": 1e-20, "g_D_eps_trace": 1e-36}
            vp = {k: x for k, x in v.items() if k != "p_index_low"}
            if sky:
                _, b_nu, b_nb = mb.band_fluxes(v)
                _, c_nu, c_nb = mp.band_fluxes(vp)
                assert np.max(np.abs(b_nu / c_nu - 1)) < tol and np.max(np.abs(b_nb / c_nb - 1)) < tol
            else:
                _, Fb, _, _ = mb.fluxes(v)
                _, Fp, _, _ = mp.fluxes(vp)
                K = mb._piece_scale(p, mb.p_nodes)
                exact = mb._scale(p) * mp._fluxes_exact(vp)[1]
                assert np.max(np.abs(Fb / (K * Fp) - 1)) < tol
                assert np.max(np.abs(Fb / exact - 1)) < tol and np.max(np.abs(K * Fp / exact - 1)) < tol


def test_broken_power_law_configuration():
    """p_index_low takes its own nodes and must come with a break; the model refuses windows set by hand."""
    from analysis import config as config_mod
    from analysis.model import FlavorModel
    raw = {"name": "bpl", "data": {"events": {"dr2": {"configs": ["IC79"]}}, "flavor_band_eV": [1e13, 3e16]},
           "solver": _solver(E_break_GeV=E_B_GEV, p_index_low=1.31),
           "parameters": {"norm_astro": {"start": 1.0, "bounds": [0.1, 10.0], "space": "log10"},
                          "p_index": {"start": 2.7, "bounds": [2.0, 3.0], "nodes": 3},
                          "p_index_low": {"start": 1.3, "bounds": [0.5, 2.5], "nodes": 3}},
           "fit": {"n_random": 0}}
    cfg = config_mod.from_dict(copy.deepcopy(raw))
    assert cfg.p_nodes() == [2.0, 2.5, 3.0] and cfg.p1_nodes() == [0.5, 1.5, 2.5]
    model = cfg.model(cache_size=1)
    assert model.E_break == E_B_GEV and model.default("p_index_low") == 1.31
    bad = copy.deepcopy(raw)
    del bad["solver"]["source"]["E_break_GeV"]
    with pytest.raises(ValueError, match="E_break_GeV"):
        config_mod.from_dict(bad)
    with pytest.raises(ValueError, match="emission windows"):
        FlavorModel(_solver(E_break_GeV=E_B_GEV, injection_E_min_GeV=1e4))
