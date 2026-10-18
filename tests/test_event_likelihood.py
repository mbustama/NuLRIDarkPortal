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


# Steps 2-3: the folds as fixed linear maps, and the HESE detector systematics per bin (synthetic Monte Carlo)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
from IceCubeRates import flux_adapters as FA  # noqa: E402
from IceCubeRates.northern_muons.dr2 import DR2Tracks, LinearFold  # noqa: E402
from IceCubeRates.starting_events import hese  # noqa: E402
import hese_reduce  # noqa: E402

E_NODES = np.logspace(4.0, 7.6, 300)                    # GeV
BANDS = np.arcsin(np.linspace(-1.0, 1.0, 21))


def _fake_mc(path, n=4000, seed=3):
    rng = np.random.default_rng(seed)
    types = rng.choice([12, -12, 14, -14, 16, -16, 13], n)
    E = 10 ** rng.uniform(4.5, 7.0, n)
    eb, zb, lb = hese_reduce.release_bins()
    d = dict(mc_type=types.astype(np.int16), mc_E_GeV=E, mc_cos_zenith=rng.uniform(-1, 1, n),
             mc_interaction=rng.integers(1, 4, n).astype(np.int8), mc_weight=rng.uniform(1e5, 1e6, n),
             mc_E_dep_GeV=E * rng.uniform(0.5, 1.0, n), mc_zenith_reco=rng.uniform(0, np.pi, n),
             mc_morphology=rng.integers(0, 3, n).astype(np.int8), mc_length_m=10 ** rng.uniform(1, 3, n),
             mc_pion_flux=rng.uniform(1e-20, 1e-18, n), mc_kaon_flux=rng.uniform(1e-20, 1e-18, n),
             mc_prompt_flux=rng.uniform(1e-21, 1e-19, n), mc_conv_veto=rng.uniform(0.1, 1, n),
             mc_prompt_veto=rng.uniform(0.1, 1, n), mc_muon_weight=np.where(types == 13, 1e-9, 0.0),
             data_E_dep_GeV=np.array([1e5, 2e5, 3e6, 7e4]), data_zenith_reco=np.array([0.3, 2.0, 1.0, 1.5]),
             data_morphology=np.array([0, 1, 2, 0], dtype=np.int8), data_length_m=np.array([0.0, 0.0, 50.0, 0.0]),
             livetime_s=2.0e8, energy_bins_GeV=eb, zenith_bins_rad=zb, length_bins_m=lb)
    np.savez(path, **d)
    grids = {"dom_eff": np.array([0.8, 0.85, 0.9, 0.95, 0.99, 1.04, 1.09, 1.14, 1.19]),
             "head_on": np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])}
    det = dict(E_dep_GeV=d["mc_E_dep_GeV"], names=np.array(list(grids)), nominal=np.array([0.99, 0.0]),
               sigma=np.array([0.1, 0.5]))
    for name, g in grids.items():
        det["grid_" + name] = g
        x = g - (0.99 if name == "dom_eff" else 0.0)
        for comp in ("Astro", "Conv", "Prompt"):
            a, b = rng.normal(0, 0.5, n), rng.normal(0, 0.3, n)        # smooth, event by event
            det[f"log10w_{name}_{comp}"] = (a[None, :] * x[:, None] + b[None, :] * x[:, None] ** 2).astype(np.float32)
    np.savez(str(path) + ".det.npz", **det)
    return d


def _windowed(per_type, gamma):
    def flux(flavor, anti, E, dec):
        E = np.asarray(E)
        return per_type * (E / 1e5) ** (-gamma) * ((E >= E_NODES[0]) & (E <= E_NODES[-1]))
    return flux


def test_node_interpolation():
    """Exact for E^-2.5, linear in the node values, zero outside the nodes."""
    E = np.array([2e4, 3.3e5, 1e7, 1e3, 1e8])
    j, lo, hi = FA.node_interpolation(E, E_NODES)
    F = (E_NODES / 1e5) ** -2.5
    assert np.allclose((lo * F[j] + hi * F[j + 1])[:3], (E[:3] / 1e5) ** -2.5, rtol=1e-13, atol=0)
    assert np.all(lo[3:] == 0) and np.all(hi[3:] == 0)


def test_hese_linear_response(tmp_path):
    """The sparse maps equal the per-event weighting: exact for E^-2.5 on the nodes; the columns' order (kind, flavor,
    band, node) checked by lighting one (kind, flavor, band) at a time."""
    d = _fake_mc(tmp_path / "mc.npz")
    ev = hese.HESEEvents(str(tmp_path / "mc.npz"))
    lin = hese.LinearResponse(ev, E_NODES, BANDS)
    F = 1e-18 * (E_NODES / 1e5) ** -2.5
    Fb = np.broadcast_to(F[None, :, None], (20, len(E_NODES), 3)).copy()
    ref = ev.flat_histogram(ev.expected(_windowed(1e-18, 2.5)))
    np.testing.assert_allclose(lin(Fb, Fb), ref, rtol=1e-12, atol=1e-14 * ref.max())
    for anti, flavor, band in ((False, 1, 3), (True, 0, 17), (True, 2, 10)):
        Fn, Fa = np.zeros_like(Fb), np.zeros_like(Fb)
        (Fa if anti else Fn)[band, :, flavor] = F
        sel = (ev.anti == anti) & (ev.flavor == flavor) & (np.clip(np.searchsorted(BANDS, ev.dec) - 1, 0, 19) == band)
        per = np.where(sel, ev.expected(_windowed(1e-18, 2.5)), 0.0)
        ref1 = ev.flat_histogram(per)
        assert ref1.sum() > 0
        np.testing.assert_allclose(lin(Fn, Fa), ref1, rtol=1e-12, atol=1e-14 * ref1.max())
    assert d["mc_type"].size == len(ev.E)


def test_hese_detector_per_bin(tmp_path):
    """BinDetector: exact on the grid (one systematic moved), as the per-event weights; its gradient by finite
    differences; with the 'Astro' weights the linear maps per grid value equal the per-event histograms."""
    _fake_mc(tmp_path / "mc.npz")
    ev = hese.HESEEvents(str(tmp_path / "mc.npz"))
    ev.load_detector(str(tmp_path / "mc.npz") + ".det.npz")
    per = ev.expected(_windowed(1e-18, 2.5))
    H0 = ev.flat_histogram(per)
    H_nodes = {s: np.stack([ev.flat_histogram(per * 10.0 ** ev.det[f"log10w_{s}_Astro"][k].astype(np.float64))
                            for k in range(len(g))]) for s, g in ev.det_grid.items()}
    bd = hese.BinDetector(ev.det_grid, H0, H_nodes)
    for s, g in ev.det_grid.items():
        for x in g[[0, 3, -1]]:
            exact = ev.flat_histogram(per * ev.detector_factor("Astro", {s: x}))
            np.testing.assert_allclose(bd({s: x}), exact, rtol=1e-10, atol=1e-14 * exact.max())
    vals = {"dom_eff": 1.02, "head_on": 0.3}
    H, dH = bd(vals, grad=True)
    for s in vals:
        h = 1e-6
        up, dn = dict(vals), dict(vals)
        up[s] += h
        dn[s] -= h
        np.testing.assert_allclose(dH[s], (bd(up) - bd(dn)) / (2 * h), rtol=1e-5, atol=1e-9 * np.abs(dH[s]).max())
    lin = hese.LinearResponse(ev, E_NODES, BANDS)
    F = 1e-18 * (E_NODES / 1e5) ** -2.5
    Fb = np.broadcast_to(F[None, :, None], (20, len(E_NODES), 3)).copy()
    at = lin.at_nodes(Fb, Fb)
    for s in ev.det_grid:
        np.testing.assert_allclose(at[s], H_nodes[s], rtol=1e-11, atol=1e-14 * H_nodes[s].max())


def test_lagrange4_cubics():
    g = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])
    f = lambda x: 2.0 - 3.0 * x + 0.5 * x ** 2 + 1.7 * x ** 3  # noqa: E731
    df = lambda x: -3.0 + x + 5.1 * x ** 2  # noqa: E731
    for x in (0.6, 0.73, 1.0, 1.27, 1.4):
        i, c, dc = hese.lagrange4(g, x)
        assert abs(c @ f(g[i:i + 4]) - f(x)) < 1e-12 and abs(dc @ f(g[i:i + 4]) - df(x)) < 1e-10


def test_dr2_linear_fold_exact():
    """LinearFold equals DR2Tracks.expected for E^-2.5 on the nodes (whatever the bands), on a selection of bins."""
    tr = DR2Tracks()
    nodes = np.logspace(2.0, 8.7, 400)
    F = 2e-18 * (nodes / 1e5) ** -2.5

    def pl(E, dlo, dhi):
        E = np.asarray(E, dtype=np.float64)
        return FA.power_law(2e-18, 2.5)(E, dlo, dhi) * ((E >= nodes[0]) & (E <= nodes[-1]))[None, :]
    de = tr.dec_edges("IC79")
    sel = np.zeros((len(de) - 1, len(tr.reco_edges) - 1), dtype=bool)
    sel[np.degrees(de[:-1]) >= -5.0, :] = True
    sel[:, tr.reco_edges[:-1] < 3.5 - 1e-9] = False
    fold = LinearFold(tr, "IC79", nodes, BANDS, select=sel)
    ref = tr.expected(pl, configs=["IC79"])["per_config"]["IC79"][sel]
    np.testing.assert_allclose(fold(np.broadcast_to(F[None, :], (20, len(nodes)))), ref, rtol=1e-11,
                               atol=1e-14 * ref.max())


# Step 4: the event term (synthetic HESE Monte Carlo, the real DR2 responses and counts) and its configuration
def test_poisson_continuation():
    """The deviance equals 2 sum [mu - n + n ln(n/mu)] where expectations are physical; below the floor it continues
    smoothly (value and gradient continuous), finite for negative mu; the gradient matches finite differences."""
    from analysis.events import _poisson, MU_FLOOR
    n = np.array([0.0, 1.0, 5.0, 2.0])
    mu = np.array([0.3, 0.8, 7.0, 3.1])
    ref = 2.0 * np.sum(mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0))
    assert abs(_poisson(mu, n) - ref) < 1e-12
    for m_low in (np.array([-0.1, MU_FLOOR * 1.0 - 1e-9, 7.0, 3.1]), np.array([0.3, -2.0, 1e-5, 3.1])):
        f, g = _poisson(m_low, n, grad=True)
        assert np.isfinite(f) and np.all(np.isfinite(g))
        for i in range(4):                              # bin by bin (the continued deviance can be ~1e6)
            h = 1e-7 * max(1.0, abs(m_low[i]))
            up = _poisson(m_low[i:i + 1] + h, n[i:i + 1])
            dn = _poisson(m_low[i:i + 1] - h, n[i:i + 1])
            assert abs(g[i] - (up - dn) / (2 * h)) < 1e-5 * max(1.0, abs(g[i]))
    edge = np.array([0.0, MU_FLOOR, 5 * MU_FLOOR, 2 * MU_FLOOR])
    for eps in (1e-10, -1e-10):
        assert abs(_poisson(edge + eps, n) - _poisson(edge, n)) < 1e-4


def _event_model(tmp_path):
    from analysis.model import FlavorModel
    from analysis.events import EventLikelihood
    _fake_mc(tmp_path / "mc.npz")
    solver = {"grid": {"n_E_bins": 60, "E_min_eV": 1e12, "E_max_eV": 3e16, "n_z_bins": 60, "n_sun_paths": 2,
                       "N_steps_earth": 8},
              "sky": {"nside": 4, "extragalactic_norm": 1.7e-16, "extragalactic_norm_E_GeV": 1.5e4},
              "dark_portal": {"g_D": 1e-60, "epsilon_prime": 0.0}}
    m = FlavorModel(solver, free_names=("norm_astro", "p_index"), p_nodes=[2.2, 2.5, 2.8], sky=True,
                    flavor_band_eV=[1e13, 3e16])
    spec = {"hese": {"mc": str(tmp_path / "mc.npz"), "detector": str(tmp_path / "mc.npz") + ".det.npz"},
            "dr2": {"configs": ["IC79"]}}
    return m, EventLikelihood(m, spec)


def test_event_term_profile_and_asimov(tmp_path):
    """The profile's gradient against finite differences; Asimov data at a point give chi^2 = 0 there and more
    elsewhere; the warm-started profile follows the astrophysical flux (no stall after a first fit)."""
    m, ev = _event_model(tmp_path)
    A = ev.astro({"norm_astro": 1.0, "p_index": 2.5})
    rng = np.random.default_rng(5)
    x = ev.centre + 0.2 * ev.sigma * rng.standard_normal(len(ev.names))
    for i, (lo, hi) in enumerate(ev.bounds):
        x[i] = min(max(x[i], -np.inf if lo is None else lo + 1e-6), np.inf if hi is None else hi - 1e-6)
    f, g = ev._expected_bins(x, A, grad=True)
    for i in range(len(x)):
        h = 1e-6 * ev.sigma[i]
        up, dn = x.copy(), x.copy()
        up[i] += h
        dn[i] -= h
        fd = (ev._expected_bins(up, A)[0] - ev._expected_bins(dn, A)[0]) / (2 * h)
        assert abs(g[i] - fd) < 1e-4 * max(1.0, abs(fd)), ev.names[i]
    truth = {"norm_astro": 1.4, "p_index": 2.6}
    ev.set_asimov(truth)
    assert abs(ev.chi2(truth)) < 1e-8
    away = ev.chi2({"norm_astro": 0.6, "p_index": 2.6})
    assert away > 0.1
    z_before = ev._z0.copy()
    ev.chi2({"norm_astro": 1.0, "p_index": 2.6})
    assert np.max(np.abs(ev._z0 - z_before)) > 1e-4 and ev.last["profile_converged"]


def test_event_term_exact_gradient(tmp_path):
    """EventLikelihood.chi2_and_grad (2026-10-18): the value is chi2's, the gradient the central difference of chi2
    with the nuisances re-profiled -- the exact evaluation's derivative in the expectations plus the nuisances'
    response through the per-bin profile -- away from the Asimov truth (residuals and detector pulls nonzero)."""
    m, ev = _event_model(tmp_path)
    ev.set_asimov({"norm_astro": 1.4, "p_index": 2.6})
    names = ["norm_astro", "p_index"]
    for v in ({"norm_astro": 1.1, "p_index": 2.45}, {"norm_astro": 1.9, "p_index": 2.7}):
        f, g = ev.chi2_and_grad(v, names)
        assert abs(f - ev.chi2(v)) < 1e-9 * max(1.0, f)
        for j, n in enumerate(names):
            h = 1e-5 * max(1.0, abs(v[n]))
            fd = (ev.chi2(dict(v, **{n: v[n] + h})) - ev.chi2(dict(v, **{n: v[n] - h}))) / (2 * h)
            assert abs(g[j] - fd) < 1e-4 * max(1.0, abs(fd)), (v, n, g[j], fd)


def test_events_configuration(tmp_path):
    """data.events and data.spectrum exclude each other; the events need the flux in physical units; the identity
    changes with the events; the model is in sky mode with the flavor band."""
    from analysis.config import from_dict
    base = {"name": "t", "data": {"surface": "pion_2050_case_5", "events": {"dr2": True},
                                  "flavor_band_eV": [1e13, 3e16]},
            "solver": {"grid": {"n_E_bins": 40, "E_min_eV": 1e12, "E_max_eV": 3e16},
                       "sky": {"nside": 4, "extragalactic_norm": 1.7e-16, "extragalactic_norm_E_GeV": 1.5e4}},
            "parameters": {"norm_astro": {"start": 1.0, "bounds": [0.01, 100.0], "space": "log10"}},
            "output_dir": str(tmp_path)}
    cfg = from_dict(base)
    m = cfg.model(cache_size=1)
    assert m.sky and m.flavor_band == (1e13, 3e16)
    other = from_dict(dict(base, data=dict(base["data"], events={"hese": True, "dr2": True})))
    assert cfg.identity() != other.identity()
    with pytest.raises(ValueError, match="exclude each other"):
        from_dict(dict(base, data=dict(base["data"], spectrum="ic26_astrophysical")))
    with pytest.raises(ValueError, match="extragalactic_norm"):
        from_dict(dict(base, solver={"grid": base["solver"]["grid"]}))
    with pytest.raises(ValueError, match="unknown keys"):
        from_dict(dict(base, data=dict(base["data"], events={"dr2": {"configs": ["IC79"], "foo": 1}})))


def test_gauss_newton_profile_equals_lbfgsb(tmp_path):
    """The HESE + DR2 nuisances' profile by projected Gauss-Newton steps (2026-10-15) reaches the L-BFGS-B profile's
    minimum (or below it) at points near and far from the data's best fit; its value and gradient are
    _expected_bins'."""
    m, ev = _event_model(tmp_path)
    rng = np.random.default_rng(11)
    x = ev.centre + 0.3 * ev.sigma * rng.standard_normal(len(ev.names))
    for i, (lo, hi) in enumerate(ev.bounds):
        x[i] = min(max(x[i], -np.inf if lo is None else lo + 1e-6), np.inf if hi is None else hi - 1e-6)
    A = ev.astro({"norm_astro": 1.0, "p_index": 2.5})
    f_ref, g_ref = ev._expected_bins(x, A, grad=True)
    f_gn, g_gn, H = ev._gn_terms(x, A)
    assert abs(f_gn - f_ref) < 1e-10 * max(1.0, abs(f_ref))
    assert np.max(np.abs(g_gn - g_ref)) < 1e-9 * max(1.0, np.max(np.abs(g_ref)))
    assert np.all(np.linalg.eigvalsh(H) > 0.0)
    for v in ({"norm_astro": 1.0, "p_index": 2.5}, {"norm_astro": 1.6, "p_index": 2.3},
              {"norm_astro": 0.3, "p_index": 2.75}, {"norm_astro": 20.0, "p_index": 2.25}):
        A = ev.astro(v)
        z0 = ev._z0.copy()
        x_gn, r_gn = ev.profile(A)
        ev._z0 = z0
        x_lb, r_lb = ev.profile_lbfgsb(A)
        assert r_gn.fun <= r_lb.fun + 1e-9 * max(1.0, r_lb.fun), (v, r_gn.fun, r_lb.fun)
        assert abs(r_gn.fun - r_lb.fun) < 1e-6 * max(1.0, r_lb.fun), (v, r_gn.fun, r_lb.fun)
        assert abs(ev.exact(x_gn, A)["total"] - ev.exact(x_lb, A)["total"]) < 1e-4 * max(1.0, r_lb.fun)
    assert ev.n_fallback == 0, ev.gn_fail


def test_empty_settings_mean_the_defaults(tmp_path, monkeypatch):
    """data.events {"hese": {}, "dr2": {}} -- what the campaign's configurations write -- turns both data sets on
    with their defaults (until 2026-10-15 an empty {} read as off, and those planes had no data: chi^2 = 0)."""
    from analysis.events import EventLikelihood, HESE_MC_ENV, HESE_DET_ENV
    m, ev_full = _event_model(tmp_path)
    monkeypatch.setenv(HESE_MC_ENV, str(tmp_path / "mc.npz"))
    monkeypatch.setenv(HESE_DET_ENV, str(tmp_path / "mc.npz") + ".det.npz")
    ev = EventLikelihood(m, {"hese": {}, "dr2": {}})
    assert ev.hese is not None and ev.dr2 is not None
    assert any(n.startswith("dr2_norm_") for n in ev.names) and "hese_conv_norm" in ev.names
    assert ev.chi2({"norm_astro": 1.0, "p_index": 2.5}) > 1.0                    # the data are there
    assert EventLikelihood(m, {"hese": {}, "dr2": False}).dr2 is None
    assert EventLikelihood(m, {"hese": True}).hese is not None
    with pytest.raises(ValueError, match="neither HESE nor DR2"):
        EventLikelihood(m, {"hese": False, "dr2": None})
