"""
The future detectors (docs/dev/FUTURE_DETECTORS_PLAN.md): NuFlavorSingleSource's detector models, vendored unchanged
(IceCubeRates/future/nfss_forecast.py, nfss_detectors.py), and the diffuse response built on them
(IceCubeRates/future/diffuse.py).
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from IceCubeRates.future import nfss_detectors as D, nfss_forecast as F  # noqa: E402
from IceCubeRates.future.diffuse import DiffuseDetector  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))


@pytest.fixture(scope="module")
def calibrated():
    return D.calibrate(verbose=False)


def test_calibration_reproduces_nuflavorsinglesource(calibrated):
    """Every calibration number equals NuFlavorSingleSource's own run of its commit 39088bd (final inputs): bit for bit
    on the machine that made the reference, to 1e-12 elsewhere (CI's libraries move the last digit: 2026-10-15)."""
    ref = json.load(open(os.path.join(REPO, "tests", "data", "nfss_calibration_39088bd.json")))
    _, info = calibrated
    got = {k: float(info[k]) for k in ref}
    assert set(got) == set(ref)
    bad = {k: (got[k], ref[k]) for k in ref if not np.isclose(got[k], ref[k], rtol=1e-12, atol=0.0)}
    assert not bad, bad


@pytest.mark.parametrize("name", ["IceCube", "ARCA_base", "GVD_base", "Gen2_new_base"])
def test_diffuse_response_matches_allsky_rates(calibrated, name):
    """Summed over directions, the per-direction response gives NuFlavorSingleSource's all-sky rate in every
    reconstructed-energy bin (its own allsky_rate with the bin's cuts, for HESE's diffuse flux), to its quadrature."""
    out, _ = calibrated
    P = out[name]
    dd = DiffuseDetector(name, P)
    det = F.Detector(name, P, build=False)
    for ch in dd.channels:
        ed = F.EDGES[ch]
        acc = P["trk_accept"] if ch == "trk" else P["shw_accept"]
        mine = np.einsum("ebz->b", sum(0.5 * F.astro_hese(F.E)[:, None, None] * dd.R[ch][k] for k in dd.R[ch])) / P["years"]
        ref = np.array([det.allsky_rate(ch, accept=acc, flux=F.astro_hese, ecut=10 ** lo, smear=True)
                        - det.allsky_rate(ch, accept=acc, flux=F.astro_hese, ecut=10 ** hi, smear=True)
                        for lo, hi in zip(ed[:-1], ed[1:])])
        ok = ref > 1e-6
        np.testing.assert_allclose(mine[ok], ref[ok], rtol=5e-5)


@pytest.mark.parametrize("case", ["dark portal", "secret interactions"])
def test_site_galactic_bands(case):
    """The Galactic flux per local cos-zenith band at a site (propagate_sky_basis(site_latitudes_deg)): at the South
    Pole the per-declination bands in cos-zenith order; at every latitude, sky averages and day averages agree --
    each direction spends the whole day somewhere; a mid-latitude site and the North Pole differ from the South
    Pole's."""
    import dark_portal_solver as dps
    from test_event_likelihood import DP, SI, _sky_cfg
    over = {"secret_interactions": SI} if case == "secret interactions" else {"dark_portal": DP}
    E, params = dps.build_solver_params(_sky_cfg(0.3, 0.7, "3d", **over))
    params["show_progress"] = False
    out = dps.propagate_sky_basis(E, params, site_latitudes_deg=(-90.0, 36.27, 90.0))
    edges = out["site_cosz_edges"]
    for kind in ("nu", "nubar"):
        pole = out["site_galactic_bands_" + kind][-90.0]
        np.testing.assert_allclose(pole, out["galactic_bands_" + kind][:, ::-1], rtol=1e-12, atol=0)
        north = out["site_galactic_bands_" + kind][90.0]           # sees the northern sky above, the Galactic Centre below
        assert np.max(np.abs(north / pole - 1)) > 1e-3
        arca = out["site_galactic_bands_" + kind][36.27]
        assert np.max(np.abs(arca / pole - 1)) > 1e-3
    W = dps.site_zenith_weights(np.radians(np.array([-60.0, 0.0, 45.0])), 36.27, edges)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    # ARCA (36.27 N): dec = -60 never rises above cos Z = sin(36.27 - 90 + 60) ... it stays below the horizon
    assert W[0, edges[:-1] >= 0.0].sum() == 0.0


def test_future_event_likelihood_end_to_end():
    """A projection with the future detectors' events alone (data.events.future, All8_base): the model provides each
    site's per-zenith flux, the Asimov data close (chi^2 = 0 at the truth), the profile's gradient is the finite
    difference's, and a different source composition costs chi^2."""
    from analysis import config as config_mod
    from analysis.future_events import config_parts, part_lat
    from test_event_likelihood import _sky_cfg
    std = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 1e-60, "epsilon_prime": 0.0}   # no new physics
    cfg = _sky_cfg(1.0 / 3.0, 1.0 / 3.0, "3d", dark_portal=std, grid={"E_min_eV": 1e11, "E_max_eV": 3e16, "n_E_bins": 24})
    raw = {"name": "future_test", "data": {"events": {"future": {"config": "All8_base"}}, "flavor_band_eV": [1e13, 3e16]},
           "solver": cfg, "parameters": {"f_e_S": {"start": 0.333, "bounds": [0.0, 1.0], "space": "linear"},
                                         "norm_astro": {"start": 1.0, "bounds": [0.01, 100.0], "space": "log10"}},
           "fit": {"n_random": 0}, "profiles": {}, "planes": [], "asimov": {"f_e_S": 1.0 / 3.0}}
    c = config_mod.from_dict(raw)
    m = c.model()
    assert set(m.site_lats) == {36.27, 51.77, 47.7, 17.4}
    out, _ = D.calibrate(verbose=False)
    assert all(abs(out[p]["lat"] - part_lat(p)) < 1e-12 for p in config_parts("All8_base"))
    obj = c.objective(m, data_centre=(1.0 / 3.0, 2.0 / 3.0))
    ev = obj.events
    v = {"f_e_S": 1.0 / 3.0, "norm_astro": 1.0}
    assert abs(ev.chi2(v)) < 1e-6
    rng = np.random.default_rng(5)
    S = ev.signals(v)
    x = ev.centre + ev.sigma * rng.normal(scale=0.3, size=len(ev.centre))
    f0, g = ev._dev(x, S, ev.data)
    for i in rng.choice(len(x), 8, replace=False):
        h = 1e-4 * ev.sigma[i]                          # central differences: rounding ~1e-16 f / h
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (ev._dev(xp, S, ev.data)[0] - ev._dev(xm, S, ev.data)[0]) / (2 * h)
        assert abs(fd - g[i]) < 1e-5 * max(1.0, abs(g[i])), (ev.names[i], fd, g[i])
    far = ev.chi2({"f_e_S": 0.9, "norm_astro": 1.0})
    assert far > 25.0, far                                        # 10 yr of eight detectors see the composition


def test_newton_profile_equals_lbfgsb():
    """The nuisances' profile by Newton's method (the expectation is linear in them: exact Hessian) gives the
    L-BFGS-B profile's chi^2, at the Asimov point and away from it; its Hessian is the gradient's finite difference."""
    from analysis import config as config_mod
    from test_event_likelihood import _sky_cfg
    std = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 1e-60, "epsilon_prime": 0.0}
    cfg = _sky_cfg(1.0 / 3.0, 1.0 / 3.0, "3d", dark_portal=std, grid={"E_min_eV": 1e11, "E_max_eV": 3e16, "n_E_bins": 24})
    raw = {"name": "future_newton", "data": {"events": {"future": {"config": "All8_base"}}, "flavor_band_eV": [1e13, 3e16]},
           "solver": cfg, "parameters": {"f_e_S": {"start": 0.333, "bounds": [0.0, 1.0], "space": "linear"},
                                         "norm_astro": {"start": 1.0, "bounds": [0.01, 100.0], "space": "log10"}},
           "fit": {"n_random": 0}, "profiles": {}, "planes": [], "asimov": {"f_e_S": 1.0 / 3.0}}
    c = config_mod.from_dict(raw)
    ev = c.objective(c.model(), data_centre=(1.0 / 3.0, 2.0 / 3.0)).events
    for v in ({"f_e_S": 1.0 / 3.0, "norm_astro": 1.0}, {"f_e_S": 0.5, "norm_astro": 1.3},
              {"f_e_S": 0.9, "norm_astro": 0.7}, {"f_e_S": 0.0, "norm_astro": 2.0}):
        S = ev.signals(v)
        z_start = ev._z0.copy()
        x_n, r_n = ev.profile(S, ev.data)
        ev._z0 = z_start
        x_l, r_l = ev.profile_lbfgsb(S, ev.data)
        assert r_n.fun <= r_l.fun + 1e-9 * max(1.0, r_l.fun), (v, r_n.fun, r_l.fun)
        assert abs(r_n.fun - r_l.fun) < 1e-6 * max(1.0, r_l.fun), (v, r_n.fun, r_l.fun)
    assert ev.n_fallback == 0 and ev.n_newton >= 4
    A = ev._design(S)
    rng = np.random.default_rng(3)
    x = x_n + 0.2 * ev.sigma * rng.normal(size=len(x_n))
    _, g, H = ev._fgh(x, A, S, ev.data)
    for i in rng.choice(len(x), 6, replace=False):
        h = 1e-5 * ev.sigma[i]
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        dg = (ev._fgh(xp, A, S, ev.data)[1] - ev._fgh(xm, A, S, ev.data)[1]) / (2 * h)
        assert np.max(np.abs(dg - H[:, i])) < 1e-5 * max(1.0, np.max(np.abs(H[:, i]))), ev.names[i]
    f_dev, g_dev = ev._dev(x, S, ev.data)
    assert abs(f_dev - ev._fgh(x, A, S, ev.data)[0]) < 1e-9 * max(1.0, f_dev)
    assert np.max(np.abs(g_dev - g)) < 1e-9 * max(1.0, np.max(np.abs(g)))


def test_vectorized_signals_equal_the_per_key_sums():
    """The signals from the stacked responses and the vectorized interpolation (2026-10-17) equal the sums over each
    (flavor, kind) of np.interp-ed fluxes times the per-key responses, to rounding."""
    from analysis import config as config_mod
    from analysis.future_events import part_lat
    from IceCubeRates.future.diffuse import KEYS
    from test_event_likelihood import _sky_cfg
    std = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 1e-60, "epsilon_prime": 0.0}
    cfg = _sky_cfg(1.0 / 3.0, 1.0 / 3.0, "3d", dark_portal=std, grid={"E_min_eV": 1e11, "E_max_eV": 3e16, "n_E_bins": 24})
    raw = {"name": "future_vec", "data": {"events": {"future": {"config": "All8_base"}}, "flavor_band_eV": [1e13, 3e16]},
           "solver": cfg, "parameters": {"f_e_S": {"start": 0.333, "bounds": [0.0, 1.0], "space": "linear"}},
           "fit": {"n_random": 0}, "profiles": {}, "planes": [], "asimov": {"f_e_S": 1.0 / 3.0}}
    c = config_mod.from_dict(raw)
    ev = c.objective(c.model(), data_centre=(1.0 / 3.0, 2.0 / 3.0)).events
    v = {"f_e_S": 0.4}
    S = ev.signals(v)
    fl_index = {"e": 0, "mu": 1, "tau": 2}
    ref_phi = {}
    for d in ev.dets:
        lat = part_lat(d.name)
        if lat not in ref_phi:
            _, F_nu, F_nb = ev.model.site_band_fluxes(v, lat)
            per = []
            for F in (F_nu, F_nb):
                L = np.log(np.maximum(F, 1e-300))
                per.append({i: (np.exp(np.array([np.interp(ev._lg, ev._lgE_model, L[z, :, i]) for z in range(L.shape[0])]))
                                * ev._inside[None, :]).T for i in range(3)})
            ref_phi[lat] = per
    for (d, ch), sk in zip(ev.blocks, S):
        pn, pb = ref_phi[part_lat(d.name)]
        for key, get in (("s", lambda k: d.R[ch][k]), ("escale", lambda k: d.RX[ch][k]["escale"]),
                         ("misid", lambda k: d.RX[ch][k]["misid"])):
            ref = sum(np.einsum("ez,ebz->bz", (pb if bar else pn)[fl_index[fl]], get((fl, bar))) for fl, bar in KEYS)
            ref = np.asarray(ref).reshape(-1)
            assert np.max(np.abs(sk[key] - ref)) <= 1e-12 * max(np.max(np.abs(ref)), 1e-300), (d.name, ch, key)
