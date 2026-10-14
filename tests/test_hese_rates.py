"""
Tests of IceCubeRates.starting_events.hese (Mauricio's task 1) on a synthetic reduced Monte Carlo: the release's
binning, the per-event weighting, the atmospheric templates with the self-veto, and the per-direction flux lookup.
The real check (the release's best fit reproduced: 50.16 astrophysical events, 39.51 / 8.43 / 2.21) runs on Tycho's
reduction (tools/hese_reduce.py) and is recorded in docs/dev/HESE_RATES_PLAN.md.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from IceCubeRates import flux_adapters as FA  # noqa: E402
from IceCubeRates.starting_events import hese  # noqa: E402
import hese_reduce  # noqa: E402


def _fake_mc(path, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    types = rng.choice([12, -12, 14, -14, 16, -16, 13], n)
    E = 10 ** rng.uniform(4.5, 7.0, n)
    morph = rng.integers(0, 3, n)
    eb, zb, lb = hese_reduce.release_bins()
    d = dict(mc_type=types.astype(np.int16), mc_E_GeV=E, mc_cos_zenith=rng.uniform(-1, 1, n),
             mc_interaction=rng.integers(1, 4, n).astype(np.int8), mc_weight=rng.uniform(1e5, 1e6, n),
             mc_E_dep_GeV=E * rng.uniform(0.5, 1.0, n), mc_zenith_reco=rng.uniform(0, np.pi, n),
             mc_morphology=morph.astype(np.int8), mc_length_m=10 ** rng.uniform(1, 3, n),
             mc_pion_flux=rng.uniform(1e-20, 1e-18, n), mc_kaon_flux=rng.uniform(1e-20, 1e-18, n),
             mc_prompt_flux=rng.uniform(1e-21, 1e-19, n), mc_conv_veto=rng.uniform(0.1, 1, n),
             mc_prompt_veto=rng.uniform(0.1, 1, n), mc_muon_weight=np.where(types == 13, 1e-9, 0.0),
             data_E_dep_GeV=np.array([1e5, 2e5, 3e6, 7e4]), data_zenith_reco=np.array([0.3, 2.0, 1.0, 1.5]),
             data_morphology=np.array([0, 1, 2, 0], dtype=np.int8), data_length_m=np.array([0.0, 0.0, 50.0, 0.0]),
             livetime_s=2.0e8, energy_bins_GeV=eb, zenith_bins_rad=zb, length_bins_m=lb)
    np.savez(path, **d)
    return d


def test_release_bins():
    eb, zb, lb = hese_reduce.release_bins()
    assert len(eb) == 22 and eb[0] == pytest.approx(6e4) and np.allclose(np.diff(np.log10(eb)), 0.111)
    assert len(zb) == 11 and zb[0] == 0.0 and zb[-1] == pytest.approx(np.pi)
    np.testing.assert_allclose(np.cos(zb), np.linspace(1, -1, 11), atol=1e-15)
    assert len(lb) == 21 and lb[0] == pytest.approx(10.0) and lb[-1] == pytest.approx(1000.0)


def test_weighting_and_histograms(tmp_path):
    d = _fake_mc(tmp_path / "mc.npz")
    ev = hese.HESEEvents(str(tmp_path / "mc.npz"))
    per = ev.expected(hese.power_law_flux(1e-18, 2.5))
    nu = np.isin(np.abs(d["mc_type"]), (12, 14, 16))
    np.testing.assert_allclose(per[nu], 2e8 * d["mc_weight"][nu] * 1e-18 * (d["mc_E_GeV"][nu] / 1e5) ** -2.5,
                               rtol=1e-14)
    assert np.all(per[~nu] == 0.0)
    h = ev.histogram(per)
    assert h["cascades"].shape == (10, 21) and h["double cascades"].shape == (20, 21)
    inside = (d["mc_E_dep_GeV"] >= 6e4) & (d["mc_E_dep_GeV"] < ev.E_bins[-1])
    assert sum(x.sum() for x in h.values()) == pytest.approx(per[inside].sum(), rel=1e-12)
    dat = ev.data()
    assert dat["cascades"].sum() == 2 and dat["tracks"].sum() == 1 and dat["double cascades"].sum() == 1
    assert ev.histogram(ev.muons(2.0))["tracks"].sum() + ev.histogram(ev.muons(2.0))["cascades"].sum() > 0


def test_atmospheric_templates_apply_the_veto(tmp_path):
    d = _fake_mc(tmp_path / "mc.npz")
    ev = hese.HESEEvents(str(tmp_path / "mc.npz"))
    from test_icecube_rates import _synthetic_atm
    _synthetic_atm(str(tmp_path / "atm.npz"))
    t = ev.atmospheric_templates(str(tmp_path / "atm.npz"))
    nu = np.isin(np.abs(d["mc_type"]), (12, 14, 16)) & (d["mc_E_GeV"] <= 1e8)
    # synthetic table: nominal exp(0.4 |cos|) E^-3.7, prompt 1%, d(barr_h_pi+) = 0.2 x nominal x log10(E / 1 TeV)
    phi = np.exp(0.4 * np.abs(d["mc_cos_zenith"])) * d["mc_E_GeV"] ** -3.7
    base = 2e8 * d["mc_weight"] * phi
    np.testing.assert_allclose(t["conv"][nu], (base * 0.99 * d["mc_conv_veto"])[nu], rtol=1e-9)
    np.testing.assert_allclose(t["prompt"][nu], (base * 0.01 * d["mc_prompt_veto"])[nu], rtol=1e-9)
    np.testing.assert_allclose(t["barr_h_pi+"][nu], (base * 0.2 * np.log10(d["mc_E_GeV"] / 1e3) * d["mc_conv_veto"])[nu],
                               rtol=1e-9, atol=1e-12 * base.max())
    assert np.all(t["barr_g_pi+"] == 0.0) and np.all(t["conv"][~nu] == 0.0)


def test_sky_flux_lookup():
    E = np.geomspace(1e13, 1e17, 9)
    edges = np.radians(np.array([-90.0, 0.0, 90.0]))
    B = np.ones((2, len(E), 3)) * (E / 1e9)[None, :, None] ** -2.0
    B[1] *= 2.0
    sky = dict(E_eV=E, dec_band_edges=edges, bands_nu=B, bands_nubar=3 * B, meta=dict(extragalactic_scale=1.0))
    f = hese.sky_flux(sky)
    fl, anti = np.array([0, 1, 2, 1]), np.array([False, False, True, True])
    Eq, dec = np.array([1e5, 1e5, 2e6, 1e9]), np.radians(np.array([-30.0, 30.0, 10.0, 10.0]))
    np.testing.assert_allclose(f(fl, anti, Eq, dec), [1e-10, 2e-10, 3 * 2 * 2e6 ** -2.0, 0.0], rtol=1e-12)
    with pytest.raises(ValueError):
        hese.sky_flux(dict(sky, meta={}))


def test_detector_factor_interpolates_the_exported_grid(tmp_path):
    d = _fake_mc(tmp_path / "mc.npz")
    ev = hese.HESEEvents(str(tmp_path / "mc.npz"))
    n = len(d["mc_E_GeV"])
    rng = np.random.default_rng(7)
    a, b, c = rng.normal(0, 1, n), rng.normal(0, 0.5, n), rng.normal(0, 0.2, n)
    grids = {"dom_eff": np.array([0.80, 0.85, 0.90, 0.95, 0.99, 1.04, 1.09, 1.14, 1.19]),
             "head_on": np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]),
             "anisotropy": np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])}
    nominal = {"dom_eff": 0.99, "head_on": 0.0, "anisotropy": 1.0}
    det = dict(E_dep_GeV=d["mc_E_dep_GeV"], names=np.array(list(grids)), nominal=np.array(list(nominal.values())),
               sigma=np.array([0.1, 0.5, 0.2]))

    def cubic(x, x0):                                          # log10 w, zero at the nominal value
        t = x - x0
        return a * t + b * t ** 2 + c * t ** 3
    for name, g in grids.items():
        det["grid_" + name] = g
        for comp in ("Astro", "Conv", "Prompt"):
            det[f"log10w_{name}_{comp}"] = np.array([cubic(x, nominal[name]) for x in g])
    np.savez(tmp_path / "det.npz", **det)
    ev.load_detector(str(tmp_path / "det.npz"))
    np.testing.assert_allclose(ev.detector_factor("Conv", nominal), 1.0, rtol=1e-12)
    v = {"dom_eff": 1.02, "head_on": -0.33, "anisotropy": 1.15}
    expect = 10 ** sum(cubic(v[k], nominal[k]) for k in v)
    np.testing.assert_allclose(ev.detector_factor("Astro", v), expect, rtol=1e-10)
    with pytest.raises(ValueError):
        ev.detector_factor("Astro", {"dom_eff": 1.3})
    bad = dict(det, E_dep_GeV=det["E_dep_GeV"][:-1])
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError):
        ev.load_detector(str(tmp_path / "bad.npz"))
