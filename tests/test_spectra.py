"""
Tests of the task-2 redesign (Mauricio, 2026-10-14): the measured astrophysical spectrum as a data term
(analysis/spectra.py), the spectral index floated by interpolation between solver runs (analysis.model p_nodes),
and the configurations that use them (tools/make_galactic_configs.py).

Run: python -m pytest tests/test_spectra.py -q
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import config as config_mod  # noqa: E402
from analysis import spectra  # noqa: E402
from analysis.model import FlavorModel  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs")


def test_segments_exact_for_power_laws():
    """C = int Phi dE / int E^-2 dE: A for A E^-2; the closed form for E^-2.5; any node spacing."""
    E = np.geomspace(5e3, 3e6, 37)
    lo, hi = np.array([1e4, 2.15e4, 1e5]), np.array([2.15e4, 4.64e4, 2.15e5])
    np.testing.assert_allclose(spectra.segments(E, 3e-8 * E ** -2.0, lo, hi), 3e-8, rtol=1e-12)
    C = spectra.segments(E, 1e-6 * E ** -2.5, lo, hi)
    exact = 1e-6 * (lo ** -1.5 - hi ** -1.5) / 1.5 / (1 / lo - 1 / hi)
    np.testing.assert_allclose(C, exact, rtol=1e-12)
    # a segment reaching 1% past the last node is extrapolated log-log (exact for a power law)
    np.testing.assert_allclose(spectra.segments(E, 1e-6 * E ** -2.5, np.array([2.9e6]), np.array([3.03e6])),
                               1e-6 * (2.9e6 ** -1.5 - 3.03e6 ** -1.5) / 1.5 / (1 / 2.9e6 - 1 / 3.03e6), rtol=1e-10)


def test_spectrum_chi2_is_asymmetric_and_loads_inside_the_grid():
    d = spectra.load("ic26_astrophysical", 1e4, 3e7)
    assert len(d["value"]) == 6 and d["E_lo"][0] == pytest.approx(9999.6) and d["E_hi"][-1] == pytest.approx(999961.4)
    assert spectra.chi2(d["value"], d) == 0.0
    up, down = d["value"] + (d["hi"] - d["value"]), d["value"] - (d["value"] - d["lo"])
    assert spectra.chi2(up, d) == pytest.approx(6.0) and spectra.chi2(down, d) == pytest.approx(6.0)
    assert len(spectra.load("ic26_astrophysical")["value"]) == 7
    with pytest.raises(ValueError):
        spectra.load("ic26_astrophysical", 2e6, 3e7)
    with pytest.raises(ValueError):
        spectra.load("nope")


def _gal_raw(name="pion_2050_case_5_si_e_gal", n_E=60):
    raw = json.load(open(os.path.join(CONFIGS, name + ".json")))
    raw["solver"]["grid"].update(n_E_bins=n_E)
    return raw


def test_configs_wire_the_spectrum_and_the_index_nodes():
    cfg = config_mod.from_dict(_gal_raw(), "")
    assert cfg.spectrum == "ic26_astrophysical"
    assert cfg.p_nodes() == pytest.approx(list(np.linspace(2.0, 3.0, 7)))
    assert "f_e_S" not in cfg.free_names and "f_e_S_gal" not in cfg.free_names
    assert {p.name for p in cfg.priors} == {"norm_gal"}
    x = config_mod.from_dict(_gal_raw("pion_2050_case_5_si_e_galx"), "")
    assert {"f_e_S", "f_e_S_gal"} <= set(x.free_names) and x.identity() != cfg.identity()
    bad = _gal_raw()
    bad["solver"]["galactic"]["enabled"] = False
    with pytest.raises(ValueError, match="Galactic component"):
        config_mod.from_dict(bad, "")
    bad = _gal_raw()
    bad["data"]["spectrum"] = "nope"
    with pytest.raises(ValueError):
        config_mod.from_dict(bad, "")


def test_index_interpolation_exact_at_nodes_and_without_new_physics():
    raw = _gal_raw()
    cfg = config_mod.from_dict(raw, "")
    mi = cfg.model()
    md = FlavorModel(cfg.solver, cfg.free_names)
    for p in (2.0, 2.5, 3.0):                                    # nodes: the solver's own runs
        v = dict(m_phi_eV=5.6e6, g_phi=0.18, p_index=p)
        np.testing.assert_allclose(mi.flux(v)[1], md.flux(v)[1], rtol=1e-12)
    for p in (2.07, 2.41, 2.93):                                 # no new physics: E^-p, exact in log
        v = dict(m_phi_eV=5.6e6, g_phi=1e-9, p_index=p)
        np.testing.assert_allclose(mi.flux(v)[1], md.flux(v)[1], rtol=1e-6)
        np.testing.assert_allclose(mi.fractions(v), md.fractions(v), atol=1e-12)
    with pytest.raises(ValueError):
        mi.flux(dict(m_phi_eV=5.6e6, g_phi=0.18, p_index=3.2))


def test_objective_adds_the_spectrum_and_extragalactic_excludes_the_galactic():
    cfg = config_mod.from_dict(_gal_raw(), "")
    obj = cfg.objective()
    v = dict(m_phi_eV=5.6e6, g_phi=1e-9, p_index=2.5, norm_astro=1.3, norm_gal=0.9)
    E, F_eg = obj.model.extragalactic(v)
    _, F_all = obj.model.flux(v)
    _, F_off = obj.model.flux(dict(v, norm_gal=0.0))
    np.testing.assert_allclose(F_eg, F_off, rtol=1e-14)
    assert np.all(F_all > F_eg)
    C = spectra.segments(E / 1e9, F_eg.sum(-1) / 3.0, obj.spectrum["E_lo"], obj.spectrum["E_hi"])
    assert obj.spectrum_chi2(v) == pytest.approx(spectra.chi2(C, obj.spectrum), rel=1e-14)
    # without new physics, norm_astro = 1 and p = 2.5 put E^2 Phi at the anchor's energy (the segment's geometric
    # centre) at the anchor's value; the segment's E^-2 normalization of that E^-2.5 flux is 3.07% above it (the
    # production grid of 400 bins: the redshift grid's offset is 4e-3 there)
    prod = config_mod.from_dict(_gal_raw(n_E=400), "").objective()
    C1 = prod.spectrum_segments(dict(v, norm_astro=1.0))
    Ec, lo, hi = 10 ** (4 + 1 / 6), prod.spectrum["E_lo"][0], prod.spectrum["E_hi"][0]
    exact = 3.623e-8 * Ec ** 0.5 * (lo ** -1.5 - hi ** -1.5) / 1.5 / (1 / lo - 1 / hi)
    assert C1[0] == pytest.approx(exact, rel=5e-3)
    f = obj.model.fractions(v)
    from analysis import surfaces
    assert obj.chi2(v) == pytest.approx(surfaces.chi2(f[0], f[1], obj.data) + obj.spectrum_chi2(v)
                                        + obj.prior_chi2(v), rel=1e-12)
