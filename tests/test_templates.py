"""
Mixing templates (analysis/templates.py, 2026-10-18): the interpolation weights, the model with templates (the same
fluxes as without at theta0, the template runs' own fluxes at their points, close to the exact runs between them,
never the parameter passed to the solver outside its runs), the exact gradient in templated parameters and in the
mass-sum penalty, the runs in parallel, and the configuration.
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

from analysis import config as config_mod  # noqa: E402
from analysis.templates import MixingTemplate, from_spec  # noqa: E402


# ------------------------------------------------------------------------------------------- the weights
def test_lagrange_weights_reproduce_polynomials():
    t = from_spec("s12sq", {"points": [0.29, 0.33, 0.35]}, 0.31)
    nodes = t.nodes()
    f = lambda x: 2.0 - 3.0 * x + 5.0 * x ** 2 - 7.0 * x ** 3              # degree 3 through 4 nodes: exact
    df = lambda x: -3.0 + 10.0 * x - 21.0 * x ** 2
    for x in (0.28, 0.3, 0.31, 0.345, 0.36):
        w, dw = t.weights(x)
        assert abs(np.sum(w) - 1.0) < 1e-12 and abs(np.sum(dw)) < 1e-9
        assert abs(w @ f(nodes) - f(x)) < 1e-12 and abs(dw @ f(nodes) - df(x)) < 1e-9
    w, _ = t.weights(0.31)
    assert np.array_equal(w, np.eye(len(nodes))[0])
    lin = from_spec("dm21_sq", {"step": 4e-7}, 7.537e-5)
    w, dw = lin.weights(7.537e-5 + 1e-7)
    assert np.allclose(w, [0.75, 0.25]) and np.allclose(dw, [-1 / 4e-7, 1 / 4e-7])


def test_turn_weights_reproduce_trigonometric_polynomials():
    t = from_spec("d_cp_deg", {"turn": 5}, 212.0)
    nodes = t.nodes()
    assert np.allclose(np.sort(nodes % 360.0), np.sort((212.0 + 72.0 * np.arange(5)) % 360.0))
    r = np.radians
    f = lambda d: 0.3 + 0.2 * np.cos(r(d)) - 0.1 * np.sin(r(d)) + 0.05 * np.cos(2 * r(d)) + 0.02 * np.sin(2 * r(d))
    df = lambda d: (-0.2 * np.sin(r(d)) - 0.1 * np.cos(r(d)) - 0.1 * np.sin(2 * r(d)) + 0.04 * np.cos(2 * r(d))) * np.pi / 180
    for d in (0.0, 100.0, 212.0, 250.0, 359.0, 572.0):
        w, dw = t.weights(d)
        assert abs(w @ f(nodes) - f(d)) < 1e-12 and abs(dw @ f(nodes) - df(d)) < 1e-12
    w, _ = t.weights(212.0)
    assert np.array_equal(w, np.eye(5)[0])
    with pytest.raises(ValueError):
        from_spec("d_cp_deg", {"turn": 4}, 212.0)
    with pytest.raises(ValueError):
        from_spec("s12sq", {"turn": 5}, 0.3)
    with pytest.raises(ValueError):
        from_spec("s12sq", {"step": 0.01, "points": [0.3]}, 0.31)


# ------------------------------------------------------------------------------------------- the model
S12 = {"step": 0.01}
DCP = {"turn": 7}       # 3 runs per turn leave 3% (the flux has harmonics of delta_CP up to the third), 5 leave 2e-3


def _raw(templates=True, free_mixing=True):
    from test_gradient_fit import _raw as base
    raw = base()
    if free_mixing:
        raw["parameters"]["s12sq"] = {"start": 0.3035, "bounds": [0.27, 0.34]}
        raw["parameters"]["d_cp_deg"] = {"start": 132.0, "bounds": [0.0, 360.0]}
        if templates:
            raw["parameters"]["s12sq"]["template"] = S12
            raw["parameters"]["d_cp_deg"]["template"] = DCP
    return raw


@pytest.fixture(scope="module")
def models():
    os.environ.pop("NULRI_FLUX_CACHE", None)
    tc = config_mod.from_dict(_raw())
    ec = config_mod.from_dict(_raw(templates=False))
    return tc, tc.model(), ec.model()


POINT = dict(m_Zd_eV=1e-13, g_D_eps_trace=1e-40, norm_astro=1.1, norm_gal=0.95, p_index=2.5, p_index_low=1.5,
             f_e_S=0.36, f_e_S_gal=0.30)


def _close(a, b, rel):
    sc = np.max(np.abs(b), axis=-1, keepdims=True)
    return float(np.max(np.abs(a - b) / np.maximum(sc, 1e-300)))


def test_theta0_is_the_model_without_templates(models):
    """At theta0 (or with the templated parameters absent) the fluxes are bit for bit those of the same point
    without templates, and the solver never sees a templated parameter outside the template runs."""
    tc, mt, me = models
    t0 = {t.name: t.theta0 for t in tc.mixing_templates()}
    for v in (POINT, dict(POINT, **t0)):
        for lat in [-89.99] + list(mt.site_lats):
            a, b = mt.site_band_fluxes(v, lat), me.site_band_fluxes(POINT, lat)
            for x, y in zip(a, b):
                assert np.array_equal(x, y)
    assert mt._overrides(dict(POINT, s12sq=0.33, d_cp_deg=200.0)) == mt._overrides(POINT)


def test_at_a_template_point_the_flux_is_that_run(models):
    """At a template's point, at the central index node (2.5) and with the lower index at its own (1.5), the
    model's flux is the exact run's at that value to rounding: the extragalactic part and the Galactic one."""
    tc, mt, me = models
    th = {t.name: t for t in tc.mixing_templates()}
    for name, x in (("s12sq", th["s12sq"].points[0]), ("d_cp_deg", th["d_cp_deg"].points[1])):
        v = dict(POINT, **{name: x})
        for lat in [-89.99] + list(mt.site_lats):
            got = mt.site_band_fluxes(v, lat)
            ref = me.site_band_fluxes(v, lat)
            for a, b in zip(got[1:], ref[1:]):
                assert _close(a, b, 0) < 1e-6, (name, lat)                # the runs are single precision


def test_between_points_close_to_the_exact_runs(models):
    """Away from the points and the central node: the template's error (the parameter's effect assumed the same at
    every index, linear in s12 between its points, a trigonometric polynomial of degree 3 in delta_CP): 1e-4 of
    each bin's largest flux here (2026-10-18: 1e-5 from s12, nothing measurable from the index, 9e-5 from delta_CP;
    3e-2 with 3 runs per turn of delta_CP, 2e-3 with 5)."""
    tc, mt, me = models
    v = dict(POINT, p_index=2.8, p_index_low=1.1, s12sq=0.3035 + 0.006, d_cp_deg=190.0)
    worst = 0.0
    for lat in [-89.99] + list(mt.site_lats):
        got, ref = mt.site_band_fluxes(v, lat), me.site_band_fluxes(v, lat)
        worst = max(worst, _close(got[1], ref[1], 0), _close(got[2], ref[2], 0))
    print(f"template error: {worst:.3g}")
    assert worst < 5e-4


def test_jacobian_in_templated_parameters(models):
    """site_band_flux_jacobian's derivatives in s12sq and d_cp_deg: central differences of site_band_fluxes; the
    fluxes themselves bit for bit site_band_fluxes'."""
    tc, mt, me = models
    names = ["norm_astro", "p_index", "s12sq", "d_cp_deg"]
    v = dict(POINT, s12sq=0.31, d_cp_deg=150.0)
    for lat in [-89.99] + list(mt.site_lats)[:1]:
        _, F_nu, F_nb, dF = mt.site_band_flux_jacobian(v, lat, names)
        _, R_nu, R_nb = mt.site_band_fluxes(v, lat)
        assert np.array_equal(F_nu, R_nu) and np.array_equal(F_nb, R_nb)
        for n in ("s12sq", "d_cp_deg"):
            h = 1e-5 if n == "s12sq" else 1e-3
            _, p_nu, p_nb = mt.site_band_fluxes(dict(v, **{n: v[n] + h}), lat)
            _, q_nu, q_nb = mt.site_band_fluxes(dict(v, **{n: v[n] - h}), lat)
            for d, a, b in ((dF[n][0], p_nu, q_nu), (dF[n][1], p_nb, q_nb)):
                fd = (a - b) / (2 * h)
                assert np.max(np.abs(d - fd)) < 1e-6 * max(np.max(np.abs(fd)), 1e-300), (n, lat)


def test_exact_gradient_with_templates_and_priors():
    """The future detectors' chi^2 with templated mixing, Gaussian priors on it and the half-Gaussian on the mass sum
    (which depends on the splittings): the exact gradient is offered and equals central differences."""
    from analysis.objective import Objective
    raw = _raw()
    raw["parameters"]["dm31_sq"] = {"start": 2.511e-3, "bounds": [2.4e-3, 2.6e-3], "template": {"step": 2e-5}}
    raw["priors"] = raw["priors"] + [{"name": "s12sq", "type": "gaussian", "centre": 0.3088, "width": 0.0016},
                                     {"name": "sum_masses_eV", "type": "half_gaussian", "limit": 0.09, "cl": 0.90,
                                      "quantile": "half"}]
    c = config_mod.from_dict(raw)
    obj = c.objective(c.model(), data_centre=(1.0 / 3.0, 2.0 / 3.0))
    names = ["norm_astro", "p_index", "s12sq", "d_cp_deg", "dm31_sq"]
    assert obj.supports_gradient(names)
    v = dict(POINT, s12sq=0.31, d_cp_deg=150.0, dm31_sq=2.52e-3)
    f, g = obj.chi2_and_grad(v, names)
    assert abs(f - obj.chi2(v)) < 1e-9 * max(1.0, f)
    for j, n in enumerate(names):
        h = 1e-5 * max(1.0, abs(v[n])) if n != "dm31_sq" else 1e-7
        fd = (obj.chi2(dict(v, **{n: v[n] + h})) - obj.chi2(dict(v, **{n: v[n] - h}))) / (2 * h)
        assert abs(g[j] - fd) < 1e-4 * max(1.0, abs(fd)), (n, g[j], fd)


def test_template_runs_in_parallel(models):
    """prefetch hands the template runs to the executor too: the fluxes bit for bit those computed one by one."""
    tc, _, _ = models
    seq, par, other = tc.model(), tc.model(), tc.model()
    tasks = []

    def ex(t):
        tasks.extend(t)
        return [other._sky_compute(cfg, g) for _, cfg, g in t]
    par.executor = ex
    v = dict(POINT, s12sq=0.31, d_cp_deg=150.0)
    for lat in [-89.99]:
        for a, b in zip(par.site_band_fluxes(v, lat), seq.site_band_fluxes(v, lat)):
            assert np.array_equal(a, b)
    n_nodes = len(par.p_nodes) + len(par.p1_nodes)
    n_tmpl = sum(len(t.points) for t in tc.mixing_templates())
    assert len(tasks) == n_nodes + 2 * n_tmpl        # every node, and each piece's template runs


def test_configuration():
    """Templates only for the six mixing parameters, around the solver's value; in the identity."""
    c = config_mod.from_dict(_raw())
    (s12, dcp) = c.mixing_templates()
    assert s12.name == "s12sq" and abs(s12.theta0 - np.sin(np.radians(33.41)) ** 2) < 1e-15
    assert dcp.turn and dcp.theta0 == 132.0 and len(dcp.points) == 6
    assert c.identity() != config_mod.from_dict(_raw(templates=False)).identity()
    bad = _raw()
    bad["parameters"]["norm_astro"]["template"] = {"step": 0.1}
    with pytest.raises(ValueError, match="unknown keys"):
        config_mod.from_dict(bad)
