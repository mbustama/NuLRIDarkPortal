"""
The exact profile over the mixing and the lightest mass (analysis/exact_profile.py, 2026-10-18): a model centred at
some mixing is the exact model there (bit for bit, no template run), its surrogate's gradient in the mixing is the
exact chi^2's (finite differences of the exact model), and the profile from a start is certified and not above a
brute-force fit with the solver at every value (small grid, future detectors, Table II's Gaussians and the mass sum).
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from analysis import config as config_mod  # noqa: E402
from analysis.exact_profile import M_LIGHTEST_MIN, ExactProfiler, exact_config  # noqa: E402

TABLE2 = dict(s12sq=(0.3088, 0.0016416), s13sq=(0.02248, 0.00059), s23sq=(0.470, 0.00418), d_cp_deg=(212.0, 6.687),
              dm21_sq=(7.537e-5, 3.7685e-7), dm31_sq=(2.511e-3, 1.2555e-5))
BOUNDS = dict(s12sq=(0.29, 0.33), s13sq=(0.02, 0.025), s23sq=(0.44, 0.50), d_cp_deg=(150.0, 270.0),
              dm21_sq=(7.3e-5, 7.8e-5), dm31_sq=(2.47e-3, 2.55e-3))


def _raw():
    from test_gradient_fit import _raw as base
    raw = base()
    sm = raw["solver"].setdefault("standard_mixing", {})
    sm.update(th12_deg=float(np.degrees(np.arcsin(np.sqrt(0.3088)))), th13_deg=float(np.degrees(np.arcsin(np.sqrt(0.02248)))),
              th23_deg=float(np.degrees(np.arcsin(np.sqrt(0.470)))), d_cp_deg=212.0, dm21_sq=7.537e-5, dm31_sq=2.511e-3)
    for n, (c, w) in TABLE2.items():
        raw["parameters"][n] = {"start": c, "bounds": list(BOUNDS[n])}
        raw["priors"].append({"name": n, "type": "gaussian", "centre": c, "width": w})
    raw["priors"].append({"name": "sum_masses_eV", "type": "half_gaussian", "limit": 0.09, "cl": 0.95, "quantile": "half"})
    return exact_config(raw)


# a point where the potential moves the flavors (the plane's coupling product, balanced split)
POINT = dict(m_Zd_eV=1e-13, g_D_eps_trace=1e-30)


@pytest.fixture(scope="module")
def prof():
    os.environ.pop("NULRI_FLUX_CACHE", None)
    cfg = config_mod.from_dict(_raw())
    return ExactProfiler(cfg, data_centre=(1.0 / 3.0, 2.0 / 3.0))


def _start(prof):
    return {p.name: p.start for p in prof.cfg.parameters}


def test_slope_templates():
    """The exact profile's difference templates: log-linear with the forward, central or one-sided second-order
    slope -- exact for a quadratic's slope at theta0 (central, one-sided) or for a linear function (forward); the
    identity at theta0; the derivative of the weights constant."""
    from analysis.templates import MixingTemplate
    f = lambda x: 0.3 - 2.0 * (x - 0.31) + 7.0 * (x - 0.31) ** 2         # its slope at 0.31: -2
    for pts, sl, exact in (((0.32,), (1.0,), False), ((0.32, 0.30), (0.5, -0.5), True),
                           ((0.32, 0.33), (2.0, -0.5), True), ((0.30, 0.29), (2.0, -0.5), True)):
        t = MixingTemplate("s12sq", 0.31, pts, slope=sl)
        w0, dw0 = t.weights(0.31)
        assert np.array_equal(w0, np.eye(len(pts) + 1)[0])
        slope = dw0 @ f(t.nodes())
        if exact:
            assert abs(slope + 2.0) < 1e-9
        w, dw = t.weights(0.335)
        assert abs(np.sum(w) - 1.0) < 1e-12 and np.allclose(dw, dw0)
        assert abs(w @ f(t.nodes()) - (f(0.31) + slope * 0.025)) < 1e-12        # linear in x with that slope
    with pytest.raises(ValueError):
        MixingTemplate("s12sq", 0.31, (0.32, 0.30), slope=(1.0,))


def test_split_top_coupling_is_not_a_wall():
    """A fixed split's plane ends at P = x; Tycho's numpy computes logspace(-60, -20, 41)[-1] = 1.0000000000000001e-20,
    a rounding above x = 1e-20: until 2026-10-18 the model refused it (the 35 WALL points of the October campaign's
    x = 1e-20 planes). Within SPLIT_TOP_ROUNDING it is x (epsilon' = 1); clearly above it, still refused."""
    raw = json.load(open(os.path.join(REPO, "analysis", "configs", "campaign_mix", "proj_future",
                                      "dp_tau_pot_x1m20_pos.json")))
    for spec in raw["parameters"].values():
        spec.pop("template", None)
    m = config_mod.from_dict(raw).model(cache_size=1)
    ov = m._overrides({"m_Zd_eV": 1e-27, "g_D_eps_trace": 1.0000000000000001e-20})
    assert ov[("dark_portal", "epsilon_prime")] == 1.0
    with pytest.raises(ValueError):
        m._overrides({"m_Zd_eV": 1e-27, "g_D_eps_trace": 1.001e-20})


def test_config_frees_m_lightest():
    raw = _raw()
    assert raw["parameters"]["m_lightest_eV"]["bounds"][0] == M_LIGHTEST_MIN
    assert all("template" not in s for s in raw["parameters"].values())


def test_centred_model_is_the_exact_model(prof):
    """At its centre the centred model is the exact model, bit for bit, and makes no template run."""
    v = dict(_start(prof), **POINT, norm_astro=1.1, p_index=2.6, s12sq=0.31, d_cp_deg=200.0, m_lightest_eV=0.01)
    centre = {n: v[n] for n in prof.eta}
    m = prof.centre_model(centre)
    for lat in [-89.99] + list(m.site_lats):
        a, b = m.site_band_fluxes(v, lat), prof.exact_model.site_band_fluxes(v, lat)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
    n_nodes = len(m.p_nodes) + len(m.p1_nodes)
    assert m.n_solver_calls == n_nodes                # the index nodes of both pieces, no template run
    prof.use(m)
    try:
        a = prof.obj.chi2(v)
    finally:
        prof.use(prof.exact_model)
    # the same up to the event likelihood's own profile, which starts from the previous call's nuisances
    assert abs(a - prof.obj.chi2(v)) < 1e-9 * max(1.0, a)


def test_surrogate_gradient_is_the_exact_gradient(prof):
    """The surrogate's chi^2 gradient in each mixing parameter at its centre: the exact model's central difference
    (relative 3e-2, or 3e-3 absolute per prior width)."""
    v = dict(_start(prof), **POINT, norm_astro=1.05, s12sq=0.31, s23sq=0.475, m_lightest_eV=0.012)
    centre = {n: v[n] for n in prof.eta}
    m = prof.centre_model(centre)
    for n in prof.eta:
        w = prof.w[n]
        h = 0.02 * w
        prof.use(m)
        try:
            gs = (prof.obj.chi2(dict(v, **{n: v[n] + h})) - prof.obj.chi2(dict(v, **{n: v[n] - h}))) / (2 * h) * w
        finally:
            prof.use(prof.exact_model)
        ge = (prof.obj.chi2(dict(v, **{n: v[n] + h})) - prof.obj.chi2(dict(v, **{n: v[n] - h}))) / (2 * h) * w
        print(f"{n}: surrogate {gs:.5g}, exact {ge:.5g} (chi^2 per prior width)")
        assert abs(gs - ge) <= max(3e-2 * abs(ge), 3e-3), (n, gs, ge)


@pytest.mark.slow
def test_plane_with_the_exact_profile(tmp_path):
    """A small plane through analysis.scans with exact_profile set: the global fit at the smallest coupling, chains
    along the coupling, every point certified, the couplings below the reach of new physics inheriting the chain's
    anchor after one evaluation, the records carrying the certificate."""
    from analysis import scans
    raw = _raw()
    raw["exact_profile"] = {}
    raw["planes"] = [{"x": "m_Zd_eV", "y": "g_D_eps_trace", "x_range": [1e-13, 1e-12], "y_range": [1e-62, 1e-59],
                      "nx": 2, "ny": 4, "raster": "m_Zd_eV"}]
    cfg = config_mod.from_dict(raw)
    centre = (1.0 / 3.0, 2.0 / 3.0)
    best = scans.global_fit(cfg, centre, workers=1, checkpoint=str(tmp_path / "fit.jsonl"), log=print)
    s = cfg.planes[0]
    assert best["values"][s.y] == s.y_grid[0]
    r = scans.plane(cfg, s.x, s.y, s.x_grid, s.y_grid, best["values"], centre, workers=1,
                    checkpoint=str(tmp_path / "plane.jsonl"), max_passes=1, log=print)
    lines = [f"{p['index']} {p['chi2']:.6f} {p.get('solver_runs')} {p['message'][:120]}" for p in r.points]
    for p in r.points:
        assert p["certified"] and "solver_runs" in p, lines
    # the global fit (at 1e-62, no new physics) anchors each column; 1e-62..1e-59 lie below the reach of new physics
    # there (the Earth's potential at a range of ~2000 km: a phase ~1e-10 across the Earth) and inherit its
    # certificate after one evaluation each. (At 1e-51 the phase is already ~0.05: profiled.)
    assert all(p["message"].startswith("inherited") for p in r.points), lines


@pytest.mark.slow
def test_profile_certified_and_not_above_bruteforce(prof):
    """From the priors' centre: certified, and a fit with the solver at every value, started at the profile and at
    the priors' centre, finds nothing more than 1e-3 lower."""
    start = _start(prof)
    r = prof.profile(POINT, start, log=print)
    print(r.message, r.chi2, r.solver_runs, len(r.centres))
    assert r.certified
    # the reported chi^2 is the exact model's at the reported values
    assert abs(prof.obj.chi2(r.values) - r.chi2) < 1e-9 * max(1.0, r.chi2)
    bf = prof.bruteforce(POINT, [r.values, start])
    print("brute force", bf.chi2, bf.nfev)
    assert r.chi2 <= bf.chi2 + 1e-3
