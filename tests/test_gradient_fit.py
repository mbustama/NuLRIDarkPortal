"""
The exact gradient of the future detectors' chi^2 (2026-10-17; analysis/model.py site_band_flux_jacobian,
analysis/future_events.py chi2_and_grad, analysis/objective.py, analysis/fitting.py): the flux derivatives against
finite differences, the profiled chi^2's gradient against re-profiled finite differences (inside the bounds and with
nuisances at them), the fitter's internal coordinates, the fits against finite-difference fits, the end-of-fit check,
and that every other analysis keeps the finite differences.
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

from test_event_likelihood import _sky_cfg  # noqa: E402

NAMES = ["norm_astro", "norm_gal", "p_index", "p_index_low", "f_e_S", "f_e_S_gal"]
E_B_GEV = 10 ** 4.39


def _raw(free=tuple(NAMES), broken=True):
    std = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-13, "g_D": 1e-60, "epsilon_prime": 0.1,
           "coupling_split": "balanced"}
    src = {"E_break_GeV": E_B_GEV, "p_index_low": 1.31, "p_index": 2.735} if broken else {"p_index": 2.6}
    solver = _sky_cfg(1.0 / 3.0, 1.0 / 3.0, "3d", dark_portal=std, source=src,
                      grid={"E_min_eV": 1e11, "E_max_eV": 3e16, "n_E_bins": 24})
    params = {"m_Zd_eV": {"start": 1e-13, "bounds": [1e-31, 1e5], "space": "log10"},
              "g_D_eps_trace": {"start": 1e-40, "bounds": [1e-62, 1e-6], "space": "log10"},
              "norm_astro": {"start": 1.0, "bounds": [0.01, 1000.0], "space": "log10"},
              "norm_gal": {"start": 0.91, "bounds": [0.0, 3.0], "space": "linear"},
              "p_index": {"start": 2.735 if broken else 2.6, "bounds": [2.0, 3.0], "nodes": 3},
              "p_index_low": {"start": 1.31, "bounds": [0.5, 2.5], "nodes": 3},
              "f_e_S": {"start": 1.0 / 3.0, "bounds": [0.0, 1.0]},
              "f_e_S_gal": {"start": 1.0 / 3.0, "bounds": [0.0, 1.0]}}
    keep = set(free) | {"m_Zd_eV", "g_D_eps_trace"}
    return {"name": "gradient_test", "data": {"events": {"future": {"config": "All8_base"}},
                                              "flavor_band_eV": [1e13, 3e16]},
            "solver": solver, "parameters": {k: v for k, v in params.items() if k in keep},
            "priors": [{"name": "norm_gal", "type": "gaussian", "centre": 0.91, "width": 0.18}],
            "fit": {"n_random": 0, "max_evals": 400, "fd_step": 1e-3, "ftol": 1e-9}, "profiles": {}, "planes": [],
            "asimov": {"g_D_eps_trace": 1e-62, "f_e_S": 1.0 / 3.0, "f_e_S_gal": 1.0 / 3.0}, "flux_basis": "pure_flavors"}


@pytest.fixture(scope="module")
def setup():
    from analysis import config as config_mod
    c = config_mod.from_dict(_raw())
    obj = c.objective(c.model(), data_centre=(1.0 / 3.0, 2.0 / 3.0))
    return c, obj


INTERIOR = dict(m_Zd_eV=1e-13, g_D_eps_trace=1e-40, norm_astro=1.1, norm_gal=0.95, p_index=2.70, p_index_low=1.40,
                f_e_S=0.36, f_e_S_gal=0.30)


def test_site_jacobian_matches_finite_differences(setup):
    """The model's flux per site and its closed-form derivatives: the flux equals site_band_fluxes', each derivative
    the central difference of site_band_fluxes (a pole site and a mid-latitude one; the broken power law, both
    indices, fitted compositions)."""
    c, obj = setup
    m = obj.model
    for lat in (-89.99, 36.27):
        _, F_nu, F_nb, dF = m.site_band_flux_jacobian(INTERIOR, lat, NAMES)
        _, R_nu, R_nb = m.site_band_fluxes(INTERIOR, lat)
        assert np.array_equal(F_nu, R_nu) and np.array_equal(F_nb, R_nb)
        for n in NAMES:
            h = 1e-6 * max(1.0, abs(INTERIOR[n]))
            _, p_nu, p_nb = m.site_band_fluxes(dict(INTERIOR, **{n: INTERIOR[n] + h}), lat)
            _, q_nu, q_nb = m.site_band_fluxes(dict(INTERIOR, **{n: INTERIOR[n] - h}), lat)
            for d, a, b in ((dF[n][0], p_nu, q_nu), (dF[n][1], p_nb, q_nb)):
                fd = (a - b) / (2 * h)
                scale = max(np.max(np.abs(fd)), 1e-300)
                assert np.max(np.abs(d - fd)) < 1e-6 * scale, (lat, n, np.max(np.abs(d - fd)) / scale)


def test_single_power_law_jacobian():
    """Without a break: the index's derivative through the spline and K(p)."""
    from analysis import config as config_mod
    c = config_mod.from_dict(_raw(free=("norm_astro", "p_index", "f_e_S"), broken=False))
    m = c.model()
    v = dict(m_Zd_eV=1e-13, g_D_eps_trace=1e-40, norm_astro=1.2, p_index=2.45, f_e_S=0.4)
    _, F_nu, _, dF = m.site_band_flux_jacobian(v, 47.7, ["norm_astro", "p_index", "f_e_S"])
    for n in ("norm_astro", "p_index", "f_e_S"):
        h = 1e-6 * max(1.0, abs(v[n]))
        fd = (m.site_band_fluxes(dict(v, **{n: v[n] + h}), 47.7)[1]
              - m.site_band_fluxes(dict(v, **{n: v[n] - h}), 47.7)[1]) / (2 * h)
        assert np.max(np.abs(dF[n][0] - fd)) < 1e-6 * np.max(np.abs(fd)), n


def _at_bounds(ev, x):
    return int(np.sum((np.abs(x - ev._lo) <= 1e-9 * np.maximum(1.0, np.abs(ev._lo)))
                      | (np.abs(x - ev._hi) <= 1e-9 * np.maximum(1.0, np.abs(ev._hi)))))


def test_profiled_gradient_matches_refit_differences(setup):
    """The gradient of the profiled chi^2 (nuisances held at their optimum) is the re-profiled central difference,
    inside the bounds and with nuisances at them; the value is chi2's."""
    c, obj = setup
    ev = obj.events
    far = dict(INTERIOR, norm_astro=3.0, p_index=2.0 + 1e-3, p_index_low=0.5 + 1e-3, f_e_S=1e-3)
    for v, want_bounds in ((INTERIOR, False), (far, True)):
        val, g = ev.chi2_and_grad(v, NAMES)
        assert abs(val - ev.chi2(v)) < 1e-9 * max(1.0, val)
        x, _ = ev.profile(ev.signals(v), ev.data)
        if want_bounds:
            assert _at_bounds(ev, x) > 0
        for j, n in enumerate(NAMES):
            h = 1e-5 * max(1.0, abs(v[n]))
            h = min(h, 0.5 * (v[n] - {"p_index": 2.0, "p_index_low": 0.5, "f_e_S": 0.0}.get(n, -np.inf)))
            fd = (ev.chi2(dict(v, **{n: v[n] + h})) - ev.chi2(dict(v, **{n: v[n] - h}))) / (2 * h)
            assert abs(g[j] - fd) < 1e-4 * max(1.0, abs(fd)), (n, g[j], fd)


def test_gradient_in_the_fitters_coordinates(setup):
    """on_with_grad: the chain rule through log10 and linear maps, against central differences in the internal
    vector; the prior on norm_gal included."""
    c, obj = setup
    pset = c.parameter_set().holding({"m_Zd_eV": 1e-13, "g_D_eps_trace": 1e-40})
    assert obj.supports_gradient(pset.names)
    fg = obj.on_with_grad(pset)
    u = pset.to_internal(INTERIOR)
    f, g = fg(u)
    assert abs(f - obj.chi2(pset.as_values(u))) < 1e-9 * max(1.0, f)
    for i in range(len(u)):
        h = 1e-5
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        fd = (obj.chi2(pset.as_values(up)) - obj.chi2(pset.as_values(um))) / (2 * h)
        assert abs(g[i] - fd) < 1e-4 * max(1.0, abs(fd)), (pset.names[i], g[i], fd)


def test_gradient_only_where_it_is_exact(setup, tmp_path):
    """The gradient path is taken only for the future detectors' events alone with every free name in
    GRADIENT_NAMES: a free mass or coupling, a flavor surface, HESE and DR2 keep the finite differences."""
    from analysis import config as config_mod
    from analysis.objective import Objective
    c, obj = setup
    assert obj.supports_gradient(NAMES)
    assert not obj.supports_gradient(NAMES + ["m_Zd_eV"])
    assert not obj.supports_gradient(["g_D_eps_trace"])
    surf = Objective(obj.model, "pion_2050_case_5", obj.priors)
    assert not surf.supports_gradient(["norm_astro"])
    from test_event_likelihood import _event_model
    m_ev, ev = _event_model(tmp_path)
    assert not Objective(m_ev, None, (), events=ev).supports_gradient(["norm_astro", "p_index"])


def test_gradient_fit_matches_finite_difference_fit(setup):
    """At held (m, coupling): the fit with the exact gradient reaches the finite-difference fit's chi^2 or lower, with
    fewer evaluations, and its end-of-fit check passes."""
    from analysis.fitting import fit
    c, obj = setup
    for held in ({"m_Zd_eV": 1e-13, "g_D_eps_trace": 1e-40}, {"m_Zd_eV": 1e-20, "g_D_eps_trace": 1e-45}):
        pset = c.parameter_set().holding(held)
        start = dict(INTERIOR, norm_astro=1.4, p_index=2.6, f_e_S=0.45)
        r_g = fit(obj, pset, starts=[start], settings=c.fit)
        r_f = fit(obj, pset, starts=[start], settings=c.fit, gradient=False)
        assert "gradient" in r_g.message and "failed" not in r_g.message, r_g.message
        assert r_g.chi2 <= r_f.chi2 + 1e-6, (held, r_g.chi2, r_f.chi2)
        assert r_g.nfev < r_f.nfev, (r_g.nfev, r_f.nfev)


def test_gradient_check_catches_a_wrong_gradient(setup, monkeypatch):
    """A wrong gradient (here half the true one) lets L-BFGS-B stop away from the minimum; the end-of-fit check
    catches it and the finite-difference refit recovers the minimum."""
    from analysis.fitting import fit
    c, obj = setup
    pset = c.parameter_set().holding({"m_Zd_eV": 1e-13, "g_D_eps_trace": 1e-40})
    start = dict(INTERIOR, norm_astro=1.4, p_index=2.6, f_e_S=0.45)
    r_f = fit(obj, pset, starts=[start], settings=c.fit, gradient=False)
    true = obj.chi2_and_grad

    def wrong(values, names):
        f, g = true(values, names)
        g = np.array(g)
        g[names.index("p_index")] *= 0.5
        g[names.index("norm_astro")] = 0.0
        return f, g
    monkeypatch.setattr(obj, "chi2_and_grad", wrong)
    r = fit(obj, pset, starts=[start], settings=c.fit)
    assert "gradient check failed" in r.message, r.message
    assert r.chi2 <= r_f.chi2 + 1e-6
