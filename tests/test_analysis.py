"""
Tests of the analysis package (analysis/): data surfaces, parameters and priors, the model's
linear shortcuts, the objective, fitting, intervals, profile scans with their repair and
checkpoints, configurations, and the command line end to end.

Run: python -m pytest tests/test_analysis.py -q
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import config as config_mod  # noqa: E402
from analysis import intervals, scans, surfaces  # noqa: E402
from analysis.fitting import FitSettings, fit  # noqa: E402
from analysis.parameters import GaussianPrior, HalfGaussianPrior, Parameter, ParameterSet  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs")

# ln L of FrequentistAnalysis' detector_flavor_log_likelihood (retired 2026-10-12) at the
# points X, Y, recorded before its removal: the surfaces must keep reproducing it bit for bit.
X = np.array([0.30, 0.3264281, 0.20, 0.45, 0.33])
Y = np.array([0.36, 0.34311662, 0.45, 0.25, 0.30])
FA_LOG_L = {
    "pion_2050_case_5": [-3.023170394198111, 7.552893319562026, -379.3487346265384, -329.9506706530841,
                         -40.776510365636646],
    "uhe_pion_ic-gen2_flux_0": [-0.4734627673349497, 0.0, -2.4024956989211366, -1.677043562041514,
                                -0.15304671171364975],
    "hnl_benchmark_A_2050_case_5": [-24.404356153923924, -0.7443200974080932, -381.9526657139512,
                                    -139.86086732780606, -6.493761361774319],
}


def _tiny_raw(**over):
    """A configuration with a tiny solver grid (60 x 60, one path, two chords)."""
    raw = json.load(open(os.path.join(CONFIGS, "pion_2050_case_5.json")))
    raw["solver"]["grid"].update(n_E_bins=60, n_z_bins=60, n_mw_paths=1, n_earth_chords=2)
    raw["fit"].update(n_random=1, max_evals=120)
    raw.update(over)
    return raw


# ---------------------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(FA_LOG_L))
def test_surfaces_reproduce_frequentist_analysis(name):
    """Asymmetric-smooth, polar ray-casting and symmetric shapes, bit for bit."""
    got = surfaces.log_likelihood(X, Y, surfaces.surface(name))
    assert np.array_equal(got, np.array(FA_LOG_L[name]))


def test_chi2_is_zero_at_the_centre_and_recentring_moves_it():
    s = surfaces.surface("pion_2050_case_5")
    assert surfaces.chi2(s["mu_x"], s["mu_y"], "pion_2050_case_5") == pytest.approx(0.0, abs=1e-12)
    moved = surfaces.recentred("pion_2050_case_5", 0.31, 0.35)
    assert surfaces.chi2(0.31, 0.35, moved) == pytest.approx(0.0, abs=1e-12)
    assert surfaces.chi2(s["mu_x"], s["mu_y"], moved) > 1.0
    # the shape is moved rigidly: chi^2 depends only on the offset from the centre
    a = surfaces.chi2(s["mu_x"] + 0.01, s["mu_y"] - 0.002, "pion_2050_case_5")
    b = surfaces.chi2(0.31 + 0.01, 0.35 - 0.002, moved)
    assert a == pytest.approx(b, rel=1e-12)
    with pytest.raises(KeyError):
        surfaces.surface("no_such_surface")


def test_ported_configs_fix_the_mixing_of_the_projection_centre():
    """The projection pion_2050_case_5 is centred on pion-decay sources oscillated with
    NuFIT 6.0 (normal ordering, with SK): the configurations' fixed mixing must reproduce
    that centre, or standard physics would sit 0.7 sigma away from the data (as it did in
    FrequentistAnalysis, at delta_CP = 132 deg)."""
    import dark_portal_solver as dps
    for fn in ("pion_2050_case_5.json", "pion_2050_case_5_mZd_1e-30.json", "pion_2050_case_5_exclusion.json"):
        raw = json.load(open(os.path.join(CONFIGS, fn)))
        m = raw["solver"]["standard_mixing"]
        U = dps.construct_PMNS(*np.radians([m["th12_deg"], m["th13_deg"], m["th23_deg"], m["d_cp_deg"]]))
        P = np.abs(U) ** 2 @ (np.abs(U) ** 2).T
        src = raw["solver"]["source"]
        f = np.array([src["f_e_S"], src["f_mu_S"], src["f_tau_S"]]) @ P
        s = surfaces.surface(raw["data"]["surface"])
        assert abs(f[0] - s["mu_x"]) < 2e-5 and abs(f[1] - s["mu_y"]) < 2e-5, (fn, f)


# ---------------------------------------------------------------------------------------
# Parameters and priors
# ---------------------------------------------------------------------------------------
def test_parameter_transforms_round_trip_and_hold():
    p = Parameter("m_Zd_eV", 2e6, (1e-30, 1e10), "log10")
    q = Parameter("u_e", 0.5, (0.0, 1.0), "linear")
    for x in (1e-30, 3.7e-5, 2e6, 1e10):
        assert p.from_u(p.to_u(x)) == pytest.approx(x, rel=1e-13)
    assert q.step == pytest.approx(0.1)
    ps = ParameterSet([p, q])
    held = ps.holding({"u_e": 0.25})
    assert held.names == ["m_Zd_eV"] and held.held == {"u_e": 0.25}
    assert held.as_values(np.array([6.0])) == {"u_e": 0.25, "m_Zd_eV": pytest.approx(1e6)}
    with pytest.raises(ValueError):
        Parameter("g", 2.0, (0.0, 1.0))
    with pytest.raises(ValueError):
        Parameter("g", 0.5, (0.0, 1.0), "log10")
    with pytest.raises(ValueError):
        ps.holding({"g_D": 1.0})


def test_priors():
    g = GaussianPrior("s12sq", 0.308, 0.01)
    assert g.chi2(0.318) == pytest.approx(1.0)
    h = HalfGaussianPrior("sum_masses_eV", 0.055, 0.95)
    assert h.sigma == pytest.approx(0.055 / 1.6448536269514722)
    assert h.chi2(0.055) == pytest.approx(1.6448536269514722 ** 2)
    assert h.chi2(-0.1) == 0.0


# ---------------------------------------------------------------------------------------
# Model and objective
# ---------------------------------------------------------------------------------------
def test_free_source_composition_uses_the_linear_basis():
    """With the source composition free, the model combines pure-flavor runs; the result must
    equal a direct run at that composition (the solver is linear in it)."""
    raw = _tiny_raw()
    raw["parameters"] = {"f_e_S": {"start": 0.3, "bounds": [0.0, 1.0]},
                         "g_D": {"start": 1e-4, "bounds": [1e-10, 3.54], "space": "log10"}}
    raw["solver"]["dark_portal"].update(u_e=0.5, u_mu=0.5, m_Zd_eV=2e6)
    raw["profiles"], raw["planes"] = {}, []
    cfg = config_mod.from_dict(raw)
    model = cfg.model()
    v = {"f_e_S": 0.2, "g_D": 2e-4}                 # f_tau_S fixed at 0: f_mu_S = 0.8
    np.testing.assert_allclose(model.composition(v), [0.2, 0.8, 0.0])
    f_basis = model.fractions(v)
    direct = copy.deepcopy(raw)
    direct["solver"]["source"].update(f_e_S=0.2, f_mu_S=0.8, f_tau_S=0.0)
    direct["parameters"] = {"g_D": raw["parameters"]["g_D"]}
    f_direct = config_mod.from_dict(direct).model().fractions({"g_D": 2e-4})
    np.testing.assert_allclose(f_basis, f_direct, rtol=1e-12, atol=1e-14)
    n = model.n_solver_calls
    model.fractions({**v, "f_e_S": 0.4})                       # same propagation point
    assert model.n_solver_calls == n


def test_coupling_products_are_what_the_data_see():
    """Exclusion planes scan the coupling products (decided 2026-10-13). With epsilon' = 0
    the fractions depend on g_D and the mixing only through g_D |U|^2; the coherent
    potential depends on g_D epsilon' |U|^2 (up to the Debye mass, negligible here)."""
    raw = _tiny_raw()
    raw["profiles"], raw["planes"] = {}, []
    out = []
    for u, eps in ((0.0316, 0.0), (0.5, 0.0)):
        r = copy.deepcopy(raw)
        r["solver"]["dark_portal"].update(u_e=u, u_mu=u, u_tau=0.0, epsilon_prime=eps)
        r["parameters"] = {"g_D_u2": {"start": 1e-5, "bounds": [1e-12, 1e-1], "space": "log10"}}
        out.append(config_mod.from_dict(r).model().fractions({"g_D_u2": 3e-5, "m_Zd_eV": 2e6}))
    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=1e-12)
    out = []
    for u, eps in ((0.0316, 1e-4), (0.1, 1e-2)):
        r = copy.deepcopy(raw)
        r["solver"]["dark_portal"].update(u_e=u, u_mu=u, u_tau=0.0, epsilon_prime=eps)
        r["parameters"] = {"g_D_eps_u2": {"start": 1e-30, "bounds": [1e-62, 1e-8], "space": "log10"}}
        out.append(config_mod.from_dict(r).model().fractions({"g_D_eps_u2": 1e-29, "m_Zd_eV": 1e-2}))
    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        config_mod.from_dict(raw).model().fractions({"g_D_u2": 1e-5, "g_D": 1e-4, "m_Zd_eV": 2e6})


def test_objective_charges_mixing_beyond_unitarity_continuously():
    cfg = config_mod.from_dict(_tiny_raw())
    obj = cfg.objective()
    base = {"m_Zd_eV": 2e6, "g_D": 1e-4, "u_tau": 0.0}
    inside = obj.chi2({**base, "u_e": 0.7, "u_mu": np.sqrt(1 - 0.49) * (1 - 1e-9)})
    edge = obj.chi2({**base, "u_e": 0.7, "u_mu": np.sqrt(1 - 0.49) * (1 + 1e-9)})
    beyond = obj.chi2({**base, "u_e": 0.7, "u_mu": 0.8})
    assert np.isfinite(inside) and abs(edge - inside) < 1e-3
    assert beyond > inside + 1e4 * 0.1


# ---------------------------------------------------------------------------------------
# A synthetic objective for fitting and scans: chi^2 = (a - 1)^2 / 0.04 + 25 (b - a^2)^2,
# plus a side minimum at b = -1 (chi^2 floor 3) that a fit started there falls into.
# ---------------------------------------------------------------------------------------
class _Synthetic:
    def __init__(self):
        self.calls = 0

    def chi2(self, v):
        self.calls += 1
        a, b = v["a"], v["b"]
        main = (a - 1.0) ** 2 / 0.04 + 25.0 * (b - a * a) ** 2
        side = 3.0 + (a - 1.0) ** 2 / 0.04 + 25.0 * (b + 1.0) ** 2
        return float(min(main, side))

    def on(self, pset):
        return lambda u: self.chi2(pset.as_values(u))


def _synthetic_cfg():
    raw = _tiny_raw()
    raw["parameters"] = {"a": {"start": 0.5, "bounds": [-2.0, 3.0]}, "b": {"start": 0.0, "bounds": [-3.0, 9.0]}}
    raw["profiles"], raw["planes"] = {}, []
    cfg = config_mod.AnalysisConfig(name="synthetic", surface="pion_2050_case_5", solver={},
                                    parameters=[Parameter("a", 0.5, (-2.0, 3.0)), Parameter("b", 0.0, (-3.0, 9.0))],
                                    priors=[], fit=FitSettings(n_random=2, max_evals=400), raw=raw)
    return cfg


@pytest.fixture
def synthetic(monkeypatch):
    cfg = _synthetic_cfg()
    obj = _Synthetic()

    def init(raw, path, centre):
        scans._W["cfg"] = cfg
        scans._W["objective"] = obj
    monkeypatch.setattr(scans, "_init_worker", init)
    return cfg, obj


def test_fit_finds_the_global_minimum(synthetic):
    cfg, obj = synthetic
    res = scans.global_fit(cfg, workers=1, log=lambda *a: None)
    assert res["chi2"] < 1e-8
    assert res["values"]["a"] == pytest.approx(1.0, abs=1e-4)
    assert res["values"]["b"] == pytest.approx(1.0, abs=1e-4)
    r = fit(obj, cfg.parameter_set().holding({"a": 0.5}), settings=FitSettings())
    assert r.values["b"] == pytest.approx(0.25, abs=1e-4) and r.chi2 == pytest.approx(6.25, abs=1e-6)


def test_profile_matches_the_analytic_profile_and_resumes(synthetic, tmp_path):
    cfg, obj = synthetic
    grid = np.linspace(-1.0, 2.5, 15)
    ck = str(tmp_path / "profile_a.jsonl")
    res = scans.profile(cfg, "a", grid, {"a": 1.0, "b": 1.0}, workers=1, checkpoint=ck, log=lambda *a: None)
    expected = (grid - 1.0) ** 2 / 0.04                # the main valley, b = a^2, at every a
    np.testing.assert_allclose(res.chi2_array(), expected, atol=1e-6)
    n = obj.calls
    again = scans.profile(cfg, "a", grid, {"a": 1.0, "b": 1.0}, workers=1, checkpoint=ck, log=lambda *a: None)
    np.testing.assert_array_equal(again.chi2_array(), res.chi2_array())
    assert obj.calls - n <= 3 * len(grid)              # no fits, only the repair probes


def test_repair_pass_lifts_a_point_out_of_a_side_minimum(synthetic, tmp_path):
    """A checkpoint holding one point stuck in the side minimum (b = -1, chi^2 >= 3 above
    the valley) is repaired from its neighbours' solutions."""
    cfg, obj = synthetic
    grid = np.linspace(0.0, 2.0, 9)
    ck = str(tmp_path / "profile_a.jsonl")
    scans.profile(cfg, "a", grid, {"a": 1.0, "b": 1.0}, workers=1, checkpoint=ck, max_passes=0, log=lambda *a: None)
    recs = [json.loads(line) for line in open(ck)]
    bad = recs[4]
    bad["values"]["b"] = -1.0
    bad["chi2"] = obj.chi2(bad["values"])
    with open(ck, "w") as f:                       # the checkpoint keeps the lowest record per point
        for r in recs:
            if r["index"] != bad["index"]:
                f.write(json.dumps(r) + "\n")
        f.write(json.dumps(bad) + "\n")
    res = scans.profile(cfg, "a", grid, {"a": 1.0, "b": 1.0}, workers=1, checkpoint=ck, max_passes=2, log=lambda *a: None)
    np.testing.assert_allclose(res.chi2_array(), (grid - 1.0) ** 2 / 0.04, atol=1e-6)


def test_checkpoints_of_another_configuration_are_ignored(synthetic, tmp_path):
    """Results computed under different physics must never be reused: records carry the
    configuration's identity (data, solver, parameters, priors, grid), and a scan under a
    changed configuration refits every point. Fit settings are not part of it."""
    cfg, obj = synthetic
    grid = np.linspace(0.0, 2.0, 5)
    ck = str(tmp_path / "profile_a.jsonl")
    scans.profile(cfg, "a", grid, {"a": 1.0, "b": 1.0}, workers=1, checkpoint=ck, log=lambda *a: None)
    n = obj.calls
    cfg.fit = FitSettings(n_random=0, max_evals=300)            # settings change: resume
    scans.profile(cfg, "a", grid, {"a": 1.0, "b": 1.0}, workers=1, checkpoint=ck, log=lambda *a: None)
    assert obj.calls - n <= 3 * len(grid)
    n = obj.calls
    cfg.solver = {"dark_portal": {"epsilon_prime": 0.2}}      # physics change: refit everything
    msgs = []
    scans.profile(cfg, "a", grid, {"a": 1.0, "b": 1.0}, workers=1, checkpoint=ck, log=msgs.append)
    assert obj.calls - n > 10 * len(grid)
    assert any("another configuration" in m for m in msgs)
    assert cfg.identity() != cfg.identity(data_centre=(0.3, 0.3))
    assert cfg.identity() != cfg.identity(extra=dict(grids=[[0.0, 1.0]]))


def test_plane_of_the_synthetic_objective(synthetic):
    """With both parameters held, every point is one evaluation and no repair pass runs."""
    cfg, obj = synthetic
    xa, xb = np.linspace(0.5, 1.5, 5), np.linspace(0.0, 2.0, 4)
    n = obj.calls
    res = scans.plane(cfg, "a", "b", xa, xb, {"a": 1.0, "b": 1.0}, workers=1, log=lambda *a: None)
    A, B = np.meshgrid(xa, xb, indexing="ij")
    exact = np.minimum((A - 1) ** 2 / 0.04 + 25 * (B - A * A) ** 2, 3 + (A - 1) ** 2 / 0.04 + 25 * (B + 1) ** 2)
    np.testing.assert_allclose(res.chi2_array(), exact, atol=1e-12)
    assert res.passes == 0 and obj.calls - n == xa.size * xb.size


# ---------------------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------------------
def test_intervals_of_a_parabola_and_open_ends():
    x = np.linspace(-3, 3, 61)
    ivs = intervals.intervals(x, x ** 2, 1.0)
    assert len(ivs) == 1 and ivs[0].lo == pytest.approx(-1.0, abs=2e-3) and ivs[0].hi == pytest.approx(1.0, abs=2e-3)
    assert not ivs[0].lo_open and not ivs[0].hi_open
    up = intervals.intervals(x, np.maximum(x, 0.0) ** 2, 1.0)          # upper limit only
    assert up[0].lo_open and up[0].lo == -3.0 and up[0].hi == pytest.approx(1.0, abs=2e-3)
    two = intervals.intervals(x, (x ** 2 - 4.0) ** 2, 1.0)               # two disconnected intervals
    assert len(two) == 2
    xl = np.logspace(-6, 0, 61)
    lg = intervals.intervals(xl, (np.log10(xl) + 3.0) ** 2, 1.0, log=True)
    assert lg[0].lo == pytest.approx(1e-4, rel=1e-2) and lg[0].hi == pytest.approx(1e-2, rel=1e-2)
    d = x ** 2
    d[25] = np.inf                                                       # a failed point is skipped
    iv = intervals.intervals(x, d, 1.0)
    assert len(iv) == 1 and iv[0].lo == pytest.approx(-1.0, abs=2e-3)


def test_raster_plane_reads_each_column_from_its_own_minimum():
    """Exclusion planes are read per mediator mass (decided 2026-10-13): in each column
    Delta chi^2 counts from that column's minimum, limits come from the 1D thresholds, and a
    column whose minimum sits above the best fit by more than the largest threshold is
    flagged (the scanned couplings never reached the region without an effect)."""
    other = np.logspace(-10, -6, 5)
    chi2 = np.array([[0.0, 0.5, 2.0, 5.0, 9.0],
                     [5.0, 5.5, 6.0, 7.0, 9.0],
                     [np.inf] * 5])
    thr = {"95%": 3.841}
    rs = intervals.summarize_raster([1.0, 2.0, 3.0], other, chi2, thr, 0.0, log_other=True)
    c0, c1, c2 = rs["columns"]
    iv0 = c0["intervals"]["95%"]
    assert len(iv0) == 1 and iv0[0]["lo_open"] and not iv0[0]["hi_open"]
    u = -8.0 + (3.841 - 2.0) / (5.0 - 2.0)                    # crossing between 1e-8 and 1e-7, in log10
    assert iv0[0]["hi"] == pytest.approx(10.0 ** u, rel=1e-12)
    assert c1["column_min_dchi2"] == pytest.approx(5.0) and rs["elevated"] == [2.0]
    u1 = -7.0 + (3.841 - 2.0) / (4.0 - 2.0)                   # column-relative: 0, .5, 1, 2, 4
    assert c1["intervals"]["95%"][0]["hi"] == pytest.approx(10.0 ** u1, rel=1e-12)
    assert c2["column_min_dchi2"] is None
    np.testing.assert_allclose(rs["dchi2"][1], [0.0, 0.5, 1.0, 2.0, 4.0])


# ---------------------------------------------------------------------------------------
# Configurations and the command line
# ---------------------------------------------------------------------------------------
def test_configs_validate():
    for fn in sorted(os.listdir(CONFIGS)):
        cfg = config_mod.load(os.path.join(CONFIGS, fn))
        assert cfg.free_names
    raw = _tiny_raw()
    with pytest.raises(ValueError):
        config_mod.from_dict({**raw, "unknown_key": 1})
    bad = copy.deepcopy(raw)
    bad["profiles"]["g_D"]["range"] = [1e-20, 1.0]
    with pytest.raises(ValueError):
        config_mod.from_dict(bad)
    bad = copy.deepcopy(raw)
    bad["solver"]["grid"]["no_such_key"] = 1
    with pytest.raises(Exception):
        config_mod.from_dict(bad)
    bad = copy.deepcopy(raw)                        # FrequentistAnalysis' "dcp" prior went unused
    bad["priors"] = [{"name": "dcp", "type": "gaussian", "centre": 212.0, "width": 6.7}]
    with pytest.raises(KeyError):
        config_mod.from_dict(bad)
    ok = copy.deepcopy(raw)
    ok["priors"] = [{"name": "d_cp_deg", "type": "gaussian", "centre": 212.0, "width": 6.7},
                    {"name": "sum_masses_eV", "type": "half_gaussian", "limit": 0.055}]
    assert len(config_mod.from_dict(ok).priors) == 2


def test_command_line_end_to_end(tmp_path):
    """Fit, a three-point profile and a 2 x 2 plane with the real solver at a tiny grid:
    the outputs exist, intervals are reported, and Delta chi^2 is measured from the lowest
    chi^2 found."""
    from analysis import run
    raw = _tiny_raw()
    raw["parameters"] = {"g_D": {"start": 1e-4, "bounds": [1e-10, 3.54], "space": "log10"},
                         "u_e": {"start": 0.5, "bounds": [1e-10, 1.0], "space": "log10"}}
    raw["solver"]["dark_portal"].update(m_Zd_eV=2e6, u_mu=0.5)
    raw["profiles"] = {"g_D": {"range": [1e-6, 1e-2], "n": 3}}
    raw["planes"] = [{"x": "g_D", "y": "u_e", "x_range": [1e-6, 1e-2], "y_range": [1e-3, 0.5], "nx": 2, "ny": 2,
                      "raster": "g_D"}]
    raw["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))
    assert run.main([str(path), "--workers", "1"]) == 0
    out = tmp_path / "out" / "data"
    for f in ("fit.json", "profile_g_D.json", "profile_g_D.pdf", "plane_g_D_u_e.json", "plane_g_D_u_e.pdf", "manifest.json"):
        assert (out / f).exists(), f
    prof = json.load(open(out / "profile_g_D.json"))
    assert min(prof["dchi2"]) >= -1e-9 and set(prof["intervals"]) == {"68.27%", "90%", "95%"}
    plane = json.load(open(out / "plane_g_D_u_e.json"))
    assert plane["raster"] == "g_D" and plane["thresholds"]["95%"] == pytest.approx(3.841)
    assert len(plane["columns"]) == 2 and all(min(r) == pytest.approx(0.0, abs=1e-12) for r in plane["dchi2"])
    man = json.load(open(out / "manifest.json"))
    assert man["runs"][-1]["provenance"]["git_commit"]
