"""tools/coverage_limits.py: the window, its widening, and the reading (the limits' own, as tools/export_limits.py
reads a plane). The profiles themselves run on Tycho."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import coverage_limits as cl  # noqa: E402
import export_limits  # noqa: E402

Y = [float(g) for g in np.logspace(-10, -2, 41)]          # the L_e-L_alpha MeV planes' couplings (0.2 decade)
NAMES = ("m_Zd_eV", "g_D")
M = 1333521.432163324


def test_window_spans_one_decade_below_to_0p4_above():
    w = np.log10(cl.window(Y, -3.7))
    assert np.allclose(w, [-4.6, -4.4, -4.2, -4.0, -3.8, -3.6, -3.4])
    assert Y[0] not in cl.window(Y, -9.9)                    # the smallest coupling is added apart, not here


def test_window_reaches_a_lower_lowest_coupling():
    w = np.log10(cl.window(Y, -2.42, hi=0.5, lowest=-4.63))  # the refit's L_e-L_tau limit at 1.3 MeV
    assert np.allclose(w, np.arange(-4.6, -1.95, 0.2))
    assert np.allclose(np.log10(cl.window(Y, -3.7, lowest=-4.0)), np.log10(cl.window(Y, -3.7)))  # never narrower


def test_widen_adds_the_next_coupling_beyond_an_edge_minimum():
    w = cl.window(Y, -3.7)
    chi = {Y[0]: 10.0, **{g: 12.0 for g in w}}
    chi[w[0]] = 9.0                                          # lowest at the window's bottom
    assert np.allclose(np.log10(cl.widen(Y, chi)), [-4.8])
    chi[w[0]], chi[w[-1]] = 12.0, 9.0                        # lowest at its top
    assert np.allclose(np.log10(cl.widen(Y, chi)), [-3.2])
    chi[w[-1]], chi[w[3]] = 12.0, 9.0                        # inside: nothing to add
    assert cl.widen(Y, chi) == []
    chi = {Y[0]: 10.0, Y[-2]: 12.0, Y[-1]: 9.0}              # at the grid's top: nothing beyond
    assert cl.widen(Y, chi) == []


def _column(lg_cross):
    """No effect at 1e-10 (chi^2 10), then Delta chi^2 rising linearly in log10 g through 3.84 at lg_cross."""
    w = cl.window(Y, -3.7)
    chi = {Y[0]: 10.0}
    for g in w:
        chi[g] = 10.0 + max(0.0, 3.84 + 8.0 * (np.log10(g) - lg_cross))
    return chi


def test_read_covers_below_the_edge_and_not_above():
    chi = _column(-3.9)
    cov_lo, lim = cl.read(NAMES, M, chi, 10 ** -4.1)
    cov_hi, _ = cl.read(NAMES, M, chi, 10 ** -3.7)
    assert cov_lo and not cov_hi
    assert abs(lim - (-3.9)) < 0.03


def test_read_matches_export_limits(tmp_path):
    chi = _column(-3.9)
    gs = sorted(chi)
    d = tmp_path / "now_icecube" / "p_fx" / "data"
    os.makedirs(d)
    json.dump(dict(kind="plane", names=list(NAMES), grids=[[M], gs], log=[True, True], chi2=[[chi[g] for g in gs]]),
              open(d / "plane_m_Zd_eV_g_D.json", "w"))
    pl = export_limits.planes(str(tmp_path))
    res = export_limits.export("now_icecube/p_fx", pl["now_icecube/p_fx"], str(tmp_path / "out"))
    _, lim = cl.read(NAMES, M, chi, 10 ** -3.7)
    assert res["columns"][0]["lowest_excluded_log10_g"] == round(lim, 4)


def test_reread_reads_a_finished_run(tmp_path):
    chi_lo, chi_hi = _column(-3.9), _column(-4.5)          # edges at 10^-3.9 and 10^-4.5: 10^-4.1 covered, then not
    out = dict(names=list(NAMES), mass_eV=M, log10g_true=-4.1, real={repr(g): c for g, c in _column(-3.9).items()},
               results=[dict(toy=0, chi2={repr(g): c for g, c in chi_lo.items()}),
                        dict(toy=1, chi2={repr(g): c for g, c in chi_hi.items()})])
    p = tmp_path / "cov.json"
    json.dump(out, open(p, "w"))
    o = cl.reread(str(p))
    assert [r["covered"] for r in o["results"]] == [True, False]
    assert o["coverage"] == 0.5 and o["real_read"]["covered"]


def test_main_keeps_the_true_coupling_apart_and_the_window_wide(tmp_path, monkeypatch):
    """main's bookkeeping with the fits stubbed (2026-11-26): the window from --lowest-log10g, the true coupling fitted
    for every pseudo-experiment but kept out of the column, the partial results dropped at the end."""
    class Par:
        name, start = "nu", 1.0

    class Plane:
        x, y = NAMES
        x_grid, y_grid = [M], Y

    class Cfg:
        planes, parameters = [Plane], [Par]

    class Obj:
        surface = "stub"

    def chi_of(k, g):                                        # no effect: 10; lowest at 10^-4.2
        lg = np.log10(g)
        return 10.0 + (0.5 if lg < -9 else (lg + 4.2) ** 2) + 0.1 * (k or 0)

    seen = []

    def init(path, n_random=0):
        seen.append(n_random)
        cl._W.update(cfg=Cfg, obj=Obj, real=(None, None, {}), toy=None)

    def task(t):
        k, _, _, held, _ = t
        return dict(toy=k, held=held, chi2=chi_of(k, held[NAMES[1]]), values=dict(held, nu=2.0 + (k or 0)), message="",
                    seconds=0.0)

    def truth(t):
        held, _ = t
        return dict(chi2=chi_of(None, held[NAMES[1]]), values=dict(held, nu=1.0), f_true=[0.3, 0.3], C_true=[1.0, 2.0])

    class Result:
        def __init__(self, r):
            self.r = r

        def get(self):
            return self.r

    class Pool:
        def __init__(self, *a, **k):
            init(*k["initargs"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def apply_async(self, f, args):
            return Result(f(*args))

        def map(self, f, it, chunksize=1):
            return [f(x) for x in it]

        def imap_unordered(self, f, it, chunksize=1):
            return (f(x) for x in it)

    class Ctx:
        pass

    Ctx.Pool = Pool
    monkeypatch.setattr(cl.mp, "get_context", lambda method: Ctx)
    for name, f in (("_init", init), ("_task", task), ("_truth", truth),
                    ("flavor_draws", lambda s, n, rng: np.zeros((n, 2))),
                    ("segment_draws", lambda C, spec, n, rng: np.tile(C, (n, 1)))):
        monkeypatch.setattr(cl, name, f)
    monkeypatch.setenv("NULRI_FLUX_CACHE", str(tmp_path))
    out = str(tmp_path / "cov.json")
    cl.main(["cfg.json", "--mass", str(M), "--true-log10g", "-2.42", "--toys", "3", "--workers", "1",
             "--lowest-log10g", "-4.63", "--window-hi", "0.5", "--out", out])
    o = json.load(open(out))
    assert "partial" not in o and len(o["results"]) == 3 and o["window_unresolved"] == []
    for k, r in enumerate(o["results"]):
        lg = np.log10(sorted(float(g) for g in r["chi2"]))
        assert np.allclose(lg, [-10.0] + list(np.arange(-4.6, -1.95, 0.2)))      # the grid only: no -2.42
        assert np.isclose(r["chi2_true"], chi_of(k, 10.0 ** -2.42))
        assert np.isclose(r["dchi2_true"], (-2.42 + 4.2) ** 2, atol=1e-6)        # from the column's lowest (at -4.2)
        assert r["covered"]                                                       # 3.17 < 3.84
        assert r["values"][repr(10.0 ** -2.42)] == {"nu": 2.0 + k} and len(r["values"]) == len(r["chi2"]) + 1
    assert o["n_random"] == 0 and o["only"] == [0, 3] and set(seen) == {0}
    # 2026-11-26: the extra starts reach the workers, and --only fits a part of the same draws
    del seen[:]
    cl.main(["cfg.json", "--mass", str(M), "--true-log10g", "-2.42", "--toys", "3", "--workers", "1", "--only", "1:3",
             "--n-random", "16", "--lowest-log10g", "-4.63", "--window-hi", "0.5", "--out", out])
    p = json.load(open(out))
    assert set(seen) == {16} and p["n_random"] == 16 and [r["toy"] for r in p["results"]] == [1, 2]
    assert p["results"][0]["chi2"] == o["results"][1]["chi2"] and p["coverage"] == 1.0
    with pytest.raises(SystemExit):
        cl.main(["cfg.json", "--mass", str(M), "--true-log10g", "-2.42", "--toys", "3", "--only", "2:5", "--out", out])
