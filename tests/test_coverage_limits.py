"""tools/coverage_limits.py: the window, its widening, and the reading (the limits' own, as tools/export_limits.py
reads a plane). The profiles themselves run on Tycho."""
import json
import os
import sys

import numpy as np

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
