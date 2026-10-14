"""
The profiled curve of a potential plane (tools/combine_split_planes.py; decided 2026-10-14): the smallest chi^2 over
the balanced run and the variants that cover each point, only where the variant's eps' = P / x <= 1, read per mass.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, REPO)

import combine_split_planes as CS  # noqa: E402

THR = {"68.27%": 1.0, "90%": 2.706, "95%": 3.841}


def _summary(m, P, chi2, chi2_min=0.0):
    return dict(kind="plane", names=["m_Zd_eV", "g_D_eps_u2"], grids=[list(m), list(P)], log=[True, True],
                chi2=np.asarray(chi2).tolist(), chi2_min=chi2_min, thresholds=THR, title="t")


def test_profiled_split_takes_the_smallest_valid_chi2():
    m = np.logspace(-24, -18, 4)
    P = np.logspace(-60, -30, 31)
    # balanced: excluded above P = 1e-55 at every mass (chi2 grows by 10 per decade beyond it)
    bal = 10.0 * np.clip(np.log10(P) + 55.0, 0.0, None)[None, :].repeat(4, axis=0)
    # a screening variant x = 1e-40 covering the two lightest masses: only the cosmic potential, excluded above 1e-36
    var = 10.0 * np.clip(np.log10(P) + 36.0, 0.0, None)[None, :].repeat(2, axis=0)
    out = CS.combine(_summary(m, P, bal), [(_summary(m[:2], P, var), 1e-40)])
    c = np.asarray(out["chi2"])
    # where the variant covers and eps' <= 1 (P <= 1e-40), the smaller chi2; above 1e-40 the variant is not allowed
    valid = P <= 1e-40
    np.testing.assert_array_equal(c[:2, valid], np.minimum(bal[:2, valid], var[:, valid]))
    np.testing.assert_array_equal(c[:2, ~valid], bal[:2, ~valid])
    np.testing.assert_array_equal(c[2:], bal[2:])
    used = np.asarray(out["split_used"])
    assert used[:2][:, (P > 1e-55) & valid].min() == 1 and used[2:].max() == 0
    # read per mass at 95%: the light masses are allowed up to the variant's edge (1e-40, where eps' = 1), then the
    # balanced run's chi2 takes over; the heavy ones up to 1e-55 as before
    hi = [c_["intervals"]["95%"][0]["hi"] for c_ in out["columns"]]
    assert abs(np.log10(hi[2]) + 55 - 0.3841) < 1e-9   # chi2 = 10 per decade above 1e-55: crossing 0.3841 decades up
    assert hi[0] > 1e-41


def test_variant_grid_must_match():
    m = np.logspace(-24, -18, 4)
    P = np.logspace(-60, -30, 31)
    bal = np.zeros((4, 31))
    with pytest.raises(ValueError, match="grid"):
        CS.combine(_summary(m, P, bal), [(_summary(m[:2] * 1.5, P, np.zeros((2, 31))), 1e-40)])


def test_triangle_map_search_finds_the_first_crossing():
    """tools/ratio_triangle_maps.search: the first upward crossing of Delta chi^2 = 3.841 in log10 of the coupling,
    to 0.02 decades and then interpolated: exact for a linear Delta chi^2, and the lower of two crossings."""
    import ratio_triangle_maps as RT
    calls = []

    def lin(x):                                   # Delta chi^2 = 2 per decade above -55.3
        calls.append(x)
        return 10.0 + max(0.0, 2.0 * (x + 55.3))
    b, ref = RT.search(lin, -62.0, -6.0)
    assert ref == 10.0 and abs(b - (-55.3 + 3.841 / 2.0)) < 1e-9 and len(calls) < 30

    def bump(x):                                  # an excluded band at [-50, -48], then allowed, then excluded above -30
        return 4.0 if -50.0 <= x <= -48.0 else (10.0 if x >= -30.0 else 0.0)
    b, _ = RT.search(bump, -62.0, -6.0)
    assert -50.0 - 0.02 <= b <= -50.0
    b, _ = RT.search(lambda x: 0.0, -62.0, -6.0)
    assert b is None
    assert len(RT.ratio_points()) == 66 and all(abs(sum(w) - 1) < 1e-12 for w in RT.ratio_points())


def test_campaign_summary_reads_bounds_and_profiles(tmp_path):
    """tools/campaign_summary.py on a synthetic campaign: the balanced plane's bound per mass, its profiled curve from
    a split variant, a missing plane listed."""
    import campaign_summary as SUM
    m = np.logspace(-24, -18, 4)
    P = np.logspace(-60, -30, 31)
    bal = 10.0 * np.clip(np.log10(P) + 55.0, 0.0, None)[None, :].repeat(4, axis=0)
    var = 10.0 * np.clip(np.log10(P) + 36.0, 0.0, None)[None, :].repeat(2, axis=0)
    from analysis import intervals
    for name, mm, c, split in (("dp_emu_pot_bal_pos", m, bal, "balanced"), ("dp_emu_pot_x1m12_pos", m[:2], var, 1e-12)):
        d = tmp_path / "set" / name / "data"
        d.mkdir(parents=True)
        s = _summary(mm, P, c)
        rs = intervals.summarize_raster(mm, P, np.asarray(c), THR, 0.0, True)
        s.update(columns=rs["columns"], raster="m_Zd_eV")
        (d / "plane_m_Zd_eV_g_D_eps_trace.json").write_text(__import__("json").dumps(s))
        (d / "manifest.json").write_text(__import__("json").dumps(
            {"config": {"solver": {"dark_portal": {"coupling_split": split}}}}))
    SUM.main(["--set", "set", "--root", str(tmp_path)])
    out = __import__("json").loads((tmp_path / "set" / "summary" / "bounds.json").read_text())
    mb, b = out["bounds"]["dp_emu_pot_bal_pos"]
    assert abs(np.log10(b[3]) + 55 - 0.3841) < 1e-9
    mp, bp = out["bounds"]["dp_emu_pot_profiled_pos"]
    assert bp[0] > 1e-41 and abs(np.log10(bp[3]) + 55 - 0.3841) < 1e-9
    assert "dp_emu_pot_bal_neg" in out["missing"]
    assert (tmp_path / "set" / "summary" / "potential_bounds.png").exists()
