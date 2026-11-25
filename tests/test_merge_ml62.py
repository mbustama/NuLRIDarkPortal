"""tools/merge_ml62.py: the refit's blocks join the stored plane point by point in tools/export_limits.py, at the
stored grid's floats, keeping the lower chi^2."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import export_limits  # noqa: E402
import merge_ml62  # noqa: E402

NAMES = ["m_Zd_eV", "g_D"]


def _write_plane(d, xs, ys, chi):
    os.makedirs(d, exist_ok=True)
    json.dump(dict(kind="plane", names=NAMES, grids=[list(xs), list(ys)], log=[True, True],
                   chi2=np.asarray(chi).tolist()), open(os.path.join(d, "plane_m_Zd_eV_g_D.json"), "w"))
    with open(os.path.join(d, "plane_m_Zd_eV_g_D.jsonl"), "w") as f:
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                f.write(json.dumps({"index": [i, j], "held": {NAMES[0]: x, NAMES[1]: y}, "chi2": chi[i][j],
                                    "pass": 0}) + "\n")


def test_refit_blocks_join_the_stored_plane_at_its_floats(tmp_path):
    xs = np.logspace(4, 9, 41)
    ys = np.logspace(-10, -2, 41)
    stored = tmp_path / "campaign_ml" / "now_icecube" / "p_fx"
    chi_s = np.full((41, 41), 10.0)
    chi_s[17, 30:] = 20.0                                   # excluded above 1e-4 at column 17
    _write_plane(str(stored / "data"), xs, ys, chi_s.tolist())
    # a refit block over columns 16-17, its grid rebuilt from its ends as analysis.config._grid does (ulp off)
    bx = np.array([np.nextafter(xs[16], np.inf), np.nextafter(xs[17], 0.0)])
    assert bx[0] != xs[16] and bx[1] != xs[17]
    chi_b = np.full((2, 41), 10.5)                          # higher than stored at most points ...
    chi_b[1, 30:] = 11.0                                    # ... lower inside column 17's excluded range
    refit = tmp_path / "campaign_ml62" / "now_icecube" / "p_fx"
    _write_plane(str(refit / "b8" / "data"), bx, ys, chi_b.tolist())
    assert merge_ml62.merge(str(refit), str(stored)) == 1
    planes = export_limits.planes(str(tmp_path / "campaign_ml"))
    assert list(planes) == ["now_icecube/p_fx"] and len(planes["now_icecube/p_fx"]) == 2
    res = export_limits.export("now_icecube/p_fx", planes["now_icecube/p_fx"], str(tmp_path / "out"))
    s = json.load(open(tmp_path / "out" / "now_icecube" / "p_fx" / "data" / "plane_m_Zd_eV_g_D.json"))
    assert len(s["grids"][0]) == 41                          # no near-duplicate columns
    chi = np.array(s["chi2"], float)
    assert chi[17, 35] == 11.0 and chi[17, 0] == 10.0 and chi[16, 0] == 10.0   # the lower of the two everywhere
    col = [c for c in res["columns"] if abs(c["log10_m"] - np.log10(xs[17])) < 1e-6][0]
    assert col["excluded_log10_g"] == []                     # the refit's lower chi^2 removes the exclusion


def test_off_grid_points_are_refused(tmp_path):
    xs, ys = np.logspace(4, 9, 41), np.logspace(-10, -2, 41)
    stored = tmp_path / "s"
    _write_plane(str(stored / "data"), xs, ys, np.full((41, 41), 1.0).tolist())
    refit = tmp_path / "r"
    _write_plane(str(refit / "b0" / "data"), [1.1e6, 1.3e6], ys, np.full((2, 41), 1.0).tolist())
    with pytest.raises(ValueError):
        merge_ml62.merge(str(refit), str(stored))
