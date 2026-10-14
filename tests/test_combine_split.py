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
