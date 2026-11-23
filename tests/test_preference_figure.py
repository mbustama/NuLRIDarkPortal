"""Tests of analysis/preference_figure.py (the paper's Fig. 12) on its stored data."""
import os

import numpy as np
import pytest

from analysis import preference_figure as pf

pytestmark = pytest.mark.skipif(not os.path.exists(pf.DATA), reason="preference data not present")


def test_load_sorted_and_finite():
    c = pf.load()
    assert set(c) == {"now_icecube", "now_mese_dr2"}
    for ds in c:
        for plane, _, _ in pf.FAMILIES:
            x, p, g = c[ds][plane]
            assert np.all(np.diff(x) > 0)
            assert np.all(np.isfinite(p)) and np.all(p >= -1e-6)   # the smallest coupling is a candidate
            assert len(x) == len(p) == len(g)


def test_peaks_as_quoted_in_the_paper():
    c = pf.load()
    p, x, g = pf.peak(c, "now_icecube", "dp_tau_mev_fx")
    assert x == pytest.approx(6.375) and 4.0 < p < 5.0 and g == pytest.approx(-3.0)
    p, x, _ = pf.peak(c, "now_mese_dr2", "dp_tau_mev_fx")
    assert x == pytest.approx(5.625) and 3.0 < p < 3.84


def test_figure_builds():
    import matplotlib
    matplotlib.use("Agg")
    fig = pf.figure()
    assert len(fig.axes) == 1 and len(fig.axes[0].lines) >= 6
