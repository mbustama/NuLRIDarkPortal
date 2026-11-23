"""Tests of analysis/event_rates_figure.py (2026-11-22): the stored runs reproduce data set 2's fit without a new
interaction, and the figure's three views of the events add up to the fit's bins."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import event_rates_figure as erf  # noqa: E402

HAVE = all(os.path.exists(os.path.join(erf.DIR, f"{k}.npz")) for k in erf.KEYS)
pytestmark = pytest.mark.skipif(not HAVE, reason="no stored event rates")


@pytest.mark.parametrize("key", erf.KEYS)
def test_runs_reproduce_the_fit_without_a_new_interaction(key):
    d = np.load(os.path.join(erf.DIR, f"{key}.npz"))
    assert float(d["chi2_sm"]) == pytest.approx(float(d["chi2_sm_stored"]), abs=1e-3)


@pytest.mark.parametrize("key", erf.KEYS)
def test_views_add_up_to_the_fits_bins(key):
    d = np.load(os.path.join(erf.DIR, f"{key}.npz"))
    v = erf.load(key)
    le = erf.dr2_edges()
    for tag, field in (("sm", "sm"), ("ex", "ex")):
        tot = sum(d[f"{tag}_dr2_{c}_mu"].sum() for c in d["dr2_configs"])
        assert v["tracks"][field].sum() == pytest.approx(tot, rel=1e-12)
        high = sum(d[f"{tag}_dr2_{c}_mu"][le[np.nonzero(d[f"dr2_{c}_select"])[1]] >= np.log10(erf.E_DIR_MIN_GEV) - 1e-9].sum()
                   for c in d["dr2_configs"])
        assert v["directions"][field].sum() == pytest.approx(high, rel=1e-12)
    assert v["tracks"]["data"].sum() == sum(d[f"dr2_{c}_data"].sum() for c in d["dr2_configs"])
    n = len(v["cascades"]["edges"]) - 1
    assert len(v["cascades"]["data"]) == n and v["cascades"]["data"].sum() > 0
    # the background is the same with and without the new interaction (the nuisances are held)
    for s in ("cascades", "tracks", "directions"):
        np.testing.assert_allclose(v[s]["sm"] - v[s]["sm_astro"], v[s]["ex"] - v[s]["ex_astro"], rtol=1e-9, atol=1e-9)


def test_figure_draws():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = erf.figure()
    assert len([a for a in fig.axes if a.get_label() != "<colorbar>"]) >= 9
    plt.close(fig)
