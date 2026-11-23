"""Tests of analysis/secondary_spectra.py (2026-11-22): the number of neutrinos per process, and the spectra against
dp_si's and dp_decay's own normalizations."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import mean_free_paths as mf  # noqa: E402
from analysis import secondary_spectra as ss  # noqa: E402

X = np.linspace(0.0, 1.0, 200001)
trapz = getattr(np, "trapezoid", None) or np.trapz      # NumPy 2 renamed it (CI runs NumPy 2)


def test_secret_interaction_products_count():
    """Two neutrinos per 2 -> 2 collision, four per pair production (dp_si.coupling_tensors' sum rules); pair
    production's neutrinos stay below the scalar's largest energy fraction, (1 + sqrt(1 - 4 M^2/s)) / 2."""
    out = ss.si_spectra(x=X)
    assert trapz(out["s"], X) == pytest.approx(2.0, rel=1e-9)
    assert trapz(out["tu"], X) == pytest.approx(2.0, rel=1e-6)
    assert trapz(out["pairs"], X) == pytest.approx(4.0, rel=1e-3)
    xmax = 0.5 * (1.0 + np.sqrt(1.0 - 4.0 / ss.S_OVER_M2))
    assert np.all(out["pairs"][X > xmax + 1e-3] == 0.0)


@pytest.mark.parametrize("key, model, M, g", mf.EXAMPLES)
def test_one_daughter_per_decay(key, model, M, g):
    """keep + flip integrate to 1 (dp_decay.daughter_spectra: each kind's integral is its share of the rate); for a
    massless mediator and daughter they would be x and 1 - x."""
    x = np.geomspace(1e-12, 1.0, 400001)
    keep, flip = ss.decay_spectra(model, M, g, x=x)
    assert trapz(keep + flip, x) == pytest.approx(1.0, rel=1e-4)
    assert np.all(keep >= -1e-9) and np.all(flip >= -1e-9)


def test_figure_draws():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = ss.figure()
    assert len(fig.axes) == 3
    plt.close(fig)
