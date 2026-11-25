"""tools/coverage.py: the pseudo-data draws (the fits run on Tycho, in its pilot)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import coverage  # noqa: E402
from analysis import surfaces  # noqa: E402


def test_segment_draws_scatter_with_the_errors_chi2_applies():
    spec = dict(value=np.array([4.0, 0.0, 2.0]), lo=np.array([3.0, 0.0, 1.0]), hi=np.array([6.0, 1.0, 2.5]))
    C = np.array([10.0, 0.5, 2.0])      # far above 0: no floor on the first segment
    D = coverage.segment_draws(C, spec, 200000, np.random.default_rng(1))
    assert D.shape == (200000, 3) and np.all(D >= 0.0)
    up, dn = D[:, 0] > C[0], D[:, 0] < C[0]
    # above the truth the lower error (1.0) sets the spread, below it the upper one (2.0): half-normal means
    assert abs(np.mean(D[up, 0] - C[0]) - 1.0 * np.sqrt(2 / np.pi)) < 0.01
    assert abs(np.mean(C[0] - D[dn, 0]) - 2.0 * np.sqrt(2 / np.pi)) < 0.02
    # a segment measured at 0 (lower error 0) never scatters upward; downward it is floored at 0
    assert np.all(D[:, 1] <= C[1] + 1e-15)


def test_flavor_draws_follow_the_released_likelihood():
    name = "icecube_mese_2025_grid"     # set 1's released grid
    p = surfaces.surface(name)
    off = coverage.flavor_draws(name, 4000, np.random.default_rng(2))
    f = off + np.array([p["true_mu_x"], p["true_mu_y"]])
    assert off.shape == (4000, 2)
    assert np.all(f > -0.01) and np.all(f.sum(axis=1) < 1.01)            # inside the triangle (to the jitter)
    ts = np.array([surfaces.chi2(a, b, p) for a, b in f[:1000]]) - p["ts_min"]
    # the draws concentrate where the likelihood does: most lie inside its 95% region for 2 dof
    assert np.mean(ts < 5.99) > 0.85
