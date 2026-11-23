"""IceCube's released MESE flavor likelihood as the data surface (audit decision A2, 2026-11-09)."""
import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist

from analysis import surfaces

BPL, SPL = "icecube_mese_2025_grid", "icecube_mese_2025_grid_spl"


def _min_on_ftau0(name):
    return min(surfaces.chi2(x, 1.0 - x, name) for x in np.linspace(0.0, 0.98, 981))


def _min_on_fe0(name):
    return min(surfaces.chi2(0.0, y, name) for y in np.linspace(0.0, 0.999, 1000))


def test_reproduces_icecubes_quoted_levels():
    """arXiv:2510.24957: f_tau = 0 rejected at 91.9% and f_e = 0 at 98.7% C.L., from Wilks' 2-D regions."""
    assert 100 * chi2_dist.cdf(_min_on_ftau0(BPL), 2) == pytest.approx(91.9, abs=0.05)
    assert 100 * chi2_dist.cdf(_min_on_fe0(BPL), 2) == pytest.approx(98.7, abs=0.05)


def test_best_node_and_minimum():
    """IceCube's best fit 0.30 : 0.37 : 0.33; the grid's best node (0.297, f_tau 0.326), TS 0.038 there."""
    s = surfaces.surface(BPL)
    assert s["mu_x"] == pytest.approx(0.297, abs=0.006) and s["mu_y"] == pytest.approx(0.377, abs=0.006)
    assert surfaces.chi2(s["mu_x"], s["mu_y"], BPL) == pytest.approx(s["ts_min"], abs=1e-12)
    assert s["ts_min"] == pytest.approx(0.038, abs=0.001)
    xs = np.linspace(0.0, 0.98, 99)
    vals = [surfaces.chi2(x, y, BPL) for x in xs for y in np.linspace(0.0, 1.0 - x, 50)]
    assert min(vals) >= s["ts_min"] - 1e-12


def test_spikes_only_in_the_broken_power_law_grid():
    assert surfaces._grid(surfaces.surface(BPL)["file"])[5] > 0
    assert surfaces._grid(surfaces.surface(SPL)["file"])[5] == 0


def test_recentred_moves_rigidly():
    s = surfaces.surface(BPL)
    moved = surfaces.recentred(BPL, 0.33, 0.34)
    assert surfaces.chi2(0.33, 0.34, moved) == pytest.approx(s["ts_min"], abs=1e-12)
    d = (0.33 - s["mu_x"], 0.34 - s["mu_y"])
    for fe, fmu in [(0.25, 0.40), (0.35, 0.30), (0.20, 0.50)]:
        assert surfaces.chi2(fe + d[0], fmu + d[1], moved) == pytest.approx(surfaces.chi2(fe, fmu, BPL), abs=1e-12)


def test_old_surface_unchanged():
    """The July surface stays available (for comparisons): its centre still gives 0."""
    s = surfaces.surface("icecube_mese_2025")
    assert surfaces.chi2(s["mu_x"], s["mu_y"], "icecube_mese_2025") == pytest.approx(0.0, abs=1e-12)
