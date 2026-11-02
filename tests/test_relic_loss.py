"""dp_relic_loss: the beam's energy loss in relic mass changes (notes Sec. 12.8 item 1, 2026-10-24)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_relic_loss as rl  # noqa: E402


def test_shift_keeps_number_and_moves_the_mean():
    E = np.geomspace(1e12, 1e16, 200)
    h = np.sqrt(E[1] / E[0])
    dE = E * h - E / h
    F = np.zeros((len(E), 1))
    F[150, 0] = 1.0
    n0 = (F[:, 0] * dE).sum()
    mean0 = (F[:, 0] * dE * E).sum() / n0
    d = 0.3 * E[150]
    rl.shift_step(E, F, np.full((len(E), 1), d))
    assert np.isclose((F[:, 0] * dE).sum(), n0, rtol=1e-12)
    # the content was uniform over the parent bin: its mean moves by exactly d
    lo, hi = E[150] / h, E[150] * h
    exact = 0.5 * (lo + hi) - d
    mean = (F[:, 0] * dE * E).sum() / n0
    assert abs(mean - exact) / exact < 0.02          # bin-centre bookkeeping of the target bins


def test_shift_below_the_grid_leaves_it():
    E = np.geomspace(1e12, 1e16, 50)
    F = np.ones((50, 3))
    rl.shift_step(E, F, np.full((50, 3), 2e16))
    assert not F.any()


def test_coefficient_signs_and_scaling():
    U = np.eye(3, dtype=complex)
    m = np.array([1e-6, 8.68e-3, 0.0501])
    G = 1e-6 * np.ones((3, 3), dtype=complex)
    c1 = rl.loss_coefficient(G, U, m, np.ones(3), 1.676e-4, 0.0, 1e-9, 0.0)
    assert c1 > 0.0                                    # up-conversions of the light relic dominate
    c2 = rl.loss_coefficient(2 * G, U, m, np.ones(3), 1.676e-4, 0.0, 1e-9, 0.0)
    assert np.isclose(c2, 4 * c1)                      # |G'_sr|^2 (the beam factor (G G^dag)_aa apart)
    c3 = rl.loss_coefficient(G, U, m, np.ones(3), 1.676e-4, 0.0, 2e-9, 0.0)
    assert np.isclose(c3, c1 / 4)                      # 1 / M_t^2
    assert rl.loss_coefficient(0.74 * np.eye(3), U, m, np.ones(3), 1.676e-4, 0.0, 1e-9, 0.0) == 0.0
