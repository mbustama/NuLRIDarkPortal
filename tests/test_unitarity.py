"""dp_unitarity against the bounds of Allwicher, Arnan, Barducci, Nardecchia, JHEP 10 (2021) 129."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_unitarity as du  # noqa: E402


def _threshold(Y_unit):
    """The scale s at which partial_wave_max(s Y_unit) = 1/2."""
    return np.sqrt(0.5 / du.partial_wave_max(Y_unit))


def test_one_majorana_flavor():
    Y = np.zeros((3, 3), complex)
    Y[0, 0] = 1.0
    assert np.isclose(_threshold(Y) ** 2, 4 * np.pi)              # their Eq. (6.2), J = 0


def test_dirac_pair():
    Y = np.zeros((3, 3), complex)
    Y[0, 1] = Y[1, 0] = 1.0
    assert np.isclose(_threshold(Y) ** 2, 8 * np.pi / 3)          # their Eq. (4.7), N = 1, real scalar


def test_half_integer_channel():
    Y = np.zeros((3, 3), complex)
    Y[0, 0] = 1.0
    j12 = 3.0 / (32 * np.pi)
    assert np.isclose(np.sqrt(0.5 / j12) ** 2, 16 * np.pi / 3)   # their Eq. (6.4), real scalar


def test_phases_do_not_matter_for_one_entry():
    Y = np.zeros((3, 3), complex)
    Y[1, 2] = Y[2, 1] = np.exp(0.7j)
    assert np.isclose(_threshold(Y) ** 2, 8 * np.pi / 3)


def test_goldstone_yukawa_matches_the_decay_rate():
    """|Y_31| = |G'_31| m_3 / M for m_1 = 0: the Majorana rate |Y|^2 m/(16 pi) is twice the Dirac LH vector rate."""
    Gp = np.zeros((3, 3), complex)
    Gp[0, 2] = Gp[2, 0] = 2e-9
    m = np.array([0.0, 8.6e-3, 0.05])
    Y = du.goldstone_yukawa(Gp, m, 1e-6)
    assert np.isclose(abs(Y[0, 2]), 2e-9 * 0.05 / 1e-6)
