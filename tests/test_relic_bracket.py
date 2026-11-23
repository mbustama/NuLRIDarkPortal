"""Decision C's bracket settings (2026-11-07): dp_decay.BRACKET_DAUGHTERS (the share of the relics' decay daughters
kept as relics) and dp_relic_loss.BRACKET_INV_E (<1/E> of the grown relic states in the mass-change loss). At their
defaults (1) everything is the model; the tool tools/bracket_relic_population.py moves them in its own processes."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dp_decay  # noqa: E402
import dp_relic_loss  # noqa: E402
import dp_si_light  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

M_NU = np.array([1e-6, 8.68e-3, 0.0501])
H0, OM, OL, T0 = 1.4377e-33, 0.315, 0.685, 1.676e-4
GAM_FAST = np.zeros((3, 3))
GAM_FAST[2, 1] = GAM_FAST[2, 0] = GAM_FAST[1, 0] = 1e-20      # all decayed long before z = 1


def test_defaults_are_the_model():
    assert dp_decay.BRACKET_DAUGHTERS == 1.0
    assert dp_relic_loss.BRACKET_INV_E == 1.0


def test_daughters_share(monkeypatch):
    eta = np.ones(3)
    full = dp_decay.relic_composition(GAM_FAST, M_NU, eta, T0, H0, OM, OL)
    assert np.allclose(full, [3.0, 0.0, 0.0], atol=1e-9)            # the model: the number kept
    monkeypatch.setattr(dp_decay, "BRACKET_DAUGHTERS", 0.0)
    none = dp_decay.relic_composition(GAM_FAST, M_NU, eta, T0, H0, OM, OL)
    assert np.allclose(none, [1.0, 0.0, 0.0], atol=1e-9)            # x1: the decayed states' daughters dropped
    monkeypatch.setattr(dp_decay, "BRACKET_DAUGHTERS", 0.5)
    half = dp_decay.relic_composition(GAM_FAST, M_NU, eta, T0, H0, OM, OL)
    assert 1.0 < half[0] < 3.0 and np.allclose(half[1:], 0.0, atol=1e-9)
    # nothing decays: the setting has nothing to act on
    assert np.allclose(dp_decay.relic_composition(GAM_FAST * 1e-40, M_NU, eta, T0, H0, OM, OL), eta, atol=1e-9)


def _G_and_U():
    U = construct_PMNS(0.5903, 0.1503, 0.8587, 3.7)
    G = 1e-6 * np.diag([1.0, -1.0, 0.0]).astype(np.complex128)
    return G, U


def test_inv_E_scales_only_grown_states(monkeypatch):
    G, U = _G_and_U()
    M, z = 1e-3, 1.0
    for f in (dp_relic_loss.loss_coefficient,
              lambda *a: dp_si_light.loss_coefficient(*a, majorana=False)):
        grown = np.array([3.0, 0.0, 0.0])                            # all nu_1, grown by decays
        plain = np.ones(3)                                           # undecayed
        c_grown, c_plain = f(G, U, M_NU, grown, T0, z, M, 0.0), f(G, U, M_NU, plain, T0, z, M, 0.0)
        monkeypatch.setattr(dp_relic_loss, "BRACKET_INV_E", 2.0)
        assert np.isclose(f(G, U, M_NU, grown, T0, z, M, 0.0), 2.0 * c_grown, rtol=1e-12)
        assert f(G, U, M_NU, plain, T0, z, M, 0.0) == c_plain       # untouched where nothing grew
        monkeypatch.setattr(dp_relic_loss, "BRACKET_INV_E", 1.0)
        assert f(G, U, M_NU, grown, T0, z, M, 0.0) == c_grown       # back to the model, bit for bit
