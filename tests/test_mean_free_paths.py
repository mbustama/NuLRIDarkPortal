"""Tests of analysis/mean_free_paths.py, the paper's figure of the mean free paths of each family (2026-11-22): the
scattering rate against the transport's own function, the conversions undamped, the longitudinal mode left out as the
transport leaves it out, the crossings of the Hubble length inside the plotted range, and the cache."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import mean_free_paths as mf  # noqa: E402
from analysis import si_mean_free_path as si_mfp  # noqa: E402

E = np.geomspace(si_mfp.E_FIG_MIN, si_mfp.E_FIG_MAX, 51)


@pytest.mark.parametrize("key, model, M, g", mf.EXAMPLES)
def test_examples(key, model, M, g):
    r = mf.rates(model, M, g, E)
    # decay lengths grow as E; nu_1 is stable; nu_2 and nu_3 cross the Hubble length inside the plotted range
    assert np.all(r["decay"][:, 0] == 0.0)
    lam = mf.HBARC_EV_MPC / r["decay"][:, 1:]
    np.testing.assert_allclose(lam / E[:, None], np.broadcast_to(lam[:1] / E[0], lam.shape), rtol=1e-12)
    # the longitudinal mode's collisions: in the transport (and the figure) for L_e - L_mu above its threshold
    import dp_si_light
    if model == "L_e-L_mu" and r["Ymax"] >= dp_si_light.GOLDSTONE_Y_MIN:
        assert np.all(r["goldstone"] > 0)
    else:
        assert np.all(r["goldstone"] == 0)
    # the relics have decayed into nu_1, their number kept
    assert r["eta"].sum() == pytest.approx(3.0, rel=1e-12) and r["eta"][0] > 2.99



@pytest.mark.parametrize("key, model, M, g", mf.EXAMPLES)
def test_scattering_rate_is_the_transports(key, model, M, g):
    """Gamma_u from dark_portal_solver._soft_scattering_rates at z = 0, at a coupling large enough for it to act."""
    import dark_portal_solver as dps
    import dp_absorption
    gbig = 1e-2 if model == "dark_portal_tau" else 1e-12
    U, m = mf.neutrinos()
    G = mf.coupling(model, gbig)
    r = mf.rates(model, M, gbig, E[:1])
    ab0 = dp_absorption.AbsorptionSetup(G, U, M, 0.0, anti=False)
    out = dps._soft_scattering_rates(ab0, U, M, m, si_mfp.T_NU0, r["eta"], np.array([1e-3, 0.0]), mf.H0_EV,
                                     mf.OMEGA_M, mf.OMEGA_L, G_jump=G)
    assert out is not None
    assert r["Gu"] == pytest.approx(out[-1], rel=1e-12)


@pytest.mark.parametrize("key, model, M, g", mf.EXAMPLES)
def test_conversions_are_undamped(key, model, M, g):
    """The transport's factor w^2 / (w^2 + D^2) on the conversions: D, the scatterings' damping plus the decay widths,
    is far below the splitting w = (m_a^2 - m_b^2) / 2E at every plotted energy, so the factor is 1."""
    U, m = mf.neutrinos()
    r = mf.rates(model, M, g, E)
    Gm = U.conj().T @ mf.coupling(model, g) @ U
    GG = np.real(np.diag(Gm @ Gm.conj().T))
    for a in range(3):
        for b in range(a + 1, 3):
            w = abs(m[a] ** 2 - m[b] ** 2) / (2.0 * E)
            D = r["Gu"] * (0.5 * (GG[a] + GG[b])) + 0.5 * (r["decay"][:, a] + r["decay"][:, b])
            assert np.all((D / w) ** 2 < 1e-12)       # the factor w^2 / (w^2 + D^2) is 1 to 1e-12


def test_cache_matches_compute():
    if not os.path.exists(mf.CACHE):
        pytest.skip("no cache")
    d = np.load(mf.CACHE)
    out = mf.compute(path=None)
    for k in out:
        np.testing.assert_allclose(out[k], d[k], rtol=1e-12)


def test_figure_draws():
    if not (os.path.exists(mf.CACHE) and os.path.exists(si_mfp.CACHE)):
        pytest.skip("no cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = mf.figure()
    assert len(fig.axes) == 3
    plt.close(fig)
