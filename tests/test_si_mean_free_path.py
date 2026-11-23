"""Tests of analysis/si_mean_free_path.py, the appendix figure of nu_3's mean free path on the relic neutrinos
(2026-11-22): the relics' distribution, the cross-section recipe, the s channel against its narrow-width limit, the
quadrature, and the port against the cache made by the original script (benchmark_rates.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import si_mean_free_path as mfp  # noqa: E402

X, W = np.polynomial.legendre.leggauss(mfp.NODES)


@pytest.fixture(scope="module")
def by_rel():
    return {r: mfp.cases(r) for r in range(3)}


def test_masses_and_widths(by_rel):
    masses = [by_rel[r][1][0]["m"] for r in range(3)]
    assert masses[0] == pytest.approx(1e-6, rel=1e-12)
    assert masses[1] == pytest.approx(np.sqrt(1e-12 + 7.537e-5), rel=1e-12)
    assert masses[2] == pytest.approx(np.sqrt(1e-12 + 2.511e-3), rel=1e-12)
    for c in by_rel[0][1]:
        assert 0.0 < c["Gam"] / mfp.M_MED < 1e-6


@pytest.mark.parametrize("rel", [0, 1, 2])
def test_relic_distribution_is_normalized_and_splits_by_chirality(by_rel, rel):
    _, si = mfp.code()
    m = by_rel[rel][1][0]["m"]
    edges = mfp.panel_edges(mfp.M_MED ** 2 / (2.0 * max(m, mfp.T_NU0)), 1e-8 * mfp.M_MED, m)
    norm = {p: mfp.panel_integral(lambda e, p=p: si._sigma_density(e, m, mfp.T_NU0, p), edges, X, W)
            for p in (0, 1, -1)}
    assert abs(norm[0] - 1.0) < 1e-8
    assert abs(norm[1] + norm[-1] - norm[0]) < 1e-8


def test_tensor_recipe_matches_the_closed_form(by_rel):
    """The profile-sum recipe (used for the Dirac scalar) against the closed form (scalar, Majorana)."""
    _, si = mfp.code()
    _, cs, scalar_factors = by_rel[0]
    A = cs[0]
    Creg = si.coupling_tensors(A["coupling"])[1][:, mfp.INC, 0, :].sum(axis=-1)
    for s in mfp.M_MED ** 2 * np.array([0.01, 0.5, 0.999, 1.0, 1.003, 3.9, 4.5, 30.0, 1e3]):
        recipe = np.sum(si.SCALAR_W * Creg * scalar_factors(s, A["Gam"]))
        closed = si.sigma_scalar_majorana_total(s, mfp.M_MED, A["Gam"], A["coupling"], mfp.INC, 0)
        assert recipe == pytest.approx(closed, rel=1e-12)


@pytest.mark.parametrize("key", list(mfp.MODELS))
def test_s_channel_at_its_narrow_width_limit(by_rel, key):
    """On the fast relic at 0.2, 1, 5 PeV; on the slow ones at E = M^2 / 2m and 0.3 of the relics' speed either side
    (where rho(eps_R) is not in its tail)."""
    case = next(c for c in by_rel[0][1] if c["key"] == key)
    for E in (2e14, 1e15, 5e15):
        num, nw = mfp.narrow_width_s_channel(E, case)
        assert num == pytest.approx(nw, rel=1e-4)
    for rel in (1, 2):
        case = next(c for c in by_rel[rel][1] if c["key"] == key)
        v = 3.15 * mfp.T_NU0 / case["m"]
        for E in mfp.M_MED ** 2 / (2.0 * case["m"]) * np.array([1 - 0.3 * v, 1.0, 1 + 0.3 * v]):
            num, nw = mfp.narrow_width_s_channel(E, case)
            assert num == pytest.approx(nw, rel=1e-4)


def test_doubling_the_nodes_changes_little(by_rel):
    x2, w2 = np.polynomial.legendre.leggauss(2 * mfp.NODES)
    for rel in range(3):
        m = by_rel[rel][1][0]["m"]
        for case in by_rel[rel][1]:
            nch = mfp.n_collision_channels(case["key"], case["cols"])
            for E in (3e12, mfp.M_MED ** 2 / (2.0 * max(m, mfp.T_NU0)), 1e15, 5e16):
                a = mfp.thermal_rate(E, case, X, W)
                b = mfp.thermal_rate(E, case, x2, w2)
                assert np.max(np.abs(b - a)) < 1e-8 * abs(a[:nch].sum())


@pytest.mark.skipif(not os.path.exists(mfp.CACHE), reason="no cache")
def test_the_port_reproduces_the_cache():
    """compute() on a sample of the cache's energies (both resonances of the slow relics included) against the cache
    that the original script wrote; and the same energy grid."""
    E, rows, cols, masses = mfp.load()
    assert np.allclose(mfp.relic_energies(masses), E, rtol=1e-14, atol=0)
    idx = np.unique(np.r_[0, len(E) - 1, np.argmin(np.abs(E - mfp.M_MED ** 2 / (2 * masses[1]))),
                          np.argmin(np.abs(E - mfp.M_MED ** 2 / (2 * masses[2]))), 300])
    out = mfp.compute(workers=1, energies=E[idx], path=None, conv_stride=1)
    assert float(out["node_doubling"]) < 1e-8
    for k in mfp.MODELS:
        assert [str(c) for c in out[f"{k}_columns"]] == cols[k]
        for r in range(3):
            ref = rows[k][r][idx]
            tol = 1e-9 * np.abs(ref[:, :mfp.n_collision_channels(k, cols[k])].sum(axis=1))[:, None]
            assert np.all(np.abs(out[f"{k}_relic{r + 1}"] - ref) <= tol)


@pytest.mark.skipif(not os.path.exists(mfp.CACHE), reason="no cache")
def test_figure_draws():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = mfp.figure()
    assert len(fig.axes) == 16
    lines = fig.axes[0].get_lines()
    assert any(np.allclose(ln.get_ydata(), mfp.HUBBLE_MPC) for ln in lines)
    plt.close(fig)
