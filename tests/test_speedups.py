"""
Speed-ups that change no number beyond rounding (2026-10-18): the Galactic emission averaged onto the coarse map from
sums kept per density pattern (dp_galactic.emission_structure), the zenith Filon weights compiled
(dp_spatial._filon_intervals_nb), the MeV transport's profile tables kept per relic (dp_si._profile_tables). Each
against the direct path it replaces.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def test_galactic_emission_regrouped_equals_direct():
    """dark_portal_solver._galactic_emission_populations with the regrouped sums equals the direct path (emission per
    full-resolution pixel, then the weighted average) to rounding, NaN pattern included, at a point of a campaign
    plane (two calls: the structure built, then reused)."""
    import dark_portal_solver as dps
    from analysis import config as config_mod
    from analysis.exact_profile import exact_config
    path = os.path.join(REPO, "analysis", "configs", "campaign_exact", "now_ic26", "dp_mu_pot_bal_pos.json")
    cfg = config_mod.from_dict(exact_config(json.load(open(path))), path)
    m = cfg.model()
    v = {p.name: p.start for p in cfg.parameters}
    v.update(m_Zd_eV=1e-22, g_D_eps_trace=7e-40)
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    setup = dps._flux_setup(E, params)
    nside = int(params["sky"]["nside"])
    flag = dps.GALACTIC_EMISSION_REGROUPED
    try:
        dps.GALACTIC_EMISSION_REGROUPED = False
        ref = dps._galactic_emission_populations(E, params, setup, nside)
        dps.GALACTIC_EMISSION_REGROUPED = True
        for _ in range(2):
            new = dps._galactic_emission_populations(E, params, setup, nside)
            for a, b in zip(ref, new):
                assert a.shape == b.shape
                assert np.array_equal(np.isnan(a), np.isnan(b))
                ok = np.isfinite(a)
                assert np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1e-300)) < 1e-12
    finally:
        dps.GALACTIC_EMISSION_REGROUPED = flag


def test_zenith_filon_weights_compiled_equal_reference():
    import dp_spatial as s
    rng = np.random.default_rng(3)
    nodes = np.linspace(-1.0, 1.0, 25)
    c_fine, idx = s._earth_zenith_fine_grid(nodes, 1.5, 400)
    for scale in (0.0, 1e-3, 0.3, 30.0, 3000.0):
        ph = scale * np.cumsum(rng.random(len(c_fine)))
        for new, ref in ((s._zenith_filon_weights_intervals, s._zenith_filon_weights_intervals_ref),
                         (s._zenith_filon_weights, s._zenith_filon_weights_ref)):
            a, b = ref(c_fine, idx, ph), new(c_fine, idx, ph)
            assert np.max(np.abs(a - b)) <= 1e-13 * np.max(np.abs(a))


def test_profile_tables_per_relic_identical():
    """The MeV transport's tables kept per relic: a set whose relics were built for other sets is bit for bit the set
    built from scratch."""
    import dp_si
    args = dict(prof=1, M=3e5, E0=1e13, N=40, dlnE=0.05, q=1, K=3, top_full=False, L_top_mid=0.01, n_u=8)

    def build(masses):
        return dp_si._profile_tables(args["prof"], args["M"], masses, args["E0"], args["N"], args["dlnE"], args["q"],
                                     args["K"], args["top_full"], args["L_top_mid"], args["n_u"])
    dp_si._TABLE_CACHE.clear()
    dp_si._RELIC_TABLES.clear()
    fresh = build((0.01, 0.0131, 0.051))
    dp_si._TABLE_CACHE.clear()
    dp_si._RELIC_TABLES.clear()
    build((0.01, 0.02, 0.051))           # relics 1 and 3 kept from here
    build((0.03, 0.0131, 0.06))          # relic 2 from here
    dp_si._TABLE_CACHE.clear()
    again = build((0.01, 0.0131, 0.051))
    for x, y in zip(fresh, again):
        if x is None:
            assert y is None
            continue
        for a, b in zip(x, y):
            assert np.array_equal(a, b)


def test_poisson_compiled_equals_numpy():
    """analysis.events._poisson (compiled) against its NumPy original: values and gradients equal to rounding, in every
    branch (bins with data above and below MU_FLOOR n, empty bins above and below zero)."""
    from analysis import events
    rng = np.random.default_rng(5)
    for size in (1, 7, 300):
        n = rng.poisson(3.0, size).astype(float)
        n[rng.random(size) < 0.3] = 0.0
        mu = rng.normal(3.0, 3.0, size)
        mu[rng.random(size) < 0.2] *= 1e-5
        a, ga = events._poisson_ref(mu, n, grad=True)
        b, gb = events._poisson(mu, n, grad=True)
        assert abs(a - b) <= 1e-12 * max(1.0, abs(a))
        assert np.allclose(ga, gb, rtol=1e-12, atol=1e-12)
        assert abs(events._poisson(mu, n) - a) <= 1e-12 * max(1.0, abs(a))
