"""
Tests of the Galactic-plane component (docs/dev/DIRECTIONS_PLAN.md, step 5; tests T6 and T8).

T6: the reduced templates (src/data, tools/prepare_galactic.py) times IceCube's 2026 normalizations
reproduce the inner-Galaxy spectra of arXiv:2607.25966 Fig. 4 (read from the SVG's path coordinates,
src/data/ic26_fig4_digitized.json) -- which fixes the conventions: per flavor, nu + nubar, and CRINGE as
diffuse plus unresolved sources; the pi0 template's sky-integrated normalization and E^-2.7 law; the maps'
frame (the plane at b = 0 after rotation to equatorial pixels).
T8: under secret interactions the component is the averaged oscillation of (x, 1 - x, 0), unattenuated.
The dark portal's local emission: flavor fractions decohere into the local eigenstates
(dp_transit.flavor_populations against a direct eigendecomposition), the vacuum limit is the averaged
oscillation, and the transfer conserves the flux and is linear in the composition.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dark_portal_solver as dps  # noqa: E402
import dp_galactic as G  # noqa: E402
import dp_sky  # noqa: E402


def _fig4():
    with open(os.path.join(REPO, "src", "data", "ic26_fig4_digitized.json")) as f:
        return json.load(f)


def _inner_mean_E2(name, E_GeV, nside=16):
    grid = dp_sky.SkyGrid(nside)
    return G.intensity(name, E_GeV, nside)[G.inner_galaxy(grid)].mean(axis=0) * E_GeV ** 2


def test_templates_reproduce_icecube_2026_fig4():
    """T6: CRINGE (diffuse + unresolved) x 0.91 and pi0 x 4.99, averaged over |b| < 15 deg, |l| < 20 deg,
    lie on IceCube's best-fit curves to 1% over their whole range (measured: 0.25% and 0.47%); the diffuse
    map alone falls short by a factor 2-4, so IceCube's "CRINGE" includes the unresolved sources."""
    fig = _fig4()
    for name, curve, norm in (("cringe", "cringe", 0.91), ("pi0", "pi0", 4.99)):
        c = np.array(fig["curves_E_GeV_E2Phi"][curve])
        r = norm * _inner_mean_E2(name, c[:, 0]) / c[:, 1]
        assert np.max(np.abs(r - 1)) < 0.01, (name, r.min(), r.max())
        assert norm == G.NORM_DEFAULT[name]
    c = np.array(fig["curves_E_GeV_E2Phi"]["cringe"])
    r = 0.91 * _inner_mean_E2("cringe_diffuse", c[:, 0]) / c[:, 1]
    assert r.max() < 0.6 and r.min() > 0.2


def test_pi0_normalization_and_spectrum():
    """T6: the pi0 template's sky-integrated per-flavor E^2 Phi(100 TeV) is the model's 4.4e-12 TeV cm^-2
    s^-1 (float32 storage), and the interpolation is exact for its E^-2.7 law between the nodes."""
    E = np.array([1e5, 3.3e4, 2.71e6])
    I = G.intensity("pi0", E, 16)
    sky = I.sum(axis=0) * 4 * np.pi / I.shape[0] * E ** 2 / 1e3          # TeV cm^-2 s^-1
    assert abs(sky[0] / 4.4e-12 - 1) < 1e-6
    assert np.allclose(sky / sky[0], (E / 1e5) ** -0.7, rtol=1e-6)


def test_templates_frame_and_content():
    """T6: in the equatorial pixels the plane lies at b = 0 (|b| < 5 deg, 8.7% of the sky, holds most of the
    flux) and the Galactic centre's pixel outshines the anticentre's; all intensities are non-negative;
    the stored energies cover the solver's range."""
    grid = dp_sky.SkyGrid(16)

    def pixel(l, b):
        ra, dec = dp_sky.galactic_to_equatorial(l, b)
        return int(dp_sky.ang2pix_ring(16, 0.5 * np.pi - dec, ra))
    gc, ac = pixel(0.0, 0.0), pixel(np.pi, 0.0)
    for name in G.TEMPLATES:
        E, I, meta = G.load_template(name, 16)
        assert I.shape == (grid.npix, len(E)) and np.all(I >= 0) and E[0] < 1e3 and E[-1] >= 1e8 * 0.999
        k = int(np.argmin(np.abs(E - 1e5)))
        assert I[np.abs(grid.b) < np.radians(5), k].sum() / I[:, k].sum() > 0.4
        assert I[gc, k] > 10 * I[ac, k]


def test_intensity_interpolation_and_degrade():
    """Exact at the nodes (to exp(log) rounding); out-of-range energies raise; N_side 8 averages N_side 16's children exactly
    (the all-sky mean is unchanged)."""
    E, I, _ = G.load_template("cringe", 16)
    assert np.allclose(G.intensity("cringe", E[3:9], 16), I[:, 3:9], rtol=1e-14, atol=0)
    with pytest.raises(ValueError):
        G.intensity("cringe", np.array([10.0]), 16)
    I8 = G.intensity("cringe", E[:5], 8)
    assert I8.shape == (768, 5) and np.allclose(I8.mean(axis=0), I[:, :5].mean(axis=0), rtol=1e-12)
    with pytest.raises(ValueError):
        G.load_template("cringe", 32)


def test_source_composition_and_averaged_transfer():
    """f_S = (x, 1 - x, 0); the averaged-oscillation matrix is symmetric and doubly stochastic."""
    U = dps.construct_PMNS(np.radians(33.4), np.radians(8.6), np.radians(49.0), np.radians(197.0))
    P = G.averaged_transfer(U)
    assert np.allclose(P, P.T, atol=1e-15) and np.allclose(P.sum(0), 1, atol=1e-14) and np.allclose(P.sum(1), 1, atol=1e-14)
    assert np.array_equal(G.source_composition(0.25), [0.25, 0.75, 0.0])
    for bad in (-0.1, 1.2):
        with pytest.raises(ValueError):
            G.source_composition(bad)


def _cfg(**over):
    cfg = {"grid": {"n_E_bins": 8, "n_z_bins": 40, "n_sun_paths": 2, "N_steps_earth": 8, "E_min_eV": 1e13,
                    "E_max_eV": 1e16},
           "execution": {"n_cores": 1, "verbose": 0}, "sky": {"output": "healpix", "nside": 8},
           "galactic": {"enabled": True, "emission": "local", "f_e_S_gal": 0.3},
           # the decay into the mediator off (2026-10-24): these check the Galaxy's paths (tests/test_decay.py has it)
           "dark_portal": {"neutrino_decay": False}}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


def _run(cfg):
    E, params = dps.build_solver_params(cfg)
    params["show_progress"] = False
    return E, params, dps.propagate_sky(E, params)


def _check_outputs(E, params, out, norm):
    """Shapes; flux conservation (every kind's flavors sum to its emitted intensity); band means and the
    all-sky mean are pixel means of the maps; the meta records the physics."""
    nb, npix = len(out["dec_band_edges"]) - 1, 12 * 8 ** 2
    em = G.emitted("cringe", E / 1e9, 8, norm)
    assert out["galactic_emitted"].shape == (npix, len(E)) and np.allclose(out["galactic_emitted"], em, rtol=1e-14)
    for kind in ("nu", "nubar"):
        T = out["galactic_transfer_" + kind]
        assert T.shape == (nb, len(E), 3) and np.max(np.abs(T.sum(-1) - 1)) < 1e-12
        maps = G.pixel_flux(out, kind)
        assert np.allclose(maps.mean(axis=0), out["galactic_allsky_" + kind], rtol=1e-13, atol=0)
        assert np.allclose(G.band_means(maps, out["pixel_band"], nb), out["galactic_bands_" + kind], rtol=1e-13, atol=0)
    tot = (out["galactic_allsky_nu"] + out["galactic_allsky_nubar"]).sum(-1)
    assert np.allclose(tot, 3 * norm * G.intensity("cringe", E / 1e9, 8).mean(axis=0), rtol=1e-13)
    assert np.allclose(out["meta"]["galactic"]["f_S"], [0.3, 0.7, 0.0], rtol=0, atol=1e-16)
    assert out["meta"]["galactic"]["norm_gal"] == norm


def test_galactic_secret_interactions_unattenuated():
    """T8: under secret interactions the Galactic flux is the averaged oscillation of (x, 1 - x, 0) in every
    band, for neutrinos and antineutrinos, whatever the coupling; the emission setting does not apply."""
    cfg = _cfg(secret_interactions={"enabled": True, "texture": "tau", "m_phi_eV": 5e6, "g_phi": 0.05},
               galactic={"emission": "3d"})
    E, params, out = _run(cfg)
    U = dps.construct_PMNS(params["th12"], params["th13"], params["th23"], params["d_cp"])
    T = G.averaged_transfer(U) @ np.array([0.3, 0.7, 0.0])
    for kind in ("nu", "nubar"):
        assert np.max(np.abs(out["galactic_transfer_" + kind] - T)) < 1e-15
    _check_outputs(E, params, out, 0.91)


def test_galactic_dark_portal_vacuum_limit_and_conservation():
    """T8: without the dark potential (epsilon' = 0) the local emission gives the averaged oscillation in
    every band, to rounding when the Earth's standard matter potential is off. With it on, the bands through
    the Earth move by the MSW effect on the arriving mass-state mixture, first order in
    (Delta m^2 / 2E) / V_CC: 2.4e-5 at 10 TeV through the core, falling as 1/E; the down-going bands, which
    cross only the ice, stay below 1e-9. The outputs are consistent."""
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-20, "g_D": 1e-60, "epsilon_prime": 0.0}
    for sm in (False, True):
        E, params, out = _run(_cfg(dark_portal=dp, standard_mixing={"use_SM_potential": sm}))
        U = dps.construct_PMNS(params["th12"], params["th13"], params["th23"], params["d_cp"])
        T = G.averaged_transfer(U) @ np.array([0.3, 0.7, 0.0])
        for kind in ("nu", "nubar"):
            dev = np.max(np.abs(out["galactic_transfer_" + kind] - T), axis=2)      # (band, E), dec order
            if not sm:
                assert dev.max() < 1e-12
                continue
            assert 1e-5 < dev.max() < 5e-5 and dev[:9].max() < 1e-9
            assert np.allclose(dev[-1] * E, dev[-1, 0] * E[0], rtol=0.3)         # 1/E: first order in Delta m^2 / 2E
    _check_outputs(E, params, out, 0.91)


def test_galactic_dark_portal_linear_in_composition():
    """The transfer is linear in x (one evaluation at x = 0 and one at x = 1 give every x, Section 3.4), at a
    point where the Sun's potential reshapes the flavors (m_Zd = 1e-20 eV, kappa = 1)."""
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-20, "g_D": 1.66e-54 / 0.05 * 1e19, "epsilon_prime": 1e-20, "use_debye_screening": False}
    T = {}
    for x in (0.0, 1.0, 0.3):
        _, params, out = _run(_cfg(dark_portal=dp, galactic={"f_e_S_gal": x}))
        T[x] = out["galactic_transfer_nu"]
    assert np.max(np.abs(T[0.3] - (0.7 * T[0.0] + 0.3 * T[1.0]))) < 1e-13
    U = dps.construct_PMNS(params["th12"], params["th13"], params["th23"], params["d_cp"])
    assert np.max(np.abs(T[1.0][:, -1] - G.averaged_transfer(U)[:, 0])) > 1e-3     # the potential acts


def test_flavor_populations_against_eigendecomposition():
    """dp_transit.flavor_populations: the populations of the local eigenstates (ascending eigenvalues of
    M_rot + sigma S) of a flavor mixture, against numpy's eigh at moderate sigma, where it is accurate."""
    import dp_transit
    U = dps.construct_PMNS(np.radians(33.4), np.radians(8.6), np.radians(49.0), np.radians(197.0))
    M_rot = U @ np.diag([0.0, 7.4e-5, 2.5e-3]).astype(complex) @ U.conj().T
    S = np.outer([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]).astype(complex)
    tab = dp_transit.frame_table(M_rot, S, 0.002)
    E = np.geomspace(1e13, 1e16, 7)
    coef2, n_bg = 2e-3 / 1e14, 1.0                                  # sigma = E coef2 n_bg: 2e-4 .. 0.2 eV^2
    F = np.tile([0.2, 0.5, 0.3], (len(E), 1))
    out = np.empty_like(F)
    dp_transit.flavor_populations(E, F, coef2, n_bg, tab["Q"], tab["Mp"], tab["s"], tab["floor"], out)
    for i, e in enumerate(E):
        _, W = np.linalg.eigh(M_rot + e * coef2 * n_bg * S)
        assert np.allclose(out[i], (np.abs(W) ** 2).T @ F[i], atol=1e-10)
    assert np.allclose(out.sum(1), 1.0, atol=1e-14)


def test_galactic_config():
    """norm_gal None takes the template's IceCube fit; templates and emission settings are validated."""
    for name, n in (("cringe", 0.91), ("pi0", 4.99), ("cringe_diffuse", 1.0)):
        _, params = dps.build_solver_params({"galactic": {"template": name}, "execution": {"verbose": 0}})
        assert params["galactic"]["norm_gal"] == n
    _, params = dps.build_solver_params({"galactic": {"template": "pi0", "norm_gal": 3.0}, "execution": {"verbose": 0}})
    assert params["galactic"]["norm_gal"] == 3.0
    for bad in ({"template": "kra"}, {"norm_gal": -1.0}, {"emission": "volume"}):
        with pytest.raises(ValueError):
            dps.build_solver_params({"galactic": bad, "execution": {"verbose": 0}})


# =============================================================================
# Step 6: emission points in 3-D (tests T7 and the emission model)
# =============================================================================

def test_neff_table_against_the_suns_integral():
    """T7: the table's n_eff at the Sun's node equals dp_transit.galactic_effective_density (the same integral
    on the same distance nodes) at its masses, and between them to the log interpolation's 1e-3; above 1e-12 eV
    it is the model's local density at any point, and at short ranges it follows it between the nodes. Far
    beyond the Galaxy's size it is not uniform (the Galaxy's 'Coulomb' potential, deeper toward the Centre,
    exceeds the cosmic floor down to ~4e-31 eV); beyond the Hubble radius it is uniform to the Galaxy's share
    of the potential, ~1e-4."""
    import dp_transit
    from dp_core import cm_inv3_to_eV3
    from dp_spatial import calc_MW_neutron_density
    t = np.load(os.path.join(REPO, "src", "data", "mw_neff_tables.npz"))
    n0 = float(t["n_n0_cm3"]) * cm_inv3_to_eV3
    R_H = 1.0 / 1.4e-33
    for lm in list(t["log10_m_eV"][::24]) + [-27.97, -25.53, -21.01, -14.4]:
        m = 10.0 ** lm
        mine = G.effective_density(np.array([G.R_SUN_KPC]), np.array([0.0]), m, n0, R_H)[0]
        ref = dp_transit.galactic_effective_density(m, n0, R_H)
        on_node = np.min(np.abs(t["log10_m_eV"] - lm)) < 1e-9
        assert abs(mine / ref - 1) < (1e-9 if on_node else 1e-3), (lm, mine / ref - 1)
    R, z = np.array([0.0, 3.2, 8.33, 14.0]), np.array([0.01, 0.37, 0.0, 2.5])
    loc = G.effective_density(R, z, 1e-9, n0, R_H) / cm_inv3_to_eV3
    for i in range(len(R)):
        u = 8.33 - R[i]
        r = np.hypot(u, z[i])
        assert abs(loc[i] / calc_MW_neutron_density(r, np.arctan2(abs(u), z[i]), 0.0 if u >= 0 else np.pi) - 1) < 1e-12
    v = G.effective_density(R, z, 1e-20, n0, R_H) / cm_inv3_to_eV3
    assert np.all(np.abs(v / loc - 1) < 1e-4)
    v = G.effective_density(R, z, 1e-29, n0, R_H)
    assert v[0] > 3 * v[2] > 3 * v[3]                     # the Galactic Centre's potential is deeper
    for m in (1e-34, 1e-36):
        v = G.effective_density(R, z, m, n0, R_H)
        assert np.ptp(v) / v[2] < 1e-3
        assert abs(v[2] / dp_transit.galactic_effective_density(m, n0, R_H) - 1) < 1e-9


def test_emission_model_reproduces_cringe_morphology():
    """The emission distances' models, integrated along each line of sight, reproduce the morphology of the
    maps they are used for: the gas (Soding et al. 2025) x CRINGE's cosmic rays vs CRINGE's diffuse map
    (correlation of the logarithms over the sky 0.991), the Lorimer sources weighted by distance vs its
    unresolved map (0.985). Emission is distant toward the Galactic Centre and near toward the anticentre."""
    gas, src = G.emission_line_integrals(16)
    I = G.intensity("cringe_diffuse", np.array([1e5]), 16)[:, 0]
    U = G.intensity("cringe", np.array([1e5]), 16)[:, 0] - I
    ok = U > 0
    assert np.corrcoef(np.log(gas), np.log(I))[0, 1] > 0.985
    assert np.corrcoef(np.log(src[ok]), np.log(U[ok]))[0, 1] > 0.98
    q_gas, q_src = G.emission_weights(16)
    s = G.emission_geometry(16)[0]
    assert np.allclose(q_gas.sum(1), 1) and np.allclose(q_src.sum(1), 1)

    def pixel(l, b):
        ra, dec = dp_sky.galactic_to_equatorial(l, b)
        return int(dp_sky.ang2pix_ring(16, 0.5 * np.pi - dec, ra))
    gc, ac = pixel(0.0, 0.0), pixel(np.pi, 0.0)
    assert 4.0 < (q_gas[gc] * s).sum() < 8.0 and (q_gas[ac] * s).sum() < 2.0
    assert 8.0 < (q_src[gc] * s).sum() < 12.0 and (q_src[ac] * s).sum() < 5.0


def test_emission_populations_on_nodes_and_mixing():
    """dp_galactic.emission_populations: emission at one density gives that density's populations (exactly on
    a node, to second order in the node spacing between nodes); a mixture of two densities is the weighted
    mixture of their populations; the diffuse/sources split mixes the two weights linearly."""
    import dp_transit
    U = dps.construct_PMNS(np.radians(33.4), np.radians(8.6), np.radians(49.0), np.radians(197.0))
    M_rot = U @ np.diag([0.0, 7.4e-5, 2.5e-3]).astype(complex) @ U.conj().T
    S = np.outer([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]).astype(complex)
    tab = dp_transit.frame_table(M_rot, S, 0.002)
    E = np.geomspace(1e13, 1e16, 9)
    f = np.array([0.3, 0.7, 0.0])
    coef2 = 2e-3 / 1e14

    def direct(n):
        out = np.empty((len(E), 3))
        dp_transit.flavor_populations(E, np.ascontiguousarray(np.tile(f, (len(E), 1))), coef2, n, tab["Q"], tab["Mp"],
                                      tab["s"], tab["floor"], out)
        return out
    n_node = 10.0 ** (7 / G.NEFF_NODES_PER_DECADE)
    q = np.full((2, 4), 0.25)
    F = G.emission_populations(E, f, np.ones((2, len(E))), q, q, np.full((2, 4), n_node), coef2, tab)
    assert np.allclose(F, direct(n_node)[None], atol=1e-12)
    F = G.emission_populations(E, f, np.ones((1, len(E))), q[:1], q[:1], np.array([[1.0, 1.0, 30.0, 30.0]]), coef2, tab)
    ref = 0.5 * (direct(1.0) + direct(30.0))
    assert np.max(np.abs(F[0] - ref)) < 2e-3
    qa, qb = np.array([[1.0, 0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0, 1.0]])
    n = np.array([[1.0, 1.0, 1.0, 1e3]])
    w = np.linspace(0, 1, len(E))[None]
    F = G.emission_populations(E, f, w, qa, qb, n, coef2, tab)
    Fa = G.emission_populations(E, f, np.ones_like(w), qa, qb, n, coef2, tab)
    Fb = G.emission_populations(E, f, np.zeros_like(w), qa, qb, n, coef2, tab)
    assert np.allclose(F, w[:, :, None] * Fa + (1 - w[:, :, None]) * Fb, atol=1e-14)
    assert np.allclose(F.sum(-1), 1.0, atol=1e-13)


def _cfg3d(**over):
    cfg = _cfg(galactic={"emission": "3d"}, sky={"nside": 8})
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


def test_galactic_3d_long_range_is_local_and_outputs_consistent():
    """Emission points in 3-D: for ranges beyond the Hubble radius (m_Zd = 1e-36 eV) the cosmic mean dominates the
    potential everywhere in the Galaxy (the Galaxy's share ~1e-4), so the pixel maps equal those of local
    emission to that share's effect; the outputs are consistent (flux conserved per kind, band and all-sky means
    of the pixel maps). The coupling puts the cosmic potential near the vacuum term in the band (the Sun's
    potential at the Earth is 2e-6 of Delta m^2_31 / 2E at 100 TeV)."""
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-36, "g_D": 2e-6 * 1.66e-54 / 0.05 * 1e19, "epsilon_prime": 1e-20, "use_debye_screening": False}
    E, params, out3 = _run(_cfg3d(dark_portal=dp))
    _, _, outl = _run(_cfg3d(dark_portal=dp, galactic={"emission": "local"}))
    for kind in ("nu", "nubar"):
        a, b = G.pixel_flux(out3, kind), G.pixel_flux(outl, kind)
        assert np.max(np.abs(a / b - 1)) < 1e-3
        em = out3["galactic_emitted"]
        assert np.max(np.abs(a.sum(-1) / em - 1)) < 1e-12
        nb = len(out3["dec_band_edges"]) - 1
        assert np.allclose(G.band_means(a, out3["pixel_band"], nb), out3["galactic_bands_" + kind], rtol=1e-12, atol=0)
        assert np.allclose(a.mean(axis=0), out3["galactic_allsky_" + kind], rtol=1e-12, atol=0)
    assert out3["meta"]["galactic"]["emission"] == "3d" and out3["meta"]["galactic"]["emission_model"]["B"] == 1.9


def test_galactic_3d_direction_dependence():
    """Emission points in 3-D at a short range (m_Zd = 1e-24 eV, range 6 pc: the potential at emission is close to
    the local density there, ~10-700 cm^-3 in the inner Galaxy against ~1 near the Sun, and the Sun's own at the
    Earth is comparable, ~3 cm^-3; kappa = 1 puts it at the vacuum term at 100 TeV): the Galactic flavors then
    differ between the Galactic Centre's direction and the anticentre's, which local emission cannot produce."""
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 1.66e-54 / 0.05 * 1e19, "epsilon_prime": 1e-20, "use_debye_screening": False}
    E, params, out3 = _run(_cfg3d(dark_portal=dp))
    _, _, outl = _run(_cfg3d(dark_portal=dp, galactic={"emission": "local"}))

    def pixel(l, b):
        ra, dec = dp_sky.galactic_to_equatorial(l, b)
        return int(dp_sky.ang2pix_ring(8, 0.5 * np.pi - dec, ra))
    gc, ac = pixel(0.0, 0.0), pixel(np.pi, 0.0)
    f3 = G.pixel_flux(out3, "nu") / G.pixel_flux(out3, "nu").sum(-1, keepdims=True)
    fl = G.pixel_flux(outl, "nu") / G.pixel_flux(outl, "nu").sum(-1, keepdims=True)
    band = out3["pixel_band"]
    # local emission: two pixels in the same band have the same flavors; 3-D emission: GC and anticentre differ
    assert np.max(np.abs(f3[gc] - f3[ac])) > 1e-3
    same_band = np.nonzero(band == band[gc])[0]
    assert np.max(np.abs(fl[same_band] - fl[gc])) < 1e-12


def test_galactic_source_at_the_sun_is_the_host_galaxy_path():
    """T10: a Galactic source at the Sun's position (emission 'local') and a source in a Milky-Way-like host at a
    Sun-like position (source.host_galaxy_scale = 1) at z -> 0 are born into the same potential, so their flavors
    at the detector agree in every band and energy. The two potentials differ by the cosmic mean, which the host
    path adds on top of the Galaxy (3e-8 of the local density here; up to 1.2e-3 at 1e-31-1e-29 eV, where the
    Galaxy's halo meets the cosmic mean and the local potential takes the larger of the two)."""
    for m, kappa in ((1e-24, 1.0), (1e-27, 30.0)):
        dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": m, "g_D": kappa * 1.66e-54 / 0.05 * 1e19,
              "epsilon_prime": 1e-20, "use_debye_screening": False}
        _, _, host = _run(_cfg(dark_portal=dp, galactic={"enabled": False},
                               source={"z_max": 1e-4, "host_galaxy_scale": 1.0, "f_e_S": 0.3, "f_mu_S": 0.7,
                                       "f_tau_S": 0.0}))
        _, _, gal = _run(_cfg(dark_portal=dp, galactic={"emission": "local", "f_e_S_gal": 0.3}))
        for kind in ("nu", "nubar"):
            fh = host["bands_" + kind] / host["bands_" + kind].sum(-1, keepdims=True)
            fg = gal["galactic_transfer_" + kind]
            assert np.max(np.abs(fh - fg)) < 1e-5, (m, kind, np.max(np.abs(fh - fg)))
        _, params, vac = _run(_cfg(dark_portal=dict(dp, epsilon_prime=0.0), galactic={"emission": "local",
                                                                                 "f_e_S_gal": 0.3}))
        assert np.max(np.abs(gal["galactic_transfer_nu"] - vac["galactic_transfer_nu"])) > 1e-2   # the potential acts


def test_galactic_at_coarse_resolution():
    """HESE-like resolution (Mauricio, 2026-10-14: cascades can use coarser pixels than tracks): N_side 4 with MESE's
    ten bands of 0.2 in sin(dec). The degraded template keeps the sky total (mean of the N_side 16 children); the
    component with emission points in 3-D conserves the flux, and its all-sky flavors agree with N_side 8's to
    < 1e-3 (the coarse pixels' band assignment): the populations are computed on the stored N_side 16 pixels,
    each along its own line of sight, and averaged with their intensities (a coarse pixel's centre can lie
    degrees off the plane its gas column contains; computed at the centre, the difference was 6e-3)."""
    edges = list(np.degrees(np.arcsin(np.linspace(-1, 1, 11))))
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 300 * 1.66e-54 / 0.05 * 1e19, "epsilon_prime": 1e-20, "use_debye_screening": False}
    outs = {}
    for ns in (4, 8):
        E, params, out = _run(_cfg3d(dark_portal=dp, sky={"nside": ns, "dec_band_edges_deg": edges}))
        a = G.pixel_flux(out, "nu")
        assert a.shape[0] == 12 * ns * ns and np.max(np.abs(a.sum(-1) / out["galactic_emitted"] - 1)) < 1e-12
        outs[ns] = out
        I16 = G.intensity("cringe", E / 1e9, 16)
        assert np.allclose(G.intensity("cringe", E / 1e9, ns).mean(0), I16.mean(0), rtol=1e-12)
    f = {ns: o["galactic_allsky_nu"] / o["galactic_allsky_nu"].sum(-1, keepdims=True) for ns, o in outs.items()}
    assert np.max(np.abs(f[4] - f[8])) < 1e-3


def test_extragalactic_physical_normalization():
    """sky.extragalactic_norm: without new physics the extragalactic per-flavor nu + nubar intensity is
    norm x (E / 100 TeV)^-p at every energy, to the redshift grid's first-order error (3.6e-3 at 400 steps, the
    same at every energy); None keeps the solver's units; the factor scales the flux with new physics too."""
    base = {"dark_portal": {"u_e": 0.0, "u_mu": 0.0, "u_tau": 0.0, "g_D": 1e-30, "epsilon_prime": 0.0},
            "grid": {"n_E_bins": 12, "n_z_bins": 400, "n_sun_paths": 2, "E_min_eV": 1e13, "E_max_eV": 1e16},
            "source": {"p_index": 2.87}, "execution": {"n_cores": 1, "verbose": 0}}
    E, params = dps.build_solver_params(dict(base, sky={"output": "bands", "extragalactic_norm": 2.12e-18}))
    params["show_progress"] = False
    out = dps.propagate_sky(E, params)
    per_flavor = (out["allsky_nu"] + out["allsky_nubar"]).sum(-1) / 3.0
    r = per_flavor / (2.12e-18 * (E / 1e14) ** -2.87)
    assert np.ptp(r) < 1e-10 and abs(r.mean() - 1) < 6e-3
    E, params = dps.build_solver_params(dict(base, sky={"output": "bands"}))
    params["show_progress"] = False
    raw = dps.propagate_sky(E, params)
    assert np.allclose(out["bands_nu"], raw["bands_nu"] * out["meta"]["extragalactic_scale"], rtol=1e-13, atol=0)
    with pytest.raises(ValueError):
        dps.build_solver_params({"sky": {"extragalactic_norm": -1.0}, "execution": {"verbose": 0}})


# =============================================================================
# The Galactic component in the flavor analysis (analysis.model; Mauricio, 2026-10-14)
# =============================================================================

def _model_cfg(**over):
    norm, E_ref = G.icecube_2026_anchor()
    cfg = {"grid": {"n_E_bins": 8, "n_z_bins": 40, "n_sun_paths": 2, "N_steps_earth": 8, "E_min_eV": 1e13,
                    "E_max_eV": 1e16},
           "sky": {"nside": 8, "extragalactic_norm": norm, "extragalactic_norm_E_GeV": E_ref},
           "galactic": {"enabled": True, "emission": "3d"}}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


def test_analysis_galactic_component_exact():
    """The flavor model with the Galactic component: norm_gal = 0 gives the extragalactic-only fractions; floating
    norm_gal, x_gal and norm_astro on the cached fluxes equals recomputing them from a config with those values;
    galactic_allsky equals propagate_sky's sky-averaged Galactic flux (nu + nubar)."""
    from analysis.model import FlavorModel
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 300 * 1.66e-54 / 0.05 * 1e19, "epsilon_prime": 1e-20, "use_debye_screening": False}
    m = FlavorModel(_model_cfg(dark_portal=dp), free_names=("norm_gal", "f_e_S_gal", "norm_astro"))
    eg = FlavorModel(_model_cfg(dark_portal=dp, galactic={"enabled": False}))
    assert np.allclose(m.fractions({"norm_gal": 0.0}), eg.fractions({}), rtol=0, atol=1e-14)
    vals = {"norm_gal": 1.3, "f_e_S_gal": 0.6, "norm_astro": 0.8}
    norm, E_ref = G.icecube_2026_anchor()
    direct = FlavorModel(_model_cfg(dark_portal=dp, galactic={"norm_gal": 1.3, "f_e_S_gal": 0.6},
                                    sky={"extragalactic_norm": 0.8 * norm}))
    assert np.allclose(m.fractions(vals), direct.fractions({}), rtol=0, atol=1e-13)
    assert m.n_solver_calls == 1 and abs(m.fractions(vals)[0] - m.fractions({"norm_gal": 0.0})[0]) > 1e-4
    E, params = dps.build_solver_params(dict(_model_cfg(dark_portal=dp), execution={"n_cores": 1, "verbose": 0},
                                             sky={"output": "healpix", "nside": 8, "extragalactic_norm": norm,
                                                  "extragalactic_norm_E_GeV": E_ref}))
    params["show_progress"] = False
    sky = dps.propagate_sky(E, params)
    G2 = dps.galactic_allsky(E, params, compositions=((1 / 3, 2 / 3, 0.0),))[0]
    params["galactic"]["f_e_S_gal"] = 1 / 3
    sky = dps.propagate_sky(E, params)
    assert np.allclose(G2, sky["galactic_allsky_nu"] + sky["galactic_allsky_nubar"], rtol=1e-12, atol=0)


def test_analysis_galactic_share_and_priors():
    """Secret interactions (unattenuated Galactic flux) and no new physics: with the extragalactic flux anchored
    to IceCube's 2026 astrophysical flux at 14.7 TeV, its per-flavor intensity there is the anchor's; the
    Galactic share of the energy-integrated flux is a few percent; a prior on norm_astro (not a solver key) reads
    its default 1; galactic.enabled without sky.extragalactic_norm is refused."""
    from analysis.model import FlavorModel
    from analysis.objective import Objective
    from analysis.parameters import GaussianPrior
    si = {"enabled": True, "texture": "tau", "m_phi_eV": 5e6, "g_phi": 1e-6}
    m = FlavorModel(_model_cfg(secret_interactions=si, grid={"E_min_eV": 1e13, "E_max_eV": 3e16, "n_E_bins": 40}))
    E, F = m.flux({"norm_gal": 0.0})
    norm, E_ref = G.icecube_2026_anchor()
    k = np.argmin(np.abs(E / 1e9 - E_ref))
    per_flavor = F[k].sum() / 3.0 * (E[k] / 1e9 / E_ref) ** m.base["source"]["p_index"]
    assert abs(per_flavor / norm - 1) < 0.02
    share = m.galactic_share({})
    assert 0.01 < share < 0.2
    obj = Objective(m, "pion_2050_case_5", priors=[GaussianPrior("norm_astro", 1.0, 0.26), GaussianPrior("norm_gal", 0.91, 0.18)])
    assert obj.prior_chi2({}) == 0.0 and abs(obj.prior_chi2({"norm_astro": 1.26}) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        FlavorModel(_model_cfg(sky={"extragalactic_norm": None}))


def test_emission_rejects_non_finite_gas(tmp_path):
    """2026-10-14: NaN gas columns (an N_side 64 reduction from native N_side 64 maps left pixels without a native
    centre) made emission_populations drop every density node they touched and zero most pixels, silently.
    Both the data check and the populations' own check must now raise."""
    import shutil
    src = os.path.join(REPO, "src", "data", "galactic_emission.npz")
    with np.load(src, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["gas_column_nside16"] = d["gas_column_nside16"].copy()
    d["gas_column_nside16"][100, :] = np.nan
    np.savez(tmp_path / "galactic_emission.npz", **d)
    for f in ("mw_neff_tables.npz",):
        shutil.copy(os.path.join(REPO, "src", "data", f), tmp_path / f)
    G.use_data_dir(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="non-finite gas columns"):
            G.emission_weights(16)
    finally:
        G.use_data_dir(None)
    q = np.full((4, 3), 1.0 / 3.0)
    q[1, 0] = np.nan
    n = np.tile(np.array([1e-20, 2e-20, 3e-20]), (4, 1))
    tab = None
    with pytest.raises(ValueError, match="non-finite"):
        G.emission_populations(np.array([1e14]), np.array([1 / 3, 2 / 3, 0.0]), np.ones((4, 1)), q, q, n, 1.0, tab)
