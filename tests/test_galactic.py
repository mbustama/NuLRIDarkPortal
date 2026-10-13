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
           "galactic": {"enabled": True, "emission": "local", "f_e_S_gal": 0.3}}
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
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-20, "g_D": 1.66e-54 / 0.05, "epsilon_prime": 0.1}
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


def test_galactic_config_and_emission_3d():
    """norm_gal None takes the template's IceCube fit; templates are validated; the dark portal's '3d'
    emission is step 6 (raises before any propagation)."""
    for name, n in (("cringe", 0.91), ("pi0", 4.99), ("cringe_diffuse", 1.0)):
        _, params = dps.build_solver_params({"galactic": {"template": name}, "execution": {"verbose": 0}})
        assert params["galactic"]["norm_gal"] == n
    _, params = dps.build_solver_params({"galactic": {"template": "pi0", "norm_gal": 3.0}, "execution": {"verbose": 0}})
    assert params["galactic"]["norm_gal"] == 3.0
    for bad in ({"template": "kra"}, {"norm_gal": -1.0}, {"emission": "volume"}):
        with pytest.raises(ValueError):
            dps.build_solver_params({"galactic": bad, "execution": {"verbose": 0}})
    E, params = dps.build_solver_params(_cfg(galactic={"emission": "3d"}))
    with pytest.raises(NotImplementedError):
        dps.propagate_sky(E, params)
