"""
Tests of the per-direction output (docs/dev/DIRECTIONS_PLAN.md): the Earth's zenith average per
declination band (step 2). T1: the bands weighted by their widths return the whole-sky average on
the same chords. T2: each band's 'phase' average equals the exact integral over its range of the
potential-dominated evolution (dense trapezoid), and its decoherent limit. Through the solver's Earth
phase: the same partition, isotropy without the dark portal, and the argument checks.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dark_portal_solver as dps  # noqa: E402
import dp_sky  # noqa: E402
import dp_spatial  # noqa: E402

trapz = getattr(np, "trapezoid", None) or np.trapz          # NumPy 2 renamed it
S = np.outer([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]).astype(complex)


def _rho_in(seed):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    rho = B @ B.conj().T
    return rho / np.trace(rho).real


def _synthetic(nodes, alpha, rho_in):
    """Chord density matrices exp(-i Psi S) rho exp(+i Psi S), Psi = alpha * column (the evolution the
    'phase' rule assumes), the evolution itself and the projectors of S."""
    P, lam = dp_spatial._lri_texture_projectors(S)
    col = dp_spatial._earth_columns(nodes, 1.5, 0.0, dps.earth_yukawa_table(None))

    def evolve(psi):
        U = sum(np.exp(-1j * l * psi) * p for l, p in zip(lam, P))
        return U @ rho_in @ U.conj().T
    return np.array([evolve(alpha * c) for c in col])[:, None], evolve, P


BANDS = dp_sky.SkyGrid(16).costh_edges            # 20 bands, 0.1 wide in cos(zenith)


def test_earth_chord_nodes():
    nodes = dp_spatial.earth_chord_nodes(BANDS, 2)
    assert len(nodes) == 41 and np.allclose(nodes, np.linspace(-1, 1, 41), atol=1e-15)
    assert np.all(np.isin(BANDS, nodes))
    assert np.array_equal(dp_spatial.earth_chord_nodes([-1.0, 0.0, 1.0], 1), [-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        dp_spatial.earth_chord_nodes([0.0, -1.0], 2)
    with pytest.raises(ValueError):
        dp_spatial.earth_chord_nodes(BANDS, 0)


@pytest.mark.parametrize("psi_core", [0.1, 3.0, 52.0, 1e3, 1e8])
def test_bands_partition_the_sky(psi_core):
    """T1: sum_b (width_b / 2) <rho>_b equals the whole-sky 'phase' average on the same chords, to
    rounding, from the vacuum-like to the decoherent regime (the band weights are the whole range's
    Filon weights split by interval)."""
    nodes = dp_spatial.earth_chord_nodes(BANDS, 2)
    col0 = dp_spatial._earth_columns(np.array([-1.0]), 1.5, 0.0, dps.earth_yukawa_table(None))[0]
    alpha = psi_core / col0
    rho_k, _, _ = _synthetic(nodes, alpha, _rho_in(5))
    coef = alpha / dp_spatial._KM_IN_EV_INV
    yk = dps.earth_yukawa_table(None)
    whole = dps.average_earth_chords(rho_k, nodes, coef, S, 1.5, 0.0, 1.0, yk, "phase")
    bands = dp_spatial.average_earth_chords_bands(rho_k, nodes, BANDS, coef, S, 1.5, 0.0, 1.0, yk, "phase")
    recombined = np.tensordot(np.diff(BANDS) / 2.0, bands, axes=(0, 0))
    assert np.max(np.abs(recombined - whole)) < 1e-14
    # every band is a density matrix: Hermitian, unit trace
    assert np.max(np.abs(bands - np.conj(np.swapaxes(bands, -1, -2)))) < 1e-14
    assert np.max(np.abs(np.trace(bands, axis1=-2, axis2=-1) - 1.0)) < 1e-12


def test_bands_plain_is_the_trapezoid():
    """'plain' per band is the trapezoid inside the band; weighted by width, the trapezoid of the range."""
    nodes = dp_spatial.earth_chord_nodes(BANDS, 3)
    rho_k = _rho_in(6)[None, None] * (1.0 + nodes)[:, None, None, None]
    yk = dps.earth_yukawa_table(None)
    bands = dp_spatial.average_earth_chords_bands(rho_k, nodes, BANDS, 1.0, S, 1.5, 0.0, 1.0, yk, "plain")
    centres = 0.5 * (BANDS[1:] + BANDS[:-1])
    assert np.allclose(bands, _rho_in(6)[None, None] * (1.0 + centres)[:, None, None, None], atol=1e-14)


@pytest.mark.parametrize("psi_core", [0.1, 3.0, 52.0, 1e3])
def test_band_average_matches_continuum(psi_core):
    """T2: each band's average equals the exact integral over its own range of cos(zenith) (dense
    trapezoid, 20,000 directions per band), for Earth phases up to 1e3 rad through the core."""
    nodes = dp_spatial.earth_chord_nodes(BANDS, 2)
    c_dense = np.linspace(-1.0, 1.0, 400_001)
    col_dense = dp_spatial._earth_columns(c_dense, 1.5, 0.0, dps.earth_yukawa_table(None))
    alpha = psi_core / col_dense[0]
    rho_in = _rho_in(7)
    rho_k, evolve, _ = _synthetic(nodes, alpha, rho_in)
    yk = dps.earth_yukawa_table(None)
    bands = dp_spatial.average_earth_chords_bands(rho_k, nodes, BANDS, alpha / dp_spatial._KM_IN_EV_INV, S,
                                                  1.5, 0.0, 1.0, yk, "phase")
    dense = np.array([evolve(alpha * c) for c in col_dense])
    for b in range(len(BANDS) - 1):
        sel = (c_dense >= BANDS[b] - 1e-12) & (c_dense <= BANDS[b + 1] + 1e-12)
        ref = trapz(dense[sel], c_dense[sel], axis=0) / (BANDS[b + 1] - BANDS[b])
        assert np.max(np.abs(bands[b, 0] - ref)) < 1e-4, (b, np.max(np.abs(bands[b, 0] - ref)))


def test_band_average_at_huge_phases():
    """At 1e8 rad through the core the up-going bands and the near-horizon ones are decoherent,
    sum_g P_g rho P_g; straight down the chords cross only 1.5-15 km of ice, the band's phase spread is a
    few hundred radians, and the residual coherence (~1/spread) must match the dense continuum, which
    resolves it there (dense points closer than 0.1 rad in phase)."""
    nodes = dp_spatial.earth_chord_nodes(BANDS, 2)
    c_dense = np.linspace(-1.0, 1.0, 400_001)
    col_dense = dp_spatial._earth_columns(c_dense, 1.5, 0.0, dps.earth_yukawa_table(None))
    alpha = 1e8 / col_dense[0]
    rho_in = _rho_in(8)
    rho_k, evolve, P = _synthetic(nodes, alpha, rho_in)
    bands = dp_spatial.average_earth_chords_bands(rho_k, nodes, BANDS, alpha / dp_spatial._KM_IN_EV_INV, S,
                                                  1.5, 0.0, 1.0, dps.earth_yukawa_table(None), "phase")
    limit = sum(p @ rho_in @ p for p in P)
    psi = alpha * col_dense
    n_decoherent = n_resolved = 0
    for b in range(len(BANDS) - 1):
        sel = (c_dense >= BANDS[b] - 1e-12) & (c_dense <= BANDS[b + 1] + 1e-12)
        spread = psi[sel].max() - psi[sel].min()
        if np.max(np.abs(np.diff(psi[sel]))) < 0.1:
            dense = np.array([evolve(x) for x in psi[sel]])
            ref = trapz(dense, c_dense[sel], axis=0) / (BANDS[b + 1] - BANDS[b])
            assert np.max(np.abs(bands[b, 0] - ref)) < 1e-4, (b, spread)
            n_resolved += 1
        elif spread > 1e5:
            assert np.max(np.abs(bands[b, 0] - limit)) < 1e-4, (b, spread)
            n_decoherent += 1
    assert n_decoherent == 10 and n_resolved == 7      # bands 0-9 (up-going) and 13-19 (straight down)


def _earth_args(u_mag=0.5):
    E = np.geomspace(1e13, 1e15, 6)
    F = np.tile([0.2, 0.3, 0.5], (len(E), 1))
    U = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    M_rot = U @ np.diag([0.0, 7.41e-5, 2.511e-3]).astype(complex) @ U.conj().T
    u = np.array([u_mag, u_mag, 0.0], dtype=complex)
    return (E, F, M_rot, u, 2e6, 1e-4, 0.1, 1.676389e-4, True, np.ones(3), 1.5, False, 0.0, False, 1, 12)


def test_earth_phase_bands_through_the_solver():
    """propagate_earth_spatial with band edges: the whole-sky flux equals the width-weighted bands to
    rounding, and is close to the default chords' (another node set: the rule's discretization); with
    no dark portal (u = 0) every band equals the whole sky (isotropy)."""
    kw = dict(N_steps_earth=25, show_progress=False, verbose=0, earth_step_scheme="avg",
              earth_zenith_average="phase", earth_sky="full")
    F_all, F_b = dps.propagate_earth_spatial(*_earth_args(), earth_band_costh=BANDS, **kw)
    assert F_b.shape == (20, 6, 3)
    assert np.max(np.abs(np.tensordot(np.diff(BANDS) / 2, F_b, axes=(0, 0)) - F_all)) < 1e-14
    F_default = dps.propagate_earth_spatial(*_earth_args(), **kw)
    assert np.max(np.abs(F_all - F_default)) < 1e-4
    F0_all, F0_b = dps.propagate_earth_spatial(*_earth_args(0.0), earth_band_costh=BANDS, **kw)
    assert np.max(np.abs(F0_b - F0_all[None])) < 1e-12
    with pytest.raises(ValueError, match="earth_band_costh"):
        dps.propagate_earth_spatial(*_earth_args(), earth_band_costh=[-1.0, 0.0, 0.5], **kw)


# =============================================================================
# The solver's per-direction output (step 3): propagate_sky and the config sections
# =============================================================================

def _cfg(**over):
    cfg = {"grid": {"n_E_bins": 8, "n_z_bins": 40, "n_sun_paths": 2, "N_steps_earth": 8, "E_min_eV": 1e13,
                    "E_max_eV": 1e16},
           "execution": {"n_cores": 1, "verbose": 0}, "sky": {"output": "healpix", "nside": 4}}
    for k, v in over.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


def test_propagate_sky_dark_portal():
    """Bands weighted by solid angle return the whole-sky flux to rounding, for neutrinos and
    antineutrinos; the whole sky agrees with propagate_weighted's (another chord set) to the rule's
    discretization; the declination order puts the Earth's strongest effect (straight up-going, dec ~ +90)
    in the last band; the pixel map assigns every pixel its band."""
    E, params = dps.build_solver_params(_cfg())
    params["show_progress"] = False
    out = dps.propagate_sky(E, params)
    nb = len(out["dec_band_edges"]) - 1
    assert nb == 20 and out["bands_nu"].shape == (20, 8, 3) and out["bands_nubar"].shape == (20, 8, 3)
    w = out["band_omega"] / (4 * np.pi)
    for kind in ("nu", "nubar"):
        assert np.max(np.abs(np.tensordot(w, out[f"bands_{kind}"], axes=(0, 0)) - out[f"allsky_{kind}"])) \
            < 1e-14 * np.max(out[f"allsky_{kind}"])
    F_pw = dps.propagate_weighted(E, params)
    tot = out["allsky_nu"] + out["allsky_nubar"]
    assert np.max(np.abs(tot - F_pw)) < 1e-4 * np.max(F_pw)
    dev = np.max(np.abs(out["bands_nu"] - out["allsky_nu"][None]), axis=(1, 2))
    assert dev[0] < dev[-1]                        # dec -90 (straight down, ice) vs dec +90 (through the core)
    g = dp_sky.SkyGrid(4)
    assert np.array_equal(out["pixel_band"], g.band) and out["nside"] == 4


def test_propagate_sky_secret_interactions_isotropic():
    """Secret interactions: every band is the whole sky (isotropic transport)."""
    E, params = dps.build_solver_params(_cfg(secret_interactions={"enabled": True, "texture": "tau",
                                                                  "m_phi_eV": 5e6, "g_phi": 0.05}))
    out = dps.propagate_sky(E, params)
    for kind in ("nu", "nubar"):
        assert np.array_equal(out[f"bands_{kind}"], np.repeat(out[f"allsky_{kind}"][None], 20, axis=0))


def test_sky_and_galactic_config():
    """Defaults keep today's behaviour; values are validated; the dark portal's Galactic emission '3d' (the
    default) is step 6 (tests/test_galactic.py covers the component)."""
    E, params = dps.build_solver_params({"execution": {"verbose": 0}})
    assert params["sky"]["output"] == "allsky" and params["galactic"]["enabled"] is False
    assert params["galactic"]["template"] == "cringe" and abs(params["galactic"]["f_e_S_gal"] - 1 / 3) < 1e-5
    E, params = dps.build_solver_params({"sky": {"dec_band_edges_deg": [-90, 0, 90]}, "execution": {"verbose": 0}})
    assert np.allclose(params["sky"]["dec_band_edges"], [-np.pi / 2, 0, np.pi / 2])
    for bad in ({"sky": {"output": "maps"}}, {"sky": {"nside": 12}}, {"sky": {"sun_weighting": "yearly"}},
                {"sky": {"dec_band_edges_deg": [-90, 10, 5, 90]}}, {"galactic": {"template": "kra"}},
                {"galactic": {"f_e_S_gal": 1.5}}):
        with pytest.raises(ValueError):
            dps.build_solver_params(dict(bad, execution={"verbose": 0}))
    E, params = dps.build_solver_params(_cfg(galactic={"enabled": True}))
    with pytest.raises(NotImplementedError):
        dps.propagate_sky(E, params)


# =============================================================================
# The Sun per direction (step 4): single paths and yearly weights
# =============================================================================

def _sun_window_cfg(m_zd=1e-20, kappa=1.0):
    """A point where the Sun's potential at the Earth is kappa times Delta m^2_31 / (2 x 100 TeV):
    for ranges much longer than 1 AU it is g_D eps |u|^2 K0 N_sun / (4 pi d), independent of m."""
    gu2 = kappa * 1.66e-54
    return _cfg(dark_portal={"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": m_zd,
                             "g_D": gu2 / (0.1 * 0.5), "epsilon_prime": 0.1},
                grid={"n_sun_paths": 4})


def test_sun_path_weights():
    """Rows sum to 1; over the sky the yearly weights average to the Gauss-Legendre ones."""
    import dp_transit
    x, w = dp_transit.sun_paths(8)
    g = dp_sky.SkyGrid(16)
    A = dp_sky.sun_path_weights(g.elon, g.elat, x, n_times=360)
    assert np.allclose(A.sum(1), 1.0, atol=1e-12)
    assert np.max(np.abs(A.mean(0) - w)) < 2e-3


def test_sun_paths_recombine_to_the_isotropic_average():
    """The transit and the Earth are linear in the state: the single-path runs weighted by w_p equal
    the isotropic run, band by band, to rounding (a point where the Sun's potential matters)."""
    import dp_transit
    E, params = dps.build_solver_params(_sun_window_cfg())
    params["show_progress"] = False
    iso = dps.propagate_sky(E, params)
    x, w = dp_transit.sun_paths(params["n_sun_paths"])
    acc = {k: 0.0 for k in ("bands_nu", "bands_nubar")}
    per_path = []
    for xp, wp in zip(x, w):
        params["sun_path_cos"] = float(xp)
        o = dps.propagate_sky(E, params)
        per_path.append(o["allsky_nu"] + o["allsky_nubar"])
        for k in acc:
            acc[k] = acc[k] + wp * o[k]
    params.pop("sun_path_cos")
    for k in acc:
        assert np.max(np.abs(acc[k] - iso[k])) < 1e-12 * np.max(iso[k])
    spread = np.max(np.abs(np.array(per_path) - per_path[0][None])) / np.max(per_path[0])
    assert spread > 1e-6                                   # the Sun's direction matters here
