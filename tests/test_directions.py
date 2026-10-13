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
