"""
Reuse of the Earth crossing within a parameter point (src/dp_spatial.py, 2026-10-18): the handover's responses to
the three unit inputs, combined, equal the direct handover to rounding -- whole sky and per band, neutrinos and
antineutrinos, with the potential on and off, eigenstate and flavor arrivals; in mode 'repeat' a parameter set's
first call takes the direct path; and a sky with its Galactic component is the same in modes 'off' (the solver's
default), 'repeat' and 'always' (the analysis').
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import dark_portal_solver as dps  # noqa: E402
import dp_sky  # noqa: E402
import dp_spatial  # noqa: E402

BANDS = dp_sky.SkyGrid(16).costh_edges
KW = dict(N_steps_earth=25, show_progress=False, verbose=0, earth_step_scheme="avg", earth_zenith_average="phase",
          earth_sky="full", earth_transfer="repeat")


def _args(u_mag, anti, m_Zd=2e6, g_D=1e-4):
    E = np.geomspace(1e13, 1e15, 6)
    U = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    if anti:
        U = U.conj()
    M_rot = U @ np.diag([0.0, 7.41e-5, 2.511e-3]).astype(complex) @ U.conj().T
    u = np.array([u_mag, u_mag, 0.0], dtype=complex)
    return (E, M_rot, u, m_Zd, g_D, 0.1, 1.676389e-4, True, np.ones(3), 1.5, False, 0.0, anti, 1, 12)


def _call(fn, F, args, **kw):
    E, rest = args[0], args[1:]
    kw = dict(KW, **kw)
    if fn is dp_spatial._propagate_earth_spatial_direct:
        kw.pop("earth_transfer")
    return fn(E, F, *rest, **kw)


def _close(a, b):
    """Equal to rounding: per energy bin, within 1e-13 of the bin's largest flux."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.max(np.abs(b), axis=-1, keepdims=True)
    return np.all(np.abs(a - b) <= 1e-13 * np.maximum(scale, 1e-300))


@pytest.mark.parametrize("bands", [None, "bands"])
@pytest.mark.parametrize("arrival", ["eigen", "flavor"])
@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("u_mag", [0.5, 0.0])
def test_combination_equals_direct(u_mag, anti, arrival, bands):
    """First call: the direct path, bit for bit. Later calls with other inputs: the unit responses combined, equal
    to the direct handover to rounding; a unit input returns its own response."""
    dp_spatial._EARTH_TRANSFER.clear()
    args = _args(u_mag, anti, m_Zd=1e-9 if u_mag else 2e6, g_D=2e-40 if u_mag else 1e-4)   # Earth phase ~ 4 rad
    kw = dict(arrival=arrival, earth_band_costh=None if bands is None else BANDS)
    direct = dp_spatial._propagate_earth_spatial_direct
    rng = np.random.default_rng(3)
    F1 = rng.uniform(0.1, 1.0, (6, 3))
    first = _call(dps.propagate_earth_spatial, F1, args, **kw)
    ref1 = _call(direct, F1, args, **kw)
    if bands is None:
        assert np.array_equal(first, ref1)
    else:
        assert np.array_equal(first[0], ref1[0]) and np.array_equal(first[1], ref1[1])
    for F in (rng.uniform(0.1, 1.0, (6, 3)), np.tile([0.0, 1.0, 0.0], (6, 1)), rng.uniform(0.0, 2.0, (6, 3))):
        got = _call(dps.propagate_earth_spatial, F, args, **kw)
        ref = _call(direct, F, args, **kw)
        if bands is None:
            assert _close(got, ref)
        else:
            assert _close(got[0], ref[0]) and _close(got[1], ref[1])
    assert len(dp_spatial._EARTH_TRANSFER) == 1
    (entry,) = dp_spatial._EARTH_TRANSFER.values()
    assert sorted(entry["cols"]) == [0, 1, 2]


def test_parameters_key_the_reuse():
    """Another potential (or the other kind) is another entry: never the first one's responses."""
    dp_spatial._EARTH_TRANSFER.clear()
    F = np.tile([0.2, 0.3, 0.5], (6, 1))
    direct = dp_spatial._propagate_earth_spatial_direct
    for g in (2e-40, 4e-40, 2e-40, 4e-40):
        for anti in (False, True):
            args = _args(0.5, anti, m_Zd=1e-9, g_D=g)
            assert _close(_call(dps.propagate_earth_spatial, F, args), _call(direct, F, args))
    assert len(dp_spatial._EARTH_TRANSFER) == 4


def test_sky_with_galactic_component_unchanged():
    """propagate_sky_basis (extragalactic and Galactic bands, dark portal, 3-D emission): the same with the reuse on
    and off."""
    from test_event_likelihood import DP, _sky_cfg
    E, params = dps.build_solver_params(_sky_cfg(0.3, 0.7, "3d", dark_portal=DP))
    params["show_progress"] = False
    assert params["earth_transfer"] == "off"                     # the solver's default: direct, as before
    out = {}
    for mode in ("off", "repeat", "always"):
        dp_spatial._EARTH_TRANSFER.clear()
        out[mode] = dps.propagate_sky_basis(E, dict(params, earth_transfer=mode))
    for mode in ("repeat", "always"):
        for k in ("bands_nu", "bands_nubar", "galactic_bands_nu", "galactic_bands_nubar"):
            assert _close(out[mode][k], out["off"][k]), (mode, k)
    with pytest.raises(ValueError, match="earth_transfer"):
        _call(dps.propagate_earth_spatial, np.ones((6, 3)), _args(0.5, False), earth_transfer="sometimes")
