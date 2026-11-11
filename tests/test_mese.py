"""MESE release (IceCubeRates/data/mese_2025) and the event-weighted flavor composition (audit decision A3)."""
import numpy as np
import pytest

from analysis import mese


def test_area_nodes_and_range():
    (Ec, Ac), (Ed, Ad), (Et, At) = mese._area_nodes()
    assert len(Ec) == len(Et) == 31 and len(Ed) == 21
    assert Ec[0] == pytest.approx(1e3) and Ed[0] == pytest.approx(1e4) and Ec[-1] == pytest.approx(1e7)
    assert np.all(Ac > 0) and np.all(At > 0)
    out = mese.flavor_area(np.array([5e11, 1e12, 1e14, 1e16, 2e16]))
    assert np.all(out[0] == 0) and np.all(out[-1] == 0) and np.all(out[1:4] > 0)


def test_glashow_resonance_in_nu_e_only():
    """The release's Glashow column (cascades only) enters nu_e: its area rises near 6.3 PeV."""
    E = np.logspace(15.5, 16, 40)
    A = mese.flavor_area(E)
    r = A[:, 0] / A[:, 1]
    assert r[np.argmin(np.abs(E - 6.3e15))] > 1.5 * r[0]


def test_constant_composition_comes_back_exactly():
    E = np.logspace(np.log10(5e12), 16, 200)
    for f in [(1 / 3, 1 / 3, 1 / 3), (0.2, 0.5, 0.3), (1.0, 0.0, 0.0)]:
        F = np.outer(E ** -2.5, f)
        np.testing.assert_allclose(mese.event_weighted_fractions(E, F), f, atol=1e-14)


def test_weights_follow_the_areas_not_the_flux():
    """A composition that changes with energy: the result weights each flavor's share by its own area."""
    E = np.logspace(np.log10(5e12), 16, 300)
    x = np.log(E / E[0]) / np.log(E[-1] / E[0])             # 0 -> 1 across the band
    F = np.stack([1 - x, x, np.zeros_like(x)], axis=1) * E[:, None] ** -2.5
    f = mese.event_weighted_fractions(E, F)
    from dark_portal_solver import compute_dE_geometric
    w = mese.flavor_area(E) * compute_dE_geometric(E)[:, None]
    tot = F.sum(axis=1)
    raw = np.array([np.sum(F[:, a] * w[:, a]) / np.sum(tot * w[:, a]) for a in range(3)])
    np.testing.assert_allclose(f, raw / raw.sum(), rtol=1e-13)
    assert f[2] == 0.0 and 0 < f[0] < 1


def test_model_fractions_with_mese_weighting():
    """data.flavor_weighting 'mese' (audit decision A3): the model's fractions are the event-weighted ones of its
    flux in the flavor band; 'flux' (the default) keeps the flux-weighted average."""
    import json
    import os
    from analysis import config as config_mod
    from analysis import REPO_DIR
    raw = json.load(open(os.path.join(REPO_DIR, "analysis", "configs", "pion_2050_case_5_si_e_gal.json")))
    raw["solver"]["grid"].update(n_E_bins=60)
    raw["data"]["flavor_band_eV"] = [5e12, 1e16]
    raw["data"]["flavor_weighting"] = "mese"
    cfg = config_mod.from_dict(raw, "")
    m = cfg.model()
    v = dict(m_phi_eV=5.6e6, g_phi=3e-2, p_index=2.5, norm_astro=1.3, norm_gal=0.9)
    E, F = m.flux(v)
    sel = (E >= 5e12 * (1 - 1e-9)) & (E <= 1e16 * (1 + 1e-9))
    f = mese.event_weighted_fractions(E[sel], F[sel])
    np.testing.assert_allclose(m.fractions(v), f, atol=1e-15)
    raw["data"]["flavor_weighting"] = "flux"
    assert not np.allclose(config_mod.from_dict(raw, "").model().fractions(v), f, atol=1e-6)
    raw["data"]["flavor_weighting"] = "areas"
    with pytest.raises(ValueError):
        config_mod.from_dict(raw, "")
