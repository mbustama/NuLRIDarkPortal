"""MESE's event fold (IceCubeRates/starting_events/mese.py; set 2, audit decisions 2026-11-09)."""
import numpy as np
import pytest

from IceCubeRates.starting_events import mese


def _g1_integrals():
    """IceCube's flavor best fit (NuFlavorSingleSource gate G1's flux), per flavor nu + nubar, on the fine bins."""
    fe = mese.response()["fine_edges"]
    Eg = np.logspace(fe[0], fe[-1], 20 * (len(fe) - 1) + 1)
    eb = 10 ** 4.5
    f = 2.72e-18 * np.where(Eg >= eb, (Eg / 1e5) ** -2.81, (eb / 1e5) ** -2.81 * (Eg / eb) ** -1.76)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(Eg))])
    return np.interp(10 ** fe[1:], Eg, cum) - np.interp(10 ** fe[:-1], Eg, cum), f, Eg


def test_histograms():
    """The release's totals: data 4960 cascades, 9 double cascades, 4919 tracks; no muons among double cascades."""
    tot = {t: mese.histograms(t)["data"].sum() for t in mese.TOPOLOGIES}
    assert tot == dict(cascades=4960, double_cascades=9, tracks=4919)
    assert mese.histograms("double_cascades")["muon"].sum() == 0.0
    assert len(mese.histograms("cascades")["data"]) == 22          # arXiv:2510.24957 Table I
    assert len(mese.histograms("tracks")["data"]) == 13


def test_g1_flux_regression():
    """IceCube's best fit through the response, all sky, 11.4 yr (0.30 : 0.37 : 0.33): the totals when the table was
    built (2026-11-09; release Astro 692.6, 5.98, 165.5: the common offset NuFlavorSingleSource's gate G1 reports)."""
    I, _, _ = _g1_integrals()
    fr = (0.30, 0.37, 0.33)
    got = {}
    for t in mese.TOPOLOGIES:
        R = mese.isotropic_kernels(t)
        got[t] = (sum(3 * fr[a] * I @ R[a] for a in range(3)) + (1.5 * fr[0] * I) @ R["gr"]).sum()
    assert got["cascades"] == pytest.approx(795.00, abs=0.01)
    assert got["double_cascades"] == pytest.approx(5.84, abs=0.01)
    assert got["tracks"] == pytest.approx(187.61, abs=0.01)


def test_fold_of_isotropic_band_fluxes():
    """The band fold of an isotropic flux on energy nodes equals the isotropic kernels (to the node interpolation)."""
    _, _, _ = _g1_integrals()
    E = np.logspace(2, 8, 601)
    eb = 10 ** 4.5
    phi = 2.72e-18 * np.where(E >= eb, (E / 1e5) ** -2.81, (eb / 1e5) ** -2.81 * (E / eb) ** -1.76)
    F = np.zeros((6, len(E), 3))
    for a, f in enumerate((0.30, 0.37, 0.33)):
        F[:, :, a] = 3 * f * phi / 2
    out = mese.MESEFold(E, np.radians(np.linspace(-90, 90, 7)))(F, F)
    assert out["cascades"].sum() == pytest.approx(795.00, rel=2e-3)
    assert out["tracks"].sum() == pytest.approx(187.61, rel=2e-3)


def test_glashow_takes_only_nubar_e():
    E = np.logspace(2, 8, 601)
    F0 = np.zeros((1, len(E), 3))
    Fb = F0.copy()
    Fb[:, :, 0] = np.where((E > 4e6) & (E < 8e6), 1e-20, 0.0)
    fold = mese.MESEFold(E, np.radians([-90.0, 90.0]), topologies=("cascades",))
    nu_only = fold(Fb, F0)["cascades"].sum()                         # the same flux as nu_e: no Glashow events
    nubar_only = fold(F0, Fb)["cascades"].sum()
    assert nubar_only > 2 * nu_only > 0
