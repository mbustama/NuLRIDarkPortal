"""
Tests of IceCubeRates (docs/dev/DR2_TRACKS_PLAN.md, task 3): the DR2 fold's geometry (the point-spread kernel
against Monte Carlo, probability conservation), its units on a synthetic response (an analytic count), the flux
adapters, and the committed DR2 reduction's own normalizations.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from IceCubeRates import flux_adapters as FA  # noqa: E402
from IceCubeRates.northern_muons import dr2  # noqa: E402

PSF_EDGES = np.append(np.round(np.arange(-2.0, 2.25 + 1e-9, 0.25), 6), np.log10(180.0))
TRUE_DEC = np.radians(np.array([-90.0, -30.0, -5.0, 0.0, 10.0, 30.0, 60.0, 85.0, 90.0]))
RECO_DEC = np.radians(np.array([-90.0, -60.0, -20.0, -5.0, 0.0, 5.0, 15.0, 30.0, 50.0, 70.0, 88.0, 90.0]))


def test_psf_kernel_conserves_probability():
    G = dr2.psf_kernel(TRUE_DEC, PSF_EDGES, RECO_DEC)
    assert G.shape == (len(TRUE_DEC) - 1, len(PSF_EDGES) - 1, len(RECO_DEC) - 1)
    np.testing.assert_allclose(G.sum(-1), 1.0, rtol=0, atol=1e-13)
    assert G.min() > -1e-14
    # a partial reconstructed range keeps only its share
    Gp = dr2.psf_kernel(TRUE_DEC, PSF_EDGES, RECO_DEC[3:])
    np.testing.assert_allclose(Gp, G[:, :, 3:], rtol=0, atol=1e-13)


def test_psf_kernel_small_angles_is_the_overlap():
    G = dr2.psf_kernel(TRUE_DEC, PSF_EDGES, RECO_DEC)
    G0 = dr2.overlap_kernel(TRUE_DEC, len(PSF_EDGES) - 1, RECO_DEC)
    # psi < 0.018 deg against bins of >= 2 deg: differences of order psi / (bin width)
    assert np.abs(G[:, 0] - G0[:, 0]).max() < 3e-3


@pytest.mark.parametrize("k", range(len(TRUE_DEC) - 1))
def test_psf_kernel_against_monte_carlo(k):
    """Each (true bin, PSF bin): tracks uniform in sin(dec), psi log-uniform in the bin, azimuth uniform; the
    reconstructed declination by spherical geometry, histogrammed (4 sigma, ~1600 comparisons in all)."""
    rng = np.random.default_rng(100 + k)
    G = dr2.psf_kernel(TRUE_DEC, PSF_EDGES, RECO_DEC)
    n = 400_000
    for q in range(len(PSF_EDGES) - 1):
        s = rng.uniform(np.sin(TRUE_DEC[k]), np.sin(TRUE_DEC[k + 1]), n)
        psi = np.radians(10.0 ** rng.uniform(PSF_EDGES[q], PSF_EDGES[q + 1], n))
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        sr = np.clip(s * np.cos(psi) + np.sqrt(1.0 - s * s) * np.sin(psi) * np.cos(phi), -1.0, 1.0)
        h = np.histogram(sr, bins=np.sin(RECO_DEC))[0] / n
        sig = np.sqrt(np.maximum(G[k, q] * (1.0 - G[k, q]), 1.0 / n) / n)
        assert np.max(np.abs(h - G[k, q]) / sig) < 4.5, (k, q)


def _synthetic_irfs(path, A_cm2=1e4, T_s=1e7, reco_bin=40, psf_bin=0):
    """A response with A_eff constant, every event in one reconstructed-energy bin and (by default) the smallest
    PSF bin; the smearing's binning differs from the effective area's."""
    le = np.round(np.arange(2.0, 8.0 + 1e-9, 0.2), 6)
    de = np.degrees(np.arcsin(np.linspace(-1.0, 1.0, 21)))
    sle = np.round(np.arange(2.0, 8.0 + 1e-9, 0.5), 6)
    sde = np.degrees(np.arcsin(np.linspace(-1.0, 1.0, 9)))
    reco = np.round(np.arange(1.0, 9.0 + 1e-9, 0.1), 6)
    P = np.zeros((len(sle) - 1, len(sde) - 1, len(reco) - 1, len(PSF_EDGES) - 1), dtype=np.float32)
    P[:, :, reco_bin, psf_bin] = 1.0
    np.savez(path, configs=np.array(["X"]), reco_edges_log10E=reco, psf_edges_log10deg=PSF_EDGES,
             X_aeff_log10E_edges=le, X_aeff_dec_edges_deg=de, X_aeff_cm2=np.full((len(le) - 1, len(de) - 1), A_cm2),
             X_smear_log10E_edges=sle, X_smear_dec_edges_deg=sde, X_smear=P, X_livetime_s=T_s)
    return le, de


def test_fold_units_against_an_analytic_count(tmp_path):
    """N = T A int dE E^-2 x 4 pi (isotropic, every bin): exact for a power law up to the 6-node rule per bin."""
    p = tmp_path / "irfs.npz"
    A, T = 1e4, 1e7
    le, de = _synthetic_irfs(p, A, T)
    tr = dr2.DR2Tracks(str(p))
    phi0, gamma = 1e-8, 2.0
    r = tr.expected(FA.power_law(phi0, gamma, pivot_GeV=1.0))
    exact = T * A * phi0 * (10.0 ** -le[0] - 10.0 ** -le[-1]) * 4.0 * np.pi
    assert r["counts"].shape == (len(de) - 1, len(tr.reco_edges) - 1)
    np.testing.assert_allclose(r["counts"].sum(), exact, rtol=1e-10)
    assert np.count_nonzero(r["counts"].sum(0)) == 1                 # one reconstructed-energy bin
    # the true-declination counts are uniform in sin(dec), and so (smallest PSF) are the reconstructed ones
    np.testing.assert_allclose(r["counts"].sum(1), exact / (len(de) - 1), rtol=1e-6)
    r0 = tr.expected(FA.power_law(phi0, gamma, pivot_GeV=1.0), psf=False)
    np.testing.assert_allclose(r0["counts"], np.where(r["counts"] > 0, r0["counts"], 0.0))
    np.testing.assert_allclose(r0["counts"].sum(1), exact / (len(de) - 1), rtol=1e-12)


def test_fold_psf_moves_tracks_but_keeps_them(tmp_path):
    p = tmp_path / "irfs.npz"
    _synthetic_irfs(p, psf_bin=14)                                    # psi of 31.6-56 deg
    tr = dr2.DR2Tracks(str(p))
    fl = FA.power_law(1e-8, 2.0, pivot_GeV=1.0)
    north = np.radians(np.array([-90.0, -5.0, 90.0]))
    r, r0 = tr.expected(fl, reco_dec_edges=north), tr.expected(fl, reco_dec_edges=north, psf=False)
    np.testing.assert_allclose(r["counts"].sum(), r0["counts"].sum(), rtol=1e-12)
    # an isotropic sky stays isotropic under any PSF (the kernel is the sphere's own rotation average)
    np.testing.assert_allclose(r["counts"].sum(1) / r["counts"].sum(), [0.5 * (1 - np.sin(np.radians(5.0))),
                                                                        0.5 * (1 + np.sin(np.radians(5.0)))], rtol=2e-4)


def test_from_sky_band_averages():
    E = np.geomspace(1e12, 1e17, 11)
    edges = np.radians(np.array([-90.0, 0.0, 90.0]))
    B = np.zeros((2, len(E), 3))
    B[0, :, 1], B[1, :, 1] = 1.0, 3.0
    sky = dict(E_eV=E, dec_band_edges=edges, bands_nu=B, bands_nubar=B, meta=dict(extragalactic_scale=1.0),
               galactic_bands_nu=0.5 * B, galactic_bands_nubar=0.5 * B)
    f = FA.from_sky(sky)
    out = f(np.array([1e4, 1e6, 1e9]), np.radians(np.array([-30.0, -10.0, 20.0])), np.radians(np.array([-10.0, 20.0, 40.0])))
    s = np.sin(np.radians([-10.0, 0.0, 20.0]))
    mixed = (1.0 * (s[1] - s[0]) + 3.0 * (s[2] - s[1])) / (s[2] - s[0]) * 3.0
    np.testing.assert_allclose(out[:, 0], [3.0, mixed, 9.0])
    assert out[0, 2] == 0.0                                           # above the solver's grid
    eg, ga = FA.from_sky(sky, galactic=False), FA.from_sky(sky, extragalactic=False)
    np.testing.assert_allclose(eg(np.array([1e5]), edges[:-1], edges[1:]) + ga(np.array([1e5]), edges[:-1], edges[1:]),
                               f(np.array([1e5]), edges[:-1], edges[1:]))
    sky["meta"] = {}
    with pytest.raises(ValueError):
        FA.from_sky(sky)


def _synthetic_atm(path):
    """A table in tools/atmospheric_background.py's format: nominal flux exp(0.4 |cos|) E^-3.7 (exact under the
    log-linear interpolation), prompt 1% of it, a derivative 0.2 x nominal x log10(E / 1 TeV) for one Barr parameter,
    the others zero; daemonflux exp(0.3 cos) E^-3.7 on its own grid."""
    E = np.geomspace(10.0, 1e8, 30)
    cos = np.linspace(0.0, 1.0, 11)
    nom = np.outer(np.exp(0.4 * cos), E ** -3.7)
    params = list(FA.BARR_PARAMS) + ["cr_dgamma"]
    z = dict(E_GeV=E, cos_zenith=cos, params=np.array(params), prior=np.array([0.3] * 12 + [np.nan]),
             models=np.array("test"), daemonflux_cos_zenith_detector=np.linspace(-1.0, 1.0, 41))
    for s in ("nue", "antinue", "numu", "antinumu", "nutau", "antinutau"):
        z["nom_" + s] = nom
        z["pr_" + s] = 0.01 * nom
        for p in params:
            z[f"d_{p}__{s}"] = (0.2 * nom * np.log10(E / 1e3)[None, :]) if p == "barr_h_pi+" else np.zeros_like(nom)
    for q in ("numu", "antinumu"):
        z[f"daemonflux_conv_{q}"] = np.outer(np.exp(0.3 * np.linspace(-1.0, 1.0, 41)), E ** -3.7)
        z[f"daemonflux_conv_{q}_err"] = 0.1 * z[f"daemonflux_conv_{q}"]
    np.savez(path, **z)
    return E


def test_atmospheric_adapter(tmp_path):
    p = str(tmp_path / "atm.npz")
    _synthetic_atm(p)
    dec = np.radians(np.array([-90.0, -30.0, 0.0, 30.0, 90.0]))
    E = np.array([50.0, 1e3, 1e5])
    nom = FA.Atmospheric(term="nom", path=p)(E, dec[:-1], dec[1:])
    # nu_mu + nubar_mu, exp(0.4 |cos zenith|) with |cos zenith| = |sin dec|: the band means in closed form
    a, b = np.array([0.5, 0.0, 0.0, 0.5]), np.array([1.0, 0.5, 0.5, 1.0])     # |sin dec| ranges of the bands
    band = (np.exp(0.4 * b) - np.exp(0.4 * a)) / (0.4 * (b - a))
    np.testing.assert_allclose(nom, 2.0 * np.outer(band, E ** -3.7), rtol=1e-12)
    np.testing.assert_allclose(nom[:2], nom[2:][::-1], rtol=1e-12)          # up-going mirrors down-going
    # the full flux at zero nuisances is the nominal one; it is linear in every nuisance
    np.testing.assert_allclose(FA.Atmospheric(path=p)(E, dec[:-1], dec[1:]), nom, rtol=1e-12)
    th = {"barr_h_pi+": 0.7, "prompt": -0.4, "barr_y_K-": 1.3}
    full = FA.Atmospheric(th, path=p)(E, dec[:-1], dec[1:])
    lin = nom + sum(v * FA.Atmospheric(term=k, path=p)(E, dec[:-1], dec[1:]) for k, v in th.items())
    np.testing.assert_allclose(full, lin, rtol=1e-12)
    np.testing.assert_allclose(FA.Atmospheric(term="barr_h_pi+", path=p)(E, dec[:-1], dec[1:]),
                               nom * 0.2 * np.log10(E / 1e3)[None, :], rtol=1e-9, atol=1e-15 * nom.max())
    np.testing.assert_allclose(FA.Atmospheric(term="prompt", path=p)(E, dec[:-1], dec[1:]), 0.01 * nom, rtol=1e-9)
    pri = FA.Atmospheric(path=p).priors()
    assert set(pri) == set(FA.ATM_PARAMS) and pri["barr_h_pi+"] == 0.3 and pri["prompt"] == 0.5
    assert pri["cr_dgamma"] == 0.05
    # daemonflux on its own grid (no mirror), cos zenith = -sin dec from 1 (south pole) to -1
    dm = FA.Atmospheric(term="daemonflux", path=p)(E, dec[:-1], dec[1:])
    sd = np.sin(dec)
    exact = (np.exp(-0.3 * sd[:-1]) - np.exp(-0.3 * sd[1:])) / (0.3 * (sd[1:] - sd[:-1]))
    np.testing.assert_allclose(dm[:, 1], 2.0 * exact * 1e3 ** -3.7, rtol=1e-12)
    assert np.all(FA.Atmospheric(term="nom", path=p)(np.array([1.0, 1e9]), dec[:-1], dec[1:]) == 0.0)
    with pytest.raises(ValueError):
        FA.Atmospheric({"barr_q_pi+": 1.0}, path=p)


@pytest.mark.skipif(not os.path.exists(dr2.DATA), reason="IceCubeRates/data/dr2_irfs.npz not present")
def test_committed_dr2_reduction():
    tr = dr2.DR2Tracks()
    assert tr.configs == ["IC40", "IC59", "IC79", "IC86"]
    for c in tr.configs:
        norm = tr.data[f"{c}_smear_norm"]
        np.testing.assert_allclose(norm[norm > 0], 1.0, atol=1e-9)
        np.testing.assert_allclose(tr.data[f"{c}_smear"].sum(axis=(2, 3)), norm, atol=1e-5)
        le, de, A = tr.aeff(c)
        assert A.shape == (len(le) - 1, len(de) - 1) and np.all(A >= 0)
        assert abs(de[0] + np.pi / 2) < 1e-9 and abs(de[-1] - np.pi / 2) < 1e-9
    days = {c: tr.livetime(c) / 86400.0 for c in tr.configs}
    assert 300 < days["IC40"] < 400 and 3800 < days["IC86"] < 4000
