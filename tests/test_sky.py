"""
Tests of the sky geometry (src/dp_sky.py; docs/dev/DIRECTIONS_PLAN.md, step 1): HEALPix RING
pixelization against its defining properties (and healpy where installed), the Galactic and ecliptic
frames against fixed points (and astropy where installed), the declination bands, and the yearly
distribution of the angle to the Sun, whose average over the sky must be the isotropic one.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_sky as sky  # noqa: E402

DEG = np.pi / 180.0


# =============================================================================
# HEALPix
# =============================================================================

@pytest.mark.parametrize("nside", [1, 2, 4, 8, 16, 32, 64, 256])
def test_healpix_round_trip_and_rings(nside):
    """Every pixel centre maps back to its pixel; 4 nside - 1 iso-latitude rings, with the ring
    heights z of Gorski et al. 2005 (caps: 1 - i^2/(3 nside^2); equator: 4/3 - 2 i/(3 nside));
    equal areas and a symmetric layout (the centres' mean vector vanishes)."""
    npix = sky.nside_to_npix(nside)
    th, ph = sky.pix2ang_ring(nside, np.arange(npix))
    assert np.array_equal(sky.ang2pix_ring(nside, th, ph), np.arange(npix))
    z = np.round(np.cos(th), 12)
    rings = np.unique(z)
    assert len(rings) == 4 * nside - 1
    i = np.arange(1, 4 * nside)
    zr = np.where(i < nside, 1 - i ** 2 / (3 * nside ** 2),
                  np.where(i <= 3 * nside, 4 / 3 - 2 * i / (3 * nside), -1 + (4 * nside - i) ** 2 / (3 * nside ** 2)))
    assert np.allclose(np.sort(rings)[::-1], zr, atol=1e-12)
    v = sky.unit_vector(ph, 0.5 * np.pi - th)
    assert np.abs(v.mean(0)).max() < 1e-12


def test_healpix_nside1_centres():
    """N_side = 1: three rings of four pixels at z = 2/3, 0, -2/3; the polar rings at
    phi = (j - 1/2) pi/2, the equatorial one at phi = j pi/2."""
    th, ph = sky.pix2ang_ring(1, np.arange(12))
    assert np.allclose(np.cos(th), [2 / 3] * 4 + [0] * 4 + [-2 / 3] * 4, atol=1e-14)
    assert np.allclose(ph[:4], (np.arange(4) + 0.5) * np.pi / 2)
    assert np.allclose(ph[4:8], np.arange(4) * np.pi / 2)


def test_healpix_random_directions_land_in_the_nearest_pixel_region():
    """Random directions fall in the pixel whose centre is within the pixel's angular size."""
    rng = np.random.default_rng(1)
    nside = 16
    v = rng.normal(size=(20000, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    lon, lat = sky.lonlat(v)
    p = sky.ang2pix_ring(nside, 0.5 * np.pi - lat, lon)
    th, ph = sky.pix2ang_ring(nside, p)
    c = sky.unit_vector(ph, 0.5 * np.pi - th)
    ang = np.arccos(np.clip(np.sum(c * v, axis=1), -1, 1))
    assert ang.max() < 1.5 * np.sqrt(4 * np.pi / sky.nside_to_npix(nside))


def test_healpix_against_healpy():
    hp = pytest.importorskip("healpy")
    for nside in [1, 8, 64]:
        p = np.arange(sky.nside_to_npix(nside))
        th, ph = sky.pix2ang_ring(nside, p)
        th2, ph2 = hp.pix2ang(nside, p)
        assert np.allclose(th, th2, atol=1e-13) and np.allclose(ph, ph2, atol=1e-13)


# =============================================================================
# Frames
# =============================================================================

def test_galactic_frame_fixed_points():
    """The Galactic Centre at (RA, dec) = (266.40499 deg, -28.93617 deg) and the north Galactic pole at
    (192.8594812, 27.1282512) (FK5 J2000, as astropy); the celestial pole at l = 122.9319186 deg;
    round trips."""
    ra, dec = sky.galactic_to_equatorial(0.0, 0.0)
    assert abs(ra / DEG - 266.40499) < 1e-4 and abs(dec / DEG + 28.93617) < 1e-4
    l, b = sky.equatorial_to_galactic(192.8594812 * DEG, 27.1282512 * DEG)
    assert abs(b / DEG - 90.0) < 1e-9
    l, b = sky.equatorial_to_galactic(0.0, 0.5 * np.pi)
    assert abs(l / DEG - 122.9319186) < 1e-9
    rng = np.random.default_rng(2)
    ra = rng.uniform(0, 2 * np.pi, 1000)
    dec = np.arcsin(rng.uniform(-1, 1, 1000))
    r2, d2 = sky.galactic_to_equatorial(*sky.equatorial_to_galactic(ra, dec))
    assert np.allclose(sky.unit_vector(r2, d2), sky.unit_vector(ra, dec), atol=1e-14)
    assert np.allclose(sky.EQ_TO_GAL @ sky.EQ_TO_GAL.T, np.eye(3), atol=1e-15)


def test_galactic_frame_against_astropy():
    coords = pytest.importorskip("astropy.coordinates")
    units = pytest.importorskip("astropy.units")
    rng = np.random.default_rng(3)
    ra = rng.uniform(0, 360, 500)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 500)))
    g = coords.SkyCoord(ra=ra * units.deg, dec=dec * units.deg, frame="fk5").galactic
    l, b = sky.equatorial_to_galactic(ra * DEG, dec * DEG)
    va = sky.unit_vector(g.l.rad, g.b.rad)
    vb = sky.unit_vector(l, b)
    # chord length (arccos of a dot product floors at 1.5e-8); measured 7e-10 rad (0.15 mas): astropy
    # derives the frame through its own transformation chain
    assert np.linalg.norm(va - vb, axis=1).max() < 1e-8


def test_ecliptic_frame():
    """The north ecliptic pole at RA = 270 deg, dec = 90 deg - obliquity; the equinox at (0, 0)."""
    lon, lat = sky.equatorial_to_ecliptic(270.0 * DEG, (90.0 - sky.OBLIQUITY_DEG) * DEG)
    assert abs(lat / DEG - 90.0) < 1e-9
    lon, lat = sky.equatorial_to_ecliptic(0.0, 0.0)
    assert abs(lon) < 1e-14 and abs(lat) < 1e-14


def test_zenith_at_the_south_pole():
    """cos(zenith) = -sin(dec) at the Pole for every hour angle; the general formula agrees there."""
    dec = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 7)
    assert np.allclose(sky.zenith_cos(dec), -np.sin(dec))
    for H in [0.0, 1.0, 3.0]:
        assert np.allclose(sky.zenith_cos(dec, -90.0, H), -np.sin(dec), atol=1e-15)
    with pytest.raises(ValueError):
        sky.zenith_cos(dec, -30.0)
    gc_dec = sky.galactic_to_equatorial(0.0, 0.0)[1]
    assert abs(sky.zenith_cos(gc_dec) - 0.4838) < 1e-3          # the Galactic Centre is down-going


# =============================================================================
# The sky grid and the bands
# =============================================================================

def test_sky_grid_bands():
    """Default bands: 20 of equal solid angle (0.1 in sin(dec)), partitioning the sky; every pixel
    centre inside its band; the bands' zenith edges are cos(theta_z) = -1, -0.9, ..., 1."""
    g = sky.SkyGrid(16)
    assert g.npix == 3072 and abs(g.omega * g.npix - 4 * np.pi) < 1e-12
    assert g.n_bands == 20
    assert np.allclose(g.band_omega, 4 * np.pi / 20) and abs(g.band_omega.sum() - 4 * np.pi) < 1e-12
    lo, hi = g.band_edges[g.band], g.band_edges[g.band + 1]
    assert np.all((g.dec >= lo) & (g.dec <= hi))
    assert np.allclose(g.costh_edges, np.linspace(-1, 1, 21), atol=1e-15)
    with pytest.raises(ValueError):
        sky.SkyGrid(16, dec_band_edges=[0.0, 0.5])
    with pytest.raises(ValueError):
        sky.SkyGrid(12)


# =============================================================================
# The angle to the Sun over a year
# =============================================================================

def test_kepler_sun_longitudes():
    """The Sun's longitude advances fastest at perihelion; for a circular orbit it is uniform; the
    distance stays within 1 +- e."""
    lam, r = sky.sun_ecliptic_longitudes(3600)
    assert abs(r.min() - (1 - sky.ECCENTRICITY)) < 1e-6 and abs(r.max() - (1 + sky.ECCENTRICITY)) < 1e-6
    dl = np.diff(np.unwrap(lam))
    assert dl.max() / dl.min() > 1.0 + 3.5 * sky.ECCENTRICITY
    lam0, r0 = sky.sun_ecliptic_longitudes(3600, eccentric=False)
    assert np.allclose(np.diff(np.unwrap(lam0)), 2 * np.pi / 3600) and np.all(r0 == 1.0)


def test_yearly_angle_to_sun():
    """For one direction (circular orbit) cos(psi) follows the arcsine law of its ecliptic latitude
    (moments <x^2> = cos^2(beta)/2, <x^4> = 3 cos^4(beta)/8); over the whole sky (equal-area pixels,
    with the eccentric orbit) cos(psi) is uniform on [-1, 1], as the isotropic average assumes."""
    beta = 0.4
    x = sky.cos_angle_to_sun(np.array([1.0]), np.array([beta]), 20000, eccentric=False)[0]
    c = np.cos(beta)
    assert abs(np.mean(x ** 2) - c ** 2 / 2) < 1e-6 and abs(np.mean(x ** 4) - 3 * c ** 4 / 8) < 1e-6
    xs = np.linspace(-c + 1e-9, c - 1e-9, 400001)
    dens = sky.arcsine_density(xs, beta)
    trapz = getattr(np, "trapezoid", None) or np.trapz      # NumPy 2 renamed it
    assert abs(trapz(dens, xs) - 1) < 2e-3                 # endpoint singularities, cut at 1e-9
    g = sky.SkyGrid(16)
    X = sky.cos_angle_to_sun(g.elon, g.elat, 360)
    assert abs(X.mean()) < 1e-3
    assert abs(np.mean(X ** 2) - 1 / 3) < 1e-3 and abs(np.mean(X ** 4) - 1 / 5) < 1e-3
