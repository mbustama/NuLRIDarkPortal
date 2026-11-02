"""
dp_sky.py

Sky geometry for the per-direction output (docs/dev/DIRECTIONS_PLAN.md, step 1; agreed with
Mauricio 2026-10-13): HEALPix pixels, the equatorial, Galactic and ecliptic frames, the zenith of
a direction at the detector, declination bands, and the yearly distribution of the angle to the
Sun. No physics and no third-party sky library: the solver's environments (laptop, Tycho, CI) have
neither healpy nor astropy, and the few operations needed here have closed forms. The tests compare
them with astropy where it is installed.

Conventions (stated in every file the solver writes):
  - frame: equatorial J2000 (FK5; IceCube's sky maps are equatorial J2000, arXiv:2307.04427), with
    Galactic coordinates by the IAU 1958 definition (Blaauw et al., MNRAS 121, 123 (1960)) in the
    FK5 J2000 realization used by astropy: north Galactic pole at (RA, dec) = (192.8594812 deg,
    +27.1282512 deg), Galactic longitude of the north celestial pole 122.9319186 deg;
  - HEALPix (Gorski et al., ApJ 622, 759 (2005)), RING ordering, colatitude theta = pi/2 - dec and
    longitude phi = RA (healpy's convention; not the north-south flipped theta = pi/2 + dec of
    ../NuAnisotropy's fit script);
  - zenith at the South Pole: cos(theta_z) = -sin(dec), the same at every time (Pierre Auger, JCAP
    11 (2019) 004; IceCube, EPJC 79, 234 (2019)). Up-going means dec > 0.
"""
import numpy as np

# -----------------------------------------------------------------------------
# Frames
# -----------------------------------------------------------------------------

DEG = np.pi / 180.0
NGP_RA_DEG = 192.8594812
NGP_DEC_DEG = 27.1282512
L_NCP_DEG = 122.9319186
# mean obliquity of the ecliptic at J2000 (IAU 2006: 84381.406 arcsec)
OBLIQUITY_DEG = 84381.406 / 3600.0


def unit_vector(lon, lat):
    """Cartesian unit vectors (..., 3) of longitudes and latitudes (rad)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    c = np.cos(lat)
    return np.stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)], axis=-1)


def lonlat(v):
    """Longitude in [0, 2pi) and latitude in [-pi/2, pi/2] of vectors (..., 3)."""
    v = np.asarray(v, dtype=np.float64)
    lon = np.mod(np.arctan2(v[..., 1], v[..., 0]), 2.0 * np.pi)
    lat = np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1]))
    return lon, lat


def _equatorial_to_galactic_matrix():
    """Rows: the Galactic x (towards l = b = 0), y (l = 90 deg) and z (north Galactic pole) axes in
    equatorial components. x follows from the pole and the longitude of the celestial pole: the
    celestial pole's projection on the Galactic plane lies at l = L_NCP."""
    z_g = unit_vector(NGP_RA_DEG * DEG, NGP_DEC_DEG * DEG)
    ncp = np.array([0.0, 0.0, 1.0])
    p = ncp - np.dot(ncp, z_g) * z_g
    p /= np.linalg.norm(p)
    q = np.cross(z_g, p)
    lc = L_NCP_DEG * DEG
    x_g = np.cos(lc) * p - np.sin(lc) * q
    y_g = np.cross(z_g, x_g)
    return np.array([x_g, y_g, z_g])


EQ_TO_GAL = _equatorial_to_galactic_matrix()
_eps = OBLIQUITY_DEG * DEG
EQ_TO_ECL = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(_eps), np.sin(_eps)],
                      [0.0, -np.sin(_eps), np.cos(_eps)]])


def equatorial_to_galactic(ra, dec):
    """(l, b) in rad from (RA, dec) in rad."""
    return lonlat(unit_vector(ra, dec) @ EQ_TO_GAL.T)


def galactic_to_equatorial(l, b):
    """(RA, dec) in rad from (l, b) in rad."""
    return lonlat(unit_vector(l, b) @ EQ_TO_GAL)


def equatorial_to_ecliptic(ra, dec):
    """(ecliptic longitude, latitude) in rad from (RA, dec) in rad (mean equinox and ecliptic of J2000)."""
    return lonlat(unit_vector(ra, dec) @ EQ_TO_ECL.T)


def zenith_cos(dec, latitude_deg=-90.0, hour_angle=None):
    """cos(zenith angle) of a direction at declination dec (rad) for a detector at the given latitude:
    sin(lat) sin(dec) + cos(lat) cos(dec) cos(H). At the South Pole (the default) it is -sin(dec) for
    every hour angle; elsewhere the hour angle H (rad) is needed."""
    lat = latitude_deg * DEG
    if hour_angle is None:
        if abs(latitude_deg + 90.0) > 1e-12:
            raise ValueError("zenith_cos: the hour angle is needed away from the South Pole")
        return -np.sin(np.asarray(dec, dtype=np.float64))
    return np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)


# -----------------------------------------------------------------------------
# HEALPix, RING ordering (the algorithms of Gorski et al. 2005 / chealpix)
# -----------------------------------------------------------------------------

def nside_to_npix(nside):
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"HEALPix: nside must be a power of 2, got {nside}")
    return 12 * nside * nside


def pix2ang_ring(nside, ipix):
    """(theta, phi) in rad of the centres of RING-ordered pixels ipix (array)."""
    npix = nside_to_npix(nside)
    ip = np.asarray(ipix, dtype=np.int64)
    if np.any(ip < 0) or np.any(ip >= npix):
        raise ValueError("HEALPix: pixel index out of range")
    ncap = 2 * nside * (nside - 1)
    fact2 = 4.0 / npix
    z = np.empty(ip.shape)
    phi = np.empty(ip.shape)
    north = ip < ncap
    south = ip >= npix - ncap
    equ = ~(north | south)
    if np.any(north):
        p = ip[north]
        iring = (1 + np.floor(np.sqrt(1 + 2 * p + 0.5)).astype(np.int64)) >> 1
        iring = _fix_isqrt_ring(iring, p, north_cap=True)
        iphi = (p + 1) - 2 * iring * (iring - 1)
        z[north] = 1.0 - iring * iring * fact2
        phi[north] = (iphi - 0.5) * (0.5 * np.pi) / iring
    if np.any(equ):
        p = ip[equ] - ncap
        iring = p // (4 * nside) + nside
        iphi = p % (4 * nside) + 1
        fodd = np.where(((iring + nside) & 1) == 1, 1.0, 0.5)
        z[equ] = (2 * nside - iring) * (2 * nside * fact2)
        phi[equ] = (iphi - fodd) * np.pi / (2 * nside)
    if np.any(south):
        p = npix - ip[south]
        iring = (1 + np.floor(np.sqrt(2 * p - 1 + 0.5)).astype(np.int64)) >> 1
        iring = _fix_isqrt_ring(iring, p, north_cap=False)
        iphi = 4 * iring + 1 - (p - 2 * iring * (iring - 1))
        z[south] = -1.0 + iring * iring * fact2
        phi[south] = (iphi - 0.5) * (0.5 * np.pi) / iring
    return np.arccos(np.clip(z, -1.0, 1.0)), phi


def _fix_isqrt_ring(iring, p, north_cap):
    """Guard the integer square root against floating-point rounding: the ring of a cap pixel p is the
    largest i with 2 i (i - 1) <= p (north, p = ipix) or 2 i (i - 1) < p (south, p = npix - ipix)."""
    for _ in range(2):
        lo = 2 * iring * (iring - 1)
        too_big = (lo > p) if north_cap else (lo >= p)
        iring = np.where(too_big, iring - 1, iring)
        hi = 2 * (iring + 1) * iring
        too_small = (hi <= p) if north_cap else (hi < p)
        iring = np.where(too_small, iring + 1, iring)
    return iring


def ang2pix_ring(nside, theta, phi):
    """RING-ordered pixel indices of directions (theta, phi) in rad."""
    npix = nside_to_npix(nside)
    z = np.cos(np.asarray(theta, dtype=np.float64))
    za = np.abs(z)
    tt = np.mod(np.asarray(phi, dtype=np.float64), 2.0 * np.pi) / (0.5 * np.pi)   # in [0, 4)
    z, za, tt = np.broadcast_arrays(z, za, tt)
    out = np.empty(z.shape, dtype=np.int64)
    equ = za <= 2.0 / 3.0
    if np.any(equ):
        t1 = nside * (0.5 + tt[equ])
        t2 = nside * z[equ] * 0.75
        jp = np.floor(t1 - t2).astype(np.int64)
        jm = np.floor(t1 + t2).astype(np.int64)
        ir = nside + 1 + jp - jm
        kshift = 1 - (ir & 1)
        ip = (jp + jm - nside + kshift + 1) // 2
        ip = np.mod(ip, 4 * nside)
        out[equ] = 2 * nside * (nside - 1) + (ir - 1) * 4 * nside + ip
    cap = ~equ
    if np.any(cap):
        tc = tt[cap]
        tp = tc - np.floor(tc)
        tmp = nside * np.sqrt(3.0 * (1.0 - za[cap]))
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.floor(tc * ir).astype(np.int64)
        ip = np.mod(ip, 4 * ir)
        out[cap] = np.where(z[cap] > 0, 2 * ir * (ir - 1) + ip, npix - 2 * ir * (ir + 1) + ip)
    return out


# -----------------------------------------------------------------------------
# The sky grid of the per-direction output
# -----------------------------------------------------------------------------

def default_dec_band_edges(n_bands=20):
    """Declination band edges (rad, ascending): uniform in sin(dec), so uniform in cos(zenith) at the
    South Pole and each band of equal solid angle (decision D1: 20 bands of 0.1 in sin(dec), mergeable
    into the MESE release's 10 bins of 0.2)."""
    return np.arcsin(np.linspace(-1.0, 1.0, n_bands + 1))


class SkyGrid:
    """HEALPix pixels (equatorial, RING) and declination bands.

    Attributes: nside, npix, theta, phi (pixel centres, rad), ra, dec, l, b (Galactic), elon, elat
    (ecliptic), omega (sr per pixel), band_edges (dec, ascending), band (band index per pixel, by its
    centre), band_omega (sr per band), costh_edges (cos(zenith) of the band edges at the South Pole,
    ascending: the Earth chords' nodes)."""

    def __init__(self, nside=16, dec_band_edges=None):
        self.nside = int(nside)
        self.npix = nside_to_npix(self.nside)
        self.theta, self.phi = pix2ang_ring(self.nside, np.arange(self.npix))
        self.ra = self.phi
        self.dec = 0.5 * np.pi - self.theta
        self.l, self.b = equatorial_to_galactic(self.ra, self.dec)
        self.elon, self.elat = equatorial_to_ecliptic(self.ra, self.dec)
        self.omega = 4.0 * np.pi / self.npix
        edges = default_dec_band_edges() if dec_band_edges is None else np.asarray(dec_band_edges, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("SkyGrid: declination band edges must increase")
        if abs(edges[0] + 0.5 * np.pi) > 1e-12 or abs(edges[-1] - 0.5 * np.pi) > 1e-12:
            raise ValueError("SkyGrid: declination bands must cover the sky, from -pi/2 to pi/2")
        self.band_edges = edges
        self.band = np.clip(np.searchsorted(edges, self.dec, side="right") - 1, 0, len(edges) - 2)
        self.band_omega = 2.0 * np.pi * np.diff(np.sin(edges))
        self.costh_edges = np.sort(-np.sin(edges))

    @property
    def n_bands(self):
        return len(self.band_edges) - 1


# -----------------------------------------------------------------------------
# The angle to the Sun over a year
# -----------------------------------------------------------------------------

ECCENTRICITY = 0.0167086
# ecliptic longitude of the Sun at perihelion (deg, J2000: the Earth's perihelion longitude
# 102.9373 deg plus 180 deg)
SUN_PERIHELION_LON_DEG = 282.9373


def sun_ecliptic_longitudes(n_times=720, eccentric=True):
    """The Sun's ecliptic longitudes (rad) at n_times instants equally spaced over a year, and its
    distance in units of the semi-major axis: Kepler's equation for the Earth's orbit (eccentricity
    0.0167), or a circular orbit. Uniform in time is uniform in the mean anomaly."""
    M = 2.0 * np.pi * (np.arange(n_times) + 0.5) / n_times
    if not eccentric:
        return np.mod(M + SUN_PERIHELION_LON_DEG * DEG, 2 * np.pi), np.ones(n_times)
    e = ECCENTRICITY
    E = M.copy()
    for _ in range(30):
        E -= (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return np.mod(nu + SUN_PERIHELION_LON_DEG * DEG, 2 * np.pi), 1.0 - e * np.cos(E)


def cos_angle_to_sun(elon, elat, n_times=720, eccentric=True):
    """cos(psi) between directions at ecliptic (lon, lat) (rad, arrays (P,)) and the Sun, at n_times
    instants over a year: (P, n_times). The Sun lies on the ecliptic, so
    cos(psi) = cos(lat) cos(lon - lon_sun)."""
    lam, _ = sun_ecliptic_longitudes(n_times, eccentric)
    return np.cos(np.asarray(elat))[:, None] * np.cos(np.asarray(elon)[:, None] - lam[None, :])


def arcsine_density(x, beta):
    """The yearly density of x = cos(psi) for a circular orbit at ecliptic latitude beta:
    1 / (pi sqrt(cos^2 beta - x^2)) on |x| < cos(beta)."""
    c = np.cos(beta)
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    inside = np.abs(x) < c
    out[inside] = 1.0 / (np.pi * np.sqrt(c * c - x[inside] ** 2))
    return out


def sun_path_weights(elon, elat, x_nodes, n_times=360, eccentric=True):
    """Per-direction yearly weights on the Sun transit's path nodes (DIRECTIONS_PLAN step 4): with the
    transit known at the cosines x_nodes (the Gauss-Legendre nodes of dp_transit.sun_paths), a direction
    at ecliptic (lon, lat) sees the Sun at x(t) = cos(psi(t)) over the year, and its flux is
    sum_k a_k F_k with a_k = <L_k(x(t))>_year, L_k the Lagrange basis polynomials on the nodes. Each row
    sums to 1, and averaged over the sky (uniform in x) the weights are the Gauss-Legendre ones, so the
    isotropic average is recovered. Returns (P, K)."""
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    X = cos_angle_to_sun(elon, elat, n_times, eccentric)                  # (P, T)
    K = len(x_nodes)
    W = np.empty((X.shape[0], K))
    for k in range(K):
        Lk = np.ones_like(X)
        for j in range(K):
            if j != k:
                Lk *= (X - x_nodes[j]) / (x_nodes[k] - x_nodes[j])
        W[:, k] = Lk.mean(axis=1)
    return W

