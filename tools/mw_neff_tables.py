"""
mw_neff_tables.py -- the Milky Way's neutron density averaged over spheres around any point of the Galaxy
(docs/dev/DIRECTIONS_PLAN.md, step 6): for emission points at galactocentric (R, z),

    nbar_x(r) = (1/4 pi) int dOmega n(x + r Omega),

so that the Yukawa-smoothed density there, the emission potential of a Galactic neutrino, is the same
one-dimensional integral the solver uses at the Sun (dp_transit.galactic_effective_density):

    n_eff(x; m) = int_0^inf m^2 r exp(-m r) max(nbar_x(r), n_n0) dr.

The model is dp_spatial.calc_MW_neutron_density (stars, hot halo, cold HI and H2 discs), evaluated through
its own Sun-centred coordinates, so the table and the solver share one model (its digest is stored). It is
axisymmetric and symmetric under z -> -z, so the nodes are R in [0, 20] kpc (and the Sun's 8.33) and z >= 0.
Around a point at (R, 0, z) the sphere is symmetric under y -> -y: nbar = (1/2 pi) int_-1^1 dmu
int_0^pi dphi n, with phi = 0 toward the Galactic Centre's side. Composite Gauss-Legendre (16 points per
panel), panels graded toward the plane crossing (mu* = -z/r, breakpoints 2 pc x 2^j / r) and toward phi = 0
(1 mrad x 2^j), as tools/mw_sky_density.py does for the Sun.

Stored: n_eff(R, z; m) on 20 masses per decade from 1e-34 to 1e-12 eV (above it, the contact limit: the local
density, n_local_cm3), integrated on the solver's distance nodes (200 per decade) with its head and floor
(galactic_effective_density without the cosmic tail beyond 2e4 kpc, which is the same everywhere in the Galaxy
and added at run time), for the default cosmic neutron density n_n0 (stored; dp_galactic checks it).

  python tools/mw_neff_tables.py [--out src/data/mw_neff_tables.npz]        (Tycho, ~10 min on 32 cores)

Checks: at the Sun's node the sphere average equals mw_sky_density's own quadrature to rounding (a direct
comparison at the same distances, printed), and n_eff there equals galactic_effective_density (test T7).
"""
import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

from dp_core import njit, prange  # noqa: E402
from dp_spatial import calc_MW_neutron_density  # noqa: E402

R_SUN = 8.33                     # kpc, as in calc_MW_neutron_density
GL16_X, GL16_W = np.polynomial.legendre.leggauss(16)
# graded toward the Galactic Centre (the bulge's 75-pc core), 0.5 kpc beyond 1 kpc, and the Sun's radius
R_NODES = np.unique(np.concatenate([[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75], np.arange(1.0, 20.0 + 1e-9, 0.5),
                                    [R_SUN]]))
LOG10_M = np.linspace(-34.0, -12.0, 441)        # eV: 20 per decade
Z_NODES = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.13, 0.17, 0.22, 0.3, 0.4, 0.5,
                    0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@njit(cache=True)
def density_galactocentric(X, Y, Z):
    """calc_MW_neutron_density at galactocentric (X, Y, Z) kpc (the Sun at (R_SUN, 0, 0)): its own
    Sun-centred coordinates, r, theta from the north Galactic pole, phi from the Galactic Centre's direction."""
    u = R_SUN - X                  # toward the Galactic Centre
    rho = np.sqrt(u * u + Y * Y)
    r = np.sqrt(rho * rho + Z * Z)
    if r == 0.0:
        return calc_MW_neutron_density(0.0, 0.5 * np.pi, 0.0)
    return calc_MW_neutron_density(r, np.arctan2(rho, Z), np.arctan2(Y, u))


@njit(cache=True)
def _panels_mu(z0, r, out):
    """Breakpoints in mu on [-1, 1], graded toward mu* = -z0 / r at 2 pc x 2^j / r; returns the count."""
    n = 0
    out[n] = -1.0
    n += 1
    ms = -z0 / r
    d = 0.002
    pts = np.empty(128)
    k = 0
    while d < 2.0 * r and k < 120:
        pts[k] = ms - d / r
        pts[k + 1] = ms + d / r
        k += 2
        d *= 2.0
    if -1.0 < ms < 1.0:
        pts[k] = ms
        k += 1
    srt = np.sort(pts[:k])
    for i in range(k):
        v = srt[i]
        if -1.0 < v < 1.0 and v > out[n - 1] + 1e-15:
            out[n] = v
            n += 1
    out[n] = 1.0
    return n + 1


@njit(cache=True)
def sphere_average(R0, z0, r, glx, glw, phi_b):
    """nbar at distance r (kpc) from the point (R0, 0, z0), in cm^-3."""
    mu_b = np.empty(260)
    nmu = _panels_mu(z0, r, mu_b)
    total = 0.0
    for i in range(nmu - 1):
        m_mid = 0.5 * (mu_b[i] + mu_b[i + 1])
        m_half = 0.5 * (mu_b[i + 1] - mu_b[i])
        for a in range(len(glx)):
            mu = m_mid + m_half * glx[a]
            st = np.sqrt(max(0.0, 1.0 - mu * mu))
            Z = z0 + r * mu
            acc = 0.0
            for j in range(len(phi_b) - 1):
                p_mid = 0.5 * (phi_b[j] + phi_b[j + 1])
                p_half = 0.5 * (phi_b[j + 1] - phi_b[j])
                s = 0.0
                for b in range(len(glx)):
                    ph = p_mid + p_half * glx[b]
                    # phi = 0 toward the Galactic Centre's side: X decreases
                    s += glw[b] * density_galactocentric(R0 - r * st * np.cos(ph), r * st * np.sin(ph), Z)
                acc += s * p_half
            total += glw[a] * acc * m_half
    return total / (2.0 * np.pi)


@njit(cache=True)
def neff_head_body(m, r, nbar, n_loc, n_n0):
    """galactic_effective_density's integral without the cosmic tail beyond r[-1], all in eV units:
    int_0^r[-1] m^2 r e^{-mr} max(nbar, n_n0) dr, linear from n_loc below the first node (the cusp)."""
    x0 = m * r[0]
    body = 0.0
    fprev = x0 * x0 * np.exp(-x0) * max(nbar[0], n_n0)
    for q in range(1, len(r)):
        x = m * r[q]
        f = x * x * np.exp(-x) * max(nbar[q], n_n0)
        body += 0.5 * (f + fprev) * (np.log(r[q]) - np.log(r[q - 1]))
        fprev = f
    beta = (nbar[0] - n_loc) / r[0]
    e0 = np.exp(-x0)
    head = (n_loc * (-np.expm1(-x0) - x0 * e0)
            + beta / m * (2.0 * (-np.expm1(-x0)) - (2.0 * x0 + x0 * x0) * e0))
    return head + body


@njit(cache=True, parallel=True)
def _table(R_nodes, z_nodes, r_nodes, glx, glw, phi_b, m_eV, r_eV, cm3_to_eV3, n_n0_eV3, neff, n_loc, order):
    """prange splits the points into contiguous blocks, one per thread: they are visited in the shuffled
    `order` so that the slow ones (near the Galactic Centre) do not pile up in one block."""
    nR, nz = len(R_nodes), len(z_nodes)
    for kk in prange(nR * nz):
        k = order[kk]
        i, j = k // nz, k % nz
        n_loc[i, j] = density_galactocentric(R_nodes[i], 0.0, z_nodes[j])
        nbar = np.empty(len(r_nodes))
        for q in range(len(r_nodes)):
            nbar[q] = sphere_average(R_nodes[i], z_nodes[j], r_nodes[q], glx, glw, phi_b) * cm3_to_eV3
        for a in range(len(m_eV)):
            neff[i, j, a] = neff_head_body(m_eV[a], r_eV, nbar, n_loc[i, j] * cm3_to_eV3, n_n0_eV3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "src", "data", "mw_neff_tables.npz"))
    ap.add_argument("--quick", action="store_true", help="a few nodes only (a check of the code)")
    a = ap.parse_args(argv)
    import mw_sky_density
    from dp_core import m_to_eV_inv, cm_inv3_to_eV3
    from dp_config import DEFAULT_CONFIG
    d = np.load(os.path.join(REPO, "src", "data", "mw_sky_density.npz"))
    digest = mw_sky_density.model_digest()
    if str(d["model_digest"]) != digest:
        raise RuntimeError("src/data/mw_sky_density.npz was made from another density model")
    r = d["r_kpc"]                                    # the solver's own distance nodes at the Sun
    r_eV = r * 3.08567758e19 * m_to_eV_inv
    n_n0_cm3 = float(DEFAULT_CONFIG["cosmology"]["n_n0_cm3"])
    Rn, zn = (np.array([0.0, R_SUN, 15.0]), np.array([0.0, 0.3])) if a.quick else (R_NODES, Z_NODES)
    phi_b = np.ascontiguousarray(mw_sky_density._breaks(1e-3, np.pi, 1.0))
    neff = np.empty((len(Rn), len(zn), len(LOG10_M)))
    n_loc = np.empty((len(Rn), len(zn)))
    t0 = time.perf_counter()
    order = np.random.default_rng(0).permutation(len(Rn) * len(zn))
    _table(Rn, zn, r, GL16_X, GL16_W, phi_b, 10.0 ** LOG10_M, r_eV, cm_inv3_to_eV3, n_n0_cm3 * cm_inv3_to_eV3,
           neff, n_loc, order)
    dt = time.perf_counter() - t0
    # direct check of the sphere average at the Sun against mw_sky_density's own quadrature
    i_sun = int(np.argmin(np.abs(Rn - R_SUN)))
    rs = np.geomspace(1e-3, 300.0, 13)
    dev = max(abs(sphere_average(R_SUN, 0.0, x, GL16_X, GL16_W, phi_b) / mw_sky_density.sky_average(x) - 1) for x in rs)
    note = (f"sphere average at the Sun vs mw_sky_density.sky_average at {len(rs)} distances: max |ratio - 1| "
            f"{dev:.1e}")
    np.savez_compressed(a.out, R_kpc=Rn, z_kpc=zn, log10_m_eV=LOG10_M, neff_cm3=neff / cm_inv3_to_eV3,
                        n_local_cm3=n_loc, n_n0_cm3=n_n0_cm3, r_max_kpc=float(r[-1]), model_digest=np.array(digest),
                        r_sun_kpc=R_SUN, check=np.array(note),
                        commit=np.array(os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()),
                        host=np.array(os.uname().nodename), seconds=dt)
    print(f"wrote {a.out}: {len(Rn)} x {len(zn)} points, {len(r)} distances, {len(LOG10_M)} masses in {dt:.0f} s; "
          f"{note}; n_local(Sun) = {n_loc[i_sun, 0]:.6g} cm^-3")


if __name__ == "__main__":
    main()
