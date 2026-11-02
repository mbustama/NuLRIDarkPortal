"""
mw_sky_density.py

Tabulates the Milky Way neutron density averaged over the sky as seen from the Sun,
    nbar(r) = (1/4 pi) int dOmega n(r, Omega),
the only property of the Galactic model that the Yukawa potential at the Sun needs:
    n_eff(m) = m^2 int d^3x n(x) exp(-m |x|) / (4 pi |x|) = int_0^inf m^2 r exp(-m r) nbar(r) dr
(dp_transit.galactic_effective_density). The model is dp_spatial.calc_MW_neutron_density,
symmetric under z -> -z and phi -> -phi, so the sky integral is 4 times one quadrant. The
quadrature is composite Gauss-Legendre (16 points per panel) with panels graded toward the
Galactic plane (mu = z / r breakpoints at z = 2 pc x 2^j) and toward the Galactic Centre's
direction (phi breakpoints at 1 mrad x 2^j), which resolves the thinnest structures (the H2 disc,
z_d = 45 pc; the bulge core, 75 pc) at every distance.

  python tools/mw_sky_density.py [--species neutron|electron] [--out ...] [--per-decade 200]

Writes r_kpc and nbar_cm3 (the pure Galactic model: no cosmic floor, which the solver applies)
with the model's source text digest. Run it again whenever calc_MW_neutron_density changes;
tests/test_transit.py checks the table against fresh quadrature. --species electron tabulates the
electron (= proton) density instead (dp_spatial.calc_MW_electron_density), for matter charges other
than the neutron's (dp_matter; src/data/mw_sky_density_electrons.npz).
"""
import argparse
import hashlib
import inspect
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

from dp_core import njit  # noqa: E402
import dp_spatial  # noqa: E402
from dp_spatial import calc_MW_neutron_density, calc_MW_species_density  # noqa: E402

SPECIES = {"neutron": 0, "electron": 1}
OUT = {0: "mw_sky_density.npz", 1: "mw_sky_density_electrons.npz"}

GL16_X, GL16_W = np.polynomial.legendre.leggauss(16)


def _breaks(scale_min, top, cap):
    """0, scale_min * 2^j below top, and top (all divided by cap's scale already)."""
    b = [0.0]
    x = scale_min
    while x < top:
        b.append(x)
        x *= 2.0
    b.append(top)
    return np.array(b)


@njit(cache=True)
def _sky_average(r_kpc, mu_b, phi_b, glx, glw, species):
    """(1/pi) int_0^1 dmu int_0^pi dphi n(r, theta = acos(mu), phi) on the given panels (dp_spatial.SPECIES_*)."""
    total = 0.0
    for i in range(len(mu_b) - 1):
        m_mid = 0.5 * (mu_b[i] + mu_b[i + 1])
        m_half = 0.5 * (mu_b[i + 1] - mu_b[i])
        for a in range(len(glx)):
            mu = m_mid + m_half * glx[a]
            theta = np.arccos(mu)
            acc_phi = 0.0
            for j in range(len(phi_b) - 1):
                p_mid = 0.5 * (phi_b[j] + phi_b[j + 1])
                p_half = 0.5 * (phi_b[j + 1] - phi_b[j])
                s = 0.0
                for b in range(len(glx)):
                    s += glw[b] * calc_MW_species_density(r_kpc, theta, p_mid + p_half * glx[b], species)
                acc_phi += s * p_half
            total += glw[a] * acc_phi * m_half
    return total / np.pi


def sky_average(r_kpc: float, species: int = 0) -> float:
    """nbar(r) in cm^-3 at r_kpc from the Sun, of neutrons (species 0) or electrons (1)."""
    mu_b = np.unique(np.minimum(_breaks(0.002, max(r_kpc, 0.002), 1.0) / max(r_kpc, 1e-300), 1.0))
    mu_b = mu_b[(mu_b >= 0.0) & (mu_b <= 1.0)]
    if mu_b[-1] < 1.0:
        mu_b = np.append(mu_b, 1.0)
    phi_b = _breaks(1e-3, np.pi, 1.0)
    return float(_sky_average(r_kpc, np.ascontiguousarray(mu_b), np.ascontiguousarray(phi_b), GL16_X, GL16_W,
                              int(species)))


def _body(fn) -> str:
    """A function's source without its decorator lines: compiler options do not change the model."""
    return "\n".join(line for line in inspect.getsource(fn).split("\n") if not line.lstrip().startswith("@"))


def model_digest(species: int = 0) -> str:
    """The density model's digest: the neutron model's (unchanged since the first table), or the electron
    model's (its components and electron fractions)."""
    if species == 0:
        src = _body(dp_spatial.calc_MW_neutron_density) + _body(dp_spatial._gas_disc_density)
        consts = repr((dp_spatial._MW_NEUTRONS_PER_NUCLEON_STARS, dp_spatial._MW_NEUTRONS_PER_ELECTRON_GAS,
                       dp_spatial._NUCLEONS_PER_CM3_PER_MSUN_PC3, dp_spatial._MW_NEUTRONS_PER_NUCLEON_COLD_GAS,
                       dp_spatial._HI_DISC, dp_spatial._H2_DISC))
    else:
        src = (_body(dp_spatial.calc_MW_electron_density) + _body(dp_spatial._MW_components)
               + _body(dp_spatial._gas_disc_density))
        consts = repr((dp_spatial._MW_ELECTRONS_PER_NUCLEON_STARS, dp_spatial._MW_ELECTRONS_PER_NUCLEON_COLD_GAS,
                       dp_spatial._NUCLEONS_PER_CM3_PER_MSUN_PC3, dp_spatial._HI_DISC, dp_spatial._H2_DISC))
    return hashlib.sha256((src + consts).encode()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--species", choices=sorted(SPECIES), default="neutron")
    ap.add_argument("--out", default=None, help="default: src/data/mw_sky_density[_electrons].npz")
    ap.add_argument("--per-decade", type=int, default=200)
    ap.add_argument("--r-min", type=float, default=1e-6, help="kpc")
    ap.add_argument("--r-max", type=float, default=2e4, help="kpc")
    args = ap.parse_args(argv)
    species = SPECIES[args.species]
    out = args.out or os.path.join(REPO, "src", "data", OUT[species])
    n = int(round(args.per_decade * np.log10(args.r_max / args.r_min))) + 1
    r = np.geomspace(args.r_min, args.r_max, n)
    t0 = time.perf_counter()
    nbar = np.array([sky_average(x, species) for x in r])
    np.savez_compressed(out, r_kpc=r, nbar_cm3=nbar, model_digest=np.array(model_digest(species)),
                        species=np.array(args.species))
    print(f"wrote {out}: {n} distances from {args.r_min:g} to {args.r_max:g} kpc in {time.perf_counter() - t0:.1f} s; "
          f"nbar(r_min) = {nbar[0]:.6g} cm^-3 (local {calc_MW_species_density(0.0, 0.0, 0.0, species):.6g})")


if __name__ == "__main__":
    main()
