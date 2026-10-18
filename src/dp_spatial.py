"""
dp_spatial.py

The spatial handover: Milky Way and Earth density models, PREM layer geometry and
the phase-exact step-averaged potential, the Earth's exterior, the per-step density-matrix
evolution, the pool workers and persistent pool, transit_to_earth (the Galaxy, the Sun and the
Earth's exterior beyond r_split, through dp_transit) and propagate_earth_spatial. Split
verbatim out of dark_portal_solver.py (HAS_TQDM references now read through dp_core, where
that mutable flag lives); the Milky Way trajectories were replaced on 2026-10-13.
"""
import multiprocessing

import numpy as np

import dp_core
from dp_core import (njit, HAS_NUMBA, numba, safe_print, tqdm, m_to_eV_inv, available_cores,
                     cm_inv3_to_eV3, GeV, G_F, calc_debye_mass_sq,
                     d_sun_eV_inv, d_moon_eV_inv, N_n_sun, N_n_earth, N_n_moon)
from dp_propagator import propagator_3x3
from dp_thermal import _GL10_X, _GL10_W

# =============================================================================
# SPATIAL HANDOVER HELPER FUNCTIONS
# =============================================================================

# Neutrons per nucleon of the Galaxy's stellar mass, and per electron of its hot gas.
# Stars: McKee, Parravano & Hollenbach (2015, arXiv:1509.05334) put the local stars and
# remnants at 33.4 Msun/pc^2 -- 81% visible stars, 15% white dwarfs, 4% brown dwarfs. With
# neutrons per nucleon (Y + Z)/2 = 0.146 for disc stars (Y ~ 0.275, Z ~ 0.017), 0.5 for
# carbon/oxygen white dwarfs and 0.142 for brown dwarfs, the mass-weighted fraction is 0.20.
# Hot gas: the Miller & Bregman (2013) profile is an ELECTRON density; fully ionized gas with
# Y ~ 0.26 and Z ~ 0.3 Zsun has (Y + Z)/2 / (1 - (Y + Z)/2) = 0.15 neutrons per electron.
# Until 2026-10-12 both were 0.5 (half of all nucleons as neutrons, right for rock but not
# for hydrogen-dominated matter), which made the Galactic potential ~2.5x (stars) and ~3.3x
# (gas) too strong.
_MW_NEUTRONS_PER_NUCLEON_STARS = 0.20
_MW_NEUTRONS_PER_ELECTRON_GAS = 0.15
# Nucleons per cm^3 in 1 Msun/pc^3: 1.98847e33 g / (3.08568e18 cm)^3 / m_p = 40.464.
_NUCLEONS_PER_CM3_PER_MSUN_PC3 = 40.464
# Cold disc gas: the HI and H2 discs of McMillan (2017, MNRAS 465, 76; arXiv:1608.00971),
#   rho = Sigma_0 / (4 z_d) exp(-R_m/R - R/R_d) sech^2(z / 2 z_d),
# with Sigma_0 in Msun/pc^2 INCLUDING helium and metals (so no further correction) -- HI:
# 53.1, R_d 7, R_m 4, z_d 0.085 kpc; H2: 2180, R_d 1.5, R_m 12, z_d 0.045 kpc. At the Sun they give
# 10 and 2 Msun/pc^2 (McKee et al. 2015 measure 13.7 in all gas). Neutrons per nucleon (Y + Z)/2
# for protosolar Y = 0.2703, Z = 0.0142 (Asplund et al. 2009). Absent before 2026-10-12.
_MW_NEUTRONS_PER_NUCLEON_COLD_GAS = 0.142
_HI_DISC = (53.1, 7.0, 4.0, 0.085)      # Sigma_0 [Msun/pc^2], R_d, R_m, z_d [kpc]
_H2_DISC = (2180.0, 1.5, 12.0, 0.045)


@njit("float64(float64, float64, float64, float64, float64, float64)", fastmath=True, error_model="numpy", cache=True)
def _gas_disc_density(R_kpc: float, z_kpc: float, Sigma_0: float, R_d: float, R_m: float,
                      z_d: float) -> float:
    """Mass density (Msun/pc^3) of a McMillan (2017) gas disc at cylindrical radius R_kpc and
    height z_kpc. sech^2 is written as 4 e^{-2|x|} / (1 + e^{-2|x|})^2, which cannot overflow,
    and the central hole exp(-R_m/R) is taken as zero on the axis."""
    if R_kpc < 1e-3:
        return 0.0
    x = np.abs(z_kpc) / (2.0 * z_d)
    e = np.exp(-2.0 * x)
    sech2 = 4.0 * e / ((1.0 + e) * (1.0 + e))
    return Sigma_0 / (4.0 * z_d * 1e3) * np.exp(-R_m / R_kpc - R_kpc / R_d) * sech2


@njit("float64(float64, float64, float64)", fastmath=True, error_model="numpy", cache=True)
def calc_MW_neutron_density(r_kpc: float, theta: float, phi: float) -> float:
    """Calculates the Milky Way neutron number density (cm^-3) at a given location.

    Arguments: r_kpc, theta, phi -- position relative to the Sun (kpc, rad, rad). The
    stellar bulge and discs (McMillan) are mass densities in Msun/pc^3, converted to
    neutrons with _MW_NEUTRONS_PER_NUCLEON_STARS; the hot-gas halo (Miller & Bregman 2013)
    is an electron density, converted with _MW_NEUTRONS_PER_ELECTRON_GAS; the cold HI and H2
    discs (McMillan 2017) are mass densities, converted with _MW_NEUTRONS_PER_NUCLEON_COLD_GAS
    (see above)."""
    R_0 = 8.33
    sth = np.sin(theta)
    # The law-of-cosines argument is mathematically >= 0, but evaluating it as a
    # difference suffers catastrophic cancellation on the exact line through the
    # galactic center (r ~ R_0, theta = pi/2, phi = 0 -- which IS one of the six
    # default Milky Way trajectories), and under fastmath the reassociated sum can
    # land a hair below zero: sqrt then yields NaN, which propagated into a
    # ZeroDivisionError from the bulge term's power. Any negative value here is pure
    # rounding, so clamp to zero.
    R_gal_sq = (r_kpc * sth)**2 + R_0**2 - 2.0 * r_kpc * R_0 * sth * np.cos(phi)
    R_gal = np.sqrt(max(R_gal_sq, 0.0))
    z_gal = r_kpc * np.cos(theta)
    r_gal = np.sqrt(R_gal**2 + z_gal**2)
    
    rp = np.sqrt(R_gal**2 + (z_gal / 0.5)**2)
    rho_bulge = (94.1 / (1.0 + rp / 0.075)**1.8) * np.exp(-(rp / 2.1)**2)
    rho_thin = (753.0 / 600.0) * np.exp(-np.abs(z_gal) / 0.3 - np.abs(R_gal) / 3.0)
    rho_thick = (182.0 / 1800.0) * np.exp(-np.abs(z_gal) / 0.9 - np.abs(R_gal) / 3.5)
    n_e_gas = 0.46 * (1.0 + (r_gal / 0.35)**2)**(-1.065)    # Miller & Bregman 2013, cm^-3

    rho_cold = (_gas_disc_density(R_gal, z_gal, _HI_DISC[0], _HI_DISC[1], _HI_DISC[2], _HI_DISC[3])
                + _gas_disc_density(R_gal, z_gal, _H2_DISC[0], _H2_DISC[1], _H2_DISC[2], _H2_DISC[3]))

    n_n_MW = ((rho_bulge + rho_thin + rho_thick) * _NUCLEONS_PER_CM3_PER_MSUN_PC3
              * _MW_NEUTRONS_PER_NUCLEON_STARS
              + rho_cold * _NUCLEONS_PER_CM3_PER_MSUN_PC3 * _MW_NEUTRONS_PER_NUCLEON_COLD_GAS
              + n_e_gas * _MW_NEUTRONS_PER_ELECTRON_GAS)
    return n_n_MW


# Electrons (= protons) per nucleon of the same components, for matter charges other than the neutron's
# (docs/dev/SCOPE_WIDENING_PLAN.md; dp_matter): in neutral matter every nucleon that is not a neutron is a
# proton, with its electron; the hot gas's profile is the electron density itself.
_MW_ELECTRONS_PER_NUCLEON_STARS = 1.0 - _MW_NEUTRONS_PER_NUCLEON_STARS
_MW_ELECTRONS_PER_NUCLEON_COLD_GAS = 1.0 - _MW_NEUTRONS_PER_NUCLEON_COLD_GAS
SPECIES_NEUTRON = 0
SPECIES_ELECTRON = 1


@njit(fastmath=True, error_model="numpy", cache=True)
def _MW_components(r_kpc: float, theta: float, phi: float):
    """(rho_stars [Msun/pc^3], rho_cold_gas [Msun/pc^3], n_e of the hot gas [cm^-3]) at a position relative
    to the Sun (kpc, rad, rad): calc_MW_neutron_density's components before their conversion to neutrons, with
    the same expressions (a regression test holds the two equal; that function is left as it was so the
    neutron path and its tables' digest stay unchanged)."""
    R_0 = 8.33
    sth = np.sin(theta)
    R_gal_sq = (r_kpc * sth)**2 + R_0**2 - 2.0 * r_kpc * R_0 * sth * np.cos(phi)
    R_gal = np.sqrt(max(R_gal_sq, 0.0))
    z_gal = r_kpc * np.cos(theta)
    r_gal = np.sqrt(R_gal**2 + z_gal**2)
    rp = np.sqrt(R_gal**2 + (z_gal / 0.5)**2)
    rho_bulge = (94.1 / (1.0 + rp / 0.075)**1.8) * np.exp(-(rp / 2.1)**2)
    rho_thin = (753.0 / 600.0) * np.exp(-np.abs(z_gal) / 0.3 - np.abs(R_gal) / 3.0)
    rho_thick = (182.0 / 1800.0) * np.exp(-np.abs(z_gal) / 0.9 - np.abs(R_gal) / 3.5)
    n_e_gas = 0.46 * (1.0 + (r_gal / 0.35)**2)**(-1.065)    # Miller & Bregman 2013, cm^-3
    rho_cold = (_gas_disc_density(R_gal, z_gal, _HI_DISC[0], _HI_DISC[1], _HI_DISC[2], _HI_DISC[3])
                + _gas_disc_density(R_gal, z_gal, _H2_DISC[0], _H2_DISC[1], _H2_DISC[2], _H2_DISC[3]))
    return rho_bulge + rho_thin + rho_thick, rho_cold, n_e_gas


@njit("float64(float64, float64, float64)", fastmath=True, error_model="numpy", cache=True)
def calc_MW_electron_density(r_kpc: float, theta: float, phi: float) -> float:
    """The Milky Way's electron (= proton) number density (cm^-3) at a position relative to the Sun (kpc, rad,
    rad): calc_MW_neutron_density's model, its stars and cold gas at their electrons per nucleon and the hot
    gas at its own electron density."""
    rho_stars, rho_cold, n_e_gas = _MW_components(r_kpc, theta, phi)
    return (rho_stars * _NUCLEONS_PER_CM3_PER_MSUN_PC3 * _MW_ELECTRONS_PER_NUCLEON_STARS
            + rho_cold * _NUCLEONS_PER_CM3_PER_MSUN_PC3 * _MW_ELECTRONS_PER_NUCLEON_COLD_GAS
            + n_e_gas)


@njit("float64(float64, float64, float64, int64)", fastmath=True, error_model="numpy", cache=True)
def calc_MW_species_density(r_kpc: float, theta: float, phi: float, species: int) -> float:
    """calc_MW_neutron_density (species SPECIES_NEUTRON) or calc_MW_electron_density (SPECIES_ELECTRON),
    cm^-3, for the Galactic tables of either species (tools/mw_sky_density.py, tools/mw_neff_tables.py)."""
    if species == SPECIES_ELECTRON:
        return calc_MW_electron_density(r_kpc, theta, phi)
    return calc_MW_neutron_density(r_kpc, theta, phi)

@njit("float64(float64)", fastmath=True, error_model="numpy", cache=True)
def calc_earth_density(r_km: float) -> float:
    """Returns the Earth's density at a distance r_km from the center using PREM layers."""
    x = r_km / 6371.0
    if r_km <= 1221.5: return 13.0885 - 8.8381*x*x
    elif r_km <= 3480.0: return 12.5815 - 1.2638*x - 3.6426*x*x - 5.5281*x*x*x
    elif r_km <= 5701.0: return 7.9565 - 6.4761*x + 5.5283*x*x - 3.0807*x*x*x
    elif r_km <= 5771.0: return 5.3197 - 1.4836*x
    elif r_km <= 5971.0: return 11.2494 - 8.0298*x
    elif r_km <= 6151.0: return 7.1089 - 3.8045*x
    elif r_km <= 6346.6: return 2.691 + 0.6924*x
    elif r_km <= 6356.0: return 2.9
    elif r_km <= 6368.0: return 2.6
    elif r_km <= 6371.0: return 1.02
    else: return 0.0


# The same PREM, as data: shell edges (km) and, per shell (edge_s, edge_s+1], the coefficients
# of rho(x) = c0 + c1 x + c2 x^2 + c3 x^3 (g/cm^3, x = r/6371 km) that calc_earth_density
# spells out branch by branch. The Yukawa integral below needs the polynomials themselves;
# a regression test holds the two copies equal.
_PREM_EDGES_KM = np.array([0.0, 1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0,
                           6346.6, 6356.0, 6368.0, 6371.0])
_PREM_RHO_COEFFS = np.array([[13.0885, 0.0, -8.8381, 0.0],
                             [12.5815, -1.2638, -3.6426, -5.5281],
                             [7.9565, -6.4761, 5.5283, -3.0807],
                             [5.3197, -1.4836, 0.0, 0.0],
                             [11.2494, -8.0298, 0.0, 0.0],
                             [7.1089, -3.8045, 0.0, 0.0],
                             [2.691, 0.6924, 0.0, 0.0],
                             [2.9, 0.0, 0.0, 0.0],
                             [2.6, 0.0, 0.0, 0.0],
                             [1.02, 0.0, 0.0, 0.0]])


# Neutrons per gram of Earth matter, by PREM region: (1 - Y_e) nucleons per gram, where Y_e is
# the electron (= proton) fraction per nucleon -- 0.4656 in the iron-nickel core and 0.4957
# in the silicate mantle and crust, the values used with PREM in neutrino-oscillation
# tomography -- and 8/18.015 per molecular mass for the water (ice) of PREM's top layer.
# Until 2026-10-12 every layer used 0.5 (3.011e23 per gram): the core was 6.9% low, the
# mantle 0.9% low and the ocean 12.6% high.
_AVOGADRO = 6.02214076e23                       # nucleons per gram (1/m_u)
_EARTH_YE_CORE = 0.4656
_EARTH_YE_MANTLE = 0.4957
_EARTH_R_CORE_KM = 3480.0
_EARTH_R_OCEAN_FLOOR_KM = 6368.0


@njit("float64(float64)", fastmath=True, error_model="numpy", cache=True)
def earth_neutrons_per_gram(r_km: float) -> float:
    """Neutrons per gram of PREM matter at geocentric radius r_km: core, mantle and crust
    from their electron fractions, the top layer as water (see the constants above)."""
    if r_km <= _EARTH_R_CORE_KM:
        return (1.0 - _EARTH_YE_CORE) * _AVOGADRO
    elif r_km <= _EARTH_R_OCEAN_FLOOR_KM:
        return (1.0 - _EARTH_YE_MANTLE) * _AVOGADRO
    else:
        return 8.0 / 18.015 * _AVOGADRO


@njit("float64(float64)", fastmath=True, error_model="numpy", cache=True)
def earth_electrons_per_gram(r_km: float) -> float:
    """Electrons (= protons) per gram of PREM matter at geocentric radius r_km: Y_e nucleons per gram in the
    core, mantle and crust, 10/18.015 per molecular mass for the water of the top layer -- the complement of
    earth_neutrons_per_gram, for matter charges other than the neutron's (dp_matter)."""
    if r_km <= _EARTH_R_CORE_KM:
        return _EARTH_YE_CORE * _AVOGADRO
    elif r_km <= _EARTH_R_OCEAN_FLOOR_KM:
        return _EARTH_YE_MANTLE * _AVOGADRO
    else:
        return 10.0 / 18.015 * _AVOGADRO


# A radius inside each of the regions of constant composition (core; mantle and crust; the top layer's
# water): three numbers give the charge per gram of any matter charge (earth_charge_per_gram).
_EARTH_REGION_SAMPLE_KM = (1000.0, 5000.0, 6370.0)


def earth_charge_per_gram(q_n: float = 1.0, q_pe: float = 0.0) -> np.ndarray:
    """(core, mantle and crust, top layer) charge per gram of the matter charge q_n n_n + q_pe n_e (neutral
    matter: n_p = n_e). The neutron charge (1, 0) gives earth_neutrons_per_gram's values bit for bit."""
    return np.array([q_n * earth_neutrons_per_gram(r) + q_pe * earth_electrons_per_gram(r)
                     for r in _EARTH_REGION_SAMPLE_KM])


@njit(fastmath=True, error_model="numpy", cache=True)
def _earth_region_value(r_km: float, v: np.ndarray) -> float:
    """v[0] in the core, v[1] in the mantle and crust, v[2] in the top layer (earth_neutrons_per_gram's
    regions)."""
    if r_km <= _EARTH_R_CORE_KM:
        return v[0]
    elif r_km <= _EARTH_R_OCEAN_FLOOR_KM:
        return v[1]
    return v[2]


@njit("float64(float64, float64)", fastmath=True, error_model="numpy", cache=True)
def get_baseline_earth(d_km: float, costh: float) -> float:
    """Path length (km) inside the Earth from the surface to a detector at depth d_km, for a
    neutrino arriving at zenith cosine costh (-1: up through the centre, 0: horizontal).

    The chord from the detector, at radius f2 = R - d, meets the surface where
    u^2 + 2 f2 costh u - d (2R - d) = 0. Until 2026-10-12 this was evaluated with the d^2
    terms dropped, which made the chords too long by up to 8 m (horizontal), starting them in
    vacuum above the surface. The root is written without cancellation for either sign of
    costh."""
    R_E = 6371.0
    f2 = R_E - d_km
    sq = np.sqrt(d_km * (2.0 * R_E - d_km) + (f2 * costh)**2)
    if costh <= 0.0:
        return sq - f2 * costh
    return d_km * (2.0 * R_E - d_km) / (sq + f2 * costh)

# =============================================================================
# EXTRACTED NUMBA KERNELS FOR PROGRESS TRACKING
# =============================================================================


@njit(fastmath=True, error_model="numpy", cache=True)
def chord_radius(f1: float, f2: float, costh: float) -> float:
    """Geocentric radius sqrt(f1^2 + f2^2 + 2 f1 f2 costh) of the chord point at distance f1
    from the detector (radius f2), written as a sum of squares,
        (f1 + f2 costh)^2 + f2^2 (1 - costh)(1 + costh),
    which cannot cancel. The expanded form loses ~1e-8 km^2 to rounding at the centre of the
    vertical chord, where the radius vanishes: on Tycho's fused multiply-adds it rounded to
    exactly zero at a Gauss-Legendre node 1e-7 km from the centre, and the Yukawa terms (m/r)
    divided by it (found 2026-10-13; latent since the Yukawa integral of 2026-10-12)."""
    a = f1 + f2 * costh
    return np.sqrt(a * a + f2 * f2 * ((1.0 - costh) * (1.0 + costh)))


@njit(fastmath=True, error_model="numpy", cache=True)
def spatial_total_density_earth(l_km: float, baseline_km: float, costh: float,
                                detector_depth_km: float, n_n_amb_eV3: float) -> float:
    """Total neutron number density (eV^3) at path position l_km along an Earth chord:
    the PREM profile at the corresponding geocentric radius plus the ambient
    (galactic + Sun/Moon Yukawa) background. Shared by step_earth_transit and the GPU
    spatial path's density-sequence precompute."""
    f1 = baseline_km - l_km
    f2 = 6371.0 - detector_depth_km
    r_center_km = chord_radius(f1, f2, costh)

    rho_earth_g = calc_earth_density(r_center_km)
    n_n_earth_cm3 = rho_earth_g * earth_neutrons_per_gram(r_center_km)

    return n_n_amb_eV3 + n_n_earth_cm3 * cm_inv3_to_eV3


@njit(fastmath=True, error_model="numpy", cache=True)
def spatial_total_density_earth_q(l_km: float, baseline_km: float, costh: float,
                                  detector_depth_km: float, n_n_amb_eV3: float, cpg: np.ndarray) -> float:
    """spatial_total_density_earth for a general matter charge: the PREM density times its region's charge per
    gram (cpg, earth_charge_per_gram), plus the ambient charge density n_n_amb_eV3 (eV^3). A separate function,
    so the neutron path keeps its compiled form (and its last bits)."""
    f1 = baseline_km - l_km
    f2 = 6371.0 - detector_depth_km
    r_center_km = chord_radius(f1, f2, costh)
    n_q_earth_cm3 = calc_earth_density(r_center_km) * _earth_region_value(r_center_km, cpg)
    return n_n_amb_eV3 + n_q_earth_cm3 * cm_inv3_to_eV3


# =============================================================================
# THE EARTH'S OWN POTENTIAL: THE SPHERICAL YUKAWA INTEGRAL
# =============================================================================
# The dark-portal potential of a body is a Yukawa integral over its neutrons,
#     V(x) = Coef m^2 int d^3x' n(x') exp(-m |x - x'|) / (4 pi |x - x'|),   m = m_eff,
# where Coef (built in propagate_earth_spatial) already carries 1/m^2, so V = Coef n_eff with
# n_eff the neutron density smoothed over the mediator range 1/m. Until 2026-10-12 the Earth
# chords used the contact limit n_eff = n(r). That limit holds while 1/m is far below every
# length over which PREM varies (a 1e-9 eV mediator already reaches only 200 m), but a
# mediator whose range rivals the Earth's radius sees the enclosed neutrons instead, and the
# contact form then overstates the Earth's potential by ~1/(m R)^2 -- 1e13 at 1e-20 eV.
#
# For a spherical profile the angular integral is elementary and the potential becomes
# one-dimensional -- the method of Magnus (docs/dev/measurements/issue192_sec6_numbers/D/lri.py,
# for the Sun) and of NuOscProbExact (notebooks/20_arbitrary_hamiltonian.ipynb, for the Earth):
#     n_eff(r) = m^2 [ e^{-mr} I_in(r)/r + shc(mr) I_out(r) ],   shc(x) = sinh(x)/x,
#     I_in(r) = int_0^r r'^2 n shc(m r') dr',   I_out(r) = int_r^R r' n e^{-m r'} dr'.
# Here it is written with f(r) = r n(r) and decaying exponentials only,
#     n_eff(r) = (m/r) [ P(r) + (1 - e^{-2mr}) Q(r)/2 ],
#     P(r) = int_0^r g(r') e^{-m(r - r')} dr',   g = f (1 - e^{-2mr'})/2,
#     Q(r) = int_r^R f(r') e^{-m(r' - r)} dr',
# since e^{mR} itself is 10^(10^19) at MeV masses. Two evaluations cover every mass, both
# exact for PREM's piecewise polynomials:
#   'direct' (m R < 50): P and Q accumulated over sub-intervals no longer than 1/m with
#     10-point Gauss-Legendre, Q inward from the surface (as NuOscProbExact does: written as
#     the total minus the interior part, it would vanish into round-off);
#   'heavy' (m R >= 50): integrating by parts, exact for a polynomial f, gives in a shell (a, b]
#         n_eff(r) = n(r) + [f''(r)/m^2 + f''''(r)/m^4]/r + (m/r) E(r),
#         E(r) = e^{-m(r-a)} delta + (1 - e^{-2mr})/2 e^{-m(b-r)} gamma,
#     with per-shell constants delta = P(a) - D_-(a)/2 - e^{-2ma} D_+(a)/2 and
#     gamma = Q(b) - D_+(b), where D_+-(x) = sum_k (+-1)^k f^(k)(x)/m^(k+1). (delta folds the
#     image term e^{-m(r+a)} into the first: kept apart, the two nearly cancel in the centre
#     shell, where fastmath's fused multiply-adds round them differently and 1/r magnifies
#     the residue. For PREM, whose centre density is even in r, delta is exactly zero
#     there.) E holds the boundary layers: half-way between the two densities at a jump,
#     relaxing over 1/m. Away from them every exponential underflows and n_eff is n(r) to
#     O((m R)^-2): at MeV masses the result is bit-identical to the contact form.
# A table is the tuple (par, edges, F, abg, nodes): par = [mode, m (1/km), R (km), P(R)];
# shell edges (km); F[s, k], the coefficients of f(r) = sum_k F[s, k] r^k (cm^-3 km); the
# heavy-mode delta, D_+(a)/2 (kept for diagnostics) and gamma per shell; the direct-mode nodes
# (r, P, Q, shell). Densities are in cm^-3 and lengths in km throughout.
_YUKAWA_CONTACT = 0
_YUKAWA_HEAVY = 1
_YUKAWA_DIRECT = 2
_YUKAWA_HEAVY_MIN_MR = 50.0
_KM_IN_EV_INV = 1e3 * m_to_eV_inv            # 1 km in eV^-1, so m [1/km] = m [eV] * this
# Boundary layers thinner than this fraction of an Earth step are left unresolved along the
# chords: each contributes below this fraction of the step's phase.
_YUKAWA_LAYER_REL_TOL = 1e-10


def _yukawa_D(F_s: np.ndarray, x: float, m: float, sign: float) -> float:
    """D_+-(x) = sum_k (+-1)^k f^(k)(x) / m^(k+1) for f(r) = sum_k F_s[k] r^k (degree <= 4).

    Arguments: F_s -- the shell's five coefficients; x -- radius (km); m -- mediator mass
    (1/km); sign -- +1 or -1. Evaluated in Horner form in sign/m, so no power of m is formed.
    """
    f0 = F_s[0] + x * (F_s[1] + x * (F_s[2] + x * (F_s[3] + x * F_s[4])))
    f1 = F_s[1] + x * (2.0 * F_s[2] + x * (3.0 * F_s[3] + x * 4.0 * F_s[4]))
    f2 = 2.0 * F_s[2] + x * (6.0 * F_s[3] + x * 12.0 * F_s[4])
    f3 = 6.0 * F_s[3] + x * 24.0 * F_s[4]
    f4 = 24.0 * F_s[4]
    u = sign / m
    return (f0 + u * (f1 + u * (f2 + u * (f3 + u * f4)))) / m


def yukawa_profile_table(edges_km, fcoef, m_km, mode=None):
    """Builds the Yukawa table of a spherical profile (see the section notes above).

    Arguments:
      edges_km: shell edges (km), increasing from 0 to the body's radius R.
      fcoef: (n_shells, 5) coefficients of f(r) = r n(r) = sum_k fcoef[s, k] r^k (cm^-3 km),
        the density n being a polynomial of degree <= 3 on each shell (edge_s, edge_s+1].
      m_km: mediator mass (1/km); None gives the contact table (n_eff = n).
      mode: None (choose by m R), or force _YUKAWA_HEAVY / _YUKAWA_DIRECT (tests).

    Returns: the tuple (par, edges, F, abg, nodes) read by the numba kernels below.
    """
    edges = np.ascontiguousarray(edges_km, dtype=np.float64)
    F = np.ascontiguousarray(fcoef, dtype=np.float64)
    S = len(edges) - 1
    R = float(edges[-1])
    abg = np.zeros((S, 3))
    nodes = np.zeros((1, 4))
    if m_km is None:
        return (np.array([_YUKAWA_CONTACT, 0.0, R, 0.0]), edges, F, abg, nodes)
    m = float(m_km)
    if not (np.isfinite(m) and m > 0.0):
        raise ValueError(f"The Yukawa mediator mass must be positive and finite (got {m_km} /km).")
    if mode is None:
        mode = _YUKAWA_HEAVY if m * R >= _YUKAWA_HEAVY_MIN_MR else _YUKAWA_DIRECT
    if mode == _YUKAWA_HEAVY:
        P = 0.0
        for s in range(S):
            a, b = edges[s], edges[s + 1]
            Dm_a, Dm_b = _yukawa_D(F[s], a, m, -1.0), _yukawa_D(F[s], b, m, -1.0)
            Dp_a, Dp_b = _yukawa_D(F[s], a, m, 1.0), _yukawa_D(F[s], b, m, 1.0)
            abg[s, 0] = (P - 0.5 * Dm_a) - np.exp(-2.0 * m * a) * (0.5 * Dp_a)
            abg[s, 1] = 0.5 * Dp_a
            e_ba = np.exp(-m * (b - a))
            P = (e_ba * P + 0.5 * (Dm_b - e_ba * Dm_a)
                 - 0.5 * (np.exp(-m * (b + a)) * Dp_a - np.exp(-2.0 * m * b) * Dp_b))
        P_R = P
        Q = 0.0
        for s in range(S - 1, -1, -1):
            a, b = edges[s], edges[s + 1]
            Dp_a, Dp_b = _yukawa_D(F[s], a, m, 1.0), _yukawa_D(F[s], b, m, 1.0)
            abg[s, 2] = Q - Dp_b
            e_ba = np.exp(-m * (b - a))
            Q = e_ba * Q + Dp_a - e_ba * Dp_b
    elif mode == _YUKAWA_DIRECT:
        r_list, s_list = [0.0], []
        for s in range(S):
            k = max(1, int(np.ceil(m * (edges[s + 1] - edges[s]))))
            r_list.extend(np.linspace(edges[s], edges[s + 1], k + 1)[1:])
            s_list.extend([s] * k)
        J = len(s_list)
        nodes = np.zeros((J + 1, 4))
        nodes[:, 0] = r_list
        nodes[:J, 3] = s_list
        for j in range(J):                    # P outward from the centre
            s, lo, hi = s_list[j], nodes[j, 0], nodes[j + 1, 0]
            half, x = 0.5 * (hi - lo), 0.5 * (hi + lo) + 0.5 * (hi - lo) * _GL10_X
            f = F[s, 0] + x * (F[s, 1] + x * (F[s, 2] + x * (F[s, 3] + x * F[s, 4])))
            g = f * (-0.5 * np.expm1(-2.0 * m * x))
            nodes[j + 1, 1] = (np.exp(-m * (hi - lo)) * nodes[j, 1]
                               + half * np.sum(_GL10_W * g * np.exp(-m * (hi - x))))
        for j in range(J - 1, -1, -1):        # Q inward from the surface
            s, lo, hi = s_list[j], nodes[j, 0], nodes[j + 1, 0]
            half, x = 0.5 * (hi - lo), 0.5 * (hi + lo) + 0.5 * (hi - lo) * _GL10_X
            f = F[s, 0] + x * (F[s, 1] + x * (F[s, 2] + x * (F[s, 3] + x * F[s, 4])))
            nodes[j, 2] = (np.exp(-m * (hi - lo)) * nodes[j + 1, 2]
                           + half * np.sum(_GL10_W * f * np.exp(-m * (x - lo))))
        P_R = nodes[J, 1]
    else:
        raise ValueError(f"Unknown Yukawa table mode {mode}.")
    return (np.array([float(mode), m, R, P_R]), edges, F, abg, nodes)


def _prem_f_coeffs(cpg=None) -> np.ndarray:
    """f(r) = r n(r) coefficients (cm^-3 km) of the PREM neutron profile, shell by shell:
    the rho(x) polynomials of _PREM_RHO_COEFFS times earth_neutrons_per_gram, which is
    constant on every shell (its regions end on PREM boundaries). cpg: None, or the charge per
    gram of each region (earth_charge_per_gram) for a general matter charge's profile."""
    F = np.zeros((len(_PREM_EDGES_KM) - 1, 5))
    for s in range(len(F)):
        mid = 0.5 * (_PREM_EDGES_KM[s] + _PREM_EDGES_KM[s + 1])
        npg = earth_neutrons_per_gram(mid) if cpg is None else _earth_region_value(mid, cpg)
        c = _PREM_RHO_COEFFS[s]
        F[s, 1:] = npg * np.array([c[0], c[1] / 6371.0, c[2] / 6371.0**2, c[3] / 6371.0**3])
    return F


_PREM_F_COEFFS = _prem_f_coeffs()


def earth_yukawa_table(m_eff_eV, mode=None, cpg=None):
    """The Earth's Yukawa table (PREM neutrons, see the section notes above).

    Arguments: m_eff_eV -- the mediator's effective mass sqrt(m_Zd^2 + m_D^2) (eV), or None
    for the contact table (the pre-2026-10-12 local density); mode -- None, or a forced
    _YUKAWA_HEAVY / _YUKAWA_DIRECT (tests); cpg -- None (neutrons), or the charge per gram of
    each region (earth_charge_per_gram) for a general matter charge: the table is then that
    charge's, and its par array carries a flag and cpg after the four Yukawa numbers
    (par[4] = 1, par[5:8] = cpg; _yk_charged), which the charged chord functions read.
    Returns the table tuple.
    """
    m_km = None if m_eff_eV is None else float(m_eff_eV) * _KM_IN_EV_INV
    if cpg is None:
        return yukawa_profile_table(_PREM_EDGES_KM, _PREM_F_COEFFS, m_km, mode)
    cpg = np.ascontiguousarray(cpg, dtype=np.float64)
    if cpg.shape != (3,):
        raise ValueError(f"cpg must hold the charge per gram of the core, the mantle and the top layer, got {cpg!r}")
    par, edges, F, abg, nodes = yukawa_profile_table(_PREM_EDGES_KM, _prem_f_coeffs(cpg), m_km, mode)
    return (np.concatenate([par, [1.0], cpg]), edges, F, abg, nodes)


def yk_charged(yk) -> bool:
    """Whether an Earth table carries a general matter charge (earth_yukawa_table's cpg)."""
    par = yk[0]
    return len(par) > 4 and par[4] != 0.0


@njit(fastmath=True, error_model="numpy", cache=True)
def _yk_shell(r: float, edges: np.ndarray) -> int:
    """Index s of the shell (edges[s], edges[s+1]] holding r, with calc_earth_density's
    convention that a boundary radius belongs to the shell below it."""
    S = len(edges) - 1
    for s in range(S - 1):
        if r <= edges[s + 1]:
            return s
    return S - 1


@njit(fastmath=True, error_model="numpy", cache=True)
def _yk_delta_heavy(r: float, yk) -> float:
    """Heavy-mode n_eff(r) - n(r) (cm^-3) for 0 <= r <= R: the polynomial correction plus the
    boundary layers E(r) of the section notes. Arguments: r -- radius (km); yk -- a table."""
    par, edges, F, abg, nodes = yk
    m = par[1]
    s = _yk_shell(r, edges)
    a = edges[s]
    b = edges[s + 1]
    rr = max(r, 1e-9)
    f2 = 2.0 * F[s, 2] + rr * (6.0 * F[s, 3] + rr * 12.0 * F[s, 4])
    f4 = 24.0 * F[s, 4]
    im2 = 1.0 / (m * m)
    E = (np.exp(-m * (rr - a)) * abg[s, 0]
         - 0.5 * np.expm1(-2.0 * m * rr) * np.exp(-m * (b - rr)) * abg[s, 2])
    return (f2 + f4 * im2) * im2 / rr + (m / rr) * E


@njit(fastmath=True, error_model="numpy", cache=True)
def _yk_direct(r: float, yk) -> float:
    """Direct-mode n_eff(r) (cm^-3) for 0 <= r <= R: P and Q at the node below and above r,
    plus Gauss-Legendre over the partial sub-interval. Arguments: r (km); yk -- a table."""
    par, edges, F, abg, nodes = yk
    m = par[1]
    rr = max(r, 1e-9)
    lo = 0
    hi = nodes.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if nodes[mid, 0] <= rr:
            lo = mid
        else:
            hi = mid
    j = lo
    rj = nodes[j, 0]
    rj1 = nodes[j + 1, 0]
    s = int(nodes[j, 3])
    hp = 0.5 * (rr - rj)
    mp = 0.5 * (rr + rj)
    hq = 0.5 * (rj1 - rr)
    mq = 0.5 * (rj1 + rr)
    acc_p = 0.0
    acc_q = 0.0
    for q in range(len(_GL10_X)):
        x = mp + hp * _GL10_X[q]
        fx = F[s, 0] + x * (F[s, 1] + x * (F[s, 2] + x * (F[s, 3] + x * F[s, 4])))
        acc_p += _GL10_W[q] * fx * (-0.5 * np.expm1(-2.0 * m * x)) * np.exp(-m * (rr - x))
        y = mq + hq * _GL10_X[q]
        fy = F[s, 0] + y * (F[s, 1] + y * (F[s, 2] + y * (F[s, 3] + y * F[s, 4])))
        acc_q += _GL10_W[q] * fy * np.exp(-m * (y - rr))
    P = np.exp(-m * (rr - rj)) * nodes[j, 1] + hp * acc_p
    Q = np.exp(-m * (rj1 - rr)) * nodes[j + 1, 2] + hq * acc_q
    return (m / rr) * (P - 0.5 * np.expm1(-2.0 * m * rr) * Q)


@njit(fastmath=True, error_model="numpy", cache=True)
def _yk_extra_cm3(r: float, yk) -> float:
    """The part of n_eff (cm^-3) that the contact density n(r) leaves out: zero for the
    contact table, n_eff - n in heavy mode, all of n_eff in direct mode, and the exterior
    potential (m/r) e^{-m(r-R)} P(R) outside the body, where n = 0. Arguments: r (km), yk."""
    par = yk[0]
    mode = int(par[0])
    if mode == _YUKAWA_CONTACT:
        return 0.0
    m = par[1]
    R = par[2]
    if r > R:
        return (m / r) * np.exp(-m * (r - R)) * par[3]
    if mode == _YUKAWA_DIRECT:
        return _yk_direct(r, yk)
    return _yk_delta_heavy(r, yk)


@njit(fastmath=True, error_model="numpy", cache=True)
def yukawa_profile_density(r: float, yk) -> float:
    """n_eff(r) (cm^-3) of the profile a table was built from, its local density taken from the
    table's own polynomials. Arguments: r -- radius (km); yk -- a yukawa_profile_table."""
    par = yk[0]
    mode = int(par[0])
    if mode == _YUKAWA_DIRECT or r > par[2]:
        return _yk_extra_cm3(r, yk)
    F = yk[2]
    s = _yk_shell(r, yk[1])
    rr = max(r, 1e-9)
    n = F[s, 0] / rr + F[s, 1] + rr * (F[s, 2] + rr * (F[s, 3] + rr * F[s, 4]))
    if mode == _YUKAWA_HEAVY:
        n += _yk_delta_heavy(r, yk)
    return n


@njit(fastmath=True, error_model="numpy", cache=True)
def earth_neutron_density_cm3(r_km: float, yk) -> float:
    """The Earth's effective neutron density n_eff (cm^-3) at geocentric radius r_km: the
    density whose contact potential Coef * n_eff equals the Yukawa potential of all of PREM's
    neutrons. Arguments: r_km -- radius (km); yk -- an earth_yukawa_table. The local part is
    calc_earth_density * earth_neutrons_per_gram, exactly as the contact form computes it."""
    mode = int(yk[0][0])
    if mode == _YUKAWA_DIRECT or r_km > yk[0][2]:
        return _yk_extra_cm3(r_km, yk)
    n = calc_earth_density(r_km) * earth_neutrons_per_gram(r_km)
    if mode == _YUKAWA_HEAVY:
        d = _yk_delta_heavy(r_km, yk)
        if d != 0.0:
            n = n + d
    return n


@njit(fastmath=True, error_model="numpy", cache=True)
def earth_total_density(l_km: float, baseline_km: float, costh: float,
                        detector_depth_km: float, n_n_amb_eV3: float, yk) -> float:
    """spatial_total_density_earth with the Earth's own neutrons entering through their
    Yukawa integral (eV^3). Arguments: as spatial_total_density_earth, plus yk -- an
    earth_yukawa_table. Where the Yukawa correction underflows (always, at MeV masses) the
    result is spatial_total_density_earth's, bit for bit."""
    mode = int(yk[0][0])
    f1 = baseline_km - l_km
    f2 = 6371.0 - detector_depth_km
    if mode == _YUKAWA_DIRECT:
        r_center_km = chord_radius(f1, f2, costh)
        return n_n_amb_eV3 + _yk_extra_cm3(r_center_km, yk) * cm_inv3_to_eV3
    n_tot = spatial_total_density_earth(l_km, baseline_km, costh, detector_depth_km, n_n_amb_eV3)
    if mode == _YUKAWA_HEAVY:
        r_center_km = chord_radius(f1, f2, costh)
        d = _yk_extra_cm3(r_center_km, yk)
        if d != 0.0:
            n_tot = n_tot + d * cm_inv3_to_eV3
    return n_tot


@njit(fastmath=True, error_model="numpy", cache=True)
def _yk_layer_width(m: float, r_b: float, t_abs: float) -> float:
    """Width along a chord of the boundary layer at radius r_b, met at distance t_abs from the
    chord's closest approach to the centre: 1/(m |dr/dl|) with |dr/dl| = t_abs/r_b, capped at
    the grazing-incidence width sqrt(r_b/m)/2. Arguments: m (1/km), r_b (km), t_abs (km)."""
    w_graze = 0.5 * np.sqrt(r_b / m)
    s = t_abs / r_b
    if s * m * w_graze > 1.0:
        return 1.0 / (m * s)
    return w_graze


@njit(fastmath=True, error_model="numpy", cache=True)
def _earth_chord_features(baseline_km: float, costh: float, detector_depth_km: float,
                          cross_ls: np.ndarray, yk):
    """Positions along an Earth chord (km) where n_eff has structure on the scale of the
    mediator range, with that structure's width along the chord: the surface at entry, every
    PREM boundary crossing, the surface just above the detector (when within 40/m of it), and
    the closest approach when it passes within 40/m of a boundary. Arguments: as
    _avg_earth_density_seq; yk -- a non-contact earth_yukawa_table. Returns (l_f, w_f)."""
    m = yk[0][1]
    edges = yk[1]
    R = edges[len(edges) - 1]
    f2 = R - detector_depth_km
    r_min = f2 * np.sqrt(max(1.0 - costh * costh, 0.0))
    l_t = baseline_km + f2 * costh
    nc = len(cross_ls)
    feat_l = np.empty(nc + 3)
    feat_w = np.empty(nc + 3)
    feat_l[0] = 0.0
    feat_w[0] = _yk_layer_width(m, R, abs(l_t))
    n = 1
    for i in range(nc):
        f1 = baseline_km - cross_ls[i]
        r_c = chord_radius(f1, f2, costh)
        feat_l[n] = cross_ls[i]
        feat_w[n] = _yk_layer_width(m, r_c, abs(cross_ls[i] - l_t))
        n += 1
    if m * detector_depth_km < 40.0:
        feat_l[n] = baseline_km
        feat_w[n] = _yk_layer_width(m, f2, abs(baseline_km - l_t))
        n += 1
    if 0.0 < l_t < baseline_km:
        for k in range(1, len(edges)):
            if abs(r_min - edges[k]) < 40.0 / m:
                feat_l[n] = l_t
                feat_w[n] = 0.5 * np.sqrt(max(r_min, 1e-9) / m)
                n += 1
                break
    return feat_l[:n].copy(), feat_w[:n].copy()


@njit(fastmath=True, error_model="numpy", cache=True)
def _yk_graded_integral(A: float, B: float, feat_l: np.ndarray, feat_w: np.ndarray,
                        w_min: float, baseline_km: float, costh: float,
                        detector_depth_km: float, yk, glx: np.ndarray, glw: np.ndarray) -> float:
    """Integral over l in [A, B] (km) of _yk_extra_cm3 along an Earth chord (cm^-3 km), with
    Gauss-Legendre sub-segments graded toward every boundary layer that is thinner than a
    quarter of the segment but thicker than w_min: edges at l_f +- w (1, 2, 3, 4, then
    sqrt(2) apart), so each sub-segment is at most ~1/2.4 of its distance from the layer
    and 10 points integrate its exponential to fp accuracy. Layers wider than a quarter of
    the segment need no grading. Arguments: A, B -- the segment; feat_l, feat_w -- from
    _earth_chord_features; w_min -- the thinnest layer resolved; the chord geometry; yk;
    glx, glw -- Gauss-Legendre nodes and weights on [-1, 1]."""
    L = B - A
    nf = len(feat_l)
    buf = np.empty(4 + 160 * nf)
    cap = len(buf) - 2
    buf[0] = A
    buf[1] = B
    n = 2
    for f in range(nf):
        w = feat_w[f]
        if w < w_min or w >= 0.25 * L:
            continue
        lf = feat_l[f]
        dist = 0.0
        if lf < A:
            dist = A - lf
        elif lf > B:
            dist = lf - B
        if dist > 40.0 * w:
            continue
        if A < lf < B:
            buf[n] = lf
            n += 1
        x = w
        i = 1
        while x < L + dist and n < cap:
            e1 = lf - x
            e2 = lf + x
            if A < e1 < B:
                buf[n] = e1
                n += 1
            if A < e2 < B:
                buf[n] = e2
                n += 1
            if i < 4:
                x = (i + 1) * w
            else:
                x *= 1.4142135623730951
            i += 1
    edges = np.sort(buf[:n])
    f2 = 6371.0 - detector_depth_km
    total = 0.0
    for i in range(n - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if hi <= lo:
            continue
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        acc = 0.0
        for q in range(len(glx)):
            f1 = baseline_km - (mid + half * glx[q])
            r = chord_radius(f1, f2, costh)
            acc += glw[q] * _yk_extra_cm3(r, yk)
        total += acc * half
    return total


@njit(fastmath=True, error_model="numpy", cache=True)
def _evolve_rho_all_bins(E_bins: np.ndarray, rho: np.ndarray, M_rot: np.ndarray,
                         V_const: np.ndarray, dr_eV_inv: float,
                         H_buf: np.ndarray, U_buf: np.ndarray, T_buf: np.ndarray):
    """Applies one spatial step rho -> U rho U^dagger, U = exp(-i (H_vac(E) + V_const) dr),
    to every energy bin, using the closed-form propagator_3x3 and fully hand-rolled 3x3
    products into pre-allocated buffers. The hand-rolling matters: the previous
    formulation (np.linalg.eigh + `U @ rho @ conj(U).T` with ascontiguousarray casts)
    allocated several temporaries per energy bin per step, and at production settings the
    spatial phase performs millions of these steps -- the allocations and the LAPACK call
    dominated its runtime.
    """
    N_E = len(E_bins)
    for i in range(N_E):
        inv2E = 1.0 / (2.0 * E_bins[i])
        for a in range(3):
            for b in range(3):
                H_buf[a, b] = inv2E * M_rot[a, b] + V_const[a, b]

        propagator_3x3(H_buf, dr_eV_inv, U_buf)

        # T = U @ rho_i ; rho_i = T @ U^dagger  (hand-rolled, allocation-free).
        # rho is Hermitian and stays Hermitian under U rho U^dagger, so only the upper
        # triangle is computed; the lower triangle is mirrored by conjugation and the
        # diagonal is forced real -- ~1/3 fewer multiplies in the second product, and
        # exact Hermiticity is enforced instead of drifting at the ulp level.
        for a in range(3):
            for b in range(3):
                T_buf[a, b] = (U_buf[a, 0] * rho[i, 0, b]
                               + U_buf[a, 1] * rho[i, 1, b]
                               + U_buf[a, 2] * rho[i, 2, b])
        for a in range(3):
            rho[i, a, a] = complex((T_buf[a, 0] * np.conjugate(U_buf[a, 0])
                                    + T_buf[a, 1] * np.conjugate(U_buf[a, 1])
                                    + T_buf[a, 2] * np.conjugate(U_buf[a, 2])).real, 0.0)
            for b in range(a + 1, 3):
                val = (T_buf[a, 0] * np.conjugate(U_buf[b, 0])
                       + T_buf[a, 1] * np.conjugate(U_buf[b, 1])
                       + T_buf[a, 2] * np.conjugate(U_buf[b, 2]))
                rho[i, a, b] = val
                rho[i, b, a] = np.conjugate(val)

@njit(fastmath=True, error_model="numpy", cache=True)
def _evolve_U_all_bins(E_bins: np.ndarray, U: np.ndarray, M_rot: np.ndarray, V_const: np.ndarray, dr_eV_inv: float,
                       H_buf: np.ndarray, S_buf: np.ndarray, T_buf: np.ndarray):
    """U -> exp(-i (H_vac(E) + V_const) dr) U in every energy bin: a chord's evolution operator accumulated step by
    step (2026-10-18), the steps of _evolve_rho_all_bins, so that U rho U^dagger is their result to rounding for any
    rho (one crossing serves every input)."""
    N_E = len(E_bins)
    for i in range(N_E):
        inv2E = 1.0 / (2.0 * E_bins[i])
        for a in range(3):
            for b in range(3):
                H_buf[a, b] = inv2E * M_rot[a, b] + V_const[a, b]
        propagator_3x3(H_buf, dr_eV_inv, S_buf)
        for a in range(3):
            for b in range(3):
                T_buf[a, b] = S_buf[a, 0] * U[i, 0, b] + S_buf[a, 1] * U[i, 1, b] + S_buf[a, 2] * U[i, 2, b]
        for a in range(3):
            for b in range(3):
                U[i, a, b] = T_buf[a, b]


@njit(fastmath=True, error_model="numpy", cache=True)
def step_earth_transit(l_km: float, baseline_km: float, costh: float, dr_earth_eV_inv: float,
                       E_bins: np.ndarray, rho_earth: np.ndarray, M_rot: np.ndarray,
                       n_n_amb_eV3: float, Coef: float, U_LRI_struct: np.ndarray,
                       V_SM: np.ndarray, detector_depth_km: float,
                       U_step_buffer: np.ndarray, H_buf: np.ndarray, T_buf: np.ndarray, yk):
    """Executes a single path-length step through the Earth interior using pre-allocated contiguous matrix buffers.

    V_SM is the complete, prebuilt 3x3 SM matter potential (zeros when use_SM_potential is
    off). It is constant along the entire chord -- it depends only on the configured
    electron density and the antineutrino sign convention, never on position -- so it is
    built exactly once in propagate_earth_spatial rather than being reconstructed here on
    every single path-length step as it previously was. yk is the Earth's Yukawa table
    (earth_yukawa_table): the density is sampled at the step's left endpoint ('left' scheme).
    """
    n_tot_eV3 = earth_total_density(l_km, baseline_km, costh, detector_depth_km, n_n_amb_eV3, yk)
    V_LRI = (Coef * n_tot_eV3) * U_LRI_struct + V_SM

    _evolve_rho_all_bins(E_bins, rho_earth, M_rot, V_LRI, dr_earth_eV_inv, H_buf, U_step_buffer, T_buf)

# =============================================================================
# REVISED SOLVERS WITH VERBOSE PROGRESS TRACKING
# =============================================================================

# PREM layer boundary radii (km) -- the breakpoints of calc_earth_density
_PREM_BOUNDARIES_KM = _PREM_EDGES_KM[1:].copy()


def _earth_layer_crossings(baseline_km: float, costh: float, detector_depth_km: float) -> np.ndarray:
    """Path positions l (km, sorted, interior) where a chord crosses a PREM layer
    boundary. Closed form: with u = baseline - l and f2 = 6371 - depth, the geocentric
    radius obeys r^2 = u^2 + 2 f2 costh u + f2^2, so r = r_b gives
    u = -f2 costh +- sqrt(r_b^2 - f2^2 (1 - costh^2))."""
    f2 = 6371.0 - detector_depth_km
    ls = []
    for r_b in _PREM_BOUNDARIES_KM:
        disc = r_b * r_b - f2 * f2 * (1.0 - costh * costh)
        if disc <= 0.0:
            continue
        sq = np.sqrt(disc)
        for u in (-f2 * costh + sq, -f2 * costh - sq):
            l = baseline_km - u
            # margin excludes the entry-point artifact: every chord ENTERS exactly on
            # the r = 6371 surface boundary, so that root lands at l = O(fp noise);
            # genuine slivers below ~1e-6 * baseline are integration-negligible anyway.
            # (Exactly true only since get_baseline_earth became exact, 2026-10-12: the
            # chords used to start up to 8 m above the surface, and where that vacuum
            # lead-in fell inside this margin the 'avg' integration ran the top layer
            # out to l = 0.)
            if 1e-6 * baseline_km < l < baseline_km * (1.0 - 1e-6):
                ls.append(l)
    return np.array(sorted(ls), dtype=np.float64)


@njit(fastmath=True, error_model="numpy", cache=True)
def _avg_earth_density_seq(baseline_km: float, costh: float, N_steps: int,
                           detector_depth_km: float, n_n_amb_eV3: float,
                           cross_ls: np.ndarray, glx: np.ndarray, glw: np.ndarray,
                           yk) -> np.ndarray:
    """Per-step EXACT path-averaged total density along an Earth chord:
    nbar_s = (1/dr) * integral of earth_total_density over [l_s, l_{s+1}],
    with the integral split at PREM layer crossings (cross_ls, sorted) so every
    Gauss-Legendre subsegment integrates a smooth piecewise-polynomial -- the density
    is then exact to fp accuracy for any step count. This is what makes the 'avg'
    stepping scheme's accumulated LRI phase equal Coef * integral(n dl) EXACTLY,
    independent of N_steps -- see propagate_earth_spatial's scheme note.

    yk is the Earth's Yukawa table (earth_yukawa_table). The contact part (ambient plus the
    local PREM density) is integrated exactly as before 2026-10-12; the Yukawa remainder
    (_yk_extra_cm3) is added from _yk_graded_integral, whose sub-segments resolve the boundary
    layers near crossings, the surface and the detector. At MeV masses that remainder
    underflows to zero and nbar is bit-identical to the contact form. In direct mode (a
    mediator range beyond R/50) n_eff replaces the local density entirely."""
    dr = baseline_km / N_steps
    nbar = np.empty(N_steps, dtype=np.float64)
    n_cross = len(cross_ls)
    n_gl = len(glx)
    mode = int(yk[0][0])
    if mode != _YUKAWA_CONTACT:
        feat_l, feat_w = _earth_chord_features(baseline_km, costh, detector_depth_km, cross_ls, yk)
    else:
        feat_l = np.empty(0)
        feat_w = np.empty(0)
    w_min = _YUKAWA_LAYER_REL_TOL * dr
    for s in range(N_steps):
        a = s * dr
        b = a + dr
        total = 0.0
        extra = 0.0
        seg_a = a
        for k in range(n_cross + 1):
            seg_b = b
            if k < n_cross:
                c = cross_ls[k]
                if c <= a:
                    continue
                if c < b:
                    seg_b = c
                else:
                    seg_b = b
            if seg_b <= seg_a:
                continue
            if mode != _YUKAWA_DIRECT:
                mid = 0.5 * (seg_a + seg_b)
                half = 0.5 * (seg_b - seg_a)
                acc = 0.0
                for q in range(n_gl):
                    l = mid + half * glx[q]
                    acc += glw[q] * spatial_total_density_earth(l, baseline_km, costh,
                                                               detector_depth_km, n_n_amb_eV3)
                total += acc * half
            if mode != _YUKAWA_CONTACT:
                extra += _yk_graded_integral(seg_a, seg_b, feat_l, feat_w, w_min, baseline_km,
                                             costh, detector_depth_km, yk, glx, glw)
            seg_a = seg_b
            if seg_a >= b:
                break
        if mode == _YUKAWA_DIRECT:
            total = n_n_amb_eV3 * dr
        nbar[s] = (total + extra * cm_inv3_to_eV3) / dr
    return nbar


@njit(fastmath=True, error_model="numpy", cache=True)
def _avg_earth_density_seq_q(baseline_km: float, costh: float, N_steps: int,
                             detector_depth_km: float, n_n_amb_eV3: float,
                             cross_ls: np.ndarray, glx: np.ndarray, glw: np.ndarray,
                             yk) -> np.ndarray:
    """_avg_earth_density_seq for a table with a general matter charge (earth_yukawa_table's cpg, read from
    yk[0][5:8]): the same integration, its contact part from spatial_total_density_earth_q. A twin rather than a
    branch, so the neutron path keeps its compiled form; a regression test holds the two equal for the neutron
    charge."""
    cpg = yk[0][5:8]
    dr = baseline_km / N_steps
    nbar = np.empty(N_steps, dtype=np.float64)
    n_cross = len(cross_ls)
    n_gl = len(glx)
    mode = int(yk[0][0])
    if mode != _YUKAWA_CONTACT:
        feat_l, feat_w = _earth_chord_features(baseline_km, costh, detector_depth_km, cross_ls, yk)
    else:
        feat_l = np.empty(0)
        feat_w = np.empty(0)
    w_min = _YUKAWA_LAYER_REL_TOL * dr
    for s in range(N_steps):
        a = s * dr
        b = a + dr
        total = 0.0
        extra = 0.0
        seg_a = a
        for k in range(n_cross + 1):
            seg_b = b
            if k < n_cross:
                c = cross_ls[k]
                if c <= a:
                    continue
                if c < b:
                    seg_b = c
                else:
                    seg_b = b
            if seg_b <= seg_a:
                continue
            if mode != _YUKAWA_DIRECT:
                mid = 0.5 * (seg_a + seg_b)
                half = 0.5 * (seg_b - seg_a)
                acc = 0.0
                for q in range(n_gl):
                    l = mid + half * glx[q]
                    acc += glw[q] * spatial_total_density_earth_q(l, baseline_km, costh,
                                                                 detector_depth_km, n_n_amb_eV3, cpg)
                total += acc * half
            if mode != _YUKAWA_CONTACT:
                extra += _yk_graded_integral(seg_a, seg_b, feat_l, feat_w, w_min, baseline_km,
                                             costh, detector_depth_km, yk, glx, glw)
            seg_a = seg_b
            if seg_a >= b:
                break
        if mode == _YUKAWA_DIRECT:
            total = n_n_amb_eV3 * dr
        nbar[s] = (total + extra * cm_inv3_to_eV3) / dr
    return nbar


def chord_density_seq(yk):
    """The chord integrator for a table: _avg_earth_density_seq, or its charged twin (yk_charged)."""
    return _avg_earth_density_seq_q if yk_charged(yk) else _avg_earth_density_seq


# =============================================================================
# ZENITH AVERAGE OF THE EARTH CHORDS
# =============================================================================
# The flux at the detector is the average over the sky, uniform in costh over [-1, 1]
# ('full', the default since 2026-10-13) or [-1, 0] ('up'); per declination band, the same
# average restricted to each band (average_earth_chords_bands). Inside the Earth the dark-portal potential exceeds the vacuum term at
# every energy of interest whenever its phase is non-negligible (a phase of 1 rad across
# the Earth needs V ~ 1e-14 eV, against Delta m^2/2E <~ 1e-16 eV above 10 TeV), so to
# O(H_vac/V) a chord evolves as
#     U(c) = sum_g exp(-i lambda_g Psi(c)) P_g,      Psi(c) = Coef * int n_tot dl,
# with P_g the eigenprojectors of the LRI texture S (eigenvalues lambda_g). The block
# P_g rho P_h of the evolved density matrix is then A_gh(c) exp(-i omega_gh Psi(c)),
# omega_gh = lambda_g - lambda_h, with an amplitude A_gh that varies slowly with c (the
# vacuum term's own phases across the Earth are <~ 1e-2 rad). Psi reaches 1e5-1e8 rad at
# MeV masses: from chord to chord the cross-block phases are effectively random, and an
# equal-weight mean over a finite set of chords keeps them as sampling noise -- the
# "chord-sampling floor", ~1e-2 in the energy-averaged flavor fractions at 50 chords
# (found 2026-10-12).
# The 'phase' zenith average integrates each block against its known phase (a Filon-type
# rule): A_gh is interpolated linearly between the chords and the oscillatory factor is
# integrated exactly on a fine grid of Psi(c) with piecewise-linear phase, giving complex
# per-chord weights
#     w_k^gh = int hat_k(c) exp(-i omega_gh [Psi(c) - Psi(c_k)]) dc,
# energy-independent, since Psi is. For omega = 0 they are the trapezoid weights; where the
# phase turns many times between neighbouring chords they fall to ~1/(omega Psi'), and the
# average becomes the block-diagonal sum_g P_g rho P_g that the continuum of zeniths
# produces; in between (Earth phases of ~1-1e4 rad) they resolve the oscillation that a
# chord sum would alias. Phases other than the LRI one (the SM potential, when switched
# on: <~ 12 rad) stay in A_gh and are interpolated. The rule needs the phase-exact 'avg'
# Earth steps: under 'left' each chord's phase carries a Riemann error of thousands of
# radians that no smooth Psi(c) describes. 'plain' keeps the historical equal-weight mean.
_ZENITH_AVERAGES = ("phase", "plain")
# Fine-grid size for Psi(c), and the tangency grading: Psi' diverges like
# 1/sqrt(c_t - c) where a chord starts to cross a PREM boundary (c_t), so points are
# added at c_t - 1e-8 * 2^j. Phase errors of the piecewise-linear rule are
# ~omega Psi'' dc^2 / 8, below 1e-3 rad up to Psi ~ 1e5 on 4096 points; beyond that
# the cross weights are ~1/(omega Psi') and their errors are of the same size.
_ZENITH_FINE_POINTS = 4096
_ZENITH_FINE_POINTS_YUKAWA = 1024
# Below this mediator mass the Yukawa smoothing of the Earth's density shifts Psi(c) by
# more than ~1e-8 of itself (the range 1/m exceeds ~10 m), enough to matter for the
# weights in the coherent regime; above it the contact column is used and cached.
_ZENITH_YUKAWA_MIN_M_EV = 2e-5
_zenith_column_cache = {}


@njit(fastmath=True, error_model="numpy", cache=True)
def _earth_layer_crossings_nb(baseline_km: float, costh: float, detector_depth_km: float) -> np.ndarray:
    """numba twin of _earth_layer_crossings (same closed form and end margin), for the
    zenith fine grid, which needs thousands of chords. Arguments and return as there."""
    f2 = 6371.0 - detector_depth_km
    out = np.empty(2 * len(_PREM_EDGES_KM))
    n = 0
    for i in range(1, len(_PREM_EDGES_KM)):
        r_b = _PREM_EDGES_KM[i]
        disc = r_b * r_b - f2 * f2 * (1.0 - costh * costh)
        if disc <= 0.0:
            continue
        sq = np.sqrt(disc)
        for sgn in (1.0, -1.0):
            l = baseline_km - (-f2 * costh + sgn * sq)
            if 1e-6 * baseline_km < l < baseline_km * (1.0 - 1e-6):
                out[n] = l
                n += 1
    return np.sort(out[:n])


@njit(fastmath=True, error_model="numpy", cache=True)
def _earth_columns(costh_arr: np.ndarray, detector_depth_km: float, n_n_amb_eV3: float,
                   yk) -> np.ndarray:
    """Column of total density int n_tot dl (eV^3 km) along the chord of each costh, with
    the same layer-split, boundary-layer-resolving integration as the 'avg' steps (one
    step per chord). Arguments: costh_arr; detector_depth_km; n_n_amb_eV3; yk -- an
    earth_yukawa_table. Coef times this, in eV^-1, is the chord's LRI phase scale Psi."""
    out = np.empty(len(costh_arr))
    for i in range(len(costh_arr)):
        c = costh_arr[i]
        B = get_baseline_earth(detector_depth_km, c)
        cross = _earth_layer_crossings_nb(B, c, detector_depth_km)
        out[i] = _avg_earth_density_seq(B, c, 1, detector_depth_km, n_n_amb_eV3, cross,
                                        _GL10_X, _GL10_W, yk)[0] * B
    return out


@njit(fastmath=True, error_model="numpy", cache=True)
def _earth_columns_q(costh_arr: np.ndarray, detector_depth_km: float, n_n_amb_eV3: float,
                     yk) -> np.ndarray:
    """_earth_columns for a table with a general matter charge (_avg_earth_density_seq_q)."""
    out = np.empty(len(costh_arr))
    for i in range(len(costh_arr)):
        c = costh_arr[i]
        B = get_baseline_earth(detector_depth_km, c)
        cross = _earth_layer_crossings_nb(B, c, detector_depth_km)
        out[i] = _avg_earth_density_seq_q(B, c, 1, detector_depth_km, n_n_amb_eV3, cross,
                                          _GL10_X, _GL10_W, yk)[0] * B
    return out


def earth_columns(costh_arr, detector_depth_km, n_n_amb_eV3, yk):
    """_earth_columns, or its charged twin for a table with a general matter charge."""
    fn = _earth_columns_q if yk_charged(yk) else _earth_columns
    return fn(costh_arr, detector_depth_km, n_n_amb_eV3, yk)


def _earth_zenith_fine_grid(nodes: np.ndarray, detector_depth_km: float, n_points: int):
    """Fine costh grid for the zenith weights: every chord node, n_points spread uniformly
    over the node intervals, and the tangency grading (see _ZENITH_FINE_POINTS). Arguments:
    nodes -- the chords' costh, increasing; detector_depth_km; n_points. Returns
    (c_fine, node_index) with c_fine[node_index[k]] == nodes[k] exactly."""
    K = len(nodes)
    n_sub = max(1, int(np.ceil(n_points / (K - 1))))
    pts = [np.linspace(nodes[j], nodes[j + 1], n_sub + 1) for j in range(K - 1)]
    f2 = 6371.0 - detector_depth_km
    for r_b in _PREM_EDGES_KM[1:-1]:
        if r_b >= f2:
            continue
        c_t = -np.sqrt(1.0 - (r_b / f2) ** 2)
        if not (nodes[0] < c_t < nodes[-1]):
            continue
        d = 1e-8
        grade = [c_t]
        while d < (nodes[-1] - nodes[0]) / n_points:
            grade.append(c_t - d)
            d *= 2.0
        pts.append(np.array(grade))
    c_fine = np.unique(np.concatenate(pts + [nodes]))
    c_fine = c_fine[(c_fine >= nodes[0]) & (c_fine <= nodes[-1])]
    node_index = np.searchsorted(c_fine, nodes)
    assert np.array_equal(c_fine[node_index], nodes)
    return c_fine, node_index


def _filon_J(z: np.ndarray):
    """J0 = int_0^1 e^{zt} dt and J1 = int_0^1 t e^{zt} dt for purely imaginary z (the
    phase change across a sub-interval), with series below |z| = 0.1 where the closed
    forms cancel. Argument: z -- complex array. Returns (J0, J1)."""
    J0 = np.empty_like(z)
    J1 = np.empty_like(z)
    small = np.abs(z) < 0.1
    zs = z[small]
    t0 = np.zeros_like(zs)
    t1 = np.zeros_like(zs)
    term = np.ones_like(zs)                       # z^k / k!
    for k in range(12):
        t0 += term / (k + 1)
        t1 += term / (k + 2)
        term = term * zs / (k + 1)
    J0[small], J1[small] = t0, t1
    zl = z[~small]
    ez = np.exp(zl)
    J0[~small] = (ez - 1.0) / zl
    J1[~small] = (ez * (zl - 1.0) + 1.0) / (zl * zl)
    return J0, J1


@njit(fastmath=False, error_model="numpy", cache=True)
def _filon_intervals_nb(c_fine, node_index, phase):
    """_zenith_filon_weights_intervals in one compiled loop (2026-10-18; the same rule and sums, to rounding):
    W[j, 0] and W[j, 1], the interval [c_j, c_j+1]'s contributions to nodes j and j + 1."""
    K = len(node_index)
    W = np.zeros((K - 1, 2), dtype=np.complex128)
    for j in range(K - 1):
        i0 = node_index[j]
        i1 = node_index[j + 1]
        span = c_fine[i1] - c_fine[i0]
        ph0 = phase[i0]
        ph1 = phase[i1]
        acc0 = 0.0 + 0.0j
        acc1 = 0.0 + 0.0j
        for i in range(i0, i1):
            dc = c_fine[i + 1] - c_fine[i]
            up_a = (c_fine[i] - c_fine[i0]) / span
            up_b = (c_fine[i + 1] - c_fine[i0]) / span
            z = -1j * (phase[i + 1] - phase[i])
            if abs(z) < 0.1:
                J0 = 0.0 + 0.0j
                J1 = 0.0 + 0.0j
                term = 1.0 + 0.0j
                for k in range(12):
                    J0 += term / (k + 1)
                    J1 += term / (k + 2)
                    term = term * z / (k + 1)
            else:
                ez = np.exp(z)
                J0 = (ez - 1.0) / z
                J1 = (ez * (z - 1.0) + 1.0) / (z * z)
            # node j: hat_dn = 1 - hat_up, ref = phase at c_j; node j + 1: hat_up, ref = phase at c_j+1
            dn_a = 1.0 - up_a
            dn_b = 1.0 - up_b
            acc0 += dc * np.exp(-1j * (phase[i] - ph0)) * (dn_a * J0 + (dn_b - dn_a) * J1)
            acc1 += dc * np.exp(-1j * (phase[i] - ph1)) * (up_a * J0 + (up_b - up_a) * J1)
        W[j, 0] = acc0
        W[j, 1] = acc1
    return W


def _zenith_filon_weights(c_fine: np.ndarray, node_index: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """_zenith_filon_weights_ref (its docstring), from the compiled intervals (_filon_intervals_nb, 2026-10-18)."""
    W = _filon_intervals_nb(np.ascontiguousarray(c_fine, dtype=np.float64),
                            np.ascontiguousarray(node_index, dtype=np.int64),
                            np.ascontiguousarray(phase, dtype=np.float64))
    w = np.zeros(len(node_index), dtype=np.complex128)
    w[:-1] += W[:, 0]
    w[1:] += W[:, 1]
    return w


def _zenith_filon_weights_intervals(c_fine: np.ndarray, node_index: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """_zenith_filon_weights_intervals_ref (its docstring), compiled (_filon_intervals_nb, 2026-10-18)."""
    return _filon_intervals_nb(np.ascontiguousarray(c_fine, dtype=np.float64),
                               np.ascontiguousarray(node_index, dtype=np.int64),
                               np.ascontiguousarray(phase, dtype=np.float64))


def _zenith_filon_weights_ref(c_fine: np.ndarray, node_index: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Per-chord weights w_k = int hat_k(c) exp(-i [phase(c) - phase(c_k)]) dc over the
    node range, with hat_k the linear interpolation basis on the chord nodes and the phase
    linear on each fine sub-interval (exact for it). Arguments: c_fine, node_index -- from
    _earth_zenith_fine_grid; phase -- omega * Psi on c_fine (rad). Returns complex (K,);
    for phase = 0, the trapezoid weights."""
    K = len(node_index)
    w = np.zeros(K, dtype=np.complex128)
    for j in range(K - 1):
        i0, i1 = node_index[j], node_index[j + 1]
        c = c_fine[i0:i1 + 1]
        ph = phase[i0:i1 + 1]
        dc = np.diff(c)
        hat_up = (c - c[0]) / (c[-1] - c[0])      # hat_{j+1} on this interval
        hat_dn = 1.0 - hat_up                     # hat_j
        J0, J1 = _filon_J(-1j * np.diff(ph))
        for k, hat, ref in ((j, hat_dn, ph[0]), (j + 1, hat_up, ph[-1])):
            base = np.exp(-1j * (ph[:-1] - ref))
            w[k] += np.sum(dc * base * (hat[:-1] * J0 + np.diff(hat) * J1))
    return w


def _zenith_filon_weights_intervals_ref(c_fine: np.ndarray, node_index: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """The contributions of each node interval to _zenith_filon_weights: W[j, 0] to node j and
    W[j, 1] to node j + 1 from the interval [c_j, c_{j+1}] (same rule, same sums). Summed over the
    intervals inside a declination band they give that band's weights; over all intervals, the
    whole range's. Arguments as _zenith_filon_weights. Returns complex (K - 1, 2)."""
    K = len(node_index)
    W = np.zeros((K - 1, 2), dtype=np.complex128)
    for j in range(K - 1):
        i0, i1 = node_index[j], node_index[j + 1]
        c = c_fine[i0:i1 + 1]
        ph = phase[i0:i1 + 1]
        dc = np.diff(c)
        hat_up = (c - c[0]) / (c[-1] - c[0])
        hat_dn = 1.0 - hat_up
        J0, J1 = _filon_J(-1j * np.diff(ph))
        for side, hat, ref in ((0, hat_dn, ph[0]), (1, hat_up, ph[-1])):
            base = np.exp(-1j * (ph[:-1] - ref))
            W[j, side] = np.sum(dc * base * (hat[:-1] * J0 + np.diff(hat) * J1))
    return W


def _lri_texture_projectors(S: np.ndarray):
    """Eigenprojectors of the Hermitian LRI texture S, one per distinct eigenvalue (within
    1e-9 of the largest |eigenvalue|), and those eigenvalues. Argument: S (3x3). Returns
    (list of 3x3 projectors, array of eigenvalues)."""
    lam, vec = np.linalg.eigh(np.asarray(S, dtype=np.complex128))
    tol = 1e-9 * max(np.max(np.abs(lam)), 1e-300)
    groups = [[0]]
    for i in range(1, 3):
        if lam[i] - lam[groups[-1][-1]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    projectors = [vec[:, g] @ np.conjugate(vec[:, g]).T for g in groups]
    return projectors, np.array([np.mean(lam[g]) for g in groups])


def average_earth_chords(rho_chords: np.ndarray, earth_costh: np.ndarray, Coef: float,
                         U_LRI_struct: np.ndarray, detector_depth_km: float, n_n_amb_eV3: float,
                         m_eff_eV: float, earth_yk, zenith_average: str = "phase") -> np.ndarray:
    """Average of the chords' density matrices over their costh range (uniform in costh: the
    whole sky or the up-going hemisphere), by the 'phase' rule of the section notes above or the
    historical 'plain' mean.

    Arguments:
      rho_chords: (K, N_E, 3, 3) density matrices at the detector, one per chord.
      earth_costh: (K,) the chords' costh, increasing.
      Coef: signed LRI coefficient (negative for antineutrinos), eV^-2.
      U_LRI_struct: the LRI texture S in the convention of this call.
      detector_depth_km, n_n_amb_eV3, earth_yk: the chords' geometry and densities.
      m_eff_eV: the mediator's effective mass (selects the cached contact column above
        _ZENITH_YUKAWA_MIN_M_EV).
      zenith_average: 'phase' or 'plain'.
    Returns: (N_E, 3, 3) averaged density matrix.
    """
    K = len(earth_costh)
    if zenith_average == "plain" or K == 1:
        return np.mean(rho_chords, axis=0)
    rho_avg = np.zeros(rho_chords.shape[1:], dtype=np.complex128)
    for Pg, Ph, w in _zenith_weights(earth_costh, None, Coef, U_LRI_struct, detector_depth_km, n_n_amb_eV3, m_eff_eV,
                                     earth_yk):
        R = np.tensordot(w, rho_chords, axes=(0, 0))          # (N_E, 3, 3)
        rho_avg += Pg @ R @ Ph
    return rho_avg


_ZENITH_WEIGHTS = {}
_ZENITH_WEIGHTS_MAX = 16


def _zenith_weights(earth_costh, band_costh, Coef, U_LRI_struct, detector_depth_km, n_n_amb_eV3, m_eff_eV, earth_yk):
    """The 'phase' rule's weights per pair of the texture's projectors, [(Pg, Ph, w)]: w (K,) over the whole range
    (divided by its span) with band_costh None, else (N_b, K), each band's (divided by its width; band edges among
    the chords). Kept per argument set (2026-10-18): they do not depend on the chords' density matrices, and every
    run of a point, and each mixing template's, repeats them."""
    yk_key = tuple(np.ascontiguousarray(x).tobytes() for x in earth_yk)
    key = (np.ascontiguousarray(earth_costh, dtype=np.float64).tobytes(),
           None if band_costh is None else np.ascontiguousarray(band_costh, dtype=np.float64).tobytes(),
           float(Coef), np.ascontiguousarray(U_LRI_struct, dtype=np.complex128).tobytes(), float(detector_depth_km),
           float(n_n_amb_eV3), float(m_eff_eV), yk_key)
    hit = _ZENITH_WEIGHTS.get(key)
    if hit is not None:
        return hit
    projectors, lam = _lri_texture_projectors(U_LRI_struct)
    c_fine, node_index, col = _zenith_phase_grid(earth_costh, detector_depth_km, n_n_amb_eV3, m_eff_eV, earth_yk)
    psi = Coef * col * _KM_IN_EV_INV                       # Psi(c), rad per unit eigenvalue
    out = []
    if band_costh is None:
        span = earth_costh[-1] - earth_costh[0]
        for g, Pg in enumerate(projectors):
            for h, Ph in enumerate(projectors):
                out.append((Pg, Ph, _zenith_filon_weights(c_fine, node_index, (lam[g] - lam[h]) * psi) / span))
    else:
        pos = np.searchsorted(earth_costh, band_costh)
        nb = len(band_costh) - 1
        for g, Pg in enumerate(projectors):
            for h, Ph in enumerate(projectors):
                W = _zenith_filon_weights_intervals(c_fine, node_index, (lam[g] - lam[h]) * psi)
                w = np.zeros((nb, len(earth_costh)), dtype=np.complex128)
                for b in range(nb):
                    for j in range(pos[b], pos[b + 1]):
                        w[b, j] += W[j, 0]
                        w[b, j + 1] += W[j, 1]
                    w[b] /= (band_costh[b + 1] - band_costh[b])
                out.append((Pg, Ph, w))
    if len(_ZENITH_WEIGHTS) >= _ZENITH_WEIGHTS_MAX:
        _ZENITH_WEIGHTS.pop(next(iter(_ZENITH_WEIGHTS)))
    _ZENITH_WEIGHTS[key] = out
    return out


def _zenith_phase_grid(earth_costh: np.ndarray, detector_depth_km: float, n_n_amb_eV3: float,
                       m_eff_eV: float, earth_yk):
    """The fine costh grid of the 'phase' zenith rule and the density column on it (cached):
    the contact column above _ZENITH_YUKAWA_MIN_M_EV, the Yukawa column (with the Earth's exterior
    when active) below. Returns (c_fine, node_index, col)."""
    charge = tuple(float(x) for x in earth_yk[0][5:8]) if yk_charged(earth_yk) else ()
    if m_eff_eV > _ZENITH_YUKAWA_MIN_M_EV:
        key = (float(detector_depth_km), tuple(np.round(earth_costh, 15)), float(n_n_amb_eV3)) + charge
        hit = _zenith_column_cache.get(key)
        if hit is None:
            c_fine, node_index = _earth_zenith_fine_grid(earth_costh, detector_depth_km, _ZENITH_FINE_POINTS)
            contact = earth_yukawa_table(None, cpg=np.array(charge) if charge else None)
            col = earth_columns(c_fine, detector_depth_km, n_n_amb_eV3, contact)
            if len(_zenith_column_cache) > 64:
                _zenith_column_cache.clear()
            _zenith_column_cache[key] = hit = (c_fine, node_index, col)
        c_fine, node_index, col = hit
    else:
        # the Yukawa columns depend on m_eff only through earth_yk = earth_yukawa_table(m_eff):
        # cached per m_eff, so the antineutrino pass reuses the neutrino pass's columns
        key = ("yukawa", float(m_eff_eV), float(detector_depth_km), tuple(np.round(earth_costh, 15)),
               float(n_n_amb_eV3)) + charge
        hit = _zenith_column_cache.get(key)
        if hit is None:
            c_fine, node_index = _earth_zenith_fine_grid(earth_costh, detector_depth_km, _ZENITH_FINE_POINTS_YUKAWA)
            col = earth_columns(c_fine, detector_depth_km, n_n_amb_eV3, earth_yk)
            if earth_exterior_active(m_eff_eV):
                col = col + _earth_exterior_columns(c_fine, detector_depth_km, n_n_amb_eV3, earth_yk,
                                                    _GL10_X, _GL10_W)
            if len(_zenith_column_cache) > 64:
                _zenith_column_cache.clear()
            _zenith_column_cache[key] = hit = (c_fine, node_index, col)
        c_fine, node_index, col = hit
    return c_fine, node_index, col


def earth_chord_nodes(band_costh: np.ndarray, nodes_per_band: int = 2) -> np.ndarray:
    """Chord nodes for per-band output: every band edge, and each band split into nodes_per_band
    equal intervals (decision D1, docs/dev/DIRECTIONS_PLAN.md: the bands' edges must be nodes for
    the 'phase' rule's weights to partition). Arguments: band_costh -- cos(zenith) edges,
    increasing; nodes_per_band >= 1. Returns the increasing nodes."""
    band_costh = np.asarray(band_costh, dtype=np.float64)
    if band_costh.ndim != 1 or len(band_costh) < 2 or np.any(np.diff(band_costh) <= 0):
        raise ValueError("earth_chord_nodes: band edges must increase")
    n = int(nodes_per_band)
    if n < 1:
        raise ValueError(f"earth_chord_nodes: nodes_per_band must be >= 1, got {nodes_per_band}")
    parts = [np.linspace(band_costh[b], band_costh[b + 1], n + 1)[:-1] for b in range(len(band_costh) - 1)]
    return np.concatenate(parts + [band_costh[-1:]])


def average_earth_chords_bands(rho_chords: np.ndarray, earth_costh: np.ndarray, band_costh: np.ndarray,
                               Coef: float, U_LRI_struct: np.ndarray, detector_depth_km: float,
                               n_n_amb_eV3: float, m_eff_eV: float, earth_yk,
                               zenith_average: str = "phase") -> np.ndarray:
    """The zenith average of average_earth_chords restricted to each band [c_b, c_{b+1}] of cos(zenith)
    (per-direction output, docs/dev/DIRECTIONS_PLAN.md step 2): the same Filon weights, summed over
    the node intervals inside the band and divided by its width, so that the bands weighted by their
    widths return the whole range's average. A single chord is not a per-direction answer -- it
    carries its own fast Earth phase -- but its band's average is. 'plain' uses the trapezoid weights
    inside each band (the zero-phase limit of the rule).

    Arguments: as average_earth_chords, plus band_costh -- the bands' cos(zenith) edges, increasing,
    each one a chord node. Returns: (N_b, N_E, 3, 3)."""
    band_costh = np.asarray(band_costh, dtype=np.float64)
    pos = np.searchsorted(earth_costh, band_costh)
    if np.any(pos >= len(earth_costh)) or not np.array_equal(earth_costh[np.minimum(pos, len(earth_costh) - 1)], band_costh):
        raise ValueError("average_earth_chords_bands: every band edge must be a chord node")
    nb = len(band_costh) - 1
    out = np.zeros((nb,) + rho_chords.shape[1:], dtype=np.complex128)
    if zenith_average == "plain":
        dc = np.diff(earth_costh)
        for b in range(nb):
            w = np.zeros(len(earth_costh))
            for j in range(pos[b], pos[b + 1]):
                w[j] += 0.5 * dc[j]
                w[j + 1] += 0.5 * dc[j]
            out[b] = np.tensordot(w / (band_costh[b + 1] - band_costh[b]), rho_chords, axes=(0, 0))
        return out
    for Pg, Ph, w in _zenith_weights(earth_costh, band_costh, Coef, U_LRI_struct, detector_depth_km, n_n_amb_eV3,
                                     m_eff_eV, earth_yk):
        R = np.tensordot(w, rho_chords, axes=(1, 0))          # (N_b, N_E, 3, 3)
        out += Pg @ R @ Ph
    return out


# =============================================================================
# THE EARTH'S EXTERIOR (2026-10-13)
# =============================================================================
# Outside a spherical body the Yukawa integral is a point source with the body's effective
# neutron number: with the table's P(R) (see the section notes on the Earth's own potential),
#     n_eff(r > R) = (m / r) exp(-m (r - R)) P(R),
# m^2 N / (4 pi r) when m R << 1, and n(R)/2 decaying over 1/m just above the surface when
# m R >> 1. Neutrinos cross it before every chord, and for down-going chords (1.5 to 140 km of
# ice) it can carry most of the phase. Out to r_split = 2 R the vacuum term turns the state by
# at most Delta m^2 R / E ~ 0.03 rad above 10 TeV, so there the exterior is integrated along
# each chord's own line, in _EXT_N_STEPS 'avg' steps geometric in r; beyond r_split every line
# runs within R / r of radial, so the transit there is one radial path for all chords
# (dp_transit.source_transit), and each chord adds the phase of its line's extra column,
#     dcol(b) = int_{r_split}^inf n_eff(r) [r / sqrt(r^2 - b^2) - 1] dr ~ n b^2 / (2 r^2),
# b = (R - d) sin(zenith) the line's distance from the centre: that extra potential sits
# within a few r_split, where the vacuum term is negligible, so it acts as a pure phase.
# Above _ZENITH_YUKAWA_MIN_M_EV the exterior is a layer thinner than 1 cm and is left out,
# keeping heavier mediators bit for bit as before.
_EXT_R_SPLIT = 2.0                  # r_split / R
_EXT_N_STEPS = 8


@njit(fastmath=True, error_model="numpy", cache=True)
def _ext_density_cm3(r_km: float, yk) -> float:
    """The Earth's exterior n_eff (cm^-3) at r_km >= R, from the Yukawa table yk."""
    par = yk[0]
    mode = int(par[0])
    if mode == _YUKAWA_CONTACT:
        return 0.0
    m = par[1]
    R = par[2]
    if r_km <= R:
        r_km = R
    return (m / r_km) * np.exp(-m * (r_km - R)) * par[3]


@njit(fastmath=True, error_model="numpy", cache=True)
def _ext_line_column(b_km: float, r_a: float, r_b: float, yk, minus_radial: bool,
                     glx: np.ndarray, glw: np.ndarray) -> float:
    """int_{r_a}^{r_b} n_eff(r) J(r) dr (cm^-3 km) along a line at distance b from the centre
    (b < R <= r_a), with J = r / sqrt(r^2 - b^2) (the line's length per unit r), or J - 1 =
    b^2 / (sqrt(r^2 - b^2) (r + sqrt(r^2 - b^2))) with minus_radial. Panels graded
    geometrically away from the surface resolve the 1/m layer and the grazing Jacobian."""
    par = yk[0]
    m = par[1]
    R = par[2]
    if r_b <= r_a:
        return 0.0
    q = R * R - b_km * b_km
    d0 = min(0.05 / m, 0.05 * q / R, 0.1)
    if d0 <= 0.0:
        d0 = 1e-6
    total = 0.0
    lo = r_a
    while lo < r_b:
        x = lo - R
        hi = R + max(2.0 * x, x + d0)
        if hi > r_b:
            hi = r_b
        if m * (lo - R) > 60.0:
            break
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        acc = 0.0
        for k in range(len(glx)):
            r = mid + half * glx[k]
            w = np.sqrt(r * r - b_km * b_km)
            if minus_radial:
                J = b_km * b_km / (w * (r + w))
            else:
                J = r / w
            acc += glw[k] * _ext_density_cm3(r, yk) * J
        total += acc * half
        lo = hi
    return total


@njit(fastmath=True, error_model="numpy", cache=True)
def earth_exterior_chord(costh: float, detector_depth_km: float, yk, glx: np.ndarray,
                         glw: np.ndarray):
    """The exterior part of the chord at zenith cosine costh (section notes above): returns
    (L, col, dcol) -- the lengths (km) and Earth columns (cm^-3 km) of the _EXT_N_STEPS steps
    from r_split down to the surface, in travel order, and the extra column beyond r_split."""
    R = yk[0][2]
    b = (R - detector_depth_km) * np.sqrt(max(1.0 - costh * costh, 0.0))
    r_split = _EXT_R_SPLIT * R
    L = np.empty(_EXT_N_STEPS)
    col = np.empty(_EXT_N_STEPS)
    for j in range(_EXT_N_STEPS):
        # step j runs from r_hi down to r_lo
        r_hi = R * _EXT_R_SPLIT ** ((_EXT_N_STEPS - j) / _EXT_N_STEPS)
        r_lo = R * _EXT_R_SPLIT ** ((_EXT_N_STEPS - j - 1) / _EXT_N_STEPS)
        if j == 0:
            r_hi = r_split
        if j == _EXT_N_STEPS - 1:
            r_lo = R
        L[j] = np.sqrt(r_hi * r_hi - b * b) - np.sqrt(r_lo * r_lo - b * b)
        col[j] = _ext_line_column(b, r_lo, r_hi, yk, False, glx, glw)
    dcol = _ext_line_column(b, r_split, r_split * 1e9, yk, True, glx, glw)
    return L, col, dcol


def earth_exterior_active(m_eff_eV: float) -> bool:
    """Whether the exterior enters the chords and their zenith columns (section notes)."""
    return 0.0 < m_eff_eV <= _ZENITH_YUKAWA_MIN_M_EV


@njit(fastmath=True, error_model="numpy", cache=True)
def _earth_exterior_columns(costh_arr: np.ndarray, detector_depth_km: float, n_n_amb_eV3: float,
                            yk, glx: np.ndarray, glw: np.ndarray) -> np.ndarray:
    """Exterior column (eV^3 km) of each costh: its steps' Earth columns and ambient density
    times their lengths, plus dcol (earth_exterior_chord) -- what the chord accumulates before
    the surface beyond the radial transit common to all chords."""
    out = np.empty(len(costh_arr))
    for i in range(len(costh_arr)):
        L, col, dcol = earth_exterior_chord(costh_arr[i], detector_depth_km, yk, glx, glw)
        acc = dcol * cm_inv3_to_eV3
        for j in range(len(L)):
            acc += col[j] * cm_inv3_to_eV3 + n_n_amb_eV3 * L[j]
        out[i] = acc
    return out


def chord_exterior(costh: float, detector_depth_km: float, earth_yk, Coef: float,
                   U_LRI_struct: np.ndarray):
    """The exterior data of one chord for apply_earth_exterior: (U_dcol, L_eV_inv, nbar_eV3) --
    the unitary exp(-i Coef dcol S) of the line's extra column beyond r_split, and the lengths
    and Earth densities of its exterior steps (earth_exterior_chord)."""
    L, col, dcol = earth_exterior_chord(costh, detector_depth_km, earth_yk, _GL10_X, _GL10_W)
    U_dcol = np.empty((3, 3), dtype=np.complex128)
    H = np.ascontiguousarray((Coef * dcol * cm_inv3_to_eV3 * _KM_IN_EV_INV) * U_LRI_struct,
                             dtype=np.complex128)
    propagator_3x3(H, 1.0, U_dcol)
    nbar = np.where(L > 0.0, col / np.maximum(L, 1e-300), 0.0) * cm_inv3_to_eV3
    return U_dcol, L * _KM_IN_EV_INV, nbar


def apply_earth_exterior(E_bins: np.ndarray, rho: np.ndarray, M_rot: np.ndarray, Coef: float,
                         U_LRI_struct: np.ndarray, n_n_amb_eV3: float, ext) -> None:
    """Evolves rho (N_E, 3, 3) in place from r_split to the surface along one chord's line:
    the phase of its extra column beyond r_split, then its exterior steps (vacuum term, ambient
    and Earth exterior potentials; no standard matter potential outside the Earth)."""
    U_dcol, L_eV_inv, nbar = ext
    T = np.empty((3, 3), dtype=np.complex128)
    Ud = np.conjugate(U_dcol).T
    for i in range(rho.shape[0]):
        rho[i] = U_dcol @ rho[i] @ Ud
    U_step_buffer = np.zeros((3, 3), dtype=np.complex128)
    H_buf = np.zeros((3, 3), dtype=np.complex128)
    for j in range(len(L_eV_inv)):
        if L_eV_inv[j] <= 0.0:
            continue
        V = np.ascontiguousarray((Coef * (n_n_amb_eV3 + nbar[j])) * U_LRI_struct, dtype=np.complex128)
        _evolve_rho_all_bins(E_bins, rho, M_rot, V, L_eV_inv[j], H_buf, U_step_buffer, T)


def _earth_chord_worker(args) -> np.ndarray:
    """Propagates a single Earth chord (one zenith angle) end-to-end to the detector.

    Module-level (picklable) so it can be dispatched to a multiprocessing pool: each
    chord is fully independent of the others, so this is the unit of parallel work for
    the Earth phase of the spatial handover.

    earth_scheme selects the potential discretization per step:
      'left' -- historical scheme: V sampled at the step's left endpoint (kept
                byte-identical);
      'avg'  -- V is the EXACT path average over the step (layer-split Gauss-Legendre
                integration of the PREM profile). The accumulated LRI phase then equals
                Coef * integral(n dl) exactly for ANY step count, removing the
                Riemann-sum phase jitter that made results step-count-ambiguous at the
                ~1e-2 level (the round-7 fringe-aliasing finding). Same evolution
                kernel; only the V value per step differs.
    """
    (costh, baseline_km, N_steps_earth, E_bins, rho_surf_avg, M_rot, n_n_amb_eV3, Coef,
     U_LRI_struct, V_SM, detector_depth_km, earth_scheme, earth_yk, ext) = args

    dr_earth_km = baseline_km / N_steps_earth
    dr_earth_eV_inv = dr_earth_km * 1e3 * m_to_eV_inv
    rho_earth = rho_surf_avg.copy()
    U_step_buffer = np.zeros((3, 3), dtype=np.complex128)
    H_buf = np.zeros((3, 3), dtype=np.complex128)
    T_buf = np.zeros((3, 3), dtype=np.complex128)
    if ext is not None:
        apply_earth_exterior(E_bins, rho_earth, M_rot, Coef, U_LRI_struct, n_n_amb_eV3, ext)
    if earth_scheme == "avg":
        cross_ls = _earth_layer_crossings(baseline_km, costh, detector_depth_km)
        nbar = chord_density_seq(earth_yk)(baseline_km, costh, N_steps_earth, detector_depth_km,
                                           n_n_amb_eV3, cross_ls, _GL10_X, _GL10_W, earth_yk)
        for step in range(N_steps_earth):
            V_LRI = np.ascontiguousarray((Coef * nbar[step]) * U_LRI_struct + V_SM)
            _evolve_rho_all_bins(E_bins, rho_earth, M_rot, V_LRI, dr_earth_eV_inv,
                                 H_buf, U_step_buffer, T_buf)
    else:
        for step in range(N_steps_earth):
            l_km = step * dr_earth_km
            step_earth_transit(l_km, baseline_km, costh, dr_earth_eV_inv, E_bins, rho_earth, M_rot,
                               n_n_amb_eV3, Coef, U_LRI_struct, V_SM,
                               detector_depth_km, U_step_buffer, H_buf, T_buf, earth_yk)
    return rho_earth

def _earth_chord_unitary_worker(args) -> np.ndarray:
    """_earth_chord_worker's crossing as its evolution operator (N_E, 3, 3): the exterior's phase and steps, then the
    chord's steps, accumulated (_evolve_U_all_bins) instead of applied to one density matrix (2026-10-18)."""
    (costh, baseline_km, N_steps_earth, E_bins, M_rot, n_n_amb_eV3, Coef, U_LRI_struct, V_SM, detector_depth_km,
     earth_scheme, earth_yk, ext) = args
    N_E = len(E_bins)
    U = np.zeros((N_E, 3, 3), dtype=np.complex128)
    for a in range(3):
        U[:, a, a] = 1.0
    S_buf = np.zeros((3, 3), dtype=np.complex128)
    H_buf = np.zeros((3, 3), dtype=np.complex128)
    T_buf = np.zeros((3, 3), dtype=np.complex128)
    if ext is not None:
        U_dcol, L_eV_inv, nbar_ext = ext
        U = np.ascontiguousarray(np.einsum("ab,nbc->nac", U_dcol, U))
        for j in range(len(L_eV_inv)):
            if L_eV_inv[j] <= 0.0:
                continue
            V = np.ascontiguousarray((Coef * (n_n_amb_eV3 + nbar_ext[j])) * U_LRI_struct, dtype=np.complex128)
            _evolve_U_all_bins(E_bins, U, M_rot, V, L_eV_inv[j], H_buf, S_buf, T_buf)
    dr_earth_km = baseline_km / N_steps_earth
    dr_earth_eV_inv = dr_earth_km * 1e3 * m_to_eV_inv
    if earth_scheme == "avg":
        cross_ls = _earth_layer_crossings(baseline_km, costh, detector_depth_km)
        nbar = chord_density_seq(earth_yk)(baseline_km, costh, N_steps_earth, detector_depth_km,
                                           n_n_amb_eV3, cross_ls, _GL10_X, _GL10_W, earth_yk)
        for step in range(N_steps_earth):
            V_LRI = np.ascontiguousarray((Coef * nbar[step]) * U_LRI_struct + V_SM)
            _evolve_U_all_bins(E_bins, U, M_rot, V_LRI, dr_earth_eV_inv, H_buf, S_buf, T_buf)
    else:
        for step in range(N_steps_earth):
            n_tot = earth_total_density(step * dr_earth_km, baseline_km, costh, detector_depth_km, n_n_amb_eV3,
                                        earth_yk)
            V_LRI = np.ascontiguousarray((Coef * n_tot) * U_LRI_struct + V_SM)
            _evolve_U_all_bins(E_bins, U, M_rot, V_LRI, dr_earth_eV_inv, H_buf, S_buf, T_buf)
    return U


def _split_E_blocks(N_E: int, n_units: int, n_workers: int):
    """Contiguous energy-index blocks for splitting spatial-transit tasks.

    Every energy bin evolves independently through the spatial transit (H depends on E
    only through its own bin), so a (path-or-chord, E-block) pair is just as valid a
    unit of parallel work as a whole path or chord -- and much better balanced when the
    unit count doesn't divide the worker count (e.g. the default 12 Earth chords on 8
    workers leave 4 workers idle for half the phase). Blocks are sized so that
    n_units * n_blocks comfortably exceeds the worker count. Returns [(0, N_E)] (no
    splitting) when a single worker is in use, preserving the sequential path exactly.
    """
    if n_workers <= 1 or n_units <= 0:
        return [(0, N_E)]
    blocks_per_unit = int(np.ceil(2.0 * n_workers / n_units))
    blocks_per_unit = max(1, min(blocks_per_unit, N_E))
    edges = np.linspace(0, N_E, blocks_per_unit + 1).astype(np.int64)
    return [(int(edges[b]), int(edges[b + 1])) for b in range(blocks_per_unit)
            if edges[b + 1] > edges[b]]

# =============================================================================
# PERSISTENT SPATIAL POOL (standalone-solver side)
# =============================================================================
# A single run_solver call previously created and destroyed FOUR multiprocessing
# pools (MW + Earth phase, neutrino + antineutrino pass), and a scan re-pays that
# per point. This module-level pool is created lazily at first use, resized
# only when the requested worker count changes, torn down at exit. It is only ever
# touched from processes that call the spatial handover with n_cores > 1 and no
# caller-managed external_pool -- scanner/analysis pool workers run with n_cores=1
# and never reach it (no nested pools).
_spatial_pool = None
_spatial_pool_size = None
_spatial_pool_atexit = False


def _close_spatial_pool():
    global _spatial_pool
    if _spatial_pool is not None:
        _spatial_pool.terminate()
        _spatial_pool.join()
        _spatial_pool = None


def _get_spatial_pool(n_workers: int):
    """Returns the persistent spatial pool with exactly n_workers processes (creating
    or resizing it as needed), or None for n_workers <= 1."""
    global _spatial_pool, _spatial_pool_size, _spatial_pool_atexit
    if n_workers <= 1:
        return None
    if _spatial_pool is not None and _spatial_pool_size == n_workers:
        return _spatial_pool
    _close_spatial_pool()
    _spatial_pool = multiprocessing.Pool(processes=n_workers)
    _spatial_pool_size = n_workers
    if not _spatial_pool_atexit:
        import atexit
        atexit.register(_close_spatial_pool)
        _spatial_pool_atexit = True
    return _spatial_pool


def _run_spatial_tasks(worker_fn, tasks, desc, n_workers, show_progress, external_pool=None):
    """Dispatches independent spatial-handover tasks (MW paths / Earth chords) either
    sequentially, across a freshly-created multiprocessing pool, or across a caller-managed
    persistent pool (external_pool), depending on n_workers and whether one was supplied.

    external_pool exists for callers that invoke this many times in a tight loop (e.g. a
    minimizer evaluating the physics once per iteration) and want to reuse the same worker
    processes across all of them, rather than spinning up a fresh Pool every call. Creating a
    Pool is cheap under the 'fork' start method (a copy-on-write process clone) but expensive
    under 'spawn' (a full interpreter relaunch plus reimport per worker), which the analysis
    workers use. Without reuse, using more than one core here made every evaluation of the
    retired FrequentistAnalysis' global fit roughly 10x *slower*, not faster (measured directly). With external_pool=None (the
    default), the module-level persistent pool (_get_spatial_pool) is used instead of the
    historical per-call Pool creation -- a single run previously built and tore down four
    pools (MW/Earth x nu/nubar), and scans re-paid that per point.
    """
    if n_workers > 1 and len(tasks) > 1:
        pool = external_pool if external_pool is not None else _get_spatial_pool(n_workers)
        if pool is not None:
            iterator = pool.imap(worker_fn, tasks)
            if dp_core.HAS_TQDM and show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc=desc, leave=False)
            results = list(iterator)
        else:
            with multiprocessing.Pool(processes=min(n_workers, len(tasks))) as pool:
                iterator = pool.imap(worker_fn, tasks)
                if dp_core.HAS_TQDM and show_progress:
                    iterator = tqdm(iterator, total=len(tasks), desc=desc, leave=False)
                results = list(iterator)
    else:
        iterator = tqdm(tasks, desc=desc, leave=False) if (dp_core.HAS_TQDM and show_progress) else tasks
        results = [worker_fn(t) for t in iterator]
    return results

def lri_coefficient(m_Zd: float, g_D: float, eps_prime: float, u_arr: np.ndarray, T_nu0: float,
                    use_debye_screening: bool, eta_bg: np.ndarray, is_antineutrino: bool, debye_sq=None,
                    potential_product=None):
    """(Coef, m_eff): the dark-portal potential per unit neutron density today, Coef = g_D eps' / m_eff^2 x
    4 c_W m_Z^2 / g x G_F / sqrt(2) (eV^-2; negative for antineutrinos), and the mediator's effective mass
    m_eff = sqrt(m_Zd^2 + m_D^2) with the relic neutrinos' Debye mass when screening is on. Shared by the
    spatial handover and the Galactic emission points (dp_galactic). debye_sq: the Debye mass squared to use
    (dp_absorption.debye_mass_sq, the default since 2026-10-14), or None for the theory notes' Eq. T23."""
    if debye_sq is not None:
        m_D_sq = float(debye_sq) if use_debye_screening else 0.0
    else:
        m_D_sq = calc_debye_mass_sq(g_D, u_arr, T_nu0, eta_bg) if use_debye_screening else 0.0
    m_eff = np.sqrt(m_Zd**2 + m_D_sq)

    c_W = 0.88
    m_Z = 91.18 * GeV
    g_weak = 0.65
    pp = g_D * eps_prime if potential_product is None else float(potential_product)
    Coef = pp / (m_Zd**2 + m_D_sq) * (4.0 * c_W * m_Z**2 / g_weak) * (np.sqrt(2.0)/2.0 * G_F)
    if is_antineutrino:
        Coef = -Coef
    return Coef, m_eff


def transit_to_earth(E_bins: np.ndarray, F_mass_arrival: np.ndarray, M_rot: np.ndarray,
                     U_LRI_struct: np.ndarray, Coef: float, m_eff: float, n_n0_eV3: float,
                     R_H_eV_inv: float, earth_yk, n_sun_paths: int = 8,
                     transit_dtheta: float = 0.002, sun_path_cos=None, arrival: str = "eigen",
                     matter_charge=None, m_bare=None):
    """The transit from the cosmological arrival to the sphere r = r_split around the Earth
    (dp_transit's module notes): the adiabatic Galactic map at the Yukawa-smoothed local
    density, the coherent transit through the Sun's potential (n_sun_paths directions) and the
    radial transit through the Earth's exterior beyond r_split.

    Arguments: E_bins; F_mass_arrival (N_E, 3) -- populations of the cosmological arrival's
    eigenstates; M_rot; U_LRI_struct (this kind's texture); Coef (signed, eV^-2); m_eff (eV);
    n_n0_eV3 -- the cosmic neutron density, taken out to the Hubble radius R_H_eV_inv; earth_yk;
    n_sun_paths; transit_dtheta -- the frame table's spacing (dp_transit.build_frame_table);
    sun_path_cos -- None (the isotropic Gauss-Legendre average over n_sun_paths directions) or the
    cosine of one angle to the Sun, for that single path (per-direction Sun weighting: the average is
    linear, so the paths' results weighted by w_p return the isotropic one; DIRECTIONS_PLAN step 4);
    arrival -- 'eigen' (F_mass_arrival as above) or 'flavor': F_mass_arrival holds instead the flavor
    fractions of neutrinos emitted inside the Galaxy at the local Galactic potential (n_gal, without the
    Moon's, which only the Earth-Moon system feels), which reach the Sun decohered in the local eigenstates
    (dp_transit.flavor_populations; the Galactic component, DIRECTIONS_PLAN step 5); matter_charge -- None
    (neutrons) or a dp_matter.MatterCharge: the Galaxy's, the Sun's and the Moon's densities are then that
    charge's (n_n0_eV3 stays the cosmic NEUTRON density, the neutron species' floor; the charge carries the
    electrons'), and earth_yk must be the same charge's table; m_bare -- None, or the mediator's bare mass when
    m_eff includes the relics' screening: the Galaxy, the Sun and the Moon are screened (range 1/m_eff) but the
    uniform cosmic mean beyond the Galaxy is not (decided 2026-10-14), so its unscreened potential replaces the
    screened one in the Galactic integral's floor, n0 m_eff^2 [Y(m_bare) - Y(m_eff)] with Y the uniform medium's
    Yukawa factor out to the Hubble radius (dp_cosmo.yukawa_horizon_inv_m2) -- zero without screening.

    Returns (rho (N_E, 3, 3) at r_split, n_amb_eV3 -- the ambient density the chords add:
    Galaxy, Sun and Moon at the Earth -- and a dict of diagnostics).
    """
    import dp_transit
    S_signed = np.ascontiguousarray(np.sign(Coef) * np.asarray(U_LRI_struct, dtype=np.complex128))
    if Coef == 0.0:
        S_signed = np.ascontiguousarray(np.asarray(U_LRI_struct, dtype=np.complex128))
    tab = dp_transit.frame_table(M_rot, S_signed, transit_dtheta)
    coef_abs = abs(float(Coef))
    if matter_charge is None:
        n_gal = dp_transit.galactic_effective_density(m_eff, n_n0_eV3, R_H_eV_inv) if m_eff > 0 else 0.0
        N_sun_q, N_moon_q = N_n_sun, N_n_moon
        n_floor = n_n0_eV3
    else:
        import dp_matter
        n_gal = (dp_matter.galactic_effective_density(matter_charge, m_eff, n_n0_eV3, R_H_eV_inv)
                 if m_eff > 0 else 0.0)
        N_sun_q, N_moon_q, _ = dp_matter.point_sources(matter_charge)
        n_floor = dp_matter.cosmic_density(matter_charge, n_n0_eV3)
    if m_bare is not None and m_eff > m_bare and m_eff > 0:
        from dp_cosmo import yukawa_horizon_inv_m2
        n_gal = n_gal + n_floor * m_eff * m_eff * (yukawa_horizon_inv_m2(m_bare * m_bare, R_H_eV_inv)
                                                   - yukawa_horizon_inv_m2(m_eff * m_eff, R_H_eV_inv))
    n_sun, n_moon = 0.0, 0.0
    if m_eff > 0:
        n_sun = N_sun_q * (m_eff**2) / (4.0 * np.pi * d_sun_eV_inv) * np.exp(-m_eff * d_sun_eV_inv)
        n_moon = N_moon_q * (m_eff**2) / (4.0 * np.pi * d_moon_eV_inv) * np.exp(-m_eff * d_moon_eV_inv)
    n_bg = n_gal + n_moon
    N_E = len(E_bins)
    E_c = np.ascontiguousarray(E_bins, dtype=np.float64)
    F_arr = np.ascontiguousarray(F_mass_arrival, dtype=np.float64)
    if arrival == "flavor":
        # born far from the Earth-Moon system, at the Galaxy's potential alone; the map below then carries the
        # populations adiabatically onto the eigenstates at n_bg (which adds the Moon), as for every arrival
        F_pop = np.empty_like(F_arr)
        dp_transit.flavor_populations(E_c, F_arr, 2.0 * coef_abs, n_gal, tab["Q"], tab["Mp"], tab["s"],
                                      tab["floor"], F_pop)
        F_arr = F_pop
    elif arrival != "eigen":
        raise ValueError(f"arrival must be 'eigen' or 'flavor', got {arrival!r}")
    rho = np.empty((N_E, 3, 3), dtype=np.complex128)
    dp_transit.galactic_map(E_c, F_arr, 2.0 * coef_abs, n_bg, tab["Q"], tab["Mp"], tab["s"], tab["floor"], rho)
    if sun_path_cos is None:
        ca, w = dp_transit.sun_paths(n_sun_paths)
    else:
        ca, w = np.array([float(sun_path_cos)]), np.array([1.0])
    rho, n_active_sun = dp_transit.source_transit(E_bins, rho, tab, coef_abs, n_bg, n_sun, m_eff,
                                                  d_sun_eV_inv, ca, w)
    n_amb = n_bg + n_sun
    n_active_ext = 0
    if earth_exterior_active(m_eff) and int(earth_yk[0][0]) != _YUKAWA_CONTACT:
        R_km = float(earth_yk[0][2])
        r_split_km = _EXT_R_SPLIT * R_km
        C_E = _ext_density_cm3(r_split_km, earth_yk) * cm_inv3_to_eV3
        outer = _ext_line_column(0.0, r_split_km, r_split_km * 1e9, earth_yk, False, _GL10_X, _GL10_W)
        phase = coef_abs * float(np.max(np.abs(tab["s"]))) * outer * cm_inv3_to_eV3 * _KM_IN_EV_INV
        rho, n_active_ext = dp_transit.source_transit(E_bins, rho, tab, coef_abs, n_amb, C_E, m_eff,
                                                      r_split_km * _KM_IN_EV_INV, np.array([-1.0]),
                                                      np.array([1.0]), force_active=phase > 1e-7)
    info = dict(n_gal_eV3=n_gal, n_sun_eV3=n_sun, n_moon_eV3=n_moon, n_active_sun=n_active_sun,
                n_active_exterior=n_active_ext, table_nodes=len(tab["sig"]))
    return rho, n_amb, info


def _earth_setup(M_rot, u_arr, m_Zd, g_D, eps_prime, T_nu0, use_debye_screening, eta_bg, detector_depth_km,
                 use_SM_potential, n_e_earth_eV3, is_antineutrino, n_sun_paths, n_earth_chords, U_LRI_struct,
                 V_SM_struct, use_gpu, earth_step_scheme, earth_zenith_average, earth_sky, earth_band_costh,
                 earth_nodes_per_band, matter_charge, debye_sq_z0, potential_product) -> dict:
    """Everything of the spatial handover that does not depend on its input (_propagate_earth_spatial_direct's
    arguments of the same names): the checks, the chords, the potential's coefficient and effective mass, the Earth's
    Yukawa table, the texture, the standard matter potential and each chord's exterior. Returns a dict of them."""
    if earth_zenith_average not in _ZENITH_AVERAGES:
        raise ValueError(f"earth_zenith_average must be one of {_ZENITH_AVERAGES}, got {earth_zenith_average!r}")
    if earth_sky not in ("full", "up"):
        raise ValueError(f"earth_sky must be 'full' or 'up', got {earth_sky!r}")
    if earth_zenith_average == "phase" and earth_step_scheme != "avg":
        raise ValueError("The 'phase' zenith average needs the phase-exact earth_step_scheme = 'avg' "
                         "(under 'left' each chord's phase carries a Riemann error); use 'plain' with 'left'.")
    if int(n_sun_paths) < 1:
        raise ValueError(f"n_sun_paths must be >= 1, got {n_sun_paths}")

    # --- Dynamic Earth Chords ---
    # n_earth_chords evenly spaced chords from exact core transit (costh = -1) to the horizon
    # (0); with earth_sky = 'full' (decided 2026-10-13) the down-going hemisphere gets the
    # mirror image, up to straight down (+1), whose chords cross only the ice above the
    # detector (1.5 km straight down, ~140 km at the horizon). The zenith average then
    # weights every cos(zenith) of the whole sky equally.
    if earth_band_costh is not None:
        earth_band_costh = np.asarray(earth_band_costh, dtype=np.float64)
        top = 0.0 if earth_sky == "up" else 1.0
        if abs(earth_band_costh[0] + 1.0) > 1e-12 or abs(earth_band_costh[-1] - top) > 1e-12:
            raise ValueError(f"earth_band_costh must run from -1 to {top:g} for earth_sky = '{earth_sky}'")
        earth_costh = earth_chord_nodes(earth_band_costh, earth_nodes_per_band)
    elif n_earth_chords <= 1:
        earth_costh = np.array([-1.0]) if earth_sky == "up" else np.array([-1.0, 1.0])
    else:
        up = np.linspace(-1.0, 0.0, int(n_earth_chords))
        earth_costh = up if earth_sky == "up" else np.concatenate([up, -up[-2::-1]])

    Coef, m_eff = lri_coefficient(m_Zd, g_D, eps_prime, u_arr, T_nu0, use_debye_screening, eta_bg, is_antineutrino,
                                  debye_sq=debye_sq_z0, potential_product=potential_product)
    # A general matter charge (dp_matter, docs/dev/SCOPE_WIDENING_PLAN.md): its sign rides on the coefficient,
    # its densities (Galaxy, Sun, Moon, the Earth's table) are its own. Neutrons: None, the historical path.
    cpg = None
    if matter_charge is not None:
        import dp_matter
        Coef = Coef * matter_charge.sign
        cpg = dp_matter.earth_cpg(matter_charge)
        if earth_step_scheme != "avg":
            raise ValueError("A matter charge other than the neutron's needs earth_step_scheme = 'avg'")
        if use_gpu:
            raise ValueError("The GPU path supports the neutron charge only")

    # The Earth's own neutrons enter through their spherical Yukawa integral over the mediator
    # range 1/m_eff (see earth_yukawa_table); at MeV masses this is the contact density.
    earth_yk = earth_yukawa_table(m_eff, cpg=cpg)

    if U_LRI_struct is None:
        # Default texture: outer product of the sterile mixing vector. u_arr arrives
        # pre-conjugated for antineutrinos, so this construction is automatically in the
        # right convention for either case.
        U_LRI_struct = np.outer(u_arr, np.conjugate(u_arr))
    U_LRI_struct = np.ascontiguousarray(U_LRI_struct, dtype=np.complex128)

    # The SM matter potential is constant along every chord (it depends only on the
    # configured electron density, the flavor structure, and the antineutrino sign
    # convention -- never on position), so it is built exactly once here and handed to
    # step_earth_transit prebuilt. V_SM_struct defaults to the standard charged-current
    # texture diag(1, 0, 0); a user override arrives here already conjugated for
    # antineutrinos (propagate_spectrum's convention).
    V_SM = np.zeros((3, 3), dtype=np.complex128)
    if use_SM_potential:
        if V_SM_struct is None:
            S_SM = np.zeros((3, 3), dtype=np.complex128)
            S_SM[0, 0] = 1.0
        else:
            S_SM = np.asarray(V_SM_struct, dtype=np.complex128)
        V_SM_scale = np.sqrt(2.0) * G_F * n_e_earth_eV3
        if is_antineutrino:
            V_SM_scale = -V_SM_scale
        V_SM = V_SM_scale * S_SM
    V_SM = np.ascontiguousarray(V_SM)

    # Each chord's exterior below r_split (None when the exterior is out of range)
    exterior = earth_exterior_active(m_eff) and int(earth_yk[0][0]) != _YUKAWA_CONTACT
    ext_data = [chord_exterior(c, detector_depth_km, earth_yk, Coef, U_LRI_struct) if exterior else None
                for c in earth_costh]
    return dict(earth_band_costh=earth_band_costh, earth_costh=earth_costh, Coef=Coef, m_eff=m_eff, cpg=cpg,
                earth_yk=earth_yk, U_LRI_struct=U_LRI_struct, V_SM=V_SM, ext_data=ext_data)


def _propagate_earth_spatial_direct(E_bins: np.ndarray, F_mass_arrival: np.ndarray,
                            M_rot: np.ndarray,
                            u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
                            T_nu0: float, use_debye_screening: bool, eta_bg: np.ndarray,
                            detector_depth_km: float, use_SM_potential: bool, n_e_earth_eV3: float,
                            is_antineutrino: bool, n_sun_paths: int, n_earth_chords: int,
                            transit_dtheta: float = 0.002, n_n0_eV3: float = 0.0,
                            R_H_eV_inv: float = 0.0,
                            show_progress: bool = True, verbose: int = 1, n_cores: int = 1,
                            N_steps_earth: int = 500, external_pool=None,
                            U_LRI_struct=None, V_SM_struct=None, use_gpu: bool = False,
                            earth_step_scheme: str = "left",
                            earth_zenith_average: str = "plain",
                            earth_sky: str = "up",
                            earth_band_costh=None, earth_nodes_per_band: int = 2,
                            sun_path_cos=None, arrival: str = "eigen", matter_charge=None, debye_sq_z0=None,
                            potential_product=None):
    """
    The spatial handover: from the cosmological arrival to the detector, through the Galaxy,
    the Sun, the Earth's exterior and the Earth (transit_to_earth, then the chords). Earth
    chords are mutually independent; with n_cores > 1 they are distributed across a
    multiprocessing pool.

    Arguments:
      E_bins: energy grid (eV).
      F_mass_arrival: (N_E, 3) populations of the cosmological arrival's eigenstates
        (ascending eigenvalues of the Hamiltonian at z = 0).
      M_rot: (3, 3) U_pmns @ M_sq @ conj(U_pmns).T, precomputed once per run.
      u_arr: (3,) complex sterile mixing amplitudes, pre-conjugated for antineutrinos.
      m_Zd, g_D, eps_prime: dark mediator mass (eV) and couplings.
      T_nu0: CnuB temperature (eV); use_debye_screening: thermal Debye mass toggle;
      eta_bg: (3,) CnuB flavor asymmetries (all three feed the screening mass).
      detector_depth_km: detector depth below the surface (sets chord geometry).
      use_SM_potential / n_e_earth_eV3: standard MSW potential inside the Earth.
      is_antineutrino: flips the potential sign (Coef) for this call.
      n_sun_paths: directions to the Sun averaged in its transit (Gauss-Legendre in the
        cosine of the angle); n_earth_chords: zenith-average count.
      transit_dtheta: largest turn of the eigenbasis between the transit's frame-table nodes
        (dp_transit.build_frame_table); the transit's error scales with it (~1e-5 at 0.002).
      n_n0_eV3, R_H_eV_inv: the cosmic neutron density and the Hubble radius, for the Galactic
        potential's long-range part (dp_transit.galactic_effective_density).
      show_progress, verbose: progress bars / logging.
      n_cores: pool size for the (chord, E-block) task distribution.
      N_steps_earth, earth_step_scheme: Earth-transit resolution and potential
        discretization -- see the dedicated paragraph below.
      earth_zenith_average: 'phase' (integrate each block of the chords' density matrices
        against its LRI phase over costh -- see average_earth_chords; the config default)
        or 'plain' (historical equal-weight mean, the signature default);
        'phase' requires earth_step_scheme = 'avg'.
      earth_sky: 'full' (the whole sky, equal weight per costh; the config default since
        2026-10-13) or 'up' (up-going directions only, the signature default and the only
        option until then).
      external_pool, U_LRI_struct, V_SM_struct, use_gpu: see below.

      earth_band_costh: None (the default: one whole-sky flux, as always) or the cos(zenith) edges of
        declination bands, increasing over the sky's range ([-1, 1] for 'full', [-1, 0] for 'up'):
        the chords are then placed on the band edges, earth_nodes_per_band intervals per band
        (earth_chord_nodes; n_earth_chords is unused), and each band's flux is returned as well
        (average_earth_chords_bands; per-direction output, docs/dev/DIRECTIONS_PLAN.md step 2).
      sun_path_cos: None (the Sun averaged isotropically, as always) or one cosine of the angle to the
        Sun: the transit along that single path (transit_to_earth), for per-direction Sun weighting.
      arrival: 'eigen' (F_mass_arrival as above, the extragalactic flux) or 'flavor' (F_mass_arrival holds
        the flavor fractions of neutrinos emitted inside the Galaxy at the local potential: the Galactic
        component, transit_to_earth).

    Returns: (N_E, 3) flavor fluxes at the detector depth; with earth_band_costh, the tuple
    (whole-sky (N_E, 3), per band (N_b, N_E, 3)).

    Until 2026-10-13 the Milky Way phase integrated n_mw_paths (6) trajectories from 100 kpc in
    N_steps_MW (25) left-endpoint steps with the local Galactic density at every mediator mass,
    the Sun, the Moon and the Earth as point sources sampled at those steps, and the Earth's
    potential started at its surface. That treatment broke adiabaticity with its 4-kpc steps,
    lost the vacuum term to rounding once the potential exceeded it ~1e16 times, overstated the
    Galactic potential up to 5e6 times at ranges beyond a kiloparsec, and applied the Sun's
    potential only along the Earth chords (notes, Appendix on the transit).

    external_pool, if given, is a caller-managed multiprocessing.Pool reused across many
    calls instead of creating a fresh one each time -- see _run_spatial_tasks's docstring.

    The Earth's own neutrons enter through their spherical Yukawa integral over the mediator
    range 1/m_eff (earth_yukawa_table, built once per call from m_eff); at MeV masses that is
    the local density bit for bit, and it differs once the range approaches PREM's layer
    thicknesses (m_eff <~ 1e-9 eV). Below 2e-5 eV the chords start at r_split = 2 R, crossing
    the Earth's exterior first (see the section notes on it).

    U_LRI_struct / V_SM_struct, if given, are dimensionless 3x3 Hermitian flavor structure
    matrices overriding the default LRI texture (u_i * conj(u_j)) and SM matter-potential
    texture (diag(1, 0, 0)) respectively. IMPORTANT CONVENTION: this function expects them
    already conjugated for antineutrinos, exactly as u_arr itself arrives pre-conjugated --
    propagate_spectrum handles that conjugation centrally, so pass neutrino-convention
    matrices there, not here. The overrides affect only the coherent forward-scattering
    potentials; the Debye screening mass (and, upstream, the thermal attenuation and
    regeneration rates) remain tied to the u_e/u_mu/u_tau mixing amplitudes.

    earth_step_scheme: 'avg' (exact step-averaged potential; step-count-independent
    LRI phase, certified converged -- the config default) or 'left' (historical
    left-endpoint sampling). The function-signature defaults (N_steps_earth=500,
    earth_step_scheme='left') preserve historical behavior for direct callers;
    run_solver / the analysis pipeline always pass the DEFAULT_CONFIG values
    ('avg', 25). See _earth_chord_worker and the scheme NOTE below.
    """
    N_E = len(E_bins)
    lbl = "Anti-Nu" if is_antineutrino else "Nu"
    n_workers = max(1, min(int(n_cores), available_cores()))
    st = _earth_setup(M_rot, u_arr, m_Zd, g_D, eps_prime, T_nu0, use_debye_screening, eta_bg, detector_depth_km,
                      use_SM_potential, n_e_earth_eV3, is_antineutrino, n_sun_paths, n_earth_chords,
                      U_LRI_struct, V_SM_struct, use_gpu, earth_step_scheme, earth_zenith_average, earth_sky,
                      earth_band_costh, earth_nodes_per_band, matter_charge, debye_sq_z0, potential_product)
    earth_band_costh, earth_costh, Coef, m_eff, cpg, earth_yk, U_LRI_struct, V_SM, ext_data = (
        st[k] for k in ("earth_band_costh", "earth_costh", "Coef", "m_eff", "cpg", "earth_yk", "U_LRI_struct", "V_SM",
                        "ext_data"))

    # PHASE 1: Galaxy, Sun and the Earth's exterior beyond r_split (dp_transit)
    if verbose == 2:
        safe_print(f"\n--- Spatial Handover Commencing ({lbl}) ---")
    rho_surf_avg, n_n_amb_eV3, tinfo = transit_to_earth(
        E_bins, F_mass_arrival, M_rot, U_LRI_struct, Coef, m_eff, n_n0_eV3, R_H_eV_inv, earth_yk,
        n_sun_paths=n_sun_paths, transit_dtheta=transit_dtheta, sun_path_cos=sun_path_cos,
        arrival=arrival, matter_charge=matter_charge,
        m_bare=(m_Zd if debye_sq_z0 is not None else None))
    if verbose == 2:
        safe_print(f"Transit to the Earth: Galactic density {tinfo['n_gal_eV3']:.3e} eV^3, Sun "
                   f"{tinfo['n_sun_eV3']:.3e} eV^3 ({tinfo['n_active_sun']} bins coherent), Earth's "
                   f"exterior {tinfo['n_active_exterior']} bins; frame table {tinfo['table_nodes']} nodes.")

    # PHASE-NOISE ADVISORY: when the LRI potential is so large (1/m_Zd^2-enhanced at tiny
    # mediator masses) that a single Earth step accrues more phase than double precision
    # can represent (eps * spread * dr > 0.1 rad of rounding noise), propagator_3x3 keeps
    # the evolution exactly unitary but the oscillation phases carry no information --
    # the returned fluxes are decoherence-averaged only. Estimated here with the
    # PREM-core density (the worst case along any chord).
    if verbose >= 1:
        s_lri = float(np.max(np.abs(np.linalg.eigvalsh(U_LRI_struct))))
        n_core_cm3 = yukawa_profile_density(0.0, earth_yk) if cpg is not None else earth_neutron_density_cm3(0.0, earth_yk)
        n_core_eV3 = n_core_cm3 * cm_inv3_to_eV3
        dr_core_eV_inv = (get_baseline_earth(detector_depth_km, -1.0) / N_steps_earth) * 1e3 * m_to_eV_inv
        if 2.2e-16 * abs(Coef) * n_core_eV3 * s_lri * dr_core_eV_inv > 0.1:
            safe_print(
                f"[WARNING] LRI potential scale (|Coef|*n_core ~ {abs(Coef) * n_core_eV3 * s_lri:.2e} eV) "
                f"is phase-noise-dominated for the Earth transit at these parameters "
                f"(m_Zd_eV={m_Zd:.3g}, g_D={g_D:.3g}, epsilon_prime={eps_prime:.3g}): "
                f"per-step phases exceed double precision. Fluxes are unitary but "
                f"decoherence-averaged only (see propagator_3x3's docstring).")

    # Optional CUDA path: the chords as one batched GPU workload (same physics -- shared
    # density helpers, same propagator algorithm ported to batched arrays; see
    # gpu_spatial.py, including its validation-status note). Falls back to the CPU path
    # with a warning when cupy or a CUDA device is missing.
    if use_gpu:
        try:
            from gpu_spatial import gpu_available, run_spatial_handover_gpu
            _gpu_ok = gpu_available()
        except ImportError:
            _gpu_ok = False
        if not _gpu_ok:
            safe_print("[GPU] use_gpu requested but cupy with a CUDA device is unavailable -- using the CPU path.")
        else:
            if verbose == 2:
                safe_print(f"\n--- Earth chords on GPU ({lbl}): {len(earth_costh)} chord(s) ---")
            baselines = np.array([get_baseline_earth(detector_depth_km, c) for c in earth_costh])
            dr_earth_km = baselines / N_steps_earth
            n_seq_earth = np.empty((len(earth_costh), N_steps_earth))
            for c in range(len(earth_costh)):
                if earth_step_scheme == "avg":
                    cross_ls = _earth_layer_crossings(baselines[c], earth_costh[c], detector_depth_km)
                    n_seq_earth[c, :] = _avg_earth_density_seq(baselines[c], earth_costh[c],
                                                               N_steps_earth, detector_depth_km,
                                                               n_n_amb_eV3, cross_ls, _GL10_X, _GL10_W,
                                                               earth_yk)
                else:
                    for s in range(N_steps_earth):
                        n_seq_earth[c, s] = earth_total_density(s * dr_earth_km[c], baselines[c],
                                                                earth_costh[c], detector_depth_km,
                                                                n_n_amb_eV3, earth_yk)
            rho0_chords = np.broadcast_to(rho_surf_avg, (len(earth_costh), N_E, 3, 3)).copy()
            for c in range(len(earth_costh)):
                if ext_data[c] is not None:
                    apply_earth_exterior(E_bins, rho0_chords[c], M_rot, Coef, U_LRI_struct, n_n_amb_eV3,
                                         ext_data[c])
            rho_chords = run_spatial_handover_gpu(E_bins, rho0_chords, n_seq_earth,
                                                  dr_earth_km * 1e3 * m_to_eV_inv,
                                                  M_rot, Coef, U_LRI_struct, V_SM)
            rho_earth_avg = average_earth_chords(rho_chords, earth_costh, Coef, U_LRI_struct,
                                                    detector_depth_km, n_n_amb_eV3, m_eff, earth_yk,
                                                    earth_zenith_average)
            F_earth = np.real(np.diagonal(rho_earth_avg, axis1=1, axis2=2)).copy()
            if verbose == 2:
                safe_print(f"Spatial propagation finished ({lbl}, GPU).")
            if earth_band_costh is not None:
                rho_bands = average_earth_chords_bands(rho_chords, earth_costh, earth_band_costh, Coef,
                                                       U_LRI_struct, detector_depth_km, n_n_amb_eV3, m_eff,
                                                       earth_yk, earth_zenith_average)
                return F_earth, np.real(np.diagonal(rho_bands, axis1=2, axis2=3)).copy()
            return F_earth

    # PHASE 2: Earth Transit (PREM Layers)
    if verbose == 2:
        safe_print(f"Initiating the Earth transit ({len(earth_costh)} chord(s), {n_workers} worker(s))...")

    # NOTE ON STEP SCHEMES ('avg' vs 'left', grid.earth_step_scheme): under the default
    # 'avg' scheme each step uses the EXACT path-averaged potential, so the accumulated
    # LRI phase equals Coef * integral(n dl) exactly for ANY step count -- measured at
    # the production validation points (below/inband/strong classes), fluxes at
    # N_steps_earth = 8 and 1000 agree to <= 4.5e-9 with energy-averaged ratios to
    # <= 6.5e-10, which is what justifies the N_steps_earth = 25 default (a ~3x
    # margin over the observed knee; revalidated 100 vs 25 at the four standard
    # points on adoption). The historical 'left' scheme (V sampled at each
    # step's left endpoint) carries a Riemann-sum phase error of thousands of radians
    # that RE-ALIASES the Earth-transit fringes on every step-count change (500 vs 501
    # measured 1.6-17% pointwise flux / up to 1.7e-2 energy-averaged-ratio scatter);
    # everything in the historical note below applies to THAT scheme. Potential-
    # dominated points (m_Zd = 40 eV) were long thought pointwise-unconvergeable
    # (2.9e-3 between step counts); that was the non-unitary propagator before
    # 2026-10-12, and they now agree across 8, 25 and 100 steps and 12, 24 and 50
    # chords to 1e-14.
    #
    # HISTORICAL NOTE ('left' scheme): an earlier version scaled n_steps_earth down for
    # shorter chords (same physical km/step as the costh=-1 chord, since that seemed to only cost
    # resolution nobody needed -- verified via a synthetic, purely-diagonal test state to
    # agree with a 2000-step reference to 1e-12). That test state was misleading: the real
    # rho_surf_avg arriving from the Milky Way phase carries off-diagonal coherences, and at
    # production defaults this showed a genuine, energy-localized ~10% error for the shorter
    # chords, concentrated at the exact energy where the Earth matter potential drives a
    # near-degenerate (MSW-like) eigenvalue crossing -- near such a crossing, eigenvectors
    # rotate rapidly with position *independent of total path length*, so "shorter chord
    # needs fewer steps" is the wrong criterion. Fixed step count for every chord (now
    # caller-configurable via N_steps_earth; 500 was the pinned value throughout the
    # 'left'-scheme era, lowered to 100 once the 'avg' scheme's certificate landed) until a
    # scheme that actually tracks local eigenvalue splitting is worth the complexity.
    # WARNING: reducing N_steps_earth trades away exactly the resolution that protects
    # against the failure mode above. It has NOT been validated as safe at any value below
    # the 500 default across a resonance-scanning parameter range -- verify convergence
    # against a higher-resolution reference for your own m_Zd range before trusting a lower
    # value in a real fit.
    # (chord, E-block) task splitting (see _split_E_blocks).
    earth_blocks = _split_E_blocks(N_E, len(earth_costh), n_workers)
    earth_tasks = [(costh, get_baseline_earth(detector_depth_km, costh), N_steps_earth, E_bins[b0:b1],
                    rho_surf_avg[b0:b1], M_rot, n_n_amb_eV3, Coef, U_LRI_struct, V_SM,
                    detector_depth_km, earth_step_scheme, earth_yk, ext_data[c])
                   for c, costh in enumerate(earth_costh) for (b0, b1) in earth_blocks]
    earth_block_results = _run_spatial_tasks(_earth_chord_worker, earth_tasks, f"Earth chords ({lbl})", n_workers, show_progress, external_pool=external_pool)
    earth_results = []
    for c in range(len(earth_costh)):
        rho_full = np.empty((N_E, 3, 3), dtype=np.complex128)
        for b, (b0, b1) in enumerate(earth_blocks):
            rho_full[b0:b1] = earth_block_results[c * len(earth_blocks) + b]
        earth_results.append(rho_full)
    rho_chords = np.stack(earth_results, axis=0)
    rho_earth_avg = average_earth_chords(rho_chords, earth_costh, Coef,
                                            U_LRI_struct, detector_depth_km, n_n_amb_eV3, m_eff,
                                            earth_yk, earth_zenith_average)
    F_earth = np.real(np.diagonal(rho_earth_avg, axis1=1, axis2=2)).copy()

    if verbose == 2:
        safe_print(f"Spatial propagation finished ({lbl}). Flux density matrix mapped to detector depth.")

    if earth_band_costh is not None:
        rho_bands = average_earth_chords_bands(rho_chords, earth_costh, earth_band_costh, Coef, U_LRI_struct,
                                               detector_depth_km, n_n_amb_eV3, m_eff, earth_yk,
                                               earth_zenith_average)
        return F_earth, np.real(np.diagonal(rho_bands, axis1=2, axis2=3)).copy()
    return F_earth


# =============================================================================
# REUSE OF THE EARTH CROSSING WITHIN A PARAMETER POINT (2026-10-18)
# =============================================================================
# The spatial handover is linear in its input, per energy bin: every step maps the arriving populations to density
# matrices (dp_transit.galactic_map, flavor_populations), conjugates them by unitaries (the transits, the chords) or
# averages them (Sun paths, zenith chords and bands), and none of its branches depends on the input. So with
# T_k = the handover of the unit input e_k (every bin), the handover of any F is sum_k F[:, k] T_k -- exact up to
# rounding. An analysis point runs the handover tens of times with the same parameters (spectral-index nodes,
# source compositions, the Galactic responses), each through the same Earth: those calls now cost three handovers
# in total (the Galactic responses, _galactic_responses, are exactly the three unit inputs and fill the table for
# free). earth_transfer chooses: 'off' (the default: every call direct, bit for bit as before), 'repeat' (a
# parameter set's table built on its second call with a general input, so a one-off call costs what it always did)
# or 'always' (built on its first, so every result is the combination whatever the order of the calls: the
# analysis, whose sets all repeat -- one handover fewer per set, and runs computed by another process are bit for
# bit those computed here). EARTH_TRANSFER_REUSE = False forces the direct path everywhere.
EARTH_TRANSFER_REUSE = True
EARTH_TRANSFER_MODES = ("off", "repeat", "always")
_EARTH_TRANSFER_MAX = 32
_EARTH_TRANSFER = {}
_EARTH_NONRESULT_ARGS = ("F_mass_arrival", "show_progress", "verbose", "n_cores", "external_pool")


def _earth_transfer_columns(a: dict) -> list:
    """The handover's responses to the three unit inputs e_k, [(whole sky (N_E, 3), per band (N_b, N_E, 3) or None)]
    for k = 0, 1, 2: _propagate_earth_spatial_direct's arithmetic to rounding (a: its bound arguments), with each
    chord crossed once -- its evolution operator applied to the three arrivals -- and one zenith average for the
    three (their density matrices stacked along the energy axis, which the average does not mix)."""
    E_bins = np.ascontiguousarray(a["E_bins"], dtype=np.float64)
    N_E = len(E_bins)
    st = _earth_setup(a["M_rot"], a["u_arr"], a["m_Zd"], a["g_D"], a["eps_prime"], a["T_nu0"],
                      a["use_debye_screening"], a["eta_bg"], a["detector_depth_km"], a["use_SM_potential"],
                      a["n_e_earth_eV3"], a["is_antineutrino"], a["n_sun_paths"], a["n_earth_chords"],
                      a["U_LRI_struct"], a["V_SM_struct"], a["use_gpu"], a["earth_step_scheme"],
                      a["earth_zenith_average"], a["earth_sky"], a["earth_band_costh"], a["earth_nodes_per_band"],
                      a["matter_charge"], a["debye_sq_z0"], a["potential_product"])
    earth_costh, Coef, m_eff, earth_yk, S = st["earth_costh"], st["Coef"], st["m_eff"], st["earth_yk"], st["U_LRI_struct"]
    surf, n_amb = [], None
    for k in range(3):
        e = np.zeros((N_E, 3))
        e[:, k] = 1.0
        rho, n_amb, _ = transit_to_earth(
            E_bins, e, a["M_rot"], S, Coef, m_eff, a["n_n0_eV3"], a["R_H_eV_inv"], earth_yk,
            n_sun_paths=a["n_sun_paths"], transit_dtheta=a["transit_dtheta"], sun_path_cos=a["sun_path_cos"],
            arrival=a["arrival"], matter_charge=a["matter_charge"],
            m_bare=(a["m_Zd"] if a["debye_sq_z0"] is not None else None))
        surf.append(rho)
    n_workers = max(1, min(int(a["n_cores"]), available_cores()))
    blocks = _split_E_blocks(N_E, len(earth_costh), n_workers)
    tasks = [(costh, get_baseline_earth(a["detector_depth_km"], costh), a["N_steps_earth"], E_bins[b0:b1], a["M_rot"],
              n_amb, Coef, S, st["V_SM"], a["detector_depth_km"], a["earth_step_scheme"], earth_yk, st["ext_data"][c])
             for c, costh in enumerate(earth_costh) for (b0, b1) in blocks]
    res = _run_spatial_tasks(_earth_chord_unitary_worker, tasks, "Earth chords (operators)", n_workers, False,
                             external_pool=a["external_pool"])
    U = np.empty((len(earth_costh), N_E, 3, 3), dtype=np.complex128)
    for c in range(len(earth_costh)):
        for b, (b0, b1) in enumerate(blocks):
            U[c, b0:b1] = res[c * len(blocks) + b]
    # rho_k = U surf_k U^dagger per chord and bin: two batched products (2026-10-18; the three-operand einsum's
    # single loop took ~6% of a potential point) -- the same sums, equal to rounding
    UcT = np.conjugate(np.swapaxes(U, -1, -2))
    rho = np.concatenate([np.matmul(np.matmul(U, surf[k][None]), UcT) for k in range(3)], axis=1)
    rho = 0.5 * (rho + np.conjugate(np.swapaxes(rho, -1, -2)))              # Hermitian, as each step keeps it
    whole = average_earth_chords(rho, earth_costh, Coef, S, a["detector_depth_km"], n_amb, m_eff, earth_yk,
                                 a["earth_zenith_average"])
    W = np.real(np.diagonal(whole, axis1=1, axis2=2))
    B = None
    if st["earth_band_costh"] is not None:
        bands = average_earth_chords_bands(rho, earth_costh, st["earth_band_costh"], Coef, S, a["detector_depth_km"],
                                           n_amb, m_eff, earth_yk, a["earth_zenith_average"])
        B = np.real(np.diagonal(bands, axis1=2, axis2=3))
    return [(np.ascontiguousarray(W[k * N_E:(k + 1) * N_E]),
             None if B is None else np.ascontiguousarray(B[:, k * N_E:(k + 1) * N_E])) for k in range(3)]


def _earth_key_part(v):
    """A hashable, exact stand-in for one argument of the handover (arrays by dtype, shape and bytes)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.bool_, np.integer, np.floating)):
        return v.item()
    if isinstance(v, (complex, np.complexfloating)):
        return ("complex", float(v.real), float(v.imag))
    if hasattr(v, "_fields"):                         # NamedTuple (dp_matter.MatterCharge)
        return (type(v).__name__,) + tuple(_earth_key_part(x) for x in v)
    if isinstance(v, (np.ndarray, list, tuple)):
        a = np.ascontiguousarray(v)
        return (str(a.dtype), a.shape, a.tobytes())
    raise TypeError(f"cannot key the Earth transfer on a {type(v).__name__}")


def _unit_column(F: np.ndarray):
    """k if every row of F is exactly e_k, else None."""
    if F.ndim != 2 or F.shape[1] != 3 or F.shape[0] == 0:
        return None
    for k in range(3):
        if np.all(F[:, k] == 1.0) and np.all(np.delete(F, k, axis=1) == 0.0):
            return k
    return None


def _earth_signature():
    import inspect
    return inspect.signature(_propagate_earth_spatial_direct)


_EARTH_SIG = _earth_signature()


def propagate_earth_spatial(*args, earth_transfer: str = "off", **kwargs):
    """The spatial handover (_propagate_earth_spatial_direct, whose arguments and returns it takes), reusing the
    handover's response to the three unit inputs for repeated calls with the same parameters (section notes);
    earth_transfer: 'repeat' or 'always' (section notes)."""
    if earth_transfer not in EARTH_TRANSFER_MODES:
        raise ValueError(f"earth_transfer must be one of {EARTH_TRANSFER_MODES}, got {earth_transfer!r}")
    if not EARTH_TRANSFER_REUSE or earth_transfer == "off":
        return _propagate_earth_spatial_direct(*args, **kwargs)
    ba = _EARTH_SIG.bind(*args, **kwargs)
    ba.apply_defaults()
    a = dict(ba.arguments)
    if a["use_gpu"]:
        return _propagate_earth_spatial_direct(*args, **kwargs)
    key = tuple((n, _earth_key_part(v)) for n, v in a.items() if n not in _EARTH_NONRESULT_ARGS)
    F = np.ascontiguousarray(a["F_mass_arrival"], dtype=np.float64)
    bands = a["earth_band_costh"] is not None
    entry = _EARTH_TRANSFER.get(key)
    if entry is None:
        if len(_EARTH_TRANSFER) >= _EARTH_TRANSFER_MAX:
            _EARTH_TRANSFER.pop(next(iter(_EARTH_TRANSFER)))
        entry = _EARTH_TRANSFER[key] = dict(cols={}, calls=0)

    def column(k):
        if len(entry["cols"]) < 3:                     # every chord crossed once for the three (2026-10-18)
            entry["cols"] = dict(enumerate(_earth_transfer_columns(a)))
        return entry["cols"][k]

    def out(whole, per_band):
        return (whole.copy(), per_band.copy()) if bands else whole.copy()

    k = _unit_column(F)
    if k is not None:
        return out(*column(k))
    entry["calls"] += 1
    if earth_transfer == "repeat" and len(entry["cols"]) < 3 and entry["calls"] == 1:
        r = _propagate_earth_spatial_direct(**a)
        return r
    cols = [column(j) for j in range(3)]
    whole = np.einsum("ik,kib->ib", F, np.stack([c[0] for c in cols]))
    if not bands:
        return whole
    return whole, np.einsum("ik,knib->nib", F, np.stack([c[1] for c in cols]))

