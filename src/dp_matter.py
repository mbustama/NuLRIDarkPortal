"""
dp_matter.py

The matter charge of the potential (docs/dev/SCOPE_WIDENING_PLAN.md, step 1). The force is sourced by
q_n n_n + q_p n_p + q_e n_e; in neutral matter n_p = n_e, so only q_n and q_pe = q_p + q_e enter. Every density
the solver uses -- the cosmic mean, the Galaxy's Yukawa integrals (at the Sun and at the Galactic emission
points), the Sun, the Moon, the Earth's profile -- is linear in its source, so it is computed for neutrons and
for electrons (the same models with each body's composition) and combined: n_Q = q_n n_n + q_pe n_e.

The solver carries the potential's sign in its coefficient (Coef, its absolute value and its sign travel
separately through the transit), so it needs a source of one sign: normalize() folds a negative charge
(B - 3L_e: n - 2e < 0 in every body) into MatterCharge.sign, which multiplies the coefficient, and refuses a
charge whose density changes sign between bodies. The neutron charge (1, 0, 0) -- also B - L in neutral
matter -- normalizes to None: the solver's historical path, bit for bit.
"""
from typing import NamedTuple, Optional

import numpy as np

import dp_core


class MatterCharge(NamedTuple):
    """A normalized matter charge: q_n per neutron and q_pe per proton-electron pair, signed so that the source
    density is non-negative everywhere; sign (+1 or -1), which multiplies the potential's coefficient; and the
    cosmic electron density today (eV^3), the electron species' floor in the Galaxy's Yukawa integrals."""
    q_n: float
    q_pe: float
    sign: float
    n_e0_eV3: float


def _compositions(n_n0_eV3: float, n_e0_eV3: float) -> dict:
    """(neutrons, electrons) in a unit of each body's matter -- per nucleon (stars, cold gas), per electron (hot
    gas), per gram (the Earth's regions), in total (the point sources), today's cosmic densities -- for the sign
    check."""
    import dp_spatial
    out = {
        "cosmic mean": (n_n0_eV3, n_e0_eV3),
        "Galactic stars": (dp_spatial._MW_NEUTRONS_PER_NUCLEON_STARS, dp_spatial._MW_ELECTRONS_PER_NUCLEON_STARS),
        "Galactic cold gas": (dp_spatial._MW_NEUTRONS_PER_NUCLEON_COLD_GAS,
                              dp_spatial._MW_ELECTRONS_PER_NUCLEON_COLD_GAS),
        "Galactic hot gas": (dp_spatial._MW_NEUTRONS_PER_ELECTRON_GAS, 1.0),
        "Sun": (dp_core.N_n_sun, dp_core.N_e_sun),
        "Moon": (dp_core.N_n_moon, dp_core.N_e_moon),
    }
    for name, r in zip(("Earth's core", "Earth's mantle and crust", "Earth's top layer"),
                       dp_spatial._EARTH_REGION_SAMPLE_KM):
        out[name] = (dp_spatial.earth_neutrons_per_gram(r), dp_spatial.earth_electrons_per_gram(r))
    return out


def normalize(q_n: float, q_p: float, q_e: float, n_n0_eV3: float, n_e0_eV3: float) -> Optional[MatterCharge]:
    """The MatterCharge of q_n n_n + q_p n_p + q_e n_e, or None for the neutron charge (q_n = 1, q_p + q_e = 0).

    Arguments: the charges per neutron, proton and electron; the cosmic neutron and electron densities today
    (eV^3). Raises ValueError if the charge density is positive in some bodies and negative in others."""
    q_n, q_pe = float(q_n), float(q_p) + float(q_e)
    if not (np.isfinite(q_n) and np.isfinite(q_pe)):
        raise ValueError(f"matter charges must be finite, got q_n = {q_n}, q_p + q_e = {q_pe}")
    if q_n == 1.0 and q_pe == 0.0:
        return None
    vals = {k: q_n * a + q_pe * b for k, (a, b) in _compositions(n_n0_eV3, n_e0_eV3).items()}
    pos = sorted(k for k, v in vals.items() if v > 0.0)
    neg = sorted(k for k, v in vals.items() if v < 0.0)
    if pos and neg:
        raise ValueError(f"The matter charge (q_n = {q_n:g}, q_p + q_e = {q_pe:g}) is positive in {pos} and "
                         f"negative in {neg}: the solver needs a source density of one sign.")
    sign = -1.0 if neg else 1.0
    return MatterCharge(sign * q_n, sign * q_pe, sign, float(n_e0_eV3))


def from_config(config: dict, n_n0_eV3: float) -> Optional[MatterCharge]:
    """normalize() from a merged solver configuration's matter_charge section and cosmology.n_e0_cm3."""
    mc = config["matter_charge"]
    n_e0_eV3 = float(config["cosmology"]["n_e0_cm3"]) * dp_core.cm_inv3_to_eV3
    return normalize(mc["q_n"], mc["q_p"], mc["q_e"], n_n0_eV3, n_e0_eV3)


def cosmic_density(charge: Optional[MatterCharge], n_n0_eV3: float) -> float:
    """The cosmic mean charge density today (eV^3); n_n0_eV3 itself for the neutron charge (None)."""
    if charge is None:
        return n_n0_eV3
    return charge.q_n * n_n0_eV3 + charge.q_pe * charge.n_e0_eV3


def point_sources(charge: Optional[MatterCharge]):
    """(Sun, Moon, Earth): the charge each body holds (its neutrons for None)."""
    pairs = ((dp_core.N_n_sun, dp_core.N_e_sun), (dp_core.N_n_moon, dp_core.N_e_moon),
             (dp_core.N_n_earth, dp_core.N_e_earth))
    if charge is None:
        return tuple(a for a, _ in pairs)
    return tuple(charge.q_n * a + charge.q_pe * b for a, b in pairs)


def earth_cpg(charge: MatterCharge) -> np.ndarray:
    """The Earth's charge per gram, (core, mantle and crust, top layer) (dp_spatial.earth_charge_per_gram)."""
    import dp_spatial
    return dp_spatial.earth_charge_per_gram(charge.q_n, charge.q_pe)


def galactic_effective_density(charge: MatterCharge, m_eV: float, n_n0_eV3: float, R_H_eV_inv: float,
                               n_e0_eV3: Optional[float] = None) -> float:
    """The Galaxy's Yukawa-smoothed charge density at the Sun, with the cosmic mean beyond it out to the Hubble
    radius (eV^3): dp_transit.galactic_effective_density per species, each with its own cosmic floor -- n_n0_eV3
    for the neutrons, n_e0_eV3 (default: the charge's) for the electrons; pass zeros for a host galaxy alone."""
    import dp_transit
    n_e0 = charge.n_e0_eV3 if n_e0_eV3 is None else n_e0_eV3
    out = 0.0
    if charge.q_n != 0.0:
        out += charge.q_n * dp_transit.galactic_effective_density(m_eV, n_n0_eV3, R_H_eV_inv, species=0)
    if charge.q_pe != 0.0:
        out += charge.q_pe * dp_transit.galactic_effective_density(m_eV, n_e0, R_H_eV_inv, species=1)
    return out


def galactic_emission_density(charge: MatterCharge, R, z, m_eV: float, n_n0_eV3: float, R_H_eV_inv: float):
    """The Yukawa-smoothed charge density at Galactic emission points (R, |z|) kpc (eV^3):
    dp_galactic.effective_density per species, combined."""
    import dp_galactic
    out = np.zeros(np.shape(R))
    if charge.q_n != 0.0:
        out = out + charge.q_n * dp_galactic.effective_density(R, z, m_eV, n_n0_eV3, R_H_eV_inv, species=0)
    if charge.q_pe != 0.0:
        out = out + charge.q_pe * dp_galactic.effective_density(R, z, m_eV, charge.n_e0_eV3, R_H_eV_inv,
                                                                species=1)
    return out
