"""
dp_config.py

Canonical configuration: the single-source DEFAULT_CONFIG, the unknown-key-rejecting
merge, the JSON structure-matrix parser, and default-config generation. Split
verbatim out of dark_portal_solver.py.
"""
import os
import copy
import json
import multiprocessing

import numpy as np
from typing import Dict

from dp_core import safe_print

# =============================================================================
# MULTIPROCESSING WRAPPER & CONFIGURATION MANAGER
# =============================================================================

def parse_structure_matrix_3x3(entry, name: str) -> np.ndarray:
    """Parses a JSON-friendly 3x3 flavor structure matrix into a complex Hermitian ndarray.

    Each of the nine entries may be either a plain number (real) or a two-element list
    [re, im] (complex) -- JSON has no native complex type. The result must be Hermitian
    to machine precision relative to its own scale: a non-Hermitian potential would make
    the eigh-based propagation silently wrong (eigh only reads one triangle), so this
    raises loudly instead. Used for the config keys dark_portal.U_LRI_struct and
    standard_mixing.V_SM_struct.
    """
    arr = np.zeros((3, 3), dtype=np.complex128)
    try:
        if len(entry) != 3:
            raise ValueError
        for i in range(3):
            row = entry[i]
            if len(row) != 3:
                raise ValueError
            for j in range(3):
                v = row[j]
                if isinstance(v, (list, tuple)):
                    if len(v) != 2:
                        raise ValueError
                    arr[i, j] = complex(float(v[0]), float(v[1]))
                else:
                    arr[i, j] = complex(float(v), 0.0)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a 3x3 nested list; each entry a number or a two-element [re, im] list."
        )

    scale = np.max(np.abs(arr))
    if scale > 0 and np.max(np.abs(arr - np.conjugate(arr).T)) > 1e-12 * scale:
        raise ValueError(
            f"{name} must be Hermitian (M == conj(M).T): the eigh-based propagation reads only "
            f"one triangle of the Hamiltonian, so a non-Hermitian structure would be silently wrong."
        )
    return arr


# =============================================================================
# CANONICAL CONFIGURATION DEFAULTS
# =============================================================================
# Single source of truth for every configuration default. create_default_config, the
# CLI baseline in __main__, run_solver's normalization, and FrequentistAnalysis's
# physics_constants fallbacks all read from THIS dict -- defaults previously lived in
# four separately-maintained copies, and keeping them synchronized by hand produced
# exactly the class of silent drift this codebase's audits kept finding.
def _build_default_config() -> Dict:
    return {
        "cosmology": {
            "H0_km_s_Mpc": 67.4,
            "Omega_m": 0.315,
            "Omega_L": 0.685,
            # Relic-neutrino temperature today: (4/11)^(1/3) * T_CMB = 1.9454 K for
            # T_CMB = 2.7255 K, i.e. 1.6764e-4 eV (k_B = 8.617333e-5 eV/K). The value was
            # 1.95e-4 eV until 2026-10-12 -- "1.95 K" entered as eV, 16% too hot; since the
            # thermal kernel's relic density follows from the Fermi-Dirac integral (~T^3),
            # the absorption rates were ~57% too high.
            "T_nu0_eV": 1.676389e-4,
            # Cosmological NEUTRON density today. Neutrons are bound almost entirely in He-4,
            # Y_p/2 per baryon: n_b = 2.515e-7 cm^-3 (PDG 2024, CMB) x 0.2458/2 (Y_p, LBT Y_p
            # project 2026) = 3.091e-8 cm^-3. The value was 0.25e-6 until 2026-10-12 -- the
            # total BARYON density, which made the cosmological dark-portal potential 8.1x too
            # strong (it is sourced by neutrons only: Z-like couplings of protons and electrons
            # cancel in neutral matter).
            "n_n0_cm3": 3.091e-8,
            "m_lightest_eV": 0.01,
            "mass_ordering": "normal",
            "eta_e": 1.0,
            "eta_mu": 1.0,
            "eta_tau": 1.0
        },
        "standard_mixing": {
            "dm21_sq": 7.41e-5,
            "dm31_sq": 2.511e-3,
            "th12_deg": 33.41,
            "th13_deg": 8.58,
            "th23_deg": 49.1,
            "d_cp_deg": 132.0,
            "use_SM_potential": False,
            "n_e_earth_cm3": 1.5e24,
            "detector_depth_km": 1.5,
            "V_SM_struct": None
        },
        "dark_portal": {
            "u_e": 0.5,
            "u_mu": 0.5,
            "u_tau": 0.0,
            "m_Zd_eV": 2.0e6,
            "g_D": 1.e-4,
            "epsilon_prime": 1e-1,
            "use_debye_screening": True,
            "U_LRI_struct": None
        },
        "source": {
            "p_index": 2.5,
            "Phi_0": 1.0,
            "z_max": 4.0,
            "rho_evol": 0.0,
            "f_e_S": 0.333333,
            "f_mu_S": 0.666667,
            "f_tau_S": 0.0,
            "is_antineutrino": "sum",
            "nubar_to_nu_ratio": 1.0
        },
        "grid": {
            "E_min_eV": 1e13,
            "E_max_eV": 3e16,
            "n_E_bins": 1000,
            "n_z_bins": 1000,
            "n_mw_paths": 6,
            "n_earth_chords": 12,
            "N_steps_MW": 25,
            "n_z_thermal": 0,
            "thermal_analytic": True,
            "earth_step_scheme": "avg",
            "N_steps_earth": 25,
            "r_start_kpc": 100.0
        },
        "output": {
            "flux_filename": "../output/data/output_flux.txt",
            "plot_filename": "../output/plots/flux_plot.png"
        },
        "execution": {
            "n_cores": multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1,
            "verbose": 1,
            "use_gpu": False
        }
    }

DEFAULT_CONFIG = _build_default_config()

def merge_config_with_defaults(config: Dict) -> Dict:
    """Overlays a (possibly partial) user config onto the canonical defaults, REJECTING
    unknown sections and keys.

    The rejection matters as much as the merging: a typo like 'n_eart_chords' previously
    fell through every .get() fallback and silently did nothing -- the same
    parsed-but-dropped failure mode behind several bugs fixed in past audits
    (n_e_earth_cm3, use_SM_potential, execution.n_cores). Keys and sections whose names
    begin with '_comment' are ignored, matching the FrequentistAnalysis convention.
    """
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for section, entries in config.items():
        if section.startswith("_comment"):
            continue
        if section not in merged:
            raise ValueError(f"Unknown config section '{section}'. Valid sections: {sorted(merged)}")
        if not isinstance(entries, dict):
            raise ValueError(f"Config section '{section}' must be a dict of settings.")
        for key, value in entries.items():
            if key.startswith("_comment"):
                continue
            if key not in merged[section]:
                raise ValueError(
                    f"Unknown config key '{section}.{key}'. Valid keys for [{section}]: "
                    f"{sorted(merged[section])}")
            merged[section][key] = value
    return merged

# =============================================================================
# ENTRY POINT & CONFIGURATION GENERATION
# =============================================================================

def create_default_config(filename: str = "../config/config_dark_portal.json") -> Dict:
    """Writes (and returns) the canonical default configuration -- see DEFAULT_CONFIG."""
    default_config = copy.deepcopy(DEFAULT_CONFIG)
    # The optional structure-matrix overrides default to None; omit them from the
    # generated JSON so the file shows only actively-used settings.
    default_config["standard_mixing"].pop("V_SM_struct", None)
    default_config["dark_portal"].pop("U_LRI_struct", None)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(default_config, f, indent=4)
    safe_print(f"Created default configuration file: {filename}")
    return default_config

