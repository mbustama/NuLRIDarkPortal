"""
dark_portal_solver.py -- public facade of the dark-portal solver.

The implementation was split (2026-07, pure verbatim move) into:
    dp_core.py        environment guards, constants, shared physics utilities
    dp_propagator.py  the exactly unitary 3x3 propagator (deflated eigenbasis)
    dp_thermal.py     thermal attenuation rates + two-component coupling cache
    dp_cosmo.py       cosmological-phase kernels (z-stepper, sigma table)
    dp_spatial.py     Milky Way / Earth transit, workers, persistent pool
    dp_config.py      DEFAULT_CONFIG, config merging/validation

This module re-exports the complete historical surface (every name external code
imported or accessed as dps.<name> keeps working), and itself retains only the
cross-phase orchestration: propagate_spectrum, propagation_worker, run_solver,
file/plot output, and the CLI.

Backward-compatibility note: HAS_TQDM is mutable state owned by dp_core; reading
dark_portal_solver.HAS_TQDM is delegated there, but WRITING it must now target
dp_core (`import dp_core; dp_core.HAS_TQDM = False`), as the in-repo scanner and
convergence tool do.
"""
import dp_core  # noqa: F401  (environment guards run on this first import)

import os
import copy
import json
import argparse
import multiprocessing
import numpy as np
from typing import Dict, Tuple

from dp_core import (  # noqa: F401  (re-exports + orchestration dependencies)
    eV, keV, MeV, GeV, TeV, PeV,
    m_to_eV_inv, cm_inv3_to_eV3, Mpc_to_eV_inv, G_F, km_per_s,
    d_sun_eV_inv, d_moon_eV_inv, R_earth_eV_inv, R_core_eV_inv,
    N_n_sun, N_n_earth, N_n_moon,
    HAS_NUMBA, HAS_MATPLOTLIB, _TBB_LAYER, numba, njit, prange, tqdm, safe_print,
    construct_PMNS, calc_debye_mass_sq, get_mass_eigenvalues, compute_dE_geometric,
)
if HAS_MATPLOTLIB:
    from dp_core import plt  # noqa: F401
from dp_propagator import propagator_3x3, eigenvectors_3x3  # noqa: F401
from dp_thermal import (  # noqa: F401
    thermal_averaged_rates_flavor, _seg_moment_atan, _seg_moment_log,
    thermal_averaged_rates_flavor_analytic, precompute_all_thermal_rates_analytic,
    precompute_all_thermal_rates, _interp_thermal_grid_z, build_thermal_grid, absorption_normalization,
    thermal_grid_cached, _tc_cache, _tc_try_fit, compute_gamma_grid_inputs,
    _GL16_X, _GL16_W, _GL10_X, _GL10_W, _FD16,
)
from dp_cosmo import (  # noqa: F401
    calc_H_Hubble, calc_R_source, calc_V_LRI_cosmo, get_adiabatic_mixing_cosmo,
    _build_sigma_projection_table, step_cosmological_pair, cascade_kernel_weights, cross_kernel_weights,
)
from dp_spatial import (  # noqa: F401
    calc_MW_neutron_density, calc_earth_density, get_baseline_earth,
    spatial_total_density_mw, spatial_total_density_earth, init_mw_density_matrix,
    _evolve_rho_all_bins, step_mw_approach, step_earth_transit,
    _mw_path_worker, _earth_chord_worker, _PREM_BOUNDARIES_KM,
    _earth_layer_crossings, _avg_earth_density_seq, _split_E_blocks,
    _PREM_EDGES_KM, _PREM_RHO_COEFFS, _PREM_F_COEFFS, yukawa_profile_table,
    earth_yukawa_table, yukawa_profile_density, earth_neutron_density_cm3,
    earth_total_density, _earth_chord_features, _yk_extra_cm3, average_earth_chords,
    _run_spatial_tasks, _get_spatial_pool, _close_spatial_pool,
    propagate_earth_spatial,
)
from dp_config import (  # noqa: F401
    parse_structure_matrix_3x3, _build_default_config, DEFAULT_CONFIG,
    merge_config_with_defaults, create_default_config,
)


def __getattr__(name):
    # HAS_TQDM lives in dp_core and is mutated at runtime (scanner/convergence tool
    # silence the progress bars); delegate reads so stale readers stay correct.
    if name == "HAS_TQDM":
        return dp_core.HAS_TQDM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def propagate_spectrum(E_bins: np.ndarray, z_bins: np.ndarray, rho_src: np.ndarray,
                       H0: float, Omega_m: float, Omega_L: float,
                       U_pmns: np.ndarray, M_sq: np.ndarray,
                       u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
                       m_nu_bg_arr: np.ndarray, n_n0: float, T_nu0: float,
                       use_debye_screening: bool, detector_depth_km: float,
                       use_SM_potential: bool, n_e_earth: float,
                       injection, p_index: float, rho_evol: float,
                       Phi_0: float, eta_bg: np.ndarray, n_mw_paths: int, n_earth_chords: int,
                       N_steps_MW: int = 25, r_start_kpc: float = 100.0,
                       show_progress: bool = True, verbose: int = 1, n_cores: int = 1,
                       precomputed_Gamma_grid_all: np.ndarray = None, N_steps_earth: int = 500,
                       external_pool=None, U_LRI_struct=None, V_SM_struct=None,
                       n_z_thermal: int = 0, use_gpu: bool = False,
                       thermal_analytic: bool = True,
                       earth_step_scheme: str = "left",
                       earth_zenith_average: str = "plain") -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes the complete spectrum integration, driving the underlying JIT kernels.
    Neutrinos and antineutrinos are evolved together through the cosmological phase (an
    absorption of either emits one secondary of each kind) and then separately through the
    Milky Way and the Earth.
    Includes verbose level 2 for periodic redshift status updates.

    Arguments:
      E_bins: energy grid (eV), geometric spacing assumed by the cascading integral.
      z_bins: redshift grid, DESCENDING from z_max to 0 (the integration direction).
      rho_src: (3,) source flavor composition (f_e, f_mu, f_tau).
      H0, Omega_m, Omega_L: flat-LCDM cosmology (H0 already converted to eV).
      U_pmns: (3, 3) PMNS matrix, neutrino convention (conjugated here for antineutrinos).
      M_sq: (3, 3) complex diagonal mass-squared matrix diag(0, dm21, dm31_signed).
      u_arr: (3,) complex sterile mixing amplitudes (u_e, u_mu, u_tau), neutrino convention.
      m_Zd, g_D, eps_prime: dark mediator mass (eV) and couplings.
      m_nu_bg_arr: (3,) neutrino background mass eigenvalues (eV).
      n_n0: cosmological neutron density today (eV^3); T_nu0: CnuB temperature (eV).
      use_debye_screening: include the thermal Debye mass in the mediator propagator.
      detector_depth_km: detector depth below the Earth surface.
      use_SM_potential / n_e_earth: standard MSW matter potential inside the Earth
        (electron density in eV^3).
      injection: (w_nu, w_nb), the weights of the neutrino and antineutrino injection
        (e.g. (1, 0) for neutrino sources only, (1/2, 1/2) for nubar/nu = 1). Until
        2026-10-13 this was a single-species flag and each call propagated one kind alone.
      p_index, rho_evol, Phi_0: source spectrum E^-p, redshift evolution, normalization.
      eta_bg: (3,) CnuB flavor asymmetries.
      n_mw_paths, n_earth_chords: spatial-averaging counts (solid angle / zenith).
      N_steps_MW, r_start_kpc: Milky Way transit resolution and entry radius.
      show_progress, verbose: progress bars / logging (verbose=2 adds z updates).
      n_cores: intra-run parallelism of the spatial handover AND the numba thread cap.
      precomputed_Gamma_grid_all, N_steps_earth, external_pool, U_LRI_struct,
        V_SM_struct, earth_step_scheme: see the dedicated paragraphs below.
      n_z_thermal: coarse z-sampling of the thermal grid (0 = exact; see
        build_thermal_grid).
      use_gpu: route the spatial handover through gpu_spatial when cupy+CUDA exist.
      thermal_analytic: semi-analytic thermal momentum kernel (True, default) vs the
        midpoint reference kernel.

    Returns: (F_nu, F_nb), the (N_E, 3) flavor fluxes of neutrinos and of antineutrinos
    at the detector; a detector without charge separation sees their sum.

    U_LRI_struct / V_SM_struct, if given, are dimensionless 3x3 Hermitian flavor structure
    matrices overriding the default LRI texture (u_i * conj(u_j)) and SM matter-potential
    texture (diag(1, 0, 0)). Pass them in NEUTRINO convention: the antineutrino
    conjugation (V -> -conj(V)) is applied centrally here, as for U_pmns and u_arr. The
    overrides affect only the coherent forward-scattering potentials -- the Debye
    screening mass and the thermal attenuation/regeneration rates remain tied to the
    u_e/u_mu/u_tau amplitudes (config keys: dark_portal.U_LRI_struct,
    standard_mixing.V_SM_struct).

    external_pool, if given, is a caller-managed multiprocessing.Pool passed straight through
    to propagate_earth_spatial, reused across many calls instead of creating a fresh one each
    time -- see _run_spatial_tasks's docstring for why this matters under 'spawn'.

    precomputed_Gamma_grid_all lets a caller reuse a thermal-rate grid across calls (it is
    the same for both kinds, and independent of the source composition; see
    compute_gamma_grid_inputs).

    N_steps_earth controls the number of path-length steps per Earth chord (analogous to
    N_steps_MW for the Milky Way phase), and earth_step_scheme selects the potential
    discretization ('avg' = exact step-averaged potential, 'left' = historical
    left-endpoint sampling). Under 'avg' -- the CONFIG default -- the accumulated LRI
    phase is step-count-independent and results carry a convergence certificate
    (N = 8..1000 agree to <= 5e-9 at the standard validation points; DEFAULT_CONFIG
    therefore ships N_steps_earth = 25, a ~3x margin over the flat-at-8 knee,
    revalidated against N = 100 at the four standard points on adoption). Under
    'left', results scatter at the ~1e-2 level with ANY step-count change (fringe
    re-aliasing) and 500 was the pinned historical value. NOTE the function-signature
    defaults here (500, 'left') deliberately preserve the historical behavior for
    direct callers that omit these kwargs; run_solver and the analysis pipeline
    always pass the config values.
    """
    if HAS_NUMBA:
        # The cosmological step below is parallelized internally via numba prange; cap its
        # thread pool to n_cores so a caller that already parallelizes across many processes
        # at a higher level (dark_portal_scanner.py, the analysis package's scan workers --
        # both of which pass n_cores=1 here) doesn't also oversubscribe every core
        # from inside each of its own worker processes.
        numba.set_num_threads(max(1, int(n_cores)))

    if isinstance(injection, (bool, np.bool_)) or np.ndim(injection) != 1 or len(injection) != 2:
        raise TypeError("propagate_spectrum: injection must be the pair (w_nu, w_nb); the single-species "
                        "is_antineutrino flag was replaced on 2026-10-13")
    w_nu, w_nb = float(injection[0]), float(injection[1])
    N_E = len(E_bins)
    dz = z_bins[0] - z_bins[1]

    sum_G, sum_G_alpha, eta_mass = compute_gamma_grid_inputs(U_pmns, u_arr, g_D, eps_prime, eta_bg)
    BR = np.zeros(3, dtype=np.float64)
    if sum_G > 0:
        BR = sum_G_alpha / sum_G

    # Pre-compute the thermal interaction rates for the entire grid before sequential stepping
    # (the same for both kinds); reuse the caller's grid when given one.
    if precomputed_Gamma_grid_all is not None:
        Gamma_grid_all = precomputed_Gamma_grid_all
    else:
        if verbose == 2:
            safe_print(f"Pre-computing thermal interaction rate grids...")
        Gamma_grid_all = build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, m_Zd, m_nu_bg_arr, T_nu0, eta_mass,
                                            n_z_thermal=n_z_thermal, thermal_analytic=thermal_analytic)

    # DOMAIN CHECK: a non-finite rate grid would propagate NaN silently through the whole
    # z-loop and out of the final flux (historically: an all-NaN spectrum with no hint of
    # which parameter was responsible). Fail loudly, naming the offending values instead.
    if not np.isfinite(Gamma_grid_all).all():
        raise ValueError(
            f"Non-finite thermal-rate grid (Gamma) at m_Zd_eV={m_Zd:.6g}, g_D={g_D:.6g}, "
            f"epsilon_prime={eps_prime:.6g}: these dark-portal parameter values are outside "
            f"the numerically supported domain of thermal_averaged_rates_flavor.")

    # The LRI coefficient along the redshift grid (neutrino sign): c_z = 2 (1+z) A_LRI(z), the
    # sigma-table coordinate per unit energy (see _build_sigma_projection_table).
    c_z_arr = np.empty(len(z_bins), dtype=np.float64)
    c_W_ = 0.88
    m_Z_ = 91.18 * GeV
    g_weak_ = 0.65
    for k in range(len(z_bins)):
        zk = z_bins[k]
        m_D_sq_k = calc_debye_mass_sq(g_D, u_arr, T_nu0 * (1.0 + zk), eta_bg) if use_debye_screening else 0.0
        denom_k = max(m_Zd**2 + m_D_sq_k, 1e-25)
        A_k = (g_D * eps_prime) / denom_k * (4.0 * c_W_ * m_Z_**2 / g_weak_) * (np.sqrt(2.0) / 2.0 * G_F * n_n0 * (1.0 + zk)**3)
        c_z_arr[k] = 2.0 * (1.0 + zk) * A_k

    # Per kind: the vacuum matrix U M U^dag (computed once), the LRI and SM textures with the
    # antineutrino transform (U -> conj U, u -> conj u, V -> -conj V) applied here, centrally,
    # and the sigma-table of the cosmological mixing.
    species = []
    for anti in (False, True):
        U_s = np.ascontiguousarray(np.conjugate(U_pmns)) if anti else np.ascontiguousarray(U_pmns)
        u_s = np.ascontiguousarray(np.conjugate(u_arr)) if anti else np.ascontiguousarray(u_arr)
        M_rot = np.ascontiguousarray(U_s @ M_sq @ np.ascontiguousarray(np.conjugate(U_s).T))
        if U_LRI_struct is None:
            S_eff = np.outer(u_s, np.conjugate(u_s))
        else:
            S_eff = np.conjugate(U_LRI_struct) if anti else U_LRI_struct
        S_eff = np.ascontiguousarray(S_eff, dtype=np.complex128)
        V_SM_eff = None
        if V_SM_struct is not None:
            V_SM_eff = np.ascontiguousarray(np.conjugate(V_SM_struct) if anti else V_SM_struct, dtype=np.complex128)
        c_z = -c_z_arr if anti else c_z_arr
        # H(sigma) = M_rot + sigma * S_signed, tabulated over |sigma| with the sign folded into S
        S_signed = np.ascontiguousarray(-S_eff if anti else S_eff)
        abs_c = np.abs(c_z)
        sig_min = max(E_bins[0] * abs_c.min(), 1e-30 * max(np.max(np.abs(M_rot)), 1e-300))
        sig_max = max(E_bins[-1] * abs_c.max(), sig_min * 2.0)
        N_sigma = 4096
        sigma_nodes = np.geomspace(sig_min, sig_max, N_sigma)
        P_table = _build_sigma_projection_table(M_rot, S_signed, sigma_nodes)
        species.append(dict(anti=anti, u=u_s, M_rot=M_rot, S_eff=S_eff, V_SM_eff=V_SM_eff, c_z=c_z,
                            P_table=P_table, log_s0=np.log(sigma_nodes[0]),
                            inv_dlogs=(N_sigma - 1) / np.log(sig_max / sig_min),
                            F=np.zeros((N_E, 3), dtype=np.float64), P_all=np.zeros((N_E, 3, 3), dtype=np.float64),
                            R=np.zeros(N_E, dtype=np.float64)))
    nu, nb = species

    # The cascade kernels' exact bin integrals depend only on the (fixed) energy grid
    W3, pref3, self_same = cascade_kernel_weights(E_bins)
    W1, W2, c1, c2, self_cross = cross_kernel_weights(E_bins)

    iterator = tqdm(range(1, len(z_bins)), desc="Cosmo Phase", leave=False) if (dp_core.HAS_TQDM and show_progress) else range(1, len(z_bins))
    z_print_mod = max(1, len(z_bins) // 10)
    if verbose == 2:
        safe_print(f"\n--- Cosmological Integration Started (z_max = {z_bins[0]:.2f}) ---")
    for k in iterator:
        z = z_bins[k]
        dt = dz / (calc_H_Hubble(z, H0, Omega_m, Omega_L) * (1.0 + z))
        R_z = calc_R_source(z, rho_evol)
        if verbose == 2 and k % z_print_mod == 0:
            safe_print(f"Cosmological propagation: currently integrating at redshift z = {z:.4f}")
        step_cosmological_pair(z, dt, R_z, E_bins, nu["F"], nb["F"], w_nu, w_nb, rho_src,
                               nu["P_table"], nu["log_s0"], nu["inv_dlogs"], nu["c_z"][k],
                               nb["P_table"], nb["log_s0"], nb["inv_dlogs"], nb["c_z"][k],
                               p_index, Phi_0, BR, W3, pref3, self_same, W1, W2, c1, c2, self_cross,
                               nu["P_all"], nb["P_all"], np.ascontiguousarray(Gamma_grid_all[k, :, :]),
                               nu["R"], nb["R"])
    if verbose == 2:
        safe_print(f"Cosmological baseline finished at z = {z_bins[-1]:.4f}.")

    out = []
    for sp in species:
        if not np.any(sp["F"]):
            out.append(np.zeros((N_E, 3), dtype=np.float64))   # nothing of this kind arrives
            continue
        F_earth = propagate_earth_spatial(E_bins, sp["F"], sp["M_rot"], sp["u"], m_Zd, g_D, eps_prime,
                                          T_nu0, use_debye_screening, eta_bg, detector_depth_km,
                                          use_SM_potential, n_e_earth, sp["anti"], n_mw_paths, n_earth_chords,
                                          N_steps_MW=N_steps_MW, r_start_kpc=r_start_kpc,
                                          show_progress=show_progress, verbose=verbose, n_cores=n_cores,
                                          N_steps_earth=N_steps_earth, external_pool=external_pool,
                                          U_LRI_struct=sp["S_eff"], V_SM_struct=sp["V_SM_eff"],
                                          use_gpu=use_gpu, earth_step_scheme=earth_step_scheme,
                                          earth_zenith_average=earth_zenith_average)
        # DOMAIN CHECK: same contract as the thermal-grid check above -- never return NaN
        # silently. (The known non-finite mechanism here, unitarity loss of the spatial
        # propagator at large phases, is excluded by propagator_3x3's exactly unitary form
        # since 2026-10-12; this is the safety net for anything not yet diagnosed.)
        if not np.isfinite(F_earth).all():
            raise ValueError(
                f"Non-finite flux out of the spatial handover at m_Zd_eV={m_Zd:.6g}, "
                f"g_D={g_D:.6g}, epsilon_prime={eps_prime:.6g} "
                f"({'antineutrino' if sp['anti'] else 'neutrino'} pass): these dark-portal "
                f"parameter values are outside the numerically supported domain.")
        out.append(F_earth)
    return out[0], out[1]


# =============================================================================
# FILE EXPORT ROUTINE
# =============================================================================

def write_run_manifest(filename: str, config: Dict, runtime_s: float = None):
    """Writes a reproducibility manifest (JSON) alongside an output file: the full
    resolved configuration, the git commit of the code that produced it, package
    versions, host, timestamp, and runtime. Paper-level numbers should be traceable to
    an exact code + configuration state; this makes that automatic."""
    import platform
    import subprocess
    import datetime

    git_hash = None
    try:
        git_hash = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass

    versions = {"python": platform.python_version(), "numpy": np.__version__}
    if HAS_NUMBA:
        versions["numba"] = numba.__version__

    manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_hash,
        "hostname": platform.node(),
        "versions": versions,
        "runtime_s": runtime_s,
        "config": config,
    }
    with open(filename, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    safe_print(f"Run manifest saved to: {os.path.abspath(filename)}")

def save_flux_to_file(E_array: np.ndarray, Flux_array: np.ndarray, filename: str = "../output/data/output_flux.txt"):
    """Saves the generated flux components (Energy, nue, numu, nutau) to an external text file."""
    header = "Energy[eV]    Flux_nue[eV^-1 cm^-2 s^-1 sr^-1]    Flux_numu[eV^-1 cm^-2 s^-1 sr^-1]    Flux_nutau[eV^-1 cm^-2 s^-1 sr^-1]"
    data = np.column_stack((E_array, Flux_array))
    np.savetxt(filename, data, header=header, fmt="%.6e")
    safe_print(f"\nFlux data saved to: {os.path.abspath(filename)}")


def _flux_setup(E_bins: np.ndarray, params: Dict) -> Dict:
    """Per-run quantities shared by every flux of one parameter point: the redshift grid,
    the source vector, the mass and mixing matrices and the thermal-rate grid (identical
    for neutrinos and antineutrinos, and independent of the source composition).

    Arguments: E_bins -- energy grid (eV); params -- the dict built by build_solver_params.
    Returns: a dict of those quantities for _flux_from_setup.
    """
    z_bins = np.linspace(params['z_max'], 0.0, int(params['n_z_bins']))
    rho_src = np.array([params['f_e_S'], params['f_mu_S'], params['f_tau_S']], dtype=np.float64)
    
    ordering = str(params.get('mass_ordering', 'normal')).strip().lower()
    dm31_sq_signed = -params['dm31_sq'] if ordering == "inverted" else params['dm31_sq']
    
    # Enforce C-contiguous alignment for the initial matrix state
    M_sq = np.ascontiguousarray(np.diag([0.0, params['dm21_sq'], dm31_sq_signed]).astype(np.complex128))
    
    u_arr_base = np.ascontiguousarray(np.array([params['u_e'], params['u_mu'], params['u_tau']], dtype=np.complex128))
    eta_bg = params['eta_bg']
    U_pmns_base = np.ascontiguousarray(construct_PMNS(params['th12'], params['th13'], params['th23'], params['d_cp']))

    # Pin numba's thread pool BEFORE the thermal-grid precompute below: that prange kernel
    # previously ran ahead of propagate_spectrum's own set_num_threads call, so inside a
    # scanner pool worker (n_cores=1) the first evaluation used numba's default thread
    # count (all cores) -- transient oversubscription multiplied across every worker.
    if HAS_NUMBA:
        numba.set_num_threads(max(1, int(params.get('n_cores', 1))))

    # The thermal-rate grid depends only on |U_pmns| and |u_arr| magnitudes, which
    # conjugation never changes -- so it is identical for the neutrino and antineutrino
    # calls below. Computing it once here and sharing it (instead of letting each flavor
    # call redo this expensive O(N_z * N_E) precomputation from scratch) roughly halves
    # that stage's cost whenever both flavors are evaluated (is_antineutrino="sum", the
    # default). build_thermal_grid can additionally sample z on a coarse subgrid and
    # interpolate (grid.n_z_thermal; OFF by default -- see its docstring for the measured
    # error budget before enabling).
    sum_G, sum_G_alpha, eta_mass = compute_gamma_grid_inputs(U_pmns_base, u_arr_base, params['g_D'], params['epsilon_prime'], eta_bg)
    Gamma_grid_all = thermal_grid_cached(E_bins, z_bins, sum_G, sum_G_alpha, params['m_Zd_eV'], params['m_nu_bg_arr'], params['T_nu0_eV'], eta_mass,
                                         n_z_thermal=params.get('n_z_thermal', 0),
                                         thermal_analytic=params.get('thermal_analytic', True))

    return dict(z_bins=z_bins, rho_src=rho_src, M_sq=M_sq, u_arr_base=u_arr_base, eta_bg=eta_bg,
                U_pmns_base=U_pmns_base, Gamma_grid_all=Gamma_grid_all)


def _flux_from_setup(E_bins: np.ndarray, params: Dict, setup: Dict, injection,
                     rho_src: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """The neutrino and antineutrino fluxes at the detector, each (N_E, 3), from a _flux_setup.

    Arguments: E_bins, params, setup -- as for _flux_setup; injection -- (w_nu, w_nb), the
    weights of the neutrino and antineutrino sources; rho_src -- source flavor composition
    (default: the configured one).
    """
    if rho_src is None:
        rho_src = setup['rho_src']
    return propagate_spectrum(
        E_bins, setup['z_bins'], rho_src,
        params['H0_eV'], params['Omega_m'], params['Omega_L'],
        setup['U_pmns_base'], setup['M_sq'], setup['u_arr_base'],
        params['m_Zd_eV'], params['g_D'], params['epsilon_prime'],
        params['m_nu_bg_arr'], params['n_n0_eV3'], params['T_nu0_eV'],
        params['use_debye_screening'], params['detector_depth_km'],
        params['use_SM_potential'], params['n_e_earth_eV3'],
        injection, params['p_index'], params['rho_evol'], params['Phi_0'], setup['eta_bg'],
        params['n_mw_paths'], params['n_earth_chords'],
        params['N_steps_MW'], params['r_start_kpc'],
        show_progress=params.get('show_progress', True), verbose=params.get('verbose', 1),
        n_cores=params.get('n_cores', 1),
        precomputed_Gamma_grid_all=setup['Gamma_grid_all'], N_steps_earth=params.get('N_steps_earth', 500),
        U_LRI_struct=params.get('U_LRI_struct', None), V_SM_struct=params.get('V_SM_struct', None),
        use_gpu=params.get('use_gpu', False),
        thermal_analytic=params.get('thermal_analytic', True),
        earth_step_scheme=params.get('earth_step_scheme', 'left'),
        earth_zenith_average=params.get('earth_zenith_average', 'plain')
    )


def injection_weights(params: Dict) -> Tuple[float, float]:
    """(w_nu, w_nb), the weights of the neutrino and antineutrino sources: from
    source.is_antineutrino ('false': neutrinos only; 'true': antineutrinos only; 'sum':
    both, in the ratio source.nubar_to_nu_ratio)."""
    flag_val = str(params.get('is_antineutrino', 'sum')).strip().lower()
    if flag_val in ['sum', 'both']:
        ratio = float(params.get('nubar_to_nu_ratio', 1.0))
        if not np.isfinite(ratio):
            return 0.0, 1.0
        return 1.0 / (1.0 + ratio), ratio / (1.0 + ratio)
    if flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        return 0.0, 1.0
    return 1.0, 0.0


def propagate_weighted(E_bins: np.ndarray, params: Dict, sources=None) -> np.ndarray:
    """The flux at the detector (neutrinos plus antineutrinos, (N_E, 3)) for the configured
    sources (injection_weights) -- one coupled evolution. With sources, a sequence of (3,)
    source compositions sharing one thermal-rate grid, returns (n_sources, N_E, 3)."""
    setup = _flux_setup(E_bins, params)
    w = injection_weights(params)
    if sources is None:
        return sum(_flux_from_setup(E_bins, params, setup, w))
    return np.stack([sum(_flux_from_setup(E_bins, params, setup, w, np.ascontiguousarray(np.asarray(r, dtype=np.float64))))
                     for r in sources])


def propagate_nu_nubar(E_bins: np.ndarray, params: Dict, sources=None):
    """The flux at the detector (neutrinos plus antineutrinos) for neutrino sources alone and
    for antineutrino sources alone, per unit injection -- for callers that weight the two
    themselves, such as the analysis package with a free nubar/nu ratio (the fluxes are
    linear in the injection). Since 2026-10-13 each is a coupled evolution: neutrino sources
    also produce antineutrinos, through the secondaries of every absorption.

    Arguments:
      E_bins: energy grid (eV); params: the dict built by build_solver_params.
      sources: None for the configured source composition, or a sequence of (3,)
        compositions sharing one thermal-rate grid.
    Returns: (F_from_nu, F_from_nb), each (N_E, 3) -- or (n_sources, N_E, 3) when sources
    is given.
    """
    setup = _flux_setup(E_bins, params)
    srcs = [None] if sources is None else [np.ascontiguousarray(np.asarray(r, dtype=np.float64)) for r in sources]
    out_nu = [sum(_flux_from_setup(E_bins, params, setup, (1.0, 0.0), r)) for r in srcs]
    out_nb = [sum(_flux_from_setup(E_bins, params, setup, (0.0, 1.0), r)) for r in srcs]
    if sources is None:
        return out_nu[0], out_nb[0]
    return np.stack(out_nu), np.stack(out_nb)


def propagation_worker(args) -> Tuple[np.ndarray, np.ndarray]:
    """Unpacks arguments and executes the physics simulation: one coupled evolution of the
    configured neutrino and antineutrino sources, summed at the detector."""
    E_bins, params = args
    setup = _flux_setup(E_bins, params)
    F_nu, F_nb = _flux_from_setup(E_bins, params, setup, injection_weights(params))
    return E_bins, F_nu + F_nb


def build_solver_params(config: Dict) -> Tuple[np.ndarray, Dict]:
    """Validates a (possibly partial) configuration and converts it into the energy grid
    and the parameter dict the flux computation reads -- run_solver's parsing, without its
    console output, for callers that evaluate many parameter points (the analysis package
    overrides the varied parameters in a solver config and calls this per evaluation, so
    the solver's defaults and checks stay the single source of truth).

    Arguments: config -- as for run_solver.
    Returns: (E_bins, params).
    """
    config = merge_config_with_defaults(config)
    params = {}
    params['H0_eV'] = float((config['cosmology']['H0_km_s_Mpc'] * km_per_s) / Mpc_to_eV_inv)
    params['Omega_m'] = float(config['cosmology']['Omega_m'])
    params['Omega_L'] = float(config['cosmology']['Omega_L'])
    params['T_nu0_eV'] = float(config['cosmology']['T_nu0_eV'])
    params['n_n0_eV3'] = float(config['cosmology']['n_n0_cm3'] * cm_inv3_to_eV3)
    params['mass_ordering'] = config['cosmology']['mass_ordering']
    mo = str(config['cosmology']['mass_ordering']).strip().lower()
    if mo not in ['normal', 'inverted']:
        raise ValueError(f"mass_ordering must be strictly 'normal' or 'inverted'. Received: {mo}")
    
    params['eta_bg'] = np.array([
        float(config['cosmology']['eta_e']),
        float(config['cosmology']['eta_mu']),
        float(config['cosmology']['eta_tau'])
    ], dtype=np.float64)
    if config['cosmology']['eta_e'] < 0 or config['cosmology']['eta_mu'] < 0 or config['cosmology']['eta_tau'] < 0:
        raise ValueError("CnuB flavor asymmetries (eta) must be >= 0 to avoid complex Debye screening masses.")

    params['dm21_sq'] = float(config['standard_mixing']['dm21_sq'])
    params['dm31_sq'] = float(config['standard_mixing']['dm31_sq'])
    params['th12'] = float(np.radians(config['standard_mixing']['th12_deg']))
    params['th13'] = float(np.radians(config['standard_mixing']['th13_deg']))
    params['th23'] = float(np.radians(config['standard_mixing']['th23_deg']))
    params['d_cp'] = float(np.radians(config['standard_mixing']['d_cp_deg']))
    params['use_SM_potential'] = config['standard_mixing']['use_SM_potential']
    params['n_e_earth_eV3'] = float(config['standard_mixing']['n_e_earth_cm3'] * cm_inv3_to_eV3)
    # Optional flavor-structure overrides for the coherent forward-scattering potentials
    # (neutrino convention; antineutrino conjugation is applied centrally in
    # propagate_spectrum). See parse_structure_matrix_3x3 for the accepted JSON format.
    params['V_SM_struct'] = None
    if config['standard_mixing']['V_SM_struct'] is not None:
        params['V_SM_struct'] = parse_structure_matrix_3x3(
            config['standard_mixing']['V_SM_struct'], "standard_mixing.V_SM_struct")
    params['U_LRI_struct'] = None
    if config['dark_portal']['U_LRI_struct'] is not None:
        params['U_LRI_struct'] = parse_structure_matrix_3x3(
            config['dark_portal']['U_LRI_struct'], "dark_portal.U_LRI_struct")
    params['detector_depth_km'] = float(config['standard_mixing']['detector_depth_km'])
    depth_km = float(config['standard_mixing']['detector_depth_km'])
    if depth_km < 0.0 or depth_km > 6371.0:
        raise ValueError(f"detector_depth_km ({depth_km}) must be between 0 and the Earth's radius (6371.0 km).")

    params['m_nu_bg_arr'] = get_mass_eigenvalues(
        float(config['cosmology']['m_lightest_eV']),
        float(params['dm21_sq']),
        float(params['dm31_sq']),
        params['mass_ordering']
    )
    
    params['u_e'] = float(config['dark_portal']['u_e'])
    params['u_mu'] = float(config['dark_portal']['u_mu'])
    params['u_tau'] = float(config['dark_portal']['u_tau'])
    u_sq_sum = np.abs(params['u_e'])**2 + np.abs(params['u_mu'])**2 + np.abs(params['u_tau'])**2
    if u_sq_sum > 1.0:
        raise ValueError(f"Sterile mixing elements violate unitarity. Sum of squares is {u_sq_sum:.4f} (must be <= 1.0).")
    params['m_Zd_eV'] = float(config['dark_portal']['m_Zd_eV'])
    params['g_D'] = float(config['dark_portal']['g_D'])
    params['epsilon_prime'] = float(config['dark_portal']['epsilon_prime'])
    # Physical positivity bounds
    if params['m_Zd_eV'] <= 0:
        raise ValueError("Dark mediator mass m_Zd_eV must be strictly positive.")
    if params['g_D'] <= 0:
        raise ValueError("Dark mediator gauge coupling g_D must be strictly positive.")
    if params['g_D'] >= np.sqrt(4 * np.pi):
        raise ValueError(f"Dark gauge coupling g_D ({params['g_D']:.2f}) exceeds the perturbativity bound (sqrt(4*pi)).")
    if params['epsilon_prime'] < 0:
        raise ValueError("Kinetic mixing epsilon_prime must be non-negative (0 switches off the coupling "
                         "to matter: no coherent potential, absorption only).")
    params['use_debye_screening'] = config['dark_portal']['use_debye_screening']
    
    params['p_index'] = float(config['source']['p_index'])
    if config['source']['p_index'] <= 0:
        raise ValueError("p_index should be positive. The solver assumes a spectrum proportional to E^(-p_index).")
    params['Phi_0'] = float(config['source']['Phi_0'])
    if config['source']['Phi_0'] <= 0:
        raise ValueError("Flux normalization Phi_0 should be strictly positive.")
    params['z_max'] = float(config['source']['z_max'])
    if params['z_max'] <= 0.0:
        raise ValueError("Maximum redshift z_max must be strictly positive.")
    params['rho_evol'] = float(config['source']['rho_evol'])
    params['f_e_S'] = float(config['source']['f_e_S'])
    params['f_mu_S'] = float(config['source']['f_mu_S'])
    params['f_tau_S'] = float(config['source']['f_tau_S'])
    flavor_sum = params['f_e_S'] + params['f_mu_S'] + params['f_tau_S']
    if not np.isclose(flavor_sum, 1.0, atol=1e-5):
        raise ValueError(f"Source flavor ratios must sum to 1.0. Current sum: {flavor_sum}")
    if params['f_e_S'] < 0 or params['f_mu_S'] < 0 or params['f_tau_S'] < 0:
        raise ValueError("Source flavor fractions must be strictly >= 0. Negative flux injection is unphysical.")
    params['is_antineutrino'] = config['source']['is_antineutrino']
    params['nubar_to_nu_ratio'] = config['source']['nubar_to_nu_ratio']
    
    params['n_z_bins'] = int(config['grid']['n_z_bins'])
    params['n_mw_paths'] = int(config['grid']['n_mw_paths'])
    # Defaults set by the production convergence study (tools/convergence_study.py,
    # report in output/convergence/): n_earth_chords=3 (the old default) carried
    # 1.6-4.9e-2 absolute flavor-ratio error against a 100-chord reference at every
    # tested dark-portal point -- by far the largest numerical systematic of any default
    # -- while 12 chords bring it to ~1e-2 (percent level). N_steps_MW=25 already agrees
    # with a 400-step reference to ~1e-10 everywhere (the Milky Way transit is deeply
    # adiabatic), so the old default of 100 was pure surplus.
    params['n_earth_chords'] = int(config['grid']['n_earth_chords'])
    params['N_steps_MW'] = int(config['grid']['N_steps_MW'])
    # Coarse z-sampling of the thermal-rate grid (0 disables interpolation -- exact grid).
    params['n_z_thermal'] = int(config['grid']['n_z_thermal'])
    # Semi-analytic momentum integration of the thermal rates (True, the default) vs
    # the 100-point midpoint reference kernel (False) -- see build_thermal_grid.
    params['thermal_analytic'] = bool(config['grid']['thermal_analytic'])
    # Earth-transit potential discretization: 'left' (historical) or 'avg' (exact
    # step-averaged potential -- see _earth_chord_worker's docstring).
    params['earth_step_scheme'] = str(config['grid']['earth_step_scheme']).strip().lower()
    if params['earth_step_scheme'] not in ('left', 'avg'):
        raise ValueError(f"grid.earth_step_scheme must be 'left' or 'avg', got {params['earth_step_scheme']!r}")
    # Average over the Earth chords: 'phase' (exact integration of each block's LRI phase
    # over cos(zenith)) or 'plain' (historical equal-weight mean) -- see
    # dp_spatial.average_earth_chords. 'phase' needs the phase-exact 'avg' steps.
    params['earth_zenith_average'] = str(config['grid']['earth_zenith_average']).strip().lower()
    if params['earth_zenith_average'] not in ('phase', 'plain'):
        raise ValueError(f"grid.earth_zenith_average must be 'phase' or 'plain', got {params['earth_zenith_average']!r}")
    if params['earth_zenith_average'] == 'phase' and params['earth_step_scheme'] != 'avg':
        # 'left' exists to reproduce historical results, which used the plain mean (and its
        # Riemann phase errors defeat the 'phase' rule), so it implies 'plain'.
        params['earth_zenith_average'] = 'plain'
        if int(config['execution'].get('verbose', 1)) >= 1:
            safe_print("[NOTE] earth_step_scheme = 'left' uses the historical 'plain' zenith average.")
    params['N_steps_earth'] = int(config['grid']['N_steps_earth'])
    params['r_start_kpc'] = float(config['grid']['r_start_kpc'])
    paths_mw = config['grid']['n_mw_paths']
    chords_earth = config['grid']['n_earth_chords']
    if not isinstance(paths_mw, int) or paths_mw < 1:
        raise ValueError("n_mw_paths must be an integer >= 1.")
    if not isinstance(chords_earth, int) or chords_earth < 1:
        raise ValueError("n_earth_chords must be an integer >= 1.")
    
    if config['grid']['n_E_bins'] <= 1 or config['grid']['n_z_bins'] <= 1:
        raise ValueError("Grid resolution (n_E_bins, n_z_bins) must be greater than 1 to define integration steps.")

    # Energy grid sanity check
    if config['grid']['E_min_eV'] >= config['grid']['E_max_eV']:
        raise ValueError("E_max_eV must be strictly greater than E_min_eV for the cascading integral to function properly.")
    E_bins = np.logspace(np.log10(config['grid']['E_min_eV']), 
                         np.log10(config['grid']['E_max_eV']), 
                         int(config['grid']['n_E_bins']))
    
    # A single call to run_solver always evaluates exactly one physics configuration, so
    # there is nothing to parallelize at the config level (unlike dark_portal_scanner.py,
    # which parallelizes across many configs). Instead, n_cores drives genuine intra-run
    # parallelism across the independent Milky Way paths / Earth chords in the spatial
    # handover (see propagate_earth_spatial), which is where a single run actually has
    # independent work to distribute.
    n_cores = max(1, int(config['execution']['n_cores']))
    params['n_cores'] = n_cores
    # Optional CUDA spatial handover (requires cupy + a CUDA device; silently falls back
    # to the CPU path with a warning when unavailable -- see gpu_spatial.py).
    params['use_gpu'] = bool(config['execution']['use_gpu'])
    params['show_progress'] = True
    params['verbose'] = int(config['execution']['verbose'])
    return E_bins, params


def run_solver(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Orchestrates configuration and parallelization.

    Arguments:
      config: (possibly partial) configuration dict with the DEFAULT_CONFIG section
        layout -- cosmology, standard_mixing, dark_portal, source, grid, output,
        execution (see dp_config.DEFAULT_CONFIG for every key and its default,
        including grid.N_steps_earth = 25 under the certified 'avg' Earth step
        scheme). It is first overlaid onto DEFAULT_CONFIG with unknown-key
        rejection (see merge_config_with_defaults), so every read below can index
        directly -- no per-call fallback literals to drift out of sync.

    Returns:
      (E_bins, Flux_array): the energy grid (eV) and the (N_E, 3) flavor fluxes at
      the detector, weighted across neutrinos/antineutrinos per source.is_antineutrino.
    """
    E_bins, params = build_solver_params(config)
    n_cores = params['n_cores']
    safe_print(f"Initializing solver across {len(E_bins)} coupled energy bins...")
    if n_cores > 1:
        safe_print(f"Using up to {n_cores} core(s) to parallelize the Milky Way / Earth spatial handover.")
    else:
        safe_print("Deploying single-core worker.")

    E_out, Flux_out = propagation_worker((E_bins, params))

    return E_out, Flux_out


def plot_flavor_fluxes(E_array: np.ndarray, Flux_array: np.ndarray, p_index: float, is_antineutrino_flag: str, output_filename: str = "../output/plots/flux_plot.png"):
    """Plots and maps the total astrophysical neutrino spectrum ratios at Earth layers."""
    E_GeV = E_array / 1e9
    scaling_factor = E_GeV**2.0
    
    flux_e = Flux_array[:, 0]
    flux_mu = Flux_array[:, 1]
    flux_tau = Flux_array[:, 2]
    flux_total = np.sum(Flux_array, axis=1)
    
    s_flux_e = flux_e * scaling_factor
    s_flux_mu = flux_mu * scaling_factor
    s_flux_tau = flux_tau * scaling_factor
    s_flux_total = flux_total * scaling_factor
    
    safe_total = np.maximum(flux_total, 1e-100)
    ratio_e = flux_e / safe_total
    ratio_mu = flux_mu / safe_total
    ratio_tau = flux_tau / safe_total
    
    # Evaluate flag for dynamic LaTeX labels
    flag_val = str(is_antineutrino_flag).strip().lower()
    if flag_val in ['sum', 'both']:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\nu + \bar{\nu}$)', r'$\nu_e + \bar{\nu}_e$', r'$\nu_\mu + \bar{\nu}_\mu$', r'$\nu_\tau + \bar{\nu}_\tau$'
    elif flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\bar{\nu}$)', r'$\bar{\nu}_e$', r'$\bar{\nu}_\mu$', r'$\bar{\nu}_\tau$'
    else:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\nu$)', r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    ax1.plot(E_GeV, s_flux_total, color='black', lw=2.5, label=lbl_tot)
    ax1.plot(E_GeV, s_flux_e, color='crimson', linestyle='--', lw=2.0, label=lbl_e)
    ax1.plot(E_GeV, s_flux_mu, color='royalblue', linestyle='-.', lw=2.0, label=lbl_mu)
    ax1.plot(E_GeV, s_flux_tau, color='forestgreen', linestyle=':', lw=2.0, label=lbl_tau)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Scaled Flux $E_0^2 \Phi$ [Arb. Units]', fontsize=14)
    ax1.set_title('Astrophysical Neutrino Diffuse Flux at Earth', fontsize=15, pad=15)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(fontsize=12, loc='lower right')
    
    ax2.plot(E_GeV, ratio_e, color='crimson', linestyle='--', lw=2.0)
    ax2.plot(E_GeV, ratio_mu, color='royalblue', linestyle='-.', lw=2.0)
    ax2.plot(E_GeV, ratio_tau, color='forestgreen', linestyle=':', lw=2.0)
    
    ax2.set_ylabel('Flavor Ratio', fontsize=14)
    ax2.set_xlabel(r'Observed Energy $E_0$ [GeV]', fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    safe_print(f"\nPlot successfully saved to: {os.path.abspath(output_filename)}")
    plt.show()


if __name__ == "__main__":
    # Initialize argument flags for terminal overrides mapping to the JSON variables
    parser = argparse.ArgumentParser(description="Dark Portal Neutrino Propagation Solver command-line flag orchestrator.")
    parser.add_argument("--config", type=str, default="../config/config_dark_portal.json", help="Path to JSON parameter file.")
    
    # Cosmology Overrides
    parser.add_argument("--H0_km_s_Mpc", type=float, default=None)
    parser.add_argument("--Omega_m", type=float, default=None)
    parser.add_argument("--Omega_L", type=float, default=None)
    parser.add_argument("--T_nu0_eV", type=float, default=None)
    parser.add_argument("--n_n0_cm3", type=float, default=None)
    parser.add_argument("--m_lightest_eV", type=float, default=None)
    parser.add_argument("--mass_ordering", type=str, default=None)
    parser.add_argument("--eta_e", type=float, default=None)
    parser.add_argument("--eta_mu", type=float, default=None)
    parser.add_argument("--eta_tau", type=float, default=None)
    
    # Standard Oscillation Mixing Overrides
    parser.add_argument("--dm21_sq", type=float, default=None)
    parser.add_argument("--dm31_sq", type=float, default=None)
    parser.add_argument("--th12_deg", type=float, default=None)
    parser.add_argument("--th13_deg", type=float, default=None)
    parser.add_argument("--th23_deg", type=float, default=None)
    parser.add_argument("--d_cp_deg", type=float, default=None)
    parser.add_argument("--use_SM_potential", type=str, default=None, help="Set to 'true' or 'false'")
    parser.add_argument("--n_e_earth_cm3", type=float, default=None)
    parser.add_argument("--detector_depth_km", type=float, default=None)
    
    # Dark Portal Vector Interactions Overrides
    parser.add_argument("--u_e", type=float, default=None)
    parser.add_argument("--u_mu", type=float, default=None)
    parser.add_argument("--u_tau", type=float, default=None)
    parser.add_argument("--m_Zd_eV", type=float, default=None)
    parser.add_argument("--g_D", type=float, default=None)
    parser.add_argument("--epsilon_prime", type=float, default=None)
    parser.add_argument("--use_debye_screening", type=str, default=None, help="Set to 'true' or 'false'")
    
    # Primary Flux Injectors Overrides
    parser.add_argument("--p_index", type=float, default=None)
    parser.add_argument("--Phi_0", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--rho_evol", type=float, default=None)
    parser.add_argument("--f_e_S", type=float, default=None)
    parser.add_argument("--f_mu_S", type=float, default=None)
    parser.add_argument("--f_tau_S", type=float, default=None)
    parser.add_argument("--is_antineutrino", type=str, default=None)
    parser.add_argument("--nubar_to_nu_ratio", type=float, default=None)
    
    # Grid Configurations Overrides
    parser.add_argument("--E_min_eV", type=float, default=None)
    parser.add_argument("--E_max_eV", type=float, default=None)
    parser.add_argument("--n_E_bins", type=int, default=None)
    parser.add_argument("--n_z_bins", type=int, default=None)
    parser.add_argument("--n_z_thermal", type=int, default=None, help="Coarse z-samples for the thermal-rate grid (default 0 = exact, no interpolation; see build_thermal_grid before enabling)")
    parser.add_argument("--thermal_analytic", type=str, default=None, help="Set to 'false' to use the 100-point midpoint reference kernel for the thermal rates instead of the semi-analytic default (see build_thermal_grid)")
    parser.add_argument("--earth_step_scheme", type=str, default=None, help="Earth-transit potential discretization: 'avg' (default: exact step-averaged potential, step-count-independent LRI phase -- see dp_spatial._earth_chord_worker) or 'left' (historical left-endpoint sampling, kept for reproducing pre-2026-07 results)")
    parser.add_argument("--earth_zenith_average", type=str, default=None, help="Average over the Earth chords: 'phase' (default: integrate each block of the chords' density matrices against its dark-portal phase over cos(zenith), removing the chord-sampling floor) or 'plain' (historical equal-weight mean; required with --earth_step_scheme left)")
    parser.add_argument("--n_mw_paths", type=int, default=None, help="Number of Milky Way trajectories for solid-angle average")
    parser.add_argument("--n_earth_chords", type=int, default=None, help="Number of Earth chords for PREM evaluation")
    parser.add_argument("--N_steps_MW", type=int, default=None, help="Number of spatial integration steps in the Milky Way")
    parser.add_argument("--N_steps_earth", type=int, default=None, help="Number of spatial integration steps per Earth chord (default 25 under the 'avg' step scheme, whose results are certified step-count-independent; under the legacy 'left' scheme results scatter ~1e-2 with any step-count change and 500 was the pinned value)")
    parser.add_argument("--r_start_kpc", type=float, default=None, help="Starting distance for Milky Way integration in kpc")
    
    # Export Slated Locations Overrides
    parser.add_argument("--flux_filename", type=str, default=None)
    parser.add_argument("--plot_filename", type=str, default=None)
    
    # Execution Architecture Overrides
    parser.add_argument("--n_cores", type=int, default=None)
    parser.add_argument("--use_gpu", type=str, default=None, help="Set to 'true' to run the spatial handover on a CUDA GPU via cupy (falls back to CPU if unavailable)")
    parser.add_argument("--verbose", type=int, default=None)
    
    args = parser.parse_args()

    # Baseline defaults come from the single canonical DEFAULT_CONFIG (no more
    # separately-maintained copy here).
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Precedence Step 2: Overlay values from JSON file if one exists
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            for section, entries in file_config.items():
                if section in config and isinstance(entries, dict):
                    config[section].update(entries)
                else:
                    # Unknown section (or malformed entry): pass through untouched so
                    # run_solver's merge_config_with_defaults rejects it with a clear
                    # error, rather than silently dropping it here.
                    config[section] = entries
    else:
        print(f"Config file {args.config} not found. Running under strict system defaults.")

    def str_to_bool(val):
        if val is None: return None
        return str(val).lower().strip() in ['true', '1', 'yes', 'y']

    # Precedence Step 3: Overwrite entries dynamically via shell CLI arguments
    if args.H0_km_s_Mpc is not None: config["cosmology"]["H0_km_s_Mpc"] = args.H0_km_s_Mpc
    if args.Omega_m is not None: config["cosmology"]["Omega_m"] = args.Omega_m
    if args.Omega_L is not None: config["cosmology"]["Omega_L"] = args.Omega_L
    if args.T_nu0_eV is not None: config["cosmology"]["T_nu0_eV"] = args.T_nu0_eV
    if args.n_n0_cm3 is not None: config["cosmology"]["n_n0_cm3"] = args.n_n0_cm3
    if args.m_lightest_eV is not None: config["cosmology"]["m_lightest_eV"] = args.m_lightest_eV
    if args.mass_ordering is not None: config["cosmology"]["mass_ordering"] = args.mass_ordering
    if args.eta_e is not None: config["cosmology"]["eta_e"] = args.eta_e
    if args.eta_mu is not None: config["cosmology"]["eta_mu"] = args.eta_mu
    if args.eta_tau is not None: config["cosmology"]["eta_tau"] = args.eta_tau

    if args.dm21_sq is not None: config["standard_mixing"]["dm21_sq"] = args.dm21_sq
    if args.dm31_sq is not None: config["standard_mixing"]["dm31_sq"] = args.dm31_sq
    if args.th12_deg is not None: config["standard_mixing"]["th12_deg"] = args.th12_deg
    if args.th13_deg is not None: config["standard_mixing"]["th13_deg"] = args.th13_deg
    if args.th23_deg is not None: config["standard_mixing"]["th23_deg"] = args.th23_deg
    if args.d_cp_deg is not None: config["standard_mixing"]["d_cp_deg"] = args.d_cp_deg
    if args.use_SM_potential is not None: config["standard_mixing"]["use_SM_potential"] = str_to_bool(args.use_SM_potential)
    if args.n_e_earth_cm3 is not None: config["standard_mixing"]["n_e_earth_cm3"] = args.n_e_earth_cm3
    if args.detector_depth_km is not None: config["standard_mixing"]["detector_depth_km"] = args.detector_depth_km

    if args.u_e is not None: config["dark_portal"]["u_e"] = args.u_e
    if args.u_mu is not None: config["dark_portal"]["u_mu"] = args.u_mu
    if args.u_tau is not None: config["dark_portal"]["u_tau"] = args.u_tau
    if args.m_Zd_eV is not None: config["dark_portal"]["m_Zd_eV"] = args.m_Zd_eV
    if args.g_D is not None: config["dark_portal"]["g_D"] = args.g_D
    if args.epsilon_prime is not None: config["dark_portal"]["epsilon_prime"] = args.epsilon_prime
    if args.use_debye_screening is not None: config["dark_portal"]["use_debye_screening"] = str_to_bool(args.use_debye_screening)

    if args.p_index is not None: config["source"]["p_index"] = args.p_index
    if args.Phi_0 is not None: config["source"]["Phi_0"] = args.Phi_0
    if args.z_max is not None: config["source"]["z_max"] = args.z_max
    if args.rho_evol is not None: config["source"]["rho_evol"] = args.rho_evol
    if args.f_e_S is not None: config["source"]["f_e_S"] = args.f_e_S
    if args.f_mu_S is not None: config["source"]["f_mu_S"] = args.f_mu_S
    if args.f_tau_S is not None: config["source"]["f_tau_S"] = args.f_tau_S
    if args.is_antineutrino is not None: config["source"]["is_antineutrino"] = args.is_antineutrino
    if args.nubar_to_nu_ratio is not None: config["source"]["nubar_to_nu_ratio"] = args.nubar_to_nu_ratio

    if args.E_min_eV is not None: config["grid"]["E_min_eV"] = args.E_min_eV
    if args.E_max_eV is not None: config["grid"]["E_max_eV"] = args.E_max_eV
    if args.n_E_bins is not None: config["grid"]["n_E_bins"] = args.n_E_bins
    if args.n_z_bins is not None: config["grid"]["n_z_bins"] = args.n_z_bins
    if args.n_z_thermal is not None: config["grid"]["n_z_thermal"] = args.n_z_thermal
    if args.thermal_analytic is not None: config["grid"]["thermal_analytic"] = str_to_bool(args.thermal_analytic)
    if args.earth_step_scheme is not None: config["grid"]["earth_step_scheme"] = args.earth_step_scheme
    if args.earth_zenith_average is not None: config["grid"]["earth_zenith_average"] = args.earth_zenith_average
    if args.n_mw_paths is not None: config["grid"]["n_mw_paths"] = args.n_mw_paths
    if args.n_earth_chords is not None: config["grid"]["n_earth_chords"] = args.n_earth_chords
    if args.N_steps_MW is not None: config["grid"]["N_steps_MW"] = args.N_steps_MW
    if args.N_steps_earth is not None: config["grid"]["N_steps_earth"] = args.N_steps_earth
    if args.r_start_kpc is not None: config["grid"]["r_start_kpc"] = args.r_start_kpc

    if args.flux_filename is not None: config["output"]["flux_filename"] = args.flux_filename
    if args.plot_filename is not None: config["output"]["plot_filename"] = args.plot_filename

    if args.n_cores is not None: config["execution"]["n_cores"] = args.n_cores
    if args.use_gpu is not None: config["execution"]["use_gpu"] = str_to_bool(args.use_gpu)
    if args.verbose is not None: config["execution"]["verbose"] = args.verbose
            
    import time as _time
    _t0 = _time.perf_counter()
    E_array, Flux_array = run_solver(config)
    _runtime = _time.perf_counter() - _t0

    output_cfg = config.get("output", {})
    flux_filename = output_cfg.get("flux_filename", "../output/data/output_flux.txt")
    plot_filename = output_cfg.get("plot_filename", "../output/plots/flux_plot.png")

    os.makedirs(os.path.dirname(os.path.abspath(flux_filename)), exist_ok=True)
    save_flux_to_file(E_array, Flux_array, filename=flux_filename)
    write_run_manifest(flux_filename + ".manifest.json", config, runtime_s=_runtime)
    
    Flux_array = np.clip(Flux_array, 1e-50, None) 
    Total_Flux = np.sum(Flux_array, axis=1)
    f_e, f_mu, f_tau = Flux_array[:, 0] / Total_Flux, Flux_array[:, 1] / Total_Flux, Flux_array[:, 2] / Total_Flux
    
    safe_print("\nIntegration Complete. Sample Output (Mid-grid Energy):")
    mid_idx = len(E_array) // 2
    safe_print(f"Energy: {E_array[mid_idx] / 1e9:.2e} GeV")
    safe_print(f"Flavor Ratios at Earth [e, mu, tau]: [{f_e[mid_idx]:.3f}, {f_mu[mid_idx]:.3f}, {f_tau[mid_idx]:.3f}]")

    if HAS_MATPLOTLIB:
        os.makedirs(os.path.dirname(os.path.abspath(plot_filename)), exist_ok=True)
        plot_flavor_fluxes(E_array, Flux_array, config['source']['p_index'], config['source'].get('is_antineutrino', 'sum'), output_filename=plot_filename)