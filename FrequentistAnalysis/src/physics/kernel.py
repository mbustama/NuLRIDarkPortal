"""
kernel.py

USER MODIFICATION REQUIRED HERE:
This module isolates the actual physics evaluation and mathematical payload.
Configure your specific observable topologies, arrays, and labels here.
"""

import sys
import os
import atexit
import multiprocessing
import numpy as np

# Get the absolute path of the directory containing THIS file (kernel.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define relative path FROM kernel.py TO the dark portal folder
relative_path_to_dark_portal = "../../../src/" 

# Join them and resolve to an absolute path dynamically
dark_portal_dir = os.path.abspath(os.path.join(current_dir, relative_path_to_dark_portal))

# Inject into sys.path
if dark_portal_dir not in sys.path:
    sys.path.insert(0, dark_portal_dir)

# Import external physics engines. construct_PMNS/get_mass_eigenvalues are the single
# canonical implementations (also used by dark_portal_solver.py itself); importing them
# rather than keeping a local copy avoids the two drifting out of sync.
from dark_portal_solver import (propagate_spectrum, construct_PMNS, get_mass_eigenvalues,
                                compute_gamma_grid_inputs, build_thermal_grid,  # noqa: F401
                                thermal_grid_cached,
                                cm_inv3_to_eV3, DEFAULT_CONFIG as _DC)
# Imported from flux_postprocessing (not dark_portal_scanner, its previous home):
# importing the scanner has side effects -- it globally sets
# dark_portal_solver.HAS_TQDM = False and imports matplotlib -- that this kernel
# never wanted.
from flux_postprocessing import compute_energy_averaged_ratios

# Updated LaTeX labels mapping both dark portal and cosmo-physical parameter space
LATEX_LABELS = {
    "m_Zd_eV": r"m_{Z_D} \text{ [eV]}",
    "g_D": r"g_D",
    "epsilon_prime": r"\epsilon'",
    "u_e": r"u_e",
    "u_mu": r"u_\mu",
    "u_tau": r"u_\tau",
    "f_e_S": r"f_{e, S}",
    "f_mu_S": r"f_{\mu, S}",
    "f_tau_S": r"f_{\tau, S}",
    "th12_deg": r"\theta_{12} (^\circ)",
    "th13_deg": r"\theta_{13} (^\circ)",
    "th23_deg": r"\theta_{23} (^\circ)",
    "dm21_sq": r"\Delta m^2_{21}",
    "dm31_sq": r"\Delta m^2_{31}",
}

def get_label(param):
    """
    Fetches LaTeX labels for plotting, parsing internal underscores if unassigned.
    """
    base_label = LATEX_LABELS.get(param, param.replace('_', r'\_'))
    return rf"${base_label}$"

def format_title_stats(param, results_dict):
    """
    Parses frequentist extracted limits and format them dynamically into LaTeX strings
    used for matplotlib sub-plot titles.
    """
    base_label = LATEX_LABELS.get(param, param.replace('_', r'\_'))
    if param not in results_dict: return rf"${base_label}$"
    res = results_dict[param]
    bf = res.get('best_fit')
    if bf is None: return rf"${base_label}$"
    intervals = res.get('1sigma')
    
    if not intervals or len(intervals) == 0: return rf"${base_label} = {bf:.3g}$"
    
    target_interval = intervals[0]
    for interval in intervals:
        if interval[0] <= bf <= interval[1]:
            target_interval = interval
            break
            
    low, high = target_interval
    return rf"${base_label} = {bf:.3g}_{{-{bf - low:.3g}}}^{{+{high - bf:.3g}}}$"

# =============================================================================
# PERSISTENT SPATIAL-HANDOVER POOL (MainProcess-only)
# =============================================================================
# compute_user_model_single runs once per Minuit function evaluation -- potentially
# hundreds of times during main.py's initial global fit alone. propagate_spectrum's spatial
# handover distributes independent Milky Way paths / Earth chords across a
# multiprocessing.Pool when n_cores > 1; creating a fresh Pool per call is cheap under the
# 'fork' start method but expensive under 'spawn' (a full interpreter relaunch plus reimport
# per worker) -- and main.py forces 'spawn'. A pool reused across the whole fit instead pays
# that cost once. This module-level pool is only ever touched from the main process: scan/
# repair Pool workers always evaluate with n_cores=1 and never call _get_persistent_pool.
_persistent_pool = None
_persistent_pool_size = None
_persistent_pool_atexit_registered = False

def _close_persistent_pool():
    global _persistent_pool
    if _persistent_pool is not None:
        _persistent_pool.terminate()
        _persistent_pool.join()
        _persistent_pool = None

def _get_persistent_pool(n_cores):
    """Returns a persistent multiprocessing.Pool of exactly n_cores workers, creating (or
    resizing) it on first use. Returns None if n_cores <= 1 (no pool needed)."""
    global _persistent_pool, _persistent_pool_size, _persistent_pool_atexit_registered
    if n_cores <= 1:
        return None
    if _persistent_pool is not None and _persistent_pool_size == n_cores:
        return _persistent_pool
    _close_persistent_pool()
    _persistent_pool = multiprocessing.Pool(processes=n_cores)
    _persistent_pool_size = n_cores
    if not _persistent_pool_atexit_registered:
        atexit.register(_close_persistent_pool)
        _persistent_pool_atexit_registered = True
    return _persistent_pool

def prewarm_worker():
    """Pool initializer: warms a fresh worker process's numba JIT caches with one tiny
    end-to-end kernel evaluation BEFORE any real task lands.

    Under the 'spawn' start method (forced by main.py/run_limits.py) every worker is a
    fresh interpreter: without this, each worker lazily imports and loads/compiles every
    numba kernel on its FIRST real task -- at HPC scale (up to ~200 workers) that is a
    simultaneous cold-start stampede serialized against real work. The evaluation below
    uses a 10-bin grid with minimal spatial settings (~0.1 s warm) and touches the whole
    compiled path: thermal grid, sigma table, cosmological stepper, spatial handover,
    propagator. Best-effort by design -- a failure here must never kill the pool; real
    tasks will surface real errors."""
    try:
        E_bins = np.geomspace(1e13, 3e16, 10)
        z_bins = np.linspace(4.0, 0.0, 10)
        static = (
            0.33, 0.33, 0.05, 0.05, E_bins, z_bins,
            np.array([0.333333, 0.666667, 0.0]),
            1.4374e-33, 0.315, 0.685,
            construct_PMNS(0.5831, 0.1498, 0.8570, 2.3038),
            np.ascontiguousarray(np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128)),
            get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal"),
            _DC['cosmology']['n_n0_cm3'] * cm_inv3_to_eV3, 1.95e-4,
            np.ones(3, dtype=np.float64),
            1.5, False, 2.5, 0.0, 1.0, 1.0,
            1, 1, 2, 100.0, 2,          # n_mw_paths, n_earth_chords, N_steps_MW, r_start_kpc, N_steps_earth
            False, 0.0, True, 1,        # use_SM_potential, n_e_earth, use_debye_screening, n_cores_standalone
            None, None, 0, True, "avg")  # V_SM_struct, U_LRI_struct, n_z_thermal, thermal_analytic, earth_step_scheme
        compute_user_model_single({"m_Zd_eV": 2.0e6, "g_D": 1e-4, "epsilon_prime": 0.1}, static)
    except Exception:
        pass


def compute_user_model_single(params, static_data):
    """
    Standard Python wrapper. Evaluates the dark portal propagation by calling 
    external solvers and returns the predicted flavor fractions at Earth.
    """
    # Unpack the immutable data tuple (currently 36 elements; every element beyond the
    # original 26 -- N_steps_earth, use_SM_potential, n_e_earth, use_debye_screening,
    # n_cores_standalone, V_SM_struct, U_LRI_struct, n_z_thermal, thermal_analytic,
    # earth_step_scheme --
    # defaults to a safe value for any caller still passing an older, shorter tuple.
    # N_steps_earth=500 is the only value validated against resonance-crossing
    # sensitivity, see propagate_earth_spatial's docstring in dark_portal_solver.py).
    (obs_f_e, obs_f_mu, obs_err_e, obs_err_mu,
     E_bins, z_bins, rho_src, H0, Omega_m, Omega_L,
     U_pmns, M_sq, m_nu_bg_arr, n_n0, T_nu0, eta_bg,
     detector_depth_km, is_antineutrino, p_index, rho_evol, Phi_0,
     nubar_to_nu_ratio, n_mw_paths, n_earth_chords,
     N_steps_MW, r_start_kpc) = static_data[:26]
    N_steps_earth = static_data[26] if len(static_data) > 26 else _DC['grid']['N_steps_earth']
    # use_SM_potential/n_e_earth_cm3/use_debye_screening are present in every shipped
    # physics_constants block but were previously hardcoded here (False/0.0/True)
    # regardless of what the config said -- dormant since every shipped config wants
    # use_SM_potential=false anyway, but silently dropped the config value the moment
    # anyone flipped it (same bug class as the already-fixed n_e_earth_cm3 disconnect in
    # dark_portal_solver.py's own run_solver).
    use_SM_potential = bool(static_data[27]) if len(static_data) > 27 else _DC['standard_mixing']['use_SM_potential']
    n_e_earth = float(static_data[28]) if len(static_data) > 28 else _DC['standard_mixing']['n_e_earth_cm3'] * cm_inv3_to_eV3
    use_debye_screening = bool(static_data[29]) if len(static_data) > 29 else _DC['dark_portal']['use_debye_screening']
    n_cores_standalone = int(static_data[30]) if len(static_data) > 30 else 1
    # Optional flavor-structure overrides (neutrino convention; propagate_spectrum applies
    # the antineutrino conjugation centrally). None means the built-in defaults.
    V_SM_struct = static_data[31] if len(static_data) > 31 else None
    U_LRI_struct = static_data[32] if len(static_data) > 32 else None
    # Coarse z-sampling of the thermal-rate grid (0 = exact); see build_thermal_grid.
    n_z_thermal = int(static_data[33]) if len(static_data) > 33 else _DC['grid']['n_z_thermal']
    # Semi-analytic momentum integration of the thermal rates (default True); False
    # selects the 100-point midpoint reference kernel -- see build_thermal_grid.
    thermal_analytic = bool(static_data[34]) if len(static_data) > 34 else _DC['grid']['thermal_analytic']
    # Earth-transit potential discretization ('avg' = exact step-averaged, the default;
    # 'left' = historical left-endpoint sampling) -- see _earth_chord_worker.
    earth_step_scheme = str(static_data[35]) if len(static_data) > 35 else _DC['grid']['earth_step_scheme']

    # 1. Unpack scanning parameters from the generic Minuit engine wrapper.
    #    Handles backwards compatibility if passed as a positional array.
    if isinstance(params, dict):
        m_Zd_eV       = params.get('m_Zd_eV', 2.0e6)
        g_D           = params.get('g_D', 1e-4)
        epsilon_prime = params.get('epsilon_prime', 0.1)
        u_e           = params.get('u_e', 0.5)
        u_mu          = params.get('u_mu', 0.5)
        u_tau         = params.get('u_tau', 0.0)
        
        u_arr_base = np.array([u_e, u_mu, u_tau], dtype=np.complex128)

        # Reconstruct dynamic Numba arrays internally if their underlying components are floating
        f_e_S = params.get('f_e_S', rho_src[0])
        f_mu_S = params.get('f_mu_S', rho_src[1])
        f_tau_S = params.get('f_tau_S', rho_src[2])
        rho_src = np.array([f_e_S, f_mu_S, f_tau_S], dtype=np.float64)

        eta_e = params.get('eta_e', eta_bg[0])
        eta_mu = params.get('eta_mu', eta_bg[1])
        eta_tau = params.get('eta_tau', eta_bg[2])
        eta_bg = np.array([eta_e, eta_mu, eta_tau], dtype=np.float64)

        # Standard Oscillation Mixing recalculation hooks
        if any(k in params for k in ('th12_deg', 'th13_deg', 'th23_deg', 'd_cp_deg')):
            t12 = np.radians(params.get('th12_deg', 33.41))
            t13 = np.radians(params.get('th13_deg', 8.58))
            t23 = np.radians(params.get('th23_deg', 49.1))
            dcp = np.radians(params.get('d_cp_deg', 132.0))
            U_pmns = construct_PMNS(t12, t13, t23, dcp)

        if any(k in params for k in ('dm21_sq', 'dm31_sq', 'm_lightest_eV', 'mass_ordering')):
            d21 = params.get('dm21_sq', 7.41e-5)
            d31 = params.get('dm31_sq', 2.511e-3)
            ml = params.get('m_lightest_eV', 0.01)
            mo = params.get('mass_ordering', 'normal')
            
            dm31_sq_signed = -d31 if str(mo).lower().strip() == "inverted" else d31
            M_sq = np.ascontiguousarray(np.diag([0.0, d21, dm31_sq_signed]).astype(np.complex128))
            m_nu_bg_arr = get_mass_eigenvalues(ml, d21, d31, mo)
    else:
        # Fallback for raw arrays 
        m_Zd_eV       = params[0]
        g_D           = params[1]
        epsilon_prime = params[2]
        u_arr_base = np.array([params[3], params[4], params[5]], dtype=np.complex128)
    
    # ---------------------------------------------------------
    # SANITIZE AND RECONSTRUCT MOCK ASIMOV ARRAYS
    # If the GenericLikelihoodWrapper injects a mock tuple of scalars 
    # (e.g., 1.0) because the config is missing, Numba will violently 
    # crash on dimension and broadcast checks. We must rebuild them.
    # ---------------------------------------------------------
    
    # Cast expected scalars strictly to float
    H0 = float(H0)
    Omega_m = float(Omega_m)
    Omega_L = float(Omega_L)
    n_n0 = float(n_n0)
    T_nu0 = float(T_nu0)
    detector_depth_km = float(detector_depth_km)
    p_index = float(p_index)
    rho_evol = float(rho_evol)
    Phi_0 = float(Phi_0)
    nubar_to_nu_ratio = float(nubar_to_nu_ratio)
    
    # Cast expected spatial grid variables
    n_mw_paths = int(n_mw_paths)
    n_earth_chords = int(n_earth_chords)
    N_steps_MW = int(N_steps_MW)
    N_steps_earth = int(N_steps_earth)
    r_start_kpc = float(r_start_kpc)
    
    # Rebuild physics meshes if they are mock scalars
    E_bins = np.atleast_1d(np.array(E_bins, dtype=np.float64))
    if E_bins.size < 2: 
        E_bins = np.geomspace(1e4, 1e8, 10)
        
    z_bins = np.atleast_1d(np.array(z_bins, dtype=np.float64))
    if z_bins.size < 2: 
        z_bins = np.linspace(0, 4, 10)
        
    # FIX: Enforce flavor array size to strictly 3, not E_bins.size
    rho_src = np.atleast_1d(np.array(rho_src, dtype=np.float64))
    if rho_src.size != 3: 
        rho_src = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float64)
        
    m_nu_bg_arr = np.atleast_1d(np.array(m_nu_bg_arr, dtype=np.float64))
    if m_nu_bg_arr.size != 3: 
        m_nu_bg_arr = np.array([0.0, 0.01, 0.05], dtype=np.float64)
        
    # Reconstruct the eta_bg array safely to dodge Numba float broadcasting crashes
    eta_bg = np.atleast_1d(np.array(eta_bg, dtype=np.float64))
    if eta_bg.size != 3:
        eta_bg = np.ones(3, dtype=np.float64)
        
    # Rebuild quantum matrices (Force 3x3 C-contiguous)
    U_pmns_complex = np.array(U_pmns, dtype=np.complex128)
    if U_pmns_complex.shape != (3, 3):
        if U_pmns_complex.size == 9:
            U_pmns_complex = U_pmns_complex.reshape((3, 3))
        else:
            U_pmns_complex = np.eye(3, dtype=np.complex128)
    U_pmns_complex = np.ascontiguousarray(U_pmns_complex)
    
    M_sq_complex = np.array(M_sq, dtype=np.complex128)
    if M_sq_complex.shape != (3, 3):
        if M_sq_complex.size == 9:
            M_sq_complex = M_sq_complex.reshape((3, 3))
        else:
            M_sq_complex = np.diag([0.0, 7.5e-5, 2.5e-3]).astype(np.complex128)
    M_sq_complex = np.ascontiguousarray(M_sq_complex)
    # ---------------------------------------------------------

    # use_debye_screening/use_SM_potential/n_e_earth were unpacked from static_data above.
    show_progress = False
    verbose = 0
    
    # 3. Propagate the spectrum through the cosmological and spatial boundaries
    flag_val = str(is_antineutrino).strip().lower()

    # propagate_spectrum's own n_cores must stay 1 whenever this call is running inside
    # main.py's scan/repair multiprocessing.Pool (each worker must not itself spin up a
    # nested pool -- oversubscription hazard, see the BLAS/numba thread-pinning guards in
    # dark_portal_solver.py). But the initial global m.simplex()/m.migrad() fit in main.py
    # runs in the *main* process, before any pool worker ever touches this function, and
    # every one of Migrad's function evaluations there was previously paying full single-core
    # cost for no reason. multiprocessing.current_process().name is 'MainProcess' only in the
    # former case, never inside a Pool worker (those are named e.g. 'SpawnPoolWorker-N'), so
    # this distinguishes the two safely.
    #
    # A persistent, module-level Pool (_get_persistent_pool) is reused across every one of
    # these MainProcess calls rather than creating a fresh one per evaluation -- essential
    # under main.py's forced 'spawn' start method, where per-call Pool creation costs far more
    # than the multi-core speedup saves (measured ~10x *slower* without reuse). With reuse, a
    # single-flavor propagate_spectrum call at FrequentistAnalysis-scale grid settings
    # (n_mw_paths=50, n_earth_chords=50, N_steps_earth=500) measured ~2.9x faster at n_cores=8
    # vs 1.
    if multiprocessing.current_process().name == 'MainProcess':
        n_cores_for_call = n_cores_standalone
        persistent_pool = _get_persistent_pool(n_cores_standalone)
    else:
        n_cores_for_call = 1
        persistent_pool = None

    # Pin numba's thread pool BEFORE the thermal-grid precompute below. That prange kernel
    # previously ran before propagate_spectrum's own set_num_threads call, so on the FIRST
    # evaluation inside each scan-pool worker it used numba's default thread count (all
    # cores) -- transient but real oversubscription when hundreds of workers hit their
    # first evaluation together. propagate_spectrum sets the same value again downstream.
    try:
        import numba
        numba.set_num_threads(max(1, int(n_cores_for_call)))
    except ImportError:
        pass

    # The thermal-rate grid (precompute_all_thermal_rates) depends only on |U_pmns| and
    # |u_arr| magnitudes, which conjugation never changes -- so it is identical for the
    # neutrino and antineutrino calls below. compute_user_model_single runs on every single
    # Minuit/MCMC/scan evaluation, so computing this grid once and sharing it (instead of
    # letting each flavor call redo this from scratch) matters a lot more here in aggregate
    # than it does for a single dark_portal_solver.py run.
    sum_G, sum_G_alpha, eta_mass = compute_gamma_grid_inputs(U_pmns_complex, u_arr_base, g_D, epsilon_prime, eta_bg)
    # thermal_grid_cached: the two-component coupling cache -- profiled fits and
    # coupling scans at fixed m_Zd reuse a verified I_A + I_B/sum_G surrogate after
    # their first ~3 evaluations (see the cache NOTE in dark_portal_solver.py);
    # m_Zd-scan evaluations pay exactly one exact build per point, as before.
    Gamma_grid_all = thermal_grid_cached(E_bins, z_bins, sum_G, sum_G_alpha, m_Zd_eV, m_nu_bg_arr, T_nu0, eta_mass,
                                         n_z_thermal, thermal_analytic=thermal_analytic)

    def run_propagation(is_anti_flag):
        local_U_pmns = np.ascontiguousarray(np.conjugate(U_pmns_complex)) if is_anti_flag else np.ascontiguousarray(U_pmns_complex)
        local_u_arr = np.ascontiguousarray(np.conjugate(u_arr_base)) if is_anti_flag else np.ascontiguousarray(u_arr_base)

        return propagate_spectrum(
            E_bins, z_bins, rho_src, H0, Omega_m, Omega_L, local_U_pmns, M_sq_complex,
            local_u_arr, m_Zd_eV, g_D, epsilon_prime, m_nu_bg_arr, n_n0, T_nu0,
            use_debye_screening, detector_depth_km, use_SM_potential, n_e_earth,
            is_anti_flag, p_index, rho_evol, Phi_0, eta_bg,
            n_mw_paths, n_earth_chords, N_steps_MW, r_start_kpc,
            show_progress, verbose, n_cores_for_call, Gamma_grid_all, N_steps_earth,
            external_pool=persistent_pool,
            U_LRI_struct=U_LRI_struct, V_SM_struct=V_SM_struct,
            earth_step_scheme=earth_step_scheme
        )

    # Evaluate flags to match solver's native worker capabilities
    if flag_val in ['sum', 'both']:
        # Degenerate weights: skip the propagation whose weight is exactly zero (free 2x
        # for nu-only/nubar-only studies configured through the ratio).
        if nubar_to_nu_ratio == 0.0:
            Flux_array = run_propagation(False)
        elif not np.isfinite(nubar_to_nu_ratio):
            Flux_array = run_propagation(True)
        else:
            w_nu = 1.0 / (1.0 + nubar_to_nu_ratio)
            w_nubar = nubar_to_nu_ratio / (1.0 + nubar_to_nu_ratio)
            Flux_array = w_nu * run_propagation(False) + w_nubar * run_propagation(True)
    elif flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        Flux_array = run_propagation(True)
    else:
        Flux_array = run_propagation(False)
    
    # 4. Collapse the integrated flux spectrum into energy-averaged flavor fractions
    ratios = compute_energy_averaged_ratios(E_bins, Flux_array)
    f_e_th = ratios[0]
    f_mu_th = ratios[1]
    
    # Derive the tau fraction enforcing unitarity
    f_tau_th = 1.0 - f_e_th - f_mu_th
    
    return f_e_th, f_mu_th, f_tau_th