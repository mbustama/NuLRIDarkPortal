"""
kernel.py

USER MODIFICATION REQUIRED HERE:
This module isolates the actual physics evaluation and mathematical payload.
Configure your specific observable topologies, arrays, and labels here.
"""

import sys
import os
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

# Import external physics engines
from dark_portal_solver import propagate_spectrum
from dark_portal_scanner import compute_energy_averaged_ratios

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

def construct_PMNS_kernel(th12: float, th13: float, th23: float, d_cp: float) -> np.ndarray:
    """Internal kernel rebuild mechanism for standard 3x3 unitary PMNS mixing."""
    c12, s12 = np.cos(th12), np.sin(th12)
    c13, s13 = np.cos(th13), np.sin(th13)
    c23, s23 = np.cos(th23), np.sin(th23)
    phase = np.exp(1j * d_cp)
    phase_conj = np.exp(-1j * d_cp)
    R12 = np.zeros((3, 3), dtype=np.complex128)
    R12[0, 0] = c12;  R12[0, 1] = s12
    R12[1, 0] = -s12; R12[1, 1] = c12
    R12[2, 2] = 1.0
    R13 = np.zeros((3, 3), dtype=np.complex128)
    R13[0, 0] = c13;  R13[0, 2] = s13 * phase_conj
    R13[1, 1] = 1.0
    R13[2, 0] = -s13 * phase; R13[2, 2] = c13
    R23 = np.zeros((3, 3), dtype=np.complex128)
    R23[0, 0] = 1.0
    R23[1, 1] = c23;  R23[1, 2] = s23
    R23[2, 1] = -s23; R23[2, 2] = c23
    return R23 @ R13 @ R12

def get_mass_eigenvalues_kernel(m_lightest: float, dm21_sq: float, dm31_sq: float, ordering: str) -> np.ndarray:
    """Internal kernel rebuild mechanism for absolute mass mapping."""
    masses = np.zeros(3, dtype=np.float64)
    if str(ordering).lower().strip() == "inverted":
        masses[2] = m_lightest
        masses[1] = np.sqrt(masses[2]**2 + dm31_sq)
        masses[0] = np.sqrt(masses[1]**2 - dm21_sq)
    else:
        masses[0] = m_lightest
        masses[1] = np.sqrt(masses[0]**2 + dm21_sq)
        masses[2] = np.sqrt(masses[0]**2 + dm31_sq)
    return masses

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

def compute_user_model_single(params, static_data):
    """
    Standard Python wrapper. Evaluates the dark portal propagation by calling 
    external solvers and returns the predicted flavor fractions at Earth.
    """
    # Unpack the immutable data tuple (Expanded to 26 elements)
    (obs_f_e, obs_f_mu, obs_err_e, obs_err_mu, 
     E_bins, z_bins, rho_src, H0, Omega_m, Omega_L, 
     U_pmns, M_sq, m_nu_bg_arr, n_n0, T_nu0, eta_bg, 
     detector_depth_km, is_antineutrino, p_index, rho_evol, Phi_0,
     nubar_to_nu_ratio, n_mw_paths, n_earth_chords, 
     N_steps_MW, r_start_kpc) = static_data
     
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
            U_pmns = construct_PMNS_kernel(t12, t13, t23, dcp)

        if any(k in params for k in ('dm21_sq', 'dm31_sq', 'm_lightest_eV', 'mass_ordering')):
            d21 = params.get('dm21_sq', 7.41e-5)
            d31 = params.get('dm31_sq', 2.511e-3)
            ml = params.get('m_lightest_eV', 0.01)
            mo = params.get('mass_ordering', 'normal')
            
            dm31_sq_signed = -d31 if str(mo).lower().strip() == "inverted" else d31
            M_sq = np.ascontiguousarray(np.diag([0.0, d21, dm31_sq_signed]).astype(np.complex128))
            m_nu_bg_arr = get_mass_eigenvalues_kernel(ml, d21, d31, mo)
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
    r_start_kpc = float(r_start_kpc)
    
    # Rebuild physics meshes if they are mock scalars
    E_bins = np.atleast_1d(np.array(E_bins, dtype=np.float64))
    if E_bins.size < 2: 
        E_bins = np.geomspace(1e4, 1e8, 10)
        
    z_bins = np.atleast_1d(np.array(z_bins, dtype=np.float64))
    if z_bins.size < 2: 
        z_bins = np.linspace(0, 4, 10)
        
    rho_src = np.atleast_1d(np.array(rho_src, dtype=np.float64))
    if rho_src.size != E_bins.size: 
        rho_src = np.ones_like(E_bins)
        
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

    # Hardcoded structural booleans and integers required by the solver signature
    use_debye_screening = True
    use_SM_potential = False
    n_e_earth = 0.0 # Bypassed if use_SM_potential is False
    show_progress = False
    verbose = 0
    
    # 3. Propagate the spectrum through the cosmological and spatial boundaries
    flag_val = str(is_antineutrino).strip().lower()

    def run_propagation(is_anti_flag):
        local_U_pmns = np.ascontiguousarray(np.conjugate(U_pmns_complex)) if is_anti_flag else np.ascontiguousarray(U_pmns_complex)
        local_u_arr = np.ascontiguousarray(np.conjugate(u_arr_base)) if is_anti_flag else np.ascontiguousarray(u_arr_base)

        return propagate_spectrum(
            E_bins, z_bins, rho_src, H0, Omega_m, Omega_L, local_U_pmns, M_sq_complex,
            local_u_arr, m_Zd_eV, g_D, epsilon_prime, m_nu_bg_arr, n_n0, T_nu0,
            use_debye_screening, detector_depth_km, use_SM_potential, n_e_earth,
            is_anti_flag, p_index, rho_evol, Phi_0, eta_bg,
            n_mw_paths, n_earth_chords, N_steps_MW, r_start_kpc,
            show_progress, verbose
        )

    # Evaluate flags to match solver's native worker capabilities
    if flag_val in ['sum', 'both']:
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