"""
kernel.py

USER MODIFICATION REQUIRED HERE:
This module isolates the actual physics evaluation and mathematical payload.
Configure your specific observable topologies, arrays, and labels here.
"""

import sys
import os
import numpy as np

# ==============================================================================
# NUMBA JIT COMPILATION (WITH FALLBACK)
# ==============================================================================
try:
    from numba import njit
    NUMBA_AVAILABLE = True
    print("Numba is available. JIT compilation enabled for core math kernels.")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba is not installed. Falling back to standard Python execution.")
    def njit(*args, **kwargs):
        """Dummy decorator for systems lacking Numba. Preserves code functionality."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

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

# Updated LaTeX labels to map the dark portal parameter space
LATEX_LABELS = {
    "m_Zd_eV": r"m_{Z_D} \text{ [eV]}",
    "g_D": r"g_D",
    "epsilon_prime": r"\epsilon'",
    "u_e": r"u_e",
    "u_mu": r"u_\mu",
    "u_tau": r"u_\tau"
}

def get_label(param):
    """
    Fetches LaTeX labels for plotting, parsing internal underscores if unassigned.
    
    Parameters:
    - param (str): Exact dictionary parameter key.
    
    Returns:
    - str: Fully formatted LaTeX string.
    """
    base_label = LATEX_LABELS.get(param, param.replace('_', r'\_'))
    return rf"${base_label}$"

def format_title_stats(param, results_dict):
    """
    Parses frequentist extracted limits and format them dynamically into LaTeX strings
    used for matplotlib sub-plot titles.

    Parameters:
    - param (str): Target physics parameter name.
    - results_dict (dict): The active frequentist_results.json block.

    Returns:
    - str: A formatted LaTeX title (e.g. \beta = 0.5 +0.1 -0.2).
    """
    base_label = LATEX_LABELS.get(param, param.replace('_', r'\_'))
    if param not in results_dict: return rf"${base_label}$"
    res = results_dict[param]
    bf = res.get('best_fit')
    if bf is None: return rf"${base_label}$"
    intervals = res.get('1sigma')
    
    # Fallback if no valid 1-sigma intervals were mathematically resolved
    if not intervals or len(intervals) == 0: return rf"${base_label} = {bf:.3g}$"
    
    # Locate the specific interval bounding the best-fit value
    target_interval = intervals[0]
    for interval in intervals:
        if interval[0] <= bf <= interval[1]:
            target_interval = interval
            break
            
    low, high = target_interval
    return rf"${base_label} = {bf:.3g}_{{-{bf - low:.3g}}}^{{+{high - bf:.3g}}}$"

@njit(cache=True)
def compute_user_model_single(params, static_data):
    """
    Numba JIT-compiled mathematical kernel. Evaluates the dark portal propagation 
    and returns the predicted flavor fractions at Earth.
    
    Parameters:
    - params (np.ndarray): Contiguous 1D array of active float evaluations.
                           Expected strict order: [m_Zd_eV, g_D, epsilon_prime, u_e, u_mu, u_tau].
    - static_data (tuple): Immutable 22-element data block containing observation constraints, 
                           grid meshes, and physical constants.

    Returns:
    - tuple: (f_e_th, f_mu_th, f_tau_th) The expected Earth flavor fractions.
    """
    # 1. Unpack scanning parameters from the generic Minuit engine wrapper
    m_Zd_eV       = params[0]
    g_D           = params[1]
    epsilon_prime = params[2]
    
    # Construct the sterile mixing vector dynamically from the scan coordinates
    u_arr = np.array([params[3], params[4], params[5]], dtype=np.complex128)
    
    # 2. Unpack the immutable data tuple
    # NOTE: The IceCube observable variables (obs_f_e, obs_f_mu, obs_err_e, obs_err_mu) 
    # are unpacked here purely to maintain strict memory alignment with the 22-element 
    # configuration tuple. They are not utilized internally as the chi-square 
    # calculation has been offloaded to the external detector likelihood module.
    (obs_f_e, obs_f_mu, obs_err_e, obs_err_mu, 
     E_bins, z_bins, rho_src, H0, Omega_m, Omega_L, 
     U_pmns, M_sq, m_nu_bg_arr, n_n0, T_nu0, eta_bg, 
     detector_depth_km, is_antineutrino, p_index, rho_evol, Phi_0,
     nubar_to_nu_ratio) = static_data
     
    # Hardcoded structural booleans and integers required by the strict Numba signature
    use_debye_screening = True
    use_SM_potential = False
    n_e_earth = 0.0 # Bypassed if use_SM_potential is False
    n_mw_paths = 6
    n_earth_chords = 3
    N_steps_MW = 100
    r_start_kpc = 100.0
    show_progress = False
    verbose = 0
    
    # 3. Propagate the spectrum through the cosmological and spatial boundaries
    Flux_array = propagate_spectrum(
        E_bins, z_bins, rho_src, H0, Omega_m, Omega_L, U_pmns, M_sq, 
        u_arr, m_Zd_eV, g_D, epsilon_prime, m_nu_bg_arr, n_n0, T_nu0, 
        use_debye_screening, detector_depth_km, use_SM_potential, n_e_earth, 
        is_antineutrino, p_index, rho_evol, Phi_0, eta_bg, 
        n_mw_paths, n_earth_chords, N_steps_MW, r_start_kpc, 
        show_progress, verbose, nubar_to_nu_ratio
    )
    
    # 4. Collapse the integrated flux spectrum into energy-averaged flavor fractions
    ratios = compute_energy_averaged_ratios(E_bins, Flux_array)
    f_e_th = ratios[0]
    f_mu_th = ratios[1]
    
    # Derive the tau fraction enforcing unitarity
    f_tau_th = 1.0 - f_e_th - f_mu_th
    
    return f_e_th, f_mu_th, f_tau_th