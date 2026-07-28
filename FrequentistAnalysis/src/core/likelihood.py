"""
likelihood.py

Encapsulates the Minuit and Emcee objective function logic. Handles dynamic
evaluation caching and automated empirical prior spline construction.
"""

import os
import sys
import traceback
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, PchipInterpolator

# Import internal physics and external detector likelihood module
from src.physics.kernel import compute_user_model_single
from src.core.detector_flavor_likelihood import detector_flavor_log_likelihood_wrapper

# construct_PMNS/get_mass_eigenvalues are the single canonical implementations, defined
# once in dark_portal_solver.py and imported everywhere else (kernel.py does the same),
# rather than kept as separate local copies that can silently drift out of sync.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dark_portal_dir = os.path.abspath(os.path.join(_current_dir, "../../../src/"))
if _dark_portal_dir not in sys.path:
    sys.path.insert(0, _dark_portal_dir)
from dark_portal_solver import construct_PMNS, get_mass_eigenvalues

def load_and_grid_2d(filename):
    """
    Loads empirical 2D array coordinates and maps them onto a uniform, sorted grid
    for use in SciPy's RegularGridInterpolator.

    Parameters:
    - filename (str): Path to a 3-column text file (param1, param2, delta-chi2).

    Returns:
    - tuple: (unique param1 values, unique param2 values, structured chi2 grid)
             Returns (None, None, None) if the file cannot be accessed.
    """
    try: data = np.loadtxt(filename)
    except Exception as e:
        print(f"  [Prior Error] Could not load {filename}: {e}")
        return None, None, None
    
    p1, p2, chi2 = data[:, 0], data[:, 1], data[:, 2]
    u_p1, u_p2 = np.unique(p1), np.unique(p2)
    u_p1.sort(); u_p2.sort()
    
    grid_chi2 = np.full((len(u_p1), len(u_p2)), 9999.0)
    for v1, v2, c2 in zip(p1, p2, chi2):
        i, j = np.searchsorted(u_p1, v1), np.searchsorted(u_p2, v2)
        if i < len(u_p1) and j < len(u_p2): grid_chi2[i, j] = c2
    return u_p1, u_p2, grid_chi2

class PchipBoundsWrapper:
    """
    Picklable wrapper class for SciPy's PchipInterpolator to handle out-of-bounds 
    errors consistently without relying on lambda closures. This ensures safe 
    HPC multiprocessing serialization via pickle.
    """
    def __init__(self, pchip_func): 
        self.pchip = pchip_func
    
    def __call__(self, x):
        val = self.pchip(x)
        if np.isnan(val): raise ValueError("Out of bounds")
        return val

class GenericLikelihoodWrapper:
    """
    State-machine encapsulating parameter evaluation. Acts as a bridge transferring 
    dynamic variables mapped by Minuit/Emcee into the isolated Numba physics kernel
    and calculating the penalty via the external detector likelihoods.
    """
    def __init__(self, config, param_names, asimov=False):
        """
        Initializes the log-likelihood environment, cache management, and data structures.
        
        Parameters:
        - config (dict): Full dictionary parsed from the master JSON configuration.
        - param_names (list): Active floating parameter names assigned by the scanner.
        - asimov (bool): Flag to trigger mock data generation for sensitivity projections.
        """
        self.config = config
        self.param_names = param_names
        
        # Load the target likelihood module (e.g., IceCube MESE 2025)
        self.experiment = config.get('experiment', 'icecube_mese_2025')
        
        self.asimov = asimov
        self.fixed_params_cfg = config.get('fixed_params', {})
        self.use_fuzzy_cache = config.get("scan_settings", {}).get("use_fuzzy_cache", False)
        
        # ==========================================================
        # DIAGNOSTIC: Check for scanner disconnects
        # ==========================================================
        passed_params = set(self.param_names)
        
        if 'param1' in passed_params and len(passed_params) > 0:
            print("\n" + "!"*70)
            print(" [WARNING: SCANNER DISCONNECT DETECTED]")
            print(f" src.main provided these parameters to scan: {self.param_names}")
            print(" Because src.main isn't correctly identifying the scan parameters in")
            print(" your JSON, it is varying dummy variables (param1), doing nothing to")
            print(" the actual physics. The kernel will be forced to fallback to static")
            print(" 'start_values'. Please check how src.main extracts target parameters.")
            print("!"*70 + "\n")
        # ==========================================================
        
        # MCMC Memory Guard Toggle
        self._cache = {}
        self._cache_enabled = True

        # Ensure the tuple expects 26 elements now to allow dynamic spatial tracking overrides
        self.static_data = self.config.get("static_data_tuple", None)
        if self.static_data is None or len(self.static_data) != 26:
            try:
                # Direct read from the flat physics_constants dict in the active frequentist config
                pc = self.config.get("physics_constants", {})
                
                # Physical constants and conversion factors
                eV = 1.0
                m_to_eV_inv = 5.0677e6
                cm_inv3_to_eV3 = 1.0 / ((m_to_eV_inv * 1e-2)**3)
                Mpc_to_eV_inv = 3.08567758e22 * m_to_eV_inv
                km_per_s = 1.0 / 299792.458
                
                # Cosmology
                H0_eV = float((pc.get("H0_km_s_Mpc", 67.4) * km_per_s) / Mpc_to_eV_inv)
                Omega_m = float(pc.get("Omega_m", 0.315))
                Omega_L = float(pc.get("Omega_L", 0.685))
                T_nu0_eV = float(pc.get("T_nu0_eV", 0.000195))
                n_n0_eV3 = float(pc.get("n_n0_cm3", 2.5e-07) * cm_inv3_to_eV3)
                m_lightest_eV = float(pc.get("m_lightest_eV", 0.01))
                mass_ordering = pc.get("mass_ordering", "normal")
                
                # Formulated as a NumPy array for Numba compatibility
                eta_e = float(pc.get("eta_e", 1.0))
                eta_mu = float(pc.get("eta_mu", 1.0))
                eta_tau = float(pc.get("eta_tau", 1.0))
                eta_bg_arr = np.array([eta_e, eta_mu, eta_tau], dtype=np.float64)
                
                # Standard Mixing
                dm21_sq = float(pc.get("dm21_sq", 7.41e-05))
                dm31_sq = float(pc.get("dm31_sq", 0.002511))
                th12 = float(np.radians(pc.get("th12_deg", 33.41)))
                th13 = float(np.radians(pc.get("th13_deg", 8.58)))
                th23 = float(np.radians(pc.get("th23_deg", 49.1)))
                d_cp = float(np.radians(pc.get("d_cp_deg", 132.0)))
                detector_depth_km = float(pc.get("detector_depth_km", 1.5))
                
                # Source
                p_index = float(pc.get("p_index", 2.5))
                Phi_0 = float(pc.get("Phi_0", 1.0))
                z_max = float(pc.get("z_max", 4.0))
                rho_evol = float(pc.get("rho_evol", 0.0))
                f_e_S = float(pc.get("f_e_S", 0.333333))
                f_mu_S = float(pc.get("f_mu_S", 0.666667))
                f_tau_S = float(pc.get("f_tau_S", 0.0))
                is_antineutrino = pc.get("is_antineutrino", "sum")
                nubar_to_nu_ratio = float(pc.get("nubar_to_nu_ratio", 1.0))
                
                # Grid & Interpolation parameters
                E_min_eV = float(pc.get("E_min_eV", 1e13))
                E_max_eV = float(pc.get("E_max_eV", 3e16))
                n_E_bins = int(pc.get("n_E_bins", 1000))
                n_z_bins = int(pc.get("n_z_bins", 1000))
                
                # Extracted grid spatial tracking overrides
                n_mw_paths = int(pc.get("n_mw_paths", 6))
                n_earth_chords = int(pc.get("n_earth_chords", 3))
                N_steps_MW = int(pc.get("N_steps_MW", 100))
                r_start_kpc = float(pc.get("r_start_kpc", 100.0))
                
                # Computed structures
                E_bins = np.logspace(np.log10(E_min_eV), np.log10(E_max_eV), n_E_bins)
                z_bins = np.linspace(z_max, 0.0, n_z_bins)
                rho_src = np.array([f_e_S, f_mu_S, f_tau_S], dtype=np.float64)
                
                U_pmns = construct_PMNS(th12, th13, th23, d_cp)
                
                dm31_sq_signed = -dm31_sq if mass_ordering.lower().strip() == "inverted" else dm31_sq
                M_sq = np.ascontiguousarray(np.diag([0.0, dm21_sq, dm31_sq_signed]).astype(np.complex128))
                
                m_nu_bg_arr = np.ascontiguousarray(get_mass_eigenvalues(m_lightest_eV, dm21_sq, dm31_sq, mass_ordering), dtype=np.float64)
                
                # Construct expanded 26-element static_data tuple
                self.static_data = (
                    0.33,                           # 0. obs_f_e (not used)
                    0.33,                           # 1. obs_f_mu (not used)
                    0.05,                           # 2. obs_err_e (not used)
                    0.05,                           # 3. obs_err_mu (not used)
                    E_bins,                         # 4. E_bins
                    z_bins,                         # 5. z_bins
                    rho_src,                        # 6. rho_src
                    H0_eV,                          # 7. H0 (natural units)
                    Omega_m,                        # 8. Omega_m
                    Omega_L,                        # 9. Omega_L
                    U_pmns,                         # 10. U_pmns
                    M_sq,                           # 11. M_sq
                    m_nu_bg_arr,                    # 12. m_nu_bg_arr
                    n_n0_eV3,                       # 13. n_n0 (natural units)
                    T_nu0_eV,                       # 14. T_nu0
                    eta_bg_arr,                     # 15. eta_bg (Numpy Array)
                    detector_depth_km,              # 16. detector_depth_km
                    is_antineutrino,                # 17. is_antineutrino
                    p_index,                        # 18. p_index
                    rho_evol,                       # 19. rho_evol
                    Phi_0,                          # 20. Phi_0
                    nubar_to_nu_ratio,              # 21. nubar_to_nu_ratio
                    n_mw_paths,                     # 22. n_mw_paths
                    n_earth_chords,                 # 23. n_earth_chords
                    N_steps_MW,                     # 24. N_steps_MW
                    r_start_kpc                     # 25. r_start_kpc
                )
                print(f"  [Server] Successfully loaded physical constants from active config.")
            except Exception as e:
                print(f"  [Warning] Failed to parse physics config from self.config: {e}. Falling back to default mock static_data.")
                self.static_data = None
                    
        if self.static_data is None:
            print("  [Warning] Valid 26-element 'static_data_tuple' not found in config. Injecting mock tuple for initialization.")
            self.static_data = (
                0.33, 0.33, 0.05, 0.05,                         
                np.logspace(5, 8, 10), np.linspace(0, 4, 10),   
                np.ones(10),                                    
                67.4, 0.315, 0.685,                             
                np.eye(3, dtype=np.complex128),                 
                np.array([0., 7.4e-5, 2.51e-3]),                
                np.array([0., 0.0086, 0.0501]),                 
                112.0, 1.95e-4, np.ones(3, dtype=np.float64),   
                1.5, False, 2.0, 0.0, 1e-18,                    
                1.0, 6, 3, 100, 100.0                           
            )

        if self.asimov:
            print("  [Asimov] Mode activated. Generating mock expected data...")
            asimov_injections = self.config.get("asimov_injections", {})
            if asimov_injections:
                print(f"  [Asimov] Injecting null-hypothesis parameters: {asimov_injections}")
            
            # Construct dictionary state specifically for Asimov projection overrides
            mock_params = {}
            mock_params.update(self.config.get("physics_constants", {}))
            mock_params.update(self.config.get("start_values", {}))
            mock_params.update(self.fixed_params_cfg)
            mock_params.update(asimov_injections)
            
            # Print expected flavor baseline for the null hypothesis
            mock_f_e, mock_f_mu, mock_f_tau = compute_user_model_single(mock_params, self.static_data)
            print(f"  [Asimov] Mock expectation initialized at: e={mock_f_e:.3f}, mu={mock_f_mu:.3f}, tau={mock_f_tau:.3f}")
        
        self._build_prior_interpolators()
        
    def _get_val(self, vals, key):
        """
        Fetches dynamic floating values or static pinned values uniformly,
        strictly preventing 0.0 fallback crashes.
        """
        if key in vals: return vals[key]
        if key in self.fixed_params_cfg: return self.fixed_params_cfg[key]
            
        start_vals = self.config.get('start_values', {})
        if key in start_vals:
            return start_vals[key]
            
        return None
        
    def _build_prior_interpolators(self):
        """Automatically ingests empirical chi2 datafiles into active cubic splines."""
        self.prior_interpolators = []
        self.covered_params = set()
        input_configs = self.config.get('input_configs', [])
        bounds_error_setting = self.config.get('use_quadratic_fallback', False)
        use_linear_interp = self.config.get("scan_settings", {}).get("use_linear_interp", False)
        
        for entry in input_configs:
            fname, ftype, params = entry.get('file'), entry.get('type'), entry.get('params')
            units = entry.get('units', [])
            if not fname or not os.path.exists(fname): continue
                
            if ftype == '1D':
                try:
                    data = np.loadtxt(fname)
                    x_val, chi2_val = data[:, 0], data[:, 1]
                    if use_linear_interp:
                        fill_val = None if bounds_error_setting else "extrapolate"
                        interp = interp1d(x_val, chi2_val, kind='linear', bounds_error=bounds_error_setting, fill_value=fill_val)
                    else:
                        pchip = PchipInterpolator(x_val, chi2_val, extrapolate=not bounds_error_setting)
                        interp = PchipBoundsWrapper(pchip) if bounds_error_setting else pchip
                    self.prior_interpolators.append({'type': '1D', 'func': interp, 'params': params, 'units': units})
                    self.covered_params.add(params[0])
                except Exception as e: print(f"  [Prior Error] Loading 1D {fname}: {e}")

            elif ftype == '2D':
                try:
                    u_p1, u_p2, grid_chi2 = load_and_grid_2d(fname)
                    if u_p1 is None: continue
                    interp = RegularGridInterpolator((u_p1, u_p2), grid_chi2, bounds_error=bounds_error_setting, fill_value=None)
                    self.prior_interpolators.append({'type': '2D', 'func': interp, 'params': params, 'units': units})
                    self.covered_params.update(params)
                except Exception as e: print(f"  [Prior Error] Loading 2D {fname}: {e}")

    def _calculate_gaussian_fallback(self, param_name, val):
        """Computes quadratic penalties for parameters without empirical constraints."""
        fallbacks = self.config.get('fallbacks', {})
        if param_name in fallbacks:
            spec = fallbacks[param_name]
            if isinstance(spec, list) and len(spec) >= 3 and str(spec[0]).lower() == 'gauss':
                mu, sigma = spec[1], spec[2]
                if len(spec) > 3 and 'degree' in str(spec[3]).lower(): 
                    val_check = np.degrees(val)
                    diff = (val_check - mu) % 360.0
                    if diff > 180.0: diff -= 360.0
                    return (diff / sigma)**2
                else: 
                    return ((val - mu) / sigma)**2
        return 0.0

    def _compute_user_coupled_priors(self, vals):
        """User injection hook for mathematical multi-variable priors."""
        penalty = 0.0
        return penalty

    def _compute_prior_penalty(self, vals):
        """Aggregates active 1D/2D empirical constraints and fallback penalties."""
        penalty = 0.0
        for prior in self.prior_interpolators:
            p_vals = []
            for idx, p in enumerate(prior['params']):
                val = self._get_val(vals, p)
                if val is None: break
                
                # Apply localized unit conversions (e.g. log space or degrees)
                units = prior.get('units', [])
                if idx < len(units):
                    unit_str = str(units[idx]).lower()
                    if 'degree' in unit_str:
                        val = np.degrees(val)
                    elif 'log10' in unit_str:
                        val = np.log10(np.abs(val) + 1e-30)
                    elif '1e-3' in unit_str:
                        val = np.abs(val) * 1e3
                p_vals.append(val)
                
            if len(p_vals) != len(prior['params']): continue
            
            try:
                if prior['type'] == '1D': chi2_i = prior['func'](p_vals[0])
                elif prior['type'] == '2D': chi2_i = prior['func'](np.array([p_vals]))[0]
                if np.isnan(chi2_i): chi2_i = 0.0
                penalty += max(0.0, float(chi2_i)) 
            except ValueError: pass
                
        for p in self.param_names:
            if p not in self.covered_params:
                val = self._get_val(vals, p)
                if val is not None:
                    penalty += self._calculate_gaussian_fallback(p, val)
                    
        penalty += self._compute_user_coupled_priors(vals)
        return penalty

    def __call__(self, *args):
        """
        The objective function invoked directly by Minuit gradient algorithms and MCMC samplers.
        Evaluates the physics kernel and processes the external detector likelihood.
        """
        if self.use_fuzzy_cache: cache_key = tuple(round(x, 5) for x in args)
        else: cache_key = tuple(args)
            
        if self._cache_enabled and cache_key in self._cache: return self._cache[cache_key]
        vals = dict(zip(self.param_names, args))
        
        # Build comprehensive dictionary
        full_params = {}
        full_params.update(self.config.get("physics_constants", {}))
        full_params.update(self.config.get("start_values", {}))
        full_params.update(vals)
        full_params.update(self.fixed_params_cfg)
        
        for k in vals.keys():
            # Only sanitize if it's a numeric parameter Minuit is floating
            if k in full_params and isinstance(full_params[k], (float, int)):
                val = float(full_params[k])
                # Enforce floor (1e-35) if the parameter is too close to 0.0
                if abs(val) < 1e-35:
                    full_params[k] = 1e-35 * np.sign(val or 1.0)

        # =====================================================================
        # 1. TRANSLATION LAYER: Sine-Squared to Degrees
        # =====================================================================
        if "s12sq" in full_params: 
            val = np.clip(float(full_params["s12sq"]), 1e-10, 1.0)
            full_params["th12_deg"] = np.degrees(np.arcsin(np.sqrt(val)))
        if "s13sq" in full_params: 
            val = np.clip(float(full_params["s13sq"]), 1e-10, 1.0)
            full_params["th13_deg"] = np.degrees(np.arcsin(np.sqrt(val)))
        if "s23sq" in full_params: 
            val = np.clip(float(full_params["s23sq"]), 1e-10, 1.0)
            full_params["th23_deg"] = np.degrees(np.arcsin(np.sqrt(val)))

        # =====================================================================
        # 2. EXHAUSTIVE SANITIZATION
        # Force all critical physical constants to be strictly non-zero
        # =====================================================================
        safe_keys = ["m_Zd_eV", "g_D", "epsilon_prime", "dm21_sq", "dm31_sq", "m_lightest_eV"]
        for k in safe_keys:
            if k in full_params:
                # Force a tiny floor to prevent ZeroDivisionError
                full_params[k] = max(abs(float(full_params[k])), 1e-30) * np.sign(float(full_params[k]) or 1)

        # =====================================================================
        # 3. DYNAMIC MATRIX RECONSTRUCTION
        # =====================================================================
        osc_keys = {"s12sq", "s13sq", "s23sq", "th12_deg", "th13_deg", "th23_deg", "d_cp_deg", "dm21_sq", "dm31_sq", "mass_ordering", "m_lightest_eV"}
        
        if any(k in full_params for k in osc_keys) and self.static_data is not None:
            th12 = np.radians(float(full_params.get("th12_deg", 33.41)))
            th13 = np.radians(float(full_params.get("th13_deg", 8.58)))
            th23 = np.radians(float(full_params.get("th23_deg", 49.1)))
            d_cp = np.radians(float(full_params.get("d_cp_deg", 132.0)))
            new_U_pmns = construct_PMNS(th12, th13, th23, d_cp)
            
            dm21_sq = float(full_params.get("dm21_sq", 7.41e-05))
            dm31_sq = float(full_params.get("dm31_sq", 0.002511))
            ordering = str(full_params.get("mass_ordering", "normal"))
            m_lightest_eV = float(full_params.get("m_lightest_eV", 0.01))
            
            dm31_sq_signed = -dm31_sq if ordering.lower().strip() == "inverted" else dm31_sq
            new_M_sq = np.ascontiguousarray(np.diag([0.0, dm21_sq, dm31_sq_signed]).astype(np.complex128))
            new_m_nu_bg = np.ascontiguousarray(get_mass_eigenvalues(m_lightest_eV, dm21_sq, dm31_sq, ordering), dtype=np.float64)

            sd_list = list(self.static_data)
            sd_list[10] = new_U_pmns
            sd_list[11] = new_M_sq
            sd_list[12] = new_m_nu_bg
            current_static_data = tuple(sd_list)
        else:
            current_static_data = self.static_data

        try: 
            f_e, f_mu, f_tau = compute_user_model_single(full_params, current_static_data)
            fE = np.array([[f_e, f_mu, f_tau]])
            log_l = detector_flavor_log_likelihood_wrapper(fE, experiment=self.experiment)
            chi2_data = -2.0 * log_l[0] if isinstance(log_l, np.ndarray) else -2.0 * log_l
        except Exception as e:
            print(f"\n  [CRITICAL KERNEL CRASH] Evaluation failed! Reason: {e}")
            traceback.print_exc()
            return 1e9 
            
        if np.isnan(chi2_data) or np.isinf(chi2_data): chi2_data = 1e9

        chi2_prior = self._compute_prior_penalty(vals)
        total_chi2 = chi2_data + chi2_prior
        
        if self._cache_enabled: self._cache[cache_key] = total_chi2
        return total_chi2