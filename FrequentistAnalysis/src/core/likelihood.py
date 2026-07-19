"""
likelihood.py

Encapsulates the Minuit and Emcee objective function logic. Handles dynamic
evaluation caching and automated empirical prior spline construction.
"""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, PchipInterpolator

# Import internal physics and external detector likelihood module
from src.physics.kernel import compute_user_model_single
from detector_flavor_likelihood import detector_flavor_log_likelihood_wrapper

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
        
        # EXACT variable order the dark portal Numba math kernel requires.
        self.all_params = ['m_Zd_eV', 'g_D', 'epsilon_prime', 'u_e', 'u_mu', 'u_tau']
        
        self.fixed_params_cfg = config.get('fixed_params', {})
        self.use_fuzzy_cache = config.get("scan_settings", {}).get("use_fuzzy_cache", False)
        
        # MCMC Memory Guard Toggle
        self._cache = {}
        self._cache_enabled = True

        # Fetch specifically constructed 22-element tuple from main script configuration
        self.static_data = self.config.get("static_data_tuple", None)
        if self.static_data is None or len(self.static_data) != 22:
            print("  [Warning] Valid 22-element 'static_data_tuple' not found in config. Injecting mock tuple for initialization.")
            self.static_data = (
                0.33, 0.33, 0.05, 0.05,                         
                np.logspace(5, 8, 10), np.linspace(0, 4, 10),   
                np.ones(10),                                    
                67.4, 0.315, 0.685,                             
                np.eye(3, dtype=np.complex128),                 
                np.array([0., 7.4e-5, 2.51e-3]),                
                np.array([0., 0.0086, 0.0501]),                 
                112.0, 1.95e-4, 6e-10,                          
                1.5, False, 2.0, 0.0, 1e-18,                    
                1.0                                             
            )
        
        if self.asimov:
            print("  [Asimov] Mode activated. Generating mock expected data...")
            asimov_injections = self.config.get("asimov_injections", {})
            if asimov_injections:
                print(f"  [Asimov] Injecting null-hypothesis parameters: {asimov_injections}")
            
            mock_args = [asimov_injections.get(p, self.fixed_params_cfg.get(p, 0.0)) for p in self.all_params]
            
            # Print expected flavor baseline for the null hypothesis
            mock_f_e, mock_f_mu, mock_f_tau = compute_user_model_single(np.array(mock_args, dtype=np.float64), self.static_data)
            print(f"  [Asimov] Mock expectation initialized at: e={mock_f_e:.3f}, mu={mock_f_mu:.3f}, tau={mock_f_tau:.3f}")
        
        self._build_prior_interpolators()
        
    def _get_val(self, vals, key):
        """
        Fetches dynamic floating values or static pinned values uniformly.
        
        Parameters:
        - vals (dict): Current parameter coordinates provided by Minuit.
        - key (str): Parameter name to lookup.
        
        Returns:
        - float: The parameter value, or None if completely unassigned.
        """
        if key in vals: return vals[key]
        return self.fixed_params_cfg.get(key)
        
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
        """
        Computes quadratic penalties for parameters without empirical constraints.
        
        Parameters:
        - param_name (str): Target parameter name.
        - val (float): Current evaluated parameter value.
        
        Returns:
        - float: Calculated penalty mapped as a Chi-Square addition.
        """
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
        
        Parameters:
        - args: Dynamic floating point coordinates passed continuously by the Minuit/Emcee optimizer.
        
        Returns:
        - float: The total calculated frequentist Chi-Square penalty.
        """
        if self.use_fuzzy_cache: cache_key = tuple(round(x, 5) for x in args)
        else: cache_key = tuple(args)
            
        # Conditional Memory Check
        if self._cache_enabled and cache_key in self._cache: return self._cache[cache_key]
        vals = dict(zip(self.param_names, args))
        
        # Unpack floating coordinates in strict order sequence expected by compiler.
        model_args = [self._get_val(vals, p) for p in self.all_params]
        p_arr = np.array([val if val is not None else 0.0 for val in model_args], dtype=np.float64)
        
        try: 
            # 1. Obtain raw flavor fractions from the physics kernel
            f_e, f_mu, f_tau = compute_user_model_single(p_arr, self.static_data)
            
            # 2. Package into the specific 2D array structure the detector wrapper requires
            fE = np.array([[f_e, f_mu, f_tau]])
            
            # 3. Retrieve the Log-Likelihood and convert it into a frequentist Chi-Square
            log_l = detector_flavor_log_likelihood_wrapper(fE, experiment=self.experiment)
            
            # Unpack array scalar safely depending on the detector module return signature
            chi2_data = -2.0 * log_l[0] if isinstance(log_l, np.ndarray) else -2.0 * log_l
        except Exception: 
            return 1e9 
            
        # Safely reject unphysical domain cliffs
        if np.isnan(chi2_data) or np.isinf(chi2_data): chi2_data = 1e9

        chi2_prior = self._compute_prior_penalty(vals)
        if np.isnan(chi2_prior) or np.isinf(chi2_prior): chi2_prior = 1e9 if chi2_prior > 0 else 0.0

        total_chi2 = chi2_data + chi2_prior
        if np.isnan(total_chi2) or np.isinf(total_chi2): total_chi2 = 1e9
        
        # Save evaluation to tracking dictionary only if toggle allows it.
        if self._cache_enabled: self._cache[cache_key] = total_chi2
        return total_chi2