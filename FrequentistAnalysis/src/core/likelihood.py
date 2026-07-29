"""
likelihood.py

Encapsulates the Minuit and Emcee objective function logic. Handles dynamic
evaluation caching and automated empirical prior spline construction.
"""

import os
import sys
import traceback
import multiprocessing
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d, PchipInterpolator

# Import internal physics and external detector likelihood module
from src.physics.kernel import compute_user_model_single
from src.core.detector_flavor_likelihood import detector_flavor_log_likelihood_wrapper

# construct_PMNS/get_mass_eigenvalues are the single canonical implementations, defined
# once in the dp_* solver modules and re-exported through the dark_portal_solver facade
# (kernel.py imports the same way), rather than kept as separate local copies that can
# silently drift out of sync.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dark_portal_dir = os.path.abspath(os.path.join(_current_dir, "../../../src/"))
if _dark_portal_dir not in sys.path:
    sys.path.insert(0, _dark_portal_dir)
from dark_portal_solver import (construct_PMNS, get_mass_eigenvalues, parse_structure_matrix_3x3,
                                cm_inv3_to_eV3, m_to_eV_inv, Mpc_to_eV_inv, km_per_s,
                                DEFAULT_CONFIG)

# Flat fallback map for physics_constants keys, sourced from the solver's canonical
# DEFAULT_CONFIG -- these fallbacks previously repeated the literals by hand (a
# synchronized-by-luck fourth copy of the defaults). Doubles as the known-key set
# for the unknown-key warning below.
_PC_DEFAULTS = {}
for _sec in ("cosmology", "standard_mixing", "source", "grid"):
    _PC_DEFAULTS.update(DEFAULT_CONFIG[_sec])
_PC_DEFAULTS["use_debye_screening"] = DEFAULT_CONFIG["dark_portal"]["use_debye_screening"]
_PC_DEFAULTS["U_LRI_struct"] = None
_PC_DEFAULTS["V_SM_struct"] = None


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

        # Cores available for the physics kernel to use *internally* (via propagate_spectrum's
        # own n_cores, distributing the Milky Way/Earth spatial handover) when a call is not
        # running inside main.py's scan/repair multiprocessing.Pool -- see kernel.py's
        # run_propagation, which is the only place that reads this back out of static_data.
        # Mirrors main.py's own num_cores derivation (scan_settings.num_cores, clamped to
        # actually-available cores) so the two don't drift apart.
        _actual_cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        _req_cores = int(config.get("scan_settings", {}).get("num_cores", _actual_cores))
        self.n_cores_standalone = max(1, min(_req_cores, _actual_cores))

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

        # A parameter listed in BOTH the floating set and fixed_params would be floated
        # by Minuit yet silently overridden by the fixed value inside every evaluation
        # (fixed_params is applied last in __call__): the fit would report a spurious
        # flat direction. main.py's validate_config catches the scan-target cases; this
        # catches every other constructor.
        _overlap = sorted(set(self.param_names) & set(self.fixed_params_cfg))
        if _overlap:
            print("\n" + "!"*70)
            print(" [WARNING] Parameters both floating AND in fixed_params:")
            print(f"   {_overlap}")
            print(" The fixed value silently wins inside every likelihood evaluation,")
            print(" so Minuit will see an exactly flat direction. Remove them from one")
            print(" of the two lists.")
            print("!"*70 + "\n")

        # NOTE: this wrapper used to keep a per-instance evaluation cache
        # (_cache/_cache_enabled/use_fuzzy_cache). Measured on real scans it hit only
        # 9.3% in-process and ~0% across processes (every pool task pickles a fresh
        # wrapper), while costing a dict insert per evaluation and an unbounded memory
        # footprint that the MCMC path had to special-case around. Removed outright; no
        # caller relies on repeat-evaluation identity (perform_sanity_check compares
        # against Minuit's stored fval, and the emcee wrapper re-evaluates by design).

        # Ensure the tuple expects 36 elements now (spatial tracking overrides plus
        # use_SM_potential/n_e_earth/use_debye_screening/n_cores_standalone/
        # V_SM_struct/U_LRI_struct/n_z_thermal/thermal_analytic/earth_step_scheme)
        self.static_data = self.config.get("static_data_tuple", None)
        if self.static_data is None or len(self.static_data) != 36:
            try:
                # Direct read from the flat physics_constants dict in the active frequentist config
                pc = self.config.get("physics_constants", {})
                _unknown = sorted(k for k in pc
                                  if not k.startswith("_comment") and k not in _PC_DEFAULTS)
                if _unknown:
                    print("\n" + "!"*70)
                    print(" [WARNING] Unrecognized physics_constants keys (typos silently do")
                    print(f" nothing -- check them): {_unknown}")
                    print("!"*70 + "\n")
                
                # Physical constants and unit conversions come from dark_portal_solver
                # (module-level imports above) -- previously redefined locally here with
                # identical values, a silent-drift risk of the same class as the
                # construct_PMNS duplication that was centralized earlier.

                # Cosmology
                H0_eV = float((pc.get("H0_km_s_Mpc", _PC_DEFAULTS["H0_km_s_Mpc"]) * km_per_s) / Mpc_to_eV_inv)
                Omega_m = float(pc.get("Omega_m", _PC_DEFAULTS["Omega_m"]))
                Omega_L = float(pc.get("Omega_L", _PC_DEFAULTS["Omega_L"]))
                T_nu0_eV = float(pc.get("T_nu0_eV", _PC_DEFAULTS["T_nu0_eV"]))
                n_n0_eV3 = float(pc.get("n_n0_cm3", _PC_DEFAULTS["n_n0_cm3"]) * cm_inv3_to_eV3)
                m_lightest_eV = float(pc.get("m_lightest_eV", _PC_DEFAULTS["m_lightest_eV"]))
                mass_ordering = pc.get("mass_ordering", _PC_DEFAULTS["mass_ordering"])
                
                # Formulated as a NumPy array for Numba compatibility
                eta_e = float(pc.get("eta_e", _PC_DEFAULTS["eta_e"]))
                eta_mu = float(pc.get("eta_mu", _PC_DEFAULTS["eta_mu"]))
                eta_tau = float(pc.get("eta_tau", _PC_DEFAULTS["eta_tau"]))
                eta_bg_arr = np.array([eta_e, eta_mu, eta_tau], dtype=np.float64)
                
                # Standard Mixing
                dm21_sq = float(pc.get("dm21_sq", _PC_DEFAULTS["dm21_sq"]))
                dm31_sq = float(pc.get("dm31_sq", _PC_DEFAULTS["dm31_sq"]))
                th12 = float(np.radians(pc.get("th12_deg", _PC_DEFAULTS["th12_deg"])))
                th13 = float(np.radians(pc.get("th13_deg", _PC_DEFAULTS["th13_deg"])))
                th23 = float(np.radians(pc.get("th23_deg", _PC_DEFAULTS["th23_deg"])))
                d_cp = float(np.radians(pc.get("d_cp_deg", _PC_DEFAULTS["d_cp_deg"])))
                detector_depth_km = float(pc.get("detector_depth_km", _PC_DEFAULTS["detector_depth_km"]))

                # Standard-model MSW potential and Debye screening toggle: these are present
                # in every shipped physics_constants block but were never actually read here
                # -- kernel.py hardcoded use_SM_potential=False, n_e_earth=0.0,
                # use_debye_screening=True regardless of what the config said. Currently
                # dormant since every shipped config wants use_SM_potential=false anyway, but
                # silently dropped a config value exactly like the (already-fixed)
                # n_e_earth_cm3 disconnect in dark_portal_solver.py's own run_solver.
                use_SM_potential = bool(pc.get("use_SM_potential", _PC_DEFAULTS["use_SM_potential"]))
                n_e_earth_eV3 = float(pc.get("n_e_earth_cm3", _PC_DEFAULTS["n_e_earth_cm3"]) * cm_inv3_to_eV3)
                use_debye_screening = bool(pc.get("use_debye_screening", _PC_DEFAULTS["use_debye_screening"]))

                # Optional flavor-structure overrides for the coherent forward-scattering
                # potentials (neutrino convention; propagate_spectrum conjugates for
                # antineutrinos centrally). None means the defaults: diag(1,0,0) for the
                # SM potential, outer(u, conj(u)) for the LRI potential. NOTE: an
                # U_LRI_struct override freezes the potential's flavor texture -- if
                # u_e/u_mu/u_tau are simultaneously being scanned, they then affect only
                # the attenuation/regeneration rates and Debye mass, not the potential;
                # a warning is printed for that combination below.
                V_SM_struct = None
                if pc.get("V_SM_struct", _PC_DEFAULTS["V_SM_struct"]) is not None:
                    V_SM_struct = parse_structure_matrix_3x3(pc["V_SM_struct"], "physics_constants.V_SM_struct")
                U_LRI_struct = None
                if pc.get("U_LRI_struct", _PC_DEFAULTS["U_LRI_struct"]) is not None:
                    U_LRI_struct = parse_structure_matrix_3x3(pc["U_LRI_struct"], "physics_constants.U_LRI_struct")
                    scanned_u = [p for p in self.param_names if p in ("u_e", "u_mu", "u_tau")]
                    if scanned_u:
                        print("\n" + "!"*70)
                        print(" [WARNING] physics_constants.U_LRI_struct override is set, but the")
                        print(f" sterile mixing parameters {scanned_u} are being scanned. With the")
                        print(" override active, those parameters affect ONLY the thermal")
                        print(" attenuation/regeneration rates and the Debye screening mass --")
                        print(" NOT the coherent LRI potential's flavor structure.")
                        print("!"*70 + "\n")

                # Source
                p_index = float(pc.get("p_index", _PC_DEFAULTS["p_index"]))
                Phi_0 = float(pc.get("Phi_0", _PC_DEFAULTS["Phi_0"]))
                z_max = float(pc.get("z_max", _PC_DEFAULTS["z_max"]))
                rho_evol = float(pc.get("rho_evol", _PC_DEFAULTS["rho_evol"]))
                f_e_S = float(pc.get("f_e_S", _PC_DEFAULTS["f_e_S"]))
                f_mu_S = float(pc.get("f_mu_S", _PC_DEFAULTS["f_mu_S"]))
                f_tau_S = float(pc.get("f_tau_S", _PC_DEFAULTS["f_tau_S"]))
                is_antineutrino = pc.get("is_antineutrino", _PC_DEFAULTS["is_antineutrino"])
                nubar_to_nu_ratio = float(pc.get("nubar_to_nu_ratio", _PC_DEFAULTS["nubar_to_nu_ratio"]))
                
                # Grid & Interpolation parameters
                E_min_eV = float(pc.get("E_min_eV", _PC_DEFAULTS["E_min_eV"]))
                E_max_eV = float(pc.get("E_max_eV", _PC_DEFAULTS["E_max_eV"]))
                n_E_bins = int(pc.get("n_E_bins", _PC_DEFAULTS["n_E_bins"]))
                n_z_bins = int(pc.get("n_z_bins", _PC_DEFAULTS["n_z_bins"]))
                
                # Extracted grid spatial tracking overrides. Fallback defaults follow the
                # production convergence study (output/convergence/): 12 chords for
                # percent-level flavor-ratio accuracy (3 carried 1.6-4.9e-2 absolute
                # error), 25 MW steps (converged to ~1e-10; 100 was pure surplus).
                n_mw_paths = int(pc.get("n_mw_paths", _PC_DEFAULTS["n_mw_paths"]))
                n_earth_chords = int(pc.get("n_earth_chords", _PC_DEFAULTS["n_earth_chords"]))
                N_steps_MW = int(pc.get("N_steps_MW", _PC_DEFAULTS["N_steps_MW"]))
                # N_steps_earth defaults to the solver's DEFAULT_CONFIG value (25 under
                # the 'avg' step scheme, whose results are certified step-count-
                # independent; the historical 'left' scheme needed the pinned 500 --
                # see propagate_earth_spatial's docstring in dp_spatial.py).
                N_steps_earth = int(pc.get("N_steps_earth", _PC_DEFAULTS["N_steps_earth"]))
                r_start_kpc = float(pc.get("r_start_kpc", _PC_DEFAULTS["r_start_kpc"]))
                # Coarse z-sampling of the thermal-rate grid (0 = exact grid; see
                # dp_thermal.build_thermal_grid, re-exported by dark_portal_solver).
                n_z_thermal = int(pc.get("n_z_thermal", _PC_DEFAULTS["n_z_thermal"]))
                # Semi-analytic momentum integration of the thermal rates (default
                # True; False = midpoint reference kernel).
                thermal_analytic = bool(pc.get("thermal_analytic", _PC_DEFAULTS["thermal_analytic"]))
                # Earth-transit potential discretization (see dark_portal_solver's
                # _earth_chord_worker: 'avg' = exact step-averaged, step-count-
                # independent LRI phase; 'left' = historical scheme).
                earth_step_scheme = str(pc.get("earth_step_scheme", _PC_DEFAULTS["earth_step_scheme"]))
                
                # Computed structures
                E_bins = np.logspace(np.log10(E_min_eV), np.log10(E_max_eV), n_E_bins)
                z_bins = np.linspace(z_max, 0.0, n_z_bins)
                rho_src = np.array([f_e_S, f_mu_S, f_tau_S], dtype=np.float64)
                
                U_pmns = construct_PMNS(th12, th13, th23, d_cp)
                
                dm31_sq_signed = -dm31_sq if mass_ordering.lower().strip() == "inverted" else dm31_sq
                M_sq = np.ascontiguousarray(np.diag([0.0, dm21_sq, dm31_sq_signed]).astype(np.complex128))
                
                m_nu_bg_arr = np.ascontiguousarray(get_mass_eigenvalues(m_lightest_eV, dm21_sq, dm31_sq, mass_ordering), dtype=np.float64)
                
                # Construct expanded 36-element static_data tuple
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
                    r_start_kpc,                    # 25. r_start_kpc
                    N_steps_earth,                  # 26. N_steps_earth
                    use_SM_potential,               # 27. use_SM_potential
                    n_e_earth_eV3,                  # 28. n_e_earth (natural units)
                    use_debye_screening,             # 29. use_debye_screening
                    self.n_cores_standalone,          # 30. n_cores_standalone
                    V_SM_struct,                     # 31. V_SM_struct (None = default diag(1,0,0))
                    U_LRI_struct,                    # 32. U_LRI_struct (None = outer(u, conj(u)))
                    n_z_thermal,                     # 33. n_z_thermal (z-interp; 0 = off/exact)
                    thermal_analytic,                # 34. thermal_analytic (semi-analytic p-integral)
                    earth_step_scheme                # 35. earth_step_scheme ('avg' | 'left')
                )
                print(f"  [Server] Successfully loaded physical constants from active config.")
            except Exception as e:
                print(f"  [Warning] Failed to parse physics config from self.config: {e}. Falling back to default mock static_data.")
                self.static_data = None

        if self.static_data is None:
            print("  [Warning] Valid 36-element 'static_data_tuple' not found in config. Injecting mock tuple for initialization.")
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
                1.0, 6, 3, 100, 100.0, 25,
                False, 1.5e24 * cm_inv3_to_eV3, True,
                self.n_cores_standalone,
                None, None, 0, True, "avg"
            )

        # PHASE-NOISE DOMAIN ADVISORY (from the task-#3 diagnosis in
        # dp_propagator.propagator_3x3): below m_Zd ~ sqrt(3e6 * g_D * eps') eV the
        # 1/m_Zd^2-enhanced Earth potential accrues more phase per spatial step than
        # double precision can represent. Since the propagator's phase-noise guard the
        # solver returns finite, unitary fluxes there -- but they are decoherence-
        # averaged only (oscillation phases are machine noise; historically this domain
        # produced all-NaN fluxes -> the 1e9 sentinel). Warn once at startup when the
        # configured scan window overlaps that domain, using the largest coupling
        # product the scan can reach (upper limits for floating parameters).
        _plim = self.config.get("parameter_limits", {})

        def _coupling_reach(name, default):
            if name in self.param_names:
                lim = _plim.get(name)
                if lim and len(lim) == 2 and lim[1] is not None:
                    return abs(float(lim[1]))
            v = self.fixed_params_cfg.get(name, self.config.get("start_values", {}).get(name, default))
            try:
                return abs(float(v))
            except (TypeError, ValueError):
                return default

        _mzd_lim = _plim.get("m_Zd_eV")
        if "m_Zd_eV" in self.param_names and _mzd_lim and len(_mzd_lim) == 2 and _mzd_lim[0] is not None:
            _m_bad = (3e6 * _coupling_reach("g_D", 1e-4) * _coupling_reach("epsilon_prime", 0.1)) ** 0.5
            if float(_mzd_lim[0]) < _m_bad:
                print("\n" + "!"*70)
                print(" [WARNING] The m_Zd_eV scan window overlaps the phase-noise domain:")
                print(f"   lower limit {float(_mzd_lim[0]):.3g} eV < ~{_m_bad:.3g} eV"
                      f" (= sqrt(3e6 * g_D * epsilon_prime) at the scan's largest couplings).")
                print(" Below that scale the 1/m_Zd^2-enhanced Earth potential exceeds what")
                print(" double precision can phase-track: fluxes are finite and unitary (the")
                print(" propagator's phase-noise guard) but DECOHERENCE-AVERAGED ONLY --")
                print(" oscillation phases carry no information there. Narrow the window,")
                print(" or interpret those rows as decoherence-limit physics.")
                print("!"*70 + "\n")

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
        return chi2_data + chi2_prior