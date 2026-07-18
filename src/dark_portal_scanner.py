import os
import copy
import json
import multiprocessing
import numpy as np
from typing import Dict, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: 'matplotlib' not found. Plotting will be disabled.")

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: 'numba' not found. Falling back to pure Python (will be significantly slower).")
    def njit(*args, **kwargs):
        """Dummy decorator if numba is unavailable or string signatures are passed."""
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        if len(args) == 1 and isinstance(args[0], str): return decorator
        return decorator

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: 'tqdm' not found. Progress bars will be disabled.")
    def tqdm(iterable, *args, **kwargs): return iterable

# Import the core physics solver from the adjacent module
try:
    import dark_portal_solver as dps
    dps.HAS_TQDM = False  # Suppresses all internal progress bars from the solver
except ImportError:
    raise ImportError("Could not find 'dark_portal_solver.py'. Ensure this script is in the same directory.")

@njit("float64[:](float64[:], float64[:,:])", fastmath=True, cache=True)
def compute_energy_averaged_ratios(E_bins: np.ndarray, Flux_array: np.ndarray) -> np.ndarray:
    """
    Computes the energy-averaged flavor ratios over the fully integrated spectrum.
    Utilizes exact geometric midpoints to perfectly match the cascading energy conservation weights.
    """
    N_E = len(E_bins)
    dE = np.zeros(N_E, dtype=np.float64)
    if N_E > 1:
        bin_ratio = E_bins[1] / E_bins[0]
        half_ratio = np.sqrt(bin_ratio)
        for i in range(N_E):
            E_lower = E_bins[i] / half_ratio
            E_upper = E_bins[i] * half_ratio
            dE[i] = E_upper - E_lower
    elif N_E == 1:
        dE[0] = 0.0

    flux_e = Flux_array[:, 0]
    flux_mu = Flux_array[:, 1]
    flux_tau = Flux_array[:, 2]
    
    int_e = np.sum(flux_e * dE)
    int_mu = np.sum(flux_mu * dE)
    int_tau = np.sum(flux_tau * dE)
    int_total = int_e + int_mu + int_tau
    
    # Fallback to standard 1:1:1 if the entire spectrum is catastrophically attenuated
    if int_total > 1e-100:
        return np.array([int_e / int_total, int_mu / int_total, int_tau / int_total])
    else:
        return np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])

def scan_worker(args: Tuple[str, float, Dict]) -> Tuple[float, np.ndarray]:
    """
    Multiprocessing worker to execute a single point in the parameter scan.
    Suppresses internal solver verbosity to maintain a clean top-level progress bar.
    """
    param_name, param_val, base_config = args
    
    # Isolate the config specifically for this worker
    worker_config = copy.deepcopy(base_config)
    worker_config['dark_portal'][param_name] = param_val
    worker_config['execution']['n_cores'] = 1
    worker_config['execution']['verbose'] = 0
    
    # Suppress internal stdout from the solver
    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        E_bins, Flux_array = dps.run_solver(worker_config)
    finally:
        sys.stdout = old_stdout
        
    ratios = compute_energy_averaged_ratios(E_bins, Flux_array)
    return param_val, ratios

def run_parameter_scan(param_name: str, param_values: np.ndarray, base_config: Dict, output_file: str):
    """
    Orchestrates a parallel 1D parameter scan across the provided grid values.
    """
    n_cores = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
    
    tasks = [(param_name, val, base_config) for val in param_values]
    results_val = []
    results_ratios = []

    if HAS_TQDM and hasattr(tqdm, 'write'):
        tqdm.write(f"\n--- Scanning {param_name} across {len(param_values)} grid points ---")
    else:
        print(f"\n--- Scanning {param_name} across {len(param_values)} grid points ---")

    if n_cores > 1:
        with multiprocessing.Pool(processes=n_cores) as pool:
            if HAS_TQDM:
                for res in tqdm(pool.imap(scan_worker, tasks), total=len(tasks), desc=f"Optimizing {param_name}"):
                    results_val.append(res[0])
                    results_ratios.append(res[1])
            else:
                for res in pool.imap(scan_worker, tasks):
                    results_val.append(res[0])
                    results_ratios.append(res[1])
    else:
        for task in (tqdm(tasks, desc=f"Optimizing {param_name}") if HAS_TQDM else tasks):
            res = scan_worker(task)
            results_val.append(res[0])
            results_ratios.append(res[1])
            
    # Sort results mathematically to guarantee cleanly ordered output files
    results_val = np.array(results_val)
    results_ratios = np.array(results_ratios)
    sort_idx = np.argsort(results_val)
    
    sorted_vals = results_val[sort_idx]
    sorted_ratios = results_ratios[sort_idx]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    header = f"{param_name}    f_e    f_mu    f_tau"
    data = np.column_stack((sorted_vals, sorted_ratios))
    np.savetxt(output_file, data, header=header, fmt="%.6e")
    
    if HAS_TQDM and hasattr(tqdm, 'write'):
        tqdm.write(f"Scan data successfully saved to: {os.path.abspath(output_file)}")
    else:
        print(f"Scan data successfully saved to: {os.path.abspath(output_file)}")

def plot_parameter_scans(m_Zd_file: str, g_D_file: str, u_e_file: str, u_mu_file: str, base_config: Dict, output_filename: str = "../output/plots/parameter_scan_plot.png"):
    """
    Parses the exported multi-dimensional scan files and generates a 2x2 multi-panel visualization.
    Includes mass, gauge coupling, and sterile mixing variations.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib is disabled. Skipping plot generation.")
        return

    # Extract baseline parameters for text labels
    dp_conf = base_config.get('dark_portal', {})
    u_e_base = dp_conf.get('u_e', 0.0)
    u_mu_base = dp_conf.get('u_mu', 0.0)
    g_D_base = dp_conf.get('g_D', 0.0)
    m_Zd_base = dp_conf.get('m_Zd_eV', 0.0)
    eps_prime_base = dp_conf.get('epsilon_prime', 0.0)

    # Load data from the exported text arrays
    m_Zd_data = np.loadtxt(m_Zd_file)
    g_D_data = np.loadtxt(g_D_file)
    u_e_data = np.loadtxt(u_e_file)
    u_mu_data = np.loadtxt(u_mu_file)

    m_Zd_vals, m_Zd_fe, m_Zd_fmu, m_Zd_ftau = m_Zd_data[:, 0], m_Zd_data[:, 1], m_Zd_data[:, 2], m_Zd_data[:, 3]
    g_D_vals, g_D_fe, g_D_fmu, g_D_ftau = g_D_data[:, 0], g_D_data[:, 1], g_D_data[:, 2], g_D_data[:, 3]
    u_e_vals, u_e_fe, u_e_fmu, u_e_ftau = u_e_data[:, 0], u_e_data[:, 1], u_e_data[:, 2], u_e_data[:, 3]
    u_mu_vals, u_mu_fe, u_mu_fmu, u_mu_ftau = u_mu_data[:, 0], u_mu_data[:, 1], u_mu_data[:, 2], u_mu_data[:, 3]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharey=True)
    
    # --- Top Left Panel: Mass Scan ---
    axes[0, 0].plot(m_Zd_vals, m_Zd_fe, color='crimson', linestyle='--', lw=2.5, label=r'$\langle f_e \rangle$')
    axes[0, 0].plot(m_Zd_vals, m_Zd_fmu, color='royalblue', linestyle='-.', lw=2.5, label=r'$\langle f_\mu \rangle$')
    axes[0, 0].plot(m_Zd_vals, m_Zd_ftau, color='forestgreen', linestyle=':', lw=2.5, label=r'$\langle f_\tau \rangle$')
    
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlim(min(m_Zd_vals), max(m_Zd_vals))
    axes[0, 0].set_ylim(0, 0.5)
    axes[0, 0].set_xlabel(r'Dark mediator mass, $m_{Z_D}$ [eV]', fontsize=14)
    axes[0, 0].set_ylabel('Energy-averaged flavor ratio', fontsize=14)
    axes[0, 0].set_title(r'Flavor composition vs. $m_{Z_D}$', fontsize=15, pad=12)
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0, 0].legend(fontsize=12, loc='upper right')
    text_tl = f"$u_e = {u_e_base}$\n$u_\\mu = {u_mu_base}$\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    axes[0, 0].text(0.05, 0.05, text_tl, transform=axes[0, 0].transAxes, fontsize=11, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # --- Top Right Panel: Gauge Coupling Scan ---
    axes[0, 1].plot(g_D_vals, g_D_fe, color='crimson', linestyle='--', lw=2.5)
    axes[0, 1].plot(g_D_vals, g_D_fmu, color='royalblue', linestyle='-.', lw=2.5)
    axes[0, 1].plot(g_D_vals, g_D_ftau, color='forestgreen', linestyle=':', lw=2.5)
    
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlim(min(g_D_vals), max(g_D_vals))
    axes[0, 1].set_xlabel(r'Dark gauge coupling, $g_D$', fontsize=14)
    axes[0, 1].set_title(r'Flavor composition vs. $g_D$', fontsize=15, pad=12)
    axes[0, 1].grid(True, which="both", ls="--", alpha=0.4)
    text_tr = f"$u_e = {u_e_base}$\n$u_\\mu = {u_mu_base}$\n$m_{{Z_D}} = {m_Zd_base}$ eV\n$\\epsilon' = {eps_prime_base}$"
    axes[0, 1].text(0.05, 0.05, text_tr, transform=axes[0, 1].transAxes, fontsize=11, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # --- Bottom Left Panel: u_e Mixing Scan ---
    axes[1, 0].plot(u_e_vals, u_e_fe, color='crimson', linestyle='--', lw=2.5)
    axes[1, 0].plot(u_e_vals, u_e_fmu, color='royalblue', linestyle='-.', lw=2.5)
    axes[1, 0].plot(u_e_vals, u_e_ftau, color='forestgreen', linestyle=':', lw=2.5)
    
    axes[1, 0].set_xlim(min(u_e_vals), max(u_e_vals))
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_xlabel(r'Sterile electron mixing, $u_e$', fontsize=14)
    axes[1, 0].set_ylabel('Energy-averaged flavor ratio', fontsize=14)
    axes[1, 0].set_title(r'Flavor composition vs. $u_e$', fontsize=15, pad=12)
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.4)
    # Add parameter text label (Hardcoding u_mu to 0.0 to match the scan logic)
    # text_bl = f"$u_\\mu = {u_mu_base}$\n$m_{{Z_D}} = {m_Zd_base}$ eV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    text_bl = f"$u_\\mu = 0.0$\n$m_{{Z_D}} = {m_Zd_base}$ eV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    axes[1, 0].text(0.05, 0.05, text_bl, transform=axes[1, 0].transAxes, fontsize=11, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # --- Bottom Right Panel: u_mu Mixing Scan ---
    axes[1, 1].plot(u_mu_vals, u_mu_fe, color='crimson', linestyle='--', lw=2.5)
    axes[1, 1].plot(u_mu_vals, u_mu_fmu, color='royalblue', linestyle='-.', lw=2.5)
    axes[1, 1].plot(u_mu_vals, u_mu_ftau, color='forestgreen', linestyle=':', lw=2.5)
    
    axes[1, 1].set_xlim(min(u_mu_vals), max(u_mu_vals))
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_xlabel(r'Sterile muon mixing, $u_\mu$', fontsize=14)
    axes[1, 1].set_title(r'Flavor composition vs. $u_\mu$', fontsize=15, pad=12)
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.4)
    # Add parameter text label (Hardcoding u_e to 0.0 to match the scan logic)
    # text_br = f"$u_e = {u_e_base}$\n$m_{{Z_D}} = {m_Zd_base}$ eV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    text_br = f"$u_e = 0.0$\n$m_{{Z_D}} = {m_Zd_base}$ eV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    axes[1, 1].text(0.05, 0.05, text_br, transform=axes[1, 1].transAxes, fontsize=11, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    plt.savefig(output_filename, dpi=300)
    print(f"\nScan plot successfully saved to: {os.path.abspath(output_filename)}")
    plt.show()

if __name__ == "__main__":
    # Generate the baseline configuration using the solver's native builder
    config_file = "../config/config_dark_portal.json"
    if not os.path.exists(config_file):
        base_config = dps.create_default_config(config_file)
    else:
        with open(config_file, 'r') as f:
            base_config = json.load(f)

    # Compile Numba kernels in advance using a lightweight dummy execution.
    # This prevents all independent multiprocessing workers from attempting to JIT-compile 
    # simultaneously on the first iteration, which risks bottlenecking the LLVM cache.
    # print("Initiating Numba compiler cache warm-up...")
    # warmup_config = copy.deepcopy(base_config)
    # warmup_config['grid']['n_E_bins'] = 200
    # warmup_config['grid']['n_z_bins'] = 200
    # warmup_config['grid']['n_mw_paths'] = 100
    # warmup_config['grid']['n_earth_chords'] = 100
    # warmup_config['execution']['n_cores'] = 11
    # warmup_config['execution']['verbose'] = 0
    
    # import sys, io
    # old_stdout = sys.stdout
    # sys.stdout = io.StringIO()
    # try:
    #     _ = dps.run_solver(warmup_config)
    #     # Warm up the newly decorated ratio compiler
    #     dummy_E = np.logspace(13, 16, 200)
    #     dummy_Flux = np.zeros((200, 3))
    #     _ = compute_energy_averaged_ratios(dummy_E, dummy_Flux)
    # finally:
    #     sys.stdout = old_stdout
    # print("Warm-up complete. Numba kernels mapped to physical disk cache.")

    # Lowering the integration grid resolution to speed up 1D parameter scans
    base_config['grid']['n_E_bins'] = 200
    base_config['grid']['n_z_bins'] = 200
    base_config['grid']['n_mw_paths'] = 100
    base_config['grid']['n_earth_chords'] = 100
    base_config["grid"]['N_steps_MW'] = 100
    base_config["grid"]['r_start_kpc'] = 100.0
    base_config['execution']['n_cores'] = 11

    # Define high-resolution evaluation grids
    m_Zd_range = np.logspace(-10, 10, 100)      # Mediator mass scan from 0.1 neV to 10 GeV
    g_D_range = np.logspace(-10, -1, 100)       # Gauge coupling scan from 10^-10 to 10^-1
    u_e_range = np.logspace(-5, 0, 100)         # Sterile electron mixing evaluation space (log scale)
    u_mu_range = np.logspace(-5, 0, 100)        # Sterile muon mixing evaluation space (log scale)
    
    # Establish output directory paths
    m_Zd_output = "../output/data/scan_m_Zd.txt"
    g_D_output = "../output/data/scan_g_D.txt"
    u_e_output = "../output/data/scan_u_e.txt"
    u_mu_output = "../output/data/scan_u_mu.txt"
    plot_output = "../output/plots/parameter_scan_plot.png"

    # Execute decoupled configuration mappings
    # run_parameter_scan('m_Zd_eV', m_Zd_range, base_config, m_Zd_output)
    # run_parameter_scan('g_D', g_D_range, base_config, g_D_output)
    u_e_base  = base_config['dark_portal']['u_e']
    u_mu_base  = base_config['dark_portal']['u_mu']
    base_config['dark_portal']['u_mu'] = 0.0
    run_parameter_scan('u_e', u_e_range, base_config, u_e_output)
    base_config['dark_portal']['u_mu'] = u_mu_base
    base_config['dark_portal']['u_e'] = 0.0
    run_parameter_scan('u_mu', u_mu_range, base_config, u_mu_output)
    base_config['dark_portal']['u_e'] = u_e_base

    # Invoke 2x2 dual-panel Matplotlib rendering
    plot_parameter_scans(m_Zd_output, g_D_output, u_e_output, u_mu_output, base_config, output_filename=plot_output)