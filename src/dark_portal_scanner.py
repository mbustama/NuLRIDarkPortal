import os

# Must run before numpy/BLAS is imported -- see the matching guard and full explanation
# in dark_portal_solver.py. This scanner already parallelizes across parameter configs via
# multiprocessing, so each worker's BLAS backend must be pinned to a single thread or the
# per-worker thread pools oversubscribe every core.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

# compute_energy_averaged_ratios now lives in flux_postprocessing.py (it is flux
# post-processing, not a scanning routine); re-exported here so existing code that
# imports it from this module keeps working.
from flux_postprocessing import compute_energy_averaged_ratios  # noqa: F401

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

    The pool size is driven by base_config['execution']['n_cores'] (falling back to
    cpu_count()-1 if unset), clamped to the cores actually available. Previously this
    function always recomputed its own hardcoded cpu_count()-1 pool size regardless of
    what base_config['execution']['n_cores'] said, so that config field was silently
    dropped -- setting it in __main__ had no effect on the actual scan parallelism.
    """
    available_cores = multiprocessing.cpu_count()
    requested_cores = int(base_config.get('execution', {}).get('n_cores', available_cores - 1 if available_cores > 1 else 1))
    n_cores = max(1, min(requested_cores, available_cores))

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
    
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlim(min(u_e_vals), max(u_e_vals))
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_xlabel(r'Sterile electron mixing, $u_e$', fontsize=14)
    axes[1, 0].set_ylabel('Energy-averaged flavor ratio', fontsize=14)
    axes[1, 0].set_title(r'Flavor composition vs. $u_e$', fontsize=15, pad=12)
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.4)
    # Add parameter text label (Hardcoding u_mu to 0.0 to match the scan logic)
    # text_bl = f"$u_\\mu = {u_mu_base}$\n$m_{{Z_D}} = {m_Zd_base}$ eV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    text_bl = f"$u_\\mu = 0.0$\n$m_{{Z_D}} = {m_Zd_base/1e6}$ MeV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    axes[1, 0].text(0.05, 0.05, text_bl, transform=axes[1, 0].transAxes, fontsize=11, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # --- Bottom Right Panel: u_mu Mixing Scan ---
    axes[1, 1].plot(u_mu_vals, u_mu_fe, color='crimson', linestyle='--', lw=2.5)
    axes[1, 1].plot(u_mu_vals, u_mu_fmu, color='royalblue', linestyle='-.', lw=2.5)
    axes[1, 1].plot(u_mu_vals, u_mu_ftau, color='forestgreen', linestyle=':', lw=2.5)
    
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlim(min(u_mu_vals), max(u_mu_vals))
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_xlabel(r'Sterile muon mixing, $u_\mu$', fontsize=14)
    axes[1, 1].set_title(r'Flavor composition vs. $u_\mu$', fontsize=15, pad=12)
    axes[1, 1].grid(True, which="both", ls="--", alpha=0.4)
    # Add parameter text label (Hardcoding u_e to 0.0 to match the scan logic)
    # text_br = f"$u_e = {u_e_base}$\n$m_{{Z_D}} = {m_Zd_base}$ eV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    text_br = f"$u_e = 0.0$\n$m_{{Z_D}} = {m_Zd_base/1e6}$ MeV\n$g_D = {g_D_base}$\n$\\epsilon' = {eps_prime_base}$"
    axes[1, 1].text(0.05, 0.05, text_br, transform=axes[1, 1].transAxes, fontsize=11, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    plt.savefig(output_filename, dpi=300)
    print(f"\nScan plot successfully saved to: {os.path.abspath(output_filename)}")
    plt.show()

if __name__ == "__main__":
    import argparse

    SCANNABLE = ["m_Zd_eV", "g_D", "u_e", "u_mu"]
    DEFAULT_RANGES = {
        "m_Zd_eV": (1e-10, 1e10),   # 0.1 neV to 10 GeV
        "g_D":     (1e-10, 1e-1),
        "u_e":     (1e-5, 1.0),
        "u_mu":    (1e-5, 1.0),
    }

    parser = argparse.ArgumentParser(
        description="1D dark-portal parameter scans (energy-averaged flavor ratios per grid "
                    "point) with a 2x2 summary figure. Grids are logarithmic.")
    parser.add_argument("--config", type=str, default="../config/config_dark_portal.json",
                        help="Base solver config JSON (created with defaults if missing).")
    parser.add_argument("--params", nargs="*", default=SCANNABLE, choices=SCANNABLE,
                        help="Which parameters to scan (default: all four).")
    parser.add_argument("--n_points", type=int, default=100, help="Grid points per scan.")
    for _p in SCANNABLE:
        lo, hi = DEFAULT_RANGES[_p]
        parser.add_argument(f"--{_p}_min", type=float, default=lo)
        parser.add_argument(f"--{_p}_max", type=float, default=hi)
    parser.add_argument("--n_E_bins", type=int, default=200,
                        help="Energy bins during scans (reduced from the solver default for speed).")
    parser.add_argument("--n_z_bins", type=int, default=200)
    parser.add_argument("--n_mw_paths", type=int, default=12,
                        help="Milky Way trajectories per point (default 12; the convergence study "
                             "in output/convergence/ found >= 6 sufficient at the studied points, "
                             "but raise this and re-run tools/convergence_study.py when exploring "
                             "very small m_Zd, where the unscreened galactic potential makes the "
                             "solid-angle average anisotropic).")
    parser.add_argument("--n_earth_chords", type=int, default=50,
                        help="Earth chords per point (default 50: the sub-percent knee of the "
                             "convergence study in output/convergence/; 12 suffices for "
                             "percent-level ratios).")
    parser.add_argument("--N_steps_MW", type=int, default=25,
                        help="25 agrees with a 400-step reference to ~1e-10 (tools/convergence_study.py).")
    parser.add_argument("--n_cores", type=int,
                        default=multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    parser.add_argument("--output_dir", type=str, default="../output",
                        help="Scan data goes to <output_dir>/data/, the figure to <output_dir>/plots/.")
    parser.add_argument("--no_plot", action="store_true", help="Skip the 2x2 summary figure.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        base_config = dps.create_default_config(args.config)
    else:
        with open(args.config, 'r') as f:
            base_config = json.load(f)

    # Scan-resolution overrides (CLI-controlled; previously hardcoded here)
    base_config.setdefault('grid', {})
    base_config['grid']['n_E_bins'] = args.n_E_bins
    base_config['grid']['n_z_bins'] = args.n_z_bins
    base_config['grid']['n_mw_paths'] = args.n_mw_paths
    base_config['grid']['n_earth_chords'] = args.n_earth_chords
    base_config['grid']['N_steps_MW'] = args.N_steps_MW
    base_config.setdefault('execution', {})
    # Drives run_parameter_scan's pool size (see its docstring).
    base_config['execution']['n_cores'] = args.n_cores

    # Compile Numba kernels in advance using a lightweight dummy execution: prevents all
    # independent pool workers from JIT-compiling simultaneously on their first task.
    print("Initiating Numba compiler cache warm-up...")
    warmup_config = copy.deepcopy(base_config)
    warmup_config['execution']['verbose'] = 0

    import sys, io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        _ = dps.run_solver(warmup_config)
        dummy_E = np.logspace(13, 16, args.n_E_bins)
        dummy_Flux = np.zeros((args.n_E_bins, 3))
        _ = compute_energy_averaged_ratios(dummy_E, dummy_Flux)
    finally:
        sys.stdout = old_stdout
    print("Warm-up complete. Numba kernels mapped to physical disk cache.")

    grids = {p_: np.logspace(np.log10(getattr(args, f"{p_}_min")),
                             np.log10(getattr(args, f"{p_}_max")), args.n_points)
             for p_ in args.params}
    outputs = {p_: os.path.join(args.output_dir, "data", f"scan_{p_.replace('_eV', '')}.txt")
               for p_ in args.params}

    for p_ in args.params:
        # The sterile-mixing scans zero out the *other* mixing element, matching the
        # panel labels in plot_parameter_scans.
        saved = {}
        if p_ == "u_e":
            saved["u_mu"] = base_config['dark_portal']['u_mu']
            base_config['dark_portal']['u_mu'] = 0.0
        elif p_ == "u_mu":
            saved["u_e"] = base_config['dark_portal']['u_e']
            base_config['dark_portal']['u_e'] = 0.0
        run_parameter_scan(p_, grids[p_], base_config, outputs[p_])
        for k, v in saved.items():
            base_config['dark_portal'][k] = v

    if not args.no_plot:
        if set(args.params) == set(SCANNABLE):
            plot_parameter_scans(outputs["m_Zd_eV"], outputs["g_D"], outputs["u_e"], outputs["u_mu"],
                                 base_config,
                                 output_filename=os.path.join(args.output_dir, "plots", "parameter_scan_plot.png"))
        else:
            print("Summary figure requires all four scans; skipping (ran a subset).")
