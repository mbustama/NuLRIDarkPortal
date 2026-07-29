"""
repair.py

Hosts autonomous topological mapping algorithms. Fixes false-valleys dynamically generated 
when Minuit solvers fail to adapt within extreme discontinuity boundaries.
"""

import numpy as np
from src.engines.optimizers import minimize_point, minimize_point_2d

def _global_1d_repair_wrapper(task_args):
    """Exposes 1D repair sequences universally to the active multiprocessing pool."""
    return minimize_point(*task_args)

def _global_repair_wrapper(task_args):
    """Exposes 2D repair sequences universally to the active multiprocessing pool."""
    return minimize_point_2d(*task_args)

def repair_scan_outliers(x_scan, chi2_scan, param_histories, param_name, func, limits, param_names, global_errors, 
                         threshold=0.2, max_passes=10, use_simplex=True, minuit_retries=3, pool=None, num_cores=1,
                         minuit_strategy=0, retry_seed=-1, minuit_maxcall=0):
    """Scans computational array identifying artificial positive data jumps triggered by Minuit stalling and recalculates.

    Arguments:
      x_scan, chi2_scan: the 1D grid and its profiled chi2 values (chi2 copied, never
        mutated in place; the repaired copy is returned).
      param_histories: per-point profiled-parameter dicts; a spike is re-minimized
        warm-started from its better neighbor's solution (and the history entry is
        updated on success).
      param_name, func, limits, param_names, global_errors: the same profiled-fit
        context as scan_parameter_parallel / minimize_point.
      threshold (repair_settings.spike_threshold): a point is a spike when it exceeds
        BOTH neighbors by more than this delta-chi2.
      max_passes (repair_settings.max_repair_passes): full-grid sweeps; the loop
        already exits early when a sweep finds no spikes or fixes none (measured on a
        clean profile: zero tasks, microseconds).
      use_simplex, minuit_retries, minuit_strategy, retry_seed: forwarded to
        minimize_point.
      pool / num_cores: repair tasks are pooled when >= 5 are queued.

    Returns: (x_scan, repaired chi2 array)."""
    chi2_rep = chi2_scan.copy()
    for _ in range(max_passes):
        tasks, indices = [], []
        for i in range(len(chi2_rep)):
            if chi2_rep[i] > 10000.0: continue
            val_p = chi2_rep[i-1] if i > 0 else np.inf
            val_n = chi2_rep[i+1] if i < len(chi2_rep)-1 else np.inf
            
            if chi2_rep[i] > val_p + threshold and chi2_rep[i] > val_n + threshold:
                neighbor = param_histories[i-1] if val_p < val_n else param_histories[i+1]
                tasks.append((x_scan[i], param_name, neighbor, global_errors, func, limits, param_names, use_simplex, minuit_retries, minuit_strategy, retry_seed, minuit_maxcall))
                indices.append(i)
                
        n_tasks = len(tasks)
        if not tasks: break
        
        if pool is None or n_tasks < 5:
            results = [_global_1d_repair_wrapper(t) for t in tasks]
        else:
            chunksize = max(1, n_tasks//(num_cores*4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_1d_repair_wrapper, tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing (Spikes)"))
            except ImportError:
                results = pool.map(_global_1d_repair_wrapper, tasks, chunksize=chunksize)
                
        fixes = sum(1 for k, (fval, p) in enumerate(results) if fval < chi2_rep[indices[k]] and (chi2_rep.__setitem__(indices[k], fval) or param_histories.__setitem__(indices[k], p) or True))
        if fixes == 0: break
    return x_scan, chi2_rep

def repair_scan_continuity(x_scan, chi2_scan, param_histories, param_name, func, limits, param_names, global_errors, 
                           use_simplex=True, continuity_threshold=1.0, max_passes=5, minuit_retries=3, pool=None, num_cores=1,
                           minuit_strategy=0, retry_seed=-1, minuit_maxcall=0):
    """Verifies gradient transitions targeting localized unphysical numerical slope derivations for recalculation.

    Arguments: as repair_scan_outliers, except the trigger: a point is flagged when
    its second difference exceeds continuity_threshold
    (repair_settings.continuity_threshold) AND it sits more than 0.1 above the mean
    of its neighbors; each flagged point is re-minimized twice, warm-started from
    each neighbor. Returns: (x_scan, repaired chi2 array)."""
    print(f"  [Continuity] Checking for slope discontinuities in {param_name} scan...")
    chi2_rep = chi2_scan.copy()
    for pass_idx in range(max_passes):
        repair_tasks, repair_indices = [], []
        for i in range(1, len(chi2_rep) - 1):
            val_p, val_c, val_n = chi2_rep[i-1], chi2_rep[i], chi2_rep[i+1]
            if val_c > 10000.0 or val_p > 10000.0 or val_n > 10000.0: continue
            slope_change = abs((val_n - val_c) - (val_c - val_p))
            deviation = val_c - 0.5 * (val_p + val_n)
            
            if slope_change > continuity_threshold and deviation > 0.1:
                repair_tasks.extend([(x_scan[i], param_name, param_histories[i-1], global_errors, func, limits, param_names, use_simplex, minuit_retries, minuit_strategy, retry_seed, minuit_maxcall),
                                     (x_scan[i], param_name, param_histories[i+1], global_errors, func, limits, param_names, use_simplex, minuit_retries, minuit_strategy, retry_seed, minuit_maxcall)])
                repair_indices.extend([i, i])
        
        n_tasks = len(repair_tasks)
        if n_tasks == 0: break
        
        if pool is None or n_tasks < 5:
            results = [_global_1d_repair_wrapper(t) for t in repair_tasks]
        else:
            chunksize = max(1, n_tasks//(num_cores*4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing (Continuity)"))
            except ImportError:
                results = pool.map(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize)
        
        fixes = sum(1 for k, (fval, p) in enumerate(results) if fval < chi2_rep[repair_indices[k]] and (chi2_rep.__setitem__(repair_indices[k], fval) or param_histories.__setitem__(repair_indices[k], p) or True))
        if fixes == 0: break
    return x_scan, chi2_rep

def repair_steep_gradients(x_scan, chi2_scan, param_histories, param_name, func, limits, param_names, global_errors, 
                           use_simplex=True, threshold=3.0, max_passes=5, minuit_retries=3, pool=None, num_cores=1,
                           minuit_strategy=0, retry_seed=-1, minuit_maxcall=0):
    """Repairs jagged structural edges ensuring physically continuous objective surfaces.

    Arguments: as repair_scan_outliers, except the trigger: any adjacent pair
    differing by more than threshold (repair_settings.gradient_threshold) flags its
    HIGHER member, re-minimized warm-started from the lower one. Returns:
    (x_scan, repaired chi2 array)."""
    print(f"  [Gradient] Checking for steep gradients in {param_name} scan...")
    chi2_rep = chi2_scan.copy()
    for pass_idx in range(max_passes):
        repair_tasks, repair_indices = [], []
        for i in range(len(chi2_rep) - 1):
            val_a, val_b = chi2_rep[i], chi2_rep[i+1]
            if val_a > 10000.0 or val_b > 10000.0: continue
            
            if abs(val_a - val_b) > threshold:
                bad_idx, good_idx = (i, i+1) if val_a > val_b else (i+1, i)
                if bad_idx in repair_indices: continue
                repair_tasks.append((x_scan[bad_idx], param_name, param_histories[good_idx], global_errors, func, limits, param_names, use_simplex, minuit_retries, minuit_strategy, retry_seed, minuit_maxcall))
                repair_indices.append(bad_idx)
                
        n_tasks = len(repair_tasks)
        if n_tasks == 0: break
        
        if pool is None or n_tasks < 5:
            results = [_global_1d_repair_wrapper(t) for t in repair_tasks]
        else:
            chunksize = max(1, n_tasks//(num_cores*4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing (Gradients)"))
            except ImportError:
                results = pool.map(_global_1d_repair_wrapper, repair_tasks, chunksize=chunksize)
        
        fixes = sum(1 for k, (fval, p) in enumerate(results) if fval < chi2_rep[repair_indices[k]] and (chi2_rep.__setitem__(repair_indices[k], fval) or param_histories.__setitem__(repair_indices[k], p) or True))
        if fixes == 0: break
    return x_scan, chi2_rep

def repair_2d_scan_outliers(x_grid, y_grid, Z_chi2, param_grid, name_pair, func, limits, param_names, global_errors,
                            use_simplex=True, threshold=0.2, max_passes=100, minuit_retries=3, pool=None, num_cores=1,
                            minuit_strategy=0, retry_seed=-1, minuit_maxcall=0):
    """Maps nearest-neighbor evaluations tracing 2D anomalies to mathematically verify structural depressions.

    Arguments: the 2D analogue of repair_scan_outliers -- x_grid/y_grid/Z_chi2/
    param_grid come from scan_pair_parallel (Z rows index y, columns x); a cell is
    flagged when it exceeds its best 4-neighbor by more than threshold, and is
    re-minimized via minimize_point_2d warm-started from that neighbor's solution.
    threshold/max_passes/use_simplex/minuit_retries/minuit_strategy/retry_seed/pool/
    num_cores: as in the 1D repairs. Returns: (x_grid, y_grid, repaired Z)."""
    Z_repaired = Z_chi2.copy()
    rows, cols = Z_repaired.shape
    for pass_idx in range(max_passes):
        repair_tasks, repair_indices = [], []
        for i in range(rows):
            for j in range(cols):
                if Z_repaired[i, j] > 10000.0: continue
                neighbors = [(n_i, n_j, Z_repaired[n_i, n_j]) for n_i, n_j in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)] if 0 <= n_i < rows and 0 <= n_j < cols]
                if not neighbors: continue
                min_n = min(neighbors, key=lambda x: x[2])
                if Z_repaired[i, j] > min_n[2] + threshold:
                    best_params = param_grid[min_n[0]][min_n[1]]
                    if best_params:
                        repair_tasks.append(((x_grid[j], y_grid[i]), name_pair, best_params, global_errors, func, limits, param_names, use_simplex, minuit_retries, minuit_strategy, retry_seed, minuit_maxcall))
                        repair_indices.append((i, j))
        
        n_tasks = len(repair_tasks)
        if n_tasks == 0: break
        if pool is None or n_tasks < 5: 
            results = [_global_repair_wrapper(t) for t in repair_tasks]
        else: 
            chunksize = max(1, n_tasks // (num_cores * 4))
            try:
                from tqdm import tqdm
                results = list(tqdm(pool.imap(_global_repair_wrapper, repair_tasks, chunksize=chunksize), total=n_tasks, desc="       Repairing"))
            except ImportError:
                results = pool.map(_global_repair_wrapper, repair_tasks, chunksize=chunksize)
                
        fixes_applied = sum(1 for idx, (fval, params) in enumerate(results) if fval < Z_repaired[repair_indices[idx][0], repair_indices[idx][1]] and (Z_repaired.__setitem__((repair_indices[idx][0], repair_indices[idx][1]), fval) or param_grid[repair_indices[idx][0]].__setitem__(repair_indices[idx][1], params) or True))
        if fixes_applied == 0: break
    return x_grid, y_grid, Z_repaired