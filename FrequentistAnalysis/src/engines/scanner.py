"""
scanner.py

Coordinates the execution of parameter mesh grids utilizing High-Performance Computing
parallel processing architecture (Python Multiprocessing Pools).

Grid points are dispatched to the pool in contiguous SEGMENTS rather than one by one:
within a segment the minimizer is warm-started from the previous point's profiled
solution (the same neighbor-seeding strategy the topology/repair passes already use,
applied up front), which typically cuts Migrad's iteration count substantially since
adjacent grid points have nearly identical profiled minima. Each segment starts from
the global best-fit values, segments never cross a 2D row boundary, and a failed point
resets its chain to the global start -- so a bad local minimum can contaminate at most
the remainder of one segment, which the existing repair passes then catch. Warm
starting is controlled by scan_settings.use_warm_start (default true).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from src.engines.optimizers import minimize_point, minimize_point_2d


def _segment_scan_worker_1d(task_args):
    """Minimizes a contiguous segment of 1D grid points in one worker, optionally
    warm-starting each point from the previous point's profiled solution."""
    (vals, param_name, global_start_values, global_errors, func, limits, param_names,
     use_simplex, minuit_retries, warm_start) = task_args
    out = []
    start = dict(global_start_values)
    for v in vals:
        fval, params = minimize_point(v, param_name, start, global_errors, func, limits,
                                      param_names, use_simplex, minuit_retries)
        out.append((fval, params))
        if warm_start and params and np.isfinite(fval):
            start = {**global_start_values, **params}
        else:
            start = dict(global_start_values)
    return out


def _segment_scan_worker_2d(task_args):
    """Minimizes a contiguous (same-row) segment of 2D grid points in one worker,
    optionally warm-starting each point from the previous point's profiled solution."""
    (pairs, name_pair, global_start_values, global_errors, func, limits, param_names,
     use_simplex, minuit_retries, warm_start) = task_args
    out = []
    start = dict(global_start_values)
    for pair in pairs:
        fval, params = minimize_point_2d(pair, name_pair, start, global_errors, func,
                                         limits, param_names, use_simplex, minuit_retries)
        out.append((fval, params))
        if warm_start and params and np.isfinite(fval):
            start = {**global_start_values, **params}
        else:
            start = dict(global_start_values)
    return out


def _split_segments(items, num_cores):
    """Splits a task list into contiguous segments: enough for good pool load balancing
    (~3 per core), short enough that a warm-start chain gone wrong cannot contaminate
    much of the grid."""
    n = len(items)
    n_segments = min(n, max(1, num_cores * 3))
    edges = np.linspace(0, n, n_segments + 1).astype(np.int64)
    return [items[edges[s]:edges[s + 1]] for s in range(n_segments) if edges[s + 1] > edges[s]]


def _run_segments(worker_fn, segments, pool, desc):
    if pool is None or len(segments) == 1:
        nested = [worker_fn(seg_task) for seg_task in segments]
    else:
        try:
            from tqdm import tqdm
            nested = list(tqdm(pool.imap(worker_fn, segments), total=len(segments), desc=desc))
        except ImportError:
            nested = pool.map(worker_fn, segments)
    return [r for seg in nested for r in seg]


def scan_parameter_parallel(m_global, param_name, start_val, sigma_guess, func, limits, param_names, output_dir,
                            n_steps=40, use_n_points_internal=False, n_points_internal=100, scan_range_sigma=4.0,
                            minuit_retries=3, use_log_scan=False, force_rigid_bounds=False, pool=None, num_cores=1,
                            use_warm_start=True):
    """
    Constructs distributed HPC tasks dynamically iterating bounds constraints and returns
    synthesized 1D profiling vectors. Features intelligent mesh refinements.
    """
    if force_rigid_bounds and param_name in limits and limits[param_name][0] is not None and limits[param_name][1] is not None:
        scan_min, scan_max = limits[param_name][0], limits[param_name][1]
        print(f"  [1D Scan] Enforcing rigid bounds for {param_name}: [{scan_min}, {scan_max}]")
    else:
        calc_min = start_val - scan_range_sigma * sigma_guess
        calc_max = start_val + scan_range_sigma * sigma_guess
        lim_min = limits.get(param_name, [None, None])[0]
        lim_max = limits.get(param_name, [None, None])[1]

        scan_min = max(calc_min, lim_min) if lim_min is not None else calc_min
        scan_max = min(calc_max, lim_max) if lim_max is not None else calc_max

    if use_log_scan:
        scan_min = max(scan_min, 1e-30); scan_max = max(scan_max, 2e-30)
        base_grid = np.geomspace(scan_min, scan_max, n_steps)
        exec_grid = np.geomspace(scan_min, scan_max, n_points_internal) if use_n_points_internal else base_grid
    else:
        base_grid = np.linspace(scan_min, scan_max, n_steps)
        exec_grid = np.linspace(scan_min, scan_max, n_points_internal) if use_n_points_internal else base_grid

    global_start_values = dict(zip(m_global.parameters, m_global.values))
    global_errors = dict(zip(m_global.parameters, m_global.errors))

    if pool is None or len(exec_grid) < 5:
        segments = [list(exec_grid)]
    else:
        segments = _split_segments(list(exec_grid), num_cores)
    seg_tasks = [(seg, param_name, global_start_values, global_errors, func, limits,
                  param_names, False, minuit_retries, use_warm_start) for seg in segments]
    results = _run_segments(_segment_scan_worker_1d, seg_tasks,
                            None if pool is None else pool, "  Progress")

    chi2_arr_exec = np.array([r[0] for r in results])
    param_hists_exec = [r[1] for r in results]

    if use_n_points_internal and n_points_internal < n_steps:
        valid_mask = chi2_arr_exec < 1e5
        if np.sum(valid_mask) > 1:
            pchip = PchipInterpolator(exec_grid[valid_mask], chi2_arr_exec[valid_mask])
            chi2_arr = np.maximum(pchip(base_grid), 0.0)
        else: chi2_arr = np.full(n_steps, np.inf)

        param_histories = [param_hists_exec[np.abs(exec_grid - x).argmin()] for x in base_grid]
        return base_grid, chi2_arr, param_histories

    return exec_grid, chi2_arr_exec, param_hists_exec

def scan_pair_parallel(m_global, param_x, param_y, limits, func, param_names, n_steps=20,
                       scan_range_sigma=3.5, minuit_retries=3, force_rigid_bounds=False, pool=None, num_cores=1,
                       use_log_x=False, use_log_y=False, use_warm_start=True):
    """Iterates a distributed two-dimensional mesh matrix computing profile isolines contour projections.

    use_log_x / use_log_y switch the corresponding grid to geometric (logarithmic)
    spacing -- the 2D analogue of scan_parameter_parallel's use_log_scan, needed for
    parameters spanning decades (m_Zd_eV, couplings, sterile mixings). Warm-start
    segments never cross a row boundary (both members of a chain share the same y)."""
    sigma_x = m_global.errors[param_x] if m_global.errors[param_x] > 0 else 0.1
    sigma_y = m_global.errors[param_y] if m_global.errors[param_y] > 0 else 0.1
    val_x, val_y = m_global.values[param_x], m_global.values[param_y]

    if force_rigid_bounds and param_x in limits and limits[param_x][0] is not None and limits[param_x][1] is not None:
        min_x, max_x = limits[param_x][0], limits[param_x][1]
    else:
        calc_min_x = val_x - scan_range_sigma*sigma_x
        calc_max_x = val_x + scan_range_sigma*sigma_x
        lim_min_x = limits.get(param_x, [None, None])[0]
        lim_max_x = limits.get(param_x, [None, None])[1]
        min_x = max(calc_min_x, lim_min_x) if lim_min_x is not None else calc_min_x
        max_x = min(calc_max_x, lim_max_x) if lim_max_x is not None else calc_max_x

    if force_rigid_bounds and param_y in limits and limits[param_y][0] is not None and limits[param_y][1] is not None:
        min_y, max_y = limits[param_y][0], limits[param_y][1]
    else:
        calc_min_y = val_y - scan_range_sigma*sigma_y
        calc_max_y = val_y + scan_range_sigma*sigma_y
        lim_min_y = limits.get(param_y, [None, None])[0]
        lim_max_y = limits.get(param_y, [None, None])[1]
        min_y = max(calc_min_y, lim_min_y) if lim_min_y is not None else calc_min_y
        max_y = min(calc_max_y, lim_max_y) if lim_max_y is not None else calc_max_y

    if use_log_x:
        min_x = max(min_x, 1e-30)
        max_x = max(max_x, 2e-30)
        x_grid = np.geomspace(min_x, max_x, n_steps)
    else:
        x_grid = np.linspace(min_x, max_x, n_steps)
    if use_log_y:
        min_y = max(min_y, 1e-30)
        max_y = max(max_y, 2e-30)
        y_grid = np.geomspace(min_y, max_y, n_steps)
    else:
        y_grid = np.linspace(min_y, max_y, n_steps)

    global_vals = dict(zip(m_global.parameters, m_global.values))
    global_errs = dict(zip(m_global.parameters, m_global.errors))

    # Segments per row: warm-start chains must not leap from the end of one row to the
    # start of the next (the y value jumps there, invalidating the seed).
    n_rows = len(y_grid)
    seg_per_row = max(1, int(np.ceil(3.0 * num_cores / max(1, n_rows))))
    segments = []
    for y in y_grid:
        row = [(x, y) for x in x_grid]
        edges = np.linspace(0, len(row), min(seg_per_row, len(row)) + 1).astype(np.int64)
        for s in range(len(edges) - 1):
            if edges[s + 1] > edges[s]:
                segments.append(row[edges[s]:edges[s + 1]])

    n_tasks = n_steps * n_steps
    seg_tasks = [(seg, (param_x, param_y), global_vals, global_errs, func, limits,
                  param_names, True, minuit_retries, use_warm_start) for seg in segments]
    results = _run_segments(_segment_scan_worker_2d, seg_tasks,
                            None if (pool is None or n_tasks < 5) else pool,
                            f"  2D Grid ({n_steps}x{n_steps})")

    chi2_list, params_list = [r[0] for r in results], [r[1] for r in results]
    Z = np.array(chi2_list).reshape((n_steps, n_steps))
    param_grid = [[params_list[i*n_steps + j] for j in range(n_steps)] for i in range(n_steps)]
    return x_grid, y_grid, Z, param_grid
