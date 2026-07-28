"""
flux_postprocessing.py

Post-processing of solved neutrino flux spectra -- routines that consume the
(E_bins, Flux_array) output of dark_portal_solver.propagate_spectrum but are not
themselves part of the propagation physics.

Extracted from dark_portal_scanner.py so that (a) the scanner holds only routines
directly related to parameter scanning, and (b) consumers that need only this
post-processing (e.g. FrequentistAnalysis's kernel.py) no longer have to import the
whole scanner module -- which has import-time side effects (it globally sets
dark_portal_solver.HAS_TQDM = False, silencing the solver's progress bars for every
other user in the same process, and it imports matplotlib).
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        """Dummy decorator if numba is unavailable or string signatures are passed."""
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        if len(args) == 1 and isinstance(args[0], str): return decorator
        return decorator

# compute_dE_geometric is the single canonical bin-width implementation (also used by
# dark_portal_solver.py's own propagate_spectrum); importing it rather than keeping a
# local copy avoids the two silently drifting out of sync.
from dark_portal_solver import compute_dE_geometric


@njit("float64[:](float64[:], float64[:,:])", fastmath=True, cache=True)
def compute_energy_averaged_ratios(E_bins: np.ndarray, Flux_array: np.ndarray) -> np.ndarray:
    """
    Computes the energy-averaged flavor ratios over the fully integrated spectrum.
    Utilizes exact geometric midpoints to perfectly match the cascading energy conservation weights.
    """
    dE = compute_dE_geometric(E_bins)

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
