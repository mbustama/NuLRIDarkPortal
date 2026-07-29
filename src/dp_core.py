"""
dp_core.py

Import-order-sensitive environment setup (BLAS thread pins, TBB threading layer),
optional-dependency detection (numba/tqdm/matplotlib fallbacks), physical constants,
and the small shared physics utilities used across every dark-portal module.

MUST be the first dark-portal module imported (every other dp_* module imports it),
so the environment guards below run before numpy/BLAS and numba load. Split verbatim
out of dark_portal_solver.py (2026-07); that module remains the public facade.

NOTE: HAS_TQDM is mutable module state owned HERE -- silence the solver's progress
bars with `import dp_core; dp_core.HAS_TQDM = False` (the facade only provides a
read-through for backward compatibility).
"""
import os

# =============================================================================
# THREAD-OVERSUBSCRIPTION GUARD (must run before numpy/BLAS is imported)
# =============================================================================
# Every linear-algebra call in this solver diagonalizes a fixed 3x3 Hamiltonian, far too
# small to benefit from a multi-threaded BLAS/LAPACK backend (MKL, OpenBLAS). Left at
# their defaults, these libraries spin up one thread pool per *process*, and since this
# solver's own parallelism comes from numba (prange, scaled via numba.set_num_threads in
# propagate_spectrum) and from multiprocessing (across independent MW paths/Earth chords,
# and across parameter configs in dark_portal_scanner.py / Minuit workers in
# FrequentistAnalysis), an unconstrained BLAS thread pool multiplies against every one of
# those layers and can *slow down* a parallel run by 3-5x. Pinning BLAS to a single thread
# here removes that hazard everywhere this module is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


import numpy as np

# =============================================================================
# OPTIONAL DEPENDENCIES VERIFICATION
# =============================================================================
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: 'matplotlib' not found. Plotting will be disabled.")

# TBB THREADING LAYER (when available): numba's default workqueue layer is not
# fork/spawn-safe -- a prange kernel followed by multiprocessing pool use in the same
# process deadlocked the forced-'spawn' persistent-pool combination during the
# (since-reverted) threads-mode spatial experiments, and the same hazard class applies
# in principle to the thermal-grid prange kernels running alongside the scan pools.
# TBB coexists safely with fork and spawn (verified by the prange->spawn-pool stress
# test in the suite). Selected at import time, before any parallel kernel compiles,
# and only when the runtime is actually loadable -- machines without libtbb keep the
# default layer and identical behavior to before.
_TBB_LAYER = False
try:
    # numba's compiled tbb layer links libtbb directly: this import succeeds exactly
    # when both numba's tbbpool extension and a loadable libtbb runtime are present
    # (pip package 'tbb' ships the runtime without any importable python module).
    from numba.np.ufunc import tbbpool as _tbbpool  # noqa: F401
    import numba as _numba_cfg
    os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")   # inherited by pool children
    if os.environ["NUMBA_THREADING_LAYER"] == "tbb":
        # the tbbpool import above already imported numba, whose config snapshot of the
        # env predates our setdefault -- set the live config attribute too (it is read
        # at layer-load time, i.e. at the first parallel kernel launch)
        _numba_cfg.config.THREADING_LAYER = "tbb"
        _TBB_LAYER = True
except ImportError:
    pass

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None
    print("Warning: 'numba' not found. Falling back to pure Python (will be significantly slower).")
    def njit(*args, **kwargs):
        """Dummy decorator if numba is unavailable or string signatures are passed."""
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        if len(args) == 1 and isinstance(args[0], str): return decorator
        return decorator
    prange = range

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: 'tqdm' not found. Progress bars will be disabled.")
    def tqdm(iterable, *args, **kwargs): return iterable

def safe_print(msg: str):
    """Prints output safely without breaking the tqdm progress bar alignment."""
    if HAS_TQDM and hasattr(tqdm, 'write'):
        tqdm.write(msg)
    else:
        print(msg)


# =============================================================================
# PHYSICAL CONSTANTS (Natural Units: eV = 1)
# =============================================================================
eV = 1.0
keV = 1e3 * eV
MeV = 1e6 * eV
GeV = 1e9 * eV
TeV = 1e12 * eV
PeV = 1e15 * eV

# Length & volume conversions
m_to_eV_inv = 5.0677e6
cm_inv3_to_eV3 = 1.0 / ((m_to_eV_inv * 1e-2)**3)
Mpc_to_eV_inv = 3.08567758e22 * m_to_eV_inv

# Electroweak & Cosmological Constants
G_F = 1.1663787e-23 / (eV**2)
km_per_s = 1.0 / 299792.458

# Local Celestial Constants for Spatial Handover
d_sun_eV_inv = 1.496e11 * m_to_eV_inv
d_moon_eV_inv = 3.84e8 * m_to_eV_inv
R_earth_eV_inv = 6371e3 * m_to_eV_inv
R_core_eV_inv = 3480e3 * m_to_eV_inv

# Total number of neutrons per celestial body (assuming Y_n = Y_p = 0.5 where applicable)
N_n_sun = 2.972e56    
N_n_earth = 1.785e51  
N_n_moon = 1.098e49   


@njit(cache=True)
def construct_PMNS(th12: float, th13: float, th23: float, d_cp: float) -> np.ndarray:
    """Constructs the standard 3x3 unitary PMNS neutrino mixing matrix."""
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


@njit(fastmath=True, cache=True)
def calc_debye_mass_sq(g_D: float, u_arr: np.ndarray, T_nu: float, eta_bg: np.ndarray) -> float:
    """Calculates the Debye screening mass squared generated by the thermal CnuB plasma."""
    u_mag_4 = np.abs(u_arr)**4
    T_nu_alpha_sq = (T_nu**2) * (eta_bg**(2.0/3.0))
    return (g_D**2 / 6.0) * np.sum(u_mag_4 * T_nu_alpha_sq)


def get_mass_eigenvalues(m_lightest: float, dm21_sq: float, dm31_sq: float, ordering: str) -> np.ndarray:
    """Calculates the three absolute mass eigenvalues based on the lightest mass and ordering hierarchy."""
    masses = np.zeros(3, dtype=np.float64)
    if ordering.lower().strip() == "inverted":
        masses[2] = m_lightest
        masses[1] = np.sqrt(masses[2]**2 + dm31_sq)
        masses[0] = np.sqrt(masses[1]**2 - dm21_sq)
    else:
        masses[0] = m_lightest
        masses[1] = np.sqrt(masses[0]**2 + dm21_sq)
        masses[2] = np.sqrt(masses[0]**2 + dm31_sq)
    return masses


@njit(fastmath=True, cache=True)
def compute_dE_geometric(E_bins: np.ndarray) -> np.ndarray:
    """Computes per-bin energy widths from the geometric midpoints of a log-spaced E_bins grid.

    Single canonical implementation shared by propagate_spectrum here and
    compute_energy_averaged_ratios in dark_portal_scanner.py -- both need the exact same
    per-bin dE to stay consistent with the cascading integral's energy-conservation weights,
    so this is centralized rather than kept as two copies that could silently drift apart
    (the same DRY concern that motivated centralizing construct_PMNS/get_mass_eigenvalues).
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
    return dE

