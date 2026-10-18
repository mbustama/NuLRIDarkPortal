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
# and across parameter configs in dark_portal_scanner.py / the analysis package's scan
# workers), an unconstrained BLAS thread pool multiplies against every one of
# those layers and can *slow down* a parallel run by 3-5x. Pinning BLAS to a single thread
# here removes that hazard everywhere this module is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# =============================================================================
# NUMBA CACHE KEYED BY THE SOURCES (must run before numba is imported)
# =============================================================================
# numba validates an on-disk cache entry against the stamp of the function's OWN source
# file only, so a jitted function calling a jitted function from another module
# (dp_spatial -> dp_propagator.propagator_3x3, dp_cosmo -> eigenvectors_3x3, every kernel
# -> the helpers below) keeps the callee's old machine code after the callee's module is
# edited -- silently: on 2026-10-12 the spatial kernels kept running a replaced
# propagator. The cache directory is therefore keyed by a hash of every source file in
# this directory, so any edit starts a fresh cache (one full recompile). Directories of
# other versions left unused for a day are removed. A NUMBA_CACHE_DIR set by the user is
# respected as is.
def _source_digest() -> str:
    import hashlib
    src = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            h.update(name.encode())
            with open(os.path.join(src, name), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def _configure_numba_cache():
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    import shutil
    import time
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__", "numba")
    digest = _source_digest()
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(base, digest)    # inherited by pool children
    try:
        now = time.time()
        for name in os.listdir(base):
            path = os.path.join(base, name)
            if name != digest and now - os.path.getmtime(path) > 86400.0:
                shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


_configure_numba_cache()


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

# FASTMATH KERNELS USE error_model="numpy" (2026-10-13). On Tycho's AMD EPYC 9554 (Zen 4)
# nodes, numba's division-by-zero check fired spuriously under the full fastmath flag set:
# dp_spatial._yk_direct raised ZeroDivisionError at r = 6370.96 km for m_eff = 1e-14 eV, while
# the same code returned the exact value (equal to pure Python's) without the check, without
# any one of the nnan/arcp/afn flags, or compiled for a generic CPU; the laptop never raised.
# It failed every potential-plane job's global fit. Every fastmath kernel therefore uses
# numpy's error model, which drops the checks (a true division by zero gives inf or nan); the
# solver's own isfinite checks on the thermal grid and on every flux still raise loudly.
try:
    import numba
    from numba import njit, prange
    # numba reads NUMBA_CACHE_DIR once, at its own import, which may have come first
    numba.config.CACHE_DIR = os.environ["NUMBA_CACHE_DIR"]
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None

# CACHE ENTRIES NAMED BY THEIR SIGNATURE (2026-10-18). numba names a new cache entry's data
# file by the first number free in the index as ITS process last read it, so two processes
# compiling different signatures of one function at once (pool workers, pytest-xdist) both
# write <function>.1.nbc, and the index can end up pointing one signature at the other's
# machine code: run on the wrong argument types it returned garbage (the Earth's Yukawa
# density ~1e58 cm^-3 on Tycho). Named by a hash of the key (signature, CPU, bytecode), a
# data file only ever holds its own signature's code, and it is written before the index
# points at it; a concurrent index update can at worst drop an entry, which recompiles.
if HAS_NUMBA:
    from numba.core import caching as _nb_caching

    def _save_by_key(self, key, data):
        import hashlib
        data_name = "%s.%s.nbc" % (self._index_name[:-len(".nbi")],
                                   hashlib.sha256(repr(key).encode()).hexdigest()[:16])
        self._save_data(data_name, data)
        overloads = self._load_index()
        if overloads.get(key) != data_name:
            overloads[key] = data_name
            self._save_index(overloads)

    _nb_caching.IndexDataCacheFile.save = _save_by_key

try:
    if not HAS_NUMBA:
        raise ImportError
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

def available_cores() -> int:
    """Cores this process may use: its CPU affinity mask, capped by SLURM_CPUS_PER_TASK when
    SLURM sets it, else os.cpu_count(). cpu_count() alone reports the whole machine -- 128 cores
    on a Tycho node, whatever the job was given -- which oversubscribed every pool sized from it."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if slurm.isdigit() and int(slurm) > 0:
        n = min(n, int(slurm))
    return max(1, n)


def default_n_cores() -> int:
    """Default pool size: every available core inside a SLURM job (the allocation is the
    job's to use), all but one elsewhere, so a laptop stays responsive."""
    n = available_cores()
    if os.environ.get("SLURM_JOB_ID"):
        return n
    return n - 1 if n > 1 else 1


def set_numba_threads(n: int) -> int:
    """numba.set_num_threads(n), clamped to numba's pool (NUMBA_NUM_THREADS: the machine's cores
    unless set). Returns the count set. numba raises for more threads than its pool has, and a
    configuration may ask for more cores than the machine running it has: n_cores = 4 crashed
    the solver on GitHub's 2-core runners (found 2026-10-13, when CI first ran with TBB)."""
    if not HAS_NUMBA:
        return 1
    m = max(1, min(int(n), int(numba.config.NUMBA_NUM_THREADS)))
    numba.set_num_threads(m)
    return m


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

# Total number of neutrons per celestial body -- point sources of the dark-portal potential,
# which matter only for mediators light enough to reach across AU-scale distances.
# Sun: present bulk composition, the protosolar X/Y/Z = 0.7154/0.2703/0.0142 (Asplund et al.
#   2009) plus the ~0.038 Msun of hydrogen burned to helium (integrated luminosity, ~0.85 L_sun
#   over 4.57 Gyr): Y = 0.308, i.e. 0.161 neutrons per nucleon.
# Earth: PREM integrated with dp_spatial.earth_neutrons_per_gram (core Y_e 0.4656, mantle and
#   crust 0.4957, top layer water): 0.514 neutrons per nucleon (PREM's mass is 5.973e27 g).
# Moon: silicate rock (Y_e 0.4957) with a ~1.5% iron core (0.4656): 0.505 neutrons per nucleon.
# Until 2026-10-12 the Sun and the Moon used 0.248 per nucleon (2.972e56, 1.098e49) and the
# Earth 0.496 (1.785e51).
N_n_sun = 1.929e56
N_n_earth = 1.849e51
N_n_moon = 2.232e49
# Electrons (= protons) per body, for matter charges other than the neutron's (dp_matter,
# docs/dev/SCOPE_WIDENING_PLAN.md): each body's nucleons less its neutrons above -- the Sun's
# 1.98847e33 g and the Moon's 7.342e25 g at 1/m_u nucleons per gram (1.1975e57, 4.4215e49), and
# the Earth's PREM integrated with dp_spatial.earth_electrons_per_gram (Y_e by layer, water on top).
N_e_sun = 1.0046e57
N_e_earth = 1.7480e51
N_e_moon = 2.1895e49


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


@njit(fastmath=True, error_model="numpy", cache=True)
def calc_debye_mass_sq(g_D: float, u_arr: np.ndarray, T_nu: float, eta_bg: np.ndarray) -> float:
    """Calculates the Debye screening mass squared generated by the thermal CnuB plasma."""
    u_mag_4 = np.abs(u_arr)**4
    T_nu_alpha_sq = (T_nu**2) * (eta_bg**(2.0/3.0))
    return (g_D**2 / 6.0) * np.sum(u_mag_4 * T_nu_alpha_sq)


def get_mass_eigenvalues(m_lightest: float, dm21_sq: float, dm31_sq: float, ordering: str) -> np.ndarray:
    """Calculates the three absolute mass eigenvalues based on the lightest mass and ordering hierarchy.
    dm31_sq is |Delta m^2_3l| (NuFIT): Delta m^2_31 for normal ordering, |Delta m^2_32| for inverted."""
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


@njit(fastmath=True, error_model="numpy", cache=True)
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

