"""
dp_spatial.py

The spatial handover: Milky Way and Earth density models, PREM layer geometry and
the phase-exact step-averaged potential, the per-step density-matrix evolution,
the pool workers and persistent pool, and propagate_earth_spatial. Split verbatim
out of dark_portal_solver.py (HAS_TQDM references now read through dp_core, where
that mutable flag lives).
"""
import multiprocessing

import numpy as np

import dp_core
from dp_core import (njit, HAS_NUMBA, numba, safe_print, tqdm, m_to_eV_inv,
                     cm_inv3_to_eV3, GeV, G_F, calc_debye_mass_sq,
                     d_sun_eV_inv, d_moon_eV_inv, N_n_sun, N_n_earth, N_n_moon)
from dp_propagator import propagator_3x3
from dp_thermal import _GL10_X, _GL10_W

# =============================================================================
# SPATIAL HANDOVER HELPER FUNCTIONS
# =============================================================================

# Neutrons per nucleon of the Galaxy's stellar mass, and per electron of its hot gas.
# Stars: McKee, Parravano & Hollenbach (2015, arXiv:1509.05334) put the local stars and
# remnants at 33.4 Msun/pc^2 -- 81% visible stars, 15% white dwarfs, 4% brown dwarfs. With
# neutrons per nucleon (Y + Z)/2 = 0.146 for disc stars (Y ~ 0.275, Z ~ 0.017), 0.5 for
# carbon/oxygen white dwarfs and 0.142 for brown dwarfs, the mass-weighted fraction is 0.20.
# Hot gas: the Miller & Bregman (2013) profile is an ELECTRON density; fully ionized gas with
# Y ~ 0.26 and Z ~ 0.3 Zsun has (Y + Z)/2 / (1 - (Y + Z)/2) = 0.15 neutrons per electron.
# Until 2026-10-12 both were 0.5 (half of all nucleons as neutrons, right for rock but not
# for hydrogen-dominated matter), which made the Galactic potential ~2.5x (stars) and ~3.3x
# (gas) too strong.
_MW_NEUTRONS_PER_NUCLEON_STARS = 0.20
_MW_NEUTRONS_PER_ELECTRON_GAS = 0.15
# Nucleons per cm^3 in 1 Msun/pc^3: 1.98847e33 g / (3.08568e18 cm)^3 / m_p = 40.464.
_NUCLEONS_PER_CM3_PER_MSUN_PC3 = 40.464


@njit("float64(float64, float64, float64)", fastmath=True, cache=True)
def calc_MW_neutron_density(r_kpc: float, theta: float, phi: float) -> float:
    """Calculates the Milky Way neutron number density (cm^-3) at a given location.

    Arguments: r_kpc, theta, phi -- position relative to the Sun (kpc, rad, rad). The
    stellar bulge and discs (McMillan) are mass densities in Msun/pc^3, converted to
    neutrons with _MW_NEUTRONS_PER_NUCLEON_STARS; the hot-gas halo (Miller & Bregman 2013)
    is an electron density, converted with _MW_NEUTRONS_PER_ELECTRON_GAS (see above)."""
    R_0 = 8.33
    sth = np.sin(theta)
    # The law-of-cosines argument is mathematically >= 0, but evaluating it as a
    # difference suffers catastrophic cancellation on the exact line through the
    # galactic center (r ~ R_0, theta = pi/2, phi = 0 -- which IS one of the six
    # default Milky Way trajectories), and under fastmath the reassociated sum can
    # land a hair below zero: sqrt then yields NaN, which propagated into a
    # ZeroDivisionError from the bulge term's power. Any negative value here is pure
    # rounding, so clamp to zero.
    R_gal_sq = (r_kpc * sth)**2 + R_0**2 - 2.0 * r_kpc * R_0 * sth * np.cos(phi)
    R_gal = np.sqrt(max(R_gal_sq, 0.0))
    z_gal = r_kpc * np.cos(theta)
    r_gal = np.sqrt(R_gal**2 + z_gal**2)
    
    rp = np.sqrt(R_gal**2 + (z_gal / 0.5)**2)
    rho_bulge = (94.1 / (1.0 + rp / 0.075)**1.8) * np.exp(-(rp / 2.1)**2)
    rho_thin = (753.0 / 600.0) * np.exp(-np.abs(z_gal) / 0.3 - np.abs(R_gal) / 3.0)
    rho_thick = (182.0 / 1800.0) * np.exp(-np.abs(z_gal) / 0.9 - np.abs(R_gal) / 3.5)
    n_e_gas = 0.46 * (1.0 + (r_gal / 0.35)**2)**(-1.065)    # Miller & Bregman 2013, cm^-3

    n_n_MW = ((rho_bulge + rho_thin + rho_thick) * _NUCLEONS_PER_CM3_PER_MSUN_PC3
              * _MW_NEUTRONS_PER_NUCLEON_STARS
              + n_e_gas * _MW_NEUTRONS_PER_ELECTRON_GAS)
    return n_n_MW

@njit("float64(float64)", fastmath=True, cache=True)
def calc_earth_density(r_km: float) -> float:
    """Returns the Earth's density at a distance r_km from the center using PREM layers."""
    x = r_km / 6371.0
    if r_km <= 1221.5: return 13.0885 - 8.8381*x*x
    elif r_km <= 3480.0: return 12.5815 - 1.2638*x - 3.6426*x*x - 5.5281*x*x*x
    elif r_km <= 5701.0: return 7.9565 - 6.4761*x + 5.5283*x*x - 3.0807*x*x*x
    elif r_km <= 5771.0: return 5.3197 - 1.4836*x
    elif r_km <= 5971.0: return 11.2494 - 8.0298*x
    elif r_km <= 6151.0: return 7.1089 - 3.8045*x
    elif r_km <= 6346.6: return 2.691 + 0.6924*x
    elif r_km <= 6356.0: return 2.9
    elif r_km <= 6368.0: return 2.6
    elif r_km <= 6371.0: return 1.02
    else: return 0.0


@njit("float64(float64, float64)", fastmath=True, cache=True)
def get_baseline_earth(d_km: float, costh: float) -> float:
    """Calculates the exact baseline distance traversed inside the Earth."""
    R_E = 6371.0
    return np.sqrt((R_E**2 - 2.0*R_E*d_km) * costh**2 + 2.0*R_E*d_km) - (R_E - d_km) * costh

# =============================================================================
# EXTRACTED NUMBA KERNELS FOR PROGRESS TRACKING
# =============================================================================


@njit(fastmath=True, cache=True)
def spatial_total_density_mw(r_kpc: float, theta: float, phi: float, m_eff: float) -> float:
    """Total ambient number density (eV^3) at a point on a Milky Way trajectory: the
    galactic profile plus the Yukawa-enveloped Sun/Moon/Earth point-source terms.

    Single canonical implementation shared by init_mw_density_matrix, step_mw_approach
    (which previously each carried an identical copy of this block) and the GPU spatial
    path's density-sequence precompute -- one source of truth instead of three.
    """
    n_n_MW = calc_MW_neutron_density(r_kpc, theta, phi) * cm_inv3_to_eV3
    r_eV = r_kpc * 3.08567758e16 * 1e3 * m_to_eV_inv

    r_sun_eV = np.sqrt(r_eV**2 + d_sun_eV_inv**2)
    r_moon_eV = np.sqrt(r_eV**2 + d_moon_eV_inv**2)

    n_sun_eV3, n_moon_eV3, n_earth_eV3 = 0.0, 0.0, 0.0
    if m_eff > 0:
        n_sun_eV3 = N_n_sun * (m_eff**2) / (4.0 * np.pi * r_sun_eV) * np.exp(-m_eff * r_sun_eV)
        n_moon_eV3 = N_n_moon * (m_eff**2) / (4.0 * np.pi * r_moon_eV) * np.exp(-m_eff * r_moon_eV)
        n_earth_eV3 = N_n_earth * (m_eff**2) / (4.0 * np.pi * r_eV) * np.exp(-m_eff * r_eV)

    return n_n_MW + n_sun_eV3 + n_moon_eV3 + n_earth_eV3

@njit(fastmath=True, cache=True)
def spatial_total_density_earth(l_km: float, baseline_km: float, costh: float,
                                detector_depth_km: float, n_n_amb_eV3: float) -> float:
    """Total neutron number density (eV^3) at path position l_km along an Earth chord:
    the PREM profile at the corresponding geocentric radius plus the ambient
    (galactic + Sun/Moon Yukawa) background. Shared by step_earth_transit and the GPU
    spatial path's density-sequence precompute."""
    f1 = baseline_km - l_km
    f2 = 6371.0 - detector_depth_km
    r_center_km = np.sqrt(f1**2 + f2**2 + 2.0 * f1 * f2 * costh)

    rho_earth_g = calc_earth_density(r_center_km)
    n_n_earth_cm3 = rho_earth_g * 3.011e23

    return n_n_amb_eV3 + n_n_earth_cm3 * cm_inv3_to_eV3

@njit(fastmath=True, cache=True)
def init_mw_density_matrix(r_start_kpc: float, theta: float, phi: float, E_bins: np.ndarray, F_mass_arrival: np.ndarray,
                           M_rot: np.ndarray, m_eff: float, Coef: float, U_LRI_struct: np.ndarray) -> np.ndarray:
    """Calculates the initial density matrix mapping at the 30 kpc galactic edge.

    M_rot = U_pmns @ M_sq @ conj(U_pmns).T is precomputed once by the caller (see
    get_adiabatic_mixing_cosmo's docstring) since it never changes within a run.
    """
    N_E = len(E_bins)
    rho_f = np.zeros((N_E, 3, 3), dtype=np.complex128)

    # n_tot_edge depends only on r_start_kpc/theta/phi/m_eff, all fixed for this whole
    # call, never on the energy bin i -- computed once outside the loop.
    n_tot_edge = spatial_total_density_mw(r_start_kpc, theta, phi, m_eff)
    V_LRI_edge = (Coef * n_tot_edge) * U_LRI_struct

    for i in range(N_E):
        H_vac = (1.0 / (2.0 * E_bins[i])) * M_rot
        H_arr = H_vac + V_LRI_edge
        evals_a, evecs_a = np.linalg.eigh(H_arr)
        
        for a in range(3):
            for b in range(3):
                for k in range(3):
                    rho_f[i, a, b] += evecs_a[a, k] * F_mass_arrival[i, k] * np.conjugate(evecs_a[b, k])
    return rho_f

@njit(fastmath=True, cache=True)
def _evolve_rho_all_bins(E_bins: np.ndarray, rho: np.ndarray, M_rot: np.ndarray,
                         V_const: np.ndarray, dr_eV_inv: float,
                         H_buf: np.ndarray, U_buf: np.ndarray, T_buf: np.ndarray):
    """Applies one spatial step rho -> U rho U^dagger, U = exp(-i (H_vac(E) + V_const) dr),
    to every energy bin, using the closed-form propagator_3x3 and fully hand-rolled 3x3
    products into pre-allocated buffers. The hand-rolling matters: the previous
    formulation (np.linalg.eigh + `U @ rho @ conj(U).T` with ascontiguousarray casts)
    allocated several temporaries per energy bin per step, and at production settings the
    spatial phase performs millions of these steps -- the allocations and the LAPACK call
    dominated its runtime.
    """
    N_E = len(E_bins)
    for i in range(N_E):
        inv2E = 1.0 / (2.0 * E_bins[i])
        for a in range(3):
            for b in range(3):
                H_buf[a, b] = inv2E * M_rot[a, b] + V_const[a, b]

        propagator_3x3(H_buf, dr_eV_inv, U_buf)

        # T = U @ rho_i ; rho_i = T @ U^dagger  (hand-rolled, allocation-free).
        # rho is Hermitian and stays Hermitian under U rho U^dagger, so only the upper
        # triangle is computed; the lower triangle is mirrored by conjugation and the
        # diagonal is forced real -- ~1/3 fewer multiplies in the second product, and
        # exact Hermiticity is enforced instead of drifting at the ulp level.
        for a in range(3):
            for b in range(3):
                T_buf[a, b] = (U_buf[a, 0] * rho[i, 0, b]
                               + U_buf[a, 1] * rho[i, 1, b]
                               + U_buf[a, 2] * rho[i, 2, b])
        for a in range(3):
            rho[i, a, a] = complex((T_buf[a, 0] * np.conjugate(U_buf[a, 0])
                                    + T_buf[a, 1] * np.conjugate(U_buf[a, 1])
                                    + T_buf[a, 2] * np.conjugate(U_buf[a, 2])).real, 0.0)
            for b in range(a + 1, 3):
                val = (T_buf[a, 0] * np.conjugate(U_buf[b, 0])
                       + T_buf[a, 1] * np.conjugate(U_buf[b, 1])
                       + T_buf[a, 2] * np.conjugate(U_buf[b, 2]))
                rho[i, a, b] = val
                rho[i, b, a] = np.conjugate(val)

@njit(fastmath=True, cache=True)
def step_mw_approach(r_current_kpc: float, theta: float, phi: float, dr_MW_eV_inv: float,
                     E_bins: np.ndarray, rho_f: np.ndarray, M_rot: np.ndarray,
                     m_eff: float, Coef: float, U_LRI_struct: np.ndarray, U_step_buffer: np.ndarray,
                     H_buf: np.ndarray, T_buf: np.ndarray):
    """Executes a single path-length step through the Milky Way using pre-allocated contiguous matrix buffers."""
    n_tot = spatial_total_density_mw(r_current_kpc, theta, phi, m_eff)
    V_LRI = (Coef * n_tot) * U_LRI_struct

    _evolve_rho_all_bins(E_bins, rho_f, M_rot, V_LRI, dr_MW_eV_inv, H_buf, U_step_buffer, T_buf)

@njit(fastmath=True, cache=True)
def step_earth_transit(l_km: float, baseline_km: float, costh: float, dr_earth_eV_inv: float,
                       E_bins: np.ndarray, rho_earth: np.ndarray, M_rot: np.ndarray,
                       n_n_amb_eV3: float, Coef: float, U_LRI_struct: np.ndarray,
                       V_SM: np.ndarray, detector_depth_km: float,
                       U_step_buffer: np.ndarray, H_buf: np.ndarray, T_buf: np.ndarray):
    """Executes a single path-length step through the Earth interior using pre-allocated contiguous matrix buffers.

    V_SM is the complete, prebuilt 3x3 SM matter potential (zeros when use_SM_potential is
    off). It is constant along the entire chord -- it depends only on the configured
    electron density and the antineutrino sign convention, never on position -- so it is
    built exactly once in propagate_earth_spatial rather than being reconstructed here on
    every single path-length step as it previously was.
    """
    n_tot_eV3 = spatial_total_density_earth(l_km, baseline_km, costh, detector_depth_km, n_n_amb_eV3)
    V_LRI = (Coef * n_tot_eV3) * U_LRI_struct + V_SM

    _evolve_rho_all_bins(E_bins, rho_earth, M_rot, V_LRI, dr_earth_eV_inv, H_buf, U_step_buffer, T_buf)

# =============================================================================
# REVISED SOLVERS WITH VERBOSE PROGRESS TRACKING
# =============================================================================

def _mw_path_worker(args) -> np.ndarray:
    """Propagates a single Milky Way trajectory (one theta/phi direction) end-to-end.

    Module-level (picklable) so it can be dispatched to a multiprocessing pool: each
    trajectory is fully independent of the others, so this is the unit of parallel work
    for the Milky Way phase of the spatial handover.
    """
    (theta, phi, r_start_kpc, dr_kpc, dr_MW_eV_inv, N_steps_MW, E_bins, F_mass_arrival,
     M_rot, m_eff, Coef, U_LRI_struct) = args

    rho_f = init_mw_density_matrix(r_start_kpc, theta, phi, E_bins, F_mass_arrival, M_rot, m_eff, Coef, U_LRI_struct)
    U_step_buffer = np.zeros((3, 3), dtype=np.complex128)
    H_buf = np.zeros((3, 3), dtype=np.complex128)
    T_buf = np.zeros((3, 3), dtype=np.complex128)
    for step in range(N_steps_MW):
        r_current_kpc = r_start_kpc - step * dr_kpc
        step_mw_approach(r_current_kpc, theta, phi, dr_MW_eV_inv, E_bins, rho_f, M_rot, m_eff, Coef, U_LRI_struct, U_step_buffer, H_buf, T_buf)
    return rho_f

# PREM layer boundary radii (km) -- the breakpoints of calc_earth_density
_PREM_BOUNDARIES_KM = np.array([1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0,
                                6346.6, 6356.0, 6368.0, 6371.0])


def _earth_layer_crossings(baseline_km: float, costh: float, detector_depth_km: float) -> np.ndarray:
    """Path positions l (km, sorted, interior) where a chord crosses a PREM layer
    boundary. Closed form: with u = baseline - l and f2 = 6371 - depth, the geocentric
    radius obeys r^2 = u^2 + 2 f2 costh u + f2^2, so r = r_b gives
    u = -f2 costh +- sqrt(r_b^2 - f2^2 (1 - costh^2))."""
    f2 = 6371.0 - detector_depth_km
    ls = []
    for r_b in _PREM_BOUNDARIES_KM:
        disc = r_b * r_b - f2 * f2 * (1.0 - costh * costh)
        if disc <= 0.0:
            continue
        sq = np.sqrt(disc)
        for u in (-f2 * costh + sq, -f2 * costh - sq):
            l = baseline_km - u
            # margin excludes the entry-point artifact: every chord ENTERS exactly on
            # the r = 6371 surface boundary, so that root lands at l = O(fp noise);
            # genuine slivers below ~1e-6 * baseline are integration-negligible anyway
            if 1e-6 * baseline_km < l < baseline_km * (1.0 - 1e-6):
                ls.append(l)
    return np.array(sorted(ls), dtype=np.float64)


@njit(fastmath=True, cache=True)
def _avg_earth_density_seq(baseline_km: float, costh: float, N_steps: int,
                           detector_depth_km: float, n_n_amb_eV3: float,
                           cross_ls: np.ndarray, glx: np.ndarray, glw: np.ndarray) -> np.ndarray:
    """Per-step EXACT path-averaged total density along an Earth chord:
    nbar_s = (1/dr) * integral of spatial_total_density_earth over [l_s, l_{s+1}],
    with the integral split at PREM layer crossings (cross_ls, sorted) so every
    Gauss-Legendre subsegment integrates a smooth piecewise-polynomial -- the density
    is then exact to fp accuracy for any step count. This is what makes the 'avg'
    stepping scheme's accumulated LRI phase equal Coef * integral(n dl) EXACTLY,
    independent of N_steps -- see propagate_earth_spatial's scheme note."""
    dr = baseline_km / N_steps
    nbar = np.empty(N_steps, dtype=np.float64)
    n_cross = len(cross_ls)
    n_gl = len(glx)
    for s in range(N_steps):
        a = s * dr
        b = a + dr
        total = 0.0
        seg_a = a
        for k in range(n_cross + 1):
            seg_b = b
            if k < n_cross:
                c = cross_ls[k]
                if c <= a:
                    continue
                if c < b:
                    seg_b = c
                else:
                    seg_b = b
            if seg_b <= seg_a:
                continue
            mid = 0.5 * (seg_a + seg_b)
            half = 0.5 * (seg_b - seg_a)
            acc = 0.0
            for q in range(n_gl):
                l = mid + half * glx[q]
                acc += glw[q] * spatial_total_density_earth(l, baseline_km, costh,
                                                           detector_depth_km, n_n_amb_eV3)
            total += acc * half
            seg_a = seg_b
            if seg_a >= b:
                break
        nbar[s] = total / dr
    return nbar


def _earth_chord_worker(args) -> np.ndarray:
    """Propagates a single Earth chord (one zenith angle) end-to-end to the detector.

    Module-level (picklable) so it can be dispatched to a multiprocessing pool: each
    chord is fully independent of the others, so this is the unit of parallel work for
    the Earth phase of the spatial handover.

    earth_scheme selects the potential discretization per step:
      'left' -- historical scheme: V sampled at the step's left endpoint (kept
                byte-identical);
      'avg'  -- V is the EXACT path average over the step (layer-split Gauss-Legendre
                integration of the PREM profile). The accumulated LRI phase then equals
                Coef * integral(n dl) exactly for ANY step count, removing the
                Riemann-sum phase jitter that made results step-count-ambiguous at the
                ~1e-2 level (the round-7 fringe-aliasing finding). Same evolution
                kernel; only the V value per step differs.
    """
    (costh, baseline_km, N_steps_earth, E_bins, rho_surf_avg, M_rot, n_n_amb_eV3, Coef,
     U_LRI_struct, V_SM, detector_depth_km, earth_scheme) = args

    dr_earth_km = baseline_km / N_steps_earth
    dr_earth_eV_inv = dr_earth_km * 1e3 * m_to_eV_inv
    rho_earth = rho_surf_avg.copy()
    U_step_buffer = np.zeros((3, 3), dtype=np.complex128)
    H_buf = np.zeros((3, 3), dtype=np.complex128)
    T_buf = np.zeros((3, 3), dtype=np.complex128)
    if earth_scheme == "avg":
        cross_ls = _earth_layer_crossings(baseline_km, costh, detector_depth_km)
        nbar = _avg_earth_density_seq(baseline_km, costh, N_steps_earth, detector_depth_km,
                                      n_n_amb_eV3, cross_ls, _GL10_X, _GL10_W)
        for step in range(N_steps_earth):
            V_LRI = np.ascontiguousarray((Coef * nbar[step]) * U_LRI_struct + V_SM)
            _evolve_rho_all_bins(E_bins, rho_earth, M_rot, V_LRI, dr_earth_eV_inv,
                                 H_buf, U_step_buffer, T_buf)
    else:
        for step in range(N_steps_earth):
            l_km = step * dr_earth_km
            step_earth_transit(l_km, baseline_km, costh, dr_earth_eV_inv, E_bins, rho_earth, M_rot,
                               n_n_amb_eV3, Coef, U_LRI_struct, V_SM,
                               detector_depth_km, U_step_buffer, H_buf, T_buf)
    return rho_earth

def _split_E_blocks(N_E: int, n_units: int, n_workers: int):
    """Contiguous energy-index blocks for splitting spatial-transit tasks.

    Every energy bin evolves independently through the spatial transit (H depends on E
    only through its own bin), so a (path-or-chord, E-block) pair is just as valid a
    unit of parallel work as a whole path or chord -- and much better balanced when the
    unit count doesn't divide the worker count (e.g. the default 12 Earth chords on 8
    workers leave 4 workers idle for half the phase). Blocks are sized so that
    n_units * n_blocks comfortably exceeds the worker count. Returns [(0, N_E)] (no
    splitting) when a single worker is in use, preserving the sequential path exactly.
    """
    if n_workers <= 1 or n_units <= 0:
        return [(0, N_E)]
    blocks_per_unit = int(np.ceil(2.0 * n_workers / n_units))
    blocks_per_unit = max(1, min(blocks_per_unit, N_E))
    edges = np.linspace(0, N_E, blocks_per_unit + 1).astype(np.int64)
    return [(int(edges[b]), int(edges[b + 1])) for b in range(blocks_per_unit)
            if edges[b + 1] > edges[b]]

# =============================================================================
# PERSISTENT SPATIAL POOL (standalone-solver side)
# =============================================================================
# A single run_solver call previously created and destroyed FOUR multiprocessing
# pools (MW + Earth phase, neutrino + antineutrino pass), and a scan re-pays that
# per point. This module-level pool mirrors the persistent pool already used by
# FrequentistAnalysis/src/physics/kernel.py: created lazily at first use, resized
# only when the requested worker count changes, torn down at exit. It is only ever
# touched from processes that call the spatial handover with n_cores > 1 and no
# caller-managed external_pool -- scanner/analysis pool workers run with n_cores=1
# and never reach it (no nested pools).
_spatial_pool = None
_spatial_pool_size = None
_spatial_pool_atexit = False


def _close_spatial_pool():
    global _spatial_pool
    if _spatial_pool is not None:
        _spatial_pool.terminate()
        _spatial_pool.join()
        _spatial_pool = None


def _get_spatial_pool(n_workers: int):
    """Returns the persistent spatial pool with exactly n_workers processes (creating
    or resizing it as needed), or None for n_workers <= 1."""
    global _spatial_pool, _spatial_pool_size, _spatial_pool_atexit
    if n_workers <= 1:
        return None
    if _spatial_pool is not None and _spatial_pool_size == n_workers:
        return _spatial_pool
    _close_spatial_pool()
    _spatial_pool = multiprocessing.Pool(processes=n_workers)
    _spatial_pool_size = n_workers
    if not _spatial_pool_atexit:
        import atexit
        atexit.register(_close_spatial_pool)
        _spatial_pool_atexit = True
    return _spatial_pool


def _run_spatial_tasks(worker_fn, tasks, desc, n_workers, show_progress, external_pool=None):
    """Dispatches independent spatial-handover tasks (MW paths / Earth chords) either
    sequentially, across a freshly-created multiprocessing pool, or across a caller-managed
    persistent pool (external_pool), depending on n_workers and whether one was supplied.

    external_pool exists for callers that invoke this many times in a tight loop (e.g.
    FrequentistAnalysis's kernel.py evaluating the physics kernel once per Migrad iteration
    during the initial global fit) and want to reuse the same worker processes across all of
    them, rather than spinning up a fresh Pool every call. Creating a Pool is cheap under the
    'fork' start method (a copy-on-write process clone) but expensive under 'spawn' (a full
    interpreter relaunch plus reimport per worker) -- and FrequentistAnalysis's main.py forces
    'spawn'. Without reuse, using more than one core here made every single Migrad evaluation
    roughly 10x *slower*, not faster (measured directly). With external_pool=None (the
    default), the module-level persistent pool (_get_spatial_pool) is used instead of the
    historical per-call Pool creation -- a single run previously built and tore down four
    pools (MW/Earth x nu/nubar), and scans re-paid that per point.
    """
    if n_workers > 1 and len(tasks) > 1:
        pool = external_pool if external_pool is not None else _get_spatial_pool(n_workers)
        if pool is not None:
            iterator = pool.imap(worker_fn, tasks)
            if dp_core.HAS_TQDM and show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc=desc, leave=False)
            results = list(iterator)
        else:
            with multiprocessing.Pool(processes=min(n_workers, len(tasks))) as pool:
                iterator = pool.imap(worker_fn, tasks)
                if dp_core.HAS_TQDM and show_progress:
                    iterator = tqdm(iterator, total=len(tasks), desc=desc, leave=False)
                results = list(iterator)
    else:
        iterator = tqdm(tasks, desc=desc, leave=False) if (dp_core.HAS_TQDM and show_progress) else tasks
        results = [worker_fn(t) for t in iterator]
    return results

def propagate_earth_spatial(E_bins: np.ndarray, F_mass_arrival: np.ndarray,
                            M_rot: np.ndarray,
                            u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
                            T_nu0: float, use_debye_screening: bool, eta_bg: np.ndarray,
                            detector_depth_km: float, use_SM_potential: bool, n_e_earth_eV3: float,
                            is_antineutrino: bool, n_mw_paths: int, n_earth_chords: int,
                            N_steps_MW: int = 25, r_start_kpc: float = 100.0,
                            show_progress: bool = True, verbose: int = 1, n_cores: int = 1,
                            N_steps_earth: int = 500, external_pool=None,
                            U_LRI_struct=None, V_SM_struct=None, use_gpu: bool = False,
                            earth_step_scheme: str = "left") -> np.ndarray:
    """
    Executes pure spatial ODE through Milky Way and Earth with active tracking.
    Milky Way trajectories and Earth chords are mutually independent, so when n_cores > 1
    each set is distributed across a multiprocessing pool instead of being run sequentially.

    Arguments:
      E_bins: energy grid (eV).
      F_mass_arrival: (N_E, 3) mass-basis fluxes arriving from the cosmological phase.
      M_rot: (3, 3) U_pmns @ M_sq @ conj(U_pmns).T, precomputed once per run.
      u_arr: (3,) complex sterile mixing amplitudes, pre-conjugated for antineutrinos.
      m_Zd, g_D, eps_prime: dark mediator mass (eV) and couplings.
      T_nu0: CnuB temperature (eV); use_debye_screening: thermal Debye mass toggle;
      eta_bg: (3,) CnuB flavor asymmetries (all three feed the screening mass).
      detector_depth_km: detector depth below the surface (sets chord geometry).
      use_SM_potential / n_e_earth_eV3: standard MSW potential inside the Earth.
      is_antineutrino: flips the potential sign (Coef) for this call.
      n_mw_paths, n_earth_chords: solid-angle / zenith averaging counts.
      N_steps_MW, r_start_kpc: Milky Way step count and entry radius (kpc).
      show_progress, verbose: progress bars / logging.
      n_cores: pool size for the (path-or-chord, E-block) task distribution.
      N_steps_earth, earth_step_scheme: Earth-transit resolution and potential
        discretization -- see the dedicated paragraph below.
      external_pool, U_LRI_struct, V_SM_struct, use_gpu: see below.

    Returns: (N_E, 3) flavor fluxes at the detector depth.

    external_pool, if given, is a caller-managed multiprocessing.Pool reused across many
    calls instead of creating a fresh one each time -- see _run_spatial_tasks's docstring.

    U_LRI_struct / V_SM_struct, if given, are dimensionless 3x3 Hermitian flavor structure
    matrices overriding the default LRI texture (u_i * conj(u_j)) and SM matter-potential
    texture (diag(1, 0, 0)) respectively. IMPORTANT CONVENTION: this function expects them
    already conjugated for antineutrinos, exactly as u_arr itself arrives pre-conjugated --
    propagate_spectrum handles that conjugation centrally, so pass neutrino-convention
    matrices there, not here. The overrides affect only the coherent forward-scattering
    potentials; the Debye screening mass (and, upstream, the thermal attenuation and
    regeneration rates) remain tied to the u_e/u_mu/u_tau mixing amplitudes.

    earth_step_scheme: 'avg' (exact step-averaged potential; step-count-independent
    LRI phase, certified converged -- the config default) or 'left' (historical
    left-endpoint sampling). The function-signature defaults (N_steps_earth=500,
    earth_step_scheme='left') preserve historical behavior for direct callers;
    run_solver / the analysis pipeline always pass the DEFAULT_CONFIG values
    ('avg', 25). See _earth_chord_worker and the scheme NOTE below.
    """
    N_E = len(E_bins)
    lbl = "Anti-Nu" if is_antineutrino else "Nu"
    n_workers = max(1, min(int(n_cores), multiprocessing.cpu_count()))

    # --- Dynamic Earth Chords ---
    if n_earth_chords <= 1:
        earth_costh = np.array([-1.0])
    else:
        # Evenly spaced chords from exact core transit (-1.0) to grazing (0.0)
        earth_costh = np.linspace(-1.0, 0.0, int(n_earth_chords))

    # --- Dynamic Milky Way Paths ---
    if n_mw_paths <= 1:
        mw_theta = np.array([np.pi/2])
        mw_phi = np.array([0.0])
    elif n_mw_paths == 6:
        # Preserve original standard baseline exact cardinal axis mapping
        mw_theta = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, 0.0, np.pi])
        mw_phi = np.array([0.0, np.pi, np.pi/2, 3.0*np.pi/2, 0.0, 0.0])
    else:
        # Fibonacci Spherical Lattice for perfectly isotropic uniform solid-angle averaging
        indices = np.arange(0, n_mw_paths, dtype=np.float64)
        mw_phi = (np.pi * (1.0 + np.sqrt(5.0)) * indices) % (2 * np.pi)
        mw_theta = np.arccos(1.0 - 2.0 * (indices + 0.5) / n_mw_paths)

    m_D_sq = calc_debye_mass_sq(g_D, u_arr, T_nu0, eta_bg) if use_debye_screening else 0.0
    m_eff = np.sqrt(m_Zd**2 + m_D_sq)

    c_W = 0.88
    m_Z = 91.18 * GeV
    g_weak = 0.65
    Coef = (g_D * eps_prime) / (m_Zd**2 + m_D_sq) * (4.0 * c_W * m_Z**2 / g_weak) * (np.sqrt(2.0)/2.0 * G_F)
    if is_antineutrino:
        Coef = -Coef

    if U_LRI_struct is None:
        # Default texture: outer product of the sterile mixing vector. u_arr arrives
        # pre-conjugated for antineutrinos, so this construction is automatically in the
        # right convention for either case.
        U_LRI_struct = np.outer(u_arr, np.conjugate(u_arr))
    U_LRI_struct = np.ascontiguousarray(U_LRI_struct, dtype=np.complex128)

    # Ambient (galactic-center + Sun/Moon Yukawa) density and the SM matter potential are
    # pure functions of the inputs -- computed here, ahead of both phases, so the CPU and
    # GPU orchestrations below share them.
    n_sun_eV3, n_moon_eV3 = 0.0, 0.0
    if m_eff > 0:
        n_sun_eV3 = N_n_sun * (m_eff**2) / (4.0 * np.pi * d_sun_eV_inv) * np.exp(-m_eff * d_sun_eV_inv)
        n_moon_eV3 = N_n_moon * (m_eff**2) / (4.0 * np.pi * d_moon_eV_inv) * np.exp(-m_eff * d_moon_eV_inv)
    n_n_amb_eV3 = calc_MW_neutron_density(0.0, 0.0, 0.0) * cm_inv3_to_eV3 + n_sun_eV3 + n_moon_eV3

    # The SM matter potential is constant along every chord (it depends only on the
    # configured electron density, the flavor structure, and the antineutrino sign
    # convention -- never on position), so it is built exactly once here and handed to
    # step_earth_transit prebuilt. V_SM_struct defaults to the standard charged-current
    # texture diag(1, 0, 0); a user override arrives here already conjugated for
    # antineutrinos (propagate_spectrum's convention).
    V_SM = np.zeros((3, 3), dtype=np.complex128)
    if use_SM_potential:
        if V_SM_struct is None:
            S_SM = np.zeros((3, 3), dtype=np.complex128)
            S_SM[0, 0] = 1.0
        else:
            S_SM = np.asarray(V_SM_struct, dtype=np.complex128)
        V_SM_scale = np.sqrt(2.0) * G_F * n_e_earth_eV3
        if is_antineutrino:
            V_SM_scale = -V_SM_scale
        V_SM = V_SM_scale * S_SM
    V_SM = np.ascontiguousarray(V_SM)

    # PHASE-NOISE ADVISORY: when the LRI potential is so large (1/m_Zd^2-enhanced at tiny
    # mediator masses) that a single Earth step accrues more phase than double precision
    # can represent (eps * spread * dr > 0.1 rad of pure node noise), propagator_3x3's
    # projector guard keeps the evolution unitary but the oscillation phases carry no
    # information -- the returned fluxes are decoherence-averaged only. Estimated here
    # with the PREM-core density (the worst case along any chord).
    if verbose >= 1:
        s_lri = float(np.max(np.abs(np.linalg.eigvalsh(U_LRI_struct))))
        n_core_eV3 = 13.0885 * 3.011e23 * cm_inv3_to_eV3
        dr_core_eV_inv = (get_baseline_earth(detector_depth_km, -1.0) / N_steps_earth) * 1e3 * m_to_eV_inv
        if 2.2e-16 * abs(Coef) * n_core_eV3 * s_lri * dr_core_eV_inv > 0.1:
            safe_print(
                f"[WARNING] LRI potential scale (|Coef|*n_core ~ {abs(Coef) * n_core_eV3 * s_lri:.2e} eV) "
                f"is phase-noise-dominated for the Earth transit at these parameters "
                f"(m_Zd_eV={m_Zd:.3g}, g_D={g_D:.3g}, epsilon_prime={eps_prime:.3g}): "
                f"per-step phases exceed double precision. Fluxes are unitary but "
                f"decoherence-averaged only (see propagator_3x3's phase-noise guard).")

    # PHASE 1: Milky Way & Solar System Approach
    r_end_kpc = 6371.0 / 3.08567758e16
    dr_kpc = (r_start_kpc - r_end_kpc) / N_steps_MW
    dr_MW_eV_inv = dr_kpc * 3.08567758e16 * 1e3 * m_to_eV_inv

    # Optional CUDA path: the whole handover as one batched GPU workload (same physics --
    # shared density helpers, same closed-form propagator and degeneracy guard; see
    # gpu_spatial.py, including its validation-status note). Falls back to the CPU path
    # with a warning when cupy or a CUDA device is missing.
    if use_gpu:
        try:
            from gpu_spatial import gpu_available, run_spatial_handover_gpu
            _gpu_ok = gpu_available()
        except ImportError:
            _gpu_ok = False
        if not _gpu_ok:
            safe_print("[GPU] use_gpu requested but cupy with a CUDA device is unavailable -- using the CPU path.")
        else:
            if verbose == 2:
                safe_print(f"\n--- Spatial Handover on GPU ({lbl}): {len(mw_theta)} path(s), {len(earth_costh)} chord(s) ---")
            rho0_paths = np.stack([init_mw_density_matrix(r_start_kpc, mw_theta[k], mw_phi[k], E_bins,
                                                          F_mass_arrival, M_rot, m_eff, Coef, U_LRI_struct)
                                   for k in range(len(mw_theta))], axis=0)
            n_seq_mw = np.empty((len(mw_theta), N_steps_MW))
            for k in range(len(mw_theta)):
                for s in range(N_steps_MW):
                    n_seq_mw[k, s] = spatial_total_density_mw(r_start_kpc - s * dr_kpc, mw_theta[k], mw_phi[k], m_eff)
            rho_paths = run_spatial_handover_gpu(E_bins, rho0_paths, n_seq_mw,
                                                 np.full(len(mw_theta), dr_MW_eV_inv),
                                                 M_rot, Coef, U_LRI_struct, None)
            rho_surf_avg = np.mean(rho_paths, axis=0)

            baselines = np.array([get_baseline_earth(detector_depth_km, c) for c in earth_costh])
            dr_earth_km = baselines / N_steps_earth
            n_seq_earth = np.empty((len(earth_costh), N_steps_earth))
            for c in range(len(earth_costh)):
                if earth_step_scheme == "avg":
                    cross_ls = _earth_layer_crossings(baselines[c], earth_costh[c], detector_depth_km)
                    n_seq_earth[c, :] = _avg_earth_density_seq(baselines[c], earth_costh[c],
                                                               N_steps_earth, detector_depth_km,
                                                               n_n_amb_eV3, cross_ls, _GL10_X, _GL10_W)
                else:
                    for s in range(N_steps_earth):
                        n_seq_earth[c, s] = spatial_total_density_earth(s * dr_earth_km[c], baselines[c],
                                                                        earth_costh[c], detector_depth_km, n_n_amb_eV3)
            rho0_chords = np.broadcast_to(rho_surf_avg, (len(earth_costh), N_E, 3, 3)).copy()
            rho_chords = run_spatial_handover_gpu(E_bins, rho0_chords, n_seq_earth,
                                                  dr_earth_km * 1e3 * m_to_eV_inv,
                                                  M_rot, Coef, U_LRI_struct, V_SM)
            rho_earth_avg = np.mean(rho_chords, axis=0)
            F_earth = np.real(np.diagonal(rho_earth_avg, axis1=1, axis2=2)).copy()
            if verbose == 2:
                safe_print(f"Spatial propagation finished ({lbl}, GPU).")
            return F_earth

    if verbose == 2:
        safe_print(f"\n--- Spatial Handover Commencing ({lbl}) ---")
        safe_print(f"Entering Milky Way geometry at r = {r_start_kpc:.1f} kpc from Earth ({len(mw_theta)} path(s), {n_workers} worker(s))...")

    # Tasks are (path, E-block) pairs: energy bins evolve independently, so splitting the
    # E-grid gives even worker loads when the path count doesn't divide the worker count
    # (see _split_E_blocks). Per-path results are reassembled to full arrays first, so the
    # np.mean over paths below is arithmetically identical to the unsplit version.
    mw_blocks = _split_E_blocks(N_E, len(mw_theta), n_workers)
    mw_tasks = [(mw_theta[k], mw_phi[k], r_start_kpc, dr_kpc, dr_MW_eV_inv, N_steps_MW, E_bins[b0:b1],
                 F_mass_arrival[b0:b1], M_rot, m_eff, Coef, U_LRI_struct)
                for k in range(len(mw_theta)) for (b0, b1) in mw_blocks]
    mw_block_results = _run_spatial_tasks(_mw_path_worker, mw_tasks, f"MW paths ({lbl})", n_workers, show_progress, external_pool=external_pool)
    mw_results = []
    for k in range(len(mw_theta)):
        rho_full = np.empty((N_E, 3, 3), dtype=np.complex128)
        for b, (b0, b1) in enumerate(mw_blocks):
            rho_full[b0:b1] = mw_block_results[k * len(mw_blocks) + b]
        mw_results.append(rho_full)
    rho_surf_avg = np.mean(np.stack(mw_results, axis=0), axis=0)

    # PHASE 2: Earth Transit (PREM Layers)
    if verbose == 2:
        safe_print(f"Milky Way transit complete. Initiating Solar System point-source locks and Earth PREM transit ({len(earth_costh)} chord(s))...")

    # NOTE ON STEP SCHEMES ('avg' vs 'left', grid.earth_step_scheme): under the default
    # 'avg' scheme each step uses the EXACT path-averaged potential, so the accumulated
    # LRI phase equals Coef * integral(n dl) exactly for ANY step count -- measured at
    # the production validation points (below/inband/strong classes), fluxes at
    # N_steps_earth = 8 and 1000 agree to <= 4.5e-9 with energy-averaged ratios to
    # <= 6.5e-10, which is what justifies the N_steps_earth = 25 default (a ~3x
    # margin over the observed knee; revalidated 100 vs 25 at the four standard
    # points on adoption). The historical 'left' scheme (V sampled at each
    # step's left endpoint) carries a Riemann-sum phase error of thousands of radians
    # that RE-ALIASES the Earth-transit fringes on every step-count change (500 vs 501
    # measured 1.6-17% pointwise flux / up to 1.7e-2 energy-averaged-ratio scatter);
    # everything in the historical note below applies to THAT scheme. Phase-noise-
    # regime points (e.g. m_Zd ~ 40 eV, potential-dominated Earth transit) remain
    # pointwise-unconvergeable under ANY scheme (decoherence-averaged only, see
    # propagator_3x3's phase-noise guard).
    #
    # HISTORICAL NOTE ('left' scheme): an earlier version scaled n_steps_earth down for
    # shorter chords (same physical km/step as the costh=-1 chord, since that seemed to only cost
    # resolution nobody needed -- verified via a synthetic, purely-diagonal test state to
    # agree with a 2000-step reference to 1e-12). That test state was misleading: the real
    # rho_surf_avg arriving from the Milky Way phase carries off-diagonal coherences, and at
    # production defaults this showed a genuine, energy-localized ~10% error for the shorter
    # chords, concentrated at the exact energy where the Earth matter potential drives a
    # near-degenerate (MSW-like) eigenvalue crossing -- near such a crossing, eigenvectors
    # rotate rapidly with position *independent of total path length*, so "shorter chord
    # needs fewer steps" is the wrong criterion. Fixed step count for every chord (now
    # caller-configurable via N_steps_earth; 500 was the pinned value throughout the
    # 'left'-scheme era, lowered to 100 once the 'avg' scheme's certificate landed) until a
    # scheme that actually tracks local eigenvalue splitting is worth the complexity.
    # WARNING: reducing N_steps_earth trades away exactly the resolution that protects
    # against the failure mode above. It has NOT been validated as safe at any value below
    # the 500 default across a resonance-scanning parameter range -- verify convergence
    # against a higher-resolution reference for your own m_Zd range before trusting a lower
    # value in a real fit.
    # Same (chord, E-block) task splitting as the Milky Way phase above.
    earth_blocks = _split_E_blocks(N_E, len(earth_costh), n_workers)
    earth_tasks = [(costh, get_baseline_earth(detector_depth_km, costh), N_steps_earth, E_bins[b0:b1],
                    rho_surf_avg[b0:b1], M_rot, n_n_amb_eV3, Coef, U_LRI_struct, V_SM,
                    detector_depth_km, earth_step_scheme) for costh in earth_costh for (b0, b1) in earth_blocks]
    earth_block_results = _run_spatial_tasks(_earth_chord_worker, earth_tasks, f"Earth chords ({lbl})", n_workers, show_progress, external_pool=external_pool)
    earth_results = []
    for c in range(len(earth_costh)):
        rho_full = np.empty((N_E, 3, 3), dtype=np.complex128)
        for b, (b0, b1) in enumerate(earth_blocks):
            rho_full[b0:b1] = earth_block_results[c * len(earth_blocks) + b]
        earth_results.append(rho_full)
    rho_earth_avg = np.mean(np.stack(earth_results, axis=0), axis=0)
    F_earth = np.real(np.diagonal(rho_earth_avg, axis1=1, axis2=2)).copy()

    if verbose == 2:
        safe_print(f"Spatial propagation finished ({lbl}). Flux density matrix mapped to detector depth.")

    return F_earth


