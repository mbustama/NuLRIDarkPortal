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

import copy
import json
import argparse
import multiprocessing
import numpy as np
from typing import Dict, Tuple

# =============================================================================
# OPTIONAL DEPENDENCIES VERIFICATION
# =============================================================================
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: 'matplotlib' not found. Plotting will be disabled.")

try:
    import numba
    from numba import njit, prange
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

# =============================================================================
# NUMBA-ACCELERATED PHYSICS KERNELS
# =============================================================================

@njit(fastmath=True, cache=True)
def calc_H_Hubble(z: float, H0: float, Omega_m: float, Omega_L: float) -> float:
    """Calculates the Hubble expansion rate H(z) at redshift z in a flat LCDM universe."""
    return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)

@njit(fastmath=True, cache=True)
def calc_R_source(z: float, rho_evol: float) -> float:
    """Calculates the cosmological source distribution R(z) relative to z=0."""
    if z < 1.0:
        sfr = (1.0 + z)**3.4
    elif z < 4.0:
        sfr = (2.0**3.4) * ((1.0 + z) / 2.0)**(-0.3)
    else:
        sfr = (2.0**3.4) * (5.0 / 2.0)**(-0.3) * ((1.0 + z) / 5.0)**(-3.5)
        
    return sfr * (1.0 + z)**rho_evol

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

@njit(cache=True)
def calc_V_LRI_cosmo(z: float, u_arr: np.ndarray, U_LRI_struct: np.ndarray,
                     m_Zd: float, g_D: float, eps_prime: float,
                     n_n0: float, T_nu0: float, use_debye_screening: bool, eta_bg: np.ndarray) -> np.ndarray:
    """Computes the macroscopic effective potential V_{LRI} sourced ONLY by cosmological neutrons.

    U_LRI_struct is the dimensionless 3x3 Hermitian flavor structure of the potential --
    by default the outer product u_i * conj(u_j) of the sterile mixing vector, but
    caller-overridable (config key dark_portal.U_LRI_struct) for arbitrary textures.
    Callers pass the already-conjugated structure for antineutrinos (see
    propagate_spectrum), matching the pre-conjugated-u_arr convention used before this
    matrix was threaded through. u_arr itself is still needed here for the Debye
    screening mass, which remains tied to the u magnitudes regardless of any override.
    """
    n_n_z = n_n0 * (1.0 + z)**3

    m_D_sq = 0.0
    if use_debye_screening:
        T_nu_z = T_nu0 * (1.0 + z)
        m_D_sq = calc_debye_mass_sq(g_D, u_arr, T_nu_z, eta_bg)

    c_W = 0.88
    m_Z = 91.18 * GeV
    g_weak = 0.65

    denominator = max(m_Zd**2 + m_D_sq, 1e-25)
    A_LRI = (g_D * eps_prime) / denominator * (4.0 * c_W * m_Z**2 / g_weak) * (np.sqrt(2)/2.0 * G_F * n_n_z)

    return A_LRI * U_LRI_struct

@njit(fastmath=True, cache=True)
def thermal_averaged_rates_flavor(E0: float, z: float, sum_G: float, sum_G_alpha: np.ndarray, 
                                  m_Zd: float, m_nu_bg_arr: np.ndarray, T_nu0: float, 
                                  eta_mass: np.ndarray) -> np.ndarray:
    """Calculates the exact, flavor-dependent flux attenuation rate \\Gamma(E, z)_\\alpha with vectorized momentum sequencing."""
    T_nu = T_nu0 * (1.0 + z)
    n_nu_base = 56.0 * (1.0 + z)**3 * cm_inv3_to_eV3 
    
    Gamma_width = (sum_G * m_Zd) / (12.0 * np.pi)
    MG = m_Zd * Gamma_width
    MG_sq = MG**2
    M2 = m_Zd**2
    
    rates = np.zeros(3, dtype=np.float64)
    if MG == 0.0:
        return rates 
        
    prefactor = 1.0 / (2.0 * np.pi)
    
    for i in range(3):
        if eta_mass[i] <= 0:
            continue
            
        n_nu_i = n_nu_base * eta_mass[i]
        m_i_sq = m_nu_bg_arr[i]**2
        T_i = T_nu * (eta_mass[i]**(1.0/3.0))
        E_inc = E0 * (1.0 + z)
        
        n_p = 100
        p_max = 20.0 * T_i
        dp = p_max / n_p
        
        # Vectorized generation of momentum sequences and Fermi-Dirac distribution weights
        p = (np.arange(n_p) + 0.5) * dp
        E_bg = np.sqrt(m_i_sq + p**2)
        
        f_FD = 1.0 / (np.exp(p / T_i) + 1.0)
        p2_f_FD = (p**2) * f_FD
        
        A = m_i_sq + 2.0 * E_inc * E_bg
        B = 2.0 * E_inc * p
        
        u_plus = A - M2 + B
        u_minus = A - M2 - B
        C = 1.0 - (A - M2) / B
        D = 1.0 - (A - 2.0 * M2) / B
        F = C * M2 - MG_sq / B
        
        # Parallel element-wise calculation of Breit-Wigner s-channel poles
        I_ang = (2.0 / B) + (D / (2.0 * B)) * np.log((u_plus**2 + MG_sq) / (u_minus**2 + MG_sq)) + (F / (B * MG)) * (np.arctan(u_plus / MG) - np.arctan(u_minus / MG))
        
        term = prefactor * p2_f_FD * np.maximum(I_ang, 0.0) * dp
        integral_sum = np.sum(term)
        
        for alpha in range(3):
            rates[alpha] += integral_sum * sum_G_alpha[alpha] * (n_nu_i / n_nu_base)
                    
    return rates

@njit(cache=True)
def get_adiabatic_mixing_cosmo(E0: float, z: float, M_rot: np.ndarray,
                               u_arr: np.ndarray, U_LRI_struct: np.ndarray,
                               m_Zd: float, g_D: float, eps_prime: float,
                               n_n0: float, T_nu0: float, use_debye_screening: bool,
                               is_antineutrino: bool, eta_bg: np.ndarray) -> np.ndarray:
    """Computes the instantaneous mass eigenstates (eigenvectors) of the total effective Hamiltonian.

    M_rot = U_pmns @ M_sq @ conj(U_pmns).T is fixed for an entire propagate_spectrum call (U_pmns
    and M_sq never change within it), so callers precompute it once and pass it in here rather
    than recomputing the same two 3x3 matrix products on every single (E, z) evaluation.
    U_LRI_struct is likewise fixed per call -- see calc_V_LRI_cosmo's docstring.
    """
    E_z = E0 * (1.0 + z)
    H_vac = (1.0 / (2.0 * E_z)) * M_rot

    V_LRI = calc_V_LRI_cosmo(z, u_arr, U_LRI_struct, m_Zd, g_D, eps_prime, n_n0, T_nu0, use_debye_screening, eta_bg)

    if is_antineutrino:
        # Callers already pass in a pre-conjugated u_arr AND a pre-conjugated U_LRI_struct
        # for antineutrinos (see propagate_spectrum), so V_LRI here is already built from
        # the conjugated structure; only the sign needs flipping. Conjugating again would
        # double-apply the transform and corrupt off-diagonal phases whenever the structure
        # is complex (CP-violating textures). Must match the single-transform convention
        # used in the spatial handover (propagate_earth_spatial's Coef sign flip).
        V_LRI = -V_LRI

    H_tot = H_vac + V_LRI
    # Closed-form eigenvectors (ascending-eigenvalue columns, matching eigh's convention)
    # with an internal eigh fallback near level crossings -- see eigenvectors_3x3.
    return eigenvectors_3x3(H_tot)

# =============================================================================
# SPATIAL HANDOVER HELPER FUNCTIONS
# =============================================================================

@njit("float64(float64, float64, float64)", fastmath=True, cache=True)
def calc_MW_neutron_density(r_kpc: float, theta: float, phi: float) -> float:
    """Calculates the Milky Way neutron number density at a given location."""
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
    n_gas = 0.5 * 0.46 * (1.0 + (r_gal / 0.35)**2)**(-1.065) 
    
    n_n_MW = (rho_bulge + rho_thin + rho_thick) * 20.235 + n_gas
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

@njit(cache=True)
def propagator_3x3(H: np.ndarray, t: float, out: np.ndarray):
    """Writes exp(-i H t) for a Hermitian 3x3 H into out. No eigendecomposition, no
    fallback -- uniformly accurate at any eigenvalue gap. Three cooperating pieces:

    1. Everything runs on the traceless shift D = H - q I (q = tr(H)/3), with the global
       phase e^{-i q t} factored out, so all degeneracy logic is scale-invariant.
    2. NODES. Closed-form trigonometric roots of D are used when well-separated. Near a
       double root the cubic formula's roots carry the classic sqrt(eps)-conditioned
       error -- measured on a production Earth-transit Hamiltonian, the close-pair GAP
       came out 257x too large (5.1e-15 vs a true 2.0e-17), which rotates the
       near-degenerate subspace by a wildly wrong angle each step and accumulated to a
       1e-4-level flux error. So whenever the trig roots' closest pair sits below 1e-4
       of the spread, the pair is recomputed by DEFLATION: the well-separated root and
       the pair mean (= -far/2, since tr D = 0) are accurate, the far eigenvector comes
       from the stable projector (D - mean I)^2 (pair contamination suppressed by
       (gap/separation)^2), and the pair's mean and half-gap are read off the 2x2
       complement block -- whose gap formula sqrt(((b11-b22)/2)^2 + |b12|^2) is a sum of
       squares, cancellation-free.
    3. COEFFICIENTS. Newton divided differences of f(x) = e^{-i x t} over the nodes,
       closest pair first: f[a,b] via the exact sinc form when |(a-b)t/2| < 0.5 (where
       the direct difference cancels), and via the direct node-exact quotient
       (f_a - f_b)/(a - b) otherwise -- at large gap phases (Milky Way steps reach
       ~1e9 rad) only the direct form keeps the polynomial interpolating the COMPUTED
       exponentials exactly, which is what keeps the propagator unitary. A second-order
       Taylor branch handles total phase spreads below 1.2e-4 (truncation < 3e-13).

    NOTE: deliberately compiled WITHOUT fastmath -- its approximate-function flag
    degrades sin/cos/exp at these argument sizes.
    """
    q = (H[0, 0].real + H[1, 1].real + H[2, 2].real) / 3.0
    p1 = (H[0, 1].real**2 + H[0, 1].imag**2 +
          H[0, 2].real**2 + H[0, 2].imag**2 +
          H[1, 2].real**2 + H[1, 2].imag**2)
    d0 = H[0, 0].real - q
    d1 = H[1, 1].real - q
    d2 = H[2, 2].real - q
    p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)

    scale = np.abs(q) + p
    if p <= 1e-14 * max(scale, 1e-300):
        # H ~ q * I: the propagator is a pure global phase
        phase = np.exp(-1j * q * t)
        out.fill(0.0 + 0.0j)
        out[0, 0] = phase
        out[1, 1] = phase
        out[2, 2] = phase
        return

    # Trigonometric roots of the traceless D = H - q I
    b01 = H[0, 1]
    b02 = H[0, 2]
    b12 = H[1, 2]
    detB = (d0 * (d1 * d2 - (b12.real**2 + b12.imag**2))
            - (b01 * (np.conjugate(b01) * d2 - b12 * np.conjugate(b02))).real
            + (b02 * (np.conjugate(b01) * np.conjugate(b12) - d1 * np.conjugate(b02))).real)
    r = detB / (2.0 * p * p * p)
    if r < -1.0:
        r = -1.0
    elif r > 1.0:
        r = 1.0
    phi = np.arccos(r) / 3.0

    e_hi = 2.0 * p * np.cos(phi)
    e_lo = 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e_mid = -e_hi - e_lo

    spread = e_hi - e_lo
    global_phase = np.exp(-1j * q * t)

    if abs(spread * t) < 1.2e-4:
        # Tiny total phase spread: second-order Taylor in X = -i t D around the global
        # phase; truncation ||X||^3 / 6 < 3e-13. Covers every all-three-close case.
        for a in range(3):
            for b in range(3):
                x_ab = -1j * t * H[a, b]
                if a == b:
                    x_ab += 1j * t * q
                x2_ab = 0.0 + 0.0j
                for k in range(3):
                    x_ak = -1j * t * H[a, k]
                    x_kb = -1j * t * H[k, b]
                    if a == k:
                        x_ak += 1j * t * q
                    if k == b:
                        x_kb += 1j * t * q
                    x2_ab += x_ak * x_kb
                val = x_ab + 0.5 * x2_ab
                if a == b:
                    val += 1.0
                out[a, b] = global_phase * val
        return

    # Order so the closest pair is (a, b); c is the loner.
    if (e_hi - e_mid) <= (e_mid - e_lo):
        a_ = e_hi
        b_ = e_mid
        c_ = e_lo
    else:
        a_ = e_mid
        b_ = e_lo
        c_ = e_hi

    if abs(a_ - b_) < 1e-4 * spread:
        # DEFLATION: trig roots of a near-double pair are sqrt(eps)-conditioned; rebuild
        # the pair from the 2x2 complement of the (accurately known) far eigenvector.
        mean_pair = -0.5 * c_          # tr D = 0  =>  a + b = -c
        # Far eigenvector: largest column of M = (D - mean_pair I)^2; the pair's
        # contribution is (±gap/2)^2 vs (c - mean_pair)^2 for the far root.
        m00 = complex(d0 - mean_pair, 0.0)
        m11 = complex(d1 - mean_pair, 0.0)
        m22 = complex(d2 - mean_pair, 0.0)
        # M columns (M = Dm @ Dm with Dm Hermitian)
        v0 = np.zeros(3, dtype=np.complex128)
        best_norm = -1.0
        for col in range(3):
            if col == 0:
                y0 = m00
                y1 = np.conjugate(b01)
                y2 = np.conjugate(b02)
            elif col == 1:
                y0 = b01
                y1 = m11
                y2 = np.conjugate(b12)
            else:
                y0 = b02
                y1 = b12
                y2 = m22
            c0 = m00 * y0 + b01 * y1 + b02 * y2
            c1 = np.conjugate(b01) * y0 + m11 * y1 + b12 * y2
            c2 = np.conjugate(b02) * y0 + np.conjugate(b12) * y1 + m22 * y2
            nrm = (c0.real**2 + c0.imag**2 + c1.real**2 + c1.imag**2 + c2.real**2 + c2.imag**2)
            if nrm > best_norm:
                best_norm = nrm
                v0[0] = c0
                v0[1] = c1
                v0[2] = c2
        inv = 1.0 / np.sqrt(best_norm)
        v0[0] *= inv
        v0[1] *= inv
        v0[2] *= inv
        # Orthonormal complement {x1, x2}: Gram-Schmidt the least-aligned axis, then a
        # second axis orthogonalized against both.
        a0 = v0[0].real**2 + v0[0].imag**2
        a1 = v0[1].real**2 + v0[1].imag**2
        a2 = v0[2].real**2 + v0[2].imag**2
        if a0 <= a1 and a0 <= a2:
            k1 = 0
        elif a1 <= a2:
            k1 = 1
        else:
            k1 = 2
        x1 = np.zeros(3, dtype=np.complex128)
        x1[k1] = 1.0
        ov = np.conjugate(v0[k1])
        x1[0] -= ov * v0[0]
        x1[1] -= ov * v0[1]
        x1[2] -= ov * v0[2]
        n1 = np.sqrt(x1[0].real**2 + x1[0].imag**2 + x1[1].real**2 + x1[1].imag**2 + x1[2].real**2 + x1[2].imag**2)
        x1[0] /= n1
        x1[1] /= n1
        x1[2] /= n1
        k2 = 0 if k1 != 0 else 1
        x2 = np.zeros(3, dtype=np.complex128)
        x2[k2] = 1.0
        ov = np.conjugate(v0[k2])
        x2[0] -= ov * v0[0]
        x2[1] -= ov * v0[1]
        x2[2] -= ov * v0[2]
        ox = np.conjugate(x1[0]) * x2[0] + np.conjugate(x1[1]) * x2[1] + np.conjugate(x1[2]) * x2[2]
        x2[0] -= ox * x1[0]
        x2[1] -= ox * x1[1]
        x2[2] -= ox * x1[2]
        n2 = np.sqrt(x2[0].real**2 + x2[0].imag**2 + x2[1].real**2 + x2[1].imag**2 + x2[2].real**2 + x2[2].imag**2)
        x2[0] /= n2
        x2[1] /= n2
        x2[2] /= n2
        # 2x2 block of D in the {x1, x2} basis (D applied via its Hermitian elements)
        # D @ x vectors
        Dx1_0 = complex(d0, 0.0) * x1[0] + b01 * x1[1] + b02 * x1[2]
        Dx1_1 = np.conjugate(b01) * x1[0] + complex(d1, 0.0) * x1[1] + b12 * x1[2]
        Dx1_2 = np.conjugate(b02) * x1[0] + np.conjugate(b12) * x1[1] + complex(d2, 0.0) * x1[2]
        Dx2_0 = complex(d0, 0.0) * x2[0] + b01 * x2[1] + b02 * x2[2]
        Dx2_1 = np.conjugate(b01) * x2[0] + complex(d1, 0.0) * x2[1] + b12 * x2[2]
        Dx2_2 = np.conjugate(b02) * x2[0] + np.conjugate(b12) * x2[1] + complex(d2, 0.0) * x2[2]
        b11_ = (np.conjugate(x1[0]) * Dx1_0 + np.conjugate(x1[1]) * Dx1_1 + np.conjugate(x1[2]) * Dx1_2).real
        b22_ = (np.conjugate(x2[0]) * Dx2_0 + np.conjugate(x2[1]) * Dx2_1 + np.conjugate(x2[2]) * Dx2_2).real
        b12_ = (np.conjugate(x1[0]) * Dx2_0 + np.conjugate(x1[1]) * Dx2_1 + np.conjugate(x1[2]) * Dx2_2)
        mean_b = 0.5 * (b11_ + b22_)
        half_diff = 0.5 * (b11_ - b22_)
        delta = np.sqrt(half_diff * half_diff + b12_.real**2 + b12_.imag**2)
        a_ = mean_b + delta
        b_ = mean_b - delta

    f_a = np.exp(-1j * a_ * t)
    f_b = np.exp(-1j * b_ * t)
    f_c = np.exp(-1j * c_ * t)

    # First divided differences: sinc form only where the direct difference cancels
    # (|gap phase| < 0.5); the direct node-exact quotient everywhere else (mandatory at
    # large gap phases -- see docstring).
    u_half = 0.5 * (a_ - b_) * t
    if abs(u_half) < 0.5:
        sinc_ab = np.sin(u_half) / u_half if u_half != 0.0 else 1.0
        dd_ab = np.exp(-0.5j * (a_ + b_) * t) * (-1j * t) * sinc_ab
    else:
        dd_ab = (f_a - f_b) / (a_ - b_)

    v_half = 0.5 * (b_ - c_) * t
    if abs(v_half) < 0.5:
        sinc_bc = np.sin(v_half) / v_half if v_half != 0.0 else 1.0
        dd_bc = np.exp(-0.5j * (b_ + c_) * t) * (-1j * t) * sinc_bc
    else:
        dd_bc = (f_b - f_c) / (b_ - c_)

    dd_abc = (dd_ab - dd_bc) / (a_ - c_)

    # Newton -> power basis in D: U = e^{-iqt} (A2 D^2 + A1 D + A0 I)
    A2 = dd_abc
    A1 = dd_ab - (a_ + b_) * dd_abc
    A0 = f_a - a_ * dd_ab + a_ * b_ * dd_abc

    # Evaluate on D = H - q I without materializing it
    for a in range(3):
        for b in range(3):
            # D^2 element
            dd2 = 0.0 + 0.0j
            for k in range(3):
                h_ak = H[a, k]
                h_kb = H[k, b]
                if a == k:
                    h_ak -= q
                if k == b:
                    h_kb -= q
                dd2 += h_ak * h_kb
            d_ab = H[a, b]
            if a == b:
                d_ab -= q
            val = A2 * dd2 + A1 * d_ab
            if a == b:
                val += A0
            out[a, b] = global_phase * val

@njit(fastmath=True, cache=True)
def eigenvectors_3x3(H: np.ndarray) -> np.ndarray:
    """Eigenvectors of a Hermitian 3x3 H, columns ordered by ASCENDING eigenvalue --
    the same convention as np.linalg.eigh, whose call in the cosmological mixing loop
    this replaces (the adiabatic mass-eigenstate bookkeeping relies on that ordering
    staying consistent across every (E, z) evaluation).

    Method: closed-form trigonometric eigenvalues (identical to propagator_3x3), then
    for each eigenvalue lam_k the spectral projector numerator (H - lam_j I)(H - lam_l I)
    -- every nonzero column of which is proportional to the eigenvector -- taking the
    largest-norm column and normalizing. Column phases are arbitrary, exactly as they
    are for eigh; every downstream consumer uses only |U|^2 projections.

    Same precision guard as propagator_3x3: relative eigenvalue gaps below 1e-4 fall
    back to np.linalg.eigh (near an MSW-like level crossing the projector columns
    suffer the same catastrophic cancellation as the Lagrange denominators).
    """
    q = (H[0, 0].real + H[1, 1].real + H[2, 2].real) / 3.0
    p1 = (H[0, 1].real**2 + H[0, 1].imag**2 +
          H[0, 2].real**2 + H[0, 2].imag**2 +
          H[1, 2].real**2 + H[1, 2].imag**2)
    d0 = H[0, 0].real - q
    d1 = H[1, 1].real - q
    d2 = H[2, 2].real - q
    p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)

    scale = np.abs(q) + p
    if p <= 1e-14 * max(scale, 1e-300):
        # H ~ q * I: any orthonormal basis works; the identity matches eigh's output here.
        out = np.zeros((3, 3), dtype=np.complex128)
        out[0, 0] = 1.0
        out[1, 1] = 1.0
        out[2, 2] = 1.0
        return out

    b01 = H[0, 1]
    b02 = H[0, 2]
    b12 = H[1, 2]
    detB = (d0 * (d1 * d2 - (b12.real**2 + b12.imag**2))
            - (b01 * (np.conjugate(b01) * d2 - b12 * np.conjugate(b02))).real
            + (b02 * (np.conjugate(b01) * np.conjugate(b12) - d1 * np.conjugate(b02))).real)
    r = detB / (2.0 * p * p * p)
    if r < -1.0:
        r = -1.0
    elif r > 1.0:
        r = 1.0
    phi = np.arccos(r) / 3.0

    lam_hi = q + 2.0 * p * np.cos(phi)
    lam_lo = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    lam_mid = 3.0 * q - lam_hi - lam_lo

    g01 = abs(lam_hi - lam_mid)
    g02 = abs(lam_hi - lam_lo)
    g12 = abs(lam_mid - lam_lo)
    if min(g01, min(g02, g12)) < 1e-4 * (abs(lam_hi) + abs(lam_mid) + abs(lam_lo) + 1e-300):
        evals, evecs = np.linalg.eigh(H)
        return np.ascontiguousarray(evecs)

    out = np.empty((3, 3), dtype=np.complex128)
    lams_asc = (lam_lo, lam_mid, lam_hi)
    for k in range(3):
        lam_k = lams_asc[k]
        lam_a = lams_asc[(k + 1) % 3]
        lam_b = lams_asc[(k + 2) % 3]
        # M = (H - lam_a I)(H - lam_b I); columns proportional to the lam_k eigenvector
        best_norm = -1.0
        v0 = 0.0 + 0.0j
        v1 = 0.0 + 0.0j
        v2 = 0.0 + 0.0j
        for c in range(3):
            # column c of (H - lam_b I): H[:, c] - lam_b * e_c
            y0 = H[0, c]
            y1 = H[1, c]
            y2 = H[2, c]
            if c == 0:
                y0 -= lam_b
            elif c == 1:
                y1 -= lam_b
            else:
                y2 -= lam_b
            # m = (H - lam_a I) @ y
            m0 = (H[0, 0] - lam_a) * y0 + H[0, 1] * y1 + H[0, 2] * y2
            m1 = H[1, 0] * y0 + (H[1, 1] - lam_a) * y1 + H[1, 2] * y2
            m2 = H[2, 0] * y0 + H[2, 1] * y1 + (H[2, 2] - lam_a) * y2
            nrm = (m0.real**2 + m0.imag**2 + m1.real**2 + m1.imag**2 + m2.real**2 + m2.imag**2)
            if nrm > best_norm:
                best_norm = nrm
                v0 = m0
                v1 = m1
                v2 = m2
        inv_nrm = 1.0 / np.sqrt(best_norm)
        out[0, k] = v0 * inv_nrm
        out[1, k] = v1 * inv_nrm
        out[2, k] = v2 * inv_nrm
    return out

@njit("float64(float64, float64)", fastmath=True, cache=True)
def get_baseline_earth(d_km: float, costh: float) -> float:
    """Calculates the exact baseline distance traversed inside the Earth."""
    R_E = 6371.0
    return np.sqrt((R_E**2 - 2.0*R_E*d_km) * costh**2 + 2.0*R_E*d_km) - (R_E - d_km) * costh

# =============================================================================
# EXTRACTED NUMBA KERNELS FOR PROGRESS TRACKING
# =============================================================================

@njit(fastmath=True, cache=True, parallel=True)
def precompute_all_thermal_rates(E_bins: np.ndarray, z_bins: np.ndarray, sum_G: float,
                                 sum_G_alpha: np.ndarray, m_Zd: float, m_nu_bg_arr: np.ndarray,
                                 T_nu0: float, eta_mass: np.ndarray) -> np.ndarray:
    """Pre-computes the thermal attenuation rates for the entire energy and redshift grid.

    Every (k, i) redshift/energy pair is an independent evaluation writing to a disjoint
    Gamma_grid[k, i, :] slice, so the outer loop is parallelized across threads via prange.
    """
    N_E = len(E_bins)
    N_z = len(z_bins)
    Gamma_grid = np.zeros((N_z, N_E, 3), dtype=np.float64)
    for k in prange(N_z):
        for i in range(N_E):
            Gamma_grid[k, i, :] = thermal_averaged_rates_flavor(E_bins[i], z_bins[k], sum_G, sum_G_alpha, m_Zd, m_nu_bg_arr, T_nu0, eta_mass)
    return Gamma_grid

@njit(fastmath=True, cache=True, parallel=True)
def _interp_thermal_grid_z(Gamma_sub: np.ndarray, N_z: int) -> np.ndarray:
    """Linearly interpolates a thermal-rate grid computed on a coarse uniform z-subgrid
    onto the full uniform z-grid. Both grids are uniform linspaces over the same
    [z_max, 0] range (in the same direction), so the fractional-index mapping
    t = k * (n_sub - 1) / (N_z - 1) is exact -- no searching required."""
    n_sub = Gamma_sub.shape[0]
    N_E = Gamma_sub.shape[1]
    out = np.empty((N_z, N_E, 3), dtype=np.float64)
    scale = (n_sub - 1.0) / (N_z - 1.0)
    for k in prange(N_z):
        t = k * scale
        i0 = int(t)
        if i0 > n_sub - 2:
            i0 = n_sub - 2
        w = t - i0
        for i in range(N_E):
            for a in range(3):
                out[k, i, a] = (1.0 - w) * Gamma_sub[i0, i, a] + w * Gamma_sub[i0 + 1, i, a]
    return out

def build_thermal_grid(E_bins: np.ndarray, z_bins: np.ndarray, sum_G: float,
                       sum_G_alpha: np.ndarray, m_Zd: float, m_nu_bg_arr: np.ndarray,
                       T_nu0: float, eta_mass: np.ndarray, n_z_thermal: int = 0) -> np.ndarray:
    """Builds the (N_z, N_E, 3) thermal attenuation-rate grid, optionally via a coarse
    z-subgrid plus linear interpolation.

    Each grid entry is a full Breit-Wigner momentum integration
    (thermal_averaged_rates_flavor), making the exact N_z x N_E precompute one of the
    most expensive per-evaluation costs at large N_z.

    INTERPOLATION IS OFF BY DEFAULT (n_z_thermal = 0 -> exact grid), deliberately: the
    tempting smoothness argument -- the (1+z) scalings sweep a thermally-broadened
    resonance slowly through the band -- FAILS for the heavier CnuB mass states, which
    are non-relativistic (p ~ T << m), so their resonance condition s ~ 2 E (1+z) m_i
    is met over a z-window of relative width only ~ p/m_i ~ 0.4-2%. Measured at
    production settings (N_z = 1000, subgrid 201): up to 3.5e-3 absolute flavor-ratio
    error and ~10% pointwise flux error in the absorption dip at a below-band
    resonance point (m_Zd = 3e5 eV), vs ~4e-5 ratio error at an in-band point. Enable
    it (config key grid.n_z_thermal, CLI --n_z_thermal) only when that error budget is
    acceptable for your parameter region, or with a subgrid fine enough to resolve the
    narrow-in-z resonance of the heaviest background state.

    n_z_thermal: number of coarse z-samples. Values < 2, or >= the full N_z, disable
    interpolation and compute the exact grid.
    """
    N_z = len(z_bins)
    n_sub = int(n_z_thermal)
    if n_sub < 2 or n_sub >= N_z:
        return precompute_all_thermal_rates(E_bins, z_bins, sum_G, sum_G_alpha, m_Zd,
                                            m_nu_bg_arr, T_nu0, eta_mass)
    z_sub = np.linspace(z_bins[0], z_bins[-1], n_sub)
    Gamma_sub = precompute_all_thermal_rates(E_bins, z_sub, sum_G, sum_G_alpha, m_Zd,
                                             m_nu_bg_arr, T_nu0, eta_mass)
    return _interp_thermal_grid_z(np.ascontiguousarray(Gamma_sub), N_z)

@njit(fastmath=True, cache=True, parallel=True)
def step_cosmological_propagation(z: float, dt: float, R_z: float, E_bins: np.ndarray,
                                  F_mass: np.ndarray, rho_src: np.ndarray,
                                  M_rot: np.ndarray,
                                  u_arr: np.ndarray, U_LRI_struct: np.ndarray,
                                  m_Zd: float, g_D: float, eps_prime: float,
                                  n_n0: float, T_nu0: float,
                                  use_debye_screening: bool, is_antineutrino: bool,
                                  p_index: float, Phi_0: float, eta_bg: np.ndarray,
                                  BR: np.ndarray, W_kernel: np.ndarray,
                                  F_flavor: np.ndarray, Gamma_flavor: np.ndarray,
                                  dF_casc_flavor: np.ndarray, P_all: np.ndarray,
                                  rate_F_sum_arr: np.ndarray, precomputed_Gamma_flavor: np.ndarray):
    """Executes a single cosmological integration step using pre-allocated buffers and pre-computed rates.

    Each of the four loops below only ever reads state fully computed by the *previous*
    loop and writes to a disjoint per-i (or per-j) slice of its own output buffer, so each
    loop is independently safe to parallelize across energy bins via prange -- there is no
    dependency across bins within a single z-step (the physical z-recursion itself, across
    calls to this function, remains sequential).
    """
    N_E = len(E_bins)

    # Zero out buffers at start of step
    F_flavor.fill(0.0)
    Gamma_flavor.fill(0.0)
    dF_casc_flavor.fill(0.0)
    rate_F_sum_arr.fill(0.0)

    # First loop: Calculate flavor composition and assign pre-computed interaction rates.
    # Only the projection probabilities |U|^2 are ever consumed downstream (here and in
    # the fourth loop), so they are computed once per (i, z) into the P_all buffer
    # instead of storing the complex mixing matrix and re-deriving |U|^2 element-wise in
    # both loops as before. np.abs(.)**2 is kept verbatim so the stored values are
    # bit-identical to the previous inline computation.
    for i in prange(N_E):
        U_z = get_adiabatic_mixing_cosmo(E_bins[i], z, M_rot, u_arr, U_LRI_struct, m_Zd, g_D, eps_prime,
                                         n_n0, T_nu0, use_debye_screening, is_antineutrino, eta_bg)
        for a in range(3):
            for k in range(3):
                P_all[i, a, k] = np.abs(U_z[a, k])**2
            F_flavor[i, a] = (P_all[i, a, 0] * F_mass[i, 0]
                              + P_all[i, a, 1] * F_mass[i, 1]
                              + P_all[i, a, 2] * F_mass[i, 2])

        # Use the pre-computed grid values instead of the thermal kernel
        Gamma_flavor[i, :] = precomputed_Gamma_flavor[i, :]

    # Second loop: Energy cascading integral, reduced from O(N_E^2) to O(N_E).
    # The cascading kernel W(E_i, E_j) = 3 E_i^2 * dE_j / E_j^3 is separable: the entire
    # i-dependence is the prefactor 3 E_i^2, so the per-bin integral
    #     integral_i = sum_{j >= i} rate_F_sum[j] * W_kernel[j]
    # is a pure suffix sum over a fixed vector. The previous formulation recomputed each
    # suffix independently -- N_E^2/2 multiply-adds per z-step, the single dominant cost
    # of the whole cosmological phase at the production 1000x1000 grid (10^9 operations
    # per run) -- where one reversed sweep produces every suffix in N_E operations.
    # rate_F_sum_arr[j] now stores the suffix value sum_{k >= j} rate*W rather than the
    # per-bin summand; nothing downstream reads the old meaning.
    suffix = 0.0
    for j in range(N_E - 1, -1, -1):
        s = (Gamma_flavor[j, 0] * F_flavor[j, 0]
             + Gamma_flavor[j, 1] * F_flavor[j, 1]
             + Gamma_flavor[j, 2] * F_flavor[j, 2])
        suffix += s * W_kernel[j]
        rate_F_sum_arr[j] = suffix

    # Third loop: apply the i-dependent prefactor and branching ratios
    for i in prange(N_E):
        casc_i = 3.0 * (E_bins[i]**2) * rate_F_sum_arr[i]
        for a in range(3):
            dF_casc_flavor[i, a] = BR[a] * casc_i

    # Fourth loop: Update mass eigenstate flux with decay and injection (projections
    # read from the P_all buffer filled in the first loop; summation order over flavors
    # is unchanged, so the arithmetic is identical).
    for i in prange(N_E):
        E_z = E_bins[i] * (1.0 + z)
        inj_spectrum = Phi_0 * (E_z**(-p_index))
        for a in range(3):
            Gamma_mass_a = (P_all[i, 0, a] * Gamma_flavor[i, 0]
                            + P_all[i, 1, a] * Gamma_flavor[i, 1]
                            + P_all[i, 2, a] * Gamma_flavor[i, 2])
            dF_casc_mass_a = (P_all[i, 0, a] * dF_casc_flavor[i, 0]
                              + P_all[i, 1, a] * dF_casc_flavor[i, 1]
                              + P_all[i, 2, a] * dF_casc_flavor[i, 2])
            inj_rate_mass_a = (P_all[i, 0, a] * rho_src[0]
                               + P_all[i, 1, a] * rho_src[1]
                               + P_all[i, 2, a] * rho_src[2]) * inj_spectrum * R_z
            total_rate = dF_casc_mass_a + inj_rate_mass_a
            
            if Gamma_mass_a > 1e-30:
                decay = np.exp(-Gamma_mass_a * dt)
                # Numerically stable calculation for (1 - exp(-x)) / x
                # We use -np.expm1(-x) which is equivalent to 1 - exp(-x) but stable near 0
                term = total_rate * (-np.expm1(-Gamma_mass_a * dt) / Gamma_mass_a)
                F_mass[i, a] = F_mass[i, a] * decay + term
            else:
                F_mass[i, a] += total_rate * dt

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

        # T = U @ rho_i ; rho_i = T @ U^dagger  (hand-rolled, allocation-free)
        for a in range(3):
            for b in range(3):
                T_buf[a, b] = (U_buf[a, 0] * rho[i, 0, b]
                               + U_buf[a, 1] * rho[i, 1, b]
                               + U_buf[a, 2] * rho[i, 2, b])
        for a in range(3):
            for b in range(3):
                rho[i, a, b] = (T_buf[a, 0] * np.conjugate(U_buf[b, 0])
                                + T_buf[a, 1] * np.conjugate(U_buf[b, 1])
                                + T_buf[a, 2] * np.conjugate(U_buf[b, 2]))

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

def _earth_chord_worker(args) -> np.ndarray:
    """Propagates a single Earth chord (one zenith angle) end-to-end to the detector.

    Module-level (picklable) so it can be dispatched to a multiprocessing pool: each
    chord is fully independent of the others, so this is the unit of parallel work for
    the Earth phase of the spatial handover.
    """
    (costh, baseline_km, N_steps_earth, E_bins, rho_surf_avg, M_rot, n_n_amb_eV3, Coef,
     U_LRI_struct, V_SM, detector_depth_km) = args

    dr_earth_km = baseline_km / N_steps_earth
    dr_earth_eV_inv = dr_earth_km * 1e3 * m_to_eV_inv
    rho_earth = rho_surf_avg.copy()
    U_step_buffer = np.zeros((3, 3), dtype=np.complex128)
    H_buf = np.zeros((3, 3), dtype=np.complex128)
    T_buf = np.zeros((3, 3), dtype=np.complex128)
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
    roughly 10x *slower*, not faster (measured directly). Passing None here (the default)
    preserves the exact original per-call pool creation behavior for every existing caller.
    """
    if n_workers > 1 and len(tasks) > 1:
        if external_pool is not None:
            iterator = external_pool.imap(worker_fn, tasks)
            if HAS_TQDM and show_progress:
                iterator = tqdm(iterator, total=len(tasks), desc=desc, leave=False)
            results = list(iterator)
        else:
            with multiprocessing.Pool(processes=min(n_workers, len(tasks))) as pool:
                iterator = pool.imap(worker_fn, tasks)
                if HAS_TQDM and show_progress:
                    iterator = tqdm(iterator, total=len(tasks), desc=desc, leave=False)
                results = list(iterator)
    else:
        iterator = tqdm(tasks, desc=desc, leave=False) if (HAS_TQDM and show_progress) else tasks
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
                            U_LRI_struct=None, V_SM_struct=None, use_gpu: bool = False) -> np.ndarray:
    """
    Executes pure spatial ODE through Milky Way and Earth with active tracking.
    Milky Way trajectories and Earth chords are mutually independent, so when n_cores > 1
    each set is distributed across a multiprocessing pool instead of being run sequentially.

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

    # NOTE: An earlier version of this code scaled n_steps_earth down for shorter chords
    # (same physical km/step as the costh=-1 chord, since that seemed to only cost
    # resolution nobody needed -- verified via a synthetic, purely-diagonal test state to
    # agree with a 2000-step reference to 1e-12). That test state was misleading: the real
    # rho_surf_avg arriving from the Milky Way phase carries off-diagonal coherences, and at
    # production defaults this showed a genuine, energy-localized ~10% error for the shorter
    # chords, concentrated at the exact energy where the Earth matter potential drives a
    # near-degenerate (MSW-like) eigenvalue crossing -- near such a crossing, eigenvectors
    # rotate rapidly with position *independent of total path length*, so "shorter chord
    # needs fewer steps" is the wrong criterion. Fixed step count for every chord (now
    # caller-configurable via N_steps_earth, default 500 -- unchanged from before) until a
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
                    detector_depth_km) for costh in earth_costh for (b0, b1) in earth_blocks]
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


@njit(cache=True)
def compute_gamma_grid_inputs(U_pmns: np.ndarray, u_arr: np.ndarray, g_D: float, eps_prime: float,
                              eta_bg: np.ndarray):
    """Computes (sum_G, sum_G_alpha, eta_mass): the only inputs precompute_all_thermal_rates
    needs beyond the (E, z) grid itself.

    These depend only on |U_pmns| and |u_arr| element-wise magnitudes, which are identical
    for neutrinos and antineutrinos (conjugation never changes a magnitude) -- so the
    resulting Gamma_grid_all is exactly the same for both, and callers that evaluate both
    flavors (e.g. is_antineutrino="sum") should compute it once and share it rather than
    calling precompute_all_thermal_rates twice for an identical result.
    """
    eta_mass = np.zeros(3, dtype=np.float64)
    for i in range(3):
        eta_mass[i] = np.sum(np.abs(U_pmns[:, i])**2 * eta_bg)

    u_mag = np.abs(u_arr)
    sum_G = 0.0
    sum_G_alpha = np.zeros(3, dtype=np.float64)
    for a in range(3):
        for g in range(3):
            coupling_prod = u_mag[a] * u_mag[g]
            val = g_D * coupling_prod * (g_D * coupling_prod + eps_prime * (0.65 / 0.88))
            sum_G += val
            sum_G_alpha[a] += val

    return sum_G, sum_G_alpha, eta_mass

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

def propagate_spectrum(E_bins: np.ndarray, z_bins: np.ndarray, rho_src: np.ndarray,
                       H0: float, Omega_m: float, Omega_L: float,
                       U_pmns: np.ndarray, M_sq: np.ndarray,
                       u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
                       m_nu_bg_arr: np.ndarray, n_n0: float, T_nu0: float,
                       use_debye_screening: bool, detector_depth_km: float,
                       use_SM_potential: bool, n_e_earth: float,
                       is_antineutrino: bool, p_index: float, rho_evol: float,
                       Phi_0: float, eta_bg: np.ndarray, n_mw_paths: int, n_earth_chords: int,
                       N_steps_MW: int = 25, r_start_kpc: float = 100.0,
                       show_progress: bool = True, verbose: int = 1, n_cores: int = 1,
                       precomputed_Gamma_grid_all: np.ndarray = None, N_steps_earth: int = 500,
                       external_pool=None, U_LRI_struct=None, V_SM_struct=None,
                       n_z_thermal: int = 0, use_gpu: bool = False) -> np.ndarray:
    """
    Executes the complete spectrum integration, driving the underlying JIT kernels.
    Includes verbose level 2 for periodic redshift status updates.

    U_LRI_struct / V_SM_struct, if given, are dimensionless 3x3 Hermitian flavor structure
    matrices overriding the default LRI texture (u_i * conj(u_j)) and SM matter-potential
    texture (diag(1, 0, 0)). Pass them in NEUTRINO convention regardless of
    is_antineutrino: the antineutrino conjugation (V -> -conj(V)) is applied centrally
    here, matching how u_arr/U_pmns arrive pre-conjugated from propagation_worker. The
    overrides affect only the coherent forward-scattering potentials -- the Debye
    screening mass and the thermal attenuation/regeneration rates remain tied to the
    u_e/u_mu/u_tau amplitudes (config keys: dark_portal.U_LRI_struct,
    standard_mixing.V_SM_struct).

    external_pool, if given, is a caller-managed multiprocessing.Pool passed straight through
    to propagate_earth_spatial, reused across many calls instead of creating a fresh one each
    time -- see _run_spatial_tasks's docstring for why this matters under 'spawn'.

    precomputed_Gamma_grid_all lets a caller that evaluates both neutrinos and antineutrinos
    (e.g. is_antineutrino="sum") compute the thermal-rate grid once and pass it to both calls,
    since it is identical for both (see compute_gamma_grid_inputs).

    N_steps_earth controls the number of path-length steps per Earth chord (analogous to
    N_steps_MW for the Milky Way phase). The 500 default is the only value validated against
    resonance-crossing sensitivity (see propagate_earth_spatial's docstring) -- lowering it
    trades away exactly the resolution that protects against silently wrong results near a
    matter resonance; verify convergence for your own m_Zd range before trusting a lower value.
    """
    if HAS_NUMBA:
        # The cosmological step below is parallelized internally via numba prange; cap its
        # thread pool to n_cores so a caller that already parallelizes across many processes
        # at a higher level (dark_portal_scanner.py, FrequentistAnalysis's Minuit/emcee
        # workers -- both of which pass n_cores=1 here) doesn't also oversubscribe every core
        # from inside each of its own worker processes.
        numba.set_num_threads(max(1, int(n_cores)))

    N_E = len(E_bins)
    dz = z_bins[0] - z_bins[1]
    lbl = "Anti-Nu" if is_antineutrino else "Nu"

    # U_pmns and M_sq never change within this call, so the vacuum-oscillation matrix product
    # U_pmns @ M_sq @ conj(U_pmns).T is precomputed exactly once here instead of being
    # recomputed (identically) on every single (E, z) evaluation downstream -- see
    # get_adiabatic_mixing_cosmo's docstring for where this is consumed.
    M_rot = np.ascontiguousarray(U_pmns @ M_sq @ np.ascontiguousarray(np.conjugate(U_pmns).T))

    # Resolve the LRI flavor structure once for the whole call. The default is the outer
    # product of the sterile mixing vector; since u_arr arrives pre-conjugated for
    # antineutrinos, building it here lands in the right convention automatically. A user
    # override arrives in neutrino convention and is conjugated here (single-transform
    # convention: the downstream kernels then only flip the sign).
    if U_LRI_struct is None:
        U_LRI_struct_eff = np.outer(u_arr, np.conjugate(u_arr))
    else:
        U_LRI_struct_eff = np.conjugate(U_LRI_struct) if is_antineutrino else U_LRI_struct
    U_LRI_struct_eff = np.ascontiguousarray(U_LRI_struct_eff, dtype=np.complex128)

    V_SM_struct_eff = None
    if V_SM_struct is not None:
        V_SM_struct_eff = np.conjugate(V_SM_struct) if is_antineutrino else V_SM_struct
        V_SM_struct_eff = np.ascontiguousarray(V_SM_struct_eff, dtype=np.complex128)

    sum_G, sum_G_alpha, eta_mass = compute_gamma_grid_inputs(U_pmns, u_arr, g_D, eps_prime, eta_bg)

    F_mass = np.zeros((N_E, 3), dtype=np.float64)

    BR = np.zeros(3, dtype=np.float64)
    if sum_G > 0:
        BR = sum_G_alpha / sum_G

    dE = compute_dE_geometric(E_bins)

    # The cascading integral's kernel dE[j]/E_bins[j]^3 depends only on the (fixed) energy
    # grid, never on i or z, but was previously recomputed inside the O(N_E^2) inner loop of
    # step_cosmological_propagation for every (i, j) pair. Precomputing it once here removes
    # a division and a cube from what is otherwise the single most expensive loop in the solver.
    W_kernel = dE / E_bins**3


    # Pre-allocated buffers for the compiled Numba stepper to prevent garbage-collection throttling
    F_flavor = np.zeros((N_E, 3), dtype=np.float64)
    Gamma_flavor = np.zeros((N_E, 3), dtype=np.float64)
    dF_casc_flavor = np.zeros((N_E, 3), dtype=np.float64)
    P_all = np.zeros((N_E, 3, 3), dtype=np.float64)
    rate_F_sum_arr = np.zeros(N_E, dtype=np.float64)

    # Pre-compute the thermal interaction rates for the entire grid before sequential stepping.
    # Reuse the caller's precomputed grid when given one (see the docstring above) instead of
    # redoing this identical, expensive O(N_z * N_E) calculation for the antineutrino call.
    if precomputed_Gamma_grid_all is not None:
        Gamma_grid_all = precomputed_Gamma_grid_all
    else:
        if verbose == 2:
            safe_print(f"Pre-computing thermal interaction rate grids...")
        Gamma_grid_all = build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, m_Zd, m_nu_bg_arr, T_nu0, eta_mass,
                                            n_z_thermal=n_z_thermal)

    iterator = tqdm(range(1, len(z_bins)), desc=f"Cosmo Phase ({lbl})", leave=False) if (HAS_TQDM and show_progress) else range(1, len(z_bins))
    z_print_mod = max(1, len(z_bins) // 10)

    if verbose == 2:
        safe_print(f"\n--- Cosmological Integration Started (z_max = {z_bins[0]:.2f}) ({lbl}) ---")

    for k in iterator:
        z = z_bins[k]
        dt = dz / (calc_H_Hubble(z, H0, Omega_m, Omega_L) * (1.0 + z))
        R_z = calc_R_source(z, rho_evol)

        if verbose == 2 and k % z_print_mod == 0:
            safe_print(f"Cosmological propagation: currently integrating at redshift z = {z:.4f}")

        step_cosmological_propagation(z, dt, R_z, E_bins, F_mass, rho_src, M_rot, u_arr,
                                      U_LRI_struct_eff,
                                      m_Zd, g_D, eps_prime, n_n0, T_nu0,
                                      use_debye_screening, is_antineutrino, p_index, Phi_0,
                                      eta_bg, BR, W_kernel,
                                      F_flavor, Gamma_flavor, dF_casc_flavor, P_all, rate_F_sum_arr,
                                      Gamma_grid_all[k, :, :])

    if verbose == 2:
        safe_print(f"Cosmological baseline finished at z = {z_bins[-1]:.4f}.")

    F_earth = propagate_earth_spatial(E_bins, F_mass, M_rot, u_arr, m_Zd, g_D, eps_prime,
                                      T_nu0, use_debye_screening, eta_bg, detector_depth_km,
                                      use_SM_potential, n_e_earth, is_antineutrino, n_mw_paths, n_earth_chords,
                                      N_steps_MW=N_steps_MW, r_start_kpc=r_start_kpc,
                                      show_progress=show_progress, verbose=verbose, n_cores=n_cores,
                                      N_steps_earth=N_steps_earth, external_pool=external_pool,
                                      U_LRI_struct=U_LRI_struct_eff, V_SM_struct=V_SM_struct_eff,
                                      use_gpu=use_gpu)
            
    return F_earth

# =============================================================================
# FILE EXPORT ROUTINE
# =============================================================================

def write_run_manifest(filename: str, config: Dict, runtime_s: float = None):
    """Writes a reproducibility manifest (JSON) alongside an output file: the full
    resolved configuration, the git commit of the code that produced it, package
    versions, host, timestamp, and runtime. Paper-level numbers should be traceable to
    an exact code + configuration state; this makes that automatic."""
    import platform
    import subprocess
    import datetime

    git_hash = None
    try:
        git_hash = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass

    versions = {"python": platform.python_version(), "numpy": np.__version__}
    if HAS_NUMBA:
        versions["numba"] = numba.__version__

    manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_hash,
        "hostname": platform.node(),
        "versions": versions,
        "runtime_s": runtime_s,
        "config": config,
    }
    with open(filename, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    safe_print(f"Run manifest saved to: {os.path.abspath(filename)}")

def save_flux_to_file(E_array: np.ndarray, Flux_array: np.ndarray, filename: str = "../output/data/output_flux.txt"):
    """Saves the generated flux components (Energy, nue, numu, nutau) to an external text file."""
    header = "Energy[eV]    Flux_nue[eV^-1 cm^-2 s^-1 sr^-1]    Flux_numu[eV^-1 cm^-2 s^-1 sr^-1]    Flux_nutau[eV^-1 cm^-2 s^-1 sr^-1]"
    data = np.column_stack((E_array, Flux_array))
    np.savetxt(filename, data, header=header, fmt="%.6e")
    safe_print(f"\nFlux data saved to: {os.path.abspath(filename)}")

# =============================================================================
# MULTIPROCESSING WRAPPER & CONFIGURATION MANAGER
# =============================================================================

def parse_structure_matrix_3x3(entry, name: str) -> np.ndarray:
    """Parses a JSON-friendly 3x3 flavor structure matrix into a complex Hermitian ndarray.

    Each of the nine entries may be either a plain number (real) or a two-element list
    [re, im] (complex) -- JSON has no native complex type. The result must be Hermitian
    to machine precision relative to its own scale: a non-Hermitian potential would make
    the eigh-based propagation silently wrong (eigh only reads one triangle), so this
    raises loudly instead. Used for the config keys dark_portal.U_LRI_struct and
    standard_mixing.V_SM_struct.
    """
    arr = np.zeros((3, 3), dtype=np.complex128)
    try:
        if len(entry) != 3:
            raise ValueError
        for i in range(3):
            row = entry[i]
            if len(row) != 3:
                raise ValueError
            for j in range(3):
                v = row[j]
                if isinstance(v, (list, tuple)):
                    if len(v) != 2:
                        raise ValueError
                    arr[i, j] = complex(float(v[0]), float(v[1]))
                else:
                    arr[i, j] = complex(float(v), 0.0)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a 3x3 nested list; each entry a number or a two-element [re, im] list."
        )

    scale = np.max(np.abs(arr))
    if scale > 0 and np.max(np.abs(arr - np.conjugate(arr).T)) > 1e-12 * scale:
        raise ValueError(
            f"{name} must be Hermitian (M == conj(M).T): the eigh-based propagation reads only "
            f"one triangle of the Hamiltonian, so a non-Hermitian structure would be silently wrong."
        )
    return arr

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

def propagation_worker(args) -> Tuple[np.ndarray, np.ndarray]:
    """Unpacks arguments and executes the physics simulation."""
    E_bins, params = args
    show_progress = params.get('show_progress', True)
    verbose = params.get('verbose', 1)
    
    z_bins = np.linspace(params['z_max'], 0.0, int(params['n_z_bins']))
    rho_src = np.array([params['f_e_S'], params['f_mu_S'], params['f_tau_S']], dtype=np.float64)
    
    ordering = str(params.get('mass_ordering', 'normal')).strip().lower()
    dm31_sq_signed = -params['dm31_sq'] if ordering == "inverted" else params['dm31_sq']
    
    # Enforce C-contiguous alignment for the initial matrix state
    M_sq = np.ascontiguousarray(np.diag([0.0, params['dm21_sq'], dm31_sq_signed]).astype(np.complex128))
    
    u_arr_base = np.ascontiguousarray(np.array([params['u_e'], params['u_mu'], params['u_tau']], dtype=np.complex128))
    eta_bg = params['eta_bg']
    U_pmns_base = np.ascontiguousarray(construct_PMNS(params['th12'], params['th13'], params['th23'], params['d_cp']))

    # Pin numba's thread pool BEFORE the thermal-grid precompute below: that prange kernel
    # previously ran ahead of propagate_spectrum's own set_num_threads call, so inside a
    # scanner pool worker (n_cores=1) the first evaluation used numba's default thread
    # count (all cores) -- transient oversubscription multiplied across every worker.
    if HAS_NUMBA:
        numba.set_num_threads(max(1, int(params.get('n_cores', 1))))

    # The thermal-rate grid depends only on |U_pmns| and |u_arr| magnitudes, which
    # conjugation never changes -- so it is identical for the neutrino and antineutrino
    # calls below. Computing it once here and sharing it (instead of letting each flavor
    # call redo this expensive O(N_z * N_E) precomputation from scratch) roughly halves
    # that stage's cost whenever both flavors are evaluated (is_antineutrino="sum", the
    # default). build_thermal_grid can additionally sample z on a coarse subgrid and
    # interpolate (grid.n_z_thermal; OFF by default -- see its docstring for the measured
    # error budget before enabling).
    sum_G, sum_G_alpha, eta_mass = compute_gamma_grid_inputs(U_pmns_base, u_arr_base, params['g_D'], params['epsilon_prime'], eta_bg)
    Gamma_grid_all = build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, params['m_Zd_eV'], params['m_nu_bg_arr'], params['T_nu0_eV'], eta_mass,
                                        n_z_thermal=params.get('n_z_thermal', 0))

    def compute_flux(is_anti: bool) -> np.ndarray:
        # Enforce C-contiguous alignment when passing matrices to Numba kernels
        U_pmns = np.ascontiguousarray(np.conjugate(U_pmns_base)) if is_anti else U_pmns_base
        u_arr = np.ascontiguousarray(np.conjugate(u_arr_base)) if is_anti else u_arr_base

        return propagate_spectrum(
            E_bins, z_bins, rho_src,
            params['H0_eV'], params['Omega_m'], params['Omega_L'],
            U_pmns, M_sq, u_arr, params['m_Zd_eV'], params['g_D'], params['epsilon_prime'],
            params['m_nu_bg_arr'], params['n_n0_eV3'], params['T_nu0_eV'],
            params['use_debye_screening'], params['detector_depth_km'],
            params['use_SM_potential'], params['n_e_earth_eV3'],
            is_anti, params['p_index'], params['rho_evol'], params['Phi_0'], eta_bg,
            params['n_mw_paths'], params['n_earth_chords'],
            params['N_steps_MW'], params['r_start_kpc'],
            show_progress=show_progress, verbose=verbose, n_cores=params.get('n_cores', 1),
            precomputed_Gamma_grid_all=Gamma_grid_all, N_steps_earth=params.get('N_steps_earth', 500),
            U_LRI_struct=params.get('U_LRI_struct', None), V_SM_struct=params.get('V_SM_struct', None),
            use_gpu=params.get('use_gpu', False)
        )

    flag_val = str(params.get('is_antineutrino', 'sum')).strip().lower()
    
    if flag_val in ['sum', 'both']:
        ratio = params.get('nubar_to_nu_ratio', 1.0)
        # Degenerate weights: skip the propagation whose weight is exactly zero instead
        # of computing a full spectrum and multiplying it away -- a free 2x for
        # nu-only/nubar-only studies configured through the ratio.
        if ratio == 0.0:
            flux_at_earth = compute_flux(False)
        elif not np.isfinite(ratio):
            flux_at_earth = compute_flux(True)
        else:
            w_nu = 1.0 / (1.0 + ratio)
            w_nubar = ratio / (1.0 + ratio)
            flux_at_earth = w_nu * compute_flux(False) + w_nubar * compute_flux(True)
    elif flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        flux_at_earth = compute_flux(True)
    else:
        flux_at_earth = compute_flux(False)
    
    return E_bins, flux_at_earth

def run_solver(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Orchestrates configuration and parallelization.

    The incoming config is first overlaid onto DEFAULT_CONFIG with unknown-key
    rejection (see merge_config_with_defaults), so every read below can index
    directly -- no per-call fallback literals to drift out of sync.
    """
    config = merge_config_with_defaults(config)
    params = {}
    params['H0_eV'] = float((config['cosmology']['H0_km_s_Mpc'] * km_per_s) / Mpc_to_eV_inv)
    params['Omega_m'] = float(config['cosmology']['Omega_m'])
    params['Omega_L'] = float(config['cosmology']['Omega_L'])
    params['T_nu0_eV'] = float(config['cosmology']['T_nu0_eV'])
    params['n_n0_eV3'] = float(config['cosmology']['n_n0_cm3'] * cm_inv3_to_eV3)
    params['mass_ordering'] = config['cosmology']['mass_ordering']
    mo = str(config['cosmology']['mass_ordering']).strip().lower()
    if mo not in ['normal', 'inverted']:
        raise ValueError(f"mass_ordering must be strictly 'normal' or 'inverted'. Received: {mo}")
    
    params['eta_bg'] = np.array([
        float(config['cosmology']['eta_e']),
        float(config['cosmology']['eta_mu']),
        float(config['cosmology']['eta_tau'])
    ], dtype=np.float64)
    if config['cosmology']['eta_e'] < 0 or config['cosmology']['eta_mu'] < 0 or config['cosmology']['eta_tau'] < 0:
        raise ValueError("CnuB flavor asymmetries (eta) must be >= 0 to avoid complex Debye screening masses.")

    params['dm21_sq'] = float(config['standard_mixing']['dm21_sq'])
    params['dm31_sq'] = float(config['standard_mixing']['dm31_sq'])
    params['th12'] = float(np.radians(config['standard_mixing']['th12_deg']))
    params['th13'] = float(np.radians(config['standard_mixing']['th13_deg']))
    params['th23'] = float(np.radians(config['standard_mixing']['th23_deg']))
    params['d_cp'] = float(np.radians(config['standard_mixing']['d_cp_deg']))
    params['use_SM_potential'] = config['standard_mixing']['use_SM_potential']
    params['n_e_earth_eV3'] = float(config['standard_mixing']['n_e_earth_cm3'] * cm_inv3_to_eV3)
    # Optional flavor-structure overrides for the coherent forward-scattering potentials
    # (neutrino convention; antineutrino conjugation is applied centrally in
    # propagate_spectrum). See parse_structure_matrix_3x3 for the accepted JSON format.
    params['V_SM_struct'] = None
    if config['standard_mixing']['V_SM_struct'] is not None:
        params['V_SM_struct'] = parse_structure_matrix_3x3(
            config['standard_mixing']['V_SM_struct'], "standard_mixing.V_SM_struct")
    params['U_LRI_struct'] = None
    if config['dark_portal']['U_LRI_struct'] is not None:
        params['U_LRI_struct'] = parse_structure_matrix_3x3(
            config['dark_portal']['U_LRI_struct'], "dark_portal.U_LRI_struct")
    params['detector_depth_km'] = float(config['standard_mixing']['detector_depth_km'])
    depth_km = float(config['standard_mixing']['detector_depth_km'])
    if depth_km < 0.0 or depth_km > 6371.0:
        raise ValueError(f"detector_depth_km ({depth_km}) must be between 0 and the Earth's radius (6371.0 km).")

    params['m_nu_bg_arr'] = get_mass_eigenvalues(
        float(config['cosmology']['m_lightest_eV']),
        float(params['dm21_sq']),
        float(params['dm31_sq']),
        params['mass_ordering']
    )
    
    params['u_e'] = float(config['dark_portal']['u_e'])
    params['u_mu'] = float(config['dark_portal']['u_mu'])
    params['u_tau'] = float(config['dark_portal']['u_tau'])
    u_sq_sum = np.abs(params['u_e'])**2 + np.abs(params['u_mu'])**2 + np.abs(params['u_tau'])**2
    if u_sq_sum > 1.0:
        raise ValueError(f"Sterile mixing elements violate unitarity. Sum of squares is {u_sq_sum:.4f} (must be <= 1.0).")
    params['m_Zd_eV'] = float(config['dark_portal']['m_Zd_eV'])
    params['g_D'] = float(config['dark_portal']['g_D'])
    params['epsilon_prime'] = float(config['dark_portal']['epsilon_prime'])
    # Physical positivity bounds
    if params['m_Zd_eV'] <= 0:
        raise ValueError("Dark mediator mass m_Zd_eV must be strictly positive.")
    if params['g_D'] <= 0:
        raise ValueError("Dark mediator gauge coupling g_D must be strictly positive.")
    if params['g_D'] >= np.sqrt(4 * np.pi):
        raise ValueError(f"Dark gauge coupling g_D ({params['g_D']:.2f}) exceeds the perturbativity bound (sqrt(4*pi)).")
    if params['epsilon_prime'] <= 0:
        raise ValueError("Kinetic mixing epsilon_prime must be strictly positive.")
    params['use_debye_screening'] = config['dark_portal']['use_debye_screening']
    
    params['p_index'] = float(config['source']['p_index'])
    if config['source']['p_index'] <= 0:
        raise ValueError("p_index should be positive. The solver assumes a spectrum proportional to E^(-p_index).")
    params['Phi_0'] = float(config['source']['Phi_0'])
    if config['source']['Phi_0'] <= 0:
        raise ValueError("Flux normalization Phi_0 should be strictly positive.")
    params['z_max'] = float(config['source']['z_max'])
    if params['z_max'] <= 0.0:
        raise ValueError("Maximum redshift z_max must be strictly positive.")
    params['rho_evol'] = float(config['source']['rho_evol'])
    params['f_e_S'] = float(config['source']['f_e_S'])
    params['f_mu_S'] = float(config['source']['f_mu_S'])
    params['f_tau_S'] = float(config['source']['f_tau_S'])
    flavor_sum = params['f_e_S'] + params['f_mu_S'] + params['f_tau_S']
    if not np.isclose(flavor_sum, 1.0, atol=1e-5):
        raise ValueError(f"Source flavor ratios must sum to 1.0. Current sum: {flavor_sum}")
    if params['f_e_S'] < 0 or params['f_mu_S'] < 0 or params['f_tau_S'] < 0:
        raise ValueError("Source flavor fractions must be strictly >= 0. Negative flux injection is unphysical.")
    params['is_antineutrino'] = config['source']['is_antineutrino']
    params['nubar_to_nu_ratio'] = config['source']['nubar_to_nu_ratio']
    
    params['n_z_bins'] = int(config['grid']['n_z_bins'])
    params['n_mw_paths'] = int(config['grid']['n_mw_paths'])
    # Defaults set by the production convergence study (tools/convergence_study.py,
    # report in output/convergence/): n_earth_chords=3 (the old default) carried
    # 1.6-4.9e-2 absolute flavor-ratio error against a 100-chord reference at every
    # tested dark-portal point -- by far the largest numerical systematic of any default
    # -- while 12 chords bring it to ~1e-2 (percent level). N_steps_MW=25 already agrees
    # with a 400-step reference to ~1e-10 everywhere (the Milky Way transit is deeply
    # adiabatic), so the old default of 100 was pure surplus.
    params['n_earth_chords'] = int(config['grid']['n_earth_chords'])
    params['N_steps_MW'] = int(config['grid']['N_steps_MW'])
    # Coarse z-sampling of the thermal-rate grid (0 disables interpolation -- exact grid).
    params['n_z_thermal'] = int(config['grid']['n_z_thermal'])
    params['N_steps_earth'] = int(config['grid']['N_steps_earth'])
    params['r_start_kpc'] = float(config['grid']['r_start_kpc'])
    paths_mw = config['grid']['n_mw_paths']
    chords_earth = config['grid']['n_earth_chords']
    if not isinstance(paths_mw, int) or paths_mw < 1:
        raise ValueError("n_mw_paths must be an integer >= 1.")
    if not isinstance(chords_earth, int) or chords_earth < 1:
        raise ValueError("n_earth_chords must be an integer >= 1.")
    
    if config['grid']['n_E_bins'] <= 1 or config['grid']['n_z_bins'] <= 1:
        raise ValueError("Grid resolution (n_E_bins, n_z_bins) must be greater than 1 to define integration steps.")

    # Energy grid sanity check
    if config['grid']['E_min_eV'] >= config['grid']['E_max_eV']:
        raise ValueError("E_max_eV must be strictly greater than E_min_eV for the cascading integral to function properly.")
    E_bins = np.logspace(np.log10(config['grid']['E_min_eV']), 
                         np.log10(config['grid']['E_max_eV']), 
                         int(config['grid']['n_E_bins']))
    
    # A single call to run_solver always evaluates exactly one physics configuration, so
    # there is nothing to parallelize at the config level (unlike dark_portal_scanner.py,
    # which parallelizes across many configs). Instead, n_cores drives genuine intra-run
    # parallelism across the independent Milky Way paths / Earth chords in the spatial
    # handover (see propagate_earth_spatial), which is where a single run actually has
    # independent work to distribute.
    n_cores = max(1, int(config['execution']['n_cores']))
    params['n_cores'] = n_cores
    # Optional CUDA spatial handover (requires cupy + a CUDA device; silently falls back
    # to the CPU path with a warning when unavailable -- see gpu_spatial.py).
    params['use_gpu'] = bool(config['execution']['use_gpu'])
    params['show_progress'] = True
    params['verbose'] = int(config['execution']['verbose'])

    safe_print(f"Initializing solver across {len(E_bins)} coupled energy bins...")
    if n_cores > 1:
        safe_print(f"Using up to {n_cores} core(s) to parallelize the Milky Way / Earth spatial handover.")
    else:
        safe_print("Deploying single-core worker.")

    E_out, Flux_out = propagation_worker((E_bins, params))

    return E_out, Flux_out

def plot_flavor_fluxes(E_array: np.ndarray, Flux_array: np.ndarray, p_index: float, is_antineutrino_flag: str, output_filename: str = "../output/plots/flux_plot.png"):
    """Plots and maps the total astrophysical neutrino spectrum ratios at Earth layers."""
    E_GeV = E_array / 1e9
    scaling_factor = E_GeV**2.0
    
    flux_e = Flux_array[:, 0]
    flux_mu = Flux_array[:, 1]
    flux_tau = Flux_array[:, 2]
    flux_total = np.sum(Flux_array, axis=1)
    
    s_flux_e = flux_e * scaling_factor
    s_flux_mu = flux_mu * scaling_factor
    s_flux_tau = flux_tau * scaling_factor
    s_flux_total = flux_total * scaling_factor
    
    safe_total = np.maximum(flux_total, 1e-100)
    ratio_e = flux_e / safe_total
    ratio_mu = flux_mu / safe_total
    ratio_tau = flux_tau / safe_total
    
    # Evaluate flag for dynamic LaTeX labels
    flag_val = str(is_antineutrino_flag).strip().lower()
    if flag_val in ['sum', 'both']:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\nu + \bar{\nu}$)', r'$\nu_e + \bar{\nu}_e$', r'$\nu_\mu + \bar{\nu}_\mu$', r'$\nu_\tau + \bar{\nu}_\tau$'
    elif flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\bar{\nu}$)', r'$\bar{\nu}_e$', r'$\bar{\nu}_\mu$', r'$\bar{\nu}_\tau$'
    else:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\nu$)', r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    ax1.plot(E_GeV, s_flux_total, color='black', lw=2.5, label=lbl_tot)
    ax1.plot(E_GeV, s_flux_e, color='crimson', linestyle='--', lw=2.0, label=lbl_e)
    ax1.plot(E_GeV, s_flux_mu, color='royalblue', linestyle='-.', lw=2.0, label=lbl_mu)
    ax1.plot(E_GeV, s_flux_tau, color='forestgreen', linestyle=':', lw=2.0, label=lbl_tau)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Scaled Flux $E_0^2 \Phi$ [Arb. Units]', fontsize=14)
    ax1.set_title('Astrophysical Neutrino Diffuse Flux at Earth', fontsize=15, pad=15)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(fontsize=12, loc='lower right')
    
    ax2.plot(E_GeV, ratio_e, color='crimson', linestyle='--', lw=2.0)
    ax2.plot(E_GeV, ratio_mu, color='royalblue', linestyle='-.', lw=2.0)
    ax2.plot(E_GeV, ratio_tau, color='forestgreen', linestyle=':', lw=2.0)
    
    ax2.set_ylabel('Flavor Ratio', fontsize=14)
    ax2.set_xlabel(r'Observed Energy $E_0$ [GeV]', fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    safe_print(f"\nPlot successfully saved to: {os.path.abspath(output_filename)}")
    plt.show()

# =============================================================================
# CANONICAL CONFIGURATION DEFAULTS
# =============================================================================
# Single source of truth for every configuration default. create_default_config, the
# CLI baseline in __main__, run_solver's normalization, and FrequentistAnalysis's
# physics_constants fallbacks all read from THIS dict -- defaults previously lived in
# four separately-maintained copies, and keeping them synchronized by hand produced
# exactly the class of silent drift this codebase's audits kept finding.
def _build_default_config() -> Dict:
    return {
        "cosmology": {
            "H0_km_s_Mpc": 67.4,
            "Omega_m": 0.315,
            "Omega_L": 0.685,
            "T_nu0_eV": 1.95e-4,
            "n_n0_cm3": 0.25e-6,
            "m_lightest_eV": 0.01,
            "mass_ordering": "normal",
            "eta_e": 1.0,
            "eta_mu": 1.0,
            "eta_tau": 1.0
        },
        "standard_mixing": {
            "dm21_sq": 7.41e-5,
            "dm31_sq": 2.511e-3,
            "th12_deg": 33.41,
            "th13_deg": 8.58,
            "th23_deg": 49.1,
            "d_cp_deg": 132.0,
            "use_SM_potential": False,
            "n_e_earth_cm3": 1.5e24,
            "detector_depth_km": 1.5,
            "V_SM_struct": None
        },
        "dark_portal": {
            "u_e": 0.5,
            "u_mu": 0.5,
            "u_tau": 0.0,
            "m_Zd_eV": 2.0e6,
            "g_D": 1.e-4,
            "epsilon_prime": 1e-1,
            "use_debye_screening": True,
            "U_LRI_struct": None
        },
        "source": {
            "p_index": 2.5,
            "Phi_0": 1.0,
            "z_max": 4.0,
            "rho_evol": 0.0,
            "f_e_S": 0.333333,
            "f_mu_S": 0.666667,
            "f_tau_S": 0.0,
            "is_antineutrino": "sum",
            "nubar_to_nu_ratio": 1.0
        },
        "grid": {
            "E_min_eV": 1e13,
            "E_max_eV": 3e16,
            "n_E_bins": 1000,
            "n_z_bins": 1000,
            "n_mw_paths": 6,
            "n_earth_chords": 12,
            "N_steps_MW": 25,
            "n_z_thermal": 0,
            "N_steps_earth": 500,
            "r_start_kpc": 100.0
        },
        "output": {
            "flux_filename": "../output/data/output_flux.txt",
            "plot_filename": "../output/plots/flux_plot.png"
        },
        "execution": {
            "n_cores": multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1,
            "verbose": 1,
            "use_gpu": False
        }
    }

DEFAULT_CONFIG = _build_default_config()

def merge_config_with_defaults(config: Dict) -> Dict:
    """Overlays a (possibly partial) user config onto the canonical defaults, REJECTING
    unknown sections and keys.

    The rejection matters as much as the merging: a typo like 'n_eart_chords' previously
    fell through every .get() fallback and silently did nothing -- the same
    parsed-but-dropped failure mode behind several bugs fixed in past audits
    (n_e_earth_cm3, use_SM_potential, execution.n_cores). Keys and sections whose names
    begin with '_comment' are ignored, matching the FrequentistAnalysis convention.
    """
    merged = copy.deepcopy(DEFAULT_CONFIG)
    for section, entries in config.items():
        if section.startswith("_comment"):
            continue
        if section not in merged:
            raise ValueError(f"Unknown config section '{section}'. Valid sections: {sorted(merged)}")
        if not isinstance(entries, dict):
            raise ValueError(f"Config section '{section}' must be a dict of settings.")
        for key, value in entries.items():
            if key.startswith("_comment"):
                continue
            if key not in merged[section]:
                raise ValueError(
                    f"Unknown config key '{section}.{key}'. Valid keys for [{section}]: "
                    f"{sorted(merged[section])}")
            merged[section][key] = value
    return merged

# =============================================================================
# ENTRY POINT & CONFIGURATION GENERATION
# =============================================================================

def create_default_config(filename: str = "../config/config_dark_portal.json") -> Dict:
    """Writes (and returns) the canonical default configuration -- see DEFAULT_CONFIG."""
    default_config = copy.deepcopy(DEFAULT_CONFIG)
    # The optional structure-matrix overrides default to None; omit them from the
    # generated JSON so the file shows only actively-used settings.
    default_config["standard_mixing"].pop("V_SM_struct", None)
    default_config["dark_portal"].pop("U_LRI_struct", None)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(default_config, f, indent=4)
    safe_print(f"Created default configuration file: {filename}")
    return default_config

if __name__ == "__main__":
    # Initialize argument flags for terminal overrides mapping to the JSON variables
    parser = argparse.ArgumentParser(description="Dark Portal Neutrino Propagation Solver command-line flag orchestrator.")
    parser.add_argument("--config", type=str, default="../config/config_dark_portal.json", help="Path to JSON parameter file.")
    
    # Cosmology Overrides
    parser.add_argument("--H0_km_s_Mpc", type=float, default=None)
    parser.add_argument("--Omega_m", type=float, default=None)
    parser.add_argument("--Omega_L", type=float, default=None)
    parser.add_argument("--T_nu0_eV", type=float, default=None)
    parser.add_argument("--n_n0_cm3", type=float, default=None)
    parser.add_argument("--m_lightest_eV", type=float, default=None)
    parser.add_argument("--mass_ordering", type=str, default=None)
    parser.add_argument("--eta_e", type=float, default=None)
    parser.add_argument("--eta_mu", type=float, default=None)
    parser.add_argument("--eta_tau", type=float, default=None)
    
    # Standard Oscillation Mixing Overrides
    parser.add_argument("--dm21_sq", type=float, default=None)
    parser.add_argument("--dm31_sq", type=float, default=None)
    parser.add_argument("--th12_deg", type=float, default=None)
    parser.add_argument("--th13_deg", type=float, default=None)
    parser.add_argument("--th23_deg", type=float, default=None)
    parser.add_argument("--d_cp_deg", type=float, default=None)
    parser.add_argument("--use_SM_potential", type=str, default=None, help="Set to 'true' or 'false'")
    parser.add_argument("--n_e_earth_cm3", type=float, default=None)
    parser.add_argument("--detector_depth_km", type=float, default=None)
    
    # Dark Portal Vector Interactions Overrides
    parser.add_argument("--u_e", type=float, default=None)
    parser.add_argument("--u_mu", type=float, default=None)
    parser.add_argument("--u_tau", type=float, default=None)
    parser.add_argument("--m_Zd_eV", type=float, default=None)
    parser.add_argument("--g_D", type=float, default=None)
    parser.add_argument("--epsilon_prime", type=float, default=None)
    parser.add_argument("--use_debye_screening", type=str, default=None, help="Set to 'true' or 'false'")
    
    # Primary Flux Injectors Overrides
    parser.add_argument("--p_index", type=float, default=None)
    parser.add_argument("--Phi_0", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--rho_evol", type=float, default=None)
    parser.add_argument("--f_e_S", type=float, default=None)
    parser.add_argument("--f_mu_S", type=float, default=None)
    parser.add_argument("--f_tau_S", type=float, default=None)
    parser.add_argument("--is_antineutrino", type=str, default=None)
    parser.add_argument("--nubar_to_nu_ratio", type=float, default=None)
    
    # Grid Configurations Overrides
    parser.add_argument("--E_min_eV", type=float, default=None)
    parser.add_argument("--E_max_eV", type=float, default=None)
    parser.add_argument("--n_E_bins", type=int, default=None)
    parser.add_argument("--n_z_bins", type=int, default=None)
    parser.add_argument("--n_z_thermal", type=int, default=None, help="Coarse z-samples for the thermal-rate grid (default 0 = exact, no interpolation; see build_thermal_grid before enabling)")
    parser.add_argument("--n_mw_paths", type=int, default=None, help="Number of Milky Way trajectories for solid-angle average")
    parser.add_argument("--n_earth_chords", type=int, default=None, help="Number of Earth chords for PREM evaluation")
    parser.add_argument("--N_steps_MW", type=int, default=None, help="Number of spatial integration steps in the Milky Way")
    parser.add_argument("--N_steps_earth", type=int, default=None, help="Number of spatial integration steps per Earth chord (default 500; only validated at the default -- see propagate_earth_spatial's docstring before lowering it for a resonance-scanning fit)")
    parser.add_argument("--r_start_kpc", type=float, default=None, help="Starting distance for Milky Way integration in kpc")
    
    # Export Slated Locations Overrides
    parser.add_argument("--flux_filename", type=str, default=None)
    parser.add_argument("--plot_filename", type=str, default=None)
    
    # Execution Architecture Overrides
    parser.add_argument("--n_cores", type=int, default=None)
    parser.add_argument("--use_gpu", type=str, default=None, help="Set to 'true' to run the spatial handover on a CUDA GPU via cupy (falls back to CPU if unavailable)")
    parser.add_argument("--verbose", type=int, default=None)
    
    args = parser.parse_args()

    # Baseline defaults come from the single canonical DEFAULT_CONFIG (no more
    # separately-maintained copy here).
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Precedence Step 2: Overlay values from JSON file if one exists
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            for section, entries in file_config.items():
                if section in config and isinstance(entries, dict):
                    config[section].update(entries)
                else:
                    # Unknown section (or malformed entry): pass through untouched so
                    # run_solver's merge_config_with_defaults rejects it with a clear
                    # error, rather than silently dropping it here.
                    config[section] = entries
    else:
        print(f"Config file {args.config} not found. Running under strict system defaults.")

    def str_to_bool(val):
        if val is None: return None
        return str(val).lower().strip() in ['true', '1', 'yes', 'y']

    # Precedence Step 3: Overwrite entries dynamically via shell CLI arguments
    if args.H0_km_s_Mpc is not None: config["cosmology"]["H0_km_s_Mpc"] = args.H0_km_s_Mpc
    if args.Omega_m is not None: config["cosmology"]["Omega_m"] = args.Omega_m
    if args.Omega_L is not None: config["cosmology"]["Omega_L"] = args.Omega_L
    if args.T_nu0_eV is not None: config["cosmology"]["T_nu0_eV"] = args.T_nu0_eV
    if args.n_n0_cm3 is not None: config["cosmology"]["n_n0_cm3"] = args.n_n0_cm3
    if args.m_lightest_eV is not None: config["cosmology"]["m_lightest_eV"] = args.m_lightest_eV
    if args.mass_ordering is not None: config["cosmology"]["mass_ordering"] = args.mass_ordering
    if args.eta_e is not None: config["cosmology"]["eta_e"] = args.eta_e
    if args.eta_mu is not None: config["cosmology"]["eta_mu"] = args.eta_mu
    if args.eta_tau is not None: config["cosmology"]["eta_tau"] = args.eta_tau

    if args.dm21_sq is not None: config["standard_mixing"]["dm21_sq"] = args.dm21_sq
    if args.dm31_sq is not None: config["standard_mixing"]["dm31_sq"] = args.dm31_sq
    if args.th12_deg is not None: config["standard_mixing"]["th12_deg"] = args.th12_deg
    if args.th13_deg is not None: config["standard_mixing"]["th13_deg"] = args.th13_deg
    if args.th23_deg is not None: config["standard_mixing"]["th23_deg"] = args.th23_deg
    if args.d_cp_deg is not None: config["standard_mixing"]["d_cp_deg"] = args.d_cp_deg
    if args.use_SM_potential is not None: config["standard_mixing"]["use_SM_potential"] = str_to_bool(args.use_SM_potential)
    if args.n_e_earth_cm3 is not None: config["standard_mixing"]["n_e_earth_cm3"] = args.n_e_earth_cm3
    if args.detector_depth_km is not None: config["standard_mixing"]["detector_depth_km"] = args.detector_depth_km

    if args.u_e is not None: config["dark_portal"]["u_e"] = args.u_e
    if args.u_mu is not None: config["dark_portal"]["u_mu"] = args.u_mu
    if args.u_tau is not None: config["dark_portal"]["u_tau"] = args.u_tau
    if args.m_Zd_eV is not None: config["dark_portal"]["m_Zd_eV"] = args.m_Zd_eV
    if args.g_D is not None: config["dark_portal"]["g_D"] = args.g_D
    if args.epsilon_prime is not None: config["dark_portal"]["epsilon_prime"] = args.epsilon_prime
    if args.use_debye_screening is not None: config["dark_portal"]["use_debye_screening"] = str_to_bool(args.use_debye_screening)

    if args.p_index is not None: config["source"]["p_index"] = args.p_index
    if args.Phi_0 is not None: config["source"]["Phi_0"] = args.Phi_0
    if args.z_max is not None: config["source"]["z_max"] = args.z_max
    if args.rho_evol is not None: config["source"]["rho_evol"] = args.rho_evol
    if args.f_e_S is not None: config["source"]["f_e_S"] = args.f_e_S
    if args.f_mu_S is not None: config["source"]["f_mu_S"] = args.f_mu_S
    if args.f_tau_S is not None: config["source"]["f_tau_S"] = args.f_tau_S
    if args.is_antineutrino is not None: config["source"]["is_antineutrino"] = args.is_antineutrino
    if args.nubar_to_nu_ratio is not None: config["source"]["nubar_to_nu_ratio"] = args.nubar_to_nu_ratio

    if args.E_min_eV is not None: config["grid"]["E_min_eV"] = args.E_min_eV
    if args.E_max_eV is not None: config["grid"]["E_max_eV"] = args.E_max_eV
    if args.n_E_bins is not None: config["grid"]["n_E_bins"] = args.n_E_bins
    if args.n_z_bins is not None: config["grid"]["n_z_bins"] = args.n_z_bins
    if args.n_z_thermal is not None: config["grid"]["n_z_thermal"] = args.n_z_thermal
    if args.n_mw_paths is not None: config["grid"]["n_mw_paths"] = args.n_mw_paths
    if args.n_earth_chords is not None: config["grid"]["n_earth_chords"] = args.n_earth_chords
    if args.N_steps_MW is not None: config["grid"]["N_steps_MW"] = args.N_steps_MW
    if args.N_steps_earth is not None: config["grid"]["N_steps_earth"] = args.N_steps_earth
    if args.r_start_kpc is not None: config["grid"]["r_start_kpc"] = args.r_start_kpc

    if args.flux_filename is not None: config["output"]["flux_filename"] = args.flux_filename
    if args.plot_filename is not None: config["output"]["plot_filename"] = args.plot_filename

    if args.n_cores is not None: config["execution"]["n_cores"] = args.n_cores
    if args.use_gpu is not None: config["execution"]["use_gpu"] = str_to_bool(args.use_gpu)
    if args.verbose is not None: config["execution"]["verbose"] = args.verbose
            
    import time as _time
    _t0 = _time.perf_counter()
    E_array, Flux_array = run_solver(config)
    _runtime = _time.perf_counter() - _t0

    output_cfg = config.get("output", {})
    flux_filename = output_cfg.get("flux_filename", "../output/data/output_flux.txt")
    plot_filename = output_cfg.get("plot_filename", "../output/plots/flux_plot.png")

    os.makedirs(os.path.dirname(os.path.abspath(flux_filename)), exist_ok=True)
    save_flux_to_file(E_array, Flux_array, filename=flux_filename)
    write_run_manifest(flux_filename + ".manifest.json", config, runtime_s=_runtime)
    
    Flux_array = np.clip(Flux_array, 1e-50, None) 
    Total_Flux = np.sum(Flux_array, axis=1)
    f_e, f_mu, f_tau = Flux_array[:, 0] / Total_Flux, Flux_array[:, 1] / Total_Flux, Flux_array[:, 2] / Total_Flux
    
    safe_print("\nIntegration Complete. Sample Output (Mid-grid Energy):")
    mid_idx = len(E_array) // 2
    safe_print(f"Energy: {E_array[mid_idx] / 1e9:.2e} GeV")
    safe_print(f"Flavor Ratios at Earth [e, mu, tau]: [{f_e[mid_idx]:.3f}, {f_mu[mid_idx]:.3f}, {f_tau[mid_idx]:.3f}]")

    if HAS_MATPLOTLIB:
        os.makedirs(os.path.dirname(os.path.abspath(plot_filename)), exist_ok=True)
        plot_flavor_fluxes(E_array, Flux_array, config['source']['p_index'], config['source'].get('is_antineutrino', 'sum'), output_filename=plot_filename)