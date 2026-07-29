"""
dp_cosmo.py

Cosmological-phase kernels: Hubble rate, source evolution, the cosmological LRI
potential and adiabatic mixing, the sigma projection table, and the per-z-step
propagation kernel (cascade suffix sum). Split verbatim out of dark_portal_solver.py.
"""
import numpy as np

from dp_core import njit, prange, GeV, G_F, cm_inv3_to_eV3, calc_debye_mass_sq
from dp_propagator import eigenvectors_3x3

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


@njit(fastmath=True, cache=True, parallel=True)
def _build_sigma_projection_table(M_rot: np.ndarray, S_eff: np.ndarray,
                                  sigma_nodes: np.ndarray) -> np.ndarray:
    """|U|^2 projection tables for the one-parameter family H(sigma) = M_rot + sigma * S.

    The cosmological mixing matrix at any (E, z) is the eigenvector matrix of
    M_rot / (2 E (1+z)) + A_LRI(z) * S, and eigenvectors are scale-invariant, so it
    depends ONLY on sigma = 2 E (1+z) A_LRI(z) -- verified to 8e-15 numerically. The
    ~1e6 per-run eigenvector computations collapse onto this table; level-crossing
    structure in log sigma is mixing-angle-broad (no sub-grid features, unlike the
    thermal resonance), so a few thousand nodes with linear interpolation suffice.
    """
    n = len(sigma_nodes)
    table = np.zeros((n, 3, 3), dtype=np.float64)
    for k in prange(n):
        H = M_rot + sigma_nodes[k] * S_eff
        U = eigenvectors_3x3(np.ascontiguousarray(H))
        for a in range(3):
            for b in range(3):
                table[k, a, b] = np.abs(U[a, b])**2
    return table

@njit(fastmath=True, cache=True, parallel=True)
def step_cosmological_propagation(z: float, dt: float, R_z: float, E_bins: np.ndarray,
                                  F_mass: np.ndarray, rho_src: np.ndarray,
                                  P_table: np.ndarray, log_s0: float, inv_dlogs: float,
                                  c_z: float,
                                  p_index: float, Phi_0: float,
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

    # First loop: flavor composition via the sigma-table (see
    # _build_sigma_projection_table): the projection probabilities |U|^2 at this z-step
    # depend on E only through sigma = E * c_z, interpolated linearly in log|sigma|.
    N_sig = P_table.shape[0]
    for i in prange(N_E):
        sig = np.abs(E_bins[i] * c_z)
        if sig <= 0.0:
            # Vanishing potential (e.g. n_n0 = 0): vacuum-limit plateau = lowest node
            f = 0.0
        else:
            f = (np.log(sig) - log_s0) * inv_dlogs
        k0 = int(f)
        if k0 < 0:
            k0 = 0
        elif k0 > N_sig - 2:
            k0 = N_sig - 2
        w = f - k0
        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0
        for a in range(3):
            for k in range(3):
                P_all[i, a, k] = (1.0 - w) * P_table[k0, a, k] + w * P_table[k0 + 1, a, k]
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

