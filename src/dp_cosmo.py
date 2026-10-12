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

@njit(fastmath=True, cache=True)
def _absorption_factors(G: float, dt: float):
    """For dF/dt = -G F + S over a step dt with constant G and S: returns
    (e, a1, g) = (exp(-G dt), (1 - exp(-G dt))/G, dt - a1), so that
        F(dt) = F(0) e + S a1   and   (number absorbed) = F(0)(1 - e) + S g.
    Series below G dt = 1e-3, where the closed forms cancel; exact at G = 0 (e = 1,
    a1 = dt, g = 0)."""
    x = G * dt
    if x < 1e-3:
        e = np.exp(-x)
        a1 = dt * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0))))
        g = dt * x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x * (1.0 / 120.0 - x / 720.0))))
        return e, a1, g
    e = np.exp(-x)
    a1 = -np.expm1(-x) / G
    return e, a1, dt - a1


@njit(fastmath=True, cache=True, parallel=True)
def _cosmo_projections(E_bins: np.ndarray, P_table: np.ndarray, log_s0: float, inv_dlogs: float,
                       c_z: float, P_all: np.ndarray):
    """|U|^2 projections at every energy of one z-step from the sigma-table (see
    _build_sigma_projection_table): they depend on E only through sigma = E * c_z,
    interpolated linearly in log|sigma|."""
    N_E = len(E_bins)
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


@njit(fastmath=True, cache=True)
def step_cosmological_propagation(z: float, dt: float, R_z: float, E_bins: np.ndarray,
                                  F_mass: np.ndarray, rho_src: np.ndarray,
                                  P_table: np.ndarray, log_s0: float, inv_dlogs: float,
                                  c_z: float, p_index: float, Phi_0: float,
                                  BR: np.ndarray, W_kernel: np.ndarray, pref_kernel: np.ndarray,
                                  self_frac: np.ndarray, P_all: np.ndarray,
                                  Gamma_flavor: np.ndarray, R_out: np.ndarray):
    """One cosmological z-step: absorption on the relic neutrinos, regeneration of the
    absorbed flux at lower energies (the cascade), and injection, per mass state.

    Arguments: z, dt -- redshift and time step (eV^-1); R_z -- source evolution;
    E_bins -- observed energies (eV); F_mass -- (N_E, 3) mass-state fluxes, updated in
    place; rho_src -- source flavor composition; P_table, log_s0, inv_dlogs, c_z -- the
    sigma-table and this step's coordinate; p_index, Phi_0 -- source spectrum;
    BR -- flavor branching of the regenerated neutrinos (sum 1); W_kernel, pref_kernel,
    self_frac -- the discrete cascade kernel (cascade_kernel_weights); P_all -- (N_E, 3, 3)
    projection buffer; Gamma_flavor -- (N_E, 3) flavor absorption
    rates at this z (eV); R_out -- (N_E,) receives each bin's absorbed number per unit
    energy per unit time, averaged over the step (diagnostics and tests).

    The cascade sends each absorbed neutrino of energy E_j to energy E_i < E_j with
    spectrum 3 E_i^2 / E_j^3 (one neutrino per absorption, flavor split by BR). The
    absorption per step is often enormous -- G dt up to 1e10 near resonance -- so the
    regenerated source must be what each bin ACTUALLY absorbs during the step, not its
    initial rate G F held over the whole step: until 2026-10-12 it was the latter, which
    re-emitted up to G dt times more neutrinos than were absorbed, step after step, and
    inflated the fluxes by up to 1e70. Bins are therefore swept from high to low energy:
    each solves its linear equation over the step exactly (constant rates and sources),
    and passes down the number it absorbed, averaged over the step. A bin's share of its
    own regeneration (the i = j term of the kernel) makes its absorbed number appear on
    both sides; that linear relation is solved directly. With the kernel's exact bin
    integrals (cascade_kernel_weights) the step conserves number exactly for any G dt,
    the only loss being secondaries below the grid."""
    N_E = len(E_bins)
    _cosmo_projections(E_bins, P_table, log_s0, inv_dlogs, c_z, P_all)
    above = 0.0          # sum over j > i of R_j W_j (R: absorbed number per energy per time)
    for i in range(N_E - 1, -1, -1):
        E_z = E_bins[i] * (1.0 + z)
        inj_spectrum = Phi_0 * (E_z**(-p_index))
        pref = pref_kernel[i]
        num = 0.0
        qg = 0.0
        e_a = np.empty(3)
        a1_a = np.empty(3)
        s0_a = np.empty(3)
        q_a = np.empty(3)
        for a in range(3):
            G = (P_all[i, 0, a] * Gamma_flavor[i, 0]
                 + P_all[i, 1, a] * Gamma_flavor[i, 1]
                 + P_all[i, 2, a] * Gamma_flavor[i, 2])
            brp = (P_all[i, 0, a] * BR[0] + P_all[i, 1, a] * BR[1] + P_all[i, 2, a] * BR[2])
            inj = (P_all[i, 0, a] * rho_src[0]
                   + P_all[i, 1, a] * rho_src[1]
                   + P_all[i, 2, a] * rho_src[2]) * inj_spectrum * R_z
            e, a1, g = _absorption_factors(G, dt)
            s0 = inj + brp * pref * above
            q = brp * self_frac[i]
            num += F_mass[i, a] * (1.0 - e) + s0 * g
            qg += q * g
            e_a[a] = e
            a1_a[a] = a1
            s0_a[a] = s0
            q_a[a] = q
        R = num / (dt - qg)        # qg <= 3 dE/E * dt: the denominator stays near dt
        R_out[i] = R
        for a in range(3):
            F_mass[i, a] = F_mass[i, a] * e_a[a] + (s0_a[a] + q_a[a] * R) * a1_a[a]
        above += R * W_kernel[i]


def cascade_kernel_weights(E_bins: np.ndarray):
    """The cascade kernel 3 E^2 / E_j^3 (secondaries of a neutrino absorbed at E_j, per unit
    energy, for E <= E_j) integrated exactly over the geometric bins of E_bins.

    Returns (W_kernel, pref_kernel, self_frac): a parent in bin j sends to a lower bin i the
    fraction pref_kernel[i] dE_i W_kernel[j], with pref_kernel[i] = (E_hi,i^3 - E_lo,i^3)/dE_i
    and W_kernel[j] = dE_j / E_j^3, and keeps in its own bin the fraction
    self_frac[j] = 1 - (E_lo,j / E_j)^3 -- secondaries above E_j are kinematically forbidden.
    The fractions add up to 1 - (E_lo,0 / E_j)^3 exactly. The midpoint weights used until
    2026-10-12 (3 E_i^2 for every bin, the parent's own bin included in full) kept about
    1 + 1.5 dlnE of a neutrino per absorption: ~1.2% created per generation of rescattering
    on the 1000-bin grid."""
    from dp_core import compute_dE_geometric
    N = len(E_bins)
    dE = compute_dE_geometric(E_bins)
    if N > 1:
        h = np.sqrt(E_bins[1] / E_bins[0])
        E_lo, E_hi = E_bins / h, E_bins * h
    else:
        E_lo, E_hi = E_bins.copy(), E_bins.copy()
    W_kernel = dE / E_bins**3
    with np.errstate(divide="ignore", invalid="ignore"):
        pref = np.where(dE > 0, (E_hi**3 - E_lo**3) / dE, 3.0 * E_bins**2)
    self_frac = 1.0 - (E_lo / E_bins)**3
    return (np.ascontiguousarray(W_kernel), np.ascontiguousarray(pref),
            np.ascontiguousarray(self_frac))
