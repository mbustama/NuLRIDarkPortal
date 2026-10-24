"""
dp_cosmo.py

Cosmological-phase kernels: Hubble rate, source evolution, the cosmological LRI
potential and adiabatic mixing, the sigma projection table, and the per-z-step
propagation kernel (cascade suffix sum). Split verbatim out of dark_portal_solver.py.
"""
import numpy as np

from dp_core import njit, prange, GeV, G_F, cm_inv3_to_eV3, calc_debye_mass_sq
from dp_propagator import eigenvectors_3x3
from dp_transit import frame_at

@njit(fastmath=True, error_model="numpy", cache=True)
def calc_H_Hubble(z: float, H0: float, Omega_m: float, Omega_L: float) -> float:
    """Calculates the Hubble expansion rate H(z) at redshift z in a flat LCDM universe."""
    return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)

@njit(fastmath=True, error_model="numpy", cache=True)
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
def yukawa_horizon_inv_m2(m_sq: float, R_H: float) -> float:
    """The Yukawa potential of a uniform medium per unit density, cut at the Hubble radius:
        int_{r < R_H} exp(-m r) / (4 pi r) d^3r = (1 - (1 + x) e^{-x}) / m^2,   x = m R_H,
    i.e. 1/m^2 for ranges well inside the horizon and R_H^2 / 2 beyond it (decided 2026-10-13).
    Until then the cosmological potential used 1 / max(m^2, 1e-25 eV^2), which capped it for
    every mediator lighter than 3e-13 eV (1e35 times too weak at 1e-30 eV). With R_H infinite
    this is 1/m^2. Arguments: m_sq = m_Zd^2 + m_D^2 (eV^2); R_H (eV^-1)."""
    if not np.isfinite(R_H):
        return 1.0 / m_sq
    if m_sq <= 0.0:
        return 0.5 * R_H * R_H
    x = np.sqrt(m_sq) * R_H
    if x < 1e-3:
        return R_H * R_H * (0.5 - x * (1.0 / 3.0 - x * (0.125 - x / 30.0)))
    return (-np.expm1(-x) - x * np.exp(-x)) / m_sq


@njit(cache=True)
def calc_V_LRI_cosmo(z: float, u_arr: np.ndarray, U_LRI_struct: np.ndarray,
                     m_Zd: float, g_D: float, eps_prime: float,
                     n_n0: float, T_nu0: float, use_debye_screening: bool, eta_bg: np.ndarray,
                     R_H: float = np.inf) -> np.ndarray:
    """Computes the macroscopic effective potential V_{LRI} sourced ONLY by cosmological neutrons.
    R_H: the Hubble radius c/H(z) (eV^-1) at which the uniform medium's Yukawa integral is cut
    (yukawa_horizon_inv_m2); infinite by default.

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

    A_LRI = (g_D * eps_prime) * yukawa_horizon_inv_m2(m_Zd**2 + m_D_sq, R_H) \
        * (4.0 * c_W * m_Z**2 / g_weak) * (np.sqrt(2)/2.0 * G_F * n_n_z)

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


@njit(cache=True, parallel=True)
def _build_sigma_projection_table(Q: np.ndarray, Mp: np.ndarray, s: np.ndarray, floor: float,
                                  sigma_nodes: np.ndarray) -> np.ndarray:
    """|U|^2 projection tables for the one-parameter family H(sigma) = M_rot + sigma * S,
    given as the texture basis (Q, Mp, s, floor) of dp_transit.texture_scales.

    The cosmological mixing matrix at any (E, z) is the eigenvector matrix of
    M_rot / (2 E (1+z)) + A_LRI(z) * S, and eigenvectors are scale-invariant, so it
    depends ONLY on sigma = 2 E (1+z) A_LRI(z) -- verified to 8e-15 numerically. The
    ~1e6 per-run eigenvector computations collapse onto this table; level-crossing
    structure in log sigma is mixing-angle-broad (no sub-grid features, unlike the
    thermal resonance), so a few thousand nodes with linear interpolation suffice.
    The eigenvectors come from dp_transit.frame_at, accurate at any sigma: until 2026-10-13
    the closed form (eigenvectors_3x3) lost the vacuum splitting of a rank-one texture's two
    degenerate eigenvalues once sigma exceeded M ~1e13 times, returning rounding noise.
    """
    n = len(sigma_nodes)
    table = np.zeros((n, 3, 3), dtype=np.float64)
    for k in prange(n):
        vals = np.empty(3)
        U = np.empty((3, 3), dtype=np.complex128)
        frame_at(Q, Mp, s, floor, sigma_nodes[k], vals, U)
        for a in range(3):
            for b in range(3):
                table[k, a, b] = U[a, b].real**2 + U[a, b].imag**2
    return table

@njit(cache=True, parallel=True)
def _build_sigma_coupling_tables(Q: np.ndarray, Mp: np.ndarray, s: np.ndarray, floor: float,
                                 sigma_nodes: np.ndarray, A: np.ndarray, B: np.ndarray):
    """The tables of the coherent absorption (dp_absorption; docs/dev/PHYSICS_AUDIT.md, 1a-1c) on the sigma nodes:
    with W the eigenvectors of M_rot + sigma S (flavor basis, ascending eigenvalues; dp_transit.frame_at),
      P[k, f, a]    = |W_fa|^2                       the flavor content of eigenstate a (injection),
      Qabs[k, r, a] = |(A W)_ra|^2 / 2,  A = U^dag G  the weight of eigenstate a on relic mass state r,
      Qreg[k, a]    = Re (W^dag B W)_aa / 2, B = G G^dag  the weight of the decays' neutrinos in eigenstate a,
    in the absorption kernel's units (Gamma_in = w M / 12 pi). Like P, they depend on (E, z) only through sigma."""
    n = len(sigma_nodes)
    P = np.zeros((n, 3, 3), dtype=np.float64)
    Qabs = np.zeros((n, 3, 3), dtype=np.float64)
    Qreg = np.zeros((n, 3), dtype=np.float64)
    for k in prange(n):
        vals = np.empty(3)
        W = np.empty((3, 3), dtype=np.complex128)
        frame_at(Q, Mp, s, floor, sigma_nodes[k], vals, W)
        for f in range(3):
            for a in range(3):
                P[k, f, a] = W[f, a].real**2 + W[f, a].imag**2
        for r in range(3):
            for a in range(3):
                acc = 0.0 + 0.0j
                for f in range(3):
                    acc += A[r, f] * W[f, a]
                Qabs[k, r, a] = 0.5 * (acc.real**2 + acc.imag**2)
        for a in range(3):
            acc = 0.0 + 0.0j
            for f in range(3):
                for g in range(3):
                    acc += np.conj(W[f, a]) * B[f, g] * W[g, a]
            Qreg[k, a] = 0.5 * acc.real
    return P, Qabs, Qreg


@njit(cache=True, parallel=True)
def _build_sigma_jump_table(Q: np.ndarray, Mp: np.ndarray, s: np.ndarray, floor: float,
                            sigma_nodes: np.ndarray, G: np.ndarray):
    """The soft relic scatterings' tables on the sigma nodes (2026-10-22; notes, 'Soft scatterings on the relics'):
    with W the eigenvectors of M_rot + sigma S (flavor basis, ascending eigenvalues; dp_transit.frame_at) and G the
    flavor coupling (Hermitian),
      J[k, b, a]  = |(W^dag G W)_ba|^2        the weight of a scattering that takes eigenstate a to b,
      Gd[k, a]    = Re (W^dag G W)_aa,  GG[k, a] = (W^dag G^dag G W)_aa   (the damping of the coherences),
      lam[k, a]   = the eigenvalues (M units, eV^2; their differences over 2E are the oscillation frequencies)."""
    n = len(sigma_nodes)
    J = np.zeros((n, 3, 3), dtype=np.float64)
    Gd = np.zeros((n, 3), dtype=np.float64)
    GG = np.zeros((n, 3), dtype=np.float64)
    lam = np.zeros((n, 3), dtype=np.float64)
    for k in prange(n):
        vals = np.empty(3)
        W = np.empty((3, 3), dtype=np.complex128)
        frame_at(Q, Mp, s, floor, sigma_nodes[k], vals, W)
        GW = np.zeros((3, 3), dtype=np.complex128)
        for f in range(3):
            for a in range(3):
                acc = 0.0 + 0.0j
                for g in range(3):
                    acc += G[f, g] * W[g, a]
                GW[f, a] = acc
        for a in range(3):
            lam[k, a] = vals[a]
            nrm = 0.0
            for f in range(3):
                nrm += GW[f, a].real ** 2 + GW[f, a].imag ** 2
            GG[k, a] = nrm
            for b in range(3):
                acc = 0.0 + 0.0j
                for f in range(3):
                    acc += np.conj(W[f, b]) * GW[f, a]
                J[k, b, a] = acc.real ** 2 + acc.imag ** 2
                if a == b:
                    Gd[k, a] = acc.real
    return J, Gd, GG, lam


@njit(cache=True, parallel=True)
def _build_sigma_mass_table(Q: np.ndarray, Mp: np.ndarray, s: np.ndarray, floor: float,
                            sigma_nodes: np.ndarray, U: np.ndarray):
    """The eigenstates' mass content on the sigma nodes (2026-10-24; dp_decay): T[k, i, a] = |(U^dag W)_ia|^2, W the
    eigenvectors of M_rot + sigma S (flavor basis, ascending eigenvalues; dp_transit.frame_at), U the mixing matrix of
    the kind (conj U for antineutrinos). Interpolated per step like the projections (_cosmo_projections)."""
    n = len(sigma_nodes)
    T = np.zeros((n, 3, 3), dtype=np.float64)
    for k in prange(n):
        vals = np.empty(3)
        W = np.empty((3, 3), dtype=np.complex128)
        frame_at(Q, Mp, s, floor, sigma_nodes[k], vals, W)
        for i in range(3):
            for a in range(3):
                acc = 0.0 + 0.0j
                for f in range(3):
                    acc += np.conj(U[f, i]) * W[f, a]
                T[k, i, a] = acc.real ** 2 + acc.imag ** 2
    return T


@njit(cache=True, parallel=True)
def _soft_jumps(E_bins: np.ndarray, zp1: float, J_table: np.ndarray, Gd_table: np.ndarray, GG_table: np.ndarray,
                lam_table: np.ndarray, log_s0: float, inv_dlogs: float, c_z: float, Gamma_unit: float, dt: float,
                F: np.ndarray, Wd: np.ndarray, use_wd: bool):
    """One z-step of the soft scatterings on the relics, applied to the eigenstate populations F (N_E, 3) in place.
    Wd (N_E, 3), with use_wd: the eigenstates' decay widths into the mediator (lab frame, eV; dp_decay, 2026-10-24),
    which damp the coherences too: D_ab += (Wd_a + Wd_b) / 2 (a state that decays fast is Zeno-protected from the
    conversions into it).
    Per energy (sigma = E c_z, the tables interpolated linearly in log sigma as _cosmo_projections), the populations
    relax with the rates a -> b
        R_ba = Gamma_unit J_ba w^2 / (w^2 + D^2),   w = (lam_a - lam_b) / 2E(1+z),
        D = Gamma_unit [ (GG_a + GG_b) / 2 - Gd_a Gd_b ]   (the damping of the coherence between a and b),
    the Lindblad dynamics d rho/dt = Gamma_unit (G rho G - {G^2, rho}/2) with the coherences eliminated: the secular
    rates Gamma_unit J_ba where the coherences oscillate faster than they are damped (D << w), the Zeno-suppressed
    ones (w^2 / D) where not. R is symmetric (G Hermitian), so the step is exact through its eigenbasis: the
    populations' sum is kept, and rates far above 1/dt drive them to equal shares of the states they connect."""
    N_E = len(E_bins)
    N_sig = J_table.shape[0]
    for i in prange(N_E):
        tot = F[i, 0] + F[i, 1] + F[i, 2]
        if tot <= 0.0:
            continue
        sig = np.abs(E_bins[i] * c_z)
        f = 0.0 if sig <= 0.0 else (np.log(sig) - log_s0) * inv_dlogs
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
        two_E = 2.0 * E_bins[i] * zp1
        A = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(a + 1, 3):
                Jab = 0.5 * ((1.0 - w) * (J_table[k0, b, a] + J_table[k0, a, b])
                             + w * (J_table[k0 + 1, b, a] + J_table[k0 + 1, a, b]))
                om = ((1.0 - w) * (lam_table[k0, a] - lam_table[k0, b])
                      + w * (lam_table[k0 + 1, a] - lam_table[k0 + 1, b])) / two_E
                ga = (1.0 - w) * Gd_table[k0, a] + w * Gd_table[k0 + 1, a]
                gb = (1.0 - w) * Gd_table[k0, b] + w * Gd_table[k0 + 1, b]
                gga = (1.0 - w) * GG_table[k0, a] + w * GG_table[k0 + 1, a]
                ggb = (1.0 - w) * GG_table[k0, b] + w * GG_table[k0 + 1, b]
                D = Gamma_unit * max(0.5 * (gga + ggb) - ga * gb, 0.0)
                if use_wd:
                    D += 0.5 * (Wd[i, a] + Wd[i, b])
                r = Gamma_unit * Jab
                if D > 0.0:
                    r *= om * om / (om * om + D * D)
                A[a, b] = r
                A[b, a] = r
        rmax = 0.0
        for a in range(3):
            A[a, a] = -(A[0, a] + A[1, a] + A[2, a] - A[a, a])
            rmax = max(rmax, -A[a, a])
        if rmax * dt <= 0.0:
            continue
        # as a correction, F + Q W (e^(-mu dt) - 1) W^T Q^T F, in the complement Q of the uniform mode (exact, 2026-10-24:
        # through eigh the uniform mode's zero eigenvalue came out as ~1e-16 of the largest rate, which at rates x dt
        # beyond ~1e16 (the Zeno-transition band, g ~ m^1/2) removed up to half of the total). On Q, the Laplacian of
        # the rates is B = [[(4 r01 + r02 + r12)/2, sqrt(3)(r02 - r12)/2], [., 3 (r02 + r12)/2]]; its eigenvalues from
        # tr B and det B = 3 (r01 r02 + r01 r12 + r02 r12) (no cancellation), the small one det/large. A slow step
        # moves the populations by its own small amount only (the analysis interpolates fluxes in their logarithm)
        r01 = A[0, 1]
        r02 = A[0, 2]
        r12 = A[1, 2]
        b11 = 0.5 * (4.0 * r01 + r02 + r12)
        b22 = 1.5 * (r02 + r12)
        b12 = 0.8660254037844386 * (r02 - r12)
        det = 3.0 * (r01 * r02 + r01 * r12 + r02 * r12)
        mu1 = 0.5 * ((b11 + b22) + np.sqrt((b11 - b22) ** 2 + 4.0 * b12 * b12))
        mu2 = det / mu1 if mu1 > 0.0 else 0.0
        th = 0.5 * np.arctan2(2.0 * b12, b11 - b22)
        c = np.cos(th)
        sn = np.sin(th)
        y1 = (F[i, 0] - F[i, 1]) * 0.7071067811865476
        y2 = (F[i, 0] + F[i, 1] - 2.0 * F[i, 2]) * 0.4082482904638631
        p1 = (c * y1 + sn * y2) * np.expm1(-mu1 * dt)
        p2 = (-sn * y1 + c * y2) * np.expm1(-mu2 * dt)
        d1 = c * p1 - sn * p2
        d2 = sn * p1 + c * p2
        x = np.empty(3)
        x[0] = 0.7071067811865476 * d1 + 0.4082482904638631 * d2
        x[1] = -0.7071067811865476 * d1 + 0.4082482904638631 * d2
        x[2] = -0.8164965809277261 * d2
        clipped = False
        for a in range(3):
            F[i, a] += x[a]
            if F[i, a] < 0.0:
                F[i, a] = 0.0
                clipped = True
        if clipped:
            s_new = F[i, 0] + F[i, 1] + F[i, 2]
            if s_new > 0.0:
                for a in range(3):
                    F[i, a] *= tot / s_new


@njit(fastmath=True, error_model="numpy", cache=True, parallel=True)
def _cosmo_coherent_rates(E_bins: np.ndarray, P_table: np.ndarray, Qabs_table: np.ndarray,
                          Qreg_table: np.ndarray, log_s0: float, inv_dlogs: float, c_z: float,
                          Gamma_state: np.ndarray, inv_sum_G: float, P_out: np.ndarray, G_out: np.ndarray,
                          brp_out: np.ndarray):
    """One z-step's rates per eigenstate from the coupling tables (interpolated linearly in log|sigma|, as
    _cosmo_projections): G_out[i, a] = sum_r Gamma_state[i, r] Qabs[r, a] (Gamma_state: each relic state's rate per
    unit weight, dp_thermal.relic_state_grid_cached), brp_out[i, a] = Qreg[a] / sum_G (the fraction of the decays
    giving a neutrino in eigenstate a; their sum is Gamma_nu / Gamma), P_out the flavor contents."""
    N_E = len(E_bins)
    N_sig = P_table.shape[0]
    for i in prange(N_E):
        sig = np.abs(E_bins[i] * c_z)
        if sig <= 0.0:
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
            for b in range(3):
                P_out[i, a, b] = (1.0 - w) * P_table[k0, a, b] + w * P_table[k0 + 1, a, b]
            g = 0.0
            for r in range(3):
                g += Gamma_state[i, r] * ((1.0 - w) * Qabs_table[k0, r, a] + w * Qabs_table[k0 + 1, r, a])
            G_out[i, a] = g
            brp_out[i, a] = ((1.0 - w) * Qreg_table[k0, a] + w * Qreg_table[k0 + 1, a]) * inv_sum_G


@njit(fastmath=True, error_model="numpy", cache=True)
def _cosmo_flavor_rates(P: np.ndarray, Gamma_flavor: np.ndarray, BR: np.ndarray, G_out: np.ndarray,
                        brp_out: np.ndarray):
    """The theory notes' form (absorption_model 'theory_notes'): flavor rates and branching projected on the
    eigenstates without interference, G[i, a] = sum_f P[i, f, a] Gamma_f, brp[i, a] = sum_f P[i, f, a] BR_f."""
    for i in range(P.shape[0]):
        for a in range(3):
            G_out[i, a] = (P[i, 0, a] * Gamma_flavor[i, 0] + P[i, 1, a] * Gamma_flavor[i, 1]
                           + P[i, 2, a] * Gamma_flavor[i, 2])
            brp_out[i, a] = P[i, 0, a] * BR[0] + P[i, 1, a] * BR[1] + P[i, 2, a] * BR[2]


@njit(fastmath=True, error_model="numpy", cache=True)
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


@njit(fastmath=True, error_model="numpy", cache=True, parallel=True)
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


@njit(fastmath=True, error_model="numpy", cache=True)
def step_cosmological_rates(z: float, dt: float, R_z: float, E_bins: np.ndarray,
                            F_nu: np.ndarray, F_nb: np.ndarray, w_nu: float, w_nb: float,
                            rho_src: np.ndarray, Pi_nu: np.ndarray, Pi_nb: np.ndarray,
                            G_nu: np.ndarray, G_nb: np.ndarray, brp_nu: np.ndarray, brp_nb: np.ndarray,
                            p_index: float, Phi_0: float,
                            W3: np.ndarray, pref3: np.ndarray, self_same: np.ndarray,
                            W1: np.ndarray, W2: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                            self_cross: np.ndarray, R_nu: np.ndarray, R_nb: np.ndarray, inj_w: np.ndarray):
    """step_cosmological_pair with this step's rates given per eigenstate: Pi_* (N_E, 3, 3) the flavor contents
    of the eigenstates at the sources (injection), G_* (N_E, 3) the absorption rates (eV), brp_* (N_E, 3) the
    fraction of each absorption's decays that gives a neutrino of that kind in each eigenstate (_cosmo_coherent_rates
    or _cosmo_flavor_rates). The rest as below.

    One cosmological z-step for neutrinos and antineutrinos together: absorption on the
    relic neutrinos, regeneration of the absorbed flux at lower energies (the cascade), and
    injection, per mass state.

    c_inj_nu, c_inj_nb -- the sigma-table coordinate where the neutrinos are emitted, and
    P_inj_nu, P_inj_nb -- buffers for its projections: the injected flavors split over the
    eigenstates of the potential at the sources, which carry their populations adiabatically
    (sorted) into the intergalactic eigenstates. Equal to c_z_* unless the sources sit in a
    host galaxy (source.host_galaxy_scale > 0, 2026-10-13); then the buffers are filled.

    Arguments: z, dt -- redshift and time step (eV^-1); R_z -- source evolution (sources per comoving
    volume, relative to z = 0: the injection per unit observed energy carries dE_z/dE = 1+z); E_bins --
    observed energies (eV); F_nu, F_nb -- (N_E, 3) mass-state fluxes of neutrinos and
    antineutrinos, updated in place; w_nu, w_nb -- their injection weights; rho_src -- source
    flavor composition; P_table_*, log_s0_*, inv_dlogs_*, c_z_* -- each species' sigma-table
    and this step's coordinate; p_index, Phi_0 -- source spectrum; BR -- flavor branching of
    the regenerated neutrinos (sum 1); W3, pref3, self_same -- the same-species kernel
    (cascade_kernel_weights); W1, W2, c1, c2, self_cross -- the cross-species kernel
    (cross_kernel_weights); P_nu, P_nb -- (N_E, 3, 3) projection buffers; Gamma_flavor --
    (N_E, 3) flavor absorption rates at this z (eV), the same for both species; R_nu, R_nb --
    (N_E,) receive each bin's absorbed number per unit energy per unit time, averaged over
    the step (diagnostics and tests); inj_w -- (N_E,) the sources' emission window per bin (1 inside, 0
    outside; source.injection_E_min/max_GeV, 2026-10-15).

    An absorption nu(E_j) + relic nubar -> Z_D -> nu + nubar emits TWO secondaries. The
    Z_D is polarized along the incoming neutrino, and the outgoing particle of the same kind
    carries spectrum 3 E^2 / E_j^3 (forward, 3/4 of E_j on average), the other kind
    3 (E_j - E)^2 / E_j^3 (backward, the remaining 1/4); likewise for an absorbed
    antineutrino. Until 2026-10-13 only the same-kind secondary was kept and the two kinds
    were propagated independently (the theory notes' Eq. 6 regenerates rho only from rho),
    which dropped half of the regenerated particles and a quarter of the absorbed energy.

    The absorption per step is often enormous (G dt up to 1e10 near resonance), so the
    regenerated sources are what each bin ACTUALLY absorbs during the step: bins are swept
    from high to low energy, each solving its linear equations over the step exactly
    (constant rates and sources) and passing down the numbers it absorbed. A bin's own
    regeneration (the i = j terms of both kernels) puts both species' absorbed numbers on
    both sides: a 2 x 2 linear system per bin, solved directly. With the kernels' exact bin
    integrals the step balances number exactly for any G dt: every absorption adds one
    particle, less the secondaries that fall below the grid."""
    N_E = len(E_bins)
    # running sums over parents j > i of R_j W_j for each species (W3 doubles as the
    # same-species kernel and the cross kernel's third term)
    s3_nu = 0.0
    s3_nb = 0.0
    s1_nu = 0.0
    s1_nb = 0.0
    s2_nu = 0.0
    s2_nb = 0.0
    e_n = np.empty(3)
    a1_n = np.empty(3)
    s0_n = np.empty(3)
    q_n = np.empty(3)
    c_n = np.empty(3)
    e_b = np.empty(3)
    a1_b = np.empty(3)
    s0_b = np.empty(3)
    q_b = np.empty(3)
    c_b = np.empty(3)
    for i in range(N_E - 1, -1, -1):
        E_z = E_bins[i] * (1.0 + z)
        # sources emit Phi_0 E_z^-p per unit source energy E_z = E (1+z); per unit observed energy E
        # that is (1+z) times more (dE_z/dE), so the flux today is int dz R(z) Phi_0 (E (1+z))^-p / H(z),
        # the standard result (number of emitted neutrinos conserved). Without the (1+z), until
        # 2026-10-13, a source at z was weighted by R(z)/(1+z): 45% of the emitted neutrinos lost
        # (SFR to z = 4) and the absorption/potential effects weighted toward low z.
        # inj_w: the sources' emission window per bin of today's energy (source.injection_E_min/max_GeV; 1 inside, 0
        # outside; all ones for the whole power law)
        inj_spectrum = Phi_0 * (E_z**(-p_index)) * R_z * (1.0 + z) * inj_w[i]
        # regeneration densities into bin i from absorptions above it
        from_nu_same = pref3[i] * s3_nu
        from_nb_same = pref3[i] * s3_nb
        from_nu_cross = c1[i] * s1_nu - c2[i] * s2_nu + pref3[i] * s3_nu
        from_nb_cross = c1[i] * s1_nb - c2[i] * s2_nb + pref3[i] * s3_nb
        reg_nu = from_nu_same + from_nb_cross
        reg_nb = from_nb_same + from_nu_cross
        num_n = 0.0
        num_b = 0.0
        A_n = dt
        A_b = dt
        C_n = 0.0
        C_b = 0.0
        for a in range(3):
            # neutrinos
            G = G_nu[i, a]
            brp = brp_nu[i, a]
            inj = (Pi_nu[i, 0, a] * rho_src[0] + Pi_nu[i, 1, a] * rho_src[1]
                   + Pi_nu[i, 2, a] * rho_src[2]) * inj_spectrum * w_nu
            e, a1, g = _absorption_factors(G, dt)
            s0 = inj + brp * reg_nu
            q = brp * self_same[i]               # per unit R_nu (own bin, same kind)
            c = brp * self_cross[i]              # per unit R_nb (own bin, other kind)
            num_n += F_nu[i, a] * (1.0 - e) + s0 * g
            A_n -= q * g
            C_n += c * g
            e_n[a] = e
            a1_n[a] = a1
            s0_n[a] = s0
            q_n[a] = q
            c_n[a] = c
            # antineutrinos
            G = G_nb[i, a]
            brp = brp_nb[i, a]
            inj = (Pi_nb[i, 0, a] * rho_src[0] + Pi_nb[i, 1, a] * rho_src[1]
                   + Pi_nb[i, 2, a] * rho_src[2]) * inj_spectrum * w_nb
            e, a1, g = _absorption_factors(G, dt)
            s0 = inj + brp * reg_nb
            q = brp * self_same[i]
            c = brp * self_cross[i]
            num_b += F_nb[i, a] * (1.0 - e) + s0 * g
            A_b -= q * g
            C_b += c * g
            e_b[a] = e
            a1_b[a] = a1
            s0_b[a] = s0
            q_b[a] = q
            c_b[a] = c
        # R_nu dt = num_n + (sum q g) R_nu + (sum c g) R_nb, and the same for antineutrinos:
        # [A_n, -C_n; -C_b, A_b] (R_nu, R_nb) = (num_n, num_b); the self fractions are
        # O(dlnE), so A ~ dt and the determinant stays near dt^2
        det = A_n * A_b - C_n * C_b
        Rn = (num_n * A_b + C_n * num_b) / det
        Rb = (num_b * A_n + C_b * num_n) / det
        R_nu[i] = Rn
        R_nb[i] = Rb
        for a in range(3):
            F_nu[i, a] = F_nu[i, a] * e_n[a] + (s0_n[a] + q_n[a] * Rn + c_n[a] * Rb) * a1_n[a]
            F_nb[i, a] = F_nb[i, a] * e_b[a] + (s0_b[a] + q_b[a] * Rb + c_b[a] * Rn) * a1_b[a]
        s3_nu += Rn * W3[i]
        s3_nb += Rb * W3[i]
        s1_nu += Rn * W1[i]
        s1_nb += Rb * W1[i]
        s2_nu += Rn * W2[i]
        s2_nb += Rb * W2[i]


@njit(fastmath=True, error_model="numpy", cache=True)
def step_cosmological_pair(z: float, dt: float, R_z: float, E_bins: np.ndarray,
                           F_nu: np.ndarray, F_nb: np.ndarray, w_nu: float, w_nb: float,
                           rho_src: np.ndarray,
                           P_table_nu: np.ndarray, log_s0_nu: float, inv_dlogs_nu: float, c_z_nu: float,
                           P_table_nb: np.ndarray, log_s0_nb: float, inv_dlogs_nb: float, c_z_nb: float,
                           p_index: float, Phi_0: float, BR: np.ndarray,
                           W3: np.ndarray, pref3: np.ndarray, self_same: np.ndarray,
                           W1: np.ndarray, W2: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                           self_cross: np.ndarray, P_nu: np.ndarray, P_nb: np.ndarray,
                           Gamma_flavor: np.ndarray, R_nu: np.ndarray, R_nb: np.ndarray,
                           c_inj_nu: float, c_inj_nb: float, P_inj_nu: np.ndarray,
                           P_inj_nb: np.ndarray, inj_w: np.ndarray):
    """One cosmological z-step in the theory notes' absorption form (flavor rates Gamma_flavor (N_E, 3) and flavor
    branching BR projected on the eigenstates without interference; absorption_model 'theory_notes'): the
    projections, then step_cosmological_rates."""
    N_E = len(E_bins)
    _cosmo_projections(E_bins, P_table_nu, log_s0_nu, inv_dlogs_nu, c_z_nu, P_nu)
    _cosmo_projections(E_bins, P_table_nb, log_s0_nb, inv_dlogs_nb, c_z_nb, P_nb)
    Pi_nu = P_nu
    Pi_nb = P_nb
    if c_inj_nu != c_z_nu:
        _cosmo_projections(E_bins, P_table_nu, log_s0_nu, inv_dlogs_nu, c_inj_nu, P_inj_nu)
        Pi_nu = P_inj_nu
    if c_inj_nb != c_z_nb:
        _cosmo_projections(E_bins, P_table_nb, log_s0_nb, inv_dlogs_nb, c_inj_nb, P_inj_nb)
        Pi_nb = P_inj_nb
    G_nu = np.empty((N_E, 3))
    G_nb = np.empty((N_E, 3))
    b_nu = np.empty((N_E, 3))
    b_nb = np.empty((N_E, 3))
    _cosmo_flavor_rates(P_nu, Gamma_flavor, BR, G_nu, b_nu)
    _cosmo_flavor_rates(P_nb, Gamma_flavor, BR, G_nb, b_nb)
    step_cosmological_rates(z, dt, R_z, E_bins, F_nu, F_nb, w_nu, w_nb, rho_src, Pi_nu, Pi_nb, G_nu, G_nb,
                            b_nu, b_nb, p_index, Phi_0, W3, pref3, self_same, W1, W2, c1, c2, self_cross,
                            R_nu, R_nb, inj_w)


def cascade_kernel_weights(E_bins: np.ndarray):
    """The same-kind cascade kernel 3 E^2 / E_j^3 (the secondary of the absorbed particle's
    kind, per unit energy, for E <= E_j) integrated exactly over the geometric bins of E_bins.

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


def cross_kernel_weights(E_bins: np.ndarray):
    """The kernel of the secondary of the OTHER kind, 3 (E_j - E)^2 / E_j^3 for E <= E_j,
    integrated exactly over the geometric bins of E_bins. Its integral over a lower bin
    [E_lo, E_hi] of the parent's spectrum is ((E_j - E_lo)^3 - (E_j - E_hi)^3) / E_j^3
        = 3 dE / E_j - 3 (E_hi^2 - E_lo^2) / E_j^2 + (E_hi^3 - E_lo^3) / E_j^3,
    separable in the parent, so the cascade accumulates three running sums over parents.

    Returns (W1, W2, c1, c2, self_cross): a parent in bin j sends to a lower bin i the
    density c1[i] R_j W1[j] - c2[i] R_j W2[j] + pref3[i] R_j W3[j] (per unit energy, with
    pref3, W3 from cascade_kernel_weights), with W1 = dE / E, W2 = dE / E^2, c1 = 3 dE / dE,
    c2 = 3 (E_hi^2 - E_lo^2) / dE; and keeps in its own bin the fraction
    self_cross[j] = (1 - E_lo,j / E_j)^3. The fractions add up to (1 - E_lo,0 / E_j)^3:
    everything but what falls below the grid."""
    from dp_core import compute_dE_geometric
    N = len(E_bins)
    dE = compute_dE_geometric(E_bins)
    if N > 1:
        h = np.sqrt(E_bins[1] / E_bins[0])
        E_lo, E_hi = E_bins / h, E_bins * h
    else:
        E_lo, E_hi = E_bins.copy(), E_bins.copy()
    W1 = dE / E_bins
    W2 = dE / E_bins**2
    with np.errstate(divide="ignore", invalid="ignore"):
        c1 = np.where(dE > 0, 3.0 * (E_hi - E_lo) / dE, 3.0)
        c2 = np.where(dE > 0, 3.0 * (E_hi**2 - E_lo**2) / dE, 6.0 * E_bins)
    self_cross = (1.0 - E_lo / E_bins)**3
    return (np.ascontiguousarray(W1), np.ascontiguousarray(W2), np.ascontiguousarray(c1),
            np.ascontiguousarray(c2), np.ascontiguousarray(self_cross))
