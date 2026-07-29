"""
dp_thermal.py

Thermal (CnuB) attenuation rates: the exact closed-form angular integral, the
semi-analytic momentum kernel (default), the midpoint reference kernel, grid
builders with optional coarse-z interpolation, and the two-component coupling
cache. Split verbatim out of dark_portal_solver.py.
"""
import numpy as np

from dp_core import njit, prange, cm_inv3_to_eV3

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


# =============================================================================
# SEMI-ANALYTIC MOMENTUM INTEGRATION OF THE THERMAL RATES (default path)
# =============================================================================
# The p-integral of p^2 f_FD(p/T) * I_ang(p) has exactly one sharp feature: the
# Breit-Wigner resonance crosses the angular range where u_-(p) or u_+(p) crosses
# zero (u_pm = A - M2 -+ B). Both u's are strictly monotone in p, so there is at most
# ONE crossing, and its location has a CLOSED FORM: with delta = (M2 - m^2)/(2x),
# u_-(p)=0  <=>  E_bg - p = delta  <=>  p_edge = (m^2 - delta^2)/(2 delta)   (delta < m)
# u_+(p)=0  <=>  E_bg + p = delta  <=>  p_edge = (delta^2 - m^2)/(2 delta)   (delta > m)
# and |u'(p_edge)| = 2 x delta / E_bg(p_edge) for either branch. The scheme:
#   * a narrow ZONE p_edge +- 50*MG/|u'| around the crossing is integrated with exact
#     polynomial x atan / polynomial x log antiderivatives under a secant-linear u(p)
#     (node pinned AT p_edge with v = 0 exactly, so the resonance step position is
#     exact regardless of how MG compares to fp noise on u) -- valid at ANY width;
#   * the structure-free remainder intervals use 16-point Gauss-Legendre on the
#     exact pointwise closed-form I_ang (smooth there by construction);
#   * wide widths (zone would cover >= 1/4 of the range) or no crossing: single
#     GL-16 over the whole range -- the integrand is then broad/smooth.
# The integrand splits with division-free prefactors (B = 2xp cancels):
#   p^2 f * 2/B          = p f / x
#   p^2 f * D/(2B)       = f (M2 - u_-) / (8 x^2)
#   p^2 f * F/(B MG)     = -f (M2 u_- + MG^2) / (4 x^2 MG)
# which are exact identities of the closed-form I_ang used everywhere else.

@njit(cache=True)
def _seg_moment_atan(h, Qa, Qb, va, vb, at_a, at_b, lg_a, lg_b, MG):
    """Integral over one zone segment of Q(p) * atan(v(p)/MG) dp with Q linear in p and
    v(p) linear (secant). at_/lg_ are the cached atan(v/MG), ln(v^2+MG^2) at the two
    nodes. Exact for the linearized inputs at any width MG."""
    if h <= 0.0:
        return 0.0
    if abs(at_b - at_a) < 1e-3:
        # atan essentially constant across the segment (and monotone in between):
        # trapezoid of the product is accurate to ~|d_at|^2 and avoids the cancellation
        # of the closed form at tiny relative v-spans.
        return 0.5 * (Qa * at_a + Qb * at_b) * h
    dv = vb - va
    beta = dv / h
    A1 = (Qb - Qa) / dv
    A0 = Qa - A1 * va
    P0a = va * at_a - 0.5 * MG * lg_a
    P0b = vb * at_b - 0.5 * MG * lg_b
    P1a = 0.5 * (va * va + MG * MG) * at_a - 0.5 * MG * va
    P1b = 0.5 * (vb * vb + MG * MG) * at_b - 0.5 * MG * vb
    return (A0 * (P0b - P0a) + A1 * (P1b - P1a)) / beta

@njit(cache=True)
def _seg_moment_log(h, Qa, Qb, va, vb, at_a, at_b, lg_a, lg_b, MG):
    """Same as _seg_moment_atan for Q(p) * ln(v(p)^2 + MG^2)."""
    if h <= 0.0:
        return 0.0
    if abs(lg_b - lg_a) < 1e-3 and va * vb > 0.0:
        # log flat AND no interior dip (the dip requires a sign change of v)
        return 0.5 * (Qa * lg_a + Qb * lg_b) * h
    dv = vb - va
    if dv == 0.0:
        return 0.5 * (Qa * lg_a + Qb * lg_b) * h
    beta = dv / h
    A1 = (Qb - Qa) / dv
    A0 = Qa - A1 * va
    L0a = va * lg_a - 2.0 * va + 2.0 * MG * at_a
    L0b = vb * lg_b - 2.0 * vb + 2.0 * MG * at_b
    L1a = 0.5 * (va * va + MG * MG) * (lg_a - 1.0)
    L1b = 0.5 * (vb * vb + MG * MG) * (lg_b - 1.0)
    return (A0 * (L0b - L0a) + A1 * (L1b - L1a)) / beta

@njit(cache=True)
def thermal_averaged_rates_flavor_analytic(E0: float, z: float, sum_G: float,
                                           sum_G_alpha: np.ndarray, m_Zd: float,
                                           m_nu_bg_arr: np.ndarray, T_nu0: float,
                                           eta_mass: np.ndarray, glx: np.ndarray,
                                           glw: np.ndarray, glx10: np.ndarray,
                                           glw10: np.ndarray, fd16: np.ndarray) -> np.ndarray:
    """Semi-analytic replacement for thermal_averaged_rates_flavor: same physics,
    same closed-form I_ang, but the p-integral is Gauss-Legendre on the smooth
    intervals plus exact atan/log moments across the (closed-form) resonance
    crossing -- see the scheme NOTE above. Validated against dense brute-force
    quadrature across a g_D ladder 1e-8..1 (see the regression tests).

    Node-count tuning (referee-validated like everything else here): the bulk
    structure-free cells integrate a single [0, p_max] panel whose GL nodes sit at
    FIXED scaled momenta q_k = 10 (1 + x_k) regardless of (E, z, state) -- so their
    Fermi-Dirac factors are the precomputed constants fd16 (saves the exp, the most
    expensive transcendental of that loop after atan/log); and panels shorter than
    p_max / 5 (the geometric resonance shells) use the 10-point rule glx10/glw10 --
    their integrand varies over at most ~a decade of the transition, far inside
    GL-10's accuracy for sub-p_max/5 spans."""
    T_nu = T_nu0 * (1.0 + z)

    Gamma_width = (sum_G * m_Zd) / (12.0 * np.pi)
    MG = m_Zd * Gamma_width
    MG2 = MG * MG
    M2 = m_Zd * m_Zd

    rates = np.zeros(3, dtype=np.float64)
    if MG == 0.0 or MG2 == 0.0:
        # MG2 == 0.0 also guards subnormal-underflow widths (m_Zd < ~1e-35), where
        # ln(v^2 + MG^2) at the resonance node would be ln(0) = -inf.
        return rates

    prefactor = 1.0 / (2.0 * np.pi)
    x = E0 * (1.0 + z)
    n_gl = len(glx)

    for i in range(3):
        if eta_mass[i] <= 0:
            continue
        m = m_nu_bg_arr[i]
        m2 = m * m
        T_i = T_nu * (eta_mass[i] ** (1.0 / 3.0))
        p_max = 20.0 * T_i

        # --- resonance-crossing geometry (closed form) ---
        delta = (M2 - m2) / (2.0 * x)
        p_edge = -1.0
        s_sharp = 0.0
        pe_inside = False
        use_moments = False
        if delta > 0.0:
            if delta < m:
                p_edge = (m2 - delta * delta) / (2.0 * delta)
            else:
                p_edge = (delta * delta - m2) / (2.0 * delta)
            E_bg_e = np.sqrt(m2 + p_edge * p_edge)
            uprime = 2.0 * x * delta / E_bg_e
            s_sharp = MG / uprime
            pe_inside = 0.0 < p_edge < p_max
            # exact moments only for a NARROW transition core (secant-u and linear-Q
            # both excellent there); a wide core means the atan/log vary on scales the
            # per-panel GL rule resolves pointwise anyway.
            use_moments = pe_inside and (3.0 * s_sharp < T_i / 30.0)

        # --- panel breakpoints: geometric shells pe +- K * (MG/|u'|) around the
        # crossing capture each decade of the atan/log transition; the innermost
        # +-3-width core gets the exact moments when narrow. Cells with no structure
        # in range keep the single [0, p_max] panel. ---
        brk = np.empty(14, dtype=np.float64)
        nb = 0
        brk[nb] = 0.0
        nb += 1
        brk[nb] = p_max
        nb += 1
        if delta > 0.0:
            for ik in range(5):
                K = 3.0 * 10.0 ** ik
                c = p_edge - K * s_sharp
                if 0.0 < c < p_max:
                    brk[nb] = c
                    nb += 1
                c = p_edge + K * s_sharp
                if 0.0 < c < p_max:
                    brk[nb] = c
                    nb += 1
            if pe_inside:
                brk[nb] = p_edge
                nb += 1
        # insertion sort (nb <= 13)
        for ii in range(1, nb):
            key = brk[ii]
            jj = ii - 1
            while jj >= 0 and brk[jj] > key:
                brk[jj + 1] = brk[jj]
                jj -= 1
            brk[jj + 1] = key

        core_lo = p_edge - 3.0 * s_sharp
        core_hi = p_edge + 3.0 * s_sharp
        crossing_minus = delta < m

        acc = 0.0
        for jp in range(nb - 1):
            a_ = brk[jp]
            b_ = brk[jp + 1]
            if b_ - a_ <= 1e-15 * p_max:
                continue

            if use_moments and (a_ == p_edge or b_ == p_edge) and a_ >= core_lo * 0.999999 and b_ <= core_hi * 1.000001:
                # --- transition-core panel: exact polynomial x atan / x log moments
                # under secant-linear u, with v = 0 EXACTLY at the p_edge endpoint (the
                # step position stays exact even when MG is below the fp noise of u).
                h = b_ - a_
                S0v = np.empty(2, dtype=np.float64)
                Qlv = np.empty(2, dtype=np.float64)
                Qav = np.empty(2, dtype=np.float64)
                umv = np.empty(2, dtype=np.float64)
                upv = np.empty(2, dtype=np.float64)
                atmv = np.empty(2, dtype=np.float64)
                atpv = np.empty(2, dtype=np.float64)
                lgmv = np.empty(2, dtype=np.float64)
                lgpv = np.empty(2, dtype=np.float64)
                for k in range(2):
                    p = a_ if k == 0 else b_
                    E_bg = np.sqrt(m2 + p * p)
                    f_FD = 1.0 / (np.exp(p / T_i) + 1.0)
                    A_ = m2 + 2.0 * x * E_bg
                    B_ = 2.0 * x * p
                    um = A_ - M2 - B_
                    up = A_ - M2 + B_
                    if p == p_edge:
                        if crossing_minus:
                            um = 0.0
                        else:
                            up = 0.0
                    S0v[k] = p * f_FD / x
                    Qlv[k] = f_FD * (M2 - um) / (8.0 * x * x)
                    Qav[k] = -f_FD * (M2 * um + MG2) / (4.0 * x * x * MG)
                    umv[k] = um
                    upv[k] = up
                    atmv[k] = np.arctan(um / MG)
                    atpv[k] = np.arctan(up / MG)
                    lgmv[k] = np.log(um * um + MG2)
                    lgpv[k] = np.log(up * up + MG2)
                acc += _seg_moment_atan(h, Qav[0], Qav[1], upv[0], upv[1],
                                        atpv[0], atpv[1], lgpv[0], lgpv[1], MG)
                acc -= _seg_moment_atan(h, Qav[0], Qav[1], umv[0], umv[1],
                                        atmv[0], atmv[1], lgmv[0], lgmv[1], MG)
                acc += _seg_moment_log(h, Qlv[0], Qlv[1], upv[0], upv[1],
                                       atpv[0], atpv[1], lgpv[0], lgpv[1], MG)
                acc -= _seg_moment_log(h, Qlv[0], Qlv[1], umv[0], umv[1],
                                       atmv[0], atmv[1], lgmv[0], lgmv[1], MG)
                acc += 0.5 * (S0v[0] + S0v[1]) * h
                continue

            # --- Gauss-Legendre panel on the exact pointwise closed-form I_ang ---
            mid = 0.5 * (a_ + b_)
            half = 0.5 * (b_ - a_)
            full_panel = a_ == 0.0 and b_ == p_max
            short_panel = (not full_panel) and (b_ - a_) < 0.2 * p_max
            n_panel = len(glx10) if short_panel else n_gl
            for k in range(n_panel):
                if short_panel:
                    p = mid + half * glx10[k]
                else:
                    p = mid + half * glx[k]
                E_bg = np.sqrt(m2 + p * p)
                if full_panel:
                    f_FD = fd16[k]
                else:
                    f_FD = 1.0 / (np.exp(p / T_i) + 1.0)
                A_ = m2 + 2.0 * x * E_bg
                B_ = 2.0 * x * p
                u_plus = A_ - M2 + B_
                u_minus = A_ - M2 - B_
                C = 1.0 - (A_ - M2) / B_
                D = 1.0 - (A_ - 2.0 * M2) / B_
                F = C * M2 - MG2 / B_
                I_ang = ((2.0 / B_)
                         + (D / (2.0 * B_)) * np.log((u_plus * u_plus + MG2) / (u_minus * u_minus + MG2))
                         + (F / (B_ * MG)) * (np.arctan(u_plus / MG) - np.arctan(u_minus / MG)))
                if I_ang < 0.0:
                    I_ang = 0.0
                w_k = glw10[k] if short_panel else glw[k]
                acc += w_k * half * p * p * f_FD * I_ang

        if acc < 0.0:
            acc = 0.0
        integral_sum = prefactor * acc
        for alpha in range(3):
            rates[alpha] += integral_sum * sum_G_alpha[alpha] * eta_mass[i]

    return rates

@njit(fastmath=True, cache=True, parallel=True)
def precompute_all_thermal_rates_analytic(E_bins: np.ndarray, z_bins: np.ndarray, sum_G: float,
                                          sum_G_alpha: np.ndarray, m_Zd: float, m_nu_bg_arr: np.ndarray,
                                          T_nu0: float, eta_mass: np.ndarray,
                                          glx: np.ndarray, glw: np.ndarray,
                                          glx10: np.ndarray, glw10: np.ndarray,
                                          fd16: np.ndarray) -> np.ndarray:
    """Grid builder for the semi-analytic thermal kernel (same layout/parallelism as
    precompute_all_thermal_rates)."""
    N_E = len(E_bins)
    N_z = len(z_bins)
    Gamma_grid = np.zeros((N_z, N_E, 3), dtype=np.float64)
    for k in prange(N_z):
        for i in range(N_E):
            Gamma_grid[k, i, :] = thermal_averaged_rates_flavor_analytic(
                E_bins[i], z_bins[k], sum_G, sum_G_alpha, m_Zd, m_nu_bg_arr, T_nu0,
                eta_mass, glx, glw, glx10, glw10, fd16)
    return Gamma_grid

# Gauss-Legendre nodes/weights on [-1, 1] for the semi-analytic thermal kernel
# (16-point for long panels, 10-point for the short resonance-shell panels), plus
# the Fermi-Dirac factors at the FIXED scaled nodes q_k = 10 (1 + x16_k) of the
# full-range [0, 20 T] panel -- identical for every (E, z, state).
_GL16_X, _GL16_W = np.polynomial.legendre.leggauss(16)
_GL16_X = np.ascontiguousarray(_GL16_X)
_GL16_W = np.ascontiguousarray(_GL16_W)
_GL10_X, _GL10_W = np.polynomial.legendre.leggauss(10)
_GL10_X = np.ascontiguousarray(_GL10_X)
_GL10_W = np.ascontiguousarray(_GL10_W)
_FD16 = np.ascontiguousarray(1.0 / (np.exp(10.0 * (1.0 + _GL16_X)) + 1.0))

# NOTE ON APPROXIMATING THIS GRID: two tabulation/interpolation schemes were built and
# validation-rejected. (1) Coarse-z interpolation at fixed E: the heavier CnuB states
# are non-relativistic, so the resonance sweeps z with relative width only ~p/m ~
# 0.4-2% -- a 201-node subgrid produced 3.5e-3 absolute flavor-ratio errors (kept as
# the opt-in n_z_thermal). (2) (x = E(1+z), tau) tabulation, where the resonance sits
# still in x: rejected outright, because the Breit-Wigner spike in x is NARROWER THAN
# ANY PRACTICAL GRID SPACING -- this "exact" grid itself merely point-samples an
# unresolved feature at its own E(1+z) positions, and interpolating between offset
# x-nodes produced factor-of-few rate errors at resonance-edge points (~20% even on
# energy-integrated rates). Any future fast scheme must treat the resonant spike
# analytically (e.g. narrow-width splitting of the angular integral), not by sampling.
# That analytic path now exists (thermal_averaged_rates_flavor_analytic above, the
# default); THIS midpoint kernel remains the reference implementation, selectable via
# grid.thermal_analytic = false.
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
                       T_nu0: float, eta_mass: np.ndarray, n_z_thermal: int = 0,
                       thermal_analytic: bool = True) -> np.ndarray:
    """Builds the (N_z, N_E, 3) thermal attenuation-rate grid, optionally via a coarse
    z-subgrid plus linear interpolation.

    thermal_analytic (config key grid.thermal_analytic, default True) selects the
    semi-analytic momentum integration (thermal_averaged_rates_flavor_analytic:
    Gauss-Legendre on the smooth intervals + exact closed-form moments across the
    resonance crossing) -- both faster and more accurate than the 100-point midpoint
    rule, whose ~p_max/100 sampling granularity cannot resolve resonance windows
    narrower than a step and carries edge errors up to the percent level at dip cells.
    Set it to False to fall back to the midpoint reference implementation
    (thermal_averaged_rates_flavor / precompute_all_thermal_rates).

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
    N_E = len(E_bins)

    def _rates_grid(zb):
        if thermal_analytic:
            return precompute_all_thermal_rates_analytic(E_bins, zb, sum_G, sum_G_alpha,
                                                         m_Zd, m_nu_bg_arr, T_nu0, eta_mass,
                                                         _GL16_X, _GL16_W, _GL10_X, _GL10_W,
                                                         _FD16)
        return precompute_all_thermal_rates(E_bins, zb, sum_G, sum_G_alpha, m_Zd,
                                            m_nu_bg_arr, T_nu0, eta_mass)

    n_sub = int(n_z_thermal)
    if n_sub < 2 or n_sub >= N_z:
        return _rates_grid(z_bins)
    z_sub = np.linspace(z_bins[0], z_bins[-1], n_sub)
    Gamma_sub = _rates_grid(z_sub)
    return _interp_thermal_grid_z(np.ascontiguousarray(Gamma_sub), N_z)

# =============================================================================
# TWO-COMPONENT THERMAL-GRID CACHE (coupling-sliced reuse)
# =============================================================================
# The momentum integral depends on the couplings ONLY through MG = m_Zd^2 sum_G/(12 pi),
# and its structure is two-component to high accuracy over moderate coupling ranges:
# off-resonance cells scale linearly with sum_G (whence Gamma/sum_G_alpha is constant),
# while resonance-window cells carry a narrow-width plateau ~ 1/MG (whence
# Gamma/sum_G_alpha ~ 1/sum_G). Writing U(s) = Gamma[k,i,:]/sum_G_alpha (the
# "unit-weight" grid, identical for all three flavors) the model
#     U(s) ~ I_A + I_B / s
# was MEASURED to hold to ~7e-8 over 2-decade brackets of s = sum_G at in-band
# mediator masses (degrading to ~1e-6..8e-4 over 6 decades, and failing over wider
# spans -- higher-order MG-dependence). The cache below exploits it LAZILY and with a
# built-in verification gate:
#   * every exact build stores its unit-weight grid keyed by everything else the grid
#     depends on (m_Zd, masses, T, eta_mass, grid geometry);
#   * once two stored points bracket a requested s with a 0.3-2.5 decade log-spread,
#     the two-point fit is verified against the EXACT grid just built at the interior
#     point; only if the significant-cell relative error is <= _TC_GATE does the
#     surrogate activate, valid strictly inside the verified bracket (no
#     extrapolation ever);
#   * out-of-bracket or unverified requests pay the normal exact build (which slides
#     the bracket for later requests); repeated gate failures disable the key.
# Coupling scans and profiled fits at fixed m_Zd hit the surrogate for all but their
# first ~3 evaluations; m_Zd scans never pay any overhead (one exact build per point,
# same as before, since each point is a fresh key). The reference paths
# (thermal_analytic=False, n_z_thermal >= 2) bypass the cache entirely.

_TC_MAX_KEYS = 4
_TC_MAX_POINTS = 6
_TC_GATE = 3e-7          # same accuracy class as the semi-analytic kernel itself
_TC_MIN_SPREAD_DEX = 0.3
_TC_MAX_SPREAD_DEX = 2.5
_TC_MAX_FAILURES = 3
_tc_cache = {}


def _tc_try_fit(state, s_new):
    """Attempts to activate the two-component surrogate using the freshly built exact
    point at s_new as the interior verification point of the closest stored bracket."""
    pts = state["points"]
    lower = [s for s, _ in pts if s < s_new]
    upper = [s for s, _ in pts if s > s_new]
    if not lower or not upper:
        return
    s1 = max(lower)
    s2 = min(upper)
    spread = np.log10(s2 / s1)
    if not (_TC_MIN_SPREAD_DEX <= spread <= _TC_MAX_SPREAD_DEX):
        return
    U1 = dict(pts)[s1]
    U2 = dict(pts)[s2]
    U_new = dict(pts)[s_new]
    I_B = (U1 - U2) / (1.0 / s1 - 1.0 / s2)
    I_A = U1 - I_B / s1
    sur = I_A + I_B / s_new
    u_max = U_new.max()
    if u_max <= 0.0:
        rel = 0.0
    else:
        sig = U_new > 1e-3 * u_max
        rel = float(np.max(np.abs(sur[sig] - U_new[sig]) / U_new[sig]))
    if rel <= _TC_GATE:
        state["fit"] = {"I_A": I_A, "I_B": I_B, "s_lo": s1, "s_hi": s2}
    else:
        state["failures"] += 1
        if state["failures"] >= _TC_MAX_FAILURES:
            state["disabled"] = True


def thermal_grid_cached(E_bins: np.ndarray, z_bins: np.ndarray, sum_G: float,
                        sum_G_alpha: np.ndarray, m_Zd: float, m_nu_bg_arr: np.ndarray,
                        T_nu0: float, eta_mass: np.ndarray, n_z_thermal: int = 0,
                        thermal_analytic: bool = True) -> np.ndarray:
    """build_thermal_grid with the two-component coupling cache in front (see the
    NOTE above). Exact-path results differ from build_thermal_grid only by the
    factoring of the sum_G_alpha weights out of the state sum (ulp level)."""
    if (not thermal_analytic) or int(n_z_thermal) >= 2 or sum_G <= 0.0:
        return build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, m_Zd,
                                  m_nu_bg_arr, T_nu0, eta_mass,
                                  n_z_thermal=n_z_thermal, thermal_analytic=thermal_analytic)

    weights = np.asarray(sum_G_alpha, dtype=np.float64)
    key = (float(m_Zd), tuple(float(v) for v in m_nu_bg_arr), float(T_nu0),
           tuple(float(v) for v in eta_mass), len(E_bins), float(E_bins[0]),
           float(E_bins[-1]), len(z_bins), float(z_bins[0]), float(z_bins[-1]))
    state = _tc_cache.get(key)
    if state is None:
        if len(_tc_cache) >= _TC_MAX_KEYS:
            _tc_cache.clear()
        state = {"points": [], "fit": None, "failures": 0, "disabled": False}
        _tc_cache[key] = state

    s = float(sum_G)
    fit = state["fit"]
    if fit is not None and fit["s_lo"] <= s <= fit["s_hi"]:
        U = fit["I_A"] + fit["I_B"] / s
        return np.ascontiguousarray(U[:, :, None] * weights[None, None, :])

    # exact unit-weight build (the couplings enter the kernel only through sum_G)
    U = build_thermal_grid(E_bins, z_bins, s, np.ones(3), m_Zd, m_nu_bg_arr, T_nu0,
                           eta_mass, thermal_analytic=True)[:, :, 0].copy()

    if not state["disabled"]:
        pts = [(ps, pu) for (ps, pu) in state["points"] if ps != s]
        pts.append((s, U))
        if len(pts) > _TC_MAX_POINTS:
            pts.sort(key=lambda t: abs(np.log10(t[0] / s)))
            pts = pts[:_TC_MAX_POINTS]
        pts.sort(key=lambda t: t[0])
        state["points"] = pts
        _tc_try_fit(state, s)

    return np.ascontiguousarray(U[:, :, None] * weights[None, None, :])



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

