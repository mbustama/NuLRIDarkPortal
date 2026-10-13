"""
dp_si.py

Secret neutrino interactions (begun 2026-10-13): a mediator phi of mass M through which neutrinos
interact with each other, couplings g_ab in the flavor basis. High-energy neutrinos scatter on the
relic neutrinos, which absorbs them and regenerates lower-energy ones (Bustamante, Rosenstroem,
Shalgar & Tamborra, arXiv:2001.04994). This module holds the cross sections, the secondaries'
spectra and the cosmological transport; it replaces the dark portal when
secret_interactions.enabled is true (dark_portal_solver).

Scalar mediator, Majorana neutrinos: the cross sections of Esteban, Pandey, Brdar & Beacom
(arXiv:2107.13568, Appendix A; their code is not used) -- s, t and u channels and their
interference (Eqs. A11, A12) and double-scalar production above s = 4 M^2 (A11, A13), for an
ultrarelativistic left-handed incident neutrino on a relic at rest and unpolarized; width A15,
decay spectrum A16 (flat in the lab).

Conventions and choices:
  - Mass-basis couplings g_ij = sum_ab U*_ai U_bj g_ab, taken real (no CP violation, as in
    2107.13568; exact for a single coupled flavor, where the phases drop out of every rate, and
    for a flavor-universal coupling).
  - s = 2 E m_j for incident energy E (physical, at the redshift of the interaction) on a relic
    of mass m_j at rest. Relics with m_j = 0 do not scatter (s = 0); their thermal momenta are
    neglected throughout, as in both papers (an open item for a nearly massless lightest state).
  - Targets: the cross sections are summed over the relic's spin, and the appendix's "half the
    number of targets" is the density per helicity state, n = 3 zeta(3) T^3 / (4 pi^2) per mass
    state (56 cm^-3 today; nuSIprop's get_nd). Narrow-width check: an incident nu_L on an
    unpolarized relic has sigma = (pi g^2 / 2) delta(s - M^2) per relic, against 2 x 56 relics;
    the appendix's s-channel term at resonance is pi g^2 delta(s - M^2), against 56.
  - Double-scalar production: the amplitudes A3, A4 sum the exchanged nu_k coherently (the
    neutrino masses neglected in the propagators), so the rate carries ((g^2)_ij)^2 =
    (sum_k g_ik g_kj)^2. The paper's Eqs. A10, A11, A13 print sum_k g_ik^2 g_jk^2, the
    incoherent sum -- equal for a flavor-universal coupling, smaller by sum_k |U_fk|^4 (~0.35
    for nu_tau) for a single flavor f. Default "coherent"; "paper" reproduces their figures.
  - Final states: t = (p_i - p_k)^2 for the outgoing neutrino k, so E_k = E (1 + t/s): the
    energy fraction x = E_k / E has dsigma/dx = s dsigma/dt at t = -s (1 - x), u = -s x. For a
    scalar final state, x_A = 1 + (t - M^2)/s. Each scalar decays at once into two neutrinos,
    flat in [0, E_phi] (A16).
  - The transport follows the sum of both helicities of each mass state (neutrinos plus
    antineutrinos). With real couplings the interaction is CP-symmetric, so this sum evolves
    exactly whatever the source's nubar/nu ratio; only its split is not modelled.
"""
import numpy as np

from dp_core import njit, prange

_ZETA3 = 1.2020569031595942


def _gl01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return np.ascontiguousarray(0.5 * (x + 1.0)), np.ascontiguousarray(0.5 * w)


_GL4_X, _GL4_W = _gl01(4)
_GL6_X, _GL6_W = _gl01(6)
_GL8_X, _GL8_W = _gl01(8)
_GL16_X, _GL16_W = _gl01(16)
_GL24_X, _GL24_W = _gl01(24)

# Profile indices of the regeneration kernels (secondary energy fraction x = E_k / E):
#   0 flat (s channel)        1  P2 = (1-x)^2/((1-x)+mu)^2 (t)   2  P3 = x^2/(x+mu)^2 (u)
#   3 P4 = (1-x)/((1-x)+mu)   4  P5 = x(1-x)/((x+mu)(1-x+mu))    5  P6 = x/(x+mu)
#   6 the neutrinos from double-scalar production (normalized to 1 per neutrino)
# with mu = M^2 / s. Absorption channels: 0 s, 1 t+u, 2 s-t and s-u interference, 3 t-u
# interference, 4 double-scalar production.
N_PROFILES = 7
N_CHANNELS = 5


# =============================================================================
# COUPLINGS AND WIDTH
# =============================================================================

def flavor_couplings(texture, g, g_matrix=None):
    """The flavor-basis coupling matrix g_ab from a texture name: 'universal' (g delta_ab),
    'e', 'mu', 'tau' (g on that flavor alone), or 'matrix' (g times the real symmetric 3x3
    g_matrix)."""
    t = str(texture).strip().lower()
    out = np.zeros((3, 3))
    if t == "universal":
        out[:] = g * np.eye(3)
    elif t in ("e", "mu", "tau"):
        f = {"e": 0, "mu": 1, "tau": 2}[t]
        out[f, f] = g
    elif t == "matrix":
        if g_matrix is None:
            raise ValueError("secret interactions: texture 'matrix' needs g_matrix")
        m = np.asarray(g_matrix, dtype=np.float64)
        if m.shape != (3, 3) or np.max(np.abs(m - m.T)) > 1e-12 * max(np.max(np.abs(m)), 1e-300):
            raise ValueError("secret interactions: g_matrix must be a real symmetric 3x3 matrix")
        out[:] = g * m
    else:
        raise ValueError(f"secret interactions: unknown texture {texture!r} "
                         "(universal, e, mu, tau, matrix)")
    return out


def mass_basis_couplings(g_flavor, U):
    """g_ij = sum_ab U*_ai U_bj g_ab. Returns the real 3x3 matrix used in the cross sections.

    For a single coupled flavor (g_ab = g delta_af delta_bf) the phases of U_fi factor out of
    every rate (2107.13568, Appendix A), and g_ij is taken as g |U_fi| |U_fj|; a flavor-universal
    coupling gives g delta_ij exactly. Other textures must give real g_ij (to 1e-12 of the
    largest), or this raises: their interference terms would need the CP-violating amplitudes."""
    g_flavor = np.asarray(g_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    nz = np.argwhere(np.abs(g_flavor) > 0.0)
    if len(nz) == 1 and nz[0][0] == nz[0][1]:
        f = nz[0][0]
        a = np.abs(U[f, :])
        return np.real(g_flavor[f, f]) * np.outer(a, a)
    g = np.conjugate(U).T @ g_flavor @ U          # g_ij = sum_ab conj(U_ai) g_ab U_bj
    scale = max(np.max(np.abs(g)), 1e-300)
    if np.max(np.abs(np.imag(g))) > 1e-12 * scale:
        raise ValueError("secret interactions: the mass-basis couplings are complex; only textures "
                         "with real g_ij (flavor-universal, a single flavor, or real matrices with "
                         "delta_CP = 0) are supported")
    return np.ascontiguousarray(np.real(g))


def scalar_width(M, g, majorana=True):
    """Total width of the scalar, Eq. (A15): Gamma = c M / (8 pi) sum_kl g_kl^2, c = 1/2 for
    Majorana neutrinos (identical final states), 1 for Dirac."""
    c = 0.5 if majorana else 1.0
    return c * M / (8.0 * np.pi) * float(np.sum(np.asarray(g) ** 2))


def relic_density_per_helicity(T_nu0):
    """Relic neutrinos per helicity state and mass state today (eV^3): 3 zeta(3) T^3 / (4 pi^2)
    (56.0 cm^-3 at T = 1.676389e-4 eV); the 'half the number of targets' of 2107.13568."""
    return 3.0 * _ZETA3 * T_nu0 ** 3 / (4.0 * np.pi ** 2)


# =============================================================================
# KINEMATIC INTEGRALS (stable at any s / M^2)
# =============================================================================

@njit(cache=True)
def _r_minus_log1p(r):
    """r - log(1 + r), without the cancellation below r ~ 1."""
    if r < 0.5:
        term = r * r
        tot = 0.0
        sgn = 1.0
        for n in range(2, 200):
            add = sgn * term / n
            tot += add
            if abs(add) <= 1e-17 * abs(tot):
                break
            term *= r
            sgn = -sgn
        return tot
    return r - np.log1p(r)


@njit(cache=True)
def profile_norms(r, glx, glw):
    """(N2, N4, N5): the integrals over x in [0, 1] of the profiles P2 (= P3), P4 (= P6) and P5
    at mu = 1/r, r = s / M^2. N2 = s I_tt, N5 = s I_tu with I_tt, I_tu the t- and tu-terms of
    Eq. (A11); N4 (1 - M^2/s) s / BW is its interference term. Below r = 0.5, where the closed
    forms cancel (N2 ~ r^2/3, N5 ~ r^2/6), Gauss-Legendre of the cancellation-free integrands
    r^2 y^2/(1+ry)^2 and r^2 y(1-y)/((1+ry)(1+r-ry)) (glx, glw on [0, 1], 24 nodes)."""
    N4 = _r_minus_log1p(r) / r
    if r < 0.5:
        a2 = 0.0
        a5 = 0.0
        for k in range(len(glx)):
            y = glx[k]
            d = 1.0 + r * y
            a2 += glw[k] * y * y / (d * d)
            a5 += glw[k] * y * (1.0 - y) / (d * (1.0 + r - r * y))
        return r * r * a2, N4, r * r * a5
    L = np.log1p(r)
    N2 = 1.0 + 1.0 / (1.0 + r) - 2.0 * L / r
    N5 = 1.0 - 2.0 * (1.0 + r) * L / (r * (2.0 + r))
    return N2, N4, N5


@njit(cache=True)
def phiphi_kinematic(s, M2):
    """I_phiphi(s) = (s - 2 M^2) Phi(w) / s^2 for s > 4 M^2 (0 below): the double-scalar cross
    section is sigma = C I_phiphi / (32 pi) (Majorana), C the coupling factor. The bracket of
    Eq. (A11) is (s - 2 M^2) Phi(w) with w = r / A, r = sqrt(s (s - 4 M^2)), A = s - 2 M^2,
        Phi(w) = 2 (3 - w^2) atanh(w) - 6 w = sum_{n>=2} 8 (n-1) / (4 n^2 - 1) w^(2n+1):
    the series near threshold (w -> 0), where the closed form cancels to ~w^5; above, the closed
    form with A^2 - r^2 = 4 M^4, i.e. atanh(w) = ln((A + r) / (2 M^2)) and 3 - w^2 = 2 + 4 M^4/A^2,
    exact as w -> 1 (where atanh of a rounded w loses ~eps / (1 - w))."""
    if s <= 4.0 * M2:
        return 0.0
    A = s - 2.0 * M2
    r = np.sqrt(s * (s - 4.0 * M2))
    w = r / A
    if w < 0.7:
        w2 = w * w
        term = w2 * w2 * w
        Phi = 0.0
        for n in range(2, 200):
            add = 8.0 * (n - 1) / (4.0 * n * n - 1.0) * term
            Phi += add
            if add <= 1e-17 * Phi:
                break
            term *= w2
    else:
        q = 2.0 * M2 / A
        Phi = 2.0 * (2.0 + q * q) * np.log((A + r) / (2.0 * M2)) - 6.0 * w
    return A * Phi / (s * s)


# =============================================================================
# CROSS SECTIONS (python, for checks and plots)
# =============================================================================

def _bw(s, M, Gam):
    return (s - M * M) ** 2 + M * M * Gam * Gam


def dsigma_dt_scalar_majorana(s, t, M, Gam, g, i, j, k, l):
    """dsigma/dt for nu_i + nu_j -> nu_k + nu_l, Eq. (A12), with u = -s - t (massless final
    states, relic mass neglected against s). The 1/2 for identical final states (k = l) is
    included. Arguments: s, t (eV^2; t in [-s, 0]); M, Gam (eV); g -- 3x3 mass-basis couplings;
    i, j, k, l -- mass indices. Returns eV^-4."""
    u = -s - t
    M2 = M * M
    bw = _bw(s, M, Gam)
    gij, gkl, gik, gjl, gil, gjk = g[i, j], g[k, l], g[i, k], g[j, l], g[i, l], g[j, k]
    br = (gij ** 2 * gkl ** 2 / bw
          + gik ** 2 * gjl ** 2 * (t * t / (s * s)) / (t - M2) ** 2
          + gil ** 2 * gjk ** 2 * (u * u / (s * s)) / (u - M2) ** 2
          + gij * gkl * gik * gjl * t * (1.0 - M2 / s) / (bw * (t - M2))
          + gik * gjl * gil * gjk * (t * u / (s * s)) / ((u - M2) * (t - M2))
          + gij * gkl * gil * gjk * u * (1.0 - M2 / s) / (bw * (u - M2)))
    sym = 0.5 if k == l else 1.0
    return sym * br / (8.0 * np.pi)


def phiphi_coupling(g, i, j, mode="coherent"):
    """Coupling factor of nu_i nu_j -> phi phi: ((g^2)_ij)^2 (coherent, the default) or
    sum_k g_ik^2 g_jk^2 (as printed in 2107.13568)."""
    if mode == "coherent":
        return float(np.dot(g[i, :], g[:, j])) ** 2
    if mode == "paper":
        return float(np.sum(g[i, :] ** 2 * g[j, :] ** 2))
    raise ValueError(f"phiphi coupling mode must be 'coherent' or 'paper', got {mode!r}")


def dsigma_dt_scalar_majorana_phiphi(s, t, M, g, i, j, mode="coherent"):
    """dsigma/dt for nu_i + nu_j -> phi phi, Eq. (A13) (coupling factor per `mode`), present for
    s > 4 M^2, with t = (p_i - p_A)^2 for the scalar A, inside phiphi_t_range. Returns eV^-4."""
    if s <= 4.0 * M * M:
        return 0.0
    M2 = M * M
    c = phiphi_coupling(g, i, j, mode)
    return -c / (32.0 * np.pi * s * s) * ((-2.0 * M2 + s + 2.0 * t) ** 2
                                          / (t * t * (-2.0 * M2 + s + t) ** 2)) * (s * t + (t - M2) ** 2)


def phiphi_t_range(s, M):
    """Kinematic limits of t = (p_i - p_A)^2 for double-scalar production:
    t_pm = M^2 - s/2 -+ sqrt(s (s/4 - M^2)), where s t + (t - M^2)^2 vanishes. (The appendix
    prints the limits without the factor sqrt(s) and the shift M^2.)"""
    M2 = M * M
    r = np.sqrt(max(s * (0.25 * s - M2), 0.0))
    return M2 - 0.5 * s - r, M2 - 0.5 * s + r


def sigma_scalar_majorana_channels(s, M, Gam, g, i, j, mode="coherent"):
    """The total cross section of nu_i on a relic nu_j, Eq. (A11), split into the absorption
    channels (s, t+u, s-t and s-u interference, t-u interference, double scalar), eV^-2. Uses
    the coupling identities of coupling_tensors (sums over the final states done)."""
    M2 = M * M
    r = s / M2
    N2, N4, N5 = profile_norms(r, _GL24_X, _GL24_W)
    bw = _bw(s, M, Gam)
    g2 = g @ g
    g3 = g2 @ g
    G2 = float(np.sum(g * g))
    out = np.empty(5)
    out[0] = g[i, j] ** 2 * G2 / (16.0 * np.pi) * s / bw
    out[1] = 2.0 * g2[i, i] * g2[j, j] / (16.0 * np.pi) * N2 / s
    out[2] = 2.0 * g[i, j] * g3[i, j] / (16.0 * np.pi) * (s - M2) / bw * N4
    out[3] = g2[i, j] ** 2 / (16.0 * np.pi) * N5 / s
    out[4] = phiphi_coupling(g, i, j, mode) / (32.0 * np.pi) * phiphi_kinematic(s, M2)
    return out


def sigma_scalar_majorana_total(s, M, Gam, g, i, j, mode="coherent"):
    """Total cross section of nu_i on a relic nu_j, Eq. (A11). eV^-2."""
    return float(np.sum(sigma_scalar_majorana_channels(s, M, Gam, g, i, j, mode)))


def sigma_scalar_majorana_A11(s, M, Gam, g, i, j, mode="paper"):
    """Eq. (A11) as printed (closed forms, sum over all k, l), for checks; accurate only for
    s / M^2 >~ 0.5 (its t and tu terms cancel below)."""
    M2 = M * M
    bw = _bw(s, M, Gam)
    L = np.log(M2 / (M2 + s))
    tot = 0.0
    for k in range(3):
        for l in range(3):
            gij, gkl, gik, gjl, gil, gjk = g[i, j], g[k, l], g[i, k], g[j, l], g[i, l], g[j, k]
            tot += (gij ** 2 * gkl ** 2 * s / bw
                    + (gik ** 2 * gjl ** 2 + gil ** 2 * gjk ** 2) * ((s + 2 * M2) / (s * (s + M2)) + 2 * M2 / s ** 2 * L)
                    - gij * gkl * (gik * gjl + gil * gjk) * (M2 / s - 1.0) * (s + M2 * L) / bw
                    + gik * gjl * gil * gjk * (1.0 / s + 2 * M2 * (M2 + s) / (s ** 2 * (2 * M2 + s)) * L))
    tot /= 16.0 * np.pi
    if s > 4.0 * M2:
        r = np.sqrt(s * (s - 4.0 * M2))
        c = phiphi_coupling(g, i, j, mode)
        tot += c / (32.0 * np.pi * s * s) * ((s * s - 4 * M2 * s + 6 * M2 * M2) / (s - 2 * M2)
                                            * np.log(((r + s - 2 * M2) / (r - s + 2 * M2)) ** 2) - 6.0 * r)
    return tot


# =============================================================================
# COUPLING TENSORS OF THE TRANSPORT
# =============================================================================

def coupling_tensors(g, mode="coherent", channels="full"):
    """Coefficients of the absorption and of the secondaries, summed over the final states.

    Absorption of mass state a on relic r (cross section, eV^-2):
        sigma_ar = Cabs[0] s/BW + (Cabs[1] N2 + Cabs[3] N5)/s + Cabs[2] N4 (s - M^2)/BW
                   + Cabs[4] I_phiphi(s)
    with N2, N4, N5 = profile_norms(s/M^2). Secondaries of mass state c per unit energy fraction x
    (the same normalization, summed over the partner):
        dN_c/dx = Creg[0] s/BW + (Creg[1] P2 + Creg[2] P3 + Creg[4] P5)/s
                  + (Creg[3] P4 + Creg[5] P6) (s - M^2)/BW + Creg[6] I_phiphi p_phiphi(x)
    with p_phiphi normalized to 1. Every 2 -> 2 absorption yields two secondaries and every
    double-scalar one four: sum_c Creg[0] = 2 Cabs[0], sum_c (Creg[1] + Creg[2]) = 2 Cabs[1], ...

    channels: 'full'; 's' or 'delta' keep the s channel alone (2107.13568's approximations).
    Returns (Cabs (5, 3, 3) [p, a, r], Creg (7, 3, 3, 3) [q, a, r, c], G2 = sum g_kl^2)."""
    g = np.asarray(g, dtype=np.float64)
    g2 = g @ g
    g3 = g2 @ g
    G2 = float(np.sum(g * g))
    Cabs = np.zeros((N_CHANNELS, 3, 3))
    Creg = np.zeros((N_PROFILES, 3, 3, 3))
    pi8, pi16 = 8.0 * np.pi, 16.0 * np.pi
    for a in range(3):
        for r in range(3):
            Cabs[0, a, r] = g[a, r] ** 2 * G2 / pi16
            Cabs[1, a, r] = 2.0 * g2[a, a] * g2[r, r] / pi16
            Cabs[2, a, r] = 2.0 * g[a, r] * g3[a, r] / pi16
            Cabs[3, a, r] = g2[a, r] ** 2 / pi16
            Cabs[4, a, r] = phiphi_coupling(g, a, r, mode) / (32.0 * np.pi)
            for c in range(3):
                Creg[0, a, r, c] = g[a, r] ** 2 * g2[c, c] / pi8
                Creg[1, a, r, c] = g[a, c] ** 2 * g2[r, r] / pi8
                Creg[2, a, r, c] = g2[a, a] * g[r, c] ** 2 / pi8
                Creg[3, a, r, c] = g[a, r] * g[a, c] * g2[c, r] / pi8
                Creg[4, a, r, c] = g[a, c] * g[r, c] * g2[r, a] / pi8
                Creg[5, a, r, c] = g[a, r] * g[r, c] * g2[c, a] / pi8
                Creg[6, a, r, c] = (4.0 * Cabs[4, a, r] * g2[c, c] / G2) if G2 > 0 else 0.0
    ch = str(channels).strip().lower()
    if ch in ("s", "delta"):
        Cabs[1:] = 0.0
        Creg[1:] = 0.0
    elif ch != "full":
        raise ValueError(f"secret interactions: channels must be 'full', 's' or 'delta', got {channels!r}")
    return np.ascontiguousarray(Cabs), np.ascontiguousarray(Creg), G2


# =============================================================================
# REGENERATION PROFILES: BIN INTEGRALS
# =============================================================================

@njit(cache=True)
def _profiles_at(x, mu, out):
    """P2..P6 at x (out[0..4])."""
    v = 1.0 - x
    out[0] = v * v / ((v + mu) * (v + mu))
    out[1] = x * x / ((x + mu) * (x + mu))
    out[2] = v / (v + mu)
    out[3] = x * v / ((x + mu) * (v + mu))
    out[4] = x / (x + mu)


@njit(cache=True)
def _profile_bin_integrals(x_lo, x_hi, mu, glx, glw, out, tmp):
    """Integrals of P2..P6 over [x_lo, x_hi] (out[0..4]). Gauss-Legendre (glx, glw on [0, 1])
    unless the poles at x = 1 + mu come within the bin's width (mu small, top bins): there the
    closed forms, which are stable for mu < bin width."""
    D = x_hi - x_lo
    if mu < D and (1.0 - x_hi) < 2.0 * D:
        ylo = 1.0 - x_lo + mu
        yhi = 1.0 - x_hi + mu
        ly = np.log(yhi / ylo)
        lx = np.log((x_hi + mu) / (x_lo + mu))
        out[0] = D + 2.0 * mu * ly + mu * mu * D / (ylo * yhi)
        out[1] = D - 2.0 * mu * lx + mu * mu * D / ((x_lo + mu) * (x_hi + mu))
        out[2] = D + mu * ly
        out[3] = D - mu * (1.0 + mu) / (1.0 + 2.0 * mu) * (lx - ly)
        out[4] = D - mu * lx
        return
    for p in range(5):
        out[p] = 0.0
    for k in range(len(glx)):
        _profiles_at(x_lo + D * glx[k], mu, tmp)
        for p in range(5):
            out[p] += glw[k] * tmp[p]
    for p in range(5):
        out[p] *= D


@njit(cache=True)
def _phiphi_g(u, beta, delta):
    """Integrand of the scalar's energy distribution in u = ln((x - x_-)/(x_+ - x)):
    (2x - 1)^2 (a/(a+delta))^2 (b/(b+delta))^2 / beta with a = x - x_-, b = x_+ - x; its integral
    over u is (1 - 2 mu) Phi(w). Returns (g, x)."""
    if u >= 0.0:
        e = np.exp(-u)
        a = beta / (1.0 + e)
        b = beta * e / (1.0 + e)
    else:
        e = np.exp(u)
        a = beta * e / (1.0 + e)
        b = beta / (1.0 + e)
    x = 0.5 * (1.0 - beta) + a
    ra = a / (a + delta)
    rb = b / (b + delta)
    return (2.0 * x - 1.0) ** 2 * ra * ra * rb * rb / beta, x


@njit(cache=True)
def _phiphi_nu_bins(mu, edges, n_u, out):
    """Bin integrals of the neutrino spectrum of double-scalar production per neutrino,
    p(y) = int_y^1 f(x)/x dx with f the scalar's normalized energy distribution (Eq. A13 in
    x = E_A/E) and each scalar decaying flat in [0, x] (A16): out[d] = int over
    [edges[d+1], edges[d]] (edges descending from 1). Integrates in u = ln((x - x_-)/(x_+ - x)),
    where the distribution's endpoint peaks (width ~mu^2) are smooth, on n_u nodes with the
    end-corrected trapezoid rule; H(y) = int_0^y p = F0(y) + y F1(y), F0 = int_{x_-}^y f,
    F1 = int_y^{x_+} f/x, by cubic Hermite interpolation of the cumulative integrals."""
    nb = len(edges) - 1
    for d in range(nb):
        out[d] = 0.0
    if mu >= 0.25:
        return
    beta = np.sqrt(1.0 - 4.0 * mu)
    delta = 4.0 * mu * mu / ((1.0 + beta) * (1.0 + beta))
    x_m = mu + delta
    x_p = 1.0 - mu - delta
    U = np.log(beta / delta) + 25.0
    h = 2.0 * U / (n_u - 1)
    gv = np.empty(n_u)
    gx = np.empty(n_u)
    xv = np.empty(n_u)
    for k in range(n_u):
        gv[k], xv[k] = _phiphi_g(-U + k * h, beta, delta)
        gx[k] = gv[k] / xv[k]
    # cumulative trapezoid with the Euler-Maclaurin end correction (O(h^4)): C0 from the left,
    # C1 from the right
    C0 = np.empty(n_u)
    C1 = np.empty(n_u)
    C0[0] = 0.0
    for k in range(1, n_u):
        C0[k] = C0[k - 1] + 0.5 * h * (gv[k - 1] + gv[k])
    C1[n_u - 1] = 0.0
    for k in range(n_u - 2, -1, -1):
        C1[k] = C1[k + 1] + 0.5 * h * (gx[k] + gx[k + 1])
    d0 = np.empty(n_u)
    d1 = np.empty(n_u)
    for k in range(n_u):
        km = max(k - 1, 0)
        kp = min(k + 1, n_u - 1)
        d0[k] = (gv[kp] - gv[km]) / ((kp - km) * h)
        d1[k] = (gx[kp] - gx[km]) / ((kp - km) * h)
    for k in range(n_u):
        C0[k] -= h * h / 12.0 * (d0[k] - d0[0])
        C1[k] -= h * h / 12.0 * (d1[n_u - 1] - d1[k])
    Z = C0[n_u - 1]
    T1 = C1[0]
    Hprev = 0.0
    for d in range(nb, -1, -1):
        y = edges[d]
        if y <= x_m:
            H = y * T1
        elif y >= x_p:
            H = Z
        else:
            u = np.log((y - x_m) / (x_p - y))
            f = (u + U) / h
            k = int(f)
            if k < 0:
                k = 0
            elif k > n_u - 2:
                k = n_u - 2
            tt = f - k
            h00 = 2 * tt ** 3 - 3 * tt ** 2 + 1
            h10 = tt ** 3 - 2 * tt ** 2 + tt
            h01 = -2 * tt ** 3 + 3 * tt ** 2
            h11 = tt ** 3 - tt ** 2
            F0 = h00 * C0[k] + h10 * h * gv[k] + h01 * C0[k + 1] + h11 * h * gv[k + 1]
            F1 = h00 * C1[k] - h10 * h * gx[k] + h01 * C1[k + 1] - h11 * h * gx[k + 1]
            H = F0 + y * F1
        if d < nb:
            out[d] = (H - Hprev) / Z       # bin d = [edges[d+1], edges[d]]
        Hprev = H


@njit(cache=True, parallel=True)
def build_profile_table(mu_list, N_child, dlnE, n_u):
    """Bin integrals of the seven regeneration profiles for every mu in mu_list and every child
    offset d = 0..N_child-1 on a geometric grid of log-spacing dlnE: child d covers the energy
    fractions [h^(-2d-1), h^(-2d+1)] of its parent (d = 0: [1/h, 1]), h = exp(dlnE/2).
    Returns B (len(mu_list), N_child, 7): flat, P2, P3, P4, P5, P6, double-scalar neutrinos."""
    n_mu = len(mu_list)
    B = np.zeros((n_mu, N_child, N_PROFILES))
    h = np.exp(0.5 * dlnE)
    edges = np.empty(N_child + 1)
    edges[0] = 1.0
    for d in range(1, N_child + 1):
        edges[d] = h ** (-(2 * d - 1))
    glx16, glw16 = _GL16_X, _GL16_W
    glx6, glw6 = _GL6_X, _GL6_W
    for n in prange(n_mu):
        mu = mu_list[n]
        tmp = np.empty(5)
        out5 = np.empty(5)
        for d in range(N_child):
            x_hi = edges[d]
            x_lo = edges[d + 1]
            B[n, d, 0] = x_hi - x_lo
            if d < 8:
                _profile_bin_integrals(x_lo, x_hi, mu, glx16, glw16, out5, tmp)
            else:
                _profile_bin_integrals(x_lo, x_hi, mu, glx6, glw6, out5, tmp)
            for p in range(5):
                B[n, d, 1 + p] = out5[p]
        if mu < 0.25:
            pb = np.empty(N_child)
            _phiphi_nu_bins(mu, edges, n_u, pb)
            for d in range(N_child):
                B[n, d, 6] = pb[d]
    return B


@njit(cache=True, parallel=True)
def build_profile_table_tri(mu_list, d_max, dlnE, n_u):
    """build_profile_table in triangular storage: node n keeps the children d = 0..d_max[n]
    only (a parent j reaches d <= j). Returns (B (total, 7), offsets (len(mu_list) + 1)), the
    rows of node n being B[offsets[n]:offsets[n+1]]."""
    n_mu = len(mu_list)
    offsets = np.zeros(n_mu + 1, dtype=np.int64)
    for n in range(n_mu):
        offsets[n + 1] = offsets[n] + d_max[n] + 1
    B = np.zeros((offsets[n_mu], N_PROFILES))
    h = np.exp(0.5 * dlnE)
    n_edges = 0
    for n in range(n_mu):
        n_edges = max(n_edges, d_max[n] + 2)
    edges = np.empty(n_edges)
    edges[0] = 1.0
    for d in range(1, n_edges):
        edges[d] = h ** (-(2 * d - 1))
    glx16, glw16 = _GL16_X, _GL16_W
    glx6, glw6 = _GL6_X, _GL6_W
    for n in prange(n_mu):
        mu = mu_list[n]
        o = offsets[n]
        nd = d_max[n] + 1
        tmp = np.empty(5)
        out5 = np.empty(5)
        for d in range(nd):
            x_hi = edges[d]
            x_lo = edges[d + 1]
            B[o + d, 0] = x_hi - x_lo
            if d < 8:
                _profile_bin_integrals(x_lo, x_hi, mu, glx16, glw16, out5, tmp)
            else:
                _profile_bin_integrals(x_lo, x_hi, mu, glx6, glw6, out5, tmp)
            for p in range(5):
                B[o + d, 1 + p] = out5[p]
        if mu < 0.25:
            pb = np.empty(nd)
            _phiphi_nu_bins(mu, edges[:nd + 1], n_u, pb)
            for d in range(nd):
                B[o + d, 6] = pb[d]
    return B, offsets


# =============================================================================
# COSMOLOGICAL TRANSPORT
# =============================================================================
#
# Variables as in the dark-portal solver (dp_cosmo.step_cosmological_pair): F[j, a] is the
# comoving number per unit OBSERVED energy E_j of mass state a (both helicities), on a fixed
# geometric grid, so redshift losses need no term of their own; the physical energy at redshift z
# is E_j (1 + z). The source term is Phi_0 (E_j (1+z))^-p R(z) per unit time, split over the mass
# states as sum_alpha |U_alpha a|^2 f_alpha,S.
#
# Steps are uniform in ln(1+z), Delta = dlnE / z_substeps, so that the scale mu = M^2 / s of a
# bin at a step's midpoint, mu = M^2 / (2 E_0 m_r) exp(-(j q + m + 1/2) Delta), falls on one lattice
# per relic state: the regeneration profiles are tabulated once per run (build_profile_table_tri)
# and depend on the mediator mass and the relic masses only, not on the couplings. The top step,
# shortened so that it starts at z_max exactly, has a table of its own.
#
# Per step and bin, the optical depth is integrated exactly over the step: the Breit-Wigner terms
# (s channel and its interference) through theta = atan((s - M^2)/(M Gamma)), in which any
# resonance -- however narrow -- is smooth, and the others by Gauss-Legendre in s. The bins are
# then swept from high to low energy, each solving its equations over the step exactly (constant
# rates and sources; dp_cosmo._absorption_factors) and passing down what it absorbed: its own
# secondaries (a 3 x 3 system over the mass states) and those landing in lower bins, with the
# exact bin integrals of their spectra. Every 2 -> 2 absorption yields two secondaries and every
# double-scalar one four, less what falls below the grid.

@njit(cache=True)
def _absorption_factors(G, dt):
    """(exp(-G dt), (1 - exp(-G dt))/G, dt - a1); as dp_cosmo._absorption_factors."""
    x = G * dt
    if x < 1e-3:
        e = np.exp(-x)
        a1 = dt * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0))))
        g = dt * x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x * (1.0 / 120.0 - x / 720.0))))
        return e, a1, g
    e = np.exp(-x)
    a1 = -np.expm1(-x) / G
    return e, a1, dt - a1


@njit(cache=True)
def _w_of_s(s, Em2, n0r, H0, Om, OL):
    """n_r(z) dt/ds at s = Em2 (1 + z), Em2 = 2 E m_r: n0r (1+z)^2 / (Em2 H(z))."""
    zp1 = s / Em2
    return n0r * zp1 * zp1 / (Em2 * H0 * np.sqrt(Om * zp1 ** 3 + OL))


@njit(cache=True, parallel=True)
def _step_integrals(E, zp1_lo, zp1_hi, zp1_mid, masses, n0, M2, MG, H0, Om, OL, delta_mode,
                    Js, Jint, Jinv, Jpp, N2, N4, N5):
    """Per bin j and relic r, the step's time integrals of the target density times
        s/BW (Js), (s - M^2)/BW (Jint), 1/s (Jinv), I_phiphi(s) (Jpp),
    and the profile norms at the step's midpoint (N2, N4, N5). BW = (s - M^2)^2 + (M Gamma)^2."""
    N = len(E)
    b = MG
    for j in prange(N):
        for r in range(3):
            Js[j, r] = 0.0
            Jint[j, r] = 0.0
            Jinv[j, r] = 0.0
            Jpp[j, r] = 0.0
            N2[j, r] = 0.0
            N4[j, r] = 0.0
            N5[j, r] = 0.0
            m = masses[r]
            if m <= 0.0 or n0[r] <= 0.0:
                continue
            Em2 = 2.0 * E[j] * m
            s_lo = Em2 * zp1_lo
            s_hi = Em2 * zp1_hi
            rr = Em2 * zp1_mid / M2
            n2, n4, n5 = profile_norms(rr, _GL24_X, _GL24_W)
            N2[j, r] = n2
            N4[j, r] = n4
            N5[j, r] = n5
            # smooth terms: Gauss-Legendre in s
            ds = s_hi - s_lo
            a = 0.0
            for k in range(len(_GL8_X)):
                s = s_lo + ds * _GL8_X[k]
                a += _GL8_W[k] * _w_of_s(s, Em2, n0[r], H0, Om, OL) / s
            Jinv[j, r] = a * ds
            if s_hi > 4.0 * M2:
                sa = max(s_lo, 4.0 * M2)
                dsp = s_hi - sa
                a = 0.0
                for k in range(len(_GL8_X)):
                    s = sa + dsp * _GL8_X[k]
                    a += _GL8_W[k] * _w_of_s(s, Em2, n0[r], H0, Om, OL) * phiphi_kinematic(s, M2)
                Jpp[j, r] = a * dsp
            # Breit-Wigner terms
            v_lo = s_lo - M2
            v_hi = s_hi - M2
            if delta_mode:
                if v_lo <= 0.0 < v_hi:
                    Js[j, r] = M2 * _w_of_s(M2, Em2, n0[r], H0, Om, OL) * np.pi / b
                continue
            near = 0.0
            if v_lo > 0.0:
                near = v_lo
            elif v_hi < 0.0:
                near = -v_hi
            if near > 2.0 * ds:
                # the resonance is far outside the step: plain Gauss-Legendre in s
                a1_ = 0.0
                a2_ = 0.0
                for k in range(len(_GL8_X)):
                    s = s_lo + ds * _GL8_X[k]
                    v = s - M2
                    wbw = _w_of_s(s, Em2, n0[r], H0, Om, OL) / (v * v + b * b)
                    a1_ += _GL8_W[k] * wbw * s
                    a2_ += _GL8_W[k] * wbw * v
                Js[j, r] = a1_ * ds
                Jint[j, r] = a2_ * ds
            else:
                # theta = atan(v / b): int A(s) ds / (v^2 + b^2) = (1/b) int A(s(theta)) dtheta
                dth = np.arctan2(b * (v_hi - v_lo), b * b + v_hi * v_lo)
                a1_ = 0.0
                a2_ = 0.0
                for k in range(len(_GL8_X)):
                    tp = np.tan(dth * _GL8_X[k])
                    v = (v_lo + b * tp) / (1.0 - v_lo * tp / b)
                    s = v + M2
                    wv = _w_of_s(s, Em2, n0[r], H0, Om, OL)
                    a1_ += _GL8_W[k] * wv * s
                    a2_ += _GL8_W[k] * wv * v
                Js[j, r] = a1_ * dth / b
                Jint[j, r] = a2_ * dth / b


@njit(cache=True)
def _si_sweep(F, E, dE, S0, Js, Jint, Jinv, Jpp, N2, N4, N5, Cabs, Creg,
              tabB, tabO, node, dt, reg, diag):
    """One redshift step of the transport (module notes above). F (N, 3) is updated in place.
    S0 (N, 3): injection rate per unit energy over the step. tabB (3, rows, 7) / tabO (3, nodes+1):
    each relic's profile table (triangular); node (N, 3): the table node of bin j and relic r.
    reg (N, 3): scratch for the regeneration rates. diag (6,) accumulates: absorbed number,
    secondaries in the grid, secondaries expected (in the grid or below it), absorbed energy,
    energy of the secondaries in the grid (at the bin centres), injected number."""
    N = F.shape[0]
    for i in range(N):
        for c in range(3):
            reg[i, c] = 0.0
    tau = np.empty(3)
    Wn = np.zeros((3, N_PROFILES, 3, 3))       # [r, q, a, c] per absorbed a
    Q = np.empty((3, 3))
    Y = np.empty((3, N_PROFILES, 3))
    e_ = np.empty(3)
    a1_ = np.empty(3)
    g_ = np.empty(3)
    rhs = np.empty(3)
    R = np.empty(3)
    S = np.empty(3)
    Am = np.empty((3, 3))
    Jq = np.empty(N_PROFILES)
    Nq = np.empty(N_PROFILES)
    for j in range(N - 1, -1, -1):
        for a in range(3):
            t = 0.0
            for r in range(3):
                t += (Cabs[0, a, r] * Js[j, r] + (Cabs[1, a, r] * N2[j, r] + Cabs[3, a, r] * N5[j, r]) * Jinv[j, r]
                      + Cabs[2, a, r] * N4[j, r] * Jint[j, r] + Cabs[4, a, r] * Jpp[j, r])
            tau[a] = t
        expected = 0.0
        for r in range(3):
            Jq[0] = Js[j, r]
            Jq[1] = Jinv[j, r]
            Jq[2] = Jinv[j, r]
            Jq[3] = Jint[j, r]
            Jq[4] = Jinv[j, r]
            Jq[5] = Jint[j, r]
            Jq[6] = Jpp[j, r]
            Nq[0] = 1.0
            Nq[1] = N2[j, r]
            Nq[2] = N2[j, r]
            Nq[3] = N4[j, r]
            Nq[4] = N5[j, r]
            Nq[5] = N4[j, r]
            Nq[6] = 1.0
            for q in range(N_PROFILES):
                for a in range(3):
                    for c in range(3):
                        Wn[r, q, a, c] = Creg[q, a, r, c] * Jq[q] / tau[a] if tau[a] > 0.0 else 0.0
        # own-bin secondaries per absorbed a
        for c in range(3):
            for a in range(3):
                Q[c, a] = 0.0
        for r in range(3):
            o = tabO[r, node[j, r]]
            for q in range(N_PROFILES):
                b0 = tabB[r, o, q]
                for a in range(3):
                    for c in range(3):
                        Q[c, a] += Wn[r, q, a, c] * b0
        for a in range(3):
            e_[a], a1_[a], g_[a] = _absorption_factors(tau[a] / dt, dt)
            S[a] = S0[j, a] + reg[j, a]
            rhs[a] = F[j, a] * (1.0 - e_[a]) + S[a] * g_[a]
        for c in range(3):
            for a in range(3):
                Am[c, a] = (dt if c == a else 0.0) - g_[c] * Q[c, a]
        # 3 x 3 solve (cofactors; Am ~ dt (1 - O(dlnE)), well conditioned)
        c00 = Am[1, 1] * Am[2, 2] - Am[1, 2] * Am[2, 1]
        c01 = Am[1, 2] * Am[2, 0] - Am[1, 0] * Am[2, 2]
        c02 = Am[1, 0] * Am[2, 1] - Am[1, 1] * Am[2, 0]
        det = Am[0, 0] * c00 + Am[0, 1] * c01 + Am[0, 2] * c02
        c10 = Am[0, 2] * Am[2, 1] - Am[0, 1] * Am[2, 2]
        c11 = Am[0, 0] * Am[2, 2] - Am[0, 2] * Am[2, 0]
        c12 = Am[0, 1] * Am[2, 0] - Am[0, 0] * Am[2, 1]
        c20 = Am[0, 1] * Am[1, 2] - Am[0, 2] * Am[1, 1]
        c21 = Am[0, 2] * Am[1, 0] - Am[0, 0] * Am[1, 2]
        c22 = Am[0, 0] * Am[1, 1] - Am[0, 1] * Am[1, 0]
        R[0] = (c00 * rhs[0] + c10 * rhs[1] + c20 * rhs[2]) / det
        R[1] = (c01 * rhs[0] + c11 * rhs[1] + c21 * rhs[2]) / det
        R[2] = (c02 * rhs[0] + c12 * rhs[1] + c22 * rhs[2]) / det
        own = 0.0
        for c in range(3):
            sc = S[c]
            for a in range(3):
                sc += Q[c, a] * R[a]
                own += Q[c, a] * R[a]
            F[j, c] = F[j, c] * e_[c] + sc * a1_[c]
            diag[5] += S0[j, c] * dt * dE[j]
        absorbed = 0.0
        for a in range(3):
            absorbed += R[a]
        diag[0] += absorbed * dt * dE[j]
        diag[3] += absorbed * dt * dE[j] * E[j]
        diag[1] += own * dt * dE[j]
        diag[4] += own * dt * dE[j] * E[j]
        # secondaries expected per absorbed a (in the grid or below it)
        for r in range(3):
            Nq[0] = 1.0
            Nq[1] = N2[j, r]
            Nq[2] = N2[j, r]
            Nq[3] = N4[j, r]
            Nq[4] = N5[j, r]
            Nq[5] = N4[j, r]
            Nq[6] = 1.0
            for q in range(N_PROFILES):
                for a in range(3):
                    for c in range(3):
                        expected += R[a] * Wn[r, q, a, c] * Nq[q]
        diag[2] += expected * dt * dE[j]
        # pass the secondaries down
        for r in range(3):
            for q in range(N_PROFILES):
                for c in range(3):
                    y = 0.0
                    for a in range(3):
                        y += R[a] * Wn[r, q, a, c]
                    Y[r, q, c] = y * dE[j]
        pushed = 0.0
        for r in range(3):
            o = tabO[r, node[j, r]]
            nrow = tabO[r, node[j, r] + 1] - o
            dmax = min(j, nrow - 1)
            for d in range(1, dmax + 1):
                i = j - d
                inv = 1.0 / dE[i]
                for c in range(3):
                    acc = 0.0
                    for q in range(N_PROFILES):
                        acc += Y[r, q, c] * tabB[r, o + d, q]
                    reg[i, c] += acc * inv
                    pushed += acc
                    diag[4] += acc * dt * E[i]
        diag[1] += pushed * dt


def _step_time_and_injection(L_lo, L_hi, H0, Om, OL, p_index, rho_evol, Phi_0):
    """(dt, T_inj) over the step ln(1+z) in [L_lo, L_hi]: dt = int dL / H(z) and
    T_inj = int Phi_0 (1+z)^-p R(z) dL / H(z) (dt = dz / (H (1+z)) = dL / H), by Gauss-Legendre
    split at the kinks of R(z) (z = 1 and z = 4)."""
    from dp_cosmo import calc_R_source
    cuts = [L_lo] + [L for L in (np.log(2.0), np.log(5.0)) if L_lo < L < L_hi] + [L_hi]
    dt = 0.0
    T = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        for x, w in zip(_GL8_X, _GL8_W):
            L = a + (b - a) * x
            zp1 = np.exp(L)
            Hz = H0 * np.sqrt(Om * zp1 ** 3 + OL)
            dt += w * (b - a) / Hz
            T += w * (b - a) * Phi_0 * zp1 ** (-p_index) * calc_R_source(zp1 - 1.0, rho_evol) / Hz
    return dt, T


def _tri_table(mu0, n_nodes, stride, N, dlnE, Delta, n_u, offset=0.5, mu_list=None):
    """Triangular profile table for nodes n = 0..n_nodes-1 at mu = mu0 exp(-(n + offset) Delta)
    (or at mu_list), children d <= min(n // stride, N - 1)."""
    n = np.arange(n_nodes)
    if mu_list is None:
        mu_list = mu0 * np.exp(-(n + offset) * Delta)
    d_max = np.minimum(n // stride, N - 1).astype(np.int64)
    return build_profile_table_tri(np.ascontiguousarray(mu_list, dtype=np.float64), d_max, dlnE, n_u)


def _stack_tables(tabs):
    """Pad the relics' (B, offsets) into (3, rows, 7) and (3, nodes + 1) arrays."""
    rows = max(t[0].shape[0] for t in tabs)
    nodes = max(len(t[1]) for t in tabs)
    B = np.zeros((3, rows, N_PROFILES))
    O = np.zeros((3, nodes), dtype=np.int64)
    for r, (b, o) in enumerate(tabs):
        B[r, :b.shape[0]] = b
        O[r, :len(o)] = o
        O[r, len(o):] = o[-1]
    return B, O


_TABLE_CACHE = {}
_TABLE_CACHE_MAX = 2
# Memory: the lattice tables grow as N^2 z_substeps (about 30 MB for the three relics at the
# analysis grid, 515 bins with the extension and one step per bin; 1.5 GB at 1830 bins and four
# substeps). Only sets up to this size are kept between calls (scans over the coupling at a fixed
# mediator mass reuse them); larger ones are rebuilt each call and freed.
_TABLE_CACHE_MAX_BYTES = 128 * 2 ** 20


def _tables_nbytes(tabs):
    lat, top = tabs
    n = lat[0].nbytes + lat[1].nbytes
    if top is not None:
        n += top[0].nbytes + top[1].nbytes
    return n


def _profile_tables(M, masses, E0, N, dlnE, q, K, top_full, L_top_mid, n_u):
    """Lattice and top-step profile tables for the three relics, cached (up to
    _TABLE_CACHE_MAX_BYTES each) by everything they depend on: the mediator and relic masses and
    the grids -- not the couplings."""
    key = (float(M), tuple(float(m) for m in masses), float(E0), int(N), float(dlnE), int(q), int(K),
           bool(top_full), float(L_top_mid), int(n_u))
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    Delta = dlnE / q
    m_max = K - 1 if top_full else K - 2
    n_lat = (N - 1) * q + m_max + 1
    lat, top = [], []
    for r in range(3):
        if masses[r] <= 0.0:
            dummy = (np.zeros((1, N_PROFILES)), np.array([0, 1], dtype=np.int64))
            lat.append(dummy)
            top.append(dummy)
            continue
        mu0 = M * M / (2.0 * E0 * masses[r])
        lat.append(_tri_table(mu0, n_lat, q, N, dlnE, Delta, n_u))
        if not top_full:
            mu_top = mu0 * np.exp(-np.arange(N) * dlnE - L_top_mid)
            top.append(_tri_table(mu0, N, 1, N, dlnE, Delta, n_u, mu_list=mu_top))
    out = (_stack_tables(lat), _stack_tables(top) if not top_full else None)
    del lat, top
    if _tables_nbytes(out) <= _TABLE_CACHE_MAX_BYTES:
        while len(_TABLE_CACHE) >= _TABLE_CACHE_MAX:
            _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
        _TABLE_CACHE[key] = out
    return out


def propagate_si(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_src,
                 p_index, rho_evol, Phi_0, M, g_mass, majorana=True, channels="full",
                 phiphi="coherent", z_substeps=1, E_extension=10.0, n_u=4096,
                 return_diagnostics=False, regeneration=True):
    """Cosmological transport of the astrophysical neutrinos under secret interactions.

    Arguments: E_bins -- the (geometric) energy grid of the result, eV; z_max, H0 (eV),
    Omega_m, Omega_L -- sources up to z_max in flat LCDM; U_pmns -- (3, 3) mixing matrix;
    masses -- (3,) neutrino masses, eV; T_nu0 -- relic temperature today, eV; eta_mass -- (3,)
    relic density per mass state relative to the standard one; rho_src -- (3,) source flavor
    composition; p_index, rho_evol, Phi_0 -- source spectrum E^-p, evolution R(z) (1+z)^rho_evol
    (dp_cosmo.calc_R_source), normalization; M -- mediator mass, eV; g_mass -- (3, 3) real
    mass-basis couplings (mass_basis_couplings); majorana -- neutrino nature (Dirac: not yet);
    channels -- 'full', 's' (s channel only) or 'delta' (its narrow-width limit), the
    approximations compared in 2107.13568; phiphi -- 'coherent' or 'paper' (module notes);
    z_substeps -- redshift steps per energy bin (Delta ln(1+z) = dlnE / z_substeps);
    E_extension -- the grid is extended above E_bins[-1] by this factor so that what is injected
    above it cascades into it; n_u -- nodes of the double-scalar spectra; regeneration -- False
    drops the secondaries (absorption alone, for validation).

    Returns the (N_E, 3) mass-state fluxes at z = 0 (both helicities), on E_bins; with
    return_diagnostics also a dict of the number and energy bookkeeping."""
    from dp_core import compute_dE_geometric
    if not majorana:
        raise NotImplementedError("secret interactions: Dirac neutrinos are not implemented yet")
    E_bins = np.asarray(E_bins, dtype=np.float64)
    N_E = len(E_bins)
    dlnE = float(np.log(E_bins[1] / E_bins[0]))
    n_ext = int(np.ceil(np.log(E_extension) / dlnE - 1e-9)) if E_extension > 1.0 else 0
    N = N_E + n_ext
    E = E_bins[0] * np.exp(dlnE * np.arange(N))
    E[:N_E] = E_bins
    dE = compute_dE_geometric(E)
    q = int(z_substeps)
    if q < 1:
        raise ValueError("secret interactions: z_substeps must be >= 1")
    Delta = dlnE / q
    Lmax = float(np.log1p(z_max))
    K = max(1, int(np.ceil(Lmax / Delta - 1e-9)))
    L_edges = np.arange(K + 1) * Delta
    L_edges[-1] = Lmax
    top_full = abs((L_edges[K] - L_edges[K - 1]) - Delta) <= 1e-12 * Delta
    L_top_mid = 0.5 * (L_edges[K - 1] + L_edges[K])

    masses = np.ascontiguousarray(masses, dtype=np.float64)
    g_mass = np.ascontiguousarray(g_mass, dtype=np.float64)
    M2 = M * M
    Gam = scalar_width(M, g_mass, majorana)
    MG = M * Gam
    Cabs, Creg, G2 = coupling_tensors(g_mass, phiphi, channels)
    if not regeneration:
        Creg[:] = 0.0
    delta_mode = str(channels).strip().lower() == "delta"
    n0 = relic_density_per_helicity(T_nu0) * np.asarray(eta_mass, dtype=np.float64)
    P_src = (np.abs(np.asarray(U_pmns)) ** 2).T @ np.asarray(rho_src, dtype=np.float64)

    (lat_B, lat_O), top = _profile_tables(M, masses, E[0], N, dlnE, q, K, top_full, L_top_mid, n_u)

    F = np.zeros((N, 3))
    reg = np.zeros((N, 3))
    diag = np.zeros(6)
    Js, Jint, Jinv, Jpp, N2, N4, N5 = (np.zeros((N, 3)) for _ in range(7))
    node = np.zeros((N, 3), dtype=np.int64)
    jj = np.arange(N, dtype=np.int64)
    Ep = E ** (-p_index)
    for m in range(K - 1, -1, -1):
        L_lo, L_hi = L_edges[m], L_edges[m + 1]
        dt, T_inj = _step_time_and_injection(L_lo, L_hi, H0, Omega_m, Omega_L, p_index, rho_evol, Phi_0)
        S0 = np.ascontiguousarray(np.outer(Ep * (T_inj / dt), P_src))
        _step_integrals(E, np.exp(L_lo), np.exp(L_hi), np.exp(0.5 * (L_lo + L_hi)), masses, n0, M2, MG,
                        H0, Omega_m, Omega_L, delta_mode, Js, Jint, Jinv, Jpp, N2, N4, N5)
        if m == K - 1 and not top_full:
            tB, tO = top
            node[:] = jj[:, None]
        else:
            tB, tO = lat_B, lat_O
            node[:] = (jj * q + m)[:, None]
        for r in range(3):
            if masses[r] <= 0.0:
                node[:, r] = 0
        _si_sweep(F, E, dE, S0, Js, Jint, Jinv, Jpp, N2, N4, N5, Cabs, Creg, tB, tO, node, dt, reg, diag)
    out = np.ascontiguousarray(F[:N_E])
    if not return_diagnostics:
        return out
    info = dict(absorbed=diag[0], secondaries_in_grid=diag[1], secondaries_expected=diag[2],
                energy_absorbed=diag[3], energy_secondaries_in_grid=diag[4], injected=diag[5],
                final_number=float(np.sum(F * dE[:, None])), final_number_in_E_bins=float(np.sum(out * dE[:N_E, None])),
                final_energy=float(np.sum(F * (E * dE)[:, None])), width=Gam, n_steps=K, N_grid=N, top_full=top_full,
                F_extended=F.copy(), E_extended=E.copy())
    return out, info


def free_streaming_flux(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, rho_src, p_index, rho_evol, Phi_0,
                        n=4000):
    """The mass-state fluxes without interactions, (N_E, 3): E^-p P_a int_0^z_max Phi_0 (1+z)^-p
    R(z) / (H (1+z)) dz, by Gauss-Legendre on n nodes split at the kinks of R(z) -- the reference
    for the no-coupling limit of propagate_si."""
    from dp_cosmo import calc_R_source
    Lmax = np.log1p(z_max)
    cuts = [0.0] + [L for L in (np.log(2.0), np.log(5.0)) if L < Lmax] + [Lmax]
    x, w = np.polynomial.legendre.leggauss(n)
    T = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        L = a + (b - a) * 0.5 * (x + 1.0)
        zp1 = np.exp(L)
        R = np.array([calc_R_source(z, rho_evol) for z in zp1 - 1.0])
        T += np.sum(0.5 * w * (b - a) * Phi_0 * zp1 ** (-p_index) * R / (H0 * np.sqrt(Omega_m * zp1 ** 3 + Omega_L)))
    P_src = (np.abs(np.asarray(U_pmns)) ** 2).T @ np.asarray(rho_src, dtype=np.float64)
    return np.outer(np.asarray(E_bins, dtype=np.float64) ** (-p_index) * T, P_src)
