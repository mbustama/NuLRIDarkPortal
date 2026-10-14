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
  - Mass-basis couplings lambda = U^T G U (complex symmetric) for flavor-basis couplings G (real
    symmetric): for Majorana fields the scalar bilinear is nubar_a nu_b = sum_ij nubar_i
    [(U^T G U)*_ij P_R + (U^T G U)_ij P_L] nu_j. 2107.13568 writes U^dag G U and takes it real,
    which agrees only without CP violation; with delta_CP != 0 even a flavor-universal coupling is
    off-diagonal in the mass basis (g U^T U). The cross sections are generalized to complex
    couplings (moduli and real parts of the coupling products; they reduce to the paper's for real
    ones). The Majorana phases drop out of every rate, and for a single coupled flavor so does
    delta_CP (|lambda_ij| = g |U_fi| |U_fj|).
  - CP: with complex couplings the s-t and s-u interference terms carry Re[C (s - M^2 - i M Gamma)];
    the part Im(C) M Gamma, odd under nu <-> nubar, is a CP asymmetry of relative size Gamma/M =
    sum |lambda|^2 / 16 pi. The transport, which follows both helicities together, uses the
    CP-even part (exact for the sum up to O((Gamma/M)^2) for equal nu and nubar sources).
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
from functools import lru_cache

import numpy as np

from dp_core import njit, prange
from dp_thermal import (relic_state_rates_analytic, _GL16_X as _TGL16_X, _GL16_W as _TGL16_W,
                        _GL10_X as _TGL10_X, _GL10_W as _TGL10_W, _FD16 as _TFD16)

_ZETA3 = 1.2020569031595942


@lru_cache(maxsize=8)
def _leggauss(n):
    """Gauss-Legendre nodes and weights, computed once per order: numpy builds them from an n x n eigenproblem,
    6 s at n = 4000 (and ~30 s with sixteen processes doing it at once), which free_streaming_flux paid on every
    call until 2026-10-14 (it made the *_gal plane points ~30 times slower)."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


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

_FLAVOR_INDEX = {"e": 0, "mu": 1, "tau": 2}


def flavor_couplings(texture, g, g_matrix=None):
    """The flavor-basis coupling matrix g_ab from a texture name: 'universal' (g delta_ab),
    'e', 'mu', 'tau' (g on that flavor alone), 'e+mu', 'e+tau', 'mu+tau' (g on each of two
    flavors, diagonal), or 'matrix' (g times the real symmetric 3x3 g_matrix, any texture)."""
    t = str(texture).strip().lower()
    out = np.zeros((3, 3))
    if t == "universal":
        out[:] = g * np.eye(3)
    elif t in _FLAVOR_INDEX:
        f = _FLAVOR_INDEX[t]
        out[f, f] = g
    elif t in ("e+mu", "e+tau", "mu+tau"):
        for f in t.split("+"):
            out[_FLAVOR_INDEX[f], _FLAVOR_INDEX[f]] = g
    elif t == "matrix":
        if g_matrix is None:
            raise ValueError("secret interactions: texture 'matrix' needs g_matrix")
        m = np.asarray(g_matrix, dtype=np.float64)
        if m.shape != (3, 3) or np.max(np.abs(m - m.T)) > 1e-12 * max(np.max(np.abs(m)), 1e-300):
            raise ValueError("secret interactions: g_matrix must be a real symmetric 3x3 matrix")
        out[:] = g * m
    else:
        raise ValueError(f"secret interactions: unknown texture {texture!r} "
                         "(universal, e, mu, tau, e+mu, e+tau, mu+tau, matrix)")
    return out


def mass_basis_couplings(g_flavor, U):
    """The couplings of the mass states of Majorana neutrinos, lambda = U^T G U (complex
    symmetric 3x3), for flavor-basis couplings G (-1/2 G_ab nubar_a nu_b phi; see the module notes).
    U may include Majorana phases, which drop out of every rate."""
    G = np.asarray(g_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    return np.ascontiguousarray(U.T @ G @ U)


def scalar_width(M, g, majorana=True):
    """Total width of the scalar, Eq. (A15): Gamma = c M / (8 pi) sum_kl |g_kl|^2, c = 1/2 for
    Majorana neutrinos (identical final states), 1 for Dirac."""
    c = 0.5 if majorana else 1.0
    return c * M / (8.0 * np.pi) * float(np.sum(np.abs(np.asarray(g)) ** 2))


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


def dsigma_dt_scalar_majorana(s, t, M, Gam, g, i, j, k, l, cp=0):
    """dsigma/dt for nu_i + nu_j -> nu_k + nu_l, Eq. (A12) generalized to complex couplings, with
    u = -s - t (massless final states, relic mass neglected against s). The 1/2 for identical
    final states (k = l) is included. Arguments: s, t (eV^2; t in [-s, 0]); M, Gam (eV); g -- 3x3
    mass-basis couplings (complex symmetric); i, j, k, l -- mass indices; cp -- 0 for the CP-even
    part (the average of neutrino and antineutrino; the transport's), -1 for an incident neutrino
    (left-handed), +1 for an antineutrino: the s-t and s-u interference carry
    Re[C (s - M^2 - i M Gamma)] = Re(C) (s - M^2) - cp Im(C) M Gamma, C the coupling product.
    Returns eV^-4.

    Helicity structure behind the generalization (incident h = -1/2, relic summed over its
    helicity): with the relic at h = -1/2 the s channel (lambda_ij lambda_kl), t channel
    (lambda_ik lambda_jl) and u channel (lambda_il lambda_jk) produce the same final state (both
    outgoing at h = +1/2) and interfere; the s channel into two h = -1/2 neutrinos (lambda_ij
    lambda*_kl) and, with the relic at h = +1/2, the t channel (lambda_ik lambda*_jl) and u channel
    (lambda_il lambda*_jk) add incoherently. For real couplings this is Eq. (A12)."""
    u = -s - t
    M2 = M * M
    bw = _bw(s, M, Gam)
    MG = M * Gam
    lij, lkl, lik, ljl, lil, ljk = g[i, j], g[k, l], g[i, k], g[j, l], g[i, l], g[j, k]
    Cst = lij * lkl * np.conjugate(lik * ljl)
    Csu = lij * lkl * np.conjugate(lil * ljk)
    Ctu = lik * ljl * np.conjugate(lil * ljk)
    br = (abs(lij) ** 2 * abs(lkl) ** 2 / bw
          + abs(lik) ** 2 * abs(ljl) ** 2 * (t * t / (s * s)) / (t - M2) ** 2
          + abs(lil) ** 2 * abs(ljk) ** 2 * (u * u / (s * s)) / (u - M2) ** 2
          + (Cst.real * (s - M2) - cp * Cst.imag * MG) * t / (s * bw * (t - M2))
          + Ctu.real * (t * u / (s * s)) / ((u - M2) * (t - M2))
          + (Csu.real * (s - M2) - cp * Csu.imag * MG) * u / (s * bw * (u - M2)))
    sym = 0.5 if k == l else 1.0
    return sym * br / (8.0 * np.pi)


def phiphi_coupling(g, i, j, mode="coherent"):
    """Coupling factor of nu_i nu_j -> phi phi: |(lambda lambda^dag)_ij|^2 = |sum_k lambda_ik
    lambda*_kj|^2 (coherent, the default: the exchanged nu_k connects lambda at one vertex to
    lambda* at the other, the relic at the opposite helicity) or sum_k |lambda_ik|^2 |lambda_jk|^2
    (as printed in 2107.13568, real couplings); 'off' drops double-scalar production (validation:
    nuSIprop's default)."""
    if mode == "off":
        return 0.0
    if mode == "coherent":
        return float(abs(np.dot(g[i, :], np.conjugate(g[:, j]))) ** 2)
    if mode == "paper":
        return float(np.sum(np.abs(g[i, :]) ** 2 * np.abs(g[j, :]) ** 2))
    raise ValueError(f"phiphi coupling mode must be 'coherent', 'paper' or 'off', got {mode!r}")


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
    g = np.asarray(g, dtype=np.complex128)
    H = g @ np.conjugate(g)                       # lambda lambda^dag (lambda symmetric)
    Q = np.conjugate(g) @ g @ np.conjugate(g)     # lambda* lambda lambda*
    G2 = float(np.sum(np.abs(g) ** 2))
    out = np.empty(5)
    out[0] = abs(g[i, j]) ** 2 * G2 / (16.0 * np.pi) * s / bw
    out[1] = 2.0 * H[i, i].real * H[j, j].real / (16.0 * np.pi) * N2 / s
    out[2] = 2.0 * (g[i, j] * Q[i, j]).real / (16.0 * np.pi) * (s - M2) / bw * N4
    out[3] = abs(H[i, j]) ** 2 / (16.0 * np.pi) * N5 / s
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
    Complex couplings (lambda = U^T G U): with H = lambda lambda^dag and Q = lambda* lambda lambda*,
        Cabs = |l_ar|^2 G2, 2 H_aa H_rr, 2 Re(l_ar Q_ar), |H_ar|^2 (all / 16 pi), |H_ar|^2 / 32 pi;
        Creg = |l_ar|^2 H_cc, |l_ac|^2 H_rr, H_aa |l_rc|^2, Re(l_ar l*_ac H_cr), Re(l_ac l*_rc H_ra),
               Re(l_ar l*_rc H_ca) (all / 8 pi), 4 Cabs[4] H_cc / G2
    (CP-even parts; for real couplings H = g^2, Q = g^3).
    Returns (Cabs (5, 3, 3) [p, a, r], Creg (7, 3, 3, 3) [q, a, r, c], G2 = sum |l_kl|^2)."""
    g = np.asarray(g, dtype=np.complex128)
    gc = np.conjugate(g)
    H = g @ gc
    Q = gc @ g @ gc
    G2 = float(np.sum(np.abs(g) ** 2))
    Cabs = np.zeros((N_CHANNELS, 3, 3))
    Creg = np.zeros((N_PROFILES, 3, 3, 3))
    pi8, pi16 = 8.0 * np.pi, 16.0 * np.pi
    for a in range(3):
        for r in range(3):
            Cabs[0, a, r] = abs(g[a, r]) ** 2 * G2 / pi16
            Cabs[1, a, r] = 2.0 * H[a, a].real * H[r, r].real / pi16
            Cabs[2, a, r] = 2.0 * (g[a, r] * Q[a, r]).real / pi16
            Cabs[3, a, r] = abs(H[a, r]) ** 2 / pi16
            Cabs[4, a, r] = phiphi_coupling(g, a, r, mode) / (32.0 * np.pi)
            for c in range(3):
                Creg[0, a, r, c] = abs(g[a, r]) ** 2 * H[c, c].real / pi8
                Creg[1, a, r, c] = abs(g[a, c]) ** 2 * H[r, r].real / pi8
                Creg[2, a, r, c] = H[a, a].real * abs(g[r, c]) ** 2 / pi8
                Creg[3, a, r, c] = (g[a, r] * gc[a, c] * H[c, r]).real / pi8
                Creg[4, a, r, c] = (g[a, c] * gc[r, c] * H[r, a]).real / pi8
                Creg[5, a, r, c] = (g[a, r] * gc[r, c] * H[c, a]).real / pi8
                Creg[6, a, r, c] = (4.0 * Cabs[4, a, r] * H[c, c].real / G2) if G2 > 0 else 0.0
    ch = str(channels).strip().lower()
    if ch in ("s", "delta"):
        Cabs[1:] = 0.0
        Creg[1:] = 0.0
    elif ch != "full":
        raise ValueError(f"secret interactions: channels must be 'full', 's' or 'delta', got {channels!r}")
    return np.ascontiguousarray(Cabs), np.ascontiguousarray(Creg), G2


# =============================================================================
# DIRAC NEUTRINOS AND THE VECTOR MEDIATOR (2026-10-13, after the brute-force checks of
# tests/amplitudes_bruteforce.py; choices decided with Mauricio)
# =============================================================================
#
# Dirac, scalar: L = -phi h_ij nubar_i nu_j with h = U^dag G U (Hermitian; the right-handed neutrinos
# rotating like the left-handed ones, 2107.13568's convention -- nothing observable fixes their
# basis). Every vertex joins a state that interacts weakly (nu_L, nubar_R: visible) to a sterile one
# (nu_R, nubar_L). The relics, at rest and unpolarized, present both helicities to the incident
# neutrino, and the scalar couples to both. Per incident visible nu_a on relic mass state r (relic nu
# and nubar, half the density per helicity each) the absorption is the appendix's Eq. (A8) and the
# secondaries split as in dirac_scalar_tensors (derived and checked against the brute force; the
# appendix's Table I overstates the visible part by up to 3.5%). By parity an incident sterile
# neutrino gives the same with visibility swapped (checked), so the visible flux is (S + D)/2 with S
# evolving under the kernel visible + invisible and D under visible - invisible ('track', decided);
# 'drop' discards the sterile secondaries, as the paper does.
#
# Vector, left-handed currents (decided): L = -V_mu G_ab nubar_a gamma^mu P_L nu_b, mass basis
# U^dag G U for Majorana and Dirac alike. Only the relic components that behave as nu_L (helicity -)
# and as nubar_R (+) interact: a Majorana relic offers both, 56 + 56 per mass state, a Dirac one half
# of them (28 nu_L + 28 nubar_R), so Dirac rates are half the Majorana ones (kappa). Every secondary
# is visible. Squared amplitudes (checked): nu nubar -> nu nubar 4 u^2 |A_s/(s - M^2 + i M Gamma) +
# A_t/(t - M^2)|^2 with A_s = G_ra G_kl, A_t = G_ka G_rl; nu nu -> nu nu 4 s^2 |A_t/(t - M^2) +
# A_u/(u - M^2)|^2 with A_t = G_ka G_lr, A_u = G_la G_kr; nu nubar -> V V 4 |(G G^dag)_ar|^2
# [t/u + u/t + 4 M^2 s/(t u) - M^4 (1/t^2 + 1/u^2)], each V decaying at once into nu nubar with the
# spectrum of its polarization in the relic rest frame, where it is ultrarelativistic: longitudinal
# fraction f_L = 8 s M^2 (t - M^2)^2 / (t^2 (u - M^2)^2) / |M|^2 (the light-cone polarization
# k/M - M p2/(k.p2); checked), decays 6z(1-z) per neutrino, transverse ones 3z^2 + 3(1-z)^2 for the
# pair. Width M sum |G|^2 / 24 pi. The s channel reproduces the Z-burst <sigma> = 2 sqrt(2) pi G_F
# for the Standard-Model coupling.
#
# Above the resonance the vector's t channel is forward-peaked, sigma ~ |G|^4 / (4 pi M^2): nearly
# every scattering leaves the incident neutrino in its own energy bin and moves ~M^2 / 2 m into the
# recoiling relic. The transport removes those no-op returns from a bin's own equations exactly
# (QE profiles: the absorption counts only incident daughters that leave the bin or change mass
# state, from complementary integrals computed without cancellation) and keeps every recoil.

def mass_basis_couplings_model(g_flavor, U, mediator="scalar", majorana=True):
    """Mass-basis couplings for flavor-basis G: U^T G U for a scalar and Majorana neutrinos
    (mass_basis_couplings), U^dag G U for a scalar and Dirac neutrinos and for the vector."""
    if mediator == "scalar" and majorana:
        return mass_basis_couplings(g_flavor, U)
    G = np.asarray(g_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    return np.ascontiguousarray(np.conjugate(U).T @ G @ U)


def mediator_width(M, g, mediator="scalar", majorana=True):
    """Total width into neutrinos: scalar M sum |g|^2 / 8 pi (1/2 of it for Majorana, Eq. A15),
    vector coupled to left-handed currents M sum |G|^2 / 24 pi."""
    S = float(np.sum(np.abs(np.asarray(g)) ** 2))
    if mediator == "vector":
        return M * S / (24.0 * np.pi)
    return (0.5 if majorana else 1.0) * M * S / (8.0 * np.pi)


def dirac_scalar_tensors(h, channels="full"):
    """Coupling tensors of the scalar with Dirac neutrinos, in the profile set of coupling_tensors
    (flat, P2, P3, P4, P5, P6, double scalar): (Cvis, Cinv, G2), the visible and the sterile
    secondaries of mass state c per incident visible nu_a on relic r (CP-even; the relic weights,
    1/2 for each of nu and nubar, included). With w = (1/2, ..., 1/4) the absorption is
    sum_q w_q sum_c (Cvis + Cinv) J_q N_q, which is Eq. (A8) plus double-scalar production (A10)."""
    h = np.asarray(h, dtype=np.complex128)
    hc = np.conjugate(h)
    H = h @ hc.T
    G2 = float(np.sum(np.abs(h) ** 2))
    Cv = np.zeros((N_PROFILES, 3, 3, 3))
    Ci = np.zeros((N_PROFILES, 3, 3, 3))
    p16 = 16.0 * np.pi
    for a in range(3):
        for r in range(3):
            Cpp = abs(H[a, r]) ** 2 / (64.0 * np.pi)
            for c in range(3):
                Cv[0, a, r, c] = abs(h[a, r]) ** 2 * H[c, c].real / p16
                Ci[0, a, r, c] = Cv[0, a, r, c]
                Ci[1, a, r, c] = abs(h[c, a]) ** 2 * H[r, r].real / (8.0 * np.pi)
                Cv[2, a, r, c] = H[a, a].real * abs(h[r, c]) ** 2 / p16
                Ci[2, a, r, c] = Cv[2, a, r, c]
                Ci[3, a, r, c] = (h[r, a] * hc[c, a] * H[c, r]).real / p16
                Ci[4, a, r, c] = (h[c, a] * hc[c, r] * H[a, r]).real / p16
                Cv[5, a, r, c] = (h[r, a] * hc[r, c] * H[a, c]).real / p16
                if G2 > 0:
                    Cv[6, a, r, c] = 2.0 * Cpp * H[c, c].real / G2
                    Ci[6, a, r, c] = Cv[6, a, r, c]
    ch = str(channels).strip().lower()
    if ch in ("s", "delta"):
        Cv[1:] = 0.0
        Ci[1:] = 0.0
    elif ch != "full":
        raise ValueError(f"secret interactions: channels must be 'full', 's' or 'delta', got {channels!r}")
    return np.ascontiguousarray(Cv), np.ascontiguousarray(Ci), G2


# The vector's ten regeneration profiles, in the energy fraction of the secondary itself
# (mu = M^2/s; x: the neutrino of the incident line or of the nu nubar s channel, y: the
# antineutrino or the recoiling relic):
#   0 x^2 (s channel, nu)                       1 (1-y)^2 (s channel, nubar)
#   2 x^2/((1-x)+mu)^2 (t channel, nu nubar: the incident's daughter -- forward-peaked)
#   3 (1-y)^2/(y+mu)^2 (its recoiling antineutrino)
#   4 x^2/((1-x)+mu) (s-t interference, nu)     5 (1-y)^2/(y+mu) (s-t interference, nubar)
#   6 1/((1-x)+mu)^2 (t channel, nu nu: the incident's daughter -- forward-peaked)
#   7 1/(x+mu)^2 (its recoil)                    8 1/((1-x+mu)(x+mu)) (t-u interference)
#   9 the neutrinos of vector-pair production (normalized to 1)
N_VPROFILES = 10
VECTOR_JT = np.array([0, 0, 2, 2, 1, 1, 2, 2, 2, 3], dtype=np.int64)
VECTOR_W = np.array([0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.5, 0.25])
VECTOR_QE = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.int64)
SCALAR_JT = np.array([0, 2, 2, 1, 2, 1, 3], dtype=np.int64)
SCALAR_W = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
SCALAR_QE = np.zeros(7, dtype=np.int64)


def vector_tensors(G, majorana=True, channels="full"):
    """Coupling tensors of the vector coupled to left-handed currents, profile set above:
    (C (10, 3, 3, 3) [q, a, r, c], G2). Kinematic factors: profiles 0, 1 times s/BW; 2, 3, 6, 7, 8
    times 1/s; 4, 5 times (s - M^2)/BW; 9 times vv_kinematic(s). kappa = 1 (Majorana), 1/2
    (Dirac) multiplies everything. The absorption is sum_q VECTOR_W[q] sum_c C J N (with the
    forward-peaked daughters' no-op returns removed in the transport)."""
    G = np.asarray(G, dtype=np.complex128)
    Gc = np.conjugate(G)
    H = G @ Gc.T
    G2 = float(np.sum(np.abs(G) ** 2))
    kap = 1.0 if majorana else 0.5
    C = np.zeros((N_VPROFILES, 3, 3, 3))
    p4 = 4.0 * np.pi
    for a in range(3):
        for r in range(3):
            Cvv = kap * abs(H[a, r]) ** 2 / (8.0 * np.pi)
            for c in range(3):
                C[0, a, r, c] = kap * abs(G[r, a]) ** 2 * H[c, c].real / p4
                C[1, a, r, c] = C[0, a, r, c]
                C[2, a, r, c] = kap * abs(G[c, a]) ** 2 * H[r, r].real / p4
                C[3, a, r, c] = kap * H[a, a].real * abs(G[r, c]) ** 2 / p4
                C[4, a, r, c] = -2.0 * kap * (G[r, a] * Gc[c, a] * H[c, r]).real / p4
                C[5, a, r, c] = -2.0 * kap * (G[r, a] * Gc[r, c] * H[a, c]).real / p4
                C[6, a, r, c] = kap * abs(G[c, a]) ** 2 * H[r, r].real / p4
                C[7, a, r, c] = kap * H[a, a].real * abs(G[c, r]) ** 2 / p4
                C[8, a, r, c] = 2.0 * kap * (G[c, a] * Gc[c, r] * H[a, r]).real / p4
                if G2 > 0:
                    C[9, a, r, c] = 4.0 * Cvv * H[c, c].real / G2
    ch = str(channels).strip().lower()
    if ch in ("s", "delta"):
        C[2:] = 0.0
    elif ch != "full":
        raise ValueError(f"secret interactions: channels must be 'full', 's' or 'delta', got {channels!r}")
    return np.ascontiguousarray(C), G2


@njit(cache=True)
def vv_kinematic(s, M2):
    """I_VV(s): the vector-pair cross section is |(G G^dag)_ar|^2 I_VV / (8 pi) (times kappa),
    I_VV = [2 (s^2 + 4 M^4) Lambda / (s - 2 M^2) - 4 r] / s^2, r = sqrt(s (s - 4 M^2)),
    Lambda = ln((A + r)/(A - r)) = 2 ln((A + r)/(2 M^2)), A = s - 2 M^2; 0 below s = 4 M^2."""
    if s <= 4.0 * M2:
        return 0.0
    A = s - 2.0 * M2
    r = np.sqrt(s * (s - 4.0 * M2))
    Lam = 2.0 * np.log((A + r) / (2.0 * M2))
    return (2.0 * (s * s + 4.0 * M2 * M2) * Lam / A - 4.0 * r) / (s * s)


def vector_dsigma_dt(s, t, M, Gam, G, a, r, k, l, relic="nubar", cp=0):
    """dsigma/dt for an incident nu_a (left-handed) on a relic of mass state r acting as 'nubar'
    (nu nubar -> nu_k nubar_l, s and t channels) or as 'nu' (nu nu -> nu_k nu_l, t and u channels,
    1/2 for k = l), t = (p_a - p_k)^2, fixed relic helicity (no kappa), eV^-4; for checks. cp: 0 for
    the CP-even part, -1 for an incident neutrino, +1 for an antineutrino (as
    dsigma_dt_scalar_majorana)."""
    M2 = M * M
    u = -s - t
    if relic == "nubar":
        As = G[r, a] * G[k, l]
        At = G[k, a] * G[r, l]
        bw = (s - M2) ** 2 + M2 * Gam ** 2
        X = As * np.conjugate(At)
        amp2 = (abs(As) ** 2 / bw + abs(At) ** 2 / (t - M2) ** 2
                + 2.0 * (X.real * (s - M2) - cp * X.imag * M * Gam) / (bw * (t - M2)))
        return 4.0 * u * u * amp2 / (16.0 * np.pi * s * s)
    At = G[k, a] * G[l, r]
    Au = G[l, a] * G[k, r]
    sym = 0.5 if k == l else 1.0
    return sym * 4.0 * s * s * abs(At / (t - M2) + Au / (u - M2)) ** 2 / (16.0 * np.pi * s * s)


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


# --- the vector's profiles ---------------------------------------------------------------------

@njit(cache=True)
def _vec_profiles_at(x, mu, out):
    """Vector profiles 0..8 at x (out[0..8]); see N_VPROFILES."""
    v = 1.0 - x
    out[0] = x * x
    out[1] = v * v
    out[2] = x * x / ((v + mu) * (v + mu))
    out[3] = v * v / ((x + mu) * (x + mu))
    out[4] = x * x / (v + mu)
    out[5] = v * v / (x + mu)
    out[6] = 1.0 / ((v + mu) * (v + mu))
    out[7] = 1.0 / ((x + mu) * (x + mu))
    out[8] = 1.0 / ((v + mu) * (x + mu))


@njit(cache=True)
def _vec_bin_integrals(x_lo, x_hi, mu, glx, glw, out, tmp):
    """Integrals of the vector profiles 0..8 over [x_lo, x_hi]: exact for 0, 1, 6, 7, 8; for the
    forward-peaked 2 and 4 the closed forms on the top bins when mu < bin width (stable there),
    Gauss-Legendre otherwise and for 3 and 5 (peaked at x -> 0, where geometric bins resolve them)."""
    D = x_hi - x_lo
    wlo = 1.0 - x_lo + mu
    whi = 1.0 - x_hi + mu
    a = 1.0 - x_lo
    b = 1.0 - x_hi
    out[0] = D * (x_hi * x_hi + x_hi * x_lo + x_lo * x_lo) / 3.0
    out[1] = D * (a * a + a * b + b * b) / 3.0
    out[6] = D / (wlo * whi)
    out[7] = D / ((x_lo + mu) * (x_hi + mu))
    out[8] = (np.log1p(D / (x_lo + mu)) + np.log1p(D / whi)) / (1.0 + 2.0 * mu)
    top = mu < D and (1.0 - x_hi) < 2.0 * D
    for p in (2, 3, 4, 5):
        out[p] = 0.0
    for k in range(len(glx)):
        _vec_profiles_at(x_lo + D * glx[k], mu, tmp)
        out[3] += glw[k] * tmp[3]
        out[5] += glw[k] * tmp[5]
        if not top:
            out[2] += glw[k] * tmp[2]
            out[4] += glw[k] * tmp[4]
    out[3] *= D
    out[5] *= D
    if top:
        ly = np.log1p(D / whi)
        out[2] = (1.0 + mu) ** 2 * D / (wlo * whi) - 2.0 * (1.0 + mu) * ly + D
        out[4] = (1.0 + mu) ** 2 * ly - 2.0 * (1.0 + mu) * D + 0.5 * D * (wlo + whi)
    else:
        out[2] *= D
        out[4] *= D


@njit(cache=True)
def vector_norms(mu, glx, glw, out):
    """Integrals over [0, 1] of the ten vector profiles at mu (the last, the pair spectrum, 1):
    closed forms below mu = 0.5, Gauss-Legendre of the smooth integrands above (where the closed
    forms of profiles 2-5 cancel, ~1/(3 mu^2))."""
    L = np.log1p(1.0 / mu)
    out[0] = 1.0 / 3.0
    out[1] = 1.0 / 3.0
    if mu < 0.5:
        NB = (1.0 + mu) / mu - 2.0 * (1.0 + mu) * L + 1.0
        NC = (1.0 + mu) ** 2 * L - 1.5 - mu
    else:
        NB = 0.0
        NC = 0.0
        for k in range(len(glx)):
            x = glx[k]
            w = 1.0 - x + mu
            NB += glw[k] * x * x / (w * w)
            NC += glw[k] * x * x / w
    out[2] = NB
    out[3] = NB
    out[4] = NC
    out[5] = NC
    out[6] = 1.0 / (mu * (1.0 + mu))
    out[7] = out[6]
    out[8] = 2.0 * L / (1.0 + 2.0 * mu)
    out[9] = 1.0


@njit(cache=True)
def vector_qe_complements(mu, h, glx, glw, out):
    """For the forward-peaked daughters (profiles 2 and 6), the integral over [0, 1/h] -- everything
    but the parent's own bin [1/h, 1] -- computed without the cancellation of N - B(0) when nearly all
    of the profile sits in the top bin (mu << bin width). out[2], out[6]; the rest 0."""
    for q in range(N_VPROFILES):
        out[q] = 0.0
    xh = 1.0 / h
    whi = 1.0 - xh + mu
    wlo = 1.0 + mu
    out[6] = xh / (whi * wlo)
    if mu < 0.1:
        out[2] = (1.0 + mu) ** 2 * xh / (whi * wlo) - 2.0 * (1.0 + mu) * np.log(wlo / whi) + xh
    else:
        if mu < 0.5:
            NB = (1.0 + mu) / mu - 2.0 * (1.0 + mu) * np.log1p(1.0 / mu) + 1.0
        else:
            NB = 0.0
            for k in range(len(glx)):
                x = glx[k]
                NB += glw[k] * x * x / ((1.0 - x + mu) * (1.0 - x + mu))
        top = 0.0
        for k in range(len(glx)):
            x = xh + (1.0 - xh) * glx[k]
            top += glw[k] * x * x / ((1.0 - x + mu) * (1.0 - x + mu))
        out[2] = NB - top * (1.0 - xh)


@njit(cache=True)
def vector_qe_drift(mu, h, glx, glw, out):
    """For the forward-peaked daughters (profiles 2 and 6), the integral over the parent's own bin
    [1/h, 1] of -ln x times the profile: the shift in ln E of the daughters that stay in the bin, which
    the transport carries as a drift to the bin below (the events themselves being no-ops or
    conversions within the bin). With y = 1 - x and Y = 1 - 1/h, profile 6 is 1/(y + mu)^2 and 2 is
    (1 - y)^2/(y + mu)^2, and -ln(1 - y) = y f(y) with f(0) = 1: each integral is that of
    y/(y + mu)^2, log1p(Y/mu) - Y/(Y + mu) (its series when Y << mu), plus the bounded remainder
    (f_q - 1) y/(y + mu)^2 by Gauss-Legendre. out[2], out[6]; the rest 0."""
    for q in range(N_VPROFILES):
        out[q] = 0.0
    Y = 1.0 - 1.0 / h
    t = Y / mu
    if t < 1e-3:
        main = t * t * (0.5 - t * (2.0 / 3.0 - t * (0.75 - t * 0.8)))
    else:
        main = np.log1p(t) - t / (1.0 + t)
    # the remainder is O(Y) against the main part's log(Y/mu): relative accuracy 1e-7 or better where it
    # varies on the scale mu << Y between the nodes (40-digit check), plenty for a drift that is itself
    # first order in the bin width
    r2 = 0.0
    r6 = 0.0
    for k in range(len(glx)):
        y = Y * glx[k]
        if y < 1e-4:
            e6 = y * (0.5 + y * (1.0 / 3.0 + y * 0.25))
        else:
            e6 = -np.log1p(-y) / y - 1.0
        e2 = e6 - (2.0 * y - y * y) * (1.0 + e6)
        kern = y / ((y + mu) * (y + mu))
        r6 += glw[k] * e6 * kern
        r2 += glw[k] * e2 * kern
    out[6] = main + Y * r6
    out[2] = main + Y * r2


@njit(cache=True)
def _vv_g(u, beta, delta, mu):
    """The vector pair's energy distribution in u = ln((x - x_-)/(x_+ - x)), with its longitudinal
    part: returns (g, gL, x), g = D a b / (beta T^2 U^2) and gL = f_L g = 2 mu (1-x)^2 a b /
    (beta T^2 x^2), where a = x - x_-, b = x_+ - x, T = b + delta, U = a + delta,
    D = (T^2 + U^2) a b + 4 mu T U (in units of s: t = -T, u = -U)."""
    if u >= 0.0:
        e = np.exp(-u)
        a = beta / (1.0 + e)
        b = beta * e / (1.0 + e)
    else:
        e = np.exp(u)
        a = beta * e / (1.0 + e)
        b = beta / (1.0 + e)
    x = 0.5 * (1.0 - beta) + a
    T = b + delta
    U = a + delta
    D = (T * T + U * U) * a * b + 4.0 * mu * T * U
    g = D * a * b / (beta * T * T * U * U)
    gL = 2.0 * mu * (1.0 - x) * (1.0 - x) * a * b / (beta * T * T * x * x)
    return g, gL, x


@njit(cache=True)
def _cum_left(f, h):
    """Cumulative integral from the left on a uniform grid: trapezoid with the Euler-Maclaurin end
    correction (O(h^4))."""
    n = len(f)
    C = np.empty(n)
    C[0] = 0.0
    for k in range(1, n):
        C[k] = C[k - 1] + 0.5 * h * (f[k - 1] + f[k])
    d0 = (f[1] - f[0]) / h
    for k in range(n):
        km = max(k - 1, 0)
        kp = min(k + 1, n - 1)
        C[k] -= h * h / 12.0 * ((f[kp] - f[km]) / ((kp - km) * h) - d0)
    return C


@njit(cache=True)
def _cum_right(f, h):
    """Cumulative integral from the right (C[k] = int_{u_k}^{u_end} f), as _cum_left."""
    n = len(f)
    C = np.empty(n)
    C[n - 1] = 0.0
    for k in range(n - 2, -1, -1):
        C[k] = C[k + 1] + 0.5 * h * (f[k] + f[k + 1])
    dn = (f[n - 1] - f[n - 2]) / h
    for k in range(n):
        km = max(k - 1, 0)
        kp = min(k + 1, n - 1)
        C[k] -= h * h / 12.0 * (dn - (f[kp] - f[km]) / ((kp - km) * h))
    return C


@njit(cache=True)
def _herm(C, f, k, tt, h, sign):
    """Cubic Hermite interpolation of a cumulative integral C (derivative sign * f) at node k + tt."""
    h00 = 2 * tt ** 3 - 3 * tt ** 2 + 1
    h10 = tt ** 3 - 2 * tt ** 2 + tt
    h01 = -2 * tt ** 3 + 3 * tt ** 2
    h11 = tt ** 3 - tt ** 2
    return h00 * C[k] + sign * h10 * h * f[k] + h01 * C[k + 1] + sign * h11 * h * f[k + 1]


@njit(cache=True)
def _vv_nu_bins(mu, edges, n_u, out):
    """Bin integrals of the neutrino spectrum of vector-pair production per neutrino (normalized to
    1): each V at energy fraction x decays into nu nubar, transverse ones (1 - f_L) with the pair's
    spectrum (3 z^2 + 3 (1-z)^2)/2 per neutrino, longitudinal ones 6 z (1-z), z = E_nu / E_V. With
    G_T(z) = z^3 - 3/2 z^2 + 3/2 z and G_L(z) = 3 z^2 - 2 z^3 the cumulative spectra,
    H(y) = F0(y) + 3/2 y F1T - 3/2 y^2 F2T + y^3 F3T + 3 y^2 F2L - 2 y^3 F3L,
    F0 = int_{x<y} f, FnT = int_{x>y} f (1 - f_L) / x^n, FnL = int_{x>y} f f_L / x^n."""
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
    g = np.empty(n_u)
    t1 = np.empty(n_u)
    t2 = np.empty(n_u)
    t3 = np.empty(n_u)
    l2 = np.empty(n_u)
    l3 = np.empty(n_u)
    for k in range(n_u):
        gk, gL, x = _vv_g(-U + k * h, beta, delta, mu)
        gT = gk - gL
        g[k] = gk
        t1[k] = gT / x
        t2[k] = gT / (x * x)
        t3[k] = gT / (x * x * x)
        l2[k] = gL / (x * x)
        l3[k] = gL / (x * x * x)
    C0 = _cum_left(g, h)
    C1T = _cum_right(t1, h)
    C2T = _cum_right(t2, h)
    C3T = _cum_right(t3, h)
    C2L = _cum_right(l2, h)
    C3L = _cum_right(l3, h)
    Z = C0[n_u - 1]
    Hprev = 0.0
    for d in range(nb, -1, -1):
        y = edges[d]
        if y <= x_m:
            H = 1.5 * y * C1T[0] - 1.5 * y * y * C2T[0] + y ** 3 * C3T[0] + 3.0 * y * y * C2L[0] - 2.0 * y ** 3 * C3L[0]
        elif y >= x_p:
            H = Z
        else:
            uu = np.log((y - x_m) / (x_p - y))
            f = (uu + U) / h
            k = int(f)
            if k < 0:
                k = 0
            elif k > n_u - 2:
                k = n_u - 2
            tt = f - k
            H = (_herm(C0, g, k, tt, h, 1.0) + 1.5 * y * _herm(C1T, t1, k, tt, h, -1.0)
                 - 1.5 * y * y * _herm(C2T, t2, k, tt, h, -1.0) + y ** 3 * _herm(C3T, t3, k, tt, h, -1.0)
                 + 3.0 * y * y * _herm(C2L, l2, k, tt, h, -1.0) - 2.0 * y ** 3 * _herm(C3L, l3, k, tt, h, -1.0))
        if d < nb:
            out[d] = (H - Hprev) / Z
        Hprev = H


@njit(cache=True, parallel=True)
def build_vector_table_tri(mu_list, d_max, dlnE, n_u):
    """The vector's profile table in triangular storage, as build_profile_table_tri:
    (B (total, 10), offsets)."""
    n_mu = len(mu_list)
    offsets = np.zeros(n_mu + 1, dtype=np.int64)
    for n in range(n_mu):
        offsets[n + 1] = offsets[n] + d_max[n] + 1
    B = np.zeros((offsets[n_mu], N_VPROFILES))
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
        tmp = np.empty(9)
        out9 = np.empty(9)
        for d in range(nd):
            x_hi = edges[d]
            x_lo = edges[d + 1]
            if d < 8:
                _vec_bin_integrals(x_lo, x_hi, mu, glx16, glw16, out9, tmp)
            else:
                _vec_bin_integrals(x_lo, x_hi, mu, glx6, glw6, out9, tmp)
            for p in range(9):
                B[o + d, p] = out9[p]
        if mu < 0.25:
            pb = np.empty(nd)
            _vv_nu_bins(mu, edges[:nd + 1], n_u, pb)
            for d in range(nd):
                B[o + d, 9] = pb[d]
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
# per relic state: the regeneration profiles are tabulated once per run (build_profile_table_tri,
# build_vector_table_tri) and depend on the mediator mass and the relic masses only, not on the
# couplings. The top step, shortened so that it starts at z_max exactly, has a table of its own.
#
# A model is a profile set (scalar: 7, vector: 10), for each profile q the kinematic factor it
# carries (jt: 0 s/BW, 1 (s - M^2)/BW, 2 1/s, 3 the pair cross section), its weight in the
# absorption (w: 1/2 for each secondary of a 2 -> 2 event, 1/4 for pair production, 1 and 0 for a
# forward-peaked daughter and its recoil), whether it is such a daughter (qe), and coupling tensors
# Ctau (absorption) and Creg (secondaries) [q, a, r, c].
#
# Per step and bin, the optical depth is integrated exactly over the step: the Breit-Wigner terms
# through theta = atan((s - M^2)/(M Gamma)), in which any resonance -- however narrow -- is smooth,
# the others by Gauss-Legendre in s. The bins are then swept from high to low energy, each solving
# its equations over the step exactly (constant rates and sources; dp_cosmo._absorption_factors)
# and passing down what it absorbed: its own secondaries (a 3 x 3 system over the mass states) and
# those landing in lower bins, with the exact bin integrals of their spectra. Every 2 -> 2 event
# yields two secondaries and every pair-production one four, less what falls below the grid. For a
# forward-peaked daughter (qe) the absorption counts only the events whose daughter leaves the bin or
# changes mass state -- the others change nothing -- while their recoils are all kept.

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


_PADE6 = np.array([1.0, 0.5, 5.0 / 44.0, 1.0 / 66.0, 1.0 / 792.0, 1.0 / 23760.0, 1.0 / 1235520.0])


@njit(cache=True)
def _expm_small(A):
    """exp(A) of a small dense matrix: scaling and squaring with the diagonal Pade approximant of
    degree 6 (the scaled norm below 1/2, where its error is below 1e-17)."""
    n = A.shape[0]
    nrm = 0.0
    for i in range(n):
        rs = 0.0
        for j in range(n):
            rs += abs(A[i, j])
        nrm = max(nrm, rs)
    sq = 0
    if nrm > 0.5:
        sq = int(np.ceil(np.log2(nrm / 0.5)))
    X = A / (2.0 ** sq)
    I = np.eye(n)
    X2 = X @ X
    X4 = X2 @ X2
    X6 = X4 @ X2
    Uo = X @ (_PADE6[1] * I + _PADE6[3] * X2 + _PADE6[5] * X4)
    Ve = _PADE6[0] * I + _PADE6[2] * X2 + _PADE6[4] * X4 + _PADE6[6] * X6
    E = np.linalg.solve(Ve - Uo, Ve + Uo)
    for _ in range(sq):
        E = E @ E
    return E


@njit(cache=True)
def _w_of_s(s, Em2, n0r, H0, Om, OL):
    """n_r(z) dt/ds at s = Em2 (1 + z), Em2 = 2 E m_r: n0r (1+z)^2 / (Em2 H(z))."""
    zp1 = s / Em2
    return n0r * zp1 * zp1 / (Em2 * H0 * np.sqrt(Om * zp1 ** 3 + OL))


@njit(cache=True)
def _pair_kinematic(prof, s, M2):
    if prof == 0:
        return phiphi_kinematic(s, M2)
    return vv_kinematic(s, M2)


@njit(cache=True, parallel=True)
def _step_integrals(E, zp1_lo, zp1_hi, zp1_mid, masses, n0, M2, MG, H0, Om, OL, delta_mode, prof,
                    hbin, J, Nq, Lq, Dq):
    """Per bin j and relic r, the step's time integrals of the target density times the four
    kinematic factors, J[j, r, :] = (s/BW, (s - M^2)/BW, 1/s, pair cross section), with
    BW = (s - M^2)^2 + (M Gamma)^2; the profile norms at the step's midpoint, Nq[j, r, :], and
    for forward-peaked daughters their part outside the parent's bin, Lq[j, r, :] (hbin = exp(dlnE/2)),
    and the shift in ln E of those staying in it, Dq[j, r, :] (vector_qe_drift)."""
    N = len(E)
    P = Nq.shape[2]
    b = MG
    for j in prange(N):
        tmp = np.empty(10)
        tmpL = np.empty(10)
        tmpD = np.empty(10)
        for r in range(3):
            for k in range(4):
                J[j, r, k] = 0.0
            for q in range(P):
                Nq[j, r, q] = 0.0
                Lq[j, r, q] = 0.0
                Dq[j, r, q] = 0.0
            m = masses[r]
            if m <= 0.0 or n0[r] <= 0.0:
                continue
            Em2 = 2.0 * E[j] * m
            s_lo = Em2 * zp1_lo
            s_hi = Em2 * zp1_hi
            mu = M2 / (Em2 * zp1_mid)
            if prof == 0:
                n2, n4, n5 = profile_norms(1.0 / mu, _GL24_X, _GL24_W)
                Nq[j, r, 0] = 1.0
                Nq[j, r, 1] = n2
                Nq[j, r, 2] = n2
                Nq[j, r, 3] = n4
                Nq[j, r, 4] = n5
                Nq[j, r, 5] = n4
                Nq[j, r, 6] = 1.0
            else:
                vector_norms(mu, _GL24_X, _GL24_W, tmp)
                vector_qe_complements(mu, hbin, _GL24_X, _GL24_W, tmpL)
                vector_qe_drift(mu, hbin, _GL24_X, _GL24_W, tmpD)
                for q in range(P):
                    Nq[j, r, q] = tmp[q]
                    Lq[j, r, q] = tmpL[q]
                    Dq[j, r, q] = tmpD[q]
            ds = s_hi - s_lo
            a = 0.0
            for k in range(len(_GL8_X)):
                s = s_lo + ds * _GL8_X[k]
                a += _GL8_W[k] * _w_of_s(s, Em2, n0[r], H0, Om, OL) / s
            J[j, r, 2] = a * ds
            if s_hi > 4.0 * M2:
                sa = max(s_lo, 4.0 * M2)
                dsp = s_hi - sa
                a = 0.0
                for k in range(len(_GL8_X)):
                    s = sa + dsp * _GL8_X[k]
                    a += _GL8_W[k] * _w_of_s(s, Em2, n0[r], H0, Om, OL) * _pair_kinematic(prof, s, M2)
                J[j, r, 3] = a * dsp
            v_lo = s_lo - M2
            v_hi = s_hi - M2
            if delta_mode:
                if v_lo <= 0.0 < v_hi:
                    J[j, r, 0] = M2 * _w_of_s(M2, Em2, n0[r], H0, Om, OL) * np.pi / b
                continue
            near = 0.0
            if v_lo > 0.0:
                near = v_lo
            elif v_hi < 0.0:
                near = -v_hi
            a1_ = 0.0
            a2_ = 0.0
            if near > 2.0 * ds:
                for k in range(len(_GL8_X)):
                    s = s_lo + ds * _GL8_X[k]
                    v = s - M2
                    wbw = _w_of_s(s, Em2, n0[r], H0, Om, OL) / (v * v + b * b)
                    a1_ += _GL8_W[k] * wbw * s
                    a2_ += _GL8_W[k] * wbw * v
                J[j, r, 0] = a1_ * ds
                J[j, r, 1] = a2_ * ds
            else:
                dth = np.arctan2(b * (v_hi - v_lo), b * b + v_hi * v_lo)
                for k in range(len(_GL8_X)):
                    tp = np.tan(dth * _GL8_X[k])
                    v = (v_lo + b * tp) / (1.0 - v_lo * tp / b)
                    s = v + M2
                    wv = _w_of_s(s, Em2, n0[r], H0, Om, OL)
                    a1_ += _GL8_W[k] * wv * s
                    a2_ += _GL8_W[k] * wv * v
                J[j, r, 0] = a1_ * dth / b
                J[j, r, 1] = a2_ * dth / b


@njit(cache=True, parallel=True)
def _thermal_resonance_terms(E, L_lo, L_hi, masses, T_nu0, eta_mass, M, MG, H0, Om, OL, J,
                             tglx, tglw, tglx10, tglw10, tfd16):
    """Replace J[j, r, 0] (s/BW) and J[j, r, 1] ((s - M^2)/BW) of _step_integrals by their averages over the relics'
    thermal momenta (frozen Fermi-Dirac, Moller flux factor; dp_thermal.relic_state_rates_analytic, modes 0 and 1)
    where the relics' motion matters: bins and relic states whose step lies within the thermally broadened
    resonance's reach, |2 E m (1+z) - M^2| < 4 (2v + Gamma/M) M^2 (v the typical relic speed), and whose broadened
    resonance is wider than two steps (narrower ones stay at rest, integrated exactly). The time integral over the
    step is Gauss-Legendre (8 nodes) in L = ln(1+z): J = int dL n <(s or s - M^2)/BW v> / H = int dL raw / (2 pi H).
    Added 2026-10-14 (docs/dev/PHYSICS_AUDIT.md, 1d): at rest, a lightest neutrino below ~1e-3 eV puts its
    resonance above the band, where thermally it falls near M^2 / (6 T)."""
    N = len(E)
    M2 = M * M
    sumG_eq = 12.0 * np.pi * MG / M2                  # the kernel's width coordinate: MG = M^2 sum_G / 12 pi
    dL = L_hi - L_lo
    zp1_mid = np.exp(0.5 * (L_lo + L_hi))
    for j in prange(N):
        flags = np.zeros(3, dtype=np.bool_)
        need = False
        for r in range(3):
            m = masses[r]
            if m <= 0.0 or eta_mass[r] <= 0.0:
                continue
            T_r = T_nu0 * zp1_mid * eta_mass[r] ** (1.0 / 3.0)
            p_t = 3.15 * T_r
            v = p_t / np.sqrt(m * m + p_t * p_t)
            width = 2.0 * v + MG / M2
            if width < 2.0 * dL:
                continue
            if abs(2.0 * E[j] * m * zp1_mid - M2) <= 4.0 * width * M2:
                flags[r] = True
                need = True
        if not need:
            continue
        a0 = np.zeros(3)
        a1 = np.zeros(3)
        for k in range(len(_GL8_X)):
            L = L_lo + dL * _GL8_X[k]
            zp1 = np.exp(L)
            H = H0 * np.sqrt(Om * zp1 ** 3 + OL)
            st0 = relic_state_rates_analytic(E[j], zp1 - 1.0, sumG_eq, M, masses, T_nu0, eta_mass, tglx, tglw,
                                             tglx10, tglw10, tfd16, True, 0)
            st1 = relic_state_rates_analytic(E[j], zp1 - 1.0, sumG_eq, M, masses, T_nu0, eta_mass, tglx, tglw,
                                             tglx10, tglw10, tfd16, True, 1)
            for r in range(3):
                a0[r] += _GL8_W[k] * st0[r] / (2.0 * np.pi * H)
                a1[r] += _GL8_W[k] * st1[r] / (2.0 * np.pi * H)
        for r in range(3):
            if flags[r]:
                J[j, r, 0] = a0[r] * dL
                J[j, r, 1] = a1[r] * dL


@njit(cache=True)
def _si_sweep(F, E, dE, S0, J, Nq, Lq, Dq, jt, w, qe, Ctau, Creg, tabB, tabO, node, dt, reg, diag, exact_bin):
    """One redshift step of the transport (module notes above). F (N, 3) is updated in place.
    S0 (N, 3): injection rate per unit energy over the step. J, Nq, Lq, Dq: _step_integrals. jt, w, qe,
    Ctau, Creg: the model. tabB (3, rows, P) / tabO (3, nodes+1): each relic's profile table
    (triangular); node (N, 3): the table node of bin j and relic r. reg (N, 3): scratch for the
    regeneration rates. diag (6,) accumulates: absorbed number (events changing something),
    secondaries in the grid, secondaries expected (in the grid or below it), absorbed energy, energy
    of the secondaries in the grid (at the bin centres), injected number.

    A bin's own equations: dF_c/dt = -G_c F_c + sum_a Q[c, a] G_a F_a + S_c over the step, G = tau/dt,
    Q the secondaries landing in the bin per event. With exact_bin false (the scalar: Q ~ dlnE) the
    secondaries landing in the bin are held at their step average (each state's exponential, plus a
    3 x 3 system); with exact_bin true (the vector, whose quasi-elastic scatterings change the mass
    state within the bin at rates far above the bin's losses -- a stiff system that the first scheme
    would integrate to first order only) the equations are solved exactly, F(dt) and int F dt from the
    exponential of the 7 x 7 augmented matrix [[-A, 0, S], [1, 0, 0], [0, 0, 0]].

    The forward-peaked daughters that stay in their parent's bin (no-ops, or conversions within the
    bin) each lose a little energy, ~M^2/s; at small M they are so many that the losses add up to many
    bins. They move as a drift in ln E: an upwind flux to the bin below, in the daughter's state, at
    d[a, c] = sum_r,q Creg J Dq / dlnE per step -- a loss of the bin and a source of the next one,
    which the top-down sweep passes on within the step however fast it is."""
    N = F.shape[0]
    P = len(jt)
    dlnE = np.log(E[1] / E[0])
    Dm = np.zeros((3, 3))
    for i in range(N):
        for c in range(3):
            reg[i, c] = 0.0
    tau = np.empty(3)
    Wn = np.zeros((3, P, 3, 3))       # [r, q, a, c] per absorbed a
    Q = np.empty((3, 3))
    Y = np.empty((3, P, 3))
    e_ = np.empty(3)
    a1_ = np.empty(3)
    g_ = np.empty(3)
    rhs = np.empty(3)
    R = np.empty(3)
    S = np.empty(3)
    Am = np.empty((3, 3))
    Bm = np.zeros((7, 7))
    Fnew = np.empty(3)
    for j in range(N - 1, -1, -1):
        for a in range(3):
            t = 0.0
            for r in range(3):
                for q in range(P):
                    if w[q] == 0.0:
                        continue
                    Jv = J[j, r, jt[q]]
                    if Jv == 0.0:
                        continue
                    if qe[q] != 0:
                        for c in range(3):
                            t += w[q] * Ctau[q, a, r, c] * Jv * (Lq[j, r, q] if c == a else Nq[j, r, q])
                    else:
                        sc = Ctau[q, a, r, 0] + Ctau[q, a, r, 1] + Ctau[q, a, r, 2]
                        t += w[q] * sc * Jv * Nq[j, r, q]
            tau[a] = t
        # the drift of the daughters staying in the bin (vector; zero for the scalar)
        drift = False
        for a in range(3):
            for c in range(3):
                dd = 0.0
                for r in range(3):
                    for q in range(P):
                        if qe[q] != 0:
                            dd += w[q] * Creg[q, a, r, c] * J[j, r, jt[q]] * Dq[j, r, q]
                Dm[a, c] = dd / dlnE
                if dd > 0.0:
                    drift = True
                    tau[a] += Dm[a, c]
        for r in range(3):
            for q in range(P):
                Jv = J[j, r, jt[q]]
                for a in range(3):
                    for c in range(3):
                        Wn[r, q, a, c] = Creg[q, a, r, c] * Jv / tau[a] if tau[a] != 0.0 else 0.0
        # own-bin secondaries per absorbed a (no forward-peaked daughter of the same state: no-ops)
        for c in range(3):
            for a in range(3):
                Q[c, a] = 0.0
        for r in range(3):
            o = tabO[r, node[j, r]]
            for q in range(P):
                b0 = tabB[r, o, q]
                for a in range(3):
                    for c in range(3):
                        if qe[q] != 0 and c == a:
                            continue
                        Q[c, a] += Wn[r, q, a, c] * b0
        for a in range(3):
            S[a] = S0[j, a] + reg[j, a]
        if exact_bin:
            # in units of the step (tau_s = t/dt) and of the step's source, so that every entry is
            # of order the rates: dF/dtau_s = -(A dt) F + (S dt / ssc) ssc, d<F>/dtau_s = F
            ssc = 0.0
            for c in range(3):
                ssc = max(ssc, abs(S[c]) * dt)
            if ssc == 0.0:
                ssc = 1.0
            for c in range(7):
                for a in range(7):
                    Bm[c, a] = 0.0
            for c in range(3):
                for a in range(3):
                    Ga = max(tau[a], 0.0)
                    Bm[c, a] = -(Ga if c == a else 0.0) + Q[c, a] * Ga
                Bm[c, 6] = S[c] * dt / ssc
                Bm[3 + c, c] = 1.0
            Ex = _expm_small(Bm)
            own = 0.0
            for a in range(3):
                # events of state a over the step (per unit energy), as a rate: (tau_a/dt) <F_a>
                avgF = Ex[3 + a, 6] * ssc
                for b in range(3):
                    avgF += Ex[3 + a, b] * F[j, b]
                R[a] = max(tau[a], 0.0) / dt * avgF
            for c in range(3):
                fn = Ex[c, 6] * ssc
                for b in range(3):
                    fn += Ex[c, b] * F[j, b]
                for a in range(3):
                    own += Q[c, a] * R[a]
                Fnew[c] = fn
            for c in range(3):
                F[j, c] = Fnew[c]
                diag[5] += S0[j, c] * dt * dE[j]
        else:
            for a in range(3):
                e_[a], a1_[a], g_[a] = _absorption_factors(max(tau[a], 0.0) / dt, dt)
                rhs[a] = F[j, a] * (1.0 - e_[a]) + S[a] * g_[a]
            for c in range(3):
                for a in range(3):
                    Am[c, a] = (dt if c == a else 0.0) - g_[c] * Q[c, a]
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
        if drift:
            for c in range(3):
                mv = 0.0
                for a in range(3):
                    if tau[a] > 0.0:
                        mv += R[a] * Dm[a, c] / tau[a]
                mv *= dE[j]
                diag[2] += mv * dt
                if j > 0:
                    reg[j - 1, c] += mv / dE[j - 1]
                    diag[1] += mv * dt
                    diag[4] += mv * dt * E[j - 1]
        absorbed = R[0] + R[1] + R[2]
        diag[0] += absorbed * dt * dE[j]
        diag[3] += absorbed * dt * dE[j] * E[j]
        diag[1] += own * dt * dE[j]
        diag[4] += own * dt * dE[j] * E[j]
        expected = 0.0
        for r in range(3):
            for q in range(P):
                for a in range(3):
                    for c in range(3):
                        expected += R[a] * Wn[r, q, a, c] * (Lq[j, r, q] if (qe[q] != 0 and c == a) else Nq[j, r, q])
        diag[2] += expected * dt * dE[j]
        for r in range(3):
            for q in range(P):
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
                    for q in range(P):
                        acc += Y[r, q, c] * tabB[r, o + d, q]
                    reg[i, c] += acc * inv
                    pushed += acc
                    diag[4] += acc * dt * E[i]
        diag[1] += pushed * dt


def _step_time_and_injection(L_lo, L_hi, H0, Om, OL, p_index, rho_evol, Phi_0):
    """(dt, T_inj) over the step ln(1+z) in [L_lo, L_hi]: dt = int dL / H(z) and
    T_inj = int Phi_0 (1+z)^(1-p) R(z) dL / H(z) (dt = dz / (H (1+z)) = dL / H), by Gauss-Legendre
    split at the kinks of R(z) (z = 1 and z = 4). The injection per unit observed energy is
    R Phi_0 (E (1+z))^-p times dE_z/dE = 1+z: the flux today int dz R (1+z)^-p / H(z) E^-p, the
    standard result (dp_cosmo.step_cosmological_pair; the (1+z) was missing until 2026-10-13)."""
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
            T += w * (b - a) * Phi_0 * zp1 ** (1.0 - p_index) * calc_R_source(zp1 - 1.0, rho_evol) / Hz
    return dt, T


def _tri_table(prof, mu0, n_nodes, stride, N, dlnE, Delta, n_u, offset=0.5, mu_list=None):
    """Triangular profile table (prof 0: scalar, 1: vector) for nodes n = 0..n_nodes-1 at
    mu = mu0 exp(-(n + offset) Delta) (or at mu_list), children d <= min(n // stride, N - 1)."""
    n = np.arange(n_nodes)
    if mu_list is None:
        mu_list = mu0 * np.exp(-(n + offset) * Delta)
    d_max = np.minimum(n // stride, N - 1).astype(np.int64)
    mu_list = np.ascontiguousarray(mu_list, dtype=np.float64)
    if prof == 0:
        return build_profile_table_tri(mu_list, d_max, dlnE, n_u)
    return build_vector_table_tri(mu_list, d_max, dlnE, n_u)


def _stack_tables(tabs, P):
    """Pad the relics' (B, offsets) into (3, rows, P) and (3, nodes + 1) arrays."""
    rows = max(t[0].shape[0] for t in tabs)
    nodes = max(len(t[1]) for t in tabs)
    B = np.zeros((3, rows, P))
    O = np.zeros((3, nodes), dtype=np.int64)
    for r, (b, o) in enumerate(tabs):
        B[r, :b.shape[0]] = b
        O[r, :len(o)] = o
        O[r, len(o):] = o[-1]
    return B, O


_TABLE_CACHE = {}
_TABLE_CACHE_MAX = 2
# Memory: the lattice tables grow as N^2 z_substeps (about 30 MB for the three relics at the
# analysis grid with the scalar's seven profiles, 515 bins with the extension and one step per bin;
# 1.5 GB at 1830 bins and four substeps). Only sets up to this size are kept between calls (scans
# over the coupling at a fixed mediator mass reuse them); larger ones are rebuilt each call and freed.
_TABLE_CACHE_MAX_BYTES = 128 * 2 ** 20


def _tables_nbytes(tabs):
    lat, top = tabs
    n = lat[0].nbytes + lat[1].nbytes
    if top is not None:
        n += top[0].nbytes + top[1].nbytes
    return n


def _profile_tables(prof, M, masses, E0, N, dlnE, q, K, top_full, L_top_mid, n_u):
    """Lattice and top-step profile tables for the three relics, cached (up to
    _TABLE_CACHE_MAX_BYTES each) by everything they depend on: the profile set, the mediator and
    relic masses and the grids -- not the couplings."""
    key = (int(prof), float(M), tuple(float(m) for m in masses), float(E0), int(N), float(dlnE), int(q), int(K),
           bool(top_full), float(L_top_mid), int(n_u))
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    P = N_PROFILES if prof == 0 else N_VPROFILES
    Delta = dlnE / q
    m_max = K - 1 if top_full else K - 2
    n_lat = (N - 1) * q + m_max + 1
    lat, top = [], []
    for r in range(3):
        if masses[r] <= 0.0:
            dummy = (np.zeros((1, P)), np.array([0, 1], dtype=np.int64))
            lat.append(dummy)
            top.append(dummy)
            continue
        mu0 = M * M / (2.0 * E0 * masses[r])
        lat.append(_tri_table(prof, mu0, n_lat, q, N, dlnE, Delta, n_u))
        if not top_full:
            mu_top = mu0 * np.exp(-np.arange(N) * dlnE - L_top_mid)
            top.append(_tri_table(prof, mu0, N, 1, N, dlnE, Delta, n_u, mu_list=mu_top))
    out = (_stack_tables(lat, P), _stack_tables(top, P) if not top_full else None)
    del lat, top
    if _tables_nbytes(out) <= _TABLE_CACHE_MAX_BYTES:
        while len(_TABLE_CACHE) >= _TABLE_CACHE_MAX:
            _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
        _TABLE_CACHE[key] = out
    return out


def model_spec(g_mass, mediator="scalar", majorana=True, phiphi="coherent", channels="full",
               dirac_sterile="track"):
    """The transport's description of a model: dict(prof, jt, w, qe, Ctau, runs, G2) where runs is a
    list of (Creg, weight), the visible flux being sum weight * F(run). Scalar and Majorana: one run;
    scalar and Dirac: 'track' -> runs (Cvis + Cinv, 1/2) and (Cvis - Cinv, 1/2) (sum and difference
    of visible and sterile, exact by parity), 'drop' -> (Cvis, 1) as 2107.13568; vector: one run."""
    med = str(mediator).strip().lower()
    if med == "scalar":
        if majorana:
            _, C, G2 = coupling_tensors(g_mass, phiphi, channels)
            return dict(prof=0, jt=SCALAR_JT, w=SCALAR_W, qe=SCALAR_QE, Ctau=C, runs=[(C, 1.0)], G2=G2)
        Cv, Ci, G2 = dirac_scalar_tensors(g_mass, channels)
        Ctau = np.ascontiguousarray(Cv + Ci)
        mode = str(dirac_sterile).strip().lower()
        if mode == "track":
            runs = [(Ctau, 0.5), (np.ascontiguousarray(Cv - Ci), 0.5)]
        elif mode == "drop":
            runs = [(Cv, 1.0)]
        else:
            raise ValueError(f"secret interactions: dirac_sterile must be 'track' or 'drop', got {dirac_sterile!r}")
        return dict(prof=0, jt=SCALAR_JT, w=SCALAR_W, qe=SCALAR_QE, Ctau=Ctau, runs=runs, G2=G2)
    if med == "vector":
        C, G2 = vector_tensors(g_mass, majorana, channels)
        return dict(prof=1, jt=VECTOR_JT, w=VECTOR_W, qe=VECTOR_QE, Ctau=C, runs=[(C, 1.0)], G2=G2)
    raise ValueError(f"secret interactions: mediator must be 'scalar' or 'vector', got {mediator!r}")


_STEP_TERMS = {}                        # one physics point's step terms (propagate_si)
_STEP_TERMS_MAX_BYTES = 1.5e9


def _step_key(E, L_edges, masses, n0, M2, MG, H0, Om, OL, delta_mode, prof, hbin, thermal, M, T_nu0, eta, jt):
    """Everything the steps' collision terms depend on (_step_integrals, _thermal_resonance_terms)."""
    import hashlib
    h = hashlib.sha256()
    for a in (E, L_edges, masses, n0, eta, jt):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((float(M2), float(MG), float(H0), float(Om), float(OL), bool(delta_mode), int(prof), float(hbin),
                   bool(thermal), float(M), float(T_nu0))).encode())
    return h.hexdigest()


def propagate_si(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_src,
                 p_index, rho_evol, Phi_0, M, g_mass, majorana=True, channels="full",
                 phiphi="coherent", z_substeps=1, E_extension=10.0, n_u=4096,
                 return_diagnostics=False, regeneration=True, mediator="scalar", dirac_sterile="track",
                 Gamma_vis=0.0, thermal=False):
    """Cosmological transport of the astrophysical neutrinos under secret interactions.

    Arguments: E_bins -- the (geometric) energy grid of the result, eV; z_max, H0 (eV),
    Omega_m, Omega_L -- sources up to z_max in flat LCDM; U_pmns -- (3, 3) mixing matrix;
    masses -- (3,) neutrino masses, eV; T_nu0 -- relic temperature today, eV; eta_mass -- (3,)
    relic density per mass state relative to the standard one; rho_src -- (3,) source flavor
    composition; p_index, rho_evol, Phi_0 -- source spectrum E^-p, evolution R(z) (1+z)^rho_evol
    (dp_cosmo.calc_R_source), normalization; M -- mediator mass, eV; g_mass -- (3, 3) mass-basis
    couplings (mass_basis_couplings_model); majorana -- the neutrinos' nature; mediator -- 'scalar'
    or 'vector' (left-handed currents); channels -- 'full', 's' (s channel only) or 'delta' (its
    narrow-width limit), the approximations compared in 2107.13568; phiphi -- 'coherent' or
    'paper' (scalar, Majorana; module notes); dirac_sterile -- 'track' or 'drop' (scalar, Dirac);
    z_substeps -- redshift steps per energy bin (Delta ln(1+z) = dlnE / z_substeps); E_extension --
    the grid is extended above E_bins[-1] by this factor so that what is injected above it cascades
    into it; n_u -- nodes of the pair-production spectra; regeneration -- False drops the
    secondaries (absorption alone, for validation); Gamma_vis -- a visible width (eV; e.g. e+e- of a mediator coupled
    to electrons): it widens the resonance and the s-channel events into it are absorbed without secondaries (vector
    only); thermal -- the s-channel terms averaged over the relics' thermal momenta where that matters
    (_thermal_resonance_terms; the dark portal's MeV regime, 2026-10-14); False keeps the relics at rest, as both
    papers and the secret-interaction planes.

    Returns the (N_E, 3) mass-state fluxes at z = 0 of the visible neutrinos (both helicities), on
    E_bins; with return_diagnostics also a dict of the number and energy bookkeeping (of the first
    run: for Dirac 'track', visible plus sterile)."""
    from dp_core import compute_dE_geometric
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
    g_mass = np.ascontiguousarray(g_mass, dtype=np.complex128)
    if not np.any(g_mass):
        # no coupling: free streaming (the transport's Breit-Wigner would have zero width)
        out = free_streaming_flux(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, rho_src, p_index, rho_evol, Phi_0)
        return (out, dict(no_coupling=True)) if return_diagnostics else out
    M2 = M * M
    Gam = mediator_width(M, g_mass, mediator, majorana) + float(Gamma_vis)
    MG = M * Gam
    spec = model_spec(g_mass, mediator, majorana, phiphi, channels, dirac_sterile)
    if Gamma_vis > 0.0:
        if spec["prof"] != 1:
            raise ValueError("secret interactions: a visible width is implemented for the vector only")
        # the s channel's events into visible pairs: absorbed, no secondaries. In the tensors' units the final-state
        # sum Tr(G G^dag) stands for Gamma_nu = M Tr / 24 pi, so the visible width adds 24 pi Gamma_vis / M,
        # spread over the three secondary slots of the absorption tensor only (Creg keeps the neutrino channels)
        X = 24.0 * np.pi * float(Gamma_vis) / M
        kap = 1.0 if majorana else 0.5
        Ctau = spec["Ctau"].copy()
        Gm = np.asarray(g_mass, dtype=np.complex128)
        for a in range(3):
            for r in range(3):
                add = kap * abs(Gm[r, a]) ** 2 * X / (4.0 * np.pi) / 3.0
                for q in (0, 1):
                    Ctau[q, a, r, :] += add
        spec = dict(spec, Ctau=np.ascontiguousarray(Ctau))
    eta_arr = np.asarray(eta_mass, dtype=np.float64)
    prof = spec["prof"]
    P = len(spec["jt"])
    delta_mode = str(channels).strip().lower() == "delta"
    n0 = relic_density_per_helicity(T_nu0) * np.asarray(eta_mass, dtype=np.float64)
    P_src = (np.abs(np.asarray(U_pmns)) ** 2).T @ np.asarray(rho_src, dtype=np.float64)

    (lat_B, lat_O), top = _profile_tables(prof, M, masses, E[0], N, dlnE, q, K, top_full, L_top_mid, n_u)

    runs = spec["runs"]
    Fs = [np.zeros((N, 3)) for _ in runs]
    diags = [np.zeros(6) for _ in runs]
    reg = np.zeros((N, 3))
    J = np.zeros((N, 3, 4))
    Nq = np.zeros((N, 3, P))
    Lq = np.zeros((N, 3, P))
    Dq = np.zeros((N, 3, P))
    node = np.zeros((N, 3), dtype=np.int64)
    jj = np.arange(N, dtype=np.int64)
    Ep = E ** (-p_index)
    hbin = float(np.exp(0.5 * dlnE))
    Cregs = [np.ascontiguousarray(C if regeneration else np.zeros_like(C)) for C, _ in runs]
    # the steps' collision terms do not depend on the source (spectral index, evolution, composition): kept from one
    # call to the next with the same physics (2026-10-14: an analysis runs every spectral-index node of a point in a
    # row, and the terms were half of each run)
    skey = _step_key(E, L_edges, masses, n0, M2, MG, H0, Omega_m, Omega_L, delta_mode, prof, hbin, thermal, M,
                     T_nu0, eta_arr, spec["jt"])
    steps = _STEP_TERMS.get(skey)
    store = None
    if steps is None and K * J.nbytes + 3 * K * Nq.nbytes <= _STEP_TERMS_MAX_BYTES:
        store = []
    for m in range(K - 1, -1, -1):
        L_lo, L_hi = L_edges[m], L_edges[m + 1]
        dt, T_inj = _step_time_and_injection(L_lo, L_hi, H0, Omega_m, Omega_L, p_index, rho_evol, Phi_0)
        S0 = np.ascontiguousarray(np.outer(Ep * (T_inj / dt), P_src))
        if steps is not None:
            J, Nq, Lq, Dq = steps[K - 1 - m]
        else:
            _step_integrals(E, np.exp(L_lo), np.exp(L_hi), np.exp(0.5 * (L_lo + L_hi)), masses, n0, M2, MG,
                            H0, Omega_m, Omega_L, delta_mode, prof, hbin, J, Nq, Lq, Dq)
            if thermal and not delta_mode:
                _thermal_resonance_terms(E, L_lo, L_hi, masses, T_nu0, eta_arr, M, MG, H0, Omega_m, Omega_L, J,
                                         _TGL16_X, _TGL16_W, _TGL10_X, _TGL10_W, _TFD16)
            if store is not None:
                store.append((J.copy(), Nq.copy(), Lq.copy(), Dq.copy()))
        if m == K - 1 and not top_full:
            tB, tO = top
            node[:] = jj[:, None]
        else:
            tB, tO = lat_B, lat_O
            node[:] = (jj * q + m)[:, None]
        for r in range(3):
            if masses[r] <= 0.0:
                node[:, r] = 0
        for k in range(len(runs)):
            _si_sweep(Fs[k], E, dE, S0, J, Nq, Lq, Dq, spec["jt"], spec["w"], spec["qe"], spec["Ctau"], Cregs[k],
                      tB, tO, node, dt, reg, diags[k], prof == 1)
    if store is not None:
        _STEP_TERMS.clear()
        _STEP_TERMS[skey] = store
    F = sum(wt * Fk for (_, wt), Fk in zip(runs, Fs))
    out = np.ascontiguousarray(F[:N_E])
    if not return_diagnostics:
        return out
    diag = diags[0]
    F0 = Fs[0]
    info = dict(absorbed=diag[0], secondaries_in_grid=diag[1], secondaries_expected=diag[2],
                energy_absorbed=diag[3], energy_secondaries_in_grid=diag[4], injected=diag[5],
                final_number=float(np.sum(F0 * dE[:, None])), final_number_in_E_bins=float(np.sum(F0[:N_E] * dE[:N_E, None])),
                final_energy=float(np.sum(F0 * (E * dE)[:, None])), width=Gam, n_steps=K, N_grid=N, top_full=top_full,
                F_extended=F.copy(), E_extended=E.copy(), visible_number=float(np.sum(F * dE[:, None])))
    return out, info


def free_streaming_flux(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, rho_src, p_index, rho_evol, Phi_0,
                        n=4000):
    """The mass-state fluxes without interactions, (N_E, 3): E^-p P_a int_0^z_max Phi_0 (1+z)^-p
    R(z) / H(z) dz (the standard result: every emitted neutrino arrives), by Gauss-Legendre on n
    nodes in ln(1+z) split at the kinks of R(z) -- the reference for the no-coupling limit of
    propagate_si."""
    from dp_cosmo import calc_R_source
    Lmax = np.log1p(z_max)
    cuts = [0.0] + [L for L in (np.log(2.0), np.log(5.0)) if L < Lmax] + [Lmax]
    x, w = _leggauss(n)
    T = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        L = a + (b - a) * 0.5 * (x + 1.0)
        zp1 = np.exp(L)
        R = np.array([calc_R_source(z, rho_evol) for z in zp1 - 1.0])
        T += np.sum(0.5 * w * (b - a) * Phi_0 * zp1 ** (1.0 - p_index) * R / (H0 * np.sqrt(Omega_m * zp1 ** 3 + Omega_L)))
    P_src = (np.abs(np.asarray(U_pmns)) ** 2).T @ np.asarray(rho_src, dtype=np.float64)
    return np.outer(np.asarray(E_bins, dtype=np.float64) ** (-p_index) * T, P_src)
