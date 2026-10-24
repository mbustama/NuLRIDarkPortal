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
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from dp_core import njit, prange
import numba  # after dp_core (it sets numba's CPU features)
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


def dirac_scalar_chiral_tensors(h, channels="full"):
    """dirac_scalar_tensors split by the relic component the incident visible nu_a meets (2026-10-21; derived from the
    helicity structure and checked against the brute force, tests/test_si.py): (Lv, Li, Rv, Ri, G2), the visible and
    sterile secondaries on a relic's left-chiral component (a nu of helicity -1/2, a nubar of +1/2: the weakly
    interacting ones) and on its right-chiral one, each at unit weight; dirac_scalar_tensors is their mean (a relic
    at rest presents both halves). Every scalar vertex flips chirality, so with the final neutrinos massless each
    final helicity configuration picks one component: the left-chiral one takes part in nu nu -> nu nu through t and
    u together (both final neutrinos sterile: the interference P5), in nu nubar -> nu nubar through t alone (both
    sterile) and in double-scalar production; the right-chiral one in nu nu -> nu nu through t or u apart (the
    relic's partner visible) and in nu nubar -> nu nubar through s and t (their interference P4, P6; the recoil
    visible). The incident's own daughter (P2) is sterile on either. An incident sterile neutrino on one component
    gives the visible one's secondaries on the other with visibility swapped (parity; checked)."""
    Cv, Ci, G2 = dirac_scalar_tensors(h, channels)
    Lv, Li, Rv, Ri = (np.zeros_like(Cv) for _ in range(4))
    Li[1] = Ci[1]
    Ri[1] = Ci[1]
    Li[2] = 2.0 * Ci[2]
    Rv[2] = 2.0 * Cv[2]
    Li[4] = 2.0 * Ci[4]
    Lv[6] = 2.0 * Cv[6]
    Li[6] = 2.0 * Ci[6]
    Rv[0] = 2.0 * Cv[0]
    Ri[0] = 2.0 * Ci[0]
    Ri[3] = 2.0 * Ci[3]
    Rv[5] = 2.0 * Cv[5]
    return Lv, Li, Rv, Ri, G2


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
def _bin_matrix(tau, lose, Q, Qn, Mx):
    """A bin's rate matrix over the step (exact_bin): Mx[c, a] = Q[c, a] tau_a off the diagonal (the events of state a
    whose secondary lands in the bin as c), and on it minus the sum of its column's other entries minus the bin's net
    loss to state a's events, lose_a - sum_c Qn[c, a] tau_a (lose: the events whose daughters leave the bin, the
    non-forward-peaked events, the drift; Qn: the secondaries of the latter landing in the bin), floored at zero.
    The same matrix as -tau_a delta_ca + Q[c, a] tau_a, but the quasi-elastic conversions within the bin, which at
    small mu = M^2/s outnumber the losses by ~1/mu (2026-10-22: 1e30 below 1 eV), now cancel exactly in each column
    instead of leaving their rounding (1e-16 of them) as a spurious growing mode (beyond STIFF_BIN; below it the
    matrix as it was formed before). Leading the first n rows of Mx."""
    n = tau.shape[0]
    big = 0.0
    for a in range(n):
        big = max(big, tau[a])
    if big <= BIN_ROUNDING_MAX:
        # every MeV plane: the matrix as it was formed before, bit for bit
        for a in range(n):
            Ga = max(tau[a], 0.0)
            for c in range(n):
                Mx[c, a] = -(Ga if c == a else 0.0) + Q[c, a] * Ga
        return
    # beyond it the conversions within the bin, rescaled (2026-10-24): at 1e16-1e30 per step against losses of order
    # one, the losses vanish in the diagonal's rounding (-col - loss == -col) and the exponential keeps only its
    # rounding -- 1e-16 of the conversions, a flux loss of up to 100% (synthetic bin: 1e12 per step +-1.5e-4 per step,
    # 1e16 a mean 0.22 for 0.99). Scaled down to CONVERSION_CAP per step (at least CONVERSION_RATIO x the bin's
    # largest loss, every state's own at least CONVERSION_MIN): their equilibrium within the bin is the same (the
    # ratios kept), reached within the step as before, and the bin's loss rate on it the same to ~loss/conversions.
    cmax = 0.0
    cmin = np.inf
    lmax = 0.0
    for a in range(n):
        Ga = max(tau[a], 0.0)
        col = 0.0
        gain = 0.0
        for c in range(n):
            if c != a:
                col += Q[c, a] * Ga
            gain += Qn[c, a] * Ga
        cmax = max(cmax, col)
        if col > 0.0:
            cmin = min(cmin, col)
        lmax = max(lmax, lose[a] - gain)
    cap = max(CONVERSION_CAP, CONVERSION_RATIO * lmax)
    sc = 1.0
    if cmax > cap:
        sc = max(cap / cmax, CONVERSION_MIN / cmin)
        if sc > 1.0:
            sc = 1.0
    for a in range(n):
        Ga = max(tau[a], 0.0)
        col = 0.0
        gain = 0.0
        for c in range(n):
            if c != a:
                Mx[c, a] = sc * Q[c, a] * Ga
                col += Mx[c, a]
            gain += Qn[c, a] * Ga
        Mx[a, a] = -col - max(lose[a] - gain, 0.0)


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
    Ex = np.zeros((7, 7))
    Fnew = np.empty(3)
    Qn = np.empty((3, 3))
    lose = np.empty(3)
    for j in range(N - 1, -1, -1):
        for a in range(3):
            t = 0.0
            tl = 0.0
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
                            tl += w[q] * Ctau[q, a, r, c] * Jv * Lq[j, r, q]
                    else:
                        sc = Ctau[q, a, r, 0] + Ctau[q, a, r, 1] + Ctau[q, a, r, 2]
                        t += w[q] * sc * Jv * Nq[j, r, q]
                        tl += w[q] * sc * Jv * Nq[j, r, q]
            tau[a] = t
            lose[a] = tl
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
                    lose[a] += Dm[a, c]
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
                Qn[c, a] = 0.0
        for r in range(3):
            o = tabO[r, node[j, r]]
            for q in range(P):
                b0 = tabB[r, o, q]
                for a in range(3):
                    for c in range(3):
                        if qe[q] != 0 and c == a:
                            continue
                        Q[c, a] += Wn[r, q, a, c] * b0
                        if qe[q] == 0:
                            Qn[c, a] += Wn[r, q, a, c] * b0
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
            _bin_matrix(tau, lose, Q, Qn, Bm)
            for c in range(3):
                Bm[c, 6] = S[c] * dt / ssc
                Bm[3 + c, c] = 1.0
            big = 0.0
            for c in range(3):
                for a in range(3):
                    big = max(big, abs(Bm[c, a]))
            if big > STIFF_BIN:
                # the augmented exponential's blocks from _phi_eig: F(dt) = e^M F + phi1 S dt, <F> = phi1 F + phi2 S dt
                Em, P1, P2 = _phi_eig(np.ascontiguousarray(Bm[0:3, 0:3]))
                for c in range(3):
                    for a in range(3):
                        Ex[c, a] = Em[c, a]
                        Ex[3 + c, a] = P1[c, a]
                    Ex[c, 6] = 0.0
                    Ex[3 + c, 6] = 0.0
                    for b in range(3):
                        Ex[c, 6] += P1[c, b] * Bm[b, 6]
                        Ex[3 + c, 6] += P2[c, b] * Bm[b, 6]
            else:
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


# A bin's rate matrix (per step) beyond this is exponentiated keeping its blocks those of a rate matrix (_phi_eig):
# rates ~1/mu, which below ~1 eV reach 1e30 per step, leave a rounding of 1e-16 of them in the eigenvalue that should
# be zero -- e^+1e14 in plain scaling and squaring (2026-10-22). Well below it (every MeV plane), the augmented
# exponential as before.
STIFF_BIN = 1e12
# _bin_matrix: events per step beyond which the conversions within the bin are rescaled (below it, bit for bit as
# before; a synthetic bin there is exact to 1e-6), and the conversions per step kept: CONVERSION_CAP, at least
# CONVERSION_RATIO x the bin's largest loss and every state's at least CONVERSION_MIN (exact where the true ones are).
BIN_ROUNDING_MAX = 1e8
CONVERSION_CAP = 1e6
CONVERSION_RATIO = 1e7
CONVERSION_MIN = 1e3


@njit(cache=True)
def _phi_eig(Mx):
    """(e^M, phi1(M), phi2(M)) of a stiff n x n rate matrix M (off the diagonal >= 0, columns summing to <= 0; its
    exponential non-negative with columns summing to <= 1): the augmented exponential of M 2^-s (norm below 1/2,
    the diagonal Pade approximant) squared s times, with the blocks kept as the rate matrix's exponential must be --
    e^(M t) non-negative with columns summing to <= 1, t phi1 and t^2 phi2 non-negative -- after every squaring, so that
    the rounding of rates ~1e30 (1e-16 of them, either sign) cannot grow (2026-10-22: an eigenvector route blew up
    where two eigenvalues nearly meet). Doubling t: A' = A A, B' = A B + B, C' = A C + t B + C, for A = e^(M t),
    B = t phi1(M t), C = t^2 phi2(M t); at t = 1 they are the three functions."""
    n = Mx.shape[0]
    nrm = 0.0
    for c in range(n):
        rs = 0.0
        for a in range(n):
            rs += abs(Mx[c, a])
        nrm = max(nrm, rs)
    sq = 0
    if nrm > 0.5:
        sq = int(np.ceil(np.log2(nrm / 0.5)))
    t = 1.0 / (2.0 ** sq)
    Aug = np.zeros((3 * n, 3 * n))
    for c in range(n):
        for a in range(n):
            Aug[c, a] = Mx[c, a] * t
        Aug[c, n + c] = t
        Aug[n + c, 2 * n + c] = t
    Ex = _expm_small(Aug)                    # its norm is below ~1/2 + 2t: no squaring inside
    A = np.ascontiguousarray(Ex[0:n, 0:n])
    B = np.ascontiguousarray(Ex[0:n, n:2 * n])
    C = np.ascontiguousarray(Ex[0:n, 2 * n:3 * n])
    _keep_rate_blocks(A, B, C)
    for _ in range(sq):
        A2 = A @ A
        B2 = A @ B + B
        C2 = A @ C + t * B + C
        A, B, C = A2, B2, C2
        t *= 2.0
        _keep_rate_blocks(A, B, C)
    return A, B, C


@njit(cache=True)
def _keep_rate_blocks(A, B, C):
    """A non-negative with columns summing to <= 1, B and C non-negative (in place)."""
    n = A.shape[0]
    for a in range(n):
        col = 0.0
        for c in range(n):
            if A[c, a] < 0.0:
                A[c, a] = 0.0
            col += A[c, a]
            if B[c, a] < 0.0:
                B[c, a] = 0.0
            if C[c, a] < 0.0:
                C[c, a] = 0.0
        if col > 1.0:
            for c in range(n):
                A[c, a] /= col


@njit(cache=True)
def _phi_functions(Mx):
    """(e^M, phi1(M), phi2(M)) of an n x n matrix M, phi1 = sum M^k/(k+1)!, phi2 = sum M^k/(k+2)!, from the
    exponential of the augmented 3n x 3n matrix [[M, I, 0], [0, 0, I], [0, 0, 0]] (whose first block row is
    [e^M, phi1, phi2]); beyond STIFF_BIN through the eigenvalues (_phi_eig)."""
    n = Mx.shape[0]
    big = 0.0
    for c in range(n):
        for a in range(n):
            big = max(big, abs(Mx[c, a]))
    if big > STIFF_BIN:
        return _phi_eig(Mx)
    A = np.zeros((3 * n, 3 * n))
    for c in range(n):
        for a in range(n):
            A[c, a] = Mx[c, a]
        A[c, n + c] = 1.0
        A[n + c, 2 * n + c] = 1.0
    Ex = _expm_small(A)
    return (np.ascontiguousarray(Ex[0:n, 0:n]), np.ascontiguousarray(Ex[0:n, n:2 * n]),
            np.ascontiguousarray(Ex[0:n, 2 * n:3 * n]))


@njit(cache=True)
def _si_sweep_cols(F, E, dE, S0, J, Nq, Lq, Dq, jt, w, qe, Ctau, Creg, tabB, tabO, node, dt, reg, diag, exact_bin,
                   dec_rate, dec_br, zp1, use_dec):
    """_si_sweep for several sources at once (2026-10-18): F, S0 and reg are (N, n, C), one column per source, diag
    (C, 6). With use_dec (2026-10-24; dp_decay, Majorana): the decay into the mediator as one more channel of every
    bin -- state a decays at dec_rate[a] / (E zp1) (rest rate x mass over the energy, the lab rate), its daughter in
    state c with dec_br[a, c]; for the one Majorana flux holding both kinds (the kept and the flipped halves) their
    spectrum is flat in the share z of the parent's energy. The bin's own daughters enter its matrix (conversions and
    decays solved together, exactly), those below go down by running sums before their bins are solved: a cascade of
    conversion-decay cycles within the step, however fast. The rest of the docstring as before:
    (C, 6). The collision operator of a bin does not depend on the source, so it is built once per bin for every
    column: the absorption rates, the own-bin secondaries, and (exact_bin) e^M, phi1(M) and phi2(M) of the bin's
    rate matrix M (the blocks of the 7 x 7 augmented exponential _si_sweep takes per source: F(dt) = e^M F + phi1 S dt,
    <F> = phi1 F + phi2 S dt). The secondaries pushed to the bins below go through the source-independent kernel
    K_d[a, c] = sum_r,q Wn[r, q, a, c] tabB[r, o_r + d, q] (9 numbers per child bin, then 9 products per column)
    instead of each source's 90 per child bin. Every column's arithmetic is the same whatever the other columns, so
    a column's result does not depend on the batch it was computed in; against _si_sweep it agrees to rounding.
    n transported states (F.shape[1]: the 3 mass states, or 6 for a Dirac scalar on polarized relics -- visible then
    sterile) and nr relic slots (Nq.shape[1]: the 3 relic states, or 6 -- their left- then right-chiral shares, 2026-10-21);
    the tensors are [q, a, r, c] over those. With n = nr = 3 the arithmetic is the earlier 3-state sweep's."""
    N = F.shape[0]
    n = F.shape[1]
    C = F.shape[2]
    nr = Nq.shape[1]
    P = len(jt)
    dlnE = np.log(E[1] / E[0])
    Dm = np.zeros((n, n))
    for i in range(N):
        for c in range(n):
            for k in range(C):
                reg[i, c, k] = 0.0
    tau = np.empty(n)
    Wn = np.zeros((nr, P, n, n))
    Q = np.empty((n, n))
    e_ = np.empty(n)
    a1_ = np.empty(n)
    g_ = np.empty(n)
    Am = np.empty((n, n))
    Ainv = np.empty((n, n))
    Mx = np.empty((n, n))
    S = np.empty((n, C))
    R = np.empty((n, C))
    Fnew = np.empty(n)
    rhs = np.empty(n)
    Kd = np.empty((n, n))
    active = np.zeros((nr, P), dtype=np.bool_)
    oo = np.empty(nr, dtype=np.int64)
    dm = np.empty(nr, dtype=np.int64)
    Qn = np.empty((n, n))
    lose = np.empty(n)
    Sdec = np.zeros((n, C))           # the decays' daughters per unit energy below the bins swept so far
    tdec = np.zeros(n)
    hgeo = np.sqrt(E[1] / E[0])
    for j in range(N - 1, -1, -1):
        if use_dec:
            # the daughters of the bins above, a flat density per unit energy below each parent
            for c in range(n):
                for k in range(C):
                    reg[j, c, k] += Sdec[c, k]
        for a in range(n):
            t = 0.0
            tl = 0.0
            for r in range(nr):
                for q in range(P):
                    if w[q] == 0.0:
                        continue
                    Jv = J[j, r, jt[q]]
                    if Jv == 0.0:
                        continue
                    if qe[q] != 0:
                        for c in range(n):
                            t += w[q] * Ctau[q, a, r, c] * Jv * (Lq[j, r, q] if c == a else Nq[j, r, q])
                            tl += w[q] * Ctau[q, a, r, c] * Jv * Lq[j, r, q]
                    else:
                        sc = Ctau[q, a, r, 0]
                        for c in range(1, n):
                            sc += Ctau[q, a, r, c]
                        t += w[q] * sc * Jv * Nq[j, r, q]
                        tl += w[q] * sc * Jv * Nq[j, r, q]
            tdec[a] = dec_rate[a] / (E[j] * zp1) * dt if use_dec else 0.0
            tau[a] = t + tdec[a]
            lose[a] = tl + tdec[a]
        drift = False
        for a in range(n):
            for c in range(n):
                dd = 0.0
                for r in range(nr):
                    for q in range(P):
                        if qe[q] != 0:
                            dd += w[q] * Creg[q, a, r, c] * J[j, r, jt[q]] * Dq[j, r, q]
                Dm[a, c] = dd / dlnE
                if dd > 0.0:
                    drift = True
                    tau[a] += Dm[a, c]
                    lose[a] += Dm[a, c]
        for r in range(nr):
            for q in range(P):
                Jv = J[j, r, jt[q]]
                act = False
                for a in range(n):
                    for c in range(n):
                        v = Creg[q, a, r, c] * Jv / tau[a] if tau[a] != 0.0 else 0.0
                        Wn[r, q, a, c] = v
                        if v != 0.0:
                            act = True
                active[r, q] = act
        for c in range(n):
            for a in range(n):
                Q[c, a] = 0.0
                Qn[c, a] = 0.0
        for r in range(nr):
            o = tabO[r, node[j, r]]
            for q in range(P):
                b0 = tabB[r, o, q]
                for a in range(n):
                    for c in range(n):
                        if qe[q] != 0 and c == a:
                            continue
                        Q[c, a] += Wn[r, q, a, c] * b0
                        if qe[q] == 0:
                            Qn[c, a] += Wn[r, q, a, c] * b0
        if use_dec:
            # the decays' daughters staying in the bin: z from E_lo / E_j to 1 of the flat spectrum
            own_z = 1.0 - 1.0 / hgeo
            for a in range(n):
                if tdec[a] > 0.0 and tau[a] > 0.0:
                    for c in range(n):
                        v = tdec[a] / tau[a] * dec_br[a, c] * own_z
                        Q[c, a] += v
                        Qn[c, a] += v
        for k in range(C):
            for a in range(n):
                S[a, k] = S0[j, a, k] + reg[j, a, k]
        if exact_bin:
            _bin_matrix(tau, lose, Q, Qn, Mx)
            Em, P1, P2 = _phi_functions(Mx)
            for k in range(C):
                for a in range(n):
                    avg = 0.0
                    for b in range(n):
                        avg += P1[a, b] * F[j, b, k] + P2[a, b] * S[b, k] * dt
                    R[a, k] = max(tau[a], 0.0) / dt * avg
                for c in range(n):
                    fn = 0.0
                    for b in range(n):
                        fn += Em[c, b] * F[j, b, k] + P1[c, b] * S[b, k] * dt
                    Fnew[c] = fn
                for c in range(n):
                    F[j, c, k] = Fnew[c]
                    diag[k, 5] += S0[j, c, k] * dt * dE[j]
        else:
            for a in range(n):
                e_[a], a1_[a], g_[a] = _absorption_factors(max(tau[a], 0.0) / dt, dt)
            for c in range(n):
                for a in range(n):
                    Am[c, a] = (dt if c == a else 0.0) - g_[c] * Q[c, a]
            if n == 3:
                c00 = Am[1, 1] * Am[2, 2] - Am[1, 2] * Am[2, 1]
                c01 = Am[1, 2] * Am[2, 0] - Am[1, 0] * Am[2, 2]
                c02 = Am[1, 0] * Am[2, 1] - Am[1, 1] * Am[2, 0]
                det = Am[0, 0] * c00 + Am[0, 1] * c01 + Am[0, 2] * c02
                Ainv[0, 0], Ainv[1, 0], Ainv[2, 0] = c00 / det, c01 / det, c02 / det
                Ainv[0, 1] = (Am[0, 2] * Am[2, 1] - Am[0, 1] * Am[2, 2]) / det
                Ainv[1, 1] = (Am[0, 0] * Am[2, 2] - Am[0, 2] * Am[2, 0]) / det
                Ainv[2, 1] = (Am[0, 1] * Am[2, 0] - Am[0, 0] * Am[2, 1]) / det
                Ainv[0, 2] = (Am[0, 1] * Am[1, 2] - Am[0, 2] * Am[1, 1]) / det
                Ainv[1, 2] = (Am[0, 2] * Am[1, 0] - Am[0, 0] * Am[1, 2]) / det
                Ainv[2, 2] = (Am[0, 0] * Am[1, 1] - Am[0, 1] * Am[1, 0]) / det
            else:
                Ainv[:, :] = np.linalg.inv(Am)
            for k in range(C):
                for a in range(n):
                    rhs[a] = F[j, a, k] * (1.0 - e_[a]) + S[a, k] * g_[a]
                for a in range(n):
                    acc = Ainv[a, 0] * rhs[0]
                    for b in range(1, n):
                        acc += Ainv[a, b] * rhs[b]
                    R[a, k] = acc
                for c in range(n):
                    sc = S[c, k]
                    for a in range(n):
                        sc += Q[c, a] * R[a, k]
                    F[j, c, k] = F[j, c, k] * e_[c] + sc * a1_[c]
                    diag[k, 5] += S0[j, c, k] * dt * dE[j]
        if use_dec:
            # the decay rate of each state over the step (R carries every event: its decay share tdec / tau), its
            # daughters below the bin: dN/dE = rate dE_j br / E_j for E < E_lo,j
            for k in range(C):
                for a in range(n):
                    if tdec[a] > 0.0 and tau[a] > 0.0:
                        rate = R[a, k] * tdec[a] / tau[a] * dE[j]
                        for c in range(n):
                            Sdec[c, k] += rate * dec_br[a, c] / E[j]
        for k in range(C):
            own = 0.0
            for c in range(n):
                for a in range(n):
                    own += Q[c, a] * R[a, k]
            if drift:
                for c in range(n):
                    mv = 0.0
                    for a in range(n):
                        if tau[a] > 0.0:
                            mv += R[a, k] * Dm[a, c] / tau[a]
                    mv *= dE[j]
                    diag[k, 2] += mv * dt
                    if j > 0:
                        reg[j - 1, c, k] += mv / dE[j - 1]
                        diag[k, 1] += mv * dt
                        diag[k, 4] += mv * dt * E[j - 1]
            absorbed = R[0, k]
            for a in range(1, n):
                absorbed += R[a, k]
            diag[k, 0] += absorbed * dt * dE[j]
            diag[k, 3] += absorbed * dt * dE[j] * E[j]
            diag[k, 1] += own * dt * dE[j]
            diag[k, 4] += own * dt * dE[j] * E[j]
            expected = 0.0
            for r in range(nr):
                for q in range(P):
                    if not active[r, q]:
                        continue
                    for a in range(n):
                        for c in range(n):
                            expected += R[a, k] * Wn[r, q, a, c] * (Lq[j, r, q] if (qe[q] != 0 and c == a)
                                                                    else Nq[j, r, q])
            diag[k, 2] += expected * dt * dE[j]
        # the secondaries below the bin, through the source-independent kernel
        dmax_all = 0
        for r in range(nr):
            oo[r] = tabO[r, node[j, r]]
            nrow = tabO[r, node[j, r] + 1] - oo[r]
            dm[r] = min(j, nrow - 1)
            dmax_all = max(dmax_all, dm[r])
        for d in range(1, dmax_all + 1):
            i = j - d
            for a in range(n):
                for c in range(n):
                    Kd[a, c] = 0.0
            for r in range(nr):
                if d > dm[r]:
                    continue
                row = oo[r] + d
                for q in range(P):
                    if not active[r, q]:
                        continue
                    tb = tabB[r, row, q]
                    if tb == 0.0:
                        continue
                    for a in range(n):
                        for c in range(n):
                            Kd[a, c] += Wn[r, q, a, c] * tb
            for k in range(C):
                for c in range(n):
                    acc = R[0, k] * Kd[0, c]
                    for a in range(1, n):
                        acc += R[a, k] * Kd[a, c]
                    acc *= dE[j]
                    reg[i, c, k] += acc / dE[i]
                    diag[k, 1] += acc * dt
                    diag[k, 4] += acc * dt * E[i]


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
# over the coupling at a fixed mediator mass reuse them, and every run of a plane point shares them);
# larger ones are rebuilt each call and freed. Raised from 128 MB on 2026-10-18: the vector's set at the
# future detectors' grid (745 bins with the extension) is 145 MB, so every run of their MeV points rebuilt it
# (a quarter of the point's time). A set above 128 MB is kept alone.
_TABLE_CACHE_MAX_BYTES = 320 * 2 ** 20
_TABLE_CACHE_ALONE_BYTES = 128 * 2 ** 20


# one relic's lattice and top tables, per (profile set, mediator mass, relic mass, grids): _profile_tables
_RELIC_TABLES = {}
_RELIC_TABLES_MAX_BYTES = 384 * 2 ** 20


def _relic_nbytes(v):
    return v[0][0].nbytes + (0 if v[1] is None else v[1][0].nbytes)


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
        # each relic's tables depend on its own mass only (2026-10-18): kept per relic, so that a run moving one
        # relic's mass (Delta m^2_21, Delta m^2_31) rebuilds only that relic's
        rkey = key[:2] + (float(masses[r]),) + key[3:]
        hit_r = _RELIC_TABLES.get(rkey)
        if hit_r is None:
            mu0 = M * M / (2.0 * E0 * masses[r])
            t_lat = _tri_table(prof, mu0, n_lat, q, N, dlnE, Delta, n_u)
            t_top = None
            if not top_full:
                mu_top = mu0 * np.exp(-np.arange(N) * dlnE - L_top_mid)
                t_top = _tri_table(prof, mu0, N, 1, N, dlnE, Delta, n_u, mu_list=mu_top)
            hit_r = (t_lat, t_top)
            nb = t_lat[0].nbytes + (0 if t_top is None else t_top[0].nbytes)
            if nb <= _RELIC_TABLES_MAX_BYTES // 3:
                while _RELIC_TABLES and sum(_relic_nbytes(v) for v in _RELIC_TABLES.values()) + nb > _RELIC_TABLES_MAX_BYTES:
                    _RELIC_TABLES.pop(next(iter(_RELIC_TABLES)))
                _RELIC_TABLES[rkey] = hit_r
        lat.append(hit_r[0])
        if not top_full:
            top.append(hit_r[1])
    out = (_stack_tables(lat, P), _stack_tables(top, P) if not top_full else None)
    del lat, top
    nbytes = _tables_nbytes(out)
    if nbytes <= _TABLE_CACHE_MAX_BYTES:
        while len(_TABLE_CACHE) >= (1 if nbytes > _TABLE_CACHE_ALONE_BYTES else _TABLE_CACHE_MAX):
            _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
        _TABLE_CACHE[key] = out
    return out


def model_spec(g_mass, mediator="scalar", majorana=True, phiphi="coherent", channels="full",
               dirac_sterile="track", polarized=False):
    """The transport's description of a model: dict(prof, jt, w, qe, Ctau, runs, G2, pols, n) where runs is a
    list of (Creg, weight), the visible flux being sum weight * F(run) over the first three states. Scalar and
    Majorana: one run; scalar and Dirac: 'track' -> runs (Cvis + Cinv, 1/2) and (Cvis - Cinv, 1/2) (sum and difference
    of visible and sterile, exact by parity), 'drop' -> (Cvis, 1) as 2107.13568; vector: one run. pols: the relics'
    weightings, one block of three relic slots each (0: rho); n: the states transported.

    polarized (Dirac, thermal="full", 2026-10-21): a moving Dirac relic keeps its helicity, so the beam meets its
    left- and right-chiral components in shares (1 +- pi)/2 that depend on its speed and direction (notes; tests:
    after the average over the final azimuth the polarized relic is exactly that mixture of massless chiral relics, to
    m^2/s). The products come weighted by rho_L (relic slots 0-2) and rho_R (3-5). Vector: Majorana's tensors on
    rho_L alone (only the left-chiral component couples; at rest rho_L = rho/2, the earlier kappa). Scalar: the chiral
    tensors (dirac_scalar_chiral_tensors) on both; 'track' transports the visible (states 0-2) and the sterile (3-5)
    neutrinos together -- a polarized background breaks the parity that ran them as sum and difference -- and 'drop'
    the visible ones."""
    med = str(mediator).strip().lower()
    pol = bool(polarized) and not majorana
    if med == "scalar":
        if majorana:
            _, C, G2 = coupling_tensors(g_mass, phiphi, channels)
            return dict(prof=0, jt=SCALAR_JT, w=SCALAR_W, qe=SCALAR_QE, Ctau=C, runs=[(C, 1.0)], G2=G2, pols=(0,),
                        n=3)
        mode = str(dirac_sterile).strip().lower()
        if mode not in ("track", "drop"):
            raise ValueError(f"secret interactions: dirac_sterile must be 'track' or 'drop', got {dirac_sterile!r}")
        if pol:
            Lv, Li, Rv, Ri, G2 = dirac_scalar_chiral_tensors(g_mass, channels)
            if mode == "track":
                C = np.zeros((N_PROFILES, 6, 6, 6))
                for inc, (onL, onR) in enumerate((((Lv, Li), (Rv, Ri)), ((Ri, Rv), (Li, Lv)))):
                    # a visible incident (inc 0) on the left share (relic slots 0-2) and the right one (3-5): its
                    # visible (states 0-2) and sterile (3-5) secondaries; a sterile one (inc 1): the visible one's on
                    # the other share, visibility swapped
                    a = slice(3 * inc, 3 * inc + 3)
                    C[:, a, 0:3, 0:3], C[:, a, 0:3, 3:6] = onL
                    C[:, a, 3:6, 0:3], C[:, a, 3:6, 3:6] = onR
                C = np.ascontiguousarray(C)
                return dict(prof=0, jt=SCALAR_JT, w=SCALAR_W, qe=SCALAR_QE, Ctau=C, runs=[(C, 1.0)], G2=G2,
                            pols=(1, -1), n=6)
            Creg = np.ascontiguousarray(np.concatenate([Lv, Rv], axis=2))
            Ctau = np.ascontiguousarray(np.concatenate([Lv + Li, Rv + Ri], axis=2))
            return dict(prof=0, jt=SCALAR_JT, w=SCALAR_W, qe=SCALAR_QE, Ctau=Ctau, runs=[(Creg, 1.0)], G2=G2,
                        pols=(1, -1), n=3)
        Cv, Ci, G2 = dirac_scalar_tensors(g_mass, channels)
        Ctau = np.ascontiguousarray(Cv + Ci)
        if mode == "track":
            runs = [(Ctau, 0.5), (np.ascontiguousarray(Cv - Ci), 0.5)]
        else:
            runs = [(Cv, 1.0)]
        return dict(prof=0, jt=SCALAR_JT, w=SCALAR_W, qe=SCALAR_QE, Ctau=Ctau, runs=runs, G2=G2, pols=(0,), n=3)
    if med == "vector":
        C, G2 = vector_tensors(g_mass, majorana or pol, channels)
        return dict(prof=1, jt=VECTOR_JT, w=VECTOR_W, qe=VECTOR_QE, Ctau=C, runs=[(C, 1.0)], G2=G2,
                    pols=(1,) if pol else (0,), n=3)
    raise ValueError(f"secret interactions: mediator must be 'scalar' or 'vector', got {mediator!r}")


# ------------------------------------------------------------------------------------------------------------------
# Thermal relics in every collision term (2026-10-18: propagate_si thermal="full"; notes, "Relic motion in the
# collision transport"; docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md 6a, 6b). A relic of mass m at temperature T meets a
# neutrino of energy E at s = 2 E sigma, sigma = E_p (1 - v cos theta) distributed (with the flux factor) as
# rho(sigma); every at-rest quantity depends on the relic only through s, so each product of a rate kernel with a
# profile total, in-bin complement, drift or produced spectrum is its rho-average over at-rest relics of mass sigma.
# The at-rest products live on one lattice in mu = M^2 / s, ln mu_n = -n Delta, whose spectra depend on nothing else;
# rho enters through the weights of cubic interpolation between every s-th node, so a parent bin's products are sums
# of a few lattice products. The resonant kernels are not smooth on the lattice: s/BW and (s - M^2)/BW are the exact
# thermal averages (dp_thermal.relic_state_rates_analytic) and the interference's profiles enter as
# X(1) <(s - M^2)/BW> + lattice[(s - M^2)/BW (X(mu) - X(1))] (the s channel's profiles do not depend on mu).

@njit(cache=True)
def _sigma_density(sig, m, T, pol=0):
    """rho(sigma) of a relic of mass m at temperature T, normalized (tools/thermal_reference.rho):
    sigma / (3 zeta(3) T^3) int_{p_X}^inf p f(p/T) / E_p dp, X = max(sigma, m^2/sigma), p_X = (X^2 - m^2) / 2X; the
    inner integral by Gauss-Legendre on panels split at 2m, 20m, 200m and 2000m, then up to p_X + 4T and p_X + 44T.
    pol +1 (-1): the share of a Dirac relic's left- (right-) chiral component, rho_L (rho_R) = rho (1 +- pi) / 2 with
    pi = (E_p sigma - m^2) / (p sigma) (dp_thermal.relic_state_rates_analytic):
    rho_L = 1 / (3 zeta(3) T^3) int f (E_p + p) (sigma - E_p + p) / 2E_p dp, rho_R the same with p -> -p."""
    X = max(sig, m * m / sig)
    pX = (X * X - m * m) / (2.0 * X)
    edges = np.empty(7)
    edges[0] = pX
    ne = 1
    for f in (2.0, 20.0, 200.0, 2000.0):
        v = f * m
        if pX < v < pX + 4.0 * T:
            edges[ne] = v
            ne += 1
    edges[ne] = pX + 4.0 * T
    edges[ne + 1] = pX + 44.0 * T
    ne += 2
    acc = 0.0
    for e in range(ne - 1):
        a = edges[e]
        h = edges[e + 1] - a
        if h <= 0.0:
            continue
        for i in range(16):
            p = a + h * _GL16_X[i]
            x = p / T
            if x < 700.0:
                if pol == 0:
                    acc += h * _GL16_W[i] * p / (np.exp(x) + 1.0) / np.sqrt(p * p + m * m)
                else:
                    Ep = np.sqrt(p * p + m * m)
                    lo = m * m / (Ep + p)                  # E_p - p without cancellation
                    if pol > 0:
                        y = (Ep + p) * (sig - lo)
                    else:
                        y = lo * (Ep + p - sig)
                    acc += h * _GL16_W[i] * max(y, 0.0) / (2.0 * Ep) / (np.exp(x) + 1.0)
    if pol != 0:
        return acc / (3.0 * _ZETA3 * T * T * T)
    return sig * acc / (3.0 * _ZETA3 * T * T * T)


@njit(cache=True)
def _thermal_weights(m, T, c0, Delta, smax, sdiv, thr, W, pol=0):
    """One relic state and step: a parent at lattice coordinate nu = j q + c0 + ln(sigma) / Delta has the products
    sum_w W[w] G[j q + i0 + s w], G interpolated cubically (4-point Lagrange) between every s-th node and integrated
    against rho over its support (rho sigma above thr times its maximum); the stride s, 1 to smax nodes, is the
    support over sdiv. Fills W; returns (i0, s, number of weights), the number -1 if W is too short. pol +-1: the
    weights of a Dirac relic's chiral shares (_sigma_density) on rho's own support, so i0, s and the number are
    those of pol 0."""
    pm = 40.0 * T
    epm = np.sqrt(pm * pm + m * m)
    lo = np.log(m * m / (epm + pm))
    hi = np.log(epm + pm)
    ng = 192
    y = np.empty(ng)
    ymax = 0.0
    for g in range(ng):
        sg = np.exp(lo + (hi - lo) * g / (ng - 1))
        y[g] = _sigma_density(sg, m, T) * sg
        ymax = max(ymax, y[g])
    g0 = 0
    while g0 < ng - 1 and y[g0] <= thr * ymax:
        g0 += 1
    g1 = ng - 1
    while g1 > 0 and y[g1] <= thr * ymax:
        g1 -= 1
    a_lo = lo + (hi - lo) * max(g0 - 1, 0) / (ng - 1)
    a_hi = lo + (hi - lo) * min(g1 + 1, ng - 1) / (ng - 1)
    s = max(1, min(smax, int((a_hi - a_lo) / Delta / sdiv)))
    i0 = int(np.floor(c0 + a_lo / Delta)) - s
    k_lo = int(np.floor((c0 + a_lo / Delta - i0) / s))
    k_hi = int(np.floor((c0 + a_hi / Delta - i0) / s))
    nw = k_hi + 3
    if nw > len(W):
        return i0, s, -1
    for w in range(nw):
        W[w] = 0.0
    hmax = min(1.0, (a_hi - a_lo) / 12.0) / 8.0          # sub-panels: an eighth of rho's scale
    lm = np.log(m)
    for k in range(k_lo, k_hi + 1):
        x0 = max(a_lo, (i0 + s * k - c0) * Delta)
        x1 = min(a_hi, (i0 + s * (k + 1) - c0) * Delta)
        if x1 <= x0:
            continue
        nsub = int(np.ceil((x1 - x0) / hmax))
        h = (x1 - x0) / nsub
        for u in range(nsub):
            ua = x0 + h * u
            ub = x0 + h * (u + 1)
            if pol != 0 and ua < lm < ub:
                # split at sigma = m (2026-10-21): the chiral shares' densities have a kink there (rho a weaker one;
                # pol 0 unsplit, its arithmetic that of the campaign planes)
                for sgm in range(2):
                    pa = ua if sgm == 0 else lm
                    hh = (lm - ua) if sgm == 0 else (ub - lm)
                    for g in range(8):
                        x = pa + hh * _GL8_X[g]
                        sg = np.exp(x)
                        _lagrange_add(W, k, (c0 + x / Delta - i0) / s - k,
                                      hh * _GL8_W[g] * sg * _sigma_density(sg, m, T, pol))
                continue
            for g in range(8):
                x = x0 + h * (u + _GL8_X[g])
                sg = np.exp(x)
                _lagrange_add(W, k, (c0 + x / Delta - i0) / s - k, h * _GL8_W[g] * sg * _sigma_density(sg, m, T, pol))
    return i0, s, nw


@njit(cache=True)
def _lagrange_add(W, k, f, wt):
    """W[k-1..k+2] += wt times the cubic Lagrange basis at f (nodes -1, 0, 1, 2)."""
    W[k - 1] += wt * (-f * (f - 1.0) * (f - 2.0) / 6.0)
    W[k] += wt * ((f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0)
    W[k + 1] += wt * (-(f + 1.0) * f * (f - 2.0) / 2.0)
    W[k + 2] += wt * ((f + 1.0) * f * (f - 1.0) / 6.0)


@njit(cache=True, parallel=True)
def _thermal_weights_all(masses, Tm, c0, Delta, smax, sdiv, thr, live, W, i0, st, nw, pol=0):
    """_thermal_weights for every step m (temperatures Tm[m, r], offsets c0[m]) and live relic state r."""
    for t in prange(Tm.shape[0] * 3):
        m = t // 3
        r = t % 3
        if live[r]:
            a, b, c = _thermal_weights(masses[r], Tm[m, r], c0[m], Delta, smax, sdiv, thr, W[m, r], pol)
            i0[m, r] = a
            st[m, r] = b
            nw[m, r] = c


@njit(cache=True, parallel=True)
def _resonance_grid(lna, zp1, M, sumG_eq, masses, T_nu0, eta, tglx, tglw, tglx10, tglw10, tfd16, f0, f1, pol=0):
    """The exact thermal averages of s/BW and (s - M^2)/BW over 2 pi (dp_thermal.relic_state_rates_analytic, modes 0
    and 1, relics at 1+z = zp1; pol: a Dirac relic's chiral share) per relic state, at s-scale a = 2 E (1+z) =
    exp(lna[g])."""
    for g in prange(len(lna)):
        E0 = np.exp(lna[g]) / (2.0 * zp1)
        st0 = relic_state_rates_analytic(E0, zp1 - 1.0, sumG_eq, M, masses, T_nu0, eta, tglx, tglw, tglx10, tglw10,
                                         tfd16, True, 0, True, pol)
        st1 = relic_state_rates_analytic(E0, zp1 - 1.0, sumG_eq, M, masses, T_nu0, eta, tglx, tglw, tglx10, tglw10,
                                         tfd16, True, 1, True, pol)
        for r in range(3):
            f0[g, r] = st0[r] / (2.0 * np.pi)
            f1[g, r] = st1[r] / (2.0 * np.pi)


@njit(cache=True)
def _interp4(xg, yg, r, x, i):
    """4-point Lagrange interpolation of yg[:, r] (on the sorted xg) at x in cell i."""
    a = min(max(i - 1, 0), len(xg) - 4)
    acc = 0.0
    for p in range(4):
        lp = 1.0
        for k in range(4):
            if k != p:
                lp *= (x - xg[a + k]) / (xg[a + p] - xg[a + k])
        acc += lp * yg[a + p, r]
    return acc


@njit(cache=True, parallel=True)
def _resonance_step(E, L_lo, L_hi, L_mid, H0, Om, OL, lna, f0, f1, log0, JT0, JT1):
    """JT[j, r] = int_{L_lo}^{L_hi} dL f(ln 2E_j + L) e^{3 (L - L_mid)} / H(L) for both kernels (the grid's relic
    density is the step middle's): Gauss-Legendre (4 nodes) between the grid points, 4-point Lagrange of f1 and of f0
    (of its logarithm where log0)."""
    N = len(E)
    G = len(lna)
    for j in prange(N):
        b = np.log(2.0 * E[j])
        x1 = b + L_hi
        for r in range(3):
            JT0[j, r] = 0.0
            JT1[j, r] = 0.0
        a = b + L_lo
        i = min(max(np.searchsorted(lna, a) - 1, 0), G - 2)
        while a < x1:
            e = min(x1, lna[i + 1]) if i + 1 < G - 1 else x1
            h = e - a
            if h > 0.0:
                for g in range(4):
                    x = a + h * _GL4_X[g]
                    L = x - b
                    zp1 = np.exp(L)
                    wt = h * _GL4_W[g] * np.exp(3.0 * (L - L_mid)) / (H0 * np.sqrt(Om * zp1 ** 3 + OL))
                    for r in range(3):
                        v0 = _interp4(lna, f0, r, x, i)
                        JT0[j, r] += wt * (np.exp(v0) if log0[r] else v0)
                        JT1[j, r] += wt * _interp4(lna, f1, r, x, i)
            a = e
            i = min(i + 1, G - 2)


@njit(cache=True, parallel=True)
def _thermal_products(q, jt, W, i0, st, nw, n0, n_first, Jn, Nn, Ln, Dn, BT, JT0, JT1, Nab, Lab, Dab, R, offs,
                      tasks):
    """One step's absolute products (the sweep's inputs with J = 1): for parent j and relic state r the lattice sums
    sum_w W[r, w] n0[r] Jn[n, jt[q]] X[n, q] at n = j q + i0[r] + st[r] w of the profile totals Nn, in-bin complements
    Ln, drifts Dn and produced spectra BT[n, q, d] (d <= j; R[r, offs[j] + d, q]); the s channel (jt 0) and the
    interference's pole term from the exact thermal kernels JT0, JT1 times the profiles at mu = 1 (node 0).
    tasks[t] = (r, j_first, nk): the parents j_first + k st[r], k < nk, of one residue class, which share nodes
    (n = j_first q + i0 + st (k q + w)): each node's spectra are read once per task and added to every parent that
    uses them, in a buffer along d (the sums are bound by memory traffic otherwise)."""
    P = len(jt)
    one = -n_first
    for t in prange(tasks.shape[0]):
        r = tasks[t, 0]
        jf = tasks[t, 1]
        nk = tasks[t, 2]
        s = st[r]
        buf = np.zeros((nk, P, jf + (nk - 1) * s + 1))
        tot = np.zeros((nk, 3, P))
        sc = np.zeros((nk, P))
        for u in range((nk - 1) * q + nw[r]):
            n = jf * q + i0[r] + s * u - n_first
            for k in range(nk):
                w = u - k * q
                if w < 0:
                    break
                if w >= nw[r] or W[r, w] == 0.0:
                    continue
                j = jf + k * s
                for qq in range(P):
                    if jt[qq] == 0:
                        continue
                    cq = W[r, w] * n0[r] * Jn[n, jt[qq]]
                    tot[k, 0, qq] += cq * Nn[n, qq]
                    tot[k, 1, qq] += cq * Ln[n, qq]
                    tot[k, 2, qq] += cq * Dn[n, qq]
                    sc[k, qq] += cq
                    b = buf[k, qq]
                    src = BT[n, qq]
                    for d in range(j + 1):
                        b[d] += cq * src[d]
        for k in range(nk):
            j = jf + k * s
            for qq in range(P):
                if jt[qq] == 0:
                    a = JT0[j, r]
                elif jt[qq] == 1:
                    a = JT1[j, r] - sc[k, qq]
                else:
                    a = 0.0
                if a != 0.0:
                    tot[k, 0, qq] += a * Nn[one, qq]
                    tot[k, 1, qq] += a * Ln[one, qq]
                    tot[k, 2, qq] += a * Dn[one, qq]
                    b = buf[k, qq]
                    src = BT[one, qq]
                    for d in range(j + 1):
                        b[d] += a * src[d]
                Nab[j, r, qq] = tot[k, 0, qq]
                Lab[j, r, qq] = tot[k, 1, qq]
                Dab[j, r, qq] = tot[k, 2, qq]
            o = offs[j]
            for d in range(j + 1):
                for qq in range(P):
                    R[r, o + d, qq] = buf[k, qq, d]


def _thermal_tasks(N, st, nw, n0, BJ=16):
    """_thermal_products' tasks for one step: per live relic state and residue class of the parents modulo its stride,
    runs of BJ parents, the costliest first ((T, 3) int64)."""
    rows = []
    for r in range(3):
        if nw[r] <= 0 or n0[r] <= 0.0:
            continue
        s = int(st[r])
        for cls in range(s):
            js = np.arange(cls, N, s)
            for a in range(0, len(js), BJ):
                b = js[a:a + BJ]
                rows.append((int(nw[r]) * int((b + 1).sum()), r, int(b[0]), len(b)))
    rows.sort(reverse=True)
    return np.array([x[1:] for x in rows], dtype=np.int64).reshape(-1, 3)


_THERMAL_TABLE = {}
# numerical settings of thermal="full": the weights' largest stride (nodes) and the support's division into strides,
# rho's support (rho sigma above this fraction of its maximum), the resonant kernels' grid step (in Delta)
THERMAL_STRIDE_MAX = 8
THERMAL_STRIDE_DIV = 16.0
THERMAL_SUPPORT = 1e-9
THERMAL_GRID_STEP = 1.0
# the resonant kernels' grid: neighbours at least this fraction of its finest spacing apart (_thin_grid; 2026-10-21)
THERMAL_GRID_MIN_GAP = 0.5


def _thin_grid(x, gap):
    """Sorted nodes x thinned so that neighbours are at least gap apart (the first and the last kept). The kernels'
    grid merges a regular grid with finer ones around slow relics' resonances, which puts nodes arbitrarily close
    together; deep in the thermal tail the kernel's quadrature is good to ~1%, and 4-point Lagrange (of its
    logarithm) through two nodes eps apart overshoots by ~1% x spacing / eps -- e^66 at eps ~ 1e-6, infinity below
    ~1e-7 (2026-10-21: a crash at 453 keV; isolated bins absorbed outright otherwise). gap <= 0 keeps x."""
    if gap <= 0.0 or len(x) < 3:
        return x
    keep = [x[0]]
    for v in x[1:-1]:
        if v - keep[-1] >= gap:
            keep.append(v)
    if x[-1] - keep[-1] < gap and len(keep) > 1:
        keep[-1] = x[-1]
    else:
        keep.append(x[-1])
    return np.array(keep)


def _thermal_table(dlnE, Delta, N, n_u, n_lo, n_hi, prof=1):
    """The model's produced spectra (prof 0: the scalar's 7 profiles, 1: the vector's 10), full rows d = 0..N-1, on
    the lattice mu_n = exp(-n Delta) for n_lo <= n <= n_hi (they depend on mu, the bin width and N only): (n_first,
    BT (nodes, P, N): profile-major, so that the products' sums run along d), kept per grid and model and widened as
    needed."""
    key = (float(dlnE), float(Delta), int(N), int(n_u), int(prof))
    hit = _THERMAL_TABLE.get(key)
    if hit is not None:
        if hit[0] <= n_lo and hit[0] + hit[1].shape[0] - 1 >= n_hi:
            return hit
        n_lo = min(n_lo, hit[0])
        n_hi = max(n_hi, hit[0] + hit[1].shape[0] - 1)
    n_lo -= 32
    n_hi += 32
    nodes = np.arange(n_lo, n_hi + 1)
    build = build_profile_table_tri if prof == 0 else build_vector_table_tri
    Bt, _ = build(np.ascontiguousarray(np.exp(-nodes * Delta)), np.full(len(nodes), N - 1, dtype=np.int64), dlnE, n_u)
    out = (int(n_lo), np.ascontiguousarray(Bt.reshape(len(nodes), N, Bt.shape[1]).transpose(0, 2, 1)))
    _THERMAL_TABLE.clear()
    _THERMAL_TABLE[key] = out
    return out


def _thermal_terms(E, L_edges, q, Delta, dlnE, masses, n0, T_nu0, eta, M, MG, H0, Om, OL, hbin, n_u, prof=1,
                   pols=(0,)):
    """Everything _thermal_products needs but the bins' rows: per step m the relic states' weights (W, i0, st, nw),
    the at-rest kernels on the lattice (Jn) and the exact thermal resonant kernels per bin (JT0, JT1); the lattice's
    profile totals (Nn, Ln, Dn) and produced spectra (n_first, BT). pols: the relics' weightings (0: rho; +1, -1: a
    Dirac relic's left- and right-chiral shares), one set of W (K, 3, 4096), JT0 and JT1 (K, N, 3) each, stacked
    along a first axis; i0, st and nw are common to them."""
    N = len(E)
    K = len(L_edges) - 1
    M2 = M * M
    L_mid = 0.5 * (L_edges[:-1] + L_edges[1:])
    eta = np.asarray(eta, dtype=np.float64)
    live = (masses > 0.0) & (n0 > 0.0)
    Tm = np.ascontiguousarray(T_nu0 * np.exp(L_mid)[:, None] * np.cbrt(eta)[None, :])
    c0 = (np.log(2.0 * E[0]) + L_mid - np.log(M2)) / Delta
    npol = len(pols)
    W = np.zeros((npol, K, 3, 4096))
    i0 = np.zeros((K, 3), dtype=np.int64)
    st = np.ones((K, 3), dtype=np.int64)
    nw = np.zeros((K, 3), dtype=np.int64)
    for k, pl in enumerate(pols):
        _thermal_weights_all(masses, Tm, c0, Delta, THERMAL_STRIDE_MAX, THERMAL_STRIDE_DIV, THERMAL_SUPPORT, live,
                             W[k], i0, st, nw, int(pl))
    if np.any(nw[:, live] < 0):
        raise ValueError("thermal products: a relic's distribution is wider than the weights' storage")
    sel = np.zeros((K, 3), dtype=np.bool_)
    sel[:, live] = True
    n_lo = min(0, int(i0[sel].min())) if sel.any() else 0
    n_hi = max(0, (N - 1) * q + int((i0 + st * (nw - 1))[sel].max())) if sel.any() else 0
    n_first, BT = _thermal_table(dlnE, Delta, N, n_u, n_lo, n_hi, prof)
    nodes = n_first + np.arange(BT.shape[0])
    nn = len(nodes)
    P = BT.shape[1]
    one = np.array([1.0, 0.0, 0.0])
    Jf = np.zeros((nn, 3, 4))
    Nf, Lf, Df = (np.zeros((nn, 3, P)) for _ in range(3))
    Jn = np.zeros((K, nn, 4))
    JT0 = np.zeros((npol, K, N, 3))
    JT1 = np.zeros((npol, K, N, 3))
    sumG_eq = 12.0 * np.pi * MG / M2
    for m in range(K):
        zlo, zhi, zmid = np.exp(L_edges[m]), np.exp(L_edges[m + 1]), np.exp(L_mid[m])
        # the lattice's at-rest kernels: a relic of 1 eV at the step's energy that puts the step's middle at mu_n
        _step_integrals(M2 * np.exp(nodes * Delta) / (2.0 * zmid), zlo, zhi, zmid, one, one, M2, MG, H0, Om, OL,
                        False, prof, hbin, Jf, Nf, Lf, Df)
        Jn[m] = Jf[:, 0, :]
        # the resonant kernels' grid in ln a, a = 2 E (1+z): every THERMAL_GRID_STEP Delta, finer (a sixth of the
        # speed, ten speeds around the resonance) where a slow relic state's distribution is narrower than eight steps
        # (every 2 Delta, a quarter, eight: 6e-4 at 4.2 MeV, for 5% of the time)
        x_lo = np.log(2.0 * E[0]) + L_edges[m] - 3.0 * Delta
        x_hi = np.log(2.0 * E[-1]) + L_edges[m + 1] + 3.0 * Delta
        pts = [np.arange(x_lo, x_hi + Delta, THERMAL_GRID_STEP * Delta)]
        h_min = THERMAL_GRID_STEP * Delta
        for r in range(3):
            if not live[r]:
                continue
            pt = 3.15 * Tm[m, r]
            v = pt / np.hypot(masses[r], pt)
            if 2.0 * v < 8.0 * Delta:
                cen = np.log(M2 / masses[r])
                pts.append(np.arange(max(cen - np.log1p(10.0 * v), x_lo), min(cen - np.log1p(-min(10.0 * v, 0.9)), x_hi),
                                     v / 6.0))
                h_min = min(h_min, v / 6.0)
        lna = _thin_grid(np.unique(np.concatenate(pts)), THERMAL_GRID_MIN_GAP * h_min)
        for k, pl in enumerate(pols):
            f0 = np.zeros((len(lna), 3))
            f1 = np.zeros((len(lna), 3))
            _resonance_grid(lna, zmid, M, sumG_eq, masses, T_nu0, eta, _TGL16_X, _TGL16_W, _TGL10_X, _TGL10_W,
                            _TFD16, f0, f1, int(pl))
            log0 = np.all(f0 > 0.0, axis=0)
            f0[:, log0] = np.log(f0[:, log0])
            _resonance_step(E, L_edges[m], L_edges[m + 1], L_mid[m], H0, Om, OL, lna, f0, f1, log0, JT0[k, m],
                            JT1[k, m])
    tasks = [_thermal_tasks(N, st[m], nw[m], n0) for m in range(K)]
    return dict(W=W, i0=i0, st=st, nw=nw, tasks=tasks, n_first=n_first, BT=BT, Nn=np.ascontiguousarray(Nf[:, 0, :]),
                Ln=np.ascontiguousarray(Lf[:, 0, :]), Dn=np.ascontiguousarray(Df[:, 0, :]), Jn=Jn, JT0=JT0, JT1=JT1)


_STEP_TERMS = {}                        # one physics point's step terms (propagate_si)
_STEP_TERMS_MAX_BYTES = 1.5e9


def _step_key(E, L_edges, masses, n0, M2, MG, H0, Om, OL, delta_mode, prof, hbin, thermal, M, T_nu0, eta, jt):
    """Everything the steps' collision terms depend on (_step_integrals, _thermal_resonance_terms)."""
    import hashlib
    h = hashlib.sha256()
    for a in (E, L_edges, masses, n0, eta, jt):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((float(M2), float(MG), float(H0), float(Om), float(OL), bool(delta_mode), int(prof), float(hbin),
                   thermal if thermal == "full" else bool(thermal), float(M), float(T_nu0))).encode())
    return h.hexdigest()


def propagate_si(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_src,
                 p_index, rho_evol, Phi_0, M, g_mass, majorana=True, channels="full",
                 phiphi="coherent", z_substeps=1, E_extension=10.0, n_u=4096,
                 return_diagnostics=False, regeneration=True, mediator="scalar", dirac_sterile="track",
                 Gamma_vis=0.0, thermal=False, inj_window=(0.0, np.inf), decay=None, relic_loss=None):
    """Cosmological transport of the astrophysical neutrinos under secret interactions.

    Arguments: E_bins -- the (geometric) energy grid of the result, eV; z_max, H0 (eV),
    Omega_m, Omega_L -- sources up to z_max in flat LCDM; U_pmns -- (3, 3) mixing matrix;
    masses -- (3,) neutrino masses, eV; T_nu0 -- relic temperature today, eV; eta_mass -- (3,)
    relic density per mass state relative to the standard one; rho_src -- (3,) source flavor
    composition; p_index, rho_evol, Phi_0 -- source spectrum E^-p, evolution R(z) (1+z)^rho_evol
    (dp_cosmo.calc_R_source), normalization (inj_window: (E_min, E_max) in eV, 2026-10-15: the sources emit
    only where the energy today, E_source / (1+z), lies in [E_min, E_max) -- one piece of a broken power law);
    M -- mediator mass, eV; g_mass -- (3, 3) mass-basis
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
    papers and the secret-interaction planes; "full" (vector 2026-10-18, scalar 2026-10-21) averages every collision
    term -- rates, in-bin terms and produced spectra -- over the relics' thermal momenta (_thermal_products):
    Majorana relics are unpolarized at rest and in motion; a Dirac relic keeps its helicity, and the beam meets its
    left- and right-chiral components in shares that depend on its speed and direction (Dirac 2026-10-21:
    model_spec, polarized).

    Returns the (N_E, 3) mass-state fluxes at z = 0 of the visible neutrinos (both helicities), on
    E_bins; with return_diagnostics also a dict of the number and energy bookkeeping (of the first
    run: for Dirac 'track', visible plus sterile).

    The transport is linear in the source (2026-10-18): the result is Phi_0 sum_j P_j T_j, P = |U|^2^T rho_src the
    sources' mass-state composition and T_j the transport of a unit source in mass state j (Phi_0 = 1) with this
    spectral index and emission window. The three T_j are computed together (one sweep, _si_sweep_cols) and kept for
    the same physics (everything but the source: _COLUMNS), so the other source compositions of a point -- and, with
    batch_hint, its other spectral-index nodes and emission windows -- reuse one collision operator per bin and
    step. With return_diagnostics the source is transported as given (one column), its bookkeeping returned."""
    E_bins = np.asarray(E_bins, dtype=np.float64)
    g_arr = np.ascontiguousarray(g_mass, dtype=np.complex128)
    # the 2026-10-24 processes only when set (a substituted _transport_columns, as the tests' reference, need not
    # know them)
    extra = {k: v for k, v in (("decay", decay), ("relic_loss", relic_loss)) if v is not None}
    if return_diagnostics or not np.any(g_arr):
        P_src = (np.abs(np.asarray(U_pmns)) ** 2).T @ np.asarray(rho_src, dtype=np.float64)
        out = _transport_columns(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_evol, M,
                                 g_arr, majorana, channels, phiphi, z_substeps, E_extension, n_u, regeneration,
                                 mediator, dirac_sterile, Gamma_vis, thermal,
                                 [(P_src, float(p_index), (float(inj_window[0]), float(inj_window[1])), float(Phi_0),
                                   np.asarray(rho_src, dtype=np.float64))], return_diagnostics, **extra)
        return out[0]
    phys = _physics_key(E_bins, z_max, H0, Omega_m, Omega_L, masses, T_nu0, eta_mass, rho_evol, M, g_arr, majorana,
                        channels, phiphi, z_substeps, E_extension, n_u, regeneration, mediator, dirac_sterile,
                        Gamma_vis, thermal, decay, relic_loss)
    col = _column_key(p_index, inj_window)
    T = _COLUMNS.get((phys, col))
    if T is None:
        want = [col] + [c for c in dict.fromkeys(_BATCH_HINT) if c != col and (phys, c) not in _COLUMNS]
        unit = np.eye(3)
        specs = [(unit[j], c[0], (c[1], c[2]), 1.0, None) for c in want for j in range(3)]
        Fs = _transport_columns(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_evol, M,
                                g_arr, majorana, channels, phiphi, z_substeps, E_extension, n_u, regeneration,
                                mediator, dirac_sterile, Gamma_vis, thermal, specs, False, **extra)
        for i, c in enumerate(want):
            _COLUMNS[(phys, c)] = np.stack(Fs[3 * i:3 * i + 3])          # (3 unit sources, N_E, 3)
        while len(_COLUMNS) > _COLUMNS_MAX:
            _COLUMNS.popitem(last=False)
        T = _COLUMNS[(phys, col)]
    else:
        _COLUMNS.move_to_end((phys, col))
    P_src = (np.abs(np.asarray(U_pmns)) ** 2).T @ np.asarray(rho_src, dtype=np.float64)
    return np.ascontiguousarray(float(Phi_0) * (P_src[0] * T[0] + P_src[1] * T[1] + P_src[2] * T[2]))


# The transports of the unit mass-state sources, per physics and (spectral index, emission window) (propagate_si):
# {(physics key, column key): (3, N_E, 3)}, least recently used first; and the columns a transport computes along
# with its own (batch_hint).
_COLUMNS = OrderedDict()
_COLUMNS_MAX = 256
_BATCH_HINT = []


def _column_key(p_index, inj_window):
    return (float(p_index), float(inj_window[0]), float(inj_window[1]))


def _physics_key(E_bins, z_max, H0, Omega_m, Omega_L, masses, T_nu0, eta_mass, rho_evol, M, g_mass, majorana,
                 channels, phiphi, z_substeps, E_extension, n_u, regeneration, mediator, dirac_sterile, Gamma_vis,
                 thermal, decay=None, relic_loss=None):
    """Everything a unit-source transport depends on (propagate_si's arguments but the source's composition,
    spectral index, normalization and emission window)."""
    import hashlib
    h = hashlib.sha256()
    for a in (E_bins, masses, eta_mass, g_mass):
        h.update(np.ascontiguousarray(a).tobytes())
    if decay is not None:
        h.update(b"decay")
        for a in decay:
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    if relic_loss is not None:
        h.update(b"relic_loss")
        for a in relic_loss:
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(repr((float(z_max), float(H0), float(Omega_m), float(Omega_L), float(T_nu0), float(rho_evol), float(M),
                   bool(majorana), str(channels), str(phiphi), int(z_substeps), float(E_extension), int(n_u),
                   bool(regeneration), str(mediator), str(dirac_sterile), float(Gamma_vis),
                   thermal if thermal == "full" else bool(thermal))).encode())
    return h.hexdigest()


class batch_hint:
    """Within this context, a transport of propagate_si also computes these (spectral index, emission window (eV))
    columns of the same physics, kept for the calls that follow (a point's other spectral-index nodes and broken
    power-law pieces; analysis.model sets it per evaluation). Nothing else changes: each column's arithmetic does not
    depend on the batch."""

    def __init__(self, columns):
        self.columns = [_column_key(p, w) for p, w in columns]

    def __enter__(self):
        global _BATCH_HINT
        self._old = _BATCH_HINT
        _BATCH_HINT = list(self.columns)
        return self

    def __exit__(self, *exc):
        global _BATCH_HINT
        _BATCH_HINT = self._old


def _decay_columns(Fs, E, L_lo, L_hi, dt, decay, masses, majorana):
    """Neutrino decay into the mediator over one step (dp_decay, 2026-10-24), after the step's collisions: the
    visible states of every column, one Majorana flux holding both kinds with equal populations (the flipped
    daughters into it too: dp_decay's fold)."""
    import dp_decay
    if not majorana:
        raise NotImplementedError("neutrino decay in dp_si: Majorana neutrinos only")
    zp1 = float(np.exp(0.5 * (L_lo + L_hi)))
    none = np.zeros((1, 3, 3))
    for Fk in Fs:
        for c in range(Fk.shape[2]):
            F = Fk[:, :3, c]
            dp_decay.decay_step(E, zp1, dt, decay[0], decay[1], masses, F, F, none, none, False, True)


def _loss_columns(Fs, E, L_lo, L_hi, dt, relic_loss):
    """The beam's energy loss in relic mass changes over one step (dp_relic_loss, 2026-10-24): relic_loss = (z nodes,
    C(z) on them, (G'G'^dag)_aa per mass state); each visible state's content shifts down by C (G'G'^dag)_aa dt/(1+z)."""
    import dp_relic_loss
    z_nodes, C_nodes, beam = relic_loss
    zp1 = float(np.exp(0.5 * (L_lo + L_hi)))
    o = np.argsort(z_nodes)
    C = float(np.interp(zp1 - 1.0, np.asarray(z_nodes)[o], np.asarray(C_nodes)[o]))
    dE = np.ascontiguousarray(np.broadcast_to(C * np.asarray(beam, dtype=np.float64) * dt / zp1, (len(E), 3)))
    for Fk in Fs:
        for c in range(Fk.shape[2]):
            dp_relic_loss.shift_step(E, Fk[:, :3, c], dE)


def _transport_columns(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_evol, M, g_mass,
                       majorana, channels, phiphi, z_substeps, E_extension, n_u, regeneration, mediator,
                       dirac_sterile, Gamma_vis, thermal, columns, return_diagnostics=False, decay=None,
                       relic_loss=None):
    """The transport (propagate_si's physics) of several sources at once: columns is a list of (P_src (3,) the
    mass-state composition, p_index, inj_window (eV), Phi_0, rho_src (3,) or None -- the flavor composition, used only
    without a coupling, by the free-streaming closed form). Returns the list of (N_E, 3) fluxes, or with
    return_diagnostics the list of (flux, info) pairs (propagate_si's)."""
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
        outs = []
        for P_c, p_c, w_c, phi_c, rho_c in columns:
            if rho_c is None:
                # a mass-state source: the closed form with U = 1 transports the composition P_c as given
                o = free_streaming_flux(E_bins, z_max, H0, Omega_m, Omega_L, np.eye(3), P_c, p_c, rho_evol, phi_c,
                                        inj_window=w_c)
            else:
                o = free_streaming_flux(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, rho_c, p_c, rho_evol, phi_c,
                                        inj_window=w_c)
            outs.append((o, dict(no_coupling=True)) if return_diagnostics else o)
        return outs
    M2 = M * M
    Gam = mediator_width(M, g_mass, mediator, majorana) + float(Gamma_vis)
    MG = M * Gam
    full = thermal == "full"
    # thermal="full" with Dirac neutrinos: the relics' chiral shares as separate relic slots (model_spec, polarized)
    spec = model_spec(g_mass, mediator, majorana, phiphi, channels, dirac_sterile, polarized=full)
    pols = spec["pols"]
    n_st = spec["n"]
    n_rel = 3 * len(pols)
    if Gamma_vis > 0.0:
        if spec["prof"] != 1:
            raise ValueError("secret interactions: a visible width is implemented for the vector only")
        # the s channel's events into visible pairs: absorbed, no secondaries. In the tensors' units the final-state
        # sum Tr(G G^dag) stands for Gamma_nu = M Tr / 24 pi, so the visible width adds 24 pi Gamma_vis / M,
        # spread over the three secondary slots of the absorption tensor only (Creg keeps the neutrino channels);
        # Dirac at rest: half (kappa), polarized: on rho_L, which carries that half
        X = 24.0 * np.pi * float(Gamma_vis) / M
        kap = 1.0 if (majorana or pols != (0,)) else 0.5
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

    if full and delta_mode:
        raise ValueError("secret interactions: thermal='full' is implemented for the full channels")
    if not full:
        (lat_B, lat_O), top = _profile_tables(prof, M, masses, E[0], N, dlnE, q, K, top_full, L_top_mid, n_u)

    C = len(columns)
    runs = spec["runs"]
    Fs = [np.zeros((N, n_st, C)) for _ in runs]
    # the decay into the mediator (dp_decay, 2026-10-24), inside the sweep: per state the rest rate x mass and the
    # branching to each daughter state (the visible states; Majorana only)
    use_dec = decay is not None
    dec_rate = np.zeros(n_st)
    dec_br = np.zeros((n_st, n_st))
    if use_dec:
        if not majorana:
            raise NotImplementedError("neutrino decay in dp_si: Majorana neutrinos only")
        Gam_d = np.asarray(decay[0], dtype=np.float64)
        tot_d = Gam_d.sum(axis=1)
        for a_ in range(3):
            if tot_d[a_] > 0.0:
                dec_rate[a_] = tot_d[a_] * masses[a_]
                dec_br[a_, :3] = Gam_d[a_] / tot_d[a_]
    diags = [np.zeros((C, 6)) for _ in runs]
    reg = np.zeros((N, n_st, C))
    J = np.zeros((N, 3, 4))
    Nq = np.zeros((N, 3, P))
    Lq = np.zeros((N, 3, P))
    Dq = np.zeros((N, 3, P))
    node = np.zeros((N, 3), dtype=np.int64)
    jj = np.arange(N, dtype=np.int64)
    Eps = []
    for P_c, p_c, w_c, phi_c, rho_c in columns:
        Ep = E ** (-p_c)
        if w_c[0] > 0.0 or np.isfinite(w_c[1]):
            # the sources' emission window in today's energy (2026-10-15): one piece of a broken power law
            Ep = np.where((E >= w_c[0]) & (E < w_c[1]), Ep, 0.0)
        Eps.append(Ep)
    hbin = float(np.exp(0.5 * dlnE))
    Cregs = [np.ascontiguousarray(Cr if regeneration else np.zeros_like(Cr)) for Cr, _ in runs]
    # the steps' collision terms do not depend on the source (spectral index, evolution, composition): kept from one
    # call to the next with the same physics (2026-10-14: an analysis runs every spectral-index node of a point in a
    # row, and the terms were half of each run)
    skey = _step_key(E, L_edges, masses, n0, M2, MG, H0, Omega_m, Omega_L, delta_mode, prof, hbin, thermal, M,
                     T_nu0, eta_arr, spec["jt"])
    if full:
        # thermal relics in every collision term (2026-10-18): the steps' products from the lattice
        # (_thermal_products), fed to the sweep as absolute products with J = 1 and per-bin rows; one block of relic
        # slots per weighting (Dirac: rho_L, rho_R; 2026-10-21)
        skey = skey + "/full/%d/%s" % (n_u, pols)
        th = _STEP_TERMS.get(skey)
        if th is None:
            th = _thermal_terms(E, L_edges, q, Delta, dlnE, masses, n0, T_nu0, eta_arr, M, MG, H0, Omega_m, Omega_L,
                                hbin, n_u, prof, pols)
            _STEP_TERMS.clear()
            _STEP_TERMS[skey] = th
        offs = np.zeros(N + 1, dtype=np.int64)
        offs[1:] = np.cumsum(jj + 1)
        blocks = [(np.zeros((N, 3, P)), np.zeros((N, 3, P)), np.zeros((N, 3, P)), np.zeros((3, int(offs[-1]), P)))
                  for _ in pols]
        Oth = np.ascontiguousarray(np.tile(offs, (n_rel, 1)))
        node_th = np.ascontiguousarray(np.tile(jj[:, None], (1, n_rel)))
        Jone = np.ones((N, n_rel, 4))
    steps = None if full else _STEP_TERMS.get(skey)
    store = None
    if not full and steps is None and K * J.nbytes + 3 * K * Nq.nbytes <= _STEP_TERMS_MAX_BYTES:
        store = []
    S0 = np.zeros((N, n_st, C))
    for m in range(K - 1, -1, -1):
        L_lo, L_hi = L_edges[m], L_edges[m + 1]
        for k, (P_c, p_c, w_c, phi_c, rho_c) in enumerate(columns):
            dt, T_inj = _step_time_and_injection(L_lo, L_hi, H0, Omega_m, Omega_L, p_c, rho_evol, phi_c)
            S0[:, :3, k] = np.outer(Eps[k] * (T_inj / dt), np.asarray(P_c, dtype=np.float64))
        if full:
            chunk = numba.set_parallel_chunksize(1)
            for b, (Nb, Lb, Db, Rb) in enumerate(blocks):
                _thermal_products(q, spec["jt"], th["W"][b][m], th["i0"][m], th["st"][m], th["nw"][m], n0,
                                  th["n_first"], th["Jn"][m], th["Nn"], th["Ln"], th["Dn"], th["BT"], th["JT0"][b][m],
                                  th["JT1"][b][m], Nb, Lb, Db, Rb, offs, th["tasks"][m])
            numba.set_parallel_chunksize(chunk)
            if len(blocks) == 1:
                Nq, Lq, Dq, Rth = blocks[0]
            else:
                Nq, Lq, Dq = (np.ascontiguousarray(np.concatenate([bl[i] for bl in blocks], axis=1)) for i in range(3))
                Rth = np.ascontiguousarray(np.concatenate([bl[3] for bl in blocks], axis=0))
            for k in range(len(runs)):
                _si_sweep_cols(Fs[k], E, dE, S0, Jone, Nq, Lq, Dq, spec["jt"], spec["w"], spec["qe"], spec["Ctau"],
                               Cregs[k], Rth, Oth, node_th, dt, reg, diags[k], prof == 1, dec_rate, dec_br,
                               float(np.exp(0.5 * (L_lo + L_hi))), use_dec)
            if relic_loss is not None:
                _loss_columns(Fs, E, L_lo, L_hi, dt, relic_loss)
            continue
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
            _si_sweep_cols(Fs[k], E, dE, S0, J, Nq, Lq, Dq, spec["jt"], spec["w"], spec["qe"], spec["Ctau"],
                           Cregs[k], tB, tO, node, dt, reg, diags[k], prof == 1, dec_rate, dec_br,
                           float(np.exp(0.5 * (L_lo + L_hi))), use_dec)
        if relic_loss is not None:
            _loss_columns(Fs, E, L_lo, L_hi, dt, relic_loss)
    if store is not None:
        _STEP_TERMS.clear()
        _STEP_TERMS[skey] = store
    F_all = sum(wt * Fk for (_, wt), Fk in zip(runs, Fs))
    outs = []
    for k in range(C):
        F = np.ascontiguousarray(F_all[:, :3, k])          # the visible neutrinos (states 3-5: sterile ones)
        out = np.ascontiguousarray(F[:N_E])
        if not return_diagnostics:
            outs.append(out)
            continue
        diag = diags[0][k]
        F0 = Fs[0][:, :, k]
        info = dict(absorbed=diag[0], secondaries_in_grid=diag[1], secondaries_expected=diag[2],
                    energy_absorbed=diag[3], energy_secondaries_in_grid=diag[4], injected=diag[5],
                    final_number=float(np.sum(F0 * dE[:, None])),
                    final_number_in_E_bins=float(np.sum(F0[:N_E] * dE[:N_E, None])),
                    final_energy=float(np.sum(F0 * (E * dE)[:, None])), width=Gam, n_steps=K, N_grid=N,
                    top_full=top_full, F_extended=F.copy(), E_extended=E.copy(),
                    visible_number=float(np.sum(F * dE[:, None])))
        outs.append((out, info))
    return outs


def free_streaming_flux(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, rho_src, p_index, rho_evol, Phi_0,
                        n=4000, inj_window=(0.0, np.inf)):
    """The mass-state fluxes without interactions, (N_E, 3): E^-p P_a int_0^z_max Phi_0 (1+z)^-p
    R(z) / H(z) dz (the standard result: every emitted neutrino arrives), by Gauss-Legendre on n
    nodes in ln(1+z) split at the kinks of R(z) -- the reference for the no-coupling limit of
    propagate_si; zero outside the emission window inj_window (eV, in today's energy)."""
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
    E = np.asarray(E_bins, dtype=np.float64)
    spec = E ** (-p_index)
    if inj_window[0] > 0.0 or np.isfinite(inj_window[1]):
        spec = np.where((E >= inj_window[0]) & (E < inj_window[1]), spec, 0.0)
    return np.outer(spec * T, P_src)
