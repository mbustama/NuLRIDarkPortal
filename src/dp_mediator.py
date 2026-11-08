"""
dp_mediator.py -- the mediator made in flight by the beam's decays nu_i -> nu_j + Z_D, and its decay back into neutrino
pairs, Z_D -> nu_k nu_l for m_Zd > m_k + m_l, before the beam reaches the Earth (decision B: Mauricio, 2026-11-07;
docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md item 22 (f); tests/test_mediator.py checks the formulas below against the
traces).

Width. The Majorana vertex gamma^mu (G'_kl P_L - G'*_kl P_R), G' = U^dag G U the WHOLE coupling in the mass basis: its
identity part (a flavor-universal coupling, the Z mixing's kappa) decays the mediator into nu_k nu_k, though it changes
no mass state in nu_i -> nu_j + Z_D (dp_decay.rest_rates takes it out). Summed over spins and polarizations,
    sum |M|^2 = 2 |G'_kl|^2 [2 M^2 - m_k^2 - m_l^2 - (m_k^2 - m_l^2)^2 / M^2] - 12 m_k m_l Re(G'_kl^2),
    Gamma(Z -> nu_k nu_l) = S p sum|M|^2 / (24 pi M^2),  p = lambda^(1/2)(M^2, m_k^2, m_l^2) / (2 M),
S = 1/2 for k = l (identical particles), 1 for k < l (each unordered pair once). Massless daughters: Tr(G G^dag) M /
24 pi in all (the absorption's width, dp_absorption); k = l: G'_kk^2 M beta^3 / 24 pi (the axial coupling's P wave).
Dirac (the P_L piece alone, Z -> nu_k nubar_l for every ordered pair): |G'_kl|^2 [2 M^2 - m_k^2 - m_l^2 - (m_k^2 -
m_l^2)^2 / M^2] p / (24 pi M^2). Each light-front helicity of the mediator decays at a third of the summed rate
(rotational invariance; checked to 1e-15), so the three populations below decay alike, at Gamma_Z M / E in the lab.
A visible width (e+e-, above 1 MeV) lowers the neutrino branching; its decays leave the beam.

Light-front helicities. Every particle of the beam is ultra-relativistic: its state is its light-front (LF) helicity
along the beam, its energy share its LF momentum fraction (as dp_decay.daughter_spectra; a fermion's h = -1 is the
beam's neutrino, F_nu, +1 its antineutrino, F_nb). The mediator is kept as three populations, LF helicity 0
(longitudinal), +1 and -1, with polarization vectors eps_0 = q/M - M n/q^+ and eps_+- = (e.q_T/q^+, e_+-, -e.q_T/q^+),
e_+- = -+(1, +-i)/sqrt 2 (they sum to -g + q q / M^2).
  Production: in nu_i -> nu_j + Z_D the mediator takes y = 1 - z of the parent's energy (z the daughter's share). Its
LF helicity depends on z: the traces with eps_lam eps_lam* in place of the polarization sum, which add up to
daughter_spectra's. E.g. m_i = 0.05, M = 0.01 eV, a kept daughter at z = 0.48 leaves a longitudinal mediator 66% of the
time, at z = 0.91 1%; rest_rates' f_L (0.93, the rest-frame helicity) is not the share that sets the decay in flight.
A neutrino parent emits LF helicity -1 rather than +1, an antineutrino the mirror. At the collinear ends (k_T = 0)
the angular momentum along the beam fixes it: a kept daughter leaves lam = 0, a flipped one lam = h (checked).
  Decay: a mediator of LF helicity lam gives nu_k (LF helicity h1, share x) and nu_l (h2, share 1 - x). Massless
daughters come in (h1, h2) = (-1, +1) or (+1, -1) only, the neutrino's share 6 x (1 - x) (lam = 0), 3 (1 - x)^2 (+1),
3 x^2 (-1): the d^1_{lam,-1} distributions (checked). Massive daughters also give same-helicity pairs (M = 0.01,
m_l = 8.7e-3 eV: a lam = +1 mediator gives two antineutrinos 49% of the time); at the collinear ends
lam = (h1 + h2) / 2 (checked).
  Each density is computed exactly (Density, below): |M|^2 times t (1-t) (decay) or t (1-t)^2 (production), t the
product's share, is a polynomial in c = cos th* of degree <= 5, kept as a Chebyshev series fitted to the traces on
[-1, 1] (well conditioned however narrow the window: near a threshold it shrinks to a point), and the cumulative number
is integrated in ln(t / (1-t)) by Gauss-Legendre panels (exact to rounding; the light-front 1/t, 1/(1-t) of a nearly
massless product cancel there). No closed-form fit, so nothing fails at the thresholds (2026-11-07, the review's
finding 1: the earlier fits raised within ~1e-4 of them, and the analysis' objective made such a point a wall).

Numerics (vacuum kinematics). Tables per run on the energy grid's offsets (a product k bins below its parent's bin,
the parent spread uniformly over its bin as dp_decay.window_plan): production[ch, lam, k], the mediator per decay
through dp_medium's ladder channel ch; decay[lam, kind, state, k], the neutrinos per decayed mediator (two x the
neutrino branching). Number: each table's offsets 0..N+1 sum to exactly one mediator (two neutrinos x branching)
per decay, what falls below every bin kept in the last offset, N + 1 (telescoping cumulative integrals). Energy,
counted at the bins' centres: each row's (a production channel's mediators of one LF helicity; a decay's neutrinos of
one kind and mass state) number is its exact share and its energy is set exactly (energy_fix: a tilt of order
(ln h)^2 where that changes no entry by more than half, else a share moved into the next bin, the linear rule) -- the
decay rows to their exact continuous energies (which add up to the mediator's times the branching), a production
channel's rows to theirs scaled to add up to the parent's less the daughter's as the vacuum decay places it (its box
table, which dp_medium's plans reproduce in vacuum). Every decay then conserves the number and the energy on the grid
to rounding, up to what leaves the grid.
  In a potential the ladder's channels join eigenstates; the tables keep the vacuum labels (eigenstate a <-> the a-th
lightest mass state) and the daughters' mass states are projected on the eigenstates at their bin (the ladder's P9,
incoherently). Exact in vacuum, and to O(sigma |s| / splitting) in a potential: the mediators made on a rung off the
vacuum are counted (diag[10]) with the largest sigma met (diag[11]; the solver records sigma |s| / splitting as
medium_measure). The dark portal's band (eps' = 0) has no potential; L_e - L_mu's at m_Zd = 1e-2 eV reaches sigma ~
1e-3 Delta m^2_21 at g' = 1e-5 at the grid's top energy at z = 4 (most mediators are made far below). Channels the
potential alone opens (nu_i -> nubar_i flips) give a longitudinal mediator spread uniformly in y (number kept,
energy approximate).

The secret interactions below 1 keV (2026-11-08; the last section, SIMediator): the same construction for the four
vertices of dp_si_light -- the scalar (one population, isotropic decay, same-helicity pairs) and the vector, Majorana
or Dirac (the Dirac daughters visible or sterile, dp_si's six states) -- on dp_si's mass-state populations.

Out of scope here (docs item 22 (f)): the mediator's scattering on the relics; its production and decay on the
Galactic lines of sight (dp_galactic_path, dp_si_light's Galactic paths; ~1e-5 of the path; mediators alive at z = 0
are invisible); the dark portal's ladder tables (Mediator) for Dirac neutrinos -- the dark-portal transport has none
(check_supported refuses them when the run is configured).
"""
import numpy as np

from dp_core import njit       # dp_core first: it sets numba's CPU features
from dp_decay import _GAM, _PL, _PR, _ETA, _NLC, _rho_lf

NPOL = 3          # the mediator's LF helicities: index 0 -> 0 (longitudinal), 1 -> +1, 2 -> -1
LAMS = (0, 1, -1)
NCH = 36          # dp_medium's channels, ch = ((s * 3 + a) * 2 + rel) * 3 + b
# diag slots (the kernels' accumulators: numbers, and energies at the bins' centres in eV): the mediators produced on
# and below the grid, decayed, their neutrinos on and below the grid; those produced on a ladder rung off the vacuum
# (sigma > 0) and the largest such sigma (eV^2; dp_medium.Ladder.sig_vac is 1e-6 of the smallest splitting / max |s|)
(D_PROD, D_PROD_E, D_PROD_LOW, D_PROD_LOW_E, D_DEC, D_DEC_E, D_DAU, D_DAU_E, D_DAU_LOW, D_DAU_LOW_E, D_MEDIUM,
 D_SIGMA) = range(12)
NDIAG = 12


def coupling_mass_basis(G_flavor, U):
    """G' = U^dag G U, the identity part exactly (G - c I rotated, c I added back: the off-diagonal entries without the
    rounding of a large universal part)."""
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    c = G[0, 0].real
    return U.conj().T @ (G - c * np.eye(3)) @ U + c * np.eye(3)


def _p_star(M, mk, ml):
    lam = (M * M - (mk + ml) ** 2) * (M * M - (mk - ml) ** 2)
    return np.sqrt(max(lam, 0.0)) / (2.0 * M)


def pair_widths(G_flavor, U, masses, M, majorana=True):
    """W (3, 3): Majorana, W[k, l] (k <= l) the width (eV) of Z -> nu_k nu_l, zero below k > l and where closed; Dirac,
    W[k, l] of Z -> nu_k nubar_l for every ordered pair (module docstring)."""
    Gp = coupling_mass_basis(G_flavor, U)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    W = np.zeros((3, 3))
    if M <= 0.0:
        return W
    for k in range(3):
        for l in range(3):
            if majorana and l < k:
                continue
            if m[k] + m[l] >= M:
                continue
            g = Gp[k, l]
            p = _p_star(M, m[k], m[l])
            kin = 2.0 * M * M - m[k] ** 2 - m[l] ** 2 - (m[k] ** 2 - m[l] ** 2) ** 2 / (M * M)
            if majorana:
                s2 = 2.0 * abs(g) ** 2 * kin - 12.0 * m[k] * m[l] * (g * g).real
                S = 0.5 if k == l else 1.0
            else:
                s2 = abs(g) ** 2 * kin
                S = 1.0
            W[k, l] = max(S * p * s2 / (24.0 * np.pi * M * M), 0.0)
    return W


# ---------------------------------------------------------------------------------------------------------------------
# light-front traces


def eps_lf(q, M, lam):
    """The polarization vector (contravariant) of LF helicity lam of a vector of momentum q, mass M (boost along +z)."""
    qp = q[0] + q[3]
    if lam == 0:
        return (q / M - M * _NLC / qp).astype(np.complex128)
    e = -lam * np.array([1.0, lam * 1j]) / np.sqrt(2.0)
    eq = e[0] * q[1] + e[1] * q[2]
    return np.array([eq / qp, e[0], e[1], -eq / qp], dtype=np.complex128)


def _vertex(g, majorana):
    V = g * _PL - np.conj(g) * _PR if majorana else g * _PL
    Gm = np.einsum("mab,bc->mac", _GAM, V)
    Gb = np.einsum("ab,mcb,cd->mad", _GAM[0], Gm.conj(), _GAM[0])
    return Gm, Gb


def _contract(T, e, outgoing):
    el = _ETA @ e
    if outgoing:       # an outgoing vector: eps* in the amplitude
        return float(np.real(np.einsum("m,n,mn->", el.conj(), el, T)))
    return float(np.real(np.einsum("m,n,mn->", el, el.conj(), T)))


def msq_decay(M, mk, ml, cth, lam, h1, h2, g, majorana=True):
    """|M|^2 of Z (at rest, LF helicity lam along z) -> nu_k (angle th to z, LF helicity h1) + nu_l (h2), the vertex
    gamma^mu (g P_L - g* P_R) (Majorana) or g gamma^mu P_L (Dirac; nu_l then the antiparticle). Returns (|M|^2, x, y),
    x = k1^+ / q^+ the first daughter's share, y = 1 - x the second's (each without cancellation)."""
    p = _p_star(M, mk, ml)
    E1 = np.sqrt(p * p + mk * mk)
    E2 = np.sqrt(p * p + ml * ml)
    s = np.sqrt(max(1.0 - cth * cth, 0.0))
    k1 = np.array([E1, p * s, 0.0, p * cth])
    k2 = np.array([E2, -p * s, 0.0, -p * cth])
    # k^+ without cancellation: E - p |cos| = (m^2 + p^2 sin^2) / (E + p |cos|) on the backward side
    kp1 = E1 + p * cth if cth >= 0.0 else (mk * mk + p * p * (1.0 - cth * cth)) / (E1 - p * cth)
    kp2 = E2 - p * cth if cth <= 0.0 else (ml * ml + p * p * (1.0 - cth * cth)) / (E2 + p * cth)
    Gm, Gb = _vertex(g, majorana)
    A = _rho_lf(k1, mk, h1, kp1)
    B = _rho_lf(k2, -ml, -h2, kp2)            # v vbar of physical LF helicity h2: (k - m)(1 + g5 s) / 2
    T = np.einsum("ab,mbc,cd,nda->mn", A, Gm, B, Gb)
    q = np.array([M, 0.0, 0.0, 0.0])
    return _contract(T, eps_lf(q, M, lam), False), kp1 / M, kp2 / M


def msq_production(mi, mj, M, cth, h, hp, lam, g):
    """|M|^2 of nu_i (at rest, LF helicity h along z) -> nu_j (angle th, LF helicity hp) + Z (LF helicity lam; None:
    summed, -g + q q / M^2), the Majorana vertex (dp_decay's traces). Returns (|M|^2, z, y): z = k^+ / m_i the
    daughter's share, y = q^+ / m_i the mediator's (each without cancellation)."""
    pf = np.sqrt(max((mi * mi - (mj + M) ** 2) * (mi * mi - (mj - M) ** 2), 0.0)) / (2.0 * mi)
    Ej = np.sqrt(pf * pf + mj * mj)
    EZ = (mi * mi - mj * mj + M * M) / (2.0 * mi)
    s = np.sqrt(max(1.0 - cth * cth, 0.0))
    p = np.array([mi, 0.0, 0.0, 0.0])
    k = np.array([Ej, pf * s, 0.0, pf * cth])
    q = p - k
    # E - p |cos| = (m^2 + p^2 sin^2) / (E + p |cos|) where it cancels: the daughter backwards, the mediator forwards
    kp = Ej + pf * cth if cth >= 0.0 else (mj * mj + pf * pf * (1.0 - cth * cth)) / (Ej - pf * cth)
    qp = EZ - pf * cth if cth <= 0.0 else (M * M + pf * pf * (1.0 - cth * cth)) / (EZ + pf * cth)
    Gm, Gb = _vertex(g, True)
    A = _rho_lf(k, mj, hp, kp)
    B = _rho_lf(p, mi, h, mi)
    T = np.einsum("ab,mbc,cd,nda->mn", A, Gm, B, Gb)
    if lam is None:
        P = -_ETA + np.outer(_ETA @ q, _ETA @ q) / (M * M)
        return float(np.real(np.sum(T * P))), kp / mi, qp / mi
    return _contract(T, _eps_qplus(q, M, lam, qp), True), kp / mi, qp / mi


def _eps_qplus(q, M, lam, qp):
    """eps_lf with q^+ given (computed without cancellation by the caller)."""
    if lam == 0:
        return (q / M - M * _NLC / qp).astype(np.complex128)
    e = -lam * np.array([1.0, lam * 1j]) / np.sqrt(2.0)
    eq = e[0] * q[1] + e[1] * q[2]
    return np.array([eq / qp, e[0], e[1], -eq / qp], dtype=np.complex128)


# ---------------------------------------------------------------------------------------------------------------------
# the densities, exactly (2026-11-07, the review's finding 1: the closed-form fits and the energy tilt failed near the
# kinematic thresholds)
#
# A product's share t (decay: x; production: z) is linear in c = cos th* (th* in the parent's rest frame), on
# [lo, hi], and dN/dc is |M|^2(c). |M|^2 D(t) is a polynomial P(c) of degree <= 5, D = t (1-t) (decay) or t (1-t)^2
# (production): the light-front spinors' and polarizations' 1/k^+ and 1/q^+ (the earlier closed forms, multiplied
# out). P is kept as a Chebyshev series in c, fitted to the traces at 33 Chebyshev points of [-1, 1] (well conditioned
# whatever the window's width: near a threshold the window shrinks to a point, c does not). The cumulative number is
# integrated in w = ln(t / (1-t)), where dN/dw = P t^(1-a) (1-t)^(1-b) (D = t^a (1-t)^b) is smooth to both ends of the
# window (the 1/t, 1/(1-t) of a nearly massless product cancel there), by 8-point Gauss-Legendre panels of width
# <= 0.25: exact to rounding at every threshold.

_CHEB_N = 33
_CHEB_DEG = 8
_GL8X, _GL8W = np.polynomial.legendre.leggauss(8)
_W_FLOOR = -60.0              # a share below e^-60 (a massless product's end): below every grid, never counted


class Density:
    """The number density of a product's share t on [lo, hi] (1 - t on [ylo, yhi], given without cancellation),
    t = mid + half c: P(c) / (t^a (1-t)^b) per unit c, P the Chebyshev series coef. .total its integral (units of the
    traces), .cdf(t) and .cdf_other(s) the cumulative shares of t and of s = 1 - t, .mean the mean of t, .pdf(t) the
    density per unit t normalized to .total."""

    def __init__(self, coef, lo, hi, ylo, yhi, a, b):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.lo, self.hi, self.ylo, self.yhi = float(lo), float(hi), float(ylo), float(yhi)
        self.mid, self.half = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
        self.ymid = 0.5 * (self.ylo + self.yhi)
        self.a, self.b = int(a), int(b)
        self.wlo = max(self._logit(self.lo, self.yhi), _W_FLOOR)
        self.whi = self._logit(self.hi, self.ylo)
        npan = max(16, int(np.ceil((self.whi - self.wlo) / 0.25)))
        self.edges = np.linspace(self.wlo, self.whi, npan + 1)
        w0, w1 = self.edges[:-1], self.edges[1:]
        nodes = 0.5 * (w1 - w0)[:, None] * _GL8X[None, :] + 0.5 * (w1 + w0)[:, None]
        pan = 0.5 * (w1 - w0) * (self._dndw(nodes) @ _GL8W)
        self.cum = np.concatenate([[0.0], np.cumsum(pan)])
        self.total = float(self.cum[-1])
        self._nodes, self._hw = nodes, 0.5 * (w1 - w0)
        self.mean = self.moment(lambda t, y: t) if self.total > 0.0 else 0.5 * (self.lo + self.hi)

    def moment(self, fn):
        """The mean of fn(t, 1 - t) over the products (the panels' quadrature)."""
        t, y = self._tw(self._nodes)
        return float(np.sum(self._hw * ((self._dndw(self._nodes) * fn(t, y)) @ _GL8W)) / self.total)

    @staticmethod
    def _logit(t, y):
        return (np.log(t) if t > 0.0 else -np.inf) - np.log(y)

    @staticmethod
    def _tw(w):
        return 1.0 / (1.0 + np.exp(-w)), 1.0 / (1.0 + np.exp(w))

    def _c(self, t, y):
        # c from whichever of t, 1 - t is the smaller (no cancellation near either end)
        return np.where(t <= 0.5, (t - self.mid) / self.half if self.half > 0.0 else 0.0,
                        (self.ymid - y) / self.half if self.half > 0.0 else 0.0)

    def _dndw(self, w):
        t, y = self._tw(w)
        P = np.polynomial.chebyshev.chebval(np.clip(self._c(t, y), -1.0, 1.0), self.coef)
        return P * t ** (1 - self.a) * y ** (1 - self.b)

    def _cdf_w(self, w):
        w = np.clip(np.asarray(w, dtype=np.float64), self.wlo, self.whi)
        dw = self.edges[1] - self.edges[0]
        i = np.clip(((w - self.wlo) / dw).astype(np.int64), 0, len(self.edges) - 2)
        w0 = self.edges[i]
        hw = 0.5 * (w - w0)
        nodes = hw[..., None] * _GL8X + (0.5 * (w + w0))[..., None]
        if not self.total > 0.0:
            return np.zeros_like(w)
        return (self.cum[i] + hw * (self._dndw(nodes) @ _GL8W)) / self.total

    def cdf(self, t):
        """The share of the products with share below t."""
        t = np.asarray(t, dtype=np.float64)
        with np.errstate(divide="ignore"):
            w = np.log(np.clip(t, 1e-300, None)) - np.log1p(-np.clip(t, None, 1.0 - 1e-16))
        return np.where(t <= self.lo, 0.0, np.where(t >= self.hi, 1.0, self._cdf_w(w)))

    def cdf_other(self, s):
        """The share of the products whose partner's share 1 - t is below s."""
        s = np.asarray(s, dtype=np.float64)
        with np.errstate(divide="ignore"):
            w = np.log1p(-np.clip(s, None, 1.0 - 1e-16)) - np.log(np.clip(s, 1e-300, None))
        return np.where(s <= self.ylo, 0.0, np.where(s >= self.yhi, 1.0, 1.0 - self._cdf_w(w)))

    def pdf(self, t):
        """dN/dt (normalized to .total) at t in the window."""
        t = np.asarray(t, dtype=np.float64)
        y = 1.0 - t
        P = np.polynomial.chebyshev.chebval(self._c(t, y), self.coef)
        return P / (t ** self.a * y ** self.b)


def _chebyshev_fit(c, vals):
    """P's Chebyshev series from its values at the Chebyshev points c (least squares, degree _CHEB_DEG) and the largest
    absolute residual (the traces' rounding; recorded relative to the decay's largest |P|)."""
    coef = np.polynomial.chebyshev.chebfit(c, vals, _CHEB_DEG)
    return coef, float(np.max(np.abs(np.polynomial.chebyshev.chebval(c, coef) - vals)))


def _significant(totals):
    """The keys of the densities that carry the decay (total above 1e-14 of the largest: the rest is the traces'
    rounding of an exactly forbidden configuration) and their summed total."""
    big = max([t for t in totals.values()] + [0.0])
    keys = [k for k, t in totals.items() if t > 1e-14 * big]
    return keys, sum(totals[k] for k in keys)


_CHEB_C = np.cos(np.pi * np.arange(_CHEB_N) / (_CHEB_N - 1))[::-1]


def production_densities(G_flavor, U, masses, M):
    """dens[(s, i, j, rel, lam index)] = Density of the daughter's share z in nu_i -> nu_j + Z_D (s the parent's kind,
    0 the neutrino, LF helicity -1; rel 0 kept, 1 flipped; the mediator's LF helicity LAMS[lam index]), all of one
    (s, i, j) in the same units (their totals add up to its rate x a constant); and the largest fit residual. Majorana
    (the dark portal's vertex); only open pairs with a coupling."""
    Gp = coupling_mass_basis(G_flavor, U)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    dens = {}
    worst = 0.0
    if M <= 0.0:
        return dens, worst
    for i in range(3):
        for j in range(3):
            g = Gp[j, i]
            if i == j or m[i] <= m[j] + M or g == 0.0:
                continue
            mi, mj = float(m[i]), float(m[j])
            pf = np.sqrt(max((mi * mi - (mj + M) ** 2) * (mi * mi - (mj - M) ** 2), 0.0)) / (2.0 * mi)
            if not pf > 0.0:
                continue
            Ej = np.sqrt(pf * pf + mj * mj)
            EZ = (mi * mi - mj * mj + M * M) / (2.0 * mi)
            zlo, zhi = mj * mj / (Ej + pf) / mi, (Ej + pf) / mi
            ylo, yhi = M * M / (EZ + pf) / mi, (EZ + pf) / mi
            for s_, h in ((0, -1), (1, 1)):
                scale = 0.0
                res_max = 0.0
                for rel, hp in ((0, h), (1, -h)):
                    for li, lam in enumerate(LAMS):
                        v = np.array([msq_production(mi, mj, M, c, h, hp, lam, g) for c in _CHEB_C])
                        P = v[:, 0] * v[:, 1] * v[:, 2] ** 2           # |M|^2 z (1-z)^2
                        coef, res = _chebyshev_fit(_CHEB_C, P)
                        scale = max(scale, float(np.max(np.abs(P))))
                        res_max = max(res_max, res)
                        dens[(s_, i, j, rel, li)] = Density(coef, zlo, zhi, ylo, yhi, 1, 2)
                worst = max(worst, res_max / scale if scale > 0.0 else 0.0)
    return dens, worst


def decay_densities(G_flavor, U, masses, M, majorana=True):
    """The pairs Z -> nu_k nu_l (pair_widths' W[k, l] > 0) per mediator LF helicity index and daughters' LF helicities
    (h1, h2), hi = (h1 > 0) * 2 + (h2 > 0): dens[(k, l, lam index, hi)] the Density of the first daughter's share x, the
    four of one (k, l, lam) in the same units; and W, the largest fit residual."""
    Gp = coupling_mass_basis(G_flavor, U)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    W = pair_widths(G_flavor, U, masses, M, majorana)
    dens = {}
    worst = 0.0
    for k in range(3):
        for l in range(3):
            if W[k, l] <= 0.0:
                continue
            g = Gp[k, l]
            mk, ml = float(m[k]), float(m[l])
            p = _p_star(M, mk, ml)
            E1 = np.sqrt(p * p + mk * mk)
            E2 = np.sqrt(p * p + ml * ml)
            xlo, xhi = mk * mk / (E1 + p) / M, (E1 + p) / M
            ylo, yhi = ml * ml / (E2 + p) / M, (E2 + p) / M
            scale = 0.0
            res_max = 0.0
            for li, lam in enumerate(LAMS):
                for h1 in (-1, 1):
                    for h2 in (-1, 1):
                        v = np.array([msq_decay(M, mk, ml, c, lam, h1, h2, g, majorana) for c in _CHEB_C])
                        P = v[:, 0] * v[:, 1] * v[:, 2]                  # |M|^2 x (1-x)
                        coef, res = _chebyshev_fit(_CHEB_C, P)
                        scale = max(scale, float(np.max(np.abs(P))))
                        res_max = max(res_max, res)
                        dens[(k, l, li, (h1 > 0) * 2 + (h2 > 0))] = Density(coef, xlo, xhi, ylo, yhi, 1, 1)
            worst = max(worst, res_max / scale if scale > 0.0 else 0.0)
    return dens, W, worst


# ---------------------------------------------------------------------------------------------------------------------
# tables on the grid's offsets

_GLX, _GLW = np.polynomial.legendre.leggauss(16)


def _bin_average(cdf, lo, hi, h, a):
    """I(a) = (1 / (h - 1/h)) int_{1/h}^{h} du C(a / u) for every a (array): the share below a E_b of a product whose
    share of its parent's energy has the cumulative C on [lo, hi] (0 below, 1 above), the parent spread uniformly over
    its bin [E_b / h, E_b h]. Gauss-Legendre on the smooth piece between C's ends."""
    a = np.asarray(a, dtype=np.float64)
    u1, u2 = 1.0 / h, h
    b1 = np.clip(a / hi, u1, u2)
    b2 = np.clip(a / lo, u1, u2) if lo > 0.0 else np.full_like(a, u2)
    hw = 0.5 * (b2 - b1)
    uu = hw[:, None] * _GLX[None, :] + (0.5 * (b2 + b1))[:, None]
    smooth = hw * (cdf(a[:, None] / uu) @ _GLW)
    return ((b1 - u1) + np.where(b2 > b1, smooth, 0.0)) / (u2 - u1)


def offset_table(cdf, lo, hi, h, K):
    """f[k], k = 0..K: the share of the products in the bin k below their parent's (energies [h^(-2k-1), h^(-2k+1)]
    E_b), the parent spread over its bin; the cumulative made monotone (rounding), so the shares telescope exactly:
    sum f = 1 - (the share below offset K)."""
    a = h ** (1.0 - 2.0 * np.arange(K + 2))
    I = _bin_average(cdf, lo, hi, h, a)
    I = np.minimum.accumulate(np.clip(I, 0.0, 1.0))
    if hi <= 1.0:
        I[0] = 1.0
    return I[:-1] - I[1:]


def energy_fix(f, h, target):
    """f (K1,) over the offsets, its mean energy at the bins' centres (sum f h^-2k) set exactly to target, its number
    and its entries' signs kept (2026-11-07; the review's findings 1 and 4: per block, never failing):
      a tilt f_k (1 + alpha (h^-2k - mu)) where it changes no entry by more than half (alpha ~ (ln h)^2: the binning's
      error, 1e-4 relative at 515 bins over 4.5 decades) -- the shape kept;
      else (a product narrower than a bin or two, near a threshold) a share theta of it moved into the bin next to
      it on the side the target lies (one above its highest bin or below its lowest), theta = the defect over that
      bin's distance from the mean; where that is not enough (theta > 1), the whole of it at the target, shared
      between the two bins around it (energy and number kept: the linear rule)."""
    f = np.asarray(f, dtype=np.float64)
    K1 = f.shape[0]
    e = h ** (-2.0 * np.arange(K1))
    n = float(np.sum(f))
    if n <= 0.0:
        return f
    E = float(np.sum(f * e))
    d = float(target) - E
    if d == 0.0:
        return f.copy()
    mu = E / n
    occ = np.nonzero(f[:K1 - 1] > 0.0)[0]          # the last offset (below every bin) never chosen
    if len(occ) == 0:
        return f.copy()
    var = float(np.sum(f * (e - mu) ** 2))
    if var > 0.0:
        alpha = d / var
        if abs(alpha) * float(np.max(np.abs(e[occ] - mu))) <= 0.5:
            return f * (1.0 + alpha * (e - mu))
    tau = float(target) / n
    kt = int(occ[0]) - 1 if d > 0.0 else int(occ[-1]) + 1
    kt = min(max(kt, 0), K1 - 2)
    theta = (tau - mu) / (e[kt] - mu) if e[kt] != mu else 2.0
    if 0.0 < theta <= 1.0:
        out = (1.0 - theta) * f
        out[kt] += theta * n
        return out
    # the whole at the target: the two bins around tau (tau within (e[K1-1], 1])
    tau = min(max(tau, e[-1]), 1.0)
    k = int(np.clip(np.floor(-0.5 * np.log(tau) / np.log(h)), 0, K1 - 2))
    w = (tau - e[k + 1]) / (e[k] - e[k + 1])
    out = np.zeros_like(f)
    out[k] = w * n
    out[k + 1] = (1.0 - w) * n
    return out


def _K(lo, h, N):
    """The deepest offset a share >= lo reaches, capped at N + 1 (below every bin of an N-bin grid)."""
    if lo <= 0.0:
        return N + 1
    return int(min(N + 1, np.floor(0.5 * (1.0 - np.log(lo / h) / np.log(h))) + 1))


def _last(t):
    """1 + the deepest offset with an entry (over the leading axes)."""
    nz = np.nonzero(np.any(np.reshape(t, (-1, t.shape[-1])) != 0.0, axis=0))[0]
    return int(nz[-1]) + 1 if len(nz) else 0


def _uniform_cdf(y):
    return np.clip(y, 0.0, 1.0)


def _sums(t, h):
    """Per table row (all axes but the offsets): (its sum, its energy at the bins' centres, sum t h^-2k), (..., 2)."""
    e = h ** (-2.0 * np.arange(t.shape[-1]))
    return np.ascontiguousarray(np.stack([t.sum(axis=-1), (t * e).sum(axis=-1)], axis=-1))


def _rest(t):
    """The share an offset table leaves below its last offset (1 - its sum), zero where it is rounding."""
    r = 1.0 - float(np.sum(t))
    return r if r > 1e-13 else 0.0


def _table(cdf, lo, hi, h, N, K1):
    """offset_table of a product (share on [lo, hi]) padded to K1 offsets, what falls below offset N + 1 in the last."""
    t = offset_table(cdf, lo, hi, h, _K(lo, h, N))
    out = np.zeros(K1)
    out[:len(t)] = t
    out[K1 - 1] += _rest(t)
    return out


class DecayBackUnsupported(RuntimeError):
    """A run the decay back does not cover yet (raised when the run is configured, never inside a fit: not a
    ValueError, which the analysis' objective turns into its wall)."""


def check_supported(majorana=True):
    """Raises DecayBackUnsupported for a dark-portal configuration with dark_portal.mediator_decay_back on that the
    ladder's tables (Mediator) do not cover: Dirac neutrinos (the dark-portal transport has none; the secret
    interactions' Dirac neutrinos below 1 keV are SIMediator's)."""
    if not majorana:
        raise DecayBackUnsupported(
            "dark_portal.mediator_decay_back = True: the dark portal's tables (dp_mediator.Mediator) are for Majorana "
            "neutrinos only (the secret interactions' Dirac neutrinos below 1 keV: dp_mediator.SIMediator); set "
            "dark_portal.mediator_decay_back = False (docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md item 22 (f))")


class Mediator:
    """The mediator's tables for one run on a grid (module docstring): .Gamma its rest width (eV), .rate = Gamma M (the
    lab rate Gamma M / E_phys), .br_nu the neutrino branching, .W the pair widths; .prod (NCH, NPOL, N + 2) the
    mediator's offsets per decay through each of dp_medium's ladder channels (vacuum labels: eigenstate a <-> mass
    state order[a]), .nprod (NCH,) the offsets used; .dec (NPOL, 2, 3, N + 2) the neutrinos per decayed mediator by
    kind (0 the neutrino, LF helicity -1; 1 the antineutrino) and mass state, .ndec (NPOL,); .arrays for the kernels;
    .fold() dp_si's tables; .fit_residual the traces' rounding in the densities. Majorana neutrinos.
      Number and energy per block (2026-11-07, the review's finding 4): each table row -- a production channel's
    mediators of one LF helicity, a decay's neutrinos of one kind and mass state -- holds exactly its share of the
    number (the traces'), and its energy at the bins' centres is set exactly (energy_fix): a decay row to its exact
    continuous energy (the pair's shares add up to the mediator's, times the neutrino branching); a production
    channel's rows to their exact continuous energies scaled together so that they add up to the parent's (its bin's
    centre) less the daughter's as the vacuum decay places it (its box table: what dp_medium's plans do)."""

    def __init__(self, G_flavor, U, masses, M, E_bins, Gamma_vis=0.0, majorana=True):
        check_supported(majorana)
        E = np.asarray(E_bins, dtype=np.float64)
        N = len(E)
        h = float(np.sqrt(E[1] / E[0]))
        m = np.asarray(masses, dtype=np.float64)
        self.h, self.N, self.M = h, N, float(M)
        self.order = np.argsort(m)
        K1 = N + 2
        # the decay back: the pair widths, the neutrino branching and the daughters per LF helicity
        ddens, W, res_d = decay_densities(G_flavor, U, m, M)
        self.W = W
        self.Gamma = float(W.sum()) + float(Gamma_vis)
        self.br_nu = float(W.sum()) / self.Gamma if self.Gamma > 0.0 else 0.0
        self.rate = self.Gamma * float(M)
        dec = np.zeros((NPOL, 2, 3, K1))
        dtarget = np.zeros((NPOL, 2, 3))
        for k in range(3):
            for l in range(3):
                if W[k, l] <= 0.0:
                    continue
                br = W[k, l] / self.Gamma
                for li in range(NPOL):
                    keys, tot = _significant({hi: ddens[(k, l, li, hi)].total for hi in range(4)})
                    for hi in keys:
                        d = ddens[(k, l, li, hi)]
                        w = br * d.total / tot
                        dec[li, hi // 2, k] += w * _table(d.cdf, d.lo, d.hi, h, N, K1)
                        dec[li, hi % 2, l] += w * _table(d.cdf_other, d.ylo, d.yhi, h, N, K1)
                        dtarget[li, hi // 2, k] += w * d.mean
                        dtarget[li, hi % 2, l] += w * (1.0 - d.mean)
        for li in range(NPOL):
            for kd in range(2):
                for st in range(3):
                    dec[li, kd, st] = energy_fix(dec[li, kd, st], h, dtarget[li, kd, st])
        self.dec = np.ascontiguousarray(dec)
        self.ndec = np.array([_last(dec[li]) for li in range(NPOL)], dtype=np.int64)
        # the production: per ladder channel, the mediator's share y = 1 - z by LF helicity
        pdens, res_p = production_densities(G_flavor, U, m, M)
        self.fit_residual = max(res_d, res_p)
        prod = np.zeros((NCH, NPOL, K1))
        self.fallback = np.zeros(NCH, dtype=np.bool_)
        self.shares = np.zeros((2, 3, 3, 2))          # each rel's share of nu_i -> nu_j (the traces')
        for s_ in range(2):
            for i in range(3):
                for j in range(3):
                    if (s_, i, j, 0, 0) not in pdens:
                        continue
                    keys, tot = _significant({(rel, li): pdens[(s_, i, j, rel, li)].total for rel in range(2)
                                              for li in range(NPOL)})
                    for rel, li in keys:
                        self.shares[s_, i, j, rel] += pdens[(s_, i, j, rel, li)].total / tot
        for s_ in range(2):
            for a in range(3):
                for rel in range(2):
                    for b in range(3):
                        ch = ((s_ * 3 + a) * 2 + rel) * 3 + b
                        i, j = self.order[a], self.order[b]
                        if (s_, i, j, rel, 0) not in pdens or self.shares[s_, i, j, rel] <= 1e-14:
                            # closed in vacuum (a channel only a potential opens): longitudinal, uniform in y
                            self.fallback[ch] = True
                            prod[ch, 0] = _table(_uniform_cdf, 0.0, 1.0, h, N, K1)
                            continue
                        keys, _ = _significant({(r_, li): pdens[(s_, i, j, r_, li)].total for r_ in range(2)
                                                for li in range(NPOL)})
                        ds = [pdens[(s_, i, j, rel, li)] if (rel, li) in keys else None for li in range(NPOL)]
                        tot = sum(d.total for d in ds if d is not None)
                        # the daughter as the decay places it (its box table) and its energy at the bins' centres
                        d0 = pdens[(s_, i, j, rel, 0)]
                        dcdf = lambda z, ds=ds, tot=tot: sum(d.total * d.cdf(z) for d in ds  # noqa: E731
                                                             if d is not None) / tot
                        dtab = _table(dcdf, d0.lo, d0.hi, h, N, K1)
                        T = 1.0 - float(np.sum(dtab * h ** (-2.0 * np.arange(K1))))
                        ybar = sum(d.total * (1.0 - d.mean) for d in ds if d is not None) / tot
                        for li, d in enumerate(ds):
                            if d is None:
                                continue
                            row = d.total / tot * _table(d.cdf_other, d.ylo, d.yhi, h, N, K1)
                            prod[ch, li] = energy_fix(row, h, d.total / tot * (1.0 - d.mean) * T / ybar)
        self.prod = np.ascontiguousarray(prod)
        self.nprod = np.array([_last(prod[ch]) for ch in range(NCH)], dtype=np.int64)
        self.psum = _sums(self.prod, h)
        self.dsum = _sums(self.dec, h)
        self.arrays = (self.prod, self.nprod, self.psum, self.dec, self.ndec, self.dsum, float(self.rate))

    def fold(self):
        """dp_si's tables: one population holding both kinds of each mass state (dp_si_light.transport_arrays: the
        neutrino parent's channels, kept and flipped added by their shares): prod (3, 3, NPOL, N + 2) per decay of
        mass state i into j, nprod (3, 3); dec (NPOL, 3, N + 2) the neutrinos of both kinds per decayed mediator,
        ndec (NPOL,), and their sums (_sums). Returns (prod, nprod, psum, dec, ndec, dsum, rate)."""
        K1 = self.prod.shape[2]
        prod = np.zeros((3, 3, NPOL, K1))
        inv = np.empty(3, dtype=np.int64)
        inv[self.order] = np.arange(3)
        for i in range(3):
            for j in range(3):
                tot = 0.0
                for rel in range(2):
                    ch = ((0 * 3 + inv[i]) * 2 + rel) * 3 + inv[j]
                    w = self.shares[0, i, j, rel]
                    if self.fallback[ch] or w <= 0.0:
                        continue
                    prod[i, j] += w * self.prod[ch]
                    tot += w
                if tot > 0.0:
                    prod[i, j] /= tot
        dec = np.ascontiguousarray(self.dec.sum(axis=1))
        nprod = np.array([[_last(prod[i, j]) for j in range(3)] for i in range(3)], dtype=np.int64)
        ndec = np.array([_last(dec[li]) for li in range(NPOL)], dtype=np.int64)
        return (np.ascontiguousarray(prod), nprod, _sums(prod, self.h), dec, ndec, _sums(dec, self.h),
                float(self.rate))


def path_exponent(M_rate, E_min, H0, Omega_m, Omega_L, z_max):
    """The mediator's decay exponent over the whole path at energy E_min: M_rate / E_min int_0^z_max dz / ((1+z)^2 H)
    (M_rate = Gamma_Z m_Zd)."""
    zg, wg = np.polynomial.legendre.leggauss(32)
    zz = 0.5 * float(z_max) * (zg + 1.0)
    I = 0.5 * float(z_max) * float(np.sum(wg / ((1.0 + zz) ** 2 * H0 * np.sqrt(Omega_m * (1.0 + zz) ** 3 + Omega_L))))
    return float(M_rate) / float(E_min) * I


def setup(G_flavor, U, masses, M, E_bins, H0, Omega_m, Omega_L, z_max, Gamma_vis=0.0, tau_min=1e-8):
    """The run's Mediator, or None where the decay back cannot act: no pair open (m_Zd <= 2 m_lightest), or the
    mediator's decay exponent over the whole path at the grid's lowest energy (path_exponent) below tau_min (the
    mediators then stay invisible, as with the setting off, bit for bit)."""
    m = np.asarray(masses, dtype=np.float64)
    if float(M) <= 2.0 * float(np.min(m)):
        return None
    W = pair_widths(G_flavor, U, m, M)
    G_nu = float(W.sum())
    if not G_nu > 0.0:
        return None
    if path_exponent((G_nu + float(Gamma_vis)) * float(M), float(np.asarray(E_bins)[0]), H0, Omega_m, Omega_L,
                     z_max) < tau_min:
        return None
    return cached(G_flavor, U, m, M, E_bins, Gamma_vis)


_CACHE = {}


def cached(G_flavor, U, masses, M, E_bins, Gamma_vis=0.0):
    """Mediator(G_flavor, U, masses, M, E_bins, Gamma_vis), kept per arguments (the tables take ~1 s; a parameter
    point runs many sources): the last 8."""
    E = np.asarray(E_bins, dtype=np.float64)
    key = (np.ascontiguousarray(G_flavor, dtype=np.complex128).tobytes(),
           np.ascontiguousarray(U, dtype=np.complex128).tobytes(),
           np.ascontiguousarray(masses, dtype=np.float64).tobytes(), float(M), float(E[0]), float(E[1]), len(E),
           float(Gamma_vis))
    if key not in _CACHE:
        if len(_CACHE) >= 8:
            _CACHE.pop(next(iter(_CACHE)))
        try:
            _CACHE[key] = Mediator(G_flavor, U, masses, M, E, Gamma_vis=Gamma_vis)
        except ValueError as err:
            # as si_cached: never a ValueError inside a fit (the objective's wall); a failure of the tables stops the run
            raise DecayBackUnsupported(f"dp_mediator: the decay-back tables failed at M = {M!r} eV, masses "
                                       f"{np.asarray(masses).tolist()}: {err}") from err
    return _CACHE[key]


# ---------------------------------------------------------------------------------------------------------------------
# kernels


@njit(cache=True)
def produce(prod, nprod, psum, plan_tot, r0, w, ch0, n, b, Eb, h, dt, pendZ, diag, sgm):
    """A state of grid bin b decayed through the ladder's channels ch0 .. ch0 + 5 (rel, daughter state), n its decayed
    number per unit time in the plans' unit (dp_medium.deposit's n), the channels' rates blended between the bin's
    rungs r0, r0 + 1 (upper weight w): its mediators in the bins below into pendZ[e, lam] (numbers per unit time, a
    constant source of bin e over the step); those in the bin itself are in the bin's linear system (own_rows).
    diag: the produced numbers and energies (at the bins' centres), on (the bin's own included) and below the grid
    (the tables' sums psum less the grid's), and those made off the vacuum (sgm: the bin's largest rung sigma, > 0
    there) with the largest sigma met."""
    q = 1.0 / (h * h)
    for x in range(6):
        ch = ch0 + x
        rate = (1.0 - w) * plan_tot[r0, ch]
        if w > 0.0:
            rate += w * plan_tot[r0 + 1, ch]
        if rate == 0.0:
            continue
        nc = n * rate
        if sgm > 0.0:
            diag[10] += nc * dt
            if sgm > diag[11]:
                diag[11] = sgm
        K = min(nprod[ch], b + 1)
        for lam in range(3):
            sn = prod[ch, lam, 0]
            se = sn
            ek = q
            for k in range(1, K):
                v = prod[ch, lam, k]
                if v != 0.0:
                    pendZ[b - k, lam] += nc * v
                    sn += v
                    se += v * ek
                ek *= q
            diag[0] += nc * dt * sn
            diag[1] += nc * dt * Eb * se
            diag[2] += nc * dt * (psum[ch, lam, 0] - sn)
            diag[3] += nc * dt * Eb * (psum[ch, lam, 1] - se)


@njit(cache=True)
def decay_exponent(rate, Eb, decay_dt):
    """The mediators' decay exponent over a step, rate / E_obs x the step's decay time (rate = Gamma_Z m_Zd), capped at
    dp_si.DECAY_STEP_CAP (1e9: beyond it the mediators drain within the step all the same, and their neutrinos keep
    their shares; the bin's linear system stays clear of the stiff path's rounding). Returns (exponent, the uncapped
    one)."""
    tau = rate / Eb * decay_dt
    return (tau if tau <= 1e9 else 1e9), tau


@njit(cache=True)
def daughters_below(D, b, Eb, h, dt, dec, ndec, dsum, pendM, diag):
    """The neutrinos of the mediators of grid bin b decayed over a step, D (3,) their numbers by LF helicity: those in
    the bins below into pendM[e, kind, state] (numbers per unit time, a constant source of bin e over the step; one
    kind for dp_si's folded flux); those in the bin itself are in the bin's linear system (own_rows). diag: the
    decayed and the neutrinos on (the bin's own included) and below the grid."""
    q = 1.0 / (h * h)
    for lam in range(3):
        Dl = D[lam]
        if Dl == 0.0:
            continue
        diag[4] += Dl
        diag[5] += Dl * Eb
        K = min(ndec[lam], b + 1)
        Ddt = Dl / dt
        for kd in range(dec.shape[1]):
            for st in range(3):
                v = dec[lam, kd, st, 0]
                sn = v
                se = v
                ek = q
                for k in range(1, K):
                    v = dec[lam, kd, st, k]
                    if v != 0.0:
                        pendM[b - k, kd, st] += Ddt * v
                        sn += v
                        se += v * ek
                    ek *= q
                diag[6] += Dl * sn
                diag[7] += Dl * Eb * se
                diag[8] += Dl * (dsum[lam, kd, st, 0] - sn)
                diag[9] += Dl * Eb * (dsum[lam, kd, st, 1] - se)


# ---------------------------------------------------------------------------------------------------------------------
# The secret interactions below 1 keV (2026-11-08; dp_si_light's transports through dp_si; docs/SIMPLIFICATIONS_AND_
# ASSUMPTIONS.md item 22 (g)). The mediator X of nu_i -> nu_j + X, a scalar phi or a vector V coupled to neutrinos
# alone (paper Eq. lag_si: L_phi = -kappa phi sum G nubar nu, kappa 1/2 Majorana, 1 Dirac; L_V = -V_mu sum G nubar
# gamma^mu P_L nu), decays back, X -> nu_k nu_l, where M > m_k + m_l. The construction above -- exact light-front
# densities, tables on the grid's offsets with every block's number and energy exact -- for the four vertices
# (dp_si_light):
#   scalar, Majorana: lambda_kl P_L + lambda*_kl P_R,  lambda = U^T G U (complex symmetric; the whole coupling)
#   scalar, Dirac:    h_kl (both chiralities),         h = U^dag G U (Hermitian; nu_R rotates as nu_L)
#   vector, Majorana: gamma^mu (G'_kl P_L - G'*_kl P_R) (the dark portal's vertex: msq_decay, msq_production)
#   vector, Dirac:    gamma^mu G'_kl P_L,               G' = U^dag G U
# the identity part of G rotated exactly (coupling_mass_basis): it decays X into nu_k nu_k (nu_k nubar_k), though it
# changes no mass state in nu_i -> nu_j + X.
#   Widths, summed over spins (the vector's: pair_widths), p = lambda^(1/2)(M^2, m_k^2, m_l^2) / (2 M):
#     scalar, Majorana: sum|M|^2 = 2 |l_kl|^2 (M^2 - m_k^2 - m_l^2) - 4 m_k m_l Re(l_kl^2), Gamma = S p sum|M|^2 /
#       (8 pi M^2), S = 1/2 for k = l (each unordered pair once). Massless: Tr(l l^dag) M / 16 pi (dp_si.scalar_width);
#       k = l, real l: l^2 M beta^3 / 16 pi (the CP-even scalar's P wave).
#     scalar, Dirac: sum|M|^2 = 2 |h_kl|^2 (M^2 - (m_k + m_l)^2) for X -> nu_k nubar_l, every ordered pair, S = 1.
#       Massless: Tr(h h^dag) M / 8 pi (dp_si.scalar_width).
#   The scalar has no polarization: one population (slot 0 of the NPOL slots; 1 and 2 stay empty). Summed over the
# daughters' helicities its decay is isotropic, each daughter's share uniform on its window; massless daughters come
# in same-helicity pairs (J = 0: two neutrinos or two antineutrinos for Majorana; for Dirac a visible neutrino with a
# sterile antineutrino or a sterile neutrino with a visible antineutrino -- one visible daughter per decay); massive
# daughters also in opposite-helicity pairs (checked).
#   Kinds. Majorana: a daughter of LF helicity -1 is the beam's neutrino, +1 its antineutrino; dp_si's one flux holds
# both (the kinds summed). Dirac: the decay gives nu_k (the u spinor, LF helicity h1) and nubar_l (the v spinor, h2);
# nu_k is visible for h1 = -1, nubar_l for h2 = +1, the others sterile -- dp_si's six states, visible 0-2 and sterile
# 3-5, each holding both kinds (the CP mirror, X -> nubar_k(-h1) nu_l(-h2) at the same rate and shares, has the same
# visibilities: the summed populations are exact). The parents: Majorana, the neutrino (LF helicity -1; the
# antineutrino's mediators are the CP mirror, alike once the kinds are summed); Dirac, the visible (-1) and the sterile
# (+1) neutrino -- dp_si_light.transport_arrays' decay a -> c is (parent block, mass state) -> (daughter block, mass
# state), the daughter's block the parent's (kept) or the other (flipped).

def _si_med(mediator):
    med = str(mediator).strip().lower()
    if med not in ("scalar", "vector"):
        raise DecayBackUnsupported(f"dp_mediator: the secret interactions' mediator is 'scalar' or 'vector', got "
                                   f"{mediator!r}")
    return med


def si_coupling(G_flavor, U, mediator, majorana):
    """The mediator's whole coupling in the mass basis (the section's notes): lambda = U^T G U for the scalar with
    Majorana neutrinos, else coupling_mass_basis (U^dag G U, its identity part exact). For k != l the entries are
    dp_si_light.mass_couplings', bit for bit."""
    if _si_med(mediator) == "scalar" and majorana:
        G = np.asarray(G_flavor, dtype=np.complex128)
        U = np.asarray(U, dtype=np.complex128)
        return U.T @ G @ U
    return coupling_mass_basis(G_flavor, U)


def si_vertex(g, mediator, majorana):
    """The 4 x 4 vertex of the coupling g (as dp_si_light._vertex): the scalar's Gamma, or the vector's V with
    Gamma^mu = gamma^mu V."""
    if _si_med(mediator) == "scalar":
        return g * _PL + np.conj(g) * _PR if majorana else g * np.eye(4, dtype=np.complex128)
    return g * _PL - np.conj(g) * _PR if majorana else g * _PL


def si_pair_widths(G_flavor, U, masses, M, mediator, majorana):
    """W (3, 3), the width (eV) of X -> nu_k nu_l (Majorana: k <= l, zero below) or X -> nu_k nubar_l (Dirac: every
    ordered pair), zero where closed (the section's notes); the vector's is pair_widths."""
    med = _si_med(mediator)
    if med == "vector":
        return pair_widths(G_flavor, U, masses, M, majorana)
    lam = si_coupling(G_flavor, U, med, majorana)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    W = np.zeros((3, 3))
    if M <= 0.0:
        return W
    for k in range(3):
        for l in range(3):
            if majorana and l < k:
                continue
            if m[k] + m[l] >= M:
                continue
            g = lam[k, l]
            p = _p_star(M, m[k], m[l])
            if majorana:
                s2 = 2.0 * abs(g) ** 2 * (M * M - m[k] ** 2 - m[l] ** 2) - 4.0 * m[k] * m[l] * (g * g).real
                S = 0.5 if k == l else 1.0
            else:
                s2 = 2.0 * abs(g) ** 2 * (M * M - (m[k] + m[l]) ** 2)
                S = 1.0
            W[k, l] = max(S * p * s2 / (8.0 * np.pi * M * M), 0.0)
    return W


def _decay_kinematics(M, mk, ml, cth):
    """msq_decay's momenta: X at rest, nu_k at angle th to z; (k1, k2, k1^+, k2^+), the k^+ without cancellation."""
    p = _p_star(M, mk, ml)
    E1 = np.sqrt(p * p + mk * mk)
    E2 = np.sqrt(p * p + ml * ml)
    s = np.sqrt(max(1.0 - cth * cth, 0.0))
    k1 = np.array([E1, p * s, 0.0, p * cth])
    k2 = np.array([E2, -p * s, 0.0, -p * cth])
    kp1 = E1 + p * cth if cth >= 0.0 else (mk * mk + p * p * (1.0 - cth * cth)) / (E1 - p * cth)
    kp2 = E2 - p * cth if cth <= 0.0 else (ml * ml + p * p * (1.0 - cth * cth)) / (E2 + p * cth)
    return k1, k2, kp1, kp2


def msq_decay_si(M, mk, ml, cth, lam, h1, h2, g, mediator, majorana):
    """|M|^2 of X (at rest) -> nu_k (angle th to z, LF helicity h1) + nu_l (h2; Dirac: the antiparticle nubar_l): the
    vector of LF helicity lam as msq_decay; the scalar (lam ignored) Tr[u ubar Gamma v vbar Gamma-bar], Gamma
    si_vertex's. Returns (|M|^2, x, 1 - x), x the first daughter's share."""
    if _si_med(mediator) == "vector":
        return msq_decay(M, mk, ml, cth, lam, h1, h2, g, majorana)
    k1, k2, kp1, kp2 = _decay_kinematics(M, mk, ml, cth)
    V = si_vertex(g, "scalar", majorana)
    Vb = _GAM[0] @ V.conj().T @ _GAM[0]
    A = _rho_lf(k1, mk, h1, kp1)
    B = _rho_lf(k2, -ml, -h2, kp2)            # v vbar of physical LF helicity h2
    return float(np.real(np.trace(A @ V @ B @ Vb))), kp1 / M, kp2 / M


def msq_production_si(mi, mj, M, cth, h, hp, lam, g, mediator, majorana):
    """|M|^2 of nu_i (at rest, LF helicity h) -> nu_j (angle th, LF helicity hp) + X, the vertex si_vertex(g): the
    vector with Majorana neutrinos is msq_production; the vector's LF helicity lam (None: summed); the scalar's lam
    ignored. Returns (|M|^2, z, y), z = k^+ / m_i the daughter's share, y = q^+ / m_i the mediator's."""
    med = _si_med(mediator)
    if med == "vector" and majorana:
        return msq_production(mi, mj, M, cth, h, hp, lam, g)
    pf = np.sqrt(max((mi * mi - (mj + M) ** 2) * (mi * mi - (mj - M) ** 2), 0.0)) / (2.0 * mi)
    Ej = np.sqrt(pf * pf + mj * mj)
    EZ = (mi * mi - mj * mj + M * M) / (2.0 * mi)
    s = np.sqrt(max(1.0 - cth * cth, 0.0))
    p = np.array([mi, 0.0, 0.0, 0.0])
    k = np.array([Ej, pf * s, 0.0, pf * cth])
    q = p - k
    kp = Ej + pf * cth if cth >= 0.0 else (mj * mj + pf * pf * (1.0 - cth * cth)) / (Ej - pf * cth)
    qp = EZ - pf * cth if cth <= 0.0 else (M * M + pf * pf * (1.0 - cth * cth)) / (EZ + pf * cth)
    V = si_vertex(g, med, majorana)
    A = _rho_lf(k, mj, hp, kp)
    B = _rho_lf(p, mi, h, mi)
    if med == "scalar":
        Vb = _GAM[0] @ V.conj().T @ _GAM[0]
        return float(np.real(np.trace(A @ V @ B @ Vb))), kp / mi, qp / mi
    Gm = np.einsum("mab,bc->mac", _GAM, V)
    Gb = np.einsum("ab,mcb,cd->mad", _GAM[0], Gm.conj(), _GAM[0])
    T = np.einsum("ab,mbc,cd,nda->mn", A, Gm, B, Gb)
    if lam is None:
        P = -_ETA + np.outer(_ETA @ q, _ETA @ q) / (M * M)
        return float(np.real(np.sum(T * P))), kp / mi, qp / mi
    return _contract(T, _eps_qplus(q, M, lam, qp), True), kp / mi, qp / mi


def si_npol(mediator):
    """The mediator's populations: 1 for the scalar, NPOL (LF helicities 0, +1, -1) for the vector."""
    return 1 if _si_med(mediator) == "scalar" else NPOL


def si_production_densities(G_flavor, U, masses, M, mediator, majorana):
    """production_densities for the four vertices: dens[(s, i, j, rel, li)] the Density of the daughter's share z in
    nu_i -> nu_j + X, s the parent's LF helicity -1 (0) or +1 (1) -- Majorana the neutrino and the antineutrino, Dirac
    the visible and the sterile neutrino --, rel 0 the daughter keeps it, 1 flips it, li the mediator's population (the
    vector's LAMS; the scalar's one, 0); all of one (s, i, j) in the same units. And the largest fit residual."""
    med = _si_med(mediator)
    cpl = si_coupling(G_flavor, U, med, majorana)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    lams = LAMS if med == "vector" else (None,)
    dens = {}
    worst = 0.0
    if M <= 0.0:
        return dens, worst
    for i in range(3):
        for j in range(3):
            g = cpl[j, i]
            if i == j or m[i] <= m[j] + M or g == 0.0:
                continue
            mi, mj = float(m[i]), float(m[j])
            pf = np.sqrt(max((mi * mi - (mj + M) ** 2) * (mi * mi - (mj - M) ** 2), 0.0)) / (2.0 * mi)
            if not pf > 0.0:
                continue
            Ej = np.sqrt(pf * pf + mj * mj)
            EZ = (mi * mi - mj * mj + M * M) / (2.0 * mi)
            zlo, zhi = mj * mj / (Ej + pf) / mi, (Ej + pf) / mi
            ylo, yhi = M * M / (EZ + pf) / mi, (EZ + pf) / mi
            for s_, h in ((0, -1), (1, 1)):
                scale = 0.0
                res_max = 0.0
                for rel, hp in ((0, h), (1, -h)):
                    for li, lam in enumerate(lams):
                        v = np.array([msq_production_si(mi, mj, M, c, h, hp, lam, g, med, majorana) for c in _CHEB_C])
                        P = v[:, 0] * v[:, 1] * v[:, 2] ** 2           # |M|^2 z (1-z)^2
                        coef, res = _chebyshev_fit(_CHEB_C, P)
                        scale = max(scale, float(np.max(np.abs(P))))
                        res_max = max(res_max, res)
                        dens[(s_, i, j, rel, li)] = Density(coef, zlo, zhi, ylo, yhi, 1, 2)
                worst = max(worst, res_max / scale if scale > 0.0 else 0.0)
    return dens, worst


def si_decay_densities(G_flavor, U, masses, M, mediator, majorana):
    """decay_densities for the four vertices: dens[(k, l, li, hi)] the Density of the first daughter's share x in
    X -> nu_k nu_l (Dirac: nu_k nubar_l), hi = (h1 > 0) * 2 + (h2 > 0) the daughters' LF helicities, li the mediator's
    population; and W (si_pair_widths), the largest fit residual."""
    med = _si_med(mediator)
    cpl = si_coupling(G_flavor, U, med, majorana)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    W = si_pair_widths(G_flavor, U, m, M, med, majorana)
    lams = LAMS if med == "vector" else (None,)
    dens = {}
    worst = 0.0
    for k in range(3):
        for l in range(3):
            if W[k, l] <= 0.0:
                continue
            g = cpl[k, l]
            mk, ml = float(m[k]), float(m[l])
            p = _p_star(M, mk, ml)
            E1 = np.sqrt(p * p + mk * mk)
            E2 = np.sqrt(p * p + ml * ml)
            xlo, xhi = mk * mk / (E1 + p) / M, (E1 + p) / M
            ylo, yhi = ml * ml / (E2 + p) / M, (E2 + p) / M
            scale = 0.0
            res_max = 0.0
            for li, lam in enumerate(lams):
                for h1 in (-1, 1):
                    for h2 in (-1, 1):
                        v = np.array([msq_decay_si(M, mk, ml, c, lam, h1, h2, g, med, majorana) for c in _CHEB_C])
                        P = v[:, 0] * v[:, 1] * v[:, 2]                  # |M|^2 x (1-x)
                        coef, res = _chebyshev_fit(_CHEB_C, P)
                        scale = max(scale, float(np.max(np.abs(P))))
                        res_max = max(res_max, res)
                        dens[(k, l, li, (h1 > 0) * 2 + (h2 > 0))] = Density(coef, xlo, xhi, ylo, yhi, 1, 1)
            worst = max(worst, res_max / scale if scale > 0.0 else 0.0)
    return dens, W, worst


class SIMediator:
    """The mediator's tables for dp_si's transport of a secret-interaction point below 1 keV (2026-11-08; the section's
    notes), on the transport's grid E_bins: .Gamma its rest width into neutrino pairs (eV; the secret interactions'
    mediators have no other), .rate = Gamma M (the lab rate Gamma M / E_phys), .W the pair widths, .npol its
    populations (1 scalar, 3 vector), .n dp_si's states (3 Majorana, 6 Dirac), .nk the kinds of dp_si's populations the
    daughters land in (1 Majorana: both kinds in one flux; 2 Dirac: visible, sterile).
      .dec (NPOL, 2, 3, N + 2): per decayed mediator of each population, its neutrinos by kind (Majorana: 0 LF helicity
    -1, 1 +1; Dirac: 0 visible, 1 sterile) and mass state; each row's number its exact share and its energy at the
    bins' centres exact (energy_fix), together two neutrinos and the mediator's energy.
      .rows (2, 3, 3, 2, NPOL, N + 2): per decay of a parent of LF helicity index s, i -> j, the daughter kept (rel 0) or
    flipped (1), the mediator's offsets per population -- each population's share exact, their energies scaled together
    to the parent's less the daughter's as its exact density places it (the box table, as Mediator); .shares (2, 3, 3,
    2) each rel's share of nu_i -> nu_j (the traces'), .have the rows built.
      .arrays dp_si's: (prod (n, n, NPOL, N + 2) per decay of dp_si's state a into c, nprod, psum, dec (NPOL, nk, 3,
    N + 2), ndec, dsum, rate) -- Majorana: the neutrino parent's rows, kept and flipped added by their shares (as
    Mediator.fold), the kinds summed; Dirac: a = 3 s + i, c = 3 (s if kept else 1 - s) + j, the kinds visible and
    sterile."""

    def __init__(self, G_flavor, U, masses, M, E_bins, mediator="scalar", majorana=True):
        med = _si_med(mediator)
        E = np.asarray(E_bins, dtype=np.float64)
        N = len(E)
        h = float(np.sqrt(E[1] / E[0]))
        m = np.asarray(masses, dtype=np.float64)
        self.h, self.N, self.M = h, N, float(M)
        self.mediator, self.majorana = med, bool(majorana)
        self.npol = npol = si_npol(med)
        self.n = n = 3 if majorana else 6
        self.nk = 1 if majorana else 2
        K1 = N + 2
        # the decay back: the pair widths and the daughters per population
        ddens, W, res_d = si_decay_densities(G_flavor, U, m, M, med, majorana)
        self.W = W
        self.Gamma = float(W.sum())
        self.br_nu = 1.0 if self.Gamma > 0.0 else 0.0
        self.rate = self.Gamma * float(M)
        dec = np.zeros((NPOL, 2, 3, K1))
        dtarget = np.zeros((NPOL, 2, 3))
        for k in range(3):
            for l in range(3):
                if W[k, l] <= 0.0:
                    continue
                br = W[k, l] / self.Gamma
                for li in range(npol):
                    keys, tot = _significant({hi: ddens[(k, l, li, hi)].total for hi in range(4)})
                    for hi in keys:
                        d = ddens[(k, l, li, hi)]
                        w = br * d.total / tot
                        k1 = hi // 2                                 # nu_k at LF helicity +1: antineutrino / sterile
                        k2 = hi % 2 if majorana else 1 - hi % 2      # nu_l (Majorana); nubar_l sterile at -1 (Dirac)
                        dec[li, k1, k] += w * _table(d.cdf, d.lo, d.hi, h, N, K1)
                        dec[li, k2, l] += w * _table(d.cdf_other, d.ylo, d.yhi, h, N, K1)
                        dtarget[li, k1, k] += w * d.mean
                        dtarget[li, k2, l] += w * (1.0 - d.mean)
        for li in range(npol):
            for kd in range(2):
                for st in range(3):
                    dec[li, kd, st] = energy_fix(dec[li, kd, st], h, dtarget[li, kd, st])
        self.dec = np.ascontiguousarray(dec)
        # the production: per parent helicity, pair and rel, the mediator's share y = 1 - z by population
        pdens, res_p = si_production_densities(G_flavor, U, m, M, med, majorana)
        self.fit_residual = max(res_d, res_p)
        rows = np.zeros((2, 3, 3, 2, NPOL, K1))
        self.shares = np.zeros((2, 3, 3, 2))
        self.have = np.zeros((2, 3, 3, 2), dtype=np.bool_)
        for s_ in range(2):
            for i in range(3):
                for j in range(3):
                    if (s_, i, j, 0, 0) not in pdens:
                        continue
                    keys, tot = _significant({(rel, li): pdens[(s_, i, j, rel, li)].total for rel in range(2)
                                              for li in range(npol)})
                    for rel, li in keys:
                        self.shares[s_, i, j, rel] += pdens[(s_, i, j, rel, li)].total / tot
                    for rel in range(2):
                        if self.shares[s_, i, j, rel] <= 1e-14:
                            continue
                        ds = [pdens[(s_, i, j, rel, li)] if (rel, li) in keys else None for li in range(npol)]
                        tr = sum(d.total for d in ds if d is not None)
                        # the daughter as its exact density places it (its box table) and its energy at the bins' centres
                        d0 = pdens[(s_, i, j, rel, 0)]
                        dcdf = lambda z, ds=ds, tr=tr: sum(d.total * d.cdf(z) for d in ds  # noqa: E731
                                                           if d is not None) / tr
                        dtab = _table(dcdf, d0.lo, d0.hi, h, N, K1)
                        T = 1.0 - float(np.sum(dtab * h ** (-2.0 * np.arange(K1))))
                        ybar = sum(d.total * (1.0 - d.mean) for d in ds if d is not None) / tr
                        for li, d in enumerate(ds):
                            if d is None:
                                continue
                            row = d.total / tr * _table(d.cdf_other, d.ylo, d.yhi, h, N, K1)
                            rows[s_, i, j, rel, li] = energy_fix(row, h, d.total / tr * (1.0 - d.mean) * T / ybar)
                        self.have[s_, i, j, rel] = True
        self.rows = np.ascontiguousarray(rows)
        # dp_si's arrays
        prod = np.zeros((n, n, NPOL, K1))
        if majorana:
            for i in range(3):
                for j in range(3):
                    tot = 0.0
                    for rel in range(2):
                        w = self.shares[0, i, j, rel]
                        if not self.have[0, i, j, rel] or w <= 0.0:
                            continue
                        prod[i, j] += w * rows[0, i, j, rel]
                        tot += w
                    if tot > 0.0:
                        prod[i, j] /= tot
            decf = np.ascontiguousarray(self.dec.sum(axis=1)[:, None])
        else:
            for s_ in range(2):
                for rel in range(2):
                    db = s_ if rel == 0 else 1 - s_
                    for i in range(3):
                        for j in range(3):
                            if self.have[s_, i, j, rel]:
                                prod[3 * s_ + i, 3 * db + j] = rows[s_, i, j, rel]
            decf = self.dec
        self.prod = np.ascontiguousarray(prod)
        self.nprod = np.array([[_last(prod[a, c]) for c in range(n)] for a in range(n)], dtype=np.int64)
        self.dec_si = np.ascontiguousarray(decf)
        self.ndec = np.array([_last(decf[li]) for li in range(NPOL)], dtype=np.int64)
        self.arrays = (self.prod, self.nprod, _sums(self.prod, h), self.dec_si, self.ndec, _sums(self.dec_si, h),
                       float(self.rate))


def si_path_exponent(G_flavor, U, masses, M, mediator, majorana, E_min, H0, Omega_m, Omega_L, z_max):
    """The secret-interaction mediator's decay exponent over the whole path at energy E_min (path_exponent); 0 where no
    pair is open (M <= 2 m_lightest)."""
    m = np.asarray(masses, dtype=np.float64)
    if float(M) <= 2.0 * float(np.min(m)):
        return 0.0
    G_nu = float(si_pair_widths(G_flavor, U, m, M, mediator, majorana).sum())
    if not G_nu > 0.0:
        return 0.0
    return path_exponent(G_nu * float(M), float(E_min), H0, Omega_m, Omega_L, z_max)


_SI_CACHE = {}


def si_cached(G_flavor, U, masses, M, E_bins, mediator, majorana):
    """SIMediator(G_flavor, U, masses, M, E_bins, mediator, majorana), kept per arguments (a point runs many sources;
    the tables take ~1 s): the last 8."""
    E = np.asarray(E_bins, dtype=np.float64)
    key = (np.ascontiguousarray(G_flavor, dtype=np.complex128).tobytes(),
           np.ascontiguousarray(U, dtype=np.complex128).tobytes(),
           np.ascontiguousarray(masses, dtype=np.float64).tobytes(), float(M), float(E[0]), float(E[1]), len(E),
           _si_med(mediator), bool(majorana))
    if key not in _SI_CACHE:
        if len(_SI_CACHE) >= 8:
            _SI_CACHE.pop(next(iter(_SI_CACHE)))
        try:
            _SI_CACHE[key] = SIMediator(G_flavor, U, masses, M, E, mediator, majorana)
        except ValueError as err:
            # never a ValueError inside a fit (the analysis' objective would make the point a wall, chi^2 = 1e9):
            # a failure of the tables stops the run (numpy's LinAlgError is a ValueError)
            raise DecayBackUnsupported(f"dp_mediator: the secret interactions' decay-back tables failed at M = {M!r} "
                                       f"eV, masses {np.asarray(masses).tolist()}, {mediator} "
                                       f"{'Majorana' if majorana else 'Dirac'}: {err}") from err
    return _SI_CACHE[key]
