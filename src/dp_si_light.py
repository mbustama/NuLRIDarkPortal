"""
dp_si_light.py -- the secret interactions below 1 keV (2026-10-25; notes, "Secret interactions below 1 keV").

Below 1 keV the resonance falls below the band and the secret interactions add, to dp_si's hard 2 -> 2 and pair
channels and the vector's quasi-elastic conversions, the processes that the light mediator opens:

  - the decay into the mediator, nu_i -> nu_j + X, for M < m_i - m_j, for the four models (scalar or vector,
    Majorana or Dirac neutrinos); this module's rates and daughter spectra;
  - for the vector, the relics' Debye screening of the forward peak (M_t^2 = M^2 + m_D^2) and the energy the beam
    loses when a soft scattering changes the relic's mass state (dp_relic_loss), as in the dark-portal phase;
  - where the models hold: the vector's longitudinal mode (perturbative unitarity of its Goldstone couplings,
    dp_unitarity; here also for Dirac neutrinos) and the scalar's background field sourced by the relics' scalar
    density (scalar_field_shift).

The models have no potential (the mediator couples to neutrinos only; the relics' own first-order potential vanishes
for the vector, the relics holding as many neutrinos as antineutrinos, and its second-order part is below 1e-3 g^2 of
Delta m^2_21 at every mass; notes). So the propagation eigenstates are the mass states and everything here is in the
mass basis.

Couplings (as dp_si.mass_basis_couplings_model): the scalar with Majorana neutrinos lambda = U^T G U (complex
symmetric; even a flavor-universal G is off-diagonal at delta_CP != 0, so it decays), with Dirac neutrinos
h = U^dag G U (Hermitian; the right-handed neutrinos rotate as the left-handed ones); the vector G' = U^dag G U for
both. Vertices of nu_i -> nu_j + X (the amplitude u-bar_j Gamma u_i):
  scalar, Majorana:  lambda_ji P_L + lambda*_ji P_R
  scalar, Dirac:     h_ji (both chiralities: the scalar flips chirality, a visible nu_L into a sterile nu_R)
  vector, Majorana:  gamma^mu (G'_ji P_L - G'*_ji P_R)    (dp_decay, unchanged)
  vector, Dirac:     gamma^mu G'_ji P_L
Rest-frame rates (spin-averaged; checked against the traces in the tests), p* the daughter's momentum:
  scalar, Majorana:  p* / (8 pi m_i^2) [|lambda_ji|^2 (m_i^2 + m_j^2 - M^2) + 2 m_i m_j Re(lambda_ji^2)]
  scalar, Dirac:     p* / (8 pi m_i^2) |h_ji|^2 ((m_i + m_j)^2 - M^2)
  vector:            dp_decay.rest_rates (Majorana; Dirac the P_L piece alone)
The scalar's rate has no 1/M^2: there is no longitudinal mode. The Majorana phases are zero (construct_PMNS), as for
dp_decay; with massive daughters the m_i m_j terms depend on them.

Daughters. A parent in flight has a light-front helicity h; its daughter keeps it ('kept') or flips it ('flipped').
The spectra in the daughter's share z of the parent's energy are covariant traces with light-front spin vectors, as
dp_decay.daughter_spectra (the same construction, general vertex and polarization sum), dGamma/dz = a + b z + c/z on
the window [z-, z+] of two-body kinematics (fitted at nine angles; the tests check the residual and the totals).
Majorana: both daughters are visible (the flipped one of the other kind) and dp_si follows one flux holding both
kinds. Dirac: a flipped daughter of a visible neutrino has the sterile helicity (h = +1 for a neutrino; the scalar
flips chirality, so for it most daughters are sterile), and a sterile neutrino decays as fast as a visible one (the
total rate does not depend on the spin), its flipped daughters visible again: dp_si follows six states, visible
(0-2) and sterile (3-5), the kinds alike (CP: the rates depend on |coupling|^2 and Re(coupling^2) only).
"""
import numpy as np

import dp_decay
from dp_core import njit
from dp_si import _phi_functions as _phi_functions_nb

_GAM = dp_decay._GAM
_G5 = dp_decay._G5
_PL = dp_decay._PL
_PR = dp_decay._PR
_ETA = dp_decay._ETA
_rho_lf = dp_decay._rho_lf

MEDIATORS = ("scalar", "vector")


def mass_couplings(G_flavor, U, mediator, majorana):
    """The decay vertex's coupling matrix in the mass basis: lambda = U^T G U (scalar, Majorana), else U^dag G U. For
    the couplings conjugated by U^dag ... U an identity part c I (c = G[0, 0], exact) is taken out first, so that a
    flavor-universal coupling leaves nothing off the diagonal (dp_decay.rest_rates); U^T (c I) U is not diagonal at
    delta_CP != 0 and is kept."""
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    med = str(mediator).strip().lower()
    if med == "scalar" and majorana:
        return U.T @ G @ U
    c = G[0, 0].real
    return U.conj().T @ (G - c * np.eye(3)) @ U


def _vertex(g, mediator, majorana):
    """The 4 x 4 vertex Gamma (scalar) or V with Gamma^mu = gamma^mu V (vector) for the coupling g = coupling[j, i]."""
    if mediator == "scalar":
        return g * _PL + np.conj(g) * _PR if majorana else g * np.eye(4, dtype=np.complex128)
    return g * _PL - np.conj(g) * _PR if majorana else g * _PL


def _msq(p, k, mi, mj, M, h, hp, V, mediator, kplus):
    """|M|^2 of nu_i(p, light-front helicity h) -> nu_j(k, hp) + X, summed over the mediator's polarizations."""
    A = _rho_lf(k, mj, hp, kplus)
    B = _rho_lf(p, mi, h, mi)
    if mediator == "scalar":
        Vb = _GAM[0] @ V.conj().T @ _GAM[0]
        return float(np.real(np.trace(A @ V @ B @ Vb)))
    q = p - k
    Gm = np.einsum("mab,bc->mac", _GAM, V)
    Gb = np.einsum("ab,mcb,cd->mad", _GAM[0], Gm.conj(), _GAM[0])
    T = np.einsum("ab,mbc,cd,nda->mn", A, Gm, B, Gb)
    Pp = -_ETA + np.outer(_ETA @ q, _ETA @ q) / (M * M)
    return float(np.real(np.sum(T * Pp)))


def _p_star(mi, mj, M):
    return np.sqrt(max((mi * mi - (mj + M) ** 2) * (mi * mi - (mj - M) ** 2), 0.0)) / (2.0 * mi)


def rest_rates(G_flavor, U, masses, M, mediator="vector", majorana=True):
    """Gam[i, j], the rest-frame rate (eV) of nu_i -> nu_j + X, spin-averaged, for the four models (module notes).
    The vector is dp_decay.rest_rates itself."""
    med = str(mediator).strip().lower()
    if med not in MEDIATORS:
        raise ValueError(f"mediator must be 'scalar' or 'vector', got {mediator!r}")
    if med == "vector":
        return dp_decay.rest_rates(G_flavor, U, masses, M, majorana)[0]
    lam = mass_couplings(G_flavor, U, med, majorana)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    Gam = np.zeros((3, 3))
    if M <= 0.0:
        return Gam
    for i in range(3):
        for j in range(3):
            if i == j or m[i] <= m[j] + M:
                continue
            mi, mj = m[i], m[j]
            g = lam[j, i]
            if g == 0.0:
                continue
            p = _p_star(mi, mj, M)
            if majorana:
                br = abs(g) ** 2 * (mi * mi + mj * mj - M * M) + 2.0 * mi * mj * (g * g).real
            else:
                br = abs(g) ** 2 * ((mi + mj) ** 2 - M * M)
            Gam[i, j] = max(p / (8.0 * np.pi * mi * mi) * br, 0.0)
    return Gam


def daughter_spectra(G_flavor, U, masses, M, mediator="vector", majorana=True, return_totals=False):
    """spec[s, i, j, rel] = (z-, z+, a, b, c): dGamma/dz / Gamma_ij = a + b z + c / z on [z-, z+] for a parent of
    light-front helicity -1 (s = 0) or +1 (s = 1), its daughter kept (rel 0) or flipped (rel 1); for each s the two
    integrate to 1 (dp_decay.daughter_spectra's convention; for Majorana s = 1 is the antineutrino, for Dirac the
    sterile neutrino). The vector with Majorana neutrinos is dp_decay.daughter_spectra itself. With return_totals
    also tot[s, i, j], each parent helicity's own total rate (eV) from the traces (the tests: equal to rest_rates')."""
    med = str(mediator).strip().lower()
    if med not in MEDIATORS:
        raise ValueError(f"mediator must be 'scalar' or 'vector', got {mediator!r}")
    if med == "vector" and majorana and not return_totals:
        return dp_decay.daughter_spectra(G_flavor, U, masses, M)
    lam = mass_couplings(G_flavor, U, med, majorana)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    spec = np.zeros((2, 3, 3, 2, 5))
    tot_out = np.zeros((2, 3, 3))
    if M <= 0.0:
        return (spec, tot_out) if return_totals else spec
    cth = np.cos(np.linspace(np.pi, 0.0, 9))
    for i in range(3):
        for j in range(3):
            if i == j or m[i] <= m[j] + M or lam[j, i] == 0.0:
                continue
            mi, mj = float(m[i]), float(m[j])
            pf = _p_star(mi, mj, M)
            Ej = np.sqrt(pf * pf + mj * mj)
            emp = mj * mj / (Ej + pf)
            zm, zp = emp / mi, (Ej + pf) / mi
            p = np.array([mi, 0.0, 0.0, 0.0])
            z = (emp + pf * (1.0 + cth)) / mi
            V = _vertex(lam[j, i], med, majorana)
            for s_, h in ((0, -1.0), (1, 1.0)):
                coef = []
                for hp in (h, -h):
                    f = np.array([_msq(p, np.array([Ej, pf * np.sqrt(max(1.0 - c * c, 0.0)), 0.0, pf * c]),
                                       mi, mj, M, h, hp, V, med, emp + pf * (1.0 + c)) for c in cth]) / (16.0 * np.pi * mi)
                    if zm > 0.0:
                        # the 1/z column scaled by z-: at z- ~ 1e-16 (a nearly massless daughter) the unscaled column
                        # reached 1e16 and the least squares cut the other two columns as below its cutoff
                        B = np.stack([np.ones_like(z), z, zm / z], axis=1)
                        a_, b_, c_ = np.linalg.lstsq(B, f, rcond=None)[0]
                        c_ = c_ * zm
                    else:
                        B = np.stack([np.ones_like(z), z], axis=1)
                        (a_, b_), c_ = np.linalg.lstsq(B, f, rcond=None)[0], 0.0
                    coef.append((a_, b_, c_))
                ints = [a_ * (zp - zm) + 0.5 * b_ * (zp * zp - zm * zm) + (c_ * np.log(zp / zm) if c_ != 0.0 else 0.0)
                        for a_, b_, c_ in coef]
                tot = ints[0] + ints[1]
                tot_out[s_, i, j] = tot
                if tot <= 0.0:
                    continue
                for rel in range(2):
                    a_, b_, c_ = coef[rel]
                    spec[s_, i, j, rel] = (zm, zp, a_ / tot, b_ / tot, c_ / tot)
    return (spec, tot_out) if return_totals else spec


def transport_arrays(Gam, spec, masses, E, majorana):
    """dp_si's decay channel (_si_sweep_cols' dec_* arrays) for n = 3 states (Majorana: one flux holding both kinds,
    the kept and flipped spectra of the neutrino parent added, as dp_si did for the vector) or n = 6 (Dirac: visible
    0-2, sterile 3-5; a visible parent's kept daughter visible and its flipped one sterile, a sterile parent's kept
    daughter sterile and its flipped one visible). Gam (3, 3) rest rates, spec as daughter_spectra, E the transport's
    geometric grid. Returns dict(rate, br, own, kfull, kex, fex, mom), each over (parent state, daughter state)."""
    m = np.asarray(masses, dtype=np.float64)
    Gam = np.asarray(Gam, dtype=np.float64)
    n = 3 if majorana else 6
    N = len(E)
    h = float(np.sqrt(E[1] / E[0]))
    tot = Gam.sum(axis=1)
    rate = np.zeros(n)
    br = np.zeros((n, n))
    for i in range(3):
        if tot[i] <= 0.0:
            continue
        for b in range(n // 3):
            rate[3 * b + i] = tot[i] * m[i]
    # the (parent, daughter) spectra as dp_decay's (2, 3, 3, 2, 5) blocks: one window_plan per block
    blocks = []
    if majorana:
        sp = np.zeros_like(spec)
        sp[0, :, :, 0, :2] = spec[0, :, :, 0, :2]
        sp[0, :, :, 0, 2:] = spec[0, :, :, 0, 2:] + spec[0, :, :, 1, 2:]
        blocks.append(((0, 0), sp, (0, 0)))                         # (parent block, daughter block), spec, (s, rel)
        for i in range(3):
            if tot[i] > 0.0:
                br[i, :3] = Gam[i] / tot[i]
    else:
        for pb, s_ in ((0, 0), (1, 1)):
            for rel in (0, 1):
                db = pb if rel == 0 else 1 - pb
                blocks.append(((pb, db), spec, (s_, rel)))
                for i in range(3):
                    if tot[i] <= 0.0:
                        continue
                    for j in range(3):
                        sh = _share(spec[s_, i, j, rel])
                        br[3 * pb + i, 3 * db + j] = Gam[i, j] / tot[i] * sh
    kfull = np.zeros((n, n, 2), dtype=np.int64)
    kfull[:, :, 0] = 1
    kex = np.full((n, n, dp_decay.NSLOT), -1, dtype=np.int64)
    fex = np.zeros((n, n, dp_decay.NSLOT))
    mom = np.zeros((n, n, 3))
    own = np.zeros((n, n))
    for (pb, db), sp, (s_, rel) in blocks:
        one = np.zeros((2, 3, 3, 2, 5))
        one[0, :, :, 0, :2] = sp[s_, :, :, rel, :2]
        sh = np.ones((3, 3))
        if not majorana:
            # each block's spectrum normalized to 1 over its window (its share of Gamma_ij is in br)
            for i in range(3):
                for j in range(3):
                    s_ij = _share(sp[s_, i, j, rel])
                    sh[i, j] = s_ij if s_ij > 0.0 else 1.0
        one[0, :, :, 0, 2:] = sp[s_, :, :, rel, 2:] / sh[:, :, None]
        kf = np.zeros((2, 3, 3, 2, 2), dtype=np.int64)
        kx = np.zeros((2, 3, 3, 2, dp_decay.NSLOT), dtype=np.int64)
        fx = np.zeros((2, 3, 3, 2, dp_decay.NSLOT))
        dp_decay.window_plan(one, h, N, kf, kx, fx)
        a0, c0 = 3 * pb, 3 * db
        kfull[a0:a0 + 3, c0:c0 + 3] = kf[0, :, :, 0]
        kex[a0:a0 + 3, c0:c0 + 3] = kx[0, :, :, 0]
        fex[a0:a0 + 3, c0:c0 + 3] = fx[0, :, :, 0]
        mom[a0:a0 + 3, c0:c0 + 3, 0] = one[0, :, :, 0, 4]
        mom[a0:a0 + 3, c0:c0 + 3, 1] = one[0, :, :, 0, 2]
        mom[a0:a0 + 3, c0:c0 + 3, 2] = one[0, :, :, 0, 3]
        own[a0:a0 + 3, c0:c0 + 3] = np.where(kx[0, :, :, 0] == 0, fx[0, :, :, 0], 0.0).sum(axis=-1)
    return dict(rate=rate, br=br, own=own, kfull=kfull, kex=kex, fex=fex, mom=mom)


def _share(sp):
    """The integral of a + b z + c / z over its window (a spectrum row (z-, z+, a, b, c))."""
    zm, zp, a, b, c = sp
    if zp <= zm:
        return 0.0
    return a * (zp - zm) + 0.5 * b * (zp * zp - zm * zm) + (c * np.log(zp / zm) if c != 0.0 and zm > 0.0 else 0.0)


# ------------------------------------------------------------------------------------------------------------------
# dp_si's inputs below 1 keV

SUBKEV_M_EV = 1e3            # below it the processes of this module act (dark_portal_solver.SI_COSMOLOGY_MIN_M_EV)
DECAY_MIN_H = 1e-8           # the decays are kept where the lab rate at the grid's lowest energy exceeds this x H0
LOSS_MIN = 1e-6              # the relic loss is kept where it moves the grid's lowest energy by more than this
LOSS_Z_NODES = 101           # C(z) on these nodes, uniform in z (linear interpolation in dp_si._loss_columns)


def _mean_left_inv_E(m, T, n=600):
    """<(1 + v) / (2 E)> over a Fermi-Dirac momentum distribution: a Dirac relic of the populated helicity couples to
    a left-handed current through its left-chiral share (1 + v)/2 (1/2 at rest, 1 relativistic)."""
    x = np.linspace(1e-5, 40.0, n)
    f = x * x / (np.exp(x) + 1.0)
    p = x * T
    E = np.sqrt(p * p + m * m)
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(f * 0.5 * (1.0 + p / E) / E, x) / trap(f, x))


def loss_coefficient(G_flavor, U, masses, eta, T_nu0, z, M, mD2, majorana=True):
    """C(z) (eV^2) of the relic mass-change loss, dE/dt = -C (G'G'^dag)_aa (dp_relic_loss.loss_coefficient, the
    vector with Majorana relics); Dirac relics couple through their left-chiral share, <1/E_r> -> <(1+v)/(2E_r)>
    (exact at rest and for relativistic relics; between them the share is averaged over the relics' directions
    before the 1/E weight, an approximation: the loss acts only within about a decade of the unitarity line)."""
    import dp_relic_loss
    if majorana:
        return dp_relic_loss.loss_coefficient(G_flavor, U, masses, eta, T_nu0, z, M, mD2)
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    Gp = U.conj().T @ (G - G[0, 0].real * np.eye(3)) @ U
    n_hel0 = 3.0 * dp_relic_loss.ZETA3 * T_nu0 ** 3 / (4.0 * np.pi ** 2)
    acc = 0.0
    for r in range(3):
        if eta[r] <= 0.0:
            continue
        Tr = T_nu0 * (1.0 + z) * eta[r] ** (1.0 / 3.0)
        inv = _mean_left_inv_E(m[r], Tr)
        for s in range(3):
            if s != r:
                acc += eta[r] * abs(Gp[s, r]) ** 2 * 0.5 * (m[s] ** 2 - m[r] ** 2) * inv
    return 2.0 * n_hel0 * (1.0 + z) ** 3 * acc / (4.0 * np.pi * (M * M + mD2))


def subkev_inputs(G_flavor, U, masses, M, mediator, majorana, T_nu0, eta_mass, H0, Omega_m, Omega_L, z_max, E_min,
                  decay_on=True, relic_scattering=True):
    """dp_si's inputs for a secret-interaction point below 1 keV (2026-10-25): dict(M_coll, eta, decay, relic_loss,
    meta).

    eta: the relics per mass state after their own decays by z = 1 (dp_decay.relic_composition, with this model's
    rates; as the dark-portal phase). decay: dict(Gam, spec, majorana) for dp_si (rest rates and daughter spectra),
    or None where the lab rate at E_min stays below DECAY_MIN_H x H0. M_coll, the mass in dp_si's collision terms:
    vector, M_t = sqrt(M^2 + m_D^2) with the relics' Debye mass today (dp_absorption.debye_mass_sq, Majorana or Dirac,
    the relics after their decays; the bare M where m_D^2 < 1e-10 M^2), as the dark-portal model's runs through
    dp_si; scalar, max(M, m_heaviest): its channels depend on M only through M^2 / s (O(M^2/s) <= 1e-12 at the floor)
    but for the pair production's collinear logarithm ln(s / M^2), which with massive neutrinos is cut by the
    exchanged neutrino's mass, not by M (here at the heaviest mass; the region where this matters lies four or more
    decades in g inside the one the decays exclude). relic_loss (vector only): (z nodes, C(z), (G'G'^dag)_aa) for
    dp_si._loss_columns, or None where it moves E_min by less than LOSS_MIN over the path."""
    import dp_absorption
    import dp_relic_loss
    med = str(mediator).strip().lower()
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    eta = np.asarray(eta_mass, dtype=np.float64).copy()
    M = float(M)
    meta = {}
    Gam = rest_rates(G, U, m, M, med, majorana) if decay_on else np.zeros((3, 3))
    if np.any(Gam > 0.0):
        eta = dp_decay.relic_composition(Gam, m, eta, T_nu0, H0, Omega_m, Omega_L)
    decay = None
    lab = float(np.max(Gam.sum(axis=1) * m)) / float(E_min) if np.any(Gam > 0.0) else 0.0
    meta["decay_lab_over_H0"] = lab / H0
    if lab > DECAY_MIN_H * H0:
        decay = dict(Gam=Gam, spec=daughter_spectra(G, U, m, M, med, majorana), majorana=bool(majorana))
    relic_loss = None
    if med == "scalar":
        M_coll = max(M, float(np.max(m)))
    else:
        Gp = U.conj().T @ (G - G[0, 0].real * np.eye(3)) @ U + G[0, 0].real * np.eye(3)
        G2d = np.abs(np.diag(Gp)) ** 2
        mD0 = dp_absorption.debye_mass_sq(G2d, m, T_nu0, eta, bool(majorana))
        M_coll = float(np.sqrt(M * M + mD0)) if mD0 > 1e-10 * M * M else M
        meta["debye_sq_over_M2"] = mD0 / (M * M)
        if relic_scattering:
            z_nodes = np.linspace(0.0, float(z_max), LOSS_Z_NODES)
            C = np.array([loss_coefficient(G, U, m, eta, T_nu0, zk, M,
                                           dp_absorption.debye_mass_sq(G2d, m, T_nu0 * (1.0 + zk), eta, bool(majorana)),
                                           majorana) for zk in z_nodes])
            beam = np.real(np.diag(Gp @ Gp.conj().T)).copy()
            # the loss over the path, int C (G'G'^dag)_aa dz / ((1+z)^2 H), at the grid's lowest energy
            Hz = H0 * np.sqrt(Omega_m * (1.0 + z_nodes) ** 3 + Omega_L)
            trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            path = float(trap(np.abs(C) * float(np.max(beam)) / ((1.0 + z_nodes) ** 2 * Hz), z_nodes))
            meta["loss_over_E_min"] = path / float(E_min)
            if path > LOSS_MIN * float(E_min):
                relic_loss = (z_nodes, C, beam)
    meta["M_coll"] = M_coll
    return dict(M_coll=M_coll, eta=eta, decay=decay, relic_loss=relic_loss, meta=meta)


# ------------------------------------------------------------------------------------------------------------------
# The Galactic lines of sight (2026-10-25): the neutrinos emitted in the Galaxy cross 0.05-30 kpc to the Sun at z = 0.
# Without a potential every line of sight evolves alike but for where along it the emission sits, so each output pixel
# is one line of sight with its emission's distance distribution (its children's, intensity-weighted: exact by
# linearity) and the processes the cosmological phase has below 1 keV: the decays, the vector's soft conversions (the
# jumps, with their Zeno factor, as the dark-portal phase's Galactic paths) and the relic mass-change loss. The hard
# scatterings are left out on the Galactic paths, as they were for the secret interactions at 1 keV and above (the
# Galaxy's relic column is ~1e-5 of the cosmological one).

KPC_EV_INV = 3.0856775814913673e21 / 1.973269804e-5      # 1 kpc in eV^-1 (dp_galactic_path.KPC_EV_INV)
GAL_JUMP_MIN = 1e-4          # as dark_portal_solver.SOFT_JUMP_MIN: conversions kept above this over 30 kpc
GAL_DECAY_MIN = 1e-6         # decays and loss kept above this over 30 kpc at the grid's lowest energy


@njit(cache=True)
def _segment(F, E, dE, S, L, lam, Rc, br, own, kfull, kex, fex, mom, use_dec):
    """One distance segment of length L (eV^-1) at z = 0, in place: F (N, n, C) the populations per unit energy;
    S (N, n, C) the emission per unit length (constant over the segment); lam (N, n) the decay rates per unit length;
    Rc (N, n, n) the conversions a -> c per unit length [a, c]; br, own, kfull, kex, fex, mom the decays' branching
    and windows (transport_arrays). Each energy bin from the top: the daughters of the bins above (constant over the
    segment, as dp_si's sweep), then the bin's conversions, decays and sources solved exactly over the segment (the
    matrix exponential and its phi functions), its decays' daughters passed down."""
    N = F.shape[0]
    n = F.shape[1]
    C = F.shape[2]
    hgeo = np.sqrt(E[1] / E[0])
    Lgeo = 2.0 * np.log(hgeo)
    uinv = Lgeo / (hgeo - 1.0 / hgeo)
    Nd = N if use_dec else 1
    Sm = np.zeros((n, C, 3))
    addm = np.zeros((Nd, n, C, 3))
    subm = np.zeros((Nd, n, C, 3))
    pend = np.zeros((Nd, n, C))
    Mx = np.zeros((n, n))
    Ssrc = np.zeros((n, C))
    avg = np.zeros((n, C))
    Fn = np.zeros(n)
    for j in range(N - 1, -1, -1):
        for c in range(n):
            for k in range(C):
                Ssrc[c, k] = S[j, c, k]
        if use_dec:
            elo = E[j] / hgeo
            ehi = E[j] * hgeo
            dEj = ehi - elo
            q2j = 0.5 * (ehi * ehi - elo * elo)
            for c in range(n):
                for k in range(C):
                    for mo in range(3):
                        Sm[c, k, mo] += addm[j, c, k, mo]
                    Ssrc[c, k] += (Sm[c, k, 0] * Lgeo + Sm[c, k, 1] * dEj + Sm[c, k, 2] * q2j + pend[j, c, k]) / dEj
                    for mo in range(3):
                        Sm[c, k, mo] -= subm[j, c, k, mo]
        act = False
        for a in range(n):
            out = lam[j, a]
            for c in range(n):
                Mx[c, a] = 0.0
            for c in range(n):
                if c != a:
                    out += Rc[j, a, c]
                    Mx[c, a] += Rc[j, a, c] * L
            if use_dec and lam[j, a] > 0.0:
                for c in range(n):
                    Mx[c, a] += lam[j, a] * br[a, c] * own[a, c] * L
            Mx[a, a] -= out * L
            if out != 0.0:
                act = True
        if not act:
            for c in range(n):
                for k in range(C):
                    F[j, c, k] += Ssrc[c, k] * L
            continue
        Em, P1, P2 = _phi_functions_nb(Mx)
        for k in range(C):
            for a in range(n):
                v = 0.0
                for b in range(n):
                    v += P1[a, b] * F[j, b, k] + P2[a, b] * Ssrc[b, k] * L
                avg[a, k] = v
            for c in range(n):
                fn = 0.0
                for b in range(n):
                    fn += Em[c, b] * F[j, b, k] + P1[c, b] * Ssrc[b, k] * L
                Fn[c] = fn
            for c in range(n):
                F[j, c, k] = Fn[c]
        if use_dec:
            # the decays over the segment: rate x the population integrated over it (L avg), daughters below the bin
            for k in range(C):
                for a in range(n):
                    if lam[j, a] <= 0.0:
                        continue
                    rate = lam[j, a] * avg[a, k] * dE[j]       # per unit length, over the segment's mean
                    for c in range(n):
                        if br[a, c] == 0.0:
                            continue
                        nd = rate * br[a, c]
                        for x in range(kex.shape[2]):
                            kk = kex[a, c, x]
                            if kk < 0:
                                break
                            if kk >= 1 and j - kk >= 0:
                                pend[j - kk, c, k] += nd * fex[a, c, x]
                        k1 = kfull[a, c, 0]
                        k2 = kfull[a, c, 1]
                        if k1 <= k2 and j - k1 >= 0:
                            v0 = nd * mom[a, c, 0]
                            v1 = nd * mom[a, c, 1] * uinv / E[j]
                            v2 = nd * mom[a, c, 2] / (E[j] * E[j])
                            addm[j - k1, c, k, 0] += v0
                            addm[j - k1, c, k, 1] += v1
                            addm[j - k1, c, k, 2] += v2
                            if j - k2 >= 0:
                                subm[j - k2, c, k, 0] += v0
                                subm[j - k2, c, k, 1] += v1
                                subm[j - k2, c, k, 2] += v2


def galactic_rates(E, G_flavor, U, masses, M, mediator, majorana, T_nu0, eta_cosmic, H0, Omega_m, Omega_L,
                   decay_on=True, relic_scattering=True):
    """The processes on the Galactic paths at z = 0 (per unit length, eV): dict(lam (N, n), Rc (N, n, n), dec
    (transport_arrays or None), loss (C(0) (G'G'^dag)_aa per visible state, eV^2, or None), n, meta), or None where none
    acts over 30 kpc (conversions below GAL_JUMP_MIN, decays and loss below GAL_DECAY_MIN at the grid's lowest
    energy). The relics are today's, after their own decays (dp_decay.relic_composition, z_ref = 0)."""
    import dp_absorption
    E = np.asarray(E, dtype=np.float64)
    med = str(mediator).strip().lower()
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    n = 3 if majorana else 6
    N = len(E)
    L_max = 30.25 * KPC_EV_INV
    meta = {}
    Gam = rest_rates(G, U, m, M, med, majorana) if decay_on else np.zeros((3, 3))
    eta = np.asarray(eta_cosmic, dtype=np.float64)
    if np.any(Gam > 0.0):
        eta = dp_decay.relic_composition(Gam, m, eta, T_nu0, H0, Omega_m, Omega_L, z_ref=0.0)
    lam = np.zeros((N, n))
    dec = None
    if np.any(Gam > 0.0) and float(np.max(Gam.sum(axis=1) * m)) / E[0] * L_max >= GAL_DECAY_MIN:
        spec = daughter_spectra(G, U, m, M, med, majorana)
        dec = transport_arrays(Gam, spec, m, E, majorana)
        lam = np.ascontiguousarray(dec["rate"][None, :] / E[:, None])
    meta["decay_depth_30kpc"] = float(np.max(lam[0])) * L_max if dec is not None else 0.0
    Rc = np.zeros((N, n, n))
    loss = None
    if med == "vector" and relic_scattering:
        Gp = U.conj().T @ (G - G[0, 0].real * np.eye(3)) @ U + G[0, 0].real * np.eye(3)
        G2d = np.abs(np.diag(Gp)) ** 2
        mD2 = dp_absorption.debye_mass_sq(G2d, m, T_nu0, eta, bool(majorana))
        n_hel0 = 3.0 * 1.2020569031595942 * T_nu0 ** 3 / (4.0 * np.pi ** 2)
        share = np.ones(3)
        if not majorana:
            share = np.array([_mean_left_inv_E(m[r], T_nu0 * eta[r] ** (1.0 / 3.0)) /
                              _mean_inv_E(m[r], T_nu0 * eta[r] ** (1.0 / 3.0)) if eta[r] > 0 else 0.0 for r in range(3)])
        GG = Gp @ Gp.conj().T
        relic = float(np.sum(eta * share * np.real(np.diag(GG))))
        unit = 2.0 * n_hel0 * relic / (4.0 * np.pi * (M * M + mD2))
        off = np.abs(Gp - np.diag(np.diag(Gp))) ** 2
        if unit * float(np.max(off)) * L_max >= GAL_JUMP_MIN:
            G2 = Gp @ Gp
            widths = lam[:, :3]
            for a in range(3):
                for c in range(3):
                    if a == c or off[c, a] == 0.0:
                        continue
                    om = (m[a] ** 2 - m[c] ** 2) / (2.0 * E)
                    D = unit * (0.5 * (G2[a, a].real + G2[c, c].real) - (Gp[a, a] * Gp[c, c]).real) \
                        + 0.5 * (widths[:, a] + widths[:, c])
                    Rc[:, a, c] = unit * off[c, a] * om * om / (om * om + D * D)
        meta["jump_depth_30kpc"] = float(np.max(Rc)) * L_max
        Cl = loss_coefficient(G, U, m, eta, T_nu0, 0.0, M, mD2, majorana)
        beam = np.real(np.diag(GG))
        if abs(Cl) * float(np.max(beam)) * L_max >= GAL_DECAY_MIN * E[0]:
            loss = Cl * beam
        meta["loss_30kpc_over_E_min"] = abs(Cl) * float(np.max(beam)) * L_max / E[0]
    if dec is None and not np.any(Rc > 0.0) and loss is None:
        return None
    return dict(lam=np.ascontiguousarray(lam), Rc=np.ascontiguousarray(Rc), dec=dec, loss=loss, n=n, meta=meta)


def _mean_inv_E(m, T, n=600):
    import dp_relic_loss
    return dp_relic_loss.mean_inv_E(m, T, n)


GAL_SVD_TOL = 1e-10          # the emission's spectral shapes kept to this fraction of the largest singular value


def emission_rows(template, E, nside_out):
    """(X (N_pix, N_s, N_E), I_out (N_pix, N_E)): each output pixel's Galactic emission per distance bin, the mean
    of its work-resolution children's I [w q_gas + (1 - w) q_src] (dp_galactic's intensity, gas share and distance
    weights; the pixel's intensity is the children's mean, as dp_galactic.degrade), and that intensity."""
    import dp_galactic
    import dp_sky
    E_GeV = np.asarray(E, dtype=np.float64) / 1e9
    work = max(int(nside_out), 16)
    I = dp_galactic.intensity(template, E_GeV, work)
    w = dp_galactic.gas_share(template, E_GeV, work)
    q_gas, q_src = dp_galactic.emission_weights(work)
    th, ph = dp_sky.pix2ang_ring(work, np.arange(dp_sky.nside_to_npix(work)))
    parent = dp_sky.ang2pix_ring(int(nside_out), th, ph)
    npo = dp_sky.nside_to_npix(int(nside_out))
    X = np.zeros((npo, q_gas.shape[1], len(E)))
    Iout = np.zeros((npo, len(E)))
    for o in range(npo):
        ch = np.nonzero(parent == o)[0]
        X[o] = (q_gas[ch].T @ (I[ch] * w[ch]) + q_src[ch].T @ (I[ch] * (1.0 - w[ch]))) / len(ch)
        Iout[o] = I[ch].mean(axis=0)
    return X, Iout


_SHAPES = {}


def _emission_shapes(template, E, nside_out):
    """(coef (N_pix, N_s, r), shapes (r, N_E), I_out (N_pix, N_E), dropped): the SVD of the emission rows
    (emission_rows, scaled per energy by the sky-mean intensity), kept per template, grid and N_side (it depends on
    nothing else)."""
    import hashlib
    E = np.ascontiguousarray(E, dtype=np.float64)
    key = (str(template), hashlib.blake2b(E.tobytes(), digest_size=16).hexdigest(), int(nside_out))
    hit = _SHAPES.get(key)
    if hit is not None:
        return hit
    X, Iout = emission_rows(template, E, nside_out)            # (P, K, N), (P, N)
    P_, K, N = X.shape
    Ibar = np.maximum(Iout.mean(axis=0), 1e-300)
    rows = (X / Ibar[None, None, :]).reshape(P_ * K, N)
    u_, sv, vt = np.linalg.svd(rows, full_matrices=False)
    r = int(np.sum(sv > GAL_SVD_TOL * sv[0])) if sv[0] > 0 else 0
    dropped = float(sv[r] / sv[0]) if r < len(sv) else 0.0
    coef = np.ascontiguousarray((u_[:, :r] * sv[:r]).reshape(P_, K, r))
    shapes = np.ascontiguousarray(vt[:r] * Ibar[None, :])
    if len(_SHAPES) >= 4:
        _SHAPES.pop(next(iter(_SHAPES)))
    _SHAPES[key] = (coef, shapes, Iout, dropped)
    return _SHAPES[key]


def galactic_transfer(E, U, rates, template, nside_out, compositions, diag=None):
    """Per source composition (each (3,), flavor, normalized), the Galactic flavor flux at the Sun per unit emitted
    all-flavor flux per kind, per output pixel: a list of (N_pix, N_E, 3); with no process (rates None) it is
    dp_galactic.averaged_transfer(U) @ f in every pixel. rates: galactic_rates'.

    Every line of sight evolves alike (no potential, the relics uniform), so the emission's spectra (per pixel and
    distance bin, scaled per energy by the sky mean) are split into a few shapes by an SVD (kept to GAL_SVD_TOL of the
    largest singular value), and each shape, in each mass state, is evolved once from the Sun outward as a constant
    source (_segment per distance bin, the relic loss in two halves around each): the populations at consecutive
    edges differ by what each bin's uniform emission sends to the Sun (linear and time-invariant: the arrival from
    emission over [0, L] is the state reached from zero with that source over L). The pixels combine the shapes'
    arrivals with their coefficients. diag: the rank and the SVD's largest dropped share."""
    import dp_galactic
    import dp_relic_loss
    import dp_sky
    E = np.asarray(E, dtype=np.float64)
    U = np.asarray(U, dtype=np.complex128)
    A = np.abs(U) ** 2                                          # A[alpha, i]
    comps = [np.asarray(c, dtype=np.float64) for c in compositions]
    npix = dp_sky.nside_to_npix(int(nside_out))
    if rates is None:
        P = dp_galactic.averaged_transfer(U)
        return [np.ascontiguousarray(np.broadcast_to(P @ c, (npix, len(E), 3))) for c in comps]
    e = dp_galactic._data_file("galactic_emission.npz")["gas_s_edges_kpc"]
    Ls = np.diff(e) * KPC_EV_INV
    L0 = float(e[0]) * KPC_EV_INV
    N = len(E)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1.0 / h)
    n = rates["n"]
    dec = rates["dec"]
    use_dec = dec is not None
    if use_dec:
        br, own, kf, kx, fx, mom = (dec["br"], dec["own"], dec["kfull"], dec["kex"], dec["fex"], dec["mom"])
    else:
        br, own = np.zeros((n, n)), np.zeros((n, n))
        kf, kx = np.zeros((n, n, 2), dtype=np.int64), np.full((n, n, 1), -1, dtype=np.int64)
        fx, mom = np.zeros((n, n, 1)), np.zeros((n, n, 3))
    lam, Rc, loss = rates["lam"], rates["Rc"], rates["loss"]
    coef, shapes, Iout, dropped = _emission_shapes(template, E, int(nside_out))
    P_, K, r = coef.shape
    if diag is not None:
        diag["rank"] = r
        diag["dropped"] = dropped
    # each shape in each visible mass state, from the Sun outward as a constant source; arrival per bin
    C = 3 * r
    F = np.zeros((N, n, C))
    S = np.zeros((N, n, C))
    for j in range(3):
        S[:, j, j::3] = shapes.T
    S = np.ascontiguousarray(S)
    Z = np.zeros((N, n, C))
    if loss is not None:
        _loss_half(E, F, loss, 0.5 * L0, dp_relic_loss)
    _segment(F, E, dE, S, L0, lam, Rc, br, own, kf, kx, fx, mom, use_dec)
    if loss is not None:
        _loss_half(E, F, loss, 0.5 * L0, dp_relic_loss)
    # the emission of bin k is X[:, k] / Ls[k] per unit length over [e_k, e_k+1]: its arrival (Y(e_k+1) - Y(e_k)) /
    # Ls[k] for the unit source per unit length, Y the state reached with that source from the Sun
    arr = np.zeros((K, N, 3, r, 3))                              # [bin, E, state i, shape, unit state j]
    prev = F[:, :3, :].copy()
    for k in range(K):
        if loss is not None:
            _loss_half(E, F, loss, 0.5 * Ls[k], dp_relic_loss)
        _segment(F, E, dE, S, Ls[k], lam, Rc, br, own, kf, kx, fx, mom, use_dec)
        if loss is not None:
            _loss_half(E, F, loss, 0.5 * Ls[k], dp_relic_loss)
        cur = F[:, :3, :].copy()
        arr[k] = ((cur - prev) / Ls[k]).reshape(N, 3, r, 3)
        prev = cur
    del Z
    outs = []
    for c in comps:
        p = A.T @ c                                             # the composition's mass-state shares
        B = np.einsum("keirj,j->krei", arr, p).reshape(K * r, N * 3)
        mass = (coef.reshape(P_, K * r) @ B).reshape(P_, N, 3)
        mass = np.divide(mass, Iout[:, :, None], out=np.zeros_like(mass), where=Iout[:, :, None] > 0)
        outs.append(np.ascontiguousarray(mass @ A.T))
    return outs


def _loss_half(E, F, loss, L, dp_relic_loss):
    """The relic loss over length L on the visible states of every column (dp_relic_loss.shift_step; z = 0)."""
    dEo = np.ascontiguousarray(np.broadcast_to(np.asarray(loss, dtype=np.float64) * L, (len(E), 3)))
    for c in range(F.shape[2]):
        Fc = np.ascontiguousarray(F[:, :3, c])
        dp_relic_loss.shift_step(E, Fc, dEo)
        F[:, :3, c] = Fc


# ------------------------------------------------------------------------------------------------------------------
# Where the models hold (2026-10-25)

FIELD_SHIFT_MAX = 1e-3       # the scalar's relic field: not modelled where it shifts the masses squared by more than
                             # this x Delta m^2_21 (Mauricio 2026-10-25, decision (a))


def _partial_wave_max_n(Y):
    """dp_unitarity.partial_wave_max for any number of Weyl fermions (-L = 1/2 Y_ab phi psi_a psi_b + h.c., Y
    symmetric n x n): the largest |eigenvalue| of the J = 0 and J = 1/2 partial-wave matrices (tree level)."""
    Y = np.asarray(Y, dtype=np.complex128)
    nf = Y.shape[0]
    pairs = [(i, j) for i in range(nf) for j in range(i, nf)]
    n = len(pairs)
    c = np.array([1.0 / np.sqrt(2.0) if i == j else 1.0 for i, j in pairs])
    A = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            w = c[a] * c[b] / (16.0 * np.pi)
            A[a, b] = -np.conj(Y[i, j]) * Y[k, l] * w
            A[n + a, n + b] = np.conj(A[a, b])
            stu = (Y[i, k] * Y[j, l] + Y[i, l] * Y[j, k] + Y[i, j] * Y[k, l]) * w
            A[a, n + b] = stu
            A[n + b, a] = np.conj(stu)
    j0 = np.max(np.abs(np.linalg.eigvalsh(0.5 * (A + A.conj().T))))
    j12 = 3.0 * np.max(np.abs(np.linalg.eigvalsh(Y @ Y.conj().T))) / (32.0 * np.pi)
    return max(j0, j12)


def goldstone_measure(G_flavor, U, masses, M, majorana=True):
    """The vector's longitudinal mode (dp_unitarity; notes Sec. 12.8 item 3): >= 1 where its Goldstone couplings break
    perturbative unitarity. Majorana: dp_unitarity.neutrino_unitarity of G' = U^dag G U (the identity part rotated
    exactly). Dirac (2026-10-25): the current nubar gamma^mu G' P_L nu is not conserved through the Dirac masses,
    d.J = i sum G'_ij (m_i nubar_i P_L nu_j - m_j nubar_i P_R nu_j), so the Goldstone couples as
    -L = phi N_i Y_ij nu_L,j + h.c., Y_ij = i G'_ij m_i / M (N = (nu_R)^c), i.e. 1/2 Y' psi psi over the six Weyl
    fields psi = (nu_L, N) with Y' = [[0, Y^T], [Y, 0]]; 2 x its largest partial wave (_partial_wave_max_n)."""
    import dp_unitarity
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    c = G[0, 0].real
    Gp = U.conj().T @ (G - c * np.eye(3)) @ U + c * np.eye(3)
    if majorana:
        return dp_unitarity.neutrino_unitarity(Gp, m, M)
    Y = 1j * Gp * m[:, None] / float(M)
    Yp = np.zeros((6, 6), dtype=np.complex128)
    Yp[0:3, 3:6] = Y.T
    Yp[3:6, 0:3] = Y
    return 2.0 * _partial_wave_max_n(Yp)


def scalar_density(masses, eta, T_nu0, z):
    """The relics' scalar density per mass state at redshift z (eV^3): both helicities, 2 n_hel eta_r (1+z)^3
    <m_r / E_r> at T_r = T_nu0 (1+z) eta_r^(1/3) (the code's relic temperatures)."""
    m = np.asarray(masses, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    n_hel0 = 3.0 * 1.2020569031595942 * T_nu0 ** 3 / (4.0 * np.pi ** 2)
    out = np.zeros(3)
    for r in range(3):
        if eta[r] > 0.0 and m[r] > 0.0:
            out[r] = 2.0 * n_hel0 * eta[r] * (1.0 + z) ** 3 * dp_decay._mean_m_over_E(m[r], T_nu0 * (1.0 + z) *
                                                                                        eta[r] ** (1.0 / 3.0))
    return out


def scalar_field_measure(G_flavor, U, masses, M, majorana, T_nu0, eta_cosmic, H0, Omega_m, Omega_L, z_max, dm21_sq,
                         decay_on=True, return_parts=False):
    """The scalar's background field sourced by the relics (2026-10-25; Mauricio's decision (a): not modelled where it
    matters): >= 1 where it shifts the neutrinos' masses squared by FIELD_SHIFT_MAX x Delta m^2_21 or more at z_max.

    The relics' scalar density n_s,r (both kinds add: the scalar density is C-even; scalar_density) sources a static
    field; minimizing V(phi) + rho_relics(m_eff(phi)) gives M^2 phi = -sum_r Re(c_rr) n_s,r, c the mass-basis coupling
    (lambda = U^T G U Majorana, h = U^dag G U Dirac), and every neutrino -- beam and relic -- has the mass matrix
    D + c phi. The shift of the masses squared, (D + c phi)(D + c phi)^dag - D^2, grows as (g / M)^2; its largest
    |eigenvalue| over FIELD_SHIFT_MAX Delta m^2_21 is the measure. The relics are those after their own decays by
    z = 1 (the transport's: dp_decay.relic_composition with this point's rates), and the density is largest at z_max
    ((1+z)^3, (1+z)^2 for relativistic relics). Above the line the relics also clump (the attraction beats their
    pressure for g^2 n_s / M^2 >~ <p^2>/m, g/M >~ 1e3 eV^-1 today): included. return_parts: also (phi c, n_s)."""
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    M = float(M)
    # the whole coupling, its identity part included (a flavor-universal Dirac coupling shifts every mass alike; the
    # decays' mass_couplings drop that part, which moves no neutrino between states)
    cpl = U.T @ G @ U if majorana else U.conj().T @ G @ U
    eta = np.asarray(eta_cosmic, dtype=np.float64)
    if decay_on:
        Gam = rest_rates(G, U, m, M, "scalar", majorana)
        if np.any(Gam > 0.0):
            eta = dp_decay.relic_composition(Gam, m, eta, T_nu0, H0, Omega_m, Omega_L)
    ns = scalar_density(m, eta, T_nu0, float(z_max))
    phi = -float(np.sum(np.real(np.diag(cpl)) * ns)) / (M * M)
    D = np.diag(m).astype(np.complex128)
    Me = D + cpl * phi
    shift = Me @ Me.conj().T - D @ D
    X = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (shift + shift.conj().T))))) / (FIELD_SHIFT_MAX * dm21_sq)
    if return_parts:
        return X, (cpl * phi, ns)
    return X


# ------------------------------------------------------------------------------------------------------------------
# The vector's longitudinal-mode scatterings (2026-10-25; notes, "Secret interactions below 1 keV", Mauricio: on the
# lepton-number planes). With massive neutrinos the q^mu q^nu / M^2 part of the vector's propagator does not drop
# out: between the currents it is exactly the exchange of a scalar with the Goldstone couplings
# Y = i (G'_ij m_i + G'_ji m_j) / M (the Dirac equation; tests/test_goldstone.py, explicit spinors), adding to the
# transverse part without interference (the beam's helicity flips). dp_si's Majorana scalar channels with lambda = Y
# carry it (the normalization is the decays' longitudinal part). Where the potential leaves the eigenstates mass
# states at the energies where it absorbs, it acts on the dark-portal phase's populations as a factor per mass state.

GOLDSTONE_MODELS = ("L_e-L_mu", "L_e-L_tau")    # Mauricio 2026-10-25; the other vector planes: the post-campaign check
GOLDSTONE_Y_MIN = 1e-4       # below this largest |Y| the depth is < ~1e-9 (scalar channels: depth ~0.01 at 5e-3)
GOLDSTONE_OFF = 1e-30        # the reference run's Y scale: collisions ~1e-120 of the decays, i.e. none


def goldstone_yukawa(G_flavor, U, masses, M):
    """Y (3, 3) complex symmetric, the Goldstone couplings of the vector with Majorana neutrinos (dp_unitarity), the
    identity part of G rotated exactly."""
    import dp_unitarity
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    c = G[0, 0].real
    Gp = U.conj().T @ (G - c * np.eye(3)) @ U + c * np.eye(3)
    return dp_unitarity.goldstone_yukawa(Gp, np.asarray(masses, dtype=np.float64), float(M))


def goldstone_factors(E, G_flavor, U, masses, M, T_nu0, eta, H0, Omega_m, Omega_L, z_max, p_index, rho_evol, decay,
                      z_substeps=2):
    """(R (N_E, 3), R_tot (N_E,)) or None where the largest |Y| is below GOLDSTONE_Y_MIN: per mass state, dp_si's flux
    with the Goldstone channels (Majorana scalar channels, lambda = Y; relics thermal) over the same without them, the
    decays in both (decay: dp_si's (Gam, spec) or None), for a fixed reference source (equal flavors, the whole
    spectrum, index p_index), so that the runs the analysis adds up add up exactly (as the 'combined' mode's factor);
    R_tot the same for the sum over the states. The collision mass is max(M, m_heaviest), as the scalar's
    (subkev_inputs: only the pair channel's collinear logarithm depends on it)."""
    import dp_si
    m = np.asarray(masses, dtype=np.float64)
    Y = goldstone_yukawa(G_flavor, U, m, M)
    if float(np.max(np.abs(Y))) < GOLDSTONE_Y_MIN:
        return None
    E = np.asarray(E, dtype=np.float64)
    src = np.full(3, 1.0 / 3.0)
    Mc = max(float(M), float(np.max(m)))
    args = (E, float(z_max), H0, Omega_m, Omega_L, U, m, T_nu0, np.asarray(eta, dtype=np.float64), src,
            float(p_index), float(rho_evol), 1.0, Mc)
    F1 = dp_si.propagate_si(*args, np.ascontiguousarray(Y), majorana=True, mediator="scalar", thermal="full",
                            z_substeps=z_substeps, decay=decay)
    F0 = dp_si.propagate_si(*args, np.ascontiguousarray(Y * GOLDSTONE_OFF), majorana=True, mediator="scalar",
                            thermal=False, z_substeps=z_substeps, decay=decay)
    R = np.divide(F1, F0, out=np.ones_like(F1), where=F0 > 0.0)
    t0 = F0.sum(axis=1)
    Rt = np.divide(F1.sum(axis=1), t0, out=np.ones_like(t0), where=t0 > 0.0)
    return np.ascontiguousarray(R), Rt
