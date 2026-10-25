"""
dp_decay.py -- decay of the neutrinos into the mediator, nu_i -> nu_j + Z_D, for m_Zd < m_i - m_j (notes Sec. 12.8
item 2 and Sec. 12.9 (e), 2026-10-24).

Rates. The Majorana vertex gamma^mu (G'_ji P_L - G'*_ji P_R), G' = U^dag G U the couplings in the mass basis; the
rest-frame rate, averaged over the parent's spin (checked against the traces, tools/check_decay_rate.py),
    Gamma_ij = |p| / (16 pi m_i^2) [2 |G'_ji|^2 (m_i^2 + m_j^2 - 2M^2 + (m_i^2 - m_j^2)^2 / M^2)
                                    + 12 Re(G'_ji^2) m_i m_j],
twice the Dirac rate for m_j -> 0; in flight slower by m_i / E. The identity part of G (a flavor-universal coupling,
the Z mixing's kappa) has no off-diagonal entries in any basis: G - G[0,0] I is rotated, so that a universal part
cancels exactly and its rounding (1e-16 kappa) never shows up as a decay.

Daughters (2026-10-24, exact kinematics; the review's finding 1). A parent in flight (E >> m_i) has a definite
helicity h, its daughter h' ('kept' h' = h, 'flipped' h' = -h; Majorana: the flipped daughter is of the other kind).
For a boost to infinity helicity is light-front helicity, so each (h, h') rate is a covariant trace with light-front
spin vectors (daughter_spectra; tools/derive_decay_kinematics.py), and the daughter's share of the parent's energy is
its light-front fraction z = (E* + p* cos th*) / m_i, in [z-, z+] = [(E* - p*)/m_i, (E* + p*)/m_i] (rest-frame E*, p*;
the mediator takes the rest). On that window dGamma/dz = a + b z + c / z exactly (fitted at nine angles; residual
~1e-15, 1e-7 for m_j ~ 1e-6 eV), and the kept and flipped integrals add up to rest_rates' Gamma_ij (checked to 1e-6).
For m_j, M -> 0 the window is [0, 1], the kept half 2(1-z) f_T + 2 z f_L, the flipped half the mirror (the earlier
massless kernels, f_L = 1/(1 + 2 M^2/m_i^2)); with massive daughters the kept share is not 1/2 (1/3 at m_i = 0.112,
m_j = 0.1 eV for real couplings; it depends on the coupling's phase) and nearly degenerate masses leave the daughter
almost all the energy (z in [0.80, 1.00] at m_i = 0.112, m_j = 0.1, m_Zd = 1e-3 eV). The mediator is taken as
invisible.

The step (decay_step): a top-down sweep over the energy bins, like the cosmological cascade (dp_cosmo): the
daughters of the bins above arrive first, then the bin's own populations decay, exactly for any rate x step, heaviest
first, so a chain nu_3 -> nu_2 -> nu_1 runs within the step. A parent's daughters fall on the bins whose energies are
z E_parent, z in its window: the bins inside the window through running sums of the spectrum's three moments (entered
at the window's top bin, removed below its bottom bin), the one or two bins its edges cut and the parent's own bin by
their exact integrals (the parent at its bin's centre). The number of neutrinos is kept exactly, except what falls
below the grid. The populations are mass states (dp_si, and the tests). In a potential the decays act between the
propagation eigenstates with their own kinematics: dp_medium (2026-10-25); the model that made each eigenstate decay
as a whole at its mass content's mean rate, with vacuum kinematics, is removed (it let the most stable eigenstate decay
into itself).
"""
import numpy as np
from numba import njit

# NumPy 2 renamed trapz to trapezoid (CI runs NumPy 2, the pinned environment 1.26)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def rest_rates(G_flavor, U, masses, M, majorana=True):
    """(Gam, fL): Gam[i, j] the rest-frame rate (eV) of nu_i -> nu_j + Z_D, fL[i, j] its longitudinal share.
    G_flavor (3, 3) Hermitian in the flavor basis: pass the non-universal coupling (the dark portal's g_D u u^dag,
    the gauge models' charges), never one with a large universal kappa I already added -- the sum rounds the small
    diagonal entries away (G - kappa I then misses them); U the mixing matrix (mass states as columns), masses (3,) eV,
    M the mediator mass (eV). Dirac (majorana False): the P_L piece alone, half the rate."""
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    # only off-diagonal entries of G' enter: take out an identity part c I exactly (c = G[0, 0], a representable
    # number, so a universal coupling cancels to zero; tr G / 3 would leave 1e-16 c, a spurious decay of the relics)
    c = G[0, 0].real
    Gp = U.conj().T @ (G - c * np.eye(3)) @ U
    Gam = np.zeros((3, 3))
    fL = np.zeros((3, 3))
    M = float(M)
    if M <= 0.0:
        return Gam, fL
    for i in range(3):
        for j in range(3):
            if i == j or m[i] <= m[j] + M:
                continue
            mi, mj = m[i], m[j]
            p = np.sqrt(max((mi * mi - (mj + M) ** 2) * (mi * mi - (mj - M) ** 2), 0.0)) / (2.0 * mi)
            g = Gp[j, i]
            if majorana:
                bracket = (2.0 * abs(g) ** 2 * (mi * mi + mj * mj - 2.0 * M * M + (mi * mi - mj * mj) ** 2 / (M * M))
                           + 12.0 * (g * g).real * mi * mj)
            else:
                bracket = abs(g) ** 2 * (mi * mi + mj * mj - 2.0 * M * M + (mi * mi - mj * mj) ** 2 / (M * M))
            Gam[i, j] = max(p / (16.0 * np.pi * mi * mi) * bracket, 0.0)
            fL[i, j] = 1.0 / (1.0 + 2.0 * (M / mi) ** 2)
    return Gam, fL


def total_lab_rate(Gam, masses, E):
    """The lab-frame rate (eV) of each mass state at physical energy E (eV): sum_j Gam[i, j] m_i / E."""
    return np.asarray(Gam).sum(axis=1) * np.asarray(masses) / float(E)


@njit(cache=True)
def _add(F, b, j, n_per_E, T, use_T):
    if use_T:
        for a in range(3):
            F[b, a] += n_per_E * T[b, j, a]
    else:
        F[b, j] += n_per_E


# Dirac matrices (the Dirac representation) for daughter_spectra's traces
_s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
_Z2 = np.zeros((2, 2), dtype=np.complex128)
_GAM = np.array([np.block([[_Z2, _I2], [_I2, _Z2]])] + [np.block([[_Z2, s_], [-s_, _Z2]]) for s_ in (_s1, _s2, _s3)])
_G5 = np.block([[-_I2, _Z2], [_Z2, _I2]])
_PL = (np.eye(4) - _G5) / 2
_PR = (np.eye(4) + _G5) / 2
_ETA = np.diag([1.0, -1.0, -1.0, -1.0])
_NLC = np.array([1.0, 0.0, 0.0, -1.0])           # n.k = k^+ for a boost along +z


def _slash(v):
    return np.einsum("m,mab->ab", _ETA @ v, _GAM)


def _rho_lf(k, m, h, kplus):
    """u u-bar of light-front helicity h for momentum k, mass m, k^+ = n.k given (computed without cancellation by the
    caller): (k + m)(1 + h g5 s)/2 with s = k/m - m n/k^+, written without 1/m (for m = 0 the chiral projector)."""
    ks = _slash(k)
    I4 = np.eye(4)
    out = (ks + m * I4) + h * (_G5 @ (ks - m * I4))
    if m != 0.0:
        out = out - h * (m / kplus) * ((ks + m * I4) @ _G5 @ _slash(_NLC))
    return 0.5 * out


def _msq_lf(p, k, mi, mj, M, h, hp, g, kplus):
    q = p - k
    V = g * _PL - np.conj(g) * _PR
    Gm = np.einsum("mab,bc->mac", _GAM, V)
    Gb = np.einsum("ab,mcb,cd->mad", _GAM[0], Gm.conj(), _GAM[0])
    A = _rho_lf(k, mj, hp, kplus)
    B = _rho_lf(p, mi, h, mi)
    T = np.einsum("ab,mbc,cd,nda->mn", A, Gm, B, Gb)
    P = -_ETA + np.outer(_ETA @ q, _ETA @ q) / (M * M)
    return float(np.real(np.sum(T * P)))


def daughter_spectra(G_flavor, U, masses, M):
    """The daughters' spectra per unit share z of the parent's energy, spec[s, i, j, rel] = (z-, z+, a, b, c) with
    dGamma/dz / Gamma_ij = a + b z + c / z on [z-, z+] (module docstring): s the parent's kind (0 the neutrino,
    helicity -1; 1 the antineutrino), rel 0 kept / 1 flipped; each rel's integral is its share of Gamma_ij (the two
    add up to 1). Zeros where nu_i -> nu_j is closed. G_flavor, U, masses, M as rest_rates."""
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    Gp = U.conj().T @ (G - G[0, 0].real * np.eye(3)) @ U
    M = float(M)
    spec = np.zeros((2, 3, 3, 2, 5))
    if M <= 0.0:
        return spec
    cth = np.cos(np.linspace(np.pi, 0.0, 9))
    for i in range(3):
        for j in range(3):
            if i == j or m[i] <= m[j] + M or Gp[j, i] == 0.0:
                continue
            mi, mj = float(m[i]), float(m[j])
            pf = np.sqrt(max((mi * mi - (mj + M) ** 2) * (mi * mi - (mj - M) ** 2), 0.0)) / (2.0 * mi)
            Ej = np.sqrt(pf * pf + mj * mj)
            emp = mj * mj / (Ej + pf)                 # Ej - pf without the cancellation
            zm, zp = emp / mi, (Ej + pf) / mi
            p = np.array([mi, 0.0, 0.0, 0.0])
            z = (emp + pf * (1.0 + cth)) / mi
            for s_, h in ((0, -1.0), (1, 1.0)):
                coef = []
                for hp in (h, -h):
                    f = np.array([_msq_lf(p, np.array([Ej, pf * np.sqrt(max(1.0 - c * c, 0.0)), 0.0, pf * c]),
                                          mi, mj, M, h, hp, Gp[j, i], emp + pf * (1.0 + c)) for c in cth]) \
                        / (16.0 * np.pi * mi)
                    if zm > 0.0:
                        B = np.stack([np.ones_like(z), z, 1.0 / z], axis=1)
                        a_, b_, c_ = np.linalg.lstsq(B, f, rcond=None)[0]
                    else:
                        B = np.stack([np.ones_like(z), z], axis=1)
                        (a_, b_), c_ = np.linalg.lstsq(B, f, rcond=None)[0], 0.0
                    coef.append((a_, b_, c_))
                ints = [a_ * (zp - zm) + 0.5 * b_ * (zp * zp - zm * zm) + (c_ * np.log(zp / zm) if c_ != 0.0 else 0.0)
                        for a_, b_, c_ in coef]
                tot = ints[0] + ints[1]
                for rel in range(2):
                    a_, b_, c_ = coef[rel]
                    spec[s_, i, j, rel] = (zm, zp, a_ / tot, b_ / tot, c_ / tot)
    return spec


def massless_spectra(fL):
    """The earlier kernels (massless daughter and mediator), as daughter_spectra's array: z in [0, 1], the kept
    half f_T + (f_L - f_T) z, the flipped half the mirror, per pair from fL (3, 3), for both kinds."""
    spec = np.zeros((2, 3, 3, 2, 5))
    for i in range(3):
        for j in range(3):
            fl = float(fL[i, j])
            ft = 1.0 - fl
            for s_ in range(2):
                spec[s_, i, j, 0] = (0.0, 1.0, ft, fl - ft, 0.0)
                spec[s_, i, j, 1] = (0.0, 1.0, fl, ft - fl, 0.0)
    return spec


@njit(cache=True)
def _spec_integral(zm, zp, a, b, c, z1, z2):
    """int of a + b z + c/z over [z1, z2] within [zm, zp]."""
    lo = max(z1, zm)
    hi = min(z2, zp)
    if hi <= lo:
        return 0.0
    v = a * (hi - lo) + 0.5 * b * (hi * hi - lo * lo)
    if c != 0.0 and lo > 0.0:
        v += c * np.log(hi / lo)
    return v


# window_plan's slots for the bins a window cuts (its own bin, the edges' bins with the parent spread over its bin)
NSLOT = 10


@njit(cache=True)
def _spec_G(z, a, b, c):
    """The spectrum's antiderivative a z + b z^2/2 + c ln z (c = 0 where z may be 0)."""
    v = a * z + 0.5 * b * z * z
    if c != 0.0:
        v += c * np.log(z)
    return v


@njit(cache=True)
def _cdf_int(alpha, u1, u2, zm, zp, a, b, c):
    """int_{u1}^{u2} F(alpha / u) du, F the spectrum's cumulative integral from zm (0 below zm, its total above zp),
    in closed form (the parent's energy u E_b spread over its bin, a daughter bin edge at alpha E_b)."""
    if u2 <= u1:
        return 0.0
    Gm = _spec_G(zm, a, b, c) if zm > 0.0 else 0.0
    tot = _spec_G(zp, a, b, c) - Gm
    # alpha / u >= zp for u <= alpha / zp: the whole spectrum below the edge
    ua = alpha / zp
    ub = alpha / zm if zm > 0.0 else np.inf
    out = 0.0
    lo, hi = u1, min(u2, ua)
    if hi > lo:
        out += tot * (hi - lo)
    lo, hi = max(u1, ua), min(u2, ub)
    if hi > lo:
        # int (G(alpha/u) - G(zm)) du: a alpha ln u - b alpha^2 / (2u) + c u (ln alpha - ln u + 1) - G(zm) u
        def_hi = a * alpha * np.log(hi) - 0.5 * b * alpha * alpha / hi - Gm * hi
        def_lo = a * alpha * np.log(lo) - 0.5 * b * alpha * alpha / lo - Gm * lo
        if c != 0.0:
            def_hi += c * hi * (np.log(alpha) - np.log(hi) + 1.0)
            def_lo += c * lo * (np.log(alpha) - np.log(lo) + 1.0)
        out += def_hi - def_lo
    return out


@njit(cache=True)
def window_plan(spec, h, N, kfull, kex, fex):
    """For a parent spread uniformly over its bin (2026-10-25, the convergence audit's finding 3: at the bin's centre a
    daughter keeping more than 1/h of the energy never left it, and nearly degenerate decays lost no energy per cycle)
    on a geometric grid (half-bin ratio h): kfull[s, i, j, rel] = (k1, k2), the offsets (parent bin - daughter bin) of
    the bins inside the window wherever the parent sits in its bin (k1 > k2: none); kex/fex[s, i, j, rel, :NSLOT] the
    other offsets the window touches and their exact fractions (kex -1: unused). Offset k covers the daughter energies
    [h^(-2k-1), h^(-2k+1)] E_b, the own bin (k = 0) included."""
    lh = np.log(h)
    span = h - 1.0 / h
    nslot = kex.shape[4]
    for s in range(2):
        for i in range(3):
            for j in range(3):
                for rel in range(2):
                    zm = spec[s, i, j, rel, 0]
                    zp = spec[s, i, j, rel, 1]
                    a = spec[s, i, j, rel, 2]
                    b = spec[s, i, j, rel, 3]
                    c = spec[s, i, j, rel, 4]
                    for x in range(nslot):
                        kex[s, i, j, rel, x] = -1
                        fex[s, i, j, rel, x] = 0.0
                    kfull[s, i, j, rel, 0] = 1
                    kfull[s, i, j, rel, 1] = 0
                    if zp <= zm:
                        continue
                    # inside for every parent position u in [1/h, h]: h^(-2k+1) h <= zp and h^(-2k-1) / h >= zm
                    k1 = int(np.ceil(1.0 - 0.5 * np.log(zp) / lh - 1e-12))
                    if k1 < 1:
                        k1 = 1
                    if zm > 0.0:
                        k2 = int(np.floor(-1.0 - 0.5 * np.log(zm) / lh + 1e-12))
                    else:
                        k2 = N + 1
                    if k2 > N + 1:
                        k2 = N + 1
                    kfull[s, i, j, rel, 0] = k1
                    kfull[s, i, j, rel, 1] = k2
                    # the lowest daughter energy zm E_b / h: the deepest offset touched
                    kbot = N + 1 if zm <= 0.0 else int(np.floor(0.5 * (1.0 - np.log(zm / h) / lh))) + 1
                    x = 0
                    k = 0
                    while k <= kbot and k <= N + 1:
                        if k >= k1 and k <= k2:
                            k = k2 + 1
                            continue
                        a_hi = h ** (-2.0 * k + 1.0)
                        a_lo = h ** (-2.0 * k - 1.0)
                        v = (_cdf_int(a_hi, 1.0 / h, h, zm, zp, a, b, c)
                             - _cdf_int(a_lo, 1.0 / h, h, zm, zp, a, b, c)) / span
                        if v > 0.0:
                            if x >= nslot:
                                raise ValueError("window_plan: more cut bins than NSLOT")
                            kex[s, i, j, rel, x] = k
                            fex[s, i, j, rel, x] = v
                            x += 1
                        k += 1


@njit(cache=True)
def decay_step(E, zp1, dt, Gam, spec, m, F_nu, F_nb, T_nu, T_nb, use_T, fold):
    """One step of the decays, in place. E (N,) observed energies on a geometric grid (eV), zp1 = 1 + z, dt the step
    (eV^-1), Gam from rest_rates, spec from daughter_spectra (2, 3, 3, 2, 5), m (3,) masses (eV); F_nu, F_nb (N, 3)
    populations per unit energy of the mass states; fold: one population holding both kinds (dp_si's Majorana flux,
    equal kinds), the flipped daughters into it too (F_nb unused). T_nu, T_nb, use_T: kept for the signature; use_T
    True (eigenstates decaying as wholes) was removed on 2026-10-25 -- in a potential, dp_medium."""
    if use_T:
        raise ValueError("dp_decay.decay_step: eigenstates decay through dp_medium (2026-10-25)")
    N = E.shape[0]
    tot = np.zeros(3)
    for i in range(3):
        for j in range(3):
            tot[i] += Gam[i, j]
    if tot[0] + tot[1] + tot[2] == 0.0:
        return
    h = np.sqrt(E[1] / E[0]) if N > 1 else 1.0
    Lb = 2.0 * np.log(h)
    nsp = 1 if fold else 2
    kfull = np.zeros((2, 3, 3, 2, 2), dtype=np.int64)
    kex = np.zeros((2, 3, 3, 2, NSLOT), dtype=np.int64)
    fex = np.zeros((2, 3, 3, 2, NSLOT))
    window_plan(spec, h, N, kfull, kex, fex)
    uinv = 2.0 * np.log(h) / (h - 1.0 / h)      # <1/u> over the parent's bin (the a-moment; <1/u^2> = 1)
    S = np.zeros((2, 3, 3))                 # running moments per kind and daughter state: c, a / E_p, b / E_p^2
    add = np.zeros((2, 3, 3, N))
    sub = np.zeros((2, 3, 3, N))
    pend = np.zeros((2, 3, N))
    hm = np.zeros(3)
    order = np.zeros(3, dtype=np.int64)
    for b in range(N - 1, -1, -1):
        Eb = E[b]
        Elo = Eb / h
        Ehi = Eb * h
        dEb = Ehi - Elo
        q2 = 0.5 * (Ehi * Ehi - Elo * Elo)
        for t in range(nsp):
            F = F_nu if t == 0 else F_nb
            T = T_nu if t == 0 else T_nb
            for j in range(3):
                for mo in range(3):
                    S[t, j, mo] += add[t, j, mo, b]
                n = S[t, j, 0] * Lb + S[t, j, 1] * dEb + S[t, j, 2] * q2 + pend[t, j, b]
                if n != 0.0:
                    _add(F, b, j, n / dEb, T, use_T)
                for mo in range(3):
                    S[t, j, mo] -= sub[t, j, mo, b]
        for s in range(nsp):
            F = F_nu if s == 0 else F_nb
            T = T_nu if s == 0 else T_nb
            # heaviest content first
            for a in range(3):
                acc = 0.0
                for i in range(3):
                    acc += (T[b, i, a] if use_T else (1.0 if i == a else 0.0)) * m[i]
                hm[a] = acc
            for a in range(3):
                order[a] = a
            for x in range(3):
                for y in range(2 - x):
                    if hm[order[y]] < hm[order[y + 1]]:
                        tmp = order[y]
                        order[y] = order[y + 1]
                        order[y + 1] = tmp
            for oi in range(3):
                a = order[oi]
                Na = F[b, a] * dEb
                if Na == 0.0:
                    continue
                # the state's content out of the bin first, its survivors back at the end (Na (1 - p), never below
                # zero): no subtraction leaves a rounding residue (2026-10-24: residues of -1e-40 where the exact
                # kinematics leave nothing broke the analysis' log-spline in the spectral index)
                Lam = 0.0
                for i in range(3):
                    if tot[i] == 0.0:
                        continue
                    w = T[b, i, a] if use_T else (1.0 if i == a else 0.0)
                    if w <= 0.0:
                        continue
                    Lam += w * tot[i] * m[i] / (Eb * zp1) * dt
                if not Lam > 0.0:
                    continue
                ptot = -np.expm1(-Lam)
                keep = 1.0 - ptot
                F[b, a] = 0.0
                for i in range(3):
                    if tot[i] == 0.0:
                        continue
                    w = T[b, i, a] if use_T else (1.0 if i == a else 0.0)
                    if w <= 0.0:
                        continue
                    lam = tot[i] * m[i] / (Eb * zp1) * dt
                    dec = Na * ptot * (w * lam / Lam)
                    if dec == 0.0:
                        continue
                    for j in range(3):
                        if Gam[i, j] == 0.0:
                            continue
                        nd = dec * Gam[i, j] / tot[i]
                        for rel in range(2):
                            tt = s if (rel == 0 or fold) else 1 - s
                            Ft = F_nu if tt == 0 else F_nb
                            Tt = T_nu if tt == 0 else T_nb
                            for x in range(NSLOT):
                                k = kex[s, i, j, rel, x]
                                if k < 0:
                                    break
                                if k == 0:
                                    _add(Ft, b, j, nd * fex[s, i, j, rel, x] / dEb, Tt, use_T)
                                elif b - k >= 0:
                                    pend[tt, j, b - k] += nd * fex[s, i, j, rel, x]
                            k1 = kfull[s, i, j, rel, 0]
                            k2 = kfull[s, i, j, rel, 1]
                            if k1 <= k2 and b - k1 >= 0:
                                v0 = nd * spec[s, i, j, rel, 4]
                                v1 = nd * spec[s, i, j, rel, 2] * uinv / Eb
                                v2 = nd * spec[s, i, j, rel, 3] / (Eb * Eb)
                                add[tt, j, 0, b - k1] += v0
                                add[tt, j, 1, b - k1] += v1
                                add[tt, j, 2, b - k1] += v2
                                if b - k2 >= 0:
                                    sub[tt, j, 0, b - k2] += v0
                                    sub[tt, j, 1, b - k2] += v1
                                    sub[tt, j, 2, b - k2] += v2
                F[b, a] += Na * max(keep, 0.0) / dEb


def _mean_m_over_E(m, T, n=400):
    """<m/E> over a Fermi-Dirac momentum distribution at temperature T (eV) for mass m (eV)."""
    x = np.linspace(1e-4, 40.0, n)
    f = x * x / (np.exp(x) + 1.0)
    p = x * T
    return float(_trapz(f * m / np.sqrt(p * p + m * m), x) / _trapz(f, x))


def relic_composition(Gam, masses, eta, T_nu0, H0, Omega_m, Omega_L, z_ref=1.0, Omega_r=9.1e-5):
    """The relics' occupation per mass state after their decays by z_ref (2026-10-24; notes Sec. 12.8 item 2): each
    state i decays with D_i = Gamma_i int_{z_ref}^inf <m_i/E_i>(z) dz / ((1+z) H(z)) (time-dilated while
    relativistic; H with radiation), heaviest first, its daughters to the lighter states by branching (the number
    kept). Returns eta' (3,); unchanged where nothing decays. The code ties a state's density to its temperature
    (T_r = T eta^(1/3)), so the grown nu_1 population is a hotter Fermi-Dirac gas: an approximation (docs item 22)."""
    Gam = np.asarray(Gam, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64).copy()
    tot = Gam.sum(axis=1)
    if not np.any(tot > 0.0):
        return eta
    u = np.linspace(np.log1p(z_ref), np.log(1e8), 600)
    zp1 = np.exp(u)
    H = H0 * np.sqrt(Omega_r * zp1 ** 4 + Omega_m * zp1 ** 3 + Omega_L)
    for i in np.argsort(m)[::-1]:
        if tot[i] <= 0.0 or eta[i] <= 0.0:
            continue
        mE = np.array([_mean_m_over_E(m[i], T_nu0 * x) for x in zp1])
        D = tot[i] * float(_trapz(mE / H, u))
        dec = eta[i] * (-np.expm1(-D))
        eta[i] -= dec
        for j in range(3):
            if Gam[i, j] > 0.0:
                eta[j] += dec * Gam[i, j] / tot[i]
    return eta
