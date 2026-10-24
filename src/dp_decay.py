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

Daughters. Half keep the parent's helicity (the P_L piece), half flip it (the P_R piece: a neutrino gives an
antineutrino). With z the daughter's share of the parent's energy (the massless-daughter kinematics), the kept half
follows dN/dz = 2(1-z) from the transverse polarizations and 2z from the longitudinal one, the flipped half the
mirror; f_L = 1/(1 + 2 M^2/m_i^2) is the longitudinal share. Per decay: same kind fT + (fL - fT) z, other kind
fL + (fT - fL) z, each integrating to 1/2. The mediator is taken as invisible.

The step (decay_step): a top-down sweep over the energy bins, like the cosmological cascade (dp_cosmo): the
daughters of the bins above arrive first (separable kernels, two running sums per kind and daughter state), then the
bin's own populations decay, exactly for any rate x step, heaviest first, so a chain nu_3 -> nu_2 -> nu_1 runs within
the step. The number of neutrinos is kept exactly, except what falls below the grid. The populations are either mass
states (dp_si) or eigenstates of the potential (the dark-portal phase), with T[b, i, a] = |<i|a>|^2 their mass
content: each mass component decays on its own (incoherently), and a daughter in mass state j is split over the
eigenstates by T[b, j, :].
"""
import numpy as np
from numba import njit


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


@njit(cache=True)
def decay_step(E, zp1, dt, Gam, fL, m, F_nu, F_nb, T_nu, T_nb, use_T, fold):
    """One step of the decays, in place. E (N,) observed energies on a geometric grid (eV), zp1 = 1 + z, dt the step
    (eV^-1), Gam/fL from rest_rates, m (3,) masses (eV); F_nu, F_nb (N, 3) populations per unit energy (eigenstates
    with mass content T_nu, T_nb (N, 3, 3) if use_T, else mass states); fold: one population holding both kinds
    (dp_si's Majorana flux, equal kinds), the flipped daughters into it too (F_nb, T_nb unused)."""
    N = E.shape[0]
    tot = np.zeros(3)
    for i in range(3):
        for j in range(3):
            tot[i] += Gam[i, j]
    if tot[0] + tot[1] + tot[2] == 0.0:
        return
    h = np.sqrt(E[1] / E[0]) if N > 1 else 1.0
    nsp = 1 if fold else 2
    S1 = np.zeros((2, 3))
    S2 = np.zeros((2, 3))
    hm = np.zeros(3)
    order = np.zeros(3, dtype=np.int64)
    for b in range(N - 1, -1, -1):
        Eb = E[b]
        Elo = Eb / h
        Ehi = Eb * h
        dEb = Ehi - Elo
        q2 = 0.5 * (Ehi * Ehi - Elo * Elo)
        z0 = Elo / Eb
        for t in range(nsp):
            F = F_nu if t == 0 else F_nb
            T = T_nu if t == 0 else T_nb
            for j in range(3):
                n = S1[t, j] * dEb + S2[t, j] * q2
                if n != 0.0:
                    _add(F, b, j, n / dEb, T, use_T)
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
                if Na <= 0.0:
                    continue
                for i in range(3):
                    if tot[i] == 0.0:
                        continue
                    w = T[b, i, a] if use_T else (1.0 if i == a else 0.0)
                    if w <= 0.0:
                        continue
                    lam = tot[i] * m[i] / (Eb * zp1) * dt
                    dec = Na * w * (-np.expm1(-lam))
                    if dec <= 0.0:
                        continue
                    F[b, a] -= dec / dEb
                    for j in range(3):
                        if Gam[i, j] == 0.0:
                            continue
                        nd = dec * Gam[i, j] / tot[i]
                        fl = fL[i, j]
                        ft = 1.0 - fl
                        for rel in range(2):
                            if rel == 0:
                                A = ft
                                B = fl - ft
                            else:
                                A = fl
                                B = ft - fl
                            tt = s if (rel == 0 or fold) else 1 - s
                            own = A * (1.0 - z0) + 0.5 * B * (1.0 - z0 * z0)
                            Ft = F_nu if tt == 0 else F_nb
                            Tt = T_nu if tt == 0 else T_nb
                            _add(Ft, b, j, nd * own / dEb, Tt, use_T)
                            S1[tt, j] += nd * A / Eb
                            S2[tt, j] += nd * B / (Eb * Eb)


def _mean_m_over_E(m, T, n=400):
    """<m/E> over a Fermi-Dirac momentum distribution at temperature T (eV) for mass m (eV)."""
    x = np.linspace(1e-4, 40.0, n)
    f = x * x / (np.exp(x) + 1.0)
    p = x * T
    return float(np.trapz(f * m / np.sqrt(p * p + m * m), x) / np.trapz(f, x))


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
        D = tot[i] * float(np.trapz(mE / H, u))
        dec = eta[i] * (-np.expm1(-D))
        eta[i] -= dec
        for j in range(3):
            if Gam[i, j] > 0.0:
                eta[j] += dec * Gam[i, j] / tot[i]
    return eta
