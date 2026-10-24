"""
dp_relic_loss.py -- the beam's energy loss in soft scatterings that leave the relic in another mass state (notes
Sec. 12.8 item 1 and Sec. 12.9 (c'), 2026-10-24).

A soft t-channel scattering nu_a + nu_r -> nu_c + nu_s with s != r has the forward peak of the elastic one (the relic's
light-cone momentum p^- is kept), but the beam pays omega_rs = (m_s^2 - m_r^2) / (2 p_r^-) per event; weighted with
the flux factor p_r^-/E_r, the cost per unit time is exact over the relic's directions. A neutrino in state a thus
loses energy at a rate that does not depend on its energy,
    dE/dt = -C(z) (G G^dag)_aa,
    C(z) = sum_{r != s} 2 n_hel (1+z)^3 eta_r |G'_sr|^2 (m_s^2 - m_r^2) / 2 <1/E_r> / (4 pi M_t^2),
with the soft scatterings' factors (dark_portal_solver._soft_scattering_rates): n_hel the relics per helicity and mass
state today, eta_r their occupation (T_r = T eta_r^(1/3)), M_t^2 = m_Zd^2 + m_D^2(z). Up-conversions cost energy,
down-conversions return it.

The step (shift_step): each bin's content moves down by its loss over the step, spread over the bins it then
overlaps -- the number kept exactly (what crosses the grid's lower edge leaves it), for any loss per step.
"""
import numpy as np
from numba import njit

# NumPy 2 renamed trapz to trapezoid (CI runs NumPy 2, the pinned environment 1.26)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

ZETA3 = 1.2020569031595942


def mean_inv_E(m, T, n=600):
    """<1/E> over a Fermi-Dirac momentum distribution at temperature T (eV), mass m (eV)."""
    x = np.linspace(1e-5, 40.0, n)
    f = x * x / (np.exp(x) + 1.0)
    p = x * T
    return float(_trapz(f / np.sqrt(p * p + m * m), x) / _trapz(f, x))


def loss_coefficient(G_flavor, U, masses, eta, T_nu0, z, M, mD2):
    """C(z) (eV^2): dE/dt = -C(z) (G G^dag)_aa for a beam neutrino in state a. G_flavor the non-universal coupling
    (as dp_decay.rest_rates), U the mixing matrix, masses and eta per mass state, T_nu0 the relic temperature today,
    M the mediator mass, mD2 the Debye mass squared at z (eV^2)."""
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    m = np.asarray(masses, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    Gp = U.conj().T @ (G - G[0, 0].real * np.eye(3)) @ U
    n_hel0 = 3.0 * ZETA3 * T_nu0 ** 3 / (4.0 * np.pi ** 2)
    acc = 0.0
    for r in range(3):
        if eta[r] <= 0.0:
            continue
        Tr = T_nu0 * (1.0 + z) * eta[r] ** (1.0 / 3.0)
        inv = mean_inv_E(m[r], Tr)
        for s in range(3):
            if s == r:
                continue
            acc += eta[r] * abs(Gp[s, r]) ** 2 * 0.5 * (m[s] ** 2 - m[r] ** 2) * inv
    return 2.0 * n_hel0 * (1.0 + z) ** 3 * acc / (4.0 * np.pi * (M * M + mD2))


@njit(cache=True)
def shift_step(E, F, dE_obs):
    """Move each bin's content down by dE_obs[j, a] (observed energy, eV; negative moves it up), spread uniformly over
    the shifted bin, in place. E (N,) a geometric grid, F (N, n) populations per unit energy."""
    N = E.shape[0]
    n = F.shape[1]
    h = np.sqrt(E[1] / E[0])
    h2 = h * h
    lnh2 = np.log(h2)
    edge0 = E[0] / h
    new = np.zeros_like(F)
    for a in range(n):
        for j in range(N):
            lo = E[j] / h
            hi = E[j] * h
            Nj = F[j, a] * (hi - lo)
            if Nj == 0.0:
                continue
            d = dE_obs[j, a]
            if d == 0.0:
                new[j, a] += F[j, a]
                continue
            lo2 = lo - d
            hi2 = hi - d
            if hi2 <= edge0:
                continue
            k0 = int(np.floor(np.log(max(lo2, edge0) / edge0) / lnh2))
            k1 = int(np.floor(np.log(hi2 / edge0) / lnh2))
            if k0 < 0:
                k0 = 0
            if k1 > N - 1:
                k1 = N - 1
            w = hi2 - lo2
            for k in range(k0, k1 + 1):
                eklo = edge0 * np.exp(lnh2 * k)
                ekhi = eklo * h2
                ov = min(hi2, ekhi) - max(lo2, eklo)
                if ov > 0.0:
                    new[k, a] += Nj * ov / w / (ekhi - eklo)
    for j in range(N):
        for a in range(n):
            F[j, a] = new[j, a]


@njit(cache=True)
def interp_sigma_table(E_bins, table, log_s0, inv_dlogs, c_z, out):
    """A per-state table on the sigma nodes (n, 3) at every energy of a step, linear in log|sigma| (as
    dp_cosmo._cosmo_projections)."""
    N_sig = table.shape[0]
    for i in range(E_bins.shape[0]):
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
        for a in range(3):
            out[i, a] = (1.0 - w) * table[k0, a] + w * table[k0 + 1, a]
