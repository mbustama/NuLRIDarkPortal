"""
dp_galactic_path.py -- the Galactic component's incoherent processes along each line of sight (2026-10-24; notes
Sec. 12.8): the soft scatterings on the relics, the decay into the mediator and the relic mass-change energy loss,
which act on the Galactic neutrinos over their kpc to the Sun as on the extragalactic ones over cosmological
distances (dark_portal_solver's cosmological phase), at z = 0.

Each pixel's line of sight is swept from its farthest emission bin to the Sun: at each bin its emission is added (the
populations of the local eigenstates at the emission point, dp_galactic's cloud-in-cell nodes, times the emitted
intensity and the gas and source weights), then the populations evolve over the distance to the next bin -- in the
eigenstates of the local potential at the segment's midpoint (sigma = E coef2 n, as the emission) -- with the jumps
and the decays alternating n_sub times (dark_portal_solver._jump_decay_cycles: the cycles' pace), and the energy loss.
At the Sun the populations, divided by the emitted intensity, are what dp_galactic's emission populations stand for
(per unit flux emitted, the local eigenstates at the Sun): the caller's band responses and intensities apply as
before. Both kinds are swept together (a decay's flipped daughter belongs to the other kind).
"""
import numpy as np
from numba import njit, prange

import dp_cosmo
from dp_decay import decay_step
from dp_relic_loss import shift_step, interp_sigma_table

# serial copies of the transport's energy-parallel steps (prange runs as range without parallel=True), for the
# pixel-parallel sweep (with the parallel ones inside, the 32-core sweep was no faster than one core, 2026-10-24)
_soft_jumps = njit(dp_cosmo._soft_jumps.py_func)
_cosmo_projections = njit(fastmath=True, error_model="numpy")(dp_cosmo._cosmo_projections.py_func)

KPC_EV_INV = 3.0856775814913673e21 / 1.973269804e-5      # 1 kpc in eV^-1
CYCLE_STEP = 0.25
CYCLE_MAX = 64


def build_tables(M_rot, S_signed, U_kind, G_jump_kind, sig_min, sig_max, n_sigma=2048):
    """The jumps' and the mass content's tables on sigma nodes for one kind (dp_cosmo's builders), over the
    Galaxy's sigma range: dict of arrays and the log-sigma grid."""
    from dp_cosmo import _build_sigma_jump_table, _build_sigma_mass_table
    from dp_transit import texture_scales
    Q_t, s_t, Mp_t, floor_t, sig_lo_t, sig_hi_t = texture_scales(M_rot, S_signed)
    lo = max(float(sig_min), 1e-30 * max(float(np.max(np.abs(M_rot))), 1e-300))
    hi = max(float(sig_max), lo * 2.0)
    if sig_hi_t > 0.0 and hi > sig_hi_t:
        hi = max(sig_hi_t, lo * 2.0)
    nodes = np.geomspace(lo, hi, n_sigma)
    J, Gd, GG, lam = _build_sigma_jump_table(Q_t, Mp_t, s_t, floor_t, nodes, np.ascontiguousarray(G_jump_kind))
    T = _build_sigma_mass_table(Q_t, Mp_t, s_t, floor_t, nodes, np.ascontiguousarray(U_kind))
    return dict(J=J, Gd=Gd, GG=GG, lam=lam, T=T, log_s0=float(np.log(nodes[0])),
                inv_dlogs=float((n_sigma - 1) / np.log(hi / lo)))


@njit(cache=True)
def _evolve(E, Fk0, Fk1, L, c0, c1, J0, Gd0, GG0, lam0, T0, ls0, ids0, J1, Gd1, GG1, lam1, T1, ls1, ids1,
            jump_unit, use_jumps, jmax, Gam, spec, m, width_unit, rdec_max, use_dec, loss_C, use_loss, clip,
            Tb0, Tb1, Wd0, Wd1, GGb, dEo):
    """Both kinds' populations over a segment of length L (eV^-1) at sigma coefficients c0, c1: the jumps and the
    decays alternating (the cycles' pace, symmetric), the energy loss in halves around them. Linear in the populations with clip off."""
    N_E = E.shape[0]
    if use_dec:
        _cosmo_projections(E, T0, ls0, ids0, c0, Tb0)
        _cosmo_projections(E, T1, ls1, ids1, c1, Tb1)
        for i in range(N_E):
            for a in range(3):
                acc0 = 0.0
                acc1 = 0.0
                for mm in range(3):
                    acc0 += Tb0[i, mm, a] * width_unit[mm]
                    acc1 += Tb1[i, mm, a] * width_unit[mm]
                Wd0[i, a] = acc0 / E[i]
                Wd1[i, a] = acc1 / E[i]
    if use_loss:
        _loss(E, Fk0, Fk1, 0.5 * L, c0, c1, GG0, ls0, ids0, GG1, ls1, ids1, loss_C, GGb, dEo)
    n_sub = 1
    if use_dec and use_jumps:
        pace = min(jump_unit * jmax * L, rdec_max * L)
        n_sub = int(min(CYCLE_MAX, max(1.0, np.ceil(pace / CYCLE_STEP))))
    # symmetric (Strang): half a sub-step's decays before the first jumps and after the last
    if use_dec:
        decay_step(E, 1.0, 0.5 * L / n_sub, Gam, spec, m, Fk0, Fk1, Tb0, Tb1, True, False)
    for s_ in range(n_sub):
        if use_jumps:
            _soft_jumps(E, 1.0, J0, Gd0, GG0, lam0, ls0, ids0, c0, jump_unit, L / n_sub, Fk0, Wd0, use_dec, clip)
            _soft_jumps(E, 1.0, J1, Gd1, GG1, lam1, ls1, ids1, c1, jump_unit, L / n_sub, Fk1, Wd1, use_dec, clip)
        if use_dec:
            decay_step(E, 1.0, (0.5 if s_ == n_sub - 1 else 1.0) * L / n_sub, Gam, spec, m, Fk0, Fk1, Tb0, Tb1, True,
                       False)
    if use_loss:
        _loss(E, Fk0, Fk1, 0.5 * L, c0, c1, GG0, ls0, ids0, GG1, ls1, ids1, loss_C, GGb, dEo)


@njit(cache=True)
def _loss(E, Fk0, Fk1, L, c0, c1, GG0, ls0, ids0, GG1, ls1, ids1, loss_C, GGb, dEo):
    """The relic mass-change energy loss over a path L (eV^-1), both kinds."""
    N_E = E.shape[0]
    interp_sigma_table(E, GG0, ls0, ids0, c0, GGb)
    for i in range(N_E):
        for a in range(3):
            dEo[i, a] = loss_C * GGb[i, a] * L
    shift_step(E, Fk0, dEo)
    interp_sigma_table(E, GG1, ls1, ids1, c1, GGb)
    for i in range(N_E):
        for a in range(3):
            dEo[i, a] = loss_C * GGb[i, a] * L
    shift_step(E, Fk1, dEo)


@njit(cache=True, parallel=True)
def _sweep(E, I, w_gas, q_gas, q_src, Pn, jn, tn, cseg, seglen,
           J0, Gd0, GG0, lam0, T0, ls0, ids0, J1, Gd1, GG1, lam1, T1, ls1, ids1,
           jump_unit, use_jumps, Gam, spec, m, width_unit, use_dec, loss_C, use_loss, n_split, out):
    """The sweep (module docstring). Pn (2, n_nodes, N_E, 3) the populations at the emission nodes per kind; jn, tn
    (2, N_pix, N_s) each emission bin's node and weight; cseg (2, N_pix, N_s) the sigma coefficient over the segment
    from bin j to the next nearer (j = 0: to the Sun); seglen (N_s,) those lengths (eV^-1); out (2, N_pix, N_E, 3)."""
    N_E = E.shape[0]
    npix = I.shape[0]
    ns = q_gas.shape[1]
    rdec_max = 0.0
    for a in range(3):
        rdec_max = max(rdec_max, width_unit[a] / E[0])
    jmax0 = 0.0
    jmax1 = 0.0
    for k in range(J0.shape[0]):
        for b in range(3):
            for a in range(3):
                jmax0 = max(jmax0, J0[k, b, a])
                jmax1 = max(jmax1, J1[k, b, a])
    jmax = max(jmax0, jmax1)
    for p in prange(npix):
        # the pixels are independent lines of sight: each thread its own work arrays
        Fk0 = np.zeros((N_E, 3))
        Fk1 = np.zeros((N_E, 3))
        Tb0 = np.zeros((N_E, 3, 3))
        Tb1 = np.zeros((N_E, 3, 3))
        Wd0 = np.zeros((N_E, 3))
        Wd1 = np.zeros((N_E, 3))
        GGb = np.zeros((N_E, 3))
        dEo = np.zeros((N_E, 3))
        for j in range(ns - 1, -1, -1):
            # the bin's emission, both kinds
            for kind in range(2):
                F = Fk0 if kind == 0 else Fk1
                n0 = jn[kind, p, j]
                t = tn[kind, p, j]
                for i in range(N_E):
                    wq = w_gas[p, i] * q_gas[p, j] + (1.0 - w_gas[p, i]) * q_src[p, j]
                    if wq == 0.0:
                        continue
                    amp = I[p, i] * wq
                    for a in range(3):
                        F[i, a] += amp * ((1.0 - t) * Pn[kind, n0, i, a] + t * Pn[kind, n0 + 1, i, a])
            L = seglen[j]
            if L <= 0.0:
                continue
            for _ in range(n_split):
                _evolve(E, Fk0, Fk1, L / n_split, cseg[0, p, j], cseg[1, p, j], J0, Gd0, GG0, lam0, T0, ls0, ids0,
                        J1, Gd1, GG1, lam1, T1, ls1, ids1, jump_unit, use_jumps, jmax, Gam, spec, m, width_unit,
                        rdec_max, use_dec, loss_C, use_loss, True, Tb0, Tb1, Wd0, Wd1, GGb, dEo)
        for i in range(N_E):
            inv = 1.0 / I[p, i] if I[p, i] > 0.0 else 0.0
            for a in range(3):
                out[0, p, i, a] = Fk0[i, a] * inv
                out[1, p, i, a] = Fk1[i, a] * inv


def node_populations(E_eV, f_S, n_emit, coef2, tab, nodes_per_decade):
    """The cloud-in-cell nodes of the emission densities (as dp_galactic.emission_populations): the populations at
    each node (n_nodes, N_E, 3) and each emission bin's node and weight (N_pix, N_s)."""
    import dp_transit
    E_eV = np.ascontiguousarray(E_eV, dtype=np.float64)
    F_fl = np.ascontiguousarray(np.broadcast_to(np.asarray(f_S, dtype=np.float64), (len(E_eV), 3)))
    logn = np.log10(np.maximum(n_emit, 1e-300))
    d = 1.0 / nodes_per_decade
    lo = np.floor(logn.min() / d) * d
    x = (logn - lo) / d
    j = np.floor(x).astype(np.int64)
    t = x - j
    n_nodes = int(j.max()) + 2
    P = np.zeros((n_nodes, len(E_eV), 3))
    out = np.empty((len(E_eV), 3))
    for node in np.unique(np.concatenate([j.ravel(), j.ravel() + 1])):
        dp_transit.flavor_populations(E_eV, F_fl, coef2, 10.0 ** (lo + node * d), tab["Q"], tab["Mp"], tab["s"],
                                      tab["floor"], out)
        P[node] = out
    return P, j, t


def sweep_populations(E_eV, I, w_gas, q_gas, q_src, s_kpc, kinds, rates, n_split=1):
    """The Galactic populations at the Sun per unit flux emitted, per kind, (2, N_pix, N_E, 3), with the processes.
    kinds: two dicts (neutrinos, antineutrinos) with P (n_nodes, N_E, 3), j, t (N_pix, N_s), n (N_pix, N_s) the
    emission densities, coef2, tables (build_tables). rates: dict(jump_unit, Gam, spec, m, loss_C) (spec:
    dp_decay.daughter_spectra). n_split: each
    segment evolved in that many equal pieces (the convergence check)."""
    E = np.ascontiguousarray(E_eV, dtype=np.float64)
    npix, ns = q_gas.shape
    n_nodes = max(k["P"].shape[0] for k in kinds)
    Pn = np.zeros((2, n_nodes, len(E), 3))
    jn = np.zeros((2, npix, ns), dtype=np.int64)
    tn = np.zeros((2, npix, ns))
    cseg = np.zeros((2, npix, ns))
    for kk, k in enumerate(kinds):
        Pn[kk, :k["P"].shape[0]] = k["P"]
        jn[kk] = k["j"]
        tn[kk] = k["t"]
        n = k["n"]
        nmid = np.empty_like(n)
        nmid[:, 1:] = np.sqrt(np.maximum(n[:, 1:], 1e-300) * np.maximum(n[:, :-1], 1e-300))
        nmid[:, 0] = n[:, 0]
        cseg[kk] = k["coef2"] * nmid
    s = np.asarray(s_kpc, dtype=np.float64)
    seglen = np.empty(ns)
    seglen[0] = s[0] * KPC_EV_INV
    seglen[1:] = np.diff(s) * KPC_EV_INV
    t0, t1 = kinds[0]["tables"], kinds[1]["tables"]
    Gam = np.ascontiguousarray(rates["Gam"], dtype=np.float64)
    m = np.ascontiguousarray(rates["m"], dtype=np.float64)
    width_unit = Gam.sum(axis=1) * m
    out = np.zeros((2, npix, len(E), 3))
    _sweep(E, np.ascontiguousarray(I), np.ascontiguousarray(w_gas), np.ascontiguousarray(q_gas),
           np.ascontiguousarray(q_src), Pn, jn, tn, cseg, seglen,
           t0["J"], t0["Gd"], t0["GG"], t0["lam"], t0["T"], t0["log_s0"], t0["inv_dlogs"],
           t1["J"], t1["Gd"], t1["GG"], t1["lam"], t1["T"], t1["log_s0"], t1["inv_dlogs"],
           float(rates["jump_unit"]), rates["jump_unit"] > 0.0, Gam, np.ascontiguousarray(rates["spec"]), m,
           width_unit, bool(np.any(Gam > 0.0)), float(rates["loss_C"]), rates["loss_C"] != 0.0, int(n_split), out)
    return out


@njit(cache=True)
def _basis_paths(E, Y0, Y1, seglen, J0, Gd0, GG0, lam0, T0, ls0, ids0, J1, Gd1, GG1, lam1, T1, ls1, ids1,
                 jump_unit, use_jumps, Gam, spec, m, use_dec, loss_C, use_loss, n_split, X):
    """No potential in the Galaxy (sigma = 0): the evolution depends only on the path length, and paths compose
    (homogeneous). X[j] (2, N_E, 3) = the pair (Y0, Y1) evolved over s_j, built incrementally over the segment
    lengths (seglen[0] = s_0, then s_j - s_{j-1}); linear (clip off: the spectral bases change sign)."""
    N_E = E.shape[0]
    Tb0 = np.zeros((N_E, 3, 3))
    Tb1 = np.zeros((N_E, 3, 3))
    Wd0 = np.zeros((N_E, 3))
    Wd1 = np.zeros((N_E, 3))
    GGb = np.zeros((N_E, 3))
    dEo = np.zeros((N_E, 3))
    width_unit = np.zeros(3)
    rdec_max = 0.0
    for a in range(3):
        for b in range(3):
            width_unit[a] += Gam[a, b]
        width_unit[a] *= m[a]
        rdec_max = max(rdec_max, width_unit[a] / E[0])
    jmax = 0.0
    for k in range(J0.shape[0]):
        for b in range(3):
            for a in range(3):
                jmax = max(jmax, max(J0[k, b, a], J1[k, b, a]))
    F0 = Y0.copy()
    F1 = Y1.copy()
    for j in range(seglen.shape[0]):
        for _ in range(n_split):
            _evolve(E, F0, F1, seglen[j] / n_split, 0.0, 0.0, J0, Gd0, GG0, lam0, T0, ls0, ids0, J1, Gd1, GG1, lam1,
                    T1, ls1, ids1, jump_unit, use_jumps, jmax, Gam, spec, m, width_unit, rdec_max, use_dec, loss_C,
                    use_loss, False, Tb0, Tb1, Wd0, Wd1, GGb, dEo)
        for i in range(N_E):
            for a in range(3):
                X[j, 0, i, a] = F0[i, a]
                X[j, 1, i, a] = F1[i, a]


def vacuum_populations(E_eV, I, w_gas, q_gas, q_src, s_kpc, P_vac, tables, rates, work, nside, tol=1e-10,
                       n_split=1):
    """sweep_populations where the Galaxy's potential is negligible (2026-10-24): every line of sight evolves alike
    but for its length, so the emitted spectra, split into a few shapes (the gas and the sources' parts, each a
    truncated SVD over the pixels, singular values above tol of the largest), are evolved once over the distance
    bins and combined per pixel with the distance weights -- at the output resolution (the children's sums, the
    populations then divided by the summed intensity: dp_galactic.degrade_weighted's average). P_vac (2, N_E, 3)
    the vacuum populations at emission per kind. Returns (2, N_pix_out, N_E, 3)."""
    import dp_galactic
    E = np.ascontiguousarray(E_eV, dtype=np.float64)
    s = np.asarray(s_kpc, dtype=np.float64)
    seglen = np.empty(len(s))
    seglen[0] = s[0] * KPC_EV_INV
    seglen[1:] = np.diff(s) * KPC_EV_INV
    t0, t1 = tables
    Gam = np.ascontiguousarray(rates["Gam"], dtype=np.float64)
    m = np.ascontiguousarray(rates["m"], dtype=np.float64)

    def down(x):
        return x if work == nside else dp_galactic.degrade(x, work, nside)

    I_out = down(np.asarray(I, dtype=np.float64))
    total = None
    # the SVD of the emission scaled by the sky-mean intensity per energy, so the truncation is relative at every
    # energy (unscaled, the steep spectrum let it drop 3e-4 of the faint high energies)
    Ibar = np.asarray(I, dtype=np.float64).mean(axis=0)
    Ibar = np.where(Ibar > 0.0, Ibar, 1.0)
    for A, q in ((np.asarray(I) * np.asarray(w_gas), q_gas), (np.asarray(I) * (1.0 - np.asarray(w_gas)), q_src)):
        U_, S_, Vt = np.linalg.svd(A / Ibar[None, :], full_matrices=False)
        keep = S_ >= tol * S_[0] if S_[0] > 0.0 else np.zeros_like(S_, dtype=bool)
        for kb in np.nonzero(keep)[0]:
            u = Vt[kb] * Ibar
            X = np.zeros((len(s), 2, len(E), 3))
            _basis_paths(E, np.ascontiguousarray(u[:, None] * P_vac[0]), np.ascontiguousarray(u[:, None] * P_vac[1]),
                         seglen, t0["J"], t0["Gd"], t0["GG"], t0["lam"], t0["T"], t0["log_s0"], t0["inv_dlogs"],
                         t1["J"], t1["Gd"], t1["GG"], t1["lam"], t1["T"], t1["log_s0"], t1["inv_dlogs"],
                         float(rates["jump_unit"]), rates["jump_unit"] > 0.0, Gam, np.ascontiguousarray(rates["spec"]),
                         m, bool(np.any(Gam > 0.0)), float(rates["loss_C"]), rates["loss_C"] != 0.0, int(n_split), X)
            W = down((U_[:, kb] * S_[kb])[:, None] * np.asarray(q))          # (N_pix_out, N_s)
            c = W @ X.reshape(len(s), -1)
            total = c if total is None else total + c
    out = total.reshape(-1, 2, len(E), 3).transpose(1, 0, 2, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(I_out[None, :, :, None] > 0.0, out / I_out[None, :, :, None], 0.0)
