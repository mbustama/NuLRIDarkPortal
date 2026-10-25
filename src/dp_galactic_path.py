"""
dp_galactic_path.py -- the Galactic component's incoherent processes along each line of sight (2026-10-24; notes
Sec. 12.8): the soft scatterings on the relics, the decay into the mediator and the relic mass-change energy loss,
which act on the Galactic neutrinos over their kpc to the Sun as on the extragalactic ones over cosmological
distances (dark_portal_solver's cosmological phase), at z = 0.

Each pixel's line of sight is swept from its farthest emission bin to the Sun, bin by bin between the distance bins'
edges: over a bin its emission is a constant source (the populations of the local eigenstates at the emission point,
dp_galactic's cloud-in-cell nodes, times the emitted intensity and the gas and source weights, spread over the bin's
length), and the populations evolve in the eigenstates of the local potential (sigma = E coef2 n, the bin's density)
-- every energy bin solved exactly over the segment with the jumps, the decays and the source together
(dp_sweep.step_full at z = 0, 2026-10-25: the review's finding 4; the jumps and the decays alternated before), the
energy loss in two halves around it; then from the nearest bin's edge to the Sun without emission. The decays act
between the eigenstates of the local potential with their own kinematics (dp_medium: the plans of a ladder built once
per run over the Galaxy's sigma range, the nearest rung per segment).
At the Sun the populations, divided by the emitted intensity, are what dp_galactic's emission populations stand for
(per unit flux emitted, the local eigenstates at the Sun): the caller's band responses and intensities apply as
before. Both kinds are swept together (a decay's flipped daughter belongs to the other kind).
"""
import numpy as np
from numba import njit, prange

import dp_medium
import dp_sweep
from dp_relic_loss import shift_step, interp_sigma_table

KPC_EV_INV = 3.0856775814913673e21 / 1.973269804e-5      # 1 kpc in eV^-1
# a segment's decays (jumps) are left out where no state's width (conversion rate) x length exceeds this at any energy
DECAY_SKIP = 1e-12
JUMP_SKIP = 1e-12
# the segment's ladder coordinate rounded to a rung (False: the two rungs around it interpolated, as the cosmology)
GALACTIC_NEAREST_RUNG = True


def edges_from_geometry(s_kpc, ds_kpc):
    """The distance bins' edges (N_s + 1,) from their geometric centres s and widths ds (dp_galactic.emission_geometry:
    s = sqrt(e_j e_j+1), ds = e_j+1 - e_j)."""
    s = np.asarray(s_kpc, dtype=np.float64)
    ds = np.asarray(ds_kpc, dtype=np.float64)
    lo = 0.5 * (np.sqrt(ds * ds + 4.0 * s * s) - ds)
    return np.append(lo, lo[-1] + ds[-1])


def build_tables(M_rot, S_signed, U_kind, G_jump_kind, sig_min, sig_max, n_sigma=2048):
    """The jumps' tables on sigma nodes for one kind (dp_cosmo's builder), over the Galaxy's sigma range: dict of
    arrays and the log-sigma grid (U_kind unused since the decays moved to dp_medium, 2026-10-25)."""
    from dp_cosmo import _build_sigma_jump_table
    from dp_transit import texture_scales
    Q_t, s_t, Mp_t, floor_t, sig_lo_t, sig_hi_t = texture_scales(M_rot, S_signed)
    lo = max(float(sig_min), 1e-30 * max(float(np.max(np.abs(M_rot))), 1e-300))
    hi = max(float(sig_max), lo * 2.0)
    if sig_hi_t > 0.0 and hi > sig_hi_t:
        hi = max(sig_hi_t, lo * 2.0)
    nodes = np.geomspace(lo, hi, n_sigma)
    J, Gd, GG, lam = _build_sigma_jump_table(Q_t, Mp_t, s_t, floor_t, nodes, np.ascontiguousarray(G_jump_kind))
    return dict(J=J, Gd=Gd, GG=GG, lam=lam, log_s0=float(np.log(nodes[0])),
                inv_dlogs=float((n_sigma - 1) / np.log(hi / lo)))


@njit(cache=True)
def _work(n, C):
    """Per-thread work arrays of _segment for C columns and n energy bins: dp_medium's buffers per column (one rung set:
    the nearest rung), the validity measure, the widths and the loss's."""
    S = 1 if GALACTIC_NEAREST_RUNG else 2
    return (np.zeros((C, n, 2, 3)), np.zeros((C, 2, dp_medium.NCL, dp_medium.NMOM, 9)),
            np.zeros((C, n, S, dp_medium.NCL, dp_medium.NMOM, 9)),
            np.zeros((C, n, S, dp_medium.NCL, dp_medium.NMOM, 9)),
            np.zeros((C, 2, dp_medium.NCL), dtype=np.bool_), np.zeros(2), np.zeros((n, 3)), np.zeros((n, 3)),
            np.zeros((n, 3)), np.zeros((n, 3)))


@njit(cache=True)
def _segment(E, F, L, c0, c1, src, use_src, J0, Gd0, GG0, lam0, ls0, ids0, J1, Gd1, GG1, lam1, ls1, ids1,
             jump_unit, use_jumps, jmax, lad, use_dec, loss_C, use_loss, clip, wk):
    """The C columns of both kinds' populations F (C, N_E, 2, 3) over a segment of length L (eV^-1) at sigma
    coefficients c0, c1, with the sources src (C, N_E, 2, 3) per unit energy and length if use_src: the energy loss in
    halves around dp_sweep.step_columns (z = 0, the segment as its time; the jumps, the decays and the sources of each
    bin solved together, exactly; the columns share every bin's exponentials). Linear in the populations and the
    sources with clip off. wk: _work; wk[5][0] the largest validity measure met (updated)."""
    N_E = E.shape[0]
    h = np.sqrt(E[1] / E[0])
    pend, acc, add, sub, act, diag, Wd0, Wd1, GGb, dEo = wk
    x0 = 0.0
    dec = False
    if use_dec:
        x0 = dp_medium.ladder_x0(lad, E[0], h, c0, N_E)
        if GALACTIC_NEAREST_RUNG and x0 >= 0.0:
            # the segment's coefficient to the nearest rung of the ladder (its sigma within the grid's half-bin ratio,
            # 1%: below the segment's own density approximation; one plan per bin instead of two)
            x0 = np.floor(x0 + 0.5)
        dp_medium.widths(lad, x0, E, Wd0, Wd1)
        rmax = 0.0
        for i in range(N_E):
            for a in range(3):
                rmax = max(rmax, Wd0[i, a], Wd1[i, a])
        dec = rmax * L > DECAY_SKIP
    jumps = use_jumps and jump_unit * jmax * L > JUMP_SKIP
    if use_loss:
        for col in range(F.shape[0]):
            _loss(E, F[col, :, 0, :], F[col, :, 1, :], 0.5 * L, c0, c1, GG0, ls0, ids0, GG1, ls1, ids1, loss_C, GGb,
                  dEo)
    if dec or jumps or use_src:
        dp_sweep.step_columns(L, E, F, jumps, jump_unit, J0, Gd0, GG0, lam0, ls0, ids0, c0, J1, Gd1, GG1, lam1, ls1,
                              ids1, c1, lad, x0, dec, clip, diag, pend, acc, add, sub, act, src, use_src)
    if use_loss:
        for col in range(F.shape[0]):
            _loss(E, F[col, :, 0, :], F[col, :, 1, :], 0.5 * L, c0, c1, GG0, ls0, ids0, GG1, ls1, ids1, loss_C, GGb,
                  dEo)


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
def _sweep(E, I, w_gas, q_gas, q_src, Pn, jn, tn, gstart, gchild, cseg, seglen, L_sun,
           J0, Gd0, GG0, lam0, ls0, ids0, J1, Gd1, GG1, lam1, ls1, ids1,
           jump_unit, use_jumps, lad, use_dec, loss_C, use_loss, n_split, out, zeta):
    """The sweep (module docstring), per line of sight g of the path sky (its pixels gchild[gstart[g]:gstart[g+1]]
    of the emission's sky, whose emission it carries, at the densities' mean): Pn (C, 2, n_nodes, N_E, 3) the
    populations at the emission nodes per column (composition) and kind; jn, tn (2, N_pix, N_s) each emission bin's
    node and weight; cseg (2, N_g, N_s) the sigma coefficient over bin j; seglen (N_s,) the bins' lengths and L_sun
    the nearest bin's distance to the Sun (eV^-1); out (C, 2, N_pix, N_E, 3): every pixel of a line of sight its
    populations per unit intensity emitted; zeta (N_g,) the largest validity measure per line of sight."""
    N_E = E.shape[0]
    ng = gstart.shape[0] - 1
    ns = q_gas.shape[1]
    C = Pn.shape[0]
    jmax = 0.0
    for k in range(J0.shape[0]):
        for b in range(3):
            for a in range(3):
                jmax = max(jmax, J0[k, b, a], J1[k, b, a])
    for g in prange(ng):
        # the lines of sight are independent: each thread its own work arrays
        F = np.zeros((C, N_E, 2, 3))
        src = np.zeros((C, N_E, 2, 3))
        Isum = np.zeros(N_E)
        wk = _work(N_E, C)
        for x in range(gstart[g], gstart[g + 1]):
            p = gchild[x]
            for i in range(N_E):
                Isum[i] += I[p, i]
        for j in range(ns - 1, -1, -1):
            L = seglen[j]
            if L <= 0.0:
                continue
            for col in range(C):
                for i in range(N_E):
                    for kind in range(2):
                        for a in range(3):
                            src[col, i, kind, a] = 0.0
            # the bin's emission per unit length, both kinds, every pixel of the line of sight
            for x in range(gstart[g], gstart[g + 1]):
                p = gchild[x]
                for kind in range(2):
                    n0 = jn[kind, p, j]
                    t = tn[kind, p, j]
                    for i in range(N_E):
                        wq = w_gas[p, i] * q_gas[p, j] + (1.0 - w_gas[p, i]) * q_src[p, j]
                        amp = I[p, i] * wq / L
                        if amp == 0.0:
                            continue
                        for col in range(C):
                            for a in range(3):
                                src[col, i, kind, a] += amp * ((1.0 - t) * Pn[col, kind, n0, i, a]
                                                               + t * Pn[col, kind, n0 + 1, i, a])
            for _ in range(n_split):
                _segment(E, F, L / n_split, cseg[0, g, j], cseg[1, g, j], src, True, J0, Gd0, GG0, lam0, ls0, ids0,
                         J1, Gd1, GG1, lam1, ls1, ids1, jump_unit, use_jumps, jmax, lad, use_dec, loss_C, use_loss,
                         True, wk)
        if L_sun > 0.0:
            for _ in range(n_split):
                _segment(E, F, L_sun / n_split, cseg[0, g, 0], cseg[1, g, 0], src, False, J0, Gd0, GG0, lam0, ls0,
                         ids0, J1, Gd1, GG1, lam1, ls1, ids1, jump_unit, use_jumps, jmax, lad, use_dec, loss_C,
                         use_loss, True, wk)
        zeta[g] = wk[5][0]
        for x in range(gstart[g], gstart[g + 1]):
            p = gchild[x]
            for i in range(N_E):
                inv = 1.0 / Isum[i] if Isum[i] > 0.0 else 0.0
                for col in range(C):
                    for kind in range(2):
                        for a in range(3):
                            out[col, kind, p, i, a] = F[col, i, kind, a] * inv


@njit(cache=True, parallel=True)
def _cross(E, P, cseg, seglen, L_sun, J0, Gd0, GG0, lam0, ls0, ids0, J1, Gd1, GG1, lam1, ls1, ids1,
           jump_unit, use_jumps, lad, use_dec, loss_C, use_loss, n_split, out, zeta):
    """The extragalactic beam across the Galaxy: the populations P (C, N_E, 2, 3) entering every line of sight g at
    its far end (the local eigenstates in order: the transit is adiabatic) carried to the Sun through its distance
    bins with the processes and no emission; out (N_g, C, N_E, 2, 3); zeta (N_g,)."""
    N_E = E.shape[0]
    ng = cseg.shape[1]
    ns = seglen.shape[0]
    C = P.shape[0]
    jmax = 0.0
    for k in range(J0.shape[0]):
        for b in range(3):
            for a in range(3):
                jmax = max(jmax, J0[k, b, a], J1[k, b, a])
    for g in prange(ng):
        F = P.copy()
        src = np.zeros((1, 1, 2, 3))
        wk = _work(N_E, C)
        for j in range(ns - 1, -1, -1):
            L = seglen[j]
            if L <= 0.0:
                continue
            for _ in range(n_split):
                _segment(E, F, L / n_split, cseg[0, g, j], cseg[1, g, j], src, False, J0, Gd0, GG0, lam0, ls0, ids0,
                         J1, Gd1, GG1, lam1, ls1, ids1, jump_unit, use_jumps, jmax, lad, use_dec, loss_C, use_loss,
                         False, wk)
        if L_sun > 0.0:
            for _ in range(n_split):
                _segment(E, F, L_sun / n_split, cseg[0, g, 0], cseg[1, g, 0], src, False, J0, Gd0, GG0, lam0, ls0,
                         ids0, J1, Gd1, GG1, lam1, ls1, ids1, jump_unit, use_jumps, jmax, lad, use_dec, loss_C,
                         use_loss, False, wk)
        zeta[g] = wk[5][0]
        out[g] = F


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


def _segments(edges_kpc):
    """(seglen (N_s,), L_sun): the distance bins' lengths and the nearest bin's distance to the Sun, eV^-1."""
    e = np.asarray(edges_kpc, dtype=np.float64)
    return np.diff(e) * KPC_EV_INV, float(e[0]) * KPC_EV_INV


def groups_of(nside_pix, nside_path):
    """The path sky's lines of sight (N_side nside_path, a divisor of nside_pix) as groups of the emission sky's
    pixels (RING): (gstart (N_g + 1,), gchild (N_pix,)); every pixel lies inside one coarser pixel (HEALPix)."""
    import dp_sky
    npix = dp_sky.nside_to_npix(nside_pix)
    if nside_path == nside_pix:
        return np.arange(npix + 1, dtype=np.int64), np.arange(npix, dtype=np.int64)
    if nside_pix % nside_path:
        raise ValueError(f"N_side {nside_path} does not divide {nside_pix}")
    th, ph = dp_sky.pix2ang_ring(nside_pix, np.arange(npix))
    parent = dp_sky.ang2pix_ring(nside_path, th, ph)
    order = np.argsort(parent, kind="stable")
    counts = np.bincount(parent, minlength=dp_sky.nside_to_npix(nside_path))
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64), order.astype(np.int64)


def _group_cseg(kinds, gstart, gchild):
    """(2, N_g, N_s): each line of sight's sigma coefficient per distance bin, coef2 x its pixels' mean density."""
    ns = np.asarray(kinds[0]["n"]).shape[1]
    ng = len(gstart) - 1
    cseg = np.zeros((2, ng, ns))
    for kk, k in enumerate(kinds):
        n = np.asarray(k["n"], dtype=np.float64)
        for g in range(ng):
            cseg[kk, g] = k["coef2"] * n[gchild[gstart[g]:gstart[g + 1]]].mean(axis=0)
    return cseg


def sweep_populations(E_eV, I, w_gas, q_gas, q_src, edges_kpc, kinds, rates, n_split=1, diag=None, groups=None):
    """The Galactic populations at the Sun per unit flux emitted, per kind, (2, N_pix, N_E, 3) -- or (C, 2, N_pix,
    N_E, 3) for C compositions at once -- with the processes. edges_kpc (N_s + 1,): the distance bins' edges
    (edges_from_geometry). kinds: two dicts (neutrinos, antineutrinos) with P (n_nodes, N_E, 3), or (C, n_nodes,
    N_E, 3) for C compositions, j, t (N_pix, N_s), n (N_pix, N_s) the emission densities, coef2, tables
    (build_tables). rates: dict(jump_unit, ladder, loss_C) (ladder: a dp_medium.Ladder over the Galaxy's sigma range,
    or None without decays). n_split: each segment evolved in that many equal pieces (the convergence check). diag: a
    dict that gets 'zeta', the largest decay width over splitting met (dp_medium's validity measure). groups: None
    (every pixel its own line of sight) or groups_of's (gstart, gchild): the processes along the coarser path sky's
    lines of sight, each carrying its pixels' emission, at their mean density (a shortcut, 2026-10-25)."""
    E = np.ascontiguousarray(E_eV, dtype=np.float64)
    npix, ns = q_gas.shape
    single = np.asarray(kinds[0]["P"]).ndim == 3
    Ps = [np.asarray(k["P"], dtype=np.float64)[None] if single else np.asarray(k["P"], dtype=np.float64)
          for k in kinds]
    C = Ps[0].shape[0]
    n_nodes = max(P.shape[1] for P in Ps)
    Pn = np.zeros((C, 2, n_nodes, len(E), 3))
    jn = np.zeros((2, npix, ns), dtype=np.int64)
    tn = np.zeros((2, npix, ns))
    for kk, k in enumerate(kinds):
        Pn[:, kk, :Ps[kk].shape[1]] = Ps[kk]
        jn[kk] = k["j"]
        tn[kk] = k["t"]
    gstart, gchild = (np.arange(npix + 1, dtype=np.int64), np.arange(npix, dtype=np.int64)) if groups is None \
        else groups
    cseg = _group_cseg(kinds, gstart, gchild)
    seglen, L_sun = _segments(edges_kpc)
    t0, t1 = kinds[0]["tables"], kinds[1]["tables"]
    lad = rates.get("ladder")
    out = np.zeros((C, 2, npix, len(E), 3))
    zeta = np.zeros(len(gstart) - 1)
    _sweep(E, np.ascontiguousarray(I), np.ascontiguousarray(w_gas), np.ascontiguousarray(q_gas),
           np.ascontiguousarray(q_src), Pn, jn, tn, gstart, gchild, cseg, seglen, L_sun,
           t0["J"], t0["Gd"], t0["GG"], t0["lam"], t0["log_s0"], t0["inv_dlogs"],
           t1["J"], t1["Gd"], t1["GG"], t1["lam"], t1["log_s0"], t1["inv_dlogs"],
           float(rates["jump_unit"]), rates["jump_unit"] > 0.0, _lad_arrays(lad), lad is not None,
           float(rates["loss_C"]), rates["loss_C"] != 0.0, int(n_split), out, zeta)
    if diag is not None:
        diag["zeta"] = max(float(diag.get("zeta", 0.0)), float(zeta.max()) if len(zeta) else 0.0)
    return out[0] if single else out


def crossing_populations(E_eV, P, edges_kpc, kinds, rates, groups, n_split=1, diag=None):
    """The extragalactic beam across the Galaxy (2026-10-25): its populations P (C, N_E, 2, 3) at z = 0 (the
    cosmological phase's eigenstates, which the adiabatic transit carries in order into the local ones) along each
    line of sight g of the path sky (groups_of; kinds' n at the emission sky's pixels, averaged per line of sight)
    from the Galaxy's edge to the Sun with the processes: (N_g, C, N_E, 2, 3)."""
    E = np.ascontiguousarray(E_eV, dtype=np.float64)
    gstart, gchild = groups
    cseg = _group_cseg(kinds, gstart, gchild)
    seglen, L_sun = _segments(edges_kpc)
    t0, t1 = kinds[0]["tables"], kinds[1]["tables"]
    lad = rates.get("ladder")
    P = np.ascontiguousarray(P, dtype=np.float64)
    out = np.zeros((len(gstart) - 1,) + P.shape)
    zeta = np.zeros(len(gstart) - 1)
    _cross(E, P, cseg, seglen, L_sun, t0["J"], t0["Gd"], t0["GG"], t0["lam"], t0["log_s0"], t0["inv_dlogs"],
           t1["J"], t1["Gd"], t1["GG"], t1["lam"], t1["log_s0"], t1["inv_dlogs"],
           float(rates["jump_unit"]), rates["jump_unit"] > 0.0, _lad_arrays(lad), lad is not None,
           float(rates["loss_C"]), rates["loss_C"] != 0.0, int(n_split), out, zeta)
    if diag is not None:
        diag["zeta"] = max(float(diag.get("zeta", 0.0)), float(zeta.max()) if len(zeta) else 0.0)
    return out


def _lad_arrays(lad):
    """A ladder's arrays for the kernels, or a dummy of the same types (no decays)."""
    if lad is not None:
        return lad.arrays
    return _DUMMY_LADDER


_NC, _NE, _NR = dp_medium.NCH, dp_medium.NEXP, dp_medium.NRUN
_DUMMY_LADDER = (np.ones(2), np.zeros((2, 2, 3)), np.zeros((2, 2, 3, 9)), np.zeros((2, 3)), np.zeros((2, _NC)),
                 np.zeros((2, _NC)), np.zeros((2, _NC), dtype=np.int64), np.zeros((2, _NC, _NE), dtype=np.int64),
                 np.zeros((2, _NC, _NE)), np.zeros((2, _NC), dtype=np.int64),
                 np.zeros((2, _NC, _NR, 2), dtype=np.int64),
                 np.zeros((2, dp_medium.NCL, dp_medium.NMOM, 9)), np.zeros(2), 1.0, 1.0)


_NC, _NE, _NR = dp_medium.NCH, dp_medium.NEXP, dp_medium.NRUN
_DUMMY_LADDER = (np.ones(2), np.zeros((2, 2, 3)), np.zeros((2, 2, 3, 9)), np.zeros((2, 3)), np.zeros((2, _NC)),
                 np.zeros((2, _NC)), np.zeros((2, _NC), dtype=np.int64), np.zeros((2, _NC, _NE), dtype=np.int64),
                 np.zeros((2, _NC, _NE)), np.zeros((2, _NC), dtype=np.int64),
                 np.zeros((2, _NC, _NR, 2), dtype=np.int64),
                 np.zeros((2, dp_medium.NCL, dp_medium.NMOM, 9)), np.zeros(2), 1.0, 1.0)


@njit(cache=True)
def _basis_paths(E, Y, seglen, L_sun, J0, Gd0, GG0, lam0, ls0, ids0, J1, Gd1, GG1, lam1, ls1, ids1,
                 jump_unit, use_jumps, lad, use_dec, loss_C, use_loss, n_split, X, zeta):
    """No potential in the Galaxy (sigma = 0): the evolution depends only on the path length. X[j] (C, N_E, 2, 3) =
    what reaches the Sun per unit length of the constant sources Y (C, N_E, 2, 3) (the columns: spectral bases and
    compositions) spread over bin j, (G(e_j+1) - G(e_j)) / ds_j with G(e) = int_0^e T(s) Y ds the populations after
    a constant source Y over a path e from zero (homogeneous paths compose), built outward over the edges; linear
    (clip off: the spectral bases change sign). zeta[0] updated."""
    N_E = E.shape[0]
    C = Y.shape[0]
    jmax = 0.0
    for k in range(J0.shape[0]):
        for b in range(3):
            for a in range(3):
                jmax = max(jmax, J0[k, b, a], J1[k, b, a])
    wk = _work(N_E, C)
    G = np.zeros((C, N_E, 2, 3))
    for _ in range(n_split):
        _segment(E, G, L_sun / n_split, 0.0, 0.0, Y, True, J0, Gd0, GG0, lam0, ls0, ids0, J1, Gd1, GG1, lam1, ls1,
                 ids1, jump_unit, use_jumps, jmax, lad, use_dec, loss_C, use_loss, False, wk)
    Pv = G.copy()
    for j in range(seglen.shape[0]):
        L = seglen[j]
        for _ in range(n_split):
            _segment(E, G, L / n_split, 0.0, 0.0, Y, True, J0, Gd0, GG0, lam0, ls0, ids0, J1, Gd1, GG1, lam1, ls1,
                     ids1, jump_unit, use_jumps, jmax, lad, use_dec, loss_C, use_loss, False, wk)
        for col in range(C):
            for i in range(N_E):
                for kind in range(2):
                    for a in range(3):
                        X[j, col, i, kind, a] = (G[col, i, kind, a] - Pv[col, i, kind, a]) / L
                        Pv[col, i, kind, a] = G[col, i, kind, a]
    zeta[0] = max(zeta[0], wk[5][0])


def vacuum_populations(E_eV, I, w_gas, q_gas, q_src, edges_kpc, P_vac, tables, rates, work, nside, tol=1e-10,
                       n_split=1, diag=None):
    """sweep_populations where the Galaxy's potential is negligible (2026-10-24): every line of sight evolves alike
    but for its length, so the emitted spectra, split into a few shapes (the gas and the sources' parts, each a
    truncated SVD over the pixels, singular values above tol of the largest), are evolved once over the distance
    bins -- every shape and composition together, the columns of one sweep -- and combined per pixel with the distance
    weights, at the output resolution (the children's sums, the populations then divided by the summed intensity:
    dp_galactic.degrade_weighted's average). P_vac (2, N_E, 3), or (C, 2, N_E, 3) for C compositions, the vacuum
    populations at emission per kind. Returns (2, N_pix_out, N_E, 3), or (C, 2, N_pix_out, N_E, 3)."""
    import dp_galactic
    E = np.ascontiguousarray(E_eV, dtype=np.float64)
    seglen, L_sun = _segments(edges_kpc)
    ns = len(seglen)
    t0, t1 = tables
    lad = rates.get("ladder")
    zeta = np.zeros(1)
    P_vac = np.asarray(P_vac, dtype=np.float64)
    single = P_vac.ndim == 3
    Pc = P_vac[None] if single else P_vac
    nc = Pc.shape[0]

    def down(x):
        return x if work == nside else dp_galactic.degrade(x, work, nside)

    I_out = down(np.asarray(I, dtype=np.float64))
    # the SVD of the emission scaled by the sky-mean intensity per energy, so the truncation is relative at every
    # energy (unscaled, the steep spectrum let it drop 3e-4 of the faint high energies)
    Ibar = np.asarray(I, dtype=np.float64).mean(axis=0)
    Ibar = np.where(Ibar > 0.0, Ibar, 1.0)
    cols, weights = [], []
    for A, q in ((np.asarray(I) * np.asarray(w_gas), q_gas), (np.asarray(I) * (1.0 - np.asarray(w_gas)), q_src)):
        U_, S_, Vt = np.linalg.svd(A / Ibar[None, :], full_matrices=False)
        keep = S_ >= tol * S_[0] if S_[0] > 0.0 else np.zeros_like(S_, dtype=bool)
        for kb in np.nonzero(keep)[0]:
            u = Vt[kb] * Ibar
            weights.append(down((U_[:, kb] * S_[kb])[:, None] * np.asarray(q)))          # (N_pix_out, N_s)
            for c in range(nc):
                cols.append(np.stack([u[:, None] * Pc[c, 0], u[:, None] * Pc[c, 1]], axis=1))   # (N_E, 2, 3)
    Y = np.ascontiguousarray(np.stack(cols))
    X = np.zeros((ns, len(cols), len(E), 2, 3))
    _basis_paths(E, Y, seglen, L_sun, t0["J"], t0["Gd"], t0["GG"], t0["lam"], t0["log_s0"], t0["inv_dlogs"],
                 t1["J"], t1["Gd"], t1["GG"], t1["lam"], t1["log_s0"], t1["inv_dlogs"],
                 float(rates["jump_unit"]), rates["jump_unit"] > 0.0, _lad_arrays(lad), lad is not None,
                 float(rates["loss_C"]), rates["loss_C"] != 0.0, int(n_split), X, zeta)
    outs = []
    for c in range(nc):
        total = None
        for ib, W in enumerate(weights):
            Xc = X[:, ib * nc + c].transpose(0, 2, 1, 3)                                  # (N_s, 2, N_E, 3)
            t = W @ Xc.reshape(ns, -1)
            total = t if total is None else total + t
        o = total.reshape(-1, 2, len(E), 3).transpose(1, 0, 2, 3)
        with np.errstate(invalid="ignore", divide="ignore"):
            outs.append(np.where(I_out[None, :, :, None] > 0.0, o / I_out[None, :, :, None], 0.0))
    if diag is not None:
        diag["zeta"] = max(float(diag.get("zeta", 0.0)), float(zeta[0]))
    return outs[0] if single else np.stack(outs)
