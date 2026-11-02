"""
dp_sweep.py -- the dark-portal phase's step where the neutrinos decay (2026-10-25; the convergence audit,
docs/dev/CONVERGENCE_AUDIT_2026-10-25.md, finding 2; Mauricio: port dp_si's method), with the decays between the
propagation eigenstates of the medium (dp_medium; docs/dev/DECAY_IN_MEDIUM_2026-10-25.md); and, the same step at z = 0
with the emission as a source over the segment, the Galactic lines of sight's segments (dp_galactic_path, 2026-10-25:
the review's finding 4, which replaced their jump-decay alternation).

Every process of a bin is in one linear system solved exactly over the step, the bins swept from the highest energy
down as in dp_si's column sweep: per bin, the six populations per unit energy (both kinds, since a flipped daughter
changes kind; the eigenstates of the potential) evolve as dx/dt = M x + S,

    M  = the soft scatterings within each kind (dp_cosmo._soft_jumps' rates, Zeno-damped by the eigenstates' decay
         widths) - the absorption (step_cosmological_rates' G) - the decays (each eigenstate's in-medium width, the
         sum of its channels, dp_medium)
         + the bin's own secondaries: the absorptions' (brp x self_same, self_cross) and the decays' (the channels'
           own-bin shares, into the daughter eigenstates directly)
    S  = the injection (or, use_src, a given source per unit energy and time: the Galactic emission) + the
         absorptions' secondaries from the bins above + the decays' daughters from the bins above, constant over the
         step (as dp_si and step_cosmological_rates),

x(dt) = e^(M dt) x0 + dt phi1(M dt) S; its time average phi1(M dt) x0 + dt phi2(M dt) S gives the numbers absorbed and
decayed that the bins below receive (the absorptions' running sums as step_cosmological_rates, the decays' through
dp_medium's plans: the cut bins directly, the runs of bins wholly inside a window by running moments). The times: dt
the step's (step_cosmological_rates' end-point dt, for the injection and the absorption as before); the scatterings'
rates scaled to the step's exact int dz / ((1+z) H) (jump_I), the decays' to its int dz / ((1+z)^2 H) (decay_I). Stiff
bins as dp_si's: a decay beyond dp_si.DECAY_STEP_CAP per step and DECAY_STEP_RATIO x the state's other events capped
there (it drains the state within the step either way; its channels' shares kept); conversions beyond
dp_si.BIN_ROUNDING_MAX per step scaled down as dp_si._bin_matrix (their ratios, hence the bin's equilibrium, kept).
The relic mass-change loss stays in two halves around the step.
"""
import numpy as np
from numba import njit

import dp_medium
from dp_si import (_phi_eig, _keep_rate_blocks, STIFF_BIN, DECAY_STEP_CAP, DECAY_STEP_RATIO, BIN_ROUNDING_MAX,
                   CONVERSION_CAP, CONVERSION_RATIO, CONVERSION_MIN, CONVERSION_GUARD)

# the Taylor terms of e^X, phi1(X), phi2(X) at |X| <= 1/2 at most (remainder 0.5^15 / 15! = 2e-17; fewer where |X|
# is smaller: until |X|^k / k! < 1e-18)
_TAYLOR_K = 14


@njit(cache=True)
def _jump_rates(i, E_bins, zp1, J_table, Gd_table, GG_table, lam_table, log_s0, inv_dlogs, c_z, Gamma_unit, wd, A):
    """dp_cosmo._soft_jumps' rates at bin i (per unit time, eV), with the bin's decay widths wd (3,): A[b, a] (a -> b)
    off the diagonal, symmetric; the diagonal minus the column's other entries."""
    N_sig = J_table.shape[0]
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
    two_E = 2.0 * E_bins[i] * zp1
    for a in range(3):
        for b in range(3):
            A[a, b] = 0.0
    for a in range(3):
        for b in range(a + 1, 3):
            Jab = 0.5 * ((1.0 - w) * (J_table[k0, b, a] + J_table[k0, a, b])
                         + w * (J_table[k0 + 1, b, a] + J_table[k0 + 1, a, b]))
            om = ((1.0 - w) * (lam_table[k0, a] - lam_table[k0, b])
                  + w * (lam_table[k0 + 1, a] - lam_table[k0 + 1, b])) / two_E
            ga = (1.0 - w) * Gd_table[k0, a] + w * Gd_table[k0 + 1, a]
            gb = (1.0 - w) * Gd_table[k0, b] + w * Gd_table[k0 + 1, b]
            gga = (1.0 - w) * GG_table[k0, a] + w * GG_table[k0 + 1, a]
            ggb = (1.0 - w) * GG_table[k0, b] + w * GG_table[k0 + 1, b]
            D = Gamma_unit * max(0.5 * (gga + ggb) - ga * gb, 0.0)
            D += 0.5 * (wd[a] + wd[b])
            r = Gamma_unit * Jab
            if D > 0.0:
                r *= om * om / (om * om + D * D)
            A[a, b] = r
            A[b, a] = r
    for a in range(3):
        A[a, a] = -(A[0, a] + A[1, a] + A[2, a])


@njit(cache=True)
def _mm(A, B, out):
    """out = A B (small square matrices; out distinct from A and B)."""
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            v = 0.0
            for k in range(n):
                v += A[i, k] * B[k, j]
            out[i, j] = v


@njit(cache=True)
def _phi_rate(Mx, Em, P1, P2, W1, W2, W3):
    """e^M, phi1(M), phi2(M) of a rate matrix M (off the diagonal >= 0, columns summing to <= 0), in place: scaling
    to |M 2^-s| <= 1/2, the three Taylor series there, then s doublings of t (A = e^(M t), B = t phi1(M t),
    C = t^2 phi2(M t): A' = A A, B' = A B + B, C' = A C + t B + C) with the blocks kept as a rate matrix's must be
    (dp_si._keep_rate_blocks) -- dp_si._phi_functions' result (its 3n x 3n augmented exponential, the same squarings)
    at 6 x 6 cost; beyond dp_si.STIFF_BIN dp_si._phi_eig. W1-W3: work matrices."""
    n = Mx.shape[0]
    nrm = 0.0
    big = 0.0
    for r in range(n):
        rs = 0.0
        for c in range(n):
            rs += abs(Mx[r, c])
            big = max(big, abs(Mx[r, c]))
        nrm = max(nrm, rs)
    if big > STIFF_BIN:
        A, B, C = _phi_eig(Mx)
        for r in range(n):
            for c in range(n):
                Em[r, c] = A[r, c]
                P1[r, c] = B[r, c]
                P2[r, c] = C[r, c]
        return
    sq = 0
    if nrm > 0.5:
        sq = int(np.ceil(np.log2(nrm / 0.5)))
    t = 1.0 / (2.0 ** sq)
    # Taylor: W1 = X^k
    for r in range(n):
        for c in range(n):
            W1[r, c] = 1.0 if r == c else 0.0
            Em[r, c] = W1[r, c]
            P1[r, c] = W1[r, c]
            P2[r, c] = 0.5 * W1[r, c]
    f0 = 1.0
    xs = nrm * t                              # the scaled norm (<= 1/2): the series stops once its terms are below 1e-18
    bound = 1.0
    for k in range(1, _TAYLOR_K + 1):
        bound *= xs / k
        if bound < 1e-18:
            break
        _mm(W1, Mx, W2)
        for r in range(n):
            for c in range(n):
                W1[r, c] = W2[r, c] * t
        f0 /= k
        f1 = f0 / (k + 1)
        f2 = f1 / (k + 2)
        for r in range(n):
            for c in range(n):
                v = W1[r, c]
                Em[r, c] += f0 * v
                P1[r, c] += f1 * v
                P2[r, c] += f2 * v
    for r in range(n):
        for c in range(n):
            P1[r, c] *= t
            P2[r, c] *= t * t
    for _ in range(sq):
        _mm(Em, P2, W1)                       # C' = A C + t B + C
        for r in range(n):
            for c in range(n):
                W1[r, c] += t * P1[r, c] + P2[r, c]
        _mm(Em, P1, W2)                       # B' = A B + B
        for r in range(n):
            for c in range(n):
                W2[r, c] += P1[r, c]
        _mm(Em, Em, W3)                       # A' = A A
        for r in range(n):
            for c in range(n):
                Em[r, c] = W3[r, c]
                P1[r, c] = W2[r, c]
                P2[r, c] = W1[r, c]
        t *= 2.0
        _keep_rate_blocks(Em, P1, P2)


@njit(cache=True)
def _phi_diagonal(Mx, Em, P1, P2):
    """e^M, phi1(M), phi2(M) of a diagonal M, in place: e^m, (e^m - 1) / m, (e^m - 1 - m) / m^2 (series near 0)."""
    n = Mx.shape[0]
    for r in range(n):
        for c in range(n):
            Em[r, c] = 0.0
            P1[r, c] = 0.0
            P2[r, c] = 0.0
        m = Mx[r, r]
        Em[r, r] = np.exp(m)
        if abs(m) < 1e-3:
            P1[r, r] = 1.0 + m * (0.5 + m * (1.0 / 6.0 + m * (1.0 / 24.0 + m / 120.0)))
            P2[r, r] = 0.5 + m * (1.0 / 6.0 + m * (1.0 / 24.0 + m * (1.0 / 120.0 + m / 720.0)))
        else:
            e1 = np.expm1(m)
            P1[r, r] = e1 / m
            P2[r, r] = (e1 - m) / (m * m)


@njit(cache=True)
def step_full(inj_rate, dt, zp1_mid, jump_dt, decay_dt, E_bins, F_nu, F_nb, w_nu, w_nb, rho_src, Pi_nu, Pi_nb,
              G_nu, G_nb, brp_nu, brp_nb, p_index, Phi_0, W3, pref3, self_same, W1, W2, c1, c2, self_cross,
              R_nu, R_nb, inj_w, use_jumps, Gamma_unit,
              J_n, Gd_n, GG_n, lam_n, ls_n, ids_n, cz_n, J_b, Gd_b, GG_b, lam_b, ls_b, ids_b, cz_b,
              lad, x0, use_dec, clip, diag, pend, acc, add, sub, act, src, use_src, cache, have, use_cache):
    """One step (module docstring), F_nu, F_nb (N, 3) in place. inj_rate: the sources' emission per unit time over the
    step, R(z) (1+z)^(1-p) averaged (the injection Phi_0 (E (1+z))^-p R(z) (1+z) of step_cosmological_rates without
    E^-p); dt and the absorption and cascade arguments as step_cosmological_rates' (G, brp this step's rates; zeros for
    the Galactic segments); zp1_mid 1 + the step's midpoint z; jump_dt = jump_I, decay_dt = decay_I (the step's exact
    times for the scatterings and the decays); with use_jumps the scatterings' tables per kind (_n, _b: J, Gd, GG, lam
    tables, log_s0, inv_dlogs, the midpoint sigma coefficient) and Gamma_unit (the midpoint rate per unit |G|^2, eV);
    with use_dec the decays: lad a dp_medium.Ladder's arrays, x0 its coordinate of this step's coefficient
    (Ladder.coord), pend, acc, add, sub, act its work arrays (Ladder.buffers; left zero). clip: negative populations
    from rounding set to zero (off for the Galactic paths' sign-indefinite spectral bases: the step is then linear).
    diag[0]: the largest decay width over its kind's smallest splitting met (dp_medium's validity measure, updated).
    src (N, 2, 3), use_src: a further source per unit energy and time, constant over the step (the Galactic emission).
    A bin whose matrix is diagonal (no conversions, no own-bin daughters) takes the scalar exponentials.
    cache (N, 4, 6, 6), have (N,), use_cache: the runs of one parameter point that differ only in their sources (the
    flavors and kinds of the analysis' basis) share every bin's matrix; with use_cache a bin whose matrix equals the
    stored one exactly reuses its exponentials (e^M, phi1, phi2: functions of the matrix alone, so the result is the
    same to the last bit), and stores them otherwise (2026-10-25: the exponentials were 90% of a decay point's cost)."""
    N = E_bins.shape[0]
    h = np.sqrt(E_bins[1] / E_bins[0])
    jsc = jump_dt / dt                            # the scatterings' and the decays' rates per unit dt
    dsc = decay_dt / dt
    Ms = lad[13]
    zeno = lad[12]
    # the decays' daughters below each bin: the cut bins' numbers (pend) and the runs' running moments (acc, add, sub)
    if use_dec:
        for t in range(2):
            for cl in range(dp_medium.NCL):
                act[t, cl] = False
                for k in range(dp_medium.NMOM):
                    for q in range(9):
                        acc[t, cl, k, q] = 0.0
    gin = np.zeros((2, 3))
    Lam = np.zeros((2, 3))
    own = np.zeros((2, 3, 2, 3))
    # the absorptions' running sums over the bins above (step_cosmological_rates)
    s3_nu = 0.0
    s3_nb = 0.0
    s1_nu = 0.0
    s1_nb = 0.0
    s2_nu = 0.0
    s2_nb = 0.0
    Mx = np.zeros((6, 6))
    S = np.zeros(6)
    x0v = np.zeros(6)
    xbar = np.zeros(6)
    Ld = np.zeros(6)
    fcap = np.ones(6)
    Gs = np.zeros(6)
    A = np.zeros((2, 3, 3))
    Ak = np.zeros((3, 3))
    wd = np.zeros(3)
    Em = np.zeros((6, 6))
    P1 = np.zeros((6, 6))
    P2 = np.zeros((6, 6))
    Wk1 = np.zeros((6, 6))
    Wk2 = np.zeros((6, 6))
    Wk3 = np.zeros((6, 6))
    for b in range(N - 1, -1, -1):
        Eb = E_bins[b]
        dEb = Eb * h - Eb / h
        # the sources: the injection, the absorptions' secondaries from above, the decays' daughters from above
        inj_spectrum = Phi_0 * Eb ** (-p_index) * inj_rate * inj_w[b] if Phi_0 != 0.0 else 0.0
        reg_nu = pref3[b] * s3_nu + c1[b] * s1_nb - c2[b] * s2_nb + pref3[b] * s3_nb
        reg_nb = pref3[b] * s3_nb + c1[b] * s1_nu - c2[b] * s2_nu + pref3[b] * s3_nu
        for a in range(3):
            S[a] = ((Pi_nu[b, 0, a] * rho_src[0] + Pi_nu[b, 1, a] * rho_src[1] + Pi_nu[b, 2, a] * rho_src[2])
                    * inj_spectrum * w_nu + brp_nu[b, a] * reg_nu)
            S[3 + a] = ((Pi_nb[b, 0, a] * rho_src[0] + Pi_nb[b, 1, a] * rho_src[1] + Pi_nb[b, 2, a] * rho_src[2])
                        * inj_spectrum * w_nb + brp_nb[b, a] * reg_nb)
            Gs[a] = max(G_nu[b, a], 0.0)
            Gs[3 + a] = max(G_nb[b, a], 0.0)
            if use_src:
                S[a] += src[b, 0, a]
                S[3 + a] += src[b, 1, a]
        if use_dec:
            dp_medium.gains(lad, x0, b, Eb, h, acc, add, sub, pend, gin, act)
            for t in range(2):
                for a in range(3):
                    S[3 * t + a] += gin[t, a] / dEb
            # the bin's decay rates per unit dt (pref: the plans' unit 1 / (16 pi M^2 E), the step's decay time)
            pref = dsc / (16.0 * np.pi * Ms * Eb)
            dp_medium.bin_rates(lad, x0, b, pref, Lam, own)
            r0, wr = dp_medium.rung_of_bin(x0, b)
            zb = (1.0 - wr) * zeno[r0] + (wr * zeno[r0 + 1] if wr > 0.0 else 0.0)
            if zb > diag[0]:
                diag[0] = zb
        # the scatterings (per dt), Zeno-damped by the physical decay widths (per unit time)
        for t in range(2):
            if use_jumps:
                for a in range(3):
                    # the physical width (per unit time at the midpoint): the plans' rate per dt / dsc is E_phys
                    # Gamma / E_obs = Gamma (1+z)
                    wd[a] = Lam[t, a] / (dsc * zp1_mid) if use_dec and dsc > 0.0 else 0.0
                if t == 0:
                    _jump_rates(b, E_bins, zp1_mid, J_n, Gd_n, GG_n, lam_n, ls_n, ids_n, cz_n, Gamma_unit, wd, Ak)
                else:
                    _jump_rates(b, E_bins, zp1_mid, J_b, Gd_b, GG_b, lam_b, ls_b, ids_b, cz_b, Gamma_unit, wd, Ak)
                for r in range(3):
                    for c in range(3):
                        A[t, r, c] = Ak[r, c] * jsc
        # stiff bins (per step): a decay beyond DECAY_STEP_CAP and DECAY_STEP_RATIO x the state's other events capped
        # (its channels' shares kept); the conversions beyond BIN_ROUNDING_MAX scaled as dp_si._bin_matrix
        cmax = 0.0
        cmin = np.inf
        lmax = 0.0
        for c in range(6):
            t = c // 3
            a = c - 3 * t
            out = -A[t, a, a] * dt
            other = out + Gs[c] * dt
            Ld[c] = Lam[t, a] * dt if use_dec else 0.0
            fcap[c] = 1.0
            if Ld[c] > DECAY_STEP_CAP and Ld[c] > DECAY_STEP_RATIO * other:
                fcap[c] = max(DECAY_STEP_CAP, DECAY_STEP_RATIO * other) / Ld[c]
                Ld[c] *= fcap[c]
            cmax = max(cmax, out)
            if out > 0.0:
                cmin = min(cmin, out)
            lmax = max(lmax, Gs[c] * dt + Ld[c])
        sc = 1.0
        if cmax > BIN_ROUNDING_MAX:
            cap = max(CONVERSION_CAP, CONVERSION_RATIO * lmax)
            if cmax > cap:
                sc = max(cap / cmax, CONVERSION_MIN / cmin)
                sc = min(sc, max(cap, CONVERSION_GUARD) / cmax)
                if sc > 1.0:
                    sc = 1.0
        # the matrix (per step)
        for r in range(6):
            for c in range(6):
                Mx[r, c] = 0.0
        for t in range(2):
            o = 3 * t
            for c in range(3):
                col = 0.0
                for r in range(3):
                    if r != c:
                        v = A[t, r, c] * sc * dt
                        Mx[o + r, o + c] = v
                        col += v
                Mx[o + c, o + c] = -col - Gs[o + c] * dt - Ld[o + c]
        # the bin's own secondaries: the absorptions' (same kind: self_same; the other kind: self_cross)
        for t in range(2):
            o = 3 * t
            po = 3 - o
            brp = brp_nu if t == 0 else brp_nb
            brp_o = brp_nb if t == 0 else brp_nu
            for c in range(3):
                g = Gs[o + c] * dt
                if g == 0.0:
                    continue
                for r in range(3):
                    Mx[o + r, o + c] += brp[b, r] * self_same[b] * g
                    Mx[po + r, o + c] += brp_o[b, r] * self_cross[b] * g
        # ... and the decays' (each channel's own-bin share, into the daughter eigenstate)
        if use_dec:
            for t in range(2):
                for c in range(3):
                    for td in range(2):
                        for r in range(3):
                            v = own[t, c, td, r]
                            if v != 0.0:
                                Mx[3 * td + r, 3 * t + c] += v * dt * fcap[3 * t + c]
        diagonal = True
        for r in range(6):
            for c in range(6):
                if r != c and Mx[r, c] != 0.0:
                    diagonal = False
        same = False
        if use_cache and have[b]:
            same = True
            for r in range(6):
                for c in range(6):
                    if Mx[r, c] != cache[b, 0, r, c]:
                        same = False
        if same:
            for r in range(6):
                for c in range(6):
                    Em[r, c] = cache[b, 1, r, c]
                    P1[r, c] = cache[b, 2, r, c]
                    P2[r, c] = cache[b, 3, r, c]
        else:
            if diagonal:
                _phi_diagonal(Mx, Em, P1, P2)
            else:
                _phi_rate(Mx, Em, P1, P2, Wk1, Wk2, Wk3)
            if use_cache:
                for r in range(6):
                    for c in range(6):
                        cache[b, 0, r, c] = Mx[r, c]
                        cache[b, 1, r, c] = Em[r, c]
                        cache[b, 2, r, c] = P1[r, c]
                        cache[b, 3, r, c] = P2[r, c]
                have[b] = True
        for a in range(3):
            x0v[a] = F_nu[b, a]
            x0v[3 + a] = F_nb[b, a]
        for r in range(6):
            fn = 0.0
            av = 0.0
            for c in range(6):
                fn += Em[r, c] * x0v[c] + P1[r, c] * S[c] * dt
                av += P1[r, c] * x0v[c] + P2[r, c] * S[c] * dt
            if clip:
                xbar[r] = av if av > 0.0 else 0.0
                fn = fn if fn > 0.0 else 0.0
            else:
                xbar[r] = av
            if r < 3:
                F_nu[b, r] = fn
            else:
                F_nb[b, r - 3] = fn
        # the numbers absorbed per unit energy and time (the cascade's running sums, as step_cosmological_rates)
        Rn = 0.0
        Rb = 0.0
        for a in range(3):
            Rn += Gs[a] * xbar[a]
            Rb += Gs[3 + a] * xbar[3 + a]
        R_nu[b] = Rn
        R_nb[b] = Rb
        s3_nu += Rn * W3[b]
        s3_nb += Rb * W3[b]
        s1_nu += Rn * W1[b]
        s1_nb += Rb * W1[b]
        s2_nu += Rn * W2[b]
        s2_nb += Rb * W2[b]
        # the decays' daughters below the bin (numbers per unit time)
        if use_dec:
            pref = dsc / (16.0 * np.pi * Ms * Eb)
            for t in range(2):
                for c in range(3):
                    xb = xbar[3 * t + c]
                    if xb == 0.0 or Lam[t, c] == 0.0:
                        continue
                    dp_medium.deposit(lad, x0, b, t, c, xb * dEb * pref * fcap[3 * t + c], Eb, h, pend, add, sub,
                                      act)


@njit(cache=True)
def step_columns(L, E_bins, F, use_jumps, Gamma_unit, J_n, Gd_n, GG_n, lam_n, ls_n, ids_n, cz_n,
                 J_b, Gd_b, GG_b, lam_b, ls_b, ids_b, cz_b, lad, x0, use_dec, clip, diag, pend, acc, add, sub, act,
                 src, use_src):
    """The Galactic segments' step (2026-10-25): step_full at z = 0 over a path L (eV^-1, the time), without
    absorption or injection, for C columns of populations F (C, N, 2, 3) per unit energy (both kinds), in place --
    the columns (the analysis' sources, the Galactic emission's compositions, the spectral bases) share every bin's
    matrix, so its exponentials are computed once per bin for all of them. Per bin: the scatterings (Zeno-damped by
    the states' decay widths), the decays and their own-bin daughters, and with use_src the sources src (C, N, 2, 3)
    per unit energy and length, solved together exactly over L; the daughters below by dp_medium's plans, per column
    (pend (C, N, 2, 3), acc (C, 2, NCL, NMOM, 9), add, sub (C, N, S, NCL, NMOM, 9), act (C, 2, NCL); S = 1 suffices on
    the nearest rung, x0 an integer or -1; left zero). The stiff caps as step_full's. diag[0]: the validity measure."""
    N = E_bins.shape[0]
    C = F.shape[0]
    h = np.sqrt(E_bins[1] / E_bins[0])
    Ms = lad[13]
    zeno = lad[12]
    if use_dec:
        for col in range(C):
            for t in range(2):
                for cl in range(dp_medium.NCL):
                    act[col, t, cl] = False
                    for k in range(dp_medium.NMOM):
                        for q in range(9):
                            acc[col, t, cl, k, q] = 0.0
    gin = np.zeros((2, 3))
    Lam = np.zeros((2, 3))
    own = np.zeros((2, 3, 2, 3))
    Mx = np.zeros((6, 6))
    S = np.zeros((C, 6))
    xbar = np.zeros((C, 6))
    Ld = np.zeros(6)
    fcap = np.ones(6)
    A = np.zeros((2, 3, 3))
    Ak = np.zeros((3, 3))
    wd = np.zeros(3)
    Em = np.zeros((6, 6))
    P1 = np.zeros((6, 6))
    P2 = np.zeros((6, 6))
    Wk1 = np.zeros((6, 6))
    Wk2 = np.zeros((6, 6))
    Wk3 = np.zeros((6, 6))
    x0v = np.zeros(6)
    for b in range(N - 1, -1, -1):
        Eb = E_bins[b]
        dEb = Eb * h - Eb / h
        for col in range(C):
            for c in range(6):
                S[col, c] = src[col, b, c // 3, c % 3] if use_src else 0.0
        pref = 1.0 / (16.0 * np.pi * Ms * Eb)
        if use_dec:
            for col in range(C):
                dp_medium.gains(lad, x0, b, Eb, h, acc[col], add[col], sub[col], pend[col], gin, act[col])
                for t in range(2):
                    for a in range(3):
                        S[col, 3 * t + a] += gin[t, a] / dEb
            dp_medium.bin_rates(lad, x0, b, pref, Lam, own)
            r0, wr = dp_medium.rung_of_bin(x0, b)
            zb = (1.0 - wr) * zeno[r0] + (wr * zeno[r0 + 1] if wr > 0.0 else 0.0)
            if zb > diag[0]:
                diag[0] = zb
        for t in range(2):
            for r in range(3):
                for c in range(3):
                    A[t, r, c] = 0.0
            if use_jumps:
                for a in range(3):
                    wd[a] = Lam[t, a] if use_dec else 0.0
                if t == 0:
                    _jump_rates(b, E_bins, 1.0, J_n, Gd_n, GG_n, lam_n, ls_n, ids_n, cz_n, Gamma_unit, wd, Ak)
                else:
                    _jump_rates(b, E_bins, 1.0, J_b, Gd_b, GG_b, lam_b, ls_b, ids_b, cz_b, Gamma_unit, wd, Ak)
                for r in range(3):
                    for c in range(3):
                        A[t, r, c] = Ak[r, c]
        cmax = 0.0
        cmin = np.inf
        lmax = 0.0
        for c in range(6):
            t = c // 3
            a = c - 3 * t
            out = -A[t, a, a] * L
            Ld[c] = Lam[t, a] * L if use_dec else 0.0
            fcap[c] = 1.0
            if Ld[c] > DECAY_STEP_CAP and Ld[c] > DECAY_STEP_RATIO * out:
                fcap[c] = max(DECAY_STEP_CAP, DECAY_STEP_RATIO * out) / Ld[c]
                Ld[c] *= fcap[c]
            cmax = max(cmax, out)
            if out > 0.0:
                cmin = min(cmin, out)
            lmax = max(lmax, Ld[c])
        sc = 1.0
        if cmax > BIN_ROUNDING_MAX:
            cap = max(CONVERSION_CAP, CONVERSION_RATIO * lmax)
            if cmax > cap:
                sc = max(cap / cmax, CONVERSION_MIN / cmin)
                sc = min(sc, max(cap, CONVERSION_GUARD) / cmax)
                if sc > 1.0:
                    sc = 1.0
        for r in range(6):
            for c in range(6):
                Mx[r, c] = 0.0
        diagonal = True
        for t in range(2):
            o = 3 * t
            for c in range(3):
                colsum = 0.0
                for r in range(3):
                    if r != c:
                        v = A[t, r, c] * sc * L
                        Mx[o + r, o + c] = v
                        colsum += v
                        if v != 0.0:
                            diagonal = False
                Mx[o + c, o + c] = -colsum - Ld[o + c]
        if use_dec:
            for t in range(2):
                for c in range(3):
                    for td in range(2):
                        for r in range(3):
                            v = own[t, c, td, r]
                            if v != 0.0:
                                Mx[3 * td + r, 3 * t + c] += v * L * fcap[3 * t + c]
                                diagonal = False
        if diagonal:
            _phi_diagonal(Mx, Em, P1, P2)
        else:
            _phi_rate(Mx, Em, P1, P2, Wk1, Wk2, Wk3)
        for col in range(C):
            for c in range(6):
                x0v[c] = F[col, b, c // 3, c % 3]
            for r in range(6):
                fn = 0.0
                av = 0.0
                for c in range(6):
                    fn += Em[r, c] * x0v[c] + P1[r, c] * S[col, c] * L
                    av += P1[r, c] * x0v[c] + P2[r, c] * S[col, c] * L
                if clip:
                    xbar[col, r] = av if av > 0.0 else 0.0
                    fn = fn if fn > 0.0 else 0.0
                else:
                    xbar[col, r] = av
                F[col, b, r // 3, r % 3] = fn
        if use_dec:
            for col in range(C):
                for t in range(2):
                    for c in range(3):
                        xb = xbar[col, 3 * t + c]
                        if xb == 0.0 or Lam[t, c] == 0.0:
                            continue
                        dp_medium.deposit(lad, x0, b, t, c, xb * dEb * pref * fcap[3 * t + c], Eb, h, pend[col],
                                          add[col], sub[col], act[col])
