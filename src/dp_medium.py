"""
dp_medium.py -- the decay into the mediator between the propagation eigenstates of a medium (2026-10-25; notes,
'Decay in a medium', and docs/dev/DECAY_IN_MEDIUM_2026-10-25.md).

Physics. In a potential V the neutrinos propagate as the eigenstates a of H = U M^2 U^dag / 2E + V (antineutrinos:
U -> U*, V -> -V*), with effective masses squared mu_a(sigma), the eigenvalues of M^2 + sigma S in the mass basis
(sigma = 2 E V / S; dp_cosmo's sigma tables). A state in flight is the superposition sum_i c_ia |nu_i> of the mass
states' light-front spinors (to O(V/E)), so the decay a(p) -> b(z p) + Z_D has the amplitude
    A = sum_ij conj(c'_jb) c_ia V_ji(z, k_T)
with V_ji the vacuum light-front helicity vertex of the Majorana coupling gamma^mu (G'_ji P_L - G'*_ji P_R) (the
Lagrangian masses m_i in the spinors and in the longitudinal mode's q.J), c the parent's eigenvector at its momentum
and c' the daughter's at its own, and the in-medium light-front energy conservation
    k_T^2 = (1 - z) (z mu_a(sigma) - mu'_b(z sigma)) - z M^2 >= 0.
The mediator's polarizations are summed as -J.J* + |q.J|^2 / M^2 with the in-medium q = p_a - p_b: written in
k_T the result below is exact for any texture S (the vacuum energy mismatch of each mass pair in q.J cancels against
the mass terms identically; checked against explicit light-front spinors to 1e-9, tests/test_medium.py, a coupling
that does not commute with S included): the potential enters only through the eigenvectors and the kinematics --
never as a mass in the Goldstone coupling. Summed over the polarizations (in vacuum dp_decay's traces):
  kept (same helicity, same kind):
    |A|^2 = 2 k_T^2 (z + 1/z) / (1-z)^2 |alpha|^2 + |(z M^2 - k_T^2) / (1-z) alpha + (1-z) beta|^2 / (M^2 z),
    alpha = c'^dag G_s c,  beta = c'^dag D G_s* D c,
  flipped (Majorana, the other kind):
    |A|^2 = (2/z) |alpha + z beta|^2 + k_T^2 |alpha + beta|^2 / (M^2 z),
    alpha = c'^dag D G_s c,  beta = c'^dag G_s* D c,
with G_s = G' (parent a neutrino) or G'* (an antineutrino), D = diag(m_i); and E dGamma/dz = |A|^2 / 16 pi. With
Q = z mu_a - mu'_b every weight is (A z + B + C / z) / M^2, A a polynomial in mu_a, C in mu'_b, B in both (COEF below).
In vacuum (c = 1) this is dp_decay.rest_rates and daughter_spectra exactly. A kept decay a -> a is closed: mu'_b(z
sigma) / z >= mu_b(sigma) (D^2 / z >= D^2, Weyl), so a kept decay needs mu_a > mu_b + M^2, a lower state; a flipped
one mu_a > mu-bar_b + M^2. A potential opens flips nu_i -> nu-bar_i of the same mass state (the axial coupling's
2 m_i G'_ii / M), closed in vacuum.

Numerics. E Gamma depends on (E, z) only through sigma, so every parent's decay plan is built once per run on a
'ladder' of sigma values with the energy grid's bin ratio, sigma_r = sig0 h^(2 r): a parent at rung r has its
daughters k bins below at rung r - k (the same coefficient sigma / E). A plan (per rung and channel) holds the total
rate, the share into the parent's own bin, the bins the window cuts (exact integrals over the parent spread uniformly
over its bin, as dp_decay.window_plan: closed forms in y = 1 - z near z = 1, in z below, 8-point Gauss-Legendre over the
parent's spread) and the runs of bins wholly inside the window. Inside a daughter bin the daughter's eigenvector is held
at the bin's centre (projector P9) and its eigenvalue follows the potential's slope through the bin, mu' = alpha -
sigma delta with alpha, delta the bin's (second order in the bin width; exact in vacuum and where nothing varies); the
runs are then exact sums of NPS prefix-summed weights, deposited by the parent's NMOM running moments and collected by
each daughter bin's psi vectors. Number is kept by construction: each decay's daughters are counted once, on or below
the grid. The ladder runs down to where the eigenstates are the vacuum ones (SIGMA_VAC: 1e-6 of the smallest splitting
over the largest |s|; rungs below are the vacuum, sigma = 0 exactly), the decays below the grid counted in the total
(they leave the grid). Each step interpolates linearly in log sigma between the two rungs around its coefficient (as
every sigma table here): a parent's decays go (1 - w) by the lower rung's plan and w by the upper's, each conserving the
number. Assumes a small width over splitting (zeta = 2 E Gamma / min |mu_a - mu_b|, Ladder.zeno; Ladder.diag[0] the
largest a run met): the eigenstates of H, not of H - i Gamma / 2.
"""
import numpy as np
from numba import njit, prange

from dp_transit import frame_at, texture_scales

# packed Hermitian 3x3: [Re00, Re11, Re22, Re01, Im01, Re02, Im02, Re12, Im12]; tr(A B) = sum TRW * a * b
TRW = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
# the ladder reaches the vacuum where sigma |s| < SIGMA_VAC x the smallest vacuum splitting
SIGMA_VAC = 1e-6
# a plan's explicit bins and runs per channel to start with: a ladder whose plans need more is rebuilt with twice the
# room (Ladder; 2026-10-25, the review's finding 1: near a threshold every bin of a wide window can be a cut bin)
NEXP = 24
NRUN = 16
# channels: ch = ((s * 3 + a) * 2 + rel) * 3 + b; classes (s, rel, b): cl = (s * 2 + rel) * 3 + b
NCH = 36
NCL = 12


def coupling_mass_basis(G_flavor, U):
    """G' = U^dag G U with its identity part added back exactly (dp_decay.rest_rates subtracts it: there only the
    off-diagonal entries matter; here the diagonal ones do -- the flips nu_i -> nu-bar_i)."""
    G = np.asarray(G_flavor, dtype=np.complex128)
    U = np.asarray(U, dtype=np.complex128)
    c = G[0, 0].real
    return U.conj().T @ (G - c * np.eye(3)) @ U + c * np.eye(3)


def vertex_matrices(G_flavor, U, m):
    """K[s, rel, X] (2, 2, 2, 3, 3): the matrices whose element between the daughter's and the parent's mass-basis
    vectors gives alpha (X = 0) and beta (X = 1) (module docstring); s the parent's kind, rel 0 kept / 1 flipped."""
    Gp = coupling_mass_basis(G_flavor, U)
    D = np.diag(np.asarray(m, dtype=np.float64)).astype(np.complex128)
    K = np.zeros((2, 2, 2, 3, 3), dtype=np.complex128)
    for s in range(2):
        Gs = Gp if s == 0 else Gp.conj()
        K[s, 0, 0] = Gs
        K[s, 0, 1] = D @ Gs.conj() @ D
        K[s, 1, 0] = D @ Gs
        K[s, 1, 1] = Gs.conj() @ D
    return K


def coefficients(M):
    """COEF[rel, XY, t, n]: the weights (A z + B + C / z) / M^2 of |alpha|^2 (XY 0), |beta|^2 (1) and 2 Re alpha
    beta* (2) as polynomials: A = sum_n COEF[., ., 0, n] mu_a^n (n <= 2); B = B0 + B1 mu_a + B2 mu' + B3 mu_a mu'
    (t = 1, n = 0..3); C = sum_n COEF[., ., 2, n] mu'^n (n <= 2). Without the overall 1 / M^2."""
    Ms = float(M) * float(M)
    C = np.zeros((2, 3, 3, 4))
    # kept: alpha alpha: [Q^2 + 2 (1-z) M^2 Q - 2 z M^4] / z; beta beta: (1-z)^2 / z; cross: [2 z M^2 - (1-z) Q] / z
    C[0, 0, 0, :3] = (0.0, -2.0 * Ms, 1.0)
    C[0, 0, 1, :] = (-2.0 * Ms * Ms, 2.0 * Ms, 2.0 * Ms, -2.0)
    C[0, 0, 2, :3] = (0.0, -2.0 * Ms, 1.0)
    C[0, 1, 0, 0] = 1.0
    C[0, 1, 1, 0] = -2.0
    C[0, 1, 2, 0] = 1.0
    C[0, 2, 0, 1] = 1.0
    C[0, 2, 1, :3] = (2.0 * Ms, -1.0, -1.0)
    C[0, 2, 2, 1] = 1.0
    # flipped: k_T^2 = (1-z) Q - z M^2; alpha alpha: M^2 2/z + k_T^2/z; beta beta: M^2 2z + k_T^2/z;
    # cross: M^2 2 + k_T^2/z
    C[1, 0, 0, 1] = -1.0
    C[1, 0, 1, :3] = (-Ms, 1.0, 1.0)
    C[1, 0, 2, :2] = (2.0 * Ms, -1.0)
    C[1, 1, 0, :2] = (2.0 * Ms, -1.0)
    C[1, 1, 1, :3] = (-Ms, 1.0, 1.0)
    C[1, 1, 2, 1] = -1.0
    C[1, 2, 0, 1] = -1.0
    C[1, 2, 1, :3] = (Ms, 1.0, 1.0)
    C[1, 2, 2, 1] = -1.0
    return C


@njit(cache=True)
def _pack(Hc, out):
    out[0] = Hc[0, 0].real
    out[1] = Hc[1, 1].real
    out[2] = Hc[2, 2].real
    out[3] = Hc[0, 1].real
    out[4] = Hc[0, 1].imag
    out[5] = Hc[0, 2].real
    out[6] = Hc[0, 2].imag
    out[7] = Hc[1, 2].real
    out[8] = Hc[1, 2].imag


@njit(cache=True)
def _unpack(v, Hc):
    Hc[0, 0] = v[0]
    Hc[1, 1] = v[1]
    Hc[2, 2] = v[2]
    Hc[0, 1] = complex(v[3], v[4])
    Hc[1, 0] = complex(v[3], -v[4])
    Hc[0, 2] = complex(v[5], v[6])
    Hc[2, 0] = complex(v[5], -v[6])
    Hc[1, 2] = complex(v[7], v[8])
    Hc[2, 1] = complex(v[7], -v[8])


@njit(cache=True)
def _sandwich(A, H, B, out):
    """out = A H B^dag (3x3 complex)."""
    T = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                acc += A[i, k] * H[k, j]
            T[i, j] = acc
    for i in range(3):
        for j in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                acc += T[i, k] * np.conj(B[j, k])
            out[i, j] = acc


@njit(cache=True, parallel=True)
def _ladder_eigen(Qs, Mps, ss, floors, shifts, moff, Ucs, Sps, m, sig, mu, m2, dd, P9):
    """At the ladder's sigma values, per kind s and state a (ascending): mu the eigenvalue of M^2 + sigma S_s
    (absolute: the texture's identity shift added back, the kinds compared), m2 = <a|D^2|a> its mass content's mass
    squared, dd = <a|S_s|a> (mu = m2 + sigma dd), P9 the projector c c^dag on the mass basis (packed); per kind the
    texture basis of dp_transit.texture_scales (Qs, Mps, ss, floors), its shift, the offset moff[s] of the masses
    squared over M_rot's eigenvalues (the solver's M_rot holds the differences only), its mixing matrix Ucs[s] and its
    texture on the mass basis Sps[s]. At sigma = 0 the mass states exactly (no rounding of the two kinds'
    diagonalizations: the flips nu_i -> nubar_i stay closed)."""
    n = sig.shape[0]
    order = np.argsort(m)
    for r in prange(n):
        vals = np.empty(3)
        W = np.empty((3, 3), dtype=np.complex128)
        c = np.empty((3, 3), dtype=np.complex128)
        Hc = np.empty((3, 3), dtype=np.complex128)
        out = np.empty(9)
        for s in range(2):
            if sig[r] == 0.0:
                for i in range(3):
                    for a in range(3):
                        c[i, a] = 1.0 if i == order[a] else 0.0
                for a in range(3):
                    i = order[a]
                    mu[r, s, a] = m[i] * m[i]
                    m2[r, s, a] = m[i] * m[i]
                    dd[r, s, a] = Sps[s, i, i].real
                    for k in range(3):
                        for j in range(3):
                            Hc[k, j] = c[k, a] * np.conj(c[j, a])
                    _pack(Hc, out)
                    for q in range(9):
                        P9[r, s, a, q] = out[q]
                continue
            frame_at(Qs[s], Mps[s], ss[s], floors[s], sig[r], vals, W)
            for i in range(3):
                for a in range(3):
                    acc = 0.0 + 0.0j
                    for f in range(3):
                        acc += np.conj(Ucs[s, f, i]) * W[f, a]
                    c[i, a] = acc
            for a in range(3):
                mu[r, s, a] = vals[a] + shifts[s] * sig[r] + moff[s]
                acc_m = 0.0
                for i in range(3):
                    acc_m += (c[i, a].real ** 2 + c[i, a].imag ** 2) * m[i] * m[i]
                m2[r, s, a] = acc_m
                acc_r = 0.0
                for k in range(3):
                    v = 0.0 + 0.0j
                    for f in range(3):
                        v += np.conj(Qs[s, f, k]) * W[f, a]
                    acc_r += (v.real * v.real + v.imag * v.imag) * ss[s, k]
                dd[r, s, a] = acc_r + shifts[s]
                for i in range(3):
                    for j in range(3):
                        Hc[i, j] = c[i, a] * np.conj(c[j, a])
                _pack(Hc, out)
                for q in range(9):
                    P9[r, s, a, q] = out[q]


@njit(cache=True)
def _sparse(v, t, is_max):
    """Sparse table of range max (or min) of v (n,) into t (L, n)."""
    n = v.shape[0]
    for i in range(n):
        t[0, i] = v[i]
    span = 1
    for l in range(1, t.shape[0]):
        for i in range(n):
            j = i + span
            if j < n:
                t[l, i] = max(t[l - 1, i], t[l - 1, j]) if is_max else min(t[l - 1, i], t[l - 1, j])
            else:
                t[l, i] = t[l - 1, i]
        span *= 2


@njit(cache=True)
def _range_q(t, lo, hi, is_max):
    """Range max (or min) of the sparse table t over [lo, hi]."""
    n = hi - lo + 1
    l = 0
    while (1 << (l + 1)) <= n:
        l += 1
    a = t[l, lo]
    b = t[l, hi - (1 << l) + 1]
    return max(a, b) if is_max else min(a, b)


# prefix sums over the daughter rungs rho of P'w x: h^(4 rho) {1, delta, delta^2}; h^(2 rho) {1, m2, delta, delta m2};
# {1, m2, m2^2} (the runs' z, 1 and 1/z terms); NPS of them
NPS = 10
# the parent moments of the runs' gains: z term {1, alpha, alpha^2, sigma, alpha sigma, sigma^2} / E^2; 1 term
# {1, alpha, sigma} <1/u> / E; 1/z term {1}
NMOM = 10


@njit(cache=True)
def _daughter_side(h, m2, dd, D0, P9, P9w, SPup, SPlo, SPdmax, SPdmin, PS):
    """Per daughter kind and state over the ladder: the trace-weighted projectors P9w, the classification bounds'
    sparse tables (range max of eta h^2 and min of eta / h^2, eta = m2 h^(-2 rho); range max and min of
    delta = dd - D0) and the prefix sums PS (NPS weights above)."""
    nr = m2.shape[0]
    lh = np.log(h)
    up = np.empty(nr)
    lo = np.empty(nr)
    dl = np.empty(nr)
    for sd in range(2):
        for b in range(3):
            for r in range(nr):
                for q in range(9):
                    P9w[sd, b, r, q] = P9[r, sd, b, q] * TRW[q]
                et = m2[r, sd, b] * np.exp(-2.0 * lh * r)
                up[r] = et * h * h
                lo[r] = et / (h * h)
                dl[r] = dd[r, sd, b] - D0[sd, b]
            _sparse(up, SPup[sd, b], True)
            _sparse(lo, SPlo[sd, b], False)
            _sparse(dl, SPdmax[sd, b], True)
            _sparse(dl, SPdmin[sd, b], False)
            for w in range(NPS):
                for q in range(9):
                    PS[sd, b, w, 0, q] = 0.0
            for r in range(nr):
                e4 = np.exp(4.0 * lh * r)
                e2 = np.exp(2.0 * lh * r)
                mb = m2[r, sd, b]
                de = dl[r]
                for q in range(9):
                    pw = P9w[sd, b, r, q]
                    PS[sd, b, 0, r + 1, q] = PS[sd, b, 0, r, q] + pw * e4
                    PS[sd, b, 1, r + 1, q] = PS[sd, b, 1, r, q] + pw * e4 * de
                    PS[sd, b, 2, r + 1, q] = PS[sd, b, 2, r, q] + pw * e4 * de * de
                    PS[sd, b, 3, r + 1, q] = PS[sd, b, 3, r, q] + pw * e2
                    PS[sd, b, 4, r + 1, q] = PS[sd, b, 4, r, q] + pw * e2 * mb
                    PS[sd, b, 5, r + 1, q] = PS[sd, b, 5, r, q] + pw * e2 * de
                    PS[sd, b, 6, r + 1, q] = PS[sd, b, 6, r, q] + pw * e2 * de * mb
                    PS[sd, b, 7, r + 1, q] = PS[sd, b, 7, r, q] + pw
                    PS[sd, b, 8, r + 1, q] = PS[sd, b, 8, r, q] + pw * mb
                    PS[sd, b, 9, r + 1, q] = PS[sd, b, 9, r, q] + pw * mb * mb


@njit(cache=True)
def _parent_N(K, s, rel, Pa, N9, Nc, T2, tmp9):
    """N9[XY] = K_X P_a K_Y^dag (XY 0: alpha alpha, 1: beta beta, 2: both cross terms), packed."""
    for xy in range(3):
        if xy == 0:
            _sandwich(K[s, rel, 0], Pa, K[s, rel, 0], Nc)
        elif xy == 1:
            _sandwich(K[s, rel, 1], Pa, K[s, rel, 1], Nc)
        else:
            _sandwich(K[s, rel, 0], Pa, K[s, rel, 1], Nc)
            _sandwich(K[s, rel, 1], Pa, K[s, rel, 0], T2)
            for i in range(3):
                for j in range(3):
                    Nc[i, j] += T2[i, j]
        _pack(Nc, tmp9)
        for q in range(9):
            N9[xy, q] = tmp9[q]


_GLX, _GLW = np.polynomial.legendre.leggauss(8)


@njit(cache=True)
def _window_y(muh, Dl, mb, Ms):
    """The window of k_T^2 = y (Dl - y muh) - (1 - y) Ms >= 0 in y = 1 - z (Q = z muh - mb = Dl - y muh,
    mb = muh - Dl >= 0 the daughter's mass content): (ylo, yhi, zl), zl = 1 - yhi computed as the product of the
    z-roots over the other, mb / (muh (1 - ylo)), so that both ends keep their precision (a window near z = 1 by
    its y, one reaching down to z ~ 1e-16 by its z). Empty when yhi <= ylo. k_T^2 is -Ms at y = 0 and -mb at y = 1:
    one interval."""
    if not muh > 0.0:
        return 0.0, 0.0, 1.0
    bq = Dl + Ms
    if bq <= 0.0:
        return 0.0, 0.0, 1.0
    disc = bq * bq - 4.0 * muh * Ms
    if disc < 0.0:
        return 0.0, 0.0, 1.0
    sq = np.sqrt(disc)
    yhi = (bq + sq) / (2.0 * muh)
    ylo = 2.0 * Ms / (bq + sq)
    zl = mb / (muh * (1.0 - ylo)) if mb > 0.0 else 0.0
    if yhi > 1.0:
        yhi = 1.0
    if zl > 1.0 - ylo:
        zl = 1.0 - ylo
    return ylo, yhi, zl


@njit(cache=True)
def _ycoeffs(rel, tau, Dl, muh, Ms):
    """z |A|^2-weights in y = 1 - z, N(y) = n0 + n1 y + n2 y^2 (the local spectrum is N(y) / z, in the plans' units),
    from the three traces tau[XY], directly in Dl = Q(z = 1) and q1 = -muh (no cancellation near z = 1). Returns
    (n0, n1, N1), N1 = N(1) = n0 + n1 + n2 the 1/z term's coefficient, from its own closed form (exactly zero for a
    massless daughter, m2 = muh - Dl = 0)."""
    q0 = Dl
    q1 = -muh
    mb = muh - Dl
    n0 = 0.0
    n1 = 0.0
    N1 = 0.0
    if rel == 0:
        t = tau[0]
        n0 += t * (q0 * q0 - 2.0 * Ms * Ms)
        n1 += t * (2.0 * q0 * q1 + 2.0 * Ms * q0 + 2.0 * Ms * Ms)
        N1 += t * (mb * mb - 2.0 * Ms * mb)
        N1 += tau[1]
        t = tau[2]
        n0 += t * 2.0 * Ms
        n1 += t * (-2.0 * Ms - q0)
        N1 += t * mb
    else:
        t = tau[0]
        n0 += t * Ms
        n1 += t * (q0 + Ms)
        N1 += t * (2.0 * Ms - mb)
        t = tau[1]
        n0 += t * Ms
        n1 += t * (q0 - 3.0 * Ms)
        N1 -= t * mb
        t = tau[2]
        n0 += t * Ms
        n1 += t * (q0 - Ms)
        N1 -= t * mb
    return n0, n1, N1


@njit(cache=True)
def _Fy(y, n0, n1, N1):
    """int_0^y N(t) / (1 - t) dt = n0 y + (n0 + n1) y^2 / 2 + N1 L3(y), L3 = -ln(1-y) - y - y^2/2 (N(t) / (1 - t)
    = n0 + (n0 + n1) t + N1 t^2 / (1 - t))."""
    if N1 == 0.0:
        return n0 * y + 0.5 * (n0 + n1) * y * y
    if y < 0.05:
        L3 = 0.0
        p = y * y * y
        for k in range(3, 40):
            L3 += p / k
            p *= y
            if p < 1e-18 * L3:
                break
    else:
        L3 = -np.log1p(-y) - y - 0.5 * y * y
    return n0 * y + 0.5 * (n0 + n1) * y * y + N1 * L3


@njit(cache=True)
def _Gz(z, n0, n1, N1):
    """int N(y) / z dz in z (an antiderivative): with N = N1 - (n1 + 2 n2) z + n2 z^2, n2 = N1 - n0 - n1,
    N1 ln z - (n1 + 2 n2) z + n2 z^2 / 2 (well conditioned away from z = 1)."""
    n2 = N1 - n0 - n1
    v = -(n1 + 2.0 * n2) * z + 0.5 * n2 * z * z
    if N1 != 0.0:
        v += N1 * np.log(z)
    return v


@njit(cache=True)
def _seg(zlo, ylo_, zhi, yhi_, n0, n1, N1):
    """int_{zlo}^{zhi} N / z dz, each end given in z and in y = 1 - z (the y form above z = 1/2, the z form below)."""
    if zlo >= 0.5:
        return _Fy(ylo_, n0, n1, N1) - _Fy(yhi_, n0, n1, N1)
    if zhi <= 0.5:
        return _Gz(zhi, n0, n1, N1) - _Gz(zlo, n0, n1, N1)
    return (_Gz(0.5, n0, n1, N1) - _Gz(zlo, n0, n1, N1)) + (_Fy(0.5, n0, n1, N1) - _Fy(yhi_, n0, n1, N1))


@njit(cache=True)
def _inner(u, n0, n1, N1, ylo, yhi, zl, a_lo, a_hi):
    """int of N(y) / z over z in [a_lo / u, a_hi / u] cap the window [zl, 1 - ylo]."""
    if a_lo <= 0.0 or a_lo / u <= zl:
        lo, ylo_e = zl, yhi
    else:
        lo = a_lo / u
        ylo_e = 1.0 - lo
    zh = 1.0 - ylo
    if a_hi / u >= zh:
        hi, yhi_e = zh, ylo
    else:
        hi = a_hi / u
        yhi_e = 1.0 - hi
    if hi <= lo:
        return 0.0
    return _seg(lo, ylo_e, hi, yhi_e, n0, n1, N1)


@njit(cache=True)
def _bin_frac(n0, n1, N1, ylo, yhi, zl, a_lo, a_hi, h):
    """The parent spread uniformly over [E_p / h, E_p h] (as dp_decay.window_plan): (1 / span) int du of _inner,
    8-point Gauss-Legendre on each piece between the clamps' breakpoints (the integrand smooth on each)."""
    u1 = 1.0 / h
    u2 = h
    zh = 1.0 - ylo
    bp = np.empty(6)
    nb = 0
    bp[nb] = u1
    nb += 1
    for v in ((a_hi / zl) if zl > 0.0 else 0.0, a_hi / zh, (a_lo / zl) if a_lo > 0.0 and zl > 0.0 else 0.0,
              (a_lo / zh) if a_lo > 0.0 else 0.0):
        if v > u1 and v < u2:
            bp[nb] = v
            nb += 1
    bp[nb] = u2
    nb += 1
    srt = np.sort(bp[:nb])
    tot = 0.0
    for i in range(nb - 1):
        a = srt[i]
        b = srt[i + 1]
        if b <= a:
            continue
        hm = 0.5 * (b - a)
        cm = 0.5 * (b + a)
        acc = 0.0
        for g in range(8):
            acc += _GLW[g] * _inner(cm + hm * _GLX[g], n0, n1, N1, ylo, yhi, zl, a_lo, a_hi)
        tot += hm * acc
    return tot / (u2 - u1)


@njit(cache=True)
def _explicit(rel, N9, Pw, muh, Dl, mb, Ms, h, a_lo, a_hi, tau):
    """The rate (units of 1 / (16 pi M^2 E_p)) into the daughter energies [a_lo, a_hi] E_p: exact over the parent's
    bin and over the window (the daughter bin's eigenvector held: Q = z muh - mb, mb = <D^2>_b, muh = Dl + mb;
    module docstring), integrated in y = 1 - z near z = 1 and in z below 1/2. a_lo = 0: everything below a_hi."""
    ylo, yhi, zl = _window_y(muh, Dl, mb, Ms)
    if yhi <= ylo:
        return 0.0
    for xy in range(3):
        acc = 0.0
        for q in range(9):
            acc += N9[xy, q] * Pw[q]
        tau[xy] = acc
    n0, n1, N1 = _ycoeffs(rel, tau, Dl, muh, Ms)
    if a_lo <= 0.0 and zl <= 0.0 and N1 != 0.0:
        return np.nan                    # a 1/z term down to z = 0: not integrable (the caller's error)
    return _bin_frac(n0, n1, N1, ylo, yhi, zl, a_lo, a_hi, h)


@njit(cache=True, parallel=True)
def _plans(h, Ms, CO, K, mu, m2, dd, D0, P9, sig, P9w, SPup, SPlo, SPdmax, SPdmin, PS, plan_tot, plan_own,
           plan_nexp, plan_k, plan_f, plan_nrun, plan_run, err):
    """The decay plan of every rung r and channel ch (module docstring), in units of 1 / (16 pi M^2 E_parent): the total
    rate plan_tot[r, ch] (with the daughters below the ladder by the vacuum tail), the own bin's plan_own[r, ch], the
    cut bins plan_k / plan_f[r, ch, :plan_nexp] (offsets k >= 1, rates), the runs of bins wholly inside
    plan_run[r, ch, :plan_nrun] = (k1, k2) (offsets, k1 <= k2). With the daughter bin's eigenvector held, the kinematics
    is Q = z alpha_hat - z sigma delta(rho) - m2(rho), alpha_hat = mu_a - sigma D0_b, delta = <S>_b - D0_b (D0_b the
    texture's own eigenvalue the state tends to, so that sigma delta stays small where sigma is large). err[r] != 0: the
    rung's plan failed (1: more cut bins than plan_k holds, 2: more runs than plan_run holds, 3: the stack, 4: a
    non-integrable tail)."""
    nr = mu.shape[0]
    nexp_cap = plan_k.shape[2]
    nrun_cap = plan_run.shape[2]
    lh = np.log(h)
    span = h - 1.0 / h
    uinv = 2.0 * lh / span
    q2rel = 0.5 * (h * h - 1.0 / (h * h))
    Lb = 2.0 * lh
    for r in prange(nr):
        Pa = np.empty((3, 3), dtype=np.complex128)
        Nc = np.empty((3, 3), dtype=np.complex128)
        T2 = np.empty((3, 3), dtype=np.complex128)
        N9 = np.zeros((3, 9))
        tmp9 = np.empty(9)
        tau = np.zeros(3)
        stack_lo = np.empty(512, dtype=np.int64)
        stack_hi = np.empty(512, dtype=np.int64)
        e2r = np.exp(2.0 * lh * r)
        e2m = np.exp(-2.0 * lh * r)
        e4m = np.exp(-4.0 * lh * r)
        sr = sig[r]
        for ch in range(NCH):
            plan_tot[r, ch] = 0.0
            plan_own[r, ch] = 0.0
            plan_nexp[r, ch] = 0
            plan_nrun[r, ch] = 0
        for s in range(2):
            for a in range(3):
                for q in range(9):
                    tmp9[q] = P9[r, s, a, q]
                _unpack(tmp9, Pa)
                for rel in range(2):
                    sd = s if rel == 0 else 1 - s
                    _parent_N(K, s, rel, Pa, N9, Nc, T2, tmp9)
                    for b in range(3):
                        ch = ((s * 3 + a) * 2 + rel) * 3 + b
                        if rel == 0 and b >= a:
                            continue                     # kept decays go to lower states only: a -> a is closed
                        if mu[r, s, a] <= mu[r, sd, b] + Ms:
                            continue                     # closed (mu'_b(z sigma) / z >= mu'_b(sigma), Weyl)
                        al = mu[r, s, a] - sr * D0[sd, b]
                        tot = 0.0
                        ne = 0
                        nru = 0
                        # own bin (k = 0) and k = 1: always cut (the window's top, the parent's spread)
                        for k in range(2):
                            rho = r - k
                            if rho < 0:
                                break
                            Dl = m2[r, s, a] - m2[rho, sd, b] + sr * (dd[r, s, a] - dd[rho, sd, b])
                            mb = m2[rho, sd, b]
                            f = _explicit(rel, N9, P9w[sd, b, rho], Dl + mb, Dl, mb, Ms, h,
                                          np.exp(lh * (-2.0 * k - 1.0)), np.exp(lh * (-2.0 * k + 1.0)), tau)
                            if k == 0:
                                plan_own[r, ch] = f
                            elif f != 0.0:
                                plan_k[r, ch, ne] = k
                                plan_f[r, ch, ne] = f
                                ne += 1
                            tot += f
                        nst = 0
                        if r - 2 >= 0:
                            stack_lo[0] = 0
                            stack_hi[0] = r - 2
                            nst = 1
                        while nst > 0 and err[r] == 0:
                            nst -= 1
                            lo = stack_lo[nst]
                            hi = stack_hi[nst]
                            if (e2r * _range_q(SPlo[sd, b], lo, hi, False) + sr * _range_q(SPdmin[sd, b], lo, hi, False)
                                    + Ms > al):
                                continue                 # empty for every daughter energy and parent position
                            zmax = np.exp(-2.0 * lh * (r - hi) + 2.0 * lh)
                            if (e2r * _range_q(SPup[sd, b], lo, hi, True) + sr * _range_q(SPdmax[sd, b], lo, hi, True)
                                    + Ms / (1.0 - zmax) <= al):
                                # wholly inside for every parent position: a run, its rate from the prefix sums (the
                                # ranges come from the top down: one adjacent to the last run extends it)
                                if nru > 0 and plan_run[r, ch, nru - 1, 1] == r - hi - 1:
                                    plan_run[r, ch, nru - 1, 1] = r - lo
                                else:
                                    if nru >= nrun_cap:
                                        err[r] = 2
                                        break
                                    plan_run[r, ch, nru, 0] = r - hi
                                    plan_run[r, ch, nru, 1] = r - lo
                                    nru += 1
                                for xy in range(3):
                                    A0 = CO[rel, xy, 0, 0]
                                    A1 = CO[rel, xy, 0, 1]
                                    A2 = CO[rel, xy, 0, 2]
                                    B0 = CO[rel, xy, 1, 0]
                                    B1 = CO[rel, xy, 1, 1]
                                    B2 = CO[rel, xy, 1, 2]
                                    B3 = CO[rel, xy, 1, 3]
                                    fz = q2rel * e4m
                                    f1 = uinv * span * e2m
                                    w = np.empty(NPS)
                                    w[0] = fz * (A0 + A1 * al + A2 * al * al)
                                    w[1] = -fz * sr * (A1 + 2.0 * A2 * al)
                                    w[2] = fz * A2 * sr * sr
                                    w[3] = f1 * (B0 + B1 * al)
                                    w[4] = f1 * (B2 + B3 * al)
                                    w[5] = -f1 * sr * B1
                                    w[6] = -f1 * sr * B3
                                    w[7] = Lb * CO[rel, xy, 2, 0]
                                    w[8] = Lb * CO[rel, xy, 2, 1]
                                    w[9] = Lb * CO[rel, xy, 2, 2]
                                    acc = 0.0
                                    for q in range(9):
                                        yq = 0.0
                                        for ww in range(NPS):
                                            if w[ww] != 0.0:
                                                yq += w[ww] * (PS[sd, b, ww, hi + 1, q] - PS[sd, b, ww, lo, q])
                                        acc += N9[xy, q] * yq
                                    tot += acc
                                continue
                            if lo == hi:
                                k = r - lo
                                Dl = m2[r, s, a] - m2[lo, sd, b] + sr * (dd[r, s, a] - dd[lo, sd, b])
                                mb = m2[lo, sd, b]
                                f = _explicit(rel, N9, P9w[sd, b, lo], Dl + mb, Dl, mb, Ms, h,
                                              np.exp(lh * (-2.0 * k - 1.0)), np.exp(lh * (-2.0 * k + 1.0)), tau)
                                if f != 0.0:
                                    if ne >= nexp_cap:
                                        err[r] = 1
                                        break
                                    plan_k[r, ch, ne] = k
                                    plan_f[r, ch, ne] = f
                                    ne += 1
                                    tot += f
                                continue
                            if nst + 2 > stack_lo.shape[0]:
                                err[r] = 3
                                break
                            mid = (lo + hi) // 2
                            stack_lo[nst] = lo
                            stack_hi[nst] = mid
                            nst += 1
                            stack_lo[nst] = mid + 1
                            stack_hi[nst] = hi
                            nst += 1
                        # the daughters below the ladder (rung 0: the vacuum daughter)
                        Dl = m2[r, s, a] - m2[0, sd, b] + sr * (dd[r, s, a] - dd[0, sd, b])
                        mb = m2[0, sd, b]
                        ft = _explicit(rel, N9, P9w[sd, b, 0], Dl + mb, Dl, mb, Ms, h, 0.0,
                                       np.exp(lh * (-2.0 * r - 1.0)), tau)
                        if ft != ft:
                            err[r] = 4
                        else:
                            tot += ft
                        plan_tot[r, ch] = tot
                        plan_nexp[r, ch] = ne
                        plan_nrun[r, ch] = nru


@njit(cache=True, parallel=True)
def _psi(K, CO, mu, m2, dd, D0, P9, psi, zeno, plan_tot, Ms):
    """psi[r, cl, k] (trace-weighted, packed): the daughter side of the runs' gains at rung r for class
    cl = (s, rel, b): sum_XY coef_k R_XY, R_XY = K_Y^dag P'_b K_X, coef_k the daughter's factor of parent moment k
    (NMOM, deposit). zeno[r]: max over the states of 2 E Gamma_a / min_b |mu_a - mu_b| over the other states of its
    kind (the decay width over the smallest splitting: the eigenstate picture needs it small)."""
    nr = mu.shape[0]
    for r in prange(nr):
        Pb = np.empty((3, 3), dtype=np.complex128)
        Rc = np.empty((3, 3), dtype=np.complex128)
        T2 = np.empty((3, 3), dtype=np.complex128)
        tmp9 = np.empty(9)
        R9 = np.zeros((3, 9))
        KYd = np.empty((3, 3), dtype=np.complex128)
        KXd = np.empty((3, 3), dtype=np.complex128)
        cf = np.empty(NMOM)
        for s in range(2):
            for rel in range(2):
                sd = s if rel == 0 else 1 - s
                for b in range(3):
                    for q in range(9):
                        tmp9[q] = P9[r, sd, b, q]
                    _unpack(tmp9, Pb)
                    for xy in range(3):
                        X = 1 if xy == 1 else 0
                        Y = 0 if xy == 0 else 1
                        # R_XY = K_Y^dag P' K_X = A P' B^dag with A = K_Y^dag, B = K_X^dag (and the swapped term)
                        for i in range(3):
                            for j in range(3):
                                KYd[i, j] = np.conj(K[s, rel, Y, j, i])
                                KXd[i, j] = np.conj(K[s, rel, X, j, i])
                        _sandwich(KYd, Pb, KXd, Rc)
                        if xy == 2:
                            _sandwich(KXd, Pb, KYd, T2)
                            for i in range(3):
                                for j in range(3):
                                    Rc[i, j] += T2[i, j]
                        _pack(Rc, tmp9)
                        for q in range(9):
                            R9[xy, q] = tmp9[q] * TRW[q]
                    cl = (s * 2 + rel) * 3 + b
                    mb = m2[r, sd, b]
                    de = dd[r, sd, b] - D0[sd, b]
                    for k in range(NMOM):
                        for q in range(9):
                            psi[r, cl, k, q] = 0.0
                    for xy in range(3):
                        A1 = CO[rel, xy, 0, 1]
                        A2 = CO[rel, xy, 0, 2]
                        B1 = CO[rel, xy, 1, 1]
                        B3 = CO[rel, xy, 1, 3]
                        cf[0] = CO[rel, xy, 0, 0]
                        cf[1] = A1
                        cf[2] = A2
                        cf[3] = -A1 * de
                        cf[4] = -2.0 * A2 * de
                        cf[5] = A2 * de * de
                        cf[6] = CO[rel, xy, 1, 0] + CO[rel, xy, 1, 2] * mb
                        cf[7] = B1 + B3 * mb
                        cf[8] = -de * (B1 + B3 * mb)
                        cf[9] = CO[rel, xy, 2, 0] + CO[rel, xy, 2, 1] * mb + CO[rel, xy, 2, 2] * mb * mb
                        for k in range(NMOM):
                            if cf[k] == 0.0:
                                continue
                            for q in range(9):
                                psi[r, cl, k, q] += cf[k] * R9[xy, q]
        z = 0.0
        for s in range(2):
            for a in range(3):
                W = 0.0
                for rel in range(2):
                    for b in range(3):
                        W += plan_tot[r, ((s * 3 + a) * 2 + rel) * 3 + b]
                if W <= 0.0:
                    continue
                W /= 16.0 * np.pi * Ms
                gap = np.inf
                for b in range(3):
                    if b != a:
                        gap = min(gap, abs(mu[r, s, a] - mu[r, s, b]))
                z = max(z, 2.0 * W / gap) if gap > 0.0 else np.inf
        zeno[r] = z


class Ladder:
    """The decay plans of a run (module docstring) on the sigma values sig[r] = sig0 h^(2 r) (h the grid's half-bin
    ratio): .arrays for the njit kernels (sig, mu, P9, D0, plan_tot, plan_own, plan_nexp, plan_k, plan_f, plan_nrun,
    plan_run, psi, zeno, Ms, sig0: the rungs' nominal sigma is sig0 h^(2 r), sig the physical one, zero below the
    vacuum); .coord(c) the ladder coordinate of the grid's bin 0 for a coefficient c (sigma = E |c|);
    .zeno[r] the validity measure; .diag[0] the largest one the steps met."""

    def __init__(self, E_bins, G_flavor, U, m, M, kinds, c_max, c_min=0.0):
        """E_bins the energy grid (geometric); G_flavor the mediator's coupling (flavor basis, the whole of it); U the
        mixing matrix, m the masses, M the mediator mass; kinds: two (M_rot, S_signed) of
        dark_portal_solver.propagate_spectrum (H(sigma) = M_rot + sigma S_signed, sigma = |E c|, per kind); c_max the
        largest |c| of the steps that will use it (c_min unused)."""
        E = np.asarray(E_bins, dtype=np.float64)
        self.E0 = float(E[0])
        self.h = float(np.sqrt(E[1] / E[0]))
        self.N = len(E)
        lh = np.log(self.h)
        tex = []
        shifts = np.zeros(2)
        moff = np.zeros(2)
        D0 = np.zeros((2, 3))
        gaps = []
        smax = 0.0
        for s, (M_rot, S_signed) in enumerate(kinds):
            Q_t, s_t, Mp_t, floor_t, _, _ = texture_scales(M_rot, S_signed)
            tex.append((Q_t, Mp_t, s_t, floor_t))
            S_h = 0.5 * (np.asarray(S_signed, dtype=np.complex128) + np.asarray(S_signed, dtype=np.complex128).conj().T)
            shifts[s] = (np.trace(S_h).real - float(np.sum(s_t))) / 3.0
            D0[s] = np.sort(np.asarray(s_t, dtype=np.float64)) + shifts[s]
            ev = np.sort(np.linalg.eigvalsh(np.asarray(M_rot, dtype=np.complex128)))
            gaps.append(float(np.min(np.diff(ev))))
            # the masses squared above M_rot's eigenvalues (the solver's M_rot holds U diag(0, dm21, dm31) U^dag)
            moff[s] = float(np.sum(np.asarray(m, dtype=np.float64) ** 2) - np.trace(np.asarray(M_rot)).real) / 3.0
            if np.max(np.abs(np.sort(np.asarray(m, dtype=np.float64) ** 2) - (ev + moff[s]))) > 1e-9 * float(
                    np.max(np.asarray(m, dtype=np.float64) ** 2)):
                raise ValueError("dp_medium.Ladder: M_rot's eigenvalues are not the masses squared up to a constant")
            smax = max(smax, float(np.max(np.abs(np.linalg.eigvalsh(S_h)))))
        # the ladder's bottom: a grid's length below the vacuum (SIGMA_VAC relative), so that a step whose bin 0 lies
        # below it (c = 0 included) has every bin's plan on vacuum rungs (bin p at rung p + 2)
        sig_vac = SIGMA_VAC * min(gaps) / max(smax, 1e-300)
        self.sig_vac = sig_vac
        top = self.E0 * float(c_max) * np.exp(2.0 * lh * (self.N + 2))
        if not top > sig_vac:
            top = sig_vac
        bottom = sig_vac * np.exp(-2.0 * lh * (self.N + 4))
        nr = int(np.ceil(np.log(top / bottom) / (2.0 * lh))) + 3
        self.sig0 = bottom
        sig = bottom * np.exp(2.0 * lh * np.arange(nr))
        # below sig_vac the vacuum (to SIGMA_VAC): exactly sigma = 0 there
        sig[sig < sig_vac * (1.0 - 1e-12)] = 0.0
        self.nr = nr
        Qs = np.ascontiguousarray(np.array([t[0] for t in tex]))
        Mps = np.ascontiguousarray(np.array([t[1] for t in tex]))
        ss = np.ascontiguousarray(np.array([t[2] for t in tex]))
        floors = np.array([t[3] for t in tex])
        Ucs = np.ascontiguousarray(np.array([np.asarray(U, dtype=np.complex128),
                                              np.conj(np.asarray(U, dtype=np.complex128))]))
        mvec = np.ascontiguousarray(np.asarray(m, dtype=np.float64))
        Sps = np.ascontiguousarray(np.array([Ucs[s].conj().T @ np.asarray(kinds[s][1], dtype=np.complex128) @ Ucs[s]
                                             for s in range(2)]))
        mu = np.zeros((nr, 2, 3))
        m2 = np.zeros((nr, 2, 3))
        dd = np.zeros((nr, 2, 3))
        P9 = np.zeros((nr, 2, 3, 9))
        _ladder_eigen(Qs, Mps, ss, floors, shifts, moff, Ucs, Sps, mvec, sig, mu, m2, dd, P9)
        K = np.ascontiguousarray(vertex_matrices(G_flavor, U, m))
        CO = coefficients(M)
        Ms = float(M) * float(M)
        L = max(1, int(np.ceil(np.log2(nr))) + 1)
        P9w = np.zeros((2, 3, nr, 9))
        SPup = np.zeros((2, 3, L, nr))
        SPlo = np.zeros((2, 3, L, nr))
        SPdmax = np.zeros((2, 3, L, nr))
        SPdmin = np.zeros((2, 3, L, nr))
        PS = np.zeros((2, 3, NPS, nr + 1, 9))
        _daughter_side(self.h, m2, dd, D0, P9, P9w, SPup, SPlo, SPdmax, SPdmin, PS)
        plan_tot = np.zeros((nr, NCH))
        plan_own = np.zeros((nr, NCH))
        plan_nexp = np.zeros((nr, NCH), dtype=np.int64)
        plan_nrun = np.zeros((nr, NCH), dtype=np.int64)
        err = np.zeros(nr, dtype=np.int64)
        nexp, nrun = NEXP, NRUN
        while True:
            # the room per channel grows until every plan fits: a plan cuts at most one bin, and has at most one run,
            # per rung below its own, so nr bounds both
            plan_k = np.zeros((nr, NCH, nexp), dtype=np.int64)
            plan_f = np.zeros((nr, NCH, nexp))
            plan_run = np.zeros((nr, NCH, nrun, 2), dtype=np.int64)
            err[:] = 0
            _plans(self.h, Ms, CO, K, mu, m2, dd, D0, P9, sig, P9w, SPup, SPlo, SPdmax, SPdmin, PS, plan_tot,
                   plan_own, plan_nexp, plan_k, plan_f, plan_nrun, plan_run, err)
            if not np.any(err):
                break
            codes = set(int(c) for c in err[err != 0])
            if not codes <= {1, 2} or nexp > nr + 2 or nrun > nr + 2:
                bad = int(np.argmax(err != 0))
                raise ValueError(f"dp_medium.Ladder: the plan of rung {bad} (sigma {sig[bad]:.3e}) failed, code "
                                 f"{int(err[bad])} (1 cut bins, 2 runs, 3 stack, 4 vacuum tail)")
            if 1 in codes:
                nexp *= 2
            if 2 in codes:
                nrun *= 2
        self.capacity = (nexp, nrun)
        psi = np.zeros((nr, NCL, NMOM, 9))
        zeno = np.zeros(nr)
        _psi(K, CO, mu, m2, dd, D0, P9, psi, zeno, plan_tot, Ms)
        self.sig = sig
        self.mu = mu
        self.m2 = m2
        self.dd = dd
        self.D0 = D0
        self.P9 = P9
        self.Ms = Ms
        self.zeno = zeno
        self.diag = np.zeros(2)
        self.arrays = (sig, mu, P9, D0, plan_tot, plan_own, plan_nexp, plan_k, plan_f, plan_nrun, plan_run, psi, zeno,
                       Ms, float(bottom))

    def coord(self, c):
        """The ladder coordinate x0 of the grid's bin 0 for the coefficient c (bin p between rungs floor(x0 + p) and
        the next, weight x0 + p - floor on the upper); -1 where bin 0 lies below the ladder (the vacuum)."""
        c = abs(float(c))
        if c == 0.0 or self.E0 * c <= self.sig0:
            return -1.0
        x0 = np.log(self.E0 * c / self.sig0) / (2.0 * np.log(self.h))
        if x0 + self.N + 1 > self.nr - 1:
            raise ValueError(f"dp_medium.Ladder: coefficient {c:.3e} beyond the ladder's top")
        return float(x0)

    def width(self, E, c):
        """E Gamma (eV^2) per kind and state, (2, 3), at energy E (observed) and coefficient c, interpolated (the total
        lab width is this / (E (1+z)))."""
        x = np.log(E * abs(c) / self.sig0) / (2.0 * np.log(self.h)) if c != 0.0 else 2.0
        r0 = int(np.floor(x))
        w = x - r0
        tot = self.arrays[4]
        out = np.zeros((2, 3))
        for s in range(2):
            for a in range(3):
                for rel in range(2):
                    for b in range(3):
                        ch = ((s * 3 + a) * 2 + rel) * 3 + b
                        out[s, a] += (1.0 - w) * tot[r0, ch] + w * tot[min(r0 + 1, self.nr - 1), ch]
        return out / (16.0 * np.pi * self.Ms)

    def buffers(self):
        """Work arrays of the kernels (deposit, gains), zero, reusable: (pend, acc, add, sub, act), act[set, cl] the
        classes a sweep has deposited into (the others skipped); every call leaves them zero again."""
        return (np.zeros((self.N, 2, 3)), np.zeros((2, NCL, NMOM, 9)), np.zeros((self.N, 2, NCL, NMOM, 9)),
                np.zeros((self.N, 2, NCL, NMOM, 9)), np.zeros((2, NCL), dtype=np.bool_))


@njit(cache=True)
def rung_of_bin(x0, p):
    """(r_lo, w): the lower rung and the upper rung's weight of grid bin p (Ladder.coord's x0; x0 < 0: the vacuum)."""
    if x0 < 0.0:
        return p + 2, 0.0
    x = x0 + p
    r = int(np.floor(x))
    return r, x - r


@njit(cache=True)
def bin_rates(lad, x0, p, pref, Lam, own):
    """The decay rates of grid bin p's states per unit population: Lam[s, a] (total) and own[s, a, s', b] (into the
    bin itself), the plans of the two rungs around it blended; pref = 1 / (16 pi M^2 E_p) x the time unit."""
    plan_tot = lad[4]
    plan_own = lad[5]
    r0, w = rung_of_bin(x0, p)
    for s in range(2):
        for a in range(3):
            Lam[s, a] = 0.0
            for sd in range(2):
                for b in range(3):
                    own[s, a, sd, b] = 0.0
            for rel in range(2):
                sd = s if rel == 0 else 1 - s
                for b in range(3):
                    ch = ((s * 3 + a) * 2 + rel) * 3 + b
                    t = (1.0 - w) * plan_tot[r0, ch]
                    o = (1.0 - w) * plan_own[r0, ch]
                    if w > 0.0:
                        t += w * plan_tot[r0 + 1, ch]
                        o += w * plan_own[r0 + 1, ch]
                    Lam[s, a] += t * pref
                    own[s, a, sd, b] += o * pref


@njit(cache=True)
def deposit(lad, x0, p, s, a, n, Ep, h, pend, add, sub, act):
    """State (s, a) of grid bin p decayed (n: its decayed number x pref, the plans' unit): its daughters below its bin
    -- the cut bins' numbers into pend[e, s', b], the runs into the gains' running sums add / sub[e, set, cl, k, :]
    (set 0 the lower rung's plan, 1 the upper's; k the NMOM parent moments of the parent's projector)."""
    sig = lad[0]
    mu = lad[1]
    P9 = lad[2]
    D0 = lad[3]
    plan_nexp = lad[6]
    plan_k = lad[7]
    plan_f = lad[8]
    plan_nrun = lad[9]
    plan_run = lad[10]
    r0, w = rung_of_bin(x0, p)
    uinv = 2.0 * np.log(h) / (h - 1.0 / h)
    mom = np.empty(NMOM)
    vec = np.empty((NMOM, 9))
    for st in range(2):
        r = r0 + st
        om = (1.0 - w) if st == 0 else w
        if om == 0.0:
            continue
        nw = n * om
        sr = sig[r]
        for rel in range(2):
            sd = s if rel == 0 else 1 - s
            for b in range(3):
                ch = ((s * 3 + a) * 2 + rel) * 3 + b
                for x in range(plan_nexp[r, ch]):
                    e = p - plan_k[r, ch, x]
                    if e >= 0:
                        pend[e, sd, b] += nw * plan_f[r, ch, x]
                nru = plan_nrun[r, ch]
                if nru == 0:
                    continue
                cl = (s * 2 + rel) * 3 + b
                act[st, cl] = True
                al = mu[r, s, a] - sr * D0[sd, b]
                fz = nw / (Ep * Ep)
                f1 = nw * uinv / Ep
                mom[0] = fz
                mom[1] = fz * al
                mom[2] = fz * al * al
                mom[3] = fz * sr
                mom[4] = fz * al * sr
                mom[5] = fz * sr * sr
                mom[6] = f1
                mom[7] = f1 * al
                mom[8] = f1 * sr
                mom[9] = nw
                for k in range(NMOM):
                    for q in range(9):
                        vec[k, q] = mom[k] * P9[r, s, a, q]
                for x in range(nru):
                    top = p - plan_run[r, ch, x, 0]
                    if top < 0:
                        continue
                    bot = p - plan_run[r, ch, x, 1]
                    if bot < 0:
                        bot = 0
                    for k in range(NMOM):
                        for q in range(9):
                            add[top, st, cl, k, q] += vec[k, q]
                            sub[bot, st, cl, k, q] += vec[k, q]


@njit(cache=True)
def gains(lad, x0, e, Ee, h, acc, add, sub, pend, out, act):
    """The daughters arriving in grid bin e (numbers, out (2, 3) per kind and state): pend's and the running sums'
    (acc[set, cl, k, q]: entered with add[e], read, left with sub[e]); contracted with psi at the bin's rungs and the
    bin's moments (q2, dE, L). Clears pend[e], add[e], sub[e]."""
    psi = lad[11]
    r0, w = rung_of_bin(x0, e)
    dE = Ee * (h - 1.0 / h)
    q2 = 0.5 * Ee * Ee * (h * h - 1.0 / (h * h))
    Lb = 2.0 * np.log(h)
    for sd in range(2):
        for b in range(3):
            out[sd, b] = pend[e, sd, b]
            pend[e, sd, b] = 0.0
    for st in range(2):
        r = r0 + st
        for cl in range(NCL):
            if not act[st, cl]:
                continue
            s = cl // 6
            rel = (cl // 3) % 2
            b = cl % 3
            sd = s if rel == 0 else 1 - s
            nz = False
            for k in range(NMOM):
                for q in range(9):
                    v = add[e, st, cl, k, q]
                    if v != 0.0:
                        acc[st, cl, k, q] += v
                        add[e, st, cl, k, q] = 0.0
                    if acc[st, cl, k, q] != 0.0:
                        nz = True
            if nz:
                v = 0.0
                for k in range(NMOM):
                    fk = q2 if k <= 5 else (dE if k <= 8 else Lb)
                    t = 0.0
                    for q in range(9):
                        t += acc[st, cl, k, q] * psi[r, cl, k, q]
                    v += fk * t
                out[sd, b] += v
            for k in range(NMOM):
                for q in range(9):
                    sv = sub[e, st, cl, k, q]
                    if sv != 0.0:
                        acc[st, cl, k, q] -= sv
                        sub[e, st, cl, k, q] = 0.0
            if e == 0:
                act[st, cl] = False
                for k in range(NMOM):
                    for q in range(9):
                        acc[st, cl, k, q] = 0.0


@njit(cache=True)
def ladder_x0(lad, E0, h, c, N):
    """Ladder.coord inside the kernels: the ladder coordinate of the grid's bin 0 (energy E0, half-bin ratio h, N
    bins) for the coefficient c; -1 where bin 0 lies below the ladder (the vacuum). Raises beyond the ladder's top."""
    sig0 = lad[14]
    c = abs(c)
    if c == 0.0 or E0 * c <= sig0:
        return -1.0
    x0 = np.log(E0 * c / sig0) / (2.0 * np.log(h))
    if x0 + N + 1 > lad[0].shape[0] - 1:
        raise ValueError("dp_medium.ladder_x0: coefficient beyond the ladder's top")
    return x0


@njit(cache=True)
def widths(lad, x0, E, W0, W1):
    """The states' decay widths per unit time (or length) at z = 0, W0, W1 (N, 3) per kind (the jumps' damping)."""
    N = E.shape[0]
    Ms = lad[13]
    Lam = np.zeros((2, 3))
    own = np.zeros((2, 3, 2, 3))
    for i in range(N):
        bin_rates(lad, x0, i, 1.0 / (16.0 * np.pi * Ms * E[i]), Lam, own)
        for a in range(3):
            W0[i, a] = Lam[0, a]
            W1[i, a] = Lam[1, a]


@njit(cache=True)
def decay_sweep(lad, x0, E, F0, F1, L, pend, acc, add, sub, act):
    """The decays alone over a path L (eV^-1; z = 0, rates per unit length), both kinds' populations per unit energy
    in place (dp_galactic_path's segments): from the highest energy down, per bin the daughters of the bins above
    arrive first, then the bin's six states decay exactly, in descending order of their effective masses squared
    (absolute: the kinds compared), so that the bin's own daughters reach the lower states before these decay (as
    dp_decay.decay_step); the daughters below by the plans. Linear in the populations (the sign-indefinite spectral
    bases of the vacuum path). pend, acc, add, sub, act: Ladder.buffers (left zero)."""
    N = E.shape[0]
    h = np.sqrt(E[1] / E[0])
    Ms = lad[13]
    mu = lad[1]
    gin = np.zeros((2, 3))
    Lam = np.zeros((2, 3))
    own = np.zeros((2, 3, 2, 3))
    mabs = np.zeros(6)
    order = np.zeros(6, dtype=np.int64)
    for st in range(2):
        for cl in range(NCL):
            act[st, cl] = False
            for k in range(NMOM):
                for q in range(9):
                    acc[st, cl, k, q] = 0.0
    for b in range(N - 1, -1, -1):
        Eb = E[b]
        dEb = Eb * h - Eb / h
        gains(lad, x0, b, Eb, h, acc, add, sub, pend, gin, act)
        for a in range(3):
            if gin[0, a] != 0.0:
                F0[b, a] += gin[0, a] / dEb
            if gin[1, a] != 0.0:
                F1[b, a] += gin[1, a] / dEb
        pref = 1.0 / (16.0 * np.pi * Ms * Eb)
        bin_rates(lad, x0, b, pref, Lam, own)
        r0, w = rung_of_bin(x0, b)
        for s in range(2):
            for a in range(3):
                v = (1.0 - w) * mu[r0, s, a]
                if w > 0.0:
                    v += w * mu[r0 + 1, s, a]
                mabs[3 * s + a] = v
        for i in range(6):
            order[i] = i
        for x in range(6):
            for y in range(5 - x):
                if mabs[order[y]] < mabs[order[y + 1]]:
                    tmp = order[y]
                    order[y] = order[y + 1]
                    order[y + 1] = tmp
        for oi in range(6):
            c = order[oi]
            s = c // 3
            a = c - 3 * s
            F = F0 if s == 0 else F1
            La = Lam[s, a]
            if La <= 0.0:
                continue
            Na = F[b, a] * dEb
            if Na == 0.0:
                continue
            p = -np.expm1(-La * L)
            Nd = Na * p
            F[b, a] = Na * (1.0 - p) / dEb
            for sd in range(2):
                Fd = F0 if sd == 0 else F1
                for bb in range(3):
                    o = own[s, a, sd, bb]
                    if o != 0.0:
                        Fd[b, bb] += Nd * o / La / dEb
            deposit(lad, x0, b, s, a, Nd * pref / La, Eb, h, pend, add, sub, act)
