"""
dp_transit.py

The flavor transit from the cosmological arrival to the Earth's surface (2026-10-13):

  Galaxy  -- adiabatic. The Galactic potential changes over >= 10 pc, against oscillation
             lengths <= 0.005 pc at 30 PeV, so each matter eigenstate keeps its population
             and the state arriving in the solar neighbourhood is diagonal in the local
             eigenbasis. Only the local potential matters: the Yukawa integral of the
             Galactic neutrons (and the cosmic mean beyond) over the mediator's range
             (galactic_effective_density).
  Sun     -- coherent. Its 1/r potential turns on over AU scales, comparable to the
             oscillation lengths, so the transit can be adiabatic, sudden or in between; we
             integrate it along paths at Gauss-Legendre angles to the Sun and average them.
  Earth's exterior -- coherent beyond r_split (radial), and per chord inside it (see
             dp_spatial.propagate_earth_spatial).

The integrator works in the instantaneous eigenbasis of H = (M_rot + sigma S)/2E, sigma =
2E |Coef| n >= 0 (S carries the sign of Coef). Its eigenvectors depend on sigma only, so a
table of them (build_frame_table) serves every energy and path. Between consecutive table
nodes the eigenbasis turns by K_n = log(W_n^dag W_n+1), at most `dtheta`; the dynamic phases
are integrated exactly; and the non-adiabatic coupling K_n/h is integrated against the phase
difference (first-order Magnus with a Filon rule, exact for a linear phase across the step).
The scheme is exact in the adiabatic and sudden limits and second order in dtheta between
them.

The eigenvectors come from a Jacobi iteration in the eigenbasis of the texture S
(eigh_graded). The potential can exceed the vacuum term by 1e30, and a rank-one texture
leaves two of its eigenvalues degenerate, split only by the vacuum term: any method that
forms H first (the closed form, LAPACK) loses that splitting to rounding (eps * sigma).
Jacobi in the texture basis keeps it: the large entries sit on the diagonal, the texture's
degenerate eigenvalues are shifted to exactly zero (texture_basis), and every rotation acts on
entries of comparable size.
"""
import numpy as np

from dp_core import njit, prange
from dp_propagator import propagator_3x3

_EPS = 2.220446049250313e-16


# =============================================================================
# EIGENSYSTEMS OF M + sigma S AT ANY sigma
# =============================================================================

def texture_basis(S):
    """Eigenbasis of the Hermitian texture S, with any degenerate eigenvalues made exactly
    equal and shifted to zero (a multiple of the identity changes no flavor evolution).

    Arguments: S -- 3x3 Hermitian. Returns (Q, s): Q unitary (columns the eigenvectors), s the
    shifted eigenvalues, ascending. Eigenvalues within 64 eps of the largest |eigenvalue| of
    each other count as degenerate: u u^dag has two exact zeros that eigh returns as rounding.
    """
    S = np.asarray(S, dtype=np.complex128)
    S = 0.5 * (S + np.conjugate(S).T)
    lam, Q = np.linalg.eigh(S)
    scale = max(float(np.max(np.abs(lam))), 1e-300)
    tol = 64.0 * _EPS * scale
    s = lam.astype(np.float64).copy()
    if s[2] - s[0] <= tol:
        s[:] = 0.0                                   # S ~ identity: no flavor structure
    elif s[1] - s[0] <= tol:
        c = 0.5 * (s[0] + s[1])
        s -= c
        s[0] = s[1] = 0.0
    elif s[2] - s[1] <= tol:
        c = 0.5 * (s[1] + s[2])
        s -= c
        s[1] = s[2] = 0.0
    return np.ascontiguousarray(Q), np.ascontiguousarray(s)


@njit(cache=True)
def _jacobi_herm3(A, V, floor):
    """Diagonalizes the Hermitian 3x3 A in place by cyclic complex Jacobi rotations,
    accumulating them into V (V <- V G). A pair (p, q) is rotated unless
    |A_pq| <= 4 eps max(sqrt(|A_pp A_qq|), floor) -- the relative-accuracy criterion of
    Demmel and Veselic, with an absolute floor at the scale below which nothing matters
    (callers pass the vacuum term's size). Both triangles of A are kept."""
    for sweep in range(60):
        rotated = False
        for pair in range(3):
            if pair == 0:
                p, q, r = 0, 1, 2
            elif pair == 1:
                p, q, r = 0, 2, 1
            else:
                p, q, r = 1, 2, 0
            apq = A[p, q]
            mag = np.sqrt(apq.real * apq.real + apq.imag * apq.imag)
            if mag == 0.0:
                continue
            app = A[p, p].real
            aqq = A[q, q].real
            if mag <= 4.0 * _EPS * max(np.sqrt(abs(app * aqq)), floor):
                A[p, q] = 0.0
                A[q, p] = 0.0
                continue
            rotated = True
            ph = apq / mag                               # e^{i phi}
            ph_c = np.conjugate(ph)
            theta = (aqq - app) / (2.0 * mag)
            if theta >= 0.0:
                t = 1.0 / (theta + np.sqrt(theta * theta + 1.0))
            else:
                t = -1.0 / (-theta + np.sqrt(theta * theta + 1.0))
            c = 1.0 / np.sqrt(t * t + 1.0)
            sn = t * c
            # G on (p, q): G_pp = c, G_pq = sn, G_qp = -sn e^{-i phi}, G_qq = c e^{-i phi}
            arp = A[r, p]
            arq = A[r, q]
            nrp = c * arp - sn * ph_c * arq
            nrq = sn * arp + c * ph_c * arq
            A[r, p] = nrp
            A[p, r] = np.conjugate(nrp)
            A[r, q] = nrq
            A[q, r] = np.conjugate(nrq)
            A[p, p] = app - t * mag
            A[q, q] = aqq + t * mag
            A[p, q] = 0.0
            A[q, p] = 0.0
            for k in range(3):
                vkp = V[k, p]
                vkq = V[k, q]
                V[k, p] = c * vkp - sn * ph_c * vkq
                V[k, q] = sn * vkp + c * ph_c * vkq
        if not rotated:
            break


@njit(cache=True)
def eigh_graded(Mp, sigma, s, floor, vals, vecs):
    """Eigensystem of Mp + sigma diag(s), all in the texture basis (Mp = Q^dag M Q), with
    eigenvalues ascending into vals and eigenvectors (columns, in the texture basis) into
    vecs. Accurate to eps relative to each eigenvalue's own scale for any sigma (module
    notes). floor: the size of Mp (the scale of the vacuum term)."""
    A = np.empty((3, 3), dtype=np.complex128)
    V = np.zeros((3, 3), dtype=np.complex128)
    for a in range(3):
        V[a, a] = 1.0
        for b in range(3):
            A[a, b] = Mp[a, b]
    for a in range(3):
        A[a, a] = complex(Mp[a, a].real + sigma * s[a], 0.0)
    _jacobi_herm3(A, V, floor)
    d0 = A[0, 0].real
    d1 = A[1, 1].real
    d2 = A[2, 2].real
    # ascending order
    i0, i1, i2 = 0, 1, 2
    if d0 > d1:
        i0, i1 = i1, i0
    dd = (d0, d1, d2)
    if dd[i1] > d2:
        i1, i2 = i2, i1
    if dd[i0] > dd[i1]:
        i0, i1 = i1, i0
    order = (i0, i1, i2)
    for k in range(3):
        j = order[k]
        vals[k] = dd[j]
        for a in range(3):
            vecs[a, k] = V[a, j]


@njit(cache=True)
def _inv3(B, out):
    """Inverse of a complex 3x3 by the adjugate."""
    c00 = B[1, 1] * B[2, 2] - B[1, 2] * B[2, 1]
    c01 = B[1, 2] * B[2, 0] - B[1, 0] * B[2, 2]
    c02 = B[1, 0] * B[2, 1] - B[1, 1] * B[2, 0]
    det = B[0, 0] * c00 + B[0, 1] * c01 + B[0, 2] * c02
    inv_det = 1.0 / det
    out[0, 0] = c00 * inv_det
    out[1, 0] = c01 * inv_det
    out[2, 0] = c02 * inv_det
    out[0, 1] = (B[0, 2] * B[2, 1] - B[0, 1] * B[2, 2]) * inv_det
    out[1, 1] = (B[0, 0] * B[2, 2] - B[0, 2] * B[2, 0]) * inv_det
    out[2, 1] = (B[0, 1] * B[2, 0] - B[0, 0] * B[2, 1]) * inv_det
    out[0, 2] = (B[0, 1] * B[1, 2] - B[0, 2] * B[1, 1]) * inv_det
    out[1, 2] = (B[0, 2] * B[1, 0] - B[0, 0] * B[1, 2]) * inv_det
    out[2, 2] = (B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]) * inv_det


@njit(cache=True)
def log_unitary3(X, K):
    """K = log X for a 3x3 unitary X with no eigenvalue at -1 (rotation angles below pi),
    anti-Hermitian to rounding: the Cayley transform C = (X - I)(X + I)^-1 is anti-Hermitian
    with eigenvalues i tan(theta/2); diagonalizing -iC gives theta = 2 atan(tau) on the same
    eigenvectors. Accurate to eps absolute for X near the identity."""
    B = np.empty((3, 3), dtype=np.complex128)
    Bi = np.empty((3, 3), dtype=np.complex128)
    Xm = np.empty((3, 3), dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            B[a, b] = X[a, b]
            Xm[a, b] = X[a, b]
        B[a, a] += 1.0
        Xm[a, a] -= 1.0
    _inv3(B, Bi)
    H = np.empty((3, 3), dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                acc += Xm[a, k] * Bi[k, b]
            H[a, b] = -1j * acc
    # Hermitize -iC
    for a in range(3):
        H[a, a] = complex(H[a, a].real, 0.0)
        for b in range(a + 1, 3):
            hab = 0.5 * (H[a, b] + np.conjugate(H[b, a]))
            H[a, b] = hab
            H[b, a] = np.conjugate(hab)
    V = np.zeros((3, 3), dtype=np.complex128)
    for a in range(3):
        V[a, a] = 1.0
    _jacobi_herm3(H, V, 1e-300)
    th0 = 2.0 * np.arctan(H[0, 0].real)
    th1 = 2.0 * np.arctan(H[1, 1].real)
    th2 = 2.0 * np.arctan(H[2, 2].real)
    for a in range(3):
        for b in range(3):
            K[a, b] = 1j * (th0 * V[a, 0] * np.conjugate(V[b, 0])
                            + th1 * V[a, 1] * np.conjugate(V[b, 1])
                            + th2 * V[a, 2] * np.conjugate(V[b, 2]))
    # exact anti-Hermiticity
    for a in range(3):
        K[a, a] = complex(0.0, K[a, a].imag)
        for b in range(a + 1, 3):
            kab = 0.5 * (K[a, b] - np.conjugate(K[b, a]))
            K[a, b] = kab
            K[b, a] = -np.conjugate(kab)


@njit(cache=True)
def frame_at(Q, Mp, s, floor, sigma, vals, W):
    """Eigenvalues (ascending, M units) and eigenvectors (columns, flavor basis) of
    M + sigma S, for M = Q Mp Q^dag and S = Q diag(s) Q^dag (up to the identity shift)."""
    V = np.empty((3, 3), dtype=np.complex128)
    eigh_graded(Mp, sigma, s, floor, vals, V)
    for a in range(3):
        for b in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                acc += Q[a, k] * V[k, b]
            W[a, b] = acc


@njit(cache=True)
def align_columns(W_ref, W):
    """Rephases each column of W so that its overlap with the same column of W_ref is real
    and non-negative (parallel transport to first order)."""
    for k in range(3):
        ov = 0.0 + 0.0j
        for a in range(3):
            ov += np.conjugate(W_ref[a, k]) * W[a, k]
        mag = abs(ov)
        if mag > 0.0:
            ph = np.conjugate(ov) / mag
            for a in range(3):
                W[a, k] = W[a, k] * ph


@njit(cache=True)
def _overlap(Wa, Wb, X):
    """X = Wa^dag Wb."""
    for a in range(3):
        for b in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                acc += np.conjugate(Wa[k, a]) * Wb[k, b]
            X[a, b] = acc


@njit(cache=True)
def _frames_on_nodes(Q, Mp, s, floor, sig):
    """Eigenvalues and (unaligned) frames at every node of sig."""
    n = len(sig)
    nu = np.empty((n, 3))
    W = np.empty((n, 3, 3), dtype=np.complex128)
    vals = np.empty(3)
    Wk = np.empty((3, 3), dtype=np.complex128)
    for i in range(n):
        frame_at(Q, Mp, s, floor, sig[i], vals, Wk)
        for k in range(3):
            nu[i, k] = vals[k]
        W[i] = Wk
    return nu, W


@njit(cache=True)
def _rotation_proxy(Wa, Wb):
    """Largest off-diagonal |(Wa^dag Wb)_jk|: the angle the eigenbasis turns between two
    nodes, to first order, independent of the columns' phases."""
    X = np.empty((3, 3), dtype=np.complex128)
    _overlap(Wa, Wb, X)
    best = 0.0
    for a in range(3):
        for b in range(3):
            if a != b:
                best = max(best, abs(X[a, b]))
    return best


def texture_scales(M_rot, S):
    """(Q, s, Mp, floor, sig_lo, sig_hi) for build_frame_table: the texture basis, M in it,
    the vacuum scale, and the sigma range outside which the eigenbasis is constant to 1e-9
    (the vacuum one below, the texture's own above)."""
    Q, s = texture_basis(S)
    Mp = np.conjugate(Q).T @ np.asarray(M_rot, dtype=np.complex128) @ Q
    Mp = np.ascontiguousarray(0.5 * (Mp + np.conjugate(Mp).T))
    floor = float(np.max(np.abs(Mp)))
    ev = np.linalg.eigvalsh(np.asarray(M_rot, dtype=np.complex128))
    gap_min = float(np.min(np.diff(np.sort(ev))))
    ds = np.abs(np.subtract.outer(s, s))
    ds = ds[ds > 0.0]
    if ds.size == 0:
        return Q, s, Mp, floor, 0.0, 0.0
    sig_lo = 1e-9 * gap_min / float(np.max(np.abs(s)))
    sig_hi = 1e9 * floor / float(np.min(ds))
    return Q, s, Mp, floor, sig_lo, sig_hi


def build_frame_table(M_rot, S, dtheta=0.01, per_decade=8, max_nodes=200000):
    """The sigma-table of eigenbases of M_rot + sigma S (module notes), sigma >= 0.

    Arguments: M_rot (3x3, eV^2); S (3x3 texture, carrying the sign of the potential);
    dtheta -- the largest rotation of the eigenbasis between consecutive nodes; per_decade --
    the initial geometric spacing. Nodes are bisected (geometrically) until the rotation
    proxy is below dtheta, or the interval is narrower than 1e-9 in ln sigma (an exact level
    crossing, which the integrator then crosses suddenly, as it should).

    Returns a dict: sig (N,), nu (N, 3) eigenvalues ascending (eV^2), W (N, 3, 3) frames in
    the flavor basis with phases aligned along the table, K (N-1, 3, 3) = log(W_n^dag W_n+1),
    and Q, s, Mp, floor (texture_scales). With a texture proportional to the identity the
    table is the single vacuum node.
    """
    Q, s, Mp, floor, sig_lo, sig_hi = texture_scales(M_rot, S)
    if sig_hi <= 0.0:
        sig = np.array([0.0])
    else:
        n0 = int(np.ceil(per_decade * np.log10(sig_hi / sig_lo))) + 1
        sig = np.geomspace(sig_lo, sig_hi, n0)
    nu, W = _frames_on_nodes(Q, Mp, s, floor, sig)
    for _ in range(64):
        if len(sig) < 2:
            break
        bad = np.array([_rotation_proxy(W[i], W[i + 1]) > dtheta and sig[i + 1] / sig[i] > 1.0 + 1e-9
                        for i in range(len(sig) - 1)])
        if not bad.any():
            break
        new = np.sqrt(sig[:-1][bad] * sig[1:][bad])
        nu_new, W_new = _frames_on_nodes(Q, Mp, s, floor, new)
        sig = np.concatenate([sig, new])
        nu = np.concatenate([nu, nu_new])
        W = np.concatenate([W, W_new])
        order = np.argsort(sig)
        sig, nu, W = sig[order], nu[order], W[order]
        if len(sig) > max_nodes:
            raise RuntimeError("build_frame_table: node limit reached")
    W = np.ascontiguousarray(W)
    for i in range(1, len(sig)):
        align_columns(W[i - 1], W[i])
    K = np.zeros((max(len(sig) - 1, 0), 3, 3), dtype=np.complex128)
    X = np.empty((3, 3), dtype=np.complex128)
    for i in range(len(sig) - 1):
        _overlap(W[i], W[i + 1], X)
        log_unitary3(X, K[i])
    return dict(sig=np.ascontiguousarray(sig), nu=np.ascontiguousarray(nu), W=W,
                K=np.ascontiguousarray(K), Q=Q, s=s, Mp=Mp, floor=floor)


# =============================================================================
# THE PATH INTEGRATOR
# =============================================================================
# A path ends at the point of interest (the Earth, or a sphere around it) at distance D from a
# point source of Yukawa range 1/m. Measured backwards along the path by t >= 0, the distance
# to the source is r(t) = sqrt((t - D ca)^2 + D^2 (1 - ca^2)), ca the cosine of the angle
# between the backward direction and the source, and the source's density term is
#     n(t) = C exp(-m (r - D)) D / r,        C = n at the end point.
# With ca > 0 the path passes its closest approach at t* = D ca: the potential rises to it
# and falls after it, and the march runs the two monotonic segments in turn. The Earth's
# exterior seen radially is the case ca = -1.

@njit(cache=True)
def _path_r(t, D, ca):
    dt = t - D * ca
    return np.sqrt(dt * dt + D * D * (1.0 - ca * ca))


@njit(cache=True)
def _path_n(t, C, m, D, ca):
    r = _path_r(t, D, ca)
    return C * np.exp(-m * (r - D)) * D / r


@njit(cache=True)
def _r_of_n(n_target, C, m, D, r_lo, r_hi):
    """The distance r in [r_lo, r_hi] at which C exp(-m (r - D)) D / r = n_target (the
    profile falls monotonically with r): safeguarded Newton on ln r + m (r - D) = ln(C D /
    n_target)."""
    L = np.log(C * D / n_target)
    a, b = r_lo, r_hi
    r = min(max(np.exp(L), a), b)
    if m * r > 1.0:
        r = min(max(L / m, a), b)
    for it in range(200):
        f = np.log(r) + m * (r - D) - L
        if f > 0.0:
            b = r
        else:
            a = r
        rn = r - f / (1.0 / r + m)
        if rn <= a or rn >= b:
            rn = 0.5 * (a + b) if b / a < 4.0 else np.sqrt(a * b)
        if abs(rn - r) <= 1e-15 * r:
            return rn
        r = rn
    return r


@njit(cache=True)
def _t_of_r(r, D, ca, far):
    """Path parameter at distance r on the far (t >= D ca) or near branch."""
    q = r * r - D * D * (1.0 - ca * ca)
    w = np.sqrt(max(q, 0.0))
    return D * ca + w if far else D * ca - w


@njit(cache=True)
def _column_minus(t_lo, t_hi, n_ref, C, m, D, ca, glx, glw):
    """int_{t_lo}^{t_hi} (n(t) - n_ref) dt, on sub-intervals geometric in r (within a step r
    is monotonic in t) and no wider than 0.5/m, with Gauss-Legendre on each."""
    r_lo = _path_r(t_lo, D, ca)
    r_hi = _path_r(t_hi, D, ca)
    rr = max(r_lo, r_hi) / min(r_lo, r_hi)
    n_sub = 1 + int(np.log(rr) / 0.1) + int(m * abs(r_hi - r_lo) / 0.5)
    if n_sub > 4000:
        n_sub = 4000
    far = t_lo >= D * ca      # the step lies on one branch: both ends on the same side of t*
    total = 0.0
    ta = t_lo
    for j in range(n_sub):
        if j == n_sub - 1:
            tb = t_hi
        else:
            f = (j + 1.0) / n_sub
            rb = r_lo * (r_hi / r_lo) ** f
            tb = _t_of_r(rb, D, ca, far)
        mid = 0.5 * (ta + tb)
        half = 0.5 * (tb - ta)
        acc = 0.0
        for q in range(len(glx)):
            acc += glw[q] * (_path_n(mid + half * glx[q], C, m, D, ca) - n_ref)
        total += acc * half
        ta = tb
    return total


@njit(cache=True)
def _E1i(x):
    """(e^{ix} - 1) / (ix) for real x."""
    if abs(x) < 1e-3:
        x2 = x * x
        return complex(1.0 - x2 / 6.0 + x2 * x2 / 120.0, x / 2.0 - x * x2 / 24.0)
    sh = np.sin(0.5 * x)
    return complex(np.sin(x) / x, 2.0 * sh * sh / x)


@njit(cache=True)
def _transit_step(rho, K, nu_a, nu_b, sig_a, sig_b, h, J, twoE, Om, Uo, T):
    """One step of the adiabatic-frame integrator (module notes), in place on the density
    matrix rho expressed in the step's starting eigenbasis; on return it is expressed in the
    final one. K: the eigenbasis' rotation over the step; nu_a, nu_b: eigenvalues at the ends
    (eV^2); sig_a, sig_b: sigma at the ends; h: step length (eV^-1); J = int (sigma - sig_a)
    dt over the step (eV); twoE = 2E (eV)."""
    Phi = np.empty(3)
    dsig = sig_b - sig_a
    for k in range(3):
        slope = 0.0
        if abs(dsig) > 1e-13 * max(abs(sig_a), abs(sig_b)):
            slope = (nu_b[k] - nu_a[k]) / dsig
        Phi[k] = (nu_a[k] * h + slope * J) / twoE
    # Omega = -K o E1(i dPhi) (elementwise); exp(Omega) = exp(-i (i Omega))
    for a in range(3):
        for b in range(3):
            if a == b:
                Om[a, b] = 1j * (-K[a, a])
            else:
                Om[a, b] = 1j * (-K[a, b] * _E1i(Phi[a] - Phi[b]))
    # Hermitize i Omega (rounding)
    for a in range(3):
        Om[a, a] = complex(Om[a, a].real, 0.0)
        for b in range(a + 1, 3):
            v = 0.5 * (Om[a, b] + np.conjugate(Om[b, a]))
            Om[a, b] = v
            Om[b, a] = np.conjugate(v)
    propagator_3x3(Om, 1.0, Uo)
    # rho <- Uo rho Uo^dag, then the dynamic phases
    for a in range(3):
        for b in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                acc += Uo[a, k] * rho[k, b]
            T[a, b] = acc
    for a in range(3):
        for b in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                acc += T[a, k] * np.conjugate(Uo[b, k])
            rho[a, b] = acc
    for a in range(3):
        rho[a, a] = complex(rho[a, a].real, 0.0)
        for b in range(a + 1, 3):
            dph = Phi[a] - Phi[b]
            v = rho[a, b] * complex(np.cos(dph), -np.sin(dph))
            rho[a, b] = v
            rho[b, a] = np.conjugate(v)


@njit(cache=True)
def _first_node_above(sig_nodes, x):
    """Index of the first node > x (len if none)."""
    lo, hi = 0, len(sig_nodes)
    while lo < hi:
        mid = (lo + hi) // 2
        if sig_nodes[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _sandwich(W, rho, out, T, adjoint_first):
    """out = W^dag rho W (adjoint_first) or W rho W^dag."""
    for a in range(3):
        for b in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                if adjoint_first:
                    acc += np.conjugate(W[k, a]) * rho[k, b]
                else:
                    acc += W[a, k] * rho[k, b]
            T[a, b] = acc
    for a in range(3):
        for b in range(3):
            acc = 0.0 + 0.0j
            for k in range(3):
                if adjoint_first:
                    acc += T[a, k] * W[k, b]
                else:
                    acc += T[a, k] * np.conjugate(W[b, k])
            out[a, b] = acc


@njit(cache=True)
def march_path(E, kappa, n_bg, C, m, D, ca, t_max, rho_in, sig_nodes, nu_nodes, W_nodes,
               K_nodes, Q, Mp, s, floor, glx, glw, rho_out):
    """Evolves the flavor density matrix rho_in from t = t_max to t = 0 along one path
    (section notes above) at energy E, with sigma(t) = kappa (n_bg + n(t)), kappa = 2E |Coef|
    (eV^-1 per unit density, so that sigma is in eV^2). The table (sig_nodes, nu_nodes,
    W_nodes, K_nodes) and (Q, Mp, s, floor) come from build_frame_table. Writes the flavor
    density matrix at t = 0 into rho_out."""
    twoE = 2.0 * E
    rho = np.empty((3, 3), dtype=np.complex128)
    T = np.empty((3, 3), dtype=np.complex128)
    Om = np.empty((3, 3), dtype=np.complex128)
    Uo = np.empty((3, 3), dtype=np.complex128)
    X = np.empty((3, 3), dtype=np.complex128)
    Kp = np.empty((3, 3), dtype=np.complex128)
    Wc = np.empty((3, 3), dtype=np.complex128)
    Wn = np.empty((3, 3), dtype=np.complex128)
    nu_c = np.empty(3)
    nu_n = np.empty(3)
    N = len(sig_nodes)

    t_c = t_max
    n_c = _path_n(t_c, C, m, D, ca)
    sig_c = kappa * (n_bg + n_c)
    frame_at(Q, Mp, s, floor, sig_c, nu_c, Wc)
    # align the starting frame with the table where it starts
    j0 = _first_node_above(sig_nodes, sig_c)
    if N > 0:
        align_columns(W_nodes[min(j0, N - 1)], Wc)
    _sandwich(Wc, rho_in, rho, T, True)
    c_node = -1                 # index of the table node the current frame is, or -1

    n_seg = 1
    t_turn = D * ca
    if ca > 0.0 and t_turn < t_max:
        n_seg = 2
    for seg in range(n_seg):
        if n_seg == 2 and seg == 0:
            t_end = t_turn
            far = True
        else:
            t_end = 0.0
            far = n_seg == 1
        n_end = _path_n(t_end, C, m, D, ca)
        sig_end = kappa * (n_bg + n_end)
        rising = sig_end >= sig_c
        r_c = _path_r(t_c, D, ca)
        r_end = _path_r(t_end, D, ca)
        # table nodes strictly between sig_c and sig_end, in travel order
        if rising:
            j = _first_node_above(sig_nodes, sig_c)
            if c_node >= 0:
                j = c_node + 1
        else:
            j = _first_node_above(sig_nodes, sig_c) - 1
            if j >= 0:
                if sig_nodes[j] >= sig_c:
                    j -= 1
            if c_node >= 0:
                j = c_node - 1
        while True:
            if rising:
                if j >= N or sig_nodes[j] >= sig_end:
                    break
            else:
                if j < 0 or sig_nodes[j] <= sig_end:
                    break
            sig_j = sig_nodes[j]
            n_target = sig_j / kappa - n_bg
            r_j = _r_of_n(n_target, C, m, D, min(r_c, r_end), max(r_c, r_end))
            t_j = _t_of_r(r_j, D, ca, far)
            if t_j > t_c:
                t_j = t_c
            if t_j < t_end:
                t_j = t_end
            for k in range(3):
                nu_n[k] = nu_nodes[j, k]
            if c_node >= 0:
                if rising:
                    for a in range(3):
                        for b in range(3):
                            Kp[a, b] = K_nodes[c_node, a, b]
                else:
                    for a in range(3):
                        for b in range(3):
                            Kp[a, b] = -K_nodes[j, a, b]
            else:
                _overlap(Wc, W_nodes[j], X)
                log_unitary3(X, Kp)
            h = t_c - t_j
            J = kappa * _column_minus(t_j, t_c, n_c, C, m, D, ca, glx, glw)
            _transit_step(rho, Kp, nu_c, nu_n, sig_c, sig_j, h, J, twoE, Om, Uo, T)
            t_c = t_j
            n_c = _path_n(t_c, C, m, D, ca)
            sig_c = sig_j
            for k in range(3):
                nu_c[k] = nu_n[k]
            for a in range(3):
                for b in range(3):
                    Wc[a, b] = W_nodes[j, a, b]
            c_node = j
            r_c = _path_r(t_c, D, ca)
            j = j + 1 if rising else j - 1
        # partial step to the segment's end
        frame_at(Q, Mp, s, floor, sig_end, nu_n, Wn)
        align_columns(Wc, Wn)
        _overlap(Wc, Wn, X)
        log_unitary3(X, Kp)
        h = t_c - t_end
        J = kappa * _column_minus(t_end, t_c, n_c, C, m, D, ca, glx, glw) if h > 0.0 else 0.0
        _transit_step(rho, Kp, nu_c, nu_n, sig_c, sig_end, h, J, twoE, Om, Uo, T)
        t_c = t_end
        n_c = n_end
        sig_c = sig_end
        for k in range(3):
            nu_c[k] = nu_n[k]
        for a in range(3):
            for b in range(3):
                Wc[a, b] = Wn[a, b]
        c_node = -1
    _sandwich(Wc, rho, rho_out, T, False)


# =============================================================================
# THE GALAXY: YUKAWA-SMOOTHED LOCAL DENSITY AND THE ADIABATIC MAP
# =============================================================================
# Above this mediator mass (a range of 0.2 m) the smoothing changes the local density by
# ~1e-14: the contact value is used, bit for bit the pre-2026-10-13 one. The correction falls
# only as 1/m, not 1/m^2, because the model's exponential discs have a cusp at z = 0, where the
# Sun sits: nbar(r) = n_loc (1 - 1.2 r/kpc) near the Sun.
_MW_CONTACT_MIN_M_EV = 1e-12
_MW_TABLE = None


def _mw_table():
    """(r in eV^-1, nbar in eV^3) from src/data/mw_sky_density.npz (tools/mw_sky_density.py)."""
    global _MW_TABLE
    if _MW_TABLE is None:
        import os
        from dp_core import m_to_eV_inv, cm_inv3_to_eV3
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mw_sky_density.npz")
        d = np.load(path)
        r = d["r_kpc"] * 3.08567758e19 * m_to_eV_inv
        _MW_TABLE = (np.ascontiguousarray(r), np.ascontiguousarray(d["nbar_cm3"] * cm_inv3_to_eV3))
    return _MW_TABLE


def galactic_effective_density(m_eV: float, n_n0_eV3: float, R_H_eV_inv: float) -> float:
    """The Yukawa-smoothed neutron density (eV^3) at the Sun from the Galaxy and the cosmic
    mean beyond it,
        n_eff = int_0^inf m^2 r exp(-m r) max(nbar(r), n_n0) dr,
    with nbar the sky-averaged Galactic density (the r^2 of the volume element cancels the
    1/r of the kernel), the cosmic mean taken out to the Hubble radius R_H. The potential is
    Coef n_eff with Coef ~ 1/m^2: n_eff is the local density for short ranges and m^2 times the
    enclosed neutrons over their distance for long ones. Above _MW_CONTACT_MIN_M_EV the local
    density itself (dp_spatial.calc_MW_neutron_density at the Sun)."""
    from dp_spatial import calc_MW_neutron_density
    from dp_core import cm_inv3_to_eV3
    n_loc = calc_MW_neutron_density(0.0, 0.0, 0.0) * cm_inv3_to_eV3
    if m_eV >= _MW_CONTACT_MIN_M_EV:
        return n_loc
    r, nbar = _mw_table()
    n = np.maximum(nbar, n_n0_eV3)
    x = m_eV * r
    f = x * x * np.exp(-x) * n                     # m^2 r^2 e^{-mr} n, per unit ln r
    lnr = np.log(r)
    body = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(lnr)))
    x0, x1 = x[0], x[-1]
    # below the table, nbar is linear from n_loc (the cusp, see above):
    # int_0^r0 m^2 r e^{-mr} (n_loc + beta r) dr
    beta = (nbar[0] - n_loc) / r[0]
    e0 = np.exp(-x0)
    head = (n_loc * (-np.expm1(-x0) - x0 * e0)
            + beta / m_eV * (2.0 * (-np.expm1(-x0)) - (2.0 * x0 + x0 * x0) * e0))
    xH = m_eV * R_H_eV_inv
    tail = n_n0_eV3 * ((1.0 + x1) * np.exp(-x1) - (1.0 + xH) * np.exp(-xH)) if xH > x1 else 0.0
    return head + body + tail


@njit(cache=True, parallel=True)
def galactic_map(E_bins, F, coef2, n_bg, Q, Mp, s, floor, rho_out):
    """The adiabatic Galactic transit: the populations F (N_E, 3) of the cosmological arrival's
    eigenstates (ascending eigenvalues) become the same populations of the local eigenstates,
    sigma = E coef2 n_bg (coef2 = 2 |Coef|): rho = W diag(F) W^dag, written to rho_out."""
    for i in prange(len(E_bins)):
        vals = np.empty(3)
        W = np.empty((3, 3), dtype=np.complex128)
        frame_at(Q, Mp, s, floor, E_bins[i] * coef2 * n_bg, vals, W)
        for a in range(3):
            for b in range(3):
                acc = 0.0 + 0.0j
                for k in range(3):
                    acc += W[a, k] * F[i, k] * np.conjugate(W[b, k])
                rho_out[i, a, b] = acc


@njit(cache=True, parallel=True)
def flavor_populations(E_bins, F_flavor, coef2, n_bg, Q, Mp, s, floor, F_out):
    """Neutrinos emitted inside the Galaxy at the local potential (docs/dev/DIRECTIONS_PLAN.md,
    Section 3.6, emission 'local'): flavor fractions F_flavor (N_E, 3) decohere over the kpc to the
    Sun into the local eigenstates (ascending eigenvalues, sigma = E coef2 n_bg), with populations
    F_out[i, k] = sum_a F_flavor[i, a] |W[a, k]|^2 -- the input galactic_map expects."""
    for i in prange(len(E_bins)):
        vals = np.empty(3)
        W = np.empty((3, 3), dtype=np.complex128)
        frame_at(Q, Mp, s, floor, E_bins[i] * coef2 * n_bg, vals, W)
        for k in range(3):
            acc = 0.0
            for a in range(3):
                acc += F_flavor[i, a] * (W[a, k].real ** 2 + W[a, k].imag ** 2)
            F_out[i, k] = acc


@njit(cache=True, parallel=True)
def nearfield_transit(E_bins, rho_in, active, coef2, n_bg, C, m, D, ca, w, t_max, sig_nodes,
                      nu_nodes, W_nodes, K_nodes, Q, Mp, s, floor, glx, glw, rho_out):
    """March every active energy bin along the paths ca[p] (weights w[p], starts t_max[p]) of
    one point source (march_path), averaging the resulting density matrices; inactive bins
    are copied through. coef2 = 2 |Coef|."""
    for i in prange(len(E_bins)):
        if not active[i]:
            for a in range(3):
                for b in range(3):
                    rho_out[i, a, b] = rho_in[i, a, b]
            continue
        acc = np.zeros((3, 3), dtype=np.complex128)
        tmp = np.empty((3, 3), dtype=np.complex128)
        rin = np.empty((3, 3), dtype=np.complex128)
        for a in range(3):
            for b in range(3):
                rin[a, b] = rho_in[i, a, b]
        for p in range(len(ca)):
            march_path(E_bins[i], E_bins[i] * coef2, n_bg, C, m, D, ca[p], t_max[p], rin,
                       sig_nodes, nu_nodes, W_nodes, K_nodes, Q, Mp, s, floor, glx, glw, tmp)
            for a in range(3):
                for b in range(3):
                    acc[a, b] += w[p] * tmp[a, b]
        for a in range(3):
            for b in range(3):
                rho_out[i, a, b] = acc[a, b]


@njit(cache=True)
def _proxy_between(Q, Mp, s, floor, sig_a, sig_b):
    va = np.empty(3)
    vb = np.empty(3)
    Wa = np.empty((3, 3), dtype=np.complex128)
    Wb = np.empty((3, 3), dtype=np.complex128)
    frame_at(Q, Mp, s, floor, sig_a, va, Wa)
    frame_at(Q, Mp, s, floor, sig_b, vb, Wb)
    return _rotation_proxy(Wa, Wb)


_TRANSIT_TABLES = {}
# Paths start where the source's density falls to this fraction of the larger of the
# background and the density whose potential equals the smallest vacuum gap: the adiabatic
# hand-over there is exact to that fraction.
_TRANSIT_START_FRACTION = 1e-8
# Energy bins whose eigenbasis the source turns by less than this are left unchanged.
_TRANSIT_ACTIVE_MIN_ROTATION = 1e-7
_GLX8, _GLW8 = np.polynomial.legendre.leggauss(8)
_GLX8 = np.ascontiguousarray(_GLX8)
_GLW8 = np.ascontiguousarray(_GLW8)


def frame_table(M_rot, S_signed, dtheta):
    """build_frame_table, cached on (M_rot, S_signed, dtheta): within an analysis the mixing and
    the texture are fixed while the couplings and mass vary, so one table serves every call."""
    key = (np.asarray(M_rot, dtype=np.complex128).tobytes(), np.asarray(S_signed, dtype=np.complex128).tobytes(),
           float(dtheta))
    tab = _TRANSIT_TABLES.get(key)
    if tab is None:
        if len(_TRANSIT_TABLES) > 16:
            _TRANSIT_TABLES.clear()
        tab = _TRANSIT_TABLES[key] = build_frame_table(M_rot, S_signed, dtheta=dtheta)
    return tab


def source_transit(E_bins, rho_in, tab, coef_abs, n_bg, C, m, D, ca, w, force_active=False):
    """The coherent transit through one point source's potential (module notes), for every
    energy bin it affects: paths at cosines ca (weights w, summing to 1) to the source, ending
    at distance D from it, where its density is C (eV^3); background density n_bg; mediator
    mass m (eV, the Yukawa range 1/m); |Coef| (eV^-2). A bin is affected when the source turns
    its eigenbasis by more than _TRANSIT_ACTIVE_MIN_ROTATION, or always with force_active (for
    a state that may carry coherences, on which the source's phase acts even then).
    Returns (rho_out, n_active)."""
    E_bins = np.ascontiguousarray(E_bins, dtype=np.float64)
    rho_out = np.array(rho_in, dtype=np.complex128, copy=True)
    if C <= 0.0 or coef_abs <= 0.0 or len(tab["sig"]) < 2:
        return rho_out, 0
    coef2 = 2.0 * coef_abs
    Q, Mp, s, floor = tab["Q"], tab["Mp"], tab["s"], tab["floor"]
    active = np.zeros(len(E_bins), dtype=np.bool_)
    for i, E in enumerate(E_bins):
        active[i] = force_active or _proxy_between(Q, Mp, s, floor, E * coef2 * n_bg, E * coef2 * (n_bg + C)) \
            > _TRANSIT_ACTIVE_MIN_ROTATION
    if not active.any():
        return rho_out, 0
    # start: the source's density down to the fraction above of max(n_bg, n_gap(E_max))
    ev = np.sort(np.linalg.eigvalsh(tab["Mp"]))
    gap_min = float(np.min(np.diff(ev)))
    n_gap = gap_min / (coef2 * float(E_bins[-1]) * max(float(np.max(np.abs(s))), 1e-300))
    n_thr = _TRANSIT_START_FRACTION * max(n_bg, n_gap)
    if n_thr >= C:
        r_far = D * 1.0000001
    else:
        r_hi = D
        while C * np.exp(-m * (r_hi - D)) * D / r_hi > n_thr:
            r_hi *= 2.0
        r_far = _r_of_n(n_thr, C, m, D, D, r_hi)
    ca = np.ascontiguousarray(ca, dtype=np.float64)
    t_max = np.array([D * c + np.sqrt(max(r_far * r_far - D * D * (1.0 - c * c), 0.0)) for c in ca])
    nearfield_transit(E_bins, np.ascontiguousarray(rho_in, dtype=np.complex128), active, coef2, n_bg,
                      float(C), float(m), float(D), ca, np.ascontiguousarray(w, dtype=np.float64), t_max,
                      tab["sig"], tab["nu"], tab["W"], tab["K"], Q, Mp, s, floor, _GLX8, _GLW8, rho_out)
    return rho_out, int(active.sum())


def sun_paths(n_paths: int):
    """Gauss-Legendre cosines of the angle between the arrival direction and the Sun, and
    their weights (summing to 1): an isotropic sky is uniform in that cosine."""
    x, wgt = np.polynomial.legendre.leggauss(int(n_paths))
    return np.ascontiguousarray(x), np.ascontiguousarray(0.5 * wgt)
