"""
dp_propagator.py

The 3x3 propagator (exactly unitary spectral sum over a deflated eigenbasis) and the
closed-form eigenvector routine -- the most delicate numerics in the repository; read
the docstrings before touching anything.
"""
import numpy as np

from dp_core import njit

# A pair half-gap below this many eps * (|q| + p) is not resolved by a double-precision
# Hamiltonian: rounding when H is formed already splits an exact degeneracy by up to
# 3.1 eps * (|q| + p) (measured on 1500 random rotated degenerate spectra), and the
# half-gap computed below is accurate to 0.5 eps * spread (both against 60-digit
# arithmetic). Such pairs are propagated as exactly degenerate.
PAIR_FLOOR = 16.0
_EPS = 2.220446049250313e-16


@njit(cache=True)
def propagator_3x3(H: np.ndarray, t: float, out: np.ndarray):
    """Writes exp(-i H t) for a Hermitian 3x3 H into out, as the spectral sum
    sum_k e^{-i lam_k t} v_k v_k^dag over an orthonormal eigenbasis built so that it stays
    orthonormal to rounding at any eigenvalue gap. U is therefore unitary to ~1e-15 at any
    phase (spread * t up to 1e30 rad and beyond); errors in the eigenvalues only shift
    phases. H is read as the Hermitian matrix of its upper triangle and real diagonal.

    1. SHIFT. D = H - q I with q = tr(H)/3, and the rounding residue of tr(D) moved back
       into q, so D is traceless to rounding -- which the trigonometric roots assume -- and
       e^{-i q t} factors out.
    2. ORDER. Closed-form trigonometric roots of D give the closest pair (a, b) and the
       loner c. Near a double root the pair's roots carry the cubic formula's
       sqrt(eps)-conditioned error (measured: a 2.0e-17 gap came out 5.1e-15), but the
       loner is accurate to eps * spread at any gap.
    3. LONER EIGENVECTOR. The largest column of (D - a I)(D - b I), normalized. The pair's
       admixture is (lam - a)(lam - b) / ((c - a)(c - b)) for its eigenvalues lam: the
       roots' error times the gap, ~eps p^2 at any gap (the roots are off by
       ~eps p^2 / gap, and by sqrt(eps) p only once the gap itself is below
       sqrt(eps) p), so the admixture stays at the eps level.
    4. COMPLEMENT. x = the coordinate axis least aligned with v, Gram-Schmidted against it
       (norm >= sqrt(2/3)), and y = conj(v cross x), orthonormal to both.
    5. PAIR. D restricted to {x, y} is a 2x2 Hermitian block whose eigenvalues
       mean +- delta (delta = sqrt(((bxx - byy)/2)^2 + |bxy|^2), a sum of squares) and
       eigenvectors are cancellation-free. When delta is below PAIR_FLOOR eps (|q| + p)
       the pair is propagated as degenerate, U = e^{-iqt} [e^{-ict} v v^dag +
       e^{-i mean t} (1 - v v^dag)]: its splitting is not resolved by H in double
       precision (e.g. the vacuum term under a rank-1 potential 1e17 times larger), and
       evolving it would rotate the pair by rounding noise.
    6. Spreads below 1.2e-4 / t use the second-order Taylor series (truncation < 1e-13,
       unitary to |X|^4/4), and H ~ q I is a pure phase.

    History: until 2026-10-12 (v3) U was a polynomial in D through Newton divided
    differences of e^{-ixt} at the eigenvalues. A polynomial evaluated at rounded nodes is
    unitary only to ~eps * spread * min(t, 2/gap): up to 5e-3 per Earth step at
    potentials of 70 eV (the cosmoX2 benchmark), where the spatial phase then created up
    to 0.7% of the flux. A guard switched to the degenerate projector form above
    eps * spread * min(t, 1/delta) = 0.1 only. v3 also read both triangles of an H that
    is Hermitian only to rounding (M_rot = U M U^dag), and its complement basis divided
    by zero for single-flavor sterile mixing (u_mu = 0 or u_e = 0) at potentials above
    ~1e7 eV.

    NOTE: deliberately compiled WITHOUT fastmath -- its approximate-function flag
    degrades sin/cos/exp at these argument sizes.
    """
    q = (H[0, 0].real + H[1, 1].real + H[2, 2].real) / 3.0
    d0 = H[0, 0].real - q
    d1 = H[1, 1].real - q
    d2 = H[2, 2].real - q
    tau = (d0 + d1 + d2) / 3.0
    d0 -= tau
    d1 -= tau
    d2 -= tau
    q += tau
    b01 = H[0, 1]
    b02 = H[0, 2]
    b12 = H[1, 2]
    p1 = (b01.real**2 + b01.imag**2 + b02.real**2 + b02.imag**2 + b12.real**2 + b12.imag**2)
    p = np.sqrt((d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1) / 6.0)
    scale = np.abs(q) + p
    gp = np.exp(-1j * q * t)

    if p <= 1e-14 * max(scale, 1e-300):
        # H ~ q * I: the propagator is a pure global phase
        out.fill(0.0 + 0.0j)
        out[0, 0] = gp
        out[1, 1] = gp
        out[2, 2] = gp
        return

    detB = (d0 * (d1 * d2 - (b12.real**2 + b12.imag**2))
            - (b01 * (np.conjugate(b01) * d2 - b12 * np.conjugate(b02))).real
            + (b02 * (np.conjugate(b01) * np.conjugate(b12) - d1 * np.conjugate(b02))).real)
    r = detB / (2.0 * p * p * p)
    if r < -1.0:
        r = -1.0
    elif r > 1.0:
        r = 1.0
    phi = np.arccos(r) / 3.0
    e_hi = 2.0 * p * np.cos(phi)
    e_lo = 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e_mid = -e_hi - e_lo
    spread = e_hi - e_lo

    c01 = np.conjugate(b01)
    c02 = np.conjugate(b02)
    c12 = np.conjugate(b12)

    if abs(spread * t) < 1.2e-4:
        # U = e^{-iqt} (1 - i t D - t^2 D^2 / 2)
        s00 = d0 * d0 + (b01.real**2 + b01.imag**2) + (b02.real**2 + b02.imag**2)
        s11 = (b01.real**2 + b01.imag**2) + d1 * d1 + (b12.real**2 + b12.imag**2)
        s22 = (b02.real**2 + b02.imag**2) + (b12.real**2 + b12.imag**2) + d2 * d2
        s01 = d0 * b01 + b01 * d1 + b02 * c12
        s02 = d0 * b02 + b01 * b12 + b02 * d2
        s12 = c01 * b02 + d1 * b12 + b12 * d2
        h = -0.5 * t * t
        it = -1j * t
        out[0, 0] = gp * (1.0 + it * d0 + h * s00)
        out[1, 1] = gp * (1.0 + it * d1 + h * s11)
        out[2, 2] = gp * (1.0 + it * d2 + h * s22)
        out[0, 1] = gp * (it * b01 + h * s01)
        out[0, 2] = gp * (it * b02 + h * s02)
        out[1, 2] = gp * (it * b12 + h * s12)
        out[1, 0] = gp * (it * c01 + h * np.conjugate(s01))
        out[2, 0] = gp * (it * c02 + h * np.conjugate(s02))
        out[2, 1] = gp * (it * c12 + h * np.conjugate(s12))
        return

    # Closest pair (a, b); c is the loner.
    if (e_hi - e_mid) <= (e_mid - e_lo):
        a_ = e_hi
        b_ = e_mid
        c_ = e_lo
    else:
        a_ = e_mid
        b_ = e_lo
        c_ = e_hi

    # Loner eigenvector: the largest column of (D - a I)(D - b I)
    best = -1.0
    v0 = 0.0 + 0.0j
    v1 = 0.0 + 0.0j
    v2 = 0.0 + 0.0j
    for col in range(3):
        if col == 0:
            y0 = complex(d0 - b_, 0.0)
            y1 = c01
            y2 = c02
        elif col == 1:
            y0 = b01
            y1 = complex(d1 - b_, 0.0)
            y2 = c12
        else:
            y0 = b02
            y1 = b12
            y2 = complex(d2 - b_, 0.0)
        z0 = (d0 - a_) * y0 + b01 * y1 + b02 * y2
        z1 = c01 * y0 + (d1 - a_) * y1 + b12 * y2
        z2 = c02 * y0 + c12 * y1 + (d2 - a_) * y2
        nrm = z0.real**2 + z0.imag**2 + z1.real**2 + z1.imag**2 + z2.real**2 + z2.imag**2
        if nrm > best:
            best = nrm
            v0 = z0
            v1 = z1
            v2 = z2
    inv = 1.0 / np.sqrt(best)
    v0 *= inv
    v1 *= inv
    v2 *= inv

    # Complement {x, y}: the least-aligned axis minus its projection on v; y = conj(v cross x)
    n0 = v0.real**2 + v0.imag**2
    n1 = v1.real**2 + v1.imag**2
    n2 = v2.real**2 + v2.imag**2
    if n0 <= n1 and n0 <= n2:
        x0 = complex(1.0 - n0, 0.0)
        x1 = -np.conjugate(v0) * v1
        x2 = -np.conjugate(v0) * v2
    elif n1 <= n2:
        x0 = -np.conjugate(v1) * v0
        x1 = complex(1.0 - n1, 0.0)
        x2 = -np.conjugate(v1) * v2
    else:
        x0 = -np.conjugate(v2) * v0
        x1 = -np.conjugate(v2) * v1
        x2 = complex(1.0 - n2, 0.0)
    inv = 1.0 / np.sqrt(x0.real**2 + x0.imag**2 + x1.real**2 + x1.imag**2 + x2.real**2 + x2.imag**2)
    x0 *= inv
    x1 *= inv
    x2 *= inv
    y0 = np.conjugate(v1 * x2 - v2 * x1)
    y1 = np.conjugate(v2 * x0 - v0 * x2)
    y2 = np.conjugate(v0 * x1 - v1 * x0)

    # The pair's 2x2 block of D on {x, y}
    Dx0 = d0 * x0 + b01 * x1 + b02 * x2
    Dx1 = c01 * x0 + d1 * x1 + b12 * x2
    Dx2 = c02 * x0 + c12 * x1 + d2 * x2
    Dy0 = d0 * y0 + b01 * y1 + b02 * y2
    Dy1 = c01 * y0 + d1 * y1 + b12 * y2
    Dy2 = c02 * y0 + c12 * y1 + d2 * y2
    bxx = (np.conjugate(x0) * Dx0 + np.conjugate(x1) * Dx1 + np.conjugate(x2) * Dx2).real
    byy = (np.conjugate(y0) * Dy0 + np.conjugate(y1) * Dy1 + np.conjugate(y2) * Dy2).real
    bxy = np.conjugate(x0) * Dy0 + np.conjugate(x1) * Dy1 + np.conjugate(x2) * Dy2
    mean = 0.5 * (bxx + byy)
    hd = 0.5 * (bxx - byy)
    delta = np.sqrt(hd * hd + bxy.real**2 + bxy.imag**2)

    gc = gp * np.exp(-1j * c_ * t)
    if delta <= PAIR_FLOOR * _EPS * scale:
        # Unresolved pair: U = e^{-iqt} [f_m 1 + (f_c - f_m) v v^dag]
        gm = gp * np.exp(-1j * mean * t)
        dg = gc - gm
        out[0, 0] = gm + dg * n0
        out[1, 1] = gm + dg * n1
        out[2, 2] = gm + dg * n2
        out[0, 1] = dg * (v0 * np.conjugate(v1))
        out[0, 2] = dg * (v0 * np.conjugate(v2))
        out[1, 2] = dg * (v1 * np.conjugate(v2))
        out[1, 0] = dg * (v1 * np.conjugate(v0))
        out[2, 0] = dg * (v2 * np.conjugate(v0))
        out[2, 1] = dg * (v2 * np.conjugate(v1))
        return

    # Pair eigenvectors: w+ = al x + be y (eigenvalue mean + delta), w- = -conj(be) x + conj(al) y
    if hd >= 0.0:
        al = complex(hd + delta, 0.0)
        be = np.conjugate(bxy)
    else:
        al = bxy
        be = complex(delta - hd, 0.0)
    inv = 1.0 / np.sqrt(al.real**2 + al.imag**2 + be.real**2 + be.imag**2)
    al *= inv
    be *= inv
    p0 = al * x0 + be * y0
    p1_ = al * x1 + be * y1
    p2 = al * x2 + be * y2
    m0 = np.conjugate(al) * y0 - np.conjugate(be) * x0
    m1 = np.conjugate(al) * y1 - np.conjugate(be) * x1
    m2 = np.conjugate(al) * y2 - np.conjugate(be) * x2
    ga = gp * np.exp(-1j * (mean + delta) * t)
    gb = gp * np.exp(-1j * (mean - delta) * t)
    out[0, 0] = gc * n0 + ga * (p0.real**2 + p0.imag**2) + gb * (m0.real**2 + m0.imag**2)
    out[1, 1] = gc * n1 + ga * (p1_.real**2 + p1_.imag**2) + gb * (m1.real**2 + m1.imag**2)
    out[2, 2] = gc * n2 + ga * (p2.real**2 + p2.imag**2) + gb * (m2.real**2 + m2.imag**2)
    out[0, 1] = gc * (v0 * np.conjugate(v1)) + ga * (p0 * np.conjugate(p1_)) + gb * (m0 * np.conjugate(m1))
    out[0, 2] = gc * (v0 * np.conjugate(v2)) + ga * (p0 * np.conjugate(p2)) + gb * (m0 * np.conjugate(m2))
    out[1, 2] = gc * (v1 * np.conjugate(v2)) + ga * (p1_ * np.conjugate(p2)) + gb * (m1 * np.conjugate(m2))
    out[1, 0] = gc * (v1 * np.conjugate(v0)) + ga * (p1_ * np.conjugate(p0)) + gb * (m1 * np.conjugate(m0))
    out[2, 0] = gc * (v2 * np.conjugate(v0)) + ga * (p2 * np.conjugate(p0)) + gb * (m2 * np.conjugate(m0))
    out[2, 1] = gc * (v2 * np.conjugate(v1)) + ga * (p2 * np.conjugate(p1_)) + gb * (m2 * np.conjugate(m1))


@njit(fastmath=True, error_model="numpy", cache=True)
def eigenvectors_3x3(H: np.ndarray) -> np.ndarray:
    """Eigenvectors of a Hermitian 3x3 H, columns ordered by ASCENDING eigenvalue --
    the same convention as np.linalg.eigh, whose call in the cosmological mixing loop
    this replaces (the adiabatic mass-eigenstate bookkeeping relies on that ordering
    staying consistent across every (E, z) evaluation).

    Method: closed-form trigonometric eigenvalues (identical to propagator_3x3), then
    for each eigenvalue lam_k the spectral projector numerator (H - lam_j I)(H - lam_l I)
    -- every nonzero column of which is proportional to the eigenvector -- taking the
    largest-norm column and normalizing. Column phases are arbitrary, exactly as they
    are for eigh; every downstream consumer uses only |U|^2 projections.

    Same precision guard as propagator_3x3: relative eigenvalue gaps below 1e-4 fall
    back to np.linalg.eigh (near an MSW-like level crossing the projector columns
    suffer the same catastrophic cancellation as the Lagrange denominators).
    """
    q = (H[0, 0].real + H[1, 1].real + H[2, 2].real) / 3.0
    p1 = (H[0, 1].real**2 + H[0, 1].imag**2 +
          H[0, 2].real**2 + H[0, 2].imag**2 +
          H[1, 2].real**2 + H[1, 2].imag**2)
    d0 = H[0, 0].real - q
    d1 = H[1, 1].real - q
    d2 = H[2, 2].real - q
    p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)

    scale = np.abs(q) + p
    if p <= 1e-14 * max(scale, 1e-300):
        # H ~ q * I: any orthonormal basis works; the identity matches eigh's output here.
        out = np.zeros((3, 3), dtype=np.complex128)
        out[0, 0] = 1.0
        out[1, 1] = 1.0
        out[2, 2] = 1.0
        return out

    b01 = H[0, 1]
    b02 = H[0, 2]
    b12 = H[1, 2]
    detB = (d0 * (d1 * d2 - (b12.real**2 + b12.imag**2))
            - (b01 * (np.conjugate(b01) * d2 - b12 * np.conjugate(b02))).real
            + (b02 * (np.conjugate(b01) * np.conjugate(b12) - d1 * np.conjugate(b02))).real)
    r = detB / (2.0 * p * p * p)
    if r < -1.0:
        r = -1.0
    elif r > 1.0:
        r = 1.0
    phi = np.arccos(r) / 3.0

    lam_hi = q + 2.0 * p * np.cos(phi)
    lam_lo = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    lam_mid = 3.0 * q - lam_hi - lam_lo

    g01 = abs(lam_hi - lam_mid)
    g02 = abs(lam_hi - lam_lo)
    g12 = abs(lam_mid - lam_lo)
    if min(g01, min(g02, g12)) < 1e-4 * (abs(lam_hi) + abs(lam_mid) + abs(lam_lo) + 1e-300):
        evals, evecs = np.linalg.eigh(H)
        return np.ascontiguousarray(evecs)

    out = np.empty((3, 3), dtype=np.complex128)
    lams_asc = (lam_lo, lam_mid, lam_hi)
    for k in range(3):
        lam_k = lams_asc[k]
        lam_a = lams_asc[(k + 1) % 3]
        lam_b = lams_asc[(k + 2) % 3]
        # M = (H - lam_a I)(H - lam_b I); columns proportional to the lam_k eigenvector
        best_norm = -1.0
        v0 = 0.0 + 0.0j
        v1 = 0.0 + 0.0j
        v2 = 0.0 + 0.0j
        for c in range(3):
            # column c of (H - lam_b I): H[:, c] - lam_b * e_c
            y0 = H[0, c]
            y1 = H[1, c]
            y2 = H[2, c]
            if c == 0:
                y0 -= lam_b
            elif c == 1:
                y1 -= lam_b
            else:
                y2 -= lam_b
            # m = (H - lam_a I) @ y
            m0 = (H[0, 0] - lam_a) * y0 + H[0, 1] * y1 + H[0, 2] * y2
            m1 = H[1, 0] * y0 + (H[1, 1] - lam_a) * y1 + H[1, 2] * y2
            m2 = H[2, 0] * y0 + H[2, 1] * y1 + (H[2, 2] - lam_a) * y2
            nrm = (m0.real**2 + m0.imag**2 + m1.real**2 + m1.imag**2 + m2.real**2 + m2.imag**2)
            if nrm > best_norm:
                best_norm = nrm
                v0 = m0
                v1 = m1
                v2 = m2
        inv_nrm = 1.0 / np.sqrt(best_norm)
        out[0, k] = v0 * inv_nrm
        out[1, k] = v1 * inv_nrm
        out[2, k] = v2 * inv_nrm
    return out

