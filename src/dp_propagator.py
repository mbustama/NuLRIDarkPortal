"""
dp_propagator.py

The deflation-based divided-difference 3x3 propagator (v3) and the closed-form
eigenvector routine -- the most delicate numerics in the repository; read the
docstrings before touching anything. Split verbatim out of dark_portal_solver.py.
"""
import numpy as np

from dp_core import njit

@njit(cache=True)
def propagator_3x3(H: np.ndarray, t: float, out: np.ndarray):
    """Writes exp(-i H t) for a Hermitian 3x3 H into out. No eigendecomposition, no
    fallback -- uniformly accurate at any eigenvalue gap. Three cooperating pieces:

    1. Everything runs on the traceless shift D = H - q I (q = tr(H)/3), with the global
       phase e^{-i q t} factored out, so all degeneracy logic is scale-invariant.
    2. NODES. Closed-form trigonometric roots of D are used when well-separated. Near a
       double root the cubic formula's roots carry the classic sqrt(eps)-conditioned
       error -- measured on a production Earth-transit Hamiltonian, the close-pair GAP
       came out 257x too large (5.1e-15 vs a true 2.0e-17), which rotates the
       near-degenerate subspace by a wildly wrong angle each step and accumulated to a
       1e-4-level flux error. So whenever the trig roots' closest pair sits below 1e-4
       of the spread, the pair is recomputed by DEFLATION: the well-separated root and
       the pair mean (= -far/2, since tr D = 0) are accurate, the far eigenvector comes
       from the stable projector (D - mean I)^2 (pair contamination suppressed by
       (gap/separation)^2), and the pair's mean and half-gap are read off the 2x2
       complement block -- whose gap formula sqrt(((b11-b22)/2)^2 + |b12|^2) is a sum of
       squares, cancellation-free.
    3. COEFFICIENTS. Newton divided differences of f(x) = e^{-i x t} over the nodes,
       closest pair first: f[a,b] via the exact sinc form when |(a-b)t/2| < 0.5 (where
       the direct difference cancels), and via the direct node-exact quotient
       (f_a - f_b)/(a - b) otherwise -- at large gap phases (Milky Way steps reach
       ~1e9 rad) only the direct form keeps the polynomial interpolating the COMPUTED
       exponentials exactly, which is what keeps the propagator unitary. A second-order
       Taylor branch handles total phase spreads below 1.2e-4 (truncation < 3e-13).
    4. PHASE-NOISE GUARD. Within the deflation branch only: when the pair splitting is
       at the cancellation floor AND the amplification factor eps * spread * min(|t|,
       1/delta) reaches O(0.1) -- e.g. the 1/m_Zd^2-enhanced Earth potential at
       m_Zd ~ 1e-6 eV, where spread*t ~ 1e27 rad -- polynomial evaluation at the
       noise-limited nodes is non-unitary by many orders (measured singular values
       ~5e11; 500 steps overflowed rho to NaN, the pre-2026-07 all-NaN flux bug at
       extreme mediator masses). The propagator then switches to the exactly-unitary
       spectral-projector form built from the stable far eigenvector, dropping the
       (machine-noise) pair-splitting phase. Outside the deflation branch this hazard
       is unreachable (gap >= 1e-4 * spread bounds the amplification at ~2e-12).

    NOTE: deliberately compiled WITHOUT fastmath -- its approximate-function flag
    degrades sin/cos/exp at these argument sizes.
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
        # H ~ q * I: the propagator is a pure global phase
        phase = np.exp(-1j * q * t)
        out.fill(0.0 + 0.0j)
        out[0, 0] = phase
        out[1, 1] = phase
        out[2, 2] = phase
        return

    # Trigonometric roots of the traceless D = H - q I
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

    e_hi = 2.0 * p * np.cos(phi)
    e_lo = 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    e_mid = -e_hi - e_lo

    spread = e_hi - e_lo
    global_phase = np.exp(-1j * q * t)

    if abs(spread * t) < 1.2e-4:
        # Tiny total phase spread: second-order Taylor in X = -i t D around the global
        # phase; truncation ||X||^3 / 6 < 3e-13. Covers every all-three-close case.
        for a in range(3):
            for b in range(3):
                x_ab = -1j * t * H[a, b]
                if a == b:
                    x_ab += 1j * t * q
                x2_ab = 0.0 + 0.0j
                for k in range(3):
                    x_ak = -1j * t * H[a, k]
                    x_kb = -1j * t * H[k, b]
                    if a == k:
                        x_ak += 1j * t * q
                    if k == b:
                        x_kb += 1j * t * q
                    x2_ab += x_ak * x_kb
                val = x_ab + 0.5 * x2_ab
                if a == b:
                    val += 1.0
                out[a, b] = global_phase * val
        return

    # Order so the closest pair is (a, b); c is the loner.
    if (e_hi - e_mid) <= (e_mid - e_lo):
        a_ = e_hi
        b_ = e_mid
        c_ = e_lo
    else:
        a_ = e_mid
        b_ = e_lo
        c_ = e_hi

    if abs(a_ - b_) < 1e-4 * spread:
        # DEFLATION: trig roots of a near-double pair are sqrt(eps)-conditioned; rebuild
        # the pair from the 2x2 complement of the (accurately known) far eigenvector.
        mean_pair = -0.5 * c_          # tr D = 0  =>  a + b = -c
        # Far eigenvector: largest column of M = (D - mean_pair I)^2; the pair's
        # contribution is (±gap/2)^2 vs (c - mean_pair)^2 for the far root.
        m00 = complex(d0 - mean_pair, 0.0)
        m11 = complex(d1 - mean_pair, 0.0)
        m22 = complex(d2 - mean_pair, 0.0)
        # M columns (M = Dm @ Dm with Dm Hermitian)
        v0 = np.zeros(3, dtype=np.complex128)
        best_norm = -1.0
        for col in range(3):
            if col == 0:
                y0 = m00
                y1 = np.conjugate(b01)
                y2 = np.conjugate(b02)
            elif col == 1:
                y0 = b01
                y1 = m11
                y2 = np.conjugate(b12)
            else:
                y0 = b02
                y1 = b12
                y2 = m22
            c0 = m00 * y0 + b01 * y1 + b02 * y2
            c1 = np.conjugate(b01) * y0 + m11 * y1 + b12 * y2
            c2 = np.conjugate(b02) * y0 + np.conjugate(b12) * y1 + m22 * y2
            nrm = (c0.real**2 + c0.imag**2 + c1.real**2 + c1.imag**2 + c2.real**2 + c2.imag**2)
            if nrm > best_norm:
                best_norm = nrm
                v0[0] = c0
                v0[1] = c1
                v0[2] = c2
        inv = 1.0 / np.sqrt(best_norm)
        v0[0] *= inv
        v0[1] *= inv
        v0[2] *= inv
        # Orthonormal complement {x1, x2}: Gram-Schmidt the least-aligned axis, then a
        # second axis orthogonalized against both.
        a0 = v0[0].real**2 + v0[0].imag**2
        a1 = v0[1].real**2 + v0[1].imag**2
        a2 = v0[2].real**2 + v0[2].imag**2
        if a0 <= a1 and a0 <= a2:
            k1 = 0
        elif a1 <= a2:
            k1 = 1
        else:
            k1 = 2
        x1 = np.zeros(3, dtype=np.complex128)
        x1[k1] = 1.0
        ov = np.conjugate(v0[k1])
        x1[0] -= ov * v0[0]
        x1[1] -= ov * v0[1]
        x1[2] -= ov * v0[2]
        n1 = np.sqrt(x1[0].real**2 + x1[0].imag**2 + x1[1].real**2 + x1[1].imag**2 + x1[2].real**2 + x1[2].imag**2)
        x1[0] /= n1
        x1[1] /= n1
        x1[2] /= n1
        k2 = 0 if k1 != 0 else 1
        x2 = np.zeros(3, dtype=np.complex128)
        x2[k2] = 1.0
        ov = np.conjugate(v0[k2])
        x2[0] -= ov * v0[0]
        x2[1] -= ov * v0[1]
        x2[2] -= ov * v0[2]
        ox = np.conjugate(x1[0]) * x2[0] + np.conjugate(x1[1]) * x2[1] + np.conjugate(x1[2]) * x2[2]
        x2[0] -= ox * x1[0]
        x2[1] -= ox * x1[1]
        x2[2] -= ox * x1[2]
        n2 = np.sqrt(x2[0].real**2 + x2[0].imag**2 + x2[1].real**2 + x2[1].imag**2 + x2[2].real**2 + x2[2].imag**2)
        x2[0] /= n2
        x2[1] /= n2
        x2[2] /= n2
        # 2x2 block of D in the {x1, x2} basis (D applied via its Hermitian elements)
        # D @ x vectors
        Dx1_0 = complex(d0, 0.0) * x1[0] + b01 * x1[1] + b02 * x1[2]
        Dx1_1 = np.conjugate(b01) * x1[0] + complex(d1, 0.0) * x1[1] + b12 * x1[2]
        Dx1_2 = np.conjugate(b02) * x1[0] + np.conjugate(b12) * x1[1] + complex(d2, 0.0) * x1[2]
        Dx2_0 = complex(d0, 0.0) * x2[0] + b01 * x2[1] + b02 * x2[2]
        Dx2_1 = np.conjugate(b01) * x2[0] + complex(d1, 0.0) * x2[1] + b12 * x2[2]
        Dx2_2 = np.conjugate(b02) * x2[0] + np.conjugate(b12) * x2[1] + complex(d2, 0.0) * x2[2]
        b11_ = (np.conjugate(x1[0]) * Dx1_0 + np.conjugate(x1[1]) * Dx1_1 + np.conjugate(x1[2]) * Dx1_2).real
        b22_ = (np.conjugate(x2[0]) * Dx2_0 + np.conjugate(x2[1]) * Dx2_1 + np.conjugate(x2[2]) * Dx2_2).real
        b12_ = (np.conjugate(x1[0]) * Dx2_0 + np.conjugate(x1[1]) * Dx2_1 + np.conjugate(x1[2]) * Dx2_2)
        mean_b = 0.5 * (b11_ + b22_)
        half_diff = 0.5 * (b11_ - b22_)
        delta = np.sqrt(half_diff * half_diff + b12_.real**2 + b12_.imag**2)

        # PHASE-NOISE GUARD (piece 4 of the docstring). Every quantity above carries an
        # absolute cancellation floor ~ eps * spread from arithmetic on O(spread) matrix
        # entries. Evaluating the interpolating polynomial at D's true eigenvalues (which
        # sit within that floor of the nodes) amplifies by ~ node_error * |p'|, with
        # |p'| ~ |t| on the sinc branch (pair phase < O(1)) and ~ 1/delta on the direct-
        # quotient branch. When eps * spread * min(|t|, 1/delta) reaches O(0.1), NO
        # polynomial in D can remain unitary -- measured 5e11 singular values at
        # m_Zd = 1e-6 eV, where the 1/m_Zd^2-enhanced Earth potential reaches ~1e17 eV
        # and spread*t ~ 6e27 rad; 500 such steps overflow rho to Inf/NaN. In that
        # regime the pair-splitting phase is machine noise whatever we do, so the only
        # meaningful propagator is the exactly-unitary projector form
        #     U = e^{-iqt} [ f_c v0 v0^dag + f_pair (I - v0 v0^dag) ],
        # which drops the sub-noise splitting but preserves unitarity exactly (the flux
        # then carries decoherence-averaged information, the best double precision can
        # represent there). The far eigenvector v0 is already computed stably above.
        slope = np.abs(t)
        if delta > 0.0 and delta * slope > 1.0:
            slope = 1.0 / delta
        if 2.2e-16 * spread * slope > 0.1:
            f_c = np.exp(-1j * c_ * t)
            f_p = np.exp(-1j * mean_b * t)
            df = f_c - f_p
            for a in range(3):
                for b in range(3):
                    val = df * (v0[a] * np.conjugate(v0[b]))
                    if a == b:
                        val += f_p
                    out[a, b] = global_phase * val
            return

        a_ = mean_b + delta
        b_ = mean_b - delta

    f_a = np.exp(-1j * a_ * t)
    f_b = np.exp(-1j * b_ * t)
    f_c = np.exp(-1j * c_ * t)

    # First divided differences: sinc form only where the direct difference cancels
    # (|gap phase| < 0.5); the direct node-exact quotient everywhere else (mandatory at
    # large gap phases -- see docstring).
    u_half = 0.5 * (a_ - b_) * t
    if abs(u_half) < 0.5:
        sinc_ab = np.sin(u_half) / u_half if u_half != 0.0 else 1.0
        dd_ab = np.exp(-0.5j * (a_ + b_) * t) * (-1j * t) * sinc_ab
    else:
        dd_ab = (f_a - f_b) / (a_ - b_)

    v_half = 0.5 * (b_ - c_) * t
    if abs(v_half) < 0.5:
        sinc_bc = np.sin(v_half) / v_half if v_half != 0.0 else 1.0
        dd_bc = np.exp(-0.5j * (b_ + c_) * t) * (-1j * t) * sinc_bc
    else:
        dd_bc = (f_b - f_c) / (b_ - c_)

    dd_abc = (dd_ab - dd_bc) / (a_ - c_)

    # Newton -> power basis in D: U = e^{-iqt} (A2 D^2 + A1 D + A0 I)
    A2 = dd_abc
    A1 = dd_ab - (a_ + b_) * dd_abc
    A0 = f_a - a_ * dd_ab + a_ * b_ * dd_abc

    # Evaluate on D = H - q I without materializing it: fully unrolled straight-line
    # code, operation-for-operation identical to the original a/b/k loop nest (same
    # products, same summation order -- bit-identical by construction), with the D
    # entries hoisted into locals once instead of re-read per (a, b) pair.
    e00 = H[0, 0] - q
    e11 = H[1, 1] - q
    e22 = H[2, 2] - q
    e01 = H[0, 1]
    e02 = H[0, 2]
    e12 = H[1, 2]
    e10 = H[1, 0]
    e20 = H[2, 0]
    e21 = H[2, 1]

    out[0, 0] = global_phase * (A2 * (e00 * e00 + e01 * e10 + e02 * e20) + A1 * e00 + A0)
    out[0, 1] = global_phase * (A2 * (e00 * e01 + e01 * e11 + e02 * e21) + A1 * e01)
    out[0, 2] = global_phase * (A2 * (e00 * e02 + e01 * e12 + e02 * e22) + A1 * e02)
    out[1, 0] = global_phase * (A2 * (e10 * e00 + e11 * e10 + e12 * e20) + A1 * e10)
    out[1, 1] = global_phase * (A2 * (e10 * e01 + e11 * e11 + e12 * e21) + A1 * e11 + A0)
    out[1, 2] = global_phase * (A2 * (e10 * e02 + e11 * e12 + e12 * e22) + A1 * e12)
    out[2, 0] = global_phase * (A2 * (e20 * e00 + e21 * e10 + e22 * e20) + A1 * e20)
    out[2, 1] = global_phase * (A2 * (e20 * e01 + e21 * e11 + e22 * e21) + A1 * e21)
    out[2, 2] = global_phase * (A2 * (e20 * e02 + e21 * e12 + e22 * e22) + A1 * e22 + A0)


@njit(fastmath=True, cache=True)
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

