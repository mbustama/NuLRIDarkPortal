"""
gpu_spatial.py

Optional CUDA (cupy) implementation of the Earth chords of the spatial handover
(dp_spatial.propagate_earth_spatial, re-exported by the dark_portal_solver facade),
evolving every (chord, energy-bin) density matrix as one batched GPU workload instead of
distributing units across a CPU multiprocessing pool. The transit to the Earth (dp_transit)
and each chord's exterior run on the CPU first; until 2026-10-13 the Milky Way trajectories
ran here too.

Activated only when the user sets execution.use_gpu = true (CLI --use_gpu) AND cupy
with a working CUDA device is importable; otherwise propagate_earth_spatial falls back
to the CPU path with a warning. The physics is identical by construction:

- the per-step densities come from the same numba helpers the CPU kernels use
  (dp_spatial's chord densities -- single source of truth);
- the propagator is propagator_3x3's spectral sum over a deflated eigenbasis, ported
  to batched array operations (masks in place of branches): traceless shift with the
  global phase factored out, the pure-phase and second-order-Taylor branches, the
  loner's eigenvector from (D - a I)(D - b I), the orthonormal complement and the pair's
  2x2 block, and the degenerate form for pair splittings below double-precision
  resolution. Read propagator_3x3's docstring for why every one of those pieces exists;
- everything runs in complex128; no reduced-precision intrinsics.

VALIDATION STATUS: this module was developed on a machine without a CUDA device.
Because _batched_propagator_gpu receives its array backend as a parameter and uses
only numpy-compatible operations, its MATHEMATICS is validated on the CPU by running
it with numpy as the backend against propagator_3x3 across every branch regime
(test_gpu_batched_propagator_matches_cpu_numpy_backend) -- but execution under cupy
itself is NOT tested here. Before trusting --use_gpu results, run the device
regression test (skipped without a GPU; compares GPU vs CPU output end-to-end at
production-like settings):

    python3 -m pytest tests/test_physics_regression.py -k gpu -v
"""

import numpy as np

from dp_propagator import PAIR_FLOOR, _EPS

_CUPY = None


def _get_cupy():
    """Returns the cupy module if importable with at least one CUDA device, else None.
    Result is cached (including negative results) so repeated checks are free."""
    global _CUPY
    if _CUPY is None:
        try:
            import cupy
            if cupy.cuda.runtime.getDeviceCount() > 0:
                _CUPY = cupy
            else:
                _CUPY = False
        except Exception:
            _CUPY = False
    return _CUPY if _CUPY is not False else None


def gpu_available() -> bool:
    return _get_cupy() is not None


def _batched_propagator_gpu(cp, H, t):
    """exp(-i H t) for a batch of Hermitian 3x3 matrices: propagator_3x3 (the spectral sum
    over a deflated eigenbasis, exactly unitary at any phase) ported to batched array
    operations.

    H: (B, 3, 3) complex128, read as the Hermitian matrix of its upper triangle and real
    diagonal; t: (B,) float64. Returns (B, 3, 3) complex128.
    `cp` is the array backend -- cupy on the GPU, or numpy for the CPU validation of
    this port's mathematics (every operation used exists identically in both).
    Branches become masks; lanes a mask will overwrite still compute with safe
    substituted denominators. Override priority (last write wins): resolved pair ->
    unresolved pair -> Taylor -> pure phase, matching the CPU early-returns in reverse.
    See propagator_3x3's docstring for why each piece exists.
    """
    eye = cp.eye(3, dtype=cp.complex128)

    def sq(z):
        return z.real**2 + z.imag**2

    q = cp.real(H[:, 0, 0] + H[:, 1, 1] + H[:, 2, 2]) / 3.0
    d0 = cp.real(H[:, 0, 0]) - q
    d1 = cp.real(H[:, 1, 1]) - q
    d2 = cp.real(H[:, 2, 2]) - q
    tau = (d0 + d1 + d2) / 3.0
    d0 = d0 - tau
    d1 = d1 - tau
    d2 = d2 - tau
    q = q + tau
    b01 = H[:, 0, 1]
    b02 = H[:, 0, 2]
    b12 = H[:, 1, 2]
    p = cp.sqrt((d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * (sq(b01) + sq(b02) + sq(b12))) / 6.0)
    scale = cp.abs(q) + p
    iso_mask = p <= 1e-14 * cp.maximum(scale, 1e-300)
    p_safe = cp.where(iso_mask, 1.0, p)
    gp = cp.exp(-1j * q * t)

    # Hermitian D from the upper triangle
    D = cp.empty_like(H)
    D[:, 0, 0] = d0
    D[:, 1, 1] = d1
    D[:, 2, 2] = d2
    D[:, 0, 1] = b01
    D[:, 0, 2] = b02
    D[:, 1, 2] = b12
    D[:, 1, 0] = cp.conj(b01)
    D[:, 2, 0] = cp.conj(b02)
    D[:, 2, 1] = cp.conj(b12)

    # Trigonometric roots of D: ordering and the loner
    detB = (d0 * (d1 * d2 - sq(b12))
            - cp.real(b01 * (cp.conj(b01) * d2 - b12 * cp.conj(b02)))
            + cp.real(b02 * (cp.conj(b01) * cp.conj(b12) - d1 * cp.conj(b02))))
    r = cp.clip(detB / (2.0 * p_safe**3), -1.0, 1.0)
    phi = cp.arccos(r) / 3.0
    e_hi = 2.0 * p * cp.cos(phi)
    e_lo = 2.0 * p * cp.cos(phi + 2.0 * np.pi / 3.0)
    e_mid = -e_hi - e_lo
    spread = e_hi - e_lo
    taylor_mask = cp.abs(spread * t) < 1.2e-4
    upper = (e_hi - e_mid) <= (e_mid - e_lo)
    a_ = cp.where(upper, e_hi, e_mid)
    b_ = cp.where(upper, e_mid, e_lo)
    c_ = cp.where(upper, e_lo, e_hi)

    # Loner eigenvector: the largest column of (D - a I)(D - b I)
    M = cp.matmul(D - a_[:, None, None] * eye, D - b_[:, None, None] * eye)
    csel = cp.argmax(cp.sum(sq(M), axis=1), axis=1)
    v = cp.take_along_axis(M, csel[:, None, None], axis=2)[:, :, 0]
    v = v / cp.sqrt(cp.maximum(cp.sum(sq(v), axis=1), 1e-300))[:, None]

    # Complement {x, y}: least-aligned axis minus its projection on v; y = conj(v cross x)
    k1 = cp.argmin(sq(v), axis=1)
    x = eye[k1] - cp.conj(cp.take_along_axis(v, k1[:, None], axis=1)) * v
    x = x / cp.sqrt(cp.maximum(cp.sum(sq(x), axis=1), 1e-300))[:, None]
    y = cp.conj(cp.stack([v[:, 1] * x[:, 2] - v[:, 2] * x[:, 1],
                          v[:, 2] * x[:, 0] - v[:, 0] * x[:, 2],
                          v[:, 0] * x[:, 1] - v[:, 1] * x[:, 0]], axis=1))

    # The pair's 2x2 block on {x, y}
    Dx = cp.einsum('bij,bj->bi', D, x)
    Dy = cp.einsum('bij,bj->bi', D, y)
    bxx = cp.real(cp.sum(cp.conj(x) * Dx, axis=1))
    byy = cp.real(cp.sum(cp.conj(y) * Dy, axis=1))
    bxy = cp.sum(cp.conj(x) * Dy, axis=1)
    mean = 0.5 * (bxx + byy)
    hd = 0.5 * (bxx - byy)
    delta = cp.sqrt(hd * hd + sq(bxy))
    unresolved = delta <= PAIR_FLOOR * _EPS * scale

    gc = gp * cp.exp(-1j * c_ * t)
    vv = v[:, :, None] * cp.conj(v)[:, None, :]

    # Resolved pair: w+ = al x + be y (mean + delta), w- = conj(al) y - conj(be) x
    pos = hd >= 0.0
    al = cp.where(pos, hd + delta + 0j, bxy)
    be = cp.where(pos, cp.conj(bxy), delta - hd + 0j)
    nrm = cp.sqrt(sq(al) + sq(be))
    nrm = cp.where(nrm > 0.0, nrm, 1.0)
    al = al / nrm
    be = be / nrm
    wp = al[:, None] * x + be[:, None] * y
    wm = cp.conj(al)[:, None] * y - cp.conj(be)[:, None] * x
    ga = gp * cp.exp(-1j * (mean + delta) * t)
    gb = gp * cp.exp(-1j * (mean - delta) * t)
    U = (gc[:, None, None] * vv
         + ga[:, None, None] * (wp[:, :, None] * cp.conj(wp)[:, None, :])
         + gb[:, None, None] * (wm[:, :, None] * cp.conj(wm)[:, None, :]))

    # Overrides, lowest to highest priority
    if bool(cp.any(unresolved)):
        gm = gp * cp.exp(-1j * mean * t)
        U_deg = gm[:, None, None] * eye + (gc - gm)[:, None, None] * vv
        U = cp.where(unresolved[:, None, None], U_deg, U)

    if bool(cp.any(taylor_mask)):
        X = (-1j * t)[:, None, None] * D
        U_tay = gp[:, None, None] * (eye + X + 0.5 * cp.matmul(X, X))
        U = cp.where(taylor_mask[:, None, None], U_tay, U)

    if bool(cp.any(iso_mask)):
        U = cp.where(iso_mask[:, None, None], gp[:, None, None] * eye, U)

    return U


def run_spatial_handover_gpu(E_bins, rho0_units_host, n_tot_seq_host, dr_units_host,
                             M_rot, Coef, U_LRI_struct, V_SM=None):
    """Evolves independent units (MW trajectories or Earth chords) through their step
    sequences as one batched GPU workload and returns the final (U, N_E, 3, 3) density
    matrices on the host.

    Inputs are host-side numpy arrays prepared by propagate_earth_spatial:
      E_bins            (N_E,)            energy grid
      rho0_units_host   (U, N_E, 3, 3)    initial density matrices per unit
      n_tot_seq_host    (U, N_steps)      total density per unit per step, built by the
                                          caller under the active earth_step_scheme
                                          ('avg': exact step averages; 'left': left-
                                          endpoint samples of spatial_total_density_*)
      dr_units_host     (U,)              step length per unit in eV^-1
      M_rot             (3, 3)            vacuum mass matrix rotated to flavor basis
      Coef              scalar            LRI coefficient (antineutrino sign included)
      U_LRI_struct      (3, 3)            LRI flavor structure (conjugated upstream)
      V_SM              (3, 3) or None    prebuilt SM potential (Earth phase only)
    """
    cp = _get_cupy()
    if cp is None:
        raise RuntimeError("run_spatial_handover_gpu called without an available GPU")

    n_units, N_E = rho0_units_host.shape[0], rho0_units_host.shape[1]
    n_steps = n_tot_seq_host.shape[1]
    B = n_units * N_E

    rho = cp.asarray(rho0_units_host, dtype=cp.complex128).reshape(B, 3, 3)
    M_rot_d = cp.asarray(M_rot, dtype=cp.complex128)
    S_d = cp.asarray(U_LRI_struct, dtype=cp.complex128)
    V_SM_d = cp.zeros((3, 3), dtype=cp.complex128) if V_SM is None else cp.asarray(V_SM, dtype=cp.complex128)

    inv2E = cp.asarray(1.0 / (2.0 * np.asarray(E_bins)), dtype=cp.float64)      # (N_E,)
    H_vac = inv2E[:, None, None] * M_rot_d[None, :, :]                          # (N_E, 3, 3)
    H_vac_B = cp.tile(H_vac, (n_units, 1, 1))                                   # (B, 3, 3)

    n_tot_seq = cp.asarray(n_tot_seq_host, dtype=cp.float64)                    # (U, S)
    t_B = cp.repeat(cp.asarray(dr_units_host, dtype=cp.float64), N_E)           # (B,)

    for s in range(n_steps):
        V_scale = Coef * n_tot_seq[:, s]                                        # (U,)
        V_B = V_scale[:, None, None] * S_d[None, :, :] + V_SM_d[None, :, :]     # (U, 3, 3)
        H = H_vac_B + cp.repeat(V_B, N_E, axis=0)                               # (B, 3, 3)
        U = _batched_propagator_gpu(cp, H, t_B)
        rho = cp.matmul(U, cp.matmul(rho, cp.conj(cp.swapaxes(U, 1, 2))))

    return cp.asnumpy(rho.reshape(n_units, N_E, 3, 3))
