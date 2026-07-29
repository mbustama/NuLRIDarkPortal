"""
gpu_spatial.py

Optional CUDA (cupy) implementation of the spatial handover -- the Milky Way and Earth
transit phases of dark_portal_solver.propagate_earth_spatial -- evolving every
(trajectory-or-chord, energy-bin) density matrix as one batched GPU workload instead of
distributing units across a CPU multiprocessing pool.

Activated only when the user sets execution.use_gpu = true (CLI --use_gpu) AND cupy
with a working CUDA device is importable; otherwise propagate_earth_spatial falls back
to the CPU path with a warning. The physics is identical by construction:

- the per-step densities come from the same numba helpers the CPU kernels use
  (spatial_total_density_mw / spatial_total_density_earth -- single source of truth);
- the propagator is the v3 deflation-based divided-difference scheme of
  propagator_3x3, ported branch-for-branch to batched array operations (masks in
  place of branches, NO eigendecomposition anywhere): traceless shift with the
  global phase factored out, the pure-phase and second-order-Taylor branches, the
  closest-pair ordering with node-exact first divided differences (sinc form only
  below |gap*t/2| = 0.5, direct quotient otherwise), the deflation of
  sqrt(eps)-conditioned near-double trig roots via the far eigenvector's 2x2
  complement block, and the phase-noise projector guard for potentials beyond
  double-precision phase tracking. Read propagator_3x3's docstring for why every
  one of those pieces exists;
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
    """exp(-i H t) for a batch of Hermitian 3x3 matrices: the v3 propagator_3x3
    ported branch-for-branch to batched array operations.

    H: (B, 3, 3) complex128; t: (B,) float64. Returns (B, 3, 3) complex128.
    `cp` is the array backend -- cupy on the GPU, or numpy for the CPU validation of
    this port's mathematics (every operation used exists identically in both).
    Branches become masks; lanes a mask will overwrite still compute with safe
    substituted denominators. Override priority (last write wins): polynomial ->
    phase-noise projector -> Taylor -> pure phase, matching the CPU early-returns
    in reverse. See propagator_3x3's docstring for why each piece exists.
    """
    eye = cp.eye(3, dtype=cp.complex128)

    q = cp.real(H[:, 0, 0] + H[:, 1, 1] + H[:, 2, 2]) / 3.0
    d0 = cp.real(H[:, 0, 0]) - q
    d1 = cp.real(H[:, 1, 1]) - q
    d2 = cp.real(H[:, 2, 2]) - q
    b01 = H[:, 0, 1]
    b02 = H[:, 0, 2]
    b12 = H[:, 1, 2]
    p1 = cp.abs(b01)**2 + cp.abs(b02)**2 + cp.abs(b12)**2
    p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    p = cp.sqrt(p2 / 6.0)

    scale = cp.abs(q) + p
    iso_mask = p <= 1e-14 * cp.maximum(scale, 1e-300)
    p_safe = cp.where(iso_mask, 1.0, p)

    # Trigonometric roots of the traceless D = H - q I
    detB = (d0 * (d1 * d2 - cp.abs(b12)**2)
            - cp.real(b01 * (cp.conj(b01) * d2 - b12 * cp.conj(b02)))
            + cp.real(b02 * (cp.conj(b01) * cp.conj(b12) - d1 * cp.conj(b02))))
    r = cp.clip(detB / (2.0 * p_safe**3), -1.0, 1.0)
    phi = cp.arccos(r) / 3.0
    e_hi = 2.0 * p * cp.cos(phi)
    e_lo = 2.0 * p * cp.cos(phi + 2.0 * np.pi / 3.0)
    e_mid = -e_hi - e_lo

    spread = e_hi - e_lo
    gp = cp.exp(-1j * q * t)
    D = H - q[:, None, None] * eye

    taylor_mask = cp.abs(spread * t) < 1.2e-4

    # Closest-pair ordering: (a_, b_) the close pair, c_ the loner
    upper = (e_hi - e_mid) <= (e_mid - e_lo)
    a_ = cp.where(upper, e_hi, e_mid)
    b_ = cp.where(upper, e_mid, e_lo)
    c_ = cp.where(upper, e_lo, e_hi)

    # DEFLATION (computed for every lane, selected by mask): far eigenvector from the
    # largest column of (D - mean_pair I)^2, Gram-Schmidt complement, pair mean and
    # half-gap from the cancellation-free 2x2 block.
    defl_mask = (cp.abs(a_ - b_) < 1e-4 * spread) & (~iso_mask)
    mean_pair = -0.5 * c_
    Dm = D - mean_pair[:, None, None] * eye
    M = cp.matmul(Dm, Dm)
    col_norm = cp.sum(cp.abs(M)**2, axis=1)                       # (B, 3) per column
    csel = cp.argmax(col_norm, axis=1)
    v0 = cp.take_along_axis(M, csel[:, None, None], axis=2)[:, :, 0]
    v0 = v0 / cp.sqrt(cp.maximum(cp.sum(cp.abs(v0)**2, axis=1), 1e-300))[:, None]
    k1 = cp.argmin(cp.abs(v0)**2, axis=1)
    x1 = eye[k1] - cp.conj(cp.take_along_axis(v0, k1[:, None], axis=1)) * v0
    x1 = x1 / cp.sqrt(cp.maximum(cp.sum(cp.abs(x1)**2, axis=1), 1e-300))[:, None]
    k2 = cp.where(k1 != 0, 0, 1)
    x2 = eye[k2] - cp.conj(cp.take_along_axis(v0, k2[:, None], axis=1)) * v0
    ox = cp.sum(cp.conj(x1) * x2, axis=1)
    x2 = x2 - ox[:, None] * x1
    x2 = x2 / cp.sqrt(cp.maximum(cp.sum(cp.abs(x2)**2, axis=1), 1e-300))[:, None]
    Dx1 = cp.einsum('bij,bj->bi', D, x1)
    Dx2 = cp.einsum('bij,bj->bi', D, x2)
    b11_ = cp.real(cp.sum(cp.conj(x1) * Dx1, axis=1))
    b22_ = cp.real(cp.sum(cp.conj(x2) * Dx2, axis=1))
    b12_ = cp.sum(cp.conj(x1) * Dx2, axis=1)
    mean_b = 0.5 * (b11_ + b22_)
    half_diff = 0.5 * (b11_ - b22_)
    delta = cp.sqrt(half_diff * half_diff + cp.abs(b12_)**2)
    a_ = cp.where(defl_mask, mean_b + delta, a_)
    b_ = cp.where(defl_mask, mean_b - delta, b_)

    # PHASE-NOISE PROJECTOR GUARD (v3 piece 4), inside the deflation branch only
    abs_t = cp.abs(t)
    slope = cp.where(delta * abs_t > 1.0, 1.0 / cp.maximum(delta, 1e-300), abs_t)
    noise_mask = defl_mask & (2.2e-16 * spread * slope > 0.1)

    # Newton divided differences of e^{-i x t}: node-exact branch selection
    f_a = cp.exp(-1j * a_ * t)
    f_b = cp.exp(-1j * b_ * t)
    f_c = cp.exp(-1j * c_ * t)

    u_half = 0.5 * (a_ - b_) * t
    u_safe = cp.where(u_half != 0.0, u_half, 1.0)
    sinc_ab = cp.where(u_half != 0.0, cp.sin(u_safe) / u_safe, 1.0)
    dd_ab = cp.where(cp.abs(u_half) < 0.5,
                     cp.exp(-0.5j * (a_ + b_) * t) * (-1j * t) * sinc_ab,
                     (f_a - f_b) / cp.where(a_ != b_, a_ - b_, 1.0))

    v_half = 0.5 * (b_ - c_) * t
    v_safe = cp.where(v_half != 0.0, v_half, 1.0)
    sinc_bc = cp.where(v_half != 0.0, cp.sin(v_safe) / v_safe, 1.0)
    dd_bc = cp.where(cp.abs(v_half) < 0.5,
                     cp.exp(-0.5j * (b_ + c_) * t) * (-1j * t) * sinc_bc,
                     (f_b - f_c) / cp.where(b_ != c_, b_ - c_, 1.0))

    dd_abc = (dd_ab - dd_bc) / cp.where(a_ != c_, a_ - c_, 1.0)

    A2 = dd_abc
    A1 = dd_ab - (a_ + b_) * dd_abc
    A0 = f_a - a_ * dd_ab + a_ * b_ * dd_abc

    D2 = cp.matmul(D, D)
    U = gp[:, None, None] * (A2[:, None, None] * D2 + A1[:, None, None] * D
                             + A0[:, None, None] * eye)

    # Overrides, lowest to highest priority
    if bool(cp.any(noise_mask)):
        f_p = cp.exp(-1j * mean_b * t)
        P = v0[:, :, None] * cp.conj(v0)[:, None, :]
        U_proj = gp[:, None, None] * (f_p[:, None, None] * eye
                                      + (f_c - f_p)[:, None, None] * P)
        U = cp.where(noise_mask[:, None, None], U_proj, U)

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
      n_tot_seq_host    (U, N_steps)      total density per unit per step (from the
                                          shared spatial_total_density_* helpers)
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
