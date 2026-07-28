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
- the propagator is the same closed-form trigonometric-eigenvalue Cayley-Hamilton
  expression as propagator_3x3, vectorized over the batch, with the same 1e-4
  relative-gap guard: near-degenerate batch elements are recomputed through cupy's
  batched Hermitian eigendecomposition, mirroring the CPU eigh fallback;
- everything runs in complex128.

VALIDATION STATUS: this module was developed on a machine without a CUDA device, so
the guards and the CPU fallback are tested here but the GPU numerics are NOT. Before
trusting --use_gpu results, run the regression test (it is skipped without a GPU and
compares GPU vs CPU output at production-like settings):

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
    """exp(-i H t) for a batch of Hermitian 3x3 matrices on the GPU.

    H: (B, 3, 3) complex128; t: (B,) float64. Returns (B, 3, 3) complex128.
    Same mathematics and the same near-degeneracy guard as the CPU propagator_3x3;
    guarded elements are recomputed via cupy.linalg.eigh (batched), exactly as the CPU
    path falls back to numpy's eigh.
    """
    q = cp.real(H[:, 0, 0] + H[:, 1, 1] + H[:, 2, 2]) / 3.0
    p1 = (cp.abs(H[:, 0, 1])**2 + cp.abs(H[:, 0, 2])**2 + cp.abs(H[:, 1, 2])**2)
    d0 = cp.real(H[:, 0, 0]) - q
    d1 = cp.real(H[:, 1, 1]) - q
    d2 = cp.real(H[:, 2, 2]) - q
    p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    p = cp.sqrt(p2 / 6.0)

    scale = cp.abs(q) + p
    iso_mask = p <= 1e-14 * cp.maximum(scale, 1e-300)
    p_safe = cp.where(iso_mask, 1.0, p)

    b01 = H[:, 0, 1]
    b02 = H[:, 0, 2]
    b12 = H[:, 1, 2]
    detB = (d0 * (d1 * d2 - cp.abs(b12)**2)
            - cp.real(b01 * (cp.conj(b01) * d2 - b12 * cp.conj(b02)))
            + cp.real(b02 * (cp.conj(b01) * cp.conj(b12) - d1 * cp.conj(b02))))
    r = cp.clip(detB / (2.0 * p_safe**3), -1.0, 1.0)
    phi = cp.arccos(r) / 3.0

    lam0 = q + 2.0 * p * cp.cos(phi)
    lam2 = q + 2.0 * p * cp.cos(phi + 2.0 * np.pi / 3.0)
    lam1 = 3.0 * q - lam0 - lam2

    g_min = cp.minimum(cp.minimum(cp.abs(lam0 - lam1), cp.abs(lam0 - lam2)), cp.abs(lam1 - lam2))
    degen_mask = g_min < 1e-4 * (cp.abs(lam0) + cp.abs(lam1) + cp.abs(lam2) + 1e-300)

    # Closed-form path (safe denominators substituted where masks will overwrite)
    denom_guard = cp.where(iso_mask | degen_mask, 1.0, 0.0)
    e0 = cp.exp(-1j * lam0 * t)
    e1 = cp.exp(-1j * lam1 * t)
    e2 = cp.exp(-1j * lam2 * t)
    c0 = e0 / ((lam0 - lam1) * (lam0 - lam2) + denom_guard)
    c1 = e1 / ((lam1 - lam0) * (lam1 - lam2) + denom_guard)
    c2 = e2 / ((lam2 - lam0) * (lam2 - lam1) + denom_guard)

    A2 = c0 + c1 + c2
    A1 = c0 * (lam1 + lam2) + c1 * (lam0 + lam2) + c2 * (lam0 + lam1)
    A0 = c0 * lam1 * lam2 + c1 * lam0 * lam2 + c2 * lam0 * lam1

    H2 = cp.matmul(H, H)
    eye = cp.eye(3, dtype=cp.complex128)
    U = (A2[:, None, None] * H2 - A1[:, None, None] * H + A0[:, None, None] * eye)

    # Pure-phase override for H ~ q*I
    if bool(cp.any(iso_mask)):
        phase = cp.exp(-1j * q * t)
        U_iso = phase[:, None, None] * eye
        U = cp.where(iso_mask[:, None, None], U_iso, U)

    # eigh fallback for near-degenerate spectra (rare: only around level crossings)
    if bool(cp.any(degen_mask)):
        idx = cp.where(degen_mask)[0]
        Hm = H[idx]
        w, v = cp.linalg.eigh(Hm)
        phases = cp.exp(-1j * w * t[idx][:, None])
        Um = cp.einsum('bik,bk,bjk->bij', v, phases, cp.conj(v))
        U[idx] = Um

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
