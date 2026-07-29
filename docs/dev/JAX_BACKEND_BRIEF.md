# Brief: JAX backend for the dark-portal likelihood (`--use_jax`)

Instructions for a future session implementing an optional JAX forward + gradient
backend for this repository. Written 2026-07-29, immediately after speed-up round 7
(HEAD `ca816a4`); read this fully before touching code. The goal is exact gradients
for the profiled fits — Migrad currently estimates them by finite differences at
O(2N) likelihood evaluations per iteration, ~3 s each (single-core) after round 7.
`jax.grad` delivers exact gradients for ~1 forward-pass cost, typically cutting
profiled-scan cost 5–10× and eliminating most repair-pass triggers (which are
finite-difference-noise stalls).

**This is a multi-session project.** Do not attempt it in one sitting. The phases
below have individual acceptance gates; stop at any gate that fails and document the
negative result (this campaign has rejected two of its own optimizations that way —
see the README's "attempted twice, rejected twice" and "Rejected optimization"
notes for the expected style).

---

## 0. Repository context you must know first

* Solver: `src/dark_portal_solver.py`. Analysis pipeline:
  `FrequentistAnalysis/src/core/likelihood.py` (`GenericLikelihoodWrapper`) →
  `src/physics/kernel.py` (`compute_user_model_single`) → `propagate_spectrum`.
  Configuration flows from `DEFAULT_CONFIG` (solver) with unknown-key rejection;
  the analysis mirrors it through a **36-element `static_data` tuple** built in
  `likelihood.py` (element 34 is `thermal_analytic`, element 35 is
  `earth_step_scheme`; extend the tuple for new flags, with the established
  `len(static_data) > N` backward-compat unpack pattern in `kernel.py`). The
  solver itself was split (2026-07) into `dp_core/dp_propagator/dp_thermal/
  dp_cosmo/dp_spatial/dp_config` behind the `dark_portal_solver` facade —
  import through the facade, but read implementation in the dp_* modules.
* Test suite: `python3 -m pytest tests/test_physics_regression.py` — 60 passed +
  1 skipped (GPU device test) at the time of writing. Must stay green throughout,
  including when JAX is **not** installed (skip pattern: see
  `test_gpu_spatial_matches_cpu` / `test_use_gpu_falls_back_to_cpu_without_cuda`).
* Validation discipline (non-negotiable, same as every round): production-settings
  runs (`n_E_bins=1000, n_z_bins=1000, n_mw_paths=6, n_earth_chords=12,
  N_steps_MW=25, N_steps_earth=25, earth_step_scheme="avg", n_cores=4,
  verbose=0` — the certified step-count-independent defaults (N_steps_earth
  100 → 25 on 2026-07-29, revalidated at the four points; historical rounds
  used `left`/500) at four parameter
  points, reference and candidate in **separate processes**:
  - `below`:  `m_Zd_eV=3.0e5, g_D=1e-4, epsilon_prime=0.1`
  - `inband`: `m_Zd_eV=2.0e6, g_D=1e-4, epsilon_prime=0.1`
  - `strong`: `m_Zd_eV=2.0e7, g_D=1e-2, epsilon_prime=0.1`
  - `cosmoX2`: `m_Zd_eV=40.0, g_D=1e-4`
* **Parity tolerances for a different arithmetic stack** (measured amplification
  factors from round 7 — do NOT demand bit-identity from XLA, it is unattainable
  even between two numba compilations of identical source):
  - `below`/`inband`: expect ≤ ~1e-8 max relative flux, flavor ratios ≤ ~1e-10.
  - `strong`: known conditioning floor — ANY last-ulp reordering produces ~7.5e-8
    max relative flux, common-mode across flavors; judge on ratios (≤ ~1e-10).
  - `cosmoX2`: the spatial steps there are in the phase-noise regime; a last-ulp
    perturbation was **measured** to produce 1.5e-3 max relative flux and 3.9e-4
    per-bin ratio differences. Judge cosmoX2 on **energy-averaged flavor ratios
    and dip-location sanity only**; pointwise flux parity is meaningless there.
* Numerical lessons already paid for (respect them; they all transfer to XLA):
  node-exactness of divided differences at large phases; √ε-conditioning of cubic
  roots at double roots (deflation required); the Breit–Wigner spike is narrower
  than any grid (integrate analytically, never sample); phases reach 1e10–1e27 rad
  (float64 mandatory, and phase *values* there are representation noise — only
  unitarity and |U| structure are meaningful).

## 0.1 Two round-7 artifacts are your head start

1. **`src/gpu_spatial.py`** — the v3 propagator already exists as branch-free,
   batched, *backend-parameterized* array code (`_batched_propagator_gpu(cp, H, t)`
   takes its array module as an argument and is CPU-validated by passing numpy).
   A JAX port is nearly a transliteration: `cp` → `jnp`, in-place writes → chained
   `jnp.where`. All masks (pure-phase, Taylor, deflation, phase-noise projector)
   are already mask-form.
2. **The semi-analytic thermal kernel** (`dp_thermal.thermal_averaged_rates_flavor_analytic`)
   — closed-form window edge `p_edge = |m² − δ²|/(2δ)`, `δ = (M2 − m²)/(2x)`,
   `|u'(p_edge)| = 2xδ/E_bg(p_edge)`; geometric shells `p_edge ± {3,30,…,3e4}·MG/|u'|`
   with GL-16 per panel; exact polynomial×atan / polynomial×log antiderivatives
   across the core. Everything is smooth compositions of `sqrt/exp/log/atan` —
   analytically differentiable. The *panel decomposition* depends on parameters:
   keep panel **counts** static (pad unused panels to zero length) so the trace
   has fixed shape; integrand continuity across moving panel edges keeps gradients
   correct.

---

## 1. Activation design (`--use_jax`)

Mirror the existing `use_gpu` pattern exactly:

* `DEFAULT_CONFIG["execution"]["use_jax"] = False` in the solver; CLI flag
  `--use_jax` (str_to_bool, like `--use_gpu`); validated by
  `merge_config_with_defaults` automatically.
* New module `src/dark_portal_jax.py` with `jax_available()` (import guard — JAX
  plus x64 check) and the forward entry point. `propagation_worker` routes to it
  when `use_jax` is set, **falling back to the numba path with a one-line warning**
  when JAX is unavailable (same courtesy as the GPU path).
* FrequentistAnalysis: extend `static_data` to 37 elements
  (`use_jax` at index 36), read from `physics_constants.use_jax` in
  `likelihood.py` with `_PC_DEFAULTS` fallback; `kernel.compute_user_model_single`
  routes. Gradient wiring (phase c) adds a `value_and_grad` method on the wrapper —
  do NOT change `__call__`'s signature (iminuit and the pool workers depend on it).
* At import: `jax.config.update("jax_enable_x64", True)` before anything else, and
  refuse to run (raise, loudly) if x64 cannot be enabled. float32 is a non-starter
  at these phase magnitudes.

## 2. What must be ported (the whole differentiable path)

Order of porting inside `src/dark_portal_jax.py` (each piece unit-tested against
its numba counterpart before moving on):

1. **Propagator**: transliterate `_batched_propagator_gpu` to `jnp`. Branches are
   `jnp.where` masks (already). Write every `|z|²` as `z.real**2 + z.imag**2`
   (never `jnp.abs(z)**2`) for clean Wirtinger gradients. complex128 throughout.
2. **Thermal rates**: port the semi-analytic kernel with static panel counts
   (2 outer + 8 shell + 2 core panels, zero-length padding; GL nodes as constants;
   `vmap` over (E, z) grid cells × 3 states). The `max(I_ang, 0)` clip becomes
   `jnp.maximum` (subgradient at 0 is fine — it fires only on cancellation noise).
3. **Cosmological z-loop**: `jax.lax.scan` over 1000 z-steps carrying `F_mass`;
   the cascade suffix sum is `jnp.cumsum` on the reversed rate·W vector. Either
   port `eigenvectors_3x3` directly (closed form; near-crossing eigh fallback
   becomes an issue — prefer evaluating the |U|² projections through the σ-scale
   invariance directly per (E, z) under `vmap`, dropping the interpolation table:
   tables with parameter-dependent nodes are awkward under JIT and the table was
   only ever a CPU-cost optimization).
4. **Spatial transit**: `lax.scan` over steps (25 MW / 100 Earth at current defaults), `vmap` over
   (bins × units). PREM density: `jnp.select` over the 10 polynomial layers.
   Densities per step can be precomputed per unit as arrays (they depend only on
   geometry + m_eff) — mirror round 7's `V_seq` idea, which is the natural
   JAX-friendly formulation anyway.
5. **Ratios + detector likelihood**: `compute_energy_averaged_ratios` is a
   weighted sum (trivial); the detector likelihood in
   `core/detector_flavor_likelihood.py` is vectorized numpy — port or wrap.
6. **`jax.checkpoint` on the Earth-transit scan body.** Reverse-mode through the
   Earth scan stores every intermediate ρ per step — many GB at production shapes
   even at the current 100-step default.
   Rematerialization brings memory to O(segment) at ~2× forward FLOPs. This is
   the single most important structural decision; do it from the start, not as a
   retrofit.

## 3. Technical hazards (each has bitten someone)

* **Branch seams**: the propagator masks make the function piecewise; gradients
  are correct a.e. but undefined exactly on the switching surfaces. Both sides
  agree to the branch tolerance by construction (that is what the thresholds were
  calibrated for). Document, don't fight.
* **Compile times**: the full pipeline's first trace will take minutes. Fix all
  shapes (grid sizes, panel counts, step counts) per configuration; enable the
  persistent compilation cache (see §6) from day one.
* **Fork-safety**: JAX must never be initialized before a `fork`. main.py forces
  `spawn` already; the scanner uses the OS default (`fork` on Linux) — the JAX
  path must therefore be **lazy-imported inside workers** or the scanner switched
  to spawn when `use_jax` is on. The natural endgame replaces process pools with
  `vmap` batching of grid points in one process (a further win), but do not
  restructure the pools in phase (a).
* **Determinism**: CPU XLA is deterministic; GPU reductions may not be — if GPU
  execution is ever used, set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` during
  validation runs (perf cost is acceptable there).

## 4. Gradient correctness harness (phase b)

* Compare `jax.grad` of the full likelihood against high-order central finite
  differences of the **numba** pipeline (not the JAX one — that would only test
  self-consistency) at ~10 parameter points: on and off the absorption dips, at
  `below`/`inband`-class masses, for each floating parameter class
  (m_Zd log-scale, couplings log-scale, mixing angles, source fractions).
* Use Richardson-extrapolated central differences with per-parameter steps
  (relative 1e-4 … 1e-6 ladder); accept ~1e-4 relative gradient agreement at
  smooth points. Near dip edges finite differences are themselves ill-conditioned
  — compare there qualitatively (sign + magnitude), not to tight tolerance.
* Add `tests/` entries that skip cleanly without JAX installed.

## 5. Fitting integration (phase c)

Escalate in this order, benchmarking a real 1-D profile scan
(`config/pion_2050_case_5_fixed_nuisance.json`, small grid first) at each step:

1. iminuit `Minuit(func, grad=...)` — smallest change, keeps everything else.
2. Inner profiled fits → `scipy.optimize.minimize(..., method="L-BFGS-B",
   jac=True)` with bounds from `parameter_limits`; Migrad stays as the global-fit
   orchestrator. Expect ~15–30 value_and_grad calls per point vs the current ~134
   evaluations (post-`minuit_strategy=0`).
3. (Optional, later) full-JAX scans: `vmap` the whole profiled objective over
   grid points.

**Decision gate**: if step 2 does not show ≥3× wall-time improvement on a real
scan, stop and document; steps beyond are not worth the maintenance surface.

## 6. HPC / SLURM compatibility

JAX runs fine under SLURM — it is a self-contained Python library, no MPI, no
daemon. The considerations that actually matter for THIS pipeline:

* **CPU-only JAX is the safe default** for the existing 200-worker scan
  architecture: `pip install jax` (CPU wheel) works on any cluster with no
  CUDA/driver matching. Each pool worker can run the JAX forward+grad model
  independently — the scans are embarrassingly parallel and need no cross-process
  JAX communication.
* **Compile stampede** (the JAX analogue of the numba cold-start stampede fixed in
  round 7): 200 spawn workers each re-tracing the pipeline for minutes would be
  disastrous. Enable JAX's **persistent compilation cache** on the shared
  filesystem — `jax.config.update("jax_compilation_cache_dir", <shared path>)` or
  `JAX_COMPILATION_CACHE_DIR` — so one warm run populates it and workers load
  compiled artifacts (exactly like numba's `cache=True`). Extend the existing
  pool `initializer=` prewarm (`src/physics/kernel.py::prewarm_worker`) to touch
  the JAX path when `use_jax` is on.
* **Thread pinning**: XLA-CPU spins its own intra-op thread pool per process; on
  a node running many workers this oversubscribes exactly like the BLAS pools the
  solver already pins. Per worker set `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=1"` (and keep `--cpus-per-task` consistent) — add
  it next to the existing `OMP_NUM_THREADS` guards.
* **Start method**: JAX is not fork-safe. main.py already forces `spawn`; under
  SLURM keep spawn (or one-process-per-task, see next point) and lazy-import JAX
  inside workers.
* **The better SLURM architecture** (phase-c+): instead of one job with a
  200-worker pool, submit a **SLURM array job** where each task is one process
  that `vmap`s a chunk of grid points. Fewer, fatter processes amortize the JIT
  once each, drop the pickling entirely, and map cleanly onto
  `#SBATCH --array=0-39` + a chunk index. The repo's existing
  `slurm_batch_frequentist_dark_portal.sh` is the file to extend.
* **GPU nodes (optional)**: needs a CUDA-matched wheel (`pip install "jax[cuda12]"`
  or the cluster's module). Two flags matter on shared GPUs: JAX preallocates 75%
  of device memory by default — set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or
  `XLA_PYTHON_CLIENT_MEM_FRACTION`) — and request the device with
  `#SBATCH --gres=gpu:1` (JAX honors `CUDA_VISIBLE_DEVICES`). One process per
  GPU; never point a worker pool at a single GPU.
* **What does NOT apply**: `jax.distributed`/multi-node collectives — this
  workload has no cross-node coupling; array jobs are simpler and equally fast.

## 7. Phasing summary with acceptance gates

| Phase | Deliverable | Gate |
|---|---|---|
| a | Forward-parity JAX kernel behind `--use_jax` (solver + analysis routing, fallback warning, tests skip without JAX) | 4-point validation at the §0 tolerances; suite green with and without JAX installed |
| b | Gradient harness | `jax.grad` vs numba finite differences per §4 |
| c | `grad=` / L-BFGS-B wiring + one real 1-D profile scan benchmark | ≥3× wall-time on the real scan, chi2 curves agreeing within repair-threshold (0.2) |
| d | Decision point | User decides on wider adoption (SLURM array-job restructure, GPU) |

Honest cost estimate: the thermal kernel and propagator are each a careful day;
the checkpointed spatial scan another; validation at this repo's standards at
least as much again. The payoff only materializes for large scan campaigns — at
0.72 s per pooled evaluation, single runs do not need this.
