# Brief: potential-texture generality answer + deep scanner/FrequentistAnalysis speed-up review

You are picking up a long-running performance/correctness campaign on this repository.
This brief is self-contained: read it fully before touching code. It specifies two
tasks — (1) a precise **written answer** on whether the potential textures can take
arbitrary forms, and (2) a **deep review + implementation pass** on speed-ups and
improvements to `src/dark_portal_scanner.py` and the `FrequentistAnalysis/` pipeline —
plus the mandatory validation discipline and the pitfalls previous rounds paid for.

---

## 0. Repository state and environment

* Repo: `/home/mbustamante/Research/NuLRIDarkPortal`, branch `dev` (tracked on
  `origin/dev`, GitHub `mbustama/NuLRIDarkPortal`). HEAD at the time of writing:
  the commit adding this brief, on top of `cb5f51f`. A PR merging `dev` into `main`
  is open (PR #1) — **never merge to `main` yourself**; the user does that. Commit
  locally as you go; push to `origin dev` when a task lands (standing authorization).
* **Module layout** (post-split, 2026-07-29): the solver lives in
  `src/dp_core.py` (env guards — BLAS pins + TBB layer — MUST stay the first import;
  constants; PMNS/mass/dE utilities), `dp_propagator.py`, `dp_thermal.py`,
  `dp_cosmo.py`, `dp_spatial.py`, `dp_config.py`, with **`dark_portal_solver.py` as
  the byte-compatible public facade** (import through it; read implementation in the
  dp_* modules). Two rules learned the hard way: **monkeypatches must target the
  owner module** (e.g. `dp_spatial._earth_chord_worker`), not the facade; and the
  mutable `HAS_TQDM` flag lives in `dp_core` (write it there).
* Test suite: `python3 -m pytest tests/test_physics_regression.py` — currently
  **60 passed + 1 skipped** (GPU device test; no CUDA here). Must stay green after
  every task. Warm runs ~20 s; touching numba kernels recompiles (~2 min once).
* Environment: laptop-class **hybrid** CPU (i5-1334U: 2 P-cores + 8 E-cores,
  12 threads) that **throttles to ~1/3 clock under sustained load** — wall-clock
  comparisons are only meaningful same-conditions, best-of-3, interleaved.
  Anaconda Python 3.12, numba 0.60 / llvmlite 0.43.0 build `_1`
  (**SVML ACTIVE**: `numba.core.config.USING_SVML` is True — it accelerates
  array-vectorized fastmath kernels ~1.6x but does nothing for scalar/branchy
  loops), `tbb` installed (TBB threading layer active; prange + spawn pools
  coexist safely), `iminuit`, no CUDA/cupy.
  `FrequentistAnalysis/src/main.py` forces the `spawn` start method — **every
  multiprocessing timing/behavior claim must be checked under `spawn`**, and a
  spawn test script must be a real file (spawn workers cannot re-import a stdin
  `__main__`; a heredoc + Pool deadlocks by construction).
* Current benchmarks (8 cores unless stated; throttle caveat applies):
  - Solver defaults (1000×1000 grid, 6 paths / 12 chords, `avg` scheme, 100 Earth
    steps): **~1.1 s** full run.
  - Analysis likelihood evaluation (`config/pion_2050_case_5_fixed_nuisance.json`,
    200×200, 50 paths / 50 chords): **0.84 s** single-core, **0.25 s** pooled.
  - Thermal grid at production size: 0.28–0.8 s (8 threads); one exact grid build
    at analysis settings: ~113 ms single-core; two-component cache hit: 0.3 ms.
* A persistent memory directory exists whose summaries agree with this brief;
  where they conflict, this brief wins.

## 0.1 Validation discipline (non-negotiable)

Physics-affecting changes are validated **against the real pipeline at production
settings across multiple parameter points**, reference and candidate in **separate
processes**, comparing the max relative difference over the full spectrum plus
flavor ratios. Standard recipe: worktree at the pre-change commit + a runner that
calls `dps.run_solver` at `n_E_bins=1000, n_z_bins=1000, n_mw_paths=6,
n_earth_chords=12, N_steps_MW=25, N_steps_earth=100, earth_step_scheme="avg",
n_cores=4, verbose=0` for the four standard points:

  - `below`:  `m_Zd_eV=3.0e5, g_D=1e-4, epsilon_prime=0.1`
  - `inband`: `m_Zd_eV=2.0e6, g_D=1e-4, epsilon_prime=0.1`
  - `strong`: `m_Zd_eV=2.0e7, g_D=1e-2, epsilon_prime=0.1`
  - `cosmoX2`: `m_Zd_eV=40.0, g_D=1e-4`

Acceptance for arithmetic-preserving changes: `below`/`inband`/`strong` at the
fp floor (cross-run reproducibility floor is ~2e-12–7e-12; `strong` additionally
has a known ~7.5e-8 common-mode conditioning floor under ANY last-ulp reorder,
ratios ≤1e-12 — do not chase). `cosmoX2`'s Earth transit is in the phase-noise
regime: judge it on **energy-averaged flavor ratios and dip locations only**
(a last-ulp perturbation measurably reshuffles its pointwise fluxes at 1e-3).
Under the default `avg` Earth scheme, results carry a step-count convergence
certificate (N = 8…1000 agree ≤ 5e-9 at below/inband/strong).

**Paid-for numerical lessons** (respect them):
* fastmath FMA contraction is **compilation-unit-dependent**: two textually
  identical numba kernels measured bit-different (3.7e-16). Restructuring any
  spatial kernel breaks bit-identity and lights up cosmoX2. Do not restructure
  `dp_spatial`/`dp_propagator` kernels for a speed-up unless the win is large and
  judged by the ratio metrics.
* The user's hard constraint: **no precision or accuracy sacrifice** (sub-percent
  must hold everywhere it holds today). Accuracy-trading knobs (chord/path
  counts, grid sizes) are the user's call, not yours — you may *measure and
  present* such trades, never silently apply them.
* Global thermal-grid rescaling across couplings is physically invalid at
  resonance cells (Γ there is coupling-independent); the LOCAL verified
  two-component cache (`dp_thermal.thermal_grid_cached`) is the correct form and
  already landed. Don't re-derive either conclusion.
* Benchmark multiprocessing under `spawn`, never `fork` (a 10x regression once
  shipped from fork-only timing).

---

## Task 1 — Answer: can the potential textures (SM and LRI) take arbitrary forms?

Write the user a precise, evidence-backed answer (chat message; no code changes
required unless you find a genuine defect). The machinery to inspect:

* Config keys: `dark_portal.U_LRI_struct` and `standard_mixing.V_SM_struct`
  (solver JSON/CLI-less — JSON only), mirrored as `physics_constants.U_LRI_struct`
  / `V_SM_struct` in the analysis configs. Parser:
  `dp_config.parse_structure_matrix_3x3` — 3×3 nested lists, each entry a real
  number or `[re, im]`; **Hermiticity enforced** at 1e-12 relative (raises
  loudly otherwise, because the eigh-based/upper-triangle machinery would be
  silently wrong on a non-Hermitian input).
* Conventions: overrides are given in **neutrino convention**; the antineutrino
  transform (V → −V*) is applied centrally in `propagate_spectrum` (facade), and
  the spatial/cosmological paths share it (regression-tested including complex,
  CP-violating textures: `test_structure_override_*`,
  `test_parse_structure_matrix_3x3_formats_and_validation`,
  `test_run_solver_accepts_structure_overrides_in_config`).
* Known, documented physics limitation to state clearly: the override changes
  ONLY the coherent forward-scattering potential's flavor structure. The **Debye
  screening mass and the thermal attenuation/regeneration rates remain tied to
  the u_e/u_mu/u_tau amplitudes** (`compute_gamma_grid_inputs`); scanning u's
  with an override active triggers a loud warning in `likelihood.py`.
* Structural assumptions worth verifying before you answer "arbitrary":
  (a) the texture must be **constant per run** — no z-, E-, or position-dependence
  (the cosmological σ-table in `dp_cosmo` assumes H(σ) = M_rot + σ·S for ONE
  fixed S; V_LRI(z) = A_LRI(z)·S honors that by construction);
  (b) rank/definiteness are NOT restricted — the v3 propagator and
  `eigenvectors_3x3` handle any Hermitian H (the default rank-1 texture is the
  degenerate worst case, and the deflation/phase-noise machinery is
  texture-agnostic — but sanity-check an all-zero and a full-rank complex
  texture through a tiny run);
  (c) V_SM applies only inside the Earth when `use_SM_potential` is on, constant
  along each chord; default texture diag(1,0,0);
  (d) the GPU path and the `avg` step scheme thread the texture through
  (`dp_spatial`, `gpu_spatial`) — confirm by reading, and note the GPU device
  path is still unvalidated on hardware.
* Deliver: a direct answer ("yes, any fixed Hermitian 3×3 texture, with these
  caveats: …"), the exact JSON syntax with a worked complex example, and any
  edge case you actually tested.

## Task 2 — Deep speed-up and improvement review: `dark_portal_scanner.py` + `FrequentistAnalysis/`

Note: there is no `FrequentistAnalysis.py` — the target is the package:
`FrequentistAnalysis/src/main.py` (1D/2D profile scans), `run_limits.py`
(exclusion planes), `engines/scanner.py` (warm-started segment grids),
`engines/optimizers.py` (Minuit inner fits, MCMC, sanity check),
`topology/repair.py` (spike/continuity/gradient repairs),
`core/likelihood.py` (`GenericLikelihoodWrapper`), `physics/kernel.py`
(`compute_user_model_single` → solver).

**Measure first.** Start by re-profiling one likelihood evaluation (per-phase:
thermal build / cosmo z-loop / MW / Earth / detector likelihood + priors) and one
small real 1-D scan (evaluation counts per point, repair-pass counts, wall time),
single-core AND pooled, under `spawn`. The last full profile predates the `avg`
scheme; the balance has shifted (Earth phase shrank ~5x).

**What already landed — do not redo:** Minuit `strategy=0` for inner fits (62%
fewer evals; `scan_settings.minuit_strategy`), warm-started same-row scan
segments, spawn-worker JIT prewarm initializers, persistent pools (analysis +
standalone), likelihood cache REMOVED (9.3%/~0% hit rates — do not resurrect a
naive one), two-component thermal cache (worker-local; hits after ~3 evals at
fixed m_Zd), scanner defaults 12 paths/50 chords, semi-analytic thermal kernel,
`avg` Earth scheme.

**Concrete leads worth evaluating** (verify by measurement; some may be duds):
1. **σ-table reuse in `dp_cosmo`**: `_build_sigma_projection_table` (4096
   eigenvector solves) depends only on (M_rot, S-direction); for g_D/ε′ scans at
   fixed mixing the table is identical up to its σ-range. A range-checked memo
   (same key discipline as `thermal_grid_cached`) could remove it from every
   evaluation where only couplings move. Measure its actual cost first.
2. **Inner-fit economics** (`optimizers.py`): `m_scan.tol = 0.1` and
   `minuit_retries=3` are unmeasured knobs; the retry jitter uses
   `np.random.seed(os.urandom)` — irreproducible fits (an *improvement* target:
   seedable determinism, e.g. a config-driven base seed, without changing
   default behavior). Also: `use_simplex` is off for 1D scan segments but on for
   repairs/2D second attempts — is simplex ever worth its cost now that
   strategy=0 landed?
3. **AMR interplay** (`scanner.py`): `n_points_internal=100` internal evaluations
   PCHIP-interpolated down to `n_steps=40` — is 100 still right when each
   evaluation is 0.25 s pooled? Are repair passes re-running points the AMR then
   discards?
4. **Repair-pass budgets** (`repair.py`): `max_repair_passes=10` with full-grid
   rescans per pass; measure how many passes/points real scans actually consume,
   and whether the pass loop can early-out smarter without changing results.
5. **2D scans / `run_limits`**: segment sizing per row vs pool occupancy at
   20×20; checkpointing granularity (per-plane only — a crashed plane loses
   everything; per-row checkpoints are an improvement candidate).
6. **Scanner (`dark_portal_scanner.py`) improvements**: no checkpointing at all
   (a killed scan loses all points); `SCANNABLE` is hardcoded to 4 parameters
   (no `epsilon_prime`, `u_tau`); per-point manifests absent; results stream
   into memory then sort (fine) — and the pool warmup runs a tiny solve per
   worker even under fork where it's redundant (harmless, but measure).
7. **Detector likelihood + priors** (`core/`): presumed negligible — confirm
   once with the profiler and then stop thinking about it.
8. **HPC angle**: everything above is desktop-measured; note in your findings
   which wins scale with worker count (prewarm, checkpointing, pickling volume
   of `static_data` per task) — the user runs SLURM campaigns.
9. **The JAX track is out of scope** for this session (a separate self-contained
   plan exists at `docs/dev/JAX_BACKEND_BRIEF.md`); mention it only where a
   finding changes its calculus.

**Deliverables**: per finding — measurement, implementation (or documented
rejection with numbers, as this campaign has done six times), tests where
behavior is pinned, 4-point validation where physics is touched, README/config
docs where user-facing, commit-per-item in the established detailed style,
everything pushed to `dev`. Close by deleting this brief file in the final
commit and reporting fresh benchmarks against the §0 numbers.
