# Brief: Speed-up round 7 — implement findings #1–#10 (except #8) in NuLRIDarkPortal

You are picking up a long-running performance/correctness campaign on this repository.
This brief is self-contained: read it fully before touching code. It specifies nine
implementation tasks, one explanation task, the validation discipline that is
**mandatory** for every change, and the pitfalls previous rounds hit so you do not
repeat them.

---

## 0. Repository state and environment

* Repo: `/home/mbustamante/Research/NuLRIDarkPortal`, branch `dev` (tracked on
  `origin/dev`, GitHub `mbustama/NuLRIDarkPortal`). HEAD at the time of writing:
  `0b98376`. Commit locally as you go, push to `origin dev` when a task lands (the
  user has standing authorization for pushes to `dev`). Never merge to `main`.
* Test suite: `python3 -m pytest tests/test_physics_regression.py` — currently
  **44 passed + 1 skipped** (the skip is a GPU test on this CUDA-less machine). It must
  stay green after every task. The first run after touching numba kernels recompiles
  for ~2 minutes; warm runs take ~25 s.
* Environment: 12-core Linux desktop, anaconda Python 3.12, numba 0.60 / llvmlite
  0.43 (conda build **without** SVML support — `llvmlite.binding.has_svml()` is
  False; `icc-rt` is already pip-installed but inert). `iminuit` installed. **No CUDA
  device and no cupy** — GPU code can only be written and CPU-fallback-tested here.
  `tbb` is NOT installed (relevant to task #4). `FrequentistAnalysis/src/main.py`
  forces the `spawn` multiprocessing start method — every multiprocessing-related
  timing/behavior claim must be checked under `spawn`, not the OS default `fork`
  (a previous round shipped a 10x slowdown by benchmarking under `fork`).
* Current benchmarks to compare against (8 cores unless stated):
  - Solver defaults (`src/dark_portal_solver.py`, 1000×1000 grid, 12 chords,
    25 MW steps): **6.23 s** full run.
  - Profile of that run: thermal-rate grid **3.7 s (59%)**, spatial handover
    **1.13 s/flavor (36%)**, cosmological z-loop **0.18 s/flavor (6%)**.
  - FrequentistAnalysis likelihood evaluation
    (`config/pion_2050_case_5_fixed_nuisance.json`, 200×200 grid, 50 paths/50
    chords): **6.14 s** single-core (scan-worker mode), **1.58 s** in the pooled
    main-process mode.
* There is a persistent memory directory whose summaries agree with this brief;
  where they conflict, this brief wins.

## 0.1 Architecture in five sentences

`src/dark_portal_solver.py` computes neutrino flavor fluxes: a cosmological phase
(z-loop over `step_cosmological_propagation`, with a precomputed thermal-rate grid
`precompute_all_thermal_rates` and mixing projections from a σ-table
`_build_sigma_projection_table`), then a spatial handover
(`propagate_earth_spatial`: Milky Way paths + Earth chords, each evolving the
density matrix per energy bin with `propagator_3x3` inside
`_evolve_rho_all_bins`, distributed over a process pool as (unit, E-block)
tasks). `src/dark_portal_scanner.py` runs 1-D parameter scans of the solver;
`src/flux_postprocessing.py` holds the energy-averaged flavor ratios.
`FrequentistAnalysis/` wraps the solver in an iminuit profile-likelihood pipeline
(`src/core/likelihood.py` → `src/physics/kernel.py` → solver), with parallel scan
engines (`src/engines/scanner.py`, warm-started segments), repair passes, and a
limits module (`src/run_limits.py`). Configuration flows from a single
`DEFAULT_CONFIG` in the solver with unknown-key rejection
(`merge_config_with_defaults`); FrequentistAnalysis mirrors it through a
34-element `static_data` tuple built in `likelihood.py`.

## 0.2 The validation discipline (non-negotiable)

Every change that can affect physics output must be validated **against the real
pipeline at production settings across multiple dark-portal parameter points**,
comparing the **max** relative difference over the **full** output spectrum plus the
flavor ratios — never a synthetic state, never a single bin, never the median.
History that motivates this: a step-count optimization validated on a synthetic
diagonal state silently produced a ~10% energy-localized error with the real
coherence-carrying state (see the long comment in `propagate_earth_spatial`); a
divided-difference propagator passed all unit tests yet failed end-to-end at
1.3e-4 (two separate root causes, both now documented in `propagator_3x3`'s
docstring); two thermal-grid interpolation schemes passed plausibility arguments
and failed validation (see the NOTE above `precompute_all_thermal_rates`).

**Standard validation recipe** (rebuild this; previous scratchpad scripts are gone):

1. `git worktree add --detach /tmp/<somewhere>/wt_ref HEAD` → the reference code.
2. A small runner script that, given a src dir and a parameter-point name, runs
   `dps.run_solver` at production settings — `n_E_bins=1000, n_z_bins=1000,
   n_mw_paths=6, n_earth_chords=12, N_steps_MW=25, N_steps_earth=500,
   n_cores=4, verbose=0` — with `dark_portal.update(point)` and saves `E, F` to
   `.npz`. Run reference and modified code in **separate processes** (same-process
   dual imports produce module-cache artifacts).
3. Parameter points:
   - `below`:  `m_Zd_eV=3.0e5, g_D=1e-4, epsilon_prime=0.1`
   - `inband`: `m_Zd_eV=2.0e6, g_D=1e-4, epsilon_prime=0.1`
   - `strong`: `m_Zd_eV=2.0e7, g_D=1e-2, epsilon_prime=0.1`
   - `cosmoX2` (cosmological level-crossing in range): `m_Zd_eV=40.0, g_D=1e-4`
4. Acceptance: `below`/`inband`/`cosmoX2` at the fp-noise floor (≤ ~4e-11 max
   relative flux). **Known conditioning floor**: the `strong` point responds to ANY
   last-ulp arithmetic reordering with ~7.5e-8 max relative flux, always
   common-mode across flavors at the lowest-E resonant bin, with ratios ~1e-12.
   This has been confirmed three times; do not chase it. Ratios there must stay
   ≤ ~1e-12.
5. For genuinely approximate changes (quadrature swaps etc.), additionally compare
   the intermediate quantity directly (e.g. the Γ grid) and use the energy-averaged
   flavor ratios as the physical acceptance metric.

**Numerical lessons already paid for** (respect them):
* `fastmath=True` enables LLVM's approximate-function flag: transcendentals at
  large arguments (phases here reach 1e10 rad) degrade to ~1e-7. `propagator_3x3`
  deliberately compiles without fastmath. Never benchmark a trig-heavy kernel's
  accuracy only at small arguments.
* Closed-form cubic roots are √ε-conditioned at double roots (a measured
  close-pair gap came out 257× too large). `propagator_3x3` handles this by
  deflation; any new spectral code must too.
* Interpolation cannot reproduce point samples of a sub-grid-width feature: the
  Breit–Wigner resonance in the thermal rates is narrower than any practical grid.
* Coefficients built from transcendental identities must interpolate the
  *computed* exponentials exactly at the nodes ("node-exactness"), or phase
  rounding becomes per-step non-unitarity.

---

## The tasks

Recommended order: **#6 → #3 → #10 → #9 → #1 → #7 → #5 → #2 → #4**, then the #8
explanation. (#6/#3/#10 are quick wins that de-risk the rest; #1 is the big one;
#2 and #4 are riskier and self-contained.) Commit per task with the detailed
commit-message style used throughout this repo (what/why/validation evidence).
Update `README.md` where behavior or guidance changes.

### #1 — Analytic momentum integration of the resonant thermal term (largest win)

**Problem**: `thermal_averaged_rates_flavor` (src/dark_portal_solver.py) evaluates,
per (E, z) grid point and per CνB mass state, a 100-point midpoint rule over
momentum p ∈ [0, 20 T_i] of
`p² f_FD(p/T_i) · I_ang(A(p), B(p), M2, MG)` with `A = m_i² + 2 x E_bg(p)`,
`B = 2 x p`, `x = E(1+z)`, `E_bg = sqrt(m_i² + p²)`, and `I_ang` the existing
closed-form angular integral (2/B term + log term + arctan pair with width MG).
At the production 1000×1000 grid that is 3×10⁶ momentum loops = **59% of the
solver run (3.7 s)** and the dominant cost of every FrequentistAnalysis
evaluation's grid build.

**Approach**: add a second closed-form layer. Split `I_ang` into (a) the resonant
arctan-pair contribution, which for narrow width behaves as ≈ π·F/(B·MG) inside
the kinematic window `A − M2 ∈ [−B, +B]` and decays outside, and (b) the smooth
remainder (2/B, log term, non-resonant tails). Integrate (a) over p
**analytically**: within the window, change variables to u = A − M2 (with
dA/dp = 2x·p/E_bg, dB/dp = 2x locally linear or exactly integrable for
E_bg = sqrt(m²+p²)); the arctan pair integrates in closed form under a
locally-linear (or piecewise-exact) treatment of A(p), B(p), exactly in the
spirit of the existing angular closed form. Integrate (b) with a small
Gauss–Legendre rule (16–24 nodes) — it is smooth by construction.

**Constraints and checks**:
* Exact linear scaling of Γ with `sum_G` in the narrow-width regime must be
  preserved: currently verified numerically (Γ ratio exactly 0.1000 per decade of
  g_D down to 1e-8). Re-verify after the change.
* The wide-width regime (g_D → O(1), sum_G ~ 1) must remain correct: there the
  "resonant" term is broad and the split must still sum to the exact `I_ang`
  (the split is algebraic, only the *integration method* differs per piece).
* Keep `precompute_all_thermal_rates` (the current exact quadrature) available as
  the reference implementation; add the new path as the default with a config
  switch back (follow the pattern of `n_z_thermal`, and read the NOTE above
  `precompute_all_thermal_rates` — it documents two prior *rejected* fast schemes
  and why; your scheme is the "treat the resonant spike analytically" path that
  NOTE calls for).
* Regression tests: `test_angular_integral_closed_form_matches_quadrature` and
  `test_thermal_averaged_rates_matches_reimplementation` pin the current kernel.
  Add an analogous independent-reimplementation test for the new integral
  (closed-form vs brute-force dense quadrature of the resonant piece over p, with
  quadrature told where the window is), across a g_D ladder spanning 1e-8 … 1.
* Grid-level acceptance: max relative Γ difference vs the exact grid where rates
  are significant, across the four validation points; then the full 4-point
  pipeline validation. Expect the flux dips to be sensitive — judge with the
  ratio metric plus dip-location sanity.
* Expected gain: thermal phase 3.7 s → ~0.1–0.3 s; solver defaults → ~2.5–3 s;
  analysis evaluations shed ~1–1.5 s each.

### #2 — Port the GPU spatial path to the v3 propagator

`src/gpu_spatial.py` still implements the **old** batched Lagrange propagator with
a masked batched-eigh fallback. The CPU propagator has since been replaced
(`propagator_3x3` v3) and the old scheme is both slower (the fallback mask fires
for ~100% of Earth evolutions at default parameters — rank-1 LRI texture makes H
doubly-degenerate wherever the potential rivals the vacuum term) and numerically
inconsistent with the CPU path. Port v3 exactly, batched over the (unit, E-bin)
axis with cupy:

* Traceless shift D = H − qI, global phase factored out.
* Mask 1: `p ≤ 1e-14·scale` → pure phase.
* Mask 2: `|spread·t| < 1.2e-4` → second-order Taylor `e^{-iqt}(I + X + X²/2)`.
* Closest-pair ordering; **node-exact branch**: sinc form only where
  `|gap·t/2| < 0.5`, direct quotient `(f_a − f_b)/(a − b)` otherwise.
* **Deflation mask**: trig pair-gap `< 1e-4·spread` → far eigenvector from the
  largest column of `(D − mean_pair I)²` (batched), Gram–Schmidt complement,
  pair mean/half-gap from the 2×2 block (`delta = sqrt(((b11−b22)/2)² + |b12|²)`).
* complex128 throughout; no fastmath analog concerns on GPU but avoid low-precision
  intrinsics.

Read `propagator_3x3`'s docstring first — it records why each branch exists.
You cannot run GPU numerics here: keep the guards and CPU-fallback behavior
(`test_use_gpu_falls_back_to_cpu_without_cuda` must pass), keep
`test_gpu_spatial_matches_cpu` as the skipped-on-CPU validation gate, and state
the validation status honestly in the module docstring and README (the user will
run `pytest -k gpu` on their GPU machine).

### #3 — NaN guards at extreme mediator masses

Reproducible: `m_Zd_eV=1e-6, g_D=1e-4, epsilon_prime=0.1` at production settings →
**all-NaN flux**, in current and older code alike (pre-existing). First diagnose
the actual overflow/NaN origin (suspects: `MG` astronomically small →
`1/MG`-type terms in `I_ang`; `Coef`/Debye-mass expressions; `m_eff`-Yukawa
exponentials). Then:
* If the origin is benignly fixable (e.g. a guard on an intermediate), fix it and
  validate that the four standard points are bit-unchanged.
* Regardless, add a loud domain check: `run_solver`/`propagation_worker` should
  detect a non-finite Γ grid or non-finite final flux and raise a clear error
  naming the offending parameter values, instead of returning NaN silently.
* FrequentistAnalysis: verify end-to-end that a NaN-producing point yields the 1e9
  sentinel through `GenericLikelihoodWrapper`'s existing isnan → 1e9 path (write a
  small test with a tiny grid at the pathological point).
* The shipped configs allow `m_Zd` down to 1e-30 (see #9). Add a regression test
  pinning whatever contract you establish.

### #4 — Threads-based spatial parallelism, retried with TBB

A prange/threads spatial mode was implemented and **reverted**: with numba's
default threading layer it deadlocked the `spawn`-subprocess + persistent-pool
combination (420 s hang in
`test_frequentist_persistent_pool_reused_under_spawn_not_recreated_per_call`)
and segfaulted once locally. The known-good recipe to retry:
* `pip install tbb`, then force `numba.config.THREADING_LAYER = 'tbb'` (or the
  `NUMBA_THREADING_LAYER` env var) before the first parallel compile, in
  `dark_portal_solver.py`'s import-time setup. Verify the active layer via
  `numba.threading_layer()` after a parallel call.
* The reverted design (recreate it): `_evolve_rho_all_bins` gets
  `parallel=True` with `prange` over bins and **thread-local 3×3 scratch**
  (`np.empty` inside the loop body — the caller-supplied shared buffers would race);
  `propagate_earth_spatial` sets `use_threads = (external_pool is None and
  n_workers > 1)`, and in that mode skips process pools and E-block splitting
  entirely (tasks run sequentially; `numba.set_num_threads(n_workers)` carries the
  parallelism). The process-pool path must remain for `external_pool is not None`
  (FrequentistAnalysis persistent pool; its single-threaded workers make prange
  serial — identical arithmetic).
* Acceptance gates, all mandatory: full suite green including the 420 s-budget
  spawn/pool test; `test_spatial_E_block_parallelism_is_exact` (exact equality
  n_cores=1 vs >1); a NEW stress test that runs a parallel kernel and then
  creates/uses a spawn Pool in the same process; the 4-point pipeline validation
  (should be bit-identical — prange over disjoint bins is deterministic); and
  benchmarks showing real improvement (target: spatial scaling from 3.2× toward
  ~6–7× on 8 cores). **If TBB still misbehaves with spawn, revert again and
  instead persist the standalone solver's process pools** (currently created and
  destroyed per phase per flavor — 4 pools per run; a module-level persistent pool
  as in `FrequentistAnalysis/src/physics/kernel.py` is the fallback win).

### #5 — Propagator micro-optimizations (measure-first, low priority)

* Exploit Hermiticity in the final polynomial evaluation of `propagator_3x3`:
  D² of a Hermitian D is Hermitian — compute its upper triangle and mirror
  (~25% fewer multiplies in that loop). U itself is NOT Hermitian; only D² is.
* Optional experiment: SoA (separate real/imag arrays) layout for ρ in
  `_evolve_rho_all_bins` to unlock SIMD. Only adopt on a measured win at
  production shapes; keep the AoS path if inconclusive.
* Both must be bit-identical or fp-noise-level; validate with the chord-product
  check (compose 500 Earth steps, compare against `np.linalg.eigh` construction —
  see `test_propagator_3x3_*` tests for the pattern) plus one pipeline point.

### #6 — Scanner defaults from the convergence study (nearly free)

`src/dark_portal_scanner.py`'s CLI defaults ignore the repo's own convergence
study (`output/convergence/`, README §11): it scans with `n_mw_paths=100`
(study: ≥6 suffices at the studied points) and `n_earth_chords=100` (50 is the
sub-percent knee). Change the argparse defaults to `n_mw_paths=12` and
`n_earth_chords=50`, cite the study in the help strings, and note in the README
scanner section that users exploring very small m_Zd (where the galactic
potential matters) should raise `n_mw_paths` and re-run the convergence tool.
~2–3× per scan point. No physics validation needed beyond one smoke scan
(it is a resolution-settings change, documented as such).

### #7 — Thermal-grid reuse across coupling scans via exact linear scaling

Probe result on record: Γ scales **exactly linearly** with `sum_G` (ratio 0.1000
per decade of g_D down to 1e-8) at fixed (E, z, m_Zd) in the narrow-width regime —
because `MG = m_Zd²·sum_G/(12π)` enters `I_ang` only through width terms whose
integrated contribution is linear in the narrow-width limit. If that holds across
the full grid to high precision, coupling scans (u_e, u_mu, g_D at fixed m_Zd —
i.e. the scanner's u-scans and much of FrequentistAnalysis's profiling) can
compute the grid **once** at a reference `sum_G` and rescale per point.

Implementation with a built-in safety gate:
* In `dark_portal_scanner.run_parameter_scan` (and optionally as a solver-level
  helper), before a scan over a coupling parameter at fixed m_Zd/masses/T/eta:
  compute the exact grid at two couplings spanning the scan range; if
  `Γ(c₂)/Γ(c₁)` matches `sum_G₂/sum_G₁` element-wise to < 1e-6 everywhere rates
  are significant, enable rescaling for the scan; otherwise fall back to exact
  per-point grids. Log the decision.
* Note the interplay with #1: if #1 lands first, the thermal grid is already
  ~20–50× cheaper and #7's marginal value shrinks — implement #7 only if the
  verification machinery is cheap, and say so in the commit either way.

### #9 — Minuit tuning, worker prewarm, physical parameter windows

Three independent sub-items in FrequentistAnalysis:
1. **Minuit strategy for inner profiled fits**: in
   `src/engines/optimizers.py` (`minimize_point`, `minimize_point_2d`), set
   `m_scan.strategy = 0` (skip the expensive error-matrix refinements nobody
   consumes mid-scan; the repair passes catch quality regressions). Measure the
   per-point evaluation-count/time change on a small real scan (warm vs cold
   smoke-test pattern exists in git history: tiny physics grid, one 1-D scan,
   compare chi2 arrays and wall time). Make it configurable
   (`scan_settings.minuit_strategy`, default 0) and document.
2. **Pool worker JIT prewarm**: the scan pools (main.py, run_limits.py) spawn up
   to `num_cores` workers that each lazily load/compile numba caches on their
   first task — at 200 HPC workers this is a simultaneous cold-start stampede.
   Add a pool `initializer=` that imports `src.physics.kernel` and runs one tiny
   `compute_user_model_single` (10-bin grid) to warm each worker before tasks
   land. Same for the scanner's pool (it already warms the parent; extend to
   workers via initializer under `spawn`— note the scanner runs under the OS
   default start method, main.py under forced spawn).
3. **Physical parameter windows**: shipped configs allow
   `m_Zd_eV ∈ [1e-30, 1e10]` — 40 decades, much of it in the all-NaN domain
   (#3). Do not silently change the user's physics scope: add a
   `_comment_limits` note in both `pion_2050_*` configs documenting the NaN
   domain and the recommendation, add a startup warning in
   `GenericLikelihoodWrapper` when scan windows overlap the known-bad domain
   (below whatever threshold #3's diagnosis establishes), and let the user pick
   final windows.

### #10 — Drop (or trivialize) the likelihood cache

`GenericLikelihoodWrapper._cache` measured 9.3% hit rate in-process and ~0%
cross-process (each pool task pickles a fresh wrapper). Remove `_cache`,
`_cache_enabled`, `use_fuzzy_cache` and the MCMC cache-guard choreography in
`run_mcmc_warm_start` (`func._cache_enabled = False` etc.), OR replace with a
one-entry last-call cache if any code path relies on repeat-evaluation identity
(check `perform_sanity_check` and the emcee wrapper; neither should need it).
Also remove `use_fuzzy_cache` from `DEFAULT_SCAN_SETTINGS`/config docs/README.
Verify: full suite + one smoke likelihood evaluation + grep that no other module
touches the removed attributes.

---

## After implementing: explain #8 (JAX autodiff) — do not implement

Write the user a clear, technically concrete explanation of what porting the
likelihood to JAX for gradient-based fitting would entail. Cover at least:

* **Why**: Migrad estimates gradients numerically — O(2N) likelihood evaluations
  per iteration for N floating parameters, at ~6 s per evaluation. `jax.grad`
  delivers exact gradients for ~1 forward-cost, typically cutting profiled-scan
  cost 5–10× and improving convergence robustness (fewer repair-pass triggers).
* **What must be ported**: the whole differentiable path — thermal-rate grid,
  σ-table or direct mixing, the z-loop (as `jax.lax.scan`), the spatial transit
  (500-step chord products as `lax.scan` over steps with `vmap` over bins/chords),
  the propagator (its branches become `lax.cond`/`jnp.where` masks — the
  deflation and Taylor branches are exactly maskable, as the GPU port shows), the
  PREM piecewise density (`jnp.select`), energy-averaged ratios, and the detector
  likelihood (already vectorized numpy — trivial).
* **Technical hazards**: float64 must be enabled (`jax.config.update
  ("jax_enable_x64", True)`); complex differentiation (JAX handles it, but
  `abs()²` should be written as `real² + imag²` for clean Wirtinger gradients);
  non-differentiable branch boundaries (degeneracy masks) — gradients are still
  correct almost everywhere, but document the measure-zero seams; long `lax.scan`
  chains and reverse-mode memory (use `jax.checkpoint`/rematerialization for the
  500-step transit); compile times for the first trace; JAX vs multiprocessing
  (JAX is not fork-safe — the natural architecture replaces the process pool with
  `vmap`/`pmap` batching of grid points, which is itself a further win); and the
  validation burden (forward parity vs numba at fp tolerance on the standard
  4-point suite, then `jax.grad` vs high-order finite differences).
* **How it plugs into fitting**: iminuit accepts `grad=`; or move the inner
  profiled minimizations to `scipy.optimize` L-BFGS-B/trust-constr with exact
  gradients; Migrad stays as the orchestrator if preferred.
* **Phasing**: (a) forward-parity JAX port of the kernel behind a flag,
  (b) gradient correctness harness, (c) `grad=` wiring + benchmark on one real
  1-D profile scan, (d) decision point before any wider adoption.
* **Honest cost estimate**: this is a multi-session project (the thermal integral
  and propagator are each a careful day, validation likewise), with a real payoff
  only for large scan campaigns.

## Final deliverables checklist

1. Tasks #6, #3, #10, #9, #1, #7, #5, #2, #4 implemented (or, where a task's own
   acceptance gate fails — as has legitimately happened twice in this campaign —
   reverted with the negative result documented in the commit message and README).
2. Suite green (expect 44+ tests; add regression tests per task as specified).
3. 4-point validation table reported for every physics-touching task.
4. Fresh benchmarks: solver defaults, analysis eval single-core and pooled;
   compare to §0 numbers.
5. README and config comments updated; this brief file may be deleted in the
   final commit of the round.
6. The #8 explanation delivered as the closing message.
7. Everything committed on `dev` and pushed.
