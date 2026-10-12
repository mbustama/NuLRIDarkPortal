# Work plan: event rates, systematics, statistics, Tycho

Agreed with Mauricio between 2026-07-29 and 2026-10-12. Each item lists the decision and, where
it was not obvious, why. Nothing below is implemented yet unless marked **done**.

## Goal

Turn the dark-portal fluxes into predicted IceCube event distributions (reconstructed energy,
per event class), with realistic detector response, backgrounds and systematics; run the heavy
work on Tycho; show the statistical analysis is sound with coverage tests.

## Order of work

0. **Solver corrections and the new analysis** -- the solver part is **done** 2026-10-12: three
   errors present since the first commit (runaway cosmological cascade, `7dc95e9`; resonant
   absorption independent of the couplings, `944360d`; non-unitary spatial propagator,
   `2066f3b`; details in the notes' corrections appendix). Every analysis result before them
   is void. **In progress:** the new analysis package `analysis/` (replaces
   FrequentistAnalysis; chi^2 bit-identical to it on all 35 flavor surfaces), then removing
   FrequentistAnalysis from code, tests, docs and outputs. Validation now includes absolute
   checks (energy and number at Earth <= the no-interaction run, total conservation through the
   spatial phase, u -> 0 recovers standard propagation), not only old-vs-new comparisons.
1. **Core-count fix** -- **done** 2026-10-12: `dp_core.available_cores()` (affinity mask capped by
   `SLURM_CPUS_PER_TASK`) and `default_n_cores()` (all but one core locally, the whole allocation
   under SLURM) now size every pool, FrequentistAnalysis included. Original note: the solver side
   sized its pools with `cpu_count()`, which on a SLURM node
   reports the whole node (128 cores) regardless of the allocation: `DEFAULT_CONFIG
   execution.n_cores`, the clamp in `dp_spatial.propagate_earth_spatial`, `dark_portal_scanner`,
   and `tools/convergence_study.py`. Use `SLURM_CPUS_PER_TASK`, then `os.sched_getaffinity`, then
   `cpu_count`. No change on the laptop. (FrequentistAnalysis already uses `sched_getaffinity`.)
2. **Tycho setup**, following GRBsim and NuFlavorSingleSource: clone over SSH to
   `/lustre/astro/mbustama/NuLRIDarkPortal`, conda environment on lustre pinned to the laptop's
   versions, `slurm/common.sh` plus job scripts, logs in `logs/`. First job: the four-point
   validation against laptop results, judged by the usual cross-machine rules (fp floor at
   below/inband/cosmoX2, the ~7.5e-8 floor at strong; cosmoX2 has been as reproducible as the
   others since `2066f3b`).
   Partitions `astro3_devel` / `astro3_short` / `astro3_long`; 16 cores by default, up to 32 per
   job, more only on request. Installs and large downloads go on Tycho, not the laptop.
3. **Per-direction solver output.** The solver returns one flux, the average over up-going
   Earth chords. The detector responses depend strongly on direction (DR2: straight up-going is
   ~20x below horizontal at 1 PeV), and starting events also come from above. Return the flux
   per zenith bin and add down-going chords (their path is only the overburden; the geometry code
   already handles them). Per bin, not per chord: a single chord's flux carries its own random
   Earth phase, so each bin needs the phase-aware integral of `average_earth_chords` restricted to
   its range in cos(zenith) (the same weights, computed on the bin's sub-range).
4. **Northern-Hemisphere muons** (through-going tracks).
   - Response: IceCube's IceTracks-DR2 release (2008-2022; doi:10.7910/DVN/MMIIZA): per-season
     effective areas (IC40, IC59, IC79, IC86) and 5D smearing tables -- per true (energy,
     declination) bin, fractional counts over (reconstructed muon energy, PSF, AngErr); 41
     true-declination bins against DR1's 3. About 3.4 GB of smearing tables, on Tycho only.
   - Fold with PLEnuM (pinned dependency, unmodified): our adapter turns DR2 into PLEnuM-format
     grids; PLEnuM supplies the fold, the multi-detector rotation and the MCEq atmospheric
     background. Sum over seasons with their own responses and uptimes.
   - Cut on reconstructed declination; turn PSF into declination migration geometrically.
   - Muon neutrinos only, as in DR1 and PLEnuM.
5. **Starting events** (cascades, starting tracks, double cascades).
   - Response: the HESE 7.5-year public per-event MC (github.com/icecube/HESE-7-year-data-release),
     the only public per-event MC with all three morphologies: weights per event with flavor
     (nu and nubar separate), interaction (CC/NC/Glashow), reconstructed deposited energy and
     morphology; per-event atmospheric fluxes and self-veto factors.
   - Multi-detector: IceCube-Gen2 TDR effective areas (doi:10.7910/DVN/XANWAT), as Gen2/IceCube
     ratios per channel, flavor, nu/nubar, energy and zenith. The tables have no smearing, so
     IceCube's HESE response supplies it.
   - Not used: the MESE 11.4-year release (binned only, no MC; its smearing is averaged over
     flavor), and the 11-year cascade analysis (proceedings only, no release).
6. **Systematics** (baseline), wired in as the channels are built:
   - astrophysical normalization and spectral index;
   - source flavor composition `f_e,S` with `f_tau,S = 0`;
   - atmospheric: conventional normalization, cosmic-ray tilt, and the 12 Barr parameters used
     by NuFlavorSingleSource (MCEq, SIBYLL-2.3d x H3a; pion regions G, H, I and kaon regions W,
     Y, Z, each charge; Barr et al.'s energy-dependent priors); prompt and atmospheric-muon
     normalizations; self-veto;
   - mixing parameters varied, all six, constrained by NuFIT 6.1's joint Delta chi^2 (the
     analysis with Super-Kamiokande and IceCube atmospheric data, "TByes", per ordering; the
     release GeoneutrinoDecay already holds in `input/mix_params/nufit_6.1_release/`). NuFIT
     publishes every 1D and 2D projection and one 3D table, so each parameter goes into
     exactly one table: the 3D (sin^2 theta_23, Delta m^2_31, delta_CP) table, where the
     octant and CP correlations live; the 2D (sin^2 theta_12, Delta m^2_21) table; and the 1D
     sin^2 theta_13 table. The surfaces are interpolated, not replaced by Gaussians, as in
     GeoneutrinoDecay. The correlations this partition neglects (between tables) are checked
     from NuFIT's other 2D projections. The inverted-ordering tables do not reach zero; their
     floor is NuFIT's evidence against that ordering and is kept. The 2050 projection keeps
     projected Gaussian widths. First fix the three bugs in the configs' `fallbacks` (the code
     matches `"gauss"` but the configs say `"gaussian"`; `dcp`/`dm3l_sq` don't match
     `d_cp_deg`/`dm31_sq`; the centres disagree with the fixed values);
   - lightest neutrino mass free, with a half-Gaussian prior on the sum of the masses, as in
     GeoneutrinoDecay: chi2 += (Sum/sigma)^2 above zero, sigma = limit / Phi^-1(0.95).
     Limit: Sum < 0.055 eV at 95%, CMB (Planck + ACT + SPT-3G) + DESI DR2, LCDM, prior from
     zero (SPT-3G D1, arXiv:2506.20707, PRD). sigma = 0.0334 eV. Both orderings are run; the
     prior disfavours inverted ordering (penalty 8.85 at its minimum sum, 3.08 for normal).
   - **Not** included: cross sections (the public responses fix them; floating them would mean
     generating our own effective areas), and the detector-systematic splines (DOM efficiency,
     hole ice, ice anisotropy; deferred, they need the PHOTOSPLINE package).
7. **Statistical soundness**, the GeoneutrinoDecay sequence:
   1. closure: reproduce a published IceCube fit with each response (HESE ships its own
      power-law fit) before any new-physics fit;
   2. injection and recovery on Asimov data, with and without a dark-portal signal;
   3. fits with the current profile-likelihood procedure;
   4. coverage tests with pseudo-experiments (Poisson-fluctuated counts, fitted exactly like the
      data, 200 per point for +-2% on a 10% rate) at: no signal (the test statistic against
      Wilks/Chernoff), the expected limit and half a decade deeper, parameter boundaries
      (couplings, lightest mass near zero, `f_e,S` at 0 or 1), where the mediator mass and the
      lightest mass trade off in setting the resonance energy, and low counts (double cascades,
      highest-energy tracks). Toys change only the data, so the solver predictions on the scan
      grid are computed once and reused;
   5. decide: no under-coverage, keep the profile likelihood; under-coverage, Feldman-Cousins
      with PyFC (github.com/mbustama/FeldmanCousins), whose L_Eff likelihood also handles the
      finite statistics of the HESE MC.

   Coverage also settles the limit thresholds: `run_limits.py` uses delta-chi2 = 2.71 / 3.84 for
   90% / 95%, the two-sided values; for an upper limit on a coupling bounded at zero the
   one-sided values are 1.64 / 2.71. The present flavor-ratio analysis has no events to fluctuate
   (its data term is a projected measurement), so its pseudo-experiments shift the projected
   measurement's centre by its own width -- an approximation to state with the results.

## Output conventions

Absolute event rates; livetime kept as a scalar applied after the fold (every response is
linear in it), so it can float. Backgrounds included from the start.

## Known approximations, to state with results

- Flavor evolution (solver) and Earth absorption (responses) are applied as separate factors
  along each path; exact if collisions remove all flavors equally, broken slightly by the
  Glashow resonance, nu_tau regeneration and NC scatterings that continue at lower energy.
- Gen2 starting events use IceCube's smearing.
- DR2 events must not be combined with other IceCube releases in a joint data fit (overlap).
- Under a dominant potential, the vacuum splitting inside the potential's degenerate subspace is
  below double-precision resolution and evolves as zero: <= 5e-6 in flavor populations across
  the Earth's diameter at 10 TeV, 5e-8 at 100 TeV (`2066f3b`).

## Done

- 2026-10-12: solver corrections. Cascade (`7dc95e9`): the regenerated source was held at its
  start-of-step value while Gamma*dt reached 1e10, so the energy at Earth was up to 1e67 times
  the no-interaction value; now an exact per-step sweep that conserves number. Absorption
  (`944360d`): the Breit-Wigner numerator carried one coupling, so the resonance's strength did
  not depend on the couplings; now Gamma_in*Gamma_tot (decided with Mauricio), rates x
  Sum G/(6 pi^2). Propagator (`2066f3b`): the divided-difference polynomial was unitary only to
  ~eps*spread*min(dr, 2/gap) (0.7% of the flux created or destroyed at cosmoX2); now an exactly
  unitary spectral sum, and the numba cache is keyed by a digest of src/ (cross-module callers
  had kept stale machine code). Production record: energy (number) at Earth / no interaction
  0.99898 (0.99805) below, 0.84068 (0.95504) inband, 0.98239 (0.99993) strong, 0.99934
  (0.99883) cosmoX2. Solver defaults 0.71 s on 8 cores, 1.04 s on one (0.80 / 0.84 s before
  the corrections; the July figure of 0.32 s is not reproduced -- to investigate).

- 2026-10-12: relic-neutrino temperature fixed, 1.95e-4 -> 1.676389e-4 eV (`5ecf10e`); spectra
  and event rates change materially, energy-averaged flavor fractions by <= 3.8e-4.
- 2026-10-12: chord-sampling floor removed: the Earth chords are combined with the phase-aware
  zenith average (`grid.earth_zenith_average = "phase"`), which integrates each block of their
  density matrices against its dark-portal phase over cos(zenith). 12 chords agree with 50 to
  <= 1.4e-7 at below/inband/strong (was ~1e-3-4e-3 between 12 and 13); fractions move by up to
  0.029. Residual ~6e-4 at cosmoX2 (then attributed to precision in the phase-noise regime; it
  was the non-unitary propagator, gone since `2066f3b`). The analysis must be rerun; its
  200-step redshift grid (~1.1e-2) is now its largest numerical error.
- 2026-10-12: Earth chords start exactly on the surface (`get_baseline_earth` had dropped the d^2
  terms); the Earth's own potential is its spherical Yukawa integral (Magnus/NuOscProbExact
  method), bit-identical at the benchmark points, 7-8% on the Earth effect at ranges >~ 100 km.
- 2026-10-12: `N_steps_earth` default 100 -> 25 (`95fef9f`); scan/plane checkpointing,
  global-fit checkpoint, `min_segment_length`, `minuit_maxcall`, thermal-cache fix (round 8).
