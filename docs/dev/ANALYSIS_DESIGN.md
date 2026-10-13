# The analysis package (`analysis/`)

Replaces FrequentistAnalysis (retired 2026-10-12, at Mauricio's request: "abandon
FrequentistAnalysis everywhere ... if you can write more suitable, optimized code"). The design
follows GeoneutrinoDecay's: a small package on top of the solver, one JSON configuration per
analysis, a command line, checkpoints, provenance.

## Layers

| Module | Role |
|---|---|
| `model.py` | Named values -> solver configuration -> fluxes -> energy-averaged flavor fractions. Calls the solver's own `build_solver_params` / `propagate_nu_nubar`, so its defaults and checks stay the single source of truth. Caches solver runs (exact keys, LRU). The source composition, the nubar/nu ratio and the normalization need no new run: the fluxes are linear in them (checked to 1.6e-15). |
| `surfaces.py` | The 35 flavor-composition likelihoods (measurements and projections) moved from FrequentistAnalysis, bit-identical in ln L. chi^2 = -2 ln(L/L_max), zero at the centre. `recentred()` moves a surface rigidly (Asimov data, pseudo-experiments). |
| `parameters.py` | Free parameters in linear or log10 internal coordinates (steps of order one); Gaussian and half-Gaussian priors, on parameters or derived quantities (sum of masses). |
| `objective.py` | chi^2 = data + priors. Sterile mixing beyond unitarity is evaluated on the unit sphere plus a continuous wall (FrequentistAnalysis returned finite chi^2 there). |
| `fitting.py` | L-BFGS-B from several starts (seeded random starts in the bounds), then a Powell polish. |
| `scans.py` | Global fit and profiles (1D) / planes (2D) in a spawn pool. Chains walked outward from the best fit, each point warm-started from its predecessor and the best fit; repair passes refit points whose neighbours' solutions beat them, points sticking out above their neighbours, and (once) edge points. JSONL checkpoints; resume skips finished points. |
| `intervals.py` | Intervals at the configured thresholds from Delta chi^2, interpolated in log10 for log parameters; open ends flagged; failed points skipped. |
| `run.py` | `python -m analysis CONFIG.json [--fit] [--profiles ...] [--planes] [--asimov] [--workers N] [--summarize]`; summaries measure Delta chi^2 from the lowest chi^2 found anywhere; `manifest.json` records code version, configuration digest, machine and timing of every run. |

## What changed relative to FrequentistAnalysis

- **Solver**: three errors present since the first commit were fixed first (notes,
  Appendices "cascade", "bwnorm", "unitarity"). Every FrequentistAnalysis result predates them.
- **Mixing**: fixed at NuFIT 6.0 NO with SK (the values the 2050 projection is centred on);
  FrequentistAnalysis used delta_CP = 132 deg etc., 0.7 sigma off the centre in f_mu, which a
  fit could only absorb with new physics.
- **Priors**: applied (FrequentistAnalysis' "fallbacks" never were: shape name, parameter names).
- **Thresholds**: 1D 1/2.706/3.841, 2D 2.296/4.605/5.991, labelled 68.27/90/95%
  (FrequentistAnalysis labelled 4.61 as 2 sigma). **Exclusion planes are read per mediator
  mass** (decided by Mauricio, 2026-10-13): a plane with `"raster": "m_Zd_eV"` measures
  Delta chi^2 in each mass column from that column's own minimum and excludes above 3.84 (95%,
  one parameter); columns whose minimum lies above the best fit by more than that are flagged,
  since their coupling range never reached the region without an effect. Coverage tests still
  have to confirm the threshold (one-sided 2.71 would be the asymptotic value for a coupling
  bounded at zero).
- **Scan ranges**: explicit in the configuration (FrequentistAnalysis scanned +-4 sigma around
  the best fit, so its m_Zd profile covered only 1.2-2.8 MeV).
- **Resolution**: 400 energy bins x 200 redshift steps, 6 Milky Way paths, 12 Earth chords
  (2026-10-12, corrected solver: <= 1.5e-5 on the averaged fractions at inband, 2e-4 at
  m_Zd = 1e-2 eV; chords and paths converged to 1e-7). FrequentistAnalysis used 200 x 200 with
  50 and 50.

## Findings to keep in mind

- **Null data and free couplings**: the 2050 projection is centred on standard physics, and
  with every coupling free the fit can always switch the new physics off, so 1D profiles stay at
  Delta chi^2 ~ 0. Exclusion regions need fixed couplings or textures
  (`configs/pion_2050_case_5_exclusion.json`: e-mu texture, u = 0.5).
- **Degeneracies**: flavor fractions alone are degenerate. Asimov data injected at
  m_Zd = 2 MeV, g_D = 3e-4, u = 0.5 were fitted to chi^2 = 7e-7 at m_Zd = 8e-14 eV: with
  u_e = u_mu, both the resonant absorption and the coherent potential pull f_e and f_mu together. Injection-recovery
  tests must check that the fit reaches chi^2 ~ 0, not the injected values.
- **Cost**: an evaluation at the analysis resolution takes ~0.35 s on one core. Below
  m_eff ~ 2e-5 eV the Earth's Yukawa columns depend on the couplings (through the Debye mass)
  and are recomputed for every new point (~0.3 s, cached per m_eff since 2026-10-12, so the
  antineutrino pass reuses them). A long-range expansion of those columns would remove most of it.

## Not yet done

Toys and coverage (the statistical-soundness plan: `--toy` re-centring the surface by its own
width), NuFIT 6.1 joint chi^2 for current-data analyses, event-rate likelihoods (IceCubeRates).
