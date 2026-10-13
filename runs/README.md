# Runs

Records of the four-point validation (`tools/validate_record.py`) and notable production runs,
with the machine and commit that made them. Each record is a JSON (numbers and provenance) with
the fluxes in an `.npz` of the same name.

| Date | Machine | Commit | Run | Result |
|---|---|---|---|---|
| 2026-10-13 | laptop (8 cores) | `1be2b97` | `record_laptop` | the reference: energy at the detector / no interaction 0.99910 (below), 0.94135 (inband), 0.99995 (strong), 0.99944 (cosmoX2) |
| 2026-10-13 | Tycho node800 (16 cores), job 66293089 | `1102066` | `record_node800_66293089` | agrees with the laptop: fluxes to 9e-14 (below, inband), 9e-10 (strong, its known floor), 6e-12 (cosmoX2); fractions to 2e-14 |
| 2026-10-13 | laptop (8 cores) | transit redesign (uncommitted at the run; committed with it) | `record_laptop` (regenerated) | the new reference: below, inband, strong unchanged to 3e-12; cosmoX2 1.4e-3 pointwise, 1.8e-8 in the fractions -- the old Milky Way stepping's own error, which with 50 trajectories converges onto the new adiabatic result to 4e-6 |
| 2026-10-13 | laptop (8 cores) | with the cancellation-free chord radius | `record_laptop` (regenerated) | moves below/inband/strong by <= 5e-10 in flux and 6e-11 in the fractions (the Earth's ~1e5-rad phases amplifying ulp changes of the radius); cosmoX2 by 6e-16 |
| 2026-10-13 | laptop (8 cores) | fastmath kernels with numpy's error model | `record_laptop` (regenerated) | strong moves by 7e-6 pointwise in its absorption dip (fractions 5e-10): the thermal rates' far tails (1e-6 to 1e-10 of the peak) are rounding-dominated and follow any change of compilation; near resonance they agree to 3e-12. Others <= 1e-9 |
| 2026-10-13 | Tycho node810 (16 cores each), jobs 66296034/36/37/38 | `52f36fc` | potential planes eμ, eτ, μτ and the eμ exclusion plane (`output/analysis/`) | ~3 min each; the first planes with the transit of `81c04b0` (and, for the exclusion plane, the first with the regeneration fix and the whole sky); notes, Sec. Scans and limits |
| 2026-10-13 | Tycho node801, job 66296001 | `52f36fc` | `record_node801_66296001` | agrees with the laptop: 5e-14 (below), 7e-14 (inband), 5e-12 (strong), 1.5e-15 (cosmoX2) |
