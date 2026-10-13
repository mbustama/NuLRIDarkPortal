# Runs

Records of the four-point validation (`tools/validate_record.py`) and notable production runs,
with the machine and commit that made them. Each record is a JSON (numbers and provenance) with
the fluxes in an `.npz` of the same name.

| Date | Machine | Commit | Run | Result |
|---|---|---|---|---|
| 2026-10-13 | laptop (8 cores) | `1be2b97` | `record_laptop` | the reference: energy at the detector / no interaction 0.99910 (below), 0.94135 (inband), 0.99995 (strong), 0.99944 (cosmoX2) |
| 2026-10-13 | Tycho node800 (16 cores), job 66293089 | `1102066` | `record_node800_66293089` | agrees with the laptop: fluxes to 9e-14 (below, inband), 9e-10 (strong, its known floor), 6e-12 (cosmoX2); fractions to 2e-14 |
| 2026-10-13 | laptop (8 cores) | transit redesign (uncommitted at the run; committed with it) | `record_laptop` (regenerated) | the new reference: below, inband, strong unchanged to 3e-12; cosmoX2 1.4e-3 pointwise, 1.8e-8 in the fractions -- the old Milky Way stepping's own error, which with 50 trajectories converges onto the new adiabatic result to 4e-6 |
