# Analysis results

Summaries (`plane_*.json`, `fit.json`), figures (`plane_*.pdf`) and run manifests of the analysis package
(`python -m analysis CONFIG`). The JSONL checkpoints that make runs resumable are not committed; rerun the
configuration (on Tycho, `slurm/analysis.sh`) to regenerate them. `output/figures/` holds the comparison
figures of the notes.

| Folder | Plane | Commit | Run (UTC+2) | Host |
|---|---|---|---|---|
| `analysis` | `pion_2050_case_5_exclusion` | `124fd03` | 2026-10-13T21:28 | node803.cluster |
| `analysis` | `pion_2050_case_5_mev_emu` | `124fd03` | 2026-10-13T21:24 | node803.cluster |
| `analysis` | `pion_2050_case_5_mev_etau` | `124fd03` | 2026-10-13T21:25 | node803.cluster |
| `analysis` | `pion_2050_case_5_mev_mutau` | `124fd03` | 2026-10-13T21:26 | node803.cluster |
| `analysis` | `pion_2050_case_5_potential_emu` | `124fd03` | 2026-10-13T21:08 | node803.cluster |
| `analysis` | `pion_2050_case_5_potential_emu_host` | `124fd03` | 2026-10-13T21:17 | node803.cluster |
| `analysis` | `pion_2050_case_5_potential_etau` | `124fd03` | 2026-10-13T21:12 | node803.cluster |
| `analysis` | `pion_2050_case_5_potential_etau_host` | `124fd03` | 2026-10-13T21:19 | node803.cluster |
| `analysis` | `pion_2050_case_5_potential_mutau` | `124fd03` | 2026-10-13T21:15 | node803.cluster |
| `analysis` | `pion_2050_case_5_potential_mutau_host` | `124fd03` | 2026-10-13T21:22 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_dirac_e` | `124fd03` | 2026-10-13T21:08 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_dirac_mu` | `124fd03` | 2026-10-13T21:19 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_dirac_tau` | `124fd03` | 2026-10-13T21:28 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_dirac_universal` | `124fd03` | 2026-10-13T21:08 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_e` | `124fd03` | 2026-10-13T21:08 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_e_mu` | `124fd03` | 2026-10-13T21:34 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_e_tau` | `124fd03` | 2026-10-13T21:39 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_mu` | `124fd03` | 2026-10-13T21:15 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_mu_tau` | `124fd03` | 2026-10-13T21:45 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_tau` | `124fd03` | 2026-10-13T21:22 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_universal` | `124fd03` | 2026-10-13T21:26 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_vector_dirac_e` | `124fd03` | 2026-10-13T21:25 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_vector_dirac_mu` | `124fd03` | 2026-10-13T21:08 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_vector_dirac_tau` | `124fd03` | 2026-10-13T21:22 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_vector_dirac_universal` | `124fd03` | 2026-10-13T21:28 | node803.cluster |
| `analysis` | `pion_2050_case_5_si_vector_e` | `124fd03` | 2026-10-13T21:19 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_vector_mu` | `124fd03` | 2026-10-13T21:27 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_vector_tau` | `124fd03` | 2026-10-13T21:08 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_vector_universal` | `124fd03` | 2026-10-13T21:17 | node801.cluster |
| `analysis_pre_transit` | `pion_2050_case_5_exclusion` | `19f953d` | 2026-10-13T00:13 | laptop-6 |
| `analysis_pre_transit` | `pion_2050_case_5_potential_emu` | `abe5c24` | 2026-10-13T10:07 | laptop-6 |
| `analysis_pre_transit` | `pion_2050_case_5_potential_etau` | `1be2b97` | 2026-10-13T10:14 | laptop-6 |
| `analysis_pre_transit` | `pion_2050_case_5_potential_mutau` | `1be2b97` | 2026-10-13T10:22 | laptop-6 |

`output/analysis_pre_transit/` keeps the planes computed before the transit redesign of 2026-10-13
(`81c04b0`) for comparison; its exclusion plane also predates the regeneration fix (`7f0095e`) and the
whole-sky average (`6cf6b2b`). Not for quoting.

The `*_host` planes put the sources inside a Milky-Way-like host galaxy (`source.host_galaxy_scale = 1`);
the others keep them in the cosmic mean density (the default).

The `*_si_*` planes replace the dark portal by secret neutrino interactions (`src/dp_si.py`):
nu_e, nu_mu, nu_tau, flavor-universal or two-flavor couplings, in (m_phi, g); `*_si_<texture>` for a
scalar mediator with Majorana neutrinos, `*_si_dirac_*` for Dirac neutrinos (sterile secondaries
tracked), `*_si_vector_*` and `*_si_vector_dirac_*` for a vector coupled to left-handed currents.
`output/si_validation/` holds the validation runs of that module (`tools/si_validation.py`, Tycho job
66300967 at `781ae43`: absorption against direct quadrature, 2107.13568 Fig. 5; job 66302931 at
`1b7d29b`: Fig. 3; jobs 66304704 at `d82231c`: the four models on one benchmark, convergence on
nested grids), and
`output/si_planes/flavor_paths.json` the flavor paths of `tools/si_flavor_paths.py` (job 66301153 at
`5b6ef9a`).
