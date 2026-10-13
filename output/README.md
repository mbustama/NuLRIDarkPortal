# Analysis results

Summaries (`plane_*.json`, `fit.json`), figures (`plane_*.pdf`) and run manifests of the analysis package
(`python -m analysis CONFIG`). The JSONL checkpoints that make runs resumable are not committed; rerun the
configuration (on Tycho, `slurm/analysis.sh`) to regenerate them. `output/figures/` holds the comparison
figures of the notes.

| Folder | Plane | Commit | Run (UTC+2) | Host |
|---|---|---|---|---|
| `analysis` | `pion_2050_case_5_exclusion` | `52f36fc` | 2026-10-13T12:15 | node810.cluster |
| `analysis` | `pion_2050_case_5_mev_emu` | `6cf6b2b` | 2026-10-13T09:51 | laptop-6 |
| `analysis` | `pion_2050_case_5_mev_etau` | `6cf6b2b` | 2026-10-13T09:57 | laptop-6 |
| `analysis` | `pion_2050_case_5_mev_mutau` | `6cf6b2b` | 2026-10-13T10:01 | laptop-6 |
| `analysis` | `pion_2050_case_5_potential_emu` | `52f36fc` | 2026-10-13T12:15 | node810.cluster |
| `analysis` | `pion_2050_case_5_potential_etau` | `52f36fc` | 2026-10-13T12:15 | node810.cluster |
| `analysis` | `pion_2050_case_5_potential_mutau` | `52f36fc` | 2026-10-13T12:15 | node810.cluster |
| `analysis` | `pion_2050_case_5_potential_emu_host` | `971e44b` | 2026-10-13T12:41 | node801.cluster |
| `analysis` | `pion_2050_case_5_potential_etau_host` | `971e44b` | 2026-10-13T12:41 | node810.cluster |
| `analysis` | `pion_2050_case_5_potential_mutau_host` | `971e44b` | 2026-10-13T12:41 | node810.cluster |
| `analysis` | `pion_2050_case_5_si_e` | `fe8e2dc` | 2026-10-13T16:12 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_mu` | `fe8e2dc` | 2026-10-13T16:12 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_tau` | `fe8e2dc` | 2026-10-13T16:12 | node801.cluster |
| `analysis` | `pion_2050_case_5_si_universal` | `fe8e2dc` | 2026-10-13T16:12 | node801.cluster |
| `analysis_pre_transit` | `pion_2050_case_5_exclusion` | `19f953d` | 2026-10-13T00:13 | laptop-6 |
| `analysis_pre_transit` | `pion_2050_case_5_potential_emu` | `abe5c24` | 2026-10-13T10:07 | laptop-6 |
| `analysis_pre_transit` | `pion_2050_case_5_potential_etau` | `1be2b97` | 2026-10-13T10:14 | laptop-6 |
| `analysis_pre_transit` | `pion_2050_case_5_potential_mutau` | `1be2b97` | 2026-10-13T10:22 | laptop-6 |

`output/analysis_pre_transit/` keeps the planes computed before the transit redesign of 2026-10-13
(`81c04b0`) for comparison; its exclusion plane also predates the regeneration fix (`7f0095e`) and the
whole-sky average (`6cf6b2b`). Not for quoting.

The `*_host` planes put the sources inside a Milky-Way-like host galaxy (`source.host_galaxy_scale = 1`);
the others keep them in the cosmic mean density (the default).

The `*_si_*` planes replace the dark portal by secret neutrino interactions (`src/dp_si.py`; scalar
mediator, Majorana neutrinos): nu_e, nu_mu, nu_tau or flavor-universal couplings, in (m_phi, g).
`output/si_validation/` holds the validation runs of that module (`tools/si_validation.py`, Tycho job
66300967 at `781ae43`: absorption against direct quadrature, convergence, 2107.13568 Fig. 5), and
`output/si_planes/flavor_paths.json` the flavor paths of `tools/si_flavor_paths.py` (job 66301153 at
`5b6ef9a`).
