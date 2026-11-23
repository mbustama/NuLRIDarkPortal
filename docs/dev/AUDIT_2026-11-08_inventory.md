# Audit 2026-11-08, step 1: inventory of external inputs

Plan: docs/dev/AUDIT_PLAN_2026-11-08.md.

This file records only facts I read, each with where I read it. No judgments are made here; checks start in step 3.

"Origin as recorded" means where our repository or Tycho says the file came from. It is a lead, not evidence.

| # | Input | Our file(s) | Origin as recorded (where I read it) | Source to read in step 3 | Access |
|---|---|---|---|---|---|
| I1 | Flavor composition (data set "spectrum and flavor") | `analysis/data/flavor_surfaces.json`, entry `icecube_mese_2025`: an analytic "Asymmetric (Smooth) Gaussian" surface, centre (0.300, 0.375), Lmax 12.8029 | Moved verbatim from FrequentistAnalysis on 2026-10-12 (the file's `_comment`). It entered this repo in 8fe48cc (2026-07-19), whose docstring says the surfaces are fitted to "measured likelihood contours". **Which figure or contours it was fitted to is not recorded** anywhere I found: FrequentistAnalysis's current tree no longer has the file. | IceCube:2025ole, arXiv:2510.24957 (HTML) | online. The paper's text mentions no data release (searched: data release, github, zenodo, dataverse, supplemental). IceCube's data-release page (read 2026-11-08) lists none for it. |
| I2 | Energy spectrum (same data set) | `src/data/icecube_cf_2025_table10.json` (13 segments; 10 used) | Copied by hand from the arXiv HTML, Table 10, CF column, on 2026-11-07 (the file's `method` field) | IceCube:2025ewu, arXiv:2507.22234v2 (HTML): Secs. II.3, III.5, V.5, App. B, Tables 3, 7, 10 | online. No data release is listed on IceCube's page. |
| I3 | HESE events | Tycho `external/hese_7yr/reduced/hese_mc.npz` and `hese_detector_systematics.npz`; `IceCubeRates/starting_events/hese.py` | github.com/icecube/HESE-7-year-data-release at 2d1ed03 (docs/dev/HESE_RATES_PLAN.md, line 21). The clone is on Tycho with SHA256SUMS. | IceCube:2020wum, arXiv:2011.03545; the release's README and toolkit (Tycho `external/hese_7yr/HESE-7-year-data-release`) | Tycho, local. (Fact: IceCube's page also lists a HESE 12-year release, 2023-07.) |
| I4 | Through-going tracks | `IceCubeRates/data/dr2_irfs.npz` and `dr2_counts.npz`, with livetimes in s and the uptime file names inside | IceTracks-DR2 v3.1, Harvard Dataverse doi:10.7910/DVN/MMIIZA (DR2_TRACKS_PLAN.md, line 22). The originals are on Tycho with md5.expected. | Tycho `external/icetracks_dr2/DR2_readme.txt`; the release's page | Tycho, local |
| I5 | Galactic template and its prior | config `solver.galactic.template = cringe`, `emission = 3d`; `src/data/galactic_cringe.npz` and others; prior on `norm_gal` 0.91 ± 0.18, bounds [0, 3] (config `priors`) | Template: Tycho `external/galactic_data` (CRINGE FITS files, SHA256SUMS). Prior: paper Sec. IV cites IceCube:2026plr. | IceCube:2026plr, arXiv:2607.25966; Schwefer:2022zly, arXiv:2211.15607 | online plus Tycho |
| I6 | Astrophysical flux shape | config `solver.source.E_break_GeV = 24547.09`; `solver.sky.extragalactic_norm = 1.77e-18` at 1e5 GeV | docs/dev/BROKEN_POWER_LAW.md, lines 4 and 15: "PRD 113, 062002 (2026), Table 3, 'CF BPL'" | IceCube:2025ewu, Table 3; IceCube:2025tgp, arXiv:2507.22233 | online |
| I7 | Atmospheric background | `IceCubeRates/data/atmospheric_mceq_southpole.npz`; Tycho `external/atm_background` | The file's `models` field: MCEq, SIBYLL-2.3d × H3a, MSIS00_IC South Pole (January + July)/2, derivatives as NuFlavorSingleSource/code/atm_flux_syst.py. tools/atmospheric_background.py, lines 4–20. | Fedynitch:2015zma, Riehn:2019jet, Gaisser:2011klf, Barr:2006it; ~/Research/NuFlavorSingleSource | online plus local |
| I8 | Future telescopes | `IceCubeRates/future/*` (nfss_forecast.py, nfss_detectors.py vendored; diffuse.py ours); `IceCubeRates/data/future` | "NuFlavorSingleSource's detector models … vendored unchanged" (analysis/future_events.py docstring) | Song:2020nfh, arXiv:2012.12893; ~/Research/NuFlavorSingleSource (the original code, to diff the vendored copy) | online plus local |
| I9 | Mixing parameters | `analysis/data/nufit_6.1/pairwise.npz` and README; the fixed best-fit values in the configs | README: converted from v61.release-TByes-{NO,IO}.txt.xz, downloaded 2026-10-09 from nu-fit.org; kept in GeoneutrinoDecay | Esteban:2024eli; nu-fit.org v6.1 release (local in GeoneutrinoDecay) | local |
| I10 | Prior on the sum of masses | config `priors`: `sum_masses_eV`, half-Gaussian, limit 0.12 at 95% | paper Sec. IV cites Barenboim:2026mvn | Barenboim:2026mvn, arXiv:2606.17132 | online |
| I11 | Model physics | src/ (solver), configs | paper Sec. II and App.; notes/notes.tex | the physics itself: derivations in the notes, checked against the code | local |

## Not yet inventoried

These are inputs that step 2's mechanical enumeration may add. Any it finds are appended here with the same columns.
- Cross sections inside the HESE, DR2 and forecast responses.
- Earth model (PREM) and electron fractions.
- Relic-neutrino temperature and density.
- Milky Way matter and relic profiles.
- Sun and Moon compositions.
- Unit constants.

## Downloads

None is needed for step 3 so far:
- the papers are readable online;
- the HESE, DR2, Galactic and atmospheric originals are on Tycho;
- NuFIT and NuFlavorSingleSource are local.

If step 3 needs the original of I1 (IceCube's flavor contours or likelihood), I will ask first. That would be a figure's data or a request to IceCube; no public release was found.
