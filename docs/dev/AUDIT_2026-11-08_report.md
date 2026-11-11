# Audit 2026-11-08: report (step 6)

Plan: AUDIT_PLAN_2026-11-08.md. Items and evidence: AUDIT_2026-11-08_items.csv. Per data term: AUDIT_2026-11-08_template.md. Independent pass: AUDIT_2026-11-08_step5_diff.md.

Every finding below cites what was read. "Size" is measured, or says "not measured". Nothing has been fixed.

**Fix types:**
- **Paper:** text only.
- **Re-evaluation:** recompute χ² from the cached fluxes, then refit the points that move; no new transport.
- **Rerun:** new transport.

Costs are not measured until step 7.

**Correction to step 1:** IceCube did release MESE data (Harvard Dataverse doi:10.7910/DVN/CBNMEB, v2.0). NuFlavorSingleSource keeps a copy in data/mese/: effective areas per flavor, interaction and morphology, and energy histograms. **Second correction (2026-11-08, after his pointer):** the release also contains the flavor likelihood itself, `mese_contour.json` (TS on a 90 x 90 grid in f_e, f_tau); NuFlavorRatiosSteriles uses it (input/IceCube_MESE_2025_Dataverse/). The earlier line "It has no flavor likelihood" was wrong.

---

## 1. Spectrum and flavor composition (data set now_ic26)

| # | Finding | Evidence | Size | Fix type |
|---|---|---|---|---|
| A1 | We average the flavor composition over 10 TeV–30 PeV; IceCube measured it over 5 TeV–10 PeV. | arXiv:2510.24957 abstract; `data.flavor_band_eV` in all configs | not measured | re-evaluation |
| A2 | Our flavor likelihood is an analytic surface fitted to IceCube's contours in July; which contours were used is not recorded. At the edges it is looser than IceCube. | f_tau = 0: chi^2 3.91 vs 5.03 for IceCube's 91.9%. f_e = 0: 7.60 vs 8.69 (98.7%). Neutron decay: 6.01 vs 6.12 (95.3%). Under a 2-dof reading; both audits computed this. | ~1.1 in chi^2 at both edges | re-evaluation (refit the surface) |
| A3 | We weight flavors by flux. A measurement weights them by its effective areas and cross sections. | flux_postprocessing.py:36-56; docs 12 | not measured | re-evaluation (weight by MESE's released areas) |
| A4 | IceCube's flavor likelihood was profiled under its own broken power law (gamma_1 1.76, gamma_2 2.81, break 31.6 TeV), not the CF fit we use. | 2510.24957, Method | not measured | paper (state it); no fix without their likelihood |
| A5 | IceCube's spectrum segments come from a fit with no Galactic component, so they contain the Galactic flux. Our counterpart uses the extragalactic flux alone. | 2507.22234 Sec. V.5; objective.py:88-91 | not measured | re-evaluation (add the Galactic flux) |
| A6 | Our segment counterpart is the E^-2 flux with the same number of neutrinos. IceCube's is the E^-2 normalization that best fits the events. | spectra.py:64-80 | refit of CF's BPL with our counterpart: phi_0 1.615, gamma_1 1.48, gamma_2 2.82, against 1.77, 1.31, 2.735 (-0.9 sigma, +1.2 sigma) | paper (state it); CF's response is not public |
| A7 | **The flavor and spectrum terms share events**: 2,514 cascades and 1,799 tracks are in both MESE and CF (CF's Table 1). We add the two chi^2 as independent. | 2507.22234 Table 1; objective.py:109-112 | not measured | decision |

## 2. Events (data set now_events)

| # | Finding | Evidence | Size | Fix type |
|---|---|---|---|---|
| B1 | **HESE and DR2 may share events.** DR2's README says its events "should not be combined with any other releases"; we combine them with HESE in one likelihood. HESE's data have no event times, so the shared events cannot be matched. | DR2_readme.txt; events.py | not measured | decision |
| B2 | HESE's release uses a likelihood that accounts for limited MC statistics (arXiv:1901.04645); ours is plain Poisson. | HESE README; events.py:58-117 | not measured | re-evaluation (code change) |
| B3 | DR2: tracks from nu_tau -> tau -> mu are left out, because the release gives nu_mu areas only. | events.py:303 | not measured | paper (state it) or an estimate |
| B4 | DR2: Earth absorption is assumed to be inside the released areas; the README does not say. The data/expectation ratio being flat in declination is indirect support. | dr2.py:4-5; DR2_TRACKS_PLAN.md:99 | — | paper (state it) |
| B5 | Our HESE nuisances follow NuFlavorSingleSource (your 2026-10-14 decision), so they differ from the release: the cosmic-ray shift is centred at 0 (release: -0.05); the prompt flux has a 1 ± 0.5 prior (release: free); there is no nu/nubar nuisance (release: 1 ± 0.1). | HESE_fit.py:177-190; events.py:44-46 | not measured | paper (state it), or switch |
| B6 | The fluxes are zero outside the energy grid. HESE events with true energies above 30 PeV, and DR2 tracks above 3.2 TeV from neutrinos below 1 TeV, get no astrophysical flux. | flux_adapters.py:32-41 | not measured | step 7 measures it |
| B7 | Our pipeline's reproduction of IceCube's own nu_mu spectral index from DR2 has not been redone on current code. The independent pass found 2.60 (old stored output) against IceCube's 2.37 ± 0.09. | lead only | — | step 7 could measure it |
| B8 | The paper calls HESE's exposure 7.5 yr; its live time is 7.22 yr (227,708,167.68 s). | data-table caption | — | paper |
| B9 | The paper says DR2 is binned in 0.2 of log10 energy; the code uses 0.1. | dr2_counts.npz | — | paper |

## 3. Forecast (data set proj_future)

| # | Finding | Evidence | Size | Fix type |
|---|---|---|---|---|
| C1 | **The forecast uses the detector model you declared invalid in NuFlavorSingleSource on Oct 15** ("STOP ... every result below is invalid": unvalidated responses, Gandhi 1998 cross sections, no NC or nu_tau regeneration). That project replaced it with "Plan A", built from IceCube's public responses. Our copy is commit 39088bd, taken on Oct 14–15. **The SI jobs running now compute forecast planes with it.** | NuFlavorSingleSource CLAUDE.md:14 and README; FD1 diff | — | decision |
| C2 | For telescopes away from the South Pole, the forecast reuses the South Pole's sky bands as local-zenith bands. That assumes the flux is isotropic before the Earth, which fails wherever the Milky Way changes the flux by direction (ultralight L_e - L_alpha). | model.py:1360-1374 | not measured | rerun of those planes, or paper |
| C3 | The paper credits Song:2020nfh with the forecast's detector models, backgrounds and priors (three places). Nothing in the code links them to it, and that paper has no TRIDENT, NEON or HUNT models. | FD5; second pass | — | paper |
| C4 | Table III gives the sum-of-masses prior as 0.12 eV for every data set; the forecasts use 0.09 eV. | configs; parameters.py | — | paper |

## 4. Model, figures and paper text

| # | Finding | Evidence | Fix type |
|---|---|---|---|
| D1 | Table I's longitudinal-mode row is imprecise. The vector decay rate keeps the 1/M^2 term in every vector model, and L_e - L_alpha has the mode below 1 keV only. | dp_decay.py:76-79; dp_si_light.py:843 | paper |
| D2 | Table I gives soft scatterings below 1 keV for the scalar. On the Galactic paths the code has them for the vector only; the cosmological phase is unverified. | dp_si_light.py:549 | paper or code (decide after checking the cosmological phase) |
| D3 | The paper says the lightest-mass scan runs in the MeV planes. It also runs in the dark portal's ultralight planes. | configs | paper |
| D4 | The paper ties thin-band removal to fitted compositions, but the code removes thin bands in every ultralight plane. | exclusion.py:296 | decision (code or paper) |
| D5 | "The coverage tests check both effects": no coverage tests exist yet. | main.tex:625; grep | paper (until they exist) |
| D6 | "9–13%" should read "1.1–13%" (the off-diagonals of g U^T U). | computed | paper |
| D7 | "We use the likelihood of the flavor measurement": we use a surface fitted to its contours. | A2 | paper |
| D8 | The figure notebook still reads campaign_ml, including the dark portal's potential planes, which were dropped on 2026-10-24. Figs. 4–7 are mock-ups for now. | notebook cells 1, 12-13 | at the final merge |
| D9 | Leads, not verified: naming (the README says "IceT-DR2"); "tens of MeV" vs ~100 MeV; arXiv:2606.17132 shows a "superseded" notice. | second pass | check |

## 5. Choices I made without your sign-off

| # | Choice | Where | Size or evidence |
|---|---|---|---|
| E1 | Neutrinos and antineutrinos 1:1 at the sources (this matters for HESE's Glashow events) | docs 10; configs | not measured |
| E2 | New physics acts on the astrophysical flux only, not on atmospheric neutrinos | docs 4 | conservative (stated) |
| E3 | Sources follow the star-formation rate to z = 4, with no extra evolution | docs 8 | — |
| E4 | Galactic propagation adiabatic; the MeV transport's approximations; SI relics at rest, and the Dirac helicity | docs 6, 6a-6c | sizes in docs (leads) |
| E5 | Numerics: 200/400 redshift steps, 3/7 index nodes, N_side 8, 12 Earth chords | docs 17-19 | sizes in docs (leads); not checked in each run's regime |
| E6 | HESE detector systematics: the profile uses a per-bin model; the chi^2 is evaluated per event | docs 14 | <= 0.02 in -2 ln L |
| E7 | The exact profile's inheritance shortcut | docs 11e | stated limits (lead) |
| E8 | The bound gamma_low >= 0.5 | configs | measured: shifts Delta chi^2 at the 95% edges by median 0.00, extremes -0.23/+0.25 (MeV) and -0.01 (below 1 keV) |
| E9 | Limits read as upper limits only; specks of at most 3 points (none above 5) dropped | exclusion.py | — |
| E10 | The SM potential is on only for L_e - L_alpha below 1 keV, on my bound (it moves a mass-state mixture by <= 1.4e-3 at 100 GeV) | configs | measured bound |
| E11 | The flux cache's key covers the solver code but not its data files | dp_core.py:49-58 | risk; whether a data file changed after the cached runs is unchecked |
| E12 | The cosmology constants (H0 67.4, Omega_m 0.315) are uncited in the code | dp_config.py:72-73 | — |

## 6. Still unverified (30 items)

These are mostly numbers from earlier analyses whose outputs I did not re-read:
- decisions B and C;
- zeta 0.1–0.5 near the unitarity line;
- 840 and 3.4e5;
- the Earth's phases;
- the 1e-5 sky average;
- "factors 0.08–2".

Also unverified:
- two figure details (495 TeV, cos theta_z = -0.6);
- the dark portal's Z-mixing bound (the paper's "8 to 30 orders");
- PAP68, PAP82 and PAP213.

## 7. What step 7 would measure (one Tycho batch, after your decisions)

From the cached fluxes, at the stored best fits and 95% edges of one representative plane per family and data set, step 7 re-evaluates the fixes that do not need new transport: A1, A3, A5 (and A2 if the surface is refitted), B2 and B6. It reports the shift in Delta chi^2 and in coupling for each. Before submitting, I measure the cost on one point and state the jobs, cores and that time.
