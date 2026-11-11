# Audit plan (2026-11-08): the whole analysis against its sources and his decisions

**Why.** On 2026-11-08 two mismatches with the data turned up by accident, while writing the data-set table:
- the flavor term averages our flux over 10 TeV–30 PeV, while IceCube measured 5 TeV–10 PeV;
- the spectrum term compares our extragalactic flux with IceCube's Table 10 segments, which were fitted with no Galactic component.

Earlier audits (PHYSICS_AUDIT.md, the 2026-10-24/25 reviews, and the 2026-11-08 audit of my own choices) missed both.

They had two flaws:
- they listed the items from memory;
- they checked our code against our own docs instead of against the sources.

**Goal.** Find every point where the analysis departs from its sources (data releases, papers, the physics) or from his decisions. Find every choice made without his sign-off. Measure the size of each. Then decide all fixes in one round, before the final runs are merged.

**Scope.** Everything that produces a number in the paper:
- the final runs: campaign_final, campaign_final_sm, campaign_ml, campaign_si, campaign_si_b;
- all three data sets: spectrum + flavor, events, forecast;
- every model family in them;
- Secs. II–IV, Tables I–III and the appendices of the paper.

**Ground rule (his words, 2026-11-08): "Now you will not assume anything."**
- An item's status comes only from a source I have read (section, table or line cited) or from code I have read (file:line cited).
- Anything I have not read is marked *unverified*, never guessed.
- Memory, earlier audits, docs and my own summaries are leads to check, not evidence.
- A size comes only from a measurement.

---

## 1. How the items are listed (no list from memory)

The item list is built mechanically, from two directions, before anything is judged.

### 1a. From our side, outward

1. **Configurations.** Dump every key and value of one config per family and data set. The families are L_e−L_μ and L_e−L_τ below 1 keV and at MeV, the dark portal MeV e, μ, τ, and SI scalar/vector Majorana/Dirac below 1 keV and at MeV. Dump `DEFAULT_CONFIG` too, and diff them. Every key whose value changes physics, data or statistics becomes one item. A script writes the list; it is not typed by hand.
2. **The χ² call graph.** Walk `analysis/objective.py` → `model.py` → `spectra.py`, `surfaces.py`, `events.py`, `future_events.py` → `IceCubeRates/*`, line by line. Write down every transformation the flux undergoes between the solver's output and each χ² term: averaging, weighting, unit conversion, energy window, sky weighting, flavor sum, ν/ν̄ sum, component selection, interpolation, cuts. Each one becomes an item.
3. **Hard-coded numbers.** Grep the likelihood path and the solver's physics switches for numeric literals: windows, cuts, priors, bounds, grid sizes, thresholds. Each one that is not a unit constant becomes an item.
4. **Docs.** Every entry of SIMPLIFICATIONS_AND_ASSUMPTIONS.md (1–22, all letters) and of PHYSICS_AUDIT.md. Each status is re-checked here; none is taken as settled.
5. **Paper.** Every statement of method in Secs. II–IV, Tables I–III and the appendices, mapped to the code line that implements it.

### 1b. From each source, inward

For every external input, read its analysis section and list each thing it assumes or provides about the flux it measured:
- energy range, sky, flavor ratio, ν/ν̄ ratio, spectral shape inside bins;
- which flux components it fits (isotropic, Galactic, prompt, muons);
- units and normalization energy, errors, correlations, overlaps with other samples.

Each line becomes an item, to be matched to our counterpart. The sources:

| Input | Source to read | What we take from it |
|---|---|---|
| Flavor composition | IceCube:2025ole (arXiv:2510.24957) and its released likelihood | likelihood in (f_e, f_μ) |
| Energy spectrum | IceCube:2025ewu (arXiv:2507.22234): Secs. II.3, III.5, V.5, Table 10 | 10 segment normalizations |
| HESE events | IceCube:2020wum + the 7.5-yr release README | Monte Carlo, binning, detector systematics, muons |
| Through-going tracks | IceTracks-DR2 v3.1 release README (Dataverse) | effective areas, smearing, livetimes, counts |
| Galactic component | IceCube:2026plr; template Schwefer:2022zly | prior 0.91 ± 0.18, template variant |
| Spectral break | IceCube:2025tgp, IceCube:2025ewu | E_b = 24.5 TeV |
| Future telescopes | Song:2020nfh + vendored NuFlavorSingleSource code | detector models, systematics, exposures |
| Atmospheric background | MCEq, SIBYLL-2.3d, H3a, Barr:2006it | fluxes, Barr widths, priors |
| Oscillations, masses | NuFIT 6.1; Barenboim:2026mvn | mixing at best fit; Σm prior |
| Model physics | paper Sec. II, notes/notes.tex | charges, couplings, mediator widths and channels |

Downloads needed for 1b (release READMEs, data files), if any, are listed and asked for first.

### 1c. Merge

The two lists are merged into one table. An item found from only one side is itself a finding: either an assumption of the source with no counterpart in our code, or a choice of ours with no source.

---

## 2. What is checked for each item

The **interface template** is applied to each of the five data terms: flavor, spectrum, HESE, DR2, future. Every row is filled for every term, so nothing is skipped because it "obviously" does not apply.

| # | Row | Question |
|---|---|---|
| T1 | Observable | What exactly did the source measure or release (quantity, units, normalization energy, per flavor or all flavors, ν + ν̄)? |
| T2 | Source's assumptions | What did the source assume about the flux to obtain it (shape inside bins, flavor ratio, isotropy, Galactic in or out, ν/ν̄, cross sections)? |
| T3 | Our counterpart | Which function computes our version, and does it reproduce T2's assumptions? |
| T4 | Energies | The source's range and bins; our window; our grid's coverage of every bin; any extrapolation. |
| T5 | Sky | The source's sky coverage and cuts; our weighting over directions; who applies Earth absorption (solver or response), and only once. |
| T6 | Components | Extragalactic, Galactic, conventional, prompt, muons: is each in signal or background exactly as in the source? |
| T7 | Response | Version, livetime, effective areas, smearing, detector systematics: as in the release? |
| T8 | Background and nuisances | Model and priors as in the source, or our choice? |
| T9 | Likelihood form | The source's form (Poisson, Gaussian, tabulated), asymmetric errors, correlations, upper limits, low counts. |
| T10 | Overlaps | Shared events between terms, and whether the data sets are kept disjoint. |
| T11 | Reproduction | Does our pipeline, fed the source's own baseline model, recover the source's own result (its best fit, its event counts)? Is that check still valid for the current files (e.g., after the 2026-11-07 switch to Table 10)? |

**Model-side rows**, for each model family in the final runs:
- **M1** Lagrangian, charges and signs (paper ↔ code);
- **M2** mediator mass, widths and decay channels, per mass range;
- **M3** each cell of Table I ↔ the code switch that turns the term on, per model;
- **M4** the regime boundaries (1 keV, unitarity, validity);
- **M5** sources (spectrum, composition, redshift evolution, Galactic);
- **M6** Earth, Sun, Moon;
- **M7** numerics settings in the run's regime (the convergence checks that cover it).

**Statistics rows:**
- **S1** each parameter of Table II: range, prior, source, and whether a bound binds;
- **S2** the profile: free versus fixed parameters and the minimizer's convergence;
- **S3** the Δχ² ≥ 3.84 reading and the boundary at g = 0;
- **S4** forecasts on Asimov data;
- **S5** the correction layers in `exclusion.corrected` and their order.

**Recorded per item:**
- ID;
- what we do (file:line, config key);
- what the source says (section, table or equation);
- status: match / mismatch / unverified;
- who decided it (him, with date, or me);
- its size (section 3), the proposed fix and the fix's cost.

---

## 3. How sizes are measured

- **No size without a measurement.** Each mismatch gets a re-evaluation at stored fits, from the flux cache, with no new transport. Where the fix changes the solver itself, the item is marked "needs transport" and costed separately.
- **Points.** Best fit, plus the 95% edge points of one representative plane per family and data set, chosen before any results are seen.
- **Outputs.** The shift in Δχ² at the edge, the shift in coupling, and whether an exclusion flips.
- **Threshold for "material"** (for him to set): for example, an edge moving by more than 0.1 in Δχ² or by more than 2% in coupling.
- **One batch.** All sizing goes into one Tycho submission, after the item list is complete. Before submitting, its cost is measured on one point, and the jobs, cores and that measured time are stated.

---

## 4. Order of work

| Step | What | Compute | His input |
|---|---|---|---|
| 0 | This plan | — | approve or change it |
| 1 | Inventory: collect the sources in 1b; list any downloads | none | OK the downloads |
| 2 | Mechanical enumeration (1a, 1b, 1c) into one raw table | local scripts, seconds | — |
| 3 | Fill the template (section 2) for every item, against source and code | none | — |
| 4 | Completeness check: every config key, every χ²-path line and every docs entry maps to an item | local | — |
| 5 | Independent second pass: an agent given only the sources, the code and the template, not my table; diff the two | one agent | OK launching it |
| 6 | Report: one document, with every mismatch and every choice of mine without his sign-off, grouped by data term | none | decisions, grouped |
| 7 | Sizing batch (section 3) | Tycho, one submission | OK the batch, with its measured cost |
| 8 | Fixes in one code batch: tests, the reproduction checks (T11), absolute checks in the runs' regime | Tycho tests | — |
| 9 | Plan for re-evaluation or reruns of the affected planes, with cost | — | go |

Nothing is fixed, rerun or submitted before step 6. The two known items (the flavor window and the spectrum's Galactic component) enter at step 2, like every other item; they do not shape the list.

## 5. Output files

- `docs/dev/AUDIT_2026-11-08_items.csv`: the merged table (steps 2–4).
- `docs/dev/AUDIT_2026-11-08_report.md`: the report (step 6).
- `scratch_audit/` (not committed): the enumeration scripts and the second pass's raw output.
