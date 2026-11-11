# Audit 2026-11-08, step 3: the interface template per data term

Status values: **match**, **mismatch** or **unverified**.
- Evidence: the source location or file:line actually read.
- Item ids refer to docs/dev/AUDIT_2026-11-08_items.csv.
- "Decided by": "him (date)" only where his decision is recorded; otherwise "me" or "not recorded".

## Step 3 status (2026-11-08, evening)

Of the 1,194 items in AUDIT_2026-11-08_items.csv:
- **30 are mismatches.**
- **30 are unverified** after the model-side pass (his choice, 'I check them now'):
  - numbers from earlier analyses whose outputs I did not re-read (decision B and C results; zeta 0.1–0.5; 840 and 3.4e5; Earth phases; the 1e-5 sky average; 'factors 0.08–2');
  - figure details not found in the scripts (495 TeV, cos theta_z = -0.6);
  - the uncited cosmology values (H0, Omega);
  - EV3 (the flux outside the grid) and EV4 (DR2's Earth absorption), which need a measurement or the release's word;
  - the dark portal's Z-mixing bound (PAP25, 26, 29);
  - PAP68, PAP82 and PAP213.
- The rest are match, choice (with its decider), numerics, units or N/A, each with its evidence.

The mismatches, by data set (details in the rows below):
- **Spectrum + flavor:**
  - the flavor window (10 TeV–30 PeV vs 5 TeV–10 PeV);
  - number weighting rather than event weighting;
  - the flavor surface is ~1.1 in chi^2 too permissive at the f_tau = 0 and f_e = 0 edges;
  - the Galactic flux is left out of the spectrum term although Table 10 contains it;
  - MESE and CF share 2,514 cascades and 1,799 tracks, counted twice;
  - the flavor likelihood assumes IceCube's own BPL, not CF's.
- **Events:**
  - HESE/DR2 overlap not removed (the DR2 release says not to combine);
  - nu_tau-induced tracks left out of DR2;
  - no MC-statistics term (HESE's release uses one);
  - DR2's Earth absorption assumed inside A_eff (the release does not say).
- **Forecast:** the detector model is NuFlavorSingleSource 39088bd's, which he declared invalid there on Oct 15 (STOP).
- **Paper:**
  - DR2 bins are 0.1 dex, not 0.2;
  - Song:2020nfh is credited for the forecast's detector models, backgrounds and priors without support;
  - Table II gives 0.12 eV, but the forecasts use 0.09;
  - "9–13%" should read 1.1–13%;
  - Table I's longitudinal-mode row: the decays keep the 1/M^2 term for every vector model, and L_e - L_alpha has the mode below 1 keV only;
  - "we use the likelihood" of the flavor measurement, when we use a fitted surface.
- **My undocumented choice:** nu:nubar = 1 at the sources (docs 10 says Accepted; no decision by him).

Measured: the counterpart's BPL refit to Table 10; the flavor surface against IceCube's quoted levels; the gamma_low bound (jobs 36613/36614). Nothing is fixed.

## Flavor composition (data set "spectrum and flavor")

| Row | Source says (evidence) | We do (evidence) | Status | Decided by | Size |
|---|---|---|---|---|---|
| T1 observable | (f_e, f_mu, f_tau) of nu + nubar, all-sky, 11.4 yr (MF1, MF2) | analytic surface in (f_e, f_mu), `icecube_mese_2025` (SURF1) | match in kind | — | — |
| T2 assumptions | the same ratio at all energies; BPL fitted jointly with IceCube's own parameters (gamma_1 1.76, gamma_2 2.81, E_b 10^4.5 GeV); no Galactic component modeled (MF2, MF3, MF5) | flux-weighted average of our energy-dependent flux, which includes the Galactic flux; our spectrum is CF's BPL, fitted separately (MOD2) | mismatch: our flux's composition varies with energy, theirs is assumed constant. Including the Galactic flux matches what the all-sky events contain. | me (docs 12 "Accepted for now", no name) | not measured |
| T3 counterpart | — | MOD2: int F_alpha dE / int F dE (number-weighted), not event-weighted | mismatch: a measurement weights flavors by its effective areas and cross sections (docs 12; PHYSICS_AUDIT 6c) | me | not measured |
| T4 energies | 5 TeV–10 PeV (MF1 abstract) | 10 TeV–30 PeV (`data.flavor_band_eV`, all 738 configs; MOD2) | **mismatch** | me (from the July grid; not recorded as a decision) | not measured |
| T5 sky | all sky (MF1) | solid-angle average of bands + Galactic pixel shares (MOD3) | match | — | — |
| T6 components | astrophysical only; atmospheric fitted by IceCube | our astrophysical flux only | match | — | — |
| T7 response | forward folding with MESE's response (MF4) | none: the surface is applied to the flux average | mismatch (docs 12) | me | not measured |
| T8 background | inside IceCube's fit | none needed | match | — | — |
| T9 likelihood | profile likelihood, 68% and 95% contours by Wilks (Fig. 1; MF4) | analytic asymmetric Gaussian fitted to contours whose origin is not recorded (I1) | **mismatch, measured**: our surface vs IceCube's quoted rejections under a 2-dof reading. Neutron decay: chi^2 6.01 vs 6.12 (95.0% vs 95.3%). f_tau = 0: 3.91 vs 5.03 (85.8% vs 91.9%). f_e = 0: 7.60 vs 8.69 (97.8% vs 98.7%). Which dof IceCube used for the fractions is unverified. | not recorded (the surface came from FrequentistAnalysis, July) | ~1.1 in chi^2 too permissive at both edges |
| T10 overlaps | CF's Table 1 (2507.22234): MESE and CF share 2514 cascades and 1799 tracks | data set 1 adds the MESE flavor term and the CF spectrum term as independent | **mismatch**: shared events counted twice | not recorded | not measured |
| T11 reproduction | best fit (0.30 : 0.37 : 0.33) | surface centre (0.300, 0.375) | match at the centre; edges as in T9 | — | — |

## Energy spectrum (same data set)

| Row | Source says (evidence) | We do (evidence) | Status | Decided by | Size |
|---|---|---|---|---|---|
| T1 observable | Table 10, CF column: 13 E^-2 normalizations, 1e-18 GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV (CF4) | 10 segments of `icecube_cf_2025_table10.json` (SPEC1, OBJ2) | values match the table (copied 2026-11-07). The per-flavor reading is unverified in the text: "total flux Phi_nu+nubar" with 1:1:1 (CF3); our JSON says per flavor. | him (2026-11-07, table switch) | — |
| T2 assumptions | isotropic, no Galactic component in the baseline fits (CF2; GL3); flavor 1:1:1 (CF3); E^-2 inside each segment, all fitted together (CF4) | extragalactic flux only (OBJ1); per flavor = all-flavor/3; C = int Phi dE / int E^-2 dE (SPEC1) | **mismatch** on the Galactic flux, which IceCube's segments contain. The rest is matched in form. | me (the counterpart was written for the 2026 Milky Way paper's segments, fitted with Galactic templates; not re-matched at the 2026-11-07 switch) | not measured |
| T3 counterpart | the segment normalization that best fits the events in that band | the E^-2 flux with the same number of neutrinos over the band (SPEC1) | mismatch in kind: equal only for a flat exposure across the band or an E^-2 flux | me | see T11 |
| T4 energies | 13 bands, 1 TeV–100 PeV | 10 inside the grid (4.6 TeV–10 PeV); grid E_min = 4.57 TeV (configs) | match (the excluded bands are outside the grid) | him (2026-11-07) | — |
| T5 sky | tracks from the Northern sky; cascades all-sky (CF1) | all-sky average | match for an isotropic flux; with new physics, the all-flavor sum is unchanged by flavor conversion in the Earth | — | — |
| T6 components | astrophysical (isotropic) + atmospheric in their fit | extragalactic only | **mismatch** (as T2) | me | not measured |
| T7 response | forward folding (CF6) | none | mismatch in kind (T3) | me | see T11 |
| T8 background | inside IceCube's fit | none needed | match | — | — |
| T9 likelihood | 1-D profile errors (Wilks); best fit 0 -> 68% upper limit; correlations not published (CF4) | asymmetric Gaussian, independent segments; zero lower error at 0 (SPEC2) | approximation, forced by what is published | docs 13 (Accepted; his name for the table switch) | not measurable without the correlations |
| T10 overlaps | shared events with MESE (as above) | counted twice | **mismatch** | not recorded | not measured |
| T11 reproduction | CF BPL: phi_0 1.77 (+0.19 -0.18), gamma_1 1.31 (+0.51 -1.30), gamma_2 2.735 (+0.067 -0.075), E_b 10^4.39 GeV (CF5) | our counterpart + chi^2, fitted to Table 10 with E_b fixed (local, seconds) | **measured**: phi_0 1.615, gamma_1 1.48, gamma_2 2.82 (chi^2 8.70 for 10 segments); at CF's own best fit, chi^2 = 9.83 | — | phi_0 -0.9 sigma, gamma_2 +1.2 sigma off CF's |

## HESE starting events (data set "events")

| Row | Source says (evidence) | We do (evidence) | Status | Decided by | Size |
|---|---|---|---|---|---|
| T1 observable | 102 events; deposited energy, morphology, zenith, length; cuts 60 TeV–10 PeV deposited, double cascades 10–1000 m (HS1; tools/hese_check.py docstring quoting data_loader.py) | the release's binning and cuts (hese.py; events.py) | match | him (2026-10-14, HESE plan) | — |
| T2 assumptions | the example fit: SPL with a nu/nubar ratio prior; astro per particle type = norm x 1e-18/6 (weighter.py:25-27) | per-particle nu and nubar fluxes from the solver (EV1, EV2) | match | — | — |
| T3 counterpart | N = T sum_i w_i Phi_type(E_i, dec_i) (release) | the same, per event (hese.py:204-252) | match (T11) | — | — |
| T4 energies | MC true energies span the release's simulation | flux zero outside 0.995 TeV–30 PeV (EV3) | unverified: events with true E > 30 PeV get no astrophysical flux | me | not measured |
| T5 sky | MC has no azimuth; true zenith | band of true dec (EV2) | match | — | — |
| T6 components | astro, conventional (pion + kaon), prompt, muons; self-veto per event (HS1) | astro, MCEq conventional and prompt with the release's self-veto, release muons (EV6) | match in components; the conventional model is ours (MCEq) | him (2026-10-14, "MCEq + Barr with the release's self-veto") | — |
| T7 response | MC weights + detector systematics by splines (HS2) | the release's per-event detector weights, exported (EV6, EV9) | match | him (2026-10-14) | per-bin model up to 0.02 in -2 ln L at fixed nuisances (events.py:19-22) |
| T8 nuisances | the release's priors (HESE_fit.py) | conv 0 +- 0.4, muons 1 +- 0.5, detector priors from the release (EV7) | unverified against HESE_fit.py line by line | him (2026-10-14: "the release's priors") | — |
| T9 likelihood | derived from arXiv:1901.04645 (accounts for limited MC statistics) (HS2) | plain Poisson with floors (EV8) | **mismatch** | not recorded | not measured |
| T10 overlaps | DR2 README: "should not be combined with any other releases" (DR1) | HESE and DR2 in one likelihood, no overlap removal | **mismatch**: shared events (starting tracks in DR2's selection) not removed | him for the combination (2026-10-14); overlap not recorded | not measured |
| T11 reproduction | the release's own code | our per-event weighting equals it to 1e-14 (736,814 events, case A); the solver path to a uniform 1.0036 (output/sky/hese_check.json, commit e28f70f, 2026-10-14) | match as of e28f70f; not rerun on the current code | — | the 0.36% is the redshift grid's normalization offset |

## Through-going tracks, IceTracks-DR2 (data set "events")

| Row | Source says (evidence) | We do (evidence) | Status | Decided by | Size |
|---|---|---|---|---|---|
| T1 observable | per-season events with reconstructed energy and direction (DR1) | counts per (reco dec, reco E), IC79 + IC86 (EV5) | match | him (2026-10-12/14) | — |
| T2 assumptions | point-source selection; "muon neutrino effective areas", nu_mu vs nubar_mu convention not stated (DR2) | A_eff applied to F_nu_mu + F_nubar_mu (EV4) | supported by data/expectation 1.06–1.12 per configuration (DR2_TRACKS_PLAN.md:99; a factor 2 would show); not stated by the release | me | — |
| T3 counterpart | A_eff x smearing (DR2, DR4) | dr2.py fold (EV4) | match in form | — | — |
| T4 energies | simulated 100 GeV–100 PeV; areas to 10^8.7 GeV | flux zero outside 0.995 TeV–30 PeV (EV3); reco > 10^3.5 GeV | unverified: muons above 3.2 TeV reconstructed from neutrinos below 1 TeV, and neutrinos above 30 PeV, get nothing | me | not measured |
| T5 sky | true dec; Earth absorption not stated (DR2) | dec_reco >= -5 deg; absorption assumed inside A_eff (dr2.py:4-5) | unverified (the release does not say); a flat data/expectation ratio in declination (DR2_TRACKS_PLAN.md:99) is indirect support | me | — |
| T6 components | muon tracks from nu_mu CC, plus nu_tau CC with tau -> mu (in the data) | astro nu_mu + nubar_mu only (events.py:303) | **mismatch**: nu_tau-induced tracks left out | not recorded | not measured |
| T7 response | IC86 area for all IC86 seasons (DR3) | as the release | match | — | — |
| T8 nuisances | — | per-configuration norm 0 +- 0.1; Barr, cr index, prompt shared (EV7) | our choice | him (2026-10-14, per-config norms) | — |
| T9 likelihood | limited MC statistics in the smearing (DR4) | Poisson, no MC-statistics term (EV8) | mismatch (small expected, not measured) | not recorded | not measured |
| T10 overlaps | as for HESE | as for HESE | **mismatch** | — | not measured |
| T11 reproduction | IceCube's own diffuse nu_mu result with these events: none in the release | data/expectation with the atmospheric nominal only (DR2_TRACKS_PLAN.md:99) | unverified: no reproduction of a published astrophysical fit | — | — |

## Forecast, eight telescopes (data set "forecast")

| Row | Source says (evidence) | We do (evidence) | Status | Decided by | Size |
|---|---|---|---|---|---|
| T1 observable | — (no data; Asimov) | events per channel, reco energy, local zenith (FUT1, FUT3) | — | him (2026-10-14/15) | — |
| T2 assumptions | NuFlavorSingleSource 39088bd's detector models (FD1). **His STOP in that project (CLAUDE.md:14, Oct 15, 08:54): "every result below is invalid. Unvalidated detector responses (shapes chosen by us, one or two parameters fixed by reported numbers), Gandhi et al. 1998 cross sections, Earth propagation without NC or nu_tau regeneration, and the cross-section uncertainty applied at detection only."** Replaced there by Plan A (FD2). | the vendored 39088bd model, unchanged (FD1; analysis/future_events.py) | **mismatch with his own verdict on that model** | the vendoring: him (2026-10-14/15, "the eight detectors as in that project"); its invalidation in the other project the same day was not carried over here | not measured |
| T3 counterpart | — | per-site flux x responses (FUT1) | match in form | — | — |
| T4 energies | reco 0.1 TeV–3.2 PeV (double cascades from 63 TeV) (nfss_forecast.py:486-488) | flux zero outside 99 GeV–30 PeV (FUT1) | match at the lower edge; the upper edge is unverified (responses above 30 PeV true energy) | me | not measured |
| T5 sky | sites (SITE_LAT) | per-site cos-zenith bands (FUT1) | match | — | — |
| T6 components | atmospheric (MCEq) + astro | the same; Earth absorption inside the responses (FUT4) | see T2 (no NC or nu_tau regeneration) | — | — |
| T7 response | see T2; [ASSUMPTION]s flagged in the code (FD4) | vendored | see T2 | — | — |
| T8 nuisances | NuFlavorSingleSource's priors | FUT2 | match to the vendored code | him (2026-10-15) | — |
| T9 likelihood | Asimov, profiled Poisson | FUT3 | match | — | — |
| T10 overlaps | — | — | N/A | — | — |
| T11 reproduction | — | none | unverified | — | — |
| attribution | — | paper Sec. IV: "the detector models of Ref. [Song:2020nfh]"; no link to Song:2020nfh anywhere in the code, the plan or NuFlavorSingleSource (FD5) | **mismatch** (paper text) | me (paper draft) | — |

## Statistics (S rows)

| Row | Check | Evidence | Status | Decided by |
|---|---|---|---|---|
| S1 | each Table II row against the code | configs; events.py:44-46; flux_adapters.py:59 and Atmospheric().priors() (Barr widths 0.122-0.40); NF.DET_PRIOR (eff 0.1, escale log10(1.1), misid 0.5); future_events.py:50 | match, except: (a) the sum-of-masses prior is 0.09 eV for the forecasts (configs), while Table II gives 0.12 for all; (b) Sec. IV attributes the priors to Song:2020nfh, but they are NuFlavorSingleSource's (flux_adapters.py:53-59) | him (values) |
| S2 | profile and minimizer convergence in the final runs | leads only: repair passes, exact profile (docs 11e), campaign findings ("20 cosmetic fit misses") | unverified for the final runs (coverage jobs final-cov running) | me |
| S3 | the reading: per mass, Delta chi^2 >= 3.84 from min(column, no-effect fit); upper limits only; specks dropped; linear interpolation in log10 g | exclusion.py:1-35, 181, 199 | match with paper Sec. IV (PAP214-218) | upper-limit reading and specks rule: not recorded; figures shaded: him |
| S4 | forecasts on Asimov data at the CF BPL | FUT3 | match | him |
| S5 | correction layers and their order (recheck, refine_recheck, rerun_rs, reeval_events, rho_fix last) | exclusion.py:67-103; tests/test_exclusion.py::test_rho_fix_replaces_last | match | me |

## Model side (M rows)

| Row | Check | Evidence so far | Status |
|---|---|---|---|
| M1 | Lagrangian, charges, signs | dark portal: G = g_D u u^dag (+ kappa I; kappa = 0 in the final planes) (dp_absorption.dark_portal_coupling); SI: lambda = U^T G U (dp_si.py:18-22); gauge models: matter charge (q_n, q_p, q_e) = (0, 0, 1), texture diag(1, -1, 0) or (1, 0, -1) (dp_models.spec, texture; dark_portal_solver.py:2665-2673) | match for L_e - L_mu, L_e - L_tau. Dark portal and secret interactions: PENDING (dp_absorption, dp_si couplings; PHYSICS_AUDIT 1a-1l are leads). |
| M2 | mediator widths and channels | Gamma_nu = M Tr(GG^dag)/24 pi (dp_absorption; dp_mediator pair widths with masses); gauge: charged-lepton pairs by direct coupling (dp_models.visible_width); kinetic mixing at loop level not included | partial; the loop-mixing omission is unmeasured. Dark portal and SI widths: PENDING (dp_mediator). |
| M3 | each Table I cell <-> the code switch per model | regime switch SI_COSMOLOGY_MIN_M_EV = 1e3 eV and SI_HARD_MIN_G = 1e-4 (dark_portal_solver.py:891-894, _cosmology_mode 1100-1135): hard scatterings and pairs above 1 keV, below it only for max|eig G| >= 1e-4 (Table I row 4: match for L_e-L_alpha and the dark portal). Soft scatterings: dark-portal phase with relic_scattering (379), dropped where the integrated jump < 1e-4 (SOFT_JUMP_MIN, 898: not in Table I; numerics). Energy loss: dp_relic_loss in the dark-portal phase (395-399); SI: cosmological loss for the vector only (dp_si_light.py:366-379; Table I 'scalar None': match). Longitudinal mode: dp_si_light.GOLDSTONE_MODELS = (L_e-L_mu, L_e-L_tau), below 1 keV with relic scattering, Y >= 1e-4 (dp_si_light.py:843-845; dark_portal_solver.py:1400-1407). | partial. Table I's 'Included' for the longitudinal mode lacks 'below 1 keV' (paper precision). Decays, decays of the emitted mediators, absorption cells, and SI 'All masses' for hard scatterings: PENDING. |
| M4 | regime boundaries (1 keV, unitarity, validity) | — | PENDING |
| M5 | sources | SFR to z = 4 (docs 8; paper L299); BPL (CF5); compositions (MOD6); Galactic (GL1) | match; nu:nubar = 1 at the sources is my choice (docs 10) |
| M6 | Earth, Sun, Moon | SM potential with PREM electrons, tested (test_sm_earth_potential.py); on only for L_e - L_alpha below 1 keV (CFG use_SM_potential) | partial; the rest PENDING (PHYSICS_AUDIT 4a-4c are leads) |
| M7 | numerics in the runs' regime | docs 17-19; CONVERGENCE_AUDIT_2026-10-25 (lead) | PENDING |

## Step 4: completeness (scratch_audit/i_completeness.py)

- (a) Every item has a status: 0 without.
- (b) Every config key of the final runs (all 738 configs) is an item: none missing.
- (c) Every docs entry (82) and paper sentence (424) is an item.
- (d) Of the 357 functions on the chi^2 path (static graph, over-inclusive), 203 are cited by an item. The other 154 were classified (scratch_audit/i_unreferenced_classified.csv):
  - 69 are machinery: caching, IO, parsing, pools, checkpoints, internal coordinates, serialization;
  - 21 are not on the final runs' path (mixing templates, NuFIT prior, flavor Asimov);
  - 44 are covered by existing items (EV, FUT, MOD, HESE rows);
  - 1 is a diagnostic;
  - 19 led to five new items:
    - **MAP1**: parameters mapped to the solver configuration (tested);
    - **ASM1**: the Asimov truth model, with relic scattering and decays off at a coupling of 1e-34 (match);
    - **GRAD1**: the exact gradients (tested);
    - **CACHE1**: the flux cache's key covers the solver code but not its data files (a risk, unmeasured);
    - **PROF1**: the exact profile's inheritance shortcut (my choice, docs 11e).
- (e) Every file:line cited in evidence exists. One citation was out of range (spectra.py:83-90) and is corrected to 83-88.

Limitation: the call graph is static. A dynamic trace (coverage of one chi^2 evaluation per data term) would need the flux cache on Tycho; it was not run.
