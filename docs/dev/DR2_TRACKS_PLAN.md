# Plan: northern tracks with IceTracks-DR2 at N_side 64 (Mauricio's task 3)

Status: **agreed 2026-10-14** (downloads of DR2 and PLEnuM to Tycho; all four seasons; rates and figures first;
Mauricio: "You could compute your own MCEq background, too" -- adopted, Section 2; "No need to compare against
DR1" -- step 5(a)'s PLEnuM/DR1 cross-check dropped, 2026-10-14; "For atmospheric background, you can use the same
treatment as NuFlavorSingleSource: MCEq with Barr parameters to vary as part of the statistical analysis" -- adopted,
Section 5, same day). Builds on the
decisions recorded on 2026-07-29 and 2026-10-12 for the event-rate module (memory: IceCube event-rates plan) and on
the per-direction output (docs/dev/DIRECTIONS_PLAN.md, steps 1-8, done; D1 addendum: tracks at N_side 64-128).

## 1. Goal

Expected through-going muon-track event distributions in IceCube for any flux the solver produces (dark portal,
secret interactions, standard), extragalactic and Galactic, per direction: rates per season in reconstructed
energy and declination, with the atmospheric background, from IceCube's own public response (IceTracks-DR2).
Not yet a likelihood (that comes with task 1, the HESE/IceCubeRates module).

## 2. Inputs (verified 2026-10-14)

| Item | Source | Size | Use |
|---|---|---|---|
| IceTracks-DR2 v3.1 (2026-10-05, CC0) | Harvard Dataverse doi:10.7910/DVN/MMIIZA | 3.65 GB, 41 files | per season (IC40, IC59, IC79, IC86 = 2011-2022): effective area A(E_nu, dec) (50 dec x 34 E bins up to 10^8.7 GeV, nu_mu + nubar_mu), smearing to (reco energy proxy, PSF, AngErr) per true (E_nu, dec) bin (14 x 0.5-dex E, 41 dec bins; 20^3 reco cells; 0.6-1.0 GB each), uptime (livetime) lists; event lists (not used for rates) |
| PLEnuM (upstream main, pinned commit; BSD-2) | github.com/PLEnuM-group/Plenum | 0.68 GB (mostly pickled resources) | the fold (aeff x smearing), the atmospheric background (MCEq fluxes), helpers; unmodified (decision 2026-10-12: our adapter ingests DR2) |
| Atmospheric background, our own | MCEq (Fedynitch et al.) and daemonflux (Yanez & Fedynitch 2023, MCEq calibrated to muon data), in their own conda environment on Tycho (MCEq's database, ~0.3 GB, fetched at first use) | small tables | nu_mu + nubar_mu at the South Pole versus energy (10 GeV-100 PeV) and zenith on DR2's declination bins: conventional from daemonflux, prompt from MCEq with SIBYLL-2.3d charm (2.3c's successor; MCEq 1.4 ships no 2.3c); MCEq + H3a + SIBYLL-2.3d as a cross-check; NRLMSISE-00 South Pole atmosphere, seasonal average. Model choices are configuration, not code. PLEnuM's own MCEq pickle only as an optional cross-check (converted once, in its environment) |
| Galactic templates and emission data at N_side 64 | tools/prepare_galactic.py --nside 64 on Tycho (CRINGE native 256, pi0 128, gas 64) | ~20-30 MB | kept outside the repository (lustre), with a config key pointing to them |

Facts recorded before (2026-10-12, streamed tables): DR2's smearing has DR1's 5-D format with 41 declination
bins (DR1: 3); the direction smearing is the PSF (true opening angle), not reconstructed declination, so a
reconstructed-declination cut needs a geometric PSF -> dec step; PLEnuM's resolution.py special-cases DR1's three
bands, so DR2 needs that step in our adapter. Earth absorption is in the effective areas (do not add it). DR2's
readme: do not combine its events with other IceCube releases (matters for later joint fits with HESE).

## 3. Steps

1. **Data on Tycho**: download DR2 (3.65 GB) to external/icetracks_dr2 with checksums; clone PLEnuM at a pinned
   commit to external/plenum; its own conda environment (MCEq, healpy, scikit-learn, mephistogram) -- its
   resources are pickles, loaded only there (as the HESE toolkit, isolated).
2. **Readers** (`IceCubeRates/northern_muons/dr2.py`, ours): effective areas, smearing (streamed to compact
   per-season arrays: marginalize what the rates do not need, keep (true E, true dec) -> (reco E proxy, reco dec
   via the PSF geometry)), uptimes; tests against the tables' own normalizations (each smearing slice sums to 1).
3. **Fluxes per direction for tracks**: propagate_sky with DR2's declination bins as bands (northern sky and the
   horizon; up-going tracks), N_side 64 for the Galactic maps (the finer reductions of item 3 above; a
   `galactic.data_dir` config key); nu_mu + nubar_mu per band and pixel.
4. **The fold** (ours): rate(reco E, reco dec) = sum over seasons of livetime x sum over (E, dec) of Phi x A x
   smearing, checked against PLEnuM's fold on DR1's table; the background from our MCEq/daemonflux tables
   (item 2); the Galactic component per pixel through the PSF (its plane is narrower than the tracks' resolution
   only below ~1 deg).
5. **Validation**: (a) ~~PLEnuM's own DR1 (IC86_II) fold reproduced by our adapter~~ (dropped, Mauricio
   2026-10-14) -- instead the reduction's own normalizations and the point-spread kernel against Monte Carlo
   (tests/test_icecube_rates.py); (b) with IceCube's 9.5-yr northern-tracks diffuse fit (ApJ 928, 50 (2022), arXiv:2111.10299, checked
   2026-10-14: nu_mu + nubar_mu 1.44 (+0.25/-0.26) x 1e-18 GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV, gamma 2.37 +- 0.09,
   constrained 15 TeV-5 PeV) plus
   the MCEq background, the expected reco-energy distribution against DR2's observed event counts per season
   (a units check, within the fit's uncertainties); (c) the Galactic component's track rate against IceCube's
   Galactic-plane track results where published.
6. **Figures and notes**: expected track rates per declination band and energy for standard physics, a dark-portal
   point and a secret-interaction point, with and without the Galactic component; the Galactic component's
   contribution in the northern sky.

## 4. Decisions for Mauricio

- **Downloads**: DR2 (3.65 GB) and PLEnuM (0.68 GB) to Tycho (Recommended), or DR2 only (our own fold, no PLEnuM).
- **Seasons**: all four configurations with their uptimes (Recommended, as decided 2026-10-12), or IC86 only.
- **Scope of the first deliverable**: rates and figures (Recommended), or straight to a tracks likelihood.

## 5. Implementation status (2026-10-14)

- **Step 1** done: DR2 v3.1 on Tycho (external/icetracks_dr2, 41 files, every MD5 checked against Dataverse's);
  PLEnuM cloned (external/plenum, e82343b) but no longer used (step 5(a) dropped; our own background).
- **Background**, NuFlavorSingleSource's treatment (Mauricio, 2026-10-14): MCEq SIBYLL-2.3d x H3a, the nominal flux
  (conventional + prompt) and its derivatives with respect to the 12 Barr parameters above 30 GeV (G, H, I for
  pi+-; W, Y, Z for K+-; Barr et al. 2006 priors) and the cosmic-ray index (prior 0.05), plus the prompt
  normalization (prior 0.5), all linear in the fit (IceCubeRates.flux_adapters.Atmospheric, templates by term);
  NuFlavorSingleSource's regions, priors, yield modification and cosmic-ray tilt copied unchanged (checked
  identical). Different, for IceCube: the South Pole's atmosphere (MSIS00_IC, January + July) and 20 production
  zenith angles dense near the horizon (theirs: US Standard, 8). daemonflux's calibrated conventional flux is
  stored as a cross-check. tools/atmospheric_background.py -> IceCubeRates/data/atmospheric_mceq_southpole.npz
  (Tycho, conda_envs/mceq: MCEq 1.4.2, daemonflux 0.9.0). A first table (daemonflux conventional + MCEq prompt,
  no nuisances; superseded) gave, straight up-going: daemonflux's 1-sigma error 11% at 1 TeV, 29% at 100 TeV;
  MCEq conventional / daemonflux = 1.01, 1.14, 1.00, 0.66 at 0.9, 9, 90, 900 TeV; prompt / conventional = 0.03,
  0.25, 1.4 at 9, 90, 900 TeV.
- **Fit check** (tools/dr2_validation.py): the nuisances above, the astrophysical normalization and index (priors
  25%, 0.2) and one normalization per configuration (10%; NuFlavorSingleSource's per-block background
  normalization) fitted to DR2's events in reconstructed energy and declination (Poisson) -- whether the treatment
  covers the data; not yet the analysis.
- **Step 2** done: tools/dr2_reduce.py -> IceCubeRates/data/dr2_irfs.npz (2.3 MB, 26 s on Tycho). Effective areas
  on 0.2 dex x 50 declination bins (uniform in sin dec, 0.04); smearing on 14 x (30, 52, 49, 41) (E_nu, dec) bins for
  IC40, IC59, IC79, IC86, onto 80 reconstructed-energy bins (0.1 dex) x 18 PSF bins (0.25 dex, 0.01-180 deg), every
  filled slice summing to 1 (to 4e-13); livetimes 376.4, 353.6, 312.8 and 3920.6 days (IC86: 11 seasons).
- **Step 4** done (IceCubeRates/northern_muons/dr2.py): the fold with the reconstructed declination from the PSF by
  spherical geometry (exact in the azimuth; each true bin split where the circle of radius psi reaches a
  reconstructed edge, directly or over a pole). A first version used plain Gauss-Legendre in sin(dec) and missed by
  6-9% wherever an edge cut a true bin, then 1.5% for psi near 180 deg at the poles; the Monte Carlo test caught
  both (now: worst 2.8 sigma over 1584 entries, statistics). The PSF moves 0.1-0.3% of an isotropic flux's
  tracks across dec = -5 deg and conserves the all-sky total exactly.
- **Step 5(b)-(c), 6**: tools/dr2_validation.py (output/tracks/dr2_validation.json, output/figures/dr2_validation.png,
  dr2_tracks.png).
