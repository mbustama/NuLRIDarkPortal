# Audit 2026-11-08, step 5: the independent pass compared with mine

The independent agent read only the sources and the code (not my table, plan, decision log or memory); its table is scratch_audit/second_pass/table.md.
- "Verified by me" means I re-read the code or source myself after its report.
- Its other claims are leads.

## Found by both
1. MESE (flavor) and CF (spectrum) share events (2514 cascades, 1799 tracks; CF Table 1), and we count them twice.
2. The spectrum term leaves out the Galactic flux, while CF's Table 10 fit has no Galactic component.
3. The spectrum counterpart (E^-2 with the same number of neutrinos, all flavors over 3) is not the measured normalization.
4. The flavor surface vs IceCube's quoted levels: f_tau = 0 at 85.8% (IceCube 91.9%), f_e = 0 at 97.8% (98.7%). Both of us computed the same numbers.
5. HESE: plain Poisson, while the release accounts for MC statistics.
6. HESE nuisances differ from the release: the cosmic-ray centre is 0 (release -0.05) and the prompt has a prior (release: free).
7. The paper says DR2 bins of 0.2 in log10 E; the code uses 0.1.
8. The HESE–DR2 overlap is not removed; the DR2 README says not to combine its events.
9. The forecast uses NuFlavorSingleSource 39088bd's model, which that project declared invalid (STOP, Oct 15).
10. The paper credits Song:2020nfh with the forecast models, backgrounds and priors. The agent adds that Song et al. have no TRIDENT, NEON or HUNT models and no Barr priors.
11. Table II/III says 0.12 eV, but the forecasts use 0.09.
12. Table I's longitudinal-mode row is imprecise.
13. The flavor window, 10 TeV–30 PeV vs IceCube's 5 TeV–10 PeV (a mismatch row in its table, T4; not in its summary list).

## Found only by the independent pass
- **Verified by me:**
  - (a) **HESE live time is 7.22 yr** (227,708,167.68 s), not the 7.5 yr in our data table, whose caption says "the exposure is the live time".
  - (b) **Non-polar telescopes reuse the South Pole's declination bands as local-zenith bands** (model.py:1360-1374). The docstring assumes the flux is isotropic before the Earth. That fails wherever the Milky Way's crossing acts, as for ultralight L_e - L_alpha. Size not measured.
  - (c) **The paper-figure notebook reads campaign_ml.** That includes the dark portal's potential planes with Z_D–Z mixing (dp_*_pot_*, cells 12-13), dropped on 2026-10-24. The paper's Fig. 4-7 are mock-ups for now. The final figures must read the final runs (the merge is pending).
  - (d) **The lightest-mass scan also runs in the dark portal's ultralight planes** (campaign_final: 162 dp_ul configs with ml_scan). Paper Sec. IV says "in the planes of MeV mediators".
  - (e) **Thin-band removal is applied to every ultralight plane**, fixed composition too (exclusion.merged, exclusion.py:296, unconditional). The paper ties the band to fitted compositions.
  - (f) **Soft scatterings for secret interactions:** on the Galactic paths, the vector only (dp_si_light.py:549; docstring 404-407). Table I lists them "below 1 keV" for the scalar too. The cosmological phase: unverified.
  - (g) **The paper says "The coverage tests check both effects"** (main.tex:625). No coverage-test code exists in this repository (grep).
  - (h) **The "final-cov" and "final-band" jobs I reported as ours belong to GeoneutrinoDecay** (/lustre/astro/mbustama/GeoneutrinoDecay-final). Ours are si-ul (81), si-mevpilot (1) and sm-rerun (96).
- **Leads, not verified by me:**
  - (i) The release's nu/nubar nuisance is dropped, and the HESE detector-systematic ranges are cut to about ±2 sigma. Our bounds are the exported grid edges (events.py:280); the grid range was not read.
  - (j) The DR2 spectral index from a stored output of older code: 2.60, against IceCube's 2.37 ± 0.09.
  - (k) Naming: "IceTracks-DR2" vs the README's "IceT-DR2". Also "tens of MeV" (Sec. II) vs ≈100 MeV (Table I).
  - (l) arXiv:2606.17132's page shows a notice that it is superseded.

## Found only by me (the agent did not list them)
- Number weighting vs event weighting of the flavor composition.
- The flavor likelihood assumes IceCube's own BPL, not CF's.
- nu_tau-induced tracks are left out of DR2.
- My undocumented choice nu : nubar = 1 at the sources.
- "9–13%" should read 1.1–13%.
- "We use the likelihood" of the flavor measurement, when we use a fitted surface.
- The flux cache's key does not cover data files.
- Measured: the BPL refit to Table 10 with our counterpart (phi_0 1.615, gamma_1 1.48, gamma_2 2.82 vs 1.77, 1.31, 2.735), and the gamma_low bound's cost.
