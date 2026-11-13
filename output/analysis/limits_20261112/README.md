# Limit curves, 2026-11-12

95% C.L. excluded couplings of every plane run on Tycho up to 2026-11-12 22:00 (set1-ml: 22:47), finished or partial, exported by
`tools/export_limits.py` from the plane results copied from the Tycho checkouts (summaries, records, fits,
manifests; no solver caches; the local copy was deleted after the export, the records stay on Tycho until the limits
are final, then `tools/collect_limits.sh` exports the final limits and deletes the checkout).

Each plane folder holds `limits.json`: per mass column the excluded intervals of log10 g, the lowest excluded coupling,
`limit_below_grid` (excluded down to the smallest coupling computed), whether the column is fully computed; the
blocks read (finished or partial); the no-effect chi^2; the not-modelled boundary of its configuration. `data/` holds
the joined plane in the format `analysis/exclusion.py` reads. `index.json` per checkout lists its planes.

| Folder | What | State |
|---|---|---|
| nulri_refit_set/campaign_si/now_icecube | Secret interactions, set 1, MeV and ultralight (64 planes) | complete |
| nulri_refit_set/campaign_si/now_mese_dr2 | Secret interactions, set 2, MeV (29 planes) | 22 complete; run stopped 2026-11-12 (disk) |
| nulri_refit_set/campaign_si/now_mese_dr2_dec10 | Set 2 variant (3 planes) | complete |
| nulri_set1_final/campaign_final/now_icecube | Dark portal, set 1, ultralight (6 planes) | complete |
| nulri_set1_b/campaign_final_b/now_icecube | Dark portal, set 1, the 1e-2 eV column (6 planes) | complete |
| nulri_set1_sm/campaign_final_sm/now_icecube | L_e - L_mu, L_e - L_tau, set 1, ultralight (4 planes) | partial, running |
| nulri_set1_ml/campaign_ml/now_icecube | Dark portal and L_e - L_alpha, set 1, MeV and ultralight (56 planes) | 53 complete (run ended 2026-11-12 22:47); the 3 partial ones (dp_mu_ul, dp_tau_ul, dp_tau_ul_fx) are older runs, replaced by the complete ones in nulri_set1_final |
| nulri_forecast/campaign_si/proj_future_planA | Forecast pilots, Plan A (2 planes) | pilots only |
| */check_v3a_flux, check_v3b_spl, check_v5_rho | Review items 3 and 5 checks | checks |

No-effect chi^2: 9.879 (set 1, pion-decay composition), 9.613 (set 1, fitted composition), 944.9 / 948.2 (set 2),
1.639 (forecasts). Not-modelled regions are recorded, not applied: hatch them as the figures do.
