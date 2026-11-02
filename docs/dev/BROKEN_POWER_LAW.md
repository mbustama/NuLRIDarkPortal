# The extragalactic flux as IceCube's broken power law

Decided 2026-10-15 (Mauricio, "Add the bend to the model"): IceCube's 2026 combined fit of tracks and cascades
(PRD 113, 062002 (2026), Table 3, "CF BPL"; NuFlavorSingleSource's CF26) bends below ~25 TeV: 1.77e-18
GeV^-1 cm^-2 s^-1 sr^-1 per flavor at 100 TeV, E^-2.735 above E_b = 10^4.39 GeV = 24.5 TeV, E^-1.31 (+0.51 -1.30)
below. A single power law fitted to the same data and extended down carries ~3x IceCube's flux at 10 TeV, ~60x at
1 TeV, ~1000x at 100 GeV (output/figures/future_low_energy_flux.png). The future detectors see down to 100 GeV,
DR2 down to 3 TeV, IceCube's own spectrum segments down to 4.6 TeV: the model takes the bend for every data set.

## The model (analysis/model.py)

Two source populations split at the energy today E_b, each a power law: p_index above the break (the upper index,
free as before) and p_index_low below (a new analysis parameter), normalized so that without new physics the flux
at Earth is exactly the broken power law, continuous at E_b (sky.extragalactic_norm is the upper piece's flux at its
reference energy: 1.77e-18 at 100 TeV, so norm_astro = 1 is IceCube's fit).

Each piece is its own solver run with the sources emitting only on its side: source.injection_E_min_GeV (upper)
and injection_E_max_GeV (lower), a window on the energy the neutrino has today, E_source / (1+z) (dp_cosmo's
injection kernel and dp_si's transport; src/dp_config.py). The transport is linear in its sources, so the pieces
add up to the whole power law (tested to rounding), and everything downstream -- flavor conversion, absorption, the
secondaries an absorption above E_b sends below it -- is each population's own. Each piece is interpolated in its
own index (p_nodes above, p1_nodes below; cubic spline in log flux, as before), and joined in physical units:

  F = K(p) F_upper(p) + (E_b / E_ref)^(p_low - p) K(p_low) F_lower(p_low)

with K the whole power law's extragalactic scale (the closed form). E_b is fixed at IceCube's value (its
uncertainty, 0.1 in log10, would multiply the solver runs: each break position is a run). The Galactic component
keeps its own spectrum (its template).

Choices (technical): p_index_low in [0.5, 2.5] on three nodes (IceCube's 1-sigma range reaches 0.01; the lower
bound keeps the solver's power laws positive and the interpolation short); p_index on its nodes as before (3 for
the potential planes, 7 for the MeV ones).

## The campaign (tools/make_campaign_configs.py --spectrum bpl)

analysis/configs/campaign_bpl/<set>/, outputs under output/analysis/campaign_bpl/; the single power law's campaign
(analysis/configs/campaign/) stays as the comparison. Sets: now_ic26 and proj_2050 (their grid down to 4.57 TeV,
where IceCube's lowest spectrum segment starts: two segments below the break; the flavor surfaces keep their
10 TeV - 30 PeV band), now_events (HESE + DR2, from ~1 TeV), proj_future (the future detectors, from 100 GeV, on
Asimov data at IceCube's fit: norm_astro 1, p_index 2.735, p_index_low 1.31). Starts at IceCube's values.

Cost: each point runs both pieces (3 + 3 runs for the potential planes instead of 3; 7 + 3 for the MeV ones); any
change of src/ invalidates the solver-run cache anyway. tools/campaign_packs.py estimates ~6,300 core-hours for the
four sets (to calibrate on Tycho).

## Tests (tests/test_spectral_break.py)

The windows add up to the whole power law (dp_si with strong absorption and secondaries; the dark portal's kernel
with a potential), nothing moves up in energy; without new physics the model's flux is the broken power law (each
piece's shape exact, joined at the break); with p_index_low = p_index the pieces give the single power law (exactly
at the nodes, within the interpolation's error between them; fluxes and per-direction bands); the configuration
keys.
