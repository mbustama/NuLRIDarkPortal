"""
IceCubeRates -- expected IceCube event distributions for the fluxes the solver computes (the event-rate module of
docs/dev/ANALYSIS_DESIGN.md; layout agreed 2026-07-29, memory 'IceCube event-rates plan'):

  northern_muons/   through-going muon tracks with IceTracks-DR2 (docs/dev/DR2_TRACKS_PLAN.md, task 3);
  flux_adapters.py  fluxes as functions of energy and declination: power laws, the atmospheric background
                    (tools/atmospheric_background.py: daemonflux and MCEq), and propagate_sky's output;
  starting_events/  HESE (task 1; tools/hese_check.py has the exact weighting).

Fluxes are in GeV^-1 cm^-2 s^-1 sr^-1; energies in GeV; declinations in radians (equatorial J2000).
"""
