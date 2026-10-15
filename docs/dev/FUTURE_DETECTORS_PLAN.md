# Plan: future detectors' event forecasts for the projections

Status: AGREED 2026-10-15 (Mauricio took the three recommended options below). Decided 2026-10-14 (memory: future-detectors): port
NuFlavorSingleSource's eight detectors (IceCube, IceCube-Gen2 = IceCube's strings + the new ones, KM3NeT/ARCA,
Baikal-GVD, P-ONE, TRIDENT, NEON, HUNT), each at full size for 10 years; their event forecasts REPLACE the 2050
flavor surface in the projections (the same future events: no double counting); present-day fits stay as they are.

## What NuFlavorSingleSource has (its committed HEAD 39088bd, read-only; its working tree has another session's edits)

`code/detectors.py`: each detector's parameters, `calibrate()` against published numbers (track and shower
effective areas, thresholds, volumes, depths, latitudes, PSFs, double-cascade efficiencies anchored on IceCube's MESE
measurement and on the water detectors' studies, mis-identification fractions). `code/forecast.py`, class `Detector`:
per channel (tracks 'trk', showers 'shw', double cascades 'dc'), flavor and nu/nubar, effective areas on a true-energy
grid (100 GeV - 10 PeV) with an energy proxy (the muon's energy for tracks, the deposited energy for showers),
Gaussian migration to reconstructed-energy bins, a sky acceptance and the Earth's absorption versus the local
zenith (`earth_survival`: two densities, no regeneration), atmospheric neutrinos (one MCEq table at the local
|cos Z|, Barr and cosmic-ray-index derivatives, prompt), atmospheric muons for down-going showers (per-detector
multipliers and gates), and systematics (per-block background normalization, efficiency, energy scale,
mis-identification, cross sections). It is built for a point source (rings around it); the diffuse astrophysical
flux enters only as a background, and `allsky_rate` gives all-sky rates for the calibration.

## What the port does here

A diffuse analysis per detector: expected events in (channel, reconstructed-energy bin, local cos-zenith bin) for
the solver's astrophysical flux (extragalactic and Galactic, per flavor and nu/nubar), plus the atmospheric
backgrounds, in a Poisson likelihood with the same machinery as the HESE and DR2 blocks (analysis/events.py): the
detector systematics and background normalizations profiled inside the block, shared atmospheric (Barr) and
astrophysical parameters. The data of a projection: the Asimov expectation of standard physics with pion-decay
sources (as the 2050 surface's centre).

The flux per local zenith at a site of latitude phi: the solver's Earth crossing depends on the zenith only, so
- extragalactic (isotropic): the solver's flux per cos-zenith band (its per-direction output, whose bands are
  zeniths) is the same at every site;
- Galactic (anisotropic): the emission's eigenstate populations per declination (before the Earth;
  `_galactic_emission_populations`) are carried to the site's zeniths over a sidereal day (cos Z = sin phi sin dec +
  cos phi cos dec cos H) and through the Earth with the solver's per-zenith responses (`_galactic_responses`);
- then times the Earth's absorption per flavor (`earth_survival`, CC + NC): the solver propagates flavor, not
  absorption, so nothing is counted twice.

## Decisions (Mauricio, 2026-10-15: all three as recommended)

1. Sky binning: local cos-zenith bins per channel (the Earth's absorption and potential, the atmospheric
   background's zenith shape) [recommended], or all-sky per channel (as NuFlavorSingleSource's allsky rates).
2. The Galactic component at non-polar sites: the full mapping above [recommended], or isotropic.
3. Backgrounds and systematics as NuFlavorSingleSource [recommended: consistency], no self-veto (it has none).

## Steps

1. Port `detectors.py`'s parameters and `calibrate()` and `forecast.py`'s response pieces (effective areas,
   migration, acceptance, survival, atmospheric tables) into `IceCubeRates/future/`; check every detector's
   calibration numbers against NuFlavorSingleSource's own output.
2. The solver: per-zenith fluxes for any site (extragalactic bands; Galactic populations per declination and the
   per-zenith responses), tested against propagate_sky at the South Pole.
3. The event block (analysis/events.py): expected counts, Asimov data, profiled nuisances; validated by reproducing
   NuFlavorSingleSource's all-sky rates for a reference flux, and against IceCube's HESE/DR2 blocks where they
   overlap (IceCube's own detector).
4. Projection configurations (the campaign's third data set, in place of the 2050 surface) and their planes on
   Tycho.

**No cross-section variation (Mauricio, 2026-10-15).** NuFlavorSingleSource varies the CC and NC cross sections by
5% as templates of the in-detector rates; a proper variation needs the Earth propagation with regeneration and
effective areas rebuilt with the varied cross sections (its calibration anchors embed the nominal ones), so the port
leaves it out, as this project's analysis does. (A first port carried the templates, and a linearized absorption
term; with them the flavor chi^2 moved by ~1%. Removed.)
