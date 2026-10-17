# Simplifications and assumptions

Started 2026-10-17 at Mauricio's request. One entry per simplification or assumption in the analysis: what it is,
why it is acceptable (or not yet), how large its effect is where known, and its status. Keep it current: add an entry
whenever a new one is made, and update the status when one is quantified or removed.

Status words: **accepted** (deliberate, effect understood to be small or conservative), **to quantify** (believed
small, not yet measured), **open** (decided to remove or change, not yet done).

## Propagation

1. **Absorption in the Earth applied after the flavor conversion.** *To quantify.* (2026-10-17)
   The solver propagates only flavor through the Earth (the new force's potential), never removing neutrinos; the
   standard absorption comes from the detector responses (HESE's Monte Carlo propagates each event through the
   Earth; DR2's effective areas include absorption; the future detectors multiply by NuFlavorSingleSource's survival
   factor). So the flavors are converted over the whole path, then absorbed with the flavor mix that arrives. This is
   exact where absorption is the same for every flavor, which holds for CC + NC absorption over most of the range;
   it is approximate where absorption depends on flavor: anti-electron neutrinos near the Glashow resonance (6.3 PeV)
   and tau-neutrino regeneration, where the flavor mix along the path matters. A proper treatment propagates flavor
   and absorption together.

2. **No cross-section variation.** *Accepted.* (Mauricio, 2026-10-12 and 2026-10-15)
   Cross sections are fixed at the values in the public responses. Varying them properly needs the Earth propagation
   with regeneration and effective areas rebuilt with the varied cross sections, which we do not do.

3. **Standard matter effects off in the Earth** (`use_SM_potential: false`). *Accepted.*
   At TeV-PeV the standard matter potential barely changes the flavor of neutrinos arriving as decohered states.

4. **New physics acts on the astrophysical flux only, not on atmospheric neutrinos.** *Accepted (conservative).*
   The atmospheric neutrinos crossing the Earth would also feel the new potential; leaving that out can only weaken
   the bounds.

5. **No t-channel scattering on the relic neutrinos.** *Accepted, with a range restriction.*
   Not modeled; above couplings of ~1e-7 it matters, so the coupling-split variants stop at x = 1e-8.

6. **Galactic component propagated adiabatically** from its emission points to the Sun (Landau-Zener parameter
   >= 1.5e5 wherever checked; `tools/galactic_adiabaticity.py`). *Accepted.*

## Sources and spectrum

7. **The extragalactic spectrum is IceCube's 2026 broken power law with the break fixed** at 10^4.39 GeV, both
   indices free (the lower one in [0.5, 2.5], while IceCube's 1-sigma range reaches 0.01). *Accepted.* (2026-10-15;
   `docs/dev/BROKEN_POWER_LAW.md`) The two pieces are split at the energy today, each a population with its own
   redshift weighting.

8. **Sources without redshift evolution, up to z = 4.** *Accepted (fixed choice).*

9. **Source flavor composition** at pion decay (1/3, 2/3, 0) in the nominal planes; fitted as (x, 1 - x, 0) in the
   `_fx` twins, extragalactic and Galactic independently. No tau neutrinos at the sources. *Accepted.*

10. **Neutrinos and antineutrinos 1:1 at the sources.** *Accepted.*

## Standard neutrino physics

11. **Mixing parameters and the lightest neutrino mass fixed in the campaign** (NuFIT 6.0, normal ordering with SK;
    m_lightest = 0.01 eV). *Open.* (2026-10-12: decided to vary the mixing with NuFIT 6.1's chi^2 and m_lightest
    with the prior on the mass sum, both orderings; the campaign's planes do not do it yet.)

## Data and detectors

12. **Flavor-composition data (IceCube's MESE 2025 measurement, the 2050 projection) compared with flux-weighted
    flavor fractions** averaged over 10 TeV-30 PeV, not folded through the detectors' responses. *Accepted for now.*
    A detector measures flavor with each flavor's own effective area and cross section; folding the flux through each
    measurement's response would fix it (the event data sets do).

13. **IceCube's 2026 spectrum segments taken as independent** (their correlations are not published). *Accepted.*

14. **HESE's detector systematics: the profile uses a per-bin model, and the reported chi^2 is the exact per-event
    evaluation at the profiled nuisances.** The release's four-point interpolation of the systematics has kinks at
    its grid values, where the profile's minimum can sit. *Accepted.*

15. **DR2: IC79 + IC86, reconstructed energy above 10^3.5 GeV, declination above -5 deg,** a normalization per
    configuration (10% prior). *Accepted.*

16. **Future detectors: NuFlavorSingleSource's detector models** (effective areas parametrized and calibrated to
    published numbers, not the detectors' own tables), its systematics, no self-veto, each detector at full size for
    10 years. Projections use Asimov data. *Accepted.* (2026-10-14/15)

## Numerics

17. **The solver's redshift grid** (200 steps in the campaign): the flux normalization is accurate to a few 1e-3
    (first-order error). *Accepted.*

18. **Interpolation in the spectral indices** between solver runs at 3 nodes (7 for the MeV planes' upper index):
    at worst 0.35% in flux and 9e-4 in flavor fractions for the lower index (strong MeV absorption); ~1e-5 for the
    potential. *Accepted.* (2026-10-15)

19. **Sky resolution:** HEALPix N_side 8, the Earth crossed on 2 chords per direction band, the Sun averaged
    isotropically. *Accepted.*
