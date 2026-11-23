# Spine of the paper (draft 2026-11-07)

The prose is written one section at a time against this file (paper-writing skill, Step 6). Every number names its
source; numbers marked *prelim.* change after the final runs. The full planning is in `paper_planning.md`.

## Vocabulary: one name per concept

| Use | For | Not |
|---|---|---|
| the trip | the path from the sources to the detector | journey, propagation path |
| stop | one place on the trip: the arrival at the Milky Way, after it, near the Earth, the detector | stage (in text), checkpoint |
| the flux along the trip | the flux vs. position, stage by stage (Fig. 4, `fig:species`); "stops" only for the four places of Appendix C | trajectory |
| species | each of $\nu_e, \nu_\mu, \nu_\tau, \bar\nu_e, \bar\nu_\mu, \bar\nu_\tau$ | channel, type |
| state, eigenstate | an eigenstate of the Hamiltonian in the local potential; the mass states without a potential | level (in text) |
| mediator | the new particle, generically; $Z'$ ($L_e - L_\alpha$), $Z_D$ (dark portal), $\phi$ / $V$ (secret interactions) | boson, force carrier |
| relic neutrinos | the cosmic neutrino background | CνB (except once, at definition), cosmic neutrinos |
| potential | the coherent, flavor-changing potential from matter | matter effect, refraction |
| collisions on the relic neutrinos | absorption, scattering, together | interactions (ambiguous) |
| absorption | resonant absorption on a relic neutrino | attenuation |
| secondaries | the neutrinos an absorption returns at lower energy | regeneration (except in the equation's name) |
| decay | $\nu_i \to \nu_j$ + mediator | |
| secret interactions | neutrino-neutrino interactions through a mediator coupled only to neutrinos | self-interactions (define the equivalence once) |
| the dark portal | the mediator coupled through the mixing with a dark neutrino | dark neutrino portal (except once) |
| flavor composition; source flavor composition | $(f_e, f_\mu, f_\tau)$ at Earth; $(f_{e,S}, f_{\mu,S}, f_{\tau,S})$ | flavor ratios |
| the full calculation; the earlier standard | ours; the potential evaluated only at the detector | naive, approximate |
| where the models hold | below the line of perturbative unitarity | validity region |

Data sets, as in the figures' legends (never internal names such as `now_ic26`):
- **IceCube flavor + spectrum:** the flavor composition from 11.4 years of medium-energy starting events
  (IceCube:2025ole) and the spectrum of tracks and cascades combined (IceCube:2025ewu, its Table 10, CF
  column; the break from its Table 3, CF BPL).
- **IceCube events:** 7.5 years of high-energy starting events (IceCube:2020wum) and 11.6 years of through-going
  tracks (IceTracks-DR2).
- **Future telescopes:** the eight detectors combined, 10 years each.

## Section by section: claims and their sources

**1. Introduction.** Earlier work evaluated the potential at the detector (ABDN23, its Sec. 5, read 2026-11-07);
high-energy secret-interaction work treated the relics at rest until Wang, Xu & Zhou 2025 (text, read).

**2. Models.** Couplings and the potential: equation plan, Eqs. (1)--(3). The unitarity line for $L_e - L_\mu$,
$g' = 10^{1.76} M/{\rm eV}$ (`analysis.exclusion.not_allowed_boundary`, computed 2026-11-07).

**3. The trip.** Supporting results 1--5 of Step 1, with their sources ([nb], [pf]). The resonance energies, 1.0 and
5.76 PeV at $M$ = 10 MeV ([nb]). The four state-to-flavor mappings where the potential dominates (numerical check,
2026-11-07). Fig. 4, the species along the trip (`analysis/trajectory_figure.py`, Tycho job 25276): secret interactions
and the dark portal act only in the Universe; $L_e - L_\mu$ changes the flavor in the Milky Way at $10^{-26}$ eV (with
$\bar\nu_e$ rising to 0.53 and 22% lost) and near the Sun at $10^{-20}$ eV, between ~2000 and 300 AU; nothing
changes inside the Earth at 495 TeV.

**Figures (numbering of 2026-11-07):** 1 money plot; 2 the trip (illustration); 3 processes; 4 the species along the
trip; 5 the collision terms; 6--9 results; Appendix: 10 vertices, 11 the flux at the stops, 12 the relics' helicity.

**4. Data and method.** 95% C.L. per mass column, $\Delta\chi^2 \geq 3.84$ (decided 2026-10-13); the data sets above.

**5. Results.** Step 1's main-result table (*prelim.*); the money plot's first number: at $10^{-20}$ eV,
$f_\tau$ = 0.386 (full) against 0.000 (the earlier standard) (`tests/test_trips.py` set-up, 2026-11-07); the MeV
preference, $\Delta\chi^2$ = 3.6--10 (campaign notes, *prelim.*).

**6. Summary.** No new numbers.
