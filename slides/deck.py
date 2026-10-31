#!/usr/bin/env python
"""New forces on high-energy cosmic neutrinos: progress report to collaborators (2026-10-31). Content.

The machinery is build_deck.py (as for the TAMBO viewfinder talk), the figures make_figures.py. Every number is
from this repo -- notes/notes.tex, docs/, the plane summaries under output/analysis -- or from the Tycho runs logged
in the decision log, and is cited on its slide.

    python make_figures.py
    python deck.py
"""

import build_deck as bd
from build_deck import (
    FIGS, W, MARGIN, BODY_W, BLACK, BLUE, DEEPBLUE, GREY, WHITE,
    build, fit, picture, section, slide, table, textbox, title_slide,
)

UCPH = "1000000100000AC8000004049F3C1DF5.png"
VILLUM = "10000001000003280000010A710211B6.png"

BODY = 17


def fig(name, max_w, max_h, y, x=None):
    href, iw, ih = fit(FIGS / f"{name}.png", max_w, max_h)
    return picture(href, (W - iw) / 2 if x is None else x, y, iw, ih), ih


def column(paragraphs, y=2.6, size=BODY, w=BODY_W, space=0.32, x=MARGIN):
    t, h = textbox(x, y, w, paragraphs, size=size, line_height=120, space_after=space)
    return t, h


def deck():
    # ------------------------------------------------------------------ 1
    title_slide(
        ["New forces on high-energy", "cosmic neutrinos: progress"],
        "Mauricio Bustamante",
        "Niels Bohr Institute, University of Copenhagen",
        ["Progress report", "October 31, 2026"],
        [(UCPH, 21.137, 10.32, 6.286, 2.326),
         (VILLUM, 20.779, 12.628, 7.567, 2.413)],
        notes=("Where the analysis stands: the models, the full propagation, the data, first results,\n"
               "and what is left before the final plots."),
    )

    # ------------------------------------------------------------------ 2
    t, h = column(
        ["IceCube sees cosmic neutrinos of 10 TeV to tens of PeV, from sources out to redshift ~4. On the way "
         "they cross the sea of relic neutrinos, the Milky Way, the Sun and the Earth.",
         "A new, light mediator can ~~absorb~~ them, ~~scatter~~ them, make them ~~decay~~, or ~~change how "
         "their flavors mix~~, and so mark the energy spectrum or the flavor composition.",
         "**We compute all of these together, from the sources to the detector, and fit them to IceCube's data.**"],
        size=16)
    f, _ = fig("trip", 25.4, 6.4, 7.75)
    slide(lead="The idea: a new force acts on neutrinos on their way to us", extras=[t, f],
          ref="notes/notes.tex, Sec. “The transport equation in full”",
          notes="The one-line pitch: a light mediator changes the flux on the way; we model every step.")

    # ------------------------------------------------------------------ 3
    rows = [
        ["Model", "The mediator couples to", "Matter potential from", "Mediator masses"],
        ["**Dark portal** (`ν_e`, `ν_μ` or `ν_τ`)",
         "Neutrinos, through their mixing with a heavy neutrino: coupling `g_D |U_{α4}|^2`",
         "None in these planes", "10{{−30}} eV – 1 GeV"],
        ["`L_e − L_μ`, `L_e − L_τ`",
         "Lepton-number differences: neutrinos and electrons, coupling `g′`",
         "Electrons: cosmic, Galaxy, Sun, Earth", "10{{−30}} eV – 1 GeV"],
        ["**Secret interactions**: scalar or vector; Majorana or Dirac neutrinos",
         "Neutrinos only: `ν_e`, `ν_μ`, `ν_τ`, or all flavors equally",
         "None", "10{{−30}} eV – 1 GeV"],
    ]
    parts, end = table(MARGIN, 2.6, [0, 7.2, 15.6, 21.2], rows, size=14.5, row_h="auto", width=BODY_W, pad=0.25)
    t, _ = column(
        ["Each plane is one exclusion region, mediator mass against coupling. Every choice below is its own plane:",
         ("•", "~~Dark portal and lepton number~~: 5 models × 2 choices of source flavors (pion decay, or fitted) "
               "× 3 data sets = 30 planes below 10 keV, and 30 above."),
         ("•", "~~Secret interactions~~: 2 mediators (scalar, vector) × 2 kinds of neutrino (Majorana, Dirac) "
               "× 4 flavor choices × 2 mass ranges (below and above 1 keV) × 2 choices of source flavors "
               "× 3 data sets = 192 planes.")],
        y=end + 0.3, size=14.5, space=0.15)
    slide(lead="Three families of models", extras=parts + [t],
          ref="analysis/configs/campaign_ml, campaign_si · lepton-number models as in Agarwalla, Bustamante, Das, "
              "Narang, JHEP 08 (2023) 113",
          notes="The coupling axes: g_D times the heavy-neutrino mixing for the dark portal; g' for lepton number.")

    # ------------------------------------------------------------------ 3b
    f, ih = fig("feyn_couplings", 25.4, 6.8, 2.6)
    t, _ = column(["The dark portal reaches the light neutrinos only through their mixing with a heavy neutrino, "
                   "`ν_4`: its planes are drawn in the product `g_D |U_{α4}|^2`. The lepton-number boson also "
                   "couples to electrons, which makes the matter potential. The scalar and the vector secret "
                   "interactions are separate models, with planes of their own."], y=2.6 + ih + 0.3, size=16)
    slide(lead="How the mediator couples", extras=[f, t],
          ref="notes/notes.tex, Sec. “Matter charges and textures” (benchmark table)",
          notes="f stands for any fermion carrying the lepton number: nu_e and e with +g', nu_mu and mu with -g'.")

    # ------------------------------------------------------------------ 4
    f, ih = fig("feyn_processes", 25.4, 9.9, 2.6)
    t, _ = column(["All of these act together, in one equation (next slide). Grey lines: relic neutrinos, "
                   "56 per cm{{3}} per state today."], y=2.6 + ih + 0.25, size=16)
    slide(lead="What the mediator does, by its mass", extras=[f, t],
          ref="notes/notes.tex, Sec. “The transport equation in full”",
          notes=("Absorption: s channel on a relic. Scattering: t channel, forward-peaked below 1 keV.\n"
                 "Decay: open when the mediator is lighter than a mass difference. Potential: forward scattering\n"
                 "on matter, coherent. Longitudinal mode: the vector's q q / M^2 part, couplings ~ m_nu g / M."))

    # ------------------------------------------------------------------ 5
    t, h = column(
        ["`F_a(E, z)`: neutrinos per unit energy in state `a`, an eigenstate of `H = UM^2U^†/2E(1 + z) + V`; "
         "`E` is the energy observed today. The same for antineutrinos. Solved from `z` = 4 to today: 439 energy "
         "bins, 200 redshift steps (400 where decays act), each bin's processes together and exactly."],
        size=15)
    f, ih = fig("eq_cosmo", 25.4, 9.3, 4.95)
    slide(lead="The propagation equation, term by term", extras=[t, f],
          ref="notes/notes.tex, Eq. (transport_cosmo) · code: src/dp_sweep.py, src/dp_medium.py",
          notes=("Sources: star formation up to z = 4, a power law, flavors split over the eigenstates.\n"
                 "Absorption: Breit-Wigner on each relic mass state, thermal average; two secondaries.\n"
                 "Soft scatterings: conversions between states, damped where states decay fast.\n"
                 "Energy loss: scatterings that change the relic's mass state.\n"
                 "Decay: between eigenstates of the medium; daughters keep or flip helicity."))

    # ------------------------------------------------------------------ 6
    t1, h1 = column(["~~Galactic neutrinos~~ (CRINGE template) are emitted along each line of sight and "
                     "carried to the Sun with the same processes:"], size=16)
    f1, i1 = fig("eq_gal", 22.0, 2.0, 2.6 + h1 + 0.05)
    y = 2.6 + h1 + i1 + 0.2
    t2, h2 = column(["~~Extragalactic neutrinos~~ crossing the Galaxy: there, the lightest neutrino and "
                     "antineutrino states can decay into each other, which they cannot do between galaxies. So "
                     "we follow them along every line of sight too.",
                     "~~Sun and Earth~~: no collisions or decays there, only coherent flavor evolution in the new "
                     "potential, chord by chord through the Earth:"], y=y, size=16)
    f2, i2 = fig("eq_det", 22.0, 1.75, y + h2 + 0.05)
    t3, _ = column(["At the detector: the flux of each flavor, per band of declination or for the whole sky."],
                   y=y + h2 + i2 + 0.2, size=16)
    slide(lead="After the Universe: the Galaxy, the Sun and the Earth", extras=[t1, f1, t2, f2, t3],
          ref="notes/notes.tex, Eq. (transport_gal) and Sec. “From the Sun to the detector” · standard matter "
              "effects left out (negligible at these energies)",
          notes=("The Galactic relic column is ~1e-5 of the cosmological one, so no resonant absorption there.\n"
                 "The crossing matters for g/M >~ 1e-2 eV^-1."))

    # ------------------------------------------------------------------ 6b
    t, _ = column(
        ["The new potential is `V = AS`: `A` sets its size, `S` its flavor structure. For `L_e − L_μ`, "
         "`S = \"diag\"(1, −1, 0)`.",
         "The states that travel are the eigenstates of `UM^2U^† + σS`, with `σ = 2E(1 + z)A`. Their effective "
         "masses squared are on the right.",
         "~~Weak potential~~: they are the mass states, `ν_1`, `ν_2`, `ν_3`.",
         "~~Strong potential~~: they become flavor states; `ν_e` and `ν_μ` get effective masses squared "
         "of about `+σ` and `−σ`.",
         "A decay `a → b` is open only where the effective masses allow it: the parent heavier than the daughter, "
         "with room for the mediator.",
         "**The old code decayed each state with vacuum masses, which let the most stable state decay into "
         "itself. Now each state decays with its own effective mass.**"],
        w=11.6, size=15, space=0.28)
    f, _ = fig("levels", 13.6, 11.6, 2.6, x=W - 13.6 - 0.55)
    slide(lead="In a potential, the states that travel change", extras=[t, f],
          ref="`L_e − L_μ`, lightest mass 0.01 eV, NuFIT 6.1 · notes/notes.tex, “What is followed” and "
              "“Decay in a medium”",
          notes="Eigenvalues of U M^2 U^dagger + sigma S; antineutrinos have -sigma.")

    # ------------------------------------------------------------------ 7
    rows = [
        ["New effect", "What it is", "How much it matters"],
        ["**Decay inside a potential**",
         "In a potential, the states that travel are not mass states. Decays now go between these states, "
         "each with its own effective mass. The old treatment let a state decay into itself.",
         "Galactic flux at 100 TeV, `L_e − L_μ` at `M` = 10{{−16}} eV: 3% of the standard one before, 57% now"],
        ["**Extragalactic neutrinos crossing the Galaxy**",
         "Inside the Galaxy, the lightest neutrino and antineutrino states can decay into each other. Between "
         "galaxies they cannot. We now follow these decays.",
         "Lowers the flux at the top of the band by 13–41%, at `g/M` = 0.1 eV{{−1}}"],
        ["**The vector's longitudinal mode**",
         "With massive neutrinos, the vector also scatters as a scalar would, with couplings ∝ `m_ν g/M`",
         "Flux at 10 TeV × 0.95 (lightest mass 10{{−6}} eV) or × 0.03 (0.1 eV), at `g/M` = 0.1 eV{{−1}}"],
    ]
    parts, end = table(MARGIN, 2.75, [0, 5.6, 16.0], rows, size=15, row_h="auto", width=BODY_W, pad=0.3)
    t, _ = column(["Also: each step is now solved exactly per energy bin, which cut the time per point from "
                   "600–750 s to 4–8 s."], y=end + 0.3, size=15)
    slide(lead="Three new effects", extras=parts + [t],
          ref="notes/notes.tex, Sec. “Decay in a medium” · docs/dev/DECAY_IN_MEDIUM_2026-10-25.md",
          notes=("The first one was a physics error, not a numerical one: the old model let the most stable\n"
                 "state decay into itself and slid the flux down in energy."))

    # ------------------------------------------------------------------ 9
    rows = [
        ["Data set", "What it is", "Role"],
        ["**IceCube spectrum + flavor**",
         "IceCube's 2026 combined fit of the astrophysical spectrum, in 1/3-decade segments from 4.6 TeV to 1 PeV; "
         "its 2025 flavor-composition measurement with medium-energy starting events (11.4 yr)",
         "Main present-day result"],
        ["**IceCube events**",
         "HESE: 7.5 years of high-energy starting events. DR2: through-going muon tracks, IC79 + IC86, above 3 TeV. "
         "We fold our fluxes with IceCube's public detector responses.",
         "Present-day, event by event, in energy and direction"],
        ["**Future telescopes**",
         "IceCube, IceCube-Gen2, KM3NeT/ARCA, Baikal-GVD, P-ONE, TRIDENT, NEON, HUNT: 10 years each; tracks, "
         "showers, double cascades",
         "Projection, taking today's best fit as the truth"],
    ]
    parts, end = table(MARGIN, 2.75, [0, 5.3, 19.6], rows, size=15, row_h="auto", width=BODY_W, pad=0.3)
    t, _ = column(["Atmospheric backgrounds: MCEq, Barr et al. uncertainties. The event set drops the flavor "
                   "measurement, whose events overlap HESE's."], y=end + 0.3, size=15)
    slide(lead="The data we fit", extras=parts + [t],
          ref="IceCube 2026: arXiv:2607.25966 · analysis/spectra.py, config.py · detectors as in NuFlavorSingleSource",
          notes="The spectrum's segment correlations are not published; we take them as independent.")

    # ------------------------------------------------------------------ 10
    rows = [
        ["Free parameter", "Range or prior", "Fitted in"],
        ["**Mediator mass** `M` and **coupling**", "On a grid; each mass read on its own", "All fits"],
        ["**Astrophysical flux**: normalization", "0.01–1000 × IceCube's best fit; no prior", "All fits"],
        ["**Spectral indices** `p`, `p_\"low\"`, above and below the break at 25 TeV", "2–3 and 0.5–2.5; no prior",
         "All fits"],
        ["**Flavors at the sources**, extragalactic and Galactic",
         "Pion decay, (1/3, 2/3, 0); or (`x`, 1 − `x`, 0), `x` fitted", "Half the planes"],
        ["**Galactic emission**: normalization of the CRINGE template", "Gaussian prior, 0.91 ± 0.18", "All fits"],
        ["**Lightest neutrino mass** (normal ordering)", "10{{−6}}–0.1 eV; sum of masses < 0.12 eV (95%)",
         "All fits"],
        ["**Event fits only**: atmospheric flux (12 Barr parameters, cosmic-ray index, prompt flux); HESE: two "
         "normalizations, three detector systematics; DR2: one normalization per configuration",
         "Gaussian priors: the releases' for HESE, 10% for DR2", "Events"],
    ]
    parts, end = table(MARGIN, 2.45, [0, 13.4, 21.6], rows, size=13.5, row_h="auto", width=BODY_W, pad=0.14)
    t, _ = column(["**Fixed**: mixing angles and CP phase at the NuFIT 6.1 central values; sources follow the "
                   "star-formation rate up to `z` = 4; as many neutrinos as antineutrinos."],
                  y=end + 0.15, size=13.5)
    slide(lead="Free parameters", extras=parts + [t],
          ref="analysis/configs/campaign_ml · analysis/events.py · NuFIT 6.1 · CRINGE: Schwefer, Mertsch, "
              "Wiebusch (2023)",
          notes=("The mixing is held fixed in the likelihood; its effect is estimated outside it.\n"
                 "The lightest mass is profiled exactly (a scan of 11 values, then refined), only where the model "
                 "holds."))

    # ------------------------------------------------------------------ 10b
    t, _ = column(
        ["~~Frequentist~~: profile likelihood, `χ^2 = −2 \"ln\" L`.",
         "For each mediator mass, we scan the coupling; at each coupling, we refit every other parameter.",
         "~~Excluded at 95% CL~~: `Δχ^2` > 3.84 above the best fit (or above the no-new-physics fit, where lower).",
         "Small couplings change nothing: `Δχ^2` is flat. The limit is where it crosses 3.84.",
         "~~In the likelihood~~: the data, plus the priors of the previous slide as Gaussian penalties. No other "
         "corrections. 3.84 is the two-sided value (one-sided: 2.71, tighter).",
         "~~Planned~~: coverage tests with pseudo-experiments; Feldman–Cousins if the limits under-cover."],
        w=11.6, size=14.5, space=0.25)
    f, _ = fig("column", 13.6, 11.6, 2.6, x=W - 13.6 - 0.55)
    slide(lead="How a limit is read", extras=[t, f],
          ref="Right: `L_e − L_μ`, `M` = 10 keV, IceCube spectrum + flavor, pion-decay flavors (the plane's own grid)",
          notes=("Here the limit is g' = 5.7e-3 on the grid; with the refinements, 5.9e-3.\n"
                 "HESE's finite simulation statistics are not corrected for (PyFC's L_Eff would do it)."))

    # ------------------------------------------------------------------ 10c
    t, _ = column(
        ["A beam neutrino meets a relic neutrino of mass `m`, almost at rest. They make the mediator when "
         "`E = M^2/2m`.",
         "For relics of `m_2` ≈ 0.009 eV to 0.1 eV and IceCube's energies, 10 TeV – 30 PeV, this needs "
         "`M` ≈ 0.4–80 MeV.",
         "There the absorption carves a dip in the spectrum, and the products pile up below it.",
         "The data's best fit, `M` ≈ 3.2 MeV, puts the resonance at ~100 TeV on `m_3` and ~0.6 PeV on `m_2`."],
        w=11.6, size=16)
    f, _ = fig("resonance", 13.6, 11.6, 2.6, x=W - 13.6 - 0.55)
    slide(lead="Why MeV mediators: where the resonance falls", extras=[t, f],
          ref="Normal ordering, lightest mass ≈ 0 (NuFIT 6.1 splittings), or all masses 0.1 eV · notes, “Thermal absorption rates”",
          notes=("The lightest relic, nearly massless, is not at rest: its resonance is smeared by the relics'\n"
                 "thermal momenta (included in the rates through the thermal average)."))

    # ------------------------------------------------------------------ 11
    t, h = column(
        ["Excluded: narrow windows near 0.5–5 MeV, where the resonance is in IceCube's band, down to couplings "
         "of 1.2–3.8 × 10{{−4}}. IceCube's spectrum mildly prefers a resonance at `M` ≈ 3 MeV "
         "(`Δχ^2` = 5.7–7.0 against no new physics)."],
        size=15)
    f, _ = fig("mev", 25.4, 9.45, 4.9)
    slide(lead="MeV mediators: what IceCube excludes today", extras=[t, f],
          ref="Pion-decay flavors at the sources · unchanged by the October code",
          notes=("Below ~300 keV only couplings above 6-9e-3 are excluded.\nFinal for IceCube's spectrum + flavor. HESE + DR2 fits are to be re-evaluated with a warm-started\n"
                 "profile before they are final."))

    # ------------------------------------------------------------------ 12
    rows = [
        ["Mediator mass", "Excluded `g_D |U_{μ4}|^2`"],
        ["≤ 10{{−10}} eV", "Outside the model's validity"],
        ["10{{−8}}, 10{{−6}} eV", "Every coupling ≥ 10{{−8}}"],
        ["10{{−4}} eV", "≥ 1.8 × 10{{−8}}"],
        ["10{{−2}} eV", "≥ 1.8 × 10{{−5}}"],
        ["≥ 1 eV", "≥ 10{{−2}}"],
    ]
    t0, h0 = column(["Dark portal, `ν_μ`, from IceCube's spectrum + flavor:"], w=12.6, size=15.5)
    parts, end = table(MARGIN, 2.6 + h0 + 0.1, [0, 4.6], rows, size=15, row_h=0.7, width=12.6)
    t, _ = column(
        ["At light masses, every coupling in the plane makes the heavier neutrinos decay: the beam arrives as "
         "`ν_1`, mostly `ν_e`. IceCube's flavors disfavor that (right): `Δχ^2` = 10–11.",
         "`L_e − L_μ`, a first test at `M` = 10{{−12}} eV: `Δχ^2` = 2.3 at `g/M` = 0.03 eV{{−1}}, 31 at "
         "0.3 eV{{−1}}."],
        y=end + 0.3, w=12.6, size=15.5)
    f, _ = fig("triangle", 12.6, 11.2, 2.75, x=W - 12.6 - 0.45)
    slide(lead="Ultralight mediators: first results (preliminary)", extras=[t0] + parts + [t, f],
          ref="Preliminary: one plane of the final run, limits not yet refined · flavors: IceCube's medium-energy "
              "starting events (2025)",
          notes=("Preliminary: one plane merged, refinements not done yet.\n"
                 "Triangle: pure nu_1 at Earth is (0.68, 0.11, 0.21); the MESE 2025 surface alone gives 11.5."))

    # ------------------------------------------------------------------ 14
    rows = [
        ["Step", "Status"],
        ["Dark portal + lepton number, MeV (30 planes)", "%%Done%% · HESE + DR2 fits to be re-evaluated"],
        ["Dark portal + lepton number, below 10 keV (30 planes)",
         "!!Running!!: about half done at the last check (26 Oct); the cluster is offline now"],
        ["Refine the limits, final plots", "Next, when the cluster is back"],
        ["Checks: mediator decaying back into neutrinos; relic population after decays; "
         "every vector point with `g/M` ≥ 10{{−2}} eV{{−1}} excluded", "To do"],
        ["Coverage tests with pseudo-experiments", "Planned"],
        ["Secret interactions (192 planes, 10{{−30}} eV – 1 GeV)", "Configured; runs after the ultralight run"],
        ["One discovery case; then the paper", "Last"],
    ]
    parts, end = table(MARGIN, 2.75, [0, 14.2], rows, size=16, row_h="auto", width=BODY_W)
    slide(lead="Status and next steps", extras=parts,
          ref="Fits take 19–47 min per point with every process on (test of 2026-10-25)",
          notes="One final run per family: every physics choice is settled before it starts.")

    # ------------------------------------------------------------------ 15
    slide(
        lead="Summary",
        body=["~~One transport equation~~ for everything a light mediator does to cosmic neutrinos: absorption, "
              "scattering, energy loss, decay, potentials. From the sources, through the Universe, the Galaxy, "
              "the Sun and the Earth, to the detector.",
              "~~Three families of models~~, mediator masses 10{{−30}} eV – 1 GeV, three data sets: today's "
              "IceCube spectrum + flavor, today's IceCube events, eight future telescopes.",
              "~~MeV mediators~~: limits from IceCube's spectrum + flavor are done.",
              "~~Ultralight mediators~~: the final run is more than half done. Where neutrinos decay, IceCube's "
              "flavors exclude even tiny couplings.",
              "~~Next~~: finish the run, final plots, secret interactions, one discovery case, the paper."],
        body_size=BODY)

    # ------------------------------------------------------------------ backup
    section(["Backup"])

    # ------------------------------------------------------------------ B1
    slide(
        lead="Where we trust the models, and what we shade",
        body=["~~Vector mediators below 1 keV~~: the vector's longitudinal mode couples as `m_ν g/M`. At large "
              "`g/M` the theory breaks down (perturbative unitarity fails), and the heavier the neutrinos, the "
              "sooner. So at each point we fit only the lightest-neutrino masses where the theory holds. Where "
              "none does, we shade the point as not allowed.",
              "~~Scalar secret interactions below 1 keV~~: the relic neutrinos themselves create a scalar field "
              "that shifts every neutrino mass. We do not model this. It matters above `g/M` ≈ 30 to "
              "10{{4}} eV{{−1}}, depending on the flavor and the lightest mass; we shade that region.",
              "~~Decays inside a potential~~: our decay rates assume each decay is slow against the oscillations "
              "between the states that travel. Every run records how well this holds. The correction is at most "
              "1.5% at our test points; it reaches 10–50% only for `g/M` ≈ 10–30 eV{{−1}}, near where unitarity "
              "fails."],
        body_size=BODY,
        ref="docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md, items 22–24 · docs/dev/REVIEW_2026-10-25.md · notes/notes.tex",
        notes="The vector's relic background cancels between neutrinos and antineutrinos; the scalar's does not.")

    # ------------------------------------------------------------------ B2
    slide(
        lead="Validation",
        body=["~~Steps in redshift~~: where the potential sets the states, 200 steps converge to ≤ 1.3 × 10{{−4}} "
              "in spectral shape; where decays and scatterings cycle, 400 steps to ≤ 1.7 × 10{{−3}}.",
              "~~Coarse sky for the Galactic paths~~ (`N_\"side\"` = 4 vs 8): the crossing flux moves by 3 × 10{{−4}}, "
              "the Galactic emission by 5.5% (its prior is 18%), the `χ^2` by 0.1–0.3 at `Δχ^2` ≈ 30.",
              "~~Conservation~~: each decay keeps one neutrino. Without a potential, the vacuum decay rates come "
              "back exactly.",
              "~~Nothing else moved~~: the MeV and secret-interaction planes above 1 keV are unchanged by the "
              "October code (952 test units, identical to 10{{−14}}).",
              "~~Review~~: an independent review of all October changes; the full test suite passes on the "
              "cluster and in continuous integration."],
        body_size=BODY,
        ref="docs/dev/DECAY_IN_MEDIUM_2026-10-25.md · docs/dev/REVIEW_2026-10-25.md · notes/notes.tex",
        notes="Shape = ratio to a 3184-step reference with its median divided out, over 10 TeV - 10 PeV.")

    # ------------------------------------------------------------------ B3
    t1, h1 = column(["~~Sources~~: star-formation rate `R(z)` up to `z` = 4, spectrum `E^{−p}`, flavors `f_α` "
                     "split over the states at the source:"], size=16)
    f1, i1 = fig("eq_src", 22.0, 1.6, 2.6 + h1)
    y = 2.6 + h1 + i1 + 0.1
    t2, h2 = column(["~~The states~~ followed are the eigenstates of", ], y=y, size=16)
    f2, i2 = fig("eq_ham", 22.0, 1.6, y + h2)
    y += h2 + i2 + 0.1
    t3, _ = column(["~~Absorption~~: a Breit–Wigner on each relic mass state, averaged over the relics' thermal "
                    "momenta; each absorption re-emits a neutrino and an antineutrino.",
                    "~~Soft scatterings~~: the conversion rate between states is damped where they decay fast "
                    "(the decay widths enter the damping).",
                    "~~Decay in a medium~~, `a → b`: allowed where `(1 − x)[x μ_a(σ) − μ′_b(xσ)] − x M^2` ≥ 0, "
                    "with `μ` the effective masses squared and `x = E/E′`."], y=y, size=16)
    slide(lead="Backup: the terms in more detail", extras=[t1, f1, t2, f2, t3],
          ref="notes/notes.tex, Eqs. (sources), (transport_cosmo), (medium_window)")


if __name__ == "__main__":
    build(deck)
