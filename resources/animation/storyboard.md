# Promotional animation: plan of Concept A, "One trip, four stops"

Status (2026-11-27): **Concept A chosen** (Mauricio, from three concepts: A, one trip with four stops, about 90 s;
B, three trips, one per family, about 2:30; C, the flux changing, without a trip, about 60 s). This storyboard
awaits his review; no stills or renders yet.

Line numbers refer to `paper/main.tex` in the main checkout (branch `dev`, 2026-11-27). Numbers from Sec. V are still
being finalized by the cluster runs: each is marked **(re-verify)** and must be re-read before any render.

## 1. Message, length, format

- **Message:** on their way from distant sources to IceCube, high-energy neutrinos can feel a new light mediator at
  every stop of the trip, so we follow them the whole way to test it (l. 85, 135).
- **Length:** about 1:55 (since 2026-11-28, with the telescopes' globe), 12 scenes (scenes 3-7 are one continuous trip). Over the 90 s pitched, to leave pauses at
  the key events ("the viewer needs to have time to digest what's on screen").
- **Format:** 1920 × 1080, 30 fps, MP4 (H.264), silent; captions only. Matplotlib, Palatino through the repository's
  `matplotlibrc` (`analysis/figure_style.use()`).
- **Background:** dark (`#0b0f1a`, as GeoneutrinoDecay's and NuFlavorGalacticPlane's videos), so that the trip reads
  as space and matches the paper's own rendition of it (Fig. 2). NuLRIEarth's was white. **Decision 1.**
- **Layout during the trip:** the scene on the left 62%; one panel on the right; one line of caption at the bottom.

## 2. The examples

The paper's three examples, "the same in every illustrative figure" (l. 398):
- **Secret interactions:** a scalar coupled to $\nu_\tau$, Majorana neutrinos, $M = 10$ MeV, $g = 0.18$.
- **Dark portal:** coupled to $\nu_\tau$, $M = 10^{-3}$ eV, $g_D |U_{\tau 4}|^2 = 5.6 \times 10^{-7}$.
- **$L_e - L_\mu$:** $M = 10^{-20}$ eV, $g' = 10^{-21}$.

The trip follows the flux of Fig. 4 (l. 454): sources in a shell at $z = 1.8$-$2.2$, neutrinos that reach the Earth
with 495 TeV. Every family is shown on the same trip, **one mediator at a time**; the panel says so, so that the video
does not suggest that the three act together.

## 3. How each physical ingredient is shown

**The panel: one bar per mediator.** Three horizontal bars, one per example, labeled by family. Each bar's length is
the flux at 495 TeV relative to the flux leaving the sources; its three segments are the shares of $\nu_e$, $\nu_\mu$,
$\nu_\tau$ (neutrinos plus antineutrinos), in Fig. 4's colors ($\nu_e$ `#d95f02`, $\nu_\mu$ `#7570b3`, $\nu_\tau$
`#1b9e77`, brightened if the dark background needs it). The bars follow the camera: at each frame they show the
stored flux at the neutrinos' position. A row lights up while its process acts. Exact: from the arrays of Fig. 4,
`output/analysis/trajectory_figure_fig7.npz` (`analysis/trajectory_figure.py`, rows `si`, `dp`, `le_20`). What the
arrays show, checked 2026-11-27:
- Secret interactions: the flux falls to zero near $z = 1$, where $495\,{\rm TeV} \times (1+z)$ crosses the resonance
  on the heaviest relic, 1.0 PeV (l. 237); secondaries refill a few percent by today.
- Dark portal: the flux drains steadily, to about a third, as $\nu_3$ and $\nu_2$ decay; the shares move toward
  $\nu_1$'s flavor content (l. 638).
- $L_e - L_\mu$: the decays have already acted within about a megaparsec of the sources (l. 405), so the bar starts in
  $\nu_1$'s flavor content; between about 15 and 2 kpc from the Sun, the Milky Way's potential turns it into $\nu_\mu$
  and $\bar\nu_e$ (no $\nu_\tau$ left) and removes about a tenth of it (l. 457).
- Near the Sun and inside the Earth, no bar changes (l. 464 for the Earth; the arrays for the Sun).

**The potential of matter** (redesigned 2026-11-28; Mauricio: "I am not satisfied with how the potentials for
Le-Lmu are depicted. In fact, they aren't", then chose "Tint the matter" from three options: tint, a strip chart along
the bottom, a meter in the panel). The $L_e - L_\mu$ example's potential, $|V| = g'^2 \Phi_e$ (Eq. 2), relative to
$\Delta m^2_{31}/(2E(1+z))$ at 495 TeV, is a tint over each world: its color gives log10 of the ratio (violet to pale
pink over $10^0$-$10^{16}$); it is transparent where the ratio is below 1 and fully opaque (0.42) from $10^5$ (until 2026-11-28 it started at
0.1, and its canvas was smaller than the region it covers: the Milky Way showed inside a faint rectangle). The
neutrino glows in the color of the value where it is; under the $L_e - L_\mu$ bar of the panel, the same scale ("Its
potential at the neutrino, over $\Delta m^2_{31}/2E$", $10^{-4}$ to $10^{16}$) carries a marker at that value. Exact,
from the paper's density models (`video_inputs.potential`): the range of the force is about 130 AU here, so in the
Milky Way $\Phi_e = n_e/M^2$ with `dp_spatial.calc_MW_electron_density` (axisymmetric: one radial profile in the
disk's plane, each screen point taken back to the plane through the drawing's tilt); near the Sun its electrons'
$e^{-Mr}/(4\pi r)$ (`dp_core.N_e_sun`); near the Earth the 1/r potential of PREM's electrons (integrated:
$1.7482\times10^{51}$, against the code's $N_e = 1.7480\times10^{51}$) on top of the Sun's and the Moon's; in the
Universe the cosmic mean, $n_{e,0}(1+z)^3$ (`dp_config`). The values, checked 2026-11-28: Universe $10^{-5.2}$-$10^{-3.4}$
(no tint); the Milky Way's plane $10^{5}$ at its center, $10^{2.1}$ at the Sun's radius, about 10 at 20 kpc (so the
potential dominates all along the Galaxy, where Fig. 4's flavor changes); near the Sun $10^{13.6}$ at 1 AU; at the
Earth $10^{13.64}$, the Sun's, with the Earth's own adding 7%. So the Sun's and the Earth's tints fill their views and
are drawn behind the Sun and the Earth, which keep their colors. The Milky Way's line of sight is schematic (it passes
2 kpc from the center); the bars come from Fig. 4's own line. Only $L_e - L_\mu$ feels the potential (l. 142).

**Resonant absorption on relic neutrinos, and the dip.** The relic neutrinos fill the Universe scene as faint teal
dots (Fig. 2). Where the secret-interaction bar collapses, one of them meets the beam: a flash, the mediator (a short
wavy line), and its decay into two neutrinos of lower energy (orange, as Fig. 2's secondaries; l. 388). The dip
appears at the detector, in the spectrum (scene 8), near 1.0 PeV, the resonance on the heaviest relic (l. 237). The
event is schematic; its redshift and the bar are exact.

*Checked 2026-11-27, and the reason there is no resonance gauge in the trip:* at 495 TeV, the arrays show the
secret-interaction flux falling from 1 to 0.15 between $z = 1.84$ and 1.70, and to 0.002 by $z = 1.2$, before the
energy at the time, $495\,{\rm TeV} \times (1+z)$, reaches the 1.0 PeV resonance at $z \approx 1.0$. At these
energies the collisions act within a Hubble length even off the peak, "most strongly at the resonances" (l. 407). A
gauge that fired the absorption at the resonance would misplace it.

**The decays and what they return.** On the beam, a heavier neutrino turns into a lighter one and emits the mediator
(a pink wavy line, as Fig. 2); the daughter continues with less energy (l. 392, 399). Schematic; the bars are exact.

**The flavor composition and the spectrum at the detector.** Two panels, redrawn in the video's style from Fig. 7's
arrays (`output/analysis/three_classes_figure.npz`, the full calculation of each example; `analysis/three_classes_figure.py`):
a flavor triangle (his ternary style: grid every 0.1, "Fraction of $\nu_e$, $f_{e,\oplus}$") with IceCube's
measurement at 68% and 95% C.L. (`analysis.surfaces.chi2`, "icecube_mese_2025_grid") and one point per example plus
no new interaction (`analysis.mese.event_weighted_fractions`, 5 TeV-10 PeV); and the flux of all flavors at Earth
relative to no new interaction, versus energy. Exact.

**The limits.** Redrawn from the limits files of the paper's figures (`output/analysis/limits_<date>/`, read with
`analysis/exclusion.py` as `notebooks/paper_figures.ipynb` does). Stamped PRELIMINARY until the final files are given.
**Decision 2** (what the last scene shows).

## 4. The trip: scales and camera

As Fig. 2's panels, each scale opens from a box in the previous one (a 2-s zoom through the box, its subtitle
updating): the Universe, "Gigaparsecs"; the Milky Way, "Kiloparsecs"; near the Sun, "1 AU"; inside the Earth,
"6371 km". The beam enters each scale where it left the previous one and keeps its direction. The camera's speed is
not to scale; redshift and distances drive the bars, and a small readout gives them ("Redshift $z$", "Distance to the
Sun [kpc]", "Distance to the Earth [AU]", as Fig. 4's axes). Coastlines: Natural Earth 1:50m, copied from
NuLRIEarth's `resources/naturalearth/land_50m.json` (no download); the Earth does not rotate while IceCube is shown.

## 5. Shots

Every line on screen is checked against the paper (main.tex line) or the code named. One caption line per beat;
labels in the scene carry the rest.

| # | Time | Visual | Text on screen | Source |
|---|---|---|---|---|
| 1. Title | 0-5 s | Dark sky with distant galaxies; the title card, dimmed out at the end. | Title: "New interactions of high-energy cosmic neutrinos, from the Hubble scale to Earth". (arXiv number once known.) | l. 67 |
| 2. Question | 5-12 s | The sky; three small labels appear in turn, one per family. | (a) "Do neutrinos feel new interactions, carried by light particles?" (b) "We test three kinds of mediator, one at a time." Labels: "Coupled to electrons, muons and their neutrinos: $L_e - L_\mu$"; "Coupled through a dark neutrino: the dark portal"; "Coupled to neutrinos alone: secret interactions". | (a) l. 103; (b) l. 104, 163-165, 175 |
| 3. Source | 12-19 s | The camera closes on a distant galaxy; a neutrino beam leaves it (Fig. 2's glowing wavy line). Label at the source: "$\nu_e : \nu_\mu : \nu_\tau = 1 : 2 : 0$". Scale subtitle "Gigaparsecs". The panel slides in at 17 s, its bars full. | "High-energy neutrinos leave a distant source, around redshift 2." Panel title: "Neutrinos detected at about 500 TeV, one mediator at a time" (Fig. 4's 495 TeV, the solver's grid point nearest 500 TeV; rounded on screen on Mauricio's question, 2026-11-28). | l. 124, 298, 454 |
| 4a. Relics | 19-24 s | The beam crosses the Universe; relic neutrinos as faint dots; readout $z$ = 1.80 → 1.77. Label: "Relic neutrinos". | "On the way, they cross the relic neutrinos left over from the early Universe." | l. 126 |
| 4b. Absorption | 24-33 s | A relic meets the beam: flash (pause about 1.5 s), mediator, two orange neutrinos of lower energy. The secret-interaction row lights up; its bar collapses. $z$ = 1.77 → 1.40. | (a) "With secret interactions, a relic neutrino can absorb a high-energy one." (b) "The mediator then decays into two neutrinos of lower energy." | l. 236-238, 388, 456; arrays |
| 4c. Decays | 33-40 s | A decay on the beam: a neutrino emits the pink mediator, the daughter continues. The dark-portal and $L_e - L_\mu$ rows light up; the dark-portal bar drains. Label on the $L_e - L_\mu$ row: "Decayed near the source". $z$ = 1.40 → 0.70. | "With the other two, heavier neutrinos decay into lighter ones." | l. 240, 392, 405, 457 |
| 4d. Today | 40-45 s | The beam reaches the Milky Way's box; $z$ → 0. | — | — |
| 5. Milky Way | 45-59 s | Zoom through the box into the Milky Way (Fig. 2's spiral); the beam crosses to the Sun; the glow of stars and gas, labeled "Potential of its stars and gas". The $L_e - L_\mu$ row lights up; its shares change between 15 and 2 kpc (readout in kpc). | (a) "Of the three, only $L_e - L_\mu$ feels the Milky Way's stars and gas." (b) "Their potential changes the flavor of the neutrinos." | l. 142, 431, 457; arrays |
| 6. Sun, Moon, Earth | 59-67 s | Zoom to the Sun ("1 AU"), the Earth and the Moon; then the Earth's cut (layers, real coastlines on the far half, as Fig. 2). Bars unchanged. | "The Sun, the Moon and the Earth add their own potentials." | l. 140, 438 |
| 7. IceCube | 67-75 s | Zoom to the South Pole: ice, strings of light sensors; the neutrino arrives; a schematic track lights up. Label "Track" (his "Remove stylized", 2026-11-28). | "IceCube, at the South Pole, measures their flavors and energies." | l. 125, 141 |
| 8a. Flavor | 75-82 s | The flavor triangle alone, full size (his "too crowded", 2026-11-28): faint grid every 0.1, IceCube's region in blue with its key outside the triangle, the points appearing one by one (no new interaction and secret interactions share one point and one leader). | "The flavor composition disfavors $L_e - L_\mu$ and the dark portal." | l. 643; Fig. 7 arrays |
| 8b. Spectrum | 82-89 s | The flux at Earth versus energy, relative to no new interaction, alone, full size; the curves draw in; the two secret-interaction dips labeled "Dips from absorption" (1.0 and 5.8 PeV, l. 237). | "The spectrum disfavors secret interactions." | l. 643; Fig. 7 arrays |
| 9a. Limits | 89-95 s | Three panels (decision 2): $L_e - L_\mu$, the dark portal on $\nu_\tau$, the scalar of secret interactions on $\nu_\tau$; existing bounds in gray; where no claim is made, hatched; today's region (IceCube) fills in; the earlier estimate dashed in the $L_e - L_\mu$ panel, "Earlier estimate: potential at the detector only". The dark portal's panel keeps the paper's "Smallest coupling computed" line. PRELIMINARY until the final files. | "Following the whole trip moves the limits by many orders of magnitude." | abstract (search "The full transport moves the limits"); dashed line: Fig. 8 caption |
| 9b. Telescopes | 95-103 s | Added 2026-11-28 (Mauricio: "we also should include a later scene with all detectors, like we have done for NuFlavorGalacticPlane"). A still globe (Natural Earth 1:50m, no rotation), the sites appearing in turn, the running ones first: "IceCube", "KM3NeT/ARCA", "Baikal-GVD" (Running, light blue); "P-ONE (being built, far side)" (hollow, on the limb), "TRIDENT, NEON, HUNT" (one site, as in the detector models), "IceCube-Gen2" (Being built or planned, darker blue); a legend of the two classes. The classes, colors and labels are NuFlavorGalacticPlane's video's (Mauricio, 2026-11-28: "Add the same classification ... as in DMGalacticPlaneNu"), from that paper's main.tex ("now running or being built (IceCube, KM3NeT, Baikal-GVD, P-ONE) and planned (TRIDENT, NEON, HUNT, IceCube-Gen2)"); this paper's main.tex gives no status per telescope. Sites: that analysis's code/schematic.py SITES. | "A forecast for eight neutrino telescopes, each taking data for ten years." | l. 491, 799 (Table I lists the eight, l. 499) |
| 9c. Projected | 103-109 s | The three panels again; the eight telescopes' regions draw in under today's. | "Together, they could tighten the limits by factors of about 2 to 60." | abstract, l. 98; Sec. V **(re-verify)** |
| 10. Card | 109-115 s | Title, authors, arXiv number once known. | Mauricio Bustamante, Mariia Petropavlova, Sudipta Das, Sanjib Kumar Agarwalla | l. 67-77; Mariia Petropavlova second on Mauricio's word (2026-11-27), not yet in main.tex |

Kept off the screen: the claims of leading limits (l. 91-92: they carry caveats about the CMB and BBN that a caption
cannot hold), the MeV preference (l. 699-707), any number from Sec. V other than (9b).

## 6. What is computed, and what is schematic

| On screen | How | Source |
|---|---|---|
| The bars (flux and flavor shares along the trip, per example) | exact, read | `output/analysis/trajectory_figure_fig7.npz` |
| Redshift of the absorption; the resonance energy | exact | the same arrays; l. 237 |
| Flavor triangle, IceCube's region, flux versus energy at Earth | exact, read and weighted as Fig. 7 | `three_classes_figure.npz`; `analysis.mese`, `analysis.surfaces` |
| Limits | exact, read | limits files, `analysis/exclusion.py` |
| Coastlines | real | Natural Earth 1:50m (copied) |
| The $L_e - L_\mu$ potential (tint, the neutrino's glow, the scale's marker) | exact, computed from the paper's density models | `video_inputs.potential`; `dp_spatial`, `dp_core`, `dp_config` |
| Galaxies, the Milky Way, the Sun, the Earth's layers | artist's rendition | `analysis/trip_illustration.py`'s drawing functions, imported read-only |
| Beam, relic dots, absorption, decays, potentials' glow, IceCube's track | schematic | drawn here |
| Speed of the camera, sizes, numbers of relics | not to scale | — |

Nothing runs the solver or the fits. `video_inputs.py` reads the arrays once (one process, `ulimit -v 2500000`) and
writes `data/inputs.npz`, which the frames read.

## 7. Files in `resources/animation/`

- `storyboard.md`: this plan, its decisions, and the source of every line on screen.
- `README.md`: status, how to regenerate, decision log.
- `video_inputs.py`: reads the repository's arrays and limits; writes `data/inputs.npz`.
- `make_video.py`: draws every frame as a function of time, pipes it to ffmpeg; `verify()` checks the title and every
  number on screen against `main.tex` before drawing; `--stills`, `--preview`, `-j`.
- `build/`: stills, previews, the final video.

## 8. Decisions for Mauricio

1. Background: dark, as Fig. 2 and the last two videos (recommended), or white, as NuLRIEarth. **Decided
   2026-11-27: dark.**
2. Last scene: the $L_e - L_\mu$ limit against the earlier estimate, then the future telescopes; or three small
   panels, one limit per family. **Decided 2026-11-27: three panels**, one per family, for the examples' couplings:
   $L_e - L_\mu$ (with the earlier estimate dashed, which caption (9a) needs), the dark portal on $\nu_\tau$, the
   Majorana scalar on $\nu_\tau$; today's region shaded, the eight future telescopes as a line; flavor composition at
   the sources fitted, as the paper's limit figures.

Defaults unless he objects: the paper's three examples; Fig. 4's flavor colors; about 1:55.
