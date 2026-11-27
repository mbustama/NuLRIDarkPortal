# Promotional animation

A captions-only animation of the paper "New interactions of high-energy cosmic neutrinos, from the Hubble scale to
Earth": one trip of high-energy neutrinos from a distant source to IceCube, with what each family of mediators does at
each stop, what IceCube would see, and the limits. 16:9, 1920 × 1080, 30 fps, for talks, the group's web page and
sharing online. Modeled on NuLRIEarth's `resources/animation/` and on the videos of GeoneutrinoDecay and
NuFlavorGalacticPlane.

**Status (2026-11-27, 23:58):** Concept A chosen; storyboard (`storyboard.md`) written; dark background and three
limit panels decided; stills reviewed by him (he added Mariia Petropavlova to the end card). Full video rendered
on Tycho on his "Render the full-resolution video already" (2026-11-28), job 909792 (16 CPUs, 48 GB requested, 2 min
47 s, peak 26.1 GB): `build/video.mp4`, 1:55, 1920 × 1080, 30 fps, H.264, 17.1 MB (md5 c9b9b3f8...), copied to the
laptop; inputs refreshed the same night (main checkout 6034c6d, limits_20261127). The limit panels carry the
PRELIMINARY stamp until he confirms the final limits files; the card has no arXiv number yet. Nothing committed.

The animation reads the main checkout's `paper/main.tex` and `output/analysis/` (read-only); it runs neither the
solver nor the fits.

## Files

- `video_inputs.py`: reads the paper's arrays (the species along the trip, Fig. 4; the flux at Earth, Fig. 7), IceCube's
  MESE flavor likelihood and the limits of the paper's figures, and writes `data/inputs.npz` (2.9 MB, 5 s, one
  process, 0.34 GB).
- `make_video.py`: draws every frame as a function of time (scene lengths in `SPEC`, captions in `CAPTIONS`);
  `verify()` checks the title and every line on screen against `main.tex` before drawing. `--stills universe:7`
  (scene:seconds into it, or seconds), `--storyboard` (one still per beat and `build/storyboard.png`). The raster art
  is built once into `build/cache/` (about 10 s per world). A 1920 × 1080 still takes about 15 s and 0.8 GB, the
  first one of a session longer (LaTeX).
- `data/land_50m.json`: Natural Earth's 1:50m land (public domain), copied on 2026-11-27 from
  `~/Research/NuLRIEarth/resources/naturalearth/land_50m.json` (SHA-256 18140153...e68829, as GeoneutrinoDecay's
  copy). Nothing was downloaded.
- `tycho_render.sh`: the render on Tycho (`sbatch tycho_render.sh preview|video`, 16 CPUs on astro3_short), from
  `/lustre/astro/mbustama/dp_video/anim`, a mirror of this folder's `make_video.py`, `data/` and `build/cache/`;
  `../repo` mirrors the repository files the script reads (`analysis/__init__.py`, `figure_style.py`, `matplotlibrc`,
  `trip_illustration.py`; `paper/main.tex`), found through `NULRI_REPO`. Python and ffmpeg come from
  NuFlavorGalacticPlane's render venv (`/lustre/astro/mbustama/gp_video/venv`), used as it is; nothing was installed.
  The job draws two stills first, then the video in 16 chunks joined without re-encoding (`-j`). `make_video.py`
  refuses to draw where LaTeX is missing (the text would not be Palatino).
- `build/`: stills, cache, renders (not tracked).

Run from this folder, one process at a time on the laptop, under `ulimit -v 2500000`:

```bash
python video_inputs.py
```

```bash
python make_video.py --storyboard
```

## Decision log

- 2026-11-27 (Mauricio): Concept A, "One trip, four stops", from three concepts (A: one trip, about 90 s; B: three
  trips, one per family, about 2:30; C: the flux changing, without a trip, about 60 s).
- 2026-11-27 (Mauricio): dark background (as Fig. 2 and the last two videos); the last scene shows three limit
  panels, one per family.
- 2026-11-27 (Mauricio): "Render preview on Tycho" (the brief had kept this work off the cluster; his instruction
  overrides it for the render). Its own folder and job (908841, astro3_short, 16 CPUs); the paper's runs, their clones
  and their queue settings untouched.
- 2026-11-27 (mine, from the first preview, job 908841: 73 s on 16 CPUs, peak 12 GB): the zooms between worlds
  showed the next world's art as a dark rectangle, went through blurry enlargements, and the beam jumped sideways at
  each one. Now each world draws the next one's object (galaxy, Sun, Earth) as a transparent sprite mapped exactly
  through its box, so a zoom lands on the same image and only the star fields cross-fade; the beam is one straight
  line through the four worlds, passing the Earth perpendicular to the Sun's direction (as Fig. 4's path past the
  Sun); every zoom keeps its target at a fixed point on screen; the IceCube roll turns about the detector, after the
  zoom; the Moon sits off the beam's side.
- 2026-11-28 (Mauricio): "Add the same classification (existing, future, etc., or whatever we used) as in
  DMGalacticPlaneNu for the other detectors": the globe now shows NuFlavorGalacticPlane's two classes, Running
  (IceCube, KM3NeT/ARCA, Baikal-GVD) and Being built or planned (P-ONE, TRIDENT, NEON, HUNT, IceCube-Gen2), its
  colors and legend, from that paper's main.tex (this paper's gives no status per telescope).
- 2026-11-28 (Mauricio): "495 TeV is a very oddly specific number. Why not 500 TeV?" Fig. 4's energy is the solver's
  grid point nearest 500 TeV (`trajectory_figure.E_OBS = 5e14`); the panel now reads "about 500 TeV" (main.tex's
  caption still says 495: his call). "The MW appears to be inside a lightly colored dim purple rectangle": the tint's
  canvas was smaller than the region where the potential exceeds the oscillation term (out to about 35 kpc), and the
  tint began at a tenth of it; now a larger canvas and no tint below 1. "Remove 'stylized'": the label reads "Track".
  "We also should include a later scene with all detectors, like we have done for NuFlavorGalacticPlane": a still
  globe with the eight telescopes between today's limits and the projected ones (storyboard 9b), names only (main.tex
  gives no status per telescope); 1:55 in all.
- 2026-11-28 (Mauricio): "The flavor triangle looks too crowded": scene 8 split in two beats of 7 s, the triangle
  alone (faint grid, IceCube's key outside it, one label for the shared point), then the spectrum alone; the captions
  now say which measurement disfavors which (l. 643).
- 2026-11-28 (Mauricio): "I am not satisfied with how the potentials for Le-Lmu are depicted. In fact, they aren't";
  he chose "Tint the matter" from three options. The potential relative to Delta m^2_31/2E, computed from the paper's
  density models (`video_inputs.potential`), tints each world; the neutrino glows with its value; a scale with a
  marker sits under the L_e - L_mu bar. Near the Sun and the Earth it fills the view (10^13.6, the Sun's), so it is
  drawn behind them (storyboard Sec. 3).
- 2026-11-27 (Mauricio): "Add Mariia Petropavlova as second author" (end card). main.tex does not list her yet, so
  `verify()` does not check her name; add it to `CHECKS` once the paper does.
- 2026-11-27 (mine, from the arrays): no resonance gauge in the trip. At 495 TeV, the secret-interaction flux
  collapses between $z = 1.84$ and 1.5, before the energy at the time reaches the 1.0 PeV resonance ($z \approx 1$);
  the flash is placed where the bar collapses, and the resonances appear as the two dips of the spectrum at Earth.
- Standing, from his earlier animations: the principle, not the paper's figures; captions only, no narration or
  music; one short line per beat, labels in the scene carry the rest; pauses at key events; no rotating Earth while a
  detector is shown; real, sharp coastlines; his matplotlib style.

## Open

- Stills for his review, then a preview; the full render only on his go.
- Before any render: re-read every number marked "re-verify" in the storyboard against `main.tex`; final limits files.
