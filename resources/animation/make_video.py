"""
make_video.py -- the promotional animation of "New interactions of high-energy cosmic neutrinos, from the Hubble
scale to Earth" (storyboard.md): one trip of high-energy neutrinos from a distant source to IceCube, what each family
of mediators does on the way, what IceCube would see, and the limits.

  python make_video.py --stills 3:2,4:7            # scene:seconds into it -> build/stills/
  python make_video.py --storyboard                # one still per beat and build/storyboard.png
  python make_video.py --preview                   # 960x540, 15 fps -> build/preview.mp4
  python make_video.py                             # 1920x1080, 30 fps -> build/video.mp4

Every frame is a function of time. The trip runs through four worlds, each zoomed into from a box in the previous
one, as the panels of the paper's Fig. 2: the Universe (arbitrary units, about gigaparsecs), the Milky Way (kpc),
near the Sun (AU), the Earth (km). Raster art (stars, galaxies, the Sun, the Earth) is drawn once with the drawing
helpers of analysis/trip_illustration.py and cached in build/cache/; the rest is vector. What comes from the paper's
results is read from data/inputs.npz (video_inputs.py). verify() checks the title and every number on screen against
paper/main.tex before drawing.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
CACHE = os.path.join(BUILD, "cache")
INPUTS = os.path.join(HERE, "data", "inputs.npz")
LAND50 = os.path.join(HERE, "data", "land_50m.json")


def main_checkout():
    if os.environ.get("NULRI_REPO"):
        return os.environ["NULRI_REPO"]
    common = subprocess.run(["git", "-C", HERE, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return os.path.dirname(common)


REPO = main_checkout()
sys.path.insert(0, REPO)
from analysis import figure_style            # noqa: E402  (his matplotlibrc)
from analysis import trip_illustration as ti  # noqa: E402  (raster helpers of the paper's Fig. 2)

# ------------------------------------------------------------------------------------------------ look

W_IN, H_IN = 19.2, 10.8
ASPECT = H_IN / W_IN
BG = "#0b0f1a"
WHITE, GRAY, DIM = "white", "#b8bfcc", "#7d8596"
#: Fig. 4's flavor colors (analysis/trajectory_figure.FLAVOR_COLORS), brightened a little for the dark background.
FLAV = ["#ef7d2e", "#9a93e6", "#2fbf93"]
FLAV_LABEL = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
#: One color per family, the same in the panel, the flavor triangle and the spectrum.
FAM = {"si": ("Secret interactions", "#c792ea"), "dp": ("Dark portal", "#ff8c69"), "le": (r"$L_e - L_\mu$", "#ffd166")}
ROWS = ("si", "dp", "le")                      # the order in which they act along the trip (Fig. 4)
BEAM, BEAM_GLOW, SECONDARY, MEDIATOR, RELIC = ti.BEAM, ti.BEAM_GLOW, ti.SECONDARY, "#ff7eb6", "#5be0bd"
#: The limits' data sets, the paper's colors (analysis/figure_style.DATASETS), brightened for the dark background.
TODAY, FUTURE = ("#9b2226", "#ef5b5f"), ("#0a9396", "#47d6cc")
HALO = [patheffects.withStroke(linewidth=4, foreground=BG, alpha=0.85)]

# ------------------------------------------------------------------------------------------------ timeline (s)

SPEC = [("title", 0.0, 5.0), ("question", 5.0, 12.0), ("source", 12.0, 19.0), ("universe", 19.0, 45.0),
        ("galaxy", 45.0, 59.0), ("sun", 59.0, 67.0), ("icecube", 67.0, 75.0), ("earth", 75.0, 89.0),
        ("limits", 89.0, 95.0), ("telescopes", 95.0, 103.0), ("future", 103.0, 109.0), ("card", 109.0, 115.0)]
CARD_T0 = 109.0
DURATION = SPEC[-1][2]

CAPTIONS = [
    (5.3, 8.6, "Do neutrinos feel new interactions, carried by light particles?"),
    (8.6, 12.4, "We test three kinds of mediator, one at a time."),
    (12.6, 18.8, "High-energy neutrinos leave a distant source, around redshift 2."),
    (19.0, 24.0, "On the way, they cross the relic neutrinos left over from the early Universe."),
    (24.0, 28.5, "With secret interactions, a relic neutrino can absorb a high-energy one."),
    (28.5, 33.0, "The mediator then decays into two neutrinos of lower energy."),
    (33.0, 40.0, "With the other two, heavier neutrinos decay into lighter ones."),
    (46.3, 51.5, r"Of the three, only $L_e - L_\mu$ feels the Milky Way's stars and gas."),
    (51.5, 58.2, "Their potential changes the flavor of the neutrinos."),
    (59.3, 66.8, "The Sun, the Moon and the Earth add their own potentials."),
    (67.5, 74.7, "IceCube, at the South Pole, measures their flavors and energies."),
    (75.6, 81.8, r"The flavor composition disfavors $L_e - L_\mu$ and the dark portal."),
    (82.2, 88.7, "The spectrum disfavors secret interactions."),
    (89.5, 94.8, "Following the whole trip moves the limits by many orders of magnitude."),
    (95.4, 102.7, "A forecast for eight neutrino telescopes, each taking data for ten years."),
    (103.4, 108.8, "Together, they could tighten the limits by factors of about 2 to 60."),
]
TITLE = ("New interactions of high-energy cosmic neutrinos,", "from the Hubble scale to Earth")
#: Mariia Petropavlova second (Mauricio, 2026-11-27); not yet in main.tex's author block, so verify() does not check
#: her name -- add it to CHECKS once the paper lists her.
AUTHORS = "Mauricio Bustamante, Mariia Petropavlova, Sudipta Das, Sanjib Kumar Agarwalla"
#: What verify() requires in main.tex, with the line it was read from on 2026-11-27 (main checkout, branch dev).
CHECKS = [("New interactions of high-energy cosmic neutrinos, from the Hubble scale to Earth", "l. 67, title"),
          ("Mauricio Bustamante", "l. 69"), ("Sudipta Das", "l. 73"), ("Sanjib Kumar Agarwalla", "l. 77"),
          ("Do neutrinos feel new interactions, carried by light particles that couple to them?", "l. 103"),
          ("shell at $z = 1.8$--$2.2$", "l. 454, the sources' shell"), ("observed energy of 495~TeV", "l. 454"),
          ("$\\nu_e : \\nu_\\mu : \\nu_\\tau = 1 : 2 : 0$", "l. 650, pion decay"),
          ("left over from the early Universe", "l. 126"),
          ("for $M = 10$~MeV, it falls at 1.0 and 5.8~PeV on the two heavier relics", "l. 237"),
          ("the first through absorption, the second through decays and scatterings", "l. 456"),
          ("Near the Sun, the potentials of the Sun, the Moon, and the Earth switch on", "l. 140"),
          ("IceCube now measures both their energy spectrum and their flavor composition", "l. 125"),
          ("the flavor composition for $L_e - L_\\mu$ and the dark portal, the spectrum for secret interactions",
           "l. 643"),
          ("The full transport moves the limits by many orders of magnitude", "l. 89, abstract"),
          ("Eight future neutrino telescopes could tighten the limits by factors of about 2 to 60", "l. 98 (re-verify)"),
          ("the events that eight neutrino telescopes would detect in ten years each", "l. 491"),
          ("Eight neutrino telescopes, each taking data for ten years", "l. 799"),
          ("at its own site and size", "l. 519")]


def verify():
    tex = open(os.path.join(REPO, "paper", "main.tex")).read()
    missing = [f"{s!r} ({where})" for s, where in CHECKS if s not in tex]
    if missing:
        raise SystemExit("main.tex no longer says:\n  " + "\n  ".join(missing))


def ramp(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return float(x * x * (3 - 2 * x))


def window(t, t0, t1, fade=0.35):
    return min(ramp(t, t0, t0 + fade), 1.0 - ramp(t, t1 - fade, t1))


def logmix(a, b, f):
    return float(np.exp(np.log(a) + f * (np.log(b) - np.log(a))))


def scene_at(t):
    for name, t0, t1 in SPEC:
        if t0 <= t < t1:
            return name, t - t0
    return SPEC[-1][0], t - SPEC[-1][1]


# ------------------------------------------------------------------------------------------------ the physics shown

D_IN = np.load(INPUTS, allow_pickle=False)
DATA = {k: D_IN[k] for k in D_IN.files}

#: Redshift of the beam's head versus time (monotone; the pause at the absorption at 25.5-27 s).
Z_KEYS = PchipInterpolator([15.0, 19.0, 24.0, 25.5, 27.0, 33.0, 40.0, 45.0],
                           [1.80, 1.80, 1.77, 1.745, 1.74, 1.40, 0.70, 0.0])
T_ABS, T_DEC = 25.5, 34.5                       # the absorption (paused) and the decay drawn on the beam


def z_of(t):
    return float(Z_KEYS(np.clip(t, 15.0, 45.0)))


def x_of(t):
    return float(X_KEYS(np.clip(t, 46.0, 58.5)))


def comoving(z, om=0.315):
    """Comoving distance in units of c/H0 (Planck 2018's matter density; the paper's cosmology, l. 334)."""
    zz = np.linspace(0.0, z, 400)
    return float(np.trapz(1.0 / np.sqrt(om * (1 + zz) ** 3 + 1 - om), zz))


DC18 = comoving(1.8)


def flux(key, t):
    """The flux at 495 TeV per flavor (nu + nubar) relative to leaving the sources, for the family `key`, where the
    head is at time t (Fig. 4's arrays; data/inputs.npz)."""
    k = {"le": "le_20"}.get(key, key)
    z, U = DATA[f"trip/{k}/z"], DATA[f"trip/{k}/universe"]
    zq = z_of(t)
    if t < 46.0 or f"trip/{k}/galaxy" not in DATA:
        return np.array([np.interp(zq, z[::-1], U[::-1, f]) for f in range(3)])
    x, G = DATA[f"trip/{k}/galaxy_x"], DATA[f"trip/{k}/galaxy"]
    xq = x_of(t)
    return np.array([np.interp(xq, x[::-1], G[::-1, f]) for f in range(3)])


# ------------------------------------------------------------------------------------------------ the worlds

D = np.array([np.sqrt(1 - 0.64 ** 2), -0.64])  # the beam's direction on screen: the chord into IceCube, cos(theta_z)
                                               # = -0.64 (Fig. 4's caption), kept in every world (no bends)

# The Milky Way (kpc): Fig. 2's spiral, tilted (ti.draw_galaxy's incl, rot), disk radius 15 kpc, the Sun at 8.3 kpc
MW_R, MW_INCL, MW_ROT, SUN_R = 15.0, 0.5, -0.3, 8.33   # the Sun at dp_spatial's R_0
MW_VIEW = (0.0, 0.0, 48.0)                     # center, width (kpc)
_M = np.array([[np.cos(MW_ROT), -MW_INCL * np.sin(MW_ROT)], [np.sin(MW_ROT), MW_INCL * np.cos(MW_ROT)]])
_dir_plane = np.linalg.solve(_M, D)
_dir_plane /= np.linalg.norm(_dir_plane)
_alpha = np.arctan2(_dir_plane[1], _dir_plane[0])
_phi = _alpha - np.arcsin(2.0 / SUN_R)         # the line of sight passes 2 kpc from the Galactic Centre (schematic)
SUN_PLANE = SUN_R * np.array([np.cos(_phi), np.sin(_phi)])


def mw_project(p):
    return _M @ np.asarray(p)


SUN_MW = mw_project(SUN_PLANE)
L_MW = 30.25                                   # the line of sight of Fig. 4's arrays (galaxy_x), kpc

# Each world's view fills a box in the world before it (a uniform scale and shift), so the beam is one straight line
# through all four, and each world draws the next one's object (the galaxy, the Sun, the Earth) mapped into its box:
# the zooms land on the same image, and only the star fields cross-fade.

# Near the Sun (AU): the Sun at the origin; the beam passes the Earth perpendicular to the direction of the Sun, as
# Fig. 4's path past the Sun (its caption)
E_AU = np.array([-D[1], D[0]])
SUN_VIEW = (0.45, 0.35, 3.6)
SUN_BOX = 0.7                                  # kpc, the half-width of the Milky Way's box around the Sun
EARTH_BOX = 0.07                               # AU, the half-width of the box around the Earth
MOON_AU = E_AU + np.array([0.06, 0.08])        # schematic: off the beam's side, outside the Earth's box

# The Earth (km): its center at the origin, IceCube at the South Pole, the chord at cos(theta_z) = -0.64
R_E = 6371.0
EARTH_VIEW = (6500.0, -1200.0, 30000.0)        # the Earth left of the panel, clear of the caption
SP = np.array([0.0, -R_E])
CHORD = 2 * R_E * 0.64
ENTRY = SP - CHORD * D
DET = SP + np.array([0.0, 1.95])               # IceCube's center, 1.95 km deep (1.45-2.45 km)

K_E = EARTH_BOX / (EARTH_VIEW[2] / 2)          # AU per km, across the Earth's box
K_S = SUN_BOX / (SUN_VIEW[2] / 2)              # kpc per AU, across the Sun's box
EARTH_BOX_C = E_AU + np.array(EARTH_VIEW[:2]) * K_E
MW_SUNBOX_C = SUN_MW + np.array(SUN_VIEW[:2]) * K_S


def earth_to_sun(p):
    return E_AU + np.asarray(p) * K_E


def sun_to_mw(p):
    return SUN_MW + np.asarray(p) * K_S


END_SUN = earth_to_sun(ENTRY)                  # where the beam meets the Earth, in the Sun's world
OFFSET_MW = sun_to_mw(END_SUN) - SUN_MW        # so the Milky Way's line of sight passes the Sun by that much


def mw_head(xkpc):
    return mw_project(SUN_PLANE - xkpc * _dir_plane) + OFFSET_MW


# The Universe (arbitrary units, 48 x 27): the Milky Way's view fills a box around MW_U
UW, UH = 48.0, 27.0
MW_U, MW_BOX = np.array([30.0, 5.5]), 1.4      # the box's center and half-width in the Universe
U_PER_KPC = MW_BOX / (MW_VIEW[2] / 2)


def mw_to_u(p):
    return MW_U + np.asarray(p) * U_PER_KPC


def _in_view(c, half_w, margin=0.97):
    return lambda p: abs(p[0] - c[0]) <= margin * half_w and abs(p[1] - c[1]) <= margin * half_w * ASPECT


def _last_inside(f, lo, hi, inside, n=6000):
    """The largest x in [lo, hi] up to which f(x) stays inside the view (it is inside at lo)."""
    xs = np.linspace(lo, hi, n)
    ok = np.array([inside(f(x)) for x in xs])
    return float(xs[-1] if ok.all() else xs[max(int(np.argmin(ok)) - 1, 0)])


X_EDGE = _last_inside(mw_head, 0.0, L_MW, _in_view(MW_VIEW[:2], MW_VIEW[2] / 2))   # enters the Milky Way's view
X_BOX = _last_inside(mw_head, 0.0, L_MW, _in_view(MW_SUNBOX_C, SUN_BOX))            # enters the Sun's box
SUN_START = END_SUN - _last_inside(lambda l: END_SUN - l * D, 0.0, 20.0,
                                   _in_view(SUN_VIEW[:2], SUN_VIEW[2] / 2)) * D       # enters the Sun's view
assert X_EDGE > 15.0 and X_BOX < 2.0, (X_EDGE, X_BOX)
#: Distance to the Sun (kpc) of the head in the Milky Way, versus time: the flavor changes between 15 and 2 kpc.
X_KEYS = PchipInterpolator([46.0, 48.5, 55.0, 58.5], [X_EDGE, 15.0, 2.0, X_BOX])
H_END_U = mw_to_u(mw_head(X_EDGE))             # where the beam enters the Milky Way's view
S_U = H_END_U - 27.0 * D                       # the source, upstream on the same line
S0 = 0.03                                      # the fraction of the path the beam has covered when the bars appear


def sun_head(t):
    return SUN_START + ramp(t, 59.0, 61.6) * (END_SUN - SUN_START)


def earth_head(t):
    """The head in the Earth's world: at the limb until 64.5 s, along the chord to IceCube by 67 s, then into the
    array (70.4-71.3 s)."""
    end = DET - 2.5 * D
    if t < 64.5:
        return ENTRY
    if t < 70.4:
        return ENTRY + ramp(t, 64.5, 67.0) * (end - ENTRY)
    return end + ramp(t, 70.4, 71.3) * ((DET - 0.35 * D) - end)


def u_head(t):
    if t < 15.0:
        s = 0.0
    elif t < 17.0:
        s = S0 * ramp(t, 15.0, 17.0)
    else:
        s = S0 + (1 - S0) * (1 - comoving(z_of(t)) / DC18)
    return S_U + s * (H_END_U - S_U)


# ------------------------------------------------------------------------------------------------ the potential

#: The L_e - L_mu potential of the example, relative to Delta m^2_31 / (2E) (video_inputs.potential), drawn as a tint
#: over the scene: its color gives log10 of the ratio from 0 to 16; it is transparent where the ratio is below 1 (the
#: potential does not set the flavor there) and reaches its full opacity at 10^5.
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
POT_CMAP = LinearSegmentedColormap.from_list("potential", ["#2d1b78", "#7b2fa8", "#d1478f", "#ff9ec8", "#ffe6f2"])
POT_LO, POT_HI, POT_ALPHA = -6.0, 16.0, 0.42


def pot_rgba(logr):
    """RGBA of the tint for log10(potential / oscillation term)."""
    logr = np.asarray(logr, dtype=float)
    rgba = np.array(POT_CMAP(np.clip(logr / POT_HI, 0.0, 1.0)))
    rgba[..., 3] = POT_ALPHA * np.clip(logr / 5.0, 0.0, 1.0)
    return rgba


def pot_mw(plane_xy):
    """log10 of the ratio in the Milky Way's plane, at galactocentric (u, v) kpc."""
    R = np.hypot(*np.asarray(plane_xy, dtype=float))
    return np.log10(np.interp(R, DATA["pot/mw_R"], DATA["pot/mw"]))


def pot_at_head(t):
    """log10 of the ratio where the neutrino is."""
    if t < 45.7:
        return float(np.log10(np.interp(max(z_of(t), 0.0), DATA["pot/z"], DATA["pot/universe"])))
    if t < 58.75:
        return float(pot_mw(SUN_PLANE - x_of(t) * _dir_plane))
    if t < 64.25:
        r = float(np.hypot(*sun_head(t)))
        return float(np.log10(np.interp(r, DATA["pot/sun_r_au"], DATA["pot/sun"])))
    return float(np.log10(np.interp(float(np.hypot(*earth_head(t))), DATA["pot/earth_r_km"], DATA["pot/earth"])))


class Cam:
    def __init__(self, cx, cy, w, roll=0.0, pivot=None):
        """A view centered at (cx, cy), w wide; rolled by `roll` degrees about `pivot` (default: the center)."""
        self.cx, self.cy, self.w, self.roll = float(cx), float(cy), float(w), float(roll)
        self.pivot = (self.cx, self.cy) if pivot is None else tuple(pivot)

    @property
    def h(self):
        return self.w * ASPECT

    def mapped(self, cA, hA, cB, hB):
        """This camera in world A, seen in world B, whose view (center cB, half-width hB) fills A's box (cA, hA)."""
        k = hB / hA
        return Cam(cB[0] + (self.cx - cA[0]) * k, cB[1] + (self.cy - cA[1]) * k, self.w * k, self.roll)

    @staticmethod
    def mix(a, b, f):
        return Cam(a.cx + f * (b.cx - a.cx), a.cy + f * (b.cy - a.cy), logmix(a.w, b.w, f), a.roll + f * (b.roll - a.roll))

    @staticmethod
    def zoom(a, b, f):
        """From view a to view b about their fixed point, the one place that sits at the same spot on screen in both:
        everything moves straight away from it, so what the zoom aims at never leaves the frame."""
        w = logmix(a.w, b.w, f)
        if abs(b.w - a.w) < 1e-9 * a.w:
            return Cam.mix(a, b, f)
        P = (np.array([a.cx, a.cy]) * b.w - np.array([b.cx, b.cy]) * a.w) / (b.w - a.w)
        c = P + (np.array([a.cx, a.cy]) - P) * (w / a.w)
        return Cam(c[0], c[1], w)


def follow_u(t, w=12.0):
    """The Universe's camera following the head, which sits 30% from the left and 58% from the bottom."""
    hx, hy = u_head(t)
    return Cam(hx + 0.20 * w, hy + (0.5 - 0.58) * w * ASPECT, w)


U_WIDE = Cam(UW / 2, UH / 2, UW)
MW_FULL = Cam(*MW_VIEW)
SUN_FULL = Cam(*SUN_VIEW)
EARTH_FULL = Cam(*EARTH_VIEW)
#: The last view, before its roll by 180 degrees about IceCube's center (which stays put on screen), so that the ice
#: ends on top and the neutrino arrives from below: the surface 88% up the frame, the deepest sensors (2.45 km) 15%
#: up, the array a third of the way across, left of the panel.
PIVOT = (0.0, -R_E + 1.95)
ICECUBE = Cam(1.0, -R_E + 2.625, 6.0)
_U_ZOOM_END = Cam(MW_U[0], MW_U[1], 2 * MW_BOX)


def cam_u(t):
    if t < 12.0:
        return Cam(UW / 2, UH / 2, logmix(UW, 0.92 * UW, t / 12.0))
    if t < 15.0:
        return Cam.zoom(Cam(UW / 2, UH / 2, 0.92 * UW), follow_u(15.0), ramp(t, 12.0, 15.0))
    if t < 43.0:
        return follow_u(t)
    return Cam.zoom(follow_u(43.0), _U_ZOOM_END, ramp(t, 43.0, 46.0))


def cam_mw(t):
    if t < 46.0:
        return cam_u(t).mapped(MW_U, MW_BOX, MW_VIEW[:2], MW_VIEW[2] / 2)
    if t < 56.5:
        return Cam(0.0, 0.0, logmix(MW_VIEW[2], 0.92 * MW_VIEW[2], ramp(t, 46.0, 56.5)))
    start = Cam(0.0, 0.0, 0.92 * MW_VIEW[2])
    return Cam.zoom(start, Cam(MW_SUNBOX_C[0], MW_SUNBOX_C[1], 2 * SUN_BOX), ramp(t, 56.5, 59.0))


def cam_sun(t):
    if t < 59.0:
        return cam_mw(t).mapped(MW_SUNBOX_C, SUN_BOX, SUN_VIEW[:2], SUN_VIEW[2] / 2)
    if t < 62.3:
        return SUN_FULL
    return Cam.zoom(SUN_FULL, Cam(EARTH_BOX_C[0], EARTH_BOX_C[1], 2 * EARTH_BOX), ramp(t, 62.3, 64.5))


def cam_earth(t):
    if t < 64.5:
        return cam_sun(t).mapped(EARTH_BOX_C, EARTH_BOX, EARTH_VIEW[:2], EARTH_VIEW[2] / 2)
    if t < 67.0:
        return EARTH_FULL
    c = Cam.zoom(EARTH_FULL, ICECUBE, ramp(t, 67.0, 69.8))
    return Cam(c.cx, c.cy, c.w, 180.0 * ramp(t, 69.2, 70.6), PIVOT)


# ------------------------------------------------------------------------------------------------ raster art

def _art_canvas(w_units, h_units, px_per_unit):
    nx, ny = int(round(w_units * px_per_unit)), int(round(h_units * px_per_unit))
    acc = np.zeros((ny, nx, 3), dtype=np.float32)
    return acc, ti.background(acc).astype(np.float32)


def _galaxy(acc, cx_px, cy_px, R_px, n, amp, seed):
    """ti.draw_galaxy on a canvas in pixels: its coordinates are inches at ti.DPI_ART, the rows counted from ti.Y1."""
    ny = acc.shape[0]
    ti.DPI_ART, ti.Y1 = 420, ny / 420.0
    ti.draw_galaxy(acc, (0.0, acc.shape[1] / 420.0), cx_px / 420.0, ti.Y1 - cy_px / 420.0, R_px / 420.0,
                   incl=MW_INCL, rot=MW_ROT, n=n, amp=amp, seed=seed)


def _local_blob(acc, c, r, sigma, color, amp):
    ny, nx = acc.shape[:2]
    k = int(4 * sigma) + 1
    r0, r1, c0, c1 = max(0, int(r) - k), min(ny, int(r) + k), max(0, int(c) - k), min(nx, int(c) + k)
    if r0 >= r1 or c0 >= c1:
        return
    yy, xx = np.mgrid[r0:r1, c0:c1]
    g = amp * np.exp(-((xx - c) ** 2 + (yy - r) ** 2) / (2 * sigma ** 2))
    acc[r0:r1, c0:c1] += (g[..., None] * np.asarray(color)[None, None, :]).astype(np.float32)


def _local_ellipse(acc, c, r, sx, sy, ang, color, amp):
    ny, nx = acc.shape[:2]
    k = int(4 * max(sx, sy)) + 1
    r0, r1, c0, c1 = max(0, int(r) - k), min(ny, int(r) + k), max(0, int(c) - k), min(nx, int(c) + k)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    ca, sa = np.cos(ang), np.sin(ang)
    u = (xx - c) * ca + (yy - r) * sa
    v = -(xx - c) * sa + (yy - r) * ca
    g = amp * np.exp(-u ** 2 / (2 * sx ** 2) - v ** 2 / (2 * sy ** 2))
    acc[r0:r1, c0:c1] += (g[..., None] * np.asarray(color)[None, None, :]).astype(np.float32)


def _to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def art_universe():
    """The Universe: stars, a faint cosmic web, distant galaxies, the source (with jets), the Milky Way at MW_U."""
    ppu = 80
    acc, base = _art_canvas(UW, UH, ppu)
    ny, nx = acc.shape[:2]
    ti.rng = np.random.default_rng(7)
    rng = ti.rng
    ti.stars(acc, 9000, 0.5)
    nodes = np.c_[rng.uniform(0, nx, 90), rng.uniform(0, ny, 90)]
    web = []
    for a in nodes:
        d = np.hypot(*(nodes - a).T)
        for j in np.argsort(d)[1:3]:
            s = rng.uniform(0, 1, 900)[:, None]
            web.append(a + s * (nodes[j] - a) + rng.normal(0, 10, (900, 2)))
    web = np.vstack(web)
    ti.splat(acc, web[:, 0], web[:, 1], 0.05 * np.ones(len(web)), [0.55, 0.35, 0.85], 8)
    px = lambda p: (p[0] * ppu, (UH - p[1]) * ppu)
    # distant galaxies, away from the beam's line
    g = np.random.default_rng(11)
    for _ in range(60):
        p = np.array([g.uniform(0, UW), g.uniform(0, UH)])
        off = p - S_U
        if abs(off[0] * D[1] - off[1] * D[0]) < 1.5:
            continue
        c, r = px(p)
        s = g.uniform(0.06, 0.22) * ppu
        _local_ellipse(acc, c, r, s, s * g.uniform(0.35, 0.8), g.uniform(0, np.pi), [1.0, 0.85, 0.65], 1.0)
        _local_blob(acc, c, r, 2.5, [1.0, 0.95, 0.85], 2.5)
    # the source: a bright galaxy with jets
    c, r = px(S_U)
    _local_ellipse(acc, c, r, 0.45 * ppu, 0.2 * ppu, 0.5, [1.0, 0.85, 0.65], 1.5)
    _local_blob(acc, c, r, 5.0, [1.0, 0.95, 0.85], 5.0)
    for sgn in (1, -1):                         # short, faint jets, across the beam's direction
        L = np.linspace(0, 0.55 * ppu, 500)
        ang = np.arctan2(D[0], D[1]) + 0.35
        ti.splat(acc, c + sgn * L * np.cos(ang), r - sgn * L * np.sin(ang), np.exp(-L / (0.25 * ppu)) * 0.35,
                 [0.6, 0.75, 1.0], 1.6)
    return _to_uint8(ti.tone(acc, base)), (0.0, UW, 0.0, UH)


def _rgba(acc):
    """Additive light -> RGBA over nothing (alpha from the brightest channel): an object without a background."""
    light = 1.0 - np.exp(-acc)
    a = np.clip(light.max(axis=2) * 1.15, 0.0, 1.0)
    rgb = np.clip(light / np.maximum(a, 1e-6)[..., None], 0.0, 1.0)
    return (np.dstack([rgb, a[..., None]]) * 255).astype(np.uint8)


def _starfield(c, w, ppu, n, seed):
    """A world's background: stars over three times its view, so that the zoom into it never shows an edge."""
    h = w * ASPECT
    acc, base = _art_canvas(w, h, ppu)
    ti.rng = np.random.default_rng(seed)
    ti.stars(acc, n, 0.5)
    return _to_uint8(ti.tone(acc, base)), (c[0] - w / 2, c[0] + w / 2, c[1] - h / 2, c[1] + h / 2)


def art_galaxy_bg():
    return _starfield(MW_VIEW[:2], 3 * MW_VIEW[2], 30, 14000, 8)


def art_galaxy():
    """The Milky Way (Fig. 2's spiral, tilted), radius 15 kpc, without a background."""
    ppu, half = 60, 17.0
    n = int(2 * half * ppu)
    acc = np.zeros((n, n, 3), dtype=np.float32)
    _galaxy(acc, n / 2, n / 2, MW_R * ppu, n=400000, amp=0.30, seed=3)
    return _rgba(acc), (-half, half, -half, half)


def art_sun_bg():
    return _starfield(SUN_VIEW[:2], 3 * SUN_VIEW[2], 300, 6000, 9)


def art_sun():
    """The Sun's glow (Fig. 2), without a background."""
    ppu, half = 350, 2.8
    n = int(2 * half * ppu)
    acc = np.zeros((n, n, 3), dtype=np.float32)
    c = n / 2
    _local_blob(acc, c, c, 0.55 * ppu, [1.0, 0.55, 0.15], 0.5)
    _local_blob(acc, c, c, 0.24 * ppu, [1.0, 0.75, 0.35], 1.4)
    _local_blob(acc, c, c, 0.10 * ppu, [1.0, 0.95, 0.8], 4.0)
    return _rgba(acc), (-half, half, -half, half)


def art_earth_bg():
    return _starfield(EARTH_VIEW[:2], 3 * EARTH_VIEW[2], 0.05, 4000, 10)


LON0 = -15.0                                   # the central meridian, as Fig. 2 (Africa and Europe)


def land_rings():
    ti.LAND = LAND50
    return ti.globe_rings(LON0)


def art_earth():
    """The Earth as Fig. 2 (ti.panel_earth), without a background: the left half cut open on PREM's layers, the right
    half the surface with Natural Earth's 1:50m land, lit from the upper left; the atmosphere's rim."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
    ppu = 0.15                                 # pixels per km
    half = int(np.ceil(1.03 * R_E * ppu))
    n = 2 * half
    f32 = np.float32
    xs = ((np.arange(n) + 0.5 - half) / ppu).astype(f32)
    u, v = np.meshgrid(xs / f32(R_E), -xs / f32(R_E))
    rho = np.hypot(u, v)
    inside = rho <= 1.0
    nz = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
    L = np.array([-0.35, 0.15, 0.92]) / np.linalg.norm([-0.35, 0.15, 0.92])
    light = (0.33 + 0.75 * np.clip(L[0] * u + L[1] * v + L[2] * nz, 0, None)).astype(f32)
    del nz
    mask = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(mask)
    for ring in land_rings():
        cols = ring[:, 0] * R_E * ppu + half
        rows = -ring[:, 1] * R_E * ppu + half
        draw.polygon(list(zip(cols.tolist(), rows.tolist())), fill=255)
    land = (np.asarray(mask) > 127) & inside
    lat = np.degrees(np.arcsin(np.clip(v, -1, 1))).astype(f32)
    land |= inside & (lat < -69.5)
    patchy = gaussian_filter(np.random.default_rng(12).random(lat.shape).astype(f32), 6)
    patchy = (patchy - patchy.min()) / (patchy.max() - patchy.min())
    desert = ((np.exp(-((lat - 22.0) / 8.0) ** 2) + 0.6 * np.exp(-((lat + 24.0) / 5.0) ** 2))
              * np.clip(0.4 + 1.2 * patchy, 0, 1)).astype(f32)
    del patchy
    out = np.zeros((n, n, 4), dtype=f32)
    for ch, (ocean, green, tan, ice) in enumerate(zip((0.12, 0.35, 0.75), (0.27, 0.45, 0.22), (0.72, 0.6, 0.4),
                                                      (0.93, 0.95, 1.0))):
        land_c = np.where(lat < -60.0, ice, (1 - desert) * green + desert * tan)
        out[..., ch] = np.where(land, land_c, ocean) * light
    del desert, light, land, lat
    cut = inside & (u < 0)                     # the left half shows PREM's layers (ti.LAYERS)
    shade = (0.82 + 0.18 * (1 - rho)).astype(f32)
    for r_km, color in ti.LAYERS:
        m = cut & (rho <= r_km / R_E)
        for ch in range(3):
            out[..., ch] = np.where(m, color[ch] * shade, out[..., ch])
    rim = (np.exp(-((rho - 1.0) / 0.015) ** 2) * (rho > 0.985)).astype(f32)
    for ch, k in enumerate((0.42, 0.65, 1.0)):
        out[..., ch] = np.where(inside, out[..., ch] + 0.35 * k * rim, k)
    out[..., 3] = np.where(inside, 1.0, np.clip(0.6 * rim, 0, 1))
    e = half / ppu
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), (-e, e, -e, e)


def _tint(ext, nx, f):
    """A tint layer: pot_rgba of f(x, y) (log10 of the ratio, world coordinates) on an nx-wide grid over ext."""
    x0, x1, y0, y1 = ext
    ny = int(round(nx * (y1 - y0) / (x1 - x0)))
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y1, y0, ny))
    return (pot_rgba(f(X, Y)) * 255).astype(np.uint8), ext


def art_galaxy_tint():
    """The Milky Way's potential: each point of the screen taken back to the disk's plane (the drawing's tilt)."""
    Minv = np.linalg.inv(_M)
    return _tint((-64.0, 64.0, -36.0, 36.0), 2048,
                 lambda X, Y: pot_mw((Minv[0, 0] * X + Minv[0, 1] * Y, Minv[1, 0] * X + Minv[1, 1] * Y)))


def art_sun_tint():
    w = 3 * SUN_VIEW[2]
    ext = (SUN_VIEW[0] - w / 2, SUN_VIEW[0] + w / 2, SUN_VIEW[1] - w * ASPECT / 2, SUN_VIEW[1] + w * ASPECT / 2)
    return _tint(ext, 1620, lambda X, Y: np.log10(np.interp(np.hypot(X, Y), DATA["pot/sun_r_au"], DATA["pot/sun"])))


def art_earth_tint():
    w = 3 * EARTH_VIEW[2]
    ext = (EARTH_VIEW[0] - w / 2, EARTH_VIEW[0] + w / 2, EARTH_VIEW[1] - w * ASPECT / 2,
           EARTH_VIEW[1] + w * ASPECT / 2)
    return _tint(ext, 900, lambda X, Y: np.log10(np.interp(np.hypot(X, Y), DATA["pot/earth_r_km"],
                                                           DATA["pot/earth"])))


ART = {"galaxy_tint": art_galaxy_tint, "sun_tint": art_sun_tint, "earth_tint": art_earth_tint,
       "universe": art_universe, "galaxy_bg": art_galaxy_bg, "galaxy": art_galaxy, "sun_bg": art_sun_bg,
       "sun": art_sun, "earth_bg": art_earth_bg, "earth": art_earth}
_ART_CACHE = {}


def art(name):
    if name not in _ART_CACHE:
        path = os.path.join(CACHE, f"{name}.npz")
        if not os.path.exists(path):
            os.makedirs(CACHE, exist_ok=True)
            t0 = time.time()
            img, ext = ART[name]()
            np.savez(path, img=img, ext=np.asarray(ext))
            print(f"  art {name}: {img.shape[1]}x{img.shape[0]}, {time.time() - t0:.0f} s", flush=True)
        d = np.load(path)
        _ART_CACHE[name] = (d["img"], tuple(d["ext"]))
    return _ART_CACHE[name]


# ------------------------------------------------------------------------------------------------ drawing a world

def world_axes(fig, cam):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(cam.cx - cam.w / 2, cam.cx + cam.w / 2)
    ax.set_ylim(cam.cy - cam.h / 2, cam.cy + cam.h / 2)
    ax.axis("off")
    ax.patch.set_alpha(0.0)
    T = Affine2D().rotate_deg_around(*cam.pivot, cam.roll) + ax.transData if cam.roll else ax.transData
    return ax, T


def view_bbox(cam):
    """The part of the world in view: the view's corners turned back about the pivot when the camera is rolled."""
    xs = cam.cx + np.array([-1, 1, 1, -1]) * cam.w / 2
    ys = cam.cy + np.array([-1, -1, 1, 1]) * cam.h / 2
    if cam.roll:
        a = np.radians(-cam.roll)
        px, py = cam.pivot
        xs, ys = (px + (xs - px) * np.cos(a) - (ys - py) * np.sin(a),
                  py + (xs - px) * np.sin(a) + (ys - py) * np.cos(a))
    return xs.min(), xs.max(), ys.min(), ys.max()


def raster(ax, T, name, cam, alpha, to_world=None, z=0):
    """Art `name` in this world, cropped to the view. to_world maps the art's own coordinates into this world (a
    uniform scale and shift: the next world's object seen through its box)."""
    if alpha <= 0.0:
        return
    img, (x0, x1, y0, y1) = art(name)
    if to_world is not None:
        (x0, y0), (x1, y1) = to_world((x0, y0)), to_world((x1, y1))
    ny, nx = img.shape[:2]
    bx0, bx1, by0, by1 = view_bbox(cam)
    sx, sy = nx / (x1 - x0), ny / (y1 - y0)
    c0 = int(np.clip(np.floor((bx0 - x0) * sx) - 2, 0, nx))
    c1 = int(np.clip(np.ceil((bx1 - x0) * sx) + 2, 0, nx))
    r0 = int(np.clip(np.floor((y1 - by1) * sy) - 2, 0, ny))
    r1 = int(np.clip(np.ceil((y1 - by0) * sy) + 2, 0, ny))
    if c1 - c0 < 2 or r1 - r0 < 2:
        return
    ext = (x0 + c0 / sx, x0 + c1 / sx, y1 - r1 / sy, y1 - r0 / sy)
    ax.imshow(img[r0:r1, c0:c1], extent=ext, origin="upper", interpolation="bilinear", alpha=alpha, zorder=z,
              transform=T)


def glow(ax, T, x, y, alpha=1.0, lw=2.4, color=BEAM, glow_color=BEAM_GLOW, z=10):
    for w, a in ((7 * lw, 0.05), (4.5 * lw, 0.09), (2.6 * lw, 0.2)):
        ax.plot(x, y, color=glow_color, lw=w, alpha=a * alpha, solid_capstyle="round", zorder=z, transform=T)
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z + 1, transform=T)


def head_dot(ax, T, p, alpha, size=1.0, t=None):
    if t is not None and 40.0 <= t < 68.5:     # the potential's glow on the neutrino (L_e - L_mu)
        c = pot_rgba(pot_at_head(t))
        if c[3] > 0:
            for s, a in ((5200, 0.35), (2400, 0.6)):
                ax.scatter([p[0]], [p[1]], s=s * size, color=c[:3], alpha=a * c[3] / POT_ALPHA * alpha, lw=0,
                           zorder=11, transform=T)
    for s, a in ((900, 0.06), (380, 0.12), (130, 0.3)):
        ax.scatter([p[0]], [p[1]], s=s * size, color=BEAM_GLOW, alpha=a * alpha, lw=0, zorder=12, transform=T)
    ax.scatter([p[0]], [p[1]], s=34 * size, color="white", alpha=alpha, lw=0, zorder=13, transform=T)


def label(ax, T, text, xy, xytext, alpha=1.0, fs=22, ha="center", color=WHITE, z=30, italic=False, halo=True):
    """A label at xytext with a leader that ends on its object, xy (no leader if xy is None)."""
    if alpha <= 0.01:
        return
    kw = dict(arrowprops=dict(arrowstyle="-", lw=1.1, color=color, alpha=0.85 * alpha, shrinkA=3, shrinkB=0)) \
        if xy is not None else {}
    a = ax.annotate(text, xy=xy if xy is not None else xytext, xytext=xytext, ha=ha, va="center", fontsize=fs,
                    color=color, alpha=alpha, zorder=z, xycoords=T, textcoords=T, style="italic" if italic else "normal",
                    **kw)
    if halo:
        a.set_path_effects(HALO)


def wiggle(p0, p1, amp, n_waves, n=300):
    s = np.linspace(0, 1, n)
    d = np.asarray(p1) - np.asarray(p0)
    L = np.hypot(*d)
    nrm = np.array([-d[1], d[0]]) / L
    return (p0[0] + s * d[0] + nrm[0] * amp * np.sin(2 * np.pi * n_waves * s),
            p0[1] + s * d[1] + nrm[1] * amp * np.sin(2 * np.pi * n_waves * s))


_RELICS = np.random.default_rng(21).uniform([0, 0], [UW, UH], (2600, 2))


def draw_universe(fig, t, cam, alpha):
    ax, T = world_axes(fig, cam)
    raster(ax, T, "universe", cam, alpha)
    raster(ax, T, "galaxy", cam, alpha, to_world=mw_to_u, z=1)      # the Milky Way, as its own world sees it
    raster(ax, T, "galaxy_tint", cam, alpha, to_world=mw_to_u, z=1.5)
    # relic neutrinos (vector, so that they stay sharp), and the one that absorbs the neutrino
    p_abs = u_head(T_ABS)
    rel = np.vstack([_RELICS, p_abs + 0.02 * np.array([-D[1], D[0]])])
    inview = (abs(rel[:, 0] - cam.cx) < cam.w * 0.6) & (abs(rel[:, 1] - cam.cy) < cam.h * 0.6)
    ra = alpha * (0.35 + 0.45 * window(t, 18.5, 24.5, 0.8))
    ax.scatter(*rel[inview].T, s=60, color=RELIC, alpha=0.08 * ra, lw=0, zorder=4)
    ax.scatter(*rel[inview].T, s=9, color=RELIC, alpha=0.8 * ra, lw=0, zorder=5)
    if t >= 15.0:
        h = u_head(t)
        x, y = ti.wave(S_U[0], S_U[1], h[0], h[1], 0.045, 0.30, 0.30 + 0.25 * np.hypot(*(h - S_U)) / 27.0,
                       n=max(40, int(60 * np.hypot(*(h - S_U)))))
        glow(ax, T, x, y, alpha)
        head_dot(ax, T, h, alpha, t=t)
    # the absorption: flash, the mediator, its two neutrinos at lower energy
    if T_ABS - 0.2 < t < 33.5:
        f = ramp(t, T_ABS, T_ABS + 0.4) * (1 - ramp(t, 27.4, 28.6))
        if f > 0:
            ti.burst(ax, p_abs[0], p_abs[1], 0.25 * f, "#fff2b3", z=15)
        m_end = p_abs + 0.7 * D + 0.25 * np.array([-D[1], D[0]])
        fm = ramp(t, 25.9, 26.8) * (1 - ramp(t, 31.5, 33.0))
        if fm > 0:
            pe = p_abs + fm * (m_end - p_abs)
            ax.plot(*wiggle(p_abs, pe, 0.04, 6 * max(fm, 0.1)), color=MEDIATOR, lw=2.2, alpha=alpha * min(1, 2 * fm),
                    zorder=14, transform=T)
            label(ax, T, "Mediator", pe, pe + np.array([0.3, 1.0]), alpha * window(t, 26.3, 31.5), fs=20,
                  color=MEDIATOR)
        fs_ = ramp(t, 26.8, 29.2) * (1 - ramp(t, 31.5, 33.5))
        if fs_ > 0:
            for ang in (0.32, -0.30):
                dd = np.array([D[0] * np.cos(ang) - D[1] * np.sin(ang), D[0] * np.sin(ang) + D[1] * np.cos(ang)])
                p1 = m_end + 2.6 * ramp(t, 26.8, 29.2) * dd
                xs, ys = ti.wave(m_end[0], m_end[1], p1[0], p1[1], 0.03, 0.42, 0.55, n=200)
                glow(ax, T, xs, ys, alpha * fs_, lw=1.8, color=SECONDARY, glow_color=SECONDARY, z=8)
    # the decay: a heavier neutrino emits the mediator
    if T_DEC - 0.1 < t < 40.5:
        p = u_head(T_DEC)
        f = ramp(t, T_DEC, T_DEC + 1.4) * (1 - ramp(t, 39.0, 40.5))
        nrm = np.array([-D[1], D[0]])
        pe = p + f * (0.9 * nrm + 0.9 * D)
        ax.plot(*wiggle(p, pe, 0.04, 8 * max(f, 0.1)), color=MEDIATOR, lw=2.2, alpha=alpha * min(1, 2 * f),
                zorder=14, transform=T)
        ax.scatter([p[0]], [p[1]], s=40, color="white", alpha=alpha * f, lw=0, zorder=15, transform=T)
        label(ax, T, r"$\nu_i \to \nu_j$ + mediator", pe, pe + np.array([0.2, 0.9]),
              alpha * window(t, T_DEC + 0.6, 40.0), fs=20, color=MEDIATOR)
    # labels
    q = _RELICS[np.argmin(np.hypot(*(_RELICS - (u_head(21.0) + np.array([0.9, -1.6]))).T))]
    label(ax, T, "Relic neutrinos", q, q + np.array([1.2, -1.1]), alpha * window(t, 19.3, 24.0), fs=22)
    label(ax, T, "Source", S_U + np.array([-0.06, -0.04]), S_U + np.array([-1.5, -1.0]),
          alpha * window(t, 15.0, 19.0), fs=22)
    label(ax, T, r"Emits $\nu_e : \nu_\mu : \nu_\tau = 1 : 2 : 0$", None, S_U + np.array([-1.5, -1.42]),
          alpha * window(t, 15.3, 19.0), fs=20, color=GRAY)
    return ax


def draw_galaxy_world(fig, t, cam, alpha):
    ax, T = world_axes(fig, cam)
    raster(ax, T, "galaxy_bg", cam, alpha)
    raster(ax, T, "galaxy", cam, alpha, z=1)
    raster(ax, T, "galaxy_tint", cam, alpha, z=1.5)                 # the potential of its stars and gas
    raster(ax, T, "sun", cam, alpha, to_world=sun_to_mw, z=2)       # the Sun, as its own world sees it
    start = mw_head(L_MW)
    if t >= 44.0:
        h = mw_head(x_of(t))
        glow(ax, T, [start[0], h[0]], [start[1], h[1]], alpha, lw=2.2)
        if t < 58.9:
            head_dot(ax, T, h, alpha, t=t)
    label(ax, T, "Sun", SUN_MW, SUN_MW + np.array([2.4, -3.6]), alpha * window(t, 46.5, 58.5), fs=22)
    disk = mw_project(SUN_PLANE - 12.0 * _dir_plane + 2.2 * np.array([-_dir_plane[1], _dir_plane[0]]))
    label(ax, T, "Potential of its stars and gas", disk, disk + np.array([-9.0, -6.0]),
          alpha * window(t, 47.0, 57.5), fs=22, italic=True)
    return ax


def draw_sun_world(fig, t, cam, alpha):
    ax, T = world_axes(fig, cam)
    raster(ax, T, "sun_bg", cam, alpha)
    # the Sun's potential (the Galaxy's beneath): fills the view, so it is drawn behind the Sun and the Earth,
    # which keep their colors; the neutrino's glow shows it where the neutrino is
    raster(ax, T, "sun_tint", cam, 0.7 * alpha, z=0.5)
    raster(ax, T, "sun", cam, alpha, z=1)
    raster(ax, T, "earth", cam, alpha, to_world=earth_to_sun, z=2)  # the Earth, as its own world sees it
    ax.add_patch(Circle(MOON_AU, 1737.0 * K_E, fc="#b9bcc4", ec="none", alpha=alpha, zorder=2, transform=T))
    start = END_SUN - 12.0 * D
    h = sun_head(t)
    glow(ax, T, [start[0], h[0]], [start[1], h[1]], alpha, lw=2.2)
    if t < 61.6:
        head_dot(ax, T, h, alpha, 0.8, t=t)
    label(ax, T, "Sun", (-0.10, 0.05), (-0.9, 0.4), alpha * window(t, 59.3, 62.5), fs=22)
    label(ax, T, "Earth and Moon", E_AU + np.array([0.0, -0.02]), E_AU + np.array([-0.15, -0.55]),
          alpha * window(t, 59.3, 62.5), fs=22)
    return ax


def _coast_lines():
    out = []
    for ring in land_rings():
        x, y = ring[:, 0].copy(), ring[:, 1].copy()
        x[x < 0] = np.nan
        out.append((x * R_E, y * R_E))
    return out


_COAST = None


def draw_earth_world(fig, t, cam, alpha):
    global _COAST
    ax, T = world_axes(fig, cam)
    fade = 1 - ramp(cam.w, 400.0, 60.0) if cam.w < 400.0 else 1.0     # the raster gives way to the ice and rock
    raster(ax, T, "earth_bg", cam, alpha)
    raster(ax, T, "earth_tint", cam, 0.7 * alpha * (1 - ramp(t, 67.0, 68.5)), z=0.5)   # the Sun's and the Earth's
    raster(ax, T, "earth", cam, alpha * fade, z=1)
    if cam.w > 300.0:
        if _COAST is None:
            _COAST = _coast_lines()
        for x, y in _COAST:
            ax.plot(x, y, color="#e8f0ff", lw=0.5, alpha=0.55 * alpha * fade, zorder=2, transform=T)
    # the beam: from outside, along the chord, to IceCube
    far = ENTRY - 9000.0 * D
    end = DET - 2.5 * D
    h = earth_head(t)
    glow(ax, T, [far[0], h[0]], [far[1], h[1]], alpha * (1 - 0.6 * ramp(t, 71.3, 72.2)), lw=2.4)
    if t < 71.3:
        head_dot(ax, T, h, alpha, t=t)
    label(ax, T, "IceCube", SP, SP + np.array([3600.0, -900.0]), alpha * window(t, 65.0, 67.6), fs=22)
    if cam.w < 400.0:
        icecube(ax, T, t, alpha * (1 - fade))
    return ax


_DOMS = None


def icecube(ax, T, t, alpha):
    """The South Pole at kilometer scale: ice (2.8 km), rock, IceCube's strings and light sensors (schematic: 9
    strings 125 m apart in a side view, 60 sensors each between 1.45 and 2.45 km deep), and a stylized track."""
    global _DOMS
    X0, X1 = -300.0, 300.0
    ax.add_patch(Rectangle((X0, -R_E), X1 - X0, 2.8, fc="#cfe3f2", ec="none", alpha=alpha, zorder=1, transform=T))
    ax.add_patch(Rectangle((X0, -R_E + 2.8), X1 - X0, 40.0, fc="#5d4a3b", ec="none", alpha=alpha, zorder=1,
                           transform=T))
    ax.add_patch(Rectangle((X0, -R_E - 40.0), X1 - X0, 40.0, fc="#0f1830", ec="none", alpha=alpha, zorder=1,
                           transform=T))
    ax.plot([X0, X1], [-R_E, -R_E], color="white", lw=2.0, alpha=alpha, zorder=2, transform=T)
    if _DOMS is None:
        xs = np.linspace(-0.5, 0.5, 9)
        ds = np.linspace(1.45, 2.45, 60)
        _DOMS = np.array([(x, -R_E + d) for x in xs for d in ds])
    for x in np.linspace(-0.5, 0.5, 9):
        ax.plot([x, x], [-R_E, -R_E + 2.45], color="#3a4a5c", lw=1.0, alpha=alpha, zorder=3, transform=T)
    ax.scatter(*_DOMS.T, s=6, color="#2b3440", alpha=alpha, lw=0, zorder=4, transform=T)
    # the track: from the interaction point along the neutrino's direction, lighting the sensors near it
    v0 = DET - 0.35 * D
    f = ramp(t, 71.3, 72.2)
    if f > 0:
        p1 = v0 + f * 1.9 * D
        glow(ax, T, [v0[0], p1[0]], [v0[1], p1[1]], alpha, lw=2.0, color="#fff4d6", glow_color="#ffb347", z=6)
        rel = _DOMS - v0
        along = rel @ D
        perp = np.abs(rel @ np.array([-D[1], D[0]]))
        hit = (perp < 0.10) & (along > 0) & (along < f * 1.9)
        cmap = plt.get_cmap("plasma")
        ax.scatter(*_DOMS[hit].T, s=10 + 70 * (1 - perp[hit] / 0.10), c=cmap(0.15 + 0.75 * along[hit] / 1.9),
                   alpha=alpha, lw=0, zorder=7, transform=T)
    # labels (the view is rolled by 180 degrees: +x in the world is to the left on screen, +y downwards)
    a = alpha * window(t, 70.4, 75.0)
    label(ax, T, "IceCube", None, (0.85, -R_E + 1.95), a, fs=24, color="#16324a", halo=False)
    label(ax, T, "Ice", None, (1.6, -R_E + 0.35), a, fs=22, color="#16324a", halo=False)
    label(ax, T, "South Pole", None, (1.6, -R_E - 0.17), a, fs=22)
    p = v0 + 1.6 * D
    label(ax, T, "Track", p, p + np.array([0.35, -0.25]), alpha * window(t, 72.0, 75.0), fs=22,
          color="#16324a", halo=False)


# ------------------------------------------------------------------------------------------------ overlays

def caption(fig, t):
    for t0, t1, text in CAPTIONS:
        a = window(t, t0, t1, 0.3)
        if a > 0:
            fig.text(0.5, 0.05, text, ha="center", va="center", fontsize=29, color=WHITE, alpha=a, zorder=100,
                     path_effects=HALO)
    if any(window(t, t0, t1, 0.3) > 0 for t0, t1, _ in CAPTIONS):
        fig.patches.append(Rectangle((0, 0), 1, 0.1, transform=fig.transFigure, fc=BG, ec="none", alpha=0.6,
                                     zorder=99))


STAGES = [(15.0, 45.6, "The Universe", "Gigaparsecs"), (45.6, 58.9, "The Milky Way", "Kiloparsecs"),
          (58.9, 64.4, "Near the Sun", "1 AU"), (64.4, 69.0, "Inside the Earth", "6371 km")]


def scale_title(fig, t):
    """The world's name and scale, top right, above the panel (the beam comes from the upper left)."""
    for t0, t1, title, sub in STAGES:
        a = window(t, t0, t1, 0.5)
        if a > 0:
            fig.text(0.98, 0.945, title, fontsize=32, color=WHITE, alpha=a, ha="right", va="center",
                     path_effects=HALO, zorder=90)
            fig.text(0.98, 0.895, sub, fontsize=22, color=GRAY, alpha=a, ha="right", va="center", style="italic",
                     path_effects=HALO, zorder=90)


def where_now(t):
    """Where the head is, for the panel: the bars are the flux there."""
    if t < 45.6:
        return rf"At redshift $z = {max(z_of(t), 0.0):.2f}$"
    if t < 58.9:
        return rf"At ${x_of(t):.1f}$ kpc from the Sun"
    return "Near the Sun" if t < 64.4 else "Inside the Earth"


def highlight(t):
    on = {"si": window(t, 24.0, 33.0, 0.5), "dp": window(t, 33.0, 40.0, 0.5), "le": max(window(t, 33.0, 40.0, 0.5),
                                                                                     window(t, 46.3, 58.2, 0.5))}
    return on


def panel(fig, t, alpha):
    """One bar per mediator: the flux at 495 TeV relative to leaving the sources, split into its flavors; under the
    L_e - L_mu bar, the scale of its potential (the tint's colors), with a marker at the neutrino's value."""
    if alpha <= 0:
        return
    x0, y0, w, h = 0.655, 0.15, 0.325, 0.71
    fig.patches.append(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=BG, ec="#3a4256", lw=1.2, alpha=0.82 * alpha,
                                      zorder=80))
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_zorder(85)                           # above its backing, which is a figure patch at zorder 80
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Fig. 4's energy, 495 TeV (l. 454): the solver's grid point nearest 500 TeV (trajectory_figure.E_OBS = 5e14)
    ax.text(0.5, 0.94, r"Neutrinos detected at about 500 TeV", ha="center", va="center", fontsize=26, color=WHITE,
            alpha=alpha, zorder=81)
    ax.text(0.5, 0.88, "One mediator at a time", ha="center", va="center", fontsize=21, color=GRAY, alpha=alpha,
            style="italic", zorder=81)
    on = highlight(t)
    any_on = max(on.values())
    bx0, bw, bh = 0.07, 0.86, 0.068
    for i, key in enumerate(ROWS):
        yb = 0.70 - i * 0.165
        name, col = FAM[key]
        a_row = alpha * (1.0 - 0.5 * any_on * (1 - on[key]))
        if on[key] > 0:
            ax.add_patch(FancyBboxPatch((0.025, yb - 0.03), 0.95, 0.14, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc="none", ec=col, lw=2.0, alpha=alpha * on[key], zorder=81))
        ax.text(bx0, yb + 0.08, name, ha="left", va="center", fontsize=24, color=col, alpha=a_row, zorder=82)
        ax.add_patch(Rectangle((bx0, yb - 0.01), bw, bh, fc="none", ec="#59627a", lw=1.2, alpha=a_row, zorder=82))
        F = flux(key, t) if t >= 17.0 else np.zeros(3)
        left = bx0
        for f in range(3):
            ww = bw * float(np.clip(F[f], 0, 1.2))
            ax.add_patch(Rectangle((left, yb - 0.01), ww, bh, fc=FLAV[f], ec="none", alpha=a_row, zorder=83))
            left += ww
        if key == "le":
            a_note = alpha * window(t, 33.6, 40.0, 0.4)
            if a_note > 0:
                ax.text(bx0 + bw, yb + 0.08, "Decayed near the source", ha="right", va="center", fontsize=19,
                        color=GRAY, alpha=a_note, style="italic", zorder=84)
            potential_scale(ax, t, alpha * ramp(t, 40.0, 41.0) * (1 - ramp(t, 68.0, 68.8)) * (a_row / alpha),
                            bx0, bw, yb - 0.115)
    ax.text(0.5, 0.165, where_now(t), ha="center", va="center", fontsize=22, color=WHITE, alpha=alpha, zorder=82)
    # legend: the flavors, and what the bar's length means
    for f in range(3):
        xl = 0.12 + 0.27 * f
        ax.add_patch(Rectangle((xl, 0.085), 0.045, 0.032, fc=FLAV[f], ec="none", alpha=alpha, zorder=82))
        ax.text(xl + 0.06, 0.101, FLAV_LABEL[f], ha="left", va="center", fontsize=23, color=WHITE, alpha=alpha,
                zorder=82)
    ax.text(0.5, 0.035, "Bar length: flux relative to leaving the sources", ha="center", va="center", fontsize=18,
            color=GRAY, alpha=alpha, zorder=82)


def potential_scale(ax, t, a, x0, w, y):
    """The tint's scale, log10(potential / Delta m^2_31 / 2E) from POT_LO to POT_HI, its title above, decades below,
    and a marker at the neutrino's value."""
    if a <= 0.01:
        return
    lo, hi = POT_LO, POT_HI
    g = np.linspace(lo, hi, 400)
    strip = pot_rgba(g)[None, :, :]
    strip[..., 3] = np.maximum(strip[..., 3], 0.12) * a          # below 1, a faint trace so the scale reads whole
    ax.imshow(strip, extent=(x0, x0 + w, y, y + 0.03), aspect="auto", zorder=83, interpolation="bilinear")
    ax.add_patch(Rectangle((x0, y), w, 0.03, fc="none", ec="#59627a", lw=1.0, alpha=a, zorder=84))
    ax.text(x0, y + 0.075, r"Its potential at the neutrino, over $\Delta m^2_{31}/2E$", ha="left", va="center",
            fontsize=18, color=GRAY, alpha=a, zorder=84)
    for k in range(-4, 17, 4):
        xk = x0 + w * (k - lo) / (hi - lo)
        ax.plot([xk, xk], [y - 0.008, y], color=GRAY, lw=1.0, alpha=a, zorder=84)
        ax.text(xk, y - 0.03, rf"$10^{{{k}}}$", ha="center", va="center", fontsize=16, color=GRAY, alpha=a, zorder=84)
    v = float(np.clip(pot_at_head(t), lo, hi))
    xm = x0 + w * (v - lo) / (hi - lo)
    ax.add_patch(Polygon([[xm - 0.016, y + 0.052], [xm + 0.016, y + 0.052], [xm, y + 0.031]], closed=True,
                         fc=WHITE, ec="none", alpha=a, zorder=85))


def title_card(fig, alpha, y=0.56, authors=False):
    fig.patches.append(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc=BG, ec="none", alpha=0.55 * alpha,
                                 zorder=95))
    fig.text(0.5, y + 0.05, TITLE[0], ha="center", va="center", fontsize=50, color=WHITE, alpha=alpha, zorder=96)
    fig.text(0.5, y - 0.04, TITLE[1], ha="center", va="center", fontsize=50, color=WHITE, alpha=alpha, zorder=96)
    if authors:
        fig.text(0.5, y - 0.17, AUTHORS, ha="center", va="center", fontsize=28, color=GRAY, alpha=alpha, zorder=96)


def families(fig, t):
    items = [("le", "Coupled to electrons, muons\nand their neutrinos", 8.8),
             ("dp", "Coupled through\na dark neutrino", 9.6),
             ("si", "Coupled to\nneutrinos alone", 10.4)]
    for k, (key, desc, t0) in enumerate(items):
        a = ramp(t, t0, t0 + 0.6) * (1 - ramp(t, 12.0, 13.0))
        if a <= 0:
            continue
        x = 0.22 + 0.28 * k
        name = {"le": r"$L_e - L_\mu$", "dp": "The dark portal", "si": "Secret interactions"}[key]
        fig.text(x, 0.60, name, ha="center", va="center", fontsize=38, color=FAM[key][1], alpha=a, zorder=96,
                 path_effects=HALO)
        fig.text(x, 0.49, desc, ha="center", va="center", fontsize=25, color=GRAY, alpha=a, zorder=96,
                 linespacing=1.3, path_effects=HALO)


# ------------------------------------------------------------------------------------------------ scenes 8-10

def _style_axes(ax, a):
    for s in ax.spines.values():
        s.set_color(GRAY)
        s.set_alpha(a)
    ax.tick_params(colors=GRAY, which="both", labelsize=21)
    ax.patch.set_alpha(0.0)


def _xy(fe, fm):
    return fe + 0.5 * fm, 0.5 * np.sqrt(3.0) * fm


def at_earth(fig, t):
    """Scene 8, in two beats, each with the frame to itself: the flavor composition at Earth (75-82 s), then the
    spectrum at Earth (82-89 s), as Fig. 7, for the full calculation of each example."""
    a_tri = ramp(t, 75.3, 76.2) * (1 - ramp(t, 81.6, 82.2))
    a_spec = ramp(t, 82.0, 82.8) * (1 - ramp(t, 88.6, 89.3))
    if a_tri > 0:
        triangle(fig, t, a_tri)
    if a_spec > 0:
        spectrum(fig, t, a_spec)


def triangle(fig, t, a0):
    """The flavor triangle in his ternary style (grid every 0.1, here faint; f_e along the bottom, f_mu up the right
    side), IceCube's MESE measurement at 68% and 95% C.L., one point per example and one without a new interaction."""
    fig.text(0.5, 0.935, "Flavor composition at Earth", ha="center", va="center", fontsize=32, color=WHITE, alpha=a0)
    ax = fig.add_axes([0.2, 0.12, 0.6, 0.77])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.17, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in range(1, 10):
        q = k / 10
        for p, r in ((_xy(q, 0.0), _xy(q, 1 - q)), (_xy(0.0, q), _xy(1 - q, q)), (_xy(q, 0.0), _xy(0.0, q))):
            ax.plot([p[0], r[0]], [p[1], r[1]], color="#262c3b", lw=0.9, alpha=a0, zorder=1)
    tri = np.array([_xy(0, 0), _xy(1, 0), _xy(0, 1), _xy(0, 0)])
    ax.plot(*tri.T, color=GRAY, lw=1.8, alpha=a0, zorder=3)
    for k in range(0, 11, 2):
        q = k / 10
        x, y = _xy(q, 0)
        ax.text(x, y - 0.04, f"{q:.1f}", ha="center", va="center", fontsize=18, color=GRAY, alpha=a0)
        x, y = _xy(1 - q, q)
        ax.text(x + 0.04, y, f"{q:.1f}", ha="left", va="center", fontsize=18, color=GRAY, alpha=a0)
        x, y = _xy(0, 1 - q)
        ax.text(x - 0.03, y, f"{q:.1f}", ha="right", va="center", fontsize=18, color=GRAY, alpha=a0)
    ax.text(0.5, -0.125, r"Fraction of $\nu_e$, $f_{e,\oplus}$", ha="center", va="center", fontsize=24,
            color=WHITE, alpha=a0)
    ax.text(0.75 + 0.2 * 0.866, 0.433 + 0.2 * 0.5, r"Fraction of $\nu_\mu$, $f_{\mu,\oplus}$", ha="center",
            va="center", fontsize=24, color=WHITE, alpha=a0, rotation=-60)
    ax.text(0.25 - 0.2 * 0.866, 0.433 + 0.2 * 0.5, r"Fraction of $\nu_\tau$, $f_{\tau,\oplus}$", ha="center",
            va="center", fontsize=24, color=WHITE, alpha=a0, rotation=60)
    g = DATA["earth/mese_grid"]
    FE, FM = np.meshgrid(g, g)
    X, Y = _xy(FE, FM)
    C = DATA["earth/mese_dchi2"]
    a_ic = a0 * ramp(t, 76.0, 77.0)
    ax.contourf(X, Y, C, levels=[0, 2.30], colors=["#5b8fd1"], alpha=0.85 * a_ic, zorder=2)
    ax.contourf(X, Y, C, levels=[2.30, 5.99], colors=["#5b8fd1"], alpha=0.35 * a_ic, zorder=2)
    # IceCube's key, outside the triangle (top left of the frame)
    fig.patches.append(Rectangle((0.06, 0.80), 0.022, 0.03, transform=fig.transFigure, fc="#5b8fd1", ec="none",
                                 alpha=0.85 * a_ic, zorder=90))
    fig.text(0.09, 0.815, "IceCube's measurement", fontsize=22, color="#9cc0ec", va="center", alpha=a_ic, zorder=90)
    fig.text(0.09, 0.775, r"68\% and 95\% C.L.", fontsize=19, color="#9cc0ec", va="center", alpha=a_ic, zorder=90)
    # the points; no new interaction and secret interactions share a point: one leader, two lines of label
    fl = {k: _xy(*DATA[f"earth/{k}_flavor"][:2]) for k in ("none", "si", "dp", "le")}
    a = a0 * ramp(t, 77.0, 77.5)
    if a > 0:
        ax.scatter(*fl["none"], s=420, facecolor="none", edgecolor=WHITE, lw=2.2, alpha=a, zorder=6)
        label(ax, ax.transData, "No new interaction", fl["none"], (0.30, 0.095), a, fs=22)
    a = a0 * ramp(t, 77.8, 78.3)
    if a > 0:
        ax.scatter(*fl["si"], s=150, color=FAM["si"][1], edgecolor=BG, lw=1.0, alpha=a, zorder=7)
        label(ax, ax.transData, "Secret interactions", None, (0.30, 0.045), a, fs=22, color=FAM["si"][1])
    for key, mk, t0, at in (("dp", "s", 78.6, (0.62, 0.045)), ("le", "D", 79.4, (0.80, 0.25))):
        a = a0 * ramp(t, t0, t0 + 0.5)
        if a > 0:
            ax.scatter(*fl[key], s=170, marker=mk, color=FAM[key][1], edgecolor=BG, lw=1.0, alpha=a, zorder=7)
            label(ax, ax.transData, FAM[key][0] if key == "le" else "Dark portal", fl[key], at, a, fs=22,
                  color=FAM[key][1])


def spectrum(fig, t, a0):
    """The flux at Earth versus energy, relative to no new interaction (Fig. 7, bottom), for the full calculation."""
    fig.text(0.5, 0.935, "Energy spectrum at Earth", ha="center", va="center", fontsize=32, color=WHITE, alpha=a0)
    bx = fig.add_axes([0.17, 0.2, 0.66, 0.67])
    _style_axes(bx, a0)
    E = DATA["earth/E"] / 1e12
    bx.set_xscale("log")
    bx.set_xlim(5.0, 1e4)
    bx.set_ylim(0.0, 1.18)
    bx.axhline(1.0, color=WHITE, lw=1.6, ls=(0, (5, 3)), alpha=a0, zorder=2)
    bx.text(1500.0, 1.03, "No new interaction", fontsize=21, color=WHITE, alpha=a0, va="bottom", ha="center")
    bx.set_xlabel(r"Neutrino energy, $E$ [TeV]", fontsize=25, color=WHITE, alpha=a0)
    bx.set_ylabel("Flux at Earth / without\na new interaction", fontsize=25, color=WHITE, alpha=a0)
    # labels beside their curves, at low energies where the three are apart
    for key, t0, (xl, yl) in (("le", 82.6, (7.0, 0.19)), ("dp", 83.2, (7.0, 0.40)), ("si", 83.8, (7.0, 0.88))):
        f = ramp(t, t0, t0 + 1.6)
        if f <= 0:
            continue
        r = DATA[f"earth/{key}_ratio"]
        n = max(2, int(f * len(E)))
        bx.plot(E[:n], r[:n], color=FAM[key][1], lw=3.0, alpha=a0, zorder=4)
        bx.text(xl, yl, FAM[key][0], fontsize=23, color=FAM[key][1], alpha=a0 * ramp(t, t0 + 1.0, t0 + 1.6),
                zorder=5, path_effects=HALO, va="center")
    # the dips: the resonances on the two heavier relics, 1.0 and 5.8 PeV (l. 237), found in the curve itself
    a_dip = a0 * ramp(t, 85.6, 86.3)
    if a_dip > 0:
        r = DATA["earth/si_ratio"]
        dips = [E[(E > lo) & (E < hi)][np.argmin(r[(E > lo) & (E < hi)])] for lo, hi in ((500, 2000), (3000, 9000))]
        bx.text(2600.0, 0.27, "Dips from\nabsorption", fontsize=21, color=FAM["si"][1], alpha=a_dip, ha="center",
                va="center", zorder=6, path_effects=HALO)
        for e in dips:
            bx.annotate("", xy=(e, 0.01), xytext=(2600.0, 0.19), zorder=6,
                        arrowprops=dict(arrowstyle="-", color=FAM["si"][1], lw=1.2, alpha=a_dip, shrinkA=0, shrinkB=0))


def _region(ax, lx, G, F, edge, fill, a, fill_alpha):
    ax.contourf(lx, G, F, levels=[0, max(1.0, float(np.nanmax(F))) + 1], colors=[fill], alpha=fill_alpha * a,
                zorder=2)
    ax.contour(lx, G, F, levels=[0], colors=[edge], linewidths=2.2, alpha=a, zorder=4)


LIMIT_AXES = {"le": ((-30, 9), (-32, -2), r"Coupling, $\log_{10} g'$", r"$L_e - L_\mu$"),
              "dp": ((-9, 9), (-8.6, -2), r"Coupling, $\log_{10} (g_D |U_{\tau 4}|^2)$", r"Dark portal, $\nu_\tau$"),
              "si": ((-9, 9), (-8, 0), r"Coupling, $\log_{10} g$", r"Secret interactions, scalar, $\nu_\tau$")}


def limits(fig, t, preliminary=True):
    a0 = max(ramp(t, 89.0, 89.8) * (1 - ramp(t, 94.8, 95.4)), ramp(t, 102.8, 103.4) * (1 - ramp(t, 108.8, 109.4)))
    if a0 <= 0:
        return
    a_today = a0 * (ramp(t, 89.8, 90.8) if t < 100.0 else 1.0)
    a_fut = a0 * ramp(t, 103.8, 104.8)
    a_old = a0 * ramp(t, 91.5, 92.3)
    for k, key in enumerate(("le", "dp", "si")):
        (x0, x1), (y0, y1), ylab, name = LIMIT_AXES[key]
        ax = fig.add_axes([0.06 + 0.322 * k, 0.21, 0.26, 0.62])
        _style_axes(ax, a0)
        b = DATA[f"limits/{key}/bounds"]
        ok = np.isfinite(b[:, 1])
        ax.fill_between(b[ok, 0], b[ok, 1], 5.0, color="#4a5164", alpha=0.6 * a0, lw=0, zorder=0)
        if a_fut > 0:
            d = "proj_future_planA"
            _region(ax, DATA[f"limits/{key}/{d}/lx"], DATA[f"limits/{key}/{d}/G"], DATA[f"limits/{key}/{d}/F"],
                    FUTURE[1], FUTURE[0], a_fut, 0.55)
        if a_today > 0:
            d = "now_icecube"
            _region(ax, DATA[f"limits/{key}/{d}/lx"], DATA[f"limits/{key}/{d}/G"], DATA[f"limits/{key}/{d}/F"],
                    TODAY[1], TODAY[0], a_today, 0.9)
        if key == "le" and a_old > 0:
            e = DATA["limits/le/earlier"]
            ax.plot(e[:, 0], e[:, 1], color=WHITE, lw=2.0, ls=(0, (5, 3)), alpha=a_old, zorder=6)
            ax.text(-1.0, -29.6, "Earlier estimate:\npotential at the\ndetector only", fontsize=17, color=WHITE,
                    ha="center", va="center", alpha=a_old, zorder=6, path_effects=HALO)
        # where the paper makes no claim, hatched as in its figures
        if f"limits/{key}/hatch_line" in DATA:
            slope, c = DATA[f"limits/{key}/hatch_line"]
            xs = np.linspace(x0, x1, 200)
            hx, hy = xs, slope * xs + c
        else:
            hx, hy = DATA[f"limits/{key}/hatch_curve"].T
        ax.fill_between(hx, np.maximum(hy, y0), y1 + 50.0, facecolor="none", edgecolor="#8a92a6", hatch="\\\\\\",
                        lw=0.0, alpha=a0, zorder=5.5)
        ax.plot(hx, hy, color="#8a92a6", lw=1.0, alpha=a0, zorder=5.5)
        if key == "dp":
            ax.fill_between([x0, x1], y0, -8.0, color=BG, zorder=5, alpha=a0)
            ax.axhline(-8.0, color=GRAY, lw=1.2, ls="--", alpha=a0, zorder=6)
            ax.text(x0 + 0.3, -8.12, "Smallest coupling computed", fontsize=15, color=GRAY, va="top", alpha=a0,
                    zorder=7)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_ylabel(ylab, fontsize=21, color=WHITE, alpha=a0)
        ax.set_xlabel(r"Mediator mass, $\log_{10}(M/{\rm eV})$", fontsize=21, color=WHITE, alpha=a0)
        ax.set_title(name, fontsize=25, color=WHITE, alpha=a0, pad=12)
    # one legend for the three panels, above them
    items = [("#4a5164", None, "Existing bounds", a0), (TODAY[0], TODAY[1], "Disfavored today, IceCube", a_today),
             (FUTURE[0], FUTURE[1], "Eight future telescopes", a_fut), ("none", "#8a92a6", "No claim made", a0)]
    for k, (fc, ec, text, a) in enumerate(items):
        x = 0.07 + 0.235 * k
        fig.patches.append(Rectangle((x, 0.925), 0.022, 0.03, transform=fig.transFigure, fc=fc, ec=ec or fc, lw=2,
                                     alpha=a, zorder=90, hatch="\\\\\\" if fc == "none" else None))
        fig.text(x + 0.03, 0.94, text, fontsize=21, color=WHITE, va="center", alpha=a, zorder=90)
    if preliminary:
        fig.text(0.5, 0.52, "PRELIMINARY", fontsize=110, color=WHITE, alpha=0.07 * a0, ha="center", va="center",
                 rotation=12, zorder=91)


# ------------------------------------------------------------------------------------------------ the telescopes

#: The eight telescopes of the forecast (Table I, l. 499), at the sites of their detector models: the coordinates of
#: NuFlavorGalacticPlane's analysis (its code/schematic.py SITES, as copied into its resources/video/data/inputs.json),
#: where NEON and HUNT sit at TRIDENT's site (an assumption of those models). One marker per site.
#: Their status as NuFlavorGalacticPlane's video shows it (Mauricio, 2026-11-28: "Add the same classification ... as
#: in DMGalacticPlaneNu"), from that paper's main.tex: "now running or being built (IceCube, KM3NeT, Baikal-GVD,
#: P-ONE) and planned (TRIDENT, NEON, HUNT, IceCube-Gen2)"; its two classes and colors (its SET, the paper's #88CCEE
#: and #4477AA, brightened). This paper's main.tex gives no status per telescope.
STATUS = {"running": ("Running", "#88CCEE"), "future": ("Being built or planned", "#5b8fd6")}
#: name, latitude, longitude, status, label position (globe units), leader-free side shift of the marker (globe units)
SITES = [("IceCube", -90.0, 0.0, "running", (-0.42, -1.12), -0.035),
         ("KM3NeT/ARCA", 36.2667, 16.1, "running", (-1.05, 0.95), 0.0),
         ("Baikal-GVD", 51.7667, 104.4, "running", (0.05, 1.12), 0.0),
         ("P-ONE (being built, far side)", 47.7667, -127.7667, "future", (1.42, 0.92), 0.0),
         ("TRIDENT, NEON, HUNT", 17.4, 114.0, "future", (1.38, 0.30), 0.0),
         ("IceCube-Gen2", -90.0, 0.0, "future", (0.55, -1.12), 0.035)]
GLOBE_LON0, GLOBE_LAT0 = 75.0, -12.0           # a still view that holds four of the five sites; P-ONE behind
GLOBE_C, GLOBE_R = (0.40, 0.55), 0.36          # figure fractions: center; radius as a fraction of the height


def ortho(lat, lon):
    """Orthographic projection on the unit disk, and whether the point faces the viewer."""
    la, lo = np.radians(lat), np.radians(np.asarray(lon) - GLOBE_LON0)
    la0 = np.radians(GLOBE_LAT0)
    x = np.cos(la) * np.sin(lo)
    y = np.cos(la0) * np.sin(la) - np.sin(la0) * np.cos(la) * np.cos(lo)
    vis = np.sin(la0) * np.sin(la) + np.cos(la0) * np.cos(la) * np.cos(lo) > 0
    return x, y, vis


def art_globe():
    """The Earth as a still globe (Natural Earth 1:50m land, lit from the upper left, Antarctica's ice), without a
    background."""
    from PIL import Image, ImageDraw
    n = 1400
    s = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(s, -s)
    rho = np.hypot(X, Y)
    inside = rho <= 1.0
    c = np.arcsin(np.clip(rho, 0, 1))
    la0, lo0 = np.radians(GLOBE_LAT0), np.radians(GLOBE_LON0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.degrees(np.arcsin(np.clip(np.cos(c) * np.sin(la0) + np.where(rho > 0, Y * np.sin(c) * np.cos(la0)
                                                                                 / rho, 0), -1, 1)))
        lon = np.degrees(lo0 + np.arctan2(X * np.sin(c), rho * np.cos(c) * np.cos(la0) - Y * np.sin(c) * np.sin(la0)))
    lon = (lon + 180.0) % 360.0 - 180.0
    W, H = 3600, 1800                          # the land on a 0.1-degree grid of longitude and latitude
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for ring in json.load(open(LAND50)):
        a = np.asarray(ring, dtype=float)
        draw.polygon(list(zip(((a[:, 0] + 180) * 10).tolist(), ((90 - a[:, 1]) * 10).tolist())), fill=255)
    m = np.asarray(mask) > 127
    land = m[np.clip(((90 - lat) * 10).astype(int), 0, H - 1), np.clip(((lon + 180) * 10).astype(int), 0, W - 1)]
    land &= inside
    nz = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
    L = np.array([-0.35, 0.25, 0.9]) / np.linalg.norm([-0.35, 0.25, 0.9])
    light = 0.38 + 0.7 * np.clip(L[0] * X + L[1] * Y + L[2] * nz, 0, None)
    rgb = np.where(land[..., None], np.where((lat < -60)[..., None], [0.93, 0.95, 1.0], [0.27, 0.45, 0.22]),
                   [0.12, 0.35, 0.75]) * light[..., None]
    rim = np.exp(-((rho - 1.0) / 0.012) ** 2) * (rho > 0.985)
    rgba = np.zeros((n, n, 4))
    rgba[..., :3] = np.where(inside[..., None], rgb + 0.3 * rim[..., None] * [0.42, 0.65, 1.0], [0.42, 0.65, 1.0])
    rgba[..., 3] = np.where(inside, 1.0, np.clip(0.6 * rim, 0, 1))
    return (np.clip(rgba, 0, 1) * 255).astype(np.uint8), (-1.0, 1.0, -1.0, 1.0)


ART["globe"] = art_globe
_GLOBE_COAST = None


def telescopes(fig, t):
    """Scene 10: the eight telescopes on a still globe (no rotation while they show), appearing one site at a time,
    IceCube's first."""
    global _GLOBE_COAST
    a0 = ramp(t, 95.0, 95.8) * (1 - ramp(t, 102.6, 103.2))
    if a0 <= 0:
        return
    h = GLOBE_R
    w = h * ASPECT                             # a square axes: its width as a fraction of the frame
    ax = fig.add_axes([GLOBE_C[0] - w, GLOBE_C[1] - h, 2 * w, 2 * h])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.patch.set_alpha(0.0)
    img, ext = art("globe")
    ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear", alpha=a0, zorder=1)
    if _GLOBE_COAST is None:                   # sharp coastlines over the raster, the facing side only
        _GLOBE_COAST = []
        for ring in json.load(open(LAND50)):
            a = np.asarray(ring, dtype=float)
            x, y, vis = ortho(a[:, 1], a[:, 0])
            _GLOBE_COAST.append((np.where(vis, x, np.nan), np.where(vis, y, np.nan)))
    for x, y in _GLOBE_COAST:
        ax.plot(x, y, color="#e8f0ff", lw=0.5, alpha=0.55 * a0, zorder=2)
    # the sites, in turn (the running ones first); labels beside them, leaders ending on the markers
    for k, (name, lat, lon, st, (lx, ly), shift) in enumerate(SITES):
        a = a0 * ramp(t, 96.0 + 0.6 * k, 96.5 + 0.6 * k)
        if a <= 0:
            continue
        col = STATUS[st][1]
        x, y, vis = ortho(lat, lon)
        x, y = float(x) + shift, float(y)
        if not vis:                            # on the far side: a hollow marker on the limb, behind the globe
            r = np.hypot(x, y)
            x, y = x / r, y / r
            ax.scatter([x], [y], s=170, facecolor="none", edgecolor=col, lw=2.2, alpha=a, zorder=4)
        else:
            ax.scatter([x], [y], s=170, color=col, edgecolor=BG, lw=1.6, alpha=a, zorder=4)
        label(ax, ax.transData, name, (x, y), (lx, ly), a, fs=24, ha="center")
    # the two classes, right of the globe
    la = a0 * ramp(t, 99.8, 100.6)
    from matplotlib.patches import Ellipse
    for i, (st, (name, col)) in enumerate(STATUS.items()):
        y = 0.40 - 0.065 * i
        d = 0.016                              # a round dot: its width in frame fractions is its height times H/W
        fig.patches.append(Ellipse((0.725, y), d * ASPECT, d, transform=fig.transFigure, fc=col, ec=BG, lw=1.2,
                                   alpha=la, zorder=90))
        fig.text(0.74, y, name, fontsize=24, color=WHITE, va="center", alpha=la, zorder=90)


# ------------------------------------------------------------------------------------------------ the frame

def frame(t, dpi=100):
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=dpi, facecolor=BG)
    # the worlds of the trip, with their cross-fades through the zoom boxes
    if t < 46.0:
        draw_universe(fig, t, cam_u(t), 1.0 if t < 75.0 else 0.0)
    # each zoom ends on the next world's object, drawn identically in both: only the star fields cross-fade
    a_mw = ramp(t, 45.4, 46.0) * (1 - ramp(t, 58.5, 59.0))
    if a_mw > 0:
        draw_galaxy_world(fig, t, cam_mw(t), a_mw)
    a_sun = ramp(t, 58.5, 59.0) * (1 - ramp(t, 64.0, 64.5))
    if a_sun > 0:
        draw_sun_world(fig, t, cam_sun(t), a_sun)
    a_earth = ramp(t, 64.0, 64.5) * (1 - ramp(t, 74.8, 75.3))
    if a_earth > 0:
        draw_earth_world(fig, t, cam_earth(t), a_earth)
    if t < 15.0 or t > 99.5:
        pass
    scale_title(fig, t)
    panel(fig, t, ramp(t, 17.0, 18.0) * (1 - ramp(t, 74.5, 75.2)))
    if t < 5.6:
        title_card(fig, 1.0 - ramp(t, 4.4, 5.4))
    families(fig, t)
    at_earth(fig, t)
    limits(fig, t)
    telescopes(fig, t)
    if t >= CARD_T0:
        if t > CARD_T0 + 0.4:
            draw_universe(fig, 5.0, Cam(UW / 2, UH / 2, 0.92 * UW), 1.0)
        title_card(fig, ramp(t, CARD_T0 + 0.4, CARD_T0 + 1.4), authors=True)
    caption(fig, t)
    return fig


def render_still(t, path, dpi=100):
    fig = frame(t, dpi)
    fig.savefig(path, dpi=dpi, facecolor=BG)
    plt.close(fig)


FFMPEG = os.environ.get("FFMPEG", "/snap/bin/ffmpeg")      # snap's ffmpeg cannot write to /tmp: outputs in build/


def render_frames(path, dpi, fps, k0, k1):
    """Draws frames k0..k1-1 (frame k at t = k/fps) in this process and pipes them to ffmpeg (H.264, yuv420p)."""
    w, h = int(W_IN * dpi), int(H_IN * dpi)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags",
           "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    start = time.time()
    for k in range(k0, k1):
        fig = frame(k / fps, dpi)
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())
        plt.close(fig)
        if (k - k0) % (5 * fps) == 0:
            el = time.time() - start
            print(f"  {os.path.basename(path)}: {k - k0}/{k1 - k0} frames, {el:.0f} s", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {path}")
    print(f"{path}: {k1 - k0} frames in {time.time() - start:.0f} s", flush=True)


def render_video(path, dpi, fps, t0=0.0, t1=None, jobs=1):
    """The video from t0 to t1. With jobs > 1, contiguous chunks of frames are drawn by separate processes, each into
    its own file with the same encoding, then joined without re-encoding (ffmpeg's concat demuxer)."""
    t1 = DURATION if t1 is None else t1
    n0, n1 = int(round(t0 * fps)), int(round(t1 * fps))
    if jobs <= 1:
        render_frames(path, dpi, fps, n0, n1)
        return
    start = time.time()
    parts_dir = os.path.join(BUILD, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    bounds = np.linspace(n0, n1, jobs + 1).astype(int)
    parts, procs = [], []
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        part = os.path.join(parts_dir, f"{base}_{k:02d}.mp4")
        parts.append(part)
        procs.append(subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--frames", f"{a}:{b}",
                                       "--fps", str(fps), "--dpi", str(dpi), "--out", part]))
    if any(p.wait() != 0 for p in procs):
        raise SystemExit("a chunk failed; see the log")
    listing = os.path.join(parts_dir, f"{base}.txt")
    with open(listing, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy",
                    "-movflags", "+faststart", path], check=True)
    for p in parts + [listing]:
        os.remove(p)
    print(f"{path}: {n1 - n0} frames, {jobs} processes, {time.time() - start:.0f} s", flush=True)


def setup():
    import shutil
    if shutil.which("latex") is None:      # figure_style would fall back to DejaVu: the video's text must be Palatino
        raise SystemExit("LaTeX is missing on this machine: the video's text would not be in Palatino")
    figure_style.use()
    plt.rcParams.update({"text.color": WHITE, "axes.labelcolor": WHITE, "xtick.color": GRAY, "ytick.color": GRAY,
                         "axes.edgecolor": GRAY, "savefig.facecolor": BG, "figure.facecolor": BG})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default=None, help="scene:seconds,... (e.g. universe:7,galaxy:5) or seconds")
    ap.add_argument("--storyboard", action="store_true")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--preview", action="store_true", help="960x540, 15 fps -> build/preview.mp4")
    ap.add_argument("--video", action="store_true", help="1920x1080, 30 fps -> build/video.mp4")
    ap.add_argument("--span", default=None, help="t0:t1 in seconds, for a partial render")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="processes (1 on the laptop)")
    ap.add_argument("--frames", default=None, help=argparse.SUPPRESS)      # a chunk: k0:k1, with --fps, --out
    ap.add_argument("--fps", type=int, default=30, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    verify()
    setup()
    if a.frames:
        k0, k1 = (int(x) for x in a.frames.split(":"))
        render_frames(a.out, a.dpi, a.fps, k0, k1)
        return
    if a.preview or a.video:
        t0, t1 = (float(x) for x in a.span.split(":")) if a.span else (0.0, DURATION)
        name = "preview" if a.preview else "video"
        render_video(os.path.join(BUILD, f"{name}.mp4"), 50 if a.preview else 100, 15 if a.preview else 30, t0, t1,
                     a.jobs)
        return
    out = os.path.join(BUILD, "stills")
    os.makedirs(out, exist_ok=True)
    starts = {n: t0 for n, t0, _ in SPEC}
    times = []
    if a.storyboard:
        times = STORYBOARD
    elif a.stills:
        for item in a.stills.split(","):
            if ":" in item:
                n, s = item.split(":")
                times.append(starts[n] + float(s))
            else:
                times.append(float(item))
    for t in times:
        name, local = scene_at(t)
        path = os.path.join(out, f"t{t:06.2f}_{name}.png")
        t0 = time.time()
        render_still(t, path, a.dpi)
        print(f"{path}  ({time.time() - t0:.1f} s)", flush=True)
    if a.storyboard:
        storyboard_sheet([os.path.join(out, f"t{t:06.2f}_{scene_at(t)[0]}.png") for t in times])


#: One still per beat, for his review (build/storyboard.png).
STORYBOARD = [3.0, 10.8, 16.5, 21.5, 26.2, 30.5, 36.5, 47.5, 54.0, 61.0, 66.0, 72.5, 80.5, 87.5, 93.0, 100.5,
              107.5, 112.0]


def storyboard_sheet(paths, cols=3):
    from PIL import Image
    ims = [Image.open(p) for p in paths]
    w, h = ims[0].size
    s = 0.33
    tw, th = int(w * s), int(h * s)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 12, rows * th + (rows + 1) * 12), (255, 255, 255))
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        sheet.paste(im.convert("RGB").resize((tw, th), Image.LANCZOS), (12 + c * (tw + 12), 12 + r * (th + 12)))
    path = os.path.join(BUILD, "storyboard.png")
    sheet.save(path)
    print(path)


if __name__ == "__main__":
    main()
