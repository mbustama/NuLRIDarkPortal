"""
schematic_dark_portal.py -- the schematic of the dark-portal calculation for the paper (two columns wide).

v8 (2026-10-14, designed with Mauricio). Six stages on one axis -- distant sources, cosmological propagation, the Milky
Way, the Sun, the Earth, detection -- with the neutrino flux ("TeV-PeV nu") as strands, the propagation eigenstates:
parallel where the evolution is adiabatic, twisted where the states can mix (the Sun, the Earth, which the bundle
crosses along its diameter to the neutrino telescope). The matter potential: the same Yukawa sphere around the
neutrino in every stage (lambda = 1/m_ZD is fixed by the mediator), the Z_D exchanged with the matter inside it, and
the potential's box ("Long-range potential from n", V ~ g_D eps' |U_a4|^2 n_eff, the active-sterile mixing U_a4).
Resonant absorption on relic neutrinos drawn and labelled in full (nubar_relic, Z_D, the nu and nubar it decays to),
with its box (the reaction, E_res); a key for relic neutrinos and neutrons. Detection as inference: the event types
(cascade, track, double cascade), and from them, together, flavour, energy and direction. Illustrations: two
galaxies with a host halo, the expanding comoving grid, the Milky Way as stars along its arms, the glowing Sun, the
cut-away Earth (land outlines from tools/data/naturalearth, as in the GeoneutrinoDecay schematic). One type family,
one line per stage. Earlier versions (v1-v7, the same day) are in the history of this file.

  python tools/schematic_dark_portal.py [--out PATH]   -> output/figures/schematic_dark_portal.pdf, .png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
LAND = os.path.join(REPO, "tools", "data", "naturalearth", "land_110m_decimated.json")

# ---------------------------------------------------------------- the system
W_IN = 7.0
X1, Y_LO, Y1 = 100.0, 4.3, 28.9                     # data units, equal aspect (100 units = 7 in)
H_IN = W_IN * (Y1 - Y_LO) / X1
Y0 = 16.6                                           # the strands' axis
SEP = 0.85                                          # the strands' spacing
EDGES = [0.5, 13.0, 37.5, 51.5, 65.5, 79.5, 99.5]   # columns: sources, cosmology (wide), Galaxy, Sun, Earth, detection
TITLES = ["Distant sources", "Cosmological propagation", "Milky Way", "Sun", "Earth", "Detection"]
LINES = [r"Power law, $z\lesssim4$", r"Refraction; absorption on relic $\nu$", r"Adiabatic; Galactic $\nu$",
         "States can mix", "States can mix", "HESE + through-going tracks"]
Y_TITLE, Y_LINE = 27.5, 5.6
INK, MID, LIGHT, PALE = "#333333", "#808080", "#d4d4d4", "#f3f3f3"
STRANDS = ("#0072B2", "#D55E00", "#009E73")         # the propagation eigenstates (Okabe-Ito)
GOLD = "#E69F00"                                     # the Galactic flux (Okabe-Ito)
SIZE_TITLE, SIZE_FORMULA = 7.0, 6.0
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
SIZE = {"label": 6.0}                                # the labels' size, fitted in main() so every stage's line fits
MEASURE = {}                                         # the renderer and the scale, set in main(), for sizing boxes to text
LW_STRAND = 1.15

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": SIZE_FORMULA,
                     "axes.linewidth": 0.5})


def centre(i):
    return 0.5 * (EDGES[i] + EDGES[i + 1])


def text(ax, x, y, s, size=None, ha="center", va="center", weight="normal", z=20, color=INK, **kw):
    size = SIZE["label"] if size is None else size
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=z, **kw)


# ---------------------------------------------------------------- strands
def parallel(ax, x0, x1, z=6):
    for k, c in enumerate(STRANDS):
        ax.plot([x0, x1], [Y0 + (k - 1) * SEP] * 2, color=c, lw=LW_STRAND, solid_capstyle="butt", zorder=z)


def twisted(ax, x0, x1, wavelength=6.0, z=6):
    """A helix seen side-on, phases 120 degrees apart (each strand starts where its parallel one ends); the half in
    front is drawn on top with a white edge, so the strands pass over and under."""
    x = np.linspace(x0, x1, 700)
    amp = SEP / np.sin(np.pi / 3.0)
    for k, c in enumerate(STRANDS):
        ph = 2.0 * np.pi * (x - x0) / wavelength + 4.0 * np.pi / 3.0 + 2.0 * np.pi * k / 3.0
        y = Y0 + amp * np.sin(ph)
        front = np.cos(ph) >= 0.0
        cuts = np.flatnonzero(np.diff(front.astype(int))) + 1
        for run in np.split(np.arange(x.size), cuts):
            s = slice(run[0], min(run[-1] + 2, x.size))
            f = bool(front[run[0]])
            ax.plot(x[s], y[s], color=c, lw=LW_STRAND, zorder=z + (0.5 if f else 0.0), solid_capstyle="butt",
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")] if f else None)


ZD = "#ae017e"                                       # the Z_D, exchanged (the potential) or resonant (absorption)
PURPLE = "#6a51a3"                                   # the cosmological stage: relic neutrinos' lines, the boxes
SPHERE_R = 4.0                                       # the Yukawa sphere: the same everywhere (lambda is fixed by m_ZD)
BEAM_SPLIT = 15.6                                    # where the beam splits into the propagation eigenstates


def wiggle(ax, p0, p1, lw=0.6, amp=0.15, waves=3.0, z=7):
    """A Z_D: a sine wave from p0 to p1, tapered at its ends so that it starts and ends on its points."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    t = np.linspace(0.0, 1.0, 160)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    env = np.sin(np.pi * t) ** 0.4
    pts = p0[None, :] + t[:, None] * d[None, :] + (amp * env * np.sin(2 * np.pi * waves * t))[:, None] * nrm[None, :]
    ax.plot(pts[:, 0], pts[:, 1], color=ZD, lw=lw, zorder=z, solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=lw + 1.3, foreground="white")])


def sphere(ax, xy, z=3):
    """The neutrino's Yukawa sphere: the matter within lambda = 1/m_ZD of it sources the potential it feels."""
    ax.add_patch(Circle(xy, SPHERE_R, fc=PURPLE, ec="none", alpha=0.06, zorder=z))
    ax.add_patch(Circle(xy, SPHERE_R, fill=False, ec="#54278f", lw=0.6, ls=(0, (3, 1.5)), zorder=z + 0.1))


def exchange(ax, p_nu, p_matter, marker=False, z=7):
    """The matter potential: a Z_D exchanged between the neutrino, on a strand, and the matter around it (a neutron:
    a small square)."""
    length = np.hypot(*(np.asarray(p_matter, dtype=float) - np.asarray(p_nu, dtype=float)))
    wiggle(ax, p_nu, p_matter, lw=0.5, amp=0.12, waves=max(2.0, length / 0.7), z=z)
    if marker:
        ax.scatter([p_matter[0]], [p_matter[1]], s=7, marker="s", color="#4d4d4d", lw=0, zorder=z + 1)


def absorption(ax, vx, vy, sg, labels=False):
    """Resonant absorption: a relic antineutrino (a background dot, its line) meets the neutrino at (vx, vy), on a
    strand; the Z_D; its nu and nubar, at lower energy, rejoin the flux. sg = +1 drawn above the strands, -1 below;
    labels names each piece."""
    relic = (vx - 1.5, vy + sg * 1.9)
    ax.plot([relic[0], vx], [relic[1], vy], color=PURPLE, lw=0.6, zorder=7)
    ax.scatter([relic[0]], [relic[1]], s=7, color="#9e9ac8", lw=0, zorder=8)
    tip = (vx + 1.1, vy + sg * 2.4)
    wiggle(ax, (vx, vy), tip, lw=0.8, amp=0.17, waves=3.0)
    ax.scatter([vx], [vy], s=26, marker=(8, 1, 0), color=ZD, lw=0, zorder=9)
    for dx, rad, lab in ((1.7, 0.3, r"$\nu$"), (4.3, 0.2, r"$\bar\nu$")):
        end = (tip[0] + dx, vy + sg * 0.78)                          # stopping short of the strands
        ax.add_patch(FancyArrowPatch(tip, end, arrowstyle="-|>", mutation_scale=3.6, lw=0.6, color="#4d4d4d",
                                     zorder=7, connectionstyle=f"arc3,rad={-rad * sg}"))
        if labels:
            text(ax, end[0] + 0.45, end[1] + sg * 0.05, lab, ha="left", size=SIZE_FORMULA, path_effects=HALO)
    if labels:
        text(ax, relic[0], relic[1] + sg * 0.85, r"$\bar\nu_{\rm relic}$", size=SIZE_FORMULA, path_effects=HALO)
        text(ax, tip[0], tip[1] + sg * 0.6, r"$Z_D$", size=SIZE_FORMULA, path_effects=HALO)


def textbox(ax, xc, yc, lines, pad=0.55, gap=1.25, z=15):
    """Lines of text (each (string, size)), centred at (xc, yc), in a white rounded box sized to them."""
    r, per_unit = MEASURE["renderer"], MEASURE["per_unit"]
    widths = []
    for s_, size in lines:
        t = ax.text(0, 0, s_, fontsize=size)
        widths.append(t.get_window_extent(r).width / per_unit)
        t.remove()
    w = max(widths) + 2 * pad
    h = gap * len(lines) + 0.5
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.55",
                                fc="white", ec=PURPLE, lw=0.5, zorder=z))
    for k, (s_, size) in enumerate(lines):
        text(ax, xc, yc + gap * (0.5 * (len(lines) - 1) - k), s_, size=size, z=z + 1)
    return w, h


def glow(ax, xy, r, color, n=14, alpha=0.3, z=1, squash=1.0):
    """A radial gradient: concentric discs (ellipses, if squashed) of decreasing opacity."""
    for i in range(n, 0, -1):
        ax.add_patch(Ellipse(xy, 2 * r * i / n, 2 * r * squash * i / n, color=color, alpha=alpha / n * 1.6, lw=0,
                             zorder=z))


# ---------------------------------------------------------------- the stages
def sources(ax):
    """Two distant galaxies, one in its host halo (a radial glow), their fluxes converging into the beam."""
    xb = 9.8                                                        # where the beam starts
    for gx, gy, ang, s, halo in ((5.6, Y0 + 5.4, 22.0, 1.0, False), (6.2, Y0 - 5.0, -15.0, 1.1, True)):
        if halo:
            glow(ax, (gx, gy), 3.5, "#6f6f6f", alpha=0.36, z=1)
        glow(ax, (gx, gy), 1.5 * s, "#fde68a", n=8, alpha=0.55, z=2)
        ax.add_patch(Ellipse((gx, gy), 3.6 * s, 1.2 * s, angle=ang, fc="#dcdcdc", ec="#636363", lw=0.5, zorder=3))
        ax.add_patch(Ellipse((gx, gy), 1.2 * s, 0.62 * s, angle=ang, fc="#fff3b0", ec="none", zorder=4))
        up = gy > Y0
        ax.add_patch(FancyArrowPatch((gx + 1.1, gy + (-0.7 if up else 0.7)), (xb, Y0 + (0.45 if up else -0.45)),
                                     arrowstyle="-|>", mutation_scale=4, lw=0.7, color="#7f7f7f", zorder=3,
                                     connectionstyle="arc3,rad=-0.12" if up else "arc3,rad=0.12"))
    ax.plot([xb, BEAM_SPLIT], [Y0, Y0], color="#7f7f7f", lw=1.8, solid_capstyle="round", zorder=6)
    text(ax, xb - 0.45, Y0, r"TeV$-$PeV $\nu$", ha="right", size=SIZE_FORMULA, path_effects=HALO)   # the beam


def cosmology(ax):
    """The expanding Universe: a comoving grid whose spacing grows towards today, relic neutrinos (purple) and neutrons
    (grey) thinning out on the way; the strands fan out into the propagation eigenstates; resonant absorption on relic
    neutrinos at several points (each energy resonates at its own redshift), each Z_D decaying to a nu nubar pair."""
    x0, x1 = EDGES[1], EDGES[2]
    xa, xb, h0 = x0 + 0.4, x1 - 0.4, 8.4
    for k in range(10):                                             # the comoving grid
        f = (k / 9) ** 0.75
        x = xa + f * (xb - xa)
        hh = h0 * (0.82 + 0.18 * f)
        ax.plot([x, x], [Y0 - hh, Y0 + hh], color="#dedede", lw=0.4, zorder=1)
    for j in range(9):
        t = j / 8 - 0.5
        ax.plot([xa, xb], [Y0 + 2 * t * h0 * 0.82, Y0 + 2 * t * h0], color="#dedede", lw=0.4, zorder=1)
    boxes = [(26.0, 36.0, Y0 + 0.9, Y0 + 4.2), (30.4, 37.5, Y0 - 3.9, Y0 - 0.9),       # the absorption chains
             (13.6, 21.0, Y0 - 8.4, Y0 - 5.8)]

    def clear(x, y):                                                # off the strands, the labels, the vertices
        ok = np.abs(y - Y0) > 1.55
        for (xa_, xb_, ya_, yb_) in boxes:
            ok &= ~((x > xa_) & (x < xb_) & (y > ya_) & (y < yb_))
        return ok
    rng = np.random.default_rng(3)
    u = rng.random(260) ** 1.3                                      # relic neutrinos, thinning towards today
    xs, ys = xa + u * (xb - xa), Y0 + (rng.random(260) - 0.5) * 2 * h0 * (0.8 + 0.18 * u)
    keep = clear(xs, ys)
    ax.scatter(xs[keep], ys[keep], s=1.3, color="#9e9ac8", lw=0, zorder=2)
    v = rng.random(50) ** 1.3                                       # and neutrons
    xn, yn = xa + v * (xb - xa), Y0 + (rng.random(50) - 0.5) * 2 * h0 * (0.8 + 0.18 * v)
    keep = clear(xn, yn)
    ax.scatter(xn[keep], yn[keep], s=2.2, color="#a3a3a3", lw=0, marker="s", zorder=2)
    for k, c in enumerate(STRANDS):                                  # the split into eigenstates
        ax.plot([BEAM_SPLIT, BEAM_SPLIT + 1.8], [Y0, Y0 + (k - 1) * SEP], color=c, lw=LW_STRAND, zorder=6,
                solid_capstyle="butt")
    parallel(ax, BEAM_SPLIT + 1.8, x1)
    yx = 22.0                                                        # the Yukawa sphere around the neutrino
    sphere(ax, (yx, Y0))
    nx, ny = yx + 0.7, Y0 + 2.6                                      # a neutron inside it, and the Z_D exchanged
    ax.scatter([nx], [ny], s=4.5, color="#8c8c8c", lw=0, marker="s", zorder=8)
    exchange(ax, (yx - 0.5, Y0 + SEP), (nx, ny))
    text(ax, yx - 0.75, Y0 + 2.25, r"$Z_D$", ha="right", size=SIZE_FORMULA, path_effects=HALO)
    textbox(ax, yx, Y0 + 6.45, [("Long-range potential from $n$", SIZE["label"]),
                                (r"$V\propto g_D\,\epsilon'\,|U_{\alpha4}|^2\,n_{\rm eff}$", SIZE_FORMULA),
                                (r"$U_{\alpha4}$: $\nu_\alpha$$-$$\nu_s$ mixing", SIZE["label"])], gap=1.2)
    absorption(ax, 28.8, Y0 + SEP, 1.0, labels=True)                # resonant absorption on relic neutrinos
    absorption(ax, 32.6, Y0 - SEP, -1.0)
    textbox(ax, 30.6, Y0 - 6.35, [(r"$\nu+\bar\nu_{\rm relic}\to Z_D\to\nu+\bar\nu$", SIZE_FORMULA),
                                  (r"$E_{\rm res}\approx m_{Z_D}^2/2m_\nu$", SIZE_FORMULA)])
    ax.scatter([14.5], [Y0 - 6.55], s=9, color="#9e9ac8", lw=0, zorder=8)          # the background's key
    text(ax, 15.1, Y0 - 6.55, r"Relic $\nu$", ha="left")
    ax.scatter([14.5], [Y0 - 7.75], s=9, color="#a3a3a3", lw=0, marker="s", zorder=8)
    text(ax, 15.1, Y0 - 7.75, "Neutron", ha="left")


def milky_way(ax):
    """A barred spiral seen nearly face-on, as stars scattered about its arms (two major, two minor, logarithmic, from
    the bar's ends) over the disc's glow, with a glowing bulge; the Sun on a major arm, on the strands' axis; Galactic
    neutrinos, born in the disc, join there."""
    cx, cy, Rd, q, b = centre(2) - 0.4, Y0 + 2.7, 6.2, 0.64, 0.25
    sx, sy = cx + 0.9, Y0                                            # the Sun
    r0 = 0.17 * Rd
    rs, a_s = np.hypot(sx - cx, (sy - cy) / q), np.arctan2((sy - cy) / q, sx - cx)
    phi0 = a_s - np.log(rs / r0) / b                                 # a major arm through the Sun
    th_max = np.log(Rd / r0) / b
    glow(ax, (cx, cy), 1.06 * Rd, "#2b6cb0", n=18, alpha=0.26, z=1, squash=q)
    rng = np.random.default_rng(11)
    for arm in range(4):
        major = arm % 2 == 0
        n = 900 if major else 450
        t = th_max * rng.random(n) ** 0.85
        r = r0 * np.exp(b * t) * (1.0 + (0.05 + 0.05 * t / th_max) * rng.standard_normal(n))
        a = t + phi0 + 0.5 * np.pi * arm + 0.05 * rng.standard_normal(n)
        rgba = np.zeros((n, 4))
        rgba[:, :3] = matplotlib.colors.to_rgb("#1f5aa6" if major else "#4a7fc1")
        rgba[:, 3] = (0.55 if major else 0.32) * (1.0 - 0.55 * t / th_max)
        ax.scatter(cx + r * np.cos(a), cy + q * r * np.sin(a), s=rng.uniform(0.15, 1.0, n), c=rgba, lw=0, zorder=2,
                   rasterized=True)
    t = th_max * (0.15 + 0.8 * rng.random(60))                       # bright young stars on the major arms
    a = t + phi0 + np.pi * (rng.random(60) < 0.5) + 0.04 * rng.standard_normal(60)
    r = r0 * np.exp(b * t)
    ax.scatter(cx + r * np.cos(a), cy + q * r * np.sin(a), s=1.5, color="white", edgecolors="#9ecae1", lw=0.2, zorder=2,
               rasterized=True)
    bar = np.degrees(np.arctan2(q * np.sin(phi0), np.cos(phi0)))
    glow(ax, (cx, cy), 0.3 * Rd, "#fdb813", n=12, alpha=0.95, z=3, squash=q)
    ax.add_patch(Ellipse((cx, cy), 2 * r0 * np.hypot(np.cos(phi0), q * np.sin(phi0)), 0.38 * r0, angle=bar,
                         fc="#fde68a", ec="none", alpha=0.95, zorder=3))
    ax.add_patch(Ellipse((cx, cy), 0.7, 0.5, fc="#fffbea", ec="none", zorder=3))
    parallel(ax, EDGES[2], EDGES[3])
    for frac, arm in ((0.38, 2), (0.66, 2), (0.9, 0)):              # Galactic neutrinos, born in the disc
        tt = frac * th_max
        aa, rr = tt + phi0 + 0.5 * np.pi * arm, r0 * np.exp(b * tt)
        gx, gy = cx + rr * np.cos(aa), cy + q * rr * np.sin(aa)
        if np.hypot(gx - sx, gy - sy) < 1.8 or gy < Y0 + 1.6:
            continue
        ax.plot([gx, sx], [gy, sy], color=GOLD, lw=0.75, zorder=7, solid_capstyle="round")
        ax.scatter([gx], [gy], s=16, color=GOLD, edgecolors="#8c6d00", lw=0.3, marker="*", zorder=8)
    ax.scatter([sx], [sy], s=18, color="#fdb813", edgecolors="#d95f02", lw=0.5, zorder=9)
    sphere(ax, (sx, Y0), z=6)                                        # the same sphere, around the neutrino
    exchange(ax, (sx - 0.8, Y0 + SEP), (sx - 2.1, Y0 + 3.3))       # the potential: the Galaxy's matter
    exchange(ax, (sx + 0.9, Y0 + SEP), (sx + 2.0, Y0 + 3.2))


def earth_geometry():
    """(cx, cy, R) of the drawn Earth; IceCube is at (cx + R, cy)."""
    return centre(4) + 0.3, Y0, 5.0


def sun(ax):
    """The Sun, glowing -- its potential falls off with distance -- above the strands, which twist: the evolution
    through its potential is coherent."""
    sx, sy = centre(3) - 0.8, Y0 + 5.3
    glow(ax, (sx, sy), 3.9, "#fd8d3c", n=16, alpha=0.5, z=2)
    ax.add_patch(Circle((sx, sy), 1.5, fc="#fdb813", ec="#d95f02", lw=0.6, zorder=4))
    ex, ey, eR = earth_geometry()
    twisted(ax, EDGES[3], ex + eR - 0.35, z=13)                       # on through the Earth to IceCube
    sphere(ax, (sx, Y0), z=5)                                        # the same sphere, around the neutrino
    for x_nu, a in ((sx - 1.1, 240.0), (sx + 1.3, 300.0)):            # the potential: the Sun's matter
        exchange(ax, (x_nu, Y0 + 1.05), (sx + 1.5 * np.cos(np.radians(a)), sy + 1.5 * np.sin(np.radians(a))),
                 marker=False)


def globe_rings(lon0_deg):
    """The land outlines on a globe seen from above the equator at longitude lon0_deg, the South Pole at the bottom
    of the limb (orthographic, in Earth radii); far-side points pushed onto the limb, rings mostly hidden dropped."""
    with open(LAND) as fh:
        rings = json.load(fh)
    lon0 = np.radians(lon0_deg)
    out = []
    for ring in rings:
        a = np.radians(np.asarray(ring, dtype=float))
        lon, lat = a[:, 0], a[:, 1]
        cosc = np.cos(lat) * np.cos(lon - lon0)
        if (cosc > 0).sum() < 3:
            continue
        x, y = np.cos(lat) * np.sin(lon - lon0), np.sin(lat)
        far = cosc <= 0
        if far.any():
            n = np.hypot(x, y)
            n[n == 0] = 1.0
            x, y = np.where(far, x / n, x), np.where(far, y / n, y)
        out.append(np.column_stack([x, y]))
    return out


def earth(ax):
    """The Earth with a quarter cut away (mantle, outer and inner core), turned so that IceCube, at the South Pole on
    the right, faces the neutrinos' entry on the left; its potential reaches beyond it (the halo). The bundle (drawn
    with the Sun's) crosses it along its diameter, still twisted -- the states can mix inside -- with no focusing; the
    Z_D exchanged with the Earth's matter."""
    cx, cy, R = earth_geometry()
    glow(ax, (cx, cy), 1.32 * R, "#238b45", n=10, alpha=0.18, z=2)
    surface = Wedge((cx, cy), R, 180.0, 450.0, fc="#cfe3f5", ec="none", zorder=8)
    ax.add_patch(surface)
    for ring in globe_rings(-40.0):                                  # turned: the South Pole to the right
        land = Polygon(np.column_stack([cx - R * ring[:, 1], cy + R * ring[:, 0]]), closed=True, fc="#8fb98a",
                       ec="#4f7a55", lw=0.2, zorder=9)
        ax.add_patch(land)
        land.set_clip_path(surface)
    for r_out, r_in, col in ((1.0, 0.546, "#f6e8c3"), (0.546, 0.192, "#d4d4d4"), (0.192, 0.0, "#ececec")):
        ax.add_patch(Wedge((cx, cy), r_out * R, 90.0, 180.0, width=(r_out - r_in) * R, fc=col, ec="white", lw=0.5,
                           zorder=9))
    ax.add_patch(Circle((cx, cy), R, fill=False, ec="#262626", lw=0.6, zorder=10))
    px, py = cx + R, cy                                              # IceCube
    ax.scatter([px], [py], s=46, marker="*", color="#2b6cb0", edgecolors="#08306b", lw=0.4, zorder=14)
    sphere(ax, (cx - 1.4, Y0), z=11.5)                                # the same sphere, around the neutrino
    for xc, end in ((cx - 2.6, (cx - 3.3, cy + 3.0)), (cx - 0.8, (cx - 1.3, cy + 2.0))):   # the Earth's matter
        exchange(ax, (xc, Y0 + 1.05), end, z=12)
    text(ax, cx - 1.3, Y0 - 7.6, "Neutrino telescope")              # IceCube, pointed out
    ax.add_patch(FancyArrowPatch((cx + 4.6, Y0 - 7.0), (px + 0.25, py - 0.55), arrowstyle="-|>", mutation_scale=5,
                                 lw=0.6, color="#4d4d4d", zorder=14, connectionstyle="arc3,rad=0.3"))


def detection(ax):
    """What is measured, and from what. On the right the event types (cascade, track, double cascade); from them,
    together, flavour, energy and direction are inferred (the left column): a flavour triangle with a measured region,
    an energy spectrum with an absorption dip, the sky with the half seen through the Earth shaded and the Galactic
    plane. Event types and flavours are not one to one, so the event types feed the column as a whole."""
    xo = EDGES[5] + 4.6                                              # the inferred quantities
    xe = EDGES[5] + 15.6                                             # the event types
    y = Y0 + 6.2                                                     # flavour
    side = 3.4
    hgt = side * np.sqrt(3.0) / 2.0
    tri = np.array([[xo, y + 2 * hgt / 3], [xo - side / 2, y - hgt / 3], [xo + side / 2, y - hgt / 3]])
    ax.add_patch(Polygon(tri, closed=True, fc="#fafafa", ec="#666666", lw=0.5, zorder=4))
    ax.add_patch(Ellipse((xo + 0.08, y - 0.05), 1.35, 0.75, angle=25, fc="#cfcfcf", ec="#525252", lw=0.4, zorder=5))
    ax.scatter([xo + 0.08], [y - 0.05], s=4, color="#252525", lw=0, zorder=6)
    text(ax, xo, y + 2 * hgt / 3 + 0.55, r"$\nu_e$")
    text(ax, xo - side / 2 - 0.6, y - hgt / 3 - 0.2, r"$\nu_\tau$")
    text(ax, xo + side / 2 + 0.6, y - hgt / 3 - 0.2, r"$\nu_\mu$")
    text(ax, xo, y - hgt / 3 - 1.5, "Flavor")
    y = Y0 + 0.2                                                     # energy
    ax.plot([xo - 3.0, xo - 3.0, xo + 3.1], [y + 1.6, y - 1.5, y - 1.5], color="#666666", lw=0.5, zorder=4)

    def spec(e):
        return 0.68 * (1.85 - 3.4 * e - 0.9 * np.exp(-0.5 * ((e - 0.62) / 0.055) ** 2)
                       + 0.25 * np.exp(-0.5 * ((e - 0.42) / 0.07) ** 2))
    e = np.linspace(0.0, 1.0, 200)
    ax.plot(xo - 2.7 + 5.6 * e, y + spec(e), color="#08519c", lw=0.8, zorder=5)
    ep = np.array([0.08, 0.24, 0.4, 0.56, 0.7, 0.86])
    fp = spec(ep) + np.array([0.07, -0.05, 0.04, -0.07, 0.035, -0.03])
    ax.errorbar(xo - 2.7 + 5.6 * ep, y + fp, yerr=0.13 + 0.09 * ep, fmt="o", ms=1.4, color="#252525", lw=0.5, zorder=6)
    text(ax, xo, y - 2.35, "Energy")
    y = Y0 - 5.6                                                     # direction
    mw_, mh = 6.2, 2.9
    sky = Ellipse((xo, y), mw_, mh, fc="#f4f6fb", ec="#666666", lw=0.5, zorder=4)
    ax.add_patch(sky)
    north = Wedge((xo, y), mw_, 0.0, 180.0, fc="#cfe0c6", ec="none", alpha=0.7, zorder=5)   # seen through the Earth
    ax.add_patch(north)
    north.set_clip_path(sky)
    tt = np.linspace(-0.97, 0.97, 80)
    ax.plot(xo + tt * mw_ / 2, y + 0.45 * np.sin(np.pi * tt) * mh / 2, color=GOLD, lw=1.4, alpha=0.9, zorder=6,
            solid_capstyle="round")
    ax.plot([xo - mw_ / 2, xo + mw_ / 2], [y, y], color="#666666", lw=0.4, ls=(0, (2, 1.5)), zorder=6)
    text(ax, xo, y - mh / 2 - 0.85, "Direction")
    # the event types
    ys = (Y0 + 5.8, Y0 + 1.2, Y0 - 3.4)
    y = ys[0]                                                        # cascade
    glow(ax, (xe, y), 1.0, "#fdae6b", n=8, alpha=0.5, z=3)
    ax.add_patch(Circle((xe, y), 0.62, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=4))
    text(ax, xe, y - 1.75, "Cascade")
    y = ys[1]                                                        # track
    xs = np.linspace(xe - 1.3, xe + 1.3, 5)
    ax.plot([xs[0], xs[-1]], [y - 0.55, y + 0.55], color="#08519c", lw=0.9, zorder=4)
    ax.scatter(xs, np.linspace(y - 0.55, y + 0.55, 5), s=3.5, color="#08519c", lw=0, zorder=5)
    text(ax, xe, y - 1.75, "Track")
    y = ys[2]                                                        # double cascade
    for dx in (-0.85, 0.85):
        ax.add_patch(Circle((xe + dx, y), 0.42, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=5))
    ax.plot([xe - 0.43, xe + 0.43], [y, y], color="#d94801", lw=0.5, ls=(0, (1, 1.0)), zorder=4)
    text(ax, xe, y - 2.05, "Double\ncascade", linespacing=0.95)
    # from the event types, together, to what is inferred: a bracket spanning them, and an arrow
    xb_, top, bot = xe - 3.3, ys[0] + 1.0, ys[2] - 2.9
    ax.plot([xb_ + 0.45, xb_, xb_, xb_ + 0.45], [top, top, bot, bot], color="#8a8a8a", lw=0.6, zorder=3,
            solid_joinstyle="miter")
    ax.add_patch(FancyArrowPatch((xb_, 0.5 * (top + bot)), (xo + 3.6, 0.5 * (top + bot)), arrowstyle="-|>",
                                 mutation_scale=5, lw=0.6,
                                 color="#8a8a8a", zorder=3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "output", "figures", "schematic_dark_portal"),
                    help="output path without extension (.pdf and .png are written)")
    a = ap.parse_args()
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X1)
    ax.set_ylim(Y_LO, Y1)
    ax.set_aspect("equal")
    ax.axis("off")
    # the labels: one size for all, the largest at which each stage's line fits its column
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    size = SIZE_FORMULA
    while size > 4.5:
        widths = []
        for line in LINES:
            t = ax.text(0, 0, line, fontsize=size)
            widths.append(t.get_window_extent(renderer).width / per_unit)
            t.remove()
        if all(w <= (EDGES[i + 1] - EDGES[i]) - 0.8 for i, w in enumerate(widths)):
            break
        size -= 0.1
    SIZE["label"] = size
    MEASURE.update(renderer=renderer, per_unit=per_unit)
    for i, (title, line) in enumerate(zip(TITLES, LINES)):
        text(ax, centre(i), Y_TITLE, title, size=SIZE_TITLE, weight="bold")
        text(ax, centre(i), Y_LINE, line)
    print(f"labels at {size:.1f} pt")
    sources(ax)
    cosmology(ax)
    milky_way(ax)
    sun(ax)
    earth(ax)
    detection(ax)
    out = a.out
    fig.savefig(out + ".pdf", dpi=600)                     # dpi: the rasterized stars of the Milky Way
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf, .png")


if __name__ == "__main__":
    main()
