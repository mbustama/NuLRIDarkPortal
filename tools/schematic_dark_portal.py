"""
schematic_dark_portal.py -- the schematic of the dark-portal calculation for the paper (two columns wide).

v6 (2026-10-14): redesigned as one visual system, after the brief Mauricio approved. Flat vector icons of equal
weight -- an active galaxy, the cosmic web, a face-on spiral, the solar disc, the cut-away Earth (land outlines from
tools/data/naturalearth, as in the GeoneutrinoDecay schematic) -- on a six-column grid, each centred on the axis of the
strands. The strands are the propagation eigenstates: parallel where the evolution is adiabatic, twisted where it is
coherent; with the Galactic flux joining as a fourth, they are the only saturated colours (Okabe-Ito). The physics is
drawn on them: resonant absorption on relic neutrinos, each Z_D decaying to a nu nubar pair. One type family in three
sizes, two formulas, and one short line per stage; everything else belongs in the caption.

Earlier versions (v1-v5c, the same day) are in the history of this file.

  python tools/schematic_dark_portal.py      -> output/figures/schematic_dark_portal.pdf, .png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, RegularPolygon, Wedge  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
         "Coherent near the Sun", "Coherent in the Earth", "HESE + through-going tracks"]
Y_TITLE, Y_LINE = 27.5, 5.6
INK, MID, LIGHT, PALE = "#333333", "#808080", "#d4d4d4", "#f3f3f3"
STRANDS = ("#0072B2", "#D55E00", "#009E73")         # the propagation eigenstates (Okabe-Ito)
GOLD = "#E69F00"                                     # the Galactic flux (Okabe-Ito)
SIZE_TITLE, SIZE_FORMULA = 7.0, 6.0
SIZE = {"label": 6.0}                                # the labels' size, fitted in main() so every stage's line fits
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


# ---------------------------------------------------------------- the stages
def sources(ax):
    """An active galaxy: an inclined disc, its core, two jets; the flux leaves it as one state."""
    cx, cy = centre(0) - 1.2, Y0
    for sgn in (1.0, -1.0):                                         # jets and lobes
        ax.plot([cx, cx + 0.9 * sgn], [cy, cy + 5.2 * sgn], color=MID, lw=0.7, zorder=3, solid_capstyle="round")
        ax.add_patch(Ellipse((cx + 1.05 * sgn, cy + 6.0 * sgn), 1.5, 1.0, angle=10, fc=LIGHT, ec="none", zorder=3))
    ax.add_patch(Ellipse((cx, cy), 7.2, 2.3, angle=-8, fc=PALE, ec=MID, lw=0.5, zorder=4))
    ax.add_patch(Ellipse((cx, cy), 2.4, 1.1, angle=-8, fc="#f4e3a9", ec="none", zorder=5))
    ax.add_patch(Circle((cx, cy), 0.35, fc="#f7f1dc", ec="none", zorder=5))
    ax.plot([cx + 0.4, EDGES[1] + 1.0], [Y0, Y0], color="#8a8a8a", lw=1.8, solid_capstyle="butt", zorder=6)


def cosmology(ax):
    """The cosmic web, faint; the strands fan out into the propagation eigenstates; resonant absorption on relic
    neutrinos at several points (each energy resonates at its own redshift), each Z_D decaying to a nu nubar pair."""
    x0, x1 = EDGES[1], EDGES[2]
    rng = np.random.default_rng(7)
    n = 70
    pts = np.column_stack([rng.uniform(x0 + 0.6, x1 - 0.6, n), rng.uniform(Y0 - 8.0, Y0 + 8.0, n)])
    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    for i in range(n):                                              # each node to its nearest neighbours
        for j in np.argsort(d[i])[1:3]:
            ax.plot(pts[[i, j], 0], pts[[i, j], 1], color="#e6e6e6", lw=0.45, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=2.0, color="#d0d0d0", lw=0, zorder=1)
    xs = x0 + 1.0                                                   # the split into eigenstates
    for k, c in enumerate(STRANDS):
        ax.plot([xs, xs + 1.8], [Y0, Y0 + (k - 1) * SEP], color=c, lw=LW_STRAND, zorder=6, solid_capstyle="butt")
    parallel(ax, xs + 1.8, x1)
    w_ = x1 - x0
    for f_, k, up in ((0.357, 2, True), (0.589, 0, False), (0.821, 2, True)):
        vx = x0 + f_ * w_
        vy = Y0 + (k - 1) * SEP
        sg = 1.0 if up else -1.0
        tip = np.array([vx + 1.0, vy + sg * 2.3])                    # the Z_D, then its nu and nubar
        t = np.linspace(0.0, 1.0, 60)
        dv = tip - np.array([vx, vy])
        nrm = np.array([-dv[1], dv[0]]) / np.hypot(*dv)
        w = np.array([vx, vy])[None, :] + t[:, None] * dv[None, :] + 0.17 * np.sin(2 * np.pi * 3 * t)[:, None] * nrm
        ax.plot(w[:, 0], w[:, 1], color="#555555", lw=0.6, zorder=7)
        for ang in (35.0, 5.0):
            a = np.radians(ang)
            ax.plot([tip[0], tip[0] + 1.5 * np.cos(a)], [tip[1], tip[1] + sg * 1.5 * np.sin(a)], color="#555555",
                    lw=0.6, zorder=7, solid_capstyle="round")
        ax.scatter([vx], [vy], s=11, color="white", edgecolors="#333333", lw=0.6, zorder=8)
    text(ax, x0 + 0.339 * w_, Y0 + 3.3, r"$Z_D$", ha="right")
    text(ax, x0 + 0.268 * w_, Y0 + 6.6, r"$V\propto g_D\,\epsilon'\,u\;n_{\rm eff}$", size=SIZE_FORMULA)
    text(ax, x0 + 0.732 * w_, Y0 - 6.6, r"$E_{\rm res}\approx m_{Z_D}^2/2m_\nu$", size=SIZE_FORMULA)


def milky_way(ax):
    """A face-on barred spiral, flat: a pale disc, two tapered arms, a bulge; the Sun on an arm, on the strands' axis;
    the Galactic flux, born in the disc, joins there as a fourth strand."""
    cx, cy, Rd, q, b = centre(2) - 0.6, Y0 + 2.5, 6.4, 0.6, 0.27
    sx, sy = cx + 1.6, Y0                                            # the Sun
    r0 = 0.16 * Rd
    rs, a_s = np.hypot(sx - cx, (sy - cy) / q), np.arctan2((sy - cy) / q, sx - cx)
    phi0 = a_s - np.log(rs / r0) / b                                 # an arm through the Sun
    ax.add_patch(Ellipse((cx, cy), 2.08 * Rd, 2.08 * Rd * q, fc="#eef2f8", ec="none", zorder=2))
    th = np.linspace(0.0, np.log(Rd / r0) / b, 200)
    r = r0 * np.exp(b * th)
    width = 0.55 * (1.0 - 0.65 * th / th[-1]) + 0.12
    for arm in range(2):
        a = th + phi0 + np.pi * arm
        X, Y = r * np.cos(a), r * np.sin(a)
        dX, dY = np.gradient(X), np.gradient(Y)
        L = np.hypot(dX, dY)
        nx, ny = -dY / L, dX / L
        up = np.column_stack([cx + X + 0.5 * width * nx, cy + q * (Y + 0.5 * width * ny)])
        dn = np.column_stack([cx + X - 0.5 * width * nx, cy + q * (Y - 0.5 * width * ny)])
        ax.add_patch(Polygon(np.vstack([up, dn[::-1]]), closed=True, fc="#b8c8e0", ec="none", zorder=3))
    bar = np.degrees(np.arctan2(q * np.sin(phi0), np.cos(phi0)))
    ax.add_patch(Ellipse((cx, cy), 2.4 * r0 * np.hypot(np.cos(phi0), q * np.sin(phi0)), 0.75, angle=bar,
                         fc="#f1dfa6", ec="none", zorder=4))
    ax.add_patch(Ellipse((cx, cy), 1.5, 1.0, fc="#f1dfa6", ec="none", zorder=4))
    parallel(ax, EDGES[2], EDGES[3])
    gx, gy = cx - 1.9, cy + 1.3                                      # a Galactic emission point
    t = np.linspace(0.0, 1.0, 60)
    ctrl = np.array([sx - 0.6, gy + 0.2])
    bez = ((1 - t) ** 2)[:, None] * np.array([gx, gy]) + (2 * (1 - t) * t)[:, None] * ctrl + (t ** 2)[:, None] * np.array([sx, sy])
    ax.plot(bez[:, 0], bez[:, 1], color=GOLD, lw=LW_STRAND, zorder=7, solid_capstyle="round")
    ax.scatter([gx], [gy], s=9, color=GOLD, lw=0, zorder=8)
    ax.scatter([sx], [sy], s=16, color="#f6d36b", edgecolors="#b8902a", lw=0.5, zorder=9)


def sun(ax):
    """The solar disc, flat, above the strands, which twist: the evolution through its potential is coherent."""
    ax.add_patch(Circle((centre(3) - 0.8, Y0 + 5.4), 2.3, fc="#f6d36b", ec="#c99a2e", lw=0.5, zorder=4))
    twisted(ax, EDGES[3], EDGES[4] + 2.6)


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
    """The Earth, flat, a quarter cut away (mantle, outer and inner core), turned so that IceCube, at the South Pole on
    the right, faces the neutrinos' entry on the left; the strands converge on it through the Earth."""
    cx, cy, R = centre(4) + 0.3, Y0, 5.0
    surface = Wedge((cx, cy), R, 180.0, 450.0, fc="#dde8f2", ec="none", zorder=8)
    ax.add_patch(surface)
    for ring in globe_rings(-40.0):                                  # turned: the South Pole to the right
        land = Polygon(np.column_stack([cx - R * ring[:, 1], cy + R * ring[:, 0]]), closed=True, fc="#b5c8a0",
                       ec="none", zorder=9)
        ax.add_patch(land)
        land.set_clip_path(surface)
    for r_out, r_in, col in ((1.0, 0.546, "#efe1c2"), (0.546, 0.192, "#ddd3c0"), (0.192, 0.0, "#cbbfa9")):
        ax.add_patch(Wedge((cx, cy), r_out * R, 90.0, 180.0, width=(r_out - r_in) * R, fc=col, ec="white", lw=0.5,
                           zorder=9))
    ax.add_patch(Circle((cx, cy), R, fill=False, ec=MID, lw=0.6, zorder=10))
    px, py = cx + R, cy                                              # IceCube
    for k, c in enumerate(STRANDS):
        y_in = Y0 + (k - 1) * SEP
        x_in = cx - np.sqrt(R * R - (y_in - cy) ** 2)
        ax.plot([x_in, px], [y_in, py], color=c, lw=0.9, zorder=11, solid_capstyle="butt")
    ax.add_patch(RegularPolygon((px, py), numVertices=6, radius=0.62, orientation=np.pi / 6, fc="#3d3d3d", ec="white",
                                lw=0.5, zorder=12))


def detection(ax):
    """The three event morphologies, flat."""
    gx, lx = EDGES[5] + 4.0, EDGES[5] + 7.0
    y = Y0 + 5.0                                                     # cascade
    ax.add_patch(Circle((gx, y), 1.05, fc="#ececec", ec="#555555", lw=0.5, zorder=4))
    ax.add_patch(Circle((gx, y), 0.45, fc="#bdbdbd", ec="none", zorder=5))
    text(ax, lx, y, "Cascade", ha="left")
    y = Y0                                                           # track
    xs = np.linspace(gx - 2.0, gx + 2.0, 6)
    ax.plot([xs[0], xs[-1]], [y - 0.7, y + 0.7], color="#555555", lw=0.6, zorder=4)
    ax.scatter(xs, np.linspace(y - 0.7, y + 0.7, 6), s=5, color="#bdbdbd", edgecolors="#555555", lw=0.4, zorder=5)
    text(ax, lx, y, "Track", ha="left")
    y = Y0 - 5.0                                                     # double cascade
    for dx in (-1.25, 1.25):
        ax.add_patch(Circle((gx + dx, y), 0.6, fc="#ececec", ec="#555555", lw=0.5, zorder=5))
    ax.plot([gx - 0.65, gx + 0.65], [y, y], color="#555555", lw=0.5, ls=(0, (1, 1.2)), zorder=4)
    text(ax, lx, y, "Double cascade", ha="left")


def main():
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
    out = os.path.join(REPO, "output", "figures", "schematic_dark_portal")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf, .png")


if __name__ == "__main__":
    main()
