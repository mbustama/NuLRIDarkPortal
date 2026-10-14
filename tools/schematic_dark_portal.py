"""
schematic_dark_portal.py -- the schematic of the dark-portal calculation for the paper (two columns wide).

v7 (2026-10-14): v6's system with the richer illustrations of v1-v5 back (Mauricio: "cleaner, but also poorer"):
the two distant galaxies with a host halo, the expanding comoving grid with relic neutrinos and neutrons, the Milky Way
as stars along its arms, the glowing Sun, the Earth's coastlines and halo, coloured detection glyphs.
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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Wedge  # noqa: E402

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


def glow(ax, xy, r, color, n=14, alpha=0.3, z=1, squash=1.0):
    """A radial gradient: concentric discs (ellipses, if squashed) of decreasing opacity."""
    for i in range(n, 0, -1):
        ax.add_patch(Ellipse(xy, 2 * r * i / n, 2 * r * squash * i / n, color=color, alpha=alpha / n * 1.6, lw=0,
                             zorder=z))


# ---------------------------------------------------------------- the stages
def sources(ax):
    """Two distant galaxies, one in its host halo (a radial glow), their fluxes converging into the beam."""
    xb = EDGES[1] - 0.6                                             # where the beam starts
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
    ax.plot([xb, EDGES[1] + 1.0], [Y0, Y0], color="#7f7f7f", lw=1.8, solid_capstyle="round", zorder=6)


def cosmology(ax):
    """The expanding Universe: a comoving grid whose spacing grows towards today, relic neutrinos (purple) and neutrons
    (grey) thinning out on the way; the strands fan out into the propagation eigenstates; resonant absorption on relic
    neutrinos at several points (each energy resonates at its own redshift), each Z_D decaying to a nu nubar pair."""
    x0, x1 = EDGES[1], EDGES[2]
    w_ = x1 - x0
    xa, xb, h0 = x0 + 0.4, x1 - 0.4, 8.4
    for k in range(10):                                             # the comoving grid
        f = (k / 9) ** 0.75
        x = xa + f * (xb - xa)
        hh = h0 * (0.82 + 0.18 * f)
        ax.plot([x, x], [Y0 - hh, Y0 + hh], color="#dedede", lw=0.4, zorder=1)
    for j in range(9):
        t = j / 8 - 0.5
        ax.plot([xa, xb], [Y0 + 2 * t * h0 * 0.82, Y0 + 2 * t * h0], color="#dedede", lw=0.4, zorder=1)
    fV, fE = (x0 + 0.268 * w_, Y0 + 6.6), (x0 + 0.732 * w_, Y0 - 6.6)

    def clear(x, y):                                                # off the strands and the formulas
        return ((np.abs(y - Y0) > 1.55) & ~((np.abs(x - fV[0]) < 6.2) & (np.abs(y - fV[1]) < 1.15))
                & ~((np.abs(x - fE[0]) < 5.6) & (np.abs(y - fE[1]) < 1.15)))
    rng = np.random.default_rng(3)
    u = rng.random(260) ** 1.3                                      # relic neutrinos, thinning towards today
    xs, ys = xa + u * (xb - xa), Y0 + (rng.random(260) - 0.5) * 2 * h0 * (0.8 + 0.18 * u)
    keep = clear(xs, ys)
    ax.scatter(xs[keep], ys[keep], s=1.3, color="#9e9ac8", lw=0, zorder=2)
    v = rng.random(50) ** 1.3                                       # and neutrons
    xn, yn = xa + v * (xb - xa), Y0 + (rng.random(50) - 0.5) * 2 * h0 * (0.8 + 0.18 * v)
    keep = clear(xn, yn)
    ax.scatter(xn[keep], yn[keep], s=2.2, color="#a3a3a3", lw=0, marker="s", zorder=2)
    xs0 = x0 + 1.0                                                  # the split into eigenstates
    for k, c in enumerate(STRANDS):
        ax.plot([xs0, xs0 + 1.8], [Y0, Y0 + (k - 1) * SEP], color=c, lw=LW_STRAND, zorder=6, solid_capstyle="butt")
    parallel(ax, xs0 + 1.8, x1)
    purple = "#6a51a3"
    for f_, k, up in ((0.357, 2, True), (0.589, 0, False), (0.821, 2, True)):
        vx, vy = x0 + f_ * w_, Y0 + (k - 1) * SEP
        sg = 1.0 if up else -1.0
        tip = np.array([vx + 1.0, vy + sg * 2.3])                    # the Z_D, then its nu and nubar
        t = np.linspace(0.0, 1.0, 60)
        dv = tip - np.array([vx, vy])
        nrm = np.array([-dv[1], dv[0]]) / np.hypot(*dv)
        wv = np.array([vx, vy])[None, :] + t[:, None] * dv[None, :] + 0.17 * np.sin(2 * np.pi * 3 * t)[:, None] * nrm
        ax.plot(wv[:, 0], wv[:, 1], color=purple, lw=0.65, zorder=7)
        for ang in (38.0, 4.0):
            a = np.radians(ang)
            ax.add_patch(FancyArrowPatch(tuple(tip), (tip[0] + 1.6 * np.cos(a), tip[1] + sg * 1.6 * np.sin(a)),
                                         arrowstyle="-|>", mutation_scale=3.2, lw=0.55, color=purple, zorder=7))
        ax.scatter([vx], [vy], s=30, marker=(8, 1, 0), color=purple, lw=0, zorder=8)
    text(ax, x0 + 0.339 * w_, Y0 + 3.3, r"$Z_D$", ha="right")
    text(ax, fV[0], fV[1], r"$V\propto g_D\,\epsilon'\,u\;n_{\rm eff}$", size=SIZE_FORMULA)
    text(ax, fE[0], fE[1], r"$E_{\rm res}\approx m_{Z_D}^2/2m_\nu$", size=SIZE_FORMULA)


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


def sun(ax):
    """The Sun, glowing -- its potential falls off with distance -- above the strands, which twist: the evolution
    through its potential is coherent."""
    sx, sy = centre(3) - 0.8, Y0 + 5.3
    glow(ax, (sx, sy), 3.9, "#fd8d3c", n=16, alpha=0.5, z=2)
    ax.add_patch(Circle((sx, sy), 1.5, fc="#fdb813", ec="#d95f02", lw=0.6, zorder=4))
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
    """The Earth with a quarter cut away (mantle, outer and inner core), turned so that IceCube, at the South Pole on
    the right, faces the neutrinos' entry on the left; its potential reaches beyond it (the halo); the strands
    converge on IceCube through the Earth."""
    cx, cy, R = centre(4) + 0.3, Y0, 5.0
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
    for k, c in enumerate(STRANDS):
        y_in = Y0 + (k - 1) * SEP
        x_in = cx - np.sqrt(R * R - (y_in - cy) ** 2)
        ax.plot([x_in, px], [y_in, py], color=c, lw=0.9, zorder=11, solid_capstyle="butt")
    ax.scatter([px], [py], s=46, marker="*", color="#2b6cb0", edgecolors="#08306b", lw=0.4, zorder=12)


def detection(ax):
    """The three event morphologies."""
    gx, lx = EDGES[5] + 4.0, EDGES[5] + 7.0
    y = Y0 + 5.0                                                     # cascade
    glow(ax, (gx, y), 1.5, "#fdae6b", n=8, alpha=0.5, z=3)
    ax.add_patch(Circle((gx, y), 0.95, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=4))
    text(ax, lx, y, "Cascade", ha="left")
    y = Y0                                                           # track
    xs = np.linspace(gx - 2.0, gx + 2.0, 6)
    ax.plot([xs[0], xs[-1]], [y - 0.7, y + 0.7], color="#08519c", lw=0.9, zorder=4)
    ax.scatter(xs, np.linspace(y - 0.7, y + 0.7, 6), s=4, color="#08519c", lw=0, zorder=5)
    text(ax, lx, y, "Track", ha="left")
    y = Y0 - 5.0                                                     # double cascade
    for dx in (-1.3, 1.3):
        ax.add_patch(Circle((gx + dx, y), 0.62, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=5))
    ax.plot([gx - 0.68, gx + 0.68], [y, y], color="#d94801", lw=0.5, ls=(0, (1, 1.1)), zorder=4)
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
    fig.savefig(out + ".pdf", dpi=600)                     # dpi: the rasterized stars of the Milky Way
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf, .png")


if __name__ == "__main__":
    main()
