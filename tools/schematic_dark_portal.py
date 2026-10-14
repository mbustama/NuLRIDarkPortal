"""
schematic_dark_portal.py -- the schematic of the dark-portal calculation for the paper (two-column wide; Mauricio,
2026-10-14: all stages from the sources through the Milky Way, the Sun and the Earth to detection, alluding to the
calculation's main ingredients, especially the new ones). Layout agreed in text first: a ruler of the mediator mass
on top (what sources the potential the neutrinos feel), the journey in six stages, the ingredients per stage below
(new ones starred).

  python tools/schematic_dark_portal.py      -> output/figures/schematic_dark_portal.pdf, .png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W_IN, H_IN = 7.0, 2.9
X1, Y1 = 100.0, 100.0 * H_IN / W_IN                     # data units: equal aspect
C = dict(src="#525252", cosmo="#6a51a3", mw="#2b6cb0", sun="#d95f02", earth="#238b45", det="#525252",
         gal="#c9a227", relic="#9e9ac8", neutron="#969696", contact="#bdbdbd")
STRANDS = ("#3b5b92", "#c44e52", "#55a868")                # matter eigenstates
STAGES = [("Sources", 0.5, 12.0, C["src"]), ("Cosmological propagation", 12.0, 36.0, C["cosmo"]),
          ("Milky Way", 36.0, 57.0, C["mw"]), ("Sun", 57.0, 68.5, C["sun"]), ("Earth", 68.5, 83.5, C["earth"]),
          ("Detection", 83.5, 99.5, C["det"])]
Y0 = 21.0                                                  # the neutrinos' path
STAR = r"$\bigstar$"

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 5.2,
                     "axes.linewidth": 0.5})


def glow(ax, xy, r, color, n=12, alpha=0.25, z=1):
    """A radial gradient: concentric discs of decreasing alpha."""
    for i in range(n, 0, -1):
        ax.add_patch(Circle(xy, r * i / n, color=color, alpha=alpha / n * 1.6, lw=0, zorder=z))


def strands(ax, x0, x1, y0=Y0, sep=0.45, widths=(0.9, 0.9, 0.9), z=4, style="-"):
    for k, (c, w) in enumerate(zip(STRANDS, widths)):
        ax.plot([x0, x1], [y0 + (k - 1) * sep] * 2, color=c, lw=w, solid_capstyle="round", zorder=z, ls=style)


def text(ax, x, y, s, size=5.2, color="k", ha="center", va="center", style="normal", weight="normal", z=10, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, style=style, weight=weight, zorder=z, **kw)


def ruler(ax):
    """The mediator-mass ruler: which body sources the potential the neutrinos feel."""
    lo, hi, xa, xb, y, h = -34.0, 9.0, 11.5, 98.5, 36.6, 1.5
    X = lambda lg: xa + (lg - lo) / (hi - lo) * (xb - xa)  # noqa: E731
    segs = [(-34, -31, C["cosmo"], "Cosmos"), (-31, -26, C["mw"], "Galaxy"), (-26, -18, C["sun"], "Sun"),
            (-18, -12, C["earth"], "Earth"), (-12, 5.0, C["contact"], "contact: the local density"),
            (5.0, 9.0, C["cosmo"], "")]
    for a, b, col, lab in segs:
        ax.add_patch(FancyBboxPatch((X(a), y), X(b) - X(a), h, boxstyle="square,pad=0", fc=col, ec="white",
                                    lw=0.6, alpha=0.85 if lab else 0.35, zorder=3,
                                    hatch="////" if not lab else None))
        if lab:
            text(ax, 0.5 * (X(a) + X(b)), y + h / 2, lab, size=5.0, color="white" if col != C["contact"] else "k",
                 weight="bold")
    text(ax, X(7.0), y + h / 2, "absorption", size=4.6, color=C["cosmo"], weight="bold")
    for lg in range(-30, 7, 5):
        ax.plot([X(lg)] * 2, [y + h, y + h + 0.5], color="k", lw=0.4, zorder=3)
        text(ax, X(lg), y + h + 1.1, rf"$10^{{{lg}}}$", size=4.8)
    text(ax, 0.6, y + h + 1.1, r"mediator mass $m_{Z_D}$ [eV]", size=5.0, ha="left")
    for name, m in (("Gpc", 1.97e-7 / 3.086e25), ("Mpc", 1.97e-7 / 3.086e22), ("kpc", 1.97e-7 / 3.086e19),
                    ("pc", 1.97e-7 / 3.086e16), ("AU", 1.97e-7 / 1.496e11), (r"$R_\oplus$", 1.97e-7 / 6.371e6),
                    ("km", 1.97e-7 / 1e3)):
        lg = np.log10(m)
        ax.plot([X(lg)] * 2, [y - 0.5, y], color="k", lw=0.4, zorder=3)
        text(ax, X(lg), y - 1.05, name, size=4.8)
    text(ax, 0.6, y - 1.05, r"its range $\lambda = 1/m_{Z_D}$", size=5.0, ha="left")
    text(ax, 0.6, y + h / 2, "the potential\nfelt comes from:", size=4.6, ha="left", color="#525252")


def sources(ax):
    glow(ax, (5.4, 15.6), 3.6, C["src"], alpha=0.35)                       # the host galaxy's potential well
    for (cx, cy, ang, s) in ((4.6, 26.2, 25, 1.0), (5.4, 15.6, -15, 1.1)):
        ax.add_patch(Ellipse((cx, cy), 3.4 * s, 1.15 * s, angle=ang, fc="#d9d9d9", ec="#636363", lw=0.5, zorder=3))
        ax.add_patch(Ellipse((cx, cy), 1.1 * s, 0.6 * s, angle=ang, fc="#fff3b0", ec="none", zorder=4))
    for (sx, sy) in ((6.3, 25.4), (6.6, 16.4)):
        ax.add_patch(FancyArrowPatch((sx, sy), (11.8, Y0 + (0.6 if sy > Y0 else -0.6)), arrowstyle="-|>",
                                     mutation_scale=4, lw=0.7, color="#7f7f7f", zorder=3,
                                     connectionstyle="arc3,rad=-0.08" if sy > Y0 else "arc3,rad=0.08"))
    text(ax, 3.0, 12.0, f"host halo {STAR}", size=4.8, color=C["src"], ha="left")
    text(ax, 6.2, 29.6, r"$\Phi_\nu\propto E^{-p}$, $z\lesssim4$", size=5.0)
    text(ax, 6.2, 10.0, r"$(f_e{:}f_\mu{:}f_\tau)_{\rm S}$", size=5.2)


def cosmology(ax):
    xa, xb, ya, yb = 12.5, 35.5, 12.0, 30.0
    rng = np.random.default_rng(3)
    for k in range(9):                                     # an expanding grid: spacing grows towards today
        f = (k / 8) ** 0.75
        x = xa + f * (xb - xa)
        ax.plot([x, x], [ya + 0.5 * (1 - f) * 2, yb - 0.5 * (1 - f) * 2], color="#d9d9d9", lw=0.4, zorder=1)
    for j in range(7):
        t = j / 6
        ax.plot([xa, xb], [Y0 + (t - 0.5) * 12.0, Y0 + (t - 0.5) * 18.0], color="#d9d9d9", lw=0.4, zorder=1)
    n = 140                                                # relic neutrinos, diluting with the expansion
    u = rng.random(n) ** 0.7
    xs, ys = xa + u * (xb - xa), Y0 + (rng.random(n) - 0.5) * (12 + 6 * u)
    keep = (np.abs(ys - Y0) > 1.4) & ~((xs > 15) & (xs < 26) & (ys > 26.4)) & ~((xs > 25.2) & (ys < 15.6))
    ax.scatter(xs[keep], ys[keep], s=0.9, color=C["relic"], zorder=2, lw=0)
    m = 35
    v = rng.random(m)
    xn, yn = xa + v * (xb - xa), Y0 + (rng.random(m) - 0.5) * (12 + 6 * v)
    keep = (np.abs(yn - Y0) > 1.4) & ~((xn > 15) & (xn < 26) & (yn > 26.4)) & ~((xn > 25.2) & (yn < 15.6))
    ax.scatter(xn[keep], yn[keep], s=1.6, color=C["neutron"], zorder=2, lw=0, marker="s")
    # the path: one state at the source, matter eigenstates after; an absorption vertex thins one strand
    ax.plot([12.0, 14.2], [Y0, Y0], color="#525252", lw=2.0, solid_capstyle="round", zorder=4)
    for k, c in enumerate(STRANDS):
        ax.plot([14.2, 15.4], [Y0, Y0 + (k - 1) * 0.45], color=c, lw=0.9, zorder=4)
    strands(ax, 15.4, 27.5)
    strands(ax, 27.5, 36.0, widths=(0.9, 0.45, 0.9))
    ax.scatter([27.5], [Y0], s=26, marker=(8, 1, 0), color=C["cosmo"], zorder=6, lw=0)
    text(ax, 27.8, Y0 + 1.8, r"$\nu+\bar\nu_{\rm relic}\to Z_D$", size=4.9, color=C["cosmo"])
    for dx, dy in ((1.6, -1.6), (2.6, -1.0), (1.2, -2.4)):  # the secondaries, at lower energy
        ax.add_patch(FancyArrowPatch((27.7, Y0 - 0.3), (27.7 + dx, Y0 - 0.3 + dy), arrowstyle="-|>",
                                     mutation_scale=3, lw=0.5, color=C["cosmo"], ls=(0, (2, 1)), zorder=5))
    # inset (a): refraction
    ax.add_patch(FancyBboxPatch((14.6, 26.5), 11.4, 4.6, boxstyle="round,pad=0.25,rounding_size=0.6",
                                fc="white", ec=C["cosmo"], lw=0.5, zorder=6))
    text(ax, 20.3, 30.1, "refraction", size=5.0, color=C["cosmo"], weight="bold")
    text(ax, 20.3, 28.75, r"$V\propto g_D\,\epsilon'\,u\;n_{\rm eff}$", size=5.2)
    text(ax, 20.3, 27.4, r"$n_{\rm eff}$ smoothed over $\lambda$", size=4.6, color="#525252")
    # inset (b): resonant absorption and the cascade
    bx, by, bw, bh = 24.6, 10.2, 11.0, 5.8
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.25,rounding_size=0.6", fc="white",
                                ec=C["cosmo"], lw=0.5, zorder=6))
    lx = np.linspace(0, 1, 200)
    base = 0.95 - 0.55 * lx
    dip = 0.42 * np.exp(-0.5 * ((lx - 0.66) / 0.06) ** 2)
    bump = 0.13 * np.exp(-0.5 * ((lx - 0.38) / 0.09) ** 2)
    ax.plot(bx + 0.6 + lx * (bw - 1.2), by + 1.3 + 0.7 * (bh - 2.4) * base, color="#bdbdbd", lw=0.6, ls="--", zorder=7)
    ax.plot(bx + 0.6 + lx * (bw - 1.2), by + 1.3 + 0.7 * (bh - 2.4) * (base - dip + bump), color=C["cosmo"], lw=0.8,
            zorder=7)
    text(ax, bx + bw / 2, by + bh - 0.95, r"absorption $+$ cascade", size=4.8, color=C["cosmo"], weight="bold")
    text(ax, bx + 0.5, by + 0.3, r"$E_{\rm res}\approx m_{Z_D}^2/2m_\nu$", size=4.6, va="bottom", ha="left", z=8)


def milky_way(ax):
    cx, cy = 45.0, Y0
    for i in range(10, 0, -1):                             # the disk, denser towards the Centre
        ax.add_patch(Ellipse((cx, cy), 19.0 * i / 10, 2.4 * i / 10, fc=C["mw"], ec="none", alpha=0.07, zorder=2))
    ax.add_patch(Ellipse((cx, cy), 3.4, 2.0, fc=C["mw"], ec="none", alpha=0.25, zorder=2))
    sx, sy = 51.0, Y0 + 0.1
    for r, lab in ((0.9, "pc"), (2.4, "kpc"), (4.3, r"$\gg$Galaxy")):
        ax.add_patch(Circle((sx, sy), r, fill=False, ec=C["mw"], lw=0.45, ls=(0, (2, 1.5)), zorder=3, alpha=0.8))
        text(ax, sx + r * 0.71 + 0.2, sy + r * 0.71 + 0.3, lab, size=4.4, color=C["mw"], ha="left")
    ax.add_patch(Circle((sx, sy), 0.32, fc="#fdb813", ec="#d95f02", lw=0.4, zorder=6))
    strands(ax, 36.0, 57.0)
    text(ax, 41.8, Y0 + 3.6, f"adiabatic {STAR}", size=5.0, color=C["mw"], weight="bold")
    text(ax, 41.8, Y0 + 2.5, "populations kept", size=4.6, color="#525252")
    # Galactic neutrinos: born along a line of sight in the local eigenstates
    pts = [(40.0, Y0 - 1.2), (43.0, Y0 - 0.9), (46.5, Y0 - 0.6)]
    for (px, py) in pts:
        ax.scatter([px], [py], s=7, color=C["gal"], zorder=7, lw=0.3, edgecolors="#8c6d00", marker="*")
    gx = np.linspace(40.0, 57.0, 60)
    ax.plot(gx, Y0 - 1.2 + 0.68 * (gx - 40.0) / 17.0, color=C["gal"], lw=1.1, zorder=5)
    text(ax, 43.0, Y0 - 3.0, f"Galactic $\\nu$ born in 3-D {STAR}", size=5.0, color="#8c6d00", weight="bold")
    text(ax, 43.0, Y0 - 4.1, "gas $\\times$ cosmic rays; sources", size=4.6, color="#525252")


def sun(ax):
    sx, sy = 62.0, Y0 + 3.3
    glow(ax, (sx, sy), 4.0, "#fd8d3c", alpha=0.45)
    ax.add_patch(Circle((sx, sy), 1.25, fc="#fdb813", ec="#d95f02", lw=0.6, zorder=5))
    text(ax, sx, sy + 3.2, r"$V\propto e^{-r/\lambda}/r$", size=4.8, color=C["sun"])
    ex, ey = 67.0, Y0
    ax.add_patch(Arc((sx, sy), 2 * np.hypot(ex - sx, ey - sy), 2 * np.hypot(ex - sx, ey - sy), angle=0,
                     theta1=-75, theta2=-10, lw=0.5, color="#636363", ls=(0, (3, 2)), zorder=3))
    for ang in (-28, -8, 12):                              # paths at different angles to the Sun
        a = np.radians(ang)
        x0, y0 = ex - 9.5 * np.cos(a), ey - 9.5 * np.sin(a) + 0.0
        ax.plot([x0, ex], [y0, ey], color="#636363", lw=0.4, zorder=4, alpha=0.8)
    strands(ax, 57.0, 68.5)
    text(ax, 62.7, Y0 - 3.0, f"coherent {STAR}", size=5.0, color=C["sun"], weight="bold")
    text(ax, 62.7, Y0 - 4.1, "adiabatic $\\leftrightarrow$ sudden", size=4.6, color="#525252")


def earth(ax):
    cx, cy, R = 76.0, Y0 + 0.2, 5.0
    ax.add_patch(Circle((cx, cy), R * 1.32, fill=False, ec=C["earth"], lw=0.45, ls=(0, (2, 1.5)), zorder=2))
    for r, col in ((R, "#c7e9c0"), (0.55 * R, "#74c476"), (0.19 * R, "#238b45")):
        ax.add_patch(Circle((cx, cy), r, fc=col, ec="#31a354", lw=0.4, zorder=3))
    px, py = cx, cy - R                                    # IceCube at the South Pole (bottom)
    for ang in (100, 125, 150, 60, 35):                    # up-going chords, through the mantle and the core
        a = np.radians(ang)
        ax.plot([cx + R * np.cos(a), px], [cy + R * np.sin(a), py], color="#525252", lw=0.45, zorder=4)
    for dx in (-3.2, 3.0):                                 # down-going: short paths from outside
        ax.add_patch(FancyArrowPatch((px + dx, py - 2.4), (px, py), arrowstyle="-|>", mutation_scale=3, lw=0.45,
                                     color="#525252", zorder=4))
    ax.add_patch(Polygon([[px - 0.55, py - 0.35], [px + 0.55, py - 0.35], [px, py + 0.5]], fc="#2b6cb0",
                         ec="none", zorder=6))
    strands(ax, 68.5, cx - R * 0.98, y0=Y0 + 0.2)
    text(ax, cx - R * 0.95, cy + R * 1.18, r"exterior", size=4.4, color=C["earth"], ha="right")
    text(ax, cx, py - 3.3, r"IceCube: $\cos z=-\sin\delta$", size=4.8, color="#2b6cb0")


def detection(ax):
    y = 27.2
    # morphologies
    ax.add_patch(Circle((86.0, y), 0.8, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=5))
    ax.plot([88.6, 91.0], [y - 0.8, y + 0.8], color="#08519c", lw=0.9, zorder=5)
    for t in np.linspace(0, 1, 5):
        ax.scatter([88.6 + 2.4 * t], [y - 0.8 + 1.6 * t], s=2.0, color="#08519c", zorder=6, lw=0)
    for xx in (93.6, 96.4):
        ax.add_patch(Circle((xx, y), 0.55, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=5))
    ax.plot([94.15, 95.85], [y, y], color="#d94801", lw=0.5, ls=(0, (1, 1)), zorder=5)
    for xx, lab in ((86.0, "cascade"), (89.8, "track"), (95.0, "double\ncascade")):
        text(ax, xx, y - 2.0, lab, size=4.4, color="#525252")
    text(ax, 91.5, y + 2.3, "HESE; DR2 tracks", size=5.0, weight="bold", color=C["det"])
    # flavor triangle
    tx, ty, s = 85.0, 14.8, 6.0
    tri = np.array([[tx, ty], [tx + s, ty], [tx + s / 2, ty + s * np.sqrt(3) / 2]])
    ax.add_patch(Polygon(tri, fill=False, ec="#525252", lw=0.5, zorder=4))
    for (vx, vy), lab, off in zip(tri, (r"$\nu_\tau$", r"$\nu_\mu$", r"$\nu_e$"), ((-0.7, -0.4), (0.7, -0.4), (0, 0.6))):
        text(ax, vx + off[0], vy + off[1], lab, size=4.6)
    ccx, ccy = tx + s * 0.52, ty + s * 0.30
    for k, a in enumerate((0.55, 0.3)):
        ax.add_patch(Ellipse((ccx, ccy), 1.3 * (k + 1), 0.6 * (k + 1), angle=20, fc=C["det"], alpha=a * 0.5,
                             ec="none", zorder=5))
    ax.scatter([ccx + 0.5], [ccy + 0.25], s=5, color="#c44e52", zorder=6, lw=0)
    text(ax, tx + s / 2, ty - 1.4, "flavor", size=4.6, color="#525252")
    # sky map (Mollweide) with the Galactic plane
    mx, my, mw_, mh = 95.8, 17.6, 5.4, 2.7
    ax.add_patch(Ellipse((mx, my), mw_, mh, fc="#eff3ff", ec="#525252", lw=0.5, zorder=4))
    tt = np.linspace(-0.96, 0.96, 80)
    ax.plot(mx + tt * mw_ / 2, my + 0.5 * np.sin(np.pi * tt) * mh / 2 * 0.9, color=C["gal"], lw=2.0, alpha=0.85,
            zorder=5, solid_capstyle="round")
    text(ax, mx, my - mh / 2 - 1.0, f"sky maps {STAR}", size=4.6, color="#525252")


def ingredients(ax):
    rows = {
        "Sources": ["extragalactic,", "with evolution;", f"or in a host galaxy {STAR}"],
        "Cosmological propagation": [f"refraction $+$ absorption, one calculation {STAR}",
                                     f"thermal relic $\\nu$, Debye screening; cascade {STAR}",
                                     f"exactly unitary propagator {STAR}"],
        "Milky Way": ["$n_{\\rm eff}$: Yukawa integral of", f"the Galaxy's neutrons {STAR}",
                      "non-uniform to $4\\times10^{-31}$ eV"],
        "Sun": ["integrated", f"along paths {STAR}", "yearly: $\\lesssim10^{-3}$"],
        "Earth": ["Yukawa integral of", f"the Earth's neutrons {STAR}", "exact chords, PREM"],
        "Detection": ["IceCube responses;", "MCEq $+$ Barr;", "flavor, spectrum, direction"],
    }
    for name, xa, xb, col in STAGES:
        for i, line in enumerate(rows[name]):
            text(ax, 0.5 * (xa + xb), 7.6 - 1.3 * i, line, size=4.7, color="#252525")
    text(ax, 99.3, 1.0, f"{STAR} new in this work", size=4.8, ha="right", color="#252525")
    text(ax, 0.8, 1.0, "strands: matter eigenstates;  gold: Galactic neutrinos", size=4.8, ha="left", color="#252525")


def main():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X1)
    ax.set_ylim(0, Y1)
    ax.set_aspect("equal")
    ax.axis("off")
    for k, (name, xa, xb, col) in enumerate(STAGES):
        text(ax, 0.5 * (xa + xb), 33.0, name, size=6.0, color=col, weight="bold")
        if k:
            ax.plot([xa, xa], [9.2, 32.4], color="#d9d9d9", lw=0.5, zorder=0)
    ax.plot([0.5, 99.5], [9.0, 9.0], color="#d9d9d9", lw=0.5, zorder=0)
    ruler(ax)
    sources(ax)
    cosmology(ax)
    milky_way(ax)
    sun(ax)
    earth(ax)
    detection(ax)
    ingredients(ax)
    out = os.path.join(REPO, "output", "figures", "schematic_dark_portal")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf, .png")


if __name__ == "__main__":
    main()
