"""
schematic_dark_portal.py -- the schematic of the dark-portal calculation for the paper (two-column wide; Mauricio,
2026-10-14: all stages from the sources through the Milky Way, the Sun and the Earth to detection, alluding to the
calculation's main ingredients). Layout agreed in text first: a ruler of the mediator mass on top (what sources the
potential the neutrinos feel), the journey in six stages, the ingredients per stage below.

v5b (2026-10-14, Mauricio: "too much text at the top"): no key strip; n_eff is drawn instead, a pictogram in the
refraction box (the neutrons within lambda = 1/m_ZD of the neutrino); fewer labels in the drawing.
v5 (2026-10-14): the range lambda = 1/m_ZD is fixed by the mediator and the same all along the way (Mauricio), so the
stages carry no spheres: each has a dimension line for its distance (1 Gpc along the cosmological path, 8 kpc from the
Sun to the Galactic Centre, 1 AU from the Sun to the Earth, the Earth's radius), and one key strip on top shows a
neutrino carrying its sphere of radius lambda and which matter that sphere encloses for each range of lambda.
v4 (2026-10-14): the mass ruler dropped -- it relates the range to the mass, not to where the neutrinos are -- and
in its place, in each stage, the neutrino's Yukawa sphere: the matter within lambda = 1/m_ZD of it, which sources the
potential it feels, sized to the stage (1 Gpc, 8 kpc to the Galactic Centre, 1 AU, the Earth's radius) and labelled
with lambda and m_ZD -- the stages' distance markers; resonant absorption at several points of the cosmological path,
each Z_D decaying to a nu nubar pair (the relic column elsewhere is ~1e-5 of the cosmological one); 'Distant sources'.
v3 (2026-10-14): the ruler aligned with the stages (a piecewise scale, values at the dividers), absorption in the
cosmological inset; no bottom legend; the Earth turned so that IceCube faces the neutrinos' entry; plainer labels.
v2, after Mauricio's edits (2026-10-14): every phrase capitalized; no marks for what is new; the Earth as a globe
with a quarter cut away, in the style of the GeoneutrinoDecay schematic (its land outlines, tools/data/naturalearth);
the Sun drawn as what it is in the calculation, a point source whose potential falls as exp(-r/lambda)/r; a barred
spiral for the Milky Way, the Sun on an arm; the strands (propagation eigenstates) further apart, twisted where the
evolution is coherent; larger labels on the ruler.

  python tools/schematic_dark_portal.py      -> output/figures/schematic_dark_portal.pdf, .png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon,  # noqa: E402
                                Wedge)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAND = os.path.join(REPO, "tools", "data", "naturalearth", "land_110m_decimated.json")
W_IN = 7.0
X1, Y_LO, Y1 = 100.0, 3.9, 34.3                         # data units (equal aspect)
H_IN = W_IN * (Y1 - Y_LO) / X1
C = dict(src="#525252", cosmo="#6a51a3", mw="#2b6cb0", sun="#d95f02", earth="#238b45", det="#525252",
         gal="#c9a227", relic="#9e9ac8", neutron="#969696", contact="#bdbdbd")
STRANDS = ("#3b5b92", "#c44e52", "#55a868")                # the propagation eigenstates
SEP = 0.8                                                  # their separation (v1: 0.45)
STAGES = [("Distant sources", 0.5, 12.0, C["src"]), ("Cosmological propagation", 12.0, 36.0, C["cosmo"]),
          ("Milky Way", 36.0, 57.0, C["mw"]), ("Sun", 57.0, 68.5, C["sun"]), ("Earth", 68.5, 83.5, C["earth"]),
          ("Detection", 83.5, 99.5, C["det"])]
Y0 = 21.0                                                  # the neutrinos' path
INK = "0.15"
HALO = [pe.withStroke(linewidth=1.3, foreground="white")]

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 5.2,
                     "axes.linewidth": 0.5})


def glow(ax, xy, r, color, n=12, alpha=0.25, z=1, squash=1.0):
    """A radial gradient: concentric discs (ellipses, if squashed) of decreasing alpha."""
    for i in range(n, 0, -1):
        ax.add_patch(Ellipse(xy, 2 * r * i / n, 2 * r * squash * i / n, color=color, alpha=alpha / n * 1.6, lw=0,
                             zorder=z))


def dimension(ax, p0, p1, label, color, side=1.0, off=0.75, z=9, size=5.2, box=False, t=0.5):
    """A dimension line from p0 to p1 with end ticks, labelled at the fraction t along it, off it on `side` (+1 left
    of p0->p1)."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    d = p1 - p0
    u = d / np.hypot(*d)
    nrm = np.array([-u[1], u[0]])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=0.7, zorder=z, solid_capstyle="butt")
    for q in (p0, p1):
        ax.plot([q[0] - 0.35 * nrm[0], q[0] + 0.35 * nrm[0]], [q[1] - 0.35 * nrm[1], q[1] + 0.35 * nrm[1]],
                color=color, lw=0.7, zorder=z)
    m = p0 + t * d + side * off * nrm
    kw = dict(bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)) if box else dict(path_effects=HALO)
    text(ax, m[0], m[1], label, size=size, color=color, weight="bold", z=z + 1, **kw)


def wavy(ax, p0, p1, color, n=3, amp=0.16, lw=0.6, z=6):
    """A mediator's propagator: a sine wave from p0 to p1."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    t = np.linspace(0.0, 1.0, 80)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    pts = p0[None, :] + t[:, None] * d[None, :] + amp * np.sin(2 * np.pi * n * t)[:, None] * nrm[None, :]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=z)


def strands(ax, x0, x1, y0=Y0, widths=(0.9, 0.9, 0.9), z=4):
    """Parallel strands, the propagation eigenstates: their populations kept (adiabatic, or decohered)."""
    for k, (c, w) in enumerate(zip(STRANDS, widths)):
        ax.plot([x0, x1], [y0 + (k - 1) * SEP] * 2, color=c, lw=w, solid_capstyle="round", zorder=z)


def twist(ax, x0, x1, y0=Y0, wavelength=7.5, z=4):
    """Twisted strands: coherent evolution, in which the eigenstates' populations change. A helix seen side-on,
    phases 120 degrees apart so that each strand starts where its parallel one ends; the half in front is drawn on
    top, with a white edge, so the strands cross over and under."""
    x = np.linspace(x0, x1, 600)
    amp = SEP / np.sin(np.pi / 3.0)
    for k, c in enumerate(STRANDS):
        ph = 2.0 * np.pi * (x - x0) / wavelength + 4.0 * np.pi / 3.0 + 2.0 * np.pi * k / 3.0
        y = y0 + amp * np.sin(ph)
        front = np.cos(ph) >= 0.0
        cuts = np.flatnonzero(np.diff(front.astype(int))) + 1
        for run in np.split(np.arange(x.size), cuts):
            s = slice(run[0], min(run[-1] + 2, x.size))
            f = bool(front[run[0]])
            ax.plot(x[s], y[s], color=c, lw=0.9, zorder=z + (0.5 if f else 0.0), solid_capstyle="butt",
                    path_effects=[pe.withStroke(linewidth=2.1, foreground="white")] if f else None)


def text(ax, x, y, s, size=5.2, color="k", ha="center", va="center", style="normal", weight="normal", z=10, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, style=style, weight=weight, zorder=z, **kw)


def curved_text(ax, s, center, radius, theta_deg, size, color):
    """Text along an arc about `center`, one character at a time, each turned to the arc's tangent (as in
    GeoneutrinoDecay's schematic)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def width(t):
        a = ax.text(0, 0, t, fontsize=size)
        w = a.get_window_extent(renderer).width
        a.remove()
        return w

    space = width("a b") - width("ab")
    widths = np.array([space if ch == " " else width(ch) for ch in s])
    per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    angles = widths / per_unit / radius
    a = np.radians(theta_deg) + angles.sum() / 2.0
    for ch, da in zip(s, angles):
        mid = a - da / 2.0
        ax.text(center[0] + radius * np.cos(mid), center[1] + radius * np.sin(mid), ch, fontsize=size, color=color,
                ha="center", va="center", rotation=np.degrees(mid) - 90.0, rotation_mode="anchor", zorder=9)
        a -= da


def globe_rings(lon0_deg):
    """The land outlines on a globe seen from above the equator at longitude lon0_deg, the South Pole at the bottom
    of the limb (an orthographic projection, in units of the Earth's radius). Points on the far side are pushed onto
    the limb, so that a ring crossing it closes along it; rings with fewer than three points in view are dropped."""
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


def sources(ax):
    glow(ax, (5.4, 15.6), 3.6, C["src"], alpha=0.35)                       # the host galaxy's potential well
    for (cx, cy, ang, s) in ((4.6, 26.2, 25, 1.0), (5.4, 15.6, -15, 1.1)):
        ax.add_patch(Ellipse((cx, cy), 3.4 * s, 1.15 * s, angle=ang, fc="#d9d9d9", ec="#636363", lw=0.5, zorder=3))
        ax.add_patch(Ellipse((cx, cy), 1.1 * s, 0.6 * s, angle=ang, fc="#fff3b0", ec="none", zorder=4))
    for (sx, sy) in ((6.3, 25.4), (6.6, 16.4)):
        ax.add_patch(FancyArrowPatch((sx, sy), (11.8, Y0 + (0.6 if sy > Y0 else -0.6)), arrowstyle="-|>",
                                     mutation_scale=4, lw=0.7, color="#7f7f7f", zorder=3,
                                     connectionstyle="arc3,rad=-0.08" if sy > Y0 else "arc3,rad=0.08"))
    text(ax, 3.0, 12.0, "Host halo", size=4.8, color=C["src"], ha="left")
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

    def clear(x, y):                                       # off the strands, the insets and the labels
        return ((np.abs(y - Y0) > 1.7) & ~((x > 14.3) & (x < 26.3) & (y > 26.2)) & ~((x > 22.6) & (y < 16.3))
                & ~((x > 13.6) & (x < 23.6) & (y > 15.7) & (y < 18.0)) & ~((x > 25.6) & (y > 24.4) & (y < 27.6)))
    u = rng.random(140) ** 0.7                             # relic neutrinos, diluting with the expansion
    xs, ys = xa + u * (xb - xa), Y0 + (rng.random(140) - 0.5) * (12 + 6 * u)
    keep = clear(xs, ys)
    ax.scatter(xs[keep], ys[keep], s=0.9, color=C["relic"], zorder=2, lw=0)
    v = rng.random(35)                                     # and neutrons
    xn, yn = xa + v * (xb - xa), Y0 + (rng.random(35) - 0.5) * (12 + 6 * v)
    keep = clear(xn, yn)
    ax.scatter(xn[keep], yn[keep], s=1.6, color=C["neutron"], zorder=2, lw=0, marker="s")
    # the path: one state at the source, propagation eigenstates after
    ax.plot([12.0, 14.2], [Y0, Y0], color="#525252", lw=2.0, solid_capstyle="round", zorder=4)
    for kk, c in enumerate(STRANDS):
        ax.plot([14.2, 15.8], [Y0, Y0 + (kk - 1) * SEP], color=c, lw=0.9, zorder=4)
    strands(ax, 15.8, 36.0)
    dimension(ax, (14.2, 17.4), (23.0, 17.4), "1 Gpc", C["cosmo"], side=-1.0, off=0.85)
    # resonant absorption on relic neutrinos, at several points: each energy resonates at its own redshift
    for (vx, k, up) in ((24.8, 2, True), (29.0, 0, False), (33.2, 2, True)):
        vy = Y0 + (k - 1) * SEP
        sgn = 1.0 if up else -1.0
        dx, dy = vx + 1.0, vy + sgn * 1.55
        wavy(ax, (vx, vy), (dx, dy), C["cosmo"])
        ax.scatter([vx], [vy], s=22, marker=(8, 1, 0), color=C["cosmo"], zorder=7, lw=0)
        ax.scatter([dx], [dy], s=6, color=C["cosmo"], zorder=7, lw=0)
        for ang in (25.0, -10.0):                          # the Z_D's nu and nubar, at lower energy
            a = np.radians(ang) * sgn
            ax.add_patch(FancyArrowPatch((dx, dy), (dx + 1.25 * np.cos(a), dy + sgn * 0.35 + 1.25 * np.sin(a)),
                                         arrowstyle="-|>", mutation_scale=3, lw=0.5, color=C["cosmo"], zorder=6))
    text(ax, 24.45, Y0 + 3.15, r"$Z_D$", size=4.8, color=C["cosmo"], ha="right")
    ax.add_patch(FancyBboxPatch((26.85, 24.75), 8.8, 2.5, boxstyle="round,pad=0.2,rounding_size=0.5", fc="white",
                                ec=C["cosmo"], lw=0.5, zorder=6))
    text(ax, 31.25, 26.0, r"$\nu+\bar\nu_{\rm relic}\to Z_D\to\nu\,\bar\nu$", size=5.5, color=C["cosmo"], z=8)
    # inset (a): refraction
    ax.add_patch(FancyBboxPatch((14.6, 26.5), 11.4, 4.6, boxstyle="round,pad=0.25,rounding_size=0.6",
                                fc="white", ec=C["cosmo"], lw=0.5, zorder=6))
    text(ax, 15.1, 30.25, "Refraction", size=5.0, color=C["cosmo"], weight="bold", ha="left")
    text(ax, 15.1, 28.3, r"$V\propto g_D\,\epsilon'\,u\;n_{\rm eff}$", size=5.0, ha="left")
    # n_eff, as a picture: the neutrons within lambda of the neutrino, the same sphere wherever it goes
    px, py, pr = 24.15, 29.0, 1.6
    ax.add_patch(Circle((px, py), pr, fc=C["relic"], ec="none", alpha=0.18, zorder=7))
    ax.add_patch(Circle((px, py), pr, fill=False, ec=C["cosmo"], lw=0.6, ls=(0, (3, 1.5)), zorder=7))
    for (qx, qy, inside) in ((-0.8, 0.6, 1), (0.55, 0.85, 1), (0.9, -0.5, 1), (-0.4, -0.85, 1), (-1.95, 1.1, 0),
                             (1.85, -1.15, 0)):
        ax.scatter([px + qx], [py + qy], s=2.4, marker="s", lw=0, zorder=8,
                   color=C["cosmo"] if inside else C["neutron"])
    a = np.radians(38.0)
    ax.plot([px, px + pr * np.cos(a)], [py, py + pr * np.sin(a)], color=C["cosmo"], lw=0.5, zorder=8)
    text(ax, px + 0.5 * pr * np.cos(a) - 0.05, py + 0.5 * pr * np.sin(a) + 0.42, r"$\lambda$", size=4.8,
         color=C["cosmo"], z=9)
    ax.scatter([px], [py], s=7, color="white", edgecolors="k", lw=0.45, zorder=9)
    text(ax, px, py - pr - 0.42, r"$\lambda=1/m_{Z_D}$", size=4.6, color="#404040", z=9)
    # inset (b): resonant absorption and the cascade
    bx, by, bw, bh = 22.9, 10.2, 12.8, 5.8                 # v3: wider
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.25,rounding_size=0.6", fc="white",
                                ec=C["cosmo"], lw=0.5, zorder=6))
    lx = np.linspace(0, 1, 200)
    base = 0.95 - 0.55 * lx
    dip = 0.42 * np.exp(-0.5 * ((lx - 0.66) / 0.06) ** 2)
    bump = 0.13 * np.exp(-0.5 * ((lx - 0.38) / 0.09) ** 2)
    ax.plot(bx + 0.6 + lx * (bw - 1.2), by + 1.3 + 0.7 * (bh - 2.4) * base, color="#bdbdbd", lw=0.6, ls="--", zorder=7)
    ax.plot(bx + 0.6 + lx * (bw - 1.2), by + 1.3 + 0.7 * (bh - 2.4) * (base - dip + bump), color=C["cosmo"], lw=0.8,
            zorder=7)
    text(ax, bx + bw / 2, by + bh - 0.95, r"Absorption $+$ cascade", size=4.8, color=C["cosmo"], weight="bold")
    text(ax, bx + 0.5, by + 0.3, r"$E_{\rm res}\approx m_{Z_D}^2/2m_\nu$", size=4.6, va="bottom", ha="left", z=8)
    text(ax, bx + bw - 0.5, by + 0.3, r"$m_{Z_D}\sim$ MeV", size=4.6, va="bottom", ha="right", z=8, color="#525252")


def milky_way(ax):
    """A barred spiral seen nearly face-on, drawn as stars scattered about its arms (two major, two minor, logarithmic,
    from the bar's ends) over the disk's glow, with a bulge; the Sun on a major arm about 0.6 of the way out, on the
    strands; Yukawa ranges drawn about it; Galactic neutrinos born in the disk, reaching it."""
    cx, cy, Rd, q, b = 46.5, Y0 + 2.75, 7.0, 0.66, 0.25     # centre, disk radius, axis ratio, arms' log pitch
    sx, sy = cx + 0.9, Y0                                   # the Sun
    r0 = 0.17 * Rd                                          # the bar's half-length, where the arms start
    rs, a_s = np.hypot(sx - cx, (sy - cy) / q), np.arctan2((sy - cy) / q, sx - cx)
    phi0 = a_s - np.log(rs / r0) / b                        # so that a major arm passes through the Sun
    th_max = np.log(Rd / r0) / b
    glow(ax, (cx, cy), 1.06 * Rd, C["mw"], n=18, alpha=0.26, z=1, squash=q)
    rng = np.random.default_rng(11)
    for arm in range(4):
        major = arm % 2 == 0
        n = 900 if major else 450
        t = th_max * rng.random(n) ** 0.85
        r = r0 * np.exp(b * t) * (1.0 + (0.05 + 0.05 * t / th_max) * rng.standard_normal(n))
        a = t + phi0 + 0.5 * np.pi * arm + 0.05 * rng.standard_normal(n)
        fade = 1.0 - 0.55 * t / th_max                      # fainter outwards
        rgba = np.zeros((n, 4))
        rgba[:, :3] = matplotlib.colors.to_rgb("#1f5aa6" if major else "#4a7fc1")
        rgba[:, 3] = (0.55 if major else 0.32) * fade
        ax.scatter(cx + r * np.cos(a), cy + q * r * np.sin(a), s=rng.uniform(0.15, 1.1, n), c=rgba, lw=0, zorder=2,
                   rasterized=True)
    bright = rng.random(60)                                 # a few bright young stars on the major arms
    t = th_max * (0.15 + 0.8 * bright)
    a = t + phi0 + np.pi * (rng.random(60) < 0.5) + 0.04 * rng.standard_normal(60)
    r = r0 * np.exp(b * t)
    ax.scatter(cx + r * np.cos(a), cy + q * r * np.sin(a), s=1.6, color="white", edgecolors="#9ecae1", lw=0.2,
               zorder=2, rasterized=True)
    bar = np.degrees(np.arctan2(q * np.sin(phi0), np.cos(phi0)))
    blen = 2 * r0 * np.hypot(np.cos(phi0), q * np.sin(phi0))
    glow(ax, (cx, cy), 0.30 * Rd, "#fdb813", n=12, alpha=0.95, z=3, squash=q)
    ax.add_patch(Ellipse((cx, cy), blen, 0.38 * r0, angle=bar, fc="#fde68a", ec="none", alpha=0.95, zorder=3))
    ax.add_patch(Ellipse((cx, cy), 0.7, 0.5, fc="#fffbea", ec="none", zorder=3))
    # Galactic neutrinos: born in the disk (gas x cosmic rays, and sources), reaching the Sun
    for arm, frac in ((0, 0.62), (2, 0.38), (2, 0.66), (0, 0.9)):
        tt = frac * th_max
        aa, rr = tt + phi0 + 0.5 * np.pi * arm, r0 * np.exp(b * tt)
        gx, gy = cx + rr * np.cos(aa), cy + q * rr * np.sin(aa)
        if np.hypot(gx - sx, gy - sy) < 1.8 or gy < Y0 + 1.6:
            continue
        ax.plot([gx, sx], [gy, sy], color=C["gal"], lw=0.6, zorder=5, alpha=0.95)
        ax.scatter([gx], [gy], s=12, color=C["gal"], zorder=7, lw=0.3, edgecolors="#8c6d00", marker="*")
    strands(ax, 36.0, 57.0)
    dimension(ax, (sx, sy), (cx, cy), "8 kpc", "#08306b", side=1.0, off=1.6, box=True, t=0.8)  # Sun to Centre
    ax.add_patch(Circle((sx, sy), 0.36, fc="#fdb813", ec="#d95f02", lw=0.4, zorder=8))
    text(ax, sx + 0.5, sy - 1.45, "Sun", size=4.4, color=C["sun"], ha="left", path_effects=HALO)
    text(ax, 46.5, Y0 - 3.6, "Adiabatic", size=5.0, color=C["mw"], weight="bold")   # level with the Sun's "Coherent"
    text(ax, 52.6, 28.0, r"Galactic $\nu$", size=4.8, color="#8c6d00", weight="bold", path_effects=HALO)


def sun(ax):
    """The Sun as the calculation has it: a point source of its neutrons, whose potential falls as e^{-r/lambda}/r
    (the glow); no matter outside it. Paths reach the Earth at every angle to it."""
    sx, sy = 60.0, Y0 + 3.9
    glow(ax, (sx, sy), 3.3, "#fd8d3c", n=16, alpha=0.5)
    ax.add_patch(Circle((sx, sy), 1.0, fc="#fdb813", ec="#d95f02", lw=0.6, zorder=5))
    text(ax, sx, sy + 2.6, r"$V\propto e^{-r/\lambda}/r$", size=5.6, color=C["sun"], path_effects=HALO)
    dimension(ax, (sx + 0.75, sy - 0.65), (67.6, Y0 + 1.25), "1 AU", C["sun"], side=1.0, off=0.85)  # to the Earth
    text(ax, 62.7, Y0 - 3.6, "Coherent", size=5.0, color=C["sun"], weight="bold")


def earth(ax):
    """The globe with a quarter cut away (in the style of GeoneutrinoDecay's schematic), turned 90 degrees
    counter-clockwise (v3) so that IceCube, at the South Pole on the right, sits opposite the neutrinos' entry from the
    northern sky on the left: mantle and core in the cut, chords through it; short paths from the southern sky reach
    IceCube from the right; the dashed circle is the exterior, whose potential matters for lambda beyond the Earth's
    radius."""
    cx, cy, R = 75.4, Y0, 5.2
    # the exterior (v4: a halo, so as not to read as a Yukawa sphere)
    ax.add_patch(Wedge((cx, cy), 1.24 * R, 0.0, 360.0, width=0.24 * R, fc=C["earth"], ec="none", alpha=0.13, zorder=2))
    surface = Wedge((cx, cy), R, 180.0, 450.0, fc="#cfe3f5", ec="none", zorder=5)
    ax.add_patch(surface)
    for ring in globe_rings(-40.0):                        # turned: (x, y) -> (-y, x), the South Pole to the right
        land = Polygon(np.column_stack([cx - R * ring[:, 1], cy + R * ring[:, 0]]), closed=True, fc="#8fb98a",
                       ec="#4f7a55", lw=0.2, zorder=6)
        ax.add_patch(land)
        land.set_clip_path(surface)
    for r_out, r_in, col in ((1.0, 0.546, "#f6e8c3"), (0.546, 0.192, "#d4d4d4"), (0.192, 0.0, "#ececec")):
        ax.add_patch(Wedge((cx, cy), r_out * R, 90.0, 180.0, width=(r_out - r_in) * R, fc=col, ec="white", lw=0.4,
                           zorder=6))
    ax.add_patch(Circle((cx, cy), R, fill=False, ec=INK, lw=0.6, zorder=7))
    curved_text(ax, "Mantle", (cx, cy), 0.80 * R, 135.0, 4.3, "0.3")
    curved_text(ax, "Core", (cx, cy), 0.40 * R, 135.0, 4.3, "0.3")
    px, py = cx + R, cy                                    # IceCube
    for beta in (118.0, 145.0, 180.0):                     # up-going: chords through the mantle and the core
        bx, by = cx + R * np.cos(np.radians(beta)), cy + R * np.sin(np.radians(beta))
        ax.plot([bx, px], [by, py], color="#3f3f3f", lw=0.55, zorder=8)
        ax.scatter([bx], [by], s=3, color="#3f3f3f", zorder=8, lw=0)
    for dy in (2.8, -2.6):                                 # down-going: short paths from the southern sky
        ax.add_patch(FancyArrowPatch((px + 2.3, py + dy), (px + 0.35, py + 0.1 * np.sign(dy)), arrowstyle="-|>",
                                     mutation_scale=3, lw=0.45, color="#3f3f3f", zorder=8))
    ax.scatter([px], [py], s=34, marker="*", color="#2b6cb0", edgecolors="#08306b", lw=0.35, zorder=10)
    text(ax, cx, cy + 1.24 * R + 0.65, "Exterior", size=4.4, color=C["earth"])
    dimension(ax, (cx, cy), (cx, cy - R), r"$R_\oplus$", C["earth"], side=1.0, off=0.75, size=5.4)   # the radius
    text(ax, cx, cy - 1.24 * R - 1.3, r"IceCube: $\cos z=-\sin\delta$", size=4.8, color="#2b6cb0")


def detection(ax):
    y = 27.2
    ax.add_patch(Circle((86.0, y), 0.8, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=5))
    ax.plot([88.6, 91.0], [y - 0.8, y + 0.8], color="#08519c", lw=0.9, zorder=5)
    for t in np.linspace(0, 1, 5):
        ax.scatter([88.6 + 2.4 * t], [y - 0.8 + 1.6 * t], s=2.0, color="#08519c", zorder=6, lw=0)
    for xx in (93.6, 96.4):
        ax.add_patch(Circle((xx, y), 0.55, fc="#fdd0a2", ec="#d94801", lw=0.5, zorder=5))
    ax.plot([94.15, 95.85], [y, y], color="#d94801", lw=0.5, ls=(0, (1, 1)), zorder=5)
    for xx, lab in ((86.0, "Cascade"), (89.8, "Track"), (95.0, "Double\ncascade")):
        text(ax, xx, y - 2.0, lab, size=4.4, color="#525252")
    text(ax, 91.5, y + 2.3, "HESE + through-going tracks", size=4.8, weight="bold", color=C["det"])
    tx, ty, s = 85.0, 14.8, 6.0                            # the flavor triangle
    tri = np.array([[tx, ty], [tx + s, ty], [tx + s / 2, ty + s * np.sqrt(3) / 2]])
    ax.add_patch(Polygon(tri, fill=False, ec="#525252", lw=0.5, zorder=4))
    for (vx, vy), lab, off in zip(tri, (r"$\nu_\tau$", r"$\nu_\mu$", r"$\nu_e$"), ((-0.7, -0.4), (0.7, -0.4), (0, 0.6))):
        text(ax, vx + off[0], vy + off[1], lab, size=4.6)
    ccx, ccy = tx + s * 0.52, ty + s * 0.30
    for k, a in enumerate((0.55, 0.3)):
        ax.add_patch(Ellipse((ccx, ccy), 1.3 * (k + 1), 0.6 * (k + 1), angle=20, fc=C["det"], alpha=a * 0.5,
                             ec="none", zorder=5))
    ax.scatter([ccx + 0.5], [ccy + 0.25], s=5, color="#c44e52", zorder=6, lw=0)
    text(ax, tx + s / 2, ty - 1.4, "Flavor", size=4.6, color="#525252")
    mx, my, mw_, mh = 95.8, 17.6, 5.4, 2.7                 # a sky map (Mollweide) with the Galactic plane
    ax.add_patch(Ellipse((mx, my), mw_, mh, fc="#eff3ff", ec="#525252", lw=0.5, zorder=4))
    tt = np.linspace(-0.96, 0.96, 80)
    ax.plot(mx + tt * mw_ / 2, my + 0.5 * np.sin(np.pi * tt) * mh / 2 * 0.9, color=C["gal"], lw=2.0, alpha=0.85,
            zorder=5, solid_capstyle="round")
    text(ax, mx, my - mh / 2 - 1.0, "Sky maps", size=4.6, color="#525252")


def ingredients(ax):
    rows = {
        "Distant sources": ["Extragalactic,", r"with $z$-evolution", "Optional host galaxy"],
        "Cosmological propagation": [r"Refraction $+$ absorption, one calculation",
                                     r"Thermal relic $\nu$, Debye screening; cascade"],
        "Milky Way": [r"$n_{\rm eff}$: Yukawa integral of", "the Galaxy's neutrons",
                      r"Galactic $\nu$ born in local eigenstates"],
        "Sun": ["Potential along paths", "at all angles to the Sun", "Its yearly motion averaged"],
        "Earth": ["Yukawa integral of", "the Earth's neutrons", "Exact chords, PREM"],
        "Detection": ["IceCube responses", r"MCEq $+$ Barr", "Flavor, spectrum, direction"],
    }
    for name, xa, xb, col in STAGES:
        for i, line in enumerate(rows[name]):
            text(ax, 0.5 * (xa + xb), 7.6 - 1.3 * i, line, size=4.7, color="#252525")


def main():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, X1)
    ax.set_ylim(Y_LO, Y1)
    ax.set_aspect("equal")
    ax.axis("off")
    for k, (name, xa, xb, col) in enumerate(STAGES):
        text(ax, 0.5 * (xa + xb), 32.8, name, size=6.0, color=col, weight="bold")
        if k:
            ax.plot([xa, xa], [9.2, 32.2], color="#d9d9d9", lw=0.5, zorder=0)
    ax.plot([0.5, 99.5], [9.0, 9.0], color="#d9d9d9", lw=0.5, zorder=0)
    sources(ax)
    cosmology(ax)
    milky_way(ax)
    sun(ax)
    twist(ax, 57.0, 70.8)                                  # through the Sun's potential and into the Earth
    earth(ax)
    detection(ax)
    ingredients(ax)
    out = os.path.join(REPO, "output", "figures", "schematic_dark_portal")
    fig.savefig(out + ".pdf", dpi=600)                     # dpi: the rasterized stars of the Milky Way
    fig.savefig(out + ".png", dpi=300)
    print(f"wrote {out}.pdf, .png")


if __name__ == "__main__":
    main()
