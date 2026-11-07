"""
trip_illustration.py -- Fig. 3 of the paper (his request, 2026-11-07: "an artist's rendition, with added elements
pointing out what's going on"): the trip of the high-energy neutrinos in four panels that zoom in -- the Universe,
the Milky Way, near the Sun, inside the Earth -- with callouts on what happens in each. Raster art (star fields,
glows, the spiral galaxy, the Earth's layers and continents) under vector overlays (the neutrino's path, the events,
the labels). Illustrative, not to scale, except: the Sun 8.3 kpc from the Galactic Centre on a disk of 15 kpc radius;
the Earth's layers at the radii of the Preliminary Reference Earth Model (src/dp_spatial.calc_earth_density); the
continents from Natural Earth's 1:110m land (tools/data/naturalearth, as tools/schematic_dark_portal.py). His
corrections of 2026-11-07: capitalized scales, every callout ends on the object it names, no bend where the
neutrino enters the Earth, real continents. Called from notebooks/paper_figures.ipynb (Sec. 7) after
analysis.figure_style.use(); figure() returns the figure, deterministic (fixed seeds).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
from scipy.ndimage import gaussian_filter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DPI_ART = 420                                # pixels per inch of the raster art
W, H = 7.0, 2.79                             # inches
Y0, Y1 = 0.08, 2.42                          # the panels' bottom and top
PANELS = {"universe": (0.04, 2.62), "galaxy": (2.80, 4.30), "sun": (4.48, 5.40), "earth": (5.58, 6.96)}
BEAM = "#bfe9ff"
BEAM_GLOW = "#4fb3ff"
SECONDARY = "#ffb347"
TEXT = "white"
FS = 6.6
rng = np.random.default_rng(7)          # reset by figure()


# ------------------------------------------------------------------------------------------------ raster helpers

def canvas(name):
    x0, x1 = PANELS[name]
    nx, ny = int((x1 - x0) * DPI_ART), int((Y1 - Y0) * DPI_ART)
    return np.zeros((ny, nx, 3)), (x0, x1)


def to_px(img, xy, xr):
    """Inches (figure) -> pixel (col, row) in a panel image."""
    x0, _ = xr
    return (xy[0] - x0) * DPI_ART, (Y1 - xy[1]) * DPI_ART


def splat(acc, cols, rows, weights, color, sigma):
    ny, nx = acc.shape[:2]
    m = np.zeros((ny, nx))
    c = np.clip(np.round(cols).astype(int), 0, nx - 1)
    r = np.clip(np.round(rows).astype(int), 0, ny - 1)
    np.add.at(m, (r, c), weights)
    if sigma > 0:
        m = gaussian_filter(m, sigma)
    acc += m[..., None] * np.asarray(color)[None, None, :]


def blob(acc, col, row, sigma, color, amp):
    ny, nx = acc.shape[:2]
    yy, xx = np.mgrid[0:ny, 0:nx]
    g = amp * np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma ** 2))
    acc += g[..., None] * np.asarray(color)[None, None, :]


def ellipse_blob(acc, col, row, sx, sy, angle, color, amp):
    ny, nx = acc.shape[:2]
    yy, xx = np.mgrid[0:ny, 0:nx]
    ca, sa = np.cos(angle), np.sin(angle)
    u = (xx - col) * ca + (yy - row) * sa
    v = -(xx - col) * sa + (yy - row) * ca
    acc += (amp * np.exp(-u ** 2 / (2 * sx ** 2) - v ** 2 / (2 * sy ** 2)))[..., None] * np.asarray(color)


def stars(acc, n, amp=0.9):
    ny, nx = acc.shape[:2]
    cols, rows = rng.uniform(0, nx, n), rng.uniform(0, ny, n)
    w = amp * rng.pareto(2.5, n).clip(0, 6) / 3 + 0.05
    tint = np.array([[0.85, 0.9, 1.0], [1.0, 0.95, 0.85], [1.0, 1.0, 1.0]])
    for t in range(3):
        sel = rng.random(n) < 1 / 3
        splat(acc, cols[sel], rows[sel], w[sel], tint[t], 0.6)


def background(img, top=(0.015, 0.02, 0.05), bottom=(0.03, 0.035, 0.08)):
    ny = img.shape[0]
    t = np.linspace(0, 1, ny)[:, None, None]
    return (1 - t) * np.asarray(top) + t * np.asarray(bottom)


def tone(acc, base):
    """Additive light -> displayable color (soft saturation), over the base."""
    return np.clip(base + 1.0 - np.exp(-acc), 0, 1)


# ------------------------------------------------------------------------------------------------ vector helpers

def glow(ax, x, y, color=BEAM, glow_color=BEAM_GLOW, lw=1.0, alpha=1.0, z=10, dashes=None):
    for w, a in ((7 * lw, 0.05), (4.5 * lw, 0.09), (2.6 * lw, 0.2)):
        ax.plot(x, y, color=glow_color, lw=w, alpha=a * alpha, solid_capstyle="round", zorder=z)
    kw = dict(dashes=dashes) if dashes else {}
    ax.plot(x, y, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z + 1, **kw)


def wave(xa, ya, xb, yb, amp, wl0, wl1, n=600):
    """A wave packet along the segment, its wavelength stretching from wl0 to wl1 (redshift)."""
    t = np.linspace(0, 1, n)
    L = np.hypot(xb - xa, yb - ya)
    wl = wl0 + (wl1 - wl0) * t
    phase = 2 * np.pi * np.cumsum(np.gradient(t * L) / wl)
    ux, uy = (xb - xa) / L, (yb - ya) / L
    x = xa + t * (xb - xa) - uy * amp * np.sin(phase)
    y = ya + t * (yb - ya) + ux * amp * np.sin(phase)
    return x, y


HALO = [patheffects.withStroke(linewidth=1.8, foreground="#05070f", alpha=0.85)]


def callout(ax, text, xy, xytext, ha="left", va="center", color=TEXT, fs=FS, z=30):
    t = ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=fs, color=color, zorder=z,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color=color, alpha=0.85, shrinkA=1.5, shrinkB=1.5))
    t.set_path_effects(HALO)
    return t


def burst(ax, x, y, r, color, z=15, n=8):
    ang = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    rad = np.where(np.arange(2 * n) % 2 == 0, r, 0.38 * r)
    ax.add_patch(Polygon(np.c_[x + rad * np.cos(ang), y + rad * np.sin(ang)], closed=True, fc=color, ec="none",
                         alpha=0.95, zorder=z))
    for k, a in ((3.0, 0.08), (2.0, 0.15)):
        ax.add_patch(Circle((x, y), k * r * 0.5, fc=color, ec="none", alpha=a, zorder=z - 1))


def zoom_box(ax, cx, cy, half, target):
    """A square around (cx, cy) and the lines from its right corners to the left corners of panel `target`."""
    ax.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False, ec="white", lw=0.6, zorder=40))
    tx = PANELS[target][0]
    for (sx, sy), ty in (((cx + half, cy + half), Y1), ((cx + half, cy - half), Y0)):
        ax.plot([sx, tx], [sy, ty], color="0.55", lw=0.5, zorder=41)


# ------------------------------------------------------------------------------------------------ the panels

def spiral_points(n, r_disk, arms=2, pitch=0.22, seed=3):
    g = np.random.default_rng(seed)
    pts, cols = [], []
    # arms
    na = int(0.55 * n)
    k = g.integers(0, arms, na)
    th = g.uniform(0.3, 3.3 * np.pi, na)
    r = 0.12 * r_disk * np.exp(pitch * th)
    keep = r < r_disk
    th, r, k = th[keep], r[keep], k[keep]
    phi = th + 2 * np.pi * k / arms
    spread = 0.025 * r_disk + 0.045 * r
    x = r * np.cos(phi) + g.normal(0, 1, len(r)) * spread
    y = r * np.sin(phi) + g.normal(0, 1, len(r)) * spread
    pts.append(np.c_[x, y]); cols.append(np.tile([0.55, 0.72, 1.0], (len(x), 1)))
    # the disk
    nd = int(0.3 * n)
    rr = g.exponential(0.3 * r_disk, nd); pp = g.uniform(0, 2 * np.pi, nd)
    pts.append(np.c_[rr * np.cos(pp), rr * np.sin(pp)]); cols.append(np.tile([0.85, 0.85, 0.95], (nd, 1)))
    # the bulge
    nb = n - na - nd
    b = g.normal(0, 0.09 * r_disk, (nb, 2))
    pts.append(b); cols.append(np.tile([1.0, 0.82, 0.55], (nb, 1)))
    return np.vstack(pts), np.vstack(cols)


def draw_galaxy(acc, xr, cx, cy, R_in, incl=0.5, rot=-0.3, n=220000, amp=0.35, seed=3, sun_kpc=None):
    """A face-on spiral, tilted, centered at (cx, cy) inches, radius R_in inches. Returns the projection function."""
    p, c = spiral_points(n, 1.0, seed=seed)
    ca, sa = np.cos(rot), np.sin(rot)

    def project(u, v):
        x, y = u * ca - v * incl * sa, u * sa + v * incl * ca
        return cx + R_in * x, cy + R_in * y

    X, Y = project(p[:, 0], p[:, 1])
    col, row = to_px(acc, (X, Y), xr)
    for ch_color in ([0.55, 0.72, 1.0], [0.85, 0.85, 0.95], [1.0, 0.82, 0.55]):
        sel = np.all(np.isclose(c, ch_color), axis=1)
        splat(acc, col[sel], row[sel], amp * np.ones(sel.sum()), ch_color, 1.0 * R_in / 0.7)
        splat(acc, col[sel], row[sel], 0.35 * amp * np.ones(sel.sum()), ch_color, 6.0 * R_in / 0.7)
    # star-forming knots along the arms
    g = np.random.default_rng(seed + 1)
    arm = np.all(np.isclose(c, [0.55, 0.72, 1.0]), axis=1)
    pick = g.choice(np.flatnonzero(arm), size=max(4, int(140 * R_in)), replace=False)
    splat(acc, col[pick], row[pick], 2.5 * np.ones(len(pick)), [1.0, 0.45, 0.7], 1.2 * R_in / 0.7)
    bc, br = to_px(acc, (cx, cy), xr)
    ellipse_blob(acc, bc, br, 0.10 * R_in * DPI_ART, 0.07 * R_in * DPI_ART, rot, [1.0, 0.85, 0.6], 2.2)
    # dust lanes: the arms' inner edges, darkened
    th = np.linspace(0.6, 3.1 * np.pi, 4000)
    dust = np.zeros(acc.shape[:2])
    for k in range(2):
        r = 0.12 * np.exp(0.22 * th) * 0.93
        keep = r < 0.95
        u, v = r[keep] * np.cos(th[keep] + np.pi * k), r[keep] * np.sin(th[keep] + np.pi * k)
        dc, dr = to_px(acc, project(u, v), xr)
        m = np.zeros_like(dust)
        np.add.at(m, (np.clip(dr.astype(int), 0, m.shape[0] - 1), np.clip(dc.astype(int), 0, m.shape[1] - 1)), 1.0)
        dust += gaussian_filter(m, 4.5 * R_in / 0.7)
    acc *= np.exp(-1.6 * dust / max(dust.max(), 1e-9))[..., None]
    return project


def panel_universe(ax):
    acc, xr = canvas("universe")
    base = background(acc)
    ny, nx = acc.shape[:2]
    stars(acc, 900, 0.5)
    # the cosmic web, faint
    nodes = np.c_[rng.uniform(0, nx, 26), rng.uniform(0, ny, 26)]
    web = []
    for i, a in enumerate(nodes):
        d = np.hypot(*(nodes - a).T)
        for j in np.argsort(d)[1:3]:
            t = rng.uniform(0, 1, 400)[:, None]
            web.append(a + t * (nodes[j] - a) + rng.normal(0, 6, (400, 2)))
    web = np.vstack(web)
    splat(acc, web[:, 0], web[:, 1], 0.05 * np.ones(len(web)), [0.55, 0.35, 0.85], 5)
    # relic neutrinos, everywhere
    rc, rr = rng.uniform(0, nx, 420), rng.uniform(0, ny, 420)
    splat(acc, rc, rr, 0.6 * np.ones(420), [0.35, 0.9, 0.75], 1.2)
    # the sources: distant galaxies, one with jets
    srcs = [(0.33, 1.82, 0.075, 0.035, 0.5), (0.36, 0.78, 0.05, 0.03, -0.3), (0.16, 1.22, 0.04, 0.025, 1.1)]
    for x, y, sx, sy, a in srcs:
        c, r = to_px(acc, (x, y), xr)
        ellipse_blob(acc, c, r, sx * DPI_ART, sy * DPI_ART, a, [1.0, 0.85, 0.65], 1.3)
        blob(acc, c, r, 3.0, [1.0, 0.95, 0.85], 4.0)
    c, r = to_px(acc, srcs[0][:2], xr)                        # jets of the first source
    for sgn in (1, -1):
        L = np.linspace(0, 0.22 * DPI_ART, 300)
        splat(acc, c + sgn * L * np.cos(2.1), r - sgn * L * np.sin(2.1), np.exp(-L / (0.12 * DPI_ART)) * 0.6,
              [0.6, 0.75, 1.0], 1.2)
    # the Milky Way, far away at the right end
    draw_galaxy(acc, xr, 2.43, 1.25, 0.13, n=30000, amp=0.9, seed=5)
    img = tone(acc, base)
    ax.imshow(img, extent=(xr[0], xr[1], Y0, Y1), zorder=1, interpolation="lanczos")
    return srcs


def panel_galaxy(ax):
    acc, xr = canvas("galaxy")
    base = background(acc)
    stars(acc, 500, 0.5)
    proj = draw_galaxy(acc, xr, 3.52, 1.22, 0.68)
    img = tone(acc, base)
    ax.imshow(img, extent=(xr[0], xr[1], Y0, Y1), zorder=1, interpolation="lanczos")
    return proj


def panel_sun(ax):
    acc, xr = canvas("sun")
    base = background(acc)
    stars(acc, 260, 0.5)
    sx, sy = 4.72, 0.95
    c, r = to_px(acc, (sx, sy), xr)
    blob(acc, c, r, 0.30 * DPI_ART, [1.0, 0.55, 0.15], 0.5)
    blob(acc, c, r, 0.13 * DPI_ART, [1.0, 0.75, 0.35], 1.4)
    blob(acc, c, r, 0.06 * DPI_ART, [1.0, 0.95, 0.8], 4.0)
    ex, ey = 5.18, 1.62
    c, r = to_px(acc, (ex, ey), xr)
    blob(acc, c, r, 0.02 * DPI_ART, [0.3, 0.55, 1.0], 1.2)
    img = tone(acc, base)
    ax.imshow(img, extent=(xr[0], xr[1], Y0, Y1), zorder=1, interpolation="lanczos")
    return (sx, sy), (ex, ey)


LAND = os.path.join(REPO, "tools", "data", "naturalearth", "land_110m_decimated.json")
LON0 = -15.0                     # the central meridian: the visible half shows Africa and Europe


def globe_rings(lon0_deg):
    """The land outlines on a globe seen from above the equator at longitude lon0_deg (orthographic, in Earth radii,
    north up); far-side points pushed onto the limb, rings mostly hidden dropped (as tools/schematic_dark_portal.py)."""
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


# PREM radii (km), as in src/dp_spatial.calc_earth_density
R_E = 6371.0
LAYERS = [(R_E, [0.55, 0.42, 0.30]), (6346.6, [0.80, 0.33, 0.18]), (5701.0, [0.62, 0.20, 0.12]),
          (3480.0, [0.98, 0.62, 0.20]), (1221.5, [1.0, 0.90, 0.55])]


def panel_earth(ax):
    acc, xr = canvas("earth")
    base = background(acc)
    stars(acc, 300, 0.5)
    img = tone(acc, base)
    ny, nx = img.shape[:2]
    cx, cy, R = 6.27, 1.22, 0.62
    xs = xr[0] + (np.arange(nx) + 0.5) / DPI_ART
    ys = Y1 - (np.arange(ny) + 0.5) / DPI_ART
    X, Y = np.meshgrid(xs, ys)
    u, v = (X - cx) / R, (Y - cy) / R
    rho = np.hypot(u, v)
    inside = rho <= 1.0
    # the surface: an ocean sphere lit from the upper left, with an ice cap at the South Pole
    nz = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
    L = np.array([-0.35, 0.15, 0.92]) / np.linalg.norm([-0.35, 0.15, 0.92])
    light = 0.33 + 0.75 * np.clip(L[0] * u + L[1] * v + L[2] * nz, 0, None)
    ocean = np.array([0.12, 0.35, 0.75])
    surf = ocean[None, None, :] * light[..., None]
    # the real land (Natural Earth 1:110m, the repository's tools/data/naturalearth), seen from above the equator
    land = np.zeros(inside.shape, dtype=bool)
    pix = np.column_stack([u[inside], v[inside]])
    for ring in globe_rings(LON0):
        land[inside] |= MplPath(ring).contains_points(pix)
    lat = np.degrees(np.arcsin(np.clip(v, -1, 1)))
    land |= inside & (lat < -69.5)              # Antarctica: its coast lies near 70 S across the visible sector
    patchy = gaussian_filter(rng.random(lat.shape), 10)
    patchy = (patchy - patchy.min()) / (patchy.max() - patchy.min())
    desert = (np.exp(-((lat - 22.0) / 8.0) ** 2) + 0.6 * np.exp(-((lat + 24.0) / 5.0) ** 2)) \
        * np.clip(0.4 + 1.2 * patchy, 0, 1)                            # the Sahara, Arabia, the Kalahari: tan
    land_rgb = (1 - desert)[..., None] * np.array([0.27, 0.45, 0.22]) + desert[..., None] * np.array([0.72, 0.6, 0.4])
    land_rgb = np.where((lat < -60.0)[..., None], np.array([0.93, 0.95, 1.0]), land_rgb)   # Antarctica's ice
    surf = np.where(land[..., None], land_rgb * light[..., None], surf)
    # the cut: the left half shows the layers
    cut = inside & (u < 0)
    sec = np.zeros_like(surf)
    for r_km, color in LAYERS:
        m = rho <= r_km / R_E
        shade = 0.82 + 0.18 * (1 - rho)
        sec = np.where(m[..., None], np.asarray(color)[None, None, :] * shade[..., None], sec)
    img = np.where(cut[..., None], sec, np.where(inside[..., None], surf, img))
    # the atmosphere's rim
    rim = np.exp(-((rho - 1.0) / 0.02) ** 2) * (rho > 0.98)
    img = np.clip(img + rim[..., None] * np.array([0.25, 0.45, 0.9]) * 0.6, 0, 1)
    ax.imshow(img, extent=(xr[0], xr[1], Y0, Y1), zorder=1, interpolation="lanczos")
    return cx, cy, R


# ------------------------------------------------------------------------------------------------ the figure

def figure():
    """The illustration, 7.0 x 2.79 inches."""
    global rng
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    srcs = panel_universe(ax)
    proj = panel_galaxy(ax)
    sun, earth = panel_sun(ax)
    ecx, ecy, ER = panel_earth(ax)

    # panel frames and titles
    titles = {"universe": ("The Universe", "Gigaparsecs"), "galaxy": ("The Milky Way", "Kiloparsecs"),
              "sun": ("Near the Sun", "1 AU"), "earth": ("Inside the Earth", "6371 km")}
    for k, (x0, x1) in PANELS.items():
        ax.add_patch(Rectangle((x0, Y0), x1 - x0, Y1 - Y0, fill=False, ec="0.25", lw=0.6, zorder=50))
        ax.text((x0 + x1) / 2, Y1 + 0.2, titles[k][0], ha="center", va="bottom", fontsize=9.5)
        ax.text((x0 + x1) / 2, Y1 + 0.05, titles[k][1], ha="center", va="bottom", fontsize=7, color="0.35",
                style="italic")

    # ---- the Universe
    sx, sy = srcs[0][0], srcs[0][1]
    mx, my = 2.43, 1.25                                     # the Milky Way, far away
    k1, k2, k3 = 0.36, 0.60, 0.80                           # where the absorption, scattering, decay happen
    px = lambda t: sx + t * (mx - sx)
    py = lambda t: sy + t * (my - sy)
    label_at, rel = np.array([0.55, 0.3]), np.array([0.86, 0.64])   # the label, and the relic it points to
    u = (rel - label_at) / np.hypot(*(rel - label_at))
    labels = [callout(ax, "Relic neutrinos", tuple(rel - 0.014 * u), tuple(label_at), ha="center")]
    x, y = wave(sx + 0.05, sy - 0.02, mx - 0.07, my, 0.022, 0.07, 0.16)
    glow(ax, x, y, lw=0.8)
    # absorption on a relic, with its secondaries
    ax_, ay_ = px(k1), py(k1)
    burst(ax, ax_, ay_, 0.055, "#fff2b3")
    for dy in (-0.05, 0.04):
        t = np.linspace(0, 1, 50)
        xs = ax_ + 0.42 * t
        ys = ay_ + (my - sy) / (mx - sx) * 0.42 * t + dy * np.sin(np.pi * t * 0.9)
        glow(ax, xs, ys, color=SECONDARY, glow_color=SECONDARY, lw=0.55, alpha=0.9, z=8)
    labels.append(callout(ax, "Absorption on a relic $\\nu$", (ax_ - 0.01, ay_ + 0.04), (ax_ + 0.18, 2.2), ha="center"))
    labels.append(callout(ax, "Secondaries,\nat lower energy", (ax_ + 0.3, ay_ - 0.1), (ax_ + 0.12, 0.98), ha="center"))
    # scattering on a relic
    bx, by = px(k2), py(k2)
    ax.add_patch(Circle((bx + 0.012, by - 0.065), 0.02, fc="#8ff2d6", ec="none", zorder=16))
    ax.annotate("", xy=(bx + 0.09, by - 0.2), xytext=(bx + 0.02, by - 0.08),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#8ff2d6", mutation_scale=6), zorder=16)
    labels.append(callout(ax, "Scattering on\na relic $\\nu$", (bx + 0.07, by - 0.17), (bx + 0.3, 0.95), ha="center"))
    # decay into a lighter neutrino and the mediator
    dx, dy_ = px(k3), py(k3)
    t = np.linspace(0, 1, 200)
    mxs, mys = dx + 0.24 * t, dy_ + 0.36 * t + 0.012 * np.sin(2 * np.pi * 9 * t)
    ax.plot(mxs, mys, color="#ff7eb6", lw=0.7, zorder=12)
    ax.add_patch(Circle((dx, dy_), 0.013, fc="white", ec="none", zorder=13))
    labels.append(callout(ax, "Decay:\n$\\nu_i \\to \\nu_j$ + mediator", (dx + 0.2, dy_ + 0.3), (dx + 0.12, 2.12), ha="center"))
    # the sources, the relics, the potential, the Milky Way
    labels.append(callout(ax, "Sources, $z \\leq 4$", (sx - 0.02, sy + 0.05), (0.42, 2.24), ha="center"))
    labels.append(ax.text(1.75, Y0 + 0.07, "Potential of the Universe's\nelectrons ($L_e - L_\\alpha$)", ha="center", va="bottom", fontsize=FS,
            color=TEXT, style="italic", zorder=30, linespacing=1.05,
            path_effects=HALO))
    labels.append(callout(ax, "Milky Way", (mx, my - 0.15), (mx - 0.05, 0.62), ha="center"))
    zoom_box(ax, mx, my, 0.15, "galaxy")
    # the relic neutrinos everywhere, kept out from under the labels and the relic label's leader
    fig.canvas.draw()
    inv = ax.transData.inverted()
    boxes = [inv.transform(Text.get_window_extent(t).expanded(1.04, 1.15)) for t in labels]   # the text alone
    leaders = [(np.array(t.xyann), np.array(t.xy)) for t in labels if hasattr(t, "xyann")]

    def near_segment(p, a, b, d):
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
        return np.hypot(*(p - (a + t * (b - a)))) < d
    g2 = np.random.default_rng(11)
    pts = np.vstack([np.c_[g2.uniform(0.1, 2.55, 150), g2.uniform(0.15, 2.35, 150)], rel[None, :]])
    def free(p):
        if np.allclose(p, rel):
            return True
        if any(b[0, 0] - 0.03 <= p[0] <= b[1, 0] + 0.03 and b[0, 1] - 0.03 <= p[1] <= b[1, 1] + 0.03 for b in boxes):
            return False
        if any(near_segment(p, a, b, 0.035) for a, b in leaders):
            return False
        return np.hypot(*(p - rel)) > 0.07
    for p_ in filter(free, pts):
        ax.add_patch(Circle(p_, 0.022, fc="#59e6bf", ec="none", alpha=0.12, zorder=4))
        ax.add_patch(Circle(p_, 0.008, fc="#8ff2d6", ec="none", alpha=0.85, zorder=4))

    # ---- the Milky Way: the beam reaches the Sun, 8.3 kpc from the centre (disk radius ~ 15 kpc drawn as 1)
    sun_g = proj(np.cos(-0.25) * 8.3 / 15, np.sin(-0.25) * 8.3 / 15)
    glow(ax, [PANELS["galaxy"][0], sun_g[0]], [1.95, sun_g[1]], lw=0.8)
    for u, v in ((0.0, 0.0), (-0.2, 0.3), (-0.45, -0.1), (0.1, -0.45)):
        gx, gy = proj(u, v)
        ax.plot([gx, sun_g[0]], [gy, sun_g[1]], color=SECONDARY, lw=0.5, alpha=0.85, zorder=9)
    ax.add_patch(Circle(sun_g, 0.016, fc="#fff3c4", ec="none", zorder=20))
    callout(ax, "Galactic neutrinos", proj(-0.25, -0.2), (3.5, 0.3), ha="center")
    callout(ax, "Potential of its\nstars and gas", proj(-0.75, 0.35), (3.25, 2.18), ha="center")
    callout(ax, "Sun", (sun_g[0] - 0.01, sun_g[1] - 0.06), (sun_g[0] - 0.1, sun_g[1] - 0.48), ha="center")
    zoom_box(ax, sun_g[0], sun_g[1], 0.06, "sun")

    # ---- near the Sun: the beam passes the Sun and reaches the Earth
    glow(ax, [PANELS["sun"][0], earth[0]], [2.05, earth[1]], lw=0.8)
    ax.add_patch(Circle((earth[0] + 0.07, earth[1] + 0.05), 0.008, fc="0.75", ec="none", zorder=20))
    th = np.linspace(1.0, 1.75, 100)
    rr = np.hypot(earth[0] - sun[0], earth[1] - sun[1])
    ax.plot(sun[0] + rr * np.cos(th - 0.35), sun[1] + rr * np.sin(th - 0.35), color="0.5", lw=0.4, ls=(0, (2, 2)),
            zorder=5)
    callout(ax, "Earth and Moon", (earth[0] - 0.03, earth[1] - 0.03), (4.84, 1.38), ha="center")
    callout(ax, "Sun", (sun[0] + 0.05, sun[1] + 0.05), (4.98, 1.0), ha="left")
    ax.text((PANELS["sun"][0] + PANELS["sun"][1]) / 2, 0.36, "Potentials of the\nSun, the Moon and\nthe Earth switch on;\n"
            "flavor evolves\ncoherently", ha="center", va="center", fontsize=FS, color=TEXT, zorder=30,
            linespacing=1.1, path_effects=HALO)
    zoom_box(ax, earth[0], earth[1], 0.06, "earth")

    # ---- inside the Earth: a chord from the Northern sky to the South Pole
    a_in = np.radians(112)
    ex0, ey0 = ecx + ER * np.cos(a_in), ecy + ER * np.sin(a_in)
    ex1, ey1 = ecx, ecy - ER * 0.985
    dxc, dyc = ex1 - ex0, ey1 - ey0                         # the chord's direction, extended back to the frame
    t_top = (Y1 - 0.01 - ey0) / dyc
    t_left = (PANELS["earth"][0] + 0.01 - ex0) / dxc
    t0 = max(t_top, t_left)                                 # both negative: the nearer frame edge
    glow(ax, [ex0 + t0 * dxc, ex0], [ey0 + t0 * dyc, ey0], lw=0.8)
    glow(ax, [ex0, ex1], [ey0, ey1], lw=0.8, z=20)
    for k in range(5):                                       # IceCube's strings in the ice
        xk = ex1 - 0.03 + 0.015 * k
        ax.plot([xk, xk], [ey1 - 0.012, ey1 + 0.035], color="#ffe066", lw=0.5, zorder=25)
    callout(ax, "IceCube", (ex1 + 0.03, ey1 + 0.0), (ex1 + 0.42, Y0 + 0.13), ha="center")
    callout(ax, "Potential along\nthe chord", (ex0 + 0.25 * (ex1 - ex0), ey0 + 0.25 * (ey1 - ey0)),
            (6.62, 2.2), ha="center")
    callout(ax, "Core", (ecx - 0.12, ecy - 0.1), (5.8, 0.42), ha="center")
    callout(ax, "Mantle", (ecx - 0.48, ecy - 0.05), (5.74, 0.95), ha="center")

    return fig
