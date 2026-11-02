#!/usr/bin/env python
"""The figures for the NuLRIDarkPortal progress talk.

Each figure is drawn at the size it occupies on its slide, in centimetres, in the palette and face of the deck
(build_deck.py), as in the TAMBO viewfinder talk, whose helpers these are. The equations are those of
notes/notes.tex, Sec. "The transport equation in full"; the limits are read from the plane summaries under
output/analysis/campaign_ml by analysis.exclusion.field. Nothing is typed in by hand except labels.

    python make_figures.py            # all figures
    python make_figures.py mev        # one figure, by name
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "figures"
sys.path[:0] = [str(REPO / "src"), str(REPO)]

# The deck's palette (build_deck.py).
BLUE = "#0262ab"
DEEPBLUE = "#0066b3"
DARKRED = "#bb0000"
GREEN = "#158466"
GREY = "#666666"
MAGENTA = "#bf0041"
BLACK = "#000000"
BLUE_BG = "#e8f0f8"
GREEN_BG = "#e3f2ec"
GREY_BG = "#ececec"
RED_BG = "#f8e1e1"

DPI = 300
FONT_SCALE = 1.12

# Placement sizes on the slide, in cm (width, height).
FIG_SIDE = (13.6, 11.6)
FIG_WIDE = (25.4, 10.6)
FIG_TRIP = (25.4, 6.2)
FIG_EQ = (25.4, 9.6)
FIG_EQ_LINE = (25.0, 1.9)


def _face(families, weight=400, style="normal"):
    for fam in families:
        for f in fm.fontManager.ttflist:
            if f.name == fam and f.style == style and abs(f.weight - weight) < 150:
                return fm.FontProperties(fname=f.fname)
    return fm.FontProperties(family="serif", weight=weight, style=style)


REG = _face(["Palatino Linotype", "TeX Gyre Pagella", "P052", "DejaVu Serif"])
BOLD = _face(["Palatino-Bold", "TeX Gyre Pagella", "P052", "DejaVu Serif"], weight=700)
ITAL = _face(["Palatino Linotype", "TeX Gyre Pagella", "P052", "DejaVu Serif"], style="italic")
for f in (REG, BOLD, ITAL):
    if f.get_file():
        fm.fontManager.addfont(f.get_file())
plt.rcParams.update({
    "font.family": REG.get_name(),
    "mathtext.fontset": "custom",
    "mathtext.rm": REG.get_name(),
    "mathtext.it": f"{ITAL.get_name()}:italic",
    "mathtext.bf": BOLD.get_name(),
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 4.5, "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
})
for f in (REG, BOLD, ITAL):
    f.set_math_fontfamily("custom")
SIZE = 13 * FONT_SCALE


def canvas(size):
    w, h = size
    return plt.figure(figsize=(w / 2.54, h / 2.54), dpi=DPI)


def axes(fig, size, left, bottom, right, top):
    """Data axes placed by margins in cm."""
    w, h = size
    ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - bottom - top) / h])
    ax.tick_params(labelsize=SIZE * 0.92)
    return ax


def label_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontproperties=REG, fontsize=SIZE)
    ax.set_ylabel(ylabel, fontproperties=REG, fontsize=SIZE)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(REG)
        lab.set_fontsize(SIZE * 0.92)


def text(ax, x, y, s, size=13, color=BLACK, ha="left", va="center", bold=False, italic=False,
         transform=None, **kw):
    fp = BOLD if bold else (ITAL if italic else REG)
    return ax.text(x, y, s, fontproperties=fp, fontsize=size * FONT_SCALE, color=color, ha=ha,
                   va=va, transform=transform or ax.transData, **kw)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  {path.relative_to(HERE)}")


# ---------------------------------------------------------------- F1: the trip

def trip():
    """The neutrino's path, stage by stage, and what the new force does in each (notes, Sec. 'The transport
    equation in full')."""
    fig = canvas(FIG_TRIP)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_TRIP[0])
    ax.set_ylim(0, FIG_TRIP[1])
    ax.axis("off")
    stages = [
        ("Sources", r"$z \leq 4$", BLUE, BLUE_BG,
         ["Follow star formation", "Broken power law", "Flavors: from pion", "decay, or fitted"]),
        ("The Universe", r"$z$ = 4 to 0", DARKRED, RED_BG,
         [r"Absorption on relic $\nu$", r"Scattering on relic $\nu$", "Neutrino decay", "Cosmic potential"]),
        ("Milky Way", "Tens of kpc", GREEN, GREEN_BG,
         ["Galactic emission", "Decays, scatterings", "along each path", "Galactic potential"]),
        ("Sun and Earth", "AU, Earth chords", MAGENTA, "#f6e0e8",
         ["Flavor evolution", "in their potential", "(coherent)"]),
        ("Detectors", "Today and future", GREY, GREY_BG,
         ["IceCube spectrum,", "flavor, events", "8 telescopes, 10 yr"]),
    ]
    n = len(stages)
    gap = 0.75
    bw = (FIG_TRIP[0] - 0.3 - (n - 1) * gap) / n
    top, bh = FIG_TRIP[1] - 0.25, FIG_TRIP[1] - 0.5
    for k, (name, scale, col, bg, lines) in enumerate(stages):
        x = 0.15 + k * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, top - bh), bw, bh, boxstyle="round,pad=0,rounding_size=0.35",
                                    fc=bg, ec=col, lw=1.6))
        text(ax, x + bw / 2, top - 0.65, name, size=12.5, color=col, ha="center", bold=True)
        text(ax, x + bw / 2, top - 1.35, scale, size=10.5, color=GREY, ha="center", italic=True)
        for j, ln in enumerate(lines):
            text(ax, x + bw / 2, top - 2.35 - 0.8 * j, ln, size=10.5, ha="center")
        if k < n - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.08, top - bh / 2), (x + bw + gap - 0.08, top - bh / 2),
                                         arrowstyle="-|>", mutation_scale=16, lw=1.6, color=BLACK))
    save(fig, "trip")


# ---------------------------------------------------------------- F2: equations

def _row(fig, y, pieces, x0=0.012, size=21, label_size=11.5, gap=0.006):
    """One row of an equation: (tex, label, colour) pieces set left to right; a coloured bracket and a label under
    each piece that has one. Positions in figure fractions."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x = x0
    for tex, label, col in pieces:
        t = fig.text(x, y, tex, ha="left", va="center", fontsize=size, color=BLACK)
        bb = t.get_window_extent(renderer=r).transformed(fig.transFigure.inverted())
        if label:
            yb = bb.y0 - 0.025
            fig.add_artist(plt.Line2D([bb.x0 + 0.004, bb.x0 + 0.004, bb.x1 - 0.004, bb.x1 - 0.004],
                                      [yb + 0.02, yb, yb, yb + 0.02], color=col, lw=1.6,
                                      transform=fig.transFigure))
            fig.text((bb.x0 + bb.x1) / 2, yb - 0.02, label, ha="center", va="top", fontsize=label_size * FONT_SCALE,
                     color=col, fontproperties=BOLD)
        x = bb.x1 + gap
    return x


def equations():
    # The cosmological phase, Eq. (eq:transport_cosmo) of the notes, all its terms.
    fig = canvas(FIG_EQ)
    _row(fig, 0.87, [
        (r"$\dfrac{\partial F_a(E,z)}{\partial t} \;=$", None, None),
        (r"$S_a$", "Sources", BLUE),
        (r"$-\,\Gamma^{\rm abs}_a F_a + \beta_a\,\int_E^\infty dE'\left[\dfrac{3E^2}{E'^3}\sum_c\Gamma^{\rm abs}_c F_c"
         r" + \dfrac{3(E'-E)^2}{E'^3}\sum_c\bar\Gamma^{\rm abs}_c\bar F_c\right]$",
         "Resonant absorption on the relic neutrinos, and what it re-emits", DARKRED),
    ], x0=0.03, gap=0.02)
    _row(fig, 0.545, [
        (r"$+\;\sum_{b\neq a}\left(R_{ab}F_b - R_{ba}F_a\right)$", "Soft scatterings on the relics", GREEN),
        (r"$+\;\dfrac{\partial}{\partial E}\left(b_a F_a\right)$", "Energy lost to the relics", MAGENTA),
    ], x0=0.19, gap=0.07)
    _row(fig, 0.24, [
        (r"$-\;\Gamma^{\rm dec}_a F_a + \sum_{a'}\,\int_E^\infty dE'\left[\dfrac{d\Gamma^{\rm keep}_{a'\to a}}{dE}"
         r"\,F_{a'}(E') + \dfrac{d\bar\Gamma^{\rm flip}_{a'\to a}}{dE}\,\bar F_{a'}(E')\right]$",
         "Decay into a lighter neutrino and the mediator, and the daughters", BLUE),
    ], x0=0.19)
    save(fig, "eq_cosmo")

    # The sources, Eq. (eq:sources).
    fig = canvas(FIG_EQ_LINE)
    fig.text(0.5, 0.5, r"$S_a = w_\nu\,\Phi_0\,[E(1+z)]^{-p}\,R(z)\,(1+z)\,\sum_\alpha f_{\alpha}\,"
             r"P_{\alpha a}(\sigma_{\rm src})$", ha="center", va="center", fontsize=17)
    save(fig, "eq_src")

    # The Hamiltonian whose eigenstates are followed.
    fig = canvas(FIG_EQ_LINE)
    fig.text(0.5, 0.5, r"$H(E,z) = \dfrac{U\,{\rm diag}(m_1^2, m_2^2, m_3^2)\,U^\dagger}{2E(1+z)} + V(z)$",
             ha="center", va="center", fontsize=17)
    save(fig, "eq_ham")

    # The Galaxy, Eq. (eq:transport_gal), and the Sun and Earth.
    fig = canvas((25.0, 2.2))
    fig.text(0.5, 0.5, r"$-\dfrac{\partial G_a}{\partial \ell} = j_a\;+\;$ the same scattering, energy-loss and "
             r"decay terms, at $z = 0$", ha="center", va="center", fontsize=17)
    save(fig, "eq_gal")
    fig = canvas((25.0, 1.9))
    fig.text(0.5, 0.5, r"$\dfrac{d\rho}{dx} = -i\,[H(x), \rho], \quad H(x) = \dfrac{UM^2U^\dagger}{2E} + V(x), \quad"
             r" \rho = \sum_a (F_a + G_a)\,|a\rangle\langle a|$",
             ha="center", va="center", fontsize=16)
    save(fig, "eq_det")


# ---------------------------------------------------------------- F3: MeV limits

MEV_MODELS = [("dp_e_mev", r"Dark portal, $\nu_e$", r"$g_D\,|U_{e4}|^2$"),
              ("dp_mu_mev", r"Dark portal, $\nu_\mu$", r"$g_D\,|U_{\mu 4}|^2$"),
              ("dp_tau_mev", r"Dark portal, $\nu_\tau$", r"$g_D\,|U_{\tau 4}|^2$"),
              ("le_lmu_mev", r"$L_e - L_\mu$", r"$g'$"),
              ("le_ltau_mev", r"$L_e - L_\tau$", r"$g'$")]


def mev():
    """The MeV planes' excluded regions (fixed pion composition): IceCube's spectrum + flavor (IC26 + MESE 2025),
    filled; HESE + DR2 events, line (preliminary: to be re-evaluated)."""
    from analysis import exclusion
    size = (25.4, 10.6)
    fig = canvas(size)
    n = len(MEV_MODELS)
    left, right, gap, bottom, top = 1.9, 0.25, 0.3, 2.45, 0.95
    pw = (size[0] - left - right - (n - 1) * gap) / n
    camp = REPO / "output/analysis/campaign_ml"
    for k, (name, title, ylab) in enumerate(MEV_MODELS):
        ax = axes(fig, size, left + k * (pw + gap), bottom, size[0] - left - k * (pw + gap) - pw, top)
        for dset, fill, col, ls in (("now_ic26", "#bcd3ea", BLUE, "-"), ("now_events", None, DARKRED, "--")):
            p = exclusion.plane(str(camp), dset, name)
            lx, G, F, _, _ = exclusion.field(p)
            if fill:
                ax.contourf(lx, G, F, levels=[0, 2 * exclusion.D], colors=[fill], zorder=1)
            ax.contour(lx, G, F, levels=[0], colors=[col], linewidths=1.7, linestyles=ls, zorder=5)
        ax.set_xlim(4, 9)
        ax.set_ylim(-5, -2)
        ax.set_xticks([4, 5, 6, 7, 8, 9])
        ax.set_xticklabels(["", "100 keV", "", "10 MeV", "", ""])
        ax.set_yticks(range(-5, -1))
        ax.set_yticklabels([f"$10^{{{v}}}$" for v in range(-5, -1)] if k == 0 else [])
        ax.tick_params(labelsize=SIZE * 0.85)
        text(ax, 0.5, 1.045, title, size=12.5, ha="center", va="bottom", transform=ax.transAxes, bold=True)
        text(ax, 0.5, -0.11, "Mediator mass", size=11.5, ha="center", va="top", transform=ax.transAxes)
        text(ax, 0.94, 0.06, ylab, size=12, color=GREY, ha="right", va="bottom", transform=ax.transAxes)
        if k == 0:
            ax.set_ylabel("Coupling", fontproperties=REG, fontsize=SIZE)
    # legend, under the panels: excluded at 95% CL
    lg = fig.add_axes([0, 0, 1, 0.09])
    lg.set_xlim(0, size[0])
    lg.set_ylim(0, 1)
    lg.axis("off")
    lg.fill_between([1.5, 2.5], [0.3, 0.3], [0.7, 0.7], color="#bcd3ea", ec=BLUE, lw=1.4)
    text(lg, 2.75, 0.5, "Excluded by IceCube's spectrum + flavor (95% CL)", size=11.5)
    lg.plot([14.6, 15.6], [0.5, 0.5], color=DARKRED, lw=1.7, ls="--")
    text(lg, 15.85, 0.5, "HESE + DR2 events (preliminary)", size=11.5)
    save(fig, "mev")


# ---------------------------------------------------------------- F4: Feynman diagrams

def _fermion(ax, p0, p1, arrow=1, color=BLACK, lw=1.6):
    """A fermion line from p0 to p1; arrow = +1 (along), -1 (against) or 0 (none) at its middle."""
    (x0, y0), (x1, y1) = p0, p1
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="round", zorder=3)
    if arrow:
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = (x1 - x0) * 0.04 * arrow, (y1 - y0) * 0.04 * arrow
        ax.add_patch(FancyArrowPatch((xm - dx, ym - dy), (xm + dx, ym + dy), arrowstyle="-|>",
                                     mutation_scale=13, lw=0, color=color, zorder=4))


def _wavy(ax, p0, p1, color=BLUE, lw=1.6, amp=0.11, waves=None):
    """A vector boson: a sine along p0 -> p1."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    n = waves or max(2, int(round(L / 0.32)))
    t = np.linspace(0, 1, 400)
    nrm = np.array([-d[1], d[0]]) / L
    pts = p0[None] + t[:, None] * d[None] + amp * np.sin(2 * np.pi * n * t)[:, None] * nrm[None]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=3)


def _dashed(ax, p0, p1, color=MAGENTA, lw=1.6):
    """A scalar: a dashed line."""
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=(0, (4, 2.5)), zorder=3)


def _dot(ax, p, color=BLACK, s=5.5):
    ax.plot([p[0]], [p[1]], "o", color=color, ms=s, zorder=5)


def _panel(fig, size, k, n, top=0.0, bottom=0.0):
    """The k-th of n equal panels across the figure, in data units of cm with (0, 0) at its bottom left."""
    w = size[0] / n
    ax = fig.add_axes([k / n, bottom / size[1], 1 / n, (size[1] - top - bottom) / size[1]])
    ax.set_xlim(0, w)
    ax.set_ylim(0, size[1] - top - bottom)
    ax.axis("off")
    return ax, w, size[1] - top - bottom


def couplings():
    """How the mediator couples, in each family: the vertices (notes Sec. 'Matter charges and textures', Table of
    benchmarks; the dark portal's texture u u^dagger, u_alpha = U_alpha4). The secret interactions' scalar and vector
    are separate models, one diagram each."""
    size = (25.4, 6.8)
    fig = canvas(size)
    panels = [
        ("Dark portal", BLUE, (r"$\nu_\alpha$", r"$\bar\nu_\beta$"), "wavy", r"$Z_D$",
         [r"Coupling $g_D\,U_{\alpha 4}U^*_{\beta 4}$:", "through the mixing with", r"a heavy neutrino, $\nu_4$"]),
        (r"$L_e - L_\mu$, $L_e - L_\tau$", BLUE, (r"$f$", r"$\bar f$"), "wavy", r"$Z'$",
         [r"$+g'$ for $\nu_e$, $e$; $-g'$ for $\nu_\mu$, $\mu$", r"(or $\nu_\tau$, $\tau$)",
          "Electrons: a matter potential"]),
        ("Secret interactions: scalar", MAGENTA, (r"$\nu_\alpha$", r"$\bar\nu_\alpha$"), "dashed", r"$\phi$",
         [r"Coupling $g$ to $\nu_e$, $\nu_\mu$ or $\nu_\tau$,", "or to all equally;",
          "neutrinos only"]),
        ("Secret interactions: vector", MAGENTA, (r"$\nu_\alpha$", r"$\bar\nu_\alpha$"), "wavy", r"$V$",
         [r"Coupling $g$ to $\nu_e$, $\nu_\mu$ or $\nu_\tau$,", "or to all equally; its longitudinal",
          r"mode couples as $m_\nu\,g/M$"]),
    ]
    n = len(panels)
    for k, (title, tcol, (top, bot), kind, boson, caption) in enumerate(panels):
        ax, w, h = _panel(fig, size, k, n)
        text(ax, w / 2, h - 0.45, title, size=12.5, color=tcol, ha="center", bold=True)
        v = (w / 2 - 0.3, h / 2 + 0.35)
        _fermion(ax, (v[0] - 1.5, v[1] + 1.2), v)
        _fermion(ax, v, (v[0] - 1.5, v[1] - 1.2))
        if kind == "wavy":
            _wavy(ax, v, (v[0] + 2.0, v[1]))
        else:
            _dashed(ax, v, (v[0] + 2.0, v[1]))
        _dot(ax, v)
        text(ax, v[0] - 1.65, v[1] + 1.2, top, size=14, ha="right")
        text(ax, v[0] - 1.65, v[1] - 1.2, bot, size=14, ha="right")
        text(ax, v[0] + 2.15, v[1], boson, size=14, color=MAGENTA if kind == "dashed" else BLUE)
        for j, ln in enumerate(caption):
            text(ax, w / 2, 1.45 - 0.55 * j, ln, size=11, ha="center")
        if k:
            fig.add_artist(plt.Line2D([k / n, k / n], [0.04, 0.96], color="#cccccc", lw=0.8,
                                      transform=fig.transFigure))
    save(fig, "feyn_couplings")


def processes():
    """What the mediator does, one diagram per process (notes Sec. 'The transport equation in full')."""
    size = (25.4, 9.6)
    fig = canvas(size)
    n = 5
    cap = 3.3
    for k in range(n):
        ax, w, h = _panel(fig, size, k, n)
        y0 = cap + (h - cap) / 2 - 0.15            # the diagram's centre line
        cx = w / 2
        if k == 0:     # s channel: beam nu + relic nubar -> mediator -> nu nubar
            a, b = (cx - 0.6, y0), (cx + 0.6, y0)
            _fermion(ax, (a[0] - 0.9, y0 + 1.2), a)
            _fermion(ax, a, (a[0] - 0.9, y0 - 1.2))
            _wavy(ax, a, b)
            _fermion(ax, b, (b[0] + 0.9, y0 + 1.2))
            _fermion(ax, (b[0] + 0.9, y0 - 1.2), b)
            for p in (a, b):
                _dot(ax, p)
            text(ax, a[0] - 0.9, y0 + 1.5, r"Beam $\nu$", size=11.5, ha="center")
            text(ax, a[0] - 0.9, y0 - 1.55, r"Relic $\bar\nu$", size=11.5, ha="center", color=GREY)
            text(ax, cx, y0 + 0.45, r"$Z'$", size=12, ha="center", color=BLUE)
            text(ax, b[0] + 0.95, y0 + 1.5, r"$\nu$", size=12, ha="center")
            text(ax, b[0] + 0.95, y0 - 1.55, r"$\bar\nu$", size=12, ha="center")
            lines = ("Resonant absorption", "10 keV – 1 GeV", [r"Peaks at $E = M^2/2m_\nu$:", "dips in the spectrum"])
        elif k == 1:   # t channel on a relic
            a, b = (cx, y0 + 0.85), (cx, y0 - 0.85)
            _fermion(ax, (cx - 1.7, a[1]), a)
            _fermion(ax, a, (cx + 1.7, a[1]))
            _fermion(ax, (cx - 1.7, b[1]), b, color=GREY)
            _fermion(ax, b, (cx + 1.7, b[1]), color=GREY)
            _wavy(ax, a, b, waves=3)
            for p in (a, b):
                _dot(ax, p)
            text(ax, cx - 1.75, a[1] + 0.4, r"$\nu_a$", size=12)
            text(ax, cx + 1.75, a[1] + 0.4, r"$\nu_b$", size=12, ha="right")
            text(ax, cx - 1.75, b[1] - 0.45, r"Relic $\nu_r$", size=11.5, color=GREY)
            text(ax, cx + 1.75, b[1] - 0.45, r"$\nu_s$", size=12, ha="right", color=GREY)
            text(ax, cx + 0.3, y0, r"$Z'$", size=12, color=BLUE)
            lines = ("Scattering on relics", "Below 1 keV", ["Changes the state:", "flavor change; energy",
                                                             r"lost if $r \neq s$"])
        elif k == 2:   # decay
            a = (cx - 0.55, y0)
            _fermion(ax, (a[0] - 1.25, y0), a)
            _fermion(ax, a, (a[0] + 1.6, y0 + 1.1))
            _wavy(ax, a, (a[0] + 1.6, y0 - 1.1), waves=3)
            _dot(ax, a)
            text(ax, a[0] - 1.3, y0 + 0.35, r"$\nu_i$", size=12)
            text(ax, a[0] + 1.7, y0 + 1.1, r"$\nu_j$", size=12)
            text(ax, a[0] + 1.7, y0 - 1.1, r"$Z'$", size=12, color=BLUE)
            lines = ("Decay", r"$M < m_i - m_j$", ["Heavier neutrinos decay:", "flux into the lightest",
                                                            "state, lower energy"])
        elif k == 3:   # coherent forward scattering on matter
            a, b = (cx, y0 + 0.85), (cx, y0 - 0.85)
            _fermion(ax, (cx - 1.7, a[1]), a)
            _fermion(ax, a, (cx + 1.7, a[1]))
            _wavy(ax, a, b, waves=3)
            _dot(ax, a)
            ax.add_patch(plt.Circle(b, 0.42, fc=GREY_BG, ec=GREY, lw=1.2, zorder=4))
            text(ax, b[0], b[1], r"$e$", size=12, ha="center", zorder=6)
            text(ax, cx - 1.75, a[1] + 0.4, r"$\nu$", size=12)
            text(ax, cx + 1.75, a[1] + 0.4, r"$\nu$", size=12, ha="right")
            text(ax, cx + 0.3, y0 + 0.1, r"$Z'$, $q \to 0$", size=11.5, color=BLUE)
            text(ax, cx, b[1] - 0.75, "Electrons of a body", size=11, ha="center", color=GREY)
            lines = ("Matter potential", r"Light; $L_e - L_\mu$, $L_e - L_\tau$", ["Every electron within", r"$1/M$ adds up:",
                                                                         "flavor mixing changes"])
        else:          # longitudinal mode: a scalar-like exchange
            a, b = (cx, y0 + 0.85), (cx, y0 - 0.85)
            _fermion(ax, (cx - 1.7, a[1]), a)
            _fermion(ax, a, (cx + 1.7, a[1]))
            _fermion(ax, (cx - 1.7, b[1]), b, color=GREY)
            _fermion(ax, b, (cx + 1.7, b[1]), color=GREY)
            _dashed(ax, a, b)
            for p in (a, b):
                _dot(ax, p)
            text(ax, cx - 1.75, a[1] + 0.4, r"$\nu$", size=12)
            text(ax, cx - 1.75, b[1] - 0.45, r"Relic $\nu$", size=11.5, color=GREY)
            text(ax, cx + 0.25, y0, r"$\propto m_\nu\,g/M$", size=11.5, color=MAGENTA)
            lines = ("Longitudinal mode", "Below 1 keV; vectors", ["The vector acts like a", "scalar too: hard",
                                                                   "scatterings, absorption"])
        title, mass, body = lines
        text(ax, cx, cap - 0.1, title, size=12.5, color=DARKRED, ha="center", bold=True)
        text(ax, cx, cap - 0.75, mass, size=11, color=GREY, ha="center", italic=True)
        for j, ln in enumerate(body):
            text(ax, cx, cap - 1.45 - 0.6 * j, ln, size=11.2, ha="center")
        if k:
            fig.add_artist(plt.Line2D([k / n, k / n], [0.04, 0.96], color="#cccccc", lw=0.8,
                                      transform=fig.transFigure))
    save(fig, "feyn_processes")


# ---------------------------------------------------------------- F5: where the resonance falls

def resonance():
    """The resonance energy E = M^2 / 2m against the mediator mass, for the relics' masses (normal ordering,
    lightest mass ~0, and a quasi-degenerate 0.1 eV), against the energies we fit."""
    import json
    cfg = json.load(open(REPO / "analysis/configs/campaign_ml/now_ic26/le_lmu_mev.json"))["solver"]["standard_mixing"]
    m2, m3 = np.sqrt(cfg["dm21_sq"]), np.sqrt(cfg["dm31_sq"])
    fig = canvas(FIG_SIDE)
    ax = axes(fig, FIG_SIDE, 2.2, 1.75, 0.75, 0.3)
    M = np.logspace(4, 9, 200)
    ax.axhspan(1e13, 3e16, color=BLUE_BG, zorder=0)
    text(ax, 1.3e4, 10 ** 14.8, "Energies\nwe fit", size=12, color=BLUE, va="center", bold=True)
    for m, lab, col in ((m3, f"$m_3$ = {m3:.3f} eV", DARKRED), (m2, f"$m_2$ = {m2:.4f} eV", GREEN),
                        (0.1, "$m$ = 0.1 eV (heaviest lightest mass)", GREY)):
        ax.plot(M, M ** 2 / (2 * m), color=col, lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e4, 1e9)
    ax.set_ylim(1e10, 1e20)
    ax.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    ax.set_xticklabels(["10 keV", "100 keV", "1 MeV", "10 MeV", "100 MeV", "1 GeV"])
    ax.set_yticks([1e12, 1e15, 1e18])
    ax.set_yticklabels(["1 TeV", "1 PeV", "1 EeV"])
    label_axes(ax, "Mediator mass, $M$", "Resonance energy, $E = M^2/2m$")
    text(ax, 1.4e4, 10 ** 19.4, f"Relic mass $m_2$ = {m2:.4f} eV", size=12, color=GREEN)
    text(ax, 1.4e4, 10 ** 18.75, f"Relic mass $m_3$ = {m3:.3f} eV", size=12, color=DARKRED)
    text(ax, 1.4e4, 10 ** 18.1, "All masses 0.1 eV", size=12, color=GREY)
    ax.axvline(3.16e6, color=GREY, lw=1.0, ls=(0, (1.5, 2)), zorder=1)
    text(ax, 3.5e6, 2e10, "Best fit,\n3.2 MeV", size=11, color=GREY, va="bottom")
    save(fig, "resonance")


# ---------------------------------------------------------------- F6: states in a potential

def levels():
    """The effective masses squared of the states that travel, the eigenvalues of U M^2 U^dagger + sigma S, against
    the potential's strength sigma = 2 E A, for L_e - L_mu (S = diag(1, -1, 0)), lightest mass 0.01 eV."""
    import json
    cfg = json.load(open(REPO / "analysis/configs/campaign_ml/now_ic26/le_lmu_mev.json"))["solver"]["standard_mixing"]
    d = np.radians
    t12, t13, t23, dcp = d(cfg["th12_deg"]), d(cfg["th13_deg"]), d(cfg["th23_deg"]), d(cfg["d_cp_deg"])
    s12, c12, s13, c13, s23, c23 = np.sin(t12), np.cos(t12), np.sin(t13), np.cos(t13), np.sin(t23), np.cos(t23)
    e = np.exp(1j * dcp)
    U = np.array([[c12 * c13, s12 * c13, s13 / e],
                  [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
                  [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])
    m1 = 0.01
    msq = np.array([m1 ** 2, m1 ** 2 + cfg["dm21_sq"], m1 ** 2 + cfg["dm31_sq"]])
    M2 = U @ np.diag(msq) @ U.conj().T
    S = np.diag([1.0, -1.0, 0.0])
    sig = np.logspace(-7, -1, 400)
    lam = np.array([np.linalg.eigvalsh(M2 + s * S) for s in sig])
    fig = canvas(FIG_SIDE)
    ax = axes(fig, FIG_SIDE, 2.2, 1.75, 0.35, 0.3)
    for a, col in zip(range(3), (GREEN, BLUE, DARKRED)):
        ax.plot(sig, lam[:, a], color=col, lw=1.9)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
    ax.set_xlim(sig[0], sig[-1])
    ax.set_ylim(-0.12, 0.12)
    ax.set_yticks([-1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["$-10^{-1}$", "$-10^{-2}$", "$-10^{-3}$", "0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax.minorticks_off()
    label_axes(ax, r"Potential's strength, $\sigma = 2EA$ [eV$^2$]", r"Effective mass squared, $\mu_a$ [eV$^2$]")
    text(ax, 2.0e-7, 4.2e-3, "$m_3^2$", size=13, color=DARKRED)
    text(ax, 2.0e-7, 2.3e-4, "$m_2^2$", size=13, color=BLUE, va="bottom")
    text(ax, 2.0e-7, 4.0e-5, "$m_1^2$", size=13, color=GREEN)
    text(ax, 2.5e-3, 4e-2, r"$\approx\nu_e$", size=13, color=DARKRED)
    text(ax, 2.5e-3, -4e-2, r"$\approx\nu_\mu$", size=13, color=GREEN)
    text(ax, 2.0e-2, 3.6e-3, r"$\approx\nu_\tau$", size=13, color=BLUE)
    save(fig, "levels")


# ---------------------------------------------------------------- F7: reading one column

def column_read():
    """Delta chi^2 against the coupling in one mass column of a plane: L_e - L_mu, IceCube spectrum + flavor,
    pion-decay flavors, M = 10 keV (the plane summary's own grid)."""
    import json
    s = json.load(open(REPO / "output/analysis/campaign_ml/now_ic26/le_lmu_mev/data/plane_m_Zd_eV_g_D.json"))
    x, y = np.array(s["grids"][0]), np.array(s["grids"][1])
    c = np.array([np.nan if v is None else v for v in s["chi2"][0]], dtype=float)
    d = c - np.nanmin(c)
    fig = canvas(FIG_SIDE)
    ax = axes(fig, FIG_SIDE, 2.0, 1.75, 0.35, 0.3)
    ax.plot(y, d, color=BLUE, lw=2.0, zorder=3)
    ax.plot(y, d, "o", ms=3.5, color=BLUE, zorder=4)
    ax.axhline(3.84, color=DARKRED, lw=1.4, ls="--")
    gl = np.interp(3.84, d[-4:], y[-4:])
    ax.axvspan(gl, y[-1] * 1.5, color=RED_BG, zorder=0)
    text(ax, 2e-10, 4.6, r"$\Delta\chi^2 = 3.84$: 95% CL", size=12, color=DARKRED)
    text(ax, gl * 0.8, 14, "Excluded", size=13, color=DARKRED, ha="right", bold=True)
    text(ax, 2e-10, 1.3, "No effect: the data can't tell", size=12, color=GREY)
    ax.set_xscale("log")
    ax.set_xlim(y[0], y[-1] * 1.5)
    ax.set_ylim(-0.5, 22)
    label_axes(ax, "Coupling, $g'$", r"$\Delta\chi^2$, every other parameter refitted")
    text(ax, 0.04, 0.96, f"$L_e - L_\\mu$, $M$ = {x[0] / 1e3:.0f} keV\nIceCube spectrum + flavor",
         size=12, transform=ax.transAxes, va="top")
    save(fig, "column")
    return f"limit {gl:.2e}"


# ---------------------------------------------------------------- F8: the flavor triangle

def _ternary_xy(fe, fm):
    """Flavor fractions to the plane of the ternary figure, python-ternary's orientation (as GeoneutrinoDecay's
    scripts/paper_figures.ternary_xy): f_e = 1 at the bottom right, f_mu = 1 at the top, f_tau = 1 at the bottom
    left, sides of unit length."""
    return fe + 0.5 * fm, 0.5 * np.sqrt(3.0) * fm


def _tint(color, amount=0.55):
    """A colour mixed with white (GeoneutrinoDecay's figure of the eigenstates' flavor content)."""
    from matplotlib.colors import to_rgb
    return tuple(1.0 - (1.0 - c) * (1.0 - amount) for c in to_rgb(color))


def _halo(t, width=3.0):
    import matplotlib.patheffects as pe
    t.set_path_effects([pe.Stroke(linewidth=width, foreground="w"), pe.Normal()])
    return t


def triangle():
    """IceCube's flavor measurement (MESE 2025: 68% and 95% C.L. contours of its likelihood surface, 2 dof) against
    the flavors at Earth from pion-decay sources and from a beam of pure nu_1, nu_2 or nu_3 (NuFIT 6.1 mixing of the
    configuration). Drawn as Mauricio's ternary figures (NuDiracVsMajorana paper_norunning/scripts/plot_ternary.py,
    GeoneutrinoDecay notebooks/paper_figures.ipynb): python-ternary, grid lines and ticks every 0.1."""
    import json
    import ternary
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from analysis import surfaces
    cfg = json.load(open(REPO / "analysis/configs/campaign_ml/now_ic26/le_lmu_mev.json"))["solver"]["standard_mixing"]
    d = np.radians
    t12, t13, t23, dcp = d(cfg["th12_deg"]), d(cfg["th13_deg"]), d(cfg["th23_deg"]), d(cfg["d_cp_deg"])
    s12, c12, s13, c13, s23, c23 = np.sin(t12), np.cos(t12), np.sin(t13), np.cos(t13), np.sin(t23), np.cos(t23)
    e = np.exp(1j * dcp)
    U = np.array([[c12 * c13, s12 * c13, s13 / e],
                  [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
                  [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])
    P = abs(U) ** 2
    pion = P @ (P.T @ np.array([1 / 3, 2 / 3, 0]))
    label, tick = 15, 11                                   # his sizes (TERNARY_LABEL, TERNARY_TICK)
    size = (12.6, 11.2)
    fig = canvas(size)
    ax = fig.add_axes([0.09, 0.11, 0.82, 0.84])
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=1)
    tax.set_background_color(color="white", alpha=1.0)
    ax.axis("off")
    tax.boundary(linewidth=1.0)
    tax.gridlines(color="gray", multiple=0.1, linewidth=0.5, ls="-", alpha=0.5)
    tax.left_axis_label(r"Fraction of $\nu_\tau$, $f_{\tau, \oplus}$", fontsize=label, offset=0.16)
    tax.right_axis_label(r"Fraction of $\nu_\mu$, $f_{\mu, \oplus}$", fontsize=label, offset=0.16)
    tax.bottom_axis_label(r"Fraction of $\nu_e$, $f_{e, \oplus}$", fontsize=label, offset=0.08)
    tax.ticks(axis="blr", linewidth=0.5, multiple=0.1, offset=0.022, clockwise=False, tick_formats="%.1f",
              fontsize=tick)
    # IceCube's surface on a fine grid, filled at 68% and 95% C.L.
    g = np.linspace(0, 1, 301)
    FE, FM = np.meshgrid(g, g)
    ok = FE + FM <= 1
    C = np.full(FE.shape, np.nan)
    for i, j in zip(*np.where(ok)):
        C[i, j] = surfaces.chi2(FE[i, j], FM[i, j], "icecube_mese_2025")
    X, Y = _ternary_xy(FE, FM)
    ic = "#4477aa"
    ax.contourf(X, Y, C, levels=[0, 2.30], colors=[ic], zorder=3)
    ax.contourf(X, Y, C, levels=[2.30, 5.99], colors=[_tint(ic)], zorder=2)
    ax.contour(X, Y, C, levels=[2.30, 5.99], colors=[ic, ic], linewidths=0.8, zorder=3)
    # pion decay, and a beam of one mass state
    px, py = _ternary_xy(pion[0], pion[1])
    ax.scatter([px], [py], marker="o", s=55, facecolor="w", edgecolor="k", linewidths=1.1, zorder=6)
    cols = (DARKRED, "#e6ce58", "#5e3576")
    for i in range(3):
        qx, qy = _ternary_xy(P[0, i], P[1, i])
        ax.scatter([qx], [qy], marker="s", s=48, color=cols[i], edgecolor="k", linewidths=0.6, zorder=6)
        _halo(text(ax, qx + 0.03, qy + (0.035 if i else -0.04), rf"$\nu_{i + 1}$", size=label / FONT_SCALE,
                   color=BLACK, ha="left", zorder=7))
    q1 = _ternary_xy(P[0, 0], P[1, 0])
    ax.add_patch(FancyArrowPatch((px + 0.015, py - 0.015), (q1[0] - 0.02, q1[1] + 0.015), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.4, color=DARKRED, zorder=5,
                                 connectionstyle="arc3,rad=-0.25"))
    _halo(text(ax, 0.5 * (px + q1[0]) + 0.06, 0.5 * (py + q1[1]) + 0.01, "Decay", size=tick / FONT_SCALE,
               color=DARKRED, ha="left", italic=True, zorder=7))
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    ax.set_aspect("equal")
    fig.legend([Patch(facecolor=ic, edgecolor=ic, lw=0.8), Patch(facecolor=_tint(ic), edgecolor=ic, lw=0.8),
                Line2D([], [], ls="", marker="o", ms=6, mfc="w", mec="k", mew=1.1),
                Line2D([], [], ls="", marker="s", ms=6, mfc=DARKRED, mec="k", mew=0.6)],
               ["IceCube 68% C.L.", "IceCube 95% C.L.", "Pion-decay sources", r"Pure $\nu_i$ beam"],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), bbox_transform=fig.transFigure, borderaxespad=0.0,
               fontsize=tick * 0.9, labelspacing=0.35, frameon=True, edgecolor="k", fancybox=False, handlelength=1.4)
    save(fig, "triangle")
    return (f"pion {pion.round(3)}, nu1 {P[:, 0].round(3)}: chi2 "
            f"{surfaces.chi2(P[0, 0], P[1, 0], 'icecube_mese_2025'):.1f}")


FIGURES = {"trip": trip, "equations": equations, "mev": mev, "couplings": couplings, "processes": processes,
           "resonance": resonance, "levels": levels, "column": column_read, "triangle": triangle}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(FIGURES):
        r = FIGURES[n]()
        if r is not None:
            print(f"    {n}: {r}")
