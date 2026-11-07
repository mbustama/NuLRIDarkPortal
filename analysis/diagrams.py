"""
diagrams.py -- the Feynman diagrams of the paper (2026-11-07; paper/paper_planning.md, Step 5, "The processes of
each model"): every process the transport includes, drawn once (processes), and each model's vertices and the
scalar's own diagrams (vertices). Drawn in the repository's style (analysis.figure_style.use() first); called from
notebooks/paper_figures.ipynb, which saves them to paper/figures/.

Conventions: the beam's neutrinos black, the relic neutrinos gray, the mediator X wavy (a vector: Z', Z_D, V) or
dashed (the scalar phi); arrows along the fermion flow.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

INK = "black"
RELIC = "#808080"
LW = 1.4
LABEL = 11.5        # pt, particle labels
TITLE = 11.0        # pt, panel titles


# ---------------------------------------------------------------- primitives (coordinates in cm)

def fermion(ax, p0, p1, arrow=1, color=INK, lw=LW):
    """A fermion line p0 -> p1, with an arrow at its middle: +1 along, -1 against, 0 none."""
    (x0, y0), (x1, y1) = p0, p1
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="round", zorder=3)
    if arrow:
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        L = np.hypot(x1 - x0, y1 - y0)
        dx, dy = (x1 - x0) / L * 0.08 * arrow, (y1 - y0) / L * 0.08 * arrow
        ax.add_patch(FancyArrowPatch((xm - dx, ym - dy), (xm + dx, ym + dy), arrowstyle="-|>",
                                     mutation_scale=10, lw=0, color=color, zorder=4))


def wavy(ax, p0, p1, color=INK, lw=LW, amp=0.09, wavelength=0.27):
    """A vector: a sine along p0 -> p1, a whole number of waves."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(*d)
    n = max(2, int(round(L / wavelength)))
    t = np.linspace(0.0, 1.0, 400)
    nrm = np.array([-d[1], d[0]]) / L
    pts = p0[None] + t[:, None] * d[None] + amp * np.sin(2 * np.pi * n * t)[:, None] * nrm[None]
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=3)


def dashed(ax, p0, p1, color=INK, lw=LW):
    """A scalar: a dashed line."""
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=(0, (4, 2.5)), zorder=3)


def mediator(ax, p0, p1, kind="vector", **kw):
    (wavy if kind == "vector" else dashed)(ax, p0, p1, **kw)


def vertex(ax, p, s=4.0):
    ax.plot([p[0]], [p[1]], "o", color=INK, ms=s, zorder=5)


def label(ax, p, s, color=INK, size=LABEL, **kw):
    ax.text(p[0], p[1], s, color=color, fontsize=size, zorder=6, **kw)


def panel(fig, rect, w, h):
    """An axes at rect (figure fractions) with data units of cm, (0, 0) at the bottom left, equal aspect."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------- the diagrams, centred on (cx, cy)

def potential(ax, cx, cy):
    """(a) Forward scattering on the electrons of a body: the potential."""
    a, b = (cx, cy + 0.65), (cx, cy - 0.55)
    fermion(ax, (cx - 1.3, a[1]), a)
    fermion(ax, a, (cx + 1.3, a[1]))
    wavy(ax, a, (b[0], b[1] + 0.32))
    vertex(ax, a)
    ax.add_patch(Circle(b, 0.32, fc="#eeeeee", ec=INK, lw=1.0, zorder=4))
    label(ax, b, r"$e$", ha="center", va="center")
    label(ax, (cx - 1.3, a[1] + 0.18), r"$\nu_\alpha$", va="bottom")
    label(ax, (cx + 1.3, a[1] + 0.18), r"$\nu_\alpha$", ha="right", va="bottom")
    label(ax, (cx + 0.2, cy + 0.15), r"$Z'$", va="center")
    label(ax, (cx + 0.45, b[1]), r"body", va="center", size=LABEL - 1.5, color=RELIC)


def s_channel(ax, cx, cy, kind="vector", relic_bar=True, out=(r"$f$", r"$\bar f$"), med=r"$X$"):
    """(b) Absorption on a relic: beam nu + relic nubar (or nu) -> X -> two particles."""
    a, b = (cx - 0.45, cy), (cx + 0.45, cy)
    fermion(ax, (a[0] - 0.85, cy + 0.85), a)
    fermion(ax, (a[0] - 0.85, cy - 0.85), a, arrow=-1 if relic_bar else 1, color=RELIC)
    mediator(ax, a, b, kind)
    fermion(ax, b, (b[0] + 0.85, cy + 0.85))
    fermion(ax, b, (b[0] + 0.85, cy - 0.85), arrow=-1 if relic_bar else 1)
    vertex(ax, a)
    vertex(ax, b)
    label(ax, (a[0] - 0.95, cy + 0.85), r"$\nu$", ha="right", va="center")
    label(ax, (a[0] - 0.95, cy - 0.85), r"$\bar\nu$" if relic_bar else r"$\nu$", ha="right", va="center",
          color=RELIC)
    label(ax, (cx, cy + 0.22), med, ha="center", va="bottom")
    label(ax, (b[0] + 0.95, cy + 0.85), out[0], va="center")
    label(ax, (b[0] + 0.95, cy - 0.85), out[1], va="center")


def t_channel(ax, cx, cy, kind="vector", med=r"$X$", crossed=False):
    """(c) Scattering on a relic through the t channel, or (d) its crossed version, the u channel."""
    a, b = (cx, cy + 0.6), (cx, cy - 0.6)
    fermion(ax, (cx - 1.2, a[1]), a)
    fermion(ax, (cx - 1.2, b[1]), b, color=RELIC)
    if crossed:
        fermion(ax, a, (cx + 1.2, b[1]))
        fermion(ax, b, (cx + 1.2, a[1]), color=RELIC)
    else:
        fermion(ax, a, (cx + 1.2, a[1]))
        fermion(ax, b, (cx + 1.2, b[1]), color=RELIC)
    mediator(ax, a, b, kind)
    vertex(ax, a)
    vertex(ax, b)
    label(ax, (cx - 1.3, a[1]), r"$\nu$", ha="right", va="center")
    label(ax, (cx - 1.3, b[1]), r"$\nu$", ha="right", va="center", color=RELIC)
    label(ax, (cx + 1.3, a[1]), r"$\nu$", va="center")
    label(ax, (cx + 1.3, b[1]), r"$\nu$", va="center")
    label(ax, (cx - 0.15, cy), med, ha="right", va="center")


def pair(ax, cx, cy, kind="vector", relic_bar=True, med=r"$X$"):
    """(e) Pair production: nu + relic nubar -> X X, through neutrino exchange (the crossed diagram as well)."""
    a, b = (cx - 0.2, cy + 0.55), (cx - 0.2, cy - 0.55)
    fermion(ax, (cx - 1.3, a[1]), a)
    fermion(ax, a, b)
    fermion(ax, (cx - 1.3, b[1]), b, arrow=-1 if relic_bar else 1, color=RELIC)
    mediator(ax, a, (cx + 1.1, a[1]), kind)
    mediator(ax, b, (cx + 1.1, b[1]), kind)
    vertex(ax, a)
    vertex(ax, b)
    label(ax, (cx - 1.4, a[1]), r"$\nu$", ha="right", va="center")
    label(ax, (cx - 1.4, b[1]), r"$\bar\nu$" if relic_bar else r"$\nu$", ha="right", va="center", color=RELIC)
    label(ax, (cx + 1.2, a[1]), med, va="center")
    label(ax, (cx + 1.2, b[1]), med, va="center")


def mediator_decay(ax, cx, cy, kind="vector", out=(r"$f$", r"$\bar f$"), med=r"$X$", anti=True):
    """(f) The mediator's decay into a pair (anti: a fermion and its antifermion; else two Majorana neutrinos)."""
    a = (cx + 0.1, cy)
    mediator(ax, (cx - 1.2, cy), a, kind)
    fermion(ax, a, (a[0] + 1.0, cy + 0.75))
    fermion(ax, a, (a[0] + 1.0, cy - 0.75), arrow=-1 if anti else 1)
    vertex(ax, a)
    label(ax, (cx - 1.3, cy), med, ha="right", va="center")
    label(ax, (a[0] + 1.1, cy + 0.75), out[0], va="center")
    label(ax, (a[0] + 1.1, cy - 0.75), out[1], va="center")


def nu_decay(ax, cx, cy, kind="vector", med=r"$X$"):
    """(g) A neutrino's decay into a lighter one and the mediator."""
    a = (cx - 0.35, cy)
    fermion(ax, (cx - 1.4, cy), a)
    fermion(ax, a, (a[0] + 1.1, cy + 0.75))
    mediator(ax, a, (a[0] + 1.1, cy - 0.75), kind)
    vertex(ax, a)
    label(ax, (cx - 1.5, cy), r"$\nu_i$", ha="right", va="center")
    label(ax, (a[0] + 1.2, cy + 0.75), r"$\nu_j$", va="center")
    label(ax, (a[0] + 1.2, cy - 0.75), med, va="center")


def longitudinal(ax, cx, cy):
    """(h) The longitudinal mode's coupling, a scalar-like vertex of strength Y_ij ~ m g / M."""
    a = (cx - 0.35, cy)
    fermion(ax, (cx - 1.4, cy), a)
    fermion(ax, a, (a[0] + 1.1, cy + 0.75))
    dashed(ax, a, (a[0] + 1.1, cy - 0.75))
    vertex(ax, a)
    label(ax, (cx - 1.5, cy), r"$\nu_i$", ha="right", va="center")
    label(ax, (a[0] + 1.2, cy + 0.75), r"$\nu_j$", va="center")
    label(ax, (a[0] + 1.2, cy - 0.75), r"$X_L$", va="center")
    label(ax, (a[0] - 0.1, cy - 0.3), r"$Y_{ij}$", ha="right", va="top", size=LABEL - 1)


# ---------------------------------------------------------------- the figures

PROCESSES = [
    (potential, "(a) Potential of matter", r"$L_e - L_\alpha$"),
    (s_channel, r"(b) Absorption, $s$ channel", "All models"),
    (t_channel, r"(c) Scattering, $t$ channel", "All models"),
    (lambda ax, cx, cy: t_channel(ax, cx, cy, crossed=True), r"(d) Scattering, $u$ channel", "All models"),
    (pair, "(e) Pair production", "All models"),
    (mediator_decay, "(f) Mediator decay", "All models"),
    (nu_decay, "(g) Neutrino decay", r"$L_e - L_\alpha$, dark portal"),
    (longitudinal, "(h) Longitudinal mode", "Vector mediators"),
]


def processes(width_in=7.0):
    """Every process the transport includes (a)-(h), two rows of four. The mediator X is drawn as a vector; the
    scalar's diagrams, with its own lines, are in vertices()."""
    ncol, nrow = 4, 2
    w, h = 4.4, 3.6                                   # cm per panel
    top = 1.2                                         # cm, the key
    W, H = ncol * w, nrow * h + top
    fig = plt.figure(figsize=(width_in, width_in * H / W))
    for k, (draw, title, models) in enumerate(PROCESSES):
        r, c = divmod(k, ncol)
        ax = panel(fig, [c / ncol, (nrow - 1 - r) * h / H, 1 / ncol, h / H], w, h)
        draw(ax, w / 2, h / 2 - 0.15)
        ax.text(w / 2, h - 0.25, title, ha="center", va="top", fontsize=TITLE)
        ax.text(w / 2, 0.15, models, ha="center", va="bottom", fontsize=TITLE - 1.5, color=RELIC, style="italic")
    key = fig.add_axes([0, nrow * h / H, 1, top / H])
    key.axis("off")
    for y, line in ((0.68, r"Mediator $X$: $Z'$ ($L_e - L_\alpha$), $Z_D$ (dark portal), $V$ or $\phi$ (secret "
                           r"interactions; $\phi$'s own diagrams in the appendix)"),
                    (0.22, r"Beam neutrinos black, relic neutrinos gray; $f$: a neutrino or a charged lepton")):
        key.text(0.5, y, line, ha="center", va="center", fontsize=TITLE - 1.5)
    return fig


def _vertex_legs(ax, cx, cy, top, bottom, kind, med, arrows=(1, -1)):
    """A vertex at (cx, cy) with two fermion legs from the left (labels top, bottom) and the mediator to the right."""
    v = (cx - 0.2, cy)
    fermion(ax, (v[0] - 1.0, cy + 0.75), v, arrow=arrows[0])
    fermion(ax, (v[0] - 1.0, cy - 0.75), v, arrow=arrows[1])
    mediator(ax, v, (v[0] + 1.2, cy), kind)
    vertex(ax, v)
    label(ax, (v[0] - 1.1, cy + 0.75), top, ha="right", va="center")
    label(ax, (v[0] - 1.1, cy - 0.75), bottom, ha="right", va="center")
    label(ax, (v[0] + 1.3, cy), med, va="center")


VERTICES = [
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_\alpha$", r"$\bar\nu_\beta$", "vector", r"$Z_D$"),
     "Dark portal\n", r"$g_D\,U_{\alpha 4} U^*_{\beta 4}$"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\ell$, $\nu_\ell$", r"$\bar\ell$, $\bar\nu_\ell$", "vector",
                                     r"$Z'$"),
     r"$L_e - L_\alpha$" + "\n", r"$+g'$ ($\ell = e$), $-g'$ ($\ell = \alpha$)"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_\alpha$", r"$\bar\nu_\beta$", "vector", r"$V$"),
     "Secret interactions:\nvector", r"$G_{\alpha\beta}$"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_\alpha$", r"$\nu_\beta$", "scalar", r"$\phi$",
                                     arrows=(0, 0)),
     "Secret interactions:\nscalar, Majorana", r"$G_{\alpha\beta}$"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_{L\alpha}$", r"$\bar\nu_{R\beta}$", "scalar",
                                     r"$\phi$"),
     "Secret interactions:\nscalar, Dirac", r"$G_{\alpha\beta}$, $\nu_L \to \nu_R$"),
]

SCALAR = [
    (lambda ax, cx, cy: s_channel(ax, cx, cy, "scalar", relic_bar=False, out=(r"$\nu$", r"$\nu$"),
                                  med=r"$\phi$"), r"$s$ channel"),
    (lambda ax, cx, cy: t_channel(ax, cx, cy, "scalar", med=r"$\phi$"), r"$t$ channel"),
    (lambda ax, cx, cy: t_channel(ax, cx, cy, "scalar", med=r"$\phi$", crossed=True), r"$u$ channel"),
    (lambda ax, cx, cy: pair(ax, cx, cy, "scalar", relic_bar=False, med=r"$\phi$"), "Pair production"),
    (lambda ax, cx, cy: mediator_decay(ax, cx, cy, "scalar", out=(r"$\nu$", r"$\nu$"), med=r"$\phi$", anti=False),
     "Mediator decay"),
]


def vertices(width_in=7.0):
    """Appendix figure: each model's vertex (top row) and the scalar's own diagrams, for Majorana neutrinos (bottom
    row; on a Majorana scalar vertex two neutrinos meet, so its s channel starts from a relic neutrino)."""
    ncol = 5
    w, h, hv = 3.6, 3.3, 3.7                          # cm: panel width, scalar row, vertex row
    W, H = ncol * w, h + hv
    fig = plt.figure(figsize=(width_in, width_in * H / W))
    for k, (draw, title, coupling) in enumerate(VERTICES):
        ax = panel(fig, [k / ncol, h / H, 1 / ncol, hv / H], w, hv)
        draw(ax, w / 2 + 0.2, hv / 2 + 0.1)
        ax.text(w / 2, hv - 0.1, title, ha="center", va="top", fontsize=TITLE - 1, linespacing=1.1)
        ax.text(w / 2, 0.3, coupling, ha="center", va="bottom", fontsize=TITLE - 1.5, color=RELIC)
    for k, (draw, title) in enumerate(SCALAR):
        ax = panel(fig, [k / ncol, 0, 1 / ncol, h / H], w, h)
        draw(ax, w / 2, h / 2 - 0.1)
        ax.text(w / 2, h - 0.2, r"$\phi$: " + title, ha="center", va="top", fontsize=TITLE - 1)
    return fig
