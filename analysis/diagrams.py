"""
diagrams.py -- the Feynman diagrams of the paper (2026-11-07; paper/paper_planning.md, Step 5, "The processes of
each model"): each model's vertex (vertices) and every process the transport includes, with the vector mediator
and with the scalar (processes); the two figures sit one above the other on the same page. Drawn in the repository's style (analysis.figure_style.use() first); called from
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

# The processes (2026-11-07, Mauricio: one figure of processes, the vector mediator above the scalar, beside the
# figure of vertices on the same page; in-place labels instead of a key). Vector diagrams (a)-(h):
VECTOR = dict(
    a=(potential, "(a) Potential of matter"),
    b=(s_channel, r"(b) Absorption, $s$ channel"),
    c=(t_channel, r"(c) Scattering, $t$ channel"),
    d=(lambda ax, cx, cy: t_channel(ax, cx, cy, crossed=True), r"(d) Scattering, $u$ channel"),
    e=(pair, "(e) Pair production"),
    f=(mediator_decay, "(f) Mediator decay"),
    g=(nu_decay, "(g) Neutrino decay"),
    h=(longitudinal, "(h) Longitudinal mode"),
)
# the scalar's, for Majorana neutrinos: two neutrinos meet at its vertex (none for (a) and (h))
SCALAR = dict(
    b=lambda ax, cx, cy: s_channel(ax, cx, cy, "scalar", relic_bar=False, out=(r"$\nu$", r"$\nu$"), med=r"$\phi$"),
    c=lambda ax, cx, cy: t_channel(ax, cx, cy, "scalar", med=r"$\phi$"),
    d=lambda ax, cx, cy: t_channel(ax, cx, cy, "scalar", med=r"$\phi$", crossed=True),
    e=lambda ax, cx, cy: pair(ax, cx, cy, "scalar", relic_bar=False, med=r"$\phi$"),
    f=lambda ax, cx, cy: mediator_decay(ax, cx, cy, "scalar", out=(r"$\nu$", r"$\nu$"), med=r"$\phi$", anti=False),
    g=lambda ax, cx, cy: nu_decay(ax, cx, cy, "scalar", med=r"$\phi$"),
)
ONLY = dict(a=r"$L_e - L_\alpha$ only", h="Vector mediators only")
SMALL = TITLE - 2.5     # pt, the in-place notes


def processes(width_in=7.0):
    """Every process the transport includes, (a)-(h), in two blocks of four columns: per block a row with the vector
    mediator X (titles above) and a row with the scalar phi (Majorana neutrinos). The first beam neutrino is marked
    "astro.", the first relic one "relic", the first f "(l or nu)"; the caption names the mediators."""
    lab = 1.0                                         # cm, the row labels
    w, hsr = 4.3, 2.55                                # cm: column, scalar row
    HVR = dict(abcd=3.35, efgh=2.95)                  # cm: vector rows (two-line titles above, one-line below)
    cyv, cys = 1.32, hsr / 2                          # cm: diagram centres
    W = lab + 4 * w
    H = sum(HVR.values()) + 2 * hsr
    fig = plt.figure(figsize=(width_in, width_in * H / W))
    y = H
    for block, hvr in HVR.items():
        y -= hvr
        for c, p in enumerate(block):
            ax = panel(fig, [(lab + c * w) / W, y / H, w / W, hvr / H], w, hvr)
            draw, title = VECTOR[p]
            cx = w / 2
            draw(ax, cx, cyv)
            ax.text(cx, hvr - 0.02, title.replace(", ", ",\n", 1), ha="center", va="top", fontsize=TITLE,
                    linespacing=1.0)
            if p in ONLY:
                ax.text(cx, 0.02, ONLY[p], ha="center", va="bottom", fontsize=TITLE - 1.5, color=RELIC,
                        style="italic")
            if p == "a":                              # the first beam neutrino (potential's incoming nu_alpha)
                ax.text(cx - 1.3, cyv + 0.65 - 0.12, "astro.", ha="left", va="top", fontsize=SMALL, color=INK)
            if p == "b":                              # the first relic neutrino and the first f (s_channel's legs)
                ax.text(cx - 0.45 - 0.95, cyv - 0.85 - 0.28, "relic", ha="right", va="top", fontsize=SMALL,
                        color=RELIC)
                ax.text(cx + 0.45 + 0.95, cyv + 0.85 - 0.28, r"($\ell$ or $\nu$)", ha="left", va="top",
                        fontsize=SMALL)
        rl = fig.add_axes([0, y / H, lab / W, hvr / H])
        rl.axis("off")
        rl.text(0.5, cyv / hvr, r"Vector $X$", rotation=90, ha="center", va="center", fontsize=TITLE - 0.5)
        y -= hsr
        for c, p in enumerate(block):
            if p in SCALAR:
                ax = panel(fig, [(lab + c * w) / W, y / H, w / W, hsr / H], w, hsr)
                SCALAR[p](ax, w / 2, cys)
        rl = fig.add_axes([0, y / H, lab / W, hsr / H])
        rl.axis("off")
        rl.text(0.5, 0.5, r"Scalar $\phi$", rotation=90, ha="center", va="center", fontsize=TITLE - 0.5)
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


# Each model's vertex, a plain junction without fermion-flow arrows so that it does not read as a process; the labels
# carry the neutrinos' kinds (two neutrinos meet at a Majorana scalar's vertex; a Dirac scalar turns nu_L into nu_R)
VERTICES = [
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_\alpha$", r"$\bar\nu_\beta$", "vector", r"$Z_D$",
                                     arrows=(0, 0)), "Dark portal", r"$g_D\,U_{\alpha 4} U^*_{\beta 4}$"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\ell$, $\nu_\ell$", r"$\bar\ell$, $\bar\nu_\ell$", "vector",
                                     r"$Z'$", arrows=(0, 0)), r"$L_e - L_\alpha$",
     r"$+g'$ ($\ell = e$), $-g'$ ($\ell = \alpha$)"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_\alpha$", r"$\bar\nu_\beta$", "vector", r"$V$",
                                     arrows=(0, 0)), "Secret interactions,\nvector", r"$G_{\alpha\beta}$"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_\alpha$", r"$\nu_\beta$", "scalar", r"$\phi$",
                                     arrows=(0, 0)), "Secret interactions,\nscalar, Majorana", r"$G_{\alpha\beta}$"),
    (lambda ax, cx, cy: _vertex_legs(ax, cx, cy, r"$\nu_{L\alpha}$", r"$\bar\nu_{R\beta}$", "scalar", r"$\phi$",
                                     arrows=(0, 0)), "Secret interactions,\nscalar, Dirac",
     r"$G_{\alpha\beta}$, $\nu_L \to \nu_R$"),
]


def vertices(width_in=7.0):
    """Each model's vertex, one row, with its coupling under it (the figure above the processes, same page)."""
    hv = 3.6                                          # cm
    W = 1.0 + 4 * 4.3                                 # cm, as processes()
    n = len(VERTICES)
    fig = plt.figure(figsize=(width_in, width_in * hv / W))
    for k, (draw, title, coupling) in enumerate(VERTICES):
        wv = W / n
        ax = panel(fig, [k / n, 0, 1 / n, 1], wv, hv)
        draw(ax, wv / 2 + 0.2, hv / 2 - 0.05)
        ax.text(wv / 2, hv - 0.05, title, ha="center", va="top", fontsize=TITLE - 1, linespacing=1.05)
        ax.text(wv / 2, 0.2, coupling, ha="center", va="bottom", fontsize=TITLE - 1.5, color=RELIC)
    return fig
