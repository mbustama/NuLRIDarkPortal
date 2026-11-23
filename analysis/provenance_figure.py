"""provenance_figure.py -- where the L_e - L_mu limits of the earlier estimate (Agarwalla:2023sng) and ours part
(2026-11-22, his request "separate [method from data], so we can explain"; "start polishing that plot").

Curves (pion-decay sources, set-1 data, 95% C.L.): the earlier estimate as published; its method (the potential at the
detector) with our data and fit; our transport without decays or collisions on the relics; our full calculation.
Arrows at one mass walk from the first to the last. Data: the tables of tools/provenance_lelmu.py (planes in
output/analysis/provenance_lelmu/), the paper's le_lmu_ul limits, and the digitized earlier estimate. Lines only (his
rule, 2026-11-22: "There is no need to use markers"). Called after analysis.figure_style.use().
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XLIM, YLIM = (-26.0, -10.0), (-31.0, -8.0)   # the masses computed at every step
GRAY, BLUE, RED, TEAL = "0.35", "#4477aa", "#9b2226", "#0a9396"
WALK_AT = -15.0      # the mass (log10 eV) where the arrows walk from the earlier estimate to ours
FS_LABEL, FS_NOTE = 15, 13
X_LABELS = -22.9     # the common left edge of the horizontal labels (his request, 2026-11-22)


def load():
    """(rows of tools.provenance_lelmu.table(), the full limit's columns, the earlier estimate's curve, the
    perturbative-unitarity line (slope, intercept))."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import provenance_lelmu as pl
    full = json.load(open(pl.FULL))
    line = full["not_modelled"]["not_perturbative"]
    return pl.table(), full["columns"], np.loadtxt(pl.EARLIER), (float(line["slope"]), float(line["intercept"]))


def _white_edge(width=3.0):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=width, foreground="white")]


def label_along(ax, x, y, slope, text, color, offset_pt, fontsize=FS_LABEL):
    """`text` centered at data (x, y) on a line of data slope `slope`, rotated to the line's angle on the page and
    moved `offset_pt` points along its normal (positive: above the line, negative: below)."""
    p0 = ax.transData.transform((x, y))
    p1 = ax.transData.transform((x + 1.0, y + slope))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    nx, ny = -np.sin(np.radians(ang)), np.cos(np.radians(ang))
    return ax.annotate(text, (x, y), xytext=(offset_pt * nx, offset_pt * ny), textcoords="offset points",
                       rotation=ang, rotation_mode="anchor", ha="center", va="center", color=color,
                       fontsize=fontsize, zorder=10, path_effects=_white_edge())


def _arrow(ax, x, y0, y1, color, text, side="right"):
    ax.annotate("", (x, y1), (x, y0), zorder=9,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=1.5, shrinkB=1.5, mutation_scale=14))
    ax.annotate(text, (x, 0.5 * (y0 + y1)), xytext=(6 if side == "right" else -6, 0), textcoords="offset points",
                ha="left" if side == "right" else "right", va="center", color=color, fontsize=FS_NOTE, zorder=10,
                linespacing=1.15, path_effects=_white_edge())


def figure():
    import matplotlib.pyplot as plt
    from analysis import exclusion as ex
    from analysis.figure_style import PANEL, ticks
    rows, full, earlier, (slope, icpt) = load()
    fig, ax = plt.subplots(figsize=(1.45 * PANEL[0], 1.45 * PANEL[0]))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.11)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    # perturbative unitarity fails above this line: hatched as in the limit figures
    ex.draw_not_allowed(ax, np.linspace(*XLIM, 200), np.array(YLIM), (slope, icpt))
    ax.text(X_LABELS, -10.3, "Perturbative unitarity fails", fontsize=FS_NOTE, color="0.25", ha="left", va="center",
            zorder=10, path_effects=_white_edge(4.0))

    # the four steps
    ax.plot(earlier[:, 0], earlier[:, 1], color=GRAY, lw=2.0, ls=(0, (5, 3)), zorder=4)
    m = np.array([r["log10_M"] for r in rows])
    ax.plot(m, [r["detector_our_data"] for r in rows], color=BLUE, lw=2.0, zorder=5)
    fx = np.array([c["log10_m"] for c in full])
    fg = np.array([np.nan if c["lowest_excluded_log10_g"] is None else c["lowest_excluded_log10_g"] for c in full])
    ax.plot(fx, fg, color=RED, lw=2.6, zorder=6)
    # without decays or collisions: the lowest coupling disfavored; at the two heaviest masses nothing is disfavored up
    # to 1e-12, the largest coupling computed, so the line stops at 1e-14 eV
    nd = np.array([(r["log10_M"], r["no_decays"]) for r in rows if r["no_decays"] is not None])
    ax.plot(nd[:, 0], nd[:, 1], color=TEAL, lw=2.0, zorder=7)

    # labels along their lines
    ax.text(X_LABELS, -28.62, "Earlier estimate", color=GRAY, fontsize=FS_LABEL, ha="left", va="top",
            path_effects=_white_edge())
    ax.text(X_LABELS, -26.3, "Earlier method, our data", color=BLUE, fontsize=FS_LABEL, ha="left", va="bottom",
            path_effects=_white_edge())
    label_along(ax, -18.0, float(np.interp(-18.0, fx, fg)), 1.0, "This work", RED, -13)
    label_along(ax, -21.0, float(np.interp(-21.0, nd[:, 0], nd[:, 1])), 1.0, "Without decays or collisions", TEAL,
                -12, fontsize=FS_NOTE + 1)

    # the walk at one mass, from the earlier method with our data: the potential along the trip, then the decays
    y_det = float(np.interp(WALK_AT, m, [r["detector_our_data"] for r in rows]))
    y_nd = float(np.interp(WALK_AT, nd[:, 0], nd[:, 1]))
    y_full = float(np.interp(WALK_AT, fx, fg))
    _arrow(ax, WALK_AT, y_det, y_nd, TEAL, "Potential along\nthe whole trip")
    _arrow(ax, WALK_AT + 0.3, y_nd, y_full, RED, "Decays")

    ax.text(0.97, 0.03, r"$L_e - L_\mu$", transform=ax.transAxes, fontsize=20, ha="right", va="bottom", zorder=20,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.25", lw=1.2))
    ticks(ax, (2, 1), (5, 1))
    ax.set_xlabel(r"Mediator mass, $\log_{10}(M/{\rm eV})$")
    ax.set_ylabel(r"Coupling, $\log_{10} g'$")
    return fig
