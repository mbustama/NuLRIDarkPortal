"""preference_figure.py -- the paper's figure of IceCube's preference for an MeV mediator (Sec. V.5; replaces the
watermarked mock-up of analysis/mockups.fig7, 2026-11-22).

Per mass, the preference is chi^2 without a new interaction minus the lowest chi^2 over the coupling, for each
present-day data set, with the source composition fitted. Data: output/analysis/preference_20261122.json, written by
tools/collect_preference.py from the final planes. One example per family, in the families' line styles; the data
sets in the colors of the limit figures (analysis.figure_style.DATASETS). Called from notebooks/paper_figures.ipynb,
after analysis.figure_style.use().
"""
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "output", "analysis", "preference_20261122.json")

#: (plane, label, line style): one example per family.
FAMILIES = [("le_lmu_mev_fx", r"$L_e - L_\mu$", (0, (5, 2.5))),
            ("dp_tau_mev_fx", r"Dark portal, $\nu_\tau$", "-"),
            ("si_scalar_majorana_tau_fx", r"Secret, scalar, $\nu_\tau$", (0, (1, 1.3)))]
XLIM, YLIM = (4.5, 7.5), (0.0, 5.0)   # his choice 2026-11-22: stop at 5
SIGMAS = (1, 2)


def load(path=DATA):
    """{data set: {plane: (log10 M array, preference array, log10 g at the lowest chi^2)}}, masses ascending."""
    raw = json.load(open(path))
    out = {}
    for ds, planes in raw.items():
        if ds.startswith("_"):
            continue
        for plane, rec in planes.items():
            xs = sorted(rec["columns"], key=float)
            out.setdefault(ds, {})[plane] = (np.array([float(x) for x in xs]),
                                             np.array([rec["columns"][x]["preference"] for x in xs]),
                                             np.array([rec["columns"][x]["log10_g_best"] for x in xs]))
    return out


def peak(curves, ds, plane):
    """(largest preference, its log10 M, its log10 g) of one plane."""
    x, p, g = curves[ds][plane]
    k = int(np.argmax(p))
    return float(p[k]), float(x[k]), float(g[k])


def figure(path=DATA):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from analysis.figure_style import DATASETS, PANEL, ticks
    curves = load(path)
    fig, ax = plt.subplots(figsize=(1.45 * PANEL[0], 1.45 * PANEL[0]))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.12)
    present = [d for d in DATASETS if d[0] in curves]
    for ds, _, color in present:
        for plane, _, ls in FAMILIES:
            x, p, _ = curves[ds][plane]
            ax.plot(x, p, color=color, ls=ls, lw=2.0)
    # local significance at one mass, one parameter (the coupling): Delta chi^2 = n^2 for n sigma (his request)
    for n in SIGMAS:
        ax.axhline(n * n, color="0.4", lw=1.1, ls=(0, (1, 2)), zorder=1)
        ax.text(XLIM[1] - 0.05, n * n + 0.1, rf"${n}\sigma$", fontsize=15, color="0.3", ha="right", va="bottom")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ticks(ax, (0.5, 0.125), (1, 0.25))
    ax.set_xlabel(r"Mediator mass, $\log_{10}(M/{\rm eV})$")
    ax.set_ylabel(r"Preference for a new interaction, $\Delta\chi^2$")
    hd = [Line2D([], [], color=c, lw=2.5, label=lab) for _, lab, c in present]
    hd += [Line2D([], [], color="none", label=" ")]     # keeps the data sets in one column, the families in the other
    hd += [Line2D([], [], color="0.15", lw=2.0, ls=ls, label=lab) for _, lab, ls in FAMILIES]
    # above the panel: with the axis stopping at 5, no corner inside is free of curves
    leg = ax.legend(handles=hd, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.0), mode="expand", ncol=2,
                    fontsize=13, frameon=True, framealpha=1, edgecolor="black", borderaxespad=0.0)
    leg.get_frame().set_linewidth(1.0)
    return fig
