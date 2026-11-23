"""
event_rates_figure.py -- IceCube's event rates with and without a new interaction, for the examples of the
ingredients figure (analysis/three_classes_figure.py), at the same sources (2026-11-22, his request: "a plot showing
event rates with and without the effects", option (a), the examples of Fig. 7, the same sources).

Rows: the medium-energy starting cascades (MESE, the release's histogram in reconstructed energy) and the through-going
tracks (IceTracks-DR2, IC79 + IC86, summed over the declinations above -5 deg, reconstructed energy above 10^3.5 GeV).
Columns: L_e - L_mu (1e-20 eV, g' = 1e-21), the dark portal on nu_tau (1e-3 eV, g_D |U_tau4|^2 = 5.62e-7), the scalar
of secret interactions on nu_tau (10 MeV, g = 0.18, Majorana). In each panel: the observed events, the events expected
without a new interaction and with it, and the atmospheric background.

"Without" is data set 2's best fit without a new interaction (its stored fit.json: norm_astro 1.155, indices 2.715 and
1.793 about the 24.5 TeV break, norm_gal 0.906, nubar:nu 0.5; chi^2 948.2012, reproduced to 1e-4 by every run); "with"
keeps those sources and the event nuisances profiled there (atmospheric, MESE conventional and muon normalizations,
DR2 normalizations), and switches the example's coupling on. The runs (Tycho, 2026-11-22, jobs 714633-5, 8 cores:
37 min, 3 min, 37 s) used each example's own checkout (nulri_set1_sm, nulri_set1_final, nulri_refit_set) and its
now_events block configuration with set 2's data (~/stage_scan/event_rates.py, a copy in
output/analysis/event_rates/event_rates.py); the results are output/analysis/event_rates/{le,dp,si}.npz.
The MESE double cascades enter the fits but are not drawn (few events).
  from analysis import event_rates_figure as erf; fig = erf.figure()       (analysis.figure_style.use() first)
"""
import os

import numpy as np

from analysis import mean_free_paths as mf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR = os.path.join(REPO, "output", "analysis", "event_rates")
KEYS = ("le", "dp", "si")
TITLES = ["$L_e - L_\\mu$\n" r"$M = 10^{-20}$ eV, $g' = 10^{-21}$",
          "Dark portal, coupled to $\\nu_\\tau$\n" r"$M = 10^{-3}$ eV, $g_D |U_{\tau 4}|^2 = 5.6 \times 10^{-7}$",
          "Secret interactions, scalar $\\phi$, Majorana $\\nu$\n" r"$M$ = 10 MeV, $g = 0.18$ (coupled to $\nu_\tau$)"]
RED, BKG = "#9b2226", "0.55"
Y_MAX = (2e3, 2e4, 275.0)  # the rows' upper y limits (his choice, 2026-11-22)
E_DIR_MIN_GEV = 1e4        # the bottom row: tracks of reconstructed energy above 10 TeV (his choice, 2026-11-22)


def dr2_edges():
    """The DR2 release's reconstructed-energy edges, log10(E / GeV)."""
    import sys
    for p in (REPO, os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from IceCubeRates import flux_adapters as FA
    return np.load(os.path.join(FA.DATA, "dr2_counts.npz"))["reco_edges_log10E"]


def dr2_dec_edges():
    """The DR2 release's declination edges (deg), the same for IC79 and IC86."""
    dr2_edges()
    from IceCubeRates import flux_adapters as FA
    z = np.load(os.path.join(FA.DATA, "dr2_counts.npz"))
    a, b = z["IC79_dec_edges_deg"], z["IC86_dec_edges_deg"]
    if not np.array_equal(a, b):
        raise ValueError("IC79 and IC86 declination bins differ")
    return a


def load(key, directory=DIR):
    """Per sample ('cascades', 'tracks'): edges (GeV), data, and the expected events without (sm) and with (ex) the
    new interaction, in total and from the astrophysical flux; the background is total minus astrophysical."""
    d = np.load(os.path.join(directory, f"{key}.npz"))
    out = {}
    tops = [str(t) for t in d["mese_topologies"]]
    n = [len(d[f"mese_{t}_edges"]) - 1 for t in tops]
    i0 = int(np.sum(n[:tops.index("cascades")]))
    sl = slice(i0, i0 + n[tops.index("cascades")])
    out["cascades"] = dict(edges=d["mese_cascades_edges"], data=d["mese_data"][sl],
                           sm=d["sm_mese_mu"][sl], ex=d["ex_mese_mu"][sl],
                           sm_astro=d["sm_mese_astro"][sl], ex_astro=d["ex_mese_astro"][sl])
    le = dr2_edges()
    nE = len(le) - 1
    acc = {k: np.zeros(nE) for k in ("data", "sm", "ex", "sm_astro", "ex_astro")}
    for c in [str(c) for c in d["dr2_configs"]]:
        idx = np.nonzero(d[f"dr2_{c}_select"])[1]
        for k, v in (("data", d[f"dr2_{c}_data"]), ("sm", d[f"sm_dr2_{c}_mu"]), ("ex", d[f"ex_dr2_{c}_mu"]),
                     ("sm_astro", d[f"sm_dr2_{c}_astro"]), ("ex_astro", d[f"ex_dr2_{c}_astro"])):
            np.add.at(acc[k], idx, v)
    keep = np.flatnonzero(acc["sm"] > 0)
    lo, hi = keep[0], keep[-1] + 1
    out["tracks"] = dict(edges=10.0 ** le[lo:hi + 1], **{k: v[lo:hi] for k, v in acc.items()})
    # the same tracks by declination, summed over the reconstructed energies (the release's declination bins, equal in
    # sin(dec); IC79 and IC86 share them)
    de = dr2_dec_edges()
    nD = len(de) - 1
    acc = {k: np.zeros(nD) for k in ("data", "sm", "ex", "sm_astro", "ex_astro")}
    for c in [str(c) for c in d["dr2_configs"]]:
        rows, cols = np.nonzero(d[f"dr2_{c}_select"])
        high = le[cols] >= np.log10(E_DIR_MIN_GEV) - 1e-9          # bins whose lower edge is at or above the cut
        for k, v in (("data", d[f"dr2_{c}_data"]), ("sm", d[f"sm_dr2_{c}_mu"]), ("ex", d[f"ex_dr2_{c}_mu"]),
                     ("sm_astro", d[f"sm_dr2_{c}_astro"]), ("ex_astro", d[f"ex_dr2_{c}_astro"])):
            np.add.at(acc[k], rows[high], v[high])
    keep = np.flatnonzero(acc["sm"] > 0)
    lo, hi = keep[0], keep[-1] + 1
    out["directions"] = dict(edges=de[lo:hi + 1], **{k: v[lo:hi] for k, v in acc.items()})
    return out


#: The insets: (parent-axes rectangle [x0, y0, w, h] in axes fractions, x range, y range).
INSET_TRACKS = ([0.51, 0.36, 0.45, 0.46], (3e4, 5e5), (0.5, 100.0))        # the tracks near 100 TeV
INSET_DIR = ([0.42, 0.30, 0.54, 0.44], (-4.59, 20.0), (90.0, 265.0))       # the tracks near the horizon


def _inset(ax, s, spec, log):
    """A zoom into ax: the same curves and points on spec's ranges, the zoomed region marked in ax with leaders."""
    rect, (x0, x1), (y0, y1) = spec
    ins = ax.inset_axes(rect)
    e = s["edges"]
    xc = np.sqrt(e[:-1] * e[1:]) if log else 0.5 * (e[:-1] + e[1:])
    ins.stairs(s["sm"] - s["sm_astro"], e, color=BKG, lw=1.2, ls=(0, (1, 1.5)))
    ins.stairs(s["sm"], e, color="k", lw=1.5)
    ins.stairs(s["ex"], e, color=RED, lw=1.5)
    m = s["data"] > 0
    ins.errorbar(xc[m], s["data"][m], yerr=np.sqrt(s["data"][m]), fmt="o", ms=3, color="k", lw=0.9, capsize=0,
                 zorder=5)
    if log:
        ins.set_xscale("log")
        ins.set_yscale("log")
    ins.set_xlim(x0, x1)
    ins.set_ylim(y0, y1)
    ins.tick_params(labelsize=12)
    ins.set_facecolor("white")
    ax.indicate_inset_zoom(ins, edgecolor="0.3", lw=0.8)    # the zoomed region and leaders to the inset
    if not log:                                               # the declination inset: minor ticks every 10 and 2
        from matplotlib.ticker import MultipleLocator
        ins.yaxis.set_minor_locator(MultipleLocator(10.0))
        ins.xaxis.set_minor_locator(MultipleLocator(2.0))
    return ins


def figure(directory=DIR):
    """Three rows by three columns (the examples): the starting cascades and the through-going tracks versus
    reconstructed energy (log axes), and the tracks versus declination summed over energies (linear axes); observed
    events with Poisson errors, expected without (black) and with (red) the new interaction, the background (gray,
    dotted)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import FixedLocator, FuncFormatter
    from analysis.figure_style import PANEL

    data = {k: load(k, directory) for k in KEYS}
    fig, axes = plt.subplots(3, 3, sharex="row", sharey="row", figsize=(3 * 1.05 * PANEL[0], 3.1 * PANEL[1]))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.895, bottom=0.10, hspace=0.10, wspace=0.05)
    for ax in axes[2]:                       # room under the centre row for its x label (the top row has none)
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 - 0.045, p.width, p.height])
    rows = (("cascades", "Starting cascades (MESE)"), ("tracks", "Through-going tracks (DR2)"))
    for c, k in enumerate(KEYS):            # bottom row: the tracks' directions, summed over energies
        ax = axes[2, c]
        s = data[k]["directions"]
        e = s["edges"]
        xc = 0.5 * (e[:-1] + e[1:])
        ax.stairs(s["sm"] - s["sm_astro"], e, color=BKG, lw=1.4, ls=(0, (1, 1.5)))
        ax.stairs(s["sm"], e, color="k", lw=1.8)
        ax.stairs(s["ex"], e, color=RED, lw=1.8)
        ax.errorbar(xc, s["data"], yerr=np.sqrt(s["data"]), fmt="o", ms=4, color="k", lw=1.0, capsize=0, zorder=5)
        ax.set_xlim(e[0], e[-1])
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0.0, 91.0, 10.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if round(v) in (0, 30, 60) else ""))
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(-5.0, 91.0, 5.0)))
        ax.tick_params(labelsize=18)
        ax.text(0.96, 0.94, "Through-going tracks (DR2),\nsummed above 10 TeV", transform=ax.transAxes, ha="right",
                va="top", fontsize=15, linespacing=1.2)
        ax.yaxis.set_minor_locator(FixedLocator(np.arange(0.0, Y_MAX[2] + 1.0, 10.0)))
        _inset(ax, s, INSET_DIR, log=False)
    axes[2, 0].set_ylim(0.0, Y_MAX[2])
    for r, (sample, name) in enumerate(rows):
        for c, k in enumerate(KEYS):
            ax = axes[r, c]
            s = data[k][sample]
            e = s["edges"]
            xc = np.sqrt(e[:-1] * e[1:])
            ax.stairs(s["sm"] - s["sm_astro"], e, color=BKG, lw=1.4, ls=(0, (1, 1.5)))
            ax.stairs(s["sm"], e, color="k", lw=1.8)
            ax.stairs(s["ex"], e, color=RED, lw=1.8)
            m = s["data"] > 0
            ax.errorbar(xc[m], s["data"][m], yerr=np.sqrt(s["data"][m]), fmt="o", ms=4, color="k", lw=1.0,
                        capsize=0, zorder=5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.tick_params(labelsize=18)
            ax.text(0.96, 0.94, name, transform=ax.transAxes, ha="right", va="top", fontsize=15)
            if sample == "tracks":
                _inset(ax, s, INSET_TRACKS, log=True)
        lo = min(float(np.min(data[k][sample]["sm"][data[k][sample]["sm"] > 0])) for k in KEYS)
        axes[r, 0].set_ylim(max(0.3 * lo, 3e-2), Y_MAX[r])
        e0, e1 = data[KEYS[0]][sample]["edges"][0], data[KEYS[0]][sample]["edges"][-1]
        for ax in axes[r]:
            ax.set_xlim(e0, e1)
            # major ticks only inside the panel, so that the labels of neighbouring panels do not meet
            ax.xaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(3, 9) if e0 < 10.0 ** k < e1]))
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    for r, label in ((1, r"Reconstructed energy [GeV]"), (2, r"Declination [deg]")):
        mf.common_xlabel(fig, axes[r], label)
    # one y label for the two top rows, and the bottom row's at the same horizontal position
    fig.canvas.draw()
    ren, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    x_lab = min(inv.transform(ax.get_tightbbox(ren).p0)[0] for ax in axes[:, 0]) - 0.006
    for rows_, y in (((0, 1), 0.5 * (axes[1, 0].get_position().y0 + axes[0, 0].get_position().y1)),
                     ((2,), 0.5 * (axes[2, 0].get_position().y0 + axes[2, 0].get_position().y1))):
        fig.text(x_lab, y, "Events per bin", rotation=90, ha="right", va="center", fontsize=22)
    # the examples above their columns, in rounded boxes as wide as the panels (as Figs. 4 and 5)
    fig.canvas.draw()
    ren, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    gap, pad = 0.008, 0.01
    texts = []
    for ax, title in zip(axes[0], TITLES):
        pos = ax.get_position()
        texts.append(fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + gap + pad, title, ha="center", va="bottom",
                              fontsize=16, linespacing=1.3, multialignment="center", zorder=3))
    fig.canvas.draw()
    top = max(inv.transform(t.get_window_extent(ren).p1)[1] for t in texts) + pad
    for ax in axes[0]:
        pos = ax.get_position()
        fig.patches.append(FancyBboxPatch((pos.x0, pos.y1 + gap), pos.x1 - pos.x0, top - pos.y1 - gap,
                                          boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                          facecolor="white", edgecolor="black", lw=1.0, zorder=2,
                                          mutation_aspect=fig.get_figwidth() / fig.get_figheight()))
    L = lambda **kw: plt.Line2D([], [], **kw)
    handles = [L(color="k", marker="o", ls="none", ms=5, label="IceCube data"),
               L(color="k", lw=1.8, label="Expected, no new interaction"),
               L(color=RED, lw=1.8, label="Expected, with the new interaction"),
               L(color=BKG, lw=1.4, ls=(0, (1, 1.5)), label="Atmospheric background")]
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, 0.997, x1 - x0, 0), mode="expand",
                     borderaxespad=0.0, ncol=4, fontsize=16, frameon=True, framealpha=1, edgecolor="black",
                     handlelength=2.0, columnspacing=1.6)
    leg.get_frame().set_linewidth(1.0)
    return fig
