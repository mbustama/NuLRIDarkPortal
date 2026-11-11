"""
mockups.py -- mock-ups of the paper's result figures (2026-11-08, Mauricio: "mock-up versions (with watermark saying
that) of the figures that can be done like that"), drawn from the latest stored results, which are NOT the final ones:
the October campaign (output/analysis/campaign_ml: before the decay back, the SM potential in the Earth, Table 10's
spectrum data and the rho fix) and the first secret-interaction pilot (output/analysis/si_pilot1). Every figure carries
a watermark. They show the layout and roughly where the regions lie; the final figures come from
notebooks/paper_figures.ipynb once the runs finish.

  python -m analysis.mockups            writes paper/figures/mockup_fig{1,4,5,6,7}.pdf
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from . import REPO_DIR
from . import exclusion as ex
from . import existing_bounds as eb
from .figure_style import DATASETS, PANEL, pct, ticks, tint, use

CAMPAIGN = os.path.join(REPO_DIR, "output", "analysis", "campaign_ml")
SI_PILOT = os.path.join(REPO_DIR, "output", "analysis", "si_pilot1")
FIGURE_DIR = os.path.join(REPO_DIR, "paper", "figures")
DRAWN = DATASETS[::-1]
WATERMARK = "MOCK-UP: preliminary results"


def watermark(fig, text=WATERMARK):
    w, h = fig.get_size_inches()
    diag = float(np.hypot(w, h))
    # the text across ~80% of the figure's diagonal (a character ~0.5 em wide)
    fs = 0.80 * diag * 72.0 / (0.5 * len(text))
    fig.text(0.5, 0.5, text, rotation=float(np.degrees(np.arctan2(h, w))), ha="center", va="center", fontsize=fs,
             color="#c00000", alpha=0.13, zorder=100, transform=fig.transFigure, fontweight="bold")


def model_label(ax, text, size=22, where="top"):
    y, va = (0.94, "top") if where == "top" else (0.05, "bottom")
    ax.text(0.015, y, text, transform=ax.transAxes, ha="left", va=va, fontsize=size,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.25", lw=1.2))


def handles():
    return [Patch(facecolor=tint(c), edgecolor=c, lw=1.5, label=label) for _, label, c in DATASETS]


def legend(ax, loc, anchor, title):
    leg = ax.legend(handles=handles(), loc=loc, bbox_to_anchor=anchor, borderaxespad=0.0, fontsize=15, frameon=True,
                    framealpha=1, edgecolor="black", title=title, title_fontsize=16)
    leg.get_frame().set_linewidth(1.0)


def bounds_gray(ax, model, label_xy=None):
    """The drawn existing bounds of `model` as one light-gray region: above the lowest of their curves at each mass
    (their union), with one label in place (orientation only; Appendix H lists them)."""
    xs = np.linspace(-31, 10, 1200)
    low = np.full(len(xs), np.inf)
    for key in eb.for_model(model):
        b = eb.BOUNDS[key]
        if not b.get("draw") or b["kind"] != "curve":
            continue
        try:
            M, g = eb.load(key, model)
        except (OSError, KeyError, ValueError):
            continue
        lx, lg = np.log10(M), np.log10(g)
        o = np.argsort(lx)
        inside = (xs >= lx[o][0]) & (xs <= lx[o][-1])
        low[inside] = np.minimum(low[inside], np.interp(xs[inside], lx[o], lg[o]))
    ok = np.isfinite(low)
    ax.fill_between(xs, np.where(ok, low, np.nan), 5.0, where=ok, color="0.88", lw=0, zorder=0)
    ax.plot(xs, np.where(ok, low, np.nan), color="0.55", lw=1.0, zorder=0.5)
    if label_xy:
        ax.text(*label_xy, "Existing bounds", fontsize=16, color="0.35", ha="center", va="center", zorder=1)


def save(fig, name):
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"))
    plt.close(fig)
    print("saved paper/figures/" + name + ".pdf")


def fig1():
    """Money plot's mock-up: L_e - L_mu, HESE + DR2, the full calculation only (the simplified calculations are not
    run yet), existing bounds in gray."""
    fig, ax = plt.subplots(figsize=(1.45 * PANEL[0], 1.45 * PANEL[0]))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.13)
    bounds_gray(ax, "le_lmu", label_xy=(-21.0, -8.0))
    lx, G, F = ex.merged(ex.plane(CAMPAIGN, "now_events", "le_lmu_ul"), ex.plane(CAMPAIGN, "now_events", "le_lmu_mev"))
    c = DATASETS[1][2]
    ex.draw(ax, lx, G, F, c, tint(c), lw=2.2)
    ax.annotate("Disfavored at 95\\% C.L.:\nfull calculation,\nIceCube HESE + DR2", xy=(-15.2, -24.9),
                xytext=(-8.6, -24.9), fontsize=14,
                color="#8a5a00", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#8a5a00", lw=1.2, shrinkA=3, shrinkB=0))
    ax.text(4.0, -12.0, "Simplified\ncalculations:\npending", fontsize=14, color="0.35", ha="center", va="center")
    ax.set_xlim(-30, 9)
    ax.set_ylim(-32, 0)
    ticks(ax, (5, 1), (5, 1))
    ax.set_xlabel(r"Mediator mass, $\log_{10}(M/{\rm eV})$")
    ax.set_ylabel(r"Coupling, $\log_{10} g'$")
    model_label(ax, r"$L_e - L_\mu$", where="bottom")
    ax.texts[-1].set_position((0.70, 0.05))
    watermark(fig)
    return fig


def _merged_panels(models, names, ylabel, xlim=(-30, 9), ylim=(-32, -2), label_where="top",
                   leg=("lower right", (0.985, 0.04)), yticks=(5, 1), ylabels=None):
    fig, axes = plt.subplots(len(models), 1, sharex=True,
                             figsize=(3.5 * PANEL[0], len(models) * 0.85 * PANEL[1] + 1.0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.11 if len(models) == 2 else 0.085, hspace=0.06)
    for ax, model, name in zip(axes, models, names):
        for ds, _, color in DRAWN:
            ul, mev = ex.plane(CAMPAIGN, ds, f"{model}_ul"), ex.plane(CAMPAIGN, ds, f"{model}_mev")
            if not (ul and mev):
                continue
            lx, G, F = ex.merged(ul, mev)
            ex.draw(ax, lx, G, F, color, tint(color), lw=2.0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ticks(ax, (5, 1), yticks)
        if ylabels is not None:
            ax.set_yticks(ylabels)
        model_label(ax, name, where=label_where)
    axes[-1].set_xlabel(r"Mediator mass, $\log_{10}(M/{\rm eV})$")
    fig.text(0.012, 0.545, ylabel, rotation=90, ha="left", va="center", fontsize=plt.rcParams["axes.labelsize"])
    legend(axes[0], leg[0], leg[1], f"Disfavored (95{pct()} C.L.), pion-decay sources")
    watermark(fig)
    return fig


def fig4():
    """L_e - L_alpha's mock-up: the notebook's figure (both data planes joined across 1 keV, the MeV corner inset)."""
    joined = [("le_lmu", r"$L_e - L_\mu$"), ("le_ltau", r"$L_e - L_\tau$")]
    inset_box, corner = [0.735, 0.13, 0.205, 0.52], ((2.5, 7.5), (-5, -2))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5 * PANEL[0], 2 * PANEL[1] + 1.0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.98, bottom=0.11, hspace=0.06)
    for ax, (model, name) in zip(axes, joined):
        regions = []
        for ds, _, color in DRAWN:
            lx, G, F = ex.merged(ex.plane(CAMPAIGN, ds, f"{model}_ul"), ex.plane(CAMPAIGN, ds, f"{model}_mev"))
            regions.append((lx, G, F, color))
            ex.draw(ax, lx, G, F, color, tint(color), lw=2.0)
        ins = ax.inset_axes(inset_box)
        for lx, G, F, color in regions:
            ex.draw(ins, lx, G, F, color, tint(color), lw=1.5)
        ins.set_xlim(*corner[0])
        ins.set_ylim(*corner[1])
        ticks(ins, (1, 0.25), (1, 0.25))
        ins.tick_params(labelsize=18, pad=4, length=6, labelleft=False, labelright=True)
        ins.tick_params(which="minor", length=3)
        ax.indicate_inset_zoom(ins, edgecolor="0.45")
        ax.set_xlim(-30, 9)
        ax.set_ylim(-32, -2)
        ticks(ax, (5, 1), (5, 1))
        model_label(ax, name)
    axes[1].set_xlabel(r"Mediator mass, $\log_{10}(M/{\rm eV})$")
    fig.text(0.012, 0.545, r"Coupling, $\log_{10} g'$", rotation=90, ha="left", va="center",
             fontsize=plt.rcParams["axes.labelsize"])
    legend(axes[0], "lower right", (inset_box[0] - 0.012, inset_box[1]), f"Disfavored (95{pct()} C.L.), pion-decay sources")
    watermark(fig)
    return fig


def fig5():
    return _merged_panels(["dp_e", "dp_mu", "dp_tau"],
                          [r"Dark portal, coupled to $\nu_e$", r"Dark portal, coupled to $\nu_\mu$",
                           r"Dark portal, coupled to $\nu_\tau$"], r"Coupling, $\log_{10} g_D$", ylim=(-5, -0.5),
                          label_where="bottom", leg=("lower center", (0.45, 0.06)),
                          yticks=(1, 0.25), ylabels=[-4, -3, -2, -1])


def fig6():
    """Secret interactions' mock-up: the one plane of the first pilot with a summary (scalar, Majorana, nu_mu,
    IceCube spectrum, below 1 keV); the rest of the run is pending."""
    paths = glob.glob(os.path.join(SI_PILOT, "now_ic26", "*", "*", "plane_*.json"))
    fig, ax = plt.subplots(figsize=(1.45 * PANEL[0], 1.45 * PANEL[0]))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.13)
    c = DATASETS[0][2]
    for p in paths:
        lx, G, F, _, _ = ex.field(p)
        ex.draw(ax, lx, G, F, c, tint(c), lw=2.2)
    ax.text(-5.5, -7.0, "Scalar, Majorana,\ncoupled to $\\nu_\\mu$:\nIceCube spectrum + flavor", fontsize=15, color=c,
            ha="center", va="top")
    ax.text(-20, -9.0, "Other models and\ndata sets: pending", fontsize=14, color="0.35", ha="center", va="center")
    ax.set_xlim(-30, 3)
    ax.set_ylim(-12, 0)
    ticks(ax, (5, 1), (2, 0.5))
    ax.set_xlabel(r"Mediator mass, $\log_{10}(M_\phi/{\rm eV})$")
    ax.set_ylabel(r"Coupling, $\log_{10} g_\phi$")
    model_label(ax, "Secret interactions", where="bottom")
    watermark(fig)
    return fig


def fig7():
    """IceCube's MeV preference's mock-up: per mass, how much lower the best coupling's chi^2 is than the fit without
    a new interaction (the column's no-effect value, its smallest coupling), for the dark portal's three textures."""
    fig, ax = plt.subplots(figsize=(1.45 * PANEL[0], 1.45 * PANEL[0]))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.96, bottom=0.13)
    styles = {"dp_e_mev": ("-", r"coupled to $\nu_e$"), "dp_mu_mev": ("--", r"coupled to $\nu_\mu$"),
              "dp_tau_mev": (":", r"coupled to $\nu_\tau$")}
    c = DATASETS[0][2]
    for model, (ls, label) in styles.items():
        s = json.load(open(ex.plane(CAMPAIGN, "now_ic26", model)))
        x = np.log10(np.array(s["grids"][0]))
        chi = np.array([[np.nan if v is None else v for v in row] for row in s["chi2"]], dtype=float)
        pref = chi[:, 0] - np.nanmin(chi, axis=1)
        ax.plot(x, pref, ls=ls, color=c, lw=1.8, marker="o", ms=3.5, label=label)
    ax.axhline(3.84, color="0.4", lw=1.0, ls="-.")
    ax.text(8.9, 3.84 + 0.15, r"95\% C.L.", fontsize=14, color="0.3", ha="right")
    ax.set_xlim(4, 9)
    ax.set_ylim(0, 9.5)
    ticks(ax, (1, 0.25), (1, 0.5))
    ax.set_xlabel(r"Mediator mass, $\log_{10}(M_{Z_D}/{\rm eV})$")
    ax.set_ylabel(r"Preference for a new interaction, $\Delta\chi^2$")
    leg = ax.legend(loc="upper right", fontsize=14, frameon=True, framealpha=1, edgecolor="black",
                    title="Dark portal, IceCube spectrum + flavor", title_fontsize=14)
    leg.get_frame().set_linewidth(1.0)
    watermark(fig)
    return fig


def main():
    use()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for name, f in (("mockup_fig1", fig1), ("mockup_fig4", fig4), ("mockup_fig5", fig5), ("mockup_fig6", fig6),
                    ("mockup_fig7", fig7)):
        try:
            save(f(), name)
        except Exception as e:                       # a missing plane: the other mock-ups still made
            print(f"{name}: not made ({e!r})")


if __name__ == "__main__":
    main()
