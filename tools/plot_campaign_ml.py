"""
plot_campaign_ml.py -- the diagnostic figures of the campaign_ml planes (the paper's are drawn in
notebooks/paper_figures.ipynb): per physics model and data set, the excluded regions at 95% (analysis/exclusion.py:
per-mass Delta chi^2 >= 3.84, upper limits only, specks and thin bands left out), all seven models on one page, one
figure with the source compositions fixed to pion decay and one with them fitted; and the thin bands the fitted planes
also exclude, as ribbons at their true width (notes, 'Thin excluded bands with fitted source compositions'). Drawn in
Mauricio's style (analysis/figure_style.py). Reads the plane summaries under output/analysis/campaign_ml/.
  python tools/plot_campaign_ml.py [output prefix, default output/figures/campaign_ml]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from analysis import exclusion  # noqa: E402
from analysis.exclusion import field, thin_bands  # noqa: E402
from analysis.figure_style import DATASETS, PANEL, pct, ticks, tint, use  # noqa: E402
MODELS = [("le_lmu_ul", r"$L_e - L_\mu$, ultralight"), ("le_ltau_ul", r"$L_e - L_\tau$, ultralight"),
          ("le_lmu_mev", r"$L_e - L_\mu$, MeV"), ("le_ltau_mev", r"$L_e - L_\tau$, MeV"),
          ("dp_e_mev", r"dark portal, $e$ texture, MeV"), ("dp_mu_mev", r"dark portal, $\mu$ texture, MeV"),
          ("dp_tau_mev", r"dark portal, $\tau$ texture, MeV")]
SETS = DATASETS
TITLE = "Disfavored (95{} C.L.)"
TITLE_COMP = {"": r"fixing $f_{e,{\rm S}} = 1/3$", "_fx": r"fitting $f_{e,{\rm S}}$"}   # the source compositions
DRAWN = SETS[::-1]
YLABEL = {"g_D": r"$g_D$", "g_D_trace": r"$g_D$ (texture trace)"}
COMPS = [("", "pion", "source compositions fixed to pion decay"), ("_fx", "fitted", "source compositions fitted")]


INSET = ((-25, -15), (-28.5, -24))   # log10 m, log10 g: the thin bands' flat stretch, enlarged


def draw(ax, lx, G, F, ribbons, color, lw=1.8):
    exclusion.draw(ax, lx, G, F, color, tint(color), lw=lw, ribbons=ribbons)


def figure(suffix, comp_title, out, bands=False):
    """The exclusion figure (7 models, thin bands left out); bands: only the ultralight planes, with their thin bands
    as ribbons and an inset on their flat stretch."""
    models = MODELS[:2] if bands else MODELS
    nr, nc = (1, 3) if bands else (2, 4)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * PANEL[0], nr * PANEL[1] + 0.5))
    for ax, (model, title) in zip(axes.flat, models):
        yname, drawn = None, []
        for ds, label, color in DRAWN:
            hits = glob.glob(f"{REPO}/output/analysis/campaign_ml/{ds}/{model}{suffix}/*/plane_*.json")
            if not hits:
                continue
            lx, G, F, yname, C = field(hits[0])
            ribbons = []
            if model.endswith("_ul"):
                F, ribbons = thin_bands(lx, G, F, C)
            draw(ax, lx, G, F, ribbons if bands else [], color)
            drawn.append((lx, G, F, ribbons, color))
        if bands:                                        # the thin bands, enlarged
            ins = ax.inset_axes([0.57, 0.06, 0.41, 0.38])
            for d in drawn:
                draw(ins, *d, lw=1.2)
            ins.set_xlim(*INSET[0])
            ins.set_ylim(*INSET[1])
            ticks(ins, (4, 1), (2, 0.5))
            ins.tick_params(labelsize=13, pad=3, length=5)
            ins.tick_params(which="minor", length=3)
            ax.indicate_inset_zoom(ins, edgecolor="0.45")
        if model.endswith("_mev"):
            ax.set_ylim(-6, -2)                          # nothing is excluded below ~1e-5 in the MeV planes
            ticks(ax, (1, 0.25), (1, 0.25))
        else:
            ticks(ax, (10, 2), (10, 2))
        ax.set_xlabel(r"$\log_{10}(m_{Z_D}$ / eV$)$")
        ax.set_ylabel(r"$\log_{10}$ " + YLABEL.get(yname, yname or "coupling"))
        ax.set_title(title)
    leg_ax = axes.flat[-1]
    leg_ax.axis("off")
    handles = [Patch(facecolor=tint(c), edgecolor=c, lw=1.5, label=lab) for _, lab, c in SETS]
    leg_ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.62), frameon=True, edgecolor="black",
                  title=TITLE.format(pct()) + ", " + TITLE_COMP[suffix] + "\n" + r"(per mass, $\Delta\chi^2 \geq 3.84$)",
                  title_fontsize=plt.rcParams["legend.fontsize"])
    if suffix and not bands:
        leg_ax.text(0.5, 0.1, "Not shown: thin excluded bands along the edge of the\npion-decay region in the "
                    "ultralight planes, which\nexist only for energy-independent source compositions.",
                    ha="center", va="center", fontsize=12, color="0.35", transform=leg_ax.transAxes)
    fig.suptitle((f"Thin excluded bands, {comp_title} (ribbons at their true width; inset enlarged)" if bands else
                  f"Campaign of 2026-10-20, {comp_title}: excluded couplings per mediator mass; mixing fixed, lightest "
                  "mass profiled, relics thermal in every collision"))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def main(prefix):
    use()
    for suffix, tag, comp_title in COMPS:
        figure(suffix, comp_title, f"{prefix}_excluded_{tag}.png")
    figure(COMPS[1][0], COMPS[1][2], f"{prefix}_thin_bands.png", bands=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "figures", "campaign_ml"))
