"""
plot_campaign_ml.py -- summary of the campaign_ml planes (2026-10-20: mixing fixed, m_lightest profiled exactly, the
relics thermal in every MeV collision term): per physics model, the 95% upper limit on the coupling at each mediator
mass (the upper end of the mass column's lowest allowed interval at Delta chi^2 = 3.84), for each data set, with the
source compositions fixed to pion decay (solid) and fitted (dashed). Masses where the limit is open or the scanned
couplings never reach the no-effect limit are left out. Reads the plane summaries under output/analysis/campaign_ml/.
  python tools/plot_campaign_ml.py output/figures/campaign_ml_limits.png
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = [("le_lmu_ul", r"$L_e - L_\mu$, ultralight"), ("le_ltau_ul", r"$L_e - L_\tau$, ultralight"),
          ("le_lmu_mev", r"$L_e - L_\mu$, MeV"), ("le_ltau_mev", r"$L_e - L_\tau$, MeV"),
          ("dp_e_mev", r"dark portal, $e$ texture, MeV"), ("dp_mu_mev", r"dark portal, $\mu$ texture, MeV"),
          ("dp_tau_mev", r"dark portal, $\tau$ texture, MeV")]
# categorical slots 1-3 of the validated default palette (dataviz skill), in fixed order
SETS = [("now_ic26", "IceCube flavor + spectrum (today)", "#2a78d6"),
        ("now_events", "HESE + DR2 events (today)", "#eb6834"),
        ("proj_future", "future detectors (projection)", "#1baf7a")]
YLABEL = {"g_D": r"$g_D$", "g_D_trace": r"$g_D$ (texture trace)"}


def limits(path):
    s = json.load(open(path))
    x = np.array(s["grids"][0])
    lim = []
    for col in s["columns"]:
        ivs = col["intervals"].get("95%", [])
        bad = (not ivs) or (col["value"] in s.get("elevated", [])) or ivs[0]["hi_open"]
        lim.append(np.nan if bad else ivs[0]["hi"])
    return x, np.array(lim), s["names"][1]


def main(out):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax, (model, title) in zip(axes.flat, MODELS):
        yname = None
        for ds, label, color in SETS:
            for suffix, ls, comp in (("", "-", "pion sources"), ("_fx", "--", "compositions fitted")):
                hits = glob.glob(f"{REPO}/output/analysis/campaign_ml/{ds}/{model}{suffix}/*/plane_*.json")
                if not hits:
                    continue
                x, lim, yname = limits(hits[0])
                ax.plot(x, lim, ls=ls, color=color, lw=2.0, label=f"{label}, {comp}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{Z_D}$ [eV]")
        ax.set_ylabel(YLABEL.get(yname, yname or "coupling"))
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, which="major", color="0.9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg_ax = axes.flat[-1]
    leg_ax.axis("off")
    leg_ax.legend(handles, labels, loc="center", fontsize=9, frameon=False,
                  title="Excluded above each line (95%, per mass)", title_fontsize=9.5)
    fig.suptitle("Campaign of 2026-10-20: 95% upper limits on the coupling per mediator mass "
                 r"($\Delta\chi^2 = 3.84$); mixing fixed, lightest mass profiled, relics thermal in every collision",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "figures", "campaign_ml_limits.png"))
