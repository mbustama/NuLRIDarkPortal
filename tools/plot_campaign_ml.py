"""
plot_campaign_ml.py -- the campaign_ml planes (2026-10-20: mixing fixed, m_lightest profiled exactly, the relics thermal
in every MeV collision term) as excluded regions: per physics model and data set, the couplings with a per-mass
Delta chi^2 >= 3.84 (95%) shaded, edges drawn. Shading rather than one limit line (Mauricio, 2026-10-20): with the
source compositions fitted, a data set can exclude a narrow band of couplings and allow larger ones again (the future
detectors near 3e-28 in the ultralight planes). One figure with the compositions fixed to pion decay, one with them
fitted. Mass columns where the scanned couplings never reach the no-effect limit ('elevated') are left unshaded.
Reads the plane summaries under output/analysis/campaign_ml/.
  python tools/plot_campaign_ml.py [output prefix, default output/figures/campaign_ml_excluded]
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
MODELS = [("le_lmu_ul", r"$L_e - L_\mu$, ultralight"), ("le_ltau_ul", r"$L_e - L_\tau$, ultralight"),
          ("le_lmu_mev", r"$L_e - L_\mu$, MeV"), ("le_ltau_mev", r"$L_e - L_\tau$, MeV"),
          ("dp_e_mev", r"dark portal, $e$ texture, MeV"), ("dp_mu_mev", r"dark portal, $\mu$ texture, MeV"),
          ("dp_tau_mev", r"dark portal, $\tau$ texture, MeV")]
# categorical slots 1-3 of the validated default palette (dataviz skill), in fixed order
SETS = [("now_ic26", "IceCube flavor + spectrum (today)", "#2a78d6"),
        ("now_events", "HESE + DR2 events (today)", "#eb6834"),
        ("proj_future", "future detectors (projection)", "#1baf7a")]
YLABEL = {"g_D": r"$g_D$", "g_D_trace": r"$g_D$ (texture trace)"}
COMPS = [("", "pion", "source compositions fixed to pion decay"), ("_fx", "fitted", "source compositions fitted")]


def excluded(path):
    """(x, y, per-mass delta chi^2 with elevated columns masked, y name)."""
    s = json.load(open(path))
    x = np.array(s["grids"][0])
    y = np.array(s["grids"][1])
    chi = np.array([[np.nan if v is None else v for v in row] for row in s["chi2"]])
    d = chi - np.nanmin(chi, axis=1, keepdims=True)
    for v in s.get("elevated", []):
        d[int(np.argmin(np.abs(x - v)))] = np.nan
    return x, y, d, s["names"][1]


def figure(suffix, comp_title, out):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax, (model, title) in zip(axes.flat, MODELS):
        yname = None
        for ds, label, color in SETS:
            hits = glob.glob(f"{REPO}/output/analysis/campaign_ml/{ds}/{model}{suffix}/*/plane_*.json")
            if not hits:
                continue
            x, y, d, yname = excluded(hits[0])
            X, Y = np.meshgrid(np.log10(x), np.log10(y), indexing="ij")
            dd = np.where(np.isfinite(d), d, -1.0)          # masked columns never shaded
            # excluded from above only (Mauricio, 2026-10-20): couplings below a mass column's lowest allowed one are
            # disfavoured only because the data prefer a non-zero coupling there -- not an exclusion; those
            # preferred couplings are drawn as a dotted segment instead
            ok = dd < 3.84
            for i in range(len(x)):
                if not ok[i].any():
                    continue
                j0 = int(np.argmax(ok[i]))
                if j0 > 0:
                    dd[i, :j0] = -1.0
                    j1 = j0 + int(np.argmax(~ok[i, j0:])) if (~ok[i, j0:]).any() else len(y)
                    ax.plot([X[i, 0]] * 2, [np.log10(y[j0]), np.log10(y[j1 - 1])], ls=":", color=color, lw=2.2)
            ax.contourf(X, Y, dd, levels=[3.84, 1e9], colors=[color], alpha=0.22)
            ax.contour(X, Y, dd, levels=[3.84], colors=[color], linewidths=1.8)
        if model.endswith("_mev"):
            ax.set_ylim(-6, -2)                          # nothing is excluded below ~1e-5 in the MeV planes
        ax.set_xlabel(r"$\log_{10}(m_{Z_D}$ / eV$)$")
        ax.set_ylabel(r"$\log_{10}$ " + YLABEL.get(yname, yname or "coupling"))
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, color="0.9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    leg_ax = axes.flat[-1]
    leg_ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=lab) for _, lab, c in SETS]
    leg_ax.legend(handles=handles, loc="center", fontsize=9.5, frameon=False,
                  title="Excluded at 95% (shaded; per mass, " + r"$\Delta\chi^2 \geq 3.84$)", title_fontsize=9.5)
    leg_ax.text(0.5, 0.16, "Dotted segment: the couplings a data set prefers at that mass\n(there, couplings too small "
                "to have an effect are disfavoured at\n95%; not shaded: a preference, not an exclusion).\n\n"
                "Unshaded masses inside a data set's range: the scanned\ncouplings never reach the no-effect limit there.",
                ha="center", va="center", fontsize=8, color="0.35", transform=leg_ax.transAxes)
    fig.suptitle(f"Campaign of 2026-10-20, {comp_title}: excluded couplings per mediator mass; mixing fixed, lightest "
                 "mass profiled, relics thermal in every collision", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)


def main(prefix):
    for suffix, tag, comp_title in COMPS:
        figure(suffix, comp_title, f"{prefix}_{tag}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "figures", "campaign_ml_excluded"))
