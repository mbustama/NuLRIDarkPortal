"""plot_cmp.py -- comparison_with_petropavlova.pdf: |her total / ours - 1| for checks A-D, her numerical rates
and her analytic formulas, from the .npy files of compare.py.
  python plot_cmp.py OUTDIR_of_compare.py OUTDIR_for_the_pdf"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "analysis"))
import figure_style
figure_style.use()
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
src, out = sys.argv[1], sys.argv[2]
labels = {"A": r"Scalar $\phi$, Majorana $\nu$", "B": r"Vector $V$, Majorana $\nu$",
          "C": r"Scalar $\phi$, Dirac $\nu$", "D": r"Vector $V$, Dirac $\nu$"}
P = figure_style.PANEL
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(2.4 * P[0], 2.15 * P[1] + 1.0))
fig.subplots_adjust(left=0.13, right=0.975, top=0.9, bottom=0.1, hspace=0.05, wspace=0.05)
box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.3", lw=0.8)
for ax, k in zip(axes.flat, "ABCD"):
    E, rn, ra = np.load(f"{src}/{k}_ratios.npy")
    ax.plot(E / 1e9, np.abs(rn), color="k", lw=2.0)
    ax.plot(E / 1e9, np.abs(ra), color="#ae2012", lw=1.6, ls=(0, (5, 2.5)))
    ax.text(0.955, 0.94, labels[k], transform=ax.transAxes, ha="right", va="top", fontsize=16, bbox=box)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e4, 1e8); ax.set_ylim(1e-12, 1e0)
    ax.xaxis.set_major_locator(FixedLocator([1e5, 1e6, 1e7]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10))); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([1e-10, 1e-7, 1e-4, 1e-1]))
    ax.yaxis.set_minor_locator(FixedLocator(10.0 ** np.arange(-12, 1))); ax.yaxis.set_minor_formatter(NullFormatter())
fig.canvas.draw()
ren, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
y_t = min(inv.transform(a.get_tightbbox(ren).p0)[1] for a in axes[-1])
x_t = min(inv.transform(a.get_tightbbox(ren).p0)[0] for a in axes[:, 0])
gx = 0.5 * (axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1)
gy = 0.5 * (axes[1, 0].get_position().y0 + axes[0, 0].get_position().y1)
size = plt.rcParams["axes.labelsize"]
fig.text(gx, y_t - 0.006, r"Neutrino energy, $E$ [GeV]", ha="center", va="top", fontsize=size)
fig.text(x_t - 0.006, gy, r"$|\Gamma_{31}^{\rm Petropavlova} / \Gamma_{31}^{\rm ours} - 1|$", rotation=90, ha="right",
         va="center", fontsize=size)
h = [plt.Line2D([], [], color="k", lw=2.0, label="Her numerical rates"),
     plt.Line2D([], [], color="#ae2012", lw=1.6, ls=(0, (5, 2.5)), label="Her analytic formulas")]
x0, x1 = axes[0, 0].get_position().x0, axes[0, 1].get_position().x1
leg = fig.legend(handles=h, loc="upper left", bbox_to_anchor=(x0, 0.995, x1 - x0, 0), mode="expand", borderaxespad=0.0,
                 ncol=2, fontsize=15, frameon=True, framealpha=1, edgecolor="black", handlelength=2.6, title_fontsize=16,
                 title=r"Total rate of $\nu_3$ on the relic $\nu_1$, checks A--D: relative difference from ours")
leg.get_frame().set_linewidth(1.0)
fig.savefig(f"{out}/comparison_with_petropavlova.pdf", bbox_inches="tight", pad_inches=0.05)
print("wrote", f"{out}/comparison_with_petropavlova.pdf")
