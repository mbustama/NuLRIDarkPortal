"""
plot_exclusion_summary.py

Summary figure of the six regime-split exclusion planes: the three benchmark textures (columns)
with absorption only (epsilon' = 0, in g_D |U|^2) and with the coherent potential (epsilon' =
0.1, in g_D epsilon' |U|^2), as per-mass Delta chi^2 with the 95% line (the upper end of each
mass column's lowest allowed interval at Delta chi^2 = 3.84). Reads the plane summaries under
output/analysis/ (python -m analysis CONFIG --planes).

  python tools/plot_exclusion_summary.py output/figures/exclusion_planes_summary.png
"""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = sys.argv[1]
tex = [("emu", r"$e\mu$ texture"), ("etau", r"$e\tau$ texture"), ("mutau", r"$\mu\tau$ texture")]
regimes = [("mev", r"$g_D\,|U_{\alpha4}|^2$", r"absorption only ($\epsilon'=0$)", "g_D_u2"),
           ("potential", r"$g_D\,\epsilon'\,|U_{\alpha4}|^2$", r"with the potential ($\epsilon'=0.1$)", "g_D_eps_u2")]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
mesh = None
for r, (reg, ylab, rtitle, yname) in enumerate(regimes):
    for c, (key, ttitle) in enumerate(tex):
        ax = axes[r, c]
        path = f"{REPO}/output/analysis/pion_2050_case_5_{reg}_{key}/data/plane_m_Zd_eV_{yname}.json"
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "not run yet", ha="center", transform=ax.transAxes)
            continue
        s = json.load(open(path))
        x = np.array(s["grids"][0]); y = np.array(s["grids"][1])
        d = np.array([[np.nan if v is None else v for v in row] for row in s["dchi2"]])
        X, Y = np.meshgrid(x, y, indexing="ij")
        mesh = ax.pcolormesh(X, Y, np.clip(d, 0, 8), cmap="Greys", vmin=0, vmax=8, shading="nearest", rasterized=True)
        lim = []
        for col in s["columns"]:
            ivs = col["intervals"].get("95%", [])
            bad = (not ivs) or (col["value"] in s.get("elevated", [])) or ivs[0]["hi_open"]
            lim.append(np.nan if bad else ivs[0]["hi"])
        ax.plot(x, np.array(lim), color="tab:red", lw=2.0, label=r"95% upper limit ($\Delta\chi^2 = 3.84$ per mass)")
        h = np.sqrt(x[1] / x[0])
        for v in s.get("elevated", []):
            k = int(np.argmin(np.abs(x - v)))
            ax.axvspan(x[k] / h, x[k] * h, facecolor="none", edgecolor="tab:blue", hatch="///", lw=0, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$m_{Z_D}$ [eV]"); ax.set_ylabel(ylab)
        ax.set_title(f"{ttitle}, {rtitle}", fontsize=10)
        if r == 0 and c == 0:
            ax.legend(fontsize=7, loc="upper left")
if mesh is not None:
    fig.colorbar(mesh, ax=axes, shrink=0.6, label=r"$\Delta\chi^2$ per mass (clipped at 8)")
fig.suptitle("Projected 2050 flavor measurement (pion sources), whole sky: excluded above the red line (95%, per mass)"
             "\nhatched: the scanned couplings never reach the no-effect limit there", fontsize=11)
fig.subplots_adjust(hspace=0.32, wspace=0.28)
fig.savefig(out, dpi=105, bbox_inches="tight")
print("saved", out)
