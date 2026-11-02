"""
plots.py

Figures of profiles and planes, read from their summary JSON files (run.summarize writes
them). Delta chi^2 is drawn against the parameter, with the thresholds of the configuration
as horizontal lines (profiles) or contour levels (planes). Drawn in Mauricio's style
(figure_style.py and its matplotlibrc; 2026-10-21).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .figure_style import PANEL, tex, use  # noqa: E402

LABELS = {
    "m_Zd_eV": r"$m_{Z_D}$ [eV]", "g_D": r"$g_D$", "epsilon_prime": r"$\epsilon'$",
    "u_e": r"$|U_{e4}|$", "u_mu": r"$|U_{\mu 4}|$", "u_tau": r"$|U_{\tau 4}|$",
    "s12sq": r"$\sin^2\theta_{12}$", "s13sq": r"$\sin^2\theta_{13}$", "s23sq": r"$\sin^2\theta_{23}$",
    "d_cp_deg": r"$\delta_{\rm CP}$ [$^\circ$]", "dm21_sq": r"$\Delta m^2_{21}$ [eV$^2$]",
    "dm31_sq": r"$\Delta m^2_{31}$ [eV$^2$]", "m_lightest_eV": r"$m_{\rm lightest}$ [eV]",
    "f_e_S": r"$f_{e,S}$", "f_mu_S": r"$f_{\mu,S}$",
    "g_D_u2": r"$g_D\,|U_{\alpha 4}|^2$", "g_D_eps_u2": r"$g_D\,\epsilon'\,|U_{\alpha 4}|^2$",
}
STYLES = [("0.25", "--"), ("0.45", "-."), ("0.65", ":")]


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", r"\_"))


def plot_profile(summary: dict, path: str):
    """Delta chi^2 of one profile with its thresholds. summary: a profile summary
    (run.summarize); path: output file (format from its extension)."""
    name = summary["names"][0]
    x = np.asarray(summary["grids"][0])
    d = np.asarray(summary["dchi2"])
    log = summary.get("log", False)
    use()
    fig, ax = plt.subplots(figsize=PANEL)
    ok = np.isfinite(d)
    ax.plot(x[ok], d[ok], "k-", lw=1.4)
    ax.plot(x[ok], d[ok], "k.", ms=3)
    top = max(1.5 * max(summary["thresholds"].values()), 1.0)
    for (lab, t), (c, ls) in zip(sorted(summary["thresholds"].items(), key=lambda kv: kv[1]), STYLES):
        ax.axhline(t, color=c, ls=ls, lw=1.0)
        ax.text(x[0], t, tex(f" {lab}"), va="bottom", ha="left", fontsize=12, color=c)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, top)
    ax.set_xlabel(label(name))
    ax.set_ylabel(r"$\Delta\chi^2$")
    ax.set_title(tex(summary.get("title", "")), fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_plane(summary: dict, path: str):
    """Delta chi^2 over a plane (grey scale) with its contours at the summary's thresholds:
    the 2D ones and the plane's lowest point for a joint region, the 1D ones with
    column-relative Delta chi^2 for a raster plane (elevated columns hatched)."""
    xn, yn = summary["names"]
    x = np.asarray(summary["grids"][0])
    y = np.asarray(summary["grids"][1])
    d = np.asarray(summary["dchi2"], dtype=float)          # [i_x, i_y]
    logx, logy = summary.get("log", [False, False])
    use()
    fig, ax = plt.subplots(figsize=(PANEL[0] + 1.3, PANEL[1]))
    levels = sorted(summary["thresholds"].items(), key=lambda kv: kv[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    dm = np.where(np.isfinite(d), d, 1e9)
    top = 2.0 * levels[-1][1]
    mesh = ax.pcolormesh(X, Y, np.clip(dm, 0.0, top), cmap="Greys", vmin=0.0, vmax=top, shading="nearest",
                         rasterized=True)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(r"$\Delta\chi^2$ (clipped at %.3g)" % top)
    contour_colors = ["tab:blue", "tab:orange", "tab:red"]
    handles = []
    for (lab, t), c in zip(levels, contour_colors):
        ax.contour(X, Y, dm, levels=[t], colors=[c], linewidths=1.2)
        handles.append(matplotlib.lines.Line2D([], [], color=c, lw=1.2,
                                               label=tex(f"{lab} ($\\Delta\\chi^2$ = {t:g})")))
    ax.legend(handles=handles, fontsize=11, loc="lower right", framealpha=0.85)
    raster = summary.get("raster")
    if raster is None:
        i, j = np.unravel_index(np.argmin(dm), dm.shape)
        ax.plot(x[i], y[j], "k*", ms=8)
    else:
        # columns whose own minimum lies far above the best fit: their column-relative map
        # would read as allowed, so they are hatched
        g = x if raster == xn else y
        lg = (logx if raster == xn else logy)
        u = np.log10(g) if lg else g
        edges = np.concatenate([[u[0] - 0.5 * (u[1] - u[0])], 0.5 * (u[1:] + u[:-1]), [u[-1] + 0.5 * (u[-1] - u[-2])]])
        edges = 10.0 ** edges if lg else edges
        for v in summary.get("elevated", []):
            k = int(np.argmin(np.abs(g - v)))
            span = ax.axvspan if raster == xn else ax.axhspan
            span(edges[k], edges[k + 1], facecolor="none", edgecolor="tab:red", hatch="///", lw=0.0, alpha=0.6)
        if summary.get("elevated"):
            handles.append(matplotlib.patches.Patch(facecolor="none", edgecolor="tab:red", hatch="///",
                                                    label="column minimum above the best fit"))
            ax.legend(handles=handles, fontsize=11, loc="lower right", framealpha=0.85)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(label(xn))
    ax.set_ylabel(label(yn))
    title = tex(summary.get("title", ""))
    if raster is not None:
        title += f"\nper {label(raster)}: $\\Delta\\chi^2$ from each column's minimum"
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
