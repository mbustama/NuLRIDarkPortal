"""
plots.py

Figures of profiles and planes, read from their summary JSON files (run.summarize writes
them). Delta chi^2 is drawn against the parameter, with the thresholds of the configuration
as horizontal lines (profiles) or contour levels (planes).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "m_Zd_eV": r"$m_{Z_D}$ [eV]", "g_D": r"$g_D$", "epsilon_prime": r"$\epsilon'$",
    "u_e": r"$|U_{e4}|$", "u_mu": r"$|U_{\mu 4}|$", "u_tau": r"$|U_{\tau 4}|$",
    "s12sq": r"$\sin^2\theta_{12}$", "s13sq": r"$\sin^2\theta_{13}$", "s23sq": r"$\sin^2\theta_{23}$",
    "d_cp_deg": r"$\delta_{\rm CP}$ [$^\circ$]", "dm21_sq": r"$\Delta m^2_{21}$ [eV$^2$]",
    "dm31_sq": r"$\Delta m^2_{31}$ [eV$^2$]", "m_lightest_eV": r"$m_{\rm lightest}$ [eV]",
    "f_e_S": r"$f_{e,S}$", "f_mu_S": r"$f_{\mu,S}$",
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
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ok = np.isfinite(d)
    ax.plot(x[ok], d[ok], "k-", lw=1.4)
    ax.plot(x[ok], d[ok], "k.", ms=3)
    top = max(1.5 * max(summary["thresholds"].values()), 1.0)
    for (lab, t), (c, ls) in zip(sorted(summary["thresholds"].items(), key=lambda kv: kv[1]), STYLES):
        ax.axhline(t, color=c, ls=ls, lw=1.0)
        ax.text(x[0], t, f" {lab}", va="bottom", ha="left", fontsize=8, color=c)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, top)
    ax.set_xlabel(label(name))
    ax.set_ylabel(r"$\Delta\chi^2$")
    ax.set_title(summary.get("title", ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_plane(summary: dict, path: str):
    """Contours of Delta chi^2 over a plane at the configuration's 2D thresholds, and the
    lowest point of the plane."""
    xn, yn = summary["names"]
    x = np.asarray(summary["grids"][0])
    y = np.asarray(summary["grids"][1])
    d = np.asarray(summary["dchi2"], dtype=float)          # [i_x, i_y]
    logx, logy = summary.get("log", [False, False])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    levels = sorted(summary["thresholds"].items(), key=lambda kv: kv[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    dm = np.where(np.isfinite(d), d, 1e9)
    for (lab, t), (c, ls) in zip(levels, STYLES):
        cs = ax.contour(X, Y, dm, levels=[t], colors=[c], linestyles=[ls], linewidths=1.2)
        ax.clabel(cs, fmt={t: lab}, fontsize=7)
    i, j = np.unravel_index(np.argmin(dm), dm.shape)
    ax.plot(x[i], y[j], "k*", ms=8)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(label(xn))
    ax.set_ylabel(label(yn))
    ax.set_title(summary.get("title", ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
