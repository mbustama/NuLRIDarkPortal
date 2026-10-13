"""
plot_si_summary.py -- figures of the secret-interaction results of the flavor-triangle analysis:

  python tools/plot_si_summary.py planes output/figures/si_planes_summary.png
      the four exclusion planes (nu_e, nu_mu, nu_tau, flavor-universal couplings): per-mass
      Delta chi^2 and the 95% upper limit (the upper end of each mass column's lowest allowed
      interval at Delta chi^2 = 3.84), from output/analysis/pion_2050_case_5_si_*/data/;
  python tools/plot_si_summary.py planes2 output/figures/si_planes_two_flavor.png
      the same for couplings to two flavors (e+mu, e+tau, mu+tau) next to the universal one;
  python tools/plot_si_summary.py limits output/figures/si_limits_models.png
      the 95% upper limits of the four models (scalar or vector mediator, Majorana or Dirac
      neutrinos), one panel per texture;
  python tools/plot_si_summary.py paths output/figures/si_flavor_paths.png
      the energy-averaged flavor composition along a coupling ladder at four mediator masses,
      against the projected 2050 measurement, and the all-flavor spectra at g = 0.1, from
      output/si_planes/flavor_paths.json (tools/si_flavor_paths.py).
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
TEX = [("e", r"$\nu_e$ couplings, $g\,\delta_{\alpha e}\delta_{\beta e}$"),
       ("mu", r"$\nu_\mu$ couplings, $g\,\delta_{\alpha\mu}\delta_{\beta\mu}$"),
       ("tau", r"$\nu_\tau$ couplings, $g\,\delta_{\alpha\tau}\delta_{\beta\tau}$"),
       ("universal", r"flavor-universal, $g\,\delta_{\alpha\beta}$")]
TEX2 = [("e_mu", r"$\nu_e$ and $\nu_\mu$, $g\,{\rm diag}(1, 1, 0)$"),
        ("e_tau", r"$\nu_e$ and $\nu_\tau$, $g\,{\rm diag}(1, 0, 1)$"),
        ("mu_tau", r"$\nu_\mu$ and $\nu_\tau$, $g\,{\rm diag}(0, 1, 1)$"),
        ("universal", r"flavor-universal, $g\,\delta_{\alpha\beta}$")]
# resonance at E = M^2 / (2 m_j) inside 10 TeV - 30 PeV for some z <= 4, m_j = 0.01 .. 0.051 eV
RES_BAND = (np.sqrt(2 * 0.01 * 1e13 / 5.0), np.sqrt(2 * 0.051 * 3e16))


def planes(out, textures=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    mesh = None
    for ax, (key, title) in zip(axes.ravel(), textures or TEX):
        path = f"{REPO}/output/analysis/pion_2050_case_5_si_{key}/data/plane_m_phi_eV_g_phi.json"
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "not run yet", ha="center", transform=ax.transAxes)
            continue
        s = json.load(open(path))
        x = np.array(s["grids"][0])
        y = np.array(s["grids"][1])
        d = np.array([[np.nan if v is None else v for v in row] for row in s["dchi2"]])
        X, Y = np.meshgrid(x, y, indexing="ij")
        mesh = ax.pcolormesh(X, Y, np.clip(d, 0, 8), cmap="Greys", vmin=0, vmax=8, shading="nearest", rasterized=True)
        lim = []
        for col in s["columns"]:
            ivs = col["intervals"].get("95%", [])
            bad = (not ivs) or (col["value"] in s.get("elevated", [])) or ivs[0]["hi_open"]
            lim.append(np.nan if bad else ivs[0]["hi"])
        ax.plot(x, np.array(lim), color="tab:red", lw=2.0, label=r"95% upper limit ($\Delta\chi^2 = 3.84$ per mass)")
        ax.axvspan(*RES_BAND, color="tab:orange", alpha=0.08, lw=0, label="resonance within 10 TeV - 30 PeV")
        h = np.sqrt(x[1] / x[0])
        for v in s.get("elevated", []):
            k = int(np.argmin(np.abs(x - v)))
            ax.axvspan(x[k] / h, x[k] * h, facecolor="none", edgecolor="tab:blue", hatch="///", lw=0, alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$M_\phi$ [eV]")
        ax.set_ylabel(r"$g$")
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.subplots_adjust(hspace=0.3, wspace=0.25, right=0.88)
    if mesh is not None:
        cax = fig.add_axes([0.905, 0.25, 0.015, 0.5])
        fig.colorbar(mesh, cax=cax, label=r"$\Delta\chi^2$ per mass (clipped at 8)")
    fig.suptitle("Secret neutrino interactions (scalar, Majorana), projected 2050 flavor measurement (pion sources):\n"
                 "excluded above the red line (95%, per mass); hatched: the scanned couplings never reach the "
                 "no-effect limit there", fontsize=10)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


MODELS = [("", "scalar, Majorana", "C1", "-"), ("dirac_", "scalar, Dirac", "C3", "--"),
          ("vector_", "vector, Majorana", "C0", "-"), ("vector_dirac_", "vector, Dirac", "C9", "--")]


def _limit_line(path):
    s = json.load(open(path))
    x = np.array(s["grids"][0])
    lim = []
    for col in s["columns"]:
        ivs = col["intervals"].get("95%", [])
        bad = (not ivs) or (col["value"] in s.get("elevated", [])) or ivs[0]["hi_open"]
        lim.append(np.nan if bad else ivs[0]["hi"])
    return x, np.array(lim)


def limits(out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes.ravel(), TEX):
        for tag, lab, col, ls in MODELS:
            path = f"{REPO}/output/analysis/pion_2050_case_5_si_{tag}{key}/data/plane_m_phi_eV_g_phi.json"
            if not os.path.exists(path):
                continue
            x, lim = _limit_line(path)
            ax.plot(x, lim, ls, color=col, lw=1.8, label=lab)
        ax.axvspan(*RES_BAND, color="tab:orange", alpha=0.08, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$M$ [eV]")
        ax.set_ylabel(r"95% upper limit on $g$")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Secret neutrino interactions, projected 2050 flavor measurement (pion sources): 95% upper limits per "
                 "mediator mass\nshaded: resonance within 10 TeV - 30 PeV", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


def paths(out):
    from analysis import surfaces
    d = json.load(open(f"{REPO}/output/si_planes/flavor_paths.json"))
    surf = surfaces.surface("pion_2050_case_5")
    fe = np.linspace(0.15, 0.55, 401)
    fm = np.linspace(0.2, 0.55, 401)
    FE, FM = np.meshgrid(fe, fm, indexing="ij")
    ll = surfaces.log_likelihood(FE.ravel(), FM.ravel(), surf).reshape(FE.shape)
    chi2 = -2.0 * (ll - np.nanmax(ll[np.isfinite(ll)]))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(d["masses_eV"])))
    gl = np.log10(d["couplings"])
    for c, (key, title) in enumerate(TEX):
        ax = axes[0, c]
        ax.contour(FE, FM, chi2, levels=[2.30, 5.99], colors=["k", "k"], linestyles=["-", "--"], linewidths=1)
        rec = d["textures"][key]
        for M, col in zip(d["masses_eV"], cols):
            p = np.array(rec["paths"][str(M)])
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1)
            sc = ax.scatter(p[1:, 0], p[1:, 1], c=gl, cmap="plasma", s=14, zorder=3, vmin=gl[0], vmax=gl[-1])
            ax.annotate(f"{M:.0e} eV".replace("e+0", "e"), p[-1, :2], fontsize=7, color=col,
                        xytext=(3, 3), textcoords="offset points")
        p0 = np.array(rec["paths"][str(d["masses_eV"][0])])[0]
        ax.plot(p0[0], p0[1], "k*", ms=10, zorder=4, label="no interaction")
        ax.set_xlim(0.15, 0.55)
        ax.set_ylim(0.2, 0.55)
        ax.set_xlabel(r"$\langle f_e \rangle$")
        if c == 0:
            ax.set_ylabel(r"$\langle f_\mu \rangle$")
        ax.set_title(title, fontsize=9)
        if c == 0:
            ax.plot([], [], "k-", lw=1, label="2050 projection, 68%")
            ax.plot([], [], "k--", lw=1, label="95%")
            ax.legend(fontsize=7, loc="upper right")
        ax2 = axes[1, c]
        E = np.array(rec["E_eV"]) / 1e9
        for M, col in zip(d["masses_eV"], cols):
            sp = rec["spectra"][str(M)]
            ax2.plot(E, np.array(sp["si"]) / np.array(sp["free"]), color=col, label=f"$M_\\phi$ = {M:.0e} eV")
        ax2.axhline(1, color="gray", lw=0.5)
        ax2.set_xscale("log")
        ax2.set_ylim(0, 2.2)
        ax2.set_xlabel(r"$E_\nu$ [GeV]")
        if c == 0:
            ax2.set_ylabel(f"all-flavor flux / no interaction ($g$ = {d['spectrum_g']})")
            ax2.legend(fontsize=7)
    cb = fig.colorbar(sc, ax=axes[0, :], fraction=0.015, pad=0.01)
    cb.set_label(r"$\log_{10} g$")
    fig.suptitle("How secret interactions move the energy-averaged flavor composition (top; one path per mediator mass, "
                 r"$g = 10^{-3}$ to 1) and the spectrum (bottom); pion sources, $E^{-2.5}$, $z \leq 4$", fontsize=10)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    if sys.argv[1] == "planes2":
        planes(sys.argv[2], TEX2)       # the two-flavor textures and the universal one
    else:
        {"planes": planes, "paths": paths, "limits": limits}[sys.argv[1]](sys.argv[2])
