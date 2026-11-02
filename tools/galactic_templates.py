"""
galactic_templates.py -- the Galactic templates as the solver uses them (docs/dev/DIRECTIONS_PLAN.md, step 5,
test T6): the CRINGE map on the sky, the inner-Galaxy spectra against IceCube's 2026 fit (arXiv:2607.25966
Fig. 4, src/data/ic26_fig4_digitized.json), and the declination-band profile the per-band output carries.
Light (N_side 16 maps); writes output/sky/galactic_templates.json and output/figures/galactic_templates.png.

  python tools/galactic_templates.py
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_galactic as G  # noqa: E402
import dp_sky  # noqa: E402


def main():
    grid = dp_sky.SkyGrid(16)
    inner = G.inner_galaxy(grid)
    fig4 = json.load(open(os.path.join(REPO, "src", "data", "ic26_fig4_digitized.json")))
    E = np.geomspace(1e3, 1e7, 81)
    models = [("cringe", 0.91, "CRINGE (diffuse + unresolved) x 0.91", "C2", "-"),
              ("cringe_diffuse", 0.91, "CRINGE diffuse only x 0.91", "C2", ":"),
              ("pi0", 4.99, r"Fermi-LAT $\pi^0$ x 4.99", "C0", "-")]
    res = dict(note="T6: per-flavor nu + nubar E^2 Phi (GeV cm^-2 s^-1 sr^-1) averaged over |b| < 15 deg, |l| < 20 deg, "
                    "against arXiv:2607.25966 Fig. 4", ratios={})
    spectra = {}
    for name, norm, *_ in models:
        spectra[name] = norm * G.intensity(name, E, 16)[inner].mean(axis=0) * E ** 2
    for name, norm, curve in (("cringe", 0.91, "cringe"), ("cringe_diffuse", 0.91, "cringe"), ("pi0", 4.99, "pi0")):
        c = np.array(fig4["curves_E_GeV_E2Phi"][curve])
        r = norm * G.intensity(name, c[:, 0], 16)[inner].mean(axis=0) * c[:, 0] ** 2 / c[:, 1]
        res["ratios"][f"{name} vs IceCube {curve}"] = dict(min=float(r.min()), max=float(r.max()),
                                                           E_range_GeV=[float(c[0, 0]), float(c[-1, 0])])
    # all-flavor sky fractions and the band profile at 30 TeV
    prof = {}
    for name, norm, *_ in models:
        I30 = norm * G.intensity(name, np.array([3e4]), 16)[:, 0]
        prof[name] = G.band_means(I30, grid.band, grid.n_bands)
        res[f"{name}_south_fraction_30TeV"] = float(I30[grid.dec < 0].sum() / I30.sum())
        res[f"{name}_unresolved_share_sky_100TeV"] = None
    Eu, Iu, _ = G.load_template("cringe", 16)
    Ed, Id, _ = G.load_template("cringe_diffuse", 16)
    k = int(np.argmin(np.abs(Eu - 1e5)))
    res["cringe_unresolved_share_sky_100TeV"] = float(1 - Id[:, k].sum() / Iu[:, k].sum())
    res = {k: v for k, v in res.items() if v is not None}
    res["commit"] = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    os.makedirs(os.path.join(REPO, "output", "sky"), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, "output", "sky", "galactic_templates.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8.6))
    ax = fig.add_subplot(2, 2, (1, 2), projection="mollweide")
    I30 = 0.91 * G.intensity("cringe", np.array([3e4]), 16)[:, 0] * 9e8
    ra = np.where(grid.ra > np.pi, grid.ra - 2 * np.pi, grid.ra)
    sc = ax.scatter(-ra, grid.dec, c=np.log10(I30), s=9, cmap="magma", vmin=np.log10(I30).max() - 2.5)
    for e in grid.band_edges[1:-1]:
        ax.plot(np.linspace(-np.pi, np.pi, 50), np.full(50, e), color="w", lw=0.3, alpha=0.4)
    lb = np.linspace(-np.pi, np.pi, 200)
    pr, pd = dp_sky.galactic_to_equatorial(lb, np.zeros_like(lb))
    pr = np.where(pr > np.pi, pr - 2 * np.pi, pr)
    o = np.argsort(-pr)
    ax.plot(-pr[o], pd[o], ".", color="c", ms=1.2)
    fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.025, pad=0.02,
                 label=r"log$_{10}$ $E^2\Phi$ per flavor, $\nu+\bar\nu$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title("CRINGE x 0.91 at 30 TeV, equatorial (RA increasing left), N_side 16; white: the 20 declination bands; "
                 "cyan: b = 0", fontsize=9)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(2, 2, 3)
    for (name, norm, lab, col, ls) in models:
        ax.loglog(E, spectra[name], color=col, ls=ls, lw=1.6, label="ours: " + lab)
    for curve, col in (("cringe", "C2"), ("pi0", "C0")):
        c = np.array(fig4["curves_E_GeV_E2Phi"][curve])
        ax.plot(c[::3, 0], c[::3, 1], "o", mfc="none", color=col, ms=5, label=f"IceCube 2026 Fig. 4: {curve}")
    for a in fig4["astrophysical_all_sky_per_flavour"]:
        x = np.sqrt(a["E_lo_GeV"] * a["E_hi_GeV"])
        if a["E2Phi"] > 3e-9:
            ax.errorbar(x, a["E2Phi"], yerr=[[a["E2Phi"] - a["E2Phi_lo"]], [a["E2Phi_hi"] - a["E2Phi"]]],
                        xerr=[[x - a["E_lo_GeV"]], [a["E_hi_GeV"] - x]], color="0.5", lw=0.8)
    ax.plot([], [], color="0.5", label="astrophysical, all-sky average (2026)")
    ax.set_xlim(1e3, 1e7); ax.set_ylim(1e-10, 3e-7)
    ax.set_xlabel("E [GeV]"); ax.set_ylabel(r"$E^2\Phi$ per flavor, $\nu+\bar\nu$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title(r"Inner Galaxy ($|b|<15^\circ$, $|l|<20^\circ$): ours vs IceCube 2026", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")

    ax = fig.add_subplot(2, 2, 4)
    sd = np.sin(grid.band_edges)
    for (name, norm, lab, col, ls) in models:
        ax.stairs(prof[name] * 9e8, sd, color=col, ls=ls, lw=1.5, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"sin(dec)  (South Pole: up-going for dec > 0)")
    ax.set_ylabel(r"band mean $E^2\Phi$ at 30 TeV [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_title("Declination-band profile (what the per-band output carries)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "output", "figures", "galactic_templates.png"), dpi=110)
    print("wrote output/sky/galactic_templates.json, output/figures/galactic_templates.png")


if __name__ == "__main__":
    main()
