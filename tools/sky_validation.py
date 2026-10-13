"""
sky_validation.py -- the per-direction output with the Galactic component, put together (docs/dev/DIRECTIONS_PLAN.md,
step 7). The extragalactic flux is normalized to IceCube's HESE 7.5-yr fit (sky.extragalactic_norm = 2.12e-18
per flavor at 100 TeV, p_index = 2.87; PRD 104, 022002) and the Galactic one is CRINGE x 0.91 (IceCube 2026),
emission points in 3-D. Light (one process, N_side 16). Writes output/sky/sky_validation.json and
output/figures/sky_validation.png:

  (a) all-sky per-flavor spectra: extragalactic, Galactic (CRINGE, pi0) and IceCube's 2026 astrophysical flux;
      the Galactic share against energy, all-sky and in the inner Galaxy (check: IceCube 2026 states the
      Galactic component is ~10% of the astrophysical flux at ~20 TeV);
  (b) the nu_e fraction of the total flux per pixel at 100 TeV for a dark-portal point (m_Zd = 1e-24 eV,
      kappa = 300, e-mu texture): the Earth's bands and the Galactic plane;
  (c) per declination band, the nu_e fraction of the total and the Galactic share, with and without the
      dark portal;
  (d) the anisotropy each stage adds: the spread of the nu_e fraction over pixels from the Earth (bands), the
      Galactic component (emission points in 3-D) and the Sun (decision D4's diagnostic).

  python tools/sky_validation.py
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dark_portal_solver as dps  # noqa: E402
import dp_galactic as G  # noqa: E402
import dp_sky  # noqa: E402

GU2_PER_KAPPA = 1.66e-54
HESE = dict(norm=2.12e-18, p=2.87)


def run(m_zd=1e-24, kappa=300.0, template="cringe", nside=16, n_E=40, coupling=True):
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": m_zd,
          "g_D": (kappa * GU2_PER_KAPPA / (0.1 * 0.5)) if coupling else 1e-60, "epsilon_prime": 0.1 if coupling else 0.0}
    cfg = {"dark_portal": dp, "source": {"p_index": HESE["p"]},
           "grid": {"n_E_bins": n_E, "E_min_eV": 1e13, "E_max_eV": 1e16, "n_z_bins": 200},
           "sky": {"output": "healpix", "nside": nside, "extragalactic_norm": HESE["norm"]},
           "galactic": {"enabled": True, "template": template, "emission": "3d"},
           "execution": {"n_cores": 1, "verbose": 0}}
    E, params = dps.build_solver_params(cfg)
    params["show_progress"] = False
    out = dps.propagate_sky(E, params)
    eg = (out["bands_nu"] + out["bands_nubar"])[out["pixel_band"]]                  # (N_pix, N_E, 3)
    gal = G.pixel_flux(out, "nu") + G.pixel_flux(out, "nubar")
    return E, out, eg, gal


def main():
    t0 = time.time()
    grid = dp_sky.SkyGrid(16)
    inner = G.inner_galaxy(grid)
    fig4 = json.load(open(os.path.join(REPO, "src", "data", "ic26_fig4_digitized.json")))
    E, out, eg, gal = run()
    _, out0, eg0, gal0 = run(coupling=False)
    _, outp, egp, galp = run(coupling=False, template="pi0")
    Eg = E / 1e9
    res = dict(note="extragalactic: HESE 7.5-yr normalization (2.12e-18 per flavor at 100 TeV, p = 2.87); Galactic: "
                    "CRINGE x 0.91 (pi0 x 4.99 for comparison), emission points in 3-D; dark portal point m_Zd = 1e-24 eV, "
                    "kappa = 300, e-mu texture u = 0.5, eps' = 0.1")
    # (a) spectra and shares
    eg_sky = eg0.mean(0).sum(-1) / 3.0
    g_sky, gp_sky = gal0.mean(0).sum(-1) / 3.0, galp.mean(0).sum(-1) / 3.0
    g_in, eg_in = gal0[inner].mean(0).sum(-1) / 3.0, eg0[inner].mean(0).sum(-1) / 3.0
    astro = fig4["astrophysical_all_sky_per_flavour"]
    a20 = [a for a in astro if a["E_lo_GeV"] <= 2e4 < a["E_hi_GeV"]][0]
    k20 = int(np.argmin(np.abs(Eg - 2e4)))
    res["galactic_over_icecube_astro_20TeV"] = dict(
        cringe=float(g_sky[k20] * Eg[k20] ** 2 / a20["E2Phi"]), pi0=float(gp_sky[k20] * Eg[k20] ** 2 / a20["E2Phi"]),
        icecube_bin_GeV=[a20["E_lo_GeV"], a20["E_hi_GeV"]], icecube_statement="~10% at ~20 TeV (arXiv:2607.25966)")
    res["galactic_share_allsky"] = {f"{e:.0f}TeV": float(g_sky[k] / (g_sky[k] + eg_sky[k]))
                                    for e, k in ((Eg[i] / 1e3, i) for i in (0, 13, 26, 39))}
    res["galactic_share_inner"] = {f"{e:.0f}TeV": float(g_in[k] / (g_in[k] + eg_in[k]))
                                   for e, k in ((Eg[i] / 1e3, i) for i in (0, 13, 26, 39))}
    # (b)-(d) the dark-portal point
    k100 = int(np.argmin(np.abs(E - 1e14)))
    tot = eg + gal
    fe = tot[:, k100, 0] / tot[:, k100].sum(-1)
    fe_eg = eg[:, k100, 0] / eg[:, k100].sum(-1)
    fe_gal = gal[:, k100, 0] / gal[:, k100].sum(-1)
    sun = json.load(open(os.path.join(REPO, "output", "sky", "sun_anisotropy.json")))
    res["anisotropy_fe_100TeV"] = dict(
        earth_bands_extragalactic=float(np.ptp(fe_eg)), galactic_component=float(np.ptp(fe_gal)),
        total=float(np.ptp(fe)),
        sun_yearly_max_over_points=float(max(r["max_abs_dfe"] for r in sun["rows"])))
    res["commit"] = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.join(REPO, "output", "sky"), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, "output", "sky", "sky_validation.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 9.5))
    ax = fig.add_subplot(2, 2, 1)
    ax.loglog(Eg, eg_sky * Eg ** 2, "k-", label="extragalactic (HESE 7.5-yr fit, no new physics)")
    ax.loglog(Eg, g_sky * Eg ** 2, "C2-", label="Galactic CRINGE x 0.91, all-sky mean")
    ax.loglog(Eg, gp_sky * Eg ** 2, "C0-", label=r"Galactic $\pi^0$ x 4.99, all-sky mean")
    ax.loglog(Eg, g_in * Eg ** 2, "C2--", label="Galactic CRINGE x 0.91, inner Galaxy")
    for a in astro:
        x = np.sqrt(a["E_lo_GeV"] * a["E_hi_GeV"])
        if a["E2Phi"] > 3e-9:
            ax.errorbar(x, a["E2Phi"], yerr=[[a["E2Phi"] - a["E2Phi_lo"]], [a["E2Phi_hi"] - a["E2Phi"]]],
                        xerr=[[x - a["E_lo_GeV"]], [a["E_hi_GeV"] - x]], color="0.5", lw=0.8)
    ax.plot([], [], color="0.5", label="IceCube 2026 astrophysical, all-sky")
    ax2 = ax.twinx()
    ax2.semilogx(Eg, g_sky / (g_sky + eg_sky), "C3:", label="Galactic share, all sky")
    ax2.semilogx(Eg, g_in / (g_in + eg_in), "C3-.", label="Galactic share, inner Galaxy")
    ax2.set_ylabel("Galactic share of the flux", color="C3"); ax2.set_ylim(0, 1)
    ax.set_xlabel("E [GeV]"); ax.set_ylabel(r"$E^2\Phi$ per flavor, $\nu+\bar\nu$ [GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    ax.set_ylim(1e-10, 3e-7); ax.set_xlim(3e3, 1.2e7)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower left")
    ax.set_title(f"Galactic vs extragalactic; at 20 TeV CRINGE is {100 * res['galactic_over_icecube_astro_20TeV']['cringe']:.0f}% "
                 f"of IceCube's astrophysical flux (they state ~10%)", fontsize=8.5)
    ax = fig.add_subplot(2, 2, 2, projection="mollweide")
    ra = np.where(grid.ra > np.pi, grid.ra - 2 * np.pi, grid.ra)
    sc = ax.scatter(-ra, grid.dec, c=fe, s=8, cmap="viridis")
    fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05,
                 label=r"$f_e$ of the total flux at 100 TeV")
    ax.set_title(r"Dark portal, $m_{Z_D}$ = 1e-24 eV, $\kappa$ = 300: total flux, equatorial (RA increasing left)", fontsize=8.5)
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(2, 2, 3)
    sd = np.sin(grid.band_edges)
    lo_hi = []
    for name, o, gl, col in (("dark portal", out, gal, "C3"), ("no new physics", out0, gal0, "k")):
        egb = (o["bands_nu"] + o["bands_nubar"])[:, k100]
        gb = (o["galactic_bands_nu"] + o["galactic_bands_nubar"])[:, k100]
        tb = egb + gb
        for f, ls, lab in ((tb, "-", "total"), (egb, ":", "extragalactic"), (gb, "--", "Galactic")):
            v = f[:, 0] / f.sum(-1)
            lo_hi += [v.min(), v.max()]
            ax.stairs(v, sd, color=col, ls=ls, lw=1.4, label=f"{name}: $f_e$ {lab}")
    ax.set_ylim(min(lo_hi) - 0.004, max(lo_hi) + 0.012)
    ax3 = ax.twinx()
    gb0 = (out0["galactic_bands_nu"] + out0["galactic_bands_nubar"])[:, k100].sum(-1)
    eb0 = (out0["bands_nu"] + out0["bands_nubar"])[:, k100].sum(-1)
    ax3.stairs(gb0 / (gb0 + eb0), sd, color="C2", lw=1.0, label="Galactic share")
    ax3.set_ylabel("Galactic share at 100 TeV", color="C2"); ax3.set_ylim(0, 0.5)
    ax.set_xlabel("sin(dec)  (South Pole: up-going for dec > 0)"); ax.set_ylabel(r"$f_e$ at 100 TeV")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Per declination band", fontsize=9)
    ax = fig.add_subplot(2, 2, 4)
    an = res["anisotropy_fe_100TeV"]
    names = ["Earth (bands),\nextragalactic", "Galactic component\n(3-D emission)", "total flux",
             "Sun, yearly\n(D4 diagnostic, max)"]
    vals = [an["earth_bands_extragalactic"], an["galactic_component"], an["total"], an["sun_yearly_max_over_points"]]
    ax.bar(range(4), vals, color=["C0", "C2", "C3", "C1"])
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8); ax.set_yscale("log")
    ax.set_ylabel(r"spread of $f_e$ over pixels")
    ax.set_title("The anisotropy each stage adds (100 TeV; the Sun: its own diagnostic points)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "output", "figures", "sky_validation.png"), dpi=110)
    print(f"wrote output/sky/sky_validation.json, output/figures/sky_validation.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
