"""
galactic_emission.py -- the Galactic emission points of the dark portal (docs/dev/DIRECTIONS_PLAN.md, step 6):
where along each line of sight the Galactic neutrinos are born, the potential there, and what it does to their
flavors at the detector. Light (one process): writes output/sky/galactic_emission.json and
output/figures/galactic_emission.png.

  (a) emission-distance distributions toward the Galactic Centre and the anticentre, diffuse (gas x cosmic
      rays) and unresolved sources (Lorimer x distance), with the morphology checks against CRINGE's maps;
  (b) the Yukawa-smoothed density n_eff along the Centre's line of sight for several mediator masses;
  (c) the Galactic nu_e fraction at 1 PeV per pixel for emission points in 3-D (m_Zd = 1e-24 eV, kappa = 300:
      the Sun's potential at the Earth is 300 Delta m^2_31 / 2E at 100 TeV, the local Galactic one ~0.1 of it,
      so the emission potentials straddle the vacuum term across the Galaxy), and
  (d) against energy for the Centre's and the anticentre's pixels and for emission at the local potential.

  python tools/galactic_emission.py
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
from dp_core import cm_inv3_to_eV3  # noqa: E402

GU2_PER_KAPPA = 1.66e-54


def pixel(nside, l_deg, b_deg):
    ra, dec = dp_sky.galactic_to_equatorial(np.radians(l_deg), np.radians(b_deg))
    return int(dp_sky.ang2pix_ring(nside, 0.5 * np.pi - dec, ra))


def run(emission, m_zd=1e-24, kappa=1.0, nside=16, n_E=40):
    cfg = {"dark_portal": {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": m_zd,
                           "g_D": kappa * GU2_PER_KAPPA / (0.1 * 0.5), "epsilon_prime": 0.1},
           "grid": {"n_E_bins": n_E, "E_min_eV": 1e13, "E_max_eV": 1e16, "n_z_bins": 100},
           "sky": {"output": "healpix", "nside": nside},
           "galactic": {"enabled": True, "emission": emission},
           "execution": {"n_cores": 1, "verbose": 0}}
    E, params = dps.build_solver_params(cfg)
    params["show_progress"] = False
    out = dps.propagate_sky(E, params)
    F = G.pixel_flux(out, "nu") + G.pixel_flux(out, "nubar")
    return E, params, out, F / F.sum(-1, keepdims=True)


def main():
    t0 = time.time()
    nside = 16
    grid = dp_sky.SkyGrid(nside)
    gc, ac = pixel(nside, 0.0, 0.0), pixel(nside, 180.0, 0.0)
    s, ds, R, z = G.emission_geometry(nside)
    q_gas, q_src = G.emission_weights(nside)
    gas, src = G.emission_line_integrals(nside)
    I = G.intensity("cringe_diffuse", np.array([1e5]), nside)[:, 0]
    U = G.intensity("cringe", np.array([1e5]), nside)[:, 0] - I
    ok = U > 0
    res = dict(morphology=dict(gas_vs_cringe_diffuse=float(np.corrcoef(np.log(gas), np.log(I))[0, 1]),
                               sources_vs_cringe_unresolved=float(np.corrcoef(np.log(src[ok]), np.log(U[ok]))[0, 1])),
               mean_emission_distance_kpc={name: dict(gas=float((q_gas[p] * s).sum()), sources=float((q_src[p] * s).sum()))
                                           for name, p in (("centre", gc), ("anticentre", ac))})
    kappa = 300.0
    E3, params, out3, f3 = run("3d", kappa=kappa)
    El, _, outl, fl = run("local", kappa=kappa)
    k1 = int(np.argmin(np.abs(E3 - 1e15)))
    res["point"] = (f"m_Zd = 1e-24 eV, kappa = {kappa:g} (the Sun's potential at the Earth in units of Delta m^2_31 / 2E "
                    f"at 100 TeV; the local Galactic one is ~3e-4 of it at this 6-pc range), e-mu texture u = 0.5, "
                    f"eps' = 0.1; nu + nubar")
    res["f_e_1PeV"] = dict(centre_3d=float(f3[gc, k1, 0]), anticentre_3d=float(f3[ac, k1, 0]),
                           centre_local=float(fl[gc, k1, 0]), sky_range_3d=[float(f3[:, k1, 0].min()), float(f3[:, k1, 0].max())])
    res["commit"] = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(REPO, "output", "sky", "galactic_emission.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 2, 1)
    for name, p, col in (("Galactic Centre", gc, "C3"), ("anticentre", ac, "C0")):
        ax.semilogx(s, q_gas[p] / np.log(s[1] / s[0]), color=col, lw=1.6, label=f"{name}: gas x cosmic rays")
        ax.semilogx(s, q_src[p] / np.log(s[1] / s[0]), color=col, ls="--", lw=1.6, label=f"{name}: unresolved sources")
    ax.set_xlabel("distance from the Sun s [kpc]"); ax.set_ylabel("emission per ln s (normalized)")
    ax.set_xlim(0.05, 30)
    ax.set_title(f"Where the Galactic neutrinos are born (morphology vs CRINGE: gas {res['morphology']['gas_vs_cringe_diffuse']:.3f}, "
                 f"sources {res['morphology']['sources_vs_cringe_unresolved']:.3f})", fontsize=9)
    ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 2, 2)
    n0 = params["n_n0_eV3"]
    for m, col in ((1e-31, "C4"), (1e-28, "C0"), (1e-26, "C2"), (1e-20, "C3"), (1e-24, "C1")):
        ax.loglog(s, G.effective_density(R[gc], z[gc], m, n0, 1.0 / params["H0_eV"]) / cm_inv3_to_eV3, color=col,
                  ls="--" if m == 1e-24 else "-", label=f"$m_{{eff}}$ = {m:.0e} eV (range {1.97e-7 / m / 3.086e19:.2g} kpc)")
    ax.set_xlabel("distance from the Sun toward the Galactic Centre [kpc]")
    ax.set_ylabel(r"$n_{eff}$ [cm$^{-3}$]"); ax.set_xlim(0.05, 30)
    ax.set_title("The potential at the emission point (Yukawa-smoothed neutron density)", fontsize=9)
    ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 2, 3, projection="mollweide")
    lw = np.where(grid.l > np.pi, grid.l - 2 * np.pi, grid.l)
    sc = ax.scatter(-lw, grid.b, c=f3[:, k1, 0], s=8, cmap="viridis")
    fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05, label=r"Galactic $f_e$ at 1 PeV")
    ax.set_title(rf"Emission in 3-D, $m_{{Z_D}}$ = 1e-24 eV, $\kappa$ = {kappa:g} (Galactic coordinates, $l$ increasing left)", fontsize=9)
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(2, 2, 4)
    for name, f, p, col, ls in (("centre, 3-D", f3, gc, "C3", "-"), ("anticentre, 3-D", f3, ac, "C0", "-"),
                                ("centre, local emission", fl, gc, "C3", ":"), ("anticentre, local emission", fl, ac, "C0", ":")):
        ax.semilogx(E3 / 1e12, f[p, :, 0], color=col, ls=ls, label=name)
    ax.set_xlabel("E [TeV]"); ax.set_ylabel(r"Galactic $f_e$ at the detector")
    ax.set_title("Flavor at the detector: emission points in 3-D vs the local potential", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "output", "figures", "galactic_emission.png"), dpi=110)
    print(f"wrote output/sky/galactic_emission.json, output/figures/galactic_emission.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
