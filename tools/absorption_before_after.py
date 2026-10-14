"""
absorption_before_after.py -- the resonant absorption before and after the physics audit of 2026-10-14
(docs/dev/PHYSICS_AUDIT.md): the flux at the detector over the flux without interactions, versus energy, for the
dark portal's e-mu texture at M = 3 MeV with eps' = 0 (no potential), for
  * the theory notes' form (absorption_model 'theory_notes', used until 2026-10-14),
  * the corrected resonance in the dark-portal solver (absorption_model 'lagrangian', cosmo_transport
    'dark_portal'),
  * every collision with the relics through dp_si (cosmo_transport 'si', the MeV regime's default),
at two couplings. Also the energy-averaged flavor fractions of each.

  python tools/absorption_before_after.py [--out output/figures/absorption_before_after.png]
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dp_core  # noqa: E402
dp_core.HAS_TQDM = False
import dark_portal_solver as dps  # noqa: E402

GRID = {"n_E_bins": 400, "n_z_bins": 800, "n_sun_paths": 2, "n_earth_chords": 4, "N_steps_earth": 10}
CASES = [("Theory notes (until 2026-10-14)", dict(absorption_model="theory_notes"), "#999999", (0, (4, 2))),
         ("Corrected resonance", dict(absorption_model="lagrangian", cosmo_transport="dark_portal"), "#D55E00", "-"),
         ("Every collision (dp_si)", dict(absorption_model="lagrangian", cosmo_transport="si"), "#0072B2", (0, (1, 1.2)))]


def run(dp):
    E, params = dps.build_solver_params({"grid": GRID, "dark_portal": dp,
                                         "execution": {"n_cores": 1, "verbose": 0}})
    params["show_progress"] = False
    return E, dps.propagate_weighted(E, params)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "output", "figures", "absorption_before_after.png"))
    a = ap.parse_args(argv)
    E, F0 = run({"u_e": 0.0, "u_mu": 0.0, "epsilon_prime": 0.0})
    dE = dp_core.compute_dE_geometric(E)
    frac = lambda F: ((F * dE[:, None]).sum(0) / (F * dE[:, None]).sum()).round(4).tolist()
    m_nu = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    M = 3e6
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    summary = {"none": frac(F0)}
    for ax, gu2 in zip(axes, (2.5e-4, 2.5e-3)):
        for name, opts, col, ls in CASES:
            _, F = run({"m_Zd_eV": M, "g_D": gu2 / 0.25, "epsilon_prime": 0.0, **opts})
            ax.semilogx(E, F.sum(1) / F0.sum(1), color=col, ls=ls, lw=1.3, label=name)
            summary[f"{name} | g|U|^2={gu2:g}"] = frac(F)
        for r in range(3):
            ax.axvline(M * M / (2 * m_nu[r]), color="#bbbbbb", lw=0.5, ls=":")
            ax.text(M * M / (2 * m_nu[r]), 0.02, f" $m_{r + 1}$", fontsize=7, color="#777777")
        ax.axhline(1.0, color="#555555", lw=0.5)
        ax.set_title(f"$e\\mu$ texture, $M = 3$ MeV, $g_D|U|^2 = {gu2:g}$", fontsize=8.5, loc="left")
        ax.set_xlabel("Neutrino energy [eV]")
        ax.set_ylim(0, 1.6)
    axes[0].set_ylabel("Flux at the detector / without interactions")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("Resonant absorption on the relic neutrinos, before and after the audit (all flavors, $\\nu+\\bar\\nu$)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
