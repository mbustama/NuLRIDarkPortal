"""
sun_anisotropy.py -- the anisotropy the Sun would add to the per-direction flux (decision D4 of
docs/dev/DIRECTIONS_PLAN.md: measure it before replacing the isotropic Sun average).

Per direction the Sun is seen at x(t) = cos(psi(t)) over the year (dp_sky.cos_angle_to_sun); the
flux is sum_p a_p F_p over the transit's Gauss-Legendre paths, with a_p the year-average of the
Lagrange basis on the path cosines (dp_sky.sun_path_weights). Today's isotropic average is
sum_p w_p F_p. For dark-portal points across the Sun's window (m_Zd = 1e-24 to 1e-18 eV, the Sun's
potential at the Earth kappa = 0.3, 1, 3 times Delta m^2_31 / (2 x 100 TeV), the e-mu texture), this
compares, pixel by pixel (N_side 16), the energy-averaged flavor fractions of the two, and the
single paths' all-sky fractions. Light (one process); writes output/sky/sun_anisotropy.json and
output/figures/sun_anisotropy.png.

  python tools/sun_anisotropy.py
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dark_portal_solver as dps  # noqa: E402
import dp_sky  # noqa: E402
import dp_transit  # noqa: E402
from flux_postprocessing import compute_energy_averaged_ratios  # noqa: E402

GU2_PER_KAPPA = 1.66e-54          # g_D eps |u|^2 giving the Sun's potential Delta m^2_31 / 2(100 TeV) at 1 AU


def config(m_zd, kappa):
    gu2 = kappa * GU2_PER_KAPPA
    return {"dark_portal": {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": m_zd,
                            "g_D": gu2 / (0.1 * 0.5), "epsilon_prime": 0.1},
            "grid": {"n_E_bins": 60, "E_min_eV": 1e13, "E_max_eV": 3e16, "n_z_bins": 200, "n_sun_paths": 8,
                     "N_steps_earth": 25},
            "sky": {"output": "healpix", "nside": 16},
            "execution": {"n_cores": 1, "verbose": 0}}


def fractions(E, F):
    return np.array([compute_energy_averaged_ratios(E, np.ascontiguousarray(f)) for f in F])


def run_point(m_zd, kappa):
    E, params = dps.build_solver_params(config(m_zd, kappa))
    params["show_progress"] = False
    grid = dp_sky.SkyGrid(16)
    x, w = dp_transit.sun_paths(params["n_sun_paths"])
    A = dp_sky.sun_path_weights(grid.elon, grid.elat, x, n_times=360)            # (P, K)
    paths = []
    for xp in x:
        params["sun_path_cos"] = float(xp)
        o = dps.propagate_sky(E, params)
        paths.append(o["bands_nu"] + o["bands_nubar"])                            # (N_b, N_E, 3)
    paths = np.array(paths)                                                       # (K, N_b, N_E, 3)
    iso_bands = np.tensordot(w, paths, axes=(0, 0))                               # (N_b, N_E, 3)
    yearly = np.einsum("pk,kpef->pef", A, paths[:, grid.band])                   # (P, N_E, 3)
    f_iso = fractions(E, iso_bands)[grid.band]                                    # (P, 3)
    f_year = fractions(E, yearly)
    d = f_year - f_iso
    allsky_paths = fractions(E, np.tensordot(np.full(len(grid.band_omega), 1.0 / len(grid.band_omega)),
                                             paths, axes=(0, 1)))                 # (K, 3), equal-area bands
    return dict(m_Zd_eV=m_zd, kappa=kappa, max_abs_dfe=float(np.max(np.abs(d[:, 0]))),
                max_abs_dfmu=float(np.max(np.abs(d[:, 1]))), rms_dfe=float(np.sqrt(np.mean(d[:, 0] ** 2))),
                sky_mean_df=np.mean(d, axis=0).tolist(),
                paths_fraction_spread=float(np.max(np.ptp(allsky_paths, axis=0))),
                f_allsky_iso=fractions(E, np.tensordot(grid.band_omega / (4 * np.pi), iso_bands,
                                                       axes=(0, 0))[None])[0].tolist()), d, grid


def main():
    t0 = time.time()
    rows, worst = [], None
    for m in (1e-24, 1e-22, 1e-20, 1e-18):
        for k in (0.3, 1.0, 3.0):
            r, d, grid = run_point(m, k)
            rows.append(r)
            if worst is None or r["max_abs_dfe"] > worst[0]["max_abs_dfe"]:
                worst = (r, d, grid)
            print(f"m_Zd {m:.0e} kappa {k:3.1f}: max |df_e| {r['max_abs_dfe']:.2e}, max |df_mu| {r['max_abs_dfmu']:.2e}, "
                  f"rms {r['rms_dfe']:.1e}, paths spread {r['paths_fraction_spread']:.1e}", flush=True)
    out = dict(note="per-pixel energy-averaged flavor fractions: yearly per-direction Sun weighting minus the "
                    "isotropic Sun average (DIRECTIONS_PLAN decision D4); e-mu texture, u = 0.5, eps' = 0.1",
               rows=rows, commit=os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
               host=os.uname().nodename, seconds=time.time() - t0)
    os.makedirs(os.path.join(REPO, "output", "sky"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "output", "sky", "sun_anisotropy.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, d, grid = worst
    fig = plt.figure(figsize=(8, 4.4))
    ax = fig.add_subplot(111, projection="mollweide")
    ra = np.where(grid.ra > np.pi, grid.ra - 2 * np.pi, grid.ra)
    sc = ax.scatter(-ra, grid.dec, c=d[:, 0], s=6, cmap="RdBu_r",
                    vmin=-np.max(np.abs(d[:, 0])), vmax=np.max(np.abs(d[:, 0])))
    fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05,
                 label=r"$\Delta f_e$ (yearly Sun minus isotropic Sun)")
    ax.set_title(f"Sun anisotropy, $m_{{Z_D}}$ = {r['m_Zd_eV']:.0e} eV, kappa = {r['kappa']:g} (equatorial; RA increasing left)",
                 fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(REPO, "output", "figures", "sun_anisotropy.png"), dpi=110, bbox_inches="tight")
    print("wrote output/sky/sun_anisotropy.json, output/figures/sun_anisotropy.png", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
