"""
trajectory_figure.py -- the paper's figure of the six species along the trip (2026-11-07, Mauricio's request: the flux
of each flavor of neutrinos and antineutrinos vs. position, from the sources to the detector, for every family of
models; columns lined up with Fig. 3's panels, the Universe, the Milky Way, near the Sun, inside the Earth).
compute() runs src/dp_trajectory.py's stages for the benchmarks (sources in one redshift shell, one observed energy)
and caches them in output/analysis/trajectory_figure.npz; figure() draws from the cache. Called from
notebooks/paper_figures.ipynb (Sec. 8), in the repository's style (analysis.figure_style.use() first).

Benchmarks, one row each, ordered by where along the trip they act:
  si     -- secret interactions, a Majorana scalar on nu_tau, 10 MeV, g = 1e-3: resonant absorption in the Universe;
  dp     -- the dark portal mixed with nu_tau (u_tau = 1/sqrt 2, as the campaign), 1e-2 eV, g_D = 4e-6: decays and
            scatterings on the relics drain the flux across the Universe;
  le_26  -- L_e - L_mu, 1e-26 eV, g' = 1e-27: the Milky Way's potential opens decays;
  le_20  -- L_e - L_mu, 1e-20 eV, g' = 1e-24: the Sun's potential turns the states into flavors.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "output", "analysis", "trajectory_figure.npz")
#: The sources' redshift shell and the observed energy (eV) of every row.
SHELL = (1.8, 2.2)
E_OBS = 5e14
#: The lightest neutrino mass (eV; normal ordering), the smallest the fits allow; cores of the local runs.
M_LIGHTEST = 1e-6
N_CORES = 4
#: The grid: 10 TeV to 30 PeV, as the method figures (method_figures.GRID), finer in energy than the redshift grid.
GRID = {"E_min_eV": 1e13, "E_max_eV": 3e16, "n_E_bins": 120, "n_z_bins": 80}

ROWS = [("si", r"Secret interactions: scalar on $\nu_\tau$, $M = 10$ MeV, $g = 10^{-3}$"),
        ("dp", r"Dark portal: mixed with $\nu_\tau$, $M = 10^{-2}$ eV, $g_D = 4 \times 10^{-6}$"),
        ("le_26", r"$L_e - L_\mu$: $M = 10^{-26}$ eV, $g' = 10^{-27}$"),
        ("le_20", r"$L_e - L_\mu$: $M = 10^{-20}$ eV, $g' = 10^{-24}$")]
FLAVOR_COLORS = ["#d95f02", "#7570b3", "#1b9e77"]
FLAVORS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
ANTI_DASH = (0, (3.2, 1.6))
#: Fig. 3's panels (analysis.trip_illustration.PANELS, inches across 7.0) and titles.
FIG3 = [(0.04, 2.62), (2.80, 4.30), (4.48, 5.40), (5.58, 6.96)]
TITLES = ["The Universe", "The Milky Way", "Near the Sun", "Inside the Earth"]


def _solver():
    sys.path.insert(0, os.path.join(REPO, "src"))
    os.environ.setdefault("TQDM_DISABLE", "1")
    import dark_portal_solver as dps
    import dp_trajectory as tj
    return dps, tj


def _mixing():
    with open(os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26", "le_lmu_mev.json")) as f:
        return json.load(f)["solver"]["standard_mixing"]


def _params(key, grid):
    dps, _ = _solver()
    base = {"cosmology": {"m_lightest_eV": M_LIGHTEST}, "standard_mixing": _mixing(), "source": {"p_index": 2.5},
            "grid": grid, "execution": {"n_cores": N_CORES, "verbose": 0}}
    if key == "si":
        cfg = dict(base, secret_interactions={"enabled": True, "mediator": "scalar", "neutrino_type": "majorana",
                                              "m_phi_eV": 1e7, "g_phi": 1e-3, "texture": "tau"})
    elif key == "dp":
        cfg = dict(base, model={"name": "dark_portal"},
                   dark_portal={"m_Zd_eV": 1e-2, "g_D": 4e-6, "u_e": 0.0, "u_mu": 0.0, "u_tau": 2 ** -0.5,
                                "epsilon_prime": 0.0, "use_debye_screening": True, "thermal_products": True})
    else:
        m, g = {"le_26": (1e-26, 1e-27), "le_20": (1e-20, 1e-24)}[key]
        cfg = dict(base, model={"name": "L_e-L_mu"},
                   dark_portal={"m_Zd_eV": m, "g_D": g, "u_e": 0.0, "u_mu": 0.0, "u_tau": 0.0, "epsilon_prime": 0.0})
    return dps.build_solver_params(cfg)


#: Per row, the grid's changes: secret interactions on 4 (n - 1) + 1 energies (the default grid's nodes among them,
#: so the same observed energy): dp_si's redshift step is one energy bin, and a resonance that absorbs fully refills
#: the flux one bin at a time -- finer bins, finer steps (sub-steps within a bin show the bins as a staircase instead)
ROW_GRID = {"si": {"n_E_bins": 4 * (GRID["n_E_bins"] - 1) + 1}}


def compute(path=CACHE, grid=None, keys=None):
    """Every row's species along the trip at the observed energy nearest E_OBS: per row, the positions and the fluxes
    (2 kinds, N_pos, 3 flavors) of each stage, saved to `path`."""
    _, tj = _solver()
    grid = dict(GRID, **(grid or {}))
    out = dict(np.load(path)) if (keys and os.path.exists(path)) else {}
    # every row at the default grid's energy nearest E_OBS (a node of the finer rows' grids too)
    E0 = np.geomspace(grid["E_min_eV"], grid["E_max_eV"], grid["n_E_bins"])
    E_target = float(E0[np.argmin(np.abs(np.log(E0 / E_OBS)))])
    for key, _ in ROWS:
        if keys and key not in keys:
            continue
        E, p = _params(key, dict(grid, **ROW_GRID.get(key, {})))
        i = int(np.argmin(np.abs(np.log(E / E_target))))
        u = tj.universe(E, p, source_z=SHELL)
        out[f"{key}/E"] = E[i]
        out[f"{key}/z"] = u["z"]
        out[f"{key}/universe"] = np.stack([u["F_nu"][:, i], u["F_nb"][:, i]])
        if not u["si"]:
            ga = tj.galaxy(E, p, u)
            su = tj.sun(E, u, ga)
            ea = tj.earth(E, u, su)
            for name, x, d in (("galaxy", ga["d_kpc"], ga), ("sun", su["x_au"], su), ("earth", ea["l_km"], ea)):
                out[f"{key}/{name}_x"] = x
                out[f"{key}/{name}"] = np.stack([d["F_nu"][:, i], d["F_nb"][:, i]])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


def load(path=CACHE):
    return dict(np.load(path)) if os.path.exists(path) else compute(path)


def figure(data=None):
    """The figure: one row per benchmark, one column per stage of the trip (Fig. 3's), the flux of each species over
    the total once the shell's sources stop emitting."""
    d = data or load()
    W, H = 7.0, 6.9
    left, right = 0.62, 0.16
    scale = (W - left - right) / (FIG3[-1][1] - FIG3[0][0])
    cols = [(left + (a - FIG3[0][0]) * scale, (b - a) * scale) for a, b in FIG3]
    top, bottom, rgap = 0.98, 0.62, 0.3
    rh = (H - top - bottom - rgap * (len(ROWS) - 1)) / len(ROWS)
    L_chord = float(d["le_20/earth_x"][-1])
    fig = plt.figure(figsize=(W, H))
    for r, (key, label) in enumerate(ROWS):
        y0 = H - top - (r + 1) * rh - r * rgap
        axs = [fig.add_axes([x / W, y0 / H, w / W, rh / H]) for x, w in cols]
        z, U = d[f"{key}/z"], d[f"{key}/universe"]
        norm = U[:, int(np.argmax(z <= SHELL[0]))].sum()
        stages = [(z, U)]
        if f"{key}/galaxy" in d:
            S = d[f"{key}/sun"]
            stages += [(np.maximum(d[f"{key}/galaxy_x"], 2e-3), d[f"{key}/galaxy"]),
                       # no potential acts beyond the Sun path's start: the Galaxy's end out to the panel's edge
                       (np.concatenate([[1e4], np.maximum(d[f"{key}/sun_x"], 5e-5)]),
                        np.concatenate([S[:, :1], S], axis=1)),
                       (d[f"{key}/earth_x"], d[f"{key}/earth"])]
        else:                       # secret interactions: no potential, no local process; the arrival, unchanged
            last = np.repeat(U[:, -1:], 2, axis=1)
            stages += [(np.array([30.0, 2e-3]), last), (np.array([1e4, 5e-5]), last), (np.array([0.0, L_chord]), last)]
        ymax = 0.0
        for ax, (x, F) in zip(axs, stages):
            for s, ls in ((0, "-"), (1, ANTI_DASH)):
                for f in range(3):
                    y = F[s, :, f] / norm
                    ax.plot(x, y, color=FLAVOR_COLORS[f], ls=ls, lw=1.5 if s == 0 else 1.2)
                    ymax = max(ymax, float(np.max(y)))
        for c, ax in enumerate(axs):
            ax.set_ylim(0.0, 1.12 * ymax)
            ax.tick_params(labelsize=8, which="major", length=3.0, width=0.6, pad=2)
            ax.tick_params(which="minor", length=1.5, width=0.5)
            if c > 0:
                ax.set_yticklabels([])
            if r < len(ROWS) - 1:
                ax.set_xticklabels([])
            if r == 0:
                fig.text((cols[c][0] + 0.5 * cols[c][1]) / W, (y0 + rh + 0.3) / H, TITLES[c], fontsize=10,
                         ha="center", va="bottom")
        axs[0].set_xlim(SHELL[1], 0.0)
        axs[0].axvspan(SHELL[0], SHELL[1], color="0.9", lw=0, zorder=0)
        axs[1].set_xscale("log")
        axs[1].set_xlim(30.0, 2e-3)
        axs[1].set_xticks([10.0, 1.0, 0.1, 0.01])
        axs[2].set_xscale("log")
        axs[2].set_xlim(1e4, 5e-5)
        axs[2].set_xticks([1e3, 1e0, 1e-3])
        axs[3].set_xlim(0.0, L_chord)
        axs[3].set_xticks([0, 2000, 4000, 6000, 8000])
        axs[3].xaxis.set_minor_locator(plt.MultipleLocator(1000))
        axs[0].set_ylabel("Flux / total\nat emission", fontsize=9)
        fig.text(cols[0][0] / W, (y0 + rh + 0.05) / H, label, fontsize=8.5, va="bottom", ha="left")
        if r == 0:
            axs[0].text(np.mean(SHELL), 0.5 * ymax, "Sources", rotation=90, ha="center", va="center", fontsize=7.5,
                        color="0.4")
        if r == len(ROWS) - 1:
            axs[0].set_xlabel(r"Redshift, $z$", fontsize=9)
            axs[1].set_xlabel("Distance to\nthe Sun [kpc]", fontsize=9)
            axs[2].set_xlabel("Distance to\nthe Earth [AU]", fontsize=9)
            axs[3].set_xlabel("Distance inside\nthe Earth [km]", fontsize=9)
    handles = [Line2D([], [], color=FLAVOR_COLORS[f], lw=1.5, label=FLAVORS[f]) for f in range(3)]
    handles += [Line2D([], [], color="k", lw=1.5, ls="-", label="Neutrinos"),
                Line2D([], [], color="k", lw=1.2, ls=ANTI_DASH, label="Antineutrinos")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8.5, frameon=True, edgecolor="k",
               fancybox=False, bbox_to_anchor=(0.5, 1.0))
    return fig
