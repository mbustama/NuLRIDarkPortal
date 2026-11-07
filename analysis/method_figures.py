"""
method_figures.py -- the paper's figures of the method (2026-11-07; paper/paper_planning.md, Step 5): Fig. 2, the
flux at the stops of the trip (dark_portal_solver.flux_stops), and Fig. 3, the flux as the collision terms are
switched on one at a time. The cases and settings are those of notebooks/fluxes_at_earth.ipynb (its Secs. 3--5).
compute() runs the solver once (222 s on 4 cores of the laptop, measured 2026-11-07) and caches the fluxes in
output/analysis/method_figures.npz; trip_figure() and collisions_figure() draw from the cache. Called from
notebooks/paper_figures.ipynb, in the repository's style (analysis.figure_style.use() first).
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "output", "analysis", "method_figures.npz")
#: The energy grid of the fluxes notebook: 10 TeV to 30 PeV.
GRID = {"E_min_eV": 1e13, "E_max_eV": 3e16, "n_E_bins": 200, "n_z_bins": 100}
#: The lightest neutrino mass (eV; normal ordering), the smallest the fits allow.
M_LIGHTEST = 1e-6
N_CORES = 4

#: The stages' lines: the first dashed black, on top; the rest colorblind-safe colors (not the data sets' colors),
#: each thinner than the one before and drawn over it, so that a stage that changes nothing shows as a thin line
#: inside a thicker one.
COLORS = ["k", "#7570b3", "#1b9e77", "#e7298a"]
STYLES = ["--", "-", "-", "-"]
WIDTHS = [1.3, 3.0, 2.1, 1.3]
FLAVORS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]

TRIP_CASES = [
    ("le_lmu_20", "L_e-L_mu", 1e-20, 1e-24, r"$L_e - L_\mu$, $M = 10^{-20}$ eV, $g' = 10^{-24}$"),
    ("le_lmu_26", "L_e-L_mu", 1e-26, 1e-27, r"$L_e - L_\mu$, $M = 10^{-26}$ eV, $g' = 10^{-27}$"),
]
STOPS = [("universe", "Arriving at the Milky Way"), ("galaxy", "After the Milky Way"),
         ("earth", "Near the Earth"), ("detector", "At the detector")]

COLLISION_CASES = [
    ("si_mev", r"Secret interactions, scalar on $\nu_\tau$, $M = 10$ MeV, $g = 10^{-3}$", [
        ("No new interaction", dict(kind="gauge", model="L_e-L_mu", m_eV=1e7, g=1e-3, potential=False,
                                    collisions=False)),
        ("+ Resonant absorption", dict(kind="si", m_eV=1e7, g=1e-3, channels="s")),
        ("+ Scattering, pair production", dict(kind="si", m_eV=1e7, g=1e-3)),
        ("+ Relics' thermal motion", dict(kind="si", m_eV=1e7, g=1e-3, thermal=True)),
    ]),
    ("le_lmu_light", r"$L_e - L_\mu$, $M = 10^{-2}$ eV, $g' = 10^{-6}$", [
        ("No new interaction", dict(kind="gauge", model="L_e-L_mu", m_eV=1e-2, g=1e-6, potential=False,
                                    collisions=False)),
        ("+ Potential", dict(kind="gauge", model="L_e-L_mu", m_eV=1e-2, g=1e-6, collisions=False)),
        ("+ Decay", dict(kind="gauge", model="L_e-L_mu", m_eV=1e-2, g=1e-6, scattering=False)),
        ("+ Scattering on relics", dict(kind="gauge", model="L_e-L_mu", m_eV=1e-2, g=1e-6)),
    ]),
]


def _solver():
    sys.path.insert(0, os.path.join(REPO, "src"))
    os.environ.setdefault("TQDM_DISABLE", "1")
    import dark_portal_solver as dps
    return dps


def _mixing():
    with open(os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26", "le_lmu_mev.json")) as f:
        return json.load(f)["solver"]["standard_mixing"]


def _gauge_params(model, m_eV, g, potential=True, collisions=True, decay=True, scattering=True, thermal=False):
    dps = _solver()
    config = {"model": {"name": model}, "cosmology": {"m_lightest_eV": M_LIGHTEST}, "standard_mixing": _mixing(),
              "dark_portal": {"m_Zd_eV": m_eV, "g_D": g, "u_e": 0.0, "u_mu": 0.0, "u_tau": 0.0,
                              "epsilon_prime": 0.0, "neutrino_decay": decay, "relic_scattering": scattering,
                              "thermal_products": thermal},
              "source": {"p_index": 2.5}, "grid": GRID, "execution": {"n_cores": N_CORES, "verbose": 0}}
    E, p = dps.build_solver_params(config)
    if not potential:
        p["potential_product"] = 0.0
    if not collisions:
        p["G_coupling"] = np.zeros((3, 3), dtype=np.complex128)
    return E, p


def _si_params(m_eV, g, channels="full", thermal=False, mediator="scalar", texture="tau"):
    dps = _solver()
    config = {"secret_interactions": {"enabled": True, "mediator": mediator, "neutrino_type": "majorana",
                                      "m_phi_eV": m_eV, "g_phi": g, "texture": texture, "channels": channels,
                                      "thermal_products": thermal},
              "cosmology": {"m_lightest_eV": M_LIGHTEST}, "standard_mixing": _mixing(),
              "source": {"p_index": 2.5}, "grid": GRID, "execution": {"n_cores": N_CORES, "verbose": 0}}
    return dps.build_solver_params(config)


def _flux(kw):
    dps = _solver()
    kw = dict(kw)
    kind = kw.pop("kind")
    E, p = _gauge_params(**kw) if kind == "gauge" else _si_params(**kw)
    return dps.propagation_worker((E, p))


def compute(path=CACHE):
    """Every case's fluxes (energies in eV; flux per flavor, N_E x 3), saved to `path`."""
    dps = _solver()
    out = {}
    for key, model, m, g, _ in TRIP_CASES:
        E, p = _gauge_params(model, m, g)
        st = dps.flux_stops(E, p)
        out[f"{key}/E"] = E
        for s, _ in STOPS:
            out[f"{key}/{s}"] = st[s]
    for key, _, steps in COLLISION_CASES:
        for k, (_, kw) in enumerate(steps):
            E, F = _flux(kw)
            out[f"{key}/E"] = E
            out[f"{key}/{k}"] = F
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


def load(path=CACHE):
    return dict(np.load(path)) if os.path.exists(path) else compute(path)


def _stage_grid(columns, first_label, width_in=7.0, top=0.87):
    """The panels for `columns` cases: the all-flavor flux relative to the first stage, then the fraction of each
    flavor; one line per stage. Returns the figure and the axes [row][column]."""
    ncol = len(columns)
    fig = plt.figure(figsize=(width_in, 6.2))
    gs = fig.add_gridspec(4, ncol, height_ratios=[1.25, 1, 1, 1], hspace=0.12, wspace=0.28,
                          left=0.1, right=0.98, top=top, bottom=0.075)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncol)] for r in range(4)]
    for c, (title, stages) in enumerate(columns):
        E0, F0 = stages[0][1]
        for k, (label, (E, F)) in enumerate(stages):
            kw = dict(color=COLORS[k], ls=STYLES[k], lw=WIDTHS[k], zorder=20 if k == 0 else 3 + k, label=label)
            axes[0][c].plot(E / 1e12, F.sum(axis=1) / F0.sum(axis=1), **kw)
            for i in range(3):
                axes[i + 1][c].plot(E / 1e12, F[:, i] / F.sum(axis=1), **kw)
        axes[0][c].set_title(title, fontsize=10.5, pad=6)
        for r in range(4):
            ax = axes[r][c]
            ax.set_xscale("log")
            ax.set_xlim(GRID["E_min_eV"] / 1e12, GRID["E_max_eV"] / 1e12)
            if r < 3:
                plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(labelsize=10)
        axes[3][c].set_xlabel(r"Neutrino energy, $E$ [TeV]", fontsize=11)
        for i in range(3):
            axes[i + 1][c].set_ylim(-0.05, 1.05)
            axes[i + 1][c].set_yticks([0, 0.5, 1])
        top = max(1.05, 1.08 * max(float(np.max(F.sum(axis=1) / F0.sum(axis=1))) for _, (_, F) in stages))
        axes[0][c].set_ylim(0, top)
    axes[0][0].set_ylabel("Flux relative\n" + first_label, fontsize=10.5)
    for i in range(3):
        axes[i + 1][0].set_ylabel("Fraction of " + FLAVORS[i], fontsize=10.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    return fig, axes, handles, labels


def trip_figure(data=None):
    """Fig. 2: the flux at the four stops of the trip, for two mediator masses."""
    data = data or load()
    columns = []
    for key, _, _, _, title in TRIP_CASES:
        E = data[f"{key}/E"]
        columns.append((title, [(label, (E, data[f"{key}/{s}"])) for s, label in STOPS]))
    fig, axes, handles, labels = _stage_grid(columns, "to its arrival", top=0.82)
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10, frameon=True, edgecolor="k",
               fancybox=False, bbox_to_anchor=(0.54, 0.985))
    return fig


def collisions_figure(data=None):
    """Fig. 3: the flux as the terms of the transport are switched on one at a time, for one MeV and one light
    mediator."""
    data = data or load()
    columns = []
    for key, title, steps in COLLISION_CASES:
        E = data[f"{key}/E"]
        columns.append((title, [(label, (E, data[f"{key}/{k}"])) for k, (label, _) in enumerate(steps)]))
    fig, axes, _, _ = _stage_grid(columns, "to no new interaction")
    for c, (title, _) in enumerate(columns):
        h, l = axes[0][c].get_legend_handles_labels()
        axes[0][c].legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=8.5,
                          frameon=True, edgecolor="k", fancybox=False, columnspacing=1.0, handlelength=1.8)
        axes[0][c].set_title(title, fontsize=10.5, pad=40)
    return fig
