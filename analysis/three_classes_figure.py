"""
three_classes_figure.py -- the paper's figure of what each ingredient of the calculation does, one column per model
class (2026-11-21, Mauricio: "All models are treated equal, we don't single out the lepton-number models"; first
drafted as Fig. 1, moved by him on 2026-11-22). In each column the ingredients are added one at a time. Top: the flavor
composition of the starting events (MESE-event-weighted over 5 TeV - 10 PeV, as in the fits;
analysis.mese.event_weighted_fractions) against IceCube's MESE flavor likelihood at 68% and 95% C.L. Bottom: the all-flavor flux at the detector relative to no new interaction.

compute() runs the solver for every column and caches the fluxes in output/analysis/three_classes_figure.npz;
figure() draws from the cache. Called from notebooks/paper_figures.ipynb (Sec. 9), in the repository's style
(analysis.figure_style.use() first).

Columns (each example sits just inside our 95% C.L. excluded region; the solver settings are those of the planes it
belongs to, extragalactic flux only):
  le -- L_e - L_mu, M = 1e-20 eV, g' = 1e-21 (campaign_final_sm le_lmu_ul): the earlier estimate (potential at the
        detector only, Agarwalla:2023sng), + the potentials along the whole trip (decays and relic scatterings off),
        + decays. 'full' (+ relic scatterings) is cached too: it differs from 'decays' by little here, so not drawn.
  dp -- the dark portal mixed with nu_tau, M = 1e-3 eV, g_D sum|U_a4|^2 = 5.62e-7 (campaign_final dp_tau_ul):
        decays (relic scatterings off), + relic scatterings.
  si -- secret interactions, a Majorana scalar on nu_tau, M = 10 MeV, g = 0.18 (campaign_si si_scalar_majorana_tau):
        absorption on the relics (the secondaries dropped: dp_si.propagate_si(regeneration=False)), + the secondaries.
The reference ('none') turns the new interaction off: no potential and no coupling (le), g_D sum|U|^2 = 1e-20 (dp),
g = 1e-8 (si).

The cache of 2026-11-21 was computed on Tycho with these settings, by the scripts that compute() ports
(output/figures/fig1_draft/bench_*.py; checkouts nulri_set1_sm, nulri_set1_final, nulri_refit_set, whose solver
configurations are identical to this repository's), and merged by build_cache_from_bench(). compute() itself has not
been rerun yet.
"""
import copy
import functools
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "output", "analysis", "three_classes_figure.npz")
CONFIGS = os.path.join(REPO, "analysis", "configs")

#: The energies over which the flavor composition is averaged, as in the fits to IceCube's MESE flavor likelihood
#: (data.flavor_band_eV of the set-1 configurations, made by tools/make_set_configs.py).
FLAVOR_BAND_EV = (5e12, 1e16)

#: Colors: the earlier estimate gray, the intermediate step teal, the full calculation red; IceCube blue.
GRAY, TEAL, RED, IC = "0.35", "#0a9396", "#9b2226", "#4477aa"

#: Per column: key, config, solver section, the example's parameters, its reference ('none'), and the steps:
#: (name, switches in the section, trip, regeneration of the secondaries).
COLUMNS = [
    ("le", "campaign_final_sm/now_ic26/le_lmu_ul__b0.json", "dark_portal", dict(m_Zd_eV=1e-20, g_D=1e-21), None,
     [("detector_only", {}, "detector", True),
      ("potentials", {"neutrino_decay": False, "relic_scattering": False}, None, True),
      ("decays", {"relic_scattering": False}, None, True),
      ("full", {}, None, True)]),
    ("dp", "campaign_final/now_ic26/dp_tau_ul__b0.json", "dark_portal", dict(m_Zd_eV=1e-3, g_D_trace=5.62e-7),
     dict(m_Zd_eV=1e-3, g_D_trace=1e-20),
     [("decays", {"relic_scattering": False}, None, True),
      ("full", {}, None, True)]),
    ("si", "campaign_si/now_ic26/si_scalar_majorana_tau.json", "secret_interactions", dict(m_phi_eV=1e7, g_phi=0.18),
     dict(m_phi_eV=1e7, g_phi=1e-8),
     [("absorption", {}, None, False),
      ("full", {}, None, True),
      # the earlier treatment (Bustamante et al. 2020, Esteban et al. 2021): every relic at rest, secondaries kept
      # (his request 2026-11-22; drawn in the flux panel only, not a step of the ingredients)
      ("rest", {"thermal_products": False}, None, True)]),
]

#: What the figure draws per column: title, parameters, steps (cache key, color), and the step labels inside the
#: triangle: (point index: 0 = no new interaction, n = step n; text; position in triangle coordinates).
PANELS = {
    "le": (r"$L_e - L_\mu$", r"$M = 10^{-20}$ eV, $g' = 10^{-21}$",
           [("detector_only", GRAY), ("potentials", TEAL), ("decays", RED)],
           [(1, "Earlier\nestimate", (0.50, 0.675)), (2, "+ Potentials\nalong the trip", (0.25, 0.05)),
            (3, "+ Decays", (0.75, 0.025))]),
    "dp": (r"Dark portal, mixed with $\nu_\tau$", r"$M = 10^{-3}$ eV, $g_D \sum_\alpha |U_{\alpha 4}|^2 = 5.6 \times 10^{-7}$",
           [("decays", TEAL), ("full", RED)],
           [(0, "No new\ninteraction", (0.24, 0.05)), (1, "Decays", (0.55, 0.03)),
            (2, "+ Relic\nscatterings", (0.875, 0.04))]),
    "si": (r"Secret interactions, scalar on $\nu_\tau$", r"$M = 10$ MeV, $g = 0.18$",
           [("absorption", TEAL), ("full", RED)],
           [(1, "Absorption\non relics", (0.48, 0.585)), (2, "+ Secondaries", (0.64, 0.03))]),
}


def _solver_params(section, values, sec, switches):
    """The column's solver configuration with the example's values; g_D_trace = g_D sum|U_a4|^2 sets g_D."""
    sc = copy.deepcopy(section)
    v = dict(values)
    if "g_D_trace" in v:
        d = sc["dark_portal"]
        v["g_D"] = v.pop("g_D_trace") / (d["u_e"] ** 2 + d["u_mu"] ** 2 + d["u_tau"] ** 2)
    sc[sec].update(**v, **switches)
    return sc


def compute(columns=None, n_cores=8, path=CACHE):
    """Runs the solver for the columns (all by default) and writes the cache (merged with what is there)."""
    sys.path[:0] = [REPO, os.path.join(REPO, "src")]
    import dark_portal_solver as dps
    import dp_si
    from analysis.model import split_analysis_keys

    out = dict(np.load(path)) if os.path.exists(path) else {}
    for key, cfg, sec, values, ref, steps in COLUMNS:
        if columns and key not in columns:
            continue
        base = copy.deepcopy(split_analysis_keys(json.load(open(os.path.join(CONFIGS, cfg)))["solver"])[0])
        base["galactic"] = dict(base.get("galactic", {}), enabled=False)
        base["execution"] = {"n_cores": n_cores, "verbose": 0}
        orig = dp_si.propagate_si
        try:
            for name, switches, trip, regen in steps:
                E, p = dps.build_solver_params(_solver_params(base, values, sec, switches))
                p["show_progress"] = False
                if trip:
                    p = dict(p, trip=trip)
                # absorption alone: the secondaries dropped (not a configuration switch; validation option)
                dp_si.propagate_si = orig if regen else functools.partial(orig, regeneration=False)
                out[f"{key}_{name}"] = dps.propagate_weighted(E, p)
                dp_si.propagate_si = orig
            if ref is None:   # L_e - L_alpha: no potential, no coupling
                E, p = dps.build_solver_params(_solver_params(base, values, sec, {}))
                p = dict(p, show_progress=False, potential_product=0.0, G_coupling=np.zeros((3, 3), complex))
            else:
                E, p = dps.build_solver_params(_solver_params(base, ref, sec, {}))
                p["show_progress"] = False
            out[f"{key}_none"] = dps.propagate_weighted(E, p)
        finally:
            dp_si.propagate_si = orig
        if "E" in out and not np.allclose(out["E"], E):
            raise ValueError(f"column {key}: energy grid differs from the cached one")
        out["E"] = E
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out


def _xy(f):
    """Ternary (f_e, f_mu) -> Cartesian, f_e along the bottom edge, f_mu up the right edge."""
    return np.array([f[0] + 0.5 * f[1], 0.5 * np.sqrt(3.0) * f[1]])


def figure(path=CACHE):
    import matplotlib.pyplot as plt
    import ternary
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    sys.path[:0] = [REPO]
    from analysis import mese, surfaces

    FX = np.load(path)
    E = FX["E"]
    fs, lfs = 7.5, 5.5          # axis-label and step-label font sizes
    # IceCube's MESE flavor likelihood on a grid (Delta chi^2 from its minimum)
    g = np.linspace(0, 1, 201)
    FE, FM = np.meshgrid(g, g)
    C = np.full(FE.shape, np.nan)
    for i, j in zip(*np.where(FE + FM <= 1)):
        C[i, j] = surfaces.chi2(FE[i, j], FM[i, j], "icecube_mese_2025_grid")
    X, Y = _xy((FE, FM))

    fig = plt.figure(figsize=(7.0, 3.9))
    for c, key in enumerate(("le", "dp", "si")):
        name, par, steps, labels = PANELS[key]
        band = (E >= FLAVOR_BAND_EV[0] * (1.0 - 1e-9)) & (E <= FLAVOR_BAND_EV[1] * (1.0 + 1e-9))
        comp = lambda k: mese.event_weighted_fractions(E[band], FX[f"{key}_{k}"][band])
        x0 = c / 3.0
        fig.patches.append(FancyBboxPatch((x0 + 0.012, 0.012), 0.31, 0.976, boxstyle="round,pad=0,rounding_size=0.015",
                                          transform=fig.transFigure, fill=False, lw=0.7, ec="0.4", zorder=0))
        fig.text(x0 + 0.167, 0.972, name, ha="center", va="top", fontsize=8.5)
        fig.text(x0 + 0.167, 0.922, par, ha="center", va="top", fontsize=7.5)

        # top: the flavor triangle
        ax = fig.add_axes([x0 + 0.031, 0.355, 0.272, 0.53])
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=1)
        ax.axis("off")
        tax.set_background_color(color="white", alpha=1.0)     # the library's default is a light gray
        tax.boundary(linewidth=0.7)
        tax.gridlines(color="gray", multiple=0.2, linewidth=0.35, ls="-", alpha=0.5)
        tax.left_axis_label(r"$f_{\tau, \oplus}$", fontsize=fs, offset=0.16)
        tax.right_axis_label(r"$f_{\mu, \oplus}$", fontsize=fs, offset=0.16)
        tax.bottom_axis_label(r"$f_{e, \oplus}$", fontsize=fs, offset=0.075)
        tax.ticks(axis="blr", linewidth=0.35, multiple=0.2, offset=0.025, clockwise=False, tick_formats="%.1f",
                  fontsize=6)
        ax.contourf(X, Y, C, levels=[0, 2.30], colors=[IC], alpha=0.85, zorder=2)
        ax.contourf(X, Y, C, levels=[2.30, 5.99], colors=[IC], alpha=0.35, zorder=2)
        # the points: no new interaction, then each step; points closer than about one marker are drawn side by side
        # around their mean (display only)
        keys = ["none"] + [k for k, _ in steps]
        cols = ["k"] + [col for _, col in steps]
        P = [_xy(comp(k)) for k in keys]
        shown = [p.copy() for p in P]
        i = 0
        while i < len(P):
            j = i
            while j + 1 < len(P) and np.hypot(*(P[j + 1] - P[i])) < 0.04:
                j += 1
            if j > i:
                mean = np.mean(P[i:j + 1], axis=0)
                for q in range(i, j + 1):
                    shown[q] = mean + np.array([(q - (i + j) / 2.0) * 0.055, 0.0])
            i = j + 1
        for q in range(1, len(P)):
            if np.hypot(*(P[q] - P[q - 1])) > 0.04:
                rad = 0.3 if (key == "le" and q < 3) else 0.0
                ax.add_patch(FancyArrowPatch(tuple(shown[q - 1]), tuple(shown[q]), arrowstyle="-|>", mutation_scale=7,
                                             lw=1.0, color=cols[q], zorder=6, shrinkA=3, shrinkB=5,
                                             connectionstyle=f"arc3,rad={rad}"))
        for q, p in enumerate(shown):
            if q == 0:
                ax.scatter(*p, s=12, color="k", zorder=8)
            else:
                ax.scatter(*p, s=58, color=cols[q], edgecolor="w", lw=0.9, zorder=7)
                ax.scatter(*p, s=14, marker=r"$%d$" % q, color="w", lw=0, zorder=8)
        ax.set_aspect("equal")
        tax.clear_matplotlib_ticks()
        tax._redraw_labels()
        for q, text, pos in labels:
            ax.annotate(text, xy=tuple(shown[q]), xytext=pos, fontsize=lfs, ha="center", va="center", linespacing=1.0,
                        zorder=9, arrowprops=dict(arrowstyle="-", lw=0.4, color="0.25", shrinkA=1.5,
                                                  shrinkB=3.5 if q else 2.5))
        if key == "dp":   # the MESE regions, labeled once, in the central triangle
            for xy_r, xy_t, t, rel in (((0.50, 0.40), (0.50, 0.553), r"MESE 68\% C.L.", (0.5, 0.0)),
                                       ((0.71, 0.43), (0.50, 0.600), r"MESE 95\% C.L.", (1.0, 0.1))):
                ax.annotate(t, xy=xy_r, xytext=xy_t, fontsize=4.8, ha="center", va="center", zorder=9,
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.25", shrinkA=0.5, shrinkB=0, relpos=rel))

        # bottom: the all-flavor flux relative to no new interaction
        bx = fig.add_axes([x0 + (0.085 if c == 0 else 0.167 - 0.1075), 0.1, 0.215, 0.22])
        ref = FX[f"{key}_none"].sum(axis=1)
        bx.axhline(1.0, color="k", lw=0.6, ls=":", zorder=1)
        for k, color in steps:
            bx.plot(E, FX[f"{key}_{k}"].sum(axis=1) / ref, color=color, lw=1.2,
                    ls="-" if color == RED else (0, (3, 1.6)))
        if key == "si" and "si_rest" in FX:   # the full calculation with every relic at rest (his request 2026-11-22)
            bx.plot(E, FX["si_rest"].sum(axis=1) / ref, color=GRAY, lw=1.0, ls=(0, (1, 1.2)), zorder=6)
            bx.text(1.4e13, 1.6e-3, "Dotted: relics at rest", fontsize=5.5, color=GRAY, ha="left", va="bottom")
        bx.set_xscale("log")
        bx.set_xlim(1e13, 3e16)
        bx.set_yscale("log")
        bx.set_ylim(1e-3, 3)
        bx.set_xlabel(r"Neutrino energy, $E$ [eV]", fontsize=fs, labelpad=1)
        if c == 0:
            bx.set_ylabel("Flux relative to\nno new interaction", fontsize=fs, labelpad=2)
        bx.tick_params(labelsize=6.5, which="major", length=2.5, width=0.5)
        bx.tick_params(axis="x", which="both", pad=2.4)
        bx.tick_params(axis="y", which="both", pad=3.2)
        bx.tick_params(which="minor", length=1.3, width=0.4)
        for s in bx.spines.values():
            s.set_linewidth(0.6)
    return fig


def build_cache_from_bench(bench_dir, path=CACHE):
    """The cache from the 2026-11-21 Tycho runs (bench_clean.npz, bench_dp_1e-3.npz, bench_si_018.npz)."""
    le = np.load(os.path.join(bench_dir, "bench_clean.npz"))
    dp = np.load(os.path.join(bench_dir, "bench_dp_1e-3.npz"))
    si = np.load(os.path.join(bench_dir, "bench_si_018.npz"))
    for d in (dp, si):
        if not np.allclose(d["E"], le["E"]):
            raise ValueError("energy grids differ")
    out = {"E": le["E"]}
    out.update({f"le_{k}": le[k] for k in ("detector_only", "potentials", "decays", "full", "none")})
    out.update({"dp_decays": dp["5.6e-07_decays"], "dp_full": dp["5.6e-07_full"], "dp_none": dp["none"]})
    out.update({"si_absorption": si["0.18_absorption"], "si_full": si["0.18_full"], "si_none": si["none"]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    return out
