"""
mean_free_paths.py -- the paper's figure of the mean free paths at z = 0, one panel per family of mediators
(2026-11-22, his request: "one plot illustrating the mean free path in each of the three classes", placed before the
figure of the species along the trip).

  L_e - L_mu and the dark portal: the decay lengths of nu_2 and nu_3, c / Gamma_i with Gamma_i the lab-frame rate of
    nu_i -> nu_j + mediator summed over j (dp_decay.rest_rates times m_i / E, Majorana, as the transport), and the
    mean free path of nu_3 for the scatterings on the relics that move it to another state (the transport's soft
    scatterings, dark_portal_solver._soft_scattering_rates at z = 0: Gamma_u sum_{b != a} |G'_ba|^2, with the relics'
    occupation after their own decays, dp_decay.relic_composition, and their screening mass,
    dp_absorption.debye_mass_sq). The damping w^2 / (w^2 + D^2) of these conversions is 1 to better than 1e-13 here
    (checked in tests/test_mean_free_paths.py). The longitudinal mode's collisions (dp_si_light.goldstone_factors)
    are left out, as the transport leaves them out, where the largest Goldstone coupling is below
    dp_si_light.GOLDSTONE_Y_MIN (both examples: ~1e-6); the cosmological potential is negligible at z = 0 for them, so
    the states are the mass states.
  Secret interactions: the scalar with Majorana neutrinos of analysis/si_mean_free_path.py's figure, on all three
    relics (his choice, 2026-11-22; M = 1 MeV, g = 1e-3 on nu_tau), with the arrows on the dip of nu_1.

Couplings (his choice, 2026-11-22: "couplings where curves cross c/H0" inside IceCube's energy range): L_e - L_mu at
M = 1e-20 eV with g' = 1e-25 (nu_2 crosses near 3.5 TeV, nu_3 near 2.3 PeV); the dark portal on nu_tau at
M = 1e-3 eV with g_D sum|U_a4|^2 = 2e-8 (near 1.8 TeV and 8 PeV). Same neutrinos as si_mean_free_path: m_1 = 1e-6 eV,
normal ordering, NuFIT 6.1. Cheap (analytic rates, < 1 s): compute() writes output/analysis/mean_free_paths.npz.
  from analysis import mean_free_paths as mf; fig = mf.figure()       (analysis.figure_style.use() first)
"""
import os

import numpy as np

from analysis import si_mean_free_path as si_mfp

REPO = si_mfp.REPO
CACHE = os.path.join(REPO, "output", "analysis", "mean_free_paths.npz")
HBARC_EV_MPC = 1.973269804e-7 / 3.0856775814913673e22     # hbar c in eV Mpc
H0_KM_S_MPC, OMEGA_M, OMEGA_L = 67.4, 0.315, 0.685         # the configs' cosmology (Planck 2018)
H0_EV = H0_KM_S_MPC / 2.99792458e5 * HBARC_EV_MPC          # as dark_portal_solver's params['H0_eV']

#: The two light-vector examples: key, model, mass (eV), coupling, flavor texture.
EXAMPLES = [("le", "L_e-L_mu", 1e-20, 1e-21), ("dp", "dark_portal_tau", 1e-3, 5.62e-7)]
#: Secret interactions: Fig. 7's scalar (Majorana, nu_tau, 10 MeV, g = 0.18), from si_mean_free_path.compute(M=1e7,
#: g=0.18, models="A") on all three relics.
SI_CACHE = os.path.join(REPO, "output", "analysis", "si_mean_free_path_fig7.npz")
COLORS = {"nu2": "#0a9396", "nu3": "#9b2226", "scat": "#6a4c93"}


def coupling(model, g):
    """The flavor-basis coupling G (3, 3) of the transport: dp_models' charges for L_e - L_mu, g_D u u^dag for the
    dark portal on nu_tau (u = (0, 0, 1), so g = g_D sum |U_a4|^2)."""
    si_mfp.code()
    import dp_absorption
    import dp_models
    if model == "dark_portal_tau":
        return dp_absorption.dark_portal_coupling(g, [0.0, 0.0, 1.0], 0.0)
    return dp_models.neutrino_coupling(model, g, 0.0)


def neutrinos():
    """(U, masses): the mixing matrix and masses of si_mean_free_path's benchmark."""
    core, _ = si_mfp.code()
    U = np.asarray(core.construct_PMNS(*np.radians(si_mfp.MIXING_DEG)), dtype=np.complex128)
    m = np.asarray(core.get_mass_eigenvalues(si_mfp.M_LIGHTEST, si_mfp.DM21_SQ, si_mfp.DM31_SQ, "normal"), float)
    return U, m


def rates(model, M, g, E):
    """The rates at z = 0 (eV): decay (N_E, 3), the lab-frame decay rate of each mass state; scatter (3,), the rate of
    the soft scatterings that move each state to another one (energy independent); and eta, mD2, Ymax, Gu."""
    si_mfp.code()
    import dp_absorption
    import dp_decay
    import dp_si
    import dp_si_light
    U, m = neutrinos()
    G = coupling(model, g)
    Gam, _ = dp_decay.rest_rates(G, U, m, M)
    decay = np.array([dp_decay.total_lab_rate(Gam, m, e) for e in np.atleast_1d(E)])
    eta = dp_decay.relic_composition(Gam, m, np.ones(3), si_mfp.T_NU0, H0_EV, OMEGA_M, OMEGA_L)
    Gm = U.conj().T @ G @ U
    mD2 = dp_absorption.debye_mass_sq(np.abs(np.diag(Gm)) ** 2, m, si_mfp.T_NU0, eta)
    relic = float(np.sum(eta * np.sum(np.abs(Gm) ** 2, axis=1)))
    Gu = 2.0 * dp_si.relic_density_per_helicity(si_mfp.T_NU0) * relic / (4.0 * np.pi * (M * M + mD2))
    scatter = np.array([Gu * sum(abs(Gm[b, a]) ** 2 for b in range(3) if b != a) for a in range(3)])
    Y = dp_si_light.goldstone_yukawa(G, U, m, M)
    Ymax = float(np.max(np.abs(Y)))
    # the longitudinal mode's collisions of nu_3 with the relics, as the transport adds them (dp_si_light.
    # goldstone_factors: L_e - L_alpha only, where the largest Goldstone coupling reaches GOLDSTONE_Y_MIN): dp_si's
    # Majorana-scalar channels with lambda = Y and the collision mass max(M, m_heaviest), on each relic state at its
    # occupation eta_r and temperature T eta_r^(1/3) (the relics after their own decays)
    gold = np.zeros(len(np.atleast_1d(E)))
    model_name = "L_e-L_mu" if model == "L_e-L_mu" else model
    if model_name in dp_si_light.GOLDSTONE_MODELS and Ymax >= dp_si_light.GOLDSTONE_Y_MIN:
        Mc = max(float(M), float(np.max(m)))
        Gam_c = dp_si.mediator_width(Mc, Y, "scalar", True)
        x, w = np.polynomial.legendre.leggauss(si_mfp.NODES)
        for r in range(3):
            if eta[r] <= 0.0:
                continue
            case = dict(key="A", Gam=Gam_c, cols=si_mfp.SCALAR_CHANNELS, coupling=Y, m=float(m[r]), M=Mc,
                        T=si_mfp.T_NU0 * float(eta[r]) ** (1.0 / 3.0),
                        terms=[(0, lambda s, r=r: dp_si.sigma_scalar_majorana_channels(s, Mc, Gam_c, Y, 2, r,
                                                                                      "coherent"))])
            gold += np.array([si_mfp.thermal_rate(e, case, x, w)[:5].sum() for e in np.atleast_1d(E)])
    return dict(decay=decay, scatter=scatter, eta=eta, mD2=mD2, Ymax=Ymax, Gu=Gu, goldstone=gold)


def compute(path=CACHE):
    """The decay and scattering lengths (Mpc) of the two examples on the SI cache's energies; writes the cache."""
    E = si_mfp.load(SI_CACHE)[0]
    out = dict(E_eV=E)
    for key, model, M, g in EXAMPLES:
        r = rates(model, M, g, E)
        Eg = E[::10]                                   # the longitudinal mode's collisions: smooth, every 10th energy
        rg = rates(model, M, g, Eg)["goldstone"]
        out[f"{key}_decay_mpc"] = HBARC_EV_MPC / np.where(r["decay"] > 0, r["decay"], np.nan)
        out[f"{key}_scatter_mpc"] = HBARC_EV_MPC / r["scatter"]
        out[f"{key}_goldstone_E_eV"] = Eg
        out[f"{key}_goldstone_mpc"] = HBARC_EV_MPC / np.where(rg > 0, rg, np.nan)
        out[f"{key}_eta"] = r["eta"]
        out[f"{key}_Ymax"] = np.array(r["Ymax"])
    if path:
        np.savez(path, **out)
    return out


def common_xlabel(fig, axes, text, size=22):
    """One x label for a row of panels, centred under them, just below their tick labels (measured on the drawn
    figure)."""
    fig.canvas.draw()
    ren, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    y = min(inv.transform(ax.get_tightbbox(ren).p0)[1] for ax in axes)
    x = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
    fig.text(x, y - 0.01, text, ha="center", va="top", fontsize=size)


def _along(ax, x0, y0, x1, y1):
    """The angle (degrees) on screen of the segment (x0, y0)-(x1, y1) in data coordinates."""
    p0, p1 = ax.transData.transform([(x0, y0), (x1, y1)])
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))


def figure(path=CACHE):
    """One row, three panels: L_e - L_mu, the dark portal, secret interactions; c/Gamma in Mpc versus E in GeV, the
    y axis of the 4 x 4 (0.1 to 1e15 Mpc), the Hubble length dashed; one legend across the top; the decays labelled
    along their lines too; in the third panel, the 4 x 4's schematic arrows."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
    from matplotlib.transforms import blended_transform_factory
    from analysis.figure_style import PANEL

    d = dict(np.load(path))
    E, rows, cols, _ = si_mfp.load(SI_CACHE)
    EG = E / 1e9
    y_lo, y_hi = 1e-8, 1e15
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(3 * 1.05 * PANEL[0], 1.2 * PANEL[1]))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.70, bottom=0.13, wspace=0.05)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(si_mfp.E_FIG_MIN / 1e9, si_mfp.E_FIG_MAX / 1e9)
        ax.set_ylim(y_lo, y_hi)
    titles = ["$L_e - L_\\mu$\n" r"$M = 10^{-20}$ eV, $g' = 10^{-21}$",
              "Dark portal, coupled to $\\nu_\\tau$\n"
              r"$M = 10^{-3}$ eV, $g_D |U_{\tau 4}|^2 = 5.6 \times 10^{-7}$",
              "Secret interactions, scalar $\\phi$, Majorana $\\nu$\n"
              r"$M$ = 10 MeV, $g = 0.18$ (coupled to $\nu_\tau$), all relics"]
    for ax, (key, *_r) in zip(axes[:2], EXAMPLES):
        dec, sc = d[f"{key}_decay_mpc"], d[f"{key}_scatter_mpc"]
        for i, name, frac, f, va in ((1, r"Decay of $\nu_2$", 0.22, 4.0, "bottom"),
                                     (2, r"Decay of $\nu_3$", 0.55, 0.25, "top")):
            color = COLORS["nu2" if i == 1 else "nu3"]
            ax.plot(EG, dec[:, i], color=color, lw=2.0)
            j = int(frac * (len(EG) - 1))
            ax.text(EG[j], dec[j, i] * f, name, color=color, fontsize=16, ha="left", va=va,
                    rotation=_along(ax, EG[j], dec[j, i], EG[j + 50], dec[j + 50, i]), rotation_mode="anchor")
        gE, gl = d.get(f"{key}_goldstone_E_eV"), d.get(f"{key}_goldstone_mpc")
        if gl is not None and np.isfinite(gl).any():      # the longitudinal mode's collisions (L_e - L_mu)
            ax.plot(gE / 1e9, 1.0 / (1.0 / gl + 1.0 / sc[2]), color=COLORS["scat"], lw=2.0)
        elif sc[2] < y_hi:
            ax.axhline(sc[2], color=COLORS["scat"], lw=2.0)
        else:
            ax.annotate("", xy=(0.06, 0.84), xytext=(0.06, 0.70), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=COLORS["scat"], lw=1.6, mutation_scale=16))
            ax.text(0.09, 0.77, "Scatterings of $\\nu_3$ on the relics:\nat $\\sim 10^{%d}$ Mpc, off the scale"
                    % int(round(np.log10(sc[2]))), transform=ax.transAxes, fontsize=15, color=COLORS["scat"],
                    ha="left", va="center")
    # secret interactions: the 4 x 4's scalar with Majorana neutrinos, on all three relics (his choice, 2026-11-22)
    ax = axes[2]
    R = rows["A"][0] + rows["A"][1] + rows["A"][2]
    total = R[:, :5].sum(axis=1)
    idx = [cols["A"].index(q) for q in si_mfp.INTERFERENCE if q in cols["A"]]
    ax.plot(EG, si_mfp.mean_free_path_mpc(total - R[:, idx].sum(axis=1)), color="#ae2012", lw=1.5, zorder=4)
    ax.plot(EG, si_mfp.mean_free_path_mpc(total), color="k", lw=2.0, zorder=5)
    for q, color in (("s_channel", "#005f73"), ("t_and_u", "#ee9b00"), ("phi_phi_pairs", "#ca6702")):
        ax.plot(EG, si_mfp.mean_free_path_mpc(R[:, cols["A"].index(q)]), color=color, lw=1.3)
    # the schematic arrows from the dip (the narrow-width s channel): larger M along lambda ~ E, larger g down
    i = int(np.argmax(total))                      # the arrows on the deepest dip (the resonance on the relic nu_3)
    to_axes = ax.transData + ax.transAxes.inverted()
    x0, y0 = to_axes.transform((EG[i], si_mfp.C_CM_S / total[i] / si_mfp.MPC_CM))
    dd = np.array([1.0 / np.log10(si_mfp.E_FIG_MAX / si_mfp.E_FIG_MIN), 1.0 / np.log10(y_hi / y_lo)])
    arrow = dict(arrowstyle="-|>", lw=1.5, color="0.15", mutation_scale=16, shrinkA=0, shrinkB=0)
    tl = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
    ax.annotate("", xy=(x0 + 0.02 + dd[0], y0 - 0.025 + dd[1]), xytext=(x0 + 0.02, y0 - 0.025),
                xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrow, zorder=8)
    ax.text(x0 + 0.04 + 0.6 * dd[0], y0 - 0.068 + 0.6 * dd[1], r"Larger $M$", transform=ax.transAxes, ha="center",
            va="top", fontsize=16, color="0.15", bbox=tl, zorder=7)
    ax.annotate("", xy=(x0, y0 - 0.115), xytext=(x0, y0 - 0.02), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=arrow, zorder=8)
    ax.text(x0 - 0.02, y0 - 0.07, r"Larger $g$", transform=ax.transAxes, ha="right", va="center", fontsize=16,
            color="0.15", bbox=tl, zorder=7)
    axes[0].text(0.04, si_mfp.HUBBLE_MPC * 10 ** 0.3, r"Hubble length, $c/H_0$", fontsize=15, color="0.3",
                 ha="left", va="bottom", transform=blended_transform_factory(axes[0].transAxes, axes[0].transData))
    for ax in axes:
        ax.axhline(si_mfp.HUBBLE_MPC, color="0.35", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6, 1e7]))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1e-5, 1e0, 1e5, 1e10, 1e15]))
        ax.yaxis.set_minor_locator(FixedLocator(10.0 ** np.arange(-8, 16)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=18)
    axes[0].set_ylabel(r"Mean free path at $z = 0$, $c/\Gamma$ [Mpc]", fontsize=22)
    common_xlabel(fig, axes, r"Neutrino energy, $E$ [GeV]")
    L = lambda color, lw, label: plt.Line2D([], [], color=color, lw=lw, label=label)
    handles = [L(COLORS["nu2"], 2.0, r"Decay of $\nu_2$"), L(COLORS["nu3"], 2.0, r"Decay of $\nu_3$"),
               L(COLORS["scat"], 2.0, r"Scatterings of $\nu_3$ on the relics"),
               L("k", 2.0, r"Total (secret interactions, $\nu_3$ on the relics)"),
               L("#ae2012", 1.5, "Total without interference"), L("#005f73", 1.3, r"$s$ channel alone"),
               L("#ee9b00", 1.3, r"$t$ and $u$ channels alone"), L("#ca6702", 1.3, "Pair production alone")]
    # the examples above their panels, in rounded boxes as wide as the panels, black-edged (his rules, 2026-11-22)
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()
    ren, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    gap, pad = 0.03, 0.012
    texts = []
    for ax, title in zip(axes, titles):
        pos = ax.get_position()
        texts.append(fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + gap + pad, title, ha="center", va="bottom",
                              fontsize=16, linespacing=1.3, multialignment="center", zorder=3))
    fig.canvas.draw()
    top = max(inv.transform(t.get_window_extent(ren).p1)[1] for t in texts) + pad
    for ax in axes:
        pos = ax.get_position()
        fig.patches.append(FancyBboxPatch((pos.x0, pos.y1 + gap), pos.x1 - pos.x0, top - pos.y1 - gap,
                                          boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                          facecolor="white", edgecolor="black", lw=1.0, zorder=2,
                                          mutation_aspect=fig.get_figwidth() / fig.get_figheight()))
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, 0.995, x1 - x0, 0), mode="expand",
                     borderaxespad=0.0, ncol=4, fontsize=16, frameon=True, framealpha=1, edgecolor="black",
                     handlelength=2.4)
    leg.get_frame().set_linewidth(1.0)
    return fig
