"""
secondary_spectra.py -- the energy spectra of the neutrinos that each process returns, one panel per family of
mediators (2026-11-22, his request "Draft it, one panel per family"), for an incident nu_3, as a companion to the
figure of the mean free paths (analysis/mean_free_paths.py), with the same examples.

The quantity is dN/dx, the number of neutrinos that come out of one process per unit x = E_out / E_in, the energy of
an outgoing neutrino over that of the incident one; its integral is the number of neutrinos per process.
  L_e - L_mu and the dark portal (Fig. 7's examples, mean_free_paths.EXAMPLES): the daughter of a decay nu_3 -> nu_j +
    mediator, summed over j with the branching ratios (dp_decay.rest_rates), split into the daughter that keeps the
    helicity (a neutrino) and the one that flips it (an antineutrino; Majorana): dp_decay.daughter_spectra, the
    spectrum a + b x + c / x on [x-, x+]. One neutrino per decay. The mediator takes 1 - x: for L_e - L_mu at 1e-20 eV
    it cannot decay into neutrinos; for the dark portal at 1e-3 eV it can, but at the example's coupling its decay
    length exceeds 1e8 Mpc (dp_mediator.pair_widths), so it carries its energy away. The soft scatterings on the
    relics turn nu_3 into nu_1 or nu_2 at the same energy, x = 1 (drawn as an arrow).
  Secret interactions (scalar, Majorana, M = 10 MeV, g = 0.18 on nu_tau, Fig. 7's example): the
    products of one collision of nu_3 on a relic nu_3 at rest, each channel as if it acted alone, from dp_si's
    coupling tensors (Cabs, Creg) and profiles: the s channel, sum_c Creg[0] / Cabs[0] (flat); the t and u channels,
    sum_c (Creg[1] P2 + Creg[2] P3) / (Cabs[1] N2); pair production, sum_c Creg[6] p_phiphi(x) / Cabs[4]. Two
    neutrinos per 2 -> 2 collision, four per pair production. At s = 10 M^2, which is E = 10 PeV on a relic nu_3
    (s = 2 E m_3). Interferences, which have no spectrum of their own, are left out, as in the mean free paths.
Analytic, < 1 s; no cache.
  from analysis import secondary_spectra as ss; fig = ss.figure()       (analysis.figure_style.use() first)
"""
import numpy as np

from analysis import mean_free_paths as mf
from analysis import si_mean_free_path as si_mfp

S_OVER_M2 = 10.0          # the secret-interaction panel's s / M^2 (E = 10 PeV on relic nu_3 at M = 10 MeV)
SI_M, SI_G = 1e7, 0.18    # Fig. 7's scalar (the spectra's shapes depend on s / M^2 only)
X = np.linspace(0.0, 1.0, 2001)
COLORS = {"decay": mf.COLORS["nu3"], "scat": mf.COLORS["scat"], "s": "#005f73", "tu": "#ee9b00", "pairs": "#ca6702"}


def decay_spectra(model, M, g, x=X):
    """(keep, flip) (len(x),): dN/dx of the daughter of a decay of nu_3 (helicity -1), summed over the lighter states
    with the branching ratios; keep + flip integrates to 1."""
    si_mfp.code()
    import dp_decay
    U, m = mf.neutrinos()
    G = mf.coupling(model, g)
    Gam, _ = dp_decay.rest_rates(G, U, m, M)
    spec = dp_decay.daughter_spectra(G, U, m, M)
    BR = Gam[2] / Gam[2].sum()
    out = np.zeros((2, len(x)))
    xs = np.where(x > 0, x, 1.0)
    for j in range(2):
        for rel in range(2):
            zm, zp, a, b, c = spec[0, 2, j, rel]
            if zp <= 0.0:
                continue
            inside = (x >= zm) & (x <= zp) & (x > 0)
            out[rel] += BR[j] * np.where(inside, a + b * xs + c / xs, 0.0)
    return out[0], out[1]


def si_spectra(s_over_M2=S_OVER_M2, x=X, a=2, r=2):
    """dict s, tu, pairs (len(x),): dN/dx of the products of one collision of mass state a on relic r at rest, each
    channel alone, for si_mean_free_path's scalar with Majorana neutrinos, at s = s_over_M2 M^2."""
    _, si = si_mfp.code()
    _, cases, _ = si_mfp.cases(r, SI_M, SI_G)
    lam = cases[0]["coupling"]
    Cabs, Creg, _ = si.coupling_tensors(lam)
    mu = 1.0 / s_over_M2
    N2, _, _ = si.profile_norms(s_over_M2, si._GL24_X, si._GL24_W)
    v = 1.0 - x
    P2, P3 = v * v / ((v + mu) ** 2), x * x / ((x + mu) ** 2)
    out = dict(s=np.full(len(x), Creg[0, a, r].sum() / Cabs[0, a, r]),
               tu=(Creg[1, a, r].sum() * P2 + Creg[2, a, r].sum() * P3) / (Cabs[1, a, r] * N2))
    # pair production: the neutrinos' spectrum per neutrino (normalized to 1), bin by bin on the grid of x
    edges = np.linspace(1.0, 0.0, len(x))
    p = np.zeros(len(x) - 1)
    si._phiphi_nu_bins(mu, edges, 4000, p)
    dens = p / (edges[:-1] - edges[1:])                    # per unit x, at the bins' centres (descending)
    xc = 0.5 * (edges[:-1] + edges[1:])
    out["pairs"] = Creg[6, a, r].sum() / Cabs[4, a, r] * np.interp(x, xc[::-1], dens[::-1])
    return out


def figure():
    """One row, three panels (the examples of the mean-free-path figure): dN/dx versus x, linear axes, the
    headers as that figure's."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import FixedLocator, NullFormatter
    from analysis.figure_style import PANEL

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(3 * 1.05 * PANEL[0], 1.2 * PANEL[1]))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.70, bottom=0.13, wspace=0.05)
    y_hi = 20.0
    for ax, (key, model, M, g) in zip(axes[:2], mf.EXAMPLES):
        keep, flip = decay_spectra(model, M, g)
        ax.plot(X, keep, color=COLORS["decay"], lw=2.0)
        ax.plot(X, flip, color=COLORS["decay"], lw=2.0, ls=(0, (5, 2.5)))
        ax.plot(X, keep + flip, color="0.35", lw=1.4, ls=(0, (1, 1.5)))
        ax.annotate("", xy=(0.975, 4.0), xytext=(0.975, 1.2e-2), arrowprops=dict(arrowstyle="-|>", lw=2.0,
                    color=COLORS["scat"], mutation_scale=18, shrinkA=0, shrinkB=0))
        ax.text(0.945, 7.0, "Scatterings on the relics: $\\nu_3$ turns\n"
                r"into $\nu_1$ or $\nu_2$, keeping its energy", color=COLORS["scat"], fontsize=15, ha="right",
                va="center")
    si = si_spectra()
    ax = axes[2]
    for k in ("s", "tu", "pairs"):
        ax.plot(X, si[k], color=COLORS[k], lw=2.0)
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, y_hi)
        ax.tick_params(labelsize=18)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticks(np.arange(0.05, 1.0, 0.05), minor=True)
        ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1e0, 1e1]))
        ax.yaxis.set_minor_locator(FixedLocator([k * 10.0 ** e for e in (-2, -1, 0, 1) for k in range(2, 10)]))
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel(r"Neutrinos returned, $dN/dx$", fontsize=22)
    mf.common_xlabel(fig, axes, r"Energy fraction, $x = E_{\rm out}/E_{\rm in}$")
    titles = ["$L_e - L_\\mu$; decay of $\\nu_3$\n" r"$M = 10^{-20}$ eV, $g' = 10^{-21}$",
              "Dark portal, coupled to $\\nu_\\tau$; decay of $\\nu_3$\n"
              r"$M = 10^{-3}$ eV, $g_D |U_{\tau 4}|^2 = 5.6 \times 10^{-7}$",
              "Secret interactions, scalar $\\phi$, Majorana $\\nu$\n"
              r"$M$ = 10 MeV, $g = 0.18$; $\nu_3$ on relic $\nu_3$, $s = 10\,M^2$"]
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
    L = lambda **kw: plt.Line2D([], [], **kw)
    handles = [L(color=COLORS["decay"], lw=2.0, label="Daughter that keeps its helicity (neutrino)"),
               L(color=COLORS["decay"], lw=2.0, ls=(0, (5, 2.5)), label="Daughter that flips it (antineutrino)"),
               L(color="0.35", lw=1.4, ls=(0, (1, 1.5)), label="Both daughters"),
               L(color=COLORS["s"], lw=2.0, label=r"$s$ channel alone"),
               L(color=COLORS["tu"], lw=2.0, label=r"$t$ and $u$ channels alone"),
               L(color=COLORS["pairs"], lw=2.0, label="Pair production alone")]
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, 0.995, x1 - x0, 0), mode="expand",
                     borderaxespad=0.0, ncol=3, fontsize=16, frameon=True, framealpha=1, edgecolor="black",
                     handlelength=2.4)
    leg.get_frame().set_linewidth(1.0)
    return fig
