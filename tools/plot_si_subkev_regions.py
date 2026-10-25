"""
plot_si_subkev_regions.py -- where the secret interactions' processes act below 1 keV (2026-10-25; notes, "Secret
interactions below 1 keV"): per model (texture mu), the decay's onset (depth 1 over the path at 10 TeV, z <= 1), the
region where the model is not modelled (vector: perturbative unitarity of its longitudinal mode; scalar: the relics'
background field, Mauricio's decision (a)) at the lightest masses 1e-6 and 0.1 eV, and, for orientation, the hard
scatterings' and the vector's conversions' scales carried down from 1 keV (the 2050 projection's limits there).
  python tools/plot_si_subkev_regions.py  ->  output/figures/si_subkev_regions.png
"""
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

import dp_si  # noqa: E402
import dp_si_light as L  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

U = construct_PMNS(*np.radians((33.75878818411903, 8.623062290036087, 43.280093616242404, 212.0)))
H0 = 67.4e3 / 3.0856775814913673e22 * 6.582119569e-16
T0 = 1.676389e-4
ARGS = (T0, np.ones(3), H0, 0.315, 0.685, 4.0, 7.537e-5)


def masses(ml):
    return np.array([ml, np.sqrt(ml * ml + 7.537e-5), np.sqrt(ml * ml + 2.511e-3)])


def decay_onset(med, maj, ml, lm):
    """log10 g where the decay depth at 10 TeV over z <= 1 reaches 1, per log10 M (nan where closed)."""
    zz = np.linspace(0, 1, 401)
    I = np.trapz(1.0 / ((1 + zz) ** 2 * H0 * np.sqrt(0.315 * (1 + zz) ** 3 + 0.685)), zz)
    m = masses(ml)
    out = []
    for x in lm:
        M = 10.0 ** x
        G = dp_si.flavor_couplings("mu", 1.0, None)
        Gam = L.rest_rates(G, U, m, M, med, maj)          # at g = 1: rate ~ g^2
        lab = float(np.max(Gam.sum(1) * m)) / 1e13
        out.append(-0.5 * np.log10(lab * I) if lab > 0 else np.nan)
    return np.array(out)


def field_line(maj, ml, lm, ly):
    """log10 g of the scalar's relic-field boundary per log10 M: the measure grows as (g / M)^2 at a fixed relic
    composition, so c <- c - log10(X) / 2 from c = 2 converges in a few steps (the composition settles with g)."""
    G1 = dp_si.flavor_couplings("mu", 1.0, None)
    m = masses(ml)
    out = []
    for x in lm:
        M = 10.0 ** x
        c = 2.0
        for _ in range(4):
            X = L.scalar_field_measure(G1 * M * 10.0 ** c, U, m, M, maj, *ARGS)
            if X <= 0:
                break
            c -= 0.5 * np.log10(X)
        out.append(x + c)
    return np.array(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis import figure_style
    figure_style.use()
    lm = np.linspace(-30, 3, 34)
    ly = np.array([-38.0, 0.0])
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True, sharey=True)
    for ax, (med, maj) in zip(axes.ravel(), (("scalar", True), ("scalar", False), ("vector", True), ("vector", False))):
        for ml, ls in ((1e-6, "-"), (0.1, "--")):
            ax.plot(lm, decay_onset(med, maj, ml, lm), color="C0", ls=ls,
                    label=f"decay onset, $m_{{\\rm l}}$ = {ml:g} eV")
            if med == "vector":
                c = -0.5 * np.log10(L.goldstone_measure(dp_si.flavor_couplings("mu", 1.0, None), U, masses(ml), 1.0,
                                                         maj))
                edge = lm + c
            else:
                edge = field_line(maj, ml, lm, ly)
            ax.plot(lm, edge, color="0.35", ls=ls, label=f"not modelled above, $m_{{\\rm l}}$ = {ml:g} eV")
            if ml == 1e-6:
                ax.fill_between(lm, np.clip(edge, -38, 0), 0.0, facecolor="none", edgecolor="0.6", hatch="////",
                                lw=0.0)
        if med == "scalar":
            ax.axhline(np.log10(6e-3), color="C3", lw=1.0, ls=":", label="hard scatterings ($g\\sim6\\times10^{-3}$)")
        else:
            ax.plot(lm[lm > -1.5], np.log10(1.4e-5 * np.sqrt(10.0 ** lm[lm > -1.5])), color="C2", lw=1.0, ls=":",
                    label="conversions ($g\\propto\\sqrt{M}$, from 1 keV)")
            ax.axhline(-3.0, color="C3", lw=1.0, ls=":", label="hard scatterings ($g\\sim10^{-3}$)")
        ax.set_title(f"{med}, {'Majorana' if maj else 'Dirac'}, $\\mu$ texture")
        ax.set_xlim(-30, 3)
        ax.set_ylim(-38, 0)
    for ax in axes[1]:
        ax.set_xlabel("$\\log_{10}(M/{\\rm eV})$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$\\log_{10} g$")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    axes[1, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path = os.path.join(REPO, "output", "figures", "si_subkev_regions.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    main()
