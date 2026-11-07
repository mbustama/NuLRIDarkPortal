"""
schematics.py -- the paper's schematic of the states in a potential (2026-11-07; paper/paper_planning.md, Step 5,
his go of the same day). levels(): the effective masses squared of the states that travel in an L_e - L_mu potential,
against the potential's strength, showing how a strong potential turns the mass states into nearly pure flavors; the
mixing is the campaign's (NuFIT 6.1, normal ordering). The trip itself is drawn by analysis/trip_illustration.py
(which replaced a to-scale version here, 2026-11-07). Called from notebooks/paper_figures.ipynb, in the repository's
style (analysis.figure_style.use() first).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _mixing():
    with open(os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26", "le_lmu_mev.json")) as f:
        return json.load(f)["solver"]["standard_mixing"]


def _pmns(c):
    t12, t13, t23, dcp = (np.radians(c[k]) for k in ("th12_deg", "th13_deg", "th23_deg", "d_cp_deg"))
    s12, c12, s13, c13, s23, c23 = np.sin(t12), np.cos(t12), np.sin(t13), np.cos(t13), np.sin(t23), np.cos(t23)
    e = np.exp(1j * dcp)
    return np.array([[c12 * c13, s12 * c13, s13 / e],
                     [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
                     [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]])


# ----------------------------------------------------------------------------------- the states in a potential

STATE_COLORS = ["#1b9e77", "#7570b3", "#c51b7d"]


def levels(m_lightest=1e-6, width_in=5.0, height_in=4.75):
    """The effective masses squared mu_a, the eigenvalues of 2 E H = U diag(m_i^2) U^dagger + sigma S_mu, against
    sigma = 2 E g'^2 Phi_e, for L_e - L_mu (S_mu = diag(1, -1, 0)), normal ordering."""
    c = _mixing()
    U = _pmns(c)
    msq = np.array([0.0, c["dm21_sq"], c["dm31_sq"]]) + m_lightest ** 2
    M2 = U @ np.diag(msq) @ U.conj().T
    S = np.diag([1.0, -1.0, 0.0])
    sig = np.logspace(-7, -1, 600)
    lam = np.array([np.linalg.eigvalsh(M2 + s * S) for s in sig])

    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.3, right=0.93, bottom=0.18, top=0.95)
    for a in range(3):
        ax.plot(sig, lam[:, a], color=STATE_COLORS[a], lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("asinh", linear_width=1e-4)
    ax.set_xlim(sig[0], sig[-1])
    ax.set_ylim(-0.13, 0.13)
    ax.set_xticks([1e-7, 1e-5, 1e-3, 1e-1])
    ax.set_yticks([-1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["$-10^{-1}$", "$-10^{-2}$", "$-10^{-3}$", "0", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"Potential, $2E g'^2 \Phi_e$ [eV$^2$]")
    ax.set_ylabel(r"Mass squared, $\mu_a$ [eV$^2$]")
    ax.text(1.6e-7, msq[2] * 1.4, r"$m_3^2$", color=STATE_COLORS[2], va="bottom")
    ax.text(1.6e-7, msq[1] * 1.4, r"$m_2^2$", color=STATE_COLORS[1], va="bottom")
    ax.text(1.6e-7, -1.5e-5, r"$m_1^2$", color=STATE_COLORS[0], va="top")
    ax.text(5e-3, 1.6e-2, r"$\approx \nu_e$", color=STATE_COLORS[2], ha="center", va="bottom")
    ax.text(2.5e-2, 8e-4, r"$\approx \nu_\tau$", color=STATE_COLORS[1], ha="center", va="top")
    ax.text(1.5e-2, -2.5e-3, r"$\approx \nu_\mu$", color=STATE_COLORS[0], ha="center", va="bottom")
    return fig
