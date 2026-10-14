"""
absorption_relic_weights.py -- how the resonant absorption on relic neutrinos weighs the three relic mass states
(docs/dev/SCOPE_WIDENING_PLAN.md, step 3; found 2026-10-14).

The rates of src/dp_thermal.py follow the theory notes (Eqs. T7, T11, T12): an incident nu_alpha meets each relic
mass state i with the same flavor coupling (Sum G)_alpha = Sum_beta |C_alpha beta|^2, whatever the state's flavor
content. The s-channel nu_alpha + nubar_i -> Z' couples through the relic's own components, amplitude
(C U*)_alpha i, so the state's weight is |(C U*)_alpha i|^2; summed over the three states this is (C C^dag)_alpha
alpha = (Sum G)_alpha, i.e. one relic species' worth of targets, where the kernel counts three. The resonance width
(Sum G) M / 12 pi is the same in both.

Computed with the kernel itself (build_thermal_grid), one relic state at a time (eta_mass = e_i), at z = 0:
the dark portal's e-mu texture (C = g u u^dag, u = (0.5, 0.5, 0)) and a flavor-diagonal coupling (L_e - L_mu,
C = g diag(1, -1, 0)), at M = 3 MeV.

  python tools/absorption_relic_weights.py [--out output/figures/absorption_relic_weights.png]
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

import dark_portal_solver as dps  # noqa: E402
from dp_thermal import build_thermal_grid  # noqa: E402
from dp_config import DEFAULT_CONFIG  # noqa: E402

FLAVORS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
COLORS = ["#D55E00", "#0072B2", "#009E73"]


def rates(E, C, U, m_nu, T0, M, resolved):
    """Gamma_alpha(E) (N_E, 3) at z = 0 for the coupling matrix C (flavor basis): the kernel's weights
    (resolved False) or the mass-resolved ones |(C U*)_alpha i|^2 (True); both with the width of Sum |C|^2."""
    W = np.abs(C) ** 2
    sG_alpha = W.sum(axis=1)
    sG = float(W.sum())
    out = np.zeros((len(E), 3))
    for i in range(3):
        eta = np.zeros(3)
        eta[i] = 1.0
        w = np.abs(C @ np.conj(U[:, i])) ** 2 if resolved else sG_alpha
        out += build_thermal_grid(E, np.array([0.0]), sG, np.ascontiguousarray(w, dtype=np.float64), M, m_nu, T0,
                                  eta)[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "output", "figures", "absorption_relic_weights.png"))
    a = ap.parse_args(argv)
    cfg = DEFAULT_CONFIG
    sm = cfg["standard_mixing"]
    U = dps.construct_PMNS(np.radians(sm["th12_deg"]), np.radians(sm["th13_deg"]), np.radians(sm["th23_deg"]),
                           np.radians(sm["d_cp_deg"]))
    m_nu = dps.get_mass_eigenvalues(cfg["cosmology"]["m_lightest_eV"], sm["dm21_sq"], sm["dm31_sq"], "normal")
    T0 = cfg["cosmology"]["T_nu0_eV"]
    M = 3e6
    g = 1e-3
    u = np.array([0.5, 0.5, 0.0])
    cases = [("Dark portal, $e\\mu$ texture: $C = g\\,uu^\\dagger$", g * np.outer(u, u)),
             ("$L_e-L_\\mu$: $C = g\\,{\\rm diag}(1,-1,0)$", g * np.diag([1.0, -1.0, 0.0]))]
    E = np.geomspace(3e13, 1e15, 2500)
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    summary = {}
    for ax, (title, C) in zip(axes, cases):
        R0 = rates(E, C, U, m_nu, T0, M, False)
        R1 = rates(E, C, U, m_nu, T0, M, True)
        tot0, tot1 = np.trapz(R0, E, axis=0), np.trapz(R1, E, axis=0)
        sG_alpha = (np.abs(C) ** 2).sum(axis=1)
        w_res = np.abs(C @ np.conj(U)) ** 2                    # (alpha, i): each relic state's own weight
        summary[title] = {"kernel_over_resolved_integrated": [float(x) for x in tot0 / np.where(tot1 > 0, tot1, np.nan)],
                          "resolved_weight_fraction": (w_res / np.where(sG_alpha[:, None] > 0, sG_alpha[:, None], np.nan)).tolist()}
        flav = [k for k in range(3) if sG_alpha[k] > 0]
        x0 = 0.0
        ticks, labels = [], []
        for k in flav:
            frac = w_res[k] / sG_alpha[k]
            xs = x0 + np.arange(3) * 0.9
            ax.bar(xs, np.ones(3), width=0.8, color="none", edgecolor="#555555", lw=0.8, ls=(0, (3, 2)))
            ax.bar(xs, frac, width=0.8, color=COLORS[k], alpha=0.85)
            for j, x in enumerate(xs):
                ax.text(x, frac[j] + 0.03, f"{frac[j]:.2f}", ha="center", fontsize=7)
            ax.text(xs[1], 1.06, f"{FLAVORS[k]}: x{tot0[k] / tot1[k]:.1f} too much", ha="center", fontsize=7.5)
            ticks += list(xs)
            labels += [f"$m_{j + 1}$" for j in range(3)]
            x0 += 3.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("Relic mass state")
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.7, x0 - 3.6 + 2.5)
    axes[0].set_ylabel("Weight of the relic state in the\nabsorption of each flavor")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="none", edgecolor="#555555", linestyle=(0, (3, 2)),
                     label="now (theory notes): every relic state at the full coupling, sum 3"),
               Patch(facecolor="#888888", label="by the state's own flavor content $|(CU^*)_{\\alpha i}|^2$, sum 1")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle("Resonant absorption on relic neutrinos ($M = 3$ MeV): the weight of each relic mass state, "
                 "and the integrated excess", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
