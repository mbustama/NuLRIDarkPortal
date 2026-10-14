"""
screening_thresholds.py -- how strongly the relic neutrinos screen the dark portal's mediator, and which couplings
switch off which body (docs/dev/PHYSICS_AUDIT.md, item 1j; the eps' profile of the potential planes).

(a) The screening (Debye) mass today versus the coupling, for the Z mixing's piece (per eps') and the e-mu texture
    (per g_D |U|^2), with today's slow Majorana relics (dp_absorption.debye_mass_sq) and with the theory notes'
    relativistic Eq. (T23); a body's potential is cut off when the screening mass exceeds 1 / (its distance).
(b) At m_Zd = 1e-20 eV (a range of 130 AU), where the Sun's potential sets the bound, the plane of the two
    couplings: the potential fixes only their product P = g_D eps' |U|^2 (the solid line: the pre-audit 95% bound,
    e-mu); the shaded regions screen the Sun, and the Earth too; there only the unscreened cosmological potential
    is left (the dashed line: roughly the bound it gives alone, P V_Sun / V_cosmic); the MICROSCOPE limit on eps'
    (a vector coupled to B - L, which is the neutron number of neutral matter, as the Z mixing gives).

  python tools/screening_thresholds.py [--out output/figures/screening_thresholds.png]
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

import dp_absorption as A  # noqa: E402
import dp_core  # noqa: E402
import dark_portal_solver as dps  # noqa: E402

HBARC_EV_M = 1.97327e-7
T0 = 1.676389e-4
BODIES = [("Earth (its radius)", 6.371e6, "#8B5A2B"), ("Sun (1 AU)", 1.496e11, "#E69F00"),
          ("Galaxy (8 kpc)", 8 * 3.0857e19, "#56B4E9")]
# MICROSCOPE (final), spin 1 coupled to B - L, m <~ 1e-14 eV: |g_{B-L}| < 1.1e-25 (Fayet 2018, 2025); the Z mixing
# couples the mediator to neutrons with eps' g / (2 c_W) (the theory notes' convention), and B - L = N for neutral matter
G_BL_MAX = 1.1e-25
P_BOUND_1EM20 = 2.02e-55  # output/analysis/pion_2050_case_5_potential_emu: 95% upper bound on g_D eps' |U|^2 at 1e-20 eV


def per_unit_coupling():
    """Screening mass per unit coupling today (eV): the Z mixing's kappa = 1, and the e-mu texture's g_D |U|^2 = 1."""
    U = dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1), np.radians(132.0))
    m = dps.get_mass_eigenvalues(0.01, 7.41e-5, 2.511e-3, "normal")
    out = {}
    for key, G in (("kappa", np.eye(3, dtype=complex)), ("emu", np.outer([1, 1, 0], [1, 1, 0]).astype(complex))):
        d2 = np.abs(np.diag(U.conj().T @ G @ U)) ** 2
        out[key] = {"majorana": float(np.sqrt(A.debye_mass_sq(d2, m, T0, np.ones(3)))),
                    "dirac": float(np.sqrt(A.debye_mass_sq(d2, m, T0, np.ones(3), majorana=False)))}
    # the theory notes' Eq. (T23): relativistic, flavor basis, g_D^2 |u_alpha|^4 (e-mu, |u|^2 = 1 per entry)
    out["emu"]["theory_notes_T23"] = float(np.sqrt(dp_core.calc_debye_mass_sq(1.0, np.array([1.0, 1.0, 0.0]), T0,
                                                                                np.ones(3))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "output", "figures", "screening_thresholds.png"))
    a = ap.parse_args(argv)
    pu = per_unit_coupling()
    k_per_eps = A.kappa_z_mixing(1.0)
    eps_max = G_BL_MAX / (0.65 / (2 * 0.88))
    thr = {name: {"inverse_distance_eV": HBARC_EV_M / R,
                  "g_D_U2": HBARC_EV_M / R / pu["emu"]["majorana"],
                  "eps_prime": HBARC_EV_M / R / (pu["kappa"]["majorana"] * k_per_eps)} for name, R, _ in BODIES}

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.8, 3.5))
    g = np.logspace(-32, 0, 200)
    ax.loglog(g, pu["emu"]["majorana"] * g, color="#D55E00", lw=1.4,
              label=r"$e\mu$ texture, per $g_D|U|^2$ (slow Majorana relics)")
    ax.loglog(g, pu["kappa"]["majorana"] * k_per_eps * g, color="#0072B2", lw=1.4, ls=(0, (1, 1.5)),
              label=r"$Z$ mixing, per $\epsilon'$ (slow Majorana relics)")
    ax.loglog(g, pu["emu"]["theory_notes_T23"] * g, color="#999999", lw=1.0, ls=(0, (4, 2)),
              label=r"$e\mu$, the theory notes' Eq. (T23) (fast relics)")
    for name, R, col in BODIES:
        ax.axhline(HBARC_EV_M / R, color=col, lw=0.8)
        ax.text(1.5e-32, HBARC_EV_M / R * 1.8, name, fontsize=7, color=col)
    ax.set_xlim(1e-32, 1)
    ax.set_ylim(1e-36, 1e-3)
    ax.set_xlabel(r"Coupling ($g_D|U|^2$ or $\epsilon'$)")
    ax.set_ylabel("Screening mass today [eV]")
    ax.set_title("(a) Screening by relic neutrinos", fontsize=8.5, loc="left")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False)

    x = np.logspace(-45, 0, 400)
    y = np.logspace(-60, 0, 400)
    X, Y = np.meshgrid(x, y)
    mD = np.sqrt((pu["emu"]["majorana"] * X) ** 2 + (pu["kappa"]["majorana"] * k_per_eps * Y) ** 2)
    m_sun, m_earth = HBARC_EV_M / BODIES[1][1], HBARC_EV_M / BODIES[0][1]
    bx.contourf(X, Y, mD, levels=[m_sun, m_earth], colors=[BODIES[1][2]], alpha=0.25)
    bx.contourf(X, Y, mD, levels=[m_earth, 1.0], colors=[BODIES[0][2]], alpha=0.30)
    bx.contour(X, Y, mD, levels=[m_sun, m_earth], colors=[BODIES[1][2], BODIES[0][2]], linewidths=0.8)
    # the bound the cosmological potential gives alone: the same total potential from n_n0 / m^2 instead of the Sun's
    # N_n / (4 pi AU) (both with the same K)
    m_zd = 1e-20
    n_n0 = 2.515e-7 * 0.2458 / 2 * (HBARC_EV_M * 100) ** 3      # cm^-3 -> eV^3
    v_sun = dp_core.N_n_sun / (4 * np.pi * BODIES[1][1] / HBARC_EV_M)
    v_cos = n_n0 / m_zd ** 2
    P_cos = P_BOUND_1EM20 * v_sun / v_cos
    sel = x > thr["Sun (1 AU)"]["g_D_U2"] * 0.3
    bx.loglog(x, P_BOUND_1EM20 / x, color="k", lw=1.3,
              label=r"pre-audit 95% bound: $g_D\epsilon'|U|^2 = 2\times10^{-55}$")
    bx.loglog(x[sel], P_cos / x[sel], color="k", lw=1.0, ls=(0, (4, 2)),
              label="the cosmological potential alone (estimate)")
    bx.axhline(eps_max, color="#009E73", lw=1.0, ls=(0, (3, 1.5)),
               label=r"MICROSCOPE: $\epsilon' < 3\times10^{-25}$")
    bx.text(thr["Sun (1 AU)"]["g_D_U2"] * 1.5, 1e-33, "Sun\noff", fontsize=6.5, color="#9a6a00")
    bx.text(thr["Earth (its radius)"]["g_D_U2"] * 8, 1e-36, "Earth\noff too", fontsize=6.5, color=BODIES[0][2])
    bx.text(3e-44, 1e-35, "nothing screened", fontsize=6.5, color="#555555")
    bx.legend(loc="lower left", fontsize=6.3, frameon=False)
    bx.set_xlim(1e-45, 1)
    bx.set_ylim(1e-60, 1)
    bx.set_xlabel(r"$g_D|U|^2$ (the neutrinos' coupling)")
    bx.set_ylabel(r"$\epsilon'$ (the matter's coupling, via $Z$ mixing)")
    bx.set_title(r"(b) $m_{Z_D} = 10^{-20}$ eV: one potential, many splits", fontsize=8.5, loc="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    summary = {"screening_mass_per_unit_coupling_eV": pu, "kappa_per_eps_prime": k_per_eps,
               "thresholds": thr, "eps_prime_max_MICROSCOPE": eps_max, "P_bound_1e-20_eV": P_BOUND_1EM20,
               "P_bound_1e-20_eV_cosmic_alone_estimate": P_cos}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
