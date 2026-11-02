"""
plot_matter_sources.py -- where the potential at IceCube comes from, for each matter charge of the benchmark
models (docs/dev/SCOPE_WIDENING_PLAN.md, step 1).

For a mediator of mass m the potential is Coef n_eff, with n_eff the Yukawa-smoothed charge density at the
detector; this plots n_eff per body versus m, for neutrons (the dark portal, L_mu - L_tau), electrons
(L_e - L_mu, L_e - L_tau), nucleons (B - 3L_mu, B - 3L_tau, B - 3/2 (L_mu + L_tau)) and 2e - n (B - 3L_e), with
the solver's own functions:
  Earth      the PREM profile's Yukawa integral at IceCube's depth (dp_spatial.earth_yukawa_table with the
             charge's charge per gram, yukawa_profile_density);
  Moon, Sun  point sources, N m^2 e^{-m d} / (4 pi d) (dp_core counts, dp_matter.point_sources);
  Galaxy     dp_transit.galactic_effective_density without the cosmic floor, per species;
  Cosmos     the cosmic mean's share of the same integral (the floor where the Galaxy is thinner, out to the
             Hubble radius): with the floor minus without.

  python tools/plot_matter_sources.py [--out output/figures/matter_charge_sources.png]
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

import dp_core as C  # noqa: E402
import dp_matter as DM  # noqa: E402
import dp_spatial as SP  # noqa: E402
import dp_transit as T  # noqa: E402
from dp_config import DEFAULT_CONFIG  # noqa: E402

CHARGES = [
    ("Neutrons", "dark portal, $L_\\mu-L_\\tau$", (1.0, 0.0, 0.0)),
    ("Electrons", "$L_e-L_\\mu$, $L_e-L_\\tau$", (0.0, 0.0, 1.0)),
    ("Nucleons", "$B-3L_\\mu$, $B-3L_\\tau$, $B-\\frac{3}{2}(L_\\mu+L_\\tau)$", (1.0, 1.0, 0.0)),
    ("$2e-n$", "$B-3L_e$ (sign folded out)", (1.0, 1.0, -3.0)),
]
BODIES = [("Cosmos", "#CC79A7"), ("Milky Way", "#E69F00"), ("Sun", "#D55E00"), ("Moon", "#999999"),
          ("Earth", "#0072B2")]
DEPTH_KM = 1.5
EV_PER_INV_M = 1.0 / C.m_to_eV_inv              # m = 1 / lambda: 1 m^-1 in eV
RANGES = [("Gpc", 3.0857e25), ("Mpc", 3.0857e22), ("kpc", 3.0857e19), ("pc", 3.0857e16), ("AU", 1.496e11),
          ("$R_\\oplus$", 6.371e6), ("km", 1e3)]


def sources(charge, m, n_n0, n_e0, R_H):
    """n_eff (cm^-3) at IceCube per body, for a normalized charge (None: neutrons) at mediator mass m (eV)."""
    q = charge if charge is not None else DM.MatterCharge(1.0, 0.0, 1.0, n_e0)
    gal_floor = DM.galactic_effective_density(q, m, n_n0, R_H)
    gal = DM.galactic_effective_density(q, m, 0.0, R_H, n_e0_eV3=0.0)
    N_sun, N_moon, _ = DM.point_sources(q)
    d_s, d_m = C.d_sun_eV_inv, C.d_moon_eV_inv
    sun = N_sun * m * m / (4 * np.pi * d_s) * np.exp(-m * d_s)
    moon = N_moon * m * m / (4 * np.pi * d_m) * np.exp(-m * d_m)
    yk = SP.earth_yukawa_table(m, cpg=DM.earth_cpg(q))
    earth = SP.yukawa_profile_density(6371.0 - DEPTH_KM, yk)
    conv = 1.0 / C.cm_inv3_to_eV3
    return dict(Cosmos=(gal_floor - gal) * conv, **{"Milky Way": gal * conv}, Sun=sun * conv, Moon=moon * conv,
                Earth=earth)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "output", "figures", "matter_charge_sources.png"))
    a = ap.parse_args(argv)
    cfg = DEFAULT_CONFIG
    n_n0 = cfg["cosmology"]["n_n0_cm3"] * C.cm_inv3_to_eV3
    n_e0 = cfg["cosmology"]["n_e0_cm3"] * C.cm_inv3_to_eV3
    R_H = 1.0 / (cfg["cosmology"]["H0_km_s_Mpc"] * C.km_per_s / C.Mpc_to_eV_inv)
    m = np.geomspace(1e-35, 1e-8, 271)
    curves = {}
    for name, _, (qn, qp, qe) in CHARGES:
        charge = DM.normalize(qn, qp, qe, n_n0, n_e0)
        rows = [sources(charge, x, n_n0, n_e0, R_H) for x in m]
        curves[name] = {b: np.array([r[b] for r in rows]) for b, _ in BODIES}
        curves[name]["Total"] = sum(curves[name][b] for b, _ in BODIES)

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "mathtext.fontset": "dejavusans"})
    fig, (ax, ar) = plt.subplots(2, 1, figsize=(6.6, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1.0]))
    # top: the neutron charge, body by body; the dominant body shaded on both panels
    c = curves["Neutrons"]
    dominant = np.array([max(BODIES, key=lambda b: c[b[0]][k])[0] for k in range(len(m))])
    edges = np.sqrt(m[1:] * m[:-1])
    lo = np.concatenate([[m[0]], edges])
    hi = np.concatenate([edges, [m[-1]]])
    colour = dict(BODIES)
    for k in range(len(m)):
        for a_ in (ax, ar):
            a_.axvspan(lo[k], hi[k], color=colour[dominant[k]], alpha=0.10, lw=0)
    for b, col in BODIES:
        ax.loglog(m, np.where(c[b] > 0, c[b], np.nan), color=col, lw=1.4, label=b)
    ax.loglog(m, c["Total"], color="black", lw=0.8, ls=(0, (4, 2)), label="Total")
    ax.set_ylim(1e-26, 1e27)
    ax.set_xlim(m[0], m[-1])
    ax.set_ylabel("Neutrons: Yukawa-smoothed\ndensity at IceCube [cm$^{-3}$]")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=3)
    ax.grid(True, which="major", lw=0.3, alpha=0.35)
    top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    top.set_xticks([EV_PER_INV_M / L for _, L in RANGES])
    top.set_xticklabels([n for n, _ in RANGES], fontsize=7.5)
    top.set_xlabel("Range $1/m$", fontsize=8)
    # bottom: each charge's total relative to the neutrons'
    styles = {"Electrons": ("#0072B2", "-"), "Nucleons": ("#009E73", "-"), "$2e-n$": ("#D55E00", "-")}
    for name, models, _ in CHARGES[1:]:
        col, ls = styles[name]
        ar.semilogx(m, curves[name]["Total"] / c["Total"], color=col, lw=1.5, ls=ls, label=f"{name}: {models}")
    ar.axhline(1.0, color="#555555", lw=0.6, ls=(0, (1, 2)))
    ar.set_ylim(0, 15)
    ar.set_ylabel("Total relative to\nthe neutron charge")
    ar.set_xlabel("Mediator mass $m$ [eV]")
    ar.legend(loc="upper right", fontsize=7, frameon=False)
    ar.grid(True, which="major", lw=0.3, alpha=0.35)
    fig.suptitle("The potential at IceCube by matter charge: hydrogen-rich distant matter weighs 4-7 times more",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200)
    # the numbers behind the figure: the electron/neutron ratio of each body's share and of the total
    summary = {}
    for x in (1e-33, 1e-28, 1e-24, 1e-18, 1e-14, 1e-10):
        k = int(np.argmin(np.abs(np.log(m / x))))
        summary[f"{m[k]:.1e}"] = {name: {b: float(curves[name][b][k]) for b in [b for b, _ in BODIES] + ["Total"]}
                                  for name, _, _ in CHARGES}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {a.out}")
    for key, row in summary.items():
        n, e = row["Neutrons"], row["Electrons"]
        print(f"m = {key} eV: total e/n = {e['Total'] / n['Total']:.2f}; dominant (n) "
              f"{max(BODIES, key=lambda b: n[b[0]])[0]}, (e) {max(BODIES, key=lambda b: e[b[0]])[0]}")


if __name__ == "__main__":
    main()
