"""provenance_lelmu.py -- where the L_e - L_mu limits of the earlier estimate (Agarwalla:2023sng) and ours part
(2026-11-22, his request: "separate [method from data], so we can explain. No need to rerun everything").

Inputs (set-1 data, now_icecube, pion source composition fixed, as in the earlier estimate):
  * the earlier estimate's published 95% C.L. curve (src/data/existing_bounds/agarwalla_2305.03675_emu.txt);
  * its method with our data and fit: dark_portal.trip = "detector" (planes le_lmu_det__b0..2, nine masses);
  * our transport without decays or relic collisions (planes le_lmu_nod__b0..2);
  * our full calculation (the paper's le_lmu_ul limits, pion composition).
The planes ran on Tycho in a copy of the nulri_set1_sm checkout (nulri_sep; configs analysis/configs/sep/ there);
their plane_m_Zd_eV_g_D.json files are copied to output/analysis/provenance_lelmu/.

  python tools/provenance_lelmu.py   -> prints the per-mass limits and writes output/figures/provenance_lelmu.pdf
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
DIR = os.path.join(REPO, "output", "analysis", "provenance_lelmu")
FULL = os.path.join(REPO, "output", "analysis", "limits_20261121", "nulri_set1_sm", "campaign_final_sm", "now_icecube",
                    "le_lmu_ul", "limits.json")
EARLIER = os.path.join(REPO, "src", "data", "existing_bounds", "agarwalla_2305.03675_emu.txt")
THRESHOLD = 3.841
UNITARITY_G_OVER_M = 100.0     # 1/eV: the planes' first 'not allowed' row (perturbative unitarity), at every mass


def lowest_excluded(path):
    """Per mass of a plane: (log10 M, log10 of the lowest coupling with Delta chi^2 >= THRESHOLD or None, log10 of
    the largest allowed coupling computed)."""
    d = json.load(open(path))
    X, Y = d["grids"]
    out = []
    for i, m in enumerate(X):
        row = d["dchi2"][i]
        exc = [np.log10(Y[j]) for j in range(len(Y)) if row[j] is not None and row[j] >= THRESHOLD]
        top = max(np.log10(Y[j]) for j in range(len(Y)) if row[j] is not None)
        out.append((np.log10(m), min(exc) if exc else None, top))
    return out


def table():
    earlier = np.loadtxt(EARLIER)
    full = {c["log10_m"]: c["lowest_excluded_log10_g"] for c in json.load(open(FULL))["columns"]}
    det = sum((lowest_excluded(f) for f in sorted(glob.glob(os.path.join(DIR, "le_lmu_det__b*.json")))), [])
    nod = {m: (g, top) for m, g, top in
           sum((lowest_excluded(f) for f in sorted(glob.glob(os.path.join(DIR, "le_lmu_nod__b*.json")))), [])}
    rows = []
    for m, g_det, _ in det:
        rows.append(dict(log10_M=m, earlier=float(np.interp(m, earlier[:, 0], earlier[:, 1])), detector_our_data=g_det,
                         no_decays=nod.get(m, (None, None))[0], no_decays_top=nod.get(m, (None, None))[1],
                         full=full.get(round(m, 1))))
    return rows


def figure(rows, path):
    from analysis import figure_style
    figure_style.use()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    e = np.loadtxt(EARLIER)
    ax.plot(e[:, 0], e[:, 1], color="0.35", ls="--", lw=1.5, label="Earlier estimate (its data)")
    m = [r["log10_M"] for r in rows]
    ax.plot(m, [r["detector_our_data"] for r in rows], "o-", color="#4477aa", ms=5, lw=1.2,
            label="Its method, our data")
    full = json.load(open(FULL))["columns"]
    ax.plot([c["log10_m"] for c in full], [c["lowest_excluded_log10_g"] for c in full], color="#9b2226", lw=2.0,
            label="Our calculation")
    for r in rows:
        if r["no_decays"] is not None:
            ax.plot(r["log10_M"], r["no_decays"], "^", color="#0a9396", ms=6)
        else:
            ax.annotate("", (r["log10_M"], r["no_decays_top"] + 0.6), (r["log10_M"], r["no_decays_top"]),
                        arrowprops=dict(arrowstyle="->", color="#0a9396", lw=1.2))
    ax.plot([], [], "^", color="#0a9396", label="No decays or collisions")
    xx = np.linspace(-27, -9, 50)
    ax.plot(xx, xx + np.log10(UNITARITY_G_OVER_M), color="0.6", lw=0.8, ls=":")
    ax.text(-15.2, -12.6, "Perturbative unitarity", color="0.45", fontsize=8, rotation=33)
    ax.set_xlim(-27, -9)
    ax.set_ylim(-32, -10)
    ax.set_xlabel(r"Mediator mass, $\log_{10}(M/{\rm eV})$")
    ax.set_ylabel(r"Coupling, $\log_{10} g'$")
    ax.legend(loc="upper left", fontsize=8, frameon=True, edgecolor="black", title=r"$L_e - L_\mu$, 95\% C.L., pion sources",
              title_fontsize=8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    rows = table()
    print(f"{'log10 M':>8} {'earlier':>8} {'its method, our data':>21} {'no decays':>10} {'full':>7}")
    for r in rows:
        nd = f"{r['no_decays']:.2f}" if r["no_decays"] is not None else f"> {r['no_decays_top']:.2f}"
        print(f"{r['log10_M']:8.1f} {r['earlier']:8.2f} {str(r['detector_our_data']):>21} {nd:>10} {str(r['full']):>7}")
    from analysis import figure_style, provenance_figure
    figure_style.use()
    fig = provenance_figure.figure()     # the polished figure (analysis/provenance_figure.py)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(REPO, "output", "figures", f"provenance_lelmu.{ext}"), bbox_inches="tight",
                    pad_inches=0.05, dpi=150)
