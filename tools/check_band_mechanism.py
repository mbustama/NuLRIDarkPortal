"""
check_band_mechanism.py -- why the fitted-composition ultralight planes keep a thin excluded band (campaign_ml,
2026-10-21). Expected: well above the band the new force shifts the flavor mix at Earth by the same amount at every
energy, which a different source composition reproduces (absorbed); on the band, where the force switches on, the
shift changes with energy and no composition reproduces it. Checked at m = 1e-22 eV in the future detectors' L_e-L_tau
plane: the flavor fractions at Earth against energy at couplings below, on and above the band, each with its fitted
parameters (stored fits), the one above also with the compositions of pion decay (the shift before absorption), and the
Asimov data (no new force). Writes output/figures/band_mechanism.png and output/analysis/band_mechanism.npz.
  python tools/check_band_mechanism.py
"""
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
from check_edge_islands import load  # noqa: E402

CFG = "analysis/configs/campaign_ml/proj_future/le_ltau_ul_fx.json"
M, CASES = -22, (("below the band", -28.0), ("on the band", -27.3), ("above the band", -26.5))


def stored(out_dir, xn, yn, m, g):
    """the stored fit (grid or refinement) nearest (log10 m, log10 g) at that mass"""
    s, x, y, chi, best = load(out_dir)
    recs = [dict(held={xn: float(x[i]), yn: float(y[j])}, chi2=r["chi2"], values=r["values"]) for (i, j), r in best.items()]
    for f in glob.glob(f"{out_dir}/refine_*.jsonl"):
        recs += [json.loads(line) for line in open(f)]
    recs = [r for r in recs if abs(np.log10(r["held"][xn]) - m) < 1e-6]
    return min(recs, key=lambda r: abs(np.log10(r["held"][yn]) - g))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.config import from_dict
    raw = json.load(open(os.path.join(REPO, CFG)))
    cfg = from_dict(raw, os.path.join(REPO, CFG))
    out_dir = os.path.join(REPO, raw["output_dir"], "asimov")
    s = json.load(open(glob.glob(f"{out_dir}/plane_*.json")[0]))
    xn, yn = s["names"]
    model = cfg.model(cache_size=8)
    rows = [("Asimov data (no new force)", cfg.asimov_values(), None)]
    for label, g in CASES:
        r = stored(out_dir, xn, yn, M, g)
        v = {**r["values"], **r["held"]}
        rows.append((f"{label}, g = 1e{np.log10(r['held'][yn]):.1f}, fitted sources "
                     f"({v.get('f_e_S', np.nan):.2f}, {v.get('f_e_S_gal', np.nan):.2f})", v, r["chi2"]))
    v = dict(rows[-1][1], f_e_S=1 / 3, f_e_S_gal=1 / 3)
    rows.append((f"above the band, pion-decay sources (1/3, 1/3)", v, None))
    out, fig, axes = {}, None, None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for k, (label, values, chi2) in enumerate(rows):
        E, F = model.flux(values)
        f = F / F.sum(-1, keepdims=True)
        out[f"E"], out[f"f{k}"] = E, f
        print(label, "" if chi2 is None else f"chi2 {chi2:.2f}", flush=True)
        for a, ax in enumerate(axes):
            ax.plot(E / 1e9, f[:, a], lw=2.2 if k == 0 else 1.5, ls="--" if k == len(rows) - 1 else "-",
                    color=["0.3", "#2a78d6", "#eb6834", "#1baf7a", "#1baf7a"][k], label=label if a == 0 else None)
    for a, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_xlim(1e4, 1e7)
        ax.set_xlabel("neutrino energy (GeV)")
        ax.set_title(["electron", "muon", "tau"][a] + " fraction at Earth")
        ax.grid(True, color="0.9")
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"Future detectors, L_e - L_tau, m = 1e{M} eV, source compositions fitted", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "output/figures/band_mechanism.png"), dpi=110)
    np.savez(os.path.join(REPO, "output/analysis/band_mechanism.npz"), labels=[r[0] for r in rows], **out)
    print("saved output/figures/band_mechanism.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
