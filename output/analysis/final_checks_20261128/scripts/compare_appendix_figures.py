"""Old against recomputed data of the two appendix figures (2026-11-28): the trip stop by stop
(output/analysis/method_figures.npz, Fig. trip) and the Dirac relics' helicity (output/analysis/dirac_polarization.npz,
Fig. helicity). Prints, per panel, the largest change of each plotted quantity on the common energy range.

  python compare_appendix_figures.py OLD_METHOD NEW_METHOD OLD_DIRAC NEW_DIRAC
"""
import sys

import numpy as np

om, nm, od, nd = (np.load(p) for p in sys.argv[1:5])
STOPS = ("universe", "galaxy", "earth", "detector")


def on(E_from, y, E_to):
    return np.interp(np.log10(E_to), np.log10(E_from), y, left=np.nan, right=np.nan)


print("== Trip stop by stop: total flux / flux on arrival at the Milky Way, and flavor fractions")
for case in ("le_lmu_20", "le_lmu_26"):
    Eo, En = om[f"{case}/E"], nm[f"{case}/E"]
    for s in STOPS:
        fo, fn = om[f"{case}/{s}"], nm[f"{case}/{s}"]
        to, tn = fo.sum(1) / om[f"{case}/universe"].sum(1), fn.sum(1) / nm[f"{case}/universe"].sum(1)
        d_tot = np.nanmax(np.abs(on(Eo, to, En) - tn))
        d_fr = max(np.nanmax(np.abs(on(Eo, fo[:, k] / fo.sum(1), En) - fn[:, k] / fn.sum(1))) for k in range(3))
        print(f"  {case:10s} {s:9s} max |d(total ratio)| {d_tot:.2e}   max |d(flavor fraction)| {d_fr:.2e}")
    # what the text says: which stop changes the flavor composition
    for s0, s1 in zip(STOPS[:-1], STOPS[1:]):
        f0, f1 = nm[f"{case}/{s0}"], nm[f"{case}/{s1}"]
        ch = max(np.max(np.abs(f1[:, k] / f1.sum(1) - f0[:, k] / f0.sum(1))) for k in range(3))
        dr = np.max(np.abs(f1.sum(1) / f0.sum(1) - 1))
        print(f"    new: {s0} -> {s1}: largest change of a flavor fraction {ch:.3f}, of the total flux {dr:.3f}")

print("== Dirac helicity: flux / free-streaming flux")
keys = sorted({k.rsplit("_", 1)[0] for k in nd.files})
for base in keys:
    Eo, Fo0 = od[f"{base}_free"]
    En, Fn0 = nd[f"{base}_free"]
    out = []
    for mode in ("rest", "unpolarized", "polarized"):
        ro = od[f"{base}_{mode}"][1] / Fo0
        rn = nd[f"{base}_{mode}"][1] / Fn0
        d = np.abs(on(Eo, ro, En) - rn)
        i = int(np.nanargmax(d))
        out.append(f"{mode} {np.nanmax(d):.3f} (at {En[i] / 1e12:.0f} TeV)")
    ip = nd[f"{base}_polarized"][1] / Fn0
    print(f"  {base:22s} max |d ratio|: " + "; ".join(out) + f"; new polarized min {np.nanmin(ip):.3f}")
