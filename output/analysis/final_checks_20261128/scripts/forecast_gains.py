"""Forecast gains of Sec. V ("Future telescopes" and the last sentence of the MeV-preference subsection), from a limits
export (tools/export_limits.py layout), fitted source composition (the _fx planes, as Figs. 9-11).

Definition (that of 2026-11-22, scratchpad msg/limits_summary_fx.py + compare_fx.py, rebuilt here): for each of the 21
models (L_e-L_mu, L_e-L_tau, dark portal e/mu/tau, 16 secret-interaction cases), the lowest excluded log10 g per mass
column of the field the limit figures draw (ultralight and MeV planes joined by analysis.exclusion.joined, then
along_ratio; SI joined on a 0.125-decade coupling grid; thin bands out), for today's set 1 (now_icecube) and the eight
telescopes (proj_future_planA). Gain at a mass = today's lowest excluded log10 g minus the projected one, at the masses
where both have a limit. Per model: the median gain over all masses, below 1 keV (log10 M < 3) and above (>= 3), and the
largest gain above 1 keV. Preference: projected limit against log10_g_best at the mass of largest preference
(preference json of analysis/preference_figure.py).

Usage (repo root): python output/analysis/final_checks_20261128/scripts/forecast_gains.py LIMITS_DIR PREFERENCE_JSON OUT_JSON
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "src")]
from analysis import exclusion as ex  # noqa: E402

LIMITS, PREF, OUT = sys.argv[1:4]
WHERE = {"le_ul": ("nulri_set1_sm", "campaign_final_sm"), "dp_ul": ("nulri_set1_final", "campaign_final"),
         "mev": ("nulri_set1_ml", "campaign_ml"), "si_now": ("nulri_refit_set", "campaign_si"),
         "si_proj": ("nulri_forecast", "campaign_si")}
NOW, PROJ = "now_icecube", "proj_future_planA"


def plane(w, ds, name):
    return ex.limits_plane(LIMITS, WHERE[w], ds, name)


def lowest(ulw, mevw, uln, mevn, ds, dg):
    ul, mev = plane(ulw, ds, uln), plane(mevw, ds, mevn)
    if not mev:
        raise FileNotFoundError(f"{ds}/{mevn}")
    lx, G, F = ex.along_ratio(*ex.joined(ul, mev, dg=dg)) if ul else ex.field(mev)[:3]
    Fn = F.filled(np.nan) if np.ma.isMaskedArray(F) else F
    out = {}
    for k, x in enumerate(lx):
        e = np.flatnonzero(Fn[:, k] > 0)
        if len(e):
            out[round(float(x), 4)] = float(G[e[0]])
    return out


models = [(m, "le_ul", "mev", None) for m in ("le_lmu", "le_ltau")]
models += [(m, "dp_ul", "mev", None) for m in ("dp_e", "dp_mu", "dp_tau")]
models += [(f"si_{a}_{b}_{c}", "si_now", "si_now", 0.125) for a in ("scalar", "vector") for b in ("majorana", "dirac")
           for c in ("e", "mu", "tau", "universal")]
pref = json.load(open(PREF))[NOW]
res = {}
for m, ulw, mevw, dg in models:
    si = m.startswith("si")
    uln, mevn = f"{m}_ul_fx", (f"{m}_fx" if si else f"{m}_mev_fx")
    now = lowest(ulw, mevw, uln, mevn, NOW, dg)
    proj = lowest("si_proj" if si else ulw, "si_proj" if si else mevw, uln, mevn, PROJ, dg)
    xs = sorted(set(now) & set(proj))
    gain = {x: now[x] - proj[x] for x in xs}
    lo = [gain[x] for x in xs if x < 3]
    hi = [gain[x] for x in xs if x >= 3]
    xhi = max((x for x in xs if x >= 3), key=lambda x: gain[x])
    cols = pref[mevn]["columns"]
    xp = max(cols, key=lambda x: cols[x]["preference"])
    gp = cols[xp]["log10_g_best"]
    xpr = round(float(xp), 4)
    res[m] = {"masses": len(xs), "median": float(np.median(list(gain.values()))),
              "median_below_1keV": float(np.median(lo)) if lo else None,
              "median_above_1keV": float(np.median(hi)) if hi else None,
              "max_above_1keV": [xhi, gain[xhi]],
              "preferred": {"log10_m": xpr, "dchi2": cols[xp]["preference"], "log10_g_best": gp,
                            "projected": proj.get(xpr), "below_by": None if xpr not in proj else gp - proj[xpr]},
              "gain": {str(x): gain[x] for x in xs}}
    r = res[m]
    print(f"{m:30s} n={r['masses']:3d} med {r['median']:.2f}  <1keV {r['median_below_1keV'] or float('nan'):.2f}"
          f"  >=1keV {r['median_above_1keV']:.2f}  max>=1keV {gain[xhi]:.2f} @ {xhi}"
          f"  pref M=10^{xpr} dchi2 {cols[xp]['preference']:.2f} g_best {gp} proj {proj.get(xpr)} below {r['preferred']['below_by']}")
json.dump(res, open(OUT, "w"), indent=1)
