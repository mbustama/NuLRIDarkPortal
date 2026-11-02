"""
validate_goldstone.py -- the vector's longitudinal-mode scatterings on the lepton-number planes (2026-10-25; notes,
"Secret interactions below 1 keV"): (1) bit for bit against the solver before them where they do not act (L_e-L_mu
below the gate, the dark portal's ultralight plane, an SI point above 1 keV); (2) their effect and the factor's
exactness bound at le_lmu (M, g) = (1e-12, 1e-13) and (1e-12, 3e-14), m_lightest 1e-6 and 0.1 eV.
  python tools/validate_goldstone.py   (tools/_dps_before_goldstone.py: the solver before them, a scratch copy)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

import dark_portal_solver as dps  # noqa: E402
from analysis.config import load  # noqa: E402

spec = importlib.util.spec_from_file_location("dps_old", os.path.join(REPO, "tools", "_dps_before_goldstone.py"))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
CF = os.path.join(REPO, "analysis", "configs")


def sky(mod, cfg_path, over):
    cfg = load(cfg_path)
    model = cfg.model()
    v = {q.name: q.start for q in cfg.parameters}
    v.update(over)
    E, pr = mod.build_solver_params(model._run_cfg(model._overrides(v)))
    t0 = time.time()
    out = mod.propagate_sky_basis(E, pr, sources=((1.0, 0, 0), (0, 1.0, 0)),
                                  galactic_compositions=((1.0, 0, 0), (0, 1.0, 0)))
    return E, out, time.time() - t0


res = {"bit_for_bit": {}, "effect": []}
for name, path, over in (
        ("le_lmu below the gate", f"{CF}/campaign_ml/now_ic26/le_lmu_ul.json", dict(m_Zd_eV=1e-12, g_D=1e-16)),
        ("dark portal ul", f"{CF}/campaign_ml/now_ic26/dp_mu_ul.json", dict(m_Zd_eV=1e-12, g_D_trace=1e-13)),
        ("SI vector 10 keV", f"{CF}/campaign_si/now_ic26/si_vector_majorana_mu.json", dict(m_phi_eV=1e4, g_phi=1e-3))):
    _, a, _ = sky(dps, path, over)
    _, b, _ = sky(old, path, over)
    keys = [k for k in a if isinstance(a[k], np.ndarray)]
    same = all(np.array_equal(a[k], b[k]) for k in keys)
    res["bit_for_bit"][name] = dict(identical=bool(same), arrays=keys, goldstone_bound=a["meta"]["zeta"].get("goldstone"))
    print(name, res["bit_for_bit"][name], flush=True)

for ml in (1e-6, 0.1):
    for g in (1e-13, 3e-14):
        over = dict(m_Zd_eV=1e-12, g_D=g, m_lightest_eV=ml)
        E, a, ta = sky(dps, f"{CF}/campaign_ml/now_ic26/le_lmu_ul.json", over)
        _, b, tb = sky(old, f"{CF}/campaign_ml/now_ic26/le_lmu_ul.json", over)
        r = (a["bands_nu"][1] + a["bands_nubar"][1]).sum(axis=(0, 2)) / (b["bands_nu"][1] + b["bands_nubar"][1]).sum(axis=(0, 2))
        fa = (a["bands_nu"][1] + a["bands_nubar"][1]).sum(axis=0)
        fb = (b["bands_nu"][1] + b["bands_nubar"][1]).sum(axis=0)
        idx = [int(np.searchsorted(E, x)) for x in (1e13, 1e14, 1e15)]
        row = dict(m_lightest=ml, g=g, g_over_M=g / 1e-12, flux_ratio_1e13_1e14_1e15=[float(r[i]) for i in idx],
                   flavor_shift_max=float(np.max(np.abs(fa / fa.sum(1, keepdims=True) - fb / fb.sum(1, keepdims=True)))),
                   bound=a["meta"]["zeta"]["goldstone"], seconds_new=ta, seconds_old=tb)
        res["effect"].append(row)
        print(row, flush=True)
out = os.path.join(REPO, "output", "analysis", "goldstone_validation.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
print("wrote", out)
