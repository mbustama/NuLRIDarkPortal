"""Which term makes the dark-portal phase depend on its redshift steps (2026-10-24): at one point, the extragalactic
band flux (1e13-3e15 eV) and the flavor at 100 TeV for n_z = 200, 800, 3200 with one configuration of the terms, or
dp_si at 1, 2, 4 substeps. Arguments: config m g case, case in full, nodecay, decayonly, noloss, nocyclecap, si.
Run on Tycho."""
import copy
import json
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "src"), os.path.join(os.path.dirname(__file__), "..")]
os.environ.pop("NULRI_FLUX_CACHE", None)
import numpy as np  # noqa: E402
import dark_portal_solver as dps  # noqa: E402
import dp_relic_loss  # noqa: E402
from analysis import config as config_mod  # noqa: E402

cfgname, x, y, case = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
p = f"analysis/configs/campaign_ml/now_ic26/{cfgname}.json"
raw = copy.deepcopy(json.load(open(p)))
raw.pop("exact_profile", None)
cfg = config_mod.from_dict(raw, p)
m = cfg.model()
pl = raw["planes"][0]
v = {q.name: q.start for q in cfg.parameters}
v.update({pl["x"]: x, pl["y"]: y})
base = m._run_cfg(m._overrides(v))
if case == "noloss":
    dp_relic_loss.loss_coefficient = lambda *a, **k: 0.0
if case == "nocyclecap":
    dps.CYCLE_MAX = 1 << 20
    dps.CYCLE_STEP = 0.05
src = [np.array([1.0, 2.0, 0.0]) / 3.0]
ref = None
runs = [dict(si=s) for s in (4, 2, 1)] if case == "si" else [dict(nz=n) for n in (3200, 800, 200)]
for r in runs:
    c = copy.deepcopy(base)
    if "nz" in r:
        c["grid"]["n_z_bins"] = r["nz"] + (0 if r["nz"] == 200 else 1)
    E, pr = dps.build_solver_params(c)
    pr["show_progress"] = False
    pr["n_cores"] = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    if case == "si":
        pr["cosmo_transport"] = "si"
        pr["si"] = dict(z_substeps=r["si"], E_extension=10.0)
    else:
        pr["cosmo_transport"] = "dark_portal"
    if case == "nodecay":
        pr["neutrino_decay"] = False
    if case == "decayonly":
        pr["relic_scattering"] = False
    E = np.asarray(E)
    t = time.time()
    F = np.asarray(dps.propagate_weighted(E, pr, sources=src)[0])
    tot = F.sum(1)
    band = (E > 1e13) & (E < 3e15)
    k = int(np.argmin(np.abs(E - 1e14)))
    line = f"{cfgname} m {x:g} g {y:g} {case} {r}: {time.time() - t:.1f} s; band mean {tot[band].mean():.4e}, flavor {np.round(F[k] / F[k].sum(), 4)}"
    if ref is None:
        ref = tot
    else:
        q = tot[band] / ref[band]
        line += f"; vs finest {q.min():.4f}-{q.max():.4f}"
    print(line, flush=True)
