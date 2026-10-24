"""The Galactic vacuum path (dp_galactic_path.vacuum_populations) against the per-pixel sweep (2026-10-24): the sky-
averaged Galactic flux at a few ultralight points, both paths, with their times. Run on Tycho; arguments: the points' indices (default all)."""
import copy
import json
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "src"), os.path.join(os.path.dirname(__file__), "..")]
os.environ.pop("NULRI_FLUX_CACHE", None)
import numpy as np  # noqa: E402
import dark_portal_solver as dps  # noqa: E402
import dp_galactic_path as gp  # noqa: E402
from analysis import config as config_mod  # noqa: E402

POINTS = [("dp_mu_ul", 1e-4, 1e-6), ("dp_mu_ul", 1e-6, 1e-6), ("dp_e_ul", 1e-5, 1e-7), ("le_lmu_ul", 1e-4, 1e-9),
          ("le_lmu_ul", 1e-4, 1e-6)]
used = []
SPLIT = [1]
for name in ("sweep_populations", "vacuum_populations"):
    f = getattr(gp, name)

    def wrap(*a, _f=f, _n=name, **k):
        t = time.time()
        r = _f(*a, n_split=SPLIT[0], **k)
        used.append((_n, time.time() - t))
        return r
    setattr(gp, name, wrap)

# (path, segment pieces): "sweep" the per-pixel sweep, "vac" the vacuum path where allowed, "force" the vacuum path
# everywhere (the potential's effect on the processes); the sweep refined to 16 pieces per segment is the reference
RUNS = [("sweep", 16), ("sweep", 4), ("sweep", 1), ("vac", 4), ("vac", 1), ("force", 1)]
ROT = dps.GALACTIC_VACUUM_ROTATION
sel = [int(a) for a in sys.argv[1:]] or range(len(POINTS))
for cfgname, x, y in [POINTS[k] for k in sel]:
    p = f"analysis/configs/campaign_ml/now_ic26/{cfgname}.json"
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    m = cfg.model()
    pl = raw["planes"][0]
    v = {q.name: q.start for q in cfg.parameters}
    v.update({pl["x"]: x, pl["y"]: y})
    E, pr = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    pr["show_progress"] = False
    E = np.asarray(E)
    ref = None
    pr["n_cores"] = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    for vac, split in RUNS:
        dps.GALACTIC_PATH_VACUUM = vac != "sweep"
        dps.GALACTIC_VACUUM_ROTATION = np.inf if vac == "force" else ROT
        SPLIT[0] = split
        used.clear()
        t = time.time()
        r = dps.galactic_allsky(E, pr)
        line = (f"{cfgname} m {x:g} g {y:g} {vac}, pieces {split}: {time.time() - t:.1f} s, "
                f"{[(n, round(s, 2)) for n, s in used]}")
        if ref is None:
            ref = r
        else:
            tot = ref.sum(axis=-1)
            rel = np.abs(r - ref).max(axis=-1) / np.where(tot > 0, tot, 1.0)
            line += f"; vs reference: max per energy {rel.max():.2e}, flux ratio {r.sum() / ref.sum():.8f}"
        print(line, flush=True)
    print(f"   reference flavor {np.round(ref.sum(axis=(0, 1)) / ref.sum(), 4)}", flush=True)
