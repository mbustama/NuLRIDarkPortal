"""
audit_transport.py -- the transport's absolute checks over every campaign family (2026-10-24, after the stiff-bin flux
loss): per plane (now_ic26 configs, pion source), on a grid of mediator masses and couplings, the flux with the routed
cosmology mode ('auto') and, below 1 keV, with each mode forced ('dark_portal', 'si'); the flux over the band
(1e13-3e15 eV) relative to the plane's no-effect flux (coupling 1e-40 of its range's bottom), the flavor at 100 TeV,
the mode used, finiteness, the time. NULRI_AUDIT_SRC (a directory put first on the path) runs another dp_si.py (the
buggy one, for the bug's footprint).

  plan <manifest.json> --tasks N
  run <manifest.json> <task> <out_dir> [--workers 32]
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [p for p in (os.environ.get("NULRI_AUDIT_SRC"),) if p] + [os.path.join(REPO, "src"), REPO]

import argparse
import copy
import glob
import json
import time
from multiprocessing import get_context

import numpy as np

CONFIGS = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26")
UL_LOGM = [-30, -27, -24, -21, -18, -15, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 2.9, 3.1, 3.5, 4]
MEV_LOGM = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9]
N_Y = 13
SOURCE = np.array([1.0, 2.0, 0.0]) / 3.0
_W = {}


def plan(manifest, n_tasks):
    units = []
    for p in sorted(glob.glob(os.path.join(CONFIGS, "*.json"))):
        if p.endswith("_fx.json"):
            continue
        d = json.load(open(p))
        if not (isinstance(d, dict) and "planes" in d and "solver" in d):
            continue
        pl = d["planes"][0]
        lo, hi = np.log10(pl["y_range"])
        ys = list(np.linspace(lo, hi, N_Y))
        if pl["y"] in ("g_D_trace", "g_D"):
            ys += [np.log10(3e-5), -4.0, np.log10(3e-4)]
        logm = MEV_LOGM if "_mev" in p else UL_LOGM
        rel = os.path.relpath(p, REPO)
        for lm in logm:
            for ly in sorted(set(round(float(v), 6) for v in ys)):
                modes = ["auto"] + (["dark_portal", "si"] if lm < 3 else [])
                for mode in modes:
                    units.append(dict(config=rel, x=pl["x"], y=pl["y"], logm=lm, logy=ly, mode=mode))
        for mode in ("dark_portal", "si"):        # the no-effect flux per transport
            units.append(dict(config=rel, x=pl["x"], y=pl["y"], logm=0.0, logy=float(lo) - 40.0, mode=mode,
                              baseline=True))
    rng = np.random.default_rng(7)
    order = rng.permutation(len(units))
    units = [units[i] for i in order]
    json.dump(dict(tasks=[units[k::n_tasks] for k in range(n_tasks)], units=len(units)), open(manifest, "w"))
    print(f"{len(units)} evaluations -> {manifest}")
    return 0


def _one(u):
    import dark_portal_solver as dps
    from analysis import config as config_mod
    os.environ.pop("NULRI_FLUX_CACHE", None)
    t0 = time.time()
    out = dict(u)
    try:
        if _W.get("path") != u["config"]:
            p = os.path.join(REPO, u["config"])
            raw = copy.deepcopy(json.load(open(p)))
            raw.pop("exact_profile", None)
            cfg = config_mod.from_dict(raw, p)
            _W.update(path=u["config"], cfg=cfg, model=cfg.model())
        cfg, m = _W["cfg"], _W["model"]
        v = {q.name: q.start for q in cfg.parameters}
        v.update({u["x"]: 10.0 ** u["logm"], u["y"]: 10.0 ** u["logy"]})
        E, pr = dps.build_solver_params(m._run_cfg(m._overrides(v)))
        pr["show_progress"] = False
        if u["mode"] != "auto":
            pr["cosmo_transport"] = u["mode"]
        E = np.asarray(E)
        out["mode_used"] = dps._cosmology_mode(pr, E)
        F = np.asarray(dps.propagate_weighted(E, pr, sources=[SOURCE])[0])
        tot = F.sum(axis=1)
        band = (E > 1e13) & (E < 3e15)
        k = int(np.argmin(np.abs(E - 1e14)))
        out.update(finite=bool(np.all(np.isfinite(F))), band=[float(x) for x in tot[band]],
                   flavor=[float(x) for x in F[k] / F[k].sum()] if F[k].sum() > 0 else None, error=None)
    except Exception as e:
        out.update(error=repr(e)[:300])
    out["seconds"] = time.time() - t0
    return out


def run(manifest, task, out_dir, workers):
    share = json.load(open(manifest))["tasks"][task]
    os.makedirs(out_dir, exist_ok=True)
    print(json.dumps(_one(share[0]))[:200], flush=True)      # compiles once before the pool
    with open(os.path.join(out_dir, f"audit.{task}.jsonl"), "a") as fh, \
            get_context("spawn").Pool(min(workers, len(share))) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, share, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(share)}", flush=True)
    print("done", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("manifest")
    a.add_argument("--tasks", type=int, required=True)
    a = sub.add_parser("run")
    a.add_argument("manifest")
    a.add_argument("task", type=int)
    a.add_argument("out_dir")
    a.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return plan(a.manifest, a.tasks)
    return run(a.manifest, a.task, a.out_dir, a.workers)


if __name__ == "__main__":
    sys.exit(main())
