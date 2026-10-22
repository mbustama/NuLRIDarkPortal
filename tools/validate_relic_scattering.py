"""
validate_relic_scattering.py -- which stored solutions the relic-scattering code (2026-10-22: the soft scatterings
in the dark-portal phase, dp_si's transport below 1 keV, the 'combined' cosmology, the two accuracy fixes) changes,
and by how much: step 2 of the campaign's finishing plan, which decides what is rerun.

  flags <out.jsonl> --data-root R [--workers N]
      every stored solution of every campaign_ml plane (the summaries' grid points and the refine_*.jsonl records,
      read from R's output/) classified by what the new code does there: the cosmology's transport (old: dp_si at
      1 keV and above where the potential is negligible, else the dark-portal phase; new: _cosmology_mode) and
      whether the soft scatterings act (_soft_scattering_rates). A solution with the same transport and no
      scatterings gets the same flux as before, bit for bit, except through dp_si's thermal kernels (the two accuracy
      fixes, ~1e-4).
  sample <flags.jsonl> <manifest.json> --per-plane N --tasks T
      the chi^2 comparison's points, per plane up to N flagged and N unflagged through dp_si (seeded).
  eval <manifest.json> <task> --data-root R --out <prefix> [--workers N]
      chi^2 at those stored solutions with the code this file runs from (run it from the old code's checkout and
      from the new), each in its own objective with the event nuisances' profile started from the priors' centres
      (the warm start's history out of the comparison); <prefix>.<task>.jsonl.
  report <flags.jsonl> <old prefix> <new prefix>
      per family: the flagged share, and new - old where both were evaluated.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
CONFIGS = os.path.join(REPO, "analysis", "configs", "campaign_ml")
SEED = 20261022
_W = {}


def _planes():
    """The plane configurations (inventories beside them have no solver)."""
    out = []
    for c in sorted(glob.glob(os.path.join(CONFIGS, "*", "*.json"))):
        raw = json.load(open(c))
        if isinstance(raw, dict) and "planes" in raw and "solver" in raw:
            out.append(c)
    return out


def _out_dir(cfgp, data_root):
    from analysis.config import from_dict
    from analysis.run import asimov_only
    cfg = from_dict(json.load(open(cfgp)), cfgp)
    d = cfg.output_path("asimov" if asimov_only(cfg) else "data")
    return os.path.join(data_root, os.path.relpath(d, REPO)), cfg


def solutions(cfgp, data_root):
    """[(source, held, values)] of a plane: its summary's grid points and every refine_*.jsonl record."""
    from check_edge_islands import load
    out_dir, _ = _out_dir(cfgp, data_root)
    if not glob.glob(os.path.join(out_dir, "plane_*.json")):
        return []
    s, x, y, chi, best = load(out_dir)
    xn, yn = s["names"]
    out = [("grid", {xn: float(x[i]), yn: float(y[j])}, b["values"]) for (i, j), b in best.items() if b.get("values")]
    for f in sorted(glob.glob(os.path.join(out_dir, "refine_*.jsonl"))):
        if os.path.basename(f).startswith("refine_recheck"):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("values"):
                out.append((os.path.basename(f), r["held"], r["values"]))
    return out


def _params(cfgp, held, values):
    if _W.get("path") != cfgp:
        from analysis.config import from_dict
        cfg = from_dict(json.load(open(cfgp)), cfgp)
        _W.clear()
        _W.update(path=cfgp, model=cfg.model())
    import dark_portal_solver as dps
    m = _W["model"]
    v = dict(values)
    v.update(held)
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    return np.asarray(E), params


def _old_mode(params, E):
    """The routing before 2026-10-22: dp_si at 1 keV and above where the potential is negligible."""
    import dark_portal_solver as dps
    if float(params["m_Zd_eV"]) < dps.SI_COSMOLOGY_MIN_M_EV:
        return "dark_portal"
    return dps._cosmology_mode(params, E)


def _flag(task):
    cfgp, src, held, values = task
    import dark_portal_solver as dps
    import dp_absorption
    from dp_thermal import compute_gamma_grid_inputs
    try:
        E, p = _params(cfgp, held, values)
        if p.get("si_enabled", False):
            return dict(config=cfgp, source=src, held=held, skip="secret interactions")
        new, old = dps._cosmology_mode(p, E), _old_mode(p, E)
        jumps = False
        if new != "si" and p.get("absorption_model", "lagrangian") == "lagrangian":
            U = dps.construct_PMNS(p["th12"], p["th13"], p["th23"], p["d_cp"])
            u = np.array([p["u_e"], p["u_mu"], p["u_tau"]], dtype=np.complex128)
            ab = dps._absorption_setup(p, U, u)
            eta_mass = compute_gamma_grid_inputs(U, u, p["g_D"], p["epsilon_prime"], p["eta_bg"])[2]
            Gj = (ab.G if p.get("G_coupling") is not None
                  else ab.G - dp_absorption.kappa_z_mixing(p["epsilon_prime"]) * np.eye(3, dtype=np.complex128))
            z = np.linspace(p["z_max"], 0.0, int(p["n_z_bins"]))
            jumps = dps._soft_scattering_rates(ab, U, p["m_Zd_eV"], p["m_nu_bg_arr"], p["T_nu0_eV"], eta_mass, z,
                                               p["H0_eV"], p["Omega_m"], p["Omega_L"], G_jump=Gj) is not None
        thermal = bool(p.get("thermal_products", False)) and new == "si"
        return dict(config=os.path.relpath(cfgp, REPO), source=src, held=held, old=old, new=new, jumps=bool(jumps),
                    thermal_si=thermal, flagged=bool(new != old or jumps))
    except Exception as e:
        return dict(config=os.path.relpath(cfgp, REPO), source=src, held=held, error=repr(e)[:300])


def flags(out, data_root, workers):
    from refine_grid import Progress
    tasks = [(c, s, h, v) for c in _planes() for s, h, v in solutions(c, data_root)]
    print(f"{len(tasks)} stored solutions", flush=True)
    progress = Progress(len(tasks), "points")
    with get_context("spawn").Pool(workers) as pool, open(out, "w") as fh:
        for k, r in enumerate(pool.imap(_flag, tasks, chunksize=64)):
            fh.write(json.dumps(r) + "\n")
            progress(k + 1)
    return 0


def _family(cfg):
    ds, name = cfg.split("/")[-2], os.path.basename(cfg)[:-5]
    base = name[:-3] if name.endswith("_fx") else name
    if "_pot_" in base:
        fam = "pot " + base.split("_pot_")[1].split("_")[0]
    elif base.startswith("dp_") and base.endswith("_ul"):
        fam = "dpul"
    else:
        fam = base.split("_")[-1] + " " + ("dp" if base.startswith("dp_") else "L")
    return ds, fam


def sample(flags_path, manifest, per_plane, n_tasks):
    rng = np.random.default_rng(SEED)
    by = {}
    for line in open(flags_path):
        r = json.loads(line)
        if "error" in r or "skip" in r:
            continue
        key = "flagged" if r["flagged"] else ("thermal_si" if r["thermal_si"] else None)
        if key:
            by.setdefault((r["config"], key), []).append(r)
    pts = []
    for (cfg, key), rows in sorted(by.items()):
        take = rng.choice(len(rows), size=min(per_plane, len(rows)), replace=False)
        pts += [dict(config=cfg, source=rows[i]["source"], held=rows[i]["held"], kind=key) for i in sorted(take)]
    rng.shuffle(pts)
    tasks = [pts[k::n_tasks] for k in range(n_tasks)]
    json.dump(dict(tasks=tasks), open(manifest, "w"))
    print(f"{len(pts)} points in {n_tasks} tasks -> {manifest}")
    return 0


def _eval(task):
    cfgp, src, held, values, data_root = task
    try:
        from analysis.config import from_dict
        from analysis.run import asimov_only
        if _W.get("obj_path") != cfgp:
            raw = json.load(open(cfgp))
            cfg = from_dict(raw, cfgp)
            dc = None
            if asimov_only(cfg):
                out_dir, _ = _out_dir(cfgp, data_root)
                runs = json.load(open(os.path.join(out_dir, "manifest.json")))["runs"]
                dc = tuple(next(r["data_centre"] for r in reversed(runs) if r.get("data_centre")))
            _W.clear()
            _W.update(obj_path=cfgp, obj=cfg.objective(data_centre=dc))
        obj = _W["obj"]
        ev = getattr(obj, "events", None)
        for t in ([] if ev is None else getattr(ev, "terms", [ev])):
            if hasattr(t, "_z0"):
                t._z0 = np.zeros_like(t._z0)
        v = dict(values)
        v.update(held)
        return dict(source=src, held=held, config=os.path.relpath(cfgp, REPO), chi2=float(obj.chi2(v)), error=None)
    except Exception as e:
        return dict(source=src, held=held, config=os.path.relpath(cfgp, REPO), chi2=None, error=repr(e)[:300])


def evaluate(manifest, task, data_root, prefix, workers):
    from refine_grid import Progress
    os.environ.pop("NULRI_FLUX_CACHE", None)
    share = json.load(open(manifest))["tasks"][task]
    jobs = []
    for cfg in sorted({p["config"] for p in share}):
        cfgp = os.path.join(REPO, cfg)
        sols = {(s, json.dumps(h, sort_keys=True)): v for s, h, v in solutions(cfgp, data_root)}
        jobs += [(cfgp, p["source"], p["held"], sols[(p["source"], json.dumps(p["held"], sort_keys=True))], data_root)
                 for p in share if p["config"] == cfg]
    print(f"{len(jobs)} points", flush=True)
    progress = Progress(len(jobs), "points")
    with get_context("spawn").Pool(min(workers, max(1, len(jobs)))) as pool, open(f"{prefix}.{task}.jsonl", "a") as fh:
        for k, r in enumerate(pool.imap(_eval, jobs, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            progress(k + 1)
    return 0


def report(flags_path, old_prefix, new_prefix):
    fam = {}
    for line in open(flags_path):
        r = json.loads(line)
        if "error" in r or "skip" in r:
            continue
        f = fam.setdefault(_family(r["config"]), dict(n=0, flagged=0, mode=0, jumps=0))
        f["n"] += 1
        f["flagged"] += r["flagged"]
        f["mode"] += r["new"] != r["old"]
        f["jumps"] += r["jumps"]

    def recs(prefix):
        out = {}
        for p in glob.glob(prefix + ".*.jsonl"):
            for line in open(p):
                r = json.loads(line)
                if r.get("chi2") is not None:
                    out[(r["config"], r["source"], json.dumps(r["held"], sort_keys=True))] = r["chi2"]
        return out
    old, new = recs(old_prefix), recs(new_prefix)
    d = {}
    for k in old.keys() & new.keys():
        d.setdefault(_family(k[0]), []).append(new[k] - old[k])
    print(f"{'data set':12s} {'family':10s} {'points':>8s} {'flagged':>8s} {'transport':>9s} {'jumps':>7s} | "
          f"{'compared':>8s} {'|d|>0.05':>8s} {'max|d|':>8s}")
    for k in sorted(fam):
        f = fam[k]
        dd = np.abs(np.array(d.get(k, [])))
        print(f"{k[0]:12s} {k[1]:10s} {f['n']:8d} {f['flagged']:8d} {f['mode']:9d} {f['jumps']:7d} | {len(dd):8d} "
              f"{int((dd > 0.05).sum()) if len(dd) else 0:8d} {dd.max() if len(dd) else 0:8.3f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("flags")
    a.add_argument("out")
    a.add_argument("--data-root", default=REPO)
    a.add_argument("--workers", type=int, default=32)
    a = sub.add_parser("sample")
    a.add_argument("flags")
    a.add_argument("manifest")
    a.add_argument("--per-plane", type=int, default=30)
    a.add_argument("--tasks", type=int, default=4)
    a = sub.add_parser("eval")
    a.add_argument("manifest")
    a.add_argument("task", type=int)
    a.add_argument("--data-root", default=REPO)
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=32)
    a = sub.add_parser("report")
    a.add_argument("flags")
    a.add_argument("old")
    a.add_argument("new")
    a = ap.parse_args(argv)
    if a.cmd == "flags":
        return flags(a.out, a.data_root, a.workers)
    if a.cmd == "sample":
        return sample(a.flags, a.manifest, a.per_plane, a.tasks)
    if a.cmd == "eval":
        return evaluate(a.manifest, a.task, a.data_root, a.out, a.workers)
    return report(a.flags, a.old, a.new)


if __name__ == "__main__":
    sys.exit(main())
