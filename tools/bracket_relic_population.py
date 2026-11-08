"""
bracket_relic_population.py -- decision C (Mauricio, 2026-10-24; run 2026-11-07): how much the ultralight planes'
limits depend on the relic population where the relics' own decays are open. The model takes the decayed relics as
all nu_1, three times the number, a hotter Fermi-Dirac gas (dp_decay.relic_composition; docs item 22): the soft
scatterings' jumps scale with the relics' number, the mass-change energy loss with number x <1/E>. The bracket:
number x1 to x3 (dp_decay.BRACKET_DAUGHTERS 0 or 1) times <1/E> x1/2 to x2 (dp_relic_loss.BRACKET_INV_E), four
corners, against the model.

Points: on each plane of a finished run, per mediator mass, the two grid points that straddle the 95% edge (the first
excluded coupling and the allowed one below it), where the relics' composition is changed by their decays at that
point. Each is re-evaluated at its stored solution (no refit; NULRI_FLUX_CACHE ignored, fresh processes per
setting): the model's own setting first (the control: must give back the stored chi^2), then the four corners. A
corner's chi^2 at the stored solution is an upper bound on its profiled chi^2 (a refit can only go lower).

  python tools/bracket_relic_population.py plan <manifest.json> --root <run output root> --tasks N [--datasets a,b]
      [--campaign <config folder>]
  python tools/bracket_relic_population.py run <manifest.json> <task> <out dir> [--workers 32]
  python tools/bracket_relic_population.py report <manifest.json> <out dir>

<run output root>: a campaign's output folder (e.g. .../output/analysis/campaign_final) holding <data set>/<plane>/
b*/{data,asimov}/plane_*.jsonl (split runs) or <data set>/<plane>/{data,asimov}/plane_*.jsonl. The configs are read
from analysis/configs/<campaign name>/<data set>/<plane>[__b0].json of this checkout, which must be the code and
configs that wrote the run (the control checks it).
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

CORNERS = ((1.0, 1.0), (0.0, 0.5), (0.0, 2.0), (1.0, 0.5), (1.0, 2.0))   # (daughters, <1/E> factor); first = model
THRESHOLD = 3.84
_W = {}


def _records(plane_dir):
    """{held key: record} (lowest chi^2 per point) and the reference chi^2 (the fits' and the plane's lowest)."""
    best, ref, kind = {}, np.inf, "data"
    files = glob.glob(os.path.join(plane_dir, "b*", "*", "plane_*.jsonl")) + \
        glob.glob(os.path.join(plane_dir, "*", "plane_*.jsonl"))
    for f in files:
        kind = os.path.basename(os.path.dirname(f))
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "held" not in r or not r.get("values"):
                continue
            c = r.get("chi2")
            if c is None or c != c:
                continue
            k = json.dumps(r["held"], sort_keys=True)
            if k not in best or c < best[k]["chi2"]:
                best[k] = dict(held=r["held"], values=r["values"], chi2=float(c))
    for fj in glob.glob(os.path.join(plane_dir, "b*", "*", "fit.json")) + glob.glob(os.path.join(plane_dir, "*", "fit.json")):
        try:
            ref = min(ref, float(json.load(open(fj))["chi2"]))
        except (KeyError, ValueError, TypeError):
            pass
    if best:
        ref = min(ref, min(r["chi2"] for r in best.values()))
    return best, ref, kind


def _config_path(campaign, ds, plane):
    base = os.path.join(REPO, "analysis", "configs", campaign, ds)
    for name in (f"{plane}__b0.json", f"{plane}.json"):
        if os.path.exists(os.path.join(base, name)):
            return os.path.join(base, name)
    raise FileNotFoundError(f"no config for {ds}/{plane} under {base}")


def _relics_decay(model, values):
    """True where the relics' composition is changed by their decays at this point (the transport's own rule)."""
    import dark_portal_solver as dps
    import dp_absorption
    import dp_decay
    from dp_core import construct_PMNS
    E, p = dps.build_solver_params(model._run_cfg(model._overrides(values)))
    if not p.get("neutrino_decay", True):
        return False
    m = np.asarray(p["m_nu_bg_arr"], dtype=np.float64)
    U = construct_PMNS(p["th12"], p["th13"], p["th23"], p["d_cp"])
    if p.get("G_coupling") is not None:
        G = np.asarray(p["G_coupling"], dtype=np.complex128)
    else:
        u = np.array([p["u_e"], p["u_mu"], p["u_tau"]], dtype=np.complex128)
        G = dp_absorption.dark_portal_coupling(p["g_D"], u, 0.0)
    Gam, _ = dp_decay.rest_rates(G, U, m, float(p["m_Zd_eV"]))
    if not np.any(Gam > 0.0):
        return False
    eta0 = (np.abs(U) ** 2).T @ np.asarray(p["eta_bg"], dtype=np.float64)
    eta = dp_decay.relic_composition(Gam, m, eta0, float(p["T_nu0_eV"]), p["H0_eV"], p["Omega_m"], p["Omega_L"])
    return bool(np.max(np.abs(eta - eta0)) > 1e-3)


def plan(manifest, root, tasks, datasets=None, campaign=""):
    from analysis.config import from_dict
    campaign = campaign or os.path.basename(os.path.normpath(root))
    items = []
    for ds in sorted(os.listdir(root)):
        if datasets and ds not in datasets:
            continue
        for pd in sorted(glob.glob(os.path.join(root, ds, "*"))):
            plane = os.path.basename(pd)
            recs, ref, kind = _records(pd)
            if not recs:
                continue
            cfgp = _config_path(campaign, ds, plane)
            raw = json.load(open(cfgp))
            model = from_dict(raw, cfgp).model()
            xn, yn = raw["planes"][0]["x"], raw["planes"][0]["y"]
            cols = {}
            for r in recs.values():
                cols.setdefault(r["held"][xn], []).append(r)
            n0 = len(items)
            for M, col in sorted(cols.items()):
                col.sort(key=lambda r: r["held"][yn])
                exc = [i for i, r in enumerate(col) if r["chi2"] - ref >= THRESHOLD]
                if not exc:
                    continue
                i = exc[0]
                for r in ([col[i - 1]] if i > 0 else []) + [col[i]]:
                    v = dict(r["values"])
                    v.update(r["held"])
                    if _relics_decay(model, v):
                        items.append(dict(config=os.path.relpath(cfgp, REPO), ds=ds, plane=plane, kind=kind,
                                          run_dir=os.path.abspath(pd), held=r["held"], values=r["values"],
                                          stored=r["chi2"], ref=ref))
            print(f"{ds}/{plane}: {len(items) - n0} edge points where the relics decay", flush=True)
    shares = [items[k::tasks] for k in range(tasks)]
    json.dump(dict(tasks=shares, points=len(items), corners=CORNERS), open(manifest, "w"))
    print(f"{len(items)} points x {len(CORNERS)} settings, {tasks} tasks -> {manifest}")


def _init(daughters, inv_E):
    """A worker of one setting: the bracket's two factors; the objectives are built lazily, one per plane (a failure
    there is recorded per point, never in the initializer, where multiprocessing would respawn the worker forever)."""
    os.environ.pop("NULRI_FLUX_CACHE", None)
    import dp_decay
    import dp_relic_loss
    dp_decay.BRACKET_DAUGHTERS = float(daughters)
    dp_relic_loss.BRACKET_INV_E = float(inv_E)
    _W.clear()


def _one(task):
    cfg_rel, dc, held, values = task
    try:
        key = (cfg_rel, None if dc is None else tuple(dc))
        if key not in _W:
            # one plane's objective per worker at a time: each holds its model's caches and the event
            # likelihoods' Monte Carlo, and a worker that kept them all ran its node out of memory (2026-11-08)
            for k in [k for k in _W if k not in set()]:
                del _W[k]
            import gc
            gc.collect()
            from analysis.config import from_dict
            cfgp = os.path.join(REPO, cfg_rel)
            _W[key] = from_dict(json.load(open(cfgp)), cfgp).objective(data_centre=key[1])
        v = dict(values)
        v.update(held)
        return cfg_rel, held, float(_W[key].chi2(v)), None
    except Exception as e:                                   # a failing point is recorded, not fatal
        return cfg_rel, held, None, repr(e)[:300]


def _data_centre(item):
    if item["kind"] != "asimov":
        return None
    for mf in sorted(glob.glob(os.path.join(item["run_dir"], "b*", "asimov", "manifest.json")) +
                     glob.glob(os.path.join(item["run_dir"], "asimov", "manifest.json"))):
        runs = json.load(open(mf)).get("runs", [])
        dc = next((r["data_centre"] for r in reversed(runs) if r.get("data_centre")), None)
        if dc:
            return tuple(dc)
    raise RuntimeError(f"no data centre for the Asimov run {item['run_dir']}")


def run(manifest, task, out_dir, workers, part=0, nparts=1):
    """Every point of the task under each setting in turn: one pool of fresh processes per setting (the model's
    in-process flux cache does not know the settings), all the task's planes in it, so every worker stays busy.
    part/nparts (a resume on more jobs): this job takes every nparts-th of the task's points, from part on; what any
    earlier job wrote (every bracket.*.jsonl in out_dir) is skipped."""
    share = json.load(open(manifest))["tasks"][task][part::nparts]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bracket.{task}.jsonl" if nparts == 1 else f"bracket.{task}.p{part}.jsonl")
    done = set()
    for f in glob.glob(os.path.join(out_dir, "bracket.*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("chi2") is not None:
                done.add((r["ds"], r["plane"], json.dumps(r["held"], sort_keys=True), tuple(r["corner"])))
    dcs = {}
    for it in share:
        if it["run_dir"] not in dcs:
            dcs[it["run_dir"]] = _data_centre(it)
    with open(path, "a") as fh:
        for corner in CORNERS:
            todo = [it for it in share
                    if (it["ds"], it["plane"], json.dumps(it["held"], sort_keys=True), tuple(corner)) not in done]
            print(f"corner {corner}: {len(todo)} points", flush=True)
            if not todo:
                continue
            by_key = {(it["config"], json.dumps(it["held"], sort_keys=True)): it for it in todo}
            tasks = [(it["config"], dcs[it["run_dir"]], it["held"], it["values"]) for it in todo]
            k = 0
            with get_context("spawn").Pool(min(workers, len(todo)), initializer=_init, initargs=tuple(corner)) as pool:
                for cfg_rel, held, chi2, err in pool.imap_unordered(_one, tasks, chunksize=1):
                    it = by_key[(cfg_rel, json.dumps(held, sort_keys=True))]
                    fh.write(json.dumps(dict(ds=it["ds"], plane=it["plane"], held=held, corner=list(corner),
                                             chi2=chi2, error=err, stored=it["stored"], ref=it["ref"])) + "\n")
                    fh.flush()
                    k += 1
                    if k % 20 == 0:
                        print(f"  {k}/{len(todo)}", flush=True)
    print("done", flush=True)


def report(manifest, out_dir):
    recs = {}
    for f in glob.glob(os.path.join(out_dir, "bracket.*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            recs.setdefault((r["ds"], r["plane"]), {}).setdefault(json.dumps(r["held"], sort_keys=True), {})[
                tuple(r["corner"])] = r
    model = tuple(CORNERS[0])
    for (ds, plane), pts in sorted(recs.items()):
        ctrl = [abs(c[model]["chi2"] - c[model]["stored"]) for c in pts.values()
                if model in c and c[model]["chi2"] is not None]
        line = [f"{ds}/{plane:14s} {len(pts):3d} pts, control max {max(ctrl) if ctrl else float('nan'):.1e}"]
        for corner in CORNERS[1:]:
            d, flips = [], 0
            for c in pts.values():
                if corner in c and model in c and c[corner]["chi2"] is not None and c[model]["chi2"] is not None:
                    d.append(c[corner]["chi2"] - c[model]["chi2"])
                    was = c[model]["chi2"] - c[model]["ref"] >= THRESHOLD
                    now = c[corner]["chi2"] - c[model]["ref"] >= THRESHOLD
                    flips += was != now
            if d:
                line.append(f"N{corner[0]:g},1/E x{corner[1]:g}: dchi2 {min(d):+.2f}..{max(d):+.2f}, flips {flips}")
        err = sum(1 for c in pts.values() for r in c.values() if r.get("error"))
        print(" | ".join(line) + (f" | errors {err}" if err else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("manifest")
    p.add_argument("--root", required=True)
    p.add_argument("--tasks", type=int, required=True)
    p.add_argument("--datasets", default="")
    p.add_argument("--campaign", default="", help="config folder name (default: the root's name)")
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("task", type=int)
    r.add_argument("out_dir")
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--nparts", type=int, default=1)
    q = sub.add_parser("report")
    q.add_argument("manifest")
    q.add_argument("out_dir")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a.manifest, a.root, a.tasks, [s for s in a.datasets.split(",") if s], a.campaign)
    elif a.cmd == "run":
        run(a.manifest, a.task, a.out_dir, a.workers, a.part, a.nparts)
    else:
        report(a.manifest, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
