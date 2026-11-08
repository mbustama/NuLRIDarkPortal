"""
check_decay_back.py -- decision B (Mauricio, 2026-10-24; implemented 2026-11-07, src/dp_mediator.py): where the
mediator's decay back into neutrino pairs (dark_portal.mediator_decay_back) changes a finished run's planes, which
took the mediator as invisible. It fixes the mass ranges to rerun.

Points, on each plane of the run: (a) every point where the beam's decays make mediators that can decay back within
the trip, by generous criteria (the heavier neutrinos' decay exponent over a Hubble time at 50 TeV >= DBEAM_MIN, and
the mediator's own, at 25 TeV, >= DZ_MIN, with its width summed over the open pairs, m_k + m_l < M, at the point's
fitted lightest mass); (b) per mediator mass from 1e-6 to 0.06 eV (where a pair can be open and the decays made),
the two grid points that straddle the 95% edge. Each is re-evaluated at its stored solution (no refit; the flux cache
ignored; fresh processes per setting) with the decay back ON; every CONTROL_EVERY-th point also OFF (the control:
must give back the stored chi^2). A refit can only lower the ON chi^2.

  python tools/check_decay_back.py plan <manifest.json> --root <run output root> --tasks N [--campaign <configs>]
  python tools/check_decay_back.py run <manifest.json> <task> <out dir> [--workers 64]
  python tools/check_decay_back.py report <manifest.json> <out dir>
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
from bracket_relic_population import _records, _config_path, _data_centre, THRESHOLD  # noqa: E402

E_NU, E_Z = 5e13, 2.5e13
DBEAM_MIN, DZ_MIN = 0.01, 0.01
M_LO, M_HI = 1e-6, 0.06
CONTROL_EVERY = 10
_W = {}


def _flags(model, values):
    """(beam decay exponent, mediator decay exponent) at this point, the solver's own couplings and rates."""
    import dark_portal_solver as dps
    import dp_absorption
    import dp_decay
    from dp_core import construct_PMNS
    E, p = dps.build_solver_params(model._run_cfg(model._overrides(values)))
    m = np.asarray(p["m_nu_bg_arr"], dtype=np.float64)
    M = float(p["m_Zd_eV"])
    U = construct_PMNS(p["th12"], p["th13"], p["th23"], p["d_cp"])
    if p.get("G_coupling") is not None:
        G = np.asarray(p["G_coupling"], dtype=np.complex128)
    else:
        u = np.array([p["u_e"], p["u_mu"], p["u_tau"]], dtype=np.complex128)
        G = dp_absorption.dark_portal_coupling(p["g_D"], u, 0.0)
    Gam, _ = dp_decay.rest_rates(G, U, m, M)
    H0 = float(p["H0_eV"])
    d_beam = max((Gam[i].sum() * m[i] / E_NU) / H0 for i in range(3))
    Gp = U.conj().T @ G @ U
    gz = sum(abs(Gp[j, k]) ** 2 * M / (24.0 * np.pi) for j in range(3) for k in range(j, 3) if m[j] + m[k] < M)
    return d_beam, gz * (M / E_Z) / H0


def plan(manifest, root, tasks, campaign=""):
    from analysis.config import from_dict
    campaign = campaign or os.path.basename(os.path.normpath(root))
    items = []
    for ds in sorted(os.listdir(root)):
        for pd in sorted(glob.glob(os.path.join(root, ds, "*"))):
            plane = os.path.basename(pd)
            recs, ref, kind = _records(pd)
            if not recs:
                continue
            cfgp = _config_path(campaign, ds, plane)
            raw = json.load(open(cfgp))
            model = from_dict(raw, cfgp).model()
            xn, yn = raw["planes"][0]["x"], raw["planes"][0]["y"]
            pick = {}
            cols = {}
            for k, r in recs.items():
                M = r["held"][xn]
                if not M_LO <= M <= M_HI:
                    continue
                cols.setdefault(M, []).append(r)
                v = dict(r["values"])
                v.update(r["held"])
                db, dz = _flags(model, v)
                if db >= DBEAM_MIN and dz >= DZ_MIN:
                    pick[k] = r
            for M, col in cols.items():                      # the 95% edge, per mass
                col.sort(key=lambda r: r["held"][yn])
                exc = [i for i, r in enumerate(col) if r["chi2"] - ref >= THRESHOLD]
                if exc:
                    i = exc[0]
                    for r in ([col[i - 1]] if i > 0 else []) + [col[i]]:
                        pick[json.dumps(r["held"], sort_keys=True)] = r
            n0 = len(items)
            for r in pick.values():
                items.append(dict(config=os.path.relpath(cfgp, REPO), ds=ds, plane=plane, kind=kind,
                                  run_dir=os.path.abspath(pd), held=r["held"], values=r["values"],
                                  stored=r["chi2"], ref=ref))
            print(f"{ds}/{plane}: {len(items) - n0} points", flush=True)
    for k, it in enumerate(items):
        it["control"] = k % CONTROL_EVERY == 0
    shares = [items[k::tasks] for k in range(tasks)]
    json.dump(dict(tasks=shares, points=len(items)), open(manifest, "w"))
    print(f"{len(items)} points ({sum(i['control'] for i in items)} controls), {tasks} tasks -> {manifest}")


def _init(flag):
    os.environ.pop("NULRI_FLUX_CACHE", None)
    _W.clear()
    _W["flag"] = bool(flag)


def _one(task):
    cfg_rel, dc, held, values = task
    try:
        key = (cfg_rel, None if dc is None else tuple(dc))
        if key not in _W:
            # one plane's objective per worker at a time: each holds its model's caches and the event
            # likelihoods' Monte Carlo, and a worker that kept them all ran its node out of memory (2026-11-08)
            for k in [k for k in _W if k not in {"flag"}]:
                del _W[k]
            import gc
            gc.collect()
            from analysis.config import from_dict
            cfgp = os.path.join(REPO, cfg_rel)
            raw = json.load(open(cfgp))
            raw.setdefault("solver", {}).setdefault("dark_portal", {})["mediator_decay_back"] = _W["flag"]
            _W[key] = from_dict(raw, cfgp).objective(data_centre=key[1])
        v = dict(values)
        v.update(held)
        return cfg_rel, held, float(_W[key].chi2(v)), None
    except Exception as e:                                   # a failing point is recorded, not fatal
        return cfg_rel, held, None, repr(e)[:300]


def run(manifest, task, out_dir, workers):
    share = json.load(open(manifest))["tasks"][task]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"decay_back.{task}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("chi2") is not None:
                done.add((r["ds"], r["plane"], json.dumps(r["held"], sort_keys=True), r["on"]))
    dcs = {}
    for it in share:
        if it["run_dir"] not in dcs:
            dcs[it["run_dir"]] = _data_centre(it)
    with open(path, "a") as fh:
        for flag in (False, True):                           # the controls first: a wrong set-up shows at once
            todo = [it for it in share if (flag or it["control"])
                    and (it["ds"], it["plane"], json.dumps(it["held"], sort_keys=True), flag) not in done]
            print(f"decay back {'ON' if flag else 'OFF (control)'}: {len(todo)} points", flush=True)
            if not todo:
                continue
            by_key = {(it["config"], json.dumps(it["held"], sort_keys=True)): it for it in todo}
            tasks = [(it["config"], dcs[it["run_dir"]], it["held"], it["values"]) for it in todo]
            k = 0
            with get_context("spawn").Pool(min(workers, len(todo)), initializer=_init, initargs=(flag,)) as pool:
                for cfg_rel, held, chi2, err in pool.imap_unordered(_one, tasks, chunksize=1):
                    it = by_key[(cfg_rel, json.dumps(held, sort_keys=True))]
                    fh.write(json.dumps(dict(ds=it["ds"], plane=it["plane"], held=held, on=flag, chi2=chi2,
                                             error=err, stored=it["stored"], ref=it["ref"])) + "\n")
                    fh.flush()
                    k += 1
                    if k % 20 == 0:
                        print(f"  {k}/{len(todo)}", flush=True)
    print("done", flush=True)


def report(manifest, out_dir):
    recs = {}
    for f in glob.glob(os.path.join(out_dir, "decay_back.*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            recs.setdefault((r["ds"], r["plane"]), []).append(r)
    for (ds, plane), rs in sorted(recs.items()):
        ctrl = [abs(r["chi2"] - r["stored"]) for r in rs if not r["on"] and r["chi2"] is not None]
        on = [r for r in rs if r["on"] and r["chi2"] is not None]
        d = [r["chi2"] - r["stored"] for r in on]
        flips = [r for r in on if (r["stored"] - r["ref"] >= THRESHOLD) != (r["chi2"] - r["ref"] >= THRESHOLD)]
        masses = sorted({float(next(iter(r["held"].values()))) for r in flips})
        err = sum(1 for r in rs if r.get("error"))
        print(f"{ds}/{plane:14s} {len(on):4d} pts, control max {max(ctrl) if ctrl else float('nan'):.1e}; "
              f"dchi2 ON - stored {min(d) if d else 0:+.2f}..{max(d) if d else 0:+.2f}; flips {len(flips)}"
              + (f" at masses {['%.0e' % x for x in masses]}" if flips else "") + (f"; errors {err}" if err else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("manifest")
    p.add_argument("--root", required=True)
    p.add_argument("--tasks", type=int, required=True)
    p.add_argument("--campaign", default="")
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("task", type=int)
    r.add_argument("out_dir")
    r.add_argument("--workers", type=int, default=64)
    q = sub.add_parser("report")
    q.add_argument("manifest")
    q.add_argument("out_dir")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a.manifest, a.root, a.tasks, a.campaign)
    elif a.cmd == "run":
        run(a.manifest, a.task, a.out_dir, a.workers)
    else:
        report(a.manifest, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
