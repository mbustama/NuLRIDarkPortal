"""
compare_code_versions.py -- the fluxes of two versions of the code at the same points (stage 2, 2026-10-24): the MeV
planes (dark portal e, mu, tau; L_e - L_mu, L_e - L_tau) and the secret-interaction planes, which the ultralight work
must leave unchanged. Each version computes what the analysis computes (analysis.model's _fluxes_compute: the
pure-flavor sources, the Galactic sky average; F, G) and the model's fluxes between its spectral-index nodes
(analysis.model.fluxes at p_index 2.63, p_index_low 1.13: the index spline and the broken power law's join; MF, MG)
with its own src/ and analysis/, from the same configuration files.

  plan <units.json> [--per-plane 40]
  run <units.json> <task> <n_tasks> <out_dir> --root <repository> [--workers 32]
  compare <dir_old> <dir_new>
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEV = ["dp_e_mev", "dp_mu_mev", "dp_tau_mev", "le_lmu_mev", "le_ltau_mev"]
_W = {}


def plan(path, per_plane):
    import numpy as np
    sys.path[:0] = [os.path.join(HERE, "src"), HERE]
    from analysis import config as config_mod
    files = [os.path.join(HERE, "analysis", "configs", "campaign_ml", "now_ic26", f"{f}.json") for f in MEV]
    files += sorted(p for p in glob.glob(os.path.join(HERE, "analysis", "configs", "pion_2050_case_5_si_*.json"))
                    if not p.endswith(("_gal.json", "_galx.json")))
    rng = np.random.default_rng(5)
    units = []
    for p in files:
        raw = copy.deepcopy(json.load(open(p)))
        raw.pop("exact_profile", None)
        cfg = config_mod.from_dict(raw, p)
        pl = cfg.planes[0]
        arrays = [v for v in vars(pl).values() if isinstance(v, np.ndarray)]
        names = [v for v in vars(pl).values() if isinstance(v, str)]
        ml = [q for q in cfg.parameters if q.name == "m_lightest_eV"]
        for _ in range(per_plane):
            ix, iy = int(rng.integers(len(arrays[0]))), int(rng.integers(len(arrays[1])))
            u = dict(config=os.path.relpath(p, HERE), x=names[0], y=names[1], xv=float(arrays[0][ix]),
                     yv=float(arrays[1][iy]))
            if ml:
                u["ml"] = float(rng.choice([1e-6, 1e-3, 1e-2, 0.1]))
            u["id"] = f"{os.path.basename(p)[:-5]}:{ix}:{iy}"
            units.append(u)
    json.dump(units, open(path, "w"))
    print(f"{len(units)} units -> {path}")
    return 0


def _one(u):
    import numpy as np
    out = dict(id=u["id"])
    t = time.time()
    try:
        import dark_portal_solver as dps
        from analysis import config as config_mod
        p = os.path.join(HERE, u["config"])
        if _W.get("p") != p:
            raw = copy.deepcopy(json.load(open(p)))
            raw.pop("exact_profile", None)
            cfg = config_mod.from_dict(raw, p)
            _W.update(p=p, cfg=cfg, model=cfg.model())
        cfg, m = _W["cfg"], _W["model"]
        v = {q.name: q.start for q in cfg.parameters}
        v.update({u["x"]: u["xv"], u["y"]: u["yv"]})
        if "ml" in u:
            v["m_lightest_eV"] = u["ml"]
        E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
        params["show_progress"] = False
        params["n_cores"] = 1
        sources = None if m.basis is None else [np.eye(3)[i] for i in m.basis]
        if m.ratio_free:
            Fa, Fb = dps.propagate_nu_nubar(E, params, sources=sources)
            F = np.concatenate([np.atleast_3d(np.asarray(Fa)), np.atleast_3d(np.asarray(Fb))], axis=0)
        else:
            F = np.asarray(dps.propagate_weighted(E, params, sources=sources))
        out["F"] = F
        if m.galactic:
            out["G"] = np.asarray(dps.galactic_allsky(E, params))
        # the analysis model's fluxes between its spectral-index nodes (the index spline and the broken power law's
        # join, analysis.model.fluxes): p_index 2.63, p_index_low 1.13 where they are parameters
        names = {q.name for q in cfg.parameters}
        vm = dict(v)
        if "p_index" in names:
            vm["p_index"] = 2.63
        if "p_index_low" in names:
            vm["p_index_low"] = 1.13
        _, Fa, _, gal = m.fluxes(vm)
        out["MF"] = np.asarray(Fa)
        if gal is not None and gal.get("G") is not None:
            out["MG"] = np.asarray(gal["G"])
    except Exception as e:
        out["error"] = repr(e)[:300]
    out["seconds"] = time.time() - t
    return out


def _init(root):
    os.environ.pop("NULRI_FLUX_CACHE", None)
    sys.path[:0] = [os.path.join(root, "src"), root]


def run(path, task, n_tasks, out_dir, root, workers):
    import numpy as np
    from multiprocessing import get_context
    units = json.load(open(path))[task::n_tasks]
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    with get_context("spawn").Pool(min(workers, len(units)), initializer=_init, initargs=(root,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, units, chunksize=1)):
            res[r["id"]] = r
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(units)}", flush=True)
    arrays = {}
    meta = {}
    for i, r in res.items():
        meta[i] = dict(error=r.get("error"), seconds=r["seconds"])
        for key in ("F", "G", "MF", "MG"):
            if key in r:
                arrays[f"{i}|{key}"] = r[key]
    np.savez_compressed(os.path.join(out_dir, f"fluxes.{task}.npz"), **arrays)
    json.dump(meta, open(os.path.join(out_dir, f"meta.{task}.json"), "w"))
    print("done", flush=True)
    return 0


def compare(d_old, d_new):
    import numpy as np

    def load(d):
        arr, meta = {}, {}
        for p in glob.glob(os.path.join(d, "fluxes.*.npz")):
            with np.load(p) as z:
                arr.update({k: z[k] for k in z.files})
        for p in glob.glob(os.path.join(d, "meta.*.json")):
            meta.update(json.load(open(p)))
        return arr, meta
    a, ma = load(d_old)
    b, mb = load(d_new)
    print(f"units: old {len(ma)} new {len(mb)}; errors old {sum(1 for x in ma.values() if x['error'])}, "
          f"new {sum(1 for x in mb.values() if x['error'])}")
    for k, x in mb.items():
        if x["error"]:
            print("  new error", k, x["error"])
    fam = {}
    for k in sorted(set(a) & set(b)):
        A, B = a[k], b[k]
        same = A.shape == B.shape and np.array_equal(A, B)
        tot = np.abs(A).sum(-1, keepdims=True)
        rel = float(np.max(np.abs(B - A) / np.maximum(tot, 1e-300))) if A.shape == B.shape else np.inf
        f = k.split(":")[0] + "|" + k.split("|")[1]
        n, ns, mx = fam.get(f, (0, 0, 0.0))
        fam[f] = (n + 1, ns + int(same), max(mx, rel))
    for f, (n, ns, mx) in sorted(fam.items()):
        print(f"{f:45s} {n:4d} compared, {ns:4d} bit for bit, largest change {mx:.2e}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("plan")
    a.add_argument("units")
    a.add_argument("--per-plane", type=int, default=40)
    a = sp.add_parser("run")
    a.add_argument("units")
    a.add_argument("task", type=int)
    a.add_argument("n_tasks", type=int)
    a.add_argument("out_dir")
    a.add_argument("--root", required=True)
    a.add_argument("--workers", type=int, default=32)
    a = sp.add_parser("compare")
    a.add_argument("old")
    a.add_argument("new")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        return plan(args.units, args.per_plane)
    if args.cmd == "run":
        return run(args.units, args.task, args.n_tasks, args.out_dir, args.root, args.workers)
    return compare(args.old, args.new)


if __name__ == "__main__":
    sys.exit(main())
