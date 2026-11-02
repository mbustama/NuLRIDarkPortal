"""
preflight_ladders.py -- the decay ladders of the final run, built before it (2026-10-25; the review's finding 1,
docs/dev/REVIEW_2026-10-25.md): at every point of the ultralight planes the final run fits (every raster column x
coupling row of analysis/configs/campaign_ml/now_ic26, allowed at the lightest mass, at lightest masses from 1e-6 to
0.1 eV), the cosmological and the Galactic dp_medium.Ladder exactly as production builds them (the solver's own
calls, stopped right after the ladder), recording failures, the plans' room (dp_medium grows it on demand) and the
validity measure zeta (the largest decay width over its kind's smallest splitting over the ladder's rungs up to the
run's largest sigma: an upper bound of what the run meets). Its records also give the zeta map (plot).

  run <out_dir> [--workers 32] [--families le_lmu_ul,...] [--ml 1e-6,...] [--all-points FAMILY,...]
      (--all-points: also the points not allowed, for those families: the zeta map's)
  summarize <out_dir>                              failures, rooms, zeta per family
  plot <out_dir> <figure.png> [--ml 1e-6,1e-3,1e-2,0.1]   zeta over the lepton-number planes, the unitarity line

Workers single-threaded (NUMBA_NUM_THREADS=1), as in the campaign.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
os.environ.pop("NULRI_FLUX_CACHE", None)

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

CONFIGS = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26")
FAMILIES = ["le_lmu_ul", "le_ltau_ul", "dp_e_ul", "dp_mu_ul", "dp_tau_ul"]
ML = [float(x) for x in np.geomspace(1e-6, 0.1, 11)]
_W = {}


class _Stop(Exception):
    pass


def _hooks():
    """Ladder.__init__ records the ladder and stops the run; the Galactic path stops after its processes' setup."""
    import dark_portal_solver as dps
    import dp_medium
    if _W.get("hooked"):
        return
    orig = dp_medium.Ladder.__init__

    def init(self, E_bins, G_flavor, U, m, M, kinds, c_max, c_min=0.0):
        t = time.time()
        try:
            orig(self, E_bins, G_flavor, U, m, M, kinds, c_max, c_min)
        except Exception as e:
            _W["rec"].update(error=repr(e)[:300])
            raise _Stop()
        E = np.asarray(E_bins, dtype=np.float64)
        top = E[-1] * abs(float(c_max)) * self.h ** 2
        use = self.sig <= max(top, self.sig_vac)
        z = self.zeno[use]
        _W["rec"].update(ok=True, zeta=float(np.max(z)) if z.size else 0.0, room=list(self.capacity),
                         nr=int(self.nr), c_max=float(c_max), t_ladder=round(time.time() - t, 3))
        raise _Stop()

    dp_medium.Ladder.__init__ = init
    orig_gpp = dps._galactic_path_processes

    def gpp(*a, **k):
        orig_gpp(*a, **k)
        _W["rec"].update(no_ladder=True)          # the Galactic processes without decays: no ladder
        raise _Stop()

    dps._galactic_path_processes = gpp
    _W["hooked"] = True


def _model(fam):
    if fam not in _W.setdefault("models", {}):
        from analysis import config as config_mod
        p = os.path.join(CONFIGS, f"{fam}.json")
        raw = copy.deepcopy(json.load(open(p)))
        raw.pop("exact_profile", None)
        cfg = config_mod.from_dict(raw, p)
        _W["models"][fam] = (cfg, cfg.model())
    return _W["models"][fam]


def _one(u):
    import dark_portal_solver as dps
    _hooks()
    out = dict(u)
    t0 = time.time()
    try:
        cfg, m = _model(u["fam"])
        v = {q.name: q.start for q in cfg.parameters}
        v.update({u["x"]: u["xv"], u["y"]: u["yv"], "m_lightest_eV": u["ml"]})
        E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
        params["show_progress"] = False
        params["n_cores"] = 1
        E = np.asarray(E, dtype=np.float64)
        out["mode"] = dps._cosmology_mode(params, E)
        out["decay_acts"] = bool(dps._decay_acts(params, E))
        if out["decay_acts"] and out["mode"] != "si":
            _W["rec"] = {}
            try:
                dps.propagate_weighted(E, params)
                out["cosmo"] = dict(reached_end=True)
            except _Stop:
                out["cosmo"] = dict(_W["rec"])
        if out["mode"] == "si" and out["decay_acts"]:
            out["zeta_vacuum"] = dps.vacuum_zeta(params)
        if m.galactic and float(params["m_Zd_eV"]) < dps.SI_COSMOLOGY_MIN_M_EV:
            _W["rec"] = {}
            try:
                dps.galactic_allsky(E, params)
                out["gal"] = dict(reached_end=True)
            except _Stop:
                out["gal"] = dict(_W["rec"])
    except Exception as e:
        out["other_error"] = repr(e)[:300] + " | " + traceback.format_exc()[-400:]
    out["sec"] = round(time.time() - t0, 2)
    return out


def _units(families, mls, all_points):
    from analysis import exclusion as ex
    units = []
    for fam in families:
        cfg, _ = _model(fam)
        pl = cfg.planes[0]
        xs = np.asarray(pl.x_grid, dtype=np.float64)
        ys = np.asarray(pl.y_grid, dtype=np.float64)
        p = os.path.join(CONFIGS, f"{fam}.json")
        for ml in mls:
            sl, cb = ex.not_allowed_boundary(p, ml)
            for ix, x in enumerate(xs):
                for iy, y in enumerate(ys):
                    allowed = bool(np.log10(y) < sl * np.log10(x) + cb)
                    if not allowed and fam not in all_points:
                        continue
                    units.append(dict(fam=fam, x=pl.x, y=pl.y, ix=ix, iy=iy, xv=float(x), yv=float(y), ml=float(ml),
                                      allowed=allowed))
    return units


def run(out_dir, workers, families, mls, all_points):
    os.makedirs(out_dir, exist_ok=True)
    units = _units(families, mls, all_points)
    path = os.path.join(out_dir, "ladders.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["fam"], r["ix"], r["iy"], r["ml"]))
    todo = [u for u in units if (u["fam"], u["ix"], u["iy"], u["ml"]) not in done]
    # the expensive ones (decays at sub-keV masses) spread over the workers
    rng = np.random.default_rng(5)
    todo = [todo[i] for i in rng.permutation(len(todo))]
    print(f"{len(units)} units, {len(todo)} to do", flush=True)
    t0 = time.time()
    with open(path, "a") as fh, get_context("spawn").Pool(max(1, min(workers, len(todo)))) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, todo, chunksize=4)):
            fh.write(json.dumps(r) + "\n")
            if k % 500 == 0:
                fh.flush()
                print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f} s", flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f} s", flush=True)
    return 0


def _load(out_dir):
    recs = []
    for p in glob.glob(os.path.join(out_dir, "*.jsonl")):
        recs += [json.loads(line) for line in open(p)]
    return recs


def summarize(out_dir):
    recs = _load(out_dir)
    bad = 0
    for fam in FAMILIES:
        rs = [r for r in recs if r["fam"] == fam]
        if not rs:
            continue
        n_c = sum(1 for r in rs if "cosmo" in r)
        n_g = sum(1 for r in rs if "gal" in r and r["gal"].get("ok"))
        fails = [r for r in rs if r.get("other_error") or r.get("cosmo", {}).get("error")
                 or r.get("gal", {}).get("error")]
        ends = sum(1 for r in rs if r.get("cosmo", {}).get("reached_end") or r.get("gal", {}).get("reached_end"))
        bad += len(fails)
        rooms = [tuple(r[k]["room"]) for r in rs for k in ("cosmo", "gal") if r.get(k, {}).get("room")]
        grown = sorted(set(x for x in rooms if x != (24, 16)))
        za = [max(r.get("cosmo", {}).get("zeta", 0.0), r.get("gal", {}).get("zeta", 0.0), r.get("zeta_vacuum", 0.0))
              for r in rs if r.get("allowed")]
        print(f"{fam}: {len(rs)} units ({sum(r.get('allowed', False) for r in rs)} allowed), cosmological ladders "
              f"{n_c}, Galactic {n_g}; failures {len(fails)}; runs without the expected ladder {ends}; rooms grown {grown}; zeta at allowed points: max "
              f"{max(za) if za else 0:.3g}, > 0.05 at {sum(z > 0.05 for z in za)}, > 0.1 at {sum(z > 0.1 for z in za)}")
        for r in fails[:5]:
            print("   ", {k: r.get(k) for k in ("xv", "yv", "ml", "mode", "other_error", "cosmo", "gal")})
    t = [r["sec"] for r in recs]
    print(f"{len(recs)} units, {bad} failures; seconds per unit: median {np.median(t):.2f}, max {max(t):.1f}")
    return 1 if bad else 0


def plot(out_dir, figure, mls):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from analysis import figure_style
    from analysis import exclusion as ex
    figure_style.use()
    recs = _load(out_dir)
    fams = ["le_lmu_ul", "le_ltau_ul"]
    names = {"le_lmu_ul": r"$L_e - L_\mu$", "le_ltau_ul": r"$L_e - L_\tau$"}
    fig, axes = plt.subplots(len(fams), len(mls), figsize=(3.2 * len(mls), 3.0 * len(fams)), sharex=True,
                             sharey=True, squeeze=False)
    norm = LogNorm(1e-8, 1.0)
    im = None
    for i, fam in enumerate(fams):
        cfg, _ = _model(fam)
        pl = cfg.planes[0]
        xs = np.asarray(pl.x_grid, dtype=np.float64)
        ys = np.asarray(pl.y_grid, dtype=np.float64)
        for j, ml in enumerate(mls):
            ax = axes[i, j]
            Z = np.full((len(ys), len(xs)), np.nan)
            for r in recs:
                if r["fam"] != fam or abs(np.log10(r["ml"] / ml)) > 1e-6:
                    continue
                z = max(r.get("cosmo", {}).get("zeta", 0.0), r.get("gal", {}).get("zeta", 0.0),
                        r.get("zeta_vacuum", 0.0))
                if z > 0.0:
                    Z[r["iy"], r["ix"]] = z
            lx, ly = np.log10(xs), np.log10(ys)
            dx, dy = 0.5 * (lx[1] - lx[0]), 0.5 * (ly[1] - ly[0])
            im = ax.imshow(Z, origin="lower", extent=(lx[0] - dx, lx[-1] + dx, ly[0] - dy, ly[-1] + dy),
                           aspect="auto", norm=norm, cmap="viridis", interpolation="nearest")
            ok = np.isfinite(Z)
            if np.any(ok):
                cs = ax.contour(lx, ly, np.where(ok, Z, 0.0), levels=[0.05, 0.1], colors=["w", "r"],
                                linewidths=1.0)
                ax.clabel(cs, fmt={0.05: "0.05", 0.1: "0.1"}, fontsize=7)
            sl, cb = ex.not_allowed_boundary(os.path.join(CONFIGS, f"{fam}.json"), ml)
            xx = np.array([lx[0], lx[-1]])
            ax.plot(xx, sl * xx + cb, "k--", lw=1.2)
            ax.set_xlim(lx[0] - dx, lx[-1] + dx)
            ax.set_ylim(ly[0] - dy, ly[-1] + dy)
            if i == 0:
                ax.set_title(rf"$m_{{\rm lightest}} = {ml:g}$ eV")
            if j == 0:
                ax.set_ylabel(names[fam] + r": $\log_{10}(g_D)$")
            if i == len(fams) - 1:
                ax.set_xlabel(r"$\log_{10}(m_{Z'}/{\rm eV})$")
    cb_ = fig.colorbar(im, ax=axes, shrink=0.85)
    cb_.set_label(r"$\zeta = 2E\Gamma/\min|\mu_a - \mu_b|$")
    fig.savefig(figure, dpi=200, bbox_inches="tight")
    print("written", figure)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--families", default=",".join(FAMILIES))
    r.add_argument("--ml", default=",".join(f"{x:.6g}" for x in ML))
    r.add_argument("--all-points", default="")
    s = sp.add_parser("summarize")
    s.add_argument("out_dir")
    p = sp.add_parser("plot")
    p.add_argument("out_dir")
    p.add_argument("figure")
    p.add_argument("--ml", default="1e-6,1e-3,1e-2,0.1")
    a = ap.parse_args()
    if a.cmd == "run":
        return run(a.out_dir, a.workers, a.families.split(","), [float(x) for x in a.ml.split(",")],
                   [f for f in a.all_points.split(",") if f])
    if a.cmd == "summarize":
        return summarize(a.out_dir)
    return plot(a.out_dir, a.figure, [float(x) for x in a.ml.split(",")])


if __name__ == "__main__":
    sys.exit(main())
