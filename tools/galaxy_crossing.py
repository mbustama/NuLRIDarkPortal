"""
galaxy_crossing.py -- what the extragalactic beam's decays while it crosses the Galaxy would change (2026-10-25; the
review's finding 3, docs/dev/REVIEW_2026-10-25.md). The transport takes the crossing as coherent and decay-free
(dp_transit); here the beam's eigenstate populations at z = 0 (the cosmological phase's, which the adiabatic transit
carries into the local eigenstates in order) are evolved along a few Galactic lines of sight, from 30 kpc to the
Sun, through the local potential's eigenstates with the transport's Galactic processes (dp_galactic_path's
crossing_populations: the jumps, the in-medium decays and the loss, each energy bin exact over each segment), and
both versions go on through the Sun and the Earth
(propagate_earth_spatial). Measured: at the detector, over 1e13-1e16 eV where the flux is >= 1e-3 of the standard
one, the total's change (range), its shape (median ratio divided out) and the flavor fractions' largest change.

  python tools/galaxy_crossing.py <out.json> [--points le_lmu_ul:M:g:ml,...] [--ratios 0.1,1,10] [--workers N]

A measurement: the transport itself crosses the Galaxy since 2026-10-25 (dark_portal_solver._flux_runs, per
declination band); this tool switches that off and does it per line of sight.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
os.environ.pop("NULRI_FLUX_CACHE", None)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

CONFIGS = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26")
# toward the Galactic centre, the anticentre, a pole and one in between (l, b in degrees)
LINES = {"centre": (0.0, 0.0), "anticentre": (180.0, 0.0), "pole": (0.0, 90.0), "l90_b30": (90.0, 30.0)}


class _Stop(Exception):
    pass


def _model(fam):
    from analysis import config as config_mod
    p = os.path.join(CONFIGS, f"{fam}.json")
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    return cfg, cfg.model()


def _galaxy(E_any, params):
    """The Galactic lines of sight's potential: per kind coef2 and the densities n (N_pix, N_s) at the working
    resolution, the bins' edges (kpc) and the pixels' (l, b), from the solver's own Galactic setup."""
    import dark_portal_solver as dps
    import dp_galactic
    import dp_galactic_path
    import dp_sky
    got = {}
    orig = dps._galactic_path_processes

    def hook(E_bins, params_, setup, kind_info, f_S, w_gas, q_gas, q_src, work, nside, template):
        got.update(kinds=[dict(coef2=float(k["coef2"]), n=np.array(k["n"]), M_rot=np.array(k["M_rot"]),
                               S_signed=np.array(k["S_signed"])) for k in kind_info], work=int(work),
                   u_arr=np.array(setup["u_arr_base"]))
        raise _Stop()

    dps._galactic_path_processes = hook
    try:
        dps.galactic_allsky(np.asarray(E_any), params)
    except _Stop:
        pass
    finally:
        dps._galactic_path_processes = orig
    s, ds, _, _ = dp_galactic.emission_geometry(got["work"])
    got["edges"] = dp_galactic_path.edges_from_geometry(s, ds)
    grid = dp_sky.SkyGrid(got["work"])
    got["l"], got["b"] = np.degrees(grid.l), np.degrees(grid.b)
    return got


def _pixel(gal, l, b):
    lr, br = np.radians(gal["l"]), np.radians(gal["b"])
    c = np.cos(br) * np.cos(np.radians(b)) * np.cos(lr - np.radians(l)) + np.sin(br) * np.sin(np.radians(b))
    return int(np.argmax(c))


def _metrics(A, X, std, E, lo=1e13, hi=1e16):
    totA, totX = A.sum(-1), X.sum(-1)
    sel = (E >= lo) & (E <= hi) & (totA >= 1e-3 * std) & (totA > 0)
    if not np.any(sel):
        return dict(seen=0)
    r = totX[sel] / totA[sel]
    fA = A[sel] / totA[sel, None]
    fX = X[sel] / totX[sel, None]
    return dict(seen=int(sel.sum()), rmin=float(r.min()), rmax=float(r.max()),
                d_shape=float(np.max(np.abs(r / np.median(r) - 1.0))), d_flavor=float(np.max(np.abs(fX - fA))))


def _one(pt):
    import dark_portal_solver as dps
    import dp_galactic_path as gp
    dps.EXTRAGALACTIC_CROSSING = False          # the transport's crossing off: this tool does it per line of sight
    fam, M, g, ml = pt
    t0 = time.time()
    out = dict(fam=fam, M=M, g=g, ml=ml)
    try:
        cfg, m = _model(fam)
        pl = cfg.planes[0]
        v = {q.name: q.start for q in cfg.parameters}
        v.update({pl.x: M, pl.y: g, "m_lightest_eV": ml})
        E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
        params["show_progress"] = False
        params["n_cores"] = 1
        E = np.asarray(E, dtype=np.float64)
        out["mode"] = dps._cosmology_mode(params, E)
        gal = _galaxy(E, params)
        pix = {name: _pixel(gal, l, b) for name, (l, b) in LINES.items()}
        calls = []
        orig = dps.propagate_earth_spatial

        def earth(E_bins, F, *a, **k):
            # every handover (the analysis' runs: both kinds of each spectral piece), kept to be redone
            calls.append((bool(a[11]), np.array(E_bins), np.array(F), a, k))
            return orig(E_bins, F, *a, **k)

        dps.propagate_earth_spatial = earth
        try:
            Fn, Fb = dps.propagate_nu_nubar(E, params)
        finally:
            dps.propagate_earth_spatial = orig
        pairs = [(calls[i], calls[i + 1]) for i in range(0, len(calls) - 1, 2)
                 if not calls[i][0] and calls[i + 1][0]]
        if not pairs or 2 * len(pairs) != len(calls):
            out["error"] = f"handovers not in (nu, nubar) pairs: {[c[0] for c in calls]}"
            return out
        out["pieces"] = len(pairs)
        E_run = pairs[0][0][1]
        # the transport's own Galactic processes and lines of sight (dark_portal_solver._crossing_bands' pieces)
        setup = dps._flux_setup(E_run, params)
        kind_info = dps._galactic_kinds(params, setup, 16)
        gr = dps._galactic_rates(E_run, params, setup, kind_info, 16)
        if gr is None:
            out["no_process"] = True
            return out
        tables = dps._galactic_tables(E_run, kind_info, gr, gr["vac"])
        kinds_x = [dict(n=kd["n"], coef2=kd["coef2"], tables=tables[kk]) for kk, kd in enumerate(kind_info)]
        names = list(pix)
        groups = (np.arange(len(names) + 1, dtype=np.int64), np.array([pix[nm] for nm in names], dtype=np.int64))
        Pc = np.zeros((len(pairs), len(E_run), 2, 3))
        for c, (cn, cb) in enumerate(pairs):
            Pc[c, :, 0, :] = cn[2]
            Pc[c, :, 1, :] = cb[2]
        Qx = gp.crossing_populations(E_run, Pc, gr["edges_kpc"], kinds_x, gr["rates"], groups)
        # the standard flux (no new physics), for the 1e-3 floor
        vs = dict(v)
        vs[pl.y] = float(np.min(pl.y_grid)) * 1e-40
        vs["m_Zd_eV"] = 1e3
        Es, ps = dps.build_solver_params(m._run_cfg(m._overrides(vs)))
        ps["show_progress"] = False
        Sn, Sb = dps.propagate_nu_nubar(np.asarray(Es), ps)
        std = (np.asarray(Sn) + np.asarray(Sb)).sum(-1)
        N = len(E)
        # the decay-free crossing through the same calls (what the transport computes)
        base = sum(np.asarray(orig(c[1], c[2], *c[3], **c[4]))[:N] for pr in pairs for c in pr)
        out["check_base"] = float(np.max(np.abs(base - (np.asarray(Fn) + np.asarray(Fb)))) / np.max(np.abs(base)))
        hb = np.sqrt(E_run[1] / E_run[0])
        dE = E_run * (hb - 1 / hb)
        sel = (E_run >= 1e13) & (E_run <= 1e16)
        for ig, name in enumerate(names):
            p = pix[name]
            X = 0.0
            n_in = n_out = 0.0
            P_sel = Q_sel = 0.0
            for c, (cn, cb) in enumerate(pairs):
                Pn, Pb = cn[2], cb[2]
                Q0, Q1 = np.ascontiguousarray(Qx[ig, c, :, 0, :]), np.ascontiguousarray(Qx[ig, c, :, 1, :])
                n_in += float(((Pn + Pb).sum(1) * dE).sum())
                n_out += float(((Q0 + Q1).sum(1) * dE).sum())
                P_sel = P_sel + (Pn + Pb)[sel].sum(1)
                Q_sel = Q_sel + (Q0 + Q1)[sel].sum(1)
                X = X + np.asarray(orig(cn[1], Q0, *cn[3], **cn[4]))[:N] + np.asarray(orig(cb[1], Q1, *cb[3],
                                                                                            **cb[4]))[:N]
            met = _metrics(base, X, std, E)
            dep = 1.0 - Q_sel / np.maximum(P_sel, 1e-300)
            met.update(number=n_out / n_in, depth_max=float(np.max(np.abs(dep))), pixel=p,
                       l=float(gal["l"][p]), b=float(gal["b"][p]))
            out[name] = met
    except Exception as e:
        import traceback
        out["error"] = repr(e)[:300] + " | " + traceback.format_exc()[-600:]
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--points", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ratios", default="0.1,1,10", help="g/M values (eV^-1) of the default points")
    a = ap.parse_args()
    if a.points:
        pts = [(p.split(":")[0], float(p.split(":")[1]), float(p.split(":")[2]), float(p.split(":")[3]))
               for p in a.points.split(",")]
    else:
        pts = [(fam, M, r * M, ml) for fam in ("le_lmu_ul", "le_ltau_ul") for M in (1e-24, 1e-18, 1e-12, 1e-8)
               for r in [float(x) for x in a.ratios.split(",")] for ml in (1e-2,)]
    with get_context("spawn").Pool(max(1, min(a.workers, len(pts)))) as pool:
        res = pool.map(_one, pts, chunksize=1)
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        if r.get("error"):
            print(r["fam"], r["M"], r["g"], r["ml"], "ERROR", r["error"][:200])
            continue
        parts = []
        if r.get("no_process"):
            print(f"{r['fam']} M {r['M']:.0e} g {r['g']:.0e}: no Galactic process acts")
            continue
        for name in LINES:
            q = r.get(name)
            if q is None:
                continue
            parts.append(f"{name}: depth {q['depth_max']:.1e} shape {q.get('d_shape', float('nan')):.1e} "
                         f"flavor {q.get('d_flavor', float('nan')):.1e}")
        print(f"{r['fam']} M {r['M']:.0e} g {r['g']:.0e} ml {r['ml']:g} [{r['mode']}] " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
