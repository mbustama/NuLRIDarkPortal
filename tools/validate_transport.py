"""
validate_transport.py -- the transport's validation over the final run's grid (stage 2, 2026-10-24): at every point
of the ultralight planes (dark portal e, mu, tau; L_e - L_mu, L_e - L_tau; the transport is the same for the pion and
the fitted compositions and for the three data sets), the fluxes the analysis computes (analysis.model's
_fluxes_compute: the pure-flavor sources, the Galactic sky average) against variants of the numerics and of the
transport:

  A   the default (what the final run computes); finiteness, negativity, the mode used
  B   time steps refined: the dark-portal phase's redshift steps x4, dp_si's substeps x2
  C   energy grid refined x2 (877 bins, every other one the default's): the sub-grid units and every dark-portal one
  H   the grid extended a decade above 30 PeV (the inflow from above): the same units; GH its Galactic part
  D   the other transport(s) below 1 keV (dark-portal phase <-> dp_si; both where 'combined')
  E   the shortcuts off (SOFT_JUMP_MIN = 0, SI_HARD_MIN_G = 0), where that changes the run
  F   decay alone (relic scatterings off) against the exact nu_3 survival averaged over the sources, at 100 TeV and
      1 PeV, where the decay is open and the potential negligible
  GA  the Galactic sky average, default; GB its line-of-sight pieces x2 (sub-grid units); GC the per-pixel sweep where
      the vacuum path was used (every fourth column and row at m_lightest 1e-6 eV: 1000 s a point); GD the energy grid
      x2 (sub-grid units)

The grid: every point at m_lightest = 1e-6 eV; every second column and row at 1e-3, 1e-2, 0.1 eV. Each comparison:
d = max over energies, sources and flavors of |F_X - F_A| / max(F_A,tot, 1e-3 F_std,tot) (F_std the flux without new
physics), and the range of F_X,tot / F_A,tot. Workers single-threaded, as in the campaign.

  plan <manifest.json> --tasks N [--pilot K] [--part all|threaded|single]   (VT_THREADS: numba threads per worker)
  run <manifest.json> <task> <out_dir> [--workers 32]
  summarize <out_dir> <figure.png>
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
os.environ.pop("NULRI_FLUX_CACHE", None)          # every flux computed, none read from a disk cache

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

CONFIGS = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26")
FAMILIES = ["dp_e_ul", "dp_mu_ul", "dp_tau_ul", "le_lmu_ul", "le_ltau_ul"]
ML_FULL = 1e-6
ML_SUB = [1e-3, 1e-2, 1e-1]
FLOOR = 1e-3
E_CHECK = (1e14, 1e15)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
_W = {}


def _load(fam):
    from analysis import config as config_mod
    p = os.path.join(CONFIGS, f"{fam}.json")
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    return raw, cfg


def _axes(cfg):
    pl = cfg.planes[0]
    fields = vars(pl)
    arrays = [v for v in fields.values() if isinstance(v, np.ndarray)]
    names = [v for v in fields.values() if isinstance(v, str)]
    return names[0], names[1], arrays[0], arrays[1]


def _threaded(u):
    """The units carrying the per-pixel sweep check (GC: ~1000 s a point on one core), run with numba threads."""
    return u["xv"] < 1e3 and u["ix"] % 4 == 0 and u["iy"] % 4 == 0 and u["ml"] == ML_FULL


def plan(manifest, n_tasks, pilot, part="all"):
    units = []
    for fam in FAMILIES:
        raw, cfg = _load(fam)
        xn, yn, xs, ys = _axes(cfg)
        G_scale = 1.0
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                sub = (ix % 2 == 0) and (iy % 2 == 0)
                for ml in [ML_FULL] + (ML_SUB if sub else []):
                    # a rough cost for the balance: dp_si's runs ~20x the dark-portal phase's; the Galactic sweep
                    # measured on the first map (2026-10-24): dp_si units ~200 s (x3 with the sub-grid's checks), the
                    # Galactic sweeps of the lepton-number planes below 1 keV ~500 s, the dark-portal phase's ~25 s
                    heavy = (x >= 1e3) or (y * G_scale >= 1e-4)
                    cost = (200.0 if heavy else 25.0) * (3.0 if sub else 1.0)
                    if x < 1e3 and fam.startswith("le"):
                        cost += 500.0 if sub else 150.0
                    if x < 1e3 and ix % 4 == 0 and iy % 4 == 0 and ml == ML_FULL:
                        cost += 1000.0
                    units.append(dict(fam=fam, x=xn, y=yn, ix=ix, iy=iy, xv=float(x), yv=float(y), ml=ml, sub=sub,
                                      cost=cost))
    if part != "all":
        units = [u for u in units if _threaded(u) == (part == "threaded")]
    if pilot:
        rng = np.random.default_rng(11)
        units = [units[i] for i in rng.choice(len(units), size=pilot, replace=False)]
    order = np.argsort([-u["cost"] for u in units])
    loads = np.zeros(n_tasks)
    tasks = [[] for _ in range(n_tasks)]
    for i in order:                                  # longest first onto the least loaded task
        k = int(np.argmin(loads))
        tasks[k].append(units[i])
        loads[k] += units[i]["cost"]
    json.dump(dict(tasks=tasks, units=len(units)), open(manifest, "w"))
    print(f"{len(units)} units -> {manifest}; load per task {loads.min():.0f}-{loads.max():.0f}")
    return 0


def _model(fam):
    if _W.get("fam") != fam:
        raw, cfg = _load(fam)
        _W.update(fam=fam, raw=raw, cfg=cfg, model=cfg.model(), std=None)
    return _W["cfg"], _W["model"]


def _flux(m, solver_cfg, params_mod=None, galactic=False, ext=True):
    """analysis.model._fluxes_compute, with the params changed by params_mod (dict) before the run."""
    import dark_portal_solver as dps
    E, params = dps.build_solver_params(solver_cfg)
    params["show_progress"] = False
    params["n_cores"] = int(os.environ.get("VT_THREADS", 1))
    if params_mod:
        params.update(params_mod)
    sources = None if m.basis is None else [np.eye(3)[i] for i in m.basis]
    out = dict(E=np.asarray(E), mode=dps._cosmology_mode(params, np.asarray(E)))
    if ext:
        if m.ratio_free:
            Fa, Fb = dps.propagate_nu_nubar(E, params, sources=sources)
            out["F"] = np.concatenate([np.atleast_3d(np.asarray(Fa)), np.atleast_3d(np.asarray(Fb))], axis=0)
        else:
            F = np.asarray(dps.propagate_weighted(E, params, sources=sources))
            out["F"] = F if F.ndim == 3 else F[None]
    if galactic:
        out["G"] = np.asarray(dps.galactic_allsky(E, params))
    return out, params


def _cmp(A, X, EA, EX, std):
    """X against A (arrays (n, N_E, 3)), on A's energies: d, the largest change per energy, flavor and source relative
    to max(F_A,tot, 1e-3 F_std,tot); and, the fits' normalization being free (0.01-1000), what it cannot absorb: d_shape,
    the largest change of F_tot after the sources' median ratio is divided out, and d_flavor, the largest change of
    the flavor fractions, both over the energies where F_A,tot >= 1e-3 F_std,tot (F_std, the flux without new physics)."""
    if len(EX) != len(EA):
        idx = np.array([int(np.argmin(np.abs(np.log(EX / e)))) for e in EA])
        if np.max(np.abs(EX[idx] / EA - 1.0)) > 1e-9:
            raise ValueError("energy grids do not nest")
        X = X[:, idx]
    totA = A.sum(-1)
    totX = X.sum(-1)
    den = np.maximum(totA, FLOOR * std[None, :])
    d = np.abs(X - A).max(-1) / np.where(den > 0, den, np.inf)
    k = np.unravel_index(int(np.argmax(d)), d.shape)
    seen = (totA >= FLOOR * std[None, :]) & (totA > 0)
    d_shape = d_flavor = 0.0
    rmin = rmax = 1.0
    if np.any(seen):
        r = np.where(seen, totX / np.where(totA > 0, totA, 1.0), np.nan)
        rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
        med = np.nanmedian(r, axis=1, keepdims=True)
        d_shape = float(np.nanmax(np.abs(r / np.where(med > 0, med, 1.0) - 1.0)))
        fA = A / np.where(totA > 0, totA, 1.0)[..., None]
        fX = X / np.where(totX > 0, totX, 1.0)[..., None]
        d_flavor = float(np.max(np.where(seen[..., None], np.abs(fX - fA), 0.0)))
    return dict(d=float(d.max()), E=float(EA[k[1]]), d_shape=d_shape, d_flavor=d_flavor, rmin=rmin, rmax=rmax,
                identical=bool(np.array_equal(A, X)))


def _health(F, std):
    tot = F.sum(-1)
    return dict(finite=bool(np.all(np.isfinite(F))), neg=float(min(0.0, F.min()) / max(F.max(), 1e-300)),
                over=float(np.max(tot / np.maximum(std[None, :], 1e-300))))


def _std(m, v, y_name, y_low, gal):
    """The flux without new physics (the plane's coupling 1e-40 of its range's bottom at m_Zd = 1 keV; per family,
    the masses then immaterial): extragalactic and Galactic, per source, energy and flavor."""
    if _W.get("std") is None:
        vs = dict(v)
        vs[y_name] = y_low * 1e-40
        vs["m_Zd_eV"] = 1e3          # no decay, a contact-range potential (no exterior chords): nothing acts
        out, _ = _flux(m, m._run_cfg(m._overrides(vs)), galactic=gal)
        _W["std"] = out
    return _W["std"]


def _survival_exact(params, energies):
    """nu_3's survival averaged over the sources (scatterings off, nothing decays into nu_3) at the given energies."""
    import dp_absorption
    import dp_decay
    from dp_core import construct_PMNS
    from dp_cosmo import calc_R_source
    if "surv_w" not in _W:
        H0, Om, OL = params["H0_eV"], params["Omega_m"], params["Omega_L"]
        Hz = lambda z: H0 * np.sqrt(Om * (1 + z) ** 3 + OL)  # noqa: E731
        zl = np.concatenate([[0.0], np.geomspace(1e-7, float(params["z_max"]), 20000)])
        wl = np.array([calc_R_source(z, float(params["rho_evol"])) for z in zl]) * (1 + zl) ** (-float(params["p_index"])) / Hz(zl)
        f = 1.0 / ((1 + zl) ** 2 * Hz(zl))
        Il = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(zl))])
        _W["surv_w"] = (zl, wl, Il)
    zl, wl, Il = _W["surv_w"]
    U = construct_PMNS(params["th12"], params["th13"], params["th23"], params["d_cp"])
    m = np.asarray(params["m_nu_bg_arr"], dtype=np.float64)
    G = (np.asarray(params["G_coupling"], dtype=np.complex128) if params.get("G_coupling") is not None
         else dp_absorption.dark_portal_coupling(params["g_D"], np.array([params["u_e"], params["u_mu"], params["u_tau"]],
                                                                         complex), 0.0))
    Gam, _ = dp_decay.rest_rates(G, U, m, float(params["m_Zd_eV"]))
    r3 = Gam[2].sum() * m[2]
    if r3 <= 0.0:
        return None, U
    return {float(e): float(_trapz(wl * np.exp(-r3 / e * Il), zl) / _trapz(wl, zl)) for e in energies}, U


def _one(u):
    import dark_portal_solver as dps
    import dp_galactic_path as gp
    t_all = time.time()
    out = dict(u)
    times = {}
    try:
        cfg, m = _model(u["fam"])
        v = {q.name: q.start for q in cfg.parameters}
        v.update({u["x"]: u["xv"], u["y"]: u["yv"], "m_lightest_eV": u["ml"]})
        base = m._run_cfg(m._overrides(v))
        gal = bool(m.galactic)
        if "ylow" not in _W or _W.get("ylow_fam") != u["fam"]:
            _W.update(ylow=float(_axes(cfg)[3][0]), ylow_fam=u["fam"])
        std = _std(m, v, u["y"], _W["ylow"], gal)
        std_tot = std["F"].sum(-1).max(0)
        std_gal = std["G"].sum(-1).max(0) if gal else None

        # the Galactic path's calls (which path, pieces)
        if "gp_wrapped" not in _W:
            _W["split"] = [1]
            _W["used"] = []
            for name in ("sweep_populations", "vacuum_populations"):
                f = getattr(gp, name)

                def wrap(*a, _f=f, _n=name, **k):
                    _W["used"].append(_n)
                    return _f(*a, n_split=_W["split"][0], **k)
                setattr(gp, name, wrap)
            _W["gp_wrapped"] = True

        def run(key, solver_cfg=None, params_mod=None, galactic=False, ext=True):
            t = time.time()
            _W["used"].clear()
            r, params = _flux(m, solver_cfg if solver_cfg is not None else base, params_mod, galactic, ext)
            r["used"] = sorted(set(_W["used"]))
            times[key] = round(time.time() - t, 2)
            return r, params

        A, pA = run("A", galactic=gal)
        EA = A["E"]
        out.update(mode=A["mode"], health=_health(A["F"], std_tot), gal_path=A["used"])
        if gal:
            out["gal_health"] = _health(A["G"], std_gal)
        sub_keV = u["xv"] < dps.SI_COSMOLOGY_MIN_M_EV
        res = {}

        # B: time steps
        n0 = int(base["grid"]["n_z_bins"])
        cB = copy.deepcopy(base)
        cB["grid"]["n_z_bins"] = (n0 - 1) * 4 + 1
        pmB = dict(si=dict(z_substeps=2, E_extension=10.0)) if A["mode"] in ("si", "combined") else None
        if A["mode"] == "si":
            cB = None
        B, _ = run("B", solver_cfg=cB, params_mod=pmB)
        res["B"] = _cmp(A["F"], B["F"], EA, B["E"], std_tot)

        # C: energy grid (sub-grid units, and every unit whose run is the dark-portal phase alone)
        if u["sub"] or A["mode"] == "dark_portal":
            cC = copy.deepcopy(base)
            cC["grid"]["n_E_bins"] = 2 * (int(base["grid"]["n_E_bins"]) - 1) + 1
            C, _ = run("C", solver_cfg=cC)
            res["C"] = _cmp(A["F"], C["F"], EA, C["E"], std_tot)

        # H: the grid extended a decade above its top (115 more bins of the same width: the grids nest), what comes
        # down from above 30 PeV (decay daughters, the energy loss, secondaries) the default leaves out
        if u["sub"] or A["mode"] == "dark_portal":
            cH = copy.deepcopy(base)
            nE0 = int(base["grid"]["n_E_bins"])
            r = (float(base["grid"]["E_max_eV"]) / float(base["grid"]["E_min_eV"])) ** (1.0 / (nE0 - 1))
            cH["grid"]["E_max_eV"] = float(base["grid"]["E_max_eV"]) * r ** 115
            cH["grid"]["n_E_bins"] = nE0 + 115
            Hx, _ = run("H", solver_cfg=cH, galactic=gal and bool(A["used"]))
            res["H"] = _cmp(A["F"], Hx["F"], EA, Hx["E"], std_tot)
            if gal and A["used"]:
                res["GH"] = _cmp(A["G"], Hx["G"], EA, Hx["E"], std_gal)

        # D: the other transport below 1 keV, where both hold (dp_si has no potential: not where the potential acts,
        # 'combined' or the dark-portal phase with a potential)
        if sub_keV:
            s0, h0 = dps.SOFT_JUMP_MIN, dps.SI_HARD_MIN_G
            try:
                dps.SOFT_JUMP_MIN, dps.SI_HARD_MIN_G = 0.0, 0.0
                mode_E = dps._cosmology_mode(pA, EA)
            finally:
                dps.SOFT_JUMP_MIN, dps.SI_HARD_MIN_G = s0, h0
            pot_negligible = (mode_E == "si")
            others = {"dark_portal": ["si"] if pot_negligible else [], "si": ["dark_portal"],
                      "combined": ["dark_portal"]}[A["mode"]]
            Dres = {}
            for mode in others:
                D, _ = run("D_" + mode, params_mod=dict(cosmo_transport=mode))
                res["D_" + mode] = _cmp(A["F"], D["F"], EA, D["E"], std_tot)
                Dres[mode] = D

            # E: the shortcuts off, where that changes the run
            s0, h0 = dps.SOFT_JUMP_MIN, dps.SI_HARD_MIN_G
            try:
                dps.SOFT_JUMP_MIN, dps.SI_HARD_MIN_G = 0.0, 0.0
                if mode_E == "si" and "si" in Dres:
                    res["E"] = dict(res["D_si"], same_as="D_si")
                elif not (mode_E == "si" and A["mode"] == "si"):            # si alone: no shortcut acts
                    Ex, _ = run("E")
                    res["E"] = _cmp(A["F"], Ex["F"], EA, Ex["E"], std_tot)
                    res["E"]["mode"] = mode_E
            finally:
                dps.SOFT_JUMP_MIN, dps.SI_HARD_MIN_G = s0, h0

            # F: decay alone against the exact nu_3 survival
            kF = [int(np.argmin(np.abs(EA - e))) for e in E_CHECK]
            exact, U = _survival_exact(pA, [EA[k] for k in kF])
            if exact is not None and pot_negligible:
                F1, _ = run("F1", params_mod=dict(relic_scattering=False, neutrino_decay=True))
                F0, _ = run("F0", params_mod=dict(relic_scattering=False, neutrino_decay=False))
                inv = np.linalg.inv(np.abs(U) ** 2)
                dev = 0.0
                rows = {}
                for k in kF:
                    for s in range(F1["F"].shape[0]):
                        a1, a0 = (inv @ F1["F"][s, k])[2], (inv @ F0["F"][s, k])[2]
                        if a0 > 1e-6 * F0["F"][s, k].sum():
                            ratio = a1 / a0
                            rows[f"{EA[k]:.4e}/{s}"] = ratio
                            dev = max(dev, abs(ratio - exact[float(EA[k])]))
                res["F"] = dict(d=dev, exact={f"{e:.4e}": x for e, x in exact.items()}, code=rows)

        # Galactic: pieces x2; the sweep where the vacuum path was used; the energy grid x2
        if gal and A["used"] and u["sub"]:
            _W["split"][0] = 2
            try:
                GB, _ = run("GB", galactic=True, ext=False)
            finally:
                _W["split"][0] = 1
            res["GB"] = _cmp(A["G"], GB["G"], EA, GB["E"], std_gal)
            if "vacuum_populations" in A["used"] and u["ix"] % 4 == 0 and u["iy"] % 4 == 0 and u["ml"] == ML_FULL:
                vac0 = dps.GALACTIC_PATH_VACUUM
                try:
                    dps.GALACTIC_PATH_VACUUM = False
                    GC, _ = run("GC", galactic=True, ext=False)
                finally:
                    dps.GALACTIC_PATH_VACUUM = vac0
                res["GC"] = _cmp(A["G"], GC["G"], EA, GC["E"], std_gal)
            if u["sub"]:
                cG = copy.deepcopy(base)
                cG["grid"]["n_E_bins"] = 2 * (int(base["grid"]["n_E_bins"]) - 1) + 1
                GD, _ = run("GD", solver_cfg=cG, galactic=True, ext=False)
                res["GD"] = _cmp(A["G"], GD["G"], EA, GD["E"], std_gal)
        out.update(res=res, error=None)
    except Exception as e:
        import traceback
        out.update(error=repr(e)[:300], trace=traceback.format_exc()[-1500:])
    out["times"] = times
    out["seconds"] = round(time.time() - t_all, 2)
    return out


def run(manifest, task, out_dir, workers):
    share = json.load(open(manifest))["tasks"][task]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"validate.{task}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done.add((r["fam"], r["ix"], r["iy"], r["ml"]))
    share = [u for u in share if (u["fam"], u["ix"], u["iy"], u["ml"]) not in done]
    if not share:
        print("done (nothing left)", flush=True)
        return 0
    with open(path, "a") as fh, get_context("spawn").Pool(min(workers, len(share))) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, share, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (k + 1) % 25 == 0:
                print(f"{k + 1}/{len(share)}", flush=True)
    print("done", flush=True)
    return 0


def summarize(out_dir, fig):
    rows = [json.loads(line) for p in sorted(glob.glob(os.path.join(out_dir, "validate.*.jsonl"))) for line in open(p)]
    print(f"{len(rows)} units")
    errs = [r for r in rows if r.get("error")]
    print(f"errors: {len(errs)}")
    for r in errs[:10]:
        print("  ", r["fam"], r["ix"], r["iy"], r["ml"], r["error"])
    from analysis import exclusion as ex
    bnd = {fam: ex.not_allowed_boundary(os.path.join(CONFIGS, f"{fam}.json")) for fam in FAMILIES}
    for r in rows:
        sl, c = bnd[r["fam"]]
        r["not_allowed"] = bool(np.log10(r["yv"]) >= sl * np.log10(r["xv"]) + c)
    keys = ["B", "C", "H", "D_si", "D_dark_portal", "E", "F", "GB", "GC", "GD", "GH"]
    for k in keys:
        for meas in ("d", "d_shape", "d_flavor"):
            ds = [r["res"][k][meas] for r in rows if not r.get("error") and k in r.get("res", {})
                  and meas in r["res"][k] and not r["not_allowed"]]
            if not ds:
                continue
            d = np.array(ds)
            print(f"{k:14s} {meas:8s} n {len(d):6d}  median {np.median(d):.1e}  p99 {np.percentile(d, 99):.1e}  "
                  f"max {d.max():.1e}  > 1e-3: {int((d > 1e-3).sum())}  > 1e-2: {int((d > 1e-2).sum())}")
    print("(allowed region only; the not-allowed region's units are in the files)")
    bad = [r for r in rows if not r.get("error") and (not r["health"]["finite"] or r["health"]["neg"] < -1e-9)]
    print(f"non-finite or negative (A): {len(bad)}")
    t = np.array([r["seconds"] for r in rows])
    print(f"seconds per unit: median {np.median(t):.0f}, mean {t.mean():.0f}, max {t.max():.0f}; total {t.sum() / 3600:.1f} core-h")
    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, REPO)
        try:
            from analysis import figure_style
            figure_style.use()
        except Exception:
            pass
        show = ["B", "C", "H", "D_si", "E", "F", "GB", "GC"]
        fams = FAMILIES
        f, axs = plt.subplots(len(fams), len(show), figsize=(3.0 * len(show), 2.6 * len(fams)), squeeze=False)
        for i, fam in enumerate(fams):
            for j, k in enumerate(show):
                ax = axs[i, j]
                pts = [(r["xv"], r["yv"], r["res"][k]["d"]) for r in rows
                       if r["fam"] == fam and r["ml"] == ML_FULL and not r.get("error") and k in r.get("res", {})]
                if pts:
                    a = np.array(pts)
                    sc = ax.scatter(np.log10(a[:, 0]), np.log10(a[:, 1]), c=np.log10(np.maximum(a[:, 2], 1e-12)),
                                    vmin=-8, vmax=-1, s=6, cmap="viridis", marker="s")
                ax.set_title(f"{fam} {k}", fontsize=8)
                ax.tick_params(labelsize=6)
        f.colorbar(sc, ax=axs, shrink=0.6, label="log10 d")
        f.savefig(fig, dpi=110)
        print("figure:", fig)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("plan")
    a.add_argument("manifest")
    a.add_argument("--tasks", type=int, required=True)
    a.add_argument("--pilot", type=int, default=0)
    a.add_argument("--part", choices=("all", "threaded", "single"), default="all",
                   help="threaded: the per-pixel sweep checks' units (run with VT_THREADS and fewer workers)")
    a = sp.add_parser("run")
    a.add_argument("manifest")
    a.add_argument("task", type=int)
    a.add_argument("out_dir")
    a.add_argument("--workers", type=int, default=32)
    a = sp.add_parser("summarize")
    a.add_argument("out_dir")
    a.add_argument("figure", nargs="?", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        return plan(args.manifest, args.tasks, args.pilot, args.part)
    if args.cmd == "run":
        return run(args.manifest, args.task, args.out_dir, args.workers)
    return summarize(args.out_dir, args.figure)


if __name__ == "__main__":
    sys.exit(main())
