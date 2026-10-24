"""The jump-decay cycles' convergence in the cosmological transport (2026-10-24): at points where the soft scatterings
and the decays compete, the extragalactic flux over the band (1e13-3e15 eV) and the flavor at 100 TeV, with the
dark-portal phase at 1, 4 and 16 times its redshift steps, against dp_si (the decays inside its column sweep, no
splitting) at 1 and 4 substeps; and the nu_3 survival against the exact source average (scatterings off). Run on
Tycho; arguments: the points' indices (default all)."""
import copy
import json
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "src"), os.path.join(os.path.dirname(__file__), "..")]
os.environ.pop("NULRI_FLUX_CACHE", None)
import numpy as np  # noqa: E402
import dark_portal_solver as dps  # noqa: E402
from analysis import config as config_mod  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
POINTS = [("dp_mu_ul", 1e-6, 1e-7), ("dp_mu_ul", 1e-6, 1e-6), ("dp_mu_ul", 1e-4, 1e-6), ("le_lmu_ul", 1e-4, 1e-6),
          ("dp_e_ul", 1e-5, 1e-7), ("survival", 1e-2, 10 ** -7.5)]
SOURCE = np.array([1.0, 2.0, 0.0]) / 3.0


def setup(cfgname, x, y):
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
    pr["n_cores"] = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    return np.asarray(E), pr


def flux(E, pr):
    return np.asarray(dps.propagate_weighted(E, pr, sources=[SOURCE])[0])


def survival(x, y):
    import dp_absorption
    import dp_decay
    from dp_core import construct_PMNS
    from dp_cosmo import calc_R_source
    E, pr = setup("dp_mu_ul", x, y)
    pr["cosmo_transport"] = "dark_portal"
    H0, Om, OL = pr["H0_eV"], pr["Omega_m"], pr["Omega_L"]
    k = int(np.argmin(abs(E - 1e14)))
    U = construct_PMNS(pr["th12"], pr["th13"], pr["th23"], pr["d_cp"])
    inv = np.linalg.inv(np.abs(U) ** 2)
    m3 = pr["m_nu_bg_arr"][2]
    G = dp_absorption.dark_portal_coupling(pr["g_D"], np.array([pr["u_e"], pr["u_mu"], pr["u_tau"]], complex), 0.0)
    Gam, _ = dp_decay.rest_rates(G, U, pr["m_nu_bg_arr"], x)
    Hz = lambda z: H0 * np.sqrt(Om * (1 + z) ** 3 + OL)  # noqa: E731
    zl = np.concatenate([[0.0], np.geomspace(1e-7, float(pr["z_max"]), 20000)])
    wl = np.array([calc_R_source(z, 0.0) for z in zl]) * (1 + zl) ** (-2.735) / Hz(zl)
    Il = np.concatenate([[0], np.cumsum(0.5 * (1 / ((1 + zl[1:]) ** 2 * Hz(zl[1:])) + 1 / ((1 + zl[:-1]) ** 2 * Hz(zl[:-1])))
                                        * np.diff(zl))])
    S3 = _trapz(wl * np.exp(-Gam[2].sum() * m3 / E[k] * Il), zl) / _trapz(wl, zl)
    for nz in (1, 4):
        q = dict(pr, n_z_bins=(int(pr["n_z_bins"]) - 1) * nz + 1)
        F0 = flux(E, dict(q, neutrino_decay=False, relic_scattering=False))
        F1 = flux(E, dict(q, neutrino_decay=True, relic_scattering=False))
        print(f"survival m {x:g} g {y:g}, steps x{nz}: nu_3 share ratio {(inv @ F1[k])[2] / (inv @ F0[k])[2]:.4f}, "
              f"exact {S3:.4f}", flush=True)
    for sub in (1,):
        q = dict(pr, cosmo_transport="si", si=dict(z_substeps=sub, E_extension=10.0))
        F0 = flux(E, dict(q, neutrino_decay=False, relic_scattering=False))
        F1 = flux(E, dict(q, neutrino_decay=True, relic_scattering=False))
        print(f"survival dp_si: nu_3 share ratio {(inv @ F1[k])[2] / (inv @ F0[k])[2]:.4f}", flush=True)


sel = [int(a) for a in sys.argv[1:]] or range(len(POINTS))
for cfgname, x, y in [POINTS[k] for k in sel]:
    if cfgname == "survival":
        survival(x, y)
        continue
    E, pr = setup(cfgname, x, y)
    band = (E > 1e13) & (E < 3e15)
    k = int(np.argmin(np.abs(E - 1e14)))
    n0 = int(pr["n_z_bins"])
    runs = [("dark_portal", dict(n_z_bins=(n0 - 1) * 16 + 1)), ("dark_portal", dict(n_z_bins=(n0 - 1) * 4 + 1)),
            ("dark_portal", dict(n_z_bins=n0)), ("si", dict(si=dict(z_substeps=4, E_extension=10.0))),
            ("si", dict(si=dict(z_substeps=1, E_extension=10.0)))]
    ref = None
    for mode, extra in runs:
        t = time.time()
        F = flux(E, dict(pr, cosmo_transport=mode, **extra))
        tot = F.sum(axis=1)
        line = (f"{cfgname} m {x:g} g {y:g} {mode} {extra}: {time.time() - t:.1f} s; band mean {tot[band].mean():.4e}, "
                f"flavor {np.round(F[k] / F[k].sum(), 4)}")
        if ref is None:
            ref = tot
        else:
            r = tot[band] / np.where(ref[band] > 0, ref[band], 1.0)
            line += f"; vs x16: band ratio {r.min():.4f}-{r.max():.4f}"
        print(line, flush=True)
