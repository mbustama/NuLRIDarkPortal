"""
validate_si_decay_back.py -- the secret interactions' decay back below 1 keV (2026-11-08; dp_mediator.SIMediator,
docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md item 22 (f)) on the production grid (the SI campaign's configurations,
analysis/configs/campaign_si/now_events, texture mu: 515 bins, 2 redshift substeps per bin, the relics' thermal
products), the Galactic component off, a pion source (1/3, 2/3, 0), one power law E^-2.735:

  absolute  the flux at Earth (all flavors) with / without the decay back for the four models at m_phi = 1e-2 eV,
            g = 1e-6, 3e-6, 1e-5 (and 1e-3 eV, g = 1e-6, 1e-5): on / off over 10 TeV - 30 PeV, the flavor fractions at
            100 TeV, the cost of each run;
  steps     z_substeps 4 against 2 (the step count doubled) at 1e-2 eV, g = 1e-6 and 1e-5: on, off and on / off;
  bins      the grid's bins doubled (1029) at 1e-2 eV, g = 1e-5: on, off and on / off;
  scheme    the scalar's transport (its step-average scheme with the mediator as one more population) against the
            exact bin solution of the same steps, with the decay back, 1e-2 eV, g = 1e-6 and 1e-5 (the collisions
            negligible there, so the two differ by the decay back's scheme alone);
  closure   the decays alone (dp_si's sweep, no collisions) on the production grid with its extension: a beam above
            1 PeV in every visible state, 400 steps over a Hubble time at z = 0, the decay back on and off: the energy
            of the beam, the live mediators and what fell below the grid against the start; the neutrinos' number
            gain against two per decayed mediator;
  cost      the transport's time off and on at 1e-2 eV, g = 1e-5, the step terms built, the column cache cleared;
  warm      every model on a small grid (numba's cache filled before parallel processes start).

  python tools/validate_si_decay_back.py <mode> <out.json> [model, e.g. scalar-dirac]
  python tools/validate_si_decay_back.py report <out dir>
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

MODELS = [("scalar", "majorana"), ("scalar", "dirac"), ("vector", "majorana"), ("vector", "dirac")]
SOURCE = np.array([1.0 / 3.0, 2.0 / 3.0, 0.0])
P_INDEX = 2.735
BAND = (1e13, 3e16)


def _setup(med, kind, vals, grid=None, si=None):
    import dark_portal_solver as dps
    from analysis.config import from_dict
    raw = json.load(open(os.path.join(REPO, "analysis", "configs", "campaign_si", "now_events",
                                      f"si_{med}_{kind}_mu_ul.json")))
    raw["solver"]["grid"].update(grid or {})
    raw["solver"]["galactic"]["enabled"] = False
    raw["solver"]["secret_interactions"].update(si or {})
    model = from_dict(raw, "").model()
    E, p = dps.build_solver_params(model._run_cfg(model._overrides(dict(vals))))
    p["show_progress"] = False
    p["p_index"] = P_INDEX
    p["inj_window_eV"] = (0.0, np.inf)
    return E, p, dps._flux_setup(E, p)


def _flux(E, p, setup, on):
    import dark_portal_solver as dps
    import dp_si
    p = dict(p, mediator_decay_back=bool(on))
    t0 = time.time()
    F_nu, F_nb = dps._flux_from_setup(E, p, setup, (0.5, 0.5), SOURCE)
    dt = time.time() - t0
    F = F_nu + F_nb
    diag = dp_si.MEDIATOR_LAST.get("diag")
    return F, dt, (None if diag is None else np.asarray(diag[0]).sum(axis=0).tolist())


def _summary(E, F_on, F_off):
    band = (E >= BAND[0]) & (E <= BAND[1])
    r = F_on.sum(1) / F_off.sum(1)
    i100 = int(np.argmin(np.abs(np.log(E / 1e14))))
    fr_on = F_on[i100] / F_on[i100].sum()
    fr_off = F_off[i100] / F_off[i100].sum()
    return dict(ratio_max=float(r[band].max()), ratio_min=float(r[band].min()), E_at_max=float(E[band][np.argmax(r[band])]),
                ratio_10TeV=float(np.interp(np.log(1e13), np.log(E), r)),
                ratio_100TeV=float(np.interp(np.log(1e14), np.log(E), r)),
                ratio_1PeV=float(np.interp(np.log(1e15), np.log(E), r)),
                flavor_shift_100TeV=float(np.max(np.abs(fr_on - fr_off))), ratio=r.tolist())


def absolute():
    out = []
    pts = [(1e-2, g) for g in (1e-6, 3e-6, 1e-5)] + [(1e-3, g) for g in (1e-6, 1e-5)]
    for med, kind in MODELS:
        for M, g in pts:
            E, p, setup = _setup(med, kind, dict(m_phi_eV=M, g_phi=g, m_lightest_eV=1e-6))
            F_off, t_off, _ = _flux(E, p, setup, False)
            F_on, t_on, diag = _flux(E, p, setup, True)
            sub = list(setup["si_subkev"].values())[-1]
            rec = dict(mode="absolute", model=f"{med}-{kind}", M=M, g=g, t_off=t_off, t_on=t_on, diag=diag,
                       mediator_path_exponent=sub["meta"].get("mediator_path_exponent"), E=E.tolist(),
                       finite=bool(np.all(np.isfinite(F_on))), **_summary(E, F_on, F_off))
            print(json.dumps({k: v for k, v in rec.items() if k not in ("ratio", "E", "diag")}), flush=True)
            out.append(rec)
    return out


def steps():
    out = []
    for med, kind in MODELS:
        for g in (1e-6, 1e-5):
            res = {}
            for q in (2, 4):
                E, p, setup = _setup(med, kind, dict(m_phi_eV=1e-2, g_phi=g, m_lightest_eV=1e-6), si=dict(z_substeps=q))
                F_off, t_off, _ = _flux(E, p, setup, False)
                F_on, t_on, _ = _flux(E, p, setup, True)
                res[q] = (F_on.sum(1), F_off.sum(1), t_on, t_off)
            band = (E >= BAND[0]) & (E <= BAND[1])
            on2, off2 = res[2][0], res[2][1]
            on4, off4 = res[4][0], res[4][1]
            rec = dict(mode="steps", model=f"{med}-{kind}", M=1e-2, g=g,
                       on=float(np.max(np.abs(on4 / on2 - 1)[band])), off=float(np.max(np.abs(off4 / off2 - 1)[band])),
                       ratio=float(np.max(np.abs((on4 / off4) / (on2 / off2) - 1)[band])),
                       t=[res[2][2], res[2][3], res[4][2], res[4][3]])
            print(json.dumps(rec), flush=True)
            out.append(rec)
    return out


def bins():
    out = []
    for med, kind in MODELS:
        res = {}
        for nb in (515, 1029):
            E, p, setup = _setup(med, kind, dict(m_phi_eV=1e-2, g_phi=1e-5, m_lightest_eV=1e-6), grid=dict(n_E_bins=nb))
            F_off, t_off, _ = _flux(E, p, setup, False)
            F_on, t_on, _ = _flux(E, p, setup, True)
            res[nb] = (E, F_on.sum(1), F_off.sum(1), t_on, t_off)
        E1, on1, off1 = res[515][:3]
        E2, on2, off2 = res[1029][:3]
        band = (E1 >= BAND[0]) & (E1 <= BAND[1])
        lin = lambda y: np.exp(np.interp(np.log(E1), np.log(E2), np.log(y)))   # noqa: E731
        rec = dict(mode="bins", model=f"{med}-{kind}", M=1e-2, g=1e-5,
                   on=float(np.max(np.abs(lin(on2) / on1 - 1)[band])), off=float(np.max(np.abs(lin(off2) / off1 - 1)[band])),
                   ratio=float(np.max(np.abs((lin(on2) / lin(off2)) / (on1 / off1) - 1)[band])),
                   t=[res[515][3], res[515][4], res[1029][3], res[1029][4]])
        print(json.dumps(rec), flush=True)
        out.append(rec)
    return out


def scheme():
    import dp_si
    orig = dp_si._si_sweep_cols_med
    out = []
    for kind in [k for m_, k in MODELS if m_ == "scalar"]:
        for g in (1e-6, 1e-5):
            E, p, setup = _setup("scalar", kind, dict(m_phi_eV=1e-2, g_phi=g, m_lightest_eV=1e-6))
            F_off, _, _ = _flux(E, p, setup, False)
            F_avg, _, _ = _flux(E, p, setup, True)
            dp_si._COLUMNS.clear()
            dp_si._COLUMNS_MED.clear()
            # the transport with the decay back calls _si_sweep_cols_med from Python: exact_bin (argument 19) forced
            dp_si._si_sweep_cols_med = lambda *a: orig(*a[:19], True, *a[20:])
            try:
                F_ex, _, _ = _flux(E, p, setup, True)
            finally:
                dp_si._si_sweep_cols_med = orig
                dp_si._COLUMNS.clear()
                dp_si._COLUMNS_MED.clear()
            band = (E >= BAND[0]) & (E <= BAND[1])
            a, x, o = F_avg.sum(1), F_ex.sum(1), F_off.sum(1)
            rec = dict(mode="scheme", model=f"scalar-{kind}", M=1e-2, g=g,
                       on=float(np.max(np.abs(a / x - 1)[band])),
                       of_the_effect=float(np.max(np.abs(a / x - 1)[band]) / np.max(np.abs(x / o - 1)[band])),
                       effect=float(np.max((x / o - 1)[band])))
            print(json.dumps(rec), flush=True)
            out.append(rec)
    return out


def closure():
    """The decays alone on the production grid (515 bins and the transport's extension, 10x above), dp_si's sweep with
    no collisions: 400 steps over a Hubble time at z = 0, a beam E^-2 above 1 PeV in every visible state."""
    import dp_mediator as dmed
    import dp_si
    import dp_si_light as dsl
    from dp_core import construct_PMNS
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from test_si_mediator import _kernel
    U = np.asarray(construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0)))
    E_bins = np.geomspace(994997970133.3439, 3e16, 515)
    dlnE = np.log(E_bins[1] / E_bins[0])
    E = E_bins[0] * np.exp(dlnE * np.arange(515 + int(np.ceil(np.log(10.0) / dlnE - 1e-9))))
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    N = len(E)
    out = []
    H0 = 1.4377e-33
    for med, kind in MODELS:
        maj = kind == "majorana"
        n = 3 if maj else 6
        for ml, M, g in ((1e-6, 1e-2, 1e-5), (1e-6, 1e-2, 1e-6), (0.01, 0.03, 1e-6)):
            m = np.sqrt(np.array([0.0, 7.537e-5, 2.511e-3]) + ml * ml)
            G = np.zeros((3, 3), complex)
            G[1, 1] = g
            S = dmed.SIMediator(G, U, m, M, E, med, maj)
            Gam = dsl.rest_rates(G, U, m, M, med, maj)
            arr = dsl.transport_arrays(Gam, dsl.daughter_spectra(G, U, m, M, med, maj), m, E, maj)
            F0 = np.zeros((N, n))
            F0[:, :3] = (E ** -2.0)[:, None]
            F0[E < 1e15] = 0.0
            Z0 = np.zeros((N, 3))
            e0 = (F0.sum(1) * E * dE).sum()
            n0 = (F0.sum(1) * dE).sum()
            rec = dict(mode="closure", model=f"{med}-{kind}", ml=ml, M=M, g=g)
            for on in (True, False):
                tabs = S.arrays if on else (S.arrays[0], S.arrays[1], S.arrays[2], S.arrays[3], S.arrays[4],
                                            S.arrays[5], 0.0)
                t0 = time.time()
                F, Z, d = _kernel(E, F0, Z0, 1.0 / H0, tabs, arr, med == "vector", n, parts=400)
                e1 = ((F.sum(1) + Z.sum(1)) * E * dE).sum()
                low = d[dmed.D_PROD_LOW_E] + d[dmed.D_DAU_LOW_E]
                key = "on" if on else "off"
                rec[key] = dict(beam_and_mediators=float(e1 / e0), below_grid=float(low / e0),
                                closure=float((e1 + low) / e0 - 1.0), beam=float((F.sum(1) * E * dE).sum() / e0),
                                number_gain_over_2dec=(float(((F.sum(1) * dE).sum() - n0 + d[dmed.D_DAU_LOW])
                                                             / (2.0 * d[dmed.D_DEC])) if d[dmed.D_DEC] > 0 else None),
                                seconds=time.time() - t0)
            print(json.dumps(rec), flush=True)
            out.append(rec)
    return out


def cost():
    """The decay back's cost on the production grid: at 1e-2 eV, g = 1e-5, the transport off and on, each after the
    point's step terms are built (a first run off) and with dp_si's column cache cleared: three repeats each."""
    import dp_si
    out = []
    for med, kind in MODELS:
        E, p, setup = _setup(med, kind, dict(m_phi_eV=1e-2, g_phi=1e-5, m_lightest_eV=1e-6))
        _flux(E, p, setup, False)
        t = {False: [], True: []}
        for _ in range(3):
            for on in (False, True):
                dp_si._COLUMNS.clear()
                dp_si._COLUMNS_MED.clear()
                t[on].append(_flux(E, p, setup, on)[1])
        rec = dict(mode="cost", model=f"{med}-{kind}", t_off=min(t[False]), t_on=min(t[True]),
                   ratio=min(t[True]) / min(t[False]))
        print(json.dumps(rec), flush=True)
        out.append(rec)
    return out


def warm():
    """Every model on a small grid, on and off: numba's cache filled before the parallel processes start."""
    out = []
    for med, kind in MODELS:
        E, p, setup = _setup(med, kind, dict(m_phi_eV=1e-2, g_phi=1e-5, m_lightest_eV=1e-6), grid=dict(n_E_bins=24))
        for on in (False, True):
            F, dt, _ = _flux(E, p, setup, on)
            out.append(dict(mode="warm", model=f"{med}-{kind}", on=on, seconds=dt, finite=bool(np.all(np.isfinite(F)))))
            print(json.dumps(out[-1]), flush=True)
    return out


def report(out_dir):
    import glob
    for f in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        for rec in json.load(open(f)):
            print(json.dumps({k: v for k, v in rec.items() if k not in ("ratio", "E", "diag")
                              or not isinstance(v, list)}))


if __name__ == "__main__":
    mode = sys.argv[1]
    if len(sys.argv) > 3:
        MODELS[:] = [tuple(sys.argv[3].split("-"))]
    if mode == "report":
        report(sys.argv[2])
    else:
        res = {"absolute": absolute, "steps": steps, "bins": bins, "scheme": scheme, "closure": closure,
               "warm": warm, "cost": cost}[mode]()
        json.dump(res, open(sys.argv[2], "w"))
