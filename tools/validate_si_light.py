"""
validate_si_light.py -- the secret interactions below 1 keV (2026-10-25; notes, "Secret interactions below 1 keV"):
Tycho checks of dp_si with the processes of src/dp_si_light.py.

  steps      convergence in redshift steps (z_substeps 1, 2, 4 against 8) and energy bins (the grid's 439 bins against
             877, nested) at benchmark points of the four models (decays, conversions, hard scatterings, near the
             lines where the models stop holding): the spectrum's shape (the ratio to the standard flux with its median
             divided out) and the largest change of a flavor fraction, over 1e13-1e16 eV where the flux is at least
             1e-3 of the standard one
  limits     the no-coupling limit (free streaming), decays alone keeping the number (a grid down to 1 GeV), and the
             sub-keV inputs switched off reproducing dp_si's earlier runs bit for bit
  unchanged  1 keV and above: dp_si as before the 2026-10-25 edits (src/dp_si.py with them undone, loaded from a
             scratch copy) against the current one, bit for bit, the four models
  crosscheck the vector with Majorana neutrinos, texture mu, against the dark-portal model at eps' = 0 (its coupling
             g_D u u^dag with |u_mu|^2 = 0.5 is the same vector at g = g_D / 2 = g_D_trace): the cosmological phase of the two
             transports (dark_portal_solver.propagate_sky_basis allsky, extragalactic) at points across the plane
  galactic   the Galactic lines of sight: halved distance bins, the SVD's tolerance (1e-10 against 1e-13)
  python tools/validate_si_light.py <check> [--out output/analysis/si_light_<check>.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

import dp_si  # noqa: E402
import dp_si_light as L  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

D_LOG = 3.477121254719662 / 399
E_LO, N_E = 10 ** (13.0 - 39 * D_LOG), 439              # the campaign's now_ic26 grid (4.57 TeV - 30 PeV)
TH = (33.75878818411903, 8.623062290036087, 43.280093616242404, 212.0)
U = construct_PMNS(*np.radians(TH))
DM21, DM31 = 7.537e-5, 2.511e-3
H0 = 67.4e3 / 3.0856775814913673e22 * 6.582119569e-16
OM, OL = 0.315, 0.685
T0 = 1.676389e-4
P_INDEX = 2.735
SRC = np.array([1 / 3, 2 / 3, 0.0])

# (mediator, majorana, texture, M eV, g, m_lightest, what)
POINTS = [
    ("vector", True, "mu", 1e-6, 1e-12, 1e-6, "decays, g/M = 1e-6"),
    ("vector", True, "mu", 1e-12, 1e-17, 1e-6, "decays, g/M = 1e-5"),
    ("vector", True, "mu", 1e-3, 3e-8, 0.1, "decays, degenerate masses"),
    ("vector", True, "mu", 1e-6, 3e-5, 1e-6, "decays + conversions + loss, g/M = 30"),
    ("vector", True, "mu", 1.0, 2e-5, 1e-6, "conversions above the splittings"),
    ("vector", True, "mu", 100.0, 1e-3, 1e-6, "conversions + hard"),
    ("vector", False, "mu", 1e-6, 1e-12, 1e-6, "decays, g/M = 1e-6"),
    ("vector", False, "mu", 1e-3, 3e-8, 0.1, "decays, degenerate masses"),
    ("vector", False, "mu", 1e-6, 3e-5, 1e-6, "decays + conversions + loss"),
    ("vector", False, "mu", 1.0, 2e-5, 1e-6, "conversions above the splittings"),
    ("scalar", True, "mu", 1e-6, 5e-8, 1e-6, "decays"),
    ("scalar", True, "mu", 1e-3, 3e-7, 0.1, "decays, degenerate masses"),
    ("scalar", True, "mu", 1e-9, 1e-7, 1e-6, "decays, relics decayed"),
    ("scalar", True, "mu", 10.0, 5e-3, 1e-6, "hard scatterings"),
    ("scalar", False, "mu", 1e-6, 5e-8, 1e-6, "decays (sterile daughters)"),
    ("scalar", False, "mu", 1e-3, 3e-7, 0.1, "decays, degenerate masses"),
    ("scalar", False, "mu", 10.0, 5e-3, 1e-6, "hard scatterings"),
    ("scalar", True, "universal", 1e-4, 1e-6, 1e-6, "universal: decays through U^T U"),
]


def masses(ml):
    return np.array([ml, np.sqrt(ml * ml + DM21), np.sqrt(ml * ml + DM31)])


def grid(n=N_E):
    return np.geomspace(E_LO, 3e16, n)


def run(point, E, z_substeps=2, inputs=True, thermal="full"):
    """dp_si's flux (mass states, both kinds) at a benchmark point, with the sub-keV inputs (dp_si_light)."""
    med, maj, tex, M, g, ml, _ = point
    m = masses(ml)
    gf = dp_si.flavor_couplings(tex, g, None)
    gm = dp_si.mass_basis_couplings_model(gf, U, med, maj)
    eta = (np.abs(U) ** 2).T @ np.ones(3)
    kw = {}
    M_run = M
    if inputs:
        inp = L.subkev_inputs(gf, U, m, M, med, maj, T0, eta, H0, OM, OL, 4.0, E[0])
        kw = dict(decay=inp["decay"], relic_loss=inp["relic_loss"])
        M_run, eta = inp["M_coll"], inp["eta"]
    return dp_si.propagate_si(E, 4.0, H0, OM, OL, U, m, T0, eta, SRC, P_INDEX, 0.0, 1.0, M_run, gm, majorana=maj,
                              mediator=med, thermal=thermal, z_substeps=z_substeps, **kw)


def metrics(F, Fref, F0):
    """(shape, flavor): the largest relative difference of the spectra's shapes (ratio to the standard flux F0, its
    median divided out) and of a flavor fraction, over 1e13-1e16 eV where the reference keeps >= 1e-3 of F0."""
    A = np.abs(U) ** 2
    fl, flr = F @ A.T, Fref @ A.T
    tot, totr, tot0 = fl.sum(1), flr.sum(1), F0.sum(1)
    E = grid(len(F))
    sel = (E >= 1e13) & (E <= 1e16) & (totr >= 1e-3 * tot0)
    if not np.any(sel):
        return None, None, 0.0
    r, rr = tot[sel] / tot0[sel], totr[sel] / tot0[sel]
    shape = float(np.max(np.abs((r / np.median(r)) / (rr / np.median(rr)) - 1.0)))
    flav = float(np.max(np.abs(fl[sel] / tot[sel, None] - flr[sel] / totr[sel, None])))
    return shape, flav, float(np.median(rr))


def check_steps():
    out = []
    E1, E2 = grid(), grid(2 * N_E - 1)
    F0 = dp_si.free_streaming_flux(E1, 4.0, H0, OM, OL, U, SRC, P_INDEX, 0.0, 1.0)
    for pt in POINTS:
        t0 = time.time()
        ref = run(pt, E1, z_substeps=8)
        row = dict(point=list(map(str, pt)))
        for q in (1, 2, 4):
            s, f, med = metrics(run(pt, E1, z_substeps=q), ref, F0)
            row[f"steps_{q}"] = dict(shape=s, flavor=f)
        row["flux_median_over_standard"] = med
        # the fine grid at one substep has the coarse grid's redshift steps at two: the bins alone change
        fine = run(pt, E2, z_substeps=1)[::2]
        s, f, _ = metrics(run(pt, E1, z_substeps=2), fine, F0)
        row["bins_877"] = dict(shape=s, flavor=f)
        row["seconds"] = time.time() - t0
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


def check_limits():
    out = {}
    E = grid()
    pt = ("vector", True, "mu", 1e-6, 0.0, 1e-6, "")
    F = run(pt, E)
    F0 = dp_si.free_streaming_flux(E, 4.0, H0, OM, OL, U, SRC, P_INDEX, 0.0, 1.0)
    out["no_coupling_vs_free_streaming"] = float(np.max(np.abs(F - F0) / F0))
    # decays alone keep the number: a grid down to 1 GeV, the collisions negligible (g/M = 1e-5 at 1e-12 eV)
    Ed = np.geomspace(1e9, 3e16, 1300)
    for med, maj in (("vector", True), ("vector", False), ("scalar", True), ("scalar", False)):
        g = 1e-17 if med == "vector" else 1e-7
        gf = dp_si.flavor_couplings("mu", g, None)
        m = masses(1e-6)
        inp = L.subkev_inputs(gf, U, m, 1e-12, med, maj, T0, (np.abs(U) ** 2).T @ np.ones(3), H0, OM, OL, 4.0, Ed[0])
        gm = dp_si.mass_basis_couplings_model(gf, U, med, maj)
        # injected above 1e14 eV only, so that the daughters (and theirs) stay far above the grid's 1 GeV bottom
        _, info = dp_si.propagate_si(Ed, 4.0, H0, OM, OL, U, m, T0, inp["eta"], SRC, P_INDEX, 0.0, 1.0,
                                     inp["M_coll"], gm, majorana=maj, mediator=med, thermal="full", z_substeps=2,
                                     decay=inp["decay"], return_diagnostics=True, inj_window=(1e14, np.inf))
        out[f"number_{med}_{'maj' if maj else 'dir'}"] = dict(
            final_over_injected=info["final_number"] / info["injected"],
            visible_over_injected=info["visible_number"] / info["injected"],
            note="final: every state (Dirac: the sterile ones too); injected above 1e14 eV, grid down to 1 GeV")
    print(json.dumps(out), flush=True)
    return out


def _old_dp_si():
    """src/dp_si.py with the 2026-10-25 edits undone (the vector pair spectrum's floor, the six-state Dirac spec, the
    decay arrays through dp_si_light), loaded as a module from a scratch copy."""
    import importlib.util
    src = open(os.path.join(REPO, "src", "dp_si.py")).read()
    a = src.index("        if x < VV_X_FLOOR:")
    b = src.index("        t1[k] = gT / x\n")
    src = src[:a] + src[b:]
    src = src.replace(",\n                      sterile_states=decay is not None and not majorana)", ")")
    path = os.path.join(REPO, "output", "scratch_dp_si_old.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("dp_si_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_unchanged():
    old = _old_dp_si()
    E = grid()
    out = {}
    for med, maj in (("vector", True), ("vector", False), ("scalar", True), ("scalar", False)):
        for M, g in ((1e3, 4e-4), (1e5, 5e-3), (5e5, 3e-5), (1e7, 0.1)):
            gf = dp_si.flavor_couplings("mu", g, None)
            gm = dp_si.mass_basis_couplings_model(gf, U, med, maj)
            m = masses(1e-6)
            eta = (np.abs(U) ** 2).T @ np.ones(3)
            args = (E, 4.0, H0, OM, OL, U, m, T0, eta, SRC, P_INDEX, 0.0, 1.0, M, gm)
            kw = dict(majorana=maj, mediator=med, thermal="full")
            a = dp_si.propagate_si(*args, **kw)
            b = old.propagate_si(*args, **kw)
            out[f"{med}_{'maj' if maj else 'dir'}_M{M:g}_g{g:g}"] = dict(bit_for_bit=bool(np.array_equal(a, b)),
                                                                         max_rel=float(np.max(np.abs(a - b) / b)))
    print(json.dumps(out), flush=True)
    return out


def check_crosscheck():
    """The vector, Majorana, texture mu through the SI route against the dark-portal model at eps' = 0."""
    from analysis.config import load
    import dark_portal_solver as dps
    out = []
    path = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26", "dp_mu_ul.json")
    cfg = load(path)
    model = cfg.model()
    pts = ((1e-6, 1e-12, 1e-6), (1e-3, 3e-8, 0.1), (1.0, 2e-5, 1e-6), (1e-6, 3e-6, 1e-6), (1e-9, 1e-14, 1e-6),
           (100.0, 1e-3, 1e-6))
    if os.environ.get("CROSS_DRAIN"):
        # where the conversion-decay cycles drain the flux: the two transports' absolute fluxes along g/M at 1e-6 eV
        pts = tuple((1e-6, gm * 1e-6, 1e-6) for gm in (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0))
    for M, g, ml in pts:
        v = {q.name: q.start for q in cfg.parameters}
        # the plane's axis g_D Sum|U|^2 (u = (0, sqrt(1/2), 0)): G = g_D u u^dag = (g_D / 2) diag(0, 1, 0), the SI g
        v.update(m_Zd_eV=M, g_D_trace=g, m_lightest_eV=ml)
        E, pr = dps.build_solver_params(model._run_cfg(model._overrides(v)))
        pr["galactic"] = dict(pr.get("galactic", {}), enabled=False)
        t0 = time.time()
        sky = dps.propagate_sky_basis(E, pr, sources=(tuple(SRC),), galactic=False)
        t_dp = time.time() - t0
        F_dp = np.tensordot(sky["band_omega"] / sky["band_omega"].sum(), sky["bands_nu"][0] + sky["bands_nubar"][0],
                            axes=(0, 0))
        F_si = run(("vector", True, "mu", M, g, ml, ""), np.asarray(E), z_substeps=2)
        A = np.abs(U) ** 2
        fl = F_si @ A.T
        sel = (E >= 1e13) & (E <= 1e16)
        r = F_dp[sel].sum(1) / fl[sel].sum(1)
        F0 = dp_si.free_streaming_flux(np.asarray(E), 4.0, H0, OM, OL, U, SRC, P_INDEX, 0.0, 1.0) @ A.T
        row = dict(M=M, g=g, m_lightest=ml, norm_ratio_median=float(np.median(r)),
                   si_over_standard_median=float(np.median(fl[sel].sum(1) / F0[sel].sum(1))),
                   shape=float(np.max(np.abs(r / np.median(r) - 1.0))),
                   flavor=float(np.max(np.abs(F_dp[sel] / F_dp[sel].sum(1, keepdims=True)
                                              - fl[sel] / fl[sel].sum(1, keepdims=True)))),
                   seconds_dark_portal=t_dp, route=dps._cosmology_mode(pr, np.asarray(E)))
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


def check_galactic():
    import dp_galactic
    dp_galactic.use_data_dir(None)
    E = np.geomspace(E_LO, 1e17, 470)
    comps = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    out = []
    for med, maj, M, g in (("vector", True, 1e-3, 1e-5), ("vector", False, 1e-6, 1e-9), ("scalar", True, 1e-3, 1e-4),
                           ("scalar", False, 1e-3, 1e-4)):
        gf = dp_si.flavor_couplings("mu", g, None)
        m = masses(1e-6)
        r = L.galactic_rates(E, gf, U, m, M, med, maj, T0, np.ones(3), H0, OM, OL)
        t0 = time.time()
        d = {}
        a = L.galactic_transfer(E, U, r, "cringe", 8, comps, diag=d)
        sec = time.time() - t0
        # the SVD's tolerance
        tol = L.GAL_SVD_TOL
        L.GAL_SVD_TOL = 1e-13
        L._SHAPES.clear()
        b = L.galactic_transfer(E, U, r, "cringe", 8, comps)
        L.GAL_SVD_TOL = tol
        L._SHAPES.clear()
        # halved distance bins: each segment twice at half length (the same emission per unit length)
        seg = L._segment

        def halves(F, E_, dE_, S, Ls, *rest):
            seg(F, E_, dE_, S, 0.5 * Ls, *rest)
            seg(F, E_, dE_, S, 0.5 * Ls, *rest)
        L._segment = halves
        c = L.galactic_transfer(E, U, r, "cringe", 8, comps)
        L._segment = seg
        sc = max(float(np.max(np.abs(x))) for x in a[1:])
        row = dict(model=f"{med}_{'maj' if maj else 'dir'}", M=M, g=g, meta=r["meta"], rank=d.get("rank"),
                   svd_1e13=float(max(np.max(np.abs(x - y)) for x, y in zip(a, b)) / sc),
                   halved_segments=float(max(np.max(np.abs(x - y)) for x, y in zip(a, c)) / sc), seconds=sec)
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=("steps", "limits", "unchanged", "crosscheck", "galactic"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = globals()["check_" + a.check]()
    path = a.out or os.path.join(REPO, "output", "analysis", f"si_light_{a.check}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
