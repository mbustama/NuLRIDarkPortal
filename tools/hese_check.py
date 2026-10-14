"""
hese_check.py -- plan step 8 (docs/dev/DIRECTIONS_PLAN.md): the per-direction output as a consumer uses it.
IceCube's HESE 7.5-yr Monte Carlo (github.com/icecube/HESE-7-year-data-release; PRD 104, 022002) gives each
simulated event weightOverFluxOverLivetime (GeV sr cm^2), so its expected rate under a flux is

    N = livetime x sum_i weightOverFluxOverLivetime_i x Phi_{type_i}(E_i, dec_i),

with Phi per particle type (nu_e, nubar_e, ... : PDG codes +-12, +-14, +-16) at the true energy and the true
direction (South Pole: sin(dec) = -cos(zenith)). The release's cuts: 60 TeV <= deposited energy <= 10 PeV,
double cascades 10-1000 m long; livetime 227708167.68 s. Its astrophysical flux per type is
astro_norm x 1e-18 / 6 x (E / 100 TeV)^-gamma (weighter.py), best fit 6.37 and 2.87.

Cases (the expected astrophysical events per morphology: cascades, tracks, double cascades):
  A  the release's flux, written down here (the reference);
  B  propagate_sky with sources (1/3, 1/3, 1/3) -- 1:1:1 at the Earth for any mixing, the averaged oscillation
     matrix being doubly stochastic -- p = 2.87 and sky.extragalactic_norm = 6.37e-18 / 3, no new physics:
     the units and conventions test (per event B / A must be the extragalactic normalization's redshift-grid
     offset, the same for every event);
  C  pion sources (1/3, 2/3, 0): the flavor composition at the Earth, (0.30, 0.36, 0.34);
  D  C plus the Galactic component (CRINGE x 0.91, emission points in 3-D, averaged over right ascension per
     declination band: the MC has no azimuth);
  E  a dark-portal point (m_Zd = 1e-24 eV, kappa = 300) with the Galactic component.
With --toolkit-weights FILE (written by tools/hese_toolkit_weights.py with the release's own code, run in its
own environment), case A is also compared event by event with the release's weights.

  python -I tools/hese_check.py --hese /lustre/astro/mbustama/external/hese_7yr/HESE-7-year-data-release/HESE-7-year-data-release

Writes output/sky/hese_check.json and output/figures/hese_check.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dark_portal_solver as dps  # noqa: E402

LIVETIME_S = 227708167.68
BEST = dict(astro_norm=6.37, gamma=2.87)
MORPH = ("cascades", "tracks", "double cascades")
GU2_PER_KAPPA = 1.66e-54


def load_mc(folder, emin=60.0e3, emax=1.0e7, lmin=10.0, lmax=1000.0):
    """The release's MC with its cuts (data_loader.load_mc), as a dict of arrays."""
    d = {}
    for f in ("HESE_mc_observable.json", "HESE_mc_flux.json", "HESE_mc_truth.json"):
        with open(os.path.join(folder, "resources", "data", f)) as fh:
            d.update(json.load(fh))
    d = {k: np.asarray(v) for k, v in d.items()}
    with np.errstate(invalid="ignore"):
        keep = ((d["recoDepositedEnergy"] >= emin) & (d["recoDepositedEnergy"] <= emax)
                & ((d["recoMorphology"] != 2) | ((d["recoLength"] >= lmin) & (d["recoLength"] <= lmax))))
    out = {k: v[keep] for k, v in d.items()}
    out["n_all"] = int(len(keep))
    return out


def per_type_flux(mc, E_eV, bands_nu, bands_nb, dec_edges):
    """Phi (GeV^-1 cm^-2 s^-1 sr^-1) of each event's particle type at its true energy and declination, from
    per-band fluxes (N_b, N_E, 3) per flavor and kind (log-log in energy between the grid's nodes)."""
    t = mc["primaryType"]
    nu = np.abs(t)
    is_nu = np.isin(nu, (12, 14, 16))
    flavor = np.where(nu == 12, 0, np.where(nu == 14, 1, 2))
    dec = np.arcsin(np.clip(-np.cos(mc["primaryZenith"]), -1.0, 1.0))
    band = np.clip(np.searchsorted(dec_edges, dec, side="right") - 1, 0, len(dec_edges) - 2)
    lE, lx = np.log(E_eV / 1e9), np.log(mc["primaryEnergy"])
    if lx[is_nu].min() < lE[0] or lx[is_nu].max() > lE[-1]:
        raise ValueError("the flux's energy grid must cover the MC's true energies")
    k = np.clip(np.searchsorted(lE, lx) - 1, 0, len(lE) - 2)
    w = (lx - lE[k]) / (lE[k + 1] - lE[k])
    phi = np.zeros(len(t))
    for src, sel in ((bands_nu, is_nu & (t > 0)), (bands_nb, is_nu & (t < 0))):
        a = np.log(np.maximum(src[band[sel], k[sel], flavor[sel]], 1e-300))
        b = np.log(np.maximum(src[band[sel], k[sel] + 1, flavor[sel]], 1e-300))
        phi[sel] = np.exp((1.0 - w[sel]) * a + w[sel] * b)
    return phi


def counts(mc, phi):
    w = LIVETIME_S * mc["weightOverFluxOverLivetime"] * phi
    return {m: float(w[mc["recoMorphology"] == i].sum()) for i, m in enumerate(MORPH)}, w


def sky(dp=None, f_S=(1 / 3, 1 / 3, 1 / 3), galactic=False, n_E=100):
    cfg = {"dark_portal": dp or {"u_e": 0.0, "u_mu": 0.0, "u_tau": 0.0, "g_D": 1e-30},
           "source": {"p_index": BEST["gamma"], "f_e_S": f_S[0], "f_mu_S": f_S[1], "f_tau_S": f_S[2]},
           "grid": {"n_E_bins": n_E, "E_min_eV": 1e13, "E_max_eV": 1e17, "n_z_bins": 400},
           "sky": {"output": "bands", "nside": 8, "extragalactic_norm": BEST["astro_norm"] * 1e-18 / 3.0},
           "galactic": {"enabled": galactic, "emission": "3d"},
           "execution": {"n_cores": 1, "verbose": 0}}
    E, params = dps.build_solver_params(cfg)
    params["show_progress"] = False
    out = dps.propagate_sky(E, params)
    return E, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hese", required=True, help="the release's folder with resources/data/*.json")
    ap.add_argument("--toolkit-weights", help="npz from tools/hese_toolkit_weights.py (the release's own weights)")
    a = ap.parse_args()
    t0 = time.time()
    mc = load_mc(a.hese)
    res = dict(release=a.hese, livetime_s=LIVETIME_S, n_events_after_cuts=int(len(mc["primaryType"])),
               n_events_total=mc["n_all"], best_fit=BEST)
    E_grid = np.geomspace(1e3, 1e9, 400)
    flat = np.broadcast_to((BEST["astro_norm"] * 1e-18 / 6.0 * (E_grid / 1e5) ** -BEST["gamma"])[None, :, None],
                           (1, len(E_grid), 3))
    phiA = per_type_flux(mc, E_grid * 1e9, flat, flat, np.array([-np.pi / 2, np.pi / 2]))
    nA, wA = counts(mc, phiA)
    res["A_release_flux"] = nA
    E, outB = sky()
    phiB = per_type_flux(mc, E, outB["bands_nu"], outB["bands_nubar"], outB["dec_band_edges"])
    nB, wB = counts(mc, phiB)
    nu = np.isin(np.abs(mc["primaryType"]), (12, 14, 16)) & (mc["weightOverFluxOverLivetime"] > 0)
    r = phiB[nu] / phiA[nu]
    res["B_propagate_sky_111"] = dict(counts=nB, per_event_ratio_min=float(r.min()), per_event_ratio_max=float(r.max()),
                                      extragalactic_scale=float(outB["meta"]["extragalactic_scale"]))
    E, outC = sky(f_S=(1 / 3, 2 / 3, 0.0))
    phiC = per_type_flux(mc, E, outC["bands_nu"], outC["bands_nubar"], outC["dec_band_edges"])
    res["C_pion_sources"] = counts(mc, phiC)[0]
    E, outD = sky(f_S=(1 / 3, 2 / 3, 0.0), galactic=True)
    phiG = per_type_flux(mc, E, outD["galactic_bands_nu"], outD["galactic_bands_nubar"], outD["dec_band_edges"])
    res["D_galactic_only"] = counts(mc, phiG)[0]
    dp = {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "g_D": 300 * GU2_PER_KAPPA / 0.05,
          "epsilon_prime": 0.1}
    E, outE = sky(dp=dp, f_S=(1 / 3, 2 / 3, 0.0), galactic=True)
    phiE = per_type_flux(mc, E, outE["bands_nu"], outE["bands_nubar"], outE["dec_band_edges"])
    phiEg = per_type_flux(mc, E, outE["galactic_bands_nu"], outE["galactic_bands_nubar"], outE["dec_band_edges"])
    res["E_dark_portal_extragalactic"] = counts(mc, phiE)[0]
    res["E_dark_portal_galactic"] = counts(mc, phiEg)[0]
    if a.toolkit_weights:
        tk = np.load(a.toolkit_weights)
        # the toolkit's events: the same cuts; matched by (primaryEnergy, primaryZenith, recoDepositedEnergy)
        key = lambda d: np.round(np.stack([d["primaryEnergy"], d["primaryZenith"], d["recoDepositedEnergy"]], 1), 6)
        order_mine = np.lexsort(key(mc).T)
        order_tk = np.lexsort(key(tk).T)
        same = np.array_equal(key(mc)[order_mine], key(tk)[order_tk])
        wa, wt = wA[order_mine], tk["astro_weight_bare"][order_tk]
        m = wt > 0
        res["toolkit"] = dict(events_match=bool(same), n=int(len(wt)),
                              bare_ratio_min=float((wa[m] / wt[m]).min()), bare_ratio_max=float((wa[m] / wt[m]).max()),
                              sum_bare=float(wt.sum()), sum_with_detector_nominal=float(tk["astro_weight_full"].sum()),
                              sum_ours=float(wA.sum()), note=str(tk["note"]))
    res["commit"] = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.join(REPO, "output", "sky"), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, "output", "sky", "hese_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    edges = np.geomspace(6e4, 1e7, 15)
    Edep = mc["recoDepositedEnergy"]
    cases = ((phiA, "A: release's flux", dict(color="k", lw=2)),
             (phiB, "B: propagate_sky, 1:1:1", dict(color="C1", ls="--")),
             (phiC, "C: pion sources", dict(color="C0")),
             (phiG, "D: Galactic (CRINGE x 0.91)", dict(color="C2")),
             (phiC + phiG, "C + D", dict(color="C2", ls=":")),
             (phiE + phiEg, "E: dark portal + Galactic", dict(color="C3")))
    for i, mname in enumerate(MORPH):
        sel = mc["recoMorphology"] == i
        hist = {}
        for phi, lab, st in cases:
            w = LIVETIME_S * mc["weightOverFluxOverLivetime"] * phi
            hist[lab], _ = np.histogram(Edep[sel], bins=edges, weights=w[sel])
            axes[0, i].stairs(hist[lab], edges, label=lab, **st)
        for phi, lab, st in cases:
            if lab.startswith(("A", "D")):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                axes[1, i].stairs(hist[lab] / hist["A: release's flux"], edges, **st)
        axes[0, i].set_xscale("log"); axes[0, i].set_yscale("log")
        axes[0, i].set_title(f"HESE {mname}, 2635 days", fontsize=10)
        axes[1, i].set_xlabel("deposited energy [GeV]"); axes[1, i].set_ylim(0.85, 1.2)
        axes[1, i].axhline(1, color="k", lw=0.6)
    axes[0, 0].set_ylabel("expected events per bin"); axes[1, 0].set_ylabel("ratio to A")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "output", "figures", "hese_check.png"), dpi=110)
    print(f"wrote output/sky/hese_check.json, output/figures/hese_check.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
