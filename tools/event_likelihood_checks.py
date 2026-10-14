"""
event_likelihood_checks.py -- checks of the event likelihoods' responses (docs/dev/EVENT_LIKELIHOOD_PLAN.md, steps 2-3):

  (a) HESE: the sparse linear maps (starting_events.hese.LinearResponse) against the per-event weighting
      (HESEEvents.expected): exact for an E^-2.5 flux on the grid; against hese.sky_flux (log-log) for solver fluxes.
  (b) HESE detector systematics per bin (BinDetector: the release's exact per-event weights summed per bin at each grid
      value, interpolated per bin) against the exact per-event product (HESEEvents.detector_factor) at random values of
      all three systematics within the grids, for the astrophysical, conventional and prompt components; in units of
      each bin's Poisson error too.
  (c) DR2: northern_muons.dr2.LinearFold against DR2Tracks.expected for the same E^-2.5 flux.

  python -I tools/event_likelihood_checks.py --mc HESE_MC.npz --detector HESE_DETSYS.npz [--solver]

Writes output/hese/event_likelihood_checks.json and output/figures/event_likelihood_checks.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from IceCubeRates.starting_events import hese  # noqa: E402
from IceCubeRates.northern_muons.dr2 import DR2Tracks, LinearFold  # noqa: E402
from IceCubeRates.flux_adapters import power_law  # noqa: E402

D = 3.477121254719662 / 399                  # the flavor fits' node spacing in log10 E (10 TeV - 30 PeV, 400 nodes)
E_NODES_GEV = np.logspace(13.0 - 115 * D, np.log10(3e16), 515) / 1e9      # the event fits' grid (step 'Energy grid')
BANDS = np.arcsin(np.linspace(-1.0, 1.0, 21))                             # the solver's default 20 bands
PER_TYPE, GAMMA = 6.37e-18 / 6.0, 2.5


def windowed_power_law(per_type, gamma, E_lo, E_hi):
    def flux(flavor, anti, E, dec):
        E = np.asarray(E)
        return per_type * (E / 1e5) ** (-gamma) * ((E >= E_lo) & (E <= E_hi))
    return flux


def grid_power_law(per_type, gamma):
    F = per_type * (E_NODES_GEV / 1e5) ** (-gamma)
    F = np.broadcast_to(F[None, :, None], (len(BANDS) - 1, len(E_NODES_GEV), 3)).copy()
    return F, F.copy()


def compare(a, b, floor=1e-3):
    big = b > floor * b.max()
    r = np.abs(a - b)[big] / b[big]
    return dict(worst=float(r.max()), median=float(np.median(r)), total=float(a.sum() / b.sum() - 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--solver", action="store_true", help="also compare with a dark-portal solver sky (~30 s)")
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--outdir", default=os.path.join(REPO, "output"))
    a = ap.parse_args()
    t0 = time.time()
    res = {}
    ev = hese.HESEEvents(a.mc)
    ev.load_detector(a.detector)
    t1 = time.time()
    lin = hese.LinearResponse(ev, E_NODES_GEV, BANDS)
    res["hese_build_seconds"] = time.time() - t1
    res["hese_nnz_per_map"] = int(lin.R.nnz)
    # (a) exact for E^-2.5 on the grid
    F_nu, F_nb = grid_power_law(PER_TYPE, GAMMA)
    t1 = time.time()
    n_lin = lin(F_nu, F_nb)
    res["hese_apply_ms"] = 1e3 * (time.time() - t1)
    ref = ev.flat_histogram(ev.expected(windowed_power_law(PER_TYPE, GAMMA, E_NODES_GEV[0], E_NODES_GEV[-1])))
    res["hese_power_law"] = compare(n_lin, ref)
    print("HESE linear map vs per-event, E^-2.5:", res["hese_power_law"], f"({n_lin.sum():.3f} events)", flush=True)
    if a.solver:
        import dark_portal_solver as dps
        cfg = {"source": {"p_index": 2.5},
               "grid": {"n_E_bins": 515, "E_min_eV": E_NODES_GEV[0] * 1e9, "E_max_eV": 3e16, "n_z_bins": 200},
               "sky": {"output": "bands", "nside": 8, "extragalactic_norm": 1.68e-16, "extragalactic_norm_E_GeV": 14678.0},
               "galactic": {"enabled": True, "template": "cringe", "emission": "3d"},
               "dark_portal": {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "epsilon_prime": 0.1,
                               "g_D": 300.0 * 1.66e-54 / (0.1 * 0.5)},
               "execution": {"verbose": 0}}
        E, p = dps.build_solver_params(cfg)
        p["show_progress"] = False
        sky = dps.propagate_sky(E, p)
        Fn = sky["bands_nu"] + sky["galactic_bands_nu"]
        Fb = sky["bands_nubar"] + sky["galactic_bands_nubar"]
        n_lin_s = lin(Fn, Fb)
        ref_s = ev.flat_histogram(ev.expected(hese.sky_flux(sky)))
        res["hese_solver_dark_portal"] = compare(n_lin_s, ref_s)
        print("HESE linear map vs per-event, dark-portal sky:", res["hese_solver_dark_portal"], flush=True)
    # (b) detector systematics per bin against the per-event product
    names = list(ev.det_grid)
    nominal_log = {s: float(np.max(np.abs(ev.det[f"log10w_{s}_{c}"][list(ev.det_grid[s]).index(ev.det_nominal[s])])))
                   for s in names for c in ("Astro",)}
    res["max_abs_log10w_at_nominal"] = nominal_log
    astro_events = ev.expected(windowed_power_law(PER_TYPE, GAMMA, E_NODES_GEV[0], E_NODES_GEV[-1]))
    tpl = ev.atmospheric_templates()
    comps = {"Astro": astro_events, "Conv": tpl["conv"], "Prompt": tpl["prompt"]}
    dets = {}
    for comp, per_event in comps.items():
        H0 = ev.flat_histogram(per_event)
        H_nodes = {}
        for s in names:
            L = ev.det[f"log10w_{s}_{comp}"]
            H_nodes[s] = np.stack([ev.flat_histogram(per_event * 10.0 ** L[k].astype(np.float64))
                                   for k in range(len(ev.det_grid[s]))])
        dets[comp] = hese.BinDetector(ev.det_grid, H0, H_nodes)
    rng = np.random.default_rng(20261014)
    total_nom = sum(d.H0 for d in dets.values()) + ev.flat_histogram(ev.muons())
    worst = {c: 0.0 for c in comps}
    worst_sigma = {c: 0.0 for c in comps}
    worst_one = {c: 0.0 for c in comps}
    draws = []
    for _ in range(a.draws):
        vals = {}
        for s in names:
            g = ev.det_grid[s]
            vals[s] = float(np.clip(ev.det_nominal[s] + ev.det_sigma[s] * rng.uniform(-2.0, 2.0), g[0], g[-1]))
        draws.append(vals)
        for comp, per_event in comps.items():
            exact = ev.flat_histogram(per_event * ev.detector_factor(comp, vals))
            approx = dets[comp](vals)
            big = exact > 1e-3 * exact.max()
            worst[comp] = max(worst[comp], float(np.max(np.abs(approx - exact)[big] / exact[big])))
            sig = np.sqrt(np.maximum(total_nom, 1e-3))
            worst_sigma[comp] = max(worst_sigma[comp], float(np.max(np.abs(approx - exact) / sig)))
    # one systematic at a time, off the grid: only the per-event cubic in log vs the per-bin one differ
    for comp, per_event in comps.items():
        for s in names:
            g = ev.det_grid[s]
            for x in 0.5 * (g[1:] + g[:-1]):
                vals = {s: float(x)}
                exact = ev.flat_histogram(per_event * ev.detector_factor(comp, vals))
                approx = dets[comp](vals)
                big = exact > 1e-3 * exact.max()
                worst_one[comp] = max(worst_one[comp], float(np.max(np.abs(approx - exact)[big] / exact[big])))
    res["detector_per_bin_vs_per_event"] = dict(
        draws=a.draws, all_three_within_2sigma_worst_relative=worst, all_three_worst_in_poisson_sigma=worst_sigma,
        one_at_a_time_midpoints_worst_relative=worst_one)
    print("detector per bin vs per event (all three moved, within 2 sigma): worst relative", worst,
          "| in Poisson sigma", worst_sigma, "| one at a time, between grid values:", worst_one, flush=True)
    # (c) DR2 linear fold against the fold, the same E^-2.5 flux (nu_mu + nubar_mu = 2 per type)
    tr = DR2Tracks()
    F2 = 2.0 * PER_TYPE * (E_NODES_GEV / 1e5) ** (-GAMMA)
    F2 = np.broadcast_to(F2[None, :], (len(BANDS) - 1, len(E_NODES_GEV)))

    def pl(E, dlo, dhi):
        E = np.asarray(E, dtype=np.float64)
        return power_law(2.0 * PER_TYPE, GAMMA)(E, dlo, dhi) * ((E >= E_NODES_GEV[0]) & (E <= E_NODES_GEV[-1]))[None, :]
    res["dr2"] = {}
    for c in ("IC79", "IC86"):
        fold = LinearFold(tr, c, E_NODES_GEV, BANDS)
        ref2 = tr.expected(pl, configs=[c])["per_config"][c].ravel()
        res["dr2"][c] = compare(fold(F2), ref2)
    print("DR2 linear fold vs fold, E^-2.5:", res["dr2"], flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.join(a.outdir, "hese"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "figures"), exist_ok=True)
    with open(os.path.join(a.outdir, "hese", "event_likelihood_checks.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    vals = draws[0]
    for ax, comp in zip(axes, comps):
        exact = ev.flat_histogram(comps[comp] * ev.detector_factor(comp, vals))
        approx = dets[comp](vals)
        nom = dets[comp].H0
        big = exact > 1e-3 * exact.max()
        x = np.arange(len(exact))[big]
        ax.semilogy(x, nom[big], ".", ms=3, color="0.6", label="no detector weights")
        ax.semilogy(x, exact[big], "o", ms=3, mfc="none", color="C0", label="per-event product (exact)")
        ax.semilogy(x, approx[big], ".", ms=2, color="C3", label="per bin (likelihood)")
        r = np.abs(approx - exact)[big] / exact[big]
        ax.set_title(f"{comp}: worst {r.max():.1e} at " + ", ".join(f"{s} {v:.3g}" for s, v in vals.items()),
                     fontsize=8.5)
        ax.set_xlabel("analysis bin (cascades, tracks, double cascades)")
        ax.set_ylabel("expected events")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "figures", "event_likelihood_checks.png"), dpi=110)
    print(f"wrote {a.outdir}/hese/event_likelihood_checks.json and the figure ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
