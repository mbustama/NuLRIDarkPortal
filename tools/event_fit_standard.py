"""
event_fit_standard.py -- step 5 of docs/dev/EVENT_LIKELIHOOD_PLAN.md: the event likelihoods against the real data with
standard physics. The astrophysical normalization and spectral index fitted to HESE alone, to DR2 tracks alone and to
both (the Galactic component at IceCube's 2026 normalization, 0.91 +- 0.18 as a prior), every event nuisance
profiled; IceCube's own fits are the reference (HESE 7.5 yr: gamma = 2.87; northern tracks 9.5 yr: 2.37 +- 0.09).

  python -I tools/event_fit_standard.py --mc HESE_MC.npz --detector HESE_DETSYS.npz

Writes output/hese/event_fit_standard.json and output/figures/event_fit_standard.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from analysis.model import FlavorModel  # noqa: E402
from analysis.events import EventLikelihood  # noqa: E402

D = 3.477121254719662 / 399                     # the flavor fits' node spacing in log10 E (10 TeV - 30 PeV, 400 nodes)
SOLVER = {"source": {"p_index": 2.5},
          "grid": {"n_E_bins": 515, "E_min_eV": 10 ** (13.0 - 115 * D), "E_max_eV": 3e16, "n_z_bins": 200},
          "sky": {"nside": 8, "extragalactic_norm": 1.6816476344179075e-16, "extragalactic_norm_E_GeV": 14677.992676220705},
          "galactic": {"enabled": True, "template": "cringe", "emission": "3d"},
          "dark_portal": {"g_D": 1e-60, "epsilon_prime": 0.0}}
P_NODES = [2.0, 2.25, 2.5, 2.75, 3.0]
NORM_GAL = (0.91, 0.18)


def fit(ev, start=(0.0, 2.5)):
    """Minimize the event term plus the norm_gal prior over (log10 norm_astro, p_index, norm_gal)."""
    def f(u):
        vals = dict(norm_astro=10.0 ** u[0], p_index=u[1], norm_gal=u[2])
        return ev.chi2(vals) + ((u[2] - NORM_GAL[0]) / NORM_GAL[1]) ** 2
    r = minimize(f, [start[0], start[1], NORM_GAL[0]], method="L-BFGS-B",
                 bounds=[(-1.5, 1.5), (P_NODES[0], P_NODES[-1]), (0.0, 3.0)], options=dict(eps=1e-4, ftol=1e-12))
    f(r.x)                                                          # leave ev.last at the best fit
    return dict(norm_astro=float(10.0 ** r.x[0]), p_index=float(r.x[1]), norm_gal=float(r.x[2]), chi2=float(r.fun),
                nfev=int(r.nfev), nuisances={k: float(v) for k, v in ev.last["nuisances"].items()},
                hese=float(ev.last["hese"]), dr2=float(ev.last["dr2"]), priors=float(ev.last["priors"]))


def profile_p(ev, best, grid):
    out = []
    for p in grid:
        def f(u):
            vals = dict(norm_astro=10.0 ** u[0], p_index=p, norm_gal=u[1])
            return ev.chi2(vals) + ((u[1] - NORM_GAL[0]) / NORM_GAL[1]) ** 2
        r = minimize(f, [np.log10(best["norm_astro"]), best["norm_gal"]], method="L-BFGS-B",
                     bounds=[(-1.5, 1.5), (0.0, 3.0)], options=dict(eps=1e-4, ftol=1e-12))
        out.append(float(r.fun))
    return out


def components(ev, values):
    """Expected events per component at the values and the last profiled nuisances (exact detector weights)."""
    A = ev.astro(values)
    v = ev.last["nuisances"]
    theta = np.array([v[p] for p in ev.atm_params])
    out = {}
    if ev.hese is not None:
        H = ev.hese
        f = H.factors({s: v[s] for s in H.names})

        def h(per):
            return np.bincount(H.bin, weights=per, minlength=H.n_bins)
        out["hese"] = dict(astro=h(A["hese_events"] * f["Astro"]),
                           conv=h((1.0 + v["hese_conv_norm"]) * (H.conv + theta @ H.deriv) * f["Conv"]),
                           prompt=h((1.0 + v["prompt"]) * H.prompt * f["Prompt"]), muons=v["hese_muon_norm"] * H.muons,
                           data=H.data)
    if ev.dr2 is not None:
        Dd = ev.dr2
        out["dr2"] = {c: dict(astro=A["dr2"][c], atm=(1.0 + v[f"dr2_norm_{c}"]) * (Dd.nom[c] + theta @ Dd.deriv[c]
                                                                                 + v["prompt"] * Dd.prompt[c]),
                              data=Dd.data[c]) for c in Dd.configs}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--outdir", default=os.path.join(REPO, "output"))
    a = ap.parse_args()
    t0 = time.time()
    model = FlavorModel(SOLVER, free_names=("norm_astro", "p_index", "norm_gal"), p_nodes=P_NODES, sky=True,
                        flavor_band_eV=[1e13, 3e16])
    hese_spec = {"mc": a.mc, "detector": a.detector}
    res, evs = {}, {}
    for name, spec in (("HESE", {"hese": hese_spec}), ("DR2", {"dr2": True}), ("HESE + DR2", {"hese": hese_spec, "dr2": True})):
        t1 = time.time()
        ev = EventLikelihood(model, spec)
        best = fit(ev)
        grid = np.round(np.arange(2.0, 3.0001, 0.05), 3)
        best["profile_p"] = dict(p=grid.tolist(), chi2=profile_p(ev, best, grid))
        best["seconds"] = time.time() - t1
        res[name], evs[name] = best, ev
        pulls = {k: round(val, 3) for k, val in best["nuisances"].items()}
        print(f"{name}: norm_astro {best['norm_astro']:.3f}, p_index {best['p_index']:.3f}, norm_gal "
              f"{best['norm_gal']:.3f}, chi2 {best['chi2']:.2f} ({best['seconds']:.0f} s)\n  nuisances {pulls}", flush=True)
    res["solver_calls"] = model.n_solver_calls
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.join(a.outdir, "hese"), exist_ok=True)
    os.makedirs(os.path.join(a.outdir, "figures"), exist_ok=True)
    with open(os.path.join(a.outdir, "hese", "event_fit_standard.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    both = res["HESE + DR2"]
    ev = evs["HESE + DR2"]
    vals = dict(norm_astro=both["norm_astro"], p_index=both["p_index"], norm_gal=both["norm_gal"])
    ev.chi2(vals)
    comp = components(ev, vals)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    # HESE: all morphologies, per deposited energy
    from IceCubeRates.starting_events import hese as hese_mod
    hv = hese_mod.HESEEvents(a.mc)
    Eb = hv.E_bins

    def per_energy(flat):
        parts = hv._split(np.asarray(flat))
        return sum(parts[m].sum(0) for m in hese_mod.MORPHOLOGIES)
    ax = axes[0]
    c = comp["hese"]
    bottom = np.zeros(len(Eb) - 1)
    for key, col, lab in (("muons", "C4", "muons"), ("conv", "C0", "conventional"), ("prompt", "C1", "prompt"),
                          ("astro", "C3", "astrophysical")):
        e = per_energy(c[key])
        ax.stairs(bottom + e, Eb, baseline=bottom, fill=True, color=col, alpha=0.55, label=lab)
        bottom = bottom + e
    d = per_energy(c["data"])
    xc = np.sqrt(Eb[1:] * Eb[:-1])
    ax.errorbar(xc, d, yerr=np.sqrt(d), fmt="o", color="k", ms=3, lw=0.8, label="HESE data")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-2, 30)
    ax.set_xlabel("deposited energy (GeV)"); ax.set_ylabel("events per bin")
    ax.set_title(f"HESE, joint best fit (gamma = {both['p_index']:.2f})", fontsize=9)
    ax.legend(fontsize=7)
    # DR2: each configuration, per reconstructed energy (northern sky)
    for ax, cfg in zip(axes[1:3], ev.dr2.configs):
        sel = ev.dr2.select[cfg]
        r_edges = ev.dr2.fold[cfg].reco_edges
        cols = np.nonzero(sel.any(axis=0))[0]
        shape = sel.shape

        def per_reco(flat):
            full = np.zeros(shape)
            full[sel] = flat
            return full.sum(0)[cols]
        e_edges = r_edges[cols[0]:cols[-1] + 2]
        cc = comp["dr2"][cfg]
        atm, ast = per_reco(cc["atm"]), per_reco(cc["astro"])
        ax.stairs(atm, e_edges, fill=True, color="C0", alpha=0.55, label="atmospheric")
        ax.stairs(atm + ast, e_edges, baseline=atm, fill=True, color="C3", alpha=0.55, label="astrophysical")
        dd = per_reco(cc["data"])
        ax.errorbar(0.5 * (e_edges[1:] + e_edges[:-1]), dd, yerr=np.sqrt(dd), fmt="o", color="k", ms=2.5, lw=0.7,
                    label="DR2 events")
        ax.set_yscale("log"); ax.set_ylim(0.3, None)
        ax.set_xlabel(r"log$_{10}$(reconstructed energy / GeV)"); ax.set_ylabel("events per bin (dec > -5 deg)")
        ax.set_title(f"DR2 {cfg}", fontsize=9)
        ax.legend(fontsize=7)
    ax = axes[3]
    for name, col in (("HESE", "C3"), ("DR2", "C0"), ("HESE + DR2", "k")):
        pr = res[name]["profile_p"]
        ch = np.array(pr["chi2"])
        ax.plot(pr["p"], ch - ch.min(), color=col, label=f"{name}: {res[name]['p_index']:.2f}")
    ax.axhline(1.0, color="0.6", lw=0.6, ls="--")
    ax.axvline(2.87, color="C3", lw=0.6, ls=":")
    ax.axvline(2.37, color="C0", lw=0.6, ls=":")
    ax.set_ylim(0, 12)
    ax.set_xlabel("spectral index p"); ax.set_ylabel(r"$\Delta\chi^2$ (profiled)")
    ax.set_title("Standard physics: spectral index (IceCube's: dotted)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "figures", "event_fit_standard.png"), dpi=110)
    print(f"wrote {a.outdir}/hese/event_fit_standard.json and the figure ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
