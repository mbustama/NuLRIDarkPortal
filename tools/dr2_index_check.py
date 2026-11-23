"""
dr2_index_check.py -- audit finding B7 (docs/dev/AUDIT_2026-11-08_resolution.md): does our DR2 track likelihood,
the one set 2 uses (analysis/events.py _DR2: IC79 + IC86, reconstructed energy >= 10^3.5 GeV, declination >= -5 deg,
atmospheric nuisances with their priors, one normalization per configuration), reproduce IceCube's own measurement
of the astrophysical nu_mu spectrum from northern tracks?

Reference: IceCube 9.5-yr northern tracks, ApJ 928, 50 (2022): single power law, nu_mu + nubar_mu
1.44 (+0.25 -0.26) x 1e-18 GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV, gamma 2.37 +- 0.09. That sample (IC59 + IC79 + IC86,
9.5 yr) is not DR2 (IC79 + IC86, 11.6 yr), so agreement is expected within the errors, not exactly.
Criterion, set before computing: consistent if |gamma_ours - 2.37| is below twice the two 1-sigma errors added in
quadrature.

The astrophysical flux is isotropic, a single power law on a dense energy grid, folded with the same LinearFold the
fits use; a cross-check folds it with DR2Tracks.expected directly (no node interpolation).

Usage: python tools/dr2_index_check.py [--diagnostics]      writes output/analysis/dr2_index_check.json, output/figures/dr2_index_check.png
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from analysis import events as EV  # noqa: E402
from IceCubeRates import flux_adapters as FA  # noqa: E402
from IceCubeRates.northern_muons.dr2 import DR2Tracks  # noqa: E402
import dp_sky  # noqa: E402

REF = dict(phi=1.44, phi_lo=0.26, phi_hi=0.25, gamma=2.37, gamma_err=0.09, pivot_GeV=1e5,
           source="IceCube 9.5-yr northern tracks, ApJ 928, 50 (2022)")
E_NODES = np.logspace(2.0, 9.0, 701)                       # GeV, 0.01 dex
GAMMAS = np.round(np.arange(1.90, 3.301, 0.02), 3)


def astro_flux(E, gamma, phi=1.0):
    return phi * 1e-18 * (np.asarray(E) / REF["pivot_GeV"]) ** (-gamma)


def main(emin=EV.DEFAULT_DR2["E_reco_min_log10"], configs=EV.DEFAULT_DR2["configs"], tag="", dec_min=EV.DEFAULT_DR2["dec_min_deg"],
         atm_scale=1.0, norm_sigma=EV.DR2_NORM_PRIOR):
    """One profile in gamma. atm_scale widens the atmospheric priors (Barr, cr_dgamma, prompt) by that factor;
    norm_sigma is the prior on each configuration's normalization (set 2: 0.1)."""
    t0 = time.time()
    bands = dp_sky.default_dec_band_edges()
    D = EV._DR2(E_NODES, bands, configs=configs, E_reco_min_log10=emin, dec_min_deg=dec_min)
    pri = FA.Atmospheric().priors()
    atm = D.atm_params
    names = ["log10_phi"] + atm + ["prompt"] + [f"norm_{c}" for c in D.configs]
    sig = np.array([atm_scale * pri[p] for p in atm] + [atm_scale * pri["prompt"]] + [norm_sigma] * len(D.configs))
    bounds = [(-2.0, 2.0)] + [(None, None)] * len(atm) + [EV.PROMPT_BOUNDS] + [(-0.9, 1.0)] * len(D.configs)
    nb = len(bands) - 1

    def templates(gamma):
        F = np.broadcast_to(astro_flux(E_NODES, gamma)[None, :], (nb, len(E_NODES)))
        return {c: D.fold[c](F) for c in D.configs}

    def objective(x, T):
        lphi, rest = x[0], x[1:]
        theta, pr, nc = rest[:len(atm)], rest[len(atm)], rest[len(atm) + 1:]
        f = float(np.sum((rest / sig) ** 2))
        for i, c in enumerate(D.configs):
            mu = 10.0 ** lphi * T[c] + (1.0 + nc[i]) * (D.nom[c] + theta @ D.deriv[c] + pr * D.prompt[c])
            f += EV._poisson(np.clip(mu, 1e-300, None), D.data[c])
        return f

    # cross-check of the fold against the direct calculation, at gamma 2.37 and the reference normalization
    tr = DR2Tracks()
    T237 = templates(REF["gamma"])
    direct = {c: tr.expected(lambda E, dlo, dhi: np.broadcast_to(astro_flux(E, REF["gamma"], REF["phi"])[None, :],
                                                                   (len(dlo), len(E))), configs=[c])
              ["per_config"][c][D.select[c]] for c in D.configs}
    fold_vs_direct = {c: float(np.sum(REF["phi"] * T237[c]) / np.sum(direct[c])) for c in D.configs}

    prof, x = [], np.zeros(len(names))
    for g in GAMMAS:
        T = templates(g)
        r = minimize(objective, x, args=(T,), method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=5000, maxfun=200000, ftol=1e-12, gtol=1e-8))
        x = r.x
        prof.append(dict(gamma=float(g), chi2=float(r.fun), phi=float(10.0 ** r.x[0]), ok=bool(r.success),
                         nuis={n: float(v) for n, v in zip(names[1:], r.x[1:])}))
    chi2 = np.array([p["chi2"] for p in prof])
    i0 = int(np.argmin(chi2))
    # parabola through the minimum and its neighbours for the best fit; Delta chi2 = 1 crossings by interpolation
    j = min(max(i0, 1), len(GAMMAS) - 2)
    a, b, c0 = np.polyfit(GAMMAS[j - 1:j + 2], chi2[j - 1:j + 2], 2)
    g_best = float(-b / (2 * a))
    chi2_min = float(c0 - b * b / (4 * a))
    d = chi2 - chi2_min
    lo = float(np.interp(1.0, d[:i0 + 1][::-1], GAMMAS[:i0 + 1][::-1]))
    hi = float(np.interp(1.0, d[i0:], GAMMAS[i0:]))
    err = 0.5 * (hi - lo)
    pull = abs(g_best - REF["gamma"]) / np.hypot(err, REF["gamma_err"])
    def parts(p):
        T, x_ = templates(p["gamma"]), p["nuis"]
        th = np.array([x_[n] for n in atm])
        res = {}
        for c in D.configs:
            ast = p["phi"] * T[c]
            bkg = (1.0 + x_[f"norm_{c}"]) * (D.nom[c] + th @ D.deriv[c] + x_["prompt"] * D.prompt[c])
            ir, ie = np.nonzero(D.select[c])
            nE, nd = D.select[c].shape[1], D.select[c].shape[0]
            res[c] = {k: dict(E=np.bincount(ie, w, nE).tolist(), dec=np.bincount(ir, w, nd).tolist())
                      for k, w in (("data", D.data[c]), ("astro", ast), ("atm", bkg))}
        return res
    p237 = prof[int(np.argmin(np.abs(GAMMAS - REF["gamma"])))]
    residuals = dict(best=parts(prof[i0]), at_2p37=parts(p237), chi2_at_2p37=p237["chi2"] - chi2_min,
                     reco_edges=D.fold[D.configs[0]].reco_edges.tolist(),
                     dec_edges_deg={c: np.degrees(D.fold[c].reco_dec_edges).tolist() for c in D.configs},
                     priors={n: float(v) for n, v in zip(names[1:], sig)})
    out = dict(reference=REF, gamma_best=g_best, gamma_1sigma=[lo, hi], phi_best_1e18=prof[i0]["phi"],
               chi2_min=chi2_min, n_bins={c: int(D.select[c].sum()) for c in D.configs},
               n_events={c: float(D.data[c].sum()) for c in D.configs}, pull_sigma=float(pull),
               consistent=bool(pull < 2.0), criterion="|gamma - 2.37| < 2 sigma (errors added in quadrature)",
               fold_vs_direct_events_ratio=fold_vs_direct,
               astro_events_at_reference={c: float(REF["phi"] * T237[c].sum()) for c in D.configs},
               all_minimizations_ok=all(p["ok"] for p in prof), E_reco_min_log10=emin, configs=list(configs), dec_min_deg=dec_min, atm_scale=atm_scale,
               norm_sigma=norm_sigma, residuals=residuals, profile=prof,
               seconds=time.time() - t0)
    os.makedirs(os.path.join(ROOT, "output", "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "output", "analysis", f"dr2_index_check{tag}.json"), "w"), indent=1)

    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from analysis import figure_style
    figure_style.use()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(GAMMAS, d, color="k", label="Our DR2 fit (IC79 + IC86)")
    ax.axvspan(REF["gamma"] - REF["gamma_err"], REF["gamma"] + REF["gamma_err"], color="tab:blue", alpha=0.25,
               label=r"IceCube 9.5 yr, $1\sigma$")
    ax.axhline(1.0, color="0.5", ls=":")
    ax.set_xlabel(r"Spectral index $\gamma$")
    ax.set_ylabel(r"$\Delta\chi^2$")
    ax.set_ylim(0, 10)
    ax.legend(frameon=True, loc="upper center")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "output", "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "output", "figures", f"dr2_index_check{tag}.png"), dpi=150)
    plt.close(fig)
    print(tag or 'base', {k: v for k, v in out.items() if k in ('gamma_best', 'gamma_1sigma', 'phi_best_1e18', 'pull_sigma')})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostics", action="store_true", help="also: thresholds 10^4 and 10^4.5 GeV, IC86 alone")
    ap.add_argument("--causes", action="store_true", help="also: declination above +10 and +30 deg, atmospheric priors x3,"
                    " configuration normalizations free, both")
    a = ap.parse_args()
    main()
    if a.diagnostics:
        main(emin=4.0, tag="_e4")
        main(emin=4.5, tag="_e4.5")
        main(configs=("IC86",), tag="_ic86")
    if a.causes:
        main(dec_min=10.0, tag="_dec10")
        main(dec_min=30.0, tag="_dec30")
        main(atm_scale=3.0, tag="_atm3")
        main(norm_sigma=1.0, tag="_norm_free")
        main(atm_scale=3.0, norm_sigma=1.0, tag="_atm3_norm_free")
