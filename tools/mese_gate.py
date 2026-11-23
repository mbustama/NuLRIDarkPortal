"""The gate of set 2's MESE event term (audit decision 2026-11-09, fixed before computing): fed a constant flavor
composition and IceCube's broken power law (profiled), our MESE fit must reproduce IceCube's released likelihood grid
(mese_contour.json): the test statistic's minimum along f_tau = 0 (IceCube 5.02) and along f_e = 0 (IceCube 8.65)
each within 1 of IceCube's, and the best fit within 0.05 of (0.30, 0.37, 0.33) in each fraction.

Our fit (IceCubeRates/starting_events/mese.py): MESE's cascades, double cascades and tracks in reconstructed energy
(the release's 1-D energy histograms; IceCube fits 2-D and 3-D), astrophysical events from the per-channel response,
backgrounds the release's Conv and Muon columns (IceCube's fit) times one normalization each, shared by the three
morphologies, free (>= 0); the release's Prompt columns are left out (IceCube fitted the prompt flux at zero; the
columns hold ~1e-7 events). Flux: IceCube's broken power law (arXiv:2510.24957, footnote 2), per flavor nu + nubar,
phi0 at 100 TeV, gamma_1 below E_break, gamma_2 above; nu = nubar (the Glashow channel takes half). Poisson
-2 ln L over every bin of the three histograms; TS = its rise above the global minimum.

Writes output/analysis/mese_gate.json and output/figures/mese_gate.png (ours next to IceCube's).
Usage: python tools/mese_gate.py [N_MAP [N_LINE]]   (map nodes per axis, default 41; nodes per line, default 101)
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from IceCubeRates.starting_events import mese  # noqa: E402

GATE = dict(ts_ftau0=5.02, ts_fe0=8.65, best=(0.30, 0.37, 0.33), tol_ts=1.0, tol_best=0.05)
TOPS = mese.TOPOLOGIES


class Fit:
    def __init__(self):
        z = mese.response()
        fe = z["fine_edges"]
        self.lo, self.hi = 10 ** fe[:-1], 10 ** fe[1:]
        self.Eg = np.logspace(fe[0], fe[-1], 20 * (len(fe) - 1) + 1)
        self.R = {t: mese.isotropic_kernels(t) for t in TOPS}
        self.h = {t: mese.histograms(t) for t in TOPS}
        for t in TOPS:
            n_r = len(z[f"{t}_reco_edges"]) - 1
            if n_r != len(self.h[t]["data"]) or not np.allclose(10 ** z[f"{t}_reco_edges"], self.h[t]["edges"], rtol=1e-9):
                raise SystemExit(f"{t}: response bins differ from the histogram's")
        self.n_eval = 0

    def integrals(self, lphi, g1, g2, lgEb):
        """Per fine bin: the integral of IceCube's broken power law (per flavor nu + nubar)."""
        E, eb = self.Eg, 10 ** lgEb
        f = 10 ** lphi * np.where(E >= eb, (E / 1e5) ** -g2, (eb / 1e5) ** -g2 * (E / eb) ** -g1)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(E))])
        return np.interp(self.hi, E, cum) - np.interp(self.lo, E, cum)

    def deviance(self, p, comp):
        lphi, g1, g2, lgEb, cn, mn = p
        I = self.integrals(lphi, g1, g2, lgEb)
        D = 0.0
        for t in TOPS:
            R, h = self.R[t], self.h[t]
            mu = sum(3 * comp[a] * I @ R[a] for a in range(3)) + (1.5 * comp[0] * I) @ R["gr"]
            mu = mu + cn * h["conv"] + mn * h["muon"]
            n = h["data"]
            mu = np.maximum(mu, 1e-12)
            D += 2.0 * np.sum(mu - n + np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0))
        self.n_eval += 1
        return D

    def profile(self, comp, x0):
        bounds = [(-20.0, -16.0), (0.0, 4.0), (1.5, 4.5), (3.3, 6.5), (0.0, 5.0), (0.0, 10.0)]
        best = None
        starts = [x0] if x0 is not None else [[np.log10(2.7e-18), 1.76, 2.81, 4.5, 1.0, 1.0],
                                              [np.log10(2.0e-18), 2.5, 2.5, 5.0, 1.0, 1.0]]
        for start in starts:
            r = minimize(self.deviance, start, args=(comp,), method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
        return best.fun, best.x


def main(n_map=41, n_line=101):
    t0 = time.time()
    fit = Fit()
    # the map on (f_e, f_tau), warm-started along each row (and from the row below)
    g = np.linspace(0.0, 1.0, n_map)
    Dm = np.full((n_map, n_map), np.nan)
    X = {}
    for i, fe_ in enumerate(g):
        x0 = X.get((i - 1, 0))
        for j, ft_ in enumerate(g):
            if fe_ + ft_ <= 1 + 1e-12:
                d, x0 = fit.profile((fe_, max(0.0, 1 - fe_ - ft_), ft_), x0 if x0 is not None else X.get((i - 1, j)))
                d2, x2 = fit.profile((fe_, max(0.0, 1 - fe_ - ft_), ft_), None) if (i + j) % 7 == 0 else (np.inf, None)
                if d2 < d:
                    d, x0 = d2, x2
                Dm[i, j], X[(i, j)] = d, x0
    k = np.unravel_index(np.nanargmin(Dm), Dm.shape)
    # the global best fit: from the best map node, the composition refined (nuisances profiled inside)
    xk = X[k]

    def over_comp(u):
        fe_, ft_ = u
        if fe_ < 0 or ft_ < 0 or fe_ + ft_ > 1:
            return 1e9
        return fit.profile((fe_, 1 - fe_ - ft_, ft_), xk)[0]
    r = minimize(over_comp, (g[k[0]], g[k[1]]), method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-7, initial_simplex=[(g[k[0]], g[k[1]]), (g[k[0]] + 0.02, g[k[1]]),
                                                                        (g[k[0]], g[k[1]] + 0.02)]))
    D0, (fe0, ft0) = min(r.fun, float(np.nanmin(Dm))), r.x
    if r.fun > np.nanmin(Dm):
        fe0, ft0 = g[k[0]], g[k[1]]
    best = (float(fe0), float(1 - fe0 - ft0), float(ft0))

    def line_min(points):
        vals, x0 = [], xk
        for comp in points:
            d, x0 = fit.profile(comp, x0)
            vals.append(d)
        vals = np.array(vals)
        k_ = int(np.argmin(vals))
        return float(vals[k_] - D0), points[k_]
    s = np.linspace(0.0, 1.0, n_line)
    ts_ftau0, at_ftau0 = line_min([(x, 1 - x, 0.0) for x in s])
    ts_fe0, at_fe0 = line_min([(0.0, 1 - y, y) for y in s])
    if min(ts_ftau0, ts_fe0) < -1e-6:
        raise SystemExit("a line minimum lies below the global best fit: the best-fit search failed")
    TS = Dm - D0
    passed = dict(ts_ftau0=abs(ts_ftau0 - GATE["ts_ftau0"]) <= GATE["tol_ts"],
                  ts_fe0=abs(ts_fe0 - GATE["ts_fe0"]) <= GATE["tol_ts"],
                  best=bool(np.all(np.abs(np.array(best) - np.array(GATE["best"])) <= GATE["tol_best"])))
    res = dict(gate=GATE, ours=dict(best_fit=best, deviance_min=float(D0), nuisances_at_best=fit.profile(best, None)[1].tolist(),
                                    ts_ftau0=ts_ftau0, ts_ftau0_at=at_ftau0, ts_fe0=ts_fe0, ts_fe0_at=at_fe0),
               passed=passed, gate_passed=all(passed.values()), n_bins=int(sum(len(fit.h[t]["data"]) for t in TOPS)),
               n_eval=fit.n_eval, seconds=time.time() - t0, map_axes=g.tolist(), map_TS_fe_ftau=np.where(np.isfinite(TS), TS, None).tolist())
    os.makedirs(os.path.join(REPO, "output", "analysis"), exist_ok=True)
    json.dump(res, open(os.path.join(REPO, "output", "analysis", "mese_gate.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("map")}, indent=1))
    plot(g, TS, res)


def plot(g, TS, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.figure_style import use
    use()
    ic = json.load(open(os.path.join(REPO, "analysis", "data", "icecube_mese_2025", "mese_contour.json")))
    fe_i, ft_i, ts_i = (np.asarray(ic[k], float) for k in ("fe", "ftau", "TS"))
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    levels = [2.30, 6.18]                                      # 68% and 95% C.L., 2 dof
    FE, FT = np.meshgrid(g, g, indexing="ij")
    ax.contour(FE, FT, TS, levels=levels, colors="firebrick", linestyles=["--", "-"])
    if ts_i.ndim == 2:
        ax.contour(fe_i, ft_i, ts_i, levels=levels, colors="k", linestyles=["--", "-"])
    else:
        ax.tricontour(fe_i, ft_i, ts_i, levels=levels, colors="k", linestyles=["--", "-"])
    b = res["ours"]["best_fit"]
    ax.plot(b[0], b[2], "o", color="firebrick")
    ax.plot(0.30, 0.33, "o", color="k")
    ax.plot([0, 1], [1, 0], color="0.6", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$f_{e,\oplus}$")
    ax.set_ylabel(r"$f_{\tau,\oplus}$")
    ax.text(0.55, 0.9, "IceCube (released)", color="k")
    ax.text(0.55, 0.82, "Ours, MESE energy only", color="firebrick")
    ax.text(0.55, 0.74, r"dashed 68\%, solid 95\%" if matplotlib.rcParams["text.usetex"] else "dashed 68%, solid 95%",
            color="0.3")
    out = os.path.join(REPO, "output", "figures", "mese_gate.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("figure", out)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
