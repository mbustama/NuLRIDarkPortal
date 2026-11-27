"""Combine the coverage tasks (reread with the repository's reading) and draw the survival of the test statistic."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import coverage_limits as cl
from scipy.stats import chi2 as c2
from scipy.interpolate import PchipInterpolator
D = sys.argv[1]
PRE = sys.argv[3] if len(sys.argv) > 3 else "cov"      # "cov2" for the rerun with multi-start fits
LE, SI = PRE + "_le_ltau", PRE + "_si"
res = {}
for tag in (LE, SI):
    q, cov, real, above = [], [], None, 0
    nt = 0
    for p in sorted(glob.glob(f"{D}/{tag}_t*.json")):
        if "finished" not in json.load(open(p)):
            continue
        o = cl.reread(p); nt += 1
        real = o["real"]; lt = o["log10g_true"]; rr = o["real_read"]
        for r in o["results"]:
            c = {np.log10(float(g)): v for g, v in r["chi2"].items()}
            gmin = min(c, key=c.get)
            below = (c[gmin] < r["chi2_true"]) and gmin < lt
            q.append(r["dchi2_true"] if below else 0.0); cov.append(r["covered"])
            # the reading and the one-sided statistic must agree, except when the column's lowest chi2 lies above the
            # truth (then the reading may exclude the truth from below: counted in "above", q = 0)
            if below:
                assert r["covered"] == (not (q[-1] > 3.84)) or abs(q[-1] - 3.84) < 0.4, (p, q[-1], r["covered"])
            elif not r["covered"]:
                above += 1
    q = np.array(q); cov = np.array(cov, bool); n = len(q); k = int(cov.sum())
    # the threshold that leaves the truth allowed in 97.5% and in 95% of the pseudo-experiments, and the limit it gives
    rc = real["chi2"] if "chi2" in real else real
    G = np.array(sorted(np.log10(float(g)) for g in rc)); C = np.array([rc[g] for g in sorted(rc, key=float)]); C -= C.min()
    out = dict(tasks=nt, toys=n, covered=k, frac=k / n, err=float(np.sqrt(k / n * (1 - k / n) / n)), real_read=rr, lt=lt, excluded_from_below=above)
    for lev in (0.95, 0.975):
        t = float(np.quantile(q, lev)); out[f"thr{lev}"] = t
        up = G > G[np.argmin(C)]; f = PchipInterpolator(G[up], C[up]); x = np.linspace(G[up][0], G[up][-1], 20001)
        hit = x[f(x) >= max(t, 3.84)]; out[f"lim{lev}"] = float(hit[0]) if len(hit) else None
    res[tag] = (q, out); print(tag, json.dumps(out))
import matplotlib; matplotlib.use("Agg")
from analysis.figure_style import use, pct; use()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.5, 6.2))
t = np.linspace(0, 26, 600)
ax.plot(t, 0.5 * c2.sf(t, 1), color="0.35", ls="--", lw=1.5)
for tag, col, lab in ((LE, "#0072B2", r"$L_e - L_\tau$"), (SI, "#D55E00", r"Secret interactions")):
    q, o = res[tag]
    ax.step(t, [(q > x).mean() for x in t], where="post", color=col, lw=2,
            label=lab + rf" ({o['covered']} of {o['toys']} allowed)")
ax.axvline(3.84, color="k", lw=1, ls=":")
ax.text(4.1, 0.0036, r"Our threshold, 3.84", rotation=90, va="bottom", ha="left", fontsize=14)
ax.text(6.9, 0.006, "Wilks's theorem", color="0.35", fontsize=14)
ax.set_yscale("log"); ax.set_ylim(3e-3, 0.7); ax.set_xlim(0, 16)
ax.set_xlabel(r"Threshold on $\Delta\chi^2$"); ax.set_ylabel(r"Fraction with the true coupling excluded", fontsize=20)
ax.legend(loc="upper right", fontsize=14, title=r"Pseudo-experiments, mediator mass 1.3 MeV", title_fontsize=14, frameon=True, edgecolor="black", framealpha=1)
fig.savefig(sys.argv[2], bbox_inches="tight", dpi=200)
