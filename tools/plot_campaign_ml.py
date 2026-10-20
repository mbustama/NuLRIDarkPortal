"""
plot_campaign_ml.py -- the campaign_ml planes (2026-10-20: mixing fixed, m_lightest profiled exactly, the relics thermal
in every MeV collision term) as excluded regions: per physics model and data set, the couplings with a per-mass
Delta chi^2 >= 3.84 (95%) shaded, edges drawn through
the per-mass crossings (field). Shading rather than one limit line (Mauricio, 2026-10-20): with the
source compositions fitted, a data set can exclude a narrow band of couplings and allow larger ones again (the future
detectors near 3e-28 in the ultralight planes). One figure with the compositions fixed to pion decay, one with them
fitted. Mass columns whose smallest scanned coupling is not at the no-effect chi^2 are left unshaded (none in this
campaign). The plane summaries' 'elevated' columns are NOT masked: in the IceCube flavor + spectrum MeV planes the data
prefer an effect near 3 MeV, which lifts every other mass above the global minimum although its scan does reach the
no-effect couplings; the per-mass reading is valid there.
Reads the plane summaries under output/analysis/campaign_ml/.
  python tools/plot_campaign_ml.py [output prefix, default output/figures/campaign_ml_excluded]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = [("le_lmu_ul", r"$L_e - L_\mu$, ultralight"), ("le_ltau_ul", r"$L_e - L_\tau$, ultralight"),
          ("le_lmu_mev", r"$L_e - L_\mu$, MeV"), ("le_ltau_mev", r"$L_e - L_\tau$, MeV"),
          ("dp_e_mev", r"dark portal, $e$ texture, MeV"), ("dp_mu_mev", r"dark portal, $\mu$ texture, MeV"),
          ("dp_tau_mev", r"dark portal, $\tau$ texture, MeV")]
# categorical slots 1-3 of the validated default palette (dataviz skill), in fixed order
SETS = [("now_ic26", "IceCube flavor + spectrum (today)", "#2a78d6"),
        ("now_events", "HESE + DR2 events (today)", "#eb6834"),
        ("proj_future", "future detectors (projection)", "#1baf7a")]
YLABEL = {"g_D": r"$g_D$", "g_D_trace": r"$g_D$ (texture trace)"}
COMPS = [("", "pion", "source compositions fixed to pion decay"), ("_fx", "fitted", "source compositions fitted")]


D = 1.5        # decades: the signed distance to the 95% edge is clipped here (see field)
STEP = 0.005   # decades: the coupling resolution of the drawn field


def signed(G, excl):
    """signed distance in log10 g from each G to the nearest edge of the excluded set (+ inside), clipped at +-D"""
    edges = 0.5 * (G[1:] + G[:-1])[excl[1:] != excl[:-1]]
    if len(edges) == 0:
        dist = np.full(len(G), D)
    else:
        k = np.clip(np.searchsorted(edges, G), 1, len(edges))
        dist = np.minimum(abs(G - edges[k - 1]), abs(G - edges[np.minimum(k, len(edges) - 1)]))
    return np.where(excl, 1.0, -1.0) * np.minimum(dist, D)


def field(path):
    """(log10 m columns, log10 g, signed-distance field, y name): per mass, the couplings with delta chi^2 >= 3.84,
    their edges interpolated linearly in log10 g between samples (grid points and the ridge refinement,
    tools/refine_ridges.py), turned into a signed distance to the nearest edge. Contouring that field at 0 joins each
    edge in a straight line between neighbouring masses -- the per-mass crossings themselves, without the staircase a
    contour of delta chi^2 draws when it jumps from ~0 to hundreds within one coupling step. A region that ends between
    two masses ends about halfway (the clipping at D). For display: columns whose smallest coupling is not at the
no-effect chi^2 (the median over masses there, within 0.3) masked (unshaded); couplings
    below a column's lowest allowed one not excluded (upper limits only); isolated specks -- excluded grid regions of at
    most three points, none above delta chi^2 = 5 -- not excluded (a hair over 3.84, they flip with the sampling)."""
    from scipy import ndimage
    s = json.load(open(path))
    x = np.array(s["grids"][0])
    y = np.array(s["grids"][1])
    G = np.arange(np.log10(y[0]), np.log10(y[-1]) + STEP / 2, STEP)
    chi = np.array([[np.nan if v is None else v for v in row] for row in s["chi2"]])
    chi0 = chi.copy()
    d = chi - np.nanmin(chi, axis=1, keepdims=True)
    lab, n = ndimage.label(d >= 3.84)
    for k in range(1, n + 1):
        sel = lab == k
        if sel.sum() <= 3 and np.nanmax(d[sel]) < 5.0:
            chi[sel] = np.nan                                   # shown as allowed below
    cols = {i: [(float(y[j]), float(chi[i, j])) for j in range(len(y)) if np.isfinite(chi[i, j])] for i in range(len(x))}
    for f in glob.glob(os.path.join(os.path.dirname(path), "refine_ridges.*.jsonl")) + \
            glob.glob(os.path.join(os.path.dirname(path), "refine_edges.*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m, g = r["held"][s["names"][0]], r["held"][s["names"][1]]
            cols[int(np.argmin(np.abs(np.log(x / m))))].append((g, r["chi2"]))
    elevated = {i for i in range(len(x)) if not abs(chi0[i, 0] - np.nanmedian(chi0[:, 0])) < 0.3}
    S, low = {}, {}                                            # per grid column: the field, the lowest allowed log g
    for i, pts in cols.items():
        if i in elevated or not pts:
            continue
        specks = [(float(y[j]), 0.0) for j in range(len(y)) if np.isnan(chi[i, j]) and np.isfinite(d[i, j])]
        pts.sort()
        g = np.log10([p[0] for p in pts])
        c = np.array([p[1] for p in pts])
        dc = c - min(c.min(), np.nanmin(chi0[i]))              # the column's minimum, refinement included
        ok = dc < 3.84
        if ok.any():
            dc[:int(np.argmax(ok))] = 0.0                       # below the lowest allowed coupling: not an exclusion
            low[i] = g[int(np.argmax(ok))]
        pts = sorted(list(zip(g, dc)) + [(np.log10(gs), v) for gs, v in specks])
        excl = np.interp(G, [p[0] for p in pts], [p[1] for p in pts]) >= 3.84
        S[i] = signed(G, excl)
    xs = {np.log10(x[i]): S.get(i) for i in range(len(x))}     # None: masked
    # the ridges and edges sampled between the grid's masses (refine_ridges.py --between, refine_edges.py): inside
    # each sampled window (runs of couplings at most 0.15 decade apart), their own points (minimum from the
    # neighbouring full columns, the no-effect minimum being the same at every mass); outside, the neighbouring
    # columns' field interpolated in log10 m
    between = {}
    for f in glob.glob(os.path.join(os.path.dirname(path), "refine_ridges_between.*.jsonl")) + \
            glob.glob(os.path.join(os.path.dirname(path), "refine_edges_between.*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            between.setdefault(round(np.log10(r["held"][s["names"][0]]), 6), []).append((r["held"][s["names"][1]],
                                                                                         r["chi2"]))
    for lm, pts in between.items():
        m = 10 ** lm
        i = int(np.searchsorted(x, m)) - 1
        if i < 0 or i + 1 >= len(x) or i not in S or i + 1 not in S:
            continue
        t = np.log(m / x[i]) / np.log(x[i + 1] / x[i])
        cmin = min(np.nanmin(chi0[i]), np.nanmin(chi0[i + 1]), min(c for _, c in pts))
        pts.sort()
        g = np.log10([p[0] for p in pts])
        dc = np.array([p[1] for p in pts]) - cmin
        excl = (1 - t) * S[i] + t * S[i + 1] > 0
        for run in np.split(np.arange(len(g)), np.flatnonzero(np.diff(g) > 0.15) + 1):
            win = (G >= g[run[0]]) & (G <= g[run[-1]])
            excl[win] = np.interp(G[win], g[run], dc[run]) >= 3.84
        if i in low and i + 1 in low:
            excl &= G >= (1 - t) * low[i] + t * low[i + 1]
        xs[lm] = signed(G, excl)
    lx = np.array(sorted(xs))
    F = np.ma.masked_invalid(np.array([xs[v] if xs[v] is not None else np.full(len(G), np.nan) for v in lx]).T)
    return lx, G, F, s["names"][1]


def figure(suffix, comp_title, out):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax, (model, title) in zip(axes.flat, MODELS):
        yname = None
        for ds, label, color in SETS:
            hits = glob.glob(f"{REPO}/output/analysis/campaign_ml/{ds}/{model}{suffix}/*/plane_*.json")
            if not hits:
                continue
            lx, G, F, yname = field(hits[0])
            ax.contourf(lx, G, F, levels=[0, 2 * D], colors=[color], alpha=0.22)
            ax.contour(lx, G, F, levels=[0], colors=[color], linewidths=1.8)
        if model.endswith("_mev"):
            ax.set_ylim(-6, -2)                          # nothing is excluded below ~1e-5 in the MeV planes
        ax.set_xlabel(r"$\log_{10}(m_{Z_D}$ / eV$)$")
        ax.set_ylabel(r"$\log_{10}$ " + YLABEL.get(yname, yname or "coupling"))
        ax.set_title(title, fontsize=10.5)
        ax.grid(True, color="0.9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    leg_ax = axes.flat[-1]
    leg_ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=lab) for _, lab, c in SETS]
    leg_ax.legend(handles=handles, loc="center", fontsize=9.5, frameon=False,
                  title="Excluded at 95% (shaded; per mass, " + r"$\Delta\chi^2 \geq 3.84$)", title_fontsize=9.5)
    fig.suptitle(f"Campaign of 2026-10-20, {comp_title}: excluded couplings per mediator mass; mixing fixed, lightest "
                 "mass profiled, relics thermal in every collision", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)


def main(prefix):
    for suffix, tag, comp_title in COMPS:
        figure(suffix, comp_title, f"{prefix}_{tag}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "figures", "campaign_ml_excluded"))
