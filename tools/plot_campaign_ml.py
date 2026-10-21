"""
plot_campaign_ml.py -- the campaign_ml planes (2026-10-20: mixing fixed, m_lightest profiled exactly, the relics thermal
in every MeV collision term) as excluded regions: per physics model and data set, the couplings with a per-mass
Delta chi^2 >= 3.84 (95%) shaded, edges drawn through
the per-mass crossings (field). One figure with the compositions fixed to pion decay, one with them fitted. With them
fitted, the ultralight planes also exclude a thin band (~0.5 decade) along the edge of the pion-decay region, which
exists only because each source's composition is taken to be the same at every energy (notes, 'Thin excluded bands
with fitted source compositions'); Mauricio, 2026-10-21: left out of the exclusion figures (thin_bands takes them out),
drawn on their own, as ribbons at their true width, in <prefix>_thin_bands.png. Drawn in Mauricio's style
(analysis/figure_style.py and its matplotlibrc). Mass columns whose smallest scanned coupling is not at the no-effect chi^2 are left unshaded (none in this
campaign). The plane summaries' 'elevated' columns are NOT masked: in the IceCube flavor + spectrum MeV planes the data
prefer an effect near 3 MeV, which lifts every other mass above the global minimum although its scan does reach the
no-effect couplings; the per-mass reading is valid there.
Reads the plane summaries under output/analysis/campaign_ml/.
  python tools/plot_campaign_ml.py [output prefix, default output/figures/campaign_ml]
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
sys.path.insert(0, REPO)
from analysis.figure_style import PANEL, pct, use  # noqa: E402
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
    S, low, cov = {}, {}, {}                                   # per column: the field, lowest allowed log g, sampled
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
        cov[np.log10(x[i])] = np.ones(len(G), bool)
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
        cov[lm] = np.zeros(len(G), bool)
        for run in np.split(np.arange(len(g)), np.flatnonzero(np.diff(g) > 0.15) + 1):
            win = (G >= g[run[0]]) & (G <= g[run[-1]])
            excl[win] = np.interp(G[win], g[run], dc[run]) >= 3.84
            cov[lm] |= win
        if i in low and i + 1 in low:
            excl &= G >= (1 - t) * low[i] + t * low[i + 1]
        xs[lm] = signed(G, excl)
    lx = np.array(sorted(xs))
    F = np.ma.masked_invalid(np.array([xs[v] if xs[v] is not None else np.full(len(G), np.nan) for v in lx]).T)
    C = np.array([cov.get(v, np.zeros(len(G), bool)) for v in lx]).T
    return lx, G, F, s["names"][1], C


def thin_bands(lx, G, F, C, r=0.4, link=0.6, step=0.05, area=3.0):
    """The thin excluded bands of an ultralight plane as ribbons. Thin: no disk of radius r decades (in log m and log g
    alike) fits inside it -- the excluded set rasterised every step decades and opened with that disk; what the
    opening removes is thin (the bands are ~0.5 decade across, the regions far more), and so is what it leaves in
    patches smaller than area square decades that do not reach the top of the scan (where a band widens). Per mass, a thin excluded run
    of couplings is taken out of the shaded field and, where the mass column was sampled there (C), its lower and upper
    edges become a node (runs less than 0.2 decade apart, one node). Nodes of neighbouring masses whose runs overlap within link
    decades are joined; a band ends at a sampled mass without a run near it (a real gap), not at a mass that was not
    sampled there; a band seen at one mass only is dropped. Returns (field without the bands, [(log m, lower log g,
    upper log g) ribbons])."""
    from scipy import ndimage
    Ff = F.filled(np.nan)
    xr = np.arange(lx[0], lx[-1] + step / 2, step)
    rows = np.arange(0, len(G), int(round(step / STEP)))
    R = np.zeros((len(rows), len(xr)), bool)
    for a_, gi in enumerate(rows):
        ok = np.isfinite(Ff[gi])
        if ok.sum() > 1:
            R[a_] = np.interp(xr, lx[ok], Ff[gi][ok]) > 0
    pad = int(np.ceil(r / step))
    yy, xx = np.mgrid[-pad:pad + 1, -pad:pad + 1]
    thick = ndimage.binary_opening(np.pad(R, pad, mode="edge"), structure=xx ** 2 + yy ** 2 <= pad ** 2)[pad:-pad, pad:-pad]
    lab, n = ndimage.label(thick)                          # a region: at least area square decades, or cut off by the
    area_ = ndimage.sum(thick, lab, range(1, n + 1)) * step ** 2   # top of the scan; smaller patches go with the bands
    top = set(lab[-1][lab[-1] > 0])
    thick = np.isin(lab, [k + 1 for k in range(n) if area_[k] >= area or k + 1 in top])
    Fs = F.copy()
    nodes = []
    for k in range(len(lx)):
        ex = np.asarray(F[:, k].filled(-1.0)) > 0
        edges = np.flatnonzero(np.diff(np.r_[False, ex, False].astype(int)))
        kr = int(round((lx[k] - lx[0]) / step))
        col = []
        for i0, i1 in zip(edges[::2], edges[1::2] - 1):
            rr = np.flatnonzero((G[rows] >= G[i0]) & (G[rows] <= G[i1]))
            if len(rr) and thick[rr, kr].any():
                continue                                         # part of a region: shaded as before
            Fs[i0:i1 + 1, k] = -STEP
            sel = np.arange(i0, i1 + 1)[C[i0:i1 + 1, k]]
            if len(sel):
                if col and G[sel[0]] - col[-1][1] < 0.2:           # runs a hair apart: one band
                    col[-1] = (col[-1][0], G[sel[-1]])
                else:
                    col.append((G[sel[0]], G[sel[-1]]))
        nodes.append(col)
    done, active = [], []
    for k, col in enumerate(nodes):
        used, keep = set(), []
        for ch in active:
            lo, hi = ch["last"]
            cand = [n for n in col if n not in used and n[0] <= hi + link and n[1] >= lo - link]
            if cand:
                n = min(cand, key=lambda n: abs(n[0] + n[1] - lo - hi))
                used.add(n)
                ch["pts"].append((lx[k], n[0], n[1]))
                ch["last"] = n
                keep.append(ch)
            elif C[(G >= lo - link) & (G <= hi + link), k].all():
                done.append(ch["pts"])                           # sampled here and no band: it ends
            else:
                keep.append(ch)                                  # not sampled here: carry on
        active = keep + [dict(pts=[(lx[k], n[0], n[1])], last=n) for n in col if n not in used]
    return Fs, [np.array(p) for p in done + [ch["pts"] for ch in active] if len(p) > 1]   # seen at one mass: dropped


INSET = ((-25, -15), (-28.5, -24))   # log10 m, log10 g: the thin bands' flat stretch, enlarged


def draw(ax, lx, G, F, ribbons, color, lw=1.8):
    ax.contourf(lx, G, F, levels=[0, 2 * D], colors=[color], alpha=0.22)
    ax.contour(lx, G, F, levels=[0], colors=[color], linewidths=lw)
    for r in ribbons:
        xs, ys = np.r_[r[:, 0], r[::-1, 0], r[0, 0]], np.r_[r[:, 2], r[::-1, 1], r[0, 2]]
        ax.fill(xs, ys, color=color, alpha=0.5, lw=0)
        ax.plot(xs, ys, color=color, lw=0.6 * lw)


def figure(suffix, comp_title, out, bands=False):
    """The exclusion figure (7 models, thin bands left out); bands: only the ultralight planes, with their thin bands
    as ribbons and an inset on their flat stretch."""
    models = MODELS[:2] if bands else MODELS
    nr, nc = (1, 3) if bands else (2, 4)
    fig, axes = plt.subplots(nr, nc, figsize=(nc * PANEL[0], nr * PANEL[1] + 0.5))
    for ax, (model, title) in zip(axes.flat, models):
        yname, drawn = None, []
        for ds, label, color in SETS:
            hits = glob.glob(f"{REPO}/output/analysis/campaign_ml/{ds}/{model}{suffix}/*/plane_*.json")
            if not hits:
                continue
            lx, G, F, yname, C = field(hits[0])
            ribbons = []
            if model.endswith("_ul"):
                F, ribbons = thin_bands(lx, G, F, C)
            draw(ax, lx, G, F, ribbons if bands else [], color)
            drawn.append((lx, G, F, ribbons, color))
        if bands:                                        # the thin bands, enlarged
            ins = ax.inset_axes([0.57, 0.06, 0.41, 0.38])
            for d in drawn:
                draw(ins, *d, lw=1.2)
            ins.set_xlim(*INSET[0])
            ins.set_ylim(*INSET[1])
            ins.set_xticks([-24, -20, -16])
            ins.set_yticks([-28, -26, -24])
            ins.tick_params(labelsize=13, pad=3, length=5)
            ax.indicate_inset_zoom(ins, edgecolor="0.45")
        if model.endswith("_mev"):
            ax.set_ylim(-6, -2)                          # nothing is excluded below ~1e-5 in the MeV planes
        ax.set_xlabel(r"$\log_{10}(m_{Z_D}$ / eV$)$")
        ax.set_ylabel(r"$\log_{10}$ " + YLABEL.get(yname, yname or "coupling"))
        ax.set_title(title)
    leg_ax = axes.flat[-1]
    leg_ax.axis("off")
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=lab) for _, lab, c in SETS]
    leg_ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.62), frameon=False,
                  title=f"Excluded at 95{pct()}\n(shaded; per mass, " + r"$\Delta\chi^2 \geq 3.84$)",
                  title_fontsize=plt.rcParams["legend.fontsize"])
    if suffix and not bands:
        leg_ax.text(0.5, 0.1, "Not shown: thin excluded bands along the edge of the\npion-decay region in the "
                    "ultralight planes, which\nexist only for energy-independent source compositions.",
                    ha="center", va="center", fontsize=12, color="0.35", transform=leg_ax.transAxes)
    fig.suptitle((f"Thin excluded bands, {comp_title} (ribbons at their true width; inset enlarged)" if bands else
                  f"Campaign of 2026-10-20, {comp_title}: excluded couplings per mediator mass; mixing fixed, lightest "
                  "mass profiled, relics thermal in every collision"))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


JOINED = [("le_lmu", r"$L_e - L_\mu$"), ("le_ltau", r"$L_e - L_\tau$")]
MEV_CORNER = ((4, 9), (-6, -2))   # log10 m, log10 g: where the MeV planes' structure lies, enlarged in an inset


def joined(suffix, comp_title, out):
    """L_e-L_mu and L_e-L_tau, each in one wide panel: the ultralight plane (1e-30 to 1e4 eV) and the MeV plane
    (1e4 to 1e9 eV) side by side, the thin bands left out, with an inset on the MeV planes' corner."""
    fig, axes = plt.subplots(2, 1, figsize=(3 * PANEL[0], 2 * PANEL[1] + 1.0))
    for ax, (model, title) in zip(axes, JOINED):
        drawn = []
        for ds, label, color in SETS:
            for kind in ("_ul", "_mev"):
                hits = glob.glob(f"{REPO}/output/analysis/campaign_ml/{ds}/{model}{kind}{suffix}/*/plane_*.json")
                if not hits:
                    continue
                lx, G, F, _, C = field(hits[0])
                if kind == "_ul":
                    F, _ = thin_bands(lx, G, F, C)
                draw(ax, lx, G, F, [], color)
                drawn.append((lx, G, F, [], color))
        ins = ax.inset_axes([0.62, 0.08, 0.36, 0.5])
        for d in drawn:
            draw(ins, *d, lw=1.4)
        ins.set_xlim(*MEV_CORNER[0])
        ins.set_ylim(*MEV_CORNER[1])
        ins.tick_params(labelsize=14, pad=4, length=6)
        ax.indicate_inset_zoom(ins, edgecolor="0.45")
        ax.set_xlim(-30, 9)
        ax.set_ylim(-32, -2)
        ax.set_xlabel(r"$\log_{10}(m_{Z_D}$ / eV$)$")
        ax.set_ylabel(r"$\log_{10} g_D$")
        ax.text(0.015, 0.94, title, transform=ax.transAxes, ha="left", va="top", fontsize=22)
    handles = [Patch(facecolor=c, edgecolor=c, alpha=0.5, label=lab) for _, lab, c in SETS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               title=f"Excluded at 95{pct()} ({comp_title}; per mass, " + r"$\Delta\chi^2 \geq 3.84$)",
               title_fontsize=plt.rcParams["legend.fontsize"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def main(prefix):
    use()
    for suffix, tag, comp_title in COMPS:
        figure(suffix, comp_title, f"{prefix}_excluded_{tag}.png")
        joined(suffix, comp_title, f"{prefix}_joined_{tag}.png")
    figure(COMPS[1][0], COMPS[1][2], f"{prefix}_thin_bands.png", bands=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "output", "figures", "campaign_ml"))
