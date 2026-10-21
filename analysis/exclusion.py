"""
exclusion.py

The excluded regions of the campaign's planes, as the figures draw them: read from the plane summaries (run.summarize)
and the refinement records next to them, per mass column, at 95% (per-mass Delta chi^2 >= 3.84). What a figure shows
is computed here, and tested (tests/test_exclusion.py); how it looks is set where it is drawn (notebooks/
paper_figures.ipynb for the paper's figures, tools/plot_campaign_ml.py for the diagnostic ones).

The reading, column by column (one mediator mass):
  - Delta chi^2 from the column's minimum (the refinement included), excluded where >= 3.84;
  - upper limits only: couplings below the column's lowest allowed one are not excluded (a column whose no-effect
    couplings are themselves disfavoured, where the data prefer an effect, would otherwise read as excluded there);
  - isolated specks -- excluded grid regions of at most three points, none above 5 -- not excluded (a hair over 3.84,
    they flip with the sampling);
  - columns whose smallest coupling is not at the no-effect chi^2 left out (none in the campaign); the summaries'
    'elevated' flag is not used: in planes whose data prefer an effect somewhere it fires at every other mass;
  - the crossings interpolated linearly in log10 g between samples, and the field contoured is the signed distance to
    the nearest crossing (field): contouring it joins each crossing to the next mass's in a straight line, without the
    staircase that contouring Delta chi^2 itself draws when it jumps from ~0 to hundreds within one coupling step.
With the source compositions fitted, the ultralight planes also exclude a thin band (~0.5 decade) along the edge of the
pion-decay region; it exists only because each source's composition is taken to be the same at every energy, and the
figures leave it out (thin_bands; notes, 'Thin excluded bands with fitted source compositions').
"""
import glob
import json
import os
from typing import List, Tuple

import numpy as np

D = 1.5        # decades: the signed distance to the 95% edge is clipped here (see field)
STEP = 0.005   # decades: the coupling resolution of the field
THRESHOLD = 3.84


def signed(G, excl):
    """Signed distance in log10 g from each G to the nearest edge of the excluded set (+ inside), clipped at +-D."""
    edges = 0.5 * (G[1:] + G[:-1])[excl[1:] != excl[:-1]]
    if len(edges) == 0:
        dist = np.full(len(G), D)
    else:
        k = np.clip(np.searchsorted(edges, G), 1, len(edges))
        dist = np.minimum(abs(G - edges[k - 1]), abs(G - edges[np.minimum(k, len(edges) - 1)]))
    return np.where(excl, 1.0, -1.0) * np.minimum(dist, D)


def _records(paths):
    for f in paths:
        for line in open(f):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def field(path):
    """(log10 m columns, log10 g, signed-distance field [g, m] (masked where a column is left out), y name, sampled
    [g, m]) of the plane summary at `path`, with the refinement records in its directory
    (tools/refine_ridges.py, refine_edges.py, refine_grid.py: at grid masses, merged into their columns, also above
    the grid's largest coupling, which then extends the field; between them, partial columns:
    their own points inside each sampled window -- runs of couplings at most 0.25 decade apart -- with the minimum of the
    neighbouring full columns, and the neighbouring columns' field interpolated in log10 m outside). `sampled` marks
    where a column holds samples (all of a grid column, the windows of a partial one)."""
    from scipy import ndimage
    s = json.load(open(path))
    here = os.path.dirname(path)
    x = np.array(s["grids"][0])
    y = np.array(s["grids"][1])
    xn, yn = s["names"]
    grid_recs = list(_records(glob.glob(os.path.join(here, "refine_ridges.*.jsonl"))
                              + glob.glob(os.path.join(here, "refine_edges.*.jsonl"))
                              + glob.glob(os.path.join(here, "refine_grid.*.jsonl"))))
    between_recs = list(_records(glob.glob(os.path.join(here, "refine_ridges_between.*.jsonl"))
                                 + glob.glob(os.path.join(here, "refine_edges_between.*.jsonl"))
                                 + glob.glob(os.path.join(here, "refine_grid_between.*.jsonl"))))
    g_top = max([np.log10(y[-1])] + [np.log10(r["held"][yn]) for r in grid_recs + between_recs])
    G = np.arange(np.log10(y[0]), g_top + STEP / 2, STEP)      # the refinement may reach above the grid
    chi = np.array([[np.nan if v is None else v for v in row] for row in s["chi2"]], dtype=float)
    chi0 = chi.copy()
    d = chi - np.nanmin(chi, axis=1, keepdims=True)
    lab, n = ndimage.label(d >= THRESHOLD)
    for k in range(1, n + 1):
        sel = lab == k
        if sel.sum() <= 3 and np.nanmax(d[sel]) < 5.0:
            chi[sel] = np.nan                                   # a speck: shown as allowed below
    cols = {i: [(float(y[j]), float(chi[i, j])) for j in range(len(y)) if np.isfinite(chi[i, j])] for i in range(len(x))}
    for r in grid_recs:
        cols[int(np.argmin(np.abs(np.log(x / r["held"][xn]))))].append((r["held"][yn], r["chi2"]))
    left_out = {i for i in range(len(x)) if not abs(chi0[i, 0] - np.nanmedian(chi0[:, 0])) < 0.3}
    S, low, cov = {}, {}, {}
    for i, pts in cols.items():
        if i in left_out or not pts:
            continue
        specks = [(float(y[j]), 0.0) for j in range(len(y)) if np.isnan(chi[i, j]) and np.isfinite(d[i, j])]
        pts.sort()
        g = np.log10([p[0] for p in pts])
        c = np.array([p[1] for p in pts])
        dc = c - min(c.min(), np.nanmin(chi0[i]))              # from the column's minimum, the refinement included
        ok = dc < THRESHOLD
        if ok.any():
            dc[:int(np.argmax(ok))] = 0.0                       # below the lowest allowed coupling: not an exclusion
            low[i] = g[int(np.argmax(ok))]
        pts = sorted(list(zip(g, dc)) + [(np.log10(gs), v) for gs, v in specks])
        S[i] = signed(G, np.interp(G, [p[0] for p in pts], [p[1] for p in pts]) >= THRESHOLD)
        cov[np.log10(x[i])] = np.ones(len(G), bool)
    xs = {np.log10(x[i]): S.get(i) for i in range(len(x))}     # None: left out
    between = {}
    for r in between_recs:
        between.setdefault(round(np.log10(r["held"][xn]), 6), []).append((r["held"][yn], r["chi2"]))
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
        for run in np.split(np.arange(len(g)), np.flatnonzero(np.diff(g) > 0.25) + 1):
            win = (G >= g[run[0]]) & (G <= g[run[-1]])
            excl[win] = np.interp(G[win], g[run], dc[run]) >= THRESHOLD
            cov[lm] |= win
        if i in low and i + 1 in low:
            excl &= G >= (1 - t) * low[i] + t * low[i + 1]
        xs[lm] = signed(G, excl)
    lx = np.array(sorted(xs))
    F = np.ma.masked_invalid(np.array([xs[v] if xs[v] is not None else np.full(len(G), np.nan) for v in lx]).T)
    C = np.array([cov.get(v, np.zeros(len(G), bool)) for v in lx]).T
    return lx, G, F, yn, C


def thin_bands(lx, G, F, C, r=0.4, link=0.6, step=0.05, area=3.0) -> Tuple[np.ma.MaskedArray, List[np.ndarray]]:
    """The thin excluded bands of a plane, taken out of its field. Thin: no disk of radius r decades (in log m and log g
    alike) fits inside it -- the excluded set rasterised every `step` decades and opened with that disk; what the opening
    removes is thin (the bands are ~0.5 decade across, the regions far more), and so is what it leaves in patches
    smaller than `area` square decades that do not reach the top of the scan (where a band widens). Per mass, a thin
    excluded run of couplings is taken out of the field and, where the column was sampled there (C), its lower and upper
    edges become a node (runs less than 0.2 decade apart: one node). Nodes of neighbouring masses whose runs overlap
    within `link` decades are joined; a band ends at a sampled mass without a run near it (a real gap), not at a mass
    that was not sampled there; a band seen at one mass only is dropped. Returns (the field without the bands,
    [(log m, lower log g, upper log g) ribbons])."""
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
    thick = ndimage.binary_opening(np.pad(R, pad, mode="edge"), structure=xx ** 2 + yy ** 2 <= pad ** 2)[pad:-pad,
                                                                                                         pad:-pad]
    lab, n = ndimage.label(thick)
    areas = ndimage.sum(thick, lab, range(1, n + 1)) * step ** 2
    top = set(lab[-1][lab[-1] > 0])
    thick = np.isin(lab, [k + 1 for k in range(n) if areas[k] >= area or k + 1 in top])
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
                continue                                         # part of a region: kept
            Fs[i0:i1 + 1, k] = -STEP
            sel = np.arange(i0, i1 + 1)[C[i0:i1 + 1, k]]
            if len(sel):
                if col and G[sel[0]] - col[-1][1] < 0.2:
                    col[-1] = (col[-1][0], G[sel[-1]])
                else:
                    col.append((G[sel[0]], G[sel[-1]]))
        nodes.append(col)
    done, active = [], []
    for k, col in enumerate(nodes):
        used, keep = set(), []
        for ch in active:
            lo, hi = ch["last"]
            cand = [nd for nd in col if nd not in used and nd[0] <= hi + link and nd[1] >= lo - link]
            if cand:
                nd = min(cand, key=lambda q: abs(q[0] + q[1] - lo - hi))
                used.add(nd)
                ch["pts"].append((lx[k], nd[0], nd[1]))
                ch["last"] = nd
                keep.append(ch)
            elif C[(G >= lo - link) & (G <= hi + link), k].all():
                done.append(ch["pts"])                           # sampled here and no band: it ends
            else:
                keep.append(ch)                                  # not sampled here: carry on
        active = keep + [dict(pts=[(lx[k], nd[0], nd[1])], last=nd) for nd in col if nd not in used]
    return Fs, [np.array(p) for p in done + [ch["pts"] for ch in active] if len(p) > 1]


def merged(ul_path, mev_path):
    """One field for a model's ultralight and MeV planes (contoured once, so that no seam shows where they meet, at
    1e4 eV): the ultralight plane's, its thin bands out, and the MeV plane's on the ultralight couplings, allowed below
    the MeV plane's smallest coupling (no effect there); at the mass both have, the MeV plane's column (finer in
    coupling). Returns (log10 m columns, log10 g, field)."""
    lu, G, Fu, _, Cu = field(ul_path)
    Fu, _ = thin_bands(lu, G, Fu, Cu)
    lm, Gm, Fm, _, _ = field(mev_path)
    Fm = np.ma.masked_invalid(np.array([np.interp(G, Gm, Fm[:, k].filled(np.nan), left=-D, right=-D)
                                        for k in range(len(lm))]).T)
    keep = lu < lm[0] - 1e-9
    return np.r_[lu[keep], lm], G, np.ma.concatenate([Fu[:, keep], Fm], axis=1)


def plane(campaign_dir: str, dataset: str, name: str) -> str:
    """The summary of plane `name` (e.g. 'le_lmu_ul_fx') of `dataset` under `campaign_dir`, or '' if not there."""
    hits = glob.glob(os.path.join(campaign_dir, dataset, name, "*", "plane_*.json"))
    return hits[0] if hits else ""


def draw(ax, lx, G, F, color, fill, lw=1.8, ribbons=()):
    """A data set's excluded region on `ax`: an opaque fill (no transparency, which mixes overlapping colours; draw the
    largest region first) and its edge in `color` above every fill; optional ribbons (thin_bands) likewise."""
    ax.contourf(lx, G, F, levels=[0, 2 * D], colors=[fill], zorder=1)
    ax.contour(lx, G, F, levels=[0], colors=[color], linewidths=lw, zorder=5)
    for r in ribbons:
        xs, ys = np.r_[r[:, 0], r[::-1, 0], r[0, 0]], np.r_[r[:, 2], r[::-1, 1], r[0, 2]]
        ax.fill(xs, ys, color=fill, lw=0, zorder=1)
        ax.plot(xs, ys, color=color, lw=0.6 * lw, zorder=5)
