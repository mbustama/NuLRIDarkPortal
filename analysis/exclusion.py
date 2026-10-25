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
OWN_COLUMN_FROM = -4.95  # log10 g: a column between grid masses sampled from here down reads its own minimum (field)


def signed(G, excl):
    """Signed distance in log10 g from each G to the nearest edge of the excluded set (+ inside), clipped at +-D."""
    edges = 0.5 * (G[1:] + G[:-1])[excl[1:] != excl[:-1]]
    if len(edges) == 0:
        dist = np.full(len(G), D)
    else:
        k = np.clip(np.searchsorted(edges, G), 1, len(edges))
        dist = np.minimum(abs(G - edges[k - 1]), abs(G - edges[np.minimum(k, len(edges) - 1)]))
    return np.where(excl, 1.0, -1.0) * np.minimum(dist, D)


def _records(paths, corrected=None, xn=None, yn=None):
    """The records of the files, each with its chi2 replaced by the corrected one where there is one."""
    for f in paths:
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if corrected:
                c = corrected.get((os.path.basename(f), _key(r["held"], xn, yn)))
                if c is not None:
                    r["chi2"] = c
            yield r


def _key(held, xn, yn):
    return round(float(np.log10(held[xn])), 6), round(float(np.log10(held[yn])), 6)


def corrected(here, xn, yn):
    """The corrections to the stored solutions. The thermal kernels' grid fix (2026-10-21; tools/recheck_thermal_grid.py,
    notes 'The thermal kernels' grid'):
    {(source, (log10 m, log10 g)): chi2} -- every stored solution that went through dp_si, re-evaluated with the fixed
    code (recheck.*.jsonl, the fixed grid's records; a solution whose stored chi2 was missing, a failed fit, only
    through a refit), a refit (refine_recheck.*.jsonl) taking over where it went lower. Source: 'grid' for the
    summary's points, else the record file's name. Then the relic-scattering reruns (rerun_rs.*.jsonl), which replace
    the value at their points. Empty without either."""
    out = {}
    for r in _records(glob.glob(os.path.join(here, "recheck.*.jsonl"))):
        st = r.get("stored")
        if r.get("gap") == 0.5 and r.get("chi2") is not None and st is not None and st == st:
            out[(r["source"], _key(r["held"], xn, yn))] = float(r["chi2"])
    for r in _records(glob.glob(os.path.join(here, "refine_recheck.*.jsonl"))):
        if r.get("chi2") is not None:
            k = (r["source"], _key(r["held"], xn, yn))
            out[k] = min(out.get(k, np.inf), float(r["chi2"]))
    # the relic-scattering reruns (2026-10-22; tools/rerun_relic_scattering.py): new physics at those points, so they
    # take over whatever their value
    for r in _records(glob.glob(os.path.join(here, "rerun_rs.*.jsonl"))):
        if r.get("chi2") is not None:
            out[(r["source"], _key(r["held"], xn, yn))] = float(r["chi2"])
    # HESE + DR2's chi^2 with the nuisance profile from fixed starts (2026-10-24; tools/reeval_events.py): the lowest
    # of the profile's minima, so it takes over where lower than the value it was evaluated against
    for r in _records(glob.glob(os.path.join(here, "reeval_events.*.jsonl"))):
        if r.get("chi2") is not None and r.get("stored") is not None:
            k = (r["source"], _key(r["held"], xn, yn))
            out[k] = min(out.get(k, float(r["stored"])), float(r["stored"]), float(r["chi2"]))
    return out


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
    corr = corrected(here, xn, yn)
    grid_recs = list(_records(glob.glob(os.path.join(here, "refine_ridges.*.jsonl"))
                              + glob.glob(os.path.join(here, "refine_edges.*.jsonl"))
                              + glob.glob(os.path.join(here, "refine_grid.*.jsonl")), corr, xn, yn))
    between_recs = list(_records(glob.glob(os.path.join(here, "refine_ridges_between.*.jsonl"))
                                 + glob.glob(os.path.join(here, "refine_edges_between.*.jsonl"))
                                 + glob.glob(os.path.join(here, "refine_grid_between.*.jsonl")), corr, xn, yn))
    g_top = max([np.log10(y[-1])] + [np.log10(r["held"][yn]) for r in grid_recs + between_recs])
    G = np.arange(np.log10(y[0]), g_top + STEP / 2, STEP)      # the refinement may reach above the grid
    chi = np.array([[np.nan if v is None else v for v in row] for row in s["chi2"]], dtype=float)
    for (src, (lm, lg)), c in corr.items():                     # the summary's points, re-evaluated or refitted
        if src == "grid":
            i, j = int(np.argmin(abs(np.log10(x) - lm))), int(np.argmin(abs(np.log10(y) - lg)))
            if abs(np.log10(x[i]) - lm) < 1e-5 and abs(np.log10(y[j]) - lg) < 1e-5:
                chi[i, j] = c
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
    null = float(np.nanmedian(chi0[:, 0]))                     # the no-effect chi^2 (the grid's smallest coupling)
    between = {}
    for r in between_recs:
        between.setdefault(round(np.log10(r["held"][xn]), 6), []).append((r["held"][yn], r["chi2"]))
    for lm, pts in between.items():
        m = 10 ** lm
        i = int(np.searchsorted(x, m)) - 1
        if i < 0 or i + 1 >= len(x) or i not in S or i + 1 not in S:
            continue
        t = np.log(m / x[i]) / np.log(x[i + 1] / x[i])
        pts.sort()
        g = np.log10([p[0] for p in pts])
        c = np.array([p[1] for p in pts])
        own = g[0] <= OWN_COLUMN_FROM
        if own:
            # sampled from the small couplings up (refine_grid's new masses): the column's own minimum, with the
            # no-effect chi^2 (the same at every mass) for the couplings below its samples (2026-10-22: the neighbours'
            # minima, where the data prefer an effect at a mass-dependent depth, made whole columns read as excluded)
            cmin = min(c.min(), null)
        else:
            # a window high in the column (refine_ridges, refine_edges): the neighbouring full columns' minima
            cmin = min(np.nanmin(chi0[i]), np.nanmin(chi0[i + 1]), c.min())
        dc = c - cmin
        excl = (1 - t) * S[i] + t * S[i + 1] > 0
        cov[lm] = np.zeros(len(G), bool)
        for run in np.split(np.arange(len(g)), np.flatnonzero(np.diff(g) > 0.25) + 1):
            win = (G >= g[run[0]]) & (G <= g[run[-1]])
            excl[win] = np.interp(G[win], g[run], dc[run]) >= THRESHOLD
            cov[lm] |= win
        if own:
            # upper limits only, from the column's own samples: below its lowest allowed coupling nothing is excluded
            # (none of it if the no-effect couplings are allowed)
            if null - cmin >= THRESHOLD and (dc < THRESHOLD).any():
                excl &= G >= g[int(np.argmax(dc < THRESHOLD))]
        elif i in low and i + 1 in low:
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

def not_allowed_boundary(cfg, m_lightest=None):
    """(slope, intercept) of the line log10 g = log10 m + c above which a plane's model is not perturbative
    (2026-10-24; notes Sec. 12.8 item 3; Mauricio: marked, ranges kept): the longitudinal mode's Goldstone couplings
    Y = i (G'_ij m_i + G'_ji m_j) / m_Zd break perturbative unitarity (src/dp_unitarity.py), with the neutrino masses
    at the lightest mass m_lightest (eV; default the configuration's, 1e-6 eV: the narrowest region -- the couplings
    grow with the masses, c falls from 1.76 to 1.23 for the mu texture at 0.1 eV; the fits use only the lightest masses
    where the model holds, Mauricio 2026-10-24, so the region where none does is this line's). The coupling matrix is linear in the plane's
    coupling, so the measure scales as (g / m_Zd)^2: one evaluation fixes c = -log10 sqrt(r0), r0 the measure at
    g = m_Zd = 1 (checked at a second coupling). Returns None for a plane whose coupling does not set G linearly.
    cfg: a configuration file's path, a loaded configuration (analysis.config.AnalysisConfig) or its raw dict."""
    import copy
    import json
    import dark_portal_solver as dps
    import dp_absorption
    import dp_unitarity
    from dp_core import construct_PMNS
    from analysis import config as config_mod
    if isinstance(cfg, (str, os.PathLike)):
        raw, path = json.load(open(cfg)), str(cfg)
    elif hasattr(cfg, "raw"):
        raw, path = cfg.raw, getattr(cfg, "path", "")
    else:
        raw, path = cfg, ""
    raw = copy.deepcopy(dict(raw))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, path)
    model = cfg.model()
    pl = raw["planes"][0]

    def measure(y):
        v = {q.name: q.start for q in cfg.parameters}
        v.update({pl["x"]: 1.0, pl["y"]: y})
        if m_lightest is not None:
            v["m_lightest_eV"] = float(m_lightest)
        _, pr = dps.build_solver_params(model._run_cfg(model._overrides(v)))
        if pr.get("si"):
            # the secret interactions (2026-10-25): the vector's Goldstone couplings, Majorana or Dirac
            # (dp_si_light.goldstone_measure); the scalar has no such line (its relic field: not_modelled_measure)
            import dp_si_light
            si = pr["si"]
            if si["mediator"] != "vector":
                return None
            U = np.asarray(construct_PMNS(pr["th12"], pr["th13"], pr["th23"], pr["d_cp"]))
            return dp_si_light.goldstone_measure(si["g_flavor"], U, np.asarray(pr["m_nu_bg_arr"], dtype=np.float64),
                                                 si["m_phi_eV"], si["majorana"])
        G = pr.get("G_coupling")
        if G is None:
            G = dp_absorption.dark_portal_coupling(pr["g_D"], np.array([pr["u_e"], pr["u_mu"], pr["u_tau"]],
                                                                       dtype=np.complex128), 0.0)
        G = np.asarray(G, dtype=np.complex128)
        U = np.asarray(construct_PMNS(pr["th12"], pr["th13"], pr["th23"], pr["d_cp"]))
        c = G[0, 0].real
        Gp = U.conj().T @ (G - c * np.eye(3)) @ U + c * np.eye(3)
        return dp_unitarity.neutrino_unitarity(Gp, np.asarray(pr["m_nu_bg_arr"], dtype=np.float64), 1.0)

    r1, r2 = measure(1e-10), measure(1e-12)
    if r1 is None or r2 is None or r1 <= 0.0 or not np.isclose(r1 / r2, 1e4, rtol=1e-6):
        return None
    r0 = r1 / 1e-20
    return 1.0, -0.5 * np.log10(r0)


def not_modelled_measure(cfg):
    """For planes whose model fails where no single line marks it (2026-10-25: the secret-interaction scalar's background
    field sourced by the relics, Mauricio's decision (a); dp_si_light.scalar_field_measure), a callable
    f(x, y, m_lightest) -> the measure (>= 1: not modelled) at the plane's point (x, y) with that lightest mass; None
    for every other plane (their region is not_allowed_boundary's line). cfg as not_allowed_boundary's."""
    import copy
    import dark_portal_solver as dps
    import dp_si_light
    from dp_core import construct_PMNS
    from analysis import config as config_mod
    if isinstance(cfg, (str, os.PathLike)):
        raw, path = json.load(open(cfg)), str(cfg)
    elif hasattr(cfg, "raw"):
        raw, path = cfg.raw, getattr(cfg, "path", "")
    else:
        raw, path = cfg, ""
    raw = copy.deepcopy(dict(raw))
    si = (raw.get("solver", {}) or {}).get("secret_interactions", {}) or {}
    if not si.get("enabled", False) or str(si.get("mediator", "scalar")).strip().lower() != "scalar":
        return None
    raw.pop("exact_profile", None)
    acfg = config_mod.from_dict(raw, path)
    model = acfg.model()
    pl = raw["planes"][0]
    base = {q.name: q.start for q in acfg.parameters}

    def f(x, y, m_lightest=None):
        v = dict(base)
        v.update({pl["x"]: float(x), pl["y"]: float(y)})
        if m_lightest is not None:
            v["m_lightest_eV"] = float(m_lightest)
        _, pr = dps.build_solver_params(model._run_cfg(model._overrides(v)))
        sp = pr["si"]
        U = np.asarray(construct_PMNS(pr["th12"], pr["th13"], pr["th23"], pr["d_cp"]))
        return dp_si_light.scalar_field_measure(sp["g_flavor"], U, np.asarray(pr["m_nu_bg_arr"], dtype=np.float64),
                                                sp["m_phi_eV"], sp["majorana"], pr["T_nu0_eV"],
                                                (np.abs(U) ** 2).T @ np.asarray(pr["eta_bg"], dtype=np.float64),
                                                pr["H0_eV"], pr["Omega_m"], pr["Omega_L"], pr["z_max"], pr["dm21_sq"],
                                                decay_on=pr.get("neutrino_decay", True))
    return f


def not_modelled_mask(cfg, lx, ly, m_lightest=None):
    """(N_y, N_x) booleans: where the plane's model is not modelled at the lightest mass given (default the
    configuration's): not_allowed_boundary's line, or not_modelled_measure's region. lx, ly: log10 of the axes."""
    lx = np.asarray(lx, dtype=np.float64)
    ly = np.asarray(ly, dtype=np.float64)
    f = not_modelled_measure(cfg)
    if f is None:
        b = not_allowed_boundary(cfg, m_lightest)
        if b is None:
            return np.zeros((len(ly), len(lx)), dtype=bool)
        return ly[:, None] >= b[0] * lx[None, :] + b[1]
    return np.array([[f(10.0 ** x, 10.0 ** y, m_lightest) >= 1.0 for x in lx] for y in ly])


def draw_not_modelled(ax, lx, ly, mask, color="0.45"):
    """Hatch a not_modelled_mask region (cells of the grid lx x ly, log10 axes), as draw_not_allowed."""
    mask = np.asarray(mask, dtype=float)
    if not np.any(mask):
        return
    ax.contourf(lx, ly, mask, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=6)
    ax.contour(lx, ly, mask, levels=[0.5], colors=color, linewidths=0.8, zorder=6)


def draw_not_allowed(ax, lx, G, boundary, color="0.45"):
    """Hatch the region where the plane's model is not perturbative (not_allowed_boundary), over the plane's ranges."""
    if boundary is None:
        return
    slope, c = boundary
    lx = np.asarray(lx, dtype=np.float64)
    edge = slope * lx + c
    top = np.full_like(lx, max(G[-1], edge.max()) + 1.0)
    ax.fill_between(lx, np.clip(edge, G[0], None), top, facecolor="none", edgecolor=color, hatch="////", lw=0.0,
                    zorder=6)
    ax.plot(lx, edge, color=color, lw=0.8, zorder=6)
