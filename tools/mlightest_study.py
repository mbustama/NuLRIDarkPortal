"""
mlightest_study.py -- how much the lightest neutrino mass moves a plane's chi^2 near its limit (2026-10-18), to choose
how the new campaign profiles it (Mauricio: "vary the lightest neutrino mass penalized with cosmological upper limits
on sum of neutrino masses").

For a plane configuration and its finished plane from an earlier campaign (the summary's Delta chi^2 grid), takes the
points where each mass column crosses the threshold (Delta chi^2 = 3.84, the coupling interpolated in log), and refits
the plane's free parameters there (mass and coupling held) at each lightest mass of --m-lightest, with the mixing fixed.
Prints and writes (JSON) chi^2 - chi^2_ref per point and lightest mass, chi^2_ref the fit at the same point with the
coupling 1e-12 times smaller (no new physics), and the mass sum per lightest mass, for any penalty to be added after.

  python tools/mlightest_study.py <config.json> <old plane summary .json> [--m-lightest 1e-5 1e-3 ...]
                                  [--masses 6] [--workers N] [--out results.json]
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

M_LIGHTEST = (1e-5, 1e-3, 3e-3, 6e-3, 0.01, 0.02, 0.03, 0.045)
THRESHOLD = 3.84


def contour_points(summary: dict, n: int):
    """(x, y) where Delta chi^2 crosses THRESHOLD going up in y, for up to n x columns spread over those that cross."""
    xs, ys = (np.asarray(g, dtype=float) for g in summary["grids"])
    d = np.array([[np.nan if v is None else v for v in row] for row in summary["dchi2"]], dtype=float)
    cols = []
    for i in range(len(xs)):
        col = d[i]
        for j in range(len(ys) - 1):
            if np.isfinite(col[j]) and np.isfinite(col[j + 1]) and col[j] < THRESHOLD <= col[j + 1]:
                t = (THRESHOLD - col[j]) / (col[j + 1] - col[j])
                cols.append((xs[i], float(10 ** (np.log10(ys[j]) + t * (np.log10(ys[j + 1]) - np.log10(ys[j]))))))
                break
    if len(cols) > n:
        cols = [cols[int(round(k))] for k in np.linspace(0, len(cols) - 1, n)]
    return cols


def _task(args):
    raw, path, x, y, held, m_l, ref = args
    from analysis import config as config_mod
    from analysis.fitting import fit
    r = copy.deepcopy(raw)
    r["solver"].setdefault("cosmology", {})["m_lightest_eV"] = float(m_l)
    r["priors"] = [p for p in r.get("priors", []) if p.get("name") != "sum_masses_eV"]
    cfg = config_mod.from_dict(r, path)
    centre = None
    from analysis.run import asimov_centre, asimov_only
    if asimov_only(cfg):
        centre = asimov_centre(cfg)
    obj = cfg.objective(data_centre=centre)
    vals = {x: held[0], y: held[1] * (1e-12 if ref else 1.0)}
    pset = cfg.parameter_set().holding(vals)
    res = fit(obj, pset, starts=[pset.start_values()], settings=cfg.fit)
    from analysis.model import sum_of_masses
    return dict(x=held[0], y=held[1], m_lightest=float(m_l), ref=ref, chi2=float(res.chi2),
                sum_masses=float(sum_of_masses({}, obj.model.base)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("summary")
    ap.add_argument("--m-lightest", type=float, nargs="+", default=list(M_LIGHTEST))
    ap.add_argument("--masses", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    raw = json.load(open(a.config))
    summary = json.load(open(a.summary))
    x, y = summary["names"]
    pts = contour_points(summary, a.masses)
    tasks = [(raw, os.path.abspath(a.config), x, y, p, m, False) for p in pts for m in a.m_lightest]
    tasks += [(raw, os.path.abspath(a.config), x, y, p, a.m_lightest[0], True) for p in pts]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        out = pool.map(_task, tasks, chunksize=1)
    refs = {(r["x"], r["y"]): r["chi2"] for r in out if r["ref"]}
    print(f"{raw['name']}: chi^2 - chi^2(no new physics) at the earlier plane's threshold crossings")
    print("  m_lightest:  " + "  ".join(f"{m:8.3g}" for m in a.m_lightest))
    print("  sum (eV):    " + "  ".join(f"{next(r['sum_masses'] for r in out if r['m_lightest'] == m and not r['ref']):8.4f}"
                                     for m in a.m_lightest))
    for p in pts:
        row = [next(r["chi2"] for r in out if (r["x"], r["y"]) == p and r["m_lightest"] == m and not r["ref"])
               - refs[p] for m in a.m_lightest]
        print(f"  {x}={p[0]:.3g}, {y}={p[1]:.3g}: " + "  ".join(f"{v:8.3f}" for v in row))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(config=a.config, summary=a.summary, points=pts, results=out), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
