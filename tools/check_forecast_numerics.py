"""
check_forecast_numerics.py -- review item 9 (2026-11-11 physics review; his go 2026-11-12, "check first"): are the
solver's grids, set against today's data, fine enough for the forecasts' statistics (1e4-1e5 events)? At the 95% edge
of chosen mass columns of one finished forecast plane, the coupling 0.3 decade above it, and the column's smallest
coupling (no effect, the reference), the points are refitted (the campaign's own fitting code, mass and coupling held)
with the solver's grids as in the campaign ('standard') or all refined twice over ('doubled': twice the energy and
redshift bins, Sun paths, Earth chords, steps inside the Earth, sky resolution and secret-interaction redshift
substeps; half the transit angle step). The Asimov events are recomputed with the same grids. The grids are fine
enough if Delta chi^2 (point minus reference) agrees to 0.1 between the two.
Review item 2 (his go 2026-11-12, "measure first"): 'mirrored' instead gives the off-pole telescopes the extragalactic
flux mirrored north to south (data.events.future.offpole_earth; analysis/future_events.py), against 'standard' the
South Pole's crossing of the Earth at every site; the two bracket the error of the South Pole's crossing elsewhere.
Points either at the edges of --columns of the finished plane (its output, or --plane-dir), or given as --points
'm:log10g,...' (each with the reference, the smallest coupling); fits start from the nearest stored point.
Written to <plane output>/asimov/forecast_numerics.<grid>.jsonl.
  python tools/check_forecast_numerics.py <config.json> <standard | doubled | mirrored> --columns 17.78e6,23.71e6
  python tools/check_forecast_numerics.py <config.json> mirrored --points 1e-17:-25.25,1e-16:-25.29 \
      --plane-dir <old plane's asimov directory> [--workers N]
"""
import argparse
import copy
import glob
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
from check_edge_islands import load  # noqa: E402

REF_G = 1e-32                  # the planes' smallest coupling (no effect)


def stored(plane_dir):
    """{(m, g): best record} from a plane's jsonl (finished or not)."""
    best = {}
    for f in glob.glob(f"{plane_dir}/plane_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("chi2") is None or not np.isfinite(r["chi2"]) or not r.get("values"):
                continue
            k = tuple(float(v) for v in r["held"].values())
            if k not in best or r["chi2"] < best[k]["chi2"]:
                best[k] = r
    return best


def nearest(best, m, g):
    k = min(best, key=lambda k: abs(np.log10(k[0] / m)) * 10 + abs(np.log10(k[1] / g)))
    return best[k]["values"]

DCHI2 = 3.84


def doubled(raw):
    raw = copy.deepcopy(raw)
    s = raw["solver"]
    g = s["grid"]
    for k in ("n_E_bins", "n_z_bins", "n_sun_paths", "n_earth_chords", "N_steps_earth"):
        if k in g:
            g[k] = 2 * int(g[k])
    if "transit_dtheta" in g:
        g["transit_dtheta"] = 0.5 * float(g["transit_dtheta"])
    if "nside" in s.get("sky", {}):
        s["sky"]["nside"] = 2 * int(s["sky"]["nside"])
    if "z_substeps" in s.get("secret_interactions", {}):
        s["secret_interactions"]["z_substeps"] = 2 * int(s["secret_interactions"]["z_substeps"])
    return raw


def mirrored(raw):
    raw = copy.deepcopy(raw)
    raw["data"]["events"]["future"]["offpole_earth"] = "mirrored"
    return raw


def edge(ly, col):
    """log10 g where Delta chi^2 first rises through 3.84 (linear in log10 g), or None."""
    d = col - np.nanmin(col)
    for j in range(len(ly) - 1):
        if np.isfinite(d[j]) and np.isfinite(d[j + 1]) and d[j] < DCHI2 <= d[j + 1]:
            return ly[j] + (DCHI2 - d[j]) / (d[j + 1] - d[j]) * (ly[j + 1] - ly[j])
    return None


def run(task):
    from analysis import scans
    label, held, start, settings = task
    r = scans._fit_held(held, [start], settings)
    return label, held, float(r.chi2), r.values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("grid", choices=("standard", "doubled", "mirrored"))
    ap.add_argument("--columns", help="masses (in the plane's x unit), comma-separated")
    ap.add_argument("--points", help="'m:log10g' pairs, comma-separated (instead of --columns)")
    ap.add_argument("--plane-dir", help="the stored plane for the edges and the starts (default: the config's)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    cfgp = os.path.join(REPO, a.config)
    raw0 = json.load(open(cfgp))
    raw = {"standard": lambda r: r, "doubled": doubled, "mirrored": mirrored}[a.grid](raw0)
    cfg = from_dict(raw, cfgp)
    dc = asimov_centre(cfg)
    out_dir = os.path.join(REPO, raw0["output_dir"], "asimov")
    os.makedirs(out_dir, exist_ok=True)
    settings = {**asdict(cfg.fit), "n_random": 0}
    tasks = []
    if a.points:
        xn, yn = raw0["planes"][0]["x"], raw0["planes"][0]["y"]
        best = stored(a.plane_dir or out_dir)
        for pair in a.points.split(","):
            m, lg = (float(v) for v in pair.split(":"))
            for label, g in (("edge", 10 ** lg), ("reference", REF_G)):
                tasks.append((f"{m:.4g}|{label}", {xn: m, yn: g}, nearest(best, m, g), settings))
    s, x, y, chi, best = load(a.plane_dir or out_dir) if not a.points else (None,) * 5
    for m in (float(v) for v in (a.columns.split(",") if a.columns and not a.points else [])):
        xn, yn = s["names"]
        lx, ly = np.log10(x), np.log10(y)
        i = int(np.argmin(abs(lx - np.log10(m))))
        e = edge(ly, chi[i])
        if e is None:
            print(f"column {x[i]:.4g}: no 95% edge, skipped", flush=True)
            continue
        for label, lg in (("edge", e), ("above", e + 0.3), ("reference", ly[0])):
            j = int(np.argmin(abs(ly - lg)))
            g = float(10 ** lg) if label != "reference" else float(y[0])
            tasks.append((f"{x[i]:.4g}|{label}", {xn: float(x[i]), yn: g}, best[(i, j)]["values"], settings))
    print(f"{raw0['name']}, {a.grid} grids: {len(tasks)} fits on {min(a.workers, len(tasks))} workers", flush=True)
    path = f"{out_dir}/forecast_numerics.{a.grid}.jsonl"
    with get_context("spawn").Pool(min(a.workers, len(tasks)), initializer=scans._init_worker,
                                   initargs=(raw, cfgp, dc)) as pool, open(path, "w") as f:
        for label, held, chi2, values in pool.imap_unordered(run, tasks):
            f.write(json.dumps(dict(label=label, held=held, chi2=chi2, values=values)) + "\n")
            f.flush()
            print(label, round(chi2, 4), flush=True)
    print("done", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
