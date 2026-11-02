"""
check_edge_islands.py -- are the small excluded islands of the fitted-composition ultralight planes (campaign_ml,
2026-10-20) real, or fits stuck in the wrong place? They sit on the edge of the pion-decay planes' excluded region,
where the best source composition changes abruptly (the IceCube events: electron fractions 0/0 below the edge, 1/1
above), so either would put them there. The points: excluded (per mass, above the lowest allowed coupling), outside
the plane's largest excluded region, 1e-26 <= m <= 1e-14 eV, g above the verified thin band at ~3e-28. Each refitted
(the campaign's own fitting code, mass and coupling held), one fit per start: its own solution, its four neighbours',
the pion-decay plane's (electron fractions 1/3), its own with the four corner compositions (0/0, 1/1, 1/0, 0/1), and its
own plus 8 random starts. Written to <plane output>/<data|asimov>/edge_islands.<part>.jsonl.
  python tools/check_edge_islands.py <future_lmu | future_ltau | events_lmu> <part> <parts> [--workers 16]
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
PLANES = {"future_lmu": ("proj_future", "le_lmu_ul_fx", "asimov"),
          "future_ltau": ("proj_future", "le_ltau_ul_fx", "asimov"),
          "events_lmu": ("now_events", "le_lmu_ul_fx", "data")}


def load(out_dir):
    s = json.load(open(glob.glob(f"{out_dir}/plane_*.json")[0]))
    best = {}
    for f in glob.glob(f"{out_dir}/plane_*.jsonl"):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "chi2" in r and (tuple(r["index"]) not in best or r["chi2"] < best[tuple(r["index"])]["chi2"]):
                best[tuple(r["index"])] = r
    chi = np.array([[np.nan if v is None else v for v in row] for row in s["chi2"]])
    return s, np.array(s["grids"][0]), np.array(s["grids"][1]), chi, best


def islands(x, y, chi):
    from scipy import ndimage
    d = chi - np.nanmin(chi, axis=1, keepdims=True)
    ex = d >= 3.84
    for i in range(len(x)):
        ok = np.flatnonzero(d[i] < 3.84)
        if len(ok):
            ex[i, :ok[0]] = False                                # upper limits only, as in plot_campaign_ml.py
    lab, n = ndimage.label(ex)
    main = 1 + int(np.argmax(ndimage.sum(ex, lab, range(1, n + 1))))
    lx, ly = np.log10(x), np.log10(y)
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(ex & (lab != main)))
            if -26 <= lx[i] <= -14 and ly[j] > -27.25]


def run(task):
    from analysis import scans
    label, index, held, starts, settings = task
    r = scans._fit_held(held, starts, settings)
    return label, index, held, float(r.chi2), r.values


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plane", choices=sorted(PLANES))
    ap.add_argument("part", type=int)
    ap.add_argument("parts", type=int)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    ds, name, kind = PLANES[a.plane]
    cfgp = os.path.join(REPO, f"analysis/configs/campaign_ml/{ds}/{name}.json")
    raw = json.load(open(cfgp))
    cfg = from_dict(raw, cfgp)
    dc = asimov_centre(cfg) if kind == "asimov" else None
    out_dir = os.path.join(REPO, raw["output_dir"], kind)
    s, x, y, chi, best = load(out_dir)
    _, _, _, _, best_pion = load(out_dir.replace(f"/{name}/", f"/{name[:-3]}/"))
    xn, yn = s["names"]
    base = {**asdict(cfg.fit), "n_random": 0}
    tasks = []
    for i, j in islands(x, y, chi):
        held = {xn: float(x[i]), yn: float(y[j])}
        own = best[(i, j)]["values"]
        starts = {"own": own}
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            starts[f"neighbour {di:+d},{dj:+d}"] = best.get((i + di, j + dj), {}).get("values")
        if (i, j) in best_pion:
            starts["pion plane"] = {**best_pion[(i, j)]["values"], "f_e_S": 1 / 3, "f_e_S_gal": 1 / 3}
        for fe, feg in ((0, 0), (1, 1), (1, 0), (0, 1)):
            starts[f"corner {fe}/{feg}"] = {**own, "f_e_S": float(fe), "f_e_S_gal": float(feg)}
        tasks += [(lab, (i, j), held, [st], base) for lab, st in starts.items() if st]
        tasks.append(("own + 8 random", (i, j), held, [own], {**base, "n_random": 8}))
    tasks = tasks[a.part::a.parts]
    print(f"{ds}/{name}: part {a.part} of {a.parts}, {len(tasks)} fits on {a.workers} workers", flush=True)
    path = f"{out_dir}/edge_islands.{a.part}.jsonl"
    with get_context("spawn").Pool(a.workers, initializer=scans._init_worker, initargs=(raw, cfgp, dc)) as pool, \
            open(path, "w") as f:
        for k, (label, index, held, chi2, values) in enumerate(pool.imap_unordered(run, tasks)):
            f.write(json.dumps(dict(label=label, index=index, held=held, chi2=chi2, values=values,
                                    stored=float(chi[index]), column_min=float(np.nanmin(chi[index[0]])))) + "\n")
            f.flush()
            if (k + 1) % 10 == 0:
                print(f"PROGRESS {100 * (k + 1) / len(tasks):.0f}% | {k + 1} of {len(tasks)} fits", flush=True)
    print("done", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
