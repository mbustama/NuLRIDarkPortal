"""
check_seam.py -- where the ultralight and MeV planes of L_e-L_mu meet (campaign_ml, 10 keV), the future detectors'
ultralight plane excludes g = 1e-4..3e-4 (delta chi^2 14-125) while the MeV plane finds no effect at the same points
(2026-10-21). Their configurations differ in ranges, starts, the spectral-index nodes and the MeV plane's shortcut
(exact_profile.ml_scan: a point whose quick check shows no effect takes the no-effect chi^2). Here the points at
m = 1e4 eV are refitted in full (one point at a time, no shortcut) with one plane's configuration, each from the stored
solution of both planes at that point, and the stored records are printed (solver runs, inheritance).
Written to output/analysis/campaign_ml/proj_future/seam_check.<ul|mev><_fx>.jsonl.
  python tools/check_seam.py <ul | mev> <pion | fitted> [--workers 16]
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
from check_edge_islands import load  # noqa: E402

COUPLINGS = (-10, -4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0)   # log10 g at m = 1e4 eV; -10: no effect
OUT = os.path.join(REPO, "output/analysis/campaign_ml/proj_future")


def run(task):
    from analysis import scans
    label, held, start, settings = task
    r = scans._fit_held(held, [start], settings)
    return label, held, float(r.chi2), r.values


def nearest(out_dir, lg):
    s, x, y, chi, best = load(out_dir)
    i, j = int(np.argmin(abs(np.log10(x) - 4))), int(np.argmin(abs(np.log10(y) - lg)))
    return best[(i, j)], float(np.log10(y[j]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=("ul", "mev"))
    ap.add_argument("comp", choices=("pion", "fitted"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    sfx = "_fx" if a.comp == "fitted" else ""
    cfgp = os.path.join(REPO, f"analysis/configs/campaign_ml/proj_future/le_lmu_{a.kind}{sfx}.json")
    raw = json.load(open(cfgp))
    cfg = from_dict(raw, cfgp)
    dc = asimov_centre(cfg)
    settings = {**asdict(cfg.fit), "n_random": 0}
    tasks = []
    for lg in COUPLINGS:
        held = {"m_Zd_eV": 1e4, "g_D": float(10 ** lg)}
        for src in ("ul", "mev"):
            rec, gl = nearest(f"{OUT}/le_lmu_{src}{sfx}/asimov", lg)
            print(f"stored {src:3s} g=1e{gl:.2f}: chi2 {rec['chi2']:.3f}, solver runs {rec.get('solver_runs')}, "
                  f"inherit_check {rec.get('inherit_check')}, certified {rec.get('certified')}, "
                  f"pass {rec.get('pass', 0)}", flush=True)
            tasks.append((f"{lg}|from {src}", held, rec["values"], settings))
    print(f"le_lmu_{a.kind}{sfx}: {len(tasks)} fits on {min(a.workers, len(tasks))} workers", flush=True)
    path = f"{OUT}/seam_check.{a.kind}{sfx}.jsonl"
    with get_context("spawn").Pool(min(a.workers, len(tasks)), initializer=scans._init_worker,
                                   initargs=(raw, cfgp, dc)) as pool, open(path, "w") as f:
        for label, held, chi2, values in pool.imap_unordered(run, tasks):
            f.write(json.dumps(dict(label=label, held=held, chi2=chi2, values=values)) + "\n")
            f.flush()
            print(label, round(chi2, 3), flush=True)
    print("done", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
