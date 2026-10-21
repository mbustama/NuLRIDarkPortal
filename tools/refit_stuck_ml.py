"""
refit_stuck_ml.py -- the ultralight planes' points stuck in the wrong basin of the lightest neutrino mass (campaign_ml,
2026-10-21). The ultralight scans profile m_lightest from 1e-6 eV (its lower bound) by local steps; where chi^2 has a
second, lower basin near 1e-4 eV, a point can stay at 1e-6 eV with chi^2 far too high (at 10 keV, g = 1e-4..1e-3.25
in L_e-L_mu: 16-335 against 1.66), and the repair pass did not try its neighbours' solutions (their m_lightest counted
as inside the point's certificate box, measured in eV). Suspects: points at m_lightest = 1e-6 eV with a neighbour at
m_lightest > 1e-5 eV and chi^2 lower by more than 1. Each is refitted with the MeV planes' scan of the lightest mass
(exact_profile.ml_scan: chi^2 on 1e-6 and 1e-4..0.1 eV, three per decade, then profile() from each basin) from its own
solution and from each neighbour's; where the best is lower than the stored chi^2 (by more than the repair passes'
1e-3), it is appended to the plane's checkpoint as a repair record (pass 4, the scan's identity kept: the likelihood is
the same, only the search differs). Every trial goes to <plane output>/<data|asimov>/refit_stuck_ml.jsonl. Afterwards:
python -m analysis <config> --summarize.
  python tools/refit_stuck_ml.py <data set>/<plane> [--workers 16] [--count]
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

ML_SCAN = [1e-6] + list(np.logspace(-4, -1, 10))   # the MeV planes' exact_profile.ml_scan
M_LOW = 1.0001e-6


def suspects(best):
    out = []
    for (i, j), r in best.items():
        if r["values"].get("m_lightest_eV", 1.0) > M_LOW:
            continue
        nb = [best.get((i + a, j + b)) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        if any(n and n["values"].get("m_lightest_eV", 0.0) > 1e-5 and n["chi2"] < r["chi2"] - 1 for n in nb):
            out.append(((i, j), [n["values"] for n in nb if n]))
    return out


def run(task):
    from analysis import scans
    index, held, start, settings, label = task
    r = scans._fit_held(held, [start], settings)
    return index, held, label, r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plane", help="e.g. proj_future/le_lmu_ul")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args(argv)
    ds, name = a.plane.split("/")
    cfgp = os.path.join(REPO, f"analysis/configs/campaign_ml/{ds}/{name}.json")
    raw = json.load(open(cfgp))
    kind = "asimov" if ds == "proj_future" else "data"
    out_dir = os.path.join(REPO, raw["output_dir"], kind)
    s, x, y, chi, best = load(out_dir)
    sus = suspects(best)
    print(f"{a.plane}: {len(sus)} suspects", flush=True)
    if a.count or not sus:
        return 0
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_centre
    raw = dict(raw, exact_profile={**raw.get("exact_profile", {}), "ml_scan": ML_SCAN})
    cfg = from_dict(raw, cfgp)
    dc = asimov_centre(cfg) if kind == "asimov" else None
    xn, yn = s["names"]
    settings = {**asdict(cfg.fit), "n_random": 0}
    tasks = []
    for (i, j), nbs in sus:
        held = {xn: float(x[i]), yn: float(y[j])}
        for k, st in enumerate([best[(i, j)]["values"]] + nbs):
            tasks.append(((i, j), held, {kk: v for kk, v in st.items() if kk not in held}, settings,
                          "own" if k == 0 else f"neighbour {k}"))
    plane_ckpt = os.path.join(out_dir, f"plane_{xn}_{yn}.jsonl")
    trials = os.path.join(out_dir, "refit_stuck_ml.jsonl")
    print(f"{len(tasks)} fits on {min(a.workers, len(tasks))} workers; checkpoint {plane_ckpt}", flush=True)
    found = {}
    with get_context("spawn").Pool(min(a.workers, len(tasks)), initializer=scans._init_worker,
                                   initargs=(raw, cfgp, dc)) as pool, open(trials, "a") as f:
        for index, held, label, r in pool.imap_unordered(run, tasks):
            f.write(json.dumps(dict(index=list(index), held=held, label=label, chi2=float(r.chi2), values=r.values,
                                    stored=best[index]["chi2"])) + "\n")
            f.flush()
            if index not in found or r.chi2 < found[index][2].chi2:
                found[index] = (held, label, r)
    new = []
    for index, (held, label, r) in sorted(found.items()):
        old = best[index]["chi2"]
        print(f"  {index}: m=1e{np.log10(held[xn]):.2f} g=1e{np.log10(held[yn]):.2f}  stored {old:.3f} -> "
              f"{r.chi2:.3f} ({label}, m_lightest {r.values.get('m_lightest_eV', np.nan):.3g})", flush=True)
        if r.chi2 < old - scans.REPAIR_TOL:
            rec = scans._record(index, held, r)
            rec.update(id=best[index].get("id"), **{"pass": 4}, note="refit_stuck_ml.py (ml_scan), 2026-10-21")
            new.append(rec)
    if new:
        scans._append_jsonl(plane_ckpt, new)
    print(f"{len(new)} of {len(found)} points improved and appended", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
