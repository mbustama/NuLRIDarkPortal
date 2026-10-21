"""
check_dips.py -- why delta chi^2 dips at some couplings of the ultralight planes (campaign_ml, 2026-10-21):
  (1) the future detectors' L_e-L_mu plane with pion-decay sources has a thin allowed channel along g = m 10^-1.5
      (delta chi^2 3.3 between 4.4 and 4.7), splitting its excluded region in two;
  (2) the MESE flavor + IceCube 2026 spectrum L_e-L_tau plane with fitted compositions is excluded at delta chi^2 ~3.9
      up to g = 1e-3 but falls to 3.7 at 10^-2.5 and 0.5 at 1e-2, while with pion-decay sources it stays at 7.6-10.
At one mass and several couplings (and the plane's smallest coupling, the no-effect reference), each point is refitted
(the campaign's own fitting code, mass and coupling held) from the nearest stored solution, with the campaign's grids
and, with --numerics, all grids doubled (as tools/check_band_numerics.py). At each fit the chi^2 is split into its
terms -- flavor surface, spectrum, event likelihoods, priors -- and the flux at Earth per flavor against energy, the
energy-averaged flavor fractions and the spectrum's segments (model and data) are kept. Asimov planes take the data
centre from manifest.json (tools/resummarize.py). Writes <plane output>/<data|asimov>/check_dips.<grid>.jsonl and
.npz (spectra).
  python tools/check_dips.py <data set>/<plane> <log10 m> <log10 g> [<log10 g> ...] [--workers 16] [--numerics]
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
from check_band_numerics import doubled  # noqa: E402
from check_edge_islands import load  # noqa: E402


def run(task):
    from analysis import scans, surfaces
    label, held, start, settings = task
    r = scans._fit_held(held, [start], settings)
    obj = scans._W["objective"]
    v = {**r.values, **held}
    terms = dict(total=float(r.chi2), spectrum=float(obj.spectrum_chi2(v)), priors=float(obj.prior_chi2(v)))
    f = obj.model.fractions(v)
    if obj.data is not None:
        terms["flavor"] = float(surfaces.chi2(f[0], f[1], obj.data))
    if obj.events is not None:
        terms["events"] = float(obj.events.chi2(v))
    E, F = obj.model.flux(v)
    seg = obj.spectrum_segments(v) if obj.spectrum is not None else None
    return label, held, r.values, terms, [float(x) for x in f], E, F, seg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plane")
    ap.add_argument("m", type=float)
    ap.add_argument("g", type=float, nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--numerics", action="store_true")
    a = ap.parse_args(argv)
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_only
    ds, name = a.plane.split("/")
    cfgp = os.path.join(REPO, f"analysis/configs/campaign_ml/{ds}/{name}.json")
    raw0 = json.load(open(cfgp))
    cfg0 = from_dict(raw0, cfgp)
    kind = "asimov" if asimov_only(cfg0) else "data"
    out_dir = cfg0.output_path(kind)
    dc = None
    if kind == "asimov":
        runs = json.load(open(os.path.join(out_dir, "manifest.json")))["runs"]
        dc = tuple(next(r["data_centre"] for r in reversed(runs) if r.get("data_centre")))
    s, x, y, chi, best = load(out_dir)
    xn, yn = s["names"]
    lx, ly = np.log10(x), np.log10(y)
    i = int(np.argmin(abs(lx - a.m)))
    points = [("reference", float(ly[0]))] + [(f"{g:+.2f}", g) for g in a.g]
    for grid in (["standard", "doubled"] if a.numerics else ["standard"]):
        raw = doubled(raw0) if grid == "doubled" else raw0
        cfg = from_dict(raw, cfgp)
        settings = {**asdict(cfg.fit), "n_random": 0}
        tasks = [(lab, {xn: float(x[i]), yn: float(10 ** g)}, best[(i, int(np.argmin(abs(ly - g))))]["values"],
                  settings) for lab, g in points]
        print(f"{a.plane} at m = 1e{lx[i]:.2f} eV, {grid} grids: {len(tasks)} fits", flush=True)
        rows, spec = [], {}
        with get_context("spawn").Pool(min(a.workers, len(tasks)), initializer=scans._init_worker,
                                       initargs=(raw, cfgp, dc)) as pool:
            for lab, held, values, terms, f, E, F, seg in pool.imap(run, tasks):
                rows.append(dict(label=lab, held=held, values=values, terms=terms, fractions=f))
                spec[f"F|{lab}"], spec["E"] = F, E
                if seg is not None:
                    spec[f"seg|{lab}"] = seg
                print(f"  {lab:10s} " + " ".join(f"{k} {v:.3f}" for k, v in terms.items())
                      + f"  f_e,f_mu,f_tau = {f[0]:.4f},{f[1]:.4f},{f[2]:.4f}", flush=True)
        ref = rows[0]["terms"]
        for r in rows[1:]:
            print(f"  {r['label']}: delta chi2 {r['terms']['total'] - ref['total']:.3f} = " + " + ".join(
                f"{k} {r['terms'][k] - ref[k]:+.3f}" for k in r["terms"] if k != "total"), flush=True)
        stem = os.path.join(out_dir, f"check_dips.{grid}")
        with open(stem + ".jsonl", "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
        name_s = raw0.get("data", {}).get("spectrum")
        if name_s:                                         # the measured segments, for the figure
            from analysis import spectra
            gr = raw0["solver"]["grid"]
            sd = spectra.load(name_s, float(gr["E_min_eV"]) / 1e9, float(gr["E_max_eV"]) / 1e9)
            spec.update({f"data|{k}": np.asarray(v) for k, v in sd.items() if np.ndim(v) == 1})
        np.savez(stem + ".npz", **spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
