"""
events_profile_check.py -- why the HESE + DR2 chi^2 does not always come back on re-evaluation (2026-10-22: the
thermal-grid recheck's control -- the old grid at stored solutions, tools/recheck_thermal_grid.py -- gave back the
stored chi^2 at every MESE + spectrum point, but not at 2% of the HESE + DR2 ones: by more than 0.05 at 0.6%, up to
1.5). Suspect: the event nuisances' profile (analysis/events.EventLikelihood.profile) starts from the nuisances of
the previous call in the same process, so wherever the profile has more than one minimum (the HESE detector
systematics' interpolation has kinks at its nodes) a chi^2 depends on what the process evaluated before.
At every mismatched control point (|control - stored| > MISMATCH) and at as many matched ones (seeded sample), the
astrophysical expectation is computed once, with the grid the stored fits used, and the profile is run from the
priors' centres and from STARTS random starts (1 sigma, seeded by the point): the full chi^2 from each, against the
stored and the control values. Writes output/analysis/events_profile_check.jsonl (appends; done points skipped).
  python tools/events_profile_check.py run [--workers 32]
  python tools/events_profile_check.py report
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO, os.path.join(REPO, "tools")]
CAMPAIGN = os.path.join(REPO, "output", "analysis", "campaign_ml", "now_events")
CONFIGS = os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_events")
OUT = os.path.join(REPO, "output", "analysis", "events_profile_check.jsonl")
MISMATCH = 1e-3                 # chi^2: a control further than this from the stored value is a mismatch
STARTS = 16                     # random starts of the profile per point, besides the priors' centres
SEED = 20261022
_W = {}


def _key(plane, source, held):
    return json.dumps([plane, source, held], sort_keys=True)


def controls():
    """{key: (plane, source, held, stored, control)} of every HESE + DR2 control evaluation."""
    out = {}
    for f in glob.glob(os.path.join(CAMPAIGN, "*", "data", "recheck.*.jsonl")):
        plane = f.split(os.sep)[-3]
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["gap"] == 0.0 and r["chi2"] is not None and r["stored"] is not None:
                out[_key(plane, r["source"], r["held"])] = (plane, r["source"], r["held"], r["stored"], r["chi2"])
    return out


def selection():
    """The mismatched controls and as many matched ones (seeded), as (key, plane, source, held, stored, control)."""
    c = controls()
    bad = sorted(k for k, v in c.items() if abs(v[4] - v[3]) > MISMATCH)
    good = sorted(k for k, v in c.items() if abs(v[4] - v[3]) <= MISMATCH)
    rng = np.random.default_rng(SEED)
    good = [good[i] for i in sorted(rng.choice(len(good), size=min(len(bad), len(good)), replace=False))]
    return [(k,) + c[k] for k in bad + good]


def _init():
    os.environ.pop("NULRI_FLUX_CACHE", None)
    import dp_si
    dp_si.THERMAL_GRID_MIN_GAP = 0.0                # the grid the stored fits and the control used


def _objective(cfgp):
    """The plane's objective, one kept per process (a process mostly sees one plane: the tasks are sorted)."""
    if _W.get("path") != cfgp:
        from analysis.config import from_dict
        _W.clear()
        obj = from_dict(json.load(open(cfgp)), cfgp).objective()
        ev = obj.events
        orig = type(ev).astro

        def astro(values):                           # the expectation of the last evaluation, kept for the starts
            _W["A"] = orig(ev, values)
            return _W["A"]
        ev.astro = astro
        _W.update(path=cfgp, obj=obj)
    return _W["obj"]


def _one(task):
    key, plane, source, held, stored, control, values = task
    try:
        obj = _objective(os.path.join(CONFIGS, plane + ".json"))
        ev = obj.events
        v = dict(values)
        v.update(held)
        n = len(ev.names)
        ev._z0 = np.zeros(n)
        total = float(obj.chi2(v))                   # the full chi^2, the profile from the priors' centres
        A = _W["A"]
        lo = np.array([-np.inf if b[0] is None else (b[0] - c) / s for b, c, s in zip(ev.bounds, ev.centre, ev.sigma)])
        hi = np.array([np.inf if b[1] is None else (b[1] - c) / s for b, c, s in zip(ev.bounds, ev.centre, ev.sigma)])
        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
        starts = [np.zeros(n)] + [np.clip(rng.standard_normal(n), lo, hi) for _ in range(STARTS)]
        ev_tot, how = [], []
        for z in starts:
            ev._z0 = z.copy()
            fb = ev.n_fallback
            x, _ = ev.profile(A)
            ev_tot.append(float(ev.exact(x, A)["total"]))
            how.append("lbfgsb" if ev.n_fallback > fb else "gn")
        other = total - ev_tot[0]                    # the rest of chi^2 (priors, spectrum): the same at every start
        chi = [other + e for e in ev_tot]
        return dict(key=key, plane=plane, source=source, held=held, stored=stored, control=control, centre=chi[0],
                    starts=chi, how=how, error=None)
    except Exception as e:                           # a failing point is recorded, not fatal
        return dict(key=key, plane=plane, source=source, held=held, stored=stored, control=control, error=repr(e)[:300])


def run(workers):
    from recheck_thermal_grid import solutions
    from refine_grid import Progress
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    sel = [s for s in selection() if s[0] not in done]
    tasks = []
    for plane in sorted({s[1] for s in sel}):
        sols = {(src, json.dumps(h, sort_keys=True)): vals
                for src, h, vals, _ in solutions(os.path.join(CONFIGS, plane + ".json"))}
        for key, pl, src, held, stored, control in sel:
            if pl == plane:
                tasks.append((key, pl, src, held, stored, control, sols[(src, json.dumps(held, sort_keys=True))]))
    print(f"{len(tasks)} points ({len(done)} done), {STARTS + 1} profiles each", flush=True)
    progress = Progress(len(tasks), "points")
    with get_context("spawn").Pool(min(workers, max(1, len(tasks))), initializer=_init) as pool, open(OUT, "a") as fh:
        for k, r in enumerate(pool.imap(_one, tasks, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            progress(k + 1)
    print("done", flush=True)


def report():
    recs = [json.loads(line) for line in open(OUT)]
    ok = [r for r in recs if not r["error"]]
    print(f"{len(recs)} points, {len(recs) - len(ok)} errors")
    for name, rows in (("mismatched", [r for r in ok if abs(r["control"] - r["stored"]) > MISMATCH]),
                       ("matched", [r for r in ok if abs(r["control"] - r["stored"]) <= MISMATCH])):
        if not rows:
            continue
        spread = np.array([max(r["starts"]) - min(r["starts"]) for r in rows])
        low = np.array([min(r["starts"]) - min(r["stored"], r["control"]) for r in rows])
        cover = np.array([min(r["starts"]) - 1e-3 <= min(r["stored"], r["control"]) and
                          max(r["stored"], r["control"]) <= max(r["starts"]) + 1e-3 for r in rows])
        fb = sum(h == "lbfgsb" for r in rows for h in r["how"])
        print(f"{name:10s} {len(rows):4d} points: spread over starts > 1e-3 at {np.mean(spread > 1e-3):.0%}, "
              f"> 0.05 at {np.mean(spread > 0.05):.0%}, max {spread.max():.2f}; stored and control both within the "
              f"starts' range at {np.mean(cover):.0%}; lowest start below both by > 1e-3 at {np.mean(low < -1e-3):.0%} "
              f"(by up to {-low.min():.2f}); L-BFGS-B fallbacks {fb} of {sum(len(r['how']) for r in rows)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    if a.command == "run":
        run(a.workers)
        return 0
    return report()


if __name__ == "__main__":
    sys.exit(main())
