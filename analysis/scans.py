"""
scans.py

The global fit and profile likelihoods: chi^2 minimized over the other free parameters at each
point of a grid of one parameter (a profile) or of two (a plane), computed in parallel.

Every grid point is a fit with the scanned parameters held. A fit started far from the
profile's valley can stop in a side minimum, so the points are fitted in chains: each half of a
grid line, on either side of the global best fit, is cut into contiguous chains that are walked
outward, and every point starts both from its predecessor's solution and from the global best
fit. A repair pass then evaluates each point at its neighbours' solutions (one chi^2 each) and
refits it from them wherever one already beats its own fit or the point sticks out above its
neighbours; the passes repeat until no point improves. Every completed fit is appended to a JSONL checkpoint, so an
interrupted scan resumes where it stopped.

Workers are spawned processes (the solver's numba kernels and pools are not fork-safe). Each
builds the analysis' objective once from the configuration, with its own cache of solver runs.
With one worker everything runs in this process.
"""
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig, from_dict
from .fitting import FitSettings, fit

# chi^2 improvement that counts as a repair (2026-10-14: 1e-6 counted the fits' own noise -- in the task-2 reruns 48%
# of the points "improved" in the first pass, by 6e-5 (median), and each change re-queued its neighbours, so every
# pass re-evaluated most of the plane; limits are read at Delta chi^2 of order 1)
REPAIR_TOL = 1e-3
SPIKE_TOL = 0.05           # chi^2 above the neighbours' midpoint that marks a point for refitting
_W: dict = {}              # per-process state: the configuration and its objective


# ----------------------------------------------------------------------------------------
# Workers
# ----------------------------------------------------------------------------------------
def _init_worker(raw: dict, path: str, data_centre):
    cfg = from_dict(raw, path)
    _W["cfg"] = cfg
    _W["objective"] = cfg.objective(data_centre=data_centre)
    _W["exact"] = None
    if cfg.exact is not None:
        from .exact_profile import ExactProfiler, ExactSettings
        _W["exact"] = ExactProfiler(cfg, objective=_W["objective"], settings=ExactSettings.from_dict(cfg.exact))


def _record(index, held, r) -> dict:
    out = dict(index=list(index), held=held, values=r.values, chi2=float(r.chi2), nfev=int(r.nfev),
               seconds=float(r.seconds), message=r.message)
    if hasattr(r, "certified"):                  # the exact profile's (exact_profile.ExactResult)
        out.update(certified=bool(r.certified), pred=float(r.pred), solver_runs=int(r.solver_runs),
                   centres=len(r.centres), starts=len(r.starts) or 1, cert=r.cert, scan_runs=int(r.scan_runs),
                   scan_seconds=float(r.scan_seconds))
        if r.inherit_check is not None:
            out["inherit_check"] = float(r.inherit_check)
    return out


def _fit_held(held: dict, starts: Sequence[dict], settings: dict):
    """The fit at the held values from the starts; with the exact profile (cfg.exact), its global profile from the
    first start given (the predecessor's solution in a chain, else the global best fit)."""
    cfg, obj = _W["cfg"], _W["objective"]
    if _W.get("exact") is not None:
        start = [s for s in starts if s][-1]
        return _W["exact"].profile_global(held, {k: v for k, v in start.items() if k not in held})
    pset = cfg.parameter_set().holding(held)
    return fit(obj, pset, starts=[s for s in starts if s], settings=FitSettings(**settings))


def _run_chain(task) -> List[dict]:
    """Fits the points of one chain in order, each from the global best fit and from its
    predecessor's solution. With the exact profile, a point whose solver runs at the last fully certified point's
    mixing are those of that point (exact_profile.ExactProfiler.inherit: new physics below the solver's reach)
    takes that point's certificate after one evaluation.

    With the lightest mass's scan (MeV planes, ExactSettings.ml_scan and ml_inherit; 2026-10-18), the column's first
    point (the plane's smallest coupling) is profiled in full and anchors it; the points that pass that test against
    it are held back until a scan of the lightest mass at a coupling at least theirs shows the solver's runs at every
    scanned mass equal to the first point's (scan_flat):
    absorption only grows with the coupling, so the new physics is then below the solver's reach at every mass
    there and below, and those points inherit. The probe goes down from the highest held-back point; a point
    whose scan is not flat is profiled in full (its scan's runs reused). Above the first point that fails the test
    every point is profiled in full.

    task: (chain, best, settings, sink) -- sink (checkpoint path, identity) writes each point's record as it is
    done (2026-10-18: a whole chain's at its end left long runs without a sign of progress), or None."""
    chain, best, settings = task[:3]
    sink = task[3] if len(task) > 3 else None
    out, prev, anchor = [], None, None
    prof = _W.get("exact")

    def emit(index, held, r):
        rec = _record(index, held, r)
        rec["t"] = time.time()                   # when it was done (tools/exact_progress.py: the rate)
        if sink is not None:
            rec["pass"] = 0
            rec["id"] = sink[1]
            _append_jsonl(sink[0], [rec])
        out.append(rec)

    if prof is not None:
        prof.curvs = []              # forward certificates take the curvature of this chain's own central ones
    ml = prof is not None and prof.settings.ml_scan is not None
    ml_inherit = ml and prof.settings.inherit and prof.settings.ml_inherit
    if prof is not None and prof.settings.inherit and not ml and chain and set(chain[0][1]) <= set(best):
        # the global fit (the exact profile without new physics, _exact_global_fit) anchors the chain's first point
        g_held = {k: float(best[k]) for k in chain[0][1]}
        key = json.dumps(sorted(best.items()))
        if _W.get("global_anchor", (None,))[0] != key:
            try:
                _W["global_anchor"] = (key, prof.anchor(g_held, SimpleNamespace(values=dict(best), chi2=float("nan"))))
            except ValueError:
                _W["global_anchor"] = (key, None)
        anchor = _W["global_anchor"][1]
    ref = mixing = None
    if ml_inherit:
        # MeV columns anchor on their own first point (the plane's smallest coupling, profiled in full with its
        # scan): the transport's numerics differ between mediator masses by more than inherit_eps even where the
        # coupling does nothing (pilot 2026-10-18), so the global fit's point cannot anchor them; its scan at the
        # scan's mixing is the reference of the flatness probes
        mixing = {n: float(best[n]) for n in prof.eta if n != "m_lightest_eV"}
        mixing.update({k: float(v) for k, v in (prof.settings.scan_mixing or {}).items() if k in mixing})
    pending, full = [], False

    def resolve():
        """The held-back points (increasing coupling): the scan probed from the top down until it is flat."""
        for j in range(len(pending) - 1, -1, -1):
            index, held, r_inh = pending[j]
            runs, k, sec = prof.scan_runs(held, mixing, best)
            flat, worst = prof.scan_flat(runs, ref)
            if flat:
                r_inh.message += f"; the m_lightest scan without new physics to {worst:.2g} here"
                for x in (r_inh,):
                    x.solver_runs += k
                    x.seconds += sec
                    x.scan_runs += k
                    x.scan_seconds += sec
                for ix, h, r in pending[:j]:
                    r.message += f"; the m_lightest scan without new physics at {held}"
                    emit(ix, h, r)
                emit(index, held, r_inh)
                break
            r = _fit_held(held, [best, r_inh.values], settings)
            r.message += f"; m_lightest scan with new physics ({worst:.2g})"
            r.solver_runs += k + r_inh.solver_runs
            r.seconds += sec + r_inh.seconds
            r.scan_runs += k
            r.scan_seconds += sec
            emit(index, held, r)
        pending.clear()

    for index, held in chain:
        r = None
        if prof is not None and anchor is not None:
            if not ml:
                r = prof.inherit(held, anchor)
            elif ref is not None and not full:
                r = prof.inherit(held, anchor, scan_pending=True)
                if r is not None:
                    pending.append((index, held, r))
                    continue
                full = True
                resolve()
        if r is None:
            r = _fit_held(held, [best, prev], settings)
            if prof is not None and not ml and r.certified and prof.settings.inherit:
                anchor = prof.anchor(held, r)
            if ml_inherit and anchor is None and r.certified:
                # the column's first point: its result anchors the column, its scan (kept models: no new run) is
                # the probes' reference
                anchor = prof.anchor(held, r)
                ref = prof.scan_runs(held, mixing, best)[0]
        emit(index, held, r)
        prev = r.values
    if pending:
        resolve()
    return out


def _run_repair(task) -> Optional[dict]:
    """Refits one point from its neighbours' solutions where they promise better: where one
    already gives a lower chi^2 at this point, or where the point sticks out above its
    neighbours (chi^2 above the midpoint of the two neighbours along every axis that has
    both, by SPIKE_TOL; a valley curving faster than the grid can defeat the first test).
    Edge points, which lack a neighbour on one side, are refitted from their neighbours once
    (force). Returns the improved record or None. A point stuck at the bottom of a convex
    profile, below its neighbours' midpoint, passes every test."""
    index, held, own_values, own_chi2, candidates, nb_mean, force, settings, own_certified = task
    obj = _W["objective"]
    prof = _W.get("exact")
    better, trials = [], []
    for c in candidates:
        if prof is not None and own_certified and prof.within_certificate(own_values, c):
            continue                 # the point's certificate already covers that mixing (exact_profile)
        trial = {**c, **held}
        x = obj.chi2(trial)
        trials.append((x, trial))
        if x < own_chi2 - REPAIR_TOL:
            better.append((x, trial))
    spike = force or (nb_mean is not None and own_chi2 > nb_mean + SPIKE_TOL)
    if prof is not None:
        if not better and not (spike and trials):
            return None
        # the exact profile from the most promising neighbour's solution
        x, start = min(better or trials, key=lambda bt: bt[0])
        r = _fit_held(held, [start], settings)
    else:
        if not better and not spike:
            return None
        starts = [t for _, t in sorted(better, key=lambda bt: bt[0])] + ([t for _, t in trials] if spike else []) \
            + [own_values]
        r = _fit_held(held, starts, settings)
    if r.chi2 < own_chi2 - REPAIR_TOL:
        return _record(index, held, r)
    return None


def _init_runner(raw: dict, path: str):
    """A worker that computes solver runs for another process's model (FlavorModel.prefetch): the model alone, without
    the likelihood or its Asimov data."""
    _W["runner"] = from_dict(raw, path).model()


def _remote_run(task):
    """One run of FlavorModel.prefetch: (kind, cfg, with_galactic) -> the run as the model computes it."""
    kind, cfg, with_galactic = task
    m = _W["runner"]
    return m._sky_compute(cfg, with_galactic) if kind == "sky" else m._fluxes_compute(cfg, with_galactic)


class _Runners:
    """For a fit in this process: its objective, built here, with the model's runs of each new point computed in
    parallel by `workers` spawned processes (FlavorModel.prefetch; 2026-10-18). A global fit from one start used one
    core of the plane's allocation while the others waited, for 10-90 minutes per plane in the October campaign."""

    def __init__(self, cfg: AnalysisConfig, data_centre, workers: int):
        self.cfg, self.data_centre, self.workers = cfg, data_centre, int(workers)
        self.pool = None

    def __enter__(self):
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.workers, initializer=_init_runner, initargs=(self.cfg.raw, self.cfg.path))
        model = self.cfg.model()
        pool = self.pool
        model.executor = lambda tasks: pool.map(_remote_run, tasks, chunksize=1)
        _W["cfg"] = self.cfg
        _W["objective"] = self.cfg.objective(model=model, data_centre=self.data_centre)
        return self

    def __exit__(self, *exc):
        _W["objective"].model.executor = None
        self.pool.close()
        self.pool.join()


def _run_start(task) -> dict:
    """One start of the global fit (no polish)."""
    start, settings = task
    cfg, obj = _W["cfg"], _W["objective"]
    r = fit(obj, cfg.parameter_set(), starts=[start], settings=FitSettings(**settings))
    return dict(values=r.values, chi2=float(r.chi2), nfev=int(r.nfev), seconds=float(r.seconds), start=start)


class _Pool:
    """A spawn pool, or this process when workers == 1."""

    def __init__(self, cfg: AnalysisConfig, data_centre, workers: int):
        self.workers = max(1, int(workers))
        self.args = (cfg.raw, cfg.path, data_centre)
        self.pool = None

    def __enter__(self):
        if self.workers == 1:
            _init_worker(*self.args)
        else:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(self.workers, initializer=_init_worker, initargs=self.args)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def imap(self, f, tasks: Iterable):
        if self.pool is None:
            return map(f, tasks)
        return self.pool.imap_unordered(f, tasks)


# ----------------------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------------------
def _read_jsonl(path: str, identity: Optional[str] = None, log=None) -> List[dict]:
    """The records of a checkpoint; with identity, only those computed under it (the others,
    from another configuration, are reported and ignored)."""
    if not os.path.exists(path):
        return []
    out, stale = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                  # a line cut by an interruption
                if identity is not None and r.get("id") != identity:
                    stale += 1
                    continue
                out.append(r)
    if stale and log is not None:
        log(f"  ignoring {stale} records of {os.path.basename(path)} computed under another configuration")
    return out


def _append_jsonl(path: str, records: Iterable[dict]):
    """Appends records, one JSON line each, in one write under an exclusive lock (the workers of a scan write their
    points to the same checkpoint; where the file system has no locks, the single O_APPEND write remains)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in records)
    with open(path, "a") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _best_by_index(records: Iterable[dict]) -> Dict[tuple, dict]:
    best = {}
    for r in records:
        if "chi2" not in r:
            continue                          # a marker, e.g. an edge point already checked
        k = tuple(r["index"])
        if k not in best or r["chi2"] < best[k]["chi2"]:
            best[k] = r
    return best


# ----------------------------------------------------------------------------------------
# Global fit
# ----------------------------------------------------------------------------------------
def _exact_global_fit(cfg: AnalysisConfig, data_centre, checkpoint: Optional[str], log) -> dict:
    """global_fit for the exact profile (its docstring)."""
    t0 = time.perf_counter()
    ident = cfg.identity(data_centre)
    done = [r for r in (_read_jsonl(checkpoint, ident, log) if checkpoint else []) if r.get("final")]
    if done:
        r = done[-1]
        return dict(values=r["values"], chi2=r["chi2"], nfev=0, seconds=0.0, starts=[], id=ident)
    from .exact_profile import ExactProfiler, ExactSettings
    if not cfg.planes:
        raise ValueError("the exact profile's global fit holds a plane's parameters: the configuration needs a plane")
    s = cfg.planes[0]
    held = {s.x: float(cfg.parameter(s.x).start), s.y: float(s.y_grid[0])}
    prof = ExactProfiler(cfg, data_centre=data_centre, settings=ExactSettings.from_dict(cfg.exact))
    r = prof.profile_global(held, {p.name: p.start for p in cfg.parameters if p.name not in held},
                            log=lambda m: log(m))
    log(f"  exact profile at {held}: chi2 = {r.chi2:.6g}, {r.message} ({r.solver_runs} solver runs, {r.seconds:.0f} s)")
    out = dict(values=r.values, chi2=r.chi2, nfev=r.nfev, seconds=time.perf_counter() - t0,
               starts=[dict(chi2=x["chi2"], values={}) for x in r.starts], id=ident)
    if checkpoint:
        _append_jsonl(checkpoint, [dict(final=True, values=out["values"], chi2=out["chi2"], id=ident,
                                        certified=r.certified, message=r.message, seconds=out["seconds"],
                                        solver_runs=int(r.solver_runs))])
    return out


def global_fit(cfg: AnalysisConfig, data_centre=None, workers: int = 1, checkpoint: Optional[str] = None,
               log=print) -> dict:
    """The best fit over all free parameters: L-BFGS-B from the configured starts and
    cfg.fit.n_random random starts inside the bounds (seeded), in parallel, then a Powell
    polish of the best. Returns {"values", "chi2", "nfev", "seconds", "starts": [...]}.

    With the exact profile (cfg.exact): the exact profile (exact_profile.py) with the plane's parameters held at the
    first plane's smallest coupling (no new physics) and the mass's start: the chains' start and the no-new-physics
    reference."""
    if cfg.exact is not None:
        return _exact_global_fit(cfg, data_centre, checkpoint, log)
    t0 = time.perf_counter()
    pset = cfg.parameter_set()
    rng = np.random.default_rng(cfg.fit.seed)
    lo, hi = pset.bounds_internal()
    starts = [pset.start_values()]
    starts += [pset.as_values(lo + (hi - lo) * rng.random(len(lo))) for _ in range(cfg.fit.n_random)]
    ident = cfg.identity(data_centre)
    done = _read_jsonl(checkpoint, ident, log) if checkpoint else []
    done_starts = {json.dumps(r["start"], sort_keys=True) for r in done if "start" in r}
    todo = [s for s in starts if json.dumps(s, sort_keys=True) not in done_starts]
    settings = {**asdict(cfg.fit), "polish": False, "n_random": 0}
    results = [r for r in done if "start" in r]
    # one start left (n_random = 0): fitted in this process with the solver runs in parallel (_Runners); several:
    # one start per worker, as before
    serial = len(todo) <= 1 and workers > 1
    with (_Runners(cfg, data_centre, workers) if serial else
          _Pool(cfg, data_centre, min(workers, max(1, len(todo))))) as pool:
        run = (lambda tasks: map(_run_start, tasks)) if serial else (lambda tasks: pool.imap(_run_start, tasks))
        for r in run([(s, settings) for s in todo]):
            r["id"] = ident
            results.append(r)
            if checkpoint:
                _append_jsonl(checkpoint, [r])
            log(f"  start {len(results)}/{len(starts)}: chi2 = {r['chi2']:.6g} ({r['nfev']} evaluations)")
        best = min(results, key=lambda r: r["chi2"])
        polish_settings = {**asdict(cfg.fit), "n_random": 0}
        polished = list(run([(best["values"], polish_settings)]))[0]
    final = polished if polished["chi2"] <= best["chi2"] else best
    out = dict(values=final["values"], chi2=final["chi2"],
               nfev=int(sum(r["nfev"] for r in results) + polished["nfev"]),
               seconds=time.perf_counter() - t0, starts=[dict(chi2=r["chi2"], values=r["values"]) for r in results])
    out["id"] = ident
    if checkpoint:
        _append_jsonl(checkpoint, [dict(final=True, values=out["values"], chi2=out["chi2"], id=ident)])
    return out


# ----------------------------------------------------------------------------------------
# Profiles and planes
# ----------------------------------------------------------------------------------------
@dataclass
class ScanResult:
    """Fits on a grid: kind '1d' or '2d', the scanned names and grids, and per point (in
    C order over the grid) the held values, the fitted values and chi^2."""
    kind: str
    names: List[str]
    grids: List[List[float]]
    points: List[dict] = field(default_factory=list)
    nfev: int = 0
    seconds: float = 0.0
    passes: int = 0

    def chi2_array(self) -> np.ndarray:
        shape = tuple(len(g) for g in self.grids)
        a = np.full(shape, np.inf)
        for p in self.points:
            a[tuple(p["index"])] = p["chi2"]
        return a

    def as_dict(self) -> dict:
        return asdict(self)


def _chains_1d(grid: np.ndarray, centre: float, log: bool, length: int) -> List[List[int]]:
    u = np.log10(grid) if log else grid
    c = np.log10(centre) if log else centre
    i0 = int(np.argmin(np.abs(u - c)))
    right = list(range(i0, len(grid)))
    left = list(range(i0 - 1, -1, -1))
    chains = []
    for side in (right, left):
        for k in range(0, len(side), length):
            chains.append(side[k:k + length])
    return [c for c in chains if c]


def _scan(cfg: AnalysisConfig, names: List[str], grids: List[np.ndarray], best: dict, data_centre,
          workers: int, checkpoint: Optional[str], max_passes: int, log=print) -> ScanResult:
    params = [cfg.parameter(n) for n in names]
    shape = tuple(len(g) for g in grids)
    indices = list(np.ndindex(*shape))
    if len(names) == len(cfg.parameters):
        max_passes = 0            # nothing left to fit: every point is one evaluation, nothing to repair

    def held_of(index):
        return {n: float(g[i]) for n, g, i in zip(names, grids, index)}

    # chains along the first axis, for each value of the others: about two per worker
    n_points = len(indices)
    chains = []
    if cfg.exact is not None and len(shape) == 2:
        # the exact profile (exact_profile.py): whole chains along the second axis (a plane's coupling), outward from
        # the global best fit's value -- the profile's mixing changes smoothly along the coupling, and a point below
        # the reach of new physics takes the chain's last certified point's certificate (ExactProfiler.inherit)
        for i in range(shape[0]):
            for c in _chains_1d(grids[1], best[names[1]], params[1].space == "log10", shape[1]):
                chains.append([(i, j) for j in c])
    else:
        rows = int(np.prod(shape[1:]))
        per_row = max(2, int(np.ceil(2 * workers / rows)))
        length = max(1, int(np.ceil(shape[0] / per_row)))
        for rest in np.ndindex(*shape[1:]):
            for c in _chains_1d(grids[0], best[names[0]], params[0].space == "log10", length):
                chains.append([(i,) + tuple(rest) for i in c])
    t0 = time.perf_counter()
    ident = scan_identity(cfg, names, grids, data_centre)
    records = _read_jsonl(checkpoint, ident, log) if checkpoint else []
    done = _best_by_index(records)
    edge_checked = {tuple(r["index"]) for r in records if r.get("edge_checked")}
    settings = {**asdict(cfg.fit), "n_random": 0}
    result = ScanResult("1d" if len(names) == 1 else "2d", list(names), [list(map(float, g)) for g in grids])
    best_start = {k: v for k, v in best.items()}
    todo = []
    # each point's record goes to the checkpoint as soon as it is done, from its worker (2026-10-18)
    sink = (checkpoint, ident) if checkpoint else None
    for ch in chains:
        rest = [ix for ix in ch if ix not in done]
        if rest:
            todo.append(([(ix, held_of(ix)) for ix in rest], best_start, settings, sink))
    # the repair passes use every worker too (2026-10-20: sized by the chains left, a resumed plane with all its points
    # done ran its repair passes, ~90 core-hours, in this one process)
    with _Pool(cfg, data_centre, workers if max_passes > 0 else min(workers, max(1, len(todo)))) as pool:
        n_done = len(done)
        for recs in pool.imap(_run_chain, todo):
            for r in recs:
                r["pass"] = 0
                r["id"] = ident
                done[tuple(r["index"])] = r
                result.nfev += r["nfev"]
            n_done += len(recs)
            log(f"  {n_done}/{n_points} points")
        # repair passes: cross-seed from the neighbours' solutions; after the first pass,
        # only around the points that changed
        def neighbours(ix):
            out = []
            for ax in range(len(shape)):
                for step in (-1, 1):
                    nb = list(ix)
                    nb[ax] += step
                    if 0 <= nb[ax] < shape[ax]:
                        out.append(tuple(nb))
            return out

        changed = set(indices)
        # each repair task's outcome goes to the checkpoint as soon as it is done (2026-10-20): the improved fit and a
        # marker {"repair_checked": k}; a restarted scan replays the passes from them and skips the points already
        # checked in the pass it stopped in (a pass of a plane's every point takes hours: written only at its end, it
        # was lost whole when a job reached its time limit)
        checked, improved_in = {}, {}
        for r in records:
            if r.get("repair_checked"):
                checked.setdefault(int(r["repair_checked"]), set()).add(tuple(r["index"]))
            if "chi2" in r and int(r.get("pass", 0)) >= 1:
                improved_in.setdefault(int(r["pass"]), set()).add(tuple(r["index"]))
        for k in range(1, max_passes + 1):
            tasks = []
            for ix in indices:
                nbs = neighbours(ix)
                if ix not in changed and not any(nb in changed for nb in nbs):
                    continue
                own = done[ix]
                cands = [done[nb]["values"] for nb in nbs if nb in done]
                # the profile interpolated between the two neighbours along each axis that has them
                mids = []
                for ax in range(len(shape)):
                    pair = [tuple(ix[:ax]) + (ix[ax] + st,) + tuple(ix[ax + 1:]) for st in (-1, 1)]
                    if all(p in done and np.isfinite(done[p]["chi2"]) for p in pair):
                        mids.append(0.5 * (done[pair[0]]["chi2"] + done[pair[1]]["chi2"]))
                nb_mid = max(mids) if mids else None
                edge = k == 1 and len(mids) < len(shape) and ix not in edge_checked   # once, from the neighbours
                tasks.append((ix, held_of(ix), own["values"], own["chi2"], cands, nb_mid, edge, settings,
                              bool(own.get("certified", False))))
            seen = checked.get(k, set())
            todo_k = [t for t in tasks if t[0] not in seen]
            changed = set(improved_in.get(k, set()))
            for t, r in zip(todo_k, pool.imap(_run_repair, todo_k)):
                recs = []
                if r is not None:
                    r["pass"] = k
                    r["id"] = ident
                    recs.append(r)
                    done[tuple(r["index"])] = r
                    result.nfev += r["nfev"]
                    changed.add(tuple(r["index"]))
                marker = dict(index=list(t[0]), repair_checked=k, id=ident)
                if t[6]:
                    marker["edge_checked"] = True
                    edge_checked.add(t[0])
                if checkpoint:
                    _append_jsonl(checkpoint, recs + [marker])
            result.passes = k
            log(f"  repair pass {k}: {len(changed)} of {len(tasks)} points improved"
                + (f" ({len(tasks) - len(todo_k)} checked before a restart)" if len(todo_k) < len(tasks) else ""))
            if not changed:
                break
    result.points = [done[ix] for ix in indices]
    result.seconds = time.perf_counter() - t0
    return result


def scan_identity(cfg: AnalysisConfig, names: Sequence[str], grids: Sequence[np.ndarray], data_centre=None) -> str:
    """The identity a scan's checkpoint records carry (configuration, data, names, grid)."""
    return cfg.identity(data_centre, extra=dict(names=list(names), grids=[[float(x) for x in g] for g in grids]))


def profile(cfg: AnalysisConfig, name: str, grid: np.ndarray, best: dict, data_centre=None,
            workers: int = 1, checkpoint: Optional[str] = None, max_passes: int = 3, log=print) -> ScanResult:
    """The profile of chi^2 in one parameter over grid (physical values), starting from the
    global best fit's values `best`."""
    return _scan(cfg, [name], [np.asarray(grid, dtype=float)], best, data_centre, workers, checkpoint,
                 max_passes, log)


def plane(cfg: AnalysisConfig, x: str, y: str, x_grid: np.ndarray, y_grid: np.ndarray, best: dict,
          data_centre=None, workers: int = 1, checkpoint: Optional[str] = None, max_passes: int = 3,
          log=print) -> ScanResult:
    """The profile of chi^2 in two parameters over the grid x_grid x y_grid (index order
    [i_x, i_y])."""
    return _scan(cfg, [x, y], [np.asarray(x_grid, dtype=float), np.asarray(y_grid, dtype=float)], best,
                 data_centre, workers, checkpoint, max_passes, log)
