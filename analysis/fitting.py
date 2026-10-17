"""
fitting.py

Minimization of an Objective over a ParameterSet's internal coordinates.

Each start runs a bounded quasi-Newton descent (L-BFGS-B, finite-difference gradient), and
the best result is then polished by Powell's derivative-free method, keeping it only if it
improves. The chi^2 of these analyses has small ripples where an absorption dip moves
across the energy grid and flat directions where several parameters trade off along one
flavor composition, so a gradient method alone can stop short; Powell walks along
coordinates and is not misled by a noisy gradient. Global fits add random starts inside
the bounds (seeded, so a fit is reproducible); profile points start from their
neighbour's solution, which makes most of them a short polish.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .objective import Objective
from .parameters import ParameterSet


@dataclass
class FitSettings:
    """Knobs of a fit. fd_step: finite-difference step in internal units (a thousandth of a
    decade for log10 parameters); max_evals: evaluation budget per start; n_random: random
    starts added to the given ones; seed: their seed; polish: run the Powell stage."""
    fd_step: float = 1e-3
    max_evals: int = 400
    ftol: float = 1e-9
    xtol: float = 1e-3
    n_random: int = 0
    seed: int = 12345
    polish: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Mapping]) -> "FitSettings":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown fit settings {sorted(unknown)}; known: {sorted(cls.__dataclass_fields__)}")
        return cls(**d)


@dataclass
class FitResult:
    values: Dict[str, float]
    chi2: float
    nfev: int
    seconds: float
    starts: int
    message: str = ""
    trace: List[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(values=self.values, chi2=self.chi2, nfev=self.nfev, seconds=self.seconds,
                    starts=self.starts, message=self.message, trace=self.trace)


def _counted(f):
    n = [0]

    def g(u):
        n[0] += 1
        return f(u)
    return g, n


def fit(objective: Objective, pset: ParameterSet, starts: Optional[Sequence[Mapping]] = None,
        settings: Optional[FitSettings] = None, gradient: Optional[bool] = None) -> FitResult:
    """Minimize chi^2 over the free parameters of pset.

    Arguments:
      objective: the Objective; pset: free parameters (and held values).
      starts: named starting values (only the free parameters are read); default: the
        parameters' own starts.
      settings: FitSettings.
      gradient: None (the default) uses chi^2's exact gradient where the objective gives it
        (Objective.supports_gradient: the future detectors; 2026-10-17), False forces the finite differences.
    Returns: FitResult at the best point found.
    """
    st = settings or FitSettings()
    t0 = time.perf_counter()
    if not pset.free:                                   # nothing to fit: one evaluation
        v = dict(pset.held)
        return FitResult(v, objective.chi2(v), 1, time.perf_counter() - t0, 0, "no free parameters")
    if gradient is not False and getattr(objective, "supports_gradient", None) is not None \
            and objective.supports_gradient(pset.names):
        return _fit_with_gradient(objective, pset, starts, st, t0)
    lo, hi = pset.bounds_internal()
    bounds = list(zip(lo, hi))
    f, n = _counted(objective.on(pset))
    u0s = [pset.to_internal({**pset.start_values(), **dict(s)}) for s in (starts or [pset.start_values()])]
    if st.n_random:
        rng = np.random.default_rng(st.seed)
        u0s += [lo + (hi - lo) * rng.random(len(lo)) for _ in range(st.n_random)]
    best_u, best_f, trace = None, np.inf, []
    for u0 in u0s:
        r = minimize(f, u0, method="L-BFGS-B", bounds=bounds,
                     options=dict(eps=st.fd_step, ftol=st.ftol, gtol=1e-8, maxfun=st.max_evals))
        trace.append(float(r.fun))
        if r.fun < best_f:
            best_u, best_f = np.array(r.x), float(r.fun)
    msg = "lbfgsb"
    if st.polish:
        r = minimize(f, best_u, method="Powell", bounds=bounds,
                     options=dict(xtol=st.xtol, ftol=st.ftol, maxfev=st.max_evals))
        if r.fun < best_f:
            best_u, best_f = np.array(r.x), float(r.fun)
            msg = "lbfgsb+powell"
    return FitResult(pset.as_values(best_u), best_f, n[0], time.perf_counter() - t0, len(u0s), msg, trace)


GRADIENT_CHECK_STEP = 1e-4          # internal units (a parameter's typical step is 1)
# absolute (chi^2 per internal unit) and relative: a slope 1e-2 off moves the fit's chi^2 by ~1e-4 / curvature, while
# a wrong gradient (a missing term, a sign) stops the fit where the true slope is of order one
GRADIENT_CHECK_TOL = (1e-2, 2e-2)


def _check_direction(pset: ParameterSet, n: int) -> np.ndarray:
    """A unit direction for the end-of-fit gradient check, set by the held values (reproducible, different from
    point to point)."""
    import hashlib
    import json
    seed = int(hashlib.sha256(json.dumps(sorted(pset.held.items()), default=str).encode()).hexdigest()[:8], 16)
    d = np.random.default_rng(seed).standard_normal(n)
    return d / np.linalg.norm(d)


def gradient_check(fg, u, g, lo, hi, d, h=GRADIENT_CHECK_STEP, tol=GRADIENT_CHECK_TOL):
    """(passed, detail): the slope g . d against the central difference of fg's chi^2 along d (components that
    cannot move both ways within the bounds left out). A wrong gradient would let L-BFGS-B stop where the true slope
    is not zero; this catches it."""
    d = np.array(d, dtype=np.float64)
    d[(u - h < lo) | (u + h > hi)] = 0.0
    norm = np.linalg.norm(d)
    if norm == 0.0:
        return True, "no free direction"
    d /= norm
    fd = (fg(u + h * d)[0] - fg(u - h * d)[0]) / (2.0 * h)
    an = float(np.dot(g, d))
    ok = abs(fd - an) <= tol[0] + tol[1] * abs(fd)
    return ok, f"slope {an:.4g} vs central difference {fd:.4g}"


def _fit_with_gradient(objective: Objective, pset: ParameterSet, starts, st: FitSettings, t0: float) -> FitResult:
    """fit() with chi^2's exact gradient (Objective.on_with_grad) in place of forward differences: the same L-BFGS-B,
    tolerances, starts and evaluation budget. At the end the gradient is checked against a central difference along
    a direction set by the held values (gradient_check, two evaluations); on a mismatch the fit is redone with
    finite differences from the best point, the better result kept, and the message says so."""
    lo, hi = pset.bounds_internal()
    bounds = list(zip(lo, hi))
    fg_raw = objective.on_with_grad(pset)
    n = [0]

    def fg(u):
        n[0] += 1
        return fg_raw(u)
    u0s = [pset.to_internal({**pset.start_values(), **dict(s)}) for s in (starts or [pset.start_values()])]
    if st.n_random:
        rng = np.random.default_rng(st.seed)
        u0s += [lo + (hi - lo) * rng.random(len(lo)) for _ in range(st.n_random)]
    best_u, best_f, best_g, trace = None, np.inf, None, []
    for u0 in u0s:
        r = minimize(fg, u0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(ftol=st.ftol, gtol=1e-8, maxfun=st.max_evals))
        trace.append(float(r.fun))
        if r.fun < best_f:
            best_u, best_f, best_g = np.array(r.x), float(r.fun), np.array(r.jac, dtype=np.float64)
    msg = "lbfgsb+gradient"
    extra = 0                                           # evaluations of a finite-difference refit
    ok, detail = gradient_check(fg, best_u, best_g, lo, hi, _check_direction(pset, len(best_u)))
    if not ok:
        r_fd = fit(objective, pset, starts=[pset.as_values(best_u)], settings=st, gradient=False)
        extra = r_fd.nfev
        msg += f"; gradient check failed ({detail}): refit with finite differences"
        if r_fd.chi2 < best_f:
            best_u, best_f = pset.to_internal(r_fd.values), float(r_fd.chi2)
    if st.polish:
        def f(u):
            return fg(u)[0]
        r = minimize(f, best_u, method="Powell", bounds=bounds,
                     options=dict(xtol=st.xtol, ftol=st.ftol, maxfev=st.max_evals))
        if r.fun < best_f:
            best_u, best_f = np.array(r.x), float(r.fun)
            msg += "+powell"
    return FitResult(pset.as_values(best_u), best_f, n[0] + extra, time.perf_counter() - t0, len(u0s), msg, trace)
