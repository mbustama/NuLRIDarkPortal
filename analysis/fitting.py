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
        settings: Optional[FitSettings] = None) -> FitResult:
    """Minimize chi^2 over the free parameters of pset.

    Arguments:
      objective: the Objective; pset: free parameters (and held values).
      starts: named starting values (only the free parameters are read); default: the
        parameters' own starts.
      settings: FitSettings.
    Returns: FitResult at the best point found.
    """
    st = settings or FitSettings()
    t0 = time.perf_counter()
    if not pset.free:                                   # nothing to fit: one evaluation
        v = dict(pset.held)
        return FitResult(v, objective.chi2(v), 1, time.perf_counter() - t0, 0, "no free parameters")
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
