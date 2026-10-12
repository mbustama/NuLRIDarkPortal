"""
parameters.py

Free parameters and priors. A fit sees a plain vector of internal coordinates; this module
owns the map between that vector and the named physical values:

  internal = transform(value) / scale,   transform = log10 or identity,

so a mediator mass spanning thirty decades and a mixing element of order one both move in
steps of order one, and box bounds stay box bounds. Parameters can be held at fixed values
(a profile holds its parameter of interest): held values leave the vector but stay in
every mapping the model reads, and priors apply to them as well, so the chi^2 of a held
fit can be compared with that of a free one.

Priors contribute to chi^2 = -2 ln L. Each declares the quantity it constrains: a
parameter, or a derived quantity the objective computes (the sum of neutrino masses).
"""
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SPACES = ("linear", "log10")


@dataclass(frozen=True)
class GaussianPrior:
    """chi^2 += ((x - centre) / width)^2 on the quantity `name`."""
    name: str
    centre: float
    width: float

    def chi2(self, x: float) -> float:
        return ((x - self.centre) / self.width) ** 2


@dataclass(frozen=True)
class HalfGaussianPrior:
    """One-sided prior for a published upper limit: chi^2 += (x / sigma)^2 for x > 0, with
    sigma = limit / Phi^-1(cl), so the limit is the one-sided quantile at cl. Used for the
    cosmological bound on the sum of neutrino masses (2026-10-12 decision: Sum < 0.055 eV at
    95%, SPT-3G D1 + DESI DR2, sigma = 0.0334 eV), applied to the derived quantity."""
    name: str
    limit: float
    cl: float = 0.95

    @cached_property
    def sigma(self) -> float:
        from scipy.stats import norm
        return self.limit / norm.ppf(self.cl)

    def chi2(self, x: float) -> float:
        return 0.0 if x <= 0.0 else (x / self.sigma) ** 2


def prior_from_spec(spec: Mapping) -> object:
    """A prior from its config entry: {"name": ..., "type": "gaussian", "centre": ...,
    "width": ...} or {"name": ..., "type": "half_gaussian", "limit": ..., "cl": ...}."""
    kind = str(spec.get("type", "")).lower()
    if kind == "gaussian":
        return GaussianPrior(spec["name"], float(spec["centre"]), float(spec["width"]))
    if kind in ("half_gaussian", "half-gaussian"):
        return HalfGaussianPrior(spec["name"], float(spec["limit"]), float(spec.get("cl", 0.95)))
    raise ValueError(f"Unknown prior type {spec.get('type')!r} in {dict(spec)}; use 'gaussian' or 'half_gaussian'.")


@dataclass(frozen=True)
class Parameter:
    """A free parameter: name, starting value, box bounds (physical units), the space it is
    fitted in, and the size of a typical step in that space (one decade for log10)."""
    name: str
    start: float
    bounds: Tuple[float, float]
    space: str = "linear"
    scale: Optional[float] = None

    def __post_init__(self):
        if self.space not in SPACES:
            raise ValueError(f"Parameter {self.name}: space must be one of {SPACES}, got {self.space!r}")
        lo, hi = self.bounds
        if not lo < hi:
            raise ValueError(f"Parameter {self.name}: bounds {self.bounds} are empty")
        if self.space == "log10" and lo <= 0:
            raise ValueError(f"Parameter {self.name}: a log10 parameter needs a positive lower bound")
        if not lo <= self.start <= hi:
            raise ValueError(f"Parameter {self.name}: start {self.start} outside bounds {self.bounds}")

    @property
    def step(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        if self.space == "log10":
            return 1.0
        return (self.bounds[1] - self.bounds[0]) / 10.0

    def to_u(self, x: float) -> float:
        return (np.log10(x) if self.space == "log10" else float(x)) / self.step

    def from_u(self, u: float) -> float:
        t = u * self.step
        return float(10.0 ** t) if self.space == "log10" else float(t)


@dataclass
class ParameterSet:
    """The free parameters of a fit, in order, and the values held fixed."""
    free: List[Parameter] = field(default_factory=list)
    held: Dict[str, float] = field(default_factory=dict)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.free]

    def parameter(self, name: str) -> Parameter:
        for p in self.free:
            if p.name == name:
                return p
        raise KeyError(name)

    def as_values(self, u: Sequence[float]) -> Dict[str, float]:
        """Named physical values (held ones included) from an internal vector."""
        out = dict(self.held)
        out.update({p.name: p.from_u(ui) for p, ui in zip(self.free, u)})
        return out

    def to_internal(self, values: Mapping[str, float]) -> np.ndarray:
        """Internal vector from named values (the free parameters must be present),
        clipped into the bounds."""
        lo, hi = self.bounds_internal()
        u = np.array([p.to_u(values[p.name]) for p in self.free], dtype=float)
        return np.clip(u, lo, hi)

    def bounds_internal(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([p.to_u(p.bounds[0]) for p in self.free], dtype=float)
        hi = np.array([p.to_u(p.bounds[1]) for p in self.free], dtype=float)
        return lo, hi

    def start_values(self) -> Dict[str, float]:
        out = dict(self.held)
        out.update({p.name: p.start for p in self.free})
        return out

    def holding(self, values: Mapping[str, float]) -> "ParameterSet":
        """A copy with the named free parameters held at the given values."""
        unknown = sorted(set(values) - set(self.names))
        if unknown:
            raise ValueError(f"Cannot hold parameters that are not free: {unknown}")
        return ParameterSet([p for p in self.free if p.name not in values],
                            {**self.held, **{k: float(v) for k, v in values.items()}})
