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
    """One-sided prior for a published upper limit: chi^2 += (x / sigma)^2 for x > 0. quantile
    'one_sided' (the default): sigma = limit / Phi^-1(cl), the limit the one-sided quantile at cl of
    a Gaussian centred at zero (the 2026-10-12 decision: Sum < 0.055 eV at 95%, sigma = 0.0334 eV);
    'half': sigma = limit / Phi^-1((1 + cl) / 2), the limit holding the fraction cl of the half-Gaussian
    bounded at zero -- the faithful reading of a cosmological upper limit at cl (a posterior bounded at
    zero), and the same threshold as the planes' Delta chi^2 = 3.84 at 95%: Table II of arXiv:2606.17132's
    limits on the mass sum, 0.12 eV now and 0.09 eV for projections, at 95%, sigma = limit / 1.960
    (2026-10-18). 'one_sided' at 95% puts only 90% of the half-Gaussian below the limit. Applied to the
    derived quantity sum_masses_eV."""
    name: str
    limit: float
    cl: float = 0.95
    quantile: str = "one_sided"

    def __post_init__(self):
        if self.quantile not in ("one_sided", "half"):
            raise ValueError(f"half_gaussian prior on {self.name}: quantile must be 'one_sided' or 'half', got "
                             f"{self.quantile!r}")
        if not 0.0 < self.cl < 1.0 or not self.limit > 0.0:
            raise ValueError(f"half_gaussian prior on {self.name}: needs 0 < cl < 1 and limit > 0")

    @cached_property
    def sigma(self) -> float:
        from scipy.stats import norm
        q = self.cl if self.quantile == "one_sided" else 0.5 * (1.0 + self.cl)
        return self.limit / norm.ppf(q)

    def chi2(self, x: float) -> float:
        return 0.0 if x <= 0.0 else (x / self.sigma) ** 2


def prior_from_spec(spec: Mapping) -> object:
    """A prior from its config entry: {"name": ..., "type": "gaussian", "centre": ...,
    "width": ...}, {"name": ..., "type": "half_gaussian", "limit": ..., "cl": ..., "quantile":
    "one_sided" | "half"}, or
    {"type": "nufit_6.1_pairwise", "ordering": "normal" | "inverted"} (nufit.py: NuFIT 6.1's
    three pairwise tables on the six mixing quantities at once; no name)."""
    kind = str(spec.get("type", "")).lower()
    if kind == "gaussian":
        return GaussianPrior(spec["name"], float(spec["centre"]), float(spec["width"]))
    if kind in ("half_gaussian", "half-gaussian"):
        return HalfGaussianPrior(spec["name"], float(spec["limit"]), float(spec.get("cl", 0.95)),
                                 str(spec.get("quantile", "one_sided")))
    if kind == "nufit_6.1_pairwise":
        from .nufit import NuFitPairwisePrior
        bad = set(spec) - {"type", "ordering"}
        if bad or "ordering" not in spec:
            raise ValueError(f"nufit_6.1_pairwise takes only 'ordering' ('normal' or 'inverted'); got {dict(spec)}")
        return NuFitPairwisePrior(str(spec["ordering"]))
    raise ValueError(f"Unknown prior type {spec.get('type')!r} in {dict(spec)}; use 'gaussian', 'half_gaussian' "
                     f"or 'nufit_6.1_pairwise'.")


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
