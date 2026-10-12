"""
config.py

An analysis configuration: one JSON file that says which data term to use, how the solver is
set up, which parameters float (with their starts, bounds and fit spaces), which priors apply,
and which fits and scans to run. Example: analysis/configs/pion_2050_case_5.json.

  {
    "name": "...",                       label of the analysis (output file stems)
    "data": {"surface": "pion_2050_case_5"},
    "solver": {section: {key: value}},   partial solver configuration (DEFAULT_CONFIG layout);
                                         anything not given takes the solver's default
    "parameters": {name: {"start", "bounds", "space": "linear"|"log10", "scale"}},
    "priors": [{"name", "type": "gaussian"|"half_gaussian", ...}],
    "fit": {FitSettings fields},         settings of every minimization
    "profiles": {name: {"range": [lo, hi], "n": int}},
    "planes": [{"x", "y", "x_range", "y_range", "nx", "ny"}],
    "thresholds": {"1d": {label: dchi2}, "2d": {label: dchi2}},
    "asimov": {name: value},             parameters of the Asimov data set (--asimov)
    "output_dir": "output/analysis/..."  relative to the repository
  }

Profile and plane ranges are in physical units; a parameter fitted in log10 space is scanned
in equal steps of log10. Keys starting with '_' are comments.
"""
import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import REPO_DIR, surfaces
from .fitting import FitSettings
from .model import FlavorModel
from .objective import Objective
from .parameters import Parameter, ParameterSet, prior_from_spec

TOP_KEYS = {"name", "data", "solver", "parameters", "priors", "fit", "profiles", "planes",
            "thresholds", "asimov", "output_dir"}
DEFAULT_THRESHOLDS = {
    # Delta chi^2 for two-sided intervals of 1 and 2 parameters (Wilks): 68.27%, 90%, 95%
    "1d": {"68.27%": 1.0, "90%": 2.706, "95%": 3.841},
    "2d": {"68.27%": 2.296, "90%": 4.605, "95%": 5.991},
}


def _strip_comments(d):
    if isinstance(d, dict):
        return {k: _strip_comments(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [_strip_comments(v) for v in d]
    return d


@dataclass
class Scan1D:
    name: str
    grid: np.ndarray          # physical values, ascending


@dataclass
class Scan2D:
    x: str
    y: str
    x_grid: np.ndarray
    y_grid: np.ndarray


@dataclass
class AnalysisConfig:
    """A validated analysis configuration (see the module docstring)."""
    name: str
    surface: str
    solver: dict
    parameters: List[Parameter]
    priors: list
    fit: FitSettings
    profiles: List[Scan1D] = field(default_factory=list)
    planes: List[Scan2D] = field(default_factory=list)
    thresholds: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_THRESHOLDS))
    asimov: dict = field(default_factory=dict)
    output_dir: str = ""
    raw: dict = field(default_factory=dict)
    path: str = ""

    @property
    def free_names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def parameter(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"{name!r} is not a free parameter of {self.name}; free: {self.free_names}")

    def parameter_set(self) -> ParameterSet:
        return ParameterSet(list(self.parameters), {})

    def model(self, cache_size: int = 512) -> FlavorModel:
        return FlavorModel(self.solver, self.free_names, cache_size=cache_size)

    def objective(self, model: Optional[FlavorModel] = None, data_centre: Optional[Tuple[float, float]] = None) -> Objective:
        """The chi^2 of this analysis; data_centre re-centres the data surface (Asimov data,
        pseudo-experiments)."""
        return Objective(model or self.model(), self.surface, self.priors, data_centre=data_centre)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def output_path(self, *parts: str) -> str:
        return os.path.join(self.output_dir, *parts)


def _grid(p: Parameter, lo: float, hi: float, n: int) -> np.ndarray:
    if not (p.bounds[0] <= lo < hi <= p.bounds[1]):
        raise ValueError(f"Scan range [{lo}, {hi}] of {p.name} is empty or outside its bounds {p.bounds}")
    if n < 2:
        raise ValueError(f"Scan of {p.name} needs at least 2 points, got {n}")
    if p.space == "log10":
        return np.logspace(np.log10(lo), np.log10(hi), int(n))
    return np.linspace(lo, hi, int(n))


def from_dict(raw: Mapping, path: str = "") -> AnalysisConfig:
    """Validates a configuration dictionary (comments allowed) and returns an AnalysisConfig."""
    d = _strip_comments(copy.deepcopy(dict(raw)))
    unknown = set(d) - TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown configuration keys {sorted(unknown)}; known: {sorted(TOP_KEYS)}")
    for key in ("name", "data", "parameters"):
        if key not in d:
            raise ValueError(f"Configuration needs '{key}'")
    surface = d["data"].get("surface")
    surfaces.surface(surface)                                  # unknown names fail here
    params = []
    for name, spec in d["parameters"].items():
        spec = dict(spec)
        bad = set(spec) - {"start", "bounds", "space", "scale"}
        if bad:
            raise ValueError(f"Parameter {name}: unknown keys {sorted(bad)}")
        params.append(Parameter(name, float(spec["start"]), tuple(float(b) for b in spec["bounds"]),
                                spec.get("space", "linear"), spec.get("scale")))
    cfg = AnalysisConfig(
        name=str(d["name"]), surface=surface, solver=d.get("solver", {}), parameters=params,
        priors=[prior_from_spec(p) for p in d.get("priors", [])],
        fit=FitSettings.from_dict(d.get("fit")), asimov=dict(d.get("asimov", {})),
        raw=dict(raw), path=path)
    thr = copy.deepcopy(DEFAULT_THRESHOLDS)
    for k, v in d.get("thresholds", {}).items():
        if k not in thr:
            raise ValueError(f"thresholds: use '1d' and '2d', got {k!r}")
        thr[k] = {str(a): float(b) for a, b in v.items()}
    cfg.thresholds = thr
    for name, spec in d.get("profiles", {}).items():
        p = cfg.parameter(name)
        cfg.profiles.append(Scan1D(name, _grid(p, float(spec["range"][0]), float(spec["range"][1]), int(spec["n"]))))
    for spec in d.get("planes", []):
        px, py = cfg.parameter(spec["x"]), cfg.parameter(spec["y"])
        cfg.planes.append(Scan2D(px.name, py.name,
                                 _grid(px, float(spec["x_range"][0]), float(spec["x_range"][1]), int(spec["nx"])),
                                 _grid(py, float(spec["y_range"][0]), float(spec["y_range"][1]), int(spec["ny"]))))
    out = d.get("output_dir") or os.path.join("output", "analysis", cfg.name)
    cfg.output_dir = out if os.path.isabs(out) else os.path.join(REPO_DIR, out)
    # Build the model once to validate every solver key and parameter name now, not in a worker
    model = cfg.model(cache_size=1)
    from .model import solver_key, LINEAR_KEYS
    for p in params:
        if p.name not in LINEAR_KEYS:
            solver_key(p.name)
    for name in cfg.asimov:
        if name not in LINEAR_KEYS:
            solver_key(name)
    del model
    return cfg


def load(path: str) -> AnalysisConfig:
    """Reads and validates a configuration file."""
    with open(path) as f:
        raw = json.load(f)
    return from_dict(raw, path=os.path.abspath(path))
