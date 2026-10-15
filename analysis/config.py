"""
config.py

An analysis configuration: one JSON file that says which data term to use, how the solver is
set up, which parameters float (with their starts, bounds and fit spaces), which priors apply,
and which fits and scans to run. Example: analysis/configs/pion_2050_case_5.json.

  {
    "name": "...",                       label of the analysis (output file stems)
    "data": {"surface": "pion_2050_case_5"[, "spectrum": "ic26_astrophysical"]
             [, "events": {"hese": {...}, "dr2": {...}}, "flavor_band_eV": [lo, hi]]},
                                         the flavor surface, and optionally a measured spectrum
                                         (spectra.py; needs the extragalactic flux in physical units),
                                         or the event likelihoods (events.py; HESE and/or DR2 tracks,
                                         never with the spectrum, which is fitted to the same events), with
                                         the flavor fractions averaged over flavor_band_eV of the grid;
                                         with the events the surface may be left out (2026-10-14: HESE's
                                         events are also MESE's, so the event data set drops its surface)
    "solver": {section: {key: value}},   partial solver configuration (DEFAULT_CONFIG layout);
                                         anything not given takes the solver's default
    "parameters": {name: {"start", "bounds", "space": "linear"|"log10", "scale"}},
    "priors": [{"name", "type": "gaussian"|"half_gaussian", ...}],
    "fit": {FitSettings fields},         settings of every minimization
    "profiles": {name: {"range": [lo, hi], "n": int}},
    "planes": [{"x", "y", "x_range", "y_range", "nx", "ny", "raster"}],
    "thresholds": {"1d": {label: dchi2}, "2d": {label: dchi2}},
    "asimov": {name: value},             parameters of the Asimov data set (--asimov)
    "output_dir": "output/analysis/..."  relative to the repository
    "flux_basis": "auto" | "pure_flavors"  the solver runs per pure-flavor source even with the composition
                                         fixed (2026-10-14): a fixed-composition analysis then shares its
                                         runs with a fitted-composition one (NULRI_FLUX_CACHE); the composition
                                         is linear (to 1.6e-15)
  }

Profile and plane ranges are in physical units; a parameter fitted in log10 space is scanned
in equal steps of log10. A plane with "raster": name is read per value of that parameter (e.g.
per mediator mass): Delta chi^2 from each column's own minimum, at the 1D thresholds -- the
convention decided for exclusion planes on 2026-10-13 (95% at 3.84). Otherwise it is read as a
joint region at the 2D thresholds. Keys starting with '_' are comments.
"""
import copy
import dataclasses
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import REPO_DIR, spectra, surfaces
from .fitting import FitSettings
from .model import P_NODES_N, FlavorModel
from .objective import Objective
from .parameters import Parameter, ParameterSet, prior_from_spec

TOP_KEYS = {"name", "data", "solver", "parameters", "priors", "fit", "profiles", "planes",
            "thresholds", "asimov", "output_dir", "flux_basis"}
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
    raster: Optional[str] = None     # read per value of this parameter (x or y), at the 1D thresholds


@dataclass
class AnalysisConfig:
    """A validated analysis configuration (see the module docstring)."""
    name: str
    surface: Optional[str]
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
    spectrum: Optional[str] = None
    p_nodes_n: Optional[int] = None
    events: Optional[dict] = None
    flavor_band: Optional[Tuple[float, float]] = None
    flux_basis: str = "auto"

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

    def p_nodes(self) -> Optional[List[float]]:
        """With p_index free: its nodes (parameters.p_index.nodes, default P_NODES_N) spanning its bounds, where the
        model runs the solver."""
        if "p_index" not in self.free_names:
            return None
        lo, hi = self.parameter("p_index").bounds
        return [float(v) for v in np.linspace(lo, hi, self.p_nodes_n or P_NODES_N)]

    def model(self, cache_size: int = 512) -> FlavorModel:
        # p_index interpolated between the nodes' solver runs: that many times the cache for the same points
        n = 1 if self.p_nodes() is None else len(self.p_nodes())
        sites = ()
        if self.events is not None and "future" in self.events:
            from .future_events import config_sites
            sites = config_sites((self.events["future"] or {}).get("config", "All8_base"))
        return FlavorModel(self.solver, self.free_names, cache_size=cache_size * n, p_nodes=self.p_nodes(),
                           sky=self.events is not None, flavor_band_eV=self.flavor_band,
                           pure_flavor_basis=self.flux_basis == "pure_flavors", sites=sites)

    def asimov_values(self) -> Dict[str, float]:
        """The values of the Asimov data set: the configuration's 'asimov' values, the free parameters' starts
        otherwise."""
        values = {p.name: p.start for p in self.parameters}
        values.update(self.asimov)
        return values

    def objective(self, model: Optional[FlavorModel] = None, data_centre: Optional[Tuple[float, float]] = None) -> Objective:
        """The chi^2 of this analysis; data_centre re-centres the data surface (Asimov data,
        pseudo-experiments). With events, data_centre also marks Asimov data: the event data become the
        expectation at asimov_values() (the event nuisances at their priors' centres)."""
        model = model or self.model()
        events = None
        if self.events is not None:
            terms = []
            icecube = {k: v for k, v in self.events.items() if k in ("hese", "dr2")}
            if icecube:
                from .events import EventLikelihood
                terms.append(EventLikelihood(model, icecube))
            if "future" in self.events:
                from .future_events import FutureEventLikelihood
                terms.append(FutureEventLikelihood(model, self.events["future"] or {}))
            events = terms[0] if len(terms) == 1 else _EventTerms(terms)
            if data_centre is not None:
                events.set_asimov(self.asimov_values())
        return Objective(model, self.surface, self.priors, data_centre=data_centre, spectrum=self.spectrum,
                         events=events)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]

    def identity(self, data_centre=None, extra=None) -> str:
        """Digest of everything that decides a chi^2 or a constrained minimum: the data term
        (and its re-centring), the solver configuration merged with the solver's defaults, the
        solver's source code (dp_core._source_digest), the free parameters with their bounds
        and spaces, and the priors; plus `extra` (a scan's grid). Checkpoints carry it, so
        results computed under another configuration, other defaults or other code are never
        reused. Fit settings and starts are left out: rerunning with better ones resumes,
        keeping the best fits."""
        from dp_config import merge_config_with_defaults
        import dp_core
        from .model import split_analysis_keys
        sc, own = split_analysis_keys(self.solver)          # the analysis' own keys (potential sign, coupling split)
        solver = merge_config_with_defaults(sc)
        solver = {k: v for k, v in solver.items() if k not in ("execution", "output")}
        if own:
            solver["analysis_dark_portal"] = own
        d = dict(data=self.surface, centre=None if data_centre is None else [float(c) for c in data_centre],
                 spectrum=self.spectrum, p_nodes=self.p_nodes(), events=self.events, flavor_band=self.flavor_band,
                 **({} if self.flux_basis == "auto" else dict(flux_basis=self.flux_basis)),
                 solver=solver, code=dp_core._source_digest(),
                 parameters=[[p.name, p.space, list(p.bounds)] for p in self.parameters],
                 priors=[[type(p).__name__, sorted(dataclasses.asdict(p).items())] for p in self.priors], extra=extra)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def output_path(self, *parts: str) -> str:
        return os.path.join(self.output_dir, *parts)


class _EventTerms:
    """Several independent event likelihoods (IceCube's HESE and DR2; the future detectors) as one term."""

    def __init__(self, terms):
        self.terms = list(terms)
        self.last = {}

    def chi2(self, values):
        total = sum(t.chi2(values) for t in self.terms)
        self.last = {type(t).__name__: t.last for t in self.terms}
        return total

    def set_asimov(self, values, nuisances=None):
        for t in self.terms:
            t.set_asimov(values)


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
    if surface is None:
        # the event likelihoods alone (Mauricio, 2026-10-14: HESE's events are also MESE's, so the event data set
        # drops the MESE flavor surface rather than count them twice; HESE's topologies carry the flavor)
        if not d["data"].get("events"):
            raise ValueError("data.surface may be left out only with data.events (the events carry the flavor)")
    else:
        surfaces.surface(surface)                              # unknown names fail here
    bad = set(d["data"]) - {"surface", "spectrum", "events", "flavor_band_eV"}
    if bad:
        raise ValueError(f"data: unknown keys {sorted(bad)}; known: surface, spectrum, events, flavor_band_eV")
    spectrum = d["data"].get("spectrum")
    if spectrum is not None and spectrum not in spectra.SPECTRA:
        raise ValueError(f"data.spectrum must be one of {sorted(spectra.SPECTRA)}, got {spectrum!r}")
    events = d["data"].get("events")
    if events is not None:
        events = dict(events)
        if spectrum is not None:
            raise ValueError("data.events and data.spectrum exclude each other: IceCube's 2026 spectrum is fitted to "
                             "largely the same events (Mauricio, 2026-10-14: events replace it)")
        if not events or set(events) - {"hese", "dr2", "future"}:
            raise ValueError(f"data.events: use 'hese', 'dr2' and/or 'future', got {sorted(events)}")
        known = {"hese": {"mc", "detector", "atm"}, "dr2": {"configs", "E_reco_min_log10", "dec_min_deg"},
                 "future": {"config"}}
        for k, v in events.items():
            if isinstance(v, dict) and set(v) - known[k]:
                raise ValueError(f"data.events.{k}: unknown keys {sorted(set(v) - known[k])}; known: {sorted(known[k])}")
        if (d.get("solver", {}).get("sky") or {}).get("extragalactic_norm") is None:
            raise ValueError("data.events needs solver.sky.extragalactic_norm (the flux in physical units)")
    flavor_band = d["data"].get("flavor_band_eV")
    if flavor_band is not None:
        flavor_band = (float(flavor_band[0]), float(flavor_band[1]))
        if not 0.0 < flavor_band[0] < flavor_band[1]:
            raise ValueError(f"data.flavor_band_eV must be [lo, hi] with 0 < lo < hi, got {flavor_band}")
    params = []
    p_nodes_n = None
    for name, spec in d["parameters"].items():
        spec = dict(spec)
        bad = set(spec) - {"start", "bounds", "space", "scale"} - ({"nodes"} if name == "p_index" else set())
        if bad:
            raise ValueError(f"Parameter {name}: unknown keys {sorted(bad)}")
        if "nodes" in spec:
            p_nodes_n = int(spec.pop("nodes"))
            if p_nodes_n < 2:
                raise ValueError(f"p_index: nodes must be >= 2, got {p_nodes_n}")
        params.append(Parameter(name, float(spec["start"]), tuple(float(b) for b in spec["bounds"]),
                                spec.get("space", "linear"), spec.get("scale")))
    cfg = AnalysisConfig(
        name=str(d["name"]), surface=surface, solver=d.get("solver", {}), parameters=params,
        priors=[prior_from_spec(p) for p in d.get("priors", [])],
        fit=FitSettings.from_dict(d.get("fit")), asimov=dict(d.get("asimov", {})),
        raw=dict(raw), path=path, spectrum=spectrum, p_nodes_n=p_nodes_n, events=events, flavor_band=flavor_band,
        flux_basis=str(d.get("flux_basis", "auto")))
    if cfg.flux_basis not in ("auto", "pure_flavors"):
        raise ValueError(f"flux_basis must be 'auto' or 'pure_flavors', got {cfg.flux_basis!r}")
    if spectrum is not None and not (cfg.solver.get("galactic") or {}).get("enabled"):
        raise ValueError("data.spectrum compares the extragalactic flux in physical units, which the model provides "
                         "with the Galactic component (galactic.enabled and sky.extragalactic_norm)")
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
        bad = set(spec) - {"x", "y", "x_range", "y_range", "nx", "ny", "raster"}
        if bad:
            raise ValueError(f"Plane {spec.get('x')}/{spec.get('y')}: unknown keys {sorted(bad)}")
        px, py = cfg.parameter(spec["x"]), cfg.parameter(spec["y"])
        raster = spec.get("raster")
        if raster is not None and raster not in (px.name, py.name):
            raise ValueError(f"Plane {px.name}/{py.name}: raster must name x or y, got {raster!r}")
        cfg.planes.append(Scan2D(px.name, py.name,
                                 _grid(px, float(spec["x_range"][0]), float(spec["x_range"][1]), int(spec["nx"])),
                                 _grid(py, float(spec["y_range"][0]), float(spec["y_range"][1]), int(spec["ny"])),
                                 raster))
    lri = (cfg.solver.get("dark_portal") or {}).get("U_LRI_struct")
    floating_u = [n for n in cfg.free_names if n in ("u_e", "u_mu", "u_tau")]
    if lri is not None and floating_u:
        warnings.warn(f"{cfg.name}: dark_portal.U_LRI_struct overrides the potential's texture, so the floating "
                      f"{floating_u} change only the absorption rates and the Debye mass, not the potential")
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
    from .objective import DERIVED
    for prior in cfg.priors:                     # a misspelt prior must fail, not go silently unused
        if prior.name not in cfg.free_names and prior.name not in DERIVED and prior.name not in LINEAR_KEYS:
            solver_key(prior.name)
    del model
    return cfg


def load(path: str) -> AnalysisConfig:
    """Reads and validates a configuration file."""
    with open(path) as f:
        raw = json.load(f)
    return from_dict(raw, path=os.path.abspath(path))
