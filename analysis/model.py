"""
model.py

The physics model: named parameter values -> fluxes at the detector -> energy-averaged
flavor fractions.

Every parameter that changes the propagation is a key of the solver's configuration, so
an evaluation overrides those keys in a copy of the base solver config and calls the
solver's own parsing and flux computation (dark_portal_solver.build_solver_params and
propagate_nu_nubar): the solver's defaults and checks stay the single source of truth.
Three kinds of parameter need no solver call, because the fluxes are linear in them
(checked to 1.6e-15): the source flavor composition, the neutrino/antineutrino ratio and
the normalization. The model therefore caches, per point of the propagation parameters,
the neutrino and antineutrino fluxes -- for the configured source composition when it is
fixed, or for the pure-flavor sources when it is free -- and combines them on demand.

Aliases: s12sq, s13sq, s23sq (sin^2 of the mixing angles) are mapped to th*_deg.

Source composition: components that are not floated keep their configured values, except
one that takes the remainder so that they sum to 1 -- f_tau_S when f_e_S and f_mu_S both
float (the theory notes' parametrization), f_mu_S otherwise (f_e_S floated with f_tau_S
fixed, the agreed systematics). A negative remainder is outside the model (ValueError).
"""
import copy
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import surfaces  # noqa: F401  (ensures the src/ path is set)
import dark_portal_solver as dps
import dp_core
from dp_config import DEFAULT_CONFIG, merge_config_with_defaults
from flux_postprocessing import compute_energy_averaged_ratios

SOURCE_KEYS = ("f_e_S", "f_mu_S", "f_tau_S")
LINEAR_KEYS = SOURCE_KEYS + ("nubar_to_nu_ratio", "Phi_0")
ALIASES = {"s12sq": "th12_deg", "s13sq": "th13_deg", "s23sq": "th23_deg"}
_SECTIONS = ("cosmology", "standard_mixing", "dark_portal", "source", "grid")


def _section_of(key: str) -> str:
    hits = [s for s in _SECTIONS if key in DEFAULT_CONFIG[s]]
    if len(hits) != 1:
        raise KeyError(f"{key!r} is not a solver parameter (or is ambiguous): known keys are those of "
                       f"DEFAULT_CONFIG sections {_SECTIONS}, plus aliases {sorted(ALIASES)}")
    return hits[0]


def sin2_to_deg(s: float) -> float:
    """Mixing angle in degrees from sin^2 (clipped to [1e-10, 1], as FrequentistAnalysis did)."""
    return float(np.degrees(np.arcsin(np.sqrt(np.clip(float(s), 1e-10, 1.0)))))


def solver_key(name: str) -> Tuple[str, str]:
    """(section, key) of the solver config entry a parameter sets."""
    key = ALIASES.get(name, name)
    return _section_of(key), key


def sum_of_masses(values: Mapping, base: Mapping) -> float:
    """Sum of the three neutrino masses (eV) from m_lightest, the splittings and the ordering,
    read from `values` where present and from the base solver config otherwise."""
    def get(name, section):
        return values.get(name, base[section][name])
    m = dp_core.get_mass_eigenvalues(float(get("m_lightest_eV", "cosmology")), float(get("dm21_sq", "standard_mixing")),
                                     float(get("dm31_sq", "standard_mixing")), str(get("mass_ordering", "cosmology")))
    return float(np.sum(m))


class FlavorModel:
    """Energy-averaged flavor fractions at the detector.

    Arguments:
      solver_config: partial solver configuration (DEFAULT_CONFIG layout) with the settings
        every evaluation shares; execution is forced to one core and silence (parallelism
        happens across fits, not inside the solver).
      free_names: the parameters a fit may vary (decides whether the source composition
        needs the pure-flavor basis).
      cache_size: number of propagation points kept.
    """

    def __init__(self, solver_config: Mapping, free_names: Sequence[str] = (), cache_size: int = 512):
        base = merge_config_with_defaults(copy.deepcopy(dict(solver_config)))
        base["execution"].update(n_cores=1, verbose=0, use_gpu=False)
        self.base = base
        free_src = [k for k in SOURCE_KEYS if k in free_names]
        # Pure-flavor sources needed when the composition is free: the floated ones, the one
        # taking the remainder, and any fixed at a nonzero value
        self.basis = None
        self.remainder = None
        if free_src:
            self.remainder = "f_tau_S" if {"f_e_S", "f_mu_S"} <= set(free_src) else "f_mu_S"
            if self.remainder in free_src:
                raise ValueError("Float at most two of f_e_S, f_mu_S, f_tau_S: the third is the remainder")
            self.basis = [i for i, k in enumerate(SOURCE_KEYS)
                          if k in free_names or k == self.remainder or float(base["source"][k]) != 0.0]
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._cache_size = int(cache_size)
        self.n_solver_calls = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------
    def _overrides(self, values: Mapping) -> Dict[Tuple[str, str], object]:
        out = {}
        for name, v in values.items():
            if name in LINEAR_KEYS:
                continue
            section, key = solver_key(name)
            if name in ALIASES:
                v = sin2_to_deg(v)
            out[(section, key)] = v
        return out

    def fluxes(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(E_bins, F_nu, F_nubar) for the propagation parameters in `values`; with the
        pure-flavor basis the fluxes have a leading source axis."""
        ov = self._overrides(values)
        key = tuple(sorted((s, k, v if isinstance(v, str) else float(v)) for (s, k), v in ov.items()))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.n_cache_hits += 1
            return hit
        cfg = copy.deepcopy(self.base)
        for (section, k), v in ov.items():
            cfg[section][k] = v
        if self.basis is not None:
            # any valid composition passes the solver's checks; the basis fluxes ignore it
            cfg["source"].update(f_e_S=1.0, f_mu_S=0.0, f_tau_S=0.0)
        E, params = dps.build_solver_params(cfg)
        params["show_progress"] = False
        sources = None if self.basis is None else [np.eye(3)[i] for i in self.basis]
        F_nu, F_nubar = dps.propagate_nu_nubar(E, params, sources=sources)
        self.n_solver_calls += 1
        out = (E, F_nu, F_nubar)
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def flux(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray]:
        """(E_bins, flux at the detector) with the source composition and the
        neutrino/antineutrino ratio applied, as run_solver weights them."""
        E, F_nu, F_nubar = self.fluxes(values)
        if self.basis is not None:
            comp = self.composition(values)
            F_nu = np.tensordot(comp[self.basis], F_nu, axes=(0, 0))
            F_nubar = np.tensordot(comp[self.basis], F_nubar, axes=(0, 0))
        mode = str(self.base["source"]["is_antineutrino"]).strip().lower()
        if mode in ("sum", "both"):
            ratio = float(values.get("nubar_to_nu_ratio", self.base["source"]["nubar_to_nu_ratio"]))
            if ratio == 0.0:
                F = F_nu
            elif not np.isfinite(ratio):
                F = F_nubar
            else:
                F = (1.0 / (1.0 + ratio)) * F_nu + (ratio / (1.0 + ratio)) * F_nubar
        elif mode in ("true", "antineutrino", "antinu", "nubar"):
            F = F_nubar
        else:
            F = F_nu
        return E, F

    def composition(self, values: Mapping) -> np.ndarray:
        """(f_e_S, f_mu_S, f_tau_S) at the source, with the remainder rule of the module
        docstring when the composition floats."""
        comp = {k: float(values.get(k, self.base["source"][k])) for k in SOURCE_KEYS}
        if self.remainder is not None:
            comp[self.remainder] = 1.0 - sum(v for k, v in comp.items() if k != self.remainder)
            if comp[self.remainder] < -1e-12:
                raise ValueError(f"Source composition {comp} has a negative {self.remainder}")
        return np.array([comp[k] for k in SOURCE_KEYS])

    def fractions(self, values: Mapping) -> np.ndarray:
        """Energy-averaged (f_e, f_mu, f_tau) at the detector, f_tau = 1 - f_e - f_mu."""
        E, F = self.flux(values)
        r = compute_energy_averaged_ratios(E, F)
        return np.array([r[0], r[1], 1.0 - r[0] - r[1]])
