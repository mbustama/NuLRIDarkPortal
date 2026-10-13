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
the flux at the detector -- for the configured sources in one coupled evolution of
neutrinos and antineutrinos when the ratio is fixed, or the responses to neutrino and to
antineutrino sources separately when it floats; for the configured source composition, or
for the pure-flavor sources when the composition floats -- and combines them on demand.

Aliases: s12sq, s13sq, s23sq (sin^2 of the mixing angles) are mapped to th*_deg.

Coupling products: the data depend on the dark coupling and the sterile mixing only through
products, so exclusion planes scan those directly (decided 2026-10-13). For a texture with
|U|^2 = max_alpha |U_alpha4|^2 (the common value of the benchmark textures' two entries):
  g_D_u2      = g_D |U|^2          sets g_D = g_D_u2 / |U|^2 (resonant absorption; with
                                   epsilon' = 0 everything depends on it alone);
  g_D_eps_u2  = g_D epsilon' |U|^2 sets g_D = g_D_eps_u2 / (epsilon' |U|^2) (the coherent
                                   potential, which is proportional to it).

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
PRODUCTS = ("g_D_u2", "g_D_eps_u2")
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
    if name in PRODUCTS:
        return "dark_portal", "g_D"
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
        self.ratio_free = "nubar_to_nu_ratio" in free_names
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
        products = [n for n in PRODUCTS if n in values]
        if len(products) > 1 or (products and "g_D" in values):
            raise ValueError(f"Set g_D through at most one of g_D, {PRODUCTS}; got {sorted(values)}")
        for name, v in values.items():
            if name in LINEAR_KEYS or name in PRODUCTS:
                continue
            section, key = solver_key(name)
            if name in ALIASES:
                v = sin2_to_deg(v)
            out[(section, key)] = v
        if products:
            dp = self.base["dark_portal"]

            def get(k):
                return float(out.get(("dark_portal", k), dp[k]))
            u2 = max(abs(get("u_e")), abs(get("u_mu")), abs(get("u_tau"))) ** 2
            name = products[0]
            scale = u2 if name == "g_D_u2" else u2 * get("epsilon_prime")
            if scale <= 0.0:
                raise ValueError(f"{name} needs a nonzero mixing{'' if name == 'g_D_u2' else ' and epsilon_prime'}")
            out[("dark_portal", "g_D")] = float(values[name]) / scale
        return out

    def fluxes(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """For the propagation parameters in `values`: (E_bins, F, None) with F the flux at the
        detector for the configured sources, or, when the nubar/nu ratio floats,
        (E_bins, F_from_nu, F_from_nb), the responses to unit neutrino and antineutrino
        sources. With the pure-flavor basis the fluxes have a leading source axis."""
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
        if self.ratio_free:
            F_a, F_b = dps.propagate_nu_nubar(E, params, sources=sources)
        else:
            F_a, F_b = dps.propagate_weighted(E, params, sources=sources), None
        self.n_solver_calls += 1
        out = (E, F_a, F_b)
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def flux(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray]:
        """(E_bins, flux at the detector) with the source composition and the
        neutrino/antineutrino ratio applied, as run_solver weights them."""
        E, F_a, F_b = self.fluxes(values)
        if self.basis is not None:
            comp = self.composition(values)
            F_a = np.tensordot(comp[self.basis], F_a, axes=(0, 0))
            if F_b is not None:
                F_b = np.tensordot(comp[self.basis], F_b, axes=(0, 0))
        if F_b is None:
            return E, F_a
        src = dict(self.base["source"])
        src["nubar_to_nu_ratio"] = float(values.get("nubar_to_nu_ratio", src["nubar_to_nu_ratio"]))
        w_nu, w_nb = dps.injection_weights(src)
        return E, w_nu * F_a + w_nb * F_b

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
