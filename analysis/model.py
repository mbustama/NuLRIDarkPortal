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
Since 2026-10-14 (docs/dev/SCOPE_WIDENING_PLAN.md) also, for free mixing ratios and the gauge models:
  g_D_trace     = g_D Sum_alpha |U_alpha4|^2       sets g_D = g_D_trace / Sum |U|^2;
  g_D_eps_trace = g_D epsilon' Sum_alpha |U_alpha4|^2 (the dark portal's potential, ratio-independent axis);
  g_D_eps       = g_D epsilon'                      (L_mu - L_tau's potential: g' times its Z-Z' mixing).
The potential's sign is free for the dark portal and L_mu - L_tau (decided 2026-10-14): the solver setting
dark_portal.potential_sign = -1 (this package only; default +1) makes epsilon' negative (g_D stays positive).
The split of a potential product (the eps products; decided 2026-10-14, docs/dev/SCOPE_WIDENING_PLAN.md decision 4):
the relics' screening grows with each coupling separately, so the solver setting dark_portal.coupling_split (this
package only) chooses how a product P sets them, with x = g_D |U|^2 the neutrinos' side in the product's own
normalization (|U|^2 as above; 1 for g_D_eps):
  'fixed' (default)  epsilon' from the configuration, g_D = P / (epsilon' |U|^2), as before;
  'balanced'         x = epsilon' = sqrt(P): screening negligible wherever the data reach (the first curve of
                     each potential plane);
  a number x         the neutrinos' side fixed at x and epsilon' = P / x (needs P <= x <= 1, so epsilon' <= 1):
                     the variants whose minimum over x is the bound profiled over all splits (the second curve;
                     tools/combine_split_planes.py).

Solver runs can also be shared on disk by every process (NULRI_FLUX_CACHE=<directory>; 2026-10-14): the planes'
repair passes re-evaluate every point on whichever worker is free, and with per-process caches most of them
re-solved (7 runs per point with p_index free). Files are keyed by the full solver configuration, the flux mode and
the solver's code digest, written atomically, and read back bit for bit; nothing is ever evicted (delete the
directory after a campaign).

The broken power law (Mauricio, 2026-10-15: IceCube's 2026 combined fit bends below ~25 TeV): with
source.E_break_GeV set, the extragalactic flux is two source populations split at the energy today E_b, each a power
law, p_index above and p_index_low below (an analysis parameter), normalized so that without new physics the flux
at Earth is exactly the broken power law, continuous at E_b (sky.extragalactic_norm then refers to the upper piece
extended to its reference energy). Each piece is its own solver run with the sources emitting only on its side
(source.injection_E_min_GeV / injection_E_max_GeV), so everything downstream -- conversion, absorption, the
secondaries an absorption above E_b sends below it -- is that population's, and each piece is interpolated in its
own index (p_nodes above, p1_nodes below). E_b is fixed.

The spectral index (Mauricio, 2026-10-14: the flux normalization and shape float): with p_index free the
model runs the solver at fixed nodes of p (p_nodes, from the parameter's bounds; P_NODES_N of them) per point of
the other propagation parameters and interpolates the fluxes between them (cubic spline in log flux), so a fit
that moves p costs no new solver runs. With 7 nodes over [2, 3] the energy-averaged fractions are interpolated to
2e-4 at worst (strong secret interactions; 4e-8 without new physics) and the flux to 0.7%. The Galactic flux does
not depend on p.

Sky mode (the event likelihoods, docs/dev/EVENT_LIKELIHOOD_PLAN.md; 2026-10-14): one solver call per point
(dark_portal_solver.propagate_sky_basis) gives the flux per declination band, flavor and kind for each pure-flavor
source and Galactic source; band_fluxes combines them, and the flavor fractions use the same run's sky average (the
bands weighted by solid angle, the Galactic ones by their pixel shares), averaged over flavor_band_eV when the grid
extends beyond the flavor measurement's band. The band arrays are rounded to single precision, so a run read back from
the disk cache (1 MB) is the run computed.

Source composition: components that are not floated keep their configured values, except
one that takes the remainder so that they sum to 1 -- f_tau_S when f_e_S and f_mu_S both
float (the theory notes' parametrization), f_mu_S otherwise (f_e_S floated with f_tau_S
fixed, the agreed systematics). A negative remainder is outside the model (ValueError).
"""
import copy
import hashlib
import json
import os
import uuid
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import surfaces  # noqa: F401  (ensures the src/ path is set)
import dark_portal_solver as dps
import dp_core
from dp_config import DEFAULT_CONFIG, merge_config_with_defaults
from flux_postprocessing import compute_energy_averaged_ratios, compute_dE_geometric

SOURCE_KEYS = ("f_e_S", "f_mu_S", "f_tau_S")
# The in-memory caches' byte budget, each (2026-10-15): the event fits' sky runs carry per-band arrays of a few MB per
# index node, and a cache sized by entries alone (512 points x the index nodes) grew past a 16-worker job's 32 GB on
# Tycho (six campaign packs killed). Override with NULRI_MODEL_CACHE_MB.
CACHE_MB_DEFAULT = 256


def _nbytes(obj) -> int:
    """The numpy arrays' bytes in a (nested) tuple, list or dict."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(_nbytes(v) for v in obj)
    return 0


class _BoundedCache(OrderedDict):
    """An LRU dict bounded by entries and by the bytes of the arrays it holds: inserting evicts the oldest entries
    until both bounds hold (keeping at least the newest)."""

    def __init__(self, maxlen: int, maxbytes: int):
        super().__init__()
        self.maxlen, self.maxbytes = int(maxlen), int(maxbytes)
        self._sizes = {}
        self.nbytes = 0

    def __setitem__(self, key, value):
        if key in self:
            self.nbytes -= self._sizes.pop(key, 0)
        super().__setitem__(key, value)
        size = _nbytes(value)
        self._sizes[key] = size
        self.nbytes += size
        while len(self) > 1 and (len(self) > self.maxlen or self.nbytes > self.maxbytes):
            self.popitem(last=False)

    def popitem(self, last=True):
        k, v = super().popitem(last)
        self.nbytes -= self._sizes.pop(k, 0)
        return k, v
PRODUCTS = ("g_D_u2", "g_D_eps_u2", "g_D_trace", "g_D_eps_trace", "g_D_eps")
# The Galactic component (galactic.enabled): its normalization and source composition (x, 1 - x, 0) are linear
# too, and so is the extragalactic normalization against it, norm_astro -- a parameter of the analysis only (a
# multiplier of sky.extragalactic_norm; default 1).
GALACTIC_KEYS = ("norm_gal", "f_e_S_gal")
ANALYSIS_DEFAULTS = {"norm_astro": 1.0}
LINEAR_KEYS = SOURCE_KEYS + ("nubar_to_nu_ratio", "Phi_0") + GALACTIC_KEYS + tuple(ANALYSIS_DEFAULTS)
ALIASES = {"s12sq": "th12_deg", "s13sq": "th13_deg", "s23sq": "th23_deg"}
DISK_CACHE_ENV = "NULRI_FLUX_CACHE"
DISK_CACHE_VERSION = 1
SITE_POLE_DEG = 89.9
P_NODES_N = 7             # solver runs per propagation point when p_index floats (fractions to ~2e-4 under strong SI)
_SECTIONS = ("cosmology", "standard_mixing", "matter_charge", "dark_portal", "secret_interactions", "source", "grid",
             "galactic")


ANALYSIS_ONLY_DARK_PORTAL = ("potential_sign", "coupling_split")
# the broken power law (2026-10-15): read here, never by the solver -- the break (GeV; None: one power law) and the
# lower piece's index (its default value; a free parameter of the analysis when floated)
ANALYSIS_ONLY_SOURCE = ("E_break_GeV", "p_index_low")
ANALYSIS_SPECTRAL = ("p_index_low",)
WINDOW_KEYS = ("injection_E_min_GeV", "injection_E_max_GeV")
# a broken power law's scale K(p) is the closed form at the base configuration's cosmology and source evolution
BPL_FIXED = ("z_max", "rho_evol", "Phi_0", "H0_km_s_Mpc", "Omega_m", "Omega_L")


def split_analysis_keys(solver_config: Mapping) -> Tuple[dict, dict]:
    """(solver configuration without this package's own keys, those keys): dark_portal.potential_sign and
    coupling_split, source.E_break_GeV and p_index_low are read here, never by the solver, whose configuration check
    would refuse them."""
    sc = copy.deepcopy(dict(solver_config))
    dpc = dict(sc.get("dark_portal", {}))
    own = {k: dpc.pop(k) for k in ANALYSIS_ONLY_DARK_PORTAL if k in dpc}
    if "dark_portal" in sc:
        sc["dark_portal"] = dpc
    src = dict(sc.get("source", {}))
    own.update({k: src.pop(k) for k in ANALYSIS_ONLY_SOURCE if k in src})
    if "source" in sc:
        sc["source"] = src
    return sc, own


def parse_coupling_split(v):
    """dark_portal.coupling_split: 'fixed', 'balanced', or the neutrinos' side x of a potential product,
    0 < x <= 1 (a number or a numeric string)."""
    if isinstance(v, str) and v.strip().lower() in ("fixed", "balanced"):
        return v.strip().lower()
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"dark_portal.coupling_split must be 'fixed', 'balanced' or a number in (0, 1], got {v!r}")
    if not (0.0 < x <= 1.0):
        raise ValueError(f"dark_portal.coupling_split must lie in (0, 1], got {x}")
    return x


def _section_of(key: str) -> str:
    hits = [s for s in _SECTIONS if key in DEFAULT_CONFIG[s]]
    if len(hits) != 1:
        raise KeyError(f"{key!r} is not a solver parameter (or is ambiguous): known keys are those of "
                       f"DEFAULT_CONFIG sections {_SECTIONS}, plus aliases {sorted(ALIASES)}")
    return hits[0]


def sin2_to_deg(s: float) -> float:
    """Mixing angle in degrees from sin^2 (clipped to [1e-10, 1], as FrequentistAnalysis did)."""
    return float(np.degrees(np.arcsin(np.sqrt(np.clip(float(s), 1e-10, 1.0)))))


_SPLINE_WEIGHTS: Dict[tuple, np.ndarray] = {}


def spline_weights(nodes, p: float) -> np.ndarray:
    """The weights of the cubic spline through `nodes` (scipy's CubicSpline, not-a-knot) on its nodes' values, at p:
    the spline is linear in them, so sum_j w_j y_j is CubicSpline(nodes, y)(p) to rounding -- without building a
    spline over every element of large arrays at every call (2026-10-15: half of an event fit's model time)."""
    key = (tuple(float(q) for q in nodes), float(p))
    w = _SPLINE_WEIGHTS.get(key)
    if w is None:
        from scipy.interpolate import CubicSpline
        x = np.asarray(nodes, dtype=np.float64)
        w = np.asarray(CubicSpline(x, np.eye(len(x)), axis=0)(float(p)), dtype=np.float64)
        if len(_SPLINE_WEIGHTS) >= 200000:
            _SPLINE_WEIGHTS.clear()
        _SPLINE_WEIGHTS[key] = w
    return w


def spline_log_interp(nodes, p: float, arrays) -> np.ndarray:
    """exp of the cubic spline in p through log(arrays) (stacked along a new first axis; floored at 1e-300)."""
    L = np.log(np.maximum(np.stack(arrays), 1e-300))
    return np.exp(np.tensordot(spline_weights(nodes, p), L, axes=(0, 0)))


_SPLINE_DERIVATIVES: Dict[tuple, np.ndarray] = {}


def spline_weight_derivatives(nodes, p: float) -> np.ndarray:
    """d/dp of spline_weights(nodes, p): the same spline's first derivative on its nodes' values (exact)."""
    key = (tuple(float(q) for q in nodes), float(p))
    w = _SPLINE_DERIVATIVES.get(key)
    if w is None:
        from scipy.interpolate import CubicSpline
        x = np.asarray(nodes, dtype=np.float64)
        w = np.asarray(CubicSpline(x, np.eye(len(x)), axis=0)(float(p), 1), dtype=np.float64)
        if len(_SPLINE_DERIVATIVES) >= 200000:
            _SPLINE_DERIVATIVES.clear()
        _SPLINE_DERIVATIVES[key] = w
    return w


# The parameters whose derivatives the model gives in closed form (site_band_flux_jacobian; 2026-10-17): the
# extragalactic and Galactic normalizations and compositions (linear) and the spectral indices (through the cubic
# spline between the solver runs at the index nodes)
GRADIENT_NAMES = ("norm_astro", "norm_gal", "f_e_S", "f_mu_S", "f_tau_S", "f_e_S_gal", "p_index", "p_index_low")


def solver_key(name: str) -> Tuple[str, str]:
    """(section, key) of the solver config entry a parameter sets."""
    if name in ANALYSIS_DEFAULTS:
        raise KeyError(f"{name!r} is a parameter of the analysis, not of the solver (default {ANALYSIS_DEFAULTS[name]})")
    if name in PRODUCTS:
        return "dark_portal", "g_D"
    key = ALIASES.get(name, name)
    return _section_of(key), key


def _json_default(o):
    """Exact JSON for the disk cache's key: arrays in full (str() would abbreviate long ones), complex as pairs."""
    if isinstance(o, np.ndarray):
        return [_json_default(x) if isinstance(x, (complex, np.complexfloating)) else x for x in o.tolist()]
    if isinstance(o, (complex, np.complexfloating)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"cannot key the disk cache on {type(o).__name__}")


def sum_of_masses(values: Mapping, base: Mapping) -> float:
    """Sum of the three neutrino masses (eV) from m_lightest, the splittings and the ordering,
    read from `values` where present and from the base solver config otherwise."""
    def get(name, section):
        return values.get(name, base[section][name])
    m = dp_core.get_mass_eigenvalues(float(get("m_lightest_eV", "cosmology")), float(get("dm21_sq", "standard_mixing")),
                                     float(get("dm31_sq", "standard_mixing")), str(get("mass_ordering", "cosmology")))
    return float(np.sum(m))


def sum_of_masses_grad(values: Mapping, base: Mapping, names: Sequence[str]) -> Dict[str, float]:
    """d(sum of the masses)/d(name), eV per unit of the name, for names among m_lightest_eV, dm21_sq and dm31_sq (=
    |Delta m^2_3l|), in closed form from dp_core.get_mass_eigenvalues: normal m = (l, sqrt(l^2 + dm21),
    sqrt(l^2 + dm31)); inverted m = (sqrt(l^2 + dm31 - dm21), sqrt(l^2 + dm31), l)."""
    def get(name, section):
        return float(values.get(name, base[section][name]))
    l, d21, d31 = get("m_lightest_eV", "cosmology"), get("dm21_sq", "standard_mixing"), get("dm31_sq", "standard_mixing")
    inverted = str(values.get("mass_ordering", base["cosmology"]["mass_ordering"])).lower().strip() == "inverted"
    m = dp_core.get_mass_eigenvalues(l, d21, d31, "inverted" if inverted else "normal")
    if inverted:
        d = {"m_lightest_eV": 1.0 + l / m[1] + l / m[0], "dm21_sq": -0.5 / m[0], "dm31_sq": 0.5 / m[1] + 0.5 / m[0]}
    else:
        d = {"m_lightest_eV": 1.0 + l / m[1] + l / m[2], "dm21_sq": 0.5 / m[1], "dm31_sq": 0.5 / m[2]}
    return {n: float(d[n]) for n in names if n in d}


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

    def __init__(self, solver_config: Mapping, free_names: Sequence[str] = (), cache_size: int = 512,
                 p_nodes: Optional[Sequence[float]] = None, sky: bool = False,
                 flavor_band_eV: Optional[Sequence[float]] = None, pure_flavor_basis: bool = False,
                 sites: Sequence[float] = (), p1_nodes: Optional[Sequence[float]] = None, templates: Sequence = ()):
        sc, own = split_analysis_keys(solver_config)
        self.potential_sign = float(own.get("potential_sign", 1.0))
        if self.potential_sign not in (1.0, -1.0):
            raise ValueError("dark_portal.potential_sign must be +1 or -1")
        self.coupling_split = parse_coupling_split(own.get("coupling_split", "fixed"))
        base = merge_config_with_defaults(sc)
        # the Earth crossing's responses reused from a point's first run on (dp_spatial): every run's result is then
        # independent of the order of the runs, whichever process computes it (prefetch)
        base["execution"].update(n_cores=1, verbose=0, use_gpu=False, earth_transfer="always")
        self.base = base
        self.sky = bool(sky)
        # detectors away from the South Pole (the future detectors, 2026-10-15): their latitudes (deg); a site within
        # 0.1 deg of a pole takes the per-declination bands (IceCube's -89.99 in NuFlavorSingleSource)
        self.site_lats = tuple(sorted({float(l) for l in sites if abs(float(l)) < SITE_POLE_DEG}))
        self.flavor_band = None if flavor_band_eV is None else (float(flavor_band_eV[0]), float(flavor_band_eV[1]))
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
        elif pure_flavor_basis:
            # the solver runs per pure-flavor source although the composition is fixed (flux_basis 'pure_flavors',
            # 2026-10-14): the same runs as a fitted-composition analysis's, which then shares them on disk
            self.basis = [i for i, k in enumerate(SOURCE_KEYS) if float(base["source"][k]) != 0.0]
        # The Galactic component: the extragalactic flux must be in physical units to be added to it
        self.galactic = bool(base["galactic"]["enabled"])
        if self.galactic and base["sky"]["extragalactic_norm"] is None:
            raise ValueError("galactic.enabled needs sky.extragalactic_norm (and sky.extragalactic_norm_E_GeV): the "
                             "extragalactic flux's normalization against the Galactic one, e.g. "
                             "dp_galactic.icecube_2026_anchor()")
        if self.sky:
            if self.ratio_free:
                raise ValueError("sky mode (the event likelihoods) does not float nubar_to_nu_ratio")
            if base["sky"]["extragalactic_norm"] is None:
                raise ValueError("sky mode needs sky.extragalactic_norm: the events need the flux in physical units")
            base["sky"]["output"] = "bands"
        self.p_nodes = None
        if p_nodes is not None:
            self.p_nodes = np.array(sorted(float(p) for p in p_nodes))
            if len(self.p_nodes) < 2 or np.any(np.diff(self.p_nodes) <= 0):
                raise ValueError(f"p_nodes must hold at least two distinct values, got {p_nodes}")
        # the broken power law (module docstring)
        self.E_break = None if own.get("E_break_GeV") is None else float(own["E_break_GeV"])
        self.p_low_default = float(own.get("p_index_low", base["source"]["p_index"]))
        self.p1_nodes = None
        if p1_nodes is not None:
            self.p1_nodes = np.array(sorted(float(p) for p in p1_nodes))
            if len(self.p1_nodes) < 2 or np.any(np.diff(self.p1_nodes) <= 0):
                raise ValueError(f"p1_nodes must hold at least two distinct values, got {p1_nodes}")
        self._scales = {}
        self._scale_params = None
        if self.E_break is None:
            if "p_index_low" in free_names or self.p1_nodes is not None:
                raise ValueError("p_index_low is the lower index of a broken power law: set source.E_break_GeV")
        else:
            if not self.E_break > 0.0:
                raise ValueError(f"source.E_break_GeV must be positive, got {self.E_break}")
            if any(base["source"].get(k) is not None for k in WINDOW_KEYS):
                raise ValueError("with source.E_break_GeV the model sets the sources' emission windows itself")
            if base["sky"]["extragalactic_norm"] is None:
                raise ValueError("source.E_break_GeV needs sky.extragalactic_norm: the pieces join in physical units")
            fixed = [k for k in BPL_FIXED if k in free_names]
            if fixed:
                raise ValueError(f"a broken power law keeps {fixed} fixed (its scale is their closed form)")
            self.E_ref = float(base["sky"]["extragalactic_norm_E_GeV"])
        budget = int(float(os.environ.get("NULRI_MODEL_CACHE_MB", CACHE_MB_DEFAULT)) * 2 ** 20)
        self._cache: "OrderedDict[tuple, tuple]" = _BoundedCache(cache_size, budget)
        self._sky_cache: "OrderedDict[tuple, dict]" = _BoundedCache(cache_size, budget)
        # the Galactic component per propagation point WITHOUT the spectral index (2026-10-14): it does not depend on
        # it (checked bit for bit), and recomputing it at every index node took half of a plane point's time
        self._gal_cache: "OrderedDict[tuple, object]" = _BoundedCache(cache_size, budget)
        self._cache_size = int(cache_size)
        self.n_solver_calls = 0
        self.n_cache_hits = 0
        self.n_disk_hits = 0
        # a solver-run cache on disk shared by every process (2026-10-14): NULRI_FLUX_CACHE names its directory
        self.disk_cache = os.environ.get(DISK_CACHE_ENV) or None
        # runs of one evaluation computed in parallel (prefetch, 2026-10-18): a callable mapping a list of
        # (kind, cfg, with_galactic) tasks to their runs, in order -- e.g. a pool of worker processes
        # (scans.global_fit); None computes them one by one, in this process
        self.executor = None
        # the mixing parameters varied through templates (templates.py, 2026-10-18): never passed to the solver but in
        # the template runs; their ratios to the theta0 run, per point and piece, kept in _tmpl_memo
        self.templates = {t.name: t for t in templates}
        self._tmpl_memo: "OrderedDict[tuple, dict]" = _BoundedCache(64, budget)

    # ------------------------------------------------------------------
    def _overrides(self, values: Mapping, templated: bool = False) -> Dict[Tuple[str, str], object]:
        """The solver configuration entries the values set; a parameter with a template only for a template's own run
        (templated)."""
        out = {}
        products = [n for n in PRODUCTS if n in values]
        if len(products) > 1 or (products and "g_D" in values):
            raise ValueError(f"Set g_D through at most one of g_D, {PRODUCTS}; got {sorted(values)}")
        for name, v in values.items():
            if name in LINEAR_KEYS or name in PRODUCTS or name in ANALYSIS_SPECTRAL:
                continue
            if name in self.templates and not templated:
                continue
            section, key = solver_key(name)
            if name in ALIASES:
                v = sin2_to_deg(v)
            out[(section, key)] = v
        if products:
            dp = self.base["dark_portal"]

            def get(k):
                return float(out.get(("dark_portal", k), dp[k]))
            name = products[0]
            if name in ("g_D_u2", "g_D_eps_u2"):
                u2 = max(abs(get("u_e")), abs(get("u_mu")), abs(get("u_tau"))) ** 2
            elif name in ("g_D_trace", "g_D_eps_trace"):
                u2 = abs(get("u_e")) ** 2 + abs(get("u_mu")) ** 2 + abs(get("u_tau")) ** 2
            else:
                u2 = 1.0
            if "eps" in name and self.coupling_split != "fixed":
                P = float(values[name])
                x = np.sqrt(P) if self.coupling_split == "balanced" else self.coupling_split
                if u2 == 0.0 or P < 0.0 or (x > 0.0 and P > x):
                    raise ValueError(f"{name} = {P:g} with coupling_split {self.coupling_split!r}: needs a nonzero "
                                     "mixing, P >= 0, and P <= x (epsilon' <= 1)")
                out[("dark_portal", "g_D")] = x / u2
                out[("dark_portal", "epsilon_prime")] = P / x if x > 0.0 else 0.0
            else:
                scale = u2 * (get("epsilon_prime") if "eps" in name else 1.0)
                if scale == 0.0:
                    raise ValueError(f"{name} needs a nonzero mixing"
                                     f"{'' if 'eps' not in name else ' and epsilon_prime'}")
                out[("dark_portal", "g_D")] = float(values[name]) / abs(scale)
        if self.potential_sign < 0.0:
            # the other sign of the potential: eps' < 0 (g_D stays positive); the direct gauge models ignore it
            dp = self.base["dark_portal"]
            out[("dark_portal", "epsilon_prime")] = -abs(float(out.get(("dark_portal", "epsilon_prime"),
                                                                       dp["epsilon_prime"])))
        return out

    @staticmethod
    def _without_index(key: tuple) -> tuple:
        # the Galactic component depends on neither the spectral index nor the extragalactic sources' window
        return tuple(item for item in key if not (item[0] == "source" and (item[1] == "p_index"
                                                                           or item[1] in WINDOW_KEYS)))

    def _gal_get(self, key: tuple):
        g = self._gal_cache.get(self._without_index(key))
        if g is not None:
            self._gal_cache.move_to_end(self._without_index(key))
        return g

    def _gal_put(self, key: tuple, g) -> None:
        k = self._without_index(key)
        if k not in self._gal_cache:
            self._gal_cache[k] = g
            if len(self._gal_cache) > self._cache_size:
                self._gal_cache.popitem(last=False)

    def default(self, name: str) -> float:
        """A parameter's value when a fit does not set it: the base solver config's, or the analysis' default."""
        if name in ANALYSIS_DEFAULTS:
            return float(ANALYSIS_DEFAULTS[name])
        if name == "p_index_low":
            return self.p_low_default
        section, key = solver_key(name)
        v = self.base[section][key]
        if name in ALIASES:                    # sin^2 of the angle the solver keeps in degrees
            return float(np.sin(np.radians(float(v))) ** 2)
        if name == "norm_gal" and v is None:
            import dp_galactic
            v = dp_galactic.NORM_DEFAULT[str(self.base["galactic"]["template"]).strip().lower()]
        return float(v)

    def _disk_path(self, cfg: Mapping) -> Optional[str]:
        """The cache file of one solver run: a digest of everything that decides it -- the solver configuration
        (without execution and output), the flux mode (basis sources, nu/nubar responses, Galactic part) and the
        solver's source code (dp_core._source_digest), so results of other code are never reused."""
        if not self.disk_cache:
            return None
        solver = {k: v for k, v in cfg.items() if k not in ("execution", "output")}
        ident = dict(solver=solver, basis=self.basis, ratio_free=self.ratio_free, galactic=self.galactic,
                     code=dp_core._source_digest(), version=DISK_CACHE_VERSION)
        if self.sky:
            ident["sky"] = "propagate_sky_basis, single-precision bands"
            if self.site_lats:
                ident["sites"] = list(self.site_lats)
        h = hashlib.sha256(json.dumps(ident, sort_keys=True, default=_json_default).encode()).hexdigest()
        return os.path.join(self.disk_cache, h[:2], h + ".npz")

    @staticmethod
    def _disk_read(path: str):
        try:
            with np.load(path, allow_pickle=False) as z:
                gal = dict(G=z["G"], K=float(z["K"])) if "G" in z.files else None
                return z["E"], z["F_a"], (z["F_b"] if "F_b" in z.files else None), gal
        except FileNotFoundError:
            return None
        except Exception:                        # a damaged file (e.g. a writer killed mid-way): recompute
            return None

    @staticmethod
    def _disk_write(path: str, out) -> None:
        E, F_a, F_b, gal = out
        arrays = dict(E=E, F_a=F_a)
        if F_b is not None:
            arrays["F_b"] = F_b
        if gal is not None:
            arrays.update(G=gal["G"], K=np.float64(gal["K"]))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)                # atomic: readers see a whole file or none
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _run_cfg(self, ov: Mapping) -> dict:
        """The full solver configuration of one run: the base with the overrides, and with the pure-flavor basis any
        valid composition (the basis fluxes ignore it; it passes the solver's checks)."""
        cfg = copy.deepcopy(self.base)
        for (section, k), v in ov.items():
            cfg[section][k] = v
        if self.basis is not None:
            cfg["source"].update(f_e_S=1.0, f_mu_S=0.0, f_tau_S=0.0)
        return cfg

    # ------------------------------------------------------------------ runs in parallel (2026-10-18)
    def _pieces(self, values: Mapping) -> list:
        """(values, nodes) of each piece: the one power law, or the broken power law's upper and lower pieces."""
        if self.E_break is None:
            return [(dict(values), self.p_nodes)]
        _, p1 = self._index_values(values)
        return [(dict(values, injection_E_min_GeV=self.E_break), self.p_nodes),
                (dict(values, injection_E_max_GeV=self.E_break, p_index=p1), self.p1_nodes)]

    def _evaluation_values(self, values: Mapping) -> list:
        """(values, templated) of every solver run one evaluation at `values` makes: each index node of each piece,
        then each piece's template runs, in the order fluxes() and band_fluxes() visit them."""
        out = []
        for v, nodes in self._pieces(values):
            out.extend((x, False) for x in ([v] if nodes is None else [dict(v, p_index=float(pk)) for pk in nodes]))
            out.extend(self._template_runs(v, nodes))
        return out

    # ------------------------------------------------------------------ mixing templates (templates.py, 2026-10-18)
    def _template_central(self, values: Mapping, nodes) -> dict:
        """A piece's run at its central index node (at its own index without nodes), every template's parameter at
        theta0."""
        c = {k: v for k, v in values.items() if k not in self.templates}
        if nodes is not None:
            c["p_index"] = float(nodes[len(nodes) // 2])
        return c

    def _template_runs(self, values: Mapping, nodes) -> list:
        """(values, True) of a piece's template runs: one parameter at one of its points, at the central node."""
        c = self._template_central(values, nodes)
        return [(dict(c, **{t.name: float(pt)}), True) for t in self.templates.values() for pt in t.points]

    def _template_stack(self, values: Mapping, nodes, kind: str) -> dict:
        """For a piece: the runs' relative differences to the theta0 run at the central node, stacked over every
        template point (in _template_runs' order): {'bands_nu', 'bands_nb'} (sky) or {'F_a', 'F_b'} (flux), each
        (n_points, ...), F_r / F_0 - 1 (0 where F_0 = 0); and the Galactic differences G_r - G_0: {'gal_nu',
        'gal_nb', 'site_gal'} (sky) or {'G'} (flux), or None. Kept per point and piece."""
        c = self._template_central(values, nodes)
        key = (kind, tuple(sorted((s, k, v if isinstance(v, str) else float(v))
                                  for (s, k), v in self._overrides(c).items())))
        hit = self._tmpl_memo.get(key)
        if hit is not None:
            self._tmpl_memo.move_to_end(key)
            return hit
        if kind == "sky":
            r0 = self._sky_exact(c)
            rs = [self._sky_exact(v, True) for v, _ in self._template_runs(values, nodes)]
            arrays = {"bands_nu": (r0["bands_nu"], [r["bands_nu"] for r in rs]),
                      "bands_nb": (r0["bands_nb"], [r["bands_nb"] for r in rs])}
        else:
            r0 = self._fluxes_exact(c)
            rs = [self._fluxes_exact(v, True) for v, _ in self._template_runs(values, nodes)]
            arrays = {"F_a": (r0[1], [r[1] for r in rs])}
            if r0[2] is not None:
                arrays["F_b"] = (r0[2], [r[2] for r in rs])
        out = {}
        for name, (a0, ar) in arrays.items():
            pos = a0 > 0.0
            safe = np.where(pos, a0, 1.0)
            out[name] = np.stack([np.where(pos, a / safe - 1.0, 0.0) for a in ar])
        if kind == "sky" and r0["gal_nu"] is not None:
            out["gal_nu"] = np.stack([r["gal_nu"] - r0["gal_nu"] for r in rs])
            out["gal_nb"] = np.stack([r["gal_nb"] - r0["gal_nb"] for r in rs])
            out["site_gal"] = {lat: (np.stack([r["site_gal"][lat][0] - r0["site_gal"][lat][0] for r in rs]),
                                     np.stack([r["site_gal"][lat][1] - r0["site_gal"][lat][1] for r in rs]))
                               for lat in (r0.get("site_gal") or {})}
        elif kind == "flux" and r0[3] is not None:
            out["G"] = np.stack([r[3]["G"] - r0[3]["G"] for r in rs])
        self._tmpl_memo[key] = out
        return out

    def _template_weights(self, values: Mapping):
        """(w, {name: dw}) over the stacked template points at the values: each point's interpolation weight, and
        its derivative in its own template's parameter (zero at the other templates' points)."""
        w, dws, n = [], {}, 0
        total = sum(len(t.points) for t in self.templates.values())
        for t in self.templates.values():
            a, da = t.weights(float(values.get(t.name, t.theta0)))
            w.extend(a[1:])
            d = np.zeros(total)
            d[n:n + len(t.points)] = da[1:]
            dws[t.name] = d
            n += len(t.points)
        return np.array(w), dws

    def _template_factor(self, values: Mapping, nodes, kind: str, names: Sequence[str] = ()):
        """For a piece: the factors 1 + sum_r w_r (F_r / F_0 - 1) of its arrays at the values, and their derivatives
        in the templated `names`: ({array: factor}, {name: {array: d factor}})."""
        st = self._template_stack(values, nodes, kind)
        w, dws = self._template_weights(values)
        keys = ("bands_nu", "bands_nb") if kind == "sky" else tuple(k for k in ("F_a", "F_b") if k in st)
        fac = {k: 1.0 + np.tensordot(w, st[k], axes=(0, 0)) for k in keys}
        der = {n: {k: np.tensordot(dws[n], st[k], axes=(0, 0)) for k in keys} for n in names if n in self.templates}
        return fac, der

    def _template_galactic(self, values: Mapping, run: dict, names: Sequence[str] = ()):
        """The upper piece's Galactic bands at the values: run's (at theta0) plus sum_r w_r (G_r - G_0); with their
        derivatives in the templated `names`. Returns (gal_nu, gal_nb, site_gal, {name: (d_nu, d_nb, d_site)})."""
        v, nodes = self._pieces(values)[0]
        st = self._template_stack(v, nodes, "sky")
        w, dws = self._template_weights(values)
        g_nu = run["gal_nu"] + np.tensordot(w, st["gal_nu"], axes=(0, 0))
        g_nb = run["gal_nb"] + np.tensordot(w, st["gal_nb"], axes=(0, 0))
        site = {lat: (run["site_gal"][lat][0] + np.tensordot(w, d[0], axes=(0, 0)),
                      run["site_gal"][lat][1] + np.tensordot(w, d[1], axes=(0, 0))) for lat, d in st["site_gal"].items()}
        der = {}
        for n in names:
            if n in self.templates:
                d = dws[n]
                der[n] = (np.tensordot(d, st["gal_nu"], axes=(0, 0)), np.tensordot(d, st["gal_nb"], axes=(0, 0)),
                          {lat: (np.tensordot(d, x[0], axes=(0, 0)), np.tensordot(d, x[1], axes=(0, 0)))
                           for lat, x in st["site_gal"].items()})
        return g_nu, g_nb, site, der

    def gradient_names(self) -> tuple:
        """The names whose derivatives extragalactic_band_jacobian gives in closed form: GRADIENT_NAMES and the
        templated mixing parameters."""
        return tuple(GRADIENT_NAMES) + tuple(self.templates)

    def prefetch(self, values_list) -> int:
        """Computes the solver runs of these values that are neither cached nor on disk in parallel, through
        self.executor, and keeps them as _fluxes_exact or _sky_exact would: the calls that follow find them cached.
        The same arithmetic in another process: the runs are bit for bit those of the calls one by one. The Galactic
        part, which depends on neither the index nor the window, is computed once per point (by its first run that
        lacks it). Returns the number of runs computed; nothing without an executor or with fewer than two to do."""
        if self.executor is None:
            return 0
        cache = self._sky_cache if self.sky else self._cache
        todo, seen = [], set()
        for item in values_list:
            v, templated = item if isinstance(item, tuple) else (item, False)
            ov = self._overrides(v, templated)
            key = tuple(sorted((s, k, x if isinstance(x, str) else float(x)) for (s, k), x in ov.items()))
            if key in cache or key in seen:
                continue
            seen.add(key)
            cfg = self._run_cfg(ov)
            disk = self._disk_path(cfg)
            if disk is not None and os.path.exists(disk):
                continue                                  # read back by the call itself
            todo.append((key, cfg, disk))
        if len(todo) < 2:
            return 0
        firsts = set()
        tasks = []
        for key, cfg, _ in todo:
            point = self._without_index(key)
            need = self.galactic and point not in firsts and self._gal_get(key) is None
            if need:
                firsts.add(point)
            tasks.append(("sky" if self.sky else "flux", cfg, need))
        results = self.executor(tasks)
        order = [i for i, t in enumerate(tasks) if t[2]] + [i for i, t in enumerate(tasks) if not t[2]]
        for i in order:                                   # the runs carrying a point's Galactic part first
            key, _, disk = todo[i]
            if self.sky:
                self._sky_store(key, results[i], disk)
            else:
                self._fluxes_store(key, results[i], disk)
        return len(todo)

    # ------------------------------------------------------------------ the broken power law
    def _index_values(self, values: Mapping) -> Tuple[float, float]:
        return (float(values.get("p_index", self.base["source"]["p_index"])),
                float(values.get("p_index_low", self.p_low_default)))

    def _scale(self, p: float) -> float:
        """K(p), the whole power law's extragalactic scale (dark_portal_solver.extragalactic_scale, the closed form;
        the solver's runs carry the same number)."""
        k = self._scales.get(p)
        if k is None:
            if self._scale_params is None:
                cfg = copy.deepcopy(self.base)
                cfg["source"].update(f_e_S=1.0, f_mu_S=0.0, f_tau_S=0.0)     # K does not depend on the composition
                self._scale_params = dps.build_solver_params(cfg)[1]
            sky = self._scale_params["sky"]
            k = dps.extragalactic_scale(dict(self._scale_params, p_index=float(p)), sky["extragalactic_norm"],
                                        sky["extragalactic_norm_E_GeV"])
            if len(self._scales) < 4096:
                self._scales[p] = k
        return k

    def _piece_scale(self, p: float, nodes) -> float:
        """K at p for a piece interpolated on `nodes`: the spline of log K through the nodes, as the fluxes are
        (their product then reproduces the power law exactly without new physics), or K(p) itself."""
        if nodes is None:
            return self._scale(p)
        return float(spline_log_interp(nodes, p, [np.float64(self._scale(float(q))) for q in nodes]))

    def _join(self, p2: float, p1: float) -> Tuple[float, float]:
        """The pieces' factors (upper, lower) into physical units: K(p2) and (E_b / E_ref)^(p1 - p2) K(p1), which
        join the two power laws continuously at E_b."""
        return (self._piece_scale(p2, self.p_nodes),
                (self.E_break / self.E_ref) ** (p1 - p2) * self._piece_scale(p1, self.p1_nodes))

    def fluxes(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[dict]]:
        """As _fluxes_exact; with p_nodes, interpolated in p_index between the solver runs at the nodes (cubic
        spline in log flux, and in log K for the extragalactic scale). With a broken power law (module docstring):
        the two pieces joined in physical units (gal K = 1)."""
        if self.executor is not None:
            self.prefetch(self._evaluation_values(values))
        if self.E_break is None:
            return self._fluxes_index(values, self.p_nodes)
        p2, p1 = self._index_values(values)
        up = self._fluxes_index(dict(values, injection_E_min_GeV=self.E_break), self.p_nodes)
        lo = self._fluxes_index(dict(values, injection_E_max_GeV=self.E_break, p_index=p1), self.p1_nodes)
        k2, k1 = self._join(p2, p1)
        F_a = k2 * up[1] + k1 * lo[1]
        F_b = None if up[2] is None else k2 * up[2] + k1 * lo[2]
        gal = None if up[3] is None else dict(G=up[3]["G"], K=1.0)
        return up[0], F_a, F_b, gal

    def _fluxes_index(self, values: Mapping, nodes) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray],
                                                               Optional[dict]]:
        """_fluxes_exact at values' p_index, or interpolated in it between the runs at `nodes`."""
        if nodes is None:
            out = self._fluxes_exact(values)
        else:
            p = float(values.get("p_index", self.base["source"]["p_index"]))
            if not nodes[0] - 1e-12 <= p <= nodes[-1] + 1e-12:
                raise ValueError(f"p_index = {p} lies outside the interpolation nodes {nodes.tolist()}")
            runs = [self._fluxes_exact(dict(values, p_index=float(pk))) for pk in nodes]
            F_a = spline_log_interp(nodes, p, [r[1] for r in runs])
            F_b = None if runs[0][2] is None else spline_log_interp(nodes, p, [r[2] for r in runs])
            gal = None
            if runs[0][3] is not None:
                K = float(spline_log_interp(nodes, p, [np.float64(r[3]["K"]) for r in runs]))
                gal = dict(G=runs[0][3]["G"], K=K)
            out = (runs[0][0], F_a, F_b, gal)
        if not self.templates:
            return out
        E, F_a, F_b, gal = out
        fac, _ = self._template_factor(values, nodes, "flux")
        st = self._template_stack(values, nodes, "flux")
        F_a = F_a * fac["F_a"]
        F_b = None if F_b is None else F_b * fac["F_b"]
        if gal is not None:
            w, _ = self._template_weights(values)
            gal = dict(G=gal["G"] + np.tensordot(w, st["G"], axes=(0, 0)), K=gal["K"])
        return E, F_a, F_b, gal

    def _fluxes_exact(self, values: Mapping, templated: bool = False) -> Tuple[np.ndarray, np.ndarray,
                                                                                Optional[np.ndarray], Optional[dict]]:
        """For the propagation parameters in `values`: (E_bins, F, None, gal) with F the flux at the
        detector for the configured sources, or, when the nubar/nu ratio floats,
        (E_bins, F_from_nu, F_from_nb, gal), the responses to unit neutrino and antineutrino
        sources. With the pure-flavor basis the fluxes have a leading source axis. gal is None, or with the
        Galactic component dict(G=(2, N_E, 3) its sky-averaged flux for pure nu_e and pure nu_mu sources at the
        base norm_gal, K=the factor putting the extragalactic flux in physical units)."""
        ov = self._overrides(values, templated)
        key = tuple(sorted((s, k, v if isinstance(v, str) else float(v)) for (s, k), v in ov.items()))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.n_cache_hits += 1
            return hit
        if self.sky:
            run = self._sky_exact(values)
            w = run["omega"] / run["omega"].sum()
            F_a = np.tensordot(w, run["bands_nu"] + run["bands_nb"], axes=(0, 1))       # (n_src, N_E, 3)
            if self.basis is None:
                F_a = F_a[0]
            gal = None
            if run["gal_nu"] is not None:
                gal = dict(G=np.tensordot(run["share"], run["gal_nu"] + run["gal_nb"], axes=(0, 1)), K=run["K"])
            out = (run["E"], F_a, None, gal)
            self._cache[key] = out
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return out
        cfg = self._run_cfg(ov)
        disk = self._disk_path(cfg)
        if disk is not None:
            out = self._disk_read(disk)
            if out is not None:
                self.n_disk_hits += 1
                if out[3] is not None:
                    self._gal_put(key, out[3]["G"])
                self._cache[key] = out
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                return out
        G = self._gal_get(key) if self.galactic else None
        out = self._fluxes_compute(cfg, with_galactic=G is None)
        return self._fluxes_store(key, out, disk)

    def _fluxes_compute(self, cfg: Mapping, with_galactic: bool = True):
        """The solver run of _fluxes_exact for the full solver configuration cfg (_run_cfg): (E, F_a, F_b, gal), gal's
        G the Galactic flux only with_galactic, else None (another run of the same point carries it; _fluxes_store).
        No caching: a worker process computes these for prefetch."""
        E, params = dps.build_solver_params(cfg)
        params["show_progress"] = False
        sources = None if self.basis is None else [np.eye(3)[i] for i in self.basis]
        if self.ratio_free:
            F_a, F_b = dps.propagate_nu_nubar(E, params, sources=sources)
        else:
            F_a, F_b = dps.propagate_weighted(E, params, sources=sources), None
        gal = None
        if self.galactic:
            gal = dict(G=dps.galactic_allsky(E, params) if with_galactic else None,
                       K=dps.extragalactic_scale(params, params["sky"]["extragalactic_norm"],
                                                 params["sky"]["extragalactic_norm_E_GeV"]))
        return E, F_a, F_b, gal

    def _fluxes_store(self, key: tuple, out, disk: Optional[str]):
        """Completes a computed run with its point's Galactic flux (computed by it, or cached), counts it, writes it
        to the disk cache and keeps it."""
        E, F_a, F_b, gal = out
        if gal is not None:
            if gal["G"] is not None:
                self._gal_put(key, gal["G"])
            else:
                G = self._gal_get(key)
                if G is None:
                    raise RuntimeError("a run without its Galactic flux, and none cached for its point")
                gal = dict(G=G, K=gal["K"])
            out = (E, F_a, F_b, gal)
        self.n_solver_calls += 1
        if disk is not None:
            self._disk_write(disk, out)
        self._cache[key] = out
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    # ------------------------------------------------------------------ sky mode
    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """(E_bins in eV, declination band edges in rad) of the solver runs, without running it."""
        import dp_sky
        cfg = copy.deepcopy(self.base)
        if self.basis is not None:
            cfg["source"].update(f_e_S=1.0, f_mu_S=0.0, f_tau_S=0.0)
        E, params = dps.build_solver_params(cfg)
        edges = params["sky"]["dec_band_edges"]
        return np.asarray(E, dtype=np.float64), (dp_sky.default_dec_band_edges() if edges is None
                                                 else np.asarray(edges, dtype=np.float64))

    def _sky_exact(self, values: Mapping, templated: bool = False) -> dict:
        """One propagate_sky_basis run for the propagation parameters in `values`: dict E (eV), band_edges (rad),
        omega (sr per band), bands_nu, bands_nb (n_src, N_b, N_E, 3) in solver units for the basis sources (or the
        configured composition), K (the extragalactic scale), gal_nu, gal_nb (2, N_b, N_E, 3) for pure nu_e and
        nu_mu Galactic sources at the base norm_gal and share (N_b,) the bands' pixel shares, or None; the band arrays
        rounded to single precision (as the disk cache keeps them)."""
        ov = self._overrides(values, templated)
        key = tuple(sorted((s, k, v if isinstance(v, str) else float(v)) for (s, k), v in ov.items()))
        hit = self._sky_cache.get(key)
        if hit is not None:
            self._sky_cache.move_to_end(key)
            return hit
        cfg = self._run_cfg(ov)
        disk = self._disk_path(cfg)
        run = self._sky_read(disk) if disk is not None else None
        if run is not None:
            self.n_disk_hits += 1
            if run["gal_nu"] is not None:
                self._gal_put(key, dict(gal_nu=run["gal_nu"], gal_nb=run["gal_nb"], share=run["share"],
                                        site_gal=run.get("site_gal", {})))
            self._sky_cache[key] = run
            if len(self._sky_cache) > self._cache_size:
                self._sky_cache.popitem(last=False)
            return run
        g_hit = self._gal_get(key) if self.galactic else None
        return self._sky_store(key, self._sky_compute(cfg, with_galactic=g_hit is None), disk)

    def _sky_compute(self, cfg: Mapping, with_galactic: bool = True) -> dict:
        """The propagate_sky_basis run of _sky_exact for the full solver configuration cfg (_run_cfg), with the
        point's Galactic bands only with_galactic (else another run of the point carries them; _sky_store). No
        caching: a worker process computes these for prefetch."""
        E, params = dps.build_solver_params(cfg)
        params["show_progress"] = False
        if self.basis is not None:
            sources = [np.eye(3)[i] for i in self.basis]
        else:
            sources = [np.array([params["f_e_S"], params["f_mu_S"], params["f_tau_S"]], dtype=np.float64)]
        K = dps.extragalactic_scale(params, params["sky"]["extragalactic_norm"],
                                    params["sky"]["extragalactic_norm_E_GeV"])
        raw = dict(params)
        raw["sky"] = dict(params["sky"], extragalactic_norm=None)     # solver units; K kept apart
        gal = self.galactic and with_galactic
        out = dps.propagate_sky_basis(E, raw, sources=sources, galactic=gal,
                                      site_latitudes_deg=self.site_lats if gal else ())

        def r32(a):
            return np.asarray(a, dtype=np.float32).astype(np.float64)
        run = dict(E=np.asarray(E, dtype=np.float64), band_edges=np.asarray(out["dec_band_edges"], dtype=np.float64),
                   omega=np.asarray(out["band_omega"], dtype=np.float64), bands_nu=r32(out["bands_nu"]),
                   bands_nb=r32(out["bands_nubar"]), K=float(K), gal_nu=None, gal_nb=None, share=None)
        if "galactic_bands_nu" in out:
            run.update(gal_nu=r32(out["galactic_bands_nu"]), gal_nb=r32(out["galactic_bands_nubar"]),
                       share=np.asarray(out["galactic_band_share"], dtype=np.float64),
                       site_gal={lat: (r32(out["site_galactic_bands_nu"][lat]),
                                       r32(out["site_galactic_bands_nubar"][lat])) for lat in self.site_lats})
        return run

    def _sky_store(self, key: tuple, run: dict, disk: Optional[str]) -> dict:
        """Completes a computed sky run with its point's Galactic bands (computed by it, or cached: every run and its
        disk file carries them), counts it, writes it to the disk cache and keeps it."""
        if run["gal_nu"] is not None:
            self._gal_put(key, dict(gal_nu=run["gal_nu"], gal_nb=run["gal_nb"], share=run["share"],
                                    site_gal=run["site_gal"]))
        elif self.galactic:
            g_hit = self._gal_get(key)
            if g_hit is None:
                raise RuntimeError("a sky run without its Galactic bands, and none cached for its point")
            run.update(g_hit)
        self.n_solver_calls += 1
        if disk is not None:
            self._sky_write(disk, run)
        self._sky_cache[key] = run
        if len(self._sky_cache) > self._cache_size:
            self._sky_cache.popitem(last=False)
        return run

    @staticmethod
    def _sky_read(path: str):
        try:
            with np.load(path, allow_pickle=False) as z:
                run = dict(E=z["E"], band_edges=z["band_edges"], omega=z["omega"],
                           bands_nu=z["bands_nu"].astype(np.float64), bands_nb=z["bands_nb"].astype(np.float64),
                           K=float(z["K"]), gal_nu=None, gal_nb=None, share=None)
                if "gal_nu" in z.files:
                    run.update(gal_nu=z["gal_nu"].astype(np.float64), gal_nb=z["gal_nb"].astype(np.float64),
                               share=z["share"])
                    lats = z["site_lats"].tolist() if "site_lats" in z.files else []
                    run["site_gal"] = {float(l): (z[f"site_nu_{i}"].astype(np.float64), z[f"site_nb_{i}"].astype(np.float64))
                                       for i, l in enumerate(lats)}
                return run
        except FileNotFoundError:
            return None
        except Exception:                        # a damaged file: recompute
            return None

    @staticmethod
    def _sky_write(path: str, run: dict) -> None:
        arrays = dict(E=run["E"], band_edges=run["band_edges"], omega=run["omega"],
                      bands_nu=run["bands_nu"].astype(np.float32), bands_nb=run["bands_nb"].astype(np.float32),
                      K=np.float64(run["K"]))
        if run["gal_nu"] is not None:
            arrays.update(gal_nu=run["gal_nu"].astype(np.float32), gal_nb=run["gal_nb"].astype(np.float32),
                          share=run["share"])
            site = run.get("site_gal") or {}
            if site:
                lats = sorted(site)
                arrays["site_lats"] = np.array(lats, dtype=np.float64)
                for i, l in enumerate(lats):
                    arrays[f"site_nu_{i}"] = site[l][0].astype(np.float32)
                    arrays[f"site_nb_{i}"] = site[l][1].astype(np.float32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            with open(tmp, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def band_fluxes(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(E_bins in eV, F_nu, F_nb): the flux per declination band, energy and flavor for neutrinos and
        antineutrinos, each (N_b, N_E, 3) in GeV^-1 cm^-2 s^-1 sr^-1 -- the extragalactic flux times norm_astro with
        the source composition applied, plus the Galactic one (norm_gal, f_e_S_gal); with p_nodes interpolated in
        p_index as fluxes() is (cubic spline in log flux, and in log K)."""
        if not self.sky:
            raise ValueError("band_fluxes needs the model in sky mode (the event likelihoods)")
        if self.executor is not None:
            self.prefetch(self._evaluation_values(values))
        if self.E_break is None:
            b_nu, b_nb, K, run = self._bands_index(values, self.p_nodes)
        else:
            # the broken power law: the two pieces joined in physical units (module docstring)
            p2, p1 = self._index_values(values)
            u_nu, u_nb, _, run = self._bands_index(dict(values, injection_E_min_GeV=self.E_break), self.p_nodes)
            l_nu, l_nb, _, _ = self._bands_index(dict(values, injection_E_max_GeV=self.E_break, p_index=p1),
                                                 self.p1_nodes)
            k2, k1 = self._join(p2, p1)
            b_nu, b_nb, K = k2 * u_nu + k1 * l_nu, k2 * u_nb + k1 * l_nb, 1.0
        if self.basis is not None:
            comp = self.composition(values)[self.basis]
            b_nu, b_nb = np.tensordot(comp, b_nu, axes=(0, 0)), np.tensordot(comp, b_nb, axes=(0, 0))
        else:
            b_nu, b_nb = b_nu[0], b_nb[0]
        scale = float(values.get("norm_astro", 1.0)) * K
        F_nu, F_nb = scale * b_nu, scale * b_nb
        if run["gal_nu"] is not None:
            x = float(values.get("f_e_S_gal", self.default("f_e_S_gal")))
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"f_e_S_gal must lie in [0, 1], got {x}")
            g = float(values.get("norm_gal", self.default("norm_gal"))) / self.default("norm_gal")
            G_nu, G_nb = run["gal_nu"], run["gal_nb"]
            if self.templates:
                G_nu, G_nb, _, _ = self._template_galactic(values, run)
            F_nu = F_nu + g * (x * G_nu[0] + (1.0 - x) * G_nu[1])
            F_nb = F_nb + g * (x * G_nb[0] + (1.0 - x) * G_nb[1])
        return run["E"], F_nu, F_nb

    def _bands_index_jac(self, values: Mapping, nodes, names: Sequence[str] = ()):
        """_bands_index (the same arithmetic) with the derivatives in p_index of log(bands), element by element, and
        of log K: (bands_nu, bands_nb, K, the first run, dlog_nu, dlog_nb, dlogK, dt); the derivatives None without
        nodes (the index fixed); dt the derivatives of the bands in the templated `names`, {name: (d_nu, d_nb)} (the
        templates' factor does not depend on the index)."""
        if nodes is None:
            run = self._sky_exact(values)
            b_nu, b_nb, K = run["bands_nu"], run["bands_nb"], run["K"]
            dl_nu = dl_nb = dlK = None
        else:
            p = float(values.get("p_index", self.base["source"]["p_index"]))
            if not nodes[0] - 1e-12 <= p <= nodes[-1] + 1e-12:
                raise ValueError(f"p_index = {p} lies outside the interpolation nodes {nodes.tolist()}")
            runs = [self._sky_exact(dict(values, p_index=float(pk))) for pk in nodes]
            w, w1 = spline_weights(nodes, p), spline_weight_derivatives(nodes, p)
            lK = np.log(np.maximum(np.stack([np.float64(r["K"]) for r in runs]), 1e-300))
            K = float(np.exp(np.tensordot(w, lK, axes=(0, 0))))
            out = []
            for key in ("bands_nu", "bands_nb"):
                L = np.log(np.maximum(np.stack([r[key] for r in runs]), 1e-300))
                out.append((np.exp(np.tensordot(w, L, axes=(0, 0))), np.tensordot(w1, L, axes=(0, 0))))
            b_nu, b_nb, run = out[0][0], out[1][0], runs[0]
            dl_nu, dl_nb, dlK = out[0][1], out[1][1], float(np.dot(w1, lK))
        dt = {}
        if self.templates:
            fac, der = self._template_factor(values, nodes, "sky", names)
            dt = {n: (b_nu * d["bands_nu"], b_nb * d["bands_nb"]) for n, d in der.items()}
            b_nu, b_nb = b_nu * fac["bands_nu"], b_nb * fac["bands_nb"]
        return b_nu, b_nb, K, run, dl_nu, dl_nb, dlK, dt

    def _piece_scale_dlog(self, p: float, nodes) -> float:
        """d log(_piece_scale(p, nodes)) / dp: the spline of log K through the nodes (as _piece_scale)."""
        lK = np.log(np.maximum(np.array([self._scale(float(q)) for q in nodes]), 1e-300))
        return float(np.dot(spline_weight_derivatives(nodes, p), lK))

    def extragalactic_band_jacobian(self, values: Mapping, names: Sequence[str]):
        """(E, F_nu, F_nb, dF): the extragalactic flux per declination band, as band_fluxes(values, norm_gal=0) gives
        it (the same arithmetic, so the same values), and its derivatives in each of `names` (GRADIENT_NAMES), in
        closed form: {name: (dF_nu, dF_nb)}, zero for names it does not depend on. norm_astro and the source
        composition enter linearly; the indices through the spline in p (exact derivatives of the same spline) and,
        with a broken power law, the pieces' join (E_b / E_ref)^(p_low - p) K(p_low)."""
        names = list(names)
        key = (tuple(sorted((k, float(v)) for k, v in values.items())), tuple(names))
        memo = getattr(self, "_eg_jac", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        p2, p1 = self._index_values(values)
        db = {}                                            # d(bands per source)/d(index): (nu, nb)
        if self.E_break is None:
            b_nu, b_nb, K, run, dl_nu, dl_nb, dlK, dt = self._bands_index_jac(values, self.p_nodes, names)
            if "p_index" in names and dl_nu is not None:
                db["p_index"] = (b_nu * (dl_nu + dlK), b_nb * (dl_nb + dlK))
            db.update(dt)
        else:
            up = self._bands_index_jac(dict(values, injection_E_min_GeV=self.E_break), self.p_nodes, names)
            lo = self._bands_index_jac(dict(values, injection_E_max_GeV=self.E_break, p_index=p1), self.p1_nodes,
                                       names)
            k2, k1 = self._join(p2, p1)
            b_nu, b_nb, K = k2 * up[0] + k1 * lo[0], k2 * up[1] + k1 * lo[1], 1.0
            run = up[3]
            for n in up[7]:
                db[n] = (k2 * up[7][n][0] + k1 * lo[7][n][0], k2 * up[7][n][1] + k1 * lo[7][n][1])
            lnr = float(np.log(self.E_break / self.E_ref))
            if "p_index" in names and up[4] is not None:
                dk2 = self._piece_scale_dlog(p2, self.p_nodes)
                db["p_index"] = (k2 * up[0] * (up[4] + dk2) - lnr * k1 * lo[0],
                                 k2 * up[1] * (up[5] + dk2) - lnr * k1 * lo[1])
            if "p_index_low" in names and lo[4] is not None:
                dk1 = lnr + self._piece_scale_dlog(p1, self.p1_nodes)
                db["p_index_low"] = (k1 * lo[0] * (lo[4] + dk1), k1 * lo[1] * (lo[5] + dk1))

        def compose(a_nu, a_nb, comp):
            if self.basis is None:
                return a_nu[0], a_nb[0]
            return np.tensordot(comp, a_nu, axes=(0, 0)), np.tensordot(comp, a_nb, axes=(0, 0))
        comp = self.composition(values)[self.basis] if self.basis is not None else None
        c_nu, c_nb = compose(b_nu, b_nb, comp)
        norm = float(values.get("norm_astro", 1.0))
        scale = norm * K
        F_nu, F_nb = scale * c_nu, scale * c_nb
        dF = {}
        for n in names:
            if n == "norm_astro":
                dF[n] = (K * c_nu, K * c_nb)
            elif n in db:
                dF[n] = tuple(scale * a for a in compose(db[n][0], db[n][1], comp))
            elif n in SOURCE_KEYS and self.basis is not None and self.remainder is not None and n != self.remainder:
                d = np.zeros(3)
                d[SOURCE_KEYS.index(n)] = 1.0
                d[SOURCE_KEYS.index(self.remainder)] = -1.0
                dF[n] = tuple(scale * a for a in compose(b_nu, b_nb, d[self.basis]))
            else:
                dF[n] = (np.zeros_like(F_nu), np.zeros_like(F_nb))
        out = (run["E"], F_nu, F_nb, dF)
        self._eg_jac = (key, out)
        return out

    def site_band_flux_jacobian(self, values: Mapping, lat: float, names: Sequence[str]):
        """(E, F_nu, F_nb, dF) at a site of latitude lat (deg): site_band_fluxes (the same arithmetic, so the same
        values) and the derivatives in each of `names` (GRADIENT_NAMES), {name: (dF_nu, dF_nb)}, per local cos-zenith
        band (increasing cos Z); norm_gal and f_e_S_gal enter the Galactic part linearly."""
        names = list(names)
        E, e_nu, e_nb, de = self.extragalactic_band_jacobian(values, names)
        F_nu, F_nb = e_nu[::-1], e_nb[::-1]                       # declination order -> cos-zenith order (South Pole)
        dF = {n: (de[n][0][::-1], de[n][1][::-1]) for n in names}
        if not self.galactic:
            return E, F_nu.copy(), F_nb.copy(), dF
        run = self._galactic_run(values)
        x = float(values.get("f_e_S_gal", self.default("f_e_S_gal")))
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"f_e_S_gal must lie in [0, 1], got {x}")
        ng0 = self.default("norm_gal")
        g = float(values.get("norm_gal", ng0)) / ng0
        a_nu, a_nb, site_gal, tder = run["gal_nu"], run["gal_nb"], run.get("site_gal", {}), {}
        if self.templates:
            a_nu, a_nb, site_gal, tder = self._template_galactic(values, run, names)
        lat = float(lat)
        if abs(lat) >= SITE_POLE_DEG:
            if lat > 0:
                raise ValueError("a site at the North Pole needs its own bands")
            G_nu, G_nb = a_nu[:, ::-1], a_nb[:, ::-1]
            dG = {n: (d[0][:, ::-1], d[1][:, ::-1]) for n, d in tder.items()}
        else:
            if lat not in site_gal:
                raise ValueError(f"the model was not built for a site at latitude {lat} (sites={self.site_lats})")
            G_nu, G_nb = site_gal[lat]
            dG = {n: d[2][lat] for n, d in tder.items()}
        gal_nu, gal_nb = x * G_nu[0] + (1.0 - x) * G_nu[1], x * G_nb[0] + (1.0 - x) * G_nb[1]
        if "norm_gal" in names:
            dF["norm_gal"] = (gal_nu / ng0, gal_nb / ng0)
        if "f_e_S_gal" in names:
            dF["f_e_S_gal"] = (g * (G_nu[0] - G_nu[1]), g * (G_nb[0] - G_nb[1]))
        for n, (d_nu, d_nb) in dG.items():
            dF[n] = (dF[n][0] + g * (x * d_nu[0] + (1.0 - x) * d_nu[1]),
                     dF[n][1] + g * (x * d_nb[0] + (1.0 - x) * d_nb[1]))
        return E, F_nu + g * gal_nu, F_nb + g * gal_nb, dF

    def _bands_index(self, values: Mapping, nodes):
        """(bands_nu, bands_nb, K, the first run) at values' p_index: one sky run, or interpolated in p_index between
        the runs at `nodes` (cubic spline in log flux, and in log K); times the mixing templates' factor."""
        if nodes is None:
            run = self._sky_exact(values)
            b_nu, b_nb, K = run["bands_nu"], run["bands_nb"], run["K"]
        else:
            p = float(values.get("p_index", self.base["source"]["p_index"]))
            if not nodes[0] - 1e-12 <= p <= nodes[-1] + 1e-12:
                raise ValueError(f"p_index = {p} lies outside the interpolation nodes {nodes.tolist()}")
            runs = [self._sky_exact(dict(values, p_index=float(pk))) for pk in nodes]
            K = float(spline_log_interp(nodes, p, [np.float64(r["K"]) for r in runs]))
            b_nu = spline_log_interp(nodes, p, [r["bands_nu"] for r in runs])
            b_nb = spline_log_interp(nodes, p, [r["bands_nb"] for r in runs])
            run = runs[0]
        if self.templates:
            fac, _ = self._template_factor(values, nodes, "sky")
            b_nu, b_nb = b_nu * fac["bands_nu"], b_nb * fac["bands_nb"]
        return b_nu, b_nb, K, run

    def _galactic_run(self, values: Mapping) -> dict:
        """A sky run of this point that carries its Galactic bands (they depend on neither the index nor the
        window): the first index node's, of the upper piece with a broken power law -- one the fluxes already ran."""
        v = dict(values)
        if self.p_nodes is not None:
            v["p_index"] = float(self.p_nodes[0])
        if self.E_break is not None:
            v["injection_E_min_GeV"] = self.E_break
        return self._sky_exact(v)

    def site_band_fluxes(self, values: Mapping, lat: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """As band_fluxes, per local cos-zenith band (increasing cos Z: up-going first) at a site of latitude lat (deg):
        the extragalactic flux is isotropic before the Earth and the Earth crossing depends on the zenith only, so it
        is the South Pole's bands in cos-zenith order; the Galactic flux is the site's own (the model's sites), or the
        South Pole's for a site within SITE_POLE_DEG of it."""
        # the extragalactic part is every site's (isotropic before the Earth): computed once per values for all of
        # them (2026-10-15; the future detectors ask for five sites per chi^2)
        key = tuple(sorted((k, float(v)) for k, v in values.items()))
        memo = getattr(self, "_site_eg", None)
        if memo is not None and memo[0] == key:
            E, F_nu, F_nb = memo[1]
        else:
            E, F_nu, F_nb = self.band_fluxes(dict(values, norm_gal=0.0)) if self.galactic else self.band_fluxes(values)
            self._site_eg = (key, (E, F_nu, F_nb))
        F_nu, F_nb = F_nu[::-1], F_nb[::-1]                       # declination order -> cos-zenith order (South Pole)
        if not self.galactic:
            return E, F_nu.copy(), F_nb.copy()
        run = self._galactic_run(values)
        x = float(values.get("f_e_S_gal", self.default("f_e_S_gal")))
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"f_e_S_gal must lie in [0, 1], got {x}")
        g = float(values.get("norm_gal", self.default("norm_gal"))) / self.default("norm_gal")
        gal_nu, gal_nb, site_gal = run["gal_nu"], run["gal_nb"], run.get("site_gal", {})
        if self.templates:
            gal_nu, gal_nb, site_gal, _ = self._template_galactic(values, run)
        lat = float(lat)
        if abs(lat) >= SITE_POLE_DEG:
            if lat > 0:
                raise ValueError("a site at the North Pole needs its own bands")
            G_nu, G_nb = gal_nu[:, ::-1], gal_nb[:, ::-1]
        else:
            if lat not in site_gal:
                raise ValueError(f"the model was not built for a site at latitude {lat} (sites={self.site_lats})")
            G_nu, G_nb = site_gal[lat]
        return E, F_nu + g * (x * G_nu[0] + (1.0 - x) * G_nu[1]), F_nb + g * (x * G_nb[0] + (1.0 - x) * G_nb[1])

    def flux(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray]:
        """(E_bins, flux at the detector) with the source composition and the
        neutrino/antineutrino ratio applied, as run_solver weights them; with the Galactic component, the
        extragalactic flux in physical units times norm_astro plus the Galactic one (norm_gal, x = f_e_S_gal)."""
        E, F, gal = self._extragalactic(values)
        if gal is None:
            return E, F
        return E, F + self._galactic(values, gal)

    def extragalactic(self, values: Mapping) -> Tuple[np.ndarray, np.ndarray]:
        """(E_bins in eV, the extragalactic flux at the detector, (N_E, 3) per flavor nu + nubar): in physical units
        (GeV^-1 cm^-2 s^-1 sr^-1, times norm_astro) with the Galactic component, solver units without."""
        E, F, gal = self._extragalactic(values)
        return E, F

    def _galactic(self, values: Mapping, gal: dict) -> np.ndarray:
        x = float(values.get("f_e_S_gal", self.default("f_e_S_gal")))
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"f_e_S_gal must lie in [0, 1], got {x}")
        scale = float(values.get("norm_gal", self.default("norm_gal"))) / self.default("norm_gal")
        return scale * (x * gal["G"][0] + (1.0 - x) * gal["G"][1])

    def _extragalactic(self, values: Mapping):
        E, F_a, F_b, gal = self.fluxes(values)
        if self.basis is not None:
            comp = self.composition(values)
            F_a = np.tensordot(comp[self.basis], F_a, axes=(0, 0))
            if F_b is not None:
                F_b = np.tensordot(comp[self.basis], F_b, axes=(0, 0))
        if F_b is None:
            F = F_a
        else:
            src = dict(self.base["source"])
            src["nubar_to_nu_ratio"] = float(values.get("nubar_to_nu_ratio", src["nubar_to_nu_ratio"]))
            w_nu, w_nb = dps.injection_weights(src)
            F = w_nu * F_a + w_nb * F_b
        if gal is None:
            return E, F, None
        return E, float(values.get("norm_astro", 1.0)) * gal["K"] * F, gal

    def galactic_share(self, values: Mapping) -> float:
        """The Galactic share of the energy-integrated flux (what the energy-averaged fractions weight)."""
        if not self.galactic:
            return 0.0
        E, F = self.flux(dict(values))
        _, F_eg = self.flux(dict(values, norm_gal=0.0))
        dE = compute_dE_geometric(E)
        return float(1.0 - (F_eg.sum(-1) * dE).sum() / (F.sum(-1) * dE).sum())

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
        if self.flavor_band is not None:
            m = (E >= self.flavor_band[0] * (1.0 - 1e-9)) & (E <= self.flavor_band[1] * (1.0 + 1e-9))
            E, F = E[m], F[m]
        r = compute_energy_averaged_ratios(E, F)
        return np.array([r[0], r[1], 1.0 - r[0] - r[1]])
