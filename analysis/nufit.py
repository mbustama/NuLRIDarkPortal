"""
nufit.py -- the penalty on the mixing parameters for present-day data: NuFIT 6.1's pairwise Delta chi^2
(Mauricio, 2026-10-18: "For the current runs, vary using the pairwise NuFIT 6.1 chi2, not 1D chi2 gaussian
distributions. Only use gaussian for projections."; as Table II of arXiv:2606.17132):

    chi^2 += T13_T12(sin^2 th13, sin^2 th12) + T23_DCP(sin^2 th23, delta_CP) + DMS_DMA(log10 dm21^2, dm3l^2),

NuFIT 6.1's two-dimensional projections (each profiled by NuFIT over the other parameters; with Super-Kamiokande's
and IceCube's atmospheric data) for one mass ordering. Tables: analysis/data/nufit_6.1/pairwise.npz (README there).
The projections use Gaussians instead (parameters.GaussianPrior; Table II's numbers).

Conventions. The solver's dm31_sq is |Delta m^2_3l| (dp_core.get_mass_eigenvalues): Delta m^2_31 in the normal
ordering, |Delta m^2_32| in the inverted, where NuFIT's DMA axis is negative. NuFIT's delta_CP runs over -180..180
deg, the solver's over 0..360; any value is wrapped. The angles are read as sin^2 (s12sq, s13sq, s23sq) or in degrees
(th12_deg, th13_deg, th23_deg), from a fit's values where present and the solver configuration otherwise.

Each table is measured from its own ordering's minimum. NuFIT quotes the inverted ordering's tables against the
global (normal-ordering) minimum, 5.908 above it; summed over three tables that would count three times, and within
one ordering it is a constant. It is subtracted and kept in `offset` (as GeoneutrinoDecay's NuFitSurface does).

Interpolation: a bicubic spline through each table (GeoneutrinoDecay found that a bilinear blend, whose gradient
jumps at every node, leaves a fit a grid cell away from the minimum). Each spline is shifted up by its undershoot
below zero, measured at SUB points per table interval, so the penalty is smooth and non-negative; the shift is a
constant that cancels in any Delta chi^2. delta_CP is periodic: the table is padded by PAD columns on each side
before the spline is built, so the interpolant is smooth across +-180 deg (the normal ordering's minimum sits at
-150). Arguments outside a table are clamped to its edges, where Delta chi^2 is in the hundreds or more;
configurations must keep their fit bounds inside the tables (check()).
"""
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nufit_6.1", "pairwise.npz")
SUFFIX = {"normal": "NO", "inverted": "IO"}
TABLE_NAMES = ("T13_T12", "T23_DCP", "DMS_DMA")
PERIODIC = "T23_DCP"     # its second axis, delta_CP, is periodic
PAD = 4                  # columns copied across +-180 deg before the spline is built
SUB = 4                  # points per table interval at which a spline's undershoot is measured
DMA_UNIT = 1e-3          # NuFIT's DMA axis is in units of 1e-3 eV^2
QUANTITIES = ("s12sq", "s13sq", "s23sq", "d_cp_deg", "dm21_sq", "dm31_sq")
ANGLES = {"s12sq": "th12_deg", "s13sq": "th13_deg", "s23sq": "th23_deg"}

_SURFACES: Dict[Tuple[str, str], tuple] = {}


def ordering_of(base: Mapping) -> str:
    """The solver configuration's mass ordering, as dp_core.get_mass_eigenvalues reads it."""
    return "inverted" if str(base["cosmology"]["mass_ordering"]).lower().strip() == "inverted" else "normal"


def _refine(a: np.ndarray, k: int) -> np.ndarray:
    """k points per interval of a, and a's last point."""
    t = np.arange(k) / k
    return np.concatenate([(a[:-1, None] + np.diff(a)[:, None] * t[None, :]).ravel(), a[-1:]])


def surface(name: str, ordering: str):
    """(spline, floor, offset, (x range, y range)) of one table, built once. floor <= 0 is the spline's undershoot,
    offset the ordering's minimum in NuFIT's table (0 normal, 5.908 inverted), both subtracted by delta_chi2."""
    from scipy.interpolate import RectBivariateSpline
    key = (name, ordering)
    if key not in _SURFACES:
        if ordering not in SUFFIX:
            raise ValueError(f"ordering must be 'normal' or 'inverted', got {ordering!r}")
        with np.load(TABLES) as z:
            x, y, c = (np.array(z[f"{name}_{SUFFIX[ordering]}_{k}"], dtype=np.float64) for k in ("x", "y", "chi2"))
        offset = float(c.min())
        c = c - offset
        ranges = ((float(x[0]), float(x[-1])), (float(y[0]), float(y[-1])))
        if name == PERIODIC:
            if not (y[0] == -180.0 and y[-1] == 180.0 and np.array_equal(c[:, 0], c[:, -1])):
                raise ValueError(f"{name}: expected delta_CP over -180..180 deg with equal end columns")
            y = np.concatenate([y[-1 - PAD:-1] - 360.0, y, y[1:1 + PAD] + 360.0])
            c = np.concatenate([c[:, -1 - PAD:-1], c, c[:, 1:1 + PAD]], axis=1)
        spline = RectBivariateSpline(x, y, c, kx=3, ky=3, s=0)
        floor = float(min(0.0, spline(_refine(x, SUB), _refine(y, SUB)).min()))
        _SURFACES[key] = (spline, floor, offset, ranges)
    return _SURFACES[key]


def wrap_dcp(d: float) -> float:
    """delta_CP in NuFIT's range, [-180, 180) deg."""
    return float((float(d) + 180.0) % 360.0 - 180.0)


def mixing_values(values: Mapping, base: Mapping) -> Dict[str, float]:
    """The six quantities NuFIT's tables constrain (QUANTITIES), from a fit's values where present (sin^2 or degrees
    for the angles) and the solver configuration otherwise."""
    sm = base["standard_mixing"]
    out = {}
    for s2, deg in ANGLES.items():
        if s2 in values:
            out[s2] = float(values[s2])
        else:
            out[s2] = float(np.sin(np.radians(float(values.get(deg, sm[deg])))) ** 2)
    for k in ("d_cp_deg", "dm21_sq", "dm31_sq"):
        out[k] = float(values.get(k, sm[k]))
    return out


def table_coordinates(q: Mapping[str, float], ordering: str) -> Dict[str, Tuple[float, float]]:
    """Each table's (x, y) at the quantities q (mixing_values)."""
    if q["dm21_sq"] <= 0.0 or q["dm31_sq"] <= 0.0:
        raise ValueError(f"dm21_sq and dm31_sq (= |Delta m^2_3l|) must be positive, got {q['dm21_sq']}, "
                         f"{q['dm31_sq']}")
    sign = 1.0 if ordering == "normal" else -1.0
    return {"T13_T12": (q["s13sq"], q["s12sq"]),
            "T23_DCP": (q["s23sq"], wrap_dcp(q["d_cp_deg"])),
            "DMS_DMA": (float(np.log10(q["dm21_sq"])), sign * q["dm31_sq"] / DMA_UNIT)}


def delta_chi2(name: str, ordering: str, x: float, y: float) -> float:
    """One table's Delta chi^2 above its ordering's minimum, interpolated (module docstring); arguments clamped to
    the table's range (delta_CP wrapped instead)."""
    spline, floor, _, ((x0, x1), (y0, y1)) = surface(name, ordering)
    x = min(max(float(x), x0), x1)
    y = wrap_dcp(y) if name == PERIODIC else min(max(float(y), y0), y1)
    return float(spline.ev(x, y)) - floor


@dataclass(frozen=True)
class NuFitPairwisePrior:
    """chi^2 += the sum of NuFIT 6.1's three pairwise tables for `ordering` (module docstring), on the six mixing
    quantities (QUANTITIES) at once. The ordering must be the solver's (check)."""
    ordering: str
    name: str = "nufit_6.1_pairwise"

    def __post_init__(self):
        if self.ordering not in SUFFIX:
            raise ValueError(f"nufit_6.1_pairwise: ordering must be 'normal' or 'inverted', got {self.ordering!r}")

    @property
    def quantities(self) -> Tuple[str, ...]:
        return QUANTITIES

    @property
    def offset(self) -> float:
        """NuFIT's minimum of this ordering above the global one (5.908 for inverted), left out of chi2_values."""
        return surface(TABLE_NAMES[0], self.ordering)[2]

    def terms(self, values: Mapping, base: Mapping) -> Dict[str, float]:
        coords = table_coordinates(mixing_values(values, base), self.ordering)
        return {n: delta_chi2(n, self.ordering, *coords[n]) for n in TABLE_NAMES}

    def chi2_values(self, values: Mapping, base: Mapping) -> float:
        return float(sum(self.terms(values, base).values()))

    def best_fit(self) -> Dict[str, float]:
        """The tables' minima (grid points) as solver values: sin^2 of the angles, delta_CP in [0, 360) deg,
        dm21_sq and dm31_sq (= |Delta m^2_3l|) in eV^2."""
        out = {}
        with np.load(TABLES) as z:
            for n in TABLE_NAMES:
                x, y, c = (z[f"{n}_{SUFFIX[self.ordering]}_{k}"] for k in ("x", "y", "chi2"))
                i, j = np.unravel_index(np.argmin(c), c.shape)
                out[n] = (float(x[i]), float(y[j]))
        return {"s13sq": out["T13_T12"][0], "s12sq": out["T13_T12"][1], "s23sq": out["T23_DCP"][0],
                "d_cp_deg": out["T23_DCP"][1] % 360.0, "dm21_sq": 10.0 ** out["DMS_DMA"][0],
                "dm31_sq": abs(out["DMS_DMA"][1]) * DMA_UNIT}

    def check(self, base: Mapping, bounds: Optional[Mapping[str, Tuple[float, float]]] = None) -> None:
        """Raise unless the solver's ordering is this prior's and every free mixing parameter's bounds (physical
        units; sin^2 or degrees for the angles) lie inside the tables, where the penalty is not clamped."""
        if ordering_of(base) != self.ordering:
            raise ValueError(f"nufit_6.1_pairwise is for the {self.ordering} ordering, but the solver's "
                             f"cosmology.mass_ordering is {base['cosmology']['mass_ordering']!r}")
        sign = 1.0 if self.ordering == "normal" else -1.0
        axes = {"s13sq": ("T13_T12", 0, lambda v: v), "s12sq": ("T13_T12", 1, lambda v: v),
                "s23sq": ("T23_DCP", 0, lambda v: v),
                "dm21_sq": ("DMS_DMA", 0, lambda v: np.log10(v) if v > 0 else -np.inf),
                "dm31_sq": ("DMS_DMA", 1, lambda v: sign * v / DMA_UNIT)}
        for name, (lo, hi) in (bounds or {}).items():
            q = next((s2 for s2, deg in ANGLES.items() if deg == name), name)
            if q not in axes:
                continue
            table, axis, to_table = axes[q]
            if q != name:                                  # bounds in degrees
                lo, hi = (float(np.sin(np.radians(b)) ** 2) for b in (lo, hi))
            a, b = sorted((float(to_table(lo)), float(to_table(hi))))
            t0, t1 = surface(table, self.ordering)[3][axis]
            if a < t0 - 1e-12 * max(1.0, abs(t0)) or b > t1 + 1e-12 * max(1.0, abs(t1)):
                raise ValueError(f"{name}: bounds {(lo, hi)} reach outside NuFIT 6.1's {table} table "
                                 f"({t0}..{t1} on its axis), where the penalty would be clamped flat")
