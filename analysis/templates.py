"""
templates.py -- the mixing parameters varied without a solver run at every value (Mauricio, 2026-10-18: vary the
mixing in every fit; "I don't understand. Explain simply": at each point, a few extra solver runs with one parameter
moved, and the fit combines them instead of running the solver again).

A parameter with a template is not passed to the solver. At each point of the propagation parameters the model runs
the solver at the configuration's value theta0 and at the template's points (one parameter moved at a time), at the
central index node of each piece, and builds the flux at any value x as

    F(x, p) = F(theta0, p) * exp[sum_k sum_n w_kn(x_k) log(F_kn / F_0)],

element by element (flavor, energy, band, source), F_kn / F_0 the ratio of the run at parameter k's n-th point to the
run at theta0, both at the central node; w_kn the interpolation weights through theta0 and k's points (Lagrange; for
delta_CP over a whole turn, trigonometric). The log of the flux is interpolated because absorption is exponential in
the mixing (interpolating the ratio itself left chi^2 errors up to 2 at the MeV planes' limits, 2026-10-18). At the
central node and theta0 the formula is exact; elsewhere it neglects how the parameter's relative effect changes with
the index (and the parameters' cross terms). The Galactic flux, which depends on neither the index nor the window, is
interpolated directly: G(x) = G0 + sum w_kn (G_kn - G0).

At theta0 itself the model skips the template runs altogether (the factor is exactly 1 there), so a model centred at
some value is also that value's exact model: the exact profile (exact_profile.py, 2026-10-18) builds one such model per
centre of its trust region, with a "step" template per parameter as the forward differences of its local surrogate.

Template specs (a parameter's "template" in the configuration), in the parameter's own units (sin^2 for s12sq,
s13sq, s23sq; degrees for d_cp_deg; eV^2 for dm21_sq and dm31_sq, the latter |Delta m^2_3l|; eV for m_lightest_eV):
  {"step": h}            one run at theta0 + h: linear;
  {"points": [a, ...]}   runs at these values (theta0 is always one): Lagrange through all of them;
  {"turn": n}            d_cp_deg only: n (odd) runs equally spaced over 360 deg, theta0 one of them: the
                         trigonometric polynomial of degree (n - 1) / 2 through them.
"""
from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

NAMES = ("s12sq", "s13sq", "s23sq", "d_cp_deg", "dm21_sq", "dm31_sq", "m_lightest_eV")


@dataclass(frozen=True)
class MixingTemplate:
    """One parameter's template: its runs at `points` besides theta0 (module docstring)."""
    name: str
    theta0: float
    points: Tuple[float, ...]
    turn: bool = False

    def __post_init__(self):
        if self.name not in NAMES:
            raise ValueError(f"a template is for one of {NAMES}, not {self.name!r}")
        if not self.points:
            raise ValueError(f"{self.name}: a template needs at least one point besides theta0")
        nodes = self.nodes()
        if self.turn:
            if self.name != "d_cp_deg" or len(nodes) % 2 == 0:
                raise ValueError("a turn template is for d_cp_deg, with an odd number of runs")
        elif len(set(np.round(nodes, 15))) != len(nodes):
            raise ValueError(f"{self.name}: template points must differ from each other and from theta0 = "
                             f"{self.theta0}")

    def nodes(self) -> np.ndarray:
        """theta0 first, then the points."""
        return np.array((float(self.theta0),) + tuple(float(p) for p in self.points))

    def weights(self, x: float) -> Tuple[np.ndarray, np.ndarray]:
        """(w, dw/dx) over nodes() at x: sum w = 1, w = e_0 at theta0."""
        t = self.nodes()
        n = len(t)
        x = float(x)
        if self.turn:
            m = (n - 1) // 2
            d = np.radians(x - t)
            j = np.arange(1, m + 1)[:, None]
            w = (1.0 + 2.0 * np.sum(np.cos(j * d), axis=0)) / n
            dw = (-2.0 * np.sum(j * np.sin(j * d), axis=0)) / n * (np.pi / 180.0)
            if x == self.theta0:              # exactly the theta0 run (the cosines leave 1e-17 at the other nodes)
                w = np.eye(n)[0]
            return w, dw
        w = np.ones(n)
        dw = np.zeros(n)
        for i in range(n):
            for k in range(n):
                if k != i:
                    w[i] *= (x - t[k]) / (t[i] - t[k])
            for k in range(n):
                if k == i:
                    continue
                term = 1.0 / (t[i] - t[k])
                for j in range(n):
                    if j != i and j != k:
                        term *= (x - t[j]) / (t[i] - t[j])
                dw[i] += term
        return w, dw


def from_spec(name: str, spec: Mapping, theta0: float) -> MixingTemplate:
    """A template from a parameter's "template" entry (module docstring) around the configuration's value theta0."""
    spec = dict(spec)
    kinds = [k for k in ("step", "points", "turn") if k in spec]
    if len(kinds) != 1 or set(spec) - {"step", "points", "turn"}:
        raise ValueError(f"{name}: a template is one of {{'step': h}}, {{'points': [...]}}, {{'turn': n}}; got {spec}")
    if "step" in spec:
        h = float(spec["step"])
        if h == 0.0:
            raise ValueError(f"{name}: a template step must be nonzero")
        return MixingTemplate(name, float(theta0), (float(theta0) + h,))
    if "points" in spec:
        pts = tuple(float(p) for p in spec["points"] if abs(float(p) - float(theta0)) > 1e-15 * max(1.0, abs(theta0)))
        return MixingTemplate(name, float(theta0), pts)
    n = int(spec["turn"])
    return MixingTemplate(name, float(theta0), tuple(float(theta0) + 360.0 * k / n for k in range(1, n)), turn=True)
