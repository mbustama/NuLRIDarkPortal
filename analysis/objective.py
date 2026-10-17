"""
objective.py

chi^2 = -2 ln L(data | model) + sum of priors, as a function of named values or of a fit's
internal vector. The data term of the flavor-composition analyses is a surface of
surfaces.py evaluated at the model's energy-averaged (f_e, f_mu), plus, with a spectrum
(data.spectrum; spectra.py), the measured astrophysical spectrum against the model's
extragalactic flux at the detector (per flavor: the all-flavor flux over 3), or, with events
(data.events; events.py), the HESE and DR2 event likelihoods profiled over their nuisances.

Sterile mixing beyond unitarity (u_e^2 + u_mu^2 + u_tau^2 > 1, which the boxes of a fit
cannot exclude) is evaluated on the unit sphere and charged UNITARITY_SLOPE per unit of
excess, a continuous wall the fitters step back from. FrequentistAnalysis called the
propagation directly and returned finite chi^2 there. Other points the solver rejects, and
non-finite results, return WALL.
"""
import time
from typing import Mapping, Sequence

import numpy as np

from . import spectra, surfaces
from .model import ALIASES, FlavorModel, sum_of_masses
from .parameters import ParameterSet

WALL = 1e9
UNITARITY_SLOPE = 1e4
U_KEYS = ("u_e", "u_mu", "u_tau")
DERIVED = {"sum_masses_eV": sum_of_masses}


class Objective:
    """chi^2 of one analysis.

    Arguments:
      model: a FlavorModel.
      surface: name of the flavor-composition surface (the data term), or None with the event likelihoods
        alone.
      priors: prior objects (parameters.GaussianPrior, HalfGaussianPrior), each on a
        parameter or on a derived quantity in DERIVED; or nufit.NuFitPairwisePrior, on the
        six mixing quantities at once.
      data_centre: (f_e, f_mu) to move the surface's centre to (Asimov data,
        pseudo-experiments); default: the surface as published.
    """

    def __init__(self, model: FlavorModel, surface: str, priors: Sequence = (), data_centre=None, spectrum=None,
                 events=None):
        self.model = model
        self.events = events
        self.surface = surface
        self.spectrum = None
        if spectrum is not None:
            g = model.base["grid"]
            self.spectrum = spectra.load(spectrum, float(g["E_min_eV"]) / 1e9, float(g["E_max_eV"]) / 1e9)
        # no surface: the event likelihoods alone carry the flavor (data.surface left out)
        self.data = (None if surface is None else surfaces.surface(surface) if data_centre is None
                     else surfaces.recentred(surface, *data_centre))
        self.data_centre = None if data_centre is None else (float(data_centre[0]), float(data_centre[1]))
        self.priors = list(priors)
        self.n_calls = 0
        self.time_s = 0.0

    def _quantity(self, name: str, values: Mapping) -> float:
        if name in values:
            return float(values[name])
        if name in ALIASES and ALIASES[name] in values:     # sin^2 of an angle the fit sets in degrees
            return float(np.sin(np.radians(float(values[ALIASES[name]]))) ** 2)
        if name in DERIVED:
            return DERIVED[name](values, self.model.base)
        return self.model.default(name)

    def prior_chi2(self, values: Mapping) -> float:
        """The priors' sum: each on one quantity, or (nufit.NuFitPairwisePrior) on several at once."""
        return float(sum(p.chi2_values(values, self.model.base) if hasattr(p, "chi2_values")
                         else p.chi2(self._quantity(p.name, values)) for p in self.priors))

    def fractions(self, values: Mapping) -> np.ndarray:
        return self.model.fractions(values)

    def spectrum_segments(self, values: Mapping) -> np.ndarray:
        """The model's E^-2 normalizations on the spectrum's segments (E^2 Phi per flavor, GeV cm^-2 s^-1 sr^-1)."""
        E, F = self.model.extragalactic(values)
        return spectra.segments(E / 1e9, F.sum(axis=-1) / 3.0, self.spectrum["E_lo"], self.spectrum["E_hi"])

    def spectrum_chi2(self, values: Mapping) -> float:
        return 0.0 if self.spectrum is None else spectra.chi2(self.spectrum_segments(values), self.spectrum)

    def chi2(self, values: Mapping) -> float:
        """chi^2 at named values (every free parameter of the analysis, and any held one)."""
        t0 = time.perf_counter()
        self.n_calls += 1
        excess = 0.0
        u2 = sum(self._quantity(k, values) ** 2 for k in U_KEYS)
        if u2 > 1.0:
            shrink = (1.0 - 1e-12) / np.sqrt(u2)     # strictly inside: the solver checks > 1.0
            values = dict(values)
            for k in U_KEYS:
                values[k] = self._quantity(k, values) * shrink
            excess = UNITARITY_SLOPE * (u2 - 1.0)
        try:
            data = self.spectrum_chi2(values)
            if self.data is not None:
                f = self.model.fractions(values)
                data += surfaces.chi2(f[0], f[1], self.data)
            if self.events is not None:
                data += self.events.chi2(values)
        except ValueError:
            data = WALL
        if not np.isfinite(data):
            data = WALL
        self.time_s += time.perf_counter() - t0
        return data + excess + self.prior_chi2(values)

    def on(self, pset: ParameterSet):
        """The objective as a function of a fit's internal vector."""
        def f(u):
            return self.chi2(pset.as_values(u))
        return f

    # ------------------------------------------------------------------ the exact gradient (2026-10-17)
    def supports_gradient(self, names: Sequence[str]) -> bool:
        """Whether chi2_and_grad can give the exact gradient in `names`: the future detectors' events alone (their
        profiled chi^2 is differentiable with the nuisances at their optimum; FutureEventLikelihood.chi2_and_grad),
        every name one whose derivative the model gives in closed form (model.GRADIENT_NAMES), and every prior on
        them Gaussian. Anything else -- a surface, a spectrum, HESE or DR2, a mass or a coupling free -- keeps the
        finite differences."""
        from .future_events import FutureEventLikelihood
        from .model import GRADIENT_NAMES
        from .parameters import GaussianPrior
        if self.surface is not None or self.spectrum is not None or not isinstance(self.events, FutureEventLikelihood):
            return False
        if not names or not set(names) <= set(GRADIENT_NAMES):
            return False
        return all(isinstance(p, GaussianPrior) or not ({p.name, *getattr(p, "quantities", ())} & set(names))
                   for p in self.priors)

    def chi2_and_grad(self, values: Mapping, names: Sequence[str]):
        """(chi2(values), its gradient in `names`) where supports_gradient(names): the same value as chi2."""
        t0 = time.perf_counter()
        self.n_calls += 1
        names = list(names)
        excess = 0.0
        u2 = sum(self._quantity(k, values) ** 2 for k in U_KEYS)
        if u2 > 1.0:
            shrink = (1.0 - 1e-12) / np.sqrt(u2)
            values = dict(values)
            for k in U_KEYS:
                values[k] = self._quantity(k, values) * shrink
            excess = UNITARITY_SLOPE * (u2 - 1.0)
        try:
            data, g = self.events.chi2_and_grad(values, names)
        except ValueError:
            data, g = WALL, np.zeros(len(names))
        if not np.isfinite(data) or not np.all(np.isfinite(g)):
            data, g = WALL, np.zeros(len(names))
        g = np.array(g, dtype=np.float64)
        for p in self.priors:
            if p.name in names:                         # Gaussian (supports_gradient)
                g[names.index(p.name)] += 2.0 * (self._quantity(p.name, values) - p.centre) / p.width ** 2
        self.time_s += time.perf_counter() - t0
        return data + excess + self.prior_chi2(values), g

    def on_with_grad(self, pset: ParameterSet):
        """(chi^2, its gradient) as a function of a fit's internal vector: the chain rule through each parameter's
        map (parameters.Parameter: x = u step, or 10^(u step) in log10 space)."""
        names = pset.names

        def fg(u):
            values = pset.as_values(u)
            f, g = self.chi2_and_grad(values, names)
            dxdu = np.array([p.step * (np.log(10.0) * values[p.name] if p.space == "log10" else 1.0)
                             for p in pset.free])
            return f, g * dxdu
        return fg
