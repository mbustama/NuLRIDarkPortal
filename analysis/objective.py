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
from .model import FlavorModel, sum_of_masses
from .parameters import ParameterSet

WALL = 1e9
UNITARITY_SLOPE = 1e4
U_KEYS = ("u_e", "u_mu", "u_tau")
DERIVED = {"sum_masses_eV": sum_of_masses}


class Objective:
    """chi^2 of one analysis.

    Arguments:
      model: a FlavorModel.
      surface: name of the flavor-composition surface (the data term).
      priors: prior objects (parameters.GaussianPrior, HalfGaussianPrior), each on a
        parameter or on a derived quantity in DERIVED.
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
        self.data = (surfaces.surface(surface) if data_centre is None
                     else surfaces.recentred(surface, *data_centre))
        self.data_centre = None if data_centre is None else (float(data_centre[0]), float(data_centre[1]))
        self.priors = list(priors)
        self.n_calls = 0
        self.time_s = 0.0

    def _quantity(self, name: str, values: Mapping) -> float:
        if name in values:
            return float(values[name])
        if name in DERIVED:
            return DERIVED[name](values, self.model.base)
        return self.model.default(name)

    def prior_chi2(self, values: Mapping) -> float:
        return float(sum(p.chi2(self._quantity(p.name, values)) for p in self.priors))

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
            f = self.model.fractions(values)
            data = surfaces.chi2(f[0], f[1], self.data) + self.spectrum_chi2(values)
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
