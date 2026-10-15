"""
future_events.py -- the future detectors' event likelihood for the projections (docs/dev/FUTURE_DETECTORS_PLAN.md;
Mauricio, 2026-10-14/15): NuFlavorSingleSource's eight detectors (IceCubeRates.future: its models vendored unchanged,
counted per channel, reconstructed energy and local cos-zenith band by IceCubeRates.future.diffuse), against the
model's flux per local zenith at each detector's site (FlavorModel.site_band_fluxes), with NuFlavorSingleSource's
main-analysis uncertainties profiled inside the term:

  mu_k = (1 + eff_k) s_k + escale_k s_k^escale + misid_k s_k^misid [+ xs s_k^xs]
         + n_k B_k + sum_g theta_g GB_k,g + escale_k BX_k^escale + misid_k BX_k^misid [+ xs GB_k^xs]

per block k = (detector part, channel): s the astrophysical events (the model's flux, the signal here), B the
atmospheric background (NuFlavorSingleSource's MCEq table, with its atmospheric-muon multipliers), n_k its
normalization (10% prior), theta_g the Barr parameters, the cosmic-ray index and the prompt normalization (shared, with
NuFlavorSingleSource's priors), eff, escale, misid per block (DET_PRIOR; no PSF: no rings), the cross sections only
with spec cross_sections (this project decided against them, 2026-10-13; NuFlavorSingleSource has them at 5%).
-2 ln L = 2 sum [mu - n + n ln(n / mu)] (continued smoothly below a floor, as the HESE and DR2 terms) + the
priors; the data of a projection: the Asimov expectation at the
configuration's asimov values (standard physics) with the nuisances at their centres.

Detector sets as NuFlavorSingleSource's run.py (config_parts): 'All8_base' (its main case: IceCube, the six water
detectors and IceCube-Gen2, published selections), 'All_<sc>', 'Current_<sc>', 'Gen2_<sc>', single parts.
"""
import os
import sys
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from . import REPO_DIR
from .events import _poisson

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from IceCubeRates.future import nfss_detectors as ND  # noqa: E402
from IceCubeRates.future import nfss_forecast as NF  # noqa: E402
from IceCubeRates.future.diffuse import ATM_PARAMS, COSZ_EDGES, KEYS, XS_PARAMS, DiffuseDetector  # noqa: E402

WATER = ("ARCA", "GVD", "P-ONE", "TRIDENT", "NEON", "HUNT")
# the detectors' latitudes (deg), as NuFlavorSingleSource's calibrate() sets them (checked in the tests)
SITE_LAT = {"IceCube": -89.99, "Gen2": -89.99, "ARCA": 36.27, "GVD": 51.77, "P-ONE": 47.7, "TRIDENT": 17.4,
            "NEON": 17.4, "HUNT": 17.4}
BKG_PRIOR = 0.10
DET_PARAMS = ("eff", "escale", "misid")


def config_parts(config: str):
    """NuFlavorSingleSource's detector configurations (code/run.py CONFIGS): the detector parts they combine."""
    if config == "IceCube":
        return ["IceCube"]
    name, sc = config.rsplit("_", 1)
    if sc not in ("base", "mid", "ded"):
        raise ValueError(f"unknown future-detector configuration {config!r}")
    gen2 = ["Gen2_IC", "Gen2_new_" + sc]
    sets = {"Current": ["IceCube", "ARCA_" + sc, "GVD_" + sc], "All": ["IceCube"] + [w + "_" + sc for w in WATER],
            "All8": ["IceCube"] + [w + "_" + sc for w in WATER] + gen2, "Gen2": gen2}
    if name in sets:
        return sets[name]
    if name in WATER:
        return [config]
    raise ValueError(f"unknown future-detector configuration {config!r}")


def part_lat(part: str) -> float:
    return SITE_LAT[part.split("_")[0]]


def config_sites(config: str):
    """The latitudes the model must provide per-zenith fluxes for."""
    return sorted({part_lat(p) for p in config_parts(config)})


class FutureEventLikelihood:
    """-2 ln L of the future detectors' events at the model's values, profiled over their nuisances (module docstring).

    Arguments: model -- a FlavorModel in sky mode built with sites=config_sites(config); spec -- {"config":
    "All8_base", "cross_sections": False}."""

    def __init__(self, model, spec: Mapping):
        spec = dict(spec)
        bad = set(spec) - {"config", "cross_sections"}
        if bad:
            raise ValueError(f"data.events.future: unknown keys {sorted(bad)}; known: config, cross_sections")
        self.config = str(spec.get("config", "All8_base"))
        self.xs = bool(spec.get("cross_sections", False))
        self.model = model
        out, _ = ND.calibrate(verbose=False)
        self.dets = [DiffuseDetector(n, out[n]) for n in config_parts(self.config)]
        E_eV, _ = model.grid()
        self._lgE_model = np.log10(np.asarray(E_eV, dtype=np.float64) / 1e9)
        lo, hi = self._lgE_model[0], self._lgE_model[-1]
        lg = np.log10(NF.E)
        self._inside = (lg >= lo - 1e-9) & (lg <= hi + 1e-9)
        self._lg = np.clip(lg, lo, hi)
        self.blocks = [(d, ch) for d in self.dets for ch in d.channels]
        gparams = list(ATM_PARAMS) + (list(XS_PARAMS) if self.xs else [])
        names, centre, sigma, bounds = [], [], [], []
        for d, ch in self.blocks:
            names.append(f"{d.name}|{ch}|bkg")
            centre.append(1.0)
            sigma.append(BKG_PRIOR)
            bounds.append((0.2, 3.0))
        for p in gparams:
            names.append(p)
            centre.append(0.0)
            s = NF.global_prior(p)
            sigma.append(s)
            bounds.append((-5.0 * s, 5.0 * s))
        self._own = []
        for d, ch in self.blocks:
            own = {}
            for p in DET_PARAMS:
                if p == "misid" and ch == "trk":
                    continue
                own[p] = len(names)
                names.append(f"{d.name}|{ch}|{p}")
                centre.append(0.0)
                s = NF.DET_PRIOR[p]
                sigma.append(s)
                bounds.append((max(-5.0 * s, -0.9), 5.0 * s))
            self._own.append(own)
        self.names, self.centre, self.sigma = names, np.array(centre), np.array(sigma)
        self.bounds = bounds
        self._gidx = {p: names.index(p) for p in gparams}
        self._gparams = gparams
        # backgrounds, flattened per block
        self._B, self._GB, self._BX = [], [], []
        for d, ch in self.blocks:
            self._B.append(d.B[ch].reshape(-1))
            self._GB.append({p: d.GB[ch][p].reshape(-1) for p in gparams})
            self._BX.append({p: d.BX[ch][p].reshape(-1) for p in ("escale", "misid")})
        self.data = None
        self._z0 = np.zeros(len(names))
        self.last = {}
        self.n_calls = 0

    # ---------------------------------------------------------------- the astrophysical events
    def _phi(self, F):
        """(N_z, N_E_model, 3) per cos-zenith band -> {flavor index: (nE, N_z)} on the detectors' energy grid
        (log-log; zero outside the model's grid)."""
        L = np.log(np.maximum(F, 1e-300))                                   # (N_z, N_E, 3)
        out = {}
        for i in range(3):
            v = np.array([np.interp(self._lg, self._lgE_model, L[z, :, i]) for z in range(L.shape[0])])   # (N_z, nE)
            out[i] = (np.exp(v) * self._inside[None, :]).T
        return out

    def signals(self, values):
        """Per block: the signal and its templates (flattened), for the model's values."""
        fluxes = {}
        for d in self.dets:
            lat = part_lat(d.name)
            if lat not in fluxes:
                _, F_nu, F_nb = self.model.site_band_fluxes(values, lat)
                if F_nu.shape[0] != len(COSZ_EDGES) - 1:
                    raise ValueError("the model's direction bands must be the detectors' cos-zenith bins")
                fluxes[lat] = (self._phi(F_nu), self._phi(F_nb))
        S = []
        fl_index = {"e": 0, "mu": 1, "tau": 2}
        for d, ch in self.blocks:
            pn, pb = fluxes[part_lat(d.name)]
            base = 0.0
            tpl = {p: 0.0 for p in ("escale", "misid") + XS_PARAMS}
            for fl, bar in KEYS:
                phi = (pb if bar else pn)[fl_index[fl]]                     # (nE, N_z)
                base = base + np.einsum("ez,ebz->bz", phi, d.R[ch][(fl, bar)])
                for p in tpl:
                    tpl[p] = tpl[p] + np.einsum("ez,ebz->bz", phi, d.RX[ch][(fl, bar)][p])
            S.append(dict(s=np.asarray(base).reshape(-1), **{p: np.asarray(v).reshape(-1) for p, v in tpl.items()}))
        return S

    # ---------------------------------------------------------------- expectation and profile
    def _mu(self, x, S, grad=False):
        mus, parts = [], []
        for k, (sk, own) in enumerate(zip(S, self._own)):
            eff = x[own["eff"]]
            mu = (1.0 + eff) * sk["s"] + x[own["escale"]] * (sk["escale"] + self._BX[k]["escale"])
            if "misid" in own:
                mu = mu + x[own["misid"]] * (sk["misid"] + self._BX[k]["misid"])
            mu = mu + x[k] * self._B[k]
            for p in self._gparams:
                g = self._GB[k][p] + (sk[p] if p in XS_PARAMS else 0.0)
                mu = mu + x[self._gidx[p]] * g
            mus.append(mu)
        return mus

    def _dev(self, x, S, data, grad=True):
        mus = self._mu(x, S)
        f = 0.0
        g = np.zeros_like(x)
        for k, (mu, n, sk, own) in enumerate(zip(mus, data, S, self._own)):
            # the Poisson deviance, continued smoothly where no physical expectation goes (events._poisson): trial
            # steps of the profile into negative expectations stay finite, with consistent gradients
            dev, r = _poisson(mu, n, grad=True)
            f += dev
            g[k] += r @ self._B[k]
            g[own["eff"]] += r @ sk["s"]
            g[own["escale"]] += r @ (sk["escale"] + self._BX[k]["escale"])
            if "misid" in own:
                g[own["misid"]] += r @ (sk["misid"] + self._BX[k]["misid"])
            for p in self._gparams:
                g[self._gidx[p]] += r @ (self._GB[k][p] + (sk[p] if p in XS_PARAMS else 0.0))
        z = (x - self.centre) / self.sigma
        f += float(z @ z)
        g += 2.0 * z / self.sigma
        return f, g

    def profile(self, S, data):
        zb = [((lo - c) / s, (hi - c) / s) for (lo, hi), c, s in zip(self.bounds, self.centre, self.sigma)]

        def fun(z):
            f, g = self._dev(self.centre + self.sigma * z, S, data)
            return f, g * self.sigma
        r = minimize(fun, np.clip(self._z0, [b[0] for b in zb], [b[1] for b in zb]), jac=True, method="L-BFGS-B",
                     bounds=zb, options=dict(maxiter=5000, ftol=1e-15, gtol=1e-9))
        self._z0 = np.array(r.x)
        return self.centre + self.sigma * r.x, r

    def chi2(self, values: Mapping) -> float:
        if self.data is None:
            raise ValueError("the future detectors' likelihood has no data: a projection needs Asimov data "
                             "(set_asimov, i.e. the analysis' --asimov or data_centre)")
        self.n_calls += 1
        S = self.signals(values)
        x, r = self.profile(S, self.data)
        self.last = dict(nuisances=dict(zip(self.names, x.tolist())), value=float(r.fun), converged=bool(r.success),
                         nit=int(r.nit))
        return float(r.fun)

    def set_asimov(self, values: Mapping, nuisances: Mapping = None) -> None:
        x = self.centre.copy()
        for k, v in (nuisances or {}).items():
            x[self.names.index(k)] = float(v)
        self.data = [np.array(m) for m in self._mu(x, self.signals(values))]
        self._z0 = (x - self.centre) / self.sigma

    def expected(self, values: Mapping):
        """{(part, channel): (nb, nz) expected events} at the nuisances' centres: signal and background."""
        mus = self._mu(self.centre, self.signals(values))
        return {(d.name, ch): m.reshape(d.B[ch].shape) for (d, ch), m in zip(self.blocks, mus)}
