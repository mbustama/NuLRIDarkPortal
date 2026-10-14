"""
flux_adapters.py -- fluxes as the event-rate folds take them: callables flux(E_GeV (N,), dec_lo (K,), dec_hi (K,))
returning (K, N) in GeV^-1 cm^-2 s^-1 sr^-1, each row the average over the declination bin [dec_lo, dec_hi]
(radians; the bins' solid angle is 2 pi d(sin dec)), for one channel's flavor (tracks: nu_mu + nubar_mu).

  power_law          IceCube-style isotropic fits (per-flavor nu + nubar normalization at a pivot);
  Atmospheric        the South Pole's atmospheric flux (tools/atmospheric_background.py: MCEq SIBYLL-2.3d x H3a)
                     with NuFlavorSingleSource's nuisances (Barr parameters, cosmic-ray index, prompt), or one of
                     its templates; cos(zenith) = -sin(dec);
  from_sky           a dark_portal_solver.propagate_sky output: extragalactic (in physical units: set
                     sky.extragalactic_norm) and Galactic per declination band, interpolated in log E.
  combined           the sum of several.
"""
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def power_law(per_flavor, gamma, pivot_GeV=1e5):
    """Phi = per_flavor (E / pivot)^-gamma, the same in every direction (per flavor, nu + nubar)."""
    def flux(E, dlo, dhi):
        return np.broadcast_to(per_flavor * (np.asarray(E) / pivot_GeV) ** (-gamma), (len(dlo), len(E))).copy()
    return flux


def _band_average(fn_cos, E, dlo, dhi):
    """Average over each declination bin, uniformly in sin(dec), of fn_cos(E, cos_zenith) with cos_zenith = -sin(dec)."""
    out = np.empty((len(dlo), len(E)))
    for k, (a, b) in enumerate(zip(np.sin(dlo), np.sin(dhi))):
        s = 0.5 * (a + b) + 0.5 * (b - a) * _GL_X
        out[k] = 0.5 * np.sum(_GL_W[:, None] * fn_cos(E, -s), axis=0)
    return out


# The atmospheric nuisances of NuFlavorSingleSource (code/forecast.py there): the Barr et al. hadron-production
# regions that matter above ~100 GeV (pi+- in G, H, I; K+- in W, Y, Z; priors stored with the table), the cosmic-ray
# spectral index, and the prompt normalization.
BARR_PARAMS = tuple(f"barr_{r}_{m}" for r in "ghi" for m in ("pi+", "pi-")) + \
    tuple(f"barr_{r}_{m}" for r in "wyz" for m in ("K+", "K-"))
ATM_PARAMS = BARR_PARAMS + ("cr_dgamma", "prompt")
OTHER_PRIOR = {"cr_dgamma": 0.05, "prompt": 0.5}


class Atmospheric:
    """The atmospheric flux at the South Pole (tools/atmospheric_background.py: MCEq SIBYLL-2.3d x H3a, the South
    Pole's atmosphere) for the chosen species (default nu_mu + nubar_mu), linear in the nuisances as in
    NuFlavorSingleSource:

        Phi = Phi_nom (1 + sum_p theta_p rel_p) + theta_prompt Phi_prompt,   rel_p = (dPhi/dp) / Phi_nom,

    p over BARR_PARAMS and cr_dgamma (ATM_PARAMS; priors from priors()). term selects one piece instead -- 'nom', 'prompt'
    (the template of theta_prompt), a parameter (its derivative, the template of theta_p), or 'daemonflux' /
    'daemonflux_err' (the calibrated conventional flux and its 1-sigma error, a cross-check; nu_mu species only).
    Interpolation as NuFlavorSingleSource's: log Phi_nom, rel_p and the prompt fraction linear in |cos zenith| and in
    log E (zero outside the table's energies); a direction at the detector takes the flux of the production zenith with
    the same |cos zenith| (up-going: produced on the far side; Earth absorption is in the effective areas)."""

    def __init__(self, nuisances=None, term=None, species=("numu", "antinumu"),
                 path=os.path.join(DATA, "atmospheric_mceq_southpole.npz")):
        nuisances = dict(nuisances or {})
        unknown = set(nuisances) - set(ATM_PARAMS)
        if unknown:
            raise ValueError(f"unknown atmospheric nuisances {sorted(unknown)}; known: {ATM_PARAMS}")
        with np.load(path, allow_pickle=False) as d:
            self.E = d["E_GeV"]
            self._prior = {str(p): float(v) for p, v in zip(d["params"], d["prior"]) if str(p).startswith("barr_")}
            self.models = str(d["models"])
            if term in ("daemonflux", "daemonflux_err"):
                if not set(species) <= {"numu", "antinumu"}:
                    raise ValueError("daemonflux is stored for nu_mu and nubar_mu only")
                key = "daemonflux_conv_{}" + ("_err" if term == "daemonflux_err" else "")
                self.cos = d["daemonflux_cos_zenith_detector"]
                self.table = sum(d[key.format(s)] for s in species)
                self.mirror, self._log, self._nom = False, True, None
                return
            nom = sum(d["nom_" + s] for s in species)
            pr = sum(d["pr_" + s] for s in species)
            deriv = {str(p): sum(d[f"d_{p}__{s}"] for s in species) for p in d["params"]}
            self.cos = d["cos_zenith"]
        self.mirror = True
        safe = np.maximum(nom, 1e-300)
        if term is None:
            rel = sum(v * deriv[p] for p, v in nuisances.items() if p != "prompt") / safe if nuisances else 0.0
            self.table = nom * (1.0 + rel) + nuisances.get("prompt", 0.0) * pr
            self._log = False
        elif term == "nom":
            self.table, self._log = nom, True
        elif term == "prompt":
            self.table, self._log = pr, False
        elif term in deriv:
            self.table, self._log = deriv[term], False
        else:
            raise ValueError(f"unknown term {term!r}")
        self._nom = nom

    def priors(self):
        """1-sigma priors of ATM_PARAMS: Barr et al.'s (from the table) and NuFlavorSingleSource's others."""
        return {p: self._prior.get(p, OTHER_PRIOR.get(p)) for p in ATM_PARAMS}

    def at_cos(self, E, cos_z):
        """(len(cos_z), len(E)): log-linear interpolation of the nominal flux, linear of the relative pieces."""
        E = np.asarray(E, dtype=np.float64)
        lE, lT = np.log(E), np.log(self.E)
        c = np.clip(np.abs(cos_z) if self.mirror else np.asarray(cos_z), self.cos[0], self.cos[-1])
        if self.mirror and not self._log:
            # relative to the nominal flux, as NuFlavorSingleSource interpolates its derivatives
            ref = np.maximum(self._nom, 1e-300)
            rows = np.array([np.interp(lE, lT, r, left=0.0, right=0.0) for r in self.table / ref])
            base = np.exp(np.array([np.interp(lE, lT, r, left=-np.inf, right=-np.inf) for r in np.log(ref)]))
        else:
            rows = np.array([np.interp(lE, lT, r, left=-np.inf, right=-np.inf)
                             for r in np.log(np.maximum(self.table, 1e-300))])
            base = None
        j = np.clip(np.searchsorted(self.cos, c) - 1, 0, len(self.cos) - 2)
        w = ((c - self.cos[j]) / (self.cos[j + 1] - self.cos[j]))[:, None]
        if base is None:
            return np.exp((1 - w) * rows[j] + w * rows[j + 1])
        lb = np.log(np.maximum(base, 1e-300))
        nomc = np.exp((1 - w) * lb[j] + w * lb[j + 1]) * (base[j] > 0) * (base[j + 1] > 0)
        return nomc * ((1 - w) * rows[j] + w * rows[j + 1])

    def __call__(self, E, dlo, dhi):
        return _band_average(self.at_cos, np.asarray(E, dtype=np.float64), dlo, dhi)


def from_sky(sky, flavor=1, extragalactic=True, galactic=True):
    """The flavor (1: nu_mu) flux, nu + nubar, of a propagate_sky output per declination band, interpolated log-log
    in energy (zero outside the solver's grid) and averaged over the requested bins by overlap in sin(dec). The
    extragalactic part must be in physical units (sky.extragalactic_norm)."""
    if extragalactic and "extragalactic_scale" not in sky["meta"]:
        raise ValueError("from_sky needs the extragalactic flux in physical units: set sky.extragalactic_norm")
    E_s = sky["E_eV"] / 1e9
    F = np.zeros(sky["bands_nu"].shape[:2])
    if extragalactic:
        F = F + sky["bands_nu"][:, :, flavor] + sky["bands_nubar"][:, :, flavor]
    if galactic and "galactic_bands_nu" in sky:
        F = F + sky["galactic_bands_nu"][:, :, flavor] + sky["galactic_bands_nubar"][:, :, flavor]
    s_edges = np.sin(sky["dec_band_edges"])

    def flux(E, dlo, dhi):
        lE = np.log(np.asarray(E, dtype=np.float64))
        with np.errstate(divide="ignore"):
            lF = np.log(np.maximum(F, 1e-300))
        per_band = np.exp(np.array([np.interp(lE, np.log(E_s), row, left=-np.inf, right=-np.inf) for row in lF]))
        out = np.empty((len(dlo), len(E)))
        for k, (a, b) in enumerate(zip(np.sin(dlo), np.sin(dhi))):
            w = np.clip(np.minimum(s_edges[1:], b) - np.maximum(s_edges[:-1], a), 0.0, None)
            out[k] = (w[:, None] * per_band).sum(0) / w.sum()
        return out
    return flux


def combined(*fluxes):
    def flux(E, dlo, dhi):
        return sum(f(E, dlo, dhi) for f in fluxes)
    return flux
