"""
spectra.py

Measured astrophysical spectra as data terms (Mauricio, 2026-10-14: the flux normalization and shape float in the
statistical analysis; the flavor fits with the Galactic component take IceCube's 2026 spectrum as data).

  ic26_astrophysical  IceCube 2026 (arXiv:2607.25966), Figure 4: the astrophysical flux of the combined fit, the
                      all-sky (isotropic) component alongside the Galactic templates, per flavor (nu + nubar, the
                      all-flavor flux over 3), in segments of 1/3 decade from 4.6 TeV to 1 PeV, each an E^-2
                      normalization with its error bar (src/data/ic26_fig4_digitized.json, read from the figure's
                      SVG). The segments' correlations are not published: they are taken as independent.

The model's counterpart of a segment [E_lo, E_hi] is the E^-2 normalization with the same number of neutrinos,
C = int Phi dE / int E^-2 dE, of the extragalactic flux at the detector (the Galactic flux is IceCube's other
component, constrained through norm_gal's prior); chi^2 = sum ((C - D) / sigma)^2 with sigma the upper error bar
where C > D and the lower one otherwise. Only segments inside the model's energy grid enter.
"""
import json
import os

import numpy as np

from . import REPO_DIR

SPECTRA = {
    "ic26_astrophysical": dict(file=os.path.join(REPO_DIR, "src", "data", "ic26_fig4_digitized.json"),
                               key="astrophysical_all_sky_per_flavour",
                               source="IceCube 2026, arXiv:2607.25966 Fig. 4 (combined fit, astrophysical component)"),
}
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def load(name: str, E_min_GeV: float = 0.0, E_max_GeV: float = np.inf) -> dict:
    """The segments of spectrum `name` lying inside [E_min_GeV, E_max_GeV] (to 1%): dict of arrays E_lo, E_hi
    (GeV), value, lo, hi (E^2 Phi, GeV cm^-2 s^-1 sr^-1, per flavor), and source."""
    if name not in SPECTRA:
        raise ValueError(f"Unknown spectrum {name!r}; known: {sorted(SPECTRA)}")
    spec = SPECTRA[name]
    with open(spec["file"]) as f:
        rows = json.load(f)[spec["key"]]
    rows = [r for r in rows if r["E_lo_GeV"] >= 0.99 * E_min_GeV and r["E_hi_GeV"] <= 1.01 * E_max_GeV]
    if not rows:
        raise ValueError(f"No segment of {name} lies inside {E_min_GeV:.3g}-{E_max_GeV:.3g} GeV")
    out = {k: np.array([float(r[s]) for r in rows]) for k, s in
           (("E_lo", "E_lo_GeV"), ("E_hi", "E_hi_GeV"), ("value", "E2Phi"), ("lo", "E2Phi_lo"), ("hi", "E2Phi_hi"))}
    if np.any(out["lo"] > out["value"]) or np.any(out["hi"] < out["value"]):
        raise ValueError(f"{name}: an error bar does not bracket its value")
    out["source"] = spec["source"]
    return out


def segments(E_GeV, phi, E_lo, E_hi) -> np.ndarray:
    """The E^-2 normalizations C_i = int Phi dE / int E^-2 dE over [E_lo_i, E_hi_i] of the flux phi(E_GeV) (per
    flavor, GeV^-1 cm^-2 s^-1 sr^-1), interpolated log-log between its nodes (Gauss-Legendre in log E per segment).
    Segments must lie inside the nodes' range to 1% (the edges are extrapolated log-log by at most that)."""
    lE, lphi = np.log(np.asarray(E_GeV, dtype=np.float64)), np.log(np.maximum(np.asarray(phi, dtype=np.float64), 1e-300))
    a, b = np.log(np.asarray(E_lo, dtype=np.float64)), np.log(np.asarray(E_hi, dtype=np.float64))
    x = 0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * _GL_X[None, :]                  # ln E nodes
    slope_lo = (lphi[1] - lphi[0]) / (lE[1] - lE[0])
    slope_hi = (lphi[-1] - lphi[-2]) / (lE[-1] - lE[-2])
    f = np.interp(x, lE, lphi)
    f = np.where(x < lE[0], lphi[0] + slope_lo * (x - lE[0]), f)
    f = np.where(x > lE[-1], lphi[-1] + slope_hi * (x - lE[-1]), f)
    num = np.sum(0.5 * (b - a)[:, None] * _GL_W[None, :] * np.exp(f + x), axis=1)       # int Phi dE = int Phi E dlnE
    den = np.exp(-a) - np.exp(-b)                                                         # int E^-2 dE
    return num / den


def chi2(C, data: dict) -> float:
    """sum ((C - D) / sigma)^2, sigma the upper error where C > D and the lower one otherwise."""
    C = np.asarray(C, dtype=np.float64)
    D = data["value"]
    sig = np.where(C > D, data["hi"] - D, D - data["lo"])
    return float(np.sum(((C - D) / sig) ** 2))
