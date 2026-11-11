"""
spectra.py

Measured astrophysical spectra as data terms (Mauricio, 2026-10-14: the flux normalization and shape float in the
statistical analysis; the flavor fits with the Galactic component take IceCube's 2026 spectrum as data).

  icecube_cf_2025     IceCube's combined fit of tracks and cascades (Phys. Rev. D 113, 062002 (2026), arXiv:2507.22234,
                      INSPIRE IceCube:2025ewu), Appendix B, Table 10, CF column (Mauricio, 2026-11-07: the published
                      table replaces the figure read below): per flavor (nu + nubar), an E^-2 normalization in each of
                      13 bands, 3 per decade from 1 TeV, then 10-100 PeV, with asymmetric errors; where the best fit is 0
                      the table's 68% upper limit is the upper error, the lower one 0
                      (src/data/icecube_cf_2025_table10.json). Inside the fits' grid (4.6 TeV - 30 PeV): the 10 bands
                      from 4.6 TeV to 10 PeV.
  ic26_astrophysical  (superseded 2026-11-07; kept for the earlier campaigns) the combined fit as plotted in IceCube
                      2026 (arXiv:2607.25966), Figure 4, whose caption cites the letter IceCube:2025tgp; its values
                      match Table 10's to a few percent, three error bars differ by up to a factor 2: the astrophysical flux of the combined fit, the
                      all-sky (isotropic) component alongside the Galactic templates, per flavor (nu + nubar, the
                      all-flavor flux over 3), in segments of 1/3 decade from 4.6 TeV to 1 PeV, each an E^-2
                      normalization with its error bar (src/data/ic26_fig4_digitized.json, read from the figure's
                      SVG). The segments' correlations are not published: they are taken as independent.

The model's counterpart of a segment [E_lo, E_hi] is the E^-2 normalization with the same number of neutrinos,
C = int Phi dE / int E^-2 dE, of the flux at the detector that the measurement's segments hold: for icecube_cf_2025,
whose segmented fit has no Galactic component (arXiv:2507.22234 Sec. V.5), the extragalactic plus the Galactic flux
(audit finding A5, decided 2026-11-09); for ic26_astrophysical, the extragalactic flux alone (the Galactic flux is
that fit's other component, constrained through norm_gal's prior); chi^2 = sum ((C - D) / sigma)^2 with sigma the upper error bar
where C > D and the lower one otherwise. Only segments inside the model's energy grid enter.
"""
import json
import os

import numpy as np

from . import REPO_DIR

SPECTRA = {
    # includes_galactic: whether the measurement's fit had no separate Galactic component, so that its segments
    # hold the Galactic neutrinos too (audit finding A5, decided 2026-11-09: Table 10's segmented fit has none,
    # arXiv:2507.22234 Sec. V.5); the model's counterpart is then the extragalactic plus the Galactic flux
    "icecube_cf_2025": dict(file=os.path.join(REPO_DIR, "src", "data", "icecube_cf_2025_table10.json"),
                            key="cf_per_flavour", includes_galactic=True,
                            source="IceCube:2025ewu (arXiv:2507.22234), Table 10, CF column (segmented fit)"),
    "ic26_astrophysical": dict(file=os.path.join(REPO_DIR, "src", "data", "ic26_fig4_digitized.json"),
                               key="astrophysical_all_sky_per_flavour", includes_galactic=False,
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
    out["includes_galactic"] = bool(spec["includes_galactic"])
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


def chi2(C, data: dict, rho: float = 0.0) -> float:
    """sum ((C - D) / sigma)^2, sigma the upper error where C > D and the lower one otherwise. A segment measured
    at 0 has a zero lower error: C = D = 0 adds nothing there (a flux is never negative). rho (default 0): a
    correlation between neighbouring segments, r^T R^-1 r with R tridiagonal (1 on the diagonal, rho beside it), over
    the segments with an error; a check of the published segments' unknown correlations (review item 5, 2026-11-11)."""
    C = np.asarray(C, dtype=np.float64)
    D = data["value"]
    sig = np.where(C > D, data["hi"] - D, D - data["lo"])
    r = np.divide(C - D, sig, out=np.zeros_like(C), where=sig > 0.0)
    if rho == 0.0:
        return float(np.sum(r ** 2))
    k = np.flatnonzero(sig > 0.0)
    n = len(k)
    R = np.eye(n) + rho * (np.eye(n, k=1) + np.eye(n, k=-1))
    rk = r[k]
    return float(rk @ np.linalg.solve(R, rk))
