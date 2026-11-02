"""
surfaces.py

Likelihoods of measured or projected TeV-PeV flavor compositions, as analytic surfaces in
(f_e, f_mu) fitted to the published or simulated contours. The data are in
data/flavor_surfaces.json (moved from FrequentistAnalysis, 2026-10-12, with the same
numbers); the shapes below are those of FrequentistAnalysis' detector_flavor_likelihood.py,
with the same arithmetic, so the two agree bit for bit:

  Q = [z_x^2 - 2 rho z_x z_y + z_y^2] / (1 - rho^2),   z_x = (f_e - mu_x)/sig_x, ...
  Gaussian:    ln L = ln Lmax - Q/2
  Student's t: ln L = ln Lmax - (nu + 2)/2 ln(1 + Q/nu)

'Asymmetric' shapes take different widths and correlations left and right of mu_x,
switched sharply or blended by w = [1 + tanh((f_e - mu_x)/sig_x_avg)]/2 ('Smooth').
'Polar Ray-Casting (Fourier Smoothed)' reads the radii of the 68% and 95% contours along
each direction from Fourier series and interpolates Q = 2.30 (r/r68)^p between them.
"""
import json
import os
from functools import lru_cache

import numpy as np

from . import ANALYSIS_DIR

_DATA = os.path.join(ANALYSIS_DIR, "data", "flavor_surfaces.json")


@lru_cache(maxsize=None)
def _all_surfaces() -> dict:
    with open(_DATA) as f:
        raw = json.load(f)["surfaces"]
    for d in raw.values():
        if d.get("nu") == "inf":
            d["nu"] = np.inf
    return raw


def surface_names() -> list:
    """Names of every available surface, sorted."""
    return sorted(_all_surfaces())


def surface(name: str) -> dict:
    """The parameters of one surface. Argument: name -- e.g. 'pion_2050_case_5'.
    Raises KeyError naming the available surfaces if it is unknown."""
    try:
        return _all_surfaces()[name]
    except KeyError:
        raise KeyError(f"Unknown flavor surface {name!r}; available: {surface_names()}") from None


def log_likelihood(f_e, f_mu, params: dict) -> np.ndarray:
    """ln L of flavor fractions (f_e, f_mu) under one surface.

    Arguments: f_e, f_mu -- scalars or arrays of the same shape; params -- a surface
    (see surface()). Returns ln L, -inf where the shape is undefined.
    """
    x = np.atleast_1d(np.asarray(f_e, dtype=float))
    y = np.atleast_1d(np.asarray(f_mu, dtype=float))
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: f_e {x.shape}, f_mu {y.shape}")
    model = params.get("fit_model", "").lower()
    Lmax = params.get("Lmax", 1.0)
    log_Lmax = np.log(Lmax) if Lmax > 0 else 0.0
    mu_x, mu_y = params["mu_x"], params["mu_y"]

    def q_of(x_in, y_in, sx, sy, r):
        with np.errstate(divide="ignore", invalid="ignore"):
            z_x = (x_in - mu_x) / sx
            z_y = (y_in - mu_y) / sy
            if np.isscalar(r) and abs(r) >= 1.0:
                return np.full_like(x_in, np.inf)
            return (1.0 / (1.0 - r**2)) * (z_x**2 - 2 * r * z_x * z_y + z_y**2)

    if "polar ray-casting" in model:
        c68 = np.array(params["coeffs_68"])
        c95 = np.array(params["coeffs_95"])
        order = params.get("order", 10)
        dx, dy = x - mu_x, y - mu_y
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)

        def fourier(th, c):
            rad = c[0] * np.ones_like(th)
            i = 1
            for k in range(1, order + 1):
                rad += c[i] * np.cos(k * th) + c[i + 1] * np.sin(k * th)
                i += 2
            return rad

        r68 = np.maximum(fourier(theta, c68), 1e-4)
        r95 = np.maximum(fourier(theta, c95), r68 * 1.05)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.log(5.99 / 2.30) / np.log(r95 / r68)
        p = np.clip(p, 1.0, 4.0)
        Q = np.where(r < 1e-8, 0.0, (2.30 / (r68**p)) * (r**p))
    elif "asymmetric" in model:
        sxl, syl, rl = params["sig_x_l"], params["sig_y_l"], params["rho_l"]
        sxr, syr, rr = params["sig_x_r"], params["sig_y_r"], params["rho_r"]
        if "smooth" in model:
            sx_avg = (sxl + sxr) / 2.0
            if sx_avg < 1e-9:
                w = np.where(x < mu_x, 0.0, 1.0)
            else:
                w = 0.5 * (1.0 + np.tanh((x - mu_x) / sx_avg))
            Q = q_of(x, y, (1.0 - w) * sxl + w * sxr, (1.0 - w) * syl + w * syr,
                     (1.0 - w) * rl + w * rr)
        else:
            left = x < mu_x
            Q = np.zeros_like(x, dtype=float)
            if np.any(left):
                Q[left] = q_of(x[left], y[left], sxl, syl, rl)
            if np.any(~left):
                Q[~left] = q_of(x[~left], y[~left], sxr, syr, rr)
    else:
        Q = q_of(x, y, params["sig_x"], params["sig_y"], params["rho"])

    out = np.full_like(Q, -np.inf)
    ok = np.isfinite(Q)
    if "student" in model:
        nu = params.get("nu", 1.0)
        if nu <= 0:
            nu = 1.0
        factor = -(nu + 2.0) / 2.0
        out[ok] = log_Lmax + factor * np.log(1.0 + Q[ok] / nu)
    else:
        out[ok] = log_Lmax - 0.5 * Q[ok]
    return out


def chi2(f_e: float, f_mu: float, name) -> float:
    """-2 ln (L / L_max) of one composition under a surface (the data term of the
    flavor-composition analyses): zero at the surface's centre. FrequentistAnalysis used
    -2 ln L, larger by the constant -2 ln L_max (-15.1 for pion_2050_case_5); differences
    of chi^2 are the same. Arguments: f_e, f_mu; name -- a surface name, or a surface's
    parameters (see recentred). Returns a float (inf where the surface is undefined)."""
    params = surface(name) if isinstance(name, str) else name
    Lmax = params.get("Lmax", 1.0)
    log_Lmax = np.log(Lmax) if Lmax > 0 else 0.0
    return float(-2.0 * (log_likelihood(f_e, f_mu, params)[0] - log_Lmax))


def recentred(name: str, f_e: float, f_mu: float) -> dict:
    """The named surface moved rigidly so that its centre sits at (f_e, f_mu): the data term
    of an Asimov data set (the model's own prediction at chosen parameters) or of a
    pseudo-experiment whose measurement fluctuated by a shift of the centre."""
    params = dict(surface(name))
    params.update(mu_x=float(f_e), mu_y=float(f_mu), true_mu_x=float(f_e), true_mu_y=float(f_mu))
    return params
