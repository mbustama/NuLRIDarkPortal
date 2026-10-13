"""
dp_galactic.py

The Galactic-plane component of the per-direction output (docs/dev/DIRECTIONS_PLAN.md, Sections 3.4-3.6;
step 5, 2026-10-13).

Templates: src/data/galactic_<file>.npz (tools/prepare_galactic.py), the per-flavor intensity, nu + nubar
combined, in GeV^-1 cm^-2 s^-1 sr^-1 on equatorial HEALPix RING pixels (dp_sky's conventions):

  'cringe'          CRINGE (Schwefer, Mertsch, Wiebusch, ApJ 949, 16 (2023)), the diffuse emission plus the
                    unresolved sources, which is what IceCube fits as "CRINGE" (arXiv:2607.25966: 0.91 x
                    the model). Above ~30 TeV the unresolved sources dominate (75% of the sky-integrated
                    flux at 100 TeV).
  'cringe_diffuse'  the diffuse part alone: cosmic rays on gas (no IceCube fit; default normalization 1).
  'pi0'             the Fermi-LAT pi0 template with E^-2.7 as IceCube fits it (4.99 x the model,
                    E^2 Phi(100 TeV) = 4.4e-12 TeV cm^-2 s^-1 per flavor, sky-integrated).

Times the IceCube normalizations the templates reproduce the inner-Galaxy spectra (|b| < 15 deg,
|l| < 20 deg) of arXiv:2607.25966 Fig. 4 to 0.5% (src/data/ic26_fig4_digitized.json, tests/test_galactic.py).

The component: the templates are flavor-blind (all flavors 3 I, 1:1:1 assumed at Earth by IceCube). We keep
their total and apply our own physics: sources of composition f_S = (x, 1 - x, 0) (decision D6) emit nu and
nubar equally (pp collisions), so each kind carries the all-flavor intensity (3/2) N_gal I with flavors
f_S; the flux at the detector is that times the kind's transfer (dark_portal_solver._galactic_transfer):
the averaged oscillation probabilities for secret interactions, the local-emission transit through the Sun
and the Earth per declination band for the dark portal.
"""
import os

import numpy as np

import dp_sky

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# name -> (file, components summed): I = the diffuse intensity, U = the unresolved sources'
TEMPLATES = {"cringe": ("galactic_cringe.npz", ("I", "U")),
             "cringe_diffuse": ("galactic_cringe.npz", ("I",)),
             "pi0": ("galactic_pi0.npz", ("I",))}
# The normalization (x the template's model flux) used when galactic.norm_gal is not set: IceCube's 2026
# combined fit (arXiv:2607.25966, Table 1) where there is one, the model's own prediction otherwise.
NORM_DEFAULT = {"cringe": 0.91, "pi0": 4.99, "cringe_diffuse": 1.0}
KIND_SHARE = 0.5          # nu : nubar = 1 : 1 at the sources

_cache = {}


def degrade(maps, nside_in, nside_out):
    """Average RING maps (N_pix_in, ...) onto N_side nside_out (a divisor): every pixel lies inside one
    coarser pixel (the HEALPix hierarchy), so the mean of its children is the exact area average."""
    if nside_in % nside_out:
        raise ValueError(f"N_side {nside_out} does not divide {nside_in}")
    th, ph = dp_sky.pix2ang_ring(nside_in, np.arange(dp_sky.nside_to_npix(nside_in)))
    p = dp_sky.ang2pix_ring(nside_out, th, ph)
    npo = dp_sky.nside_to_npix(nside_out)
    cnt = np.bincount(p, minlength=npo).astype(np.float64)
    flat = np.asarray(maps, dtype=np.float64).reshape(len(p), -1)
    out = np.stack([np.bincount(p, weights=col, minlength=npo) for col in flat.T], axis=1) / cnt[:, None]
    return out.reshape((npo,) + np.shape(maps)[1:])


def load_template(name, nside=16):
    """(E_GeV (N_E,), I (N_pix, N_E), meta) for a template: the per-flavor intensity, nu + nubar combined,
    GeV^-1 cm^-2 s^-1 sr^-1, at the template's energy nodes, on N_side `nside` (stored or coarser)."""
    key = (name, int(nside))
    if key in _cache:
        return _cache[key]
    if name not in TEMPLATES:
        raise ValueError(f"Galactic template must be one of {sorted(TEMPLATES)}, got {name!r}")
    fname, comps = TEMPLATES[name]
    with np.load(os.path.join(_DATA, fname), allow_pickle=False) as d:
        stored = sorted(int(n) for n in d["nsides"])
        src = min((n for n in stored if n >= nside and n % nside == 0), default=None)
        if src is None:
            raise ValueError(f"{fname} holds N_side {stored}; N_side {nside} needs "
                             f"tools/prepare_galactic.py --nside {nside}")
        I = sum(d[f"{c}_nside{src}"].astype(np.float64) for c in comps).T
        E = d["E_GeV"].astype(np.float64)
        meta = dict(template=name, file=fname, components=comps, source=str(d["source"]),
                    commit=str(d["commit"]), nside_stored=src)
    if src != nside:
        I = degrade(I, src, nside)
    out = (E, np.ascontiguousarray(I), meta)
    _cache[key] = out
    return out


def intensity(name, E_GeV, nside=16):
    """The template's intensity (N_pix, N_E) at energies E_GeV, interpolated linearly in log I versus log E
    between its nodes (exact for power laws: the pi0 template; linearly where a node is zero)."""
    Et, I, _ = load_template(name, nside)
    lE, lEt = np.log(np.asarray(E_GeV, dtype=np.float64)), np.log(Et)
    if lE.min() < lEt[0] - 1e-9 or lE.max() > lEt[-1] + 1e-9:
        raise ValueError(f"Galactic template {name!r} covers {Et[0]:.3g}-{Et[-1]:.3g} GeV; asked for "
                         f"{np.exp(lE.min()):.3g}-{np.exp(lE.max()):.3g} GeV")
    k = np.clip(np.searchsorted(lEt, lE) - 1, 0, len(Et) - 2)
    t = np.clip((lE - lEt[k]) / (lEt[k + 1] - lEt[k]), 0.0, 1.0)
    lo, hi = I[:, k], I[:, k + 1]
    pos = (lo > 0.0) & (hi > 0.0)
    with np.errstate(divide="ignore"):
        loglin = np.exp((1.0 - t) * np.log(np.where(pos, lo, 1.0)) + t * np.log(np.where(pos, hi, 1.0)))
    return np.where(pos, loglin, (1.0 - t) * lo + t * hi)


def source_composition(x):
    """f_S = (x, 1 - x, 0), x in [0, 1] (decision D6: x floated on its own, default 1/3 for pp)."""
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"the Galactic source composition needs x in [0, 1], got {x}")
    return np.array([x, 1.0 - x, 0.0])


def averaged_transfer(U):
    """P[beta, alpha] = sum_i |U_beta,i|^2 |U_alpha,i|^2: flavor alpha emitted, beta detected, oscillations
    averaged over the kpc of the Galaxy (the same for nu and nubar)."""
    A = np.abs(np.asarray(U)) ** 2
    return A @ A.T


def band_means(I, pixel_band, n_bands):
    """Mean over each band's pixels (equal-area pixels, assigned by their centres) of maps (N_pix, ...)."""
    I = np.asarray(I, dtype=np.float64)
    cnt = np.bincount(pixel_band, minlength=n_bands).astype(np.float64)
    if np.any(cnt == 0):
        raise ValueError("a declination band holds no pixel centre: use a finer N_side or wider bands")
    flat = I.reshape(len(pixel_band), -1)
    out = np.stack([np.bincount(pixel_band, weights=col, minlength=n_bands) for col in flat.T], axis=1)
    return (out / cnt[:, None]).reshape((n_bands,) + I.shape[1:])


def emitted(name, E_GeV, nside, norm):
    """The all-flavor intensity emitted per kind (nu or nubar), (N_pix, N_E): KIND_SHARE x 3 x norm x I."""
    return KIND_SHARE * 3.0 * float(norm) * intensity(name, E_GeV, nside)


def pixel_flux(sky, kind="nu"):
    """The Galactic flux per pixel (N_pix, N_E, 3) from a dark_portal_solver.propagate_sky output with
    sky.output = 'healpix': emitted intensity times the transfer of the pixel's band."""
    T = sky["galactic_transfer_" + ("nu" if kind == "nu" else "nubar")]
    return sky["galactic_emitted"][:, :, None] * T[sky["pixel_band"]]


def inner_galaxy(grid, b_max_deg=15.0, l_max_deg=20.0):
    """Pixels of the inner Galaxy, |b| < b_max and |l| < l_max (arXiv:2607.25966 Fig. 4: 15 and 20 deg)."""
    l = np.where(grid.l > np.pi, grid.l - 2.0 * np.pi, grid.l)
    return (np.abs(grid.b) <= np.radians(b_max_deg)) & (np.abs(l) <= np.radians(l_max_deg))
