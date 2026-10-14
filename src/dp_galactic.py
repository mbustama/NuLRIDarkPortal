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
from dp_core import njit
from dp_spatial import calc_MW_neutron_density

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DATA_DIRS = [_DATA]


def use_data_dir(path=None):
    """Look for the Galactic data files in `path` first, then in src/data (galactic.data_dir: e.g. the N_side 64
    reductions for tracks, tens of MB, kept outside the repository; tools/prepare_galactic.py --out). Clears the
    caches when the search path changes."""
    global _DATA_DIRS
    new = [os.path.abspath(path), _DATA] if path else [_DATA]
    if new != _DATA_DIRS:
        _DATA_DIRS = new
        _cache.clear()
        _emission_cache.clear()


def _path(fname):
    for d in _DATA_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{fname} not found in {_DATA_DIRS}")

# name -> (file, components summed): I = the diffuse intensity, U = the unresolved sources'
TEMPLATES = {"cringe": ("galactic_cringe.npz", ("I", "U")),
             "cringe_diffuse": ("galactic_cringe.npz", ("I",)),
             "pi0": ("galactic_pi0.npz", ("I",))}
# The normalization (x the template's model flux) used when galactic.norm_gal is not set: IceCube's 2026
# combined fit (arXiv:2607.25966, Table 1) where there is one, the model's own prediction otherwise.
NORM_DEFAULT = {"cringe": 0.91, "pi0": 4.99, "cringe_diffuse": 1.0}
KIND_SHARE = 0.5          # nu : nubar = 1 : 1 at the sources
# The extragalactic flux's anchor in the flavor fits (Mauricio, 2026-10-14): IceCube's 2026 combined-fit
# astrophysical flux (arXiv:2607.25966 Fig. 4, all-sky average, per flavor nu + nubar) in its 10-21.5 TeV bin,
# where IceCube's Galactic templates were fitted (src/data/ic26_fig4_digitized.json): E^2 Phi = 3.623e-8 GeV
# cm^-2 s^-1 sr^-1 (-20% / +33%) at the bin's geometric centre 10^4.167 GeV.
ICECUBE_2026_ANCHOR = dict(E_GeV=10 ** (4 + 1 / 6), E2Phi=3.623e-8, rel_lo=0.198, rel_hi=0.328)


def icecube_2026_anchor():
    """(per-flavor intensity in GeV^-1 cm^-2 s^-1 sr^-1, its energy in GeV) of the anchor, for
    sky.extragalactic_norm and sky.extragalactic_norm_E_GeV."""
    a = ICECUBE_2026_ANCHOR
    return a["E2Phi"] / a["E_GeV"] ** 2, a["E_GeV"]

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
    with np.load(_path(fname), allow_pickle=False) as d:
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
    sky.output = 'healpix': emitted intensity times the transfer of the pixel's band, or, for emission points
    in 3-D, times the pixel's local-eigenstate populations through its band's response."""
    k = "nu" if kind == "nu" else "nubar"
    em, band = sky["galactic_emitted"], sky["pixel_band"]
    if "galactic_populations_" + k in sky:
        F, R = sky["galactic_populations_" + k], sky["galactic_response_" + k]
        out = np.empty(F.shape)
        for b in range(R.shape[0]):
            m = band == b
            out[m] = np.einsum("pek,ekf->pef", F[m], R[b])
        return em[:, :, None] * out
    return em[:, :, None] * sky["galactic_transfer_" + k][band]


def inner_galaxy(grid, b_max_deg=15.0, l_max_deg=20.0):
    """Pixels of the inner Galaxy, |b| < b_max and |l| < l_max (arXiv:2607.25966 Fig. 4: 15 and 20 deg)."""
    l = np.where(grid.l > np.pi, grid.l - 2.0 * np.pi, grid.l)
    return (np.abs(grid.b) <= np.radians(b_max_deg)) & (np.abs(l) <= np.radians(l_max_deg))


# =============================================================================
# Emission points (step 6; DIRECTIONS_PLAN Section 3.6): the potential the Galactic neutrinos are born in
# =============================================================================
#
# A Galactic neutrino is emitted at a point x of its line of sight and travels adiabatically to the Sun
# (test T9), so the populations of its emission eigenstates -- those of H(E, sigma), sigma = 2 |Coef| E
# n_eff(x), ascending eigenvalues -- become those of the local eigenstates. n_eff(x; m) is the Yukawa-smoothed
# neutron density at x (src/data/mw_neff_tables.npz, tools/mw_neff_tables.py: the solver's Galactic model and
# integral). Where along each line of sight the neutrinos are emitted (src/data/galactic_emission.npz,
# tools/prepare_galactic.py --emission):
#   diffuse (cosmic rays on gas): the hydrogen column n_HI + 2 n_H2 of Soding et al. (2025) times CRINGE's
#     nucleon flux at E_PARENT_PER_NEUTRINO times the neutrino energy; the line-of-sight integral reproduces
#     the morphology of CRINGE's diffuse map (correlation 0.99);
#   unresolved sources (CRINGE's second map): the Lorimer et al. (2006) pulsar distribution with
#     exp(-|z| / 0.2 kpc) (Vecchiotti et al. 2022, ApJ 928, 19, the model CRINGE adopts), weighted by s: with
#     dN/dL ~ L^-1.5 and a flux threshold, the unresolved luminosity per source grows as the distance (the
#     nearby sources are resolved and removed); correlation 0.985 with CRINGE's map, which is broader at high
#     latitude and in the outer Galaxy (likely survey-dependent resolution there).
# Per pixel and energy the two mix in the proportions of the template's components (CRINGE: diffuse I and
# unresolved U; the pi0 template and cringe_diffuse are gas tracers, all diffuse).

R_SUN_KPC = 8.33                  # the Sun in the solver's Galactic model (dp_spatial.calc_MW_neutron_density)
E_PARENT_PER_NEUTRINO = 20.0      # nucleon energy per neutrino energy in pp collisions (E_nu ~ E_p / 20)
E_REF_GEV = 1e5                   # the diffuse weights' cosmic-ray energy: 20 x 100 TeV per nucleon
SOURCES = dict(B=1.9, C=5.0, R0_kpc=8.5, H_kpc=0.2, distance_power=1.0)
NEFF_NODES_PER_DECADE = 20
CONTACT_MIN_M_EV = 1e-12          # as dp_transit: above it n_eff is the local density
_emission_cache = {}


def _data_file(name):
    if name not in _emission_cache:
        _emission_cache[name] = dict(np.load(_path(name), allow_pickle=False))
    return _emission_cache[name]


def emission_geometry(nside=16):
    """(s (N_s,), ds (N_s,), R (N_pix, N_s), |z| (N_pix, N_s)): the distance bins of each pixel's line of sight
    (kpc, Sun-centred, as in galactic_emission.npz) and their galactocentric radius and height."""
    key = ("geometry", int(nside))
    if key not in _emission_cache:
        e = _data_file("galactic_emission.npz")
        edges = e["gas_s_edges_kpc"]
        s, ds = np.sqrt(edges[1:] * edges[:-1]), np.diff(edges)
        grid = dp_sky.SkyGrid(nside)
        l, b = grid.l[:, None], grid.b[:, None]
        X = R_SUN_KPC - s * np.cos(b) * np.cos(l)
        Y = s * np.cos(b) * np.sin(l)
        _emission_cache[key] = (s, ds, np.sqrt(X * X + Y * Y), np.abs(s * np.sin(b)))
    return _emission_cache[key]


def emission_weights(nside=16, E_ref_GeV=E_REF_GEV):
    """(q_gas, q_src), each (N_pix, N_s) and normalized per pixel: where along each line of sight the diffuse
    emission (gas column x nucleon flux at E_PARENT_PER_NEUTRINO x E_ref) and the unresolved sources' (Lorimer
    density x exp(-|z|/H) x s ds) are emitted."""
    key = ("weights", int(nside), float(E_ref_GeV))
    if key in _emission_cache:
        return _emission_cache[key]
    from scipy.interpolate import RegularGridInterpolator
    e = _data_file("galactic_emission.npz")
    s, ds, R, z = emission_geometry(nside)
    stored = sorted(int(n) for n in e["nsides"])
    src = min((n for n in stored if n >= nside and n % nside == 0), default=None)
    if src is None:
        raise ValueError(f"galactic_emission.npz holds N_side {stored}; N_side {nside} needs "
                         f"tools/prepare_galactic.py --emission --nside {nside}")
    col = e[f"gas_column_nside{src}"].astype(np.float64)
    if src != nside:
        col = degrade(col, src, nside)
    Ecr = e["cr_E_GeV_per_nucleon"]
    k = int(np.argmin(np.abs(np.log(Ecr / (E_PARENT_PER_NEUTRINO * E_ref_GeV)))))
    J = RegularGridInterpolator((e["cr_z_kpc"], e["cr_R_kpc"]), e["cr_nucleon_flux"][k].astype(np.float64),
                                bounds_error=False, fill_value=0.0)
    q_gas = col * J(np.stack([z.ravel(), R.ravel()], axis=-1)).reshape(R.shape)
    S = SOURCES
    q_src = ((R / S["R0_kpc"]) ** S["B"] * np.exp(-S["C"] * (R - S["R0_kpc"]) / S["R0_kpc"])
             * np.exp(-z / S["H_kpc"]) * s ** S["distance_power"] * ds)
    totals = []
    for q in (q_gas, q_src):
        tot = q.sum(axis=1, keepdims=True)
        if np.any(tot <= 0):
            raise ValueError("a pixel's line of sight carries no emission")
        q /= tot
        totals.append(tot[:, 0])
    _emission_cache[key] = (q_gas, q_src)
    _emission_cache[("totals",) + key[1:]] = tuple(totals)
    return q_gas, q_src


def emission_line_integrals(nside=16, E_ref_GeV=E_REF_GEV):
    """(gas, sources), each (N_pix,): the two emission models integrated along each line of sight (arbitrary
    units). Their morphology against CRINGE's diffuse and unresolved maps validates the emission distances
    (tests/test_galactic.py)."""
    emission_weights(nside, E_ref_GeV)
    return _emission_cache[("totals", int(nside), float(E_ref_GeV))]


@njit(cache=True)
def _density_RZ(R, z, out):
    """calc_MW_neutron_density (cm^-3) at galactocentric (R, z) kpc, through its Sun-centred coordinates."""
    for i in range(R.size):
        u = R_SUN_KPC - R.flat[i]
        r = np.sqrt(u * u + z.flat[i] * z.flat[i])
        if r == 0.0:
            out.flat[i] = calc_MW_neutron_density(0.0, 0.5 * np.pi, 0.0)
        else:
            out.flat[i] = calc_MW_neutron_density(r, np.arctan2(np.abs(u), z.flat[i]), 0.0 if u >= 0.0 else np.pi)


def effective_density(R, z, m_eff_eV, n_n0_eV3, R_H_eV_inv):
    """The Yukawa-smoothed neutron density n_eff (eV^3) at galactocentric (R, |z|) kpc for the mediator mass
    m_eff: the local density for m_eff >= 1e-12 eV (contact); below, the tabulated integral (log-linear in
    m between the table's masses, and in z and R between its nodes) plus the cosmic tail beyond the table's
    radius (the same everywhere: galactic_effective_density's tail). For ranges far beyond the Galaxy n_eff is
    not uniform: the Galaxy's share is m^2 times its neutrons' 'Coulomb' potential at x, ~6 times deeper at
    the Galactic Centre than at the Sun, and it exceeds the cosmic floor down to ~4e-31 eV. Below the table's
    lowest mass (1e-34 eV, ranges beyond the Hubble radius) both scale as m^2 and the pattern is frozen: the
    lowest mass's n_eff(x) / n_eff(Sun) times the Sun's exact value."""
    from dp_core import cm_inv3_to_eV3
    import dp_transit
    R = np.asarray(R, dtype=np.float64)
    z = np.abs(np.asarray(z, dtype=np.float64))
    if m_eff_eV >= CONTACT_MIN_M_EV:
        out = np.empty(R.shape)
        _density_RZ(np.ascontiguousarray(R), np.ascontiguousarray(z), out)
        return out * cm_inv3_to_eV3
    t = _data_file("mw_neff_tables.npz")
    lm = t["log10_m_eV"]
    if np.log10(m_eff_eV) < lm[0]:
        m0 = 10.0 ** lm[0]
        pattern = (effective_density(R, z, m0, n_n0_eV3, R_H_eV_inv)
                   / effective_density(np.array([R_SUN_KPC]), np.array([0.0]), m0, n_n0_eV3, R_H_eV_inv)[0])
        return pattern * dp_transit.galactic_effective_density(m_eff_eV, n_n0_eV3, R_H_eV_inv)
    if abs(float(t["n_n0_cm3"]) * cm_inv3_to_eV3 / n_n0_eV3 - 1) > 1e-9:
        raise ValueError("mw_neff_tables.npz was made for another cosmic neutron density; rerun tools/mw_neff_tables.py")
    a = min(int(np.searchsorted(lm, np.log10(m_eff_eV)) - 1), len(lm) - 2)
    a = max(a, 0)
    w = (np.log10(m_eff_eV) - lm[a]) / (lm[a + 1] - lm[a])
    # Between the nodes in (R, z): interpolate log n_eff - beta log n_local and add back beta log n_local of the
    # model at the point, exact. beta = 1 for ranges below ~6 pc (m >= 1e-24 eV), where n_eff follows the local
    # density (whose sums of exponentials are not log-linear between the z nodes), 0 for ranges beyond ~6 kpc
    # (m <= 1e-27 eV), where n_eff is smooth; log-linear in m between. Exact at the nodes either way.
    beta = float(np.clip((np.log10(m_eff_eV) + 27.0) / 3.0, 0.0, 1.0))
    logn = ((1.0 - w) * np.log(t["neff_cm3"][:, :, a]) + w * np.log(t["neff_cm3"][:, :, a + 1])
            - beta * np.log(t["n_local_cm3"]))
    Rn, zn = t["R_kpc"], t["z_kpc"]
    iR = np.clip(np.searchsorted(Rn, R) - 1, 0, len(Rn) - 2)
    iz = np.clip(np.searchsorted(zn, z) - 1, 0, len(zn) - 2)
    tR = np.clip((R - Rn[iR]) / (Rn[iR + 1] - Rn[iR]), 0.0, 1.0)
    tz = np.clip((z - zn[iz]) / (zn[iz + 1] - zn[iz]), 0.0, 1.0)
    v = ((1 - tR) * (1 - tz) * logn[iR, iz] + tR * (1 - tz) * logn[iR + 1, iz]
         + (1 - tR) * tz * logn[iR, iz + 1] + tR * tz * logn[iR + 1, iz + 1])
    if beta > 0.0:
        loc = np.empty(R.shape)
        _density_RZ(np.ascontiguousarray(R), np.ascontiguousarray(z), loc)
        v = v + beta * np.log(loc)
    from dp_core import m_to_eV_inv
    x1 = m_eff_eV * float(t["r_max_kpc"]) * 3.08567758e19 * m_to_eV_inv
    xH = m_eff_eV * R_H_eV_inv
    tail = n_n0_eV3 * ((1.0 + x1) * np.exp(-x1) - (1.0 + xH) * np.exp(-xH)) if xH > x1 else 0.0
    return np.exp(v) * cm_inv3_to_eV3 + tail


def emission_populations(E_eV, f_S, w_gas, q_gas, q_src, n_emit_eV3, coef2, tab):
    """Populations (N_pix, N_E, 3) of the local eigenstates (ascending eigenvalues) per unit flux emitted with
    flavors f_S, for emission points of density n_emit (N_pix, N_s) weighted by q_gas and q_src (N_pix, N_s)
    and mixed per pixel and energy with w_gas (N_pix, N_E) (the diffuse share; 1 - w_gas the sources').
    The emission densities are deposited on nodes NEFF_NODES_PER_DECADE per decade apart, linearly in
    log n between the two nearest (cloud in cell: the populations are interpolated linearly in log sigma),
    and each node's populations come from dp_transit.flavor_populations on the kind's frame table
    (sigma = E coef2 n, coef2 = 2 |Coef|)."""
    import dp_transit
    E_eV = np.ascontiguousarray(E_eV, dtype=np.float64)
    npix, ns = n_emit_eV3.shape
    F_fl = np.ascontiguousarray(np.broadcast_to(np.asarray(f_S, dtype=np.float64), (len(E_eV), 3)))
    out = np.empty((len(E_eV), 3))
    if np.ptp(n_emit_eV3) == 0.0:
        # one density everywhere (ranges beyond the Galaxy's reach): its populations, exactly
        dp_transit.flavor_populations(E_eV, F_fl, coef2, float(n_emit_eV3.flat[0]), tab["Q"], tab["Mp"], tab["s"],
                                      tab["floor"], out)
        return np.ascontiguousarray(np.broadcast_to(out, (npix, len(E_eV), 3)))
    logn = np.log10(np.maximum(n_emit_eV3, 1e-300))
    d = 1.0 / NEFF_NODES_PER_DECADE
    lo = np.floor(logn.min() / d) * d
    x = (logn - lo) / d
    j = np.floor(x).astype(np.int64)
    t = x - j
    n_nodes = int(j.max()) + 2
    rows = np.repeat(np.arange(npix), ns).reshape(npix, ns)
    H = np.zeros((2, npix, n_nodes))
    for c, q in enumerate((q_gas, q_src)):
        np.add.at(H[c], (rows, j), q * (1.0 - t))
        np.add.at(H[c], (rows, j + 1), q * t)
    used = np.nonzero(H.sum(axis=(0, 1)) > 0)[0]
    P = np.empty((len(used), len(E_eV), 3))
    for i, node in enumerate(used):
        dp_transit.flavor_populations(E_eV, F_fl, coef2, 10.0 ** (lo + node * d), tab["Q"], tab["Mp"], tab["s"],
                                      tab["floor"], out)
        P[i] = out
    P2 = P.reshape(len(used), -1)
    A_gas = (H[0][:, used] @ P2).reshape(npix, len(E_eV), 3)
    A_src = (H[1][:, used] @ P2).reshape(npix, len(E_eV), 3)
    w = np.asarray(w_gas, dtype=np.float64)[:, :, None]
    return w * A_gas + (1.0 - w) * A_src


def degrade_weighted(F, w, nside_in, nside_out):
    """Average per-pixel quantities F (N_pix_in, N_E, ...) onto N_side nside_out, weighted per energy by w
    (N_pix_in, N_E) (e.g. the emitted intensity): sum_children w F / sum_children w."""
    th, ph = dp_sky.pix2ang_ring(nside_in, np.arange(dp_sky.nside_to_npix(nside_in)))
    p = dp_sky.ang2pix_ring(nside_out, th, ph)
    npo = dp_sky.nside_to_npix(nside_out)
    w = np.asarray(w, dtype=np.float64)
    wF = (w.reshape(w.shape + (1,) * (F.ndim - 2)) * F).reshape(len(p), -1)
    num = np.stack([np.bincount(p, weights=col, minlength=npo) for col in wF.T], axis=1).reshape((npo,) + F.shape[1:])
    den = np.stack([np.bincount(p, weights=col, minlength=npo) for col in w.T], axis=1)
    return num / den.reshape(den.shape + (1,) * (F.ndim - 2))


def gas_share(name, E_GeV, nside=16):
    """(N_pix, N_E): the diffuse (cosmic rays on gas) share of the template's intensity -- CRINGE's diffuse map
    over the total; 1 for the gas tracers ('pi0', 'cringe_diffuse')."""
    if name != "cringe":
        return np.ones((dp_sky.nside_to_npix(nside), len(np.atleast_1d(E_GeV))))
    tot = intensity("cringe", E_GeV, nside)
    dif = intensity("cringe_diffuse", E_GeV, nside)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, np.clip(dif / tot, 0.0, 1.0), 1.0)
