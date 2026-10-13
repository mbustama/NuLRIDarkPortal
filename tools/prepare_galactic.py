"""
prepare_galactic.py -- reduce the Galactic-plane templates for the solver (docs/dev/DIRECTIONS_PLAN.md,
step 5). Run where the inputs are (Tycho: /lustre/astro/mbustama/external/galactic_data, downloaded
2026-10-13). Writes, small enough to commit, src/data/galactic_cringe.npz and src/data/galactic_pi0.npz:
the per-flavor intensity, nu + nubar combined, in GeV^-1 cm^-2 s^-1 sr^-1, on HEALPix RING pixels in
the equatorial frame (theta = pi/2 - dec), shape (N_E, N_pix), with provenance.

  CRINGE  Schwefer, Mertsch, Wiebusch, ApJ 949, 16 (2023), arXiv:2211.15607; Zenodo 7859442 v3
          (CC BY 4.0): the fiducial diffuse map Neutrino_AAfrag_Galprop_Ferr_Fiducial_256 (I_nside*) and
          the unresolved-source map Neutrino_Unresolved_Fiducial_256 (U_nside*), each N_side 256, RING,
          50 energies log-spaced over 10 GeV-1e8 GeV (header EMIN, EMAX), "per-flavor intensity in units
          of GeV-1 cm-2 s-1 sr-1" (the record). The paper's per-flavor relation J_nu(E) = 2 J_gamma(2E)
          (its Eq. 12) is the pp relation for nu + nubar of one flavor, IceCube's convention for model
          fluxes ("per-neutrino flavor, neutrino and anti-neutrino combined", the README of
          doi:10.21234/8v5d-rn16). The record names no frame: it is read off the maps (the plane at b = 0).
          IceCube's "CRINGE" is the SUM of the two: times 0.91, the sum reproduces the inner-Galaxy
          spectrum of arXiv:2607.25966 Fig. 4 to 0.3% from 3 to 37 TeV, the diffuse map alone only
          28-54% of it (src/data/ic26_fig4_digitized.json; tests/test_galactic.py).
  pi0     Fermi-LAT pi0 (GALPROP SS_Z4_R20_T150_C5, Ackermann et al., ApJ 750, 3 (2012)) as IceCube
          fits it: the spatial template of the 2023 release (MainResult/Fermi-LAT_pi0_map.npy,
          N_side 128, equatorial, "normalized over the sky", sr^-1) times a sky-integrated per-flavor
          (nu + nubar) power law E^-2.7 with E^2 Phi(100 TeV) = 4.4e-12 TeV cm^-2 s^-1, the model
          flux of IceCube 2026 (arXiv:2607.25966, supplement; fitted 4.99 times it; the 2023 fit,
          21.8e-12 in the same convention, is 4.95 times it); times 4.99 it reproduces Fig. 4 of
          arXiv:2607.25966 to 0.5%. Optionally checked against toise's copy of the map (Galactic frame),
          which tests the rotation used for CRINGE (correlation 0.997 at N_side 16).

  emission  (step 6: where along each line of sight the neutrinos are emitted) src/data/galactic_emission.npz:
          the gas of Soding et al., A&A 693, A139 (2025), Zenodo 12578443 (samples_densities_hpixr.fits: HI and
          H2 number densities, 8 posterior samples, on Sun-centred HEALPix N_side 64 NESTED Galactic pixels x 388
          log-spaced distances 0.05-30 kpc), reduced to columns of hydrogen nuclei n_HI + 2 n_H2 (mean of the
          samples) per output pixel and per 4 merged distance bins; and CRINGE's cosmic rays
          (Global_Hadronic_CR_Dist_CRINGE: DRAGON, R 0-20 kpc x z +-6 kpc x 63 kinetic energies per nucleon
          1-1e9 GeV, fluxes GeV^-1 m^-2 s^-1 sr^-1), reduced to the nucleon flux (p + 4 He) from 1 TeV.

The maps are rotated to the equatorial frame where needed and averaged onto the output pixels (every
input pixel goes to the output pixel holding its centre; same frame: exact area averages, since the
HEALPix hierarchy nests). Energies: the CRINGE nodes from 72 GeV to 1e8 GeV (44 nodes; the pi0 law on
the same nodes).

  python tools/prepare_galactic.py --cringe /lustre/astro/mbustama/external/galactic_data \
      --pi0 /lustre/astro/mbustama/external/galactic_data/IceCube_GP2023/Fermi-LAT_pi0_map.npy \
      [--pi0-check .../fermi_galactic_emission.npy] [--emission /lustre/astro/mbustama/external/galactic_data] \
      [--nside 16]          (--emission needs astropy and healpy: Tycho's environment has them)
"""
import argparse
import gzip
import hashlib
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_sky  # noqa: E402

CRINGE_FILE = "Neutrino_AAfrag_Galprop_Ferr_Fiducial_256.fits.gz"
CRINGE_UNRESOLVED_FILE = "Neutrino_Unresolved_Fiducial_256.fits.gz"
CRINGE_E_GEV = np.logspace(1.0, 8.0, 50)        # the record's 50 energies, 10 GeV-1e8 GeV
E_KEEP_GEV = (70.0, 1.0e8)
PI0_E2PHI_100TEV_GEV = 4.4e-12 * 1e3             # GeV cm^-2 s^-1: sky-integrated, per flavor, nu + nubar
PI0_GAMMA = 2.7


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fits_primary_image(path):
    """The primary image of a (gzipped) FITS file, read directly: header cards in 2880-byte blocks up to
    END, then the big-endian data. Returns (header dict, array with axes NAXISn ... NAXIS1)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        header = {}
        done = False
        while not done:
            block = f.read(2880)
            if len(block) < 2880:
                raise ValueError(f"{path}: header ends before END")
            for i in range(0, 2880, 80):
                card = block[i:i + 80].decode("ascii", "replace")
                if card[:8].strip() == "END":
                    done = True
                    break
                if card[8:10] == "= ":
                    header[card[:8].strip()] = card[10:].split("/")[0].strip().strip("'").strip()
        dims = [int(header[f"NAXIS{k}"]) for k in range(1, int(header["NAXIS"]) + 1)]
        dtype = {-64: ">f8", -32: ">f4"}[int(header["BITPIX"])]
        n = int(np.prod(dims))
        raw = f.read(n * np.dtype(dtype).itemsize)
        if len(raw) != n * np.dtype(dtype).itemsize:
            raise ValueError(f"{path}: data shorter than the header says")
        data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    return header, data.reshape(dims[::-1])


def pixel_directions(nside):
    th, ph = dp_sky.pix2ang_ring(nside, np.arange(dp_sky.nside_to_npix(nside)))
    return ph, 0.5 * np.pi - th                   # longitude, latitude of the map's own frame


def plane_fractions(m, nside):
    """Fractions of the map's sum within |b| < 5 deg if its pixels were Galactic and if they were
    equatorial (a Galactic-plane map holds most of its emission there; the band is 8.7% of the sky)."""
    lon, lat = pixel_directions(nside)
    _, b_eq = dp_sky.equatorial_to_galactic(lon, lat)
    cut = np.radians(5.0)
    return float(m[np.abs(lat) < cut].sum() / m.sum()), float(m[np.abs(b_eq) < cut].sum() / m.sum())


def check_frame(m, nside, frame, name):
    f_gal, f_eq = plane_fractions(m, nside)
    mine, other = (f_gal, f_eq) if frame == "galactic" else (f_eq, f_gal)
    if not (mine > 0.3 and mine > 3.0 * other):
        raise RuntimeError(f"{name}: the plane is not at b = 0 in the {frame} frame (|b| < 5 deg holds "
                           f"{f_gal:.3f} of the map read as Galactic, {f_eq:.3f} read as equatorial)")
    return (f"{frame} frame confirmed: |b| < 5 deg holds {mine:.3f} of the map ({other:.3f} if read in the "
            f"other frame)")


def to_equatorial(maps, nside_in, nside_out, frame):
    """Area averages of RING maps (..., N_pix_in) in the given frame on equatorial RING pixels."""
    lon, lat = pixel_directions(nside_in)
    if frame == "galactic":
        lon, lat = dp_sky.galactic_to_equatorial(lon, lat)
    p = dp_sky.ang2pix_ring(nside_out, 0.5 * np.pi - lat, lon)
    npo = dp_sky.nside_to_npix(nside_out)
    cnt = np.bincount(p, minlength=npo).astype(np.float64)
    if np.any(cnt == 0):
        raise RuntimeError(f"N_side {nside_in} -> {nside_out}: an output pixel holds no input pixel centre")
    flat = np.asarray(maps, dtype=np.float64).reshape(-1, len(p))
    out = np.array([np.bincount(p, weights=row, minlength=npo) / cnt for row in flat])
    return out.reshape(np.shape(maps)[:-1] + (npo,))


def summaries(E, I, nside):
    """Sky-integrated E^2 Phi and the inner Galaxy's mean E^2 Phi (|b| <= 15 deg, |l| <= 20 deg, as
    arXiv:2607.25966 Fig. 4), per flavor (nu + nubar), at 30 and 100 TeV, from (N_E, N_pix) equatorial maps."""
    ra, dec = pixel_directions(nside)
    l, b = dp_sky.equatorial_to_galactic(ra, dec)
    l = np.where(l > np.pi, l - 2 * np.pi, l)
    inner = (np.abs(b) <= np.radians(15.0)) & (np.abs(l) <= np.radians(20.0))
    omega = 4 * np.pi / I.shape[1]
    out = {}
    for e in (3e4, 1e5):
        lx = np.log(e)
        row = np.array([np.interp(lx, np.log(E), np.log(I[:, p])) for p in range(I.shape[1])])
        row = np.exp(row) * e ** 2
        out[f"sky_E2Phi_{e / 1e3:.0f}TeV_GeV_cm2_s"] = float(row.sum() * omega)
        out[f"inner_E2Phi_{e / 1e3:.0f}TeV_GeV_cm2_s_sr"] = float(row[inner].mean())
    out["inner_omega_sr"] = float(inner.sum() * omega)
    return out


def read_cringe(path):
    hdr, data = read_fits_primary_image(path)
    nside_in = int(hdr["NSIDE"])
    if data.shape != (50, 12 * nside_in ** 2):
        raise RuntimeError(f"{path}: shape {data.shape}, expected (50, {12 * nside_in ** 2})")
    if hdr.get("SCHEME", hdr.get("ORDERING", "RING")).upper() != "RING":
        raise RuntimeError(f"{path}: not RING ordered")
    if (float(hdr.get("EMIN", 10)), float(hdr.get("EMAX", 1e8))) != (10.0, 1e8):
        raise RuntimeError(f"{path}: energies {hdr.get('EMIN')}-{hdr.get('EMAX')}, expected 10-1e8 GeV")
    k100 = int(np.argmin(np.abs(CRINGE_E_GEV - 1e5)))
    return hdr, data, nside_in, check_frame(data[k100], nside_in, "galactic", os.path.basename(path))


def prepare_cringe(folder, nsides):
    """The diffuse map, and the unresolved-source map when present (both N_side 256, Galactic):
    I_nside* is the diffuse intensity, U_nside* the unresolved sources' (CRINGE's two components)."""
    path = os.path.join(folder, CRINGE_FILE)
    hdr, data, nside_in, frame_note = read_cringe(path)
    sel = (CRINGE_E_GEV >= E_KEEP_GEV[0]) & (CRINGE_E_GEV <= E_KEEP_GEV[1] * 1.001)
    E = CRINGE_E_GEV[sel]
    out = dict(E_GeV=E, template="cringe",
               source="CRINGE fiducial diffuse (Schwefer, Mertsch, Wiebusch 2023, arXiv:2211.15607), "
                      "Zenodo 7859442 v3 (CC BY 4.0), " + CRINGE_FILE,
               input_sha256=sha256(path), input_nside=nside_in, frame_check=frame_note,
               header=repr({k: hdr[k] for k in sorted(hdr)}))
    for ns in nsides:
        out[f"I_nside{ns}"] = to_equatorial(data[sel], nside_in, ns, "galactic").astype(np.float32)
    out.update(summaries(E, out[f"I_nside{nsides[0]}"].astype(np.float64), nsides[0]))
    upath = os.path.join(folder, CRINGE_UNRESOLVED_FILE)
    if os.path.exists(upath):
        del data
        uhdr, udata, unside, unote = read_cringe(upath)
        out.update(unresolved_source=CRINGE_UNRESOLVED_FILE, unresolved_sha256=sha256(upath),
                   unresolved_frame_check=unote, unresolved_header=repr({k: uhdr[k] for k in sorted(uhdr)}))
        for ns in nsides:
            out[f"U_nside{ns}"] = to_equatorial(udata[sel], unside, ns, "galactic").astype(np.float32)
        u = out[f"U_nside{nsides[0]}"].astype(np.float64)
        out.update({"unresolved_" + k: v for k, v in summaries(E, u, nsides[0]).items()})
        out.update({"total_" + k: v for k, v in summaries(E, u + out[f"I_nside{nsides[0]}"], nsides[0]).items()})
    return out


def pi0_spectrum(E):
    """Sky-integrated per-flavor (nu + nubar) dN/dE in GeV^-1 cm^-2 s^-1 at E (GeV)."""
    return PI0_E2PHI_100TEV_GEV / 1e10 * (E / 1e5) ** (-PI0_GAMMA)


def prepare_pi0(path, nsides, check_path=None):
    m = np.load(path, allow_pickle=False).astype(np.float64)
    nside_in = int(round(np.sqrt(m.size / 12)))
    if 12 * nside_in ** 2 != m.size or np.any(m < 0):
        raise RuntimeError("pi0 map: not a non-negative HEALPix map")
    frame_note = check_frame(m, nside_in, "equatorial", "pi0 (IceCube release)")
    sky_integral = m.sum() * 4 * np.pi / m.size
    E = CRINGE_E_GEV[(CRINGE_E_GEV >= E_KEEP_GEV[0]) & (CRINGE_E_GEV <= E_KEEP_GEV[1] * 1.001)]
    out = dict(E_GeV=E, template="pi0",
               source="Fermi-LAT pi0 (GALPROP SS_Z4_R20_T150_C5, Ackermann et al. 2012) spatial template of "
                      "the IceCube 2023 release (doi:10.21234/8v5d-rn16, MainResult/Fermi-LAT_pi0_map.npy) "
                      "times E^-2.7 with sky-integrated per-flavor E^2 Phi(100 TeV) = 4.4e-12 TeV cm^-2 s^-1 "
                      "(IceCube 2026, arXiv:2607.25966)",
               input_sha256=sha256(path), input_nside=nside_in, frame_check=frame_note,
               input_sky_integral=float(sky_integral))
    unit = m / sky_integral                                  # sr^-1, unit sky integral
    for ns in nsides:
        sp = to_equatorial(unit[None], nside_in, ns, "equatorial")[0]
        out[f"I_nside{ns}"] = (pi0_spectrum(E)[:, None] * sp[None, :]).astype(np.float32)
    out.update(summaries(E, out[f"I_nside{nsides[0]}"].astype(np.float64), nsides[0]))
    if check_path:
        c = np.load(check_path, allow_pickle=False).astype(np.float64)
        ns_c = int(round(np.sqrt(c.size / 12)))
        note = check_frame(c, ns_c, "galactic", "pi0 (toise copy)")
        ns = nsides[0]
        a = to_equatorial(unit[None], nside_in, ns, "equatorial")[0]
        b = to_equatorial(c[None] / (c.sum() * 4 * np.pi / c.size), ns_c, ns, "galactic")[0]
        hot = a > np.percentile(a, 90)
        out["toise_check"] = (f"toise's copy ({note}; sha256 {sha256(check_path)[:16]}) rotated to equatorial "
                              f"N_side {ns}: correlation with the release map {np.corrcoef(a, b)[0, 1]:.4f}; "
                              f"median |ratio - 1| in the brightest 10% of pixels "
                              f"{np.median(np.abs(b[hot] / a[hot] - 1)):.3f}")
    return out


GAS_FILE = "samples_densities_hpixr.fits"
CR_FILE = "Global_Hadronic_CR_Dist_CRINGE.fits.gz"
GAS_MERGE = 4


def prepare_emission(folder, nsides):
    """Gas columns per output pixel and merged distance bin, and CRINGE's nucleon flux (step 6)."""
    from astropy.io import fits
    import healpy
    gpath, cpath = os.path.join(folder, GAS_FILE), os.path.join(folder, CR_FILE)
    out = dict(gas_source="Soding et al. 2025 (A&A 693, A139), Zenodo 12578443, " + GAS_FILE,
               gas_sha256=sha256(gpath), cr_source="CRINGE (Schwefer, Mertsch, Wiebusch 2023) " + CR_FILE,
               cr_sha256=sha256(cpath))
    with fits.open(gpath, memmap=True) as h:
        edges = np.array(h[2].data.field(0), dtype=np.float64)
        hi, h2 = h["HI NUMBER DENSITY"], h["H2 NUMBER DENSITY"]
        nside_g = int(hi.header["NSIDE"])
        if hi.header["ORDERING"].lower() != "nest" or "galactic" not in hi.header["COMMENT3"].lower():
            raise RuntimeError("gas file: expected NESTED pixels in the Galactic frame")
        npix_g = 12 * nside_g ** 2
        th, ph = healpy.pix2ang(nside_g, np.arange(npix_g), nest=True)
        ra, dec = dp_sky.galactic_to_equatorial(ph, 0.5 * np.pi - th)
        nb = (len(edges) - 1) // GAS_MERGE
        e_m = edges[::GAS_MERGE][:nb + 1]
        width = np.diff(edges)[:nb * GAS_MERGE]
        out["gas_s_edges_kpc"] = e_m
        for ns in nsides:
            p = dp_sky.ang2pix_ring(ns, 0.5 * np.pi - dec, ra)
            npo = dp_sky.nside_to_npix(ns)
            col = np.zeros((npo, nb))
            cnt = np.bincount(p, minlength=npo).astype(np.float64)
            for a in range(0, npix_g, 2048):                     # chunks of pixels: ~200 MB per read
                b = min(a + 2048, npix_g)
                nH = (np.asarray(hi.section[a:b], dtype=np.float64).mean(axis=2)
                      + 2.0 * np.asarray(h2.section[a:b], dtype=np.float64).mean(axis=2))[:, :nb * GAS_MERGE]
                c = (nH * width).reshape(b - a, nb, GAS_MERGE).sum(axis=2)       # cm^-3 kpc
                np.add.at(col, p[a:b], c)
            out[f"gas_column_nside{ns}"] = (col / cnt[:, None]).astype(np.float32)
    with fits.open(cpath, memmap=False) as h:
        hd = h[0].header
        R = np.linspace(float(hd["RMIN"]), float(hd["RMAX"]), int(hd["DIMR"]))
        z = np.linspace(float(hd["ZMIN"]), float(hd["ZMAX"]), int(hd["DIMZ"]))
        Ek = float(hd["EKMIN"]) * float(hd["EKIN_FAC"]) ** np.arange(int(hd["DIME"]))
        flux = np.zeros((int(hd["DIMZ"]), int(hd["DIMR"]), int(hd["DIME"])))
        used = []
        for hdu in h[1:]:
            Z_, A_ = int(hdu.header["Z_"]), int(hdu.header["A"])
            if (Z_, A_) in ((1, 1), (2, 4)):                      # protons (primary and secondary), helium-4
                flux += A_ * np.asarray(hdu.data, dtype=np.float64)
                used.append((Z_, A_, int(hdu.header["SEC"])))
        keep = Ek >= 1e3 * 0.999
        out.update(cr_R_kpc=R, cr_z_kpc=z, cr_E_GeV_per_nucleon=Ek[keep],
                   cr_nucleon_flux=np.moveaxis(flux[:, :, keep], 2, 0).astype(np.float32),
                   cr_species=repr(used), cr_r_obs_kpc=float(hd["ROBS"]),
                   cr_units="GeV^-1 m^-2 s^-1 sr^-1 per nucleon, p + 4 He (shape only is used)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cringe", help="folder with " + CRINGE_FILE)
    ap.add_argument("--pi0", help="the IceCube 2023 release's MainResult/Fermi-LAT_pi0_map.npy")
    ap.add_argument("--pi0-check", help="toise's fermi_galactic_emission.npy (Galactic frame), a cross-check")
    ap.add_argument("--emission", help="folder with " + GAS_FILE + " and " + CR_FILE)
    ap.add_argument("--nside", type=int, nargs="+", default=[16])
    ap.add_argument("--out", default=os.path.join(REPO, "src", "data"))
    a = ap.parse_args()
    commit = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    jobs = [("cringe", lambda: prepare_cringe(a.cringe, a.nside)) if a.cringe else None,
            ("pi0", lambda: prepare_pi0(a.pi0, a.nside, a.pi0_check)) if a.pi0 else None,
            ("emission", lambda: prepare_emission(a.emission, a.nside)) if a.emission else None]
    for name, fn in filter(None, jobs):
        t0 = time.time()
        d = fn()
        d.update(commit=commit, host=os.uname().nodename, date=time.strftime("%Y-%m-%d %H:%M"),
                 nsides=np.array(a.nside), tool="tools/prepare_galactic.py")
        if name == "emission":
            d.update(frame="equatorial J2000, HEALPix RING, theta = pi/2 - dec; gas columns (N_pix, N_s) in "
                           "cm^-3 kpc of n_HI + 2 n_H2 along each pixel's line of sight from the Sun")
        else:
            d.update(units="GeV^-1 cm^-2 s^-1 sr^-1, per flavor, nu + nubar combined",
                     frame="equatorial J2000, HEALPix RING, theta = pi/2 - dec; arrays (N_E, N_pix)")
        p = os.path.join(a.out, f"galactic_{name}.npz")
        np.savez_compressed(p, **d)
        if name == "emission":
            col = d[f"gas_column_nside{a.nside[0]}"]
            print(f"emission: gas columns {col.shape} (total N_H per pixel: median {np.median(col.sum(1)):.3g}, "
                  f"max {col.sum(1).max():.3g} cm^-3 kpc), CR {d['cr_nucleon_flux'].shape} from "
                  f"{d['cr_E_GeV_per_nucleon'][0]:.3g} GeV per nucleon, species {d['cr_species']}; wrote {p} "
                  f"({os.path.getsize(p) / 1e6:.2f} MB, {time.time() - t0:.0f} s)")
            continue
        print(f"{name}: {d['frame_check']}")
        for k in sorted(d):
            if k.startswith(("sky_", "inner_", "unresolved_sky", "unresolved_inner", "total_")) or k in (
                    "toise_check", "unresolved_frame_check", "unresolved_header"):
                print(f"  {k} = {d[k]}")
        print(f"  {len(d['E_GeV'])} energies; wrote {p} ({os.path.getsize(p) / 1e6:.2f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
