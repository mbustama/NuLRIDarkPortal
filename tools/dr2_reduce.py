"""
dr2_reduce.py -- IceTracks-DR2's instrument responses reduced for the track rates (docs/dev/DR2_TRACKS_PLAN.md,
Mauricio's task 3). Reads the release (doi:10.7910/DVN/MMIIZA v3.1, CC0; on Tycho in external/icetracks_dr2) and
writes IceCubeRates/data/dr2_irfs.npz (committed), per detector configuration (IC40, IC59, IC79, IC86):

  aeff              A_eff(E_nu, dec_nu) in cm^2 on the release's own bins (0.2 dex in E, its declination bins),
                    nu_mu + nubar_mu, Earth absorption included (the release's simulation);
  smear             P(log10 E_reco, log10 PSF | E_nu bin, dec_nu bin) on the release's (E_nu, dec) bins (0.5 dex,
                    41 dec): the 5-D table's fractional counts summed over AngErr and spread onto common grids --
                    reconstructed energy in 0.1 dex (RECO), the point-spread angle (true neutrino to reconstructed
                    muon direction) in 0.25 dex (PSF) -- in proportion to each cell's overlap (uniform in log E and
                    log PSF within it; the release chooses its cells per (E_nu, dec) bin); also its total per
                    (E_nu, dec) bin (1 where events exist: the table's normalization);
  livetime_s        per configuration from the good-run lists (sum of MJD_stop - MJD_start), the IC86 seasons summed.

  python -I tools/dr2_reduce.py --dr2 /lustre/astro/mbustama/external/icetracks_dr2 [--out IceCubeRates/data/dr2_irfs.npz]

The smearing tables (0.6-1.0 GB) are streamed in chunks; nothing else from the release is copied.
"""
import argparse
import glob
import hashlib
import itertools
import os
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = ("IC40", "IC59", "IC79", "IC86")
RECO = np.round(np.arange(1.0, 9.0 + 1e-9, 0.1), 6)             # log10(E_reco / GeV) edges
# log10(PSF / deg) edges: 0.01 deg to 180 deg, 0.25 dex (the last bin 178-180 deg)
PSF = np.append(np.round(np.arange(-2.0, 2.25 + 1e-9, 0.25), 6), np.log10(180.0))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def read_table(path, chunk=2_000_000):
    """Whitespace table with a '#' header, streamed in chunks of rows."""
    with open(path) as f:
        f.readline()
        while True:
            lines = list(itertools.islice(f, chunk))
            if not lines:
                break
            yield np.loadtxt(lines, ndmin=2)


def reduce_aeff(path):
    a = np.concatenate(list(read_table(path)))
    le = np.unique(np.concatenate([a[:, 0], a[:, 1]]))
    de = np.unique(np.concatenate([a[:, 2], a[:, 3]]))
    A = np.zeros((len(le) - 1, len(de) - 1))
    i = np.searchsorted(le, a[:, 0])
    j = np.searchsorted(de, a[:, 2])
    if not (np.allclose(le[i + 1], a[:, 1]) and np.allclose(de[j + 1], a[:, 3])):
        raise ValueError(f"{path}: the effective-area bins do not tile a grid")
    A[i, j] = a[:, 4]
    if len(a) != A.size:
        raise ValueError(f"{path}: {len(a)} rows for {A.size} bins")
    return le, de, A


def overlaps(lo, hi, edges):
    """Fractions of each cell [lo, hi] (uniform within it) in the bins of edges, as K (bin, fraction) pairs per row:
    returns (index (K, N), fraction (K, N)). The part of a cell outside the edges is folded into its inside part (a
    cell wholly outside goes to the first or last bin); a cell of zero width goes to the bin holding it."""
    lo = np.clip(lo, edges[0], edges[-1])
    hi = np.clip(hi, edges[0], edges[-1])
    width = hi - lo
    first = np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, len(edges) - 2)
    last = np.maximum(np.clip(np.searchsorted(edges, hi, side="left") - 1, 0, len(edges) - 2), first)
    K = int(np.max(last - first)) + 1
    idx = np.empty((K, len(lo)), dtype=np.int64)
    frac = np.zeros((K, len(lo)))
    for t in range(K):
        b = np.minimum(first + t, len(edges) - 2)
        idx[t] = b
        ov = np.clip(np.minimum(hi, edges[b + 1]) - np.maximum(lo, edges[b]), 0.0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac[t] = np.where(width > 0, ov / np.where(width > 0, width, 1.0), 1.0 * (t == 0))
        frac[t] = np.where(first + t <= last, frac[t], 0.0)
    return idx, frac


def reduce_smearing(path):
    le_vals, de_vals, rows = set(), set(), []
    for c in read_table(path):
        le_vals.update(np.unique(c[:, 0:2]).tolist())
        de_vals.update(np.unique(c[:, 2:4]).tolist())
        rows.append(c[c[:, 10] > 0])
    a = np.concatenate(rows)
    le, de = np.array(sorted(le_vals)), np.array(sorted(de_vals))
    nE, nD, nR, nQ = len(le) - 1, len(de) - 1, len(RECO) - 1, len(PSF) - 1
    i = np.searchsorted(le, a[:, 0])
    j = np.searchsorted(de, a[:, 2])
    if not (np.allclose(le[i + 1], a[:, 1]) and np.allclose(de[j + 1], a[:, 3])):
        raise ValueError(f"{path}: the (E_nu, dec) bins do not tile a grid")
    ri, rf = overlaps(a[:, 4], a[:, 5], RECO)
    with np.errstate(divide="ignore"):
        qi, qf = overlaps(np.log10(np.maximum(a[:, 6], 1e-30)), np.log10(np.maximum(a[:, 7], 1e-30)), PSF)
    P = np.zeros(nE * nD * nR * nQ)
    base = (i * nD + j) * nR
    for t in range(ri.shape[0]):
        for u in range(qi.shape[0]):
            w = a[:, 10] * rf[t] * qf[u]
            P += np.bincount((base + ri[t]) * nQ + qi[u], weights=w, minlength=P.size)
    P = P.reshape(nE, nD, nR, nQ)
    lost = float(np.sum(a[:, 10]) - P.sum())
    return le, de, P, float(np.min(a[:, 4])), float(np.max(a[:, 5])), float(np.min(a[:, 6])), lost


def livetime(folder, config):
    pattern = "IC86_*_exp.tab" if config == "IC86" else f"{config}_exp.tab"
    total = 0.0
    files = sorted(glob.glob(os.path.join(folder, "uptime", pattern)))
    for p in files:
        u = np.loadtxt(p, ndmin=2)
        total += float(np.sum(u[:, 1] - u[:, 0])) * 86400.0
    return total, [os.path.basename(p) for p in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dr2", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "IceCubeRates", "data", "dr2_irfs.npz"))
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    a = ap.parse_args()
    t0 = time.time()
    out = dict(source=np.array("IceTracks-DR2 v3.1 (doi:10.7910/DVN/MMIIZA, CC0), reduced by tools/dr2_reduce.py"),
               reco_edges_log10E=RECO, psf_edges_log10deg=PSF, configs=np.array(a.configs))
    for c in a.configs:
        le, de, A = reduce_aeff(os.path.join(a.dr2, "irfs", f"{c}_effectiveArea.tab"))
        sp = os.path.join(a.dr2, "irfs", f"{c}_smearing.csv")
        sle, sde, P, rlo, rhi, psf_lo, lost = reduce_smearing(sp)
        lt, files = livetime(a.dr2, c)
        norm = P.sum(axis=(2, 3))
        out.update({f"{c}_aeff_log10E_edges": le, f"{c}_aeff_dec_edges_deg": de, f"{c}_aeff_cm2": A,
                    f"{c}_smear_log10E_edges": sle, f"{c}_smear_dec_edges_deg": sde, f"{c}_smear": P.astype(np.float32),
                    f"{c}_smear_norm": norm, f"{c}_livetime_s": lt, f"{c}_uptime_files": np.array(files),
                    f"{c}_smearing_md5": np.array(md5(sp)), f"{c}_reco_range": np.array([rlo, rhi])})
        print(f"{c}: aeff {A.shape} (log10 E {le[0]}-{le[-1]}, dec {de[0]}-{de[-1]}); smearing {P.shape}, "
              f"normalization min/max over filled bins {norm[norm > 0].min():.6f}/{norm.max():.6f} (lost to the grids "
              f"{lost:.2e}), reco {rlo}-{rhi}, smallest PSF {psf_lo:.3g} deg; livetime {lt / 86400:.1f} days "
              f"({len(files)} seasons); {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
