"""
dr2_counts.py -- IceTracks-DR2's events binned for the track likelihood (docs/dev/EVENT_LIKELIHOOD_PLAN.md, step 2):
each configuration's events (IC40, IC59, IC79, and IC86's eleven seasons together) histogrammed in reconstructed
declination (each configuration's effective-area bins) x log10 reconstructed energy (the reduced smearing's 0.1-dex
bins), on the full grids, so that the likelihood makes its own selection (Mauricio, 2026-10-14: IC79 + IC86 above
3 TeV). Reads the events only (columns: run, event, subevent, MJD, log10(E/GeV), AngErr, RA, Dec, Azimuth, Zenith);
the release is CC0 (Harvard Dataverse doi:10.7910/DVN/MMIIZA), and the counts are committed.

  python -I tools/dr2_counts.py --dr2 DIR [--irfs IceCubeRates/data/dr2_irfs.npz] [--out IceCubeRates/data/dr2_counts.npz]
"""
import argparse
import glob
import hashlib
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATTERNS = {"IC40": "IC40_exp.tab", "IC59": "IC59_exp.tab", "IC79": "IC79_exp.tab", "IC86": "IC86_*_exp.tab"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dr2", required=True, help="the release's directory (with events/)")
    ap.add_argument("--irfs", default=os.path.join(REPO, "IceCubeRates", "data", "dr2_irfs.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "IceCubeRates", "data", "dr2_counts.npz"))
    a = ap.parse_args()
    with np.load(a.irfs, allow_pickle=False) as d:
        reco = d["reco_edges_log10E"]
        dec = {c: d[f"{c}_aeff_dec_edges_deg"] for c in PATTERNS}
    out = dict(configs=np.array(list(PATTERNS)), reco_edges_log10E=reco,
               source=np.array("IceTracks-DR2 v3.1 events (doi:10.7910/DVN/MMIIZA, CC0); tools/dr2_counts.py"))
    for c, pat in PATTERNS.items():
        files = sorted(glob.glob(os.path.join(a.dr2, "events", pat)))
        if not files:
            raise FileNotFoundError(f"no {pat} under {a.dr2}/events")
        counts = np.zeros((len(dec[c]) - 1, len(reco) - 1), dtype=np.int64)
        n_all, digests = 0, []
        for p in files:
            ev = np.loadtxt(p, ndmin=2)
            n_all += len(ev)
            counts += np.histogram2d(ev[:, 7], ev[:, 4], bins=(dec[c], reco))[0].astype(np.int64)
            with open(p, "rb") as f:
                digests.append(f"{os.path.basename(p)} {hashlib.md5(f.read()).hexdigest()}")
        out[f"{c}_dec_edges_deg"] = dec[c]
        out[f"{c}_counts"] = counts
        out[f"{c}_n_events"] = np.int64(n_all)
        out[f"{c}_files"] = np.array(digests)
        print(f"{c}: {len(files)} files, {n_all} events, {int(counts.sum())} inside the grids", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
