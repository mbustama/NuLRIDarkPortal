"""Build IceCubeRates/data/mese_2025/mese_response.npz: MESE's response per morphology and channel, from
NuFlavorSingleSource's Plan A code (its per-channel energy model; audit decision 2026-11-09, Mauricio: "Port
per-channel model"). Nothing of the model is re-derived here: the areas, selections, deposits and resolution are
NuFlavorSingleSource's own functions, called as they are; this script only collects them on MESE's reconstructed-energy
bins, without the point-source rings.

Cascades and double cascades (NuFlavorSingleSource code/arcalike.py, ShowerResponse with IceCube's settings,
code/blocks.py icecube_showers): areas per channel and true-zenith bin = arcalike.tables(eps = MESE's selection per
zenith bin, showers.mese_eps; vr = 0.5, MESE's sum convention turned into the average); the deposited energy per kind
(showers.edep_samples: CSMS dsigma/dy, hadronic light yield, tau decays), weighted by the bin's selection, normalized,
spread log-normally with the 2023 deep-learning cascade resolution (depos.resolution_icecube) onto MESE's bins
(arcalike.lognormal_matrix with MESE's edges); one ring holding every direction.
Tracks (code/mese_tracks.py, for the gate only): area_tables (MESE's track areas, A_nu + A_nubar, halved here into the
average convention) and energy_table (MESE's one released track table, every channel; D133).

Convention of every stored area: the average one, [m^2] multiplying a flavor's nu + nubar flux; the Glashow column
(nue_gr) assumes nubar_e = half of it, so analysis code multiplies it by 2 x the nubar_e flux.
True energies: arcalike.FINE_EDGES (log10 E/GeV, 2-8 in 0.05). Zenith bins: MESE's ten, bin 0 = cos 0.8-1.

Usage: python tools/build_mese_response.py [NUFLAVORSINGLESOURCE_DIR]   (default ~/Research/NuFlavorSingleSource)
"""
import csv
import hashlib
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "IceCubeRates", "data", "mese_2025")
TOPO_FILES = dict(cascades="mese_cascades_energy_histogram_components.tab",
                  double_cascades="mese_double_cascades_energy_histogram_components.tab",
                  tracks="mese_tracks_energy_histogram_components.tab")


def reco_edges(topology):
    """log10 of MESE's reconstructed-energy edges [GeV] for a topology, from its released histogram (the last row holds
    the top edge only)."""
    with open(os.path.join(DATA, TOPO_FILES[topology])) as f:
        rows = list(csv.reader(f))[1:]
    return np.log10([float(r[0]) for r in rows if r and r[0]])


def main(nfss):
    code = os.path.join(nfss, "code")
    sys.path.insert(0, code)
    import arcalike  # noqa: E402
    import depos  # noqa: E402
    import mese_tracks  # noqa: E402
    import showers  # noqa: E402
    t0 = time.time()
    commit = subprocess.run(["git", "-C", nfss, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", nfss, "status", "--porcelain", "code"], capture_output=True,
                           text=True).stdout.strip()
    eps = [showers.mese_eps(j) for j in range(arcalike.NB)]
    fits = showers.mese_fits()
    tabs = arcalike.tables(None, eps, vr=0.5)
    out = dict(fine_edges=arcalike.FINE_EDGES, cos_edges=np.asarray(tabs["cos_edges"], float),
               E50=np.asarray(fits["E50"], float), n_sel=np.asarray(fits["n"], float))
    # the deposited-energy histograms per kind, true energy and zenith bin (selection applied, normalized)
    m = showers.model()
    lgm = np.log10(m.E)
    dep_e = 10 ** arcalike.DEP_C
    selw = np.array([eps[j](dep_e) for j in range(arcalike.NB)])
    kinds = sorted(set(showers.DEPOSIT[ch] for ch in arcalike.CHANNELS))
    H = {}
    for kind in kinds:
        h = np.zeros((arcalike.NB, len(arcalike.FINE_C), len(arcalike.DEP_C)))
        for i, lc in enumerate(arcalike.FINE_C):
            idx = int(np.argmin(abs(lgm - lc)))
            d = np.zeros(len(arcalike.DEP_C))
            for which in (0, 1):
                w, e = showers.edep_samples(kind, idx, which)
                pos = e > 0
                d += np.histogram(np.log10(e[pos]), bins=arcalike.DEP_EDGES, weights=w[pos])[0]
            for j in range(arcalike.NB):
                hb = d * selw[j]
                if hb.sum() > 0:
                    h[j, i] = hb / hb.sum()
        H[kind] = h
    for topo, key in (("cascades", "shw"), ("double_cascades", "dc")):
        lr = reco_edges(topo)
        out[f"{topo}_reco_edges"] = lr
        M = arcalike.lognormal_matrix(depos.resolution_icecube, lgr_edges=lr)       # (n_dep, n_reco)
        for kind in kinds:
            out[f"{topo}_J_{kind}"] = H[kind] @ M                                    # (NB, n_fine, n_reco)
        for ch, A in tabs[key].items():
            out[f"{topo}_A_{ch}"] = np.asarray(A, float)                             # (n_fine, NB) [m^2], average
    lr = reco_edges("tracks")
    out["tracks_reco_edges"] = lr
    At = mese_tracks.area_tables(arcalike.FINE_C)
    for ch, A in At.items():
        out[f"tracks_A_{ch}"] = 0.5 * np.asarray(A, float)                           # sum -> average convention
    P, lgr = mese_tracks.energy_table(arcalike.FINE_C)                              # (n_fine, n_reco_table)
    # onto MESE's histogram bins, uniform in log E inside each table bin (as NuFlavorSingleSource's gate G1)
    W = np.clip((np.minimum(lgr[1:, None], lr[None, 1:]) - np.maximum(lgr[:-1, None], lr[None, :-1]))
                / np.diff(lgr)[:, None], 0.0, None)                                  # (n_table, n_mese) overlaps
    out["tracks_P"] = P @ W
    out["kinds"] = np.array(kinds)
    out["deposit_kind_of"] = np.array([f"{ch}:{showers.DEPOSIT[ch]}" for ch in arcalike.CHANNELS])
    out["provenance"] = np.array(f"NuFlavorSingleSource {commit}{' (code modified)' if dirty else ''}; "
                                 f"tools/build_mese_response.py; built {time.strftime('%Y-%m-%d %H:%M')}")
    path = os.path.join(DATA, "mese_response.npz")
    np.savez_compressed(path, **out)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    print(f"wrote {path} (md5 {md5}) in {time.time() - t0:.1f} s; NuFlavorSingleSource {commit}, code dirty: {bool(dirty)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Research/NuFlavorSingleSource"))
