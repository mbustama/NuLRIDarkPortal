"""
mese.py

IceCube's MESE starting events (11.4 yr; flavor paper arXiv:2510.24957; release doi:10.7910/DVN/CBNMEB v2.0, CC0,
files in data/mese_2025/, README there): the expected events per reconstructed-energy bin of each morphology
(cascades, double cascades, tracks) as a fixed linear map of the flux per declination band, and the release's
histograms (Data, and the Conv, Prompt and Muon components at IceCube's fit).

Response: data/mese_2025/mese_response.npz, built by tools/build_mese_response.py from NuFlavorSingleSource's Plan A
code (audit decision 2026-11-09, Mauricio: its per-channel energy model): MESE's areas per channel and true-zenith bin
(divided by ln 10; the average convention, [m^2] times a flavor's nu + nubar flux), each channel's deposited energy
(CSMS inelasticity, hadronic light yield, tau decays) weighted by MESE's selection in that zenith bin, spread with the
deep-learning cascade resolution; tracks with MESE's one released track energy table. The Glashow column (nue_gr)
assumes nubar_e = half of nu_e + nubar_e, so here it takes 2 x the nubar_e flux.

Directions: at the South Pole cos(zenith) = -sin(dec); MESE's zenith bin j (bin 0: cos 0.8-1, down-going) covers dec
from arcsin(-cos_edges[j]) to arcsin(-cos_edges[j + 1]). Within a bin the flux is the bands' average weighted by
overlap in sin(dec), between energy nodes as flux_adapters.node_interpolation (as the DR2 fold).

Simplification (stated): IceCube fits cascades and tracks in (E_reco, cos theta_reco) and double cascades also in the
cascades' separation; the release gives 1-D projections, and the fit here uses reconstructed energy only.
"""
import csv
import os
from functools import lru_cache

import numpy as np

from ..flux_adapters import DATA as _DATA
from ..northern_muons.dr2 import _GL_W, _GL_X, band_overlaps, energy_weights

DATA = os.path.join(_DATA, "mese_2025")
RESPONSE = os.path.join(DATA, "mese_response.npz")
LIVETIME_S = 11.4 * 365.25 * 86400.0
TOPOLOGIES = ("cascades", "double_cascades", "tracks")
FLAVOR_OF = dict(nue_cc=0, nue_nc=0, nue_gr=0, numu_cc=1, numu_nc=1, nutau_cc=2, nutau_nc=2)
HIST_FILES = dict(cascades="mese_cascades_energy_histogram_components.tab",
                  double_cascades="mese_double_cascades_energy_histogram_components.tab",
                  tracks="mese_tracks_energy_histogram_components.tab")


@lru_cache(maxsize=None)
def response():
    """The stored response (a dict of arrays; module docstring)."""
    with np.load(RESPONSE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def histograms(topology):
    """The release's energy histogram of one morphology: dict(edges (n+1,) GeV, data, conv, prompt, muon (n,)); the
    double cascades have no Muon column (zeros)."""
    with open(os.path.join(DATA, HIST_FILES[topology])) as f:
        rows = list(csv.reader(f))
    head, body = rows[0], [r for r in rows[1:] if r and r[0]]
    col = {h.strip(): k for k, h in enumerate(head)}
    full = [r for r in body if len(r) > col["Data"] and r[col["Data"]] != ""]
    edges = np.array([float(r[0]) for r in body])
    if len(edges) != len(full) + 1:
        raise ValueError(f"{HIST_FILES[topology]}: {len(edges)} edges for {len(full)} bins")

    def column(name):
        return np.array([float(r[col[name]]) for r in full]) if name in col else np.zeros(len(full))
    return dict(edges=edges, data=column("Data"), conv=column("Conv"), prompt=column("Prompt"), muon=column("Muon"))


def _kernels(topology):
    """G[a] (NB, n_fine, n_reco): events per unit flux integral (GeV^-1 cm^-2 s^-1 sr^-1 x GeV) of flavor a's
    nu + nubar in each zenith bin and fine true-energy bin, over the livetime and the bin's solid angle; G['gr'] the
    Glashow channel per unit nubar_e flux (the factor 2 included)."""
    z = response()
    ce = z["cos_edges"]
    omega = 2.0 * np.pi * np.abs(np.diff(ce))                                     # (NB,)
    kind_of = dict(s.split(":") for s in z["deposit_kind_of"])
    nb, nf = len(omega), len(z["fine_edges"]) - 1
    n_reco = len(z[f"{topology}_reco_edges"]) - 1
    G = {a: np.zeros((nb, nf, n_reco)) for a in (0, 1, 2, "gr")}
    pre = f"{topology}_A_"
    for key in z:
        if not key.startswith(pre):
            continue
        ch = key[len(pre):]
        A = z[key].T * 1e4 * LIVETIME_S * omega[:, None]                         # (NB, n_fine) cm^2 s sr
        J = (np.broadcast_to(z["tracks_P"][None], (nb, nf, n_reco)) if topology == "tracks"
             else z[f"{topology}_J_{kind_of[ch]}"])
        if ch == "nue_gr":
            G["gr"] += 2.0 * A[:, :, None] * J
        else:
            G[FLAVOR_OF[ch]] += A[:, :, None] * J
    return G


class MESEFold:
    """Expected astrophysical MESE events per reconstructed-energy bin, per morphology, as a linear map of the model's
    band fluxes (module docstring).

    E_nodes_GeV: the flux's energy nodes; band_edges: declination band edges [rad] (ascending); topologies: which
    morphologies. __call__(F_nu, F_nb), each (n_bands, n_nodes, 3) in GeV^-1 cm^-2 s^-1 sr^-1, returns
    {topology: (n_reco,)}."""

    def __init__(self, E_nodes_GeV, band_edges, topologies=TOPOLOGIES):
        z = response()
        fe = z["fine_edges"]
        lo, hi = fe[:-1], fe[1:]
        x = 0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * _GL_X[None, :]
        E = 10.0 ** x
        dE_w = 0.5 * (hi - lo)[:, None] * _GL_W[None, :] * np.log(10.0) * E
        self.W_E = energy_weights(E, dE_w, np.asarray(E_nodes_GeV, dtype=np.float64))      # (n_fine, n_nodes)
        dec_edges = np.arcsin(np.clip(-z["cos_edges"], -1.0, 1.0))                         # ascending
        self.ov = band_overlaps(dec_edges, band_edges)                                      # (NB, n_bands)
        self.topologies = tuple(topologies)
        self.G = {t: _kernels(t) for t in self.topologies}
        self.reco_edges = {t: 10.0 ** z[f"{t}_reco_edges"] for t in self.topologies}

    def integrals(self, F):
        """(NB, n_fine): the flux F (n_bands, n_nodes) averaged over each zenith bin, integrated over each fine bin."""
        return (self.ov @ np.asarray(F, dtype=np.float64)) @ self.W_E.T

    def __call__(self, F_nu, F_nb):
        I = [self.integrals(F_nu[:, :, a] + F_nb[:, :, a]) for a in range(3)]
        I_gr = self.integrals(F_nb[:, :, 0])
        out = {}
        for t in self.topologies:
            G = self.G[t]
            out[t] = sum(np.einsum("ji,jir->r", I[a], G[a]) for a in range(3)) + np.einsum("ji,jir->r", I_gr, G["gr"])
        return out


def isotropic_kernels(topology):
    """For an isotropic flux on the response's fine grid: R[a] (n_fine, n_reco) per unit flux integral of flavor a's
    nu + nubar (a = 0, 1, 2) and R['gr'] per unit nubar_e flux integral, summed over the zenith bins."""
    return {a: g.sum(axis=0) for a, g in _kernels(topology).items()}
