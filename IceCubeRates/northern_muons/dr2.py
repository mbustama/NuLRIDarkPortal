"""
dr2.py -- expected through-going muon-track events from IceCube's IceTracks-DR2 responses (docs/dev/DR2_TRACKS_PLAN.md).

For each detector configuration c (IC40, IC59, IC79, IC86), with its livetime T_c, effective area A_c(E_nu, dec_nu)
on the release's bins (0.2 dex x its declination bins; nu_mu + nubar_mu, Earth absorption included) and smearing
P_c(E_reco, psi | E_nu, dec_nu) (the release's 5-D table summed over AngErr, on a 0.1-dex reconstructed-energy grid
and a 0.25-dex grid of the point-spread angle psi between the true neutrino and the reconstructed muon direction;
tools/dr2_reduce.py):

    N_c(E_reco, dec_reco bin m) = T_c sum_{i,k} A_c(i, k) int_{E_i} dE int_{dec_k} dOmega Phi(E, dec)
                                  x sum_q P_c(E_reco, psi_q | i, k) G(k, q, m),

with Phi the nu_mu + nubar_mu flux (GeV^-1 cm^-2 s^-1 sr^-1; A_eff applies to the sum, checked against the release's
events by tools/dr2_validation.py) and G the geometry: the probability that a track from true declination bin k
(uniform in sin dec), deflected by psi in bin q (uniform in log psi) at an azimuth uniform around the true direction,
is reconstructed in declination bin m -- exact in the azimuth, sin dec_r = sin dec cos psi + cos dec sin psi cos phi.
The energy integral uses Gauss-Legendre nodes in log E within each 0.2-dex bin; the solid angle is 2 pi d(sin dec).
The smearing bin of each effective-area bin is the one holding its centre (the two binnings do not nest).
"""
import os
from functools import lru_cache

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dr2_irfs.npz")
_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)
_GT_X, _GT_W = np.polynomial.legendre.leggauss(16)     # per smooth piece of a true bin (see psf_kernel)
PSI_STEP_DEG = 0.5      # the largest spacing of the psi nodes (at least 16 Gauss-Legendre nodes per PSF bin)


def psf_kernel(true_dec_edges, psf_edges_log10deg, reco_dec_edges):
    """G[k, q, m] (rad edges; PSF edges in log10 deg): see the module docstring. Rows sum to 1 over m when the
    reconstructed bins cover -90 to 90 deg. Cached per set of edges (read-only result).

    For a reconstructed edge dec_e and angle psi, F = P(sin dec_r <= sin dec_e) = 1 - arccos(c)/pi with
    c = (sin dec_e - sin dec cos psi) / (cos dec sin psi), clipped to [-1, 1]: as a function of the true dec it is
    flat (0 or 1) or smooth between the breakpoints dec_e -+ psi and -+(180 deg - psi) - dec_e (where the circle of
    radius psi reaches dec_e, directly or over a pole), with square-root ends there. Its mean over a true bin
    (uniform in s = sin dec) splits the bin at those points and uses Gauss-Legendre on each piece in t, with
    s = a + (b - a)(1 - cos t)/2 clustering the nodes at the ends. In psi (uniform in log psi within each PSF bin),
    Gauss-Legendre nodes at most PSI_STEP_DEG apart."""
    key = tuple(np.asarray(x, dtype=np.float64).tobytes() for x in (true_dec_edges, psf_edges_log10deg, reco_dec_edges))
    return _psf_kernel(*key)


@lru_cache(maxsize=32)
def _psf_kernel(true_b, psf_b, reco_b, chunk=16):
    true_dec_edges, psf_edges, reco_dec_edges = (np.frombuffer(b, dtype=np.float64) for b in (true_b, psf_b, reco_b))
    s_lo, s_hi = np.sin(true_dec_edges[:-1]), np.sin(true_dec_edges[1:])                       # (K,)
    lo, hi = s_lo[:, None, None, None], s_hi[:, None, None, None]
    d_e = reco_dec_edges[None, :, None]                                                        # (1, M + 1, 1)
    e = np.sin(reco_dec_edges)[None, None, :, None, None]
    t = 0.5 * np.pi * (1.0 + _GT_X)                                                            # nodes on [0, pi]
    tw = 0.5 * np.sin(t) * 0.5 * np.pi * _GT_W
    G = np.zeros((len(s_lo), len(psf_edges) - 1, len(reco_dec_edges) - 1))
    for q in range(len(psf_edges) - 1):
        q_lo, q_hi = psf_edges[q], psf_edges[q + 1]
        n = max(16, int(np.ceil((10.0 ** q_hi - 10.0 ** q_lo) / PSI_STEP_DEG)))
        x, w = np.polynomial.legendre.leggauss(n)
        for c0 in range(0, n, chunk):
            xs, ws = x[c0:c0 + chunk], w[c0:c0 + chunk]
            psi = np.minimum(np.radians(10.0 ** (0.5 * (q_lo + q_hi) + 0.5 * (q_hi - q_lo) * xs)), np.pi)[:, None, None]
            bp = np.sin(np.clip(np.concatenate([d_e - psi, d_e + psi, np.pi - psi - d_e, psi - np.pi - d_e], axis=-1),
                                -0.5 * np.pi, 0.5 * np.pi))                                    # (n_c, M + 1, 4)
            pts = np.sort(np.concatenate([np.broadcast_to(lo, lo.shape[:1] + bp.shape[:2] + (1,)),
                                          np.clip(bp[None], lo, hi),
                                          np.broadcast_to(hi, hi.shape[:1] + bp.shape[:2] + (1,))], axis=-1), axis=-1)
            a, width = pts[..., :-1], np.diff(pts, axis=-1)                                    # (K, n_c, M + 1, 5)
            sn = a[..., None] + width[..., None] * 0.5 * (1.0 - np.cos(t))                     # (..., 5, n_t)
            den = np.sqrt(np.clip(1.0 - sn * sn, 0.0, None)) * np.sin(psi)[None, :, :, :, None]
            num = e - sn * np.cos(psi)[None, :, :, :, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where(num >= 0, np.inf, -np.inf))
            F = 1.0 - np.arccos(np.clip(c, -1.0, 1.0)) / np.pi
            I = np.einsum("knmjt,knmj,t->knm", F, width, tw) / (s_hi - s_lo)[:, None, None]      # mean of F over the bin
            if reco_dec_edges[0] <= -0.5 * np.pi:
                I[..., 0] = 0.0
            if reco_dec_edges[-1] >= 0.5 * np.pi:
                I[..., -1] = 1.0
            G[:, q, :] += np.einsum("kpm,p->km", np.diff(I, axis=-1), 0.5 * ws)
    G.setflags(write=False)
    return G


def overlap_kernel(true_dec_edges, n_psf, reco_dec_edges):
    """No direction smearing: G[k, q, m] = the fraction of true bin k (uniform in sin dec) inside reconstructed bin m."""
    a, b = np.sin(true_dec_edges[:-1]), np.sin(true_dec_edges[1:])
    e = np.sin(np.asarray(reco_dec_edges, dtype=np.float64))
    ov = np.clip(np.minimum(b[:, None], e[None, 1:]) - np.maximum(a[:, None], e[None, :-1]), 0.0, None) / (b - a)[:, None]
    return np.repeat(ov[:, None, :], n_psf, axis=1)


class DR2Tracks:
    """The reduced DR2 responses (IceCubeRates/data/dr2_irfs.npz) and the fold."""

    def __init__(self, path=DATA):
        with np.load(path, allow_pickle=False) as d:
            self.data = {k: d[k] for k in d.files}
        self.configs = [str(c) for c in self.data["configs"]]
        self.reco_edges = self.data["reco_edges_log10E"]
        self.psf_edges = self.data["psf_edges_log10deg"]

    def livetime(self, c):
        return float(self.data[f"{c}_livetime_s"])

    def aeff(self, c):
        """(log10 E edges, dec edges in rad, A_eff in cm^2 (n_E, n_dec))."""
        return (self.data[f"{c}_aeff_log10E_edges"], np.radians(self.data[f"{c}_aeff_dec_edges_deg"]),
                self.data[f"{c}_aeff_cm2"])

    def dec_edges(self, c="IC86"):
        """The effective area's declination edges (rad): the bands to compute fluxes on."""
        return np.radians(self.data[f"{c}_aeff_dec_edges_deg"])

    def expected(self, flux, configs=None, reco_dec_edges=None, psf=True):
        """Expected events: dict(reco_edges (log10 GeV), reco_dec_edges (rad), counts (n_dec_reco, n_reco) summed
        over configurations, per_config {c: (n_dec_reco, n_reco)}, true_counts {c: (n_E, n_dec)} on the effective
        area's bins (true energy and declination)).

        flux(E_GeV (N,), dec_lo (K,), dec_hi (K,)) -> (K, N): the nu_mu + nubar_mu flux averaged over each true
        declination bin (GeV^-1 cm^-2 s^-1 sr^-1) -- IceCubeRates.flux_adapters. reco_dec_edges (rad): the
        reconstructed declination bins (default: the effective area's); psf=False takes the reconstructed
        declination to be the true one."""
        configs = self.configs if configs is None else list(configs)
        out, true_counts, total = {}, {}, None
        for c in configs:
            le, de, A = self.aeff(c)
            rde = de if reco_dec_edges is None else np.asarray(reco_dec_edges, dtype=np.float64)
            sle = self.data[f"{c}_smear_log10E_edges"]
            sde = np.radians(self.data[f"{c}_smear_dec_edges_deg"])
            lo, hi = le[:-1], le[1:]
            x = 0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * _GL_X[None, :]
            E = 10.0 ** x                                                       # (n_E, n_gl)
            dE_w = 0.5 * (hi - lo)[:, None] * _GL_W[None, :] * np.log(10.0) * E  # dE = E ln10 dlog10E
            phi = flux(E.ravel(), de[:-1], de[1:]).reshape(len(de) - 1, E.shape[0], E.shape[1])
            omega = 2.0 * np.pi * (np.sin(de[1:]) - np.sin(de[:-1]))           # (n_dec,)
            n_true = self.livetime(c) * A * np.einsum("kig,ig->ik", phi, dE_w) * omega[None, :]
            si = np.clip(np.searchsorted(sle, 0.5 * (lo + hi)) - 1, 0, len(sle) - 2)
            sk = np.clip(np.searchsorted(sde, 0.5 * (de[:-1] + de[1:])) - 1, 0, len(sde) - 2)
            S = self.data[f"{c}_smear"][si][:, sk].astype(np.float64)          # (n_E, n_dec, n_reco, n_psf)
            X = np.einsum("ik,ikrq->krq", n_true, S)
            G = (psf_kernel(de, self.psf_edges, rde) if psf else overlap_kernel(de, S.shape[-1], rde))
            reco = np.einsum("krq,kqm->mr", X, G)
            out[c], true_counts[c] = reco, n_true
            total = reco if total is None else total + reco
        return dict(reco_edges=self.reco_edges, reco_dec_edges=rde, counts=total, per_config=out,
                    true_counts=true_counts)
