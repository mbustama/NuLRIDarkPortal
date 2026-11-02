"""
diffuse.py -- the future detectors' diffuse analysis (docs/dev/FUTURE_DETECTORS_PLAN.md; decided with Mauricio
2026-10-15): NuFlavorSingleSource's detector models (nfss_forecast, nfss_detectors: vendored unchanged), counted per
channel (tracks, showers, double cascades), reconstructed-energy bin and local cos-zenith bin, for a flux that
depends on the local zenith (the solver's per-direction output), instead of in rings around a point source.

Per detector part and channel, as NuFlavorSingleSource's Detector.build():
  - each (flavor, nu/nubar) response: its effective-area components (Detector.aeff: area, energy proxy), the
    Gaussian migration to reconstructed energy (bias -0.15 for tracks), and, over the cos-zenith bin, the sky
    acceptance, the Earth's absorption (earth_survival; the solver propagates flavor, not absorption) and the
    down-going showers' gate, over `years` of exposure: R[ch][(fl, bar)] (nE, nb, nz), events per unit flux
    (GeV^-1 cm^-2 s^-1 sr^-1) on the true-energy grid nfss_forecast.E;
  - the atmospheric background (atm_flux at |cos Z|, times the atmospheric-muon multiplier bkgmult for showers)
    B[ch] (nb, nz), and its templates per unit of each global parameter (Barr, cosmic-ray index, prompt) GB[ch][p];
  - the per-block systematics' templates (per unit parameter): the signal's (escale, misid for showers and double
    cascades) RX[ch][(fl, bar)][p], the background's (escale, misid) BX[ch][p]; the efficiency scales the signal
    (1 + eff). No cross-section variation (Mauricio, 2026-10-15): it needs the Earth propagation with regeneration and
    effective areas rebuilt with the varied cross sections, not a template on a fixed calibration.
The diffuse astrophysical flux, which NuFlavorSingleSource counted as a background around its source, is here the
signal: the model's flux.

Direction bins: the solver's declination bands at the South Pole are cos-zenith bands (cos Z = -sin dec): 20 bins of
0.1 in cos Z by default (COSZ_EDGES), integrated on N_SUB points each (exact for the acceptances' steps at 0 and 0.17).
"""
import numpy as np

from . import nfss_forecast as F

COSZ_EDGES = np.linspace(-1.0, 1.0, 21)
N_SUB = 20                  # 0.005 in cos Z: water tracks' acceptance edge (cos Z = 0.17) falls between two points
CHANNELS = ("trk", "shw", "dc")
KEYS = [(fl, bar) for fl in F.FLAVS for bar in (0, 1)]
DET_PARAMS = ("eff", "escale", "misid")                 # NuFlavorSingleSource's per-block systematics, but the PSF
ATM_PARAMS = tuple(F.BARR_PARAMS) + ("cr_dgamma", "prompt")


def _sub_points(edges, n):
    """(cos Z, weight) per bin: n midpoints of equal width, weights summing to the bin's width."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        c = a + (np.arange(n) + 0.5) * (b - a) / n
        out.append((c, np.full(n, (b - a) / n)))
    return out


class DiffuseDetector:
    """One detector part (NuFlavorSingleSource's parameter set P) counted per channel, reconstructed energy and local
    cos-zenith bin. Attributes per channel: R, B, GB, RX, BX (module docstring), edges (reconstructed log10 E)."""

    def __init__(self, name, P, cosz_edges=COSZ_EDGES, n_sub=N_SUB):
        self.name, self.P = name, P
        self.cosz_edges = np.asarray(cosz_edges, dtype=float)
        self.nz = len(self.cosz_edges) - 1
        self.det = F.Detector(name, P, build=False)
        self.channels = [ch for ch in P.get("channels", CHANNELS) if ch in CHANNELS]
        self.T = P["years"] * F.YR
        self.R, self.B, self.GB, self.RX, self.BX, self.edges = {}, {}, {}, {}, {}, {}
        sub = _sub_points(self.cosz_edges, n_sub)
        for ch in self.channels:
            self._build(ch, sub)

    def _build(self, ch, sub):
        d, P = self.det, self.P
        ed = F.EDGES[ch]
        nb = len(ed) - 1
        sig_lg = P["trk_sig_lgE"] if ch == "trk" else P["shw_sig_lgE"]
        bias = -0.15 if ch == "trk" else 0.0
        acc = P["trk_accept"] if ch == "trk" else P["shw_accept"]
        R = {k: np.zeros((len(F.E), nb, self.nz)) for k in KEYS}
        RX = {k: {p: np.zeros((len(F.E), nb, self.nz)) for p in ("escale", "misid")} for k in KEYS}
        B = np.zeros((nb, self.nz))
        GB = {p: np.zeros((nb, self.nz)) for p in ATM_PARAMS}
        BX = {p: np.zeros((nb, self.nz)) for p in ("escale", "misid")}
        for fl, bar in KEYS:
            comps = d.aeff(ch, fl, bar)
            tags = d.aeff_tags(ch, fl)
            assert len(tags) == len(comps)
            for (A, lgobs), (kind, mis) in zip(comps, tags):
                mig = F.gauss_migration(lgobs, ed, sig_lg, bias=bias)                   # (nE, nb)
                dmig = F.gauss_migration_dshift(lgobs, ed, sig_lg, bias=bias)
                for z, (cs, ws) in enumerate(sub):
                    for c, wc in zip(cs, ws):
                        a = acc(c)
                        if a == 0:
                            continue
                        surv = F.earth_survival(bar, F.E, c) * d.gate(ch, c, lgobs)
                        # per unit flux per sr: 2 pi dcos Z of solid angle, the exposure and the true-energy bin
                        w = 2.0 * np.pi * wc * a * self.T * (A * surv * F.DE)          # (nE,)
                        R[(fl, bar)][:, :, z] += w[:, None] * mig
                        RX[(fl, bar)]["escale"][:, :, z] += w[:, None] * dmig
                        if mis:
                            RX[(fl, bar)]["misid"][:, :, z] += w[:, None] * mig
                        m = d.bkgmult(ch, c, lgobs)
                        atm = F.atm_flux(F.ATMNAME[fl], bar, F.E, c)
                        aw = w * m                                                     # m: scalar or per energy
                        B[:, z] += (atm * aw) @ mig
                        for p in ATM_PARAMS:
                            GB[p][:, z] += (F.atm_rel(p, F.ATMNAME[fl], bar, F.E, c) * atm * aw) @ mig
                        BX["escale"][:, z] += (atm * aw) @ dmig
                        if mis:
                            BX["misid"][:, z] += (atm * aw) @ mig
        self.R[ch], self.RX[ch], self.B[ch], self.GB[ch], self.BX[ch] = R, RX, B, GB, BX
        self.edges[ch] = ed

    def counts(self, ch, flux):
        """Expected astrophysical events (nb, nz) for flux[(fl, bar)] (nE, nz): GeV^-1 cm^-2 s^-1 sr^-1 per flavor and
        kind on the true-energy grid, per cos-zenith bin."""
        return sum(np.einsum("ez,ebz->bz", flux[k], self.R[ch][k]) for k in KEYS if k in flux)

    def allsky_counts(self, ch, phi):
        """Events of an isotropic flux phi(E) per flavor and kind (nu + nubar = 2 phi... as NuFlavorSingleSource's
        allsky_rate: each kind carries half of a per-flavor nu + nubar flux), all bins: for the comparison with
        Detector.allsky_rate(flux=...) per year."""
        f = 0.5 * phi(F.E)
        return sum(np.einsum("e,ebz->", f, self.R[ch][k]) for k in KEYS) / self.P["years"]
