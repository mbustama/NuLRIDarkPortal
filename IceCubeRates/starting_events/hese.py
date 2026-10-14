"""
hese.py -- expected HESE events (cascades, tracks, double cascades) from IceCube's HESE 7.5-year release
(docs/dev/HESE_RATES_PLAN.md, Mauricio's task 1). The release's Monte Carlo, reduced by tools/hese_reduce.py (kept
outside the repository: the path, or the environment variable NULRI_HESE_MC), gives each simulated neutrino event i
its particle type, true energy and direction, and weightOverFluxOverLivetime w_i, so the expected number of events
in any bin under a flux Phi (per type: nu or nubar of one flavor, GeV^-1 cm^-2 s^-1 sr^-1) is exact:

    N = T sum_{i in bin} w_i Phi_type_i(E_i, dec_i),      sin dec = -cos zenith (South Pole),

with the release's binning: cascades and tracks in deposited energy x reconstructed zenith (10 bins uniform in
cos zenith), double cascades in deposited energy x length.

Atmospheric neutrinos (Mauricio, 2026-10-14: MCEq + Barr with the release's self-veto): our MCEq table
(IceCubeRates/data/atmospheric_mceq_southpole.npz) per event, the conventional part (nominal minus prompt) times the
release's conventional self-veto correction and the prompt part times its prompt one; the Barr derivatives move the
conventional part only. The cosmic-ray tilt moves both: until the table carries the prompt part's own derivative it
takes the conventional correction (the prompt share above 60 TeV deposited is small after the veto). Atmospheric
muons: the release's muon Monte Carlo (muonWeightOverLivetime), with a normalization.
"""
import os

import numpy as np

DATA_ENV = "NULRI_HESE_MC"
MORPHOLOGIES = ("cascades", "tracks", "double cascades")
_FLAVOR = {12: 0, 14: 1, 16: 2}
_SPECIES = {12: "nue", -12: "antinue", 14: "numu", -14: "antinumu", 16: "nutau", -16: "antinutau"}


class HESEEvents:
    """The reduced HESE Monte Carlo and data, and the per-event weighting."""

    def __init__(self, path=None):
        path = path or os.environ.get(DATA_ENV)
        if not path:
            raise ValueError(f"the reduced HESE Monte Carlo (tools/hese_reduce.py): pass its path or set {DATA_ENV}")
        with np.load(path, allow_pickle=False) as d:
            self.d = {k: d[k] for k in d.files}
        d = self.d
        self.T = float(d["livetime_s"])
        self.E_bins, self.zen_bins, self.len_bins = d["energy_bins_GeV"], d["zenith_bins_rad"], d["length_bins_m"]
        t = d["mc_type"].astype(np.int64)
        self.is_nu = np.isin(np.abs(t), (12, 14, 16))
        self.flavor = np.where(self.is_nu, np.vectorize(lambda x: _FLAVOR.get(abs(int(x)), -1))(t), -1)
        self.anti = t < 0
        self.E = d["mc_E_GeV"]
        self.cosz = d["mc_cos_zenith"]
        self.dec = np.arcsin(np.clip(-self.cosz, -1.0, 1.0))
        self.w = np.where(self.is_nu, d["mc_weight"], 0.0)
        self._index = self._bin_index(d["mc_morphology"], d["mc_E_dep_GeV"], d["mc_zenith_reco"], d["mc_length_m"])
        self._data_index = self._bin_index(d["data_morphology"], d["data_E_dep_GeV"], d["data_zenith_reco"],
                                           d["data_length_m"])

    # ------------------------------------------------------------------ binning
    @property
    def shapes(self):
        nE, nz, nl = len(self.E_bins) - 1, len(self.zen_bins) - 1, len(self.len_bins) - 1
        return {"cascades": (nz, nE), "tracks": (nz, nE), "double cascades": (nl, nE)}

    def _bin_index(self, morph, E, zen, L):
        """Flat index into the concatenated (cascades, tracks, double cascades) histograms; -1 outside."""
        sh = self.shapes
        size = [int(np.prod(sh[m])) for m in MORPHOLOGIES]
        offset = np.cumsum([0] + size[:-1])
        iE = np.digitize(E, self.E_bins) - 1
        iz = np.digitize(zen, self.zen_bins) - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            il = np.digitize(L, self.len_bins) - 1
        nE = len(self.E_bins) - 1
        out = np.full(len(E), -1, dtype=np.int64)
        for m, name in enumerate(MORPHOLOGIES):
            ix = il if m == 2 else iz
            nx = sh[name][0]
            ok = (morph == m) & (iE >= 0) & (iE < nE) & (ix >= 0) & (ix < nx)
            out[ok] = offset[m] + ix[ok] * nE + iE[ok]
        self._size = int(sum(size))
        return out

    def histogram(self, per_event):
        """Per-event expected counts -> dict morphology -> histogram (release binning)."""
        ok = self._index >= 0
        flat = np.bincount(self._index[ok], weights=np.asarray(per_event, dtype=np.float64)[ok], minlength=self._size)
        return self._split(flat)

    def data(self):
        ok = self._data_index >= 0
        return self._split(np.bincount(self._data_index[ok], minlength=self._size).astype(np.float64))

    def _split(self, flat):
        out, a = {}, 0
        for name in MORPHOLOGIES:
            n = int(np.prod(self.shapes[name]))
            out[name] = flat[a:a + n].reshape(self.shapes[name])
            a += n
        return out

    # ------------------------------------------------------------------ fluxes
    def expected(self, flux):
        """Per-event expected counts T w_i Phi(flavor_i, anti_i, E_i, dec_i) for flux(flavor (N,), anti (N,), E_GeV
        (N,), dec (N,)) -> (N,) (neutrino events only; flavor 0 e, 1 mu, 2 tau)."""
        out = np.zeros(len(self.E))
        m = self.is_nu
        out[m] = self.T * self.w[m] * flux(self.flavor[m], self.anti[m], self.E[m], self.dec[m])
        return out

    def muons(self, norm=1.0):
        """Per-event expected atmospheric muons (the release's muon Monte Carlo), times norm."""
        return norm * self.T * self.d["mc_muon_weight"]

    def atmospheric_templates(self, path=None):
        """Per-event expected atmospheric-neutrino counts with the release's self-veto: dict term -> (N,), terms
        'conv' (nominal conventional), 'prompt', and each derivative parameter of the MCEq table (Barr parameters
        and cr_dgamma, per unit of the parameter). The full expectation is conv + prompt_norm x prompt (prompt_norm
        = 1 + theta_prompt) + sum_p theta_p d_p."""
        from IceCubeRates.flux_adapters import DATA
        path = path or os.path.join(DATA, "atmospheric_mceq_southpole.npz")
        with np.load(path, allow_pickle=False) as z:
            tab = {k: z[k] for k in z.files}
        lE_t, c_t = np.log(tab["E_GeV"]), tab["cos_zenith"]
        params = [str(p) for p in tab["params"]]
        out = {k: np.zeros(len(self.E)) for k in ["conv", "prompt"] + params}
        lE = np.log(self.E)
        c = np.clip(np.abs(self.cosz), c_t[0], c_t[-1])          # up-going: the mirror production zenith
        j = np.clip(np.searchsorted(c_t, c) - 1, 0, len(c_t) - 2)
        u = (c - c_t[j]) / (c_t[j + 1] - c_t[j])
        k = np.clip(np.searchsorted(lE_t, lE) - 1, 0, len(lE_t) - 2)
        v = (lE - lE_t[k]) / (lE_t[k + 1] - lE_t[k])
        inside = (lE >= lE_t[0]) & (lE <= lE_t[-1])
        cv, pv = self.d["mc_conv_veto"], self.d["mc_prompt_veto"]
        for pdg, sp in _SPECIES.items():
            idx = np.nonzero((self.d["mc_type"] == pdg) & self.is_nu & inside)[0]
            if not len(idx):
                continue
            jj, uu, kk, vv = j[idx], u[idx], k[idx], v[idx]

            def bil(A):
                return ((1 - uu) * (1 - vv) * A[jj, kk] + uu * (1 - vv) * A[jj + 1, kk]
                        + (1 - uu) * vv * A[jj, kk + 1] + uu * vv * A[jj + 1, kk + 1])
            nom = tab["nom_" + sp]
            safe = np.maximum(nom, 1e-300)
            # the nominal flux in log, the prompt fraction and the relative derivatives linearly (as the
            # adapter and NuFlavorSingleSource interpolate them)
            phi = np.exp(bil(np.log(safe)))
            fpr = bil(tab["pr_" + sp] / safe)
            base = self.T * self.w[idx] * phi
            out["conv"][idx] = base * (1.0 - fpr) * cv[idx]
            out["prompt"][idx] = base * fpr * pv[idx]
            for p in params:
                out[p][idx] = base * bil(tab[f"d_{p}__{sp}"] / safe) * cv[idx]
        return out

    def load_detector(self, path):
        """The release's detector systematics per event (tools/hese_detector_systematics.py, its own code and
        environment): log10 of the exact weight factors on a grid of each systematic's values, per flux component."""
        with np.load(path, allow_pickle=False) as z:
            det = {k: z[k] for k in z.files}
        if len(det["E_dep_GeV"]) != len(self.E) or not np.array_equal(det["E_dep_GeV"], self.d["mc_E_dep_GeV"]):
            raise ValueError("the detector systematics are not aligned with the reduced Monte Carlo")
        self.det = det
        names = [str(n) for n in det["names"]]
        self.det_nominal = dict(zip(names, det["nominal"].astype(float)))
        self.det_sigma = dict(zip(names, det["sigma"].astype(float)))
        self.det_grid = {n: det["grid_" + n].astype(float) for n in names}

    def detector_factor(self, component, values):
        """Per-event weight factor of the release's detector systematics for component 'Astro', 'Conv' or 'Prompt' at
        the values in `values` (names dom_eff, head_on, anisotropy; absent = nominal, factor 1): the product over
        systematics of 10^(log10 w), log10 w interpolated in the systematic by a cubic through the four nearest grid
        values (exact on the grid)."""
        expo = np.zeros(len(self.E))
        for name, g in self.det_grid.items():
            if name not in values:
                continue
            x = float(values[name])
            if not g[0] - 1e-12 <= x <= g[-1] + 1e-12:
                raise ValueError(f"{name} = {x} lies outside the exported grid {g[0]}-{g[-1]}")
            i = int(np.clip(np.searchsorted(g, x) - 2, 0, len(g) - 4))
            nodes = g[i:i + 4]
            L = self.det[f"log10w_{name}_{component}"]
            for a in range(4):
                c = np.prod([(x - nodes[b]) / (nodes[a] - nodes[b]) for b in range(4) if b != a])
                if c != 0.0:
                    expo += c * L[i + a].astype(np.float64)
        return 10.0 ** expo

    def release_atmospheric(self, kpi_ratio=1.0):
        """The release's own atmospheric expectation per event (its pion, kaon and prompt fluxes with its
        self-veto corrections), for comparison: dict(conv, prompt)."""
        base = self.T * self.w
        return dict(conv=base * (self.d["mc_pion_flux"] + kpi_ratio * self.d["mc_kaon_flux"]) * self.d["mc_conv_veto"],
                    prompt=base * self.d["mc_prompt_flux"] * self.d["mc_prompt_veto"])


def sky_flux(sky, extragalactic=True, galactic=True):
    """A per-event flux(flavor, anti, E_GeV, dec) from a propagate_sky output: the band holding dec, nu or nubar,
    interpolated log-log in energy (zero outside the solver's grid); the extragalactic part in physical units
    (sky.extragalactic_norm). The Galactic flux is the band's pixel mean (the release's Monte Carlo has no azimuth)."""
    if extragalactic and "extragalactic_scale" not in sky["meta"]:
        raise ValueError("sky_flux needs the extragalactic flux in physical units: set sky.extragalactic_norm")
    lE_s = np.log(sky["E_eV"] / 1e9)
    nb, nE = sky["bands_nu"].shape[:2]
    F = {False: np.zeros((nb, nE, 3)), True: np.zeros((nb, nE, 3))}
    for anti, key in ((False, "nu"), (True, "nubar")):
        if extragalactic:
            F[anti] = F[anti] + sky["bands_" + key]
        if galactic and "galactic_bands_" + key in sky:
            F[anti] = F[anti] + sky["galactic_bands_" + key]
    edges = sky["dec_band_edges"]

    def flux(flavor, anti, E, dec):
        b = np.clip(np.searchsorted(edges, dec) - 1, 0, nb - 1)
        lE = np.log(E)
        k = np.clip(np.searchsorted(lE_s, lE) - 1, 0, nE - 2)
        t = (lE - lE_s[k]) / (lE_s[k + 1] - lE_s[k])
        out = np.zeros(len(E))
        for a in (False, True):
            m = (anti == a) & (lE >= lE_s[0]) & (lE <= lE_s[-1])
            if not m.any():
                continue
            L = np.log(np.maximum(F[a], 1e-300))
            v = (1 - t[m]) * L[b[m], k[m], flavor[m]] + t[m] * L[b[m], k[m] + 1, flavor[m]]
            out[m] = np.exp(v)
        return out
    return flux


def power_law_flux(per_type_norm, gamma, pivot_GeV=1e5):
    """flux(flavor, anti, E, dec) = per_type_norm (E / pivot)^-gamma for every type (e.g. HESE's 6.37e-18 / 6)."""
    def flux(flavor, anti, E, dec):
        return per_type_norm * (np.asarray(E) / pivot_GeV) ** (-gamma)
    return flux
