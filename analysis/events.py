"""
events.py

The event likelihoods (docs/dev/EVENT_LIKELIHOOD_PLAN.md; Mauricio, 2026-10-14): IceCube's HESE starting events (the
7.5-year release, its binning above 60 TeV deposited) and through-going tracks (IceTracks-DR2's IC79 and IC86,
reconstructed energy above 3 TeV, declination above -5 deg) against the model's flux per declination band, with the
background and detector nuisances profiled inside the term:

  HESE    mu = astro (rho_Astro) + (1 + hese_conv_norm) [conv + sum_p theta_p d_p] (rho_Conv)
               + (1 + prompt) prompt (rho_Prompt) + hese_muon_norm muons,
  DR2, c  mu_c = astro_c + (1 + dr2_norm_c) [nominal_c + sum_p theta_p d_p,c + prompt prompt_c],

with theta_p the 12 Barr parameters and cr_dgamma, and 'prompt', shared (NuFlavorSingleSource's priors); for HESE its
conventional normalization (0.4) and muon normalization (1 +- 0.5), the release's priors, and the detector systematics
dom_eff (0.99 +- 0.1), head_on (0 +- 0.5) and anisotropy (1 +- 0.2) within the exported grids (rho: the release's
detector weights); for DR2 a normalization per configuration (0.1). The astrophysical flux is the model's (its own
parameters fitted outside); DR2 uses nu_mu + nubar_mu (the release's effective areas).

-2 ln L = 2 sum [mu - n + n ln(n / mu)] over both data sets plus the nuisances' priors. The nuisances are found with
the detector weights per bin (IceCubeRates.starting_events.hese.BinDetector, exact on the grid one systematic at a
time), and the value returned is evaluated with the exact per-event weights at them: per bin the weights are not exact
away from the grid (up to 0.02 in -2 ln L at fixed nuisances), so the per-bin model only steers the profile.
"""
import os
import sys
from typing import Mapping, Optional

from types import SimpleNamespace

import numpy as np
from numba import njit
from scipy.optimize import minimize

from . import REPO_DIR

if REPO_DIR not in sys.path:                    # IceCubeRates lives at the repository's top level
    sys.path.insert(0, REPO_DIR)
from IceCubeRates import flux_adapters as FA  # noqa: E402
from IceCubeRates.northern_muons.dr2 import DR2Tracks, LinearFold  # noqa: E402
from IceCubeRates.starting_events import hese  # noqa: E402

HESE_MC_ENV, HESE_DET_ENV = "NULRI_HESE_MC", "NULRI_HESE_DETSYS"
DR2_COUNTS = os.path.join(FA.DATA, "dr2_counts.npz")
HESE_PRIORS = {"hese_conv_norm": (0.0, 0.4, (-1.0, 3.0)), "hese_muon_norm": (1.0, 0.5, (0.0, 4.0))}
DR2_NORM_PRIOR = 0.1
PROMPT_BOUNDS = (-1.0, 5.0)
DEFAULT_DR2 = dict(configs=("IC79", "IC86"), E_reco_min_log10=3.5, dec_min_deg=-5.0)
MU_FLOOR = 1e-3                   # bins with data: the deviance continued quadratically below MU_FLOOR x n
NEG_PENALTY = 1e6                 # empty bins: + NEG_PENALTY mu^2 below mu = 0


def _enabled(setting) -> bool:
    """A data set's entry in data.events: on when a mapping (an empty one: the defaults) or true; off when absent,
    false or null."""
    return isinstance(setting, Mapping) or setting is True


def _poisson(mu, n, grad=False):
    """-2 ln L / L_max = 2 sum [mu - n + n ln(n / mu)], continued smoothly (C^2) where no physical expectation goes:
    below MU_FLOOR n in bins with data by its quadratic Taylor expansion there, below 0 in empty bins with a steep
    quadratic penalty -- so a trial step into unphysical nuisances returns finite values and gradients. With grad,
    also d/dmu per bin. Compiled (2026-10-18: called ~30,000 times per chi^2 evaluation by the future detectors'
    nuisance profile, its NumPy overhead took ~40% of a proj_future point); _poisson_ref is the NumPy original, equal
    to rounding (tests/test_speedups.py)."""
    mu_a = np.asarray(mu, dtype=np.float64)
    m_c = np.ascontiguousarray(mu_a).reshape(-1)
    n_c = np.ascontiguousarray(np.broadcast_to(np.asarray(n, dtype=np.float64), mu_a.shape)).reshape(-1)
    g = np.empty_like(m_c)
    total = _poisson_nb(m_c, n_c, g)
    if not grad:
        return total
    return total, g.reshape(mu_a.shape)


@njit(cache=True)
def _poisson_nb(mu, n, g):
    """_poisson's sum over bins (its docstring), filling g with d/dmu."""
    total = 0.0
    for i in range(mu.shape[0]):
        ni = n[i]
        mi = mu[i]
        if ni > 0.0:
            mu0 = MU_FLOOR * ni
            m = mu0 if mi < mu0 else mi
            D = 2.0 * (m - ni + ni * np.log(ni / m))
            D1 = 2.0 * (1.0 - ni / m)
            D2 = 2.0 * ni / (m * m)
        else:
            mu0 = 0.0
            m = mu0 if mi < mu0 else mi
            D = 2.0 * m
            D1 = 2.0
            D2 = 2.0 * NEG_PENALTY
        dm = mi - mu0 if mi < mu0 else 0.0
        total += D + D1 * dm + 0.5 * D2 * dm * dm
        g[i] = D1 + D2 * dm
    return total


def _poisson_ref(mu, n, grad=False):
    """_poisson in NumPy (the original; tests compare the compiled one with it)."""
    mu = np.asarray(mu, dtype=np.float64)
    pos = n > 0
    mu0 = np.where(pos, MU_FLOOR * n, 0.0)
    low = mu < mu0
    m = np.where(low, mu0, mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(pos, 2.0 * (m - n + n * np.log(np.where(pos, n, 1.0) / np.where(pos, m, 1.0))), 2.0 * m)
        D1 = np.where(pos, 2.0 * (1.0 - n / np.where(pos, m, 1.0)), 2.0)
        D2 = np.where(pos, 2.0 * n / np.where(pos, m, 1.0) ** 2, 2.0 * NEG_PENALTY)
    dm = np.where(low, mu - mu0, 0.0)
    total = float(np.sum(D + D1 * dm + 0.5 * D2 * dm * dm))
    if not grad:
        return total
    return total, D1 + D2 * dm


def _deviance(mu, n):
    return _poisson(mu, n)


class _HESE:
    """HESE: the per-bin templates, the per-bin detector model, and the per-event pieces of the exact evaluation."""

    def __init__(self, E_nodes_GeV, band_edges, mc=None, detector=None, atm=None):
        mc = mc or os.environ.get(HESE_MC_ENV)
        detector = detector or os.environ.get(HESE_DET_ENV)
        if not mc or not detector:
            raise ValueError(f"the HESE likelihood needs the reduced Monte Carlo and its detector systematics: set "
                             f"data.events.hese.mc/detector or {HESE_MC_ENV}/{HESE_DET_ENV}")
        ev = hese.HESEEvents(mc)
        ev.load_detector(detector)
        self.lin = hese.LinearResponse(ev, E_nodes_GeV, band_edges, detector_maps=False)
        self.n_bins = ev._size
        self.data = ev.flat_data()
        self.names = list(ev.det_grid)
        self.grids = {s: ev.det_grid[s] for s in self.names}
        self.nominal = {s: float(ev.det_nominal[s]) for s in self.names}
        self.sigma = {s: float(ev.det_sigma[s]) for s in self.names}
        sel = self.lin.events                                           # neutrino events inside the binning
        self.bin = ev._index[sel]
        tpl = ev.atmospheric_templates(atm)
        self.atm_params = [p for p in FA.ATM_PARAMS if p != "prompt"]
        # per-event pieces of the exact evaluation (single precision for the detector weights, as exported)
        self.conv, self.prompt = tpl["conv"][sel], tpl["prompt"][sel]
        self.deriv = np.stack([tpl[p][sel] for p in self.atm_params])
        self.L = {c: {s: np.asarray(ev.det[f"log10w_{s}_{c}"][:, sel], dtype=np.float32) for s in self.names}
                  for c in ("Astro", "Conv", "Prompt")}
        mu_ev = np.nonzero((~ev.is_nu) & (ev._index >= 0))[0]
        self.muons = np.bincount(ev._index[mu_ev], weights=ev.muons()[mu_ev], minlength=self.n_bins)

        # per-bin templates and their detector models (the atmospheric ones fixed; the astrophysical one's log-ratios
        # from a reference E^-2.5 flux, the same for every flavor and band: they only steer the profile)
        def bin_detector(per_event, comp):
            H0 = np.bincount(self.bin, weights=per_event, minlength=self.n_bins)
            H_nodes = {s: np.stack([np.bincount(self.bin, weights=per_event * 10.0 ** self.L[comp][s][k].astype(np.float64),
                                                minlength=self.n_bins) for k in range(len(self.grids[s]))])
                       for s in self.names}
            return hese.BinDetector(self.grids, H0, H_nodes)
        self.bd_conv = bin_detector(self.conv, "Conv")
        self.bd_deriv = [bin_detector(d, "Conv") for d in self.deriv]
        self.bd_prompt = bin_detector(self.prompt, "Prompt")
        ref = 1e-18 * (np.asarray(E_nodes_GeV) / 1e5) ** -2.5
        ref = np.broadcast_to(ref[None, :, None], (len(band_edges) - 1, len(E_nodes_GeV), 3))
        self.bd_astro_ref = bin_detector(self.lin.per_event(ref, ref), "Astro")

    def coefficients(self, values):
        """{systematic: lagrange4's (i, c, dc)} at values (all systematics): shared by every template's rho."""
        return {s: hese.lagrange4(self.grids[s], values[s]) for s in self.names}

    def rho(self, bd, values, grad, coeffs=None):
        """exp(sum_s cubic in log) per bin and its derivatives in each systematic (values: all systematics; coeffs:
        their coefficients, computed here when not given)."""
        co = coeffs if coeffs is not None else self.coefficients(values)
        expo = np.zeros(self.n_bins)
        d = {}
        for s in self.names:
            i, c, dc = co[s]
            L = bd.logr[s][i:i + 4]
            expo += c @ L
            if grad:
                d[s] = dc @ L
        r = np.exp(expo)
        return r, d

    def factors(self, values):
        """The exact per-event detector weights of each component at values (all systematics)."""
        out = {}
        co = self.coefficients(values)
        for comp, Ls in self.L.items():
            expo = np.zeros(len(self.bin))
            for s in self.names:
                i, c, _ = co[s]
                expo += c.astype(np.float32) @ Ls[s][i:i + 4]
            out[comp] = 10.0 ** expo.astype(np.float64)
        return out


class _DR2:
    """DR2: per configuration, the linear fold of the astrophysical flux, the atmospheric templates and the counts, on
    the selected bins."""

    def __init__(self, E_nodes_GeV, band_edges, configs=DEFAULT_DR2["configs"],
                 E_reco_min_log10=DEFAULT_DR2["E_reco_min_log10"], dec_min_deg=DEFAULT_DR2["dec_min_deg"]):
        tr = DR2Tracks()
        with np.load(DR2_COUNTS, allow_pickle=False) as z:
            counts = {c: z[f"{c}_counts"] for c in configs}
            dec_c = {c: z[f"{c}_dec_edges_deg"] for c in configs}
            reco_c = z["reco_edges_log10E"]
        if not np.array_equal(reco_c, tr.reco_edges):
            raise ValueError("dr2_counts.npz and dr2_irfs.npz disagree on the reconstructed-energy bins")
        self.configs = list(configs)
        self.atm_params = [p for p in FA.ATM_PARAMS if p != "prompt"]
        self.fold, self.data, self.nom, self.prompt, self.deriv, self.select = {}, {}, {}, {}, {}, {}
        for c in self.configs:
            de = tr.dec_edges(c)
            if not np.allclose(np.degrees(de), dec_c[c], rtol=0, atol=1e-9):
                raise ValueError(f"dr2_counts.npz and dr2_irfs.npz disagree on {c}'s declination bins")
            sel = np.zeros((len(de) - 1, len(tr.reco_edges) - 1), dtype=bool)
            sel[np.degrees(de[:-1]) >= dec_min_deg - 1e-9, :] = True
            sel[:, tr.reco_edges[:-1] < E_reco_min_log10 - 1e-9] = False
            self.select[c] = sel
            self.fold[c] = LinearFold(tr, c, E_nodes_GeV, band_edges, select=sel)
            self.data[c] = counts[c][sel].astype(np.float64)

            def tmpl(term):
                return tr.expected(FA.Atmospheric(term=term), configs=[c])["per_config"][c][sel]
            self.nom[c], self.prompt[c] = tmpl("nom"), tmpl("prompt")
            self.deriv[c] = np.stack([tmpl(p) for p in self.atm_params])


class EventLikelihood:
    """-2 ln L of the event data at the model's values, profiled over the event nuisances (see the module
    docstring).

    Arguments:
      model: a FlavorModel in sky mode (its energy grid and declination bands).
      spec: the configuration's data.events, {"hese": {...} or true, "dr2": {...} or true} (either or both; HESE keys:
        mc, detector -- paths, default NULRI_HESE_MC and NULRI_HESE_DETSYS; DR2 keys: configs, E_reco_min_log10,
        dec_min_deg).
    """

    def __init__(self, model, spec: Mapping):
        spec = dict(spec)
        bad = set(spec) - {"hese", "dr2"}
        if bad or not spec:
            raise ValueError(f"data.events: use 'hese' and/or 'dr2', got {sorted(spec)}")
        self.model = model
        E, self.band_edges = model.grid()
        self.E_GeV = E / 1e9
        self.hese = self.dr2 = None
        # a data set is on when named with its settings, an empty {} meaning the defaults (until 2026-10-15 an empty
        # {} read as off: the campaign's HESE + DR2 planes had no data); off only when absent, false or null
        if _enabled(spec.get("hese")):
            h = spec["hese"] if isinstance(spec["hese"], Mapping) else {}
            bad = set(h) - {"mc", "detector", "atm"}
            if bad:
                raise ValueError(f"data.events.hese: unknown keys {sorted(bad)}")
            self.hese = _HESE(self.E_GeV, self.band_edges, **h)
        if _enabled(spec.get("dr2")):
            d = dict(DEFAULT_DR2, **(spec["dr2"] if isinstance(spec["dr2"], Mapping) else {}))
            bad = set(d) - set(DEFAULT_DR2)
            if bad:
                raise ValueError(f"data.events.dr2: unknown keys {sorted(bad)}")
            self.dr2 = _DR2(self.E_GeV, self.band_edges, **d)
        if self.hese is None and self.dr2 is None:
            # an event term without data would return chi^2 = 0 everywhere (as the campaign's first HESE + DR2
            # planes did): refuse it
            raise ValueError(f"data.events {spec}: neither HESE nor DR2 is on (use {{}} for the defaults)")
        # the nuisances: name, centre, sigma, bounds
        pri = FA.Atmospheric().priors()
        nz = [(p, 0.0, pri[p], (None, None)) for p in FA.ATM_PARAMS if p != "prompt"]   # held by their priors
        nz.append(("prompt", 0.0, pri["prompt"], PROMPT_BOUNDS))
        if self.hese is not None:
            nz += [(k, c, s, b) for k, (c, s, b) in HESE_PRIORS.items()]
            nz += [(s, self.hese.nominal[s], self.hese.sigma[s], (float(self.hese.grids[s][0]), float(self.hese.grids[s][-1])))
                   for s in self.hese.names]
        if self.dr2 is not None:
            nz += [(f"dr2_norm_{c}", 0.0, DR2_NORM_PRIOR, (-0.9, 1.0)) for c in self.dr2.configs]
        self.names = [n[0] for n in nz]
        self.centre = np.array([n[1] for n in nz])
        self.sigma = np.array([n[2] for n in nz])
        self.bounds = [n[3] for n in nz]
        self.atm_params = [p for p in FA.ATM_PARAMS if p != "prompt"]
        self._z0 = np.zeros(len(self.names))                          # warm start (units of sigma from the centre)
        self.last = {}
        self.n_calls = 0
        self.n_gn = 0
        self.n_fallback = 0
        self.gn_fail = ""

    # ------------------------------------------------------------------ the astrophysical templates
    def astro(self, values: Mapping) -> dict:
        """The model's expected astrophysical events: HESE per bin (no detector weights) and per event, DR2 per
        configuration's selected bins."""
        _, F_nu, F_nb = self.model.band_fluxes(values)
        out = {}
        if self.hese is not None:
            per_event = self.hese.lin.per_event(F_nu, F_nb)
            out["hese_events"] = per_event
            out["hese"] = np.bincount(self.hese.bin, weights=per_event, minlength=self.hese.n_bins)
        if self.dr2 is not None:
            Fmu = F_nu[:, :, 1] + F_nb[:, :, 1]
            out["dr2"] = {c: self.dr2.fold[c](Fmu) for c in self.dr2.configs}
        return out

    # ------------------------------------------------------------------ the profile
    def _unpack(self, x):
        return dict(zip(self.names, x))

    def _expected_bins(self, x, A, grad=False):
        """Expected events per bin with the per-bin detector model, and the gradient of -2 ln L + priors wrt x."""
        v = self._unpack(x)
        theta = np.array([v[p] for p in self.atm_params])
        g = np.zeros(len(x))
        f = 0.0
        idx = {n: i for i, n in enumerate(self.names)}
        if self.hese is not None:
            H = self.hese
            det = {s: v[s] for s in H.names}
            cn, mn, pr = v["hese_conv_norm"], v["hese_muon_norm"], v["prompt"]
            co = H.coefficients(det)
            rA, dA = H.rho(H.bd_astro_ref, det, grad, co)
            rC, dC = H.rho(H.bd_conv, det, grad, co)
            rP, dP = H.rho(H.bd_prompt, det, grad, co)
            rD = [H.rho(bd, det, grad, co) for bd in H.bd_deriv]
            a = A["hese"] * rA
            c = H.bd_conv.H0 * rC
            dsum = sum(t * bd.H0 * r for t, bd, (r, _) in zip(theta, H.bd_deriv, rD))
            p = H.bd_prompt.H0 * rP
            mu = a + (1.0 + cn) * (c + dsum) + (1.0 + pr) * p + mn * H.muons
            dev, w = _poisson(mu, H.data, grad=True)
            f += dev
            if grad:
                for k, (bd, (r, _)) in enumerate(zip(H.bd_deriv, rD)):
                    g[idx[self.atm_params[k]]] += w @ ((1.0 + cn) * bd.H0 * r)
                g[idx["prompt"]] += w @ p
                g[idx["hese_conv_norm"]] += w @ (c + dsum)
                g[idx["hese_muon_norm"]] += w @ H.muons
                for s in H.names:
                    dmu = a * dA[s] + (1.0 + cn) * (c * dC[s] + sum(t * bd.H0 * r * d[s] for t, bd, (r, d)
                                                                    in zip(theta, H.bd_deriv, rD))) \
                        + (1.0 + pr) * p * dP[s]
                    g[idx[s]] += w @ dmu
        if self.dr2 is not None:
            D = self.dr2
            pr = v["prompt"]
            for c in D.configs:
                nc = v[f"dr2_norm_{c}"]
                B = D.nom[c] + theta @ D.deriv[c] + pr * D.prompt[c]
                mu = A["dr2"][c] + (1.0 + nc) * B
                dev, w = _poisson(mu, D.data[c], grad=True)
                f += dev
                if grad:
                    gd = (1.0 + nc) * (D.deriv[c] @ w)
                    for k, pname in enumerate(self.atm_params):
                        g[idx[pname]] += gd[k]
                    g[idx["prompt"]] += (1.0 + nc) * (w @ D.prompt[c])
                    g[idx[f"dr2_norm_{c}"]] += w @ B
        z = (x - self.centre) / self.sigma
        f += float(z @ z)
        g += 2.0 * z / self.sigma
        return f, g

    def _gn_terms(self, x, A, hess=True):
        """-2 ln L + priors (the per-bin detector model, as _expected_bins), its gradient, and the Gauss-Newton
        Hessian J^T diag(d^2 dev / dmu^2) J + the priors' (J the expectation's exact first derivatives): positive
        definite, and exact wherever the expectation is linear in the nuisances."""
        v = self._unpack(x)
        theta = np.array([v[p] for p in self.atm_params])
        n_par = len(x)
        idx = {n: i for i, n in enumerate(self.names)}
        f, g = 0.0, np.zeros(n_par)
        Hm = np.zeros((n_par, n_par)) if hess else None

        def add(mu, n, cols):
            nonlocal f, g, Hm
            dev, r = _poisson(mu, n, grad=True)
            f += dev
            if hess:
                J = np.zeros((len(mu), n_par))
                for i, col in cols.items():
                    J[:, i] += col
                g += J.T @ r
                pos = n > 0
                m = np.where(pos, np.maximum(mu, MU_FLOOR * n), 1.0)
                h = np.where(pos, 2.0 * n / (m * m), np.where(mu < 0.0, 2.0 * NEG_PENALTY, 0.0))
                Hm += J.T @ (h[:, None] * J)
        if self.hese is not None:
            H = self.hese
            det = {s: v[s] for s in H.names}
            cn, mn, pr = v["hese_conv_norm"], v["hese_muon_norm"], v["prompt"]
            co = H.coefficients(det)
            rA, dA = H.rho(H.bd_astro_ref, det, hess, co)
            rC, dC = H.rho(H.bd_conv, det, hess, co)
            rP, dP = H.rho(H.bd_prompt, det, hess, co)
            rD = [H.rho(bd, det, hess, co) for bd in H.bd_deriv]
            a = A["hese"] * rA
            c = H.bd_conv.H0 * rC
            dk = [bd.H0 * r for bd, (r, _) in zip(H.bd_deriv, rD)]
            dsum = sum(t * d for t, d in zip(theta, dk))
            p = H.bd_prompt.H0 * rP
            mu = a + (1.0 + cn) * (c + dsum) + (1.0 + pr) * p + mn * H.muons
            cols = {}
            if hess:
                for k, pname in enumerate(self.atm_params):
                    cols[idx[pname]] = (1.0 + cn) * dk[k]
                cols[idx["prompt"]] = p
                cols[idx["hese_conv_norm"]] = c + dsum
                cols[idx["hese_muon_norm"]] = H.muons
                for s_ in H.names:
                    cols[idx[s_]] = (a * dA[s_] + (1.0 + cn) * (c * dC[s_] + sum(t * d * rd[1][s_] for t, d, rd
                                                                                    in zip(theta, dk, rD)))
                                     + (1.0 + pr) * p * dP[s_])
            add(mu, H.data, cols)
        if self.dr2 is not None:
            D = self.dr2
            pr = v["prompt"]
            for c_ in D.configs:
                nc = v[f"dr2_norm_{c_}"]
                B = D.nom[c_] + theta @ D.deriv[c_] + pr * D.prompt[c_]
                mu = A["dr2"][c_] + (1.0 + nc) * B
                cols = {}
                if hess:
                    for k, pname in enumerate(self.atm_params):
                        cols[idx[pname]] = (1.0 + nc) * D.deriv[c_][k]
                    cols[idx["prompt"]] = (1.0 + nc) * D.prompt[c_]
                    cols[idx[f"dr2_norm_{c_}"]] = B
                add(mu, D.data[c_], cols)
        z = (x - self.centre) / self.sigma
        f += float(z @ z)
        if hess:
            g += 2.0 * z / self.sigma
            Hm[np.diag_indices(n_par)] += 2.0 / self.sigma ** 2
        return f, g, Hm

    def _gauss_newton(self, A, x0, tol=1e-12, max_iter=200):
        """The profile by projected Gauss-Newton steps within the bounds (as FutureEventLikelihood._newton; the
        Hessian here Gauss-Newton's, so the convergence is linear where the expectation is not linear in the
        nuisances, but the fixed point is the stationary point all the same). Returns (x, value, iterations) once
        the free nuisances' decrement falls below tol, or None (the caller falls back to L-BFGS-B)."""
        lo = np.array([-np.inf if b[0] is None else b[0] for b in self.bounds])
        hi = np.array([np.inf if b[1] is None else b[1] for b in self.bounds])
        x = np.clip(np.array(x0, dtype=np.float64), lo, hi)
        f, g, H = self._gn_terms(x, A)
        for it in range(max_iter):
            span = 1e-12 * np.maximum(1.0, np.abs(x))
            fixed = ((x <= lo + span) & (g > 0.0)) | ((x >= hi - span) & (g < 0.0))
            free = ~fixed
            d = np.zeros_like(x)
            try:
                d[free] = np.linalg.solve(H[np.ix_(free, free)], g[free])
            except np.linalg.LinAlgError:
                self.gn_fail = "singular Hessian"
                return None
            dec = float(g[free] @ d[free])
            if not np.isfinite(dec) or dec < 0.0:
                self.gn_fail = f"no descent direction (decrement {dec:g})"
                return None
            if dec < tol:
                return x, f, it
            t = 1.0
            while True:
                xn = np.clip(x - t * d, lo, hi)
                fn = self._gn_terms(xn, A, hess=False)[0]
                if fn <= f - 1e-4 * float(g @ (x - xn)):
                    break
                t *= 0.5
                if t < 1e-6:
                    # typically a detector systematic whose profile minimum sits on a node of the release's
                    # four-point interpolation (a kink: continuous, not differentiable), which the quadratic model
                    # cannot see across; L-BFGS-B takes over (it lands on the node, as before 2026-10-15)
                    self.gn_fail = f"line search failed (decrement {dec:g})"
                    return None
            x = xn
            f, g, H = self._gn_terms(x, A)
        self.gn_fail = f"no convergence in {max_iter} steps (decrement {dec:g})"
        return None

    def profile(self, A):
        """Minimize -2 ln L + priors over the nuisances (per-bin detector model) for the astrophysical templates A:
        projected Gauss-Newton steps from the last profile's nuisances (2026-10-15), L-BFGS-B if they ever stall;
        returns the nuisances."""
        out = self._gauss_newton(A, self.centre + self.sigma * self._z0)
        if out is not None:
            x, f, it = out
            self.n_gn += 1
            self._z0 = (x - self.centre) / self.sigma
            return x, SimpleNamespace(fun=f, x=self._z0.copy(), success=True, nit=it, converged=True,
                                      message="gauss-newton")
        self.n_fallback += 1
        return self.profile_lbfgsb(A)

    def profile_lbfgsb(self, A):
        """Minimize -2 ln L + priors over the nuisances (per-bin detector model) for the astrophysical templates A by
        L-BFGS-B (the method until 2026-10-15); returns the nuisances."""
        # bounds in units of sigma; the Barr parameters and cr_dgamma are unbounded (an L-BFGS-B problem bounded on
        # every side takes a full, unnormalized first step, which from a warm start landed many sigma away)
        zb = [(None if lo is None else (lo - c) / s, None if hi is None else (hi - c) / s)
              for (lo, hi), c, s in zip(self.bounds, self.centre, self.sigma)]

        def fun(z):
            x = self.centre + self.sigma * z
            f, g = self._expected_bins(x, A, grad=True)
            return f, g * self.sigma
        z0 = np.array([min(max(z, -np.inf if lo is None else lo), np.inf if hi is None else hi)
                       for z, (lo, hi) in zip(self._z0, zb)])
        opts = dict(maxiter=3000, ftol=1e-15, gtol=1e-9)
        r = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=zb, options=opts)
        for _ in range(2):              # at these tolerances a line search can stop at rounding level: judge by the
            if self._stationary(r, zb):   # projected gradient, and restart where it is genuinely short
                break
            r = minimize(fun, r.x, jac=True, method="L-BFGS-B", bounds=zb, options=opts)
        r.converged = self._stationary(r, zb)
        self._z0 = np.array(r.x)
        return self.centre + self.sigma * r.x, r

    @staticmethod
    def _stationary(r, zb, tol=1e-5):
        """The projected gradient (sigma units) below tol: zero where a bound holds the nuisance back."""
        g = np.array(r.jac, dtype=float)
        for i, (lo, hi) in enumerate(zb):
            if (lo is not None and r.x[i] <= lo + 1e-9 and g[i] > 0) or (hi is not None and r.x[i] >= hi - 1e-9 and g[i] < 0):
                g[i] = 0.0
        return bool(np.max(np.abs(g)) < tol)

    def exact(self, x, A) -> dict:
        """-2 ln L at nuisances x with the exact per-event detector weights (HESE) -- the value reported -- plus the
        priors; per data set."""
        v = self._unpack(x)
        theta = np.array([v[p] for p in self.atm_params])
        out = dict(hese=0.0, dr2=0.0)
        if self.hese is not None:
            H = self.hese
            f = H.factors({s: v[s] for s in H.names})
            per = (A["hese_events"] * f["Astro"]
                   + (1.0 + v["hese_conv_norm"]) * (H.conv + theta @ H.deriv) * f["Conv"]
                   + (1.0 + v["prompt"]) * H.prompt * f["Prompt"])
            mu = np.bincount(H.bin, weights=per, minlength=H.n_bins) + v["hese_muon_norm"] * H.muons
            out["hese"] = _deviance(mu, H.data)
            out["hese_mu"] = mu
        if self.dr2 is not None:
            D = self.dr2
            out["dr2_mu"] = {}
            for c in D.configs:
                B = D.nom[c] + theta @ D.deriv[c] + v["prompt"] * D.prompt[c]
                mu = A["dr2"][c] + (1.0 + v[f"dr2_norm_{c}"]) * B
                out["dr2"] += _deviance(mu, D.data[c])
                out["dr2_mu"][c] = mu
        z = (x - self.centre) / self.sigma
        out["priors"] = float(z @ z)
        out["total"] = out["hese"] + out["dr2"] + out["priors"]
        return out

    def chi2(self, values: Mapping) -> float:
        """The event term of chi^2 at the model's values: profiled over the nuisances, evaluated exactly."""
        self.n_calls += 1
        A = self.astro(values)
        x, r = self.profile(A)
        ex = self.exact(x, A)
        self.last = dict(nuisances=self._unpack(x), hese=ex["hese"], dr2=ex["dr2"], priors=ex["priors"],
                         profile_value=float(r.fun), profile_converged=bool(r.converged), profile_nit=int(r.nit),
                         profile_message=str(r.message))
        return ex["total"]

    # ------------------------------------------------------------------ the exact gradient (2026-10-18)
    def _astro_from(self, F_nu, F_nb) -> dict:
        """astro() from given band fluxes (linear in them)."""
        out = {}
        if self.hese is not None:
            per_event = self.hese.lin.per_event(F_nu, F_nb)
            out["hese_events"] = per_event
            out["hese"] = np.bincount(self.hese.bin, weights=per_event, minlength=self.hese.n_bins)
        if self.dr2 is not None:
            Fmu = F_nu[:, :, 1] + F_nb[:, :, 1]
            out["dr2"] = {c: self.dr2.fold[c](Fmu) for c in self.dr2.configs}
        return out

    def _exact_grads(self, x, A):
        """At nuisances x: d(exact)/d(HESE per-event astro), d(exact)/d(DR2 astro per configuration) and d(exact)/dx
        -- exact() differentiated (the per-event detector factors through lagrange4's derivatives)."""
        v = self._unpack(x)
        idx = {n: i for i, n in enumerate(self.names)}
        theta = np.array([v[p] for p in self.atm_params])
        gx = np.zeros(len(x))
        g_ev, g_dr2 = None, {}
        if self.hese is not None:
            H = self.hese
            det = {s: v[s] for s in H.names}
            co = H.coefficients(det)
            f = H.factors(det)
            atm = H.conv + theta @ H.deriv
            per = A["hese_events"] * f["Astro"] + (1.0 + v["hese_conv_norm"]) * atm * f["Conv"] \
                + (1.0 + v["prompt"]) * H.prompt * f["Prompt"]
            mu = np.bincount(H.bin, weights=per, minlength=H.n_bins) + v["hese_muon_norm"] * H.muons
            _, r = _poisson(mu, H.data, grad=True)
            re = r[H.bin]                                         # per event
            g_ev = re * f["Astro"]
            gx[idx["hese_conv_norm"]] += re @ (atm * f["Conv"])
            for k, p in enumerate(self.atm_params):
                gx[idx[p]] += (1.0 + v["hese_conv_norm"]) * (re @ (H.deriv[k] * f["Conv"]))
            gx[idx["prompt"]] += re @ (H.prompt * f["Prompt"])
            gx[idx["hese_muon_norm"]] += r @ H.muons
            ln10 = np.log(10.0)
            for s in H.names:
                i, c, dc = co[s]
                dlog = {comp: ln10 * (dc.astype(np.float32) @ H.L[comp][s][i:i + 4]).astype(np.float64)
                        for comp in ("Astro", "Conv", "Prompt")}
                dper = A["hese_events"] * f["Astro"] * dlog["Astro"] \
                    + (1.0 + v["hese_conv_norm"]) * atm * f["Conv"] * dlog["Conv"] \
                    + (1.0 + v["prompt"]) * H.prompt * f["Prompt"] * dlog["Prompt"]
                gx[idx[s]] += re @ dper
        if self.dr2 is not None:
            D = self.dr2
            for c in D.configs:
                nc = v[f"dr2_norm_{c}"]
                B = D.nom[c] + theta @ D.deriv[c] + v["prompt"] * D.prompt[c]
                mu = A["dr2"][c] + (1.0 + nc) * B
                _, r = _poisson(mu, D.data[c], grad=True)
                g_dr2[c] = r
                gx[idx[f"dr2_norm_{c}"]] += r @ B
                gx[[idx[p] for p in self.atm_params]] += (1.0 + nc) * (D.deriv[c] @ r)
                gx[idx["prompt"]] += (1.0 + nc) * (r @ D.prompt[c])
        gx += 2.0 * (x - self.centre) / self.sigma ** 2
        return g_ev, g_dr2, gx

    def _profile_shift(self, x, A, gx):
        """The nuisances follow the astrophysical expectation (the profile is the per-bin model's minimum, at which the
        exact evaluation's gradient gx is not zero): by the implicit function theorem the exact value's derivative in
        the expectation gains -(d grad_x f_bin / dA)^T H^-1 gx (H the per-bin model's Gauss-Newton Hessian, the
        nuisances held at a bound left out). Returns that term per HESE bin (per-bin expectation) and per DR2 bin."""
        lo = np.array([-np.inf if b[0] is None else b[0] for b in self.bounds])
        hi = np.array([np.inf if b[1] is None else b[1] for b in self.bounds])
        f, g, Hm = self._gn_terms(x, A)
        span = 1e-12 * np.maximum(1.0, np.abs(x))
        free = ~(((x <= lo + span) & (g > 0.0)) | ((x >= hi - span) & (g < 0.0)))
        vv = np.zeros(len(x))
        try:
            vv[free] = np.linalg.solve(Hm[np.ix_(free, free)], gx[free])
        except np.linalg.LinAlgError:
            return None, {}
        v = self._unpack(x)
        idx = {n: i for i, n in enumerate(self.names)}
        theta = np.array([v[p] for p in self.atm_params])
        vt = np.array([vv[idx[p]] for p in self.atm_params])
        c_hese, c_dr2 = None, {}
        if self.hese is not None:
            H = self.hese
            det = {s: v[s] for s in H.names}
            cn, mn, pr = v["hese_conv_norm"], v["hese_muon_norm"], v["prompt"]
            co = H.coefficients(det)
            rA, dA = H.rho(H.bd_astro_ref, det, True, co)
            rC, dC = H.rho(H.bd_conv, det, True, co)
            rP, dP = H.rho(H.bd_prompt, det, True, co)
            rD = [H.rho(bd, det, True, co) for bd in H.bd_deriv]
            a = A["hese"] * rA
            c = H.bd_conv.H0 * rC
            dk = [bd.H0 * rr for bd, (rr, _) in zip(H.bd_deriv, rD)]
            dsum = sum(t * d for t, d in zip(theta, dk))
            p = H.bd_prompt.H0 * rP
            mu = a + (1.0 + cn) * (c + dsum) + (1.0 + pr) * p + mn * H.muons
            _, r = _poisson(mu, H.data, grad=True)
            pos = H.data > 0
            m = np.where(pos, np.maximum(mu, MU_FLOOR * H.data), 1.0)
            h = np.where(pos, 2.0 * H.data / (m * m), np.where(mu < 0.0, 2.0 * NEG_PENALTY, 0.0))
            Jv = (1.0 + cn) * sum(vt[k] * dk[k] for k in range(len(dk))) + vv[idx["prompt"]] * p \
                + vv[idx["hese_conv_norm"]] * (c + dsum) + vv[idx["hese_muon_norm"]] * H.muons
            dA_v = np.zeros(H.n_bins)
            for s in H.names:
                dmu = a * dA[s] + (1.0 + cn) * (c * dC[s] + sum(t * d * rd[1][s] for t, d, rd in zip(theta, dk, rD))) \
                    + (1.0 + pr) * p * dP[s]
                Jv += vv[idx[s]] * dmu
                dA_v += vv[idx[s]] * rA * dA[s]
            c_hese = -(h * rA * Jv + r * dA_v)
        if self.dr2 is not None:
            D = self.dr2
            pr = v["prompt"]
            for c_ in D.configs:
                nc = v[f"dr2_norm_{c_}"]
                B = D.nom[c_] + theta @ D.deriv[c_] + pr * D.prompt[c_]
                mu = A["dr2"][c_] + (1.0 + nc) * B
                pos = D.data[c_] > 0
                m = np.where(pos, np.maximum(mu, MU_FLOOR * D.data[c_]), 1.0)
                h = np.where(pos, 2.0 * D.data[c_] / (m * m), np.where(mu < 0.0, 2.0 * NEG_PENALTY, 0.0))
                Jv = (1.0 + nc) * (vt @ D.deriv[c_] + vv[idx["prompt"]] * D.prompt[c_]) + vv[idx[f"dr2_norm_{c_}"]] * B
                c_dr2[c_] = -h * Jv
        return c_hese, c_dr2

    def chi2_and_grad(self, values: Mapping, names):
        """chi2(values) (the same value) and its gradient in `names` (the model's gradient_names()): the exact
        evaluation's derivative in the astrophysical expectations at the profiled nuisances, plus the nuisances'
        response through the per-bin profile (_profile_shift), through the linear maps from the band fluxes to the
        expectations (HESE per event, DR2 per bin) to the model's closed-form flux derivatives
        (FlavorModel.band_flux_jacobian)."""
        self.n_calls += 1
        names = list(names)
        _, F_nu, F_nb, dF = self.model.band_flux_jacobian(values, names)
        A = self._astro_from(F_nu, F_nb)
        x, r = self.profile(A)
        ex = self.exact(x, A)
        self.last = dict(nuisances=self._unpack(x), hese=ex["hese"], dr2=ex["dr2"], priors=ex["priors"],
                         profile_value=float(r.fun), profile_converged=bool(r.converged), profile_nit=int(r.nit),
                         profile_message=str(r.message))
        g_ev, g_dr2, gx = self._exact_grads(x, A)
        c_hese, c_dr2 = self._profile_shift(x, A, gx)
        if self.hese is not None:
            G_ev = g_ev + (0.0 if c_hese is None else c_hese[self.hese.bin])
        grad = np.zeros(len(names))
        for j, n in enumerate(names):
            d_nu, d_nb = dF[n]
            if self.hese is not None:
                grad[j] += G_ev @ self.hese.lin.per_event(d_nu, d_nb)
            if self.dr2 is not None:
                dmu = d_nu[:, :, 1] + d_nb[:, :, 1]
                for c in self.dr2.configs:
                    grad[j] += (g_dr2[c] + c_dr2.get(c, 0.0)) @ self.dr2.fold[c](dmu)
        return float(ex["total"]), grad

    # ------------------------------------------------------------------ Asimov data
    def set_asimov(self, values: Mapping, nuisances: Optional[Mapping] = None) -> None:
        """Replace the data by the expectation at the model's values and the nuisances (default: their centres), with
        the exact detector weights."""
        x = self.centre.copy()
        for k, val in (nuisances or {}).items():
            x[self.names.index(k)] = float(val)
        ex = self.exact(x, self.astro(values))
        if self.hese is not None:
            self.hese.data = ex["hese_mu"].copy()
        if self.dr2 is not None:
            for c in self.dr2.configs:
                self.dr2.data[c] = ex["dr2_mu"][c].copy()
        self._z0 = (x - self.centre) / self.sigma
