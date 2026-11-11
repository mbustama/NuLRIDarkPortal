"""
future_events.py -- the future telescopes' event likelihood for the forecasts, on NuFlavorSingleSource's Plan A
detector models (audit finding C1, 2026-11-11; docs/dev/C1_FORECAST_PORT.md). Replaces the 2026-10-15 version, which
used the models that project withdrew.

The responses (IceCubeRates/data/future_planA/<card>.pkl, from tools/build_forecast_planA.py and
tools/assemble_forecast_planA.py on Tycho, Plan A's final code unchanged) give, per card, topology and declination band
of the solver's sky (20 bands, equal in sin dec), events per live year per unit flux integral per sr over the card's
true-energy bins, per flavor (nu + nubar, the cards' average), the Glashow piece apart, and Plan A's nuisance templates
and priors. The card carries its site, the Earth's absorption by local zenith and the rotation through the day; the
input is the flux per declination band before the Earth (FlavorModel's sky bands; at the South Pole the solver's
crossing of the Earth, which changes nothing for the dark portal and the secret interactions, C2).

Blocks (his choice, 2026-11-11): tracks per declination band and measured energy; showers and double cascades in
measured energy only (their bands summed). Per block k

  mu_k = s_k + B_k + sum_p x_p (S_k,p + BX_k,p)

s the astrophysical events of the model's flux (the signal), B Plan A's background (atmosphere, muons, published
backgrounds; its diffuse flux removed: it is our signal), x_p each nuisance's deviation from its centre in its prior's
units, S and BX its templates on the signal and the background (Plan A's blocks.assemble: efficiencies, energy scales,
misidentification, the energy model's deposits, the atmosphere's pieces). Pointing nuisances act within a band only
and are left out. The Glashow piece follows the nubar_e flux: twice its share of the card's nu_e + nubar_e response.
Live time: 10 years per detector (Plan A's D105); 'All8' adds IceCube's starting events over Gen2's years (its IceCube
strings, Plan A's D49). -2 ln L = the Poisson deviance (continued smoothly below a floor, as the other event terms)
plus the Gaussian priors; the data of a forecast: the Asimov expectation at the configuration's asimov values. The
expectation is linear in every nuisance: the profile takes projected Newton steps (exact Hessian) within the bounds.
"""
import os
import pickle
from types import SimpleNamespace
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from . import REPO_DIR
from .events import MU_FLOOR, NEG_PENALTY, _poisson

DATA_DIR = os.environ.get("NULRI_FORECAST_BLOCKS", os.path.join(REPO_DIR, "IceCubeRates", "data", "future_planA"))
CARDS = ("icecube", "arca", "gvd", "pone", "trident", "neon", "hunt", "gen2")
YEARS = 10.0
FLAVORS = ("e", "mu", "tau")
DROP_KINDS = ("pointing", "dnn")
GRADIENT_TOL = 1e-20
N_SUB = 8                    # sub-points per true-energy bin for the flux integrals (log-midpoint rule)
ALIASES = {"All8_base": "All8"}


def config_entries(config: str):
    """[(card, topologies or None for all, label)] of a detector configuration: 'All8' (Plan A's eight telescopes and
    Gen2's IceCube strings), 'Current' (IceCube, ARCA, GVD), 'IceCube', or one card."""
    config = ALIASES.get(config, config)
    if config == "All8":
        return [(c, None, c) for c in CARDS] + [("icecube", ("shw", "dc"), "gen2_ic")]
    if config == "Current":
        return [(c, None, c) for c in ("icecube", "arca", "gvd")]
    if config.lower() in CARDS:
        return [(config.lower(), None, config.lower())]
    raise ValueError(f"unknown future-detector configuration {config!r}; known: All8, Current, {', '.join(CARDS)}")


def config_sites(config: str):
    """The latitudes the model must provide its own Galactic bands for: none (the cards take the sky's declination
    bands and carry their sites)."""
    config_entries(config)
    return []


def load_card(card: str) -> dict:
    with open(os.path.join(DATA_DIR, f"{card}.pkl"), "rb") as fh:
        return pickle.load(fh)


def _flux_points(lgt_edges, n_sub=N_SUB):
    """(lg points, W (n_true, n_points)): bin integrals of a flux known at the points, phi_i = W[i] @ Phi(points)."""
    lo, hi = np.asarray(lgt_edges[:-1]), np.asarray(lgt_edges[1:])
    u = (np.arange(n_sub) + 0.5) / n_sub
    lg = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    w = np.log(10.0) * 10.0 ** lg * ((hi - lo) / n_sub)[:, None]
    n = len(lo)
    W = np.zeros((n, n * n_sub))
    for i in range(n):
        W[i, i * n_sub:(i + 1) * n_sub] = w[i]
    return lg.ravel(), W


class FutureEventLikelihood:
    """-2 ln L of the future telescopes' events at the model's values, profiled over their nuisances (module docstring).

    Arguments: model -- a FlavorModel in sky mode; spec -- {"config": "All8"}."""

    def __init__(self, model, spec: Mapping):
        spec = dict(spec)
        bad = set(spec) - {"config"}
        if bad:
            raise ValueError(f"data.events.future: unknown keys {sorted(bad)}; known: config")
        self.config = ALIASES.get(str(spec.get("config", "All8")), str(spec.get("config", "All8")))
        self.model = model
        E_eV, band_edges = model.grid()
        self._lgE_model = np.log10(np.asarray(E_eV, dtype=np.float64) / 1e9)
        sin_model = np.sin(np.asarray(band_edges, dtype=np.float64))
        cards = {}
        for card, _, _ in config_entries(self.config):
            if card not in cards:
                cards[card] = load_card(card)
                if not np.allclose(cards[card]["sin_edges"], sin_model, atol=1e-9):
                    raise ValueError(f"{card}: the responses' declination bands are not the model's")
        self.nb = len(sin_model) - 1
        # the blocks: (label, topology, band or None) with their per-point responses
        self.blocks, priors = [], {}
        self._pts = {}                                   # (card, topo) -> (lg points, inside mask, gather j, t)
        for card, topos, label in config_entries(self.config):
            d = cards[card]
            priors.update(d["priors"])
            for topo, rec in d["topos"].items():
                if topos is not None and topo not in topos:
                    continue
                key = (card, topo)
                if key not in self._pts:
                    lg, W = _flux_points(rec["lgt_edges"])
                    self._pts[key] = self._gather(lg) + (W,)
                W = self._pts[key][-1]
                bands = [(b, e) for b, e in enumerate(rec["bands"]) if e is not None]
                groups = [bands] if rec["energy_only"] else [[be] for be in bands]
                for grp in groups:
                    blk = self._block(label, topo, key, W, grp, rec["energy_only"])
                    if blk is not None:
                        self.blocks.append(blk)
        # the nuisances
        names = sorted({p for b in self.blocks for p in b["cols"]})
        self.names = names
        pr = [priors[n] for n in names]
        self.gauss = np.array([p["kind"] == "gauss" for p in pr])
        self.centre = np.zeros(len(names))                # deviations from the prior's centre
        self.width = np.array([p["width"] if p["kind"] == "gauss" else 1.0 for p in pr])
        self._lo = np.array([p["lo"] for p in pr])
        self._hi = np.array([p["hi"] for p in pr])
        self.prior_centre = {n: p["centre"] for n, p in zip(names, pr)}
        idx = {n: i for i, n in enumerate(names)}
        for b in self.blocks:
            b["cidx"] = np.array([idx[p] for p in b["cols"]], dtype=np.int64)
        self.data = None
        self._x0 = np.zeros(len(names))
        self.last = {}
        self.n_calls = self.n_newton = self.n_fallback = 0
        self.newton_fail = ""

    # ---------------------------------------------------------------- set-up
    def _gather(self, lg):
        """Linear interpolation in log E onto the model's grid as gather weights; zero outside it."""
        xp = self._lgE_model
        inside = (lg >= xp[0] - 1e-9) & (lg <= xp[-1] + 1e-9)
        lgc = np.clip(lg, xp[0], xp[-1])
        j = np.clip(np.searchsorted(xp, lgc, side="right") - 1, 0, len(xp) - 2)
        return lg, inside, j, (lgc - xp[j]) / (xp[j + 1] - xp[j])

    def _block(self, label, topo, key, W, grp, energy_only):
        """One block: its signal responses per (band, flavor) and Glashow per band, as matrices on the flux points
        (M = W^T R, n_points x n_bins), for the nominal and each signal template; its background and templates."""
        T = YEARS
        sig = {}                                   # name ('' nominal or a nuisance) -> {(b, f or 'gr'): M}
        bkg, bx = 0.0, {}
        for b, e in grp:
            kinds = e["kinds"]
            gr = e.get("Rb_gr")
            for name, R, Rg in [("", e["Rb"], gr)] + [(p, e["RXb"][p], (e.get("RXb_gr") or {}).get(p))
                                                        for p in e["RXb"] if kinds.get(p) not in DROP_KINDS]:
                dst = sig.setdefault(name, {})
                for f in FLAVORS:
                    Rf = R[f] - (Rg[f] if Rg is not None else 0.0)
                    dst[(b, f)] = T * (W.T @ Rf)
                if Rg is not None:
                    dst[(b, "gr")] = T * (W.T @ Rg["e"])
            bkg = bkg + T * e["B_total"]
            for p, v in e["BX"].items():
                if p.startswith("astro_") or kinds.get(p) in DROP_KINDS:
                    continue
                bx[p] = bx.get(p, 0.0) + T * np.asarray(v, dtype=np.float64)
        n_bins = len(np.atleast_1d(bkg))
        if not any(np.any(M) for M in sig[""].values()) and not np.any(bkg):
            return None
        cols = sorted(set(sig) - {""} | set(bx))
        bx = {p: (np.asarray(bx[p], dtype=np.float64) if p in bx else np.zeros(n_bins)) for p in cols}
        return dict(label=label, topo=topo, key=key, band=None if energy_only else grp[0][0], sig=sig,
                    B=np.asarray(bkg, dtype=np.float64), BX=bx, cols=cols)

    # ---------------------------------------------------------------- the astrophysical events
    def _fluxes(self, values, names=None):
        """Per declination band (south to north), the nu and nubar fluxes on the model's grid (and derivatives)."""
        if names is None:
            _, F_nu, F_nb = self.model.site_band_fluxes(values, -90.0)
            return F_nu[::-1], F_nb[::-1], None
        _, F_nu, F_nb, dF = self.model.site_band_flux_jacobian(values, -90.0, names)
        return F_nu[::-1], F_nb[::-1], {n: (dF[n][0][::-1], dF[n][1][::-1]) for n in names}

    def _phi(self, key, F):
        """F (nb, N_E, 3) -> (nb, n_points, 3) at a block's flux points, log-log, zero outside the model's grid."""
        _, inside, j, t = self._pts[key][:4]
        L = np.log(np.maximum(F, 1e-300))
        a = L[:, j, :]
        return np.exp(a + t[None, :, None] * (L[:, j + 1, :] - a)) * inside[None, :, None]

    def _point_fluxes(self, F_nu, F_nb):
        """{key: (Phi (nb, n_pts, 3) nu + nubar, Phi_gr (nb, n_pts) = 2 nubar_e)}."""
        out = {}
        for key in self._pts:
            pn, pb = self._phi(key, F_nu), self._phi(key, F_nb)
            out[key] = (pn + pb, 2.0 * pb[:, :, 0])
        return out

    def _signals(self, P):
        """Per block: {'': nominal signal, p: signal template} from the point fluxes P."""
        S = []
        for blk in self.blocks:
            Phi, Pgr = P[blk["key"]]
            out = {}
            for name, Ms in blk["sig"].items():
                v = 0.0
                for (b, f), M in Ms.items():
                    v = v + (Pgr[b] if f == "gr" else Phi[b, :, FLAVORS.index(f)]) @ M
                out[name] = np.asarray(v, dtype=np.float64)
            S.append(out)
        return S

    def signals(self, values):
        F_nu, F_nb, _ = self._fluxes(values)
        return self._signals(self._point_fluxes(F_nu, F_nb))

    # ---------------------------------------------------------------- expectation and profile
    def _design(self, S):
        A = []
        for blk, s in zip(self.blocks, S):
            cols = [s.get(p, 0.0) + blk["BX"][p] for p in blk["cols"]]
            A.append(np.ascontiguousarray(np.stack(cols, axis=1)) if cols else np.zeros((len(blk["B"]), 0)))
        return A

    def _mu(self, x, S, A=None):
        A = self._design(S) if A is None else A
        return [s[""] + blk["B"] + Ak @ x[blk["cidx"]] for blk, s, Ak in zip(self.blocks, S, A)]

    def _fgh(self, x, A, S, data, hess=True):
        n_par = len(x)
        f, g = 0.0, np.zeros(n_par)
        H = np.zeros((n_par, n_par)) if hess else None
        for blk, Ak, s, n in zip(self.blocks, A, S, data):
            c = blk["cidx"]
            mu = s[""] + blk["B"] + Ak @ x[c]
            dev, r = _poisson(mu, n, grad=True)
            f += dev
            if hess:
                g[c] += Ak.T @ r
                pos = n > 0
                m = np.where(pos, np.maximum(mu, MU_FLOOR * n), 1.0)
                h = np.where(pos, 2.0 * n / (m * m), np.where(mu < 0.0, 2.0 * NEG_PENALTY, 0.0))
                H[np.ix_(c, c)] += Ak.T @ (h[:, None] * Ak)
        z = np.where(self.gauss, x / self.width, 0.0)
        f += float(z @ z)
        if hess:
            g += np.where(self.gauss, 2.0 * z / self.width, 0.0)
            H[np.diag_indices(n_par)] += np.where(self.gauss, 2.0 / self.width ** 2, 0.0)
        return f, g, H

    def _newton(self, S, data, x0, tol=1e-12, max_iter=100):
        """Projected Newton steps within the bounds (as the 2026-10-15 version): (x, chi^2, iterations) or None."""
        A = self._design(S)
        lo, hi = self._lo, self._hi
        x = np.clip(np.array(x0, dtype=np.float64), lo, hi)
        f, g, H = self._fgh(x, A, S, data)
        prev_dec = np.inf
        for it in range(max_iter):
            span = 1e-12 * np.maximum(1.0, np.abs(x))
            fixed = ((x <= lo + span) & (g > 0.0)) | ((x >= hi - span) & (g < 0.0))
            free = ~fixed
            d = np.zeros_like(x)
            try:
                d[free] = np.linalg.solve(H[np.ix_(free, free)], g[free])
            except np.linalg.LinAlgError:
                self.newton_fail = "singular Hessian"
                return None
            dec = float(g[free] @ d[free])
            if not np.isfinite(dec) or dec < 0.0:
                self.newton_fail = f"no descent direction (decrement {dec:g})"
                return None
            if dec < tol or (dec < 1e-9 and it > 0 and dec >= 0.5 * prev_dec):
                return x, f, it
            prev_dec = dec
            t = 1.0
            while True:
                xn = np.clip(x - t * d, lo, hi)
                fn = self._fgh(xn, A, S, data, hess=False)[0]
                if fn <= f - 1e-4 * float(g @ (x - xn)):
                    break
                t *= 0.5
                if t < 1e-12:
                    if dec < 1e-6:
                        return x, f, it
                    self.newton_fail = f"line search failed (decrement {dec:g})"
                    return None
            x = xn
            f, g, H = self._fgh(x, A, S, data)
        self.newton_fail = f"no convergence in {max_iter} steps"
        return None

    def profile(self, S, data, tol=1e-12):
        out = self._newton(S, data, self._x0, tol=tol)
        if out is not None:
            x, f, it = out
            self.n_newton += 1
            self._x0 = x.copy()
            return x, SimpleNamespace(fun=f, x=x, success=True, nit=it)
        self.n_fallback += 1
        A = self._design(S)

        def fun(x):
            f, g, _ = self._fgh(x, A, S, data, hess=True)
            return f, g
        r = minimize(fun, np.clip(self._x0, self._lo, self._hi), jac=True, method="L-BFGS-B",
                     bounds=list(zip(self._lo, self._hi)), options=dict(maxiter=5000, ftol=1e-15, gtol=1e-9))
        self._x0 = np.array(r.x)
        return np.array(r.x), r

    def _need_data(self):
        if self.data is None:
            raise ValueError("the future detectors' likelihood has no data: a projection needs Asimov data "
                             "(set_asimov, i.e. the analysis' --asimov or data_centre)")

    def chi2(self, values: Mapping) -> float:
        self._need_data()
        self.n_calls += 1
        x, r = self.profile(self.signals(values), self.data)
        self.last = dict(nuisances=dict(zip(self.names, x.tolist())), value=float(r.fun), converged=bool(r.success),
                         nit=int(r.nit))
        return float(r.fun)

    def chi2_and_grad(self, values: Mapping, names):
        """chi2(values) and its gradient in `names`: the deviance's derivative through the signal at the profiled
        nuisances (Danskin: the profile's minimum is unique, the problem strictly convex), contracted with the
        model's flux derivatives (site_band_flux_jacobian at the South Pole: the sky's bands)."""
        self._need_data()
        self.n_calls += 1
        names = list(names)
        F_nu, F_nb, dF = self._fluxes(values, names)
        P = self._point_fluxes(F_nu, F_nb)
        S = self._signals(P)
        x, r = self.profile(S, self.data, tol=GRADIENT_TOL)
        self.last = dict(nuisances=dict(zip(self.names, x.tolist())), value=float(r.fun), converged=bool(r.success),
                         nit=int(r.nit))
        # d chi2 / d Phi at the points, per key: Q (nb, n_pts, 3) for nu + nubar, Qgr (nb, n_pts) for 2 nubar_e
        Q = {k: (np.zeros(P[k][0].shape), np.zeros(P[k][1].shape)) for k in P}
        for blk, s, mu, n in zip(self.blocks, S, self._mu(x, S), self.data):
            _, res = _poisson(mu, n, grad=True)
            coef = {"": 1.0}
            coef.update({p: x[i] for p, i in zip(blk["cols"], blk["cidx"])})
            q, qg = Q[blk["key"]]
            for name, Ms in blk["sig"].items():
                c = coef.get(name, 0.0)
                if c == 0.0:
                    continue
                for (b, f), M in Ms.items():
                    v = c * (M @ res)
                    if f == "gr":
                        qg[b] += v
                    else:
                        q[b, :, FLAVORS.index(f)] += v
        grad = np.zeros(len(names))
        for key, (q, qg) in Q.items():
            pn, pb = self._phi(key, F_nu), self._phi(key, F_nb)
            _, inside, j, t = self._pts[key][:4]

            def dphi(F, dFn, phi):
                pos = F > 1e-300
                rel = np.where(pos, dFn / np.where(pos, F, 1.0), 0.0)
                a = rel[:, j, :]
                return phi * (a + t[None, :, None] * (rel[:, j + 1, :] - a))
            for i, nme in enumerate(names):
                dn, db = dphi(F_nu, dF[nme][0], pn), dphi(F_nb, dF[nme][1], pb)
                grad[i] += float(np.sum(q * (dn + db))) + float(np.sum(qg * 2.0 * db[:, :, 0]))
        return float(r.fun), grad

    def set_asimov(self, values: Mapping, nuisances: Mapping = None) -> None:
        x = np.zeros(len(self.names))
        for k, v in (nuisances or {}).items():
            x[self.names.index(k)] = float(v)
        # expectations below 1e-30 events are empty bins: a data value of ~1e-320 underflows the deviance's floor
        # (MU_FLOOR n) to zero and divides by it (2026-11-11, the first Plan A forecast pilot)
        self.data = [np.where(np.abs(m) < 1e-30, 0.0, np.array(m, dtype=np.float64))
                     for m in self._mu(x, self.signals(values))]
        self._x0 = x.copy()

    def expected(self, values: Mapping):
        """{(label, topology, band): (signal, background)} at the nuisances' centres."""
        S = self.signals(values)
        return {(b["label"], b["topo"], b["band"]): (s[""], b["B"]) for b, s in zip(self.blocks, S)}
