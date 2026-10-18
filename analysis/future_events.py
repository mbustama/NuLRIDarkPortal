"""
future_events.py -- the future detectors' event likelihood for the projections (docs/dev/FUTURE_DETECTORS_PLAN.md;
Mauricio, 2026-10-14/15): NuFlavorSingleSource's eight detectors (IceCubeRates.future: its models vendored unchanged,
counted per channel, reconstructed energy and local cos-zenith band by IceCubeRates.future.diffuse), against the
model's flux per local zenith at each detector's site (FlavorModel.site_band_fluxes), with NuFlavorSingleSource's
main-analysis uncertainties profiled inside the term:

  mu_k = (1 + eff_k) s_k + escale_k s_k^escale + misid_k s_k^misid
         + n_k B_k + sum_g theta_g GB_k,g + escale_k BX_k^escale + misid_k BX_k^misid

per block k = (detector part, channel): s the astrophysical events (the model's flux, the signal here), B the
atmospheric background (NuFlavorSingleSource's MCEq table, with its atmospheric-muon multipliers), n_k its
normalization (10% prior), theta_g the Barr parameters, the cosmic-ray index and the prompt normalization (shared, with
NuFlavorSingleSource's priors), eff, escale, misid per block (DET_PRIOR; no PSF: no rings). No cross-section variation
(Mauricio, 2026-10-15; NuFlavorSingleSource has it at 5%): it needs the Earth propagation with regeneration and
effective areas rebuilt with the varied cross sections.
-2 ln L = 2 sum [mu - n + n ln(n / mu)] (continued smoothly below a floor, as the HESE and DR2 terms) + the
priors; the data of a projection: the Asimov expectation at the
configuration's asimov values (standard physics) with the nuisances at their centres. The expectation is linear in
every nuisance, so -2 ln L is convex in them and its Hessian exact: the profile takes projected Newton steps within
their bounds (2026-10-15: 4-5 steps, ~0.03 s, where L-BFGS-B took ~2000 iterations and 8-10 s and stopped ~1e-7
short of the minimum).

Detector sets as NuFlavorSingleSource's run.py (config_parts): 'All8_base' (its main case: IceCube, the six water
detectors and IceCube-Gen2, published selections), 'All_<sc>', 'Current_<sc>', 'Gen2_<sc>', single parts.
"""
import os
import sys
from typing import Mapping

import numpy as np
from scipy.optimize import minimize

from types import SimpleNamespace

from . import REPO_DIR
from .events import MU_FLOOR, NEG_PENALTY, _poisson

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from IceCubeRates.future import nfss_detectors as ND  # noqa: E402
from IceCubeRates.future import nfss_forecast as NF  # noqa: E402
from IceCubeRates.future.diffuse import ATM_PARAMS, COSZ_EDGES, KEYS, DiffuseDetector  # noqa: E402

WATER = ("ARCA", "GVD", "P-ONE", "TRIDENT", "NEON", "HUNT")
GRADIENT_TOL = 1e-20         # the profile's Newton decrement for chi2_and_grad: to its rounding floor
# the detectors' latitudes (deg), as NuFlavorSingleSource's calibrate() sets them (checked in the tests)
SITE_LAT = {"IceCube": -89.99, "Gen2": -89.99, "ARCA": 36.27, "GVD": 51.77, "P-ONE": 47.7, "TRIDENT": 17.4,
            "NEON": 17.4, "HUNT": 17.4}
BKG_PRIOR = 0.10
DET_PARAMS = ("eff", "escale", "misid")


def config_parts(config: str):
    """NuFlavorSingleSource's detector configurations (code/run.py CONFIGS): the detector parts they combine."""
    if config == "IceCube":
        return ["IceCube"]
    name, sc = config.rsplit("_", 1)
    if sc not in ("base", "mid", "ded"):
        raise ValueError(f"unknown future-detector configuration {config!r}")
    gen2 = ["Gen2_IC", "Gen2_new_" + sc]
    sets = {"Current": ["IceCube", "ARCA_" + sc, "GVD_" + sc], "All": ["IceCube"] + [w + "_" + sc for w in WATER],
            "All8": ["IceCube"] + [w + "_" + sc for w in WATER] + gen2, "Gen2": gen2}
    if name in sets:
        return sets[name]
    if name in WATER:
        return [config]
    raise ValueError(f"unknown future-detector configuration {config!r}")


def part_lat(part: str) -> float:
    return SITE_LAT[part.split("_")[0]]


def config_sites(config: str):
    """The latitudes the model must provide per-zenith fluxes for."""
    return sorted({part_lat(p) for p in config_parts(config)})


class FutureEventLikelihood:
    """-2 ln L of the future detectors' events at the model's values, profiled over their nuisances (module docstring).

    Arguments: model -- a FlavorModel in sky mode built with sites=config_sites(config); spec -- {"config":
    "All8_base"}."""

    def __init__(self, model, spec: Mapping):
        spec = dict(spec)
        bad = set(spec) - {"config"}
        if bad:
            raise ValueError(f"data.events.future: unknown keys {sorted(bad)}; known: config")
        self.config = str(spec.get("config", "All8_base"))
        self.model = model
        out, _ = ND.calibrate(verbose=False)
        self.dets = [DiffuseDetector(n, out[n]) for n in config_parts(self.config)]
        E_eV, _ = model.grid()
        self._lgE_model = np.log10(np.asarray(E_eV, dtype=np.float64) / 1e9)
        lo, hi = self._lgE_model[0], self._lgE_model[-1]
        lg = np.log10(NF.E)
        self._inside = (lg >= lo - 1e-9) & (lg <= hi + 1e-9)
        self._lg = np.clip(lg, lo, hi)
        # np.interp's linear interpolation in log E as fixed gather weights (2026-10-17: one vectorized step for all
        # directions and flavors instead of a call per band and flavor; equal to rounding)
        xp = self._lgE_model
        j = np.clip(np.searchsorted(xp, self._lg, side="right") - 1, 0, len(xp) - 2)
        self._ij, self._it = j, (self._lg - xp[j]) / (xp[j + 1] - xp[j])
        self.blocks = [(d, ch) for d in self.dets for ch in d.channels]
        # each block's responses stacked over the six (flavor, kind) KEYS and laid out per cos-zenith bin, (nz,
        # 6 nE, nb): its signals and the gradient's adjoint pass are batched matrix products (2026-10-17: ~1000 small
        # einsums per evaluation before; the same sums to rounding)
        def stacked(get):
            A = np.stack([get(k) for k in KEYS])                         # (6, nE, nb, nz)
            return np.ascontiguousarray(A.transpose(3, 0, 1, 2).reshape(A.shape[3], -1, A.shape[2]))
        self._Rz = [dict(s=stacked(lambda k, d=d, ch=ch: d.R[ch][k]),
                         escale=stacked(lambda k, d=d, ch=ch: d.RX[ch][k]["escale"]),
                         misid=stacked(lambda k, d=d, ch=ch: d.RX[ch][k]["misid"])) for d, ch in self.blocks]
        gparams = list(ATM_PARAMS)
        names, centre, sigma, bounds = [], [], [], []
        for d, ch in self.blocks:
            names.append(f"{d.name}|{ch}|bkg")
            centre.append(1.0)
            sigma.append(BKG_PRIOR)
            bounds.append((0.2, 3.0))
        for p in gparams:
            names.append(p)
            centre.append(0.0)
            s = NF.global_prior(p)
            sigma.append(s)
            bounds.append((-5.0 * s, 5.0 * s))
        self._own = []
        for d, ch in self.blocks:
            own = {}
            for p in DET_PARAMS:
                if p == "misid" and ch == "trk":
                    continue
                own[p] = len(names)
                names.append(f"{d.name}|{ch}|{p}")
                centre.append(0.0)
                s = NF.DET_PRIOR[p]
                sigma.append(s)
                bounds.append((max(-5.0 * s, -0.9), 5.0 * s))
            self._own.append(own)
        self.names, self.centre, self.sigma = names, np.array(centre), np.array(sigma)
        self.bounds = bounds
        self._gidx = {p: names.index(p) for p in gparams}
        self._gparams = gparams
        # backgrounds, flattened per block
        self._B, self._GB, self._BX = [], [], []
        for d, ch in self.blocks:
            self._B.append(d.B[ch].reshape(-1))
            self._GB.append({p: d.GB[ch][p].reshape(-1) for p in gparams})
            self._BX.append({p: d.BX[ch][p].reshape(-1) for p in ("escale", "misid")})
        # the expectation is linear in every nuisance, mu_k = s_k + A_k x[cols_k] (_design): the profile is a convex
        # problem that Newton's method solves exactly in a few steps (profile; 2026-10-15)
        self._cols = []
        for k, own in enumerate(self._own):
            self._cols.append(np.array([k] + [own[p] for p in DET_PARAMS if p in own]
                                       + [self._gidx[p] for p in gparams], dtype=np.int64))
        self._lo = np.array([b[0] for b in bounds])
        self._hi = np.array([b[1] for b in bounds])
        self.data = None
        self._z0 = np.zeros(len(names))
        self.last = {}
        self.n_calls = 0
        self.n_newton = 0
        self.n_fallback = 0
        self.newton_fail = ""

    # ---------------------------------------------------------------- the astrophysical events
    def _interp_log(self, L):
        """np.interp of L (N_z, N_E_model, 3) along energy at the detectors' energies, vectorized: (N_z, nE, 3)."""
        j, t = self._ij, self._it
        a = L[:, j, :]
        return a + t[None, :, None] * (L[:, j + 1, :] - a)

    def _stack(self, phi_nu, phi_nb):
        """(N_z, nE, 3) per kind -> (N_z, 6 nE) in KEYS order ((e, nu), (e, nubar), (mu, nu), ...)."""
        fl_index = {"e": 0, "mu": 1, "tau": 2}
        return np.concatenate([(phi_nb if bar else phi_nu)[:, :, fl_index[fl]] for fl, bar in KEYS], axis=1)

    def _phi(self, F):
        """(N_z, N_E_model, 3) per cos-zenith band -> the flux on the detectors' energy grid, (N_z, nE, 3) (log-log;
        zero outside the model's grid)."""
        L = np.log(np.maximum(F, 1e-300))
        return np.exp(self._interp_log(L)) * self._inside[None, :, None]

    def _phi_jac(self, F, dFs):
        """_phi(F) and, for each derivative dF of F, the derivative of _phi: the log-log interpolation is linear in
        log F with fixed weights, so d phi = phi * interp(dF / F) exactly (zero where F sits at the 1e-300 floor, as
        log max(F, 1e-300) does)."""
        phi = self._phi(F)
        pos = F > 1e-300
        safe = np.where(pos, F, 1.0)
        dphi = {n: phi * self._interp_log(np.where(pos, dF / safe, 0.0)) for n, dF in dFs.items()}
        return phi, dphi

    def signals(self, values):
        """Per block: the signal and its templates (flattened), for the model's values."""
        fluxes = {}
        for d in self.dets:
            lat = part_lat(d.name)
            if lat not in fluxes:
                _, F_nu, F_nb = self.model.site_band_fluxes(values, lat)
                if F_nu.shape[0] != len(COSZ_EDGES) - 1:
                    raise ValueError("the model's direction bands must be the detectors' cos-zenith bins")
                fluxes[lat] = self._stack(self._phi(F_nu), self._phi(F_nb))
        return self._signals_from(fluxes)

    def _signals_from(self, fluxes):
        """Per block: the signal and its templates (flattened, reconstructed bin major), from each site's flux in the
        stacked layout (_stack): s[b, z] = sum over keys and energies of phi[z, k e] R[z, k e, b]."""
        S = []
        for (d, ch), Rz in zip(self.blocks, self._Rz):
            P = fluxes[part_lat(d.name)][:, None, :]                          # (nz, 1, 6 nE)
            out = {}
            for key in ("s", "escale", "misid"):
                out[key] = np.ascontiguousarray(np.matmul(P, Rz[key])[:, 0, :].T).reshape(-1)    # (nb, nz) -> flat
            S.append(out)
        return S

    # ---------------------------------------------------------------- expectation and profile
    def _mu(self, x, S, grad=False):
        mus, parts = [], []
        for k, (sk, own) in enumerate(zip(S, self._own)):
            eff = x[own["eff"]]
            mu = (1.0 + eff) * sk["s"] + x[own["escale"]] * (sk["escale"] + self._BX[k]["escale"])
            if "misid" in own:
                mu = mu + x[own["misid"]] * (sk["misid"] + self._BX[k]["misid"])
            mu = mu + x[k] * self._B[k]
            for p in self._gparams:
                mu = mu + x[self._gidx[p]] * self._GB[k][p]
            mus.append(mu)
        return mus

    def _dev(self, x, S, data, grad=True):
        mus = self._mu(x, S)
        f = 0.0
        g = np.zeros_like(x)
        for k, (mu, n, sk, own) in enumerate(zip(mus, data, S, self._own)):
            # the Poisson deviance, continued smoothly where no physical expectation goes (events._poisson): trial
            # steps of the profile into negative expectations stay finite, with consistent gradients
            dev, r = _poisson(mu, n, grad=True)
            f += dev
            g[k] += r @ self._B[k]
            g[own["eff"]] += r @ sk["s"]
            g[own["escale"]] += r @ (sk["escale"] + self._BX[k]["escale"])
            if "misid" in own:
                g[own["misid"]] += r @ (sk["misid"] + self._BX[k]["misid"])
            for p in self._gparams:
                g[self._gidx[p]] += r @ self._GB[k][p]
        z = (x - self.centre) / self.sigma
        f += float(z @ z)
        g += 2.0 * z / self.sigma
        return f, g

    def _design(self, S):
        """Per block, the columns of the expectation's linear dependence on its nuisances (in _cols' order): the
        background (its normalization multiplies it), the signal (efficiency), the energy scale's and the
        mis-identification's templates (signal and background), and each global atmospheric template."""
        A = []
        for k, (sk, own) in enumerate(zip(S, self._own)):
            cols = [self._B[k], sk["s"], sk["escale"] + self._BX[k]["escale"]]
            if "misid" in own:
                cols.append(sk["misid"] + self._BX[k]["misid"])
            cols += [self._GB[k][p] for p in self._gparams]
            A.append(np.ascontiguousarray(np.stack(cols, axis=1)))
        return A

    def _fgh(self, x, A, S, data, hess=True):
        """-2 ln L (the continued Poisson deviance plus the priors), its gradient and its Hessian at the nuisances
        x (the Hessian exact: the expectation is linear in x)."""
        n_par = len(x)
        f, g = 0.0, np.zeros(n_par)
        H = np.zeros((n_par, n_par)) if hess else None
        for k, (Ak, sk, n) in enumerate(zip(A, S, data)):
            c = self._cols[k]
            mu = sk["s"] + Ak @ x[c]
            dev, r = _poisson(mu, n, grad=True)
            f += dev
            if hess:
                g[c] += Ak.T @ r
                # the continued deviance's second derivative (events._poisson): 2 n / max(mu, MU_FLOOR n)^2 in bins
                # with data, 0 in empty bins (2 NEG_PENALTY below mu = 0)
                pos = n > 0
                m = np.where(pos, np.maximum(mu, MU_FLOOR * n), 1.0)
                h = np.where(pos, 2.0 * n / (m * m), np.where(mu < 0.0, 2.0 * NEG_PENALTY, 0.0))
                H[np.ix_(c, c)] += Ak.T @ (h[:, None] * Ak)
        z = (x - self.centre) / self.sigma
        f += float(z @ z)
        if hess:
            g += 2.0 * z / self.sigma
            H[np.diag_indices(n_par)] += 2.0 / self.sigma ** 2
        return f, g, H

    def _newton(self, S, data, x0, tol=1e-12, max_iter=100):
        # (tol below ~1e-14 reaches the rounding floor: then a decrement that stops falling ends the iteration)
        """The profile by projected Newton steps within the nuisances' bounds (Bertsekas 1982): a nuisance at a
        bound whose gradient pushes outward stays there, the others take the Newton step of their block of the
        exact Hessian, and a backtracking search on the step projected into the bounds keeps the descent. Returns
        (x, chi^2, iterations) once the free nuisances' Newton decrement falls below tol (chi^2 then within tol / 2 of
        the minimum: the problem is convex), or None (the caller falls back to L-BFGS-B)."""
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
            # converged: below tol, or stalled at f's rounding (the decrement no longer halving below 1e-9: chi^2
            # within 5e-10; GitHub's runners sat at 2e-11 - 6e-11 for 100 steps, 2026-10-18)
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
                        # converged to the rounding of f: the predicted decrease (dec / 2 < 5e-7 in chi^2) is below
                        # what f resolves, so no step can show it (machine-dependent: GitHub's runners, 2026-10-18)
                        return x, f, it
                    self.newton_fail = f"line search failed (decrement {dec:g})"
                    return None
            x = xn
            f, g, H = self._fgh(x, A, S, data)
        self.newton_fail = f"no convergence in {max_iter} steps (decrement {dec:g})"
        return None

    def profile(self, S, data, tol=1e-12):
        """The nuisances minimizing -2 ln L within their bounds: projected Newton steps from the last profile's
        nuisances (exact in a few steps: the problem is convex; 400 times faster than L-BFGS-B, which needed ~2000
        iterations), L-BFGS-B if Newton ever stalls."""
        x0 = np.clip(self.centre + self.sigma * self._z0, self._lo, self._hi)
        out = self._newton(S, data, x0, tol=tol)
        if out is not None:
            x, f, it = out
            self.n_newton += 1
            self._z0 = (x - self.centre) / self.sigma
            return x, SimpleNamespace(fun=f, x=self._z0.copy(), success=True, nit=it)
        self.n_fallback += 1
        return self.profile_lbfgsb(S, data)

    def profile_lbfgsb(self, S, data):
        zb = [((lo - c) / s, (hi - c) / s) for (lo, hi), c, s in zip(self.bounds, self.centre, self.sigma)]

        def fun(z):
            f, g = self._dev(self.centre + self.sigma * z, S, data)
            return f, g * self.sigma
        r = minimize(fun, np.clip(self._z0, [b[0] for b in zb], [b[1] for b in zb]), jac=True, method="L-BFGS-B",
                     bounds=zb, options=dict(maxiter=5000, ftol=1e-15, gtol=1e-9))
        self._z0 = np.array(r.x)
        return self.centre + self.sigma * r.x, r

    def chi2(self, values: Mapping) -> float:
        if self.data is None:
            raise ValueError("the future detectors' likelihood has no data: a projection needs Asimov data "
                             "(set_asimov, i.e. the analysis' --asimov or data_centre)")
        self.n_calls += 1
        S = self.signals(values)
        x, r = self.profile(S, self.data)
        self.last = dict(nuisances=dict(zip(self.names, x.tolist())), value=float(r.fun), converged=bool(r.success),
                         nit=int(r.nit))
        return float(r.fun)

    def chi2_and_grad(self, values: Mapping, names):
        """chi2(values) (the same value) and its gradient in `names` (model.GRADIENT_NAMES), (chi2, array).

        The profile's minimum is differentiable with the nuisances held at their optimum (Danskin's theorem: its
        minimum is unique -- the problem is strictly convex -- within bounds that do not depend on the parameters;
        checked against re-profiled finite differences on the campaign's likelihood, interior and with 31 nuisances
        at their bounds, 2026-10-17). So the gradient is the deviance's derivative through the signal templates at
        the optimum: the per-bin residuals contracted once with the responses (an adjoint pass), then with each
        derivative of the fluxes (FlavorModel.site_band_flux_jacobian), which is exact. The profile here runs to its
        rounding floor (GRADIENT_TOL): the gradient's error is first order in the nuisances' residual error."""
        if self.data is None:
            raise ValueError("the future detectors' likelihood has no data: a projection needs Asimov data "
                             "(set_asimov, i.e. the analysis' --asimov or data_centre)")
        self.n_calls += 1
        names = list(names)
        fluxes, dfluxes = {}, {}
        for d in self.dets:
            lat = part_lat(d.name)
            if lat not in fluxes:
                _, F_nu, F_nb, dF = self.model.site_band_flux_jacobian(values, lat, names)
                if F_nu.shape[0] != len(COSZ_EDGES) - 1:
                    raise ValueError("the model's direction bands must be the detectors' cos-zenith bins")
                pn, dpn = self._phi_jac(F_nu, {n: dF[n][0] for n in names})
                pb, dpb = self._phi_jac(F_nb, {n: dF[n][1] for n in names})
                fluxes[lat] = self._stack(pn, pb)
                dfluxes[lat] = {n: self._stack(dpn[n], dpb[n]) for n in names}
        S = self._signals_from(fluxes)
        x, r = self.profile(S, self.data, tol=GRADIENT_TOL)
        self.last = dict(nuisances=dict(zip(self.names, x.tolist())), value=float(r.fun), converged=bool(r.success),
                         nit=int(r.nit))
        Q = {lat: np.zeros_like(P) for lat, P in fluxes.items()}              # d chi2 / d phi, (nz, 6 nE) per site
        for (d, ch), Rz, mu, n_obs, own in zip(self.blocks, self._Rz, self._mu(x, S), self.data, self._own):
            _, res = _poisson(mu, n_obs, grad=True)
            Wt = res.reshape(d.B[ch].shape).T[:, :, None]                     # (nz, nb, 1): d chi2 / d mu
            q = (1.0 + x[own["eff"]]) * np.matmul(Rz["s"], Wt) + x[own["escale"]] * np.matmul(Rz["escale"], Wt)
            if "misid" in own:
                q = q + x[own["misid"]] * np.matmul(Rz["misid"], Wt)
            Q[part_lat(d.name)] += q[:, :, 0]
        grad = np.array([sum(float(np.sum(Q[lat] * dfluxes[lat][n])) for lat in Q) for n in names])
        return float(r.fun), grad

    def set_asimov(self, values: Mapping, nuisances: Mapping = None) -> None:
        x = self.centre.copy()
        for k, v in (nuisances or {}).items():
            x[self.names.index(k)] = float(v)
        self.data = [np.array(m) for m in self._mu(x, self.signals(values))]
        self._z0 = (x - self.centre) / self.sigma

    def expected(self, values: Mapping):
        """{(part, channel): (nb, nz) expected events} at the nuisances' centres: signal and background."""
        mus = self._mu(self.centre, self.signals(values))
        return {(d.name, ch): m.reshape(d.B[ch].shape) for (d, ch), m in zip(self.blocks, mus)}
