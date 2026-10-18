"""
exact_profile.py -- the profile over the neutrino mixing and the lightest neutrino mass, exact at every point
(2026-10-18; docs/dev/EXACT_PROFILE.md).

The problem. At a point of a plane (mediator mass and coupling held) chi^2 is minimized over the cheap nuisances nu
(normalizations, compositions, spectral indices: they need no new solver runs) and over the parameters eta that change
the propagation -- sin^2 th12, sin^2 th13, sin^2 th23, delta_CP, Delta m^2_21, Delta m^2_31 and m_lightest. Each new
value of eta needs the solver at every spectral-index node of both pieces of the broken power law (on Tycho ~3 s for a
potential point, 17-28 s for an MeV point), and a fit with finite differences tries hundreds of them.

The method: a trust-region Gauss-Newton iteration in eta whose iterates (centres) are evaluated exactly.

  Surrogate. At a centre c the model is templates.py's model centred there: the exact runs at c (every index node), and
  one forward-difference run per parameter of eta (a step h_i of a twentieth of the prior's width, at each piece's
  central index node), combined element by element (energy, flavor, band, source) as
      F(eta, p) = F(c, p) exp[sum_i (eta_i - c_i) / h_i log(F_i / F_c)].
  It is the exact model at eta = c, for every nu (templates.py skips the template runs there: the factor is exactly 1),
  and its gradient in eta there is the forward difference's. The surrogate chi^2 -- this flux in the analysis' own
  likelihood, the priors exact -- costs no solver run.

  Step. The surrogate is minimized over (eta, nu) with eta in the box |eta_i - c_i| <= Delta w_i (w_i the prior's
  width, Delta the trust radius). If its best value lies less than `stop` below the centre's exact value within a box
  of at least `cert_radius` widths, the centre is the profile (certified, below). Otherwise the step's end is
  evaluated exactly -- it becomes a candidate centre -- and the step accepted when the exact chi^2 decreased: the radius
  doubles when the surrogate predicted the decrease well (rho > 0.75) at the box's edge and halves when it did not
  (rho < 0.25); a rise is rejected and the radius cut to a quarter of the step. The accepted centre's forward
  differences are computed when its surrogate is first needed.

Why the result is exact. (i) Every chi^2 reported is the exact model's at the reported values: the solver at the
reported eta, every index node, nu fitted on those runs. (ii) The certificate: at the reported centre the surrogate
equals the exact chi^2 for every nu and has the exact gradient in eta up to the forward difference's truncation e (of
order h times the flux's curvature: below 1e-3 chi^2 per prior width at h = w / 20, measured), and no point of the
surrogate within cert_radius widths lies `stop` below. The exact chi^2's own minimum near the centre can then lie
below the reported value by at most stop plus e^2 / (2 lambda) (lambda the curvature, at least the prior's ~2 per
width^2): ~1e-6 beyond `stop`. (iii) Separate basins -- the two octants of theta23 under NuFIT, the lightest mass at
MeV absorption -- are the caller's to start in (each start is its own trust-region run; the lowest certified one is
the profile).

m_lightest. Bounded below by M_LIGHTEST_MIN (1e-6 eV), not 0: the MeV transport treats an exactly massless relic as
not interacting at all (dp_si: s = 2 E m = 0), while any m > 0 gets the thermal (relativistic) treatment, continuous
as m -> 0; the bound changes the mass-sum penalty by 3e-5.
"""
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .fitting import FitSettings, fit
from .model import ALIASES, sin2_to_deg
from .parameters import GaussianPrior, Parameter, ParameterSet
from .templates import MixingTemplate

ETA = ("s12sq", "s13sq", "s23sq", "d_cp_deg", "dm21_sq", "dm31_sq", "m_lightest_eV")
SOLVER_KEYS = {"s12sq": ("standard_mixing", "th12_deg"), "s13sq": ("standard_mixing", "th13_deg"),
               "s23sq": ("standard_mixing", "th23_deg"), "d_cp_deg": ("standard_mixing", "d_cp_deg"),
               "dm21_sq": ("standard_mixing", "dm21_sq"), "dm31_sq": ("standard_mixing", "dm31_sq"),
               "m_lightest_eV": ("cosmology", "m_lightest_eV")}
# NuFIT 6.1 (normal ordering, with SK's atmospheric data): approximate 1-sigma widths -- the trust region's units and
# the forward differences' steps only, never the penalty (nufit.py)
NUFIT_WIDTHS = dict(s12sq=0.012, s13sq=0.00057, s23sq=0.018, d_cp_deg=30.0, dm21_sq=2.0e-6, dm31_sq=2.5e-5)
M_LIGHTEST_WIDTH = 2e-3           # eV: the trust region's unit for m_lightest (the relics' thermal scale, ~10 T_nu)
M_LIGHTEST_MIN = 1e-6             # eV (module docstring)


@dataclass
class ExactSettings:
    """Knobs of the exact profile (module docstring). radius, radius_max, cert_radius: trust radii in prior widths;
    fd_rel: the differences' step in prior widths (m_lightest's: M_LIGHTEST_WIDTH); stop: the certified
    predicted decrease; max_centres: exact centres per start; accept: the smallest exact decrease accepted."""
    radius: float = 1.0
    radius_max: float = 64.0
    radius_min: float = 1e-3
    cert_radius: float = 0.5
    fd_rel: float = 0.25
    stop: float = 1e-4
    max_centres: int = 24
    accept: float = 1e-9
    # the lightest mass's global search (profile_global): the exact chi^2 on this grid of m_lightest (eV) at the
    # start's mixing, then the exact profile from every local minimum of the scan within ml_margin of its lowest
    ml_scan: Optional[Sequence[float]] = None
    ml_margin: float = 3.0
    # NuFIT's two octants of theta23 (profile_global): the exact profile also from the start mirrored about 0.5
    octants: bool = False
    # a plane's chain (analysis.scans): a point whose solver runs at the last certified point's values equal that
    # point's (relative inherit_eps per element, from each energy's largest; 2 float32 units in sky mode, whose bands
    # are single precision) takes its certificate after one evaluation (ExactProfiler.inherit); never with ml_scan
    inherit: bool = True
    inherit_eps: float = 1e-10
    inherit_eps_sky: float = 2.4e-7

    @classmethod
    def from_dict(cls, d: Optional[Mapping]) -> "ExactSettings":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown exact-profile settings {sorted(unknown)}; known: {sorted(cls.__dataclass_fields__)}")
        return cls(**d)


@dataclass
class ExactResult:
    """The profile at one point: values (every parameter, held ones included), chi2 (the exact model's there), the
    certificate (certified, the last predicted decrease and its box), the centres visited (eta, exact chi^2, accepted)
    and the cost (solver runs, seconds)."""
    values: Dict[str, float]
    chi2: float
    certified: bool
    message: str
    pred: float
    pred_radius: float
    centres: List[dict] = field(default_factory=list)
    solver_runs: int = 0
    seconds: float = 0.0
    nfev: int = 0
    scan: List[dict] = field(default_factory=list)
    starts: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(values=self.values, chi2=self.chi2, certified=self.certified, message=self.message, pred=self.pred,
                    pred_radius=self.pred_radius, centres=self.centres, solver_runs=self.solver_runs,
                    seconds=self.seconds, nfev=self.nfev, scan=self.scan, starts=self.starts)


def widths(cfg, names: Sequence[str]) -> Dict[str, float]:
    """The prior's width of each parameter of eta: a Gaussian prior's, NuFIT 6.1's approximate 1 sigma under the
    pairwise prior, M_LIGHTEST_WIDTH for m_lightest."""
    gauss = {p.name: float(p.width) for p in cfg.priors if isinstance(p, GaussianPrior)}
    out = {}
    for n in names:
        if n in gauss:
            out[n] = gauss[n]
        elif n == "m_lightest_eV":
            out[n] = M_LIGHTEST_WIDTH
        else:
            out[n] = NUFIT_WIDTHS[n]
    return out


class ExactProfiler:
    """The exact profile over eta (module docstring) for one analysis configuration.

    Arguments: cfg -- an AnalysisConfig whose eta parameters are free and carry no template (exact_config gives one
    from any campaign configuration); objective -- its Objective (built here when None; with data_centre for Asimov
    data); settings -- ExactSettings.
    """

    def __init__(self, cfg, objective=None, data_centre=None, settings: Optional[ExactSettings] = None):
        if cfg.templates:
            raise ValueError("the exact profile builds its own local templates: give it a configuration without "
                             "'template' entries (exact_config)")
        self.cfg = cfg
        self.settings = settings or ExactSettings()
        self.obj = objective if objective is not None else cfg.objective(data_centre=data_centre)
        self.exact_model = self.obj.model
        self.eta = [n for n in ETA if n in cfg.free_names]
        self.w = widths(cfg, self.eta)
        self.bounds = {p.name: tuple(p.bounds) for p in cfg.parameters}
        fs = dict(vars(cfg.fit))
        # central differences (fitting.FitSettings.fd_central: the forward ones stop short in norm_astro), with an
        # evaluation budget of ~150 gradients
        fs.update(polish=False, n_random=0, fd_central=True, max_evals=max(int(cfg.fit.max_evals), 4000))
        self.fit_settings = FitSettings(**fs)
        self.solver_runs = 0

    # ------------------------------------------------------------------ models
    def _fd_template(self, name: str, c: float, central: bool) -> MixingTemplate:
        """The difference template of one parameter at c: forward, the log of the flux linear with the slope to
        c + h (templates.MixingTemplate with a slope); central, the parabola in the log of the flux through c - h, c
        and c + h (Lagrange: the central difference's slope and the second difference's curvature), or at a bound
        through c, c + h and c + 2h inward (the one-sided second-order slope); h = fd_rel prior widths."""
        h = self.settings.fd_rel * self.w[name]
        lo, hi = self.bounds[name]
        up, down = c + h <= hi, c - h >= lo
        if central:
            if up and down:
                return MixingTemplate(name, c, (c + h, c - h))
            if up and c + 2.0 * h <= hi:
                return MixingTemplate(name, c, (c + h, c + 2.0 * h))
            if down and c - 2.0 * h >= lo:
                return MixingTemplate(name, c, (c - h, c - 2.0 * h))
        return MixingTemplate(name, c, (c + h,) if up else (c - h,), slope=(1.0,))

    def centre_model(self, centre: Mapping[str, float], central: bool = False, reuse=None):
        """The model centred at eta = centre: the solver configuration there, and per parameter of eta a difference
        template (_fd_template: log-linear with a forward difference's slope, or the central parabola).
        reuse: a model centred at the same values whose solver runs are taken over (a forward model upgraded to
        central differences then makes only the new points' runs)."""
        solver = copy.deepcopy(self.cfg.solver)
        templates = []
        for n in self.eta:
            c = float(centre[n])
            section, key = SOLVER_KEYS[n]
            solver.setdefault(section, {})[key] = sin2_to_deg(c) if n in ALIASES else c
            templates.append(self._fd_template(n, c, central))
        m = self.cfg.model(solver=solver, templates=templates)
        if reuse is not None:
            # the same base configuration: a run's cache key is the same in both models
            for name in ("_cache", "_sky_cache", "_gal_cache"):
                dst = getattr(m, name)
                for k, v in getattr(reuse, name).items():
                    dst[k] = v
        return m

    def use(self, model) -> None:
        """Puts `model` into the objective and its event likelihoods (the data stay)."""
        self.obj.model = model
        ev = self.obj.events
        if ev is not None:
            for t in getattr(ev, "terms", [ev]):
                t.model = model

    def _runs(self, model) -> int:
        return int(getattr(model, "n_solver_calls", 0))

    # ------------------------------------------------------------------ fits
    def _nu_names(self, held: Mapping) -> List[str]:
        return [p.name for p in self.cfg.parameters if p.name not in held and p.name not in self.eta]

    def _centre_value(self, held: Mapping, centre: Mapping, start: Mapping):
        """min over nu of the (exact) chi^2 at eta = centre: (chi2, values, nfev)."""
        pset = self.cfg.parameter_set().holding({**held, **{n: float(centre[n]) for n in self.eta}})
        if not pset.free:
            v = dict(pset.held)
            return float(self.obj.chi2(v)), v, 1
        r = fit(self.obj, pset, starts=[{k: v for k, v in start.items() if k in pset.names}],
                settings=self.fit_settings)
        return float(r.chi2), dict(r.values), int(r.nfev)

    def _box_pset(self, held: Mapping, centre: Mapping, radius: float) -> ParameterSet:
        free = []
        for p in self.cfg.parameters:
            if p.name in held:
                continue
            if p.name in self.eta:
                c, w = float(centre[p.name]), self.w[p.name]
                lo, hi = max(p.bounds[0], c - radius * w), min(p.bounds[1], c + radius * w)
                if not lo < hi:
                    lo, hi = min(lo, c) - 1e-12 * max(1.0, abs(c)), max(hi, c) + 1e-12 * max(1.0, abs(c))
                free.append(Parameter(p.name, min(max(c, lo), hi), (lo, hi), "linear"))
            else:
                free.append(p)
        return ParameterSet(free, dict(held))

    def _surrogate_step(self, held: Mapping, centre: Mapping, values_c: Mapping, radius: float):
        """min of the surrogate centred at `centre` over (eta in the box of `radius` widths, nu), from the centre's
        own solution: FitResult."""
        pset = self._box_pset(held, centre, radius)
        start = {k: v for k, v in values_c.items() if k in pset.names}
        start.update({n: float(centre[n]) for n in self.eta})
        return fit(self.obj, pset, starts=[start], settings=self.fit_settings)

    # ------------------------------------------------------------------ the profile
    def profile(self, held: Mapping, start: Mapping, log=None) -> ExactResult:
        """The exact profile at the held values from one start (values of eta and nu; missing ones take the
        parameters' starts)."""
        s = self.settings
        t0 = time.perf_counter()
        held = {k: float(v) for k, v in held.items()}
        st = {p.name: p.start for p in self.cfg.parameters}
        st.update({k: float(v) for k, v in start.items()})
        centre = {}
        for n in self.eta:
            lo, hi = self.bounds[n]
            centre[n] = float(min(max(st[n], lo), hi))
        nfev = 0
        runs = 0                                      # solver runs of the models dropped so far
        model_c = model_t = self.centre_model(centre)
        self.use(model_c)
        certified, message, pred, pred_radius = False, "", np.inf, 0.0
        central = False                               # the current centre's differences
        try:
            f_c, vals_c, k = self._centre_value(held, centre, st)
            nfev += k
            centres = [dict(eta=dict(centre), chi2=f_c, accepted=True)]
            radius = s.radius
            while True:
                big = max(radius, s.cert_radius)
                r = self._surrogate_step(held, centre, vals_c, big)
                nfev += r.nfev
                pred, pred_radius = f_c - float(r.chi2), big
                if pred <= s.stop:
                    if not central:
                        # the certificate takes central differences: the forward ones' truncation (O(h)) can leave
                        # a gradient error that matters where the data pull hard (2026-10-18: proj_2050 at chi^2 ~
                        # 3000 stalled on it); the centre's runs are reused, the backward points added
                        runs += self._runs(model_c)
                        model_c = model_t = self.centre_model(centre, central=True, reuse=model_c)
                        self.use(model_c)
                        central = True
                        continue
                    certified, message = True, f"certified: surrogate decrease {pred:.2g} within {big:.3g} widths"
                    break
                if radius < s.cert_radius:
                    r = self._surrogate_step(held, centre, vals_c, radius)
                    nfev += r.nfev
                    if f_c - float(r.chi2) <= 0.0:
                        message = f"stalled: no surrogate decrease within {radius:.3g} widths (predicted {pred:.3g} within {big:.3g})"
                        break
                p_step = f_c - float(r.chi2)
                trial = {n: float(r.values[n]) for n in self.eta}
                step = max(abs(trial[n] - centre[n]) / self.w[n] for n in self.eta)
                if len(centres) >= s.max_centres:
                    message = f"stopped after {len(centres)} centres (predicted decrease {pred:.3g})"
                    break
                model_t = self.centre_model(trial)
                self.use(model_t)
                f_t, vals_t, k = self._centre_value(held, trial, r.values)
                nfev += k
                rho = (f_c - f_t) / p_step if p_step > 0 else 0.0
                accepted = f_t < f_c - s.accept
                centres.append(dict(eta=dict(trial), chi2=f_t, accepted=bool(accepted), predicted=f_c - p_step,
                                    rho=rho, radius=radius, step=step))
                if log is not None:
                    log(f"    centre {len(centres) - 1}: chi2 {f_t:.6f} (predicted {f_c - p_step:.6f}, from {f_c:.6f}), "
                        f"step {step:.3g} widths, rho {rho:.3g}, {'accepted' if accepted else 'rejected'}")
                if accepted:
                    if rho > 0.75 and step > 0.9 * radius:
                        radius = min(2.0 * radius, s.radius_max)
                    elif rho < 0.25:
                        radius = max(0.5 * radius, s.radius_min)
                    runs += self._runs(model_c)
                    centre, f_c, vals_c, model_c = trial, f_t, vals_t, model_t
                    central = False
                else:
                    runs += self._runs(model_t)
                    model_t = model_c
                    radius = 0.25 * step
                    if not central and step < s.cert_radius:
                        # a short step that failed: the surrogate's slope is suspect -- central differences here,
                        # and a fresh radius for the new surrogate
                        runs += self._runs(model_c)
                        model_c = model_t = self.centre_model(centre, central=True, reuse=model_c)
                        central = True
                        radius = max(radius, s.cert_radius)
                    self.use(model_c)
                    if radius < s.radius_min:
                        message = f"stalled: trust radius {radius:.2g} widths (predicted {pred:.3g})"
                        break
        finally:
            runs += self._runs(model_c) + (self._runs(model_t) if model_t is not model_c else 0)
            self.use(self.exact_model)
            self.last_model = (dict(centre), model_c)  # the reported centre and its exact runs (anchor)
        values = dict(vals_c)
        values.update({n: float(centre[n]) for n in self.eta})
        self.solver_runs += runs
        return ExactResult(values=values, chi2=float(f_c), certified=certified, message=message, pred=float(pred),
                           pred_radius=float(pred_radius), centres=centres, solver_runs=runs,
                           seconds=time.perf_counter() - t0, nfev=nfev)

    def profile_starts(self, held: Mapping, starts: Sequence[Mapping], log=None) -> ExactResult:
        """profile() from each start; the lowest result, with every start's chi^2 in its message."""
        res = [self.profile(held, st, log=log) for st in starts]
        best = min(res, key=lambda r: (not r.certified, r.chi2))
        lowest = min(res, key=lambda r: r.chi2)
        if lowest.chi2 < best.chi2:
            best = lowest
        best.message += "; starts: " + ", ".join(f"{r.chi2:.6f}{'' if r.certified else ' (uncertified)'}" for r in res)
        best.solver_runs = sum(r.solver_runs for r in res)
        best.seconds = sum(r.seconds for r in res)
        best.nfev = sum(r.nfev for r in res)
        return best

    def value_at(self, held: Mapping, eta: Mapping, start: Mapping):
        """The exact chi^2 at eta, minimized over nu: (chi2, values, solver runs)."""
        centre = {n: float(min(max(float(eta[n]), self.bounds[n][0]), self.bounds[n][1])) for n in self.eta}
        model = self.centre_model(centre)
        self.use(model)
        try:
            f, v, _ = self._centre_value(held, centre, start)
        finally:
            self.use(self.exact_model)
        return f, v, self._runs(model)

    def profile_global(self, held: Mapping, start: Mapping, log=None) -> ExactResult:
        """The exact profile with the known separate basins searched (ExactSettings): with ml_scan, the exact chi^2
        on that grid of the lightest mass (the start's mixing, nu fitted) and profile() from each local minimum of
        the scan within ml_margin of its lowest (2026-10-18: at the MeV planes' points the lightest mass moves chi^2
        by up to ~80, with minima a factor ~2 in m apart); with octants, profile() also from the best start's
        theta23 mirrored about maximal mixing. Returns the lowest result, with every start's in `starts`."""
        s = self.settings
        t0 = time.perf_counter()
        held = {k: float(v) for k, v in held.items()}
        st = {p.name: p.start for p in self.cfg.parameters if p.name not in held}
        st.update({k: float(v) for k, v in start.items() if k not in held})
        starts, scan, runs = [dict(st)], [], 0
        if s.ml_scan is not None and "m_lightest_eV" in self.eta:
            lo, hi = self.bounds["m_lightest_eV"]
            grid = sorted({float(min(max(m, lo), hi)) for m in s.ml_scan})
            for m in grid:
                f, v, k = self.value_at(held, dict(st, m_lightest_eV=m), st)
                runs += k
                scan.append(dict(m_lightest_eV=m, chi2=f, values=v))
                if log is not None:
                    log(f"    scan m_lightest {m:.3g}: chi2 {f:.6f}")
            vals = [x["chi2"] for x in scan]
            low = min(vals)
            cand = [i for i in range(len(grid)) if vals[i] <= low + s.ml_margin
                    and (i == 0 or vals[i] <= vals[i - 1]) and (i == len(grid) - 1 or vals[i] <= vals[i + 1])]
            starts = [dict(scan[i]["values"]) for i in sorted(cand, key=lambda i: vals[i])]
        res = [self.profile(held, x, log=log) for x in starts]
        if s.octants and "s23sq" in self.eta:
            b = min(res, key=lambda r: r.chi2)
            mirror = dict(b.values)
            mirror["s23sq"] = float(min(max(1.0 - b.values["s23sq"], self.bounds["s23sq"][0]), self.bounds["s23sq"][1]))
            res.append(self.profile(held, {k: v for k, v in mirror.items() if k not in held}, log=log))
        best = min(res, key=lambda r: r.chi2)
        best.starts = [dict(start={n: r.centres[0]["eta"][n] for n in self.eta}, chi2=r.chi2, certified=r.certified,
                            centres=len(r.centres), solver_runs=r.solver_runs, message=r.message) for r in res]
        best.scan = [dict(m_lightest_eV=x["m_lightest_eV"], chi2=x["chi2"]) for x in scan]
        best.solver_runs = runs + sum(r.solver_runs for r in res)
        best.nfev = sum(r.nfev for r in res)
        best.seconds = time.perf_counter() - t0
        if len(res) > 1:
            best.message += f"; lowest of {len(res)} starts" + (f" after a {len(scan)}-point m_lightest scan" if scan
                                                                 else "")
        return best

    # ------------------------------------------------------------------ a chain's anchor (analysis.scans)
    def _runs_at(self, model, held: Mapping, values: Mapping) -> list:
        """The raw solver runs of `model` at values (held included): every index node's arrays, in the model's
        evaluation order (cached, or computed now)."""
        v = {**values, **held}
        out = []
        for x, templated in model._evaluation_values(v):
            if templated:
                continue
            if model.sky:
                r = model._sky_exact(x)
                out += [r["bands_nu"], r["bands_nb"]]
                if r["gal_nu"] is not None:
                    out += [r["gal_nu"], r["gal_nb"]]
                    for lat in sorted(r.get("site_gal") or {}):
                        out += list(r["site_gal"][lat])
            else:
                r = model._fluxes_exact(x)
                out += [a for a in (r[1], r[2]) if a is not None]
                if r[3] is not None and r[3].get("G") is not None:
                    out.append(r[3]["G"])
        return out

    def anchor(self, held: Mapping, result: "ExactResult") -> dict:
        """A certified point's anchor for the points after it in a chain: its values, its held values and its solver
        runs there (from the reported centre's model)."""
        eta = {n: float(result.values[n]) for n in self.eta}
        last = getattr(self, "last_model", None)
        model = last[1] if last is not None and last[0] == eta else self.centre_model(eta)
        return dict(held=dict(held), values=dict(result.values), runs=self._runs_at(model, held, result.values),
                    chi2=float(result.chi2))

    def inherit(self, held: Mapping, anchor: dict) -> Optional["ExactResult"]:
        """The point at `held` profiled from a chain's anchor when new physics is below the solver's precision there:
        its runs at the anchor's values (one evaluation) equal the anchor's to inherit_eps (per element, relative
        to each energy's largest), so its chi^2 is the anchor's function of every parameter up to that perturbation,
        whose effect on the minimum is first order at the anchor's solution -- where it is evaluated exactly here,
        nu refitted -- and second order elsewhere. Returns the result (certified by the anchor), or None (profile it:
        new physics shows, or the lightest mass is scanned -- its other values are not compared)."""
        s = self.settings
        if not s.inherit or s.ml_scan is not None:
            return None
        t0 = time.perf_counter()
        held = {k: float(v) for k, v in held.items()}
        eta = {n: float(anchor["values"][n]) for n in self.eta}
        model = self.centre_model(eta)
        try:
            runs = self._runs_at(model, held, anchor["values"])
        except ValueError:
            return None
        eps = s.inherit_eps_sky if model.sky else s.inherit_eps
        worst = 0.0
        for a, b in zip(runs, anchor["runs"]):
            if a.shape != b.shape:
                return None
            scale = np.maximum(np.max(np.abs(b), axis=-1, keepdims=True), 1e-300)
            worst = max(worst, float(np.max(np.abs(a - b) / scale)))
            if worst > eps:
                return None
        self.use(model)
        try:
            f, vals, k = self._centre_value(held, eta, anchor["values"])
        finally:
            self.use(self.exact_model)
        values = dict(vals)
        values.update(eta)
        return ExactResult(values=values, chi2=float(f), certified=True,
                           message=f"inherited: runs within {worst:.2g} of the anchor at {anchor['held']}",
                           pred=0.0, pred_radius=0.0, centres=[dict(eta=eta, chi2=float(f), accepted=True)],
                           solver_runs=self._runs(model), seconds=time.perf_counter() - t0, nfev=k)

    def within_certificate(self, own_values: Mapping, other: Mapping) -> bool:
        """Whether `other`'s mixing lies within a certified point's certificate box (cert_radius widths of its own):
        a repair need not try it."""
        r = self.settings.cert_radius
        return all(abs(float(other[n]) - float(own_values[n])) <= r * self.w[n] for n in self.eta if n in other)

    # ------------------------------------------------------------------ validation
    def bruteforce(self, held: Mapping, starts: Sequence[Mapping], settings: Optional[FitSettings] = None):
        """The reference for validation: the exact model with every parameter free (the solver at every value of eta
        a fit tries), L-BFGS-B from each start: FitResult."""
        self.use(self.exact_model)
        pset = self.cfg.parameter_set().holding(held)
        st = settings or FitSettings(**{**vars(self.fit_settings), "max_evals": 2000})
        return fit(self.obj, pset, starts=[{k: v for k, v in s.items() if k in pset.names} for s in starts],
                   settings=st)


def exact_config(raw: Mapping, m_lightest: Optional[Mapping] = None) -> dict:
    """A configuration for the exact profile from any campaign configuration: the mixing templates dropped (the
    parameters stay free, the solver at every value) and m_lightest_eV free from M_LIGHTEST_MIN (the mass-sum
    penalty's minimum) when the configuration does not float it; m_lightest overrides its spec."""
    out = copy.deepcopy(dict(raw))
    params = out["parameters"]
    for spec in params.values():
        spec.pop("template", None)
    if "m_lightest_eV" not in params or m_lightest:
        params["m_lightest_eV"] = dict({"start": M_LIGHTEST_MIN, "bounds": [M_LIGHTEST_MIN, 0.1], "space": "linear"},
                                       **(m_lightest or {}))
    return out
