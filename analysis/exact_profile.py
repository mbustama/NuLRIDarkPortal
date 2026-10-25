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
  per parameter of eta runs at c + h_i (h_i a quarter of the prior's width; each piece's central index node only),
  combined element by element (energy, flavor, band, source) as
      F(eta, p) = F(c, p) exp[sum_i (eta_i - c_i) / h_i log(F_i / F_c)]
  (forward differences, during the search), or with runs at c - h_i too, the parabola in log F through the three
  (central differences, for the certificate; one-sided three-point at a bound). It is the exact model at eta = c for
  every nu (templates.py skips the template runs there: the factor is exactly 1), and its gradient in eta there is the
  differences'. The surrogate chi^2 -- this flux in the analysis' own likelihood, the priors exact -- costs no solver
  run. Its fits (and the centres' nu fits) take central-difference gradients (fitting.FitSettings.fd_central).

  Step. The surrogate is minimized over (eta, nu) with eta in the box |eta_i - c_i| <= Delta w_i (w_i the prior's
  width, Delta the trust radius). If its best value lies less than `stop` below the centre's exact value within a box
  of at least `cert_radius` widths, with central differences, the centre is the profile (certified, below); with
  forward ones the centre is first upgraded to central (its runs reused). Otherwise the step's end is evaluated exactly
  -- it becomes a candidate centre -- and the step accepted when the exact chi^2 decreased: the radius doubles when
  the surrogate predicted the decrease well (rho > 0.75) at the box's edge and halves when it did not (rho < 0.25); a
  rise is rejected and the radius cut to a quarter of the step (a short failed step upgrades the centre to central
  differences: its slope was suspect).

Why the result is exact. (i) Every chi^2 reported is the exact model's at the reported values: the solver at the
reported eta, every index node, nu fitted on those runs. (ii) The certificate: at the reported centre the surrogate
equals the exact chi^2 for every nu and has its gradient in eta up to the central differences' truncation (h^2 / 6
times the third derivative: ~1% of the slope at h = w / 4), and no point of the surrogate within cert_radius widths
lies `stop` below. With the exact chi^2 convex near the centre (the surrogate's curvature along each parameter is the
second difference's), its minimum there lies below the reported value by at most stop plus e^2 / (2 lambda) (e the
gradient's error, lambda the curvature, at least the prior's): ~1e-6 beyond `stop`. (iii) Separate basins -- the two
octants of theta23 under NuFIT, the lightest mass at MeV absorption -- are searched by profile_global (each start its
own trust-region run; the lowest result is the profile).

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
ML_BASIN = 0.3                    # decades: a basin in m_lightest spans at least ~0.3 decade (profile_global)
# the MeV planes' scan of the lightest mass (profile_global): M_LIGHTEST_MIN and 1e-4 .. 0.1 eV, three per decade
# (2026-10-18; five until then): a basin in m_lightest spans at least ~0.3 decade (module docstring of profile_global)
ML_SCAN = (M_LIGHTEST_MIN,) + tuple(float(10.0 ** (-4.0 + k / 3.0)) for k in range(10))


@dataclass
class ExactSettings:
    """Knobs of the exact profile (module docstring). radius, radius_max, cert_radius: trust radii in prior widths;
    fd_rel: the differences' step in prior widths (m_lightest's: M_LIGHTEST_WIDTH); stop: the certified
    predicted decrease; max_centres: exact centres per start; accept: the smallest exact decrease accepted."""
    radius: float = 1.0
    radius_max: float = 64.0
    radius_min: float = 1e-3
    cert_radius: float = 0.5
    # the differences' step (2026-10-18: 0.05, 0.25 until then): small, so that a forward difference's truncation,
    # (h / 2) times the curvature of log F, is small enough for the certificate (trunc_tol); the solver is smooth at
    # this step (its slopes from +-0.05 and +-0.25 widths agree to 1e-6 per width)
    fd_rel: float = 0.05
    # the certified predicted decrease: 1e-3 since 2026-10-18 (1e-4 until then; Tycho job 66462115: the final chi^2
    # moved by at most 5.9e-4 between the two at eight points of every kind)
    stop: float = 1e-3
    max_centres: int = 24
    accept: float = 1e-9
    # the certificate from forward differences (2026-10-18): taken when the curvature of log F measured at the
    # chain's last central certificate (within curv_radius widths of the centre), times trunc_safety, shifts the
    # surrogate's slope by the forward differences' whole truncation and the shifted surrogate still predicts no
    # decrease beyond stop, nor more than trunc_tol beyond the unshifted one; central differences otherwise
    trunc_tol: float = 1e-4
    trunc_safety: float = 2.0
    curv_radius: float = 1.0
    # the lightest mass's global search (profile_global): the exact chi^2 on this grid of m_lightest (eV) at the
    # start's mixing, then the exact profile from every local minimum of the scan within ml_margin of its lowest
    ml_scan: Optional[Sequence[float]] = None
    ml_margin: float = 3.0
    # the scan's mixing (the six; default: the start's): a fixed one makes the scan's solver runs the same for every
    # configuration of a point -- shared through the disk cache (NULRI_FLUX_CACHE) by a plane's nominal and fitted-
    # composition twins and by the data sets of one energy grid
    scan_mixing: Optional[Mapping[str, float]] = None
    # NuFIT's two octants of theta23 (profile_global): the best solution moved into the other octant by the shift
    # between NuFIT's two minima of (sin^2 th23, delta_CP) is evaluated exactly (one evaluation), and profiled from
    # there when it comes within octant_margin of the best
    octants: bool = False
    octant_margin: float = 1.0
    # the fits' gradients (fitting.FitSettings): 'forward' differences of step fit_fd_step (internal units), or
    # 'central' ones -- both small enough that the curvature's bias does not stop L-BFGS-B short
    fit_fd: str = "forward"
    fit_fd_step: float = 1e-5
    # a plane's chain (analysis.scans): a point whose solver runs at the last certified point's values equal that
    # point's (relative inherit_eps per element, from each energy's largest) takes its certificate after one
    # evaluation (ExactProfiler.inherit). 1e-6 since 2026-10-18 (1e-10 and 2.4e-7 in sky mode until then: below the
    # solver's own differences between couplings, it never fired): the error is second order in it (inherit's
    # docstring). With ml_scan, only below a flat scan of the lightest mass (ml_inherit; analysis.scans._run_chain)
    inherit: bool = True
    inherit_eps: float = 1e-6
    inherit_eps_sky: float = 1e-6
    ml_inherit: bool = True
    # validation: at an inherited point also the central certificate from the anchor's values (its predicted
    # decrease is the inheritance's error), recorded as inherit_check -- ~30 solver runs per point
    inherit_check: bool = False
    # the plane's chains along the coupling (analysis.scans._scan): None, each half-column outward from the global best
    # fit one chain (as before); n, cut into chains of at most n points, each started from the global best fit as
    # well as from its predecessor (2026-10-25: with every process on, a fit costs ~30 min, and a 61-point column in
    # one chain ~30 h of wall time whatever the cores; shorter chains run in parallel, the repair passes as before)
    chain_length: Optional[int] = None

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
    # how the point was certified: 'central' or 'forward' differences, 'inherited' (from a chain's anchor), or ''
    cert: str = ""
    # the lightest mass's scan (and a chain's flatness probe): its solver runs and seconds, within the totals
    scan_runs: int = 0
    scan_seconds: float = 0.0
    # validation (ExactSettings.inherit_check): the central certificate's predicted decrease at an inherited point
    inherit_check: Optional[float] = None

    def as_dict(self) -> dict:
        return dict(values=self.values, chi2=self.chi2, certified=self.certified, message=self.message, pred=self.pred,
                    pred_radius=self.pred_radius, centres=self.centres, solver_runs=self.solver_runs,
                    seconds=self.seconds, nfev=self.nfev, scan=self.scan, starts=self.starts, cert=self.cert,
                    scan_runs=self.scan_runs, scan_seconds=self.scan_seconds, inherit_check=self.inherit_check)


def _stack_map(st: Mapping, f) -> dict:
    """f applied to every array of a template stack (FlavorModel._template_stack), site_gal's pairs included."""
    return {k: ({lat: tuple(f(x) for x in pair) for lat, pair in v.items()} if k == "site_gal" else f(v))
            for k, v in st.items()}


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
        self._ml_bounds0 = self.bounds.get("m_lightest_eV")
        fs = dict(vars(cfg.fit))
        # the fits' differences: forward ones of step 1e-5 (internal units; the campaign's 1e-3 stops short in
        # norm_astro, fitting.FitSettings.fd_central) or central ones; ~150 gradients each
        s_ = self.settings
        n = len(cfg.parameters)
        if s_.fit_fd == "central":
            fs.update(fd_central=True, fd_central_step=s_.fit_fd_step, max_evals=150 * (2 * n + 1))
        elif s_.fit_fd == "forward":
            fs.update(fd_central=False, fd_step=s_.fit_fd_step, max_evals=150 * (n + 1))
        else:
            raise ValueError(f"exact_profile.fit_fd must be 'forward' or 'central', got {s_.fit_fd!r}")
        fs.update(polish=False, n_random=0)
        self.fit_settings = FitSettings(**fs)
        self.solver_runs = 0
        # the chain's recent central certificates, newest first: their centres and curvatures (_curvatures), for
        # forward certificates (one per basin: present-day points profile both octants of theta23)
        self.curvs = []
        # models of the current point kept for reuse (scan_runs -> profile_global's scan): {(held, centre): model}
        self._pool = {}
        self._pool_held = None

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
                lo, hi = max(self.bounds[p.name][0], c - radius * w), min(self.bounds[p.name][1], c + radius * w)
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
        certified, message, pred, pred_radius, cert = False, "", np.inf, 0.0, ""
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
                        # forward differences certify when the chain's last measured curvature bounds their
                        # truncation (_forward_certificate); otherwise central ones: the forward ones' truncation
                        # (O(h)) can leave a gradient error that matters where the data pull hard (2026-10-18:
                        # proj_2050 at chi^2 ~ 3000 stalled on it); the centre's runs are reused, the backward added
                        ok, pred2, k = self._forward_certificate(held, centre, vals_c, big, model_c, f_c, pred)
                        nfev += k
                        if ok:
                            certified, cert = True, "forward"
                            message = (f"certified (forward differences): surrogate decrease {pred:.2g} within "
                                       f"{big:.3g} widths, {pred2:.2g} with twice the chain's curvature")
                            break
                        runs += self._runs(model_c)
                        model_c = model_t = self.centre_model(centre, central=True, reuse=model_c)
                        self.use(model_c)
                        central = True
                        continue
                    certified, cert = True, "central"
                    message = f"certified: surrogate decrease {pred:.2g} within {big:.3g} widths"
                    self._store_curvature(model_c, {**vals_c, **held, **centre}, centre)
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
                           seconds=time.perf_counter() - t0, nfev=nfev, cert=cert)

    # ------------------------------------------------------------------ forward certificates (2026-10-18)
    def _curvatures(self, model, values) -> Optional[dict]:
        """From a central model's template runs: h^2 times the second derivative in each parameter of eta of every
        stacked array (the log ratios to the centre's run, the Galactic differences), {(kind, piece): {array:
        (n_eta, ...)}} (site_gal: {lat: (nu, nb)}); None when a parameter has no second difference."""
        offs, modes, n = [], [], 0
        for t in model.templates.values():
            if t.slope is not None or len(t.points) != 2:
                return None
            a, b = t.points
            offs.append(n)
            modes.append((a - t.theta0) * (b - t.theta0) < 0.0)   # (c + h, c - h); else (c +- h, c +- 2h)
            n += 2

        def second(arr):
            return np.stack([arr[o] + arr[o + 1] if c else arr[o + 1] - 2.0 * arr[o] for o, c in zip(offs, modes)])
        out = {}
        for kind in (("flux", "sky") if model.sky else ("flux",)):
            for p, (v, nodes) in enumerate(model._pieces(values)):
                out[(kind, p)] = _stack_map(model._template_stack(v, nodes, kind), second)
        return out

    def _store_curvature(self, model, values, centre) -> None:
        """Keeps a central certificate's curvatures for the forward certificates after it (a chain's next points)."""
        K = self._curvatures(model, values)
        if K is None:
            return
        c = {n: float(centre[n]) for n in self.eta}
        # the newest first; an older one within 0.1 widths of this centre is replaced, at most four kept
        keep = [x for x in self.curvs if max(abs(x["centre"][n] - c[n]) / self.w[n] for n in self.eta) > 0.1]
        self.curvs = [dict(centre=c, K=K)] + keep[:3]

    def _forward_certificate(self, held, centre, vals_c, radius, model, f_c, pred):
        """Whether a forward model's certificate holds (ExactSettings.trunc_tol): a forward difference of step h
        carries a truncation of (h / 2) times the curvature of log F in its slope (the Galactic flux's, likewise).
        With the curvature K measured at the chain's nearest recent central certificate (within curv_radius widths), the
        surrogate is minimized again with every slope shifted by trunc_safety times that truncation (the stacked
        log ratios by -trunc_safety K / 2). Both surrogates -- the unshifted one is the truncation's other end --
        predicting no decrease beyond stop certifies every truncation between them (the predicted decrease is
        convex in the slopes), and the shift must change it by at most trunc_tol. Returns (ok, the shifted
        prediction, chi^2 evaluations)."""
        s = self.settings
        # the nearest recent curvature within curv_radius widths (the newest among equals)
        dist = [max(abs(float(centre[n]) - x["centre"][n]) / self.w[n] for n in self.eta) for x in self.curvs]
        near = [i for i, d in enumerate(dist) if d <= s.curv_radius]
        if not near:
            return False, None, 0
        c = self.curvs[min(near, key=lambda i: (dist[i], i))]
        m = -0.5 * s.trunc_safety
        model.stack_shift = {key: _stack_map(K, lambda a: m * a) for key, K in c["K"].items()}
        try:
            r = self._surrogate_step(held, centre, vals_c, radius)
        finally:
            model.stack_shift = None
        pred2 = f_c - float(r.chi2)
        return bool(pred2 <= s.stop and pred2 - pred <= s.trunc_tol), pred2, int(r.nfev)

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

    def _pooled(self, held: Mapping, centre: Mapping):
        """The model centred at `centre` for the point `held`: kept from scan_runs at the same point, else new."""
        hk = tuple(sorted((k, float(v)) for k, v in held.items()))
        if hk != self._pool_held:
            self._pool, self._pool_held = {}, hk
        ck = tuple(sorted((k, float(v)) for k, v in centre.items()))
        m = self._pool.get(ck)
        if m is None:
            m = self._pool[ck] = self.centre_model(centre)
        return m

    def value_at(self, held: Mapping, eta: Mapping, start: Mapping):
        """The exact chi^2 at eta, minimized over nu: (chi2, values, solver runs made)."""
        centre = {n: float(min(max(float(eta[n]), self.bounds[n][0]), self.bounds[n][1])) for n in self.eta}
        model = self._pooled(held, centre)
        n0 = self._runs(model)
        self.use(model)
        try:
            f, v, _ = self._centre_value(held, centre, start)
        finally:
            self.use(self.exact_model)
        return f, v, self._runs(model) - n0

    # ------------------------------------------------------------------ where the model holds (2026-10-24)
    def lightest_cap(self, held: Mapping) -> Optional[float]:
        """The largest lightest mass at which the plane's model is perturbative at the point held (Mauricio
        2026-10-24: the fits use only the lightest masses where the model holds; analysis.exclusion.not_allowed_boundary
        per lightest mass, the line log10 g = log10 m_Zd + c(m_lightest), c falling with the mass): None where every
        mass in the bounds is allowed (or the point is not one of a plane with that boundary), 0.0 where none is."""
        if self._ml_bounds0 is None or "m_lightest_eV" not in self.eta or not self.cfg.planes:
            return None
        pl = self.cfg.planes[0]
        if pl.x not in held or pl.y not in held or held[pl.x] <= 0.0 or held[pl.y] <= 0.0:
            return None
        tab = getattr(self, "_cap_table", None)
        if tab is None:
            from .exclusion import not_allowed_boundary
            lo, hi = self._ml_bounds0
            nodes = np.geomspace(max(lo, M_LIGHTEST_MIN), hi, 25)
            cs = []
            for m in nodes:
                # from the loaded configuration itself (no file needed); any error raises (2026-10-25, the review:
                # a missing path or a swallowed exception switched decision A off silently)
                b = not_allowed_boundary(self.cfg, float(m))
                if b is None or b[0] != 1.0:
                    cs = None               # a plane whose coupling does not set G linearly: no such boundary
                    break
                cs.append(float(b[1]))
            tab = self._cap_table = False if cs is None else (np.log10(nodes), np.array(cs))
        if tab is False:
            return self._measure_cap(held)
        lm, cs = tab
        t = float(np.log10(held[pl.y]) - np.log10(held[pl.x]))
        if t < cs[-1]:
            return None
        if t >= cs[0]:
            return 0.0
        k = int(np.nonzero(cs > t)[0][-1])
        f = (cs[k] - t) / (cs[k] - cs[k + 1])
        return float(10.0 ** (lm[k] + f * (lm[k + 1] - lm[k])))

    def _measure_cap(self, held: Mapping) -> Optional[float]:
        """lightest_cap for a plane whose not-modelled region is a measure per point rather than a line (2026-10-25:
        the secret-interaction scalar's relic field, analysis.exclusion.not_modelled_measure; Mauricio's decision
        (a)): the largest lightest mass whose measure stays below 1 (the measure grows with the mass: heavier relics
        carry more scalar density), interpolated in log between 25 nodes; None where every mass is modelled, 0.0
        where none is."""
        f = getattr(self, "_cap_measure", None)
        if f is None:
            from .exclusion import not_modelled_measure
            f = self._cap_measure = not_modelled_measure(self.cfg) or False
        if f is False:
            return None
        pl = self.cfg.planes[0]
        if pl.x not in held or pl.y not in held or held[pl.x] <= 0.0 or held[pl.y] <= 0.0:
            return None
        lo, hi = self._ml_bounds0
        nodes = np.geomspace(max(lo, M_LIGHTEST_MIN), hi, 25)
        lx = np.log10(nodes)
        # the largest mass first: below 1 there, every mass is modelled (one evaluation, most of the plane)
        top = f(held[pl.x], held[pl.y], float(nodes[-1]))
        if top < 1.0:
            return None
        X = np.array([f(held[pl.x], held[pl.y], float(m)) for m in nodes[:-1]] + [top])
        if X[0] >= 1.0:
            return 0.0
        k = int(np.nonzero(X < 1.0)[0][-1])
        lx0, lx1 = np.log10(X[k]), np.log10(X[k + 1])
        fr = -lx0 / (lx1 - lx0) if lx1 > lx0 else 0.0
        return float(10.0 ** (lx[k] + fr * (lx[k + 1] - lx[k])))

    def set_lightest_cap(self, cap: Optional[float]) -> None:
        """Bound m_lightest above by cap for the point to come (None: its configured bounds)."""
        if self._ml_bounds0 is None:
            return
        lo, hi = self._ml_bounds0
        self.bounds["m_lightest_eV"] = (lo, hi if cap is None else min(hi, max(float(cap), lo)))

    def _ml_grid(self) -> List[float]:
        lo, hi = self.bounds["m_lightest_eV"]
        return sorted({float(min(max(m, lo), hi)) for m in self.settings.ml_scan})

    def scan_runs(self, held: Mapping, mixing: Mapping, start: Mapping):
        """The solver runs of the lightest mass's scan at the point held: at `mixing` (the six) and every mass of
        ml_scan, every index node -- ({m: runs}, solver runs made, seconds). The models stay for profile_global's
        scan at the same point (its centres are the same when ExactSettings.scan_mixing is set)."""
        t0 = time.perf_counter()
        out, k = {}, 0
        held = {kk: float(v) for kk, v in held.items()}
        for m in self._ml_grid():
            centre = {n: float(mixing[n]) for n in self.eta if n != "m_lightest_eV"}
            centre["m_lightest_eV"] = m
            model = self._pooled(held, centre)
            n0 = self._runs(model)
            out[m] = self._runs_at(model, held, dict(start, **centre))
            k += self._runs(model) - n0
        return out, k, time.perf_counter() - t0

    def runs_differ(self, runs: Sequence, ref: Sequence, sky: bool) -> float:
        """The largest difference between two sets of solver runs, per element relative to each energy's largest
        (inf when their shapes differ)."""
        worst = 0.0
        if len(runs) != len(ref):
            return np.inf
        for a, b in zip(runs, ref):
            if a.shape != b.shape:
                return np.inf
            scale = np.maximum(np.max(np.abs(b), axis=-1, keepdims=True), 1e-300)
            worst = max(worst, float(np.max(np.abs(a - b) / scale)))
        return worst

    def scan_flat(self, runs: Mapping, ref: Mapping):
        """Whether a point's lightest-mass scan (scan_runs) equals the reference's (the plane's point without new
        physics) to inherit_eps at every mass: (flat, the largest difference)."""
        s = self.settings
        eps = s.inherit_eps_sky if self.exact_model.sky else s.inherit_eps
        worst = 0.0
        for m, r in ref.items():
            if m not in runs:
                return False, np.inf
            worst = max(worst, self.runs_differ(runs[m], r, self.exact_model.sky))
            if worst > eps:
                return False, worst
        return True, worst

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
        starts, scan, runs, probes = [dict(st)], [], 0, []
        scan_runs, scan_sec = 0, 0.0
        if s.ml_scan is not None and "m_lightest_eV" in self.eta:
            t_scan = time.perf_counter()
            grid = self._ml_grid()
            base = dict(st, **{k: float(v) for k, v in (s.scan_mixing or {}).items()})
            for m in grid:
                f, v, k = self.value_at(held, dict(base, m_lightest_eV=m), base)
                runs += k
                scan_runs += k
                scan.append(dict(m_lightest_eV=m, chi2=f, values=v))
                if log is not None:
                    log(f"    scan m_lightest {m:.3g}: chi2 {f:.6f}")
            scan_sec = time.perf_counter() - t_scan
            vals = [x["chi2"] for x in scan]
            low = min(vals)
            cand = [i for i in range(len(grid)) if vals[i] <= low + s.ml_margin
                    and (i == 0 or vals[i] <= vals[i - 1]) and (i == len(grid) - 1 or vals[i] <= vals[i + 1])]
            # each basin from the start's mixing (a chain's predecessor's) and the scan's lightest mass and nu
            starts = [dict(scan[i]["values"], **{n: st[n] for n in self.eta if n != "m_lightest_eV"})
                      for i in sorted(cand, key=lambda i: vals[i])]
        res = [self.profile(held, x, log=log) for x in starts]
        if s.octants and "s23sq" in self.eta:
            b = min(res, key=lambda r: r.chi2)
            ds23, ddcp = self._octant_shift(b.values)
            other = dict(b.values)
            other["s23sq"] = float(min(max(b.values["s23sq"] + ds23, self.bounds["s23sq"][0]), self.bounds["s23sq"][1]))
            if "d_cp_deg" in self.eta:
                other["d_cp_deg"] = float(min(max(b.values["d_cp_deg"] + ddcp, self.bounds["d_cp_deg"][0]),
                                              self.bounds["d_cp_deg"][1]))
            other = {k: v for k, v in other.items() if k not in held}
            f_o, v_o, k = self.value_at(held, other, other)
            runs += k
            probes.append(dict(octant_probe={n: other[n] for n in ("s23sq", "d_cp_deg") if n in other}, chi2=f_o))
            if log is not None:
                log(f"    other octant probe: chi2 {f_o:.6f} (best {b.chi2:.6f})")
            if f_o < b.chi2 + s.octant_margin:
                res.append(self.profile(held, {k: v for k, v in v_o.items() if k not in held}, log=log))
        best = min(res, key=lambda r: r.chi2)
        best.starts = [dict(start={n: r.centres[0]["eta"][n] for n in self.eta}, chi2=r.chi2, certified=r.certified,
                            centres=len(r.centres), solver_runs=r.solver_runs, message=r.message) for r in res] + probes
        best.scan = [dict(m_lightest_eV=x["m_lightest_eV"], chi2=x["chi2"]) for x in scan]
        best.solver_runs = runs + sum(r.solver_runs for r in res)
        best.nfev = sum(r.nfev for r in res)
        best.seconds = time.perf_counter() - t0
        best.scan_runs, best.scan_seconds = scan_runs, scan_sec
        if len(res) > 1:
            best.message += f"; lowest of {len(res)} starts" + (f" after a {len(scan)}-point m_lightest scan" if scan
                                                                 else "")
        return best

    def _octant_shift(self, values: Mapping):
        """(d sin^2 th23, d delta_CP) taking a solution into the other octant: the shift between the penalty's two
        minima of (sin^2 th23, delta_CP) (NuFIT 6.1's T23_DCP table: (0.47, -150 deg) and (0.55, -175 deg) in the
        normal ordering), or the mirror about maximal mixing without that table."""
        from .nufit import NuFitPairwisePrior, SUFFIX, TABLES
        pri = [p for p in self.cfg.priors if isinstance(p, NuFitPairwisePrior)]
        s23 = float(values["s23sq"])
        if not pri:
            return 1.0 - 2.0 * s23, 0.0
        key = ("T23_DCP", pri[0].ordering)
        mins = getattr(self, "_octant_minima", {}).get(key)
        if mins is None:
            with np.load(TABLES) as z:
                x, y, c = (np.array(z[f"T23_DCP_{SUFFIX[pri[0].ordering]}_{k}"]) for k in ("x", "y", "chi2"))
            out = []
            for sel in (x < 0.5, x > 0.5):
                cc = np.where(sel[:, None], c, np.inf)
                i, j = np.unravel_index(np.argmin(cc), cc.shape)
                out.append((float(x[i]), float(y[j])))
            mins = out
            self._octant_minima = {**getattr(self, "_octant_minima", {}), key: mins}
        (lo_s, lo_d), (hi_s, hi_d) = mins
        if s23 < 0.5:
            return hi_s - lo_s, hi_d - lo_d
        return lo_s - hi_s, lo_d - hi_d

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

    def inherit(self, held: Mapping, anchor: dict, scan_pending: bool = False) -> Optional["ExactResult"]:
        """The point at `held` profiled from a chain's anchor when new physics is below the solver's reach there:
        its runs at the anchor's values (one evaluation) equal the anchor's to inherit_eps (per element, relative
        to each energy's largest). Its chi^2 is then the anchor's function of every parameter plus a perturbation
        delta whose size is O(inherit_eps) times chi^2's sensitivity to the flux: the anchor's minimum moves by
        first order in delta's gradient, and the value there by its square over chi^2's curvature -- second order
        (with inherit_eps = 1e-6, below 1e-6 in chi^2: ExactSettings.inherit_check measures it). The point is
        evaluated exactly at the anchor's eta, nu refitted, so the reported chi^2 is exact there. Returns the result
        (certified by the anchor), or None (profile it: new physics shows). With the lightest mass scanned
        (ml_scan) only with scan_pending: the chain confirms afterwards that the scan's runs are those without new
        physics at every mass (analysis.scans._run_chain; scan_flat)."""
        s = self.settings
        if not s.inherit or (s.ml_scan is not None and not (s.ml_inherit and scan_pending)):
            return None
        t0 = time.perf_counter()
        held = {k: float(v) for k, v in held.items()}
        eta = {n: float(anchor["values"][n]) for n in self.eta}
        if any(not self.bounds[n][0] <= eta[n] <= self.bounds[n][1] for n in self.eta):
            return None                     # the anchor's solution is outside this point's bounds (lightest_cap)
        model = self.centre_model(eta)
        try:
            runs = self._runs_at(model, held, anchor["values"])
        except ValueError:
            return None
        eps = s.inherit_eps_sky if model.sky else s.inherit_eps
        worst = self.runs_differ(runs, anchor["runs"], model.sky)
        if not worst <= eps:
            return None
        self.use(model)
        try:
            f, vals, k = self._centre_value(held, eta, anchor["values"])
        finally:
            self.use(self.exact_model)
        values = dict(vals)
        values.update(eta)
        n_runs = self._runs(model)
        check, msg = None, ""
        if s.inherit_check:
            if "check" not in anchor:
                anchor["check"] = self._certificate_at(anchor["held"], eta, anchor["values"])
            p_point, k_runs = self._certificate_at(held, eta, values)
            check = p_point - anchor["check"][0]
            n_runs += k_runs + anchor["check"][1]
            anchor["check"] = (anchor["check"][0], 0)
            msg = f"; check: central certificate {p_point:.2g} here, {anchor['check'][0]:.2g} at the anchor"
        return ExactResult(values=values, chi2=float(f), certified=True,
                           message=f"inherited: runs within {worst:.2g} of the anchor at {anchor['held']}" + msg,
                           pred=0.0, pred_radius=0.0, centres=[dict(eta=eta, chi2=float(f), accepted=True)],
                           solver_runs=n_runs, seconds=time.perf_counter() - t0, nfev=k, cert="inherited",
                           inherit_check=check)

    def _certificate_at(self, held: Mapping, eta: Mapping, start: Mapping):
        """Validation: the central certificate's predicted decrease at eta for the point `held` (the surrogate with
        central differences minimized within max(radius, cert_radius) widths): (prediction, solver runs)."""
        s = self.settings
        model = self.centre_model(eta, central=True)
        self.use(model)
        try:
            f, vals, _ = self._centre_value(held, eta, start)
            r = self._surrogate_step(held, eta, vals, max(s.radius, s.cert_radius))
        finally:
            self.use(self.exact_model)
        return f - float(r.chi2), self._runs(model)

    def within_certificate(self, own_values: Mapping, other: Mapping) -> bool:
        """Whether `other`'s mixing lies within a certified point's certificate box (cert_radius widths of its own):
        a repair need not try it. The lightest mass must also lie within ML_BASIN decades (2026-10-21): chi^2 can have
        separate basins in it, at the lower bound and near 1e-4 eV, that a box a prior width across in eV does not tell
        apart -- ultralight points stuck at 1e-6 eV with chi^2 up to 335 too high were never tried at their
        neighbours' 1e-4 eV."""
        r = self.settings.cert_radius
        for n in self.eta:
            if n not in other:
                continue
            a, b = float(other[n]), float(own_values[n])
            if abs(a - b) > r * self.w[n]:
                return False
            if n == "m_lightest_eV" and abs(np.log10(max(a, M_LIGHTEST_MIN) / max(b, M_LIGHTEST_MIN))) > ML_BASIN:
                return False
        return True

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
