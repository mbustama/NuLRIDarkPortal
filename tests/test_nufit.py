"""
NuFIT 6.1's pairwise Delta chi^2 as the penalty on the mixing parameters (analysis/nufit.py, 2026-10-18): the splines
reproduce the tables, stay non-negative and smooth across delta_CP = +-180 deg, the conventions (delta_CP's range,
the inverted ordering's negative DMA axis, sin^2 or degrees), the ordering's offset counted once, and the
configuration checks (prior type, ordering, bounds inside the tables).
"""
import copy
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from analysis import nufit  # noqa: E402
from analysis.nufit import TABLE_NAMES, NuFitPairwisePrior  # noqa: E402

ORDERINGS = ("normal", "inverted")


def _base(ordering="normal", **mix):
    sm = {"th12_deg": 33.7, "th13_deg": 8.6, "th23_deg": 43.3, "d_cp_deg": 212.0, "dm21_sq": 7.5e-5,
          "dm31_sq": 2.5e-3}
    sm.update(mix)
    return {"standard_mixing": sm, "cosmology": {"mass_ordering": ordering}}


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("name", TABLE_NAMES)
def test_splines_reproduce_the_tables(name, ordering):
    """At every table node the spline is NuFIT's value above the ordering's minimum (to 1e-8); the shift that keeps
    it non-negative is small."""
    spline, floor, offset, ((x0, x1), (y0, y1)) = nufit.surface(name, ordering)
    with np.load(nufit.TABLES) as z:
        sfx = nufit.SUFFIX[ordering]
        x, y, c = (z[f"{name}_{sfx}_{k}"] for k in ("x", "y", "chi2"))
    X, Y = np.meshgrid(x, y, indexing="ij")
    assert np.max(np.abs(spline.ev(X, Y) - (c - offset))) < 1e-8
    assert offset == pytest.approx(0.0 if ordering == "normal" else 5.908214, abs=1e-6)
    assert -0.05 < floor <= 0.0, floor


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_penalty_is_non_negative_and_vanishes_at_the_best_fit(ordering):
    """Random points across every table: Delta chi^2 >= 0. At the tables' minima the whole penalty is the splines'
    small shifts, not the inverted ordering's offset three times."""
    rng = np.random.default_rng(1)
    for name in TABLE_NAMES:
        _, _, _, ((x0, x1), (y0, y1)) = nufit.surface(name, ordering)
        for x, y in zip(rng.uniform(x0, x1, 3000), rng.uniform(y0, y1, 3000)):
            assert nufit.delta_chi2(name, ordering, x, y) >= 0.0
    prior = NuFitPairwisePrior(ordering)
    best = prior.best_fit()
    shifts = -sum(nufit.surface(n, ordering)[1] for n in TABLE_NAMES)
    assert prior.chi2_values(best, _base(ordering)) == pytest.approx(shifts, abs=1e-8)
    assert shifts < 0.1


def test_conventions():
    """delta_CP over 0..360 (the solver's) or any turn of it reads NuFIT's -180..180; the inverted ordering's
    dm31_sq (= |Delta m^2_32|, positive) reads NuFIT's negative axis; angles in degrees equal sin^2."""
    no, io = NuFitPairwisePrior("normal"), NuFitPairwisePrior("inverted")
    b = no.best_fit()
    assert b["d_cp_deg"] == pytest.approx(210.0) and b["s23sq"] == pytest.approx(0.47)
    base = _base()
    t = no.terms(b, base)
    for d in (210.0 - 360.0, 210.0 + 360.0, 210.0 + 720.0):
        assert no.terms(dict(b, d_cp_deg=d), base)["T23_DCP"] == pytest.approx(t["T23_DCP"], abs=1e-12)
    bi = io.best_fit()
    assert bi["dm31_sq"] == pytest.approx(2.485e-3) and bi["d_cp_deg"] == pytest.approx(275.0)
    assert io.terms(bi, _base("inverted"))["DMS_DMA"] < 0.05
    assert io.terms(dict(bi, dm31_sq=2.51e-3), _base("inverted"))["DMS_DMA"] > 1.0
    deg = {"th12_deg": 33.5, "th13_deg": 8.55, "th23_deg": 44.0}
    s2 = {k: float(np.sin(np.radians(v)) ** 2) for k, v in zip(("s12sq", "s13sq", "s23sq"), deg.values())}
    assert no.chi2_values(deg, base) == pytest.approx(no.chi2_values(s2, base), abs=1e-12)
    assert no.chi2_values({}, _base(**deg)) == pytest.approx(no.chi2_values(s2, base), abs=1e-12)


@pytest.mark.parametrize("ordering", ORDERINGS)
def test_smooth_across_180_degrees(ordering):
    """delta_CP's table is padded periodically: value and slope agree on both sides of +-180 deg."""
    for s23 in (0.45, 0.47, 0.55, 0.6):
        def f(d):
            return nufit.delta_chi2("T23_DCP", ordering, s23, d)
        h = 1e-3
        assert f(180.0 - 1e-9) == pytest.approx(f(-180.0 + 1e-9), abs=1e-6)
        left = (f(180.0) - f(180.0 - h)) / h
        right = (f(-180.0 + h) - f(-180.0)) / h
        assert left == pytest.approx(right, abs=1e-2 * max(1.0, abs(left))), (s23, left, right)


def test_aliases_in_the_objective_and_the_model():
    """A Gaussian prior on s12sq sees sin^2 whether the fit sets s12sq, th12_deg or neither (the model's default)."""
    from test_analysis import _tiny_raw
    from analysis import config as config_mod
    cfg = config_mod.from_dict(_tiny_raw())
    obj = cfg.objective()
    th = float(obj.model.base["standard_mixing"]["th12_deg"])
    assert obj.model.default("s12sq") == pytest.approx(np.sin(np.radians(th)) ** 2, rel=1e-14)
    assert obj._quantity("s12sq", {"th12_deg": 34.0}) == pytest.approx(np.sin(np.radians(34.0)) ** 2, rel=1e-14)
    assert obj._quantity("s12sq", {"s12sq": 0.31}) == 0.31


def _nufit_raw(ordering="normal", prior_ordering=None, s23_bounds=(0.3, 0.7)):
    from test_analysis import _tiny_raw
    raw = _tiny_raw()
    raw["solver"].setdefault("cosmology", {})["mass_ordering"] = ordering
    raw["parameters"]["s23sq"] = {"start": 0.5, "bounds": list(s23_bounds)}
    raw["priors"] = [{"type": "nufit_6.1_pairwise", "ordering": prior_ordering or ordering}]
    return raw


def test_configuration():
    """The prior from a configuration enters the objective's priors; it must name the solver's ordering, and the
    fit's bounds must stay inside the tables; it changes the configuration's identity."""
    from analysis import config as config_mod
    cfg = config_mod.from_dict(_nufit_raw())
    obj = cfg.objective()
    (p,) = [q for q in obj.priors if isinstance(q, NuFitPairwisePrior)]
    v = {"s23sq": 0.55}
    assert obj.prior_chi2(v) == pytest.approx(p.chi2_values(v, obj.model.base) + sum(
        q.chi2(obj._quantity(q.name, v)) for q in obj.priors if q is not p))
    assert obj.prior_chi2({"s23sq": 0.62}) > obj.prior_chi2({"s23sq": 0.47}) + 4.0
    with pytest.raises(ValueError, match="ordering"):
        config_mod.from_dict(_nufit_raw("normal", prior_ordering="inverted"))
    with pytest.raises(ValueError, match="outside"):
        config_mod.from_dict(_nufit_raw(s23_bounds=(0.2, 0.7)))
    with pytest.raises(ValueError):
        config_mod.from_dict({**_nufit_raw(), "priors": [{"type": "nufit_6.1_pairwise"}]})
    raw = _nufit_raw()
    other = copy.deepcopy(raw)
    other["priors"] = []
    assert config_mod.from_dict(raw).identity() != config_mod.from_dict(other).identity()
    io = config_mod.from_dict(_nufit_raw("inverted"))
    assert io.objective().prior_chi2({}) > 0.0
