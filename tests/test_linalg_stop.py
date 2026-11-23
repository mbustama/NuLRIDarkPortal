"""A linear-algebra failure inside the model stops the run; it is never a wall (2026-11-08).

numpy's LinAlgError is a ValueError subclass, and the objective turns a ValueError into its wall (chi^2 = WALL), which
reads as an excluded point. The objective and the decay-back tables' caches re-raise it as a RuntimeError instead.
"""
import numpy as np
import pytest

from analysis import objective
from src import dp_mediator


class _Model:
    base = {}

    def default(self, name):
        return 0.0


class _Events:
    def __init__(self, err):
        self.err = err

    def chi2(self, values):
        raise self.err

    def chi2_and_grad(self, values, names):
        raise self.err


def test_linalg_error_stops_the_objective():
    obj = objective.Objective(_Model(), None, events=_Events(np.linalg.LinAlgError("Singular matrix")))
    with pytest.raises(RuntimeError, match="linear-algebra failure"):
        obj.chi2({"norm_astro": 1.0})
    with pytest.raises(RuntimeError, match="linear-algebra failure"):
        obj.chi2_and_grad({"norm_astro": 1.0}, ["norm_astro"])


def test_other_value_errors_stay_walls():
    obj = objective.Objective(_Model(), None, events=_Events(ValueError("point outside the solver's domain")))
    assert obj.chi2({"norm_astro": 1.0}) == objective.WALL
    assert obj.chi2_and_grad({"norm_astro": 1.0}, ["norm_astro"])[0] == objective.WALL


@pytest.mark.parametrize("err", [np.linalg.LinAlgError("Singular matrix"), ValueError("bad table")])
def test_dark_portal_tables_failure_stops_the_run(monkeypatch, err):
    def broken(*args, **kwargs):
        raise err
    monkeypatch.setattr(dp_mediator, "Mediator", broken)
    monkeypatch.setattr(dp_mediator, "_CACHE", {})
    with pytest.raises(dp_mediator.DecayBackUnsupported, match="decay-back tables failed"):
        dp_mediator.cached(np.eye(3), np.eye(3), np.array([1e-6, 8.7e-3, 5e-2]), 1e-2, np.geomspace(1e12, 1e16, 5))
    assert issubclass(dp_mediator.DecayBackUnsupported, RuntimeError)
    assert not issubclass(dp_mediator.DecayBackUnsupported, ValueError)
