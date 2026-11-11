"""Test-wide settings (2026-10-25): the extragalactic beam's crossing of the Galaxy (dark_portal_solver.
EXTRAGALACTIC_CROSSING) costs every sub-keV run a sweep of the Galaxy's lines of sight. The tests CI runs (not marked
slow) run without it -- through NULRI_CROSSING=0 too, which dark_portal_solver reads at import, so that the worker
processes a test spawns inherit it; the slow tests (Tycho's suite, slurm/test.sh) keep it and cover its code paths
(tests/test_galactic_path.py)."""
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _crossing_only_in_slow_tests(request, monkeypatch):
    on = request.node.get_closest_marker("slow") is not None
    monkeypatch.setenv("NULRI_CROSSING", "1" if on else "0")
    dps = sys.modules.get("dark_portal_solver")
    if dps is not None:
        monkeypatch.setattr(dps, "EXTRAGALACTIC_CROSSING", on)
    yield


@pytest.fixture(scope="session", autouse=True)
def _future_blocks(tmp_path_factory):
    """The forecasts' detector responses (C1, 2026-11-11; IceCubeRates/data/future_planA) are large and kept outside
    the repository: where they are missing, synthetic ones on the solver's 20 declination bands stand in
    (test_future_detectors.write_synthetic_cards), so every test that builds a forecast model runs."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (repo, os.path.join(repo, "src"), os.path.join(repo, "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from analysis import future_events as FE
    if all(os.path.exists(os.path.join(FE.DATA_DIR, f"{c}.pkl")) for c in FE.CARDS):
        yield
        return
    import numpy as np
    from test_future_detectors import write_synthetic_cards
    path = tmp_path_factory.mktemp("future_blocks_synthetic")
    write_synthetic_cards(path, sin=np.linspace(-1.0, 1.0, 21))
    old, FE.DATA_DIR = FE.DATA_DIR, str(path)
    yield
    FE.DATA_DIR = old
