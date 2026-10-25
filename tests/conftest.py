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
