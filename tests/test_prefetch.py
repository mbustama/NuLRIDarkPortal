"""
Runs in parallel (analysis/model.py FlavorModel.prefetch, analysis/scans.py _Runners; 2026-10-18): an evaluation's
solver runs computed by another model (a worker process in the campaign) and kept by this one give the same fluxes
bit for bit as the runs made one by one, in sky mode (broken power law, Galactic component, detector sites) and not;
the Galactic part is computed once per point; runs already on disk are read, not recomputed; and a global fit from
one start is the same with its runs in parallel.
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from analysis import config as config_mod  # noqa: E402


def _sky_raw():
    from test_gradient_fit import _raw
    return _raw()


def _flux_raw():
    raw = json.load(open(os.path.join(REPO, "analysis", "configs", "pion_2050_case_5_potential_emu_gal.json")))
    raw["solver"]["grid"].update(n_E_bins=24, n_z_bins=40, n_sun_paths=1, n_earth_chords=2)
    return raw


def _executor(other, calls):
    def run(tasks):
        calls.append([t[2] for t in tasks])
        return [other._sky_compute(cfg, g) if kind == "sky" else other._fluxes_compute(cfg, g)
                for kind, cfg, g in tasks]
    return run


def _evaluate(m, points):
    out = []
    for v in points:
        if m.sky:
            out.append(m.band_fluxes(v))
            out.extend(m.site_band_fluxes(v, lat) for lat in m.site_lats)
        else:
            out.append(m.fluxes(v))
            out.append(m.fractions(v))
    return out


def _flat(x):
    if isinstance(x, dict):
        return [a for k in sorted(x) for a in _flat(x[k])]
    if isinstance(x, (tuple, list)):
        return [a for y in x for a in _flat(y)]
    return [] if x is None else [np.asarray(x)]


@pytest.mark.parametrize("mode", ["sky", "flux"])
def test_prefetched_runs_equal_runs_one_by_one(mode, monkeypatch):
    monkeypatch.delenv("NULRI_FLUX_CACHE", raising=False)
    raw = _sky_raw() if mode == "sky" else _flux_raw()
    cfg = config_mod.from_dict(raw)
    if mode == "sky":
        pts = [dict(m_Zd_eV=1e-13, g_D_eps_trace=1e-40, p_index=2.6, p_index_low=1.2),
               dict(m_Zd_eV=3e-13, g_D_eps_trace=1e-39, p_index=2.8, p_index_low=1.6)]
    else:
        names = cfg.free_names
        pts = [{k: v for k, v in dict(m_Zd_eV=1e-13, g_D_eps_u2=1e-40, g_D_eps_trace=1e-40, p_index=2.6).items()
                if k in names}]
    one = cfg.model()
    ref = _evaluate(one, pts)
    par = cfg.model()
    calls = []
    par.executor = _executor(cfg.model(), calls)
    got = _evaluate(par, pts)
    for a, b in zip(_flat(got), _flat(ref)):
        assert np.array_equal(a, b)
    assert calls, "nothing was prefetched"
    assert par.n_solver_calls == one.n_solver_calls
    if one.galactic:
        assert all(sum(c) == 1 for c in calls)          # the Galactic part once per point


def test_prefetch_reads_runs_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("NULRI_FLUX_CACHE", str(tmp_path))
    cfg = config_mod.from_dict(_sky_raw())
    v = dict(m_Zd_eV=1e-13, g_D_eps_trace=1e-40, p_index=2.6, p_index_low=1.2)
    ref = cfg.model().band_fluxes(v)
    par = cfg.model()
    calls = []
    par.executor = _executor(cfg.model(), calls)
    got = par.band_fluxes(v)
    assert not calls and par.n_solver_calls == 0 and par.n_disk_hits > 0
    for a, b in zip(_flat(got), _flat(ref)):
        assert np.array_equal(a, b)


@pytest.mark.slow
def test_global_fit_with_parallel_runs(tmp_path, monkeypatch):
    """One start: the same fit, evaluation for evaluation, with its runs in two worker processes."""
    monkeypatch.delenv("NULRI_FLUX_CACHE", raising=False)
    from analysis import scans
    raw = _sky_raw()
    raw["fit"].update(max_evals=60)
    cfg = config_mod.from_dict(raw)
    r1 = scans.global_fit(cfg, None, workers=1, log=lambda *a: None)
    r2 = scans.global_fit(cfg, None, workers=2, log=lambda *a: None)
    assert r1["chi2"] == r2["chi2"] and r1["values"] == r2["values"] and r1["nfev"] == r2["nfev"]
