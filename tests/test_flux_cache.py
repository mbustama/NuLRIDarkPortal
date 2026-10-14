"""
The analysis model's solver-run cache on disk (NULRI_FLUX_CACHE, 2026-10-14): shared by every process, keyed by the
full solver configuration, the flux mode and the solver's code digest, written atomically and read back bit for bit.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis import config as config_mod  # noqa: E402
from analysis import model as model_mod  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs")


def _cfg(name="pion_2050_case_5_si_e_galx", n_E=40):
    raw = json.load(open(os.path.join(CONFIGS, name + ".json")))
    raw["solver"]["grid"].update(n_E_bins=n_E)
    return config_mod.from_dict(raw, "")


def _files(d):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".npz"))


def test_disk_cache_is_shared_and_exact(tmp_path, monkeypatch):
    monkeypatch.setenv(model_mod.DISK_CACHE_ENV, str(tmp_path))
    cfg = _cfg()
    v = dict(m_phi_eV=5.6e6, g_phi=0.1, p_index=2.4, f_e_S=0.3, f_e_S_gal=0.6, norm_astro=1.2, norm_gal=0.9)
    a = cfg.model()
    fa = a.fractions(v)
    Ea, Fa = a.flux(v)
    assert a.n_solver_calls == len(cfg.p_nodes()) and a.n_disk_hits == 0
    assert len(_files(tmp_path)) == len(cfg.p_nodes())
    b = cfg.model()                                   # another process: nothing in memory
    fb = b.fractions(v)
    Eb, Fb = b.flux(v)
    assert b.n_solver_calls == 0 and b.n_disk_hits == len(cfg.p_nodes())
    assert np.array_equal(fa, fb) and np.array_equal(Fa, Fb) and np.array_equal(Ea, Eb)
    # another point: new files; the nuisances alone: none
    b.fractions(dict(v, g_phi=0.2))
    b.fractions(dict(v, norm_astro=2.0, f_e_S=0.5))
    assert len(_files(tmp_path)) == 2 * len(cfg.p_nodes())


def test_disk_cache_keys_on_code_and_mode_and_survives_damage(tmp_path, monkeypatch):
    monkeypatch.setenv(model_mod.DISK_CACHE_ENV, str(tmp_path))
    cfg_x, cfg_n = _cfg("pion_2050_case_5_si_e_galx"), _cfg("pion_2050_case_5_si_e_gal")
    v = dict(m_phi_eV=5.6e6, g_phi=0.1, p_index=2.5)
    mx = cfg_x.model()
    mx.fractions(v)
    n0 = len(_files(tmp_path))
    mn = cfg_n.model()                                # same physics, other mode (fixed composition): own files
    mn.fractions(v)
    assert mn.n_disk_hits == 0 and len(_files(tmp_path)) == 2 * n0
    import dp_core
    monkeypatch.setattr(dp_core, "_source_digest", lambda: "other code")
    mc = cfg_x.model()
    mc.fractions(v)
    assert mc.n_disk_hits == 0 and len(_files(tmp_path)) == 3 * n0
    monkeypatch.undo()
    monkeypatch.setenv(model_mod.DISK_CACHE_ENV, str(tmp_path))
    for f in _files(tmp_path):                        # damage every file: recomputed, rewritten, same numbers
        with open(f, "wb") as fh:
            fh.write(b"not an npz")
    md = cfg_x.model()
    f_damaged = md.fractions(v)
    assert md.n_solver_calls == len(cfg_x.p_nodes())
    assert np.array_equal(f_damaged, mx.fractions(v))
    me = cfg_x.model()
    me.fractions(v)
    assert me.n_disk_hits == len(cfg_x.p_nodes())


def test_without_the_variable_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv(model_mod.DISK_CACHE_ENV, raising=False)
    m = _cfg().model()
    assert m.disk_cache is None
    m.fractions(dict(m_phi_eV=5.6e6, g_phi=0.1, p_index=2.5))
    assert m.n_disk_hits == 0
