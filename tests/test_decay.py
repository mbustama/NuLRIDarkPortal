"""dp_decay: neutrino decay into the mediator (notes Sec. 12.8 item 2, 2026-10-24)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_decay as dd  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

U = construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0))
M_NU = np.array([1e-6, 8.68e-3, 0.0501])


def _grid(N=120, lo=1e12, hi=1e16):
    return np.geomspace(lo, hi, N)


def test_rates_match_the_formula_and_ignore_a_universal_coupling():
    G = 1e-8 * np.outer([0, 1, 0], [0, 1, 0]).astype(complex)
    Gam, fL = dd.rest_rates(G, U, M_NU, 1e-4)
    Gp = U.conj().T @ G @ U
    mi, mj, M = M_NU[2], M_NU[0], 1e-4
    p = np.sqrt((mi**2 - (mj + M) ** 2) * (mi**2 - (mj - M) ** 2)) / (2 * mi)
    ref = p / (16 * np.pi * mi**2) * (2 * abs(Gp[0, 2]) ** 2 * (mi**2 + mj**2 - 2 * M**2 + (mi**2 - mj**2) ** 2 / M**2)
                                      + 12 * (Gp[0, 2] ** 2).real * mi * mj)
    assert np.isclose(Gam[2, 0], ref, rtol=1e-12)
    assert Gam[0, 1] == Gam[0, 2] == Gam[1, 2] == 0.0          # only downward
    Gk, _ = dd.rest_rates(G + 0.74 * np.eye(3), U, M_NU, 1e-4)  # a universal part changes nothing (to its rounding)
    assert np.allclose(Gk, Gam, rtol=1e-6, atol=0)
    G0, _ = dd.rest_rates(0.74 * np.eye(3), U, M_NU, 1e-20)
    assert not np.any(G0)


def _number(E, F):
    h = np.sqrt(E[1] / E[0])
    return float(np.sum(F.sum(axis=1) * (E * h - E / h)))


def test_number_conserved_and_survival():
    E = _grid()
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1e-20
    fL = np.full((3, 3), 0.9)
    Fn = np.zeros((len(E), 3))
    Fb = np.zeros((len(E), 3))
    Fn[100, 2] = 1.0
    n0 = _number(E, Fn)
    zp1, dt = 1.0, 1e30
    dd.decay_step(E, zp1, dt, Gam, dd.massless_spectra(fL), M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
    lam = Gam[2, 0] * M_NU[2] / E[100] * dt
    assert np.isclose(Fn[100, 2], np.exp(-lam), rtol=1e-12)
    # kept, but for the daughters below the grid: z < E_lo,0 / E_100 (kernels <= 1 per unit z)
    lost = n0 - (_number(E, Fn) + _number(E, Fb))
    z_min = E[0] / np.sqrt(E[1] / E[0]) / E[100]
    assert 0.0 <= lost <= n0 * (-np.expm1(-lam)) * z_min * 1.0001


def test_daughter_spectra():
    E = _grid(N=400)
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1.0
    for fl in (1.0, 0.0, 0.3):
        fL = np.full((3, 3), fl)
        Fn = np.zeros((len(E), 3))
        Fb = np.zeros((len(E), 3))
        j = 399
        Fn[j, 2] = 1.0
        dd.decay_step(E, 1.0, 1e40, Gam, dd.massless_spectra(fL), M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
        h = np.sqrt(E[1] / E[0])
        dEj = E[j] * h - E[j] / h
        # the daughters per bin against the kernels integrated over each bin (parent at E_j)
        lo, hi = np.minimum(E / h, E[j]), np.minimum(E * h, E[j])
        z_lo, z_hi = lo / E[j], hi / E[j]
        ft = 1 - fl
        same = ft * (z_hi - z_lo) + 0.5 * (fl - ft) * (z_hi**2 - z_lo**2)
        other = fl * (z_hi - z_lo) + 0.5 * (ft - fl) * (z_hi**2 - z_lo**2)
        got_same = Fn[:, 0] * (E * h - E / h) / dEj
        got_other = Fb[:, 0] * (E * h - E / h) / dEj
        assert np.allclose(got_same, same, atol=1e-12)
        assert np.allclose(got_other, other, atol=1e-12)


def test_chain_runs_within_a_step():
    E = _grid()
    Gam = np.zeros((3, 3))
    Gam[2, 1] = 1.0
    Gam[1, 0] = 1.0
    fL = np.full((3, 3), 1.0)
    Fn = np.zeros((len(E), 3))
    Fb = np.zeros((len(E), 3))
    Fn[110, 2] = 1.0
    n0 = _number(E, Fn)
    dd.decay_step(E, 1.0, 1e40, Gam, dd.massless_spectra(fL), M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
    assert Fn[:, 1:].sum() + Fb[:, 1:].sum() < 1e-12 * n0
    lost = n0 - (_number(E, Fn) + _number(E, Fb))
    assert lost >= -1e-12 * n0


def test_fold_is_flat():
    E = _grid(N=300)
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1.0
    fL = np.full((3, 3), 0.8)
    F = np.zeros((len(E), 3))
    F[299, 2] = 1.0
    dd.decay_step(E, 1.0, 1e40, Gam, dd.massless_spectra(fL), M_NU, F, F, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, True)
    h = np.sqrt(E[1] / E[0])
    dEj = E[299] * h - E[299] / h
    lo, hi = np.minimum(E / h, E[299]), np.minimum(E * h, E[299])
    flat = (hi - lo) / E[299]
    assert np.allclose(F[:, 0] * (E * h - E / h) / dEj, flat, atol=1e-12)


def test_eigenstate_content():
    """Populations as eigenstates with a mass content: a pure mass-3 eigenstate decays like mass state 3, and the
    daughters are split over the eigenstates by their mass content."""
    E = _grid(N=50)
    Gam = np.zeros((3, 3))
    Gam[2, 0] = 1.0
    fL = np.full((3, 3), 1.0)
    T = np.zeros((len(E), 3, 3))
    T[:, 0, 0] = T[:, 0, 1] = 0.5        # mass state 1 shared by eigenstates 0 and 1
    T[:, 1, 0] = T[:, 1, 1] = 0.5
    T[:, 2, 2] = 1.0
    Fn = np.zeros((len(E), 3))
    Fb = np.zeros((len(E), 3))
    Fn[49, 2] = 1.0
    dd.decay_step(E, 1.0, 1e40, Gam, dd.massless_spectra(fL), M_NU, Fn, Fb, T, T, True, False)
    assert Fn[:, 2].sum() < 1e-12
    assert np.allclose(Fn[:, 0], Fn[:, 1])


def test_relic_composition():
    H0, Om, OL, T0 = 1.4377e-33, 0.315, 0.685, 1.676e-4
    eta = np.ones(3)
    assert np.allclose(dd.relic_composition(np.zeros((3, 3)), M_NU, eta, T0, H0, Om, OL), eta)
    Gam = np.zeros((3, 3))
    Gam[2, 1] = Gam[2, 0] = Gam[1, 0] = 1e-20           # ~1e13 H0 at rest: all decayed long before z = 1
    out = dd.relic_composition(Gam, M_NU, eta, T0, H0, Om, OL)
    assert np.allclose(out, [3.0, 0.0, 0.0], atol=1e-9)
    Gam_slow = Gam * 1e-40                              # nothing decays
    assert np.allclose(dd.relic_composition(Gam_slow, M_NU, eta, T0, H0, Om, OL), eta, atol=1e-9)


def test_jumps_zeno_protected_by_decay_widths():
    """dp_cosmo._soft_jumps with the decay widths (2026-10-24): a pair at frequency omega with a damping D converts at
    Gamma J omega^2 / (omega^2 + D^2); widths Wd add (Wd_a + Wd_b) / 2 to D."""
    from dp_cosmo import _soft_jumps
    E = np.array([1e14])
    J = np.zeros((2, 3, 3))
    J[:, 0, 1] = J[:, 1, 0] = 1.0
    lam = np.zeros((2, 3))
    lam[:, 1] = 2.0 * 1e14 * 1e-20             # omega_01 = 1e-20 eV at zp1 = 1
    zeros = np.zeros((2, 3))
    for wd in (0.0, 3e-20):
        F = np.array([[1.0, 0.0, 0.0]])
        Wd = np.array([[wd, wd, 0.0]])
        rate_dt = 1e-6
        _soft_jumps(E, 1.0, J, zeros, zeros, lam, 0.0, 1.0, 0.0, 1.0, rate_dt, F, Wd, True)
        om, D = 1e-20, wd
        expect = 0.5 * (1.0 - np.exp(-2.0 * rate_dt * om * om / (om * om + D * D))) if wd else 0.5 * (1.0 - np.exp(-2.0 * rate_dt))
        assert np.isclose(F[0, 1], expect, rtol=1e-6)


def test_truth_model_has_no_decay():
    """The projections' Asimov data come from the Standard Model: relic scatterings and decay both off."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import json
    from analysis import config as config_mod
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_ml", "proj_future", "dp_mu_ul.json")
    raw = json.load(open(p))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    truth = cfg._truth_model()
    E, params = __import__("dark_portal_solver").build_solver_params(truth._run_cfg(truth._overrides(
        {q.name: q.start for q in cfg.parameters})))
    assert params["neutrino_decay"] is False and params["relic_scattering"] is False


@pytest.mark.slow     # 30-90 s on CI's runners (2026-10-24): Tycho's suite runs it
def test_solver_uses_the_decayed_relics():
    """propagate_spectrum: where the relics decay, every relic effect sees their composition after the decays (the
    Debye mass and the jumps' rates get eta' = (3, 0, 0)-like)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import json, copy
    import dark_portal_solver as dps
    import dp_absorption
    from analysis import config as config_mod
    p = os.path.join(os.path.dirname(__file__), "..", "analysis", "configs", "campaign_ml", "now_ic26", "dp_mu_ul.json")
    raw = copy.deepcopy(json.load(open(p)))
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, p)
    m = cfg.model()
    v = {q.name: q.start for q in cfg.parameters}
    v.update({"m_Zd_eV": 1e-6, "g_D_trace": 1e-6})
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    params["show_progress"] = False
    params["cosmo_transport"] = "dark_portal"
    seen = []
    orig = dp_absorption.debye_mass_sq
    def spy(G2, m_nu, T, eta, *a, **kw):
        seen.append(np.array(eta, dtype=float))
        return orig(G2, m_nu, T, eta, *a, **kw)
    dp_absorption.debye_mass_sq = spy
    try:
        dps.propagate_weighted(np.asarray(E), params, sources=[np.array([1.0, 2.0, 0.0]) / 3.0])
    finally:
        dp_absorption.debye_mass_sq = orig
    assert seen and all(np.allclose(e, [3.0, 0.0, 0.0], atol=1e-6) for e in seen), seen[:3]


def test_exact_spectra_kinematics():
    """daughter_spectra: the window [z-, z+] of two-body kinematics; averaged over the parent's helicity the
    daughter's mean share is E*/m_i (the rest-frame angle averages out); the massless limit is the earlier kernels."""
    G = 1e-8 * np.array([[0.3, 1, 0.5], [1, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    for ml, M in ((1e-6, 1e-4), (1e-6, 0.03), (0.1, 1e-3), (0.1, 0.008)):
        m = np.sqrt(np.array([0.0, 7.5e-5, 2.5e-3]) + ml * ml)
        Gam, _ = dd.rest_rates(G, U, m, M)
        sp = dd.daughter_spectra(G, U, m, M)
        for i in range(3):
            for j in range(3):
                if Gam[i, j] == 0.0:
                    assert not np.any(sp[:, i, j])
                    continue
                p = np.sqrt((m[i] ** 2 - (m[j] + M) ** 2) * (m[i] ** 2 - (m[j] - M) ** 2)) / (2 * m[i])
                Ej = np.sqrt(p * p + m[j] ** 2)
                zm, zp = sp[0, i, j, 0, :2]
                assert np.isclose(zm, (Ej - p) / m[i], rtol=1e-9, atol=1e-15) and np.isclose(zp, (Ej + p) / m[i])
                mean = 0.0
                for s_ in range(2):
                    tot = 0.0
                    for rel in range(2):
                        _, _, a, b, c = sp[s_, i, j, rel]
                        tot += a * (zp - zm) + 0.5 * b * (zp ** 2 - zm ** 2) + (c * np.log(zp / zm) if c else 0.0)
                        mean += 0.5 * (a * (zp ** 2 - zm ** 2) / 2 + b * (zp ** 3 - zm ** 3) / 3 + c * (zp - zm))
                    assert np.isclose(tot, 1.0, rtol=1e-12)
                assert np.isclose(mean, Ej / m[i], rtol=1e-6)
    m0 = np.array([0.0, 8.68e-3, 0.0501])
    G1 = 1e-8 * np.outer([0, 1, 0], [0, 1, 0]).astype(complex)
    _, fL = dd.rest_rates(G1, U, m0, 1e-6)
    sp, ref = dd.daughter_spectra(G1, U, m0, 1e-6), dd.massless_spectra(fL)
    for i, j in ((1, 0), (2, 0)):
        assert np.allclose(sp[:, i, j], ref[:, i, j], atol=1e-6)


def _brute(E, Gam, spec, Fn, Fb):
    """Every parent bin to every daughter bin by the exact integrals (parent at its bin's centre), all decaying,
    one generation (no chains)."""
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    res = [np.zeros_like(Fn), np.zeros_like(Fb)]
    for s_, F in ((0, Fn), (1, Fb)):
        for b in range(len(E)):
            for i in range(3):
                tot = Gam[i].sum()
                if F[b, i] == 0.0 or tot == 0.0:
                    res[s_][b, i] += F[b, i]
                    continue
                for j in range(3):
                    if Gam[i, j] == 0.0:
                        continue
                    nd = F[b, i] * dE[b] * Gam[i, j] / tot
                    for rel in range(2):
                        t = s_ if rel == 0 else 1 - s_
                        for c in range(b, -1, -1):
                            v = dd._spec_integral(*spec[s_, i, j, rel], E[c] / h / E[b], min(E[c] * h / E[b], 1.0))
                            res[t][c, j] += nd * v / dE[c]
    return res


def test_windowed_kernel_matches_brute_force():
    """decay_step's windows (running moments, exact edge bins) against the direct sum over all bin pairs."""
    rng = np.random.default_rng(3)
    E = _grid(N=150, lo=1e12, hi=3e16)
    G = 1e-9 * np.array([[0.3, 1, 0.5], [1, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    for ml, M in ((1e-6, 1e-4), (1e-6, 0.03), (0.1, 1e-3), (0.1, 0.008)):
        m = np.sqrt(np.array([0.0, 7.5e-5, 2.5e-3]) + ml * ml)
        Gam, _ = dd.rest_rates(G, U, m, M)
        Gam[:2] = 0.0                                         # nu_3 alone decays: one generation
        spec = dd.daughter_spectra(G, U, m, M)
        Fn = np.zeros((len(E), 3))
        Fb = np.zeros((len(E), 3))
        Fn[:, 2] = rng.uniform(0.5, 1.0, len(E))
        Fb[:, 2] = rng.uniform(0.5, 1.0, len(E))
        rn, rb = _brute(E, Gam, spec, Fn, Fb)
        a, b = Fn.copy(), Fb.copy()
        dd.decay_step(E, 1.0, 1e60, Gam, spec, m, a, b, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
        assert np.max(np.abs(a - rn)) < 1e-10 * np.max(rn) and np.max(np.abs(b - rb)) < 1e-10 * np.max(rb)
        # the fold (dp_si's one flux) gets both halves
        F = Fn.copy()
        dd.decay_step(E, 1.0, 1e60, Gam, spec, m, F, F, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, True)
        sp1 = spec.copy()
        sp1[:, :, :, 0, 2:] += sp1[:, :, :, 1, 2:]
        sp1[:, :, :, 1, 2:] = 0.0
        rf, _ = _brute(E, Gam, sp1, Fn, np.zeros_like(Fn))
        assert np.max(np.abs(F - rf)) < 1e-10 * np.max(rf)


def test_dp_si_decay_channel_matches_the_kernel():
    """dp_si's sweep with every collision off: its decay channel (the windows, the own bin in the bin's matrix) moves
    the populations as dp_decay.decay_step's fold, for slow and fast decays and the four kinematic regimes (the
    transport review 2, 2026-10-24)."""
    import dp_si
    G = 1e-9 * np.array([[0.3, 1, 0.5], [1, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    N = 120
    E = np.geomspace(1e12, 3e16, N)
    h = np.sqrt(E[1] / E[0])
    dE = E * (h - 1 / h)
    rng = np.random.default_rng(1)
    n = nr = 3
    C = P = 1
    for ml, M, lam0 in ((1e-6, 1e-4, 1e-4), (1e-6, 0.03, 1e-4), (0.1, 1e-3, 1e-4), (0.1, 0.008, 1e-4), (1e-6, 0.03, 1e3)):
        m = np.sqrt(np.array([0.0, 7.537e-5, 2.511e-3]) + ml * ml)
        Gam, _ = dd.rest_rates(G, U, m, M)
        spec = dd.daughter_spectra(G, U, m, M)
        Gam = Gam * lam0 / np.max(Gam.sum(1) * m / E[0])
        tot = Gam.sum(1)
        dec_rate = np.where(tot > 0, tot * m, 0.0)
        dec_br = np.divide(Gam, tot[:, None], out=np.zeros((3, 3)), where=tot[:, None] > 0)
        spf = np.zeros_like(spec)
        spf[0, :, :, 0, :2] = spec[0, :, :, 0, :2]
        spf[0, :, :, 0, 2:] = spec[0, :, :, 0, 2:] + spec[0, :, :, 1, 2:]
        kf = np.zeros((2, 3, 3, 2, 2), np.int64)
        kx = np.zeros((2, 3, 3, 2, 6), np.int64)
        fx = np.zeros((2, 3, 3, 2, 6))
        dd.window_plan(spf, float(h), N, kf, kx, fx)
        mom = np.stack([spf[0, :, :, 0, 4], spf[0, :, :, 0, 2], spf[0, :, :, 0, 3]], axis=-1)
        own = np.where(kx[0, :, :, 0] == 0, fx[0, :, :, 0], 0.0).sum(-1)
        F0 = (E ** -2.0)[:, None] * rng.uniform(0.5, 1.5, (N, 3))
        F = np.ascontiguousarray(F0[:, :, None].copy())
        z3 = np.zeros((N, nr, P))
        dp_si._si_sweep_cols(F, E, dE, np.zeros((N, n, C)), np.zeros((N, nr, 4)), z3, z3.copy(), z3.copy(),
                             np.zeros(P, np.int64), np.zeros(P), np.zeros(P, np.int64), np.zeros((P, n, nr, n)),
                             np.zeros((P, n, nr, n)), np.zeros((nr, 2, P)), np.zeros((nr, 2), np.int64),
                             np.zeros((N, nr), np.int64), 1.0, np.zeros((N, n, C)), np.zeros((C, 6)), True,
                             dec_rate, dec_br, own, np.ascontiguousarray(kf[0, :, :, 0]),
                             np.ascontiguousarray(kx[0, :, :, 0]), np.ascontiguousarray(fx[0, :, :, 0]),
                             np.ascontiguousarray(mom), 1.0, True)
        Fd = F0.copy()
        sp1 = spec.copy()
        sp1[:, :, :, 0, 2:] += sp1[:, :, :, 1, 2:]
        sp1[:, :, :, 1, 2:] = 0.0
        dd.decay_step(E, 1.0, 1.0, Gam, sp1, m, Fd, Fd, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, True)
        d_si, d_dd = F[:, :, 0] - F0, Fd - F0
        assert np.max(np.abs(d_si - d_dd)) < 1e-6 * np.max(np.abs(d_dd)), (ml, M, lam0)
        assert np.isclose(np.sum(F[:, :, 0] * dE[:, None]), np.sum(Fd * dE[:, None]), rtol=1e-9)
