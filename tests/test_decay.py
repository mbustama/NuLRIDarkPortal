"""dp_decay: neutrino decay into the mediator (notes Sec. 12.8 item 2, 2026-10-24)."""
import os
import sys

import numpy as np

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
    dd.decay_step(E, zp1, dt, Gam, fL, M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
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
        dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
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
    dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, Fn, Fb, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, False)
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
    dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, F, F, np.zeros((1, 3, 3)), np.zeros((1, 3, 3)), False, True)
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
    dd.decay_step(E, 1.0, 1e40, Gam, fL, M_NU, Fn, Fb, T, T, True, False)
    assert Fn[:, 2].sum() < 1e-12
    assert np.allclose(Fn[:, 0], Fn[:, 1])
