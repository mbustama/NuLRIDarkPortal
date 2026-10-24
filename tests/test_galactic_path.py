"""dp_galactic_path: the incoherent processes along the Galactic lines of sight (2026-10-24)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dp_decay  # noqa: E402
import dp_galactic_path as gp  # noqa: E402
from dp_core import construct_PMNS  # noqa: E402

M_NU = np.array([0.0, 8.6e-3, 0.05])
F_S = np.array([1.0 / 3.0, 2.0 / 3.0, 0.0])


def _setup(seed=1, decay=True, jumps=True, loss=True):
    """Twelve pixels (N_side 1), twelve equal segments to 30 kpc, no potential (coef2 0): random intensities and
    distance weights; rates set to O(1) effects over the path."""
    rng = np.random.default_rng(seed)
    U = construct_PMNS(0.59, 0.15, 0.84, 3.4)
    E = np.geomspace(1e3, 1e7, 24)
    npix, ns = 12, 12
    s = np.linspace(2.5, 30.0, ns)
    I = rng.uniform(0.2, 1.0, (npix, 1)) * (E / E[0]) ** -2.0
    w = rng.uniform(0.0, 1.0, (npix, len(E)))
    q_gas = rng.uniform(0.0, 1.0, (npix, ns))
    q_src = rng.uniform(0.0, 1.0, (npix, ns))
    G = 3e-13 * np.array([[0.0, 1.0, 0.5], [1.0, 0.2, 0.7j], [0.5, -0.7j, -0.4]])
    Gam, _ = dp_decay.rest_rates(G, U, M_NU, 1e-4)
    spec = dp_decay.daughter_spectra(G, U, M_NU, 1e-4)
    if not decay:
        Gam = np.zeros((3, 3))
    L = s[-1] * gp.KPC_EV_INV
    kinds, tables, P_vac = [], [], np.zeros((2, len(E), 3))
    for kk, Uk in enumerate((U, np.conj(U))):
        Gk = np.conj(G) if kk else G
        M_rot = Uk @ np.diag(M_NU ** 2) @ np.conj(Uk).T
        S = np.diag([1.0, -1.0, 0.0])
        t = gp.build_tables(M_rot, S, Uk, Gk, 0.0, 0.0, n_sigma=2)
        tables.append(t)
        P_vac[kk] = F_S @ np.abs(Uk) ** 2
        kinds.append(dict(P=np.stack([P_vac[kk], P_vac[kk]]), j=np.zeros((npix, ns), np.int64),
                          t=np.zeros((npix, ns)), n=np.ones((npix, ns)), coef2=0.0, tables=t))
    jump_unit = 1.0 / (L * max(float(tables[0]["J"].max()), 1e-300)) if jumps else 0.0
    loss_C = 0.3 * E[0] / (L * float(tables[0]["GG"].max())) if loss else 0.0
    rates = dict(jump_unit=jump_unit, Gam=Gam, spec=spec, m=M_NU, loss_C=loss_C)
    return E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates


def test_vacuum_path_matches_the_sweep():
    """Equal segments (the order of the path's pieces then immaterial): the spectral-basis path equals the
    per-pixel sweep to rounding, with all three processes on."""
    E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates = _setup()
    a = gp.sweep_populations(E, I, w, q_gas, q_src, s, kinds, rates)
    b = gp.vacuum_populations(E, I, w, q_gas, q_src, s, P_vac, tables, rates, 1, 1)
    scale = a.sum(axis=(0, 3))[None, :, :, None]
    assert np.max(np.abs(b - a) / scale) < 1e-9
    # the processes act: the populations moved from the emitted ones
    emitted = (w[:, :, None] * q_gas.sum(1)[:, None, None] + (1 - w[:, :, None]) * q_src.sum(1)[:, None, None])
    assert np.max(np.abs(a[0] - emitted * P_vac[0][None])) > 1e-2 * emitted.max()


def test_jumps_keep_the_number():
    """The soft scatterings alone move neutrinos between states, never in or out: per pixel and energy the
    populations at the Sun sum to the emitted number per unit intensity."""
    E, I, w, q_gas, q_src, s, kinds, tables, P_vac, rates = _setup(decay=False, loss=False)
    a = gp.sweep_populations(E, I, w, q_gas, q_src, s, kinds, rates)
    emitted = w * q_gas.sum(1)[:, None] + (1 - w) * q_src.sum(1)[:, None]
    for kk in range(2):
        assert np.allclose(a[kk].sum(-1), emitted, rtol=1e-12)
