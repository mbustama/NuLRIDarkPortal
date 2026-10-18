"""
The MeV transport for several sources at once (dp_si._si_sweep_cols, propagate_si's unit mass-state columns and
batch_hint; 2026-10-18): against the one-source sweep it replaced (_si_sweep, driven here as propagate_si drove it), to
rounding, for the vector (exact in-bin solution, thermal relics, a visible width, an emission window) and the scalar;
a column's result is the same bit for bit whatever the batch it is computed in; the bookkeeping still balances.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

import dp_si as si  # noqa: E402
from test_si import _grid, _transport_kw  # noqa: E402


def _reference(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_src, p_index, rho_evol,
               Phi_0, M, g_mass, majorana=True, channels="full", phiphi="coherent", z_substeps=1, E_extension=10.0,
               n_u=4096, regeneration=True, mediator="scalar", dirac_sterile="track", Gamma_vis=0.0, thermal=False,
               inj_window=(0.0, np.inf)):
    """propagate_si as it was until 2026-10-18: one source, the one-source sweep _si_sweep."""
    from dp_core import compute_dE_geometric
    E_bins = np.asarray(E_bins, dtype=np.float64)
    N_E = len(E_bins)
    dlnE = float(np.log(E_bins[1] / E_bins[0]))
    n_ext = int(np.ceil(np.log(E_extension) / dlnE - 1e-9)) if E_extension > 1.0 else 0
    N = N_E + n_ext
    E = E_bins[0] * np.exp(dlnE * np.arange(N))
    E[:N_E] = E_bins
    dE = compute_dE_geometric(E)
    q = int(z_substeps)
    Delta = dlnE / q
    Lmax = float(np.log1p(z_max))
    K = max(1, int(np.ceil(Lmax / Delta - 1e-9)))
    L_edges = np.arange(K + 1) * Delta
    L_edges[-1] = Lmax
    top_full = abs((L_edges[K] - L_edges[K - 1]) - Delta) <= 1e-12 * Delta
    L_top_mid = 0.5 * (L_edges[K - 1] + L_edges[K])
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    g_mass = np.ascontiguousarray(g_mass, dtype=np.complex128)
    M2 = M * M
    Gam = si.mediator_width(M, g_mass, mediator, majorana) + float(Gamma_vis)
    MG = M * Gam
    spec = si.model_spec(g_mass, mediator, majorana, phiphi, channels, dirac_sterile)
    if Gamma_vis > 0.0:
        X = 24.0 * np.pi * float(Gamma_vis) / M
        kap = 1.0 if majorana else 0.5
        Ctau = spec["Ctau"].copy()
        for a in range(3):
            for r in range(3):
                add = kap * abs(g_mass[r, a]) ** 2 * X / (4.0 * np.pi) / 3.0
                for qq in (0, 1):
                    Ctau[qq, a, r, :] += add
        spec = dict(spec, Ctau=np.ascontiguousarray(Ctau))
    eta_arr = np.asarray(eta_mass, dtype=np.float64)
    prof = spec["prof"]
    P = len(spec["jt"])
    delta_mode = str(channels).strip().lower() == "delta"
    n0 = si.relic_density_per_helicity(T_nu0) * eta_arr
    P_src = (np.abs(np.asarray(U_pmns)) ** 2).T @ np.asarray(rho_src, dtype=np.float64)
    (lat_B, lat_O), top = si._profile_tables(prof, M, masses, E[0], N, dlnE, q, K, top_full, L_top_mid, n_u)
    runs = spec["runs"]
    Fs = [np.zeros((N, 3)) for _ in runs]
    diags = [np.zeros(6) for _ in runs]
    reg = np.zeros((N, 3))
    J = np.zeros((N, 3, 4))
    Nq = np.zeros((N, 3, P))
    Lq = np.zeros((N, 3, P))
    Dq = np.zeros((N, 3, P))
    node = np.zeros((N, 3), dtype=np.int64)
    jj = np.arange(N, dtype=np.int64)
    Ep = E ** (-p_index)
    if inj_window[0] > 0.0 or np.isfinite(inj_window[1]):
        Ep = np.where((E >= inj_window[0]) & (E < inj_window[1]), Ep, 0.0)
    hbin = float(np.exp(0.5 * dlnE))
    Cregs = [np.ascontiguousarray(Cr if regeneration else np.zeros_like(Cr)) for Cr, _ in runs]
    for m in range(K - 1, -1, -1):
        L_lo, L_hi = L_edges[m], L_edges[m + 1]
        dt, T_inj = si._step_time_and_injection(L_lo, L_hi, H0, Omega_m, Omega_L, p_index, rho_evol, Phi_0)
        S0 = np.ascontiguousarray(np.outer(Ep * (T_inj / dt), P_src))
        si._step_integrals(E, np.exp(L_lo), np.exp(L_hi), np.exp(0.5 * (L_lo + L_hi)), masses, n0, M2, MG,
                           H0, Omega_m, Omega_L, delta_mode, prof, hbin, J, Nq, Lq, Dq)
        if thermal and not delta_mode:
            si._thermal_resonance_terms(E, L_lo, L_hi, masses, T_nu0, eta_arr, M, MG, H0, Omega_m, Omega_L, J,
                                        si._TGL16_X, si._TGL16_W, si._TGL10_X, si._TGL10_W, si._TFD16)
            si._thermal_nonresonant_terms(E, L_lo, L_hi, masses, T_nu0, eta_arr, n0, M2, H0, Omega_m, Omega_L, prof,
                                          J, si._FDX32, si._FDW32, si._GL12_X, si._GL12_W)
        if m == K - 1 and not top_full:
            tB, tO = top
            node[:] = jj[:, None]
        else:
            tB, tO = lat_B, lat_O
            node[:] = (jj * q + m)[:, None]
        for r in range(3):
            if masses[r] <= 0.0:
                node[:, r] = 0
        for k in range(len(runs)):
            si._si_sweep(Fs[k], E, dE, S0, J, Nq, Lq, Dq, spec["jt"], spec["w"], spec["qe"], spec["Ctau"], Cregs[k],
                         tB, tO, node, dt, reg, diags[k], prof == 1)
    F = sum(wt * Fk for (_, wt), Fk in zip(runs, Fs))
    return np.ascontiguousarray(F[:N_E])


def _vector_kw(**over):
    kw = _transport_kw()
    g = si.flavor_couplings("matrix", 1.0, np.array([[0.25, 0.25, 0.0], [0.25, 0.25, 0.0], [0.0, 0.0, 0.0]]))
    gm = si.mass_basis_couplings_model(3e-3 * g, kw["U_pmns"], "vector", True)
    kw.update(M=2e5, g_mass=gm, mediator="vector", thermal=True, Gamma_vis=1e-9)
    kw.update(over)
    return kw


CASES = {
    "vector, thermal, visible width": dict(),
    "vector, upper piece": dict(inj_window=(10 ** 4.39 * 1e9, np.inf), p_index=2.7),
    "vector, lower piece": dict(inj_window=(0.0, 10 ** 4.39 * 1e9), p_index=1.3),
    "vector, light relic": dict(masses=np.array([1e-5, np.sqrt(1e-10 + 7.49e-5), np.sqrt(1e-10 + 2.513e-3)])),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_batched_transport_matches_the_one_source_sweep(case):
    E = _grid(80)
    kw = _vector_kw(**CASES[case])
    si._COLUMNS.clear()
    F = si.propagate_si(E, **kw)
    R = _reference(E, **kw)
    scale = np.max(np.abs(R), axis=1, keepdims=True)
    err = float(np.max(np.abs(F - R) / np.maximum(scale, 1e-300)))
    print(f"{case}: {err:.2e}")
    assert np.all(np.isfinite(F)) and err < 1e-11


def test_scalar_matches_the_one_source_sweep():
    E = _grid(60)
    kw = _transport_kw()
    kw.update(M=5e6, g_mass=si.mass_basis_couplings(si.flavor_couplings("tau", 0.1), kw["U_pmns"]))
    si._COLUMNS.clear()
    F = si.propagate_si(E, **kw)
    R = _reference(E, **kw)
    scale = np.max(np.abs(R), axis=1, keepdims=True)
    assert float(np.max(np.abs(F - R) / np.maximum(scale, 1e-300))) < 1e-11


def test_a_column_does_not_depend_on_its_batch():
    """Alone, or computed with other spectral indices and windows of the same physics (batch_hint): bit for bit;
    the second source composition and the hinted columns come from the kept columns (no new transport)."""
    E = _grid(60)
    kw = _vector_kw()
    si._COLUMNS.clear()
    alone = si.propagate_si(E, **kw)
    si._COLUMNS.clear()
    hint = [(2.0, (0.0, np.inf)), (2.9, (0.0, np.inf)), (2.5, (1e14, np.inf))]
    with si.batch_hint(hint):
        batched = si.propagate_si(E, **kw)
    assert np.array_equal(alone, batched)
    n = len(si._COLUMNS)
    assert n == 1 + len(hint)
    other = si.propagate_si(E, **dict(kw, rho_src=np.array([1.0, 0.0, 0.0]), p_index=2.9))
    assert len(si._COLUMNS) == n                              # reused
    si._COLUMNS.clear()
    ref = si.propagate_si(E, **dict(kw, rho_src=np.array([1.0, 0.0, 0.0]), p_index=2.9))
    assert np.array_equal(other, ref)


def test_diagnostics_balance_with_columns():
    E = _grid(60)
    kw = _vector_kw()
    F, info = si.propagate_si(E, return_diagnostics=True, **kw)
    bal = (info["final_number"] - info["injected"]) - (info["secondaries_in_grid"] - info["absorbed"])
    assert abs(bal) < 1e-12 * info["injected"]
    F2 = si.propagate_si(E, **kw)
    assert np.max(np.abs(F - F2) / np.maximum(np.max(np.abs(F2), axis=1, keepdims=True), 1e-300)) < 1e-12
