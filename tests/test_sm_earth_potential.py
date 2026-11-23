"""The Standard Model's charged-current potential inside the Earth with PREM's electron density (2026-11-08).

Until 2026-11-08 every run left it out (use_SM_potential false), and the code, when asked, put one constant electron
density on every chord. The 'prem' profile takes the exact mean of PREM's electron density over each 'avg' step:
here against an independent evolution (scipy quadrature of rho(r) Y_e along the chord, scipy's expm per step).
"""
import os
import sys

import numpy as np
import pytest
from scipy import integrate
from scipy.linalg import expm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core  # noqa: E402
import dp_spatial as sp  # noqa: E402

AVOGADRO = 6.02214076e23


def _mixing():
    th12, th13, th23, dcp = np.radians([33.76, 8.62, 43.28, 212.0])
    c12, s12, c13, s13, c23, s23 = np.cos(th12), np.sin(th12), np.cos(th13), np.sin(th13), np.cos(th23), np.sin(th23)
    e = np.exp(-1j * dcp)
    R23 = np.array([[1, 0, 0], [0, c23, s23], [0, -s23, c23]], complex)
    R13 = np.array([[c13, 0, s13 * e], [0, 1, 0], [-s13 * np.conj(e), 0, c13]], complex)
    R12 = np.array([[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]], complex)
    return R23 @ R13 @ R12


def _electrons_per_cm3(r_km):
    rho = sp.calc_earth_density(r_km)
    if r_km <= 3480.0:
        per_g = 0.4656 * AVOGADRO
    elif r_km <= 6368.0:
        per_g = 0.4957 * AVOGADRO
    else:
        per_g = 10.0 / 18.015 * AVOGADRO
    return rho * per_g


def _reference(costh, depth_km, N, E, M_rot, rho0, sign):
    """Independent: the chord from its entry (l = 0) to the detector, r(l) from the detector's position, each step's
    mean electron density by quadrature, exp(-i H dr) by scipy."""
    L = sp.get_baseline_earth(depth_km, costh)
    f2 = 6371.0 - depth_km
    r_of = lambda l: np.sqrt(max(f2 * f2 + (L - l) ** 2 + 2.0 * f2 * (L - l) * costh, 0.0))
    dl = L / N
    dr = dl * 1e3 * dp_core.m_to_eV_inv
    cuts = []                                  # where the chord crosses a PREM boundary (density steps there)
    for R in sp._PREM_EDGES_KM[1:]:
        disc = R * R - f2 * f2 * (1.0 - costh * costh)
        if disc > 0.0:
            for u in (-f2 * costh - np.sqrt(disc), -f2 * costh + np.sqrt(disc)):
                if 0.0 < u < L:
                    cuts.append(L - u)
    out = []
    for Ei in E:
        rho = rho0.copy()
        for s in range(N):
            a, b = s * dl, (s + 1) * dl
            pts = [c for c in cuts if a < c < b]
            ne = integrate.quad(lambda l: _electrons_per_cm3(r_of(l)), a, b, points=pts or None, limit=400, epsabs=0,
                                epsrel=1e-12)[0] / dl
            V = sign * np.sqrt(2.0) * dp_core.G_F * ne * dp_core.cm_inv3_to_eV3
            H = M_rot / (2.0 * Ei) + np.diag([V, 0.0, 0.0])
            U = expm(-1j * H * dr)
            rho = U @ rho @ U.conj().T
        out.append(rho)
    return np.array(out)


@pytest.mark.parametrize("costh, anti", [(-0.6, False), (-1.0, True), (-0.2, False)])
def test_prem_profile_against_independent_evolution(costh, anti):
    Um = _mixing()
    M_rot = np.ascontiguousarray(Um @ np.diag([0.0, 7.537e-5, 2.511e-3]) @ Um.conj().T)
    rho0 = Um @ np.diag([0.55, 0.30, 0.15]) @ Um.conj().T          # a mixture of mass states, as the flux arrives
    E = np.array([1e11, 1e12])
    depth, N = 1.5, 25
    L = sp.get_baseline_earth(depth, costh)
    sign = -1.0 if anti else 1.0
    yk_e = sp.earth_yukawa_table(None, cpg=sp.earth_charge_per_gram(q_n=0.0, q_pe=1.0))
    sm = (np.ascontiguousarray(sign * np.sqrt(2.0) * dp_core.G_F * np.diag([1.0, 0.0, 0.0]).astype(complex)), yk_e)
    yk = sp.earth_yukawa_table(None)
    zero = np.zeros((3, 3), complex)
    rho_in = np.broadcast_to(rho0, (len(E), 3, 3)).copy()
    args = (costh, L, N, E, rho_in, M_rot, 0.0, 0.0, zero, zero, depth, "avg", yk, None)
    got = sp._earth_chord_worker(args + (sm,))
    off = sp._earth_chord_worker(args)
    want = _reference(costh, depth, N, E, M_rot, rho0, sign)
    assert np.max(np.abs(got - want)) < 1e-9
    # the test can tell: at 100 GeV the potential moves the flavor content by more than 1e-5
    assert np.max(np.abs(np.real(np.diagonal(got[0] - off[0])))) > 1e-5
    # the operator form (_earth_chord_unitary_worker) gives the same crossing
    U = sp._earth_chord_unitary_worker((costh, L, N, E, M_rot, 0.0, 0.0, zero, zero, depth, "avg", yk, None, sm))
    via_U = np.einsum("nab,bc,ndc->nad", U, rho0, U.conj())
    assert np.max(np.abs(via_U - got)) < 1e-12


def test_setup_profiles():
    """_earth_setup: a number keeps the constant potential (the code before); None hands the PREM profile to the
    chords, with the antineutrinos' sign; 'left' steps refuse it."""
    Um = _mixing()
    M_rot = np.ascontiguousarray(Um @ np.diag([0.0, 7.537e-5, 2.511e-3]) @ Um.conj().T)
    kw = dict(M_rot=M_rot, u_arr=np.zeros(3, complex), m_Zd=1e-12, g_D=0.0, eps_prime=0.0, T_nu0=1.68e-4,
              use_debye_screening=False, eta_bg=np.ones(3), detector_depth_km=1.5, use_SM_potential=True,
              n_sun_paths=8, n_earth_chords=12, U_LRI_struct=np.zeros((3, 3), complex), V_SM_struct=None,
              use_gpu=False, earth_step_scheme="avg", earth_zenith_average="phase", earth_sky="full",
              earth_band_costh=None, earth_nodes_per_band=None, matter_charge=None, debye_sq_z0=None,
              potential_product=None)
    ne = 1.5e24 * dp_core.cm_inv3_to_eV3
    st = sp._earth_setup(n_e_earth_eV3=ne, is_antineutrino=False, **kw)
    assert st["sm"] is None and st["V_SM"][0, 0] == pytest.approx(np.sqrt(2.0) * dp_core.G_F * ne)
    nu = sp._earth_setup(n_e_earth_eV3=None, is_antineutrino=False, **kw)
    nub = sp._earth_setup(n_e_earth_eV3=None, is_antineutrino=True, **kw)
    assert not np.any(nu["V_SM"]) and nu["sm"][0][0, 0] == pytest.approx(np.sqrt(2.0) * dp_core.G_F)
    assert nub["sm"][0][0, 0] == pytest.approx(-np.sqrt(2.0) * dp_core.G_F)
    with pytest.raises(ValueError, match="avg"):
        sp._earth_setup(n_e_earth_eV3=None, is_antineutrino=False, **dict(kw, earth_step_scheme="left",
                                                                          earth_zenith_average="plain"))


def test_config_profile_key():
    """standard_mixing.SM_potential_profile: 'prem' (default) passes no density, 'constant' n_e_earth_cm3, anything
    else is refused."""
    import copy
    import dark_portal_solver as dps
    from dp_config import DEFAULT_CONFIG
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    assert cfg["standard_mixing"]["SM_potential_profile"] == "prem"
    _, p = dps.build_solver_params(cfg)
    assert p["n_e_earth_eV3"] is None
    cfg["standard_mixing"]["SM_potential_profile"] = "constant"
    _, p = dps.build_solver_params(cfg)
    assert p["n_e_earth_eV3"] == pytest.approx(cfg["standard_mixing"]["n_e_earth_cm3"] * dp_core.cm_inv3_to_eV3)
    cfg["standard_mixing"]["SM_potential_profile"] = "uniform"
    with pytest.raises(ValueError, match="SM_potential_profile"):
        dps.build_solver_params(cfg)
