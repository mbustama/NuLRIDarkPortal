"""
Tests of the general matter charge (src/dp_matter.py; docs/dev/SCOPE_WIDENING_PLAN.md, step 1, 2026-10-14):
the potential sourced by q_n n_n + q_p n_p + q_e n_e. The neutron charge must run the historical path; every
density must be linear in the charge; the electron inputs must match the bodies' compositions; a negative
charge must equal a flipped texture.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tools"))

import dp_core as C  # noqa: E402
import dp_matter as DM  # noqa: E402
import dp_spatial as SP  # noqa: E402
import dp_transit as T  # noqa: E402
import dark_portal_solver as dps  # noqa: E402

N0 = 3.091e-8 * C.cm_inv3_to_eV3          # the configured cosmic neutron and electron densities
E0 = 2.2059e-7 * C.cm_inv3_to_eV3
M_U = 1.66054e-24
NA = 6.02214076e23


def test_normalize_signs_and_refusals():
    """Neutrons (and B - L, neutrons in neutral matter) run the historical path (None); a charge negative in
    every body has its sign folded out (B - 3L_e: n - 2e); one that changes sign between bodies is refused."""
    assert DM.normalize(1.0, 0.0, 0.0, N0, E0) is None
    assert DM.normalize(1.0, 1.0, -1.0, N0, E0) is None
    e = DM.normalize(0.0, 0.0, 1.0, N0, E0)
    assert (e.q_n, e.q_pe, e.sign) == (0.0, 1.0, 1.0) and e.n_e0_eV3 == E0
    b3e = DM.normalize(1.0, 1.0, -3.0, N0, E0)
    assert (b3e.q_n, b3e.q_pe, b3e.sign) == (-1.0, 2.0, -1.0)
    nucleons = DM.normalize(1.0, 1.0, 0.0, N0, E0)
    assert (nucleons.q_n, nucleons.q_pe, nucleons.sign) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="one sign"):
        DM.normalize(1.0, 0.0, -0.5, N0, E0)          # n - e/2: positive in the Earth, negative in the cosmos
    zero = DM.normalize(0.0, 0.0, 0.0, N0, E0)
    assert (zero.q_n, zero.q_pe, zero.sign) == (0.0, 0.0, 1.0)
    # the configuration route
    _, params = dps.build_solver_params({"matter_charge": {"q_n": 1.0, "q_p": 1.0, "q_e": -3.0}})
    assert params["matter_charge"] == b3e
    _, params = dps.build_solver_params({})
    assert params["matter_charge"] is None


def test_electron_counts_match_compositions():
    """The point sources' electrons: the bodies' nucleons less their neutrons (Sun 1.98847e33 g, Moon
    7.342e25 g), the Earth's from PREM with Y_e by layer; per gram, neutrons and electrons add to the
    nucleons (N_A per gram of rock, 18/18.015 N_A of water)."""
    from scipy.integrate import quad
    edges = [0.0, 1221.5, 3480.0, 5701.0, 5771.0, 5971.0, 6151.0, 6346.6, 6356.0, 6368.0, 6371.0]
    N_e = sum(quad(lambda r: 4 * np.pi * r * r * SP.calc_earth_density(r) * SP.earth_electrons_per_gram(r),
                   a, b, epsrel=1e-10)[0] for a, b in zip(edges[:-1], edges[1:])) * 1e15
    assert abs(C.N_e_earth / N_e - 1.0) < 1e-3
    assert abs((C.N_n_sun + C.N_e_sun) * M_U / 1.98847e33 - 1.0) < 1e-3
    assert abs((C.N_n_moon + C.N_e_moon) * M_U / 7.342e25 - 1.0) < 1e-3
    for r in (1000.0, 5000.0, 6370.0):
        total = SP.earth_neutrons_per_gram(r) + SP.earth_electrons_per_gram(r)
        assert abs(total / (NA * (18.0 / 18.015 if r > 6368.0 else 1.0)) - 1.0) < 1e-12
    # the charge per gram: the neutron charge gives the neutron function's values bit for bit
    cpg = SP.earth_charge_per_gram(1.0, 0.0)
    assert all(cpg[k] == SP.earth_neutrons_per_gram(r) for k, r in enumerate(SP._EARTH_REGION_SAMPLE_KM))
    assert DM.point_sources(None) == (C.N_n_sun, C.N_n_moon, C.N_n_earth)


def test_galactic_components_match_the_neutron_model():
    """_MW_components reproduces calc_MW_neutron_density (left untouched) at its neutron fractions, and
    calc_MW_electron_density is the same components at their electron fractions."""
    rng = np.random.default_rng(1)
    K = SP._NUCLEONS_PER_CM3_PER_MSUN_PC3
    pts = [(0.0, 0.0, 0.0), (8.33, 0.5 * np.pi, 0.0)] + [
        (10 ** rng.uniform(-4, 1.5), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)) for _ in range(200)]
    for r, th, ph in pts:
        stars, cold, hot = SP._MW_components(r, th, ph)
        n = stars * K * 0.20 + cold * K * 0.142 + hot * 0.15
        e = stars * K * 0.80 + cold * K * 0.858 + hot
        assert abs(SP.calc_MW_neutron_density(r, th, ph) / n - 1.0) < 1e-13
        assert abs(SP.calc_MW_electron_density(r, th, ph) / e - 1.0) < 1e-13
        assert SP.calc_MW_species_density(r, th, ph, 0) == SP.calc_MW_neutron_density(r, th, ph)
        assert SP.calc_MW_species_density(r, th, ph, 1) == SP.calc_MW_electron_density(r, th, ph)


def test_electron_sky_density_table_is_current():
    """src/data/mw_sky_density_electrons.npz matches fresh quadrature of the electron model (rerun
    tools/mw_sky_density.py --species electron after changing the Galactic model)."""
    import mw_sky_density as M
    d = np.load(os.path.join(REPO, "src", "data", "mw_sky_density_electrons.npz"))
    assert str(d["model_digest"]) == M.model_digest(1)
    r, nb = d["r_kpc"], d["nbar_cm3"]
    for i in np.linspace(0, len(r) - 1, 9).astype(int):
        assert abs(M.sky_average(r[i], 1) / nb[i] - 1.0) < 1e-12


@pytest.mark.parametrize("m_eV", [None, 1e-14, 3e-13, 1e-9, 2e6])
def test_earth_table_is_linear_in_the_charge(m_eV):
    """The Earth's Yukawa-smoothed density for q_n n_n + q_pe n_e equals q_n times the neutrons' plus q_pe
    times the electrons', in every mode (contact, direct, heavy) and outside the body; the charged table
    with the neutron charge reproduces the neutron table's numbers."""
    yk_n = SP.earth_yukawa_table(m_eV, cpg=SP.earth_charge_per_gram(1.0, 0.0))
    yk_e = SP.earth_yukawa_table(m_eV, cpg=SP.earth_charge_per_gram(0.0, 1.0))
    yk_q = SP.earth_yukawa_table(m_eV, cpg=SP.earth_charge_per_gram(0.3, 1.7))
    ref = SP.earth_yukawa_table(m_eV)
    assert np.array_equal(yk_n[2], ref[2]) and np.array_equal(yk_n[0][:4], ref[0][:4])
    assert SP.yk_charged(yk_q) and not SP.yk_charged(ref)
    for r in (0.0, 500.0, 1221.0, 1222.0, 3479.9, 3480.1, 5000.0, 6367.0, 6370.0, 6371.0, 7000.0, 12742.0):
        n, e, q = (SP.yukawa_profile_density(r, yk) for yk in (yk_n, yk_e, yk_q))
        if n == 0.0 and e == 0.0:
            assert q == 0.0
            continue
        assert abs(q - (0.3 * n + 1.7 * e)) <= 1e-12 * abs(0.3 * n + 1.7 * e), (r, q, n, e)
        assert n == SP.yukawa_profile_density(r, ref)


@pytest.mark.parametrize("m_eV", [None, 1e-13, 1e-9])
def test_charged_chord_integrator_matches_the_neutron_one(m_eV):
    """_avg_earth_density_seq_q (the charged twin) with the neutron charge reproduces _avg_earth_density_seq;
    with electrons it is the density times the regions' electron/neutron ratios (contact mode)."""
    yk_ref = SP.earth_yukawa_table(m_eV)
    yk_n = SP.earth_yukawa_table(m_eV, cpg=SP.earth_charge_per_gram(1.0, 0.0))
    for c in (-1.0, -0.7, -0.2, -0.01):
        B = SP.get_baseline_earth(1.5, c)
        cross = SP._earth_layer_crossings(B, c, 1.5)
        a = SP._avg_earth_density_seq(B, c, 25, 1.5, 1e-20, cross, SP._GL10_X, SP._GL10_W, yk_ref)
        b = SP.chord_density_seq(yk_n)(B, c, 25, 1.5, 1e-20, cross, SP._GL10_X, SP._GL10_W, yk_n)
        assert np.max(np.abs(b / a - 1.0)) < 1e-14
    cols_ref = SP._earth_columns(np.array([-1.0, -0.5]), 1.5, 0.0, yk_ref)
    cols_n = SP.earth_columns(np.array([-1.0, -0.5]), 1.5, 0.0, yk_n)
    assert np.max(np.abs(cols_n / cols_ref - 1.0)) < 1e-14


def test_galactic_density_is_linear_in_the_charge():
    """dp_matter's Galactic density at the Sun combines the species, each with its own cosmic floor."""
    R_H = 1.0 / (67.4 * C.km_per_s / C.Mpc_to_eV_inv)
    q = DM.MatterCharge(0.4, 1.3, 1.0, E0)
    for m in (1e-33, 1e-28, 1e-25, 1e-20, 1e-12, 1e-6):
        n = T.galactic_effective_density(m, N0, R_H)
        e = T.galactic_effective_density(m, E0, R_H, species=1)
        assert DM.galactic_effective_density(q, m, N0, R_H) == 0.4 * n + 1.3 * e
        assert e > n             # hydrogen-rich: 4-7 electrons per neutron
    # contact limit: the local densities
    assert T.galactic_effective_density(1e-6, E0, R_H, species=1) == SP.calc_MW_electron_density(0, 0, 0) * C.cm_inv3_to_eV3


def _propagate(m_Zd, charge, U_LRI_struct=None, bands=False):
    E = np.geomspace(1e13, 1e16, 10)
    z = np.linspace(4.0, 0.0, 10)
    U = np.ascontiguousarray(dps.construct_PMNS(np.radians(33.41), np.radians(8.58), np.radians(49.1),
                                                np.radians(132.0)))
    M_sq = np.ascontiguousarray(np.diag([0.0, 7.41e-5, 2.511e-3]).astype(np.complex128))
    H0 = 67.4 * dps.km_per_s / dps.Mpc_to_eV_inv
    m_nu = np.array([0.01, np.sqrt(0.01**2 + 7.41e-5), np.sqrt(0.01**2 + 2.511e-3)])
    u = np.ascontiguousarray(np.array([0.5, 0.5, 0.0], dtype=np.complex128))
    g_D = 1e-52      # the Sun's (1e-19 eV) or the Earth's (1e-13 eV) potential ~1e-17 eV, the vacuum term's size
    return dps.propagate_spectrum(E, z, np.array([1 / 3, 2 / 3, 0.0]), H0, 0.315, 0.685, U, M_sq, u, m_Zd, g_D, 0.1,
                                  m_nu, N0, 1.676389e-4, True, 1.5, False, 0.0, (0.5, 0.5), 2.5, 0.0, 1.0, np.ones(3),
                                  2, 4, show_progress=False, verbose=0, n_cores=1, N_steps_earth=25,
                                  U_LRI_struct=U_LRI_struct, earth_step_scheme="avg", earth_zenith_average="phase",
                                  earth_sky="full", host_galaxy_scale=1.0, matter_charge=charge)


@pytest.mark.parametrize("m_Zd", [1e-19, 1e-13])
def test_solver_general_path_with_the_neutron_charge(m_Zd):
    """The general path (dp_matter densities, the charged Earth table and integrator, the host galaxy) with the
    neutron charge MatterCharge(1, 0, 1) reproduces the historical path (matter_charge None) -- at a range of
    ~13 AU (Sun, Galaxy, the Earth's exterior) and of ~2000 km (the Earth's Yukawa profile)."""
    a = _propagate(m_Zd, None)
    b = _propagate(m_Zd, DM.MatterCharge(1.0, 0.0, 1.0, E0))
    for x, y in zip(a, b):
        assert np.max(np.abs(x - y)) <= 1e-12 * np.max(np.abs(x))


def test_negative_charge_equals_flipped_texture():
    """A charge negative everywhere (sign folded into the coefficient) equals the positive charge with the
    texture's sign flipped: the potential is the same, and the absorption does not see the texture."""
    S = np.outer([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]).astype(np.complex128)
    neg = DM.normalize(0.0, 0.0, -1.0, N0, E0)
    pos = DM.normalize(0.0, 0.0, 1.0, N0, E0)
    assert neg.sign == -1.0 and pos.sign == 1.0
    a = _propagate(1e-19, neg, U_LRI_struct=S)
    b = _propagate(1e-19, pos, U_LRI_struct=-S)
    for x, y in zip(a, b):
        assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(x))
    # and the sign matters: the same charge with the texture unflipped gives another flux
    c = _propagate(1e-19, pos, U_LRI_struct=S)
    assert np.max(np.abs(c[0] - a[0])) > 1e-3 * np.max(np.abs(a[0]))


def test_electron_emission_table_matches_the_sun_and_combines():
    """src/data/mw_neff_tables_electrons.npz (tools/mw_neff_tables.py --species electron, Tycho) is the electron
    model's, equals the Sun's integral (dp_transit.galactic_effective_density, species 1) at the Sun's node, and
    dp_matter combines the two species at emission points."""
    import mw_sky_density as M
    import dp_galactic as G
    d = np.load(os.path.join(REPO, "src", "data", "mw_neff_tables_electrons.npz"))
    assert str(d["model_digest"]) == M.model_digest(1) and float(d["n_e0_cm3"]) == 2.2059e-7
    R_H = 1.0 / (67.4 * C.km_per_s / C.Mpc_to_eV_inv)
    for m in (1e-33, 1e-28, 1e-24, 1e-18, 1e-13):
        mine = G.effective_density(np.array([G.R_SUN_KPC]), np.array([0.0]), m, E0, R_H, species=1)[0]
        assert abs(mine / T.galactic_effective_density(m, E0, R_H, species=1) - 1.0) < 1e-9
    R, z = np.array([2.0, 8.33, 15.0]), np.array([0.0, 0.1, 1.0])
    q = DM.MatterCharge(0.4, 1.3, 1.0, E0)
    for m in (1e-30, 1e-22, 1e-9):
        both = DM.galactic_emission_density(q, R, z, m, N0, R_H)
        sep = (0.4 * G.effective_density(R, z, m, N0, R_H) + 1.3 * G.effective_density(R, z, m, E0, R_H, species=1))
        assert np.max(np.abs(both / sep - 1.0)) < 1e-14
