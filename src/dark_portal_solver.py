"""
dark_portal_solver.py -- public facade of the dark-portal solver.

The implementation was split (2026-07, pure verbatim move) into:
    dp_core.py        environment guards, constants, shared physics utilities
    dp_propagator.py  the exactly unitary 3x3 propagator (deflated eigenbasis)
    dp_thermal.py     thermal attenuation rates + two-component coupling cache
    dp_cosmo.py       cosmological-phase kernels (z-stepper, sigma table)
    dp_spatial.py     Milky Way / Earth transit, workers, persistent pool
    dp_config.py      DEFAULT_CONFIG, config merging/validation

This module re-exports the complete historical surface (every name external code
imported or accessed as dps.<name> keeps working), and itself retains only the
cross-phase orchestration: propagate_spectrum, propagation_worker, run_solver,
file/plot output, and the CLI.

Backward-compatibility note: HAS_TQDM is mutable state owned by dp_core; reading
dark_portal_solver.HAS_TQDM is delegated there, but WRITING it must now target
dp_core (`import dp_core; dp_core.HAS_TQDM = False`), as the in-repo scanner and
convergence tool do.
"""
import dp_core  # noqa: F401  (environment guards run on this first import)

import os
import copy
import json
import argparse
import multiprocessing
import numpy as np
from typing import Dict, Tuple

from dp_core import (  # noqa: F401  (re-exports + orchestration dependencies)
    eV, keV, MeV, GeV, TeV, PeV,
    m_to_eV_inv, cm_inv3_to_eV3, Mpc_to_eV_inv, G_F, km_per_s,
    d_sun_eV_inv, d_moon_eV_inv, R_earth_eV_inv, R_core_eV_inv,
    N_n_sun, N_n_earth, N_n_moon,
    HAS_NUMBA, HAS_MATPLOTLIB, _TBB_LAYER, numba, njit, prange, tqdm, safe_print,
    construct_PMNS, calc_debye_mass_sq, get_mass_eigenvalues, compute_dE_geometric,
)
if HAS_MATPLOTLIB:
    from dp_core import plt  # noqa: F401
from dp_propagator import propagator_3x3, eigenvectors_3x3  # noqa: F401
from dp_thermal import (  # noqa: F401
    thermal_averaged_rates_flavor, _seg_moment_atan, _seg_moment_log,
    thermal_averaged_rates_flavor_analytic, precompute_all_thermal_rates_analytic,
    precompute_all_thermal_rates, _interp_thermal_grid_z, build_thermal_grid, absorption_normalization,
    thermal_grid_cached, _tc_cache, _tc_try_fit, compute_gamma_grid_inputs, relic_state_grid_cached,
    _GL16_X, _GL16_W, _GL10_X, _GL10_W, _FD16,
)
from dp_transit import texture_scales  # noqa: F401
import dp_matter
import dp_absorption
from dp_cosmo import (  # noqa: F401
    calc_H_Hubble, calc_R_source, calc_V_LRI_cosmo, get_adiabatic_mixing_cosmo, yukawa_horizon_inv_m2,
    _build_sigma_projection_table, step_cosmological_pair, cascade_kernel_weights, cross_kernel_weights,
    _build_sigma_coupling_tables, _cosmo_coherent_rates, _cosmo_projections, step_cosmological_rates,
)
from dp_spatial import (  # noqa: F401
    calc_MW_neutron_density, calc_earth_density, get_baseline_earth,
    spatial_total_density_earth,
    _evolve_rho_all_bins, step_earth_transit,
    _earth_chord_worker, _PREM_BOUNDARIES_KM,
    _earth_layer_crossings, _avg_earth_density_seq, _split_E_blocks,
    _PREM_EDGES_KM, _PREM_RHO_COEFFS, _PREM_F_COEFFS, yukawa_profile_table,
    earth_yukawa_table, yukawa_profile_density, earth_neutron_density_cm3,
    earth_total_density, _earth_chord_features, _yk_extra_cm3, average_earth_chords,
    _run_spatial_tasks, _get_spatial_pool, _close_spatial_pool,
    propagate_earth_spatial, transit_to_earth,
)
from dp_config import (  # noqa: F401
    parse_structure_matrix_3x3, _build_default_config, DEFAULT_CONFIG,
    merge_config_with_defaults, create_default_config,
)


def __getattr__(name):
    # HAS_TQDM lives in dp_core and is mutated at runtime (scanner/convergence tool
    # silence the progress bars); delegate reads so stale readers stay correct.
    if name == "HAS_TQDM":
        return dp_core.HAS_TQDM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _species_matrices(U_pmns, M_sq, u_arr, U_LRI_struct, V_SM_struct, anti: bool):
    """One kind's matrices, with the antineutrino transform (U -> conj U, u -> conj u, structure matrices
    conjugated) applied here, centrally: (u_s, M_rot = U M_sq U^dag, S_eff (the LRI texture), V_SM_eff or None)."""
    U_s = np.ascontiguousarray(np.conjugate(U_pmns)) if anti else np.ascontiguousarray(U_pmns)
    u_s = np.ascontiguousarray(np.conjugate(u_arr)) if anti else np.ascontiguousarray(u_arr)
    M_rot = np.ascontiguousarray(U_s @ M_sq @ np.ascontiguousarray(np.conjugate(U_s).T))
    if U_LRI_struct is None:
        S_eff = np.outer(u_s, np.conjugate(u_s))
    else:
        S_eff = np.conjugate(U_LRI_struct) if anti else U_LRI_struct
    S_eff = np.ascontiguousarray(S_eff, dtype=np.complex128)
    V_SM_eff = None
    if V_SM_struct is not None:
        V_SM_eff = np.ascontiguousarray(np.conjugate(V_SM_struct) if anti else V_SM_struct, dtype=np.complex128)
    return u_s, M_rot, S_eff, V_SM_eff


def propagate_spectrum(E_bins: np.ndarray, z_bins: np.ndarray, rho_src: np.ndarray,
                       H0: float, Omega_m: float, Omega_L: float,
                       U_pmns: np.ndarray, M_sq: np.ndarray,
                       u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
                       m_nu_bg_arr: np.ndarray, n_n0: float, T_nu0: float,
                       use_debye_screening: bool, detector_depth_km: float,
                       use_SM_potential: bool, n_e_earth: float,
                       injection, p_index: float, rho_evol: float,
                       Phi_0: float, eta_bg: np.ndarray, n_sun_paths: int, n_earth_chords: int,
                       transit_dtheta: float = 0.002,
                       show_progress: bool = True, verbose: int = 1, n_cores: int = 1,
                       precomputed_Gamma_grid_all: np.ndarray = None, N_steps_earth: int = 500,
                       external_pool=None, U_LRI_struct=None, V_SM_struct=None,
                       n_z_thermal: int = 0, use_gpu: bool = False,
                       thermal_analytic: bool = True,
                       earth_step_scheme: str = "left",
                       earth_zenith_average: str = "plain",
                       earth_sky: str = "up",
                       host_galaxy_scale: float = 0.0,
                       earth_band_costh=None, earth_nodes_per_band: int = 2, sun_path_cos=None,
                       arrival_flavor=None, arrival_basis: str = "flavor", matter_charge=None,
                       absorption_model: str = "lagrangian", G_coupling=None, Gamma_vis=None,
                       si_cosmology: bool = False, si_settings=None, potential_product=None,
                       inj_window=(0.0, np.inf), earth_transfer: str = "off"):
    """
    Executes the complete spectrum integration, driving the underlying JIT kernels.
    Neutrinos and antineutrinos are evolved together through the cosmological phase (an
    absorption of either emits one secondary of each kind) and then separately through the
    Galaxy, the Sun, the Earth's exterior and the Earth (propagate_earth_spatial).
    Includes verbose level 2 for periodic redshift status updates.

    Arguments:
      E_bins: energy grid (eV), geometric spacing assumed by the cascading integral.
      z_bins: redshift grid, DESCENDING from z_max to 0 (the integration direction).
      rho_src: (3,) source flavor composition (f_e, f_mu, f_tau).
      H0, Omega_m, Omega_L: flat-LCDM cosmology (H0 already converted to eV).
      U_pmns: (3, 3) PMNS matrix, neutrino convention (conjugated here for antineutrinos).
      M_sq: (3, 3) complex diagonal mass-squared matrix diag(0, dm21, dm31_signed).
      u_arr: (3,) complex sterile mixing amplitudes (u_e, u_mu, u_tau), neutrino convention.
      m_Zd, g_D, eps_prime: dark mediator mass (eV) and couplings.
      m_nu_bg_arr: (3,) neutrino background mass eigenvalues (eV).
      n_n0: cosmological neutron density today (eV^3); T_nu0: CnuB temperature (eV).
      use_debye_screening: include the thermal Debye mass in the mediator propagator.
      detector_depth_km: detector depth below the Earth surface.
      use_SM_potential / n_e_earth: standard MSW matter potential inside the Earth
        (electron density in eV^3).
      injection: (w_nu, w_nb), the weights of the neutrino and antineutrino injection
        (e.g. (1, 0) for neutrino sources only, (1/2, 1/2) for nubar/nu = 1). Until
        2026-10-13 this was a single-species flag and each call propagated one kind alone.
      p_index, rho_evol, Phi_0: source spectrum E^-p, redshift evolution, normalization.
      inj_window: (E_min, E_max) in eV (2026-10-15): the sources emit only neutrinos whose energy today,
      earth_transfer: whether and when the Earth crossing's responses are reused
        (dp_spatial.propagate_earth_spatial): 'off', 'repeat' or 'always'.
        E_source / (1+z), lies in [E_min, E_max) -- one piece of a broken power law (analysis/model.py); the
        default emits the whole power law.
      eta_bg: (3,) CnuB flavor asymmetries.
      n_sun_paths, n_earth_chords: directions to the Sun in its transit, and zenith chords.
      transit_dtheta: resolution of the transit to the Earth (dp_transit.build_frame_table).
      show_progress, verbose: progress bars / logging (verbose=2 adds z updates).
      n_cores: intra-run parallelism of the spatial handover AND the numba thread cap.
      precomputed_Gamma_grid_all, N_steps_earth, external_pool, U_LRI_struct,
        V_SM_struct, earth_step_scheme: see the dedicated paragraphs below.
      n_z_thermal: coarse z-sampling of the thermal grid (0 = exact; see
        build_thermal_grid).
      use_gpu: route the spatial handover through gpu_spatial when cupy+CUDA exist.
      thermal_analytic: semi-analytic thermal momentum kernel (True, default) vs the
        midpoint reference kernel.
      host_galaxy_scale: where the sources sit (2026-10-13). 0 (the default): in the cosmic
        mean density, which sets the potential the neutrinos are emitted in. s > 0: inside a
        host galaxy like the Milky Way seen from the Sun, its neutrons scaled by s (1: a
        Milky-Way-like host at a Sun-like position), added to the cosmic mean -- the same
        Yukawa-smoothed profile as our own Galaxy (dp_transit.galactic_effective_density), so
        right at every range. Only the injection uses it: the injected flavors split over the
        host's eigenstates, which leave the host adiabatically into the intergalactic ones.
      arrival_flavor: None (the default: sources at cosmological distances) or a (3,) flavor
        composition of sources inside the Milky Way at the local potential (the Galactic component,
        emission 'local'; docs/dev/DIRECTIONS_PLAN.md step 5): the cosmological phase is skipped, and
        each kind arrives with these flavor fractions times its injection weight at every energy
        (unit spectrum), decohered in the local eigenstates (dp_transit.flavor_populations), then
        crosses the Sun and the Earth as the extragalactic flux does. The result is the flux at the
        detector per unit flux emitted, per energy. arrival_basis = 'eigen': arrival_flavor holds instead
        the populations of the local eigenstates (ascending eigenvalues) -- the Sun and the Earth's response
        to each of them, for emission points elsewhere in the Galaxy (step 6, _galactic_transfer).
      absorption_model: 'lagrangian' (the default since 2026-10-14; dp_absorption, docs/dev/PHYSICS_AUDIT.md
        1a-1c, 1e, 1f, 1i, 1j): the resonant absorption from the mediator's coupling to the neutrinos G (default
        the dark portal's g_D u u^dag + eps' g/c_W I), each eigenstate weighted on each relic mass state by
        |(U^dag G W)_ra|^2, the width M Tr(G G^dag)/24 pi plus Gamma_vis (default the Z mixing's e+e- and mu+mu-),
        the decays' neutrinos in the eigenstates by (W^dag G G^dag W)_aa, the relics' Moller flux factor and their
        non-relativistic Debye mass; or 'theory_notes', the form used until then (T7, T11, T12). G_coupling,
        Gamma_vis override the dark portal's coupling and visible width (other models).
      si_cosmology: the cosmological phase from dp_si's vector transport instead (decided 2026-10-14 for the MeV
        regime, where the cosmological potential is negligible: _use_si_cosmology) -- every collision with the
        relics (the s-channel resonance averaged over their thermal momenta, t and u exchange, pair production),
        mass state by mass state, with the coupling's mass-basis form and the visible width; the arriving mass
        states then cross the Galaxy, the Sun and the Earth as usual. si_settings: dp_si's z_substeps and
        E_extension (default 1 and 10).
      potential_product: the coupling product of the potential, K n / m^2 times it (the theory notes' Eq. 9 has
        g_D eps', the default); dp_models gives the eight benchmarks' (g^2 / K for the direct gauge models).
      matter_charge: None (the potential's source is the neutrons, as the theory notes' Eq. T9 has it) or a
        dp_matter.MatterCharge (docs/dev/SCOPE_WIDENING_PLAN.md): every density -- the cosmic mean, the host
        galaxy, the Galaxy, the Sun, the Moon, the Earth -- is then that charge's, and its sign multiplies the
        potential. n_n0 stays the cosmic NEUTRON density (the charge carries the electrons').

    Returns: (F_nu, F_nb), the (N_E, 3) flavor fluxes of neutrinos and of antineutrinos
    at the detector; a detector without charge separation sees their sum.

    U_LRI_struct / V_SM_struct, if given, are dimensionless 3x3 Hermitian flavor structure
    matrices overriding the default LRI texture (u_i * conj(u_j)) and SM matter-potential
    texture (diag(1, 0, 0)). Pass them in NEUTRINO convention: the antineutrino
    conjugation (V -> -conj(V)) is applied centrally here, as for U_pmns and u_arr. The
    overrides affect only the coherent forward-scattering potentials -- the Debye
    screening mass and the thermal attenuation/regeneration rates remain tied to the
    u_e/u_mu/u_tau amplitudes (config keys: dark_portal.U_LRI_struct,
    standard_mixing.V_SM_struct).

    external_pool, if given, is a caller-managed multiprocessing.Pool passed straight through
    to propagate_earth_spatial, reused across many calls instead of creating a fresh one each
    time -- see _run_spatial_tasks's docstring for why this matters under 'spawn'.

    precomputed_Gamma_grid_all lets a caller reuse a thermal-rate grid across calls (it is
    the same for both kinds, and independent of the source composition; see
    compute_gamma_grid_inputs).

    N_steps_earth controls the number of path-length steps per Earth chord, and
    earth_step_scheme selects the potential
    discretization ('avg' = exact step-averaged potential, 'left' = historical
    left-endpoint sampling). Under 'avg' -- the CONFIG default -- the accumulated LRI
    phase is step-count-independent and results carry a convergence certificate
    (N = 8..1000 agree to <= 5e-9 at the standard validation points; DEFAULT_CONFIG
    therefore ships N_steps_earth = 25, a ~3x margin over the flat-at-8 knee,
    revalidated against N = 100 at the four standard points on adoption). Under
    'left', results scatter at the ~1e-2 level with ANY step-count change (fringe
    re-aliasing) and 500 was the pinned historical value. NOTE the function-signature
    defaults here (500, 'left') deliberately preserve the historical behavior for
    direct callers that omit these kwargs; run_solver and the analysis pipeline
    always pass the config values.
    """
    if HAS_NUMBA:
        # The cosmological step below is parallelized internally via numba prange; cap its
        # thread pool to n_cores so a caller that already parallelizes across many processes
        # at a higher level (dark_portal_scanner.py, the analysis package's scan workers --
        # both of which pass n_cores=1 here) doesn't also oversubscribe every core
        # from inside each of its own worker processes.
        dp_core.set_numba_threads(n_cores)

    if isinstance(injection, (bool, np.bool_)) or np.ndim(injection) != 1 or len(injection) != 2:
        raise TypeError("propagate_spectrum: injection must be the pair (w_nu, w_nb); the single-species "
                        "is_antineutrino flag was replaced on 2026-10-13")
    w_nu, w_nb = float(injection[0]), float(injection[1])
    N_E = len(E_bins)
    dz = z_bins[0] - z_bins[1]

    pp = g_D * eps_prime if potential_product is None else float(potential_product)
    if absorption_model not in ("lagrangian", "theory_notes"):
        raise ValueError(f"absorption_model must be 'lagrangian' or 'theory_notes', got {absorption_model!r}")
    lagrangian = absorption_model == "lagrangian"
    sum_G, sum_G_alpha, eta_mass = compute_gamma_grid_inputs(U_pmns, u_arr, g_D, eps_prime, eta_bg)
    BR = np.zeros(3, dtype=np.float64)
    if sum_G > 0:
        BR = sum_G_alpha / sum_G
    ab = None
    if lagrangian:
        G_c = dp_absorption.dark_portal_coupling(g_D, u_arr, eps_prime) if G_coupling is None else G_coupling
        G_vis = dp_absorption.z_mixing_visible_width(m_Zd, eps_prime) if Gamma_vis is None else float(Gamma_vis)
        ab = (dp_absorption.AbsorptionSetup(G_c, U_pmns, m_Zd, G_vis, anti=False),
              dp_absorption.AbsorptionSetup(G_c, U_pmns, m_Zd, G_vis, anti=True))
        sum_G = ab[0].sum_G

    def debye_sq(T_nu):
        """The mediator's Debye mass squared at relic temperature T_nu (0 without screening)."""
        if not use_debye_screening:
            return 0.0
        if lagrangian:
            return dp_absorption.debye_mass_sq(ab[0].G_mass_diag2, m_nu_bg_arr, T_nu, eta_mass)
        return calc_debye_mass_sq(g_D, u_arr, T_nu, eta_bg)

    # Pre-compute the thermal interaction rates for the entire grid before sequential stepping
    # (the same for both kinds); reuse the caller's grid when given one. Sources inside the Galaxy
    # (arrival_flavor) skip the cosmological phase and need none.
    if arrival_flavor is not None:
        Gamma_grid_all = np.zeros((0, N_E, 3), dtype=np.float64)
    elif precomputed_Gamma_grid_all is not None:
        Gamma_grid_all = precomputed_Gamma_grid_all
    else:
        if verbose == 2:
            safe_print(f"Pre-computing thermal interaction rate grids...")
        if lagrangian:
            Gamma_grid_all = relic_state_grid_cached(E_bins, z_bins, sum_G, m_Zd, m_nu_bg_arr, T_nu0, eta_mass,
                                                     moller=True)
        else:
            Gamma_grid_all = build_thermal_grid(E_bins, z_bins, sum_G, sum_G_alpha, m_Zd, m_nu_bg_arr, T_nu0,
                                                eta_mass, n_z_thermal=n_z_thermal, thermal_analytic=thermal_analytic)

    # DOMAIN CHECK: a non-finite rate grid would propagate NaN silently through the whole
    # z-loop and out of the final flux (historically: an all-NaN spectrum with no hint of
    # which parameter was responsible). Fail loudly, naming the offending values instead.
    if not np.isfinite(Gamma_grid_all).all():
        raise ValueError(
            f"Non-finite thermal-rate grid (Gamma) at m_Zd_eV={m_Zd:.6g}, g_D={g_D:.6g}, "
            f"epsilon_prime={eps_prime:.6g}: these dark-portal parameter values are outside "
            f"the numerically supported domain of thermal_averaged_rates_flavor.")

    # The LRI coefficient along the redshift grid (neutrino sign): c_z = 2 (1+z) A_LRI(z), the
    # sigma-table coordinate per unit energy (see _build_sigma_projection_table).
    c_z_arr = np.empty(len(z_bins), dtype=np.float64)
    n_q0 = dp_matter.cosmic_density(matter_charge, n_n0)          # n_n0 itself for the neutron charge
    q_sign = 1.0 if matter_charge is None else float(matter_charge.sign)
    c_W_ = 0.88
    m_Z_ = 91.18 * GeV
    g_weak_ = 0.65
    for k in range(len(z_bins)):
        zk = z_bins[k]
        # The uniform cosmological potential is not screened by the relics (no gradient, so no force on them,
        # and no reservoir of dark charge; decided 2026-10-14, docs/dev/PHYSICS_AUDIT.md): the bare mass under
        # the corrected absorption; the theory notes' Eq. 9 form screens it.
        m_D_sq_k = 0.0 if lagrangian else debye_sq(T_nu0 * (1.0 + zk))
        inv_m2 = yukawa_horizon_inv_m2(m_Zd**2 + m_D_sq_k, 1.0 / calc_H_Hubble(zk, H0, Omega_m, Omega_L))
        A_k = pp * inv_m2 * (4.0 * c_W_ * m_Z_**2 / g_weak_) * (np.sqrt(2.0) / 2.0 * G_F * n_q0 * (1.0 + zk)**3)
        c_z_arr[k] = 2.0 * (1.0 + zk) * A_k

    # The coordinate at emission: the cosmic mean (c_inj = c_z, bit for bit) unless the sources
    # sit in a host galaxy, whose neutrons do not dilute with (1+z)^3 (a bound structure).
    c_inj_arr = c_z_arr.copy()
    if host_galaxy_scale > 0.0 and pp != 0.0:
        import dp_transit
        K_ = (4.0 * c_W_ * m_Z_**2 / g_weak_) * (np.sqrt(2.0) / 2.0 * G_F)
        for k in range(len(z_bins)):
            zk = z_bins[k]
            m_D_sq_k = debye_sq(T_nu0 * (1.0 + zk))
            m_eff_sq = m_Zd**2 + m_D_sq_k
            R_H_k = 1.0 / calc_H_Hubble(zk, H0, Omega_m, Omega_L)
            if matter_charge is None:
                n_host = host_galaxy_scale * dp_transit.galactic_effective_density(np.sqrt(m_eff_sq), 0.0, R_H_k)
            else:
                n_host = host_galaxy_scale * dp_matter.galactic_effective_density(
                    matter_charge, np.sqrt(m_eff_sq), 0.0, R_H_k, n_e0_eV3=0.0)
            c_inj_arr[k] = c_z_arr[k] + 2.0 * (1.0 + zk) * pp * K_ * n_host / m_eff_sq
    elif host_galaxy_scale < 0.0:
        raise ValueError(f"source.host_galaxy_scale must be >= 0, got {host_galaxy_scale}")

    # Per kind: the vacuum matrix U M U^dag (computed once), the LRI and SM textures with the
    # antineutrino transform (U -> conj U, u -> conj u, V -> -conj V) applied here, centrally,
    # and the sigma-table of the cosmological mixing.
    species = []
    for anti in (False, True):
        u_s, M_rot, S_eff, V_SM_eff = _species_matrices(U_pmns, M_sq, u_arr, U_LRI_struct, V_SM_struct, anti)
        c_z = -c_z_arr if anti else c_z_arr
        c_inj = -c_inj_arr if anti else c_inj_arr
        # H(sigma) = M_rot + sigma * S_signed, tabulated over |sigma| with the sign folded into S (the
        # antineutrinos' and a negative matter charge's)
        S_signed = np.ascontiguousarray(-S_eff if anti else S_eff)
        # a negative matter charge, or a negative coupling product g_D eps' (both signs of the dark portal's
        # potential are physical, 2026-10-14): the table's sigma is |E c_z|, so the sign goes into S
        if q_sign * np.sign(pp) < 0.0:
            S_signed = np.ascontiguousarray(-S_signed)
        abs_c = np.abs(np.concatenate([c_z, c_inj]))
        Q_t, s_t, Mp_t, floor_t, sig_lo_t, sig_hi_t = texture_scales(M_rot, S_signed)
        sig_min = max(E_bins[0] * abs_c.min(), 1e-30 * max(np.max(np.abs(M_rot)), 1e-300))
        sig_max = max(E_bins[-1] * abs_c.max(), sig_min * 2.0)
        if sig_hi_t > 0.0 and sig_max > sig_hi_t:
            # above sig_hi the eigenbasis is the texture's own to 1e-9 (texture_scales): the
            # table stops there and lookups beyond it clamp to its last node
            sig_max = max(sig_hi_t, sig_min * 2.0)
        N_sigma = 4096
        sigma_nodes = np.geomspace(sig_min, sig_max, N_sigma)
        Qabs_t = Qreg_t = None
        if lagrangian:
            ab_s = ab[1] if anti else ab[0]
            P_table, Qabs_t, Qreg_t = _build_sigma_coupling_tables(Q_t, Mp_t, s_t, floor_t, sigma_nodes,
                                                                   ab_s.A, ab_s.B)
        else:
            P_table = _build_sigma_projection_table(Q_t, Mp_t, s_t, floor_t, sigma_nodes)
        species.append(dict(anti=anti, u=u_s, M_rot=M_rot, S_eff=S_eff, V_SM_eff=V_SM_eff, c_z=c_z,
                            Qabs_table=Qabs_t, Qreg_table=Qreg_t, G_step=np.zeros((N_E, 3)),
                            brp_step=np.zeros((N_E, 3)),
                            c_inj=c_inj, P_inj=np.zeros((N_E, 3, 3), dtype=np.float64),
                            P_table=P_table, log_s0=np.log(sigma_nodes[0]),
                            inv_dlogs=(N_sigma - 1) / np.log(sig_max / sig_min),
                            F=np.zeros((N_E, 3), dtype=np.float64), P_all=np.zeros((N_E, 3, 3), dtype=np.float64),
                            R=np.zeros(N_E, dtype=np.float64)))
    nu, nb = species

    inv_sum_G = 1.0 / sum_G if sum_G > 0.0 else 0.0

    # The cascade kernels' exact bin integrals depend only on the (fixed) energy grid
    W3, pref3, self_same = cascade_kernel_weights(E_bins)
    W1, W2, c1, c2, self_cross = cross_kernel_weights(E_bins)
    inj_lo, inj_hi = float(inj_window[0]), float(inj_window[1])
    inj_w = ((E_bins >= inj_lo) & (E_bins < inj_hi)).astype(np.float64)

    z_steps = range(1, len(z_bins)) if (arrival_flavor is None and not si_cosmology) else range(0)
    iterator = tqdm(z_steps, desc="Cosmo Phase", leave=False) if (dp_core.HAS_TQDM and show_progress) else z_steps
    z_print_mod = max(1, len(z_bins) // 10)
    if verbose == 2:
        safe_print(f"\n--- Cosmological Integration Started (z_max = {z_bins[0]:.2f}) ---")
    for k in iterator:
        z = z_bins[k]
        dt = dz / (calc_H_Hubble(z, H0, Omega_m, Omega_L) * (1.0 + z))
        R_z = calc_R_source(z, rho_evol)
        if verbose == 2 and k % z_print_mod == 0:
            safe_print(f"Cosmological propagation: currently integrating at redshift z = {z:.4f}")
        if not lagrangian:
            step_cosmological_pair(z, dt, R_z, E_bins, nu["F"], nb["F"], w_nu, w_nb, rho_src,
                                   nu["P_table"], nu["log_s0"], nu["inv_dlogs"], nu["c_z"][k],
                                   nb["P_table"], nb["log_s0"], nb["inv_dlogs"], nb["c_z"][k],
                                   p_index, Phi_0, BR, W3, pref3, self_same, W1, W2, c1, c2, self_cross,
                                   nu["P_all"], nb["P_all"], np.ascontiguousarray(Gamma_grid_all[k, :, :]),
                                   nu["R"], nb["R"], nu["c_inj"][k], nb["c_inj"][k], nu["P_inj"], nb["P_inj"],
                                   inj_w)
            continue
        G_state = np.ascontiguousarray(Gamma_grid_all[k, :, :])
        Pis = []
        for sp in species:
            _cosmo_coherent_rates(E_bins, sp["P_table"], sp["Qabs_table"], sp["Qreg_table"], sp["log_s0"],
                                  sp["inv_dlogs"], sp["c_z"][k], G_state, inv_sum_G, sp["P_all"], sp["G_step"],
                                  sp["brp_step"])
            Pi = sp["P_all"]
            if sp["c_inj"][k] != sp["c_z"][k]:
                _cosmo_projections(E_bins, sp["P_table"], sp["log_s0"], sp["inv_dlogs"], sp["c_inj"][k], sp["P_inj"])
                Pi = sp["P_inj"]
            Pis.append(Pi)
        step_cosmological_rates(z, dt, R_z, E_bins, nu["F"], nb["F"], w_nu, w_nb, rho_src, Pis[0], Pis[1],
                                nu["G_step"], nb["G_step"], nu["brp_step"], nb["brp_step"], p_index, Phi_0,
                                W3, pref3, self_same, W1, W2, c1, c2, self_cross, nu["R"], nb["R"], inj_w)
    if verbose == 2:
        safe_print(f"Cosmological baseline finished at z = {z_bins[-1]:.4f}.")

    arrival = "eigen"
    if si_cosmology and arrival_flavor is None:
        if not lagrangian:
            raise ValueError("si_cosmology needs absorption_model = 'lagrangian'")
        import dp_si
        st = dict(z_substeps=1, E_extension=10.0)
        st.update(si_settings or {})
        g_mass = dp_si.mass_basis_couplings_model(ab[0].G, U_pmns, "vector", True)
        w_tot = w_nu + w_nb
        F_mass = dp_si.propagate_si(E_bins, float(z_bins[0]), H0, Omega_m, Omega_L, U_pmns, m_nu_bg_arr, T_nu0,
                                    eta_mass, rho_src, p_index, rho_evol, Phi_0 * w_tot, m_Zd, g_mass,
                                    majorana=True, channels="full", mediator="vector",
                                    z_substeps=int(st["z_substeps"]), E_extension=float(st["E_extension"]),
                                    Gamma_vis=ab[0].Gamma_vis, inj_window=(inj_lo, inj_hi),
                                    thermal="full" if st.get("thermal_products") else True)
        # dp_si carries both kinds of each mass state together (exact for their sum, the interaction conserving CP
        # to O(Gamma/M)); split in the injection's proportions; the arrival's eigenstates are the mass states
        # (no potential) in ascending order of the vacuum term
        order = np.argsort(np.real(np.diag(M_sq)))
        for sp, w_s in zip(species, (w_nu, w_nb)):
            sp["F"] = np.ascontiguousarray(F_mass[:, order] * (w_s / w_tot if w_tot > 0 else 0.0))
    if arrival_flavor is not None:
        f_arr = np.asarray(arrival_flavor, dtype=np.float64)
        if f_arr.shape != (3,) or np.any(f_arr < 0.0):
            raise ValueError(f"arrival_flavor must be a (3,) non-negative composition, got {arrival_flavor!r}")
        if arrival_basis not in ("flavor", "eigen"):
            raise ValueError(f"arrival_basis must be 'flavor' or 'eigen', got {arrival_basis!r}")
        for sp, w_s in zip(species, (w_nu, w_nb)):
            sp["F"] = np.ascontiguousarray(np.broadcast_to(w_s * f_arr, (N_E, 3)))
        arrival = arrival_basis

    out = []
    bands = []
    n_b = 0 if earth_band_costh is None else len(earth_band_costh) - 1
    for sp in species:
        if not np.any(sp["F"]):
            out.append(np.zeros((N_E, 3), dtype=np.float64))   # nothing of this kind arrives
            bands.append(np.zeros((n_b, N_E, 3), dtype=np.float64))
            continue
        F_earth = propagate_earth_spatial(E_bins, sp["F"], sp["M_rot"], sp["u"], m_Zd, g_D, eps_prime,
                                          T_nu0, use_debye_screening, eta_bg, detector_depth_km,
                                          use_SM_potential, n_e_earth, sp["anti"], n_sun_paths, n_earth_chords,
                                          transit_dtheta=transit_dtheta, n_n0_eV3=n_n0,
                                          R_H_eV_inv=1.0 / H0,
                                          show_progress=show_progress, verbose=verbose, n_cores=n_cores,
                                          N_steps_earth=N_steps_earth, external_pool=external_pool,
                                          U_LRI_struct=sp["S_eff"], V_SM_struct=sp["V_SM_eff"],
                                          use_gpu=use_gpu, earth_step_scheme=earth_step_scheme,
                                          earth_zenith_average=earth_zenith_average,
                                          earth_sky=earth_sky, earth_band_costh=earth_band_costh,
                                          earth_nodes_per_band=earth_nodes_per_band,
                                          sun_path_cos=sun_path_cos, arrival=arrival,
                                          matter_charge=matter_charge,
                                          debye_sq_z0=(debye_sq(T_nu0) if lagrangian else None),
                                          potential_product=pp, earth_transfer=earth_transfer)
        if earth_band_costh is not None:
            F_earth, F_bands = F_earth
            if not np.isfinite(F_bands).all():
                raise ValueError(f"Non-finite per-band flux out of the spatial handover at m_Zd_eV={m_Zd:.6g}.")
            bands.append(F_bands)
        # DOMAIN CHECK: same contract as the thermal-grid check above -- never return NaN
        # silently. (The known non-finite mechanism here, unitarity loss of the spatial
        # propagator at large phases, is excluded by propagator_3x3's exactly unitary form
        # since 2026-10-12; this is the safety net for anything not yet diagnosed.)
        if not np.isfinite(F_earth).all():
            raise ValueError(
                f"Non-finite flux out of the spatial handover at m_Zd_eV={m_Zd:.6g}, "
                f"g_D={g_D:.6g}, epsilon_prime={eps_prime:.6g} "
                f"({'antineutrino' if sp['anti'] else 'neutrino'} pass): these dark-portal "
                f"parameter values are outside the numerically supported domain.")
        out.append(F_earth)
    if earth_band_costh is not None:
        return out[0], out[1], bands[0], bands[1]
    return out[0], out[1]


# =============================================================================
# FILE EXPORT ROUTINE
# =============================================================================

def write_run_manifest(filename: str, config: Dict, runtime_s: float = None):
    """Writes a reproducibility manifest (JSON) alongside an output file: the full
    resolved configuration, the git commit of the code that produced it, package
    versions, host, timestamp, and runtime. Paper-level numbers should be traceable to
    an exact code + configuration state; this makes that automatic."""
    import platform
    import subprocess
    import datetime

    git_hash = None
    try:
        git_hash = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass

    versions = {"python": platform.python_version(), "numpy": np.__version__}
    if HAS_NUMBA:
        versions["numba"] = numba.__version__

    manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_hash,
        "hostname": platform.node(),
        "versions": versions,
        "runtime_s": runtime_s,
        "config": config,
    }
    with open(filename, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    safe_print(f"Run manifest saved to: {os.path.abspath(filename)}")

def save_flux_to_file(E_array: np.ndarray, Flux_array: np.ndarray, filename: str = "../output/data/output_flux.txt"):
    """Saves the generated flux components (Energy, nue, numu, nutau) to an external text file."""
    header = "Energy[eV]    Flux_nue[eV^-1 cm^-2 s^-1 sr^-1]    Flux_numu[eV^-1 cm^-2 s^-1 sr^-1]    Flux_nutau[eV^-1 cm^-2 s^-1 sr^-1]"
    data = np.column_stack((E_array, Flux_array))
    np.savetxt(filename, data, header=header, fmt="%.6e")
    safe_print(f"\nFlux data saved to: {os.path.abspath(filename)}")


SI_COSMOLOGY_MIN_M_EV = 1e3


def _use_si_cosmology(params: Dict, E_bins: np.ndarray) -> bool:
    """Whether the cosmological phase goes through dp_si's vector transport (dark_portal.cosmo_transport: 'auto' --
    when the cosmological potential, and the host galaxy's, stay below 1e-4 of the smallest vacuum splitting at every
    energy and redshift, i.e. the MeV regime, decided 2026-10-14; 'si' forces it, 'dark_portal' never). The bound
    used: sigma = 2 E (1+z) V <= 2 E_max (1+z_max)^4 |g_D eps'| K n_max / m^2, n_max the larger of the cosmic charge
    density at z_max and a host galaxy's local one."""
    mode = str(params.get('cosmo_transport', 'auto')).strip().lower()
    if mode not in ('auto', 'si', 'dark_portal'):
        raise ValueError(f"dark_portal.cosmo_transport must be 'auto', 'si' or 'dark_portal', got {mode!r}")
    if mode == 'dark_portal' or params.get('absorption_model', 'lagrangian') != 'lagrangian':
        return False
    if mode == 'si':
        return True
    m = float(params['m_Zd_eV'])
    if m < SI_COSMOLOGY_MIN_M_EV:
        # dp_si's collision tables are built and validated for keV and heavier mediators (its planes); below,
        # the resonance falls below the band and the dark-portal phase (resonance only) is used
        return False
    K = (4.0 * 0.88 * (91.18 * GeV) ** 2 / 0.65) * (np.sqrt(2.0) / 2.0 * G_F)
    n_q0 = dp_matter.cosmic_density(params.get('matter_charge', None), params['n_n0_eV3'])
    zp1 = 1.0 + float(params['z_max'])
    n_max = n_q0 * zp1 ** 3
    s_host = float(params.get('host_galaxy_scale', 0.0))
    if s_host > 0.0:
        from dp_spatial import calc_MW_neutron_density
        n_max = max(n_max, s_host * calc_MW_neutron_density(0.0, 0.0, 0.0) * cm_inv3_to_eV3 * 10.0)
    pp = params.get('potential_product', None)
    coupling = abs(float(params['g_D']) * float(params['epsilon_prime']) if pp is None else float(pp))
    sigma_max = 2.0 * float(E_bins[-1]) * zp1 * coupling * K * n_max / (m * m)
    return sigma_max < 1e-4 * float(params['dm21_sq'])


def _absorption_setup(params: Dict, U_pmns: np.ndarray, u_arr: np.ndarray, anti: bool = False):
    """dp_absorption.AbsorptionSetup of a parameter point: the coupling (params['G_coupling'], else the dark
    portal's) and the visible width (params['Gamma_vis'], else the Z mixing's)."""
    G_c = params.get('G_coupling', None)
    if G_c is None:
        G_c = dp_absorption.dark_portal_coupling(params['g_D'], u_arr, params['epsilon_prime'])
    G_vis = params.get('Gamma_vis', None)
    if G_vis is None:
        G_vis = dp_absorption.z_mixing_visible_width(params['m_Zd_eV'], params['epsilon_prime'])
    return dp_absorption.AbsorptionSetup(G_c, U_pmns, params['m_Zd_eV'], G_vis, anti=anti)


def _flux_setup(E_bins: np.ndarray, params: Dict) -> Dict:
    """Per-run quantities shared by every flux of one parameter point: the redshift grid,
    the source vector, the mass and mixing matrices and the thermal-rate grid (identical
    for neutrinos and antineutrinos, and independent of the source composition).

    Arguments: E_bins -- energy grid (eV); params -- the dict built by build_solver_params.
    Returns: a dict of those quantities for _flux_from_setup.
    """
    z_bins = np.linspace(params['z_max'], 0.0, int(params['n_z_bins']))
    rho_src = np.array([params['f_e_S'], params['f_mu_S'], params['f_tau_S']], dtype=np.float64)
    
    ordering = str(params.get('mass_ordering', 'normal')).strip().lower()
    # Inverted ordering: the configured splitting is |Delta m^2_32| (NuFIT's Delta m^2_3l, l = 2), as the
    # absolute masses read it (dp_core.get_mass_eigenvalues), so m_3^2 - m_1^2 = -(|Delta m^2_32| - Delta m^2_21);
    # until 2026-10-14 this entry was -|Delta m^2_32| (docs/dev/PHYSICS_AUDIT.md, 3e). Normal: Delta m^2_31.
    dm31_sq_signed = (-params['dm31_sq'] + params['dm21_sq']) if ordering == "inverted" else params['dm31_sq']
    
    # Enforce C-contiguous alignment for the initial matrix state
    M_sq = np.ascontiguousarray(np.diag([0.0, params['dm21_sq'], dm31_sq_signed]).astype(np.complex128))
    
    u_arr_base = np.ascontiguousarray(np.array([params['u_e'], params['u_mu'], params['u_tau']], dtype=np.complex128))
    eta_bg = params['eta_bg']
    U_pmns_base = np.ascontiguousarray(construct_PMNS(params['th12'], params['th13'], params['th23'], params['d_cp']))

    # Pin numba's thread pool BEFORE the thermal-grid precompute below: that prange kernel
    # previously ran ahead of propagate_spectrum's own set_num_threads call, so inside a
    # scanner pool worker (n_cores=1) the first evaluation used numba's default thread
    # count (all cores) -- transient oversubscription multiplied across every worker.
    if HAS_NUMBA:
        dp_core.set_numba_threads(params.get('n_cores', 1))

    if params.get('si_enabled', False):
        # Secret interactions (dp_si): no thermal grid, no potentials; the couplings in the mass
        # basis and the relic densities per mass state (the cosmology's eta, as for the dark portal)
        import dp_si
        g_mass = dp_si.mass_basis_couplings_model(params['si']['g_flavor'], U_pmns_base, params['si']['mediator'],
                                                  params['si']['majorana'])
        eta_mass = (np.abs(U_pmns_base) ** 2).T @ np.asarray(eta_bg, dtype=np.float64)
        return dict(si=True, rho_src=rho_src, U_pmns_base=U_pmns_base, g_mass=g_mass, eta_mass=eta_mass)

    # The thermal-rate grid depends only on |U_pmns| and |u_arr| magnitudes, which
    # conjugation never changes -- so it is identical for the neutrino and antineutrino
    # calls below. Computing it once here and sharing it (instead of letting each flavor
    # call redo this expensive O(N_z * N_E) precomputation from scratch) roughly halves
    # that stage's cost whenever both flavors are evaluated (is_antineutrino="sum", the
    # default). build_thermal_grid can additionally sample z on a coarse subgrid and
    # interpolate (grid.n_z_thermal; OFF by default -- see its docstring for the measured
    # error budget before enabling).
    sum_G, sum_G_alpha, eta_mass = compute_gamma_grid_inputs(U_pmns_base, u_arr_base, params['g_D'], params['epsilon_prime'], eta_bg)
    si_cosmo = _use_si_cosmology(params, E_bins)
    if si_cosmo:
        Gamma_grid_all = np.zeros((len(z_bins), len(E_bins), 3))     # not used: dp_si's transport (MeV regime)
    elif params.get('absorption_model', 'lagrangian') == 'lagrangian':
        # each relic mass state's rate per unit weight (dp_absorption; propagate_spectrum's absorption_model)
        ab = _absorption_setup(params, U_pmns_base, u_arr_base)
        Gamma_grid_all = relic_state_grid_cached(E_bins, z_bins, ab.sum_G, params['m_Zd_eV'], params['m_nu_bg_arr'],
                                                 params['T_nu0_eV'], eta_mass, moller=True)
    else:
        Gamma_grid_all = thermal_grid_cached(E_bins, z_bins, sum_G, sum_G_alpha, params['m_Zd_eV'], params['m_nu_bg_arr'], params['T_nu0_eV'], eta_mass,
                                             n_z_thermal=params.get('n_z_thermal', 0),
                                             thermal_analytic=params.get('thermal_analytic', True))

    return dict(z_bins=z_bins, rho_src=rho_src, M_sq=M_sq, u_arr_base=u_arr_base, eta_bg=eta_bg,
                U_pmns_base=U_pmns_base, Gamma_grid_all=Gamma_grid_all, si_cosmology=si_cosmo)


def _flux_from_setup(E_bins: np.ndarray, params: Dict, setup: Dict, injection,
                     rho_src: np.ndarray = None, band_costh=None, arrival_flavor=None,
                     arrival_basis: str = "flavor"):
    """The neutrino and antineutrino fluxes at the detector, each (N_E, 3), from a _flux_setup.

    Arguments: E_bins, params, setup -- as for _flux_setup; injection -- (w_nu, w_nb), the
    weights of the neutrino and antineutrino sources; rho_src -- source flavor composition
    (default: the configured one); band_costh -- None, or the cos(zenith) edges of declination
    bands (per-direction output): then also the per-band fluxes, (F_nu, F_nb, B_nu, B_nb) with
    B (N_b, N_E, 3). Secret interactions are isotropic: every band is the whole sky.
    arrival_flavor, arrival_basis -- dark portal only: sources inside the Galaxy at the local potential with
    this (3,) flavor composition, or ('eigen') these local-eigenstate populations (propagate_spectrum; the
    Galactic component, _galactic_transfer).
    """
    if rho_src is None:
        rho_src = setup['rho_src']
    if setup.get('si', False):
        # Secret interactions: dp_si propagates both helicities of each mass state together (exact
        # for their sum whatever the injection, the interaction being CP-symmetric); the sum is
        # split here in the injection's proportions, which is what a detector without charge
        # separation needs -- the actual nu/nubar split of the secondaries is not modelled.
        import dp_si
        w_nu, w_nb = float(injection[0]), float(injection[1])
        w_tot = w_nu + w_nb
        if w_tot <= 0.0:
            z = np.zeros((len(E_bins), 3))
            if band_costh is not None:
                zb = np.zeros((len(band_costh) - 1, len(E_bins), 3))
                return z, z.copy(), zb, zb.copy()
            return z, z.copy()
        si_p = params['si']
        U = setup['U_pmns_base']
        F_mass = dp_si.propagate_si(E_bins, params['z_max'], params['H0_eV'], params['Omega_m'], params['Omega_L'],
                                    U, params['m_nu_bg_arr'], params['T_nu0_eV'], setup['eta_mass'], rho_src,
                                    params['p_index'], params['rho_evol'], params['Phi_0'] * w_tot,
                                    si_p['m_phi_eV'], setup['g_mass'], majorana=si_p['majorana'],
                                    channels=si_p['channels'], phiphi=si_p['phiphi'],
                                    z_substeps=si_p['z_substeps'], E_extension=si_p['E_extension'],
                                    mediator=si_p['mediator'], dirac_sterile=si_p['dirac_sterile'],
                                    inj_window=params.get('inj_window_eV', (0.0, np.inf)))
        F_fl = np.ascontiguousarray(F_mass @ (np.abs(U) ** 2).T)
        if not np.isfinite(F_fl).all():
            raise ValueError(f"Non-finite flux from the secret-interaction transport at m_phi_eV="
                             f"{si_p['m_phi_eV']:.6g}, g_phi texture {si_p['g_flavor'].tolist()}.")
        F_nu, F_nb = F_fl * (w_nu / w_tot), F_fl * (w_nb / w_tot)
        if band_costh is not None:
            n_b = len(band_costh) - 1
            return F_nu, F_nb, np.repeat(F_nu[None], n_b, axis=0), np.repeat(F_nb[None], n_b, axis=0)
        return F_nu, F_nb
    return propagate_spectrum(
        E_bins, setup['z_bins'], rho_src,
        params['H0_eV'], params['Omega_m'], params['Omega_L'],
        setup['U_pmns_base'], setup['M_sq'], setup['u_arr_base'],
        params['m_Zd_eV'], params['g_D'], params['epsilon_prime'],
        params['m_nu_bg_arr'], params['n_n0_eV3'], params['T_nu0_eV'],
        params['use_debye_screening'], params['detector_depth_km'],
        params['use_SM_potential'], params['n_e_earth_eV3'],
        injection, params['p_index'], params['rho_evol'], params['Phi_0'], setup['eta_bg'],
        params['n_sun_paths'], params['n_earth_chords'],
        params['transit_dtheta'],
        show_progress=params.get('show_progress', True), verbose=params.get('verbose', 1),
        n_cores=params.get('n_cores', 1),
        precomputed_Gamma_grid_all=setup['Gamma_grid_all'], N_steps_earth=params.get('N_steps_earth', 500),
        U_LRI_struct=params.get('U_LRI_struct', None), V_SM_struct=params.get('V_SM_struct', None),
        use_gpu=params.get('use_gpu', False),
        thermal_analytic=params.get('thermal_analytic', True),
        earth_step_scheme=params.get('earth_step_scheme', 'left'),
        earth_zenith_average=params.get('earth_zenith_average', 'plain'),
        earth_sky=params.get('earth_sky', 'up'),
        host_galaxy_scale=params.get('host_galaxy_scale', 0.0),
        earth_band_costh=band_costh,
        earth_nodes_per_band=params.get('sky', {}).get('earth_nodes_per_band', 2),
        sun_path_cos=params.get('sun_path_cos', None),
        arrival_flavor=arrival_flavor, arrival_basis=arrival_basis,
        matter_charge=params.get('matter_charge', None),
        absorption_model=params.get('absorption_model', 'lagrangian'),
        G_coupling=params.get('G_coupling', None), Gamma_vis=params.get('Gamma_vis', None),
        si_cosmology=setup.get('si_cosmology', False),
        si_settings=dict(z_substeps=params.get('si', {}).get('z_substeps', 1),
                         E_extension=params.get('si', {}).get('E_extension', 10.0),
                         thermal_products=params.get('thermal_products', False)),
        potential_product=params.get('potential_product', None),
        inj_window=params.get('inj_window_eV', (0.0, np.inf)),
        earth_transfer=params.get('earth_transfer', 'off')
    )


def _galactic_transfer(E_bins: np.ndarray, params: Dict, setup: Dict, band_costh, nside: int = 16, f_S=None):
    """The Galactic component's flavor transfer (docs/dev/DIRECTIONS_PLAN.md, steps 5 and 6), for sources of
    composition f_S (default (x, 1 - x, 0), x = galactic.f_e_S_gal), per unit flux emitted inside the Galaxy (all
    flavors, per kind); bands in the order of band_costh (cos(zenith), increasing).

    Secret interactions: the averaged oscillation probabilities, the same in every band (the Galactic relic
    column is ~1e-5 of the cosmological one: unattenuated; Section 3.5). Dark portal, emission 'local': emitted
    at the local potential, then the Sun and the Earth per band (propagate_spectrum's arrival_flavor). Both
    return (T_nu, T_nb), each (N_b, N_E, 3).

    Dark portal, emission '3d' (step 6): emitted at points along each line of sight (dp_galactic's emission
    weights and Yukawa-smoothed densities), adiabatically to the local eigenstates: returns a dict with
    F_nu, F_nb (N_pix, N_E, 3), each pixel's local-eigenstate populations, and R_nu, R_nb (N_b, N_E, 3, 3),
    each band's flavor response to a unit population of each local eigenstate (_galactic_responses)."""
    import dp_galactic
    gal = params['galactic']
    f_S = dp_galactic.source_composition(gal['f_e_S_gal']) if f_S is None else np.asarray(f_S, dtype=np.float64)
    n_b = len(band_costh) - 1
    if setup.get('si', False):
        T = np.broadcast_to(dp_galactic.averaged_transfer(setup['U_pmns_base']) @ f_S, (n_b, len(E_bins), 3))
        return np.ascontiguousarray(T), np.ascontiguousarray(T)
    if gal['emission'] == 'local':
        _, _, T_nu, T_nb = _flux_from_setup(E_bins, params, setup, (1.0, 1.0), band_costh=band_costh,
                                            arrival_flavor=f_S)
        return T_nu, T_nb
    R_nu, R_nb = _galactic_responses(E_bins, params, setup, band_costh)
    F = _galactic_emission_populations(E_bins, params, setup, nside, f_S=f_S)
    return dict(F_nu=F[0], F_nb=F[1], R_nu=R_nu, R_nb=R_nb)


def _galactic_responses(E_bins: np.ndarray, params: Dict, setup: Dict, band_costh):
    """(R_nu, R_nb), each (N_b, N_E, 3, 3) in the order of band_costh: each band's flavor flux at the detector per
    unit population of each local eigenstate (ascending eigenvalues), from three spatial runs per kind
    (propagate_spectrum, arrival_basis 'eigen'); independent of the Galactic sources."""
    R = np.empty((2, len(band_costh) - 1, len(E_bins), 3, 3))
    for k in range(3):
        e_k = np.zeros(3)
        e_k[k] = 1.0
        _, _, B_nu, B_nb = _flux_from_setup(E_bins, params, setup, (1.0, 1.0), band_costh=band_costh,
                                            arrival_flavor=e_k, arrival_basis='eigen')
        R[0, :, :, k, :], R[1, :, :, k, :] = B_nu, B_nb
    return R[0], R[1]


def _galactic_emission_populations(E_bins: np.ndarray, params: Dict, setup: Dict, nside: int, f_S=None):
    """(F_nu, F_nb), each (N_pix, N_E, 3): per pixel, the local-eigenstate populations of the Galactic flux
    emitted at its line of sight's emission points (dp_galactic.emission_populations), each kind with its
    own potential sign, texture and frame table -- the same ones the transit uses (lri_coefficient,
    _species_matrices, dp_transit.frame_table)."""
    import dp_galactic
    import dp_transit
    from dp_spatial import lri_coefficient
    gal = params['galactic']
    f_S = dp_galactic.source_composition(gal['f_e_S_gal']) if f_S is None else np.asarray(f_S, dtype=np.float64)
    E_GeV = np.asarray(E_bins, dtype=np.float64) / 1e9
    # Coarser maps (e.g. N_side 8 for cascades) are averaged from the stored resolution's pixels, each along its
    # own line of sight and weighted by its emitted intensity: a coarse pixel's centre can lie degrees off the
    # plane its gas column contains, which would misplace its emission points.
    work = max(int(nside), 16)
    w_gas = dp_galactic.gas_share(gal['template'], E_GeV, work)
    q_gas, q_src = dp_galactic.emission_weights(work)
    _, _, R, z = dp_galactic.emission_geometry(work)
    out = []
    for anti in (False, True):
        u_s, M_rot, S_eff, _ = _species_matrices(setup['U_pmns_base'], setup['M_sq'], setup['u_arr_base'],
                                                 params.get('U_LRI_struct', None), params.get('V_SM_struct', None),
                                                 anti)
        debye = None
        if params.get('absorption_model', 'lagrangian') == 'lagrangian':
            import dp_absorption as _dpa
            ab = _absorption_setup(params, setup['U_pmns_base'], setup['u_arr_base'])
            eta_m = (np.abs(setup['U_pmns_base']) ** 2).T @ np.asarray(setup['eta_bg'], dtype=np.float64)
            debye = _dpa.debye_mass_sq(ab.G_mass_diag2, params['m_nu_bg_arr'], params['T_nu0_eV'], eta_m)
        Coef, m_eff = lri_coefficient(params['m_Zd_eV'], params['g_D'], params['epsilon_prime'], u_s,
                                      params['T_nu0_eV'], params['use_debye_screening'], setup['eta_bg'], anti,
                                      debye_sq=debye, potential_product=params.get('potential_product', None))
        charge = params.get('matter_charge', None)
        if charge is not None:
            Coef = Coef * charge.sign
        S_signed = np.ascontiguousarray(np.sign(Coef) * S_eff if Coef != 0.0 else S_eff)
        tab = dp_transit.frame_table(M_rot, S_signed, params['transit_dtheta'])
        if charge is None:
            n_emit = dp_galactic.effective_density(R, z, m_eff, params['n_n0_eV3'], 1.0 / params['H0_eV'])
        else:
            n_emit = dp_matter.galactic_emission_density(charge, R, z, m_eff, params['n_n0_eV3'],
                                                         1.0 / params['H0_eV'])
        if debye is not None and m_eff > params['m_Zd_eV']:
            # the uniform cosmic mean is not screened (as dp_spatial.transit_to_earth's m_bare)
            R_H = 1.0 / params['H0_eV']
            n_floor = dp_matter.cosmic_density(charge, params['n_n0_eV3'])
            n_emit = n_emit + n_floor * m_eff * m_eff * (yukawa_horizon_inv_m2(params['m_Zd_eV'] ** 2, R_H)
                                                         - yukawa_horizon_inv_m2(m_eff * m_eff, R_H))
        st = None
        if work != nside and GALACTIC_EMISSION_REGROUPED:
            # the emission's mixing-independent sums kept per density pattern (dp_galactic.emission_structure;
            # 2026-10-18): the same result to rounding, without re-averaging the full-resolution map per run
            import hashlib
            n_c = np.ascontiguousarray(n_emit, dtype=np.float64)
            key = (gal['template'], hashlib.blake2b(E_GeV.tobytes(), digest_size=16).digest(), work, int(nside),
                   n_c.shape, hashlib.blake2b(n_c.tobytes(), digest_size=16).digest())
            st = dp_galactic.emission_structure(key, w_gas, q_gas, q_src, n_c,
                                                dp_galactic.intensity(gal['template'], E_GeV, work), work, nside)
        if work != nside and GALACTIC_EMISSION_REGROUPED and st is not None:
            F = dp_galactic.emission_populations_degraded(E_bins, f_S, n_c, 2.0 * abs(Coef), tab, st)
        else:
            F = dp_galactic.emission_populations(E_bins, f_S, w_gas, q_gas, q_src, n_emit, 2.0 * abs(Coef), tab)
            if work != nside:
                F = dp_galactic.degrade_weighted(F, dp_galactic.intensity(gal['template'], E_GeV, work), work, nside)
        out.append(F)
    return out


# dp_galactic.emission_structure / emission_populations_degraded for the coarser maps (False: the direct path, as
# before 2026-10-18; tests/test_speedups.py compares the two)
GALACTIC_EMISSION_REGROUPED = True


def injection_weights(params: Dict) -> Tuple[float, float]:
    """(w_nu, w_nb), the weights of the neutrino and antineutrino sources: from
    source.is_antineutrino ('false': neutrinos only; 'true': antineutrinos only; 'sum':
    both, in the ratio source.nubar_to_nu_ratio)."""
    flag_val = str(params.get('is_antineutrino', 'sum')).strip().lower()
    if flag_val in ['sum', 'both']:
        ratio = float(params.get('nubar_to_nu_ratio', 1.0))
        if not np.isfinite(ratio):
            return 0.0, 1.0
        return 1.0 / (1.0 + ratio), ratio / (1.0 + ratio)
    if flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        return 0.0, 1.0
    return 1.0, 0.0


def propagate_weighted(E_bins: np.ndarray, params: Dict, sources=None) -> np.ndarray:
    """The flux at the detector (neutrinos plus antineutrinos, (N_E, 3)) for the configured
    sources (injection_weights) -- one coupled evolution. With sources, a sequence of (3,)
    source compositions sharing one thermal-rate grid, returns (n_sources, N_E, 3)."""
    setup = _flux_setup(E_bins, params)
    w = injection_weights(params)
    if sources is None:
        return sum(_flux_from_setup(E_bins, params, setup, w))
    return np.stack([sum(_flux_from_setup(E_bins, params, setup, w, np.ascontiguousarray(np.asarray(r, dtype=np.float64))))
                     for r in sources])


def propagate_nu_nubar(E_bins: np.ndarray, params: Dict, sources=None):
    """The flux at the detector (neutrinos plus antineutrinos) for neutrino sources alone and
    for antineutrino sources alone, per unit injection -- for callers that weight the two
    themselves, such as the analysis package with a free nubar/nu ratio (the fluxes are
    linear in the injection). Since 2026-10-13 each is a coupled evolution: neutrino sources
    also produce antineutrinos, through the secondaries of every absorption.

    Arguments:
      E_bins: energy grid (eV); params: the dict built by build_solver_params.
      sources: None for the configured source composition, or a sequence of (3,)
        compositions sharing one thermal-rate grid.
    Returns: (F_from_nu, F_from_nb), each (N_E, 3) -- or (n_sources, N_E, 3) when sources
    is given.
    """
    setup = _flux_setup(E_bins, params)
    srcs = [None] if sources is None else [np.ascontiguousarray(np.asarray(r, dtype=np.float64)) for r in sources]
    out_nu = [sum(_flux_from_setup(E_bins, params, setup, (1.0, 0.0), r)) for r in srcs]
    out_nb = [sum(_flux_from_setup(E_bins, params, setup, (0.0, 1.0), r)) for r in srcs]
    if sources is None:
        return out_nu[0], out_nb[0]
    return np.stack(out_nu), np.stack(out_nb)


def propagate_sky(E_bins: np.ndarray, params: Dict) -> Dict:
    """The flux at the detector per direction (docs/dev/DIRECTIONS_PLAN.md, steps 3 and 5): per declination
    band, neutrinos and antineutrinos separately, for the configured sources (injection_weights); with
    sky.output = 'healpix' also the HEALPix pixel of each band (maps are the band values per pixel while
    the Sun is averaged isotropically). Fluxes are per unit solid angle, in the solver's units.

    The Earth average is taken per band of cos(zenith) = -sin(dec) (dp_spatial.average_earth_chords_bands,
    the band edges as chord nodes), so the whole-sky flux here equals the bands weighted by solid angle to
    rounding; it differs from propagate_weighted's by the chord set's discretization (~1e-5).
    Secret interactions are isotropic: every band is the whole sky.

    With galactic.enabled, also the Galactic-plane component (dp_galactic; step 5), in physical units
    (GeV^-1 cm^-2 s^-1 sr^-1, per flavor and per kind): the template per pixel times each kind's transfer
    per band (_galactic_transfer), i.e. galactic_bands_nu/nubar (N_b, N_E, 3) and galactic_allsky_nu/nubar
    (N_E, 3) (pixel means), and with sky.output = 'healpix' the factors of the maps: galactic_emitted
    (N_pix, N_E), the all-flavor intensity emitted per kind, and galactic_transfer_nu/nubar (N_b, N_E, 3)
    (dp_galactic.pixel_flux multiplies them out).

    Returns a dict: E_eV (N_E,); dec_band_edges (rad, increasing); band_omega (sr); allsky_nu,
    allsky_nubar (N_E, 3); bands_nu, bands_nubar (N_b, N_E, 3), in declination order; nside and
    pixel_band (N_pix,) when sky.output = 'healpix'; the galactic_* entries above; meta (frame,
    conventions, units)."""
    import dp_sky
    sky_p = params.get('sky') or dict(output='bands', nside=16, dec_band_edges=None, earth_nodes_per_band=2)
    gal = params.get('galactic') or dict(enabled=False)
    if not params.get('si') and params.get('earth_sky', 'up') != 'full':
        raise ValueError("propagate_sky needs grid.earth_sky = 'full' (bands cover the whole sky).")
    grid = dp_sky.SkyGrid(sky_p['nside'], sky_p['dec_band_edges'])
    if gal.get('enabled', False) and np.any(np.bincount(grid.band, minlength=grid.n_bands) == 0):
        raise ValueError(f"The Galactic component needs a pixel centre in every declination band: N_side "
                         f"{grid.nside} leaves {int(np.sum(np.bincount(grid.band, minlength=grid.n_bands) == 0))} "
                         f"of {grid.n_bands} bands empty (use a finer N_side or wider bands).")
    setup = _flux_setup(E_bins, params)
    F_nu, F_nb, B_nu, B_nb = _flux_from_setup(E_bins, params, setup, injection_weights(params),
                                              band_costh=grid.costh_edges)
    out = dict(E_eV=np.asarray(E_bins, dtype=np.float64), dec_band_edges=grid.band_edges,
               band_omega=grid.band_omega, allsky_nu=F_nu, allsky_nubar=F_nb,
               # the Earth's bands run in cos(zenith) = -sin(dec), increasing: reversed declination order
               bands_nu=np.ascontiguousarray(B_nu[::-1]), bands_nubar=np.ascontiguousarray(B_nb[::-1]),
               meta=dict(frame="equatorial J2000 (FK5)", healpix_ordering="RING",
                         healpix_theta="pi/2 - dec", detector="South Pole: cos(zenith) = -sin(dec)",
                         units="per unit solid angle, solver units (Phi_0 E^-p)",
                         sun="isotropic Gauss-Legendre average", galactic="none"))
    if sky_p.get('extragalactic_norm') is not None:
        E_ref = sky_p.get('extragalactic_norm_E_GeV', 1e5)
        K = extragalactic_scale(params, sky_p['extragalactic_norm'], E_ref)
        for key in ('allsky_nu', 'allsky_nubar', 'bands_nu', 'bands_nubar'):
            out[key] = out[key] * K
        out['meta']['units'] = ("GeV^-1 cm^-2 s^-1 sr^-1 per flavor and kind; without new physics the per-flavor "
                                f"nu + nubar intensity at {E_ref:.4g} GeV would be {sky_p['extragalactic_norm']:.4g}")
        out['meta']['extragalactic_scale'] = K
    if sky_p['output'] == 'healpix':
        out['nside'] = grid.nside
        out['pixel_band'] = grid.band
    if gal.get('enabled', False):
        import dp_galactic
        em = dp_galactic.emitted(gal['template'], np.asarray(E_bins, dtype=np.float64) / 1e9, grid.nside,
                                 gal['norm_gal'])
        share = np.bincount(grid.band, minlength=grid.n_bands) / grid.npix
        res = _galactic_transfer(E_bins, params, setup, grid.costh_edges, grid.nside)
        if isinstance(res, dict):
            # emission points in 3-D: populations per pixel, responses per band (declination order)
            for kind, key in (("nu", "nu"), ("nubar", "nb")):
                F, Rb = res['F_' + key], np.ascontiguousarray(res['R_' + key][::-1])
                G = dp_galactic.band_means(em[:, :, None] * F, grid.band, grid.n_bands)
                out['galactic_bands_' + kind] = np.einsum('bek,bekf->bef', G, Rb)
                out['galactic_allsky_' + kind] = np.tensordot(share, out['galactic_bands_' + kind], axes=(0, 0))
                if sky_p['output'] == 'healpix':
                    out['galactic_populations_' + kind] = F
                    out['galactic_response_' + kind] = Rb
            if sky_p['output'] == 'healpix':
                out['galactic_emitted'] = em
        else:
            T_nu, T_nb = np.ascontiguousarray(res[0][::-1]), np.ascontiguousarray(res[1][::-1])
            em_b = dp_galactic.band_means(em, grid.band, grid.n_bands)
            for kind, T in (("nu", T_nu), ("nubar", T_nb)):
                out['galactic_bands_' + kind] = em_b[:, :, None] * T
                out['galactic_allsky_' + kind] = np.tensordot(share, out['galactic_bands_' + kind], axes=(0, 0))
            if sky_p['output'] == 'healpix':
                out.update(galactic_emitted=em, galactic_transfer_nu=T_nu, galactic_transfer_nubar=T_nb)
        out['meta']['galactic'] = dict(
            template=gal['template'], norm_gal=gal['norm_gal'], f_S=dp_galactic.source_composition(gal['f_e_S_gal']).tolist(),
            emission='unattenuated, averaged oscillations' if params.get('si') else gal['emission'],
            emission_model=None if (params.get('si') or gal['emission'] != '3d') else dict(
                dp_galactic.SOURCES, gas="Soding et al. 2025 x CRINGE nucleon flux at 20 x 100 TeV",
                neff="src/data/mw_neff_tables.npz", nodes_per_decade=dp_galactic.NEFF_NODES_PER_DECADE),
            nu_to_nubar="1:1", units="GeV^-1 cm^-2 s^-1 sr^-1, per flavor, per kind (nu or nubar)",
            band_means="pixel means (pixels assigned to bands by their centres)",
            source=dp_galactic.load_template(gal['template'], grid.nside)[2]['source'])
    return out


def site_zenith_weights(dec: np.ndarray, lat_deg: float, cosz_edges: np.ndarray, n_hour: int = 720,
                        dec_band=None) -> np.ndarray:
    """(N_pix, N_z): the fraction of a sidereal day each sky direction (declination dec, rad) spends in each local
    cos-zenith band of a site at latitude lat_deg, from cos Z = sin(lat) sin(dec) + cos(lat) cos(dec) cos(H) at n_hour
    hour angles. At a pole each direction stays in one band, cos Z = -+sin(dec): with dec_band (the directions'
    declination bands, SkyGrid.band, whose edges map onto cosz_edges) exactly the grid's own assignment, so that the
    South Pole reproduces the per-declination bands even for pixel centres on a band edge (HEALPix rings sit on
    sin(dec) = 0, +-0.5)."""
    if dec_band is not None and abs(abs(float(lat_deg)) - 90.0) < 1e-9:
        nz = len(cosz_edges) - 1
        z = (nz - 1 - np.asarray(dec_band)) if lat_deg < 0 else np.asarray(dec_band)
        W = np.zeros((len(dec), nz))
        W[np.arange(len(dec)), z] = 1.0
        return W
    H = (np.arange(n_hour) + 0.5) * (2.0 * np.pi / n_hour)
    la = np.radians(float(lat_deg))
    cz = np.sin(la) * np.sin(dec)[:, None] + np.cos(la) * np.cos(dec)[:, None] * np.cos(H)[None, :]
    idx = np.clip(np.searchsorted(cosz_edges, cz, side="right") - 1, 0, len(cosz_edges) - 2)
    W = np.zeros((len(dec), len(cosz_edges) - 1))
    for z in range(len(cosz_edges) - 1):
        W[:, z] = np.mean(idx == z, axis=1)
    return W


def site_band_means(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Per local cos-zenith band, the time-weighted mean of the sky maps X (N_pix, ...) over the directions passing
    through it (W: site_zenith_weights): the band's time-averaged intensity. At the South Pole, the pixel means per
    declination band (dp_galactic.band_means), in cos-zenith order."""
    X = np.asarray(X, dtype=np.float64)
    flat = X.reshape(len(W), -1)
    norm = W.sum(axis=0)
    if np.any(norm == 0):
        raise ValueError("a cos-zenith band receives no sky direction: use a finer N_side or more hour angles")
    return ((W.T @ flat) / norm[:, None]).reshape((W.shape[1],) + X.shape[1:])


def propagate_sky_basis(E_bins: np.ndarray, params: Dict, sources=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                        galactic_compositions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), galactic: bool = True,
                        site_latitudes_deg=()) -> Dict:
    """propagate_sky for several source compositions at once (the analysis' event likelihoods,
    docs/dev/EVENT_LIKELIHOOD_PLAN.md): the extragalactic bands for each composition in `sources` and, with
    galactic.enabled, the Galactic bands for each composition in `galactic_compositions`, sharing one setup
    (thermal-rate grid) and, for the dark portal's emission in 3-D, one set of band responses. The fluxes are
    linear in the compositions, so the default pairs (pure nu_e, pure nu_mu sources) give every (x, 1 - x, 0).

    Returns a dict: E_eV (N_E,); dec_band_edges (rad, increasing); band_omega (sr); bands_nu, bands_nubar
    (n_sources, N_b, N_E, 3) in declination order, per unit solid angle, for the configured nu/nubar injection
    (injection_weights), in GeV^-1 cm^-2 s^-1 sr^-1 when sky.extragalactic_norm is set (meta extragalactic_scale)
    and solver units otherwise; with galactic.enabled, galactic_bands_nu, galactic_bands_nubar (n_comp, N_b, N_E,
    3) in GeV^-1 cm^-2 s^-1 sr^-1 at galactic.norm_gal (pixel means per band) and galactic_band_share (N_b,), each
    band's share of the pixels (the sky's pixel mean, galactic_allsky, weights the bands by it); meta.
    galactic=False skips the Galactic bands (a caller that has them: they do not depend on the extragalactic
    source's spectral index, 2026-10-14).
    site_latitudes_deg: detectors elsewhere than the South Pole (the future detectors, docs/dev/FUTURE_DETECTORS_PLAN.md,
    2026-10-15): with galactic.enabled (3-D emission or secret interactions), site_galactic_bands_nu, _nubar {latitude:
    (n_comp, N_z, N_E, 3)} per local cos-zenith band (site_cosz_edges, increasing): each sky direction's emission
    spread over the site's zeniths over a sidereal day (site_zenith_weights), then through the Earth with the
    per-zenith responses. The extragalactic flux is isotropic before the Earth, so its bands in cos-zenith order
    (bands_* reversed) serve every site."""
    import dp_sky
    sky_p = params.get('sky') or dict(output='bands', nside=16, dec_band_edges=None, earth_nodes_per_band=2)
    gal = params.get('galactic') or dict(enabled=False)
    if not params.get('si') and params.get('earth_sky', 'up') != 'full':
        raise ValueError("propagate_sky_basis needs grid.earth_sky = 'full' (bands cover the whole sky).")
    grid = dp_sky.SkyGrid(sky_p['nside'], sky_p['dec_band_edges'])
    if gal.get('enabled', False) and np.any(np.bincount(grid.band, minlength=grid.n_bands) == 0):
        raise ValueError(f"The Galactic component needs a pixel centre in every declination band: N_side "
                         f"{grid.nside} leaves {int(np.sum(np.bincount(grid.band, minlength=grid.n_bands) == 0))} "
                         f"of {grid.n_bands} bands empty (use a finer N_side or wider bands).")
    setup = _flux_setup(E_bins, params)
    w = injection_weights(params)
    b_nu, b_nb = [], []
    for r in sources:
        _, _, B_nu, B_nb = _flux_from_setup(E_bins, params, setup, w, np.ascontiguousarray(np.asarray(r, dtype=np.float64)),
                                            band_costh=grid.costh_edges)
        # the Earth's bands run in cos(zenith) = -sin(dec), increasing: reversed declination order
        b_nu.append(B_nu[::-1])
        b_nb.append(B_nb[::-1])
    out = dict(E_eV=np.asarray(E_bins, dtype=np.float64), dec_band_edges=grid.band_edges, band_omega=grid.band_omega,
               bands_nu=np.ascontiguousarray(np.stack(b_nu)), bands_nubar=np.ascontiguousarray(np.stack(b_nb)),
               meta=dict(frame="equatorial J2000 (FK5)", detector="South Pole: cos(zenith) = -sin(dec)",
                         sources=[list(map(float, r)) for r in sources],
                         units="per unit solid angle, solver units (Phi_0 E^-p)", galactic="none"))
    if sky_p.get('extragalactic_norm') is not None:
        E_ref = sky_p.get('extragalactic_norm_E_GeV', 1e5)
        K = extragalactic_scale(params, sky_p['extragalactic_norm'], E_ref)
        out['bands_nu'] = out['bands_nu'] * K
        out['bands_nubar'] = out['bands_nubar'] * K
        out['meta']['units'] = ("GeV^-1 cm^-2 s^-1 sr^-1 per flavor and kind; without new physics the per-flavor "
                                f"nu + nubar intensity at {E_ref:.4g} GeV would be {sky_p['extragalactic_norm']:.4g}")
        out['meta']['extragalactic_scale'] = K
    if gal.get('enabled', False) and galactic:
        import dp_galactic
        em = dp_galactic.emitted(gal['template'], np.asarray(E_bins, dtype=np.float64) / 1e9, grid.nside,
                                 gal['norm_gal'])
        g_nu, g_nb = [], []
        sites = {float(lat): site_zenith_weights(grid.dec, lat, grid.costh_edges, dec_band=grid.band)
                 for lat in site_latitudes_deg}
        s_nu = {lat: [] for lat in sites}
        s_nb = {lat: [] for lat in sites}
        if setup.get('si', False):
            # unattenuated, averaged oscillations: the same transfer in every band (_galactic_transfer)
            P = dp_galactic.averaged_transfer(setup['U_pmns_base'])
            em_b = dp_galactic.band_means(em, grid.band, grid.n_bands)
            for c in galactic_compositions:
                T = em_b[:, :, None] * (P @ np.asarray(c, dtype=np.float64))[None, None, :]
                g_nu.append(T)
                g_nb.append(T.copy())
                for lat, W in sites.items():
                    Ts = site_band_means(em, W)[:, :, None] * (P @ np.asarray(c, dtype=np.float64))[None, None, :]
                    s_nu[lat].append(Ts)
                    s_nb[lat].append(Ts.copy())
        elif gal['emission'] == 'local':
            if sites:
                raise NotImplementedError("site_latitudes_deg needs galactic.emission '3d' (or secret interactions)")
            em_b = dp_galactic.band_means(em, grid.band, grid.n_bands)
            for c in galactic_compositions:
                T_nu, T_nb = _galactic_transfer(E_bins, params, setup, grid.costh_edges, grid.nside,
                                                f_S=np.asarray(c, dtype=np.float64))
                g_nu.append(em_b[:, :, None] * T_nu[::-1])
                g_nb.append(em_b[:, :, None] * T_nb[::-1])
        else:
            R_nu_z, R_nb_z = _galactic_responses(E_bins, params, setup, grid.costh_edges)      # cos-zenith order
            R_nu, R_nb = np.ascontiguousarray(R_nu_z[::-1]), np.ascontiguousarray(R_nb_z[::-1])
            for c in galactic_compositions:
                F_nu, F_nb = _galactic_emission_populations(E_bins, params, setup, grid.nside,
                                                            f_S=np.asarray(c, dtype=np.float64))
                g_nu.append(np.einsum('bek,bekf->bef', dp_galactic.band_means(em[:, :, None] * F_nu, grid.band,
                                                                              grid.n_bands), R_nu))
                g_nb.append(np.einsum('bek,bekf->bef', dp_galactic.band_means(em[:, :, None] * F_nb, grid.band,
                                                                              grid.n_bands), R_nb))
                for lat, W in sites.items():
                    s_nu[lat].append(np.einsum('zek,zekf->zef', site_band_means(em[:, :, None] * F_nu, W), R_nu_z))
                    s_nb[lat].append(np.einsum('zek,zekf->zef', site_band_means(em[:, :, None] * F_nb, W), R_nb_z))
        out['galactic_bands_nu'] = np.ascontiguousarray(np.stack(g_nu))
        out['galactic_bands_nubar'] = np.ascontiguousarray(np.stack(g_nb))
        # each band's share of the pixels: the sky's pixel mean (galactic_allsky) is the bands weighted by it
        out['galactic_band_share'] = np.bincount(grid.band, minlength=grid.n_bands) / grid.npix
        if sites:
            out['site_cosz_edges'] = np.asarray(grid.costh_edges, dtype=np.float64)
            out['site_galactic_bands_nu'] = {lat: np.ascontiguousarray(np.stack(v)) for lat, v in s_nu.items()}
            out['site_galactic_bands_nubar'] = {lat: np.ascontiguousarray(np.stack(v)) for lat, v in s_nb.items()}
        out['meta']['galactic'] = dict(template=gal['template'], norm_gal=gal['norm_gal'],
                                       compositions=[list(map(float, c)) for c in galactic_compositions],
                                       emission='unattenuated, averaged oscillations' if params.get('si')
                                       else gal['emission'], nu_to_nubar="1:1",
                                       units="GeV^-1 cm^-2 s^-1 sr^-1, per flavor, per kind (nu or nubar)",
                                       band_means="pixel means (pixels assigned to bands by their centres)")
    return out


def extragalactic_scale(params: Dict, per_flavor: float, E_ref_GeV: float = 1e5) -> float:
    """The factor taking the solver's extragalactic flux to GeV^-1 cm^-2 s^-1 sr^-1 such that, without new
    physics, its per-flavor nu + nubar intensity at E_ref_GeV is per_flavor: the free-streaming flux
    Phi_0 E^-p int dz (1+z)^-p R(z) / H(z) (dp_si.free_streaming_flux, the closed form; the solver's own
    no-coupling flux equals it to its redshift grid's first-order error, 3e-3 at 400 steps) over three flavors.
    The same factor scales the flux with new physics, which keeps its absorption and conversion."""
    import dp_si
    U = construct_PMNS(params['th12'], params['th13'], params['th23'], params['d_cp'])
    rho = np.array([params['f_e_S'], params['f_mu_S'], params['f_tau_S']], dtype=np.float64)
    tot = float(dp_si.free_streaming_flux(np.array([float(E_ref_GeV) * 1e9]), params['z_max'], params['H0_eV'],
                                          params['Omega_m'], params['Omega_L'], U, rho / rho.sum(),
                                          params['p_index'], params['rho_evol'], params['Phi_0']).sum())
    return float(per_flavor) / (tot / 3.0)


def galactic_allsky(E_bins: np.ndarray, params: Dict, compositions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))) -> np.ndarray:
    """The Galactic component's sky-averaged flux at the detector, neutrinos plus antineutrinos per flavor, in
    GeV^-1 cm^-2 s^-1 sr^-1 with galactic.norm_gal, for each source composition: (n_comp, N_E, 3). It is linear
    in the composition, so the default pair (pure nu_e, pure nu_mu sources) gives every (x, 1 - x, 0) -- what
    the flavor analysis needs (analysis.model; the sky's pixel means, as propagate_sky's galactic_allsky_*).
    For the dark portal with emission '3d' the band responses are computed once for all compositions."""
    import dp_galactic
    import dp_sky
    sky_p = params.get('sky') or dict(nside=16, dec_band_edges=None)
    gal = params['galactic']
    grid = dp_sky.SkyGrid(sky_p['nside'], sky_p['dec_band_edges'])
    if np.any(np.bincount(grid.band, minlength=grid.n_bands) == 0):
        raise ValueError(f"The Galactic component needs a pixel centre in every declination band (N_side {grid.nside}).")
    setup = _flux_setup(E_bins, params)
    E_GeV = np.asarray(E_bins, dtype=np.float64) / 1e9
    em = dp_galactic.emitted(gal['template'], E_GeV, grid.nside, gal['norm_gal'])
    share = np.bincount(grid.band, minlength=grid.n_bands) / grid.npix
    em_b = dp_galactic.band_means(em, grid.band, grid.n_bands)
    comps = [np.asarray(c, dtype=np.float64) for c in compositions]
    out = np.empty((len(comps), len(E_bins), 3))
    if setup.get('si', False):
        P = dp_galactic.averaged_transfer(setup['U_pmns_base'])
        for i, c in enumerate(comps):
            out[i] = 2.0 * em.mean(axis=0)[:, None] * (P @ c)[None, :]
        return out
    if not params.get('si') and params.get('earth_sky', 'up') != 'full':
        raise ValueError("The Galactic component needs grid.earth_sky = 'full' (bands cover the whole sky).")
    if gal['emission'] == 'local':
        for i, c in enumerate(comps):
            T_nu, T_nb = _galactic_transfer(E_bins, params, setup, grid.costh_edges, grid.nside, f_S=c)
            T = (T_nu + T_nb)[::-1]
            out[i] = np.tensordot(share, em_b[:, :, None] * T, axes=(0, 0))
        return out
    R_nu, R_nb = _galactic_responses(E_bins, params, setup, grid.costh_edges)
    R_nu, R_nb = R_nu[::-1], R_nb[::-1]
    for i, c in enumerate(comps):
        F_nu, F_nb = _galactic_emission_populations(E_bins, params, setup, grid.nside, f_S=c)
        b = np.einsum('bek,bekf->bef', dp_galactic.band_means(em[:, :, None] * F_nu, grid.band, grid.n_bands), R_nu)
        b += np.einsum('bek,bekf->bef', dp_galactic.band_means(em[:, :, None] * F_nb, grid.band, grid.n_bands), R_nb)
        out[i] = np.tensordot(share, b, axes=(0, 0))
    return out


def propagation_worker(args) -> Tuple[np.ndarray, np.ndarray]:
    """Unpacks arguments and executes the physics simulation: one coupled evolution of the
    configured neutrino and antineutrino sources, summed at the detector."""
    E_bins, params = args
    setup = _flux_setup(E_bins, params)
    F_nu, F_nb = _flux_from_setup(E_bins, params, setup, injection_weights(params))
    return E_bins, F_nu + F_nb


def build_solver_params(config: Dict) -> Tuple[np.ndarray, Dict]:
    """Validates a (possibly partial) configuration and converts it into the energy grid
    and the parameter dict the flux computation reads -- run_solver's parsing, without its
    console output, for callers that evaluate many parameter points (the analysis package
    overrides the varied parameters in a solver config and calls this per evaluation, so
    the solver's defaults and checks stay the single source of truth).

    Arguments: config -- as for run_solver.
    Returns: (E_bins, params).
    """
    config = merge_config_with_defaults(config)
    params = {}
    params['H0_eV'] = float((config['cosmology']['H0_km_s_Mpc'] * km_per_s) / Mpc_to_eV_inv)
    params['Omega_m'] = float(config['cosmology']['Omega_m'])
    params['Omega_L'] = float(config['cosmology']['Omega_L'])
    params['T_nu0_eV'] = float(config['cosmology']['T_nu0_eV'])
    params['n_n0_eV3'] = float(config['cosmology']['n_n0_cm3'] * cm_inv3_to_eV3)
    # The potential's matter charge (dp_matter): None for the neutrons' (the default), the historical path
    params['matter_charge'] = dp_matter.from_config(config, params['n_n0_eV3'])
    params['mass_ordering'] = config['cosmology']['mass_ordering']
    mo = str(config['cosmology']['mass_ordering']).strip().lower()
    if mo not in ['normal', 'inverted']:
        raise ValueError(f"mass_ordering must be strictly 'normal' or 'inverted'. Received: {mo}")
    
    params['eta_bg'] = np.array([
        float(config['cosmology']['eta_e']),
        float(config['cosmology']['eta_mu']),
        float(config['cosmology']['eta_tau'])
    ], dtype=np.float64)
    if config['cosmology']['eta_e'] < 0 or config['cosmology']['eta_mu'] < 0 or config['cosmology']['eta_tau'] < 0:
        raise ValueError("CnuB flavor asymmetries (eta) must be >= 0 to avoid complex Debye screening masses.")

    params['dm21_sq'] = float(config['standard_mixing']['dm21_sq'])
    params['dm31_sq'] = float(config['standard_mixing']['dm31_sq'])
    params['th12'] = float(np.radians(config['standard_mixing']['th12_deg']))
    params['th13'] = float(np.radians(config['standard_mixing']['th13_deg']))
    params['th23'] = float(np.radians(config['standard_mixing']['th23_deg']))
    params['d_cp'] = float(np.radians(config['standard_mixing']['d_cp_deg']))
    params['use_SM_potential'] = config['standard_mixing']['use_SM_potential']
    params['n_e_earth_eV3'] = float(config['standard_mixing']['n_e_earth_cm3'] * cm_inv3_to_eV3)
    # Optional flavor-structure overrides for the coherent forward-scattering potentials
    # (neutrino convention; antineutrino conjugation is applied centrally in
    # propagate_spectrum). See parse_structure_matrix_3x3 for the accepted JSON format.
    params['V_SM_struct'] = None
    if config['standard_mixing']['V_SM_struct'] is not None:
        params['V_SM_struct'] = parse_structure_matrix_3x3(
            config['standard_mixing']['V_SM_struct'], "standard_mixing.V_SM_struct")
    params['U_LRI_struct'] = None
    if config['dark_portal']['U_LRI_struct'] is not None:
        params['U_LRI_struct'] = parse_structure_matrix_3x3(
            config['dark_portal']['U_LRI_struct'], "dark_portal.U_LRI_struct")
    params['detector_depth_km'] = float(config['standard_mixing']['detector_depth_km'])
    depth_km = float(config['standard_mixing']['detector_depth_km'])
    if depth_km < 0.0 or depth_km > 6371.0:
        raise ValueError(f"detector_depth_km ({depth_km}) must be between 0 and the Earth's radius (6371.0 km).")

    params['m_nu_bg_arr'] = get_mass_eigenvalues(
        float(config['cosmology']['m_lightest_eV']),
        float(params['dm21_sq']),
        float(params['dm31_sq']),
        params['mass_ordering']
    )
    
    params['u_e'] = float(config['dark_portal']['u_e'])
    params['u_mu'] = float(config['dark_portal']['u_mu'])
    params['u_tau'] = float(config['dark_portal']['u_tau'])
    u_sq_sum = np.abs(params['u_e'])**2 + np.abs(params['u_mu'])**2 + np.abs(params['u_tau'])**2
    if u_sq_sum > 1.0:
        raise ValueError(f"Sterile mixing elements violate unitarity. Sum of squares is {u_sq_sum:.4f} (must be <= 1.0).")
    params['m_Zd_eV'] = float(config['dark_portal']['m_Zd_eV'])
    params['g_D'] = float(config['dark_portal']['g_D'])
    params['epsilon_prime'] = float(config['dark_portal']['epsilon_prime'])
    # Physical positivity bounds
    if params['m_Zd_eV'] <= 0:
        raise ValueError("Dark mediator mass m_Zd_eV must be strictly positive.")
    if params['g_D'] <= 0:
        raise ValueError("Dark mediator gauge coupling g_D must be strictly positive.")
    if params['g_D'] >= np.sqrt(4 * np.pi):
        raise ValueError(f"Dark gauge coupling g_D ({params['g_D']:.2f}) exceeds the perturbativity bound (sqrt(4*pi)).")
    if not np.isfinite(params['epsilon_prime']):
        raise ValueError("epsilon_prime must be finite (0 switches off the coupling to matter: no coherent potential; "
                         "its sign, with g_D's, is the potential's: both are physical, decided 2026-10-14).")
    params['use_debye_screening'] = config['dark_portal']['use_debye_screening']
    params['cosmo_transport'] = str(config['dark_portal']['cosmo_transport']).strip().lower()
    params['absorption_model'] = str(config['dark_portal']['absorption_model']).strip().lower()
    # the MeV transport's relics thermal in every collision term (dp_si thermal="full", 2026-10-18), not only in the
    # resonance: off by default
    params['thermal_products'] = bool(config['dark_portal'].get('thermal_products', False))
    if params['absorption_model'] not in ('lagrangian', 'theory_notes'):
        raise ValueError(f"dark_portal.absorption_model must be 'lagrangian' or 'theory_notes', got "
                         f"{config['dark_portal']['absorption_model']!r}")
    # The benchmark model (dp_models; docs/dev/SCOPE_WIDENING_PLAN.md): the dark portal by default; a gauge model
    # sets the matter charge, the texture, the neutrino coupling, the potential's coupling product and the visible
    # width from its charges, dark_portal.g_D playing its gauge coupling g' (and epsilon_prime the Z-Z' mixing of
    # L_mu - L_tau).
    params['model'] = str(config['model']['name']).strip()
    if params['model'].lower() != 'dark_portal':
        import dp_models
        kind, charges, _, _ = dp_models.spec(params['model'])
        mc = config['matter_charge']
        if (float(mc['q_n']), float(mc['q_p']), float(mc['q_e'])) != (1.0, 0.0, 0.0):
            raise ValueError(f"model {params['model']!r} sets the matter charge: leave matter_charge at its default")
        if params['U_LRI_struct'] is not None:
            raise ValueError(f"model {params['model']!r} sets the texture: leave dark_portal.U_LRI_struct unset")
        if params['absorption_model'] != 'lagrangian':
            raise ValueError("the gauge models need dark_portal.absorption_model = 'lagrangian'")
        n_e0 = float(config['cosmology']['n_e0_cm3']) * cm_inv3_to_eV3
        params['matter_charge'] = dp_matter.normalize(*charges, params['n_n0_eV3'], n_e0)
        params['U_LRI_struct'] = dp_models.texture(params['model'])
        g, eps = params['g_D'], params['epsilon_prime']
        params['G_coupling'] = dp_models.neutrino_coupling(params['model'], g, eps)
        params['Gamma_vis'] = dp_models.visible_width(params['model'], params['m_Zd_eV'], g, eps)
        params['potential_product'] = dp_models.potential_product(params['model'], g, eps)
    
    params['p_index'] = float(config['source']['p_index'])
    if config['source']['p_index'] <= 0:
        raise ValueError("p_index should be positive. The solver assumes a spectrum proportional to E^(-p_index).")
    params['Phi_0'] = float(config['source']['Phi_0'])
    if config['source']['Phi_0'] <= 0:
        raise ValueError("Flux normalization Phi_0 should be strictly positive.")
    params['z_max'] = float(config['source']['z_max'])
    if params['z_max'] <= 0.0:
        raise ValueError("Maximum redshift z_max must be strictly positive.")
    params['rho_evol'] = float(config['source']['rho_evol'])
    params['f_e_S'] = float(config['source']['f_e_S'])
    params['f_mu_S'] = float(config['source']['f_mu_S'])
    params['f_tau_S'] = float(config['source']['f_tau_S'])
    flavor_sum = params['f_e_S'] + params['f_mu_S'] + params['f_tau_S']
    if not np.isclose(flavor_sum, 1.0, atol=1e-5):
        raise ValueError(f"Source flavor ratios must sum to 1.0. Current sum: {flavor_sum}")
    if params['f_e_S'] < 0 or params['f_mu_S'] < 0 or params['f_tau_S'] < 0:
        raise ValueError("Source flavor fractions must be strictly >= 0. Negative flux injection is unphysical.")
    params['is_antineutrino'] = config['source']['is_antineutrino']
    params['nubar_to_nu_ratio'] = config['source']['nubar_to_nu_ratio']
    # Sources inside a Milky-Way-like host galaxy (scale of its neutrons; 0 = the cosmic mean,
    # the default) -- see propagate_spectrum.
    params['host_galaxy_scale'] = float(config['source']['host_galaxy_scale'])
    if params['host_galaxy_scale'] < 0.0:
        raise ValueError("source.host_galaxy_scale must be >= 0 (0: sources in the cosmic mean density).")
    # The sources' emission window in today's energy (2026-10-15; propagate_spectrum's inj_window), eV
    lo = config['source'].get('injection_E_min_GeV')
    hi = config['source'].get('injection_E_max_GeV')
    lo = 0.0 if lo is None else float(lo) * 1e9
    hi = np.inf if hi is None else float(hi) * 1e9
    if not (0.0 <= lo < hi):
        raise ValueError(f"source.injection_E_min_GeV / injection_E_max_GeV must satisfy 0 <= min < max, got "
                         f"{config['source'].get('injection_E_min_GeV')}, {config['source'].get('injection_E_max_GeV')}")
    params['inj_window_eV'] = (lo, hi)
    
    # Secret neutrino interactions (dp_si, 2026-10-13): when enabled they replace the dark portal
    # (see DEFAULT_CONFIG's comments).
    si_cfg = config['secret_interactions']
    params['si_enabled'] = bool(si_cfg['enabled'])
    if params['si_enabled']:
        mediator = str(si_cfg['mediator']).strip().lower()
        if mediator not in ('scalar', 'vector'):
            raise ValueError(f"secret_interactions.mediator must be 'scalar' or 'vector', got {si_cfg['mediator']!r}")
        nu_type = str(si_cfg['neutrino_type']).strip().lower()
        if nu_type not in ('majorana', 'dirac'):
            raise ValueError(f"secret_interactions.neutrino_type must be 'majorana' or 'dirac', got {si_cfg['neutrino_type']!r}")
        sterile = str(si_cfg['dirac_sterile']).strip().lower()
        if sterile not in ('track', 'drop'):
            raise ValueError(f"secret_interactions.dirac_sterile must be 'track' or 'drop', got {si_cfg['dirac_sterile']!r}")
        m_phi, g_phi = float(si_cfg['m_phi_eV']), float(si_cfg['g_phi'])
        if not (m_phi > 0.0 and np.isfinite(m_phi)):
            raise ValueError("secret_interactions.m_phi_eV must be strictly positive.")
        if not (0.0 <= g_phi < np.sqrt(4.0 * np.pi)):
            raise ValueError(f"secret_interactions.g_phi ({g_phi}) must be in [0, sqrt(4 pi)) (perturbativity).")
        channels = str(si_cfg['channels']).strip().lower()
        if channels not in ('full', 's', 'delta'):
            raise ValueError(f"secret_interactions.channels must be 'full', 's' or 'delta', got {si_cfg['channels']!r}")
        phiphi = str(si_cfg['phiphi_coupling']).strip().lower()
        if phiphi not in ('coherent', 'paper'):
            raise ValueError(f"secret_interactions.phiphi_coupling must be 'coherent' or 'paper', got "
                             f"{si_cfg['phiphi_coupling']!r}")
        z_sub = si_cfg['z_substeps']
        if not isinstance(z_sub, int) or z_sub < 1:
            raise ValueError("secret_interactions.z_substeps must be an integer >= 1.")
        if not float(si_cfg['E_extension']) >= 1.0:
            raise ValueError("secret_interactions.E_extension must be >= 1.")
        import dp_si
        g_flavor = dp_si.flavor_couplings(si_cfg['texture'], g_phi, si_cfg['g_matrix'])   # validates the texture
        params['si'] = dict(mediator=mediator, majorana=(nu_type == 'majorana'), dirac_sterile=sterile, m_phi_eV=m_phi,
                            g_flavor=g_flavor, channels=channels, phiphi=phiphi, z_substeps=int(z_sub),
                            E_extension=float(si_cfg['E_extension']))

    # Per-direction output and the Galactic component (docs/dev/DIRECTIONS_PLAN.md, 2026-10-13).
    sky_cfg = config['sky']
    sky_out = str(sky_cfg['output']).strip().lower()
    if sky_out not in ('allsky', 'bands', 'healpix'):
        raise ValueError(f"sky.output must be 'allsky', 'bands' or 'healpix', got {sky_cfg['output']!r}")
    nside = int(sky_cfg['nside'])
    if nside < 1 or (nside & (nside - 1)) != 0:
        raise ValueError(f"sky.nside must be a power of 2, got {sky_cfg['nside']!r}")
    edges = sky_cfg['dec_band_edges_deg']
    if edges is not None:
        edges = np.radians(np.asarray(edges, dtype=np.float64))
        if (edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0)
                or abs(edges[0] + np.pi / 2) > 1e-9 or abs(edges[-1] - np.pi / 2) > 1e-9):
            raise ValueError("sky.dec_band_edges_deg must increase from -90 to 90.")
        edges[0], edges[-1] = -np.pi / 2, np.pi / 2
    nodes_per_band = int(sky_cfg['earth_nodes_per_band'])
    if nodes_per_band < 1:
        raise ValueError("sky.earth_nodes_per_band must be >= 1.")
    sun_w = str(sky_cfg['sun_weighting']).strip().lower()
    if sun_w != 'isotropic':
        raise ValueError("sky.sun_weighting: only 'isotropic' so far (the yearly weighting follows the anisotropy "
                         "diagnostic, decision D4 of docs/dev/DIRECTIONS_PLAN.md).")
    eg_norm = sky_cfg['extragalactic_norm']
    if eg_norm is not None:
        eg_norm = float(eg_norm)
        if not eg_norm > 0.0:
            raise ValueError(f"sky.extragalactic_norm must be None or > 0, got {sky_cfg['extragalactic_norm']!r}")
    eg_E = float(sky_cfg['extragalactic_norm_E_GeV'])
    if not eg_E > 0.0:
        raise ValueError(f"sky.extragalactic_norm_E_GeV must be > 0, got {sky_cfg['extragalactic_norm_E_GeV']!r}")
    params['sky'] = dict(output=sky_out, nside=nside, dec_band_edges=edges, earth_nodes_per_band=nodes_per_band,
                         sun_weighting=sun_w, extragalactic_norm=eg_norm, extragalactic_norm_E_GeV=eg_E)
    gal_cfg = config['galactic']
    import dp_galactic
    dp_galactic.use_data_dir(gal_cfg['data_dir'])
    template = str(gal_cfg['template']).strip().lower()
    if template not in dp_galactic.TEMPLATES:
        raise ValueError(f"galactic.template must be one of {sorted(dp_galactic.TEMPLATES)}, got {gal_cfg['template']!r}")
    emission = str(gal_cfg['emission']).strip().lower()
    if emission not in ('3d', 'local'):
        raise ValueError(f"galactic.emission must be '3d' or 'local', got {gal_cfg['emission']!r}")
    # None: IceCube's 2026 fit for the template (dp_galactic.NORM_DEFAULT)
    norm_gal = float(dp_galactic.NORM_DEFAULT[template] if gal_cfg['norm_gal'] is None else gal_cfg['norm_gal'])
    x_gal = float(gal_cfg['f_e_S_gal'])
    if norm_gal < 0.0 or not (0.0 <= x_gal <= 1.0):
        raise ValueError("galactic.norm_gal must be >= 0 and galactic.f_e_S_gal in [0, 1].")
    params['galactic'] = dict(enabled=bool(gal_cfg['enabled']), template=template, norm_gal=norm_gal,
                              f_e_S_gal=x_gal, emission=emission)

    params['n_z_bins'] = int(config['grid']['n_z_bins'])
    # The transit to the Earth (2026-10-13, replacing the Milky Way trajectories): directions
    # to the Sun in its coherent transit, and the frame table's spacing (dp_transit).
    params['n_sun_paths'] = int(config['grid']['n_sun_paths'])
    params['transit_dtheta'] = float(config['grid']['transit_dtheta'])
    if params['n_sun_paths'] < 1:
        raise ValueError("grid.n_sun_paths must be an integer >= 1.")
    if not (0.0 < params['transit_dtheta'] <= 0.1):
        raise ValueError("grid.transit_dtheta must be in (0, 0.1].")
    # Defaults set by the production convergence study (tools/convergence_study.py,
    # report in output/convergence/): n_earth_chords=3 (the old default) carried
    # 1.6-4.9e-2 absolute flavor-ratio error against a 100-chord reference at every
    # tested dark-portal point -- by far the largest numerical systematic of any default
    # -- while 12 chords bring it to ~1e-2 (percent level).
    params['n_earth_chords'] = int(config['grid']['n_earth_chords'])
    # Coarse z-sampling of the thermal-rate grid (0 disables interpolation -- exact grid).
    params['n_z_thermal'] = int(config['grid']['n_z_thermal'])
    # Semi-analytic momentum integration of the thermal rates (True, the default) vs
    # the 100-point midpoint reference kernel (False) -- see build_thermal_grid.
    params['thermal_analytic'] = bool(config['grid']['thermal_analytic'])
    # Earth-transit potential discretization: 'left' (historical) or 'avg' (exact
    # step-averaged potential -- see _earth_chord_worker's docstring).
    params['earth_step_scheme'] = str(config['grid']['earth_step_scheme']).strip().lower()
    if params['earth_step_scheme'] not in ('left', 'avg'):
        raise ValueError(f"grid.earth_step_scheme must be 'left' or 'avg', got {params['earth_step_scheme']!r}")
    # Average over the Earth chords: 'phase' (exact integration of each block's LRI phase
    # over cos(zenith)) or 'plain' (historical equal-weight mean) -- see
    # dp_spatial.average_earth_chords. 'phase' needs the phase-exact 'avg' steps.
    params['earth_sky'] = str(config['grid']['earth_sky']).strip().lower()
    if params['earth_sky'] not in ('full', 'up'):
        raise ValueError(f"grid.earth_sky must be 'full' or 'up', got {params['earth_sky']!r}")
    params['earth_zenith_average'] = str(config['grid']['earth_zenith_average']).strip().lower()
    if params['earth_zenith_average'] not in ('phase', 'plain'):
        raise ValueError(f"grid.earth_zenith_average must be 'phase' or 'plain', got {params['earth_zenith_average']!r}")
    if params['earth_zenith_average'] == 'phase' and params['earth_step_scheme'] != 'avg':
        # 'left' exists to reproduce historical results, which used the plain mean (and its
        # Riemann phase errors defeat the 'phase' rule), so it implies 'plain'.
        params['earth_zenith_average'] = 'plain'
        if int(config['execution'].get('verbose', 1)) >= 1:
            safe_print("[NOTE] earth_step_scheme = 'left' uses the historical 'plain' zenith average.")
    params['N_steps_earth'] = int(config['grid']['N_steps_earth'])
    chords_earth = config['grid']['n_earth_chords']
    if not isinstance(chords_earth, int) or chords_earth < 1:
        raise ValueError("n_earth_chords must be an integer >= 1.")
    
    if config['grid']['n_E_bins'] <= 1 or config['grid']['n_z_bins'] <= 1:
        raise ValueError("Grid resolution (n_E_bins, n_z_bins) must be greater than 1 to define integration steps.")

    # Energy grid sanity check
    if config['grid']['E_min_eV'] >= config['grid']['E_max_eV']:
        raise ValueError("E_max_eV must be strictly greater than E_min_eV for the cascading integral to function properly.")
    E_bins = np.logspace(np.log10(config['grid']['E_min_eV']), 
                         np.log10(config['grid']['E_max_eV']), 
                         int(config['grid']['n_E_bins']))
    
    # A single call to run_solver always evaluates exactly one physics configuration, so
    # there is nothing to parallelize at the config level (unlike dark_portal_scanner.py,
    # which parallelizes across many configs). Instead, n_cores drives genuine intra-run
    # parallelism across the independent Milky Way paths / Earth chords in the spatial
    # handover (see propagate_earth_spatial), which is where a single run actually has
    # independent work to distribute.
    n_cores = max(1, int(config['execution']['n_cores']))
    params['n_cores'] = n_cores
    # Optional CUDA spatial handover (requires cupy + a CUDA device; silently falls back
    # to the CPU path with a warning when unavailable -- see gpu_spatial.py).
    params['use_gpu'] = bool(config['execution']['use_gpu'])
    params['earth_transfer'] = str(config['execution'].get('earth_transfer', 'off'))
    params['show_progress'] = True
    params['verbose'] = int(config['execution']['verbose'])
    return E_bins, params


def run_solver(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Orchestrates configuration and parallelization.

    Arguments:
      config: (possibly partial) configuration dict with the DEFAULT_CONFIG section
        layout -- cosmology, standard_mixing, dark_portal, source, grid, output,
        execution (see dp_config.DEFAULT_CONFIG for every key and its default,
        including grid.N_steps_earth = 25 under the certified 'avg' Earth step
        scheme). It is first overlaid onto DEFAULT_CONFIG with unknown-key
        rejection (see merge_config_with_defaults), so every read below can index
        directly -- no per-call fallback literals to drift out of sync.

    Returns:
      (E_bins, Flux_array): the energy grid (eV) and the (N_E, 3) flavor fluxes at
      the detector, weighted across neutrinos/antineutrinos per source.is_antineutrino.
    """
    E_bins, params = build_solver_params(config)
    n_cores = params['n_cores']
    safe_print(f"Initializing solver across {len(E_bins)} coupled energy bins...")
    if n_cores > 1:
        safe_print(f"Using up to {n_cores} core(s) to parallelize the Milky Way / Earth spatial handover.")
    else:
        safe_print("Deploying single-core worker.")

    E_out, Flux_out = propagation_worker((E_bins, params))

    return E_out, Flux_out


def plot_flavor_fluxes(E_array: np.ndarray, Flux_array: np.ndarray, p_index: float, is_antineutrino_flag: str, output_filename: str = "../output/plots/flux_plot.png"):
    """Plots and maps the total astrophysical neutrino spectrum ratios at Earth layers."""
    E_GeV = E_array / 1e9
    scaling_factor = E_GeV**2.0
    
    flux_e = Flux_array[:, 0]
    flux_mu = Flux_array[:, 1]
    flux_tau = Flux_array[:, 2]
    flux_total = np.sum(Flux_array, axis=1)
    
    s_flux_e = flux_e * scaling_factor
    s_flux_mu = flux_mu * scaling_factor
    s_flux_tau = flux_tau * scaling_factor
    s_flux_total = flux_total * scaling_factor
    
    safe_total = np.maximum(flux_total, 1e-100)
    ratio_e = flux_e / safe_total
    ratio_mu = flux_mu / safe_total
    ratio_tau = flux_tau / safe_total
    
    # Evaluate flag for dynamic LaTeX labels
    flag_val = str(is_antineutrino_flag).strip().lower()
    if flag_val in ['sum', 'both']:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\nu + \bar{\nu}$)', r'$\nu_e + \bar{\nu}_e$', r'$\nu_\mu + \bar{\nu}_\mu$', r'$\nu_\tau + \bar{\nu}_\tau$'
    elif flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\bar{\nu}$)', r'$\bar{\nu}_e$', r'$\bar{\nu}_\mu$', r'$\bar{\nu}_\tau$'
    else:
        lbl_tot, lbl_e, lbl_mu, lbl_tau = r'Total Flux ($\nu$)', r'$\nu_e$', r'$\nu_\mu$', r'$\nu_\tau$'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    ax1.plot(E_GeV, s_flux_total, color='black', lw=2.5, label=lbl_tot)
    ax1.plot(E_GeV, s_flux_e, color='crimson', linestyle='--', lw=2.0, label=lbl_e)
    ax1.plot(E_GeV, s_flux_mu, color='royalblue', linestyle='-.', lw=2.0, label=lbl_mu)
    ax1.plot(E_GeV, s_flux_tau, color='forestgreen', linestyle=':', lw=2.0, label=lbl_tau)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Scaled Flux $E_0^2 \Phi$ [Arb. Units]', fontsize=14)
    ax1.set_title('Astrophysical Neutrino Diffuse Flux at Earth', fontsize=15, pad=15)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(fontsize=12, loc='lower right')
    
    ax2.plot(E_GeV, ratio_e, color='crimson', linestyle='--', lw=2.0)
    ax2.plot(E_GeV, ratio_mu, color='royalblue', linestyle='-.', lw=2.0)
    ax2.plot(E_GeV, ratio_tau, color='forestgreen', linestyle=':', lw=2.0)
    
    ax2.set_ylabel('Flavor Ratio', fontsize=14)
    ax2.set_xlabel(r'Observed Energy $E_0$ [GeV]', fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    safe_print(f"\nPlot successfully saved to: {os.path.abspath(output_filename)}")
    plt.show()


if __name__ == "__main__":
    # Initialize argument flags for terminal overrides mapping to the JSON variables
    parser = argparse.ArgumentParser(description="Dark Portal Neutrino Propagation Solver command-line flag orchestrator.")
    parser.add_argument("--config", type=str, default="../config/config_dark_portal.json", help="Path to JSON parameter file.")
    
    # Cosmology Overrides
    parser.add_argument("--H0_km_s_Mpc", type=float, default=None)
    parser.add_argument("--Omega_m", type=float, default=None)
    parser.add_argument("--Omega_L", type=float, default=None)
    parser.add_argument("--T_nu0_eV", type=float, default=None)
    parser.add_argument("--n_n0_cm3", type=float, default=None)
    parser.add_argument("--m_lightest_eV", type=float, default=None)
    parser.add_argument("--mass_ordering", type=str, default=None)
    parser.add_argument("--eta_e", type=float, default=None)
    parser.add_argument("--eta_mu", type=float, default=None)
    parser.add_argument("--eta_tau", type=float, default=None)
    
    # Standard Oscillation Mixing Overrides
    parser.add_argument("--dm21_sq", type=float, default=None)
    parser.add_argument("--dm31_sq", type=float, default=None)
    parser.add_argument("--th12_deg", type=float, default=None)
    parser.add_argument("--th13_deg", type=float, default=None)
    parser.add_argument("--th23_deg", type=float, default=None)
    parser.add_argument("--d_cp_deg", type=float, default=None)
    parser.add_argument("--use_SM_potential", type=str, default=None, help="Set to 'true' or 'false'")
    parser.add_argument("--n_e_earth_cm3", type=float, default=None)
    parser.add_argument("--detector_depth_km", type=float, default=None)
    
    # Dark Portal Vector Interactions Overrides
    parser.add_argument("--u_e", type=float, default=None)
    parser.add_argument("--u_mu", type=float, default=None)
    parser.add_argument("--u_tau", type=float, default=None)
    parser.add_argument("--m_Zd_eV", type=float, default=None)
    parser.add_argument("--g_D", type=float, default=None)
    parser.add_argument("--epsilon_prime", type=float, default=None)
    parser.add_argument("--use_debye_screening", type=str, default=None, help="Set to 'true' or 'false'")
    
    # Primary Flux Injectors Overrides
    parser.add_argument("--p_index", type=float, default=None)
    parser.add_argument("--Phi_0", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--rho_evol", type=float, default=None)
    parser.add_argument("--f_e_S", type=float, default=None)
    parser.add_argument("--f_mu_S", type=float, default=None)
    parser.add_argument("--f_tau_S", type=float, default=None)
    parser.add_argument("--is_antineutrino", type=str, default=None)
    parser.add_argument("--nubar_to_nu_ratio", type=float, default=None)
    
    # Grid Configurations Overrides
    parser.add_argument("--E_min_eV", type=float, default=None)
    parser.add_argument("--E_max_eV", type=float, default=None)
    parser.add_argument("--n_E_bins", type=int, default=None)
    parser.add_argument("--n_z_bins", type=int, default=None)
    parser.add_argument("--n_z_thermal", type=int, default=None, help="Coarse z-samples for the thermal-rate grid (default 0 = exact, no interpolation; see build_thermal_grid before enabling)")
    parser.add_argument("--thermal_analytic", type=str, default=None, help="Set to 'false' to use the 100-point midpoint reference kernel for the thermal rates instead of the semi-analytic default (see build_thermal_grid)")
    parser.add_argument("--earth_step_scheme", type=str, default=None, help="Earth-transit potential discretization: 'avg' (default: exact step-averaged potential, step-count-independent LRI phase -- see dp_spatial._earth_chord_worker) or 'left' (historical left-endpoint sampling, kept for reproducing pre-2026-07 results)")
    parser.add_argument("--earth_sky", type=str, default=None, help="Directions averaged at the detector: 'full' (default: the whole sky, equal weight per cos(zenith); down-going neutrinos cross only the ice above the detector) or 'up' (up-going only, through the Earth)")
    parser.add_argument("--earth_zenith_average", type=str, default=None, help="Average over the Earth chords: 'phase' (default: integrate each block of the chords' density matrices against its dark-portal phase over cos(zenith), removing the chord-sampling floor) or 'plain' (historical equal-weight mean; required with --earth_step_scheme left)")
    parser.add_argument("--n_sun_paths", type=int, default=None, help="Directions to the Sun averaged in its coherent transit (Gauss-Legendre in the cosine of the angle; default 8)")
    parser.add_argument("--n_earth_chords", type=int, default=None, help="Number of Earth chords for PREM evaluation")
    parser.add_argument("--transit_dtheta", type=float, default=None, help="Largest turn of the eigenbasis between the transit's frame-table nodes (default 0.002; the transit's error scales with it)")
    parser.add_argument("--N_steps_earth", type=int, default=None, help="Number of spatial integration steps per Earth chord (default 25 under the 'avg' step scheme, whose results are certified step-count-independent; under the legacy 'left' scheme results scatter ~1e-2 with any step-count change and 500 was the pinned value)")
    
    # Export Slated Locations Overrides
    parser.add_argument("--flux_filename", type=str, default=None)
    parser.add_argument("--plot_filename", type=str, default=None)
    
    # Execution Architecture Overrides
    parser.add_argument("--n_cores", type=int, default=None)
    parser.add_argument("--use_gpu", type=str, default=None, help="Set to 'true' to run the spatial handover on a CUDA GPU via cupy (falls back to CPU if unavailable)")
    parser.add_argument("--verbose", type=int, default=None)
    
    args = parser.parse_args()

    # Baseline defaults come from the single canonical DEFAULT_CONFIG (no more
    # separately-maintained copy here).
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Precedence Step 2: Overlay values from JSON file if one exists
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            for section, entries in file_config.items():
                if section in config and isinstance(entries, dict):
                    config[section].update(entries)
                else:
                    # Unknown section (or malformed entry): pass through untouched so
                    # run_solver's merge_config_with_defaults rejects it with a clear
                    # error, rather than silently dropping it here.
                    config[section] = entries
    else:
        print(f"Config file {args.config} not found. Running under strict system defaults.")

    def str_to_bool(val):
        if val is None: return None
        return str(val).lower().strip() in ['true', '1', 'yes', 'y']

    # Precedence Step 3: Overwrite entries dynamically via shell CLI arguments
    if args.H0_km_s_Mpc is not None: config["cosmology"]["H0_km_s_Mpc"] = args.H0_km_s_Mpc
    if args.Omega_m is not None: config["cosmology"]["Omega_m"] = args.Omega_m
    if args.Omega_L is not None: config["cosmology"]["Omega_L"] = args.Omega_L
    if args.T_nu0_eV is not None: config["cosmology"]["T_nu0_eV"] = args.T_nu0_eV
    if args.n_n0_cm3 is not None: config["cosmology"]["n_n0_cm3"] = args.n_n0_cm3
    if args.m_lightest_eV is not None: config["cosmology"]["m_lightest_eV"] = args.m_lightest_eV
    if args.mass_ordering is not None: config["cosmology"]["mass_ordering"] = args.mass_ordering
    if args.eta_e is not None: config["cosmology"]["eta_e"] = args.eta_e
    if args.eta_mu is not None: config["cosmology"]["eta_mu"] = args.eta_mu
    if args.eta_tau is not None: config["cosmology"]["eta_tau"] = args.eta_tau

    if args.dm21_sq is not None: config["standard_mixing"]["dm21_sq"] = args.dm21_sq
    if args.dm31_sq is not None: config["standard_mixing"]["dm31_sq"] = args.dm31_sq
    if args.th12_deg is not None: config["standard_mixing"]["th12_deg"] = args.th12_deg
    if args.th13_deg is not None: config["standard_mixing"]["th13_deg"] = args.th13_deg
    if args.th23_deg is not None: config["standard_mixing"]["th23_deg"] = args.th23_deg
    if args.d_cp_deg is not None: config["standard_mixing"]["d_cp_deg"] = args.d_cp_deg
    if args.use_SM_potential is not None: config["standard_mixing"]["use_SM_potential"] = str_to_bool(args.use_SM_potential)
    if args.n_e_earth_cm3 is not None: config["standard_mixing"]["n_e_earth_cm3"] = args.n_e_earth_cm3
    if args.detector_depth_km is not None: config["standard_mixing"]["detector_depth_km"] = args.detector_depth_km

    if args.u_e is not None: config["dark_portal"]["u_e"] = args.u_e
    if args.u_mu is not None: config["dark_portal"]["u_mu"] = args.u_mu
    if args.u_tau is not None: config["dark_portal"]["u_tau"] = args.u_tau
    if args.m_Zd_eV is not None: config["dark_portal"]["m_Zd_eV"] = args.m_Zd_eV
    if args.g_D is not None: config["dark_portal"]["g_D"] = args.g_D
    if args.epsilon_prime is not None: config["dark_portal"]["epsilon_prime"] = args.epsilon_prime
    if args.use_debye_screening is not None: config["dark_portal"]["use_debye_screening"] = str_to_bool(args.use_debye_screening)

    if args.p_index is not None: config["source"]["p_index"] = args.p_index
    if args.Phi_0 is not None: config["source"]["Phi_0"] = args.Phi_0
    if args.z_max is not None: config["source"]["z_max"] = args.z_max
    if args.rho_evol is not None: config["source"]["rho_evol"] = args.rho_evol
    if args.f_e_S is not None: config["source"]["f_e_S"] = args.f_e_S
    if args.f_mu_S is not None: config["source"]["f_mu_S"] = args.f_mu_S
    if args.f_tau_S is not None: config["source"]["f_tau_S"] = args.f_tau_S
    if args.is_antineutrino is not None: config["source"]["is_antineutrino"] = args.is_antineutrino
    if args.nubar_to_nu_ratio is not None: config["source"]["nubar_to_nu_ratio"] = args.nubar_to_nu_ratio

    if args.E_min_eV is not None: config["grid"]["E_min_eV"] = args.E_min_eV
    if args.E_max_eV is not None: config["grid"]["E_max_eV"] = args.E_max_eV
    if args.n_E_bins is not None: config["grid"]["n_E_bins"] = args.n_E_bins
    if args.n_z_bins is not None: config["grid"]["n_z_bins"] = args.n_z_bins
    if args.n_z_thermal is not None: config["grid"]["n_z_thermal"] = args.n_z_thermal
    if args.thermal_analytic is not None: config["grid"]["thermal_analytic"] = str_to_bool(args.thermal_analytic)
    if args.earth_step_scheme is not None: config["grid"]["earth_step_scheme"] = args.earth_step_scheme
    if args.earth_zenith_average is not None: config["grid"]["earth_zenith_average"] = args.earth_zenith_average
    if args.earth_sky is not None: config["grid"]["earth_sky"] = args.earth_sky
    if args.n_sun_paths is not None: config["grid"]["n_sun_paths"] = args.n_sun_paths
    if args.n_earth_chords is not None: config["grid"]["n_earth_chords"] = args.n_earth_chords
    if args.transit_dtheta is not None: config["grid"]["transit_dtheta"] = args.transit_dtheta
    if args.N_steps_earth is not None: config["grid"]["N_steps_earth"] = args.N_steps_earth

    if args.flux_filename is not None: config["output"]["flux_filename"] = args.flux_filename
    if args.plot_filename is not None: config["output"]["plot_filename"] = args.plot_filename

    if args.n_cores is not None: config["execution"]["n_cores"] = args.n_cores
    if args.use_gpu is not None: config["execution"]["use_gpu"] = str_to_bool(args.use_gpu)
    if args.verbose is not None: config["execution"]["verbose"] = args.verbose
            
    import time as _time
    _t0 = _time.perf_counter()
    E_array, Flux_array = run_solver(config)
    _runtime = _time.perf_counter() - _t0

    output_cfg = config.get("output", {})
    flux_filename = output_cfg.get("flux_filename", "../output/data/output_flux.txt")
    plot_filename = output_cfg.get("plot_filename", "../output/plots/flux_plot.png")

    os.makedirs(os.path.dirname(os.path.abspath(flux_filename)), exist_ok=True)
    save_flux_to_file(E_array, Flux_array, filename=flux_filename)
    write_run_manifest(flux_filename + ".manifest.json", config, runtime_s=_runtime)
    
    Flux_array = np.clip(Flux_array, 1e-50, None) 
    Total_Flux = np.sum(Flux_array, axis=1)
    f_e, f_mu, f_tau = Flux_array[:, 0] / Total_Flux, Flux_array[:, 1] / Total_Flux, Flux_array[:, 2] / Total_Flux
    
    safe_print("\nIntegration Complete. Sample Output (Mid-grid Energy):")
    mid_idx = len(E_array) // 2
    safe_print(f"Energy: {E_array[mid_idx] / 1e9:.2e} GeV")
    safe_print(f"Flavor Ratios at Earth [e, mu, tau]: [{f_e[mid_idx]:.3f}, {f_mu[mid_idx]:.3f}, {f_tau[mid_idx]:.3f}]")

    if HAS_MATPLOTLIB:
        os.makedirs(os.path.dirname(os.path.abspath(plot_filename)), exist_ok=True)
        plot_flavor_fluxes(E_array, Flux_array, config['source']['p_index'], config['source'].get('is_antineutrino', 'sum'), output_filename=plot_filename)