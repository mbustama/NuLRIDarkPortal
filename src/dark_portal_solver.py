import os
import json
import multiprocessing
import numpy as np
from typing import Dict, Tuple

# =============================================================================
# OPTIONAL DEPENDENCIES VERIFICATION
# =============================================================================
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: 'matplotlib' not found. Plotting will be disabled.")

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: 'numba' not found. Falling back to pure Python (will be significantly slower).")
    def njit(*args, **kwargs):
        """Dummy decorator if numba is unavailable."""
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: 'tqdm' not found. Progress bars will be disabled.")
    def tqdm(iterable, *args, **kwargs): return iterable

# =============================================================================
# PHYSICAL CONSTANTS (Natural Units: eV = 1)
# =============================================================================
eV = 1.0
keV = 1e3 * eV
MeV = 1e6 * eV
GeV = 1e9 * eV
TeV = 1e12 * eV
PeV = 1e15 * eV

# Length & volume conversions
m_to_eV_inv = 5.0677e6
cm_inv3_to_eV3 = 1.0 / ((m_to_eV_inv * 1e-2)**3)
Mpc_to_eV_inv = 3.08567758e22 * m_to_eV_inv

# Electroweak & Cosmological Constants
G_F = 1.1663787e-23 / (eV**2)
km_per_s = 1.0 / 299792.458

# =============================================================================
# NUMBA-ACCELERATED PHYSICS KERNELS
# =============================================================================

@njit(fastmath=True)
def calc_H_Hubble(z: float, H0: float, Omega_m: float, Omega_L: float) -> float:
    """
    Calculates the Hubble expansion rate H(z) at redshift z in a flat LCDM universe.
    
    Math:
        H(z) = H_0 * sqrt(\\Omega_m * (1 + z)^3 + \\Omega_\\Lambda)
        where \\Omega_\\Lambda (dark energy density) is typically 1 - \\Omega_m.
        This governs the cosmological time step dt = dz / [H(z) * (1+z)].
        
    Args:
        z (float): Cosmological redshift [dimensionless].
        H0 (float): Present-day Hubble constant [eV].
        Omega_m (float): Matter density parameter [dimensionless].
        Omega_L (float): Dark energy density parameter [dimensionless].
        
    Returns:
        float: Hubble expansion rate H(z) [eV].
    """
    return H0 * np.sqrt(Omega_m * (1.0 + z)**3 + Omega_L)

@njit(fastmath=True)
def calc_R_source(z: float, rho_evol: float) -> float:
    """
    Calculates the cosmological source distribution R(z) relative to z=0.
    
    Math: 
        R(z) = SFR(z) * (1 + z)^{\\rho_{evol}}
        SFR(z) follows a broken power-law parameterizing the Star Formation Rate 
        (e.g., Yuksel et al. 2008). The exponents govern the rise, plateau, and 
        decay of source density across cosmic epochs.
        
    Args:
        z (float): Cosmological redshift [dimensionless].
        rho_evol (float): Additional source evolution index modifier [dimensionless].
        
    Returns:
        float: Normalized source density R(z) at redshift z [dimensionless].
    """
    if z < 1.0:
        sfr = (1.0 + z)**3.4
    elif z < 4.0:
        sfr = (2.0**3.4) * ((1.0 + z) / 2.0)**(-0.3)
    else:
        sfr = (2.0**3.4) * (5.0 / 2.0)**(-0.3) * ((1.0 + z) / 5.0)**(-3.5)
        
    return sfr * (1.0 + z)**rho_evol

@njit
def construct_PMNS(th12: float, th13: float, th23: float, d_cp: float) -> np.ndarray:
    """
    Constructs the standard 3x3 unitary PMNS neutrino mixing matrix.
    
    Math:
        U = R_{23}(\\theta_{23}) * R_{13}(\\theta_{13}, \\delta_{CP}) * R_{12}(\\theta_{12})
        where R_{ij} are rotation matrices. The CP-violating phase \\delta_{CP} 
        is embedded natively inside R_{13}.
        
    Args:
        th12 (float): Mixing angle \\theta_{12} [radians].
        th13 (float): Mixing angle \\theta_{13} [radians].
        th23 (float): Mixing angle \\theta_{23} [radians].
        d_cp (float): CP-violating Dirac phase \\delta_{CP} [radians].
        
    Returns:
        np.ndarray: 3x3 complex PMNS matrix [dimensionless].
    """
    c12, s12 = np.cos(th12), np.sin(th12)
    c13, s13 = np.cos(th13), np.sin(th13)
    c23, s23 = np.cos(th23), np.sin(th23)
    
    phase = np.exp(1j * d_cp)
    phase_conj = np.exp(-1j * d_cp)
    
    R12 = np.zeros((3, 3), dtype=np.complex128)
    R12[0, 0] = c12;  R12[0, 1] = s12
    R12[1, 0] = -s12; R12[1, 1] = c12
    R12[2, 2] = 1.0
    
    R13 = np.zeros((3, 3), dtype=np.complex128)
    R13[0, 0] = c13;  R13[0, 2] = s13 * phase_conj
    R13[1, 1] = 1.0
    R13[2, 0] = -s13 * phase; R13[2, 2] = c13
    
    R23 = np.zeros((3, 3), dtype=np.complex128)
    R23[0, 0] = 1.0
    R23[1, 1] = c23;  R23[1, 2] = s23
    R23[2, 1] = -s23; R23[2, 2] = c23
    
    return R23 @ R13 @ R12

@njit(fastmath=True)
def calc_debye_mass_sq(g_D: float, u_arr: np.ndarray, T_nu: float, eta_bg: np.ndarray) -> float:
    """
    Calculates the Debye screening mass squared generated by the thermal CnuB plasma.
    
    Math:
        m_D^2 = (g_D^2 / 6) * \\sum_i ( |u_i|^4 * T_{\\nu, i}^2 )
        The background flavor asymmetry \\eta_{bg} shifts the effective thermal distributions 
        of the C\\nu B mass eigenstates, orthogonalizing the Debye screening.
        
    Args:
        g_D (float): Dark gauge coupling constant [dimensionless].
        u_arr (np.ndarray): 1D array of flavor-specific dark charges [dimensionless].
        T_nu (float): Baseline temperature of the CnuB [eV].
        eta_bg (np.ndarray): Dimensionless flavor asymmetry multipliers for the CnuB [dimensionless].
        
    Returns:
        float: Debye mass squared m_D^2 [eV^2].
    """
    u_mag_4 = np.abs(u_arr)**4
    # Applying the dimensionless asymmetry multiplier to the effective background temperature
    T_nu_alpha_sq = (T_nu**2) * (eta_bg**(2.0/3.0))
    return (g_D**2 / 6.0) * np.sum(u_mag_4 * T_nu_alpha_sq)

@njit
def calc_V_LRI(z: float, u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
               n_n0: float, T_nu0: float, use_debye_screening: bool, n_n_local: float, 
               eta_bg: np.ndarray) -> np.ndarray:
    """
    Computes the macroscopic effective potential V_{LRI} sourced by cosmological and local neutrons
    through kinetic mixing with the dark sector.
    
    Math:
        A_{LRI} = \\left[ \\frac{g_D \\epsilon'}{m_{Z_D}^2 + m_D^2} \\right] 
                  \\left( \\frac{4 c_W m_Z^2}{g_w} \\right) 
                  \\left( \\frac{\\sqrt{2}}{2} G_F n_n(z) \\right)
                  
        V_{LRI, ij} = A_{LRI} \\cdot u_i u_j^*
        
        This potential acts identically to an MSW matter potential but scales with the neutron 
        number density n_n(z) and incorporates thermal screening (m_D^2) from the hot Big Bang plasma.
        
    Args:
        z (float): Cosmological redshift [dimensionless].
        u_arr (np.ndarray): 1D array of dark flavor charges [dimensionless].
        m_Zd (float): Mass of the dark Z' vector boson [eV].
        g_D (float): Dark gauge coupling [dimensionless].
        eps_prime (float): Effective kinetic mixing parameter [dimensionless].
        n_n0 (float): Present-day cosmological neutron number density [eV^3].
        T_nu0 (float): Present-day CnuB temperature [eV].
        use_debye_screening (bool): Flag to apply thermal Debye screening.
        n_n_local (float): Local neutron density near Earth/Milky Way [eV^3].
        eta_bg (np.ndarray): CnuB flavor asymmetry array [dimensionless].
        
    Returns:
        np.ndarray: 3x3 complex LRI potential matrix [eV].
    """
    n_n_z = n_n0 * (1.0 + z)**3
    if z <= 1e-10:
        n_n_z += n_n_local

    m_D_sq = 0.0
    if use_debye_screening:
        T_nu_z = T_nu0 * (1.0 + z)
        m_D_sq = calc_debye_mass_sq(g_D, u_arr, T_nu_z, eta_bg)
    
    c_W = 0.88
    m_Z = 91.18 * GeV
    g_weak = 0.65
    
    A_LRI = (g_D * eps_prime) / (m_Zd**2 + m_D_sq) * (4.0 * c_W * m_Z**2 / g_weak) * (np.sqrt(2)/2.0 * G_F * n_n_z)
    
    V_LRI = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            V_LRI[i, j] = A_LRI * u_arr[i] * np.conjugate(u_arr[j])
            
    return V_LRI

@njit(fastmath=True)
def thermal_averaged_rates_flavor(E0: float, z: float, sum_G: float, sum_G_alpha: np.ndarray, 
                                  m_Zd: float, m_nu_bg_arr: np.ndarray, T_nu0: float, 
                                  eta_mass: np.ndarray) -> np.ndarray:
    """
    Calculates the exact, flavor-dependent flux attenuation rate \\Gamma(E, z)_\\alpha caused 
    by Breit-Wigner s-channel scattering against the thermal C\\nu B via the dark Z' mediator.
    
    Math:
        The total interaction rate integrates over the Fermi-Dirac momentum distribution of the C\\nu B:
        \\Gamma_\\alpha(E) = \\sum_i \\frac{1}{2\\pi} \\int p^2 f_{FD}(p) \\left[ \\int_{-1}^1 \\sigma_{B.W.}(s) d\\cos\\theta \\right] dp
        
        The Mandelstam variable s is defined dynamically by the background kinematics:
        s(p, \\theta) = m_i^2 + 2 E_{inc} E_{bg} - 2 E_{inc} p \\cos(\\theta)
        
        To avoid numerical undersampling of incredibly narrow resonance poles (e.g., small g_D), 
        the inner angular integral over d\\cos(\\theta) is evaluated EXACTLY analytically via u-substitution 
        where u = s - M_{Z_D}^2. This guarantees unconditional mathematical stability.
        
    Args:
        E0 (float): Present-day observed neutrino energy [eV].
        z (float): Cosmological redshift [dimensionless].
        sum_G (float): Total coupling magnitude for the Breit-Wigner width calculation.
        sum_G_alpha (np.ndarray): Flavor-partitioned coupling magnitudes.
        m_Zd (float): Dark mediator mass [eV].
        m_nu_bg_arr (np.ndarray): Background neutrino mass eigenvalues [eV].
        T_nu0 (float): Present-day CnuB temperature [eV].
        eta_mass (np.ndarray): CnuB asymmetry projected into the mass basis [dimensionless].
        
    Returns:
        np.ndarray: 1D array (size 3) of interaction rates for each flavor [eV].
    """
    T_nu = T_nu0 * (1.0 + z)
    n_nu_base = 56.0 * (1.0 + z)**3 * cm_inv3_to_eV3 
    
    Gamma_width = (sum_G * m_Zd) / (12.0 * np.pi)
    MG = m_Zd * Gamma_width
    MG_sq = MG**2
    M2 = m_Zd**2
    
    rates = np.zeros(3, dtype=np.float64)
    if MG == 0.0:
        return rates 
        
    prefactor = 1.0 / (2.0 * np.pi)
    
    for i in range(3):
        if eta_mass[i] <= 0:
            continue
            
        n_nu_i = n_nu_base * eta_mass[i]
        m_i_sq = m_nu_bg_arr[i]**2
        T_i = T_nu * (eta_mass[i]**(1.0/3.0))
        E_inc = E0 * (1.0 + z)
        
        n_p = 100
        p_max = 20.0 * T_i
        dp = p_max / n_p
        
        for j in range(n_p):
            p = (j + 0.5) * dp
            E_bg = np.sqrt(m_i_sq + p**2)
            if p / T_i > 50.0: continue 
                
            f_FD = 1.0 / (np.exp(p / T_i) + 1.0)
            if f_FD < 1e-16: continue
                
            p2_f_FD = p**2 * f_FD
            
            # Kinematic constants for the analytical angular integration
            A = m_i_sq + 2.0 * E_inc * E_bg
            B = 2.0 * E_inc * p
            
            if B < 1e-12: 
                I_ang = 2.0 * A / ((A - M2)**2 + MG_sq)
            else:
                u_plus = A - M2 + B
                u_minus = A - M2 - B
                C = 1.0 - (A - M2) / B
                D = 1.0 - (A - 2.0 * M2) / B
                F = C * M2 - MG_sq / B
                # The exact analytical solution for the angular integration to avoid pole-skipping
                I_ang = (2.0 / B) + (D / (2.0 * B)) * np.log((u_plus**2 + MG_sq) / (u_minus**2 + MG_sq)) + (F / (B * MG)) * (np.arctan(u_plus / MG) - np.arctan(u_minus / MG))
                
            if I_ang > 0:
                integral_val = prefactor * p2_f_FD * I_ang * dp
                for alpha in range(3):
                    rates[alpha] += integral_val * sum_G_alpha[alpha] * (n_nu_i / n_nu_base)
                    
    return rates

@njit
def get_adiabatic_mixing(E0: float, z: float, U_pmns: np.ndarray, M_sq: np.ndarray,
                         u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
                         n_n0: float, T_nu0: float, use_debye_screening: bool, n_n_local: float,
                         use_SM_potential: bool, n_e_earth: float, is_antineutrino: bool,
                         eta_bg: np.ndarray) -> np.ndarray:
    """
    Computes the instantaneous mass eigenstates (eigenvectors) of the total effective Hamiltonian,
    accounting for vacuum mixing, standard matter effects, and new dark portal interactions.
    
    Math:
        H_{vac} = \\frac{1}{2 E_z} U \\cdot M_{sq} \\cdot U^\\dagger
        H_{tot} = H_{vac} + V_{LRI} + V_{SM}
        
        Note on CP Violation: If calculating for antineutrinos, the Hamiltonian is evaluated 
        by conjugating the vacuum mixing matrices (both U_{PMNS} and U_{sterile}) and 
        reversing the sign of all coherent matter potentials (V_{LRI} -> -V_{LRI}).
        
    Args:
        E0 (float): Neutrino energy at Earth [eV].
        z (float): Cosmological redshift [dimensionless].
        U_pmns (np.ndarray): 3x3 PMNS mixing matrix [dimensionless].
        M_sq (np.ndarray): 3x3 Diagonal mass squared matrix [eV^2].
        u_arr, m_Zd, g_D, eps_prime, n_n0, T_nu0, use_debye_screening, n_n_local: Parameters for V_{LRI}.
        use_SM_potential (bool): Flag to apply standard MSW matter potential at Earth.
        n_e_earth (float): Standard matter electron density near/at Earth [eV^3].
        is_antineutrino (bool): True if evaluating the antineutrino Hamiltonian.
        eta_bg (np.ndarray): CnuB asymmetry parameters.
        
    Returns:
        np.ndarray: 3x3 matrix where the columns represent the instantaneous eigenvectors of H_{tot} [dimensionless].
    """
    E_z = E0 * (1.0 + z)
    H_vac = (1.0 / (2.0 * E_z)) * (U_pmns @ M_sq @ np.conjugate(U_pmns).T)
    
    V_LRI = calc_V_LRI(z, u_arr, m_Zd, g_D, eps_prime, n_n0, T_nu0, use_debye_screening, n_n_local, eta_bg)
    
    V_SM = np.zeros((3, 3), dtype=np.complex128)
    if use_SM_potential and z <= 1e-10: 
        V_SM[0, 0] = np.sqrt(2.0) * G_F * n_e_earth
        
    if is_antineutrino:
        V_LRI = -np.conjugate(V_LRI)
        V_SM = -np.conjugate(V_SM)
            
    H_tot = H_vac + V_LRI + V_SM
    evals, evecs = np.linalg.eigh(H_tot)
    return evecs

@njit
def propagate_spectrum(E_bins: np.ndarray, z_bins: np.ndarray, rho_src: np.ndarray, 
                       H0: float, Omega_m: float, Omega_L: float,
                       U_pmns: np.ndarray, M_sq: np.ndarray,
                       u_arr: np.ndarray, m_Zd: float, g_D: float, eps_prime: float,
                       m_nu_bg_arr: np.ndarray, n_n0: float, T_nu0: float,
                       use_debye_screening: bool, n_n_local: float,
                       use_SM_potential: bool, n_e_earth: float,
                       is_antineutrino: bool, p_index: float, rho_evol: float,
                       Phi_0: float, eta_bg: np.ndarray) -> np.ndarray:
    """
    Executes the vectorized adiabatic transport integration across the entire energy spectrum.
    
    Math:
        The fundamental integro-differential equation dictating the flux F(E, z) is:
        dF(E) / dt = -\\Gamma(E) F(E) + \\int_E^{E_{max}} dE' W(E, E') F(E') + Q(E)
        
        Where:
        - \\Gamma(E) is the attenuation rate due to dark scattering.
        - W(E, E') = \\Gamma(E') \\times (3 E^2 / E'^3) is the exact lab-frame cascade redistribution 
          kernel for an s-channel vector mediator coupling to chiral fermions.
        - Q(E) is the continuous cosmological source injection spectrum.
        
        To solve this robustly despite the extreme "stiffness" introduced by narrow resonances 
        (where \\Gamma \\cdot dt \\gg 1), the routine utilizes an unconditionally stable, semi-analytic 
        exponential time step:
        
        F_{n+1} = F_n e^{-\\Gamma \\Delta t} + \\frac{Source}{\\Gamma} (1 - e^{-\\Gamma \\Delta t})
        
        Integration occurs backwards in redshift (z_max -> 0), dynamically transforming the 
        flux between the flavor and mass basis to account for adiabatic oscillations and MSW effects.
        
    Args:
        E_bins (np.ndarray): 1D array of observed energy bins [eV].
        z_bins (np.ndarray): 1D array of redshift bins (descending order).
        rho_src (np.ndarray): Source flavor ratios [f_e, f_mu, f_tau] [dimensionless].
        H0, Omega_m, Omega_L: Cosmological parameters.
        U_pmns, M_sq: Standard mixing parameter matrices.
        u_arr, m_Zd, g_D, eps_prime, use_debye_screening: LRI parameters.
        m_nu_bg_arr, n_n0, T_nu0, n_n_local, eta_bg: Environmental background properties.
        use_SM_potential, n_e_earth: Standard Matter parameters.
        is_antineutrino (bool): Antineutrino switch for Hamiltonian symmetry.
        p_index (float): Spectral index of the primary injection spectrum.
        rho_evol (float): Source evolution index.
        Phi_0 (float): Baseline source flux normalization.
        
    Returns:
        np.ndarray: Final absolute flux mapped into flavor space at Earth (size N_E x 3) [eV^-1 cm^-2 s^-1 sr^-1].
    """
    N_E = len(E_bins)
    dz = z_bins[0] - z_bins[1] 
    
    # Project CnuB flavor asymmetry onto the background mass eigenstates using the PMNS matrix
    eta_mass = np.zeros(3, dtype=np.float64)
    for i in range(3):
        eta_mass[i] = np.sum(np.abs(U_pmns[:, i])**2 * eta_bg)
    
    F_mass = np.zeros((N_E, 3), dtype=np.float64)
            
    # Calculate NSSI coupling structures (constant over E and z) to speed up integration
    u_mag = np.abs(u_arr)
    sum_G = 0.0
    sum_G_alpha = np.zeros(3, dtype=np.float64)
    for a in range(3):
        for g in range(3):
            coupling_prod = u_mag[a] * u_mag[g]
            val = g_D * coupling_prod * (g_D * coupling_prod + eps_prime * (0.65 / 0.88))
            sum_G += val
            sum_G_alpha[a] += val
            
    BR = np.zeros(3, dtype=np.float64)
    if sum_G > 0:
        BR = sum_G_alpha / sum_G

    dE = np.zeros(N_E, dtype=np.float64)
    for i in range(N_E - 1):
        dE[i] = E_bins[i+1] - E_bins[i]
    if N_E > 1:
        dE[N_E-1] = dE[N_E-2] 

    F_flavor = np.zeros((N_E, 3), dtype=np.float64)
    Gamma_flavor = np.zeros((N_E, 3), dtype=np.float64)
    dF_casc_flavor = np.zeros((N_E, 3), dtype=np.float64)
    U_z_all = np.zeros((N_E, 3, 3), dtype=np.complex128)

    # 2. Adiabatic transport with coupled cascading and continuous continuous injection
    for k in range(1, len(z_bins)):
        z = z_bins[k]
        dt = dz / (calc_H_Hubble(z, H0, Omega_m, Omega_L) * (1.0 + z))
        R_z = calc_R_source(z, rho_evol)
        
        # Determine instantaneous rates and mixings across the spectrum
        for i in range(N_E):
            U_z = get_adiabatic_mixing(E_bins[i], z, U_pmns, M_sq, u_arr, m_Zd, g_D, eps_prime, 
                                       n_n0, T_nu0, use_debye_screening, n_n_local, 
                                       use_SM_potential, n_e_earth, is_antineutrino, eta_bg)
            U_z_all[i] = U_z
            for a in range(3):
                F_flavor[i, a] = np.sum(np.abs(U_z[a, :])**2 * F_mass[i, :])
            Gamma_flavor[i, :] = thermal_averaged_rates_flavor(E_bins[i], z, sum_G, sum_G_alpha, m_Zd, m_nu_bg_arr, T_nu0, eta_mass)
            
        # Compute cascading integral (top-down scattering)
        for i in range(N_E):
            integral = 0.0
            # Integrate from local energy bin up to max energy
            for j in range(i, N_E):
                E_prime = E_bins[j]
                rate_F_sum = 0.0
                for b in range(3):
                    rate_F_sum += Gamma_flavor[j, b] * F_flavor[j, b]
                
                # Applying the correct inelasticity distribution polynomial for vector decay: 3 * E^2 / E'^3
                integral += 1.0 * (rate_F_sum * 3.0 * (E_bins[i]**2) / (E_prime**3)) * dE[j]
                
            for a in range(3):
                dF_casc_flavor[i, a] = BR[a] * integral
                
        # Apply injection, attenuation, and cascading using a stable semi-analytic step
        for i in range(N_E):
            U_z = U_z_all[i]
            E_z = E_bins[i] * (1.0 + z)
            
            # Injection spectrum evaluated at physical emission energy E_z, scaled by Phi_0
            # Kinematic Jacobian fixed to rely purely on cosmological expansion terms
            inj_spectrum = Phi_0 * (E_z**(-p_index))
            
            for a in range(3):
                Gamma_mass_a = np.sum(np.abs(U_z[:, a])**2 * Gamma_flavor[i, :])
                dF_casc_mass_a = np.sum(np.abs(U_z[:, a])**2 * dF_casc_flavor[i, :])
                inj_rate_mass_a = np.sum(np.abs(U_z[:, a])**2 * rho_src) * inj_spectrum * R_z
                
                total_rate = dF_casc_mass_a + inj_rate_mass_a
                
                if Gamma_mass_a > 1e-30:
                    decay = np.exp(-Gamma_mass_a * dt)
                    F_mass[i, a] = F_mass[i, a] * decay + (total_rate / Gamma_mass_a) * (1.0 - decay)
                else:
                    F_mass[i, a] += total_rate * dt
                
    # 3. Final projection to flavor basis at Earth (z=0)
    F_earth = np.zeros((N_E, 3), dtype=np.float64)
    for i in range(N_E):
        U_earth = get_adiabatic_mixing(E_bins[i], 0.0, U_pmns, M_sq, u_arr, m_Zd, g_D, eps_prime, 
                                       n_n0, T_nu0, use_debye_screening, n_n_local, 
                                       use_SM_potential, n_e_earth, is_antineutrino, eta_bg)
        for a in range(3):
            F_earth[i, a] = np.sum(np.abs(U_earth[a, :])**2 * F_mass[i, :])
            
    return F_earth

# =============================================================================
# FILE EXPORT ROUTINE
# =============================================================================

def save_flux_to_file(E_array: np.ndarray, Flux_array: np.ndarray, filename: str = "../output/data/output_flux.txt"):
    """
    Saves the generated flux components (Energy, nue, numu, nutau) to an external text file.
    
    Args:
        E_array (np.ndarray): 1D Array of Energy values [eV].
        Flux_array (np.ndarray): 2D array (size N_E x 3) of unscaled absolute flux values.
        filename (str): The destination file path.
        
    Returns:
        None
    """
    header = "Energy[eV]    Flux_nue[eV^-1 cm^-2 s^-1 sr^-1]    Flux_numu[eV^-1 cm^-2 s^-1 sr^-1]    Flux_nutau[eV^-1 cm^-2 s^-1 sr^-1]"
    data = np.column_stack((E_array, Flux_array))
    np.savetxt(filename, data, header=header, fmt="%.6e")
    print(f"\nFlux data saved to: {os.path.abspath(filename)}")

# =============================================================================
# MULTIPROCESSING WRAPPER & CONFIGURATION MANAGER
# =============================================================================

def get_mass_eigenvalues(m_lightest: float, dm21_sq: float, dm31_sq: float, ordering: str) -> np.ndarray:
    """
    Calculates the three absolute mass eigenvalues based on the lightest mass and mass ordering hierarchy.
    
    Math:
        Normal Hierarchy (NH): m_1 = m_{lightest}, m_2 = \\sqrt{m_1^2 + \\Delta m_{21}^2}, m_3 = \\sqrt{m_1^2 + \\Delta m_{31}^2}.
        Inverted Hierarchy (IH): m_3 = m_{lightest}, m_2 = \\sqrt{m_3^2 + |\\Delta m_{31}^2|}, m_1 = \\sqrt{m_2^2 - \\Delta m_{21}^2}.
        
    Args:
        m_lightest (float): Mass of the lightest neutrino eigenstate [eV].
        dm21_sq (float): Solar mass squared difference [eV^2].
        dm31_sq (float): Atmospheric mass squared difference [eV^2].
        ordering (str): Hierarchy label ("normal" or "inverted").
        
    Returns:
        np.ndarray: 1D array (size 3) of mass eigenvalues [eV].
    """
    masses = np.zeros(3, dtype=np.float64)
    if ordering.lower().strip() == "inverted":
        masses[2] = m_lightest
        masses[1] = np.sqrt(masses[2]**2 + dm31_sq)
        masses[0] = np.sqrt(masses[1]**2 - dm21_sq)
    else:
        masses[0] = m_lightest
        masses[1] = np.sqrt(masses[0]**2 + dm21_sq)
        masses[2] = np.sqrt(masses[0]**2 + dm31_sq)
    return masses

def propagation_worker(args) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to unpack arguments and execute `propagate_spectrum` 
    within the multiprocessing pool loop. Identifies and routes neutrino vs. 
    antineutrino computation branches.
    
    Args:
        args (tuple): Contains (E_bins array, params dictionary).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The evaluated energy array and corresponding flux array.
    """
    E_bins, params = args
    
    z_bins = np.linspace(params['z_max'], 0.0, int(params['n_z_bins']))
    rho_src = np.array([params['f_e_S'], params['f_mu_S'], params['f_tau_S']], dtype=np.float64)
    
    ordering = str(params.get('mass_ordering', 'normal')).strip().lower()
    dm31_sq_signed = -params['dm31_sq'] if ordering == "inverted" else params['dm31_sq']
    M_sq = np.diag([0.0, params['dm21_sq'], dm31_sq_signed]).astype(np.complex128)
    
    u_arr_base = np.array([params['u_e'], params['u_mu'], params['u_tau']], dtype=np.complex128)
    eta_bg = params['eta_bg']
    
    def compute_flux(is_anti: bool) -> np.ndarray:
        U_pmns = construct_PMNS(params['th12'], params['th13'], params['th23'], params['d_cp'])
        u_arr = u_arr_base.copy()
        
        if is_anti:
            # Complex CP-violation conjugate transformations for the antineutrino solver branch
            U_pmns = np.conjugate(U_pmns)
            u_arr = np.conjugate(u_arr) 
            
        return propagate_spectrum(
            E_bins, z_bins, rho_src,
            params['H0_eV'], params['Omega_m'], params['Omega_L'],
            U_pmns, M_sq, u_arr, params['m_Zd_eV'], params['g_D'], params['epsilon_prime'],
            params['m_nu_bg_arr'], params['n_n0_eV3'], params['T_nu0_eV'],
            params['use_debye_screening'], params['n_n_local_eV3'],
            params['use_SM_potential'], params['n_e_earth_eV3'],
            is_anti, params['p_index'], params['rho_evol'], params['Phi_0'], eta_bg
        )
    
    flag_val = str(params.get('is_antineutrino', 'sum')).strip().lower()
    
    if flag_val in ['sum', 'both']:
        flux_at_earth = compute_flux(False) + compute_flux(True)
    elif flag_val in ['true', 'antineutrino', 'antinu', 'nubar']:
        flux_at_earth = compute_flux(True)
    else:
        flux_at_earth = compute_flux(False)
    
    return E_bins, flux_at_earth

def run_solver(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main orchestrator function. Sets up the computational grid, parses the JSON config, 
    evaluates physics parameters, and executes the vectorized parallel map.
    
    Args:
        config (Dict): The parsed dictionary from the config JSON file.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Evaluated energy bins [eV] and evaluated absolute flux [eV^-1 cm^-2 s^-1 sr^-1].
    """
    params = {}
    params['H0_eV'] = (config['cosmology']['H0_km_s_Mpc'] * km_per_s) / Mpc_to_eV_inv
    params['Omega_m'] = config['cosmology']['Omega_m']
    params['Omega_L'] = config['cosmology']['Omega_L']
    params['T_nu0_eV'] = config['cosmology']['T_nu0_eV']
    params['n_n0_eV3'] = config['cosmology']['n_n0_cm3'] * cm_inv3_to_eV3
    params['n_n_local_eV3'] = config['cosmology'].get('n_n_local_cm3', 1.0) * cm_inv3_to_eV3
    params['mass_ordering'] = config['cosmology'].get('mass_ordering', 'normal')
    
    params['eta_bg'] = np.array([
        config['cosmology'].get('eta_e', 1.0),
        config['cosmology'].get('eta_mu', 1.0),
        config['cosmology'].get('eta_tau', 1.0)
    ], dtype=np.float64)
    
    params['dm21_sq'] = config['standard_mixing']['dm21_sq']
    params['dm31_sq'] = config['standard_mixing']['dm31_sq']
    params['th12'] = np.radians(config['standard_mixing']['th12_deg'])
    params['th13'] = np.radians(config['standard_mixing']['th13_deg'])
    params['th23'] = np.radians(config['standard_mixing']['th23_deg'])
    params['d_cp'] = np.radians(config['standard_mixing']['d_cp_deg'])
    params['use_SM_potential'] = config['standard_mixing'].get('use_SM_potential', False)
    params['n_e_earth_eV3'] = config['standard_mixing'].get('n_e_earth_cm3', 1.5e24) * cm_inv3_to_eV3
    
    params['m_nu_bg_arr'] = get_mass_eigenvalues(
        config['cosmology']['m_lightest_eV'],
        params['dm21_sq'],
        params['dm31_sq'],
        params['mass_ordering']
    )
    
    params['u_e'] = config['dark_portal']['u_e']
    params['u_mu'] = config['dark_portal']['u_mu']
    params['u_tau'] = config['dark_portal']['u_tau']
    params['m_Zd_eV'] = config['dark_portal']['m_Zd_eV']
    params['g_D'] = config['dark_portal']['g_D']
    params['epsilon_prime'] = config['dark_portal']['epsilon_prime']
    params['use_debye_screening'] = config['dark_portal'].get('use_debye_screening', True)
    
    params['p_index'] = config['source']['p_index']
    params['Phi_0'] = config['source'].get('Phi_0', 1.0) 
    params['z_max'] = config['source']['z_max']
    params['rho_evol'] = config['source']['rho_evol']
    params['f_e_S'] = config['source']['f_e_S']
    params['f_mu_S'] = config['source']['f_mu_S']
    params['f_tau_S'] = config['source']['f_tau_S']
    params['is_antineutrino'] = config['source'].get('is_antineutrino', 'sum')
    
    params['n_z_bins'] = config['grid']['n_z_bins']
    
    E_bins = np.logspace(np.log10(config['grid']['E_min_eV']), 
                         np.log10(config['grid']['E_max_eV']), 
                         int(config['grid']['n_E_bins']))
    
    n_cores = int(config['execution'].get('n_cores', 1))
    
    tasks = [(E_bins, params)]
    
    print(f"Initializing solver across {len(E_bins)} coupled energy bins...")
    results = []
    
    if n_cores > 1:
        print(f"Deploying {n_cores} multiprocessing workers.")
        with multiprocessing.Pool(processes=n_cores) as pool:
            if HAS_TQDM:
                for res in tqdm(pool.imap(propagation_worker, tasks), total=len(tasks), desc="Integrating Spectrum"):
                    results.append(res)
            else:
                results = pool.map(propagation_worker, tasks)
    else:
        print("Deploying single-core worker.")
        iterator = tqdm(tasks, desc="Integrating Spectrum") if HAS_TQDM else tasks
        for task in iterator:
            results.append(propagation_worker(task))
            
    E_out = results[0][0]
    Flux_out = results[0][1]
    
    return E_out, Flux_out

def plot_flavor_fluxes(E_array: np.ndarray, Flux_array: np.ndarray, p_index: float, output_filename: str = "../output/plots/flux_plot.png"):
    """
    Generates, saves, and displays a 2-panel plot of the final neutrino flux at Earth.
    - Top panel: Absolute diffuse flux scaled by E_0^2.
    - Bottom panel: Relative flavor ratios (f_e, f_mu, f_tau) as a function of energy.
    
    Args:
        E_array (np.ndarray): 1D Array of Energy values [eV].
        Flux_array (np.ndarray): 2D array (size N_E x 3) of unscaled flux values [eV^-1 cm^-2 s^-1 sr^-1].
        p_index (float): Original source spectral index (used for reference/titles).
        output_filename (str): The destination file path to save the generated plot.
        
    Returns:
        None
    """
    E_GeV = E_array / 1e9
    scaling_factor = E_GeV**2.0
    
    flux_e = Flux_array[:, 0]
    flux_mu = Flux_array[:, 1]
    flux_tau = Flux_array[:, 2]
    flux_total = np.sum(Flux_array, axis=1)
    
    # Calculate scaled fluxes for the upper panel
    s_flux_e = flux_e * scaling_factor
    s_flux_mu = flux_mu * scaling_factor
    s_flux_tau = flux_tau * scaling_factor
    s_flux_total = flux_total * scaling_factor
    
    # Calculate the normalized flavor ratios for the lower panel
    # We use np.maximum to ensure we never divide by exactly zero if the flux completely attenuates
    safe_total = np.maximum(flux_total, 1e-100)
    ratio_e = flux_e / safe_total
    ratio_mu = flux_mu / safe_total
    ratio_tau = flux_tau / safe_total
    
    # Set up the 2-panel figure structure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    # --- Top Panel: Scaled Absolute Flux ---
    ax1.plot(E_GeV, s_flux_total, color='black', lw=2.5, label='Total Flux')
    ax1.plot(E_GeV, s_flux_e, color='crimson', linestyle='--', lw=2.0, label=r'$\nu_e$ Component')
    ax1.plot(E_GeV, s_flux_mu, color='royalblue', linestyle='-.', lw=2.0, label=r'$\nu_\mu$ Component')
    ax1.plot(E_GeV, s_flux_tau, color='forestgreen', linestyle=':', lw=2.0, label=r'$\nu_\tau$ Component')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(f'Scaled Flux $E_0^2 \\Phi$ [Arb. Units]', fontsize=14)
    ax1.set_title('Astrophysical Neutrino Diffuse Flux at Earth', fontsize=15, pad=15)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(fontsize=12, loc='lower right')
    
    # --- Bottom Panel: Relative Flavor Ratios ---
    ax2.plot(E_GeV, ratio_e, color='crimson', linestyle='--', lw=2.0)
    ax2.plot(E_GeV, ratio_mu, color='royalblue', linestyle='-.', lw=2.0)
    ax2.plot(E_GeV, ratio_tau, color='forestgreen', linestyle=':', lw=2.0)
    
    ax2.set_ylabel('Flavor Ratio', fontsize=14)
    ax2.set_xlabel(r'Observed Energy $E_0$ [GeV]', fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot successfully saved to: {os.path.abspath(output_filename)}")
    plt.show()

# =============================================================================
# ENTRY POINT & CONFIGURATION GENERATION
# =============================================================================

def create_default_config(filename: str = "../config/config_dark_portal.json") -> Dict:
    """
    Generates a strictly formatted JSON file if one does not exist, establishing 
    the baseline cosmological and particle physics parameters.
    
    Args:
        filename (str): Path where the default JSON configuration will be written.
        
    Returns:
        Dict: A dictionary containing the newly created default configuration parameters.
    """
    default_config = {
        "cosmology": {
            "H0_km_s_Mpc": 67.4,
            "Omega_m": 0.315,
            "Omega_L": 0.685,
            "T_nu0_eV": 1.95e-4,
            "n_n0_cm3": 0.25e-6,
            "n_n_local_cm3": 1.0, 
            "m_lightest_eV": 0.01,
            "mass_ordering": "normal",
            "eta_e": 1.0,
            "eta_mu": 1.0,
            "eta_tau": 1.0
        },
        "standard_mixing": {
            "dm21_sq": 7.41e-5,
            "dm31_sq": 2.511e-3,
            "th12_deg": 33.41,
            "th13_deg": 8.58,
            "th23_deg": 49.1,
            "d_cp_deg": 132.0,
            "use_SM_potential": False,
            "n_e_earth_cm3": 1.5e24
        },
        "dark_portal": {
            "u_e": 0.5,
            "u_mu": 0.5,
            "u_tau": 0.0,
            "m_Zd_eV": 2.0e6,
            "g_D": 1.e-4,
            "epsilon_prime": 1e-1,
            "use_debye_screening": True
        },
        "source": {
            "p_index": 2.5,
            "Phi_0": 1.0, 
            "z_max": 4.0,
            "rho_evol": 0.0,
            "f_e_S": 0.333333,
            "f_mu_S": 0.666667,
            "f_tau_S": 0.0,
            "is_antineutrino": "sum"
        },
        "grid": {
            "E_min_eV": 1e13,
            "E_max_eV": 3e16,
            "n_E_bins": 1000,
            "n_z_bins": 1000
        },
        "output": {
            "flux_filename": "../output/data/output_flux.txt",
            "plot_filename": "../output/plots/flux_plot.png"
        },
        "execution": {
            "n_cores": multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
        }
    }
    
    # Ensure directory exists before writing
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(default_config, f, indent=4)
    print(f"Created default configuration file: {filename}")
    return default_config

if __name__ == "__main__":
    config_file = "../config/config_dark_portal.json"
    
    if not os.path.exists(config_file):
        config = create_default_config(config_file)
    else:
        print(f"Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
            
    E_array, Flux_array = run_solver(config)
    
    # Safely retrieve output filenames from the config, fallback if missing
    output_cfg = config.get("output", {})
    flux_filename = output_cfg.get("flux_filename", "../output/data/output_flux.txt")
    plot_filename = output_cfg.get("plot_filename", "../output/plots/flux_plot.png")
    
    # Ensure the target directory for the flux text file exists
    os.makedirs(os.path.dirname(os.path.abspath(flux_filename)), exist_ok=True)
    save_flux_to_file(E_array, Flux_array, filename=flux_filename)
    
    Flux_array = np.clip(Flux_array, 1e-50, None) 
    Total_Flux = np.sum(Flux_array, axis=1)
    f_e, f_mu, f_tau = Flux_array[:, 0] / Total_Flux, Flux_array[:, 1] / Total_Flux, Flux_array[:, 2] / Total_Flux
    
    print("\nIntegration Complete. Sample Output (Mid-grid Energy):")
    mid_idx = len(E_array) // 2
    print(f"Energy: {E_array[mid_idx] / 1e9:.2e} GeV")
    print(f"Flavor Ratios at Earth [e, mu, tau]: [{f_e[mid_idx]:.3f}, {f_mu[mid_idx]:.3f}, {f_tau[mid_idx]:.3f}]")

    if HAS_MATPLOTLIB:
        # Ensure the target directory for the plot exists
        os.makedirs(os.path.dirname(os.path.abspath(plot_filename)), exist_ok=True)
        plot_flavor_fluxes(E_array, Flux_array, config['source']['p_index'], output_filename=plot_filename)