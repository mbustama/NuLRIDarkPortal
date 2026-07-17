# Astrophysical Neutrino Dark Portal Solver

This repository contains a high-performance, Python-based numerical solver for simulating the cosmological propagation of high-energy astrophysical neutrinos under the influence of new dark sector physics. 

The code evaluates a comprehensive framework featuring a light vector mediator ($Z_D$) that couples to active neutrinos via mixing with a sterile state, and to standard matter via kinetic mixing ($\epsilon'$). It explicitly tracks flavor evolution, resonant attenuation against the Cosmic Neutrino Background (C$\nu$B), exact energy cascading (down-scattering), and environmentally dependent long-range interactions (LRI).

---

## 1. Physics and Mathematics Framework

The solver evaluates the continuous integro-differential equation dictating the cosmological transport of the astrophysical neutrino density matrix, propagating it backwards in redshift from $z_{max}$ to $z=0$. 

The total flavor flux $F(E)$ evolves according to:
$$\frac{dF_i(E, z)}{dt} = -\Gamma_i(E, z) F_i(E, z) + \int_E^{E_{max}} dE' W(E, E') \sum_\alpha \left[ \Gamma_\alpha(E', z) F_\alpha(E', z) \right] + Q_{i}(E, z)$$

Where:
* $t$ is the comoving cosmological time tied to the Hubble expansion $dt = \frac{dz}{H(z)(1+z)}$.
* $\Gamma_i$ is the resonant absorption rate.
* $W(E, E')$ is the inelastic cascading kernel.
* $Q_i$ is the continuous source injection term.

### 1.1 Adiabatic Flavor Mixing and Long-Range Interactions (LRI)
Neutrinos propagate as instantaneous mass eigenstates of an effective Hamiltonian that incorporates vacuum mixing, Standard Model (SM) matter effects, and a new macroscopic Long-Range Interaction (LRI) potential sourced by ambient cosmological neutrons.

$$H_{tot} = H_{vac} + V_{SM} + V_{LRI}$$

* **Vacuum Hamiltonian:** $H_{vac} = \frac{1}{2E} U_{PMNS} M^2 U_{PMNS}^\dagger$. The absolute mass scale is dynamically computed using the user-defined mass of the lightest state and the mass ordering hierarchy (Normal or Inverted).
* **SM Matter Potential:** $V_{SM} = \sqrt{2} G_F n_e \text{diag}(1, 0, 0)$, activated near/inside the Earth.
* **LRI Potential:** Sourced by the local and cosmological neutron density $n_n(z)$, mediated by the dark vector $Z_D$ through kinetic mixing $\epsilon'$:

$$V_{LRI, ij} = \left[ \frac{g_D \epsilon'}{m_{Z_D}^2 + m_D^2} \right] \left( \frac{4 c_W m_Z^2}{g_w} \right) \left( \frac{\sqrt{2}}{2} G_F n_n(z) \right) u_i u_j^*$$

**Debye Screening & Flavor Asymmetry:** The LRI potential is dynamically screened by the hot plasma of the C$\nu$B, which generates a Debye mass $m_D$. The solver accounts for theoretical C$\nu$B lepton asymmetries ($\eta_\alpha \equiv n_{\nu_\alpha}/n_{\nu, \text{sym}}$), which orthogonalizes the screening across flavors:
$$m_D^2 = \frac{g_D^2}{6} \sum_i \vert{}u_i\vert{}^4 T_{\nu, i}^2 \quad \text{where} \quad T_{\nu, i}^2 = T_{\nu, \text{base}}^2 \cdot \eta_i^{2/3}$$

*(Note on CP Violation: For the antineutrino branch, the solver rigorously applies complex conjugation to both $U_{PMNS}$ and the sterile coupling vector $u_i$, while reversing the sign of both $V_{SM}$ and $V_{LRI}$.)*

### 1.2 Resonant Attenuation (Absorption)
High-energy neutrinos scatter resonantly off the C$\nu$B via $s$-channel $Z_D$ exchange. The attenuation rate $\Gamma(E)$ is determined by integrating the Breit-Wigner cross-section over the full Fermi-Dirac momentum distribution $f_{FD}(p)$ of the C$\nu$B:

$$\Gamma_\alpha(E) = \sum_{i=1}^3 \eta_i \cdot \frac{1}{2\pi} \int_0^\infty p^2 f_{FD}(p) \left[ \int_{-1}^1 \sigma_{BW}(s(p, \theta)) \, d\cos\theta \right] dp$$

To avoid numerical "pole-skipping" when the coupling $g_D$ is extremely small (resulting in a heavily stiff, narrow resonance width), the inner angular integral over $d\cos\theta$ is evaluated **exactly analytically** within the solver.

### 1.3 Energy Cascading (Down-Scattering)
For every resonant scattering event ($Z_D \to \nu + \bar{\nu}$), particles cascade down to lower energies. The solver redistributes this flux using the exact lab-frame differential cross-section for the decay of a vector mediator into chiral fermions. The inelasticity distribution polynomial is:
$$W(E, E') = 3 \frac{E^2}{(E')^3}$$

### 1.4 Continuous Source Injection
Instead of propagating a point source, the solver initializes an empty universe and integrates continuous flux injections at every redshift step $dz$. The source term $Q(E, z)$ factors in a broken power-law Star Formation Rate (SFR) source evolution $R(z)$, modulated by an arbitrary evolution index $\rho_{\text{evol}}$:
$$Q_i(E, z) = f_{\alpha, S} \cdot \Phi_0 E^{-\gamma} \cdot R(z)$$

---

## 2. Numerical Implementation

### 2.1 Stiff-Stable Semi-Analytic Integration
Because the resonance width $\Gamma_{Z_D} \propto g_D^2$, small couplings generate astronomically high, narrow interaction rates where standard explicit Euler integration ($\Delta F = -\Gamma F \Delta t$) causes catastrophic unphysical numerical explosions. 

This solver circumvents this by utilizing an exact semi-analytic exponential time step for the coupled ODE:
$$F_{n+1} = F_n e^{-\Gamma \Delta t} + \frac{\text{Total Source}}{\Gamma} \left( 1 - e^{-\Gamma \Delta t} \right)$$
In the limit $\Gamma \Delta t \to \infty$, this unconditionally stable method safely collapses to the true physical equilibrium limit $F = \text{Source}/\Gamma$.

### 2.2 Vectorization & Multiprocessing
The energy cascading mechanism rigidly couples higher-energy bins to lower-energy bins. Therefore, the solver does not integrate individual energy bins independently. Instead, it vectorizes the entire spectrum $\vec{E}$ and propagates it uniformly through $dz$. For parameter space scans, the code is structurally wrapped in a Python `multiprocessing` pool, ready to distribute distinct parameter sets across available CPU cores.

### 2.3 Spatial Handover (Local Universe)
To accurately capture the potentials of gravitationally bound local structures, the solver employs a two-phase integration strategy. First, the cosmological baseline is established by integrating the flux down from $z_{max}$ to $z=0$. The resulting density matrix is then handed over as the initial condition for a purely spatial ODE solver. Because $z=0$ is now fixed, the Hubble time step $dt$ is replaced by a flat-space path-length step $dr$.

The spatial solver dynamically updates the effective Hamiltonian as the neutrino propagates through the local environment toward the IceCube detector:
* **The Milky Way:** Treated as an extended source. The solver evaluates the ambient neutron density from the galaxy's bulge, discs, and halo at Earth's distance of 8.33 kpc from the Galactic Center.
* **The Sun and Moon:** Treated as point sources. As the neutrino crosses their respective distances (1 A.U. and $4 \cdot 10^5$ km), their potentials are calculated based on their total neutron content.
* **The Earth:** Treated as an extended source. The solver integrates through the specific arrival chord using the varying neutron densities of the Preliminary Reference Earth Model (PREM). This ensures both the long-range potential and the Standard Model MSW matter potential are captured correctly as they steepen inside the planet.

---

## 3. Repository Structure

~~~text
.
├── dark_portal_solver.py      # Main solver and integration script
├── config/
│   └── config_dark_portal.json # Autogenerated configuration parameters
└── output/
    ├── data/                  # Exported absolute flux data (.txt)
    └── plots/                 # Generated matplotlib figures (.png)
~~~

---

## 4. Installation & Dependencies

The code is written in Python 3 and strictly utilizes fast, standard numerical libraries.

**Dependencies:**
* `numpy`
* `numba` (Crucial for JIT-compiling the nested momentum/angular cross-section integrals for massive speed-ups).
* `matplotlib` (For automated 2-panel plotting).
* `tqdm` (For progress bar rendering).

~~~bash
pip install numpy numba matplotlib tqdm
~~~

---

## 5. Usage

The script is entirely self-contained. Running it directly will automatically execute the solver using the parameters defined in the JSON configuration file. If the config file does not exist, the script will generate a baseline template automatically.

~~~bash
python dark_portal_solver.py
~~~

### Example Console Output
~~~text
Created default configuration file: ../config/config_dark_portal.json
Initializing solver across 1000 coupled energy bins...
Deploying 12 multiprocessing workers.
Integrating Spectrum: 100%|██████████████████| 1/1 [00:03<00:00,  3.12s/it]

Flux data saved to: /path/to/output/data/output_flux.txt

Integration Complete. Sample Output (Mid-grid Energy):
Energy: 3.16e+05 GeV
Flavor Ratios at Earth [e, mu, tau]: [0.298, 0.351, 0.351]

Plot successfully saved to: /path/to/output/plots/flux_plot.png
~~~

---

## 6. Configuration Guide (`config_dark_portal.json`)

The solver is entirely controlled via the JSON configuration file. Below is an exhaustive dictionary of all configurable flags.

### `[cosmology]`
* `H0_km_s_Mpc` (float): Present-day Hubble constant in km/s/Mpc.
* `Omega_m` (float): Cosmological matter density parameter.
* `Omega_L` (float): Cosmological dark energy density parameter.
* `T_nu0_eV` (float): Present-day temperature of the C$\nu$B in eV.
* `n_n0_cm3` (float): Present-day cosmological background neutron density in cm$^{-3}$.
* `n_n_local_cm3` (float): Local environmental neutron density (e.g., Milky Way) in cm$^{-3}$.
* `m_lightest_eV` (float): Absolute mass of the lightest neutrino eigenstate in eV.
* `mass_ordering` (str): `"normal"` or `"inverted"`. Dictates how absolute masses are derived from $\Delta m^2$.
* `eta_e`, `eta_mu`, `eta_tau` (float): Dimensionless flavor asymmetry multipliers for the C$\nu$B. Default is `1.0` (symmetric).

### `[standard_mixing]`
* `dm21_sq` (float): Solar mass squared difference $\Delta m_{21}^2$ in eV$^2$.
* `dm31_sq` (float): Atmospheric mass squared difference $\Delta m_{31}^2$ in eV$^2$. *(Code automatically handles the negative sign for inverted ordering).*
* `th12_deg`, `th13_deg`, `th23_deg` (float): Standard PMNS mixing angles $\theta_{ij}$ in degrees.
* `d_cp_deg` (float): CP-violating Dirac phase $\delta_{CP}$ in degrees.
* `use_SM_potential` (bool): If `true`, applies standard MSW coherent scattering $V_{SM}$ near/inside the Earth.
* `n_e_earth_cm3` (float): Standard matter electron density in cm$^{-3}$ used for $V_{SM}$.

### `[dark_portal]`
* `u_e`, `u_mu`, `u_tau` (float): Sterile flavor mixing parameters $\vert{}U_{\alpha 4}\vert{}$. Represent the dimensionless magnitude of dark charges.
* `m_Zd_eV` (float): Mass of the dark vector mediator $Z_D$ in eV.
* `g_D` (float): Dark gauge coupling constant $g_D$.
* `epsilon_prime` (float): Effective kinetic mixing parameter $\epsilon'$.
* `use_debye_screening` (bool): If `true`, applies the C$\nu$B thermal Debye mass $m_D^2$ to the denominator of the LRI potential.

### `[source]`
* `p_index` (float): Power-law spectral index ($\gamma$) of the injected neutrino spectrum.
* `Phi_0` (float): Baseline source flux normalization multiplier.
* `z_max` (float): Maximum redshift to integrate back to (start of injection).
* `rho_evol` (float): Cosmological source evolution index (modifies the base SFR).
* `f_e_S`, `f_mu_S`, `f_tau_S` (float): Injection flavor ratios at the source (e.g., `0.333, 0.666, 0.0` for pion decay).
* `is_antineutrino` (str): 
  * `"sum"` or `"both"`: Computes $\nu$ and $\bar{\nu}$ separately and sums them (Default for IceCube).
  * `"true"` or `"antineutrino"`: Computes $\bar{\nu}$ flux only.
  * `"false"`: Computes $\nu$ flux only.

### `[grid]`
* `E_min_eV` (float): Minimum observed energy at Earth in eV.
* `E_max_eV` (float): Maximum observed energy at Earth in eV.
* `n_E_bins` (int): Number of logarithmically spaced energy bins.
* `n_z_bins` (int): Number of linearly spaced redshift bins.

### `[output]` & `[execution]`
* `flux_filename` (str): Target filepath for the output `.txt` data file.
* `plot_filename` (str): Target filepath for the output `.png` plot.
* `n_cores` (int): Number of CPU cores to utilize for multiprocessing.

---

## 7. Authors & Citation

**Author:** Mauricio Bustamante  
If you use this code in academic research, please cite the associated theoretical manuscripts detailing this statistical procedure and propagation framework.