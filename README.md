# Astrophysical Neutrino Dark Portal Solver

This repository contains a high-performance, Python-based numerical solver for simulating the cosmological propagation of high-energy astrophysical neutrinos under the influence of new dark sector physics. 

The code evaluates a comprehensive framework featuring a light vector mediator ($Z_D$) that couples to active neutrinos via mixing with a sterile state, and to standard matter via kinetic mixing ($\epsilon'$). It explicitly tracks flavor evolution, resonant attenuation against the Cosmic Neutrino Background (C$\nu$B), exact energy cascading (down-scattering), and environmentally dependent long-range interactions (LRI).

---

## Table of Contents
1. [Physics and Mathematics Framework](#1-physics-and-mathematics-framework)
2. [Numerical Implementation & Spatial Handover](#2-numerical-implementation--spatial-handover)
3. [Repository Structure](#3-repository-structure)
4. [Installation & Dependencies](#4-installation--dependencies)
5. [Usage & Order of Precedence](#5-usage--order-of-precedence)
6. [Configuration Guide & Command-Line Flags](#6-configuration-guide--command-line-flags)
7. [Authors & Citation](#7-authors--citation)

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

* **Vacuum Hamiltonian:** $H_{vac} = \frac{1}{2E} U_{PMNS} M^2 U_{PMNS}^\dagger$. The absolute mass scale is dynamically computed using the user-defined mass of the lightest state and the mass ordering hierarchy.
* **SM Matter Potential:** $V_{SM} = \sqrt{2} G_F n_e \text{diag}(1, 0, 0)$, activated near/inside the Earth.
* **LRI Potential:** Sourced by the local and cosmological neutron density $n_n(z)$, mediated by the dark vector $Z_D$ through kinetic mixing $\epsilon'$:

$$V_{LRI, ij} = \left[ \frac{g_D \epsilon'}{m_{Z_D}^2 + m_D^2} \right] \left( \frac{4 c_W m_Z^2}{g_w} \right) \left( \frac{\sqrt{2}}{2} G_F n_n(z) \right) u_i u_j^*$$

**Debye Screening & Flavor Asymmetry:** The LRI potential is dynamically screened by the hot plasma of the C$\nu$B, which generates a Debye mass $m_D$. The solver accounts for theoretical C$\nu$B lepton asymmetries ($\eta_\alpha \equiv n_{\nu_\alpha}/n_{\nu, \text{sym}}$), which orthogonalizes the screening across flavors:
$$m_D^2 = \frac{g_D^2}{6} \sum_i |u_i|^4 T_{\nu, i}^2 \quad \text{where} \quad T_{\nu, i}^2 = T_{\nu, \text{base}}^2 \cdot \eta_i^{2/3}$$

### 1.2 Resonant Attenuation (Absorption)
High-energy neutrinos scatter resonantly off the C$\nu$B via $s$-channel $Z_D$ exchange. The attenuation rate $\Gamma(E)$ is determined by integrating the Breit-Wigner cross-section over the full Fermi-Dirac momentum distribution $f_{FD}(p)$ of the C$\nu$B. To avoid numerical "pole-skipping" when the coupling $g_D$ is extremely small, the inner angular integral over $d\cos\theta$ is evaluated **exactly analytically**, and the outer momentum integral is parallelized using implicit vectorization.

### 1.3 Energy Cascading & Injection
For resonant scattering events ($Z_D \to \nu + \bar{\nu}$), particles cascade to lower energies following the exact lab-frame differential cross-section for the decay of a vector mediator into chiral fermions ($W(E, E') = 3 \frac{E^2}{(E')^3}$). The cascading integral is weighted precisely using exact logarithmic bin boundaries to strictly conserve energy. The code continuously integrates flux injections at every redshift utilizing a broken power-law Star Formation Rate (SFR) combined with an arbitrary source evolution modifier $\rho_{evol}$.

---

## 2. Numerical Implementation & Spatial Handover

### 2.1 Stiff-Stable Semi-Analytic Integration
Because the resonance width $\Gamma_{Z_D} \propto g_D^2$, small couplings generate narrow interaction rates where standard explicit Euler integration causes unphysical numerical explosions. The solver circumvents this by utilizing an exact semi-analytic exponential time step for the coupled ODE:
$$F_{n+1} = F_n e^{-\Gamma \Delta t} + \frac{\text{Total Source}}{\Gamma} \left( 1 - e^{-\Gamma \Delta t} \right)$$

### 2.2 Spatial Handover (Local Universe Transit)
To accurately capture the potentials of gravitationally bound local structures, the solver executes a two-phase integration. Once the cosmological baseline flux arrives at $z=0$, the resulting density matrix is passed into a pure spatial ODE solver utilizing flat-space path-length steps $dr$.

This state-of-the-art spatial handover dynamically resolves specific environmental shaping:
* **The Milky Way (Extended Source):** The Milky Way density transit begins at a 30 kpc edge radius. The specific local neutron density is evaluated iteratively using McMillan (2011) multi-component models (bulge, thin disc, thick disc) and the Miller & Bregman (2013) hot gas halo profile.
* **Milky Way Solid-Angle Averaging:** To account for anisotropic galactic potentials, the solver computes an ensemble average of transits. The user can specify the number of integration paths (`n_mw_paths`). If an arbitrary value is provided, the solver dynamically generates a **Fibonacci Spherical Lattice** (Golden Spiral) to guarantee perfectly isotropic, uniform solid-angle sky coverage.
* **The Sun & Moon (Point Sources):** As the neutrino state transits the Solar System, the Sun and Moon impose exponentially decaying Yukawa potential envelopes $\propto e^{-m_{eff}r}/r$ relative to their precise nucleon mass counts.
* **The Earth (PREM & MSW Effect):** The final transit solves the density matrix through the varying depths of the Earth using the 10-layer polynomials of the Preliminary Reference Earth Model (PREM). This generates an exact mapping of both the LRI and the Standard Model MSW matter potentials. The integration dynamically evaluates over multiple evenly-spaced Earth chords (`n_earth_chords`), sweeping continuously from core-crossing trajectories to grazing trajectories, resolving the flux at a parameterized detector depth (`detector_depth_km`).

---

## 3. Repository Structure

```text
.
├── dark_portal_solver.py      # Main solver, CLI interface, and spatial ODE integration script
├── dark_portal_scanner.py     # Multiprocessing parameter scanner and 2x2 grid visualization tool
├── config/
│   └── config_dark_portal.json # Autogenerated configuration parameters
└── output/
    ├── data/                  # Exported absolute flux data and scan grids (.txt)
    └── plots/                 # Generated matplotlib figures (.png)
```

---

## 4. Installation & Dependencies

The code is written in Python 3 and strictly utilizes fast, standard numerical libraries.

**Dependencies:**
* `numpy`
* `numba` (Crucial for JIT-compiling the nested physics kernels. Features persistent layout caching `cache=True` and static C-signatures)
* `matplotlib` (For automated plotting)
* `tqdm` (For granular progress bar rendering)

```bash
pip install numpy numba matplotlib tqdm
```

---

## 5. Usage & Order of Precedence

### 5.1 Running the Main Solver (`dark_portal_solver.py`)
The main solver script is entirely self-contained. It can be run immediately without prior setup. 

```bash
python dark_portal_solver.py
```

**Note on First Execution:** Because the solver utilizes Numba's Just-In-Time (JIT) compilation with `cache=True` to achieve C-level performance, the script will pause for several seconds during its very first execution. This is perfectly normal; it is compiling and saving the optimized machine code to your hard drive. All subsequent runs will bypass this step and initialize instantly.

#### Parameter Order of Precedence
The solver determines the active physics parameters by cascading through three levels of preference:
1. **Immutable Defaults:** Hardcoded baselines instantiated at runtime.
2. **JSON Configuration File:** Overlays and overrides the baseline parameters if a `config_dark_portal.json` file is detected (automatically generated on the first run).
3. **Command-Line Interface (CLI) Flags:** The highest priority. Terminal arguments will dynamically overwrite both the system defaults and the JSON file settings for that specific execution.

**Example CLI Execution:**
```bash
python dark_portal_solver.py --config my_custom_config.json --n_cores 8 --detector_depth_km 1.5 --g_D 1e-5 --is_antineutrino true
```

#### Output Specifications
Upon completion, the solver generates two primary files (dictated by the `[output]` configuration block):
1. **Data File (`.txt`)**: A columnar text file mapping the exact unscaled flux values $[eV^{-1} cm^{-2} s^{-1} sr^{-1}]$ across the evaluated energy grid for each distinct neutrino flavor ($\nu_e, \nu_\mu, \nu_\tau$).
2. **Flavor Spectrum Plot (`.png`)**: A two-panel figure summarizing the results:
   * **Top Panel:** Displays the absolute diffuse flux scaled by $E_0^2$ to highlight high-energy tail attenuation.
   * **Bottom Panel:** Maps the relative, normalized flavor ratios ($f_e, f_\mu, f_\tau$) as a function of observed energy at the configured detector depth.

### 5.2 Running Parameter Scans (`dark_portal_scanner.py`)
To explore the multidimensional parameter space, the repository includes a dedicated multiprocessing scanner. This script evaluates a grid of parameter values across the dark portal sector, computes the energy-averaged flavor ratios at the detector, and generates a 2x2 multi-panel diagnostic plot.

```bash
python dark_portal_scanner.py
```

**Scanner Features:**
* **Automated 1D Grids:** Conducts high-resolution logarithmic scans across the dark vector mass ($m_{Z_D}$), gauge coupling ($g_D$), and sterile mixing matrix elements ($u_e$, $u_\mu$).
* **Multiprocessing Optimization:** Distributes the grid evaluations across all available CPU cores using `multiprocessing.Pool`, suppressing standard solver verbosity to maintain clean top-level progress bars via `tqdm`.
* **Energy-Averaged Ratios:** Integrates the resulting absolute fluxes across the defined energy bins using exact geometric midpoints to extract the mean expected flavor composition $\langle f_e \rangle$, $\langle f_\mu \rangle$, $\langle f_\tau \rangle$.
* **Visualization:** Exports the raw columnar text data (`scan_m_Zd.txt`, etc.) and automatically renders a `parameter_scan_plot.png` showcasing the macroscopic flavor response. Base configuration parameters are elegantly overlaid as text labels within the individual plot panels.

---

## 6. Configuration Guide & Command-Line Flags

Below is an exhaustive dictionary of all configurable parameters, their respective JSON block locations, and their corresponding CLI flags.

### `[cosmology]`
* `--H0_km_s_Mpc` (float): Present-day Hubble constant in km/s/Mpc (Default: `67.4`).
* `--Omega_m` (float): Cosmological matter density parameter (Default: `0.315`).
* `--Omega_L` (float): Cosmological dark energy density parameter (Default: `0.685`).
* `--T_nu0_eV` (float): Present-day temperature of the C$\nu$B in eV (Default: `1.95e-4`).
* `--n_n0_cm3` (float): Present-day cosmological background neutron density in cm$^{-3}$ (Default: `0.25e-6`).
* `--m_lightest_eV` (float): Absolute mass of the lightest neutrino eigenstate in eV (Default: `0.01`).
* `--mass_ordering` (str): `"normal"` or `"inverted"`. Dictates how absolute masses are derived from $\Delta m^2$ (Default: `"normal"`).
* `--eta_e`, `--eta_mu`, `--eta_tau` (float): Dimensionless flavor asymmetry multipliers for the C$\nu$B (Default: `1.0`).

### `[standard_mixing]`
* `--dm21_sq` (float): Solar mass squared difference $\Delta m_{21}^2$ in eV$^2$ (Default: `7.41e-5`).
* `--dm31_sq` (float): Atmospheric mass squared difference $\Delta m_{31}^2$ in eV$^2$. *(Code automatically handles negative signs for inverted ordering)* (Default: `2.511e-3`).
* `--th12_deg`, `--th13_deg`, `--th23_deg` (float): PMNS mixing angles $\theta_{ij}$ in degrees (Defaults: `33.41`, `8.58`, `49.1`).
* `--d_cp_deg` (float): CP-violating Dirac phase $\delta_{CP}$ in degrees (Default: `132.0`).
* `--use_SM_potential` (str/bool): `"true"` or `"false"`. Applies standard MSW coherent scattering $V_{SM}$ near/inside the Earth (Default: `"false"`).
* `--n_e_earth_cm3` (float): Standard matter electron density in cm$^{-3}$ used for $V_{SM}$ (Default: `1.5e24`).
* `--detector_depth_km` (float): Subterranean depth of the simulated observatory in km (e.g., 1.5 for IceCube) (Default: `1.5`).

### `[dark_portal]`
* `--u_e`, `--u_mu`, `--u_tau` (float): Sterile flavor mixing parameters $|U_{\alpha 4}|$ (Defaults: `0.5`, `0.5`, `0.0`). 
* `--m_Zd_eV` (float): Mass of the dark vector mediator $Z_D$ in eV (Default: `2.0e6`).
* `--g_D` (float): Dark gauge coupling constant $g_D$ (Default: `1e-4`).
* `--epsilon_prime` (float): Effective kinetic mixing parameter $\epsilon'$ (Default: `1e-1`).
* `--use_debye_screening` (str/bool): `"true"` or `"false"`. Applies the thermal Debye mass $m_D^2$ to the LRI potential denominator (Default: `"true"`).

### `[source]`
* `--p_index` (float): Power-law spectral index ($\gamma$) of the injected neutrino spectrum (Default: `2.5`).
* `--Phi_0` (float): Baseline source flux normalization multiplier (Default: `1.0`).
* `--z_max` (float): Maximum redshift to integrate back to (Default: `4.0`).
* `--rho_evol` (float): Cosmological source evolution index (modifies the base SFR) (Default: `0.0`).
* `--f_e_S`, `--f_mu_S`, `--f_tau_S` (float): Injection flavor ratios at the source (e.g., `0.333, 0.666, 0.0` for pion decay) (Defaults: `0.333333`, `0.666667`, `0.0`).
* `--is_antineutrino` (str): (Default: `"sum"`)
  * `"sum"` or `"both"`: Computes $\nu$ and $\bar{\nu}$ independently and sums the flux.
  * `"true"` or `"antineutrino"`: Computes $\bar{\nu}$ flux only.
  * `"false"`: Computes $\nu$ flux only.

### `[grid]`
* `--E_min_eV` (float): Minimum observed energy at Earth in eV (Default: `1e13`).
* `--E_max_eV` (float): Maximum observed energy at Earth in eV (Default: `3e16`).
* `--n_E_bins` (int): Number of exact logarithmically spaced energy bins (Default: `1000`).
* `--n_z_bins` (int): Number of linearly spaced redshift bins (Default: `1000`).
* `--n_mw_paths` (int): Total Milky Way spatial integration trajectories evaluated for the solid-angle sky average (Default: `6`).
* `--n_earth_chords` (int): Total PREM-layer transits averaged across the Earth's radius (Default: `3`).
* `--N_steps_MW` (int): Number of spatial integration steps evaluated during the Milky Way transit (Default: `100`).
* `--r_start_kpc` (float): Starting distance from Earth for the Milky Way spatial integration in kpc (Default: `100.0`).

### `[output]` & `[execution]`
* `--flux_filename` (str): Target filepath for the output `.txt` data file (Default: `"../output/data/output_flux.txt"`).
* `--plot_filename` (str): Target filepath for the output `.png` plot (Default: `"../output/plots/flux_plot.png"`).
* `--n_cores` (int): Number of CPU cores to utilize for parameter-scan multiprocessing (Default: Total available system cores minus 1). (Note: The solver dynamically sets this to `1` and reveals detailed progress bars if only a single configuration is evaluated).
* `--verbose` (int): Sets the logging detail level (`1` for standard, `2` for detailed granular milestone reporting) (Default: `1`).

---

## 7. Authors & Citation

**Author:** Mauricio Bustamante  
If you use this code in academic research, please cite the associated theoretical manuscripts detailing this statistical procedure and propagation framework.