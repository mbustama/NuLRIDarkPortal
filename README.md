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

---

## 3. Repository Structure

```text
.
├── dark_portal_solver.py      # Main solver and integration script
├── config/
│   └── config_dark_portal.json # Autogenerated configuration parameters
└── output/
    ├── data/                  # Exported absolute flux data (.txt)
    └── plots/                 # Generated matplotlib figures (.png)