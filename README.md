# Astrophysical Neutrino Dark Portal Solver

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

This repository contains a high-performance, Python-based numerical solver for simulating the cosmological propagation of high-energy astrophysical neutrinos (E ≳ 10⁴ GeV) under the influence of new dark-sector physics, together with a profile-likelihood analysis package (`analysis/`) for confronting the predictions with detector flavor-ratio measurements.

The code evaluates a comprehensive framework featuring a light vector mediator ($Z_D$) that couples to active neutrinos via mixing with a sterile state, and to standard matter via kinetic mixing ($\epsilon'$). It explicitly tracks flavor evolution, resonant attenuation against the Cosmic Neutrino Background (C$\nu$B), exact energy cascading (down-scattering), and environmentally dependent long-range interactions (LRI).

---

## Table of Contents
1. [Quick Start](#1-quick-start)
2. [Physics and Mathematics Framework](#2-physics-and-mathematics-framework)
3. [Numerical Implementation & Spatial Handover](#3-numerical-implementation--spatial-handover)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Dependencies](#5-installation--dependencies)
6. [The Solver & Scanner](#6-the-solver--scanner)
7. [Configuration Guide & Command-Line Flags](#7-configuration-guide--command-line-flags)
8. [Custom Potential Structure Matrices](#8-custom-potential-structure-matrices)
9. [The Statistical Analysis (`analysis/`)](#9-the-statistical-analysis-analysis)
10. [Exclusion Planes](#10-exclusion-planes)
11. [Choosing Resolution Parameters (Convergence Tool)](#11-choosing-resolution-parameters-convergence-tool)
12. [Performance Notes](#12-performance-notes)
13. [Testing](#13-testing)
14. [Authors & Citation](#14-authors--citation)

---

## 1. Quick Start

```bash
# Install dependencies
pip install numpy scipy numba matplotlib tqdm pytest

# 1) Compute one flux spectrum at the default dark-portal parameters
#    (first run pauses ~10 s while numba JIT-compiles and caches the kernels)
cd src
python3 dark_portal_solver.py --n_cores 8

# 2) Same, overriding physics parameters from the command line
python3 dark_portal_solver.py --m_Zd_eV 2e6 --g_D 1e-4 --u_e 0.5 --u_mu 0.5 --n_cores 8

# 3) 1D parameter scans (m_Zd, g_D, u_e, u_mu) with a 2x2 summary figure
python3 dark_portal_scanner.py

# 4) Profile-likelihood analysis: global fit, profiles, planes (Section 9)
cd ..
python3 -m analysis analysis/configs/pion_2050_case_5_exclusion.json --workers 8

# 5) The same against Asimov data (the model's own prediction at the config's 'asimov' values)
python3 -m analysis analysis/configs/pion_2050_case_5.json --asimov --fit

# 6) Find the coarsest safe resolution settings for YOUR parameter region
cd tools
python3 convergence_study.py --axes n_z_bins n_E_bins --n_cores 8

# 7) Run the regression suite and the analysis tests (do this after any physics edit)
cd ..
python3 -m pytest tests/ -v
```

---

## 2. Physics and Mathematics Framework

The solver evaluates the continuous integro-differential equation dictating the cosmological transport of the astrophysical neutrino density matrix, propagating it backwards in redshift from $z_{max}$ to $z=0$.

The total flavor flux $F(E)$ evolves according to:
$$\frac{dF_i(E, z)}{dt} = -\Gamma_i(E, z) F_i(E, z) + \int_E^{E_{max}} dE' W(E, E') \sum_\alpha \left[ \Gamma_\alpha(E', z) F_\alpha(E', z) \right] + Q_{i}(E, z)$$

Where:
* $t$ is the comoving cosmological time tied to the Hubble expansion $dt = \frac{dz}{H(z)(1+z)}$.
* $\Gamma_i$ is the resonant absorption rate.
* $W(E, E')$ is the inelastic cascading kernel.
* $Q_i$ is the continuous source injection term.

### 2.1 Adiabatic Flavor Mixing and Long-Range Interactions (LRI)
Neutrinos propagate as instantaneous mass eigenstates of an effective Hamiltonian that incorporates vacuum mixing, Standard Model (SM) matter effects, and a new macroscopic Long-Range Interaction (LRI) potential sourced by ambient cosmological neutrons.

$$H_{tot} = H_{vac} + V_{SM} + V_{LRI}$$

* **Vacuum Hamiltonian:** $H_{vac} = \frac{1}{2E} U_{PMNS} M^2 U_{PMNS}^\dagger$. The absolute mass scale is dynamically computed using the user-defined mass of the lightest state and the mass ordering hierarchy.
* **SM Matter Potential:** $V_{SM} = \sqrt{2} G_F n_e \, S_{SM}$, activated near/inside the Earth, with the flavor texture $S_{SM} = \mathrm{diag}(1, 0, 0)$ by default (see [Section 8](#8-custom-potential-structure-matrices) for overrides). **Off by default** (`use_SM_potential: false`): this code targets neutrinos above $10^4$ GeV, where standard MSW matter effects are negligible — they shape the oscillations of GeV-scale neutrinos.
* **LRI Potential:** Sourced by the local and cosmological neutron density $n_n(z)$, mediated by the dark vector $Z_D$ through kinetic mixing $\epsilon'$:

$$V_{LRI, ij} = \left[ \frac{g_D \epsilon'}{m_{Z_D}^2 + m_D^2} \right] \left( \frac{4 c_W m_Z^2}{g_w} \right) \left( \frac{\sqrt{2}}{2} G_F n_n(z) \right) S_{LRI, ij}$$

with the flavor texture $S_{LRI, ij} = u_i u_j^\ast$ by default (also overridable, [Section 8](#8-custom-potential-structure-matrices)). Antineutrinos experience $V \to -V^\ast$; the code applies this transform in a single, centralized place (`propagate_spectrum`) for both potentials.

**Debye Screening & Flavor Asymmetry:** The LRI potential is dynamically screened by the hot plasma of the C$\nu$B, which generates a Debye mass $m_D$. The solver accounts for theoretical C$\nu$B lepton asymmetries ($\eta_\alpha \equiv n_{\nu_\alpha}/n_{\nu, \text{sym}}$), which orthogonalizes the screening across flavors:
$$m_D^2 = \frac{g_D^2}{6} \sum_i |u_i|^4 T_{\nu, i}^2 \quad \text{where} \quad T_{\nu, i}^2 = T_{\nu, \text{base}}^2 \cdot \eta_i^{2/3}$$

### 2.2 Resonant Attenuation (Absorption)
High-energy neutrinos scatter resonantly off the C$\nu$B via $s$-channel $Z_D$ exchange. The attenuation rate $\Gamma(E)$ is determined by integrating the Breit-Wigner cross-section over the full Fermi-Dirac momentum distribution $f_{FD}(p)$ of the C$\nu$B. To avoid numerical "pole-skipping" when the coupling $g_D$ is extremely small, the inner angular integral over $d\cos\theta$ is evaluated **exactly analytically**. The outer momentum integral is, by default, also **semi-analytic** (`thermal_averaged_rates_flavor_analytic`): both $u_\pm(p) = A - M^2 \mp B$ are strictly monotone, so the resonance crosses the angular range at most once, at a *closed-form* momentum $p_{edge} = |m^2 - \delta^2|/(2\delta)$ with $\delta = (M^2 - m^2)/(2x)$; the integral is decomposed into geometric shells $p_{edge} \pm \{3,30,\dots,3\times10^4\}\cdot MG/|u'|$ (16-point Gauss–Legendre per panel on the exact pointwise $I_{ang}$) with exact polynomial×arctan / polynomial×log moment antiderivatives across the narrow transition core, where the node is pinned at $p_{edge}$ with $u = 0$ exactly — so the resonance-step position is exact even when $MG$ is below the floating-point noise of $u$, and the split remains exact at *any* width (the wide-width $g_D \sim 1$ regime is just smooth panels). Against a dense structure-aware referee quadrature, the semi-analytic kernel is accurate to $\sim 2.5\times10^{-7}$ (significance-weighted) while the historical 100-point midpoint rule — still available via `grid.thermal_analytic = false` as the reference implementation — carries **6–9% error at on-resonance cells** (its $p_{max}/100$ granularity cannot resolve the window edges). It is also 2–5.6× faster. Node-count tuning (referee-verified to the *same* worst case, $2.524\times10^{-7}$): the bulk single-panel cells use a precomputed Fermi–Dirac table at their fixed scaled nodes, and sub-$p_{max}/5$ shell panels use a 10-point rule — a further 10–25% off the grid build.

**Normalization (corrected 2026-10-12).** The rates use the Breit–Wigner cross section of a vector resonance formed by $\nu_L + \bar\nu_R$, summed over final states,
$$\sigma_\alpha = \frac{48\pi\,\Gamma_{in,\alpha}\Gamma\, s}{M^2[(s-M^2)^2 + M^2\Gamma^2]} = \frac{(\Sigma G)_\alpha(\Sigma G)\, s}{3\pi[(s-M^2)^2 + M^2\Gamma^2]}, \qquad \Gamma = \frac{(\Sigma G)\,M}{12\pi},$$
whose integral over $s$ is $4\pi(\Sigma G)_\alpha$ (for the $Z$ it gives the classic $\langle\sigma\rangle = 2\sqrt{2}\pi G_F$). The kernels integrate the raw form $2\pi(\Sigma G)_\alpha s/[(s-M^2)^2+M^2\Gamma^2]$, and the grids leave `dp_thermal` multiplied by $\Sigma G/(6\pi^2)$ (`absorption_normalization`). Until 2026-10-12 the raw form was the rate: with one coupling factor in the numerator, the resonance's integrated strength, $12\pi^2(\Sigma G)_\alpha/\Sigma G$, did not depend on the size of the couplings (the integrated rate was identical from $\Sigma G = 7.6\times10^{-4}$ down to $3\times10^{-19}$), so any nonzero $u$, however small, gave the full absorption and only $u = 0$ exactly was standard. Tests pin the integrated rate's proportionality to $\Sigma G$, the narrow-width limit against independent quadrature, and $u = 10^{-10}$ reproducing the $u = 0$ flux to $10^{-8}$.

On top of the kernel sits the **two-component coupling cache** (`thermal_grid_cached`): the raw unit-weight grid $U(s) = \Gamma_{raw}/\text{sum}_{G\alpha}$ (the normalization rides on the weights) obeys $U(s) \approx I_A + I_B/s$ (linear cells + narrow-width $1/MG$ plateau) to $\sim 7\times10^{-8}$ over ≤2-decade brackets of $s = \text{sum}_G$. The cache activates the surrogate **lazily and only after verifying it** against an exact grid built at an interior bracket point (gate $3\times10^{-7}$, the kernel's own accuracy class), never extrapolates outside the verified bracket, and disables itself on repeated gate failures *in the wide-bracket regime where failures indicate genuine model breakdown*. Brackets are accepted down to a purely numerical floor (~$10^{-4}$ dex): the surrogate is an interpolation between exact endpoints, so narrower only means more accurate — this is what lets it fire inside profiled inner fits, whose per-step coupling moves (~0.01–0.1 dex) never spanned the original 0.3-dex floor (measured: the cache **never activated** during real scans before this fix). Exact repeats of an already-built $s$ are additionally served straight from the stored exact point. Measured on minimizer-like inner-fit traces at fixed $m_{Z_D}$ (in-band/below/strong classes, 200×200): **83% of evaluations hit** with worst cached-grid error $6.5\times10^{-9}$ vs fresh exact builds (~50× inside the gate); repeated-point evaluations drop the thermal cost from ~105 ms to 0.3 ms. $m_{Z_D}$-scan points pay exactly one exact build, as before. The reference paths bypass the cache entirely. The cosmological absorption resonance sits at $E_{res} \sim m_{Z_D}^2 / (2 m_\nu)$ and sweeps through the observed energy band as $m_{Z_D}$ varies — this drives the resolution requirements discussed in [Section 11](#11-choosing-resolution-parameters-convergence-tool).

### 2.3 Energy Cascading & Injection
Every resonant absorption $\nu(E') + \bar\nu_{\rm relic} \to Z_D \to \nu + \bar\nu$ emits two particles. The $Z_D$ is polarized along the incoming neutrino, so the outgoing particle of the absorbed kind follows $W(E, E') = 3E^2/E'^3$ (on average 3/4 of the energy) and the other kind $3(E'-E)^2/E'^3$ (the remaining 1/4); an absorbed antineutrino gives the mirror image. Neutrinos and antineutrinos are therefore evolved together through the cosmological phase (`step_cosmological_pair`); both kernels are integrated exactly over the logarithmic bins (`cascade_kernel_weights`, `cross_kernel_weights`). Until 2026-10-13 only the secondary of the absorbed kind was kept and the two kinds were propagated independently, which dropped half of the regenerated particles and a quarter of the absorbed energy; until 2026-10-12 the midpoint weights also re-emitted $1 + 1.5\,\Delta\ln E$ particles per absorption. The code continuously integrates flux injections at every redshift utilizing a broken power-law Star Formation Rate (SFR) combined with an arbitrary source evolution modifier $\rho_{evol}$.

---

## 3. Numerical Implementation & Spatial Handover

### 3.1 Stiff-Stable Semi-Analytic Integration
Near resonance the optical depth of one redshift step, $\Gamma\Delta t$, reaches $10^{10}$ (above 1 in 70% of the $(z, E)$ cells at `inband`), so every step must be exact at any optical depth. Each step sweeps the energy bins from high to low. A bin solves $dF/dt = -\Gamma F + S$ exactly over the step,
$$F_{n+1} = F_n e^{-\Gamma \Delta t} + \frac{S}{\Gamma}\left(1 - e^{-\Gamma \Delta t}\right),$$
where $S$ is the injection plus the regeneration of both kinds from the higher bins, built from the number they actually absorbed during the step, $F_n(1 - e^{-\Gamma\Delta t}) + S\,[\Delta t - (1 - e^{-\Gamma\Delta t})/\Gamma]$; a bin's own secondaries of both kinds make it a 2×2 linear system, solved directly. The step balances number to rounding for $\Gamma\Delta t$ from $10^{-8}$ to $10^{10}$ (every absorption adds one particle, less the secondaries below the grid), and the solver never returns more energy than was injected (both are tests). **Until 2026-10-12** the regenerated source was each bin's absorption rate $\Gamma F_n$ at the start of the step, held over the whole step: each step re-emitted up to $\Gamma\Delta t$ times more neutrinos than were absorbed, the overshoot compounded, and the energy flux at Earth came out up to $10^{67}$ times the injected one at the benchmark points, growing with finer redshift steps. The flavor fractions, dominated by the runaway secondaries, still looked plausible, and the validations then in use compared runs with each other rather than with conservation, so the error went unnoticed from the first commit until 2026-10-12.

### 3.2 Spatial Handover (Local Universe Transit)
To accurately capture the potentials of gravitationally bound local structures, the solver executes a two-phase integration. Once the cosmological baseline flux arrives at $z=0$, the resulting density matrix is passed into a pure spatial ODE solver utilizing flat-space path-length steps $dr$.

* **The Milky Way (Extended Source):** The local neutron density is evaluated using McMillan (2011) multi-component models (bulge, thin disc, thick disc) and the Miller & Bregman (2013) hot gas halo profile. Stellar mass is converted at 0.20 neutrons per nucleon (visible stars, white dwarfs and brown dwarfs in the local proportions of McKee, Parravano & Hollenbach 2015) and the halo's electron density at 0.15 neutrons per electron; until 2026-10-12 both used 0.5, appropriate for rock but not for hydrogen-dominated matter. The cold HI and H$_2$ discs of McMillan (2017), whose surface densities include helium and metals, add 0.142 neutrons per nucleon of gas (absent before 2026-10-12).
* **Milky Way Solid-Angle Averaging:** The solver computes an ensemble average over `n_mw_paths` trajectories. For arbitrary path counts it generates a **Fibonacci Spherical Lattice** (Golden Spiral) guaranteeing isotropic solid-angle coverage.
* **The Sun & Moon (Point Sources):** Exponentially decaying Yukawa potential envelopes $\propto e^{-m_{eff}r}/r$ tied to their neutron numbers: Sun $1.929\times10^{56}$ (0.161 neutrons per nucleon for its present bulk composition), Moon $2.232\times10^{49}$ (0.505, silicate rock with a small iron core), Earth $1.849\times10^{51}$ (PREM integrated with the layer fractions below). Until 2026-10-12 the Sun and the Moon used 0.248 neutrons per nucleon and the Earth 0.496.
* **The Earth (PREM & MSW Effect):** The final transit solves the density matrix through the 10-layer polynomials of the Preliminary Reference Earth Model (PREM), with $1-Y_e$ neutrons per nucleon ($Y_e = 0.4656$ in the core, $0.4957$ in the mantle and crust, water for the top layer; 0.5 everywhere before 2026-10-12), averaged over the up-going hemisphere (uniform in $\cos\theta_z$) from `n_earth_chords` evenly-spaced chords between core-crossing and grazing with the phase-aware zenith average of Section 3.5, resolved at a parameterized detector depth (`detector_depth_km`). The chords run exactly from the surface to the detector (until 2026-10-12 `get_baseline_earth` dropped the $d^2$ terms of the chord length, so chords started up to 8 m above the ground).
* **The Earth's own potential (Yukawa integral):** Inside the Earth the dark-portal potential is the Yukawa integral of PREM's neutrons over the mediator's range $1/m_{eff}$, $m_{eff} = \sqrt{m_{Z_D}^2 + m_D^2}$. For a spherical profile it is one-dimensional, $n_{eff}(r) = m^2[e^{-mr} I_{in}(r)/r + \mathrm{shc}(mr)\,I_{out}(r)]$ with $I_{in} = \int_0^r r'^2 n\,\mathrm{shc}(mr')dr'$, $I_{out} = \int_r^R r' n\,e^{-mr'}dr'$ and $\mathrm{shc}(x) = \sinh x/x$ — the method of Magnus (LRI in the Sun) and NuOscProbExact (notebook 20, LRI in the Earth) — and the potential is $V = \mathrm{Coef}\cdot n_{eff}$ (`earth_yukawa_table`, `earth_neutron_density_cm3` in `src/dp_spatial.py`). It is evaluated exactly for PREM's polynomials at any mass, in an overflow-free form: tabulated integrals for $m R_\oplus < 50$, integration by parts above, where the result is the local density plus boundary layers of thickness $1/m$ at every layer jump; the Earth steps integrate those layers explicitly, so the `avg` scheme stays step-count-exact. Until 2026-10-12 the code used the local (contact) limit $n_{eff} = n(r)$, which is exact while $1/m$ is far below PREM's layer thicknesses — at the benchmark points the new form reproduces it bit for bit — but overstates the Earth's potential by $\sim 1/(m R_\oplus)^2$ once the range rivals the Earth ($10^{13}$ at $10^{-20}$ eV). Outside, the integral is exactly a point source of $N_{eff} = 4\pi\int r^2 n\,\mathrm{shc}(mr)\,dr$ neutrons, the Earth's neutron number wherever the Milky Way phase can reach it.

### 3.3 The Spatial Propagator
Each path-length step applies $\rho \to U \rho U^\dagger$ with $U = \exp(-i H \, dr)$ per energy bin. $U$ is the spectral sum $\sum_k e^{-i\lambda_k dr} v_k v_k^\dagger$ over an eigenbasis that `propagator_3x3` builds to stay orthonormal to rounding at any eigenvalue gap, so $U$ is **unitary to $\sim 10^{-15}$ at any phase**: closed-form trigonometric roots of the traceless $D = H - \tfrac{1}{3}\mathrm{tr}(H)$ (with the rounding residue of the trace moved back into the global phase) give the closest pair and the loner; the loner's eigenvector is the largest column of $(D - aI)(D - bI)$ for the pair's roots $a, b$, accurate to $\varepsilon$ at any gap even though the cubic formula's roots of a near-double pair are $\sqrt{\varepsilon}$-conditioned (measured: a close-pair gap 257× too large on a production Earth Hamiltonian); the complement is completed by Gram–Schmidt and a conjugated cross product; and the pair's eigenvalues and eigenvectors come from the cancellation-free 2×2 block of $D$ there. This matters because the rank-1 LRI texture makes the Earth-transit Hamiltonian *doubly-degenerate by construction* wherever the potential rivals the vacuum term. A pair splitting below $16\,\varepsilon(|q| + p)$ is not resolved by $H$ in double precision (rounding when $H$ is formed already splits an exact degeneracy by up to $3.1\,\varepsilon(|q|+p)$) and is propagated as degenerate: under a potential $10^{18}$ times the vacuum term, evolving the rounding-level splitting would rotate the two states orthogonal to the potential by tens of radians per Earth step instead of the true $10^{-5}$. Spreads below $1.2\times10^{-4}/dr$ use the second-order Taylor series (truncation $< 10^{-13}$). $H$ is read as the Hermitian matrix of its upper triangle (the rotated mass matrix $U M U^\dagger$ is Hermitian only to rounding). The function compiles without `fastmath` (its approximate-function flag degrades large-argument transcendentals). Validation: unitarity $< 2\times10^{-15}$ over 12,880 production textures (potentials $10^{-24}$–$10^{22}$ eV, steps to $10^{33}$ eV$^{-1}$, single-flavor mixing) and 20,000 random spectra; agreement with a 50-digit matrix exponential at the phases' rounding level; the spatial handover conserves every energy bin's total to $10^{-14}$.

Until 2026-10-12 (v3) $U$ was a polynomial in $D$ through Newton divided differences of $e^{-i\lambda t}$ at the computed eigenvalues. A polynomial evaluated at rounded nodes is unitary only to $\sim \varepsilon\,\mathrm{spread}\,\min(dr, 2/\mathrm{gap})$ — $5\times10^{-3}$ per Earth step at potentials of 70 eV (the $m_{Z_D} = 40$ eV benchmark), where the spatial phase created or destroyed up to 0.7% of the flux, and, before a guard for that extreme, $\sim 10^{11}$ at $m_{Z_D} \sim 10^{-6}$ eV, where 500 steps overflowed $\rho$ to NaN (the historical *all-NaN flux*); v3 also divided by zero for single-flavor sterile mixing ($u_\mu = 0$ or $u_e = 0$) above $\sim 10^{7}$ eV. At extreme mediator masses ($m_{Z_D} \lesssim$ a few eV; shipped configs allow down to $10^{-30}$) the $1/m_{Z_D}^2$-enhanced Earth potential reaches $10^{16{-}22}$ eV and a single Earth step accrues more phase than double precision can track ($\varepsilon\cdot\mathrm{spread}\cdot dr > 0.1$ rad; the growth stops once the range $1/m_{eff}$ exceeds the Earth's radius, where the Yukawa integral saturates at the enclosed-charge value — but at the default couplings the potential is still far beyond phase tracking there): fluxes are exactly unitary but **decoherence-averaged only** (a one-line `[WARNING]` is printed at `verbose ≥ 1`). As a safety net, `propagate_spectrum` raises a loud `ValueError` naming the offending $m_{Z_D}, g_D, \epsilon'$ if a non-finite thermal-rate grid or final flux ever appears, instead of returning NaN silently. The cosmological mixing runs through a **σ-table**: $|U|^2$ at any $(E, z)$ depends only on $\sigma = 2E(1+z)A_{LRI}(z)$ (exact scale invariance), tabulated once per call on 4096 log-σ nodes.

### 3.4 A Cautionary Note on Earth-Transit Step Counts
`N_steps_earth` (default 25 under the phase-exact `earth_step_scheme = "avg"`; the historical `left` scheme used 500) is deliberately **uniform across chords regardless of chord length**. An earlier optimization that gave shorter chords proportionally fewer steps validated perfectly against a synthetic diagonal test state, yet introduced a silent, energy-localized ~10% error with the real coherence-carrying state near an Earth matter resonance — near a level crossing, eigenvectors rotate rapidly with *position*, independent of total path length. A regression test locks in the uniform behavior. Under the default `avg` scheme the step *count* now carries a genuine convergence certificate (N = 8…1000 agree to ≤5×10⁻⁹ at the standard validation points — see Section 12), so the historical step-count anxiety applies only to the legacy `left` scheme; the uniform-across-chords rule holds for both. Validate any resolution change with the convergence tool ([Section 11](#11-choosing-resolution-parameters-convergence-tool)) at *your* parameter points.

---

### 3.5 The Zenith Average of the Earth Chords
The flux at the detector is the average over the up-going hemisphere. Inside the Earth the dark-portal potential exceeds the vacuum term whenever its phase matters (a phase of 1 rad across the Earth needs $V \sim 10^{-14}$ eV, against $\Delta m^2/2E \lesssim 10^{-16}$ eV above 10 TeV), so a chord evolves as $U(c) = \sum_g e^{-i\lambda_g \Psi(c)} P_g$, with $P_g$ the eigenprojectors of the LRI texture and $\Psi(c) = \mathrm{Coef}\int n\,dl$ the chord's column of potential. Each block $P_g\rho P_h$ of the evolved density matrix therefore carries the phase $e^{-i(\lambda_g-\lambda_h)\Psi(c)}$ times a slowly varying amplitude. At MeV masses $\Psi$ reaches $10^5$–$10^8$ rad: the phases are effectively random from chord to chord, and the historical equal-weight mean over chords kept the cross blocks as **sampling noise — the chord-sampling floor**, ~$10^{-2}$ in energy-averaged flavor fractions even at 50 chords (the "fringe aliasing" of Section 11, and the reason chord-count changes never converged). The default `earth_zenith_average = "phase"` integrates each block against its known phase over $\cos\theta_z$ (`average_earth_chords`, a Filon-type rule): the amplitudes are interpolated linearly between chords and the oscillatory factor is integrated exactly on a fine grid of $\Psi(c)$ (4096 chords' columns, graded toward the layer-tangency points, computed once per process in ~10 ms), giving energy-independent complex weights per chord. For small phases these are the trapezoid weights; for large ones the average becomes the decoherent $\sum_g P_g\rho P_g$ that the continuum of zeniths produces; in between they resolve the oscillation a chord sum aliases. Against the exact zenith integral of synthetic chords it is accurate to $\lesssim10^{-5}$ for Earth phases from 0.1 to $10^8$ rad, where the equal-weight mean errs by $10^{-3}$–$10^{-1}$. At production settings, 12, 13 and 25 chords agree with 50 to $\le1.4\times10^{-7}$ in the energy-averaged fractions at below/inband/strong (equal-weight mean: $5.6\times10^{-4}$–$4.1\times10^{-3}$ between 12 and 13 chords), and the step-count certificate tightens to $\le1.2\times10^{-10}$ in flux. With $u_e = u_\mu$ the decoherent average makes $f_e$ and $f_\mu$ equal to $\sim10^{-5}$ at MeV masses; the equal-weight mean had them apart by up to 0.06. In the phase-noise regime (tens of eV) a residual of $\sim6\times10^{-4}$ remains: the potential (~70 eV in the core) is so large that the propagator's eigenvalues, precise to ~$10^{-14}$ eV, cannot resolve the vacuum splitting inside the potential's degenerate subspace. The rule needs the phase-exact `avg` steps; `earth_step_scheme = "left"` implies the historical `"plain"` mean. Phases other than the LRI one (the SM potential when switched on, $\lesssim12$ rad) are interpolated between chords, so use $\ge50$ chords with `use_SM_potential`.

## 4. Repository Structure

```text
.
├── src/
│   ├── dark_portal_solver.py      # PUBLIC FACADE: re-exports everything + orchestration
│   │                              #   (propagate_spectrum, run_solver, CLI, I/O, plotting)
│   ├── dp_core.py                 # Env guards (BLAS pins, TBB), constants, shared utilities
│   ├── dp_propagator.py           # exactly unitary 3x3 propagator (deflated eigenbasis)
│   ├── dp_thermal.py              # Thermal rates: semi-analytic kernel + coupling cache
│   ├── dp_cosmo.py                # Cosmological phase: z-stepper, sigma table, cascade
│   ├── dp_spatial.py              # MW/Earth transit, step schemes, workers, persistent pool
│   ├── dp_config.py               # DEFAULT_CONFIG, unknown-key-rejecting merge, parsing
│   ├── dark_portal_scanner.py     # Multiprocessing 1D parameter scanner + 2x2 grid figure
│   └── flux_postprocessing.py     # Energy-averaged flavor ratios (shared by scanner & fits)
├── analysis/                      # Profile-likelihood analysis (Section 9; docs/dev/ANALYSIS_DESIGN.md)
│   ├── configs/                   # One JSON per analysis
│   ├── data/flavor_surfaces.json  # The 35 measured/projected flavor-composition likelihoods
│   ├── model.py                   # Parameters -> solver -> energy-averaged flavor fractions
│   ├── surfaces.py, parameters.py, objective.py, fitting.py
│   ├── scans.py                   # Global fit, profiles, planes: spawn pool, repair, checkpoints
│   ├── intervals.py, plots.py
│   └── run.py                     # Command line: python -m analysis CONFIG.json
├── tools/
│   └── convergence_study.py       # Resolution-parameter convergence studies
├── tests/
│   ├── test_physics_regression.py # Regression suite (run before & after every change)
│   └── test_analysis.py           # The analysis package
├── config/                        # Solver configuration (autogenerated on first run)
└── output/                        # Flux data, plots, convergence-study reports
```

---

## 5. Installation & Dependencies

The code is written in Python 3 and strictly utilizes fast, standard numerical libraries.

**Core dependencies:** `numpy`, `scipy`, `numba` (JIT compilation of all physics kernels, with persistent on-disk caching), `matplotlib`, `tqdm`. The analysis package needs nothing more (its minimizers are scipy's).
**Tests:** `pytest`.

```bash
pip install numpy scipy numba matplotlib tqdm pytest
```

---

## 6. The Solver & Scanner

### 6.1 Running the Main Solver (`src/dark_portal_solver.py`)

```bash
cd src && python3 dark_portal_solver.py
```

**Note on First Execution:** numba JIT-compiles the kernels with `cache=True` on first run (a pause of ~10 s); subsequent runs load the cached machine code instantly. The cache lives in `src/__pycache__/numba/<digest>`, keyed by a hash of every file in `src/`, so any edit there starts a fresh cache: numba alone validates a cached function against its own file only, and a kernel calling a jitted function from another module (the spatial steps call `propagator_3x3`) would keep the callee's old machine code after an edit — as happened on 2026-10-12. Caches of other versions unused for a day are removed; a `NUMBA_CACHE_DIR` you set yourself is used as is (and then this protection is yours to manage).

#### Parameter Order of Precedence
1. **Immutable defaults** hardcoded at runtime.
2. **JSON configuration file** (autogenerated at `config/config_dark_portal.json` on first run) overrides the defaults.
3. **CLI flags** (highest priority) override both.

```bash
python3 dark_portal_solver.py --config my_config.json --n_cores 8 --g_D 1e-5 --is_antineutrino true
```

#### Outputs
1. **Data file (`.txt`)**: columnar flux $[\mathrm{eV}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}]$ per flavor over the energy grid.
2. **Flavor spectrum plot (`.png`)**: $E^2\Phi$ spectra (top) and flavor ratios vs energy (bottom).

### 6.2 Running Parameter Scans (`src/dark_portal_scanner.py`)

```bash
cd src && python3 dark_portal_scanner.py
```

Runs logarithmic 1D scans over $m_{Z_D}$, $g_D$, $u_e$, $u_\mu$ (plus `epsilon_prime` and `u_tau` via `--params`; those two scan on top of the base config without the cross-mixing zeroing the classic panels use), distributing configurations across a `multiprocessing.Pool` sized by the config's `execution.n_cores`, computes energy-averaged flavor ratios per point (via `flux_postprocessing.compute_energy_averaged_ratios`, which shares the solver's exact geometric bin widths), exports columnar scan data alongside a per-scan reproducibility manifest (`scan_<param>.manifest.json`: config, grid, git commit, runtime), and renders a 2×2 summary figure.

Every completed point is checkpointed to `<output>.partial.jsonl` as it arrives: a killed scan resumes from what it finished (points matched bit-exactly to the requested grid; any config change invalidates the file), and the checkpoint is removed once the final sorted output is written.

The scanner's spatial-averaging defaults follow the convergence study (Section 11): `--n_mw_paths 12` and `--n_earth_chords 50` (the sub-percent knee), replacing the historical 100/100 — roughly a 2–3× speed-up per scan point at no measured accuracy cost at the studied points. **Caveat:** the study's `n_mw_paths` conclusion was drawn at the default parameter points; at very small $m_{Z_D}$ (unscreened, longer-range galactic potential) the solid-angle average becomes anisotropic — raise `--n_mw_paths` there and re-check with `tools/convergence_study.py`.

**SOLVED on `dev-earth` — the Earth step axis (`earth_step_scheme = "avg"`)**: the fringe-aliasing problem below was traced to the *discretization*, not the physics: the historical stepper samples V at each step's left endpoint, so the accumulated LRI phase misses the true $\int V\,dl$ by a Riemann-sum error of thousands of radians that re-aliases the fringes on every grid change. The `avg` scheme (now the default) uses the **exact path-averaged potential per step** — layer-split Gauss–Legendre integration of PREM, with closed-form layer-crossing positions — so the summed phase equals $\int V\,dl$ **exactly for any step count**. Measured at the production validation points (below/inband/strong classes): fluxes at $N = 8$ and $N = 1000$ steps agree to $\le 4.5\times10^{-9}$ (ratios $\le 6.5\times10^{-10}$), across completely different step-edge geometries and an in-band $m_{Z_D}$ sweep — the scheme is *consistent* (its $N\to\infty$ limit is the true path-ordered answer), so N-independence certifies correctness. The old scheme's answers were realizations scattered around this converged limit within their own ~1e-2 band. The scheme initially shipped with `N_steps_earth = 100` (12.5× margin over the observed knee): solver defaults 1.92 → **1.11 s**; analysis eval 3.01 → **0.84 s** single-core, 0.72 → **0.25 s** pooled. **Update 2026-07-29**: the default is now `N_steps_earth = 25` (~3× margin, well inside the same certificate), revalidated 100-vs-25 at the four standard points at production settings (below/inband/strong fluxes agree to ≤1.0×10⁻⁹, energy-averaged ratios ≤1.2×10⁻¹⁰, dip bins identical; the cosmoX2 point then shifted its energy-averaged ratios by 2.9×10⁻³ between step counts — the non-unitary propagator of the time, see Section 3.3; since 2026-10-12 cosmoX2 agrees across 8, 25 and 100 steps to $10^{-14}$). The Earth phase was 56–66% of every analysis evaluation, so this is the largest win the certificate allows for free: same-conditions measurements, solver defaults 0.45 → **0.32 s**, analysis eval 0.81 → **0.39 s** single-core (0.29 s on thermal-cache hits) / 0.25 → **0.17 s** pooled. Potential-dominated points ($m_{Z_D}\sim$ tens of eV) were long believed pointwise-unconvergeable under any scheme; that was the propagator's non-unitarity, removed on 2026-10-12 (Section 3.3). The `left` scheme stays selectable for reproducing historical results.

**Superseded finding (kept for the record) — adaptive Earth stepping** (2026-07, post-round-7): a gap-tracked adaptive step grid (rotation-angle + phase-curvature budgets matched to the uniform-500 grid's own observed maxima) measured ~11× potential step savings at below/inband/strong-class points. Rejected on the accuracy bar: the resolution axis itself is **fringe-aliasing-limited** — changing `N_steps_earth` from 500 to 501 (a 0.2% grid perturbation, no scheme change) reshuffles the flux at dip bins by **1.6–13% (solver defaults) / up to 17% (analysis settings)** and the energy-averaged flavor ratios by **2×10⁻³–1.5×10⁻² (defaults) / 1.8×10⁻³–1.7×10⁻² (analysis settings, 50 chords, in-band worst)**. Near the absorption dips the Earth-transit oscillation phases are sub-resolution at *any* practical step count, so every step-grid choice is one realization drawn from that band and no regridding can be certified accuracy-neutral — the same class of trade as reducing the chord count, and above the sub-percent target at in-band points. ⚠️ **Physics implication independent of any optimization**: in-band energy-averaged flavor ratios carry a ~1.7×10⁻² step-discretization ambiguity at the shipped analysis settings; treat sub-percent statements about in-band flavor ratios with this floor in mind (it adds in quadrature with the chord-count fringe floor documented in Section 11).

**Rejected optimization — thermal-grid rescaling across coupling scans** (2026-07, round 7): the idea of computing the Γ grid once per coupling scan and rescaling by `sum_G` fails *physically* wherever it would matter. Off resonance Γ is indeed linear in `sum_G` (verified to ~10⁻⁷/decade), but at resonance-window cells the narrow-width plateau contributes Γ ∝ `sum_G_alpha`/MG ∝ *coupling-independent* (that independence was the normalization error corrected on 2026-10-12, Section 2.2; physical rates scale with the couplings) — measured at grid level (200×200, `g_D` 10⁻⁵ → 10⁻³): at in-band `m_Zd = 2×10⁶` **100% of significant cells violate** an element-wise 10⁻⁶ linearity gate, with deviations up to 0.99 (dip-cell rates barely move while `sum_G` grows 100×); even below-band `m_Zd = 3×10⁵` has 9.5% marginal violations (≤ 7×10⁻⁶). The planned safety gate would therefore always fall back to exact per-point grids. With the semi-analytic thermal kernel already cutting the grid cost 2–5.6×, the machinery is not worth its verification overhead.

---

## 7. Configuration Guide & Command-Line Flags

### `[cosmology]`
* `--H0_km_s_Mpc` (float): Present-day Hubble constant in km/s/Mpc (Default: `67.4`).
* `--Omega_m` (float): Cosmological matter density parameter (Default: `0.315`).
* `--Omega_L` (float): Cosmological dark energy density parameter (Default: `0.685`).
* `--T_nu0_eV` (float): Present-day C$\nu$B temperature in eV (Default: `1.676389e-4`, i.e. $(4/11)^{1/3}\,T_{\rm CMB} = 1.9454$ K; until 2026-10-12 the default was `1.95e-4`, "1.95 K" entered as eV, which made the relic neutrinos 16% too hot and the absorption rates ~57% too high).
* `--n_n0_cm3` (float): Present-day cosmological background neutron density in cm$^{-3}$ (Default: `3.091e-8` = $n_b\,Y_p/2$, the PDG 2024 baryon density $2.515\times10^{-7}$ cm$^{-3}$ times the neutrons per baryon bound in helium-4, $Y_p/2 = 0.1229$; until 2026-10-12 the default was `0.25e-6`, the baryon density itself, which made the cosmological potential 8.1x too strong).
* `--m_lightest_eV` (float): Mass of the lightest neutrino eigenstate in eV (Default: `0.01`).
* `--mass_ordering` (str): `"normal"` or `"inverted"` (Default: `"normal"`).
* `--eta_e`, `--eta_mu`, `--eta_tau` (float): C$\nu$B flavor asymmetry multipliers (Default: `1.0`).

### `[standard_mixing]`
* `--dm21_sq`, `--dm31_sq` (float): Mass-squared differences in eV$^2$ (Defaults: `7.41e-5`, `2.511e-3`; the sign of $\Delta m^2_{31}$ is handled automatically for inverted ordering).
* `--th12_deg`, `--th13_deg`, `--th23_deg` (float): PMNS angles in degrees (Defaults: `33.41`, `8.58`, `49.1`).
* `--d_cp_deg` (float): Dirac CP phase in degrees (Default: `132.0`).
* `--use_SM_potential` (bool): Standard MSW potential inside the Earth (Default: `false` — negligible for the $>10^4$ GeV energies this code targets; see Section 2.1).
* `--n_e_earth_cm3` (float): Electron density for $V_{SM}$ in cm$^{-3}$ (Default: `1.5e24`).
* `V_SM_struct` (JSON only): optional 3×3 flavor texture override for $V_{SM}$ ([Section 8](#8-custom-potential-structure-matrices)).
* `--detector_depth_km` (float): Detector depth in km (Default: `1.5`).

### `[dark_portal]`
* `--u_e`, `--u_mu`, `--u_tau` (float): Sterile flavor mixing parameters (Defaults: `0.5`, `0.5`, `0.0`).
* `--m_Zd_eV` (float): Dark mediator mass in eV (Default: `2.0e6`).
* `--g_D` (float): Dark gauge coupling (Default: `1e-4`).
* `--epsilon_prime` (float): Kinetic mixing parameter (Default: `1e-1`).
* `--use_debye_screening` (bool): Thermal Debye mass in the LRI denominator (Default: `true`).
* `U_LRI_struct` (JSON only): optional 3×3 flavor texture override for $V_{LRI}$ ([Section 8](#8-custom-potential-structure-matrices)).

### `[source]`
* `--p_index` (float): Injection spectral index (Default: `2.5`).
* `--Phi_0` (float): Source normalization (Default: `1.0`).
* `--z_max` (float): Maximum redshift (Default: `4.0`).
* `--rho_evol` (float): Source evolution index on top of the SFR (Default: `0.0`).
* `--f_e_S`, `--f_mu_S`, `--f_tau_S` (float): Source flavor ratios (Defaults: `0.333333, 0.666667, 0.0` — pion decay).
* `--is_antineutrino` (str): `"sum"`/`"both"` ($\nu + \bar\nu$), `"true"` ($\bar\nu$ only), `"false"` ($\nu$ only) (Default: `"sum"`).
* `--nubar_to_nu_ratio` (float): $\bar\nu/\nu$ injection ratio for `"sum"` mode (Default: `1.0`).

### `[grid]`
* `--E_min_eV`, `--E_max_eV` (float): Energy band (Defaults: `1e13`, `3e16` — i.e. $10^4$ to $3\times10^7$ GeV).
* `--n_E_bins` (int): Log-spaced energy bins (Default: `1000`).
* `--n_z_bins` (int): Linear redshift bins (Default: `1000`).
* `--n_mw_paths` (int): Milky Way trajectories in the solid-angle average (Default: `6`).
* `--n_earth_chords` (int): Earth chords in the zenith average (Default: `12` — raised from the historical 3, which the convergence study measured at $1.6\text{–}4.9\times10^{-2}$ absolute flavor-ratio error; see Section 11). Under the default phase-aware zenith average 12 chords agree with 50 to $\le1.4\times10^{-7}$ at the benchmark points (Section 3.5).
* `--earth_zenith_average` (str): Average over the Earth chords (Default: `"phase"` — integrate each block of the chords' density matrices against its dark-portal phase over $\cos\theta_z$, removing the chord-sampling floor; `"plain"` is the historical equal-weight mean, implied by `--earth_step_scheme left`). See Section 3.5.
* `--N_steps_MW` (int): Path-length steps per Milky Way trajectory (Default: `25` — lowered from the historical 100; 25 already agrees with a 400-step reference to $\sim10^{-10}$ everywhere tested).
* `--N_steps_earth` (int): Path-length steps per Earth chord (Default: `25`; **uniform across chords by design** — see Section 3.4). Under the default `avg` step scheme results are certified step-count-independent (N = 8…1000 agree to ≤5×10⁻⁹, and the 100 → 25 default change was revalidated at the four standard points); under the legacy `left` scheme results scatter at the ~10⁻² level with *any* step-count change, and 500 was the pinned value.
* `--earth_step_scheme` (str): Earth-transit potential discretization (Default: `"avg"` — exact step-averaged potential via layer-split Gauss–Legendre integration of PREM, making the accumulated LRI phase step-count-independent; `"left"` reproduces the historical left-endpoint sampling).
* `--n_z_thermal` (int): Coarse z-samples for the thermal-rate grid (Default: `0` = exact grid, no interpolation; see `build_thermal_grid`'s measured error budget before enabling).
* `--thermal_analytic` (bool): Semi-analytic momentum integration of the thermal rates (Default: `true`; set `false` for the 100-point midpoint reference kernel — Section 2.2).
* `--r_start_kpc` (float): Milky Way integration start radius in kpc (Default: `100.0`).

### `[output]` & `[execution]`
* `--flux_filename`, `--plot_filename` (str): Output paths.
* `--n_cores` (int): Cores used to parallelize the Milky Way paths / Earth chords of the spatial handover within a single run (Default: available cores − 1 on a workstation, the whole allocation inside a SLURM job). "Available" is the process's CPU affinity mask capped by `SLURM_CPUS_PER_TASK` (`dp_core.available_cores`), never `cpu_count()`, which reports a whole cluster node whatever the job was given. The scanner, the convergence tool and the analysis package size their pools the same way.
* `--verbose` (int): `1` standard, `2` granular milestones (Default: `1`).

---

## 8. Custom Potential Structure Matrices

Both coherent forward-scattering potentials accept an optional user-defined 3×3 Hermitian **flavor structure matrix** through the config dict, replacing the built-in textures:

```jsonc
{
    "standard_mixing": {
        "use_SM_potential": true,
        "V_SM_struct": [[1.0, 0.0, 0.0],
                        [0.0, 0.5, 0.0],
                        [0.0, 0.0, 0.0]]
    },
    "dark_portal": {
        "U_LRI_struct": [[0.25, [0.0, 0.1], 0.0],
                         [[0.0, -0.1], 0.25, 0.0],
                         [0.0, 0.0, 0.2]]
    }
}
```

* Each entry is a plain number (real) or a two-element `[re, im]` list (complex) — JSON has no native complex type.
* Matrices are validated at parse time: shape must be 3×3 and the matrix must be Hermitian (the eigensolver reads only one triangle, so a non-Hermitian potential would be *silently* wrong; the code raises instead).
* Supply matrices in **neutrino convention**; the antineutrino transform $V \to -V^\ast$ is applied centrally.
* Defaults when absent: $S_{SM} = \mathrm{diag}(1,0,0)$ and $S_{LRI,ij} = u_i u_j^\ast$ — bit-identical to the pre-override behavior (regression-tested).
* **Physics caveat:** the overrides change only the coherent forward-scattering potentials. The Debye screening mass and the thermal attenuation/regeneration rates remain tied to `u_e/u_mu/u_tau`. The analysis package warns when a configuration combines a `U_LRI_struct` override with *floating* `u_*` parameters, since those would then affect only the rates, not the potential.

---

## 9. The Statistical Analysis (`analysis/`)

`analysis/` fits the solver's energy-averaged flavor fractions to measured or projected flavor-composition likelihoods by profile likelihood. It replaced FrequentistAnalysis on 2026-10-12; the design, and what changed, is in `docs/dev/ANALYSIS_DESIGN.md`.

```bash
python -m analysis analysis/configs/pion_2050_case_5_exclusion.json --workers 8   # everything the config lists
python -m analysis CONFIG.json --fit                    # the global fit only
python -m analysis CONFIG.json --profiles g_D u_e       # chosen profiles
python -m analysis CONFIG.json --planes                 # the planes
python -m analysis CONFIG.json --asimov                 # against Asimov data
python -m analysis CONFIG.json --summarize              # re-summarize existing results
```

* **Configuration** (one JSON per analysis, `analysis/configs/`): the data surface (`data.surface`, one of the 35 in `analysis/data/flavor_surfaces.json`), a partial solver configuration in the layout of Section 7 (anything absent takes the solver's default; unknown keys are rejected), the free parameters with start, bounds and fit space (`linear` or `log10`), priors (`gaussian`, or `half_gaussian` for one-sided limits, also on the derived `sum_masses_eV`), fit settings, profile and plane grids, Delta chi^2 thresholds (default 1D 1/2.706/3.841, 2D 2.296/4.605/5.991 for 68.27/90/95%), the Asimov injection, and the output directory.
* **Model:** each evaluation overrides the solver configuration and calls the solver's own `build_solver_params` / `propagate_nu_nubar`; solver runs are cached. The fluxes are linear in the source composition, the antineutrino ratio and the normalization, so those float without new runs (with the source composition free, the remainder goes to `f_tau_S` if `f_e_S` and `f_mu_S` both float, to `f_mu_S` otherwise).
* **chi^2** = -2 ln(L/L_max) of the surface (zero at its centre) + priors; mixing beyond unitarity is evaluated on the unit sphere plus a continuous wall.
* **Fits and scans:** L-BFGS-B from several seeded starts, then a Powell polish. Profiles and planes run in a spawn pool: chains walked outward from the best fit with warm starts, then repair passes that refit points from their neighbours' solutions where those do better, where a point sticks out above its neighbours, and (once) at the edges. Every fit is appended to a JSONL checkpoint; rerunning resumes.
* **Outputs** (`output_dir/data/` or `.../asimov/`): `fit.json`, `profile_<name>.json` (Delta chi^2 and intervals at every threshold, open ends flagged) with its figure, `plane_<x>_<y>.json` with its figure, and `manifest.json` (code version, configuration digest, machine, timing of every run). Delta chi^2 is measured from the lowest chi^2 found anywhere.
* **Shipped configurations:** `pion_2050_case_5` (MeV mediators, m_Zd, g_D, u_e, u_mu free) and `pion_2050_case_5_mZd_1e-30` (ultralight) port FrequentistAnalysis' two analyses of the 2050 projection; `pion_2050_case_5_exclusion` is an exclusion plane. All fix the mixing at NuFIT 6.0 NO with SK, which reproduces the projection's centre (FrequentistAnalysis' fixed values put standard physics 0.7 sigma off it).

## 10. Exclusion Planes

The 2050 projection is centred on standard physics. With every coupling free, a fit can always switch the new physics off, so profiles of that data stay at Delta chi^2 ~ 0: exclusion regions need fixed couplings or a fixed texture. `analysis/configs/pion_2050_case_5_exclusion.json` fixes the e-mu texture of the theory notes at |U_e4| = |U_mu4| = 0.5 and maps Delta chi^2 over (m_Zd, g_D); nothing else floats, so each point is one evaluation. **Exclusion planes are read per mediator mass** (decided 2026-10-13): with `"raster": "m_Zd_eV"` in the plane's entry, Delta chi^2 is measured in each mass column from that column's own minimum, and couplings above Delta chi^2 = 3.84 (95%, one parameter) are excluded. Columns whose own minimum lies above the best fit by more than that are hatched in the figure and listed in the summary: their scanned couplings never reached the region without an effect, so the range must be extended there before a limit is read. Planes without `raster` are read as joint regions at the 2D thresholds.

**Planes by regime** (decided 2026-10-13). The data depend on the dark coupling and the sterile mixing only through products, so a free mixing would always slide to zero; the planes scan the products instead, and their limits hold for any mixing. For each benchmark texture of the theory notes (eμ, eτ, μτ):
* `pion_2050_case_5_mev_<texture>.json`: MeV mediators with ε′ = 0, which switches off the coherent potential and leaves the resonant absorption alone, in (m_Zd, g_D·|U|²) (`g_D_u2`), on which every observable then depends alone (tested).
* `pion_2050_case_5_potential_<texture>.json`: the coherent potential from ultralight to GeV mediators, in (m_Zd, g_D·ε′·|U|²) (`g_D_eps_u2`), to which the potential is proportional; with ε′ = 0.1 the absorption coupling ΣG ≈ g_D|U|²·ε′g/c_W depends on the same product, so the plane is a function of it alone. At MeV masses the Earth's own potential is far from negligible: with g_D·ε′ = 1e-5 at 2 MeV it is ~1e-8 eV against a vacuum term of ~1e-16 eV, and it mixes up-going flavors fully.

The detector flux averages the **whole sky** (`grid.earth_sky = "full"`, the default since 2026-10-13; `"up"` keeps the historical up-going hemisphere): down-going neutrinos cross only the ice above the detector, 1.5 km straight down to ~140 km at the horizon, against up to 12,740 km through the Earth. How much this dilutes the Earth's effect depends on the potential: by half while the ice adds little phase, not at all once it mixes the flavors from above too. At the MeV benchmarks the dark-portal phase is 6e5 rad through the core but only 7 rad straight down, so near-vertical down-going neutrinos keep part of the arriving flux's flavor structure (f_e − f_μ = 3e-3 at `strong`). Other planes follow by choosing which parameters are free and which are fixed in the solver section. Flavor fractions alone are degenerate: Asimov data injected at m_Zd = 2 MeV were fitted equally well (chi^2 = 7e-7) at m_Zd = 8e-14 eV.

---

## 11. Choosing Resolution Parameters (Convergence Tool)

**The safe resolution settings depend on the dark-portal parameters.** The absorption resonance ($E_{res} \sim m_{Z_D}^2/2m_\nu$) moves through the energy band with $m_{Z_D}$, and the Earth's MSW-like level crossing depends on the potential magnitude — a grid that converges beautifully at one parameter point can be badly under-resolved at another. `tools/convergence_study.py` measures this directly, running the **full production pipeline** over resolution ladders and comparing each rung against a finer reference via the **maximum** relative difference over the **full** spectrum (plus energy-averaged flavor ratios), at several parameter points spanning resonance-below-band / in-band / strong-coupling:

```bash
cd tools
python3 convergence_study.py --n_cores 8                          # all six axes (slow; run overnight)
python3 convergence_study.py --axes n_z_bins n_E_bins --n_cores 8 # just the cosmological grid
python3 convergence_study.py --param_points my_points.json        # your own parameter region
```

Reports (JSON + text) land in `output/convergence/`. Headline findings at the default parameter points:

* **`n_z_bins` / `n_E_bins` = 1000 are *not* overkill.** Energy-averaged flavor ratios still move at the few-$10^{-3}$ / few-$10^{-4}$ level respectively between 1000 and a 1500-bin reference when the resonance is in-band. Pointwise flux in the deep absorption dip never converges tightly — exponential attenuation amplifies any resolved-rate difference — which is why the ratio metric is the one to watch. The shipped analysis configs were upgraded from their historical 50×50 grid (which carried few-percent ratio systematics) to 200×200 on this basis.
* **`N_steps_MW` is heavily over-resolved:** 25 steps already agree with a 400-step reference to $\sim10^{-10}$ at all tested points (the Milky Way transit is deeply adiabatic; its potential is negligible at the default parameter points). The default 100 is safe margin, not necessity.
* **`n_mw_paths` = 6 suffices at the default points** for the same reason — but this is exactly the axis whose difficulty is parameter-dependent: at much smaller $m_{Z_D}$ (unscreened, longer-range galactic potential) the anisotropy matters and the tool should be re-run.
* **`n_earth_chords` was the weakest default.** The historical default of 3 chords carried $1.6\text{–}4.9\times10^{-2}$ *absolute* flavor-ratio error against a 100-chord reference at every tested parameter point — far above the other axes — improving to $\sim10^{-2}$ at 12 chords (the current default) and $0.7\text{–}11\times10^{-3}$ at 50 (the analysis configs' value). Convergence is non-monotonic (25 chords tested *worse* than 12): the zenith average samples Earth-transit interference structure, so chord-count changes partially alias rather than settle. Use ≥ 12 chords for percent-level ratio work, ≥ 50 for sub-percent, and check with the tool at your own parameter points. **Resolved 2026-10-12:** the aliasing was sampling noise of the dark-portal phase in the equal-weight chord mean; the phase-aware zenith average (Section 3.5, the default) removes it — 12 chords now agree with 50 to $\le1.4\times10^{-7}$ at below/inband/strong — and the numbers above describe the historical `"plain"` mean.

The tool's verdicts are gated by `--criterion` (default `ratio`: the energy-averaged flavor ratios the likelihood consumes; `both` restores the strict AND with the pointwise-flux gate, which is structurally unmeetable in the deep absorption dip) with `--ratio_tolerance` (default $10^{-2}$). Recommendations use an *upward-closed* rule — the smallest tested value from which every finer tested value also passes — because aliasing makes intermediate resolutions test better than finer ones, and a value is only trustworthy if refinement never degrades it. A saved study can be re-scored under new tolerances without re-running the solver:

```bash
python3 convergence_study.py --reanalyze ../output/convergence/convergence_production_<stamp>.json --criterion ratio --ratio_tolerance 1e-3
```

Under the percent-level ratio criterion, the production study's robust recommendations are: `n_z_bins` ≥ 1000, `n_E_bins` ≥ 200, `N_steps_MW` ≥ 25, `n_mw_paths` ≥ 6, with `n_earth_chords` floored at $\sim10^{-2}$ by fringe aliasing (see above; the *step*-count half of that floor was eliminated in 2026-07 by the phase-exact `earth_step_scheme = "avg"` — see Section 12 — and the *chord*-count half in 2026-10 by the phase-aware zenith average, Section 3.5). The solver defaults and the shipped analysis configs now reflect these numbers (the analysis configs use a 200×200 E/z grid: the z-component of the scatter at 200 bins, $1.1\times10^{-2}$, was comparable to the 50-chord fringe floor, so the ~3× further cost of `n_z_bins = 1000` was deferred until that floor was addressed — **it now is (Section 3.5), which leaves the 200-bin grid's scatter as the analysis settings' largest numerical error; revisit the grid** — see the `_comment_resolution` note in the configs, including the measured ~15× per-evaluation cost of the new grid vs the old 50×50).
* **[Historical — `left` scheme; superseded for the step axis]** **`N_steps_earth` likewise showed percent-level *fringe aliasing*, not clean convergence**, when a resonance is in-band: under the default `avg` scheme this axis is now certified converged (Section 12), and the description below applies to the legacy `left` scheme only. The *energy*-fringe smearing caveat at the end remains relevant regardless of scheme: the Earth transit imprints interference fringes in energy finer than the energy grid resolves, so bin-center sampling turns resolution changes into pseudo-random percent-level scatter of the ratios (125 → 2000 steps never settles below $\sim10^{-2}$ ratio scatter at the in-band point). Increasing step counts alone does not remove this — it is a property of the unsmeared observable. For robust fits near such parameter points, either apply detector energy smearing downstream or treat percent-level ratio scatter as a numerical systematic.

---

## 12. Performance Notes

* **Thread pinning:** BLAS/LAPACK pools (MKL/OpenBLAS/OMP) are pinned to one thread at import, *before* numpy loads. Every linear-algebra call here touches 3×3 matrices; unconstrained BLAS pools multiply against the code's own process-level parallelism and can slow parallel runs 3–5×.
* **Solver parallelism:** within one run, `n_cores` distributes the independent Milky Way paths / Earth chords across processes, and the cosmological phase's `prange` loops scale via `numba.set_num_threads`.
* **Scanner / analysis parallelism:** both parallelize across configurations/grid points at the process level; each worker forces its internal physics to a single core.
* **Spawn-safe pooling:** the analysis runs its workers under the `spawn` start method (numba's kernels and the solver's pools are not fork-safe), where creating a pool is expensive; each scan therefore creates one pool and each worker builds its objective once. The solver's spatial phase keeps one *persistent* internal pool for repeated calls in the same process rather than recreating one per call (per-call pool creation under `spawn` was ~10× *slower* than single-core).
* **Closed-form propagator:** the spatial hot loops use `propagator_3x3` (Section 3.3, no LAPACK call) with allocation-free, Hermitian-exploiting ρ-updates (upper triangle + conjugate mirror). Removing the 100%-firing eigh fallback roughly halved full production runs; the exactly unitary form of 2026-10-12 is 0.6–0.7× the previous cost per Earth step and 1.25× per Milky Way step.
* **Threads-based spatial mode: attempted twice, rejected twice (final).** Round 6: numba's default workqueue threading layer deadlocked the `spawn`-subprocess + persistent-pool combination. Round 7 retried under the **TBB layer** (which coexists cleanly with spawn pools — verified by a dedicated stress test now in the suite): a whole-phase fused `prange` kernel reached 4.1× scaling on 8 threads (5.0× on 12; the machine is a hybrid 2P+8E-core part, so the uniform-core 6–7× target is not physically available) — but **any** `parallel=True` recompilation of the spatial kernels changes fastmath FMA contraction at the last ulp (two *textually identical* kernels measured bit-different), and at the `cosmoX2` level-crossing point the phase-noise-regime spatial steps amplify any ulp change into ~10⁻³ flux differences, far beyond the ≤4×10⁻¹¹ validation gate. Bit-identical validation is therefore unattainable for any prange restructure of these kernels; the mode was reverted per the round's own acceptance rules. What landed instead: the **TBB threading layer** (spawn-safety for the existing thermal prange kernels) and a **persistent spatial process pool** for the standalone solver (previously four pools were created and destroyed per run — one per phase per flavor pass), which is bit-identical by construction.
* **Current benchmarks** (12-core desktop, 8 cores): solver defaults (1000×1000 grid, 12 chords) ≈ 6.2 s; analysis-config likelihood evaluation ≈ 6.1 s single-core / 1.6 s in the pooled global fit.
* **JIT cache:** first-ever execution compiles for ~10 s; the scanner additionally warms the cache once before spawning workers so they don't compile simultaneously.
* **Cosmological phase:** the cascading integral runs in O(N_E) (its kernel is separable, so the downward sweep of Section 3.1 accumulates it as it goes), and the per-(E, z) mixing projections come from the σ-table (Section 3.3) — ~10⁶ eigenvector computations per run collapsed to ~4×10³ table nodes, validated to 2.7×10⁻¹⁴ at a level-crossing-in-range stress point. The thermal-rate grid uses the **semi-analytic momentum kernel by default** (Section 2.2): the "treat the resonant spike analytically" path that two validation-rejected tabulation schemes (coarse-z interpolation, (x, τ) tabulation) pointed toward — those failed because the Breit–Wigner resonance is narrower than any practical grid spacing, so it must be integrated in closed form, not sampled (see the NOTE at `precompute_all_thermal_rates`). The midpoint reference kernel remains selectable via `grid.thermal_analytic = false`, and the legacy `--n_z_thermal` z-interpolation stays available opt-in with its measured error budget. Thermal-grid build at production size: ~2.0 s → 0.28–0.8 s (parameter-dependent, after node tuning). Full solver defaults across the 2026-07 campaigns: 6.23 s → 2.41 s (semi-analytic thermal) → 1.92 s (persistent spatial pool) → ~1.1 s (phase-exact `avg` Earth scheme at N = 100) → **~0.3 s** (N = 25, revalidated inside the same certificate; 0.45 → 0.32 s same-conditions); analysis eval 6.14/1.58 s → 0.84/0.25 s (N = 100) → **~0.4 s** single-core (0.29 s on thermal-cache hits) / **~0.17 s** pooled (N = 25).
* **SVML (operator note — ACTIVE since 2026-07):** the environment now runs the SVML-enabled `llvmlite 0.43.0 py312h6a678d5_1` build (anaconda `defaults`; the previously installed `_0` build of the *same version* reported `has_svml() = False`), with pip `icc-rt` supplying `libsvml.so` — `numba.core.config.USING_SVML` is True. **Measured effects:** 3.5× on a pure transcendental fastmath microbenchmark; **1.6× on the midpoint *reference* thermal kernel** (array-vectorized code, 1.97–2.08 → 1.26 s at production size); **no measurable change to the default paths** — the semi-analytic thermal kernel is scalar/branchy (LLVM does not auto-vectorize it, so no SVML calls are emitted) and `propagator_3x3` is scalar *and* non-fastmath (its transcendentals deliberately stay libm — verified by cosmoX2 validating at 1.7×10⁻¹³ across the swap). Under `fastmath=True`, SVML's fast variants differ from libm at up to ~5×10⁻⁷ on the thermal integrand (three orders below that kernel's own accuracy; the reimplementation test's pin was relaxed 1e-9 → 1e-5 accordingly). **Rollback:** `NUMBA_DISABLE_INTEL_SVML=1` (env var, no reinstall) or `conda install "llvmlite=0.43.0=py312h6a678d5_0"`. Verify with `python -c "from numba.core import config; print(config.USING_SVML)"`.
* **Warm-started scans:** profile and plane points are fitted in chains walked outward from the best fit, each point started from its predecessor's solution and from the best fit; repair passes follow (Section 9).
* **GPU (optional, off by default):** `--use_gpu` / `execution.use_gpu` runs the spatial handover as one batched CUDA workload via cupy (`src/gpu_spatial.py`) — same density helpers, and `propagator_3x3` **ported to batched masks** (traceless shift, pure-phase/Taylor branches, the loner's eigenvector, the orthonormal complement and the pair's 2×2 block, the degenerate form for unresolved pairs — **no LAPACK call**, replacing the old Lagrange + batched-eigh-fallback scheme that fired for ~100% of Earth evolutions and was numerically inconsistent with the CPU path). Falls back to CPU with a warning when cupy or a CUDA device is missing. **Validation status:** the port's *mathematics* is validated on CPU — the batched kernel takes its array backend as a parameter, so the suite runs it with numpy against `propagator_3x3` across every branch regime — but **execution under cupy on real hardware is not yet tested**: before trusting `--use_gpu` results, run `python3 -m pytest tests/test_physics_regression.py -k gpu -v` on the GPU machine (the device test compares GPU vs CPU output end-to-end and is skipped without a device). The analysis package does not use the GPU path: its scan workers already saturate the CPU cores, and hundreds of workers contending for one GPU is an anti-pattern.

---

## 13. Testing

```bash
python3 -m pytest tests/ -v        # the regression suite and the analysis tests
```

The regression suite pins down, among other things: the closed-form Breit-Wigner angular integral against brute-force quadrature; the semi-analytic thermal momentum kernel against a dense structure-aware referee across a g_D ladder (and the sum_G linearity the coupling cache relies on, plus the cache's verification gate, bracket bounds, and bit-exact reference-path bypass); the antineutrino conjugation conventions of both potentials (including complex, CP-violating textures) across the cosmological and spatial code paths; the structure-matrix override plumbing (explicit defaults must be bit-identical to no override); the propagator against the eigendecomposition and a 50-digit matrix exponential, its unitarity at every potential from $10^{-24}$ to $10^{22}$ eV (far-eigenvector ray, single-flavor textures), the degenerate treatment of pairs below double-precision resolution, and trace conservation through the whole spatial handover; the GPU propagator port's mathematics via a numpy backend; the PREM layer-crossing closed forms and step-averaged potentials against adaptive quadrature, plus the `avg` scheme's step-count-stability certificate; the exact chord length (every chord starts on the surface); the Earth's Yukawa integral against the uniform-ball closed form at $mR$ from $10^{-7}$ to $10^8$, brute-force quadrature of the shell kernel, the long-range (enclosed-charge) limit and the exterior point source, the agreement of its two evaluations across their switch, bit-identity with the contact form at the benchmark masses, and step-count independence with the boundary layers resolved; the uniform-across-chords Earth step count (Section 3.4); the loud-NaN contract at extreme mediator masses (finite flux end-to-end, ValueError on non-finite grids); the config-wiring of every parameter that once went silently dead (`n_e_earth_cm3`, `use_SM_potential`, `use_debye_screening`, `n_cores`); Milky Way density finiteness on the galactic-center line; the absolute checks (energy at Earth below the injected, conservation through the spatial phase, vanishing mixing recovering standard propagation); and the TBB prange → spawn-pool coexistence stress test. `tests/test_analysis.py` covers the analysis package (surfaces against recorded FrequentistAnalysis values, fits, scans with repair and resume, intervals, the command line end to end). **Run it before and after any physics change.**

---

## 14. Authors & Citation

**Author:** Mauricio Bustamante
If you use this code in academic research, please cite the associated theoretical manuscripts detailing this statistical procedure and propagation framework.
