# Plan: a general matter charge and a family of neutrino textures (the dark-portal paper's scope)

Status: agreed 2026-10-14 (Mauricio's decisions below; the rest are technical defaults, mine to decide and report).
Paper working title (Mauricio, 2026-10-14): "New forces on high-energy cosmic neutrinos, from the Hubble scale to the
Earth". No drafting yet. The spin: the most comprehensive calculation of these effects in high-energy astrophysical
neutrinos, with several kinds of current data (HESE, DR2 tracks, IceCube's 2026 spectrum and flavor), plus
projections.

## Decisions (Mauricio, 2026-10-14)

1. **Widen the model in two ways.** (1) A general matter charge: the potential is sourced by
   q_n n_n + q_p n_p + q_e n_e instead of neutrons alone. (2) A family of neutrino textures: free mixing ratios
   for the dark portal's rank-one texture u u^dagger, and the diagonal textures of gauged lepton-flavor symmetries.
2. **Eight benchmark models**, each with the matter charge and the texture that its gauge structure fixes:

   | Model | Matter charge (neutral matter) | Neutrino charges (e, mu, tau) | In ABDN23 |
   |---|---|---|---|
   | Dark portal | neutrons (Z-Z_D mass mixing) | rank one, u u^dagger, ratios free | -- |
   | L_e - L_mu | electrons | (1, -1, 0) | yes |
   | L_e - L_tau | electrons | (1, 0, -1) | yes |
   | L_mu - L_tau | neutrons (Z-Z' mass mixing) | (0, 1, -1) | yes |
   | B - 3L_e | n + p - 3e = n - 2e | (-3, 0, 0) | -- |
   | B - 3L_mu | nucleons, n + p | (0, -3, 0) | -- |
   | B - 3L_tau | nucleons | (0, 0, -3) | -- |
   | B - (3/2)(L_mu + L_tau) | nucleons | (0, -3/2, -3/2) | -- |

   ABDN23 = Agarwalla, Bustamante, Das, Narang, JHEP 08 (2023) 113, arXiv:2305.03675 (L_e - L_mu, L_e - L_tau,
   L_mu - L_tau; Earth, Moon, Sun, Milky Way and cosmological matter, with Yukawa factors).
3. **Free ratios: named points plus maps over the ratio triangle.** Full planes for named textures (pure e, mu,
   tau; e-mu, e-tau, mu-tau as now; democratic e-mu-tau); maps of the bound over the whole triangle
   |U_e4|^2 : |U_mu4|^2 : |U_tau4|^2 at a few Z_D masses.

4. **The split of the potential's coupling product (2026-10-14).** The potential fixes only P = g_D eps' |U|^2 (g' eps'
   for L_mu - L_tau); the relic neutrinos' screening grows with each coupling separately, so a split with a large
   g_D |U|^2 and a tiny eps' switches off the Earth's, the Sun's and the Galaxy's potentials (thresholds today, for a
   mediator lighter than 1/distance: g_D |U|^2 above 2e-9, 7e-14, 4e-23; output/figures/screening_thresholds.png).
   No published limit stops that split (MICROSCOPE bounds eps' < 3e-25 from above; cosmology bounds a neutrino-only
   coupling at ~1e-7). **Each potential plane shows two curves**: the bound on P where screening is negligible
   (independent of the split; computed at the balanced split g_D |U|^2 = eps' = sqrt(P)), with its condition stated on
   the plot, and the bound profiled over all splits (frequentist, as the planes: a point is excluded only if every
   split is), which at ultralight masses comes from the cosmological potential alone.
   Implementation (technical, 2026-10-14): `dark_portal.coupling_split` in the analysis ('balanced', or the neutrinos'
   side x with eps' = P/x <= 1); the profile's variants x = 1e-24, 1e-20, 1e-16, 1e-12, 1e-8, each on the masses where
   its screening (m_D = 1.9e-5 eV x) differs from the balanced run's (m_Zd <~ 2e-4 x eV), combined by the smallest chi^2
   (tools/combine_split_planes.py). The largest x is 1e-8: above ~1e-7 the high-energy neutrinos would scatter many
   times on the relics through the light mediator (t channel, momentum transfer ~ the relics' 1e-4 eV), which the
   potential solver does not model; the s channel's tail alone shifts the flavors by 0.04 at x = 0.5 (1e-20 eV).
   Check (solver, m_Zd = 1e-20 eV, P = 1e-54, e-mu): the flavor shift is 0.0064 at the balanced split and at x =
   1e-16, zero at x = 1e-12 and 1e-6 (the Sun and the Earth screened); the cosmological potential alone acts from
   P ~ 1e-38.

5. **The campaign's data sets and composition check (2026-10-14).** Three data sets: (1) present-day flavor (IceCube's
   MESE 2025 measurement) with IceCube's 2026 spectrum; (2) the HESE starting events and DR2 tracks alone -- no MESE
   surface, since HESE's events are also MESE's and would be counted twice (HESE's topologies carry the flavor); (3)
   the 2050 projection (case 5, pion centre) with the 2026 spectrum. **Every plane also with the sources' flavor
   compositions fitted** (extragalactic and Galactic (x, 1 - x, 0), independently; nominal: both at pion decay), which
   doubles the campaign (~50 h on 112 Tycho cores, estimated from task 2's timings).

## Physics

**Neutral matter.** n_p = n_e in every body, so any charge acts as q_n n_n + (q_p + q_e) n_e: the neutron density
and the electron (= proton) density, combined with one ratio. B - L gives neutrons alone (our current source) and a
flavor-universal neutrino texture (no flavor effect), so it is not a separate benchmark. The "nu_alpha only" textures
of B - 3L_alpha and B - (3/2)(L_mu + L_tau) are rank-one projectors: they also sit in the dark portal's free-ratio
family, there with neutrons as the source.

**Compositions** (all inputs already in the code; electrons per nucleon = 1 - neutrons per nucleon in neutral matter):

| Body | Neutrons per nucleon | Electrons per neutron | Source in the code |
|---|---|---|---|
| Cosmic mean | Y_p/2 = 0.1229 | 7.14 | n_b = 2.515e-7 cm^-3, Y_p = 0.2458 (dp_config) |
| Galaxy, stars | 0.20 | 4.00 | `_MW_NEUTRONS_PER_NUCLEON_STARS` |
| Galaxy, cold gas | 0.142 | 6.04 | `_MW_NEUTRONS_PER_NUCLEON_COLD_GAS` |
| Galaxy, hot gas | 0.15 per electron | 6.67 | `_MW_NEUTRONS_PER_ELECTRON_GAS` (the profile is n_e) |
| Sun | 0.161 | 5.21 | `N_n_sun`; electrons 1.005e57 |
| Earth | 0.514 (core 0.534, mantle 0.504) | 0.95 | PREM x Y_e per layer; electrons 1.748e51 |
| Moon | 0.505 | 0.98 | `N_n_moon`; electrons 2.188e49 |

The distant sources are hydrogen-rich (4-7 electrons per neutron), the Earth and the Moon are not (about 1): an
electron- or nucleon-sourced force weighs the cosmos, the Galaxy and the Sun 4-8 times more, relative to the Earth,
than a neutron-sourced one.

**Linearity.** Every effective density in the solver -- the cosmic mean out to the Hubble radius, the Galaxy's
Yukawa integrals and tables, the point sources, the Earth's Yukawa integral -- is linear in the source density. So
each is computed for neutrons and for electrons separately and combined at run time:
n_eff,Q = q_n n_eff,n + q_pe n_eff,e. The one non-linear step, max(Galactic sky average, cosmic mean) in
`galactic_effective_density`, is applied per species before combining. The solver assumes a single-signed density
(it carries |Coef| and the sign separately): the charge's sign is folded into the coefficient, and a charge whose
density changes sign between bodies is refused (none of the benchmarks does: n - 2e < 0 everywhere).

**Couplings and potentials.** For neutrinos (antineutrinos: -V*), with n_eff,Q the Yukawa-smoothed charge density
and m_eff^2 = m^2 + m_D^2:
- Dark portal (theory notes, Eq. T9): V = g_D eps' K_Z n_eff,n / m_eff^2 x u u^dagger, K_Z = (4 c_W m_Z^2 / g)
  (G_F / sqrt 2) = g / (2 c_W). Convention: the theory notes' eps' is half the dark-Z mixing eps_Z of Davoudiasl,
  Lee & Marciano (their neutron coupling is eps_Z g / (4 c_W)); the absorption coupling (T12) uses the same
  convention, eps' g / c_W for the neutrinos, so the model is self-consistent.
- L_mu - L_tau (same mechanism as the dark portal): V = g' eps' K_Z n_eff,n / m_eff^2 x diag(0, 1, -1). ABDN23 write
  G_mutau = g' xi g / (4 c_W), so their xi = 2 eps'.
- Direct gauge couplings (L_e - L_alpha, B - 3L_alpha, B - (3/2)(L_mu + L_tau)): V = g'^2 n_eff,Q / m_eff^2 x
  diag(Q_nu), with n_Q the matter charge density of the table above.
- Plane axes: the dark portal's potential planes scan P = g_D eps' Sum_alpha |U_alpha4|^2 (the trace of the
  texture, so every ratio point shares one axis; the current e-mu, e-tau, mu-tau planes, with |U|^2 = 0.25 per
  entry, convert by P = 2 g_D eps' |U|^2), MeV planes g_D Sum|U_alpha4|^2 at eps' = 0 as now; L_mu - L_tau the
  same two kinds in g' eps' and g'; the direct gauge models one kind, g' over the whole mass range (the same
  coupling sets the potential, ~g'^2, and the absorption).

**Absorption on relic neutrinos and the Debye mass** (DECIDED 2026-10-14: weight each relic state by its own flavor
content): the solver's rates followed the theory notes' flavor-biased form, (Sum G)_alpha from |u_alpha| |u_beta| (T12),
the same for every relic mass state -- every state at the full coupling, so the targets were counted ~3 times. For a general coupling matrix C (flavor basis) the direct generalization is
(Sum G)_alpha = Sum_beta |C_alpha beta| (|C_alpha beta| + eps' g/c_W); a mass-resolved form weighs relic state i by
|(C U)_alpha i|^2, as the secret-interaction solver does. Models coupled to electrons also decay to e+e- above
1.022 MeV (in the MeV window), which widens the resonance and removes neutrinos from the regeneration; muon and
hadron channels open only above ~0.2 GeV (resonances above ~100 PeV). The Debye mass generalizes with the diagonal
couplings |C_alpha alpha|^2 (the theory notes' g_D^2 |u_alpha|^4).

## Design (technical defaults)

- **Configuration.** A `matter_charge` section {q_n, q_p, q_e} (default neutrons: 1, 0, 0) and a `model` name that
  sets the charges, the neutrino texture and the coupling normalization for the eight benchmarks; the dark portal's
  texture from u_e, u_mu, u_tau as now (free ratios). Existing configurations run unchanged, bit for bit.
- **Densities.** Electron counterparts of every neutron input: `N_e_sun`, `N_e_earth`, `N_e_moon`, the cosmic
  `n_e0`, electrons per gram per PREM layer, and per-component Galactic electron densities. The Galaxy's two tables
  (`mw_sky_density.npz`, `mw_neff_tables.npz`) gain electron columns (rebuilt on Tycho; same digests otherwise).
  numba kernels take the charges as arguments (no module constants, which a cached compile would freeze).
- **Absorption.** `compute_gamma_grid_inputs` and the Debye mass take the coupling matrix; for C = g_D u u^dagger
  they return today's numbers exactly. Visible decay channels enter the total width only.

## Validation (absolute checks first)

1. Closure: the neutron charge (1, 0, 0) reproduces the current solver at the four validation points bit for bit
   (`tools/validate_record.py`), and the current analysis configurations unchanged.
2. Linearity: n_eff,Q from the separate species equals a direct evaluation with the combined density (Earth Yukawa
   integral, Galactic tables, cosmic mean), to rounding.
3. Totals: electrons and nucleons per body against independent numbers (solar mass and composition; Earth's mass x
   Y_e; ABDN23's and BA19's stated counts).
4. A flavor-universal texture (the identity) changes no flavor ratio, for any charge.
5. Potentials against ABDN23: V from each body versus the mediator mass for L_e - L_mu (electrons) and L_mu - L_tau
   (neutrons), in their conventions; differences traced to the Galaxy and cosmology models.
6. Absorption: the width against the closed form per model; number conservation of absorption + regeneration when
   only neutrino channels are open; the e+e- channel's loss.

## Steps

1. Matter charge in the solver: electron inputs for every body, run-time combination, configuration keys; the closure
   and linearity checks; figure: the effective density at IceCube versus the mediator mass, per body, for neutrons,
   electrons, nucleons and 2e - n.
2. Texture family and couplings: the `model` layer for the eight benchmarks, the dark portal's free ratios, plane
   axes; checks 4 and 5.
3. Absorption for general couplings (decision with Mauricio first, figure comparing the two forms); check 6.
4. The event term's speed (HESE ~0.4 s per call) before the plane campaign.
5. Configurations: the eight benchmarks x named textures x data sets (IC26 flavor and spectrum; HESE + DR2 events;
   projections), packed for Tycho; the ratio-triangle maps (real, non-negative u; step 0.1, 66 points; four masses:
   cosmic, AU, Earth-radius and MeV ranges; the bound per point by bisection in P).
6. The campaign on Tycho after task 2's reruns; figures, notes, README.

## Progress

| Step | Status |
|---|---|
| 1. Matter charge | DONE 2026-10-14: dp_matter; electron inputs for every body (the Galaxy's emission-point table on Tycho); tests/test_matter_charge.py (16); the four validation points bit for bit against the previous code, fresh compiles on both sides (cached-vs-fresh numba builds differ by 7e-15 on their own); figure output/figures/matter_charge_sources.png |
| 2. Texture family and couplings | DONE 2026-10-14: src/dp_models.py, the `model` config section (matter charge, texture, Lagrangian coupling, potential product, visible width); both signs of the potential (negative eps', and the analysis' `dark_portal.potential_sign`); tests/test_models.py: every gauge model against the potential-dominated closed form to 1e-8, nu and nubar (L_e - L_mu and L_e - L_tau keep the sources' flavors) |
| 3. Absorption for general couplings | DONE 2026-10-14 with the physics audit (docs/dev/PHYSICS_AUDIT.md): each relic mass state by its own flavor content, coherent amplitudes per eigenstate, the left-handed width, the Z mixing's three pieces, visible widths, the Moller factor, the slow relics' Debye mass (localized bodies only), the MeV regime through dp_si (every collision); tests/test_absorption.py |
| 4. Event term speed | DONE 2026-10-14: the detector interpolation's coefficients written out and shared by every template (they were recomputed 48 times per likelihood step with Python generators: 67% of the time); 253 -> 106 ms per objective call (laptop, cached solver runs), bit-identical (12 points: chi^2 and every nuisance). The rest: the exact per-event weights (46 ms) and the nuisance fit (44 ms); a plane point's ~23 s of sky runs now dominate |
| 5. Configurations and maps | IN PROGRESS 2026-10-14: tools/make_campaign_configs.py -> analysis/configs/campaign/<data set>/ (232 planes and 218 608 points per data set: 116 planes and their fitted-composition twins); every configuration loads, and one point of each plane kind evaluates (laptop smoke test); tools/ratio_triangle_maps.py (66 ratio points x 4 masses, bisection in the coupling); tools/combine_split_planes.py; configurations without a flavor surface (events only) |
| 6. Campaign | -- |
