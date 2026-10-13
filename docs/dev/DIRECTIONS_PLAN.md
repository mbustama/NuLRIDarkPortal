# Plan: per-direction output and Galactic sources

Status: **agreed 2026-10-13** (decisions D1-D6 answered by Mauricio: bands + N_side 16; CRINGE + pi0;
full 3-D emission; Sun diagnostic first; all downloads on Tycho; x_gal floated). Steps 1-5 implemented
(2026-10-13; outcomes of D4 and step 5 in Section 9). Written after four read-only surveys: the
solver's direction handling (this repository), the literature on Galactic diffuse neutrinos, the
literature on new physics with directions, and the sibling projects' directional data
(`../NuAnisotropy`, `../NuFlavorSingleSource`, `../toise`, `../PLEnuM`). Every physics choice below
cites its source; choices without a published precedent are marked **(new)** and come with a
validation that does not rely on one. Decisions for Mauricio are collected in Section 9.

References are keyed as in Section 11, e.g. [GP23] for IceCube's Galactic-plane observation.

---

## 1. Goal and scope

**Goal.** Predict the neutrino flux per arrival direction on the sky, per particle type (three
flavors, neutrinos and antineutrinos separately), differential in energy and solid angle, for both
models the solver handles:

- the dark portal (coherent, flavor-dependent long-range potential sourced by neutrons), and
- secret interactions (scattering on the relic neutrinos),

with an extragalactic diffuse component and a **Galactic-plane component** that can be switched off
(decided 2026-10-13, `docs/dev/WORK_PLAN.md`, "Include from the start").

**Consumers.** The event-rate work that follows (WORK_PLAN items 4-5): the HESE 7.5-yr per-event
MC (weight = `weightOverFluxOverLivetime` x Phi(E, delta) x livetime [HESE21]), IceTracks-DR2
(effective areas in 50 declination bins), the MESE effective areas (10 bins of sin(delta)), and the
Gen2 TDR areas (21 cos(zenith) bins). All of them need declination resolution; the Galactic-plane
analyses also need right ascension (templates on the sky [GP23]).

**Out of scope here.** Detector responses, event likelihoods, backgrounds (WORK_PLAN items 4-7);
Earth absorption (it is folded into every released effective area: [HESE21] states it explicitly
for HESE; to be verified for DR2, Section 10); per-event arrival times (none in the HESE MC or the
data files, Section 2.4).

**Unchanged.** Every existing output: the analysis consumes energy-averaged, all-sky, nu+nubar
flavor fractions from `(N_E, 3)` fluxes (`analysis/model.py` `FlavorModel.fluxes`, 143-173;
`src/flux_postprocessing.py` `compute_energy_averaged_ratios`, 2-D only). The per-direction output is
an addition behind a new config section, and the all-sky output must be recovered from it to
machine precision (Section 7, test T1).

---

## 2. What exists today (read from the code, HEAD `450fc15`)

### 2.1 Where direction enters

Only two stages depend on direction, and both are averaged on the spot:

| Stage | Direction variable | Where it is averaged | Per-direction data kept? |
|---|---|---|---|
| Sun (`dp_transit.nearfield_transit`, 755-781) | angle psi between arrival direction and the Sun: 8 Gauss-Legendre nodes in cos(psi), weights for an isotropic sky (`sun_paths`, 860-864) | `acc += w[p]*tmp` (773-781) | no |
| Earth (`dp_spatial.propagate_earth_spatial`, 1384-1675) | zenith: K chords evenly spaced in cos(theta_z), K = 2n-1 = 23 for the full sky (1483-1493) | `average_earth_chords` (979-1034), called at 1667-1669 | per chord, shape (K, N_E, 3, 3), then discarded |

Every chord starts from one Sun-averaged state (`rho_surf_avg`, 1535-1544). The Sun's direction and
the zenith are treated as uncorrelated (notes, 1793-1794). There is no time or yearly variable
anywhere in `src/`.

The cosmological phase (`dp_cosmo.step_cosmological_pair`) is isotropic: homogeneous medium,
isotropic relic background, diffuse sources. The Galactic transit is adiabatic and therefore
independent of direction for extragalactic neutrinos (`dp_transit` docstring 6-11; notes 1738-1746).
`dp_si` has no direction dependence at all.

### 2.2 The phase-aware zenith average (the key reusable piece)

`average_earth_chords(..., zenith_average="phase")` integrates each block (g, h) of the chords'
density matrices against its dark-portal phase over cos(theta_z), with weights from
`_zenith_filon_weights(c_fine, node_index, phase)` (941-960). Those weights are **sums over node
intervals**, so weights accumulated per interval and grouped into zenith bins partition the whole-sky
weights exactly, **provided bin edges lie on chord nodes** (the fine grid asserts
`c_fine[node_index] == nodes`, 913-914). Notes, zenith appendix (1510-1563): U(c) = sum_g
exp(-i lambda_g Psi(c)) P_g, Psi = Coef * integral(n dl), and
w_k^{gh} = integral hat_k(c) exp(-i omega_gh [Psi(c) - Psi(c_k)]) dc, independent of energy.

### 2.3 The Galaxy in the code

- 3-D neutron model `calc_MW_neutron_density(r_kpc, theta, phi)` (`dp_spatial` 65-102): McMillan
  bulge and thin and thick stellar discs [McM11, McM17], hot halo [MB13], HI and H2 discs [McM17];
  neutrons per nucleon 0.20 (stars, [MPH15]) and 0.142 (cold gas, Asplund 2009), 0.15 per electron
  (hot gas). Heliocentric coordinates, R0 = 8.33 kpc.
- `src/data/mw_sky_density.npz`: the 4pi sky average of that density versus heliocentric distance
  (2061 radii, 1e-6 to 2e4 kpc). **No angular information.**
- `galactic_effective_density(m, n_n0, R_H)` (`dp_transit` 706-735): the Yukawa-smoothed density
  at the Sun only.
- Host-galaxy injection (`host_galaxy_scale`, DPS 232-246; `dp_cosmo` 192-213): injected flavors
  split over the eigenstates of the potential at the sources, then carried adiabatically (sorted)
  into the intergalactic eigenstates. This is the mechanism Galactic emission will reuse.

**Missing for Galactic sources:** the Yukawa-smoothed density at arbitrary points in the Galaxy, an
emission template, and the distribution of emission points along each line of sight.

### 2.4 What the consumers expect (from `../NuAnisotropy`)

- HESE MC (20 column files, 821,764 rows): `primaryEnergy` [GeV], `primaryZenith` [rad],
  `primaryType` (PDG, nu and nubar separate), `interactionType`, `recoZenith`,
  `recoDepositedEnergy`, `recoMorphology`, `weightOverFluxOverLivetime` [GeV cm^2 sr]; a synthetic
  uniform azimuth `phi`; **no azimuth from IceCube and no event times**. Declination
  delta = `primaryZenith` - pi/2 at the Pole.
- Required input: Phi per particle type in GeV^-1 cm^-2 s^-1 sr^-1, at arbitrary (delta, RA) or as
  HEALPix maps with frame and ordering stated; for coarse pixels, pixel averages rather than
  centre values (NuAnisotropy's published LIV analysis computed at N_side = 32 and averaged down).
- Conventions to watch: that project's fit script maps healpy theta = pi/2 + delta (north-south
  flipped against the standard theta = pi/2 - delta). Our files will state the standard convention.

---

## 3. Physics model, component by component

### 3.1 Sky and coordinates

- **Frame:** equatorial J2000 (IceCube's sky maps are equatorial J2000 [GP23]). Galactic
  coordinates by the IAU 1958 definition [Bla60] as implemented in astropy (FK5 J2000 Galactic
  pole at (192.8594812 deg, +27.1282512 deg), l of the celestial pole 122.9319186 deg); the
  FK5/ICRS difference is negligible at IceCube resolution [Liu11].
- **Pixelization:** HEALPix [Gor05], RING ordering, theta = pi/2 - delta, phi = RA (healpy's
  convention). Pixels are equal-area, so pixel averages and all-sky integrals are plain sums.
- **Zenith at the detector:** at the South Pole cos(theta_z) = -sin(delta) for every direction at
  every time ([Auger19]; IceCube converts zenith directly to declination [IC19]). So the Earth part
  of the flux depends on declination only, and the per-direction Earth output is a set of
  **declination bands**. The Galactic Centre (delta = -28.9 deg) is down-going, cos(theta_z) = +0.48.
- **Sun at the Pole:** the Sun's zenith angle is 90 deg + delta_sun, between 66.6 and 113.4 deg,
  constant over a day [ICsun]. For an arrival direction at ecliptic latitude beta and longitude
  lambda, the angle to the Sun obeys cos(psi) = cos(beta) cos(lambda - lambda_sun(t)).

### 3.2 Extragalactic flux, dark portal

The flux at the detector for a direction n is the diagonal of

    rho(n) = E_{delta(n)} [ S_n [ G [ rho_cosmo ] ] ],

where rho_cosmo is the isotropic cosmological arrival (unchanged), G the adiabatic Galactic map
(unchanged, direction-independent [notes 1738-1746]), S_n the Sun transit for this direction, and
E_delta the Earth phase for declination delta. All three maps are linear in rho.

**Earth, per declination band (WORK_PLAN item 3).** For band b = [c_lo, c_hi] in cos(theta_z):

    rho_b = (1/(c_hi - c_lo)) sum_{k in b} sum_{g,h} w_{k,b}^{gh} P_g rho_k P_h,

with the existing Filon weights restricted to the node intervals inside the band (Section 2.2). A
single chord is not a valid per-direction answer because it carries its own fast Earth phase; the
band average is (WORK_PLAN item 3; notes, zenith appendix). Bands summed with weights
(c_hi - c_lo)/2 return today's whole-sky average exactly.

To apply the Earth map to several input states (needed once the Sun is resolved, below), each chord
returns its **unitary** U_k (per energy, 3x3) instead of the evolved state; then
rho_k = U_k rho_in U_k^dagger for any rho_in. Evolving U costs the same number of 3x3 products as
evolving rho. No physics changes: test T3 requires the two to agree to rounding.

**Sun, per direction (yearly average).** Over a year the Sun runs along the ecliptic, so for a
direction at ecliptic latitude beta the distribution of x = cos(psi) is the arcsine law
p_beta(x) = 1 / (pi sqrt(cos^2 beta - x^2)) on |x| < cos(beta), with lambda_sun uniform in time.
Eccentricity (e = 0.0167) makes lambda_sun non-uniform and the distance vary by +-1.7%; both are
included by solving Kepler's equation for the yearly orbit, as in the solar-atmospheric-neutrino
literature, which averages over the equinox-to-equinox year [EEEN17] or over r(t) [AdWFJ17]. IceCube's
solar searches instead compute the Sun's position per event [ICsun]; that needs event times, which
the HESE MC does not have (Section 2.4), so **the yearly average is the consistent choice for diffuse
predictions**. Then

    S_n[rho] = integral p_{beta(n)}(x) S_x[rho] dx.

S_x is known at the transit's path nodes; we interpolate it smoothly in x (Chebyshev, 16-32 nodes
instead of 8; the transit is smooth in x away from the solar disk, which no path samples: the closest
GL path today has impact parameter 0.28 AU [notes 1748-1767]) and integrate against the arcsine law
with Gauss-Chebyshev quadrature, exact for that weight. The Sun matters only for m_eff ~ 1e-26 to
1e-18 eV (README 130, 133; notes 1748-1750); elsewhere the transit is inactive and S_n is the identity.
Two checks pin it: the pixel-weighted sky average must reproduce today's isotropic Gauss-Legendre
average (test T4), and the anisotropy it produces must be measured before it is made the default
(decision D4).

**Combination.** Per pixel p: rho_p = sum_x a_{p,x} U_{b(p)} [ rho_x ] U_{b(p)}^dagger, where the
Earth map of band b(p) is the band-averaged channel. The cost is N_E x (Sun nodes) x (bands) 3x3
products, negligible next to the chords.

**Literature context.** Published long-range-interaction analyses evaluate every potential at the
Earth and take the flux isotropic [BA19, ABDN23]; [ABDN23] states "We do not track the propagation of
neutrinos inside the Milky Way". The per-direction treatment of the Sun and the Earth is therefore
**(new)** for TeV-PeV neutrinos. For MeV supernova neutrinos the Sun's direction was found nearly
irrelevant under adiabatic propagation [DSS26]; at 100 TeV the vacuum oscillation lengths (0.66 AU for
Delta m^2_31, 22 AU for Delta m^2_21) are comparable to the Sun's gradient, so adiabaticity is not
guaranteed and our coherent transit (already non-adiabatic-safe) is the right tool.

### 3.3 Extragalactic flux, secret interactions

Isotropic: the same `(N_E, 3)` flux in every pixel. Justification: the relic overdensity in the Milky
Way is f = n/nbar = 1.005 (0.01 eV), 1.07-1.12 (0.05 eV), 1.27-1.5 (0.1 eV), up to ~3-6 (0.3 eV)
[MPdS20, dS17, ZCA23, E23, RW04]; a 1-20 kpc path with f <= 6 adds <~ 2e-5 of the cosmological column
to a source at z = 1 (estimate from the survey, to be recomputed in the notes), so the absorption is
isotropic to ~1e-5 in optical depth. Caveat recorded: a Galactic absorption line is not smeared by
redshift; its equivalent width is ~2e-3 (g/0.01)^2 (10 MeV/M)^2 (L/10 kpc), invisible in realistic
energy bins.

### 3.4 Galactic component: template, spectrum, flavor

**Template choice.** Three public options:

| Template | What is public | Notes |
|---|---|---|
| CRINGE [SMW23] | Zenodo 7859442: HEALPix N_side 256, RING, 50 log energies 10 GeV-1e8 GeV, per-flavor intensity (all-flavor/3) in GeV^-1 cm^-2 s^-1 sr^-1; **and the 3-D cosmic-ray distribution** `Global_Hadronic_CR_Dist_CRINGE` (81 radii 0-20 kpc x 61 heights +-6 kpc x 63 energies 1-1e9 GeV) | The template closest to the 2026 IceCube combined fit: normalization 0.91 +- 0.18 x model, 5.64 sigma [IC26]. Frame to verify (the paper uses l, b). |
| pi0 [Fer12] | GALPROP SSZ4R20T150C5 gamma-ray map; a copy on disk in `../toise/toise/data/models/fermi_galactic_emission.npy` (N_side 128, Galactic, all-flavor at 1 GeV) | IceCube used it as a spatial template with a fixed E^-2.7 spectrum; fitted 21.8 (+5.3/-4.9) x 1e-12 TeV cm^-2 s^-1 per flavor at 100 TeV [GP23], i.e. 4.99 (+0.90/-0.86) x the model [IC26]. Spectrum not physical beyond the template. |
| KRA_gamma [Gag15] | Zenodo 7070823: two 807 MB pickled `.npy` files | Fitted 0.55 (5 PeV) and 0.37 (50 PeV) x model [GP23], 0.53 and 0.37 [IC26]. Pickled arrays execute code when loaded: only in an isolated process on Tycho, if at all. |

**Recommendation: CRINGE as the primary template, pi0 as the alternative** (decision D2). CRINGE is
public in the right units, fits the data best among the models with a predicted spectrum, and ships
the 3-D cosmic-ray distribution needed for emission distances (Section 3.6).

**Normalization.** A free factor N_gal (nuisance in the later likelihood), default 0.91 for CRINGE and
4.99 for pi0 [IC26]. Recorded trap: [GP23]'s supplement defines model fluxes per flavor, nu + nubar
combined, while its Eq. (5) writes Phi = Phi_nu = Phi_nubar with A_eff^{nu+nubar}; to be settled
against the release notebooks (doi:10.21234/8v5d-rn16) before any number is quoted (Section 10).

**Flavor and nu/nubar at the source: (x, 1 - x, 0), x free** (Mauricio, 2026-10-13). The source
composition is parameterized as f_S = (x, 1 - x, 0), like the extragalactic sources' f_e,S with
f_tau,S = 0 (`docs/dev/WORK_PLAN.md`, item 6), and x is floated in the analysis rather than fixed.
Default and physics expectation x = 1/3: pion decay in pp collisions gives nu_e : nubar_e : nu_mu :
nubar_mu = 1 : 1 : 2 : 2 [Spi26, Ans04], and muon damping (x -> 0) is irrelevant in the interstellar
medium (critical energy ~1e24 eV at 5 microgauss; [KW05] for the mechanism). Floating x still matters:
if unresolved Galactic sources make much of the flux [Mar25, Pre25], their composition need not be
1/3 (x = 1 for neutron decay [Ans04], x = 0 for muon-damped sources), and the data should decide; the
range x in [0, 1] spans all of these. nu = nubar per flavor by default (pp). Whether x_gal floats on
its own or is tied to the extragalactic x is decision D6. Public templates are flavor-blind
(all-flavor/3, 1:1:1 assumed at Earth [IC26]); we use their **all-flavor nu + nubar intensity** and
apply our own source composition and flavor evolution.

**Flavor at Earth, standard and secret interactions.** Oscillations are averaged over kpc: the
oscillation length 2.48 km x (E/GeV)/(Delta m^2/eV^2) is ~1e-3 pc at 1 PeV for Delta m^2_21, far below
the kpc spread of emission points; wave-packet decoherence only reinforces this [LP95, FS08]. So
f_beta,Earth(x) = sum_alpha P_beta_alpha f_alpha,S(x) with the averaged |U|^2 probabilities of our
mixing (the analysis' NuFIT parameters), linear in x: f_Earth(x) = f_Earth(0) + x [f_Earth(1) - f_Earth(0)],
so one evaluation at x = 0 and one at x = 1 give every x (for the dark portal too: the injection is
linear in f_S). Cross-check at x = 1/3: the survey's (0.30 : 0.36 : 0.34), to be recomputed.

### 3.5 Galactic component, secret interactions

Unattenuated by the relic neutrinos (Section 3.3: kpc paths). This makes the Galactic plane a nearby,
unaffected reference against which the attenuated extragalactic flux is compared; [EPBB21] call
spectral features "seen in distant but not nearby sources" a smoking gun, and no published analysis
has used the Galactic plane this way (survey, 2023-2026) **(new)**.

### 3.6 Galactic component, dark portal: emission inside the Galaxy

A neutrino emitted at x_e is injected into the eigenstates of H(E, c(x_e)), where c is set by the
Yukawa-smoothed neutron density n_eff(x_e; m_eff) plus the cosmic mean. It then travels adiabatically
to the Sun's neighbourhood (the Galactic potential changes over >= 10 pc against oscillation lengths
<= 0.005 pc at 30 PeV; `dp_transit` 6-11), so the populations of the emission eigenstates become the
populations of the local eigenstates (sorted), as for host-galaxy sources today (DPS 232-246). From
there the Sun and the Earth act as for extragalactic neutrinos (Section 3.2). Adiabaticity through
near-degeneracies is checked with the Landau-Zener parameter [Par86] along representative lines of
sight (test T9).

**Where along the line of sight.** The 2-D templates integrate along the line of sight. We need the
distribution of emission distances per pixel, q_p(s) proportional to n_gas(x(s)) J_CR(R(s), z(s), E),
normalized per pixel, used **only** for the emission potential (the flux itself stays the template's).
Inputs:

- J_CR: CRINGE's 3-D cosmic-ray distribution [SMW23] (consistent with the template we use).
- n_gas: the joint 3-D HI + H2 reconstruction of Soding et al. 2025 [Sod25] (Zenodo 12578443:
  Sun-centred HEALPix x radius grid, 50 pc-28 kpc), i.e. directly per line of sight. Alternatives:
  [MV21] (H2), [MP23] (HI).
- Published guidance: no "fraction within X kpc" is published; typical distances are ~10 kpc toward
  the Galactic Centre and a few kpc toward the anticentre [Mac25], and the path-length distribution is
  broad [LV18]. [Mac25] is the only published study of distance-dependent propagation of Galactic
  neutrinos (quasi-Dirac splittings, decay): our use of q_p(s) follows the same idea **(new for
  long-range interactions)**.
- Caveat: if unresolved sources dominate the Galactic flux [Mar25, Pre25], emission follows the source
  distribution, not the gas. A source-distribution variant (Ferriere 2001 [Fer01], as CRINGE uses for
  its sources) is a systematic check.

**The emission potential.** n_eff(x; m) = integral d^3x' (m^2/4pi) exp(-m|x-x'|)/|x-x'| n_n(x') + cosmic
floor, the same Yukawa kernel as the code's `galactic_effective_density`, now at any x. Regimes
(range = hbar c / m_eff; 6.4 kpc at 1e-27 eV): range below ~6 pc (m_eff >~ 1e-24 eV): the contact
limit, the local density at x_e; very long ranges: the Galaxy's share falls as m_eff^2 below the cosmic
mean, so the potential is the same everywhere in the Galaxy and emission inside it changes nothing
(the crossover follows from the tables, test T7); in between the 3-D convolution is needed. The McMillan model is axisymmetric, so n_eff
depends on (R, z) only: computed once per m on a Cartesian grid by FFT convolution and averaged in
azimuth (Tycho), stored as small 2-D tables in log m (Section 5). Test T7: at the Sun's position the
table reproduces `galactic_effective_density(m)` for every m.

**Evaluation per energy.** The eigenstates depend on E and c only through sigma = E c, which the solver
already tabulates (`P_table`, 4096 nodes, `propagate_spectrum` 268-278). Per pixel, the emission
distances are binned in n_eff (a few bins), so the injection is a weighted sum of table lookups; no
new eigen-decompositions.

---

## 4. Software design

### 4.1 New modules

**`src/dp_sky.py` (sky geometry, no physics)**
- `SkyGrid(nside, dec_band_edges)`: HEALPix pixel centres (RA, dec), solid angles, Galactic (l, b),
  ecliptic (lambda, beta), band index per pixel; RING ordering; theta = pi/2 - delta.
- `galactic_to_equatorial(l, b)`, `equatorial_to_galactic(ra, dec)`, `equatorial_to_ecliptic(...)`:
  rotation matrices from the IAU constants above (no runtime dependency on astropy; tests compare
  with astropy where installed).
- `zenith_cos(dec) = -sin(dec)` for the South Pole; detector latitude kept as a parameter for other
  sites (the multi-detector plan, WORK_PLAN item 5).
- `sun_yearly_weights(beta, x_nodes)`: arcsine-law weights in cos(psi), with eccentricity (Kepler).
- `band_edges_default()`: 20 bands uniform in sin(delta) (width 0.1), mergeable into MESE's 10 and
  compatible with HEALPix ring structure checks; user-defined edges allowed.

**`src/dp_galactic.py` (Galactic component)**
- `load_template(name, path)`: CRINGE / pi0 (and KRA_gamma if adopted), reduced to the solver's needs
  (Section 5), returning all-flavor nu + nubar intensity on (pixel, E) in the equatorial frame.
- `emission_distance_bins(skygrid, model)`: per pixel, weights over n_eff bins from q_p(s).
- `neff_table(m)`: interpolation in the precomputed (R, z, log m) tables.
- `galactic_flux_si(...)`, `galactic_flux_dark_portal(...)`: the component per pixel and particle type.

### 4.2 Changes to existing code

| File | Change |
|---|---|
| `dp_spatial.py` | `_earth_chord_worker` returns the chord unitary (and, for compatibility, the evolved state); new `average_earth_chords_bands(...)` with Filon weights restricted to bands; chord nodes placed at band edges (plus optional refinement nodes inside bands); `propagate_earth_spatial(..., return_bands=False)` keeps its current return when False. Fix the stale "up-going" docstrings (810-811, 982-983). |
| `dp_transit.py` | `nearfield_transit` / `source_transit` optionally return per-path states (before the weighted sum at 773-781); `sun_paths` gains Chebyshev nodes for interpolation in cos(psi). |
| `dark_portal_solver.py` | New `propagate_sky(E_bins, params) -> SkyFlux`; `_flux_from_setup` threads a `sky` option; `propagate_spectrum` returns per-band and per-path data when asked; SI branch broadcasts and adds the Galactic SI component. |
| `dp_cosmo.py` | Galactic injection reuses `_cosmo_projections` with c_inj from n_eff bins (no change to the cosmological step itself). |
| `dp_config.py` | New section `sky` (Section 4.3); added to `analysis/model.py` `_SECTIONS` so its keys can be floated (e.g. N_gal). |
| `flux_postprocessing.py` | Helpers to collapse a `SkyFlux` to today's `(N_E, 3)` and to declination bands. |

### 4.3 Configuration (new section `sky`, all defaults keep today's behaviour)

```
"sky": {
  "output": "allsky",            # "allsky" (today), "bands", "healpix"
  "nside": 16,                   # HEALPix resolution of maps (decision D1)
  "dec_band_edges": null,        # null = 20 bands uniform in sin(dec)
  "sun_weighting": "isotropic",  # "isotropic" (today) or "yearly" (decision D4)
  "galactic": {
    "enabled": false,
    "template": "cringe",        # "cringe", "pi0" (decision D2)
    "normalization": 0.91,       # x model; IceCube 2026 combined fit for CRINGE
    "emission": "3d",            # "3d" (Section 3.6) or "local" (decision D3)
    "f_e_S": 0.333333,           # source composition (x, 1 - x, 0); floatable (decision D6)
    "tie_f_e_S": false           # true: use the extragalactic source.f_e_S
  }
}
```

Unknown keys stay rejected (`dp_config.merge_config_with_defaults`). The analysis' configuration
identity hashes the merged config and the source digest, so checkpoints are never reused across this
change (by design, `analysis/config.py` 119-135).

### 4.4 Output object and files

`SkyFlux` (dataclass): `E_eV (N_E,)`; `bands_edges (N_b+1,)`; `extragalactic_bands (N_b, N_E, 3, 2)`
[band, energy, flavor, nu/nubar]; optionally `extragalactic_pixels (N_pix, N_E, 3, 2)` when the Sun is
resolved; `galactic_pixels (N_pix, N_E, 3, 2)`; `allsky (N_E, 3)` (exactly today's output); metadata:
frame "equatorial J2000", ordering "RING", theta convention "pi/2 - dec", nside, units
"GeV^-1 cm^-2 s^-1 sr^-1 per particle type", commit, config digest. Written as `.npz` (no new
dependency), readable by the HESE reweighting (Phi_type(E, delta, RA) for each MC event).

---

## 5. Data acquisition and preparation (Tycho only)

Downloads go to Tycho, never to the laptop (standing rule). Each item: source, expected size (to be
confirmed from the record before downloading), what we keep.

| Data | Source | Kept in the repository |
|---|---|---|
| CRINGE neutrino map `Neutrino_AAfrag_Galprop_Ferr_Fiducial_256.fits.gz` (174.0 MB) and 3-D CR distribution `Global_Hadronic_CR_Dist_CRINGE.fits.gz` (96.6 MB); optional unresolved-source map `Neutrino_Unresolved_Fiducial_256.fits.gz` (203.0 MB) | Zenodo 7859442 v3, CC BY 4.0 [SMW23] (sizes read 2026-10-13) | reduced map: N_side 16-64, energies 1 TeV-10 PeV (~20 log nodes), equatorial frame, float32: ~1-8 MB; a CR table cut to the 1-100 PeV parent range |
| 3-D gas (HI + H2 via CO): `samples_densities_hpixr.fits` (1.2 GB; Sun-centred HEALPix (nested) x radius, 50 pc-28 kpc, cm^-3, posterior samples); the whole record is 26.8 GB and only this file is needed | Zenodo 12578443 v1, CC BY 4.0 [Sod25] | per-pixel emission-distance bins only (small). No-download alternative: the McMillan 2017 HI and H2 discs already in `calc_MW_neutron_density` [McM17] (smooth, axisymmetric) |
| pi0 map | `../toise` (on disk already) [Fer12] | reduced as above |
| IceCube Galactic-plane release | doi:10.21234/8v5d-rn16 (26 MB) [GP23] | nothing; used to settle the per-flavor normalization convention and to cross-check template fluxes |
| n_eff(R, z; m) tables | computed on Tycho by FFT | 2-D tables, ~20 masses, ~3 MB |

`tools/prepare_galactic.py` does all reductions with provenance (DOI, file digests, commit) in the
output's metadata, like `tools/mw_sky_density.py` does for the Galactic map today.

---

## 6. Implementation steps (each lands with its tests, docs and a notes section)

1. **Sky geometry** (`dp_sky.py`): grids, rotations, bands, Sun yearly weights. Tests T5, T4a.
2. **Earth per band**: chord unitaries; band Filon weights; nodes on band edges; `return_bands`. Tests
   T1, T2, T3. Notes: extend the zenith appendix.
3. **SkyFlux API + config** (`propagate_sky`, section `sky`, `.npz` writer), extragalactic only, Sun
   isotropic. Test T1 at the API level; backward-compatibility run of the full suite.
4. **Sun per direction** (only if decision D4 says so): per-path states, Chebyshev interpolation,
   yearly weights; anisotropy diagnostic in the Sun window. Tests T4, T4b.
5. **Galactic template** (CRINGE, pi0): Tycho preparation, loading, flux per pixel, SI component.
   Tests T6, T8.
6. **Dark-portal Galactic emission**: n_eff tables (Tycho), emission-distance bins, injection via
   sigma tables, adiabatic transit, Sun and Earth. Tests T7, T9, T10.
7. **Validation tools and figures** (`tools/sky_validation.py` on Tycho): sky maps per flavor, band
   profiles, the Galactic fraction versus energy, the anisotropy induced by each stage; notes section
   "Directions and Galactic sources".
8. **Consumer check**: reweight the HESE MC with a `SkyFlux` (Phi per particle type at
   (E, delta)) and recover the standard isotropic event count with the default physics (a units and
   conventions test, not an analysis).

---

## 7. Tests and validations

| ID | What | Expected |
|---|---|---|
| T1 | Bands (and pixels) averaged over the sky with solid-angle weights versus today's all-sky output, at the four benchmark points (below, inband, strong, cosmoX2) | equal to 1e-12 |
| T2 | Each band versus a brute-force dense chord average inside the band (e.g. 2000 chords per band) | 1e-4 or better; convergence with nodes per band |
| T3 | Chord unitary applied to the input versus today's evolved chord state | rounding |
| T4 | Sun: the pixel-weighted sky average of the yearly weights equals the isotropic Gauss-Legendre weights; arcsine-law moments | exact; analytic |
| T4b | Sun anisotropy outside the window | zero (transit inactive) |
| T5 | Coordinates: Galactic Centre at (266.40 deg, -28.94 deg); poles; round trips; agreement with astropy | 1e-9 rad |
| T6 | Template: all-sky per-flavor flux at 100 TeV and the |b| profile against the published model values [SMW23], and against IceCube's fitted flux when multiplied by N_gal [IC26, GP23]; the inner-Galaxy region (|b| < 15 deg, |l| < 20 deg) | to the precision the publications quote |
| T7 | n_eff table at the Sun's position versus `galactic_effective_density(m)`; contact limit (local density) and long-range limit (same as at the Sun) | 1e-6; exact limits |
| T8 | Galactic SI component equals free propagation with averaged |U|^2 | exact |
| T9 | Adiabaticity of the Galactic transit from emission points (Landau-Zener parameter along representative lines of sight) | >> 1 except where flagged |
| T10 | A Galactic source placed at the Sun's position reproduces the host-galaxy path with host_galaxy_scale = 1 | rounding |
| T11 | u -> 0 (no dark portal): every band and pixel equal (isotropy) | rounding |
| T12 | Full existing suite and the analysis unchanged with `sky.output = "allsky"` | bit-identical |

---

## 8. Cost

- Earth bands: free when band edges are chord nodes (weights plus one tensor product per band).
  Nodes: today 23; with 20 bands and one refinement node per band, ~41 chords, i.e. ~1.8x the Earth
  phase (the Earth phase was ~2/3 of an evaluation in 2026-10-12 measurements). Measured before
  choosing the default (test T2 sets the minimum nodes per band).
- Unitaries: memory N_E x K x 9 complex numbers, ~2.4 MB at 400 energies, 41 chords.
- Sun per direction: only in the Sun window; 16-32 paths instead of 8 (the transit costs 0.2-1 ms per
  path and energy bin, ~6 s per evaluation at 8 paths and 400 bins today), so ~12-24 s per evaluation
  there. This is why decision D4 asks for a diagnostic first.
- Galactic, dark portal: per pixel x n_eff bins x energies table lookups: 3072 x 8 x 400 ~ 1e7
  lookups per evaluation (well under a second in numba); the n_eff tables are a one-off Tycho job.
- Galactic, SI: negligible.

---

## 9. Decisions for Mauricio

**Answered 2026-10-13:** D1 = bands + N_side 16; D2 = CRINGE + pi0; D3 = full 3-D; D4 = diagnostic first.
Added by Mauricio: the Galactic source composition varies as (x, 1 - x, 0) (Section 3.4).
**Answered later the same day:** D5 = all downloads on Tycho (CRINGE map, CR distribution and
unresolved-source map; the 1.2 GB 3-D gas file; IceCube's Galactic-plane release); D6 = x_gal floats
on its own.

- **D6, the Galactic x.** Float x_gal on its own, default 1/3 (Recommended: Galactic and extragalactic
  sources need not share a composition; the event data can separate them by direction); or tie it to
  the extragalactic f_e,S (one parameter fewer).

- **D1, sky resolution.** Declination bands only (20, uniform in sin(delta)) plus HEALPix maps at
  N_side 16 (Recommended: cascades resolve ~6-10 deg, the Galactic plane is a few degrees thick, and
  N_side 16 pixels are 3.7 deg); or N_side 32/64 for tracks.
- **D1 addendum (Mauricio, 2026-10-14): resolution per consumer.** HESE can use coarser pixels than DR2 tracks.
  The flux map must resolve the true sky structure (the plane is a few degrees thick) to the extent each
  sample's angular resolution can see it: HESE (mostly cascades, 10-20 deg) at N_side 8 (7.3 deg), DR2 tracks
  (~1 deg) at N_side 64-128. The stored templates and emission data (N_side 16) degrade exactly to 8, 4, 2, 1
  (N_side 8 is the coarsest keeping a pixel centre in each of the 20 default bands; N_side 4 works with MESE's
  10 bands). For tracks: finer reductions on Tycho (prepare_galactic.py --nside 64 128; CRINGE is native
  N_side 256, IceCube's pi0 128, the Soding gas 64, so emission distances stop at 64), kept outside the
  repository (tens of MB); DR2's 41 declination bins as band edges; the 3-D emission populations (N_pix x N_E)
  restricted to the declinations a sample uses.
- **D2, Galactic templates.** CRINGE primary plus pi0 (Recommended); add KRA_gamma (two 807 MB pickled
  files, loaded only in isolation); or pi0 only (already on disk, no downloads).
- **D3, dark-portal emission points.** Full 3-D (CRINGE cosmic rays x Soding et al. gas; FFT tables)
  (Recommended); or every Galactic neutrino emitted at the local potential first, 3-D later.
- **D4 outcome (2026-10-13):** the yearly per-direction Sun weighting changes per-pixel energy-averaged
  flavor fractions by <= 1.2e-3 (twelve points across the Sun's window; tools/sun_anisotropy.py):
  the Sun stays isotropic; single-path transits (sun_path_cos) and dp_sky.sun_path_weights remain.
- **Step 5 outcome (2026-10-13): the templates and their conventions, checked against IceCube.** Reduced
  on Tycho (tools/prepare_galactic.py; src/data/galactic_cringe.npz, galactic_pi0.npz; N_side 16,
  72 GeV-1e8 GeV). Frames read off the maps: CRINGE is Galactic (55.7% of the 100-TeV flux at |b| < 5 deg,
  4.9% if read as equatorial); IceCube's pi0 template is equatorial; toise's copy of it is Galactic and,
  rotated by dp_sky, correlates 0.997 with IceCube's. Conventions: per flavor, nu + nubar, for both
  (CRINGE's record; its Eq. 12 is the pp per-flavor relation; IceCube's README); the pi0 model flux is
  E^2 Phi(100 TeV) = 4.4e-12 TeV cm^-2 s^-1 sky-integrated per flavor ([IC26] supplement; the 2023 fit
  21.8e-12 is 4.95 times it, so [GP23] and [IC26] share the convention, settling the trap of Section 10).
  **Finding: IceCube's "CRINGE" is the diffuse map plus CRINGE's unresolved-source map**
  (Neutrino_Unresolved_Fiducial_256): times 0.91 the sum lies on [IC26] Fig. 4's inner-Galaxy curve to
  0.1-0.25% from 3 to 37 TeV, the diffuse map alone at 28-54% of it; pi0 x 4.99 lies on its curve to
  0.47% (Fig. 4 read from its SVG path coordinates: src/data/ic26_fig4_digitized.json; test T6 in
  tests/test_galactic.py). The unresolved sources carry 75% of CRINGE's sky-integrated flux at 100 TeV.
  **Consequence for step 6:** the emission distances must mix two distributions per pixel and energy:
  gas x cosmic rays for the diffuse share, the source distribution for the unresolved share (CRINGE's
  unresolved sources follow Ferriere 2001 [Fer01]; to be read from the CRINGE paper before step 6), the
  latter dominating above ~30 TeV -- the "unresolved-source variant" of Section 3.6 becomes the main
  term, not a systematic. Implemented: dp_galactic (templates, interpolation, composition (x, 1 - x, 0),
  averaged transfer), propagate_sky's Galactic component (secret interactions: unattenuated, averaged
  oscillations; dark portal, emission 'local': emitted at the local potential, decohered into the local
  eigenstates by dp_transit.flavor_populations, then the Sun and the Earth per band). Checks: the
  transfer conserves the flux to 3e-14; the vacuum limit is the averaged oscillation to 1e-12 with the
  Earth's standard matter potential off, and with it on moves the bands through the Earth by the MSW
  effect on the arriving mass-state mixture (2.4e-5 at 10 TeV through the core, falling as 1/E).
- **Step 6 (2026-10-14): emission points in 3-D.** n_eff(R, z; m) at any emission point
  (tools/mw_neff_tables.py, src/data/mw_neff_tables.npz): sphere averages of the solver's Galactic model around
  48 x 22 nodes (R graded toward the Centre, z toward the plane), integrated on the solver's own distance nodes
  for 20 masses per decade from 1e-34 to 1e-12 eV; T7: at the Sun's node it equals galactic_effective_density
  to 1e-9 (1.5e-14 on the first run), and the sphere average equals mw_sky_density's quadrature to 2e-15.
  Between nodes: log-linear in m, and in (R, z) with the exact local density blended in at short ranges.
  **Finding:** at ranges far beyond the Galaxy n_eff is not uniform -- the Galaxy's share is m^2 times its
  neutrons' 'Coulomb' potential at the point, ~6 times deeper at the Centre than at the Sun, and it beats the
  cosmic floor down to ~4e-31 eV (the first table, which stopped at 1e-30 eV and assumed uniformity below,
  was extended). Emission distances (tools/prepare_galactic.py --emission, src/data/galactic_emission.npz):
  diffuse = Soding et al. hydrogen columns x CRINGE's nucleon flux at 20 x 100 TeV (morphology vs CRINGE's
  diffuse map: correlation 0.991); unresolved = Lorimer et al. 2006 with exp(-|z|/0.2 kpc) (Vecchiotti et al.
  2022, CRINGE's choice) weighted by distance s (dN/dL ~ L^-1.5 with a flux threshold removes the nearby
  sources; s^1 fits CRINGE's map better than s^0 or s^0.5), correlation 0.985 (CRINGE's map is broader at
  high latitudes and in the outer Galaxy, where its resolution threshold is presumably survey-dependent;
  Case & Bhattacharya with H = 0.33 kpc is the systematic variant). Mean emission distance toward the Centre:
  5.7 kpc (gas), 9.8 kpc (sources); toward the anticentre 0.9 and 3.7 kpc. Per pixel and energy the two mix
  as CRINGE's maps do. Populations: emission densities deposited on log-n nodes 20 per decade (cloud in
  cell), each node's populations from flavor_populations on the kind's frame table; per band the response to
  each local eigenstate (three spatial runs per kind). **Fix found by T10:** 'local' emission had computed its
  populations at n_bg = n_gal + n_moon; Galactic neutrinos are born at the Galaxy's potential alone (the
  Moon's Yukawa-smoothed density at the Earth is ~13% of the Galaxy's at a 6-pc range), and the map then
  carries them onto the eigenstates at n_bg. T10: a Galactic source at the Sun's position and the host-galaxy
  path (host_galaxy_scale = 1, z -> 0) agree to < 1e-5 in every band and energy (their potentials differ by
  3e-8, the cosmic mean the host path adds; up to 1.2e-3 at 1e-31-1e-29 eV, where the halo meets the cosmic
  mean: the local potential takes the larger, the host path sums them).
- **Step 7 (2026-10-14): the pieces together** (tools/sky_validation.py, output/figures/sky_validation.png).
  sky.extragalactic_norm puts the extragalactic flux in physical units (its per-flavor intensity at 100 TeV
  without new physics; closed-form free streaming, which the solver matches to 4e-3 at 400 redshift steps).
  With HESE 7.5-yr (2.12e-18, gamma 2.87) the Galactic share (CRINGE x 0.91) is 4/6/1% of the sky-averaged
  flux at 10 TeV/100 TeV/1 PeV, 22/35/7% in the inner Galaxy; at 20 TeV the sky-averaged CRINGE flux is 11% of
  IceCube's 2026 astrophysical flux (they state ~10%). HESE's soft fit is 4x the 2026 flux at 10 TeV (the
  known tension with broken power laws), so the share depends on that choice. At m_Zd = 1e-24 eV, kappa = 300
  the f_e spread at 100 TeV over the pixels is 0.018 from the Galactic component (0.014 in the total), 1e-6
  from the Earth, <= 1.2e-3 from the Sun's yearly path.
- **D4, the Sun per direction.** Keep the isotropic Sun average now and measure the anisotropy it would
  produce in its window (Recommended); or implement the yearly per-direction weighting now.
- **D5, downloads on Tycho.** CRINGE (Zenodo 7859442), the 3-D gas (Zenodo 12578443) and IceCube's
  Galactic-plane release (26 MB): approve once the sizes are read from the records.

---

## 10. Unverified items and risks

- ~~The per-flavor normalization convention of [GP23]~~ settled 2026-10-13: per flavor, nu + nubar, as
  the release notebook states; [IC26] Fig. 4 is labelled the same way and our templates reproduce it
  (Section 9, step 5 outcome).
- ~~CRINGE map frame~~ Galactic (read off the map, step 5). The URL of IceCube's 2026 template release
  [IC26] was not found; its CRINGE is diffuse + unresolved sources (step 5 outcome).
- Whether the Fermi 2012 supplement [Fer12] is still online (the toise copy covers the pi0 map).
- Whether the DR2 / point-source effective areas include Earth absorption explicitly ([PS10] defines
  rates with A_eff but does not say; [HESE21] does for HESE); needed for the event-rate step, not here.
- Emission distances: no published per-pixel distribution exists; ours depends on the gas and
  cosmic-ray models; the unresolved-source variant brackets it.
- Adiabaticity near level crossings at emission (test T9).
- Side finding, not part of this plan: `../NuAnisotropy`'s on-disk fit script
  (`ultranest_fitting_script.py`) swaps the angle arguments of `sph_harm` (L107), reads wrong parameter
  names so the conventional and prompt normalizations stay at zero (L226-228), and reweights from
  `astrophysical_weights` made with a different reference flux (gamma = 2.5 instead of 2.89); the
  published analysis used other code, not on disk.

---

## 11. References

- [ABDN23] Agarwalla, Bustamante, Das, Narang, JHEP 08 (2023) 113, arXiv:2305.03675.
- [AdWFJ17] Arguelles, de Wasseige, Fedynitch, Jones, JCAP 07 (2017) 024, arXiv:1703.07798.
- [Ans04] Anchordoqui et al., PLB 593, 42 (2004).
- [Auger19] Pierre Auger Collaboration, JCAP 11 (2019) 004.
- [BA19] Bustamante, Agarwalla, PRL 122, 061103 (2019), arXiv:1808.02042.
- [Bla60] Blaauw, Gum, Pawsey, Westerhout, MNRAS 121, 123 (1960).
- [dS17] de Salas, Gariazzo, Lesgourgues, Pastor, JCAP 09 (2017) 034, arXiv:1706.09850.
- [DSS26] Dighe, Sahoo, Sen, arXiv:2607.24918.
- [E23] Elbers et al., JCAP 10 (2023) 010, arXiv:2307.03191.
- [EEEN17] Edsjo, Elevant, Enberg, Niblaeus, JCAP 06 (2017) 033, arXiv:1704.02892.
- [EPBB21] Esteban, Pandey, Brdar, Beacom, PRD 104, 123014 (2021), arXiv:2107.13568.
- [Fer01] Ferriere, RMP 73, 1031 (2001).
- [Fer12] Fermi-LAT (Ackermann et al.), ApJ 750, 3 (2012), arXiv:1202.4039.
- [FS08] Farzan, Smirnov, NPB 805, 356 (2008), arXiv:0803.0495.
- [Gag15] Gaggero et al., ApJL 815, L25 (2015), arXiv:1504.00227; maps doi:10.5281/zenodo.7070823.
- [Gor05] Gorski et al., ApJ 622, 759 (2005), astro-ph/0409513.
- [GP23] IceCube, Science 380, 1338 (2023), arXiv:2307.04427; data doi:10.21234/8v5d-rn16.
- [HESE21] IceCube, PRD 104, 022002 (2021), arXiv:2011.03545.
- [IC19] IceCube, EPJC 79, 234 (2019).
- [IC26] IceCube, "High-energy neutrino emission from the Milky Way", arXiv:2607.25966.
- [ICsun] IceCube, JCAP 02 (2021) 025, arXiv:1912.13135.
- [KW05] Kashti, Waxman, PRL 95, 181101 (2005).
- [Liu11] Liu, Zhu, Zhang, A&A 526, A16 (2011).
- [LP95] Learned, Pakvasa, Astropart. Phys. 3, 267 (1995).
- [LV18] Lipari, Vernetto, PRD 98, 043003 (2018), arXiv:1804.10116.
- [Mac25] MacDonald et al., "Exploring new propagation scales with Galactic neutrinos", arXiv:2512.10744.
- [Mar25] Marinos et al., arXiv:2511.09777.
- [MB13] Miller, Bregman, ApJ 770, 118 (2013).
- [McM11] McMillan, MNRAS 414, 2446 (2011).
- [McM17] McMillan, MNRAS 465, 76 (2017), arXiv:1608.00971.
- [MP23] Mertsch, Phan, A&A 671, A54 (2023), doi:10.5281/zenodo.5956696.
- [MPdS20] Mertsch et al., JCAP 01 (2020) 015, arXiv:1910.13388.
- [MPH15] McKee, Parravano, Hollenbach, ApJ 814, 13 (2015), arXiv:1509.05334.
- [MV21] Mertsch, Vittino, A&A 655, A64 (2021), doi:10.5281/zenodo.5501196.
- [Par86] Parke, PRL 57, 1275 (1986).
- [Pre25] Prevotat et al., arXiv:2507.10823.
- [PS10] IceCube, "IceCube data for neutrino point-source searches: years 2008-2018", arXiv:2101.09836.
- [RW04] Ringwald, Wong, JCAP 12 (2004) 005, hep-ph/0408241.
- [SMW23] Schwefer, Mertsch, Wiebusch, ApJ 949, 16 (2023), arXiv:2211.15607; doi:10.5281/zenodo.7859442.
- [Sod25] Soding et al., A&A 693, A139 (2025), arXiv:2407.02859; doi:10.5281/zenodo.12578443.
- [Spi26] Spiering, arXiv:2609.26436.
- [ZCA23] Zimmer, Correa, Ando, JCAP 11 (2023) 038, arXiv:2306.16444.
