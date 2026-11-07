# Paper planning

Working title (2026-10-21): *New neutrino interactions with high-energy astrophysical neutrinos*.
Earlier working title (2026-10-14): *New forces on high-energy cosmic neutrinos, from the Hubble scale to the Earth*.

Each step ends with Mauricio's sign-off, recorded in the decision log at the end, before the next step starts.

| Step | What | State |
|---|---|---|
| 0 | Is the paper worth writing, at what pitch; competing results | done 2026-11-07 |
| 1 | Main message and main result | **draft, for sign-off** |
| 2 | Structure | **draft, for sign-off** |
| 3 | Equations plan | **draft, for sign-off** |
| 4 | Citations plan | **draft, for sign-off** |
| 5 | Figures plan | **draft, for sign-off** |
| 6 | Figures, LaTeX skeleton, spine, prose | **started**: equation plan typeset, skeleton, spine, refs |

---

## Step 0. Is the paper worth writing, and at what pitch

### What the paper computes (checked against the campaign configurations)

High-energy astrophysical neutrinos feel a new force carried by a light mediator on their way from the sources to
IceCube. The calculation follows them through the Universe, the Milky Way, the Sun and the Earth. Along the way,
the mediator does two things:

- It creates a **coherent potential** sourced by matter (or by the relic neutrinos), which changes the neutrinos'
  flavor in flight.
- It lets them **collide with the relic neutrinos**: resonant absorption with regeneration, neutrino decay, and
  scattering.

Models, as run (planes in `analysis/configs/campaign_ml`, `campaign_final`, `campaign_si`):

- **The dark portal.** The mediator couples to neutrinos through their mixing with a sterile neutrino, in one of
  three flavors ($e$, $\mu$, $\tau$). Ultralight and MeV masses.
- **Gauged $L_e - L_\mu$ and $L_e - L_\tau$.** The potential is sourced by electrons. Ultralight and MeV masses.
- **Secret interactions.** Scalar or vector mediator, Majorana or Dirac neutrinos, coupled to $\nu_e$, $\nu_\mu$,
  $\nu_\tau$ or to all flavors equally: 16 models. Not run yet; on hold until the physics below 1 keV exists
  (decision of 2026-10-25).

Data sets, each fitted on its own:

- **Today, flavor + spectrum:** IceCube's MESE flavor composition (2025) and its 2026 spectrum.
- **Today, events:** IceCube's HESE starting events and DR2 through-going tracks.
- **Projection:** future neutrino telescopes, 10 years each.

Source flavor compositions: fixed to pion decay, or fitted.

### What the paper may add (candidate claims; NONE checked against the literature yet)

1. The coherent potential and the collisions on relic neutrinos computed together, along the whole trip from the
   sources to the detector, for several families of models.
2. Several current IceCube data sets, each with a full likelihood: flavor composition, spectrum, and event rates,
   with systematic uncertainties and fitted source compositions.
3. Limits that span 39 decades in mediator mass, from $10^{-30}$ to $10^{9}$ eV (the $L_e - L_\mu$ and $L_e - L_\tau$
   planes), in one framework.
4. Possibly the first limits on the dark portal from high-energy neutrinos.

Each claim must be checked against the competing results below before it goes into the text.

### Competing results

For each result: the probe, the bound and its confidence level, the mass range, and how it was checked: **text**
(read in the paper), **abstract** (abstract only), or **title** (seen in a citation list, not read). INSPIRE texkeys
in brackets. Papers read are in the session scratchpad, not in the repository.

#### Family 1: $L_e - L_\mu$ and $L_e - L_\tau$ (checked 2026-11-07)

Ultralight mediators:

| Work | Probe | Result | Checked |
|---|---|---|---|
| Bustamante & Agarwalla, PRL 122 (2019), arXiv:1808.02042 [Bustamante:2018mzu] | High-energy flavor composition; cosmological electrons | Proof-of-principle sensitivity (1 sigma) from IceCube's 2015 flavor measurement | text of ABDN23 (Fig. 1 caption) |
| Agarwalla, Bustamante, Das & Narang, JHEP 08 (2023) 113, arXiv:2305.03675 [Agarwalla:2023sng] | High-energy flavor composition; electrons and neutrons of the Earth, Moon, Sun, Milky Way and cosmos | 95% C.L. (Bayesian) limits from an **estimated** IceCube 8-year flavor measurement, not real data; values in its Table F1 (not read yet). Its own caveats (Sec. 5): the potential is computed **at the detector**, not along the trajectory; source redshift distribution neglected; spectral index fixed to 2.5 | text |
| Agarwalla, Bustamante, Das & Narang, arXiv:2501.12662 [Agarwalla:2025jbn] | Same, present and future | Sensitivity study; no journal listed (proceedings?) | abstract |
| Garg, Krishnamoorthi, Kumar & Agarwalla, arXiv:2601.01220 [Garg:2026gwx] | IceCube DeepCore, 8 years of public atmospheric $\nu_\mu$ data | 90% C.L.: $V_{e\mu} < 8.9\times10^{-15}$ eV, $V_{e\tau} < 9.2\times10^{-15}$ eV (their Eq. 7); masses below $10^{-10}$ eV; they call it world-leading | text |
| Garg et al., arXiv:2606.03443 [Garg:2026rmy] | Same data, many $U(1)'$ models | Limits on potentials, coupling and mass | abstract |
| Coloma, Gonzalez-Garcia & Maltoni, JHEP 01 (2021), arXiv:2009.14220 [Coloma:2020gfv] | Global oscillation fit (solar, atmospheric, reactor, accelerator) | Ranges from the Sun-Earth distance to contact interactions; beats gravity tests below $\sim10^{-8}$ to $10^{-11}$ eV, depending on the model | abstract |
| Joshipura & Mohanty 2004 (hep-ph/0310210); Bandyopadhyay, Dighe & Joshipura 2007 (hep-ph/0610263) | Atmospheric; solar + KamLAND | Older oscillation limits | title (as cited by ABDN23, Garg) |
| Fifth-force and equivalence-principle tests (Eöt-Wash, MICROSCOPE, lunar laser ranging); Wise & Zhang 2018 reinterpretation | Gravity tests | To identify and read | -- |
| Chauhan & Xu, JHEP 07 (2024), arXiv:2403.09783 [Chauhan:2024qew] | Screening by the relic neutrinos | Relic screening removes the lunar-laser-ranging and Sun-attractor bounds for neutrino couplings above $3\times10^{-10}$ and $5\times10^{-13}$ | abstract |
| Black-hole superradiance (90% C.L.); weak gravity conjecture | Indirect | As shown in ABDN23's Fig. 1 | text of ABDN23 (caption) |
| **Dror, PRD 101 (2020), arXiv:2004.04750 [Dror:2020fbh]** | Neutrino masses break $L_e - L_\mu$; the mediator's longitudinal mode couples like a scalar, enhanced by 1/mass: meson decays, invisible neutrino decays, neutrinoless double-beta decay, BBN and supernovae | Strongest bounds for masses below $\sim10^{-18}$ eV (standard cosmology) or $\sim10^{-24}$ eV (modified cosmology) (conclusions; the introduction says $10^{-22}$ to $10^{-17}$ eV, depending on assumptions; the extracted text is garbled here, to recheck in the PDF). His conclusion: long-range forces on galactic scales are then not observable | text |
| Dighe, Sahoo & Sen, arXiv:2607.24918; Kumar Poddar, Mohanty & Jana, arXiv:2002.02935 | Supernova neutronization burst; planetary perihelion precession | -- | title |
| DUNE, T2HK, ESSnuSB, P2SO forecasts (e.g., Singh, Bustamante & Agarwalla, arXiv:2305.05184) | Future long-baseline | Forecasts | title |

MeV mediators:

| Work | Probe | Result | Checked |
|---|---|---|---|
| Bauer, Foldenauer & Jaeckel, JHEP 07 (2018), arXiv:1803.05466 [Bauer:2018onh] | Compilation: hidden-photon searches translated to $L_e - L_\mu$, $L_e - L_\tau$; neutrino experiments; astrophysics; cosmology; white-dwarf cooling | Mass range and values to read from its figures | abstract |
| Sabti, Alvey, Escudero, Fairbairn & Blas, JCAP 01 (2020), arXiv:1910.01649 [Sabti:2019mhn] | BBN + CMB | A thermalized electrophilic or neutrinophilic particle must be heavier than 3.7 MeV (2 sigma); whether our couplings thermalize: to check | abstract |

What family 1 says for Step 0:

1. **The propagation claim holds against the closest work.** ABDN23 states that it computes the potential at the
   detector, not along the trajectory, and neglects the source distribution in redshift. How the other high-energy
   works in the citation lists treat the propagation is not yet checked (titles only).
2. **Real data.** ABDN23 used an estimated measurement; we fit real IceCube data (MESE flavor and spectrum; HESE
   and DR2 events). Garg et al. (2026) fit real data too, but atmospheric (DeepCore), at lower energies.
3. **Ultralight: checked 2026-11-07, against our preliminary figure** (`paper/figures/limits_le_lmu_le_ltau_pion.png`,
   from campaign_ml, before the final run; values read from figures to about 0.2 decade):
   - Our line of perturbative unitarity is $g' = 10^{1.76} M'/{\rm eV}$ for $L_e - L_\mu$ ($10^{1.75}$ for $L_e - L_\tau$;
     `analysis.exclusion.not_allowed_boundary`, lightest mass $10^{-6}$ eV). The region excluded by MESE flavor +
     spectrum starts at $g' \approx 10^{2.1} M'/{\rm eV}$: **it lies entirely where the model is not perturbative.**
   - The HESE + DR2 band reaches $g' \approx 10^{-28}$ for $M'$ between $10^{-28}$ and $10^{-20}$ eV. Bustamante &
     Agarwalla's 2019 sensitivity (1 sigma, estimated), which Dror plots as the oscillation limit, reaches
     $\approx 2\times10^{-28}$ over the same masses. Ours is comparable, now at 95% C.L., with real data and the full
     propagation.
   - Dror's Fig. 3 (Majorana neutrinos): neutrino decays in the early Universe exclude $g'/M' \gtrsim 10^{-8.8}$
     eV$^{-1}$ if the neutrinos are present at recombination (standard cosmology). That covers our HESE + DR2 band
     below about $10^{-15}$ eV. Without that assumption, his strongest line is BBN/supernovae at
     $g'/M' \gtrsim 10^{-3}$ eV$^{-1}$, which leaves our band standing. Dirac neutrinos (his Fig. 4): not read yet.
4. **MeV: checked 2026-11-07.** Bauer et al.'s Figs. 14 ($L_\mu - L_e$) and 15 ($L_e - L_\tau$) start at 2.4 MeV.
   From there up, current experiments (beam dumps E137, U70, NuCal, E141, LSND; Borexino, TEXONO, Super-K; electron
   $g-2$, NA64, KLOE, BaBar; white dwarfs) exclude every coupling from $10^{-8}$ to the figures' top, $2\times10^{-3}$. **Our MeV
   region ($10^{-4}$ to $10^{-2}$) lies inside it** (above $2\times10^{-3}$, outside the figures: to check). Below 2.4 MeV the figures do not reach; BBN + CMB exclude a
   thermalized boson lighter than 3.7 MeV (Sabti et al.), and couplings this large thermalize (my estimate, not
   checked).
5. **Net for family 1:** our $L_e - L_\mu$ and $L_e - L_\tau$ limits do not improve on existing ones at MeV masses,
   and at ultralight masses they match the earlier high-energy sensitivity, with real data, and survive Dror's
   bounds only under his non-standard assumptions. Their value for the paper is as a showcase of the propagation,
   not as leading limits.

#### Family 2: the dark portal (checked 2026-11-07)

Our dark-portal planes: MeV masses ($10^4$ to $10^9$ eV; regions excluded up to $10^{7.2}$ eV, couplings
$g_D|U_{\alpha4}|^2$ from $10^{-5}$ to $10^{-2}$; `notebooks/paper_figures.ipynb`, Sec. 2), and ultralight masses
without the $Z$ mixing (the potential planes were dropped 2026-10-24), where the limits come from neutrino decay
(preliminary $\nu_\mu$ plane, progress slides: every coupling $\geq 10^{-8}$ at $10^{-8}$ to $10^{-6}$ eV,
$\geq 1.8\times10^{-8}$ at $10^{-4}$ eV, $\geq 1.8\times10^{-5}$ at $10^{-2}$ eV).

| Work | Probe | Result | Checked |
|---|---|---|---|
| Jana & Swain, arXiv:2607.04441 [Jana:2026pwx] | DUNE, 10 years: the dark portal's potential (electrons and neutrons of the Earth to the cosmos; $Z$ mixing) | Our model. 3 sigma sensitivities on the potential, the coupling, the self-interaction strength (Hubble-tension band) and $\vert U_{\alpha4}\vert^2$. Quotes the mixing bounds $\vert U_{e4}\vert^2 \lesssim 8.8\times10^{-9}$ (PIENU), $\vert U_{\mu4}\vert^2 \lesssim 6\times10^{-6}$ (KEK), $\vert U_{\tau4}\vert^2 \lesssim 7.4\times10^{-4}$ (CHARM), at its benchmark $m_4 = 100$ MeV. Competes only with the potential planes we dropped | text (Figs. 6, 9 captions) |
| Blinov, Kelly, Krnjaic & McDermott, PRL 123 (2019), arXiv:1905.02727 [Blinov:2019gcj] | Neutrino self-interactions: cosmology and laboratory | A mediator with neutrino coupling $> 10^{-4}$ and mass keV to 100 MeV is excluded unless it is a scalar coupled almost only to $\nu_\tau$ | abstract |
| Berryman et al., Phys. Dark Univ. 42 (2023), arXiv:2203.01955 [Berryman:2022hds] | White paper: all probes of neutrino self-interactions | Compilation; figures to read for flavor-specific vector bounds | abstract |
| Brdar, Lindner, Vogl & Xu, PRD 101 (2020), arXiv:2003.05339 [Brdar:2020nbj] | $Z$ and $\tau$ decays | Scalar mediators; leading above 300 MeV for $\nu_\tau$ | abstract |
| Sabti et al., JCAP 01 (2020), arXiv:1910.01649 | BBN + CMB | Thermalized neutrinophilic particle heavier than 3.7 MeV (2 sigma) | abstract |
| Escudero & Fairbairn, PRD 100 (2019), arXiv:1907.05425 [Escudero:2019gfk] | CMB (Planck 2018), BBN | Invisible neutrino decay: lifetime $\tau_\nu > (1.3$--$0.3)\times10^9\,{\rm s}\,(m_\nu/0.05\,{\rm eV})^3$ (95% C.L.); BBN: $> 10^{-3}$ s | abstract |
| Bustamante, Beacom & Murase, PRD 95 (2017), arXiv:1610.02096 [Bustamante:2016ciw] | IceCube diffuse flux, decay | Sensitivity $\tau \sim 10\,{\rm s}\,(m/{\rm eV})$ for all mass states | abstract |

What family 2 says for Step 0:

1. **MeV:** our region (vector, couplings $10^{-5}$ to $10^{-2}$, 10 keV to 16 MeV) lies where Blinov et al. exclude
   vectors above $10^{-4}$, and where BBN + CMB exclude a thermalized neutrinophilic boson below 3.7 MeV. Not
   yet checked: the strip between $10^{-5}$ and $10^{-4}$ above 3.7 MeV, for each texture (white-paper figures).
   In addition, with $g_D \leq \sqrt{4\pi}$ the quoted mixing bounds cap $g_D\vert U_{\alpha4}\vert^2$ at about
   $3\times10^{-8}$ ($e$), $2\times10^{-5}$ ($\mu$) and $3\times10^{-3}$ ($\tau$) for $m_4 = 100$ MeV (my arithmetic;
   the bounds depend on $m_4$).
2. **Ultralight (decay):** IceCube sees a decay only if the lifetime is shorter than about $10$ to $10^3$ s per eV of
   mass (Bustamante et al. 2017). The CMB requires lifetimes above $10^9$ s at $m_\nu = 0.05$ eV, that is
   $\sim 10^{10}$ s/eV. So every coupling whose decays IceCube can see is excluded by the CMB, by many orders of
   magnitude, if the decays proceed in the early Universe as in vacuum (my argument from the two abstracts).
3. **Net for family 2:** as for family 1, our dark-portal limits do not improve on existing ones. One angle that
   survives: the high-energy limits do not depend on the early Universe, so they hold where a modified cosmology
   evades the CMB and BBN bounds.

#### Family 3: secret interactions (checked 2026-11-07)

Our secret-interaction planes are not run yet (on hold until the physics below 1 keV exists).

| Work | Probe | Result | Checked |
|---|---|---|---|
| Ng & Beacom, PRD 90 (2014), arXiv:1404.2288 [Ng:2014pca]; Ioka & Murase, PTEP 2014, arXiv:1404.2279 [Ioka:2014kca] | High-energy neutrinos on the relic neutrinos | The first proposals | title |
| Bustamante, Rosenstrøm, Shalgar & Tamborra, PRD 101 (2020), arXiv:2001.04994 [Bustamante:2020mep] | IceCube HESE, 6 years, public | First rigorous statistical analysis; no evidence; upper limits on the coupling for 1--100 MeV | abstract |
| Creque-Sarbinowski, Hyde & Kamionkowski, PRD 103 (2021), arXiv:2005.05332 | Analytic spectra, any coupling matrix | Method | abstract |
| Esteban, Pandey, Brdar & Beacom, PRD 104 (2021), arXiv:2107.13568 [Esteban:2021tub] | Forecast, IceCube-Gen2 | Full theory (effects neglected before), realistic detection; code nusiprop | abstract |
| Carpio, Murase, Shoemaker & Tabrizi, PRD 107 (2023), arXiv:2104.15136 | IceCube showers; $L_\mu - L_\tau$ at MeV | Four-parameter fit to data; no evidence | abstract |
| Mazumdar, Mohanty & Parashari, JCAP 10 (2022), arXiv:2011.13685 | Flavor-specific interactions; IceCube dips vs. the Hubble tension | Only $\nu_\tau$ (inverted ordering) consistent with both | abstract |
| Hyde 2023 (arXiv:2307.02361); Döring & Vogl 2023 (arXiv:2304.08533); Franklin et al., PLB 867 (2025) (arXiv:2404.02202) | Point sources (NGC 1068, TXS 0506+056) | -- | title / abstract |
| **Wang, Xu & Zhou, PRL 135 (2025), arXiv:2501.07624 [Wang:2025qap]** | Relativistic lightest relic broadens the resonance | Boltzmann equations with absorption and regeneration; real scalar, Majorana, **no flavor structure** ("upcoming work"); forecast for the diffuse supernova background at Hyper-K, $g \sim 10^{-8}$ at sub-keV masses | text |
| Machado, Wang, Xu & Zhou, JCAP 07 (2026), arXiv:2512.00165 | Same, ultra-high-energy cosmogenic neutrinos | -- | abstract |
| Blinov et al. 2019; Berryman et al. 2023; Brdar et al. 2020; Sabti et al. 2020 | Laboratory and cosmology | As in family 2: above $10^{-4}$, keV to 100 MeV, only a scalar coupled to $\nu_\tau$ survives | abstract |

What family 3 says for Step 0:

1. Fits of real IceCube data exist (HESE 2020; showers 2023; point sources). Ours would add several data sets
   (flavor, spectrum, events), fitted source compositions, flavor textures, scalar and vector, Majorana and Dirac.
2. The relics' motion is no longer new: Wang, Xu & Zhou (2025) broadened the resonance with a relativistic lightest
   relic, for the supernova background, without flavor. Ours adds TeV--PeV energies, flavor, the vector, and the Dirac
   relics' helicity. Their flavor paper is announced: a race.
3. Against laboratory and cosmology, IceCube can lead only where they are weak: the scalar coupled to $\nu_\tau$
   above $10^{-4}$ (keV to 100 MeV), and perhaps below 1 keV. To be read off our planes once they run.

### Step 0 synthesis (2026-11-07, for his decision)

- **Our limits are not leading** for $L_e - L_\mu$, $L_e - L_\tau$ or the dark portal: at MeV masses they lie inside
  laboratory and BBN limits; at ultralight masses they match earlier high-energy sensitivities or sit inside the
  CMB's neutrino-lifetime bound and Dror's longitudinal-mode bounds (standard cosmology).
- **What is new, and defensible:** the propagation (his key aspect): the flux followed through the Universe, the
  Galaxy, the Sun and the Earth, with the potential and the collisions on the relics (absorption with
  regeneration, decay, scattering, thermal relics) together, for several models, fitted to several real IceCube
  data sets. The closest work (ABDN23) computes the potential only at the detector.
- **Where high-energy neutrinos matter:** their limits do not depend on the early Universe, so they hold in modified
  cosmologies that evade the CMB and BBN bounds; and secret interactions of a scalar with $\nu_\tau$, if our planes
  reach there.

### Pitch (decided earlier)

- 2026-11-07: **Propagation first** (Mauricio, option 1). The paper presents the most complete propagation
  calculation and applies it to several models and data sets. Every limit figure shows the existing bounds; a
  limit is called leading only where that is verified.
- 2026-11-07: **Key aspect: the detailed treatment of the propagation** (Mauricio). The novelty claim and the
  literature check center on it: for each earlier work, how it treated the trip from the sources to the detector.
- 2026-10-14: "the most comprehensive calculation of these effects in high-energy astrophysical neutrinos, using
  several types of current data". The comprehensiveness claim goes in the abstract, not in the title.

### Open points for Step 0

- The precise novelty claim, after the literature check.
- Which working title.

---

## Step 1. Main message and main result (draft 2026-11-07)

Sources of the numbers: **[nb]** `notebooks/fluxes_at_earth.ipynb` (measured 2026-11-02); **[pf]**
`notebooks/paper_figures.ipynb` and `paper/figures/` (campaign_ml, before the final run); **[sl]** the progress
slides (one merged plane of the final run); **[mem]** campaign notes of 2026-10-14/20. "Read" = read off a figure,
about 0.2 decade.

### 1. The message, in one sentence (draft)

High-energy astrophysical neutrinos feel a new light mediator along their whole trip, so turning IceCube data into
limits needs a calculation that follows them from their sources through the Universe, the Milky Way, the Sun and
the Earth, with the mediator's potential and its collisions on the relic neutrinos computed together; we provide it
and apply it to three families of models (the dark portal, $L_e - L_\mu$ and $L_e - L_\tau$, secret
interactions).

### 2. The main result

Convention: 95% C.L., a coupling disfavored at its mass when $\Delta\chi^2 \geq 3.84$ relative to the best fit of
its mass column (decided 2026-10-13). Source flavor composition from pion decay unless stated. Every value below is
**preliminary**; the final ones come after the ultralight run, the refinements, the HESE + DR2 re-evaluation and the
secret-interaction campaign.

| Model, masses | IceCube flavor + spectrum (MESE 2025, 2026 spectrum) | IceCube events (HESE + DR2) | Strongest existing bound | Leading? |
|---|---|---|---|---|
| $L_e - L_\mu$, ultralight | $g' \gtrsim 10^{2.1} M'/{\rm eV}$, beyond perturbative unitarity (read, [pf]) | down to $g' \approx 10^{-28}$, $10^{-28}$ to $10^{-20}$ eV (read, [pf]) | $\approx 2\times10^{-28}$ (2019 sensitivity); Dror's early-Universe decay, $g'/M' \gtrsim 10^{-8.8}$ eV$^{-1}$ | no |
| $L_e - L_\tau$, ultralight | as $L_e - L_\mu$ ([pf]) | flat at $\approx 8\times10^{-3}$, from scattering on the relics ([mem]) | as above | no |
| $L_e - L_\mu$, $L_e - L_\tau$, MeV | down to $g' \approx 10^{-3.7}$ between 0.5 and 5 MeV (read, [pf]) | to read | $10^{-8}$ to $2\times10^{-3}$ excluded above 2.4 MeV; BBN + CMB below 3.7 MeV | no |
| Dark portal ($e$, $\mu$, $\tau$), MeV | excluded up to $10^{7.2}$ eV; coupling floor to read ([pf]) | to read | Blinov et al.: vectors above $10^{-4}$, keV to 100 MeV; BBN + CMB | no |
| Dark portal ($\mu$), ultralight | every coupling $\geq 10^{-8}$ at $10^{-8}$ to $10^{-6}$ eV; $\geq 1.8\times10^{-8}$ at $10^{-4}$ eV; $\geq 1.8\times10^{-5}$ at $10^{-2}$ eV ([sl]) | to compute | CMB neutrino lifetime | no (standard cosmology) |
| Secret interactions (16 models) | not run | not run | Blinov et al.; only scalar $\nu_\tau$ open above $10^{-4}$ | to see |

### 3. Why the results read this way

- **Ultralight, flavor:** the cosmological potential, sourced by the Universe's electrons, changes the flavor mix at
  the sources and along the way; with today's data the Sun's and the Earth's shifts move $\chi^2$ by only ~0.1
  ([mem], 2026-10-15). The excluded region follows a fixed $g'/M'$ because the potential depends on the coupling
  over the mass.
- **Ultralight, decay:** where decays are open, every coupling in the plane makes the heavier states decay; the
  beam arrives as $\nu_1$, mostly $\nu_e$, which IceCube's flavors disfavor by $\Delta\chi^2$ = 10--11 ([sl]).
- **MeV:** resonant absorption on the relic neutrinos carves dips at $E = M^2/2m_r$, one per relic mass state: 1.0
  and 5.76 PeV for $M$ = 10 MeV ([nb]).
- **Fitted source compositions** absorb any energy-independent flavor shift, so their regions shrink to where the
  shift changes with energy ([pf]).
- To explain: why the HESE + DR2 events reach lower couplings than the flavor + spectrum data at ultralight masses.

### 4. Supporting results (one sentence each, as they would appear)

1. At $10^{-20}$ eV, the Sun's $L_e - L_\mu$ potential turns each mass state into one flavor before the neutrinos
   reach the Earth ([nb]; `tests/test_flux_stops.py`).
2. At $10^{-26}$ eV, the Milky Way's potential flips the lightest neutrino and antineutrino states and removes about
   40% of the flux ([nb]).
3. Decays and scatterings on the relics drain the flux together more than either alone: 69% survives where each
   alone leaves 100% and 81% ($L_e - L_\mu$, $10^{-2}$ eV, $g' = 10^{-6}$; [nb]).
4. The relics' thermal motion changes the fluxes by up to 29.5% (10 MeV scalar secret interaction; [nb]).
5. The helicity of Dirac relics matters: for a scalar, the resonance on a relativistic lightest relic nearly
   disappears (86% of the flux at 10 TeV survives at 31.6 keV, against full absorption when helicity is ignored;
   [pf]).
6. IceCube's 2026 spectrum mildly prefers a resonance at 0.5--7.5 MeV ($\Delta\chi^2$ = 3.6--10 below no new
   physics; [mem]); a spectral-shape effect, not evidence.
7. Future telescopes reach lower couplings at every mass, e.g., $g' \sim 10^{-4.5}$ near 1 MeV for $L_e - L_\mu$
   ([pf]).

### 5. What the paper does not claim

- Leading limits for $L_e - L_\mu$, $L_e - L_\tau$ or the dark portal.
- Evidence for new physics.
- Anything beyond perturbative unitarity (marked, not used), or the dark portal's potential planes (dropped).
- Limits with the mixing parameters varied in the likelihood (they are fixed; their effect is estimated outside it).

### 6. Open points

1. **A propagation-first headline result** (my suggestion): how much each stage of the trip moves the limits, full
   calculation against simplified ones (the potential only at the detector, as in ABDN23; relics at rest; no decay
   or scattering; no Galaxy or Sun). It would be the money plot. Cost: not measured; needs reruns of a few planes.
2. Final numbers: ultralight resume and merge, refinements, HESE + DR2 re-evaluation, secret-interaction campaign
   (after the physics below 1 keV), decisions B and C.
3. Literature leftovers: the strip $10^{-5}$ to $10^{-4}$ above 3.7 MeV (dark portal); $L_e - L_a$ couplings above
   $2\times10^{-3}$; Dror's Dirac case; whether our couplings thermalize in the early Universe.
4. Wang, Xu & Zhou's announced flavor paper.

---

## Step 2. Structure (draft 2026-11-07)

**Format (proposal):** revtex4-2, PRD regular article, two columns; about 14 pages of main text plus appendices
(a comprehensive paper). Author block and co-authors: to decide.

**Working title (to decide):** *New neutrino interactions with high-energy astrophysical neutrinos* (2026-10-21), or,
closer to the pitch, *New interactions of high-energy cosmic neutrinos, from the Hubble scale to the Earth*.

**Abstract, as ordered points:**
1. The question: light mediators of new neutrino interactions act on high-energy astrophysical neutrinos all along
   their trip.
2. What we do: follow the flux from the sources through the Universe, the Milky Way, the Sun and the Earth, with
   the potential and the collisions on the relic neutrinos together, for three model families.
3. What we fit: IceCube's flavor composition and spectrum, and its starting and through-going events.
4. What we find: which stage of the trip matters where, and the resulting limits, with where they compete.
5. Context: future telescopes.

**Sections (target length; content):**

1. **Introduction** (1.5 pages). The physics question; why high-energy astrophysical neutrinos; what earlier work
   did (potential at the detector; one effect at a time; relics at rest); this work; Fig. 1 (money plot) on page 1.
2. **New light mediators** (1.5 pages). The dark portal; $L_e - L_\mu$ and $L_e - L_\tau$; secret interactions
   (scalar or vector, Majorana or Dirac, flavor textures). What each does by mass: potential, absorption, decay,
   scattering. Where the models hold (perturbative unitarity).
3. **The trip** (3.5 pages). Run-in headings: sources and the cosmological transport (states in the potential,
   absorption with regeneration, decay, scattering, thermal and helicity effects of the relics); the Milky Way
   (smooth crossing, Galactic emission, crossing where decays open); the Sun, the Moon and the Earth (coherent
   evolution, the Earth's potential); the flux at Earth, stop by stop (figure).
4. **Data and statistical method** (2 pages). The data sets, described in words; source spectrum (broken power
   law) and compositions (fixed or fitted); systematics; the test statistic, the 95% C.L. per mass column, coverage
   tests.
5. **Results** (4 pages). What each stage does to the limits; $L_e - L_\mu$ and $L_e - L_\tau$; the dark portal;
   secret interactions; the MeV preference; future telescopes. Existing bounds on every limit figure.
6. **Conclusions** (1 page).

**Appendices (one topic each):** the transport equation in full; thermal absorption rates and the relics'
helicity; decays in a medium; the Earth's potential and chords; numerical method and validation; statistics and
coverage; the existing bounds shown in the figures and their sources.

---

## Step 3. Equations plan (draft list 2026-11-07; to typeset in `paper/equation_plan.tex` after sign-off)

Main text, about ten display equations; each derived formula is checked numerically against the full calculation
(check in brackets; "to do" where none exists yet).

| # | Equation | Purpose | Section | Check |
|---|---|---|---|---|
| 1 | The couplings of the three families: $G = g_D\,uu^\dagger$ (+ the $Z$-mixing term), $g'\,{\rm diag}(1,-1,0)$ and $(1,0,-1)$, and the secret-interaction couplings $g_{\alpha\beta}$ (scalar or vector) | Define the models | 2 | definitions |
| 2 | The potential of a body: a Yukawa integral of its electrons (or neutrons) times the texture | Where the potential comes from; why the bodies switch on with the range | 2 | Earth: exact Yukawa integral (`tests`, 2026-10-12) |
| 3 | The Goldstone couplings $Y_{ij} = i(G'_{ij}m_i + G'_{ji}m_j)/M$ and the unitarity condition | Where the models hold | 2 | `tests/test_unitarity.py` |
| 4 | The Hamiltonian in the potential, $H = UM^2U^\dagger/2E(1+z) + V(z)$, and its eigenstates | The states that are followed | 3 | closed form, potential-dominated, to $10^{-11}$ |
| 5 | The cosmological transport equation, compact: sources, absorption with its secondaries, scatterings, energy loss, decays | The heart of the trip | 3 | number conservation; $u \to 0$; Z-burst normalization |
| 6 | The absorption rate on the relics, thermally averaged; resonance energy $M^2/2m_r$ in-line | Why MeV dips appear where they do | 3 | brute-force thermal average |
| 7 | The coherent evolution near the Earth, $d\rho/dx = -i[H(x),\rho]$ | Sun, Moon, Earth | 3 | unitarity; Sun mapping (`tests/test_flux_stops.py`) |
| 8 | The flux at the detector and the event rates (folding) | From flux to data | 4 | HESE check exact (2026-10-14) |
| 9 | The test statistic and the 95% C.L. per mass column | How limits are read | 4 | coverage tests (planned) |

Appendices: the full transport with every secondary spectrum; the Galactic transport and emission; the source term;
the thermal kernel and the Dirac relics' helicity mixture; decay widths in a medium; the secret-interaction
transport (all channels); the Earth's chords and zenith average. Taken from `notes/notes.tex`, rewritten.

---

## Step 4. Citations plan (draft 2026-11-07)

No `refs.bib` yet: every entry is fetched from INSPIRE verbatim at Step 6. Texkeys marked **[k]** were read from
INSPIRE on 2026-11-07 (by arXiv number); **[id]** still to identify; **[v]** arXiv number from memory, to verify.
Code references (`docs/`, `src/`, `notes/`) are included where the paper uses the same input.

**Introduction**
- Discovery and status of high-energy astrophysical neutrinos: IceCube:2013low (arXiv:1311.5238) **[k]**; reviews
  Ackermann:2022rqc (Snowmass, arXiv:2203.08096) **[k]**, Spiering:2026hqf **[k]**.
- New physics with high-energy neutrinos, flavor as a probe: Arguelles:2022tki (arXiv:2203.10811) **[k]**;
  Song:2020nfh (arXiv:2012.12893) **[k]**.
- Earlier work on each family (one block each, from the Step 0 tables): long-range forces Bustamante:2018mzu,
  Agarwalla:2023sng, Agarwalla:2025jbn, Garg:2026gwx, Garg:2026rmy, Coloma:2020gfv, Wise:2018rnb **[k]**;
  dark portal Jana:2026pwx, Bertuzzo:2018itn (arXiv:1807.09877) **[k]**; secret interactions Ng:2014pca,
  Ioka:2014kca, Bustamante:2020mep, Esteban:2021tub, Wang:2025qap, Machado:2025ltu **[k]**, Creque-Sarbinowski:2020qhz,
  Carpio:2021jhu **[k]**.

**Sec. 2, models**
- $L_e - L_\mu$, $L_e - L_\tau$ gauge models: the original papers (He, Joshi, Lew & Volkas; Foot, 1991) **[id]**.
- Dark neutrino portal: Bertuzzo:2018itn, Jana:2026pwx **[k]**.
- Secret interactions: Berryman:2022hds **[k]**; Blinov:2019gcj **[k]**; cross sections Esteban:2021tub **[k]**.
- Non-conserved currents and the longitudinal mode: Dror:2020fbh **[k]**; perturbative unitarity, Allwicher,
  Arnan, Barducci & Nardecchia, JHEP 10 (2021) 129 **[id]**.

**Sec. 3, the trip**
- Sources: star-formation rate Yuksel:2008cu **[k]** (checked 2026-11-07: the code's `dp_cosmo` uses their Eq. (1), slopes 3.4, -0.3, -3.5, breaks at z = 1 and 4; Madau:2014bja is not the code's).
- Relic neutrinos: temperature and density **[id]**; clustering Ringwald:2004np, deSalas:2017wtt, Mertsch:2019qjv,
  Zimmer:2023jbb, Elbers:2023mdr **[k]**; coherence of cosmic neutrinos Farzan:2008eg **[k]**.
- The Milky Way: mass model McMillan:2016jtx **[k]**; solar neighbourhood McKee:2015hwa **[k]**; HEALPix
  Gorski:2004by **[k]**; Galactic emission CRINGE Schwefer:2022zly **[k]**; IceCube's Galactic plane
  IceCube:2023ame **[k]** and IceCube:2026plr **[k]**.
- The Earth: PREM (Dziewonski & Anderson 1981) **[id]**; the Sun's and the Moon's matter **[id]**.
- Relic screening: Chauhan:2024qew **[k]**.

**Sec. 4, data and method**
- IceCube MESE flavor composition, 11.4 years: IceCube:2025ole (PRL 2026, arXiv:2510.24957) **[k]** (checked: 11.4
  yr, best-fit $f_e = 0.30$, as our surface).
- IceCube 2026 spectrum: IceCube:2026plr (arXiv:2607.25966) **[k]** (to confirm the figure used).
- HESE 7.5 years: IceCube:2020wum **[k]**. DR2 tracks: IceTracks-DR2 release v3.1, doi:10.7910/DVN/MMIIZA (DataCite,
  not INSPIRE) **[id]**; IceCube point-source data IceCube:2021xar **[k]** if relevant; 9.5-year tracks
  Abbasi:2021qfz **[k]**.
- Atmospheric uncertainties: Barr:2006it **[k]**. Mixing: NuFIT 6.1 (Esteban:2024eli for 6.0 **[k]** plus the
  www.nu-fit.org release of November 2025).
- Future telescopes: the eight detectors of NuFlavorSingleSource **[id]**.

**Sec. 5, existing bounds in the figures:** the Step 0 tables (Bauer:2018onh, Sabti:2019mhn, Dror:2020fbh,
Baryakhtar:2017ngi, Wise:2018rnb, Coloma:2020gfv, Garg:2026gwx, Blinov:2019gcj, Brdar:2020nbj, Escudero:2019gfk,
Bustamante:2016ciw, Chauhan:2024qew) **[k]**.

---

## Step 5. Figures plan (started 2026-11-07)

### Fig. 1 (money plot, page 1): how each stage of the trip moves the limits (decided 2026-11-07)

- **What it shows:** the 95% C.L. region for one model across the mass range, from the full calculation and from
  simplified ones, so the reader sees which stage matters where.
- **Model and data (my proposal):** $L_e - L_\mu$, $10^{-30}$ to $10^{9}$ eV, IceCube HESE + DR2 events (the data
  set whose ultralight region lies below the unitarity line), pion-decay sources. Existing bounds in gray.
- **Curves (at most three, to keep it clear; my proposal):** (i) full calculation; (ii) the earlier standard: the
  potential evaluated only at the detector, no collisions on the relics (as in ABDN23); (iii) one intermediate step
  chosen after the pilot shows which stage moves the limits most (candidates: no Galaxy and Sun; relics at rest;
  no decay or scattering).
- **Code needed:** (ii) a new flavor calculation with the potential only at the detector (averaged oscillations
  with the mixing in the detector's total potential); "no Galaxy and Sun" from `flux_stops` fed to the fit; "relics
  at rest" for the dark-portal transport has no switch yet. Decay and scattering have switches.
- **Code (2026-11-07, uncommitted):** `dark_portal.trip` = 'full', 'no_local' (the beam as it arrives at the Milky
  Way) or 'detector' (the earlier standard); `dark_portal_solver._flux_runs_simplified`; `tests/test_trips.py`
  (4 tests pass: every trip equals the full one without potential or collisions; 'no_local' is `flux_stops`'
  arrival; 'detector' keeps the source composition where the potential dominates). First number, $L_e - L_\mu$ at
  $10^{-20}$ eV, $g' = 10^{-24}$, pion-decay sources, flavor fractions $(f_e, f_\mu, f_\tau)$ averaged over energy:
  full $(0.307, 0.307, 0.386)$; arrival at the Milky Way $(0.326, 0.343, 0.330)$; potential at the detector only
  $(0.333, 0.667, 0.000)$. The earlier standard predicts no $\nu_\tau$; the full trip, 39%.
- **Runs:** a few planes per curve; cost **not measured**. First a pilot on Tycho (a few mass columns, one per
  regime) when it has room, to measure the cost and pick curve (iii). Needs his go before launching.

### Main-text figures after Fig. 1 (draft 2026-11-07)

One fixed color per data set across all figures (his choice of 2026-10-21): IceCube flavor + spectrum auburn
`#9b2226`, IceCube events gamboge `#ee9b00`, future telescopes dark cyan `#0a9396`; existing bounds gray; the region
beyond perturbative unitarity hatched. Data sets labelled by what they are, never by internal names.

| Fig. | What it shows | Source | Status |
|---|---|---|---|
| 2 | The trip stop by stop: flavor and flux arriving at the Milky Way, after it, near the Earth, at the detector, at two masses ($10^{-20}$ and $10^{-26}$ eV) | `flux_stops`; `notebooks/fluxes_at_earth.ipynb`, Sec. 5 | exists as a notebook plot; restyle |
| 3 | What the collisions do: the flux with each term switched on in turn (absorption with regeneration, the relics' motion, decay, scattering), one MeV and one light case, with the relative change under each panel | `notebooks/fluxes_at_earth.ipynb`, Secs. 2--4 | exists; choose the two cases |
| 4 | $L_e - L_\mu$ and $L_e - L_\tau$: excluded regions, $10^{-30}$ to $10^9$ eV, with the MeV inset | `paper/figures/limits_le_lmu_le_ltau_*.pdf` | redo after the final run; add existing bounds and the unitarity hatching |
| 5 | The dark portal: textures $e$, $\mu$, $\tau$, ultralight and MeV | `paper/figures/limits_dark_portal_textures_*.pdf` (MeV only) | redo after the final run |
| 6 | Secret interactions: scalar and vector, Majorana and Dirac, the four flavor textures | -- | needs the secret-interaction campaign |
| 7 | IceCube's preference for an MeV resonance, or the discovery case | -- | after "discovery case" |

Appendix figures: the Dirac relics' helicity (`paper/figures/dirac_relic_helicity.pdf`, exists); validation and
coverage (text only unless the argument needs them).

Inputs still missing: the existing bounds as curves. They must be digitized from the papers' figures or taken from
public data (to identify for each: Bauer et al., Dror, Blinov et al., the CMB lifetime bound, Coloma et al.,
Garg et al.).

### The processes of each model: Feynman diagrams (his requirement, 2026-11-07)

Every Feynman diagram involved in each model is shown. Inventory, checked against the code on 2026-11-07
(`src/dp_si.py` channels; `src/dp_absorption.py` widths; the transport equation in `notes/notes.tex`):

| Process | Diagram | $L_e - L_\alpha$ ($Z'$) | Dark portal ($Z_D$) | SI scalar ($\phi$) | SI vector ($V$) |
|---|---|---|---|---|---|
| (a) Potential of ordinary matter | $\nu f \to \nu f$, forward, $q \to 0$ | electrons | -- (Z mixing set to zero) | -- | -- |
| (b) Absorption, $s$ channel | $\nu\,\bar\nu_{\rm relic} \to X \to$ pairs | yes | yes | yes ($\nu\nu_{\rm relic}$, Majorana) | yes |
| (c) Scattering, $t$ channel | $\nu\,\nu_{\rm relic} \to \nu\,\nu$ | yes | yes | yes | yes |
| (d) Scattering, $u$ channel | crossed $t$ | yes | yes | yes | yes |
| (e) Pair production | $\nu\,\bar\nu_{\rm relic} \to XX$, neutrino exchange | yes | yes | yes | yes |
| (f) Mediator decay | $X \to \nu\bar\nu$; charged-lepton pairs where open | $\nu\bar\nu$; $e^+e^-$; $\mu^+\mu^-$ for $L_e - L_\mu$ above 211 MeV (`dp_models.visible_width`) | $\nu\bar\nu$ | $\nu\nu$ | $\nu\bar\nu$ |
| (g) Neutrino decay | $\nu_i \to \nu_j X$ (and $\bar\nu_j$, Majorana) | yes | yes | planned (below 1 keV) | planned (below 1 keV) |
| (h) Longitudinal mode | the vector's Goldstone coupling, $\propto m_\nu g/M$, in (b)--(g) | yes | yes | -- | yes |
| (i) Relic-sourced potential | forward scattering on the relics | -- | -- | planned (below 1 keV) | planned (below 1 keV) |

Proposal: a **main-text figure** with the diagrams (a)--(h) drawn once, the mediator as a generic line (wavy for
vectors, dashed for the scalar), each panel labelled with its process and the models it belongs to; and an
**appendix figure** with each model's vertices (the couplings of Eq. (1), the charged-lepton vertex of $L_e - L_\alpha$,
the longitudinal mode's, the Dirac scalar's $\nu_L\nu_R$). Drawn in matplotlib in his style, in
`notebooks/paper_figures.ipynb`, from the slides' diagram code (`slides/make_figures.py`, five diagrams already).
Dirac secret interactions: the scalar turns $\nu_L$ into $\nu_R$ (sterile); its diagrams carry the helicities.

### Step 6: argument outlines, Secs. 2 and 3 (draft 2026-11-07, for approval before any prose)

**Sec. 2, New light mediators.**
1. *Job: why these three families.* Light mediators coupled to neutrinos arise in gauged lepton-number differences,
   in dark sectors reached through neutrino mixing, and as neutrino self-interactions. They differ in what else
   they couple to: $L_e - L_\alpha$ also to electrons, so ordinary matter sources a potential; the dark portal and
   secret interactions only to neutrinos, so only the relic neutrinos act on the beam. Together they span both
   ways a mediator acts on the trip.
2. *Job: $L_e - L_\alpha$.* Eq. (1a); its texture; the potential of the electrons, Eq. (2); each body contributes
   once its distance is within the range $1/M$; above $2m_e$ the mediator also decays into charged leptons.
3. *Job: the dark portal.* Eq. (1b): the coupling reaches the active neutrinos through their mixing with the dark
   neutrino, rank one in flavor; its $Z$ mixing, which would source a potential, is set to zero because the model
   bounds it far below the reach of these data (Jana & Swain).
4. *Job: secret interactions.* Eq. (1c): scalar or vector, Majorana or Dirac, four textures; for Dirac neutrinos
   the scalar couples left- to right-handed states, which are sterile.
5. *Job: what each does, by mass.* MeV masses: the resonance $E = M^2/2m_r$ falls in IceCube's range, so
   absorption carves dips; below a keV: decays (when $M < m_i - m_j$) and soft scatterings move neutrinos between
   states; ultralight masses: the potential of every body within $1/M$. The figure of diagrams.
6. *Job: where the models hold.* A vector coupled to a current the neutrino masses do not conserve has a
   longitudinal mode that couples like a scalar, Eq. (3); perturbative unitarity bounds $g/M$; that region is
   hatched and not used.

**Sec. 3, The trip** (run-in headings).
1. *What is followed.* Neutrinos and antineutrinos separately, as populations of the eigenstates of the local
   Hamiltonian, Eq. (4). Over cosmological distances the beam decoheres, and the potential changes slowly
   (adiabatically), so populations suffice until the Sun.
2. *Sources.* Distributed in redshift as the star-formation rate, to $z = 4$; a broken power law; the flavor
   composition at the sources, split over the eigenstates there.
3. *The cosmological transport.* Eq. (5), term by term, each tied to its diagram; the redshift through the energy.
4. *Absorption.* Eq. (6): coherent in the eigenstates; the relics' thermal motion and, for Dirac relics, their
   helicity; each absorption returns a neutrino and an antineutrino at lower energy.
5. *Decays and scatterings below a keV.* What each does; together they drain more than either alone (supporting
   result 3).
6. *The Milky Way.* The beam enters the Galaxy's potential smoothly; where the Galaxy opens decays, it is followed
   along every line of sight; the Galaxy's own neutrinos are emitted along those lines.
7. *The Sun, the Moon and the Earth.* Coherent evolution, Eq. (7); the Sun can turn mass states into flavors
   (supporting result 1); the Earth's own potential; the average over chords.
8. *The flux at Earth, stop by stop.* Figs. 2 and 3.

### Step 6: argument outlines, appendices (draft 2026-11-07, for approval before any prose)

Sources: `notes/notes.tex` sections named in brackets. Validation history and internal checks stay in the notes;
the appendices say only what a reader needs to follow or reproduce the calculation.

**A. The vertices of each model** [Model and cross sections; Dirac neutrinos and a vector mediator].
1. *Job: the couplings in the basis the transport uses.* Each model's flavor couplings become mass-basis couplings:
   $U^\dagger G U$ for the vectors and for Dirac scalars (Hermitian); $U^T G U$ for Majorana scalars (complex
   symmetric). The Majorana phases drop out of every rate; for one coupled flavor, so does $\delta_{\rm CP}$.
2. *Job: fermion flow.* Majorana diagrams follow the fermion-flow rules of Denner et al.; for Dirac scalars, each
   vertex flips chirality, so its secondaries split into visible and sterile states. The vertices figure.
3. *Job: the longitudinal mode's couplings*, Eq. (goldstone), from the divergence of the current; the unitarity
   condition and its checks against Allwicher et al.

**B. The transport equation in full** [The transport equation in full; The Galactic emission].
1. *Job: every term of Eq. (transport).* The source term with the Jacobian $(1+z)$; absorption with the shares of the
   two secondaries in each state; the soft-scattering rates $R_{ab}$; the energy loss to relic mass changes $b_a$;
   the decay gain with its helicity-keeping and helicity-flipping daughters.
2. *Job: the Milky Way.* The transport along a line of sight, with the Galactic emission term (template, gas and
   unresolved sources, composition $(x, 1-x, 0)$); the crossing of the extragalactic flux where decays open.
3. *Job: how it is solved*, briefly: grids in energy and redshift, the adiabatic tables of eigenstates, the
   per-band handover to the Sun (resolution choices only; convergence tests stay in the notes).

**C. Thermal absorption rates and the helicity of Dirac relics** [Thermal absorption rates; Relic motion in the
collision transport; Dirac relics in motion].
1. *Job: the rate per relic.* The Breit--Wigner resonance, its width (the neutrino pairs plus charged leptons where
   open), integrated over the relics' Fermi--Dirac momenta at $T_{\nu,0}(1+z)$; its normalization reproduces the
   standard resonant absorption (the Z-burst check).
2. *Job: the relics' motion everywhere.* At MeV masses, the thermal average enters every collision term, not only
   the resonance (`thermal_products`).
3. *Job: Dirac relics' helicity.* A Dirac relic keeps its decoupling helicity; a moving relic is a mixture of the
   two chiral states with weights $(1 \pm \pi)/2$; consequence: the scalar's resonance on a relativistic lightest
   relic nearly disappears, the vector's deepens (the helicity figure, if kept).

**D. Decays in a medium** [Relic mass changes, neutrino decay, and where the models hold].
1. *Job: vacuum decays.* Rates and daughter spectra (helicity keep and flip), number-conserving.
2. *Job: decays between eigenstates in a potential.* The kinematic window set by the effective masses; a potential
   opens $\nu_i \to \bar\nu_i$, closed in vacuum; valid while each width is small against the splittings
   ($\zeta \ll 1$), with the size of the corrections where it is not (being mapped; state honestly).
3. *Job: what the transport leaves out.* The mediator is taken as invisible (its decays back to neutrinos are
   decision B); decays from the Sun on are negligible except where the potential already dominates.

**E. The secret-interaction transport** [The other transport: dp_si; Transport; Model and cross sections].
1. *Job: the equation in the mass basis*, with every $2 \to 2$ channel ($s$, $t$, $u$ and their interference) and
   pair production, each secondary's spectrum, and the decays with both daughters.
2. *Job: the cross sections.* Taken from Esteban et al., Appendix A, with two corrections we found (the coherent
   factor of pair production; complex couplings with CP violation); checked by brute-force amplitudes.
3. *Job: where it is used*: secret interactions at every mass; the dark portal and $L_e - L_\alpha$ at MeV masses,
   where the potential is negligible.

**F. The Earth's chords** [Earth steps; The zenith average of the Earth chords; The Earth's own potential].
1. *Job: the potential along a chord*: the Earth's electrons (PREM) integrated with the Yukawa factor, Eq.
   (potential), including the exterior where the range exceeds the radius.
2. *Job: the average over directions*: per declination band, an average over chords that keeps the oscillation
   phase (no sampling floor).

### Step 6: argument outline, Sec. 4 (draft 2026-11-07, for approval before prose)

Sources: the final configurations (`analysis/configs/campaign_final/*/le_lmu_ul__b0.json` and the `_fx` twins),
`analysis/events.py`, the decisions of 2026-10-13/14/15/18 in memory.

1. *Job: the three data sets, in words.*
   - **IceCube flavor + spectrum:** the flavor composition measured with 11.4 years of medium-energy starting
     events (IceCube:2025ole), used as its likelihood in $(f_e, f_\mu)$, against the model's composition over
     10 TeV--30 PeV; plus IceCube's 2026 measurement of the spectrum (IceCube:2026plr).
   - **IceCube events:** 7.5 years of high-energy starting events (their topologies carry the flavor) and
     through-going tracks (IceTracks-DR2, IC79 and IC86, above 3 TeV), folded with IceCube's public responses.
     This set drops the flavor measurement, whose events overlap HESE's.
   - **Future telescopes:** forecasts for eight telescopes, 10 years each, around no new interaction.
2. *Job: the model of the flux in the fits.* A broken power law (normalization and two spectral indices), free;
   the Galactic component, with a Gaussian prior on its normalization ($0.91 \pm 0.18$); the source composition
   fixed to pion decay, or fitted (extragalactic and Galactic $\nu_e$ fractions free, no $\nu_\tau$); the lightest
   neutrino mass free from $10^{-6}$ to $0.1$ eV, with a prior from cosmology ($\sum m_\nu < 0.12$ eV at 95%);
   the mixing parameters fixed to NuFIT 6.1's best fit, with their effect estimated separately; normal ordering.
3. *Job: the backgrounds and detector systematics of the event data.* Atmospheric neutrinos with the 12 Barr
   parameters, the cosmic-ray spectral tilt and the prompt normalization; HESE's conventional and muon
   normalizations; one normalization per DR2 configuration; all profiled, with their priors. Poisson likelihood.
4. *Job: how a limit is read.* $\chi^2 = -2\ln\mathcal{L}$ minimized over every other parameter at each (mass,
   coupling); a coupling is disfavored at 95% C.L. when $\Delta\chi^2 \geq 3.84$ relative to the best coupling at
   that mass. Where every coupling of a column changes the flux (decays at light masses), the reference is the fit
   without a new interaction. The region beyond perturbative unitarity is not used.
5. *Job: is the 3.84 threshold right?* Wilks's theorem may fail (boundaries, couplings with no effect); the coverage
   tests (planned: closure, Asimov recovery, pseudo-experiments) decide whether a Feldman--Cousins construction is
   needed. State plainly what is conservative.

### Step 6: argument outline, Sec. 1, Introduction (draft 2026-11-07, for approval before prose)

His rules: open with the physics question; then why other probes fall short; then the probe; then this work;
Fig. 1 on page 1; no technical caveats or estimates; the main message in bold; a short roadmap.

1. *Job: the question.* Do neutrinos feel new interactions, carried by light mediators that couple to them? Many
   extensions of the Standard Model predict such mediators (gauged lepton-number differences, dark sectors reached
   through neutrino mixing, self-interactions of neutrinos) [He et al.; Foot; Bertuzzo et al.; Berryman et al.].
2. *Job: why other probes leave room.* Laboratory experiments, oscillation experiments, and cosmology constrain
   these mediators strongly over many masses [Bauer et al.; Coloma et al.; Blinov et al.; Escudero & Fairbairn;
   Dror]. Their reach rests on short baselines, low energies, or the history of the early Universe; effects that
   build up over cosmological distances and high energies lie beyond them.
3. *Job: the probe.* High-energy astrophysical neutrinos, discovered by IceCube [IceCube:2013low], travel up to
   gigaparsecs with TeV--PeV energies; IceCube now measures their spectrum and flavor composition
   [IceCube:2025ole; IceCube:2026plr]. On the way, a new mediator can act on them in two ways: through the potential
   of the matter they cross, and through collisions with the relic neutrinos.
4. *Job: what earlier work did.* Earlier studies treated one effect at a time and simplified the trip: the
   potential evaluated only at the detector [Bustamante & Agarwalla; Agarwalla et al.], the relic neutrinos at rest
   [Ng & Beacom; Ioka & Murase; Bustamante et al. 2020; Esteban et al.], until recently [Wang, Xu & Zhou].
5. *Job: this work.* We follow the flux through the whole trip -- the Universe, the Milky Way, the Sun, the Earth --
   with the potential and the collisions together, for three families of models, and fit several IceCube data sets.
   **Bold main message:** how a new mediator changes the flux at Earth depends on where along the trip it acts, so
   reliable limits need the full trip. Fig. 1: how each stage moves the limits (the result sentence waits for the
   money-plot runs).
6. *Job: roadmap*, one sentence per section; the appendices named.

---

## Decision log

- 2026-10-14: Two papers (dark portal; secret interactions), one shared set of notes. (Mauricio) Superseded 2026-10-21.
- 2026-10-14: Working title "New forces on high-energy cosmic neutrinos, from the Hubble scale to the Earth"; no
  drafting yet. (Mauricio)
- 2026-10-14: Pitch: the most comprehensive calculation, with several types of current data. (Mauricio)
- 2026-10-18: Campaign cut to the dark portal ($e$, $\mu$, $\tau$; MeV) and $L_e-L_\mu$, $L_e-L_\tau$ (ultralight and
  MeV); the 2050 flavor projection replaced by future-detector forecasts. (Mauricio)
- 2026-10-21: The two papers merged into one, "New neutrino interactions with high-energy astrophysical
  neutrinos". (Mauricio)
- 2026-10-24: The ultralight dark-portal potential planes dropped (Jana & Swain's bound on the $Z$ mixing); the
  region beyond perturbative unitarity marked in the figures. (Mauricio)
- 2026-10-25: Secret-interaction campaign on hold until the physics below 1 keV exists; then all masses run once.
  (Mauricio)
- 2026-11-07: Secret interactions stay in this paper; splitting them off was offered and declined. (Mauricio)
- 2026-11-07: Planning file started; Step 0 in progress.
- 2026-11-07: Key aspect of the paper: the detailed treatment of the propagation. (Mauricio)
- 2026-11-07: Literature check of all three families done (tables above).
- 2026-11-07: Pitch: propagation first; existing bounds on every limit figure; leading limits claimed only where
  verified. Step 0 signed off. (Mauricio)
- 2026-11-07: Steps 1 and 2 drafted ("Don't stop"); waiting for his sign-off.
- 2026-11-07: Money plot = how each stage of the trip moves the limits (full vs simplified). (Mauricio, "Go with your pick")
- 2026-11-07: Step 3 equation list drafted; Fig. 1 specified (pilot needed).
- 2026-11-07: Step 4 citations plan drafted (texkeys checked on INSPIRE where marked).
- 2026-11-07: Step 5 figures plan drafted (Figs. 1--7).
- 2026-11-07: He said "Continue" after the request to sign off Steps 1--5; taken as provisional approval (any step can be revised). Step 6 started: `equation_plan.tex/.pdf` (compiles cleanly), `main.tex` skeleton (4 pages, builds, no undefined references), `references.bib` (51 INSPIRE entries; his number = year fix on JHEP/JCAP), `spine.md`.
- 2026-11-07: Focus on the method first: its text and figures. Every Feynman diagram involved in each model is shown. (Mauricio)
- 2026-11-07: Process inventory and diagram plan; argument outlines for Secs. 2 and 3, for approval.
- 2026-11-07: Feynman diagrams drawn (his approval of the plan): `analysis/diagrams.py`; `paper/figures/feynman_processes.pdf` (main text, (a)--(h)) and `feynman_vertices.pdf` (appendix: each model's vertex; the scalar's own diagrams); cells added to `notebooks/paper_figures.ipynb` (Sec. 5); placed in `main.tex`.
- 2026-11-07: Sec. 2, paragraphs 1--3 written in `main.tex` (the three families; $L_e - L_\alpha$ with Eqs. lag_le, potential; the dark portal with Eq. lag_dp and the Z-mixing bound); references He:1990pn, Foot:1990mn, He:1991qd added. For his review.
- 2026-11-07: Sec. 2, paragraphs 4--6 written (secret interactions, Eq. lag_si; what each mediator does by mass; where the models hold, Eq. goldstone); reference Allwicher:2021rtd added. Sec. 2 complete in draft, for his review.
- **Handoff 2026-11-07.** *Changed:* this planning file (Steps 0--5 drafted, Step 0 signed off; Steps 1--5
  provisionally approved by "Continue"); `equation_plan.tex/.pdf`; `main.tex` skeleton with Sec. 2 drafted in full
  (paragraphs 1--6, Eqs. lag_le, potential, lag_dp, lag_si, goldstone); `references.bib` (56 INSPIRE entries);
  `spine.md`; Feynman diagrams (`analysis/diagrams.py`, `figures/feynman_{processes,vertices}.pdf`, notebook Sec. 5);
  money-plot code (`dark_portal.trip`, `tests/test_trips.py`, 4 tests pass). Nothing committed.
  *Decided (Mauricio):* SI stays in this paper; key aspect = the propagation; pitch = propagation first; money plot =
  how each stage moves the limits; method first; every Feynman diagram of each model shown.
  *Open:* his review of Sec. 2; Sec. 3 prose (outline approved); the equation plan's red items; existing bounds as
  curves; the money-plot pilot on Tycho (cost not measured; Tycho full); title and co-authors; citations still to
  identify (PREM, relic inputs, IceTracks-DR2 on DataCite, the future detectors' sources).
- 2026-11-07: Sec. 3, paragraphs 1--4 written (what we follow, Eq. hamiltonian; sources, Yuksel:2008cu; the cosmological transport, Eq. transport; absorption, Eq. absorption). For his review.
- 2026-11-07: Sec. 3, paragraphs 5--8 written (decays and scatterings; the Milky Way; the Sun, the Moon and the Earth, Eq. coherent; the flux stop by stop); thermal_products is true in campaign_final, campaign_ml and campaign_si, so the text states the relics' motion enters every collision term at MeV masses; PREM (Dziewonski:1981xy) added. Sec. 3 complete in draft.
- 2026-11-07: Figs. 2 and 3 made (`analysis/method_figures.py`, cached fluxes `output/analysis/method_figures.npz`, 222 s on 4 cores; `paper/figures/flux_stops.pdf`, `collision_terms.pdf`; notebook Sec. 6); placed in `main.tex` as two-column figures. Fig. 3 cases: 10 MeV scalar on nu_tau (resonance, every channel, thermal motion) and L_e - L_mu at 1e-2 eV (potential, decay, scattering).
- 2026-11-07: Captions written for Figs. processes, trip, collisions and vertices (his caption rules); figure layout kept at four rows (asked; he said "Continue").
- 2026-11-07: Appendix outlines (A--F) drafted, for approval before prose.
- 2026-11-07: Appendices A (vertices) and D (decays in a medium) written; two to-dos left in D for the final run's mapping of zeta and for decision B; reference Denner:1992me added.
- 2026-11-07: Appendix B (the transport in full) written: source, secondaries, soft scatterings, energy loss, Galactic transport and emission, crossing, grids (439 energies 4.6 TeV--30 PeV, 200 redshift steps, 8 Sun paths, 12 chords: campaign configs).
- 2026-11-07: Appendix C (thermal rates; relics' motion everywhere at MeV; Dirac helicity, Eqs. polarization and helicity_mix) written, with the existing helicity figure (texture e, `tools/dirac_relic_polarization.py`). To do: that figure uses the data sets' colors (auburn, gamboge, cyan) for the relic treatments -- recolor; relic clustering: decide.
- 2026-11-07: Appendices E (the secret-interaction transport, Eq. si_transport; the two corrections to Esteban et al.) and F (the Earth's chords) written. All appendices drafted; to-dos left: zeta mapping and decision B (D), relic clustering and the helicity figure's colors (C).
- 2026-11-07: Overlap check (8-word sequences vs every tex in ~/Research): 31 phrases close to our notes, one to NuLRIEarth, three to old NuDecay drafts; all reworded; now zero overlaps.
- 2026-11-07: Sec. 4 argument outline drafted, for approval.
- 2026-11-07: Sec. 4 written (data, expected events Eq. events, the flux in the fits, backgrounds, how a limit is read Eq. dchi2); to-dos: cite IceTracks-DR2 and the eight telescopes; the mixing parameters' effect; the coverage tests. Reference Barenboim:2026mvn (mass-sum prior, its Table II) added.
- 2026-11-07: Small items cleared: IceTracks-DR2 cited (DataCite: v3.1, 2026, doi 10.7910/DVN/MMIIZA, entry IceCube:IceTracksDR2); the eight telescopes cited (keys as in NuFlavorSingleSource's paper, checked on INSPIRE); relic clustering stated as neglected (App. C); helicity figure recolored (gray dashed, purple dotted, black solid; notebook cell and PDF).
- 2026-11-07: Introduction argument outline drafted, for approval.
- 2026-11-07: Introduction written from the outline (bold message as proposed; Fig. 1 result left as a to-do). Citations for 'relics at rest' limited to the two works checked (Bustamante:2020mep, Esteban:2021tub, per dp_config) and Wang et al.'s statement; 'potential at the detector' to ABDN23 (its text).
