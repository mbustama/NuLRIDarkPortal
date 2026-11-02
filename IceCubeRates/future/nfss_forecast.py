# Vendored from NuFlavorSingleSource, commit 39088bd ("Final production run: the 2026 inputs"), code/forecast.py,
# unchanged except: the data directory (IceCubeRates/data/future) and the default inputs ('final', those of its final production run)
# (docs/dev/FUTURE_DETECTORS_PLAN.md; Mauricio, 2026-10-14/15: the eight detectors as in that project)
"""
Core of the forecast: flavor composition at Earth of neutrinos from a single point source.

Channels per detector: tracks, showers (single cascades), double cascades.
Likelihood: binned Poisson in (channel x reconstructed energy x angular ring
around the source). Source flavor fractions at Earth (f_e, f_mu, f_tau) are
scanned over the flavor triangle; the all-flavor normalization, the spectral
index, and one background normalization per (detector, channel) are profiled.

Contains: oscillation-averaged flavor ratios (NuFIT 6.1), cross sections, atmospheric fluxes,
the Source class, the generic Detector response, the Analysis likelihood and the triangle scan.
Detector-specific parameters live in detectors.py. Numbers marked [ASSUMPTION] are knobs.
"""
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent      # IceCubeRates
DATA, FIGS = ROOT / 'data' / 'future', None

# The inputs (Oct 14). 'stored': those of every result stored before the final production run (IceCube's 2022 fit of
# NGC 1068, its 2025 fits of the Seyferts, HESE's 7.5-yr diffuse flux, our assumed nu_mu-as-shower fractions). 'final':
# those adopted for the final run (Mauricio, Oct 14): IceCube's 2026 fits of NGC 1068 and of the Seyferts, its 2026
# combined fit of the diffuse flux, its measured nu_mu-as-shower fraction (detectors.NUMU_SHW). The environment
# variable NUFLAV_INPUTS overrides the default.
INPUTS = os.environ.get('NUFLAV_INPUTS', 'final')
assert INPUTS in ('stored', 'final'), INPUTS
FINAL = INPUTS == 'final'

NA = 6.02214e23           # nucleons per gram
YR = 3.15576e7            # s
DEG = np.pi / 180.

# --------------------------------------------------------------------------
# Source: NGC 1068. IceCube's 2022 fit (Science 378, 538) is the flux of the detector calibration's anchor (79 events in
# 3186 d, detectors.calibrate), a property of the detector; the source itself is, with the final inputs, IceCube's 2026
# fit (ApJL 1000, L26, 13.1 yr of tracks): 4.7e-11 TeV^-1 cm^-2 s^-1 at 1 TeV, gamma = 3.4.
# --------------------------------------------------------------------------
DEC_SRC = -0.0133          # deg
PHI_NUMU_1TEV = 5.0e-14    # 2022 fit: nu_mu + nubar_mu flux at 1 TeV [GeV^-1 cm^-2 s^-1]
GAMMA_TRUE = 3.2
NGC1068_FIT = (4.7e-14, 3.4) if FINAL else (PHI_NUMU_1TEV, GAMMA_TRUE)     # the source: (flux at 1 TeV, index)
E0 = 1e3                   # GeV, pivot

# Oscillations: NuFIT 6.1 (data to November 2025; www.nu-fit.org), normal ordering, IC24 with SK atmospheric data.
# NuFIT 6.0 (used until 2026-10-13): 0.308, 0.470, 0.02215, 212; the flavor ratios at Earth differ by < 0.001.
S12, S23, S13, DCP = 0.3088, 0.470, 0.02248, 212.0


def pmns(s12=S12, s23=S23, s13=S13, dcp=DCP):
    t12, t23, t13 = np.arcsin(np.sqrt([s12, s23, s13]))
    c12, c23, c13 = np.cos([t12, t23, t13])
    sn12, sn23, sn13 = np.sin([t12, t23, t13])
    d = np.exp(1j * np.radians(dcp))
    return np.array([
        [c12 * c13, sn12 * c13, sn13 / d],
        [-sn12 * c23 - c12 * sn23 * sn13 * d, c12 * c23 - sn12 * sn23 * sn13 * d, sn23 * c13],
        [sn12 * sn23 - c12 * c23 * sn13 * d, -c12 * sn23 - sn12 * c23 * sn13 * d, c23 * c13]])


def earth_ratios(fS):
    U2 = np.abs(pmns()) ** 2
    P = U2 @ U2.T                        # averaged P(alpha->beta)
    return P.T @ np.asarray(fS, float)


F_TRUE = earth_ratios([1 / 3, 2 / 3, 0])             # pion beam (full pi decay chain)


class Source:
    """All-flavor spectrum dPhi/dE = phi_tot (E/E0)^-gamma shape(E), nu:nubar = 1:1, at declination dec.
    phi_numu_E0 is the true nu_mu+nubar_mu normalization at E0 before the shape factor [GeV^-1 cm^-2 s^-1], i.e. the
    flux at E0 only if shape(E0) = 1 (NGC 1068: yes; the ~300 TeV source: no, shape(100 TeV) = e^-1.1)."""

    def __init__(self, name, E0, gamma, shape, phi_numu_E0, f_true=F_TRUE):
        self.name, self.E0, self.gamma, self.shape = name, E0, gamma, shape
        self.phi_numu_E0, self.f_true = phi_numu_E0, np.asarray(f_true)

    def with_truth(self, label, f_true):
        """The same source with another flavor composition at Earth and the same nu_mu + nubar_mu flux, the quantity
        IceCube's tracks measure (so the all-flavor flux changes as 1 / f_mu)."""
        return Source(f'{self.name}, {label}', self.E0, self.gamma, self.shape, self.phi_numu_E0, f_true)

    @property
    def lphi_true(self):
        return np.log10(self.phi_numu_E0 / self.f_true[1])

    def weights(self, Egrid, lphi=None, gamma=None):
        lphi = self.lphi_true if lphi is None else lphi
        gamma = self.gamma if gamma is None else gamma
        return 10 ** lphi * (Egrid / self.E0) ** (-gamma) * self.shape(Egrid)


NGC1068 = Source('NGC 1068', 1e3, NGC1068_FIT[1], lambda x: np.ones_like(x), NGC1068_FIT[0])
# IceCube's 2022 fit: the flux behind the calibration and behind the projections that the cross-checks compare with
# (IceCube's own 4.2 sigma in 3186 d, TRIDENT's, the IceCube-Gen2 TDR's); the same as NGC1068 with the stored inputs
NGC1068_2022 = Source('NGC 1068, 2022 fit', 1e3, GAMMA_TRUE, lambda x: np.ones_like(x), PHI_NUMU_1TEV)
BENCHMARKS = {'pion decay (1:2:0)': earth_ratios([1 / 3, 2 / 3, 0]),
              'muon damped (0:1:0)': earth_ratios([0, 1, 0]),
              'neutron decay (1:0:0)': earth_ratios([1, 0, 0])}

# Synchrotron cooling of the secondaries. Above a critical energy, E_c ~ sqrt(m^5 / tau) / B, a muon (pion, kaon) loses
# its energy faster than it decays: E_mu,c ~ 6e9 GeV (B / G)^-1, E_pi,c = 18.4 E_mu,c, E_K,c = 629 E_mu,c. Ec below is the
# neutrino energy E_mu,c / 3 (each neutrino from muon decay takes ~1/3 of E_mu): one parameter, the magnetic field.
# cooled_fractions (Oct 13): each species in steady state, so that particles that cool pile up below their critical
# energy, where they finally decay, and kaons, whose K -> mu nu_mu keeps a nu_mu-only flux far above the pions'.
# cooled_fractions_simple: the first model, decay-before-cooling fractions 1 / (1 + (E/E_c)^2), no pile-up, no kaons.
COOL_RATIO = 0.2135 * 18.41 * 3
M_MU, M_PI, M_K = 0.1056584, 0.1395704, 0.493677              # GeV
TAU_MU, TAU_PI, TAU_K = 2.196981e-6, 2.6033e-8, 1.2380e-8        # s


def _crit(m, tau):
    """A species' critical energy (synchrotron cooling time = decay time) over the muon's, in the same field."""
    return (m / M_MU) ** 2.5 * (TAU_MU / tau) ** 0.5


CRIT_PI, CRIT_K = _crit(M_PI, TAU_PI), _crit(M_K, TAU_K)        # 18.4, 629
Z_PI_NU = (1 - (M_MU / M_PI) ** 2) / 2                           # mean energy fraction of the nu_mu in pi -> mu nu, 0.2135
Z_K_NU = (1 - (M_MU / M_K) ** 2) / 2                             # in K -> mu nu, 0.477
BR_K_MUNU = 0.6356                                              # K+ -> mu+ nu_mu; the other kaon decays are left out
KAON_RATIO = 0.1                                                # kaons per pion produced, at the same energy [ASSUMPTION]
_EP = np.logspace(0.5, 10.5, 2001)                              # parent energies [GeV] of the steady-state cascade


def _at(F, y):
    """A spectrum F on _EP at energies y: log-log interpolation; zero above the grid."""
    return np.exp(np.interp(np.log(y), np.log(_EP), np.log(np.maximum(F, 1e-300)), right=-700.))


def _steady(Q, eps_c):
    """Decays per unit energy, in steady state, of particles injected as Q (on _EP) that cool by synchrotron radiation
    (dE/dt = -b E^2) and decay (in E tau / m), with eps_c where the two times are equal:
        R(eps) = (eps_c^2 / eps^3) int_eps^inf exp[-(eps_c^2 / 2)(1/eps^2 - 1/eps'^2)] Q(eps') deps'.
    Every particle decays, so R integrates to what Q does: those injected above eps_c pile up below it. Exponential
    integrator in u = 1/eps^2, exact for Q eps^3 linear in u between grid points. eps_c = inf: no cooling, R = Q."""
    if not np.isfinite(eps_c):
        return Q.copy()
    a = 0.5 * eps_c ** 2
    eps = _EP[::-1]
    u = eps ** -2.
    g = Q[::-1] * eps ** 3
    du = np.diff(u)
    x = a * du
    ex = np.exp(-x)
    small = x < 1e-4
    e0 = np.where(small, du * (1 - x / 2 + x * x / 6), -np.expm1(-x) / a)
    e1 = np.where(small, du * du * (0.5 - x / 3 + x * x / 8), (-np.expm1(-x) - x * ex) / a ** 2)
    inc = g[1:] * e0 - (g[1:] - g[:-1]) * e1 / du
    I = np.zeros_like(u)
    for k in range(len(du)):
        I[k + 1] = I[k] * ex[k] + inc[k]
    return (a * I / eps ** 3)[::-1]


def _chain(inject, crit, z_nu, br, Ec, Eg):
    """One parent species (pions or kaons): its nu_mu and its muons' nu_e (= nu_mu) per unit energy at the neutrino
    energies Eg, cooled (critical energies crit * 3 Ec and 3 Ec) and uncooled; mean energy fractions. inject(z) is the
    parents' injection spectrum on _EP for a neutrino that takes the fraction z of the parent's energy: the nu_mu takes
    z_nu, each neutrino of the muon (1 - z_nu) / 3. Each of the two is injected so that, uncooled, it follows the
    source's own spectrum (see cooled_fractions)."""
    out = {}
    z_mu = (1 - z_nu) / 3
    for tag, eps_mu in (('cooled', 3. * Ec), ('uncooled', np.inf)):
        Rp = _steady(inject(z_nu), crit * eps_mu)
        nu_p = br * _at(Rp, Eg / z_nu) / z_nu
        Rp = _steady(inject(z_mu), crit * eps_mu)
        Rmu = _steady(br * _at(Rp, _EP / (1 - z_nu)) / (1 - z_nu), eps_mu)
        out[tag] = (nu_p, 3. * _at(Rmu, 3. * Eg))
    return out


def cooled_fractions(Ec, gamma=2.0, shape=None, kaon_ratio=KAON_RATIO, Egrid=None):
    """Pion and kaon decays with synchrotron cooling of the pions, kaons and muons, each in steady state (with the
    pile-up of cooled particles), for a source whose uncooled spectrum is E^-gamma shape(E): the (nu_e, nu_mu, nu_tau)
    flux at Earth per unit uncooled all-flavor flux, shape (3, nE). Ec = inf: no cooling.
    Each neutrino of a chain (the parent's nu_mu, the muon's nu_e and nu_mu) is followed from parents injected as
    eps^-gamma shape(z eps), z its energy fraction of the parent, so that uncooled it follows the source's own spectrum
    exactly, and is taken relative to that: without cooling the composition is exactly pion decay's (1:2:0) at the
    source, and the cooled flux is the steady-state one, with no factor between the shapes of different chains. (With
    one injection for all, eps^-gamma shape(eps / 4), the chains' uncooled spectra differ where shape cuts off steeply,
    and their ratio inflated what cooling moves below such a cutoff by up to many decades: the ~300 TeV source and the
    flare suppressed below 30 TeV, until Oct 13.) The chains' shares are those of a power law, E^-gamma: the kaons'
    is kaon_ratio * BR(K -> mu nu) * [z_K^(gamma-1) + 2 ((1 - z_K)/3)^(gamma-1)] over the pions' [z_pi^(gamma-1) +
    2 ((1 - z_pi)/3)^(gamma-1)]."""
    Eg = E if Egrid is None else Egrid
    if not np.isfinite(Ec):
        return earth_ratios(np.outer([1 / 3, 2 / 3, 0], np.ones_like(Eg)))
    sh = (lambda x: 1.) if shape is None else shape

    def inject(z):
        return _EP ** -gamma * sh(z * _EP)
    chains = [(_chain(inject, CRIT_PI, Z_PI_NU, 1., Ec, Eg), 1. * (Z_PI_NU ** (gamma - 1)
                                                                    + 2 * ((1 - Z_PI_NU) / 3) ** (gamma - 1)))]
    if kaon_ratio > 0:
        chains.append((_chain(inject, CRIT_K, Z_K_NU, BR_K_MUNU, Ec, Eg),
                       kaon_ratio * BR_K_MUNU * (Z_K_NU ** (gamma - 1) + 2 * ((1 - Z_K_NU) / 3) ** (gamma - 1))))
    f = np.zeros((3, len(Eg)))
    for c, t in chains:
        w = t / sum(t_ for _, t_ in chains)                               # the chain's share of the uncooled flux
        d_p = _ratio(c['cooled'][0], c['uncooled'][0])
        d_mu = _ratio(c['cooled'][1], c['uncooled'][1])
        f += w * np.array([d_mu, d_p + d_mu, np.zeros_like(Eg)]) / 3
    return earth_ratios(f)


def _ratio(a, b):
    """a / b, and 0 where b underflows (there the source's own flux, which multiplies it, does too)."""
    return np.divide(a, b, out=np.zeros_like(a), where=b > 1e-290)


def cooled_fractions_simple(Ec, Egrid=None, ratio=COOL_RATIO):
    """The first cooling model (until Oct 13): decay-before-cooling fractions, no pile-up, no kaons. Below Ec the
    composition is pion decay's, (1:2:0) at the source; between Ec and ratio * Ec the muon-decay neutrinos are damped,
    leaving (0:1:0) with a third of the flux; above ratio * Ec all are. Ec = inf: no cooling."""
    Eg = E if Egrid is None else Egrid
    dmu = 1. / (1. + (Eg / Ec) ** 2)
    dpi = 1. / (1. + (Eg / (ratio * Ec)) ** 2)
    return earth_ratios(np.array([dpi * dmu / 3, dpi * (1 + dmu) / 3, np.zeros_like(Eg)]))

# --------------------------------------------------------------------------
# Cross sections: Gandhi, Quigg, Reno, Sarcevic (1998), CTEQ4-DIS table [cm^2]
# --------------------------------------------------------------------------
_lgE = np.arange(1, 9, dtype=float)
_XS = {
    ('cc', 0): [0.777e-37, 0.697e-36, 0.625e-35, 0.454e-34, 0.196e-33, 0.611e-33, 0.176e-32, 0.478e-32],
    ('nc', 0): [0.242e-37, 0.217e-36, 0.199e-35, 0.155e-34, 0.745e-34, 0.252e-33, 0.748e-33, 0.207e-32],
    ('cc', 1): [0.368e-37, 0.349e-36, 0.338e-35, 0.292e-34, 0.155e-33, 0.570e-33, 0.175e-32, 0.477e-32],
    ('nc', 1): [0.130e-37, 0.122e-36, 0.120e-35, 0.106e-34, 0.611e-34, 0.247e-33, 0.755e-33, 0.206e-32],
}


def xsec(kind, bar, E):
    return 10 ** np.interp(np.log10(E), _lgE, np.log10(_XS[(kind, bar)]))


def ymean(kind, bar, E):
    """Mean inelasticity, rough: nu CC 0.45 -> 0.30, nubar 0.33 -> 0.28 over 1 TeV -> 1 PeV."""
    t = np.clip((np.log10(E) - 3) / 3, 0, 1)
    if kind == 'cc':
        return (0.45 - 0.15 * t) if bar == 0 else (0.33 - 0.05 * t)
    return (0.40 - 0.10 * t) if bar == 0 else (0.32 - 0.04 * t)


# --------------------------------------------------------------------------
# Atmospheric fluxes (MCEq, SIBYLL-2.3d x H3a; conventional + prompt)
# --------------------------------------------------------------------------
_atm = np.load(DATA / 'atm_flux.npz')
_atmE = _atm['E']
_atmC = _atm['cosZ']
_order = np.argsort(_atmC)
_ATM = {(fl, bar): _atm[('anti' if bar else '') + fl][_order] for fl in ('nue', 'numu', 'nutau') for bar in (0, 1)}
_atmC = _atmC[_order]


def atm_flux(fl, bar, E, cosz_down):
    """Interpolate MCEq table in log flux, linear in |cosZ| (down-going convention)."""
    tab = np.log(np.maximum(_ATM[(fl, bar)], 1e-300))
    c = np.clip(abs(cosz_down), _atmC[0], _atmC[-1])
    j = np.clip(np.searchsorted(_atmC, c) - 1, 0, len(_atmC) - 2)
    w = (c - _atmC[j]) / (_atmC[j + 1] - _atmC[j])
    row = (1 - w) * tab[j] + w * tab[j + 1]
    return np.exp(np.interp(np.log(E), np.log(_atmE), row))


def astro_hese(E):
    """Per-flavor (nu+nubar) diffuse astrophysical flux [GeV^-1 cm^-2 s^-1 sr^-1]: IceCube's 7.5-yr HESE single power
    law (PRD 104, 022002), 6.37e-18 (E/100 TeV)^-2.87 for the six species combined, shared equally among flavors. The
    stored results' diffuse flux; with the final inputs still the reference flux of the anchors defined with it (the
    projected double-cascade rates, detectors.nutau_dc_rate)."""
    return 2.12e-18 * (E / 1e5) ** -2.87


# IceCube's combined fit of tracks and cascades (PRD 113, 062002 (2026), Table 3, "CF BPL"): a broken power law,
# 1.77 (+0.19 -0.18) e-18 GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV, E^-2.735 (+0.067 -0.075) above 10^4.39 (+-0.1) GeV and
# E^-1.31 (+0.51 -1.30) below. Per flavor (nu + nubar): the paper writes "the total flux Phi_nu+nubar", with
# 1:1:1, and its comparison table copies each paper's own convention (HESE's all-flavor 6.37 beside the per-flavor
# 1.66 and 1.44); read as all-flavor, 1.77 would sit ~3 times below every measurement it combines (Mauricio, Oct 14:
# per flavor).
CF26 = dict(phi=1.77e-18, g1=1.31, g2=2.735, lgEb=4.39)


def astro_cf26(E):
    """Per-flavor (nu+nubar) diffuse astrophysical flux, IceCube's 2026 combined fit (CF26)."""
    E, Eb = np.asarray(E, float), 10 ** CF26['lgEb']
    above = (E / 1e5) ** -CF26['g2']
    return CF26['phi'] * np.where(E >= Eb, above, (Eb / 1e5) ** -CF26['g2'] * (E / Eb) ** -CF26['g1'])


astro_diffuse = astro_cf26 if FINAL else astro_hese        # the diffuse flux of the backgrounds


def astro_rel(E):
    """The diffuse flux's systematic templates, relative: d ln Phi / d p per unit of each parameter p. Stored inputs
    (HESE's single power law): the normalization and a tilt of the index about 100 TeV. Final inputs (the broken power
    law): the normalization, the index above the break (a tilt about 100 TeV that carries the flux below the break with
    it), the index below the break, and log10 of the break energy (the flux at 100 TeV fixed)."""
    out = {'astro_norm': np.ones_like(E), 'astro_gamma': -np.log(E / 1e5)}
    if FINAL:
        Eb, below = 10 ** CF26['lgEb'], E < 10 ** CF26['lgEb']
        out['astro_gamma'] = np.where(below, -np.log(Eb / 1e5), -np.log(E / 1e5))
        out['astro_gamma1'] = np.where(below, -np.log(E / Eb), 0.)
        out['astro_lgEb'] = np.where(below, np.log(10.) * (CF26['g1'] - CF26['g2']), 0.)
    return out


# --------------------------------------------------------------------------
# Systematic uncertainties. Fit level 0: statistics, with one background normalization per detector and channel;
# level 1 adds the global parameters below; level 2 adds per-detector ones; level 3 (an alternative analysis) also frees
# the diffuse astrophysical flux's flavor composition at Earth; level 4 frees it at its sources instead, with standard
# oscillations (Analysis(syst=...)).
# --------------------------------------------------------------------------
# Global, shared by all detectors: the Barr et al. hadron-production regions that matter above ~100 GeV (pi+ and pi-
# in G, H, I; K+ and K- in W, Y, Z; their priors come with data/atm_flux_syst.npz), the cosmic-ray spectral index,
# the prompt normalization, the diffuse astrophysical flux, and the CC and NC cross sections (relative).
BARR_PARAMS = [f'barr_{r}_{m}' for r in 'ghi' for m in ('pi+', 'pi-')] + \
              [f'barr_{r}_{m}' for r in 'wyz' for m in ('K+', 'K-')]
ASTRO_PARAMS = ['astro_norm', 'astro_gamma'] + (['astro_gamma1', 'astro_lgEb'] if FINAL else [])
GLOBAL_PARAMS = BARR_PARAMS + ['cr_dgamma', 'prompt'] + ASTRO_PARAMS + ['xs_cc', 'xs_nc']
OTHER_PRIOR = {'cr_dgamma': 0.05, 'prompt': 0.5, 'astro_norm': 0.25, 'astro_gamma': 0.2, 'xs_cc': 0.05, 'xs_nc': 0.05}
if FINAL:   # the 2026 fit's own uncertainties, symmetrized (its index below the break: +0.51 -1.30 -> 0.905)
    OTHER_PRIOR.update(astro_norm=0.105, astro_gamma=0.071, astro_gamma1=0.905, astro_lgEb=0.10)
# Level 3: the diffuse flux's flavor fractions, f_e and f_tau minus their 1/3 at 1:1:1 (f_mu = 1/3 - both), free (flat
# priors) within the flavor triangle: boxes for f_e, f_tau >= 0, and a steep quadratic wall (ASTRO_WALL) for f_mu >= 0.
# They enter the backgrounds through exact linear templates, 3 (B_e - B_mu) and 3 (B_tau - B_mu), with B_alpha a
# flavor's diffuse background at 1:1:1.
ASTRO_FLAVOR_PARAMS = ['astro_fe', 'astro_ftau']
OTHER_PRIOR.update({p: np.inf for p in ASTRO_FLAVOR_PARAMS})
NUIS_BOUNDS = {p: (-1. / 3, 2. / 3) for p in ASTRO_FLAVOR_PARAMS}
ASTRO_WALL = 0.01
# An unknown intrinsic spectrum (Analysis(curvature=True)): a log-parabola, (E/E0)^-gamma exp(-beta ln^2(E/E0)), with
# beta free (flat) within +-CURV_BOUND; it is the last fit parameter.
CURV_BOUND = 1.0
DIFF_FLAV = {'astro_fe': {'e': 3., 'mu': -3., 'tau': 0.}, 'astro_ftau': {'e': 0., 'mu': -3., 'tau': 3.}}
# Level 4: the diffuse flux's flavor composition at its sources is (x, 1 - x, 0), any mix from muon damping (x = 0)
# through pion decay (1/3) to neutron decay (1): sources make no nu_tau (Mauricio, Oct 13). It is oscillated with the
# fixed mixing, so at Earth it stays on the line standard oscillations allow. In the code the level-3 parameters act
# at the sources (a change df_S moves the composition at Earth by P df_S, so each template is a fixed combination of
# the level-3 ones: (dfe_Earth, dftau_Earth) per unit of each), with f_tau at the sources held at 0 by its bounds;
# the data keep the diffuse flux at 1:1:1, within 0.02 of pion decay's composition at Earth.
_U2 = np.abs(pmns()) ** 2
_POSC = _U2 @ _U2.T
DIFF_OSC = {'astro_fe': (_POSC[0, 0] - _POSC[0, 1], _POSC[2, 0] - _POSC[2, 1]),
            'astro_ftau': (_POSC[0, 2] - _POSC[0, 1], _POSC[2, 2] - _POSC[2, 1])}
# Per detector and channel: signal efficiency, energy scale (a shift in log10 E), angular-resolution scale, and the
# channel's misidentification (nu_mu CC as showers; showers as double cascades), all relative
DET_PRIOR = {'eff': 0.10, 'escale': np.log10(1.1), 'psf': 0.10, 'misid': 0.5}
# Down-going tracks in water (the second analysis; Detector P['trkd']): the source above cosZ = TRKD_CMIN, where water's
# up-going + horizontal tracks stop. Each reconstructed-energy bin's background is free in the fit (a 100% prior), i.e.
# measured in situ, from the rings around the source, at every fit level.
TRKD_CMIN = 0.17
TRKD_BIN_PRIOR = 1.0
_SYST = None


def atm_syst():
    """The atmospheric-flux systematics table (code/atm_flux_syst.py, run with MCEq on Tycho): relative derivatives
    dPhi/Phi per unit of each parameter, and the prompt fraction, per (parameter, flavor, nubar) on the atm_flux grid;
    and the Barr priors."""
    global _SYST
    if _SYST is None:
        z = np.load(DATA / 'atm_flux_syst.npz')
        o = np.argsort(z['cosZ'])
        assert np.allclose(z['E'], _atmE) and np.allclose(z['cosZ'][o], _atmC)
        rel = {}
        for fl in ('nue', 'numu', 'nutau'):
            for bar in (0, 1):
                s = ('anti' if bar else '') + fl
                nom = np.maximum(z['nom_' + s][o], 1e-300)
                rel[('prompt', fl, bar)] = z['pr_' + s][o] / nom
                for p in z['params']:
                    rel[(str(p), fl, bar)] = z[f'd_{p}__{s}'][o] / nom
        _SYST = dict(rel=rel, prior={str(p): float(v) for p, v in zip(z['params'], z['prior'])})
    return _SYST


def global_prior(name):
    return atm_syst()['prior'][name] if name.startswith('barr_') else OTHER_PRIOR[name]


def atm_rel(name, fl, bar, E, cosz_down):
    """A relative derivative of the atmospheric flux (or its prompt fraction), interpolated as atm_flux: linearly in
    |cosZ| and in log E."""
    tab = atm_syst()['rel'][(name, fl, bar)]
    c = np.clip(abs(cosz_down), _atmC[0], _atmC[-1])
    j = np.clip(np.searchsorted(_atmC, c) - 1, 0, len(_atmC) - 2)
    w = (c - _atmC[j]) / (_atmC[j + 1] - _atmC[j])
    return np.interp(np.log(E), np.log(_atmE), (1 - w) * tab[j] + w * tab[j + 1])


def _muon_tables():
    """Atmospheric muons (mu+ + mu-) at the surface from data/atm_flux_syst.npz: log flux, prompt fraction and relative
    derivatives, on the atm_flux grid."""
    s = atm_syst()
    if 'mu' not in s:
        z = np.load(DATA / 'atm_flux_syst.npz')
        o = np.argsort(z['cosZ'])
        nom = z['nom_mu+'][o] + z['nom_mu-'][o]
        safe = np.maximum(nom, 1e-300)
        s['mu'] = {'nom': np.log(safe), 'prompt': (z['pr_mu+'][o] + z['pr_mu-'][o]) / safe}
        for p in z['params']:
            s['mu'][str(p)] = np.where(nom > 0, (z[f'd_{p}__mu+'][o] + z[f'd_{p}__mu-'][o]) / safe, 0.)
    return s['mu']


def atm_muon(E_det, cosz_down, X, name='nom'):
    """Atmospheric muons (mu+ + mu-) at energy E_det after a slant depth X [g/cm^2] of water, from MCEq's surface flux
    with continuous energy losses (A_MU, B_MU): a muon arrives with E_det if it starts with
    E0 = (E_det + A/B) exp(B X) - A/B, and dE0/dE_det = exp(B X). Returns the flux [GeV^-1 cm^-2 s^-1 sr^-1]
    (name='nom'), its prompt part ('prompt'), or its derivative with respect to a parameter of atm_flux_syst.npz.
    Single muons: bundles are not modeled (see the bundle factor of the down-going track channel)."""
    tab = _muon_tables()
    g = np.exp(min(B_MU * X, 50.))                # beyond, no muon reaches the depth (and exp would overflow)
    E0 = (E_det + A_MU / B_MU) * g - A_MU / B_MU
    c = np.clip(abs(cosz_down), _atmC[0], _atmC[-1])
    j = np.clip(np.searchsorted(_atmC, c) - 1, 0, len(_atmC) - 2)
    w = (c - _atmC[j]) / (_atmC[j + 1] - _atmC[j])
    lnE0, lnE = np.log(E0), np.log(_atmE)
    flux = np.exp(np.interp(lnE0, lnE, (1 - w) * tab['nom'][j] + w * tab['nom'][j + 1], right=-700.)) * g
    if name == 'nom':
        return flux
    return flux * np.interp(lnE0, lnE, (1 - w) * tab[name][j] + w * tab[name][j + 1])


def earth_survival(bar, E, cosz):
    """exp(-N_A sigma_tot X); X = 2 R_E |cosZ| <rho>; <rho> = 4.5 g/cm^3 (mantle only), rising to
    8.0 g/cm^3 at the nadir for core-crossing paths (|cosZ| > 0.84). Up-going only. NC regeneration ignored."""
    if cosz >= 0:
        return np.ones_like(E)
    a = abs(cosz)
    rho = 4.5 if a < 0.84 else 4.5 + 3.5 * (a - 0.84) / 0.16
    X = 2 * 6.371e8 * a * rho
    sig = xsec('cc', bar, E) + xsec('nc', bar, E)
    return np.exp(-NA * sig * X)


# --------------------------------------------------------------------------
# Generic detector-response pieces
# --------------------------------------------------------------------------
LGE = np.linspace(2.0, 7.0, 251)          # true neutrino energy grid
E = 10 ** LGE
DLNE = np.log(10) * (LGE[1] - LGE[0])
DE = E * DLNE

A_MU, B_MU = 2.6e-3, 3.6e-6               # muon energy loss in ice/water [GeV cm^2/g], [cm^2/g]


def muon_range(Emu, Eth):
    """Range [g/cm^2] for a muon to drop from Emu to Eth."""
    return np.maximum(np.log((A_MU + B_MU * Emu) / (A_MU + B_MU * Eth)) / B_MU, 0.)


def sigmoid(Ex, E50, n=2.):
    return 1. / (1. + (E50 / np.maximum(Ex, 1e-9)) ** n)


def gauss_migration(lg_true, edges, sigma, bias=0.):
    """P(reco bin | true), Gaussian in log10 E."""
    from scipy.special import erf
    z = (edges[None, :] - (lg_true[:, None] + bias)) / (np.sqrt(2) * sigma)
    cdf = 0.5 * (1 + erf(z))
    return np.diff(cdf, axis=1)


def gauss_migration_dshift(lg_true, edges, sigma, bias=0.):
    """Derivative of gauss_migration with respect to a shift of the reconstructed log10 E (an energy-scale error)."""
    z = (edges[None, :] - (lg_true[:, None] + bias)) / (np.sqrt(2) * sigma)
    return np.diff(-np.exp(-z ** 2) / (np.sqrt(2 * np.pi) * sigma), axis=1)


def vmf_cdf(psi, psi50):
    """Fraction of a von Mises-Fisher PSF within angle psi (rad), given its median psi50 (rad)."""
    kappa = np.log(2) / (1 - np.cos(psi50))
    num = -np.expm1(-kappa * (1 - np.cos(psi)))
    den = -np.expm1(-2 * kappa)
    return num / den


RING_MULT = np.array([0., 0.5, 1., 1.5, 2., 3.])     # ring edges in units of psi50 of the bin
MAX_RING_DEG = 60.

EDGES = {'trk': np.array([2.0, 3.0, 3.3, 3.6, 4.0, 4.5, 5.0, 6.5]),
         'shw': np.array([2.0, 3.0, 3.3, 3.6, 4.0, 4.5, 5.0, 5.5, 6.5]),
         'dc':  np.array([4.8, 5.3, 5.8, 6.5])}

FLAVS = ('e', 'mu', 'tau')
ATMNAME = {'e': 'nue', 'mu': 'numu', 'tau': 'nutau'}
BR_TAU_MU = 0.174
FVIS_TAU = 0.52            # mean visible fraction of tau energy in non-muonic decays


def sky_segments(lat_deg, accept, nH=720, dec_deg=DEC_SRC):
    """Time-weighted local cosZ of a source at declination dec_deg over a sidereal day, passed through accept(cosz)."""
    H = np.linspace(0, 2 * np.pi, nH, endpoint=False)
    phi, dlt = np.radians(lat_deg), np.radians(dec_deg)
    cz = np.sin(phi) * np.sin(dlt) + np.cos(phi) * np.cos(dlt) * np.cos(H)
    edges = np.linspace(-1, 1, 41)
    idx = np.clip(np.digitize(cz, edges) - 1, 0, 39)
    segs = []
    for k in np.unique(idx):
        sel = idx == k
        c = cz[sel].mean()
        w = sel.mean()
        a = accept(c)
        if a > 0:
            segs.append((w * a, c))
    return segs


class Detector:
    """
    Parametrised detector. P is a dict with keys
      lat, years,
      dec           : optional, the source's declination [deg] (default DEC_SRC, NGC 1068's)
      trk_norm      : multiplies sigma*(range+L) for tracks [cm^2 * nucleons/g]  [CALIB]
      trk_Eth       : muon threshold [GeV]
      trk_psf(E)    : median angular error [deg] vs neutrino energy
      trk_accept(c) : sky acceptance vs local cosZ for tracks (0..1)
      trk_sig_lgE   : log10 energy resolution of the track energy proxy
      shw_V         : shower effective target volume [cm^3]; shw_rho [g/cm^3]
      shw_eff(Edep) : shower selection efficiency vs deposited energy        [CALIB]
      shw_psf(Edep) : median angular error [deg]
      shw_accept(c), shw_bkgmult(c): sky acceptance and atm-muon multiplier vs cosZ
      shw_sig_lgE, numu_misid
      dc_eff(Etrue) : fraction of nutau CC (non-muonic) identified as double cascades
      dc_misid      : fraction of showers (all flavors) classified as double cascades
      dc_misid_mu   : optional, fraction of nu_mu CC events (tracks) classified as double cascades
      channels      : optional, the channels to build among trk, shw, dc (default: all three)
      trkd          : optional, down-going tracks (the second analysis): dict(depth [m] and rho [g/cm^3] of the water
                      above the detector, lgcut [log10 GeV] reconstructed-energy cut, fb muon-bundle factor)
    """

    def __init__(self, name, P, build=True, syst=False):
        """syst=True also builds the templates the fit's systematic uncertainties need (Analysis(syst=1 or 2))."""
        self.name, self.P, self.syst = name, P, syst
        if build:
            self.build()

    # ---- effective areas [cm^2] per (channel, flavor, bar), and deposited/proxy energies
    def aeff(self, ch, fl, bar):
        P = self.P
        out = []   # list of (A(E), lg E_obs(E)) components
        if ch == 'trk':
            Eth = P['trk_Eth']
            L = 1e5 * 1.0     # 1 km of detector, g/cm^2 ~ cm (water)
            if fl == 'mu':
                Emu = (1 - ymean('cc', bar, E)) * E
                A = P['trk_norm'] * xsec('cc', bar, E) * (muon_range(Emu, Eth) + L) * sigmoid(Emu, Eth)
                out.append((A, np.log10(Emu)))
            if fl == 'tau':
                Emu = 0.35 * (1 - ymean('cc', bar, E)) * E
                A = BR_TAU_MU * P['trk_norm'] * xsec('cc', bar, E) * (muon_range(Emu, Eth) + L) * sigmoid(Emu, Eth)
                out.append((A, np.log10(Emu)))
            return out
        Nn = P['shw_V'] * P['shw_rho'] * NA
        if ch == 'shw':
            y = ymean('cc', bar, E)
            if fl == 'e':
                Ed = E
                out.append((Nn * xsec('cc', bar, E) * P['shw_eff'](Ed), np.log10(Ed)))
            if fl == 'tau':
                Ed = (y + (1 - y) * FVIS_TAU) * E
                A = Nn * xsec('cc', bar, E) * (1 - BR_TAU_MU) * (1 - P['dc_eff'](E)) * P['shw_eff'](Ed)
                out.append((A, np.log10(Ed)))
            if fl == 'mu':
                Ed = 0.6 * E
                m = P['numu_misid']                       # a constant, or a function of the neutrino energy
                out.append((Nn * xsec('cc', bar, E) * (m(E) if callable(m) else m) * P['shw_eff'](Ed), np.log10(Ed)))
            yn = ymean('nc', bar, E)
            Ed = yn * E
            out.append((Nn * xsec('nc', bar, E) * P['shw_eff'](Ed), np.log10(Ed)))
            return out
        if ch == 'dc':
            if fl == 'tau':
                y = ymean('cc', bar, E)
                Ed = (y + (1 - y) * FVIS_TAU) * E
                A = Nn * xsec('cc', bar, E) * (1 - BR_TAU_MU) * P['dc_eff'](E) * P['shw_eff'](Ed)
                out.append((A, np.log10(Ed)))
            # single showers mis-identified as double cascades (all flavors)
            for A, lg in self.aeff('shw', fl, bar):
                out.append((P['dc_misid'] * A, lg))
            if fl == 'mu' and P.get('dc_misid_mu'):       # nu_mu CC events (tracks) classified as double cascades
                Ed = 0.6 * E
                out.append((Nn * xsec('cc', bar, E) * P['dc_misid_mu'] * P['shw_eff'](Ed), np.log10(Ed)))
            return out
        raise ValueError(ch)

    def aeff_tags(self, ch, fl):
        """For each component of aeff(ch, fl, bar), in the same order: its interaction ('cc' or 'nc') and whether it
        scales with the channel's misidentification (nu_mu CC as showers; showers as double cascades)."""
        if ch == 'trk':
            return [('cc', False)] if fl in ('mu', 'tau') else []
        shw = {'e': [('cc', False), ('nc', False)], 'mu': [('cc', True), ('nc', False)],
               'tau': [('cc', False), ('nc', False)]}[fl]
        if ch == 'shw':
            return shw
        return (([('cc', False)] if fl == 'tau' else []) + [(kind, True) for kind, _ in shw]
                + ([('cc', True)] if fl == 'mu' and self.P.get('dc_misid_mu') else []))

    def psf(self, ch, Eobs):
        return self.P['trk_psf'](Eobs) if ch == 'trk' else self.P['shw_psf'](Eobs)

    def segments(self, ch):
        P = self.P
        acc = P['trk_accept'] if ch == 'trk' else P['shw_accept']
        return sky_segments(P['lat'], acc, dec_deg=P.get('dec', DEC_SRC))

    def bkgmult(self, ch, c, lgobs=None):
        """Background multiplier for atmospheric muons in the shower selection: shw_bkgmult(c), or, if set,
        shw_bkgmult_E(c, E_dep), which may depend on the deposited energy (an array over the energy grid)."""
        if ch == 'trk':
            return 1.0
        fE = self.P.get('shw_bkgmult_E')
        return self.P['shw_bkgmult'](c) if fE is None else fE(c, 10 ** lgobs)

    def gate(self, ch, c, lgobs):
        """Extra efficiency for down-going showers / double cascades (atm-muon rejection): shw_down_gate(E_dep) if
        set, else a sharp cut at shw_down_Emin, if set."""
        if ch == 'trk' or c < 0:
            return 1.0
        g = self.P.get('shw_down_gate')
        if g is not None:
            return g(10 ** lgobs)
        Emin = self.P.get('shw_down_Emin')
        return 1.0 if Emin is None else sigmoid(10 ** lgobs, Emin, 8)

    def build(self):
        """Signal response R[ch][fl] (nE, nbin) and background B[ch] (nbin). With syst=True also the templates for the
        fit's systematics: GB[ch][p] (background, per global parameter p), RX[ch][fl][p] (signal, for p = xs_cc, xs_nc,
        escale, psf, misid) and BX[ch][p] (background, for p = escale, misid), each per unit of its parameter."""
        P = self.P
        T = P['years'] * YR
        self.R, self.B, self.BX, self.channels = {}, {}, {}, []
        if self.syst:
            self.GB, self.RX = {}, {}
            h = 0.05                                       # finite-difference step of the PSF scale
        for ch in P.get('channels', ('trk', 'shw', 'dc')):
            ed = EDGES[ch]
            nb = len(ed) - 1
            sig_lg = P['trk_sig_lgE'] if ch == 'trk' else P['shw_sig_lgE']
            # ring edges per reco-energy bin
            lgc = 0.5 * (ed[1:] + ed[:-1])
            psi50_bin = self.psf(ch, 10 ** lgc) * DEG
            ring_edges = np.minimum(RING_MULT[None, :] * psi50_bin[:, None], MAX_RING_DEG * DEG)   # (nb, nr+1)
            nr = len(RING_MULT) - 1
            omega = 2 * np.pi * (np.cos(ring_edges[:, :-1]) - np.cos(ring_edges[:, 1:]))          # (nb, nr)
            segs = self.segments(ch)
            Rch = {}
            Bch = np.zeros((nb, nr))
            if self.syst:
                GB = {p: np.zeros((nb, nr)) for p in GLOBAL_PARAMS + ASTRO_FLAVOR_PARAMS}
                arel = astro_rel(E)
                BX = {p: np.zeros((nb, nr)) for p in ('escale', 'misid')}
                RX = {fl: {p: np.zeros((len(E), nb, nr)) for p in ('xs_cc', 'xs_nc', 'escale', 'psf', 'misid')}
                      for fl in FLAVS}
            for fl in FLAVS:
                Rfl = np.zeros((len(E), nb, nr))
                for bar in (0, 1):
                    comps = self.aeff(ch, fl, bar)
                    tags = self.aeff_tags(ch, fl) if self.syst else [None] * len(comps)
                    assert len(tags) == len(comps)
                    for (A, lgobs), tag in zip(comps, tags):
                        mig = gauss_migration(lgobs, ed, sig_lg, bias=(-0.15 if ch == 'trk' else 0.))   # (nE, nb)
                        # tracks: PSF set by neutrino energy (kinematics); showers: by deposited energy
                        psi50_true = (self.psf(ch, E) if ch == 'trk' else self.psf(ch, 10 ** lgobs)) * DEG
                        # ring fractions for each true energy and bin: (nE, nb, nr)
                        cdf = vmf_cdf(ring_edges[None, :, :], psi50_true[:, None, None])
                        rfrac = np.diff(cdf, axis=2)
                        if self.syst:
                            kind, mis = tag
                            dmig = gauss_migration_dshift(lgobs, ed, sig_lg, bias=(-0.15 if ch == 'trk' else 0.))
                            drfrac = (np.diff(vmf_cdf(ring_edges[None], (psi50_true * (1 + h))[:, None, None]), axis=2)
                                      - np.diff(vmf_cdf(ring_edges[None], (psi50_true * (1 - h))[:, None, None]),
                                                axis=2)) / (2 * h)
                        for w, c in segs:
                            surv = earth_survival(bar, E, c) * self.gate(ch, c, lgobs)
                            # signal: half nu, half nubar (pp source); response per unit dPhi/dE
                            Rfl += 0.5 * w * T * (A * surv * DE)[:, None, None] * mig[:, :, None] * rfrac
                            # background: atm (fl, bar) + astro diffuse (per flavor, half nubar), per sr
                            phi_b = atm_flux(ATMNAME[fl], bar, E, c) + 0.5 * astro_diffuse(E)
                            m = self.bkgmult(ch, c, lgobs)
                            if np.ndim(m):                             # energy-dependent multiplier
                                rate = (phi_b * A * surv * DE * m) @ mig
                                Bch += w * T * rate[:, None] * omega
                            else:
                                rate = (phi_b * A * surv * DE) @ mig       # per sr per s, (nb,)
                                Bch += w * T * m * rate[:, None] * omega
                            if self.syst:
                                base = 0.5 * w * T * (A * surv * DE)[:, None, None]
                                RX[fl]['xs_' + kind] += base * mig[:, :, None] * rfrac
                                RX[fl]['escale'] += base * dmig[:, :, None] * rfrac
                                RX[fl]['psf'] += base * mig[:, :, None] * drfrac
                                if mis:
                                    RX[fl]['misid'] += base * mig[:, :, None] * rfrac
                                aw = w * T * A * surv * DE * m             # m: scalar or energy-dependent
                                atm = atm_flux(ATMNAME[fl], bar, E, c)
                                ast = 0.5 * astro_diffuse(E)

                                def add(tpl, key, phi, mg=mig):
                                    tpl[key] += ((phi * aw) @ mg)[:, None] * omega
                                for p in BARR_PARAMS + ['cr_dgamma', 'prompt']:
                                    add(GB, p, atm_rel(p, ATMNAME[fl], bar, E, c) * atm)
                                for p, r in arel.items():          # the diffuse flux's normalization and shape
                                    add(GB, p, r * ast)
                                for p in ASTRO_FLAVOR_PARAMS:
                                    add(GB, p, DIFF_FLAV[p][fl] * ast)
                                add(GB, 'xs_' + kind, atm + ast)
                                add(BX, 'escale', atm + ast, dmig)
                                if mis:
                                    add(BX, 'misid', atm + ast)
                Rch[fl] = Rfl.reshape(len(E), nb * nr)
            self.R[ch], self.B[ch] = Rch, Bch.reshape(-1)
            if self.syst:
                self.GB[ch] = {p: v.reshape(-1) for p, v in GB.items()}
                self.BX[ch] = {p: v.reshape(-1) for p, v in BX.items()}
                self.RX[ch] = {fl: {p: v.reshape(len(E), nb * nr) for p, v in RX[fl].items()} for fl in FLAVS}
            self.channels.append(ch)
        if 'trkd' in P:
            self._build_trkd(T)

    def _build_trkd(self, T):
        """The down-going track channel 'trkd' (the second analysis): nu_mu CC and nu_tau -> mu while the source is above
        cosZ = TRKD_CMIN, in reconstructed-energy bins of 0.5 in log10 E from P['trkd']['lgcut'] to 10^6.5 GeV.
        Signal: as tracks, but the muon must start in the water above the detector or in it (its range is capped at
        the slant depth), with no Earth absorption. Background: atmospheric and diffuse neutrinos, as for tracks, and
        atmospheric muons at the slant depth (atm_muon), through the muon area trk_norm / N_A, times fb, the ratio of
        muon bundles to single muons (calibrated in detectors.py); their energy proxy is their energy at the detector.
        Each energy bin's background also has its own template, BX['trkd']['bkg<i>'] (free in the fit); self.trkd_mu
        keeps the atmospheric-muon part of B['trkd']. With syst=True also the templates of build()."""
        P, Q = self.P, self.P['trkd']
        ed = np.arange(Q['lgcut'], 6.5 + 1e-9, 0.5)
        nb, nr = len(ed) - 1, len(RING_MULT) - 1
        sig_lg, Eth, L = P['trk_sig_lgE'], P['trk_Eth'], 1e5
        lgc = 0.5 * (ed[1:] + ed[:-1])
        ring_edges = np.minimum(RING_MULT[None, :] * (P['trk_psf'](10 ** lgc) * DEG)[:, None], MAX_RING_DEG * DEG)
        omega = 2 * np.pi * (np.cos(ring_edges[:, :-1]) - np.cos(ring_edges[:, 1:]))          # (nb, nr)
        psi50 = P['trk_psf'](E) * DEG                     # set by the neutrino energy, as for tracks
        rfrac = np.diff(vmf_cdf(ring_edges[None], psi50[:, None, None]), axis=2)                # (nE, nb, nr)
        segs = sky_segments(P['lat'], lambda c: 1.0 if c >= TRKD_CMIN else 0.0, dec_deg=P.get('dec', DEC_SRC))
        R = {fl: np.zeros((len(E), nb, nr)) for fl in FLAVS}
        bnu, bmu = np.zeros(nb), np.zeros(nb)             # backgrounds per sr: neutrinos, atmospheric muons
        if self.syst:
            h = 0.05
            drfrac = (np.diff(vmf_cdf(ring_edges[None], (psi50 * (1 + h))[:, None, None]), axis=2)
                      - np.diff(vmf_cdf(ring_edges[None], (psi50 * (1 - h))[:, None, None]), axis=2)) / (2 * h)
            GB = {p: np.zeros(nb) for p in GLOBAL_PARAMS + ASTRO_FLAVOR_PARAMS}
            arel = astro_rel(E)
            bes = np.zeros(nb)
            RX = {fl: {p: np.zeros((len(E), nb, nr)) for p in ('xs_cc', 'xs_nc', 'escale', 'psf')} for fl in FLAVS}
        Xs = [100. * Q['depth'] * Q['rho'] / c for _, c in segs]         # slant depth of water [g/cm^2]
        for fl, frac, fmu in (('mu', 1., 1.), ('tau', BR_TAU_MU, 0.35)):
            for bar in (0, 1):
                Emu = fmu * (1 - ymean('cc', bar, E)) * E
                mig = gauss_migration(np.log10(Emu), ed, sig_lg, bias=-0.15)
                if self.syst:
                    dmig = gauss_migration_dshift(np.log10(Emu), ed, sig_lg, bias=-0.15)
                for (w, c), X in zip(segs, Xs):
                    A = (frac * P['trk_norm'] * xsec('cc', bar, E) * (np.minimum(muon_range(Emu, Eth), X) + L)
                         * sigmoid(Emu, Eth))
                    base = 0.5 * w * T * (A * DE)[:, None, None]
                    R[fl] += base * mig[:, :, None] * rfrac
                    atm, ast = atm_flux(ATMNAME[fl], bar, E, c), 0.5 * astro_diffuse(E)
                    aw = w * T * A * DE
                    bnu += ((atm + ast) * aw) @ mig
                    if self.syst:
                        RX[fl]['xs_cc'] += base * mig[:, :, None] * rfrac
                        RX[fl]['escale'] += base * dmig[:, :, None] * rfrac
                        RX[fl]['psf'] += base * mig[:, :, None] * drfrac
                        for p in BARR_PARAMS + ['cr_dgamma', 'prompt']:
                            GB[p] += (atm_rel(p, ATMNAME[fl], bar, E, c) * atm * aw) @ mig
                        for p, r in arel.items():          # the diffuse flux's normalization and shape
                            GB[p] += (r * ast * aw) @ mig
                        for p in ASTRO_FLAVOR_PARAMS:
                            GB[p] += (DIFF_FLAV[p][fl] * ast * aw) @ mig
                        GB['xs_cc'] += ((atm + ast) * aw) @ mig
                        bes += ((atm + ast) * aw) @ dmig
        mig = gauss_migration(LGE, ed, sig_lg)             # atmospheric muons: their energy at the detector
        if self.syst:
            dmig = gauss_migration_dshift(LGE, ed, sig_lg)
        for (w, c), X in zip(segs, Xs):
            aw = w * T * Q['fb'] * P['trk_norm'] / NA * sigmoid(E, Eth) * DE
            phi = atm_muon(E, c, X)
            bmu += (phi * aw) @ mig
            if self.syst:
                for p in BARR_PARAMS + ['cr_dgamma', 'prompt']:
                    GB[p] += (atm_muon(E, c, X, p) * aw) @ mig
                bes += (phi * aw) @ dmig
        Bm = (bnu + bmu)[:, None] * omega
        self.R['trkd'] = {fl: R[fl].reshape(len(E), nb * nr) for fl in FLAVS}
        self.B['trkd'] = Bm.reshape(-1)
        self.trkd_mu = (bmu[:, None] * omega).reshape(-1)
        self.BX['trkd'] = {f'bkg{i}': (Bm * (np.arange(nb) == i)[:, None]).reshape(-1) for i in range(nb)}
        if self.syst:
            self.GB['trkd'] = {p: (v[:, None] * omega).reshape(-1) for p, v in GB.items()}
            self.BX['trkd']['escale'] = (bes[:, None] * omega).reshape(-1)
            self.RX['trkd'] = {fl: {p: v.reshape(len(E), nb * nr) for p, v in RX[fl].items()} for fl in FLAVS}
        self.channels.append('trkd')

    def source_counts(self, ch, flux, f):
        """Total expected signal in a channel (all energies and angles). flux = all-flavor dPhi/dE on grid E."""
        T = self.P['years'] * YR
        segs = self.segments(ch)
        n = 0.
        for i, fl in enumerate(FLAVS):
            # f[i]: a flavor fraction, or (energy-dependent composition) its value on grid E
            fx, fi = (f[i], 1.) if np.ndim(f) == 2 else (1., f[i])
            for bar in (0, 1):
                for A, lg in self.aeff(ch, fl, bar):
                    for w, c in segs:
                        n += 0.5 * fi * w * T * np.sum(flux * fx * A * earth_survival(bar, E, c) * self.gate(ch, c, lg)
                                                        * DE)
        return n

    # ---- convenience
    def signal_counts(self, flux, f):
        """Expected signal per channel (flattened bins). flux = all-flavor dPhi/dE on grid E; f = the flavor fractions,
        or their values on grid E (shape (3, nE)) for a composition that changes with energy."""
        if np.ndim(f) == 2:
            return {ch: sum((flux * f[i]) @ self.R[ch][fl] for i, fl in enumerate(FLAVS)) for ch in self.channels}
        return {ch: sum(f[i] * (flux @ self.R[ch][fl]) for i, fl in enumerate(FLAVS)) for ch in self.channels}

    def allsky_rate(self, ch, accept=lambda c: 1.0, mult=lambda c: 1.0, flux='atm', ecut=None, smear=False):
        """All-sky event rate [1/yr] for calibration (no angular cut): atmospheric (flux='atm'), the diffuse
        astrophysical flux ('astro'), or flux(E), a per-flavor nu+nubar flux [GeV^-1 cm^-2 s^-1 sr^-1]. ecut [GeV] cuts
        on the energy proxy; with smear=True it cuts on the reconstructed proxy, smeared and biased as in build()."""
        from scipy.special import erf
        P = self.P
        cz = np.linspace(-0.995, 0.995, 200)
        dc = cz[1] - cz[0]
        tot = 0.
        for fl in FLAVS:
            for bar in (0, 1):
                for A, lgobs in self.aeff(ch, fl, bar):
                    if ecut is None or not smear:
                        sel = np.ones_like(E) if ecut is None else (lgobs >= np.log10(ecut)).astype(float)
                    else:
                        sig, bias = (P['trk_sig_lgE'], -0.15) if ch == 'trk' else (P['shw_sig_lgE'], 0.)
                        sel = 0.5 * (1 - erf((np.log10(ecut) - (lgobs + bias)) / (np.sqrt(2) * sig)))
                    for c in cz:
                        a = accept(c)
                        if a == 0:
                            continue
                        phi = (atm_flux(ATMNAME[fl], bar, E, c) if flux == 'atm' else
                               0.5 * astro_diffuse(E) if flux == 'astro' else 0.5 * flux(E))
                        tot += a * mult(c) * 2 * np.pi * dc * np.sum(phi * A * earth_survival(bar, E, c) * DE * sel
                                                                     * self.gate(ch, c, lgobs))
        return tot * YR


# --------------------------------------------------------------------------
# Likelihood
# --------------------------------------------------------------------------
class Analysis:
    """Profiled Poisson likelihood. Parameters x: log10 normalization, spectral index, one background normalization
    per block (detector, channel) with a 10% prior; with syst=1 also the global systematics (GLOBAL_PARAMS); with
    syst=2 also, per block, the signal efficiency, energy scale, angular resolution and misidentification (DET_PRIOR);
    with syst=3 also the diffuse flux's flavor composition at Earth (ASTRO_FLAVOR_PARAMS, free); with syst=4 instead its
    composition at its sources, (x, 1 - x, 0) with x free, oscillated (DIFF_OSC), so that at Earth it stays where
    standard oscillations allow. With curvature=True the
    spectrum also has a free log-parabola curvature beta (the last parameter; an unknown intrinsic spectrum).
    Down-going track blocks ('trkd') also get one background normalization per energy bin (TRKD_BIN_PRIOR), at every
    level. Every nuisance enters linearly through a template, with a Gaussian prior around 0."""

    def __init__(self, dets, src=NGC1068, bkg_prior=0.10, syst=0, det_params=tuple(DET_PRIOR),
                 global_params=tuple(GLOBAL_PARAMS), curvature=False, share_det=False):
        """det_params, global_params: which per-detector (level 2) and global (levels 1, 2) systematics to include;
        all by default, fewer to attribute a loss. share_det=True: blocks of detectors with the same name (one detector
        built for several sources, for a joint fit of a stack) share their per-detector systematics, with one prior
        each; off by default, when every block has its own (names are unique within a configuration anyway)."""
        self.dets, self.src = dets, src
        self.share_det = share_det
        self.det_params, self.global_params = det_params, global_params
        self.shape = src.shape(E)
        self.blocks = []   # (det, ch)
        for d in dets:
            for ch in d.channels:
                self.blocks.append((d, ch))
        self.Rs = [{fl: d.R[ch][fl] for fl in FLAVS} for d, ch in self.blocks]
        self.Bs = [d.B[ch] for d, ch in self.blocks]
        self.sb = bkg_prior
        self.lnE = np.log(E / src.E0)
        self.lnE2 = self.lnE ** 2
        self.curv = curvature
        self.syst = syst
        # templates: for the systematics, or for the down-going tracks' per-bin backgrounds (also at level 0)
        self.templates = bool(syst) or any(ch == 'trkd' for _, ch in self.blocks)
        self.nuis, self.sig = [], np.zeros(0)     # nuisance names and 1-sigma priors, after the background norms
        self.wall = None                          # level 3: the indices of the diffuse flux's f_e and f_tau
        if self.templates:
            self._setup_syst()

    def _setup_syst(self):
        K = len(self.blocks)
        gp = [p for p in GLOBAL_PARAMS if p in self.global_params] if self.syst else []
        if self.syst >= 3:
            gp += ASTRO_FLAVOR_PARAMS
        names = list(gp)
        sig = [global_prior(p) for p in names]
        gidx = {p: 2 + K + i for i, p in enumerate(names)}
        if self.syst >= 3:
            self.wall = (gidx['astro_fe'], gidx['astro_ftau'])
        self.Rcat, self.sidx, self.eidx, self.Bcat, self.bidx = [], [], [], [], []
        for k, (d, ch) in enumerate(self.blocks):
            assert d.syst or not self.syst, f'{d.name} was built without syst=True'
            terms = [t for t in ('xs_cc', 'xs_nc') if t in gidx]     # signal templates, besides the nominal one
            sidx = [gidx[t] for t in terms]
            bterms = [d.B[ch]] + [self._gtemplate(d, ch, p) for p in gp]
            bidx = [2 + k] + [gidx[p] for p in gp]
            eidx = None
            if self.syst >= 2:
                per = [p for p in DET_PRIOR if p in self.det_params and (p != 'misid' or ch in ('shw', 'dc'))]
                own = {}
                for p in per:
                    key = f'{d.name}|{ch}|{p}'
                    if self.share_det and key in names:          # the same detector, built for another source
                        own[p] = 2 + K + names.index(key)
                    else:
                        own[p] = 2 + K + len(names)
                        names.append(key)
                        sig.append(DET_PRIOR[p])
                eidx = own.get('eff')
                for p in per:
                    if p != 'eff':
                        terms.append(p)
                        sidx.append(own[p])
                for p in ('escale', 'misid'):
                    if p in own:
                        bterms.append(d.BX[ch][p])
                        bidx.append(own[p])
            if ch == 'trkd':                             # each energy bin's background, measured in situ
                for i in range(sum(key.startswith('bkg') for key in d.BX[ch])):
                    bterms.append(d.BX[ch][f'bkg{i}'])
                    bidx.append(2 + K + len(names))
                    names.append(f'{d.name}|{ch}|bkg{i}')
                    sig.append(TRKD_BIN_PRIOR)
            self.Rcat.append({fl: np.hstack([d.R[ch][fl]] + [d.RX[ch][fl][t] for t in terms]) for fl in FLAVS})
            self.sidx.append(np.array(sidx, dtype=int))      # integer even when empty (no signal templates)
            self.eidx.append(eidx)
            self.Bcat.append(np.array(bterms))
            self.bidx.append(np.array(bidx, dtype=int))
        self.nuis, self.sig = names, np.array(sig)

    def _nuis_bounds(self, n, sg):
        if self.syst >= 4 and n == 'astro_ftau':
            return (-1. / 3, -1. / 3)             # no nu_tau at the diffuse flux's sources: (x, 1 - x, 0)
        return NUIS_BOUNDS.get(n, (max(-5 * sg, -0.9), 5 * sg))

    def _gtemplate(self, d, ch, p):
        """A global parameter's background template; at level 4 the diffuse flux's flavor parameters act at its
        sources, so their templates are the oscillated combinations of the level-3 ones."""
        if self.syst >= 4 and p in DIFF_OSC:
            a, b = DIFF_OSC[p]
            return a * d.GB[ch]['astro_fe'] + b * d.GB[ch]['astro_ftau']
        return d.GB[ch][p]

    def x0(self, lphi, g):
        """Starting point: the given normalization and index, background norms 1, nuisances 0 (and no curvature)."""
        return np.r_[lphi, g, np.ones(len(self.blocks)), np.zeros(len(self.nuis) + bool(self.curv))]

    def _w(self, x):
        """The source's weights on grid E: normalization, index, shape and (curvature=True) curvature."""
        w = 10 ** x[0] * np.exp(-x[1] * self.lnE) * self.shape
        return w * np.exp(-x[-1] * self.lnE2) if self.curv else w

    def mu(self, x, f):
        if self.templates:
            return self._mu_syst(x, f)
        w = self._w(x)
        mus, dmu_dg, Svecs, self._db = [], [], [], []
        for k, (R, B) in enumerate(zip(self.Rs, self.Bs)):
            if np.ndim(f) == 2:                     # a composition that changes with energy: f[i] on grid E
                s = sum((w * f[i]) @ R[fl] for i, fl in enumerate(FLAVS))
                ds = -sum((w * self.lnE * f[i]) @ R[fl] for i, fl in enumerate(FLAVS))
                if self.curv:
                    self._db.append(-sum((w * self.lnE2 * f[i]) @ R[fl] for i, fl in enumerate(FLAVS)))
            else:
                S = sum(f[i] * R[fl] for i, fl in enumerate(FLAVS))       # (nE, nb)
                s = w @ S
                ds = -(w * self.lnE) @ S
                if self.curv:
                    self._db.append(-(w * self.lnE2) @ S)
            mus.append(s + x[2 + k] * B)
            dmu_dg.append(ds)
            Svecs.append(s)
        return mus, Svecs, dmu_dg

    def _mu_syst(self, x, f):
        """As mu(), with the templates: per block also the signal-template vectors (for the gradient)."""
        w = self._w(x)
        wl = w * self.lnE
        mus, Svecs, dmu_dg, V, self._db = [], [], [], [], []
        for k in range(len(self.blocks)):
            Rc = self.Rcat[k]
            nb = len(self.Bs[k])
            if np.ndim(f) == 2:                     # a composition that changes with energy: f[i] on grid E
                v = sum((w * f[i]) @ Rc[fl] for i, fl in enumerate(FLAVS)).reshape(-1, nb)
                vl = sum((wl * f[i]) @ Rc[fl] for i, fl in enumerate(FLAVS)).reshape(-1, nb)
            else:
                v = sum(f[i] * (w @ Rc[fl]) for i, fl in enumerate(FLAVS)).reshape(-1, nb)      # (1 + T, nb)
                vl = sum(f[i] * (wl @ Rc[fl]) for i, fl in enumerate(FLAVS)).reshape(-1, nb)
            coef = np.r_[1. + (x[self.eidx[k]] if self.eidx[k] is not None else 0.), x[self.sidx[k]]]
            s = coef @ v
            mus.append(s + x[self.bidx[k]] @ self.Bcat[k])
            Svecs.append(s)
            dmu_dg.append(-(coef @ vl))
            V.append(v)
            if self.curv:
                wq = w * self.lnE2
                if np.ndim(f) == 2:
                    vq = sum((wq * f[i]) @ Rc[fl] for i, fl in enumerate(FLAVS)).reshape(-1, nb)
                else:
                    vq = sum(f[i] * (wq @ Rc[fl]) for i, fl in enumerate(FLAVS)).reshape(-1, nb)
                self._db.append(-(coef @ vq))
        self._V = V
        return mus, Svecs, dmu_dg

    def asimov(self, f, lphi, g):
        return self.mu(self.x0(lphi, g), f)[0]

    def chi2(self, x, f, data):
        mus, Svecs, dmu_dg = self.mu(x, f)
        c2, grad = 0., np.zeros_like(x)
        for k, (m, n, s, dg) in enumerate(zip(mus, data, Svecs, dmu_dg)):
            m = np.maximum(m, 1e-12)
            pos = n > 0
            c2 += 2 * np.sum(m - n) + 2 * np.sum(n[pos] * np.log(n[pos] / m[pos]))
            r = 2 * (1 - n / m)
            grad[0] += np.sum(r * s) * np.log(10)
            grad[1] += np.sum(r * dg)
            if self.curv:
                grad[-1] += np.sum(r * self._db[k])
            if self.templates:
                v = self._V[k]
                np.add.at(grad, self.sidx[k], v[1:] @ r)
                if self.eidx[k] is not None:
                    grad[self.eidx[k]] += v[0] @ r
                np.add.at(grad, self.bidx[k], self.Bcat[k] @ r)
            else:
                grad[2 + k] += np.sum(r * self.Bs[k])
            c2 += ((x[2 + k] - 1) / self.sb) ** 2
            grad[2 + k] += 2 * (x[2 + k] - 1) / self.sb ** 2
        if self.templates:
            K = len(self.blocks)
            N = len(self.sig)
            t = x[2 + K:2 + K + N] / self.sig
            c2 += np.sum(t ** 2)
            grad[2 + K:2 + K + N] += 2 * t / self.sig
        if self.wall is not None:                 # the diffuse flux's f_mu = 1/3 - (f_e - 1/3) - (f_tau - 1/3) >= 0
            ex = x[self.wall[0]] + x[self.wall[1]] - 1. / 3
            if ex > 0:
                c2 += (ex / ASTRO_WALL) ** 2
                grad[list(self.wall)] += 2 * ex / ASTRO_WALL ** 2
        return c2, grad

    lphi_window = 3.0         # the normalization is fitted within +-lphi_window decades of the source's

    def profile(self, f, data, x0):
        lt = self.src.lphi_true
        w = self.lphi_window
        bounds = [(lt - w, lt + w), (1.0, 5.0)] + [(0.2, 3.0)] * len(self.blocks)
        # nuisances within 5 sigma, and no relative change below -0.9 (efficiencies, misidentification, ...); the
        # diffuse flux's flavor fractions (levels 3, 4) within their own ranges
        bounds += [self._nuis_bounds(n, sg) for n, sg in zip(self.nuis, self.sig)]
        if self.curv:
            bounds.append((-CURV_BOUND, CURV_BOUND))
        res = minimize(self.chi2, x0, args=(f, data), jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500 if not self.templates else 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.fun, res.x


def walk_profile(analysis, f, data, f_from, x_from, steps=10):
    """The profile at f reached in `steps` warm-started steps along the straight line from f_from, where x_from is the
    best fit: the scans' outward walk, for one point. f and f_from may be compositions on the energy grid."""
    f, f_from = np.asarray(f, float), np.asarray(f_from, float)
    if f.ndim != f_from.ndim:              # a constant composition against one on the energy grid
        f, f_from = (np.broadcast_to(f[:, None], f_from.shape) if f.ndim == 1 else f,
                     np.broadcast_to(f_from[:, None], f.shape) if f_from.ndim == 1 else f_from)
    x = np.array(x_from, float)
    for a in np.linspace(1. / steps, 1., steps):
        c, x = analysis.profile((1. - a) * f_from + a * f, data, x)
    return c, x


def fine_walk_profile(analysis, f, data, f_from, x_from, step=0.02):
    """The scan's own walk for one point: from f_from (best fit x_from) to f in steps no larger than the scan's grid
    step in any flavor fraction, each step started from the previous best fit and from the same with the normalization
    tracks-matched (the nu_mu flux at the truth's), keeping the lower. f and f_from are constant compositions."""
    f, f_from = np.asarray(f, float), np.asarray(f_from, float)
    n = max(10, int(np.ceil(np.abs(f - f_from).max() / step)))
    x = np.array(x_from, float)
    for a in np.linspace(1. / n, 1., n):
        fi = (1. - a) * f_from + a * f
        xt = x.copy()
        xt[0] = np.log10(analysis.src.phi_numu_E0 / max(fi[1], 0.02))
        c1, x1 = analysis.profile(fi, data, x)
        c2, x2 = analysis.profile(fi, data, xt)
        c, x = (c1, x1) if c1 <= c2 else (c2, x2)
    return c, x


def best_profile(analysis, f, data, starts, f_from, x_from, steps=10):
    """The lowest chi^2 at f, from direct starts and from walks from f_from (the truth, with its best fit x_from): ten
    warm-started steps, and the scan's own finer walk (fine_walk_profile). A fit started directly far from the truth
    can stop at a false minimum, with chi^2 too high: e.g. a TXS-like flare ten times brighter, all seven detectors,
    all systematic uncertainties, against neutron decay, 83 from either direct start and 55 walked; for a neutron-
    decay truth far from muon damping, the finer walk also matters (Oct 14). The scans walk outward from the truth, so
    they are immune; the benchmark-only studies use this."""
    c = min(analysis.profile(np.asarray(f, float), data, s)[0] for s in starts)
    c = min(c, walk_profile(analysis, f, data, f_from, x_from, steps)[0])
    return min(c, fine_walk_profile(analysis, f, data, f_from, x_from)[0])


def edge_min(analysis, data, starts, f_from, x_from, n=51, steps=10):
    """The lowest chi^2 along the f_tau = 0 edge, (f_e, 1 - f_e, 0) at n points: each point from its direct starts
    (starts(f), a list), and the whole edge swept outward from its point nearest f_from, reached by both walks from
    f_from (with its best fit x_from), each point started from its neighbour's best fit and from the same with the
    normalization tracks-matched, as the scan does."""
    edge = [np.array((fe, 1. - fe, 0.)) for fe in np.linspace(0., 1., n)]
    best = min(min(analysis.profile(f, data, s)[0] for s in starts(f)) for f in edge)
    i0 = int(np.argmin([np.linalg.norm(f - np.asarray(f_from, float)) for f in edge]))
    ca, xa = walk_profile(analysis, edge[i0], data, f_from, x_from, steps)
    cb, xb = fine_walk_profile(analysis, edge[i0], data, f_from, x_from)
    c, x0 = (ca, xa) if ca <= cb else (cb, xb)
    best = min(best, c)
    for d in (1, -1):
        x, i = x0, i0 + d
        while 0 <= i < n:
            xt = x.copy()
            xt[0] = np.log10(analysis.src.phi_numu_E0 / max(edge[i][1], 0.02))
            c1, x1 = analysis.profile(edge[i], data, x)
            c2, x2 = analysis.profile(edge[i], data, xt)
            c, x = (c1, x1) if c1 <= c2 else (c2, x2)
            best = min(best, c)
            i += d
    return best


class StackAnalysis:
    """Several sources fitted together (Oct 14): each with its own normalization, spectral index and background
    normalizations, all with the same flavor composition at Earth, and the systematic uncertainties shared among
    them, the global ones and each detector's own (by detector name), each with one prior. Built from one Analysis
    per source (same fit level, no curvature); the parameter vector is [lphi, gamma per source; background
    normalizations per source; the union of the nuisances]. Summing single-source fits instead gives every source its
    own copy of each systematic; when all sources pull it the same way, as a flavor change does, that overstates
    Delta chi^2 (five NGC 1068-like sources, all seven, against neutron decay: 23.9 summed, 16.0 joint). With
    statistics only the two agree."""
    lphi_window = 3.0

    def __init__(self, analyses):
        self.an = list(analyses)
        assert not any(a.curv for a in self.an) and len({a.syst for a in self.an}) == 1
        S = len(self.an)
        self.names = []
        for a in self.an:
            self.names += [n for n in a.nuis if n not in self.names]
        self.off_bkg, off = [], 2 * S
        for a in self.an:
            self.off_bkg.append(off)
            off += len(a.blocks)
        self.off_nuis = off
        self.sig = np.zeros(len(self.names))
        self.maps = []
        for s, a in enumerate(self.an):
            K = len(a.blocks)
            idx = [self.names.index(n) for n in a.nuis]
            self.maps.append(np.array([2 * s, 2 * s + 1] + list(range(self.off_bkg[s], self.off_bkg[s] + K))
                                      + [self.off_nuis + i for i in idx], dtype=int))
            self.sig[idx] = a.sig
        # the diffuse flux's flavor wall (fit level 3) acts on shared parameters: once, here
        self.wall = None
        if self.an[0].wall is not None:
            a0 = self.an[0]
            K0 = len(a0.blocks)
            self.wall = tuple(self.off_nuis + self.names.index(a0.nuis[i - 2 - K0]) for i in a0.wall)
        self.walls = [a.wall for a in self.an]
        self.src = self.an[0].src          # for code that reads .src (the composition's tracks-matched starts use each)

    def x0(self, tracks_f=None):
        """Every source at its truth (or, with tracks_f, at the normalization whose nu_mu flux is the truth's),
        background normalizations 1, nuisances 0."""
        X = np.zeros(self.off_nuis + len(self.names))
        for s, a in enumerate(self.an):
            X[2 * s] = a.src.lphi_true if tracks_f is None else np.log10(a.src.phi_numu_E0 / max(tracks_f[1], 0.02))
            X[2 * s + 1] = a.src.gamma
            X[self.off_bkg[s]:self.off_bkg[s] + len(a.blocks)] = 1.
        return X

    def asimov(self, f):
        return [a.asimov(f, a.src.lphi_true, a.src.gamma) for a in self.an]

    def chi2(self, X, f, datas):
        c2, g = 0., np.zeros_like(X)
        for a, m, data, w in zip(self.an, self.maps, datas, self.walls):
            a.wall = None                                       # added once below
            try:
                c, gl = a.chi2(X[m], f, data)
            finally:
                a.wall = w
            K, N = len(a.blocks), len(a.sig)
            if N:                                               # the nuisance priors: once, below
                t = X[m][2 + K:2 + K + N] / a.sig
                c -= np.sum(t ** 2)
                gl[2 + K:2 + K + N] -= 2 * t / a.sig
            c2 += c
            np.add.at(g, m, gl)
        if len(self.names):
            t = X[self.off_nuis:] / self.sig
            c2 += np.sum(t ** 2)
            g[self.off_nuis:] += 2 * t / self.sig
        if self.wall is not None:
            ex = X[self.wall[0]] + X[self.wall[1]] - 1. / 3
            if ex > 0:
                c2 += (ex / ASTRO_WALL) ** 2
                g[list(self.wall)] += 2 * ex / ASTRO_WALL ** 2
        return c2, g

    def profile(self, f, datas, X0):
        bounds = []
        for a in self.an:
            bounds += [(a.src.lphi_true - self.lphi_window, a.src.lphi_true + self.lphi_window), (1.0, 5.0)]
        for a in self.an:
            bounds += [(0.2, 3.0)] * len(a.blocks)
        bounds += [self.an[0]._nuis_bounds(n, sg) for n, sg in zip(self.names, self.sig)]
        res = minimize(self.chi2, X0, args=(f, datas), jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.fun, res.x

    def _matched(self, X, f):
        Y = X.copy()
        for s, a in enumerate(self.an):
            Y[2 * s] = np.log10(a.src.phi_numu_E0 / max(f[1], 0.02))
        return Y

    def walk(self, f, datas, f_from, X_from, step=0.02, two=True, n=None):
        """Walk from f_from (best fit X_from) to f: steps no larger than `step` in any fraction (or n steps), each
        from the previous best fit and (two=True) from the same with every source's normalization tracks-matched."""
        f, f_from = np.asarray(f, float), np.asarray(f_from, float)
        n = n or max(10, int(np.ceil(np.abs(f - f_from).max() / step)))
        X = np.array(X_from, float)
        for a in np.linspace(1. / n, 1., n):
            fi = (1. - a) * f_from + a * f
            c, X = self.profile(fi, datas, X)
            if two:
                c2, X2 = self.profile(fi, datas, self._matched(X, fi))
                if c2 < c:
                    c, X = c2, X2
        return c, X

    def best(self, f, datas, f_true, X_true):
        """The lowest chi^2 at f: direct starts (the truth, tracks-matched), a ten-step walk and the fine walk."""
        f = np.asarray(f, float)
        c = min(self.profile(f, datas, X)[0] for X in (self.x0(), self.x0(tracks_f=f)))
        c = min(c, self.walk(f, datas, f_true, X_true, two=False, n=10)[0])
        return min(c, self.walk(f, datas, f_true, X_true)[0])

    def edge_min(self, datas, f_true, X_true, n=51):
        """The lowest chi^2 along the f_tau = 0 edge, as edge_min() for one source."""
        edge = [np.array((fe, 1. - fe, 0.)) for fe in np.linspace(0., 1., n)]
        best = min(min(self.profile(f, datas, X)[0] for X in (self.x0(), self.x0(tracks_f=f))) for f in edge)
        i0 = int(np.argmin([np.linalg.norm(f - np.asarray(f_true, float)) for f in edge]))
        ca, Xa = self.walk(edge[i0], datas, f_true, X_true, two=False, n=10)
        cb, Xb = self.walk(edge[i0], datas, f_true, X_true)
        c, X0 = (ca, Xa) if ca <= cb else (cb, Xb)
        best = min(best, c)
        for d in (1, -1):
            X, i = X0, i0 + d
            while 0 <= i < n:
                c1, X1 = self.profile(edge[i], datas, X)
                c2, X2 = self.profile(edge[i], datas, self._matched(X, edge[i]))
                c, X = (c1, X1) if c1 <= c2 else (c2, X2)
                best = min(best, c)
                i += d
        return best


def triangle_grid(step=0.02):
    pts = []
    n = int(round(1 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            fe, ft = i * step, j * step
            pts.append((fe, 1 - fe - ft, ft))
    return np.array(pts)


def scan(analysis, step=0.02):
    src = analysis.src
    f_true, lphi_true = src.f_true, src.lphi_true
    data = analysis.asimov(f_true, lphi_true, src.gamma)
    x0 = analysis.x0(lphi_true, src.gamma)
    c2min, _ = analysis.profile(f_true, data, x0)
    pts = triangle_grid(step)
    # order points by distance from truth so warm starts propagate outward
    order = np.argsort(np.linalg.norm(pts - f_true, axis=1))
    dchi2 = np.empty(len(pts))
    xbest = {}
    for k in order:
        f = pts[k]
        # warm start from nearest already-solved point
        if xbest:
            keys = np.array(list(xbest.keys()))
            near = keys[np.argmin(np.linalg.norm(pts[keys] - f, axis=1))]
            xs = xbest[near].copy()
        else:
            xs = x0.copy()
        c2, xb = analysis.profile(f, data, xs)
        # second try from a "tracks-matched" start to avoid local minima
        xt = xs.copy()
        xt[0] = np.log10(src.phi_numu_E0 / max(f[1], 0.02))
        c2b, xbb = analysis.profile(f, data, xt)
        if c2b < c2:
            c2, xb = c2b, xbb
        dchi2[k] = c2 - c2min
        xbest[k] = xb
    return pts, np.maximum(dchi2, 0.)


def scan_chi2(analysis, data, x0, f_start, step=0.02):
    """scan() for given data: the profiled chi^2 itself (not Delta chi^2) at every point of the triangle grid, walked
    outward from f_start with the same warm starts (the nearest solved point's best fit, and a tracks-matched start).
    For data no constant composition fits exactly (a source whose composition changes with energy): Delta chi^2 is then
    relative to the grid's minimum, and that minimum says how badly one composition fits."""
    src = analysis.src
    pts = triangle_grid(step)
    order = np.argsort(np.linalg.norm(pts - np.asarray(f_start, float), axis=1))
    chi2, xbest = np.empty(len(pts)), {}
    for k in order:
        f = pts[k]
        if xbest:
            keys = np.array(list(xbest.keys()))
            xs = xbest[keys[np.argmin(np.linalg.norm(pts[keys] - f, axis=1))]].copy()
        else:
            xs = np.array(x0, float)
        c2, xb = analysis.profile(f, data, xs)
        xt = xs.copy()
        xt[0] = np.log10(src.phi_numu_E0 / max(f[1], 0.02))
        c2b, xbb = analysis.profile(f, data, xt)
        if c2b < c2:
            c2, xb = c2b, xbb
        chi2[k], xbest[k] = c2, xb
    return pts, chi2


def scan_sector(pts, f_true, sector, nsec):
    """Indices of the grid points in one of nsec equal angular sectors around the truth (in the f_e, f_tau plane), so
    that a scan can be split among processes, each walking its sector outward from the truth."""
    ang = np.arctan2(pts[:, 2] - f_true[2], pts[:, 0] - f_true[0]) % (2 * np.pi)
    return np.nonzero(np.floor(ang / (2 * np.pi / nsec)).astype(int) % nsec == sector)[0]


def stack_scan(st, step=0.02, sector=None, nsec=1):
    """scan() for a StackAnalysis: the triangle grid walked outward from the truth, each point started from its
    nearest solved neighbour's best fit and from the same with every source's normalization tracks-matched. With
    nsec > 1, only sector `sector` (scan_sector), walked outward from the truth on its own; the rest of the grid is
    returned as nan, to be filled by the other sectors."""
    f_true = np.asarray(st.an[0].src.f_true, float)
    datas = st.asimov(f_true)
    c2min, Xb = st.profile(f_true, datas, st.x0())
    pts = triangle_grid(step)
    order = np.argsort(np.linalg.norm(pts - f_true, axis=1))
    if nsec > 1:
        keep = set(scan_sector(pts, f_true, sector, nsec).tolist())
        order = np.array([k for k in order if k in keep], dtype=int)
    d2, xbest = np.full(len(pts), np.nan), {}
    for k in order:
        f = pts[k]
        if xbest:
            keys = np.array(list(xbest.keys()))
            Xs = xbest[keys[np.argmin(np.linalg.norm(pts[keys] - f, axis=1))]].copy()
        else:
            Xs = Xb.copy()
        c, X = st.profile(f, datas, Xs)
        c2, X2 = st.profile(f, datas, st._matched(Xs, f))
        if c2 < c:
            c, X = c2, X2
        d2[k], xbest[k] = c - c2min, X
    return pts, np.where(np.isnan(d2), np.nan, np.maximum(d2, 0.))


def source_line(x):
    """The flavor composition at Earth of a source that makes (x, 1 - x, 0): nu_e fraction x at production, no nu_tau
    (x = 0 muon damping, 1/3 pion decay, 1 neutron decay)."""
    return earth_ratios([x, 1. - x, 0.])


def line_scan(profile, matched, x_true, X_true, c_true, step=0.02, side=0):
    """Delta chi^2 along the line of source compositions (x, 1 - x, 0), x in [0, 1] on a grid of `step` plus the true
    x itself: walked outward from the truth (best fit X_true, chi^2 c_true), each point started from the previous
    one's best fit and from the same with the normalization tracks-matched (matched(X, f)), keeping the lower, as the
    scans do. profile(f, X) -> (chi2, X). side=-1 or +1 walks one side only (for parallel runs), 0 both.
    Returns {x: Delta chi^2}."""
    grid = np.round(np.arange(0., 1. + 1e-9, step), 10)
    out = {float(x_true): 0.}
    for direction in ((-1, 1) if side == 0 else (side,)):
        X = np.array(X_true, float)
        for x in sorted((x for x in grid if (x - x_true) * direction > 1e-9), key=lambda x: abs(x - x_true)):
            f = source_line(x)
            c1, X1 = profile(f, X)
            c2, X2 = profile(f, matched(X, f))
            c, X = (c1, X1) if c1 <= c2 else (c2, X2)
            out[float(x)] = max(0., c - c_true)
    return out


def line_interval(curve, level):
    """The interval of x with Delta chi^2 below `level` (1: 68%, 3.84: 95%, one parameter), interpolated linearly
    between grid points; an end at 0 or 1 means the interval reaches the edge of the line."""
    xs = np.array(sorted(curve))
    d = np.array([curve[x] for x in xs])
    inside = d < level
    if not inside.any():
        return None
    i0, i1 = np.nonzero(inside)[0][[0, -1]]
    lo = xs[i0] if i0 == 0 else xs[i0 - 1] + (level - d[i0 - 1]) * (xs[i0] - xs[i0 - 1]) / (d[i0] - d[i0 - 1])
    hi = xs[i1] if i1 == len(xs) - 1 else xs[i1] + (level - d[i1]) * (xs[i1 + 1] - xs[i1]) / (d[i1 + 1] - d[i1])
    return float(lo), float(hi)

