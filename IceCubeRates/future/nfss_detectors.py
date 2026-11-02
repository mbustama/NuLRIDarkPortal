# Vendored from NuFlavorSingleSource, commit 39088bd ("Final production run: the 2026 inputs"), code/detectors.py,
# unchanged except: the import of the forecast module (nfss_forecast)
# (docs/dev/FUTURE_DETECTORS_PLAN.md; Mauricio, 2026-10-14/15: the eight detectors as in that project)
"""
Detector definitions and calibration to published anchors.

Every detector gets three shower scenarios (IceCube only one), to choose between later:
  base : the published selections (IceCube, ARCA, GVD; IceCube-Gen2's TDR). ARCA's, per unit volume, is the design
         template for P-ONE, TRIDENT and NEON; HUNT's 30 TeV threshold is assumed.
  mid  : ARCA's published up-going selection (half efficient at ~7 TeV) as a dedicated up-going selection, everywhere.
  ded  : an optimistic dedicated up-going selection at IceCube's cascade E50 = 1.7 TeV (HUNT 5 TeV) [ASSUMPTION: no
         published water selection keeps showers below ~5 TeV].
In mid and ded, down-going showers are kept only above 60 TeV; atmospheric muons scale their background.
Water shower volumes are 1.2 x the instrumented volume, the effective volume of ARCA's published selection. Its
published atm-nu shower background is 1.6 x the model's in every energy bin; the factor scales the atm-nu showers of
every water selection (but GVD's baseline, calibrated on GVD's own data).
Tracks in water: up-going + horizontal (cosZ < 0.17), as in the ARCA selection, the same in all scenarios.
IceCube-Gen2 is two Detector parts used together: 'Gen2_IC' (its IceCube strings: IceCube as calibrated, with Gen2's
track PSF) and 'Gen2_new_*' (the new strings). Its scenarios differ only in the new strings' shower selection:
  base : the TDR's HESE-like cascade selection;  mid : E50 ~ 28 TeV;  ded : E50 ~ 7 TeV [ASSUMPTIONS]
Down-going tracks (the second analysis): '<water detector>_dn' parts hold only the down-going track channel, added to
a detector's other channels; ARCA, P-ONE, TRIDENT, NEON and HUNT (not GVD, ~1 km deep and without a published
multi-cluster track selection). Same track response as up-going, above 100 TeV of reconstructed muon energy; background
from atmospheric muons at each detector's depth, times a bundle factor calibrated on ARCA's simulation [ASSUMPTION:
the same factor at every depth and above 100 TeV; bracketed in code/downgoing_study.py].

Anchors used                                                         source
-------------------------------------------------------------------  ------------------------------------------
IceCube tracks : 79 signal events from NGC 1068 in 3186 d for         IceCube, Science 378, 538 (2022)
                 Phi_numu(1 TeV)=5e-11 /TeV/cm2/s, gamma=3.2; and
                 13 events from the TXS 0506+056 2014-15 flare for     IceCube, Science 361, 147 (2018)
                 E^2 J(100 TeV) = 2.1e-4 TeV/cm2, gamma = 2.1
IceCube showers: ~5,900 cascade events/yr (0.5 TeV - 5 PeV), all     IceCube, ApJ (2024), 10-yr cascade transient
                 sky; 90% taken as nu-induced                         search (59,592 events, May 2011 - May 2021)
IceCube shower PSF: 50% radius ~17 deg @1 TeV -> ~6 deg @1 PeV          Galactic-plane DNN cascades (Science 2023),
                                                                     as used in Blanco et al., arXiv:2509.15421
ARCA tracks    : 8.5e4 atm-nu and 220 cosmic tracks/yr (theta>80),    KM3NeT, EPJC 84, 885 (2024), Table 3
                 the cosmic ones for KM3NeT's E^-2 benchmark flux      and eq. (6)
                 (eq. 6). Check: nu_mu CC A_eff vs Fig. 7b, within
                 10% at 0.1-1 PeV and 25% at 1-10 TeV
ARCA showers   : the published selection: nu_e CC A_eff, up- and      same, Figs. 6a, 7c, 8a, 8b
                 down-going (Fig. 7c); atm muons and atm nu per
                 energy bin (Fig. 6a; 290 atm-nu/yr); PSF (Fig. 8a);
                 energy resolution 6-10%. Checks: cosmic showers for
                 eq. 6, 52/yr (+~4 Glashow, not modeled) vs 60;
                 nu_e CC alone 22.6 vs 22.4 (Fig. 6a)
ARCA PSF       : 0.1 deg tracks at 300 TeV                             same
GVD tracks     : 988 upgoing nu candidates in 30 live cluster-yr       Baikal-GVD, arXiv:2603.21261;
                 (clusters 1-11, Apr 2019 - Mar 2024), single-cluster  PoS(ICRC2025)1162
                 reconstruction; 23.5 atm-nu expected above 45 TeV
                 reco energy; ~0.5 deg typical; 20 clusters ~1 km^3 planned for 2028.
                 No multi-cluster track performance published yet
GVD showers    : upgoing E>15 TeV: 2.7 atm-nu events in 2018-2022;     Baikal-GVD, arXiv:2211.09447
                 all-sky E>70 TeV: 7.4 atm-mu vs 0.8 atm-nu bkg;
                 median PSF 2-4 deg; energy resolution 10-30%
P-ONE          : 7 clusters x 10 lines, 80 m spacing, ~1 km^3;        pacific-neutrino.org; PoS(ICRC2023)1175
                 Cascadia Basin (~47.7 N); tracks <~0.1 deg
TRIDENT        : 1211 strings, 70/110 m spacing, ~7.5 km^3, 17.4 N;    Ye et al., Nat. Astron. 7, 1497 (2023)
                 tracks 0.1 deg and A_eff ~ 5e2 m^2 at ~100 TeV;
                 contained showers 3.45 deg @10 TeV, 2.0 deg @100 TeV   PoS(ICRC2023)1207 (preliminary)
                 (likelihood), as ARCA's
NEON           : 1200 strings, ~100 m spacing, ~10 km^3, South China   arXiv:2408.05122; showers: arXiv:2609.03417
                 Sea; tracks 0.1 deg @100 TeV; showers: GNN PSF per energy bin, 4.4 deg (1-3.2 TeV) to
                 1.5 deg (0.3-1 PeV); 25-37% energy resolution
HUNT           : 2304 strings, ~130 m spacing, 20-inch PMTs, 30 km^3;   PoS(ICRC2023)1080
                 sites Baikal (51.8 N) or South China Sea (17.4 N; used here);
                 up-going track efficiency >85% above 30 TeV, ~70% at 10 TeV; <0.05 deg for long tracks
IceCube-Gen2   : IceCube + 120 new strings, 240 m spacing, 7.9 km^3,   IceCube-Gen2 TDR v0.05 (Jul 2023);
                 multi-PMT modules (~3x IceCube's photocathode);       white paper, J. Phys. G 48, 060501 (2021);
                 new-string muon A_eff at the horizon turns on at       toise (github.com/icecube/toise)
                 E_mu ~ 2 TeV (Gen2 - IceCube, TDR Fig. 34);
                 TXS 0506+056 2014-15 flare: 38 events (13 in IceCube) (TDR p.29);
                 track PSF 0.41 deg @1 TeV -> 0.17 deg @100 TeV (Fig. 43);
                 cascades: HESE-like logistic, E_th = 66.7 TeV, slope 2.5 (toise), nu_e A_eff 133 m^2 @1 PeV (Fig. 42),
                 PSF 3.85 + 2190/sqrt(E/GeV) deg (Fig. 44) above 30 TeV, 45 deg below (an idealized study gives
                 >20 deg at 31.6 TeV: PoS(ICRC2023)1188); double cascades and their misID from toise
                 Checks: NGC 1068 track significance 2.04x IceCube's (TDR 2.34, without the nu-mu angle);
                 track A_eff within 6% of TDR Fig. 42 at 30 TeV - 1 PeV
Down-going     : atm-muon bundles above 10 TeV at trigger, 8.1e7/yr     KM3NeT, EPJC 84, 885, Sec. 4.1
tracks           in ARCA (simulated). Depths of the detector centres:
                 ARCA 3.1 km (sea floor 3.5 km), P-ONE 2.2 km (2.66),  KM3NeT; P-ONE; TRIDENT, NEON and HUNT
                 TRIDENT 3.1 km (3.5), NEON 3.1 km (3.5, the deeper     site studies (approximate)
                 candidate), HUNT 3.0 km (its South China Sea array:   PoS(ICRC2023)1080
                 2.56-3.42 km)
"""
import numpy as np
from scipy.optimize import brentq
from .nfss_forecast import (Detector, sigmoid, NGC1068, PHI_NUMU_1TEV, GAMMA_TRUE, E, DE, YR, xsec, muon_range, ymean, NA,
                      earth_survival, atm_muon, astro_hese, FINAL)

KM3 = 1e15  # cm^3
KM3_EQ6 = lambda E: 1.2e-8 * E ** -2.   # KM3NeT's benchmark cosmic flux, per flavor nu+nubar (EPJC 84, 885, eq. 6)
YEARS = 10.0
# Down-going tracks: depth of each water detector's centre [m], sea-water density [g/cm^3], and the reference cut on
# the reconstructed muon energy [log10 GeV]
DEPTH = {'ARCA': 3100., 'P-ONE': 2200., 'TRIDENT': 3100., 'NEON': 3100., 'HUNT': 2990.}
RHO_SEA, TRKD_LGCUT = 1.03, 5.0


def psf_ic_trk(E):  return np.maximum(0.25, 1.2 * (E / 1e3) ** -0.35)
def psf_ic_shw(E):  return np.maximum(5.0, 17.0 * (E / 1e3) ** -0.15)
def psf_arca_trk(E): return np.sqrt((0.7 * (E / 1e3) ** -0.5) ** 2 + 0.07 ** 2)
def psf_gvd_trk(E): return np.sqrt((0.7 * (E / 1e3) ** -0.5) ** 2 + 0.4 ** 2)
def psf_gvd_shw(E): return np.maximum(2.0, 2.5 * (E / 1e5) ** -0.25)
def psf_hunt_trk(E): return np.sqrt((0.7 * (E / 1e3) ** -0.5) ** 2 + 0.05 ** 2)
# Shower PSFs from published reconstructions. ARCA: fit to EPJC 84, 885, Fig. 8a (5 TeV - 1 PeV; extrapolated below);
# it also stands for TRIDENT, whose published contained-shower values at 10-100 TeV it matches, and for P-ONE (nothing
# published). NEON: its GNN reconstruction bin by bin (arXiv:2609.03417, Fig. 4), flat outside 1.8-560 TeV.
def psf_arca_shw(E): return np.maximum(1.8, 2.0 * (E / 1e5) ** -0.22)
_NEON_E, _NEON_PSF = np.array([1.8e3, 5.6e3, 1.8e4, 5.6e4, 1.8e5, 5.6e5]), np.array([4.4, 3.5, 2.9, 1.85, 1.8, 1.5])
def psf_neon_shw(E): return np.interp(np.log10(E), np.log10(_NEON_E), _NEON_PSF)
# Gen2: nu-mu kinematic angle (as for water) + Gen2 muon reconstruction 0.165 + 0.245 (E_mu/TeV)^-1 deg (fit to TDR
# Fig. 43, 1 TeV - 1 PeV), with E_mu ~ E/2; cascades: fit to the Gen2 projection in TDR Fig. 44 above 30 TeV; below it,
# no usable direction (an idealized study gives >20 deg at 31.6 TeV, PoS(ICRC2023)1188), taken as 45 deg
def psf_g2_trk(E):  return np.sqrt((0.7 * (E / 1e3) ** -0.5) ** 2 + (0.165 + 0.49 * (E / 1e3) ** -1) ** 2)
def psf_g2_shw(E):  return np.where(np.asarray(E) < 3e4, 45.0, 3.85 + 2190. / np.sqrt(E))

# ARCA's published shower selection (EPJC 84, 885, Fig. 7c): the nu_e CC effective areas of its up- and down-going
# showers, digitized and divided by the model's own full-efficiency area, i.e. with the model's cross sections and, for
# up-going showers, its Earth absorption averaged over the hemisphere. Water shower volumes are F_V x the instrumented
# volume; F_V = 1.2 makes the up-going efficiency 1 at 30-300 TeV. Up-going showers turn on between 5 and 30 TeV
# (half efficient at ~7 TeV), down-going ones slowly; above ~300 TeV both keep rising (bright showers from outside the
# instrumented volume pass), to ~2 at 10 PeV. Flat beyond 10^7.2 GeV; the Glashow resonance is left out.
F_V = 1.2
_UP_LGE = np.array([3.7, 3.75, 3.85, 3.95, 4.05, 4.15, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4,
                    5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.2])
_UP_EFF = np.array([0., 0.17, 0.47, 0.61, 0.72, 0.81, 0.88, 0.93, 0.98, 0.99, 0.99, 1.02, 1.02, 1.02, 1.02, 1.03,
                    1.07, 1.07, 1.11, 1.11, 1.11, 1.12, 1.16, 1.19, 1.23, 1.28, 1.49, 1.59, 1.69, 1.83, 1.87, 1.91,
                    1.95, 1.99, 1.99])
_DN_LGE = np.array([3.75, 3.775, 3.825, 3.875, 3.925, 4.025, 4.125, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1,
                    5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
                    7.1, 7.2])
_DN_EFF = np.array([0., 0.07, 0.12, 0.16, 0.20, 0.25, 0.28, 0.35, 0.37, 0.39, 0.43, 0.51, 0.59, 0.61, 0.61, 0.65,
                    0.67, 0.69, 0.73, 0.77, 0.77, 0.78, 0.79, 0.79, 0.80, 0.83, 0.83, 0.89, 0.93, 1.00, 1.10, 1.27,
                    1.33, 1.39, 1.45, 1.51, 1.56, 1.57])
def eff_arca_up(Ed): return np.interp(np.log10(np.maximum(Ed, 1.)), _UP_LGE, _UP_EFF, left=0.)
def eff_arca_down(Ed): return np.interp(np.log10(np.maximum(Ed, 1.)), _DN_LGE, _DN_EFF, left=0.)
def gate_arca_down(Ed): return np.minimum(1., eff_arca_down(Ed) / np.maximum(eff_arca_up(Ed), 1e-12))
# The rise above 300 TeV, for selections that turn on differently (water ded scenario, HUNT's baseline)
def hi_arca(Ed): return np.maximum(1., eff_arca_up(Ed))


def dc_ice(E):   return 0.5 * sigmoid(E, 3e5, 3)     # placeholder until calibrate() anchors the double cascades
def dc_water(E): return 0.5 * sigmoid(E, 1.5e5, 3)   # placeholder (the optimistic pre-2026-10-12 water assumption)
def dc_g2(E):    return 0.2 * sigmoid(E, 2e5, 4)     # Gen2 new strings: plateau and midpoint of the TDR's curve

# Double cascades (nu_tau identification), anchored to published numbers (2026-10-12). eps_DC(E) is the fraction of
# nu_tau CC (non-muonic) events in a detector's shower sample that are identified as double cascades, so its anchor is a
# classification fraction measured in a comparable sample, or, where none is published, a projected rate.
#  - IceCube, HUNT and IceCube-Gen2's IceCube strings: IceCube's measured fraction in its MESE sample (11.4 yr; starting
#    events from 1 TeV, double cascades above 30 TeV; data release doi:10.7910/DVN/CBNMEB, in data/mese/), whose volume at
#    high energy matches our IceCube cascade sample's; HESE's (EPJC 82, 1031) has a smaller one. DC_MESE is the fit of
#    a S(E; E50, n) to its nu_tau CC double cascades / (double cascades + cascades), 0.1-10 PeV
#    (code/checks/mese_dc.py, which also checks our area against MESE's). Not normalized to a rate.
#  - Others: MESE's shape, except TRIDENT and NEON, which take TRIDENT's double-pulse passing fraction
#    (PoS(ICRC2025)1195, Fig. 4, fit), normalized to reproduce a projected rate of identified nu_tau from the HESE 7.5-yr
#    flux (all sky, above 63 TeV deposited, dedicated showers), DC_RATE [1/yr]: P-ONE 0.5 (A. Gaertner, PhD thesis,
#    Alberta 2024), TRIDENT 4.57 (PoS(ICRC2025)1195; flux normalization not stated), Baikal-GVD 0.33 for 20 clusters
#    (PoS(ECRS)099, preliminary, 1.67e-2 per cluster). ARCA takes P-ONE's normalization and NEON TRIDENT's.
# Misidentification: 2% of showers (IceCube 1.9%, EPJC 82, 1031; MESE 1.3-3.4% of nu_e CC; GVD 1.3-2.2%) and 1% of
# nu_mu CC events (MESE 0.8-1.9%); IceCube-Gen2's new strings keep toise's curve and 5% (the TDR's nu_e), and take 2% of
# nu_mu CC.
DC_MESE = (0.56, 4.2e5, 1.55)
# nu_mu CC events counted as showers, f_mu->shw: IceCube's measured classification in MESE (code/checks/mese_dc.py,
# part 6: the fraction of nu_mu CC events classified as cascades, against neutrino energy, from the data release;
# fit c + (a - c) / (1 + (E/E50)^n) at 1 TeV - 10 PeV, rms 0.005), for every detector in the final production run
# (Mauricio, 2026-10-14: "Use IceCube's measured fraction in the final run"). NUMU_SHW = 'assumed' keeps the earlier
# 0.15 (ice) and 0.10 (water), which the stored results use; 'mese' switches every detector to the measured one.
NUMU_SHW_MESE = (0.0892, 1.59e4, 0.0146, 0.96)
NUMU_SHW = 'mese' if FINAL else 'assumed'          # the final inputs take the measured one
def numu_shw_mese(E):
    a, E50, c, n = NUMU_SHW_MESE
    return c + (a - c) / (1. + (np.asarray(E, float) / E50) ** n)
def _numu_shw(assumed):
    """A detector's f_mu->shw: the assumed constant, or IceCube's measured fraction (NUMU_SHW = 'mese')."""
    assert NUMU_SHW in ('assumed', 'mese'), NUMU_SHW
    return numu_shw_mese if NUMU_SHW == 'mese' else assumed
def dc_shape_mese(E): return DC_MESE[0] * sigmoid(E, DC_MESE[1], DC_MESE[2])
def dc_shape_dp(E):   return 0.5 * sigmoid(E, 6.5e5, 1.2)
DC_RATE = {'P-ONE': 0.5, 'TRIDENT': 4.57, 'GVD': 0.33}
DC_FAMILY = {'IceCube': ('IceCube', dc_shape_mese), 'Gen2_IC': ('IceCube', dc_shape_mese),
             'HUNT': ('IceCube', dc_shape_mese), 'ARCA': ('P-ONE', dc_shape_mese), 'P-ONE': ('P-ONE', dc_shape_mese),
             'GVD': ('GVD', dc_shape_mese), 'TRIDENT': ('TRIDENT', dc_shape_dp), 'NEON': ('TRIDENT', dc_shape_dp)}


def nutau_dc_rate(P, lgmin=4.8):
    """Identified nu_tau CC double cascades per year from the HESE 7.5-yr flux (per flavor, 1:1:1), all sky, above
    10^lgmin GeV of deposited energy (default 63 TeV: the analysis bins)."""
    d = Detector('x', P, build=False)
    cz = np.linspace(-0.995, 0.995, 200)
    out = 0.
    for bar in (0, 1):
        A, lg = d.aeff('dc', 'tau', bar)[0]
        sel = (lg >= lgmin).astype(float)
        for c in cz:
            if P['shw_accept'](c) == 0:
                continue
            out += 2 * np.pi * (cz[1] - cz[0]) * np.sum(0.5 * astro_hese(E) * A * earth_survival(bar, E, c) * DE * sel
                                                        * d.gate('dc', c, lg))
    return out * YR


def base_params(**kw):
    P = dict(years=YEARS, trk_Eth=600., trk_sig_lgE=0.35, shw_sig_lgE=0.10, dc_misid=0.005,
             trk_accept=lambda c: 1.0, shw_accept=lambda c: 1.0, shw_bkgmult=lambda c: 1.0, shw_down_Emin=None)
    P.update(kw)
    return P


def water(lat, V, trk_norm, trk_Eth, trk_psf, shw_psf, shw_eff, down_Emin=None, down_mult=1.0, up_mult=1.0,
          rho=1.04, dc=dc_water, sig_lgE=0.10, down_gate=None, mult_E=None):
    P = base_params(lat=lat, trk_norm=trk_norm, trk_Eth=trk_Eth, trk_psf=trk_psf,
                    trk_accept=lambda c: 1.0 if c < 0.17 else 0.0,
                    shw_V=F_V * V * KM3, shw_rho=rho, shw_psf=shw_psf, shw_eff=shw_eff, shw_sig_lgE=sig_lgE,
                    numu_misid=_numu_shw(0.10), dc_eff=dc, shw_accept=lambda c: 1.0, shw_down_Emin=down_Emin,
                    shw_bkgmult=lambda c, u=up_mult, d=down_mult: u if c < 0 else d)
    if down_gate is not None:        # down-going shower efficiency relative to shw_eff, instead of a cut at down_Emin
        P['shw_down_gate'] = down_gate
    if mult_E is not None:           # energy-dependent atm-muon multiplier, instead of down_mult
        P['shw_bkgmult_E'] = mult_E
    return P


def numu_aeff_100TeV(trk_norm, Eth):
    """nu_mu (nu/nubar averaged) track effective area at 100 TeV [m^2] in the model."""
    i = np.argmin(abs(E - 1e5))
    out = 0.
    for bar in (0, 1):
        Emu = (1 - ymean('cc', bar, E)) * E
        A = trk_norm * xsec('cc', bar, E) * (muon_range(Emu, Eth) + 1e5) * sigmoid(Emu, Eth)
        out += 0.5 * A[i]
    return out / 1e4


def trk_aeff_band(P, E0, lo, hi):
    """nu_mu CC track A_eff [m^2] at E0: nu/nubar averaged, with Earth absorption, averaged over cos(zenith) in [lo, hi],
    as in KM3NeT's EPJC 84, 885, Fig. 7b."""
    d = Detector('x', P, build=False)
    i = np.argmin(abs(E - E0))
    cz = np.linspace(lo, hi, 60)
    out = 0.
    for bar in (0, 1):
        A = sum(a for a, _ in d.aeff('trk', 'mu', bar))
        out += 0.5 * A[i] * np.mean([earth_survival(bar, E, c)[i] for c in cz])
    return out / 1e4


def nue_shw_aeff(P, E0):
    """nu_e shower effective area [m^2] at E0: CC + NC, nu/nubar averaged, 4pi average with Earth absorption."""
    d = Detector('x', P, build=False)
    i = np.argmin(abs(E - E0))
    cz = np.linspace(-0.995, 0.995, 200)
    out = 0.
    for bar in (0, 1):
        surv = np.mean([earth_survival(bar, E, c)[i] for c in cz])
        out += 0.5 * surv * sum(A[i] for A, _ in d.aeff('shw', 'e', bar))
    return out / 1e4


def muon_rate(depth, Emin, area):
    """Single atmospheric muons per year above Emin [GeV] at a depth [m] of sea water, from every down-going direction,
    through an area [cm^2]."""
    cz = np.linspace(0.005, 0.995, 100)
    sel = (E >= Emin) * DE
    return YR * area * 2 * np.pi * (cz[1] - cz[0]) * sum(np.sum(atm_muon(E, c, 100. * depth * RHO_SEA / c) * sel)
                                                         for c in cz)


def calibrate(verbose=True):
    out = {}
    # ---------------- IceCube ----------------
    P_ic = base_params(lat=-89.99, trk_norm=1.0, trk_psf=psf_ic_trk,
                       shw_V=1.0 * KM3, shw_rho=0.92, shw_psf=psf_ic_shw, shw_sig_lgE=0.12,
                       numu_misid=_numu_shw(0.15), dc_eff=dc_ice,
                       shw_eff=lambda Ed: sigmoid(Ed, 1e3), shw_bkgmult=lambda c: 1.10)
    # Two anchors fix the track threshold shape and normalization:
    #  (i) NGC 1068: 79 events in 3186 d for the best-fit soft flux (probes ~1-10 TeV);
    #  (ii) TXS 0506+056 2014-15 flare: 13 events for fluence E^2 J(100 TeV) = 2.1e-4 TeV/cm^2, gamma = 2.1
    #       (IceCube, Science 361, 147 (2018); probes ~100 TeV).
    w_ngc = NGC1068.weights(E, np.log10(PHI_NUMU_1TEV), GAMMA_TRUE)
    J_txs = 0.21 / 1e10 * (E / 1e5) ** -2.1          # nu_mu+nubar_mu fluence [GeV^-1 cm^-2]

    def ic_ratio(lgEth):
        d1 = Detector('x', dict(P_ic, trk_Eth=10 ** lgEth, years=3186 / 365.25), build=False)
        d2 = Detector('x', dict(P_ic, trk_Eth=10 ** lgEth, years=1 / 3.15576e7), build=False)
        return d2.source_counts('trk', J_txs, (0, 1, 0)) / d1.source_counts('trk', w_ngc, (0, 1, 0)) - 13. / 79.
    P_ic['trk_Eth'] = 10 ** brentq(ic_ratio, 2.0, 3.2, xtol=1e-3)
    d = Detector('IceCube', dict(P_ic, years=3186 / 365.25), build=False)
    P_ic['trk_norm'] = 79.0 / d.source_counts('trk', w_ngc, (0, 1, 0))

    def ic_rate(E50):
        P = dict(P_ic, shw_eff=lambda Ed: sigmoid(Ed, E50))
        return Detector('x', P, build=False).allsky_rate('shw', ecut=500.) - 0.9 * 59592 / 10.04
    E50_ic = brentq(ic_rate, 200., 2e5, xtol=1.)
    P_ic['shw_eff'] = lambda Ed, E50=E50_ic: sigmoid(Ed, E50)
    out['IceCube'] = P_ic
    # water ded scenario: IceCube's cascade turn-on, with ARCA's rise above 300 TeV
    eff_ded = lambda Ed, E50=E50_ic: sigmoid(Ed, E50) * hi_arca(Ed)

    # ---------------- ARCA (full, 230 DUs) ----------------
    P_ar = base_params(lat=36.27, trk_norm=1.0, trk_psf=psf_arca_trk,
                       trk_accept=lambda c: 1.0 if c < 0.17 else 0.0,
                       shw_V=F_V * KM3, shw_rho=1.04, shw_psf=psf_arca_shw, shw_sig_lgE=0.04,
                       numu_misid=_numu_shw(0.10), dc_eff=dc_water, shw_eff=eff_arca_up, shw_down_gate=gate_arca_down)

    # Tracks: 8.5e4 atm-nu and 220 cosmic events per year; the paper computes the cosmic ones for its eq. (6),
    # Phi_(nu_i+nubar_i) = 1.2e-8 (E/GeV)^-2 GeV^-1 cm^-2 s^-1 sr^-1 per flavor. Their ratio fixes the threshold.
    def ratio(lgEth):
        dd = Detector('x', dict(P_ar, trk_Eth=10 ** lgEth), build=False)
        a = dd.allsky_rate('trk', accept=P_ar['trk_accept'])
        c = dd.allsky_rate('trk', accept=P_ar['trk_accept'], flux=KM3_EQ6)
        return c / a - 220. / 8.5e4
    P_ar['trk_Eth'] = 10 ** brentq(ratio, 1.7, 3.5, xtol=1e-3)
    r = Detector('x', P_ar, build=False).allsky_rate('trk', accept=P_ar['trk_accept'])
    P_ar['trk_norm'] = 8.5e4 / r
    TN, TE = P_ar['trk_norm'], P_ar['trk_Eth']        # ARCA track response per km^2-ish, reused below

    # Showers: the published selection (EPJC 84, 885, Figs. 6a, 7c, 8): the efficiencies, PSF and 6-10% energy
    # resolution (0.04 in log10 E) above. With the published nu_e CC effective areas, the model reproduces the cosmic
    # nu_e CC showers of Fig. 6a, but gives fewer atm-nu showers than the 290 per year published, by the same factor
    # (1.6-1.7) in every energy bin from 5 TeV to 1 PeV: the flux models and the NC and nu_mu CC content differ. The
    # factor k_nu = 290 / model scales the atm-nu shower background of every water selection built on this one.
    d_ar = Detector('x', P_ar, build=False)
    arca_shw_atm = d_ar.allsky_rate('shw')
    k_nu = 290. / arca_shw_atm
    # Atmospheric muons in that selection (Fig. 6a; all down-going; per year in reconstructed-energy bins): 9 (4.6-10 TeV),
    # 67 (10-46), 18 (46-100), 7.5 (100-215), 2.8 (215-464). Each, over the model's down-going atm-nu showers in the bin,
    # sets the down-going background multiplier there (interpolated in log E, flat outside). It also serves the other
    # water detectors (an assumption: their depths and selections differ).
    def nu_down(lo, hi):
        down = lambda c: 1.0 if c >= 0 else 0.0
        return (d_ar.allsky_rate('shw', accept=down, ecut=lo, smear=True)
                - d_ar.allsky_rate('shw', accept=down, ecut=hi, smear=True))
    mu_bins = ((4.6e3, 1e4, 9.), (1e4, 4.6e4, 67.), (4.6e4, 1e5, 18.), (1e5, 2.15e5, 7.5), (2.15e5, 4.64e5, 2.8))
    mu_E = np.array([np.sqrt(lo * hi) for lo, hi, _ in mu_bins])
    mu_r = np.array([n / nu_down(lo, hi) for lo, hi, n in mu_bins])
    mult_mu = lambda c, Ed, e=mu_E, r=mu_r, k=k_nu: k if c < 0 else k + np.interp(np.log10(Ed), np.log10(e), r)
    E50_mid = 10 ** np.interp(0.5, _UP_EFF[:6], _UP_LGE[:6])    # where ARCA's up-going showers are half efficient
    out['ARCA_base'] = dict(P_ar, shw_bkgmult_E=mult_mu)
    # checks: cosmic showers for eq. 6 (60 published, Glashow resonance included), track A_eff vs Fig. 7b
    arca_cos = d_ar.allsky_rate('shw', flux=KM3_EQ6)
    a7b = {e: trk_aeff_band(P_ar, e, -0.995, 0.08) for e in (1e4, 1e5, 1e6)}
    out['ARCA_mid'] = water(36.27, 1.0, TN, TE, psf_arca_trk, psf_arca_shw, eff_arca_up, 6e4, mult_E=mult_mu,
                            sig_lgE=0.04)
    out['ARCA_ded'] = water(36.27, 1.0, TN, TE, psf_arca_trk, psf_arca_shw, eff_ded, 6e4, mult_E=mult_mu,
                            sig_lgE=0.04)

    # ---------------- Baikal-GVD (20 clusters, ~1 km^3) ----------------
    P_gv = base_params(lat=51.77, trk_norm=1.0, trk_psf=psf_gvd_trk,
                       trk_accept=lambda c: 1.0 if c < 0 else 0.0,
                       shw_V=1.0 * KM3, shw_rho=1.0, shw_psf=psf_gvd_shw,
                       numu_misid=_numu_shw(0.10), dc_eff=dc_water,
                       shw_accept=lambda c: 1.0 if c < 0 else 0.0)
    # Tracks: the published single-cluster up-going sample, with two anchors as for IceCube: 988 atm-nu events in
    # 30 live cluster-years, 23.5 of them expected above 45 TeV reconstructed energy (PoS(ICRC2025)1162). The ratio
    # fixes the threshold, the total the normalization. Multi-cluster tracks: no published performance yet.
    def gvd_ratio(lgEth):
        dd = Detector('x', dict(P_gv, trk_Eth=10 ** lgEth), build=False)
        return (dd.allsky_rate('trk', accept=P_gv['trk_accept'], ecut=4.5e4, smear=True)
                / dd.allsky_rate('trk', accept=P_gv['trk_accept']) - 23.5 / 988)
    P_gv['trk_Eth'] = 10 ** brentq(gvd_ratio, 2.5, 4.8, xtol=1e-3)
    r = Detector('x', P_gv, build=False).allsky_rate('trk', accept=P_gv['trk_accept'])
    P_gv['trk_norm'] = 988 / 30. * 20 / r          # 30 live cluster-years -> 20 clusters
    step = lambda Ed: sigmoid(Ed, 1.5e4, 8)
    rs = Detector('x', dict(P_gv, shw_eff=step), build=False).allsky_rate('shw', accept=P_gv['shw_accept'], ecut=1.5e4)
    eps_gvd = min(1.0, 2.7 / (23 * 0.05) / rs)
    gvd_down = dict(shw_accept=lambda c: 1.0, shw_down_Emin=7e4)     # all-sky selection above 70 TeV
    mult_gvd = lambda c: 1.19 if c < 0 else 1.0 + 7.4 / 0.8
    # same tracks in all scenarios; only the shower selection differs
    out['GVD_base'] = dict(P_gv, shw_eff=lambda Ed, e=eps_gvd: e * sigmoid(Ed, 1.5e4, 8), shw_bkgmult=mult_gvd, **gvd_down)
    # mid and ded: ARCA's selection per unit volume and its atm-nu scale, as for the other water detectors
    mult_gvd_k = lambda c, k=k_nu: k * mult_gvd(c)
    out['GVD_mid'] = dict(P_gv, shw_V=F_V * KM3, shw_eff=eff_arca_up, shw_bkgmult=mult_gvd_k, **gvd_down)
    out['GVD_ded'] = dict(P_gv, shw_V=F_V * KM3, shw_eff=eff_ded, shw_bkgmult=mult_gvd_k, **gvd_down)

    # ---------------- Next generation ----------------
    # Tracks: ARCA response scaled by (V/1 km^3)^(2/3), except TRIDENT, normalized to its quoted A_eff(100 TeV).
    # Showers: P-ONE, TRIDENT and NEON take ARCA's published selection per unit volume as their baseline (a design
    # template: nothing comparable is published for them); HUNT's 30 TeV baseline remains an assumption. All take
    # ARCA's down-going atm-muon multipliers [ASSUMPTION: their depths and selections differ].
    A_arca100 = numu_aeff_100TeV(TN, TE)
    tn_trident = 500. / numu_aeff_100TeV(1.0, 150.)      # its quoted 500 m^2 at 100 TeV, with its own threshold
    nextgen = {
        #          lat    V    trk_norm             Eth    trk_psf       shw_psf       sig_lgE  E50 ded  dc
        'P-ONE':   (47.7, 1.0, TN,                  300., psf_arca_trk, psf_arca_shw, 0.04,    E50_ic,  dc_water),
        'TRIDENT': (17.4, 7.5, tn_trident,          150., psf_arca_trk, psf_arca_shw, 0.04,    E50_ic,  dc_water),
        'NEON':    (17.4, 10., TN * 10 ** (2 / 3),  150., psf_arca_trk, psf_neon_shw, 0.12,    E50_ic,  dc_water),
        'HUNT':    (17.4, 30., TN * 30 ** (2 / 3),  3e3,  psf_hunt_trk, psf_gvd_shw,  0.10,    5e3,     dc_ice),
    }
    aeff100 = {}
    for k, (lat, V, tn, Eth, ptrk, pshw, sig, Ed_, dc) in nextgen.items():
        aeff100[k] = numu_aeff_100TeV(tn, Eth)      # with the detector's own track threshold
        common = dict(dc=dc, sig_lgE=sig, mult_E=mult_mu)
        if k == 'HUNT':
            out[k + '_base'] = water(lat, V, tn, Eth, ptrk, pshw, lambda x: sigmoid(x, 3e4) * hi_arca(x), 6e4, **common)
        else:
            out[k + '_base'] = water(lat, V, tn, Eth, ptrk, pshw, eff_arca_up, down_gate=gate_arca_down, **common)
        out[k + '_mid'] = water(lat, V, tn, Eth, ptrk, pshw, eff_arca_up, 6e4, **common)
        out[k + '_ded'] = water(lat, V, tn, Eth, ptrk, pshw, lambda x, a=Ed_: sigmoid(x, a) * hi_arca(x), 6e4,
                                **common)

    # ---------------- IceCube-Gen2 optical array (IceCube + 120 new strings) ----------------
    # 'Gen2_IC': the IceCube strings during Gen2 operation, i.e. IceCube as calibrated above with Gen2's track PSF.
    # 'Gen2_new_*': the new strings, with their own thresholds, PSFs and double cascades.
    out['Gen2_IC'] = dict(P_ic, trk_psf=psf_g2_trk)
    P_g2 = dict(P_ic, trk_norm=1.0, trk_Eth=2e3, trk_psf=psf_g2_trk, shw_V=1.0 * KM3, shw_psf=psf_g2_shw,
                shw_eff=lambda Ed: sigmoid(Ed, 2e5 / 3, 2.5), dc_eff=dc_g2, dc_misid=0.05)
    # tracks: the TXS 2014-15 flare gives 38 events in Gen2 (TDR p.29), 13 of them in the IceCube part
    n_txs = lambda P: Detector('x', dict(P, years=1 / 3.15576e7), build=False).source_counts('trk', J_txs, (0, 1, 0))
    P_g2['trk_norm'] = (38. - n_txs(P_ic)) / n_txs(P_g2)
    # showers: both parts give the TDR's nu_e cascade A_eff of 133 m^2 at 1 PeV (Fig. 42, read as toise output: 4pi
    # average with Earth absorption). Cross-check: toise's plateau, 0.75 x V_fid ~ 4.4 km^3, minus IceCube's 1 km^3
    P_g2['shw_V'] = (133. - nue_shw_aeff(P_ic, 1e6)) / nue_shw_aeff(P_g2, 1e6) * KM3
    E50_g2 = E50_ic * P_g2['trk_Eth'] / P_ic['trk_Eth']     # [ASSUMPTION] IceCube's E50 x the track-threshold ratio
    E50_g2_mid = E50_mid * P_g2['trk_Eth'] / P_ic['trk_Eth']   # the mid scenario's turn-on, scaled the same way
    out['Gen2_new_base'] = P_g2
    out['Gen2_new_mid'] = dict(P_g2, shw_eff=lambda Ed, E50=E50_g2_mid: sigmoid(Ed, E50, 6))
    out['Gen2_new_ded'] = dict(P_g2, shw_eff=lambda Ed, E50=E50_g2: sigmoid(Ed, E50))

    # ---------------- Down-going tracks in water (the second analysis) ----------------
    # Atmospheric muon bundles: ARCA's simulation has 8.1e7 per year above 10 TeV at trigger level (EPJC 84, 885,
    # Sec. 4.1). The model's single muons above 10 TeV at ARCA's depth, through its muon area trk_norm / N_A, are fewer
    # by the bundle factor fb, which then scales the single-muon background of every down-going track selection.
    mu_arca = muon_rate(DEPTH['ARCA'], 1e4, TN / NA)
    fb = 8.1e7 / mu_arca
    for k, depth in DEPTH.items():                  # the down-going track channel alone, beside '<k>_<scenario>'
        out[k + '_dn'] = dict(out[k + '_ded'], channels=(),
                              trkd=dict(depth=depth, rho=RHO_SEA, lgcut=TRKD_LGCUT, fb=fb))

    # ---------------- double cascades, anchored (see DC_RATE above); after every other calibration ----------------
    dc_norm = {'IceCube': 1.}                           # IceCube's family: MESE's measured fraction as it is
    for fam, shape in {f: s for f, s in DC_FAMILY.values() if f in DC_RATE}.items():
        ref = {'P-ONE': 'P-ONE_ded', 'TRIDENT': 'TRIDENT_ded', 'GVD': 'GVD_ded'}[fam]
        dc_norm[fam] = DC_RATE[fam] / nutau_dc_rate(dict(out[ref], dc_eff=shape))
    for k in out:
        base = 'Gen2_IC' if k == 'Gen2_IC' else k.split('_')[0]
        if base == 'Gen2':                                  # the new strings: the TDR's curve, 5% misidentification
            out[k] = dict(out[k], dc_misid_mu=0.02)
            continue
        fam, shape = DC_FAMILY[base]
        out[k] = dict(out[k], dc_eff=lambda E, n=dc_norm[fam], s=shape: n * s(E), dc_misid=0.02, dc_misid_mu=0.01)
    info_dc = {f'dc_norm_{f}': v for f, v in dc_norm.items() if f in DC_RATE}

    info = dict(ic_trk_Eth=P_ic['trk_Eth'], arca_trk_Eth=TE, gvd_trk_Eth=P_gv['trk_Eth'],
                Aeff100_gvd_m2=numu_aeff_100TeV(P_gv['trk_norm'], P_gv['trk_Eth']),
                E50_ic=E50_ic, arca_shw_atm_per_yr=arca_shw_atm, arca_shw_bkg_scale=k_nu,
                arca_shw_cosmic_eq6_per_yr=arca_cos, **{f'arca_trk_aeff_7b_{e / 1e3:.0f}TeV_m2': a for e, a in a7b.items()},
                E50_mid=E50_mid, eps_gvd=eps_gvd,
                **{f'arca_mu_per_nu_{e / 1e3:.0f}TeV': r for e, r in zip(mu_E, mu_r)},
                Aeff100_arca_m2=A_arca100, Aeff100_trident_m2=aeff100['TRIDENT'], Aeff100_neon_m2=aeff100['NEON'],
                Aeff100_hunt_m2=aeff100['HUNT'], gen2_trk_norm_ratio=P_g2['trk_norm'] / P_ic['trk_norm'],
                gen2_new_shw_V_km3=P_g2['shw_V'] / KM3, E50_gen2_ded=E50_g2, E50_gen2_mid=E50_g2_mid,
                arca_mu_single_10TeV_per_yr=mu_arca, trkd_fb=fb, **info_dc)
    if verbose:
        print('Calibration:', {k: round(float(v), 4) for k, v in info.items()})
    return out, info
