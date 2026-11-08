"""
existing_bounds.py -- the bounds of other works drawn in gray on the paper's limit figures (Appendix G; started
2026-11-07). Each curve is (mediator mass M [eV], coupling) in the coordinates of our planes, with the excluded side
above it, and a provenance record: the source, its figure, how the numbers were obtained, and what is still to be
verified before the paper uses it.

The curves on disk come from Mauricio's earlier projects (src/data/existing_bounds/README.md says from where);
coupling conventions: g' for L_e - L_alpha, the coupling of Eq. (lag_le), which is the "effective coupling G' = g_Z'"
of Singh, Bustamante & Agarwalla, JHEP 09 (2024) 055, Eq. (2.6), whose Fig. 1 drew these same files.
"""
import os

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "data", "existing_bounds")
CM_TO_INV_EV = 5.067730716e4          # 1 cm in eV^-1
G_GRAV = 1.7518e-45                   # the gravitational coupling G_N m_u^2 (as in the source script)

# key: label, the models whose figures show it, its file (log10 M/eV, log10 coupling) or None, kind, notes
BOUNDS = {
    "coloma_osc": dict(label="Oscillation data (global fit)", texkey="Coloma:2020gfv", models=["le_lmu"],
                       file="coloma_2009.14220_emu.txt", kind="curve",
                       verify="L_e - L_mu only on disk (L_e - L_tau to digitize from the same figure); C.L. "
                              "as labeled in JHEP 09 (2024) 055 (2 sigma)"),
    "dror_early_universe": dict(label="Early Universe (longitudinal mode)", texkey="Dror:2020fbh",
                                models=["le_lmu", "le_ltau"],
                                file={"le_lmu": "dror_2004.04750_fig4_dirac_emu.txt",
                                      "le_ltau": "dror_2004.04750_fig4_dirac_etau.txt"}, kind="curve",
                                verify="Dirac neutrinos (Fig. 4); our neutrinos are Majorana: take Dror's "
                                       "Majorana case if it differs"),
    "precession": dict(label="Planetary precession", texkey="KumarPoddar:2020kdz", models=["le_lmu", "le_ltau"],
                       file="poddar_2002.02935_precession.txt", kind="curve",
                       verify="C.L. to check in the source; one curve used for both models"),
    "binaries": dict(label="Compact binaries", texkey=None, models=[],
                     file="poddar_1908.09732_binaries.txt", kind="curve",
                     verify="NOT DRAWN: arXiv:1908.09732 bounds a gauged L_mu - L_tau (its title, checked "
                            "2026-11-08); in a neutron star the L_e - L_mu charge is n_e - n_mu, not n_mu, so the "
                            "curve does not carry over (JHEP 09 (2024) 055, Fig. 1, labeled it L_e - L_mu)"),
    "astro_flavor_2023": dict(label="High-energy flavor (earlier estimate)", texkey="Agarwalla:2023sng",
                              models=["le_lmu", "le_ltau"],
                              file={"le_lmu": "agarwalla_2305.03675_emu.txt",
                                    "le_ltau": "agarwalla_2305.03675_etau.txt"}, kind="curve",
                              verify="95% C.L., estimated IceCube 2020 flavor data, normal ordering"),
    "torsion": dict(label="Torsion balance", texkey=None, models=["le_lmu", "le_ltau"],
                    file="adelberger_0712.0607_torsion_raw.txt", kind="torsion",
                    verify="no charge-per-mass (Z/A) factor applied, as in the source script; INSPIRE texkey to fetch"),
    "superradiance": dict(label="Black-hole superradiance", texkey="Baryakhtar:2017ngi", models=["le_lmu", "le_ltau"],
                          file=None, kind="mass_bands",
                          bands=[(3.924e-14, 1.715e-11), (1.657e-19, 2.103e-17), (5.110e-20, 1.083e-19)],
                          verify="mass ranges excluded at weak coupling (1704.05081 Fig. 5); where they stop "
                                 "applying at strong coupling is not drawn"),
}


BOUNDS["cmb_lifetime"] = dict(label="CMB (neutrino lifetime)", texkey="Escudero:2019gfk",
                              models=["dp_e", "dp_mu", "dp_tau", "le_lmu", "le_ltau"], file=None, kind="computed",
                              verify="computed with our decay rates (cmb_lifetime below); Planck 2018, 95% C.L.")


def _path(name):
    return os.path.join(HERE, name)


def load(key, model=None):
    """(M [eV], coupling) arrays of a curve, or the list of (M_lo, M_hi) [eV] of a mass-band bound."""
    b = BOUNDS[key]
    if b["kind"] == "mass_bands":
        return b["bands"]
    f = b["file"][model] if isinstance(b["file"], dict) else b["file"]
    d = np.loadtxt(_path(f))
    if b["kind"] == "torsion":
        lam_m, alpha = d[:, 0], d[:, 1]
        M = 1.0 / (lam_m * 1e2 * CM_TO_INV_EV)
        g = np.sqrt(alpha * G_GRAV * 4.0 * np.pi)
        o = np.argsort(M)
        return M[o], g[o]
    return 10.0 ** d[:, 0], 10.0 ** d[:, 1]


def for_model(model):
    """The keys of the bounds drawn on `model`'s figure ('le_lmu', 'le_ltau', ...)."""
    return [k for k, b in BOUNDS.items() if model in b["models"]]


# ---------------------------------------------------------------- DarkCast (MeV-GeV laboratory bounds, L_e - L_alpha)
# DarkCast's recast limits (gitlab.com/darkcast/releases; GPL-2+, so not copied into this repository): one file per
# experiment, columns mass [GeV], lower[, upper] coupling, for a boson with charges +1 (e, nu_e) and -1 (alpha, nu_alpha)
# plus the loop-induced kinetic mixing -- the coupling g' of Eq. (lag_le) (checked in DarkCast's models/Le-L*_boson.py,
# 2026-11-07). The folder is read from NULRI_DARKCAST, default ~/Research/nulri_data/darkcast/darkcast.
DARKCAST = os.environ.get("NULRI_DARKCAST", os.path.expanduser("~/Research/nulri_data/darkcast/darkcast"))
DARKCAST_MODEL = {"le_lmu": "Le-Lmu_boson", "le_ltau": "Le-Ltau_boson"}


def darkcast(model):
    """[(experiment, M [eV], g_lower, g_upper or None)] of DarkCast's recast limits for `model`; [] if not available.
    The excluded region of each lies between its lower and upper curves (above the lower one where there is no
    upper)."""
    import glob as _glob
    d = os.path.join(DARKCAST, "recast", "limits", DARKCAST_MODEL.get(model, ""))
    out = []
    for f in sorted(_glob.glob(os.path.join(d, "*.lmt"))):
        cols = open(f).readline().lstrip("#").split()
        a = np.loadtxt(f, ndmin=2)
        if a.size == 0:
            continue
        up = a[:, cols.index("upper")] if "upper" in cols else None
        out.append((os.path.basename(f)[:-4], a[:, 0] * 1e9, a[:, cols.index("lower")], up))
    return out


# ---------------------------------------------------------------- computed: the CMB bound on invisible neutrino decay
# Escudero & Fairbairn, arXiv:1907.05425, Eq. (13) (Planck 2018 TTTEEE+lowE+lensing, 95% C.L.):
# tau(nu_i -> nu_j + boson) > 1.3e9 s (m_i / 0.05 eV)^3. Our rate is dp_decay.rest_rates (with the longitudinal mode's
# (m_i^2 - m_j^2)^2 / M^2 term), so for L_e - L_alpha this is the kind of bound Dror (arXiv:2004.04750, Fig. 4, red)
# draws. The lightest mass is profiled in our planes, so the curve takes, at each M, the weakest bound over the
# lightest masses given (the most conservative). Caveat: the bound assumes the boson is not converted back into
# neutrinos before recombination (decision B's decay back acts on the high-energy beam, not on this).
CMB_TAU_S = 1.3e9
S_TO_INV_EV = 1.0 / 6.582119569e-16


def cmb_lifetime(cfg_path, masses_eV, m_lightest=(1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 0.1), g_ref=1e-8):
    """(M [eV], coupling bound) of the CMB lifetime bound on the coupling of the plane in `cfg_path` (its y axis):
    each heavier state's total rate into lighter ones scales as g^2; the bound excludes couplings above the curve.
    np.nan where no decay is open at any lightest mass."""
    import json as _json
    import sys as _sys
    _sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")]
    import dark_portal_solver as dps
    import dp_absorption
    import dp_decay
    from dp_core import construct_PMNS
    from analysis.config import from_dict
    raw = _json.load(open(cfg_path))
    model = from_dict(raw, cfg_path).model()
    xn, yn = raw["planes"][0]["x"], raw["planes"][0]["y"]
    out = np.full(len(masses_eV), np.nan)
    for k, M in enumerate(masses_eV):
        best = np.nan
        for ml in m_lightest:
            v = {xn: float(M), yn: g_ref, "m_lightest_eV": ml}
            E, p = dps.build_solver_params(model._run_cfg(model._overrides(v)))
            m = np.asarray(p["m_nu_bg_arr"], dtype=np.float64)
            U = construct_PMNS(p["th12"], p["th13"], p["th23"], p["d_cp"])
            if p.get("G_coupling") is not None:
                G = np.asarray(p["G_coupling"], dtype=np.complex128)
            else:
                u = np.array([p["u_e"], p["u_mu"], p["u_tau"]], dtype=np.complex128)
                G = dp_absorption.dark_portal_coupling(p["g_D"], u, 0.0)
            Gam, _ = dp_decay.rest_rates(G, U, m, float(M))
            gs = []
            for i in range(3):
                tot = Gam[i].sum()
                if tot > 0.0:
                    tau_min = CMB_TAU_S * S_TO_INV_EV * (m[i] / 0.05) ** 3
                    gs.append(g_ref * np.sqrt(1.0 / (tot * tau_min)))     # rate = tot (g / g_ref)^2
            if gs:
                g_ml = min(gs)                                   # the most constraining state at this lightest mass
                best = g_ml if not np.isfinite(best) else max(best, g_ml)   # the weakest over lightest masses
        out[k] = best
    return np.asarray(masses_eV, dtype=float), out


# ---------------------------------------------------------------- digitized 2026-11-08 (the PDFs' vector paths)
# Each file's header records the source, figure, curve, C.L., the coupling convention as written, the excluded side and
# the method; src/data/existing_bounds/README.md lists them. Factors to our couplings (checked against each source's
# Lagrangian): Dror's g_X, Coloma's g', Bauer's g_x = our g' (L_e - L_alpha); Blinov's g_phi and Brdar's y = our G
# (scalar, Majorana: (1/2)(g phi nu nu + h.c.) = -kappa phi G nubar nu with kappa = 1/2); the white paper's figures
# follow Blinov's normalization (inferred from their coinciding lab lines). `draw` False: kept for cross-checks only.
_LE = ("le_lmu", "le_ltau")
_F = {"le_lmu": "emu", "le_ltau": "etau"}


def _reg(key, label, texkey, models, file, draw=True, verify="", kind="curve", region="above"):
    BOUNDS[key] = dict(label=label, texkey=texkey, models=list(models), file=file, kind=kind, draw=draw,
                       verify=verify, region=region)


# L_e - L_alpha, ultralight: Coloma et al. v5 (2 sigma) replaces the older file; their own fifth-force and equivalence-
# principle recasts per model replace the torsion-balance curve (which had no charge-per-mass factor)
BOUNDS["coloma_osc"]["draw"] = False
BOUNDS["coloma_osc"]["verify"] = "SUPERSEDED by coloma_osc_v5 (0.24-0.43 dex below it; an older digitization)"
BOUNDS["torsion"]["draw"] = False
BOUNDS["torsion"]["verify"] = "SUPERSEDED by Coloma et al.'s model-specific fifth-force/EP recasts"
for k in BOUNDS:
    BOUNDS[k].setdefault("draw", True)
    BOUNDS[k].setdefault("region", "above")
_reg("coloma_osc_v5", "Oscillations (global fit)", "Coloma:2020gfv", _LE,
     {m: f"coloma_2009.14220v5_fig2_{_F[m]}_osc.txt" for m in _LE}, verify="2 sigma (Delta chi^2 > 4)")
_reg("coloma_fifthforce", "Fifth-force searches", "Coloma:2020gfv", _LE,
     {m: f"coloma_2009.14220v5_fig2_{_F[m]}_fifthforce.txt" for m in _LE})
_reg("coloma_eqprinc", "Equivalence-principle tests", "Coloma:2020gfv", _LE,
     {m: f"coloma_2009.14220v5_fig2_{_F[m]}_eqprinc.txt" for m in _LE})
for curve, label, draw in (("mesons", "Meson decays (longitudinal mode)", True),
                           ("0vbb", r"$0\nu\beta\beta$ with emission", True),
                           ("bbn_sn", "BBN and supernovae (longitudinal mode)", True),
                           ("nudecay_lab", "Neutrino decay (solar neutrinos)", True),
                           ("nudecay_cmb", "Neutrino decay (CMB, Dror)", False),
                           ("unitarity", "Unitarity (Dror's estimate)", False),
                           ("prev_nuosc", "Oscillations (as in Dror)", False),
                           ("prev_fifthforce_ep", "Fifth force + EP (as in Dror)", False)):
    _reg(f"dror_{curve}", label, "Dror:2020fbh", _LE,
         {m: f"dror_2004.04750_fig4_majorana_{_F[m]}_{curve}.txt" for m in _LE}, draw=draw,
         verify={"nudecay_cmb": "replaced by cmb_lifetime (Planck 2018, our rates)",
                 "unitarity": "our own unitarity line is 0.66 dex stronger; cross-check only",
                 "prev_nuosc": "replaced by coloma_osc_v5", "prev_fifthforce_ep": "replaced by Coloma's recasts"
                 }.get(curve, "Majorana panel; C.L. not stated in the paper"))
BOUNDS["dror_early_universe"]["draw"] = False
BOUNDS["dror_early_universe"]["verify"] = "DIRAC panel: replaced by the Majorana curves dror_*"
# L_e - L_alpha, MeV: Bauer et al. (above 2 MeV; SN and BBN not shown there), besides DarkCast
for curve, label in (("whitedwarfs", "White-dwarf cooling"), ("superk", "Super-K (matter effects)")):
    _reg(f"bauer_{curve}", label, "Bauer:2018onh", _LE,
         {"le_lmu": f"bauer_1803.05466_fig14_emu_{curve}.txt", "le_ltau": f"bauer_1803.05466_fig15_etau_{curve}.txt"},
         verify="drawn by the source above 2 MeV only" + ("; trapping edge g' = 0.082 M/GeV not drawn"
                                                          if curve == "whitedwarfs" else ""))
# secret interactions, scalar, Majorana
_SI = {"e": "ee", "mu": "mumu", "tau": "tautau", "universal": "universal"}
for fl, ff in _SI.items():
    mod = [f"si_scalar_majorana_{fl}"]
    _reg(f"blinov_lab_{fl}", "Laboratory", "Blinov:2019gcj", mod, f"blinov_1905.02727_fig1_{fl}_lab.txt")
    _reg(f"brdar_zinv_{fl}", r"$Z$ invisible width", "Brdar:2020nbj", mod, f"brdar_2003.05339_fig4_{ff}_zinv.txt",
         verify="3 sigma")
    _reg(f"brdar_taudecay_{fl}", r"$\tau$ decays", "Brdar:2020nbj", mod, f"brdar_2003.05339_fig4_{ff}_taudecay.txt",
         verify="3 sigma; Brdar's meson curves dropped (swapped ee/mumu vs Blinov and the white paper)")
    _reg(f"blinov_bbn_{fl}", "BBN", "Blinov:2019gcj", mod, "blinov_1905.02727_fig1_bbn_sinu_real.txt",
         kind="vertical", region="left", verify="m_phi > 1.3 MeV (real scalar; the weakest of Blinov's BBN lines)")
_reg("wp_phibb_ee", r"$0\nu\beta\beta\phi$", "Berryman:2022hds", ["si_scalar_majorana_e"],
     "wp_2203.01955_fig9_phibb_ee.txt", verify="original: arXiv:1808.08158")
_reg("wp_hese7p5_tautau", "IceCube HESE 7.5 yr", "Esteban:2021tub", ["si_scalar_majorana_tau"],
     "wp_2203.01955_fig10_tautau_icecube_hese7p5yr.txt")
for curve, label, tk, region in (("sn1987a_prop_shalgar", "SN 1987A (propagation)", "Shalgar:2019rqe", "above"),
                                 ("sn1987a_core_shalgar", "SN 1987A (core)", "Shalgar:2019rqe", "above"),
                                 ("sn1987a_core_kolbturner", "SN 1987A (core, Kolb & Turner)", "Kolb:1987qy", "inside"),
                                 ("cmb", "CMB", "Archidiacono:2013dua", "above"), ("txs0506", "TXS 0506+056", "Kelly:2018tyg", "above"),
                                 ("icecube_hese6yr_dip1", "IceCube HESE 6 yr", "Bustamante:2020mep", "above"),
                                 ("icecube_hese6yr_dip2", "IceCube HESE 6 yr", "Bustamante:2020mep", "above")):
    _reg(f"wp_{curve}", label, tk, ["si_scalar_majorana_universal"], f"wp_2203.01955_fig9_{curve}.txt",
         region=region, verify="flavor-universal; original source in the file header" + ("" if tk else
                                                                                          "; texkey to add"))
