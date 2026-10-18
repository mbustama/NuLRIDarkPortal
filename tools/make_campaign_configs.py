"""
make_campaign_configs.py -- the dark-portal paper's plane campaign (docs/dev/SCOPE_WIDENING_PLAN.md, step 5).

Writes analysis configurations to analysis/configs/campaign/<data set>/<name>.json (outputs in
output/analysis/campaign/<data set>/<name>) and a list with each plane's points, for packing on Tycho:
analysis/configs/campaign/<data set>/campaign.json.

Per data set (DATASETS), the planes (decided with Mauricio 2026-10-14 unless marked technical):
  dark portal, for each named texture (pure e, mu, tau; e-mu, e-tau, mu-tau; democratic): the rank-one texture
    u u^dag with Sum_alpha |U_alpha4|^2 = 0.5 (technical: the planes scan products, so only the direction and the
    perturbativity bound g_D <= sqrt(4 pi) depend on it; 0.5 keeps g_D Sum|U|^2 <= 1.8 within reach);
    potential planes in (m_Zd, P = g_D eps' Sum|U|^2) over 1e-30 - 1e4 eV, both signs of the potential, two curves:
      the balanced split (dark_portal.coupling_split 'balanced': screening negligible) and the variants of the
      split's profile (x = g_D Sum|U|^2 = 1e-24 ... 1e-8, each over the masses where its screening differs from the
      balanced run's, m_Zd <= 2e-4 x eV, and P <= x so that eps' <= 1; tools/combine_split_planes.py);
    the MeV plane in (m_Zd, g_D Sum|U|^2) at eps' = 0, 1e4 - 1e9 eV (absorption; the cosmological phase through
      dp_si's every collision, cosmo_transport 'auto').
  L_mu - L_tau: the same with P = g' eps' and the MeV plane in g'.
  the six direct gauge models: (m_Zd, g') over 1e-30 - 1e4 eV (the potential) and 1e4 - 1e9 eV (absorption).
Every plane also with the sources' flavor compositions fitted (Mauricio, 2026-10-14, "All planes"): the twin
<name>_fx frees the extragalactic (f_e_S, 1 - f_e_S, 0) and the Galactic (f_e_S_gal, 1 - f_e_S_gal, 0), each flat in
[0, 1], as task 2's _galx variants; nominal: both at pion decay (1/3, 2/3, 0).
Technical defaults: one mass per decade below 1e4 eV and eight per decade above (the resonances); one coupling
value per decade for the potential (P down to 1e-60), two for g' below 1e4 eV, five above; spectral-index nodes 3
for the potential planes and 7 where the MeV absorption goes through dp_si (its secondaries make the flux's
dependence on p least smooth, as for the secret interactions).

The broken power law (Mauricio, 2026-10-15; --spectrum bpl, the default since): IceCube's 2026 combined fit bends
below ~25 TeV, so the extragalactic flux is its broken power law (analysis/model.py: two source populations split
at E_b = 10^4.39 GeV, fixed), the upper index p_index and the lower one p_index_low free (technical: [0.5, 2.5] on
three nodes), normalized at 100 TeV (1.77e-18 per flavor: norm_astro = 1 is IceCube's fit). Configurations under
analysis/configs/campaign_bpl/, outputs under output/analysis/campaign_bpl/. The 2026-spectrum sets' grid extends
down to 4.6 TeV, where IceCube's lowest spectrum segment starts (technical: two segments below the break constrain
p_index_low; the flavor surfaces keep their 10 TeV - 30 PeV band). The future detectors' projection (proj_future)
exists only with it. --spectrum pl: the single power law of the first campaign (analysis/configs/campaign/).

  python tools/make_campaign_configs.py [--spectrum bpl|pl] [--datasets now_ic26 now_events proj_2050 proj_future]
                                        [--check]
"""
import argparse
import copy
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

import dp_models  # noqa: E402

BASE = os.path.join(REPO, "analysis", "configs", "pion_2050_case_5_potential_emu_gal.json")
D_LOG = 3.477121254719662 / 399                 # the flavor fits' node spacing in log10 E (10 TeV - 30 PeV, 400 nodes)
DATASETS = {
    # present-day flavor (IceCube's MESE 2025 measurement) with IceCube's 2026 spectrum
    "now_ic26": {"data": {"surface": "icecube_mese_2025", "spectrum": "ic26_astrophysical"}},
    # the HESE starting events and DR2 tracks alone (Mauricio, 2026-10-14: HESE's events are also MESE's, so no MESE
    # surface here; HESE's topologies carry the flavor; the grid extended down to ~1 TeV for DR2)
    "now_events": {"data": {"events": {"hese": {}, "dr2": {}}, "flavor_band_eV": [1e13, 3e16]},
                   "grid": {"n_E_bins": 515, "E_min_eV": 10 ** (13.0 - 115 * D_LOG), "E_max_eV": 3e16}},
    # the 2050 projection (IceCube-Gen2, KM3NeT, Baikal, P-ONE, HUNT; pion-decay centre) with the 2026 spectrum
    "proj_2050": {"data": {"surface": "pion_2050_case_5", "spectrum": "ic26_astrophysical"}},
}
# The broken power law (module docstring): IceCube's 2026 combined fit (PRD 113, 062002 (2026), Table 3, "CF BPL",
# as NuFlavorSingleSource's CF26): 1.77e-18 GeV^-1 cm^-2 s^-1 sr^-1 per flavor at 100 TeV, E^-2.735 above
# 10^4.39 GeV, E^-1.31 below
BPL = dict(E_break_GeV=10 ** 4.39, norm=1.77e-18, norm_E_GeV=1e5, p_index=2.735, p_index_low=1.31,
           low={"start": 1.31, "bounds": [0.5, 2.5], "space": "linear", "nodes": 3})
GRID_IC26 = {"n_E_bins": 439, "E_min_eV": 10 ** (13.0 - 39 * D_LOG), "E_max_eV": 3e16}    # down to 4.57 TeV
DATASETS_BPL = {
    "now_ic26": {"data": {"surface": "icecube_mese_2025", "spectrum": "ic26_astrophysical",
                          "flavor_band_eV": [1e13, 3e16]}, "grid": GRID_IC26},
    "now_events": copy.deepcopy(DATASETS["now_events"]),
    "proj_2050": {"data": {"surface": "pion_2050_case_5", "spectrum": "ic26_astrophysical",
                           "flavor_band_eV": [1e13, 3e16]}, "grid": GRID_IC26},
    # the projection with the future detectors' events instead of the 2050 surface (docs/dev/FUTURE_DETECTORS_PLAN.md,
    # 2026-10-15): NuFlavorSingleSource's main case, 10 yr of each of the eight; always on Asimov data (analysis.run):
    # standard physics, pion-decay sources, IceCube's 2026 broken power law; the solver's grid down to ~100 GeV with
    # the same spacing, where the detectors' energy grid starts
    "proj_future": {"data": {"events": {"future": {"config": "All8_base"}}, "flavor_band_eV": [1e13, 3e16]},
                    "grid": {"n_E_bins": 630, "E_min_eV": 10 ** (13.0 - 230 * D_LOG), "E_max_eV": 3e16},
                    "asimov": {"norm_astro": 1.0, "p_index": BPL["p_index"], "p_index_low": BPL["p_index_low"]},
                    "asimov_fx": {"f_e_S": 1 / 3, "f_e_S_gal": 1 / 3}},
}
SPECTRA = {"pl": ("campaign", DATASETS), "bpl": ("campaign_bpl", DATASETS_BPL)}
# The mixing and the lightest mass varied (--mixing vary, 2026-10-18; docs/dev/MIXING_VARIATION.md): Table II of
# arXiv:2606.17132, normal ordering only. Present-day data under NuFIT 6.1's pairwise chi^2, projections under Table
# II's Gaussians (central value, width); every fit at Table II's central values (theta0) varies the six through
# templates (analysis/templates.py; the schemes per data kind technical, from the accuracy scans); the mass sum under
# the paper's half-Gaussian bounded at zero with Table II's limits, 0.12 eV now and 0.09 eV for projections (Mauricio,
# 2026-10-18), which are 95% C.L. there and in its sources (PDG 2024's Planck 2018 + BAO; Palanque-Delabrouille et al.
# 2020 and Di Valentino et al. 2021): the limit holds 95% of the half-Gaussian, sigma = limit / 1.960 ("inspect what's
# correct", 2026-10-18; docs/dev/MIXING_VARIATION.md). Configurations under analysis/configs/campaign_mix/.
TABLE2 = dict(s12sq=(0.3088, 0.0016416), s13sq=(0.02248, 0.00059), s23sq=(0.470, 0.00418), d_cp_deg=(212.0, 6.687),
              dm21_sq=(7.537e-5, 3.7685e-7), dm31_sq=(2.511e-3, 1.2555e-5))
MIX_KIND = {"now_ic26": "now", "now_events": "now", "proj_2050": "proj", "proj_future": "proj"}
MIX_BOUNDS = {"now": dict(s12sq=[0.24, 0.38], s13sq=[0.018, 0.027], s23sq=[0.38, 0.65], d_cp_deg=[0.0, 360.0],
                          dm21_sq=[6.4e-5, 8.7e-5], dm31_sq=[2.40e-3, 2.62e-3]),
              "proj": {n: [c - 8.0 * w, c + 8.0 * w] for n, (c, w) in TABLE2.items()}}
MIX_TEMPLATES = {"now": dict(s12sq={"points": [0.2888, 0.3288]}, s13sq={"step": 0.001},
                             s23sq={"points": [0.40, 0.55, 0.60]}, d_cp_deg={"turn": 7}, dm21_sq={"step": 3e-6},
                             dm31_sq={"step": 4e-5}),
                 "proj": dict(s12sq={"step": 0.003}, s13sq={"step": 0.001}, s23sq={"step": 0.008},
                              d_cp_deg={"points": [200.0, 224.0]}, dm21_sq={"step": 7e-7}, dm31_sq={"step": 2.5e-5})}
MASS_SUM = {"now": 0.12, "proj": 0.09}
M_LIGHTEST = 0.01                # eV, every plane until the lightest-mass study decides (tools/mlightest_study.py)


def apply_mixing(cfg: dict, dataset: str, m_lightest: float = M_LIGHTEST, exact: bool = False) -> dict:
    """cfg with the mixing varied (the module's notes on --mixing vary): the solver at Table II's central values and
    the given lightest mass in the normal ordering, the six mixing parameters free with their templates, and the
    data kind's penalties. exact (--mixing exact, 2026-10-18): no templates, the lightest mass free from the mass sum's
    minimum, and the exact profile (analysis/exact_profile.py) -- both octants under NuFIT, the lightest mass's scan at
    the MeV planes."""
    kind = MIX_KIND[dataset]
    if exact:
        from analysis.exact_profile import M_LIGHTEST_MIN, ML_SCAN
        m_lightest = M_LIGHTEST_MIN
    deg = lambda s2: float(np.degrees(np.arcsin(np.sqrt(s2))))
    sm = cfg["solver"].setdefault("standard_mixing", {})
    sm.update(th12_deg=deg(TABLE2["s12sq"][0]), th13_deg=deg(TABLE2["s13sq"][0]), th23_deg=deg(TABLE2["s23sq"][0]),
              d_cp_deg=TABLE2["d_cp_deg"][0], dm21_sq=TABLE2["dm21_sq"][0], dm31_sq=TABLE2["dm31_sq"][0])
    cfg["solver"].setdefault("cosmology", {}).update(mass_ordering="normal", m_lightest_eV=float(m_lightest))
    for n, (c, _) in TABLE2.items():
        cfg["parameters"][n] = {"start": c, "bounds": list(MIX_BOUNDS[kind][n]), "space": "linear"}
        if not exact:
            cfg["parameters"][n]["template"] = dict(MIX_TEMPLATES[kind][n])
    if exact:
        cfg["parameters"]["m_lightest_eV"] = {"start": M_LIGHTEST_MIN, "bounds": [M_LIGHTEST_MIN, 0.1],
                                              "space": "linear"}
        ex = {}
        if kind == "now":
            ex["octants"] = True
        if cfg["name"].endswith("_mev"):
            ex["ml_scan"] = list(ML_SCAN)
            ex["scan_mixing"] = {n: c for n, (c, _) in TABLE2.items()}     # shared by every data set's scan
        cfg["exact_profile"] = ex
    if kind == "now":
        cfg["priors"].append({"type": "nufit_6.1_pairwise", "ordering": "normal"})
    else:
        cfg["priors"].extend({"name": n, "type": "gaussian", "centre": c, "width": w} for n, (c, w) in TABLE2.items())
    cfg["priors"].append({"name": "sum_masses_eV", "type": "half_gaussian", "limit": MASS_SUM[kind], "cl": 0.95,
                          "quantile": "half"})
    if exact:
        pen = "NuFIT 6.1 pairwise" if kind == "now" else "Table II Gaussians"
        cfg["_comment"] += (f"; mixing and m_lightest profiled exactly ({pen}; analysis/exact_profile.py), mass sum "
                            f"< {MASS_SUM[kind]:g} eV (95%, half-Gaussian)")
    else:
        cfg["_comment"] += (f"; mixing varied ({'NuFIT 6.1 pairwise' if kind == 'now' else 'Table II Gaussians'}, "
                            f"templates), m_lightest {m_lightest:g} eV, mass sum < {MASS_SUM[kind]:g} eV (95%, "
                            f"half-Gaussian)")
    return cfg
TEXTURES = {"e": (1, 0, 0), "mu": (0, 1, 0), "tau": (0, 0, 1), "emu": (1, 1, 0), "etau": (1, 0, 1),
            "mutau": (0, 1, 1), "emutau": (1, 1, 1)}
SUM_U2 = 0.5
MODEL_KEYS = {"dark_portal": "dp", "L_mu-L_tau": "lmu_ltau", "L_e-L_mu": "le_lmu", "L_e-L_tau": "le_ltau",
              "B-3L_e": "b_3le", "B-3L_mu": "b_3lmu", "B-3L_tau": "b_3ltau", "B-3/2(L_mu+L_tau)": "b_32lmultau"}
SPLITS = (1e-24, 1e-20, 1e-16, 1e-12, 1e-8)
M_POT = (-30, 4)                 # log10 m_Zd range of the potential planes, one per decade
M_MEV = (4, 9, 41)               # the MeV planes: eight per decade
P_RANGE = (-60, -6)              # log10 P, one per decade
G_MEV = (-10, -2, 41)            # log10 of g_D Sum|U|^2 (or g') in the MeV planes, five per decade
G_DIRECT = (-32, -2)             # log10 g' below 1e4 eV, two per decade
COMPOSITIONS = {"f_e_S": {"start": 0.333333, "bounds": [0.0, 1.0], "space": "linear"},
                "f_e_S_gal": {"start": 0.333333, "bounds": [0.0, 1.0], "space": "linear"}}
FIT = {"n_random": 0, "seed": 20261014, "max_evals": 400, "fd_step": 0.001, "ftol": 1e-09, "xtol": 0.001,
       "polish": False}


def _texture_u(weights):
    w = np.asarray(weights, dtype=float)
    u = np.sqrt(SUM_U2 * w / w.sum())
    return dict(u_e=float(u[0]), u_mu=float(u[1]), u_tau=float(u[2]))


def _base(dataset, spectrum):
    with open(BASE) as f:
        cfg = json.load(f)
    solver = copy.deepcopy(cfg["solver"])
    solver.pop("_comment", None)
    ds = SPECTRA[spectrum][1][dataset]
    if "grid" in ds:
        solver["grid"].update(ds["grid"])
    if spectrum == "bpl":
        solver["source"].update(p_index=BPL["p_index"], E_break_GeV=BPL["E_break_GeV"], p_index_low=BPL["p_index_low"])
        solver["sky"].update(extragalactic_norm=BPL["norm"], extragalactic_norm_E_GeV=BPL["norm_E_GeV"])
    return solver, cfg["parameters"], cfg["priors"]


def _config(dataset, spectrum, name, comment, solver, params, priors, plane, asimov):
    # flux_basis 'pure_flavors' (2026-10-14): the fixed-composition planes run the same pure-flavor sources as their
    # fitted twins, and the two spectrum data sets the same physics, so one set of solver runs serves four planes
    # (NULRI_FLUX_CACHE); the composition is linear, to rounding (1e-9 through dp_si's MeV transport)
    tree, sets = SPECTRA[spectrum]
    return {"_comment": comment, "name": name, "data": copy.deepcopy(sets[dataset]["data"]),
            "solver": solver, "parameters": params, "priors": copy.deepcopy(priors), "fit": dict(FIT),
            "profiles": {}, "planes": [plane], "asimov": {**sets[dataset].get("asimov", {}), **asimov},
            "flux_basis": "pure_flavors",
            "output_dir": f"output/analysis/{tree}/{dataset}/{name}"}


def _common_params(base_params, nodes, spectrum="pl"):
    p = {k: copy.deepcopy(base_params[k]) for k in ("norm_gal", "norm_astro", "p_index")}
    p["p_index"]["nodes"] = nodes
    if spectrum == "bpl":
        p["p_index"]["start"] = BPL["p_index"]
        p["p_index_low"] = copy.deepcopy(BPL["low"])
    return p


def _decades(lo, hi):
    return lo, hi, hi - lo + 1


def planes_for(dataset, spectrum="pl", mixing="fixed"):
    """[(name, config, n_points)] for one data set; mixing 'vary' (with spectrum 'bpl'): the tree campaign_mix."""
    solver0, base_params, priors = _base(dataset, spectrum)
    sets = SPECTRA[spectrum][1]
    out = []

    def add(name, comment, solver, params, plane, asimov):
        cfg = _config(dataset, spectrum, name, comment, solver, params, priors, plane, asimov)
        if mixing in ("vary", "exact"):
            cfg = apply_mixing(cfg, dataset, exact=mixing == "exact")
            tree = "/campaign_mix/" if mixing == "vary" else "/campaign_exact/"
            cfg["output_dir"] = cfg["output_dir"].replace("/campaign_bpl/", tree)
        out.append((name, cfg, plane["nx"] * plane["ny"]))
        fx = copy.deepcopy(cfg)
        fx["name"] = name + "_fx"
        fx["_comment"] += "; the sources' flavor compositions fitted (extragalactic and Galactic, independently)"
        fx["parameters"].update(copy.deepcopy(COMPOSITIONS))
        fx["asimov"].update(sets[dataset].get("asimov_fx", {}))
        fx["output_dir"] += "_fx"
        out.append((name + "_fx", fx, plane["nx"] * plane["ny"]))

    for model in dp_models.model_names():
        kind = dp_models.spec(model)[0]
        key = MODEL_KEYS[model]
        textures = TEXTURES.items() if model == "dark_portal" else [(None, None)]
        for tex, w in textures:
            stem = key if tex is None else f"{key}_{tex}"
            solver = copy.deepcopy(solver0)
            solver["model"] = {"name": model}
            dp = solver.setdefault("dark_portal", {})
            dp.update(use_debye_screening=True)
            if model == "dark_portal":
                dp.update(_texture_u(w))
            else:
                dp.update(u_e=0.0, u_mu=0.0, u_tau=0.0)
            if kind == "mixing":
                prod = "g_D_eps_trace" if model == "dark_portal" else "g_D_eps"
                mev_axis = "g_D_trace" if model == "dark_portal" else "g_D"
                for sign in (1, -1):
                    sname = "pos" if sign > 0 else "neg"
                    for split in ("balanced",) + SPLITS:
                        s = copy.deepcopy(solver)
                        s["dark_portal"].update(coupling_split=split, potential_sign=sign, epsilon_prime=0.1)
                        if split == "balanced":
                            m_lo, m_hi = M_POT
                            p_hi = P_RANGE[1]
                            tag = "bal"
                        else:
                            m_hi = int(np.floor(np.log10(2e-4 * split)))
                            m_lo = M_POT[0]
                            if m_hi < m_lo:
                                continue
                            p_hi = int(round(np.log10(split)))
                            tag = f"x{split:.0e}".replace("e-", "m")
                        params = {"m_Zd_eV": {"start": 10.0 ** ((m_lo + m_hi) / 2), "bounds": [10.0 ** (m_lo - 1),
                                                                                             10.0 ** (m_hi + 1)],
                                              "space": "log10"},
                                  prod: {"start": 10.0 ** (P_RANGE[0] + 20), "bounds": [10.0 ** (P_RANGE[0] - 2),
                                                                                       10.0 ** p_hi],
                                         "space": "log10"}}
                        params.update(_common_params(base_params, 3, spectrum))
                        plane = {"x": "m_Zd_eV", "y": prod, "x_range": [10.0 ** m_lo, 10.0 ** m_hi],
                                 "y_range": [10.0 ** P_RANGE[0], 10.0 ** p_hi], "nx": m_hi - m_lo + 1,
                                 "ny": p_hi - P_RANGE[0] + 1, "raster": "m_Zd_eV"}
                        comment = (f"{model}{'' if tex is None else ', texture ' + tex}: the potential, "
                                   f"{'+' if sign > 0 else '-'} sign, coupling split {split} "
                                   f"(docs/dev/SCOPE_WIDENING_PLAN.md decision 4; tools/make_campaign_configs.py)")
                        add(f"{stem}_pot_{tag}_{sname}", comment, s, params, plane, {prod: 10.0 ** (P_RANGE[0] - 2)})
                s = copy.deepcopy(solver)
                s["dark_portal"].update(epsilon_prime=0.0)
                params = {"m_Zd_eV": {"start": 2e6, "bounds": [10.0 ** (M_MEV[0] - 1), 10.0 ** (M_MEV[1] + 1)],
                                      "space": "log10"},
                          mev_axis: {"start": 1e-6, "bounds": [10.0 ** (G_MEV[0] - 2), 10.0 ** (G_MEV[1] + 1)],
                                     "space": "log10"}}
                params.update(_common_params(base_params, 7, spectrum))
                plane = {"x": "m_Zd_eV", "y": mev_axis, "x_range": [10.0 ** M_MEV[0], 10.0 ** M_MEV[1]],
                         "y_range": [10.0 ** G_MEV[0], 10.0 ** G_MEV[1]], "nx": M_MEV[2], "ny": G_MEV[2],
                         "raster": "m_Zd_eV"}
                add(f"{stem}_mev", f"{model}{'' if tex is None else ', texture ' + tex}: MeV absorption at eps' = 0",
                    s, params, plane, {mev_axis: 10.0 ** (G_MEV[0] - 2)})
            else:
                s = copy.deepcopy(solver)
                s["dark_portal"].update(epsilon_prime=0.0)
                for tag, (m_lo, m_hi, nx), (g_lo, g_hi, ny), nodes in (
                        ("ul", (M_POT[0], M_POT[1], M_POT[1] - M_POT[0] + 1), (G_DIRECT[0], G_DIRECT[1],
                                                                               2 * (G_DIRECT[1] - G_DIRECT[0]) + 1), 3),
                        ("mev", M_MEV, G_MEV, 7)):
                    params = {"m_Zd_eV": {"start": 10.0 ** ((m_lo + m_hi) / 2), "bounds": [10.0 ** (m_lo - 1),
                                                                                         10.0 ** (m_hi + 1)],
                                          "space": "log10"},
                              "g_D": {"start": 10.0 ** ((g_lo + g_hi) / 2), "bounds": [10.0 ** (g_lo - 2),
                                                                                    10.0 ** (g_hi + 1)],
                                      "space": "log10"}}
                    params.update(_common_params(base_params, nodes, spectrum))
                    plane = {"x": "m_Zd_eV", "y": "g_D", "x_range": [10.0 ** m_lo, 10.0 ** m_hi],
                             "y_range": [10.0 ** g_lo, 10.0 ** g_hi], "nx": nx, "ny": ny, "raster": "m_Zd_eV"}
                    add(f"{stem}_{tag}", f"{model}: the gauge coupling g' over {tag} masses", s, params, plane,
                        {"g_D": 10.0 ** (g_lo - 2)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spectrum", choices=sorted(SPECTRA), default="bpl")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every data set of the spectrum")
    ap.add_argument("--check", action="store_true", help="load every configuration and evaluate its model once")
    ap.add_argument("--mixing", choices=("fixed", "vary", "exact"), default="fixed",
                    help="'vary': the mixing varied through templates (needs --spectrum bpl; campaign_mix); 'exact': "
                         "the mixing and the lightest mass profiled exactly (analysis/exact_profile.py; campaign_exact)")
    a = ap.parse_args(argv)
    from analysis import config as config_mod
    tree, sets = SPECTRA[a.spectrum]
    if a.mixing in ("vary", "exact"):
        if a.spectrum != "bpl":
            raise SystemExit(f"--mixing {a.mixing} needs --spectrum bpl")
        tree = "campaign_mix" if a.mixing == "vary" else "campaign_exact"
    for ds in a.datasets or list(sets):
        if ds not in sets:
            raise SystemExit(f"data set {ds!r} does not exist with --spectrum {a.spectrum}; known: {sorted(sets)}")
        d = os.path.join(REPO, "analysis", "configs", tree, ds)
        os.makedirs(d, exist_ok=True)
        rows = planes_for(ds, a.spectrum, a.mixing)
        for name, cfg, _ in rows:
            with open(os.path.join(d, name + ".json"), "w") as f:
                json.dump(cfg, f, indent=1)
            config_mod.load(os.path.join(d, name + ".json"))          # validates every key
        with open(os.path.join(d, "campaign.json"), "w") as f:
            json.dump({"dataset": ds, "planes": [{"name": n, "points": k} for n, _, k in rows],
                       "total_points": int(sum(k for _, _, k in rows))}, f, indent=1)
        print(f"{ds}: {len(rows)} planes, {sum(k for _, _, k in rows)} points -> {d}")
        if a.check:
            for name, cfg, _ in rows:
                c = config_mod.load(os.path.join(d, name + ".json"))
                pl = cfg["planes"][0]
                v = {pl["x"]: pl["x_range"][1], pl["y"]: pl["y_range"][0]}
                c.model()._overrides(v)


if __name__ == "__main__":
    main()
