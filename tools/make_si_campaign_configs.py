"""
make_si_campaign_configs.py -- the secret-interaction (SI) plane campaign over the whole mediator mass range,
1e-30 eV - 1 GeV, as the dark-portal and lepton-number planes (2026-10-25; notes, "Secret interactions below 1 keV").

Writes analysis/configs/campaign_si/<data set>/<name>.json (outputs in output/analysis/campaign_si/<data set>/<name>)
and analysis/configs/campaign_si/<data set>/campaign.json (each plane's points, for packing on Tycho).

Every configuration is the data set's dark-portal MeV plane of the campaign (analysis/configs/campaign_ml/<data
set>/dp_mu_mev[_fx].json: its data, grid, broken power law, the mixing fixed at Table II's centre, the lightest mass
profiled exactly, the relics' thermal products) with the secret interactions in place of the dark portal, as the
SI campaign's first planes (campaign_si/now_ic26/si_*_mu.json, which this reproduces). Per data set (now_ic26,
now_events, proj_future), model (scalar or vector mediator x Majorana or Dirac neutrinos), texture (e, mu, tau,
universal) and composition (pion; _fx: the sources' compositions fitted):
  si_<mediator>_<type>_<texture>      1 keV - 1 GeV, g 1e-6 - 1, 49 x 49 (the first planes' grid: the resonances)
  si_<mediator>_<type>_<texture>_ul   1e-30 eV - 1 keV, g 1e-38 - 1, one mass per decade (34) and two couplings per
                                      decade (77), as the lepton-number planes' spacing (technical); the couplings
                                      reach 1e-38 because the vector's decays act down to g/M ~ 1e-6 eV^-1 (1e-36 at
                                      1e-30 eV) and every column must reach the region without an effect; the
                                      scalar's on the same grid, for one figure layout (its region above its
                                      relic-field line is not modelled and not fitted, Mauricio 2026-10-25)
The sub-keV planes take two redshift substeps per energy bin (secret_interactions.z_substeps 2: where the decays and
the conversions cycle, as the dark-portal model's runs through dp_si below 1 keV) and no scan of the lightest mass
(exact_profile {}, as the lepton-number ultralight planes).

  python tools/make_si_campaign_configs.py [--datasets now_ic26 now_events proj_future] [--check]
"""
import argparse
import copy
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

DATASETS = ("now_ic26", "now_events", "proj_future")
MODELS = (("scalar", "majorana"), ("scalar", "dirac"), ("vector", "majorana"), ("vector", "dirac"))
TEXTURES = ("e", "mu", "tau", "universal")
RANGES = {
    "kev": dict(x_range=[1e3, 1e9], y_range=[1e-6, 1.0], nx=49, ny=49, m_bounds=[100.0, 1e10], g_bounds=[1e-7, 3.0],
                m_start=5e6, g_start=0.01, asimov_g=1e-7, z_substeps=1),
    "ul": dict(x_range=[1e-30, 1e3], y_range=[1e-38, 1.0], nx=34, ny=77, m_bounds=[1e-31, 1e4], g_bounds=[1e-40, 3.0],
               m_start=1e-13, g_start=1e-20, asimov_g=1e-40, z_substeps=2),
}
SI = {"enabled": True, "channels": "full", "phiphi_coupling": "coherent", "thermal_products": True,
      "dirac_sterile": "track", "E_extension": 10.0}


def si_config(base: dict, dataset: str, mediator: str, ntype: str, texture: str, rng: str, fx: bool) -> dict:
    """The SI configuration from the data set's dark-portal MeV configuration base (dp_mu_mev[_fx])."""
    r = RANGES[rng]
    name = f"si_{mediator}_{ntype}_{texture}" + ("_ul" if rng == "ul" else "") + ("_fx" if fx else "")
    cfg = copy.deepcopy(base)
    solver = cfg["solver"]
    solver.pop("dark_portal", None)
    solver.pop("model", None)
    solver["secret_interactions"] = dict(SI, mediator=mediator, neutrino_type=ntype, texture=texture,
                                         z_substeps=r["z_substeps"])
    params = cfg["parameters"]
    for k in ("m_Zd_eV", "g_D_trace"):
        params.pop(k, None)
    new = {"m_phi_eV": {"start": r["m_start"], "bounds": list(r["m_bounds"]), "space": "log10"},
           "g_phi": {"start": r["g_start"], "bounds": list(r["g_bounds"]), "space": "log10"}}
    cfg["parameters"] = {**new, **params}
    cfg["planes"] = [{"x": "m_phi_eV", "y": "g_phi", "x_range": list(r["x_range"]), "y_range": list(r["y_range"]),
                      "nx": r["nx"], "ny": r["ny"], "raster": "m_phi_eV"}]
    asimov = {k: v for k, v in cfg.get("asimov", {}).items() if k != "g_D_trace"}
    asimov["g_phi"] = r["asimov_g"]
    cfg["asimov"] = asimov
    if rng == "ul":
        cfg["exact_profile"] = {}
    cfg["name"] = name
    cfg["output_dir"] = f"output/analysis/campaign_si/{dataset}/{name}"
    what = "1 keV - 1 GeV" if rng == "kev" else "1e-30 eV - 1 keV (notes, 'Secret interactions below 1 keV')"
    cfg["_comment"] = (f"SI campaign (2026-10-25; tools/make_si_campaign_configs.py): the {dataset} data set's "
                       f"settings (campaign_ml dp_mu_mev{'_fx' if fx else ''}.json) with the secret interactions in "
                       f"place of the dark portal: {mediator} mediator, {ntype} neutrinos, texture {texture}, {what}; "
                       f"relics' thermal products on"
                       + ("; the sources' flavor compositions fitted" if fx else ""))
    return cfg


def planes_for(dataset: str):
    out = []
    for fx in (False, True):
        with open(os.path.join(REPO, "analysis", "configs", "campaign_ml", dataset,
                               "dp_mu_mev" + ("_fx" if fx else "") + ".json")) as f:
            base = json.load(f)
        for mediator, ntype in MODELS:
            for tex in TEXTURES:
                for rng in ("kev", "ul"):
                    cfg = si_config(base, dataset, mediator, ntype, tex, rng, fx)
                    pl = cfg["planes"][0]
                    out.append((cfg["name"], cfg, pl["nx"] * pl["ny"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--check", action="store_true", help="load every configuration and evaluate its overrides once")
    a = ap.parse_args(argv)
    from analysis import config as config_mod
    for ds in a.datasets:
        d = os.path.join(REPO, "analysis", "configs", "campaign_si", ds)
        os.makedirs(d, exist_ok=True)
        rows = planes_for(ds)
        for name, cfg, _ in rows:
            p = os.path.join(d, name + ".json")
            with open(p, "w") as f:
                json.dump(cfg, f, indent=1)
            config_mod.load(p)                                  # validates every key
        with open(os.path.join(d, "campaign.json"), "w") as f:
            json.dump({"dataset": ds, "planes": [{"name": n, "points": k} for n, _, k in rows],
                       "total_points": int(sum(k for _, _, k in rows))}, f, indent=1)
        print(f"{ds}: {len(rows)} planes, {sum(k for _, _, k in rows)} points -> {d}")
        if a.check:
            for name, cfg, _ in rows:
                c = config_mod.load(os.path.join(d, name + ".json"))
                pl = cfg["planes"][0]
                c.model()._overrides({pl["x"]: pl["x_range"][1], pl["y"]: pl["y_range"][0]})


if __name__ == "__main__":
    main()
