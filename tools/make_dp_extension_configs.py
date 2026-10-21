"""
make_dp_extension_configs.py -- the dark portal beyond the campaign of 2026-10-20 (Mauricio, 2026-10-21), in
analysis/configs/campaign_ml/<data set>/, as the campaign's (mixing fixed at Table II's centre, m_lightest profiled
exactly, the relics thermal in every collision; outputs under output/analysis/campaign_ml/):
  dp_<flavor>_ul[_fx] -- the MeV planes' physics (eps' = 0: no coupling to matter, so no long-range potential; the
      scattering and absorption on relic neutrinos) at ultralight masses, 1e-30 - 1e4 eV, one mass per decade, the
      coupling (the texture's trace g_D Sum|U|^2) 1e-12 - 1, four per decade; the MeV planes' lightest-mass scan. A
      limit that does not depend on eps', drawn with the MeV planes' in one panel per flavor.
  dp_<flavor>_pot_x<x>_<sign>[_fx] -- the long-range potential at a fixed neutrino-side coupling x = g_D Sum|U|^2
      (1e-24: no relic screening; 1e-16: the Sun and the Galaxy screened; 1e-8: every local body screened), both signs,
      over the whole ultralight range 1e-30 - 1e4 eV (tools/make_campaign_configs.py's split planes, there cut where
      the screening stops mattering): the product P = x eps' from 1e-60 to x, i.e. eps' up to 1 -- the plane of mass
      against eps'.
Flavors e, mu, tau (the sterile neutrino mixing with that flavor alone).
  python tools/make_dp_extension_configs.py [--datasets now_ic26 now_events proj_future]
"""
import argparse
import copy
import json
import os
import re
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "tools")]

import make_campaign_configs as mcc  # noqa: E402

FLAVORS = ("e", "mu", "tau")
UL_MASS = (-30, 4, 35)            # log10 m_Zd, one per decade
UL_COUPLING = (-12, 0, 49)        # log10 g_D Sum|U|^2, four per decade
POT_SPLITS = (1e-24, 1e-16, 1e-8)


def ul_config(mev_cfg: dict) -> dict:
    cfg = copy.deepcopy(mev_cfg)
    old, new = cfg["name"], cfg["name"].replace("_mev", "_ul")
    cfg["name"] = new
    cfg["output_dir"] = cfg["output_dir"].replace(old, new)
    cfg["_comment"] = cfg["_comment"].replace("MeV absorption at eps' = 0",
                                              "eps' = 0 at ultralight masses (scattering and absorption on the "
                                              "relics; no potential)")
    y = cfg["planes"][0]["y"]
    cfg["parameters"]["m_Zd_eV"] = {"start": 1e-13, "bounds": [10.0 ** (UL_MASS[0] - 1), 10.0 ** (UL_MASS[1] + 1)],
                                    "space": "log10"}
    cfg["parameters"][y] = {"start": 1e-6, "bounds": [10.0 ** (UL_COUPLING[0] - 2), 10.0 ** (UL_COUPLING[1] + 1)],
                            "space": "log10"}
    cfg["planes"] = [{"x": "m_Zd_eV", "y": y, "x_range": [10.0 ** UL_MASS[0], 10.0 ** UL_MASS[1]],
                      "y_range": [10.0 ** UL_COUPLING[0], 10.0 ** UL_COUPLING[1]], "nx": UL_MASS[2],
                      "ny": UL_COUPLING[2], "raster": "m_Zd_eV"}]
    cfg["asimov"][y] = 10.0 ** (UL_COUPLING[0] - 2)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["now_ic26", "now_events", "proj_future"])
    a = ap.parse_args(argv)
    from analysis import config as config_mod
    mcc.SPLITS = POT_SPLITS
    mcc.SPLIT_M_HI = UL_MASS[1]
    mcc.ML_KEEP = re.compile(r"^dp_(e|mu|tau)_pot_x1m(24|16|08)_(pos|neg)$")
    for ds in a.datasets:
        d = os.path.join(REPO, "analysis", "configs", "campaign_ml", ds)
        rows = []
        for fl in FLAVORS:
            for sfx in ("", "_fx"):
                mev = json.load(open(os.path.join(d, f"dp_{fl}_mev{sfx}.json")))
                cfg = ul_config(mev)
                rows.append((cfg["name"], cfg))
        rows += [(n, c) for n, c, _ in mcc.planes_for(ds, "bpl", "lightest")]
        for name, cfg in rows:
            path = os.path.join(d, name + ".json")
            with open(path, "w") as f:
                json.dump(cfg, f, indent=1)
            config_mod.load(path)                       # validates every key
        n = sum(c["planes"][0]["nx"] * c["planes"][0]["ny"] for _, c in rows)
        print(f"{ds}: {len(rows)} new planes, {n} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
