"""
si_flavor_paths.py -- how secret interactions move the energy-averaged flavor composition at the
detector: (f_e, f_mu) along a coupling ladder at a few mediator masses, for the four coupling
textures, with the physics of the pion_2050_case_5_si_* planes (their solver configuration). Also
the all-flavor spectra at one coupling per mass. Writes output/si_planes/flavor_paths.json; run on
Tycho (a few hundred solver calls):

  python tools/si_flavor_paths.py
"""
import copy
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import dark_portal_solver as dps  # noqa: E402
from flux_postprocessing import compute_energy_averaged_ratios  # noqa: E402

MASSES_EV = [1e5, 1e6, 5e6, 3e7]
COUPLINGS = list(np.logspace(-3, 0, 13))
SPECTRUM_G = 0.1


def main():
    out = dict(masses_eV=MASSES_EV, couplings=COUPLINGS, spectrum_g=SPECTRUM_G, textures={})
    for tex in ["e", "mu", "tau", "universal"]:
        cfg = json.load(open(os.path.join(REPO, "analysis", "configs", f"pion_2050_case_5_si_{tex}.json")))
        solver = copy.deepcopy(cfg["solver"])
        solver["execution"] = {"n_cores": 1, "verbose": 0}
        rec = dict(paths={}, spectra={})
        for M in MASSES_EV:
            path = []
            for g in [0.0] + COUPLINGS:
                s = copy.deepcopy(solver)
                s["secret_interactions"].update(m_phi_eV=M, g_phi=float(g))
                E, params = dps.build_solver_params(s)
                params["show_progress"] = False
                F = dps.propagate_weighted(E, params)
                path.append(compute_energy_averaged_ratios(E, F).tolist())
                if g == SPECTRUM_G or g == 0.0:
                    rec["spectra"].setdefault(str(M), {})["free" if g == 0.0 else "si"] = F.sum(1).tolist()
            rec["paths"][str(M)] = path
            print(tex, M, path[-1], flush=True)
        rec["E_eV"] = E.tolist()
        out["textures"][tex] = rec
    out.update(host=os.uname().nodename, commit=os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
               date=time.strftime("%Y-%m-%d %H:%M"))
    os.makedirs(os.path.join(REPO, "output", "si_planes"), exist_ok=True)
    p = os.path.join(REPO, "output", "si_planes", "flavor_paths.json")
    json.dump(out, open(p, "w"))
    print("wrote", p)


if __name__ == "__main__":
    main()
