"""
dirac_relic_polarization.py -- what the relics' helicity does to the secret-interaction fluxes (2026-10-21). A Dirac
relic keeps its helicity, so the beam meets its left- and right-chiral components in shares (1 +- pi)/2 that depend on
its speed and direction (dp_si.model_spec, polarized). For the Dirac scalar and vector (texture e) at the two
validation points of tools/thermal_reference_checks.py and two lightest masses: the flux at Earth summed over flavors,
over the free-streaming one, with the relics at rest, in motion ignoring their helicity (each chiral share rho/2), and
in motion with it (secret_interactions.thermal_products). Writes output/analysis/dirac_polarization.npz; --plot draws
output/figures/dirac_polarization.png from it.

  python tools/dirac_relic_polarization.py [--workers N]
  python tools/dirac_relic_polarization.py --plot
"""
import argparse
import copy
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = {"scalar": "analysis/configs/pion_2050_case_5_si_dirac_e.json",
           "vector": "analysis/configs/pion_2050_case_5_si_vector_dirac_e.json"}
MG = [(3.16e4, 2e-3), (4.22e6, 1.35e-3)]
MLS = (1e-6, 0.03)
MODES = ("free", "rest", "unpolarized", "polarized")
OUT = os.path.join(REPO, "output", "analysis", "dirac_polarization.npz")


def flux(args):
    """(mediator, point index, m_lightest, mode, E, flux summed over flavors)."""
    med, ip, m_l, mode = args
    os.environ["NUMBA_NUM_THREADS"] = "2"
    os.environ.pop("NULRI_FLUX_CACHE", None)
    sys.path[:0] = [os.path.join(REPO, "src"), REPO]
    import dark_portal_solver as dps
    import dp_si
    from analysis import config as config_mod
    if mode == "unpolarized":
        spec = dp_si.model_spec
        dp_si.model_spec = lambda *a, **k: spec(*a, **dict(k, polarized=False))
    path = os.path.join(REPO, CONFIGS[med])
    raw = copy.deepcopy(json.load(open(path)))
    raw["solver"].setdefault("cosmology", {})["m_lightest_eV"] = m_l
    raw["solver"]["secret_interactions"]["thermal_products"] = mode in ("unpolarized", "polarized")
    raw["parameters"].pop("m_lightest_eV", None)
    raw.pop("exact_profile", None)
    cfg = config_mod.from_dict(raw, path)
    m = cfg.model()
    v = {p.name: p.start for p in cfg.parameters}
    pl = raw["planes"][0]
    M, g = MG[ip]
    v.update({pl["x"]: M, pl["y"]: 1e-12 if mode == "free" else g})
    E, params = dps.build_solver_params(m._run_cfg(m._overrides(v)))
    F = np.asarray(dps.propagate_weighted(E, params))
    F = F.sum(axis=tuple(range(1, F.ndim))) if F.ndim > 1 else F
    return med, ip, m_l, mode, np.asarray(E).tolist(), F.tolist()


def plot():
    sys.path[:0] = [REPO]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis import figure_style as fs
    fs.use()
    z = np.load(OUT)
    style = {"rest": ("#9b2226", (0, (4, 2)), "relics at rest"),
             "unpolarized": ("#ee9b00", (0, (1, 1.5)), "in motion, helicity ignored"),
             "polarized": ("#0a9396", "-", "in motion, helicity kept")}
    fig, axes = plt.subplots(2, 2, figsize=(2.3 * fs.PANEL[0], 1.9 * fs.PANEL[1]), sharex=True, sharey=True)
    for i, med in enumerate(("scalar", "vector")):
        for j, (M, g) in enumerate(MG):
            ax = axes[i, j]
            for m_l, lw in zip(MLS, (1.8, 1.0)):
                E = z[f"{med}_{j}_{m_l:g}_free"][0]
                F0 = z[f"{med}_{j}_{m_l:g}_free"][1]
                for mode, (c, ls, lab) in style.items():
                    F = z[f"{med}_{j}_{m_l:g}_{mode}"][1]
                    ax.plot(np.log10(E), F / F0, color=c, ls=ls, lw=lw,
                            label=f"{lab}" if m_l == MLS[0] else None)
            ax.set_ylim(0, 1.2)
            mlab = f"{M / 1e6:.3g} MeV" if M >= 1e6 else f"{M / 1e3:.3g} keV"
            e = int(np.floor(np.log10(g)))
            ax.set_title(f"Dirac {med}, $M$ = {mlab}, $g = {g / 10 ** e:.3g}\\times 10^{{{e}}}$", fontsize=17)
    fig.supxlabel(r"Neutrino energy, $\log_{10}(E/\mathrm{eV})$")
    fig.supylabel("Flux / flux without interaction")
    axes[1, 0].legend(loc="lower right", fontsize=14, title=r"thick: $m_1 = 10^{-6}$ eV, thin: $0.03$ eV",
                      title_fontsize=14, frameon=True, edgecolor="black")
    fig.tight_layout()
    path = os.path.join(REPO, "output", "figures", "dirac_polarization.png")
    fig.savefig(path, dpi=150)
    print(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    if a.plot:
        plot()
        return 0
    tasks = [(med, ip, m_l, mode) for med in CONFIGS for ip in range(len(MG)) for m_l in MLS for mode in MODES]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(flux, tasks, chunksize=1)
    np.savez(OUT, **{f"{med}_{ip}_{m_l:g}_{mode}": np.vstack([E, F]) for med, ip, m_l, mode, E, F in res})
    got = {(med, ip, m_l, mode): np.array(F) for med, ip, m_l, mode, E, F in res}
    for med in CONFIGS:
        for ip, (M, g) in enumerate(MG):
            for m_l in MLS:
                F0 = got[(med, ip, m_l, "free")]
                pol, unp, rest = (got[(med, ip, m_l, k)] for k in ("polarized", "unpolarized", "rest"))
                ok = F0 > 0
                print(f"{med} M={M:g} g={g:g} m_l={m_l:g}: max |pol - unpol| / free = "
                      f"{np.max(np.abs(pol - unp)[ok] / F0[ok]):.3f}, max |pol - rest| / free = "
                      f"{np.max(np.abs(pol - rest)[ok] / F0[ok]):.3f}")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
