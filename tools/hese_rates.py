"""
hese_rates.py -- task 1's first deliverable (docs/dev/HESE_RATES_PLAN.md, steps 4-5): expected HESE events from the
release's Monte Carlo (IceCubeRates.starting_events.hese; reduced by tools/hese_reduce.py, detector systematics by
tools/hese_detector_systematics.py) with our atmospheric background, against the 60 events above 60 TeV.

  (a) the release's best-fit astrophysical flux (6.37e-18 for six species at 100 TeV, gamma 2.87): the per-morphology
      expectation (39.51 cascades, 8.43 tracks, 2.21 double cascades: step 8's check);
  (b) the atmospheric neutrinos from our MCEq table (SIBYLL-2.3d x H3a, the South Pole) with the release's self-veto,
      and the release's own model, at nominal; the muons; the data per morphology, deposited energy and zenith;
      the detector systematics' 1-sigma bands;
  (c) standard physics, a dark-portal point and a secret-interaction point with the Galactic component
      (propagate_sky at N_side 8, 20 declination bands; the extragalactic flux at the HESE fit, for illustration).

  python -I tools/hese_rates.py --mc HESE_MC.npz --detector HESE_DETSYS.npz [--atm ATM.npz] [--skip-solver]

Writes output/hese/hese_rates.json and output/figures/hese_rates.png, hese_cases.png.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from IceCubeRates.starting_events import hese  # noqa: E402

HESE_FIT = dict(per_type=6.37e-18 / 6.0, gamma=2.87)             # PRD 104, 022002, Eq. VI.1
GU2_PER_KAPPA = 1.66e-54
CASES = {
    "standard": dict(dp={"g_D": 1e-60, "epsilon_prime": 0.0}),
    "dark portal": dict(dp={"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "epsilon_prime": 0.1,
                            "g_D": 300.0 * GU2_PER_KAPPA / (0.1 * 0.5)}),
    "secret interactions": dict(si={"enabled": True, "mediator": "scalar", "texture": "tau", "m_phi_eV": 5e6,
                                    "g_phi": 0.05}),
}


def sky(dp=None, si=None):
    import dark_portal_solver as dps
    cfg = {"source": {"p_index": HESE_FIT["gamma"]},
           "grid": {"n_E_bins": 150, "E_min_eV": 3e13, "E_max_eV": 1e17, "n_z_bins": 400},
           "sky": {"output": "bands", "nside": 8, "extragalactic_norm": 2.0 * HESE_FIT["per_type"],
                   "extragalactic_norm_E_GeV": 1e5},
           "galactic": {"enabled": True, "template": "cringe", "emission": "3d"},
           "execution": {"verbose": 0}}
    if dp is not None:
        cfg["dark_portal"] = dp
    if si is not None:
        cfg["secret_interactions"] = si
    E, params = dps.build_solver_params(cfg)
    params["show_progress"] = False
    return dps.propagate_sky(E, params)


def total(h):
    return {m: float(h[m].sum()) for m in hese.MORPHOLOGIES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--atm", default=None)
    ap.add_argument("--skip-solver", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(REPO, "output"))
    a = ap.parse_args()
    t0 = time.time()
    ev = hese.HESEEvents(a.mc)
    ev.load_detector(a.detector)
    res = {}
    # (a) the release's astrophysical best fit
    astro = ev.expected(hese.power_law_flux(HESE_FIT["per_type"], HESE_FIT["gamma"]))
    H = {"astro": ev.histogram(astro)}
    res["astro_release_fit"] = total(H["astro"])
    # (b) backgrounds and data
    tpl = ev.atmospheric_templates(a.atm)
    rel = ev.release_atmospheric()
    H["conv"], H["prompt"] = ev.histogram(tpl["conv"]), ev.histogram(tpl["prompt"])
    H["conv_release"], H["prompt_release"] = ev.histogram(rel["conv"]), ev.histogram(rel["prompt"])
    H["muons"] = ev.histogram(ev.muons())
    H["data"] = ev.data()
    for k in ("conv", "prompt", "conv_release", "prompt_release", "muons", "data"):
        res[k] = total(H[k])
    # 1-sigma shifts, each fully correlated across bins: summed over bins first in any projection, then the
    # systematics in quadrature. Atmospheric: Barr, cosmic ray, prompt; detector: the release's three, all components.
    from IceCubeRates.flux_adapters import ATM_PARAMS, Atmospheric
    pri = Atmospheric(path=a.atm).priors() if a.atm else Atmospheric().priors()
    shifts = {"atm": {}, "det": {}}
    for p in ATM_PARAMS:
        shifts["atm"][p] = ev.histogram(pri[p] * (tpl["prompt"] if p == "prompt" else tpl[p]))
    for name, sig in ev.det_sigma.items():
        per = np.zeros(len(astro))
        for comp, w in (("Astro", astro), ("Conv", tpl["conv"]), ("Prompt", tpl["prompt"])):
            per += w * (ev.detector_factor(comp, {name: ev.det_nominal[name] + sig}) - 1.0)
        shifts["det"][name] = ev.histogram(per)

    def band(kind, m, axis=None):
        """1-sigma band of morphology m: the projection (sum over axis; None = total) of each shift, in quadrature."""
        return np.sqrt(sum((h[m].sum() if axis is None else h[m].sum(axis)) ** 2 for h in shifts[kind].values()))
    nominal = {m: H["astro"][m] + H["conv"][m] + H["prompt"][m] + H["muons"][m] for m in hese.MORPHOLOGIES}
    res["nominal_total"] = total(nominal)
    res["atmospheric_band"] = {m: float(band("atm", m)) for m in hese.MORPHOLOGIES}
    res["detector_band"] = {m: float(band("det", m)) for m in hese.MORPHOLOGIES}
    res["detector_shifts"] = {k: {m: float(h[m].sum()) for m in hese.MORPHOLOGIES} for k, h in shifts["det"].items()}
    print("expected (HESE fit astro + MCEq atmospheric with the release's veto + muons):",
          {m: round(v, 2) for m, v in res["nominal_total"].items()}, "| data:", res["data"], flush=True)
    print("atmospheric conventional / prompt, ours:", res["conv"], res["prompt"], "| release's:", res["conv_release"],
          res["prompt_release"], flush=True)
    # (c) physics cases
    cases = {}
    if not a.skip_solver:
        for name, kw in CASES.items():
            t1 = time.time()
            s = sky(**kw)
            eg = ev.histogram(ev.expected(hese.sky_flux(s, galactic=False)))
            ga = ev.histogram(ev.expected(hese.sky_flux(s, extragalactic=False)))
            cases[name] = (eg, ga)
            res.setdefault("cases", {})[name] = dict(extragalactic=total(eg), galactic=total(ga), seconds=time.time() - t1)
            print(f"{name}: extragalactic {total(eg)}, Galactic {total(ga)} ({time.time() - t1:.0f} s)", flush=True)
    res["seconds"] = time.time() - t0
    res["commit"] = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()
    for d in ("hese", "figures"):
        os.makedirs(os.path.join(a.outdir, d), exist_ok=True)
    with open(os.path.join(a.outdir, "hese", "hese_rates.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Eb = ev.E_bins
    xE = np.sqrt(Eb[1:] * Eb[:-1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, m in enumerate(hese.MORPHOLOGIES):
        ax = axes[0, col]
        e = lambda k: H[k][m].sum(0)  # noqa: E731
        nomE = nominal[m].sum(0)
        bandE = np.hypot(band("atm", m, 0), band("det", m, 0))
        ax.stairs(nomE, Eb, color="k", lw=1.4, label="expected (nominal)")
        ax.stairs(nomE + bandE, Eb, baseline=np.maximum(nomE - bandE, 0), fill=True, color="0.6", alpha=0.45,
                  label=r"atmospheric + detector 1$\sigma$")
        ax.stairs(e("astro"), Eb, color="C3", label="astrophysical (HESE fit)")
        ax.stairs(e("conv"), Eb, color="C0", label="conventional (MCEq, release veto)")
        ax.stairs(e("conv_release"), Eb, color="C0", ls=":", label="conventional (release's model)")
        ax.stairs(e("prompt"), Eb, color="C1", label="prompt (MCEq)")
        ax.stairs(e("muons"), Eb, color="C4", label="muons (release)")
        ax.errorbar(xE, e("data"), yerr=np.sqrt(e("data")), fmt="o", color="k", ms=3, lw=0.8, label="data")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-2, 50)
        ax.set_xlabel("deposited energy (GeV)"); ax.set_ylabel("events per bin (2635 days)")
        ax.set_title(f"HESE {m}", fontsize=10)
        if col == 0:
            ax.legend(fontsize=6.5)
    for col, m in enumerate(("cascades", "tracks")):
        ax = axes[1, col]
        zb = ev.zen_bins
        cz = np.cos(zb)
        z = lambda k: H[k][m].sum(1)  # noqa: E731
        ax.stairs(nominal[m].sum(1)[::-1], cz[::-1], color="k", label="expected (nominal)")
        ax.stairs(z("astro")[::-1], cz[::-1], color="C3", label="astrophysical")
        ax.stairs((z("conv") + z("prompt"))[::-1], cz[::-1], color="C0", label="atmospheric (MCEq)")
        d = z("data")[::-1]
        ax.errorbar(0.5 * (cz[::-1][1:] + cz[::-1][:-1]), d, yerr=np.sqrt(d), fmt="o", color="k", ms=3, lw=0.8,
                    label="data")
        ax.set_xlabel(r"cos(reconstructed zenith)  ($-1$: up-going)"); ax.set_ylabel("events")
        ax.set_title(f"{m}: zenith", fontsize=10); ax.legend(fontsize=7)
    ax = axes[1, 2]
    ax.axis("off")
    txt = [f"expected: {sum(res['nominal_total'].values()):.1f} events (data: {sum(res['data'].values()):.0f})"]
    for m in hese.MORPHOLOGIES:
        txt.append(f"  {m}: {res['nominal_total'][m]:.1f} +- {np.hypot(res['atmospheric_band'][m], res['detector_band'][m]):.1f}"
                   f" (data {res['data'][m]:.0f})")
    txt.append("astrophysical: the release's fit (6.37, 2.87)")
    txt.append(f"conventional: ours {sum(res['conv'].values()):.2f}, release {sum(res['conv_release'].values()):.2f}")
    txt.append(f"prompt: ours {sum(res['prompt'].values()):.2f}, release {sum(res['prompt_release'].values()):.2f}")
    ax.text(0.0, 0.95, "\n".join(txt), va="top", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "figures", "hese_rates.png"), dpi=110)
    plt.close(fig)
    if cases:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        cols = dict(zip(CASES, ("k", "C3", "C0")))
        for ax, m in zip(axes, hese.MORPHOLOGIES):
            for name, (eg, ga) in cases.items():
                ax.stairs(eg[m].sum(0), Eb, color=cols[name], label=f"{name}: extragalactic")
                ax.stairs(ga[m].sum(0), Eb, color=cols[name], ls=":", label=f"{name}: Galactic")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 30)
            ax.set_xlabel("deposited energy (GeV)"); ax.set_ylabel("astrophysical events per bin")
            ax.set_title(f"HESE {m}", fontsize=10)
        axes[0].legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, "figures", "hese_cases.png"), dpi=110)
        plt.close(fig)
    print(f"wrote {a.outdir}/hese/hese_rates.json and the figures ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
