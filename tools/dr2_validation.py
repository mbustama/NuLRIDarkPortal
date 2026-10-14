"""
dr2_validation.py -- task 3's checks and first results (docs/dev/DR2_TRACKS_PLAN.md, steps 5-6): through-going
muon tracks in IceCube from the IceTracks-DR2 responses (IceCubeRates.northern_muons.dr2, reduced by
tools/dr2_reduce.py), our atmospheric background (IceCubeRates/data/atmospheric_numu_southpole.npz, from
tools/atmospheric_background.py) and the solver's per-direction fluxes.

  (a) the responses: each smearing slice sums to 1 where simulated; livetimes; effective areas at 100 TeV;
  (b) the units check: tracks reconstructed at dec > -5 deg per configuration, in reconstructed energy and
      declination, for the atmospheric background (daemonflux conventional, with its 1-sigma error, plus MCEq
      prompt, SIBYLL-2.3d) plus IceCube's 9.5-yr astrophysical fit (ApJ 928, 50 (2022): nu_mu + nubar_mu 1.44e-18
      GeV^-1 cm^-2 s^-1 sr^-1 at 100 TeV, gamma 2.37), against DR2's events;
  (c) astrophysical tracks for standard physics, a dark-portal point and a secret-interaction point (propagate_sky
      on the effective area's declination bins, E 1 TeV-100 PeV): the extragalactic flux normalized to that fit
      (its nu_mu + nubar_mu without new physics equals the fit: checked) plus the Galactic component (CRINGE x
      0.91, emission in 3-D, N_side 64 data from --galactic-data).

  python -I tools/dr2_validation.py --dr2 /lustre/astro/mbustama/external/icetracks_dr2 \\
      --galactic-data /lustre/astro/mbustama/external/galactic_nside64      (Tycho; --skip-solver for (a)-(b);
      --sky-cache DIR keeps the solver's skies, --sky-only computes them alone)

Writes output/tracks/dr2_validation.json, output/figures/dr2_validation.png ((a)-(b)) and
output/figures/dr2_tracks.png ((c)).
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from IceCubeRates import flux_adapters as FA  # noqa: E402
from IceCubeRates.northern_muons.dr2 import DR2Tracks  # noqa: E402

ASTRO_9P5 = dict(per_flavor=1.44e-18, gamma=2.37, pivot_GeV=1e5)
GU2_PER_KAPPA = 1.66e-54                       # as tools/sky_validation.py: g_D = kappa x this / (eps' u)
DEC_CUT_DEG = -5.0
# reconstructed declination bins: the southern sky in one bin, then 19 bins uniform in sin(dec) above -5 deg
RECO_DEC_DEG = np.concatenate([[-90.0], np.degrees(np.arcsin(np.linspace(np.sin(np.radians(DEC_CUT_DEG)), 1.0, 20)))])
RECO_DEC_DEG[-1] = 90.0
CASES = {
    "standard": dict(dp={"g_D": 1e-60, "epsilon_prime": 0.0}),
    "dark portal": dict(dp={"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": 1e-24, "epsilon_prime": 0.1,
                            "g_D": 300.0 * GU2_PER_KAPPA / (0.1 * 0.5)}),
    "secret interactions": dict(si={"enabled": True, "mediator": "scalar", "texture": "tau", "m_phi_eV": 5e6,
                                    "g_phi": 0.05}),
}
CASE_LABELS = {"standard": "standard", "dark portal": r"dark portal ($m_{Z_D}$ = 1e-24 eV, $\kappa$ = 300, e-$\mu$)",
               "secret interactions": r"secret int. (scalar, $\nu_\tau$, 5 MeV, g = 0.05)"}


def observed(dr2_dir, config, reco_edges, dec_edges_deg):
    """DR2's events of a configuration histogrammed in (reconstructed dec, log10 E_reco): columns run, event,
    subevent, MJD, log10(E/GeV), AngErr, RA, Dec, Azimuth, Zenith (the release's readme)."""
    pattern = "IC86_*_exp.tab" if config == "IC86" else f"{config}_exp.tab"
    counts = np.zeros((len(dec_edges_deg) - 1, len(reco_edges) - 1))
    files = sorted(glob.glob(os.path.join(dr2_dir, "events", pattern)))
    for p in files:
        ev = np.loadtxt(p, ndmin=2)
        counts += np.histogram2d(ev[:, 7], ev[:, 4], bins=(dec_edges_deg, reco_edges))[0]
    return counts, len(files)


def sky_config(dp=None, si=None, nside=64, galactic_data=None, dec_edges_deg=None):
    cfg = {"source": {"p_index": ASTRO_9P5["gamma"]},
           "grid": {"n_E_bins": 150, "E_min_eV": 1e12, "E_max_eV": 1e17, "n_z_bins": 400},
           "sky": {"output": "bands", "nside": nside, "dec_band_edges_deg": [float(x) for x in dec_edges_deg],
                   "extragalactic_norm": ASTRO_9P5["per_flavor"], "extragalactic_norm_E_GeV": ASTRO_9P5["pivot_GeV"]},
           "galactic": {"enabled": True, "template": "cringe", "emission": "3d", "data_dir": galactic_data},
           "execution": {"verbose": 0}}
    if dp is not None:
        cfg["dark_portal"] = dp
    if si is not None:
        cfg["secret_interactions"] = si
    return cfg


def sky_flux(cfg, cache_dir=None):
    """propagate_sky for cfg; with cache_dir, kept as sky_<digest of cfg and code version>.npz and reused."""
    import hashlib
    path = None
    if cache_dir:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        key = hashlib.sha1((json.dumps(cfg, sort_keys=True) + head).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"sky_{key}.npz")
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                out = {k: z[k] for k in z.files if k != "meta_json"}
                out["meta"] = json.loads(str(z["meta_json"]))
            print(f"  (sky from {path})", flush=True)
            return out
    import dark_portal_solver as dps
    E, params = dps.build_solver_params(cfg)
    params["show_progress"] = False
    out = dps.propagate_sky(E, params)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, meta_json=np.array(json.dumps(out["meta"], default=float)),
                 **{k: v for k, v in out.items() if k != "meta"})
    return out


def windowed(flux, E_lo, E_hi):
    def f(E, dlo, dhi):
        E = np.asarray(E, dtype=np.float64)
        return flux(E, dlo, dhi) * ((E >= E_lo) & (E <= E_hi))[None, :]
    return f


FIT_PRIOR = {"astro_norm": 0.25, "astro_gamma": 0.2}        # NuFlavorSingleSource's OTHER_PRIOR
ATM_NORM_PRIOR = 0.10                                       # its per-block background normalization


def fit_nuisances(obs, T, configs, pri, sel):
    """The atmospheric (Barr, cosmic-ray index, prompt), astrophysical (normalization, index) and per-configuration
    normalization nuisances fitted to the events (Poisson, Gaussian priors), on the bins sel of reconstructed
    declination and every reconstructed energy:
      mu_c = (1 + n_c)(nom + sum_p theta_p d_p + theta_prompt prompt) + (1 + a_norm)(astro + a_gamma d astro/d gamma)."""
    from scipy.optimize import minimize
    names = list(FA.ATM_PARAMS) + ["astro_norm", "astro_gamma"] + [f"atm_norm_{c}" for c in configs]
    sig = np.array([pri[p] for p in FA.ATM_PARAMS] + [FIT_PRIOR["astro_norm"], FIT_PRIOR["astro_gamma"]] +
                   [ATM_NORM_PRIOR] * len(configs))
    n = {c: obs[c][sel] for c in configs}
    Tc = {c: {k: T[k][c][sel] for k in T} for c in configs}

    def mus(x, full=False):
        th = dict(zip(names, x))
        out = {}
        for c in configs:
            t = (T if full else Tc)
            g = (lambda k: t[k][c]) if full else (lambda k: t[c][k])
            atm = g("nom") + sum(th[p] * g(p) for p in FA.ATM_PARAMS)
            out[c] = (1.0 + th[f"atm_norm_{c}"]) * atm + (1.0 + th["astro_norm"]) * (g("astro") + th["astro_gamma"] * g("astro_gamma"))
        return out

    def deviance(x):
        tot = 0.0
        for c, m in mus(x).items():
            m = np.maximum(m, 1e-12)
            k = n[c]
            tot += 2.0 * np.sum(m - k + np.where(k > 0, k * np.log(np.where(k > 0, k, 1.0) / m), 0.0))
        return tot

    def objective(x):
        return deviance(x) + np.sum((x / sig) ** 2)

    x0 = np.zeros(len(names))
    # normalization factors (1 + n) stay positive; the rest are free (mu is floored at 1e-12)
    bounds = [(-0.95, None) if (k.startswith("atm_norm") or k == "astro_norm") else (None, None) for k in names]
    r = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=5000, maxfun=200000))
    n_bins = int(sum(np.sum(mus(x0)[c] > 0) for c in configs))
    return dict(values={k: float(v) for k, v in zip(names, r.x)}, pulls={k: float(v / s) for k, v, s in zip(names, r.x, sig)},
                priors={k: float(s) for k, s in zip(names, sig)}, deviance=float(deviance(r.x)),
                deviance_nominal=float(deviance(x0)), objective=float(r.fun), n_bins=n_bins, converged=bool(r.success),
                message=str(r.message), mu=mus(r.x, full=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dr2", required=True)
    ap.add_argument("--galactic-data", default=None)
    ap.add_argument("--skip-solver", action="store_true")
    ap.add_argument("--atm", default=None, help="the atmospheric table (default IceCubeRates/data/atmospheric_mceq_southpole.npz)")
    ap.add_argument("--outdir", default=os.path.join(REPO, "output"), help="where tracks/ and figures/ go")
    ap.add_argument("--sky-cache", default=None, help="keep and reuse the solver's sky per case (npz) here")
    ap.add_argument("--sky-only", action="store_true", help="compute the solver's skies (with --sky-cache) and stop")
    ap.add_argument("--fit-configs", nargs="*", default=None,
                    help="the configurations the nuisance fit uses (default all; e.g. IC79 IC86: IC40 has horizon muons, "
                         "IC59 a high-energy excess at every declination, 2026-10-14)")
    a = ap.parse_args()
    t0 = time.time()
    tr = DR2Tracks()
    dec_deg = np.degrees(tr.dec_edges("IC86"))
    dec_deg[0], dec_deg[-1] = -90.0, 90.0
    if a.sky_only:
        for name, kw in CASES.items():
            t1 = time.time()
            sky_flux(sky_config(galactic_data=a.galactic_data, dec_edges_deg=dec_deg, **kw), a.sky_cache)
            print(f"{name}: sky computed ({time.time() - t1:.0f} s)", flush=True)
        return
    edges, rde = tr.reco_edges, np.radians(RECO_DEC_DEG)
    north = slice(1, None)
    res = dict(reco_dec_edges_deg=RECO_DEC_DEG.tolist(), reco_log10E_edges=edges.tolist(),
               astro_fit="IceCube 9.5-yr northern tracks (ApJ 928, 50): nu_mu + nubar_mu 1.44e-18 at 100 TeV, gamma 2.37")
    # (a) the responses
    res["responses"] = {}
    for c in tr.configs:
        norm = tr.data[f"{c}_smear_norm"]
        le, de, A = tr.aeff(c)
        iE = int(np.searchsorted(le, 5.0)) - 1
        k_h = int(np.searchsorted(de, 0.0))
        res["responses"][c] = dict(livetime_days=tr.livetime(c) / 86400.0,
                                   smearing_sum_filled=[float(norm[norm > 0].min()), float(norm[norm > 0].max())],
                                   empty_smearing_bins=int(np.sum(norm == 0)),
                                   aeff_100TeV_m2_just_above_horizon=float(A[iE, k_h] / 1e4))
    # (b) the units check: every atmospheric template (NuFlavorSingleSource's treatment) folded once
    akw = {} if a.atm is None else dict(path=a.atm)
    pri = FA.Atmospheric(**akw).priors()
    terms = ["nom", "prompt"] + list(FA.BARR_PARAMS) + ["cr_dgamma", "daemonflux"]
    T = {t: tr.expected(FA.Atmospheric(term=t, **akw), reco_dec_edges=rde)["per_config"] for t in terms}
    pl = FA.power_law(**ASTRO_9P5)
    T["astro"] = tr.expected(pl, reco_dec_edges=rde)["per_config"]
    T["astro_gamma"] = tr.expected(lambda E, dlo, dhi: -np.log(np.asarray(E) / ASTRO_9P5["pivot_GeV"])[None, :]
                                   * pl(E, dlo, dhi), reco_dec_edges=rde)["per_config"]
    obs, n_files = {}, {}
    for c in tr.configs:
        obs[c], n_files[c] = observed(a.dr2, c, edges, RECO_DEC_DEG)
    fit_configs = list(a.fit_configs) if a.fit_configs else list(tr.configs)
    fit = fit_nuisances(obs, T, fit_configs, pri, north)
    res["nuisance_fit_configs"] = fit_configs
    res["nuisance_fit"] = {k: v for k, v in fit.items() if k != "mu"}
    res["northern_tracks"] = {}
    hi = edges[:-1] >= 4.0
    for c in tr.configs:
        e = {k: T[k][c][north] for k in T}
        o = obs[c][north]
        nominal = e["nom"] + e["astro"]
        best = fit["mu"][c][north] if c in fit["mu"] else np.full_like(nominal, np.nan)
        d = dict(observed=float(o.sum()), nominal=float(nominal.sum()), atmospheric=float(e["nom"].sum()),
                 prompt=float(e["prompt"].sum()), astro=float(e["astro"].sum()),
                 prior_band=float(np.sqrt(sum((pri[p] * e[p].sum()) ** 2 for p in FA.ATM_PARAMS))), seasons=n_files[c],
                 best_fit=float(best.sum()), daemonflux_conv_plus_prompt=float((e["daemonflux"] + e["prompt"]).sum()),
                 observed_above_10TeV=float(o[:, hi].sum()), nominal_above_10TeV=float(nominal[:, hi].sum()),
                 astro_above_10TeV=float(e["astro"][:, hi].sum()), best_fit_above_10TeV=float(best[:, hi].sum()),
                 observed_E=o.sum(0).tolist(), nominal_E=nominal.sum(0).tolist(), atmospheric_E=e["nom"].sum(0).tolist(),
                 prompt_E=e["prompt"].sum(0).tolist(), astro_E=e["astro"].sum(0).tolist(),
                 band_E=np.sqrt(sum((pri[p] * e[p].sum(0)) ** 2 for p in FA.ATM_PARAMS)).tolist(),
                 best_E=best.sum(0).tolist(), daemonflux_E=(e["daemonflux"] + e["prompt"]).sum(0).tolist(),
                 observed_dec=o.sum(1).tolist(), nominal_dec=nominal.sum(1).tolist(), best_dec=best.sum(1).tolist(),
                 observed_2d=o.tolist(), nominal_2d=nominal.tolist())
        res["northern_tracks"][c] = d
        print(f"{c} ({tr.livetime(c) / 86400:.0f} d): reco dec > {DEC_CUT_DEG:g}: observed {d['observed']:.0f}, nominal "
              f"{d['nominal']:.0f} (atmospheric {d['atmospheric']:.0f}, of which prompt {d['prompt']:.1f}; astro {d['astro']:.1f}), "
              f"ratio {d['observed'] / d['nominal']:.3f}, best fit {d['best_fit']:.0f}; daemonflux + prompt "
              f"{d['daemonflux_conv_plus_prompt']:.0f}; reco E > 10 TeV: observed {d['observed_above_10TeV']:.0f}, nominal "
              f"{d['nominal_above_10TeV']:.1f}, best fit {d['best_fit_above_10TeV']:.1f}", flush=True)
    print("nuisance fit: " + ", ".join(f"{k} {v:+.3f}" for k, v in fit["values"].items()) +
          f"; deviance {fit['deviance_nominal']:.0f} -> {fit['deviance']:.0f} over {fit['n_bins']} bins", flush=True)
    # (c) the solver's fluxes
    rows = {}
    if not a.skip_solver:
        pl_counts = tr.expected(windowed(pl, 1e3, 1e8), reco_dec_edges=rde)["counts"]
        res["cases"] = {}
        k_mu = None
        for name, kw in CASES.items():
            t1 = time.time()
            sky = sky_flux(sky_config(galactic_data=a.galactic_data, dec_edges_deg=dec_deg, **kw), a.sky_cache)
            if k_mu is None:
                # the 9.5-yr fit measured nu_mu + nubar_mu; sky.extragalactic_norm sets the flavor average, and pion
                # sources put more than a third into nu_mu at the Earth: rescale the extragalactic flux of every case
                # so that without new physics (the first case, standard) its nu_mu + nubar_mu matches the fit
                assert name == "standard"
                mu100 = float(FA.from_sky(sky, galactic=False)(np.array([ASTRO_9P5["pivot_GeV"]]), np.array([-0.5 * np.pi]),
                                                                np.array([0.5 * np.pi]))[0, 0])
                k_mu = ASTRO_9P5["per_flavor"] / mu100
                res["numu_normalization"] = dict(
                    factor=k_mu, numu_share_at_earth=mu100 / (3.0 * ASTRO_9P5["per_flavor"]),
                    note="extragalactic fluxes x factor: without new physics nu_mu + nubar_mu = 1.44e-18 at 100 TeV")
                print(f"nu_mu normalization: x {k_mu:.5f} (nu_mu + nubar_mu at 100 TeV, standard: "
                      f"{mu100 / (3.0 * ASTRO_9P5['per_flavor']):.4f} of three times the flavor average)", flush=True)
            eg = k_mu * tr.expected(FA.from_sky(sky, galactic=False), reco_dec_edges=rde)["counts"]
            ga = tr.expected(FA.from_sky(sky, extragalactic=False), reco_dec_edges=rde)["counts"]
            rows[name] = (eg, ga)
            res["cases"][name] = dict(extragalactic=float(eg[north].sum()), galactic=float(ga[north].sum()),
                                      extragalactic_E=eg[north].sum(0).tolist(), galactic_E=ga[north].sum(0).tolist(),
                                      extragalactic_dec=eg[north].sum(1).tolist(), galactic_dec=ga[north].sum(1).tolist(),
                                      seconds=time.time() - t1)
            if name == "standard":
                res["standard_vs_power_law"] = dict(
                    ratio_north=float(eg[north].sum() / pl_counts[north].sum()),
                    max_bin_deviation_above_1_event=float(np.max(np.abs(eg / np.where(pl_counts > 1, pl_counts, np.nan) - 1)[pl_counts > 1])))
                print(f"standard extragalactic / the 9.5-yr power law (1 TeV-100 PeV): "
                      f"{res['standard_vs_power_law']['ratio_north']:.5f}", flush=True)
            print(f"{name}: northern astrophysical tracks {eg[north].sum():.1f} extragalactic + {ga[north].sum():.1f} "
                  f"Galactic ({time.time() - t1:.0f} s)", flush=True)
    res["seconds"] = time.time() - t0
    res["commit"] = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()
    for d in ("tracks", "figures"):
        os.makedirs(os.path.join(a.outdir, d), exist_ok=True)
    with open(os.path.join(a.outdir, "tracks", "dr2_validation.json"), "w") as f:
        json.dump(res, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), gridspec_kw=dict(height_ratios=(3, 1.2)), sharex="col")
    for col, (cs, title) in enumerate(((["IC86"], "IC86 (2011-2022)"), (["IC40", "IC59", "IC79"], "IC40, IC59, IC79"))):
        ax, axr = axes[0, col], axes[1, col]
        g = lambda key: sum(np.array(res["northern_tracks"][c][key]) for c in cs)  # noqa: E731
        nom, band, best, obs = g("nominal_E"), g("band_E"), g("best_E"), g("observed_E")
        ax.stairs(nom, edges, color="k", lw=1.4, label="expected, nominal: atmospheric (MCEq) + astrophysical")
        ax.stairs(nom + band, edges, baseline=np.maximum(nom - band, 0), fill=True, color="0.6", alpha=0.45,
                  label=r"atmospheric priors (Barr, CR index, prompt), 1$\sigma$")
        ax.stairs(best, edges, color="C2", lw=1.2, label="best fit of the nuisances")
        ax.stairs(g("daemonflux_E") + g("astro_E"), edges, color="C0", ls="--", lw=0.9,
                  label="daemonflux conventional + prompt + astro (cross-check)")
        ax.stairs(g("prompt_E"), edges, color="C1", lw=0.9, label="prompt (MCEq, SIBYLL-2.3d)")
        ax.stairs(g("astro_E"), edges, color="C3", lw=0.9, label="astrophysical (IceCube 9.5 yr)")
        ax.errorbar(x, obs, yerr=np.sqrt(obs), fmt="o", ms=2.5, color="k", lw=0.8, label="DR2 events")
        ax.set_yscale("log"); ax.set_ylim(0.3, None)
        ax.set_ylabel(f"tracks per 0.1 dex, dec$_{{reco}}$ > {DEC_CUT_DEG:g}$^\\circ$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.errorbar(x, obs / nom, yerr=np.sqrt(obs) / nom, fmt="o", ms=2.5, color="k", lw=0.8)
            axr.fill_between(edges, np.append(1 - band / nom, (1 - band / nom)[-1]),
                             np.append(1 + band / nom, (1 + band / nom)[-1]), step="post", color="0.6", alpha=0.45)
            axr.stairs(best / nom, edges, color="C2", lw=1.2)
        axr.axhline(1, color="k", lw=0.6)
        axr.set_ylim(0.4, 1.6); axr.set_xlim(1.5, 7.0)
        axr.set_xlabel(r"log$_{10}$(reconstructed muon energy / GeV)"); axr.set_ylabel("data / nominal")
    ax, axr = axes[0, 2], axes[1, 2]
    s = np.sin(rde[1:])
    sx = 0.5 * (s[1:] + s[:-1])
    obs_d = sum(np.array(res["northern_tracks"][c]["observed_dec"]) for c in tr.configs)
    nom_d = sum(np.array(res["northern_tracks"][c]["nominal_dec"]) for c in tr.configs)
    best_d = sum(np.array(res["northern_tracks"][c]["best_dec"]) for c in tr.configs)
    ax.stairs(nom_d / np.diff(s), s, color="k", label="expected, nominal (all configurations)")
    ax.stairs(best_d / np.diff(s), s, color="C2", label="best fit")
    ax.errorbar(sx, obs_d / np.diff(s), yerr=np.sqrt(obs_d) / np.diff(s), fmt="o", ms=2.5, color="k", lw=0.8,
                label="DR2 events")
    ax.set_yscale("log"); ax.set_ylabel(r"tracks per unit sin(dec$_{reco}$)"); ax.legend(fontsize=7)
    ax.set_title("Declination (all reconstructed energies)", fontsize=10)
    axr.errorbar(sx, obs_d / nom_d, yerr=np.sqrt(obs_d) / nom_d, fmt="o", ms=2.5, color="k", lw=0.8)
    axr.stairs(best_d / nom_d, s, color="C2")
    axr.axhline(1, color="k", lw=0.6); axr.set_ylim(0.4, 1.6)
    axr.set_xlabel(r"sin(dec$_{reco}$)"); axr.set_ylabel("data / nominal")
    fit = res["nuisance_fit"]
    big = sorted(fit["pulls"].items(), key=lambda kv: -abs(kv[1]))[:4]
    fig.suptitle("IceTracks-DR2: our expectation (DR2 responses; MCEq SIBYLL-2.3d x H3a with NuFlavorSingleSource's "
                 "nuisances; IceCube's 9.5-yr astrophysical fit) against the events\nNuisance fit: Poisson deviance "
                 f"{fit['deviance_nominal']:.0f} -> {fit['deviance']:.0f} over {fit['n_bins']} bins; largest pulls "
                 + ", ".join(f"{k} {v:+.1f}" for k, v in big), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "figures", "dr2_validation.png"), dpi=110)
    plt.close(fig)

    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        cols = dict(zip(CASES, ("k", "C3", "C0")))
        ax = axes[0]
        for name, (eg, ga) in rows.items():
            ax.stairs(eg[north].sum(0), edges, color=cols[name], label=f"{CASE_LABELS[name]}: extragalactic")
            ax.stairs(ga[north].sum(0), edges, color=cols[name], ls=":", label="  Galactic (CRINGE x 0.91, 3-D)")
        ax.set_yscale("log"); ax.set_xlim(2.5, 7.0); ax.set_ylim(1e-2, None)
        ax.set_xlabel(r"log$_{10}$(reconstructed muon energy / GeV)")
        ax.set_ylabel(f"astrophysical tracks per 0.1 dex, dec$_{{reco}}$ > {DEC_CUT_DEG:g}$^\\circ$")
        ax.set_title("IceCube 2008-2022 (all configurations)", fontsize=10)
        ax.legend(fontsize=6.5)
        ax = axes[1]
        ref_eg = rows["standard"][0][north].sum(0)
        for name, (eg, ga) in rows.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                ax.stairs(eg[north].sum(0) / ref_eg, edges, color=cols[name], label=CASE_LABELS[name])
        ax.axhline(1, color="k", lw=0.5); ax.set_xlim(2.5, 7.0); ax.set_ylim(0, 1.5)
        ax.set_xlabel(r"log$_{10}$(reconstructed muon energy / GeV)"); ax.set_ylabel("extragalactic tracks / standard")
        ax.set_title("The new physics in the track spectrum", fontsize=10); ax.legend(fontsize=7)
        ax = axes[2]
        hi = edges[:-1] >= 4.0
        for name, (eg, ga) in rows.items():
            ax.stairs(eg[north][:, hi].sum(1) / np.diff(s), s, color=cols[name], label=f"{CASE_LABELS[name]}: extragalactic")
            ax.stairs(ga[north][:, hi].sum(1) / np.diff(s), s, color=cols[name], ls=":", label="  Galactic")
        ax.set_yscale("log"); ax.set_xlabel(r"sin(dec$_{reco}$)")
        ax.set_ylabel(r"tracks per unit sin(dec$_{reco}$), E$_{reco}$ > 10 TeV")
        ax.set_title("Per declination (reconstructed energy above 10 TeV)", fontsize=10); ax.legend(fontsize=6.5)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, "figures", "dr2_tracks.png"), dpi=110)
        plt.close(fig)
    print(f"wrote {a.outdir}/tracks/dr2_validation.json and the figures ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
