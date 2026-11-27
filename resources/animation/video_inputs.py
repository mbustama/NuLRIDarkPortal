"""
video_inputs.py -- everything the animation draws that comes from the paper's results, read once from the
repository (read-only; it runs neither the solver nor the fits) and stored in data/inputs.npz.

  python resources/animation/video_inputs.py [--repo PATH]

PATH is the checkout whose paper/main.tex and output/analysis/ the animation follows (default: the main checkout of
this repository, found through git, since the worktrees do not carry the untracked arrays). What it reads:
  - output/analysis/trajectory_figure_fig7.npz: the flux of each species along the trip, for the paper's three
    examples (Fig. 4; analysis/trajectory_figure.py), normalized as Fig. 4 (to the total once the sources stop
    emitting). Stored per stage: the coordinate and the flux per flavor, neutrinos plus antineutrinos.
  - output/analysis/three_classes_figure.npz: the flux at Earth for no new interaction and for the full calculation
    of each example (Fig. 7; analysis/three_classes_figure.py); the flavor composition weighted by the MESE events of
    each flavor over 5 TeV-10 PeV (analysis.mese.event_weighted_fractions), as Fig. 7.
  - IceCube's MESE flavor likelihood, Delta chi^2 on a grid (analysis.surfaces.chi2, 'icecube_mese_2025_grid').
  - The limits of the paper's figures (output/analysis/limits_<date>/, read with analysis/exclusion.py exactly as
    notebooks/paper_figures.ipynb does) for the three families of the examples, with the flavor composition at the
    sources fitted: today (IceCube MESE flavor + spectrum) and the eight future telescopes; the existing bounds
    (analysis/existing_bounds.py) as the lowest curve at each mass, and the earlier estimate for L_e - L_mu.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "inputs.npz")

#: The limits folder that notebooks/paper_figures.ipynb reads (its LIMITS), and its (checkout, campaign) per family.
LIMITS_DATE = "limits_20261127"
WHERE = {"le_ul": ("nulri_set1_sm", "campaign_final_sm"), "dp_ul": ("nulri_set1_final", "campaign_final"),
         "mev": ("nulri_set1_ml", "campaign_ml"), "si_now": ("nulri_refit_set", "campaign_si"),
         "si_proj": ("nulri_forecast", "campaign_si")}
DATASETS = ("now_icecube", "proj_future_planA")
#: The three panels of the last scene: (key, model, where the ultralight and MeV planes are, the MeV plane's name).
LIMIT_PANELS = [("le", "le_lmu", "le_ul", "mev", "le_lmu_mev"),
                ("dp", "dp_tau", "dp_ul", "mev", "dp_tau_mev"),
                ("si", "si_scalar_majorana_tau", "si_now", "si_now", "si_scalar_majorana_tau")]
FLAVOR_BAND_EV = (5e12, 1e16)


def main_checkout():
    common = subprocess.run(["git", "-C", HERE, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return os.path.dirname(common)


def trip(repo, out):
    d = dict(np.load(os.path.join(repo, "output", "analysis", "trajectory_figure_fig7.npz")))
    for key in ("si", "dp", "le_20"):
        z, U = d[f"{key}/z"], d[f"{key}/universe"]
        norm = U[:, int(np.argmax(z <= 1.8))].sum()          # as trajectory_figure.figure()
        out[f"trip/{key}/z"] = z
        out[f"trip/{key}/universe"] = U.sum(axis=0) / norm   # (n, 3): nu + nubar per flavor
        for st in ("galaxy", "sun", "earth"):
            if f"{key}/{st}" in d:
                out[f"trip/{key}/{st}_x"] = d[f"{key}/{st}_x"]
                out[f"trip/{key}/{st}"] = d[f"{key}/{st}"].sum(axis=0) / norm
    out["trip/E_obs_eV"] = d["si/E"]


def at_earth(repo, out):
    sys.path[:0] = [repo, os.path.join(repo, "src")]
    from analysis import mese, surfaces
    FX = dict(np.load(os.path.join(repo, "output", "analysis", "three_classes_figure.npz")))
    E = FX["E"]
    band = (E >= FLAVOR_BAND_EV[0] * (1 - 1e-9)) & (E <= FLAVOR_BAND_EV[1] * (1 + 1e-9))
    out["earth/E"] = E
    for key in ("le", "dp", "si"):
        out[f"earth/{key}_ratio"] = FX[f"{key}_full"].sum(1) / FX[f"{key}_none"].sum(1)
        out[f"earth/{key}_flavor"] = mese.event_weighted_fractions(E[band], FX[f"{key}_full"][band])
    out["earth/none_flavor"] = mese.event_weighted_fractions(E[band], FX["le_none"][band])
    g = np.linspace(0, 1, 121)
    C = np.full((len(g), len(g)), np.nan)
    for i, fm in enumerate(g):
        for j, fe in enumerate(g):
            if fe + fm <= 1 + 1e-12:
                C[i, j] = surfaces.chi2(fe, fm, "icecube_mese_2025_grid")
    out["earth/mese_grid"] = g
    out["earth/mese_dchi2"] = C - np.nanmin(C)


def limits(repo, out):
    from analysis import exclusion as ex
    from analysis import existing_bounds as eb
    root = os.path.join(repo, "output", "analysis", LIMITS_DATE)
    for key, model, w_ul, w_mev, mev_name in LIMIT_PANELS:
        for ds in DATASETS:
            wu, wm = w_ul, w_mev
            if key == "si" and ds.startswith("proj"):
                wu = wm = "si_proj"
            ul = ex.limits_plane(root, WHERE[wu], ds, f"{model}_ul_fx")
            mev = ex.limits_plane(root, WHERE[wm], ds, f"{mev_name}_fx")
            if ul and mev:
                lx, G, F = ex.along_ratio(*ex.joined(ul, mev, **({"dg": 0.125} if key == "si" else {})))
            elif mev:
                lx, G, F, _, _ = ex.field(mev)
            else:
                raise FileNotFoundError(f"no planes for {model}, {ds}")
            out[f"limits/{key}/{ds}/lx"], out[f"limits/{key}/{ds}/G"], out[f"limits/{key}/{ds}/F"] = lx, G, F
        # existing bounds: the lowest drawn curve at each mass; the dashed earlier estimate apart (as the notebook)
        xs = np.linspace(-31, 10, 1200)
        low = np.full(len(xs), np.inf)
        for b_key in eb.for_model(model):
            b = eb.BOUNDS[b_key]
            if not b.get("draw") or b["kind"] != "curve" or b.get("region") == "inside":
                continue
            try:
                M, g = eb.load(b_key, model)
            except (OSError, KeyError, ValueError):
                continue
            lx, lg = np.log10(M), np.log10(g)
            o = np.argsort(lx)
            if b.get("line") == "dashed":
                out[f"limits/{key}/earlier"] = np.c_[lx[o], lg[o]]
                continue
            inside = (xs >= lx[o][0]) & (xs <= lx[o][-1])
            low[inside] = np.minimum(low[inside], np.interp(xs[inside], lx[o], lg[o]))
        out[f"limits/{key}/bounds"] = np.c_[xs, np.where(np.isfinite(low), low, np.nan)]
        # where the paper makes no claim, hatched as in its figures: above the line of perturbative unitarity or of
        # the longitudinal mode's left-out scatterings (L_e - L_mu, dark portal: exclusion.hatched_line), above the
        # boundary of the field that the relics source (the scalar: scalar_field_boundaries.json)
        if key in ("le", "dp"):
            p = ex.limits_plane(root, WHERE[w_ul], "now_icecube", f"{model}_ul_fx")
            out[f"limits/{key}/hatch_line"] = np.array(ex.hatched_line(ex.not_modelled_lines(p)))
        else:
            sf = json.load(open(os.path.join(root, *WHERE["si_now"], "scalar_field_boundaries.json")))
            out[f"limits/{key}/hatch_curve"] = np.array([(x, y) for x, y in zip(sf["log10_m"], sf[model])
                                                         if y is not None])
    out["limits/source"] = np.array(LIMITS_DATE)


#: The L_e - L_mu example of the paper (l. 398): mass (eV) and coupling.
LE_M_EV, LE_G = 1e-20, 1e-21


def potential(repo, out):
    """The L_e - L_mu potential of the example on the trip, |V| = g'^2 Phi_e (Eq. 2), relative to the larger
    oscillation term, Delta m^2_31 / (2 E (1+z)), at the observed energy of Fig. 4 (495 TeV). The range of the force,
    1/M, is about 130 AU here: far below the Galaxy's scales, so there Phi_e = n_e / M^2 (the local density); far above
    the Earth's, so near it Phi_e is the 1/r potential of the Sun's and the Earth's electrons. Densities from the
    paper's code: the Milky Way's model (dp_spatial.calc_MW_electron_density, axisymmetric: one radial profile in the
    disk's plane), PREM (dp_spatial.calc_earth_density, earth_electrons_per_gram), the electrons of the Sun, the Earth
    and the Moon (dp_core.N_e_*), the cosmic mean (dp_config's cosmology.n_e0_cm3)."""
    sys.path[:0] = [os.path.join(repo, "src")]
    import dp_core
    import dp_spatial
    import dp_config
    E = float(out["trip/E_obs_eV"])
    with open(os.path.join(repo, "analysis", "configs", "campaign_ml", "now_ic26", "le_lmu_mev.json")) as fh:
        dm31 = float(json.load(fh)["solver"]["standard_mixing"]["dm31_sq"])
    n_e0 = float(dp_config._build_default_config()["cosmology"]["n_e0_cm3"])
    cm3 = dp_core.cm_inv3_to_eV3
    M, g2 = LE_M_EV, LE_G ** 2
    osc0 = dm31 / (2.0 * E)                                    # eV, at z = 0
    km = 1e3 * dp_core.m_to_eV_inv                             # eV^-1 per km
    au = dp_core.d_sun_eV_inv                                  # eV^-1 per AU (the code's 1 AU)
    # the Universe: the cosmic mean density, (1+z)^3, against the oscillation term at the energy then, E (1+z)
    z = np.linspace(0.0, 2.3, 231)
    out["pot/z"] = z
    out["pot/universe"] = g2 * n_e0 * cm3 * (1 + z) ** 3 / M ** 2 / (osc0 / (1 + z))
    # the Milky Way, in its plane: n_e at galactocentric radius R (a point at distance r from the Sun, towards the
    # Galactic Centre, phi = 0, theta = pi/2, sits at R = |r - R_0|; beyond the centre, r = R_0 + R)
    R0 = 8.33
    R = np.linspace(0.0, 45.0, 1801)
    r_sun = np.where(R <= R0, R0 - R, R0 + R)
    ne = np.array([dp_spatial.calc_MW_electron_density(float(r), np.pi / 2, 0.0) for r in r_sun])
    out["pot/mw_R"] = R
    out["pot/mw"] = g2 * ne * cm3 / M ** 2 / osc0
    out["pot/mw_R0"] = np.array(R0)
    # near the Sun: its electrons, e^{-Mr} / (4 pi r), outside the Sun; the Galaxy's local value beneath
    r_au = np.geomspace(0.005, 50.0, 400)
    mw_local = float(np.interp(R0, R, out["pot/mw"]))
    out["pot/sun_r_au"] = r_au
    out["pot/sun"] = g2 * dp_core.N_e_sun * np.exp(-M * r_au * au) / (4 * np.pi * r_au * au) / osc0 + mw_local
    # the Earth: its electrons (PREM), Phi(r) = Q(<r) / (4 pi r) + int_r^R n_e r' dr' (1/M >> R_E: e^{-Mr} = 1),
    # outside N_e / (4 pi r); on top of the Sun's at 1 AU and the Moon's at its distance
    rk = np.linspace(0.0, 6371.0, 6372)
    ne_e = np.array([dp_spatial.calc_earth_density(float(x)) * dp_spatial.earth_electrons_per_gram(float(x))
                     for x in rk]) * cm3                        # eV^3 (g/cm^3 x electrons/g)
    shell = 4 * np.pi * (rk * km) ** 2 * ne_e
    Q = np.concatenate([[0.0], np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(rk * km))])
    inner = np.concatenate([np.cumsum((ne_e * rk * km)[::-1][:-1] * np.diff(rk * km)[::-1])[::-1], [0.0]])
    r_out = np.linspace(6371.0, 40000.0, 400)[1:]
    phi_in = np.where(rk > 0, Q / (4 * np.pi * np.maximum(rk, 1e-9) * km), 0.0) + inner
    phi_in[0] = inner[0]
    phi_out = dp_core.N_e_earth / (4 * np.pi * r_out * km)
    background = (dp_core.N_e_sun * np.exp(-M * au) / (4 * np.pi * au)
                  + dp_core.N_e_moon / (4 * np.pi * dp_core.d_moon_eV_inv))
    out["pot/earth_r_km"] = np.concatenate([rk, r_out])
    out["pot/earth"] = g2 * (np.concatenate([phi_in, phi_out]) + background) / osc0 + mw_local
    out["pot/earth_Q_check"] = np.array([Q[-1], dp_core.N_e_earth])     # PREM's electrons against dp_core's


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    repo = a.repo or main_checkout()
    t0 = time.time()
    out = {}
    trip(repo, out)
    at_earth(repo, out)
    limits(repo, out)
    potential(repo, out)
    head = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    out["provenance"] = np.array(json.dumps({"repo": repo, "head": head.strip(), "date": time.strftime("%Y-%m-%d"),
                                             "limits": LIMITS_DATE}))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}: {len(out)} arrays, {os.path.getsize(OUT) / 1e6:.2f} MB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
