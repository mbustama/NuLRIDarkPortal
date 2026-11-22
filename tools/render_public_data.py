"""Write the public companion repository's data files (public_repo/data/).

Every number the companion holds comes from here: the limits and the chi^2 grids from a limits export
(tools/export_limits.py), read with the paper's own code (analysis.exclusion), and each figure's numbers from
the caches its module draws from. The companion then redraws every figure without the analysis package.

    python3 tools/render_public_data.py            # every file
    python3 tools/render_public_data.py limits     # data/limits.json and data/chi2/ only

SOURCE is the checkout whose analysis code and output/ the paper's figures are drawn from (by default this one;
NULRI_SOURCE sets another); OUT is public_repo/data in this checkout, its own git repository (ignored here).
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.environ.get("NULRI_SOURCE", HERE)
sys.path.insert(0, SOURCE)

from analysis import exclusion as ex  # noqa: E402

#: The paper, in one place.
PAPER = {"title": "New interactions of high-energy cosmic neutrinos, from the Hubble scale to the Earth",
         "authors": ["Mauricio Bustamante", "Sudipta Das", "Sanjib Kumar Agarwalla"],
         "arxiv": None,
         "repository": "https://github.com/mbustama/nu-new-interactions-cosmic"}
OUT = os.path.join(HERE, "public_repo", "data")
#: The limits export the paper's figures read (notebooks/paper_figures.ipynb, LIMITS).
EXPORT = os.path.join(SOURCE, "output", "analysis", "limits_20261121")
#: Where each family's planes are in the export: (checkout, campaign), as the notebook's WHERE.
WHERE = {"le_ul": ("nulri_set1_sm", "campaign_final_sm"), "dp_ul": ("nulri_set1_final", "campaign_final"),
         "mev": ("nulri_set1_ml", "campaign_ml"), "si_now": ("nulri_refit_set", "campaign_si"),
         "si_proj": ("nulri_forecast", "campaign_si")}
#: Significant digits: six in the limits, ten in the chi^2 grids and in a figure's numbers.
DIGITS = 6
FIGURE_DIGITS = 10
GZIP_ABOVE = 10_000_000

#: The data sets: the export's name, the public name, and how the figures' legends name them.
DATA_SETS = [("now_icecube", "icecube_mese_flavor_spectrum", "Today: IceCube MESE flavor + spectrum"),
             ("now_mese_dr2", "icecube_mese_dr2_events", "Today: IceCube MESE events + DR2 tracks"),
             ("proj_future_planA", "projected_eight_telescopes", "Projected: eight telescopes, 10 yr each")]
COMPOSITIONS = [("", "pion"), ("_fx", "fitted")]

SI_MODELS = [f"si_{a}_{b}_{c}" for a in ("scalar", "vector") for b in ("majorana", "dirac")
             for c in ("e", "mu", "tau", "universal")]
MODELS = ["le_lmu", "le_ltau", "dp_e", "dp_mu", "dp_tau"] + SI_MODELS

FLAVOR = {"e": "nu_e", "mu": "nu_mu", "tau": "nu_tau", "universal": "all three flavors equally"}


def describe(model: str) -> dict:
    """A model's plain description, its coupling and the name of its mass and coupling in the paper."""
    if model.startswith("le_"):
        a = {"le_lmu": "mu", "le_ltau": "tau"}[model]
        return {"family": "lepton-number gauge boson",
                "description": f"gauge boson of L_e - L_{a}",
                "mass": "M, the mass of the gauge boson, in eV",
                "coupling": "g', the gauge coupling"}
    if model.startswith("dp_"):
        f = model[3:]
        return {"family": "dark portal",
                "description": f"dark gauge boson that reaches the known neutrinos through the mixing of a dark neutrino "
                         f"with {FLAVOR[f]} only",
                "mass": "M, the mass of the dark gauge boson, in eV",
                "coupling": f"g_D |U_{f}4|^2, the dark coupling times the mixing squared"}
    _, med, nat, f = model.split("_")
    return {"family": "secret interactions",
            "description": f"{med} mediator coupled to {FLAVOR[f]}, {nat.capitalize()} neutrinos",
            "mass": "M, the mass of the mediator, in eV",
            "coupling": "g, the coupling of the mediator to the neutrinos"}


def rounded(x, digits: int = DIGITS):
    """Every float in a nested structure, to `digits` significant digits; NaN and inf become None."""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x):
            return None
        return float(f"{x:.{digits}g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return rounded(x.tolist(), digits)
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    return x


def write(path: str, payload) -> str:
    """Write `payload` as JSON (gzip-compressed above GZIP_ABOVE characters); returns the path written."""
    text = json.dumps(payload, indent=None, separators=(",", ":"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(text) > GZIP_ABOVE:
        import gzip
        path += ".gz"
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        with open(path, "w") as fh:
            fh.write(text + "\n")
    return path


# ------------------------------------------------------------------------------------------------ the limits

def planes():
    """Every plane the paper's limit figures read: (model, mass range, export data set, public data set,
    composition, path of its chi^2 summary)."""
    out = []
    for ds, pub, _ in DATA_SETS:
        for model in MODELS:
            for rng in ("ultralight", "mev"):
                if ds == "now_mese_dr2" and rng == "ultralight":
                    continue                          # this data set is used for the MeV masses only
                if model.startswith("si_"):
                    where = "si_proj" if ds.startswith("proj") else "si_now"
                    name = model + ("_ul" if rng == "ultralight" else "")
                else:
                    where = "mev" if rng == "mev" else model[:2] + "_ul"
                    name = model + ("_ul" if rng == "ultralight" else "_mev")
                for suffix, comp in COMPOSITIONS:
                    p = ex.limits_plane(EXPORT, WHERE[where], ds, name + suffix)
                    if not p:
                        raise FileNotFoundError(f"{ds}/{name + suffix}")
                    out.append((model, rng, ds, pub, comp, p))
    return out


def intervals(G, column) -> list:
    """The excluded couplings of one mass column of a field (signed distance to the 95% C.L. edge, > 0 inside), as
    [low, high] intervals in log10 g; each edge where the field changes sign, midway between samples."""
    inside = np.asarray(column.filled(-1.0) if np.ma.isMaskedArray(column) else np.nan_to_num(column, nan=-1.0)) > 0
    edges = np.flatnonzero(np.diff(np.r_[False, inside, False].astype(int)))
    out = []
    for i0, i1 in zip(edges[::2], edges[1::2] - 1):
        lo = G[i0] if i0 == 0 else 0.5 * (G[i0 - 1] + G[i0])
        hi = G[i1] if i1 == len(G) - 1 else 0.5 * (G[i1] + G[i1 + 1])
        out.append([lo, hi])
    return out


def not_modelled(path: str) -> dict:
    """The plane's not-modelled lines in plain words: {name: {slope, intercept, meaning}}."""
    meaning = {"not_perturbative": "above this line the coupling breaks perturbative unitarity",
               "longitudinal_left_out": "above this line, g/M > 1e-2 eV^-1, the scatterings of the mediator's "
                                        "longitudinal mode matter, and the calculation leaves them out"}
    lines = ex.not_modelled_lines(path)
    return {k: {"log10_coupling": f"{v['slope']:g} * log10_mass_eV + {v['intercept']:.6g}", "slope": v["slope"],
                "intercept": v["intercept"], "meaning": meaning[k]} for k, v in lines.items() if k in meaning}


def limits() -> None:
    """data/limits.json and data/chi2/*.json."""
    table = {m: {pub: {comp: None for _, comp in COMPOSITIONS} for _, pub, _ in DATA_SETS} for m in MODELS}
    for m in MODELS:
        for pub in table[m]:
            for comp in table[m][pub]:
                table[m][pub][comp] = {"masses": [], "not_modelled": {}}
    incomplete = []
    everything = planes()
    by_key = {(m, r, pub, comp): p for m, r, ds, pub, comp, p in everything}
    for model, rng, ds, pub, comp, path in everything:
        s = json.load(open(path))
        lim = json.load(open(os.path.join(os.path.dirname(os.path.dirname(path)), "limits.json")))
        if not lim.get("complete", True):
            incomplete.append(f"{model} {rng} {pub} {comp}")
        x, y = (np.log10(np.asarray(v, dtype=float)) for v in s["grids"])
        chi = [[None if v is None else v for v in row] for row in s["chi2"]]
        lines = not_modelled(path)
        payload = {
            "paper": PAPER["title"],
            "content": f"The chi^2 grid of {model}, {rng} masses, data set {pub}, source composition {comp}.",
            "model": model, "mass_range": rng, "data_set": pub, "composition": comp,
            "how_to_read": "chi2[i][j] is the lowest chi^2 over every other parameter of the fit, at mass "
                           "10**log10_mass_eV[i] eV and coupling 10**log10_coupling[j]; null where not computed. "
                           "chi2_no_new_interaction is the fit without a new interaction. The paper reads a mass "
                           "column at 95% C.L. as Delta chi^2 >= 3.84 from the column's lowest chi^2 (or from "
                           "chi2_no_new_interaction, where lower), with upper limits only; plotting/limits.py does it.",
            **describe(model),
            "log10_mass_eV": rounded(x, FIGURE_DIGITS), "log10_coupling": rounded(y, FIGURE_DIGITS),
            "chi2": rounded(chi, FIGURE_DIGITS), "chi2_no_new_interaction": rounded(s.get("chi2_sm"), FIGURE_DIGITS),
            "not_modelled": lines}
        write(os.path.join(OUT, "chi2", f"{model}__{rng}__{pub}__{comp}.json"), payload)
        table[model][pub][comp]["not_modelled"] = lines
    # the limits, mass by mass, as the figures read them: the ultralight plane without its thin bands, then the MeV
    # plane from its lowest mass up (where both have a mass, the MeV plane's column, finer in coupling)
    for model in MODELS:
        for _, pub, _ in DATA_SETS:
            for _, comp in COMPOSITIONS:
                ul, mev = by_key.get((model, "ultralight", pub, comp)), by_key[(model, "mev", pub, comp)]
                lm, Gm, Fm, _, _ = ex.field(mev)
                rows = []
                if ul:
                    lu, Gu, Fu0, _, Cu = ex.field(ul)
                    Fu, _ = ex.thin_bands(lu, Gu, Fu0, Cu)
                    for k in range(len(lu)):
                        if lu[k] < lm[0] - 1e-9:
                            rows.append((lu[k], "ultralight", Gu, Fu[:, k], Fu0[:, k]))
                for k in range(len(lm)):
                    rows.append((lm[k], "mev", Gm, Fm[:, k], None))
                masses = []
                for lx, rng, G, col, col0 in rows:
                    iv = intervals(G, col)
                    entry = {"log10_mass_eV": lx, "excluded_log10_coupling": iv,
                             "lowest_coupling_computed": G[0], "highest_coupling_computed": G[-1],
                             "excluded_down_to_lowest_coupling": bool(iv and iv[0][0] <= G[0] + 1e-9),
                             "plane": rng}
                    if col0 is not None:
                        thin = [p for p in intervals(G, col0) if p not in iv]
                        if thin:
                            entry["thin_bands_not_shown"] = thin
                    masses.append(entry)
                table[model][pub][comp]["masses"] = rounded(masses, DIGITS)
    how = {
        "limits_95cl": "Keyed as [model][data set][composition]. Each entry lists the masses computed and, at each, "
                       "the couplings disfavored at 95% C.L. (per-mass Delta chi^2 >= 3.84), as intervals "
                       "[low, high] in log10 of the coupling; an empty list where no coupling is disfavored. A mass "
                       "can have more than one interval.",
        "model": "The new interaction; see 'models'.",
        "data set": "The data the limit comes from; see 'data_sets'.",
        "composition": "pion: the flavor composition at the sources fixed to nu_e : nu_mu : nu_tau = 1 : 2 : 0; "
                       "fitted: the composition at the sources left free in the fit. The paper's limit figures show "
                       "'fitted'.",
        "lowest_coupling_computed": "Below it nothing was computed: where 'excluded_down_to_lowest_coupling' is true, "
                                    "the limit lies lower than the interval says.",
        "highest_coupling_computed": "Above it nothing was computed.",
        "thin_bands_not_shown": "Ultralight grids only: excluded intervals that the figures leave out, because no "
                                "disk 0.4 decade in radius (in log10 mass and log10 coupling) fits inside the region "
                                "they belong to; listed so that nothing is hidden.",
        "plane": "Which grid the mass comes from: the ultralight one or the MeV one. Where both have a mass, the MeV "
                 "grid's column is used, as in the figures.",
        "not_modelled": "Lines above which the paper makes no claim: log10 g = slope * log10(M/eV) + intercept. For "
                        "the scalar of secret interactions, the region not modelled is not a line: it is in "
                        "data/scalar_field_boundaries.json."}
    payload = {"paper": PAPER, "content": "Every 95% C.L. limit of the paper, mass by mass.",
               "how_to_read": how,
               "models": {m: describe(m) for m in MODELS},
               "data_sets": {pub: label for _, pub, label in DATA_SETS},
               "compositions": {"pion": "pion-decay sources, 1 : 2 : 0", "fitted": "source composition fitted"},
               "limits_95cl": table}
    write(os.path.join(OUT, "limits.json"), payload)
    print(f"limits: {len(everything)} planes; incomplete in the export: {incomplete or 'none'}")


def side_lines() -> None:
    """data/cmb_lifetime.json (the CMB bound on the neutrino lifetime as a bound on the coupling) and
    data/scalar_field_boundaries.json (where the relics' scalar field changes the neutrino masses)."""
    cmb = {"content": "The CMB bound on the neutrino lifetime, as a bound on the coupling: couplings above "
                      "log10_coupling are excluded if the relic neutrinos are present at recombination; null where no "
                      "decay is open. Drawn as the dotted line in the limit figures."}
    for model, (chk, camp) in (("le", WHERE["le_ul"]), ("dp", WHERE["dp_ul"])):
        d = json.load(open(os.path.join(EXPORT, chk, camp, "cmb_lifetime.json")))
        for k, v in d.items():
            if k == "log10_m":
                cmb.setdefault("log10_mass_eV", {})[model] = rounded(v, FIGURE_DIGITS)
            elif not k.startswith("_"):
                cmb[k] = rounded(v, FIGURE_DIGITS)
    write(os.path.join(OUT, "cmb_lifetime.json"), cmb)
    d = json.load(open(os.path.join(EXPORT, *WHERE["si_now"], "scalar_field_boundaries.json")))
    sf = {"content": "For the scalar of secret interactions: above log10_coupling, the scalar field that the relic "
                     "neutrinos source changes the neutrino masses, which the paper does not model (hatched in the "
                     "figure); null where there is no such region.",
          "log10_mass_eV": rounded(d["log10_m"], FIGURE_DIGITS)}
    sf.update({k: rounded(v, FIGURE_DIGITS) for k, v in d.items() if k.startswith("si_")})
    write(os.path.join(OUT, "scalar_field_boundaries.json"), sf)
    print("side lines: cmb_lifetime.json, scalar_field_boundaries.json")


def existing_bounds() -> None:
    """data/existing_bounds.json: every existing bound the limit figures draw, model by model."""
    from analysis import existing_bounds as eb
    out = {"content": "The existing bounds drawn in gray in the limit figures (and the earlier estimate drawn "
                      "dashed), digitized from their sources: [model][bound] -> log10 of mass (eV) and coupling. "
                      "'region': excluded above the curve, or inside it for a closed one. Cite the source, "
                      "'inspire', when you use a bound.", "bounds": {}}
    for model in ["le_lmu", "le_ltau"] + SI_MODELS:
        for key in eb.for_model(model):
            b = eb.BOUNDS[key]
            if not b.get("draw") or b["kind"] != "curve":
                continue
            try:
                M, g = eb.load(key, model)
            except (OSError, KeyError, ValueError):
                continue
            entry = {"label": b["label"], "inspire": b.get("texkey"), "region": b.get("region", "above"),
                     "log10_mass_eV": rounded(np.log10(M), FIGURE_DIGITS),
                     "log10_coupling": rounded(np.log10(g), FIGURE_DIGITS)}
            if b.get("line"):
                entry["line"] = b["line"]
                entry["line_label"] = b.get("line_label")
            out["bounds"].setdefault(model, {})[key] = entry
    write(os.path.join(OUT, "existing_bounds.json"), out)
    print("existing bounds:", {m: len(v) for m, v in out["bounds"].items()})


# ------------------------------------------------------------------------------------------------ the figures

def enc(a) -> dict:
    """One array as stored: {"dtype", "values"} (floats to FIGURE_DIGITS significant digits, NaN as null)."""
    a = np.asarray(a)
    if a.dtype.kind in "USO":
        return {"dtype": "str", "values": a.astype(str).tolist()}
    if a.dtype.kind == "b":
        return {"dtype": "bool", "values": a.tolist()}
    if a.dtype.kind in "iu":
        return {"dtype": "int64", "values": a.tolist()}
    return {"dtype": "float64", "values": rounded(a.astype(float).tolist(), FIGURE_DIGITS)}


def figure_file(name: str, content: str, arrays: dict, **extra) -> None:
    """data/<name>.json: what the figure shows, and its arrays."""
    payload = {"paper": PAPER["title"], "figure": f"figures/{name}.pdf", "content": content, **extra,
               "arrays": {k: enc(v) for k, v in arrays.items()}}
    path = write(os.path.join(OUT, name + ".json"), payload)
    print(f"{os.path.basename(path)}: {len(arrays)} arrays, {os.path.getsize(path) / 1e3:.0f} kB")


def trip_illustration() -> None:
    rings = json.load(open(os.path.join(SOURCE, "tools", "data", "naturalearth", "land_110m_decimated.json")))
    payload = {"paper": PAPER["title"], "figure": "figures/trip_illustration.pdf",
               "content": "The land outlines drawn on the Earth: Natural Earth's 1:110m land polygons (public domain), "
                          "decimated; each ring a list of [longitude, latitude] in degrees.",
               "land_rings": rounded(rings, 6)}
    path = write(os.path.join(OUT, "trip_illustration.json"), payload)
    print(f"{os.path.basename(path)}: {len(rings)} rings")


def secondary_spectra() -> None:
    from analysis import mean_free_paths as mf
    from analysis import secondary_spectra as ss
    arrays = {"x": ss.X}
    for key, model, M, g in mf.EXAMPLES:
        arrays[f"{key}_keep"], arrays[f"{key}_flip"] = ss.decay_spectra(model, M, g)
    for k, v in ss.si_spectra().items():
        arrays[f"si_{k}"] = v
    figure_file("secondary_spectra", "dN/dx, the neutrinos one process returns per unit x = E_out / E_in, for an "
                "incident nu_3. le_* and dp_*: the daughter of a decay of nu_3 that keeps its helicity (keep, a "
                "neutrino) or flips it (flip, an antineutrino). si_*: one collision of nu_3 on a relic nu_3 at rest, "
                "at s = 10 M^2, each channel alone (s channel; t and u channels; pair production).", arrays)


def mean_free_paths() -> None:
    from analysis import mean_free_paths as mf
    from analysis import si_mean_free_path as si_mfp
    d = dict(np.load(mf.CACHE))
    arrays = {}
    for key, *_ in mf.EXAMPLES:
        for q in ("decay_mpc", "scatter_mpc", "goldstone_E_eV", "goldstone_mpc"):
            if f"{key}_{q}" in d:
                arrays[f"{key}_{q}"] = d[f"{key}_{q}"]
    E, rows, cols, _ = si_mfp.load(mf.SI_CACHE)
    arrays["si_E_eV"] = E
    for r in range(3):
        arrays[f"si_relic{r + 1}"] = rows["A"][r]
    arrays["si_columns"] = np.array(cols["A"])
    figure_file("mean_free_paths", "Mean free paths at z = 0, in Mpc, at the energies si_E_eV. le_* and dp_*: the "
                "decay lengths of nu_1, nu_2, nu_3 (decay_mpc, one column per state), the length of the scatterings "
                "on the relics that move each state to another (scatter_mpc), and, for L_e - L_mu, the collisions of "
                "the mediator's longitudinal mode at goldstone_E_eV (goldstone_mpc). si_relic1..3: the rates (s^-1) "
                "of nu_3 on each relic, channel by channel (si_columns), for the scalar of secret interactions.",
                arrays)


def mean_free_paths_by_relic() -> None:
    from analysis import si_mean_free_path as si_mfp
    E, rows, cols, masses = si_mfp.load(si_mfp.CACHE)
    arrays = {"E_eV": E, "masses_eV": masses}
    for k in rows:
        arrays[f"{k}_columns"] = np.array(cols[k])
        for r in range(3):
            arrays[f"{k}_relic{r + 1}"] = rows[k][r]
    figure_file("mean_free_paths_by_relic", "Rates (s^-1) of an incident nu_3 on the relic nu_1, nu_2, nu_3 at z = 0, "
                "channel by channel (<model>_columns), at the energies E_eV, for the four models of secret "
                "interactions: A, scalar with Majorana neutrinos; B, vector with Majorana; C, scalar with Dirac; D, "
                "vector with Dirac. M = 1 MeV, g = 1e-3, coupled to nu_tau. For C, the columns after the fifth are "
                "not collision channels.", arrays)


def species_along_trip() -> None:
    from analysis import trajectory_figure as tf
    d = dict(np.load(tf.CACHE))
    figure_file("species_along_trip", "The flux of each species along the trip, at an observed energy of 495 TeV, "
                "from sources in the redshift shell 1.8-2.2: <row>/universe [nu or nubar, redshift, flavor] at the "
                "redshifts <row>/z; then in the Milky Way (distance to the Sun, kpc), near the Sun (distance to the "
                "Earth, AU) and inside the Earth (km), each at its <stage>_x. Rows: si, dp, le_20.", d)


def three_classes() -> None:
    from analysis import mese, surfaces
    from analysis import three_classes_figure as tc
    FX = np.load(tc.CACHE)
    E = FX["E"]
    arrays = {"E": E}
    band = (E >= tc.FLAVOR_BAND_EV[0] * (1.0 - 1e-9)) & (E <= tc.FLAVOR_BAND_EV[1] * (1.0 + 1e-9))
    for key in ("le", "dp", "si"):
        steps = ["none"] + [k for k, _ in tc.PANELS[key][2]] + (["rest"] if key == "si" else [])
        for k in steps:
            if f"{key}_{k}" not in FX.files:
                continue
            arrays[f"{key}_{k}"] = FX[f"{key}_{k}"]
            arrays[f"composition/{key}_{k}"] = mese.event_weighted_fractions(E[band], FX[f"{key}_{k}"][band])
    g = np.linspace(0, 1, 201)
    FE, FM = np.meshgrid(g, g)
    C = np.full(FE.shape, np.nan)
    for i, j in zip(*np.where(FE + FM <= 1)):
        C[i, j] = surfaces.chi2(FE[i, j], FM[i, j], "icecube_mese_2025_grid")
    arrays["mese_grid"] = g
    arrays["mese_delta_chi2"] = C
    figure_file("three_classes", "Per column (le, dp, si) and step: the flux at the detector [energy E, flavor] "
                "(<column>_<step>; 'none' without a new interaction), and its flavor composition weighted by IceCube's "
                "MESE events over 5 TeV - 10 PeV (composition/<column>_<step>: f_e, f_mu, f_tau). mese_delta_chi2: "
                "IceCube's MESE flavor likelihood (Delta chi^2) on a grid of f_e (columns) and f_mu (rows), "
                "mese_grid.", arrays)


def event_rates() -> None:
    from analysis import event_rates_figure as erf
    arrays = {}
    for key in erf.KEYS:
        for sample, s in erf.load(key).items():
            for q, v in s.items():
                arrays[f"{key}/{sample}/{q}"] = v
    figure_file("event_rates", "Per example (le, dp, si) and sample (cascades: MESE starting cascades vs reconstructed "
                "energy; tracks: DR2 through-going tracks vs reconstructed energy; directions: DR2 tracks above 10 TeV "
                "vs declination): the bin edges (GeV or degrees), the observed events (data), and the expected events "
                "without (sm) and with (ex) the new interaction, in total and from the astrophysical flux (*_astro); "
                "the background is the total minus the astrophysical.", arrays)


def flux_stops() -> None:
    from analysis import method_figures as mfig
    d = mfig.load()
    keys = [k for k in d if k.split("/")[0] in [c[0] for c in mfig.TRIP_CASES]]
    figure_file("flux_stops", "L_e - L_mu at two masses: the flux [energy, flavor] at the stops of the trip (arriving "
                "at the Milky Way, after it, near the Earth, at the detector), at the energies <case>/E (eV).",
                {k: d[k] for k in keys})


def dirac_relic_helicity() -> None:
    z = np.load(os.path.join(SOURCE, "output", "analysis", "dirac_polarization.npz"))
    keep = [k for k in z.files if k.split("_")[0] in ("scalar", "vector")
            and k.rsplit("_", 1)[-1] in ("free", "rest", "unpolarized", "polarized")]
    figure_file("dirac_relic_helicity", "Dirac neutrinos, secret interactions: (energy in eV, all-flavor flux) at "
                "the detector, <mediator>_<point>_<m_1>_<treatment>: point 0 is M = 31.6 keV, g = 2e-3; point 1 is "
                "M = 4.22 MeV, g = 1.35e-3; m_1 in eV; treatment free (no interaction), rest (relics at rest), "
                "unpolarized (relics in motion, helicity ignored), polarized (helicity kept).",
                {k: z[k] for k in keep})


def preference() -> None:
    from analysis import preference_figure as pf
    pub = {ds: p for ds, p, _ in DATA_SETS}
    arrays = {}
    for ds, planes_ in pf.load().items():
        for plane, (x, p, g) in planes_.items():
            base = f"{pub[ds]}/{plane.replace('_fx', '')}"
            arrays[f"{base}/log10_mass_eV"], arrays[f"{base}/preference"], arrays[f"{base}/log10_coupling_best"] = \
                x, p, g
    figure_file("preference", "Per present-day data set and model, with the source composition fitted: at each mass, "
                "the chi^2 without a new interaction minus the lowest chi^2 over the coupling (preference), and the "
                "coupling where it is lowest.", arrays)


def provenance() -> None:
    from analysis import provenance_figure as pv
    rows, full, earlier, (slope, icpt) = pv.load()
    nan = lambda v: np.nan if v is None else v
    arrays = {"steps/log10_mass_eV": [r["log10_M"] for r in rows],
              "steps/earlier_method_our_data": [nan(r["detector_our_data"]) for r in rows],
              "steps/without_decays_or_collisions": [nan(r["no_decays"]) for r in rows],
              "this_work/log10_mass_eV": [c["log10_m"] for c in full],
              "this_work/lowest_excluded_log10_coupling": [nan(c["lowest_excluded_log10_g"]) for c in full],
              "earlier_estimate": earlier,
              "not_perturbative": [slope, icpt]}
    figure_file("provenance", "L_e - L_mu, pion-decay sources, IceCube MESE flavor + spectrum, 95% C.L.: the lowest "
                "coupling disfavored (log10) at each mass for the earlier estimate as published ([log10 M, log10 g'] "
                "pairs), for its method (the potential at the detector) with our data and fit, for our transport "
                "without decays or collisions on the relics, and for our full calculation; and the line of "
                "perturbative unitarity (slope, intercept).", arrays)


FIGURES = {"trip_illustration": trip_illustration, "secondary_spectra": secondary_spectra,
           "mean_free_paths": mean_free_paths, "mean_free_paths_by_relic": mean_free_paths_by_relic,
           "species_along_trip": species_along_trip, "three_classes": three_classes, "event_rates": event_rates,
           "flux_stops": flux_stops, "dirac_relic_helicity": dirac_relic_helicity, "preference": preference,
           "provenance": provenance}
TASKS = {"limits": limits, "side_lines": side_lines, "existing_bounds": existing_bounds, **FIGURES}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(TASKS)):
        TASKS[name]()
