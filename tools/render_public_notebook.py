"""Write the public companion repository's notebook (public_repo/notebooks/figures.ipynb).

The limit figures' cells are the paper notebook's (notebooks/paper_figures.ipynb), with their data read from the
companion's data/ instead of the limits export; every other figure is drawn by its module in public_repo/plotting/.
Each replacement is checked, so that a change in the paper notebook stops this script instead of slipping through.

    python3 tools/render_public_notebook.py
"""

from __future__ import annotations

import json
import os
import re

import nbformat

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.environ.get("NULRI_SOURCE", HERE)
OUT = os.path.join(HERE, "public_repo", "notebooks", "figures.ipynb")


def paper_cells() -> list:
    nb = json.load(open(os.path.join(SOURCE, "notebooks", "paper_figures.ipynb")))
    return ["".join(c["source"]) for c in nb["cells"]]


def cell_with(cells, text: str) -> str:
    hits = [c for c in cells if text in c]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} cells hold {text!r}")
    return hits[0]


def swap(code: str, old: str, new: str, regex: bool = False, count: int = 0) -> str:
    """Replace old with new in code; stop if old is not there."""
    if regex:
        out, n = re.subn(old, new, code)
    else:
        n = code.count(old)
        out = code.replace(old, new)
    if n == 0 or (count and n != count):
        raise ValueError(f"{n} matches for {old!r}")
    return out


def data_access(code: str) -> str:
    """The limit cells' reads of the export, as reads of data/."""
    code = swap(code, "    suffix = COMPOSITIONS[composition][0]\n", "")
    code = swap(code, r'plane\("(?:le_ul|dp_ul)", ds, f"\{model\}_ul\{suffix\}"\)',
                'data.plane(model, "ultralight", ds, composition)', regex=True)
    code = swap(code, r'plane\("(?:le_ul|dp_ul)", "now_icecube", f"\{model\}_ul\{suffix\}"\)',
                'data.plane(model, "ultralight", "icecube_mese_flavor_spectrum", composition)', regex=True)
    code = swap(code, r'plane\("mev", ds, f"\{model\}_mev\{suffix\}"\)', 'data.plane(model, "mev", ds, composition)',
                regex=True)
    code = swap(code, "lx, G, F, _, _ = ex.field(mev)", "lx, G, F, _ = ex.field(mev)")
    code = swap(code, "ex.hatched_line(ex.not_modelled_lines(p))", "ex.hatched_line(p)")
    return code


SETUP = r'''import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = os.path.abspath("..")
sys.path.insert(0, REPO)
from plotting import data
from plotting import limits as ex
from plotting.style import DATASETS, PANEL, pct, ticks, tint, use

use()
#: Where the figures go.
FIGURES = os.path.join(REPO, "figures")
os.makedirs(FIGURES, exist_ok=True)
#: The source compositions: how the legend names each.
COMPOSITIONS = {"pion": r"fixing $f_{e,{\rm S}} = 1/3$", "fitted": r"fitting $f_{e,{\rm S}}$"}
#: The regions are drawn largest first, so that the smaller ones sit on top (the fills are opaque).
DRAWN = DATASETS[::-1]
#: The existing bounds, model by model, and the CMB bound on the neutrino lifetime as a bound on the coupling.
BOUNDS = data.load("existing_bounds")["bounds"]
CMB = data.load("cmb_lifetime")


def save(fig, name, png=False, **kw):
    """Writes figures/<name>.pdf (and .png if png); kw go to savefig (e.g. bbox_inches="tight")."""
    fig.savefig(os.path.join(FIGURES, f"{name}.pdf"), **kw)
    if png:
        fig.savefig(os.path.join(FIGURES, f"{name}.png"), dpi=150, **kw)
    print("saved figures/" + name + ".pdf")


def legend_title(composition):
    return f"Disfavored (95{pct()} C.L.), " + COMPOSITIONS[composition]


def handles():
    return [Patch(facecolor=tint(c), edgecolor=c, lw=1.5, label=label) for _, label, c in DATASETS]


def model_label(ax, text, size=22, where="top"):
    """The model's name in a rounded box, top left (or bottom left), above any hatching."""
    y, va = (0.94, "top") if where == "top" else (0.05, "bottom")
    ax.text(0.015, y, text, transform=ax.transAxes, ha="left", va=va, fontsize=size, zorder=20,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.25", lw=1.2))


def bounds(ax, model, dashed_label_xy=None):
    """The existing bounds of `model`: one light-gray region above the lowest of their curves at each mass, except
    the earlier estimate (the potential at the detector only), drawn as a dashed line labeled in place."""
    xs = np.linspace(-31, 10, 1200)
    low = np.full(len(xs), np.inf)
    for b in BOUNDS.get(model, {}).values():
        lx, lg = np.array(b["log10_mass_eV"]), np.array(b["log10_coupling"])
        o = np.argsort(lx)
        if b.get("line") == "dashed":
            ax.plot(lx[o], lg[o], color="0.35", lw=1.4, ls=(0, (5, 3)), zorder=7)
            if dashed_label_xy:
                ax.text(*dashed_label_xy, b["line_label"], fontsize=14, color="0.3", ha="center", va="top", zorder=7)
            continue
        inside = (xs >= lx[o][0]) & (xs <= lx[o][-1])
        low[inside] = np.minimum(low[inside], np.interp(xs[inside], lx[o], lg[o]))
    ok = np.isfinite(low)
    ax.fill_between(xs, np.where(ok, low, np.nan), 5.0, where=ok, color="0.88", lw=0, zorder=0)
    ax.plot(xs, np.where(ok, low, np.nan), color="0.55", lw=1.0, zorder=0.5)


def cmb_line(ax, family, model, label_at=None, label="CMB, neutrino lifetime"):
    """The CMB bound on the neutrino lifetime, from our decay rates (standard cosmology: the relic neutrinos present
    at recombination), as a dotted line where a decay is open; excluded above it. Labeled along the line at
    log10 M = label_at. family: "le" or "dp"."""
    pts = [(x, y) for x, y in zip(CMB["log10_mass_eV"][family], CMB[model]) if y is not None]
    if not pts:
        return
    x, y = np.array(pts).T
    ax.plot(x, y, color="0.2", lw=1.6, ls=(0, (1, 1.6)), zorder=8)
    if label_at is not None:
        y0 = float(np.interp(label_at, x, y))
        p1, p2 = ax.transData.transform([(label_at, y0), (label_at + 1.0, y0 + 1.0)])
        ang = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))
        ax.text(label_at, y0 + 0.6, label, fontsize=14, color="0.15", rotation=ang, rotation_mode="anchor",
                ha="center", va="bottom", zorder=8, clip_on=True)'''


SI_BOUNDS = r'''def si_bounds(ax, model):
    """The existing bounds of a secret-interaction panel as one gray region: above each curve (inside a closed one);
    there are bounds for the scalar with Majorana neutrinos only."""
    xs = np.linspace(-31, 10, 2000)
    low = np.full(len(xs), np.inf)
    for b in BOUNDS.get(model, {}).values():
        lx, lg = np.array(b["log10_mass_eV"]), np.array(b["log10_coupling"])
        if b.get("region") == "inside":
            ax.fill(lx, lg, color="0.88", lw=0, zorder=0)            # a closed contour, excluded inside
            ax.plot(np.r_[lx, lx[0]], np.r_[lg, lg[0]], color="0.55", lw=0.8, zorder=0.5)
        else:
            o = np.argsort(lx)
            inside = (xs >= lx[o][0]) & (xs <= lx[o][-1])
            low[inside] = np.minimum(low[inside], np.interp(xs[inside], lx[o], lg[o]))
    ok = np.isfinite(low)
    if ok.any():
        ax.fill_between(xs, np.where(ok, low, np.nan), 5.0, where=ok, color="0.88", lw=0, zorder=0)
        ax.plot(xs, np.where(ok, low, np.nan), color="0.55", lw=0.8, zorder=0.5)'''


def limit_cells(cells):
    le = data_access(cell_with(cells, "def limits_lmu_ltau("))
    le = swap(le, 'cmb_line(ax, os.path.join(LIMITS, "nulri_set1_sm", "campaign_final_sm", "cmb_lifetime.json"), model,',
              'cmb_line(ax, "le", model,')
    dp = data_access(cell_with(cells, "def limits_dark_portal("))
    dp = swap(dp, r'#: The CMB bound on the neutrino lifetime as a bound on g_D[^\n]*\nDP_CMB = [^\n]*\n', '', regex=True)
    dp = swap(dp, "cmb_line(ax, DP_CMB, model,", 'cmb_line(ax, "dp", model,')
    dp = swap(dp, r"      # down to just below the smallest coupling computed \(his fix, 2026-11-22\)",
              "      # down to just below the smallest coupling computed", regex=True)
    si = cell_with(cells, "def limits_secret_interactions(")
    si = swap(si, si[si.index("def si_bounds("):si.index("\n\n\nSCALAR_FIELD")], SI_BOUNDS)
    si = swap(si, re.search(r"SCALAR_FIELD = json\.load\([^\n]*\n", si).group(0),
              'SCALAR_FIELD = data.load("scalar_field_boundaries")\n')
    si = swap(si, 'SCALAR_FIELD["log10_m"]', 'SCALAR_FIELD["log10_mass_eV"]')
    si = swap(si, "    suffix = COMPOSITIONS[composition][0]\n", "")
    si = swap(si, '                w = "si_proj" if ds.startswith("proj") else "si_now"\n', "")
    si = swap(si, 'ul, mev = plane(w, ds, f"{model}_ul{suffix}"), plane(w, ds, f"{model}{suffix}")',
              'ul, mev = data.plane(model, "ultralight", ds, composition), data.plane(model, "mev", ds, composition)')
    si = swap(si, re.search(r'                if ds == "now_mese_dr2":\n[^\n]*\n', si).group(0), "")
    si = swap(si, "lx, G, F, _, _ = ex.field(mev)", "lx, G, F, _ = ex.field(mev)")
    si = swap(si, 'ex.hatched_line(ex.not_modelled_lines(plane("si_now", "now_icecube", f"{model}_ul{suffix}")))',
              'ex.hatched_line(data.plane(model, "ultralight", "icecube_mese_flavor_spectrum", composition))')
    si = swap(si, re.search(r'    """Fig\. 11 \(his layout requests[^"]*"""', si).group(0),
              '    """Fig. 11: four columns (the mediator and the neutrinos) by four rows (the flavor it couples to); column '
              'and\n    row titles in rounded boxes outside the panels, common axis labels, one legend across the top."""')
    si = swap(si, "    # the panels placed by hand (his request): the gap", "    # the panels placed by hand: the gap")
    si = swap(si, "    # titles and axis labels outside the panels, placed on the drawn figure (as in Fig. F.1)",
              "    # titles and axis labels outside the panels, placed on the drawn figure")
    hel = cell_with(cells, "def dirac_helicity(")
    hel = swap(hel, re.search(r"#: The data \(tools/dirac_relic_polarization\.py\)\.\n(?:[^\n]*\n){3}", hel).group(0), "")
    hel = swap(hel, "#: (Not the data sets' colors, which every limit figure reserves; 2026-11-07.)\n",
               "#: (Not the data sets' colors, which the limit figures use.)\n")
    hel = swap(hel, "    z = np.load(POLARIZATION)\n", '    z = data.arrays("dirac_relic_helicity")\n')
    return le, dp, si, hel


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbformat.v4.new_code_cell(text.strip("\n"))


def module_cell(module, name, call="figure()", tight=False, png=False):
    kw = (", png=True" if png else "") + (', bbox_inches="tight", pad_inches=0.05' if tight else "")
    return code(f"from plotting import {module}\n\nfig = {module}.{call}\nsave(fig, \"{name}\"{kw})\nplt.show()")


def main() -> int:
    cells = paper_cells()
    le, dp, si, hel = limit_cells(cells)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    C = nb.cells
    C.append(md(r"""
# New interactions of high-energy cosmic neutrinos, from the Hubble scale to the Earth

M. Bustamante, S. Das, and S. K. Agarwalla.

This notebook redraws every figure of the paper from the files in `data/`, with the paper's own plotting code, in
`plotting/`. Each section describes one figure, names the files it reads, and draws it; the figure is saved to
`figures/` as a PDF. Run the cells in order, from this folder. The notebook needs neither the transport code nor the
fits that produced the numbers.

The figures set their text with LaTeX in Palatino, as in the paper. Without a LaTeX installation, matplotlib's own
text rendering is used instead, and the figures look slightly different.
"""))
    C.append(md(r"""
## Setup

The paper's figure style; the folder for the figures; and the helpers that the limit figures (Figs. 9-11) share:
the data sets' colors and legend, the existing bounds drawn in gray, and the CMB bound on the neutrino lifetime,
drawn as a dotted line.

Three data sets appear in the limit figures, each in its own color:

* **Today: IceCube MESE flavor + spectrum.** IceCube's published flavor composition, measured with its medium-energy
  starting events (MESE), and its measured spectrum.
* **Today: IceCube MESE events + DR2 tracks.** Our own fit to IceCube's MESE events and to its through-going tracks
  (IceTracks-DR2). Used for the MeV mediators only.
* **Projected: eight telescopes, 10 yr each.** A forecast with simulated data from eight future neutrino telescopes.

Two choices of the flavor composition at the sources: fixed to that of pion decay, $\nu_e:\nu_\mu:\nu_\tau =
1:2:0$ (`pion`), or fitted (`fitted`). The paper shows `fitted`.
"""))
    C.append(code(SETUP))
    C.append(md(r"""
## Figure 1: the trip, an artist's rendition

The trip of high-energy astrophysical neutrinos, in four panels that each zoom in on the next: the Universe, the
Milky Way, near the Sun, and inside the Earth, with callouts on what happens along the way. It is a drawing, not to
scale; the continents are Natural Earth's land outlines. The PNG copy is the picture at the top of the README.

Code: `plotting/trip_illustration.py`. Data: `data/trip_illustration.json`.
"""))
    C.append(module_cell("trip_illustration", "trip_illustration", png=True))
    C.append(md(r"""
## Figures 2 and 3: the Feynman diagrams

Figure 2 shows how the mediator of each model couples to neutrinos. Figure 3 shows every process that the
calculation includes, built from those vertices, for a vector mediator and for the scalar. The relic neutrinos are
drawn in gray.

Code: `plotting/diagrams.py`. No data.
"""))
    C.append(code("from plotting import diagrams\n\nfig = diagrams.vertices()\nsave(fig, \"feynman_vertices\")\nplt.show()\n\n"
                  "fig = diagrams.processes()\nsave(fig, \"feynman_processes\")\nplt.show()"))
    C.append(md(r"""
## Figure 4: what each process returns

The energy spectra of the neutrinos that each process returns, for one example of each family, with an incident
$\nu_3$. For $L_e - L_\mu$ and the dark portal: the daughter of a decay of $\nu_3$, split into the daughter that
keeps its helicity (a neutrino) and the one that flips it (an antineutrino); the scatterings on the relics keep the
energy (the arrow). For secret interactions: the products of one collision on a relic $\nu_3$ at rest, at an
incident energy of 10 PeV, channel by channel. The area under each curve is the number of neutrinos per process.

Code: `plotting/secondary_spectra.py`. Data: `data/secondary_spectra.json`.
"""))
    C.append(module_cell("secondary_spectra", "secondary_spectra", tight=True))
    C.append(md(r"""
## Figure 5: the mean free paths

The mean free paths of high-energy neutrinos at $z = 0$, for the examples of Fig. 4. For $L_e - L_\mu$ and the dark
portal: the decay lengths of $\nu_2$ and $\nu_3$, and the scatterings of $\nu_3$ on the relics. For secret
interactions: the mean free path of $\nu_3$ on the relics, channel by channel. The dashed line is the Hubble length.

Code: `plotting/mean_free_paths.py`. Data: `data/mean_free_paths.json`.
"""))
    C.append(module_cell("mean_free_paths", "mean_free_paths", tight=True))
    C.append(md(r"""
## Figure 6: the six species along the trip

Where along the trip each family of mediators acts: the flux of each species, neutrinos and antineutrinos of each
flavor, at an observed energy of 495 TeV, from sources at $z = 1.8$-$2.2$ to the detector, divided by the total once
the sources stop emitting. The columns follow the panels of Fig. 1.

Code: `plotting/species_along_trip.py`. Data: `data/species_along_trip.json`.
"""))
    C.append(module_cell("species_along_trip", "species_along_trip"))
    C.append(md(r"""
## Figure 7: what each ingredient of the calculation does

One example per family of mediators, each disfavored at 95% C.L. by IceCube's flavor composition and spectrum. Each
numbered point, and the line of its color below, adds one ingredient to those before it. Top: the flavor composition,
averaged over 5 TeV-10 PeV with each flavor weighted by its expected number of MESE events, against IceCube's
measurement with these events (68% and 95% C.L.). Bottom: the flux at the detector relative to no new interaction.

Code: `plotting/three_classes.py`. Data: `data/three_classes.json`.
"""))
    C.append(module_cell("three_classes", "three_classes"))
    C.append(md(r"""
## Figure 8: IceCube's event rates

IceCube's event rates with and without a new interaction, for the examples of Fig. 7: the starting cascades and the
through-going tracks against reconstructed energy, and the tracks above 10 TeV against declination. Sources and
backgrounds are held at their values without a new interaction.

Code: `plotting/event_rates.py`. Data: `data/event_rates.json`.
"""))
    C.append(module_cell("event_rates", "event_rates", tight=True))
    C.append(md(r"""
## How a limit is read (Figs. 9-11)

Each plane is a grid of fits: at each mediator mass and coupling, the $\chi^2$ is minimized over every other
parameter (`data/chi2/`). At each mass, a coupling is disfavored at 95% C.L. where $\Delta\chi^2 \geq 3.84$, counted
from the lowest $\chi^2$ at that mass, or from the fit without a new interaction where that is lower. Only upper
limits are read: couplings below the lowest allowed one are not counted as excluded. Isolated specks, of at most
three grid points and none above $\Delta\chi^2 = 5$, are dropped. Between the grid's couplings, $\Delta\chi^2$ is
interpolated with a monotone cubic.

Each model has two grids, one of ultralight masses and one of MeV masses; the figures join them. The thin excluded
bands of the ultralight grids are left out of the figures; `data/limits.json` lists them under
`thin_bands_not_shown`. `plotting/limits.py` does all of this, and `data/limits.json` holds the result, mass by mass.

Each cell below draws its figure twice: with the source composition fitted (`_fitted`, the paper's figure) and with
pion-decay sources (`_pion`).
"""))
    C.append(md(r"""
## Figure 9: limits on $L_e - L_\mu$ and $L_e - L_\tau$

Limits on the coupling $g'$ of the $L_e - L_\mu$ (top) and $L_e - L_\tau$ (bottom) gauge bosons, against their mass
$M$. Each shaded region is disfavored at 95% C.L. by one data set. The insets enlarge the MeV masses. Existing bounds
are in gray; the dashed line is an earlier estimate that evaluated the potential only at the detector. The dotted line
is the CMB bound on the neutrino lifetime, which excludes the couplings above it if the relic neutrinos are present
at recombination. The region beyond perturbative unitarity is hatched.

Data: `data/chi2/le_lmu__*.json`, `data/chi2/le_ltau__*.json`, `data/existing_bounds.json`, `data/cmb_lifetime.json`.
"""))
    C.append(code(le))
    C.append(md(r"""
## Figure 10: limits on the dark portal

Limits on the coupling of the dark-portal mediator, $g_D |U_{\alpha 4}|^2$, against its mass, for the mediator coupled
to $\nu_e$, $\nu_\mu$, and $\nu_\tau$ (top to bottom). Below $10^{-4}$ eV, the regions reach the smallest coupling
computed, $10^{-8}$, so the limits lie lower. The hatched band, above $g_D/M = 10^{-2}$ eV$^{-1}$, is where the
scatterings of the mediator's longitudinal mode are left out, and the paper makes no claim.

Data: `data/chi2/dp_*.json`, `data/cmb_lifetime.json`.
"""))
    C.append(code(dp))
    C.append(md(r"""
## Figure 11: limits on secret interactions

Limits on secret neutrino interactions, against the mass of the mediator. Columns: a scalar or a vector mediator,
with Majorana or Dirac neutrinos. Rows: the mediator coupled to $\nu_e$, $\nu_\mu$, $\nu_\tau$, or all flavors. For
the scalar, the hatched region is where the field that the relic neutrinos source changes the neutrino masses, which
the paper does not model; for the vector, the hatched band, above $g/M = 10^{-2}$ eV$^{-1}$, is where the scatterings
of its longitudinal mode are left out. Existing bounds, available for the scalar with Majorana neutrinos, are in gray.

Data: `data/chi2/si_*.json`, `data/existing_bounds.json`, `data/scalar_field_boundaries.json`.
"""))
    C.append(code(si))
    C.append(md(r"""
## Figure 12: IceCube's preference for an MeV mediator

How much today's IceCube data prefer a new mediator to no new interaction, against its mass: at each mass, the
$\chi^2$ without a new interaction minus the lowest $\chi^2$ over the coupling, with the source composition fitted.
One example per family; one color per data set.

Code: `plotting/preference.py`. Data: `data/preference.json`.
"""))
    C.append(module_cell("preference", "preference", tight=True))
    C.append(md(r"""
## Figure C.1: the flux at each stop of the trip

For $L_e - L_\mu$ at two masses: the flux of all flavors relative to its value on arrival at the Milky Way (top),
and the fraction of each flavor (below), at four stops: arriving at the Milky Way, after it, near the Earth, and at
the detector.

Code: `plotting/flux_stops.py`. Data: `data/flux_stops.json`.
"""))
    C.append(module_cell("flux_stops", "flux_stops"))
    C.append(md(r"""
## Figure D.1: the helicity of Dirac relic neutrinos

The flux of high-energy neutrinos with secret interactions on Dirac relic neutrinos, relative to that without a new
interaction, for a mediator coupled to $\nu_e$: with the relics at rest, in motion with their helicity ignored, and in
motion with their helicity kept. Thick lines: lightest neutrino mass $10^{-6}$ eV; thin lines: 0.03 eV.

Data: `data/dirac_relic_helicity.json`.
"""))
    C.append(code(hel))
    C.append(md(r"""
## Figure F.1: the mean free path of $\nu_3$ on each relic, secret interactions

The mean free path of a high-energy $\nu_3$ on each relic neutrino, and on all three, for the four models of secret
interactions, channel by channel, with the motion of the relics included in every channel.

Code: `plotting/mean_free_paths_by_relic.py`. Data: `data/mean_free_paths_by_relic.json`.
"""))
    C.append(module_cell("mean_free_paths_by_relic", "mean_free_paths_by_relic", tight=True))
    C.append(md(r"""
## Figure I.1: where the earlier estimate and our limits on $L_e - L_\mu$ part

Limits at 95% C.L. for pion-decay sources and IceCube's flavor composition and spectrum, from the earlier estimate to
ours, one ingredient at a time: the earlier estimate as published; its method (the potential at the detector only)
with our data and fit; our transport without decays or collisions on the relics; and our full calculation.

Code: `plotting/provenance.py`. Data: `data/provenance.json`.
"""))
    C.append(module_cell("provenance", "provenance", tight=True))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nbformat.write(nb, OUT)
    print("wrote", OUT, len(C), "cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
