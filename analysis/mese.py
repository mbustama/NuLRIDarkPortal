"""
mese.py

IceCube's MESE release (arXiv:2510.24957; Harvard Dataverse doi:10.7910/DVN/CBNMEB v2.0, CC0), files in
IceCubeRates/data/mese_2025/ (README there, MD5s checked).

flavor_area(E_eV): the all-sky effective area per flavor, summed over the three event types (cascades, double
cascades, tracks) and over the interactions (CC, NC, and the Glashow resonance for nu_e), in true neutrino energy.
The release gives each area in 10 bins of cos(zenith) of equal width, so equal solid angle: the all-sky value is
their mean. Its energies (1 TeV - 10 PeV; double cascades from 10 TeV) are read as nodes, interpolated log-log;
outside them the area is 0. Only the cascades carry a Glashow-resonance column.
The release's units are not needed here: the weighting below uses each flavor's area relative to itself.

Audit decision A3 (2026-11-09): IceCube's flavor fit assumes one composition at every energy. Our composition
changes with energy, so the composition compared with IceCube's likelihood is the constant one that gives the same
expected MESE events per flavor as our flux, f_a ~ int Phi_a A_a dE / int Phi A_a dE (Phi the all-flavor flux),
normalized to sum 1. It weights each energy as MESE's events do, flavor by flavor.
"""
import os
from functools import lru_cache

import numpy as np

from . import REPO_DIR

DATA = os.path.join(REPO_DIR, "IceCubeRates", "data", "mese_2025")
AREA_FILES = ("EffectiveArea_cascades.tab", "EffectiveArea_double_cascades.tab", "EffectiveArea_tracks.tab")
FLAVOR_COLUMNS = (("nue_cc", "nue_nc", "nue_gr"), ("numu_cc", "numu_nc"), ("nutau_cc", "nutau_nc"))


@lru_cache(maxsize=None)
def _area_nodes():
    """Per event type: (E nodes in GeV, all-sky area per flavor (N, 3)), summed over the interactions. The files'
    energies differ (double cascades start at 10 TeV); only the cascades carry the Glashow column."""
    out = []
    for fn in AREA_FILES:
        with open(os.path.join(DATA, fn)) as f:
            head = f.readline().strip().split(",")
            rows = np.array([[float(x) for x in line.split(",")] for line in f if line.strip()])
        col = {h: k for k, h in enumerate(head)}
        E = np.unique(rows[:, col["Energy"]])
        zen = np.unique(rows[:, col["Zenith_bin"]])
        if len(zen) != 10:
            raise ValueError(f"{fn}: expected 10 zenith bins, found {len(zen)}")
        A = np.zeros((len(E), 3))
        for a, cols in enumerate(FLAVOR_COLUMNS):
            for c in cols:
                if c not in col:
                    continue
                for k, e in enumerate(E):
                    sel = rows[:, col["Energy"]] == e
                    if sel.sum() != 10:
                        raise ValueError(f"{fn}: {sel.sum()} zenith rows at {e} GeV")
                    A[k, a] += rows[sel, col[c]].mean()               # all-sky: equal solid angle per bin
        out.append((E, A))
    return tuple(out)


def _interp(x, xn, yn):
    """yn at x: log-log between two positive nodes, linear in log E where a node is 0; 0 outside the nodes."""
    out = np.zeros_like(x)
    inside = (x >= xn[0] - 1e-9) & (x <= xn[-1] + 1e-9)
    xi = np.clip(x[inside], xn[0], xn[-1])
    k = np.clip(np.searchsorted(xn, xi) - 1, 0, len(xn) - 2)
    t = (xi - xn[k]) / (xn[k + 1] - xn[k])
    y0, y1 = yn[k], yn[k + 1]
    pos = (y0 > 0) & (y1 > 0)
    out[inside] = np.where(pos, np.exp((1 - t) * np.log(np.where(pos, y0, 1.0)) + t * np.log(np.where(pos, y1, 1.0))),
                           (1 - t) * y0 + t * y1)
    return out


def flavor_area(E_eV) -> np.ndarray:
    """The all-sky MESE area per flavor at energies E_eV (N_E,) -> (N_E, 3): the sum over the event types, each
    interpolated between its own nodes (log-log; linear in log E next to a zero) and 0 outside them."""
    x = np.log(np.asarray(E_eV, dtype=np.float64) / 1e9)
    out = np.zeros((len(x), 3))
    for En, A in _area_nodes():
        for a in range(3):
            out[:, a] += _interp(x, np.log(En), A[:, a])
    return out


def event_weighted_fractions(E_eV, F) -> np.ndarray:
    """(f_e, f_mu, f_tau): the constant composition giving the same expected MESE events per flavor as the flux F
    (N_E, 3) per flavor on the energies E_eV (audit decision A3), f_a ~ int F_a A_a dE / int (sum_b F_b) A_a dE,
    normalized. The energy integral uses the solver's bin widths (compute_dE_geometric)."""
    from dark_portal_solver import compute_dE_geometric
    E = np.asarray(E_eV, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    w = flavor_area(E) * compute_dE_geometric(E)[:, None]
    num = np.sum(F * w, axis=0)
    den = np.sum(F.sum(axis=1)[:, None] * w, axis=0)
    if np.any(den <= 0.0):
        raise ValueError("No MESE area under the flux in this energy window")
    f = num / den
    return f / f.sum()
