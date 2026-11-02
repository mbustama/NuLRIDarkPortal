"""
check_si_new_physics.py -- whether the 2026-10-24 processes (notes Sec. 12.8) act in the secret-interaction planes
(m_phi 1 keV - 1 GeV, g_phi 1e-6 - 1): decay into the mediator needs m_phi < m_i - m_j <= 0.05 eV (closed); the
models' range (vector: the longitudinal coupling g Delta m / m_phi <= 5e-5 at 1 keV; scalar: the Yukawa g <= 1); the
relic mass-change energy loss (vector only: the scalar's chirality-flipping vertices give |M|^2 ~ t^2, no forward
peak). Printed: per texture, the smallest vector coupling at which the loss reaches 1% of the energy at 100 TeV, an
upper bound (the whole path from z = 4, no screening, lightest mass 1e-6 eV, the largest beam factor).
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

import numpy as np

import dp_relic_loss as rl
import dp_si
from dp_core import construct_PMNS

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def main():
    U = construct_PMNS(np.radians(33.76), np.radians(8.62), np.radians(43.28), np.radians(212.0))
    m = np.array([1e-6, 8.68e-3, 0.0501])
    H0, Om, OL = 1.4377e-33, 0.315, 0.685
    zz = np.linspace(0, 4, 401)
    Hz = H0 * np.sqrt(Om * (1 + zz) ** 3 + OL)
    print(f"longitudinal coupling at g = 1, m_phi = 1 keV: {1.0 * (m[2] - m[0]) / 1e3:.1e}")
    for tex in ("e", "mu", "tau", "e+mu", "e+tau", "mu+tau", "universal"):
        G = dp_si.flavor_couplings(tex, 1.0).astype(complex)
        Gm = U.conj().T @ G @ U
        b = float(np.max(np.real(np.diag(Gm @ Gm.conj().T))))
        row = []
        for M in (1e3, 1e4, 1e5, 1e6):
            C = np.array([rl.loss_coefficient(G, U, m, np.ones(3), 1.676389e-4, z, M, 0.0) for z in zz])
            dE1 = b * _trapz(C / ((1 + zz) ** 2 * Hz), zz) / 1e14
            row.append(f"{M:.0e} eV: g >= {(1e-2 / dE1) ** 0.25 if dE1 > 0 else np.inf:.2g}")
        print(f"vector {tex:9s}: 1% loss at 100 TeV for " + ", ".join(row))


if __name__ == "__main__":
    main()
