"""
galactic_adiabaticity.py -- test T9 of docs/dev/DIRECTIONS_PLAN.md: is the transit of Galactic neutrinos from
their emission points to the Sun adiabatic, as the emission populations assume (dp_galactic)?

Along a pixel's line of sight the potential follows the Yukawa-smoothed density n_eff(x; m). The eigenbasis of
H = (M_rot + sigma S) / 2E, sigma = 2 |Coef| E n_eff, turns between the frame table's nodes by K = log(W_n^dag
W_n+1) (dp_transit.build_frame_table), while the phases of each pair advance at |nu_i - nu_j| / 2E per unit
length. The adiabaticity parameter of a pair along a segment of the path,

    gamma_ij = (|nu_i - nu_j| / 2E) / (|K_ij| / dln(sigma) x |dln n_eff / ds|),

must be >> 1 (at a level crossing the Landau-Zener jump probability is exp(-pi gamma / 2); Parke, PRL 57, 1275
(1986)). For each point (m_Zd, the potential at the Sun kappa times Delta m^2_31 / 2E at 100 TeV, e-mu texture)
and energy, the emission-weighted fraction of each pixel's flux whose path to the Sun has min gamma below 1 and
below 10, and the smallest gamma, are reported for the diffuse and the sources' emission distances. Both kinds.

  python tools/galactic_adiabaticity.py          (light: N_side 16 pixels x 97 distance bins)

Writes output/sky/galactic_adiabaticity.json.
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dark_portal_solver as dps  # noqa: E402
import dp_galactic as G  # noqa: E402
import dp_transit  # noqa: E402
from dp_core import njit, m_to_eV_inv  # noqa: E402
from dp_spatial import lri_coefficient  # noqa: E402

KPC_EV_INV = 3.08567758e19 * m_to_eV_inv
GU2_PER_KAPPA = 1.66e-54          # g_D eps' |u|^2 whose Sun potential at 1 AU is Delta m^2_31 / (2 x 100 TeV)


@njit(cache=True)
def _min_gamma(lnn, s, lnsig_nodes, ratio, lnsig0, E, out):
    """out[p, j]: min over the path from bin j to the Sun of gamma (see the module notes); ratio[n] is
    max_ij |K_n,ij| / dln(sigma)_n / |nu_i - nu_j| on the frame table's intervals."""
    npix, ns = lnn.shape
    for p in range(npix):
        run = np.inf
        out[p, 0] = np.inf
        for j in range(1, ns):
            grad = abs(lnn[p, j] - lnn[p, j - 1]) / (s[j] - s[j - 1])     # per kpc
            if grad > 0.0:
                a, b = lnsig0 + lnn[p, j - 1], lnsig0 + lnn[p, j]
                lo, hi = min(a, b), max(a, b)
                ia = max(np.searchsorted(lnsig_nodes, lo) - 1, 0)
                ib = min(max(np.searchsorted(lnsig_nodes, hi) - 1, ia), len(ratio) - 1)
                r = 0.0
                for n in range(ia, ib + 1):
                    r = max(r, ratio[n])
                if r > 0.0:
                    g = KPC_EV_INV / (2.0 * E * r * grad)
                    run = min(run, g)
            out[p, j] = run


def frame_ratios(tab):
    """max over pairs of |K_ij| / dln(sigma) / |nu_i - nu_j| (eV^-2) per interval of the table."""
    sig, nu, K = tab["sig"], tab["nu"], tab["K"]
    if len(sig) < 2:
        return np.array([0.0]), np.array([0.0])
    dl = np.diff(np.log(sig))
    r = np.zeros(len(dl))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        gap = np.minimum(np.abs(nu[:-1, i] - nu[:-1, j]), np.abs(nu[1:, i] - nu[1:, j]))
        r = np.maximum(r, np.abs(K[:, i, j]) / dl / np.maximum(gap, 1e-300))
    return np.log(sig), r


def run_point(m_zd, kappa, energies, nside=16):
    cfg = {"dark_portal": {"u_e": 0.5, "u_mu": 0.5, "u_tau": 0.0, "m_Zd_eV": m_zd,
                           "g_D": kappa * GU2_PER_KAPPA / (0.1 * 0.5), "epsilon_prime": 0.1},
           "execution": {"n_cores": 1, "verbose": 0}}
    E_bins, params = dps.build_solver_params(cfg)
    setup = dps._flux_setup(E_bins[:2], params)
    q_gas, q_src = G.emission_weights(nside)
    s, ds, R, z = G.emission_geometry(nside)
    rows = []
    for anti in (False, True):
        u_s, M_rot, S_eff, _ = dps._species_matrices(setup['U_pmns_base'], setup['M_sq'], setup['u_arr_base'],
                                                     None, None, anti)
        Coef, m_eff = lri_coefficient(m_zd, params['g_D'], params['epsilon_prime'], u_s, params['T_nu0_eV'],
                                      params['use_debye_screening'], setup['eta_bg'], anti)
        S_signed = np.ascontiguousarray(np.sign(Coef) * S_eff if Coef != 0.0 else S_eff)
        tab = dp_transit.frame_table(M_rot, S_signed, params['transit_dtheta'])
        lnsig_nodes, ratio = frame_ratios(tab)
        n = G.effective_density(R, z, m_eff, params['n_n0_eV3'], 1.0 / params['H0_eV'])
        lnn = np.log(n)
        for E in energies:
            g = np.empty(lnn.shape)
            _min_gamma(lnn, s, lnsig_nodes, ratio, np.log(2.0 * abs(Coef) * E), E, g)
            row = dict(m_Zd_eV=m_zd, kappa=kappa, kind="nubar" if anti else "nu", E_eV=E,
                       min_gamma=float(np.min(g)))
            for name, q in (("gas", q_gas), ("sources", q_src)):
                f1, f10 = (q * (g < 1.0)).sum(1), (q * (g < 10.0)).sum(1)
                row[f"{name}_max_pixel_frac_gamma_lt_1"] = float(f1.max())
                row[f"{name}_max_pixel_frac_gamma_lt_10"] = float(f10.max())
                row[f"{name}_sky_frac_gamma_lt_10"] = float(f10.mean())
            rows.append(row)
    return rows


def main():
    t0 = time.time()
    energies = [1e13, 1e14, 1e15, 3e16]
    rows = []
    for m in (1e-28, 1e-26, 1e-24, 1e-22, 1e-20, 1e-18, 1e-16):
        for k in (1e-3, 1e-1, 10.0, 1e3):
            for r in run_point(m, k, energies):
                rows.append(r)
            worst = min(rr["min_gamma"] for rr in rows[-2 * len(energies):])
            frac = max(rr["sources_max_pixel_frac_gamma_lt_10"] for rr in rows[-2 * len(energies):])
            frac_g = max(rr["gas_max_pixel_frac_gamma_lt_10"] for rr in rows[-2 * len(energies):])
            print(f"m_Zd {m:.0e} kappa {k:g}: min gamma {worst:.3g}; worst pixel's emission with gamma < 10: "
                  f"gas {frac_g:.2e}, sources {frac:.2e}", flush=True)
    out = dict(note="T9: adiabaticity of the Galactic transit from the emission points (e-mu texture, u = 0.5, "
                    "eps' = 0.1); gamma = (|nu_i - nu_j|/2E) / (|K_ij|/dln sigma x |dln n_eff/ds|)",
               rows=rows, commit=os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
               seconds=time.time() - t0)
    os.makedirs(os.path.join(REPO, "output", "sky"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "output", "sky", "galactic_adiabaticity.json"), "w"), indent=1)
    print(f"wrote output/sky/galactic_adiabaticity.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
