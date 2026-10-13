"""
si_validation.py -- validation runs of the secret-interaction transport (src/dp_si.py), too long
for the test suite; run on Tycho (slurm/si_validation.sh). Results go to output/si_validation/.

  python tools/si_validation.py absorption     transport without regeneration against a direct
                                               quadrature of exp(-tau) along every redshift path
  python tools/si_validation.py convergence    energy bins x redshift substeps ladders against the
                                               finest run, spectra and energy-averaged flavor fractions
  python tools/si_validation.py fig5           2107.13568 Fig. 5 (full, s channel, delta function;
                                               coherent and paper double-scalar couplings)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core as C  # noqa: E402
import dp_si as si  # noqa: E402
from dp_core import njit  # noqa: E402
from dp_cosmo import calc_R_source  # noqa: E402

OUT = os.path.join(REPO, "output", "si_validation")
H0 = 67.4 * C.km_per_s / C.Mpc_to_eV_inv
OM, OL = 0.315, 0.685
T0 = 1.676389e-4
# The analysis configurations' mixing (NuFIT 6.0 NO with SK, the pion_2050_case_5 projection) and
# m_lightest = 0.01 eV; the source of pion_2050_case_5 (1/3 : 2/3 : 0, E^-2.5, z <= 4).
U_ANA = C.construct_PMNS(*np.radians([33.70916346871883, 8.559057632313843, 43.280093616242404, 212.0]))
MASSES_ANA = C.get_mass_eigenvalues(0.01, 7.49e-5, 2.513e-3, "normal")
RHO = np.array([1 / 3, 2 / 3, 0.0])


def _kw(U=U_ANA, masses=MASSES_ANA, p_index=2.5):
    return dict(z_max=4.0, H0=H0, Omega_m=OM, Omega_L=OL, U_pmns=U, masses=masses, T_nu0=T0,
                eta_mass=np.ones(3), rho_src=RHO, p_index=p_index, rho_evol=0.0, Phi_0=1.0)


def _grid(n):
    return np.logspace(13, np.log10(3e16), n)


def _fractions(E, F_mass, U):
    dE = C.compute_dE_geometric(E)
    t = ((F_mass @ (np.abs(U) ** 2).T) * dE[:, None]).sum(0)
    return t / t.sum()


def _save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    obj = dict(obj, host=os.uname().nodename, commit=os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
               date=time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", os.path.join(OUT, name + ".json"))


# =============================================================================
# absorption: transport without regeneration vs direct quadrature
# =============================================================================

@njit(cache=True)
def _kappa(z, E, masses, n0, M2, MG, Cabs, glx, glw, H0_, out):
    """d tau / dz for each mass state: sum_r n_r sigma_ar(2 E (1+z) m_r) / (H (1+z))."""
    zp1 = 1.0 + z
    for a in range(3):
        out[a] = 0.0
    for r in range(3):
        if masses[r] <= 0.0:
            continue
        s = 2.0 * E * zp1 * masses[r]
        N2, N4, N5 = si.profile_norms(s / M2, glx, glw)
        bw = (s - M2) ** 2 + MG * MG
        Ipp = si.phiphi_kinematic(s, M2)
        n = n0[r] * zp1 ** 3
        for a in range(3):
            out[a] += n * (Cabs[0, a, r] * s / bw + (Cabs[1, a, r] * N2 + Cabs[3, a, r] * N5) / s
                           + Cabs[2, a, r] * N4 * (s - M2) / bw + Cabs[4, a, r] * Ipp)
    Hz = H0_ * np.sqrt(0.315 * zp1 ** 3 + 0.685)
    for a in range(3):
        out[a] /= Hz * zp1


@njit(cache=True)
def _reference(E, edges, masses, n0, M2, MG, Cabs, p, GX, GW, glx, glw, H0_):
    """int_0^zmax dz (1+z)^-p R(z) / (H (1+z)) exp(-tau_a(E, z)) per mass state, Gauss-Legendre on
    the panels `edges` (graded around the resonance crossings), tau at each node from its panel's
    left edge by a nested rule."""
    k = np.empty(3)
    tau = np.zeros(3)
    res = np.zeros(3)
    tn = np.empty(3)
    for i in range(len(edges) - 1):
        a_, b_ = edges[i], edges[i + 1]
        for g in range(len(GX)):
            zg = a_ + (b_ - a_) * GX[g]
            for a in range(3):
                tn[a] = tau[a]
            for h in range(len(GX)):
                zh = a_ + (zg - a_) * GX[h]
                _kappa(zh, E, masses, n0, M2, MG, Cabs, glx, glw, H0_, k)
                for a in range(3):
                    tn[a] += GW[h] * (zg - a_) * k[a]
            zp1 = 1.0 + zg
            S = zp1 ** (-p) * calc_R_source(zg, 0.0) / (H0_ * np.sqrt(0.315 * zp1 ** 3 + 0.685) * zp1)
            for a in range(3):
                res[a] += GW[g] * (b_ - a_) * S * np.exp(-tn[a])
        for h in range(len(GX)):
            zh = a_ + (b_ - a_) * GX[h]
            _kappa(zh, E, masses, n0, M2, MG, Cabs, glx, glw, H0_, k)
            for a in range(3):
                tau[a] += GW[h] * (b_ - a_) * k[a]
    return res


def run_absorption():
    E = _grid(400)
    kw = _kw()
    n0 = si.relic_density_per_helicity(T0) * np.ones(3)
    P = (np.abs(U_ANA) ** 2).T @ RHO
    free1 = si.free_streaming_flux(E, 4.0, H0, OM, OL, np.eye(3), np.eye(3)[0], 2.5, 0.0, 1.0)[:, 0]
    out = []
    for tex, M, g in [("tau", 5e6, 0.01), ("tau", 5e6, 0.1), ("e", 2e5, 0.03), ("universal", 3e7, 0.3),
                      ("mu", 1e6, 1e-3)]:
        gm = si.mass_basis_couplings(si.flavor_couplings(tex, g), U_ANA)
        Gam = si.scalar_width(M, gm)
        Cabs, _, _ = si.coupling_tensors(gm)
        t0 = time.time()
        F1 = si.propagate_si(E, M=M, g_mass=gm, regeneration=False, **kw)
        F4 = si.propagate_si(E, M=M, g_mass=gm, regeneration=False, z_substeps=4, **kw)
        t1 = time.time()
        e1, e4 = [], []
        for jE in [3, 40, 90, 140, 190, 240, 290, 340, 395]:
            Ej = E[jE]
            pts = list(np.linspace(0.0, 4.0, 401)) + [1.0]
            w0 = max(Gam / M, 1e-12)
            for m in MASSES_ANA:
                c = M * M / (2 * Ej * m) - 1
                if 0 < c < 4:
                    for e in np.arange(np.log10(w0) - 3, 0.01, 0.05):
                        pts += [c - (1 + c) * 10 ** e, c + (1 + c) * 10 ** e]
                    pts.append(c)
            edges = np.unique(np.clip(np.array(pts), 0.0, 4.0))
            ref = _reference(Ej, edges, MASSES_ANA, n0, M * M, M * Gam, Cabs, 2.5, si._GL8_X, si._GL8_W,
                             si._GL24_X, si._GL24_W, H0) * P * Ej ** -2.5
            for a in range(3):
                scale = P[a] * free1[jE]
                e1.append(abs(F1[jE, a] - ref[a]) / scale)
                e4.append(abs(F4[jE, a] - ref[a]) / scale)
        rec = dict(texture=tex, M_eV=M, g=g, max_err_q1=max(e1), median_err_q1=float(np.median(e1)),
                   max_err_q4=max(e4), median_err_q4=float(np.median(e4)), t_transport=t1 - t0,
                   t_reference=time.time() - t1)
        print(rec, flush=True)
        out.append(rec)
    _save("absorption", dict(note="|transport without regeneration - direct quadrature| / free-streaming flux, "
                                   "at 9 energies x 3 mass states, 400 bins", cases=out))


# =============================================================================
# convergence
# =============================================================================

def run_convergence():
    kw = _kw()
    cases = [("tau", 5e6, 0.1), ("tau", 5e6, 0.01), ("universal", 1e5, 0.03), ("e", 1e9, 1.0)]
    ladder = [(200, 1), (200, 2), (400, 1), (400, 2), (400, 4), (800, 1), (800, 2)]
    ref_N, ref_q = 1600, 4
    out = []
    for tex, M, g in cases:
        gm = si.mass_basis_couplings(si.flavor_couplings(tex, g), U_ANA)
        Er = _grid(ref_N)
        t0 = time.time()
        Fr = si.propagate_si(Er, M=M, g_mass=gm, z_substeps=ref_q, **kw)
        t_ref = time.time() - t0
        F0r = si.free_streaming_flux(Er, 4.0, H0, OM, OL, U_ANA, RHO, 2.5, 0.0, 1.0)
        rr = Fr.sum(1) / F0r.sum(1)
        fr_ref = _fractions(Er, Fr, U_ANA)
        rows = []
        for N, q in ladder:
            E = _grid(N)
            t0 = time.time()
            F = si.propagate_si(E, M=M, g_mass=gm, z_substeps=q, **kw)
            dtm = time.time() - t0
            r = F.sum(1) / si.free_streaming_flux(E, 4.0, H0, OM, OL, U_ANA, RHO, 2.5, 0.0, 1.0).sum(1)
            d = np.abs(r - np.interp(np.log(E), np.log(Er), rr))[(E > 1.2e13) & (E < 1e16)]
            rows.append(dict(N=N, q=q, time_s=dtm, max_dev=float(d.max()), median_dev=float(np.median(d)),
                             fraction_dev=float(np.max(np.abs(_fractions(E, F, U_ANA) - fr_ref)))))
            print(tex, M, g, rows[-1], flush=True)
        out.append(dict(texture=tex, M_eV=M, g=g, reference=dict(N=ref_N, q=ref_q, time_s=t_ref,
                                                                    fractions=fr_ref.tolist()), ladder=rows))
    _save("convergence", dict(note="all-flavor flux over free streaming vs the 1600-bin, 4-substep run "
                                    "(pointwise, 1.2e4-1e7 GeV) and energy-averaged flavor fractions", cases=out))


# =============================================================================
# 2107.13568 Fig. 5
# =============================================================================

def run_fig5():
    from scipy.optimize import brentq
    U = C.construct_PMNS(np.radians(33.44), np.radians(8.57), np.radians(49.2), np.radians(197.0))
    m1 = brentq(lambda m: C.get_mass_eigenvalues(m, 7.42e-5, 2.517e-3, "normal").sum() - 0.1, 1e-6, 0.1)
    masses = C.get_mass_eigenvalues(m1, 7.42e-5, 2.517e-3, "normal")
    kw = _kw(U=U, masses=masses, p_index=2.0)
    E = _grid(400)
    F0 = si.free_streaming_flux(E, 4.0, H0, OM, OL, U, RHO, 2.0, 0.0, 1.0).sum(1)
    curves = {}
    for g in [0.1, 0.01]:
        gm = si.mass_basis_couplings(si.flavor_couplings("tau", g), U)
        for ch in ["full", "s", "delta"]:
            for mode in (["coherent", "paper"] if ch == "full" else ["coherent"]):
                F = si.propagate_si(E, M=5e6, g_mass=gm, channels=ch, phiphi=mode, **kw)
                curves[f"g{g}_{ch}_{mode}"] = (F.sum(1) / F0).tolist()
    _save("fig5", dict(note="2107.13568 Fig. 5: nu_tau, M = 5 MeV, E^-2, sum m = 0.1 eV NO; all-flavor flux / "
                            "free streaming", E_eV=E.tolist(), masses=masses.tolist(), curves=curves))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    for ax, g, col in zip(axs, [0.1, 0.01], ["C4", "C1"]):
        ax.plot(E / 1e9, curves[f"g{g}_full_coherent"], "-", color=col, label="Full computation")
        ax.plot(E / 1e9, curves[f"g{g}_full_paper"], "-", color=col, lw=0.8, alpha=0.5,
                label="Full, paper's double-scalar coupling")
        ax.plot(E / 1e9, curves[f"g{g}_s_coherent"], "--", color=col, label="s-channel scattering")
        ax.plot(E / 1e9, curves[f"g{g}_delta_coherent"], ":", color=col, label=r"$\delta$-function approximation")
        ax.axhline(1, color="gray", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlim(1e4, 1e7)
        ax.set_ylim(0, 2.7)
        ax.set_ylabel(r"$E^2\,d\Phi/dE$ / no $\nu$SI")
        ax.text(0.97, 0.9, f"g = {g}", transform=ax.transAxes, ha="right")
    axs[0].legend(loc="center right", fontsize=7)
    axs[1].set_xlabel(r"$E_\nu$ [GeV]")
    fig.suptitle(r"$\nu_\tau$ SI, $M_\phi$ = 5 MeV, $\Sigma m_\nu$ = 0.1 eV (NO), $E^{-2}$; cf. 2107.13568 Fig. 5",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5.png"), dpi=120)
    print("wrote", os.path.join(OUT, "fig5.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["absorption", "convergence", "fig5"])
    args = ap.parse_args()
    for w in args.what:
        t0 = time.time()
        {"absorption": run_absorption, "convergence": run_convergence, "fig5": run_fig5}[w]()
        print(f"{w}: {time.time() - t0:.0f} s", flush=True)
