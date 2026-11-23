#!/usr/bin/env python3
"""
benchmark_rates.py -- our answers to the four checks of si_excerpt.pdf (Sec. 5), from the NuLRIDarkPortal code.

The rate of an incident nu_3 on the relic nu_1, averaged over the relics' momenta (si_excerpt.pdf, Sec. 4),
    Gamma_31(E) = n * int d(eps) rho(eps) sigma_31(s = 2 E eps),
with n = 3 zeta(3) T^3 / (4 pi^2) the density per helicity and mass state and sigma summed over the relic's spin
and over the final states, at z = 0, for
    A. scalar, Majorana;
    B. vector, Majorana;
    C. scalar, Dirac, with the relics' helicity: n int [rho_L sigma_L + rho_R sigma_R], and the rates of active and
       of sterile neutrinos produced (the collision rate weighted by the number of each among the products);
    D. vector, Dirac, with the relics' helicity: n int rho_L sigma(Majorana).
G = g diag(0, 0, 1) in the flavor basis, g = 1e-3, M = 1 MeV, lightest mass 1e-6 eV (normal ordering),
T = 1.676389e-4 eV, E from 10 TeV to 100 PeV at 50 log-spaced points per decade.

Taken from the code (src/ of the repository, only read; nothing is written there):
  - the mixing matrix, dp_core.construct_PMNS, at the NuFIT 6.1 best fit of the analysis configs
    (analysis/configs/campaign_final, 2026-11-07);
  - the mass-basis couplings (dp_si.mass_basis_couplings_model) and the widths (dp_si.mediator_width);
  - the cross sections on a relic at rest, channel by channel: dp_si.sigma_scalar_majorana_channels (A); the
    coupling tensors times their kinematic factors and profile integrals, summed as in tests/test_si.py
    (dirac_scalar_chiral_tensors for C, as in test_bruteforce_scalar_dirac_by_relic_chirality; vector_tensors for
    B and D, as in test_vector_cross_sections);
  - the relics' distribution rho(eps) and its left- and right-chiral shares rho_L, rho_R (dp_si._sigma_density).
Done here: the integral over eps, by Gauss-Legendre on panels in eps, with geometric panels around the resonance
(s = M^2) and above the pair threshold (s = 4 M^2). The production transport averages over the relics in another
way (a lattice in M^2/s, with exact resonance terms), so these are the code's physics, not its production numerics.

Self-checks, printed before the run: the normalization of rho, rho_L + rho_R = rho, the profile-sum recipe against
the closed form (A), the s channel against its narrow-width limit, and the change from doubling the nodes.

Run:   python benchmark_rates.py [--repo PATH] [--out DIR] [--nodes N] [--no-plot]
Output (in --out, default ./our_rates next to this file): A_scalar_majorana.csv, B_vector_majorana.csv,
C_scalar_dirac.csv, D_vector_dirac.csv (E in eV, rates in s^-1), and the figures
benchmark_rates.pdf and benchmark_mean_free_paths.pdf (c / Gamma in Mpc).
"""
import argparse
import hashlib
import os
import sys
import tempfile
import time

import numpy as np

HBAR_EV_S = 6.582119569e-16      # CODATA 2018
HBARC_EV_CM = 1.973269804e-5     # CODATA 2018

G_COUPLING = 1e-3
M_MED = 1e6                      # eV
M_LIGHTEST = 1e-6                # eV, m_1 in normal ordering
T_NU0 = 1.676389e-4              # eV, dp_config "T_nu0_eV"
MIXING_DEG = (33.75878818411903, 8.623062290036087, 43.280093616242404, 212.0)   # th12, th13, th23, delta_CP
DM21_SQ, DM31_SQ = 7.537e-5, 2.511e-3   # eV^2, NuFIT 6.1 as in analysis/configs/campaign_final
INC, REL = 2, 0                  # incident nu_3, relic nu_1 (0-based)
E_MIN, E_MAX = 1e13, 1e17
E_FIG_MIN, E_FIG_MAX = 1e12, 1e17   # eV, the figure by relic (relic nu_3 resonates near 10 TeV)

SCALAR_CHANNELS = ["s_channel", "t_and_u", "s_x_t_u_interference", "t_x_u_interference", "phi_phi_pairs"]
VECTOR_CHANNELS = ["s_channel", "t_channel_nu_nubar", "s_x_t_interference", "t_and_u_nu_nu",
                   "t_x_u_interference", "VV_pairs"]
MODEL_LABELS = {"A": r"Scalar $\phi$, Majorana $\nu$", "B": r"Vector $V$, Majorana $\nu$",
                "C": r"Scalar $\phi$, Dirac $\nu$", "D": r"Vector $V$, Dirac $\nu$"}
# profiles of the scalar tensors (flat, P2, P3, P4, P5, P6, double scalar) and of the vector's (dp_si) by channel
SCALAR_GROUPS = [[0], [1, 2], [3, 5], [4], [6]]
VECTOR_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7], [8], [9]]


def load_code(repo):
    src = os.path.join(repo, "src")
    if not os.path.isfile(os.path.join(src, "dp_si.py")):
        raise SystemExit(f"no src/dp_si.py under {repo}; pass --repo")
    # numba's cache outside the repository, keyed by the sources as dp_core keys its own
    h = hashlib.sha256()
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            h.update(name.encode())
            with open(os.path.join(src, name), "rb") as fh:
                h.update(fh.read())
    os.environ.setdefault("NUMBA_CACHE_DIR",
                          os.path.join(tempfile.gettempdir(), "si_benchmark_numba", h.hexdigest()[:16]))
    sys.dont_write_bytecode = True
    sys.path.insert(0, src)
    import dp_core
    import dp_si
    return dp_core, dp_si


def build_checks(core, si, rel=REL):
    U = np.asarray(core.construct_PMNS(*np.radians(MIXING_DEG)), dtype=np.complex128)
    Gf = np.zeros((3, 3), dtype=np.complex128)
    Gf[2, 2] = G_COUPLING
    M2 = M_MED ** 2
    a, r = INC, rel
    m_r = float(core.get_mass_eigenvalues(M_LIGHTEST, DM21_SQ, DM31_SQ, "normal")[rel])

    def scalar_factors(s, Gam):
        bw = (s - M2) ** 2 + M2 * Gam ** 2
        N2, N4, N5 = si.profile_norms(s / M2, si._GL24_X, si._GL24_W)
        J = (s / bw, (s - M2) / bw, 1.0 / s, si.phiphi_kinematic(s, M2))
        N = (1.0, N2, N2, N4, N5, N4, 1.0)
        return np.array([J[si.SCALAR_JT[q]] * N[q] for q in range(7)])

    def vector_factors(s, Gam):
        bw = (s - M2) ** 2 + M2 * Gam ** 2
        Nq = np.empty(10)
        si.vector_norms(M2 / s, si._GL24_X, si._GL24_W, Nq)
        J = (s / bw, (s - M2) / bw, 1.0 / s, si.vv_kinematic(s, M2))
        return np.array([si.VECTOR_W[q] * J[si.VECTOR_JT[q]] * Nq[q] for q in range(10)])

    def grouped(v, groups):
        return np.array([sum(v[q] for q in g) for g in groups])

    checks = []
    # A. scalar, Majorana: lambda = U^T G U
    lam = si.mass_basis_couplings_model(Gf, U, "scalar", True)
    GamA = si.mediator_width(M_MED, lam, "scalar", True)
    checks.append(dict(key="A", file="A_scalar_majorana.csv", title="A. Scalar, Majorana", Gam=GamA,
                       cols=SCALAR_CHANNELS, coupling=lam,
                       terms=[(0, lambda s: si.sigma_scalar_majorana_channels(s, M_MED, GamA, lam, a, r, "coherent"))]))
    # B. vector, Majorana: G' = U^dag G U
    Gp = si.mass_basis_couplings_model(Gf, U, "vector", True)
    GamV = si.mediator_width(M_MED, Gp, "vector", True)
    cV = si.vector_tensors(Gp, True)[0][:, a, r, :].sum(axis=-1)
    fV = lambda s: grouped(cV * vector_factors(s, GamV), VECTOR_GROUPS)
    checks.append(dict(key="B", file="B_vector_majorana.csv", title="B. Vector, Majorana", Gam=GamV,
                       cols=VECTOR_CHANNELS, coupling=Gp, terms=[(0, fV)]))
    # C. scalar, Dirac: h = U^dag G U; left- and right-chiral shares of the relic
    hD = si.mass_basis_couplings_model(Gf, U, "scalar", False)
    GamC = si.mediator_width(M_MED, hD, "scalar", False)
    Lv, Li, Rv, Ri, _ = (t if np.ndim(t) == 0 else t[:, a, r, :].sum(axis=-1)
                         for t in si.dirac_scalar_chiral_tensors(hD))
    W = si.SCALAR_W

    def fC(cv, ci):
        def f(s):
            k = scalar_factors(s, GamC)
            return np.concatenate([grouped(W * (cv + ci) * k, SCALAR_GROUPS), [np.sum(cv * k), np.sum(ci * k)]])
        return f
    checks.append(dict(key="C", file="C_scalar_dirac.csv", title="C. Scalar, Dirac", Gam=GamC,
                       cols=SCALAR_CHANNELS + ["active_produced", "sterile_produced"], coupling=hD,
                       terms=[(1, fC(Lv, Li)), (-1, fC(Rv, Ri))]))
    # D. vector, Dirac: the Majorana cross sections on the left-chiral share
    checks.append(dict(key="D", file="D_vector_dirac.csv", title="D. Vector, Dirac", Gam=GamV,
                       cols=VECTOR_CHANNELS, coupling=Gp, terms=[(1, fV)]))
    for ch in checks:
        ch["m"], ch["rel"] = m_r, rel
    return U, checks, scalar_factors


def panel_edges(E, Gam, m):
    """Panels in eps = E_p - p cos(theta) for a relic of mass m: its support for momenta up to 60 T (rho ~ e^-60
    beyond), at least one panel per e-fold, breaks at m, at m +- k T (slow relics), around the resonance
    (geometric in its width) and above the pair threshold."""
    T = T_NU0
    pm = 60.0 * T
    Epm = np.sqrt(m * m + pm * pm)
    lo, hi = m * m / (Epm + pm), Epm + pm
    pts = list(np.geomspace(lo, hi, int(np.ceil(np.log(hi / lo))) + 1)) + [m, T]
    pts += [m + sgn * k * T for k in (0.5, 1, 2, 4, 8, 16, 32) for sgn in (-1, 1)]
    eR = M_MED ** 2 / (2.0 * E)
    g = Gam / M_MED
    for d in g * 10.0 ** np.arange(0, 12):
        if d <= 0.3:
            pts += [eR * (1.0 - d), eR * (1.0 + d)]
    pts += [eR, 4.0 * eR] + [4.0 * eR * (1.0 + 10.0 ** -k) for k in range(1, 10)]
    pts = np.unique(np.clip(pts, lo, hi))
    keep = np.concatenate([[True], np.diff(pts) > 1e-15 * pts[1:]])
    return pts[keep]


def thermal_rate(si, E, check, x, w):
    """n int d(eps) sum_terms rho_pol(eps) f(2 E eps), in eV."""
    edges = panel_edges(E, check["Gam"], check["m"])
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for xi, wi in zip(x, w):
            eps = c + h * xi
            s = 2.0 * E * eps
            acc = 0.0
            for pol, f in check["terms"]:
                rho = si._sigma_density(eps, check["m"], T_NU0, pol)
                if rho != 0.0:
                    acc = acc + rho * f(s)
            out = out + wi * h * acc
    return si.relic_density_per_helicity(T_NU0) * np.asarray(out, dtype=float)


def self_checks(si, checks, scalar_factors, x, w):
    print("Self-checks")
    n = si.relic_density_per_helicity(T_NU0)
    print(f"  n per helicity and mass state: {n / HBARC_EV_CM ** 3:.4f} cm^-3")
    edges = panel_edges(1e15, 1e-8 * M_MED, M_LIGHTEST)
    norm = {p: 0.0 for p in (0, 1, -1)}
    for lo, hi in zip(edges[:-1], edges[1:]):
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for xi, wi in zip(x, w):
            for p in norm:
                norm[p] += wi * h * si._sigma_density(c + h * xi, M_LIGHTEST, T_NU0, p)
    print(f"  int rho = {norm[0]:.12f} (1);  int (rho_L + rho_R) = {norm[1] + norm[-1]:.12f};  "
          f"int rho_L = {norm[1]:.9f}, int rho_R = {norm[-1]:.3e}")
    assert abs(norm[0] - 1.0) < 1e-8 and abs(norm[1] + norm[-1] - norm[0]) < 1e-8
    # the profile-sum recipe (used for C) against the closed form (used for A), scalar Majorana
    A = checks[0]
    Creg = si.coupling_tensors(A["coupling"])[1][:, INC, REL, :].sum(axis=-1)
    worst = 0.0
    for s in M_MED ** 2 * np.array([0.01, 0.5, 0.999, 1.0, 1.003, 3.9, 4.5, 30.0, 1e3]):
        recipe = np.sum(si.SCALAR_W * Creg * scalar_factors(s, A["Gam"]))
        closed = si.sigma_scalar_majorana_total(s, M_MED, A["Gam"], A["coupling"], INC, REL)
        worst = max(worst, abs(recipe / closed - 1.0))
    print(f"  scalar Majorana, tensor recipe vs closed form: max relative difference {worst:.1e}")
    assert worst < 1e-12
    # the s channel against its narrow-width limit, n C_s pi M rho_w(eps_R) / (2 E Gamma)
    for ch in checks:
        devs = []
        for E in (2e14, 1e15, 5e15):
            eR = M_MED ** 2 / (2.0 * E)
            num, nw = 0.0, 0.0
            for pol, f in ch["terms"]:
                Cs = f(M_MED ** 2)[0] * (M_MED * ch["Gam"]) ** 2 / M_MED ** 2     # coefficient of s/BW
                nw += Cs * np.pi * M_MED * si._sigma_density(eR, ch["m"], T_NU0, pol) / (2.0 * E * ch["Gam"])
                num += thermal_rate(si, E, dict(ch, terms=[(pol, lambda s, f=f: f(s)[0])]), x, w)
            nw *= si.relic_density_per_helicity(T_NU0)
            devs.append(abs(num / nw - 1.0))
        print(f"  {ch['key']}: s channel vs narrow-width limit at E = 0.2, 1, 5 PeV: "
              + ", ".join(f"{d:.1e}" for d in devs))
        assert max(devs) < 1e-4


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=os.environ.get("NULRI_REPO",
                                                     os.path.expanduser("~/Research/NuLRIDarkPortal")))
    ap.add_argument("--out", default=os.path.join(here, "our_rates"))
    ap.add_argument("--nodes", type=int, default=48, help="Gauss-Legendre nodes per panel")
    ap.add_argument("--per-decade", type=int, default=50)
    ap.add_argument("--only", default="ABCD", help="which checks, e.g. 'AC'")
    ap.add_argument("--energies", type=int, default=0, help="first N energies only (timing)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-relics", action="store_true", help="skip the figure by relic")
    ap.add_argument("--relics-only", action="store_true", help="only the figure by relic")
    ap.add_argument("--relic-sample", type=int, default=0, help="N energies only, for timing (no files)")
    ap.add_argument("--workers", type=int, default=1, help="processes for the figure by relic")
    args = ap.parse_args()

    core, si = load_code(args.repo)
    U, checks, scalar_factors = build_checks(core, si)
    x, w = np.polynomial.legendre.leggauss(args.nodes)
    x2, w2 = np.polynomial.legendre.leggauss(2 * args.nodes)
    print(f"code: {args.repo}/src;  numba cache: {os.environ['NUMBA_CACHE_DIR']}")
    print(f"|U_tau1| = {abs(U[2, 0]):.6f}, |U_tau3| = {abs(U[2, 2]):.6f};  widths Gamma/M: "
          + ", ".join(f"{c['key']} {c['Gam'] / M_MED:.4e}" for c in checks))
    self_checks(si, checks, scalar_factors, x, w)
    os.makedirs(args.out, exist_ok=True)
    if not args.no_relics:
        Er, rows_r, cols_r, masses = run_relics(core, si, x, w, x2, w2, args.workers, args.out, args.relic_sample)
        if not args.no_plot and not args.relic_sample:
            plot_relics(Er, rows_r, cols_r, masses, args.out, args.repo)
    if args.relics_only or args.relic_sample:
        return

    nE = int(round(np.log10(E_MAX / E_MIN) * args.per_decade)) + 1
    E = np.geomspace(E_MIN, E_MAX, nE)
    if args.energies:
        E = E[:args.energies]
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for ch in (c for c in checks if c["key"] in args.only.upper()):
        t0 = time.time()
        rows = np.array([thermal_rate(si, e, ch, x, w) for e in E]) / HBAR_EV_S
        nch = 5 if ch["key"] == "C" else len(ch["cols"])       # the collision channels (C: then active, sterile)
        total = rows[:, :nch].sum(axis=1)
        conv = max(np.max(np.abs(thermal_rate(si, E[i], ch, x2, w2) / HBAR_EV_S - rows[i])) / abs(total[i])
                   for i in range(0, len(E), 10))
        data = np.column_stack([E, total, rows])
        header = "\n".join([
            f"{ch['title']}: rate of an incident nu_3 on the relic nu_1, averaged over the relic momenta, s^-1",
            "Gamma = n int d(eps) rho(eps) sigma(s = 2 E eps), n = 3 zeta(3) T^3 / (4 pi^2) per helicity and mass state"
            + (" (Dirac: rho_L sigma_L + rho_R sigma_R)" if ch["key"] == "C" else
               " (Dirac: rho_L times the Majorana cross section)" if ch["key"] == "D" else ""),
            f"G = g diag(0,0,1), g = {G_COUPLING:g}, M = {M_MED:g} eV, m_1 = {M_LIGHTEST:g} eV, z = 0, "
            f"T = {T_NU0} eV, normal ordering",
            "th12, th13, th23, delta_CP = " + ", ".join(f"{v:.6f}" for v in MIXING_DEG) + " deg (NuFIT 6.1)",
            f"width Gamma/M = {ch['Gam'] / M_MED:.6e};  total = sum of the channels"
            + ("; active/sterile_produced = collision rate x number of active/sterile neutrinos among the products"
               if ch["key"] == "C" else ""),
            f"largest change from doubling the nodes, relative to the total: {conv:.1e}",
            ",".join(["E_eV", "total"] + ch["cols"])])
        path = os.path.join(args.out, ch["file"])
        np.savetxt(path, data, delimiter=",", header=header, fmt="%.10e")
        results[ch["key"]] = (ch, E, total, rows)
        print(f"{ch['title']}: {len(E)} energies in {time.time() - t0:.1f} s; peak {total.max():.4e} s^-1 at "
              f"E = {E[np.argmax(total)]:.3e} eV; node-doubling change {conv:.1e}; wrote {path}")
    if not args.no_plot and results:
        plot(results, args.out, args.repo)
        plot_mfp(results, args.out, args.repo)


def plot(results, out, repo):
    """The four checks on shared axes, in Mauricio's figure style (analysis/figure_style.py; the layout of the paper's
    2 x 2 figures in notebooks/paper_figures.ipynb): logarithmic shared axes, the model in a rounded box inside each
    panel, one framed legend across the top, colours from his palette. The vector's t channel on nubar and its t and u channels on nu are drawn together."""
    sys.path.insert(0, os.path.join(repo, "analysis"))
    import matplotlib
    matplotlib.use("Agg")
    try:
        import figure_style
        figure_style.use()
        panel, ticks = figure_style.PANEL, figure_style.ticks
    except Exception as exc:                          # outside the repository: matplotlib's defaults
        print(f"figure style not found ({exc}); default style")
        panel, ticks = (5.0, 4.75), None
    import matplotlib.pyplot as plt
    # channels, as drawn: (columns summed, legend entry, colour); the same for the scalar and the vector
    drawn = [(("s_channel",), r"$s$ channel", "#005f73"),
             (("t_and_u", "t_channel_nu_nubar", "t_and_u_nu_nu"), r"$t$ and $u$ channels", "#ee9b00"),
             (("s_x_t_u_interference", "s_x_t_interference"), r"Interference of $s$ with $t$, $u$", "#ae2012"),
             (("t_x_u_interference",), r"Interference of $t$ with $u$", "#94d2bd"),
             (("phi_phi_pairs", "VV_pairs"), "Mediator pairs", "#ca6702"),
             (("active_produced",), "Active neutrinos produced", "#0a9396"),
             (("sterile_produced",), "Sterile neutrinos produced", "#9b2226")]
    keys = [k for k in "ABCD" if k in results]
    nrow = (len(keys) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, sharex=True, sharey=True, squeeze=False,
                             figsize=(2.4 * panel[0], 2.15 * panel[1] * nrow / 2 + 1.2))
    fig.subplots_adjust(left=0.13, right=0.975, top=0.87, bottom=0.1, hspace=0.05, wspace=0.05)
    handles = {"Total": plt.Line2D([], [], color="k", lw=2.4, label="Total")}
    for ax, k in zip(axes.flat, keys):
        ch, E, total, rows = results[k]
        ax.plot(E, total, color="k", lw=2.4, zorder=5)
        for cols, label, color in drawn:
            idx = [ch["cols"].index(c) for c in cols if c in ch["cols"]]
            if not idx:
                continue
            y = rows[:, idx].sum(axis=1)
            ls = (0, (1, 1.4)) if "produced" in label else "-"
            lw = 2.0 if "produced" in label else 1.6
            ax.plot(E, np.where(y > 0, y, np.nan), color=color, lw=lw, ls=ls)
            if np.any(y < 0):                         # an interference that lowers the total, drawn as |value|
                ax.plot(E, np.where(y < 0, -y, np.nan), color=color, lw=lw, ls=(0, (5, 2.5)))
            handles.setdefault(label, plt.Line2D([], [], color=color, lw=lw, ls=ls, label=label))
        ax.text(0.045, 0.95, MODEL_LABELS[k], transform=ax.transAxes, ha="left",
                va="top", fontsize=18, zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.0))
    handles["neg"] = plt.Line2D([], [], color="0.35", lw=1.6, ls=(0, (5, 2.5)), label="Negative interference, absolute value")
    ax0 = axes.flat[0]
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(E_MIN, E_MAX)
        ax.set_ylim(1e-32, 1e-12)                       # room above the peaks for the boxes
        ax.xaxis.set_major_locator(FixedLocator([1e14, 1e15, 1e16]))   # no labels where the panels meet
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(10.0 ** np.arange(-30, -14, 3)))
        ax.yaxis.set_minor_locator(FixedLocator(10.0 ** np.arange(-32, -11)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    size = plt.rcParams["axes.labelsize"]
    fig.text(0.5375, 0.015, r"Neutrino energy, $E$ [eV]", ha="center", va="bottom", fontsize=size)
    fig.text(0.012, 0.445, r"Collision rate of $\nu_3$ with relic $\nu_1$, $\Gamma_{31}$ [s$^{-1}$]", rotation=90,
             ha="left", va="center", fontsize=size)
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    leg = fig.legend(handles=list(handles.values()), loc="upper left", bbox_to_anchor=(x0, 0.995, x1 - x0, 0),
                     mode="expand", borderaxespad=0.0, ncol=3, fontsize=15, frameon=True, framealpha=1,
                     edgecolor="black", handlelength=2.6)
    leg.get_frame().set_linewidth(1.0)
    path = os.path.join(out, "benchmark_rates.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {path}")


def plot_mfp(results, out, repo):
    """The same checks as mean free paths, lambda = c / Gamma_31 in Mpc, with the Hubble length c / H_0 (H_0 =
    67.4 km/s/Mpc, the code's Planck 2018 value) for scale. Interferences have no mean free path of their own and are
    left out; each channel's curve is the mean free path if that channel acted alone, and the total without the
    interferences shows their size."""
    sys.path.insert(0, os.path.join(repo, "analysis"))
    import matplotlib
    matplotlib.use("Agg")
    try:
        import figure_style
        figure_style.use()
        panel = figure_style.PANEL
    except Exception as exc:
        print(f"figure style not found ({exc}); default style")
        panel = (5.0, 4.75)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
    c_cm_s, mpc_cm = 2.99792458e10, 3.0856775814913673e24
    hubble_mpc = 2.99792458e5 / 67.4
    drawn = [(("s_channel",), r"$s$ channel alone", "#005f73"),
             (("t_and_u", "t_channel_nu_nubar", "t_and_u_nu_nu"), r"$t$ and $u$ channels alone", "#ee9b00"),
             (("phi_phi_pairs", "VV_pairs"), "Mediator pairs alone", "#ca6702")]
    keys = [k for k in "ABCD" if k in results]
    nrow = (len(keys) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, sharex=True, sharey=True, squeeze=False,
                             figsize=(2.4 * panel[0], 2.15 * panel[1] * nrow / 2 + 1.0))
    fig.subplots_adjust(left=0.13, right=0.975, top=0.893, bottom=0.1, hspace=0.05, wspace=0.05)
    mfp = lambda rate: np.where(rate > 0, c_cm_s / np.where(rate > 0, rate, 1.0) / mpc_cm, np.nan)
    handles = [plt.Line2D([], [], color="k", lw=2.4, label="Total"),
               plt.Line2D([], [], color="#ae2012", lw=1.8, label="Total without interference")]
    interference = ("s_x_t_u_interference", "s_x_t_interference", "t_x_u_interference")
    for ax, k in zip(axes.flat, keys):
        ch, E, total, rows = results[k]
        E = E / 1e9                                     # GeV
        ax.axhline(hubble_mpc, color="0.35", lw=1.4, ls=(0, (5, 2.5)), zorder=1)
        ax.plot(E, mfp(total), color="k", lw=2.4, zorder=5)
        idx = [ch["cols"].index(c) for c in interference if c in ch["cols"]]
        ax.plot(E, mfp(total - rows[:, idx].sum(axis=1)), color="#ae2012", lw=1.8, zorder=4)   # under the total: seen only where they differ
        for cols, label, color in drawn:
            idx = [ch["cols"].index(c) for c in cols if c in ch["cols"]]
            ax.plot(E, mfp(rows[:, idx].sum(axis=1)), color=color, lw=1.6)
            if ax is axes.flat[0]:
                handles.append(plt.Line2D([], [], color=color, lw=1.6, label=label))
        ax.text(0.955, 0.95, MODEL_LABELS[k], transform=ax.transAxes, ha="right", va="top", fontsize=18, zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.0))
    handles.append(plt.Line2D([], [], color="0.35", lw=1.4, ls=(0, (5, 2.5)),
                              label=rf"Hubble length, $c/H_0$ = {hubble_mpc:.0f} Mpc"))
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(E_MIN / 1e9, E_MAX / 1e9)
        ax.set_ylim(1e-4, 1e20)                           # room below the dips for the arrows
        ax.xaxis.set_major_locator(FixedLocator([1e5, 1e6, 1e7]))      # GeV; no labels where the panels meet
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator(10.0 ** np.arange(-2, 20, 4)))
        ax.yaxis.set_minor_locator(FixedLocator(10.0 ** np.arange(-4, 21)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    # schematic arrows, in the first panel only, from its dip (narrow-width s channel, the self-check's formula): a larger M moves it along
    # lambda ~ E (E_res and lambda_min both ~ M^2), a larger g straight down (lambda_min ~ 1 / g^2)
    for ax, k in list(zip(axes.flat, keys))[:1]:
        ch, E, total, rows = results[k]
        i = int(np.argmax(total))
        to_axes = ax.transData + ax.transAxes.inverted()
        x0, y0 = to_axes.transform((E[i] / 1e9, c_cm_s / total[i] / mpc_cm))
        nx, ny = np.log10(E_MAX / E_MIN), 24.0                     # decades across the panel, x and y
        d = np.array([1.0 / nx, 1.0 / ny]) * 0.9                     # 0.9 decade along lambda ~ E
        box = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
        arrow = dict(arrowstyle="-|>", lw=1.8, color="0.15", mutation_scale=20, shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(x0 + 0.02 + d[0], y0 - 0.055 + d[1]), xytext=(x0 + 0.02, y0 - 0.055),
                    xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrow, zorder=8)
        ax.text(x0 + 0.02 + d[0] / 2, y0 - 0.085 + d[1] / 2, r"Larger $M$", transform=ax.transAxes, ha="center",
                va="top", fontsize=16, color="0.15", bbox=box, zorder=7)
        ax.annotate("", xy=(x0, y0 - 0.17), xytext=(x0, y0 - 0.03), xycoords="axes fraction",
                    textcoords="axes fraction", arrowprops=arrow, zorder=8)
        ax.text(x0 - 0.02, y0 - 0.11, r"Larger $g$", transform=ax.transAxes, ha="right", va="center",
                fontsize=16, color="0.15", bbox=box, zorder=7)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    size = plt.rcParams["axes.labelsize"]
    # the axis labels just outside the tick labels (measured on the drawn figure)
    fig.canvas.draw()
    ren, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    y_ticks = min(inv.transform(ax.get_tightbbox(ren).p0)[1] for ax in axes[-1])
    x_ticks = min(inv.transform(ax.get_tightbbox(ren).p0)[0] for ax in axes[:, 0])
    grid_x = 0.5 * (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1)
    grid_y = 0.5 * (axes[-1, 0].get_position().y0 + axes[0, 0].get_position().y1)
    fig.text(grid_x, y_ticks - 0.006, r"Neutrino energy, $E$ [GeV]", ha="center", va="top", fontsize=size)
    fig.text(x_ticks - 0.006, grid_y, r"Mean free path of $\nu_3$ on relic $\nu_1$, $c/\Gamma_{31}$ [Mpc]",
             rotation=90, ha="right", va="center", fontsize=size)
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, 0.995, x1 - x0, 0), mode="expand",
                     borderaxespad=0.0, ncol=3, fontsize=15, frameon=True, framealpha=1, edgecolor="black",
                     handlelength=2.6, title_fontsize=16,
                     title=rf"Secret interactions at $z = 0$: $M$ = 1 MeV, $g = 10^{{-3}}$ (coupled to "
                           rf"$\nu_\tau$), $m_1 = 10^{{-6}}$ eV, normal ordering")
    leg.get_frame().set_linewidth(1.0)
    path = os.path.join(out, "benchmark_mean_free_paths.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {path}")


# ------------------------------------------------------------------------------------------------------------------
# The same rates on each relic mass state, nu_1, nu_2, nu_3, and on all three (the figure by relic)
_RELIC_STATE = {}


def relic_energies(masses):
    """The figure's energies: 40 per decade from 1 TeV to 100 PeV, and 250 more across each slow relic's resonance,
    where eps = m +- 15 T (E = M^2 / 2 eps; beyond, rho ~ e^-15)."""
    E = [np.geomspace(E_FIG_MIN, E_FIG_MAX, int(round(np.log10(E_FIG_MAX / E_FIG_MIN) * 40)) + 1)]
    for m in masses:
        if m > 3.0 * T_NU0:
            E.append(np.geomspace(M_MED ** 2 / (2.0 * (m + 15.0 * T_NU0)),
                                  M_MED ** 2 / (2.0 * max(m - 15.0 * T_NU0, 0.2 * m)), 250))
    E = np.unique(np.concatenate(E))
    return E[(E >= E_FIG_MIN) & (E <= E_FIG_MAX)]


def _relic_task(task):
    key, rel = task
    st = _RELIC_STATE
    ch = next(c for c in st["checks"][rel] if c["key"] == key)
    rows = np.array([thermal_rate(st["si"], e, ch, st["x"], st["w"]) for e in st["E"]]) / HBAR_EV_S
    nch = 5 if key == "C" else len(ch["cols"])
    conv = max(np.max(np.abs(thermal_rate(st["si"], st["E"][i], ch, st["x2"], st["w2"]) / HBAR_EV_S - rows[i]))
               / abs(rows[i, :nch].sum()) for i in range(0, len(st["E"]), 25))
    return key, rel, rows, conv


def relic_self_checks(si, checks_by_rel, x, w):
    """For the slow relics: rho normalized, rho_L + rho_R = rho, and the s channel at its narrow-width limit."""
    for rel, checks in checks_by_rel.items():
        m = checks[0]["m"]
        edges = panel_edges(M_MED ** 2 / (2.0 * m), 1e-8 * M_MED, m)
        norm = {p: 0.0 for p in (0, 1, -1)}
        for lo, hi in zip(edges[:-1], edges[1:]):
            c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
            for xi, wi in zip(x, w):
                for p in norm:
                    norm[p] += wi * h * si._sigma_density(c + h * xi, m, T_NU0, p)
        devs = []
        for ch in checks:
            v = 3.15 * T_NU0 / m        # the relics' typical speed; the formula holds where rho(eps_R) is not in
            for E in M_MED ** 2 / (2.0 * m) * np.array([1 - 0.3 * v, 1.0, 1 + 0.3 * v]):   # its tail
                eR = M_MED ** 2 / (2.0 * E)
                num, nw = 0.0, 0.0
                for pol, f in ch["terms"]:
                    Cs = f(M_MED ** 2)[0] * ch["Gam"] ** 2
                    nw += Cs * np.pi * M_MED * si._sigma_density(eR, m, T_NU0, pol) / (2.0 * E * ch["Gam"])
                    num += thermal_rate(si, E, dict(ch, terms=[(pol, lambda s, f=f: f(s)[0])]), x, w)
                nw *= si.relic_density_per_helicity(T_NU0)
                devs.append(abs(num / nw - 1.0) if nw > 0 else 0.0)
        print(f"  relic nu_{rel + 1} (m = {m:.4g} eV): int rho = {norm[0]:.10f}, int (rho_L + rho_R) - int rho = "
              f"{norm[1] + norm[-1] - norm[0]:.1e}, int rho_L = {norm[1]:.4f}; s channel vs narrow width at M^2/2m "
              f"and 0.3 v either side: max {max(devs):.1e}")
        assert abs(norm[0] - 1.0) < 1e-8 and abs(norm[1] + norm[-1] - norm[0]) < 1e-8 and max(devs) < 1e-4


def run_relics(core, si, x, w, x2, w2, workers, out, sample=0):
    checks_by_rel = {rel: build_checks(core, si, rel)[1] for rel in range(3)}
    masses = [checks_by_rel[r][0]["m"] for r in range(3)]
    print("By relic: masses " + ", ".join(f"{m:.4g}" for m in masses) + " eV")
    relic_self_checks(si, {r: checks_by_rel[r] for r in (1, 2)}, x, w)
    E = relic_energies(masses)
    if sample:
        E = E[np.linspace(0, len(E) - 1, sample).astype(int)]
    _RELIC_STATE.update(si=si, checks=checks_by_rel, E=E, x=x, w=w, x2=x2, w2=w2)
    tasks = [(k, r) for k in "ABCD" for r in range(3)]
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            done = pool.map(_relic_task, tasks, chunksize=1)
    else:
        done = [_relic_task(t) for t in tasks]
    print(f"By relic: {len(tasks)} cases x {len(E)} energies in {time.time() - t0:.1f} s on {max(workers, 1)} "
          f"process(es); largest node-doubling change {max(d[3] for d in done):.1e}")
    rows = {k: {} for k in "ABCD"}
    for k, r, R, _ in done:
        rows[k][r] = R
    cols = {c["key"]: c["cols"] for c in checks_by_rel[0]}
    if not sample:
        path = os.path.join(out, "rates_by_relic.npz")
        np.savez(path, E_eV=E, masses_eV=np.array(masses),
                 **{f"{k}_relic{r + 1}": rows[k][r] for k in "ABCD" for r in range(3)},
                 **{f"{k}_columns": np.array(cols[k]) for k in "ABCD"})
        print(f"wrote {path}")
    return E, rows, cols, masses


def plot_relics(E, rows, cols, masses, out, repo):
    """Mean free paths of nu_3 by relic (rows: nu_1, nu_2, nu_3, all three) and model (columns), energy in GeV, in
    the style of the paper's 4 x 4 figure (notebooks/paper_figures.ipynb, limits_secret_interactions): a square
    figure, the legend across the top as wide as the panels, the Hubble length labelled in the top row's second
    panel."""
    sys.path.insert(0, os.path.join(repo, "analysis"))
    import matplotlib
    matplotlib.use("Agg")
    try:
        import figure_style
        figure_style.use()
        panel = figure_style.PANEL
    except Exception as exc:
        print(f"figure style not found ({exc}); default style")
        panel = (5.0, 4.75)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
    from matplotlib.transforms import blended_transform_factory
    c_cm_s, mpc_cm = 2.99792458e10, 3.0856775814913673e24
    hubble_mpc = 2.99792458e5 / 67.4
    y_lo, y_hi = 1e-1, 1e15
    mfp = lambda rate: np.where(rate > 0, c_cm_s / np.where(rate > 0, rate, 1.0) / mpc_cm, np.nan)
    drawn = [(("s_channel",), r"$s$ channel alone", "#005f73"),
             (("t_and_u", "t_channel_nu_nubar", "t_and_u_nu_nu"), r"$t$ and $u$ channels alone", "#ee9b00"),
             (("phi_phi_pairs", "VV_pairs"), "Mediator pairs alone", "#ca6702")]
    interference = ("s_x_t_u_interference", "s_x_t_interference", "t_x_u_interference")
    box = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.3", lw=0.8)
    EG = E / 1e9
    side = 4 * 1.05 * panel[0]
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(side, 0.981 * side))   # square once trimmed
    fig.subplots_adjust(left=0.105, right=0.99, top=0.925, bottom=0.06, hspace=0.05, wspace=0.05)
    for c, k in enumerate("ABCD"):
        nch = 5 if k == "C" else len(cols[k])
        for r in range(4):
            ax = axes[r, c]
            R = rows[k][r] if r < 3 else rows[k][0] + rows[k][1] + rows[k][2]
            total = R[:, :nch].sum(axis=1)
            idx = [cols[k].index(q) for q in interference if q in cols[k]]
            ax.axhline(hubble_mpc, color="0.35", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
            ax.plot(EG, mfp(total - R[:, idx].sum(axis=1)), color="#ae2012", lw=1.5, zorder=4)
            ax.plot(EG, mfp(total), color="k", lw=2.0, zorder=5)
            for qs, label, color in drawn:
                j = [cols[k].index(q) for q in qs if q in cols[k]]
                ax.plot(EG, mfp(R[:, j].sum(axis=1)), color=color, lw=1.3)
            ax.tick_params(labelsize=18)
    relic_names = [rf"Relic $\nu_{r + 1}$, $m_{r + 1}$ = " + (r"$10^{-6}$ eV" if masses[r] < 1e-5 else
                                                              f"{masses[r] * 1e3:.3g} meV") for r in range(3)]
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(E_FIG_MIN / 1e9, E_FIG_MAX / 1e9)
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6, 1e7]))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1e0, 1e5, 1e10]))      # none where the panels meet
        ax.yaxis.set_minor_locator(FixedLocator(10.0 ** np.arange(-1, 16)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    # the Hubble length, labelled once: below its line, near the left end, in the top row's second panel
    ax = axes[0, 1]
    ax.text(0.03, hubble_mpc / 10 ** 0.35, "Hubble length,\n" + rf"$c/H_0$ = {hubble_mpc:.0f} Mpc",
            transform=blended_transform_factory(ax.transAxes, ax.transData), ha="left", va="top", fontsize=16,
            color="0.3", linespacing=1.15)
    # schematic arrows in the first panel, from its dip: larger M along lambda ~ E, larger g straight down
    ax = axes[0, 0]
    R = rows["A"][0]
    total = R[:, :5].sum(axis=1)
    i = int(np.argmax(total))
    to_axes = ax.transData + ax.transAxes.inverted()
    x0, y0 = to_axes.transform((EG[i], c_cm_s / total[i] / mpc_cm))
    d = np.array([1.0 / np.log10(E_FIG_MAX / E_FIG_MIN), 1.0 / np.log10(y_hi / y_lo)]) * 1.0
    arrow = dict(arrowstyle="-|>", lw=1.5, color="0.15", mutation_scale=16, shrinkA=0, shrinkB=0)
    lab = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
    ax.annotate("", xy=(x0 + 0.02 + d[0], y0 - 0.025 + d[1]), xytext=(x0 + 0.02, y0 - 0.025),
                xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrow, zorder=8)
    ax.text(x0 + 0.04 + 0.6 * d[0], y0 - 0.068 + 0.6 * d[1], r"Larger $M$", transform=ax.transAxes, ha="center", va="top",
            fontsize=16, color="0.15", bbox=lab, zorder=7)
    ax.annotate("", xy=(x0, y0 - 0.115), xytext=(x0, y0 - 0.02), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=arrow, zorder=8)
    ax.text(x0 - 0.02, y0 - 0.07, r"Larger $g$", transform=ax.transAxes, ha="right", va="center", fontsize=16,
            color="0.15", bbox=lab, zorder=7)
    # the axis labels just outside the tick labels (measured on the drawn figure)
    fig.canvas.draw()
    ren, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    y_ticks = min(inv.transform(ax.get_tightbbox(ren).p0)[1] for ax in axes[-1])
    x_ticks = min(inv.transform(ax.get_tightbbox(ren).p0)[0] for ax in axes[:, 0])
    grid_x = 0.5 * (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1)
    grid_y = 0.5 * (axes[-1, 0].get_position().y0 + axes[0, 0].get_position().y1)
    fig.text(grid_x, y_ticks - 0.004, r"Neutrino energy, $E$ [GeV]", ha="center", va="top", fontsize=24)
    # the models above their columns, the relics left of their rows (outside the panels), then the y label
    for c, k in enumerate("ABCD"):
        pos = axes[0, c].get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.006, MODEL_LABELS[k], ha="center", va="bottom", fontsize=18,
                 bbox=box)
    relic_texts = []
    for r, name in enumerate(relic_names + ["All three relics"]):
        pos = axes[r, 0].get_position()
        relic_texts.append(fig.text(x_ticks - 0.006, 0.5 * (pos.y0 + pos.y1), name, rotation=90, ha="right",
                                    va="center", fontsize=18, bbox=box))
    fig.canvas.draw()
    x_relics = min(inv.transform(t.get_bbox_patch().get_window_extent(ren).p0)[0] for t in relic_texts)
    fig.text(x_relics - 0.010, grid_y, r"Mean free path of $\nu_3$, $c/\Gamma$ [Mpc]", rotation=90, ha="right",
             va="center", fontsize=24)
    handles = [plt.Line2D([], [], color="k", lw=2.0, label="Total (this work)"),
               plt.Line2D([], [], color="#ae2012", lw=1.5, label="Total without interference")]
    handles += [plt.Line2D([], [], color=color, lw=1.3, label=label) for _, label, color in drawn]
    # the legend spans the panels exactly: from the first column's left edge to the last column's right edge
    x0, x1 = axes[0, 0].get_position().x0, axes[0, -1].get_position().x1
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, 0.997, x1 - x0, 0), mode="expand",
                     borderaxespad=0.0, ncol=len(handles), fontsize=18, frameon=True, framealpha=1,
                     edgecolor="black", handlelength=2.6,
                     title=rf"Secret interactions: $\nu_3$ on relic neutrinos at $z = 0$, with $M$ = 1 MeV, "
                           rf"$g = 10^{{-3}}$ (coupled to $\nu_\tau$), $m_1 = 10^{{-6}}$ eV, normal ordering",
                     title_fontsize=20)
    leg.get_frame().set_linewidth(1.0)
    path = os.path.join(out, "mean_free_paths_by_relic.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {path}")



if __name__ == "__main__":
    main()
