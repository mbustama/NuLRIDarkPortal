"""
si_mean_free_path.py -- the mean free path of an incident nu_3 on the relic neutrinos, for the four models of secret
interactions, by relic mass state and by channel (the paper's appendix figure mean_free_paths_by_relic, 4 x 4).

Rows: the relic nu_1, nu_2, nu_3, and all three. Columns: scalar phi with Majorana nu (A), vector V with Majorana nu (B),
scalar with Dirac nu (C), vector with Dirac nu (D). The quantity is c / Gamma in Mpc at z = 0, for E from 1 TeV to
100 PeV, with the rate averaged over the relics' momenta,
    Gamma(E) = n int d(eps) rho(eps) sigma(s = 2 E eps),
where n = 3 zeta(3) T^3 / (4 pi^2) is the density per helicity and mass state, eps = E_p - p cos(theta), and sigma is
summed over the relic's spin and the final states. Dirac relics: the scalar takes n int [rho_L sigma_L + rho_R sigma_R];
the vector, n int rho_L sigma(Majorana).

Benchmark, the same in every panel: G = g diag(0, 0, 1) in the flavor basis (coupled to nu_tau), g = 1e-3, M = 1 MeV,
m_1 = 1e-6 eV, normal ordering, the NuFIT 6.1 mixing and splittings of analysis/configs/campaign_final, T = 1.676389e-4 eV
(dp_config's T_nu0_eV). Thermal broadening is included in every channel; redshift is not (z = 0).

From the code (src/, only read): the mixing matrix (dp_core.construct_PMNS), the masses (dp_core.get_mass_eigenvalues),
the mass-basis couplings and widths (dp_si.mass_basis_couplings_model, dp_si.mediator_width), the cross sections on a
relic at rest, channel by channel -- dp_si.sigma_scalar_majorana_channels for A; the coupling tensors times their
kinematic factors and profile norms, summed as in tests/test_si.py, for the others (dirac_scalar_chiral_tensors for C,
vector_tensors for B and D) -- and the relics' distribution rho and its chiral shares rho_L, rho_R
(dp_si._sigma_density).
Done here: the integral over eps, by Gauss-Legendre on panels (geometric panels around the resonance s = M^2 and above
the pair threshold s = 4 M^2, breaks at m +- k T for slow relics). The production transport averages over the relics
another way (a lattice in M^2/s with exact resonance terms), so the figure shows the code's physics, not its
production numerics.

Ported 2026-11-22 from benchmark_rates.py (Petropavlova session, run_relics / plot_relics), unchanged in its physics
and numerics. The cache output/analysis/si_mean_free_path.npz is that script's rates_by_relic.npz (12 cases x 701
energies, 97 s on 4 processes); tests/test_si_mean_free_path.py recomputes a sample of it with this module.
  from analysis import si_mean_free_path as mfp; fig = mfp.figure()       (analysis.figure_style.use() first)
"""
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, "output", "analysis", "si_mean_free_path.npz")

HBAR_EV_S = 6.582119569e-16      # CODATA 2018
C_CM_S, MPC_CM = 2.99792458e10, 3.0856775814913673e24
HUBBLE_MPC = 2.99792458e5 / 67.4   # c / H_0, H_0 = 67.4 km/s/Mpc (the code's Planck 2018 value)

G_COUPLING = 1e-3
M_MED = 1e6                      # eV
M_LIGHTEST = 1e-6                # eV, m_1 in normal ordering
T_NU0 = 1.676389e-4              # eV, dp_config "T_nu0_eV"
MIXING_DEG = (33.75878818411903, 8.623062290036087, 43.280093616242404, 212.0)   # th12, th13, th23, delta_CP
DM21_SQ, DM31_SQ = 7.537e-5, 2.511e-3   # eV^2, NuFIT 6.1 as in analysis/configs/campaign_final
INC = 2                          # incident nu_3 (0-based)
E_FIG_MIN, E_FIG_MAX = 1e12, 1e17   # eV
NODES = 48                       # Gauss-Legendre nodes per panel

SCALAR_CHANNELS = ["s_channel", "t_and_u", "s_x_t_u_interference", "t_x_u_interference", "phi_phi_pairs"]
VECTOR_CHANNELS = ["s_channel", "t_channel_nu_nubar", "s_x_t_interference", "t_and_u_nu_nu",
                   "t_x_u_interference", "VV_pairs"]
INTERFERENCE = ("s_x_t_u_interference", "s_x_t_interference", "t_x_u_interference")
MODELS = "ABCD"
MODEL_LABELS = {"A": r"Scalar $\phi$, Majorana $\nu$", "B": r"Vector $V$, Majorana $\nu$",
                "C": r"Scalar $\phi$, Dirac $\nu$", "D": r"Vector $V$, Dirac $\nu$"}
# profiles of the scalar tensors (flat, P2, P3, P4, P5, P6, double scalar) and of the vector's (dp_si), by channel
SCALAR_GROUPS = [[0], [1, 2], [3, 5], [4], [6]]
VECTOR_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7], [8], [9]]


def code():
    """The repository's dp_core and dp_si."""
    for p in (REPO, os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import dp_core
    import dp_si
    return dp_core, dp_si


def n_collision_channels(key, cols):
    """The columns that are collision channels (C then carries the active and sterile neutrinos produced)."""
    return 5 if key == "C" else len(cols)


def cases(rel, M=M_MED, g=G_COUPLING):
    """The four models on the relic mass state rel (0-based), for a mediator of mass M (eV) coupled to nu_tau with g
    (default: the appendix figure's benchmark): dicts with the width, the columns and the terms (pol, f(s)) whose sum
    over terms of rho_pol(eps) f(2 E eps) is integrated (pol 0: rho; 1, -1: rho_L, rho_R).
    Returns (U, cases, scalar_factors)."""
    core, si = code()
    U = np.asarray(core.construct_PMNS(*np.radians(MIXING_DEG)), dtype=np.complex128)
    Gf = np.zeros((3, 3), dtype=np.complex128)
    Gf[2, 2] = g
    M_MED_ = float(M)
    M2 = M_MED_ ** 2
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

    out = []
    # A. scalar, Majorana: lambda = U^T G U
    lam = si.mass_basis_couplings_model(Gf, U, "scalar", True)
    GamA = si.mediator_width(M_MED_, lam, "scalar", True)
    out.append(dict(key="A", Gam=GamA, cols=SCALAR_CHANNELS, coupling=lam,
                    terms=[(0, lambda s: si.sigma_scalar_majorana_channels(s, M_MED_, GamA, lam, a, r, "coherent"))]))
    # B. vector, Majorana: G' = U^dag G U
    Gp = si.mass_basis_couplings_model(Gf, U, "vector", True)
    GamV = si.mediator_width(M_MED_, Gp, "vector", True)
    cV = si.vector_tensors(Gp, True)[0][:, a, r, :].sum(axis=-1)
    fV = lambda s: grouped(cV * vector_factors(s, GamV), VECTOR_GROUPS)
    out.append(dict(key="B", Gam=GamV, cols=VECTOR_CHANNELS, coupling=Gp, terms=[(0, fV)]))
    # C. scalar, Dirac: h = U^dag G U; the left- and right-chiral shares of the relic
    hD = si.mass_basis_couplings_model(Gf, U, "scalar", False)
    GamC = si.mediator_width(M_MED_, hD, "scalar", False)
    Lv, Li, Rv, Ri, _ = (t if np.ndim(t) == 0 else t[:, a, r, :].sum(axis=-1)
                         for t in si.dirac_scalar_chiral_tensors(hD))
    W = si.SCALAR_W

    def fC(cv, ci):
        def f(s):
            k = scalar_factors(s, GamC)
            return np.concatenate([grouped(W * (cv + ci) * k, SCALAR_GROUPS), [np.sum(cv * k), np.sum(ci * k)]])
        return f
    out.append(dict(key="C", Gam=GamC, cols=SCALAR_CHANNELS + ["active_produced", "sterile_produced"], coupling=hD,
                    terms=[(1, fC(Lv, Li)), (-1, fC(Rv, Ri))]))
    # D. vector, Dirac: the Majorana cross sections on the left-chiral share
    out.append(dict(key="D", Gam=GamV, cols=VECTOR_CHANNELS, coupling=Gp, terms=[(1, fV)]))
    for c in out:
        c["m"], c["rel"], c["M"] = m_r, rel, M_MED_
    return U, out, scalar_factors


def panel_edges(E, Gam, m, M=M_MED, T=T_NU0):
    """Panels in eps for a relic of mass m: its support for momenta up to 60 T (rho ~ e^-60 beyond), at least one
    panel per e-fold, breaks at m and at m +- k T (slow relics), around the resonance (geometric in its width) and
    above the pair threshold."""
    pm = 60.0 * T
    Epm = np.sqrt(m * m + pm * pm)
    lo, hi = m * m / (Epm + pm), Epm + pm
    pts = list(np.geomspace(lo, hi, int(np.ceil(np.log(hi / lo))) + 1)) + [m, T]
    pts += [m + sgn * k * T for k in (0.5, 1, 2, 4, 8, 16, 32) for sgn in (-1, 1)]
    eR = M ** 2 / (2.0 * E)
    g = Gam / M
    for d in g * 10.0 ** np.arange(0, 12):
        if d <= 0.3:
            pts += [eR * (1.0 - d), eR * (1.0 + d)]
    pts += [eR, 4.0 * eR] + [4.0 * eR * (1.0 + 10.0 ** -k) for k in range(1, 10)]
    pts = np.unique(np.clip(pts, lo, hi))
    keep = np.concatenate([[True], np.diff(pts) > 1e-15 * pts[1:]])
    return pts[keep]


def panel_integral(f, edges, x, w):
    """sum over panels of Gauss-Legendre sums of f(eps)."""
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        for xi, wi in zip(x, w):
            out = out + wi * h * f(c + h * xi)
    return out


def thermal_rate(E, case, x, w):
    """n int d(eps) sum_terms rho_pol(eps) f(2 E eps), in eV, channel by channel."""
    _, si = code()

    def f(eps):
        acc = 0.0
        for pol, fs in case["terms"]:
            rho = si._sigma_density(eps, case["m"], case.get("T", T_NU0), pol)
            if rho != 0.0:
                acc = acc + rho * fs(2.0 * E * eps)
        return acc
    T = case.get("T", T_NU0)
    out = panel_integral(f, panel_edges(E, case["Gam"], case["m"], case.get("M", M_MED), T), x, w)
    return si.relic_density_per_helicity(T) * np.asarray(out, dtype=float)


def narrow_width_s_channel(E, case):
    """The s channel's rate in the narrow-width limit, n C_s pi M rho(eps_R) / (2 E Gamma), eps_R = M^2 / 2E, with
    C_s the coefficient of s / BW; and the integral of the s channel alone. Returns (integral, limit), in eV."""
    _, si = code()
    x, w = np.polynomial.legendre.leggauss(NODES)
    M = case.get("M", M_MED)
    eR = M ** 2 / (2.0 * E)
    num, nw = 0.0, 0.0
    for pol, f in case["terms"]:
        Cs = f(M ** 2)[0] * case["Gam"] ** 2
        nw += Cs * np.pi * M * si._sigma_density(eR, case["m"], T_NU0, pol) / (2.0 * E * case["Gam"])
        num += thermal_rate(E, dict(case, terms=[(pol, lambda s, f=f: f(s)[0])]), x, w)
    return float(num), float(nw * si.relic_density_per_helicity(T_NU0))


def relic_energies(masses, M=M_MED):
    """The figure's energies: 40 per decade from 1 TeV to 100 PeV, and 250 more across each slow relic's resonance,
    where eps = m +- 15 T (E = M^2 / 2 eps; beyond, rho ~ e^-15)."""
    E = [np.geomspace(E_FIG_MIN, E_FIG_MAX, int(round(np.log10(E_FIG_MAX / E_FIG_MIN) * 40)) + 1)]
    for m in masses:
        if m > 3.0 * T_NU0:
            E.append(np.geomspace(M ** 2 / (2.0 * (m + 15.0 * T_NU0)),
                                  M ** 2 / (2.0 * max(m - 15.0 * T_NU0, 0.2 * m)), 250))
    E = np.unique(np.concatenate(E))
    return E[(E >= E_FIG_MIN) & (E <= E_FIG_MAX)]


_STATE = {}


def _task(task):
    key, rel = task
    st = _STATE
    case = next(c for c in st["cases"][rel] if c["key"] == key)
    rows = np.array([thermal_rate(e, case, st["x"], st["w"]) for e in st["E"]]) / HBAR_EV_S
    nch = n_collision_channels(key, case["cols"])
    conv = max(np.max(np.abs(thermal_rate(st["E"][i], case, st["x2"], st["w2"]) / HBAR_EV_S - rows[i]))
               / abs(rows[i, :nch].sum()) for i in range(0, len(st["E"]), st["conv_stride"]))
    return key, rel, rows, conv


def compute(workers=4, energies=None, path=CACHE, conv_stride=25, M=M_MED, g=G_COUPLING, models=MODELS):
    """The rates (s^-1, channel by channel) for the 4 models x 3 relics; writes the cache when path is given.
    energies: the energies in eV (default: relic_energies). The node-doubling change, relative to the total, is
    checked at every conv_stride-th energy and stored. Returns the cache's content as a dict."""
    by_rel = {rel: cases(rel, M, g)[1] for rel in range(3)}
    masses = np.array([by_rel[r][0]["m"] for r in range(3)])
    E = relic_energies(masses, M) if energies is None else np.asarray(energies, dtype=float)
    x, w = np.polynomial.legendre.leggauss(NODES)
    x2, w2 = np.polynomial.legendre.leggauss(2 * NODES)
    _STATE.update(cases=by_rel, E=E, x=x, w=w, x2=x2, w2=w2, conv_stride=conv_stride)
    tasks = [(k, r) for k in models for r in range(3)]
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            done = pool.map(_task, tasks, chunksize=1)
    else:
        done = [_task(t) for t in tasks]
    out = dict(E_eV=E, masses_eV=masses, node_doubling=np.array(max(d[3] for d in done)), M_eV=np.array(M),
               g=np.array(g))
    for k, r, R, _ in done:
        out[f"{k}_relic{r + 1}"] = R
    for c in by_rel[0]:
        if c["key"] in models:
            out[f"{c['key']}_columns"] = np.array(c["cols"])
    print(f"{len(tasks)} cases x {len(E)} energies in {time.time() - t0:.1f} s on {max(workers, 1)} process(es); "
          f"largest node-doubling change {float(out['node_doubling']):.1e}")
    if path:
        np.savez(path, **out)
        print(f"wrote {path}")
    return out


def mean_free_path_mpc(rate):
    """c / Gamma in Mpc (NaN where the rate is not positive)."""
    rate = np.asarray(rate, dtype=float)
    return np.where(rate > 0, C_CM_S / np.where(rate > 0, rate, 1.0) / MPC_CM, np.nan)


def load(path=CACHE):
    """E (eV), rows[model][relic 0..2] (s^-1, channel by channel), cols[model], masses (eV)."""
    d = np.load(path)
    present = [k for k in MODELS if f"{k}_columns" in d.files]
    rows = {k: {r: d[f"{k}_relic{r + 1}"] for r in range(3)} for k in present}
    cols = {k: [str(c) for c in d[f"{k}_columns"]] for k in present}
    return d["E_eV"], rows, cols, d["masses_eV"]


def figure(path=CACHE):
    """The 4 x 4 figure (rows: relic nu_1, nu_2, nu_3, all three; columns: the models), in the layout approved on
    2026-11-22: square once trimmed (save with bbox_inches='tight', pad_inches=0.05), y from 0.1 to 1e15 Mpc, the
    legend across the top as wide as the panels, model labels above the columns, relic labels left of the rows, the
    Hubble length labelled in the top row's second panel, schematic arrows in the first panel."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter
    from matplotlib.transforms import blended_transform_factory
    from analysis.figure_style import PANEL

    E, rows, cols, masses = load(path)
    y_lo, y_hi = 1e-1, 1e15
    drawn = [(("s_channel",), r"$s$ channel alone", "#005f73"),
             (("t_and_u", "t_channel_nu_nubar", "t_and_u_nu_nu"), r"$t$ and $u$ channels alone", "#ee9b00"),
             (("phi_phi_pairs", "VV_pairs"), "Pair production alone", "#ca6702")]
    box = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.3", lw=0.8)
    EG = E / 1e9
    side = 4 * 1.05 * PANEL[0]
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(side, 0.981 * side))   # square once trimmed
    fig.subplots_adjust(left=0.105, right=0.99, top=0.925, bottom=0.06, hspace=0.05, wspace=0.05)
    for c, k in enumerate(MODELS):
        nch = n_collision_channels(k, cols[k])
        for r in range(4):
            ax = axes[r, c]
            R = rows[k][r] if r < 3 else rows[k][0] + rows[k][1] + rows[k][2]
            total = R[:, :nch].sum(axis=1)
            idx = [cols[k].index(q) for q in INTERFERENCE if q in cols[k]]
            ax.axhline(HUBBLE_MPC, color="0.35", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
            ax.plot(EG, mean_free_path_mpc(total - R[:, idx].sum(axis=1)), color="#ae2012", lw=1.5, zorder=4)
            ax.plot(EG, mean_free_path_mpc(total), color="k", lw=2.0, zorder=5)
            for qs, label, color in drawn:
                j = [cols[k].index(q) for q in qs if q in cols[k]]
                ax.plot(EG, mean_free_path_mpc(R[:, j].sum(axis=1)), color=color, lw=1.3)
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
    ax.text(0.03, HUBBLE_MPC / 10 ** 0.35, "Hubble length,\n" + rf"$c/H_0$ = {HUBBLE_MPC:.0f} Mpc",
            transform=blended_transform_factory(ax.transAxes, ax.transData), ha="left", va="top", fontsize=16,
            color="0.3", linespacing=1.15)
    # schematic arrows in the first panel, from its dip (the narrow-width s channel): a larger M moves it along
    # lambda ~ E (E_res and lambda_min both ~ M^2), a larger g straight down (lambda_min ~ 1 / g^2)
    ax = axes[0, 0]
    total = rows["A"][0][:, :5].sum(axis=1)
    i = int(np.argmax(total))
    to_axes = ax.transData + ax.transAxes.inverted()
    x0, y0 = to_axes.transform((EG[i], C_CM_S / total[i] / MPC_CM))
    d = np.array([1.0 / np.log10(E_FIG_MAX / E_FIG_MIN), 1.0 / np.log10(y_hi / y_lo)])
    arrow = dict(arrowstyle="-|>", lw=1.5, color="0.15", mutation_scale=16, shrinkA=0, shrinkB=0)
    lab = dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none")
    ax.annotate("", xy=(x0 + 0.02 + d[0], y0 - 0.025 + d[1]), xytext=(x0 + 0.02, y0 - 0.025),
                xycoords="axes fraction", textcoords="axes fraction", arrowprops=arrow, zorder=8)
    ax.text(x0 + 0.04 + 0.6 * d[0], y0 - 0.068 + 0.6 * d[1], r"Larger $M$", transform=ax.transAxes, ha="center",
            va="top", fontsize=16, color="0.15", bbox=lab, zorder=7)
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
    for c, k in enumerate(MODELS):
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
    return fig
