"""
thermal_reference.py -- reference implementation of the MeV transport with thermal relics in every collision term
(2026-10-18; notes Sec. "Relic motion in the collision transport"; docs/SIMPLIFICATIONS_AND_ASSUMPTIONS.md 6a). Slow
and simple: for measuring the effect (Phase 1) and validating a fast version. Not used by the production code.

Every product of a kinematic factor K_q and a profile quantity G_q is averaged over the relics' collision-weighted
distribution of sigma = s / 2E' (sigma = m at rest),
    rho(sigma) = sigma / (3 zeta(3) T^3) int_{p_X}^inf p f(p/T) / E_p dp,  p_X = (X^2 - m^2) / 2X,
    X = max(sigma, m^2 / sigma),
as a mixture of relics at rest of mass sigma: at each node, the production code's own step integrals
(dp_si._step_integrals: exact in time, the resonance included), its profile totals and in-bin terms, and the produced
spectra from an extended table of the shapes (half the production spacing, cubic in ln mu). Each redshift step is
split into n_sub sub-steps, each with rho at its own temperature. The production sweep (dp_si._si_sweep_cols) then
takes the absolute products: J = 1, the totals, in-bin terms and drifts as computed here, and the produced spectra as
a table with one block of rows per parent bin -- so the sweep itself is the production one.

With kernel 'rest' (one node at sigma = m, one sub-step, the production tables) it reproduces the production
transport without its thermal resonance terms.

  import thermal_reference; thermal_reference.install()     # in a fresh process: the analysis uses the reference
"""
import hashlib
import os
import sys

import numpy as np
from numba import njit

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_si  # noqa: E402
# the production step integrals compiled serially for the mixture loop (called from compiled code, the parallel
# version crashed); not cached, so that its cache entries never meet the production function's (same source)
_step_integrals = njit(cache=False)(dp_si._step_integrals.py_func)

ZETA3 = 1.2020569031595942
OPTIONS = dict(kernel="thermal", n_sub=2, ng=24)
CACHE_DIR = os.path.join(REPO, "output", "cache", "thermal_reference")


def _gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return np.ascontiguousarray(0.5 * (b - a) * x + 0.5 * (b + a)), np.ascontiguousarray(0.5 * (b - a) * w)


_GL16X, _GL16W = _gl(0.0, 1.0, 16)


@njit(cache=True)
def rho(sig, m, T):
    """rho(sigma) for a relic of mass m at temperature T (module docstring): the inner integral by Gauss-Legendre on
    panels split at 2m, 20m, 200m and 2000m (where they lie between p_X and p_X + 4T), then
    up to p_X + 4T and p_X + 44T."""
    X = max(sig, m * m / sig)
    pX = (X * X - m * m) / (2.0 * X)
    acc = 0.0
    # panels: geometric at 2m, 20m, 200m, 2000m (p / E_p = 1 - m^2 / 2p^2 + ... changes on the scale of p itself),
    # then up to p_X + 4T and p_X + 44T
    edges = np.empty(7)
    ne = 0
    edges[ne] = pX
    ne += 1
    for f in (2.0, 20.0, 200.0, 2000.0):
        v = f * m
        if pX < v < pX + 4.0 * T:
            edges[ne] = v
            ne += 1
    edges[ne] = pX + 4.0 * T
    edges[ne + 1] = pX + 44.0 * T
    ne += 2
    for e in range(ne - 1):
        a = edges[e]
        b = edges[e + 1]
        h = b - a
        if h <= 0.0:
            continue
        for i in range(len(_GL16X)):
            p = a + h * _GL16X[i]
            x = p / T
            acc += h * _GL16W[i] * p / (np.exp(x) + 1.0) / np.sqrt(p * p + m * m) if x < 700.0 else 0.0
    return sig * acc / (3.0 * ZETA3 * T * T * T)


@njit(cache=True)
def sigma_nodes(m, T, extra, glx, glw, sig_out, w_out):
    """Nodes and weights of int d sigma rho(sigma) g(sigma): Gauss-Legendre (glx, glw on [0, 1]) in ln sigma on panels between
    the kernel's ends (momenta up to 40 T), split at m, m +- 3T, m +- 10T, 0.1T, T, 3T, 10T, (0.05, 0.5, 5) m^2/T and the
    extra points inside; each panel as two halves with the nodes clustered toward its ends. Returns the
    number of nodes."""
    pm = 40.0 * T
    epm = np.sqrt(pm * pm + m * m)
    lo = m * m / (epm + pm)
    hi = epm + pm
    pts = np.empty(14 + len(extra))
    n = 0
    pts[n] = lo
    n += 1
    mm_T = m * m / T
    for v in (m, 0.1 * T, T, 3.0 * T, 10.0 * T, m - 10.0 * T, m - 3.0 * T, m + 3.0 * T, m + 10.0 * T,
              0.05 * mm_T, 0.5 * mm_T, 5.0 * mm_T):
        if lo < v < hi:
            pts[n] = v
            n += 1
    for v in extra:
        if lo < v < hi:
            pts[n] = v
            n += 1
    pts[n] = hi
    n += 1
    pts = np.sort(pts[:n])
    k = 0
    for i in range(n - 1):
        a = np.log(pts[i])
        b = np.log(pts[i + 1])
        if b - a <= 0.0:
            continue
        # two half-panels with the nodes clustered toward each end (x = end -+ h u^2): the integrands have log
        # spikes (the resonance entering or leaving the step), square roots (the pair threshold) and kinks there
        h = 0.5 * (b - a)
        for side in range(2):
            for g in range(len(glx)):
                u = glx[g]
                x = a + h * u * u if side == 0 else b - h * u * u
                s = np.exp(x)
                sig_out[k] = s
                w_out[k] = 2.0 * h * u * glw[g] * s * rho(s, m, T)
                k += 1
    return k


@njit(cache=True)
def interp_rows(lnmu, lnmu_top, dl, B, jmax, out):
    """Rows d = 0..jmax of the extended table at ln mu: 4-point Lagrange in the node index (2-point at the ends;
    clamped outside the lattice)."""
    n_nodes = B.shape[0]
    P = B.shape[2]
    t = (lnmu_top - lnmu) / dl
    if t <= 0.0:
        for d in range(jmax + 1):
            for q in range(P):
                out[d, q] = B[0, d, q]
        return
    if t >= n_nodes - 1:
        for d in range(jmax + 1):
            for q in range(P):
                out[d, q] = B[n_nodes - 1, d, q]
        return
    i0 = int(np.floor(t))
    f = t - i0
    if i0 >= 1 and i0 + 2 <= n_nodes - 1:
        c0 = -f * (f - 1.0) * (f - 2.0) / 6.0
        c1 = (f + 1.0) * (f - 1.0) * (f - 2.0) / 2.0
        c2 = -(f + 1.0) * f * (f - 2.0) / 2.0
        c3 = (f + 1.0) * f * (f - 1.0) / 6.0
        for d in range(jmax + 1):
            for q in range(P):
                out[d, q] = (c0 * B[i0 - 1, d, q] + c1 * B[i0, d, q] + c2 * B[i0 + 1, d, q]
                             + c3 * B[i0 + 2, d, q])
    else:
        for d in range(jmax + 1):
            for q in range(P):
                out[d, q] = (1.0 - f) * B[i0, d, q] + f * B[i0 + 1, d, q]


@njit(cache=False)
def mixture_step(E, zlo, zhi, zmid, m, n0r, T, M2, MG, H0, Om, OL, delta_mode, prof, hbin, jt, glx, glw,
                 lnmu_top, dl, B, oN, oL, oD, oR, offs):
    """Adds, for one relic state and one (sub-)step, the rho-weighted mixture of at-rest relics of mass sigma:
    oN, oL, oD (N, P) += J_jt[q](sigma) times the profile total, in-bin complement and drift at the midpoint's mu;
    oR[offs[j] + d, q] += J_jt[q](sigma) times the produced spectrum of profile q at offset d (extended table).
    Per bin, the sigma panels are also split where the resonance and the pair threshold enter or leave the step."""
    N = len(E)
    P = len(jt)
    E1 = np.empty(1)
    m1 = np.zeros(3)
    n01 = np.zeros(3)
    n01[0] = n0r
    J1 = np.zeros((1, 3, 4))
    N1 = np.zeros((1, 3, P))
    L1 = np.zeros((1, 3, P))
    D1 = np.zeros((1, 3, P))
    sig = np.empty(40 * len(glx))
    wts = np.empty(40 * len(glx))
    extra = np.empty(4)
    rows = np.empty((N, P))
    for j in range(N):
        e2 = 2.0 * E[j]
        extra[0] = M2 / (e2 * zhi)
        extra[1] = M2 / (e2 * zlo)
        extra[2] = 4.0 * M2 / (e2 * zhi)
        extra[3] = 4.0 * M2 / (e2 * zlo)
        nk = sigma_nodes(m, T, extra, glx, glw, sig, wts)
        E1[0] = E[j]
        for k in range(nk):
            if wts[k] == 0.0:
                continue
            m1[0] = sig[k]
            _step_integrals(E1, zlo, zhi, zmid, m1, n01, M2, MG, H0, Om, OL, delta_mode, prof, hbin, J1, N1, L1,
                            D1)
            interp_rows(np.log(M2 / (e2 * sig[k] * zmid)), lnmu_top, dl, B, j, rows)
            for q in range(P):
                c = wts[k] * J1[0, 0, jt[q]]
                if c == 0.0:
                    continue
                oN[j, q] += c * N1[0, 0, q]
                oL[j, q] += c * L1[0, 0, q]
                oD[j, q] += c * D1[0, 0, q]
                o = offs[j]
                for d in range(j + 1):
                    oR[o + d, q] += c * rows[d, q]


def extended_table(prof, M, E, z_max, sig_lo, sig_hi, n_u, dlnE):
    """The produced spectra at mu = mu_top exp(-n dl), dl = dlnE / 2, over every mu = M^2 / (2 E' sigma) the kernels
    reach (mu capped at 1e6: above, every product is below 1e-12 of the rest), full rows d = 0..N-1:
    (ln mu_top, dl, B (n_nodes, N, P)). Cached on disk."""
    N = len(E)
    dl = 0.5 * dlnE
    mu_top = min(M * M / (2.0 * E[0] * sig_lo), 1e6)
    mu_bot = M * M / (2.0 * E[-1] * (1.0 + z_max) * sig_hi)
    n_nodes = int(np.ceil(np.log(mu_top / mu_bot) / dl)) + 3
    key = hashlib.sha256(repr((prof, M, float(E[0]), float(E[-1]), N, round(np.log(mu_top), 9), n_nodes, n_u,
                               float(dlnE))).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"table_{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return float(z["lnmu_top"]), float(z["dl"]), z["B"]
    mu_list = mu_top * np.exp(-np.arange(n_nodes) * dl)
    d_max = np.full(n_nodes, N - 1, dtype=np.int64)
    if prof == 0:
        Bt, offs = dp_si.build_profile_table_tri(mu_list, d_max, dlnE, n_u)
    else:
        Bt, offs = dp_si.build_vector_table_tri(mu_list, d_max, dlnE, n_u)
    B = np.ascontiguousarray(Bt.reshape(n_nodes, N, Bt.shape[1]))
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    np.savez(tmp, lnmu_top=np.log(mu_top), dl=dl, B=B)
    os.replace(tmp + ".npz", path)
    return float(np.log(mu_top)), dl, B


def transport_columns_ref(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass, rho_evol, M, g_mass,
                          majorana, channels, phiphi, z_substeps, E_extension, n_u, regeneration, mediator,
                          dirac_sterile, Gamma_vis, thermal, columns, return_diagnostics=False):
    """dp_si._transport_columns with the collision terms of module docstring (OPTIONS: kernel 'thermal' or 'rest',
    n_sub sub-steps, ng nodes per panel). Same arguments and returns."""
    from dp_core import compute_dE_geometric
    kernel, n_sub, ng = OPTIONS["kernel"], int(OPTIONS["n_sub"]), int(OPTIONS["ng"])
    glx, glw = _gl(0.0, 1.0, ng)
    E_bins = np.asarray(E_bins, dtype=np.float64)
    N_E = len(E_bins)
    dlnE = float(np.log(E_bins[1] / E_bins[0]))
    n_ext = int(np.ceil(np.log(E_extension) / dlnE - 1e-9)) if E_extension > 1.0 else 0
    N = N_E + n_ext
    E = E_bins[0] * np.exp(dlnE * np.arange(N))
    E[:N_E] = E_bins
    dE = compute_dE_geometric(E)
    q = int(z_substeps)
    Delta = dlnE / q
    Lmax = float(np.log1p(z_max))
    K = max(1, int(np.ceil(Lmax / Delta - 1e-9)))
    L_edges = np.arange(K + 1) * Delta
    L_edges[-1] = Lmax
    top_full = abs((L_edges[K] - L_edges[K - 1]) - Delta) <= 1e-12 * Delta
    L_top_mid = 0.5 * (L_edges[K - 1] + L_edges[K])
    masses = np.ascontiguousarray(masses, dtype=np.float64)
    g_mass = np.ascontiguousarray(g_mass, dtype=np.complex128)
    if not np.any(g_mass):
        return dp_si._transport_columns_orig(E_bins, z_max, H0, Omega_m, Omega_L, U_pmns, masses, T_nu0, eta_mass,
                                             rho_evol, M, g_mass, majorana, channels, phiphi, z_substeps,
                                             E_extension, n_u, regeneration, mediator, dirac_sterile, Gamma_vis,
                                             thermal, columns, return_diagnostics)
    M2 = M * M
    Gam = dp_si.mediator_width(M, g_mass, mediator, majorana) + float(Gamma_vis)
    MG = M * Gam
    spec = dp_si.model_spec(g_mass, mediator, majorana, phiphi, channels, dirac_sterile)
    if Gamma_vis > 0.0:
        X = 24.0 * np.pi * float(Gamma_vis) / M
        kap = 1.0 if majorana else 0.5
        Ctau = spec["Ctau"].copy()
        for a in range(3):
            for r in range(3):
                add = kap * abs(g_mass[r, a]) ** 2 * X / (4.0 * np.pi) / 3.0
                for qq in (0, 1):
                    Ctau[qq, a, r, :] += add
        spec = dict(spec, Ctau=np.ascontiguousarray(Ctau))
    eta_arr = np.asarray(eta_mass, dtype=np.float64)
    prof = spec["prof"]
    jt = spec["jt"]
    P = len(jt)
    delta_mode = str(channels).strip().lower() == "delta"
    n0 = dp_si.relic_density_per_helicity(T_nu0) * eta_arr
    hbin = float(np.exp(0.5 * dlnE))
    if kernel == "rest":
        (lat_B, lat_O), top = dp_si._profile_tables(prof, M, masses, E[0], N, dlnE, q, K, top_full, L_top_mid, n_u)
    else:
        T_hi = T_nu0 * (1.0 + z_max) * float(np.max(eta_arr)) ** (1.0 / 3.0)
        T_lo = T_nu0 * float(np.min(eta_arr)) ** (1.0 / 3.0)
        mmin, mmax = float(np.min(masses[masses > 0])), float(np.max(masses))
        sig_lo = mmin * mmin / (np.sqrt((40 * T_hi) ** 2 + mmin ** 2) + 40 * T_hi)
        sig_lo = min(sig_lo, mmin * mmin / (np.sqrt((40 * T_lo) ** 2 + mmin ** 2) + 40 * T_lo))
        sig_hi = np.sqrt((40 * T_hi) ** 2 + mmax ** 2) + 40 * T_hi
        lnmu_top, dl, Bext = extended_table(prof, M, E, z_max, sig_lo, sig_hi, n_u, dlnE)
    C = len(columns)
    runs = spec["runs"]
    Fs = [np.zeros((N, 3, C)) for _ in runs]
    diags = [np.zeros((C, 6)) for _ in runs]
    reg = np.zeros((N, 3, C))
    Jone = np.ones((N, 3, 4))
    jj = np.arange(N, dtype=np.int64)
    offs = np.zeros(N + 1, dtype=np.int64)
    offs[1:] = np.cumsum(jj + 1)
    R = np.zeros((3, int(offs[-1]), P))
    O = np.ascontiguousarray(np.tile(offs, (3, 1)))
    node = np.ascontiguousarray(np.tile(jj[:, None], (1, 3)))
    Nab = np.zeros((N, 3, P))
    Lab = np.zeros((N, 3, P))
    Dab = np.zeros((N, 3, P))
    Eps = []
    for P_c, p_c, w_c, phi_c, rho_c in columns:
        Ep = E ** (-p_c)
        if w_c[0] > 0.0 or np.isfinite(w_c[1]):
            Ep = np.where((E >= w_c[0]) & (E < w_c[1]), Ep, 0.0)
        Eps.append(Ep)
    Cregs = [np.ascontiguousarray(Cr if regeneration else np.zeros_like(Cr)) for Cr, _ in runs]
    S0 = np.zeros((N, 3, C))
    J = np.zeros((N, 3, 4))
    Nq = np.zeros((N, 3, P))
    Lq = np.zeros((N, 3, P))
    Dq = np.zeros((N, 3, P))
    for m in range(K - 1, -1, -1):
        L_lo, L_hi = L_edges[m], L_edges[m + 1]
        for k, (P_c, p_c, w_c, phi_c, rho_c) in enumerate(columns):
            dt, T_inj = dp_si._step_time_and_injection(L_lo, L_hi, H0, Omega_m, Omega_L, p_c, rho_evol, phi_c)
            S0[:, :, k] = np.outer(Eps[k] * (T_inj / dt), np.asarray(P_c, dtype=np.float64))
        R[:] = 0.0
        Nab[:] = 0.0
        Lab[:] = 0.0
        Dab[:] = 0.0
        if kernel == "rest":
            dp_si._step_integrals(E, np.exp(L_lo), np.exp(L_hi), np.exp(0.5 * (L_lo + L_hi)), masses, n0, M2, MG,
                                  H0, Omega_m, Omega_L, delta_mode, prof, hbin, J, Nq, Lq, Dq)
            if m == K - 1 and not top_full:
                tB, tO = top
                nd = jj
            else:
                tB, tO = lat_B, lat_O
                nd = jj * q + m
            for r in range(3):
                if masses[r] <= 0.0:
                    continue
                for qq in range(P):
                    Jv = J[:, r, jt[qq]]
                    Nab[:, r, qq] = Jv * Nq[:, r, qq]
                    Lab[:, r, qq] = Jv * Lq[:, r, qq]
                    Dab[:, r, qq] = Jv * Dq[:, r, qq]
                for j in range(N):
                    o = tO[r, nd[j]]
                    nrow = min(tO[r, nd[j] + 1] - o, j + 1)
                    R[r, offs[j]:offs[j] + nrow, :] = J[j, r, jt][None, :] * tB[r, o:o + nrow, :]
        else:
            for s in range(n_sub):
                a = L_lo + (L_hi - L_lo) * s / n_sub
                b = L_lo + (L_hi - L_lo) * (s + 1) / n_sub
                zlo, zhi, zmid = np.exp(a), np.exp(b), np.exp(0.5 * (a + b))
                for r in range(3):
                    if masses[r] <= 0.0:
                        continue
                    Tr = T_nu0 * zmid * eta_arr[r] ** (1.0 / 3.0)
                    oN = np.zeros((N, P))
                    oL = np.zeros((N, P))
                    oD = np.zeros((N, P))
                    oR = np.zeros((int(offs[-1]), P))
                    mixture_step(E, zlo, zhi, zmid, float(masses[r]), float(n0[r]), Tr, M2, MG, H0, Omega_m,
                                 Omega_L, delta_mode, prof, hbin, jt, glx, glw, lnmu_top, dl, Bext, oN, oL, oD, oR,
                                 offs)
                    Nab[:, r, :] += oN
                    Lab[:, r, :] += oL
                    Dab[:, r, :] += oD
                    R[r] += oR
        # the in-bin complement and drift are per event (relative to the total) only through their products: the
        # sweep uses J * L and J * D directly, so the absolute ones go in as they are
        for k in range(len(runs)):
            dp_si._si_sweep_cols(Fs[k], E, dE, S0, Jone, Nab, Lab, Dab, jt, spec["w"], spec["qe"], spec["Ctau"],
                                 Cregs[k], R, O, node, dt, reg, diags[k], prof == 1)
    F_all = sum(wt * Fk for (_, wt), Fk in zip(runs, Fs))
    outs = []
    for k in range(C):
        F = np.ascontiguousarray(F_all[:, :, k])
        out = np.ascontiguousarray(F[:N_E])
        if not return_diagnostics:
            outs.append(out)
            continue
        diag = diags[0][k]
        outs.append((out, dict(absorbed=diag[0], secondaries_in_grid=diag[1], secondaries_expected=diag[2],
                               energy_absorbed=diag[3], energy_secondaries_in_grid=diag[4], injected=diag[5])))
    return outs


def install(kernel="thermal", n_sub=2, ng=24):
    """In this process, dp_si's batched transport becomes the reference (the analysis then uses it)."""
    OPTIONS.update(kernel=kernel, n_sub=n_sub, ng=ng)
    if not hasattr(dp_si, "_transport_columns_orig"):
        dp_si._transport_columns_orig = dp_si._transport_columns
    dp_si._transport_columns = transport_columns_ref
    dp_si._COLUMNS.clear()
