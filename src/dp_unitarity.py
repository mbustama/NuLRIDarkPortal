"""
dp_unitarity.py -- where the models stop being perturbative (notes Sec. 'Relic mass changes and neutrino decay below
0.05 eV', 2026-10-24; Mauricio's decision: such points are not shown).

A vector of mass M coupled to a current that the fermion masses do not conserve has a longitudinal mode that, at
energies well above M, acts as its Goldstone boson phi (equivalence theorem) with Yukawa couplings set by the current's
divergence. For the neutrinos' current J^mu = sum_ij G'_ij chi_i^dag sigma-bar^mu chi_j (mass basis, Weyl fields with
Majorana masses m_i), d_mu J^mu = i sum_ij G'_ij (m_i chi_i chi_j - m_j chi_i^dag chi_j^dag), so that
    -L = 1/2 Y_ij phi chi_i chi_j + h.c.,   Y_ij = i (G'_ij m_i + G'_ji m_j) / M          (symmetric),
the convention of Allwicher, Arnan, Barducci, Nardecchia, JHEP 10 (2021) 129, Eq. (2.6). Tree-level perturbative
unitarity, |Re a_J| <= 1/2 for the largest eigenvalue of the partial-wave matrix (their Eq. 2.5), from their
amplitudes (Eq. A.10) for one real scalar:
    J = 0, the two-fermion states |ij> (i <= j) with helicities ++ and --:
        ++ -> ++:  T = -Y*_ij Y_kl (s channel),   -- -> ++:  T = Y_ik Y_jl + Y_il Y_jk + Y_ij Y_kl (s, t, u),
        a = T / (16 pi) c_ij c_kl, c = 1/sqrt(2) for identical fermions (i = j), 1 otherwise;
    J = 1/2, the fermion-scalar states: a = -3 (Y Y^dag)_ij / (32 pi).
Checks (tests/test_unitarity.py): one Majorana flavor gives |Y|^2 < 4 pi (their Eq. 6.2), J = 1/2 gives 16 pi / 3
(Eq. 6.4), and an off-diagonal pair (a Dirac fermion) gives 8 pi / 3 (Eq. 4.7, N = 1).

The Z mixing of the potential planes (epsilon', a Z - Z_D mass mixing) couples the mediator to the Standard Model's
axial currents too: a Dirac fermion f with axial coupling g_A f-bar gamma^mu gamma5 f gives a pseudoscalar Yukawa
y_f = 2 g_A m_f / M, bounded by y_f^2 < 8 pi / (2 N_c + 1) (their Eq. 4.7, real scalar, N = N_c colors).
"""
import numpy as np

_PAIRS = [(i, j) for i in range(3) for j in range(i, 3)]


def goldstone_yukawa(Gp, m, M):
    """Y_ij = i (G'_ij m_i + G'_ji m_j) / M for the couplings G' in the mass basis, masses m, mediator mass M."""
    Gp = np.asarray(Gp, dtype=np.complex128)
    m = np.asarray(m, dtype=float)
    return 1j * (Gp * m[:, None] + Gp.T * m[None, :]) / M


def partial_wave_max(Y):
    """The largest |eigenvalue| of the J = 0 and J = 1/2 partial-wave matrices of -L = 1/2 Y_ij phi chi_i chi_j +
    h.c. (one real scalar, Weyl fermions; tree level, massless limit). Perturbative unitarity: <= 1/2."""
    Y = np.asarray(Y, dtype=np.complex128)
    n = len(_PAIRS)
    c = np.array([1.0 / np.sqrt(2.0) if i == j else 1.0 for i, j in _PAIRS])
    A = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, l) in enumerate(_PAIRS):
            w = c[a] * c[b] / (16.0 * np.pi)
            A[a, b] = -np.conj(Y[i, j]) * Y[k, l] * w                      # ++ -> ++ (s)
            A[n + a, n + b] = np.conj(A[a, b])                               # -- -> --
            stu = (Y[i, k] * Y[j, l] + Y[i, l] * Y[j, k] + Y[i, j] * Y[k, l]) * w
            A[a, n + b] = stu                                                # -- -> ++ (s, t, u)
            A[n + b, a] = np.conj(stu)
    j0 = np.max(np.abs(np.linalg.eigvalsh(0.5 * (A + A.conj().T))))
    j12 = 3.0 * np.max(np.abs(np.linalg.eigvalsh(Y @ Y.conj().T))) / (32.0 * np.pi)
    return max(j0, j12)


def neutrino_unitarity(Gp, m, M):
    """2 x the largest partial wave of the neutrinos' Goldstone couplings: >= 1 where the model is not perturbative.
    The identity part of G' (a flavor-universal coupling) is kept: through the masses its axial part is not conserved
    either. (Rotate G - tr(G)/3 to the mass basis and add tr(G)/3 back on the diagonal, so that rounding of a large
    universal part leaves nothing off the diagonal.)"""
    return 2.0 * partial_wave_max(goldstone_yukawa(Gp, m, M))


def fermion_axial_unitarity(g_A, m_f, M, n_colors=1):
    """2 g_A m_f / M against sqrt(8 pi / (2 N + 1)): >= 1 where a Standard Model fermion's Goldstone coupling breaks
    perturbative unitarity."""
    y = 2.0 * abs(g_A) * m_f / M
    return y / np.sqrt(8.0 * np.pi / (2 * n_colors + 1))
