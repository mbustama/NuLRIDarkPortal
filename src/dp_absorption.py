"""
dp_absorption.py

Resonant absorption on the relic neutrinos from the mediator's Lagrangian coupling (docs/dev/PHYSICS_AUDIT.md,
items 1a-1c, 1e, 1f, 1i, 1j; decided with Mauricio, 2026-10-14). The mediator V couples to the neutrinos as
    L = V_mu sum_{alpha beta} G_{alpha beta} nubar_alpha gamma^mu P_L nu_beta,
G Hermitian (flavor basis, neutrino convention); for the dark portal G = g_D u u^dag + kappa I, with
kappa = eps' g / c_W the coupling its Z mixing gives every neutrino (the theory notes' eps' convention: their T12 cross
term c eps' g/c_W is exactly 2 c kappa / 2). Then:

  * width: Gamma = M Tr(G G^dag) / 24 pi (left-handed pairs) + the visible channels (charged leptons where open);
  * formation: an incident neutrino in the eigenstate a of the Hamiltonian (flavor amplitudes W_fa) meets relic mass
    state r with amplitude (U^dag G W)_ra, so the Breit-Wigner of a vector formed by nu_L + nubar_R integrates to
    int sigma ds = 48 pi^2 Gamma_in / M = 2 pi |(U^dag G W)_ra|^2 per relic target (the Z-burst normalization);
  * secondaries: the decay's neutrino carries the density matrix G G^dag (flavor basis), so it enters eigenstate a
    with weight (W^dag G G^dag W)_aa; the fraction Gamma_nu / Gamma of the decays gives neutrinos at all.

The kernels of dp_thermal take weights in their own units (Gamma_in = w M / 12 pi, Gamma = sum_G M / 12 pi), so
w = |(U^dag G W)_ra|^2 / 2 and sum_G = Tr(G G^dag) / 2 + 12 pi Gamma_vis / M. Antineutrinos: U -> U*, G -> G*.

Until 2026-10-14 the solver used the theory notes' form (T7, T11, T12): every relic mass state at the incident
flavor's full coupling (x3), the vector-Dirac width M/12 pi (x2), flavor projections without interference, a cross
term with eps' on every flavor pair and no eps'^2 term, no visible width -- kept as absorption_model 'theory_notes'
for comparisons.
"""
import numpy as np

from dp_core import njit

# The code's electroweak constants (as in the T9 coefficient and T12): g and c_W
G_WEAK = 0.65
C_W = 0.88
SIN2_W = 0.2312                    # the electron's Z vector coupling -1/2 + 2 s_W^2 (PDG, MSbar at m_Z)
M_E = 0.51099895e6                 # eV
M_MU = 105.6583755e6               # eV


def kappa_z_mixing(eps_prime: float) -> float:
    """The coupling to every left-handed neutrino that the Z-Z_D mass mixing gives the mediator, in the theory
    notes' eps' convention (eps' = eps_Z / 2 of Davoudiasl, Lee & Marciano): eps' g / c_W."""
    return float(eps_prime) * G_WEAK / C_W


def dark_portal_coupling(g_D: float, u, eps_prime: float) -> np.ndarray:
    """G = g_D u u^dag + kappa I (flavor basis, neutrino convention): the sterile-mixing coupling and the Z
    mixing's, with u the active-sterile mixing amplitudes (complex allowed)."""
    u = np.asarray(u, dtype=np.complex128)
    return np.ascontiguousarray(g_D * np.outer(u, np.conjugate(u)) + kappa_z_mixing(eps_prime) * np.eye(3))


def fermion_pair_width(M: float, m_f: float, g_V: float, g_A: float, n_c: float = 1.0) -> float:
    """Width (eV) of a vector of mass M into a fermion pair with couplings fbar gamma^mu (g_V - g_A gamma5) f:
    n_c (M / 12 pi) [g_V^2 (1 + 2 r) + g_A^2 (1 - 4 r)] sqrt(1 - 4 r), r = m_f^2 / M^2; zero below threshold.
    (For the Z and the electron, (g/2c_W)^2 with g_V = -1/2 + 2 s_W^2, g_A = -1/2, it gives 84 MeV.)"""
    if M <= 2.0 * m_f:
        return 0.0
    r = (m_f / M) ** 2
    return n_c * M / (12.0 * np.pi) * (g_V * g_V * (1.0 + 2.0 * r) + g_A * g_A * (1.0 - 4.0 * r)) * np.sqrt(1.0 - 4.0 * r)


def z_mixing_visible_width(M: float, eps_prime: float) -> float:
    """Visible width of the Z-mixed mediator: e+e- and mu+mu- with the Z's couplings times eps_Z = 2 eps' (in
    units of the neutrino coupling kappa: g_V = kappa (-1/2 + 2 s_W^2), g_A = -kappa/2). Hadronic channels open
    above ~0.28 GeV (2 m_pi), outside the absorption window, and are not included."""
    k = kappa_z_mixing(eps_prime)
    if k == 0.0:
        return 0.0
    gV, gA = k * (-0.5 + 2.0 * SIN2_W), -0.5 * k
    return fermion_pair_width(M, M_E, gV, gA) + fermion_pair_width(M, M_MU, gV, gA)


class AbsorptionSetup:
    """The absorption quantities of one parameter point for one kind (neutrinos, or antineutrinos with U -> U*,
    G -> G*): the coupling G, the kernel's width coordinate sum_G, the matrices of the coherent weights, the
    neutrino fraction of the decays, and the mass-basis diagonal couplings for the Debye mass.

    Arguments: G -- (3, 3) Hermitian coupling (flavor basis, neutrino convention); U -- PMNS (neutrino
    convention); M -- mediator mass (eV); Gamma_vis -- visible width (eV); anti -- antineutrino conventions."""

    def __init__(self, G, U, M, Gamma_vis=0.0, anti=False):
        G = np.asarray(G, dtype=np.complex128)
        U = np.asarray(U, dtype=np.complex128)
        if not np.allclose(G, G.conj().T, rtol=0.0, atol=1e-14 * max(1.0, float(np.max(np.abs(G))))):
            raise ValueError("The mediator's coupling matrix must be Hermitian")
        self.G = np.ascontiguousarray(G.conj() if anti else G)
        self.U = np.ascontiguousarray(U.conj() if anti else U)
        self.M = float(M)
        self.Gamma_vis = float(Gamma_vis)
        trGG = float(np.real(np.trace(self.G @ self.G.conj().T)))
        self.Gamma_nu = self.M * trGG / (24.0 * np.pi)
        self.Gamma = self.Gamma_nu + self.Gamma_vis
        # the kernel's coordinates: Gamma = sum_G M / 12 pi; a weight w means Gamma_in = w M / 12 pi
        self.sum_G = 0.5 * trGG + (12.0 * np.pi * self.Gamma_vis / self.M if self.M > 0.0 else 0.0)
        # coherent weights: absorption |(A W)_ra|^2 / 2 with A = U^dag G; secondaries (W^dag B W)_aa / 2, B = G G^dag
        self.A = np.ascontiguousarray(self.U.conj().T @ self.G)
        self.B = np.ascontiguousarray(self.G @ self.G.conj().T)
        Gm = self.U.conj().T @ self.G @ self.U                     # mass basis
        self.G_mass_diag2 = np.ascontiguousarray(np.abs(np.diag(Gm)) ** 2)

    def vacuum_weights(self):
        """(Qabs (3 relic, 3 mass), Qreg (3,)): the coherent weights for the mass eigenstates (no potential)."""
        W = self.U
        Qabs = 0.5 * np.abs(self.A @ W) ** 2
        Qreg = 0.5 * np.real(np.einsum("fa,fg,ga->a", W.conj(), self.B, W))
        return Qabs, Qreg


_GL32_X, _GL32_W = np.polynomial.legendre.leggauss(32)
_GL32_X = np.ascontiguousarray(_GL32_X)
_GL32_W = np.ascontiguousarray(_GL32_W)


@njit(cache=True)
def debye_mass_sq(G_mass_diag2: np.ndarray, m_nu: np.ndarray, T_nu: float, eta_mass: np.ndarray,
                  majorana: bool = True) -> float:
    """The mediator's Debye mass squared (eV^2) from the relic neutrinos, at any relic temperature T_nu: the static
    long-wavelength polarization sum_states int d^3p/(2 pi)^3 q(p)^2 (-df/dE) of a frozen Fermi-Dirac momentum
    distribution, f(p / T_r), T_r = T_nu eta_r^(1/3), per mass state r with coupling G'_rr (G' = U^dag G U).

    The mediator couples to the LEFT-HANDED current, so a relic's charge depends on its speed v = p/E:
      Majorana (the default, as the absorption's targets): the vector current of a Majorana field vanishes and the
        left-handed current is -(1/2) nubar gamma^mu gamma5 nu, whose charge on a state of helicity h is
        q = -G'_rr h v -- a slow Majorana neutrino carries almost no charge. Both helicities:
        |G'_rr|^2 2 (1 / 2 pi^2) int f (3p^2/E - p^4/E^3) dp.
      Dirac (the populated helicities, nu with h = -1 and nubar with h = +1): q = +-G'_rr (1 + v)/2, the left-chiral
        part of the state; |G'_rr|^2 2 (1 / 2 pi^2) int f (E + 4p + 4p^2/E - p^4/E^3)/4 dp.
    (The integrands are d[p E q^2/G'^2]/dp, from integrating -df/dE by parts.) Relativistic relics: |G'|^2 T^2/6 per
    state in both cases (the theory notes' Eq. T23, in the flavor basis there). Slow relics, as today's: Majorana
    9 zeta(3) |G'|^2 T^3 / (2 pi^2 m), smaller than T^2/6 by 3.3 T/m; Dirac |G'|^2 m T ln2 / (4 pi^2). Off-diagonal
    loops do not contribute at zero momentum (every state has the same occupation f(p/T)). Gauss-Legendre, 32
    points on each of [0, 8T] and [8T, 50T]."""
    total = 0.0
    for r in range(3):
        if G_mass_diag2[r] == 0.0 or eta_mass[r] <= 0.0:
            continue
        T = T_nu * eta_mass[r] ** (1.0 / 3.0)
        m2 = m_nu[r] * m_nu[r]
        acc = 0.0
        for lo, hi in ((0.0, 8.0 * T), (8.0 * T, 50.0 * T)):
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            for k in range(len(_GL32_X)):
                p = mid + half * _GL32_X[k]
                p2 = p * p
                E = np.sqrt(p2 + m2)
                if majorana:
                    w = 3.0 * p2 / E - p2 * p2 / (E * E * E)
                else:
                    w = 0.25 * (E + 4.0 * p + 4.0 * p2 / E - p2 * p2 / (E * E * E))
                acc += _GL32_W[k] * half * w / (np.exp(p / T) + 1.0)
        total += G_mass_diag2[r] * 2.0 * acc / (2.0 * np.pi ** 2)
    return total
