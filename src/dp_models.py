"""
dp_models.py

The eight benchmark models of the dark-portal paper (docs/dev/SCOPE_WIDENING_PLAN.md, decided 2026-10-14): each
fixes, from its gauge structure, the matter charge that sources the potential, the potential's flavor texture, the
mediator's Lagrangian coupling to the neutrinos (the absorption, dp_absorption), the potential's coupling product,
and the visible decay width.

  model                   matter charge (n, p, e)  nu charges (e, mu, tau)   how the mediator reaches matter
  dark_portal             (1, 0, 0) via Z mixing    u u^dag (sterile mixing)  Z-Z_D mass mixing eps'
  L_mu-L_tau              (1, 0, 0) via Z mixing    (0, 1, -1)                Z-Z' mass mixing eps'
  L_e-L_mu                (0, 0, 1)                 (1, -1, 0)                directly (charged leptons)
  L_e-L_tau               (0, 0, 1)                 (1, 0, -1)                directly
  B-3L_e                  (1, 1, -3)                (-3, 0, 0)                directly (quarks and leptons)
  B-3L_mu                 (1, 1, 0)                 (0, -3, 0)                directly
  B-3L_tau                (1, 1, 0)                 (0, 0, -3)                directly
  B-3/2(L_mu+L_tau)       (1, 1, 0)                 (0, -3/2, -3/2)           directly

The mixing models (the dark portal and L_mu - L_tau) keep the theory notes' Eq. (9) potential, coupling product
g eps' K per neutron (K = g_weak / 2 c_W in their convention), and their neutrino coupling adds the Z mixing's
kappa = eps' g_weak / c_W to every flavor (with its e+e- and mu+mu- decays). The direct models couple with the one
gauge coupling g' (dark_portal.g_D plays it): potential g'^2 Q_nu n_Q / m^2, neutrino coupling g' Q_nu, visible
decays to the charged leptons they charge (vector couplings g' q_l); hadronic channels (B models) open above ~0.28 GeV,
outside the absorption window, and are not included. L_mu - L_tau also charges muons and taus directly (g' q_l).
"""
import numpy as np

import dp_absorption

# (kind, matter charge (q_n, q_p, q_e), neutrino charges or None for the sterile texture, lepton charges for decays)
MODELS = {
    "dark_portal": ("mixing", (1.0, 0.0, 0.0), None, {}),
    "L_mu-L_tau": ("mixing", (1.0, 0.0, 0.0), (0.0, 1.0, -1.0), {"mu": 1.0, "tau": -1.0}),
    "L_e-L_mu": ("direct", (0.0, 0.0, 1.0), (1.0, -1.0, 0.0), {"e": 1.0, "mu": -1.0}),
    "L_e-L_tau": ("direct", (0.0, 0.0, 1.0), (1.0, 0.0, -1.0), {"e": 1.0, "tau": -1.0}),
    "B-3L_e": ("direct", (1.0, 1.0, -3.0), (-3.0, 0.0, 0.0), {"e": -3.0}),
    "B-3L_mu": ("direct", (1.0, 1.0, 0.0), (0.0, -3.0, 0.0), {"mu": -3.0}),
    "B-3L_tau": ("direct", (1.0, 1.0, 0.0), (0.0, 0.0, -3.0), {"tau": -3.0}),
    "B-3/2(L_mu+L_tau)": ("direct", (1.0, 1.0, 0.0), (0.0, -1.5, -1.5), {"mu": -1.5, "tau": -1.5}),
}
LEPTON_MASS_EV = {"e": dp_absorption.M_E, "mu": dp_absorption.M_MU, "tau": 1776.86e6}
# K = (4 c_W m_Z^2 / g)(G_F / sqrt2): the potential per unit g eps' and neutron in the theory notes' Eq. (9)
_GF = 1.1663787e-23
K_Z = (4.0 * 0.88 * (91.18e9) ** 2 / 0.65) * (np.sqrt(2.0) / 2.0 * _GF)


def model_names():
    return tuple(MODELS)


def spec(name: str):
    """(kind, matter charge, neutrino charges or None, lepton charges) of a model (case-insensitive name)."""
    key = {k.lower(): k for k in MODELS}.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"Unknown model {name!r}; known: {sorted(MODELS)}")
    return MODELS[key]


def texture(name: str, u=None):
    """The potential's flavor texture (neutrino convention): u u^dag for the dark portal, diag(Q_nu) otherwise."""
    kind, _, qnu, _ = spec(name)
    if qnu is None:
        u = np.asarray(u, dtype=np.complex128)
        return np.ascontiguousarray(np.outer(u, np.conjugate(u)))
    return np.ascontiguousarray(np.diag(np.asarray(qnu, dtype=np.complex128)))


def potential_product(name: str, g: float, eps_prime: float) -> float:
    """The coupling product the solver multiplies by K_Z n / m^2 (the theory notes' Eq. 9 normalization):
    g eps' for the mixing models, g^2 / K_Z for the direct ones (so that the potential is g^2 n_Q / m^2)."""
    kind = spec(name)[0]
    return float(g) * float(eps_prime) if kind == "mixing" else float(g) ** 2 / K_Z


def neutrino_coupling(name: str, g: float, eps_prime: float, u=None) -> np.ndarray:
    """The mediator's Lagrangian coupling to the neutrinos G (flavor basis, coefficient of nubar gamma^mu P_L nu)."""
    kind, _, qnu, _ = spec(name)
    if qnu is None:
        return dp_absorption.dark_portal_coupling(g, u, eps_prime)
    G = float(g) * np.diag(np.asarray(qnu, dtype=np.complex128))
    if kind == "mixing":
        G = G + dp_absorption.kappa_z_mixing(eps_prime) * np.eye(3)
    return np.ascontiguousarray(G)


def visible_width(name: str, M: float, g: float, eps_prime: float) -> float:
    """The mediator's width into charged leptons (eV): the Z mixing's (mixing models) plus the direct vector
    couplings g q_l to the charged leptons the model charges."""
    kind, _, _, lep = spec(name)
    w = dp_absorption.z_mixing_visible_width(M, eps_prime) if kind == "mixing" else 0.0
    for l, q in lep.items():
        w += dp_absorption.fermion_pair_width(M, LEPTON_MASS_EV[l], float(g) * q, 0.0)
    return w
