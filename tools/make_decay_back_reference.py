"""
make_decay_back_reference.py -- the frozen reference of tests/test_mediator.py (2026-11-07, decision B; the review's
finding 8): a few small transports' arrival populations computed by the code BEFORE the mediator's decay back (commit
917dbff), which the code with dark_portal.mediator_decay_back = False must reproduce.

  python tools/make_decay_back_reference.py <repository root of the old code> <out.npz>
  python tools/make_decay_back_reference.py --si <repository root of the old code> <out.npz>

Since 2026-11-08 both references are made with the old trees plus the rho fix of that day (dp_thermal: rho's
far-from-resonance angular quadrature, `if fine_m and pol != 0` -> `if fine_m and moller`; dp_si._thermal_weights:
the split at sigma = m for pol 0 too, `if pol != 0 and ua < lm < ub` -> `if ua < lm < ub`), which moves them by
1.2e-8 (dark portal) and 3.9e-9 (secret interactions) relative; without it each tree gives back the earlier
reference bit for bit (checked on the laptop).

Run it with the old code's tree (e.g. `git worktree add --detach /tmp/old 917dbff`); the points and grids are
POINTS below, the configurations the final run's (analysis/configs/campaign_final/now_events). The old code has no
mediator_decay_back setting, so none is set.

--si (2026-11-08, the decay back extended to the secret interactions below 1 keV): the detector fluxes of SI_POINTS,
the four models at m_phi = 1e-2 eV (and the scalar at 1e-3 eV), texture mu, the SI campaign's configurations
(analysis/configs/campaign_si/now_events), a pion source; made with the code before the extension (the decision-B code
of 2026-11-08, whose secret interactions below 1 keV ignore the setting), which the code with
dark_portal.mediator_decay_back = False must reproduce (tests/test_si_mediator.py).
"""
import json
import os
import sys

POINTS = [("dp_tau_ul", 7, dict(m_Zd_eV=1e-2, g_D_trace=1e-5, m_lightest_eV=1e-6)),     # dark-portal phase, decays
          ("dp_tau_ul", 7, dict(m_Zd_eV=1e-2, g_D_trace=1e-4, m_lightest_eV=1e-6)),     # dp_si's transport
          ("le_lmu_ul", 6, dict(m_Zd_eV=1e-4, g_D=1e-8, m_lightest_eV=1e-6)),           # a potential
          ("le_lmu_ul", 7, dict(m_Zd_eV=1e-2, g_D=3e-4, m_lightest_eV=1e-6))]           # Goldstone factors
GRID = dict(n_E_bins=24, n_z_bins=10, n_sun_paths=2, n_earth_chords=3)


def run(root, setting=None):
    """The arrival populations (nu, nb) of POINTS with the code in root; setting: dark_portal.mediator_decay_back or
    None (not set)."""
    sys.path[:0] = [os.path.join(root, "src"), root]
    import dark_portal_solver as dps
    from analysis.config import from_dict
    cfgdir = os.path.join(root, "analysis", "configs", "campaign_final", "now_events")
    out = {}
    for i, (plane, block, vals) in enumerate(POINTS):
        raw = json.load(open(os.path.join(cfgdir, f"{plane}__b{block}.json")))
        raw["solver"]["grid"].update(GRID)
        raw["solver"]["galactic"]["enabled"] = False
        if setting is not None:
            raw["solver"]["dark_portal"]["mediator_decay_back"] = bool(setting)
        model = from_dict(raw, "").model()
        E, p = dps.build_solver_params(model._run_cfg(model._overrides(dict(vals))))
        p["show_progress"] = False
        setup = dps._flux_setup(E, p)
        a = dps._flux_from_setup(E, p, setup, (0.5, 0.5), None, arrival_only=True)
        out[f"p{i}_nu"], out[f"p{i}_nb"] = a["P"]
    return out


SI_POINTS = [("si_scalar_majorana_mu_ul", dict(m_phi_eV=1e-2, g_phi=1e-5, m_lightest_eV=1e-6)),
             ("si_scalar_dirac_mu_ul", dict(m_phi_eV=1e-2, g_phi=1e-5, m_lightest_eV=1e-6)),
             ("si_vector_majorana_mu_ul", dict(m_phi_eV=1e-2, g_phi=1e-5, m_lightest_eV=1e-6)),
             ("si_vector_dirac_mu_ul", dict(m_phi_eV=1e-2, g_phi=1e-5, m_lightest_eV=1e-6)),
             ("si_scalar_majorana_mu_ul", dict(m_phi_eV=1e-3, g_phi=1e-6, m_lightest_eV=1e-6))]
SI_GRID = dict(n_E_bins=24, n_z_bins=10)
SI_SOURCE = (1.0 / 3.0, 2.0 / 3.0, 0.0)


def run_si(root, setting=None):
    """The detector fluxes (nu, nb) of SI_POINTS with the code in root; setting: dark_portal.mediator_decay_back or
    None (not set)."""
    sys.path[:0] = [os.path.join(root, "src"), root]
    import numpy as np
    import dark_portal_solver as dps
    from analysis.config import from_dict
    cfgdir = os.path.join(root, "analysis", "configs", "campaign_si", "now_events")
    out = {}
    for i, (plane, vals) in enumerate(SI_POINTS):
        raw = json.load(open(os.path.join(cfgdir, f"{plane}.json")))
        raw["solver"]["grid"].update(SI_GRID)
        raw["solver"]["galactic"]["enabled"] = False
        if setting is not None:
            raw["solver"].setdefault("dark_portal", {})["mediator_decay_back"] = bool(setting)
        model = from_dict(raw, "").model()
        E, p = dps.build_solver_params(model._run_cfg(model._overrides(dict(vals))))
        p["show_progress"] = False
        setup = dps._flux_setup(E, p)
        out[f"s{i}_nu"], out[f"s{i}_nb"] = dps._flux_from_setup(E, p, setup, (0.5, 0.5),
                                                                np.asarray(SI_SOURCE, dtype=np.float64))
    return out


if __name__ == "__main__":
    import numpy as np
    if sys.argv[1] == "--si":
        np.savez(sys.argv[3], **run_si(sys.argv[2]))
    else:
        np.savez(sys.argv[2], **run(sys.argv[1]))
