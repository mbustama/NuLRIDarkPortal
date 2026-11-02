"""
atmospheric_background.py -- the atmospheric neutrino flux at the South Pole and its systematic variations, for the
IceCube event rates (docs/dev/DR2_TRACKS_PLAN.md; Mauricio, 2026-10-14: "For atmospheric background, you can use the
same treatment as NuFlavorSingleSource: MCEq with Barr parameters to vary as part of the statistical analysis").

The treatment is NuFlavorSingleSource's (code/atm_flux_syst.py there, whose regions, priors, yield modification and
cosmic-ray tilt are copied below unchanged): MCEq (Fedynitch et al.) with SIBYLL-2.3d and the H3a primary flux
(Gaisser 2012); the derivatives of the total (conventional + prompt) flux with respect to

  barr_<r>_pi+, barr_<r>_pi-   pi+ / pi- production in Barr et al.'s regions r = g, h, i (projectile E > 30 GeV)
  barr_<r>_K+,  barr_<r>_K-    K+ / K- production in regions r = w, y, z
  cr_dgamma                    a shift of the cosmic-ray nucleon spectral index, pivoting at E_PIVOT per nucleon

by symmetric finite differences; priors (Barr et al., PRD 74, 094009 (2006), Fig. 2) stored with them. A Barr
parameter is the fractional change of the meson yield in its region of (x_lab, projectile energy E_i); in regions i
and z it is scaled by log10(E_i / 500 GeV) (capped at 0.5 / 0.122), so that its 1-sigma prior, 0.122, is Barr et
al.'s u(E_i) = 12.2% log10(E_i / 500 GeV). Regions and implementation follow A. Fedynitch's script in IceCube's PISA
(pisa/scripts/create_barr_sys_tables_mceq.py), proton projectile, isospin for neutrons.

Different from NuFlavorSingleSource, for IceCube: the atmosphere is the South Pole's (NRLMSISE-00, MCEq's MSIS00_IC
'SouthPole', the mean of January and July) rather than the US Standard; the production zenith angles are 20, dense
near the horizon (DR2's northern tracks), against its 8. Up-going directions at the detector take the flux produced at
the mirror zenith on the far side (the far side's latitude is not modelled), as there; Earth absorption is left to
the detector's effective areas. daemonflux (Yanez & Fedynitch, PRD 107, 123037 (2023): MCEq calibrated to muon data,
location 'IceCube') is stored too, as a cross-check of the conventional flux.

Writes (default IceCubeRates/data/atmospheric_mceq_southpole.npz), species s = nue, antinue, numu, antinumu, nutau,
antinutau, arrays (N_cos, N_E) on E_GeV (10 GeV - 100 PeV, MCEq's nodes) x cos_zenith (production, increasing):
  nom_<s>       total flux, conventional + prompt [GeV^-1 cm^-2 s^-1 sr^-1]
  pr_<s>        its prompt part
  d_<p>__<s>    the derivative of the total flux with respect to parameter p (params, prior)
  daemonflux_*  daemonflux's conventional nu_mu and nubar_mu and their 1-sigma errors on its own 41 zenith angles

Runs in its own environment (Tycho: conda_envs/mceq -- MCEq 1.4.2, daemonflux 0.9.0), imports nothing from this
repository; one task per (month, parameter), one process per allocated core:

  /lustre/astro/mbustama/conda_envs/mceq/bin/python tools/atmospheric_background.py --out FILE.npz [--quick]

Each task's result is saved as it finishes (<out>.parts/), so a job that stops keeps them and a rerun skips them.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

MONTHS = ('January', 'July')
# production zenith angles: |cos| in 0.02 steps near the horizon, 0.025-0.05 at mid angles, 0.1 near vertical;
# 89.99 deg for the horizon itself (MCEq is very slow at exactly 90 deg)
COS_NODES = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60,
                      0.70, 0.80, 0.90, 1.0])
SPECIES = ['nue', 'antinue', 'numu', 'antinumu', 'nutau', 'antinutau']
E_KEEP = (10.0, 1e8)             # GeV
DELTA = 1e-3                     # finite-difference step of a Barr parameter
DGAMMA = 1e-2                    # finite-difference step of the cosmic-ray spectral index
E_PIVOT = 1e5                    # GeV per nucleon: the cosmic-ray tilt leaves the flux unchanged here

# --- copied unchanged from NuFlavorSingleSource/code/atm_flux_syst.py (2026-10-14) --------------------------------
# Barr et al. regions: (x_lab min, x_lab max, E_i min, E_i max [GeV]); a parameter acts on all its sub-regions
BARR = {
    'a': [(0.0, 0.5, 0.0, 8.0)],
    'b': [(0.5, 1.0, 0.0, 8.0), (0.6, 1.0, 8.0, 15.0)],
    'c': [(0.2, 0.6, 8.0, 15.0)],
    'd': [(0.0, 0.2, 8.0, 15.0), (0.0, 0.1, 15.0, 30.0), (0.1, 0.2, 15.0, 30.0)],
    'e': [(0.2, 0.6, 15.0, 30.0)],
    'f': [(0.6, 1.0, 15.0, 30.0)],
    'g': [(0.0, 0.1, 30.0, 1e11)],
    'h': [(0.1, 1.0, 30.0, 500.0), (0.1, 1.0, 500.0, 1e11)],
    'i': [(0.1, 1.0, 500.0, 1e11)],
    'w': [(0.0, 1.0, 0.0, 8.0), (0.0, 1.0, 8.0, 15.0), (0.0, 0.1, 15.0, 30.0), (0.1, 0.2, 15.0, 30.0),
          (0.0, 0.1, 30.0, 500.0), (0.0, 0.1, 500.0, 1e11)],
    'x': [(0.2, 1.0, 15.0, 30.0)],
    'y': [(0.1, 1.0, 30.0, 500.0), (0.1, 1.0, 500.0, 1e11)],
    'z': [(0.1, 1.0, 500.0, 1e11)],
}
PRIOR = {'a': 0.10, 'b': 0.30, 'c': 0.10, 'd': 0.30, 'e': 0.05, 'f': 0.10, 'g': 0.30, 'h': 0.15, 'i': 0.122,
         'w': 0.40, 'x': 0.10, 'y': 0.30, 'z': 0.122}
MESONS = {'pi+': 211, 'pi-': -211, 'K+': 321, 'K-': -321}


def barr_unc(xmat, egrid, region, value):
    """Yield modification matrix: 1 + value inside the region (x_lab, E_i); for regions i and z the value is scaled
    by log10(E_i / 500 GeV), capped at 0.5 / 0.122 (Barr et al.'s u(E_i)). MCEq calls it as x_func(xmat, egrid, *args).
    """
    pname, sub = region[0], int(region[1:])
    xmin, xmax, emin, emax = BARR[pname][sub]
    modmat = np.ones_like(xmat)
    modmat[np.tril_indices(xmat.shape[0], -1)] = 0.0
    for j in np.where((emin < egrid) & (egrid <= emax))[0]:
        sel = np.where((xmat[:j + 1, j] >= xmin) & (xmat[:j + 1, j] <= xmax))[0]
        if not sel.size:
            continue
        if pname in ('i', 'z'):
            modmat[sel, j] += value * (min(0.5 / 0.122, np.log10(egrid[j] / 500.)) if egrid[j] > 500. else 0.)
        else:
            modmat[sel, j] += value
    return modmat


def tilted(dgamma):
    """H3a with the nucleon spectral index shifted by dgamma, pivoting at E_PIVOT per nucleon."""
    import crflux.models as pm

    class TiltedH3a(pm.HillasGaisser2012):
        def p_and_n_flux(self, E):
            pf, p, n = super().p_and_n_flux(E)
            t = (np.asarray(E, dtype=float) / E_PIVOT) ** (-dgamma)
            return pf, p * t, n * t
    return TiltedH3a
# --- end of the copy -----------------------------------------------------------------------------------------------


def thetas(cos_nodes):
    return np.degrees(np.arccos(np.maximum(cos_nodes, np.cos(np.radians(89.99)))))


def mceq(month):
    import crflux.models as pm
    from MCEq.core import MCEqRun
    return MCEqRun(interaction_model='SIBYLL23D', primary_model=(pm.HillasGaisser2012, 'H3a'), theta_deg=0.0,
                   density_model=('MSIS00_IC', ('SouthPole', month)))


def solve_all(m, ths):
    """Total and prompt flux of every species at each zenith angle: dicts of arrays (n_theta, n_E)."""
    tot = {s: [] for s in SPECIES}
    pr = {s: [] for s in SPECIES}
    for th in ths:
        m.set_theta_deg(th)
        m.solve()
        for s in SPECIES:
            tot[s].append(m.get_solution('total_' + s, 0))
            pr[s].append(m.get_solution('pr_' + s, 0))
    return {s: np.array(v) for s, v in tot.items()}, {s: np.array(v) for s, v in pr.items()}


def barr_flux(m, pname, pdg, value, ths):
    """Total fluxes with every sub-region of Barr parameter pname shifted by value, for meson pdg."""
    m.unset_mod_pprod(dont_fill=True)
    for k in range(len(BARR[pname])):
        m.set_mod_pprod(2212, pdg, barr_unc, (f'{pname}{k}', value))
    m.regenerate_matrices(skip_decay_matrix=True)
    return solve_all(m, ths)[0]


def task(job):
    """One (month, item): item 'nominal' (total and prompt fluxes) or a parameter's derivative."""
    month, name, ths, parts = job
    part = Path(parts) / f'{month}_{name}.npz'
    if part.exists():
        print(f'{month} {name:14s} already done', flush=True)
        z = np.load(part)
        return (month, name), {k: z[k] for k in z.files}
    t0 = time.time()
    m = mceq(month)
    if name == 'nominal':
        tot, pr = solve_all(m, ths)
        res = {**{'nom_' + s: tot[s] for s in SPECIES}, **{'pr_' + s: pr[s] for s in SPECIES}, 'E': m.e_grid}
    elif name == 'cr_dgamma':
        import crflux.models as pm
        m.set_primary_model(tilted(+DGAMMA), 'H3a')
        up = solve_all(m, ths)[0]
        m.set_primary_model(tilted(-DGAMMA), 'H3a')
        dn = solve_all(m, ths)[0]
        m.set_primary_model(pm.HillasGaisser2012, 'H3a')
        res = {s: (up[s] - dn[s]) / (2 * DGAMMA) for s in SPECIES}
    else:
        _, pname, meson = name.split('_')
        up = barr_flux(m, pname, MESONS[meson], +DELTA, ths)
        dn = barr_flux(m, pname, MESONS[meson], -DELTA, ths)
        res = {s: (up[s] - dn[s]) / (2 * DELTA) for s in SPECIES}
    np.savez(part, **res)
    print(f'{month} {name:14s} {time.time() - t0:7.1f} s', flush=True)
    return (month, name), res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--quick', action='store_true', help='two zenith angles, one month, two parameters (a setup test)')
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    parts = out.parent / (out.stem + '.parts')
    parts.mkdir(parents=True, exist_ok=True)
    cos_nodes = COS_NODES[[0, -1]] if a.quick else COS_NODES
    ths = thetas(cos_nodes)
    months = MONTHS[:1] if a.quick else MONTHS
    names = [f'barr_{r}_{m}' for r in 'ghi' for m in ('pi+', 'pi-')] + \
            [f'barr_{r}_{m}' for r in 'wyz' for m in ('K+', 'K-')] + ['cr_dgamma']
    if a.quick:
        names = ['barr_h_pi+', 'cr_dgamma']
    jobs = [(mo, n, ths, str(parts)) for mo in months for n in ['nominal'] + names]
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 1
    print(f'{len(jobs)} tasks on {min(ncpu, len(jobs))} processes; zenith angles {np.round(ths, 2).tolist()}', flush=True)
    with mp.get_context('fork').Pool(min(ncpu, len(jobs))) as pool:
        res = dict(pool.map(task, jobs, chunksize=1))

    E = res[(months[0], 'nominal')]['E']
    keep = (E >= E_KEEP[0]) & (E <= E_KEEP[1])
    avg = lambda key, n: np.mean([res[(mo, n)][key] for mo in months], axis=0)[:, keep]  # noqa: E731
    o = dict(E_GeV=E[keep], cos_zenith=cos_nodes, theta_deg=ths, e_pivot=E_PIVOT, params=np.array(names),
             prior=np.array([PRIOR[n.split('_')[1]] if n.startswith('barr') else np.nan for n in names]))
    for s in SPECIES:
        o['nom_' + s] = avg('nom_' + s, 'nominal')
        o['pr_' + s] = avg('pr_' + s, 'nominal')
        for n in names:
            o[f'd_{n}__{s}'] = avg(s, n)

    # daemonflux's conventional flux, a cross-check (its own zenith angles, at the detector)
    from daemonflux import Flux
    dflux = Flux(location='IceCube', debug=0)
    zen = np.array(sorted((float(z) for z in dflux.zenith_angles), reverse=True))     # cos(zenith) increasing
    Ek = E[keep]
    for q in ('numu', 'antinumu'):
        o['daemonflux_conv_' + q] = np.array([dflux.flux(Ek, f'{z:.4f}', q) / Ek ** 3 for z in zen])
        o['daemonflux_conv_' + q + '_err'] = np.array([dflux.error(Ek, f'{z:.4f}', q) / Ek ** 3 for z in zen])
    o['daemonflux_cos_zenith_detector'] = np.cos(np.radians(zen))
    o.update(units=np.array('GeV^-1 cm^-2 s^-1 sr^-1; arrays (N_cos, N_E); cos_zenith = |cos| of the production zenith'),
             models=np.array("MCEq SIBYLL-2.3d x H3a, MSIS00_IC SouthPole (January + July)/2; derivatives as "
                             "NuFlavorSingleSource/code/atm_flux_syst.py; daemonflux 'IceCube' as a cross-check"),
             versions=np.array(f"MCEq {metadata.version('MCEq')}, daemonflux {metadata.version('daemonflux')}"),
             tool=np.array('tools/atmospheric_background.py'), date=np.array(time.strftime('%Y-%m-%d %H:%M')))
    np.savez_compressed(out, **o)

    i = [int(np.argmin(abs(Ek - e))) for e in (1e3, 1e4, 1e5)]
    mu = o['nom_numu'] + o['nom_antinumu']
    print('\nrelative change of the nu_mu + nubar_mu flux per 1 sigma (cr_dgamma: per 0.05), vertical | horizon, at '
          '1 / 10 / 100 TeV:')
    for n in names:
        sig = 0.05 if n == 'cr_dgamma' else PRIOR[n.split('_')[1]]
        d = (o[f'd_{n}__numu'] + o[f'd_{n}__antinumu']) * sig / mu
        print(f'  {n:14s} ' + ' '.join(f'{d[-1, j]:+.3f}' for j in i) + '  |  ' + ' '.join(f'{d[0, j]:+.3f}' for j in i))
    pr = (o['pr_numu'] + o['pr_antinumu']) / mu
    print('prompt fraction, vertical | horizon: ' + ' '.join(f'{pr[-1, j]:.3f}' for j in i) + '  |  ' +
          ' '.join(f'{pr[0, j]:.3f}' for j in i))
    print(f'\nwrote {out} ({os.path.getsize(out) / 1e6:.2f} MB) in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
