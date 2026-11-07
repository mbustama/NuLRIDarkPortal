"""
dp_trajectory.py -- the beam along its trip, position by position (2026-11-07; Mauricio's request for the paper: the
flux of each of the six species, neutrinos and antineutrinos of each flavor, along the trip from a shell of sources
to the detector). Each stage replays the solver's own step of that stage along one path, recording after every step,
so that each stage ends on the solver's own result for that path (tests/test_trajectory.py):

  universe -- the cosmological phase: dark_portal_solver.propagate_spectrum's track (dp_si's for the MeV masses and
              for secret interactions), the sources in one redshift shell (source_z);
  galaxy   -- one line of sight of the Galaxy crossing, from its far end (30 kpc) to the Sun: dp_galactic_path._cross's
              segments where its processes act (decays, soft scatterings, energy loss), else the populations unchanged;
              decohered in the eigenstates of each segment's local potential;
  sun      -- one path past the Sun: dp_transit.source_transit from far away to points along the path, the last the
              Earth's distance (then the Earth's exterior, as dp_spatial.transit_to_earth);
  earth    -- one chord through the Earth to the detector: dp_spatial._earth_chord_worker's steps.

Each stage returns a dict with the positions and F_nu, F_nb (N_pos, N_E, 3), the flux of each flavor of each kind.
Secret interactions have no potential and no process past the cosmological phase (their Galactic column of relics is
~1e-5 of the cosmological one): their local stages are the arrival, unchanged. dp_si carries neutrinos and
antineutrinos together (CP-symmetric to O(Gamma/M)), split in the sources' proportions: on its path the two kinds'
fluxes are equal by construction.
"""
import numpy as np

import dark_portal_solver as dps
import dp_galactic_path
import dp_spatial
import dp_transit

KPC_M = 3.0856775814913673e19
AU_M = 1.495978707e11


def _flavor(E, F_pop, tab, coef2, n):
    """The flux of each flavor (N_E, 3) of populations F_pop (N_E, 3) of the eigenstates at density n."""
    rho = np.empty((len(E), 3, 3), dtype=np.complex128)
    dp_transit.galactic_map(np.ascontiguousarray(E), np.ascontiguousarray(F_pop, dtype=np.float64), float(coef2),
                            float(n), tab["Q"], tab["Mp"], tab["s"], tab["floor"], rho)
    return np.real(np.einsum('iaa->ia', rho))


def universe(E, params, source_z=None):
    """The cosmological phase with sources in the shell source_z = (z_lo, z_hi): dict(z (N_z,), F_nu, F_nb
    (N_z, N_E, 3)) after every redshift step, the last at z = 0; si -- the secret-interaction path (no handover);
    P, handover, setup -- the arrival's populations and handover (the dark portal's path), for the next stages."""
    E = np.asarray(E, dtype=np.float64)
    setup = dps._flux_setup(E, params)
    tr = []
    out = dict(setup=setup)
    if setup.get("si", False):
        F = dps._flux_from_setup(E, params, setup, dps.injection_weights(params), None, source_z=source_z, track=tr)
        out.update(si=True, arrival=(np.asarray(F[0]), np.asarray(F[1])))
    else:
        a = dps._flux_from_setup(E, params, setup, dps.injection_weights(params), None, arrival_only=True,
                                 source_z=source_z, track=tr)
        out.update(si=False, P=[np.asarray(p, dtype=np.float64) for p in a["P"]], handover=a["handover"])
    out.update(z=np.array([t["z"] for t in tr]), F_nu=np.stack([t["F_nu"] for t in tr]),
               F_nb=np.stack([t["F_nb"] for t in tr]))
    return out


def _kind_frames(E, uni):
    """Per kind: the transit's quantities (dp_spatial._earth_setup as dark_portal_solver.flux_stops)."""
    frames = []
    for args, kw in uni["handover"]:
        (M_rot, u_arr, m_Zd, g_D, eps_prime, T_nu0, use_debye, eta_bg, depth_km, use_SM, n_e_earth, anti,
         n_sun_paths, n_chords) = args
        st = dp_spatial._earth_setup(M_rot, u_arr, m_Zd, g_D, eps_prime, T_nu0, use_debye, eta_bg, depth_km, use_SM,
                                     n_e_earth, anti, n_sun_paths, n_chords, kw['U_LRI_struct'], kw['V_SM_struct'],
                                     kw['use_gpu'], kw['earth_step_scheme'], kw['earth_zenith_average'],
                                     kw['earth_sky'], kw['earth_band_costh'], kw['earth_nodes_per_band'],
                                     kw['matter_charge'], kw['debye_sq_z0'], kw['potential_product'])
        Coef, m_eff, S = st['Coef'], st['m_eff'], st['U_LRI_struct']
        S_signed = np.ascontiguousarray((np.sign(Coef) if Coef != 0.0 else 1.0) * np.asarray(S, dtype=np.complex128))
        tab = dp_transit.frame_table(M_rot, S_signed, kw['transit_dtheta'])
        frames.append(dict(args=args, kw=kw, st=st, Coef=float(Coef), m_eff=float(m_eff), S=S, tab=tab,
                           coef2=2.0 * abs(float(Coef)), M_rot=M_rot))
    return frames


def _transit_info(E, fr, F_pop, sun_path_cos):
    """dp_spatial.transit_to_earth for one Sun path: (rho at the Earth's split radius, n_amb, info)."""
    kw, st, args = fr["kw"], fr["st"], fr["args"]
    return dp_spatial.transit_to_earth(
        E, np.ascontiguousarray(F_pop), fr["M_rot"], fr["S"], fr["Coef"], fr["m_eff"], kw['n_n0_eV3'],
        kw['R_H_eV_inv'], st['earth_yk'], n_sun_paths=args[12], transit_dtheta=kw['transit_dtheta'],
        sun_path_cos=sun_path_cos, arrival=kw['arrival'], matter_charge=kw['matter_charge'],
        m_bare=(args[2] if kw['debye_sq_z0'] is not None else None))


def galaxy(E, params, uni, line=None, work=16):
    """One line of sight across the Galaxy, from its far end to the Sun: dict(d_kpc (N_pos,) the distance to the Sun,
    F_nu, F_nb, Q (2, N_E, 3) the populations at the Sun, line, processes). line: the line of sight (the path sky's,
    dp_galactic_path.groups_of); None, the one whose populations at the Sun are closest to the sky's mean."""
    E = np.asarray(E, dtype=np.float64)
    N = len(E)
    if uni["si"]:
        return None
    setup = uni["setup"]
    P = uni["P"]
    import dp_galactic
    kinds = dps._galactic_kinds(params, setup, work)
    s, ds = dp_galactic.emission_geometry(work)[:2]
    edges = dp_galactic_path.edges_from_geometry(s, ds)
    groups = dp_galactic_path.groups_of(work, min(int(dps.CROSSING_PATH_NSIDE), work))
    gstart, gchild = groups
    ng = len(gstart) - 1
    gr = None
    if (dps.EXTRAGALACTIC_CROSSING and float(params['m_Zd_eV']) < dps.SI_COSMOLOGY_MIN_M_EV
            and params.get('absorption_model', 'lagrangian') == 'lagrangian'
            and (params.get('relic_scattering', True) or params.get('neutrino_decay', True))
            and params.get('earth_sky', 'up') == 'full'):
        gr = dps._galactic_rates(E, params, setup, kinds, work, crossing=True)
    Pc = np.zeros((1, N, 2, 3))
    Pc[0, :, 0, :], Pc[0, :, 1, :] = P[0], P[1]
    tables = None
    if gr is not None:
        tables = dps._galactic_tables(E, kinds, gr, gr['vac'])
        edges = gr['edges_kpc']
    if gr is not None and gr['vac']:
        # no potential: one line of sight, 30 kpc of vacuum (as dark_portal_solver._crossing_bands)
        kinds_p = [dict(n=np.zeros((1, len(edges) - 1)), coef2=0.0, tables=tables[kk]) for kk in range(2)]
        groups = (np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64))
        gstart, gchild = groups
        ng = 1
        line = 0
    else:
        kinds_p = [dict(n=kd['n'], coef2=kd['coef2'], tables=(tables[kk] if tables is not None else None))
                   for kk, kd in enumerate(kinds)]
    cseg = dp_galactic_path._group_cseg(kinds_p, gstart, gchild)            # (2, N_g, N_s): coef2 x mean density
    seglen, L_sun = dp_galactic_path._segments(edges)
    ns = len(seglen)
    if gr is not None:
        Q_all = dp_galactic_path.crossing_populations(E, Pc, edges, kinds_p, gr['rates'], groups)   # (N_g, 1, N, 2, 3)
    else:
        Q_all = np.repeat(Pc[None], ng, axis=0)
    if line is None:
        mean = Q_all.mean(axis=0)
        line = int(np.argmin([np.sum((Q_all[g] - mean) ** 2) for g in range(ng)]))
    # replay the line's segments (dp_galactic_path._cross), recording after each
    F = Pc.copy()
    rec_d, rec_F = [], []

    def record(d_kpc, j):
        rec_d.append(d_kpc)
        rec_F.append([_flavor(E, F[0, :, kk, :], kinds[kk]['tab'], cseg[kk, line, j], 1.0) for kk in range(2)])

    record(float(edges[-1]), ns - 1)
    if gr is not None:
        t0, t1 = kinds_p[0]["tables"], kinds_p[1]["tables"]
        rates = gr['rates']
        lad = rates.get("ladder")
        jmax = 0.0
        for k in range(t0["J"].shape[0]):
            jmax = max(jmax, float(np.max(t0["J"][k])), float(np.max(t1["J"][k])))
        src = np.zeros((1, 1, 2, 3))
        wk = dp_galactic_path._work(N, 1)
        lad_a = dp_galactic_path._lad_arrays(lad)
        args_t = (t0["J"], t0["Gd"], t0["GG"], t0["lam"], t0["log_s0"], t0["inv_dlogs"],
                  t1["J"], t1["Gd"], t1["GG"], t1["lam"], t1["log_s0"], t1["inv_dlogs"],
                  float(rates["jump_unit"]), rates["jump_unit"] > 0.0, jmax, lad_a, lad is not None,
                  float(rates["loss_C"]), rates["loss_C"] != 0.0, False, wk)
        for j in range(ns - 1, -1, -1):
            if seglen[j] > 0.0:
                dp_galactic_path._segment(E, F, seglen[j], cseg[0, line, j], cseg[1, line, j], src, False, *args_t)
            record(float(edges[j]), j)
        if L_sun > 0.0:
            dp_galactic_path._segment(E, F, L_sun, cseg[0, line, 0], cseg[1, line, 0], src, False, *args_t)
    else:
        for j in range(ns - 1, -1, -1):
            record(float(edges[j]), j)
    if not np.allclose(F, Q_all[line], rtol=1e-12, atol=0.0):
        raise RuntimeError("dp_trajectory.galaxy: the replay differs from the solver's crossing")
    # at the Sun: the populations decohered at the transit's local density (as flux_stops' 'galaxy')
    frames = _kind_frames(E, uni)
    Q = np.stack([F[0, :, 0, :], F[0, :, 1, :]])
    n_gal = []
    for kk, fr in enumerate(frames):
        _, _, info = _transit_info(E, fr, np.zeros((N, 3)), None)
        n_gal.append(float(info['n_gal_eV3']))
    rec_d.append(0.0)
    rec_F.append([_flavor(E, Q[kk], frames[kk]['tab'], frames[kk]['coef2'], n_gal[kk]) for kk in range(2)])
    return dict(d_kpc=np.array(rec_d), F_nu=np.stack([r[0] for r in rec_F]), F_nb=np.stack([r[1] for r in rec_F]),
                Q=Q, line=line, processes=gr is not None)


def _sun_reach_au(E, frames, infos, tol=1e-6):
    """How far from the Earth the path starts (AU): where the Sun's density falls to tol of the background plus the
    vacuum term's equivalent density at the highest energy (beyond, it turns no eigenbasis); at least 300 AU."""
    r_far = 300.0
    D = dp_spatial.d_sun_eV_inv
    for fr, info in zip(frames, infos):
        n1, m = info['n_sun_eV3'], fr["m_eff"]
        if n1 <= 0.0 or fr["coef2"] <= 0.0:
            continue
        # sigma = E coef2 n against the smallest splitting, Delta m^2_21 ~ 7.5e-5 eV^2 (the scale that turns states)
        floor = info['n_gal_eV3'] + info['n_moon_eV3'] + 7.5e-5 / (float(np.max(E)) * fr["coef2"])
        r = np.geomspace(D, 1e9 * D, 4000)
        C = n1 * (D / r) * np.exp(-m * (r - D))
        far = r[C > tol * floor]
        if len(far):
            r_far = max(r_far, float(far[-1] / D) + 1.0)
    return r_far


def sun(E, uni, gal, sun_path_cos=0.0, x_max_au=None, n_pos=160):
    """One path past the Sun, at cosine sun_path_cos between its arrival direction and the Sun, from x_max_au (None:
    _sun_reach_au, where the Sun's potential starts to act) to the Earth: dict(x_au (N_pos,) the distance to the Earth
    along the path, F_nu, F_nb, rho (2, N_E, 3, 3) at the Earth's split radius -- after the Earth's exterior, as
    dp_spatial.transit_to_earth for this path). Along the path, the solver's transit through the Sun's potential over
    the Galaxy's background; the last point (x = 0) the solver's transit_to_earth, which adds the Moon's density and
    the Earth's exterior."""
    E = np.asarray(E, dtype=np.float64)
    N = len(E)
    frames = _kind_frames(E, uni)
    if x_max_au is None:
        infos = [_transit_info(E, fr, np.zeros((N, 3)), sun_path_cos)[2] for fr in frames]
        x_max_au = _sun_reach_au(E, frames, infos)
    x = np.geomspace(x_max_au, 1e-4, n_pos - 1) * AU_M          # m, decreasing; then the Earth (x = 0) below
    out_F, rho_end = [], []
    for kk, fr in enumerate(frames):
        rho_ref, n_amb, info = _transit_info(E, fr, gal["Q"][kk], sun_path_cos)
        # along the path the Galaxy's background alone: the solver carries the Moon's density as background over the
        # whole Sun transit (exact for the end point where the transit is adiabatic, the potentials sharing one
        # flavor structure), but the Moon acts only near the Earth; it enters at the last point, the solver's own
        n_bg = info['n_gal_eV3']
        n_sun1, m = info['n_sun_eV3'], fr["m_eff"]
        D = dp_spatial.d_sun_eV_inv
        rho0 = np.empty((N, 3, 3), dtype=np.complex128)
        dp_transit.galactic_map(E, np.ascontiguousarray(gal["Q"][kk]), fr["coef2"], n_bg, fr["tab"]["Q"],
                                fr["tab"]["Mp"], fr["tab"]["s"], fr["tab"]["floor"], rho0)
        Fk = []
        for xm in x:
            xe = xm * dp_spatial.m_to_eV_inv
            r = float(np.sqrt(D * D + xe * xe - 2.0 * D * xe * sun_path_cos))
            ca = (D * sun_path_cos - xe) / r
            C = n_sun1 * (D / r) * np.exp(-m * (r - D)) if n_sun1 > 0.0 else 0.0
            rho, _ = dp_transit.source_transit(E, rho0, fr["tab"], abs(fr["Coef"]), n_bg, C, m, r,
                                               np.array([ca]), np.array([1.0]))
            Fk.append(np.real(np.einsum('iaa->ia', rho)))
        Fk.append(np.real(np.einsum('iaa->ia', rho_ref)))       # at the Earth: the solver's, after its exterior
        out_F.append(np.stack(Fk))
        rho_end.append(rho_ref)
    return dict(x_au=np.append(x / AU_M, 0.0), F_nu=out_F[0], F_nb=out_F[1], rho=np.stack(rho_end))


def earth(E, uni, sun_out, costh=-0.6, n_steps=None):
    """One chord through the Earth to the detector, at cos(zenith) costh (< 0: up-going), from the surface:
    dict(l_km (N_pos,) the distance travelled inside the Earth, L_km the chord's length, F_nu, F_nb), the chord's
    steps as dp_spatial._earth_chord_worker's, starting from sun_out's rho (that Sun path's)."""
    E = np.asarray(E, dtype=np.float64)
    captured = []
    orig = dp_spatial._earth_chord_worker

    def spy(args):
        captured.append(args)
        return orig(args)

    out_F = []
    for kk, (args, kw) in enumerate(uni["handover"]):
        captured.clear()
        dp_spatial._earth_chord_worker = spy
        try:
            kw1 = dict(kw, n_cores=1, external_pool=None, show_progress=False, verbose=0)
            dp_spatial.propagate_earth_spatial(E, np.zeros((len(E), 3)), *args, **kw1)
        finally:
            dp_spatial._earth_chord_worker = orig
        cos_all = np.array([a[0] for a in captured])
        pick = [a for a in captured if abs(a[0] - cos_all[np.argmin(np.abs(cos_all - costh))]) < 1e-12]
        rec_l, rec_F = None, None
        for a in pick:
            (c, L_km, N_steps, E_blk, _, M_rot, n_amb, Coef, S, V_SM, depth, scheme, yk, ext) = a
            idx = np.searchsorted(E, E_blk)
            rho = np.ascontiguousarray(sun_out["rho"][kk][idx])
            Uh = np.zeros((3, 3), dtype=np.complex128)
            Hb = np.zeros((3, 3), dtype=np.complex128)
            Tb = np.zeros((3, 3), dtype=np.complex128)
            if ext is not None:
                dp_spatial.apply_earth_exterior(E_blk, rho, M_rot, Coef, S, n_amb, ext)
            dr_km = L_km / N_steps
            dr = dr_km * 1e3 * dp_spatial.m_to_eV_inv
            ls, Fs = [0.0], [np.real(np.einsum('iaa->ia', rho)).copy()]
            if scheme == "avg":
                cross_ls = dp_spatial._earth_layer_crossings(L_km, c, depth)
                nbar = dp_spatial.chord_density_seq(yk)(L_km, c, N_steps, depth, n_amb, cross_ls,
                                                       dp_spatial._GL10_X, dp_spatial._GL10_W, yk)
            for step in range(N_steps):
                if scheme == "avg":
                    V = np.ascontiguousarray((Coef * nbar[step]) * S + V_SM)
                    dp_spatial._evolve_rho_all_bins(E_blk, rho, M_rot, V, dr, Hb, Uh, Tb)
                else:
                    dp_spatial.step_earth_transit(step * dr_km, L_km, c, dr, E_blk, rho, M_rot, n_amb, Coef, S,
                                                  V_SM, depth, Uh, Hb, Tb, yk)
                ls.append((step + 1) * dr_km)
                Fs.append(np.real(np.einsum('iaa->ia', rho)).copy())
            if rec_F is None:
                rec_l, rec_F = np.array(ls), np.zeros((len(ls), len(E), 3))
            rec_F[:, idx, :] = np.stack(Fs)
        out_F.append(rec_F)
        L_chord = pick[0][1]
    return dict(l_km=rec_l, L_km=float(L_chord), costh=float(costh), F_nu=out_F[0], F_nb=out_F[1])
