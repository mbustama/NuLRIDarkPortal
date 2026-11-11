"""
assemble_forecast_planA.py -- C1 step 2 (docs/dev/C1_FORECAST_PORT.md): the per-declination outputs of
tools/build_forecast_planA.py turned into per-band blocks with NuFlavorSingleSource's own blocks.assemble (Plan A's
final code, unchanged), for the diffuse flux of the declination bands of our solver.

Per card, topology and band b (3 declinations d_j, solid angles dOmega_j = 2 pi d(sin dec) / 3):
  K_b = sum_j dOmega_j K_all(d_j)                     signal per unit flux integral per sr (all rings);
  tracks:  B_b = sum_j dOmega_j B_cap(d_j) / cap_sr   the background whose measured direction lies in the band;
  showers, double cascades (energy only): B_b = B_all dOmega_b / 4 pi (the whole sky split by solid angle: the
           analysis sums the bands); B_all averaged over the declinations, its spread recorded.
Each band's Fine has one ring (0-180 deg); its energy bins are the card's (blocks.analysis_binning on the whole sky's
signal for Plan A's nominal diffuse flux, blocks.astro_flux, from the sample's threshold make_blocks.lg_min_of). The
diffuse flux, which Plan A counts as a background, is the signal here: its pieces ('astro_*', 'd:astro_*') are zeroed
in the backgrounds and kept apart for the check against Plan A's own diffuse events. The Glashow channels ('*gr*') are
assembled apart too (Rb_gr), so the likelihood can weight them by the nubar_e flux. Deposit nuisances from the
variants as make_blocks.combined: RXb = (Rb+ - Rb-) / (2 H_DEPOSIT), BX from our model's neutrino pieces.

Usage (Tycho, Plan A's environment): python tools/assemble_forecast_planA.py --plana DIR --in OUT --card arca --decs decs.txt
writes OUT/../blocks/<card>.pkl
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plana", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--decs", required=True, help="lines 'band d1,d2,d3'")
    ap.add_argument("--band-edges-sin", default=None, help="comma-separated sin(dec) edges (default: -1..1 in 0.1)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.plana, "code"))
    os.chdir(a.plana)
    import blocks
    import make_blocks
    sin_e = (np.linspace(-1, 1, 21) if a.band_edges_sin is None
             else np.array([float(x) for x in a.band_edges_sin.split(",")]))
    bands = [(int(l.split()[0]), [float(x) for x in l.split()[1].split(",")]) for l in open(a.decs) if l.strip()]
    d_omega_band = 2 * np.pi * np.diff(sin_e)

    def load(variant, dec):
        p = os.path.join(a.inp, f"{a.card}__{variant}__{dec:g}.pkl")
        return pickle.load(open(p, "rb")) if os.path.exists(p) else None

    nom = {(b, d): load("nominal", d) for b, ds in bands for d in ds}
    missing = [k for k, v in nom.items() if v is None]
    if missing:
        raise SystemExit(f"{a.card}: {len(missing)} nominal outputs missing, e.g. {missing[:3]}")
    topos = sorted({t for v in nom.values() for t in v if t != "_meta"})
    is_astro = lambda k: k.startswith("astro_") or k.startswith("d:astro_")
    det = make_blocks.DET[a.card]
    out = dict(card=a.card, det=det, sin_edges=sin_e, lge_pts=blocks.LGE_PTS, topos={}, H_DEPOSIT=blocks.H_DEPOSIT)
    for topo in topos:
        base = blocks.base_topo(topo)
        energy_only = base in ("shw", "dc")
        decs_all = [(b, d) for b, ds in bands for d in ds if topo in nom[(b, d)]]
        r0 = nom[decs_all[0]][topo]
        lgt, lgr = r0["lgt_edges"], r0["lgr_edges"]
        keys = sorted({k for bd in decs_all for k in nom[bd][topo]["K_all"]})
        comps = sorted({k for bd in decs_all for k in nom[bd][topo]["B_all"]})
        zK = np.zeros((len(lgt) - 1, len(lgr) - 1))
        zB = np.zeros(len(lgr) - 1)
        # the whole sky's background (showers, double cascades): the same from every declination; its spread
        if energy_only:
            # a card counts a background over the hours its source is in view, so the whole sky's full-day
            # background is the one of a declination always in view: the largest (2026-11-11; IceCube's and ARCA's
            # showers, always in view, give the same at every declination). Check: the sum over the bands of the
            # background densities near each declination (B_cap / cap_sr), which counts every direction while in view.
            # the physical backgrounds only (not the derivative templates 'd:*'; code review 2026-11-11)
            phys = [k for k in comps if not is_astro(k) and not k.startswith("d:")]
            tot = np.array([sum(nom[bd][topo]["B_all"][k].sum() for k in phys) for bd in decs_all])
            i_max = int(np.argmax(tot))
            Ball = {k: nom[decs_all[i_max]][topo]["B_all"].get(k, zB) for k in comps}
            Bm_all = {k: v for k, v in nom[decs_all[i_max]][topo]["Bmisid_all"].items()}
            dens = sum(d_omega_band[b] / len(ds) * sum(nom[(b, d)][topo]["B_cap"][k].sum() for k in phys)
                       / nom[(b, d)][topo]["cap_sr"]
                       for b, ds in bands for d in ds if topo in nom[(b, d)])
            spread = dict(dec_of_max=float(decs_all[i_max][1]), max_over_min=float(tot.max() / max(tot.min(), 1e-300)),
                          density_sum_over_max=float(dens / max(tot.max(), 1e-300)))

        def whole_sky(variant):
            """(B_all, Bmisid_all) of showers or double cascades for a variant ('nominal' or a deposit variant), at the
            nominal's always-in-view declination (2026-11-11, code review: the variants' own backgrounds, so the deposit
            nuisances' background templates are not zero)."""
            if variant == "nominal":
                return Ball, Bm_all
            r = load(variant, decs_all[i_max][1])
            if r is None or topo not in r:
                return None
            return ({k: r[topo]["B_all"].get(k, zB) for k in comps}, dict(r[topo]["Bmisid_all"]))

        def band_fine(b, ds, src, which="all", variant="nominal"):
            """One band's Fine of one ring from the outputs `src` ((b, d) -> reduced); which: 'all' with the
            backgrounds, 'gr' the Glashow channels alone without backgrounds; variant: whose whole-sky background
            (showers, double cascades)."""
            dO = d_omega_band[b] / len(ds)
            K = {}
            for k in keys:
                gr = "gr" in k[0]
                if (which == "gr") != gr and which != "all":
                    continue
                K[k] = sum(dO * src[(b, d)][topo]["K_all"].get(k, zK) for d in ds)[:, :, None]
            if which == "gr":
                B, Bm = {k: zB[:, None] for k in comps}, {}
            elif energy_only:
                frac = d_omega_band[b] / (4 * np.pi)
                Bw, Bmw = whole_sky(variant)
                B = {k: (zB if is_astro(k) else frac * Bw[k])[:, None] for k in comps}
                Bm = {k: (zB if is_astro(k) else frac * Bmw[k])[:, None] for k in Bmw}
            else:
                B = {k: (zB if is_astro(k) else sum(dO * src[(b, d)][topo]["B_cap"].get(k, zB) / src[(b, d)][topo]["cap_sr"]
                                                     for d in ds))[:, None] for k in comps}
                mk = sorted({k for d in ds for k in src[(b, d)][topo]["Bmisid_cap"]})
                Bm = {k: (zB if is_astro(k) else sum(dO * src[(b, d)][topo]["Bmisid_cap"].get(k, zB)
                                                     / src[(b, d)][topo]["cap_sr"] for d in ds))[:, None] for k in mk}
            r = src[(b, ds[0])][topo]
            misid = tuple(k for k in r["misid"] if k in K)
            return blocks.Fine(det, topo, lgt, lgr, [0.0, 180.0], K, B, misid=misid, Bmisid=Bm, pub=r["pub"],
                               info={k: v for k, v in r["info"].items() if not isinstance(v, np.ndarray)})

        # Plan A's diffuse astrophysical events per band (its 'astro_<fl>' backgrounds), for the check
        def astro_events(b, ds):
            dO = d_omega_band[b] / len(ds)
            if energy_only:
                return {k: Ball[k] * d_omega_band[b] / (4 * np.pi) for k in comps if k.startswith("astro_")}
            return {k: sum(dO * nom[(b, d)][topo]["B_cap"].get(k, zB) / nom[(b, d)][topo]["cap_sr"] for d in ds)
                    for k in comps if k.startswith("astro_")}

        # common energy bins: the whole sky's nominal diffuse signal, one ring
        phi = blocks.phi_integrals(lgt, blocks.astro_flux)
        sig = 0.0
        for b, ds in bands:
            ds = [d for d in ds if topo in nom[(b, d)]]
            if ds:
                sig = sig + sum(np.tensordot(phi, v, axes=(0, 0)) for v in band_fine(b, ds, nom).K.values())
        try:
            bn = blocks.analysis_binning(sig, lgr, np.array([0.0, 180.0]), lg_min=make_blocks.lg_min_of(a.card, base, "main"))
        except blocks.EmptyBlock as e:
            print(f"{a.card} {topo}: left out ({e})", flush=True)
            continue
        binning = blocks.Binning(bn.lg_edges, np.tile([0.0, 180.0], (len(bn.lg_edges) - 1, 1)))
        nuis = blocks.nuisance_table(det, base, part=make_blocks.PART.get(a.card))
        rec = dict(lgt_edges=lgt, lg_edges=binning.lg_edges, energy_only=energy_only, bands=[],
                   spread=spread if energy_only else None, keys=keys)
        for b, ds in bands:
            ds = [d for d in ds if topo in nom[(b, d)]]
            if not ds:
                rec["bands"].append(None)
                continue
            blk = blocks.assemble(band_fine(b, ds, nom), 1.0, nuis, binning=binning)
            gr_keys = [k for k in keys if "gr" in k[0]]
            blk_gr = blocks.assemble(band_fine(b, ds, nom, "gr"), 1.0, nuis, binning=binning) if gr_keys else None
            entry = dict(Rb=blk.Rb, RXb=blk.RXb, B_total=blk.B_total, BX=blk.BX, kinds=blk.kinds, info=blk.info,
                         Rb_gr=None if blk_gr is None else blk_gr.Rb,
                         RXb_gr=None if blk_gr is None else blk_gr.RXb, astro=astro_events(b, ds))
            # the deposit nuisances (showers, double cascades): central differences of the rebuilt card
            for p in blocks.DEPOSIT_NUISANCES:
                vp = {(b, d): load(p + "+", d) for d in ds}
                vm = {(b, d): load(p + "-", d) for d in ds}
                if any(v is None or topo not in v for v in list(vp.values()) + list(vm.values())) or \
                        (energy_only and (whole_sky(p + "+") is None or whole_sky(p + "-") is None)):
                    entry.setdefault("missing_deposit", []).append(p)
                    continue
                bp = blocks.assemble(band_fine(b, ds, vp, variant=p + "+"), 1.0, nuis, binning=binning)
                bm = blocks.assemble(band_fine(b, ds, vm, variant=p + "-"), 1.0, nuis, binning=binning)
                h = blocks.H_DEPOSIT
                entry["RXb"][p] = {f: (bp.Rb[f] - bm.Rb[f]) / (2 * h) for f in blocks.FLAVORS}
                mod = [k for k in blk.info["nu_parts"] if k != "pub_nu"]
                entry["BX"][p] = sum((bp.B[k] - bm.B[k]) / (2 * h) for k in mod)
                entry["kinds"][p] = "deposit"
                if blk_gr is not None:
                    g = lambda v: blocks.assemble(band_fine(b, ds, v, "gr"), 1.0, nuis, binning=binning).Rb
                    gp, gm = g(vp), g(vm)
                    entry["RXb_gr"][p] = {f: (gp[f] - gm[f]) / (2 * h) for f in blocks.FLAVORS}
            rec["bands"].append(entry)
        out["topos"][topo] = rec
        n_ok = sum(e is not None for e in rec["bands"])
        print(f"{a.card} {topo}: {n_ok} bands, {len(binning.lg_edges) - 1} energy bins "
              f"{binning.lg_edges[0]:.2f}-{binning.lg_edges[-1]:.2f}; spread {rec['spread']}", flush=True)
    # the priors of every nuisance the blocks carry (Plan A's fit.prior_of: kind, centre, width, bounds as deviations)
    import fit
    names = set()
    for rec in out["topos"].values():
        for e in rec["bands"]:
            if e is not None:
                names |= {p for p in list(e["RXb"]) + list(e["BX"]) if not p.startswith("astro_")}
    out["priors"] = {}
    for n in sorted(names):
        k, c, w, bd = fit.prior_of(n)
        out["priors"][n] = dict(kind=k, centre=float(c), width=float(w), lo=float(bd[0]), hi=float(bd[1]))
    dst = os.path.join(os.path.dirname(os.path.normpath(a.inp)), "blocks")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, f"{a.card}.pkl.tmp"), "wb") as fh:
        pickle.dump(out, fh)
    os.replace(os.path.join(dst, f"{a.card}.pkl.tmp"), os.path.join(dst, f"{a.card}.pkl"))


if __name__ == "__main__":
    main()
