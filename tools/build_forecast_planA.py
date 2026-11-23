"""
build_forecast_planA.py -- audit finding C1 (docs/dev/AUDIT_2026-11-08_resolution.md; his choice 2026-11-11: tracks in
energy and declination band, showers and double cascades in energy only): the eight future telescopes' responses to a
diffuse flux, from NuFlavorSingleSource's Plan A detector cards (its final code, commit 56e3215, run unchanged).

A card's fine output for a point source at declination dec (blocks.fine_<card>: per live year, per unit flux integral
over each true-energy bin) gives, for each topology,
  K[(channel, flavor)] (n_true, n_reco, n_psi)   events by measured energy and ring psi around the source;
  B[component] (n_reco, n_psi)                   backgrounds (atmosphere in pieces, muons, published backgrounds).
A diffuse flux Phi(E, dec) per sr gives sum over directions of Phi dOmega times the point response, so each direction's
events are K summed over all rings. Per (card, dec) this tool keeps, for each topology:
  K_all[(channel, flavor)]  (n_true, n_reco)  K summed over psi;
  B_all[component]          (n_reco,)         the whole-sky background (all rings);
  B_cap[component]          (n_reco,)         the background within CAP_DEG of dec, and cap_sr its solid angle: the
                                              background density at dec (tracks, whose pointing is well below CAP_DEG);
  the same for Bmisid; misid, pub, the edges, the card's info.
The site, the Earth's absorption by local zenith and the rotation through the day are the card's own.

Usage (Tycho, Plan A's environment; the deposit variants set their environment as code/make_blocks.py variant_env):
  python tools/build_forecast_planA.py --plana /lustre/astro/mbustama/plan_a_final_blocks --card arca \
      --decs -40,40 --out DIR [--variant deposit_nc+]
writes DIR/<card>__<variant or nominal>__<dec>.pkl
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

CAP_DEG = 5.0


def reduce_fine(F, psi_edges):
    psi = np.asarray(psi_edges, float)
    i_cap = int(np.argmin(np.abs(psi - CAP_DEG)))
    cap_deg = float(psi[i_cap])
    out = {}
    for topo, f in F.items():
        out[topo] = dict(
            lgt_edges=np.asarray(f.lgt_edges), lgr_edges=np.asarray(f.lgr_edges),
            K_all={k: np.asarray(v).sum(axis=-1) for k, v in f.K.items()},
            B_all={k: np.asarray(v).sum(axis=-1) for k, v in f.B.items()},
            B_cap={k: np.asarray(v)[..., :i_cap].sum(axis=-1) for k, v in f.B.items()},
            Bmisid_all={k: np.asarray(v).sum(axis=-1) for k, v in (f.Bmisid or {}).items()},
            Bmisid_cap={k: np.asarray(v)[..., :i_cap].sum(axis=-1) for k, v in (f.Bmisid or {}).items()},
            misid=tuple(f.misid), pub=dict(f.pub or {}), info=dict(f.info or {}),
            cap_deg=cap_deg, cap_sr=float(2.0 * np.pi * (1.0 - np.cos(np.radians(cap_deg)))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plana", required=True, help="Plan A's repository copy (its code/ is imported)")
    ap.add_argument("--card", required=True)
    ap.add_argument("--decs", required=True, help="comma-separated declinations (deg)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default=None, help="a deposit variant's tag (its environment set by the caller)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.plana, "code"))
    os.chdir(a.plana)
    import blocks
    fine_fn = getattr(blocks, "fine_" + a.card)
    os.makedirs(a.out, exist_ok=True)
    tag = a.variant or "nominal"
    for dec in (float(x) for x in a.decs.split(",")):
        t0 = time.time()
        F = fine_fn(dec, tracks=a.variant is None)
        red = reduce_fine(F, blocks.PSI_FINE)
        red["_meta"] = dict(card=a.card, dec=dec, variant=tag, seconds=time.time() - t0,
                            deposit_scale=os.environ.get("PLANA_DEPOSIT_SCALE", ""))
        path = os.path.join(a.out, f"{a.card}__{tag}__{dec:g}.pkl")
        with open(path + ".tmp", "wb") as fh:
            pickle.dump(red, fh)
        os.replace(path + ".tmp", path)
        shapes = {t: {k: v.shape for k, v in r["K_all"].items()} for t, r in red.items() if t != "_meta"}
        comps = {t: sorted(r["B_all"]) for t, r in red.items() if t != "_meta"}
        print(f"{a.card} {tag} dec {dec:g}: {time.time() - t0:.0f} s; K {shapes}; B {comps}", flush=True)


if __name__ == "__main__":
    main()
