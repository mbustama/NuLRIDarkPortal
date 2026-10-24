"""
check_decay_cut.py -- where neutrino decay into the mediator and the model's perturbativity limit fall in the
campaign's planes (notes Sec. 'Relic mass changes and neutrino decay below 0.05 eV', 2026-10-24). Per plane (the
now_ic26 configurations), on its (m_Zd, coupling) grid, from the couplings in the mass basis G' = U^dag G U:
  the longitudinal coupling y_L = max_{i != j} |G'_ij| |m_i - m_j| / m_Zd (the model is cut where y_L >= 1);
  the beam's decay rate at 100 TeV over H0, sum over open channels of |G'_ji|^2 m_i^3/(32 pi M^2)(1-x)^2(1+2x) m_i/E;
  the relics' rest-frame decay rate over H0 (without m_i/E).
Against the stored exclusion (analysis/exclusion.field, pion and fitted compositions): per mass column, the lowest
excluded coupling, and whether the decay (rate >= 0.01 H0) or the cut reaches below it, i.e. into the allowed region.
Masses: the lightest at 1e-6 eV (the largest splittings; the cut is then the widest).

  python3 tools/check_decay_cut.py <campaign_dir> <out.json>
"""
import copy
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]

import numpy as np

H0 = 1.4377e-33
E_BEAM = 1e14


def rates(G, U, m, M):
    Gp = U.conj().T @ G @ U
    yl, beam, relic = 0.0, 0.0, 0.0
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            yl = max(yl, abs(Gp[i, j]) * abs(m[i] - m[j]) / M)
            if m[i] > m[j] + M:
                x = (M / m[i]) ** 2
                g = abs(Gp[j, i]) ** 2 * m[i] ** 3 / (32 * np.pi * M * M) * (1 - x) ** 2 * (1 + 2 * x)
                beam = max(beam, g * m[i] / E_BEAM / H0)
                relic = max(relic, g / H0)
    return yl, beam, relic


def _plane(args):
    p, campaign = args
    import dark_portal_solver as dps
    import dp_absorption
    from analysis import config as config_mod
    from analysis import exclusion as ex
    from dp_core import construct_PMNS
    name = os.path.basename(p)[:-5]
    if True:
        raw = copy.deepcopy(json.load(open(p)))
        raw.pop("exact_profile", None)
        cfg = config_mod.from_dict(raw, p)
        model = cfg.model()
        pl = raw["planes"][0]
        LM = np.linspace(np.log10(pl["x_range"][0]), np.log10(pl["x_range"][1]), 69)
        LY = np.linspace(np.log10(pl["y_range"][0]), np.log10(pl["y_range"][1]), 241)
        yl = np.zeros((len(LY), len(LM)))
        beam = np.zeros_like(yl)
        relic = np.zeros_like(yl)
        for b, ly in enumerate(LY):
            v = {q.name: q.start for q in cfg.parameters}
            v.update({pl["x"]: 1.0, pl["y"]: 10.0 ** ly})
            E, pr = dps.build_solver_params(model._run_cfg(model._overrides(v)))
            G = pr.get("G_coupling")
            if G is None:
                u = np.array([pr["u_e"], pr["u_mu"], pr["u_tau"]], dtype=np.complex128)
                G = dp_absorption.dark_portal_coupling(pr["g_D"], u, pr["epsilon_prime"])
            G = np.asarray(G, dtype=np.complex128)
            U = np.asarray(construct_PMNS(pr["th12"], pr["th13"], pr["th23"], pr["d_cp"]))
            m = np.asarray(pr["m_nu_bg_arr"], dtype=float)
            for a, lm in enumerate(LM):
                yl[b, a], beam[b, a], relic[b, a] = rates(G, U, m, 10.0 ** lm)
        res = dict(LM=LM.tolist(), LY=LY.tolist(), cols={})
        for comp, nm in (("fitted", name), ("pion", name + "_fx")):
            path = ex.plane(campaign, "now_ic26", nm)
            if not path:
                continue
            lx, Gf, F, _, _ = ex.field(path)
            for k, lm in enumerate(lx):
                col = F[:, k]
                if np.ma.is_masked(col) and col.mask.all():
                    continue
                exc = np.asarray(col.filled(-1.0)) > 0
                low = float(Gf[np.argmax(exc)]) if exc.any() else None
                a = int(np.argmin(abs(LM - lm)))
                dec = LY[beam[:, a] >= 0.01]
                cut = LY[yl[:, a] >= 1.0]
                res["cols"].setdefault(comp, []).append(dict(
                    logm=float(lm), lowest_excluded=low,
                    decay_from=float(dec.min()) if len(dec) else None,
                    cut_from=float(cut.min()) if len(cut) else None))
        c = res["cols"].get("pion", [])
        hit_d = [r["logm"] for r in c if r["decay_from"] is not None and (r["lowest_excluded"] is None
                                                                         or r["decay_from"] < r["lowest_excluded"])]
        hit_c = [r["logm"] for r in c if r["cut_from"] is not None and (r["lowest_excluded"] is None
                                                                       or r["cut_from"] < r["lowest_excluded"])]
        print(f"{name}: decay reaches the allowed region at log m {min(hit_d) if hit_d else '-'}..{max(hit_d) if hit_d else '-'}"
              f" ({len(hit_d)} cols); cut at {min(hit_c) if hit_c else '-'}..{max(hit_c) if hit_c else '-'} ({len(hit_c)} cols)",
              flush=True)
    return name, res


def main(campaign, out_path, only=None):
    from multiprocessing import get_context
    todo = [(p, campaign) for p in sorted(glob.glob(os.path.join(REPO, "analysis", "configs", "campaign_ml", "now_ic26",
                                                                 "*.json")))
            if not (p.endswith("_fx.json") or "_mev" in p or p.endswith("campaign.json"))]
    if only is not None:                     # one plane per array task: <out_path>.<k>
        name, res = _plane(todo[only])
        json.dump({name: res}, open(f"{out_path}.{only}", "w"))
        return 0
    out = {}
    with get_context("spawn").Pool(min(32, len(todo))) as pool:
        for name, res in pool.imap_unordered(_plane, todo):
            out[name] = res
    json.dump(out, open(out_path, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None))
