"""Per-energy fluxes of validate_transport's variants at chosen points (2026-10-25): A (default), B (steps x4 / dp_si
substeps x2), C (energy grid x2), H (grid extended a decade above 30 PeV), D (the other transport where both hold),
saved per point for analysis. Arguments: units.json task n_tasks out_dir. Run on Tycho."""
import copy
import json
import os
import sys
from multiprocessing import get_context

sys.path[:0] = [os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "src"),
                os.path.join(os.path.dirname(__file__), "..")]
os.environ.pop("NULRI_FLUX_CACHE", None)
import numpy as np  # noqa: E402


def _one(u):
    import dark_portal_solver as dps
    import validate_transport as vt
    out = {}
    try:
        cfg, m = vt._model(u["fam"])
        v = {q.name: q.start for q in cfg.parameters}
        v.update({u["x"]: u["xv"], u["y"]: u["yv"], "m_lightest_eV": u["ml"]})
        base = m._run_cfg(m._overrides(v))
        A, pA = vt._flux(m, base)
        out["E_A"], out["A"] = A["E"], A["F"]
        n0 = int(base["grid"]["n_z_bins"])
        cB = copy.deepcopy(base)
        cB["grid"]["n_z_bins"] = (n0 - 1) * 4 + 1
        B, _ = vt._flux(m, cB, dict(si=dict(z_substeps=2, E_extension=10.0)) if A["mode"] in ("si", "combined") else None)
        out["B"] = B["F"]
        cC = copy.deepcopy(base)
        cC["grid"]["n_E_bins"] = 2 * (int(base["grid"]["n_E_bins"]) - 1) + 1
        C, _ = vt._flux(m, cC)
        out["E_C"], out["C"] = C["E"], C["F"]
        cH = copy.deepcopy(base)
        nE0 = int(base["grid"]["n_E_bins"])
        r = (float(base["grid"]["E_max_eV"]) / float(base["grid"]["E_min_eV"])) ** (1.0 / (nE0 - 1))
        cH["grid"]["E_max_eV"] = float(base["grid"]["E_max_eV"]) * r ** 115
        cH["grid"]["n_E_bins"] = nE0 + 115
        H, _ = vt._flux(m, cH)
        out["E_H"], out["H"] = H["E"], H["F"]
        if A["mode"] in ("dark_portal", "si"):
            other = "si" if A["mode"] == "dark_portal" else "dark_portal"
            try:
                D, _ = vt._flux(m, base, dict(cosmo_transport=other))
                out["D"] = D["F"]
            except Exception as e:
                out["D_error"] = repr(e)[:120]
        std = vt._std(m, v, u["y"], float(vt._axes(cfg)[3][0]), False)
        out["std"] = std["F"]
        out["mode"] = A["mode"]
    except Exception as e:
        out["error"] = repr(e)[:300]
    return u, out


def main():
    units = json.load(open(sys.argv[1]))
    task, n = int(sys.argv[2]), int(sys.argv[3])
    od = sys.argv[4]
    os.makedirs(od, exist_ok=True)
    share = units[task::n]
    with get_context("spawn").Pool(min(32, len(share))) as pool:
        for u, out in pool.imap_unordered(_one, share):
            name = f"{u['fam']}_{u['ix']}_{u['iy']}_{u['ml']:g}"
            np.savez_compressed(os.path.join(od, name + ".npz"), **{k: v for k, v in out.items() if not isinstance(v, str)},
                                meta=json.dumps(dict(u, **{k: v for k, v in out.items() if isinstance(v, str)})))
            print(name, out.get("mode"), out.get("error", ""), out.get("D_error", ""), flush=True)


if __name__ == "__main__":
    main()
