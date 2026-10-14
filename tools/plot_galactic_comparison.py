"""
plot_galactic_comparison.py -- what the Galactic component does to the limits (Mauricio's task 2, 2026-10-14): every
plane of output/analysis/<name> against its rerun with the Galactic component, output/analysis/<name>_gal
(tools/make_galactic_configs.py: CRINGE x norm_gal, sources (x, 1 - x, 0), the extragalactic flux anchored to
IceCube 2026 with norm_astro; the three nuisances profiled at every point). Per plane, the 95% upper limit per
mass column (the upper end of the lowest allowed interval at Delta chi^2 = 3.84, as the other summaries) with and
without, their ratio, and the best limit. Writes output/analysis/galactic_comparison.json and
output/figures/galactic_comparison.png.

  python tools/plot_galactic_comparison.py
"""
import glob
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FAMILIES = [("si_vector_dirac_", "SI vector, Dirac"), ("si_vector_", "SI vector, Majorana"),
            ("si_dirac_", "SI scalar, Dirac"), ("si_", "SI scalar, Majorana"),
            ("potential_", "dark portal, potential planes"), ("mev_", "dark portal, absorption (MeV)"),
            ("exclusion", "dark portal, exclusion"), ("", "other")]


def limit_line(path):
    s = json.load(open(path))
    if not s.get("complete", False):
        return None
    x = np.array(s["grids"][0])
    lim = []
    for col in s["columns"]:
        ivs = col["intervals"].get("95%", [])
        bad = (not ivs) or (col["value"] in s.get("elevated", [])) or ivs[0]["hi_open"]
        lim.append(np.nan if bad else ivs[0]["hi"])
    return x, np.array(lim), s["names"]


def family(name):
    for key, label in FAMILIES:
        if name.startswith(key):
            return label
    return "other"


def main():
    rows = {}
    for gal_dir in sorted(glob.glob(os.path.join(REPO, "output", "analysis", "*_gal"))):
        name = os.path.basename(gal_dir)[:-4]
        base_dir = os.path.join(REPO, "output", "analysis", name)
        for gpath in sorted(glob.glob(os.path.join(gal_dir, "data", "plane_*.json"))):
            bpath = os.path.join(base_dir, "data", os.path.basename(gpath))
            if not os.path.exists(bpath):
                continue
            g, b = limit_line(gpath), limit_line(bpath)
            if g is None or b is None:
                print(f"skipped (plane not complete): {gpath if g is None else bpath}")
                continue
            (xg, lg, names), (xb, lb, _) = g, b
            if not np.array_equal(xg, xb):
                continue
            ok = np.isfinite(lg) & np.isfinite(lb)
            r = lg[ok] / lb[ok]
            short = name.replace("pion_2050_case_5_", "").replace("pion_2050_case_5", "(base)")
            rows[f"{short}:{os.path.basename(gpath)[6:-5]}"] = dict(
                family=family(short), axes=names, x=xg.tolist(), lim=lb.tolist(), lim_gal=lg.tolist(),
                median_ratio=float(np.median(r)) if r.size else None, min_ratio=float(r.min()) if r.size else None,
                max_ratio=float(r.max()) if r.size else None, best=float(np.nanmin(lb)) if np.isfinite(lb).any() else None,
                best_gal=float(np.nanmin(lg)) if np.isfinite(lg).any() else None,
                columns_lost=int(np.sum(np.isfinite(lb) & ~np.isfinite(lg))),
                columns_gained=int(np.sum(~np.isfinite(lb) & np.isfinite(lg))))
    out = {k: {kk: vv for kk, vv in v.items() if kk not in ("x", "lim", "lim_gal")} for k, v in rows.items()}
    json.dump(out, open(os.path.join(REPO, "output", "analysis", "galactic_comparison.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:45s} median gal/no-gal {v['median_ratio']}, range [{v['min_ratio']}, {v['max_ratio']}], "
              f"best {v['best']} -> {v['best_gal']}, columns lost {v['columns_lost']}, gained {v['columns_gained']}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [lab for _, lab in FAMILIES if any(v["family"] == lab for v in rows.values())]
    fig, axes = plt.subplots(2, (len(fams) + 1) // 2, figsize=(4.2 * ((len(fams) + 1) // 2), 7.6), squeeze=False)
    for ax, lab in zip(axes.ravel(), fams):
        for k, v in rows.items():
            if v["family"] != lab:
                continue
            x, lb, lg = np.array(v["x"]), np.array(v["lim"]), np.array(v["lim_gal"])
            ax.plot(x, lg / lb, lw=1.3, label=k.split(":")[0])
        ax.axhline(1, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel(rows[next(k for k, v in rows.items() if v["family"] == lab)]["axes"][0])
        ax.set_ylabel("95% limit: with / without Galactic")
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    fig.suptitle("The Galactic component in the flavor fits: ratio of the 95% upper limits per mass column "
                 "(>1: weaker with it)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "output", "figures", "galactic_comparison.png"), dpi=110)
    print("wrote output/analysis/galactic_comparison.json, output/figures/galactic_comparison.png")


if __name__ == "__main__":
    main()
