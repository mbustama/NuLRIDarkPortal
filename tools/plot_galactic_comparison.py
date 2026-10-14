"""
plot_galactic_comparison.py -- what the Galactic component does to the limits (Mauricio's task 2; redesigned
2026-10-14: tools/make_galactic_configs.py). Two comparisons per plane, of the 95% upper limit per mass column (the
upper end of the lowest allowed interval at Delta chi^2 = 3.84, as the other summaries):

  _gal / original   the nominal analysis with the Galactic component (CRINGE x norm_gal, compositions at pion
                    decay) and IceCube 2026's spectrum as data, the extragalactic normalization and index floating,
                    against the original flavor-only fit (no Galactic component, spectrum fixed);
  _galx / _gal      both source compositions free, (x, 1 - x, 0) each, against pion decay.

Writes output/analysis/galactic_comparison.json and output/figures/galactic_comparison.png. Incomplete planes are
skipped (reported).

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
        # read per mass from the column's own minimum, as the planes are, also in "elevated" columns: when the data
        # prefer new physics at some point of the plane (2026-10-14: IceCube 2026's spectrum, an SI resonance near
        # 3 MeV, Delta chi^2 ~ 5.8), every other column's minimum lies above the plane's, yet its lowest couplings
        # reach the no-effect region and its limit stands. No limit only without an upper end in the scanned range.
        ivs = col["intervals"].get("95%", [])
        bad = (not ivs) or ivs[0]["hi_open"]
        lim.append(np.nan if bad else ivs[0]["hi"])
    return x, np.array(lim), s["names"]


def preference(path):
    """chi^2 without new physics (the median over columns of the lowest coupling's) minus the plane's minimum: how
    much the data prefer new physics somewhere in the plane; and where."""
    s = json.load(open(path))
    c = np.array(s["chi2"], dtype=float)
    k = np.unravel_index(np.nanargmin(c), c.shape)
    return float(np.nanmedian(c[:, 0]) - np.nanmin(c)), [float(s["grids"][0][k[0]]), float(s["grids"][1][k[1]])]


def family(name):
    for key, label in FAMILIES:
        if name.startswith(key):
            return label
    return "other"


def compare(a_dir, b_dir, label):
    """Rows {plane key: ...} for the planes of b_dir against the same planes in a_dir (b over a)."""
    rows = {}
    for bpath in sorted(glob.glob(os.path.join(b_dir, "data", "plane_*.json"))):
        apath = os.path.join(a_dir, "data", os.path.basename(bpath))
        if not os.path.exists(apath):
            continue
        a, b = limit_line(apath), limit_line(bpath)
        if a is None or b is None:
            print(f"skipped (plane not complete): {bpath if b is None else apath}")
            continue
        (xa, la, names), (xb, lb, _) = a, b
        if not np.array_equal(xa, xb):
            continue
        ok = np.isfinite(la) & np.isfinite(lb)
        r = lb[ok] / la[ok]
        short = os.path.basename(a_dir).replace("pion_2050_case_5_", "").replace("pion_2050_case_5", "(base)")
        short = short.replace("_galx", "").replace("_gal", "")
        rows[f"{label}|{short}:{os.path.basename(bpath)[6:-5]}"] = dict(
            comparison=label, family=family(short), axes=names, x=xa.tolist(), lim_a=la.tolist(), lim_b=lb.tolist(),
            median_ratio=float(np.median(r)) if r.size else None, min_ratio=float(r.min()) if r.size else None,
            max_ratio=float(r.max()) if r.size else None,
            best_a=float(np.nanmin(la)) if np.isfinite(la).any() else None,
            best_b=float(np.nanmin(lb)) if np.isfinite(lb).any() else None,
            columns_lost=int(np.sum(np.isfinite(la) & ~np.isfinite(lb))),
            columns_gained=int(np.sum(~np.isfinite(la) & np.isfinite(lb))),
            preference_a=preference(apath), preference_b=preference(bpath))
    return rows


def main():
    base = os.path.join(REPO, "output", "analysis")
    rows = {}
    for gal_dir in sorted(glob.glob(os.path.join(base, "*_gal"))):
        name = gal_dir[:-4]
        rows.update(compare(name, gal_dir, "_gal / original"))
        if os.path.isdir(name + "_galx"):
            rows.update(compare(gal_dir, name + "_galx", "_galx / _gal"))
    out = {k: {kk: vv for kk, vv in v.items() if kk not in ("x", "lim_a", "lim_b")} for k, v in rows.items()}
    with open(os.path.join(base, "galactic_comparison.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k:60s} median {v['median_ratio']}, range [{v['min_ratio']}, {v['max_ratio']}], best {v['best_a']} -> "
              f"{v['best_b']}, columns lost {v['columns_lost']}, gained {v['columns_gained']}; preference for new "
              f"physics (Delta chi^2, where) {v['preference_a'][0]:.2f} -> {v['preference_b'][0]:.2f} at "
              f"{v['preference_b'][1]}")
    if not rows:
        print("no complete pairs of planes yet")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comps = ["_gal / original", "_galx / _gal"]
    fams = [lab for _, lab in FAMILIES if any(v["family"] == lab for v in rows.values())]
    fig, axes = plt.subplots(len(comps), len(fams), figsize=(3.6 * len(fams), 3.4 * len(comps)), squeeze=False)
    for i, comp in enumerate(comps):
        for ax, lab in zip(axes[i], fams):
            sel = {k: v for k, v in rows.items() if v["family"] == lab and v["comparison"] == comp}
            for k, v in sel.items():
                x, la, lb = np.array(v["x"]), np.array(v["lim_a"]), np.array(v["lim_b"])
                ax.plot(x, lb / la, lw=1.2, label=k.split("|")[1].split(":")[0])
                lost = np.isfinite(la) & ~np.isfinite(lb)
                if lost.any():
                    ax.plot(x[lost], np.full(lost.sum(), 1e3), "v", ms=3, color=ax.lines[-1].get_color())
            ax.axhline(1, color="k", lw=0.6)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-2, 2e3)
            ax.set_title(f"{lab}: {comp}", fontsize=8)
            if sel:
                ax.set_xlabel(next(iter(sel.values()))["axes"][0], fontsize=8)
                ax.legend(fontsize=5.5)
            ax.set_ylabel("ratio of 95% limits", fontsize=8)
    fig.suptitle("Task 2: the 95% upper limits per mass column, ratios (>1: weaker). Top: with the Galactic component and "
                 "IceCube 2026's spectrum as data, against the original fits; bottom: compositions free against pion "
                 "decay. Triangles at 1e3: no limit within the scanned couplings. The dark portal's planes use the "
                 "physics before the audit of 2026-10-14 (superseded).", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "output", "figures", "galactic_comparison.png"), dpi=110)
    print("wrote output/analysis/galactic_comparison.json, output/figures/galactic_comparison.png")


if __name__ == "__main__":
    main()
