"""
campaign_summary.py -- the plane campaign's results (docs/dev/SCOPE_WIDENING_PLAN.md, step 6): for one data set,

1. the profiled curve of every potential plane (tools/combine_split_planes.py): the balanced run and its coupling-split
   variants, per model, texture, sign and composition treatment (nominal / fitted);
2. the 95% upper bound per mediator mass of every plane (the first allowed interval's upper end, read per mass as
   the planes are), into bounds.json;
3. figures: the potential's bounds (balanced and profiled, both signs) per texture, the MeV absorption's per
   texture, and the direct gauge models' over the whole mass range.

  python tools/campaign_summary.py --set now_ic26 [--fx]

Reads output/analysis/campaign/<set>/<plane>/data/; writes output/analysis/campaign/<set>/summary/.
Planes not finished (no plane_*.json) are skipped and listed.
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

import combine_split_planes as CS  # noqa: E402
from make_campaign_configs import TEXTURES, MODEL_KEYS, SPLITS  # noqa: E402

LEVEL = "95%"


def plane_json(run_dir):
    hits = sorted(glob.glob(os.path.join(run_dir, "data", "plane_*.json")))
    hits = [h for h in hits if not h.endswith("_profiled_split.json")]
    return hits[0] if hits else None


def bound_curve(summary):
    """(masses, upper bounds) at LEVEL: per mass column, the upper end of the lowest allowed interval (None where
    the column has none, or where it is open above: no bound within the scanned range)."""
    m, b = [], []
    for col in summary["columns"]:
        ivs = col.get("intervals", {}).get(LEVEL) or []
        m.append(col["value"])
        if not ivs:
            b.append(None)
            continue
        iv = ivs[0]
        b.append(None if iv["hi_open"] else iv["hi"])
    return m, b


def split_tag(x):
    return f"x{x:.0e}".replace("e-", "m")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True)
    ap.add_argument("--fx", action="store_true", help="the fitted-composition twins instead of the nominal planes")
    ap.add_argument("--root", default=os.path.join(REPO, "output", "analysis", "campaign"))
    a = ap.parse_args(argv)
    root = os.path.join(a.root, a.set)
    out = os.path.join(root, "summary" + ("_fx" if a.fx else ""))
    os.makedirs(out, exist_ok=True)
    sfx = "_fx" if a.fx else ""
    bounds, missing = {}, []

    def read(name):
        p = plane_json(os.path.join(root, name + sfx))
        if p is None:
            missing.append(name + sfx)
            return None
        with open(p) as f:
            return json.load(f)

    stems = [f"dp_{t}" for t in TEXTURES] + ["lmu_ltau"]
    for stem in stems:
        for sign in ("pos", "neg"):
            bal = read(f"{stem}_pot_bal_{sign}")
            if bal is None:
                continue
            bounds[f"{stem}_pot_bal_{sign}"] = bound_curve(bal)
            variants = []
            for x in SPLITS:
                v = read(f"{stem}_pot_{split_tag(x)}_{sign}")
                if v is not None:
                    variants.append((v, float(x)))
            prof = CS.combine(bal, variants)
            with open(os.path.join(out, f"{stem}_pot_profiled_{sign}.json"), "w") as f:
                json.dump(prof, f)
            bounds[f"{stem}_pot_profiled_{sign}"] = bound_curve(prof)
        mev = read(f"{stem}_mev")
        if mev is not None:
            bounds[f"{stem}_mev"] = bound_curve(mev)
    for model, key in MODEL_KEYS.items():
        if key in ("dp", "lmu_ltau"):
            continue
        for part in ("ul", "mev"):
            s = read(f"{key}_{part}")
            if s is not None:
                bounds[f"{key}_{part}"] = bound_curve(s)
    with open(os.path.join(out, "bounds.json"), "w") as f:
        json.dump({"set": a.set, "level": LEVEL, "fitted_compositions": a.fx, "bounds": bounds,
                   "missing": missing}, f, indent=1)
    plot(bounds, out, a.set + (" (fitted compositions)" if a.fx else ""))
    print(f"{len(bounds)} curves; {len(missing)} planes not finished; -> {out}")


def _draw(ax, curve, **kw):
    m, b = curve
    m = np.array(m, dtype=float)
    b = np.array([np.nan if v is None else v for v in b], dtype=float)
    ax.loglog(m, b, **kw)


def plot(bounds, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    stems = [f"dp_{t}" for t in TEXTURES] + ["lmu_ltau"]
    names = {f"dp_{t}": f"dark portal, {t}" for t in TEXTURES}
    names["lmu_ltau"] = r"$L_\mu - L_\tau$"
    # the potential
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, stem in zip(axes.flat, stems):
        for sign, ls in (("pos", "-"), ("neg", "--")):
            k = f"{stem}_pot_bal_{sign}"
            if k in bounds:
                _draw(ax, bounds[k], color="#D55E00", ls=ls, lw=1.2, label=f"screening negligible, {sign}")
            k = f"{stem}_pot_profiled_{sign}"
            if k in bounds:
                _draw(ax, bounds[k], color="#0072B2", ls=ls, lw=1.2, label=f"any coupling split, {sign}")
        ax.set_title(names[stem], fontsize=8, loc="left")
    for ax in axes[1]:
        ax.set_xlabel(r"$m_{Z_D}$ [eV]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"95% bound on $g\,\epsilon'\,\Sigma|U|^2$")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"The potential: {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "potential_bounds.png"), dpi=180)
    plt.close(fig)
    # the MeV absorption
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for stem in stems:
        k = f"{stem}_mev"
        if k in bounds:
            _draw(ax, bounds[k], lw=1.1, label=names[stem])
    ax.set_xlabel(r"$m_{Z_D}$ [eV]")
    ax.set_ylabel(r"95% bound on $g\,\Sigma|U|^2$ ($\epsilon' = 0$)")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(f"MeV absorption: {title}", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mev_bounds.png"), dpi=180)
    plt.close(fig)
    # the direct gauge models
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for model, key in MODEL_KEYS.items():
        if key in ("dp", "lmu_ltau"):
            continue
        color = None
        for part in ("ul", "mev"):
            k = f"{key}_{part}"
            if k in bounds:
                line, = ax.loglog([], [])
                color = color or line.get_color()
                _draw(ax, bounds[k], color=color, lw=1.1, label=model if part == "ul" else None)
    ax.set_xlabel(r"$m_{Z'}$ [eV]")
    ax.set_ylabel(r"95% bound on $g'$")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(f"Gauge models: {title}", fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gauge_bounds.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
