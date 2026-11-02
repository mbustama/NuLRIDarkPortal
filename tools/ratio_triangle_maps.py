"""
ratio_triangle_maps.py -- the dark portal's bound over its mixing-ratio triangle |U_e4|^2 : |U_mu4|^2 : |U_tau4|^2
(docs/dev/SCOPE_WIDENING_PLAN.md decision 3 and step 5).

For every ratio point on a grid of step 0.1 (66 points, real non-negative u) and four mediator masses, the 95% upper
bound on the coupling, read as the planes are read (per mass: Delta chi^2 = 3.84 above the column's minimum, which
sits at vanishing coupling), by a scan in decades from below and a bisection in log10 to 0.02 decades:
  1e-28 eV (the Galaxy's potential), 1e-20 eV (the Sun's), 1e-14 eV (the Earth's): P = g_D eps' Sum|U|^2 at the
  balanced split (screening negligible; the first curve of the potential planes), + sign;
  2e6 eV (the absorption): g_D Sum|U|^2 at eps' = 0.
Every chi^2 is profiled over the configuration's other free parameters (norm_astro, p_index, norm_gal). The
templates are the campaign's dp_emu_pot_bal_pos and dp_emu_mev configurations of the data set
(tools/make_campaign_configs.py), with the texture replaced.

  python tools/ratio_triangle_maps.py --dataset now_ic26 [--workers 16] [--masses 1e-20 2e6]

Writes output/analysis/campaign/<data set>/triangle_maps.json (resumable: points already there are skipped) and
the figure triangle_maps.png beside it.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

MASSES = (1e-28, 1e-20, 1e-14, 2e6)
THRESHOLD = 3.841
SUM_U2 = 0.5
STEP = 0.1
TOL_DECADES = 0.02


def ratio_points(step=STEP):
    """(w_e, w_mu, w_tau) on the triangle, each a multiple of step, summing to 1."""
    n = int(round(1.0 / step))
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def _template(dataset, mass):
    name = "dp_emu_mev" if mass >= 1e3 else "dp_emu_pot_bal_pos"
    with open(os.path.join(REPO, "analysis", "configs", "campaign", dataset, name + ".json")) as f:
        return json.load(f)


def _axis(cfg):
    return cfg["planes"][0]["y"]


class _Point:
    """chi^2 at fixed (m_Zd, coupling), profiled over the configuration's other parameters, for one texture."""

    def __init__(self, dataset, mass, w):
        from analysis import config as config_mod
        from analysis.fitting import FitSettings
        raw = _template(dataset, mass)
        u = np.sqrt(SUM_U2 * np.asarray(w, dtype=float))
        raw["solver"]["dark_portal"].update(u_e=float(u[0]), u_mu=float(u[1]), u_tau=float(u[2]))
        raw["planes"], raw["profiles"] = [], {}
        self.cfg = config_mod.from_dict(raw)
        self.obj = self.cfg.objective()
        self.axis = _axis(_template(dataset, mass))
        self.mass = mass
        self.lo, self.hi = (np.log10(b) for b in self.cfg.parameter(self.axis).bounds)
        self.settings = FitSettings(**{k: v for k, v in raw["fit"].items() if k in FitSettings.__dataclass_fields__})
        self.trace = {}
        self.start = None

    def chi2(self, log10c):
        from analysis.fitting import fit
        key = round(float(log10c), 6)
        if key not in self.trace:
            pset = self.cfg.parameter_set().holding({"m_Zd_eV": self.mass, self.axis: 10.0 ** log10c})
            r = fit(self.obj, pset, [self.start] if self.start else None, self.settings)
            self.start = {k: v for k, v in r.values.items() if k in pset.names}
            self.trace[key] = float(r.chi2)
        return self.trace[key]


def search(chi2, lo, hi):
    """The first upward crossing of Delta chi^2 = THRESHOLD in log10 of the coupling over [lo, hi]: chi2(log10 c),
    Delta from chi2(lo); a scan in decades, then a bisection to TOL_DECADES and a linear interpolation of Delta chi^2
    between the last two points. Returns (log10 bound or None, reference chi^2)."""
    ref = chi2(lo)
    x = prev = lo
    while x < hi:
        x = min(x + 1.0, hi)
        if chi2(x) - ref >= THRESHOLD:
            break
        prev = x
    else:
        return None, ref
    a, b = prev, x
    while b - a > TOL_DECADES:
        c = 0.5 * (a + b)
        if chi2(c) - ref >= THRESHOLD:
            b = c
        else:
            a = c
    fa, fb = chi2(a) - ref, chi2(b) - ref
    return (float(a + (THRESHOLD - fa) / (fb - fa) * (b - a)) if fb > fa else float(b)), ref


def bound(dataset, mass, w):
    """The 95% upper bound on the coupling (log10), the reference chi^2 and the evaluations."""
    p = _Point(dataset, mass, w)
    lb, ref = search(p.chi2, p.lo, p.hi)
    out = dict(log10_bound=lb, ref=ref, trace=p.trace)
    if lb is None:
        out["note"] = "no crossing within the scanned range"
    return out


def _task(args):
    dataset, mass, w = args
    r = bound(dataset, mass, w)
    return dict(mass=mass, w=list(w), **{k: v for k, v in r.items() if k != "trace"},
                n_evals=len(r["trace"]), trace={str(k): v for k, v in r["trace"].items()})


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    masses = sorted({r["mass"] for r in results})
    fig, axes = plt.subplots(1, len(masses), figsize=(3.0 * len(masses), 3.0))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, masses):
        rs = [r for r in results if r["mass"] == m and r["log10_bound"] is not None]
        if not rs:
            continue
        we = np.array([r["w"][0] for r in rs])
        wm = np.array([r["w"][1] for r in rs])
        xs = 0.5 * we + (1 - we - wm)                   # e at the top, mu at the left, tau at the right
        ys = we * np.sqrt(3) / 2
        b = np.array([r["log10_bound"] for r in rs])
        sc = ax.scatter(xs, ys, c=b, s=60, cmap="viridis", marker="h")
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="k", lw=0.6)
        ax.text(-0.04, -0.06, r"$\mu$", ha="center")
        ax.text(1.04, -0.06, r"$\tau$", ha="center")
        ax.text(0.5, np.sqrt(3) / 2 + 0.03, "$e$", ha="center")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"$m_{{Z_D}}$ = {m:.0e} eV", fontsize=8)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="log10 bound")
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--masses", nargs="+", type=float, default=list(MASSES))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    out_dir = os.path.join(REPO, "output", "analysis", "campaign", a.dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "triangle_maps.json")
    done = []
    if os.path.exists(path):
        with open(path) as f:
            done = json.load(f)["points"]
    have = {(r["mass"], tuple(r["w"])) for r in done}
    todo = [(a.dataset, m, w) for m in a.masses for w in ratio_points() if (m, tuple(w)) not in have]
    print(f"{len(todo)} points to compute ({len(have)} done)", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for r in pool.imap_unordered(_task, todo):
            done.append(r)
            with open(path + ".tmp", "w") as f:
                json.dump({"dataset": a.dataset, "threshold": THRESHOLD, "points": done}, f)
            os.replace(path + ".tmp", path)
            print(f"  m = {r['mass']:.0e}, w = {r['w']}: log10 bound {r['log10_bound']} ({r['n_evals']} fits)",
                  flush=True)
    plot(done, os.path.join(out_dir, "triangle_maps.png"))


if __name__ == "__main__":
    main()
