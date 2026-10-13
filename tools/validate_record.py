"""
validate_record.py

The four-point validation record at production settings (the solver's defaults): for each
benchmark point, the energy and the number of neutrinos reaching the detector relative to the
same run without dark-portal interactions (absolute checks: the energy can never exceed 1 beyond
the bin-centre placement of secondaries), the energy-averaged flavor fractions, and the fluxes
themselves. Run on two machines and compare to check a new installation (e.g. Tycho against the
laptop).

  python tools/validate_record.py --out runs/record_laptop.json [--cores 8]
  python tools/validate_record.py --out runs/record_tycho.json --compare runs/record_laptop.json

Writes OUT (JSON: provenance, per-point numbers) and OUT with .npz (the fluxes). With --compare,
prints the largest relative flux difference and fraction difference per point against the
reference, and exits non-zero if any exceeds the cross-machine tolerances below.
"""
import argparse
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

import dark_portal_solver as dps  # noqa: E402
import dp_core  # noqa: E402
from dp_core import compute_dE_geometric  # noqa: E402
from flux_postprocessing import compute_energy_averaged_ratios  # noqa: E402

POINTS = {
    "below": dict(m_Zd_eV=3e5),
    "inband": dict(m_Zd_eV=2e6),
    "strong": dict(m_Zd_eV=2e7, g_D=1e-2),
    "cosmoX2": dict(m_Zd_eV=40.0),
}
# Cross-machine tolerances: numba compiles for each CPU and pip's numpy uses OpenBLAS where the
# laptop's uses MKL, so last-digit differences are expected; 'strong' carries a ~7.5e-8
# common-mode floor under any reordering (notes, Sec. Validation).
TOL_FLUX = 1e-6
TOL_FRACTION = 1e-8


def _run(dp, cores):
    cfg = {"dark_portal": dict(dp), "execution": {"n_cores": int(cores), "verbose": 0}}
    dp_core.HAS_TQDM = False
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return dps.run_solver(cfg)
    finally:
        sys.stdout = old


def _git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def record(cores: int) -> tuple:
    import numba
    import scipy
    t0 = time.perf_counter()
    E, F0 = _run(dict(u_e=0.0, u_mu=0.0), cores)
    dE = compute_dE_geometric(E)
    e0, n0 = np.sum(E * F0.sum(axis=1) * dE), np.sum(F0.sum(axis=1) * dE)
    out = dict(provenance=dict(host=socket.gethostname(), commit=_git("rev-parse", "HEAD"),
                               dirty=bool(_git("status", "--porcelain", "--", "src")), cores=int(cores),
                               python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                               numba=numba.__version__, svml=bool(numba.core.config.USING_SVML),
                               grid={k: dps.DEFAULT_CONFIG["grid"][k] for k in
                                     ("n_E_bins", "n_z_bins", "n_sun_paths", "transit_dtheta", "n_earth_chords", "N_steps_earth",
                                      "earth_sky", "earth_zenith_average")}),
               points={})
    fluxes = dict(E=E, no_interaction=F0)
    for name, dp in POINTS.items():
        _, F = _run(dp, cores)
        fluxes[name] = F
        out["points"][name] = dict(
            parameters=dp,
            energy_over_no_interaction=float(np.sum(E * F.sum(axis=1) * dE) / e0),
            number_over_no_interaction=float(np.sum(F.sum(axis=1) * dE) / n0),
            fractions=[float(x) for x in compute_energy_averaged_ratios(E, F)])
        p = out["points"][name]
        print(f"{name:8s} energy {p['energy_over_no_interaction']:.6f}  number {p['number_over_no_interaction']:.6f}  "
              f"fractions {np.round(p['fractions'], 6)}", flush=True)
    out["seconds"] = time.perf_counter() - t0
    return out, fluxes


def compare(rec, fluxes, ref_path) -> bool:
    ref = json.load(open(ref_path))
    ref_fl = np.load(os.path.splitext(ref_path)[0] + ".npz")
    ok = True
    print(f"\nagainst {ref_path} ({ref['provenance']['host']}, {ref['provenance']['commit'][:10]}):")
    print(f"{'point':8s} {'max rel flux diff':>18s} {'max fraction diff':>18s}")
    for name in POINTS:
        F, G = fluxes[name], ref_fl[name]
        scale = np.maximum(np.abs(G), 1e-300)
        dflux = float(np.max(np.abs(F - G) / scale))
        dfrac = float(np.max(np.abs(np.array(rec["points"][name]["fractions"]) - np.array(ref["points"][name]["fractions"]))))
        good = dflux <= TOL_FLUX and dfrac <= TOL_FRACTION
        ok &= good
        print(f"{name:8s} {dflux:18.2e} {dfrac:18.2e}   {'ok' if good else 'DIFFERS'}")
    print("verdict:", "agree within the cross-machine tolerances" if ok else "DIFFERENT: investigate before using these runs")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output JSON (fluxes go next to it as .npz)")
    ap.add_argument("--cores", type=int, default=dp_core.default_n_cores())
    ap.add_argument("--compare", default=None, help="reference record JSON to compare with")
    args = ap.parse_args(argv)
    rec, fluxes = record(args.cores)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", **fluxes)
    print(f"written {args.out} ({rec['seconds']:.0f} s)")
    if args.compare:
        return 0 if compare(rec, fluxes, args.compare) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
