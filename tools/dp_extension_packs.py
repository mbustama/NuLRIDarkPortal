"""
dp_extension_packs.py -- the job-array lists for the dark-portal extension and the grid extension (2026-10-21), from
the configurations of tools/make_dp_extension_configs.py and tools/refine_grid.py, balanced by the pilot's fit times
(tools/pilot_fits.py; seconds per fit on one core):
  analysis/configs/campaign_ml/packs_dp_ul.txt    one eps' = 0 ultralight plane per task (slurm/campaign_array.sh, SLOTS=1)
  analysis/configs/campaign_ml/packs_dp_pot.txt   the potential planes, packed into --pot-tasks tasks (SLOTS=4)
  analysis/configs/campaign_ml/commands_refine_grid.txt   one refine_grid.py run per task (slurm/command_array.sh)
  python tools/dp_extension_packs.py --pot-tasks 14 --grid-tasks 30 --manifest <refine_grid manifest>
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(REPO, "analysis", "configs", "campaign_ml")
# seconds per fit (pilot, 2026-10-21; from the configurations' starts: a scan's fits start from neighbours and run
# faster, and its repair passes add some)
POT_SECONDS = {"now_ic26": 40, "now_events": 120, "proj_future": 35}


def points(cfg):
    pl = json.load(open(cfg))["planes"][0]
    return pl["nx"] * pl["ny"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pot-tasks", type=int, default=14)
    ap.add_argument("--grid-tasks", type=int, default=30)
    ap.add_argument("--manifest", default="scratch/refine_grid_manifest.json")
    a = ap.parse_args(argv)
    rel = lambda p: os.path.relpath(p, REPO)
    ul = sorted(glob.glob(os.path.join(D, "*", "dp_*_ul*.json")))
    with open(os.path.join(D, "packs_dp_ul.txt"), "w") as f:
        f.write("".join(rel(c) + "\n" for c in ul))
    pot = sorted(glob.glob(os.path.join(D, "*", "dp_*_pot_*.json")),
                 key=lambda c: -points(c) * POT_SECONDS[os.path.basename(os.path.dirname(c))])
    bins = [[0.0, []] for _ in range(a.pot_tasks)]
    for c in pot:                                       # greedy: the next plane into the lightest bin
        cost = points(c) * POT_SECONDS[os.path.basename(os.path.dirname(c))]
        b = min(bins, key=lambda b: b[0])
        b[0] += cost
        b[1].append(rel(c))
    with open(os.path.join(D, "packs_dp_pot.txt"), "w") as f:
        f.write("".join(" ".join(b[1]) + "\n" for b in bins))
    with open(os.path.join(D, "commands_refine_grid.txt"), "w") as f:
        f.write("".join(f"python tools/refine_grid.py run {a.manifest} {{task}} --workers 32\n"
                        for _ in range(a.grid_tasks)))
    hours = [b[0] / 3600 for b in bins]
    print(f"{len(ul)} ultralight eps'=0 planes (one per task); {len(pot)} potential planes in {a.pot_tasks} tasks, "
          f"{min(hours):.0f}-{max(hours):.0f} core-hours each; {a.grid_tasks} refine_grid tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
