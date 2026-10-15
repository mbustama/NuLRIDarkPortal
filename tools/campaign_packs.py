"""
campaign_packs.py -- the plane campaign's job packs for slurm/campaign_array.sh (2026-10-14).

Reads analysis/configs/campaign/<set>/campaign.json (tools/make_campaign_configs.py), estimates each plane's cost and
packs the planes into packs of about --hours on --cores cores. One set of solver runs serves several planes through
the shared disk cache (NULRI_FLUX_CACHE; flux_basis 'pure_flavors'): the 2026-spectrum set's fixed-composition planes
compute them for their fitted twins and for both of the 2050 projection's (the same physics), the event set's
fixed-composition planes for their twins, and so the future detectors' set's (its own energy grid and sites). So the packs come in two waves, listed in that order: the producers (first)
and the planes that reuse their runs (cost: their fits only).

Costs in core-seconds per plane point (laptop, one core, 2026-10-14, after the speed-ups; Tycho to be calibrated):
the solver's runs at all index nodes (potential 8, MeV 18; event sky runs 9 and 20) and the fits (spectrum sets ~2;
event sets ~10 per fit, 1.5x with the fitted compositions' extra parameters).

The broken power law (2026-10-15, --tree campaign_bpl, the default): each point runs both pieces, the upper at
p_index's nodes and the lower at p_index_low's three, and the fits have one more parameter (BPL_SOLVER, BPL_FIT:
estimates until timed on Tycho).

  python tools/campaign_packs.py [--tree campaign_bpl] --sets now_ic26 now_events proj_2050 proj_future --hours 8 \
      --out analysis/configs/campaign_bpl/packs.txt
"""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOLVER = {"spectrum": {"pot": 8.0, "mev": 18.0}, "events": {"pot": 9.0, "mev": 20.0},
          # the future detectors (2026-10-15, NOT YET CALIBRATED): 630 energies instead of 515, five sites' Galactic
          # bands; their fits profile ~110 nuisances inside each chi^2
          "future": {"pot": 12.0, "mev": 26.0}}
FIT = {"spectrum": 2.0, "events": 10.0, "future": 30.0}
FX_FIT = 1.5
KIND = {"now_ic26": "spectrum", "proj_2050": "spectrum", "now_events": "events", "proj_future": "future"}
# the set whose fixed-composition planes compute the solver runs (proj_future: its own grid and sites)
PRODUCER_SETS = {"now_ic26": "now_ic26", "proj_2050": "now_ic26", "now_events": "now_events",
                 "proj_future": "proj_future"}


BPL_SOLVER = {"pot": (3 + 3) / 3, "mev": (7 + 3) / 7}
BPL_FIT = 1.3


def plane_cost(dataset, name, points, bpl=True):
    """(core-seconds, producer?) of one plane."""
    kind = KIND[dataset]
    planes = "mev" if "_mev" in name else "pot"
    fx = name.endswith("_fx")
    producer = PRODUCER_SETS[dataset] == dataset and not fx
    fit = FIT[kind] * (FX_FIT if fx else 1.0) * (BPL_FIT if bpl else 1.0)
    solver = SOLVER[kind][planes] * (BPL_SOLVER[planes] if bpl else 1.0) if producer else 0.0
    return (fit + solver) * points, producer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", default="campaign_bpl", choices=("campaign", "campaign_bpl"))
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    waves = {True: [], False: []}
    for ds in a.sets:
        d = os.path.join("analysis", "configs", a.tree, ds)
        with open(os.path.join(REPO, d, "campaign.json")) as f:
            lst = json.load(f)["planes"]
        for p in lst:
            c, prod = plane_cost(ds, p["name"], p["points"], bpl=a.tree == "campaign_bpl")
            waves[prod].append((c, os.path.join(d, p["name"] + ".json")))
    cap = a.hours * 3600.0 * a.cores
    packs = []                                    # [cost, [paths]], producers' packs first
    for prod in (True, False):
        items = sorted(waves[prod], reverse=True)
        mine = []
        for c, path in items:
            fit = [p for p in mine if p[0] + c <= cap]
            if fit:
                p = min(fit, key=lambda p: p[0])
            else:
                p = [0.0, []]
                mine.append(p)
            p[0] += c
            p[1].append(path)
        packs += mine
        print(f"{'producers' if prod else 'reusers'}: {len(items)} planes in {len(mine)} packs, "
              f"{sum(p[0] for p in mine) / 3600:.0f} core-hours")
    with open(os.path.join(REPO, a.out), "w") as f:
        for _, paths in packs:
            f.write(" ".join(paths) + "\n")
    tot = sum(p[0] for p in packs)
    print(f"{len(packs)} packs; {tot / 3600:.0f} core-hours; at 7 packs at a time ~{tot / 3600 / (7 * a.cores):.0f} h "
          f"wall; largest pack {max(p[0] for p in packs) / 3600 / a.cores:.1f} h")


if __name__ == "__main__":
    main()
