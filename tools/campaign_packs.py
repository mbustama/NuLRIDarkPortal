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
p_index's nodes and the lower at p_index_low's three, and the fits have one more parameter (BPL_COST: timed on the
laptop where marked, estimated otherwise).

  python tools/campaign_packs.py [--tree campaign_bpl] --sets now_ic26 now_events proj_2050 proj_future --hours 8 \
      --out analysis/configs/campaign_bpl/packs.txt
"""
import argparse
import json
import os
import sys

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


# the broken power law's campaign (2026-10-15): core-seconds per point, laptop, one core, steady state, after the day's
# speed-ups (spline weights, Newton and Gauss-Newton profiles): producers run both pieces' solver runs and the fit,
# reusers (the fitted-composition twins, the 2050 projection) the fit only. Measured: 2026-spectrum set potential 6.3,
# MeV 37; future detectors potential 17-19 when chained as the scan runs them (each fit starting from the previous
# point's; 29-55 from the configured starts); HESE + DR2 potential 21-38 chained. Estimated: the rest.
BPL_COST = {"spectrum": {"pot": (6.3, 2.5), "mev": (37.0, 3.0)},
            "events": {"pot": (25.0, 12.0), "mev": (50.0, 14.0)},
            "future": {"pot": (18.0, 10.0), "mev": (40.0, 12.0)}}


# the campaign with the mixing fixed and m_lightest profiled exactly (2026-10-18, --tree campaign_ml): Tycho core-seconds
# per point and per global fit from the pilot (job 66464601: two mass columns of 16 planes, every data set and kind,
# 1,528 points, no flux reuse between planes), per data set and plane kind (tools/exact_progress.kind_of); the MeV
# planes' from the re-pilot with the relics thermal in every collision term (job 66505900, 2026-10-18: 16 MeV planes x
# two mass columns, 353 points in 55 min)
ML_COST = {("now_events", "bal"): (27.2, 90), ("now_events", "mev"): (205.8, 286), ("now_events", "split"): (22.5, 137),
           ("now_events", "ul"): (29.7, 90), ("now_ic26", "bal"): (2.8, 90), ("now_ic26", "mev"): (57.3, 286),
           ("now_ic26", "split"): (4.7, 50), ("now_ic26", "ul"): (3.4, 90), ("proj_2050", "bal"): (3.0, 90),
           ("proj_2050", "mev"): (29.0, 90), ("proj_2050", "split"): (10.7, 79), ("proj_2050", "ul"): (3.0, 90),
           ("proj_future", "bal"): (13.6, 90), ("proj_future", "mev"): (171.6, 286),
           ("proj_future", "split"): (26.9, 95), ("proj_future", "ul"): (10.8, 90)}


def plane_cost_ml(dataset, name, points):
    """(core-seconds, producer?) of one campaign_ml plane: the pilot's cost, every plane priced as a producer."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    from exact_progress import kind_of
    per_point, fit = ML_COST[(dataset, kind_of(name))]
    return per_point * points + fit, PRODUCER_SETS[dataset] == dataset and not name.endswith("_fx")


def plane_cost(dataset, name, points, bpl=True):
    """(core-seconds, producer?) of one plane."""
    kind = KIND[dataset]
    planes = "mev" if "_mev" in name else "pot"
    fx = name.endswith("_fx")
    producer = PRODUCER_SETS[dataset] == dataset and not fx
    if bpl:
        prod, fit = BPL_COST[kind][planes]
        per_point = prod if producer else fit * (FX_FIT if fx else 1.0)
    else:
        fit = FIT[kind] * (FX_FIT if fx else 1.0)
        per_point = fit + (SOLVER[kind][planes] if producer else 0.0)
    return per_point * points, producer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", default="campaign_bpl", choices=("campaign", "campaign_bpl", "campaign_ml"))
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
            if a.tree == "campaign_ml":
                c, prod = plane_cost_ml(ds, p["name"], p["points"])
            else:
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
    print(f"{len(packs)} packs; {tot / 3600:.0f} core-hours; at {176 // a.cores} packs at a time ~{tot / 3600 / ((176 // a.cores) * a.cores):.0f} h "
          f"wall; largest pack {max(p[0] for p in packs) / 3600 / a.cores:.1f} h")


if __name__ == "__main__":
    main()
