"""
campaign_packs.py -- the plane campaign's job packs for slurm/campaign_array.sh (2026-10-14).

Reads analysis/configs/campaign/<set>/campaign.json (tools/make_campaign_configs.py), estimates each plane's cost
(points x core-seconds per point, by data set and plane kind -- calibrated on Tycho's pilot), and packs the planes,
longest first, into packs of about --hours on --cores cores each. Writes one line per pack (its configuration paths)
to --out.

  python tools/campaign_packs.py --sets now_ic26 now_events proj_2050 --hours 8 --out analysis/configs/campaign/packs.txt
"""
import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# core-seconds per plane point (Tycho, 16 cores per job, 4 configurations at a time): calibrated by the pilot
# (2026-10-14, job 66400185); events cost more (sky runs, the event likelihoods), MeV planes run 7 spectral-index nodes
COST = {"now_ic26": 16.0, "proj_2050": 16.0, "now_events": 32.0}
MEV_FACTOR = 2.0
FX_FACTOR = 1.3


def plane_cost(dataset, name, points):
    c = COST[dataset] * points
    if "_mev" in name:
        c *= MEV_FACTOR
    if name.endswith("_fx"):
        c *= FX_FACTOR
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = []
    for ds in a.sets:
        d = os.path.join("analysis", "configs", "campaign", ds)
        with open(os.path.join(REPO, d, "campaign.json")) as f:
            lst = json.load(f)["planes"]
        items += [(plane_cost(ds, p["name"], p["points"]), os.path.join(d, p["name"] + ".json")) for p in lst]
    items.sort(reverse=True)
    cap = a.hours * 3600.0 * a.cores
    packs = []                                    # [cost, [paths]]
    for c, path in items:
        fit = [p for p in packs if p[0] + c <= cap]
        if fit:
            p = min(fit, key=lambda p: p[0])
        else:
            p = [0.0, []]
            packs.append(p)
        p[0] += c
        p[1].append(path)
    with open(os.path.join(REPO, a.out), "w") as f:
        for _, paths in packs:
            f.write(" ".join(paths) + "\n")
    tot = sum(p[0] for p in packs)
    print(f"{len(items)} planes in {len(packs)} packs; {tot / 3600:.0f} core-hours; at 7 packs at a time "
          f"~{tot / 3600 / (7 * a.cores):.0f} h wall; largest pack {max(p[0] for p in packs) / 3600 / a.cores:.1f} h")


if __name__ == "__main__":
    main()
