"""
make_set_configs.py -- the audit's two present-data sets (docs/dev/AUDIT_2026-11-08_resolution.md, decisions of
2026-11-09) as configurations made from a campaign's existing block configurations: the data section replaced, the
solver section untouched, so every flux run the old blocks stored is read back from the cache (the cache key is the
solver configuration and code; analysis/model.py _disk_path).

  now_icecube   set 1, IceCube's published results: the released MESE likelihood grid (A2), its composition weighted
                by MESE's expected events (A3) over 5 TeV-10 PeV (A1), and the 2025 combined fit's spectrum segments,
                compared with the extragalactic plus Galactic flux (A5). Made from now_ic26 blocks.
  proj_future_planA  the forecasts on Plan A's detector models (C1, 2026-11-11): the eight telescopes ('All8'), the
                antineutrino share free as set 2's (E1). Made from proj_future blocks.
  now_mese_dr2  set 2, our event fit: MESE cascades and double cascades (energy only) and DR2 tracks, the
                antineutrino share free, nubar:nu from 1:2 to 1:1 (E1). Made from now_events blocks.

Usage: python tools/make_set_configs.py SRC_DIR DST_DIR {now_icecube,now_mese_dr2}
  e.g. python tools/make_set_configs.py analysis/configs/campaign_si_b/now_ic26 analysis/configs/campaign_si_b/now_icecube now_icecube
The output_dir of each block has the source directory's name replaced by the set's (output/.../now_ic26/... ->
output/.../now_icecube/...).
"""
import glob
import json
import os
import sys

SETS = {
    "now_icecube": dict(needs="spectrum",
                        data={"surface": "icecube_mese_2025_grid", "spectrum": "icecube_cf_2025",
                              "flavor_band_eV": [5e12, 1e16], "flavor_weighting": "mese"},
                        parameters={}),
    "proj_future_planA": dict(needs="events",
                              data={"events": {"future": {"config": "All8"}}},
                              parameters={"nubar_to_nu_ratio": {"start": 1.0, "bounds": [0.5, 1.0], "space": "linear"}}),
    "now_mese_dr2": dict(needs="events",
                         data={"events": {"mese": {}, "dr2": {}}},
                         parameters={"nubar_to_nu_ratio": {"start": 1.0, "bounds": [0.5, 1.0], "space": "linear"}}),
}


def convert(src_dir, dst_dir, which):
    spec = SETS[which]
    src_name = os.path.basename(os.path.normpath(src_dir))
    files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not files:
        raise SystemExit(f"no configurations in {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        c = json.load(open(f))
        if "output_dir" not in c:                      # a campaign file, not a block
            print(f"skipped {f} (no output_dir)")
            continue
        if spec["needs"] not in c.get("data", {}):
            raise SystemExit(f"{f}: {which} is made from blocks with data.{spec['needs']}")
        data = json.loads(json.dumps(spec["data"]))
        if which in ("now_mese_dr2", "proj_future_planA") and "flavor_band_eV" in c["data"]:
            data["flavor_band_eV"] = c["data"]["flavor_band_eV"]
        c["data"] = data
        c["parameters"] = dict(c["parameters"], **spec["parameters"])
        seg = f"/{src_name}/"
        if seg not in c["output_dir"]:
            raise SystemExit(f"{f}: output_dir {c['output_dir']} does not contain {seg}")
        c["output_dir"] = c["output_dir"].replace(seg, f"/{which}/")
        c["_comment"] = (c.get("_comment", "") + f" Data set {which} (audit decisions 2026-11-09; tools/make_set_configs.py"
                         f" from {src_name}).").strip()
        json.dump(c, open(os.path.join(dst_dir, os.path.basename(f)), "w"), indent=1)
    print(f"{len(files)} configurations -> {dst_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in SETS:
        raise SystemExit(__doc__)
    convert(*sys.argv[1:4])
