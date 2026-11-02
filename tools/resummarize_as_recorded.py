"""
resummarize_as_recorded.py -- re-summarize a plane under the identity its checkpoint records carry (2026-10-22).
run.summarize keeps only the records whose identity matches the current configuration, and the identity includes the
solver's source digest: a plane whose run outlived a code change (the potential planes, launched at 9088b28 and
summarized after the 11:06 pull of 9d31072) summarized no record at all and came out all NaN, although every fit was
done. Here the plane's own identity -- the one shared by nearly all its records (at least SHARE of them) -- stands in
for the current one; the summary is written as run.summarize writes it. Only for planes whose physics the code
change did not touch (the potential planes' dark-portal phase below 1 keV; at 1-10 keV, where dp_si runs, the
thermal-grid recheck and its refits correct them: analysis/exclusion.corrected).
  python tools/resummarize_as_recorded.py <config> [<config> ...]
"""
import collections
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
SHARE = 0.95


def count(path):
    s = json.load(open(path))
    return sum(v is not None for row in s["chi2"] for v in row)


def main(argv):
    from analysis import scans
    from analysis.config import from_dict
    from analysis.run import asimov_only, summarize
    original = scans.scan_identity
    for cfgp in argv:
        raw = json.load(open(cfgp))
        cfg = from_dict(raw, cfgp)
        kind = "asimov" if asimov_only(cfg) else "data"
        out_dir = cfg.output_path(kind)
        centre = None
        if kind == "asimov":
            runs = json.load(open(os.path.join(out_dir, "manifest.json")))["runs"]
            centre = tuple(next(r["data_centre"] for r in reversed(runs) if r.get("data_centre")))
        plane = [f for f in os.listdir(out_dir) if f.startswith("plane_") and f.endswith(".json")][0]
        path = os.path.join(out_dir, plane)
        ids = collections.Counter()
        for line in open(path[:-5] + ".jsonl"):
            try:
                ids[json.loads(line).get("id")] += 1
            except json.JSONDecodeError:
                continue
        ident, n = ids.most_common(1)[0]
        if n < SHARE * sum(ids.values()):
            print(f"{cfgp}: no identity shared by {SHARE:.0%} of the records ({dict(ids)}): left as it is", flush=True)
            continue
        before = count(path)
        scans.scan_identity = lambda *a, **k: ident
        try:
            summarize(cfg, out_dir, centre, log=lambda *a: None)
        finally:
            scans.scan_identity = original
        print(f"{cfgp}: identity {ident} ({n} of {sum(ids.values())} records), points {before} -> {count(path)}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
