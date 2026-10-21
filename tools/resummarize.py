"""
resummarize.py -- re-summarize a plane's output directory (run.summarize) with the Asimov data's centre taken from
its manifest.json (the last run's data_centre) rather than recomputed: `python -m analysis <config> --summarize`
recomputes it with a solver run, which on Tycho's nodes can take longer than a short job (2026-10-21: 30 min without
finishing). The checkpoint records carry the identity under that centre, so the stored value reproduces it exactly;
the tool stops if the new summary holds fewer points than the old one.
  python tools/resummarize.py <config> [<config> ...]
"""
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]


def count(path):
    s = json.load(open(path))
    return sum(v is not None for row in s["chi2"] for v in row)


def main(argv):
    from analysis.config import from_dict
    from analysis.run import asimov_only, summarize
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
        before = count(path)
        shutil.copy2(path, path + ".bak")                   # restored if the new summary loses points
        summarize(cfg, out_dir, centre)
        after = count(path)
        print(f"{cfgp}: centre {centre}, points {before} -> {after}", flush=True)
        if after < before:
            shutil.copy2(path + ".bak", path)
            raise SystemExit(f"{cfgp}: fewer points than before (identity mismatch?); the old summary is restored")
        os.remove(path + ".bak")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
