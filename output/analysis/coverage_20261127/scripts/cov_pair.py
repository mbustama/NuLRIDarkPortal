"""One to one: the rerun's pseudo-experiments (multi-start) against the first run's (single start), same seed and draw."""
import json, sys, glob, re
import numpy as np
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import coverage_limits as cl
new_dir, old_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]      # tag: le_ltau or si
n = dict(same=0, worse=0, better=0, flips_to_covered=0, flips_to_excluded=0, tot=0)
for p in sorted(glob.glob(f"{new_dir}/cov2_{tag}_t*.json")):
    o = json.load(open(p))
    if "finished" not in o:
        continue
    o = cl.reread(p); sd = int(re.search(r"_t(\d+)_", p).group(1))
    old = cl.reread(f"{old_dir}/cov_{tag}_t{sd}.json")
    ob = {r["toy"]: r for r in old["results"]}
    for r in o["results"]:
        q = ob[r["toy"]]
        cn = min(r["chi2"].values()); co = min(q["chi2"].values())
        common = [g for g in r["chi2"] if g in q["chi2"]]
        dmax = max(q["chi2"][g] - r["chi2"][g] for g in common); dmin = min(q["chi2"][g] - r["chi2"][g] for g in common)
        n["tot"] += 1
        n["flips_to_covered"] += (r["covered"] and not q["covered"]); n["flips_to_excluded"] += (q["covered"] and not r["covered"])
        print(f"seed {sd} draw {r['toy']}: covered {q['covered']} -> {r['covered']}; dchi2 at truth {q.get('dchi2_true', float('nan')):.2f} -> "
              f"{r['dchi2_true']:.2f}; column min {co:.3f} -> {cn:.3f}; old minus new chi2 over {len(common)} couplings: {dmin:+.3f} .. {dmax:+.3f}")
print(n)
