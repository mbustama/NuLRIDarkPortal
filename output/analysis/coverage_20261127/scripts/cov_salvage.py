"""A coverage task stopped by its time limit: its pseudo-experiments whose column is complete (every coupling of the
real column and the truth), from the file's "partial", written as a finished file <name>_salvaged.json for
tools/coverage_limits.reread (2026-11-27; the task's own results format, nothing refitted)."""
import json, sys
p = sys.argv[1]
o = json.load(open(p))
assert "finished" not in o and "partial" in o, "not a stopped task with partial results"
P = o["partial"]; need = set(o["real"])
res = []
for k, col in P["chi2"].items():
    if set(col) >= need and k in P["chi2_true"]:
        res.append(dict(toy=int(k), f_obs=P["f_obs"][int(k)], segments=P["segments"][int(k)],
                        chi2=dict(sorted(col.items(), key=lambda kv: float(kv[0]))), chi2_true=P["chi2_true"][k], values={}))
o.pop("partial"); o["results"] = res; o["finished"] = "salvaged from the partial of a task stopped by its time limit"
o["only_salvaged"] = [r["toy"] for r in res]
out = p.replace(".json", "_salvaged.json"); json.dump(o, open(out, "w"), indent=1)
print(out, "pseudo-experiments:", o["only_salvaged"])
