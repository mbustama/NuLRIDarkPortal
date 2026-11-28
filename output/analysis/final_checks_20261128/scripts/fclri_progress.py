"""Are the old ultralight forecast tasks (fc-lri, nulri_set1_sm) progressing? Read-only: for each running task, its
block's points, repair checks and the time of its last saved record."""
import collections, glob, json, os, re, subprocess, time
os.chdir("/lustre/astro/mbustama/external/nulri_set1_sm")
q = subprocess.run(["squeue", "-u", "mbustama", "-n", "fc-lri", "-h", "-o", "%i %M"], capture_output=True, text=True).stdout.split("\n")
for line in q:
    if not line.strip(): continue
    t, el = line.split()
    logs = glob.glob(f"logs/*{t}.out")
    if not logs: print(t, "no log"); continue
    txt = open(logs[0]).read()
    cfg = re.search(r"^=== (\S+)", txt, re.M).group(1)
    out = json.load(open(cfg))["output_dir"]
    fs = glob.glob(out + "/asimov/plane_*.jsonl") + glob.glob(out + "/data/plane_*.jsonl")
    if not fs: print(t, cfg, "no records"); continue
    recs = []
    for l in open(fs[0]):
        try: recs.append(json.loads(l))
        except Exception: pass
    idx = {json.dumps(r.get("index")) for r in recs if "chi2" in r}
    rep = collections.Counter(r.get("repair_checked") for r in recs if "repair_checked" in r)
    ts = sorted(r["t"] for r in recs if "t" in r)
    fmt = lambda x: time.strftime("%d %H:%M", time.localtime(x))
    n24 = sum(1 for x in ts if x > time.time() - 86400); n3 = sum(1 for x in ts if x > time.time() - 3 * 3600)
    pl = json.load(open(cfg))["planes"][0]
    print(f"{os.path.basename(cfg)[:-5]}: running {el}; {len(idx)} of {pl['nx'] * pl['ny']} points; repair checks per pass {dict(sorted(rep.items()))}; "
          f"records in the last 24 h: {n24}, last 3 h: {n3}; last record {fmt(ts[-1]) if ts else '?'}; log changed {fmt(os.path.getmtime(logs[0]))}")
