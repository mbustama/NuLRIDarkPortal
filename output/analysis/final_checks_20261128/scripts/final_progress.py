"""Progress of the last runs, judged by saved records (run on Tycho, read-only)."""
import glob, json, os, subprocess, time
X = "/lustre/astro/mbustama/external"
print(time.strftime("%d %H:%M"))
q = subprocess.run(["squeue", "-u", "mbustama", "-h", "-o", "%j %T"], capture_output=True, text=True).stdout.split("\n")
c = {}
for l in q:
    if l.strip(): c[l] = c.get(l, 0) + 1
print("; ".join(f"{v} {k}" for k, v in sorted(c.items())))
cores = sum(int(x) for x in subprocess.run(["squeue", "-u", "mbustama", "-h", "-t", "R", "-o", "%C"], capture_output=True, text=True).stdout.split())
print("cores running:", cores)
for name, pat in (("index ranges, SI", X + "/nulri_refit_set/output/analysis/final_idx/*/b*/data"), ("index ranges, L_e-L_tau", X + "/nulri_set1_ml/output/analysis/final_idx/*/b*/data")):
    n = last = 0; blocks = done = rep = 0
    for d in glob.glob(pat):
        blocks += 1; done += bool(glob.glob(d + "/plane_*.json"))
        for f in glob.glob(d + "/plane_*.jsonl"):
            for l in open(f):
                try: r = json.loads(l)
                except Exception: continue
                if "chi2" in r and "held" in r: n += 1
                if "repair_checked" in r: rep += 1
                last = max(last, r.get("t", 0))
    print(f"{name}: {blocks} blocks, {done} finished, {n} points saved, {rep} repair checks, last record {time.strftime('%H:%M', time.localtime(last)) if last else 'none'}")
for d in ("nulri_refit_set", "nulri_set1_ml", "nulri_sep"):
    print(d, "single columns done:", sorted(os.path.basename(f) for f in glob.glob(f"{X}/{d}/output/final_cols/*.json")))
caches = glob.glob("/lustre/astro/mbustama/flux_cache/job_*")
print("solver caches present:", len(caches))
errs = [f for pat in (X + "/nulri_refit_set/logs/slurm_fin-*_9[01]*.err", X + "/nulri_set1_ml/logs/slurm_fin-*_9[01]*.err", X + "/nulri_sep/logs/slurm_fin-col_909864_*.err") for f in glob.glob(pat) if "Traceback" in open(f).read()]
print("tracebacks:", errs)
