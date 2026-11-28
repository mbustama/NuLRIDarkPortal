"""fc62_keeper.py -- keeps the forecast refit (62 lightest masses, 2026-11-27) going without anyone watching.

Every call (the keeper job runs it every 10 minutes), for each of the 84 blocks (packs_fc62_si.txt in nulri_forecast,
packs_fc62_ml.txt in nulri_set1_ml):
  - finished (its summary asimov/plane_*.json exists): nothing;
  - a task of it is running or waiting (the arrays fc62-si / fc62-ml or an earlier resubmission fc62r-si / fc62r-ml,
    by array index): nothing, except that a task held by Slurm after a failed start is released with node664 excluded;
  - otherwise (its task ended without a summary: time limit, memory, node failure): its dead lock is removed and the
    block is resubmitted as a one-task array with the same index and config, with a fresh worker process per chain
    (NULRI_WORKER_MAXTASKS=1 in fc62_array.sh; the workers' memory creeps up otherwise; 80 GB for the secret-interaction
    blocks, 110 GB for the others), at most MAX_TRIES times, and only while the
    account's running cores stay at or below the cap. A resubmitted run reuses every recorded point.
Dry run: python3 fc62_keeper.py --dry"""
import glob
import json
import os
import subprocess
import sys
import time

X = "/lustre/astro/mbustama/external"
CAP, CORES, MAX_TRIES = 1500, 14, 5
KINDS = {"si": dict(dir=f"{X}/nulri_forecast", packs="packs_fc62_si.txt", mem="80G"),
         "ml": dict(dir=f"{X}/nulri_set1_ml", packs="packs_fc62_ml.txt", mem="110G")}
STATE = f"{X}/fc62_keeper_state.json"
DRY = "--dry" in sys.argv
NORESUB = "--no-resubmit" in sys.argv     # 2026-11-28 (his decision): the re-check is not resumed; only clean up


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    used = sum(int(v) for v in sh("squeue", "-h", "-u", "mbustama", "-t", "R", "-o", "%C").split())
    # the final checks (fin-col, fin-idx; 2026-11-27) go first: while any of their tasks waits, no block is resubmitted
    fin_waiting = len(sh("squeue", "-h", "-r", "-u", "mbustama", "-t", "PD", "-n", "fin-col,fin-idx", "-o", "%i").split())
    out = []
    for kind, K in KINDS.items():
        cfgs = [l.strip() for l in open(os.path.join(K["dir"], K["packs"])) if l.strip()]
        live = {}
        for l in sh("squeue", "-h", "-r", "-u", "mbustama", "-n", f"fc62-{kind},fc62r-{kind}", "-o", "%K|%i|%T|%r").split("\n"):
            if l.strip():
                k, jid, st, why = l.split("|")
                live[int(k)] = (jid, st, why)
        done = todo = 0
        for i, cfg in enumerate(cfgs):
            d = os.path.join(K["dir"], json.load(open(os.path.join(K["dir"], cfg)))["output_dir"], "asimov")
            if [f for f in glob.glob(os.path.join(d, "plane_*.json"))]:
                done += 1
                continue
            if i in live:
                jid, st, why = live[i]
                if st == "PENDING" and "held" in why:
                    out.append(f"release {kind} {i} ({why[:30]})")
                    if not DRY:
                        sh("scontrol", "update", f"JobId={jid}", "ExcNodeList=node664")
                        sh("scontrol", "release", jid)
                continue
            key = f"{kind}:{i}"
            n = state.get(key, 0)
            if n >= MAX_TRIES:
                out.append(f"GIVEN UP {kind} {i} after {n} resubmissions")
                continue
            if NORESUB or used + CORES > CAP or fin_waiting:
                todo += 1
                continue
            out.append(f"resubmit {kind} {i} (try {n + 1}) {os.path.basename(cfg)}")
            if not DRY:
                lock = os.path.join(d, "RUNNING.lock")
                if os.path.exists(lock):
                    os.remove(lock)
                r = subprocess.run(["sbatch", "--parsable", "-J", f"fc62r-{kind}", f"--array={i}", f"--mem={K['mem']}",
                                    "--exclude=node664", f"--export=ALL,PACKS={K['packs']}", "fc62_array.sh"],
                                   capture_output=True, text=True, cwd=K["dir"])
                out[-1] += " -> " + (r.stdout.strip() or r.stderr.strip()[:80])
                if r.returncode == 0:
                    state[key] = n + 1
                    used += CORES
        out.insert(0, f"{kind}: {done} of {len(cfgs)} finished, {len(live)} live" + (f", {todo} wait for cores" + (" or for the final checks" if fin_waiting else "") if todo else ""))
    # the solver caches of the refit's tasks that ended (killed tasks leave theirs behind)
    gone = 0
    for l in sh("sacct", "-X", "-n", "-S", "2026-11-27T15:00", "--name=fc62-si,fc62-ml,fc62r-si,fc62r-ml,fc62-test",
                "-o", "JobIDRaw,State").splitlines():
        f = l.split()
        if len(f) >= 2 and f[0].isdigit() and f[1] not in ("RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED"):
            for d in glob.glob(f"/lustre/astro/mbustama/flux_cache/job_{f[0]}_*"):
                gone += 1
                if not DRY:
                    subprocess.run(["rm", "-rf", d])
    if gone:
        out.append(f"{gone} caches of ended tasks removed")
    if not DRY:
        json.dump(state, open(STATE, "w"), indent=1)
    print(time.strftime("%d %H:%M"), f"cores {used};", "; ".join(out), flush=True)


if __name__ == "__main__":
    main()
