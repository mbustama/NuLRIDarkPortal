#!/bin/bash
#SBATCH --job-name=nulri-sweep
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --partition=astro3_long
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # up to 32 per job, ask before more
#SBATCH --mem=48G
# The campaign's clean-up (2026-10-17): every configuration under TREE whose plane is not finished -- no plane_*.json
# in its output's data/ or asimov/: packs that hit the time limit, failures -- run again, resuming from its
# checkpoints, SLOTS at a time with cpus / SLOTS workers each. The list is made when the task starts; several array
# tasks share it through the run locks (analysis/run.py: a plane held by a live run is skipped, a stale lock taken
# over), each starting at its own offset. Logs in logs/sweep_<array job>_<task>/<set>_<name>.out.
#   TREE=analysis/configs/campaign_bpl NULRI_FLUX_CACHE=... NULRI_HESE_MC=... NULRI_HESE_DETSYS=... SLOTS=4 \
#   sbatch --array=0-6%7 --dependency=afterany:<the campaign's array> slurm/campaign_sweep.sh
source slurm/common.sh
SLOTS=${SLOTS:-4}
W=$(( ${SLURM_CPUS_PER_TASK:-1} / SLOTS ))
[ "$W" -lt 1 ] && W=1
export NUMBA_NUM_THREADS=$W
export TREE=${TREE:-analysis/configs/campaign_bpl}
D=logs/sweep_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$D"
export D W
list=$(python - <<'EOF'
import glob, json, os
tree = os.environ["TREE"]
todo = []
for path in sorted(glob.glob(os.path.join(tree, "*", "*.json"))):
    if os.path.basename(path) == "campaign.json":
        continue
    out = json.load(open(path))["output_dir"]
    if not any(glob.glob(os.path.join(out, sub, "plane_*.json")) for sub in ("data", "asimov")):
        todo.append(path)
k = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
n = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
off = (k * len(todo)) // max(n, 1)
print(" ".join(todo[off:] + todo[:off]))
EOF
)
echo "$(date +%H:%M) $(printf '%s\n' $list | grep -c .) unfinished planes under $TREE"
printf '%s\n' $list | xargs -P "$SLOTS" -I{} bash -c \
    'n=$(basename "{}" .json); s=$(basename "$(dirname "{}")"); { echo "=== {}"; python -m analysis "{}" --workers "$W"; echo "exit $?"; } > "$D/${s}_$n.out" 2>&1; echo "$(date +%H:%M) $s/$n: $(tail -1 "$D/${s}_$n.out")"'
