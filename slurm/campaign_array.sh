#!/bin/bash
#SBATCH --job-name=nulri-campaign
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --partition=astro3_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # up to 32 per job, ask before more
#SBATCH --mem=32G
# The plane campaign (2026-10-14; tools/make_campaign_configs.py, tools/campaign_packs.py): array task k runs line k
# of PACKS (configurations separated by spaces) as analysis_pack.sh does -- SLOTS at a time, cpus / SLOTS workers each,
# logs in logs/pack_<array job>_<task>/<name>.out. The %7 throttle keeps the campaign at 7 x 16 = 112 cores:
#   NULRI_FLUX_CACHE=/lustre/astro/mbustama/flux_cache/campaign NULRI_HESE_MC=... NULRI_HESE_DETSYS=... \
#   PACKS=analysis/configs/campaign/packs.txt SLOTS=4 sbatch --array=0-<n-1>%7 --exclude=node812 slurm/campaign_array.sh
# campaign_ml (2026-10-18): 11 tasks of 16 cores at a time = 176, with PROGRESS lines for the whole campaign:
#   ... PROGRESS_TREE=output/analysis/campaign_ml PROGRESS_TOTAL=874432 PACKS=analysis/configs/campaign_ml/packs.txt \
#   SLOTS=4 sbatch --array=0-44%11 --time=12:00:00 slurm/campaign_array.sh
source slurm/common.sh
SLOTS=${SLOTS:-4}
W=$(( ${SLURM_CPUS_PER_TASK:-1} / SLOTS ))
[ "$W" -lt 1 ] && W=1
export NUMBA_NUM_THREADS=$W
line=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "${PACKS:?set PACKS to the pack list}")
[ -z "$line" ] && { echo "no pack at line $(( ${SLURM_ARRAY_TASK_ID:-0} + 1 )) of $PACKS"; exit 1; }
D=logs/pack_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$D"
export D W
# PROGRESS lines for the whole campaign in this task's .out (2026-10-18; tools/exact_progress.py), when PROGRESS_TREE
# names the campaign's output tree and PROGRESS_TOTAL its number of points
if [ -n "${PROGRESS_TREE:-}" ]; then
    python tools/exact_progress.py "$PROGRESS_TREE" --total "${PROGRESS_TOTAL:-0}" --cores 176 --watch --interval 10 \
        --heartbeat 600 &
    PROG=$!
fi
printf '%s\n' $line | xargs -P "$SLOTS" -I{} bash -c \
    'n=$(basename "{}" .json); { echo "=== {}"; python -m analysis "{}" --workers "$W"; echo "exit $?"; } > "$D/$n.out" 2>&1; echo "$(date +%H:%M) $n: $(tail -1 "$D/$n.out")"'
rc=$?
[ -n "${PROG:-}" ] && kill "$PROG" 2>/dev/null
exit $rc
