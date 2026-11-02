#!/bin/bash
#SBATCH --job-name=nulri-analysis-pack
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # up to 32 per job, ask before more
#SBATCH --mem=32G
# Several analysis configurations run concurrently in one job (2026-10-14): SLOTS at a time (default 4), each with
# cpus / SLOTS workers, the next starting as one finishes, so one configuration's global fit (few busy cores) overlaps
# the others' planes. Each configuration logs to logs/pack_<job>/<name>.out. Usage, from the repository root:
#   SLOTS=4 sbatch [--cpus-per-task=16] slurm/analysis_pack.sh CONFIG1 CONFIG2 ...
source slurm/common.sh
SLOTS=${SLOTS:-4}
W=$(( ${SLURM_CPUS_PER_TASK:-1} / SLOTS ))
[ "$W" -lt 1 ] && W=1
export NUMBA_NUM_THREADS=$W         # each slot's share of the allocation
D=logs/pack_${SLURM_JOB_ID:-local}
mkdir -p "$D"
export D W
printf '%s\n' "$@" | xargs -P "$SLOTS" -I{} bash -c \
    'n=$(basename "{}" .json); { echo "=== {}"; python -m analysis "{}" --workers "$W"; echo "exit $?"; } > "$D/$n.out" 2>&1; echo "$(date +%H:%M) $n: $(tail -1 "$D/$n.out")"'
