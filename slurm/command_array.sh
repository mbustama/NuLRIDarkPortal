#!/bin/bash
#SBATCH --job-name=nulri-cmd
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
#SBATCH --partition=astro3_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
# Array task k runs line k of COMMANDS (one shell command per line; {task} is replaced by k), e.g. the grid extension
# (tools/refine_grid.py run <manifest> {task} --workers 32). Usage, from the repository root:
#   COMMANDS=<file> sbatch --array=0-<n-1> slurm/command_array.sh
# (export NULRI_FLUX_CACHE, NULRI_HESE_MC, NULRI_HESE_DETSYS as the campaign does; NUMBA_NUM_THREADS is 1, or
# NUMBA_THREADS_OVERRIDE for a tool that uses numba threads itself: the
# parallelism is the tools' worker pools)
source slurm/common.sh
export NUMBA_NUM_THREADS=${NUMBA_THREADS_OVERRIDE:-1}
line=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "${COMMANDS:?set COMMANDS to the command list}")
[ -z "$line" ] && { echo "no command at line $(( ${SLURM_ARRAY_TASK_ID:-0} + 1 )) of $COMMANDS"; exit 1; }
line=${line//\{task\}/${SLURM_ARRAY_TASK_ID:-0}}
echo "$line"
bash -c "$line"
