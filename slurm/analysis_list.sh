#!/bin/bash
#SBATCH --job-name=nulri-analysis-list
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # an eighth of a 128-core node; up to 32 per job, ask before more
#SBATCH --mem=16G
# Several analysis configurations, one after the other in one job (python -m analysis each). Usage:
#   sbatch slurm/analysis_list.sh CONFIG1 CONFIG2 ...
source slurm/common.sh
for CONFIG in "$@"; do
    echo "=== $CONFIG"
    python -m analysis "$CONFIG" --workers "${SLURM_CPUS_PER_TASK:-1}"
done
