#!/bin/bash
#SBATCH --job-name=nulri-analysis
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # an eighth of a 128-core node; up to 32 per job, ask before more
#SBATCH --mem=16G
# One analysis configuration (python -m analysis), its workers sized by the allocation. Usage, from
# the repository root:
#   sbatch slurm/analysis.sh analysis/configs/pion_2050_case_5_potential_emu.json
#   sbatch --cpus-per-task=32 slurm/analysis.sh CONFIG --planes      # more cores, one task only
# Checkpoints make a resubmitted job resume where the last one stopped.
source slurm/common.sh
CONFIG=$1; shift
python -m analysis "$CONFIG" --workers "${SLURM_CPUS_PER_TASK:-1}" "$@"
