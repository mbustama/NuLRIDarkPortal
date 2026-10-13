#!/bin/bash
#SBATCH --job-name=nulri-tool
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # numba threads (prange) of the tool
#SBATCH --mem=8G
# One-off tools (tools/*.py), numba threads = the allocation. Usage, from the repository root:
#   sbatch [--cpus-per-task=32] [--job-name=...] slurm/tool.sh mw_neff_tables.py [ARGS]
source slurm/common.sh
python tools/"$@"
