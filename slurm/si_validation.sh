#!/bin/bash
#SBATCH --job-name=nulri-si-validation
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # numba threads of the table builds and per-step integrals
#SBATCH --mem=16G                   # the 1600-bin, 4-substep reference holds ~1.5 GB of tables
# Validation runs of the secret-interaction transport (tools/si_validation.py). Usage, from the
# repository root:
#   sbatch slurm/si_validation.sh absorption convergence fig5
# Results in output/si_validation/.
source slurm/common.sh
python tools/si_validation.py "$@"
