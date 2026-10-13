#!/bin/bash
#SBATCH --job-name=nulri-test
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_devel
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
# The regression suite and the analysis tests (a few minutes, most of it numba compiling the
# kernels on the first run on a node type). Usage, from the repository root:
#   sbatch slurm/test.sh
source slurm/common.sh
python -m pytest tests/ -q -p no:warnings
