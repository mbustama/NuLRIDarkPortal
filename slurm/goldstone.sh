#!/bin/bash
#SBATCH --job-name=goldstone
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
# The longitudinal-mode scatterings' validation (tools/validate_goldstone.py) and the new tests' timing.
source slurm/common.sh
python tools/validate_goldstone.py
python -m pytest tests/test_goldstone.py tests/test_si_light.py tests/test_unitarity.py tests/test_decay.py -q -p no:warnings --durations=8
echo "=== done $(date)"
