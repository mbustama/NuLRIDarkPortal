#!/bin/bash
#SBATCH --job-name=nulri-thermchk
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
# The thermal reference's checks on the production path (tools/thermal_reference_checks.py, 2026-10-18).
source slurm/common.sh
export NUMBA_NUM_THREADS=1
python tools/thermal_reference_checks.py analysis/configs/campaign_ml/now_ic26/dp_e_mev.json --workers 16 2>&1 \
    | grep -v -e Warning -e "Em, P1, P2" -e "Cosmo Phase"
