#!/bin/bash
#SBATCH --job-name=nulri-thermchk
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
# dp_si's thermal relics in every collision term against the reference at the analysis grid, and its time
# (tools/thermal_reference_checks.py, 2026-10-18).
source slurm/common.sh
python tools/thermal_reference_checks.py analysis/configs/campaign_ml/now_ic26/dp_e_mev.json --workers 18 2>&1 \
    | grep -v -e Warning -e "Em, P1, P2" -e "Cosmo Phase"
