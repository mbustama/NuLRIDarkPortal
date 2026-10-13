#!/bin/bash
#SBATCH --job-name=nulri-validate
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
# The four-point validation at production settings (tools/validate_record.py), compared with the
# laptop's record in runs/. Usage, from the repository root:
#   sbatch slurm/validate.sh                       # writes runs/record_<host>_<job>.json and compares
source slurm/common.sh
OUT="runs/record_$(hostname -s)_${SLURM_JOB_ID:-local}.json"
python tools/validate_record.py --out "$OUT" --cores "${SLURM_CPUS_PER_TASK:-1}" --compare runs/record_laptop.json
