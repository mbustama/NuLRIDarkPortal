#!/bin/bash
#SBATCH --partition=tycho1_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --exclude=node664
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
# Final checks, index ranges (2026-11-27): line k+1 of packs_final_idx.txt (../make_final_checks.py): the refit's
# blocks with gamma in 2-4 and gamma_low in 0-2.5; the campaign's own runner. Submit from the checkout.
cd "$SLURM_SUBMIT_DIR" || exit 1
source slurm/common.sh
export NUMBA_NUM_THREADS=1
export NULRI_HESE_MC=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_mc.npz
export NULRI_HESE_DETSYS=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_detector_systematics.npz
cfg=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" packs_final_idx.txt)
[ -z "$cfg" ] && { echo "no line $(( SLURM_ARRAY_TASK_ID + 1 ))"; exit 1; }
echo "=== $cfg"
run python -m analysis "$cfg" --workers "${SLURM_CPUS_PER_TASK}"
echo "exit $?"
