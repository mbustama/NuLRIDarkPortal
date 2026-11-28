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
# Final checks, single columns of the real data (2026-11-27): line k+1 of final_cols.txt =
# "config mass_eV true_log10g n_random out": tools/coverage_limits.py --toys 0 (the limits' own profile and reading):
# the deepest dips with 16 extra starts of every fit, and one ultralight column with a 62-mass scan of the lightest
# mass. Submit from the checkout.
cd "$SLURM_SUBMIT_DIR" || exit 1
source slurm/common.sh
export NUMBA_NUM_THREADS=1
export NULRI_HESE_MC=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_mc.npz
export NULRI_HESE_DETSYS=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_detector_systematics.npz
read -r cfg mass tg nr out <<< "$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" final_cols.txt)"
[ -z "$cfg" ] && { echo "no line $(( SLURM_ARRAY_TASK_ID + 1 ))"; exit 1; }
echo "=== $cfg mass $mass true $tg starts $nr -> $out"
mkdir -p output/final_cols
run python tools/coverage_limits.py "$cfg" --mass "$mass" --true-log10g "$tg" --window-hi 0.5 --toys 0 \
    --n-random "$nr" --workers "${SLURM_CPUS_PER_TASK}" --out "output/final_cols/$out"
echo "exit $?"
