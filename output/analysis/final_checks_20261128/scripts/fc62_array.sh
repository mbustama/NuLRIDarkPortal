#!/bin/bash
#SBATCH --partition=tycho1_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err
# Forecast refit with a 62-mass lightest-mass scan and chains of 2 (Mauricio, 2026-11-27: "Launch the forecast refit
# tonight"): line k+1 of PACKS (../make_fc62_configs.py); the campaign's own runner, 14 workers, one numba thread per
# worker as the refit of the present-day limits ran; records in new output directories. Submit from the checkout.
cd "$SLURM_SUBMIT_DIR" || exit 1
source slurm/common.sh
export NUMBA_NUM_THREADS=1
# a fresh worker process per chain or repair point (2026-11-27): the workers' memory creeps up otherwise
export NULRI_WORKER_MAXTASKS=1
export NULRI_HESE_MC=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_mc.npz
export NULRI_HESE_DETSYS=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_detector_systematics.npz
cfg=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" "${PACKS:?set PACKS}")
[ -z "$cfg" ] && { echo "no line $(( SLURM_ARRAY_TASK_ID + 1 )) in $PACKS"; exit 1; }
echo "=== $cfg"
run python -m analysis "$cfg" --workers "${SLURM_CPUS_PER_TASK}"
echo "exit $?"
