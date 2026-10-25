#!/bin/bash
#SBATCH --job-name=si-pilot2
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
# The SI sub-keV pilot at steady state (2026-10-25): 16 allowed points per model on 4 workers each (four fits per
# worker, the first one paying the worker's numba compile), the four models at once. Usage: sbatch slurm/si_pilot2.sh
source slurm/common.sh
D=analysis/configs/campaign_si/${DATASET:-now_ic26}
V="1e-25,1e-31 1e-25,1e-28 1e-18,1e-24 1e-18,1e-20 1e-12,1e-17 1e-12,1e-14 1e-9,1e-14 1e-9,1e-10 1e-6,1e-12 1e-6,1e-8 1e-6,3e-5 1e-3,1e-8 1e-3,1e-5 1,1e-6 1,1e-4 100,1e-3"
S="1e-12,1e-10 1e-12,1e-9 1e-9,1e-8 1e-9,1e-7 1e-6,1e-8 1e-6,5e-8 1e-6,1e-6 1e-6,1e-3 1e-3,1e-8 1e-3,1e-7 1e-3,1e-5 1e-3,3e-3 1,1e-7 1,1e-3 100,3e-3 100,1e-2"
for m in vector_majorana vector_dirac scalar_majorana scalar_dirac; do
    P=$V; [[ $m == scalar* ]] && P=$S
    python tools/pilot_fits.py $D/si_${m}_mu_ul${FX:-}.json --workers 4 --points $P > logs/si_pilot2_${m}_${SLURM_JOB_ID}.txt 2>&1 &
done
wait
grep -h "PILOT" logs/si_pilot2_*_${SLURM_JOB_ID}.txt
echo "=== done $(date)"
