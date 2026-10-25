#!/bin/bash
#SBATCH --job-name=si-pilot
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
# The SI sub-keV pilot (2026-10-25): tools/pilot_fits.py at four allowed points of each model's sub-keV plane
# (now_ic26, texture mu, pion sources), the four models at once, 8 workers each. Usage: sbatch slurm/si_pilot.sh
source slurm/common.sh
D=analysis/configs/campaign_si/${DATASET:-now_ic26}
V="1e-20,1e-26 1e-6,1e-11 1,1e-5 100,1e-4"
S="1e-12,1e-9 1e-6,5e-8 1e-3,1e-7 10,3e-3"
python tools/pilot_fits.py $D/si_vector_majorana_mu_ul${FX:-}.json --workers 8 --points $V > logs/si_pilot_vm_${SLURM_JOB_ID}.txt 2>&1 &
python tools/pilot_fits.py $D/si_vector_dirac_mu_ul${FX:-}.json --workers 8 --points $V > logs/si_pilot_vd_${SLURM_JOB_ID}.txt 2>&1 &
python tools/pilot_fits.py $D/si_scalar_majorana_mu_ul${FX:-}.json --workers 8 --points $S > logs/si_pilot_sm_${SLURM_JOB_ID}.txt 2>&1 &
python tools/pilot_fits.py $D/si_scalar_dirac_mu_ul${FX:-}.json --workers 8 --points $S > logs/si_pilot_sd_${SLURM_JOB_ID}.txt 2>&1 &
wait
grep -h "PILOT\|chi2" logs/si_pilot_*_${SLURM_JOB_ID}.txt
echo "=== done $(date)"
