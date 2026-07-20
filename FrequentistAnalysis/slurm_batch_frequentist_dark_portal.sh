#!/bin/bash

# --- Slurm Resource Requests ---
#SBATCH --job-name=limdp
#SBATCH --partition=astro3_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --time=0-1:00:00

# --- Logging & Notifications ---
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=mbustamante@nbi.ku.dk
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py310
export NUMBA_CPU_NAME=generic
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# --- Execution ---
# Note: These paths are placeholders. Update them to point to your new 
# dark portal JSON configurations for use with main.py.

# Example: IceCube MESE 2025 limits for Dark Portal
# bash run_batch_frequentist_dark_portal.sh ../config/limit_setting/dark_portal/icecube_mese_2025/icecube_mese_2025_scenario_A.json
# bash run_batch_frequentist_dark_portal.sh ../config/limit_setting/dark_portal/icecube_mese_2025/icecube_mese_2025_scenario_B.json

# Example: Projections for 2040/2050
# bash run_batch_frequentist_dark_portal.sh ../config/limit_setting/dark_portal/proj_2040/dark_portal_2040_case_1.json
rm -rf /lustre/astro/mbustama/NuLRIDarkPortal/FrequentistAnalysis/src/physics/__pycache__
rm -rf /lustre/astro/mbustama/NuLRIDarkPortal/src/__pycache__
bash run_batch_frequentist_dark_portal.sh config/pion_2050_case_5_fixed_nuisance.json

# Alternatively, if you have all dark portal configs in one directory, 
# you can run the directory mode instead of listing every file manually:
# bash run_batch_frequentist_dark_portal.sh ../config/limit_setting/dark_portal/icecube_mese_2025/