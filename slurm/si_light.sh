#!/bin/bash
#SBATCH --job-name=si-light
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# The secret interactions below 1 keV (tools/validate_si_light.py): the checks named in CHECKS, in turn, then the
# tests of the new module. Usage, from the repository root:  CHECKS="unchanged limits steps" sbatch slurm/si_light.sh
source slurm/common.sh
for c in ${CHECKS:-unchanged limits galactic steps crosscheck}; do
    echo "=== $c $(date)"
    python tools/validate_si_light.py "$c" || echo "FAILED $c"
done
[ -n "${PLOT:-}" ] && python tools/plot_si_subkev_regions.py
echo "=== tests $(date)"
python -m pytest tests/test_si_light.py -q -p no:warnings
echo "=== done $(date)"
