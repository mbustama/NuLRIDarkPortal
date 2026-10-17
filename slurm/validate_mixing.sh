#!/bin/bash
#SBATCH --job-name=nulri-mixval
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --partition=astro3_short
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
# Before the mixing-varied campaign (2026-10-18; docs/dev/MIXING_VARIATION.md), one node, everything at once:
#   - the template schemes against exact runs at the campaign's resolution, with timings (tools/template_validation.py),
#     at points on the earlier campaign's limit contours (OLD: its tree, read only);
#   - how much the lightest mass moves the chi^2 there (tools/mlightest_study.py), MeV and coupling-split planes.
# Usage, from the repository root: OLD=/lustre/astro/mbustama/external/nulri_bpl sbatch slurm/validate_mixing.sh
source slurm/common.sh
export NUMBA_NUM_THREADS=1
OLD=${OLD:-/lustre/astro/mbustama/external/nulri_bpl}
OUT=output/validation/mixing_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"

summary() {   # the earlier plane's summary for <data set> <name>
    ls "$OLD"/output/analysis/campaign_bpl/$1/$2/*/plane_*.json 2>/dev/null | grep -v jsonl | head -1
}
points() {    # three points on the earlier contour, as JSON
    python - "$1" <<'EOF'
import json, sys
sys.path.insert(0, "tools")
from mlightest_study import contour_points
s = json.load(open(sys.argv[1]))
x, y = s["names"]
print(json.dumps([{x: a, y: b} for a, b in contour_points(s, 3)]))
EOF
}

# template accuracy and timing: four configurations, one process each
for spec in now_ic26/dp_emu_pot_bal_pos now_ic26/b_3le_mev proj_future/dp_emu_pot_bal_pos proj_future/b_3le_mev \
            now_events/dp_emu_pot_bal_neg proj_2050/dp_mutau_mev; do
    ds=${spec%/*}; name=${spec#*/}
    s=$(summary "$ds" "$name")
    if [ -z "$s" ]; then s=$(summary now_ic26 "$name"); fi
    if [ -z "$s" ]; then echo "no earlier plane for $spec"; continue; fi
    pts=$(points "$s")
    python tools/template_validation.py analysis/configs/campaign_mix/$ds/$name.json --points "$pts" --samples 6 \
        --out "$OUT/templates_${ds}_${name}.json" > "$OUT/templates_${ds}_${name}.log" 2>&1 &
done

# the lightest mass: two MeV planes and two coupling-split planes (fixed mixing, the earlier configurations)
for spec in now_ic26/b_3le_mev proj_future/b_3le_mev now_events/dp_emu_pot_x1m08_pos now_events/dp_e_pot_x1m12_pos; do
    ds=${spec%/*}; name=${spec#*/}
    s=$(summary "$ds" "$name")
    if [ -z "$s" ]; then echo "no earlier plane for $spec"; continue; fi
    python tools/mlightest_study.py analysis/configs/campaign_bpl/$ds/$name.json "$s" --masses 5 --workers 5 \
        --out "$OUT/mlightest_${ds}_${name}.json" > "$OUT/mlightest_${ds}_${name}.log" 2>&1 &
done
wait
echo "done: $OUT"
tail -n 20 "$OUT"/*.log
