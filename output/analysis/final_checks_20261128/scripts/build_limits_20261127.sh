#!/bin/bash
# Builds output/analysis/limits_20261127 (NOT committed): limits_20261121 with the MeV planes of today's data replaced
# by the stored planes merged with the 62-lightest-mass refit (tools/merge_ml62.py, tools/export_limits.py), from JSON
# copies of the Tycho results. Run from the repository root.
set -euo pipefail
S=$(dirname "$0"); W=$S/lim27; H=mbustama@astro03.hpc.ku.dk; X=/lustre/astro/mbustama/external
INC=(--include='*/' --include='plane_*.json' --include='plane_*.jsonl' --include='fit.json' --include='manifest.json' --include='refine_*.jsonl' --include='recheck.*.jsonl' --include='rerun_rs.*.jsonl' --include='reeval_events.*.jsonl' --include='rho_fix.*.jsonl' --exclude='*')
rm -rf $W/merged $W/lim; mkdir -p $W/raw/si/stored $W/raw/si/refit $W/raw/ml/stored $W/raw/ml/refit $W/merged/si/now_icecube $W/merged/ml/now_icecube
rsync -a --delete --prune-empty-dirs --exclude='b1[0-9][0-9][0-9]/' --include='si_*_fx/***' --exclude='*' -e "ssh -o BatchMode=yes" $H:$X/nulri_refit_set/output/analysis/campaign_si/now_icecube/ $W/raw/si/stored/
rm -rf $W/raw/si/stored/*_ul_fx
rsync -a --delete --prune-empty-dirs "${INC[@]}" -e "ssh -o BatchMode=yes" $H:$X/nulri_refit_set/output/analysis/campaign_si_ml62/now_icecube/ $W/raw/si/refit/
rsync -a --delete --prune-empty-dirs --exclude='b1[0-9][0-9][0-9]/' --include='le_l*_mev_fx/***' --include='dp_*_mev_fx/***' --exclude='*' -e "ssh -o BatchMode=yes" $H:$X/nulri_set1_ml/output/analysis/campaign_ml/now_icecube/ $W/raw/ml/stored/
rsync -a --delete --prune-empty-dirs "${INC[@]}" -e "ssh -o BatchMode=yes" $H:$X/nulri_set1_ml/output/analysis/campaign_ml62/now_icecube/ $W/raw/ml/refit/
for k in si ml; do
  for p in $(ls $W/raw/$k/refit); do
    cp -r $W/raw/$k/stored/$p $W/merged/$k/now_icecube/$p
    echo -n "$k $p: "; python tools/merge_ml62.py $W/raw/$k/refit/$p $W/merged/$k/now_icecube/$p | tail -1
  done
  python tools/export_limits.py $W/merged/$k --out $W/lim/$k > $W/export_$k.log 2>&1
  grep -E "^FAILED|Traceback" $W/export_$k.log && exit 1
  cat $W/export_$k.log | cut -c1-110
done
OUT=output/analysis/limits_20261127
rm -rf $OUT; cp -r output/analysis/limits_20261121 $OUT
for p in $(ls $W/lim/si/now_icecube); do rm -rf $OUT/nulri_refit_set/campaign_si/now_icecube/$p; cp -r $W/lim/si/now_icecube/$p $OUT/nulri_refit_set/campaign_si/now_icecube/$p; done
for p in $(ls $W/lim/ml/now_icecube); do rm -rf $OUT/nulri_set1_ml/campaign_ml/now_icecube/$p; cp -r $W/lim/ml/now_icecube/$p $OUT/nulri_set1_ml/campaign_ml/now_icecube/$p; done
echo "built $OUT"; du -sh $OUT | cut -f1
