#!/bin/bash
# collect_limits.sh -- the end of a run (his rule, 2026-11-13: only the limits are kept, as JSON in the repository).
# Run on the laptop, from the repository root, once a run's limits are final (its jobs ended, refinements done):
#   1. copies the run's plane results from its Tycho checkout (summaries, point records, fits, manifests, the later
#      passes' records) to a temporary folder;
#   2. writes the limits with tools/export_limits.py to output/analysis/limits_<tag>/<checkout>;
#   3. commits and pushes that folder;
#   4. with --delete, and only after the push succeeded: deletes the checkout on Tycho (plane results, records, logs,
#      configurations) and the temporary copy.
# Refused while any job of the checkout is queued or running. Solver caches are not here: each job deletes its own
# (slurm/common.sh).
#   bash tools/collect_limits.sh <checkout> <tag> [--delete]      e.g. nulri_forecast 20261120 --delete
set -euo pipefail
CO=${1:?checkout name under /lustre/astro/mbustama/external}; TAG=${2:?tag, e.g. the date}; DEL=${3:-}
HOST=mbustama@astro03.hpc.ku.dk
EXT=/lustre/astro/mbustama/external
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
ssh -o BatchMode=yes "$HOST" "squeue -u mbustama -h -o %Z" < /dev/null | grep -qx "$EXT/$CO" \
    && { echo "refused: jobs of $CO are still queued or running"; exit 1; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
rsync -a --prune-empty-dirs --include='*/' --include='plane_*.json' --include='plane_*.jsonl' --include='fit.json' \
      --include='manifest.json' --include='refine_*.jsonl' --include='recheck.*.jsonl' --include='rerun_rs.*.jsonl' \
      --include='reeval_events.*.jsonl' --include='rho_fix.*.jsonl' --exclude='*' \
      "$HOST:$EXT/$CO/output/analysis/" "$TMP/"
OUT=output/analysis/limits_$TAG/$CO
rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"
python tools/export_limits.py "$TMP" --out "$OUT" | tee "$OUT.log"
grep -q '^FAILED' "$OUT.log" && { echo "export failed for some planes: nothing committed or deleted"; exit 1; }
git add "$OUT" "$OUT.log"
git commit -q -m "Limits of $CO (Tycho run), as JSON (tools/collect_limits.sh)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
GIT_CONFIG_NOSYSTEM=1 git push -q origin "$(git branch --show-current)"
git fetch -q origin && [ -z "$(git log "origin/$(git branch --show-current)..HEAD" --oneline)" ] \
    || { echo "push not confirmed: nothing deleted"; exit 1; }
echo "pushed $(git log --oneline -1)"
if [ "$DEL" = "--delete" ]; then
    ssh -o BatchMode=yes "$HOST" "cd $EXT && ls -d $CO/*/ | xargs -n 1 -P 16 rm -rf && rm -rf $CO && echo 'deleted $EXT/$CO'" < /dev/null
fi
