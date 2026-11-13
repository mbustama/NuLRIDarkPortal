#!/bin/bash
# clean_after_run.sh -- run on the laptop, in the background, at every launch (2026-11-13, his rule: when no job
# follows to sweep the last jobs' caches, clean them myself): waits until no job of the checkout is queued, then runs
# tools/disk_report.sh --clean on Tycho, which deletes every cache folder whose jobs have ended (job_* folders and
# the shared folders registered in flux_cache/SHARED.txt), and prints the report.
#   bash tools/clean_after_run.sh <checkout under /lustre/astro/mbustama/external> [poll seconds, default 600]
CO=${1:?checkout}; POLL=${2:-600}
HOST=mbustama@astro03.hpc.ku.dk
DIR=/lustre/astro/mbustama/external/$CO
R() { timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$@" < /dev/null; }
while :; do
    if q=$(R "squeue -u mbustama -h -o %Z"); then          # ssh or squeue failing is not "no job queued"
        [ "$(echo "$q" | grep -cx "$DIR")" = 0 ] && break
    fi
    sleep "$POLL"
done
echo "$(date '+%F %T'): no job of $CO queued; cleaning"
R "bash /lustre/astro/mbustama/external/disk_report.sh --clean"
