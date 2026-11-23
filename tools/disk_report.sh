#!/bin/bash
# disk_report.sh -- run on Tycho (2026-11-12, after the solver caches filled the astro group quota): the account's and
# the group's use, and every folder in flux_cache with its estimated size (files in one of its buckets x buckets x the
# mean size of 20 of them; exact with --exact, slow) and whether anything still reads it:
#   job_<id>_<task>   a job's own cache (slurm/common.sh): finished once job <id> is no longer queued
#   shared folders    listed in flux_cache/SHARED.txt as "<folder> <checkout>" (slurm/common.sh registers every
#                     NULRI_CACHE_SHARED folder; older ones are added by hand): finished once no job of <checkout> is
#                     queued
#   anything else     untracked: reported, never removed
# --clean deletes the finished ones (16 removals in parallel) and their SHARED.txt lines. Jobs sweep job_* folders
# when they start; when no job follows, run this (tools/clean_after_run.sh does, once a run's jobs have ended).
#   bash tools/disk_report.sh [--exact] [--clean]
ROOT=/lustre/astro/mbustama/flux_cache
REG=$ROOT/SHARED.txt
EXACT=; CLEAN=
for a in "$@"; do case $a in --exact) EXACT=1;; --clean) CLEAN=1;; esac; done
echo "== quota"
lfs quota -h -u "$USER" /lustre/astro | sed -n 3p
lfs quota -h -g astro /lustre/astro | sed -n 3p
q=$(squeue -u "$USER" -h -o '%A %F %Z') || { echo "squeue failed: nothing is removed"; CLEAN=; }
ids=" $(echo "$q" | awk '{print $1; print $2}' | sort -u | tr '\n' ' ') "
dirs="
$(echo "$q" | awk '{print $3}' | sort -u)
"
size() {
    local d=$1 b nb nf mean
    if [ -n "$EXACT" ]; then du -sh "$d" 2>/dev/null | cut -f1; return; fi
    b=$(ls -f "$d" | grep -v '^\.\.\?$' | head -1)
    [ -z "$b" ] && { echo empty; return; }
    [ -d "$d/$b" ] || { du -sh "$d" 2>/dev/null | cut -f1; return; }
    nb=$(ls -f "$d" | grep -vc '^\.\.\?$'); nf=$(ls -f "$d/$b" | grep -vc '^\.\.\?$')
    mean=$(ls -f "$d/$b" | grep -v '^\.\.\?$' | head -20 | sed "s#^#$d/$b/#" | xargs stat -c %s 2>/dev/null \
           | awk '{t+=$1;k++} END{print (k?t/k:0)}')
    awk -v nb="$nb" -v nf="$nf" -v m="$mean" 'BEGIN{printf "~%.3f TB, ~%.2f million files (estimate)", nb*nf*m/1e12, nb*nf/1e6}'
}
remove() {
    ( cd "$1" && ls -f | grep -v '^\.\.\?$' | xargs -r -n 1 -P 16 rm -rf ) 2>/dev/null
    rm -rf "$1" && echo "   removed"
}
echo "== flux_cache"
for d in "$ROOT"/*/; do
    d=${d%/}; n=$(basename "$d")
    case $n in
        job_*)
            id=${n#job_}; id=${id%%_*}
            if [[ "$ids" == *" $id "* ]]; then echo "$n $(size "$d"): job $id running"
            else echo "$n $(size "$d"): job $id ended"; [ -n "$CLEAN" ] && remove "$d"; fi ;;
        *)
            co=$(awk -v f="$d" '$1 == f {print $2}' "$REG" 2>/dev/null | sort -u)
            if [ -z "$co" ]; then echo "$n $(size "$d"): UNTRACKED (not in SHARED.txt; never removed here)"; continue; fi
            live=; for c in $co; do [[ "$dirs" == *"
$c
"* ]] && live="$live $(basename "$c")"; done
            if [ -n "$live" ]; then echo "$n $(size "$d"): shared, read by running jobs of$live"
            else echo "$n $(size "$d"): shared, no job of $(echo $co | xargs -n1 basename | tr '\n' ' ')queued"
                 [ -n "$CLEAN" ] && remove "$d" && { grep -v "^$d " "$REG" > "$REG.tmp"; mv "$REG.tmp" "$REG"; }
            fi ;;
    esac
done
