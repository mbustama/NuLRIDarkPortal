#!/bin/bash
# disk_report.sh -- run on Tycho (2026-11-12, after the solver caches filled the astro group quota): the account's and
# the group's use, every folder in flux_cache with its estimated size (files in one of its 256 buckets x 256 x the
# mean size of 20 of them; exact with --exact, slow), and job_* cache folders left by jobs no longer in the queue
# (a time limit can kill a job before its clean-up ends); --clean removes those.
#   bash tools/disk_report.sh [--exact] [--clean]
ROOT=/lustre/astro/mbustama/flux_cache
EXACT=; CLEAN=
for a in "$@"; do case $a in --exact) EXACT=1;; --clean) CLEAN=1;; esac; done
echo "== quota"
lfs quota -h -u "$USER" /lustre/astro | sed -n 3p
lfs quota -h -g astro /lustre/astro | sed -n 3p
echo "== flux_cache"
live=" $(squeue -u "$USER" -h -o '%A %F' | tr ' ' '\n' | sort -u | tr '\n' ' ') "
for d in "$ROOT"/*/; do
    d=${d%/}; n=$(basename "$d")
    if [ -n "$EXACT" ]; then
        echo "$n $(du -sh "$d" 2>/dev/null | cut -f1)"; continue
    fi
    b=$(ls -f "$d" | grep -v '^\.\.\?$' | head -1)
    if [ -z "$b" ]; then echo "$n empty"; continue; fi
    if [ -d "$d/$b" ]; then
        nb=$(ls -f "$d" | grep -vc '^\.\.\?$')
        nf=$(ls -f "$d/$b" | grep -vc '^\.\.\?$')
        mean=$(ls -f "$d/$b" | grep -v '^\.\.\?$' | head -20 | sed "s#^#$d/$b/#" | xargs stat -c %s 2>/dev/null | awk '{t+=$1;k++} END{print (k?t/k:0)}')
        awk -v n="$n" -v nb="$nb" -v nf="$nf" -v m="$mean" 'BEGIN{printf "%s ~%.2f TB, ~%.1f million files (estimate)\n", n, nb*nf*m/1e12, nb*nf/1e6}'
    else
        echo "$n $(du -sh "$d" 2>/dev/null | cut -f1)"
    fi
    case $n in job_*)
        id=${n#job_}; id=${id%%_*}
        if [[ "$live" != *" $id "* ]]; then
            echo "   left by job $id, no longer queued$([ -n "$CLEAN" ] && rm -rf "$d" && echo ': removed')"
        fi;;
    esac
done
