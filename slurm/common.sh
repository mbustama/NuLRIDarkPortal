# Sourced by every job script: the conda environment and the thread counts.
# The environment lives on lustre (/lustre/astro/mbustama/conda_envs/nulri). CONDA_SH and ENV_NAME
# (a name or a path) can be overridden at submission, e.g. ENV_NAME=myenv sbatch ...
# BLAS stays at one thread: the solver parallelizes with process pools and numba, and dp_core pins
# BLAS to one thread for the same reason (oversubscription); numba's pool is capped at the allocation.
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${ENV_NAME:-/lustre/astro/mbustama/conda_envs/nulri}"
# numba without AVX-512 from the first import on (src/dp_core.py, 2026-10-18: LLVM 14 miscompiled it on Zen 4)
export NUMBA_CPU_FEATURES=${NUMBA_CPU_FEATURES:-$(python -c "import llvmlite.binding as ll; ll.initialize(); ll.initialize_native_target(); print(ll.get_host_cpu_features().flatten().replace('+avx512', '-avx512'))" 2>/dev/null)}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
echo "host $(hostname), job ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores, $(date), commit $(git rev-parse --short HEAD)$(git diff --quiet -- src analysis || echo ' (modified)')"
# Progress lines for the Tycho monitor (2026-10-17): campaign jobs (PACKS) and clean-up jobs (TREE) print
# "PROGRESS <p>% | ..." to their output now and every PROGRESS_EVERY seconds (tools/job_progress.py: the planes'
# checkpoints, weighted by cost). The reporter runs beside the job and stops with it; any failure of it is ignored.
if [ -n "${PACKS:-}${TREE:-}" ] && [ -f tools/job_progress.py ]; then
    ( python tools/job_progress.py 2>/dev/null
      while sleep "${PROGRESS_EVERY:-300}"; do python tools/job_progress.py 2>/dev/null; done ) &
    PROGRESS_PID=$!
fi
# The solver-run cache on disk (analysis/model.py, NULRI_FLUX_CACHE). 2026-11-12: shared folders per campaign, never
# cleaned, reached ~25 TB and 36 million files and filled the astro group quota. Now each job keeps its own folder,
# deleted when the job ends -- normally, on failure, on scancel or at the time limit (Slurm sends TERM first) -- and
# each process writes at most NULRI_FLUX_CACHE_MAX_GB (analysis/model.py, default 2 GB). A job script must not set
# NULRI_FLUX_CACHE itself; a shared folder needs NULRI_CACHE_SHARED=<folder> at submission, with his OK, and is
# measured in every report (tools/disk_report.sh) and deleted when its last reader ends. A job script that exports
# NULRI_FLUX_CACHE after sourcing this file would bypass all of it; such scripts are refused below.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -f "$0" ] && grep -qE '^[^#]*NULRI_FLUX_CACHE=' "$0"; then
    echo "ERROR: this job script sets NULRI_FLUX_CACHE itself; remove it (a shared folder: NULRI_CACHE_SHARED at" \
         "submission, slurm/common.sh)" >&2
    exit 2
fi
if [ -n "${NULRI_CACHE_SHARED:-}" ]; then
    export NULRI_FLUX_CACHE="$NULRI_CACHE_SHARED"
    JOB_CACHE=
else
    JOB_CACHE=/lustre/astro/mbustama/flux_cache/job_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
    export NULRI_FLUX_CACHE="$JOB_CACHE"
fi
echo "solver cache: $NULRI_FLUX_CACHE ($([ -n "$JOB_CACHE" ] && echo 'this job only, deleted at its end' || echo SHARED))"
# Folders left by jobs that died before their clean-up ran (killed hard, node failure) are removed, in the background,
# by every Slurm job that starts: any job_<id>_* whose job is no longer in the queue. If squeue answers nothing (it
# always lists this job), nothing is removed.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    ( live=" $(squeue -u "$USER" -h -o '%A' 2>/dev/null | tr '\n' ' ') "
      [ -z "${live// /}" ] && exit 0
      for d in /lustre/astro/mbustama/flux_cache/job_*; do
          [ -d "$d" ] || continue
          id=${d##*/job_}; id=${id%%_*}
          [[ "$live" == *" $id "* ]] || { rm -rf "$d"; echo "removed leftover solver cache $d"; }
      done ) &
fi
_job_end() {
    [ -n "${PROGRESS_PID:-}" ] && kill "$PROGRESS_PID" 2>/dev/null
    case "$JOB_CACHE" in /lustre/astro/mbustama/flux_cache/job_*) rm -rf "$JOB_CACHE" ;; esac
}
trap _job_end EXIT
trap 'exit 143' TERM INT
