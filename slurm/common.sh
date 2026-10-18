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
    trap 'kill "$PROGRESS_PID" 2>/dev/null' EXIT
fi
