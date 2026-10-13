# Sourced by every job script: the conda environment and the thread counts.
# The environment lives on lustre (/lustre/astro/mbustama/conda_envs/nulri). CONDA_SH and ENV_NAME
# (a name or a path) can be overridden at submission, e.g. ENV_NAME=myenv sbatch ...
# BLAS stays at one thread: the solver parallelizes with process pools and numba, and dp_core pins
# BLAS to one thread for the same reason (oversubscription); numba's pool is capped at the allocation.
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${ENV_NAME:-/lustre/astro/mbustama/conda_envs/nulri}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
echo "host $(hostname), job ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none}, ${SLURM_CPUS_PER_TASK:-?} cores, $(date), commit $(git rev-parse --short HEAD)$(git diff --quiet -- src analysis || echo ' (modified)')"
