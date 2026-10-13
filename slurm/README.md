# Running on Tycho

Tycho is the NBI cluster (SLURM). Log in to `mbustama@astro02.hpc.ku.dk`, or `astro03` if
`astro02` is unavailable. The repository lives in `/lustre/astro/mbustama/NuLRIDarkPortal`, the
environment in `/lustre/astro/mbustama/conda_envs/nulri`, and every command below runs from the
repository. The conventions follow GRBsim, NuFlavorSingleSource and GeoneutrinoDecay: partitions
`astro3_devel` (tests, up to 2 h), `astro3_short` (up to 12 h) and `astro3_long` (up to 5 days),
conda in `$HOME/anaconda3` (override with `CONDA_SH=... ENV_NAME=... sbatch ...`).

**Use a modest share of a node.** The scripts ask for 16 cores (an eighth of a 128-core node);
up to 32 per job is fine, more only after asking. Heavy runs, installs and large downloads go
here, not on the laptop.

## Setting up, once (done 2026-10-13)

The repository is private; Tycho's SSH key is registered with GitHub, so the clone uses SSH. The
environment pins the laptop's versions; pip's numpy uses OpenBLAS where the laptop's anaconda
numpy uses MKL, and numba compiles for each CPU, so results agree at the last digits, not
bitwise (`slurm/validate.sh` checks how closely).

```bash
cd /lustre/astro/mbustama
git clone git@github.com:mbustama/NuLRIDarkPortal.git      # this clone existed (July, branch main):
                                                          # remote switched to SSH, dev checked out
cd NuLRIDarkPortal && git checkout dev
source $HOME/anaconda3/etc/profile.d/conda.sh
conda create -y -p /lustre/astro/mbustama/conda_envs/nulri python=3.12 pip
conda activate /lustre/astro/mbustama/conda_envs/nulri
pip install numpy==1.26.4 scipy==1.15.3 numba==0.60.0 llvmlite==0.43.0 matplotlib==3.10.8 \
            tqdm==4.67.3 pytest==9.0.3 mpmath==1.3.0 tbb==2023.1.0 icc-rt==2020.0.133
```

`tbb` provides numba's TBB threading layer (spawn-safe with the solver's process pools) and
`icc-rt` the SVML library, as on the laptop.

## Running

Commit and push on the laptop first; on Tycho, `git pull`. Then:

```bash
sbatch slurm/test.sh                     # the test suite, a few minutes on astro3_devel
sbatch slurm/validate.sh                 # the four-point record against the laptop's (runs/)
sbatch slurm/analysis.sh analysis/configs/pion_2050_case_5_potential_emu.json
squeue -u $USER
```

Logs are `logs/slurm_<job name>_<job>.out`. `slurm/common.sh` prints the host, job, core count
and commit at the top of each, flagging a modified working tree. The analysis resumes from its
checkpoints when a job is resubmitted; its `manifest.json` records the host, commit and timing of
every run. Numba caches compiled kernels under `src/__pycache__/numba/<digest of the sources>`,
shared by all jobs; the first job after a code change compiles them (a few minutes).

## Bringing the results back

From the laptop, copy the outputs and commit them there:

```bash
rsync -av mbustama@astro02.hpc.ku.dk:/lustre/astro/mbustama/NuLRIDarkPortal/output/analysis/ output/analysis/
rsync -av mbustama@astro02.hpc.ku.dk:/lustre/astro/mbustama/NuLRIDarkPortal/runs/ runs/
```

and list notable runs in `runs/README.md`. After test runs, restore the Tycho clone
(`git checkout -- .`) so the next `git pull` is clean.
