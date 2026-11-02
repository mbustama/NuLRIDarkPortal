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
sbatch slurm/analysis_list.sh CONFIG1 CONFIG2 ...          # several, one after the other
SLOTS=4 sbatch slurm/analysis_pack.sh CONFIG1 CONFIG2 ...  # several, 4 at a time with 4 workers each
squeue -u $USER
```

`analysis_pack.sh` (2026-10-14) keeps a job's cores busy when global fits would idle most of them (a global fit
runs one start per worker; one start wandering to strong couplings held 12-15 of 16 cores idle for 40% of a run):
one configuration's global fit overlaps the others' planes. Its per-configuration logs are
`logs/pack_<job>/<name>.out`.

**Shared solver-run cache** (2026-10-14): `NULRI_FLUX_CACHE=<directory>` makes every analysis process store and
reuse its solver runs on disk (keyed by the full solver configuration, the flux mode and the solver's code digest;
read back bit for bit). The planes' first repair pass re-evaluates every point on whichever worker is free, so
without it most repairs re-solve their point (7 runs each when the spectral index floats). Use one directory per
campaign and delete it afterwards (~0.4 GB per plane with 7 index nodes):

```bash
NULRI_FLUX_CACHE=/lustre/astro/mbustama/flux_cache/<campaign> SLOTS=4 sbatch slurm/analysis_pack.sh CONFIG1 ...
```

**The plane campaign** (2026-10-14; docs/dev/SCOPE_WIDENING_PLAN.md step 6): `tools/make_campaign_configs.py`
writes the configurations (analysis/configs/campaign/<set>/), `tools/campaign_packs.py` packs them into jobs of about
8 hours on 16 cores (longest first), and `slurm/campaign_array.sh` runs one pack per array task, throttled to 7 at a
time (112 cores):

```bash
python tools/campaign_packs.py --sets now_ic26 now_events proj_2050 --hours 8 --out analysis/configs/campaign/packs.txt
NULRI_FLUX_CACHE=/lustre/astro/mbustama/flux_cache/campaign \
NULRI_HESE_MC=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_mc.npz \
NULRI_HESE_DETSYS=/lustre/astro/mbustama/external/hese_7yr/reduced/hese_detector_systematics.npz \
PACKS=analysis/configs/campaign/packs.txt SLOTS=4 sbatch --array=0-<n-1>%7 --exclude=node812 slurm/campaign_array.sh
```

**The broken power law's campaign** (2026-10-15; docs/dev/BROKEN_POWER_LAW.md): the same with the configurations
under analysis/configs/campaign_bpl/<set>/ (four sets: the future detectors' projection proj_future too) and a fresh
cache directory (the solver's code changed, so no run of the first campaign is reused):

```bash
python tools/campaign_packs.py --tree campaign_bpl --sets now_ic26 now_events proj_2050 proj_future --hours 8 \
    --out analysis/configs/campaign_bpl/packs.txt
NULRI_FLUX_CACHE=/lustre/astro/mbustama/flux_cache/campaign_bpl ... PACKS=analysis/configs/campaign_bpl/packs.txt \
    SLOTS=4 sbatch --array=0-<n-1>%7 --exclude=node812 slurm/campaign_array.sh
```

**The clean-up** (2026-10-17): packs that hit the 24-hour limit leave planes unfinished (their checkpoints kept).
`slurm/campaign_sweep.sh` lists, when it starts, every configuration under TREE without a finished plane and runs
them again from their checkpoints; its array tasks share the list through the run locks. Queue it behind the campaign
so the total stays at 112 cores:

```bash
TREE=analysis/configs/campaign_bpl NULRI_FLUX_CACHE=... NULRI_HESE_MC=... NULRI_HESE_DETSYS=... SLOTS=4 \
    sbatch --array=0-6%7 --exclude=node812 --dependency=afterany:<the campaign's array> slurm/campaign_sweep.sh
```

**A job far below its cores** (2026-10-14): look inside it with `srun --jobid=<job> --overlap -n1 -c1 ps -o pid,time,stat,args -u $USER` and `cat /proc/<pid>/wchan`. Processes mostly in state D/I waiting in `cl_sync_io_wait` are waiting on the node's Lustre client, not computing: node812 did this twice (a quarter of the speed of node810/811 for the same code). Resubmit elsewhere with `--exclude=<node> --dependency=afterany:<old job>`, so the new job starts only once the old one is gone, and the analysis resumes from its checkpoints.

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
