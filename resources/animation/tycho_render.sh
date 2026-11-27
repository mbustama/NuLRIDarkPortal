#!/bin/bash
# Renders the animation on Tycho: sbatch tycho_render.sh preview|video, from /lustre/astro/mbustama/dp_video/anim,
# which mirrors this folder (make_video.py, data/, build/cache/); ../repo mirrors the files of the repository it reads
# (analysis/__init__.py, figure_style.py, matplotlibrc, trip_illustration.py; paper/main.tex). Python and ffmpeg
# (imageio-ffmpeg's static build) come from NuFlavorGalacticPlane's render venv, used as it is. Its own folder, its
# own job: the paper's runs and their clones are not touched.
#SBATCH --job-name=dp-video
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=astro3_short
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export NULRI_REPO=/lustre/astro/mbustama/dp_video/repo
PY=/lustre/astro/mbustama/gp_video/venv/bin/python
FFMPEG=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
export FFMPEG
MODE=${1:-preview}
echo "host $(hostname); latex $(which latex); ffmpeg $FFMPEG"
$PY -u make_video.py --stills 26.2,84 || exit 1          # one still of the trip and one of the plots, to compare
if [ "$MODE" = video ]; then
    $PY -u make_video.py --video -j "${SLURM_CPUS_PER_TASK:-16}"
else
    $PY -u make_video.py --preview -j "${SLURM_CPUS_PER_TASK:-16}"
fi
