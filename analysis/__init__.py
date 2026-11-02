"""
analysis

Statistical analysis of the dark-portal model: profile likelihoods, limits and, later,
coverage tests, built on the flux solver in src/. It replaces FrequentistAnalysis
(retired 2026-10-12); docs/dev/ANALYSIS_DESIGN.md explains the design.

The layers, from physics to statistics:
  model.py       parameter values -> fluxes at the detector (one solver call per point of
                 the parameters that change the propagation, cached; the source flavor
                 composition and the neutrino/antineutrino ratio are free, since the solver
                 is linear in them) -> energy-averaged flavor fractions.
  surfaces.py    the measured or projected flavor-composition likelihoods (f_e, f_mu).
  parameters.py  free parameters, their transformations and priors.
  objective.py   chi^2 = -2 ln L(data | model) + priors.
  fitting.py     minimization with several starts and a derivative-free polish.
  scans.py       1D and 2D profile likelihoods in parallel, with warm starts, checkpoints
                 and the repair of points a fit left above the true profile.
  intervals.py   intervals and contours from profiles.
  run.py         the command line: python -m analysis CONFIG.json

The solver modules live in src/ as plain modules; importing this package puts src/ on
sys.path.
"""
import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ANALYSIS_DIR)
