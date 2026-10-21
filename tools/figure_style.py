"""
figure_style.py -- the style every figure is drawn in: Mauricio's matplotlibrc at the repository root (copied
unchanged from GeoneutrinoDecay, 2026-10-21; Palatino through LaTeX, ticks inward on all four sides, large labels,
5 x 4.75 inch panels, 300 dpi). matplotlib reads matplotlibrc by itself only when run from the directory that holds
it, so the figure scripts call use() before they draw. The file's backend is left alone (matplotlib's rc_file skips
it): scripts save through Agg. Where LaTeX is missing (Tycho's compute nodes, CI), text falls back to matplotlib's
own rendering.
  from figure_style import use, pct; use()
"""
import os
import shutil

import matplotlib

RC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "matplotlibrc")
PANEL = (5.0, 4.75)   # inches: one panel, the matplotlibrc's figure.figsize


def use() -> None:
    """Draw every later figure in the repository's style."""
    matplotlib.rc_file(RC)
    if shutil.which("latex") is None:
        matplotlib.rcParams["text.usetex"] = False


def pct() -> str:
    """A percent sign that survives the text renderer in use (LaTeX reads a bare one as a comment)."""
    return r"\%" if matplotlib.rcParams["text.usetex"] else "%"
