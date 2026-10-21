"""
figure_style.py -- the style every figure is drawn in: Mauricio's matplotlibrc, next to this file (copied
unchanged from GeoneutrinoDecay, 2026-10-21; Palatino through LaTeX, ticks inward on all four sides, large labels,
5 x 4.75 inch panels, 300 dpi). Not at the repository root: matplotlib reads a matplotlibrc in the working
directory by itself, which would switch LaTeX and the Tk backend on for every run from the root (CI and Tycho's
compute nodes have neither); the figure scripts call use() before they draw. The file's backend is left alone (matplotlib's rc_file skips
it): scripts save through Agg. Where LaTeX is missing (Tycho's compute nodes, CI), text falls back to matplotlib's
own rendering.
  from analysis.figure_style import use, pct, tex; use()
"""
import os
import re
import shutil

import matplotlib

RC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matplotlibrc")
PANEL = (5.0, 4.75)   # inches: one panel, the matplotlibrc's figure.figsize


def use() -> None:
    """Draw every later figure in the repository's style."""
    matplotlib.rc_file(RC)
    if shutil.which("latex") is None:                # matplotlib's own text, in a serif font that has every glyph
        matplotlib.rcParams.update({"text.usetex": False, "font.serif": ["DejaVu Serif"],
                                    "mathtext.fontset": "dejavuserif"})


def pct() -> str:
    """A percent sign that survives the text renderer in use (LaTeX reads a bare one as a comment)."""
    return r"\%" if matplotlib.rcParams["text.usetex"] else "%"


def tex(text: str) -> str:
    """Plain text made safe for the renderer in use: under LaTeX, _ % & # escaped outside $...$ (configuration
    titles and parameter names carry underscores)."""
    if not matplotlib.rcParams["text.usetex"]:
        return text
    parts = text.split("$")
    for k in range(0, len(parts), 2):
        parts[k] = re.sub(r"(?<!\\)([_%&#])", r"\\\1", parts[k])
    return "$".join(parts)
