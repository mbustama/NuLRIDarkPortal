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


# The campaign's data sets, as every figure labels and colours them (Mauricio, 2026-10-21: colours from his coolors
# palette coolors.co/001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226 -- auburn, gamboge, dark
# cyan, which pass the dataviz skill's colour-blind checks; regions in opaque tints of them, edges in the colours).
# now_ic26: IceCube's MESE flavor composition (2025; 11.4 yr) and its 2026 spectrum; now_events: HESE 7.5 yr and
# IceTracks-DR2's IC79 + IC86 (312.8 + 3920.6 days, 11.6 yr); proj_future: the 8 future detectors, 10 yr each.
DATASETS = [("now_ic26", "Today: IC MESE 11.4 yr flavor + spectrum", "#9b2226"),
            ("now_events", "Today: IC HESE 7.5 yr + DR2 11.6 yr", "#ee9b00"),
            ("proj_future", "Projected: combined detectors", "#0a9396")]


def tint(color, amount=0.55):
    """`color` mixed with white (the regions' fill; NuFlavorRatiosSteriles' paper figures)."""
    from matplotlib.colors import to_rgb
    return tuple(1.0 - (1.0 - c) * (1.0 - amount) for c in to_rgb(color))


def ticks(ax, x, y):
    """Major and minor tick spacings: x = (major, minor), y = (major, minor)."""
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(x[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(x[1]))
    ax.yaxis.set_major_locator(MultipleLocator(y[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(y[1]))
