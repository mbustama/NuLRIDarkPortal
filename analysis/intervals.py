"""
intervals.py

Confidence intervals from 1D profiles and Delta chi^2 maps from planes.

Delta chi^2 is measured from the lowest chi^2 found anywhere (the global fit or any profile
point, whichever is lower: a profile point below the global fit means the global fit stopped
short, and the reported best fit moves there). An interval at a threshold is a maximal run of
grid points with Delta chi^2 below it; its ends are interpolated linearly between grid points,
in log10 of the parameter for parameters fitted in log10. An end that reaches the edge of the
scanned range is reported as open: the interval extends at least that far. Points whose fit
failed (non-finite chi^2) are skipped.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Interval:
    lo: float
    hi: float
    lo_open: bool
    hi_open: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _crossing(u0, d0, u1, d1, t):
    if d1 == d0:
        return 0.5 * (u0 + u1)
    return u0 + (t - d0) * (u1 - u0) / (d1 - d0)


def intervals(x: Sequence[float], dchi2: Sequence[float], threshold: float, log: bool = False) -> List[Interval]:
    """The intervals where the profile dchi2 (over the ascending grid x) lies below threshold.
    Arguments: x, dchi2 -- arrays of equal length; threshold; log -- interpolate in log10 x.
    Returns a list of Interval (empty if no point lies below the threshold)."""
    x = np.asarray(x, dtype=float)
    d = np.asarray(dchi2, dtype=float)
    if x.shape != d.shape or x.ndim != 1:
        raise ValueError("x and dchi2 must be 1D arrays of equal length")
    ok = np.isfinite(d)                  # failed points are skipped, not counted as excluded
    x, d = x[ok], d[ok]
    u = np.log10(x) if log else x
    inside = d <= threshold
    out = []
    i = 0
    n = len(x)
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inside[j + 1]:
            j += 1
        lo_open = i == 0
        hi_open = j == n - 1
        lo = u[0] if lo_open else _crossing(u[i - 1], d[i - 1], u[i], d[i], threshold)
        hi = u[-1] if hi_open else _crossing(u[j], d[j], u[j + 1], d[j + 1], threshold)
        out.append(Interval(float(10.0 ** lo if log else lo), float(10.0 ** hi if log else hi), lo_open, hi_open))
        i = j + 1
    return out


def summarize_profile(x: Sequence[float], chi2: Sequence[float], chi2_min: float, thresholds: Dict[str, float],
                      log: bool = False) -> dict:
    """Delta chi^2 of a profile and its intervals at every threshold:
    {"dchi2": [...], "intervals": {label: [Interval dicts]}}."""
    d = np.asarray(chi2, dtype=float) - float(chi2_min)
    return dict(dchi2=[float(v) for v in d],
                intervals={label: [iv.as_dict() for iv in intervals(x, d, t, log)] for label, t in thresholds.items()})


def describe(iv: dict, unit: str = "") -> str:
    """'a < x < b', with open ends written as bounds of the scanned range."""
    if iv["lo_open"] and not iv["hi_open"]:
        return f"x < {iv['hi']:.4g}{unit} (open below the scanned {iv['lo']:.4g})"
    if iv["hi_open"] and not iv["lo_open"]:
        return f"x > {iv['lo']:.4g}{unit} (open above the scanned {iv['hi']:.4g})"
    if iv["lo_open"] and iv["hi_open"]:
        return f"the whole scanned range [{iv['lo']:.4g}, {iv['hi']:.4g}]{unit}"
    return f"{iv['lo']:.4g} < x < {iv['hi']:.4g}{unit}"
