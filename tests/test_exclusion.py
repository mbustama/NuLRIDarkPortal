"""The excluded regions as the figures read them (analysis/exclusion.py), on synthetic plane summaries."""
import json

import numpy as np
import pytest

from analysis import exclusion as ex


def write_plane(path, x, y, chi2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(names=["m_Zd_eV", "g_D"], grids=[list(x), list(y)],
                                    chi2=[[None if not np.isfinite(v) else float(v) for v in row] for row in chi2])))
    return str(path)


def lowest_excluded(lx, G, F, m):
    col = np.asarray(F[:, int(np.argmin(abs(lx - m)))].filled(-1.0))
    hit = np.flatnonzero(col > 0)
    return G[hit[0]] if len(hit) else None


def test_signed_distance_and_clipping():
    G = np.arange(-10.0, 0.0 + 1e-9, 0.5)
    excl = G >= -4.0
    s = ex.signed(G, excl)
    assert np.all(s[excl] > 0) and np.all(s[~excl] < 0)
    assert s[np.argmin(abs(G + 4.0))] == pytest.approx(0.25)          # the edge halfway between -4.5 and -4.0
    assert s.min() == -ex.D and s.max() == ex.D                         # clipped
    assert np.all(ex.signed(G, np.zeros_like(excl)) == -ex.D)           # nothing excluded


def test_field_crossings_upper_limits_and_specks(tmp_path):
    """Per mass, the crossing at the linear interpolation of Delta chi^2 in log10 g; where the data prefer an effect
    (a dip at 1e-17), the no-effect couplings below it are disfavoured but not excluded (upper limits only); a lone
    point a hair over 3.84 not excluded."""
    x = 10.0 ** np.arange(-10, -4)                                       # 6 masses
    y = 10.0 ** np.arange(-20.0, -9.5, 0.5)                             # 21 couplings
    ly = np.log10(y)
    chi = np.where(ly[None, :] >= -14.0, 20.0, 0.0) * np.ones((len(x), 1)) + 1.0
    chi[:, np.argmin(abs(ly + 14.5))] = 1.0 + 3.84 / 2.0                # crossing at -14.5 + 0.5 * (3.84-1.92)/(20-1.92)
    chi[2, np.argmin(abs(ly + 17.0))] = 1.0 - 5.0                      # mass 2: an effect preferred at 1e-17
    chi[4, np.argmin(abs(ly + 18.0))] = 1.0 + 4.0                       # a speck
    lx, G, F, yn, C = ex.field(write_plane(tmp_path / "p" / "plane_m_Zd_eV_g_D.json", x, y, chi))
    assert yn == "g_D" and C.all()
    expect = -14.5 + 0.5 * (3.84 - 1.92) / (20.0 - 1.92)
    for m in (-10, -9, -6, -5):                                         # -8 has the preferred dip
        assert lowest_excluded(lx, G, F, m) == pytest.approx(expect, abs=2 * ex.STEP)
    assert lowest_excluded(lx, G, F, -8) == pytest.approx(-17.0 + 0.5 * 3.84 / 5.0, abs=2 * ex.STEP)   # not below
    assert lowest_excluded(lx, G, F, -6) > -17.0                        # the speck: not excluded


def test_thin_band_taken_out_region_kept():
    """A band 0.4 decade across is taken out of the field and returned as a ribbon; a region 10 decades tall stays."""
    lx = np.arange(-30.0, -9.9, 1.0)
    G = np.arange(-32.0, -2.0 + ex.STEP / 2, ex.STEP)
    excl = (G >= -12.0) | ((G >= -27.6) & (G <= -27.2))
    F = np.ma.masked_invalid(np.stack([ex.signed(G, excl)] * len(lx), axis=1))
    C = np.ones(F.shape, bool)
    Fs, ribbons = ex.thin_bands(lx, G, F, C)
    k = len(lx) // 2
    band = (G >= -27.6) & (G <= -27.2)
    assert np.all(Fs[band, k] < 0) and np.all(Fs[G >= -11.0, k] > 0)
    assert len(ribbons) == 1 and len(ribbons[0]) == len(lx)
    assert ribbons[0][:, 1].min() == pytest.approx(-27.6, abs=0.01) and ribbons[0][:, 2].max() == pytest.approx(-27.2, abs=0.01)


def test_merged_planes_share_one_field(tmp_path):
    """The ultralight and MeV planes join into one field on the ultralight couplings: one column at the mass both
    have (the MeV plane's), the MeV plane allowed below its smallest coupling."""
    xu, yu = 10.0 ** np.arange(-8, 5), 10.0 ** np.arange(-32.0, -1.5, 0.5)
    xm, ym = 10.0 ** np.arange(4.0, 9.01, 0.5), 10.0 ** np.arange(-10.0, -1.9, 0.2)
    chi_u = np.where(np.log10(yu)[None, :] >= -3.05, 30.0, 0.0) * np.ones((len(xu), 1))
    chi_m = np.where(np.log10(ym)[None, :] >= -2.65, 30.0, 0.0) * np.ones((len(xm), 1))
    pu = write_plane(tmp_path / "ul" / "plane_m_Zd_eV_g_D.json", xu, yu, chi_u)
    pm = write_plane(tmp_path / "mev" / "plane_m_Zd_eV_g_D.json", xm, ym, chi_m)
    lx, G, F = ex.merged(pu, pm)
    assert np.all(np.diff(lx) > 0) and np.sum(abs(lx - 4.0) < 1e-6) == 1
    assert F.shape == (len(G), len(lx)) and G[0] == pytest.approx(-32.0)
    assert lowest_excluded(lx, G, F, 4.0) == pytest.approx(-2.6 - 0.2 + 0.2 * 3.84 / 30.0, abs=2 * ex.STEP)
    assert lowest_excluded(lx, G, F, 0.0) == pytest.approx(-3.5 + 0.5 * 3.84 / 30.0, abs=2 * ex.STEP)
    assert np.all(F[np.ix_(G < -10.5, lx > 4.1)] < 0)                         # below the MeV scan: allowed


def test_refinement_above_the_grid_and_between_masses(tmp_path):
    """refine_grid.py's points: at grid masses above the grid's largest coupling (the field extends up to them) and at
    new masses between the grid's (partial columns from 1e-5 up)."""
    x = 10.0 ** np.arange(4.0, 8.01, 1.0)
    y = 10.0 ** np.arange(-10.0, -1.9, 0.2)
    chi = np.zeros((len(x), len(y)))                            # nothing excluded on the grid
    path = write_plane(tmp_path / "p" / "plane_m_Zd_eV_g_D.json", x, y, chi)
    rec = lambda m, g, c: json.dumps(dict(held={"m_Zd_eV": 10.0 ** m, "g_D": 10.0 ** g}, chi2=c)) + "\n"
    with open(tmp_path / "p" / "refine_grid.0.jsonl", "w") as f:
        for m in (4.0, 5.0, 6.0, 7.0, 8.0):
            for g in (-1.8, -1.6, -1.4, -1.2, -1.0):
                f.write(rec(m, g, 30.0 if g >= -1.4 else 0.0))   # excluded from 10^-1.4 up
    with open(tmp_path / "p" / "refine_grid_between.0.jsonl", "w") as f:
        for g in np.arange(-5.0, -0.99, 0.2):
            f.write(rec(5.5, g, 30.0 if g >= -3.0 else 0.0))   # a dip down to 1e-3 at 10^5.5 eV only
    lx, G, F, _, _ = ex.field(path)
    assert G[-1] == pytest.approx(-1.0, abs=ex.STEP)
    assert lowest_excluded(lx, G, F, 5.0) == pytest.approx(-1.6 + 0.2 * 3.84 / 30.0, abs=2 * ex.STEP)
    assert any(abs(lx - 5.5) < 1e-6)
    assert lowest_excluded(lx, G, F, 5.5) == pytest.approx(-3.2 + 0.2 * 3.84 / 30.0, abs=2 * ex.STEP)


def test_thermal_grid_corrections(tmp_path):
    """The kernels' grid fix (2026-10-21): a grid point re-evaluated with the fixed code (recheck, gap 0.5) takes the
    new chi^2; a refit (refine_recheck) takes over where lower; a refinement record is corrected through its file's
    name; a failed fit (stored chi^2 missing) changes only through a refit."""
    x = 10.0 ** np.arange(4.0, 8.01, 1.0)
    y = 10.0 ** np.arange(-6.0, -1.9, 0.5)
    chi = np.where(np.log10(y)[None, :] >= -3.0, 30.0, 0.0) * np.ones((len(x), 1))
    chi[3, 0] = np.nan                                           # a failed fit at the smallest coupling
    path = write_plane(tmp_path / "p" / "plane_m_Zd_eV_g_D.json", x, y, chi)
    held = lambda m, g: {"m_Zd_eV": 10.0 ** m, "g_D": 10.0 ** g}
    rc = lambda src, m, g, c, st, gap=0.5: json.dumps(dict(source=src, held=held(m, g), gap=gap, chi2=c,
                                                           stored=st)) + "\n"
    with open(tmp_path / "p" / "refine_grid.0.jsonl", "w") as f:
        f.write(json.dumps(dict(held=held(5.0, -1.5), chi2=30.0)) + "\n")
    with open(tmp_path / "p" / "recheck.0.jsonl", "w") as f:
        f.write(rc("grid", 5.0, -3.5, 10.0, 0.0))               # now excluded at 10^-3.5 (mass 1e5)
        f.write(rc("grid", 6.0, -3.5, 10.0, 0.0))
        f.write(rc("grid", 6.0, -3.5, 0.0, 0.0, gap=0.0))        # the old grid's control: not used
        f.write(rc("grid", 7.0, -6.0, 50.0, float("nan")))       # a failed fit's point: not without a refit
        f.write(rc("refine_grid.0.jsonl", 5.0, -1.5, 0.0, 30.0))  # a record no longer excluded
    with open(tmp_path / "p" / "refine_recheck.0.jsonl", "w") as f:
        f.write(json.dumps(dict(source="grid", held=held(6.0, -3.5), chi2=1.0)) + "\n")   # the refit, lower
    corr = ex.corrected(str(tmp_path / "p"), "m_Zd_eV", "g_D")
    assert corr[("grid", (5.0, -3.5))] == 10.0 and corr[("grid", (6.0, -3.5))] == 1.0
    assert ("grid", (7.0, -6.0)) not in corr and corr[("refine_grid.0.jsonl", (5.0, -1.5))] == 0.0
    lx, G, F, _, _ = ex.field(path)
    near = lambda a, b: a == pytest.approx(b, abs=2 * ex.STEP)
    assert near(lowest_excluded(lx, G, F, 4.0), -3.5 + 0.5 * 3.84 / 30.0)            # untouched
    assert near(lowest_excluded(lx, G, F, 5.0), -4.0 + 0.5 * 3.84 / 10.0)            # the re-evaluated point
    assert near(lowest_excluded(lx, G, F, 6.0), -3.5 + 0.5 * (3.84 - 1.0) / 29.0)    # the refit's, not the 10


def test_column_between_masses_reads_its_own_minimum(tmp_path):
    """A mass between the grid's sampled from 1e-5 up (refine_grid's new masses; 2026-10-22): Delta chi^2 from its own
    minimum (and the no-effect chi^2), not from its neighbours' -- where the data prefer an effect at a depth that
    changes with mass, a neighbour's deeper minimum made the whole column read as excluded."""
    x = 10.0 ** np.arange(4.0, 8.01, 1.0)
    y = 10.0 ** np.arange(-10.0, -1.9, 0.2)
    ly = np.log10(y)
    chi = np.full((len(x), len(y)), 10.0)                      # no effect: 10 everywhere ...
    chi[:, np.argmin(abs(ly + 4.0))] = 5.0                     # ... a preferred dip at 1e-4 (5.0), at every grid mass
    chi[:, ly >= -2.6] = 30.0                                  # excluded from 10^-2.6 up
    path = write_plane(tmp_path / "p" / "plane_m_Zd_eV_g_D.json", x, y, chi)
    rec = lambda m, g, c: json.dumps(dict(held={"m_Zd_eV": 10.0 ** m, "g_D": 10.0 ** g}, chi2=c)) + "\n"
    with open(tmp_path / "p" / "refine_grid_between.0.jsonl", "w") as f:
        for g in np.round(np.arange(-5.0, -1.99, 0.1), 6):    # at 10^5.5: a shallower dip (8.0); 30 from 10^-2.6 up
            f.write(rec(5.5, g, 30.0 if g >= -2.6 else (8.0 if abs(g + 4.0) < 0.05 else 10.0)))
    lx, G, F, _, _ = ex.field(path)
    # own minimum 8.0: 10 is 2.0 above it, allowed; excluded only from the 30s (interpolated between -2.7 and -2.6)
    assert lowest_excluded(lx, G, F, 5.5) == pytest.approx(-2.7 + 0.1 * (3.84 + 8.0 - 10.0) / 20.0, abs=2 * ex.STEP)
    # the grid columns: their own dips, 5 below the plateau -- excluded from just above the dip (upper limits only)
    assert lowest_excluded(lx, G, F, 5.0) == pytest.approx(-4.0 + 0.2 * 3.84 / 5.0, abs=2 * ex.STEP)
