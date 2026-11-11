"""
The future telescopes' event likelihood on NuFlavorSingleSource's Plan A detector models (C1, 2026-11-11;
docs/dev/C1_FORECAST_PORT.md; analysis/future_events.py): its arithmetic on synthetic responses and a stub model (the
signal, the Glashow piece, the Asimov profile, Newton against L-BFGS-B, the exact gradient), and the model's per-site
Galactic bands.
"""
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "tests")]

from analysis import future_events as FE  # noqa: E402

SIN = np.linspace(-1.0, 1.0, 5)                 # 4 bands
LGT = np.linspace(3.0, 7.0, 9)                  # 8 true-energy bins
LGE_MODEL = np.linspace(2.5, 7.5, 101)


def _card(rng, energy_only, sin=SIN):
    nt, nr = len(LGT) - 1, 5
    bands = []
    for b in range(len(sin) - 1):
        Rb = {f: rng.uniform(0.5, 2.0, (nt, nr)) * 1e10 for f in FE.FLAVORS}
        Rg = {f: np.zeros((nt, nr)) for f in FE.FLAVORS}
        Rg["e"] = 0.2 * Rb["e"]
        RXb = {"eff": Rb, "esc": {f: 0.1 * rng.normal(size=(nt, nr)) * 1e10 for f in FE.FLAVORS},
               "pt": {f: Rb[f] * 0.01 for f in FE.FLAVORS}}
        RXg = {"eff": Rg, "esc": {f: np.zeros((nt, nr)) for f in FE.FLAVORS}, "pt": Rg}
        B = rng.uniform(1.0, 5.0, nr)
        BX = {"eff": 0.5 * B, "conv": 0.8 * B, "astro_norm": B, "pt": 0.0 * B}
        bands.append(dict(Rb=Rb, RXb=RXb, Rb_gr=Rg, RXb_gr=RXg, B_total=B, BX=BX,
                          kinds={"eff": "eff", "esc": "escale", "pt": "pointing", "conv": "background"}))
    return dict(lgt_edges=LGT, energy_only=energy_only, bands=bands)


def write_synthetic_cards(path, sin=SIN, seed=1):
    """Synthetic responses for every card (two topologies, the given declination bands) in path."""
    rng = np.random.default_rng(seed)
    pri = {"eff": dict(kind="gauss", centre=1.0, width=0.1, lo=-0.05, hi=0.05),       # tight: the far fit reaches them
           "esc": dict(kind="gauss", centre=1.0, width=0.1, lo=-0.02, hi=0.02),     # tight: profiles reach the bounds
           "pt": dict(kind="gauss", centre=1.0, width=0.1, lo=-0.5, hi=0.5),
           "conv": dict(kind="free", centre=1.0, width=np.nan, lo=-0.9, hi=2.0)}
    for card in FE.CARDS:
        d = dict(card=card, sin_edges=np.asarray(sin), topos={"trk": _card(rng, False, sin), "shw": _card(rng, True, sin)},
                 priors=pri)
        with open(os.path.join(str(path), f"{card}.pkl"), "wb") as fh:
            pickle.dump(d, fh)


@pytest.fixture
def blocks_dir(tmp_path, monkeypatch):
    write_synthetic_cards(tmp_path)
    monkeypatch.setattr(FE, "DATA_DIR", str(tmp_path))
    return tmp_path


class _Stub:
    """A model with the sky's bands: Phi = n (E / 100 TeV)^-g per band and flavor, nubar a fraction r of nu."""

    def __init__(self):
        self.E_eV = 10.0 ** LGE_MODEL * 1e9

    def grid(self):
        return self.E_eV, np.arcsin(SIN)

    def _f(self, v):
        n, g, r = v.get("n", 1e-18), v.get("g", 2.5), v.get("r", 0.5)
        E = self.E_eV / 1e9
        base = n * (E / 1e5) ** (-g)
        tilt = np.array([1.0, 1.2, 0.9, 1.1])[:, None, None] * np.array([1.0, 0.8, 1.3])[None, None, :]
        F = base[None, :, None] * tilt                               # (nb, N_E, 3), declination order
        return F / (1 + r), F * r / (1 + r), base, E

    def site_band_fluxes(self, v, lat):
        Fn, Fb, _, _ = self._f(v)
        return self.E_eV, Fn[::-1], Fb[::-1]

    def site_band_flux_jacobian(self, v, lat, names):
        Fn, Fb, base, E = self._f(v)
        n, g, r = v.get("n", 1e-18), v.get("g", 2.5), v.get("r", 0.5)
        d = {}
        for nm in names:
            if nm == "n":
                d[nm] = (Fn[::-1] / n, Fb[::-1] / n)
            elif nm == "g":
                L = -np.log(E / 1e5)[None, :, None]
                d[nm] = ((Fn * L)[::-1], (Fb * L)[::-1])
            elif nm == "r":
                F = Fn * (1 + r)
                d[nm] = ((-F / (1 + r) ** 2)[::-1], (F / (1 + r) ** 2)[::-1])
        return self.E_eV, Fn[::-1], Fb[::-1], d


def _integral(y, x, lo, hi):
    """The trapezoid rule on the grid points within [lo, hi] (the bin edges are grid points)."""
    m = (x >= lo - 1e-9) & (x <= hi + 1e-9)
    xs, ys = x[m], y[m]
    return float(np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs)))


def test_blocks_signal_and_glashow(blocks_dir):
    """Tracks one block per band, showers one block; the dropped pointing template and the diffuse flux's background
    templates are out; the signal is the response times the flux integrals, the Glashow piece twice the nubar_e
    share times its response."""
    L = FE.FutureEventLikelihood(_Stub(), {"config": "arca"})
    assert [(b["topo"], b["band"]) for b in L.blocks] == [("trk", 0), ("trk", 1), ("trk", 2), ("trk", 3), ("shw", None)]
    assert L.names == ["conv", "eff", "esc"]
    v = dict(n=2e-18, g=2.3, r=0.4)
    S = L.signals(v)
    d = FE.load_card("arca")
    m = _Stub()
    Fn, Fb, _, E = m._f(v)
    lg = np.log10(E)
    for k, blk in enumerate(L.blocks):
        rec = d["topos"][blk["topo"]]
        bands = range(4) if blk["band"] is None else [blk["band"]]
        ref = 0.0
        for b in bands:
            e = rec["bands"][b]
            for i, f in enumerate(FE.FLAVORS):
                phi = np.array([_integral((Fn + Fb)[b, :, i] * E * np.log(10), lg, lo, hi)
                                for lo, hi in zip(LGT[:-1], LGT[1:])])
                ref = ref + phi @ (e["Rb"][f] - e["Rb_gr"][f])
                if f == "e":
                    phig = np.array([_integral(2 * Fb[b, :, 0] * E * np.log(10), lg, lo, hi)
                                     for lo, hi in zip(LGT[:-1], LGT[1:])])
                    ref = ref + phig @ e["Rb_gr"]["e"]
        np.testing.assert_allclose(S[k][""], FE.YEARS * ref, rtol=5e-3)


def test_asimov_profile_and_newton(blocks_dir):
    """Asimov data at the truth give chi^2 = 0 there; elsewhere chi^2 > 0; Newton's profile equals L-BFGS-B's."""
    L = FE.FutureEventLikelihood(_Stub(), {"config": "All8"})
    truth = dict(n=1.5e-18, g=2.5, r=0.5)
    L.set_asimov(truth)
    assert L.chi2(truth) < 1e-8
    off = dict(truth, g=2.6)
    c = L.chi2(off)
    assert c > 1e-3
    S = L.signals(off)
    L._x0 = np.zeros(len(L.names))
    x_n, r_n = L.profile(S, L.data)
    L.n_newton = 0
    L._newton = lambda *a, **k: None
    L._x0 = np.zeros(len(L.names))
    x_l, r_l = L.profile(S, L.data)
    assert abs(r_n.fun - r_l.fun) < 1e-6 * max(1.0, r_n.fun)


def test_gradient_matches_finite_differences(blocks_dir):
    L = FE.FutureEventLikelihood(_Stub(), {"config": "Current"})
    L.set_asimov(dict(n=1.5e-18, g=2.5, r=0.5))
    v = dict(n=1.8e-18, g=2.45, r=0.6)
    c, g = L.chi2_and_grad(v, ["n", "g", "r"])
    assert abs(c - L.chi2(v)) < 1e-8 * max(1.0, c)
    for i, (nm, h) in enumerate([("n", 1e-21), ("g", 1e-5), ("r", 1e-5)]):
        fd = (L.chi2(dict(v, **{nm: v[nm] + h})) - L.chi2(dict(v, **{nm: v[nm] - h}))) / (2 * h)
        assert abs(fd - g[i]) <= 1e-4 * max(abs(fd), 1e-3 * abs(c) / abs(v[nm])), (nm, fd, g[i])


@pytest.mark.parametrize("case", ["dark portal", "secret interactions"])
def test_site_galactic_bands(case):
    """The Galactic flux per local cos-zenith band at a site (propagate_sky_basis(site_latitudes_deg)): at the South
    Pole the per-declination bands in cos-zenith order; at every latitude, sky averages and day averages agree --
    each direction spends the whole day somewhere; a mid-latitude site and the North Pole differ from the South
    Pole's."""
    import dark_portal_solver as dps
    from test_event_likelihood import DP, SI, _sky_cfg
    over = {"secret_interactions": SI} if case == "secret interactions" else {"dark_portal": DP}
    E, params = dps.build_solver_params(_sky_cfg(0.3, 0.7, "3d", **over))
    params["show_progress"] = False
    out = dps.propagate_sky_basis(E, params, site_latitudes_deg=(-90.0, 36.27, 90.0))
    edges = out["site_cosz_edges"]
    for kind in ("nu", "nubar"):
        pole = out["site_galactic_bands_" + kind][-90.0]
        np.testing.assert_allclose(pole, out["galactic_bands_" + kind][:, ::-1], rtol=1e-12, atol=0)
        north = out["site_galactic_bands_" + kind][90.0]           # sees the northern sky above, the Galactic Centre below
        assert np.max(np.abs(north / pole - 1)) > 1e-3
        arca = out["site_galactic_bands_" + kind][36.27]
        assert np.max(np.abs(arca / pole - 1)) > 1e-3
    W = dps.site_zenith_weights(np.radians(np.array([-60.0, 0.0, 45.0])), 36.27, edges)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    # ARCA (36.27 N): dec = -60 never rises above cos Z = sin(36.27 - 90 + 60) ... it stays below the horizon
    assert W[0, edges[:-1] >= 0.0].sum() == 0.0


