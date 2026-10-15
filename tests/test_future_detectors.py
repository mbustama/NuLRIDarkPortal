"""
The future detectors (docs/dev/FUTURE_DETECTORS_PLAN.md): NuFlavorSingleSource's detector models, vendored unchanged
(IceCubeRates/future/nfss_forecast.py, nfss_detectors.py), and the diffuse response built on them
(IceCubeRates/future/diffuse.py).
"""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from IceCubeRates.future import nfss_detectors as D, nfss_forecast as F  # noqa: E402
from IceCubeRates.future.diffuse import DiffuseDetector  # noqa: E402


@pytest.fixture(scope="module")
def calibrated():
    return D.calibrate(verbose=False)


def test_calibration_reproduces_nuflavorsinglesource(calibrated):
    """Every calibration number equals NuFlavorSingleSource's own run of its commit 39088bd (final inputs), bit for bit."""
    ref = json.load(open(os.path.join(REPO, "tests", "data", "nfss_calibration_39088bd.json")))
    _, info = calibrated
    assert {k: float(info[k]) for k in ref} == ref


@pytest.mark.parametrize("name", ["IceCube", "ARCA_base", "GVD_base", "Gen2_new_base"])
def test_diffuse_response_matches_allsky_rates(calibrated, name):
    """Summed over directions, the per-direction response gives NuFlavorSingleSource's all-sky rate in every
    reconstructed-energy bin (its own allsky_rate with the bin's cuts, for HESE's diffuse flux), to its quadrature."""
    out, _ = calibrated
    P = out[name]
    dd = DiffuseDetector(name, P)
    det = F.Detector(name, P, build=False)
    for ch in dd.channels:
        ed = F.EDGES[ch]
        acc = P["trk_accept"] if ch == "trk" else P["shw_accept"]
        mine = np.einsum("ebz->b", sum(0.5 * F.astro_hese(F.E)[:, None, None] * dd.R[ch][k] for k in dd.R[ch])) / P["years"]
        ref = np.array([det.allsky_rate(ch, accept=acc, flux=F.astro_hese, ecut=10 ** lo, smear=True)
                        - det.allsky_rate(ch, accept=acc, flux=F.astro_hese, ecut=10 ** hi, smear=True)
                        for lo, hi in zip(ed[:-1], ed[1:])])
        ok = ref > 1e-6
        np.testing.assert_allclose(mine[ok], ref[ok], rtol=5e-5)
