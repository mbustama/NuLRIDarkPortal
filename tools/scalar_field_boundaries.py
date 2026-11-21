"""
scalar_field_boundaries.py -- for the paper's secret-interaction figure (2026-11-21): per scalar plane, the boundary of
the region where the relics' own scalar field is not modelled (his decision 2026-10-25: shaded, no claim there), at the
ultralight planes' masses (1e-30 to 1e3 eV, a decade apart), from the planes' own configurations
(analysis.exclusion.not_modelled_boundary: the smallest log10 g with not_modelled_measure >= 1, to 0.05 decade).

  python tools/scalar_field_boundaries.py <out.json> <config.json> [...]   (the configurations named si_scalar_*_ul__b*.json)
"""
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
from analysis import exclusion as ex  # noqa: E402

LX = np.arange(-30.0, 3.01, 1.0)


def main(out, cfgs):
    res = {"_comment": __doc__.strip().splitlines()[0], "log10_m": LX.tolist()}
    for c in cfgs:
        model = re.sub(r"_ul__b\d+\.json$", "", os.path.basename(c))
        res[model] = ex.not_modelled_boundary(c, LX)
        print(model, sum(v is not None for v in res[model]), "masses with a not-modelled region")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
