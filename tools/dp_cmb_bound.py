"""
dp_cmb_bound.py -- for the paper's limit figures (2026-11-21): the CMB bound on the neutrino lifetime as a bound on
the coupling of the dark portal or of L_e - L_alpha (analysis.existing_bounds.cmb_lifetime: each heavier state's rate into lighter ones from
our decay rates, against tau/m from the CMB; the weakest over the lightest masses scanned), per texture, at the
ultralight masses (1e-30 to 10 eV, a quarter decade apart). Excluded above the curve; NaN where no decay is open.
The configurations only set the model (their data sections are irrelevant here).

  python tools/dp_cmb_bound.py <out.json> <config> [...]   (ultralight plane configurations, e.g. dp_e_ul__b0.json)
"""
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "src"), REPO]
from analysis import existing_bounds as eb  # noqa: E402

LX = np.arange(-30.0, 1.001, 0.25)


def main(out, cfgs):
    res = {"_comment": __doc__.strip().splitlines()[0], "log10_m": LX.tolist()}
    for c in cfgs:
        model = re.sub(r"_ul__b\d+\.json$", "", os.path.basename(c))
        _, g = eb.cmb_lifetime(c, 10.0 ** LX)
        res[model] = [None if not np.isfinite(v) else float(np.log10(v)) for v in g]
        print(model, "open up to log10 M =", max((x for x, v in zip(LX, res[model]) if v is not None), default=None))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
