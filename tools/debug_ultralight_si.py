"""Where dp_si's route fails at ultralight masses (2026-10-24, the validation map's NaN and thermal-table errors): for
a grid of mediator masses and couplings, the routed run and, below 1 keV, the forced dp_si run; finite or not, and the
error with its cause. Arguments: family. Run on Tycho."""
import os
import sys
import traceback

sys.path[:0] = [os.path.join(os.path.dirname(__file__)), os.path.join(os.path.dirname(__file__), "..", "src"),
                os.path.join(os.path.dirname(__file__), "..")]
import numpy as np  # noqa: E402
import validate_transport as vt  # noqa: E402

fam = sys.argv[1]
cfg, m = vt._model(fam)
yn = [q.name for q in cfg.parameters][1]
for lm in ([int(sys.argv[2])] if len(sys.argv) > 2 else (-30, -27, -24, -21, -18, -15, -12, -9, -6, -4, -2, 0, 2)):
    for lg in (-30, -20, -12, -8, -6, -4, -3, -2, -1, 0):
        v = {q.name: q.start for q in cfg.parameters}
        v.update({"m_Zd_eV": 10.0 ** lm, yn: 10.0 ** lg})
        try:
            base = m._run_cfg(m._overrides(v))
        except Exception:
            continue
        for mode in ("auto", "si"):
            try:
                r, p = vt._flux(m, base, dict(cosmo_transport=mode))
                F = r["F"]
                print(f"{fam} m 1e{lm} g 1e{lg} {mode}: mode {r['mode']}, finite {bool(np.all(np.isfinite(F)))}", flush=True)
            except Exception as e:
                c = e.__context__ or e.__cause__
                tb = traceback.extract_tb(e.__traceback__)[-1]
                print(f"{fam} m 1e{lm} g 1e{lg} {mode}: ERROR {repr(e)[:90]} (cause {repr(c)[:60]}) at "
                      f"{os.path.basename(tb.filename)}:{tb.lineno}", flush=True)
