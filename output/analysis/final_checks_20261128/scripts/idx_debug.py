"""Why does the objective return its wall value with the wider index ranges? Evaluate the pieces at the start values
of one final_idx config and print the exception (run inside a checkout, with its environment)."""
import sys, traceback
sys.path.insert(0, ".")
from analysis import config as acfg
cfg = acfg.load(sys.argv[1])
obj = cfg.objective()
vals = {p.name: p.start for p in cfg.parameters}
pl = cfg.planes[0]
vals[pl.x] = float(pl.x_grid[0]); vals[pl.y] = float(pl.y_grid[0])
print("values:", vals)
for name, f in (("spectrum_chi2", lambda: obj.spectrum_chi2(vals)), ("fractions", lambda: obj.model.fractions(vals))):
    try:
        print(name, "->", f())
    except Exception:
        print(name, "RAISED:"); traceback.print_exc(limit=8)
