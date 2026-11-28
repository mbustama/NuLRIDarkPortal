"""Patch analysis/scans.py: the plane's worker pool recycles each worker after NULRI_WORKER_MAXTASKS tasks (chains or
repair points) when that variable is set; unset: unchanged. 2026-11-27: the forecast refit's workers keep about
0.65 GB more per hour (62-mass scans of the forecast fits), which a fresh process per chain returns."""
import sys
p = sys.argv[1]
s = open(p).read()
old = "            self.pool = ctx.Pool(self.workers, initializer=_init_worker, initargs=self.args)\n"
new = ("            # NULRI_WORKER_MAXTASKS=n (2026-11-27): a worker is replaced by a fresh process after n tasks (chains or\n"
       "            # repair points), which returns the memory that its fits keep; unset or 0: workers live to the end\n"
       "            recycle = int(os.environ.get(\"NULRI_WORKER_MAXTASKS\", \"0\")) or None\n"
       "            self.pool = ctx.Pool(self.workers, initializer=_init_worker, initargs=self.args,\n"
       "                                 maxtasksperchild=recycle)\n")
if "NULRI_WORKER_MAXTASKS" in s:
    print(p, "already patched")
else:
    assert s.count(old) == 1, "pool line not found exactly once"
    open(p, "w").write(s.replace(old, new))
    print(p, "patched")
