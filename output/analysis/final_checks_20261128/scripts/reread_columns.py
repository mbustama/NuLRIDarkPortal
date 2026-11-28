"""After build_limits_20261127.sh: the "columns" of every limits.json of the folder re-read with the repository's
current reading rule (tools/export_limits.py on each checkout's planes), every other field kept. Run from the repo root:
  python reread_columns.py output/analysis/limits_20261127 <scratch dir>"""
import glob, json, os, shutil, subprocess, sys
A, R = sys.argv[1], sys.argv[2]
shutil.rmtree(R, ignore_errors=True)
changed = []
for co in sorted(d for d in os.listdir(A) if os.path.isdir(os.path.join(A, d))):
    log = subprocess.run([sys.executable, "tools/export_limits.py", os.path.join(A, co), "--out", os.path.join(R, co)],
                         capture_output=True, text=True)
    assert log.returncode == 0 and "FAILED" not in log.stdout, (co, log.stdout[-400:], log.stderr[-400:])
n = 0
for lf in sorted(glob.glob(A + "/**/limits.json", recursive=True)):
    rel = os.path.relpath(lf, A)
    base, new = json.load(open(lf)), json.load(open(os.path.join(R, rel)))
    assert [c["log10_m"] for c in base["columns"]] == [c["log10_m"] for c in new["columns"]], rel
    for a, b in zip(base["columns"], new["columns"]):
        if a.get("excluded_log10_g") != b.get("excluded_log10_g"):
            changed.append((os.path.dirname(rel), a["log10_m"]))
    base["columns"] = new["columns"]
    json.dump(base, open(lf, "w"), indent=1)
    n += 1
print(n, "limits.json re-read;", len(changed), "columns differ from the files as built:", changed)
