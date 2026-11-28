"""Fetch the INSPIRE BibTeX and metadata of every entry of paper/references.bib (by texkey; else by eprint or DOI)."""
import json, re, sys, time, urllib.parse, urllib.request, os
BIB, OUT = sys.argv[1], sys.argv[2]
txt = open(BIB).read()
entries = re.split(r"\n(?=@)", txt)
def field(e, name):
    m = re.search(r"^\s*%s\s*=\s*(.*?),?\s*$" % name, e, re.M)
    if not m: return None
    v = m.group(1).strip().rstrip(",").strip()
    return v.strip('"').strip()
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bibliography check)"})
    for k in range(4):
        try:
            return urllib.request.urlopen(req, timeout=40).read().decode()
        except Exception as err:
            time.sleep(2 + 2 * k); last = err
    return "ERROR " + repr(last)
res = {}
for e in entries:
    m = re.match(r"@(\w+)\{([^,]+),", e.strip())
    if not m: continue
    typ, key = m.group(1), m.group(2).strip()
    q = None
    tries = [("texkeys", key)]
    if field(e, "eprint"): tries.append(("arxiv", field(e, "eprint").strip("{}")))
    if field(e, "doi"): tries.append(("doi", field(e, "doi").strip("{}")))
    rec = None
    for how, val in tries:
        url = "https://inspirehep.net/api/literature?q=" + urllib.parse.quote(f'{how}:"{val}"' if how != "arxiv" else f"arxiv:{val}") + "&size=2&fields=texkeys,titles,authors.full_name,arxiv_eprints,publication_info,dois,collaborations,document_type,earliest_date,control_number,report_numbers"
        j = get(url)
        try: d = json.loads(j)
        except Exception: continue
        if d["hits"]["total"] >= 1:
            rec = d["hits"]["hits"][0]["metadata"]; rec["_found_by"] = how; rec["_hits"] = d["hits"]["total"]; break
        time.sleep(0.35)
    bibtex = None
    if rec:
        bibtex = get(f"https://inspirehep.net/api/literature/{rec['control_number']}?format=bibtex")
    res[key] = dict(type=typ, local=e.strip(), inspire=rec, bibtex=bibtex)
    print(key, "->", (rec["_found_by"] + " " + str(rec.get("texkeys"))) if rec else "NOT FOUND", flush=True)
    time.sleep(0.35)
json.dump(res, open(OUT, "w"), indent=1)
