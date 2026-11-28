"""Field-by-field comparison of each local entry with INSPIRE's BibTeX; prints only differences."""
import json, re, sys
R = json.load(open(sys.argv[1]))
def parse(e):
    """{field: value} of one BibTeX entry (values without the outer quotes or braces)."""
    m = re.match(r"@(\w+)\{([^,]+),", e.strip()); out = {"_type": m.group(1).lower(), "_key": m.group(2).strip()}
    body = e.strip()[m.end():]
    i = 0
    while True:
        m2 = re.compile(r"\s*([A-Za-z]+)\s*=\s*").match(body, i)
        if not m2: break
        name = m2.group(1); j = m2.end()
        if body[j] == '"':
            k = j + 1; depth = 0
            while not (body[k] == '"' and depth == 0):
                depth += body[k] == "{"; depth -= body[k] == "}"; k += 1
            val = body[j + 1:k]; i = k + 1
        elif body[j] == "{":
            k = j; depth = 0
            while True:
                depth += body[k] == "{"; depth -= body[k] == "}"
                if depth == 0: break
                k += 1
            val = body[j + 1:k]; i = k + 1
        else:
            k = j
            while body[k] not in ",\n}": k += 1
            val = body[j:k]; i = k
        out[name] = re.sub(r"\s+", " ", val).strip()
        while i < len(body) and body[i] in ", \n": i += 1
    return out
def plain(t):
    return re.sub(r"[{}]", "", t)
n_same = 0
for key, r in R.items():
    L = parse(r["local"])
    if not r["inspire"] or not r["bibtex"] or r["bibtex"].startswith("ERROR"):
        print(f"\n## {key}: not on INSPIRE under this key, eprint or DOI ({L['_type']})"); continue
    I = parse(r["bibtex"]); diffs = []
    if I["_key"] != key: diffs.append(f"key: INSPIRE's first key is {I['_key']} (all: {r['inspire'].get('texkeys')})")
    if I["_type"] != L["_type"]: diffs.append(f"type: local @{L['_type']}, INSPIRE @{I['_type']}")
    for f in sorted((set(L) | set(I)) - {"_type", "_key"}):
        a, b = L.get(f), I.get(f)
        if a == b: continue
        if f == "title" and a and b:
            if plain(a).lower() == plain(b).lower():
                continue                      # only capitalization or braces differ: the sentence-case pass handles it
            diffs.append(f"title differs beyond case:\n      local  : {a}\n      INSPIRE: {b}")
        elif f == "number" and a and not b and a == L.get("year") and L.get("journal") in ("JHEP", "JCAP"):
            continue                          # his deliberate fix: number = year in JHEP and JCAP entries
        elif a is None: diffs.append(f"{f}: missing locally; INSPIRE has: {b}")
        elif b is None: diffs.append(f"{f}: only local: {a}")
        else: diffs.append(f"{f}:\n      local  : {a}\n      INSPIRE: {b}")
    if diffs:
        print(f"\n## {key}"); [print("   -", d) for d in diffs]
    else: n_same += 1
print(f"\n{n_same} entries identical to INSPIRE apart from title capitalization and the JHEP/JCAP number field")
