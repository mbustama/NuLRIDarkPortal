"""
numba's on-disk cache with several processes compiling at once (src/dp_core.py, 2026-10-18): a cache entry's data
file is named by its key, so two processes adding different signatures of one function at the same time can no
longer leave the index pointing one signature at the other's machine code (on Tycho, 16 processes compiling 16
signatures into one fresh cache broke 46 of 96 entries in six rounds, some returning wrong numbers).
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import dp_core  # noqa: E402,F401  (installs the naming)
from numba.core.caching import IndexDataCacheFile  # noqa: E402


def test_concurrent_entries_never_share_a_data_file(tmp_path):
    """Both writers read the index before either writes it. numba's own naming gives both entries data file 1, and
    whichever data write lands last serves both signatures; named by the key, each entry keeps its own file, and
    whichever index write lands last points only at its own entry's code."""
    f = IndexDataCacheFile(str(tmp_path), "f-1.py312", ("stamp", 1))
    stale = f._load_index()
    f._load_index = lambda: dict(stale)
    f.save(("sig A",), "code for sig A")
    f.save(("sig B",), "code for sig B")
    assert len([n for n in os.listdir(tmp_path) if n.endswith(".nbc")]) == 2
    g = IndexDataCacheFile(str(tmp_path), "f-1.py312", ("stamp", 1))
    index = g._load_index()
    assert index
    for key, name in index.items():
        assert g._load_data(name) == "code for " + key[0]


def test_entries_accumulate_and_reload(tmp_path):
    """One process saving signatures in turn: every entry is kept, each loads its own data, and saving a key again
    reuses its file."""
    f = IndexDataCacheFile(str(tmp_path), "f-1.py312", ("stamp", 1))
    for s in ("a", "b", "c"):
        f.save((s,), "code " + s)
    f.save(("a",), "code a")
    index = f._load_index()
    assert sorted(k[0] for k in index) == ["a", "b", "c"]
    assert len(set(index.values())) == 3
    assert all(f._load_data(n) == "code " + k[0] for k, n in index.items())


def test_no_avx512_in_numba_code():
    """dp_core switches AVX-512 off in numba's code generation (numba 0.60's LLVM 14 tunes Zen 4 as znver3 and
    miscompiled the Earth's Yukawa density with AVX-512 there): no AVX-512 feature is enabled in its target machine."""
    from numba.core.registry import cpu_target
    features = cpu_target.target_context.codegen()._tm_features
    assert "+avx512" not in features, features
