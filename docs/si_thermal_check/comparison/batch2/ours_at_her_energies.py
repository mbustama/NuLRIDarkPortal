"""Our rates of nu_3 on the relics nu_2 and nu_3 at Maria Petropavlova's energies (second batch), with
../../benchmark_rates.py: writes ours_at_her_energies.npz (8 cases x 501 energies, about 90 s on 4 processes).
  python ours_at_her_energies.py "../../rates all 27.11.2026" OUTDIR"""
# our rates of nu_3 on relics nu_2 and nu_3 at Maria's energies, with benchmark_rates.py's machinery
import sys, os, time, numpy as np, importlib.util
her_dir, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("br", os.path.expanduser("~/Research/NuLRIDarkPortal/docs/si_thermal_check/benchmark_rates.py"))
br = importlib.util.module_from_spec(spec); sys.modules["br"] = br; spec.loader.exec_module(br)
core, si = br.load_code(os.path.expanduser("~/Research/NuLRIDarkPortal"))
x, w = np.polynomial.legendre.leggauss(48); x2, w2 = np.polynomial.legendre.leggauss(96)
checks = {rel: br.build_checks(core, si, rel)[1] for rel in (1, 2)}
import multiprocessing as mp
res = {}
t0 = time.time()
for rel, tag in ((1, "Gamma32"), (2, "Gamma33")):
    E = np.genfromtxt(f"{her_dir}/numerical/{tag}_A_scalar_Majorana.csv", delimiter=",", names=True)["E_eV"]
    for f in ("B_vector_Majorana", "C_scalar_Dirac", "D_vector_Dirac"):
        assert np.array_equal(np.genfromtxt(f"{her_dir}/numerical/{tag}_{f}.csv", delimiter=",", names=True)["E_eV"], E)
    br._RELIC_STATE.update(si=si, checks=checks, E=E, x=x, w=w, x2=x2, w2=w2)
    with mp.get_context("fork").Pool(4) as pool:
        done = pool.map(br._relic_task, [(k, rel) for k in "ABCD"], chunksize=1)
    res[f"E_relic{rel + 1}"] = E
    for k, r, R, conv in done:
        res[f"{k}_relic{r + 1}"] = R
        print(f"relic nu_{r + 1}, case {k}: node-doubling change {conv:.1e}")
print(f"8 cases x 501 energies in {time.time() - t0:.1f} s on 4 processes")
np.savez(f"{out}/ours_at_her_energies.npz", **res, **{f"{c['key']}_columns": np.array(c["cols"]) for c in checks[1]})
