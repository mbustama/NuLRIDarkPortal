"""30-digit mpmath check of the largest gap on nu_3 (s channel, case A). The pair check (relic nu_2, case B, VV pairs at
2.19e14 eV: ours +5.7e-9, hers -1.4e-4 from the reference) was run with the same function, col="VV_pairs".
  python mp_check.py "../../rates all 27.11.2026" ours_at_her_energies.npz"""
# independent 30-digit check of the two places where Maria's rates and ours differ by more than 1e-8
import sys, os, numpy as np, importlib.util, mpmath as mp
her, npz = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("br", os.path.expanduser("~/Research/NuLRIDarkPortal/docs/si_thermal_check/benchmark_rates.py"))
br = importlib.util.module_from_spec(spec); sys.modules["br"] = br; spec.loader.exec_module(br)
core, si = br.load_code(os.path.expanduser("~/Research/NuLRIDarkPortal"))
z = np.load(npz)
mp.mp.dps = 30
T, M = mp.mpf(br.T_NU0), mp.mpf(br.M_MED); z3 = mp.zeta(3)
n = 3 * z3 * T ** 3 / (4 * mp.pi ** 2)
def rho(eps, m):
    X = max(eps, m * m / eps); pX = (X * X - m * m) / (2 * X)
    f = lambda p: eps * p / (mp.e ** (p / T) + 1) / mp.sqrt(p * p + m * m)
    return mp.quad(f, sorted(set([pX] + [v for v in (pX + T, pX + 4 * T, pX + 15 * T) ] + [pX + 60 * T]))) / (3 * z3 * T ** 3)
def check(rel, key, col, Etarget, label):
    ch = next(c for c in br.build_checks(core, si, rel)[1] if c["key"] == key)
    m = mp.mpf(ch["m"]); E_arr = z[f"E_relic{rel + 1}"]; i = int(np.argmin(np.abs(E_arr - Etarget))); E = mp.mpf(E_arr[i])
    cols = list(z[f"{key}_columns"]); j = cols.index(col)
    ours = z[f"{key}_relic{rel + 1}"][i, j]
    f = ch["terms"][0][1]
    g = lambda eps: rho(eps, m) * f(float(2 * E * eps))[[c for c in range(len(cols))].index(j)]
    eR = M * M / (2 * E); gam = mp.mpf(ch["Gam"]) / M
    pm = 60 * T; Epm = mp.sqrt(m * m + pm * pm); lo, hi = m * m / (Epm + pm), Epm + pm
    pts = [lo, m, hi] + [m + s * k * T for k in (1, 4, 16) for s in (-1, 1)]
    if col.endswith("pairs"):
        th = 4 * eR; pts += [th] + [th * (1 + mp.mpf(10) ** -k) for k in range(1, 9)]
    else:
        pts += [eR * (1 + s * gam * mp.mpf(10) ** k) for k in range(0, 7) for s in (-1, 1)] + [eR]
    pts = sorted(set(p for p in pts if lo <= p <= hi))
    if col.endswith("pairs"):
        pts = [p for p in pts if p >= 4 * eR]
    ref = n * mp.quad(g, pts) / mp.mpf(br.HBAR_EV_S)
    H = np.genfromtxt(f"{her}/numerical/Gamma3{rel + 1}_{ {'A':'A_scalar_Majorana','B':'B_vector_Majorana'}[key]}.csv", delimiter=",", names=True)
    hcol = {"s_channel": H["s_pole_per_s"] + H["s_off_per_s"], "VV_pairs": H["pair_per_s"]}[col]
    print(f"{label} at E = {float(E):.4g} eV: ours/ref - 1 = {ours / float(ref) - 1:+.1e};  hers/ref - 1 = {hcol[i] / float(ref) - 1:+.1e}")
check(2, "A", "s_channel", float(np.load(npz)["E_relic3"][np.argmax(np.abs((np.genfromtxt(f"{her}/numerical/Gamma33_A_scalar_Majorana.csv", delimiter=",", names=True)["s_pole_per_s"] + np.genfromtxt(f"{her}/numerical/Gamma33_A_scalar_Majorana.csv", delimiter=",", names=True)["s_off_per_s"]) / np.load(npz)["A_relic3"][:, list(np.load(npz)["A_columns"]).index("s_channel")] - 1))]), "relic nu_3, scalar Majorana, s channel")

