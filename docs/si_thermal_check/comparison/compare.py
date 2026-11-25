"""compare.py -- Maria Petropavlova's rates for checks A-D (si_checks/) against ours (../our_rates/):
the largest relative differences, total and channel by channel, and the ratios as .npy for plot_cmp.py.
  python compare.py ../../../si_checks ../our_rates OUTDIR"""
import sys, numpy as np, json
her_dir, ours_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
import os
os.makedirs(out, exist_ok=True)
def load(path):
    with open(path) as fh:
        lines = [l for l in fh]
    if lines[0].startswith("#"):
        names = [l for l in lines if l.startswith("#")][-1][1:].strip().split(",")
        data = np.loadtxt(path, delimiter=",")
    else:
        names = lines[0].strip().split(",")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {n: data[:, i] for i, n in enumerate(names)}
cases = {"A": ("A_scalar_majorana", "Gamma31_A_scalar_Majorana", "Gamma31_analytic_A_scalar_Majorana"),
         "B": ("B_vector_majorana", "Gamma31_B_vector_Majorana", "Gamma31_analytic_B_vector_Majorana"),
         "C": ("C_scalar_dirac", "Gamma31_C_scalar_Dirac", "Gamma31_analytic_C_scalar_Dirac"),
         "D": ("D_vector_dirac", "Gamma31_D_vector_Dirac", "Gamma31_analytic_D_vector_Dirac")}
res = {}
for k, (o, hn, ha) in cases.items():
    O = load(f"{ours_dir}/{o}.csv"); H = load(f"{her_dir}/numerical rates/{hn}.csv"); A = load(f"{her_dir}/analytical rates/{ha}.csv")
    dE = max(np.max(np.abs(H["E_eV"] / O["E_eV"] - 1)), np.max(np.abs(A["E_eV"] / O["E_eV"] - 1)))
    tot = O["total"]
    pairs = {"total": (O["total"], H["total_per_s"])}
    if k in "AC":
        pairs.update({"s channel": (O["s_channel"], H["s_pole_per_s"] + H["s_off_per_s"]),
                      "t and u": (O["t_and_u"], H["t_u_no_interf_per_s"]),
                      "s x (t,u) interference": (O["s_x_t_u_interference"], H["s_t_interf_per_s"]),
                      "t x u interference": (O["t_x_u_interference"], H["t_u_interf_per_s"]),
                      "pairs": (O["phi_phi_pairs"], H["pair_per_s"])})
    else:
        pairs.update({"s channel": (O["s_channel"], H["s_pole_per_s"] + H["s_off_per_s"]),
                      "t and u": (O["t_channel_nu_nubar"] + O["t_and_u_nu_nu"], H["t_u_no_interf_per_s"]),
                      "s x t interference": (O["s_x_t_interference"], H["s_t_interf_per_s"]),
                      "t x u interference": (O["t_x_u_interference"], H["t_u_interf_per_s"]),
                      "pairs": (O["VV_pairs"], H["pair_per_s"])})
    if k == "C":
        pairs.update({"active produced": (O["active_produced"], H["active_per_s"]),
                      "sterile produced": (O["sterile_produced"], H["sterile_per_s"])})
    r = {"max |E_hers/E_ours - 1|": float(dE)}
    for name, (a, b) in pairs.items():
        big = np.abs(a) > 1e-3 * tot
        rel = np.abs(b / np.where(a == 0, 1, a) - 1)
        r[name] = {"max rel (where >1e-3 of total)": float(rel[big].max()) if big.any() else None,
                   "max |diff| / total": float(np.max(np.abs(b - a) / tot))}
    ra = A["total_per_s"] / tot - 1
    i = int(np.argmax(np.abs(ra)))
    r["analytic total / ours - 1: max"] = [float(ra[i]), float(O["E_eV"][i])]
    r["her numerical total / her file 2 numerical_total - 1"] = float(np.max(np.abs(A["numerical_total_per_s"] / H["total_per_s"] - 1)))
    res[k] = r
    np.save(f"{out}/{k}_ratios.npy", np.vstack([O["E_eV"], H["total_per_s"] / tot - 1, ra]))
print(json.dumps(res, indent=1))
