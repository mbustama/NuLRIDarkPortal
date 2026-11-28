"""Her second batch against ours: totals and channels on nu_2, nu_3; her re-sent nu_1 rates and new analytic
formulas against ../../our_rates; writes ratios2.npz for plot_cmp2.py.
  python compare2.py "../../rates all 27.11.2026" ours_at_her_energies.npz ../../our_rates OUTDIR"""
import sys, numpy as np
her, ours_npz, ours_csv, out = sys.argv[1:5]
z = np.load(ours_npz)
names = {"A": "A_scalar_Majorana", "B": "B_vector_Majorana", "C": "C_scalar_Dirac", "D": "D_vector_Dirac"}
def hload(f):
    return np.genfromtxt(f, delimiter=",", names=True)
def channels(k, O, cols):
    c = list(cols); g = lambda n: O[:, c.index(n)]
    if k in "AC":
        d = {"s": g("s_channel"), "t,u": g("t_and_u"), "s x (t,u)": g("s_x_t_u_interference"), "t x u": g("t_x_u_interference"), "pairs": g("phi_phi_pairs")}
    else:
        d = {"s": g("s_channel"), "t,u": g("t_channel_nu_nubar") + g("t_and_u_nu_nu"), "s x (t,u)": g("s_x_t_interference"), "t x u": g("t_x_u_interference"), "pairs": g("VV_pairs")}
    nch = 5 if k == "C" else len(c)
    d["total"] = O[:, :nch].sum(axis=1)
    if k == "C":
        d["active"], d["sterile"] = g("active_produced"), g("sterile_produced")
    return d
def herch(H, k):
    d = {"s": H["s_pole_per_s"] + H["s_off_per_s"], "t,u": H["t_u_no_interf_per_s"], "s x (t,u)": H["s_t_interf_per_s"],
         "t x u": H["t_u_interf_per_s"], "pairs": H["pair_per_s"], "total": H["total_per_s"]}
    if k == "C":
        d["active"], d["sterile"] = H["active_per_s"] + 2 * H["pair_per_s"], H["sterile_per_s"] + 2 * H["pair_per_s"]
    return d
save = {}
for rel in (2, 3):
    print(f"=== relic nu_{rel}")
    for k in "ABCD":
        O = z[f"{k}_relic{rel}"]; E = z[f"E_relic{rel}"]
        a = channels(k, O, z[f"{k}_columns"]); H = hload(f"{her}/numerical/Gamma3{rel}_{names[k]}.csv"); b = herch(H, k)
        assert np.array_equal(H["E_eV"], E)
        tot = a["total"]; rt = b["total"] / tot - 1; i = np.argmax(np.abs(rt))
        line = f"  {k}: total max |her/ours-1| {abs(rt[i]):.1e} at {E[i]:.3g} eV;"
        for n in [n for n in a if n != "total"]:
            big = np.abs(a[n]) > 1e-3 * tot
            line += f" {n} {np.max(np.abs(b[n][big] / a[n][big] - 1)) if big.any() else float('nan'):.0e}"
        print(line)
        save[f"{k}_{rel}"] = np.vstack([E, rt])
    B, Dd = (z[f"{k}_relic{rel}"] for k in "BD"); E = z[f"E_relic{rel}"]
    tb, td = channels("B", B, z["B_columns"])["total"], channels("D", Dd, z["D_columns"])["total"]
    hb, hd = (hload(f"{her}/numerical/Gamma3{rel}_{names[k]}.csv")["total_per_s"] for k in "BD")
    j = np.argmax(tb)
    print(f"  vector Dirac/Majorana at the dip ({E[j]:.3g} eV): ours {td[j]/tb[j]:.4f}, hers {hd[j]/hb[j]:.4f}")
print("=== relic nu_1: her re-sent numerical vs our CSV, and her new analytic")
for k in "ABCD":
    O = np.loadtxt(f"{ours_csv}/{names[k].replace('_Majorana','_majorana').replace('_Dirac','_dirac').replace('A_scalar','A_scalar').lower() if False else ours_csv + '/' + {'A':'A_scalar_majorana','B':'B_vector_majorana','C':'C_scalar_dirac','D':'D_vector_dirac'}[k]}.csv".split('/', 1)[-1] if False else ours_csv + '/' + {'A':'A_scalar_majorana','B':'B_vector_majorana','C':'C_scalar_dirac','D':'D_vector_dirac'}[k] + '.csv', delimiter=",")
    H = hload(f"{her}/numerical/Gamma31_{names[k]}.csv"); A = hload(f"{her}/analytical UR/Gamma31_analytic_{names[k]}.csv")
    rn = H["total_per_s"] / O[:, 1] - 1; ra = A["total_per_s"] / O[:, 1] - 1; ia = np.argmax(np.abs(ra))
    print(f"  {k}: numerical max {np.max(np.abs(rn)):.1e} (below 10 PeV {np.max(np.abs(rn[O[:,0] <= 1e16])):.1e}); "
          f"analytic max {ra[ia]:+.1e} at {O[ia,0]:.3g} eV")
    save[f"{k}_1"] = np.vstack([O[:, 0], rn]); save[f"{k}_1a"] = np.vstack([O[:, 0], ra])
np.savez(f"{out}/ratios2.npz", **save)
