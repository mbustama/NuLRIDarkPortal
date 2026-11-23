"""event_rates.py -- expected MESE (cascades, double cascades) and DR2 events for one example of Fig. 7 and for no new
interaction, at the same sources: the set-2 best fit without a new interaction (its stored fit.json values), with the
event nuisances profiled at that fit and held for the example. Run inside a campaign checkout (its own code).
  python event_rates.py CHECKOUT {le,dp,si} OUT.npz [--only-sm] [--cores N]"""
import copy, json, os, sys, time
import numpy as np
ck, which, out = sys.argv[1], sys.argv[2], sys.argv[3]
only_sm = "--only-sm" in sys.argv
cores = int(sys.argv[sys.argv.index("--cores") + 1]) if "--cores" in sys.argv else 8
sys.path[:0] = [ck, os.path.join(ck, "src")]
os.chdir(ck)
from analysis import config as config_mod

# config (now_events block: the solver of the set-2 fits), example, reference coupling, set-2 no-effect fit (fit.json)
CASES = {
 "le": ("analysis/configs/campaign_final_sm/now_events/le_lmu_ul__b0.json", dict(m_Zd_eV=1e-20, g_D=1e-21),
        dict(m_Zd_eV=1e-20, g_D=1e-32),
        "/lustre/astro/mbustama/external/nulri_set1_ml/output/analysis/campaign_ml/now_mese_dr2/le_lmu_mev/data/fit.json"),
 "dp": ("analysis/configs/campaign_final/now_events/dp_tau_ul__b0.json", dict(m_Zd_eV=1e-3, g_D_trace=5.62e-7),
        dict(m_Zd_eV=1e-3, g_D_trace=1e-20),
        "/lustre/astro/mbustama/external/nulri_set1_ml/output/analysis/campaign_ml/now_mese_dr2/dp_tau_mev/data/fit.json"),
 "si": ("analysis/configs/campaign_si/now_events/si_scalar_majorana_tau.json", dict(m_phi_eV=1e7, g_phi=0.18),
        dict(m_phi_eV=1e7, g_phi=1e-8),
        "/lustre/astro/mbustama/external/nulri_refit_set/output/analysis/campaign_si/now_mese_dr2/si_scalar_majorana_tau/b0/data/fit.json"),
}
path, ex_np, ref_np, fitp = CASES[which]
fit = json.load(open(fitp))
src = {k: v for k, v in fit["values"].items() if k in ("m_lightest_eV", "norm_gal", "norm_astro", "p_index", "p_index_low",
                                                       "nubar_to_nu_ratio")}
raw = json.load(open(path))
raw["data"] = {"events": {"mese": {}, "dr2": {}}, "flavor_band_eV": raw["data"]["flavor_band_eV"]}   # set 2
raw["parameters"] = dict(raw["parameters"], nubar_to_nu_ratio={"start": 1.0, "bounds": [0.5, 1.0], "space": "linear"})
raw["solver"]["execution"] = dict(raw["solver"].get("execution", {}), n_cores=cores, verbose=0)
cfg = config_mod.from_dict(raw, os.path.abspath(path))
t0 = time.time()
obj = cfg.objective()
ev = obj.events
print(f"setup {time.time() - t0:.1f} s; names {ev.names}", flush=True)
sm = dict(src, **ref_np)
t0 = time.time()
chi2_sm = obj.chi2(sm)
t_sm = time.time() - t0
print(f"chi2 at the no-effect fit {chi2_sm:.4f} (stored {fit['chi2']:.4f}) in {t_sm:.1f} s", flush=True)
x = np.array([ev.last["nuisances"][n] for n in ev.names])
res = dict(which=which, chi2_sm=chi2_sm, chi2_sm_stored=fit["chi2"], seconds_sm=t_sm, names=np.array(ev.names), x=x,
           src=json.dumps(src), ex=json.dumps(ex_np), ref=json.dumps(ref_np))
from IceCubeRates.starting_events import mese as mese_mod
for t in ev.mese.topologies:
    res[f"mese_{t}_edges"] = mese_mod.histograms(t)["edges"]
res["mese_topologies"] = np.array(ev.mese.topologies)
res["mese_data"], res["mese_conv"], res["mese_muon"] = ev.mese.data, ev.mese.conv, ev.mese.muon
for c in ev.dr2.configs:
    res[f"dr2_{c}_select"] = ev.dr2.select[c]
    res[f"dr2_{c}_data"] = ev.dr2.data[c]
from IceCubeRates.northern_muons.dr2 import DR2Tracks
res["dr2_configs"] = np.array(ev.dr2.configs)


def evaluate(values, tag):
    A = ev.astro(values)
    e = ev.exact(x, A)
    res[f"{tag}_mese_astro"], res[f"{tag}_mese_mu"], res[f"{tag}_mese_dev"] = A["mese"], e["mese_mu"], e["mese"]
    for c in ev.dr2.configs:
        res[f"{tag}_dr2_{c}_astro"], res[f"{tag}_dr2_{c}_mu"] = A["dr2"][c], e["dr2_mu"][c]
    res[f"{tag}_dr2_dev"] = e["dr2"]


evaluate(sm, "sm")
if not only_sm:
    exv = dict(src, **ex_np)
    t0 = time.time()
    evaluate(exv, "ex")
    res["seconds_ex"] = time.time() - t0
    chi2_ex_prof = ev.chi2(exv)                    # the events' term with the nuisances re-profiled, sources held
    res["chi2_events_ex_profiled"] = chi2_ex_prof
    print(f"example: {time.time() - t0:.1f} s; events term, nuisances re-profiled: {chi2_ex_prof:.3f}", flush=True)
np.savez(out, **res)
print("wrote", out)
