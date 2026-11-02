# Transport review 2, 2026-10-24 (today's uncommitted diff first)

Scope: `git diff 5a36f8b -- src/` (dp_decay exact kinematics and windows, dp_si decay channel, midpoint rates and
exact step times in the dark-portal phase, kappa in the loss's beam factor, dp_galactic_path, the clip flag), then
dp_si's bin matrix. Findings F1-F12 of TRANSPORT_REVIEW_2026-10-24.md are not repeated. No file under src/, tests/,
analysis/, notes/ or docs/ was changed. The checks ran on the laptop in seconds each; their scripts are in the session
scratchpad (spec_check.py, fit_check.py, share.py, si_dec_check.py, phi_check.py, stiff2.py, tiny_m.py,
loss_mag.py). I did not run the solver or the test suite.

## Verdict

I found no bug in today's diff that changes the planes. One older numerical weakness in dp_si's bin exponential is
real, but where it acts the relic mass-change loss already removes the flux (L1). The cause of the si-mode NaN at
m_Zd 1e-30 to 1e-26 eV is not in the decay code (section 3).

## 1. Checked and correct (today's diff)

| Item | How checked | Result |
|---|---|---|
| dp_si decay channel (`_si_sweep_cols` windows, `dec_own`, `dec_kfull/kex/fex/mom`, running moments into `reg`, the own bin in Q/Qn) | Called `_si_sweep_cols` with every collision term off. Compared one step against `dp_decay.decay_step(fold=True)` with the folded spectrum, for (m_l, M) = (1e-6, 1e-4), (1e-6, 0.03), (0.1, 1e-3), (0.1, 0.008), at lambda = 1e-4 and 1e3 per step | dF agrees to 1.8e-9 (massless window) and 3e-12 (others). The number kept is identical. No unit test covered this before; si_dec_check.py can become one |
| dp_si folds with the neutrino parent's spectrum alone (`spf` from `sp[0]`) | Kept plus flipped, kind 0 against kind 1: dp_e, a complex texture, L_e - L_mu, six (m_l, M) cases | Equal to <= 7e-14. The sum is flat on [z-, z+] (isotropic Majorana decay), so the fold is exact |
| Kept/flipped split per kind | Same runs | The kept shares (0.34-0.61) are equal for both kinds. At nubar/nu = 1 the daughters' nu and nubar are equal whatever the split, so the helicity bookkeeping cannot move the 1:1 planes |
| Fit a + b z + c/z against the exact traces, at angles off the nine nodes | fit_check.py | Pointwise <= 1.5e-4. One exception: the flipped piece of quasi-degenerate pairs (m_l = 0.1, 2 -> 1, z in [0.9925, 1], narrower than one bin) is 2% pointwise. The pieces are normalized and the window lies inside one or two bins, so this has no effect |
| `window_plan` / `decay_step` | Read k1/k2/ktop/kbot, the tolerance cases (zp = 1, zp on a bin edge, zm -> 0, zm below the grid with the k2 = N+1 clamp), the add/sub timing, pend, the own bin (z in [1/h, 1]) and the lower edge. The existing brute-force test covers the arithmetic | Correct. The running sums only grow down the sweep (v proportional to E^(1-p) or E^(-p)), so they lose no precision |
| Midpoint arrays (`ju_mid`, `C_mid`, `c_mid`, `z_mid`) and `jump_I`, `decay_I` | Index alignment (step k spans z_bins[k-1] to z_bins[k]). The Strang decays 0.5 + (n_sub - 1) + 0.5 add up to the half step | Correct |
| kappa in the loss | GG_table and Gd_table come from G_jump = G - kappa I, so ((G + kappa)(G + kappa)^dag)_aa = GG + 2 kappa Gd + kappa^2. dp_si's `loss_beam` and the threshold's `beam_max` include kappa. The Galaxy raises for eps' != 0 | Consistent across the three transports |
| dp_galactic_path | Without processes, the sweep's sum over j of I wq P / I equals `dp_galactic.emission_populations`. The vacuum SVD path is linear (clip off, `Na == 0`). seglen and the emission order are right. In the prange, every work array is allocated inside the loop body (numba hoists it into each worker's gufunc) and the callees are serial and allocate locally | Correct and thread-safe |
| Relic factor, Debye mass and G in the Galaxy against the cosmology | Read | Same formulas. The z_ref = 0/1 difference is F6, already known |

## 2. Finding (latent, numerics)

### L1. dp_si's bin exponential erases every rate below ~1e-16 of the bin's largest

Where: `dp_si._phi_functions` -> `_phi_eig` (dp_si.py:1706-1745), fed by `_bin_matrix` (1395-1456).

Mechanism. Scaling and squaring divides the matrix by 2^s, with s set by its largest entry. Any other rate r becomes
r 2^-s. When that is below ~1e-16, e^(X)[a, a] rounds to exactly 1, and the squarings keep it at 1. When a
decaying state has tdec per step >~ 1e16 times another state's rates, the other state's losses vanish, and so does
its conversion-then-decay drain.

Check (stiff2.py; one state with loss 1 per step, conversions c both ways to a state decaying tdec per step, 1%
own-bin share):

| c | tdec | survival in code | exact |
|---|---|---|---|
| 1e10 | 1e20 | 0 | 0 |
| 1e20 | 1e40 | 1.000 | e^-1e20 |
| 1e28 | 1e46 | 1.000 | e^-1e28 |

Scenario. dp ul, 'si' mode, m_Zd << m_D (3e-9 eV at g = 1e-4). tdec at E_min is 2e33 per step at m_Zd = 1e-16 eV and
2e53 at m_Zd = 1e-26 eV (loss_mag.py), while the conversions saturate at M_t = m_D. Result: the code gives nu_1
survival 1 where the exact answer is 0.

Why the planes do not change. Wherever tdec reaches 1e16 or more, the relic mass-change loss over the path is
already >= 1e17 eV at g >= 1e-4 (loss_mag.py: 2.7e18 eV for nu_1 at g = 1e-4, m_Zd <= 1e-8 eV). That is above
E_max, so shift_step removes the flux anyway. With m_l = 0.1 the loss drops to ~1e15 eV, and the high-energy bins
survive it. In those bins, though, the conversions (~1e20 per step) dominate the matrix, so the conversion-decay
cycle is resolved and drains the flux. The case that would matter: decays on and the loss off in si mode, or a new
model with fast decays and slow couplings to the other states.

Fix if wanted. Eliminate the fast state before the exponential: treat the decaying state quasi-statically, with an
effective rate Q tau_fast / (tau_fast + ...). Or take the exponential by eigen-decomposition with block-scaled
columns. Or assert that max(tau) / min(nonzero rate) < 1e14 per bin.

## 3. The si-mode NaN at m_Zd 1e-30 to 1e-26 eV, g ~ 3e-4, texture e: not found by reading

Below m_D, m_Zd enters si mode only through the decays: rest rate proportional to 1/m_Zd^2, the relic composition
and the daughter spectra. M_t = m_D for everything else. These are all finite at m_Zd = 1e-31 to 1e-12 eV and
g = 3e-4 or 1e-2 (tiny_m.py):

- `rest_rates`: Gam_31 = 8.6e44 eV at 1e-30.
- `daughter_spectra`.
- `relic_composition`: eta' = [3, 0, 0].
- `_bin_matrix` + `_phi_functions` on a decay-dominated bin, up to 1e200 per step (phi_check.py).
- The Galaxy's `_soft_jumps` with Wd up to ~1e34 eV: D^2 ~ 1e68.

Zero-eta relic states are skipped in `relic_state_rates_analytic` and `_thermal_terms` (`live`).

Next step: rerun one failing point on Tycho with `np.seterr(all='raise')` and NUMBA_BOUNDSCHECK=1. Where the NaN
first appears names the routine. By L1, the region is excluded by the relic loss in any case.
