Gamma_31(E): rate of nu_3 on relic nu_1 for secret interactions. Files produced by Gamma31_benchmark.nb.

SETUP
  G = g diag(0,0,1) in the flavor basis, g = 1e-3, M = 1 MeV, m_1 = 1e-6 eV, T(z=0) = 1.676389e-4 eV,
  NuFIT 6.1 NO (sin^2 th12 = 0.3088, sin^2 th13 = 0.02248, sin^2 th23 = 0.470, dCP = 212 deg).
  E = 10 TeV ... 100 PeV, 50 points per decade (201 rows). Rates are in 1/s, mean free paths c/Gamma in Mpc.
  Cases: A = (S,M), B = (V-A,M), C = (S,D), D = (V-A,D). Only nu_3 on relic nu_1 (not summed over relic states).

1. NUMERICAL RATES (thermal average computed by integration over eps)   Gamma31_{A,B,C,D}_*.csv
   columns: E_eV, s_pole_per_s, s_off_per_s, t_u_no_interf_per_s, t_u_interf_per_s, s_t_interf_per_s,
            pair_per_s, total_per_s, total_no_interf_per_s, mfp_*_Mpc (c/Gamma of the same contribution).
   "s channel alone" = s_pole + s_off. "t and u alone" = t_u_no_interf (no t-u interference).
   total = everything incl. interference; total_no_interf drops s-t and t-u interference.
   File C also has active_per_s, sterile_per_s (collisions weighted by the number of active / sterile products).
   Weights for (S,D): each collision is weighted by the number of active (or sterile) neutrinos among its products, counted before
   any \phi decay. So the mediator-pair channel \nu \bar{\nu} -> \phi \phi has weight zero, and the resonant s channel is counted as
   \nu \bar{\nu} -> \nu \bar{\nu} (one active plus one sterile). If you need it counted differently, for example with the \phi decay
   products included or the resonance excluded, let me know and I will recompute.

2. ANALYTIC RATES (formulas of Appendices A-D of the paper, FD series; no integration)   Gamma31_analytic_{A,B,C,D}_*.csv
   columns: E_eV, off_pole_per_s, pole_per_s, total_per_s (their sum), numerical_total_per_s (from file 1),
            ratio_analytic_to_numerical.
   Notes: (V-A,M) is not in the paper. Gamma(V-A,D) = n (1+pi)/2 [...], Gamma(V-A,M) = n [...] with the same bracket and the same
   mediator width, so there is no extra coupling factor; the only difference is the relic chirality weight (1+pi)/2 -> 1 for
   relativistic relics (1 - pi = eta ~ 1e-6 for m_1 = 1e-6 eV). For non-relativistic relics Dirac = Majorana/2. B therefore uses the D formulas.
   Numerically B and D differ by <1e-5 except at E ~ 1e17 eV (1.4e-3 in the pole), where the resonance probes relics with p ~ m_1.
   off_pole can be slightly negative for case A (finite part of the s channel); total is positive.
   Case C (S,D, helicity-polarized relic): as in the paper, the off-resonant s channel and the s-t interference are neglected
   (helicity suppressed, ~m^2/(E T)); the s-channel pole is kept. The numerical file keeps all channels.
   Analytic and numerical agree to ~1e-6 near the resonance. At E = 10-60 TeV they differ by up to 4.5% (A), -1.8% (B, D)
   because of the approximate heavy-mediator limits (s-t interference, pair production) used in the paper; C agrees to 2e-4.
