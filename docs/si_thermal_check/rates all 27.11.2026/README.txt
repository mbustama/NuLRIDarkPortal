Rates Gamma_3r(E) of an incident nu_3 on relic nu_r, G = g diag(0,0,1) (flavor basis), g = 1e-3, M = 1 MeV,
m_1 = 1e-6 eV, normal ordering, NuFIT 6.1, T = 1.676389e-4 eV (z = 0). Cases: A (S,M), B (V-A,M), C (S,D), D (V-A,D).

numerical/   	Gamma3r_{A,B,C,D}_*.csv, r = 1, 2, 3 (nu_1 ultrarelativistic; nu_2, nu_3 non-relativistic).
             	Columns: E_eV, s_pole_per_s, s_off_per_s, t_u_no_interf_per_s, t_u_interf_per_s, s_t_interf_per_s, pair_per_s,
             	total_per_s, total_no_interf_per_s, mfp_* (Mpc); file C also: active_per_s, sterile_per_s, mfp_active_Mpc, mfp_sterile_Mpc.
             	Gamma31: 201 points, E = 1e13..1e17 eV.  Gamma32, Gamma33: 501 points, E = 1e12..1e17 eV.
analytical UR/  Gamma31_analytic_{A,B,C,D}_*.csv: analytic rate (formulas of the paper, ultrarelativistic relic nu_1).
             	Columns: E_eV, off_pole_per_s, pole_per_s, total_per_s, numerical_total_per_s, ratio_analytic_to_numerical.
             	max |analytic/numerical - 1| = 8.3e-4 (A), 1.4e-3 (B), 2.3e-4 (C), 3.1e-5 (D).
