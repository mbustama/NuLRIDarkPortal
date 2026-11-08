# Existing bounds (gray curves on the limit figures)

Read by `analysis/existing_bounds.py`, which also records what is still to be verified for each. Columns:
log10(M / eV), log10(coupling), excluded above, unless noted. Copied on 2026-11-07 from Mauricio's earlier projects:

| File | Source | Copied from |
|---|---|---|
| `coloma_2009.14220_emu.txt` | Coloma, Gonzalez-Garcia & Maltoni, arXiv:2009.14220, global oscillation fit, L_e - L_mu | `LRF-LBL2/plots/Script/fig-1-9/created-data-set.txt` (drawn in JHEP 09 (2024) 055, Fig. 1) |
| `dror_2004.04750_fig4_dirac_{emu,etau}.txt` | Dror, arXiv:2004.04750, Fig. 4, Dirac neutrinos | `LRF-LBL2/.../other-data/arXiv-2004-04750-fig4-dirac-*.txt` |
| `poddar_2002.02935_precession.txt` | Kumar Poddar, Mohanty & Jana, arXiv:2002.02935, planetary precession | `LRF-LBL2/.../other-data/arXiv-2002-02935.txt` |
| `poddar_1908.09732_binaries.txt` | arXiv:1908.09732, compact binaries | `LRF-LBL2/.../other-data/arXiv-1908-09732.txt` |
| `agarwalla_2305.03675_{emu,etau}.txt` | Agarwalla, Bustamante, Das & Narang, arXiv:2305.03675, 95% C.L. | `LRF-LBL2/.../Fig_1_published_data_LRI_Astro/contour_95per2020_*_NMO.txt` |
| `adelberger_0712.0607_torsion_raw.txt` | Torsion balance, arXiv:0712.0607 Fig. 3: range [m], alpha [units of G_N m_u^2] | typed into `LRF-LBL2/.../money-plot.py` |
| `dror_2004.04750_fig4_majorana_{emu,etau}_{unitarity,mesons,0vbb,bbn_sn,nudecay_lab,nudecay_cmb}.txt` | Dror, arXiv:2004.04750v2, Fig. 4 left (Majorana): unitarity, meson decays, 0nu2beta X, BBN/SN, nu decays (solar; CMB dashed); g_X = our g'; C.L. not stated | digitized 2026-11-08 |
| `dror_2004.04750_fig4_majorana_{emu,etau}_prev_{nuosc,fifthforce_ep}.txt` | Dror Fig. 4, gray: "nu osc" (arXiv:1808.02042) and "fifth force+EP" (arXiv:1803.00591); same path in both panels | digitized 2026-11-08 |
| `coloma_2009.14220v5_fig2_{emu,etau}_{osc,fifthforce,eqprinc}.txt` | Coloma et al., arXiv:2009.14220v5, Fig. 2: oscillations at 2 sigma (Delta chi^2 > 4); 5th force and EP as recast by the authors | digitized 2026-11-08 |
| `bauer_1803.05466_fig{14,15}_{emu,etau}_{whitedwarfs,superk}.txt` | Bauer, Foldenauer & Jaeckel, arXiv:1803.05466v3, Figs. 14-15: white-dwarf cooling (lower edge), Super-K matter effects; plotted only above 2 MeV | digitized 2026-11-08 |
| `blinov_1905.02727_fig1_{universal,e,mu,tau}_lab.txt` | Blinov, Kelly, Krnjaic & McDermott, arXiv:1905.02727v2, Fig. 1: lab bounds on a scalar, Majorana | digitized 2026-11-08 |
| `blinov_1905.02727_fig1_bbn_{sinu,minu}_{real,complex}.txt`, `..._bbn_dirac_rhn.txt` | Blinov Fig. 1: BBN lower mass limits (95%, vertical lines, excluded left); Dirac right-handed-neutrino line | digitized 2026-11-08 |
| `brdar_2003.05339_fig4_{ee,mumu,tautau,universal}_{zinv,taudecay}.txt` | Brdar, Lindner, Vogl & Xu, arXiv:2003.05339v2, Fig. 4: Z invisible width and tau decays, 3 sigma, scalar | digitized 2026-11-08 |
| `wp_2203.01955_fig9_*.txt` | Neutrino self-interaction white paper, arXiv:2203.01955v1, Fig. 9: SN 1987A, CMB, BBN, phi-beta-beta, lab, stellar cooling, TXS, HESE; original source in each header | digitized 2026-11-08 |
| `wp_2203.01955_fig10_tautau_icecube_hese7p5yr.txt` | White paper Fig. 10 = Esteban et al., arXiv:2107.13568: IceCube HESE 7.5 yr, nu_tau-only scalar | digitized 2026-11-08 |

Paths are under `/home/mbustamante/Research/`. Black-hole superradiance (arXiv:1704.05081, Fig. 5) is three mass
ranges, kept in `analysis/existing_bounds.py`.

Rows marked "digitized 2026-11-08" come from the vector paths of the arXiv PDFs (PyMuPDF), with the axes calibrated on
the drawn ticks. Each file's header gives the curve, the C.L., the coupling convention as the source writes it, the
excluded side (exceptions: vertical BBN lines, closed or open contours), and the method.
