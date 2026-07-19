#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"

import numpy as np

# ==============================================================================
# MODULE: DETECTOR PARAMETERS
# ==============================================================================
# This file serves as a database of fitted likelihood parameters for different 
# future detector scenarios.
#
# These dictionaries are loaded by 'detector_flavor_likelihood.py' to construct
# the analytical likelihood functions (Gaussian or Student's t) for the 
# observable flavor ratios (f_e, f_mu).
#
# Dictionary Key Definitions:
# - 'fit_model': String defining the shape (e.g., 'Symmetric Gaussian', 'Asymmetric (Smooth) Gaussian').
# - 'mu_x', 'mu_y': The best-fit flavor fractions for nu_e and nu_mu, respectively.
# - 'sig_x', 'sig_y': The widths (standard deviations) of the distribution.
# - 'rho': Correlation coefficient between nu_e and nu_mu fractions [-1, 1].
# - 'Lmax': The maximum likelihood value (peak normalization).
# - '_l' / '_r' suffixes: denote 'Left' and 'Right' parameters for asymmetric distributions.
# ==============================================================================

# ==============================================================================
# TeV-PeV flavor composition
# ==============================================================================

"""
Simulated cases:
- Case 1: IC only at years 2040 and 2050
- Case 2: IC, KM3NeT, Baikal at years 2040 and 2050
- Case 3: IC+Gen2 and KM3NeT and Baikal and P-ONE at years 2040 and 2050
- Case 4: IC+Gen2 and KM3NeT and Baikal and P-ONE and either NEON or TRIDENT at year 2050
- Case 5: IC+Gen2 and KM3NeT and Baikal and P-ONE and HUNT at year 2050
"""

###########################
# IceCube MESE, year 2025
###########################

params_likelihood_ic_mese_2025 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
    'true_mu_x': 0.300,
    'true_mu_y': 0.375,
    'mu_x': 0.300,
    'mu_y': 0.375,
    'nu': np.inf, # Degrees of freedom (infinite for Gaussian)
    'sig_x_l': 0.1086,
    'sig_y_l': 0.1038,
    'rho_l': 0.4001,
    'sig_x_r': 0.1407,
    'sig_y_r': 0.1071,
    'rho_r': -0.1499,
    'Lmax': 12.8029,
    'q_values': {}
}

###########################
# Pion decay, year 2040 
###########################

# Centered on pion decay, year 2040, case 1
params_likelihood_pion_2040_case_1 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x_l': 0.0624745493098947,
	'sig_y_l': 0.02304966819553198,
	'rho_l': 0.30377285605952253,
	'sig_x_r': 0.08112958419509708,
	'sig_y_r': 0.02810288274104437,
	'rho_r': 0.2305882650835452,
	'Lmax': 90.31927257951098,
	'q_values': {0.68: 2.2960, 0.95: 6.0349, 0.99: 9.1876}
}

# Centered on pion decay, year 2040, case 2
params_likelihood_pion_2040_case_2 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x': 0.05355922807075652,
	'sig_y': 0.015257397333201842,
	'rho': 0.5000443457401653,
	'Lmax': 221.6219571910622,
	'q_values': {0.68: 2.2756, 0.95: 5.9421}
}

# Centered on pion decay, year 2040, case 3
params_likelihood_pion_2040_case_3 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x_l': 0.04622821285661955,
	'sig_y_l': 0.01139069146104631,
	'rho_l': 0.41156632080544786,
	'sig_x_r': 0.034376825863877816,
	'sig_y_r': 0.010159209077010812,
	'rho_r': 0.4104091135845543,
	'Lmax': 392.51839480281336,
	'q_values': {0.68: 2.3091, 0.95: 5.9792, 0.99: 9.1567}
}

###########################
# Muon-damped, year 2040 
###########################

# Centered on muon-damped, year 2040, case 1
params_likelihood_muon_2040_case_1 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x_l': 0.04025796399856319,
	'sig_y_l': 0.021107389546793694,
	'rho_l': 0.4537334634473156,
	'sig_x_r': 0.08691419589114084,
	'sig_y_r': 0.02648641041372913,
	'rho_r': 0.45267642059858226,
	'Lmax': 111.25978994904445,
	'q_values': {0.68: 2.2864, 0.95: 5.9784, 0.99: 9.1836}
 }

# Centered on muon-damped, year 2040, case 2
params_likelihood_muon_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x_l': 0.03880058614059167,
	'sig_y_l': 0.014825991237531259,
	'rho_l': 0.2877446105853268,
	'sig_x_r': 0.057892099291301596,
	'sig_y_r': 0.020335627943267598,
	'rho_r': 0.5631902265204,
	'Lmax': 206.0405740850494,
	'q_values': {0.68: 2.2480, 0.95: 5.9945, 0.99: 9.2071}
}

# Centered on muon-damped, year 2040, case 3
params_likelihood_muon_2040_case_3 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x_l': 0.02771652913376624,
	'sig_y_l': 0.011079173148555933,
	'rho_l': 0.371859746290514,
	'sig_x_r': 0.043814877425363434,
	'sig_y_r': 0.014878974564600326,
	'rho_r': 0.4520498794696368,
	'Lmax': 361.63231038693357,
	'q_values': {0.68: 2.2752, 0.95: 5.9868, 0.99: 9.1553}
}

###########################
# Pion decay, year 2050 
###########################

# Centered on pion decay, year 2050, case 1
params_likelihood_pion_2050_case_1 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x': 0.06273761025976954,
	'sig_y': 0.02053489064456643,
	'rho': 0.32630583948432246,
	'Lmax': 130.92044950650882,
	'q_values': {0.68: 2.2763, 0.95: 5.9829, 0.99: 9.2102}
 }

# Centered on pion decay, year 2050, case 2
params_likelihood_pion_2050_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x_l': 0.04356601116358585,
	'sig_y_l': 0.011861867907108895,
	'rho_l': 0.3970366067744238,
	'sig_x_r': 0.04414329036350676,
	'sig_y_r': 0.012244676041646797,
	'rho_r': 0.38946416008757306,
	'Lmax': 324.574729931011,
	'q_values': {0.68: 2.3080, 0.95: 5.9709, 0.99: 9.2268}
}

# Centered on pion decay, year 2050, case 3
params_likelihood_pion_2050_case_3 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x_l': 0.02509725764960767,
	'sig_y_l': 0.00695917091710005,
	'rho_l': 0.44685458208674156,
	'sig_x_r': 0.03225723059251796,
	'sig_y_r': 0.008324880530243092,
	'rho_r': 0.44656727072210883,
	'Lmax': 815.7664880540253,
	'q_values': {0.68: 2.2349, 0.95: 6.0433, 0.99: 9.2340}
 }

# Centered on pion decay, year 2050, case 4
params_likelihood_pion_2050_case_4 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x': 0.023809434169848542,
	'sig_y': 0.006374611138125749,
	'rho': 0.36719019454316526,
	'Lmax': 1133.5802237040843,
	'q_values': {0.68: 2.2652, 0.95: 5.9646, 0.99: 9.1875}
 }

# Centered on pion decay, year 2050, case 5
params_likelihood_pion_2050_case_5 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.3264281,
	'true_mu_y': 0.34311662,
	'mu_x': 0.3264281,
	'mu_y': 0.34311662,
	'nu': np.inf,
	'sig_x_l': 0.020600881193007403,
	'sig_y_l': 0.004635586918749003,
	'rho_l': 0.3627804173566855,
	'sig_x_r': 0.01743432468281306,
	'sig_y_r': 0.005164077895654054,
	'rho_r': 0.5032950375387478,
	'Lmax': 1906.2501509870226,
	'q_values': {0.68: 2.2821, 0.95: 6.0445, 0.99: 9.2504}
}

###########################
# Muon-damped, year 2050 
###########################

# Centered on muon-damped, year 2050, case 1
params_likelihood_muon_2050_case_1 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x_l': 0.044754704384790255,
	'sig_y_l': 0.018622794575406065,
	'rho_l': 0.3847478497299732,
	'sig_x_r': 0.0669100346289641,
	'sig_y_r': 0.02322149178194479,
	'rho_r': 0.4880228331301833,
	'Lmax': 148.43109619332714,
	'q_values': {0.68: 2.2696, 0.95: 6.0017, 0.99: 9.2089}
 }

# Centered on muon-damped, year 2050, case 2
params_likelihood_muon_2050_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x_l': 0.03830703744915373,
	'sig_y_l': 0.014284263083466149,
	'rho_l': 0.5101235037143452,
	'sig_x_r': 0.03926921849950561,
	'sig_y_r': 0.01417125500956897,
	'rho_r': 0.3332592507584293,
	'Lmax': 324.51237777786696,
	'q_values': {0.68: 2.2732, 0.95: 6.0716, 0.99: 9.2366}
}

# Centered on muon-damped, year 2050, case 3
params_likelihood_muon_2050_case_3 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x_l': 0.017101495553919335,
	'sig_y_l': 0.007220931512686821,
	'rho_l': 0.3663403186295023,
	'sig_x_r': 0.0330887142404055,
	'sig_y_r': 0.0104266732137791,
	'rho_r': 0.4674348164622597,
	'Lmax': 768.2181706314867,
	'q_values': {0.68: 2.3062, 0.95: 6.0100, 0.99: 9.2454}
}

# Centered on muon-damped, year 2050, case 4
params_likelihood_muon_2050_case_4 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x': 0.022123644751604964,
	'sig_y': 0.007665680075581784,
	'rho': 0.43579499037063263,
	'Lmax': 998.3153576364127,
	'q_values': {0.68: 2.1559, 0.95: 5.8693}
}

# Centered on muon-damped, year 2050, case 5
params_likelihood_muon_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.21510017,
	'true_mu_y': 0.40712485,
	'mu_x': 0.21510017,
	'mu_y': 0.40712485,
	'nu': np.inf,
	'sig_x': 0.01931251968188957,
	'sig_y': 0.0062105882926369845,
	'rho': 0.35150188009005573,
	'Lmax': 1404.4213239988417,
	'q_values': {0.68: 2.2911, 0.95: 5.9148}
}

# ==============================================================================
# TeV-PeV flavor composition, HNL benchmarks
# ==============================================================================

"""
Simulated benchmarks:

(A) M = 3 GeV, |U_{e N}| = 5e-8, full pion decay
    rho = 0.00000000°, sigma = 16.24129481°
    Flavor ratios: [0.35877117, 0.32666965, 0.31455917]

(B) M = 3 GeV, |U_{mu N}| = 4e-8, muon-damped pion decay
    rho = 5.37949651°, sigma = 182.33677342°
    Flavor ratios: [0.15180555, 0.44344122, 0.40475323]

(C) M = 4 GeV, |U_{mu N}| = 1e-8, muon-damped pion decay
    rho = 5.14209452°, sigma = 181.25721314°
    Flavor ratios: [0.17815576, 0.42892295, 0.39292129]

(D) M = 1 GeV, |U_{tau N}| = 1.5e-6, full pion decay
    rho = 87.16302527°, sigma = 92.76623484°
    Flavor ratios: [0.35775892, 0.32907685, 0.31316423]

(E) M = 3 GeV, |U_{tau N}| = 6e-8, full pion decay
    rho = 90.60832584°, sigma = 91.22988747°
    Flavor ratios: [0.35256251, 0.33137133, 0.31606616]

(F) M = 1 GeV, |U_{tau N}| = 7e-7, muon-damped pion decay
    rho = 179.56121906°, sigma = 173.58374227°
    Flavor ratios: [0.15658559, 0.44239663, 0.40101778]

(G) M = 3 GeV, |U_{tau N}| = 3e-8, muon-damped pion decay
    rho = 179.73544004°, sigma = 174.04766052°
    Flavor ratios: [0.16489109, 0.43737232, 0.39773658]
"""

##############
# Benchmark A
##############

# Year 2040, case 2
params_likelihood_hnl_benchmark_A_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.35877117,
    'true_mu_y': 0.32666965,
    'mu_x': 0.3428348198347206,
    'mu_y': 0.3134841208996359,
    'nu': np.inf,
    'sig_x_l': 0.049011281260956656,
    'sig_y_l': 0.013813086383350869,
    'rho_l': 0.4002015391595438,
    'sig_x_r': 0.062158559757018846,
    'sig_y_r': 0.016033048809917693,
    'rho_r': 0.481826483276531,
    'Lmax': 213.10292041896855,
    'q_values': {0.68: 2.24708406901181, 0.95: 5.990118772320821, 0.99: 9.195333445739314}
}

# Year 2050, case 5
params_likelihood_hnl_benchmark_A_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.35877117,
	'true_mu_y': 0.32666965,
	'mu_x': 0.35877117,
	'mu_y': 0.32666965,
	'nu': np.inf,
	'sig_x': 0.01979927883986914,
	'sig_y': 0.005068220445512787,
	'rho': 0.21822997728194005,
	'Lmax': 1636.034638400174,
	'q_values': {0.68: 2.25532045710829, 0.95: 5.91676589719821, 0.99: 9.045691509425229}
}

##############
# Benchmark B
##############

# Year 2040, case 2
params_likelihood_hnl_benchmark_B_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.15180555,
	'true_mu_y': 0.44344122,
	'mu_x': 0.15180555,
	'mu_y': 0.44344122,
	'nu': np.inf,
	'sig_x_l': 0.03603888520129528,
	'sig_y_l': 0.01791839771755649,
	'rho_l': 0.4573733334892992,
	'sig_x_r': 0.05265492522668502,
	'sig_y_r': 0.01912382158808454,
	'rho_r': 0.4517712283809593,
	'Lmax': 215.78967173284954,
	'q_values': {0.68: 2.288340940749942, 0.95: 5.9781951508419695, 0.99: 9.211191940498328}
}

# Year 2050, case 5
params_likelihood_hnl_benchmark_B_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.15180555,
	'true_mu_y': 0.44344122,
	'mu_x': 0.15180555,
	'mu_y': 0.44344122,
	'nu': np.inf,
	'sig_x': 0.019946422862318958,
	'sig_y': 0.007953788851764176,
	'rho': 0.5610593616611785,
	'Lmax': 1156.38911696823,
	'q_values': {0.68: 2.3248862721881935, 0.95: 6.004187848638918, 0.99: 9.205915336714158}
}

##############
# Benchmark C
##############

# Year 2040, case 2
params_likelihood_hnl_benchmark_C_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.17815576,
	'true_mu_y': 0.42892295,
	'mu_x': 0.17815576,
	'mu_y': 0.42892295,
	'nu': np.inf,
	'sig_x_l': 0.04083032170374923,
	'sig_y_l': 0.01706661147264532,
	'rho_l': 0.4559554719881442,
	'sig_x_r': 0.04578818507970861,
	'sig_y_r': 0.017636960965678655,
	'rho_r': 0.43843207179845717,
	'Lmax': 237.37345462176643,
	'q_values': {0.68: 2.3048122443365613, 0.95: 5.985060444873069, 0.99: 9.220178265025549}
}

# Year 2050, case 5
params_likelihood_hnl_benchmark_C_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.17815576,
	'true_mu_y': 0.42892295,
	'mu_x': 0.17815576,
	'mu_y': 0.42892295,
	'nu': np.inf,
	'sig_x': 0.020515238049634857,
	'sig_y': 0.00856435425013452,
	'rho': 0.46036403520765257,
	'Lmax': 1016.93046765997,
	'q_values': {0.68: 2.236096003192375, 0.95: 5.888672287146187, 0.99: 9.107264272819352}
}

##############
# Benchmark D
##############

# Year 2040, case 2
params_likelihood_hnl_benchmark_D_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.35775892,
	'true_mu_y': 0.32907685,
	'mu_x': 0.35775892,
	'mu_y': 0.32907685,
	'nu': np.inf,
	'sig_x_l': 0.05530778087529353,
	'sig_y_l': 0.014866780284579134,
	'rho_l': 0.546527817375329,
	'sig_x_r': 0.047725816996831094,
	'sig_y_r': 0.013477956984262324,
	'rho_r': 0.3475611861500951,
	'Lmax': 241.88751415710942,
	'q_values': {0.68: 2.252332404361319, 0.95: 5.9279207742922315, 0.99: 9.16980312466501}
}

# Year 2050, case 5
params_likelihood_hnl_benchmark_D_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.35775892,
	'true_mu_y': 0.32907685,
	'mu_x': 0.35775892,
	'mu_y': 0.32907685,
	'nu': np.inf,
	'sig_x': 0.030753966287040413,
	'sig_y': 0.00882050488813104,
	'rho': 0.6254811917032034,
	'Lmax': 754.4080219613884,
	'q_values': {0.68: 2.3193178011829554, 0.95: 5.9921299322583454, 0.99: 9.167543112027312}
}

##############
# Benchmark E
##############

# Year 2040, case 2
params_likelihood_hnl_benchmark_E_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.35256251,
	'true_mu_y': 0.33137133,
	'mu_x': 0.35256251,
	'mu_y': 0.33137133,
	'nu': np.inf,
	'sig_x_l': 0.053062782933712915,
	'sig_y_l': 0.013783779329547524,
	'rho_l': 0.540913067011456,
	'sig_x_r': 0.04198269571301494,
	'sig_y_r': 0.0133357356232216,
	'rho_r': 0.3515520329696381,
	'Lmax': 272.66921991433975,
	'q_values': {0.68: 2.262977045114557, 0.95: 5.955580901714026, 0.99: 9.164232681908992}
}

# Year 2050, case 5
params_likelihood_hnl_benchmark_E_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.35256251,
	'true_mu_y': 0.33137133,
	'mu_x': 0.35256251,
	'mu_y': 0.33137133,
	'nu': np.inf,
	'sig_x': 0.018370985376463426,
	'sig_y': 0.0052418703075944074,
	'rho': 0.631100523984447,
	'Lmax': 2259.5736902507306,
	'q_values': {0.68: 2.414868160349469, 0.95: 6.21710879838563, 0.99: 9.261599100037543}
}

##############
# Benchmark F
##############

# Year 2040, case 2
params_likelihood_hnl_benchmark_F_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.15658559,
	'true_mu_y': 0.44239663,
	'mu_x': 0.15658559,
	'mu_y': 0.44239663,
	'nu': np.inf,
	'sig_x_l': 0.041846695095745234,
	'sig_y_l': 0.019403082233365065,
	'rho_l': 0.4465529291355027,
	'sig_x_r': 0.05011602180982033,
	'sig_y_r': 0.019983885458564832,
	'rho_r': 0.4502767553548945,
	'Lmax': 195.61170042817903,
	'q_values': {0.68: 2.2950032394082047, 0.95: 5.9834643859481815, 0.99: 9.163993932280313}
}

# Year 2050, case 5
params_likelihood_hnl_benchmark_F_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.15658559,
	'true_mu_y': 0.44239663,
	'mu_x': 0.15658559,
	'mu_y': 0.44239663,
	'nu': np.inf,
	'sig_x': 0.020259700186198104,
	'sig_y': 0.006395849212819049,
	'rho': 0.6517705412108619,
	'Lmax': 1582.9866848440222,
	'q_values': {0.68: 2.2315255683661612, 0.95: 5.95303191868237, 0.99: 9.21927834468753}
}

##############
# Benchmark G
##############

# Year 2040, case 2
params_likelihood_hnl_benchmark_G_2040_case_2 = {
	'fit_model': 'Asymmetric (Smooth) Gaussian',
	'true_mu_x': 0.16489109,
	'true_mu_y': 0.43737232,
	'mu_x': 0.16489109,
	'mu_y': 0.43737232,
	'nu': np.inf,
	'sig_x_l': 0.045500989753583986,
	'sig_y_l': 0.019421301798409972,
	'rho_l': 0.5115949768708165,
	'sig_x_r': 0.046537499048613935,
	'sig_y_r': 0.01855520869901489,
	'rho_r': 0.4886786555228172,
	'Lmax': 209.94258245421685,
	'q_values': {0.68: 2.3008663099624136, 0.95: 5.992955864299939, 0.99: 9.193460733931834}
}

# Year 2050, case 5
params_likelihood_hnl_benchmark_G_2050_case_5 = {
	'fit_model': 'Symmetric Gaussian',
	'true_mu_x': 0.16489109,
	'true_mu_y': 0.43737232,
	'mu_x': 0.16489109,
	'mu_y': 0.43737232,
	'nu': np.inf,
	'sig_x': 0.020603982706176867,
	'sig_y': 0.008188061814012116,
	'rho': 0.4198906231907149,
	'Lmax': 1050.1479990386927,
	'q_values': {0.68: 2.2988505699931467, 0.95: 5.959250044617936, 0.99: 9.187818226231434}
}


# ==============================================================================
# UHE flavor composition
# ==============================================================================

#######################################
# Pion decay, IceCube-Gen2 radio array
#######################################

# Uses neutrino flux 0 (high flux from fit to Auger UHECR data + 10% p)
params_likelihood_uhe_pion_ic_gen2_flux_0 = {
    'fit_model': 'Polar Ray-Casting (Fourier Smoothed)',
    'true_mu_x': 0.3264281,
    'true_mu_y': 0.34311662,
    'mu_x': 0.3264281,
    'mu_y': 0.34311662,
    'order': 10,
    'coeffs_68': [
        1.183221e-01, 4.418771e-03, -2.949472e-02, -4.538490e-02, 1.218375e-02, 3.788388e-03, 9.055966e-03, 1.167533e-02, -8.104457e-03, -3.415312e-03, -1.991886e-03, -2.734747e-03, 4.221461e-03, 1.213834e-03, -8.241564e-05, 5.073046e-07, -1.775718e-03, -4.176929e-04, 3.758724e-04, 2.606479e-04, 5.289579e-04
    ],
    'coeffs_95': [
        3.551373e-01, 9.697366e-02, 5.699344e-02, -1.238779e-01, 1.035257e-01, -4.506697e-02, 2.803580e-02, -1.066661e-02, -4.164282e-02, -1.678729e-02, -2.927188e-02, 5.013040e-03, -5.638461e-03, 1.651218e-02, -5.160060e-03, 5.811135e-03, -1.504007e-03, -1.921467e-04, 4.526411e-03, 4.885233e-04, 2.746897e-03
    ],
    'Lmax': 1.0,
    'q_values': {0.68: 2.30, 0.95: 5.99}
}

# Uses neutrino flux 1 (low flux from fit to TA UHECR data)
params_likelihood_uhe_pion_ic_gen2_flux_1 = {
    'fit_model': 'Polar Ray-Casting (Fourier Smoothed)',
    'true_mu_x': 0.3264281,
    'true_mu_y': 0.34311662,
    'mu_x': 0.3264281,
    'mu_y': 0.34311662,
    'order': 10,
    'coeffs_68': [
        2.276028e-01, 4.319337e-02, -3.663348e-02, -8.811605e-02, 1.500989e-02, -9.329551e-04, 1.816037e-02, 2.951692e-02, -7.971256e-03, -4.664204e-03, -3.860736e-03, -1.180612e-02, 5.189246e-03, 2.274593e-03, 1.791106e-04, 4.021678e-03, -3.510319e-03, -1.142956e-03, 1.967526e-04, -1.444556e-03, 1.386399e-03
    ],
    'coeffs_95': [
        1.064407e+01, 1.372352e+01, 1.355022e+01, -2.031190e-01, 1.667545e+01, -9.209160e+00, 9.384278e+00, -9.199740e+00, 1.903804e-01, -4.317978e+00, -4.040726e+00, -2.608063e-01, -3.326796e+00, 9.839734e-01, -1.249794e+00, 6.446587e-01, -1.035643e-01, 1.885882e-01, 9.605196e-02, 2.296956e-02, 2.971162e-02
    ],
    'Lmax': 1.0,
    'q_values': {0.68: 2.30, 0.95: 5.99}
}

#######################################
# Muon-damped, IceCube-Gen2 radio array
#######################################

# Uses neutrino flux 0 (high flux from fit to Auger UHECR data + 10% p)
params_likelihood_uhe_muon_ic_gen2_flux_0 = {
    'fit_model': 'Polar Ray-Casting (Fourier Smoothed)',
    'true_mu_x': 0.21510017,
    'true_mu_y': 0.40712485,
    'mu_x': 0.21510017,
    'mu_y': 0.40712485,
    'order': 10,
    'coeffs_68': [
        8.769102e-02, 5.187238e-03, 1.168749e-03, -4.961426e-02, 1.036929e-02, -1.532173e-03, 1.134141e-04, 1.974611e-02, -8.500224e-03, 2.929547e-04, -5.991607e-05, -8.291096e-03, 5.684430e-03, 3.079938e-04, 2.205843e-05, 3.411376e-03, -3.518661e-03, -4.866804e-04, 2.185326e-04, -1.577148e-03, 2.720406e-03
    ],
    'coeffs_95': [
        1.761914e-01, 2.449900e-02, 1.635149e-02, -1.006792e-01, 2.314060e-02, -1.112978e-02, -5.471863e-03, 4.016818e-02, -1.884400e-02, 4.351687e-03, 1.684891e-03, -1.674761e-02, 1.302214e-02, -1.403269e-03, -4.408803e-04, 6.494536e-03, -8.168881e-03, -4.631857e-06, -4.970116e-05, -2.297770e-03, 5.886914e-03
    ],
    'Lmax': 1.0,
    'q_values': {0.68: 2.30, 0.95: 5.99}
}

# Uses neutrino flux 1 (low flux from fit to TA UHECR data)
params_likelihood_uhe_muon_ic_gen2_flux_1 = {
    'fit_model': 'Polar Ray-Casting (Fourier Smoothed)',
    'true_mu_x': 0.21510017,
    'true_mu_y': 0.40712485,
    'mu_x': 0.21510017,
    'mu_y': 0.40712485,
    'order': 10,
    'coeffs_68': [
        1.724327e-01, 3.603867e-02, 1.486191e-02, -9.948975e-02, 2.301474e-02, -1.797368e-02, -3.543141e-03, 3.995661e-02, -1.856062e-02, 7.971236e-03, -4.109199e-04, -1.715075e-02, 1.244999e-02, -3.481211e-03, 1.529398e-03, 6.938322e-03, -7.956469e-03, 1.085728e-03, -1.776032e-03, -3.116700e-03, 6.093864e-03
    ],
    'coeffs_95': [
        4.870453e-01, 2.641075e-01, 2.980071e-01, -2.789680e-01, 2.903994e-01, -2.587847e-01, 2.837937e-02, -5.318209e-02, -1.564030e-01, 3.891189e-02, -1.201948e-01, 4.690255e-02, -1.332426e-02, 3.775783e-02, 2.420654e-02, 1.453696e-02, 1.363052e-02, -6.609625e-03, 7.567565e-03, -6.728584e-03, 6.448609e-03
    ],
    'Lmax': 1.0,
    'q_values': {0.68: 2.30, 0.95: 5.99}
}

# ==============================================================================
# Index
# ==============================================================================

# Master index mapping experiment/case strings to the parameter dictionaries above.
# This is the dictionary used by 'detector_flavor_likelihood_wrapper' to fetch parameters.
params_likelihood_index = {
	'icecube_mese_2025': params_likelihood_ic_mese_2025,
	'pion_2040_case_1': params_likelihood_pion_2040_case_1,
	'pion_2040_case_2': params_likelihood_pion_2040_case_2,
	'pion_2040_case_3': params_likelihood_pion_2040_case_3,
	'muon_2040_case_1': params_likelihood_muon_2040_case_1,
	'muon_2040_case_2': params_likelihood_muon_2040_case_2,
	'muon_2040_case_3': params_likelihood_muon_2040_case_3,
	'pion_2050_case_1': params_likelihood_pion_2050_case_1,
	'pion_2050_case_2': params_likelihood_pion_2050_case_2,
	'pion_2050_case_3': params_likelihood_pion_2050_case_3,
	'pion_2050_case_4': params_likelihood_pion_2050_case_4,
	'pion_2050_case_5': params_likelihood_pion_2050_case_5,
	'muon_2050_case_1': params_likelihood_muon_2050_case_1,
	'muon_2050_case_2': params_likelihood_muon_2050_case_2,
	'muon_2050_case_3': params_likelihood_muon_2050_case_3,
	'muon_2050_case_4': params_likelihood_muon_2050_case_4,
	'muon_2050_case_5': params_likelihood_muon_2050_case_5,
	'hnl_benchmark_A_2040_case_2': params_likelihood_hnl_benchmark_A_2040_case_2,
	'hnl_benchmark_A_2050_case_5': params_likelihood_hnl_benchmark_A_2050_case_5,
	'hnl_benchmark_B_2040_case_2': params_likelihood_hnl_benchmark_B_2040_case_2,
	'hnl_benchmark_B_2050_case_5': params_likelihood_hnl_benchmark_B_2050_case_5,
	'hnl_benchmark_C_2040_case_2': params_likelihood_hnl_benchmark_C_2040_case_2,
	'hnl_benchmark_C_2050_case_5': params_likelihood_hnl_benchmark_C_2050_case_5,
	'hnl_benchmark_D_2040_case_2': params_likelihood_hnl_benchmark_D_2040_case_2,
	'hnl_benchmark_D_2050_case_5': params_likelihood_hnl_benchmark_D_2050_case_5,
	'hnl_benchmark_E_2040_case_2': params_likelihood_hnl_benchmark_E_2040_case_2,
	'hnl_benchmark_E_2050_case_5': params_likelihood_hnl_benchmark_E_2050_case_5,
	'hnl_benchmark_F_2040_case_2': params_likelihood_hnl_benchmark_F_2040_case_2,
	'hnl_benchmark_F_2050_case_5': params_likelihood_hnl_benchmark_F_2050_case_5,
	'hnl_benchmark_G_2040_case_2': params_likelihood_hnl_benchmark_G_2040_case_2,
	'hnl_benchmark_G_2050_case_5': params_likelihood_hnl_benchmark_G_2050_case_5,
	'uhe_pion_ic-gen2_flux_0': params_likelihood_uhe_pion_ic_gen2_flux_0,
	'uhe_pion_ic-gen2_flux_1': params_likelihood_uhe_pion_ic_gen2_flux_1,
	'uhe_muon_ic-gen2_flux_0': params_likelihood_uhe_muon_ic_gen2_flux_0,
	'uhe_muon_ic-gen2_flux_1': params_likelihood_uhe_muon_ic_gen2_flux_1
}