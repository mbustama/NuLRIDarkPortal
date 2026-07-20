#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Mauricio Bustamante"
__email__ = "mbustamante@nbi.ku.dk"


"""
detector_flavor_likelihood.py:
    Routines to return the detector flavor-ratio likelihood.

This module provides the statistical framework for comparing the *expected* flavor ratios at Earth (calculated by the mixing module) against the 
*measured* likelihood contours of specific neutrino experiments.

It supports various analytical approximations for the likelihood surface:
1. Symmetric Gaussian: Standard elliptical contours.
2. Asymmetric Gaussian: Independent widths (sig_l, sig_r) on either side of the peak.
3. Student's t: Heavy-tailed distributions (nu degrees of freedom).
4. Smooth vs. Hard Transitions: Logic for blending asymmetric parameters.
5. Polar Ray-Casting: Radial power-law interpolation bounded by 1D periodic Fourier polynomials.
"""

import numpy as np
# import params_detector_flavor_likelihood
import src.core.params_detector_flavor_likelihood as params_detector_flavor_likelihood

def detector_flavor_log_likelihood(
    x_points: np.ndarray, 
    y_points: np.ndarray, 
    param_dict) -> np.ndarray:
    """
    Calculates the log-likelihood of the detector flavor ratio distribution
    at the given (x, y) points.

    The function infers the type of distribution (Gaussian/Student's t, 
    Symmetric/Asymmetric/Polar, Smooth/Hard) from the 'fit_model' key in param_dict.
    
    Mathematical Basis:
    The likelihood is typically parameterized via a Mahalanobis-like distance 'Q':
    Q = (1 / (1-rho^2)) * [ z_x^2 - 2*rho*z_x*z_y + z_y^2 ]
    where z_x = (x - mu_x) / sigma_x.

    Args:
        x_points (np.ndarray): 1D array of x-coordinates (fraction of nu_e).
        y_points (np.ndarray): 1D array of y-coordinates (fraction of nu_mu).
        param_dict (dict): Dictionary of parameters returned by fit_kde_to_contours.
                           Must contain 'Lmax', 'fit_model', 'mu_x', 'mu_y',
                           and the relevant shape parameters (sig_x, rho, or coeffs).

    Returns:
        np.ndarray: Array of log-likelihood values at the input points.
                    Returns -inf for points where probability is 0 or undefined.
    """
    
    # --- 1. Validate Inputs ---
    # Ensure inputs are at least 1D arrays for consistent vectorization
    x = np.atleast_1d(x_points)
    y = np.atleast_1d(y_points)
    
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x has shape {x.shape}, y has shape {y.shape}")

    # --- 2. Parse Model Type from param_dict ---
    # Determine the statistical model based on string flags in the config
    fit_model_str = param_dict.get('fit_model', '').lower()
    
    is_polar_ray_casting = "polar ray-casting" in fit_model_str
    is_asymmetric = "asymmetric" in fit_model_str
    is_smooth = "smooth" in fit_model_str
    is_students_t = "student" in fit_model_str
    
    # Get normalization constant Lmax (Peak Likelihood)
    Lmax = param_dict.get('Lmax', 1.0)
    if Lmax <= 0:
        # Fallback for invalid Lmax to avoid log errors
        log_Lmax = 0.0 
    else:
        log_Lmax = np.log(Lmax)

    # Get distribution centroid
    mu_x = param_dict['mu_x']
    mu_y = param_dict['mu_y']

    # --- 3. Helper Function: Vectorized Q Calculation ---
    # This computes the generalized squared distance (Q) for a 2D correlated Gaussian
    def calculate_Q_vec(x_in, y_in, sx, sy, r):
        # Avoid division by zero or invalid params using safe context
        with np.errstate(divide='ignore', invalid='ignore'):
            z_x = (x_in - mu_x) / sx
            z_y = (y_in - mu_y) / sy
            rho_sq = r**2
            
            # Safety Check: Correlation coefficient |rho| must be < 1
            if np.isscalar(r) and abs(r) >= 1.0:
                return np.full_like(x_in, np.inf)
            
            # Standard Elliptical Equation
            prefactor = 1.0 / (1.0 - rho_sq)
            q_val = prefactor * (z_x**2 - 2 * r * z_x * z_y + z_y**2)
            
        return q_val

    # --- 4. Calculate Q based on Model Type ---
    
    if is_polar_ray_casting:
        # --- Polar Ray-Casting (Fourier Smoothed) Logic ---
        # Calculates radii to 68% and 95% boundaries and interpolates via power law
        c68 = np.array(param_dict['coeffs_68'])
        c95 = np.array(param_dict['coeffs_95'])
        order = param_dict.get('order', 10)
        
        dx = x - mu_x
        dy = y - mu_y
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        
        # Helper to dynamically evaluate the bounded Fourier polynomial
        def eval_fourier(th, coeffs, ord):
            rad = coeffs[0] * np.ones_like(th)
            idx = 1
            for k in range(1, ord + 1):
                rad += coeffs[idx] * np.cos(k * th) + coeffs[idx+1] * np.sin(k * th)
                idx += 2
            return rad
            
        r68 = eval_fourier(theta, c68, order)
        r95 = eval_fourier(theta, c95, order)
        
        # Mathematically enforce boundaries to prevent crossover or inversion
        r68 = np.maximum(r68, 1e-4)
        r95 = np.maximum(r95, r68 * 1.05)
        
        ratio_Q = 5.99 / 2.30
        ratio_r = r95 / r68
        
        # Extract steepness exponent for the local parabolic bowl
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.log(ratio_Q) / np.log(ratio_r)
        p = np.clip(p, 1.0, 4.0)
        
        C = 2.30 / (r68**p)
        Q = C * (r**p)
        
        # Enforce strict 0 at the origin to prevent floating point noise
        Q = np.where(r < 1e-8, 0.0, Q)
        
    elif is_asymmetric:
        # Model has different widths/correlations on the Left vs. Right of the peak
        
        # Extract Left (L) parameters
        sig_x_l = param_dict['sig_x_l']
        sig_y_l = param_dict['sig_y_l']
        rho_l   = param_dict['rho_l']
        
        # Extract Right (R) parameters
        sig_x_r = param_dict['sig_x_r']
        sig_y_r = param_dict['sig_y_r']
        rho_r   = param_dict['rho_r']
        
        if is_smooth:
            # --- Smooth Asymmetric Logic ---
            # Instead of a hard cut at x = mu_x, we blend the parameters smoothly
            # using a tanh function.
            
            # Calculate transition weight w(x)
            # Characteristic width for the transition
            sig_x_avg = (sig_x_l + sig_x_r) / 2.0
            
            # Avoid div by zero if width is negligible
            if sig_x_avg < 1e-9:
                # Fallback to hard split logic efficiently
                mask_left = x < mu_x
                w = np.where(mask_left, 0.0, 1.0)
            else:
                z_smooth = (x - mu_x) / sig_x_avg
                # w goes from 0 (left) to 1 (right) smoothly
                w = 0.5 * (1.0 + np.tanh(z_smooth))
            
            # Blend parameters (Vectorized) for every point x
            sx = (1.0 - w) * sig_x_l + w * sig_x_r
            sy = (1.0 - w) * sig_y_l + w * sig_y_r
            r  = (1.0 - w) * rho_l   + w * rho_r
            
            Q = calculate_Q_vec(x, y, sx, sy, r)
            
        else:
            # --- Hard Split Asymmetric Logic ---
            # Parameters switch instantly at x = mu_x
            
            mask_left = x < mu_x
            mask_right = ~mask_left
            
            Q = np.zeros_like(x, dtype=float)
            
            # Calculate Q for left points using left params
            if np.any(mask_left):
                Q[mask_left] = calculate_Q_vec(
                    x[mask_left], y[mask_left], sig_x_l, sig_y_l, rho_l
                )
            # Calculate Q for right points using right params
            if np.any(mask_right):
                Q[mask_right] = calculate_Q_vec(
                    x[mask_right], y[mask_right], sig_x_r, sig_y_r, rho_r
                )

    else:
        # --- Symmetric Logic (Standard) ---
        # Single set of width/correlation parameters for the whole domain
        sig_x = param_dict['sig_x']
        sig_y = param_dict['sig_y']
        rho   = param_dict['rho']
        
        Q = calculate_Q_vec(x, y, sig_x, sig_y, rho)

    # --- 5. Calculate Log-Likelihood from Q ---
    
    # Prepare output array
    log_likelihood = np.full_like(Q, -np.inf)
    
    # Filter out NaNs or Infs in Q (invalid points remain -inf)
    valid_q = np.isfinite(Q)
    
    if is_students_t:
        # Student's t-Distribution (Heavy Tails)
        # PDF ~ (1 + Q/nu)^(-(nu+2)/2)
        # Log-PDF ~ -((nu+2)/2) * log(1 + Q/nu)
        
        nu = param_dict.get('nu', 1.0) # Degrees of freedom
        if nu <= 0: nu = 1.0 # Safety check
        
        # Calculate log-likelihood only for valid Q
        factor = -(nu + 2.0) / 2.0
        term = np.log(1.0 + Q[valid_q] / nu)
        
        log_likelihood[valid_q] = log_Lmax + factor * term
        
    else:
        # Gaussian Distribution (Standard)
        # PDF ~ exp(-0.5 * Q)
        # Log-PDF ~ -0.5 * Q
        
        log_likelihood[valid_q] = log_Lmax - 0.5 * Q[valid_q]

    return log_likelihood


def detector_flavor_log_likelihood_wrapper(
    flavor_ratios_earth: np.ndarray, 
    experiment='icecube_mese_2025',
    **kwargs) -> np.ndarray:
    """
    Wrapper to interface mixing outputs with the likelihood calculator.
    
    Args:
        flavor_ratios_earth (np.ndarray): Array of shape (N, 3) or (N, 4).
                                          Columns: [nu_e, nu_mu, nu_tau, (nu_s)]
        experiment (str): Key to look up parameter dict in 'params_detector_flavor_likelihood'.
    """

    # Retrieve experiment configuration
    param_dict = params_detector_flavor_likelihood.params_likelihood_index.get(experiment, None)

    if not (param_dict is None):
        # Extract Electron and Muon flavor fractions (X and Y coordinates)
        # Note: We ignore Tau/Sterile fractions here as the detector contours 
        # are typically defined in the 2D (fe, fmu) plane (Triangle Plot projection).
        fEe, fEmu = flavor_ratios_earth[:,0], flavor_ratios_earth[:,1]
        
        return detector_flavor_log_likelihood(fEe, fEmu, 
            params_detector_flavor_likelihood.params_likelihood_index[experiment])
    else:
        print("detector_flavor_log_likelihood_wrapper: Experiment "+experiment+ " unknown")
        quit()

def detector_flavor_log_likelihood_from_q(
    Q: np.ndarray, 
    experiment: str = 'icecube_mese_2025') -> np.ndarray:
    """
    Calculates the log-likelihood from Q values for a specific experiment.

    Args:
        Q (np.ndarray): Array of pre-computed Q values.
        experiment (str): The name of the experiment to look up in the global parameter index.

    Returns:
        np.ndarray: Log-likelihood values.
    """
    
    # --- 1. Retrieve Parameters ---
    # We look up the dictionary using the string provided
    param_dict = params_detector_flavor_likelihood.params_likelihood_index.get(experiment, None)
    
    if param_dict is None:
        raise ValueError(f"Experiment '{experiment}' not found in params_likelihood_index.")

    # --- 2. Parse Model Type ---
    fit_model_str = param_dict.get('fit_model', '').lower()
    is_students_t = "student" in fit_model_str
    
    # Get Lmax safely
    Lmax = param_dict.get('Lmax', 1.0)
    log_Lmax = np.log(Lmax) if Lmax > 0 else 0.0

    # --- 3. Compute Log-Likelihood ---
    # Ensure Q is at least 1D
    Q = np.atleast_1d(Q)
    log_likelihood = np.full_like(Q, -np.inf)
    valid_q = np.isfinite(Q)
    
    if is_students_t:
        nu = param_dict.get('nu', 1.0)
        if nu <= 0: nu = 1.0
        
        factor = -(nu + 2.0) / 2.0
        term = np.log(1.0 + Q[valid_q] / nu)
        log_likelihood[valid_q] = log_Lmax + factor * term
        
    else:
        # Gaussian
        log_likelihood[valid_q] = log_Lmax - 0.5 * Q[valid_q]
        
    return log_likelihood

# --- Example Usage / Unit Test ---
if __name__ == "__main__":
    print("Testing detector_flavor_log_likelihood...")

    # 1. Create dummy parameter dictionaries
    # Symmetric Gaussian
    params_sym_gauss = {
        'fit_model': 'Symmetric Gaussian',
        'Lmax': 10.0,
        'mu_x': 0.3, 'mu_y': 0.4,
        'sig_x': 0.1, 'sig_y': 0.2, 'rho': 0.5
    }
    
    # Asymmetric Smooth Student's t
    params_asym_smooth_t = {
        'fit_model': 'Asymmetric (Smooth) Student\'s t',
        'Lmax': 5.0,
        'mu_x': 0.3, 'mu_y': 0.4,
        'sig_x_l': 0.1, 'sig_y_l': 0.2, 'rho_l': 0.0,
        'sig_x_r': 0.2, 'sig_y_r': 0.1, 'rho_r': 0.5,
        'nu': 3.0
    }
    
    # Polar Ray-Casting (Fourier Smoothed)
    params_polar_ray = {
        'fit_model': 'Polar Ray-Casting (Fourier Smoothed)',
        'true_mu_x': 0.3264281,
        'true_mu_y': 0.34311662,
        'mu_x': 0.3264281,
        'mu_y': 0.34311662,
        'order': 10,
        'coeffs_68': [1.183221e-01, 4.418771e-03, -2.949472e-02, -4.538490e-02, 1.218375e-02, 3.788388e-03, 9.055966e-03, 1.167533e-02, -8.104457e-03, -3.415312e-03, -1.991886e-03, -2.734747e-03, 4.221461e-03, 1.213834e-03, -8.241564e-05, 5.073046e-07, -1.775718e-03, -4.176929e-04, 3.758724e-04, 2.606479e-04, 5.289579e-04],
        'coeffs_95': [3.551373e-01, 9.697366e-02, 5.699344e-02, -1.238779e-01, 1.035257e-01, -4.506697e-02, 2.803580e-02, -1.066661e-02, -4.164282e-02, -1.678729e-02, -2.927188e-02, 5.013040e-03, -5.638461e-03, 1.651218e-02, -5.160060e-03, 5.811135e-03, -1.504007e-03, -1.921467e-04, 4.526411e-03, 4.885233e-04, 2.746897e-03],
        'Lmax': 1.0,
        'q_values': {0.68: 2.30, 0.95: 5.99}
    }

    # 2. Create test points
    # Center, some points nearby
    x_test = np.array([0.3, 0.35, 0.4, 0.3264281, 10.0]) # Last one is way out
    y_test = np.array([0.4, 0.45, 0.5, 0.34311662, 10.0])

    # 3. Run and print results
    print("\n--- Symmetric Gaussian ---")
    ll_sym = detector_flavor_log_likelihood(x_test, y_test, params_sym_gauss)
    for i in range(len(x_test)):
        print(f"Point ({x_test[i]}, {y_test[i]}): LogL = {ll_sym[i]:.4f}")

    print("\n--- Asymmetric Smooth Student's t ---")
    ll_asym = detector_flavor_log_likelihood(x_test, y_test, params_asym_smooth_t)
    for i in range(len(x_test)):
        print(f"Point ({x_test[i]}, {y_test[i]}): LogL = {ll_asym[i]:.4f}")
        
    print("\n--- Polar Ray-Casting (Fourier Smoothed) ---")
    ll_polar = detector_flavor_log_likelihood(x_test, y_test, params_polar_ray)
    for i in range(len(x_test)):
        print(f"Point ({x_test[i]}, {y_test[i]}): LogL = {ll_polar[i]:.4f}")