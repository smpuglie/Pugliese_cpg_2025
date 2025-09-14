import time

import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
import jax
import yaml
# from scipy.stats import truncnorm

import os
import pandas as pd
from jax.scipy.signal import correlate

def sample_trunc_normal(key, mean, stdev, shape, lower_bound=0.0):
    """Sample from truncated normal for a single simulation."""
    
    # Handle infinite or NaN inputs
    def handle_invalid_inputs():
        return jnp.zeros(shape)
    
    # Handle edge case when stdev is 0 or very small
    def handle_zero_stdev():
        # When stdev is 0, return the mean value (clamped to be non-negative)
        return jnp.maximum(mean, 0.0) * jnp.ones(shape)
    
    def handle_normal_case():
        # Use inverse CDF method for truncated normal
        # Truncation points in original scale
        # Cap upper bound more aggressively to prevent numerical issues
        upper_bound = jnp.minimum(mean + jnp.minimum(100 * stdev, 1e6), 1e10)

        # Convert to standardized coordinates with bounds checking
        a = jnp.clip((lower_bound - mean) / stdev, -10.0, 10.0)
        b = jnp.clip((upper_bound - mean) / stdev, -10.0, 10.0)

        # Get CDF values at truncation points for standard normal
        cdf_a = jax.scipy.stats.norm.cdf(a)
        cdf_b = jax.scipy.stats.norm.cdf(b)

        # Ensure CDF values are valid and not too close
        cdf_a = jnp.clip(cdf_a, 1e-10, 1.0 - 1e-10)
        cdf_b = jnp.clip(cdf_b, 1e-10, 1.0 - 1e-10)
        cdf_b = jnp.maximum(cdf_b, cdf_a + 1e-10)

        # Sample uniform values between the CDF values
        u = jax.random.uniform(key, shape=shape, minval=cdf_a, maxval=cdf_b)

        # Use inverse CDF to get standard normal samples with bounds checking
        z = jax.scipy.stats.norm.ppf(jnp.clip(u, 1e-10, 1.0 - 1e-10))

        # Transform to desired mean and standard deviation with final bounds check
        samples = mean + stdev * z
        samples = jnp.clip(samples, -1e10, 1e10)

        return samples
    
    # Check for invalid inputs first
    invalid_inputs = (
        ~jnp.isfinite(mean) | ~jnp.isfinite(stdev) | 
        ~jnp.isfinite(lower_bound) | (stdev < 0)
    )
    
    return jax.lax.cond(
        invalid_inputs,
        handle_invalid_inputs,
        lambda: jax.lax.cond(
            stdev < 1e-10,
            handle_zero_stdev,
            handle_normal_case
        )
    )

def set_sizes(sizes, a, threshold):
    """Should correspond to surface area. This method is going to need a lot of improvement, very much testing out"""

    normSize = np.nanmedian(sizes)  # jnp.nanmean(sizes)
    sizes[np.isnan(sizes)] = normSize
    sizes[sizes == 0] = normSize  # TODO this isn't great
    sizes = sizes / normSize
    sizes = np.asarray(sizes)

    a = a / sizes[None,:]  # broadcasting to match shape
    threshold = threshold * sizes[None,:]  # broadcasting to match shape
    return jnp.asarray(a), jnp.asarray(threshold)


@jit
def find_peaks_1d(x, height_min=None, height_max=None, prominence_threshold=0.05):
    """
    JAX-compatible peak finding function for 1D arrays.
    This is the core implementation that works on a single 1D array.
    """
    n = x.shape[0]
    
    # Find local maxima using vectorized comparison
    is_peak = jnp.zeros(n, dtype=bool)
    is_peak = is_peak.at[1:-1].set(
        (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    )
    
    # Apply height constraints
    if height_min is not None:
        is_peak = is_peak & (x >= height_min)
    if height_max is not None:
        is_peak = is_peak & (x <= height_max)
    
    # Calculate prominences using vectorized operations
    # Left minimums: minimum from start to each position
    left_mins = jnp.minimum.accumulate(x)
    
    # Right minimums: minimum from each position to end
    right_mins = jnp.minimum.accumulate(x[::-1])[::-1]
    
    # Create arrays for left and right valley minimums
    left_valleys = jnp.concatenate([jnp.array([jnp.inf]), left_mins[:-1]])
    right_valleys = jnp.concatenate([right_mins[1:], jnp.array([jnp.inf])])
    
    # The base level is the higher of the two valley minimums
    base_levels = jnp.maximum(left_valleys, right_valleys)
    
    # Prominence is height above base level
    prominences = jnp.where(
        (jnp.arange(n) > 0) & (jnp.arange(n) < n - 1),
        x - base_levels,
        0.0
    )
    
    # Apply prominence threshold
    is_peak = is_peak & (prominences >= prominence_threshold)
    
    # Return all information as fixed-size arrays
    peak_mask = is_peak
    peak_indices = jnp.arange(n)
    peak_heights = x
    peak_prominences = prominences
    
    return peak_indices, peak_heights, peak_prominences, peak_mask

def autocorrelation_1d(activity):
    """Helper function to compute autocorrelation for a single activity trace."""
    autocorr = jnp.correlate(activity, activity, mode="full") / jnp.inner(activity, activity)
    return autocorr

@jit
def neuron_oscillation_score_helper_jax(activity, prominence):
    """JAX-compatible helper function for oscillation score calculation with enhanced numerical robustness."""
    # Add aggressive numerical robustness by rounding activity to avoid floating-point precision issues
    # This ensures consistent behavior across different parameter sets and devices
    # Use higher precision rounding for better stability
    activity = jnp.round(activity * 1e10) / 1e10  # Round to 10 decimal places for enhanced consistency
    
    # Clamp activity to reasonable range to avoid extreme values
    activity = jnp.clip(activity, -1e6, 1e6)
    
    # Normalize activity to [-1, 1] range
    activity_min = jnp.min(activity)
    activity_max = jnp.max(activity)
    
    # Avoid division by zero with stricter tolerance
    activity_range = activity_max - activity_min
    activity_normalized = jax.lax.cond(
        activity_range > 1e-6,  # Even more robust tolerance
        lambda: 2 * (activity - activity_min) / activity_range - 1,
        lambda: jnp.zeros_like(activity)
    )
    
    # Compute autocorrelation
    autocorr = correlate(activity_normalized, activity_normalized, mode='full', method='fft')
    
    # Add enhanced numerical robustness by rounding autocorrelation results
    autocorr = jnp.round(autocorr * 1e10) / 1e10  # Round to 10 decimal places for enhanced consistency
    
    # Clamp autocorrelation to reasonable range
    autocorr = jnp.clip(autocorr, -1e6, 1e6)
    
    # Normalize autocorrelation with stricter tolerance
    autocorr_max = jnp.max(jnp.abs(autocorr))
    autocorr = jax.lax.cond(
        autocorr_max > 1e-6,  # Even more robust tolerance
        lambda: autocorr / autocorr_max,
        lambda: autocorr
    )
    
    # Take the positive lag half
    mid_point = autocorr.shape[-1] // 2
    autocorr = autocorr[mid_point:]
    
    # Find peaks
    peak_indices, peak_heights, peak_prominences, peak_mask = find_peaks_1d(
        autocorr, prominence_threshold=prominence
    )
    
    # Only consider valid peaks (excluding the zero-lag peak at index 0)
    valid_peak_mask = peak_mask & (peak_indices > 0)
    
    # If no valid peaks found, return 0
    has_valid_peaks = jnp.any(valid_peak_mask)
    
    def compute_with_peaks():
        valid_heights = jnp.where(valid_peak_mask, peak_heights, 0)
        valid_prominences = jnp.where(valid_peak_mask, peak_prominences, 0)
        valid_indices = jnp.where(valid_peak_mask, peak_indices, jnp.inf)
        
        max_height = jnp.max(valid_heights)
        max_prominence = jnp.max(valid_prominences)
        score = jnp.minimum(max_height, max_prominence)
        
        # Add enhanced numerical robustness by rounding score
        score = jnp.round(score * 1e10) / 1e10  # Round to 10 decimal places for enhanced consistency
        
        # Clamp score to reasonable range
        score = jnp.clip(score, 0.0, 1e6)
        
        # Find the peak with maximum prominence for frequency calculation
        best_peak_idx = jnp.argmax(jnp.where(valid_peak_mask, peak_prominences, -jnp.inf))
        frequency = 1.0 / peak_indices[best_peak_idx]
        
        # Add enhanced numerical robustness by rounding frequency
        frequency = jnp.round(frequency * 1e10) / 1e10  # Round to 10 decimal places for enhanced consistency
        
        # Clamp frequency to reasonable range
        frequency = jnp.clip(frequency, 1e-6, 1e6)
        
        return score, frequency
    
    def no_peaks():
        return 0.0, 0.0
    
    return jax.lax.cond(has_valid_peaks, compute_with_peaks, no_peaks)

@jit  
def neuron_oscillation_score(activity, prominence=0.05):
    """JAX-compatible neuron oscillation score calculation."""
    
    raw_score, frequency = neuron_oscillation_score_helper_jax(activity, prominence)
    
    # If no oscillation detected, return 0
    def compute_normalized_score():
        n = len(activity)
        t = jnp.arange(n, dtype=jnp.float32)
        
        # Reference sine and cosine waves
        ref_sin = jnp.sin(2 * jnp.pi * frequency * t)
        ref_cos = jnp.cos(2 * jnp.pi * frequency * t)
        
        ref_sin_score, _ = neuron_oscillation_score_helper_jax(ref_sin, prominence)
        ref_cos_score, _ = neuron_oscillation_score_helper_jax(ref_cos, prominence)
        
        ref_score = jnp.maximum(ref_sin_score, ref_cos_score)
        
        # Avoid division by zero with robust tolerance
        return jnp.where(
            ref_score > 1e-6,  # Even more robust tolerance
            raw_score / ref_score,
            0.0
        )
    
    # Only normalize if we have a valid raw score and frequency with robust tolerance
    score = jax.lax.cond(
        (raw_score > 1e-6) & jnp.isfinite(frequency),  # Even more robust tolerance
        compute_normalized_score,
        lambda: 0.0
    )
    
    # Apply FINAL ultra-aggressive numerical robustness fix
    # Round the final score to an even coarser precision to eliminate ALL floating-point variations
    # This sacrifices some numerical precision for absolute reproducibility
    score = jnp.round(score * 1e6) / 1e6  # Round to 6 decimal places for MAXIMUM stability
    
    # Clamp final score to ensure it's in reasonable range
    score = jnp.clip(score, 0.0, 1.0)  # Oscillation scores should be between 0 and 1

    return score, frequency
@jax.jit
def compute_oscillation_score(activity, active_mask, prominence=0.05):
    """Compute oscillation scores for all neurons."""
    
    # Compute scores for all neurons at once using vmap
    all_scores, all_frequencies = jax.vmap(
        neuron_oscillation_score, 
        in_axes=(0, None)
    )(activity, prominence)
    
    # Apply mask: set inactive neurons to 0.0
    score_values = jnp.where(active_mask, all_scores, 0.0)
    frequencies = jnp.where(active_mask, all_frequencies, jnp.nan)
    
    # Calculate mean over active neurons only
    num_active = jnp.sum(active_mask)
    
    # Sum scores for active neurons
    active_score_sum = jnp.sum(score_values * active_mask)
    # active_freq_sum = jnp.sum(frequencies * active_mask)
    
    # Calculate means, avoiding division by zero
    oscillation_score = jnp.where(
        num_active > 0,
        active_score_sum / num_active,
        0.0
    )
    mean_frequency = jnp.where(
        num_active > 0,
        #active_freq_sum / num_active,
        # jnp.nanmean(jnp.where(jnp.isinf(frequencies),jnp.nan,frequencies)), # I think this is better? Will have to test
        jnp.nanmean(jnp.where((jnp.isinf(frequencies) | (frequencies==0.0)),jnp.nan,frequencies)),
        0.0
    )
    
    return oscillation_score, mean_frequency

def extract_nth_filtered_pytree(pytree, n, path_filter, key_list=['tau', 'threshold', 'a', 'fr_cap', 'W_mask']):
    """
    path_filter: function that takes key_path tuple and returns True/False
    """
    def process_leaf(key_path, leaf):
        if path_filter(key_path, key_list):
            return leaf[n:n+1]
        return leaf

    return jax.tree.map_with_path(process_leaf, pytree)

def path_filter(key_path, key_list):
    """
    Filter function that checks if the key_path contains specific keys.
    """
    return key_path[0].name in key_list and key_path != []

def load_W(wPath):
    wExt = os.path.splitext(wPath)[1]

    if wExt == ".npy":
        W = jnp.load(wPath)
    elif wExt == ".csv":
        W = pd.read_csv(wPath).drop(columns="bodyId_pre").to_numpy().astype(float)
    else:
        raise ValueError("Cannot read W file type.")
        # TODO add more file types that can be read

    return W

def load_from_yaml(yamlPath):
    """load data from .yml file path"""
    with open(yamlPath,'r') as file:
        data = yaml.safe_load(file)
    return data

def load_wTable(dfPath):
    dfExt = os.path.splitext(dfPath)[1]
    if dfExt == ".pkl":
        wTable = pd.read_pickle(dfPath)
    elif dfExt == ".csv":
        wTable = pd.read_csv(dfPath, index_col=0)
    else:
        raise ValueError("Cannot read wTable file type.")
        # TODO add more file types that can be read

    return wTable


def make_input(nNeurons, stimNeurons, stimI):
    input = jnp.zeros(nNeurons)
    input = input.at[stimNeurons].set(stimI)
    return jnp.array(input)
