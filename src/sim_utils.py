import time

import jax.numpy as jnp
import numpy as np
from jax import vmap, jit
import jax
# from scipy.stats import truncnorm

import os
import pandas as pd
from jax.lax import cond

def sample_trunc_normal(key, mean, stdev, shape):
    """Sample from truncated normal for a single simulation."""
    # Use inverse CDF method for truncated normal
    # Truncation points in original scale
    lower_bound = 0.0  # truncate at 0 (positive values only)
    upper_bound = mean + 100 * stdev  # effectively infinity

    # Convert to standardized coordinates
    a = (lower_bound - mean) / stdev  # left truncation point in standard deviations
    b = (upper_bound - mean) / stdev  # right truncation point in standard deviations

    # Get CDF values at truncation points for standard normal
    cdf_a = jax.scipy.stats.norm.cdf(a)
    cdf_b = jax.scipy.stats.norm.cdf(b)

    # Sample uniform values between the CDF values
    u = jax.random.uniform(key, shape=shape, minval=cdf_a, maxval=cdf_b)

    # Use inverse CDF to get standard normal samples
    z = jax.scipy.stats.norm.ppf(u)

    # Transform to desired mean and standard deviation
    samples = mean + stdev * z

    return samples

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
    """JAX-compatible helper function for oscillation score calculation.
        Assuming batch dimension is the first axis. """
    activity = activity - jnp.min(activity, axis=-1, keepdims=True)
    activity = 2 * activity / jnp.max(activity, axis=-1, keepdims=True) - 1

    autocorr = autocorrelation_1d(activity)
    autocorr = autocorr[...,autocorr.shape[-1]//2:]
    # Apply find_peaks_1d to each row using vmap
    peak_indices, peak_heights, peak_prominences, peak_mask = find_peaks_1d(autocorr, prominence_threshold=prominence)
    peak_indices = jnp.where(peak_mask, peak_indices, 0)
    peak_heights = jnp.where(peak_mask, peak_heights, 0)
    peak_prominences = jnp.where(peak_mask, peak_prominences, 0)
    score = jnp.min(jnp.array([jnp.max(peak_heights,axis=-1), jnp.max(peak_prominences,axis=-1)]), axis=-1)
    frequency = 1 / peak_indices[peak_prominences.argmax(axis=-1)]
    return score, frequency

@jit
def neuron_oscillation_score(activity, prominence=0.05):
    """JAX-compatible neuron oscillation score calculation."""
    
    raw_score, frequency = neuron_oscillation_score_helper_jax(activity, prominence)
    
    # Normalize to sine wave of the same frequency and duration
    def compute_normalized_score():
        n = len(activity)
        t = jnp.arange(n)
        
        # Reference sine and cosine waves
        ref_sin = jnp.sin(2 * jnp.pi * frequency * t)
        ref_cos = jnp.cos(2 * jnp.pi * frequency * t)
        
        ref_sin_score, _ = neuron_oscillation_score_helper_jax(ref_sin, prominence)
        ref_cos_score, _ = neuron_oscillation_score_helper_jax(ref_cos, prominence)
        
        ref_score = jnp.max(jnp.array([ref_sin_score, ref_cos_score]))
        
        return raw_score / ref_score
    
    score = jnp.where(
        raw_score == 0,
        0.0,
        compute_normalized_score()
    )
    
    return score, frequency

@jax.jit
def compute_oscillation_score(activity, active_mask, prominence=0.05):
    # Compute scores for all neurons (will be NaN for inactive ones)
    scores = jax.vmap(
        lambda activity_row, mask: jax.lax.cond(
            mask,
            lambda x: neuron_oscillation_score(x, prominence=prominence),
            lambda x: (jnp.nan, jnp.nan),
            activity_row
        ),
        in_axes=(0, 0)
    )(activity, active_mask)
    score_values, frequencies = scores
    # Compute mean of valid scores
    oscillation_score = jnp.nanmean(score_values)
    frequencies = jnp.nanmean(jnp.where(jnp.isinf(frequencies),jnp.nan,frequencies))
    return oscillation_score, frequencies

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
