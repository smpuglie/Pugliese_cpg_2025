"""
RNN-based implementation of neural rate equations as an alternative to diffeqsolve.
This module provides a pure JAX RNN that mimics the neural dynamics 
originally solved using differential equation solvers.
"""

import jax
import jax.numpy as jnp
from jax import jit, random
from typing import Tuple, Optional


# @jit 
# def run_single_simulation_rnn_simple(
#     W: jnp.ndarray,
#     tau: jnp.ndarray,
#     a: jnp.ndarray,
#     threshold: jnp.ndarray,
#     fr_cap: jnp.ndarray,
#     inputs: jnp.ndarray,
#     noise_stdv: jnp.ndarray,
#     t_axis: jnp.ndarray,
#     T: float,
#     dt: float,
#     pulse_start: float,
#     pulse_end: float,
#     r_tol: float,  # Not used in RNN version, kept for compatibility
#     a_tol: float,  # Not used in RNN version, kept for compatibility  
#     key: jnp.ndarray,
# ) -> jnp.ndarray:
#     """
#     Simplified RNN implementation without Flax - just pure JAX.
#     This is more efficient and easier to integrate as a drop-in replacement.
    
#     Same signature as the original run_single_simulation function.
#     """
#     n_neurons = W.shape[0]
#     n_steps = len(t_axis)
    
#         # Initialize outputs with the correct shape using float32 for memory efficiency
#     n_neurons = W.shape[0]
#     firing_rates = jnp.zeros((n_steps, n_neurons), dtype=jnp.float32)
    
#     # Pregenerate all noise for efficiency
#     # noise_key, rng_key = jax.random.split(rng_key)
#     # all_noise = jax.random.normal(noise_key, shape=(n_steps, W.shape[0]), dtype=jnp.float32)
    
#     # Storage for results
#     rates_history = []
    
#     for step in range(n_steps):
#         t = t_axis[step]
        
#         # Check if pulse is active
#         pulse_active = (t >= pulse_start) & (t <= pulse_end)
        
#         # Apply external input and noise during pulse
#         external_input = jnp.where(
#             pulse_active,
#             inputs, #+ all_noise[step] * noise_stdv,
#             jnp.zeros_like(inputs)
#         )
        
#         # Compute recurrent input
#         recurrent_input = jnp.dot(W, current_rates)
#         total_input = external_input + recurrent_input
        
#         # Apply half-tanh activation function
#         normalized_input = (a / fr_cap) * (total_input - threshold)
#         activation = jnp.maximum(
#             fr_cap * jnp.tanh(normalized_input),
#             0.0
#         )
        
#         # Update rates using Euler integration
#         dR_dt = (activation - current_rates) / tau
#         current_rates = current_rates + dt * dR_dt
        
#         # Clip for numerical stability
#         current_rates = jnp.clip(current_rates, 0.0, 1000.0)
        
#         # Store current rates
#         rates_history.append(current_rates)
    
#     # Convert to array and transpose to match original format [n_neurons, n_timepoints]
#     result = jnp.transpose(jnp.stack(rates_history))
    
#     # Handle potential numerical issues (same as original)
#     result = jnp.where(jnp.isinf(result), 0.0, result)
#     result = jnp.where(jnp.isnan(result), 0.0, result) 
#     result = jnp.clip(result, 0.0, 1000.0)
    
#     return result


@jit
def run_single_simulation_rnn_scan(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    noise_stdv: jnp.ndarray,
    t_axis: jnp.ndarray,
    T: float,
    dt: float,
    pulse_start: float,
    pulse_end: float,
    r_tol: float,  # Not used, kept for compatibility
    a_tol: float,  # Not used, kept for compatibility
    key: jnp.ndarray,
) -> jnp.ndarray:
    """
    RNN implementation with improved accuracy using smaller effective integration steps.
    
    This version addresses your question: it computes with higher accuracy (smaller effective dt)
    but outputs results at the exact t_axis time points you specify.
    
    Strategy: Use a fixed number of sub-steps (10) for each t_axis interval to improve accuracy.
    """
    n_steps = len(t_axis)

    # Use 25 sub-steps for each t_axis step - this gives us ~25x finer temporal resolution
    n_substeps = 25  # Fixed for JIT compatibility
    
    # Pre-generate all noise in a more memory-efficient way
    # Use float32 instead of float64 to halve memory usage
    all_noise = random.normal(key, shape=(n_steps, W.shape[0]), dtype=jnp.float32)
    
    # Ensure W matrix is float32 for consistent precision and memory efficiency
    W = jnp.asarray(W, dtype=jnp.float32)
    
    def step_fn(carry, inputs_at_t):
        current_rates = carry
        t, noise_at_t = inputs_at_t
        
        # Calculate sub-step size 
        sub_dt = dt / n_substeps
        
        def sub_step(rates, sub_idx):
            # Check if pulse is active at current time
            pulse_active = (t >= pulse_start) & (t <= pulse_end)
            
            # More memory-efficient computation - avoid intermediate arrays
            # Scale noise appropriately for sub-steps (precompute scaling factor)
            noise_scale = 1.0 / jnp.sqrt(n_substeps)
            
            # Compute total input in one step to reduce memory allocations
            recurrent_input = jnp.dot(W, rates)
            total_input = jnp.where(
                pulse_active,
                recurrent_input + inputs + noise_at_t * noise_stdv * noise_scale,
                recurrent_input
            )
            
            # Apply half-tanh activation with optimized memory usage
            normalized_input = (a / fr_cap) * (total_input - threshold)
            activation = jnp.maximum(
                fr_cap * jnp.tanh(normalized_input),
                0.0
            )
            
            # Update rates using Euler integration with smaller sub_dt
            # Combine operations to reduce intermediate arrays
            new_rates = jnp.clip(
                rates + (sub_dt / tau) * (activation - rates),
                0.0, 1000.0
            )
            
            return new_rates, None
        
        # Perform sub-steps for this time point (20x higher resolution)
        final_rates, _ = jax.lax.scan(sub_step, current_rates, jnp.arange(n_substeps))
        
        return final_rates, final_rates
    
    # Initialize with zeros using float32 for memory efficiency
    initial_rates = jnp.zeros(W.shape[0], dtype=jnp.float32)
    
    # Run simulation using scan
    _, rates_history = jax.lax.scan(
        step_fn, 
        initial_rates, 
        (t_axis, all_noise)
    )
    
    # Transpose to match expected format [n_neurons, n_timepoints]
    result = rates_history.T
    
    # Handle numerical issues
    result = jnp.where(jnp.isinf(result), 0.0, result)
    result = jnp.where(jnp.isnan(result), 0.0, result)
    result = jnp.clip(result, 0.0, 1000.0)
    
    return result


@jit
def _run_single_simulation_rnn_dual_dt_static(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    noise_stdv: jnp.ndarray,
    t_axis: jnp.ndarray,
    T: float,
    dt: float,
    pulse_start: float,
    pulse_end: float,
    r_tol: float,
    a_tol: float,
    key: jnp.ndarray,
    n_substeps: int,  # Static parameter for JIT
) -> jnp.ndarray:
    """
    Static JIT-compiled helper for run_single_simulation_rnn_dual_dt.
    This version has n_substeps as a static parameter for optimal JIT compilation.
    """
    n_steps = len(t_axis)
    
    # Pre-generate all noise
    all_noise = random.normal(key, shape=(n_steps, W.shape[0]))
    
    def step_fn(carry, inputs_at_t):
        current_rates = carry
        t, noise_at_t = inputs_at_t
        
        # Calculate sub-step size 
        sub_dt = dt / n_substeps
        
        def sub_step(rates, sub_idx):
            # Only perform computation if within the required n_substeps
            should_compute = sub_idx < n_substeps
            
            def do_computation():
                # Check if pulse is active at current time
                pulse_active = (t >= pulse_start) & (t <= pulse_end)
                
                # Apply external input and noise during pulse
                # Scale noise appropriately for sub-steps
                external_input = jnp.where(
                    pulse_active,
                    inputs + noise_at_t * noise_stdv / jnp.sqrt(n_substeps),
                    jnp.zeros_like(inputs)
                )
                
                # Compute recurrent input
                recurrent_input = jnp.dot(W, rates)
                total_input = external_input + recurrent_input
                
                # Apply half-tanh activation
                normalized_input = (a / fr_cap) * (total_input - threshold)
                activation = jnp.maximum(
                    fr_cap * jnp.tanh(normalized_input),
                    0.0
                )
                
                # Update rates using Euler integration with smaller sub_dt
                dR_dt = (activation - rates) / tau
                new_rates = rates + sub_dt * dR_dt
                
                # Clip for numerical stability
                new_rates = jnp.clip(new_rates, 0.0, 1000.0)
                return new_rates
            
            def skip_computation():
                return rates
            
            # Use conditional to only compute when needed
            new_rates = jnp.where(should_compute, do_computation(), skip_computation())
            
            return new_rates, None
        
        # Use fixed maximum substeps for JIT compatibility
        MAX_SUBSTEPS = 50
        final_rates, _ = jax.lax.scan(sub_step, current_rates, jnp.arange(MAX_SUBSTEPS))
        
        return final_rates, final_rates
    
    # Initialize with zeros
    initial_rates = jnp.zeros(W.shape[0])
    
    # Run simulation using scan
    _, rates_history = jax.lax.scan(
        step_fn, 
        initial_rates, 
        (t_axis, all_noise)
    )
    
    # Transpose to match expected format [n_neurons, n_timepoints]
    result = rates_history.T
    
    # Handle numerical issues
    result = jnp.where(jnp.isinf(result), 0.0, result)
    result = jnp.where(jnp.isnan(result), 0.0, result)
    result = jnp.clip(result, 0.0, 1000.0)
    
    return result


def run_single_simulation_rnn_dual_dt(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    noise_stdv: jnp.ndarray,
    t_axis: jnp.ndarray,
    T: float,
    dt: float,
    pulse_start: float,
    pulse_end: float,
    r_tol: float,  # Not used, kept for compatibility
    a_tol: float,  # Not used, kept for compatibility
    key: jnp.ndarray,
    n_substeps: int = 25  # Number of sub-steps per t_axis step
) -> jnp.ndarray:
    """
    RNN implementation with improved accuracy using configurable sub-steps.
    
    This version uses a helper function approach to make n_substeps static for JIT,
    while allowing flexible configuration from the user interface.
    
    Parameters:
    - n_substeps: Number of sub-steps per t_axis step for accuracy (default 25)
                 Higher values = better accuracy but slower computation
    """
    # Call the static JIT-compiled helper function
    return _run_single_simulation_rnn_dual_dt_static(
        W, tau, a, threshold, fr_cap, inputs, noise_stdv,
        t_axis, T, dt, pulse_start, pulse_end, r_tol, a_tol, key,
        n_substeps
    )


# Export the recommended implementations
run_single_simulation_rnn_optimized = run_single_simulation_rnn_scan  # Fast, fixed 20 substeps - RECOMMENDED
run_single_simulation_rnn_flexible = run_single_simulation_rnn_dual_dt  # Slower due to MAX_SUBSTEPS=50 overhead
# run_single_simulation_rnn = run_single_simulation_rnn_simple  # Simple version for debugging

