# Combined JAX/JIT VNC Network Simulation - Unified Implementation
import time
import jax
import jax.numpy as jnp
import numpy as np
import psutil
import gc
from typing import NamedTuple, Dict, Any, Optional, Tuple, List, Union
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve, Event
from jax import vmap, jit, random, pmap
from omegaconf import DictConfig
from pathlib import Path

# Import utility functions (assumed to exist)
from src.shuffle_utils import extract_shuffle_indices, full_shuffle
from src.sim_utils import (
    load_W, load_wTable, sample_trunc_normal, set_sizes, make_input,
    compute_oscillation_score
)
from src.data_classes import (
    NeuronParams, SimParams, SimulationConfig, Pruning_state, CheckpointState
)
from src.checkpointing import (
    load_checkpoint, save_checkpoint, find_latest_checkpoint, cleanup_old_checkpoints
)
from src.adaptive_memory import (
    aggressive_memory_cleanup, monitor_memory_usage, should_trigger_cleanup,
    get_conservative_batch_size, log_memory_status, create_memory_manager, 
    optimize_for_performance_vs_stability
)

# #############################################################################################################################=
# CORE SIMULATION FUNCTIONS - ALL JIT COMPILED
# #############################################################################################################################=

@jit
def rate_equation_half_tanh(t, R, args):
    """ODE rate equation with half-tanh activation."""
    inputs, pulse_start, pulse_end, tau, weighted_W, threshold, a, fr_cap, key, noise_stdv = args
    pulse_active = (t >= pulse_start) & (t <= pulse_end)
    I = inputs * pulse_active + pulse_active * jax.random.normal(key, shape=inputs.shape) * noise_stdv
    total_input = I + jnp.dot(weighted_W, R)
    activation = jnp.maximum(
        fr_cap * jnp.tanh((a / fr_cap) * (total_input - threshold)), 0
    )
    return (activation - R) / tau


@jit
def reweight_connectivity(W: jnp.ndarray, exc_mult: float, inh_mult: float) -> jnp.ndarray:
    """Reweight connectivity matrix with multipliers."""
    Wt = jnp.transpose(W)
    W_exc = jnp.maximum(Wt, 0)
    W_inh = jnp.minimum(Wt, 0)
    return exc_mult * W_exc + inh_mult * W_inh


@jit
def add_noise_to_weights(W: jnp.ndarray, key: jnp.ndarray, stdv_prop: float) -> jnp.ndarray:
    """Add truncated normal noise to weight matrix."""
    stdvs = W * stdv_prop
    noise = sample_trunc_normal(key, mean=0.0, stdev=stdv_prop, shape=W.shape)
    return W + stdvs * noise



def nan_inf_event_func(t, y, args, **kwargs):
    """Event function to detect NaNs or infs in the solution."""
    # Returns 0 when NaN or inf is detected, triggering the event
    has_nan = jnp.any(jnp.isnan(y))
    has_inf = jnp.any(jnp.isinf(y))
    # Return a negative value when NaN/inf detected to trigger the event
    return jnp.where(has_nan | has_inf, -1.0, 1.0)


@jit
def run_single_simulation(
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
) -> jnp.ndarray:
    """Run a single simulation with given parameters."""
    R0 = jnp.zeros(W.shape[1])
    term = ODETerm(rate_equation_half_tanh)
    solver = Dopri5()
    saveat = SaveAt(ts=t_axis)
    
    # Use more conservative tolerances for numerical stability
    # Especially important for multi-GPU setups where different devices may have different precision
    safe_rtol = jnp.maximum(r_tol, 1e-6)  # Don't go below 1e-6
    safe_atol = jnp.maximum(a_tol, 1e-8)  # Don't go below 1e-8
    controller = PIDController(rtol=safe_rtol, atol=safe_atol)
    
    # Create event to detect NaN/inf and stop early
    nan_inf_event = Event(nan_inf_event_func)

    solution = diffeqsolve(
        term, solver, 0, T, dt, R0,
        args=(inputs, pulse_start, pulse_end, tau, W, threshold, a, fr_cap, key, noise_stdv),
        saveat=saveat, stepsize_controller=controller,
        event=nan_inf_event,
        max_steps=100000, throw=False
    )
    
    # Check for numerical issues in the solution
    result = jnp.transpose(solution.ys)
    
    # Replace inf/nan with zeros and clip to reasonable bounds
    # This approach is JIT-compatible
    result = jnp.where(jnp.isinf(result), 0.0, result)
    result = jnp.where(jnp.isnan(result), 0.0, result)
    # Clip to reasonable firing rate bounds
    result = jnp.clip(result, 0.0, 1000.0)  # Max 1000 Hz
    
    return result


# #############################################################################################################################=
# SIMULATION TYPE FUNCTIONS
# #############################################################################################################################=

@jit
def run_with_shuffle(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    seed: jnp.ndarray,
    shuffle_indices: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    sim_params: SimParams
) -> jnp.ndarray:
    """Run simulation with shuffled connectivity."""
    exc_dn, inh_dn, exc_in, inh_in, mn = shuffle_indices
    W_shuffled = full_shuffle(W, seed, exc_dn, inh_dn, exc_in, inh_in, mn)
    W_reweighted = reweight_connectivity(W_shuffled, sim_params.exc_multiplier, sim_params.inh_multiplier)
    
    return run_single_simulation(
        W_reweighted, tau, a, threshold, fr_cap, inputs, sim_params.noise_stdv,
        sim_params.t_axis, sim_params.T, sim_params.dt,
        sim_params.pulse_start, sim_params.pulse_end,
        sim_params.r_tol, sim_params.a_tol, seed,
    )


@jit
def run_with_noise(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    seed: jnp.ndarray,
    noise_stdv_prop: float,
    sim_params: SimParams
) -> jnp.ndarray:
    """Run simulation with noisy connectivity."""
    W_noisy = add_noise_to_weights(W, seed, noise_stdv_prop)
    W_reweighted = reweight_connectivity(W_noisy, sim_params.exc_multiplier, sim_params.inh_multiplier)
    
    return run_single_simulation(
        W_reweighted, tau, a, threshold, fr_cap, inputs, sim_params.noise_stdv,
        sim_params.t_axis, sim_params.T, sim_params.dt,
        sim_params.pulse_start, sim_params.pulse_end,
        sim_params.r_tol, sim_params.a_tol, seed
    )


@jit
def run_baseline(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    sim_params: SimParams,
    seed: jnp.ndarray
) -> jnp.ndarray:
    """Run baseline simulation without shuffle or noise."""
    W_reweighted = reweight_connectivity(W, sim_params.exc_multiplier, sim_params.inh_multiplier)
    
    return run_single_simulation(
        W_reweighted, tau, a, threshold, fr_cap, inputs, sim_params.noise_stdv,
        sim_params.t_axis, sim_params.T, sim_params.dt,
        sim_params.pulse_start, sim_params.pulse_end,
        sim_params.r_tol, sim_params.a_tol, seed,
    )


@jit
def run_with_Wmask(
    W: jnp.ndarray,
    W_mask: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    sim_params: SimParams,
    seed: jnp.ndarray
) -> jnp.ndarray:
    """Run pruning simulation."""
    W_masked = W * W_mask
    W_reweighted = reweight_connectivity(W_masked, sim_params.exc_multiplier, sim_params.inh_multiplier)
    
    return run_single_simulation(
        W_reweighted, tau, a, threshold, fr_cap, inputs, sim_params.noise_stdv,
        sim_params.t_axis, sim_params.T, sim_params.dt,
        sim_params.pulse_start, sim_params.pulse_end,
        sim_params.r_tol, sim_params.a_tol, seed
    )


# #############################################################################################################################=
# PRUNING-SPECIFIC FUNCTIONS
# #############################################################################################################################=

@jax.jit
def removal_probability(max_frs, neurons_to_exclude_mask):
    """Assign a probability of removal to each neuron based on their max FR"""
    tmp = 1.0 / jnp.where(max_frs > 0, max_frs, 1.0)  # Avoid division by zero
    tmp = jnp.where(jnp.isfinite(tmp), tmp, 0.0)
    tmp = jnp.where(neurons_to_exclude_mask, 0.0, tmp)
    
    # Normalize probabilities
    p_sum = jnp.sum(tmp)
    p = jnp.where(p_sum > 0, tmp / p_sum, 0.0)
    
    return p


@jax.jit
def jax_choice(key, a, p):
    """JAX-compatible choice function"""
    # Handle edge case where all probabilities are zero
    p_sum = jnp.sum(p)
    p_normalized = jnp.where(p_sum > 0, p, jnp.ones_like(p) / len(p))
    
    # Use categorical sampling
    return random.categorical(key, jnp.log(p_normalized + 1e-10))

def update_single_sim_state(state, R, mn_mask, oscillation_threshold=0.5, clip_start=250):
    """Update state for a single simulation with proper round-based convergence logic"""
    
    # Unpack state
    (W_mask, interneuron_mask, level, total_removed_neurons, removed_stim_neurons,
        neurons_put_back, prev_put_back, last_removed, remove_p, min_circuit, key) = state

    # Get active MN activity using JAX-compatible approach
    max_frs = jnp.max(R[..., clip_start:], axis=-1)
    active_mask = ((max_frs > 0.01) & mn_mask)
    # Compute oscillation score
    oscillation_score, _ = compute_oscillation_score(R[..., clip_start:], active_mask, prominence=0.05)

    # Check current availability to determine if we need a new round
    current_exclude_mask = (~interneuron_mask) | total_removed_neurons | neurons_put_back
    current_available = jnp.sum(~current_exclude_mask)
    p_current = removal_probability(max_frs, current_exclude_mask)
    p_sum_current = jnp.sum(p_current)
    
    # If no neurons left to remove or all probabilities are zero, we need a new round
    need_new_round = (current_available <= 0) | (~jnp.isfinite(p_sum_current))
    
    # CHECK FOR CONVERGENCE AT THE BEGINNING
    # Convergence happens when we need a new round AND the put_back sets are equal
    # BUT only if we've actually been through at least one round (level > 0)
    sets_equal = jnp.array_equal(neurons_put_back, prev_put_back)
    has_done_meaningful_work = (level[0] > 0) | (jnp.sum(neurons_put_back) > 0) | (jnp.sum(total_removed_neurons) > 0)
    currently_converged = (need_new_round & sets_equal & has_done_meaningful_work) | min_circuit.squeeze()
    
    # Split key for potential use
    key_next, subkey_continue, subkey_reset = random.split(key, 3)

    # Check if oscillation is below threshold or NaN
    reset_condition = (oscillation_score < oscillation_threshold) | jnp.isnan(oscillation_score)

    ##### ALWAYS COMPUTE BOTH BRANCHES (for vmap compatibility) #####
    
    ##### CONTINUE BRANCH: Normal pruning (oscillation is good) #####
    # Identify currently silent interneurons - these are permanently removed
    silent_interneurons = interneuron_mask & (max_frs <= 0)
    newly_silent = silent_interneurons & (~total_removed_neurons)
    total_removed_continue = total_removed_neurons | newly_silent
    
    # Update probabilities after removing silent neurons
    exclude_mask_continue = (~interneuron_mask) | total_removed_continue | neurons_put_back
    p_continue = removal_probability(max_frs, exclude_mask_continue)
    available_neurons_continue = jnp.sum(~exclude_mask_continue)
    p_sum_continue = jnp.sum(p_continue)
    
    # Check if we need new round after silent neuron removal
    need_new_round_continue = need_new_round | (available_neurons_continue <= 0) | (~jnp.isfinite(p_sum_continue))
    
    # Select new neuron to remove (if available and not starting new round)
    neuron_idx_continue = jax_choice(subkey_continue, jnp.arange(len(max_frs)), p_continue)
    
    # Update removed neurons (only if not starting new round)
    removed_stim_continue = jax.lax.select(
        need_new_round_continue,
        removed_stim_neurons,
        removed_stim_neurons.at[neuron_idx_continue].set(True)
    )
    total_removed_continue = jax.lax.select(
        need_new_round_continue,
        total_removed_continue,
        total_removed_continue.at[neuron_idx_continue].set(True)
    )
    
    # Track what was removed this iteration
    last_removed_continue = newly_silent  # Always include newly silent neurons
    last_removed_continue = jax.lax.select(
        need_new_round_continue,
        last_removed_continue,
        last_removed_continue.at[neuron_idx_continue].set(True)
    )
    
    # Handle new round transition for continue branch
    neurons_put_back_continue = jax.lax.select(
        need_new_round_continue,
        jnp.full(neurons_put_back.shape, False, dtype=jnp.bool_),  # Reset put_back
        neurons_put_back
    )
    prev_put_back_continue = jax.lax.select(
        need_new_round_continue,
        neurons_put_back,  # Current becomes previous
        prev_put_back
    )
    level_continue = jax.lax.select(need_new_round_continue, level + 1, level)

    ##### RESET BRANCH: Restore last removed neuron and try again #####
    # Only restore the stimulated neuron from last_removed (silent neurons stay removed)
    stimulated_last_removed = last_removed & removed_stim_neurons
    
    # Restore only stimulated neurons
    removed_stim_reset = removed_stim_neurons & (~stimulated_last_removed)
    total_removed_reset = total_removed_neurons & (~stimulated_last_removed)
    
    # Add the restored neuron to neurons_put_back
    neurons_put_back_reset = neurons_put_back | stimulated_last_removed
    
    # Now select a different neuron to remove (exclude restored and put back neurons)
    exclude_mask_reset = (~interneuron_mask) | total_removed_reset | neurons_put_back_reset
    p_reset = removal_probability(max_frs, exclude_mask_reset)
    available_neurons_reset = jnp.sum(~exclude_mask_reset)
    p_sum_reset = jnp.sum(p_reset)
    
    # Check if we need new round in reset branch
    need_new_round_reset = (available_neurons_reset <= 0) | (~jnp.isfinite(p_sum_reset))
    
    # Select neuron to remove (if available)
    neuron_idx_reset = jax_choice(subkey_reset, jnp.arange(len(max_frs)), p_reset)
    
    # Update removed neurons (only if not starting new round)
    removed_stim_reset = jax.lax.select(
        need_new_round_reset,
        removed_stim_reset,
        removed_stim_reset.at[neuron_idx_reset].set(True)
    )
    total_removed_reset = jax.lax.select(
        need_new_round_reset,
        total_removed_reset,
        total_removed_reset.at[neuron_idx_reset].set(True)
    )
    
    # Track what was newly removed
    last_removed_reset = jax.lax.select(
        need_new_round_reset,
        jnp.full(last_removed.shape, False, dtype=jnp.bool_),
        jnp.full(last_removed.shape, False, dtype=jnp.bool_).at[neuron_idx_reset].set(True)
    )
    
    # Handle new round logic for reset branch
    neurons_put_back_reset = jax.lax.select(
        need_new_round_reset,
        jnp.full(neurons_put_back.shape, False, dtype=jnp.bool_),  # Reset put_back
        neurons_put_back_reset
    )
    prev_put_back_reset = jax.lax.select(
        need_new_round_reset,
        neurons_put_back,  # Current becomes previous
        prev_put_back
    )
    level_reset = jax.lax.select(need_new_round_reset, level + 1, level)

    ##### SELECT BETWEEN BRANCHES #####
    branch_total_removed = jax.lax.select(reset_condition, total_removed_reset, total_removed_continue)
    branch_removed_stim = jax.lax.select(reset_condition, removed_stim_reset, removed_stim_continue)
    branch_last_removed = jax.lax.select(reset_condition, last_removed_reset, last_removed_continue)
    branch_neurons_put_back = jax.lax.select(reset_condition, neurons_put_back_reset, neurons_put_back_continue)
    branch_prev_put_back = jax.lax.select(reset_condition, prev_put_back_reset, prev_put_back_continue)
    branch_level = jax.lax.select(reset_condition, level_reset, level_continue)
    branch_p = jax.lax.select(reset_condition, p_reset, p_continue)

    # Update W_mask to reflect removed neurons
    W_mask_init = jnp.ones_like(W_mask, dtype=jnp.bool_)
    active_neurons = ~branch_total_removed
    W_mask_new = (W_mask_init * active_neurons[:, None] * active_neurons[None, :]).astype(jnp.bool_)

    ##### FINAL SELECTION: CURRENT STATE (if converged) vs NEW STATE (if not converged) #####
    # This is the key fix: when converged, select ALL current state values
    final_total_removed = jax.lax.select(currently_converged, total_removed_neurons, branch_total_removed)
    final_removed_stim = jax.lax.select(currently_converged, removed_stim_neurons, branch_removed_stim)
    final_last_removed = jax.lax.select(currently_converged, last_removed, branch_last_removed)
    final_neurons_put_back = jax.lax.select(currently_converged, neurons_put_back, branch_neurons_put_back)
    final_prev_put_back = jax.lax.select(currently_converged, prev_put_back, branch_prev_put_back)
    # Ensure level never decreases - use max of current and branch levels
    final_level = jnp.maximum(level, branch_level)
    final_p = jax.lax.select(currently_converged, remove_p, branch_p)
    final_W_mask = jax.lax.select(currently_converged, W_mask, W_mask_new)
    final_key = jax.lax.select(currently_converged, key, key_next)

    # Debug information
    # jax.debug.print("Level: {level}, Converged: {conv}, Osc: {score}, NewRound: {nr}, SetsEq: {eq}", 
    #                level=final_level,
    #                conv=currently_converged, 
    #                score=oscillation_score,
    #                nr=need_new_round,
    #                eq=sets_equal)

    # Return the final state
    return Pruning_state(
        W_mask=final_W_mask,
        interneuron_mask=interneuron_mask,
        level=final_level,
        total_removed_neurons=final_total_removed,
        neurons_put_back=final_neurons_put_back,
        prev_put_back=final_prev_put_back,
        removed_stim_neurons=final_removed_stim,
        last_removed=final_last_removed,
        remove_p=final_p,
        min_circuit=currently_converged,
        keys=final_key,
    )


# #############################################################################################################################=
# PRUNING STATE MANAGEMENT FUNCTIONS
# #############################################################################################################################=

def reshape_state_for_pmap(state: Pruning_state, n_devices: int) -> Pruning_state:
    """Reshape state for pmap distribution across devices"""
    batch_size = state.W_mask.shape[0]
    batch_per_device = batch_size // n_devices
    
    if batch_size % n_devices != 0:
        raise ValueError(f"Batch size {batch_size} must be divisible by number of devices {n_devices}")
    
    return Pruning_state(
        W_mask=state.W_mask.reshape(n_devices, batch_per_device, *state.W_mask.shape[1:]),
        interneuron_mask=state.interneuron_mask.reshape(n_devices, batch_per_device, *state.interneuron_mask.shape[1:]),
        level=state.level.reshape(n_devices, batch_per_device, *state.level.shape[1:]),
        total_removed_neurons=state.total_removed_neurons.reshape(n_devices, batch_per_device, *state.total_removed_neurons.shape[1:]),
        removed_stim_neurons=state.removed_stim_neurons.reshape(n_devices, batch_per_device, *state.removed_stim_neurons.shape[1:]),
        neurons_put_back=state.neurons_put_back.reshape(n_devices, batch_per_device, *state.neurons_put_back.shape[1:]),
        prev_put_back=state.prev_put_back.reshape(n_devices, batch_per_device, *state.prev_put_back.shape[1:]),
        last_removed=state.last_removed.reshape(n_devices, batch_per_device, *state.last_removed.shape[1:]),
        remove_p=state.remove_p.reshape(n_devices, batch_per_device, *state.remove_p.shape[1:]),
        min_circuit=state.min_circuit.reshape(n_devices, batch_per_device, *state.min_circuit.shape[1:]),
        keys=state.keys.reshape(n_devices, batch_per_device, *state.keys.shape[1:])
    )


def reshape_state_from_pmap(state: Pruning_state) -> Pruning_state:
    """Reshape state back from pmap format to flat format"""
    n_devices, batch_per_device = state.W_mask.shape[:2]
    total_batch = n_devices * batch_per_device
    
    return Pruning_state(
        W_mask=state.W_mask.reshape(total_batch, *state.W_mask.shape[2:]),
        interneuron_mask=state.interneuron_mask.reshape(total_batch, *state.interneuron_mask.shape[2:]),
        level=state.level.reshape(total_batch, *state.level.shape[2:]),
        total_removed_neurons=state.total_removed_neurons.reshape(total_batch, *state.total_removed_neurons.shape[2:]),
        removed_stim_neurons=state.removed_stim_neurons.reshape(total_batch, *state.removed_stim_neurons.shape[2:]),
        neurons_put_back=state.neurons_put_back.reshape(total_batch, *state.neurons_put_back.shape[2:]),
        prev_put_back=state.prev_put_back.reshape(total_batch, *state.prev_put_back.shape[2:]),
        last_removed=state.last_removed.reshape(total_batch, *state.last_removed.shape[2:]),
        remove_p=state.remove_p.reshape(total_batch, *state.remove_p.shape[2:]),
        min_circuit=state.min_circuit.reshape(total_batch, *state.min_circuit.shape[2:]),
        keys=state.keys.reshape(total_batch, *state.keys.shape[2:])
    )



def pad_state_for_devices(state: Pruning_state, target_batch_size: int) -> Pruning_state:
    """Pad state to match target batch size with zeros/defaults"""
    current_batch_size = state.W_mask.shape[0]
    if current_batch_size >= target_batch_size:
        return state
    
    pad_size = target_batch_size - current_batch_size
    
    # Create zero/default padding for each field
    W_mask_pad = jnp.zeros((pad_size, *state.W_mask.shape[1:]), dtype=state.W_mask.dtype)
    interneuron_mask_pad = jnp.zeros((pad_size, *state.interneuron_mask.shape[1:]), dtype=state.interneuron_mask.dtype)
    level_pad = jnp.zeros((pad_size, *state.level.shape[1:]), dtype=state.level.dtype)
    total_removed_neurons_pad = jnp.zeros((pad_size, *state.total_removed_neurons.shape[1:]), dtype=state.total_removed_neurons.dtype)
    removed_stim_neurons_pad = jnp.zeros((pad_size, *state.removed_stim_neurons.shape[1:]), dtype=state.removed_stim_neurons.dtype)
    neurons_put_back_pad = jnp.zeros((pad_size, *state.neurons_put_back.shape[1:]), dtype=state.neurons_put_back.dtype)
    prev_put_back_pad = jnp.zeros((pad_size, *state.prev_put_back.shape[1:]), dtype=state.prev_put_back.dtype)
    last_removed_pad = jnp.zeros((pad_size, *state.last_removed.shape[1:]), dtype=state.last_removed.dtype)
    remove_p_pad = jnp.zeros((pad_size, *state.remove_p.shape[1:]), dtype=state.remove_p.dtype)
    min_circuit_pad = jnp.zeros((pad_size, *state.min_circuit.shape[1:]), dtype=state.min_circuit.dtype)
    keys_pad = jnp.zeros((pad_size, *state.keys.shape[1:]), dtype=state.keys.dtype)
    
    return Pruning_state(
        W_mask=jnp.concatenate([state.W_mask, W_mask_pad], axis=0),
        interneuron_mask=jnp.concatenate([state.interneuron_mask, interneuron_mask_pad], axis=0),
        level=jnp.concatenate([state.level, level_pad], axis=0),
        total_removed_neurons=jnp.concatenate([state.total_removed_neurons, total_removed_neurons_pad], axis=0),
        removed_stim_neurons=jnp.concatenate([state.removed_stim_neurons, removed_stim_neurons_pad], axis=0),
        neurons_put_back=jnp.concatenate([state.neurons_put_back, neurons_put_back_pad], axis=0),
        prev_put_back=jnp.concatenate([state.prev_put_back, prev_put_back_pad], axis=0),
        last_removed=jnp.concatenate([state.last_removed, last_removed_pad], axis=0),
        remove_p=jnp.concatenate([state.remove_p, remove_p_pad], axis=0),
        min_circuit=jnp.concatenate([state.min_circuit, min_circuit_pad], axis=0),
        keys=jnp.concatenate([state.keys, keys_pad], axis=0)
    )

def trim_state_padding(state: Pruning_state, actual_batch_size: int) -> Pruning_state:
    """Remove padding from state to get back to actual batch size"""
    return Pruning_state(
        W_mask=state.W_mask[:actual_batch_size],
        interneuron_mask=state.interneuron_mask[:actual_batch_size],
        level=state.level[:actual_batch_size],
        total_removed_neurons=state.total_removed_neurons[:actual_batch_size],
        removed_stim_neurons=state.removed_stim_neurons[:actual_batch_size],
        neurons_put_back=state.neurons_put_back[:actual_batch_size],
        prev_put_back=state.prev_put_back[:actual_batch_size],
        last_removed=state.last_removed[:actual_batch_size],
        remove_p=state.remove_p[:actual_batch_size],
        min_circuit=state.min_circuit[:actual_batch_size],
        keys=state.keys[:actual_batch_size]
    )


# #############################################################################################################################=
# VECTORIZED BATCH PROCESSING FUNCTIONS
# #############################################################################################################################=

@jit
def process_batch_Wmask(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    batch_indices: jnp.ndarray
) -> jnp.ndarray:
    """Process a batch of simulations with pruning."""
    def single_sim(idx):
        param_idx = idx % sim_params.n_param_sets
        stim_idx = idx // sim_params.n_param_sets
        return run_with_Wmask(
            neuron_params.W,
            neuron_params.W_mask[param_idx],
            neuron_params.tau[param_idx],
            neuron_params.a[param_idx],
            neuron_params.threshold[param_idx],
            neuron_params.fr_cap[param_idx],
            neuron_params.input_currents[stim_idx, param_idx],
            sim_params,
            neuron_params.seeds[param_idx]
        )
    return vmap(single_sim)(batch_indices)


@jit
def process_batch_shuffle(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    batch_indices: jnp.ndarray
) -> jnp.ndarray:
    """Process a batch of simulations with shuffling."""
    def single_sim(idx):
        param_idx = idx % sim_params.n_param_sets
        stim_idx = idx // sim_params.n_param_sets
        shuffle_indices = (
            neuron_params.exc_dn_idxs, neuron_params.inh_dn_idxs,
            neuron_params.exc_in_idxs, neuron_params.inh_in_idxs,
            neuron_params.mn_idxs
        )
        return run_with_shuffle(
            neuron_params.W,
            neuron_params.tau[param_idx],
            neuron_params.a[param_idx],
            neuron_params.threshold[param_idx],
            neuron_params.fr_cap[param_idx],
            neuron_params.input_currents[stim_idx, param_idx],
            neuron_params.seeds[param_idx],
            shuffle_indices,
            sim_params
        )
    return vmap(single_sim)(batch_indices)


@jit
def process_batch_noise(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    batch_indices: jnp.ndarray
) -> jnp.ndarray:
    """Process a batch of simulations with noise."""
    def single_sim(idx):
        param_idx = idx % sim_params.n_param_sets
        stim_idx = idx // sim_params.n_param_sets
        return run_with_noise(
            neuron_params.W,
            neuron_params.tau[param_idx],
            neuron_params.a[param_idx],
            neuron_params.threshold[param_idx],
            neuron_params.fr_cap[param_idx],
            neuron_params.input_currents[stim_idx, param_idx],
            neuron_params.seeds[param_idx],
            sim_params.noise_stdv_prop,
            sim_params
        )
    return vmap(single_sim)(batch_indices)


@jit
def process_batch_baseline(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    batch_indices: jnp.ndarray
) -> jnp.ndarray:
    """Process a batch of baseline simulations."""
    def single_sim(idx):
        param_idx = idx % sim_params.n_param_sets
        stim_idx = idx // sim_params.n_param_sets
        return run_baseline(
            neuron_params.W,
            neuron_params.tau[param_idx],
            neuron_params.a[param_idx],
            neuron_params.threshold[param_idx],
            neuron_params.fr_cap[param_idx],
            neuron_params.input_currents[stim_idx, param_idx],
            sim_params,
            neuron_params.seeds[param_idx]
        )
    return vmap(single_sim)(batch_indices)


# #############################################################################################################################=
# UNIFIED SIMULATION EXECUTION ENGINE
# #############################################################################################################################=

def get_batch_function(sim_config: SimulationConfig):
    """Get the appropriate batch processing function based on simulation type."""
    batch_functions = {
        "baseline": process_batch_baseline,
        "shuffle": process_batch_shuffle, 
        "noise": process_batch_noise,
        "Wmask": process_batch_Wmask
    }
    
    if sim_config.sim_type not in batch_functions:
        raise ValueError(f"Unknown simulation type: {sim_config.sim_type}")
    
    return batch_functions[sim_config.sim_type]


def calculate_optimal_batch_size(n_neurons: int, n_timepoints: int, n_replicates: int, n_devices: int = None, max_batch_size: int = 256, total_sims: int = None) -> int:
    """
    Calculate optimal batch size based on memory and device constraints.
    
    Args:
        n_neurons: Number of neurons in the model
        n_timepoints: Number of time points per sample
        n_replicates: Total number of replicates (batch dimension)
        n_devices: Number of devices available (defaults to JAX device count)
        max_batch_size: Maximum allowed batch size (default 256)
        total_sims: Total number of simulations (n_sims * n_replicates). If provided,
                   will try to process all simulations in a single batch when feasible.
    
    Returns:
        Optimal batch size that respects memory limits and device constraints
    """
    print("Calculating optimal batch size...")
    if n_devices is None:
        n_devices = jax.device_count()
    
    # Use total_sims as the basis for calculation if provided, otherwise use n_replicates
    if total_sims is not None:
        print(f"Total simulations to process: {total_sims}")
        effective_total = total_sims
    else:
        effective_total = n_replicates
        print(f"Using n_replicates as total: {effective_total}")
    
    # Handle edge case where we have fewer simulations than devices
    if effective_total < n_devices:
        return effective_total
    
    # Get memory information
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available
    total_memory = memory_info.total
    
    print(f"System memory: {total_memory / (1024 ** 3):.2f} GB total, {available_memory / (1024 ** 3):.2f} GB available")
    
    # Try to get GPU memory if available
    gpu_memory = None
    n_gpu_devices = 0
    gpu_memory_per_device = None
    
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        n_gpu_devices = len(gpu_devices)
        
        if n_gpu_devices > 0:
            # Method 1: Try to get actual GPU memory using nvidia-ml-py3 if available
            try:
                import pynvml
                pynvml.nvmlInit()
                
                device_memories = []
                for i in range(n_gpu_devices):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_gb = info.total / (1024 ** 3)
                    device_memories.append(info.total)
                    print(f"GPU {i}: {device_name} - {memory_gb:.1f}GB")
                
                # Use the memory of the first device (assume all are the same)
                gpu_memory_per_device = device_memories[0]
                total_gpu_memory = sum(device_memories)
                
                print(f"Detected {n_gpu_devices} GPU device(s) via nvidia-ml")
                if n_gpu_devices == 1:
                    print(f"GPU memory: {gpu_memory_per_device / (1024 ** 3):.1f}GB available")
                else:
                    print(f"GPU memory: {gpu_memory_per_device / (1024 ** 3):.1f}GB per device, {total_gpu_memory / (1024 ** 3):.1f}GB total")
                
                gpu_memory = total_gpu_memory
                
            except ImportError:
                print("pynvml not available, trying alternative methods...")
                
                # Method 2: Try subprocess to call nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run([
                        'nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        memory_values = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                        if len(memory_values) >= n_gpu_devices:
                            # nvidia-smi returns memory in MB
                            gpu_memory_per_device = memory_values[0] * (1024 ** 2)  # Convert MB to bytes
                            total_gpu_memory = sum(memory_values) * (1024 ** 2)
                            
                            print(f"Detected {n_gpu_devices} GPU device(s) via nvidia-smi")
                            if n_gpu_devices == 1:
                                print(f"GPU memory: {gpu_memory_per_device / (1024 ** 3):.1f}GB available")
                            else:
                                print(f"GPU memory: {gpu_memory_per_device / (1024 ** 3):.1f}GB per device, {total_gpu_memory / (1024 ** 3):.1f}GB total")
                            
                            gpu_memory = total_gpu_memory
                        else:
                            raise Exception("nvidia-smi returned unexpected number of devices")
                    else:
                        raise Exception(f"nvidia-smi failed with return code {result.returncode}")
                        
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
                    print(f"nvidia-smi method failed: {e}")
                    
                    # Method 3: Fallback - estimate based on common GPU types
                    print("Using fallback GPU memory estimation...")
                    
                    # Try to identify GPU type from device string
                    device_str = str(gpu_devices[0]).lower()
                    if 'titan' in device_str and 'rtx' in device_str:
                        gpu_memory_per_device = 24 * (1024 ** 3)  # Titan RTX: 24GB
                        print("Detected Titan RTX-like GPU, assuming 24GB per device")
                    elif 'rtx' in device_str:
                        if '4090' in device_str:
                            gpu_memory_per_device = 24 * (1024 ** 3)  # RTX 4090: 24GB
                            print("Detected RTX 4090-like GPU, assuming 24GB per device")
                        elif '3090' in device_str:
                            gpu_memory_per_device = 24 * (1024 ** 3)  # RTX 3090: 24GB
                            print("Detected RTX 3090-like GPU, assuming 24GB per device")
                        elif '4080' in device_str:
                            gpu_memory_per_device = 16 * (1024 ** 3)  # RTX 4080: 16GB
                            print("Detected RTX 4080-like GPU, assuming 16GB per device")
                        elif '3080' in device_str:
                            gpu_memory_per_device = 10 * (1024 ** 3)  # RTX 3080: 10GB
                            print("Detected RTX 3080-like GPU, assuming 10GB per device")
                        else:
                            gpu_memory_per_device = 12 * (1024 ** 3)  # Conservative RTX estimate
                            print("Detected RTX-like GPU, assuming 12GB per device (conservative)")
                    elif 'v100' in device_str:
                        gpu_memory_per_device = 32 * (1024 ** 3)  # Tesla V100: 32GB
                        print("Detected V100-like GPU, assuming 32GB per device")
                    elif 'a100' in device_str:
                        gpu_memory_per_device = 40 * (1024 ** 3)  # A100: 40GB (or 80GB, use conservative)
                        print("Detected A100-like GPU, assuming 40GB per device")
                    else:
                        gpu_memory_per_device = 8 * (1024 ** 3)  # Very conservative fallback
                        print("Unknown GPU type, assuming 8GB per device (conservative)")
                    
                    total_gpu_memory = gpu_memory_per_device * n_gpu_devices
                    gpu_memory = total_gpu_memory
                    
                    if n_gpu_devices == 1:
                        print(f"Estimated GPU memory: {gpu_memory_per_device / (1024 ** 3):.0f}GB available")
                    else:
                        print(f"Estimated GPU memory: {gpu_memory_per_device / (1024 ** 3):.0f}GB per device, {total_gpu_memory / (1024 ** 3):.0f}GB total")
                        
            except Exception as e:
                print(f"All GPU memory detection methods failed: {e}")
                print("Falling back to conservative 8GB estimate per GPU")
                gpu_memory_per_device = 8 * (1024 ** 3)
                total_gpu_memory = gpu_memory_per_device * n_gpu_devices
                gpu_memory = total_gpu_memory
        else:
            print("No GPU devices detected - using CPU mode")
            
    except Exception as e:
        print(f"GPU detection failed: {e}")
        pass
    
    # Estimate memory usage per sample with more realistic overhead
    # Each sample needs:
    # 1. Output array: n_neurons * n_timepoints * 4 bytes (float32)
    # 2. Intermediate computations: ~2x the output size for ODE solving (reduced from 3x)
    # 3. Weight matrix operations: n_neurons^2 * 4 bytes 
    # 4. Additional overhead: ~1.5x for JAX compilation and caching (reduced from 2x)
    
    output_size = n_neurons * n_timepoints * 4  # float32
    intermediate_size = output_size * 2  # ODE solver intermediates (reduced)
    weight_ops_size = n_neurons * n_neurons * 4  # Weight matrix operations
    overhead_factor = 1.75  # JAX compilation and caching overhead (reduced)
    
    bytes_per_sample = (output_size + intermediate_size + weight_ops_size) * overhead_factor
    
    print(f"Estimated memory per sample: {bytes_per_sample / (1024 ** 2):.1f} MB")
    print(f"  - Output array: {output_size / (1024 ** 2):.1f} MB")
    print(f"  - Intermediate computations: {intermediate_size / (1024 ** 2):.1f} MB") 
    print(f"  - Weight operations: {weight_ops_size / (1024 ** 2):.1f} MB")
    print(f"  - Overhead factor: {overhead_factor}x")
    
    # Choose memory limit based on available hardware
    if gpu_memory is not None:
        # Use more conservative memory allocation (70% instead of 80%) for better stability
        # This prevents memory fragmentation issues in large batch simulations
        usable_memory = gpu_memory * 0.70
        print(f"Using GPU memory constraint: {usable_memory / (1024 ** 3):.2f} GB ({gpu_memory / (1024 ** 3):.0f}GB total)")
        print("  Using conservative 70% limit to prevent memory fragmentation in large batch simulations")
    else:
        # Use system memory with reasonable safety margin (50% of available, max 32GB)
        usable_memory = min(available_memory * 0.5, 32 * (1024 ** 3))
        print(f"Using system memory constraint: {usable_memory / (1024 ** 3):.2f} GB")
    
    # Calculate memory-constrained batch size
    max_memory_samples = max(1, int(usable_memory / bytes_per_sample))
    print(f"Memory-limited batch size: {max_memory_samples}")
    
    # Start with reasonable baseline based on GPU setup
    if gpu_memory is not None:
        if n_gpu_devices == 1:
            # Single GPU: more conservative baseline but still efficient
            baseline_batch = max(16, n_devices * 12)  # 12 samples per JAX device for single GPU
            print(f"Baseline batch size (single GPU): {baseline_batch}")
        else:
            # Multi-GPU: higher baseline to utilize parallel processing
            baseline_batch = max(16, n_devices * 16)  # 16 samples per JAX device for multi-GPU
            print(f"Baseline batch size (multi-GPU): {baseline_batch}")
    else:
        baseline_batch = max(8, n_devices * 8)   # Lower baseline for CPU-only
        print(f"Baseline batch size (CPU): {baseline_batch}")
    
    # Choose the smaller of memory constraint and reasonable upper limit
    # With high-end GPUs, allow larger batch sizes
    if gpu_memory is not None:
        if n_gpu_devices == 1:
            reasonable_upper_limit = min(128, max_batch_size)  # More conservative for single GPU
        else:
            reasonable_upper_limit = min(256, max_batch_size)  # Higher limit for multi-GPU setup
    else:
        reasonable_upper_limit = min(64, max_batch_size)   # Lower limit for CPU-only
    
    memory_limited_batch = min(max_memory_samples, reasonable_upper_limit)
    
    # Take the larger of baseline and memory-limited (less conservative)
    optimal_batch = max(baseline_batch, memory_limited_batch)
    print(f"Initial optimal batch size: {optimal_batch}")
    
    # Special case: if we can fit all simulations in one batch and it's not too large, do it
    if effective_total <= max_memory_samples and effective_total <= reasonable_upper_limit:
        print(f"Can process all {effective_total} simulations in a single batch")
        optimal_batch = effective_total
    else:
        # Ensure batch size doesn't exceed available simulations
        optimal_batch = min(optimal_batch, effective_total)
        print(f"Simulation-limited batch size: {optimal_batch}")
    
    # Make batch size divisible by device count for even distribution
    # But only if we have enough samples to justify it
    if optimal_batch >= n_devices and n_devices > 1:
        optimal_batch = max(n_devices, (optimal_batch // n_devices) * n_devices)
        print(f"Device-aligned batch size: {optimal_batch}")
    
    # Final safety checks with hardware-appropriate minimums
    if gpu_memory is not None:
        if n_gpu_devices == 1:
            # Single GPU: more conservative limits
            optimal_batch = max(12, optimal_batch)  # Minimum of 12 for single GPU
            optimal_batch = min(optimal_batch, max_batch_size)  # Hard cap for single GPU stability
            print(f"Final batch size (single GPU): {optimal_batch}")
        else:
            # Multi-GPU: higher limits to utilize parallel processing
            optimal_batch = max(16, optimal_batch)  # Minimum of 16 for multi-GPU setup
            optimal_batch = min(optimal_batch, max_batch_size)  # Hard cap for multi-GPU stability
            print(f"Final batch size (multi-GPU): {optimal_batch}")
    else:
        optimal_batch = max(8, optimal_batch)   # Minimum of 8 for CPU setup
        optimal_batch = min(optimal_batch, 32)  # Hard cap of 32 for CPU
        print(f"Final batch size (CPU): {optimal_batch}")
    
    print(f'Final optimal batch size: {optimal_batch}')
    return optimal_batch


def initialize_pruning_state(
    neuron_params: NeuronParams, 
    sim_params: SimParams, 
    batch_size: int
) -> Pruning_state:
    """Initialize pruning state for a given batch size."""
    mn_idxs = neuron_params.mn_idxs
    all_neurons = jnp.arange(neuron_params.W.shape[-1])
    in_idxs = jnp.setdiff1d(all_neurons, mn_idxs)

    W_mask = jnp.full((batch_size, neuron_params.W.shape[0], neuron_params.W.shape[1]), 1, dtype=jnp.bool_)
    interneuron_mask = jnp.full((batch_size, W_mask.shape[-1]), fill_value=False, dtype=jnp.bool_)
    interneuron_mask = interneuron_mask.at[:, in_idxs].set(True)
    level = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    total_removed_neurons = jnp.full((batch_size, W_mask.shape[-1]), False, dtype=jnp.bool_)
    removed_stim_neurons = jnp.full([batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool_)
    neurons_put_back = jnp.full([batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool_)
    prev_put_back = jnp.full([batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool_)
    last_removed = jnp.full([batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool_)
    min_circuit = jnp.full((batch_size, 1), False)

    # Initialize probabilities
    exclude_mask = (~interneuron_mask)
    p_arrays = jax.vmap(removal_probability)(jnp.ones((batch_size, W_mask.shape[-1])), exclude_mask)

    # Get appropriate seeds for this batch
    batch_seeds = neuron_params.seeds[:batch_size]
    if len(batch_seeds) < batch_size:
        # Pad seeds if necessary
        pad_size = batch_size - len(batch_seeds)
        batch_seeds = jnp.concatenate([batch_seeds, jnp.repeat(batch_seeds[-1:], pad_size, axis=0)])

    return Pruning_state(
        W_mask=W_mask,
        interneuron_mask=interneuron_mask,
        level=level,
        total_removed_neurons=total_removed_neurons,
        removed_stim_neurons=removed_stim_neurons,
        neurons_put_back=neurons_put_back,
        prev_put_back=prev_put_back,
        last_removed=last_removed,
        remove_p=p_arrays,
        min_circuit=min_circuit,
        keys=batch_seeds
    )


def run_simulation_engine(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    sim_config: SimulationConfig,
    checkpoint_dir: Optional[Path] = None,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Pruning_state]]:
    """
    Unified simulation engine that handles all simulation types.
    
    Args:
        neuron_params: Neuron parameters
        sim_params: Simulation parameters  
        sim_config: Simulation configuration
    
    Returns:
        Results array, or (results, final_state) tuple if pruning enabled
    """
    total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
    n_devices = jax.device_count()
    
    # Calculate batch size
    batch_size = sim_config.batch_size
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(
            sim_params.n_neurons, len(sim_params.t_axis), sim_params.n_param_sets, n_devices, total_sims=total_sims
        )
    
    # Adjust batch size for multiple devices
    if n_devices > 1:
        batch_size = (batch_size // n_devices) * n_devices
        if batch_size == 0:
            batch_size = n_devices

    # Get batch processing function
    batch_func = get_batch_function(sim_config)
    
    # Create parallel version for multiple devices
    if n_devices > 1:
        batch_func = pmap(batch_func, axis_name="device", in_axes=(None, None, 0))

    print(f"Running {total_sims} {sim_config.sim_type} simulations with batch size {batch_size} on {n_devices} device(s)")

    # Handle pruning setup
    if sim_config.enable_pruning:
        return _run_with_pruning(
            neuron_params, sim_params, sim_config, batch_func, 
            total_sims, batch_size, n_devices, checkpoint_dir
        )
    else:
        return _run_stim_neurons(
            neuron_params, sim_params, sim_config, batch_func,
            total_sims, batch_size, n_devices, checkpoint_dir
        )


def _adjust_stimulation_for_batch(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    sim_config: SimulationConfig,
    batch_func,
    batch_indices: jnp.ndarray,
    n_devices: int,
) -> NeuronParams:
    """Adjust stimulation for a specific batch."""
    
    adjusted_neuron_params = neuron_params
    next_highest = None
    next_lowest = None
    
    for adjust_iter in range(sim_config.max_adjustment_iters):
        test_results = batch_func(adjusted_neuron_params, sim_params, batch_indices) # (Devices, Batch, Neurons, Time)
        if n_devices > 1:
            test_results = test_results.reshape(-1, *test_results.shape[2:]) # (Batch, Neurons, Time)

        max_rates = jnp.max(test_results, axis=2) # (Batch, Neurons, Time) -> (Batch, Neurons)
        n_active = jnp.sum(jnp.sum(test_results, axis=2) > 0, axis=1) # (Batch, Neurons) -> # (Batch,)
        n_high_fr = jnp.sum(max_rates > sim_config.high_fr_threshold, axis=1) # (Batch, Neurons) -> (Batch,)

        current_inputs = adjusted_neuron_params.input_currents.copy()
        new_inputs = current_inputs.copy()
        needs_adjustment = jnp.zeros_like(n_active, dtype=bool)

        for sim_idx in range(len(n_active)):
            sim_active = n_active[sim_idx]
            sim_high_fr = n_high_fr[sim_idx]

            # Calculate the actual global simulation index and its components
            if n_devices > 1:
                # batch_indices is reshaped for devices, so flatten it first
                flat_batch_indices = batch_indices.copy().reshape(-1)
                global_sim_idx = flat_batch_indices[sim_idx] if sim_idx < len(flat_batch_indices) else flat_batch_indices[-1]
            else:
                global_sim_idx = batch_indices.copy()
                global_sim_idx = global_sim_idx[sim_idx] if sim_idx < len(global_sim_idx) else global_sim_idx[-1]
            
            # Calculate replicate and stimulus indices from global simulation index
            replicate_idx = int(global_sim_idx % sim_params.n_param_sets)
            stim_idx = int(global_sim_idx // sim_params.n_param_sets)
            
            # Get the correct input for this specific stimulus and replicate
            sim_input = current_inputs[stim_idx, replicate_idx]

            if (sim_active > sim_config.n_active_upper) or (sim_high_fr > sim_config.n_high_fr_upper):
                # Too strong - reduce stimulation
                if next_lowest is None:
                    new_inputs = new_inputs.at[stim_idx, replicate_idx].set(sim_input / 2)
                else:
                    new_inputs = new_inputs.at[stim_idx, replicate_idx].set((sim_input + next_lowest[stim_idx, replicate_idx]) / 2)
                needs_adjustment = needs_adjustment.at[sim_idx].set(True)
            elif sim_active < sim_config.n_active_lower:
                # Too weak - increase stimulation
                if next_highest is None:
                    new_inputs = new_inputs.at[stim_idx, replicate_idx].set(sim_input * 2)
                else:
                    new_inputs = new_inputs.at[stim_idx, replicate_idx].set((sim_input + next_highest[stim_idx, replicate_idx]) / 2)
                needs_adjustment = needs_adjustment.at[sim_idx].set(True)
            # else: just right, no adjustment needed

            # Determine if this simulation needs adjustment
            sim_needs_adjustment = (sim_active > sim_config.n_active_upper) or (sim_high_fr > sim_config.n_high_fr_upper) or (sim_active < sim_config.n_active_lower)
            
            print(f"    Adjustment iter {adjust_iter + 1}, Stim {stim_idx}, Replicate {replicate_idx}: Max stim: {jnp.max(sim_input):.6f}, "
                  f"Active: {sim_active}, High FR: {sim_high_fr}, Needs adjustment: {sim_needs_adjustment}")
            
        # Update neuron params with new stimulation
        adjusted_neuron_params = adjusted_neuron_params._replace(
            input_currents=new_inputs
        )

        del test_results
        gc.collect()

        if not jnp.any(needs_adjustment):
            print(f"    Stimulation appropriate for all simulations in this batch - adjustment complete: True")
            break
    else:
        print(f"    Warning: Maximum adjustment iterations ({sim_config.max_adjustment_iters}) reached for this batch - adjustment complete: False")

    return adjusted_neuron_params


def _run_stim_neurons(
    neuron_params: NeuronParams,
    sim_params: SimParams, 
    sim_config: SimulationConfig,
    batch_func,
    total_sims: int,
    batch_size: int,
    n_devices: int,
    checkpoint_dir: Optional[Path] = None
) -> jnp.ndarray:
    """Run simulation without pruning, with optional per-batch automatic stimulation adjustment and checkpointing."""
    
    n_batches = (total_sims + batch_size - 1) // batch_size
    
    # Check for existing checkpoint to resume from
    checkpoint_state = None
    start_batch = 0
    all_results = []
    adjusted_neuron_params = neuron_params
    
    if sim_config.enable_checkpointing and (checkpoint_dir is not None):
        latest_checkpoint, base_name = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            try:
                checkpoint_state, adjusted_neuron_params, metadata = load_checkpoint(latest_checkpoint, base_name, neuron_params)
                start_batch = checkpoint_state.batch_index + 1
                print(f"Resuming from batch {start_batch}/{n_batches}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint {latest_checkpoint}: {e}")
                print("Starting from beginning...")
                start_batch = 0
                all_results = []

    for i in range(start_batch, n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_sims)
        actual_batch_size = end_idx - start_idx
        
        batch_indices = jnp.arange(start_idx, end_idx)

        # Pad if necessary for pmap
        if n_devices > 1 and len(batch_indices) < batch_size:
            pad_size = batch_size - len(batch_indices)
            batch_indices = jnp.concatenate([
                batch_indices,
                jnp.repeat(batch_indices[-1], pad_size)
            ])

        # Reshape for devices if using pmap
        if n_devices > 1:
            batch_indices = batch_indices.reshape(n_devices, -1)
        print('Batch Inds:', batch_indices)
        # Adjust stimulation for this specific batch if enabled
        if sim_config.adjustStimI:
            print(f"Adjusting stimulation for batch {i + 1}/{n_batches}...")
            adjusted_neuron_params = _adjust_stimulation_for_batch(
                adjusted_neuron_params, sim_params, sim_config, batch_func, batch_indices, n_devices)

        # Run the batch with adjusted parameters
        batch_results = jax.block_until_ready(batch_func(adjusted_neuron_params, sim_params, batch_indices))

        if n_devices > 1:
            batch_results = batch_results.reshape(-1, *batch_results.shape[2:])
            batch_results = batch_results[:actual_batch_size]  # Remove padding

        # Move results to CPU to save GPU memory
        batch_results = jax.device_put(batch_results, jax.devices("cpu")[0])
        all_results.append(batch_results)
        print(f"Batch {i + 1}/{n_batches} completed")

        # Save checkpoint if enabled and conditions are met
        if sim_config.enable_checkpointing and (checkpoint_dir is not None):  

            checkpoint_state = CheckpointState(
                batch_index=i,
                completed_batches=i + 1,
                total_batches=n_batches,
                n_result_batches=len(all_results),
                accumulated_mini_circuits=None,
                neuron_params=neuron_params,
                pruning_state=None
            )
            checkpoint_path = checkpoint_dir / f"checkpoint_batch_{i}.h5"
            metadata = {
                "sim_type": sim_config.sim_type,
                "enable_pruning": sim_config.enable_pruning,
                "total_sims": total_sims,
                "batch_size": batch_size,
                "n_devices": n_devices
            }
            
            print(f"Saving checkpoint for batch {i + 1}...")
            checkpoint_start_time = time.time()
            save_checkpoint(checkpoint_state, checkpoint_path, metadata, batch_start_idx=i)
            checkpoint_elapsed = time.time() - checkpoint_start_time
            print(f"Checkpoint saved in {checkpoint_elapsed:.2f} seconds")
            
            cleanup_old_checkpoints(checkpoint_dir=checkpoint_dir, keep_last_n=2)

        # Enhanced memory cleanup for large simulations
        del batch_results  # Free memory
        gc.collect()  # Force garbage collection
        
        # More aggressive cleanup for large batch simulations to prevent OOM
        if total_sims >= 1024 or batch_size >= 64:  # Large simulations
            if (i + 1) % 5 == 0:  # Every 5 batches for large sims
                jax.clear_caches()
                gc.collect()
                print(f"  Aggressive memory cleanup after batch {i + 1}")
        else:
            if (i + 1) % 10 == 0:  # Every 10 batches for normal sims
                jax.clear_caches()
                print(f"  Memory cleanup after batch {i + 1}")
                
        # Memory monitoring for large simulations
        if total_sims >= 1024 and (i + 1) % 5 == 0:
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                memory_percent = memory_info.percent
                available_gb = memory_info.available / (1024**3)
                if memory_percent > 80:
                    print(f"  ⚠️ Warning: High memory usage {memory_percent:.1f}% ({available_gb:.1f}GB free)")
            except:
                pass

    # If adjustStimI was used, run a final set of simulations with the final adjusted stimuli
    if sim_config.adjustStimI:
        print(f"\nRunning final simulations with adjusted stimulation across all {n_batches} batches...")
        import sparse
        final_all_results = []
        results_dir = checkpoint_dir / f"results_final"
        results_dir.mkdir(parents=True, exist_ok=True)
        # Process all batches with final adjusted parameters
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_sims)
            actual_batch_size = end_idx - start_idx
            
            batch_indices = jnp.arange(start_idx, end_idx)
            
            # Pad if necessary for pmap
            if n_devices > 1 and len(batch_indices) < batch_size:
                pad_size = batch_size - len(batch_indices)
                batch_indices = jnp.concatenate([
                    batch_indices,
                    jnp.repeat(batch_indices[-1], pad_size)
                ])
            
            # Reshape for devices if using pmap
            if n_devices > 1:
                batch_indices = batch_indices.reshape(n_devices, -1)
            
            # Run simulation with final adjusted parameters
            final_batch_results = jax.block_until_ready(
                batch_func(adjusted_neuron_params, sim_params, batch_indices)
            )
            
            if n_devices > 1:
                final_batch_results = final_batch_results.reshape(-1, *final_batch_results.shape[2:])
                final_batch_results = final_batch_results[:actual_batch_size]  # Remove padding
            
            # Move to CPU to save memory
            final_batch_results = jax.device_put(final_batch_results, jax.devices("cpu")[0])
            final_all_results.append(np.asarray(final_batch_results))
            sparse.save_npz(results_dir / f"Final_batch_Rs_{i}.npz", sparse.COO.from_numpy(final_batch_results))
            print(f"Final batch {i + 1}/{n_batches} completed")
            
            # Clean up memory
            del final_batch_results
            gc.collect()
        
        # Combine final results
        results = np.concatenate(final_all_results, axis=0)
        
        print(f"Final simulations with adjusted stimulation completed.")
        
        # Clean up
        del final_all_results
        gc.collect()
    else:
        # Combine results from adjustment phase
        results = jnp.concatenate(all_results, axis=0)
    
    # Combine results
    # results = jnp.concatenate(all_results, axis=0)

    # Reshape to (n_stim_configs, n_param_sets, n_neurons, n_timepoints)
    return results.reshape(
        sim_params.n_stim_configs, sim_params.n_param_sets,
        sim_params.n_neurons, len(sim_params.t_axis)
    ), None, adjusted_neuron_params


def _run_with_pruning(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    sim_config: SimulationConfig,
    batch_func,
    total_sims: int,
    batch_size: int,
    n_devices: int,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[jnp.ndarray, List[Pruning_state]]:
    """Run simulation with pruning, saving states across all batches."""
    
    # Initialize adaptive memory manager for optimal performance
    memory_manager = create_memory_manager(
        total_simulations=total_sims,
        simulation_type="pruning"
    )
    
    # Pre-compute static values that won't change during the loop
    oscillation_threshold_val = float(sim_config.oscillation_threshold)
    clip_start_val = int(sim_params.pulse_start / sim_params.dt) + 200

    # Create a version of update_single_sim_state with static args baked in
    def update_state_with_static_args(state, R, mn_mask):
        return update_single_sim_state(state, R, mn_mask, oscillation_threshold_val, clip_start_val)
    
    # Apply JIT to the wrapper function (not the original)
    jitted_update = jax.jit(update_state_with_static_args)
    
    # Now vmap this wrapper with only the traced arguments
    batch_update = jax.vmap(
        jitted_update, 
        in_axes=(0, 0, None)  # state, R, mn_mask - all batched
    )
    
    # Create parallel versions for multiple devices
    if n_devices > 1:
        batch_update = pmap(batch_update, axis_name="device", in_axes=(0, 0, None))

    all_results = []
    all_mini_circuits = []  # New: accumulate states from all batches
    n_batches = (total_sims + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_sims)
        actual_batch_size = end_idx - start_idx
        
        print(f"Processing batch {i + 1}/{n_batches} (sims {start_idx}-{end_idx-1})")
        
        # Monitor memory at batch start and apply recommendations
        memory_info = monitor_memory_usage()
        if memory_info.get('memory_pressure') in ['high', 'critical']:
            cleanup_performed = memory_manager.monitor_and_cleanup_if_needed(start_idx, warn_threshold=85.0)
            if cleanup_performed:
                print(f"  Applied memory cleanup before batch {i + 1}")
        
        batch_indices = jnp.arange(start_idx, end_idx)
        
        # Pad batch_indices if necessary for pmap
        if n_devices > 1 and len(batch_indices) < batch_size:
            pad_size = batch_size - len(batch_indices)
            batch_indices = jnp.concatenate([
                batch_indices, 
                jnp.repeat(batch_indices[-1], pad_size)
            ])
        
        mn_idxs = neuron_params.mn_idxs
        
        # Initialize state for this batch
        current_batch_size = len(batch_indices)
        state = initialize_pruning_state(neuron_params, sim_params, current_batch_size)

        # Reshape state for pmap if using multiple devices
        if n_devices > 1:
            state = reshape_state_for_pmap(state, n_devices)
            batch_indices = batch_indices.reshape(n_devices, -1)

        # Main pruning loop
        iteration = 0
        
        # Clear GPU memory before processing
        jax.clear_caches()
        
        while True:
            # Check convergence condition
            all_converged = jnp.all(state.min_circuit)
            
            if all_converged or iteration >= sim_config.max_pruning_iterations:
                break
                
            start_time = time.time()
            print(f"Batch {i + 1}/{n_batches}, Iteration {iteration}")
            
            # Use adaptive memory manager for optimized cleanup
            cleanup_result = memory_manager.should_cleanup(i * batch_size, iteration)
            if cleanup_result:
                memory_manager.perform_cleanup(context=f"batch_{i}_iter_{iteration}")
                print(f"  Memory cleanup at iteration {iteration}")
                
            # Update neuron parameters W_mask based on the current state
            if n_devices > 1:
                flat_W_mask = reshape_state_from_pmap(state).W_mask
                neuron_params = neuron_params._replace(W_mask=flat_W_mask)
            else:
                neuron_params = neuron_params._replace(W_mask=state.W_mask)

            # Run simulation
            batch_results = batch_func(neuron_params, sim_params, batch_indices)
            
            # Create mn_mask and broadcast to batch dimensions
            mn_mask = jnp.isin(jnp.arange(sim_params.n_neurons), mn_idxs)

            # Update state - now only passing traced arguments
            state = batch_update(state, batch_results, mn_mask)
            iteration += 1
            
            # Calculate available neurons for reporting
            if n_devices > 1:
                flat_state = reshape_state_from_pmap(state)
                available_neurons = jnp.sum(flat_state.interneuron_mask & (~flat_state.total_removed_neurons), axis=-1)
            else:
                available_neurons = jnp.sum(state.interneuron_mask & (~state.total_removed_neurons), axis=-1)
            
            print(f"  Available neurons: {available_neurons}")
            elapsed = time.time() - start_time
            print(f"  Sim time: {elapsed:.2f} seconds")

        
        # Run Final simulation state after convergence
        print(f"Final simulation state for batch {i + 1} after {iteration} iterations:")
        if n_devices > 1:
            flat_state = reshape_state_from_pmap(state)
            mini_circuit = (~flat_state.total_removed_neurons) | flat_state.last_removed | flat_state.prev_put_back
            W_mask_init = jnp.ones_like(flat_state.W_mask, dtype=jnp.bool_)
        else: 
            mini_circuit = (~state.total_removed_neurons) | state.last_removed | state.prev_put_back
            W_mask_init = jnp.ones_like(state.W_mask, dtype=jnp.bool_)
            flat_state = state
        
        W_mask_new = (W_mask_init * mini_circuit[:, :, None] * mini_circuit[:, None, :]).astype(jnp.bool_)
        neuron_params = neuron_params._replace(W_mask=W_mask_new)
        print(f"Final W_mask shape: {neuron_params.W_mask.shape}")
        
        # Run simulation
        batch_results = batch_func(neuron_params, sim_params, batch_indices)    
        # batch_results shape: (n_devices, batch_per_device, n_neurons, n_timepoints) if n_devices > 1
        #                      (batch_size, n_neurons, n_timepoints) if n_devices == 1
        
        # Reshape results and state back to flat format if using pmap
        if n_devices > 1:
            batch_results = batch_results.reshape(-1, *batch_results.shape[2:])
            # batch_results shape after reshape: (batch_size, n_neurons, n_timepoints)
            final_state = reshape_state_from_pmap(state)
        else:
            final_state = state
        
        # Remove padding if it was added
        if actual_batch_size < len(batch_results):
            batch_results = batch_results[:actual_batch_size]
            # batch_results shape after padding removal: (actual_batch_size, n_neurons, n_timepoints)
            final_state = trim_state_padding(final_state, actual_batch_size)
        
        # Move results to CPU to save GPU memory
        batch_results = jax.device_put(batch_results, jax.devices("cpu")[0])
        # mini_circuit = jnp.stack([((jnp.sum(batch_results[n],axis=-1)>0) & ~mn_mask) for n in range(batch_results.shape[0])],axis=0)
        mini_circuit = jax.device_put(mini_circuit, jax.devices("cpu")[0])
        
        all_results.append(batch_results)
        all_mini_circuits.append(mini_circuit)  # Save state for this batch

        print(f"Batch {i + 1}/{n_batches} completed")
        
        del batch_results, final_state
        jax.clear_caches()
        gc.collect()

    # Combine results
    results = jnp.stack(all_results, axis=0)
    all_mini_circuits = jnp.stack(all_mini_circuits, axis=0).reshape(-1, sim_params.n_neurons)  # (n_batches, batch_size, n_neurons) -> (total_sims, n_neurons)
    
    # Reshape results to (n_stim_configs, n_param_sets, n_neurons, n_timepoints)
    reshaped_results = results.reshape(
        sim_params.n_stim_configs, sim_params.n_param_sets,
        sim_params.n_neurons, len(sim_params.t_axis)
    )

    return reshaped_results, all_mini_circuits, neuron_params


def stack_pruning_states(states_list: List[Pruning_state]) -> Pruning_state:
    """Stack pruning states from multiple batches into a single state using jax.lax.map."""
    if not states_list:
        raise ValueError("Cannot stack empty list of states")
    
    if len(states_list) == 1:
        return states_list[0]
    
    # Convert list of states to a pytree structure that JAX can handle
    # This creates arrays where the first axis indexes the batch
    states_array = jax.tree.map(lambda *args: jnp.stack(args, axis=0), *states_list)
    
    # Now flatten back to single state by concatenating along batch dimension
    # Use lax.map to apply concatenation across all fields simultaneously
    def concatenate_field(field_array):
        # field_array has shape (n_batches, batch_size, ...) 
        # We want to reshape to (n_batches * batch_size, ...)
        return field_array.reshape(-1, *field_array.shape[2:])
    
    stacked_state = jax.lax.map(concatenate_field, states_array)
    
    return stacked_state


# #############################################################################################################################=
# CONFIGURATION LOADING AND SETUP FUNCTIONS
# #############################################################################################################################=

def prepare_neuron_params(cfg: DictConfig, W_table: Any, param_path: Optional[Path]=None) -> NeuronParams:
    """Prepare neuron parameters from configuration."""
    # Load connectivity
    W = jnp.array(load_W(cfg.experiment.wPath))
    
    # Sample parameters
    num_sims = cfg.experiment.n_replicates
    n_neurons = W.shape[0]
    base_seed = cfg.experiment.seed
    keys = jax.random.split(jax.random.PRNGKey(base_seed), 5)
    
    tau = sample_trunc_normal(
        keys[0], cfg.neuron_params.tauMean, cfg.neuron_params.tauStdv,
        (num_sims, n_neurons)
    )
    a = sample_trunc_normal(
        keys[1], cfg.neuron_params.aMean, cfg.neuron_params.aStdv,
        (num_sims, n_neurons)
    )
    threshold = sample_trunc_normal(
        keys[2], cfg.neuron_params.thresholdMean, cfg.neuron_params.thresholdStdv,
        (num_sims, n_neurons)
    )
    fr_cap = sample_trunc_normal(
        keys[3], cfg.neuron_params.frcapMean, cfg.neuron_params.frcapStdv,
        (num_sims, n_neurons)
    )
    seeds = jax.random.split(keys[4], num_sims)
    
    # Apply size scaling
    if "size" in W_table:
        a, threshold = set_sizes(W_table["size"].values, a, threshold)
    else:
        a, threshold = set_sizes(W_table["surf_area_um2"].values, a, threshold)
    
    # Prepare input currents for all stimulus configurations
    # Test if stimNeurons and stimI are same size, pad with element from index 0 if not
    stim_neurons_list = cfg.experiment.stimNeurons
    stim_input_list = cfg.experiment.stimI
    
    len_neurons = len(stim_neurons_list)
    len_inputs = len(stim_input_list)
    
    if len_neurons != len_inputs:
        print(f"Warning: stimNeurons length ({len_neurons}) != stimI length ({len_inputs})")
        if len_neurons > len_inputs:
            # Pad stimI with first element
            padding_needed = len_neurons - len_inputs
            stim_input_list = list(stim_input_list) + [stim_input_list[0]] * padding_needed
            print(f"Padded stimI with {padding_needed} copies of first element ({stim_input_list[0]})")
        else:
            # Pad stimNeurons with first element
            padding_needed = len_inputs - len_neurons
            stim_neurons_list = list(stim_neurons_list) + [stim_neurons_list[0]] * padding_needed
            print(f"Padded stimNeurons with {padding_needed} copies of first element ({stim_neurons_list[0]})")
    
    input_currents_list = []
    for stim_neurons, stim_input in zip(stim_neurons_list, stim_input_list):
        input_current = jnp.tile(make_input(n_neurons, jnp.array(stim_neurons), stim_input), (num_sims, 1))
        input_currents_list.append(input_current)
    input_currents = jnp.array(input_currents_list)
    
    # Extract shuffle indices
    exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs = extract_shuffle_indices(W_table)
    
    # Initialize W_mask for pruning if needed
    W_mask = jnp.ones((num_sims, W.shape[0], W.shape[1]), dtype=jnp.bool_)
    if len(cfg.experiment.removeNeurons) > 0:
        W_mask = W_mask.at[:, cfg.experiment.removeNeurons, :].set(False)
        W_mask = W_mask.at[:, :, cfg.experiment.removeNeurons].set(False)
        print(f"Initial removal of neurons at indices: {cfg.experiment.removeNeurons}")
    elif len(cfg.experiment.keepOnly) > 0:
        mn_mask = jnp.isin(jnp.arange(n_neurons), mn_idxs)
        keep_neurons = jnp.zeros(n_neurons, dtype=jnp.bool_)
        keep_neurons = keep_neurons.at[jnp.asarray(cfg.experiment.keepOnly)].set(True)
        keep_neurons = jnp.where(keep_neurons | mn_mask, True, False)
        W_mask = W_mask.at[:, keep_neurons, :].set(True)
        W_mask = W_mask.at[:, :, keep_neurons].set(True)
        print(f"Initial keeping of neurons at indices: {cfg.experiment.keepOnly}")
    if (param_path is not None) and param_path.exists():
        import src.io_dict_to_hdf5 as ioh5
        nparams = ioh5.load(param_path, enable_jax=True)
        print('Loaded neuron parameters from', param_path)
        return NeuronParams(**nparams)
    else:
        return NeuronParams(
            W=W, W_mask=W_mask, tau=tau, a=a, threshold=threshold, fr_cap=fr_cap,
            input_currents=input_currents, seeds=seeds,
            exc_dn_idxs=exc_dn_idxs, inh_dn_idxs=inh_dn_idxs,
            exc_in_idxs=exc_in_idxs, inh_in_idxs=inh_in_idxs, mn_idxs=mn_idxs,
        )


def prepare_sim_params(cfg: DictConfig, n_stim_configs: int, n_neurons: int) -> SimParams:
    """Prepare simulation parameters from configuration."""
    T = float(cfg.sim.T)
    dt = float(cfg.sim.dt)
    t_axis = jnp.arange(0, T + dt/2, dt, dtype=jnp.float32)  # Add dt/2 to ensure T is included
    # Ensure t_axis is monotonically increasing and within bounds
    t_axis_safe = jnp.sort(t_axis)  # Sort to ensure monotonicity
    t_axis_safe = jnp.clip(t_axis_safe, 0.0, T)  # Ensure within [0, T]
    
    # Remove any duplicate time points
    t_axis_safe = jnp.unique(t_axis_safe)
    return SimParams(
        n_neurons=n_neurons,
        n_param_sets=cfg.experiment.n_replicates,
        n_stim_configs=n_stim_configs,
        T=T, dt=dt,
        pulse_start=float(cfg.sim.pulseStart),
        pulse_end=float(cfg.sim.pulseEnd),
        t_axis=t_axis_safe,
        exc_multiplier=float(cfg.neuron_params.excitatoryMultiplier),
        inh_multiplier=float(cfg.neuron_params.inhibitoryMultiplier),
        noise_stdv_prop=getattr(cfg.sim, "noiseStdvProp", 0.1),
        r_tol=getattr(cfg.sim, "rtol", 1e-7),
        a_tol=getattr(cfg.sim, "atol", 1e-9),
        noise_stdv=cfg.sim.noiseStdv
    )

def parse_simulation_config(cfg: DictConfig) -> SimulationConfig:
    """Parse simulation configuration from DictConfig."""
    # Determine simulation type
    sim_type = "baseline"
    if getattr(cfg.sim, "shuffle", False):
        sim_type = "shuffle"
    elif getattr(cfg.sim, "noise", False):
        sim_type = "noise"
    elif getattr(cfg.sim, "prune_network", False) | (len(cfg.experiment.removeNeurons) > 0):
        sim_type = "Wmask"
    
    # Check if pruning is enabled
    enable_pruning = getattr(cfg.sim, "prune_network", False)
    
    return SimulationConfig(
        sim_type=sim_type,
        enable_pruning=enable_pruning,
        oscillation_threshold=getattr(cfg.experiment, "oscillation_threshold", 0.5),
        batch_size=getattr(cfg.experiment, "batch_size", None),
        max_pruning_iterations=getattr(cfg.experiment, "max_pruning_iterations", 200),
        adjustStimI=getattr(cfg.sim, "adjustStimI", False),
        max_adjustment_iters=getattr(cfg.sim, "max_adjustment_iters", 10),
        n_active_upper=getattr(cfg.sim, "n_active_upper", 500),
        n_active_lower=getattr(cfg.sim, "n_active_lower", 5),
        n_high_fr_upper=getattr(cfg.sim, "n_high_fr_upper", 100),
        high_fr_threshold=getattr(cfg.sim, "high_fr_threshold", 100.0),
        enable_checkpointing=getattr(cfg.sim, "enable_checkpointing", False),
    )

# #############################################################################################################################=
# STATE PERSISTENCE FUNCTIONS
# #############################################################################################################################=

def save_state(state: Pruning_state, path: str):
    """Save the pruning state to a file."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path: str) -> Pruning_state:
    """Load the pruning state from a file."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# #############################################################################################################################=
# UNIFIED MAIN INTERFACE FUNCTION
# #############################################################################################################################=

def prepare_vnc_simulation_params(cfg: DictConfig):
    """
    Prepare common simulation parameters and data.
    
    Args:
        cfg: Configuration containing all simulation parameters
        
    Returns:
        Tuple of (W_table, neuron_params, sim_params, sim_config, n_stim_configs)
    """
    print("Loading network configuration...")
    W_table = load_wTable(cfg.experiment.dfPath)
    
    # Handle DN screening if specified
    if getattr(cfg.experiment, "dn_screen", False):
        all_exc_dns = W_table.loc[
            (W_table["class"] == "descending neuron") & 
            (W_table["predictedNt"] == "acetylcholine")
        ]
        stim_neurons = [[neuron] for neuron in all_exc_dns.index.to_list()]
        stim_inputs = [cfg.experiment.stimI[0]] * len(stim_neurons)
        cfg.experiment.stimNeurons = stim_neurons
        cfg.experiment.stimI = stim_inputs
    
    n_stim_configs = len(cfg.experiment.stimNeurons)
    
    # Prepare parameters
    neuron_params = prepare_neuron_params(cfg, W_table)
    sim_params = prepare_sim_params(cfg, n_stim_configs, neuron_params.W.shape[0])
    sim_config = parse_simulation_config(cfg)
    
    return W_table, neuron_params, sim_params, sim_config, n_stim_configs


def run_vnc_simulation(cfg: DictConfig) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Pruning_state]]:
    """
    Unified main function to run VNC simulation with any configuration.

    Args:
        cfg: Configuration containing all simulation parameters
        
    Returns:
        Results array if pruning disabled, or (results, final_state) tuple if pruning enabled
    """
    W_table, neuron_params, sim_params, sim_config, n_stim_configs = prepare_vnc_simulation_params(cfg)

    jax.clear_caches()
    print(f"Running {sim_config.sim_type} simulation with:")
    print(f"  {n_stim_configs} stimulus configurations")
    print(f"  {sim_params.n_param_sets} parameter sets")
    print(f"  {sim_params.n_neurons} neurons")
    print(f"  {len(sim_params.t_axis)} time points")
    if sim_config.enable_pruning:
        print(f"  Pruning enabled with oscillation threshold: {sim_config.oscillation_threshold}")
    
    # Parse checkpointing configuration
    checkpoint_dir = None
    if cfg.sim.enable_checkpointing and hasattr(cfg, 'paths'):
        checkpoint_dir = Path(cfg.paths.ckpt_dir) / "checkpoints"
    # Run simulation
    start_time = time.time()
    result = run_simulation_engine(neuron_params, sim_params, sim_config, checkpoint_dir)
    jax.block_until_ready(result)
    
    elapsed = time.time() - start_time
    total_sims = n_stim_configs * sim_params.n_param_sets
    
    print(f"\nSimulation completed:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Total simulations: {total_sims}")
    print(f"  Time per simulation: {elapsed/total_sims:.4f} seconds")
    print(f"  Throughput: {total_sims/elapsed:.2f} sims/second")
    
    if sim_config.enable_pruning:
        results, final_mini_circuits, neuron_params = result
        print(f"  Results shape: {results.shape}")
        print(f"  Final pruning state returned")
    else:
        results, final_mini_circuits, neuron_params = result
        print(f"  Results shape: {results.shape}")

    return results, final_mini_circuits, neuron_params


# #############################################################################################################################=
# SUMMARY OF AVAILABLE FUNCTIONS
# #############################################################################################################################=

"""
MAIN INTERFACE FUNCTIONS:
- run_vnc_simulation(cfg): Unified function for all simulation types

CORE SIMULATION FUNCTIONS:
- run_baseline(): Standard simulation without modifications  
- run_with_shuffle(): Simulation with shuffled connectivity
- run_with_noise(): Simulation with noisy connectivity
- run_with_Wmask(): Simulation with pruned connectivity
- run_single_simulation(): Core ODE solver for individual simulation runs
- rate_equation_half_tanh(): ODE rate equation with half-tanh activation
- reweight_connectivity(): Apply excitatory/inhibitory multipliers to weights
- add_noise_to_weights(): Add truncated normal noise to weight matrix

SIMULATION ENGINE:
- run_simulation_engine(): Unified engine that handles all simulation types
- get_batch_function(): Get appropriate batch processing function
- _run_stim_neurons(): Execute simulation without pruning
- _run_with_pruning(): Execute simulation with pruning
- _adjust_stimulation_for_batch(): Adjust stimulation parameters for specific batch

BATCH PROCESSING FUNCTIONS:
- process_batch_baseline(): Process batch of baseline simulations
- process_batch_shuffle(): Process batch of shuffle simulations
- process_batch_noise(): Process batch of noise simulations
- process_batch_Wmask(): Process batch of Wmask simulations

PRUNING-SPECIFIC FUNCTIONS:
- update_single_sim_state(): Update pruning state for single simulation
- removal_probability(): Calculate neuron removal probabilities
- jax_choice(): JAX-compatible random choice function
- initialize_pruning_state(): Setup initial pruning state
- reshape_state_for_pmap(): Reshape state for multi-device processing
- reshape_state_from_pmap(): Reshape state back from multi-device format
- pad_state_for_devices(): Pad state for device alignment
- trim_state_padding(): Remove padding from state to get back to actual batch size
- stack_pruning_states(): Stack multiple pruning states into single state
- save_state()/load_state(): Persist pruning state to/from disk

CONFIGURATION AND SETUP:
- prepare_neuron_params(): Setup neuron parameters from config
- prepare_sim_params(): Setup simulation parameters from config
- parse_simulation_config(): Parse simulation configuration from config
- calculate_optimal_batch_size(): Determine optimal batch size

IMPORTED UTILITY FUNCTIONS (from src.sim_utils and src.shuffle_utils):
- load_W(): Load connectivity matrix from file
- load_wTable(): Load neuron metadata table from file
- sample_trunc_normal(): Sample from truncated normal distribution
- set_sizes(): Set neuron parameters based on surface area
- make_input(): Create input current arrays for stimulation
- compute_oscillation_score(): Compute oscillation scores for activity
- extract_shuffle_indices(): Extract neuron type indices for shuffling
- full_shuffle(): Perform complete connectivity matrix shuffling

CONFIGURATION STRUCTURES:
- NeuronParams: NamedTuple for immutable neuron parameter storage
- SimParams: NamedTuple for immutable simulation parameter storage  
- Pruning_state: NamedTuple for pruning algorithm state storage
- SimulationConfig: NamedTuple for simulation execution configuration

"""