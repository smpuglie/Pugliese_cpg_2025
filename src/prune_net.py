import jax
import jax.numpy as jnp
from jax import random, pmap
from functools import partial
from src.sim_utils import *
from src.optimized_vnc import *
import time
import gc


class Pruning_state(NamedTuple):
    """Parameters for pruning simulation."""
    W_mask: jnp.ndarray
    interneuron_mask: jnp.ndarray
    level: jnp.ndarray
    total_removed_neurons: jnp.ndarray
    removed_stim_neurons: jnp.ndarray
    neurons_put_back: jnp.ndarray
    prev_put_back: jnp.ndarray
    last_removed: jnp.ndarray
    remove_p: jnp.ndarray
    min_circuit: jnp.ndarray
    keys: jnp.ndarray
    

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


@partial(jax.jit, static_argnames=('oscillation_threshold', 'clip_start'))
def update_single_sim_state(state, R, mn_idxs, oscillation_threshold, clip_start=250):
    """Update state for a single simulation with proper round-based convergence logic"""
    
    # Unpack state
    (W_mask, interneuron_mask, level, total_removed_neurons, removed_stim_neurons,
        neurons_put_back, prev_put_back, last_removed, remove_p, min_circuit, key) = state

    # Get active MN activity using JAX-compatible approach
    max_frs = jnp.max(R, axis=-1)
    mn_mask = jnp.isin(jnp.arange(R.shape[0]), mn_idxs)
    active_mask = ((max_frs > 0) & mn_mask)

    # Compute oscillation score
    oscillation_score, _ = compute_oscillation_score(R[..., clip_start:], active_mask, prominence=0.05)

    # Check if oscillation is below threshold or NaN
    reset_condition = (oscillation_score < oscillation_threshold) | jnp.isnan(oscillation_score)

    key_next, subkey_continue, subkey_reset = random.split(key, 3)

    # === CONTINUE BRANCH: Normal pruning (oscillation is good) ===
    # Identify currently silent interneurons - these are permanently removed
    silent_interneurons = interneuron_mask & (max_frs <= 0)
    
    # Permanently remove silent interneurons (only new ones, not already removed)
    newly_silent = silent_interneurons & (~total_removed_neurons)
    total_removed_continue = total_removed_neurons | newly_silent
    
    # Update probabilities - exclude non-interneurons, removed neurons, and neurons put back this round
    exclude_mask_continue = (~interneuron_mask) | total_removed_continue | neurons_put_back
    p_continue = removal_probability(max_frs, exclude_mask_continue)
    
    # Check if we have neurons left to remove in this round
    available_neurons_continue = jnp.sum(interneuron_mask & (~exclude_mask_continue))
    p_sum_continue = jnp.sum(p_continue)
    
    # If no neurons left to remove or all probabilities are zero, start new round
    start_new_round = (available_neurons_continue <= 0) | (~jnp.isfinite(p_sum_continue))
    
    # NEW ROUND LOGIC
    # Check if neuronsPutBack == prevNeuronsPutBack (convergence condition)
    sets_equal = jnp.array_equal(neurons_put_back, prev_put_back)
    converged = start_new_round & sets_equal
    
    # If starting new round but not converged, reset for next round
    neurons_put_back_new_round = jnp.full(neurons_put_back.shape, False, dtype=jnp.bool_)
    prev_put_back_new_round = neurons_put_back  # Save current as previous
    
    # Select new neuron to remove (if available and not starting new round)
    neuron_idx_continue = jax_choice(subkey_continue, jnp.arange(len(max_frs)), p_continue)
    
    # Update removed neurons (only if not starting new round)
    removed_stim_continue = jax.lax.select(
        start_new_round,
        removed_stim_neurons,  # Keep current if starting new round
        removed_stim_neurons.at[neuron_idx_continue].set(True)
    )
    total_removed_continue = jax.lax.select(
        start_new_round,
        total_removed_continue,  # Keep current if starting new round
        total_removed_continue.at[neuron_idx_continue].set(True)
    )
    
    # Track what was removed this iteration
    last_removed_continue = jnp.full(last_removed.shape, False, dtype=jnp.bool_)
    # Only mark as removed if we actually removed something (not starting new round)
    last_removed_continue = jax.lax.select(
        start_new_round,
        newly_silent,  # Only newly silent neurons if starting new round
        last_removed_continue.at[neuron_idx_continue].set(True) | newly_silent
    )
    
    # Update state for continue branch
    level_continue = level + 1
    neurons_put_back_continue = jax.lax.select(
        start_new_round,
        neurons_put_back_new_round,
        neurons_put_back
    )
    prev_put_back_continue = jax.lax.select(
        start_new_round,
        prev_put_back_new_round,
        prev_put_back
    )
    min_circuit_continue = converged

    # === RESET BRANCH: Restore last removed neuron and try again ===
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
    
    # Check how many neurons are available
    available_neurons_reset = jnp.sum(interneuron_mask & (~exclude_mask_reset))
    p_sum_reset = jnp.sum(p_reset)
    
    # If no neurons left to remove, start new round
    start_new_round_reset = (available_neurons_reset <= 0) | (~jnp.isfinite(p_sum_reset))
    
    # Check convergence for reset branch too
    sets_equal_reset = jnp.array_equal(neurons_put_back_reset, prev_put_back)
    converged_reset = start_new_round_reset & sets_equal_reset
    
    # Select neuron to remove (if available)
    neuron_idx_reset = jax_choice(subkey_reset, jnp.arange(len(max_frs)), p_reset)
    
    # Update removed neurons (only if not starting new round)
    removed_stim_reset = jax.lax.select(
        start_new_round_reset,
        removed_stim_reset,
        removed_stim_reset.at[neuron_idx_reset].set(True)
    )
    total_removed_reset = jax.lax.select(
        start_new_round_reset,
        total_removed_reset,
        total_removed_reset.at[neuron_idx_reset].set(True)
    )
    
    # Track what was newly removed
    last_removed_reset = jnp.full(last_removed.shape, False, dtype=jnp.bool_)
    last_removed_reset = jax.lax.select(
        start_new_round_reset,
        last_removed_reset,  # Nothing new removed if starting new round
        last_removed_reset.at[neuron_idx_reset].set(True)
    )
    
    # Handle new round logic for reset branch
    neurons_put_back_reset = jax.lax.select(
        start_new_round_reset,
        jnp.full(neurons_put_back.shape, False, dtype=jnp.bool_),
        neurons_put_back_reset
    )
    prev_put_back_reset = jax.lax.select(
        start_new_round_reset,
        neurons_put_back,  # Save current as previous
        prev_put_back
    )
    
    # Keep level the same (we're trying again, not progressing)
    level_reset = level
    min_circuit_reset = converged_reset

    # === SELECT BETWEEN BRANCHES ===
    final_total_removed = jax.lax.select(reset_condition, total_removed_reset, total_removed_continue)
    final_removed_stim = jax.lax.select(reset_condition, removed_stim_reset, removed_stim_continue)
    final_last_removed = jax.lax.select(reset_condition, last_removed_reset, last_removed_continue)
    final_neurons_put_back = jax.lax.select(reset_condition, neurons_put_back_reset, neurons_put_back_continue)
    final_prev_put_back = jax.lax.select(reset_condition, prev_put_back_reset, prev_put_back_continue)
    final_level = jax.lax.select(reset_condition, level_reset, level_continue)
    final_p = jax.lax.select(reset_condition, p_reset, p_continue)
    final_min_circuit = jax.lax.select(reset_condition, min_circuit_reset, min_circuit_continue)

    # Calculate available neurons for final state
    available_neurons = jnp.sum(interneuron_mask & (~final_total_removed))

    # Update W_mask to reflect removed neurons
    W_mask_init = jnp.ones_like(W_mask, dtype=jnp.float32)
    removed_float = final_total_removed.astype(jnp.float32)
    kept_mask = 1.0 - removed_float
    W_mask_new = W_mask_init * kept_mask[:, None] * kept_mask[None, :]

    # Convert to scalar for jax.lax.select
    final_converged_scalar = jnp.squeeze(final_min_circuit) | min_circuit

    # When converged, preserve current state (don't make further changes)
    return Pruning_state(
        W_mask=jax.lax.select(final_converged_scalar, W_mask, W_mask_new),
        interneuron_mask=interneuron_mask,
        level=jax.lax.select(final_converged_scalar, level, final_level),
        total_removed_neurons=jax.lax.select(final_converged_scalar, total_removed_neurons, final_total_removed),
        neurons_put_back=jax.lax.select(final_converged_scalar, neurons_put_back, final_neurons_put_back),
        prev_put_back=jax.lax.select(final_converged_scalar, prev_put_back, final_prev_put_back),
        removed_stim_neurons=jax.lax.select(final_converged_scalar, removed_stim_neurons, final_removed_stim),
        last_removed=jax.lax.select(final_converged_scalar, last_removed, final_last_removed),
        remove_p=jax.lax.select(final_converged_scalar, remove_p, final_p),
        min_circuit=jax.lax.select(final_converged_scalar, min_circuit, final_min_circuit),
        keys=key_next,
    )


def reshape_state_for_pmap(state: Pruning_state, n_devices: int) -> Pruning_state:
    """Reshape state for pmap distribution across devices"""
    batch_size = state.W_mask.shape[0]
    batch_per_device = batch_size // n_devices
    
    # Ensure batch_size is divisible by n_devices
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
    """Pad state to match target batch size by repeating last elements"""
    current_batch_size = state.W_mask.shape[0]
    if current_batch_size >= target_batch_size:
        return state
    
    pad_size = target_batch_size - current_batch_size
    
    return Pruning_state(
        W_mask=jnp.concatenate([state.W_mask] + [state.W_mask[-1:]] * pad_size, axis=0),
        interneuron_mask=jnp.concatenate([state.interneuron_mask] + [state.interneuron_mask[-1:]] * pad_size, axis=0),
        level=jnp.concatenate([state.level] + [state.level[-1:]] * pad_size, axis=0),
        total_removed_neurons=jnp.concatenate([state.total_removed_neurons] + [state.total_removed_neurons[-1:]] * pad_size, axis=0),
        removed_stim_neurons=jnp.concatenate([state.removed_stim_neurons] + [state.removed_stim_neurons[-1:]] * pad_size, axis=0),
        neurons_put_back=jnp.concatenate([state.neurons_put_back] + [state.neurons_put_back[-1:]] * pad_size, axis=0),
        prev_put_back=jnp.concatenate([state.prev_put_back] + [state.prev_put_back[-1:]] * pad_size, axis=0),
        last_removed=jnp.concatenate([state.last_removed] + [state.last_removed[-1:]] * pad_size, axis=0),
        remove_p=jnp.concatenate([state.remove_p] + [state.remove_p[-1:]] * pad_size, axis=0),
        min_circuit=jnp.concatenate([state.min_circuit] + [state.min_circuit[-1:]] * pad_size, axis=0),
        keys=jnp.concatenate([state.keys] + [state.keys[-1:]] * pad_size, axis=0)
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


# Main simulation orchestration functions
def run_prune_batched(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    simulation_type: str = "baseline",
    batch_size: Optional[int] = None,
    oscillation_threshold: float = 0.5,
    ) -> jnp.ndarray:
    """
    Run complete simulation with efficient batching.
    
    Args:
        neuron_params: Neuron parameters
        sim_params: Simulation parameters
        simulation_type: "baseline", "shuffle", or "noise"
        batch_size: Batch size (auto-calculated if None)
    
    Returns:
        Results array of shape (n_stim_configs, n_param_sets, n_neurons, n_timepoints)
    """
    
    total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
    n_devices = jax.device_count()
    
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(
            sim_params.n_neurons, len(sim_params.t_axis)
        )
    
    # Adjust batch size for multiple devices
    if n_devices > 1:
        # Make batch_size divisible by n_devices
        batch_size = (batch_size // n_devices) * n_devices
        if batch_size == 0:
            batch_size = n_devices
    
    # Select batch processing function
    if simulation_type == "shuffle":
        batch_func = process_batch_shuffle
    elif simulation_type == "noise":
        batch_func = process_batch_noise
    elif simulation_type == "prune":
        batch_func = process_batch_prune
    else:
        batch_func = process_batch_baseline
    
    # Create batch update function - always use vmap for single device case
    batch_update = jax.vmap(
        update_single_sim_state, 
        in_axes=(0, 0, None, None, None), 
    )
    
    # Create parallel versions for multiple devices
    if n_devices > 1:
        # Apply pmap to both batch_func and batch_update
        batch_func = pmap(batch_func, axis_name="device")
        batch_update = pmap(batch_update, axis_name="device")

    print(f"Running {total_sims} simulations with batch size {batch_size} on {n_devices} device(s)")
    
    # Process in batches
    all_results = []
    n_batches = (total_sims + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_sims)
        actual_batch_size = end_idx - start_idx
        
        batch_indices = jnp.arange(start_idx, end_idx)
        
        # Pad batch_indices if necessary for pmap
        if n_devices > 1 and len(batch_indices) < batch_size:
            pad_size = batch_size - len(batch_indices)
            batch_indices = jnp.concatenate([
                batch_indices, 
                jnp.repeat(batch_indices[-1], pad_size)
            ])
        
        mn_idxs = neuron_params.mn_idxs
        all_neurons = jnp.arange(neuron_params.W.shape[-1])
        in_idxs = jnp.setdiff1d(all_neurons, mn_idxs)

        clip_start = int(sim_params.pulse_start / sim_params.dt) + 200
        
        # Initialize state for this batch
        current_batch_size = len(batch_indices)
        W_mask = jnp.full((current_batch_size, neuron_params.W.shape[0], neuron_params.W.shape[1]), 1, dtype=jnp.bool)
        interneuron_mask = jnp.full((current_batch_size, W_mask.shape[-1]), fill_value=False, dtype=jnp.bool_)
        interneuron_mask = interneuron_mask.at[:, in_idxs].set(True)
        level = jnp.zeros((current_batch_size, 1), dtype=jnp.int32)
        total_removed_neurons = jnp.full((current_batch_size, W_mask.shape[-1]), False, dtype=jnp.bool)
        removed_stim_neurons = jnp.full([current_batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool)
        neurons_put_back = jnp.full([current_batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool)
        prev_put_back = jnp.full([current_batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool)
        last_removed = jnp.full([current_batch_size, total_removed_neurons.shape[-1]], False, dtype=jnp.bool)
        min_circuit = jnp.full((current_batch_size, 1), False)

        # Initialize probabilities
        exclude_mask = (~interneuron_mask)
        p_arrays = jax.vmap(removal_probability)(jnp.ones((current_batch_size, W_mask.shape[-1])), exclude_mask)

        # Get appropriate seeds for this batch
        batch_seeds = neuron_params.seeds[start_idx:end_idx]
        if len(batch_seeds) < current_batch_size:
            # Pad seeds if necessary
            pad_size = current_batch_size - len(batch_seeds)
            batch_seeds = jnp.concatenate([batch_seeds, jnp.repeat(batch_seeds[-1:], pad_size, axis=0)])

        # Initialize pruning state
        state = Pruning_state(
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

        # Reshape state for pmap if using multiple devices
        if n_devices > 1:
            state = reshape_state_for_pmap(state, n_devices)
            # Also reshape batch_indices for pmap
            batch_indices = batch_indices.reshape(n_devices, -1)

        # Main pruning loop
        iteration = 0
        max_iterations = 200  # Safety limit
        
        # Clear GPU memory before processing
        jax.clear_caches()
        
        while True:
            # Check convergence condition
            if n_devices > 1:
                # For pmap, we need to check across all devices
                all_converged = jnp.all(state.min_circuit)
            else:
                all_converged = jnp.all(state.min_circuit)
            
            if all_converged or iteration >= max_iterations:
                break
                
            start_time = time.time()
            print(f"Iteration {iteration}")
            
            # Update neuron parameters W_mask based on the current state
            if n_devices > 1:
                # Flatten W_mask for update_params
                flat_W_mask = reshape_state_from_pmap(state).W_mask
                neuron_params = neuron_params._repalce(W_mask=flat_W_mask)
                # neuron_params = update_params(neuron_params, W_mask=flat_W_mask)
            else:
                neuron_params = neuron_params._repalce(W_mask=state.W_mask)
                # neuron_params = update_params(neuron_params, W_mask=state.W_mask)
                
            # Run simulation 
            batch_results = batch_func(neuron_params, sim_params, batch_indices)
            
            # Update state
            state = batch_update(state, batch_results, mn_idxs, oscillation_threshold, clip_start)
            iteration += 1
            
            # Calculate available neurons for reporting
            if n_devices > 1:
                flat_state = reshape_state_from_pmap(state)
                available_neurons = jnp.sum(flat_state.interneuron_mask & (~flat_state.total_removed_neurons))
            else:
                available_neurons = jnp.sum(state.interneuron_mask & (~state.total_removed_neurons))
            
            print(f"  Available neurons: {available_neurons}")
            elapsed = time.time() - start_time
            print(f"  Sim time: {elapsed:.2f} seconds")
        
        # Reshape results and state back to flat format if using pmap
        if n_devices > 1:
            batch_results = batch_results.reshape(-1, *batch_results.shape[2:])
            state = reshape_state_from_pmap(state)
        
        # Remove padding if it was added
        if actual_batch_size < len(batch_results):
            batch_results = batch_results[:actual_batch_size]
            state = trim_state_padding(state, actual_batch_size)
        
        # Move results to CPU to save GPU memory
        batch_results = jax.device_put(batch_results, jax.devices("cpu")[0])
        all_results.append(batch_results)
        print(f"Batch {i + 1}/{n_batches} completed")
        
        del batch_results  # Free memory
        gc.collect()  # Force garbage collection
    
    # Combine results
    results = jnp.concatenate(all_results, axis=0)
    
    # Reshape to (n_stim_configs, n_param_sets, n_neurons, n_timepoints)
    return results.reshape(
        sim_params.n_stim_configs, sim_params.n_param_sets,
        sim_params.n_neurons, len(sim_params.t_axis)
    ), state
    
    
# Main interface function
def run_vnc_prune_optimized(cfg: DictConfig) -> jnp.ndarray:
    """
    Main function to run optimized VNC pruning simulation.

    Args:
        cfg: Configuration containing all simulation parameters
        
    Returns:
        Results array of shape (n_stim_configs, n_param_sets, n_neurons, n_timepoints)
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
    
    # Determine simulation type
    simulation_type = "prune"

    print(f"Running {simulation_type} simulation with:")
    print(f"  {n_stim_configs} stimulus configurations")
    print(f"  {sim_params.n_param_sets} parameter sets")
    print(f"  {sim_params.n_neurons} neurons")
    print(f"  {len(sim_params.t_axis)} time points")
    
    # Run simulation
    start_time = time.time()
    results, state = run_prune_batched(
        neuron_params, sim_params, simulation_type,
        batch_size=getattr(cfg.experiment, "batch_size", None),
        oscillation_threshold=getattr(cfg.experiment, "oscillation_threshold", 0.5)
    )
    
    elapsed = time.time() - start_time
    total_sims = n_stim_configs * sim_params.n_param_sets
    
    print(f"\nSimulation completed:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Total simulations: {total_sims}")
    print(f"  Time per simulation: {elapsed/total_sims:.4f} seconds")
    print(f"  Results shape: {results.shape}")

    return results, state


def save_state(state: Pruning_state, path: str):
    """
    Save the pruning state to a file.
    
    Args:
        state: Pruning_state object containing simulation state
        path: Path where to save the state
    """
    import pickle
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path: str) -> Pruning_state:
    """
    Load the pruning state from a file.

    Args:
        path: Path to the file containing the pruning state

    Returns:
        Pruning_state object with the loaded state
    """
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)