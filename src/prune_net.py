import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from src.sim_utils import *
from src.optimized_vnc import *


class Pruning_state(NamedTuple):
    """Parameters for pruning simulation."""
    W_mask: jnp.ndarray
    interneuron_mask: jnp.ndarray
    level: jnp.ndarray
    total_removed_neurons: jnp.ndarray
    removed_stim_neurons: jnp.ndarray
    neurons_put_back: jnp.ndarray
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
    """Update state for a single simulation - fixed version with proper restoration logic"""
    
    # Unpack state
    (W_mask, interneuron_mask, level, total_removed_neurons, removed_stim_neurons,
        neurons_put_back, last_removed, remove_p, min_circuit, key) = state

    # Get active MN activity using JAX-compatible approach
    max_frs = jnp.max(R, axis=-1)
    mn_mask = jnp.isin(jnp.arange(R.shape[0]), mn_idxs)
    active_mask = ((max_frs>0) & mn_mask)

    # Compute oscillation score
    oscillation_score, _ = compute_oscillation_score(R[..., clip_start:], active_mask, prominence=0.05)

    # Check if oscillation is below threshold or NaN
    reset_condition = (oscillation_score < oscillation_threshold) | jnp.isnan(oscillation_score)

    key_next, subkey_continue, subkey_reset = random.split(key, 3)

    # === CONTINUE BRANCH: Normal pruning (oscillation is good) ===
    # Identify currently silent interneurons (these will be permanently removed)
    silent_interneurons = interneuron_mask & (max_frs <= 0)
    
    # Permanently remove silent interneurons
    total_removed_continue = total_removed_neurons | silent_interneurons
    
    # Update probabilities - exclude non-interneurons and removed neurons
    exclude_mask_continue = (~interneuron_mask) | total_removed_continue
    p_continue = removal_probability(max_frs, exclude_mask_continue)
    
    # Sample new neuron to remove (only from available interneurons)
    neuron_idx_continue = jax_choice(subkey_continue, jnp.arange(len(max_frs)), p_continue)
    
    # Update removed neurons
    removed_stim_continue = removed_stim_neurons.at[neuron_idx_continue].set(True)
    total_removed_continue = total_removed_continue.at[neuron_idx_continue].set(True)
    
    # Track what was removed this iteration (both silent and stimulated)
    newly_silent_continue = silent_interneurons & (~total_removed_neurons)  # Only newly silent
    last_removed_continue = jnp.full(last_removed.shape, False, dtype=jnp.bool_)
    last_removed_continue = last_removed_continue.at[neuron_idx_continue].set(True)  # Stimulated removal
    last_removed_continue = last_removed_continue | newly_silent_continue  # Add newly silent
    
    # Update other state
    level_continue = level + 1
    neurons_put_back_continue = neurons_put_back  # Unchanged
    min_circuit_continue = False  # Not converged yet

    # === RESET BRANCH: Restore last removed and try again ===
    # Restore ALL neurons from last_removed (both stimulated and those that went silent)
    # This includes neurons that went silent due to the last stimulated removal
    
    # Restore stimulated neurons from last removal
    removed_stim_reset = removed_stim_neurons & (~last_removed)
    
    # For total_removed: keep permanent removals from before last iteration, 
    # add current silent neurons, but restore all last_removed neurons
    permanent_before_last = total_removed_neurons & (~last_removed)
    # Current silent neurons are those silent now (may include some that weren't silent before)
    # But we need to be careful not to restore neurons that are currently silent due to OTHER reasons
    # Only add neurons to total_removed if they are silent AND were not in last_removed
    currently_silent_not_restored = (~last_removed)
    total_removed_reset = permanent_before_last | currently_silent_not_restored
    
    # Track neurons being put back - ALL neurons from last_removed
    # This includes both the stimulated neuron and any neurons that went silent due to that removal
    restored_neurons = last_removed  # All neurons from last_removed are being restored
    neurons_put_back_reset = neurons_put_back | restored_neurons
    
    # Now select a different neuron to remove (avoid the restored ones)
    exclude_mask_reset = (~interneuron_mask) | total_removed_reset | restored_neurons
    p_reset = removal_probability(max_frs, exclude_mask_reset)
    
    # Check how many neurons are available
    available_neurons_reset = jnp.sum(interneuron_mask & (~exclude_mask_reset))
    
    # Select neuron to remove
    neuron_idx_reset = jax_choice(subkey_reset, jnp.arange(len(max_frs)), p_reset)
    
    # Only update if we have available neurons (otherwise keep current state)
    should_remove_new = available_neurons_reset > 0
    removed_stim_reset = jax.lax.select(
        should_remove_new,
        removed_stim_reset.at[neuron_idx_reset].set(True),
        removed_stim_reset
    )
    total_removed_reset = jax.lax.select(
        should_remove_new,
        total_removed_reset.at[neuron_idx_reset].set(True),
        total_removed_reset
    )
    
    # Track what was newly removed this iteration
    last_removed_reset = jnp.full(last_removed.shape, False, dtype=jnp.bool_)
    last_removed_reset = jax.lax.select(
        should_remove_new,
        last_removed_reset.at[neuron_idx_reset].set(True),
        last_removed_reset
    )
    
    # Add any newly silent neurons (those that are silent now but weren't in total_removed_neurons before)
    # These are neurons that became silent due to current network state, not due to last removal
    newly_silent_reset = silent_interneurons & (~total_removed_neurons) & (~last_removed)
    last_removed_reset = last_removed_reset | newly_silent_reset
    
    # Keep level the same (we're trying again, not progressing)
    level_reset = level
    
    # Check if we've converged - either no more neurons to remove OR we're oscillating
    # Oscillation detection: if we're restoring neurons we've put back before, we're in a loop
    oscillation_detected = jnp.any(restored_neurons & neurons_put_back)
    min_circuit_reset = (available_neurons_reset <= 2) | oscillation_detected

    # === SELECT BETWEEN BRANCHES ===
    # Use jax.lax.select to choose between continue and reset results
    final_total_removed = jax.lax.select(reset_condition, total_removed_reset, total_removed_continue)
    final_removed_stim = jax.lax.select(reset_condition, removed_stim_reset, removed_stim_continue)
    final_last_removed = jax.lax.select(reset_condition, last_removed_reset, last_removed_continue)
    final_neurons_put_back = jax.lax.select(reset_condition, neurons_put_back_reset, neurons_put_back_continue)
    final_level = jax.lax.select(reset_condition, level_reset, level_continue)
    final_p = jax.lax.select(reset_condition, p_reset, p_continue)
    final_min_circuit = jax.lax.select(reset_condition, min_circuit_reset, min_circuit_continue)

    # Calculate available neurons and check for convergence
    available_neurons = jnp.sum(interneuron_mask & (~final_total_removed))
    
    # Check for oscillation: if we're in reset mode and detected oscillation, we've converged
    oscillation_converged = reset_condition & final_min_circuit
    size_converged = available_neurons <= 2
    final_converged = oscillation_converged | size_converged

    # Update W_mask to reflect removed neurons
    W_mask_init = jnp.ones_like(W_mask, dtype=jnp.float32)
    removed_float = final_total_removed.astype(jnp.float32)
    kept_mask = 1.0 - removed_float
    W_mask_new = W_mask_init * kept_mask[:, None] * kept_mask[None, :]

    # Convert to scalar for jax.lax.select
    final_converged_scalar = jnp.squeeze(final_converged)

    # Debug information
    jax.debug.print("Oscillation score: {score}", score=oscillation_score)
    jax.debug.print("Reset condition (below threshold): {condition}", condition=reset_condition)
    jax.debug.print("Available neurons: {count}", count=available_neurons)
    jax.debug.print("Level: {level}", level=final_level)
    jax.debug.print("Oscillation detected: {detected}", detected=reset_condition & (final_min_circuit & ~size_converged))
    jax.debug.print("Final converged: {converged}", converged=final_converged)
    jax.debug.print("Silent neurons removed: {count}", count=jnp.sum(silent_interneurons))
    print('\n')
    
    # When converged, preserve current state (don't make further changes)
    return Pruning_state(
        W_mask=jax.lax.select(final_converged_scalar, W_mask, W_mask_new),
        interneuron_mask=interneuron_mask,
        level=jax.lax.select(final_converged_scalar, level, final_level),
        total_removed_neurons=jax.lax.select(final_converged_scalar, total_removed_neurons, final_total_removed),
        neurons_put_back=jax.lax.select(final_converged_scalar, neurons_put_back, final_neurons_put_back),
        removed_stim_neurons=jax.lax.select(final_converged_scalar, removed_stim_neurons, final_removed_stim),
        last_removed=jax.lax.select(final_converged_scalar, last_removed, final_last_removed),
        remove_p=jax.lax.select(final_converged_scalar, remove_p, final_p),
        min_circuit=final_converged_scalar,
        keys=key_next,
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
    
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(
            sim_params.n_neurons, len(sim_params.t_axis)
        )
    
    # Select batch processing function
    if simulation_type == "shuffle":
        batch_func = process_batch_shuffle
    elif simulation_type == "noise":
        batch_func = process_batch_noise
    elif simulation_type == "prune":
        batch_func = process_batch_prune
    else:
        batch_func = process_batch_baseline
    
    # Create parallel version for multiple devices
    if jax.device_count() > 1:
        batch_func = pmap(batch_func, axis_name="device", in_axes=(None, None, 0))
        batch_size = (batch_size // jax.device_count()) * jax.device_count()
    
    print(f"Running {total_sims} simulations with batch size {batch_size}")
    
    # Process in batches
    all_results = []
    n_batches = (total_sims + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_sims)
        
        batch_indices = jnp.arange(start_idx, end_idx)
        
        # Pad if necessary for pmap
        if jax.device_count() > 1 and len(batch_indices) < batch_size:
            pad_size = batch_size - len(batch_indices)
            batch_indices = jnp.concatenate([
                batch_indices, 
                jnp.repeat(batch_indices[-1], pad_size)
            ])
        
        
        # mn_idxs = jnp.asarray(w_table.loc[w_table["class"]=="motor neuron"].index.values)
        mn_idxs = neuron_params.mn_idxs
        all_neurons = jnp.arange(neuron_params.W.shape[-1])
        in_idxs = jnp.setdiff1d(all_neurons, mn_idxs)

        clip_start = int(sim_params.pulse_start / sim_params.dt) + 100
        # Initialize state
        n_sims = sim_params.n_param_sets * sim_params.n_stim_configs
        W_mask = jnp.full((n_sims, neuron_params.W.shape[0], neuron_params.W.shape[1]), 1, dtype=jnp.float32)
        interneuron_mask = jnp.full((n_sims, W_mask.shape[-1]),fill_value=False, dtype=jnp.bool_)
        interneuron_mask = interneuron_mask.at[:,in_idxs].set(True)
        level = jnp.zeros((n_sims,1), dtype=jnp.int32)
        total_removed_neurons = jnp.full((n_sims, W_mask.shape[-1]), False, dtype=jnp.bool)
        removed_stim_neurons = jnp.full([n_sims, total_removed_neurons.shape[-1]], False, dtype=jnp.bool)
        neurons_put_back =  jnp.full([n_sims, total_removed_neurons.shape[-1]], False, dtype=jnp.bool)
        last_removed =  jnp.full([n_sims, total_removed_neurons.shape[-1]], False, dtype=jnp.bool) # Use False for None
        min_circuit = jnp.full((n_sims,1), False)

        # Initialize probabilities
        exclude_mask = (~interneuron_mask)
        p_arrays = jax.vmap(removal_probability)(jnp.ones(len(interneuron_mask)), exclude_mask)

        # initialize pruning state
        state = Pruning_state(
            W_mask=W_mask,
            interneuron_mask=interneuron_mask,
            level=level,
            total_removed_neurons=total_removed_neurons,
            removed_stim_neurons=removed_stim_neurons,
            neurons_put_back=neurons_put_back,
            last_removed=last_removed,
            remove_p=p_arrays,
            min_circuit=min_circuit,
            keys=neuron_params.seeds
        )

        iter_start = 0  # Starting iteration
        # Main pruning loop
        iteration = iter_start
        max_iterations = 200  # Safety limit
        while ~jnp.all(state.min_circuit) & (iteration < max_iterations):
            start_time = time.time()
            print(f"Iteration {iteration}")
            # Update neuron parameters W_mask based on the current state
            neuron_params = update_params(neuron_params, W_mask=state.W_mask)
            # Run simulation (this would call your simulation function)
            # Reshape for devices if using pmap
            if jax.device_count() > 1:
                batch_indices = batch_indices.reshape(jax.device_count(), -1)
                batch_results = batch_func(neuron_params, sim_params, batch_indices)
                batch_results = batch_results.reshape(-1, *batch_results.shape[2:])
                batch_results = batch_results[:end_idx - start_idx]  # Remove padding
            else:
                batch_results = batch_func(neuron_params, sim_params, batch_indices)
            
            state = jax.vmap(update_single_sim_state, in_axes=(0, 0, None, None, None))(state, batch_results, mn_idxs, oscillation_threshold, clip_start)
            iteration += 1
            
            elapsed = time.time() - start_time
            print(f"  Sim time: {elapsed:.2f} seconds")
        
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
        batch_size=getattr(cfg.experiment, "batch_size", None)
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
    
    
    
    
    
'''
the removal of silent interneurons is permanent. The silent neurons are only ever taken out on sims that passed the oscillation score threshold so there shouldn’t be a case where you took a neuron out that caused score < threshold and have silent neurons to put back alongside it.
yes I exclude neurons that were already put back this “round” when sampling ones to removed - this is the neuronsPutBack list
the convergence works like this:
start with the full network, remove neurons and have the ones you had to put back in a list (neuronsPutBack). Once you have no interneurons left in the circuit that aren’t already on this list this “round” is done.
start another “round” with the network that you’re left with: do the same procedure again on this smaller set of interneurons. Then compare the new list of neuronsPutBack with the previous list (prevNeuronsPutBack). If these two lists are not the same, repeat. If they are the same, that means you tried to remove all the neurons you started with and you couldn’t remove any of them. That’s when you converge.
if ~np.isfinite(np.sum(p)):
    # NO NEURONS LEFT TO REMOVE
    if set(neuronsPutBack) == set(prevNeuronsPutBack):
        print("converged to minimal circuit")
        return params
        
'''