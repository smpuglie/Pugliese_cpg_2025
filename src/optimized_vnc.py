# Optimized JAX/JIT VNC Network Simulation - Pure Functional Approach
import time
from typing import NamedTuple, Dict, Any, Optional, Tuple, List
import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax import vmap, jit, random, pmap, lax
from omegaconf import DictConfig
import psutil
import gc

# Import utility functions (assumed to exist)
from src.shuffle_utils import extract_shuffle_indices, full_shuffle
from src.sim_utils import (
    load_W, load_wTable, sample_trunc_normal, set_sizes, make_input
)


# Immutable configuration structures
class NeuronParams(NamedTuple):
    """Immutable neuron parameters for JIT compilation."""
    W: jnp.ndarray
    W_mask: jnp.ndarray # Mask for connectivity
    tau: jnp.ndarray # Time constants for neurons
    a: jnp.ndarray # Activation parameters
    threshold: jnp.ndarray # Thresholds for activation
    fr_cap: jnp.ndarray # Firing rate caps
    input_currents: jnp.ndarray  # Shape: (n_stim_configs, n_neurons)
    seeds: jnp.ndarray # Random seeds for reproducibility
    exc_dn_idxs: jnp.ndarray # Indices of excitatory descending neurons
    inh_dn_idxs: jnp.ndarray # Indices of inhibitory descending neurons
    exc_in_idxs: jnp.ndarray # Indices of excitatory interneurons
    inh_in_idxs: jnp.ndarray # Indices of inhibitory interneurons
    mn_idxs: jnp.ndarray # Indices of motor neurons


class SimParams(NamedTuple):
    """Immutable simulation parameters for JIT compilation."""
    n_neurons: int
    n_param_sets: int
    n_stim_configs: int
    T: float
    dt: float
    pulse_start: float
    pulse_end: float
    t_axis: jnp.ndarray
    exc_multiplier: float
    inh_multiplier: float
    noise_stdv_prop: float
    r_tol: float
    a_tol: float
    noise_stdv: jnp.ndarray  # Standard deviation for noise, if applicable

def update_params(params, **kwargs):
    """
    Update specific parameters of a NeuronParams instance.
    
    Args:
        params: The original NeuronParams instance
        **kwargs: Keyword arguments specifying which parameters to update
                 (parameter_name=new_value)
    
    Returns:
        New NeuronParams instance with updated parameters
        
    Example:
        # Update tau and threshold parameters
        new_params = update_neuron_params(
            old_params, 
            tau=new_tau_array,
            threshold=new_threshold_array
        )
    """
    # Get current values as a dict
    current_dict = params._asdict()
    
    # Update with new values
    current_dict.update(kwargs)
    
    # Create new instance
    return NeuronParams(**current_dict)

# Core simulation functions - all JIT compiled
@jit
def rate_equation_half_tanh(t, R, args):
    """ODE rate equation with half-tanh activation."""
    inputs, pulse_start, pulse_end, tau, weighted_W, threshold, a, fr_cap, key, noise_stdv = args
    pulse_active = (t >= pulse_start) & (t <= pulse_end)
    I = inputs * pulse_active + pulse_active*jax.random.normal(key, shape=inputs.shape) * noise_stdv  # Add noise to inputs
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
def add_noise_to_weights(W: jnp.ndarray, seed: int, stdv_prop: float) -> jnp.ndarray:
    """Add truncated normal noise to weight matrix."""
    stdvs = W * stdv_prop
    key = random.key(seed)
    noise = random.truncated_normal(
        key, lower=-1.0 / stdv_prop, upper=jnp.inf, shape=W.shape
    )
    return W + stdvs * noise


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
    controller = PIDController(rtol=r_tol, atol=a_tol)
    
    solution = diffeqsolve(
        term, solver, 0, T, dt, R0,
        args=(inputs, pulse_start, pulse_end, tau, W, threshold, a, fr_cap, key, noise_stdv),
        saveat=saveat, stepsize_controller=controller,
        max_steps=100000, throw=False
    )
    return jnp.transpose(solution.ys)


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
        W_reweighted, tau, a, threshold, fr_cap, inputs,
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
        W_reweighted, tau, a, threshold, fr_cap, inputs,
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
    noise_stdv: jnp.ndarray,
    sim_params: SimParams,
    seed: jnp.ndarray
) -> jnp.ndarray:
    """Run baseline simulation without shuffle or noise."""
    W_reweighted = reweight_connectivity(W, sim_params.exc_multiplier, sim_params.inh_multiplier)
    
    return run_single_simulation(
        W_reweighted, tau, a, threshold, fr_cap, inputs, noise_stdv, 
        sim_params.t_axis, sim_params.T, sim_params.dt,
        sim_params.pulse_start, sim_params.pulse_end,
        sim_params.r_tol, sim_params.a_tol, seed, 
    )


@jit
def run_with_prune(
    W: jnp.ndarray,
    W_mask: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    sim_params: SimParams
) -> jnp.ndarray:
    """Run pruning simulation."""
    
    W_masked = W * W_mask
    W_reweighted = reweight_connectivity(W_masked, sim_params.exc_multiplier, sim_params.inh_multiplier)
    return run_single_simulation(
        W_reweighted, tau, a, threshold, fr_cap, inputs,
        sim_params.t_axis, sim_params.T, sim_params.dt,
        sim_params.pulse_start, sim_params.pulse_end,
        sim_params.r_tol, sim_params.a_tol
    )


# Vectorized batch processing functions
@jit
def process_batch_prune(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    batch_indices: jnp.ndarray
) -> jnp.ndarray:
    """Process a batch of simulations with pruning."""
    
    def single_sim(idx):
        param_idx = idx % sim_params.n_param_sets
        stim_idx = idx // sim_params.n_param_sets
        
        return run_with_prune(
            neuron_params.W,
            neuron_params.W_mask[param_idx],
            neuron_params.tau[param_idx],
            neuron_params.a[param_idx],
            neuron_params.threshold[param_idx],
            neuron_params.fr_cap[param_idx],
            neuron_params.input_currents[stim_idx],
            sim_params
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
            neuron_params.input_currents[stim_idx],
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
            neuron_params.input_currents[stim_idx],
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
            neuron_params.input_currents[stim_idx],
            sim_params.noise_stdv,
            sim_params,
            neuron_params.seeds[param_idx]
        )
    
    return vmap(single_sim)(batch_indices)


# Efficient batch size calculation
def calculate_optimal_batch_size(n_neurons: int, n_timepoints: int, n_devices: int = None) -> int:
    """Calculate optimal batch size based on memory and device constraints."""
    if n_devices is None:
        n_devices = jax.device_count()
    
    # Estimate memory usage
    available_memory = psutil.virtual_memory().available
    bytes_per_sample = n_neurons * n_timepoints * 8  # float64 + overhead
    max_memory_samples = int(available_memory * 0.6 / bytes_per_sample)
    
    # Balance with device count
    optimal_batch = max(n_devices * 4, min(max_memory_samples, 256))
    
    # Ensure divisible by device count
    return (optimal_batch // n_devices) * n_devices


# Main simulation orchestration functions
def run_simulation_batched(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    simulation_type: str = "baseline",
    batch_size: Optional[int] = None
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
        
        # Reshape for devices if using pmap
        if jax.device_count() > 1:
            batch_indices = batch_indices.reshape(jax.device_count(), -1)
            batch_results = batch_func(neuron_params, sim_params, batch_indices)
            batch_results = batch_results.reshape(-1, *batch_results.shape[2:])
            batch_results = batch_results[:end_idx - start_idx]  # Remove padding
        else:
            batch_results = batch_func(neuron_params, sim_params, batch_indices)
        
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
    )


# Configuration loading and setup functions
def prepare_neuron_params(cfg: DictConfig, W_table: Any) -> NeuronParams:
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
    input_currents_list = []
    for stim_neurons, stim_input in zip(cfg.experiment.stimNeurons, cfg.experiment.stimI):
        input_current = jnp.array(make_input(n_neurons, jnp.array(stim_neurons), stim_input))
        input_currents_list.append(input_current)
    
    input_currents = jnp.array(input_currents_list)
    
    # Extract shuffle indices
    exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs = extract_shuffle_indices(W_table)
    W_mask = jnp.ones((num_sims, W.shape[0], W.shape[1]), dtype=jnp.float32)
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
    t_axis = jnp.arange(0, T, dt)
    
    return SimParams(
        n_neurons=n_neurons,
        n_param_sets=cfg.experiment.n_replicates,
        n_stim_configs=n_stim_configs,
        T=T, dt=dt,
        pulse_start=float(cfg.sim.pulseStart),
        pulse_end=float(cfg.sim.pulseEnd),
        t_axis=t_axis,
        exc_multiplier=float(cfg.neuron_params.excitatoryMultiplier),
        inh_multiplier=float(cfg.neuron_params.inhibitoryMultiplier),
        noise_stdv_prop=getattr(cfg.sim, "noiseStdvProp", 0.1),
        r_tol=getattr(cfg.sim, "rtol", 1e-7),
        a_tol=getattr(cfg.sim, "atol", 1e-9),        
        noise_stdv=cfg.sim.noiseStdv
    )


# Main interface function
def run_vnc_simulation_optimized(cfg: DictConfig) -> jnp.ndarray:
    """
    Main function to run optimized VNC simulation.
    
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
    simulation_type = "baseline"
    if getattr(cfg.sim, "shuffle", False):
        simulation_type = "shuffle"
    elif getattr(cfg.sim, "noise", False):
        simulation_type = "noise"
    
    print(f"Running {simulation_type} simulation with:")
    print(f"  {n_stim_configs} stimulus configurations")
    print(f"  {sim_params.n_param_sets} parameter sets")
    print(f"  {sim_params.n_neurons} neurons")
    print(f"  {len(sim_params.t_axis)} time points")
    
    # Run simulation
    start_time = time.time()
    results = run_simulation_batched(
        neuron_params, sim_params, simulation_type,
        batch_size=getattr(cfg.experiment, "batch_size", None)
    )
    
    elapsed = time.time() - start_time
    total_sims = n_stim_configs * sim_params.n_param_sets
    
    print(f"\nSimulation completed:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Total simulations: {total_sims}")
    print(f"  Time per simulation: {elapsed/total_sims:.4f} seconds")
    print(f"  Throughput: {total_sims/elapsed:.2f} sims/second")
    print(f"  Results shape: {results.shape}")
    
    return results