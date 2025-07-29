import time
from typing import List, NamedTuple, Dict, Any
import jax
import jax.numpy as jnp
# import numpy as np
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax import vmap, jit, random, pmap
from omegaconf import DictConfig
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import gc
from src.shuffle_utils import extract_shuffle_indices, full_shuffle
from src.sim_utils import (
    load_W,
    load_wTable,
    sample_trunc_normal,
    set_sizes,
    make_input,
)


class NeuronConfig(NamedTuple):

    W: jnp.ndarray
    W_table: Any
    tau: jnp.ndarray
    a: jnp.ndarray
    threshold: jnp.ndarray
    fr_cap: jnp.ndarray
    input_currents: jnp.ndarray
    seeds: jnp.ndarray
    stim_neurons: List[int]
    exc_dn_idxs: jnp.ndarray
    inh_dn_idxs: jnp.ndarray
    exc_in_idxs: jnp.ndarray
    inh_in_idxs: jnp.ndarray
    mn_idxs: jnp.ndarray


class SimConfig(NamedTuple):

    n_neurons: int
    num_sims: int
    n_replicates: int
    T: float
    dt: float
    pulse_start: float
    pulse_end: float
    t_axis: jnp.ndarray
    stim_input: float
    shuffle: bool
    noise: bool
    noise_stdv_prop: float
    exc_synapse_multiplier: float
    inh_synapse_multiplier: float
    r_tol: float
    a_tol: float
    rng_key: jax.random.PRNGKey


def rate_equation_half_tanh(t, R, args):
    inputs, pulseStart, pulseEnd, tau, weightedW, threshold, a, frCap = args

    pulse_active = (t >= pulseStart) & (t <= pulseEnd)
    I = inputs * pulse_active

    totalInput = I + jnp.dot(weightedW, R)
    activation = jnp.maximum(
        frCap * jnp.tanh((a / frCap) * (totalInput - threshold)), 0
    )

    return (activation - R) / tau


def run(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Pure jax function that runs the actual simulation"""
    term = ODETerm(rate_equation_half_tanh)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.array(tAxis))
    controller = PIDController(rtol=r_tol, atol=a_tol)

    odeSolution = diffeqsolve(
        term,
        solver,
        0,
        T,
        dt,
        R0,
        args=(inputs, pulseStart, pulseEnd, tau, W, threshold, a, frCap),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=100000,
        throw=False,
    )
    return jnp.transpose(odeSolution.ys)


def run_shuffle(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    seed,
    extra,
    excDnIdxs,
    inhDnIdxs,
    excInIdxs,
    inhInIdxs,
    mnIdxs,
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Pure jax function that shuffles the weight matrix"""
    W_shuffled = full_shuffle(
        W, seed, excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs
    )

    return run(
        W_shuffled,
        R0,
        tAxis,
        T,
        dt,
        inputs,
        pulseStart,
        pulseEnd,
        tau,
        threshold,
        a,
        frCap,
        r_tol,
        a_tol,
    )


def run_noise(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    seed,
    stdvProp,
    excDnIdxs,  # Added to match signature (ignored)
    inhDnIdxs,  # Added to match signature (ignored)
    excInIdxs,  # Added to match signature (ignored)
    inhInIdxs,  # Added to match signature (ignored)
    mnIdxs,  # Added to match signature (ignored)
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Pure jax function that adds noise to the weight matrix"""
    W_noisy = resample_W(W, seed, stdvProp)

    return run(
        W_noisy,
        R0,
        tAxis,
        T,
        dt,
        inputs,
        pulseStart,
        pulseEnd,
        tau,
        threshold,
        a,
        frCap,
        r_tol,
        a_tol,
    )


def run_no_shuffle(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    seed,  # Added to match signature (ignored)
    stdvProp,  # Added to match signature (ignored)
    excDnIdxs,  # Added to match signature (ignored)
    inhDnIdxs,  # Added to match signature (ignored)
    excInIdxs,  # Added to match signature (ignored)
    inhInIdxs,  # Added to match signature (ignored)
    mnIdxs,  # Added to match signature (ignored)
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Wrapper for run that ignores shuffle and noise parameters."""
    return run(
        W,
        R0,
        tAxis,
        T,
        dt,
        inputs,
        pulseStart,
        pulseEnd,
        tau,
        threshold,
        a,
        frCap,
        r_tol,
        a_tol,
    )


def validate_config(cfg: DictConfig) -> None:
    """Check if config has everything."""
    required_sections = {
        "experiment": ["name", "wPath", "dfPath", "stimNeurons", "stimI", "seed"],
        "sim": ["T", "dt", "pulseStart", "pulseEnd"],
        "neuron_params": [
            "tauMean",
            "tauStdv",
            "aMean",
            "aStdv",
            "thresholdMean",
            "thresholdStdv",
            "frcapMean",
            "frcapStdv",
            "excitatoryMultiplier",
            "inhibitoryMultiplier",
        ],
    }

    for section, keys in required_sections.items():
        if not hasattr(cfg, section):
            raise ValueError(f"Missing required config section: {section}")
        section_cfg = getattr(cfg, section)
        for key in keys:
            if not hasattr(section_cfg, key):
                raise ValueError(
                    f"Missing required config key '{key}' in section '{section}'"
                )


def load_connectivity(cfg: DictConfig) -> tuple[jnp.ndarray, Any]:
    """Load connectivity matrix and the data."""
    w_path = cfg.experiment.wPath
    df_path = cfg.experiment.dfPath
    W = load_W(w_path)
    W_table = load_wTable(df_path)
    return jnp.array(W), W_table


def sample_neuron_parameters(
    cfg: DictConfig, n_neurons: int, num_sims: int, seeds: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample neuron parameters based on configuration."""
    np_cfg = cfg.neuron_params

    keys = jax.random.split(seeds, 4)
    tau = sample_trunc_normal(
        key = keys[0],
        mean = float(np_cfg.tauMean),
        stdev = float(np_cfg.tauStdv),
        shape = (num_sims, n_neurons)
    )
    a = sample_trunc_normal(
        key = keys[1],
        mean = float(np_cfg.aMean),
        stdev = float(np_cfg.aStdv),
        shape = (num_sims, n_neurons)
    )
    threshold = sample_trunc_normal(
        key = keys[2],
        mean = float(np_cfg.thresholdMean),
        stdev = float(np_cfg.thresholdStdv),
        shape = (num_sims, n_neurons)
    )
    fr_cap = sample_trunc_normal(
        key = keys[3],
        mean = float(np_cfg.frcapMean),
        stdev = float(np_cfg.frcapStdv),
        shape = (num_sims, n_neurons)
    )

    return jnp.array(tau), jnp.array(a), jnp.array(threshold), jnp.array(fr_cap)


def create_sim_config(cfg: DictConfig, W: jnp.ndarray) -> SimConfig:
    """Create simulation configuration from config and connectivity matrix."""
    n_neurons = W.shape[0]
    num_sims = cfg.experiment.n_replicates * len(cfg.experiment.stimNeurons)
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)
    # seeds = jnp.array(cfg.experiment.seed)

    T = float(cfg.sim.T)
    dt = float(cfg.sim.dt)
    pulse_start = float(cfg.sim.pulseStart)
    pulse_end = float(cfg.sim.pulseEnd)
    t_axis = jnp.array(jnp.arange(0, T, dt))

    stim_input = cfg.experiment.stimI
    shuffle = getattr(cfg.sim, "shuffle", False)
    noise = getattr(cfg.sim, "noise", False)
    noise_stdv_prop = getattr(cfg.sim, "noiseStdvProp", 0.1)

    exc_synapse_multiplier = float(cfg.neuron_params.excitatoryMultiplier)
    inh_synapse_multiplier = float(cfg.neuron_params.inhibitoryMultiplier)

    r_tol = getattr(cfg.sim, "rtol", 1e-7)
    a_tol = getattr(cfg.sim, "atol", 1e-9)

    return SimConfig(
        n_neurons=n_neurons,
        num_sims=num_sims,
        n_replicates=cfg.experiment.n_replicates,
        T=T,
        dt=dt,
        pulse_start=pulse_start,
        pulse_end=pulse_end,
        t_axis=t_axis,
        stim_input=stim_input,
        shuffle=shuffle,
        noise=noise,
        noise_stdv_prop=noise_stdv_prop,
        exc_synapse_multiplier=exc_synapse_multiplier,
        inh_synapse_multiplier=inh_synapse_multiplier,
        r_tol=r_tol,
        a_tol=a_tol,
        rng_key=rng_key,
    )


def load_vnc_net(cfg: DictConfig) -> tuple[NeuronConfig, SimConfig]:
    """Load and configure VNC network parameters and simulation settings."""
    validate_config(cfg)

    W, W_table = load_connectivity(cfg)
    sim_config = create_sim_config(cfg, W)

    stim_neurons = jnp.array(cfg.experiment.stimNeurons, dtype=jnp.int32)
    # seeds = np.array(cfg.experiment.seed)
    newkey, subkey = jax.random.split(sim_config.rng_key)
    newkey = jax.random.split(newkey, sim_config.num_sims)
    tau, a, threshold, fr_cap = sample_neuron_parameters(
        cfg, sim_config.n_neurons, sim_config.num_sims, subkey
    )

    if "size" in W_table:
        a, threshold = set_sizes(W_table["size"].values, a, threshold)
    else:
        a, threshold = set_sizes(W_table["surf_area_um2"].values, a, threshold)

    input_currents = jnp.array(
        make_input(sim_config.n_neurons, stim_neurons, sim_config.stim_input)
    )

    exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs = (
        extract_shuffle_indices(W_table)
    )

    params = NeuronConfig(
        W=W,
        W_table=W_table,
        tau=tau,
        a=a,
        threshold=threshold,
        fr_cap=fr_cap,
        input_currents=input_currents,
        seeds=newkey,
        stim_neurons=stim_neurons,
        exc_dn_idxs=exc_dn_idxs,
        inh_dn_idxs=inh_dn_idxs,
        exc_in_idxs=exc_in_idxs,
        inh_in_idxs=inh_in_idxs,
        mn_idxs=mn_idxs,
    )
    return params, sim_config


def resample_W(W, seed, stdvProp):
    """Add noise to weight matrix."""
    stdvs = W * stdvProp
    key = random.key(seed)
    samples = random.truncated_normal(
        key, lower=-1.0 / stdvProp, upper=float("inf"), shape=W.shape
    )
    return W + stdvs * samples


@jit
def reweight_connectivity(
    W: jnp.ndarray, exc_multiplier: float, inh_multiplier: float
) -> jnp.ndarray:
    """Reweight connectivity matrix with excitatory and inhibitory multipliers."""
    Wt_exc = jnp.maximum(W, 0)
    Wt_inh = jnp.minimum(W, 0)
    W_rw = jnp.transpose(exc_multiplier * Wt_exc + inh_multiplier * Wt_inh)
    return W_rw

def run_batch(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau_batch,
    threshold_batch,
    a_batch,
    frCap_batch,
    seed_batch,
    stdvProp_batch,
    excDnIdxs,
    inhDnIdxs,
    excInIdxs,
    inhInIdxs,
    mnIdxs,
    run_func,
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Run a batch of simulations using vmap"""
    vmap_axes = (
        None, # W
        None, # R0
        None, # tAxis
        None, # T
        None, # dt
        None, # inputs
        None, # pulseStart
        None, # pulseEnd
        0, # tau
        0, # threshold
        0, # a
        0, # frCap
        0, # seed
        0, # stdvProp
        None, # excDnIdxs
        None, # inhDnIdxs
        None, # excInIdxs
        None, # inhInIdxs
        None, # mnIdxs
        None, # r_tol
        None, # a_tol
    )
    
    return vmap(run_func, in_axes=vmap_axes)(
        W,
        R0,
        tAxis,
        T,
        dt,
        inputs,
        pulseStart,
        pulseEnd,
        tau_batch,
        threshold_batch,
        a_batch,
        frCap_batch,
        seed_batch,
        stdvProp_batch,
        excDnIdxs,
        inhDnIdxs,
        excInIdxs,
        inhInIdxs,
        mnIdxs,
        r_tol,
        a_tol,
    )


def process_sequential_minibatches(params: NeuronConfig, config: SimConfig, 
                                  batch_size: int = None, 
                                  save_intermediate: bool = False,
                                  checkpoint_dir: str = None) -> jnp.ndarray:
    """
    Process data in sequential minibatches with optional checkpointing.
    
    Args:
        params: NeuronConfig with simulation parameters
        config: SimConfig with simulation configuration
        batch_size: Size of each minibatch (default: auto-determined)
        save_intermediate: Whether to save intermediate results
        checkpoint_dir: Directory to save checkpoints (if save_intermediate=True)
    
    Returns:
        Combined results from all minibatches
    """
    try:
        start_time = time.time()
        W_rw = reweight_connectivity(
            params.W, config.exc_synapse_multiplier, config.inh_synapse_multiplier
        )
        R0 = jnp.zeros((W_rw.shape[1],))
        
        # Determine which run function to use
        if config.shuffle:
            run_func = run_shuffle
            print("Running simulation with shuffle")
        elif config.noise:
            run_func = run_noise
            print("Running simulation with noise")
        else:
            run_func = run_no_shuffle
            print("Running simulation without shuffle or noise")
        
        n_devices = jax.device_count()
        n_samples = len(params.tau)
        
        # Optimize batch size for memory and devices
        if batch_size is None:
            # Heuristic: balance between memory usage and device utilization
            batch_size = max(n_devices * 4, min(64, n_samples // 4))
        
        print(f"Processing {n_samples} samples in minibatches of size {batch_size} using {n_devices} devices")
        
        # Prepare additional parameter
        if config.noise:
            additional_param = jnp.array([config.noise_stdv_prop] * n_samples)
        else:
            additional_param = jnp.array([0] * n_samples)
        
        # Pre-compile the pmap function
        print("Pre-compiling pmap function...")
        pmap_run_batch = pmap(
            lambda tau_b, threshold_b, a_b, fr_cap_b, seed_b, stdv_b: run_batch(
                W_rw, R0, config.t_axis, config.T, config.dt,
                params.input_currents, config.pulse_start, config.pulse_end,
                tau_b, threshold_b, a_b, fr_cap_b, seed_b, stdv_b,
                params.exc_dn_idxs, params.inh_dn_idxs, params.exc_in_idxs,
                params.inh_in_idxs, params.mn_idxs, run_func,
            ),
            axis_name='batch'
        )
        
        # Initialize results storage
        all_results = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Process each minibatch sequentially
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            actual_batch_size = end_idx - start_idx
            
            batch_start_time = time.time()
            print(f"\nProcessing batch {batch_idx + 1}/{n_batches}")
            print(f"  Samples: {start_idx} to {end_idx-1} ({actual_batch_size} samples)")
            
            # Extract current batch data
            batch_data = {
                'tau': params.tau[start_idx:end_idx],
                'threshold': params.threshold[start_idx:end_idx],
                'a': params.a[start_idx:end_idx],
                'fr_cap': params.fr_cap[start_idx:end_idx],
                'seeds': params.seeds[start_idx:end_idx],
                'additional_param': additional_param[start_idx:end_idx]
            }
            
            # Distribute batch across devices
            device_batches = distribute_batch_across_devices(batch_data, n_devices)
            
            # Run pmap on current batch
            batch_results = pmap_run_batch(
                device_batches['tau'],
                device_batches['threshold'],
                device_batches['a'],
                device_batches['fr_cap'],
                device_batches['seeds'],
                device_batches['additional_param']
            )
            
            # Process results
            batch_results_flat = batch_results.reshape(-1, *batch_results.shape[2:])
            batch_results_trimmed = batch_results_flat[:actual_batch_size]
            
            # Store results
            all_results.append(batch_results_trimmed)
            
            # Optional: Save intermediate results
            if save_intermediate and checkpoint_dir:
                checkpoint_path = checkpoint_dir / f"batch_{batch_idx:04d}.npy"
                jnp.save(checkpoint_path, batch_results_trimmed)
                print(f"  Saved checkpoint: {checkpoint_path}")
            
            batch_time = time.time() - batch_start_time
            print(f"  Batch completed in {batch_time:.2f}s")
            print(f"  Results shape: {batch_results_trimmed.shape}")
            
            # Memory cleanup
            del batch_results, batch_results_flat, batch_results_trimmed
            gc.collect()
        
        # Compile all results
        print(f"\nCompiling results from {len(all_results)} batches...")
        Rs = jnp.concatenate(all_results, axis=0)
        
        total_time = time.time() - start_time
        print(f"\nSimulation completed:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Final result shape: {Rs.shape}")
        print(f"  Average time per sample: {total_time/n_samples:.4f} seconds")
        
        return Rs
        
    except Exception as e:
        print(f"Error in sequential minibatch processing: {e}")
        raise

def create_device_specific_batches(params: NeuronConfig, additional_param, batch_size: int, n_devices: int):
    """Create device-specific batch queues for async processing."""
    n_samples = len(params.tau)
    
    # Create queues for each device
    device_queues = [Queue() for _ in range(n_devices)]
    
    # Distribute samples across devices in round-robin fashion
    for i in range(n_samples):
        device_idx = i % n_devices
        sample_data = {
            'tau': params.tau[i],
            'threshold': params.threshold[i],
            'a': params.a[i],
            'fr_cap': params.fr_cap[i],
            'seeds': params.seeds[i],
            'additional_param': additional_param[i],
            'sample_idx': i
        }
        device_queues[device_idx].put(sample_data)
    
    # Group samples into batches for each device
    device_batches = [[] for _ in range(n_devices)]
    
    for device_idx in range(n_devices):
        current_batch = []
        
        while not device_queues[device_idx].empty():
            sample = device_queues[device_idx].get()
            current_batch.append(sample)
            
            # Create batch when it reaches batch_size or no more samples
            if len(current_batch) >= batch_size or device_queues[device_idx].empty():
                device_batches[device_idx].append(current_batch)
                current_batch = []
    
    return device_batches

def distribute_batch_across_devices(batch_data, n_devices):
    """Distribute a batch across devices for pmap processing."""
    batch_size = len(batch_data['tau'])
    
    # Calculate how to distribute samples across devices
    samples_per_device = batch_size // n_devices
    remainder = batch_size % n_devices
    
    device_batches = {key: [] for key in batch_data.keys()}
    
    current_idx = 0
    for device_idx in range(n_devices):
        # Calculate how many samples this device gets
        device_size = samples_per_device + (1 if device_idx < remainder else 0)
        
        if device_size > 0:
            # Slice data for this device
            for key, data in batch_data.items():
                device_slice = data[current_idx:current_idx + device_size]
                device_batches[key].append(device_slice)
            current_idx += device_size
        else:
            # Empty slice for this device
            for key, data in batch_data.items():
                device_batches[key].append(jnp.array([]))
    
    # Convert to JAX arrays
    for key in device_batches:
        device_batches[key] = jnp.array(device_batches[key])
    
    return device_batches


def process_device_batch(device_id: int, batch_data: list, W_rw, R0, config, params, run_func):
    """Process a single batch on a specific device."""
    if not batch_data:
        return []
    
    # Extract batch arrays
    tau_batch = jnp.array([sample['tau'] for sample in batch_data])
    threshold_batch = jnp.array([sample['threshold'] for sample in batch_data])
    a_batch = jnp.array([sample['a'] for sample in batch_data])
    fr_cap_batch = jnp.array([sample['fr_cap'] for sample in batch_data])
    seeds_batch = jnp.array([sample['seeds'] for sample in batch_data])
    additional_param_batch = jnp.array([sample['additional_param'] for sample in batch_data])
    sample_indices = [sample['sample_idx'] for sample in batch_data]
    
    # Run batch processing with single device
    with jax.default_device(jax.devices()[device_id]):
        batch_results = run_batch(
            W_rw, R0, config.t_axis, config.T, config.dt,
            params.input_currents, config.pulse_start, config.pulse_end,
            tau_batch, threshold_batch, a_batch, fr_cap_batch,
            seeds_batch, additional_param_batch,
            params.exc_dn_idxs, params.inh_dn_idxs, params.exc_in_idxs,
            params.inh_in_idxs, params.mn_idxs, run_func
        )
    
    return [(idx, result) for idx, result in zip(sample_indices, batch_results)]

def async_process_batches(params: NeuronConfig, config: SimConfig, batch_size: int = None,
                         max_concurrent_batches: int = None) -> jnp.ndarray:
    """
    Process batches asynchronously across devices without waiting for synchronization.
    
    Args:
        params: NeuronConfig with simulation parameters
        config: SimConfig with simulation configuration
        batch_size: Size of each batch per device
        max_concurrent_batches: Maximum number of concurrent batches per device
    
    Returns:
        Compiled results from all devices
    """
    try:
        start_time = time.time()
        W_rw = reweight_connectivity(
            params.W, config.exc_synapse_multiplier, config.inh_synapse_multiplier
        )
        R0 = jnp.zeros((W_rw.shape[1],))
        
        # Determine run function
        if config.shuffle:
            run_func = run_shuffle
            print("Running simulation with shuffle")
        elif config.noise:
            run_func = run_noise
            print("Running simulation with noise")
        else:
            run_func = run_no_shuffle
            print("Running simulation without shuffle or noise")
        
        n_devices = jax.device_count()
        n_samples = len(params.tau)
        
        # Set defaults
        if batch_size is None:
            batch_size = max(8, n_samples // (n_devices * 4))  # Smaller batches for async
        
        if max_concurrent_batches is None:
            max_concurrent_batches = 2  # Allow up to 2 concurrent batches per device
        
        print(f"Async processing: {n_samples} samples across {n_devices} devices")
        print(f"Batch size: {batch_size}, Max concurrent batches per device: {max_concurrent_batches}")
        
        # Prepare additional parameters
        if config.noise:
            additional_param = jnp.array([config.noise_stdv_prop] * n_samples)
        else:
            additional_param = jnp.array([0] * n_samples)
        
        # Create device-specific batches
        device_batches = create_device_specific_batches(params, additional_param, batch_size, n_devices)
        
        # Results storage
        results_lock = threading.Lock()
        all_results = {}  # Will store results by sample index
        
        # Device-specific processing functions
        def process_device_async(device_id):
            """Process all batches for a specific device asynchronously."""
            device_batch_list = device_batches[device_id]
            device_results = []
            
            print(f"Device {device_id}: Processing {len(device_batch_list)} batches")
            
            # Process batches for this device
            for batch_idx, batch_data in enumerate(device_batch_list):
                batch_start_time = time.time()
                
                try:
                    batch_results = process_device_batch(
                        device_id, batch_data, W_rw, R0, config, params, run_func
                    )
                    
                    # Store results with thread safety
                    with results_lock:
                        for sample_idx, result in batch_results:
                            all_results[sample_idx] = result
                    
                    batch_time = time.time() - batch_start_time
                    print(f"Device {device_id}, Batch {batch_idx+1}/{len(device_batch_list)}: "
                          f"Processed {len(batch_data)} samples in {batch_time:.2f}s")
                    
                    # Memory cleanup
                    del batch_results
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error in device {device_id}, batch {batch_idx}: {e}")
                    raise
            
            return f"Device {device_id} completed"
        
        # Start async processing on all devices
        print("Starting async processing across devices...")
        
        with ThreadPoolExecutor(max_workers=n_devices) as executor:
            # Submit all device tasks
            future_to_device = {
                executor.submit(process_device_async, device_id): device_id 
                for device_id in range(n_devices)
            }
            
            # Monitor progress
            completed_devices = 0
            for future in future_to_device:
                try:
                    result = future.result()  # This will block until the device finishes
                    print(f"✓ {result}")
                    completed_devices += 1
                    print(f"Progress: {completed_devices}/{n_devices} devices completed")
                    
                except Exception as e:
                    device_id = future_to_device[future]
                    print(f"✗ Device {device_id} failed: {e}")
                    raise
        
        # Compile results in original order
        print("Compiling results...")
        compiled_results = []
        for i in range(n_samples):
            if i in all_results:
                compiled_results.append(all_results[i])
            else:
                raise ValueError(f"Missing result for sample {i}")
        
        Rs = jnp.array(compiled_results)
        
        total_time = time.time() - start_time
        print(f"\nAsync simulation completed:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Final result shape: {Rs.shape}")
        print(f"  Average time per sample: {total_time/n_samples:.4f} seconds")
        print(f"  Speedup from async processing: Devices worked independently!")
        
        return Rs
        
    except Exception as e:
        print(f"Error in async batch processing: {e}")
        raise
    
    
def simulate_vnc_net(params: NeuronConfig, config: SimConfig, batch_size: int = None,
                    save_checkpoints: bool = False,
                    checkpoint_dir: str = "./checkpoints") -> jnp.ndarray:
    """
    Simulate the VNC network with given parameters and configuration.
    
    Args:
        params: NeuronConfig with simulation parameters
        config: SimConfig with simulation configuration  
        batch_size: Size of each batch (auto-determined if None)
        save_checkpoints: Whether to save intermediate batch results
        checkpoint_dir: Directory for saving checkpoints
    
    Returns:
        Simulation results array
    """
    num_devices = jax.device_count()
    print(f"Number of devices available: {num_devices}")
    if num_devices > 1:
        print("Running simulation with multiple devices (pmap)")
        # Use async processing for multi-device setup
        return async_process_batches(params, config, batch_size, max_concurrent_batches=2)
    else:
        print("Running simulation with single device (sequential minibatches)")
        # Use sequential minibatch processing for single device
        return process_sequential_minibatches(params, config, batch_size, save_checkpoints, checkpoint_dir)


def run_vnc_simulation(cfg: DictConfig) -> jnp.ndarray:
    """Run a complete VNC simulation with the given configuration."""
    params, config = load_vnc_net(cfg)
    results = simulate_vnc_net(params, 
                               config,
                               batch_size=cfg.experiment.batch_size,
                               save_checkpoints=cfg.experiment.save_checkpoints,
                               checkpoint_dir=cfg.paths.ckpt_dir)
    return results
