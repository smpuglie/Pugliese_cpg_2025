import time
from typing import NamedTuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax import vmap, jit, random, pmap
from omegaconf import DictConfig
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import gc
import psutil
import os
from pathlib import Path
from functools import partial

# Import your existing utility functions
from src.shuffle_utils import extract_shuffle_indices, full_shuffle
from src.sim_utils import (
    load_W,
    load_wTable,
    sample_trunc_normal,
    set_sizes,
    make_input,
)

# Keep your existing config classes
class NeuronConfig(NamedTuple):
    W: jnp.ndarray
    W_table: Any
    tau: jnp.ndarray
    a: jnp.ndarray
    threshold: jnp.ndarray
    fr_cap: jnp.ndarray
    input_currents: jnp.ndarray
    seeds: jnp.ndarray
    stim_neurons: jnp.ndarray
    exc_dn_idxs: jnp.ndarray
    inh_dn_idxs: jnp.ndarray
    exc_in_idxs: jnp.ndarray
    inh_in_idxs: jnp.ndarray
    mn_idxs: jnp.ndarray

class SimConfig(NamedTuple):
    n_neurons: int
    num_sims: int
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

# Keep your existing core simulation functions
def rate_equation_half_tanh(t, R, args):
    inputs, pulseStart, pulseEnd, tau, weightedW, threshold, a, frCap = args
    pulse_active = (t >= pulseStart) & (t <= pulseEnd)
    I = inputs * pulse_active
    totalInput = I + jnp.dot(weightedW, R)
    activation = jnp.maximum(
        frCap * jnp.tanh((a / frCap) * (totalInput - threshold)), 0
    )
    return (activation - R) / tau

def run(W, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, r_tol=1e-7, a_tol=1e-9):
    """Pure jax function that runs the actual simulation"""
    term = ODETerm(rate_equation_half_tanh)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.array(tAxis))
    controller = PIDController(rtol=r_tol, atol=a_tol)
    
    odeSolution = diffeqsolve(
        term, solver, 0, T, dt, R0,
        args=(inputs, pulseStart, pulseEnd, tau, W, threshold, a, frCap),
        saveat=saveat, stepsize_controller=controller,
        max_steps=100000, throw=False,
    )
    return jnp.transpose(odeSolution.ys)

def run_shuffle(W, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, 
                seed, extra, excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs, r_tol=1e-7, a_tol=1e-9):
    """Pure jax function that shuffles the weight matrix"""
    W_shuffled = full_shuffle(W, seed, excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs)
    return run(W_shuffled, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, r_tol, a_tol)

def run_noise(W, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, 
              seed, stdvProp, excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs, r_tol=1e-7, a_tol=1e-9):
    """Pure jax function that adds noise to the weight matrix"""
    W_noisy = resample_W(W, seed, stdvProp)
    return run(W_noisy, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, r_tol, a_tol)

def run_no_shuffle(W, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, 
                   seed, stdvProp, excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs, r_tol=1e-7, a_tol=1e-9):
    """Wrapper for run that ignores shuffle and noise parameters."""
    return run(W, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd, tau, threshold, a, frCap, r_tol, a_tol)

def resample_W(W, seed, stdvProp):
    """Add noise to weight matrix."""
    stdvs = W * stdvProp
    key = random.key(seed)
    samples = random.truncated_normal(key, lower=-1.0 / stdvProp, upper=float("inf"), shape=W.shape)
    return W + stdvs * samples

@jit
def reweight_connectivity(W: jnp.ndarray, exc_multiplier: float, inh_multiplier: float) -> jnp.ndarray:
    """Reweight connectivity matrix with excitatory and inhibitory multipliers."""
    Wt_exc = jnp.maximum(W, 0)
    Wt_inh = jnp.minimum(W, 0)
    W_rw = jnp.transpose(exc_multiplier * Wt_exc + inh_multiplier * Wt_inh)
    return W_rw

# OPTIMIZATION 1: Memory-aware batch sizing
def calculate_optimal_batch_size(n_neurons: int, n_timepoints: int, n_devices: int, 
                                safety_factor: float = 0.7) -> int:
    """Calculate optimal batch size based on available memory."""
    # Get available memory
    available_memory = psutil.virtual_memory().available
    
    # Estimate memory per sample (rough calculation)
    # Each sample needs: neuron params + time series data
    bytes_per_sample = n_neurons * n_timepoints * 4  # 4 bytes per float32
    bytes_per_sample *= 3  # Factor for intermediate computations
    
    # Calculate max samples that fit in memory
    max_samples_memory = int((available_memory * safety_factor) / bytes_per_sample)
    
    # Balance with device count
    optimal_batch_size = max(n_devices * 2, min(max_samples_memory, 128))
    
    return optimal_batch_size

# OPTIMIZATION 2: Improved vmapping with better memory layout
@partial(jit, static_argnums=(19,))  # JIT compile with static run_func
def run_batch_optimized(W, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd,
                       tau_batch, threshold_batch, a_batch, frCap_batch, seed_batch, stdvProp_batch,
                       excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs, run_func, r_tol=1e-7, a_tol=1e-9):
    """Optimized batch processing with better memory layout."""
    
    # Use tree_map for better memory handling
    batch_params = {
        'tau': tau_batch,
        'threshold': threshold_batch, 
        'a': a_batch,
        'fr_cap': frCap_batch,
        'seed': seed_batch,
        'stdv_prop': stdvProp_batch
    }
    
    def single_run(params):
        return run_func(
            W, R0, tAxis, T, dt, inputs, pulseStart, pulseEnd,
            params['tau'], params['threshold'], params['a'], params['fr_cap'],
            params['seed'], params['stdv_prop'],
            excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs, r_tol, a_tol
        )
    
    # Use tree_map for efficient vmapping
    batch_results = vmap(single_run)(batch_params)
    return batch_results

# OPTIMIZATION 3: Pipeline processing with overlapped computation and I/O
class OptimizedSimulator:
    def __init__(self, params: NeuronConfig, config: SimConfig):
        self.params = params
        self.config = config
        self.W_rw = reweight_connectivity(
            params.W, config.exc_synapse_multiplier, config.inh_synapse_multiplier
        )
        self.R0 = jnp.zeros((self.W_rw.shape[1],))
        
        # Determine run function
        if config.shuffle:
            self.run_func = run_shuffle
        elif config.noise:
            self.run_func = run_noise
        else:
            self.run_func = run_no_shuffle
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile all JAX functions for better performance."""
        print("Pre-compiling JAX functions...")
        
        # Create dummy data for compilation
        dummy_size = min(4, len(self.params.tau))
        dummy_batch = {
            'tau': self.params.tau[:dummy_size],
            'threshold': self.params.threshold[:dummy_size],
            'a': self.params.a[:dummy_size],
            'fr_cap': self.params.fr_cap[:dummy_size],
            'seed': self.params.seeds[:dummy_size],
            'stdv_prop': jnp.array([self.config.noise_stdv_prop] * dummy_size)
        }
        
        # Compile batch function
        _ = run_batch_optimized(
            self.W_rw, self.R0, self.config.t_axis, self.config.T, self.config.dt,
            self.params.input_currents, self.config.pulse_start, self.config.pulse_end,
            dummy_batch['tau'], dummy_batch['threshold'], dummy_batch['a'], 
            dummy_batch['fr_cap'], dummy_batch['seed'], dummy_batch['stdv_prop'],
            self.params.exc_dn_idxs, self.params.inh_dn_idxs, self.params.exc_in_idxs,
            self.params.inh_in_idxs, self.params.mn_idxs, self.run_func
        )
        
        print("Compilation complete.")
    
    def process_with_pipeline(self, batch_size: Optional[int] = None, 
                             save_checkpoints: bool = False,
                             checkpoint_dir: Optional[str] = None) -> jnp.ndarray:
        """Process simulation with overlapped computation and I/O."""
        
        n_devices = jax.device_count()
        n_samples = len(self.params.tau)
        
        if batch_size is None:
            batch_size = calculate_optimal_batch_size(
                self.config.n_neurons, len(self.config.t_axis), n_devices
            )
        
        print(f"Processing {n_samples} samples with batch size {batch_size} on {n_devices} devices")
        
        # Prepare additional parameters
        if self.config.noise:
            additional_param = jnp.array([self.config.noise_stdv_prop] * n_samples)
        else:
            additional_param = jnp.array([0] * n_samples)
        
        # Create batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_results = []
        
        # Pipeline processing with overlapped I/O
        if n_devices > 1:
            results = self._process_multi_device_pipeline(
                batch_size, n_batches, additional_param, save_checkpoints, checkpoint_dir
            )
        else:
            results = self._process_single_device_pipeline(
                batch_size, n_batches, additional_param, save_checkpoints, checkpoint_dir
            )
        
        return jnp.array(results)
    
    def _process_multi_device_pipeline(self, batch_size: int, n_batches: int, 
                                     additional_param: jnp.ndarray,
                                     save_checkpoints: bool, checkpoint_dir: Optional[str]) -> list:
        """Multi-device pipeline processing with pmap."""
        
        n_devices = jax.device_count()
        
        # Create pmap function
        pmap_run = pmap(
            lambda tau_batch, threshold_batch, a_batch, frCap_batch, seed_batch, stdvProp_batch: run_batch_optimized(
                self.W_rw, self.R0, self.config.t_axis, self.config.T, self.config.dt,
                self.params.input_currents, self.config.pulse_start, self.config.pulse_end,
                tau_batch, threshold_batch, a_batch, frCap_batch, seed_batch, stdvProp_batch,
                excDnIdxs=self.params.exc_dn_idxs, inhDnIdxs=self.params.inh_dn_idxs,
                excInIdxs=self.params.exc_in_idxs, inhInIdxs=self.params.inh_in_idxs, mnIdxs=self.params.mn_idxs,
                run_func=self.run_func
            ),
            axis_name='device'
        )
        
        all_results = []
        
        # Process batches with device parallelization
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.params.tau))
            
            # Extract batch data
            batch_data = self._extract_batch_data(start_idx, end_idx, additional_param)
            
            # Distribute across devices
            device_data = self._distribute_to_devices(batch_data, n_devices)
            
            # Run on all devices
            batch_results = pmap_run(**device_data)
            
            # Flatten results
            batch_results_flat = batch_results.reshape(-1, *batch_results.shape[2:])
            actual_batch_size = end_idx - start_idx
            batch_results_trimmed = batch_results_flat[:actual_batch_size]

            all_results.extend(jax.device_put(batch_results_trimmed,jax.devices('cpu')[0]))

            # Async checkpoint saving
            if save_checkpoints and checkpoint_dir:
                self._save_checkpoint_async(batch_results_trimmed, batch_idx, checkpoint_dir)
            
            del batch_results_trimmed, batch_results_flat
            gc.collect()  # Force garbage collection
            print(f"Batch {batch_idx + 1}/{n_batches} completed")
        
        return all_results
    
    def _process_single_device_pipeline(self, batch_size: int, n_batches: int,
                                      additional_param: jnp.ndarray,
                                      save_checkpoints: bool, checkpoint_dir: Optional[str]) -> list:
        """Single device pipeline processing."""
        
        all_results = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.params.tau))
            
            # Extract batch data
            batch_data = self._extract_batch_data(start_idx, end_idx, additional_param)
            
            # Run batch
            batch_results = run_batch_optimized(
                self.W_rw, self.R0, self.config.t_axis, self.config.T, self.config.dt,
                self.params.input_currents, self.config.pulse_start, self.config.pulse_end,
                **batch_data, excDnIdxs=self.params.exc_dn_idxs, inhDnIdxs=self.params.inh_dn_idxs,
                excInIdxs=self.params.exc_in_idxs, inhInIdxs=self.params.inh_in_idxs, mnIdxs=self.params.mn_idxs,
                run_func=self.run_func
            )
            
            all_results.extend(jax.device_put(batch_results,jax.devices('cpu')[0]))
            # Async checkpoint saving
            if save_checkpoints and checkpoint_dir:
                self._save_checkpoint_async(batch_results, batch_idx, checkpoint_dir)
            
            del batch_results
            gc.collect()  # Force garbage collection
            print(f"Batch {batch_idx + 1}/{n_batches} completed")
        
        return all_results
    
    def _extract_batch_data(self, start_idx: int, end_idx: int, additional_param: jnp.ndarray) -> dict:
        """Extract batch data efficiently."""
        return {
            'tau_batch': self.params.tau[start_idx:end_idx],
            'threshold_batch': self.params.threshold[start_idx:end_idx],
            'a_batch': self.params.a[start_idx:end_idx],
            'frCap_batch': self.params.fr_cap[start_idx:end_idx],
            'seed_batch': self.params.seeds[start_idx:end_idx],
            'stdvProp_batch': additional_param[start_idx:end_idx]
        }
    
    def _distribute_to_devices(self, batch_data: dict, n_devices: int) -> dict:
        """Distribute batch data across devices for pmap."""
        batch_size = len(batch_data['tau_batch'])

        # Pad to make divisible by n_devices
        pad_size = n_devices - (batch_size % n_devices) if batch_size % n_devices != 0 else 0
        
        device_data = {}
        for key, data in batch_data.items():
            if pad_size > 0:
                # Pad with the last element
                padding = jnp.repeat(data[-1:], pad_size, axis=0)
                data_padded = jnp.concatenate([data, padding])
            else:
                data_padded = data
            
            # Reshape for devices
            device_data[key] = data_padded.reshape(n_devices, -1, *data_padded.shape[1:])
        
        return device_data
    
    def _save_checkpoint_async(self, results: jnp.ndarray, batch_idx: int, checkpoint_dir: str):
        """Save checkpoint asynchronously."""
        def save_fn():
            checkpoint_path = Path(checkpoint_dir) / f"batch_{batch_idx:04d}.npy"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            jnp.save(checkpoint_path, results)
        
        # Run in separate thread
        threading.Thread(target=save_fn, daemon=True).start()

# OPTIMIZATION 4: Main simulation function with automatic optimization selection
def simulate_vnc_net_optimized(params: NeuronConfig, config: SimConfig, 
                              batch_size: Optional[int] = None,
                              save_checkpoints: bool = False,
                              checkpoint_dir: Optional[str] = "./checkpoints") -> jnp.ndarray:
    """
    Optimized VNC network simulation with automatic performance tuning.
    
    Key optimizations:
    1. Memory-aware batch sizing
    2. Pre-compiled JAX functions
    3. Improved vmapping with better memory layout
    4. Pipeline processing with overlapped I/O
    5. Efficient device utilization
    """
    
    start_time = time.time()
    
    # Initialize optimized simulator
    simulator = OptimizedSimulator(params, config)
    
    # Run simulation with pipeline processing
    results = simulator.process_with_pipeline(
        batch_size=batch_size,
        save_checkpoints=save_checkpoints,
        checkpoint_dir=checkpoint_dir
    )
    
    total_time = time.time() - start_time
    n_samples = len(params.tau)
    
    print(f"\nOptimized simulation completed:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Final result shape: {results.shape}")
    print(f"  Average time per sample: {total_time/n_samples:.4f} seconds")
    print(f"  Throughput: {n_samples/total_time:.2f} samples/second")
    
    return results

# Keep your existing configuration loading functions
def validate_config(cfg: DictConfig) -> None:
    """Check if config has everything."""
    required_sections = {
        "experiment": ["name", "wPath", "dfPath", "stimNeurons", "stimI", "seed"],
        "sim": ["T", "dt", "pulseStart", "pulseEnd"],
        "neuron_params": [
            "tauMean", "tauStdv", "aMean", "aStdv", "thresholdMean", "thresholdStdv",
            "frcapMean", "frcapStdv", "excitatoryMultiplier", "inhibitoryMultiplier",
        ],
    }
    
    for section, keys in required_sections.items():
        if not hasattr(cfg, section):
            raise ValueError(f"Missing required config section: {section}")
        section_cfg = getattr(cfg, section)
        for key in keys:
            if not hasattr(section_cfg, key):
                raise ValueError(f"Missing required config key '{key}' in section '{section}'")

def load_connectivity(cfg: DictConfig) -> tuple[jnp.ndarray, Any]:
    """Load connectivity matrix and the data."""
    w_path = cfg.experiment.wPath
    df_path = cfg.experiment.dfPath
    W = load_W(w_path)
    W_table = load_wTable(df_path)
    return jnp.array(W), W_table

def sample_neuron_parameters(cfg: DictConfig, n_neurons: int, num_sims: int, seeds: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample neuron parameters based on configuration."""
    np_cfg = cfg.neuron_params
    
    keys = jax.random.split(seeds, 4)
    tau = sample_trunc_normal(
        key=keys[0], mean=float(np_cfg.tauMean), stdev=float(np_cfg.tauStdv), shape=(num_sims, n_neurons)
    )
    a = sample_trunc_normal(
        key=keys[1], mean=float(np_cfg.aMean), stdev=float(np_cfg.aStdv), shape=(num_sims, n_neurons)
    )
    threshold = sample_trunc_normal(
        key=keys[2], mean=float(np_cfg.thresholdMean), stdev=float(np_cfg.thresholdStdv), shape=(num_sims, n_neurons)
    )
    fr_cap = sample_trunc_normal(
        key=keys[3], mean=float(np_cfg.frcapMean), stdev=float(np_cfg.frcapStdv), shape=(num_sims, n_neurons)
    )
    
    return jnp.array(tau), jnp.array(a), jnp.array(threshold), jnp.array(fr_cap)

def create_sim_config(cfg: DictConfig, W: jnp.ndarray) -> SimConfig:
    """Create simulation configuration from config and connectivity matrix."""
    n_neurons = W.shape[0]
    num_sims = cfg.experiment.n_replicates
    rng_key = jax.random.PRNGKey(cfg.experiment.seed)
    
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
        n_neurons=n_neurons, num_sims=num_sims, T=T, dt=dt, pulse_start=pulse_start, pulse_end=pulse_end,
        t_axis=t_axis, stim_input=stim_input, shuffle=shuffle, noise=noise, noise_stdv_prop=noise_stdv_prop,
        exc_synapse_multiplier=exc_synapse_multiplier, inh_synapse_multiplier=inh_synapse_multiplier,
        r_tol=r_tol, a_tol=a_tol, rng_key=rng_key,
    )

def load_vnc_net(cfg: DictConfig) -> tuple[NeuronConfig, SimConfig]:
    """Load and configure VNC network parameters and simulation settings."""
    validate_config(cfg)
    
    W, W_table = load_connectivity(cfg)
    sim_config = create_sim_config(cfg, W)
    
    stim_neurons = jnp.array(cfg.experiment.stimNeurons, dtype=jnp.int32)
    newkey, subkey = jax.random.split(sim_config.rng_key)
    newkey = jax.random.split(newkey, sim_config.num_sims)
    tau, a, threshold, fr_cap = sample_neuron_parameters(cfg, sim_config.n_neurons, sim_config.num_sims, subkey)
    
    if "size" in W_table:
        a, threshold = set_sizes(W_table["size"].values, a, threshold)
    else:
        a, threshold = set_sizes(W_table["surf_area_um2"].values, a, threshold)
    
    input_currents = jnp.array(make_input(sim_config.n_neurons, stim_neurons, sim_config.stim_input))
    exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs = extract_shuffle_indices(W_table)
    
    params = NeuronConfig(
        W=W, W_table=W_table, tau=tau, a=a, threshold=threshold, fr_cap=fr_cap,
        input_currents=input_currents, seeds=newkey, stim_neurons=stim_neurons,
        exc_dn_idxs=exc_dn_idxs, inh_dn_idxs=inh_dn_idxs, exc_in_idxs=exc_in_idxs,
        inh_in_idxs=inh_in_idxs, mn_idxs=mn_idxs,
    )
    return params, sim_config

def run_vnc_simulation_optimized(cfg: DictConfig) -> jnp.ndarray:
    """Run an optimized VNC simulation with the given configuration."""
    params, config = load_vnc_net(cfg)
    
    # Use optimized simulation function
    results = simulate_vnc_net_optimized(
        params, config,
        batch_size=getattr(cfg.experiment, 'batch_size', None),
        save_checkpoints=getattr(cfg.experiment, 'save_checkpoints', False),
        checkpoint_dir=getattr(cfg.paths, 'ckpt_dir', './checkpoints')
    )
    
    return results