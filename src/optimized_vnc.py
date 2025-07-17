# Add these modifications to your existing code
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

# Modified NeuronConfig to handle multiple stimulus configurations
class NeuronConfig(NamedTuple):
    W: jnp.ndarray
    W_table: Any
    tau: jnp.ndarray
    a: jnp.ndarray
    threshold: jnp.ndarray
    fr_cap: jnp.ndarray
    input_currents_list: list  # Changed from single input_currents to list
    seeds: jnp.ndarray
    stim_neurons_list: list  # Changed from single stim_neurons to list
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
    stim_input: list
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

# Modified batch processing function to handle multiple stimuli
@partial(jit, static_argnums=(19,))  # Updated static_argnums index
def run_batch_optimized_multi_stim(W, R0, tAxis, T, dt, inputs_batch, pulseStart, pulseEnd,
                                  tau_batch, threshold_batch, a_batch, fr_cap_batch, seed_batch, stdvProp_batch,
                                  excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs, run_func, r_tol=1e-7, a_tol=1e-9):
    """Optimized batch processing with multiple stimulus configurations."""
    
    batch_params = {
        'tau': tau_batch,
        'threshold': threshold_batch, 
        'a': a_batch,
        'fr_cap': fr_cap_batch,
        'seed': seed_batch,
        'stdv_prop': stdvProp_batch,
        'inputs': inputs_batch  # Add inputs to batch params
    }
    
    def single_run(params):
        return run_func(
            W, R0, tAxis, T, dt, params['inputs'], pulseStart, pulseEnd,
            params['tau'], params['threshold'], params['a'], params['fr_cap'],
            params['seed'], params['stdv_prop'],
            excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs, r_tol, a_tol
        )
    
    batch_results = vmap(single_run)(batch_params)
    return batch_results

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
        
        # Calculate numbers
        self.n_stim_configs = len(params.input_currents_list)
        self.n_param_sets = len(params.tau)
        self.total_combinations = self.n_stim_configs * self.n_param_sets
        
        # Pre-compile functions
        self._compile_functions()
    
    def process_with_pipeline(self, batch_size: Optional[int] = None, 
                             save_checkpoints: bool = False,
                             checkpoint_dir: Optional[str] = None) -> dict:
        """
        Process simulation with efficient mixed batching and stimulus tracking.
        """
        
        n_devices = jax.device_count()
        
        if batch_size is None:
            batch_size = calculate_optimal_batch_size(
                self.config.n_neurons, len(self.config.t_axis), n_devices
            )
        
        print(f"Processing {self.total_combinations} total combinations:")
        print(f"  {self.n_stim_configs} stimulus configurations")
        print(f"  {self.n_param_sets} parameter sets per stimulus")
        print(f"  Batch size: {batch_size}")
        print(f"  Devices: {n_devices}")
        
        # Create work queue with stimulus tracking
        work_queue = self._create_work_queue()
        
        # Process with mixed batching
        all_results, all_metadata = self._process_multi_device_mixed_batching(
            work_queue, batch_size, save_checkpoints, checkpoint_dir
        )
        
        # Organize results by stimulus using metadata
        results_by_stim = self._organize_results_by_stimulus(all_results, all_metadata)
        
        return results_by_stim
    
    def _create_work_queue(self) -> list:
        """Create a work queue with stimulus tracking metadata."""
        work_queue = []
        
        # Prepare standard deviation propagation parameters
        if self.config.noise:
            additional_param = self.config.noise_stdv_prop
        else:
            additional_param = 0.0
        
        # Create work items: (stim_idx, param_idx, parameters)
        for stim_idx in range(self.n_stim_configs):
            for param_idx in range(self.n_param_sets):
                work_item = {
                    'param_idx': param_idx,
                    'stim_idx': stim_idx,
                    'tau': self.params.tau[param_idx],
                    'threshold': self.params.threshold[param_idx],
                    'a': self.params.a[param_idx],
                    'fr_cap': self.params.fr_cap[param_idx],
                    'seed': self.params.seeds[param_idx],
                    'stdvProp': additional_param,
                    'inputs': self.params.input_currents_list[stim_idx]
                }
                work_queue.append(work_item)
        
        return work_queue
    
    
    def _process_multi_device_mixed_batching(self, work_queue: list, batch_size: int,
                                            save_checkpoints: bool, checkpoint_dir: Optional[str]) -> tuple:
        """Process work queue with mixed batching across multiple devices."""
        
        n_devices = jax.device_count()
        
        # Create pmap function
        pmap_run = pmap(
            lambda tau_batch, threshold_batch, a_batch, fr_cap_batch, seed_batch, stdvProp_batch, inputs_batch: run_batch_optimized_multi_stim(
                self.W_rw, self.R0, self.config.t_axis, self.config.T, self.config.dt,
                inputs_batch, self.config.pulse_start, self.config.pulse_end,
                tau_batch, threshold_batch, a_batch, fr_cap_batch, seed_batch, stdvProp_batch,
                self.params.exc_dn_idxs, self.params.inh_dn_idxs, self.params.exc_in_idxs,
                self.params.inh_in_idxs, self.params.mn_idxs, self.run_func
            ),
            axis_name='device'
        )
        
        all_results = []
        all_metadata = []
        
        n_batches = (len(work_queue) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(work_queue))
            actual_batch_size = end_idx - start_idx
            
            batch_work = work_queue[start_idx:end_idx]
            
            # Extract batch parameters and metadata
            batch_params, batch_metadata = self._extract_batch_params_and_metadata(batch_work)
            
            # Distribute across devices
            device_data = self._distribute_to_devices_mixed_batch(batch_params, n_devices)
            
            # Run on all devices
            batch_results = pmap_run(**device_data)
            
            # Flatten and trim results
            batch_results_flat = batch_results.reshape(-1, *batch_results.shape[2:])
            batch_results_trimmed = batch_results_flat[:actual_batch_size]
            
            # Clean up NaN/Inf values
            batch_results_trimmed = jax.device_put(batch_results_trimmed, jax.devices('cpu')[0])
            batch_results_trimmed = jnp.where(
                jnp.isinf(batch_results_trimmed) | jnp.isnan(batch_results_trimmed), 
                0, batch_results_trimmed
            )
            
            all_results.extend(batch_results_trimmed)
            all_metadata.extend(batch_metadata)
            
            # Async checkpoint saving
            if save_checkpoints and checkpoint_dir:
                self._save_checkpoint_async(batch_results_trimmed, f"mixed_batch_{batch_idx}", checkpoint_dir)
            
            del batch_results_trimmed, batch_results_flat
            gc.collect()
            print(f"Mixed batch {batch_idx + 1}/{n_batches} completed")
        
        return all_results, all_metadata
    
    def _extract_batch_params_and_metadata(self, batch_work: list) -> tuple:
        """Extract batch parameters and metadata from work items."""
        
        batch_params = {
            'tau_batch': jnp.array([item['tau'] for item in batch_work]),
            'threshold_batch': jnp.array([item['threshold'] for item in batch_work]),
            'a_batch': jnp.array([item['a'] for item in batch_work]),
            'fr_cap_batch': jnp.array([item['fr_cap'] for item in batch_work]),
            'seed_batch': jnp.array([item['seed'] for item in batch_work]),
            'stdvProp_batch': jnp.array([item['stdvProp'] for item in batch_work]),
            'inputs_batch': jnp.array([item['inputs'] for item in batch_work])
        }
        
        batch_metadata = [
            {'param_idx': item['param_idx'], 'stim_idx': item['stim_idx']} 
            for item in batch_work
        ]
        
        return batch_params, batch_metadata
    
    def _distribute_to_devices_mixed_batch(self, batch_params: dict, n_devices: int) -> dict:
        """Distribute mixed batch parameters across devices."""
        
        batch_size = len(batch_params['tau_batch'])
        
        # Pad to make divisible by n_devices
        pad_size = n_devices - (batch_size % n_devices) if batch_size % n_devices != 0 else 0
        
        device_data = {}
        for key, data in batch_params.items():
            if pad_size > 0:
                # Pad with the last element
                if data.ndim == 1:
                    padding = jnp.repeat(data[-1:], pad_size, axis=0)
                else:
                    padding = jnp.repeat(data[-1:], pad_size, axis=0)
                data_padded = jnp.concatenate([data, padding])
            else:
                data_padded = data
            
            # Reshape for devices
            device_data[key] = data_padded.reshape(n_devices, -1, *data_padded.shape[1:])
        
        return device_data
    
    def _organize_results_by_stimulus(self, all_results: list, all_metadata: list) -> dict:
        """Organize results by stimulus configuration using metadata."""
        
        results_by_stim = {}
        
        # Initialize result containers
        for stim_idx in range(self.n_stim_configs):
            results_by_stim[stim_idx] = [None] * self.n_param_sets
        
        # Place results in correct positions using metadata
        for result, metadata in zip(all_results, all_metadata):
            param_idx = metadata['param_idx']
            stim_idx = metadata['stim_idx']
            
            # Validate indices
            if param_idx >= self.n_param_sets or stim_idx >= self.n_stim_configs:
                print(f"Warning: Invalid indices param_idx={param_idx}, stim_idx={stim_idx}")
                continue
            
            results_by_stim[stim_idx][param_idx] = result
        
        # Convert to arrays and validate completeness
        for stim_idx in results_by_stim:
            results_list = results_by_stim[stim_idx]
            
            # Check for missing results
            missing_indices = [i for i, result in enumerate(results_list) if result is None]
            if missing_indices:
                print(f"Warning: Missing results for stimulus {stim_idx} at parameter indices: {missing_indices}")
                # Fill missing with zeros (same shape as other results)
                ref_shape = next(r.shape for r in results_list if r is not None)
                for idx in missing_indices:
                    results_list[idx] = jnp.zeros(ref_shape)
            
            results_by_stim[stim_idx] = jnp.array(results_list)
        
        return results_by_stim
    
    def _save_checkpoint_async(self, results: jnp.ndarray, checkpoint_name: str, checkpoint_dir: str):
        """Save checkpoint asynchronously."""
        def save_checkpoint():
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.npy")
            jnp.save(checkpoint_path, results)
        
        # Run in separate thread to avoid blocking
        threading.Thread(target=save_checkpoint, daemon=True).start()
    
    def _compile_functions(self):
        """Pre-compile all JAX functions for better performance."""
        print("Pre-compiling JAX functions...")
        
        # Create dummy data for compilation
        dummy_size = min(4, self.total_combinations)
        dummy_inputs = jnp.array([self.params.input_currents_list[0]] * dummy_size)
        
        # Compile batch function
        _ = run_batch_optimized_multi_stim(
            self.W_rw, self.R0, self.config.t_axis, self.config.T, self.config.dt,
            dummy_inputs, self.config.pulse_start, self.config.pulse_end,
            self.params.tau[:dummy_size], self.params.threshold[:dummy_size], 
            self.params.a[:dummy_size], self.params.fr_cap[:dummy_size], 
            self.params.seeds[:dummy_size], 
            jnp.array([self.config.noise_stdv_prop] * dummy_size),
            self.params.exc_dn_idxs, self.params.inh_dn_idxs, self.params.exc_in_idxs,
            self.params.inh_in_idxs, self.params.mn_idxs, self.run_func
        )
        
        print("Compilation complete.")

# Modified main simulation function
def simulate_vnc_net_optimized_multi_stim(params: NeuronConfig, config: SimConfig, 
                                         batch_size: Optional[int] = None,
                                         save_checkpoints: bool = False,
                                         checkpoint_dir: Optional[str] = "./checkpoints") -> dict:
    """
    Optimized VNC network simulation with multiple stimulus configurations.
    
    Returns:
        dict: Results organized by stimulus configuration index
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
    
    print(f"\nOptimized multi-stimulus simulation completed:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Stimulus configurations: {len(results)}")
    for stim_idx, stim_results in results.items():
        print(f"    Stimulus {stim_idx}: {stim_results.shape}")
    print(f"  Total combinations: {simulator.total_combinations}")
    print(f"  Average time per combination: {total_time/simulator.total_combinations:.4f} seconds")
    print(f"  Throughput: {simulator.total_combinations/total_time:.2f} combinations/second")
    
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
    if cfg.experiment.dn_screen:
        allExcDNs = W_table.loc[(W_table["class"] == "descending neuron") & (W_table["predictedNt"] == "acetylcholine")]
        stimNeurons = allExcDNs.index.to_list()
        stimNeurons = [[neuron] for neuron in stimNeurons]
    else:
        stimNeurons = cfg.experiment.stimNeurons
    sim_config = create_sim_config(cfg, W)
    
    # Create input currents for each stimulus configuration
    input_currents_list = []
    for stim_neurons, stim_input in zip(stimNeurons, cfg.experiment.stimI):
        stim_neurons_array = jnp.array(stim_neurons, dtype=jnp.int32)
        stim_input = jnp.array(stim_input, dtype=jnp.int32)
        input_currents = jnp.array(make_input(sim_config.n_neurons, stim_neurons_array, stim_input))
        input_currents_list.append(input_currents)
    
    newkey, subkey = jax.random.split(sim_config.rng_key)
    newkey = jax.random.split(newkey, sim_config.num_sims)
    tau, a, threshold, fr_cap = sample_neuron_parameters(cfg, sim_config.n_neurons, sim_config.num_sims, subkey)
    
    if "size" in W_table:
        a, threshold = set_sizes(W_table["size"].values, a, threshold)
    else:
        a, threshold = set_sizes(W_table["surf_area_um2"].values, a, threshold)
    
    exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs = extract_shuffle_indices(W_table)
    
    params = NeuronConfig(
        W=W, W_table=W_table, tau=tau, a=a, threshold=threshold, fr_cap=fr_cap,
        input_currents_list=input_currents_list, seeds=newkey, 
        stim_neurons_list=cfg.experiment.stimNeurons,
        exc_dn_idxs=exc_dn_idxs, inh_dn_idxs=inh_dn_idxs, exc_in_idxs=exc_in_idxs,
        inh_in_idxs=inh_in_idxs, mn_idxs=mn_idxs,
    )
    return params, sim_config

# Usage example function
def run_vnc_simulation_optimized(cfg: DictConfig) -> dict:
    """
    Run an optimized VNC simulation with multiple stimulus configurations.
    
    Args:
        cfg: Configuration
        stim_neurons_list: List of stimulus configurations, e.g., [[31, 132], [1208]]
    
    Returns:
        dict: Results organized by stimulus configuration index
    """
    params, config = load_vnc_net(cfg)
    
    # Use optimized multi-stimulus simulation function
    results = simulate_vnc_net_optimized_multi_stim(
        params, config,
        batch_size=getattr(cfg.experiment, 'batch_size', None),
        save_checkpoints=getattr(cfg.experiment, 'save_checkpoints', False),
        checkpoint_dir=getattr(cfg.paths, 'ckpt_dir', './checkpoints')
    )
    results2 = jnp.stack([results[n] for n in results.keys()])

    return results2

# Example usage:
"""
# Example usage in your main script:

# Define your stimulus configurations
stim_configs = [[31, 132], [1208]]

# Run the simulation
results = run_vnc_simulation_multi_stim(cfg, stim_configs)

# Access results for each stimulus configuration
results_stim_0 = results[0]  # Results for stimulus [31, 132]
results_stim_1 = results[1]  # Results for stimulus [1208]

print(f"Results for stimulus [31, 132]: {results_stim_0.shape}")
print(f"Results for stimulus [1208]: {results_stim_1.shape}")
"""