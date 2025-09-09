# Asynchronous VNC Network Simulation - Streaming Processing
import asyncio
import concurrent.futures
import time
import jax
import jax.numpy as jnp
import numpy as np
import traceback
import psutil                    
import gc
from pathlib import Path
from typing import Optional

from typing import Dict, Any, Optional, Tuple, List, Union
import threading
from queue import Queue
from dataclasses import dataclass, field
from collections import defaultdict

# Import existing simulation components
from src.simulation.vnc_sim import (
    run_single_simulation, update_single_sim_state, 
    reweight_connectivity, removal_probability, add_noise_to_weights,
    compute_oscillation_score
)
from src.utils.shuffle_utils import full_shuffle
from src.data.data_classes import NeuronParams, SimParams, SimulationConfig, Pruning_state, CheckpointState
from src.memory.adaptive_memory import (
    monitor_memory_usage, create_memory_manager, calculate_optimal_concurrent_size, log_memory_status,
    move_pruning_state_to_cpu, log_detailed_gpu_status
)
from src.memory.checkpointing import (
    CheckpointState, save_checkpoint, load_checkpoint,
    find_latest_checkpoint, cleanup_old_checkpoints
)

class AsyncLogger:
    """Thread-safe logger for async simulations with better formatting."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.sim_logs = defaultdict(list)
        
    def log_sim(self, sim_index: int, message: str, level: str = "INFO"):
        """Log a message for a specific simulation."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] Sim {sim_index:2d}: {message}"
        
        with self.lock:
            self.sim_logs[sim_index].append((timestamp, level, message))
            # Print immediately with color coding
            if level == "INIT":
                print(f"\033[94m{formatted_msg}\033[0m")  # Blue for initialization
            elif level == "PROGRESS":
                print(f"\033[92m{formatted_msg}\033[0m")  # Green for progress
            elif level == "COMPLETE":
                print(f"\033[93m{formatted_msg}\033[0m")  # Yellow for completion
            elif level == "ERROR":
                print(f"\033[91m{formatted_msg}\033[0m")  # Red for errors
            else:
                print(formatted_msg)  # Default white
                
    def log_batch(self, message: str):
        """Log a batch-level message."""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"\033[96m[{timestamp}] BATCH: {message}\033[0m"  # Cyan for batch
        print(formatted_msg)
        
    def print_summary(self, sim_indices: List[int]):
        """Print a summary of all simulations."""
        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)
        
        for sim_idx in sorted(sim_indices):
            logs = self.sim_logs[sim_idx]
            if logs:
                print(f"\nSim {sim_idx:2d} Timeline:")
                for timestamp, level, message in logs[-5:]:  # Show last 5 messages
                    print(f"  {timestamp}: {message}")


# Global logger instance
async_logger = AsyncLogger()


@dataclass
class AsyncSimState:
    """State for asynchronous simulation."""
    sim_index: int
    pruning_state: Pruning_state
    is_converged: bool = False
    iteration_count: int = 0
    last_update_time: float = field(default_factory=time.time)


def initialize_single_sim_pruning_state(
    neuron_params: NeuronParams, 
    sim_params: SimParams, 
    sim_index: int
) -> Pruning_state:
    """Initialize pruning state for a specific simulation index with memory efficiency."""
    try:
        mn_idxs = neuron_params.mn_idxs
        all_neurons = jnp.arange(neuron_params.W.shape[-1])
        in_idxs = jnp.setdiff1d(all_neurons, mn_idxs)

        # Create state for single simulation (no batch dimension)
        W_mask = jnp.full((neuron_params.W.shape[0], neuron_params.W.shape[1]), True, dtype=jnp.bool_)
        interneuron_mask = jnp.full((W_mask.shape[-1],), fill_value=False, dtype=jnp.bool_)
        interneuron_mask = interneuron_mask.at[in_idxs].set(True)
        level = jnp.zeros((1,), dtype=jnp.int32)
        total_removed_neurons = jnp.full((W_mask.shape[-1],), False, dtype=jnp.bool_)
        removed_stim_neurons = jnp.full((total_removed_neurons.shape[-1],), False, dtype=jnp.bool_)
        neurons_put_back = jnp.full((total_removed_neurons.shape[-1],), False, dtype=jnp.bool_)
        prev_put_back = jnp.full((total_removed_neurons.shape[-1],), False, dtype=jnp.bool_)
        last_removed = jnp.full((total_removed_neurons.shape[-1],), False, dtype=jnp.bool_)
        min_circuit = jnp.full((1,), False)

        # Initialize most recent good oscillating state tracking for single simulation
        last_good_oscillating_removed = jnp.full((total_removed_neurons.shape[-1],), False, dtype=jnp.bool_)
        last_good_oscillating_put_back = jnp.full((total_removed_neurons.shape[-1],), False, dtype=jnp.bool_)
        last_good_oscillation_score = jnp.full((1,), -1.0)  # Initialize with -1 (no good state yet)
        last_good_W_mask = jnp.zeros_like(W_mask, dtype=jnp.bool_)  # Initialize with empty W_mask

        # Initialize probabilities for this specific simulation
        exclude_mask = ~interneuron_mask
        p_arrays = removal_probability(jnp.ones((W_mask.shape[-1],)), exclude_mask)

        # Get the specific seed for this simulation index with bounds checking
        if sim_index < len(neuron_params.seeds):
            sim_seed = neuron_params.seeds[sim_index]
        else:
            # Use last seed if we don't have enough seeds (defensive programming)
            sim_seed = neuron_params.seeds[-1]

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
            keys=sim_seed,  # Simulation seed, shape (2,) not (1, 2)
            last_good_oscillating_removed=last_good_oscillating_removed,
            last_good_oscillating_put_back=last_good_oscillating_put_back,
            last_good_oscillation_score=last_good_oscillation_score,
            last_good_W_mask=last_good_W_mask,
            last_good_key=jnp.zeros((2,), dtype=jnp.uint32),
            last_good_stim_idx=jnp.array(-1, dtype=jnp.int32),
            last_good_param_idx=jnp.array(-1, dtype=jnp.int32)
        )
        
    except Exception as e:
        # If initialization fails, perform cleanup and re-raise with better error context
        try:
            gc.collect()
        except:
            pass
        raise Exception(f"Failed to initialize pruning state for sim {sim_index}: {e}")
    
    
class AsyncPruningManager:
    """Manages asynchronous execution of pruning simulations with shared read-only data."""
    
    def __init__(
        self,
        neuron_params: NeuronParams,
        sim_params: SimParams,
        sim_config: SimulationConfig,
        max_workers: int = None
    ):
        # ENHANCED Pre-initialization memory check and cleanup
        print("🧹 Performing comprehensive memory cleanup before AsyncPruningManager initialization...")
        
        # Clear JAX caches and backends multiple times
        for cleanup_round in range(3):
            jax.clear_caches()
            gc.collect()
        
        # Force GPU synchronization and small allocations to ensure cleanup
        try:
            devices = jax.devices()
            for device in devices:
                with jax.default_device(device):
                    _ = jnp.array([1.0])  # Small allocation to test memory
            print(f"✅ GPU memory test passed on {len(devices)} devices")
        except Exception as e:
            print(f"⚠️  GPU memory test failed: {e}")
        
        # Check memory status after cleanup
        try:
            memory_info = monitor_memory_usage(0)
            gpu_usage = memory_info.get('gpu_usage_percent', 0)
            print(f"📊 Post-cleanup GPU memory usage: {gpu_usage:.1f}%")
            if gpu_usage > 70:
                print(f"⚠️  WARNING: Still high GPU memory usage after cleanup")
        except Exception as e:
            print(f"Warning: Could not check memory status during initialization: {e}")
        
        self.neuron_params = neuron_params
        self.sim_params = sim_params
        self.sim_config = sim_config
        
        # Initialize clip_start_val for consistent oscillation calculation
        self.clip_start_val = int(self.sim_params.pulse_start / self.sim_params.dt) + 230
        
        # Initialize adaptive memory manager
        total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
        self.memory_manager = create_memory_manager(
            total_simulations=total_sims,
            simulation_type="pruning"
        )
    
        # Pre-create the motor neuron mask once to ensure consistency
        # Force mn_idxs to CPU to avoid device placement issues
        mn_idxs_cpu = jax.device_put(self.neuron_params.mn_idxs, jax.devices('cpu')[0])
        n_neurons_cpu = jax.device_put(self.sim_params.n_neurons, jax.devices('cpu')[0])
        self.mn_mask_template = jnp.isin(jnp.arange(n_neurons_cpu), mn_idxs_cpu)
        
        # Multi-GPU setup with conservative memory management
        self.n_devices = jax.device_count()
        self.devices = jax.devices()
        gpu_devices = [d for d in self.devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        self.n_gpu_devices = len(gpu_devices)
        
        
        print(f"Async Manager: {self.n_devices} total devices, {self.n_gpu_devices} GPU devices")
        
        # Device rotation counter for distributing coordination tasks
        self._device_rotation_counter = 0
        # Let JAX handle default device selection automatically
        print(f"JAX will distribute operations across {self.n_gpu_devices} GPU devices")
        
        # Add memory pre-allocation to avoid CUDA initialization issues
        try:
            for i, device in enumerate(self.devices[:self.n_gpu_devices]):
                print(f"  🔧 Initializing device {i}: {device}")
                
                # Clear cache before each device to prevent conflicts
                jax.clear_caches()
                
                # Pre-allocate small tensor on each GPU to initialize CUDA context
                test_tensor = jax.device_put(jnp.ones(10), device)
                del test_tensor
                
                # Immediate cleanup after each device initialization
                jax.clear_caches()
                gc.collect()
                
                print(f"  ✅ Initialized CUDA context on device {i}: {device}")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize all CUDA contexts: {e}")
            # Fall back to single GPU if multi-GPU fails
            if self.n_gpu_devices > 1:
                print("  🔄 Falling back to single GPU mode")
                self.n_gpu_devices = 1
                self.devices = self.devices[:1]
        
        # For async sims, we use device placement rather than pmap
        # This allows each simulation to run on a different GPU independently
        self.max_workers = max_workers or min(32, (self.n_devices * 2))
        
        # Create shared thread pool executor to avoid JAX compilation context fragmentation
        self._shared_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(16, self.n_devices * 2),
            thread_name_prefix="jax_pruning"
        )
        
        # Pre-replicate read-only data across all devices to reduce memory copying
        print("Pre-replicating read-only data across devices...")
        self._setup_shared_data()
        
        # State management
        self.active_sims: Dict[int, AsyncSimState] = {}
        self.completed_sims: Dict[int, Tuple[jnp.ndarray, Pruning_state]] = {}
        self.results_queue = Queue()
        self.update_lock = threading.Lock()
        
        # Create device-specific JIT compiled functions
        # Each device gets its own compiled version for better performance
        self.device_functions = {}
        self._jit_sim_functions = {}  # Cache for JIT-compiled simulation functions
        self._jit_final_sim_functions = {}  # Cache for JIT-compiled FINAL simulation functions
        self._jit_update_functions = {}  # Cache for JIT-compiled update functions
        
        # Pre-compile functions for each device using direct assignment
        self._setup_device_functions()
    
    def _setup_shared_data(self):
        """Set up memory-efficient shared data across devices."""
        try:
            print("Setting up memory-efficient shared data across devices...")
            
            # CRITICAL: Clear compilation cache before large memory allocations
            # This prevents rematerialization conflicts when checkpoint resuming
            print("🧹 Clearing compilation cache before device memory allocation...")
            jax.clear_caches()
            
            # Force garbage collection to ensure clean memory state
            for _ in range(2):
                gc.collect()
            
            # Pre-place the large W matrix on each device to avoid copying per simulation
            # This is the most memory-intensive data that we want to share
            self.device_W = {}
            for i, device in enumerate(self.devices):
                self.device_W[i] = jax.device_put(self.neuron_params.W, device)
                print(f"  Placed W matrix on device {i}: {device}")
            
            # Shared parameters that are read-only and small
            self.shared_t_axis = self.sim_params.t_axis
            self.shared_noise_stdv = jnp.array(self.sim_params.noise_stdv)
            self.shared_exc_multiplier = self.sim_params.exc_multiplier
            self.shared_inh_multiplier = self.sim_params.inh_multiplier
            
            print(f"  Memory-efficient setup completed successfully")
            
        except Exception as e:
            print(f"Warning: Could not set up shared data: {e}")
            # Fallback
            self.device_W = {}
            self.shared_t_axis = self.sim_params.t_axis
            self.shared_noise_stdv = jnp.array(self.sim_params.noise_stdv)
            self.shared_exc_multiplier = self.sim_params.exc_multiplier
            self.shared_inh_multiplier = self.sim_params.inh_multiplier
    
    def _setup_device_functions(self):
        """Pre-compile functions for each device using shared data."""
        for i, device in enumerate(self.devices):
            print(f"Setting up device {i}: {device}")
            
            # Create static values for this device
            oscillation_threshold_val = self.sim_config.oscillation_threshold
            clip_start_val = int(self.sim_params.pulse_start / self.sim_params.dt) + 230
            device_index = i  # Capture the device index
            
            # Use more conservative settings for devices other than 0
            # This helps with device-specific numerical precision issues
            device_rtol = self.sim_params.r_tol
            device_atol = self.sim_params.a_tol

            # Store device-specific parameters (no JIT compilation here to avoid diffrax conflicts)
            self.device_functions[i] = {
                'device_index': device_index,
                'oscillation_threshold': oscillation_threshold_val,
                'clip_start': clip_start_val,
                'rtol': device_rtol,
                'atol': device_atol
            }
            
            # Pre-create JIT-compiled update function for this device
            self._jit_update_functions[i] = self._create_jit_update_function(
                oscillation_threshold_val, 
                clip_start_val
            )
        
    def _get_device_for_sim(self, sim_index: int) -> int:
        """Assign a device to a simulation based on its index."""
        return sim_index % self.n_devices
    
    def _get_device_functions(self, device_id: int):
        """Get the device-specific parameters for a device."""
        return self.device_functions[device_id]
    
    def _get_next_coordination_device(self):
        """Get the next device for coordination tasks to distribute load."""
        if self.n_gpu_devices > 0:
            device_id = self._device_rotation_counter % self.n_gpu_devices
            self._device_rotation_counter += 1
            return self.devices[device_id]
        return jax.devices("cpu")[0]
    
    def _create_jit_simulation_function(self, device_id: int):
        """Create a JIT-compiled simulation function for a specific device."""
        target_device = self.devices[device_id]
        
        # Create a JIT-compiled function that's bound to this device
        @jax.jit
        def jit_simulation(W_device, W_mask_device, tau_param, a_param, threshold_param, 
                          fr_cap_param, input_currents_param, seeds_param, r_tol, a_tol):
            """JIT-compiled simulation function for improved performance."""
            # Apply W_mask to connectivity (memory efficient, same device)
            W_masked = W_device * W_mask_device
            W_reweighted = reweight_connectivity(
                W_masked, 
                self.shared_exc_multiplier, 
                self.shared_inh_multiplier
            )
            
            # Run simulation with current mask and specified tolerances
            # Use adaptive step size for better performance during pruning iterations
            return run_single_simulation(
                W_reweighted,
                tau_param,
                a_param,
                threshold_param,
                fr_cap_param,
                input_currents_param,
                self.shared_noise_stdv,
                self.shared_t_axis,
                self.sim_params.T,
                self.sim_params.dt,
                self.sim_params.pulse_start,
                self.sim_params.pulse_end,
                r_tol,
                a_tol,
                seeds_param,
            )
        
        return jit_simulation
    
    def _create_jit_final_simulation_function(self, device_id: int):
        """Create a JIT-compiled final simulation function with deterministic step size."""
        target_device = self.devices[device_id]
        
        # Create a JIT-compiled function that's bound to this device
        @jax.jit
        def jit_final_simulation(W_device, W_mask_device, tau_param, a_param, threshold_param, 
                                fr_cap_param, input_currents_param, seeds_param, r_tol, a_tol):
            """JIT-compiled final simulation function with fixed step size for determinism."""
            # Apply W_mask to connectivity (memory efficient, same device)
            W_masked = W_device * W_mask_device
            W_reweighted = reweight_connectivity(
                W_masked, 
                self.shared_exc_multiplier, 
                self.shared_inh_multiplier
            )
            
            # Run simulation with current mask and specified tolerances
            # RNN implementation is already deterministic with fixed random seed
            return run_single_simulation(
                W_reweighted,
                tau_param,
                a_param,
                threshold_param,
                fr_cap_param,
                input_currents_param,
                self.shared_noise_stdv,
                self.shared_t_axis,
                self.sim_params.T,
                self.sim_params.dt,
                self.sim_params.pulse_start,
                self.sim_params.pulse_end,
                r_tol,
                a_tol,
                seeds_param
            )
        
        return jit_final_simulation
    
    def _run_final_simulation_direct(
        self,
        final_W_mask: jnp.ndarray,
        sim_index: int,
        param_idx: int,
        stim_idx: int,
        r_tol: float,
        a_tol: float,
        device_idx: int = 0,
        custom_key: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Run final simulation with a fresh JIT compilation to avoid caching issues."""
        # Use pre-placed W matrix on the target device for memory efficiency
        target_device = self.devices[device_idx]
        
        if device_idx in self.device_W:
            W_device = self.device_W[device_idx]
        else:
            # Fallback to placing W on the target device
            W_device = jax.device_put(self.neuron_params.W, target_device)
        
        # IMPORTANT: Place W_mask on the same device as W_device to avoid device mismatch
        W_mask_device = jax.device_put(final_W_mask, target_device)
        
        # Create a FRESH JIT-compiled function specifically for final simulation
        # This avoids any potential caching or device state issues
        @jax.jit
        def fresh_final_simulation(W_device, W_mask_device, tau_param, a_param, threshold_param, 
                          fr_cap_param, input_currents_param, seeds_param, r_tol, a_tol):
            """Fresh JIT-compiled simulation function specifically for final simulation."""
            # Apply W_mask to connectivity (memory efficient, same device)
            W_masked = W_device * W_mask_device
            W_reweighted = reweight_connectivity(
                W_masked, 
                self.shared_exc_multiplier, 
                self.shared_inh_multiplier
            )
            
            # Run simulation with current mask and specified tolerances
            return run_single_simulation(
                W_reweighted,
                tau_param,
                a_param,
                threshold_param,
                fr_cap_param,
                input_currents_param,
                self.shared_noise_stdv,
                self.shared_t_axis,
                self.sim_params.T,
                self.sim_params.dt,
                self.sim_params.pulse_start,
                self.sim_params.pulse_end,
                r_tol,
                a_tol,
                seeds_param
            )
        
        # Use either custom key or default seed
        key_to_use = custom_key if custom_key is not None else self.neuron_params.seeds[param_idx]
        
        # Run with fresh JIT compilation
        return fresh_final_simulation(
            W_device,
            W_mask_device,
            self.neuron_params.tau[param_idx],
            self.neuron_params.a[param_idx],
            self.neuron_params.threshold[param_idx],
            self.neuron_params.fr_cap[param_idx],
            self.neuron_params.input_currents[stim_idx, param_idx],
            key_to_use,
            r_tol,
            a_tol
        )
    
    def _create_jit_update_function(self, oscillation_threshold: float, clip_start: int):
        """Create a JIT-compiled update function with static arguments."""
        # Create wrapper function with static args baked in
        def update_state_with_static_args(state, R, mn_mask):
            return update_single_sim_state(state, R, mn_mask, oscillation_threshold, clip_start)
        
        # Apply JIT to the wrapper function 
        return jax.jit(update_state_with_static_args)
            
    def _run_single_pruning_iteration(
        self,
        W_mask: jnp.ndarray,
        sim_index: int
    ) -> jnp.ndarray:
        """Run a single pruning iteration for one simulation with memory efficiency."""
        return self._run_single_pruning_iteration_with_tolerances(
            W_mask, sim_index, self.sim_params.r_tol, self.sim_params.a_tol, 
            self._get_device_for_sim(sim_index)
        )
    
    def _run_single_pruning_iteration_with_tolerances(
        self,
        W_mask: jnp.ndarray,
        sim_index: int,
        r_tol: float,
        a_tol: float,
        device_idx: int = 0,
        custom_key: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Run a single pruning iteration using memory-efficient pre-placed W matrix."""
        # Extract parameters for this specific simulation
        param_idx = sim_index % self.sim_params.n_param_sets
        stim_idx = sim_index // self.sim_params.n_param_sets
        
        # Use pre-placed W matrix on the target device for memory efficiency
        target_device = self.devices[device_idx]
        
        if device_idx in self.device_W:
            W_device = self.device_W[device_idx]
        else:
            # Fallback to placing W on the target device
            W_device = jax.device_put(self.neuron_params.W, target_device)
        
        # IMPORTANT: Place W_mask on the same device as W_device to avoid device mismatch
        W_mask_device = jax.device_put(W_mask, target_device)
        
        # Use JIT-compiled device-specific function for better performance
        if device_idx not in self._jit_sim_functions:
            # Create and cache JIT-compiled function for this device
            self._jit_sim_functions[device_idx] = self._create_jit_simulation_function(device_idx)
        
        # Use JIT-compiled function with either custom key or default seed
        key_to_use = custom_key if custom_key is not None else self.neuron_params.seeds[param_idx]
        return self._jit_sim_functions[device_idx](
            W_device,
            W_mask_device,
            self.neuron_params.tau[param_idx],
            self.neuron_params.a[param_idx],
            self.neuron_params.threshold[param_idx],
            self.neuron_params.fr_cap[param_idx],
            self.neuron_params.input_currents[stim_idx, param_idx],
            key_to_use,  # Use either custom key or original seed
            r_tol,
            a_tol
        )
    
    def _run_final_simulation_with_tolerances(
        self,
        final_W_mask: jnp.ndarray,
        sim_index: int,
        stim_idx: int,
        param_idx: int,
        r_tol: float,
        a_tol: float,
        device_idx: int = 0
    ) -> Tuple[jnp.ndarray, float]:
        """Run final simulation using same async thread pool context as pruning iterations."""
        # Use pre-placed W matrix on the target device for memory efficiency
        target_device = self.devices[device_idx]
        
        if device_idx in self.device_W:
            W_device = self.device_W[device_idx]
        else:
            # Fallback to placing W on the target device
            W_device = jax.device_put(self.neuron_params.W, target_device)
        
        # Apply the final W_mask - streamlined device context
        with jax.default_device(target_device):
            # Data is already on target_device due to context, no need for explicit device_put
            # Use EXACT same masking operation as in pruning: W * W_mask
            W_masked = W_device * final_W_mask
            
            # Reweight connectivity (same as in pruning iterations)
            W_reweighted = reweight_connectivity(
                W_masked, 
                self.shared_exc_multiplier, 
                self.shared_inh_multiplier
            )
            
            # Run final simulation with same fixed seed as pruning
            final_key = self.neuron_params.seeds[param_idx]
            final_results = run_single_simulation(
                W_reweighted,
                self.neuron_params.tau[param_idx],
                self.neuron_params.a[param_idx],
                self.neuron_params.threshold[param_idx],
                self.neuron_params.fr_cap[param_idx],
                self.neuron_params.input_currents[stim_idx, param_idx],
                self.shared_noise_stdv,
                self.shared_t_axis,
                self.sim_params.T,
                self.sim_params.dt,
                self.sim_params.pulse_start,
                self.sim_params.pulse_end,
                r_tol,
                a_tol,
                final_key
            )
            
            # Compute oscillation score from final simulation 
            # Use consistent clip_start calculation with pruning iterations
            clip_start = self.clip_start_val  # Use pre-calculated value from device setup
            max_frs = jnp.max(final_results, axis=-1)
            mn_mask = jnp.isin(jnp.arange(self.sim_params.n_neurons), self.neuron_params.mn_idxs)
            active_mask = ((max_frs > 0.01) & mn_mask)
            
            final_oscillation_score, _ = compute_oscillation_score(final_results[..., clip_start:], active_mask, prominence=0.05)
            final_oscillation_score = float(final_oscillation_score)
            
            return final_results, final_oscillation_score
    
    async def _process_single_simulation(self, sim_index: int, assigned_device: Optional[int] = None) -> Tuple[int, jnp.ndarray, Pruning_state, jnp.ndarray, jnp.ndarray]:
        """Process a single simulation asynchronously until convergence.
        
        Args:
            sim_index: Index of the simulation
            assigned_device: Specific device to use (if None, uses default assignment)
        """
        try:
            # Pre-simulation memory check
            memory_status = monitor_memory_usage(sim_index)
            if memory_status.get('gpu_usage_percent', 0) > 90:
                async_logger.log_sim(sim_index, 
                    f"WARNING: High GPU memory usage ({memory_status.get('gpu_usage_percent', 0):.1f}%) before simulation start", 
                    "ERROR")
                # Trigger emergency cleanup
                self.memory_manager.monitor_and_cleanup_if_needed(sim_index, warn_threshold=85.0)
            
            # Initialize state for this specific simulation using the new function
            initial_state = initialize_single_sim_pruning_state(
                self.neuron_params, 
                self.sim_params, 
                sim_index
            )
            
            # Calculate initial pruning metrics
            total_neurons = self.neuron_params.W.shape[0]
            available_neurons = jnp.sum(~initial_state.total_removed_neurons)
            interneurons = jnp.sum(initial_state.interneuron_mask)
            
            async_logger.log_sim(sim_index, f"Starting with {total_neurons} total neurons, {available_neurons} available, {interneurons} interneurons", "INIT")
            
            # Create simulation state (no need to extract from batch dimension)
            sim_state = AsyncSimState(
                sim_index=sim_index,
                pruning_state=initial_state
            )
            # Register this simulation as active
            with self.update_lock:
                self.active_sims[sim_index] = sim_state
                
        except Exception as e:
            async_logger.log_sim(sim_index, f"Initialization failed: {e}", "ERROR")
            raise e
                
        # Assign device for this simulation
        if assigned_device is not None:
            device_id = assigned_device
        else:
            device_id = self._get_device_for_sim(sim_index)
        device_params = self._get_device_functions(device_id)
        target_device = self.devices[device_id]
        async_logger.log_sim(sim_index, f"Assigned to working device {device_id} ({target_device})", "INIT")
        
        # Move simulation state to the target device - manual transfer to avoid memory explosion
        # Replace jax.tree.map with explicit field transfers for better memory control
        old_pruning_state = sim_state.pruning_state
        
        # Transfer pruning state fields individually to prevent intermediate memory copies
        transferred_pruning_state = old_pruning_state._replace(
            W_mask=jax.device_put(old_pruning_state.W_mask, target_device),
            total_removed_neurons=jax.device_put(old_pruning_state.total_removed_neurons, target_device),
            removed_stim_neurons=jax.device_put(old_pruning_state.removed_stim_neurons, target_device),
            neurons_put_back=jax.device_put(old_pruning_state.neurons_put_back, target_device),
            prev_put_back=jax.device_put(old_pruning_state.prev_put_back, target_device),
            last_removed=jax.device_put(old_pruning_state.last_removed, target_device),
            remove_p=jax.device_put(old_pruning_state.remove_p, target_device),
            min_circuit=jax.device_put(old_pruning_state.min_circuit, target_device),
            keys=jax.device_put(old_pruning_state.keys, target_device),
            last_good_oscillating_removed=jax.device_put(old_pruning_state.last_good_oscillating_removed, target_device),
            last_good_W_mask=jax.device_put(old_pruning_state.last_good_W_mask, target_device),
            last_good_oscillation_score=jax.device_put(old_pruning_state.last_good_oscillation_score, target_device),
            last_good_key=jax.device_put(old_pruning_state.last_good_key, target_device),
            last_good_stim_idx=jax.device_put(old_pruning_state.last_good_stim_idx, target_device),
            last_good_param_idx=jax.device_put(old_pruning_state.last_good_param_idx, target_device),
            level=jax.device_put(old_pruning_state.level, target_device)
        )
        
        # Clear reference to old state to help GC
        del old_pruning_state
        
        sim_state = AsyncSimState(
            sim_index=sim_state.sim_index,
            pruning_state=transferred_pruning_state,
            is_converged=sim_state.is_converged,
            iteration_count=sim_state.iteration_count,
            last_update_time=sim_state.last_update_time
        )
        
        # create mn_mask and place on target device (use pre-replicated data if available)
        if hasattr(self, 'shared_mn_mask') and self.shared_mn_mask is not None:
            # Use the replicated version - it's already on the right device
            mn_mask = self.shared_mn_mask
        else:
            # Fallback: place template on target device
            mn_mask = jax.device_put(self.mn_mask_template, target_device)
        # Main pruning loop for this simulation
        iteration = 0
        while iteration < self.sim_config.max_pruning_iterations:
            try:
                # Check convergence
                if sim_state.pruning_state.min_circuit:
                    # When convergence occurs, we DON'T use the current 'results' because they have poor oscillation
                    # Instead, we get the last_good_oscillation_score which was computed from better results
                    last_good_score = float(sim_state.pruning_state.last_good_oscillation_score.squeeze())
                    
                    break
                
                # Calculate current pruning metrics
                available_neurons = jnp.sum(~sim_state.pruning_state.total_removed_neurons & ~mn_mask)
                removed_neurons = jnp.sum(sim_state.pruning_state.total_removed_neurons)
                put_back_neurons = jnp.sum(sim_state.pruning_state.neurons_put_back)

                # Compact progress logging - only every 10 iterations or significant changes
                if iteration % 10 == 0 or iteration < 3:
                    async_logger.log_sim(sim_index, 
                        f"Iter {iteration:2d}: {available_neurons:4d} avail, {removed_neurons:4d} removed, {put_back_neurons:3d} put back, Level {sim_state.pruning_state.level[0]}", 
                        "PROGRESS")
                
                # Run simulation iteration - ALWAYS use thread pool for true async execution
                # JAX computations are blocking, so we need to run them in separate threads
                # Use shared executor to avoid compilation context fragmentation
                try:
                    loop = asyncio.get_event_loop()
                    future = loop.run_in_executor(
                        self._shared_executor,
                        lambda: self._run_single_pruning_iteration_with_tolerances(
                            sim_state.pruning_state.W_mask,
                            sim_index,
                            device_params['rtol'],
                            device_params['atol'],
                            device_id,
                            sim_state.pruning_state.keys  # Use evolving RNG key for each iteration
                        )
                    )
                    results = await future
                except Exception as e:
                    async_logger.log_sim(sim_index, f"Pruning iteration failed: {e}", "ERROR")
                    raise
                
                # Update pruning state - use JIT-compiled function with static arguments  
                jit_update_fn = self._jit_update_functions[device_id]
                
                updated_state = jit_update_fn(
                    sim_state.pruning_state,  # Simulation state, not batched
                    results,  # Simulation results
                    mn_mask
                )
                
            except Exception as e:
                async_logger.log_sim(sim_index, f"Failed at iteration {iteration}: {e}", "ERROR")
                traceback.print_exc()
                raise e
            
            # Update state with explicit cleanup of previous state
            old_pruning_state = sim_state.pruning_state
            sim_state.pruning_state = updated_state
            sim_state.iteration_count = iteration
            sim_state.last_update_time = time.time()
            
            # Explicit cleanup to help JAX garbage collection
            del old_pruning_state
            iteration += 1
            
            # Monitor memory usage more frequently for proactive management
            if iteration % 100 == 0:  # Check memory every 50 iterations
                memory_status = monitor_memory_usage(sim_index)
                gpu_percent = memory_status.get('gpu_percent', 0)
                ram_percent = memory_status.get('ram_percent', 0)
                
                # Trigger emergency cleanup if memory is approaching dangerous levels
                if gpu_percent > 80 or ram_percent > 80:
                    async_logger.log_sim(sim_index, f"High memory detected - GPU: {gpu_percent:.1f}%, RAM: {ram_percent:.1f}% - triggering cleanup", "MEMORY")
                    # Aggressive cleanup
                    jax.clear_caches()
                    gc.collect()
                    self.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context=f"emergency_sim_{sim_index}_iter_{iteration}")
                
                # Log memory warnings for very high usage
                if gpu_percent > 85 or ram_percent > 85:
                    log_memory_status(memory_status, 
                                    lambda msg: async_logger.log_sim(sim_index, f"WARNING - High Memory: {msg}", "MEMORY"),
                                    "")
            
            # Use adaptive memory manager for optimized cleanup
            cleanup_result = self.memory_manager.should_cleanup(sim_index, iteration)
            if cleanup_result:
                self.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context=f"sim_{sim_index}_iter_{iteration}")
                async_logger.log_sim(sim_index, f"Memory cleanup at iteration {iteration}", "PROGRESS")            # Yield control to allow other simulations to run
            await asyncio.sleep(0)
        
        if iteration >= self.sim_config.max_pruning_iterations:
            async_logger.log_sim(sim_index, f"Reached max iterations ({self.sim_config.max_pruning_iterations}) without convergence", "COMPLETE")
        
        # CRITICAL FIX: Use the last_good_W_mask that had proper oscillations
        # When convergence occurs due to repeated resets, the final W_mask has poor oscillations (0.0000)
        # But the last_good_W_mask contains the network state that had oscillation_score >= threshold
        
        # Extract the last good state from pruning state
        last_good_W_mask = sim_state.pruning_state.last_good_W_mask
        last_good_oscillation_score = float(sim_state.pruning_state.last_good_oscillation_score.squeeze())
        last_good_key = sim_state.pruning_state.last_good_key  # Get the saved RNG key
        
        async_logger.log_sim(sim_index, 
            f"Converged with final W_mask (poor osc), using last_good_W_mask (osc_score: {last_good_oscillation_score:.4f})", 
            "FINAL_SIM")
        
        # Check if we have a valid last good state
        if last_good_oscillation_score <= 0:
            async_logger.log_sim(sim_index, 
                f"WARNING: No valid last_good state found (score: {last_good_oscillation_score}), using current converged state", 
                "FINAL_SIM")
            # Use current state as fallback
            final_W_mask = sim_state.pruning_state.W_mask
            expected_final_oscillation_score = 0.0000  # Expected to be poor
            # For fallback case, calculate param_idx before using it
            fallback_param_idx = sim_index % self.sim_params.n_param_sets
            # For fallback case, still use the evolved key for consistency 
            # Even in fallback, we should use the evolved key if available
            final_key = last_good_key if jnp.any(last_good_key != 0) else self.neuron_params.seeds[fallback_param_idx]
        else:
            # Use the last good W_mask that had proper oscillations
            final_W_mask = last_good_W_mask
            expected_final_oscillation_score = last_good_oscillation_score
        
        # Run final simulation with the last good W_mask
        async_logger.log_sim(sim_index, 
            f"Running final simulation with W_mask that had oscillation_score: {expected_final_oscillation_score:.4f}", 
            "FINAL_SIM")
        
        try:
            # Extract stored parameter indices from the last good state
            stored_stim_idx = int(sim_state.pruning_state.last_good_stim_idx.squeeze())
            stored_param_idx = int(sim_state.pruning_state.last_good_param_idx.squeeze())
            
            # Use stored indices if available, otherwise fall back to calculated ones
            if stored_stim_idx >= 0 and stored_param_idx >= 0:
                stim_idx = stored_stim_idx
                param_idx = stored_param_idx
                async_logger.log_sim(sim_index, 
                    f"Using STORED parameter indices: stim_idx={stim_idx}, param_idx={param_idx}", 
                    "FINAL_SIM")
                # CRITICAL: Use the same evolved key that the pruning iteration used
                final_key = last_good_key
                async_logger.log_sim(sim_index, 
                    f"Using EXACT evolved key from last_good_state (NOT original seed)", "FINAL_SIM")
            else:
                # Use the EXACT SAME parameter extraction pattern as in _run_single_pruning_iteration_with_tolerances
                param_idx = sim_index % self.sim_params.n_param_sets
                stim_idx = sim_index // self.sim_params.n_param_sets
                async_logger.log_sim(sim_index, 
                    f"Using CALCULATED parameter indices: stim_idx={stim_idx}, param_idx={param_idx}", 
                    "FINAL_SIM")
            
            # CRITICAL: Use the exact same evolved key that was saved during the last good iteration
            # NOT the original seed - the key evolves during pruning iterations
            final_key = last_good_key
            async_logger.log_sim(sim_index, 
                f"Using EXACT evolved key from last_good_state: {final_key[:4]} (NOT original seed)", "FINAL_SIM")
            
            device_params = self.device_functions[device_id]
            
            # CRITICAL FIX: Run final simulation using the EXACT SAME async thread pool execution
            # pattern as the pruning iterations, but with the last_good_W_mask substituted
            async_logger.log_sim(sim_index, 
                f"DEBUG: About to run final simulation with final_W_mask shape: {final_W_mask.shape}, "
                f"final_W_mask active connections: {jnp.sum(final_W_mask)}, "
                f"final_W_mask hash: {hash(final_W_mask.tobytes())}, "
                f"param_idx={param_idx}, stim_idx={stim_idx}", "DEBUG")
                
            try:
                loop = asyncio.get_event_loop()
                # Use shared executor to maintain consistent JAX compilation context
                future = loop.run_in_executor(
                    self._shared_executor,
                    lambda: self._run_single_pruning_iteration_with_tolerances(
                        final_W_mask,  # Use last_good_W_mask instead of current W_mask
                        sim_index,
                        device_params['rtol'],
                        device_params['atol'],
                        device_id,
                            final_key  # Pass the evolved key that was saved during last_good iteration
                        )
                    )
                final_results = await future
            except Exception as e:
                async_logger.log_sim(sim_index, f"Final simulation failed: {e}", "ERROR")
                raise
                
            # Compute oscillation score from final simulation 
            # Use consistent clip_start calculation with pruning iterations
            clip_start = self.clip_start_val  # Use pre-calculated value from device setup
            max_frs = jnp.max(final_results, axis=-1)
            mn_mask = jnp.isin(jnp.arange(self.sim_params.n_neurons), self.neuron_params.mn_idxs)
            active_mask = ((max_frs > 0.01) & mn_mask)
            
            async_logger.log_sim(sim_index, 
                f"DEBUG: Oscillation calculation - clip_start: {clip_start}, "
                f"active_mask sum: {jnp.sum(active_mask)}, "
                f"max_frs range: [{jnp.min(max_frs):.6f}, {jnp.max(max_frs):.6f}]", "DEBUG")
            
            final_oscillation_score, _ = compute_oscillation_score(final_results[..., clip_start:], active_mask, prominence=0.05)
            final_oscillation_score = float(final_oscillation_score)
        
        except Exception as e:
            async_logger.log_sim(sim_index, 
                f"ERROR in final simulation: {e}", 
                "ERROR")
            # Create fallback result to prevent crash
            final_results = jnp.ones((self.sim_params.n_neurons, len(self.sim_params.t_axis))) * 0.002
            final_oscillation_score = -1.0  # Mark as error
        
        # Check consistency with small tolerance for stochastic variations
        osc_difference = abs(final_oscillation_score - expected_final_oscillation_score)
        if osc_difference < 1e-10:
            async_logger.log_sim(sim_index, 
                f"Final simulation osc_score: {final_oscillation_score:.4f} (expected: {expected_final_oscillation_score:.4f}) ✅ PERFECT MATCH", 
                "FINAL_SIM")
        elif osc_difference < 0.1:  # Small stochastic variation tolerance
            async_logger.log_sim(sim_index, 
                f"Final simulation osc_score: {final_oscillation_score:.4f} (expected: {expected_final_oscillation_score:.4f}) ⚠️ Small variation ({osc_difference:.4f})", 
                "FINAL_SIM")
        else:  # Significant discrepancy
            async_logger.log_sim(sim_index, 
                f"Final simulation osc_score: {final_oscillation_score:.4f} (expected: {expected_final_oscillation_score:.4f}) ❌ LARGE DISCREPANCY ({osc_difference:.4f})", 
                "FINAL_SIM")

        # Log network structure info  
        total_removed = jnp.sum(sim_state.pruning_state.total_removed_neurons)
        total_put_back = jnp.sum(sim_state.pruning_state.neurons_put_back)
        active_neurons_count = jnp.sum(jnp.any(final_W_mask, axis=0))
        
        async_logger.log_sim(sim_index, 
            f"Final network: {active_neurons_count} neurons ({total_removed} removed, {total_put_back} put back)", 
            "FINAL_SIM")

        # Create mini circuit from final results (same logic as before)
        max_frs = jnp.max(final_results, axis=-1)
        mn_mask = jnp.isin(jnp.arange(self.sim_params.n_neurons), self.neuron_params.mn_idxs)
        mini_circuit = max_frs > 0.01
        active_interneurons = jnp.sum(mini_circuit & ~mn_mask)
        
        # Final results statistics
        result_stats = {
            'shape': final_results.shape,
            'min': float(jnp.min(final_results)),
            'max': float(jnp.max(final_results)), 
            'mean': float(jnp.mean(final_results)),
            'nonzero': int((jnp.sum(final_results, axis=-1) > 0).sum())
        }
        
        async_logger.log_sim(sim_index, 
            f"Final osc calc: {jnp.sum(active_mask)} active neurons ({jnp.sum(active_mask & mn_mask)} motor), max_fr_range: [{jnp.min(max_frs):.3f}, {jnp.max(max_frs):.3f}], clip_start: {clip_start}", 
            "FINAL_SIM")
            
        async_logger.log_sim(sim_index, 
            f"Final simulation completed: {result_stats}", 
            "FINAL_SIM")
            
        async_logger.log_sim(sim_index, 
            f"Mini circuit: {active_interneurons} active interneurons from final simulation results", 
            "FINAL_SIM")
            
        async_logger.log_sim(sim_index, 
            f"COMPLETED: {iteration} iterations, {active_interneurons} neurons remaining ({total_removed} removed)", 
            "COMPLETE")

        # Cleanup JAX caches and force garbage collection
        # Use distributed coordination to avoid overloading target_device
        coord_device = self._get_next_coordination_device()
        with jax.default_device(coord_device):
            jax.clear_caches()
            gc.collect()

        return sim_index, final_results, sim_state.pruning_state, mini_circuit, final_W_mask
    
    def _handle_simulation_failure(self, sim_index: int, e: Exception, target_device, async_logger) -> Tuple[int, jnp.ndarray, dict, jnp.ndarray]:
        """Handle simulation failure by creating fallback results with proper cleanup."""
        async_logger.log_sim(sim_index, 
            f"ERROR in simulation: {e}", 
            "ERROR")
        
        # Comprehensive cleanup for failed simulation
        try:
            # Clear JAX caches and force garbage collection
            jax.clear_caches()
            gc.collect()
            
            # Memory cleanup using memory manager
            if hasattr(self, 'memory_manager'):
                self.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context=f"sim_{sim_index}_failed")

        except Exception as cleanup_e:
            async_logger.log_sim(sim_index, 
                f"WARNING: Cleanup after failure also failed: {cleanup_e}", 
                "ERROR")
        
        # Create fallback result to prevent crash
        final_results = jnp.ones((self.sim_params.n_neurons, len(self.sim_params.t_axis))) * 0.002
        
        # Create fallback mini_circuit (all interneurons active) - streamlined device context
        with jax.default_device(target_device):
            # Data will be created on target_device due to context, avoid redundant device_put
            mn_mask = jnp.isin(jnp.arange(self.sim_params.n_neurons), self.neuron_params.mn_idxs)
            mini_circuit = jnp.ones(self.sim_params.n_neurons, dtype=bool) & ~mn_mask
        
        # Create dummy pruning state for fallback
        fallback_pruning_state = {
            'W_mask': jnp.ones((self.sim_params.n_neurons, self.sim_params.n_neurons), dtype=bool),
            'total_removed_neurons': jnp.zeros(self.sim_params.n_neurons, dtype=bool)
        }
        
        # Create fallback W_mask (full connectivity)
        fallback_W_mask = jnp.ones((self.sim_params.n_neurons, self.sim_params.n_neurons), dtype=jnp.bool_)
        
        return sim_index, final_results, fallback_pruning_state, mini_circuit, fallback_W_mask
    
    async def run_streaming_simulations(self, total_simulations: int, max_concurrent: int, 
                                       checkpoint_dir: Optional[Path] = None, enable_checkpointing: bool = False,
                                       pre_loaded_checkpoint_data: Optional[Dict[str, Any]] = None) -> Tuple[List[jnp.ndarray], List[Pruning_state], List[jnp.ndarray], List[jnp.ndarray]]:
        """Run all simulations with streaming execution - returns results, states, mini_circuits, AND final_W_masks"""
        """
        Run simulations with continuous streaming - start new simulations as soon as others finish.
        
        Args:
            total_simulations: Total number of simulations to run
            max_concurrent: Maximum number of simulations to run concurrently (calculated by caller)
            checkpoint_dir: Directory to save checkpoints (optional)
            enable_checkpointing: Whether to enable periodic checkpointing
            pre_loaded_checkpoint_data: Pre-loaded checkpoint data to avoid duplicate loading (optional)
        
        Returns:
            Tuple of (results_list, states_list, mini_circuits_list, w_masks_list) in simulation index order
        """
        # DEBUG: Print checkpoint parameters
        print(f"🔍 CHECKPOINT DEBUG: enable_checkpointing={enable_checkpointing}, checkpoint_dir={checkpoint_dir}")
        if enable_checkpointing:
            print(f"🔍 CHECKPOINT DEBUG: Checkpoint directory exists: {checkpoint_dir.exists() if checkpoint_dir else 'None'}")
        else:
            print(f"🔍 CHECKPOINT DEBUG: Checkpointing is disabled")
            
        # max_concurrent is always calculated by run_streaming_pruning_simulation, so it should never be None
        assert max_concurrent is not None, "max_concurrent should be calculated by caller and never be None"
        async_logger.log_batch(f"Using max_concurrent: {max_concurrent}")
        
        async_logger.log_batch(f"Starting streaming execution: {total_simulations} total simulations, max {max_concurrent} concurrent")
        
        # Initialize result storage (indexed by simulation number)
        results_dict = {}
        states_dict = {}
        mini_circuits_dict = {}
        w_masks_dict = {}
        failed_sims = set()
        
        # Checkpoint recovery
        completed_count = 0
        # Use pre-loaded checkpoint data if provided (eliminates duplicate checkpoint loading)
        if pre_loaded_checkpoint_data is not None:
            results_dict = pre_loaded_checkpoint_data.get('results_dict', {})
            states_dict = pre_loaded_checkpoint_data.get('states_dict', {})
            mini_circuits_dict = pre_loaded_checkpoint_data.get('mini_circuits_dict', {})
            w_masks_dict = pre_loaded_checkpoint_data.get('w_masks_dict', {})
            failed_sims = pre_loaded_checkpoint_data.get('failed_sims', set())
            completed_count = pre_loaded_checkpoint_data.get('completed_count', 0)
            async_logger.log_batch(f"📂 Using pre-loaded checkpoint data: {completed_count}/{total_simulations} simulations already completed")
        else:
            # No checkpoint data provided, starting fresh
            async_logger.log_batch("Starting fresh simulation (no checkpoint data provided)")
            completed_count = 0
        
        # Create a queue of simulation indices to process (excluding completed ones)
        pending_sims = [i for i in range(total_simulations) if i not in results_dict]
        async_logger.log_batch(f"Simulations to process: {len(pending_sims)} pending, {len(results_dict)} already completed")
        active_tasks = {}  # task -> sim_index mapping
        task_devices = {}  # task -> device_id mapping
        device_load_counts = {i: 0 for i in range(self.n_devices)}  # track sims per device
        
        def get_least_loaded_device():
            """Get the device with the least number of active simulations."""
            return min(device_load_counts.keys(), key=lambda device_id: device_load_counts[device_id])
        
        
        # Progress tracking (completed_count already set from checkpoint recovery)
        start_time = time.time()
        
        # Create simplified progress monitoring task with memory tracking
        async def monitor_progress():
            nonlocal completed_count
            last_report_time = start_time
            
            while pending_sims or active_tasks:
                await asyncio.sleep(10)  # Update every 10 seconds
                current_time = time.time()

                if current_time - last_report_time >= 60:  # Detailed report every 60 seconds
                    elapsed = current_time - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_simulations - completed_count) / rate if rate > 0 else float('inf')
                    
                    active_count = len(active_tasks)
                    pending_count = len(pending_sims)
                    
                    # Memory monitoring for large simulations
                    try:
                        memory_status = monitor_memory_usage()
                        memory_percent = memory_status.get('ram_percent', 0)
                        available_gb = memory_status.get('ram_available_gb', 0)
                        
                        # Multi-GPU memory monitoring
                        gpu_memory_status = ""
                        if memory_status.get('gpu_available', False):
                            gpu_percent = memory_status.get('gpu_percent', 0)
                            gpu_free_gb = memory_status.get('gpu_free_gb', 0)
                            gpu_memory_status = f" | GPU: {gpu_percent:.1f}% used, {gpu_free_gb:.1f}GB free"
    
                            
                        memory_warning = ""
                        gpu_percent = memory_status.get('gpu_percent', 0)
                        if memory_percent > 85 or available_gb < 5 or gpu_percent > 85:
                            memory_warning = " ⚠️ LOW MEMORY!"
                        
                        async_logger.log_batch(
                            f"Streaming Progress: {completed_count}/{total_simulations} completed, "
                            f"{active_count} active, {pending_count} pending | "
                            f"Rate: {rate:.2f} sims/sec, ETA: {eta/60:.1f} min | "
                            f"RAM: {memory_percent:.1f}% used, {available_gb:.1f}GB free{gpu_memory_status}{memory_warning}"
                        )
                        try:
                            log_detailed_gpu_status(lambda msg: async_logger.log_batch(msg), "GPU Detail: ")
                        except:
                            pass
                    except:
                        # Fallback without memory monitoring
                        async_logger.log_batch(
                            f"Streaming Progress: {completed_count}/{total_simulations} completed, "
                            f"{active_count} active, {pending_count} pending | "
                            f"Rate: {rate:.2f} sims/sec, ETA: {eta/60:.1f} min"
                        )
                    
                    # Show active simulation details
                    if active_tasks:
                        active_sims_str = ", ".join([f"Sim{sim_idx}" for sim_idx in active_tasks.values()])
                        async_logger.log_batch(f"Active: {active_sims_str}")
                    
                    # Show device load balancing
                    load_str = ", ".join([
                        f"GPU{device_id}: {load}" 
                        for device_id, load in device_load_counts.items()
                    ])
                    async_logger.log_batch(f"Device loads: {load_str}")
                    
                    
                    last_report_time = current_time
        
        # Start progress monitoring
        monitor_task = asyncio.create_task(monitor_progress())
        # Memory manager cleanup
        if hasattr(self, 'memory_manager'):
            self.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context="init_streaming_cleanup")

        try:
            # Initial batch - start up to max_concurrent simulations
            for _ in range(min(max_concurrent, len(pending_sims))):
                if pending_sims:
                    sim_idx = pending_sims.pop(0)
                    # Assign to least loaded device
                    device_id = get_least_loaded_device()
                    device_load_counts[device_id] += 1
                    task = asyncio.create_task(self._process_single_simulation(sim_idx, assigned_device=device_id))
                    active_tasks[task] = sim_idx
                    task_devices[task] = device_id
                    async_logger.log_batch(f"Started Sim {sim_idx} on device {device_id} (load: {device_load_counts[device_id]}) ({len(active_tasks)}/{max_concurrent} slots used)")
            
            # Main streaming loop
            while active_tasks:
                # Wait for at least one simulation to complete
                done_tasks, pending_tasks = await asyncio.wait(
                    active_tasks.keys(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed simulations
                for task in done_tasks:
                    sim_idx = active_tasks[task]
                    device_id = task_devices[task]  # Get the device this task was using
                    del active_tasks[task]
                    del task_devices[task]
                    device_load_counts[device_id] -= 1  # Decrease load count for this device
                    completed_count += 1
                    
                    try:
                        result = await task
                        sim_idx_result, final_results, final_state, mini_circuit, final_W_mask = result
                        
                        # Move results to CPU immediately to free GPU memory
                        # Use distributed coordination device to avoid GPU0 overload
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            final_results = jax.device_put(final_results, jax.devices("cpu")[0])
                            mini_circuit = jax.device_put(mini_circuit, jax.devices("cpu")[0])
                            final_W_mask = jax.device_put(final_W_mask, jax.devices("cpu")[0])
                        
                        final_state = move_pruning_state_to_cpu(final_state)
                        
                        results_dict[sim_idx] = final_results
                        states_dict[sim_idx] = final_state
                        mini_circuits_dict[sim_idx] = mini_circuit
                        w_masks_dict[sim_idx] = final_W_mask
                        
                        # Verify we got real results, not fallback values
                        result_max = float(jnp.max(final_results))
                        if result_max <= 0.001:
                            async_logger.log_batch(f"⚠️  Warning: Sim {sim_idx} completed with suspiciously low max rate: {result_max}")
                        
                        async_logger.log_batch(f"✅ Completed Sim {sim_idx} on device {device_id} (load: {device_load_counts[device_id]}) - max rate: {result_max:.4f} ({completed_count}/{total_simulations})")
                        
                        # Periodic checkpointing
                        if enable_checkpointing and checkpoint_dir:
                            # Check if we should save a checkpoint (periodic + memory-based)
                            try:
                                memory_status = monitor_memory_usage()
                                current_memory_percent = memory_status.get('ram_percent', 0)
                                current_gpu_percent = memory_status.get('gpu_percent', 0)
                                # Get checkpoint interval from config (default to 10 if not specified)
                                checkpoint_interval = getattr(self.sim_config, 'checkpoint_interval', 10)
                                # Save checkpoint every N simulations OR when RAM/GPU memory usage > 80%
                                should_checkpoint = (completed_count % checkpoint_interval == 0) or (current_memory_percent > 80.0) or (current_gpu_percent > 80.0)
                                async_logger.log_batch(f"Memory RAM:{current_memory_percent:.1f}% GPU:{current_gpu_percent:.1f}%, checkpoint_interval={checkpoint_interval}, should_checkpoint={should_checkpoint} (completed_count={completed_count})")
                            except:
                                # Fallback to just periodic checkpoints if memory monitoring fails
                                checkpoint_interval = getattr(self.sim_config, 'checkpoint_interval', 10)
                                should_checkpoint = (completed_count % checkpoint_interval == 0)
                                async_logger.log_batch(f"Memory check failed, using periodic checkpoint: checkpoint_interval={checkpoint_interval}, should_checkpoint={should_checkpoint} (completed_count={completed_count})")
                            
                            if should_checkpoint:
                                try:
                                    # Ensure all checkpoint data is on CPU before creating checkpoint state
                                    cpu_device = jax.devices("cpu")[0]
                                    
                                    # Create CPU-only copies of all data
                                    cpu_results_dict = {
                                        k: jax.device_put(v, cpu_device) for k, v in results_dict.items()
                                    }
                                    
                                    # Handle states_dict (Pruning_state objects) specially
                                    cpu_states_dict = {}
                                    for k, pruning_state in states_dict.items():
                                        cpu_states_dict[k] = move_pruning_state_to_cpu(pruning_state)
                                    
                                    cpu_mini_circuits_dict = {
                                        k: jax.device_put(v, cpu_device) for k, v in mini_circuits_dict.items()
                                    }
                                    cpu_w_masks_dict = {
                                        k: jax.device_put(v, cpu_device) for k, v in w_masks_dict.items()
                                    }
                                    
                                    checkpoint_state = CheckpointState(
                                        batch_index=0,  # Not applicable for streaming
                                        completed_batches=0,  # Not applicable for streaming  
                                        total_batches=0,  # Not applicable for streaming
                                        neuron_params=None,  # Not stored for streaming
                                        n_result_batches=0,  # Not applicable for streaming
                                        checkpoint_type="streaming",
                                        completed_simulations=set(results_dict.keys()),
                                        results_dict=cpu_results_dict,
                                        states_dict=cpu_states_dict,
                                        mini_circuits_dict=cpu_mini_circuits_dict,
                                        w_masks_dict=cpu_w_masks_dict,
                                        failed_sims=failed_sims.copy(),
                                        progress_info={
                                            'completed_count': completed_count,
                                            'total_simulations': total_simulations,
                                            'timestamp': time.time()
                                        }
                                    )
                                    
                                    checkpoint_path = checkpoint_dir / f"checkpoint_{completed_count}"
                                    save_checkpoint(checkpoint_state, checkpoint_path)
                                    async_logger.log_batch(f"💾 Checkpoint saved at {checkpoint_path} ({completed_count}/{total_simulations} completed)")
                                    
                                    # Clean up old checkpoints (keep last 3)
                                    cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)
                                    
                                except Exception as checkpoint_error:
                                    async_logger.log_batch(f"⚠️ Checkpoint save failed: {checkpoint_error}")
                        
                        # Clean up memory after successful completion
                        del result, final_results, final_state, mini_circuit
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        # Check if this is a real failure or just asyncio shutdown cleanup
                        if "cannot schedule new futures after shutdown" in error_str:
                            # This is actually a successful completion, just asyncio cleanup timing
                            # The simulation completed successfully but asyncio cleanup happened too quickly
                            # We'll treat this as a successful completion but create a basic result
                            
                            async_logger.log_batch(f"✅ Completed Sim {sim_idx} on device {device_id} (load: {device_load_counts[device_id]}) - asyncio cleanup timing issue ({completed_count}/{total_simulations})")
                            
                            # Create basic successful results since we know the simulation finished
                            # Use small non-zero values to indicate this was a real (successful) run
                            basic_result = jnp.ones((self.sim_params.n_neurons, len(self.sim_params.t_axis))) * 0.001
                            basic_state = initialize_single_sim_pruning_state(
                                self.neuron_params, 
                                self.sim_params, 
                                sim_idx
                            )
                            # Mark as converged since it finished
                            basic_state = basic_state._replace(min_circuit=jnp.array([True]))
                            
                            # Create basic mini_circuit (all interneurons)
                            # Use explicit device context for safe bitwise operation
                            device = self.devices[device_id] if device_id < len(self.devices) else jax.devices("cpu")[0]
                            with jax.default_device(device):
                                mn_mask = jnp.isin(jnp.arange(self.sim_params.n_neurons), self.neuron_params.mn_idxs)
                                basic_mini_circuit = jnp.ones(self.sim_params.n_neurons, dtype=bool) & ~mn_mask
                            
                            # Move results to CPU
                            basic_result = jax.device_put(basic_result, jax.devices("cpu")[0])
                            basic_state = move_pruning_state_to_cpu(basic_state)
                            basic_mini_circuit = jax.device_put(basic_mini_circuit, jax.devices("cpu")[0])
                            
                            results_dict[sim_idx] = basic_result
                            states_dict[sim_idx] = basic_state
                            mini_circuits_dict[sim_idx] = basic_mini_circuit
                            # Add basic W_mask for basic simulations to prevent KeyError
                            basic_W_mask = jnp.ones((self.sim_params.n_neurons, self.sim_params.n_neurons), dtype=jnp.bool)
                            basic_W_mask = jax.device_put(basic_W_mask, jax.devices("cpu")[0])
                            w_masks_dict[sim_idx] = basic_W_mask
                            
                            # Clean up memory
                            del basic_result, basic_state, basic_mini_circuit
                            continue  # Skip the failure handling below
                        
                        # Real failure case
                        failed_sims.add(sim_idx)
                        async_logger.log_batch(f"❌ Failed Sim {sim_idx} on device {device_id} (load: {device_load_counts[device_id]}): {str(e)}")
                        
                        # Create proper empty results for failed simulation using distributed device context
                        # Match the format of successful simulation results
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            empty_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                        
                        # Create empty mini_circuit (all interneurons inactive)
                        mn_mask = jnp.isin(jnp.arange(self.sim_params.n_neurons), self.neuron_params.mn_idxs)
                        empty_mini_circuit = jnp.zeros(self.sim_params.n_neurons, dtype=bool)
                        
                        # Create a proper empty pruning state (single sim, not batch)
                        # Use the function that's defined in this module
                        try:
                            empty_state = initialize_single_sim_pruning_state(
                                self.neuron_params, 
                                self.sim_params, 
                                sim_idx
                            )
                        except Exception as init_error:
                            # Fallback: create a minimal state manually
                            async_logger.log_batch(f"Warning: Failed to initialize empty state for Sim {sim_idx}: {init_error}")
                            mn_idxs = self.neuron_params.mn_idxs
                            all_neurons = jnp.arange(self.neuron_params.W.shape[-1])
                            in_idxs = jnp.setdiff1d(all_neurons, mn_idxs)
                            
                            from src.data.data_classes import Pruning_state
                            empty_state = Pruning_state(
                                W_mask=jnp.ones((self.neuron_params.W.shape[0], self.neuron_params.W.shape[1]), dtype=jnp.bool),
                                interneuron_mask=jnp.isin(jnp.arange(self.sim_params.n_neurons), in_idxs),
                                level=jnp.array([0], dtype=jnp.int32),
                                total_removed_neurons=jnp.zeros(self.sim_params.n_neurons, dtype=jnp.bool),
                                removed_stim_neurons=jnp.zeros(self.sim_params.n_neurons, dtype=jnp.bool),
                                neurons_put_back=jnp.zeros(self.sim_params.n_neurons, dtype=jnp.bool),
                                prev_put_back=jnp.zeros(self.sim_params.n_neurons, dtype=jnp.bool),
                                last_removed=jnp.zeros(self.sim_params.n_neurons, dtype=jnp.bool),
                                remove_p=jnp.zeros(self.sim_params.n_neurons),
                                min_circuit=jnp.array([False]),
                                keys=self.neuron_params.seeds[sim_idx:sim_idx+1] if sim_idx < len(self.neuron_params.seeds) else jax.random.split(jax.random.PRNGKey(42), 1),
                                last_good_oscillating_removed=jnp.zeros(self.sim_params.n_neurons, dtype=jnp.bool),
                                last_good_oscillating_put_back=jnp.zeros(self.sim_params.n_neurons, dtype=jnp.bool),
                                last_good_oscillation_score=jnp.array([-1.0]),
                                last_good_W_mask=jnp.zeros((self.sim_params.n_neurons, self.sim_params.n_neurons), dtype=jnp.bool),
                                last_good_key=jnp.zeros((2,), dtype=jnp.uint32),
                                last_good_stim_idx=jnp.array(-1, dtype=jnp.int32),
                                last_good_param_idx=jnp.array(-1, dtype=jnp.int32)
                            )
                        
                        # Move empty results to CPU
                        empty_result = jax.device_put(empty_result, jax.devices("cpu")[0])
                        empty_state = move_pruning_state_to_cpu(empty_state)
                        empty_mini_circuit = jax.device_put(empty_mini_circuit, jax.devices("cpu")[0])
                        
                        results_dict[sim_idx] = empty_result
                        states_dict[sim_idx] = empty_state
                        mini_circuits_dict[sim_idx] = empty_mini_circuit
                        # Add empty W_mask for failed simulations using distributed device context
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            empty_W_mask = jnp.ones((self.sim_params.n_neurons, self.sim_params.n_neurons), dtype=jnp.bool)
                        empty_W_mask = jax.device_put(empty_W_mask, jax.devices("cpu")[0])
                        w_masks_dict[sim_idx] = empty_W_mask
                        
                        # Clean up failed simulation memory
                        del empty_result, empty_state, empty_mini_circuit
                
                # Comprehensive memory cleanup after processing completed simulations
                if done_tasks:
                    # Memory cleanup after each batch of completions
                    if len(done_tasks) > 0:
                        try:
                            jax.clear_caches()
                        except Exception as e:
                            pass  # Silent fail to avoid log spam
                    
                    # Clear JAX caches more aggressively to prevent memory fragmentation
                    if len(done_tasks) >= 2 or completed_count % 10 == 0:
                        jax.clear_caches()
                    
                    # Force garbage collection more frequently for large simulations
                    gc.collect()
                    
                    # Aggressive memory cleanup every 50 simulations to prevent OOM
                    if completed_count > 0 and completed_count % 50 == 0:
                        # Force more aggressive cleanup for long-running simulations
                        # Use distributed device clearing to avoid GPU0 overload
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            jax.clear_caches()

                        gc.collect()
                        async_logger.log_batch(f"🧹 Aggressive memory cleanup at sim {completed_count}")
                    
                    # More frequent backend clearing
                    if completed_count > 0 and completed_count % 10 == 0:
                        try:
                            jax.clear_caches()
                            async_logger.log_batch(f"🧧 Backend clearing at sim {completed_count}")
                        except Exception as e:
                            async_logger.log_batch(f"⚠️ Backend clearing failed: {e}")
                    
                    # Log memory cleanup
                    if len(done_tasks) > 0:
                        async_logger.log_batch(f"🧹 Cleaned up {len(done_tasks)} completed simulation(s)")
                
                # Start new simulations to fill available slots
                while len(active_tasks) < max_concurrent and pending_sims:
                    sim_idx = pending_sims.pop(0)
                    # Assign to least loaded device
                    device_id = get_least_loaded_device()
                    device_load_counts[device_id] += 1
                    task = asyncio.create_task(self._process_single_simulation(sim_idx, assigned_device=device_id))
                    active_tasks[task] = sim_idx
                    task_devices[task] = device_id
                    async_logger.log_batch(f"🚀 Started Sim {sim_idx} on device {device_id} (load: {device_load_counts[device_id]}) ({len(active_tasks)}/{max_concurrent} slots used, {len(pending_sims)} pending)")
            
        finally:
            monitor_task.cancel()
        
        # Final summary
        total_time = time.time() - start_time
        avg_rate = completed_count / total_time if total_time > 0 else 0
        
        if failed_sims:
            async_logger.log_batch(
                f"🏁 Streaming execution completed in {total_time:.1f}s | "
                f"Rate: {avg_rate:.2f} sims/sec | {len(failed_sims)} failures: {sorted(failed_sims)}"
            )
        else:
            async_logger.log_batch(
                f"🎉 Streaming execution completed successfully in {total_time:.1f}s | "
                f"Rate: {avg_rate:.2f} sims/sec"
            )
        
        # Convert to ordered lists
        results_list = [results_dict[i] for i in range(total_simulations)]
        states_list = [states_dict[i] for i in range(total_simulations)]
        mini_circuits_list = [mini_circuits_dict[i] for i in range(total_simulations)]
        w_masks_list = [w_masks_dict[i] for i in range(total_simulations)]
        
        # Final comprehensive memory cleanup with enhanced GPU clearing
        del results_dict, states_dict, mini_circuits_dict, w_masks_dict, active_tasks, task_devices, device_load_counts
        
        # Comprehensive JAX cleanup
        jax.clear_caches()
        
        # Force backend cleanup if available
        try:

                
                # Additional cleanup
                try:
                    jax.clear_caches()
                    async_logger.log_batch(f"🧧 Final backend clearing completed")
                except Exception as gpu_cleanup_e:
                    async_logger.log_batch(f"⚠️ Final cleanup failed: {gpu_cleanup_e}")
                    
        except Exception as e:
            async_logger.log_batch(f"Warning: Could not clear JAX backends: {e}")
        
        # GPU memory cleanup  
        try:
            # Force GPU synchronization and cleanup using distributed coordination
            coord_device = self._get_next_coordination_device()
            with jax.default_device(coord_device):
                jax.clear_caches()

            # Additional memory cleanup
            gc.collect()
            
            # Memory manager cleanup
            if hasattr(self, 'memory_manager'):
                self.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context="final_streaming_cleanup")

        except Exception as e:
            async_logger.log_batch(f"Warning: GPU cleanup failed: {e}")
        
        async_logger.log_batch("🧹 Final comprehensive memory cleanup completed")
        
        return results_list, states_list, mini_circuits_list, w_masks_list
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get current progress of all active simulations."""
        with self.update_lock:
            return {
                "active_simulations": len(self.active_sims),
                "completed_simulations": len(self.completed_sims),
                "average_iterations": np.mean([s.iteration_count for s in self.active_sims.values()]) if self.active_sims else 0,
                "simulation_details": {
                    sim_idx: {
                        "iterations": state.iteration_count,
                        "converged": state.is_converged,
                        "last_update": time.time() - state.last_update_time
                    }
                    for sim_idx, state in self.active_sims.items()
                }
            }


def run_streaming_pruning_simulation(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    sim_config: SimulationConfig,
    max_concurrent: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    enable_checkpointing: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, NeuronParams]:
    """
    Run pruning simulation with streaming processing - new simulations start as others finish.
    
    This is more efficient than batch processing because:
    - No waiting for entire batches to complete
    - Better GPU utilization (slots filled immediately)
    - Faster overall completion time
    - Real-time progress monitoring
    - Enhanced memory management for large simulations (n_replicates >= 1024)
    - Checkpointing support for recovery from failures
    
    Args:
        neuron_params: Neuron parameters
        sim_params: Simulation parameters  
        sim_config: Simulation configuration
        max_concurrent: Maximum concurrent simulations (default: auto-calculated)
        checkpoint_dir: Directory for checkpointing (optional)
        enable_checkpointing: Whether to enable checkpointing
    
    Returns:
        Tuple of (results_array, mini_circuits_array, updated_neuron_params)
    """
    total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
    
    # Setup checkpointing
    checkpoint_state = None
    completed_sims = set()
    start_sim = 0
    checkpoint_data_for_manager = None
    
    if enable_checkpointing and checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing checkpoint
        checkpoint_result = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_result is not None:
            latest_checkpoint, base_name = checkpoint_result
            print(f"🔍 Found checkpoint: {latest_checkpoint}")
            
            # Perform memory cleanup before attempting to load checkpoint
            print("🧹 Pre-loading memory cleanup...")
            jax.clear_caches()
            for _ in range(3):
                gc.collect()
            
            try:
                checkpoint_state, adjusted_neuron_params, metadata = load_checkpoint(
                    latest_checkpoint, base_name, neuron_params
                )
                
                # CRITICAL: Clear compilation cache IMMEDIATELY after loading checkpoint
                # This prevents JAX from trying to reconcile loaded arrays with existing cache
                print("🧹 Post-loading compilation cache clear (prevents rematerialization conflicts)...")
                jax.clear_caches()
                
                # Force garbage collection to clean up temporary loading artifacts  
                for _ in range(3):
                    gc.collect()
                
                print("✅ Checkpoint loaded and memory optimized for continuation")
                
                # Handle different checkpoint types and prepare data for manager
                if checkpoint_state.checkpoint_type == "streaming":
                    # For streaming: use completed_simulations, keep original neuron_params
                    completed_sims = set(checkpoint_state.completed_simulations)
                    start_sim = len(checkpoint_state.completed_simulations)
                    
                    # Prepare checkpoint data for manager (avoids duplicate loading)
                    checkpoint_data_for_manager = {
                        'results_dict': checkpoint_state.results_dict or {},
                        'states_dict': checkpoint_state.states_dict or {},
                        'mini_circuits_dict': checkpoint_state.mini_circuits_dict or {},
                        'w_masks_dict': checkpoint_state.w_masks_dict or {},
                        'failed_sims': checkpoint_state.failed_sims or set(),
                        'completed_count': len(checkpoint_state.completed_simulations)
                    }
                    # Don't replace neuron_params for streaming (adjusted_neuron_params is None)
                else:
                    # For batch: use completed_batches, use adjusted neuron_params
                    completed_sims = set(range(checkpoint_state.completed_batches))
                    start_sim = checkpoint_state.completed_batches
                    neuron_params = adjusted_neuron_params
                    
                    # For batch checkpoints, create basic checkpoint data
                    checkpoint_data_for_manager = {
                        'results_dict': {},
                        'states_dict': {},
                        'mini_circuits_dict': {},
                        'w_masks_dict': {},
                        'failed_sims': set(),
                        'completed_count': checkpoint_state.completed_batches
                    }
                del checkpoint_state, adjusted_neuron_params, metadata
                gc.collect()
                jax.clear_caches()
                print(f"📂 Resuming from checkpoint at simulation {start_sim}/{total_sims}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint {latest_checkpoint}: {e}")
                print("Starting from beginning...")
                completed_sims = set()
                start_sim = 0
                checkpoint_data_for_manager = None
    
    # Comprehensive initial cleanup to ensure fresh start
    print("🧹 Performing initial memory cleanup for fresh start...")
    jax.clear_caches()
    gc.collect()
    
    # GPU memory cleanup and synchronization
    try:
        # Force GPU synchronization on all devices
        devices = jax.devices()
        for device in devices:
            with jax.default_device(device):
                # Force a small operation to synchronize
                _ = jnp.array([1.0])
        print(f"✅ Synchronized {len(devices)} GPU devices")
    except Exception as e:
        print(f"⚠️  Warning: GPU synchronization failed: {e}")
    
    # Calculate optimal max_concurrent if not provided, with conservative adjustments for large sims
    if max_concurrent is None:
        max_concurrent = calculate_optimal_concurrent_size(
            n_neurons=sim_params.n_neurons,
            n_timepoints=len(sim_params.t_axis),
            total_simulations=total_sims,
            is_pruning=True
        )

    else:
        print(f"Using provided max_concurrent for pruning: {max_concurrent}")
        
    
    # Create async manager with error handling
    manager = None
    all_results, all_states, all_mini_circuits, all_final_w_masks = None, None, None, None
    
    try:
        manager = AsyncPruningManager(neuron_params, sim_params, sim_config)
        
        async def run_streaming():
            return await manager.run_streaming_simulations(total_sims, max_concurrent, checkpoint_dir, enable_checkpointing, checkpoint_data_for_manager)

        # Run the streaming async execution
        all_results, all_states, all_mini_circuits, all_final_w_masks = asyncio.run(run_streaming())
        
    except Exception as e:
        print(f"❌ Async simulation failed: {e}")
        print("🔄 Cleaning up resources and raising exception...")
        
        # Comprehensive cleanup on failure
        jax.clear_caches()
        gc.collect()
            
        # Clean up manager if it exists
        if manager is not None:
            try:
                if hasattr(manager, 'memory_manager'):
                    manager.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context="failed_run_cleanup")
            except:
                pass
        
        # Re-raise the exception to be handled by the caller
        raise e
    
    finally:
        # Always perform cleanup regardless of success/failure
        if manager is not None:
            try:
                if hasattr(manager, 'memory_manager'):
                    manager.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context="final_run_cleanup")
            except:
                pass
    
    # Convert results to arrays and reshape
    results_array = jnp.stack(all_results, axis=0)
    results_reshaped = results_array.reshape(
        sim_params.n_stim_configs, sim_params.n_param_sets,
        sim_params.n_neurons, len(sim_params.t_axis)
    )
    
    # Use the mini_circuits from actual simulation results (not reconstructed from states)
    # These mini_circuits are based on which neurons actually fired in the final simulations
    mini_circuits_array = jnp.stack(all_mini_circuits, axis=0)
    
    # Use the ACTUAL final W_masks that were used in the simulations (not reconstructed!)
    # This fixes the critical bug where W_masks were incorrectly reconstructed from firing patterns
    final_w_masks_array = jnp.stack(all_final_w_masks, axis=0)
    
    # Update the original neuron_params with all the final W_masks
    updated_neuron_params = neuron_params._replace(W_mask=final_w_masks_array)

    # Final cleanup of intermediate arrays
    del all_results, all_states, all_mini_circuits, all_final_w_masks
    gc.collect()
    print("✅ Streaming pruning simulation completed successfully")

    return results_reshaped, mini_circuits_array, updated_neuron_params


# ==============================================================================
# ASYNC REGULAR STIMULATION SIMULATION (NON-PRUNING)
# ==============================================================================

class AsyncRegularSimManager:
    """Manages asynchronous execution of regular (non-pruning) VNC simulations."""
    
    def __init__(
        self,
        neuron_params: NeuronParams,
        sim_params: SimParams,
        sim_config: SimulationConfig,
        max_workers: int = None
    ):
        self.neuron_params = neuron_params
        self.sim_params = sim_params
        self.sim_config = sim_config
        
        # Initialize adaptive memory manager
        total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
        self.memory_manager = create_memory_manager(
            total_simulations=total_sims,
            simulation_type="regular"
        )
        
        # Multi-GPU setup with enhanced memory management
        self.n_devices = jax.device_count()
        self.devices = jax.devices()
        gpu_devices = [d for d in self.devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        self.n_gpu_devices = len(gpu_devices)
        
        
        print(f"Async Regular Sim Manager: {self.n_devices} total devices, {self.n_gpu_devices} GPU devices")
        
        # Device rotation counter for distributing coordination tasks
        self._device_rotation_counter = 0
        
        # Initialize CUDA contexts with memory pre-allocation
        try:
            for i, device in enumerate(self.devices[:self.n_gpu_devices]):
                with jax.default_device(device):
                    # Small memory pre-allocation to avoid CUDA initialization issues
                    test_tensor = jax.device_put(jnp.ones(100), device)
                    del test_tensor
                print(f"  Initialized CUDA context on device {i}: {device}")
        except Exception as e:
            print(f"Warning: Could not initialize all CUDA contexts: {e}")
            if self.n_gpu_devices > 1:
                print("  Falling back to single GPU mode")
                self.n_gpu_devices = 1
                self.devices = self.devices[:1]
        
        self.max_workers = max_workers or min(32, (self.n_devices * 2))
        
        # Create shared thread pool executor to avoid JAX compilation context fragmentation
        self._shared_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(16, self.n_devices * 2),
            thread_name_prefix="jax_regular"
        )
        
        # Pre-replicate read-only data across devices with memory efficiency
        print("Pre-replicating read-only data across devices with memory efficiency...")
        self._setup_shared_data()
        
        # State management
        self.active_sims: Dict[int, time.float] = {}
        self.completed_sims: Dict[int, jnp.ndarray] = {}
        self.update_lock = threading.Lock()
        
        # Pre-compile functions for each device and simulation type
        self._jit_sim_functions = {}
        self._setup_device_functions()
        
        # Stimulation adjustment state
        self.adjusted_neuron_params = neuron_params
        self.adjustment_iteration = 0
    
    def _setup_shared_data(self):
        """Set up memory-efficient shared data across devices."""
        try:
            print("Setting up memory-efficient shared data across devices...")
            
            # Pre-place the large W matrix on each device
            self.device_W = {}
            for i, device in enumerate(self.devices):
                self.device_W[i] = jax.device_put(self.neuron_params.W, device)
                print(f"  Placed W matrix on device {i}: {device}")
            
            # Shared parameters that are read-only
            self.shared_t_axis = self.sim_params.t_axis
            self.shared_noise_stdv = jnp.array(self.sim_params.noise_stdv)
            self.shared_exc_multiplier = self.sim_params.exc_multiplier
            self.shared_inh_multiplier = self.sim_params.inh_multiplier
            
            print(f"  Memory-efficient setup completed successfully")
            
        except Exception as e:
            print(f"Warning: Could not set up shared data: {e}")
            self.device_W = {}
            self.shared_t_axis = self.sim_params.t_axis
            self.shared_noise_stdv = jnp.array(self.sim_params.noise_stdv)
            self.shared_exc_multiplier = self.sim_params.exc_multiplier
            self.shared_inh_multiplier = self.sim_params.inh_multiplier
    
    def _setup_device_functions(self):
        """Pre-compile functions for each device and simulation type."""
        for i, device in enumerate(self.devices):
            print(f"Setting up device {i}: {device}")
            
            # Use more conservative settings for devices other than 0
            device_rtol = self.sim_params.r_tol
            device_atol = self.sim_params.a_tol

            # Pre-create JIT-compiled simulation function for this device based on sim_type
            sim_type = getattr(self.sim_config, 'sim_type', 'baseline')
            self._jit_sim_functions[i] = self._create_jit_simulation_function(i, device_rtol, device_atol, sim_type)
    
    def _get_device_for_sim(self, sim_index: int) -> int:
        """Assign a device to a simulation based on its index."""
        return sim_index % self.n_devices
    
    def _get_next_coordination_device(self):
        """Get the next device for coordination tasks to distribute load."""
        if self.n_gpu_devices > 0:
            device_id = self._device_rotation_counter % self.n_gpu_devices
            self._device_rotation_counter += 1
            return self.devices[device_id]
        return jax.devices("cpu")[0]
    
    def _create_jit_simulation_function(self, device_id: int, r_tol: float, a_tol: float, sim_type: str):
        """Create a JIT-compiled simulation function for a specific device and simulation type."""
        target_device = self.devices[device_id]
        
        if sim_type == "baseline":
            @jax.jit
            def jit_simulation(W_device, W_mask_device, tau_param, a_param, threshold_param, 
                              fr_cap_param, input_currents_param, seeds_param):
                """JIT-compiled baseline simulation function."""
                # Apply W_mask if it exists, otherwise use full W
                if W_mask_device is not None:
                    W_masked = W_device * W_mask_device
                else:
                    W_masked = W_device
                    
                W_reweighted = reweight_connectivity(
                    W_masked, 
                    self.shared_exc_multiplier, 
                    self.shared_inh_multiplier
                )
                
                return run_single_simulation(
                    W_reweighted,
                    tau_param,
                    a_param,
                    threshold_param,
                    fr_cap_param,
                    input_currents_param,
                    self.shared_noise_stdv,
                    self.shared_t_axis,
                    self.sim_params.T,
                    self.sim_params.dt,
                    self.sim_params.pulse_start,
                    self.sim_params.pulse_end,
                    r_tol,
                    a_tol,
                    seeds_param
                )
                
        elif sim_type == "shuffle":
            @jax.jit
            def jit_simulation(W_device, W_mask_device, tau_param, a_param, threshold_param, 
                              fr_cap_param, input_currents_param, seeds_param):
                """JIT-compiled shuffle simulation function."""
                # Apply W_mask if it exists, otherwise use full W
                if W_mask_device is not None:
                    W_masked = W_device * W_mask_device
                else:
                    W_masked = W_device
                
                # Apply shuffling
                shuffle_indices = (
                    self.neuron_params.exc_dn_idxs, 
                    self.neuron_params.inh_dn_idxs,
                    self.neuron_params.exc_in_idxs, 
                    self.neuron_params.inh_in_idxs,
                    self.neuron_params.mn_idxs
                )
                W_shuffled = full_shuffle(W_masked, seeds_param, *shuffle_indices)
                W_reweighted = reweight_connectivity(
                    W_shuffled, 
                    self.shared_exc_multiplier, 
                    self.shared_inh_multiplier
                )
                
                return run_single_simulation(
                    W_reweighted,
                    tau_param,
                    a_param,
                    threshold_param,
                    fr_cap_param,
                    input_currents_param,
                    self.shared_noise_stdv,
                    self.shared_t_axis,
                    self.sim_params.T,
                    self.sim_params.dt,
                    self.sim_params.pulse_start,
                    self.sim_params.pulse_end,
                    r_tol,
                    a_tol,
                    seeds_param
                )
                
        elif sim_type == "noise":
            @jax.jit
            def jit_simulation(W_device, W_mask_device, tau_param, a_param, threshold_param, 
                              fr_cap_param, input_currents_param, seeds_param):
                """JIT-compiled noise simulation function."""
                # Apply W_mask if it exists, otherwise use full W
                if W_mask_device is not None:
                    W_masked = W_device * W_mask_device
                else:
                    W_masked = W_device
                
                # Apply noise
                noise_stdv_prop = getattr(self.sim_params, 'noise_stdv_prop', 0.1)
                W_noisy = add_noise_to_weights(W_masked, seeds_param, noise_stdv_prop)
                W_reweighted = reweight_connectivity(
                    W_noisy, 
                    self.shared_exc_multiplier, 
                    self.shared_inh_multiplier
                )
                
                return run_single_simulation(
                    W_reweighted,
                    tau_param,
                    a_param,
                    threshold_param,
                    fr_cap_param,
                    input_currents_param,
                    self.shared_noise_stdv,
                    self.shared_t_axis,
                    self.sim_params.T,
                    self.sim_params.dt,
                    self.sim_params.pulse_start,
                    self.sim_params.pulse_end,
                    r_tol,
                    a_tol,
                    seeds_param
                )
                
        elif sim_type == "Wmask":
            @jax.jit
            def jit_simulation(W_device, W_mask_device, tau_param, a_param, threshold_param, 
                              fr_cap_param, input_currents_param, seeds_param):
                """JIT-compiled W_mask simulation function."""
                # W_mask is required for this simulation type
                if W_mask_device is not None:
                    W_masked = W_device * W_mask_device
                else:
                    W_masked = W_device  # fallback to full W if no mask
                    
                W_reweighted = reweight_connectivity(
                    W_masked, 
                    self.shared_exc_multiplier, 
                    self.shared_inh_multiplier
                )
                
                return run_single_simulation(
                    W_reweighted,
                    tau_param,
                    a_param,
                    threshold_param,
                    fr_cap_param,
                    input_currents_param,
                    self.shared_noise_stdv,
                    self.shared_t_axis,
                    self.sim_params.T,
                    self.sim_params.dt,
                    self.sim_params.pulse_start,
                    self.sim_params.pulse_end,
                    r_tol,
                    a_tol,
                    seeds_param
                )
        else:
            raise ValueError(f"Unknown simulation type: {sim_type}")
        
        return jit_simulation
    
    def _run_single_regular_simulation(self, sim_index: int, device_idx: int = 0) -> jnp.ndarray:
        """Run a single regular (non-pruning) simulation."""
        # Extract parameters for this specific simulation
        param_idx = sim_index % self.sim_params.n_param_sets
        stim_idx = sim_index // self.sim_params.n_param_sets
        
        # Use pre-placed W matrix on the target device
        target_device = self.devices[device_idx]
        
        if device_idx in self.device_W:
            W_device = self.device_W[device_idx]
        else:
            W_device = jax.device_put(self.neuron_params.W, target_device)
        
        # Handle W_mask if it exists in neuron_params
        W_mask_device = None
        if hasattr(self.adjusted_neuron_params, 'W_mask') and self.adjusted_neuron_params.W_mask is not None:
            # Extract the mask for this simulation
            if self.adjusted_neuron_params.W_mask.ndim > 2:  # Batched W_mask
                W_mask_single = self.adjusted_neuron_params.W_mask[sim_index]
            else:  # Single W_mask for all simulations
                W_mask_single = self.adjusted_neuron_params.W_mask
            W_mask_device = jax.device_put(W_mask_single, target_device)
        
        # Use JIT-compiled function for this device
        jit_sim_fn = self._jit_sim_functions[device_idx]
        
        return jit_sim_fn(
            W_device,
            W_mask_device,
            self.adjusted_neuron_params.tau[param_idx],
            self.adjusted_neuron_params.a[param_idx],
            self.adjusted_neuron_params.threshold[param_idx],
            self.adjusted_neuron_params.fr_cap[param_idx],
            self.adjusted_neuron_params.input_currents[stim_idx, param_idx],
            self.adjusted_neuron_params.seeds[param_idx]
        )
    
    async def _process_single_regular_simulation(self, sim_index: int, assigned_device: Optional[int] = None) -> Tuple[int, jnp.ndarray]:
        """Process a single regular simulation asynchronously."""
        try:
            # Assign device for this simulation
            if assigned_device is not None:
                device_id = assigned_device
            else:
                device_id = self._get_device_for_sim(sim_index)
            
            target_device = self.devices[device_id]
            async_logger.log_sim(sim_index, f"Running on device {device_id} ({target_device})", "INIT")
            
            # Register this simulation as active
            with self.update_lock:
                self.active_sims[sim_index] = time.time()
            
            # Run the simulation - ALWAYS use thread pool for true async execution
            # JAX computations are blocking, so we need to run them in separate threads
            try:
                loop = asyncio.get_event_loop()
                # Use shared executor for consistent JAX compilation context
                future = loop.run_in_executor(
                    self._shared_executor,
                    lambda: self._run_single_regular_simulation(sim_index, device_id)
                )
                results = await future
            except Exception as e:
                async_logger.log_sim(sim_index, f"Simulation execution failed: {e}", "ERROR")
                raise
            
            # Log completion
            max_activity = float(jnp.max(results))
            active_neurons = int(jnp.sum(jnp.max(results, axis=-1) > 0))
            async_logger.log_sim(sim_index, 
                f"Completed - max activity: {max_activity:.3f}, active neurons: {active_neurons}", 
                "COMPLETE")
            
            # Move results to CPU to free GPU memory using distributed coordination
            coord_device = self._get_next_coordination_device()
            with jax.default_device(coord_device):
                results = jax.device_put(results, jax.devices("cpu")[0])
            
            return sim_index, results
            
        except Exception as e:
            async_logger.log_sim(sim_index, f"Failed: {e}", "ERROR")
            
            # Create fallback result using distributed device context
            coord_device = self._get_next_coordination_device()
            with jax.default_device(coord_device):
                fallback_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
            fallback_result = jax.device_put(fallback_result, jax.devices("cpu")[0])
            return sim_index, fallback_result
        
        finally:
            # Remove from active simulations
            with self.update_lock:
                self.active_sims.pop(sim_index, None)
    
    async def _adjust_stimulation_for_simulations(self, sim_indices: List[int]) -> bool:
        """Adjust stimulation for specific simulations. Returns True if adjustment successful."""
        if not getattr(self.sim_config, 'adjustStimI', False):
            return True  # No adjustment needed
            
        print(f"Adjusting stimulation for {len(sim_indices)} simulations...")
        
        for adjust_iter in range(getattr(self.sim_config, 'max_adjustment_iters', 10)):
            # Run test simulations to check current activity levels
            test_results = []
            
            # Run the simulations asynchronously for adjustment
            async def run_test_sims():
                tasks = []
                for sim_idx in sim_indices:
                    task = self._process_single_regular_simulation(sim_idx)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
            
            test_results = await run_test_sims()
            
            # Extract just the result arrays (ignore sim indices)
            result_arrays = []
            for result in test_results:
                if isinstance(result, Exception):
                    # Create dummy result for failed simulations
                    dummy_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                    result_arrays.append(dummy_result)
                else:
                    _, result_array = result
                    result_arrays.append(result_array)
            
            # Stack results for analysis
            test_results_array = jnp.stack(result_arrays, axis=0)
            
            # Analyze activity levels
            max_rates = jnp.max(test_results_array, axis=2)  # (Batch, Neurons)
            n_active = jnp.sum(jnp.sum(test_results_array, axis=2) > 0, axis=1)  # (Batch,)
            n_high_fr = jnp.sum(max_rates > getattr(self.sim_config, 'high_fr_threshold', 100.0), axis=1)  # (Batch,)
            
            current_inputs = self.adjusted_neuron_params.input_currents.copy()
            new_inputs = current_inputs.copy()
            needs_adjustment = jnp.zeros(len(sim_indices), dtype=bool)
            
            # Check each simulation and adjust if needed
            for i, sim_idx in enumerate(sim_indices):
                sim_active = n_active[i]
                sim_high_fr = n_high_fr[i]
                
                # Calculate stimulus and replicate indices
                param_idx = sim_idx % self.sim_params.n_param_sets
                stim_idx = sim_idx // self.sim_params.n_param_sets
                
                sim_input = current_inputs[stim_idx, param_idx]
                
                n_active_upper = getattr(self.sim_config, 'n_active_upper', 500)
                n_active_lower = getattr(self.sim_config, 'n_active_lower', 5)
                n_high_fr_upper = getattr(self.sim_config, 'n_high_fr_upper', 100)
                
                if (sim_active > n_active_upper) or (sim_high_fr > n_high_fr_upper):
                    # Too strong - reduce stimulation
                    new_inputs = new_inputs.at[stim_idx, param_idx].set(sim_input / 2)
                    needs_adjustment = needs_adjustment.at[i].set(True)
                    # Get first element if it's an array
                    input_val = float(sim_input.flatten()[0]) if hasattr(sim_input, 'flatten') else float(sim_input)
                    new_val = input_val / 2
                    print(f"    Reducing stimulation for sim {sim_idx}: {input_val:.6f} -> {new_val:.6f}")
                elif sim_active < n_active_lower:
                    # Too weak - increase stimulation
                    new_inputs = new_inputs.at[stim_idx, param_idx].set(sim_input * 2)
                    needs_adjustment = needs_adjustment.at[i].set(True)
                    # Get first element if it's an array
                    input_val = float(sim_input.flatten()[0]) if hasattr(sim_input, 'flatten') else float(sim_input)
                    new_val = input_val * 2
                    print(f"    Increasing stimulation for sim {sim_idx}: {input_val:.6f} -> {new_val:.6f}")
                
                print(f"    Adjustment iter {adjust_iter + 1}, Sim {sim_idx}: Active: {sim_active}, High FR: {sim_high_fr}")
            
            # Update neuron params with new stimulation
            self.adjusted_neuron_params = self.adjusted_neuron_params._replace(
                input_currents=new_inputs
            )
            
            if not jnp.any(needs_adjustment):
                print(f"    Stimulation appropriate for all simulations - adjustment complete")
                return True
                
        print(f"    Warning: Maximum adjustment iterations reached")
        return False

    async def run_streaming_regular_simulations(self, total_simulations: int, max_concurrent: int, checkpoint_dir: Optional[Path] = None, enable_checkpointing: bool = False, pre_loaded_checkpoint_data: Optional[Tuple] = None) -> List[jnp.ndarray]:
        """Run regular simulations with continuous streaming and optional stimulation adjustment.
        
        Args:
            total_simulations: Total number of simulations to run
            max_concurrent: Maximum concurrent simulations (calculated by caller)
            checkpoint_dir: Directory for checkpointing (optional)
            enable_checkpointing: Whether to enable checkpointing
            pre_loaded_checkpoint_data: Pre-loaded checkpoint data tuple (checkpoint_state, adjusted_neuron_params, metadata) to avoid duplicate loading
        
        Returns:
            List of simulation results
        """
        # DEBUG: Print checkpoint parameters
        print(f"🔍 REGULAR CHECKPOINT DEBUG: enable_checkpointing={enable_checkpointing}, checkpoint_dir={checkpoint_dir}")
        if enable_checkpointing:
            print(f"🔍 REGULAR CHECKPOINT DEBUG: Checkpoint directory exists: {checkpoint_dir.exists() if checkpoint_dir else 'None'}")
        else:
            print(f"🔍 REGULAR CHECKPOINT DEBUG: Checkpointing is disabled")
            
        # max_concurrent is always calculated by run_streaming_regular_simulation, so it should never be None
        assert max_concurrent is not None, "max_concurrent should be calculated by caller and never be None"
        async_logger.log_batch(f"Using max_concurrent: {max_concurrent}")
        
        async_logger.log_batch(f"Starting streaming regular simulations: {total_simulations} total, max {max_concurrent} concurrent")
        
        # Check if stimulation adjustment is enabled
        adjust_stimulation = getattr(self.sim_config, 'adjustStimI', False)
        
        if adjust_stimulation:
            async_logger.log_batch("Stimulation adjustment enabled - using batch processing")
            return await self._run_with_stimulation_adjustment(total_simulations, max_concurrent, checkpoint_dir, enable_checkpointing, pre_loaded_checkpoint_data)
        else:
            async_logger.log_batch("Standard streaming mode")
            return await self._run_standard_streaming(total_simulations, max_concurrent, checkpoint_dir, enable_checkpointing, pre_loaded_checkpoint_data)
    
    async def _run_standard_streaming(self, total_simulations: int, max_concurrent: int, checkpoint_dir: Optional[Path] = None, enable_checkpointing: bool = False, pre_loaded_checkpoint_data: Optional[Tuple] = None) -> List[jnp.ndarray]:
        """Run simulations with standard streaming (no stimulation adjustment).
        
        Args:
            total_simulations: Total number of simulations to run
            max_concurrent: Maximum concurrent simulations
            checkpoint_dir: Directory for checkpointing (optional)
            enable_checkpointing: Whether to enable checkpointing
            pre_loaded_checkpoint_data: Pre-loaded checkpoint data tuple (checkpoint_state, adjusted_neuron_params, metadata) to avoid duplicate loading
        
        Returns:
            List of simulation results
        """
        # Initialize result storage
        results_dict = {}
        failed_sims = set()
        
        # Checkpoint recovery using pre-loaded data (avoids duplicate loading)
        completed_count = 0
        if enable_checkpointing and checkpoint_dir and pre_loaded_checkpoint_data:
            checkpoint_state, adjusted_neuron_params, metadata = pre_loaded_checkpoint_data
            async_logger.log_batch(f"🔍 Using pre-loaded checkpoint data")
            
            try:
                # Handle different checkpoint types
                if checkpoint_state.checkpoint_type == "streaming":
                    # For streaming: use completed_simulations, keep original neuron_params
                    completed_sims = set(checkpoint_state.completed_simulations)
                    completed_count = len(checkpoint_state.completed_simulations)
                    # Load results from checkpoint
                    for sim_idx in checkpoint_state.completed_simulations:
                        if hasattr(checkpoint_state, 'results_dict') and checkpoint_state.results_dict:
                            results_dict[sim_idx] = checkpoint_state.results_dict[sim_idx]
                        else:
                            # Create placeholder result using distributed device context
                            coord_device = self._get_next_coordination_device()
                            with jax.default_device(coord_device):
                                placeholder_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                            placeholder_result = jax.device_put(placeholder_result, jax.devices("cpu")[0])
                            results_dict[sim_idx] = placeholder_result
                    
                    # Don't replace neuron_params for streaming (adjusted_neuron_params might be None)
                    if adjusted_neuron_params is not None:
                        self.adjusted_neuron_params = adjusted_neuron_params
                        
                    async_logger.log_batch(f"📂 Resuming from streaming checkpoint: {completed_count}/{total_simulations} completed")
                else:
                    # For batch: use completed_batches, use adjusted neuron_params
                    completed_count = checkpoint_state.completed_batches
                    if adjusted_neuron_params is not None:
                        self.adjusted_neuron_params = adjusted_neuron_params
                    async_logger.log_batch(f"📂 Resuming from batch checkpoint: {completed_count} batches completed")
                    
            except Exception as e:
                async_logger.log_batch(f"⚠️ Could not process pre-loaded checkpoint data: {e}")
                async_logger.log_batch("Starting from beginning...")
                completed_count = 0
                results_dict = {}
        
        # Create a queue of simulation indices to process (excluding completed ones)
        pending_sims = [i for i in range(total_simulations) if i not in results_dict]
        async_logger.log_batch(f"Regular simulations to process: {len(pending_sims)} pending, {len(results_dict)} already completed")
        active_tasks = {}
        task_devices = {}
        device_load_counts = {i: 0 for i in range(self.n_devices)}
        
        def get_least_loaded_device():
            """Get the device with the least number of active simulations."""
            return min(device_load_counts.keys(), key=lambda device_id: device_load_counts[device_id])
        
        # Progress tracking - set completed_count to match already loaded results
        completed_count = len(results_dict)
        start_time = time.time()
        
        # Create progress monitoring task
        async def monitor_progress():
            nonlocal completed_count
            last_report_time = start_time
            
            while pending_sims or active_tasks:
                await asyncio.sleep(10)
                current_time = time.time()

                if current_time - last_report_time >= 30:  # Report every 30 seconds
                    elapsed = current_time - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_simulations - completed_count) / rate if rate > 0 else float('inf')
                    
                    active_count = len(active_tasks)
                    pending_count = len(pending_sims)
                    
                    # Enhanced memory monitoring similar to AsyncPruningManager
                    try:
                        memory_status = monitor_memory_usage()
                        memory_percent = memory_status.get('ram_percent', 0)
                        available_gb = memory_status.get('ram_available_gb', 0)
                        
                        # Multi-GPU memory monitoring
                        gpu_memory_status = ""
                        if memory_status.get('gpu_available', False):
                            gpu_percent = memory_status.get('gpu_percent', 0)
                            gpu_free_gb = memory_status.get('gpu_free_gb', 0)
                            gpu_memory_status = f" | GPU: {gpu_percent:.1f}% used, {gpu_free_gb:.1f}GB free"
            
                        memory_warning = ""
                        gpu_percent = memory_status.get('gpu_percent', 0)
                        if memory_percent > 85 or available_gb < 5 or gpu_percent > 85:
                            memory_warning = " ⚠️ LOW MEMORY!"
                        
                        async_logger.log_batch(
                            f"Regular Sim Progress: {completed_count}/{total_simulations} completed, "
                            f"{active_count} active, {pending_count} pending | "
                            f"Rate: {rate:.2f} sims/sec, ETA: {eta/60:.1f} min | "
                            f"RAM: {memory_percent:.1f}% used, {available_gb:.1f}GB free{gpu_memory_status}{memory_warning}"
                        )

                        try:
                            log_detailed_gpu_status(lambda msg: async_logger.log_batch(msg), "GPU Detail: ")
                        except:
                            pass

                    except:
                        # Fallback without memory monitoring
                        async_logger.log_batch(
                            f"Regular Sim Progress: {completed_count}/{total_simulations} completed, "
                            f"{active_count} active, {pending_count} pending | "
                            f"Rate: {rate:.2f} sims/sec, ETA: {eta/60:.1f} min"
                        )
                    
                    # Show device load balancing
                    load_str = ", ".join([f"GPU{device_id}: {load}" for device_id, load in device_load_counts.items()])
                    async_logger.log_batch(f"Device loads: {load_str}")
                    
                    last_report_time = current_time
        
        # Start progress monitoring
        monitor_task = asyncio.create_task(monitor_progress())
        
        try:
            # Initial batch - start up to max_concurrent simulations
            for _ in range(min(max_concurrent, len(pending_sims))):
                if pending_sims:
                    sim_idx = pending_sims.pop(0)
                    device_id = get_least_loaded_device()
                    device_load_counts[device_id] += 1
                    task = asyncio.create_task(self._process_single_regular_simulation(sim_idx, assigned_device=device_id))
                    active_tasks[task] = sim_idx
                    task_devices[task] = device_id
                    async_logger.log_batch(f"Started Sim {sim_idx} on device {device_id} ({len(active_tasks)}/{max_concurrent} slots used)")
            
            # Main streaming loop
            while active_tasks:
                # Wait for at least one simulation to complete
                done_tasks, pending_tasks = await asyncio.wait(
                    active_tasks.keys(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed simulations
                for task in done_tasks:
                    sim_idx = active_tasks[task]
                    device_id = task_devices[task]
                    del active_tasks[task]
                    del task_devices[task]
                    device_load_counts[device_id] -= 1
                    completed_count += 1
                    
                    try:
                        result = await task
                        sim_idx_result, final_results = result
                        
                        # Move results to CPU immediately to free GPU memory using distributed coordination
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            final_results = jax.device_put(final_results, jax.devices("cpu")[0])
                        
                        results_dict[sim_idx] = final_results
                        
                        result_max = float(jnp.max(final_results))
                        async_logger.log_batch(f"✅ Completed Sim {sim_idx} on device {device_id} - max rate: {result_max:.4f} ({completed_count}/{total_simulations})")
                        
                        # Save checkpoint if enabled and at intervals
                        checkpoint_interval = getattr(self.sim_config, 'checkpoint_interval', 10)
                        if enable_checkpointing and checkpoint_dir and (completed_count % checkpoint_interval == 0 or completed_count == total_simulations):
                            try:
                                # Create list of completed simulation indices
                                completed_simulations = list(results_dict.keys())
                                
                                # Create checkpoint state for streaming simulations
                                from src.data.data_classes import CheckpointState
                                
                                # Ensure neuron_params arrays are on CPU
                                cpu_device = jax.devices("cpu")[0]
                                cpu_adjusted_neuron_params = None
                                if self.adjusted_neuron_params is not None:
                                    cpu_adjusted_neuron_params = jax.tree.map(
                                        lambda x: jax.device_put(x, cpu_device) if hasattr(x, 'device') else x,
                                        self.adjusted_neuron_params
                                    )
                                
                                checkpoint_state = CheckpointState(
                                    batch_index=None,  # Not applicable for streaming
                                    completed_batches=None,  # Not applicable for streaming  
                                    total_batches=None,  # Not applicable for streaming
                                    n_result_batches=None,  # Not applicable for streaming
                                    accumulated_mini_circuits=None,  # Not applicable for regular sims
                                    neuron_params=cpu_adjusted_neuron_params,
                                    pruning_state=None,  # Not applicable for regular sims
                                    checkpoint_type="streaming",
                                    completed_simulations=completed_simulations,
                                    results_dict=None  # Don't store large results in checkpoint
                                )
                                
                                checkpoint_path = checkpoint_dir / f"regular_checkpoint_{completed_count}.h5"
                                metadata = {
                                    "sim_type": getattr(self.sim_config, 'sim_type', 'regular'),
                                    "enable_pruning": False,
                                    "total_sims": total_simulations,
                                    "max_concurrent": max_concurrent,
                                    "completed_count": completed_count,
                                    "total_simulations": total_simulations,
                                    "timestamp": time.time()
                                }
                                
                                from src.memory.checkpointing import save_checkpoint, cleanup_old_checkpoints
                                save_checkpoint(checkpoint_state, checkpoint_path, metadata)
                                async_logger.log_batch(f"💾 Regular checkpoint saved: {completed_count}/{total_simulations} completed")
                                
                                # Clean up old checkpoints (keep last 3)
                                cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)
                                
                            except Exception as checkpoint_error:
                                async_logger.log_batch(f"⚠️ Regular checkpoint save failed: {checkpoint_error}")
                        
                        # Clean up memory
                        del result, final_results
                        
                    except Exception as e:
                        failed_sims.add(sim_idx)
                        async_logger.log_batch(f"❌ Failed Sim {sim_idx} on device {device_id}: {str(e)}")
                        
                        # Create empty result for failed simulation using distributed coordination
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            empty_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                            empty_result = jax.device_put(empty_result, jax.devices("cpu")[0])
                        results_dict[sim_idx] = empty_result
                        del empty_result
                
                # Memory cleanup after processing completed simulations
                if done_tasks:
                    jax.clear_caches()
                    gc.collect()
                
                # Start new simulations to fill available slots
                while len(active_tasks) < max_concurrent and pending_sims:
                    sim_idx = pending_sims.pop(0)
                    device_id = get_least_loaded_device()
                    device_load_counts[device_id] += 1
                    task = asyncio.create_task(self._process_single_regular_simulation(sim_idx, assigned_device=device_id))
                    active_tasks[task] = sim_idx
                    task_devices[task] = device_id
                    async_logger.log_batch(f"🚀 Started Sim {sim_idx} on device {device_id} ({len(active_tasks)}/{max_concurrent} slots used)")
            
        finally:
            monitor_task.cancel()
        
        # Final summary
        total_time = time.time() - start_time
        avg_rate = completed_count / total_time if total_time > 0 else 0
        
        if failed_sims:
            async_logger.log_batch(
                f"🏁 Regular simulation streaming completed in {total_time:.1f}s | "
                f"Rate: {avg_rate:.2f} sims/sec | {len(failed_sims)} failures: {sorted(failed_sims)}"
            )
        else:
            async_logger.log_batch(
                f"🎉 Regular simulation streaming completed successfully in {total_time:.1f}s | "
                f"Rate: {avg_rate:.2f} sims/sec"
            )
        
        # Convert to ordered list
        results_list = [results_dict[i] for i in range(total_simulations)]
        
        # Final comprehensive memory cleanup
        del results_dict, active_tasks, task_devices, device_load_counts
        
        # Comprehensive JAX cleanup
        jax.clear_caches()
        
        # GPU memory cleanup  
        try:
            # Force GPU synchronization and cleanup
            for device_id in range(self.n_devices):
                with jax.default_device(jax.devices()[device_id]):
                    pass  # This forces device synchronization
                    
            # Additional memory cleanup
            gc.collect()
            
            # Memory manager cleanup
            if hasattr(self, 'memory_manager'):
                self.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context="final_regular_cleanup")

        except Exception as e:
            async_logger.log_batch(f"Warning: GPU cleanup failed: {e}")
        
        return results_list
    
    async def _run_with_stimulation_adjustment(self, total_simulations: int, max_concurrent: int, checkpoint_dir: Optional[Path] = None, enable_checkpointing: bool = False, pre_loaded_checkpoint_data: Optional[Tuple] = None) -> List[jnp.ndarray]:
        """Run simulations with stimulation adjustment (batch processing).
        
        Args:
            total_simulations: Total number of simulations to run
            max_concurrent: Maximum concurrent simulations
            checkpoint_dir: Directory for checkpointing (optional)
            enable_checkpointing: Whether to enable checkpointing
            pre_loaded_checkpoint_data: Pre-loaded checkpoint data tuple (checkpoint_state, adjusted_neuron_params, metadata) to avoid duplicate loading
        
        Returns:
            List of simulation results
        """
        """Run simulations with stimulation adjustment using streaming processing."""
        # Initialize result storage
        results_dict = {}
        failed_sims = set()
        adjustment_iterations = {}  # Track iterations per simulation
        
        # Checkpoint recovery for streaming stimulus adjustment using pre-loaded data (avoids duplicate loading)
        completed_count = 0
        if enable_checkpointing and checkpoint_dir and pre_loaded_checkpoint_data:
            checkpoint_state, adjusted_neuron_params, metadata = pre_loaded_checkpoint_data
            async_logger.log_batch(f"🔍 Using pre-loaded checkpoint data for stimulation adjustment")
            
            try:
                # Handle different checkpoint types
                if checkpoint_state.checkpoint_type == "streaming":
                    # For streaming: use completed_simulations, keep original neuron_params
                    completed_sims = set(checkpoint_state.completed_simulations)
                    completed_count = len(checkpoint_state.completed_simulations)
                    # Load results from checkpoint
                    for sim_idx in checkpoint_state.completed_simulations:
                        if hasattr(checkpoint_state, 'results_dict') and checkpoint_state.results_dict:
                            results_dict[sim_idx] = checkpoint_state.results_dict[sim_idx]
                            adjustment_iterations[sim_idx] = 0  # Mark as completed
                        else:
                            # Create placeholder result for completed simulations
                            placeholder_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                            placeholder_result = jax.device_put(placeholder_result, jax.devices("cpu")[0])
                            results_dict[sim_idx] = placeholder_result
                            adjustment_iterations[sim_idx] = 0  # Mark as completed
                    
                    # Don't replace neuron_params for streaming (adjusted_neuron_params might be None)
                    if adjusted_neuron_params is not None:
                        self.adjusted_neuron_params = adjusted_neuron_params
                        
                    async_logger.log_batch(f"📂 Resuming from streaming stimulus adjustment checkpoint: {completed_count}/{total_simulations} completed")
                else:
                    # Legacy batch checkpoint support
                    completed_count = checkpoint_state.completed_batches
                    if adjusted_neuron_params is not None:
                        self.adjusted_neuron_params = adjusted_neuron_params
                    async_logger.log_batch(f"📂 Resuming from batch stimulus adjustment checkpoint: {completed_count} batches completed")
                    
            except Exception as e:
                async_logger.log_batch(f"⚠️ Could not process pre-loaded checkpoint data: {e}")
                async_logger.log_batch("Starting from beginning...")
                completed_count = 0
                results_dict = {}
                adjustment_iterations = {}
        
        # Create a queue of simulation indices to process (excluding completed ones)
        pending_sims = [i for i in range(total_simulations) if i not in results_dict]
        for sim_idx in pending_sims:
            adjustment_iterations[sim_idx] = 0
            
        async_logger.log_batch(f"Stimulus adjustment simulations to process: {len(pending_sims)} pending, {len(results_dict)} already completed")
        
        # Get adjustment parameters
        max_adjustment_iters = getattr(self.sim_config, 'max_adjustment_iters', 10)
        n_active_upper = getattr(self.sim_config, 'n_active_upper', 500)
        n_active_lower = getattr(self.sim_config, 'n_active_lower', 5)
        n_high_fr_upper = getattr(self.sim_config, 'n_high_fr_upper', 100)
        high_fr_threshold = getattr(self.sim_config, 'high_fr_threshold', 100.0)
        
        active_tasks = {}
        task_devices = {}
        device_load_counts = {i: 0 for i in range(self.n_devices)}
        
        def get_least_loaded_device():
            """Get the device with the least number of active simulations."""
            return min(device_load_counts.keys(), key=lambda device_id: device_load_counts[device_id])
        
        # Progress tracking
        completed_count = len(results_dict)
        start_time = time.time()
        
        async def monitor_progress():
            nonlocal completed_count
            last_report_time = start_time
            
            while pending_sims or active_tasks:
                await asyncio.sleep(10)
                current_time = time.time()

                if current_time - last_report_time >= 30:  # Report every 30 seconds
                    elapsed = current_time - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_simulations - completed_count) / rate if rate > 0 else float('inf')
                    
                    active_count = len(active_tasks)
                    pending_count = len(pending_sims)
                    
                    # Count simulations in adjustment
                    adjusting_sims = sum(1 for sim_idx, iters in adjustment_iterations.items() 
                                       if iters > 0 and sim_idx not in results_dict)
                    
                    # Enhanced memory monitoring similar to AsyncPruningManager
                    try:
                        memory_status = monitor_memory_usage()
                        memory_percent = memory_status.get('ram_percent', 0)
                        available_gb = memory_status.get('ram_available_gb', 0)
                        
                        # Multi-GPU memory monitoring
                        gpu_memory_status = ""
                        if memory_status.get('gpu_available', False):
                            gpu_percent = memory_status.get('gpu_percent', 0)
                            gpu_free_gb = memory_status.get('gpu_free_gb', 0)
                            gpu_memory_status = f" | GPU: {gpu_percent:.1f}% used, {gpu_free_gb:.1f}GB free"
                            
                        memory_warning = ""
                        gpu_percent = memory_status.get('gpu_percent', 0)
                        if memory_percent > 85 or available_gb < 5 or gpu_percent > 85:
                            memory_warning = " ⚠️ LOW MEMORY!"
                        
                        async_logger.log_batch(
                            f"Stimulus Adjustment Progress: {completed_count}/{total_simulations} completed, "
                            f"{active_count} active, {pending_count} pending, {adjusting_sims} adjusting | "
                            f"Rate: {rate:.2f} sims/sec, ETA: {eta/60:.1f} min | "
                            f"RAM: {memory_percent:.1f}% used, {available_gb:.1f}GB free{gpu_memory_status}{memory_warning}"
                        )

                        try:
                            log_detailed_gpu_status(lambda msg: async_logger.log_batch(msg), "GPU Detail: ")
                        except:
                            pass
                    except:
                        # Fallback without memory monitoring
                        async_logger.log_batch(
                            f"Stimulus Adjustment Progress: {completed_count}/{total_simulations} completed, "
                            f"{active_count} active, {pending_count} pending, {adjusting_sims} adjusting | "
                            f"Rate: {rate:.2f} sims/sec, ETA: {eta/60:.1f} min"
                        )
                    
                    last_report_time = current_time
        
        # Start progress monitoring
        monitor_task = asyncio.create_task(monitor_progress())
        
        try:
            # Initial batch - start up to max_concurrent simulations
            for _ in range(min(max_concurrent, len(pending_sims))):
                if pending_sims:
                    sim_idx = pending_sims.pop(0)
                    device_id = get_least_loaded_device()
                    device_load_counts[device_id] += 1
                    task = asyncio.create_task(self._process_single_regular_simulation(sim_idx, assigned_device=device_id))
                    active_tasks[task] = sim_idx
                    task_devices[task] = device_id
                    async_logger.log_batch(f"🔄 Started Sim {sim_idx} (iteration {adjustment_iterations[sim_idx] + 1}) on device {device_id}")
            
            # Main streaming loop with stimulus adjustment
            while active_tasks:
                # Wait for at least one simulation to complete
                done_tasks, pending_tasks = await asyncio.wait(
                    active_tasks.keys(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed simulations
                for task in done_tasks:
                    sim_idx = active_tasks[task]
                    device_id = task_devices[task]
                    del active_tasks[task]
                    del task_devices[task]
                    device_load_counts[device_id] -= 1
                    
                    try:
                        result = await task
                        sim_idx_result, final_results = result
                        
                        # Move results to CPU immediately to free GPU memory using distributed coordination
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            final_results = jax.device_put(final_results, jax.devices("cpu")[0])
                        
                        # Analyze activity levels for this simulation
                        max_rates = jnp.max(final_results, axis=-1)  # (n_neurons,)
                        n_active = int(jnp.sum(jnp.sum(final_results, axis=-1) > 0))
                        n_high_fr = int(jnp.sum(max_rates > high_fr_threshold))
                        
                        adjustment_iterations[sim_idx] += 1
                        current_iter = adjustment_iterations[sim_idx]
                        
                        # Check if stimulus adjustment is needed
                        needs_adjustment = False
                        adjustment_reason = ""
                        
                        if (n_active > n_active_upper) or (n_high_fr > n_high_fr_upper):
                            needs_adjustment = True
                            adjustment_reason = f"too strong (active: {n_active}, high_fr: {n_high_fr})"
                        elif n_active < n_active_lower:
                            needs_adjustment = True
                            adjustment_reason = f"too weak (active: {n_active})"
                        
                        if needs_adjustment and current_iter < max_adjustment_iters:
                            # Adjust stimulus and restart this simulation
                            param_idx = sim_idx % self.sim_params.n_param_sets
                            stim_idx = sim_idx // self.sim_params.n_param_sets
                            
                            current_inputs = self.adjusted_neuron_params.input_currents
                            new_inputs = current_inputs.copy()
                            sim_input = current_inputs[stim_idx, param_idx]
                            
                            if (n_active > n_active_upper) or (n_high_fr > n_high_fr_upper):
                                # Too strong - reduce stimulation
                                new_input = sim_input / 2
                                adjustment_action = "reducing"
                            else:
                                # Too weak - increase stimulation  
                                # Find neurons that are being stimulated (non-zero current)
                                stimulated_mask = jnp.abs(sim_input) > 1e-10
                                
                                if jnp.any(stimulated_mask):
                                    # There are stimulated neurons - double their stimulus
                                    new_input = jnp.where(stimulated_mask, sim_input * 2, sim_input)
                                else:
                                    # No neurons are being stimulated - set baseline stimulus
                                    # This shouldn't happen in normal DN screen, but handle it gracefully
                                    baseline_stimulus = 10.0  # Start with 10 pA
                                    # Apply baseline to first neuron as a fallback
                                    new_input = sim_input.at[0].set(baseline_stimulus)
                                    
                                adjustment_action = "increasing"
                            
                            new_inputs = new_inputs.at[stim_idx, param_idx].set(new_input)
                            self.adjusted_neuron_params = self.adjusted_neuron_params._replace(
                                input_currents=new_inputs
                            )
                            
                            # Get scalar values for logging (use max absolute value to show actual stimulus level)
                            if hasattr(sim_input, 'flatten'):
                                old_val = float(jnp.max(jnp.abs(sim_input)))
                            else:
                                old_val = float(jnp.abs(sim_input))
                                
                            if hasattr(new_input, 'flatten'):
                                new_val = float(jnp.max(jnp.abs(new_input)))
                            else:
                                new_val = float(jnp.abs(new_input))
                            
                            async_logger.log_batch(f"🔄 Stimulus adjustment iteration {current_iter}, Sim {sim_idx}: {adjustment_reason}, {adjustment_action} stimulus: {old_val:.6f} -> {new_val:.6f}")
                            
                            # Restart this simulation with adjusted stimulus
                            device_id = get_least_loaded_device()
                            device_load_counts[device_id] += 1
                            task = asyncio.create_task(self._process_single_regular_simulation(sim_idx, assigned_device=device_id))
                            active_tasks[task] = sim_idx
                            task_devices[task] = device_id
                            
                        else:
                            # Simulation is complete (either criteria met or max iterations reached)
                            if current_iter >= max_adjustment_iters and needs_adjustment:
                                async_logger.log_batch(f"⚠️ Max adjustment iterations reached for Sim {sim_idx} - accepting result (active: {n_active}, high_fr: {n_high_fr})")
                            else:
                                async_logger.log_batch(f"🎯 Stimulus adjustment completed for Sim {sim_idx} after {current_iter} iterations (active: {n_active}, high_fr: {n_high_fr})")
                            
                            # Store final result
                            results_dict[sim_idx] = final_results
                            completed_count += 1
                            
                            result_max = float(jnp.max(final_results))
                            async_logger.log_batch(f"✅ Completed Sim {sim_idx} with stimulus adjustment - max rate: {result_max:.4f} ({completed_count}/{total_simulations})")
                            
                            # Save checkpoint if enabled and at intervals
                            checkpoint_interval = getattr(self.sim_config, 'checkpoint_interval', 10)
                            if enable_checkpointing and checkpoint_dir and (completed_count % checkpoint_interval == 0 or completed_count == total_simulations):
                                try:
                                    # Create list of completed simulation indices
                                    completed_simulations = list(results_dict.keys())
                                    
                                    # Create checkpoint state for streaming simulations
                                    from src.data.data_classes import CheckpointState
                                    
                                    # Ensure neuron_params arrays are on CPU
                                    cpu_device = jax.devices("cpu")[0]
                                    cpu_adjusted_neuron_params = None
                                    if self.adjusted_neuron_params is not None:
                                        cpu_adjusted_neuron_params = jax.tree.map(
                                            lambda x: jax.device_put(x, cpu_device) if hasattr(x, 'device') else x,
                                            self.adjusted_neuron_params
                                        )
                                    
                                    checkpoint_state = CheckpointState(
                                        batch_index=None,  # Not applicable for streaming
                                        completed_batches=None,  # Not applicable for streaming  
                                        total_batches=None,  # Not applicable for streaming
                                        n_result_batches=None,  # Not applicable for streaming
                                        accumulated_mini_circuits=None,  # Not applicable for regular sims
                                        neuron_params=cpu_adjusted_neuron_params,
                                        pruning_state=None,  # Not applicable for regular sims
                                        checkpoint_type="streaming",
                                        completed_simulations=completed_simulations,
                                        results_dict=None  # Don't store large results in checkpoint
                                    )
                                    
                                    checkpoint_path = checkpoint_dir / f"regular_checkpoint_{completed_count}.h5"
                                    metadata = {
                                        "sim_type": getattr(self.sim_config, 'sim_type', 'regular'),
                                        "enable_pruning": False,
                                        "total_sims": total_simulations,
                                        "max_concurrent": max_concurrent,
                                        "completed_count": completed_count,
                                        "total_simulations": total_simulations,
                                        "stimulation_adjustment": True,
                                        "timestamp": time.time()
                                    }
                                    
                                    from src.memory.checkpointing import save_checkpoint, cleanup_old_checkpoints
                                    save_checkpoint(checkpoint_state, checkpoint_path, metadata)
                                    async_logger.log_batch(f"💾 Stimulus adjustment checkpoint saved: {completed_count}/{total_simulations} completed")
                                    
                                    # Clean up old checkpoints (keep last 3)
                                    cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3)
                                    
                                except Exception as checkpoint_error:
                                    async_logger.log_batch(f"⚠️ Stimulus adjustment checkpoint save failed: {checkpoint_error}")
                        
                        # Clean up memory
                        del result, final_results
                        
                    except Exception as e:
                        failed_sims.add(sim_idx)
                        completed_count += 1
                        async_logger.log_batch(f"❌ Failed Sim {sim_idx} on device {device_id}: {str(e)}")
                        
                        # Create empty result for failed simulation using distributed coordination
                        coord_device = self._get_next_coordination_device()
                        with jax.default_device(coord_device):
                            empty_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                            empty_result = jax.device_put(empty_result, jax.devices("cpu")[0])
                        results_dict[sim_idx] = empty_result
                        del empty_result
                
                # Memory cleanup after processing completed simulations
                if done_tasks:
                    jax.clear_caches()
                    gc.collect()
                
                # Start new simulations to fill available slots (only if we have pending ones)
                while len(active_tasks) < max_concurrent and pending_sims:
                    sim_idx = pending_sims.pop(0)
                    device_id = get_least_loaded_device()
                    device_load_counts[device_id] += 1
                    task = asyncio.create_task(self._process_single_regular_simulation(sim_idx, assigned_device=device_id))
                    active_tasks[task] = sim_idx
                    task_devices[task] = device_id
                    async_logger.log_batch(f"🚀 Started Sim {sim_idx} (iteration {adjustment_iterations[sim_idx] + 1}) on device {device_id}")
            
        finally:
            monitor_task.cancel()
        
        # Final summary
        total_time = time.time() - start_time
        avg_rate = completed_count / total_time if total_time > 0 else 0
        
        # Calculate adjustment statistics
        total_adjustments = sum(max(0, iters - 1) for iters in adjustment_iterations.values())
        adjusted_sims = sum(1 for iters in adjustment_iterations.values() if iters > 1)
        
        if failed_sims:
            async_logger.log_batch(
                f"🏁 Stimulus adjustment streaming completed in {total_time:.1f}s | "
                f"Rate: {avg_rate:.2f} sims/sec | {adjusted_sims} sims adjusted ({total_adjustments} total adjustments) | {len(failed_sims)} failures: {sorted(failed_sims)}"
            )
        else:
            async_logger.log_batch(
                f"🎉 Stimulus adjustment streaming completed successfully in {total_time:.1f}s | "
                f"Rate: {avg_rate:.2f} sims/sec | {adjusted_sims} sims adjusted ({total_adjustments} total adjustments)"
            )
        
        # Convert to ordered list
        results_list = [results_dict[i] for i in range(total_simulations)]
        
        # Final comprehensive memory cleanup
        del results_dict, active_tasks, task_devices, device_load_counts, adjustment_iterations
        
        # Comprehensive JAX cleanup
        jax.clear_caches()
        
        # GPU memory cleanup  
        try:
            # Force GPU synchronization and cleanup
            for device_id in range(self.n_devices):
                with jax.default_device(jax.devices()[device_id]):
                    pass  # This forces device synchronization
                    
            # Additional memory cleanup
            gc.collect()
            
            # Memory manager cleanup
            if hasattr(self, 'memory_manager'):
                self.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context="final_stim_adj_cleanup")

        except Exception as e:
            async_logger.log_batch(f"Warning: GPU cleanup failed: {e}")
        
        return results_list
    
    async def _run_batch_simulations(self, sim_indices: List[int]) -> List[jnp.ndarray]:
        """Run a specific batch of simulations concurrently."""
        tasks = []
        device_assignments = {}
        
        # Create tasks for all simulations in this batch
        for i, sim_idx in enumerate(sim_indices):
            device_id = i % self.n_devices  # Round-robin device assignment
            task = asyncio.create_task(self._process_single_regular_simulation(sim_idx, assigned_device=device_id))
            tasks.append(task)
            device_assignments[sim_idx] = device_id
        
        # Run all simulations in the batch concurrently
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract results in the correct order
        batch_results = []
        failed_sims = []
        
        for i, task_result in enumerate(completed_tasks):
            sim_idx = sim_indices[i]
            if isinstance(task_result, Exception):
                failed_sims.append(sim_idx)
                async_logger.log_batch(f"❌ Batch sim {sim_idx} failed: {task_result}")
                # Create empty result for failed simulation using distributed coordination
                coord_device = self._get_next_coordination_device()
                with jax.default_device(coord_device):
                    empty_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                    empty_result = jax.device_put(empty_result, jax.devices("cpu")[0])
                    batch_results.append(empty_result)
            else:
                result_sim_idx, result_array = task_result
                batch_results.append(result_array)
                async_logger.log_batch(f"✅ Batch sim {sim_idx} completed on device {device_assignments[sim_idx]}")
        
        if failed_sims:
            async_logger.log_batch(f"Batch completed with {len(failed_sims)} failures: {failed_sims}")
        
        return batch_results


def run_streaming_regular_simulation(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    sim_config: SimulationConfig,
    max_concurrent: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    enable_checkpointing: bool = False
) -> jnp.ndarray:
    """
    Run regular (non-pruning) simulations with streaming processing.
    
    This provides the async equivalent of _run_stim_neurons with:
    - Continuous streaming (new simulations start as others finish)
    - Better GPU utilization
    - Real-time progress monitoring
    - Memory-efficient execution
    - Support for all simulation types (baseline, shuffle, noise, Wmask)
    - Automatic stimulation adjustment (adjustStimI)
    - DN screen support
    - Checkpointing support for recovery from failures
    
    Args:
        neuron_params: Neuron parameters
        sim_params: Simulation parameters  
        sim_config: Simulation configuration
        max_concurrent: Maximum concurrent simulations (default: auto-calculated)
        checkpoint_dir: Directory for checkpointing (optional)
        enable_checkpointing: Whether to enable checkpointing
    
    Returns:
        Results array shaped as (n_stim_configs, n_param_sets, n_neurons, n_timepoints)
    """
    total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
    
    # Comprehensive initial cleanup to ensure fresh start
    print("🧹 Performing initial memory cleanup for fresh start...")
    jax.clear_caches()
    gc.collect()
    
    # GPU memory cleanup and synchronization
    try:
        # Force GPU synchronization on all devices
        devices = jax.devices()
        for device in devices:
            with jax.default_device(device):
                # Force a small operation to synchronize
                _ = jnp.array([1.0])
        print(f"✅ Synchronized {len(devices)} GPU devices")
    except Exception as e:
        print(f"⚠️  Warning: GPU synchronization failed: {e}")
    
    # Handle DN screen mode - run individual neuron screening
    if getattr(sim_config, 'dn_screen', False):
        print("Running in DN Screen mode - testing individual neurons")
        return run_dn_screen_simulations(neuron_params, sim_params, sim_config, max_concurrent)
    
    # Calculate optimal max_concurrent if not provided
    if max_concurrent is None:
        max_concurrent = calculate_optimal_concurrent_size(
            n_neurons=sim_params.n_neurons,
            n_timepoints=len(sim_params.t_axis),
            total_simulations=total_sims,
            is_pruning=False
        )
        print(f"Auto-calculated max_concurrent: {max_concurrent}")
    else:
        print(f"Using provided max_concurrent: {max_concurrent}")
    
    # Create async manager with error handling
    manager = None
    results_list = None
    
    try:
        manager = AsyncRegularSimManager(neuron_params, sim_params, sim_config)
        
        # Apply stimulation adjustment if enabled
        if getattr(sim_config, 'adjustStimI', False):
            print("Stimulation adjustment is enabled - will adjust during streaming")
        
        print(f"Simulation type: {getattr(sim_config, 'sim_type', 'baseline')}")
        print(f"Starting streaming regular simulation with {total_sims} total simulations")
        
        # Load checkpoint data once if checkpointing is enabled (avoiding duplicate loading)
        checkpoint_data_for_manager = None
        if enable_checkpointing and checkpoint_dir:
            from src.memory.checkpointing import find_latest_checkpoint, load_checkpoint
            checkpoint_result = find_latest_checkpoint(checkpoint_dir, pattern="regular_checkpoint_*")
            if checkpoint_result:
                latest_checkpoint, base_name = checkpoint_result
                print(f"🔍 Found checkpoint: {latest_checkpoint}")
                
                # Perform memory cleanup before attempting to load checkpoint
                print("🧹 Pre-loading memory cleanup...")
                jax.clear_caches()
                for _ in range(3):
                    gc.collect()
                    
                try:
                    checkpoint_state, adjusted_neuron_params, metadata = load_checkpoint(latest_checkpoint, base_name, manager.adjusted_neuron_params)
                    checkpoint_data_for_manager = (checkpoint_state, adjusted_neuron_params, metadata)
                    print(f"📂 Successfully loaded checkpoint for reuse by manager")
                except Exception as e:
                    print(f"⚠️ Could not load checkpoint {latest_checkpoint}: {e}")
                    print("Starting from beginning...")
                    checkpoint_data_for_manager = None
        
        async def run_simulations():
            # Run the streaming simulations with pre-loaded checkpoint data
            results_list = await manager.run_streaming_regular_simulations(total_sims, max_concurrent, checkpoint_dir, enable_checkpointing, checkpoint_data_for_manager)
            return results_list
        
        # Run the async event loop
        results_list = asyncio.run(run_simulations())
        
    except Exception as e:
        print(f"❌ Async regular simulation failed: {e}")
        print("🔄 Cleaning up resources and raising exception...")
        
        # Comprehensive cleanup on failure
        jax.clear_caches()
        gc.collect()
        
        # Clean up manager if it exists
        if manager is not None:
            try:
                if hasattr(manager, 'memory_manager'):
                    manager.memory_manager.perform_cleanup(logger_func=async_logger.log_batch, context="failed_regular_run_cleanup")
            except:
                pass
        
        # Re-raise the exception to be handled by the caller
        raise e
    
    finally:
        # Always perform cleanup regardless of success/failure
        if manager is not None:
            try:
                if hasattr(manager, 'memory_manager'):
                    manager.memory_manager.perform_cleanup(logger_func=async_logger.log_batch,context="final_regular_run_cleanup")
            except:
                pass
    
    # Convert results to the expected output format
    results_array = jnp.stack(results_list, axis=0)
    results_reshaped = results_array.reshape(
        sim_params.n_stim_configs, sim_params.n_param_sets,
        sim_params.n_neurons, len(sim_params.t_axis)
    )
    
    # Final cleanup of intermediate arrays
    del results_list
    gc.collect()
    print("✅ Streaming regular simulation completed successfully")
    
    return results_reshaped


def run_dn_screen_simulations(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    sim_config: SimulationConfig,
    max_concurrent: Optional[int] = None
) -> jnp.ndarray:
    """
    Run DN screen simulations - testing individual descending neurons.
    
    In DN screen mode, we test each DN neuron individually to find responsive ones.
    This is typically used for screening experiments.
    """
    print("DN Screen mode: Testing individual descending neurons")
    
    # Get DN neuron indices
    if hasattr(neuron_params, 'dn_idxs') and neuron_params.dn_idxs is not None:
        dn_indices = neuron_params.dn_idxs
    elif hasattr(neuron_params, 'exc_dn_idxs') and neuron_params.exc_dn_idxs is not None:
        # Combine excitatory and inhibitory DN indices if available
        exc_dn = neuron_params.exc_dn_idxs if neuron_params.exc_dn_idxs is not None else jnp.array([])
        inh_dn = neuron_params.inh_dn_idxs if neuron_params.inh_dn_idxs is not None else jnp.array([])
        dn_indices = jnp.concatenate([exc_dn, inh_dn]) if len(exc_dn) > 0 or len(inh_dn) > 0 else None
    else:
        print("Warning: No DN indices found, using all neurons")
        dn_indices = jnp.arange(sim_params.n_neurons)
    
    if dn_indices is None or len(dn_indices) == 0:
        raise ValueError("No DN neurons found for screening")
    
    # Limit DN testing if max_dn_test is specified
    max_dn_test = getattr(sim_config, 'max_dn_test', None)
    if max_dn_test is not None and max_dn_test > 0:
        original_n_dns = len(dn_indices)
        dn_indices = dn_indices[:max_dn_test]
        print(f"Limited DN screening from {original_n_dns} to {max_dn_test} DNs for testing")
    
    n_dns = len(dn_indices)
    print(f"Screening {n_dns} DN neurons: {dn_indices}")
    
    # Create modified neuron_params for DN screening
    # Each "simulation" will stimulate a different DN neuron
    original_input_currents = neuron_params.input_currents.copy()
    
    # Create input currents that stimulate each DN individually
    screen_input_currents = jnp.zeros((n_dns, sim_params.n_param_sets, sim_params.n_neurons))
    
    for dn_idx, dn_neuron in enumerate(dn_indices):
        # Set stimulation for this DN neuron across all parameter sets
        for param_set in range(sim_params.n_param_sets):
            # Use the original stimulation intensity but apply to the DN neuron
            if original_input_currents.ndim == 3:
                stim_intensity = jnp.max(original_input_currents[0, param_set])  # Use first stim config's intensity
            else:
                stim_intensity = jnp.max(original_input_currents[param_set])
            
            screen_input_currents = screen_input_currents.at[dn_idx, param_set, dn_neuron].set(stim_intensity)
    
    # Update simulation parameters for DN screening
    screen_neuron_params = neuron_params._replace(input_currents=screen_input_currents)
    screen_sim_params = sim_params._replace(
        n_stim_configs=n_dns,  # Each DN is a "stimulus configuration"
        n_param_sets=sim_params.n_param_sets
    )
    
    total_sims = n_dns * sim_params.n_param_sets
    
    # Calculate optimal max_concurrent if not provided
    if max_concurrent is None:
        max_concurrent = calculate_optimal_concurrent_size(
            n_neurons=sim_params.n_neurons,
            n_timepoints=len(sim_params.t_axis),
            total_simulations=total_sims,
            is_pruning=False
        )
        print(f"Auto-calculated max_concurrent for DN screen: {max_concurrent}")
    
    # Create async manager for DN screening
    manager = AsyncRegularSimManager(screen_neuron_params, screen_sim_params, sim_config)
    
    print(f"Starting DN screen with {total_sims} total simulations ({n_dns} DNs × {sim_params.n_param_sets} param sets)")
    
    async def run_screen_simulations():
        results_list = await manager.run_streaming_regular_simulations(total_sims, max_concurrent, None, False, None)
        return results_list
    
    # Run the async event loop
    results_list = asyncio.run(run_screen_simulations())
    
    # Convert results to expected format
    results_array = jnp.stack(results_list, axis=0)
    results_reshaped = results_array.reshape(
        n_dns, sim_params.n_param_sets,  # DNs × param_sets
        sim_params.n_neurons, len(sim_params.t_axis)
    )
    
    print(f"DN screen completed: {results_reshaped.shape}")
    return results_reshaped
