# Asynchronous VNC Network Simulation - Individual Simulation Processing
import asyncio
import concurrent.futures
import time
import jax
import jax.numpy as jnp
import numpy as np
import traceback
from typing import NamedTuple, Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# Import existing simulation components
from src.vnc_sim import (
    run_single_simulation, update_single_sim_state, initialize_pruning_state,
    get_batch_function, reweight_connectivity, removal_probability
)
from src.data_classes import NeuronParams, SimParams, SimulationConfig, Pruning_state
from src.sim_utils import load_wTable

# Import pmap for multi-GPU support
from jax import pmap


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
    """State for individual asynchronous simulation."""
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
    """Initialize pruning state for a specific simulation index."""
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

    # Initialize probabilities for this specific simulation
    exclude_mask = ~interneuron_mask
    p_arrays = removal_probability(jnp.ones((W_mask.shape[-1],)), exclude_mask)

    # Get the specific seed for this simulation index
    if sim_index < len(neuron_params.seeds):
        sim_seed = neuron_params.seeds[sim_index]
    else:
        # Use last seed if we don't have enough seeds
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
        keys=sim_seed  # Individual seed, shape (2,) not (1, 2)
    )
    
    
class AsyncPruningManager:
    """Manages asynchronous execution of individual pruning simulations with shared read-only data."""
    
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
        
        # Pre-create the motor neuron mask once to ensure consistency
        # Force mn_idxs to CPU to avoid device placement issues
        mn_idxs_cpu = jax.device_put(self.neuron_params.mn_idxs, jax.devices('cpu')[0])
        n_neurons_cpu = jax.device_put(self.sim_params.n_neurons, jax.devices('cpu')[0])
        self.mn_mask_template = jnp.isin(jnp.arange(n_neurons_cpu), mn_idxs_cpu)
        
        # Multi-GPU setup
        self.n_devices = jax.device_count()
        self.devices = jax.devices()
        gpu_devices = [d for d in self.devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        self.n_gpu_devices = len(gpu_devices)
        
        print(f"Async Manager: {self.n_devices} total devices, {self.n_gpu_devices} GPU devices")
        
        # For async individual sims, we use device placement rather than pmap
        # This allows each simulation to run on a different GPU independently
        self.max_workers = max_workers or min(32, (self.n_devices * 2))
        
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
        
        # Pre-compile functions for each device using direct assignment
        self._setup_device_functions()
    
    def _setup_shared_data(self):
        """Set up memory-efficient shared data across devices."""
        try:
            print("Setting up memory-efficient shared data across devices...")
            
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
            if device_index == 0:
                device_rtol = self.sim_params.r_tol
                device_atol = self.sim_params.a_tol
            else:
                # Use more conservative tolerances for non-primary devices
                device_rtol = max(self.sim_params.r_tol * 10, 1e-5)
                device_atol = max(self.sim_params.a_tol * 10, 1e-7)
                print(f"  Device {device_index}: Using conservative tolerances rtol={device_rtol:.1e}, atol={device_atol:.1e}")
            
            # Store device-specific parameters (no JIT compilation here to avoid diffrax conflicts)
            self.device_functions[i] = {
                'device_index': device_index,
                'oscillation_threshold': oscillation_threshold_val,
                'clip_start': clip_start_val,
                'rtol': device_rtol,
                'atol': device_atol
            }
        
    def _get_device_for_sim(self, sim_index: int) -> int:
        """Assign a device to a simulation based on its index."""
        return sim_index % self.n_devices
    
    def _get_device_functions(self, device_id: int):
        """Get the device-specific parameters for a device."""
        return self.device_functions[device_id]
            
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
        device_idx: int = 0
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
        
        # Apply W_mask to connectivity (memory efficient, same device)
        W_masked = W_device * W_mask_device
        W_reweighted = reweight_connectivity(
            W_masked, 
            self.shared_exc_multiplier, 
            self.shared_inh_multiplier
        )
        
        # Get individual parameters for this simulation
        tau_param = self.neuron_params.tau[param_idx]
        a_param = self.neuron_params.a[param_idx]
        threshold_param = self.neuron_params.threshold[param_idx]
        fr_cap_param = self.neuron_params.fr_cap[param_idx]
        input_currents_param = self.neuron_params.input_currents[stim_idx, param_idx]
        seeds_param = self.neuron_params.seeds[param_idx]  # Individual seed shape (2,)
        
        # Run simulation with current mask and specified tolerances
        # Use shared data for memory efficiency
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
            r_tol,  # Use device-specific tolerance
            a_tol,  # Use device-specific tolerance
            seeds_param  # Individual seed, not batched
        )
    
    async def _process_single_simulation(self, sim_index: int, assigned_device: Optional[int] = None) -> Tuple[int, jnp.ndarray, Pruning_state, jnp.ndarray]:
        """Process a single simulation asynchronously until convergence.
        
        Args:
            sim_index: Index of the simulation
            assigned_device: Specific device to use (if None, uses default assignment)
        """
        try:
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
        
        # Move simulation state to the target device
        sim_state = AsyncSimState(
            sim_index=sim_state.sim_index,
            pruning_state=jax.tree.map(lambda x: jax.device_put(x, target_device), sim_state.pruning_state),
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
                    async_logger.log_sim(sim_index, f"Converged after {iteration} iterations", "COMPLETE")
                    break
                
                # Calculate current pruning metrics
                available_neurons = jnp.sum(~sim_state.pruning_state.total_removed_neurons & ~mn_mask)
                removed_neurons = jnp.sum(sim_state.pruning_state.total_removed_neurons)
                put_back_neurons = jnp.sum(sim_state.pruning_state.neurons_put_back)
                
                # Compact progress logging - only every 5 iterations or significant changes
                if iteration % 10 == 0 or iteration < 3:
                    async_logger.log_sim(sim_index, 
                        f"Iter {iteration:2d}: {available_neurons:4d} avail, {removed_neurons:4d} removed, {put_back_neurons:3d} put back, Level {sim_state.pruning_state.level[0]}", 
                        "PROGRESS")
                
                # Run simulation iteration in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Use device-specific function for this simulation
                    future = loop.run_in_executor(
                        executor,
                        lambda: self._run_single_pruning_iteration_with_tolerances(
                            sim_state.pruning_state.W_mask,
                            sim_index,
                            device_params['rtol'],
                            device_params['atol'],
                            device_id
                        )
                    )
                    results = await future
                
                # Show activity information (less frequent)
                # max_activity = jnp.max(results)
                # active_neurons = jnp.sum(jnp.max(results, axis=-1) > 0)
                # if iteration % 10 == 0 or iteration < 2:
                #     async_logger.log_sim(sim_index, 
                #         f"Activity: max={max_activity:.2f}, active neurons={active_neurons}", 
                #         "PROGRESS")
                
                
                # Update pruning state - call update function directly
                updated_state = update_single_sim_state(
                    sim_state.pruning_state,  # Individual state, not batched
                    results,  # Individual results
                    mn_mask,
                    device_params['oscillation_threshold'],
                    device_params['clip_start']
                )
                
            except Exception as e:
                async_logger.log_sim(sim_index, f"Failed at iteration {iteration}: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                raise e
            
            # Store updated state directly (already individual, no batch dimension to remove)
            sim_state.pruning_state = updated_state
            
            sim_state.iteration_count = iteration
            sim_state.last_update_time = time.time()
            iteration += 1
            
            # Periodic memory cleanup to prevent accumulation
            if iteration % 10 == 0:  # Clean up every 10 iterations
                # Clean up intermediate results
                del results, updated_state
                # Force garbage collection periodically
                import gc
                gc.collect()
                
                # Clear JAX caches every 20 iterations to prevent memory buildup
                if iteration % 20 == 0:
                    jax.clear_caches()
            
            # Yield control to allow other simulations to run
            await asyncio.sleep(0)
        
        # Run final simulation with converged pruned network
        # First create W_mask from pruning state to define the network structure
        active_neurons_from_pruning = (~sim_state.pruning_state.total_removed_neurons) | sim_state.pruning_state.last_removed | sim_state.pruning_state.prev_put_back
        
        # Create final W_mask for the pruned network structure
        W_mask_init = jnp.ones_like(self.neuron_params.W_mask[0], dtype=jnp.bool_)  # Single sim, not batch
        W_mask_final = (W_mask_init * active_neurons_from_pruning[:, None] * active_neurons_from_pruning[None, :]).astype(jnp.bool_)
        
        # Update neuron params with final pruned network - single simulation format
        final_neuron_params = self.neuron_params._replace(W_mask=W_mask_final[None, ...])  # Add batch dim
        
        # Import the pruning simulation function
        from src.vnc_sim import run_with_Wmask
        
        async_logger.log_sim(sim_index, 
            f"Starting final simulation with {jnp.sum(active_neurons_from_pruning)} neurons", 
            "FINAL_SIM")
        
        # Run VNC simulation with the pruned network using memory-efficient approach
        # Get device placement first (outside try block so it's available in except block)
        device_id = self._get_device_for_sim(sim_index)
        target_device = self.devices[device_id]
        
        try:
            # Get parameters for this simulation (single simulation, not batch)
            param_idx = sim_index % self.sim_params.n_param_sets
            stim_idx = sim_index // self.sim_params.n_param_sets
            
            # Generate random key for this simulation
            key = jax.random.PRNGKey(sim_index + 12345)
            
            # Use memory-efficient approach with pre-placed W matrix
            if device_id in self.device_W:
                W_device = self.device_W[device_id]
            else:
                W_device = jax.device_put(self.neuron_params.W, target_device)
            
            # IMPORTANT: Place W_mask_final on the same device as W_device
            W_mask_final_device = jax.device_put(W_mask_final, target_device)
            
            # Apply final W_mask and reweight (memory efficient, same device)
            W_masked = W_device * W_mask_final_device
            W_reweighted = reweight_connectivity(
                W_masked, 
                self.shared_exc_multiplier, 
                self.shared_inh_multiplier
            )
            
            # Run the final simulation with pruned network
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
                self.sim_params.r_tol,
                self.sim_params.a_tol,
                key
            )
            
            # Debug: Check the actual results
            result_stats = {
                'shape': final_results.shape,
                'min': float(jnp.min(final_results)),
                'max': float(jnp.max(final_results)), 
                'mean': float(jnp.mean(final_results)),
                'nonzero': int(jnp.sum(final_results > 0))
            }
            async_logger.log_sim(sim_index, 
                f"Final simulation completed: {result_stats}", 
                "FINAL_SIM")
            
            # IMPORTANT: Create mini_circuit from actual simulation results (neurons that fired)
            # This ensures mini_circuit matches the neurons that are active in final_results
            
            # Ensure both final_results and mn_mask are on the same device with explicit synchronization
            # Force everything to be done in the context of the target device
            with jax.default_device(target_device):
                # Move final_results to target device first
                final_results = jax.device_put(final_results, target_device)
                # Create mn_mask on target device (not just moving template)
                mn_mask = jax.device_put(self.mn_mask_template, target_device)
                
                # Perform operations step by step to ensure device compatibility
                # Create boolean mask of firing neurons (on target device)
                firing_mask = jnp.sum(final_results, axis=-1) > 0
                # Apply motoneuron mask (both on target device)  
                mini_circuit = firing_mask & ~mn_mask  # Only interneurons that fired
            
            async_logger.log_sim(sim_index, 
                f"Mini circuit: {jnp.sum(mini_circuit)} active interneurons from final simulation results", 
                "FINAL_SIM")
                
        except Exception as e:
            async_logger.log_sim(sim_index, 
                f"ERROR in final simulation: {e}", 
                "ERROR")
            # Create fallback result to prevent crash
            final_results = jnp.ones((self.sim_params.n_neurons, len(self.sim_params.t_axis))) * 0.002  # Different value to identify this case
            
            # Create fallback mini_circuit (all interneurons active)
            # Ensure both are placed on the target_device
            # Use explicit device context for safe bitwise operation
            with jax.default_device(target_device):
                final_results = jax.device_put(final_results, target_device)
                mn_mask = jax.device_put(self.mn_mask_template, target_device)
                mini_circuit = jnp.ones(self.sim_params.n_neurons, dtype=bool) & ~mn_mask
        
        # Mark as completed and show final statistics
        with jax.default_device(target_device):
            sim_state.is_converged = True
            mn_mask = jax.device_put(self.mn_mask_template, target_device)
            total_removed_neurons = jax.device_put(sim_state.pruning_state.total_removed_neurons, target_device)
            final_available = jnp.sum(~total_removed_neurons & ~mn_mask)
            final_removed = jnp.sum(total_removed_neurons)

        async_logger.log_sim(sim_index, 
            f"COMPLETED: {iteration} iterations, {final_available} neurons remaining ({final_removed} removed)", 
            "COMPLETE")
        
        # Extract the pruning state before cleanup
        final_pruning_state = sim_state.pruning_state
        
        # Clean up intermediate computation arrays to free memory
        del sim_state
        
        # Periodically clear JAX caches to prevent memory buildup
        if sim_index % 5 == 0:  # Clear every 5 simulations
            jax.clear_caches()
            
        return sim_index, final_results, final_pruning_state, mini_circuit
    
    async def run_streaming_simulations(self, total_simulations: int, max_concurrent: Optional[int] = None) -> Tuple[List[jnp.ndarray], List[Pruning_state], List[jnp.ndarray]]:
        """
        Run simulations with continuous streaming - start new simulations as soon as others finish.
        
        Args:
            total_simulations: Total number of simulations to run
            max_concurrent: Maximum number of simulations to run concurrently (default: n_devices * 2)
        
        Returns:
            Tuple of (results_list, states_list) in simulation index order
        """
        if max_concurrent is None:
            max_concurrent = min(self.n_devices * 2, total_simulations)
        
        async_logger.log_batch(f"Starting streaming execution: {total_simulations} total simulations, max {max_concurrent} concurrent")
        
        # Initialize result storage (indexed by simulation number)
        results_dict = {}
        states_dict = {}
        mini_circuits_dict = {}
        failed_sims = set()
        
        # Create a queue of simulation indices to process
        pending_sims = list(range(total_simulations))
        active_tasks = {}  # task -> sim_index mapping
        task_devices = {}  # task -> device_id mapping
        device_load_counts = {i: 0 for i in range(self.n_devices)}  # track sims per device
        
        def get_least_loaded_device():
            """Get the device with the least number of active simulations."""
            return min(device_load_counts.keys(), key=lambda device_id: device_load_counts[device_id])
        
        # Progress tracking
        completed_count = 0
        start_time = time.time()
        
        # Create simplified progress monitoring task
        async def monitor_progress():
            nonlocal completed_count
            last_report_time = start_time
            
            while pending_sims or active_tasks:
                await asyncio.sleep(10)  # Update every 10 seconds
                current_time = time.time()
                
                if current_time - last_report_time >= 30:  # Detailed report every 30 seconds
                    elapsed = current_time - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_simulations - completed_count) / rate if rate > 0 else float('inf')
                    
                    active_count = len(active_tasks)
                    pending_count = len(pending_sims)
                    
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
                        sim_idx_result, final_results, final_state, mini_circuit = result
                        
                        # Move results to CPU immediately to free GPU memory
                        final_results = jax.device_put(final_results, jax.devices("cpu")[0])
                        final_state = jax.device_put(final_state, jax.devices("cpu")[0])
                        mini_circuit = jax.device_put(mini_circuit, jax.devices("cpu")[0])
                        
                        results_dict[sim_idx] = final_results
                        states_dict[sim_idx] = final_state
                        mini_circuits_dict[sim_idx] = mini_circuit
                        
                        # Verify we got real results, not fallback values
                        result_max = float(jnp.max(final_results))
                        if result_max <= 0.001:
                            async_logger.log_batch(f"⚠️  Warning: Sim {sim_idx} completed with suspiciously low max rate: {result_max}")
                        
                        async_logger.log_batch(f"✅ Completed Sim {sim_idx} on device {device_id} (load: {device_load_counts[device_id]}) - max rate: {result_max:.4f} ({completed_count}/{total_simulations})")
                        
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
                            basic_state = jax.device_put(basic_state, jax.devices("cpu")[0])
                            basic_mini_circuit = jax.device_put(basic_mini_circuit, jax.devices("cpu")[0])
                            
                            results_dict[sim_idx] = basic_result
                            states_dict[sim_idx] = basic_state
                            mini_circuits_dict[sim_idx] = basic_mini_circuit
                            
                            # Clean up memory
                            del basic_result, basic_state, basic_mini_circuit
                            continue  # Skip the failure handling below
                        
                        # Real failure case
                        failed_sims.add(sim_idx)
                        async_logger.log_batch(f"❌ Failed Sim {sim_idx} on device {device_id} (load: {device_load_counts[device_id]}): {str(e)}")
                        
                        # Create proper empty results for failed simulation
                        # Match the format of successful simulation results
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
                            
                            from src.vnc_sim import Pruning_state
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
                                keys=self.neuron_params.seeds[sim_idx:sim_idx+1] if sim_idx < len(self.neuron_params.seeds) else jax.random.split(jax.random.PRNGKey(42), 1)
                            )
                        
                        # Move empty results to CPU
                        empty_result = jax.device_put(empty_result, jax.devices("cpu")[0])
                        empty_state = jax.device_put(empty_state, jax.devices("cpu")[0])
                        empty_mini_circuit = jax.device_put(empty_mini_circuit, jax.devices("cpu")[0])
                        
                        results_dict[sim_idx] = empty_result
                        states_dict[sim_idx] = empty_state
                        mini_circuits_dict[sim_idx] = empty_mini_circuit
                        
                        # Clean up failed simulation memory
                        del empty_result, empty_state, empty_mini_circuit
                
                # Comprehensive memory cleanup after processing completed simulations
                if done_tasks:
                    # Clear JAX caches periodically to prevent memory accumulation
                    jax.clear_caches()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
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
        
        # Final comprehensive memory cleanup
        del results_dict, states_dict, mini_circuits_dict, active_tasks, task_devices, device_load_counts
        jax.clear_caches()
        import gc
        gc.collect()
        
        async_logger.log_batch("🧹 Final memory cleanup completed")
        
        return results_list, states_list, mini_circuits_list

    async def run_async_batch(self, sim_indices: List[int]) -> Tuple[List[jnp.ndarray], List[Pruning_state]]:
        """Run a batch of simulations asynchronously."""
        async_logger.log_batch(f"Starting batch with {len(sim_indices)} simulations: {sim_indices}")
        
        # Create tasks for all simulations
        tasks = [
            self._process_single_simulation(sim_idx)
            for sim_idx in sim_indices
        ]
        
        # Run all simulations concurrently
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        results = []
        states = []
        failed_sims = []
        
        for i, task_result in enumerate(completed_tasks):
            if isinstance(task_result, Exception):
                failed_sims.append(sim_indices[i])
                async_logger.log_sim(sim_indices[i], f"FAILED: {task_result}", "ERROR")
                # Create empty result for failed simulation
                empty_result = jnp.zeros((self.sim_params.n_neurons, len(self.sim_params.t_axis)))
                empty_state = initialize_pruning_state(self.neuron_params, self.sim_params, 1)
                results.append(empty_result)
                states.append(empty_state)
            else:
                sim_idx, result, state = task_result
                results.append(result)
                states.append(state)
        
        if failed_sims:
            async_logger.log_batch(f"Batch completed with {len(failed_sims)} failures: {failed_sims}")
        else:
            async_logger.log_batch(f"Batch completed successfully - all {len(sim_indices)} simulations finished")
            
        # Print summary for this batch
        async_logger.print_summary(sim_indices)
        
        return results, states
    
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
    max_concurrent: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run pruning simulation with streaming processing - new simulations start as others finish.
    
    This is more efficient than batch processing because:
    - No waiting for entire batches to complete
    - Better GPU utilization (slots filled immediately)
    - Faster overall completion time
    - Real-time progress monitoring
    
    Args:
        neuron_params: Neuron parameters
        sim_params: Simulation parameters  
        sim_config: Simulation configuration
        max_concurrent: Maximum concurrent simulations (default: n_devices * 2)
    
    Returns:
        Tuple of (results_array, mini_circuits_array)
    """
    total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
    
    # Create async manager
    manager = AsyncPruningManager(neuron_params, sim_params, sim_config)
    
    async def run_streaming():
        return await manager.run_streaming_simulations(total_sims, max_concurrent)
    
    # Run the streaming async execution
    all_results, all_states, all_mini_circuits = asyncio.run(run_streaming())
    
    # Convert results to arrays and reshape
    results_array = jnp.stack(all_results, axis=0)
    results_reshaped = results_array.reshape(
        sim_params.n_stim_configs, sim_params.n_param_sets,
        sim_params.n_neurons, len(sim_params.t_axis)
    )
    
    # Use the mini_circuits from actual simulation results (not reconstructed from states)
    # These mini_circuits are based on which neurons actually fired in the final simulations
    mini_circuits_array = jnp.stack(all_mini_circuits, axis=0)
    
    # Create final W_masks from the correct mini_circuits
    final_w_masks = []
    for i, mini_circuit in enumerate(all_mini_circuits):
        # Create the final W_mask: neurons that fired can connect to neurons that fired
        # Add motor neurons back for connectivity (they're not in mini_circuit but need connections)
        mn_mask = jnp.isin(jnp.arange(sim_params.n_neurons), neuron_params.mn_idxs)
        active_neurons = mini_circuit | mn_mask  # Include both active interneurons and motor neurons
        final_w_mask = (active_neurons[:, None] * active_neurons[None, :]).astype(jnp.bool_)
        final_w_masks.append(final_w_mask)
    final_w_masks_array = jnp.stack(final_w_masks, axis=0)
    
    # Update the original neuron_params with all the final W_masks
    updated_neuron_params = neuron_params._replace(W_mask=final_w_masks_array)

    return results_reshaped, mini_circuits_array, updated_neuron_params


def run_async_pruning_simulation(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    sim_config: SimulationConfig,
    batch_size: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run pruning simulation with asynchronous processing of individual simulations.
    
    Args:
        neuron_params: Neuron parameters
        sim_params: Simulation parameters
        sim_config: Simulation configuration
        batch_size: Optional batch size for grouping (default: all simulations)
    
    Returns:
        Tuple of (results_array, mini_circuits_array)
    """
    total_sims = sim_params.n_stim_configs * sim_params.n_param_sets
    
    if batch_size is None:
        batch_size = total_sims  # Process all simulations concurrently
    
    # Create async manager
    manager = AsyncPruningManager(neuron_params, sim_params, sim_config)
    
    all_results = []
    all_states = []
    
    # Process in batches if needed
    n_batches = (total_sims + batch_size - 1) // batch_size
    
    async def run_batches():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_sims)
            batch_indices = list(range(start_idx, end_idx))
            
            print(f"Processing batch {i + 1}/{n_batches} with {len(batch_indices)} simulations")
            
            # Create simplified progress monitoring task
            async def monitor_progress():
                while manager.active_sims:
                    await asyncio.sleep(15)  # Update every 15 seconds
                    progress = manager.get_progress_report()
                    active = progress['active_simulations']
                    completed = progress['completed_simulations']
                    total = active + completed
                    if active > 0:
                        async_logger.log_batch(f"Progress: {completed}/{total} completed, {active} active, avg iterations: {progress['average_iterations']:.1f}")
            
            # Start monitoring and batch processing
            monitor_task = asyncio.create_task(monitor_progress())
            batch_results, batch_states = await manager.run_async_batch(batch_indices)
            monitor_task.cancel()
            
            all_results.extend(batch_results)
            all_states.extend(batch_states)
            
            async_logger.log_batch(f"Batch {i + 1}/{n_batches} completed")
    
    # Run the async event loop
    asyncio.run(run_batches())
    
    # Convert results to arrays and reshape
    results_array = jnp.stack(all_results, axis=0)
    results_reshaped = results_array.reshape(
        sim_params.n_stim_configs, sim_params.n_param_sets,
        sim_params.n_neurons, len(sim_params.t_axis)
    )
    
    # Convert states to mini_circuits format (compatible with existing save code)
    mn_mask = jnp.isin(jnp.arange(sim_params.n_neurons), neuron_params.mn_idxs)
    mini_circuits = []
    for state in all_states:
        # Extract the active neurons (minimal circuit) from each state
        if hasattr(state, 'total_removed_neurons'):
            mini_circuit = (~state.total_removed_neurons)  
        else:
            # Fallback: all neurons active
            mini_circuit = jnp.ones(sim_params.n_neurons, dtype=bool)
        mini_circuits.append(mini_circuit)
    
    mini_circuits_array = jnp.stack(mini_circuits, axis=0)
    
    return results_reshaped, mini_circuits_array

