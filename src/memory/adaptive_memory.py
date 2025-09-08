"""
Simplified memory management for VNC simulations.

Focuses on the core memory management essentials:
- Conservative batch sizing based on GPU count and simulation size
- Basic memory monitoring and cleanup
- JAX configuration for large simulations
- AdaptiveMemoryManager for async_vnc_sim compatibility
"""

import jax
import jax.numpy as jnp
import gc
import time
import os
import psutil
from typing import Optional, Dict, Any
from src.data.data_classes import Pruning_state


def move_pruning_state_to_cpu(pruning_state: Pruning_state) -> Pruning_state:
    """Move all arrays in a Pruning_state to CPU device.
    
    This is a memory management utility that ensures all JAX arrays within
    a Pruning_state NamedTuple are placed on CPU devices for consistent
    memory management and checkpointing.
    
    Args:
        pruning_state: The Pruning_state object with arrays potentially on GPU
        
    Returns:
        A new Pruning_state object with all arrays moved to CPU
    """
    cpu_device = jax.devices("cpu")[0]
    return Pruning_state(
        W_mask=jax.device_put(pruning_state.W_mask, cpu_device),
        interneuron_mask=jax.device_put(pruning_state.interneuron_mask, cpu_device),
        level=jax.device_put(pruning_state.level, cpu_device),
        total_removed_neurons=jax.device_put(pruning_state.total_removed_neurons, cpu_device),
        removed_stim_neurons=jax.device_put(pruning_state.removed_stim_neurons, cpu_device),
        neurons_put_back=jax.device_put(pruning_state.neurons_put_back, cpu_device),
        prev_put_back=jax.device_put(pruning_state.prev_put_back, cpu_device),
        last_removed=jax.device_put(pruning_state.last_removed, cpu_device),
        remove_p=jax.device_put(pruning_state.remove_p, cpu_device),
        min_circuit=jax.device_put(pruning_state.min_circuit, cpu_device),
        keys=jax.device_put(pruning_state.keys, cpu_device),
        last_good_oscillating_removed=jax.device_put(pruning_state.last_good_oscillating_removed, cpu_device),
        last_good_oscillating_put_back=jax.device_put(pruning_state.last_good_oscillating_put_back, cpu_device),
        last_good_oscillation_score=jax.device_put(pruning_state.last_good_oscillation_score, cpu_device),
        last_good_W_mask=jax.device_put(pruning_state.last_good_W_mask, cpu_device),
        last_good_key=jax.device_put(pruning_state.last_good_key, cpu_device),
        last_good_stim_idx=jax.device_put(pruning_state.last_good_stim_idx, cpu_device),
        last_good_param_idx=jax.device_put(pruning_state.last_good_param_idx, cpu_device)
    )


def get_gpu_count() -> int:
    """Get the number of available GPU devices."""
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        return max(1, len(gpu_devices))
    except:
        return 1

def monitor_memory_usage(simulation_idx: Optional[int] = None, warn_threshold: float = 85.0) -> Dict[str, Any]:
    """Basic memory monitoring for RAM and GPU with warnings."""
    status = {}
    
    try:
        # System memory
        memory_info = psutil.virtual_memory()
        status['ram_percent'] = memory_info.percent
        status['ram_available_gb'] = memory_info.available / (1024**3)
        status['ram_warning'] = memory_info.percent > warn_threshold
        
        # GPU memory if available
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            total_used = 0
            total_capacity = 0
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_used += info.used
                total_capacity += info.total
            
            status['gpu_percent'] = (total_used / total_capacity) * 100 if total_capacity > 0 else 0
            status['gpu_free_gb'] = (total_capacity - total_used) / (1024**3)
            status['gpu_warning'] = status['gpu_percent'] > warn_threshold
            status['gpu_available'] = True
            
        except Exception:
            status['gpu_available'] = False
            
    except Exception as e:
        status['error'] = str(e)
    
    return status


def log_detailed_gpu_status(logger_func=print, prefix: str = ""):
    """Log detailed GPU status for debugging."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
            gpu_name = gpu_name_raw.decode('utf-8') if isinstance(gpu_name_raw, bytes) else gpu_name_raw
            
            used_gb = info.used / (1024**3)
            total_gb = info.total / (1024**3)
            percent = (info.used / info.total) * 100
            
            logger_func(f"{prefix}GPU{i} ({gpu_name}): {percent:.1f}% used, {used_gb:.1f}GB/{total_gb:.1f}GB")
            
    except Exception as e:
        logger_func(f"{prefix}GPU status unavailable: {e}")

def log_memory_status(status: dict, logger_func=print, prefix: str = ""):
    """
    Log memory status information with multi-GPU support.
    
    Args:
        status: Status dictionary from monitor_memory_usage()
        logger_func: Function to use for logging (default: print)
        prefix: Prefix string for log messages
    """
    if 'error' in status:
        logger_func(f"{prefix}Memory monitoring unavailable: {status['error']}")
        return
    
    ram_status = f"RAM: {status['ram_percent']:.1f}% used, {status['ram_available_gb']:.1f}GB free"
    
    # Multi-GPU support
    if status.get('gpu_available', False):
        gpu_devices = status.get('gpu_devices', [])
        gpu_count = status.get('gpu_count', 0)
        
        if gpu_count > 1:
            # Show aggregated totals for multiple GPUs
            gpu_status = f" | GPU Total ({gpu_count} GPUs): {status['gpu_percent']:.1f}% used, {status['gpu_free_gb']:.1f}GB free"
            
            # Log detailed per-GPU info if requested (can be controlled via prefix)
            if "detailed" in prefix.lower():
                logger_func(f"{prefix}{ram_status}{gpu_status}")
                for gpu in gpu_devices:
                    warning_icon = " ⚠️" if gpu['warning'] else ""
                    logger_func(f"{prefix}  GPU{gpu['device_id']}: {gpu['percent']:.1f}% used, {gpu['free_gb']:.1f}GB free ({gpu['name']}){warning_icon}")
                return
        else:
            # Single GPU - show detailed info
            gpu = gpu_devices[0] if gpu_devices else {}
            gpu_name = gpu.get('name', 'Unknown')
            gpu_status = f" | GPU: {status['gpu_percent']:.1f}% used, {status['gpu_free_gb']:.1f}GB free ({gpu_name})"
    else:
        if 'gpu_error' in status:
            gpu_status = f" | GPU: Error - {status['gpu_error']}"
        else:
            gpu_status = " | GPU: Not available"
    
    warning = ""
    if status.get('ram_warning', False) or status.get('gpu_warning', False):
        warning = " ⚠️ HIGH MEMORY USAGE!"
        
    logger_func(f"{prefix}{ram_status}{gpu_status}{warning}")
    

def basic_memory_cleanup(logger_func=print):
    """Basic memory cleanup - just the essentials."""
    gc.collect()
    jax.clear_caches()
    # Force backend cleanup if available
    try:
        if hasattr(jax, 'clear_backends'):
            jax.clear_backends()
    except Exception as e:
        logger_func.log_batch(f"Warning: Could not clear JAX backends: {e}")
    time.sleep(0.05)  # Brief pause for memory to actually free


def calculate_optimal_concurrent_size(
    n_neurons: int, 
    n_timepoints: int, 
    total_simulations: int,
    n_devices: int = None, 
    is_pruning: bool = False
) -> int:
    """
    Calculate optimal max_concurrent size based on GPU memory and device count.
    """
    if n_devices is None:
        n_devices = get_gpu_count()
    
    # Get available GPU memory
    gpu_memory_gb = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        total_memory = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory += info.total
        
        gpu_memory_gb = total_memory / (1024**3)
        
    except Exception:
        # Fallback: assume 40GB per device (conservative for L40S)
        gpu_memory_gb = 40 * n_devices
        print(f"GPU memory detection failed, assuming {gpu_memory_gb}GB total")
    
    # Estimate memory per simulation
    # Each simulation needs: output array + intermediate computations
    output_size_mb = (n_neurons * n_timepoints * 4) / (1024**2)  # float32
    intermediate_mb = output_size_mb * 1.5  # ODE solver overhead
    
    total_per_sim_mb = (output_size_mb + intermediate_mb) * 1.5
    
    # Weight matrix memory (shared across simulations)
    weight_matrix_mb = (n_neurons * n_neurons * 4) / (1024**2) * 1.5
    
    memory_fraction = 0.75  # Standard allocation for non-pruning
    
    usable_memory_mb = (gpu_memory_gb * 1024) * memory_fraction
    available_for_concurrent_mb = usable_memory_mb - weight_matrix_mb
    
    if available_for_concurrent_mb <= 0:
        memory_limited = 1
    else:
        memory_limited = max(1, int(available_for_concurrent_mb / total_per_sim_mb))

    if is_pruning and n_devices > 1:
        # For Pruning, hard code 8 sims per GPU to avoid OOMs
        memory_limited = 8 * n_devices

    # Ensure we don't exceed total simulations
    optimal = min(memory_limited, total_simulations)
    
    print(f"Memory calculation: {total_per_sim_mb:.1f}MB per sim, "
          f"{memory_fraction:.0%} mem fraction, {memory_limited} memory-limited -> {optimal} optimal")
    
    return max(1, optimal)


class AdaptiveMemoryManager:
    """Simplified memory manager for async_vnc_sim compatibility."""
    
    def __init__(self, total_simulations: int, simulation_type: str = "pruning"):
        self.total_simulations = total_simulations
        self.is_large_simulation = total_simulations >= 1024
        
        print(f"Memory Manager: {total_simulations} sims ({'large' if self.is_large_simulation else 'standard'})")
    
    def should_cleanup(self, sim_index: int, iteration: int, batch_index: Optional[int] = None) -> bool:
        """Simple cleanup decision based on simulation progress."""
        if iteration < 5:
            return False
            
        if self.is_large_simulation:
            # More frequent cleanup for large simulations
            if sim_index >= 500:  # Near known OOM point
                return iteration % 10 == 0
            else:
                return iteration % 50 == 0
        else:
            return iteration % 100 == 0

    def perform_cleanup(self, logger_func=print, context: str = ""):
        """Perform memory cleanup."""
        basic_memory_cleanup(logger_func=logger_func)
        if context and self.is_large_simulation:  # Only log for large sims to reduce noise
            logger_func(f"  Memory cleanup: {context}")

    def monitor_and_cleanup_if_needed(self, sim_index: int, warn_threshold: float = 85.0) -> bool:
        """Monitor memory and cleanup if needed."""
        # Only check every 10 simulations to reduce overhead
        if sim_index % 10 != 0:
            return False
            
        status = monitor_memory_usage(sim_index, warn_threshold)
        
        # Emergency cleanup if memory is very high
        if status.get('ram_percent', 0) > 90 or status.get('gpu_percent', 0) > 90:
            print(f"⚠️ Emergency cleanup at sim {sim_index}")
            self.perform_cleanup(force_aggressive=True, context="emergency")
            return True
            
        return False


def create_memory_manager(total_simulations: int, simulation_type: str = "pruning") -> AdaptiveMemoryManager:
    """Create memory manager for async_vnc_sim compatibility."""
    return AdaptiveMemoryManager(total_simulations, simulation_type)
