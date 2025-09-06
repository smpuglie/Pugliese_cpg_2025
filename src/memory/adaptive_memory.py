"""
Comprehensive memory management for VNC simulations that balances performance and stability.

This module provides adaptive memory management that:
1. Reduces cleanup frequency for better performance (every 100 vs every 10 iterations)
2. Applies aggressive strategies only when needed (large sims, danger zones)
3. Uses conservative JAX settings only for large simulations
4. Maintains memory monitoring (minimal overhead)
5. GPU-aware concurrent simulation limits (16 concurrent simulations per GPU maximum)
6. Provides utilities for memory estimation, monitoring, and conservative sizing

MAIN FUNCTIONS:
- AdaptiveMemoryManager: Main class for adaptive memory management
- create_memory_manager(): Factory function for creating memory managers
- get_gpu_count(): Get number of available GPU devices
- monitor_memory_usage(): Monitor current RAM and GPU memory usage
- log_memory_status(): Log memory status with formatting
- get_conservative_batch_size(): Calculate GPU-aware conservative batch sizes
- aggressive_memory_cleanup(): Force aggressive memory cleanup
- should_trigger_cleanup(): Determine if cleanup should be triggered
- configure_jax_for_large_simulations(): Configure JAX for conservative memory usage
- estimate_memory_requirements(): Estimate memory requirements for simulations
- calculate_optimal_concurrent_size(): Calculate optimal concurrent size for async streaming
- optimize_for_performance_vs_stability(): Get optimized settings dictionary

This module combines functionality from the former memory_utils.py and adaptive_memory.py
to provide a unified interface for all memory management operations.
"""

import jax
import jax.numpy as jnp
import gc
import time
import os
import psutil
from typing import Optional, Dict, Any


def get_gpu_count() -> int:
    """Get the number of available GPU devices."""
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        return max(1, len(gpu_devices))
    except:
        return 1


def aggressive_memory_cleanup(force_gpu_cache_clear: bool = True):
    """
    Perform aggressive memory cleanup to prevent OOM errors.
    
    Args:
        force_gpu_cache_clear: Whether to clear JAX GPU caches (expensive but effective)
    """
    # Clear Python objects
    gc.collect()
    
    # Clear JAX caches if requested
    if force_gpu_cache_clear:
        jax.clear_caches()
    
    # Give system time to actually free memory
    time.sleep(0.1)


def monitor_memory_usage(simulation_idx: Optional[int] = None, warn_threshold: float = 85.0) -> dict:
    """
    Monitor current memory usage and return status.
    
    Args:
        simulation_idx: Current simulation index for logging
        warn_threshold: RAM usage percentage threshold for warnings
        
    Returns:
        Dictionary with memory status information
    """
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
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            status['gpu_used_gb'] = info.used / (1024**3)
            status['gpu_free_gb'] = info.free / (1024**3)
            status['gpu_percent'] = (info.used / info.total) * 100
            status['gpu_warning'] = status['gpu_percent'] > warn_threshold
        except:
            status['gpu_available'] = False
            
    except Exception as e:
        status['error'] = str(e)
    
    return status


def log_memory_status(status: dict, logger_func=print, prefix: str = ""):
    """
    Log memory status information.
    
    Args:
        status: Status dictionary from monitor_memory_usage()
        logger_func: Function to use for logging (default: print)
        prefix: Prefix string for log messages
    """
    if 'error' in status:
        logger_func(f"{prefix}Memory monitoring unavailable: {status['error']}")
        return
    
    ram_status = f"RAM: {status['ram_percent']:.1f}% used, {status['ram_available_gb']:.1f}GB free"
    
    if status.get('gpu_available', True):
        gpu_status = f" | GPU: {status['gpu_percent']:.1f}% used, {status['gpu_free_gb']:.1f}GB free"
    else:
        gpu_status = ""
    
    warning = ""
    if status.get('ram_warning', False) or status.get('gpu_warning', False):
        warning = " ⚠️ HIGH MEMORY USAGE!"
        
    logger_func(f"{prefix}{ram_status}{gpu_status}{warning}")


def should_trigger_cleanup(simulation_idx: int, batch_idx: int, total_sims: int, batch_size: int) -> bool:
    """
    Determine if aggressive memory cleanup should be triggered based on simulation progress.
    
    Args:
        simulation_idx: Current simulation index
        batch_idx: Current batch index  
        total_sims: Total number of simulations
        batch_size: Batch size being used
        
    Returns:
        True if cleanup should be triggered
    """
    # For large simulations (n_replicates >= 1024), be more aggressive
    if total_sims >= 1024:
        # Clean up every 50 simulations or every 5 batches, whichever is more frequent
        return (simulation_idx > 0 and simulation_idx % 50 == 0) or (batch_idx > 0 and batch_idx % 5 == 0)
    
    # For medium simulations (256-1024), moderate cleanup
    elif total_sims >= 256:
        return (simulation_idx > 0 and simulation_idx % 100 == 0) or (batch_idx > 0 and batch_idx % 10 == 0)
        
    # For small simulations, minimal cleanup
    else:
        return batch_idx > 0 and batch_idx % 20 == 0


def get_conservative_batch_size(total_sims: int, proposed_batch_size: int, max_concurrent: int, n_gpus: Optional[int] = None) -> int:
    """
    Calculate a conservative batch size or max_concurrent to prevent OOM errors.
    
    Args:
        total_sims: Total number of simulations to run
        proposed_batch_size: Originally proposed batch size (can be None)
        max_concurrent: Originally proposed max concurrent simulations (can be None)
        n_gpus: Number of GPUs available (auto-detected if None)
        
    Returns:
        Conservative batch size or max concurrent value
    """
    if n_gpus is None:
        n_gpus = get_gpu_count()
    
    # GPU-aware maximum concurrent simulations (16 per GPU)
    gpu_aware_max = 16 * n_gpus
    
    # For very large simulations, reduce batch size/concurrency significantly
    if total_sims >= 1024:
        # Cap based on GPU count: 4 per GPU for very large sims, but respect GPU limit
        conservative_max = min(4 * n_gpus, gpu_aware_max)
        if proposed_batch_size is not None:
            return min(proposed_batch_size, conservative_max)
        elif max_concurrent is not None:
            return min(max_concurrent, conservative_max)
    
    # For large simulations, moderate reduction
    elif total_sims >= 512:
        # Cap based on GPU count: 8 per GPU for large sims, but respect GPU limit  
        conservative_max = min(8 * n_gpus, gpu_aware_max)
        if proposed_batch_size is not None:
            return min(proposed_batch_size, conservative_max)
        elif max_concurrent is not None:
            return min(max_concurrent, conservative_max)
    
    # Return original values for smaller simulations, but still respect GPU limit
    if proposed_batch_size is not None:
        return min(proposed_batch_size, gpu_aware_max)
    elif max_concurrent is not None:
        return min(max_concurrent, gpu_aware_max)
    else:
        return min(32, gpu_aware_max)  # Default fallback, GPU-aware


def configure_jax_for_large_simulations():
    """
    Configure JAX settings for better memory management in large simulations.
    """
    # Configure JAX for more conservative memory usage
    jax.config.update("jax_platform_name", "gpu")
    
    # Enable memory preallocation to avoid fragmentation
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"  # Use only 85% of GPU memory
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    # Reduce JAX cache size
    # jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    
    print("JAX configured for conservative memory usage in large simulations")


def estimate_memory_requirements(n_neurons: int, n_timepoints: int, batch_size: int) -> dict:
    """
    Estimate memory requirements for a simulation.
    
    Args:
        n_neurons: Number of neurons in the model
        n_timepoints: Number of time points per simulation
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Basic memory estimates (similar to the calculation functions)
    output_size = n_neurons * n_timepoints * 4 * batch_size  # float32
    intermediate_size = output_size * 2  # ODE solver intermediates
    weight_ops_size = n_neurons * n_neurons * 4  # Weight matrix (shared)
    overhead_factor = 1.75  # JAX compilation and caching overhead
    
    total_per_batch_gb = ((output_size + intermediate_size) * overhead_factor + weight_ops_size) / (1024**3)
    
    return {
        'output_gb': output_size / (1024**3),
        'intermediate_gb': intermediate_size / (1024**3), 
        'weight_matrix_gb': weight_ops_size / (1024**3),
        'total_per_batch_gb': total_per_batch_gb,
        'overhead_factor': overhead_factor
    }


class AdaptiveMemoryManager:
    """Adaptive memory management that adjusts strategy based on simulation characteristics."""
    
    def __init__(self, total_simulations: int, simulation_type: str = "pruning"):
        self.total_simulations = total_simulations
        self.simulation_type = simulation_type
        self.cleanup_stats = {"cleanups": 0, "time_spent": 0.0}
        
        # Determine if this is a large simulation requiring aggressive management
        self.is_large_simulation = total_simulations >= 1024
        self.is_very_large_simulation = total_simulations >= 2048
        
        # Configure JAX only for large simulations
        if self.is_large_simulation:
            self._configure_conservative_jax()
        
        print(f"Adaptive Memory Manager initialized for {total_simulations} simulations")
        print(f"Strategy: {'Conservative' if self.is_large_simulation else 'Standard'}")
    
    def _configure_conservative_jax(self):
        """Configure JAX for conservative memory usage in large simulations only."""
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.65"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        print("  JAX configured for conservative memory usage")
    
    def should_cleanup(self, sim_index: int, iteration: int, batch_index: Optional[int] = None) -> bool:
        """
        Determine if cleanup should be performed based on adaptive strategy.
        
        Args:
            sim_index: Current simulation index
            iteration: Current iteration within simulation
            batch_index: Current batch index (for batch processing)
            
        Returns:
            True if cleanup should be performed
        """
        # Never cleanup on the first few iterations to avoid unnecessary overhead
        if iteration < 5:
            return False
            
        # Adaptive strategy based on simulation size and progress
        if self.is_very_large_simulation:
            # Very large simulations: More aggressive in danger zone
            if sim_index >= 800:  # Approaching known OOM point
                return iteration % 5 == 0
            elif sim_index >= 600:
                return iteration % 10 == 0
            elif sim_index >= 300:
                return iteration % 20 == 0
            else:
                return iteration % 50 == 0
                
        elif self.is_large_simulation:
            # Large simulations: Progressive strategy
            if sim_index >= 700:
                return iteration % 10 == 0
            elif sim_index >= 400:
                return iteration % 25 == 0
            else:
                return iteration % 100 == 0  # Performance analysis recommendation
                
        else:
            # Small/medium simulations: Minimal cleanup
            return iteration % 100 == 0
    
    def should_cleanup_batch(self, batch_index: int, total_batches: int) -> bool:
        """Determine if batch-level cleanup should be performed."""
        if self.is_large_simulation:
            # More frequent cleanup for large simulations
            return batch_index % 5 == 0
        else:
            # Less frequent for smaller simulations
            return batch_index % 10 == 0
    
    def perform_cleanup(self, force_aggressive: bool = False, context: str = ""):
        """
        Perform memory cleanup with timing.
        
        Args:
            force_aggressive: Force aggressive cleanup regardless of settings
            context: Context string for logging
        """
        start_time = time.time()
        
        # Always do basic cleanup
        gc.collect()
        
        # JAX cache clearing based on strategy
        if force_aggressive or self.is_large_simulation:
            jax.clear_caches()
        
        # Allow system time to actually free memory
        time.sleep(0.05)  # Reduced from 0.1s for better performance
        
        elapsed = time.time() - start_time
        self.cleanup_stats["cleanups"] += 1
        self.cleanup_stats["time_spent"] += elapsed
        
        if context and elapsed > 0.1:  # Only log slow cleanups
            print(f"  Memory cleanup ({context}): {elapsed:.3f}s")
    
    def monitor_and_cleanup_if_needed(self, sim_index: int, warn_threshold: float = 85.0) -> bool:
        """
        Monitor memory and perform emergency cleanup if needed.
        
        Returns True if emergency cleanup was performed.
        """
        # Only monitor every 10 simulations to reduce overhead
        if sim_index % 10 != 0:
            return False
            
        status = monitor_memory_usage(sim_index, warn_threshold)
        
        # Emergency cleanup if memory is very high
        if status.get('ram_percent', 0) > 90 or status.get('gpu_percent', 0) > 90:
            print(f"⚠️  Emergency memory cleanup at sim {sim_index}")
            log_memory_status(status, prefix="  Before cleanup: ")
            
            self.perform_cleanup(force_aggressive=True, context="emergency")
            
            # Check status after cleanup
            status_after = monitor_memory_usage(sim_index)
            log_memory_status(status_after, prefix="  After cleanup: ")
            return True
            
        # Warning if memory is high but not critical
        elif status.get('ram_warning', False) or status.get('gpu_warning', False):
            log_memory_status(status, prefix="  Memory status: ")
            
        return False
    
    def get_optimal_concurrent_size(self, base_concurrent: int, n_gpus: Optional[int] = None) -> int:
        """Get optimal concurrent size based on simulation characteristics and GPU count."""
        if n_gpus is None:
            n_gpus = get_gpu_count()
        
        # Calculate GPU-aware maximum
        max_concurrent_per_gpu = 16
        gpu_aware_max = max_concurrent_per_gpu * n_gpus
        
        if self.is_very_large_simulation:
            # Very conservative for very large simulations
            conservative_max = min(4 * n_gpus, gpu_aware_max)
            return min(base_concurrent, conservative_max)
        elif self.is_large_simulation:
            # Conservative for large simulations  
            conservative_max = min(8 * n_gpus, gpu_aware_max)
            return min(base_concurrent, conservative_max)
        else:
            # Standard for smaller simulations, capped at 16*n_gpus
            return min(base_concurrent, gpu_aware_max)
    
    def get_optimal_batch_size(self, base_batch_size: int) -> int:
        """Get optimal batch size based on simulation characteristics."""
        if self.is_very_large_simulation:
            return min(base_batch_size, 32)
        elif self.is_large_simulation:
            return min(base_batch_size, 64)
        else:
            return base_batch_size
    
    def get_memory_fraction(self) -> float:
        """Get optimal GPU memory fraction based on simulation size."""
        if self.is_very_large_simulation:
            return 0.75  # Very conservative
        elif self.is_large_simulation:
            return 0.85  # Conservative
        else:
            return 0.95  # Standard
    
    def print_performance_summary(self):
        """Print summary of cleanup performance impact."""
        if self.cleanup_stats["cleanups"] > 0:
            avg_cleanup_time = self.cleanup_stats["time_spent"] / self.cleanup_stats["cleanups"]
            total_cleanup_time = self.cleanup_stats["time_spent"]
            
            print(f"\nMemory Management Performance Summary:")
            print(f"  Total cleanups: {self.cleanup_stats['cleanups']}")
            print(f"  Total cleanup time: {total_cleanup_time:.2f}s")
            print(f"  Average cleanup time: {avg_cleanup_time:.3f}s")
            print(f"  Estimated overhead: ~{(total_cleanup_time / 3600) * 100:.1f}% of 1hr runtime")


def create_memory_manager(total_simulations: int, simulation_type: str = "pruning") -> AdaptiveMemoryManager:
    """
    Create an adaptive memory manager for the given simulation parameters.
    
    Args:
        total_simulations: Total number of simulations to run
        simulation_type: Type of simulation ("pruning", "regular", etc.)
        
    Returns:
        Configured AdaptiveMemoryManager instance
    """
    return AdaptiveMemoryManager(total_simulations, simulation_type)


def optimize_for_performance_vs_stability(total_simulations: int, n_gpus: Optional[int] = None) -> Dict[str, Any]:
    """
    Get optimized settings that balance performance vs stability.
    
    Args:
        total_simulations: Total number of simulations
        n_gpus: Number of GPUs available (auto-detected if None)
        
    Returns:
        Dictionary of optimized settings
    """
    manager = AdaptiveMemoryManager(total_simulations)
    
    if n_gpus is None:
        n_gpus = get_gpu_count()
    
    return {
        "memory_fraction": manager.get_memory_fraction(),
        "max_concurrent": manager.get_optimal_concurrent_size(32, n_gpus),  # Base of 32, GPU-aware cap
        "batch_size": manager.get_optimal_batch_size(64),  # Base of 64
        "cleanup_frequency": "adaptive",  # Use adaptive strategy
        "enable_monitoring": True,  # Always enable (minimal overhead)
        "conservative_jax": manager.is_large_simulation,  # Only for large sims
        "performance_impact": "low" if not manager.is_large_simulation else "moderate"
    }


# Performance tuning recommendations based on analysis
# Note: max_concurrent values are per GPU and should be multiplied by n_gpus, capped at 16*n_gpus
PERFORMANCE_RECOMMENDATIONS = {
    "small_simulations": {
        "max_concurrent_per_gpu": 16,  # Multiply by n_gpus, cap at 16*n_gpus total
        "batch_size": 64,
        "cleanup_every_n": 100,
        "memory_fraction": 0.75,
        "expected_overhead": "< 2%"
    },
    "medium_simulations": {
        "max_concurrent_per_gpu": 16,  # Multiply by n_gpus, cap at 16*n_gpus total
        "batch_size": 48,
        "cleanup_every_n": 50,
        "memory_fraction": 0.70,
        "expected_overhead": "< 5%"
    },
    "large_simulations": {
        "max_concurrent_per_gpu": 16,   # Multiply by n_gpus, cap at 16*n_gpus total
        "batch_size": 32,
        "cleanup_every_n": "adaptive",
        "memory_fraction": 0.65,
        "expected_overhead": "5-10%"
    },
    "very_large_simulations": {
        "max_concurrent_per_gpu": 16,   # Multiply by n_gpus, cap at 16*n_gpus total
        "batch_size": 16,
        "cleanup_every_n": "adaptive",
        "memory_fraction": 0.60,
        "expected_overhead": "10-15%"
    }
}


def calculate_optimal_concurrent_size(
    n_neurons: int, 
    n_timepoints: int, 
    total_simulations: int,
    n_devices: int = None, 
    max_concurrent: int = 32,
    is_pruning: bool = False
) -> int:
    """
    Calculate optimal max_concurrent size for async streaming based on memory and device constraints.
    
    Args:
        n_neurons: Number of neurons in the model
        n_timepoints: Number of time points per simulation
        total_simulations: Total number of simulations to run
        n_devices: Number of devices available (defaults to JAX device count)
        max_concurrent: Maximum allowed concurrent simulations (default 32)
        is_pruning: Whether this is for pruning simulations (affects memory calculation)
    
    Returns:
        Optimal max_concurrent size for async streaming execution
    """
    print("Calculating optimal concurrent batch size for async streaming...")
    
    if n_devices is None:
        n_devices = jax.device_count()
    
    # Handle edge case where we have fewer simulations than devices
    if total_simulations < n_devices:
        return total_simulations
    
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
            # Try different methods to get GPU memory
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
                
                gpu_memory_per_device = device_memories[0]
                total_gpu_memory = sum(device_memories)
                gpu_memory = total_gpu_memory
                
                print(f"Detected {n_gpu_devices} GPU device(s) via nvidia-ml")
                
            except ImportError:
                print("pynvml not available, using fallback GPU memory estimation...")
                
                # Fallback - estimate based on common GPU types
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
        else:
            print("No GPU devices detected - using CPU mode")
            
    except Exception as e:
        print(f"GPU detection failed: {e}")
        pass
    
    # Estimate memory usage per concurrent simulation
    # For async streaming, each concurrent simulation needs:
    # 1. Output array: n_neurons * n_timepoints * 4 bytes (float32)
    # 2. Intermediate computations: ~1.5x output for async overhead (less than sync)
    # 3. Weight matrix operations: n_neurons^2 * 4 bytes (shared across sims)
    # 4. Pruning overhead: Additional state if pruning enabled
    
    output_size = n_neurons * n_timepoints * 4  # float32
    intermediate_size = output_size * 1.5  # Async has less intermediate storage
    weight_ops_size = n_neurons * n_neurons * 4  # Weight matrix operations (shared)
    
    # Pruning simulations need additional memory for state tracking
    if is_pruning:
        pruning_overhead = n_neurons * 20 * 4  # Approximate pruning state overhead
        overhead_factor = 2.0  # Higher overhead for pruning iterations
        print("Accounting for pruning simulation overhead")
    else:
        pruning_overhead = 0
        overhead_factor = 1.5  # Lower overhead for regular simulations
    
    # Memory per concurrent simulation (weight matrix is shared)
    bytes_per_concurrent_sim = (output_size + intermediate_size + pruning_overhead) * overhead_factor
    # Shared memory (weight matrix) - only counted once regardless of concurrent sims
    shared_memory = weight_ops_size * overhead_factor
    
    print(f"Estimated memory per concurrent simulation: {bytes_per_concurrent_sim / (1024 ** 2):.1f} MB")
    print(f"Shared memory (weight matrix): {shared_memory / (1024 ** 2):.1f} MB")
    print(f"Overhead factor: {overhead_factor}x")
    
    # Choose memory limit based on available hardware
    if gpu_memory is not None:
        # Use more conservative memory allocation (65% instead of 75%) for long-running simulations
        # This prevents memory fragmentation issues that occur around simulation 800-900
        usable_memory = gpu_memory * 0.65
        print(f"Using GPU memory constraint: {usable_memory / (1024 ** 3):.2f} GB ({gpu_memory / (1024 ** 3):.0f}GB total)")
        print("  Using conservative 65% limit to prevent memory fragmentation in long-running simulations")
    else:
        # Use system memory with safety margin
        usable_memory = min(available_memory * 0.4, 16 * (1024 ** 3))
        print(f"Using system memory constraint: {usable_memory / (1024 ** 3):.2f} GB")
    
    # Calculate memory-constrained concurrent size
    available_for_concurrent = usable_memory - shared_memory
    if available_for_concurrent <= 0:
        print("Warning: Shared memory exceeds available memory, using minimal concurrent size")
        max_memory_concurrent = 1
    else:
        max_memory_concurrent = max(1, int(available_for_concurrent / bytes_per_concurrent_sim))
    
    print(f"Memory-limited max concurrent: {max_memory_concurrent}")
    
    # Start with device-based baseline for async streaming
    if gpu_memory is not None:
        if n_gpu_devices == 1:
            # Single GPU: moderate concurrency for stability
            baseline_concurrent = max(4, n_devices * 3)
            print(f"Baseline concurrent (single GPU): {baseline_concurrent}")
        else:
            # Multi-GPU: higher concurrency to utilize parallel processing
            baseline_concurrent = max(8, n_devices * 4)
            print(f"Baseline concurrent (multi-GPU): {baseline_concurrent}")
    else:
        # CPU: lower concurrency
        baseline_concurrent = max(2, n_devices * 2)
        print(f"Baseline concurrent (CPU): {baseline_concurrent}")
    
    # Choose reasonable upper limit based on hardware
    if gpu_memory is not None:
        if n_gpu_devices == 1:
            reasonable_upper_limit = min(16, max_concurrent)  # Conservative for single GPU
        else:
            reasonable_upper_limit = min(16 * n_gpu_devices, max_concurrent)  # Higher for multi-GPU
    else:
        reasonable_upper_limit = min(8, max_concurrent)  # Lower for CPU
    
    # Balance memory constraint with baseline and upper limits
    memory_limited_concurrent = min(max_memory_concurrent, reasonable_upper_limit)
    optimal_concurrent = max(baseline_concurrent, memory_limited_concurrent)
    
    print(f"Initial optimal concurrent: {optimal_concurrent}")
    
    # Special case: if we have very few simulations, limit concurrency
    if total_simulations <= optimal_concurrent:
        optimal_concurrent = total_simulations
        print(f"Limited by total simulations: {optimal_concurrent}")
    
    # Ensure we don't exceed the hard limit
    optimal_concurrent = min(optimal_concurrent, max_concurrent)
    
    # Final safety checks with more conservative limits for large simulations
    if gpu_memory is not None:
        if n_gpu_devices == 1:
            optimal_concurrent = max(2, optimal_concurrent)  # Minimum of 2 for single GPU
            # More conservative max for large simulations to prevent OOM
            if total_simulations >= 1024:
                optimal_concurrent = min(optimal_concurrent, 8)  # Max of 8 for large sims on single GPU
            else:
                optimal_concurrent = min(optimal_concurrent, 16)  # Max of 16 for normal sims on single GPU
            print(f"Final concurrent size (single GPU): {optimal_concurrent}")
        else:
            optimal_concurrent = max(4, optimal_concurrent)  # Minimum of 4 for multi-GPU
            optimal_concurrent = min(optimal_concurrent, 16*n_gpu_devices)  # Max of 16*n_gpus for multi-GPU
            print(f"Final concurrent size (multi-GPU): {optimal_concurrent}")
    else:
        optimal_concurrent = max(1, optimal_concurrent)  # Minimum of 1 for CPU
        optimal_concurrent = min(optimal_concurrent, 8)  # Max of 8 for CPU
        print(f"Final concurrent size (CPU): {optimal_concurrent}")
    
    print(f'Final optimal max_concurrent for async streaming: {optimal_concurrent}')
    return optimal_concurrent
