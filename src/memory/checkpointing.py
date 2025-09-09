"""
Checkpointing functionality for VNC simulations.

This module provides comprehensive checkpointing and resume capabilities
for long-running neural network simulations, including:
- Automatic checkpoint creation at configurable intervals
- Resume functionality with complete state preservation
- Memory-efficient storage using HDF5, YAML, and sparse formats
- CPU-first loading strategy for compressed HDF5 reliability
- Integration with OmegaConf configuration management
"""

import time
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
import yaml
import jax
import jax.numpy as jnp
import sparse
from omegaconf import OmegaConf
from natsort import natsorted
from src.utils import io_dict_to_hdf5 as ioh5
from src.data.data_classes import NeuronParams, SimParams, SimulationConfig, Pruning_state, CheckpointState


def extract_neuron_params_batch(neuron_params: NeuronParams, start_idx: int, end_idx: int) -> NeuronParams:
    """
    Extract a batch slice from neuron_params to save memory in checkpoints.
    
    Args:
        neuron_params: Full neuron parameters
        start_idx: Start index of the batch (inclusive)
        end_idx: End index of the batch (exclusive)
        
    Returns:
        NeuronParams containing only the specified batch slice
    """
    if neuron_params is None:
        return None
    
    # Create a new NeuronParams with only the batch slice
    batch_params = {}
    
    for field_name, field_value in neuron_params._asdict().items():
        if field_value is None:
            batch_params[field_name] = None
        elif field_name == 'input_currents':
            # input_currents shape: (n_stim_configs, n_param_sets, n_neurons)
            # Extract the batch slice from the n_param_sets dimension
            batch_params[field_name] = field_value[:, start_idx:end_idx, :]
        elif field_name == 'seeds':
            # seeds shape: (n_param_sets,)
            # Extract the batch slice
            batch_params[field_name] = field_value[start_idx:end_idx]
        else:
            # For other fields (W, tau, a, threshold, fr_cap, indices), keep as-is
            # These are typically shared across all parameter sets
            batch_params[field_name] = field_value
    
    return NeuronParams(**batch_params)


def merge_neuron_params_batch(full_neuron_params: NeuronParams, 
                             batch_neuron_params: NeuronParams, 
                             start_idx: int, end_idx: int) -> NeuronParams:
    """
    Merge batch neuron_params back into the full neuron_params structure.
    
    Args:
        full_neuron_params: Complete neuron parameters (may have modified batch data)
        batch_neuron_params: Batch-specific neuron parameters from checkpoint
        start_idx: Start index where batch data should be placed (inclusive)
        end_idx: End index where batch data should be placed (exclusive)
        
    Returns:
        Updated NeuronParams with the batch data merged in
    """
    if full_neuron_params is None or batch_neuron_params is None:
        return full_neuron_params
    
    # Create a copy of the full parameters
    updated_params = {}
    
    for field_name, full_value in full_neuron_params._asdict().items():
        batch_value = getattr(batch_neuron_params, field_name)
        
        if batch_value is None:
            # Keep the full value if batch value is None
            updated_params[field_name] = full_value
        elif field_name == 'input_currents' and full_value is not None:
            # Update the batch slice in input_currents
            # input_currents shape: (n_stim_configs, n_param_sets, n_neurons)
            updated_currents = full_value.at[:, start_idx:end_idx, :].set(batch_value)
            updated_params[field_name] = updated_currents
        elif field_name == 'seeds' and full_value is not None:
            # Update the batch slice in seeds
            # seeds shape: (n_param_sets,)
            updated_seeds = full_value.at[start_idx:end_idx].set(batch_value)
            updated_params[field_name] = updated_seeds
        else:
            # For other fields, prefer the batch value if available, otherwise keep full value
            updated_params[field_name] = batch_value if batch_value is not None else full_value
    
    return NeuronParams(**updated_params)


def save_checkpoint(checkpoint_state: CheckpointState, checkpoint_path: Path, 
                   metadata: Dict[str, Any] = None, results: List[jnp.ndarray] = None,
                   batch_start_idx: Optional[int] = None, batch_end_idx: Optional[int] = None,
                   compression_opts: int = 5):
    """Save simulation checkpoint to disk using HDF5, YAML, and sparse NPZ files.
    
    Unified function that handles both batch and streaming checkpoint modes.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint type
    is_streaming = checkpoint_state.checkpoint_type == "streaming"
    
    # Save metadata as YAML for easy inspection
    if is_streaming:
        metadata_dict = {
            "checkpoint_type": "streaming",
            "total_simulations": len(checkpoint_state.completed_simulations or set()) + len(checkpoint_state.failed_sims or set()),
            "completed_simulations": list(checkpoint_state.completed_simulations or set()),
            "failed_simulations": list(checkpoint_state.failed_sims or set()),
            "timestamp": float(time.time()),
            "progress_info": dict(checkpoint_state.progress_info or {}),
        }
    else:
        metadata_dict = {
            "checkpoint_type": "batch",
            "batch_index": int(checkpoint_state.batch_index),
            "completed_batches": int(checkpoint_state.completed_batches),
            "total_batches": int(checkpoint_state.total_batches),
            "timestamp": float(time.time()),
            "has_mini_circuits": checkpoint_state.accumulated_mini_circuits is not None,
            "has_pruning_state": checkpoint_state.pruning_state is not None,
            "n_result_batches": checkpoint_state.n_result_batches,
        }
    
    if metadata:
        metadata_dict.update(metadata)
    
    # Save accumulated results as separate sparse NPZ files
    results_dir = checkpoint_path.parent / f"results_{checkpoint_path.stem}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the metadata YAML file (with all configuration data) in the same directory as results
    metadata_path = results_dir / f"{checkpoint_path.stem}.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata_dict, f, default_flow_style=False, indent=2)
    

    # Handle batch mode results (legacy compatibility)
    if results is not None and not is_streaming:
        result_path = results_dir / f"batch_{batch_start_idx}.npz"
        sparse.save_npz(result_path, sparse.COO.from_numpy(jnp.asarray(results)))

    # Create dictionary structure for checkpoint data (only JAX arrays go in HDF5)
    checkpoint_dict = {}
    
    if is_streaming:
        # Convert streaming results to saveable format
        if checkpoint_state.results_dict:
            results_data = {}
            for sim_idx, result_array in checkpoint_state.results_dict.items():
                results_data[f"sim_{sim_idx}"] = result_array
            checkpoint_dict["results"] = results_data
        
        # Convert states_dict to saveable format
        if checkpoint_state.states_dict:
            states_data = {}
            for sim_idx, state in checkpoint_state.states_dict.items():
                # Convert Pruning_state to dict
                # Note: Arrays will be saved as-is but should be loaded back to CPU
                state_dict = state._asdict() if hasattr(state, '_asdict') else state
                states_data[f"sim_{sim_idx}"] = state_dict
            checkpoint_dict["states"] = states_data
        
        # Convert mini_circuits_dict to saveable format
        if checkpoint_state.mini_circuits_dict:
            mini_circuits_data = {}
            for sim_idx, mini_circuit in checkpoint_state.mini_circuits_dict.items():
                mini_circuits_data[f"sim_{sim_idx}"] = mini_circuit
            checkpoint_dict["mini_circuits"] = mini_circuits_data
        
        # Convert w_masks_dict to saveable format
        if checkpoint_state.w_masks_dict:
            w_masks_data = {}
            for sim_idx, w_mask in checkpoint_state.w_masks_dict.items():
                w_masks_data[f"sim_{sim_idx}"] = w_mask
            checkpoint_dict["w_masks"] = w_masks_data
            
    else:
        # Batch mode fields
        checkpoint_dict.update({
            "batch_index": checkpoint_state.batch_index,
            "completed_batches": checkpoint_state.completed_batches,
            "total_batches": checkpoint_state.total_batches,
            "n_result_batches": checkpoint_state.n_result_batches,
        })
        
        if checkpoint_state.neuron_params:
            checkpoint_dict["neuron_params"] = checkpoint_state.neuron_params._asdict()
    
    # Save checkpoint data using HDF5 (in the same directory as results)
    checkpoint_h5_path = results_dir / f"{checkpoint_path.stem}.h5"
    ioh5.save(checkpoint_h5_path, checkpoint_dict, compression_opts=compression_opts)

    if is_streaming:
        print(f"Streaming checkpoint saved: {checkpoint_h5_path}")
        print(f"Completed {len(checkpoint_state.completed_simulations or set())} simulations")
    else:
        print(f"Batch checkpoint saved: {checkpoint_h5_path}")
        print(f"Completed {checkpoint_state.completed_batches}/{checkpoint_state.total_batches} batches")
    print(f"Metadata saved: {metadata_path}")
    


def load_checkpoint(checkpoint_path: Path, base_name: str, full_neuron_params: Optional[NeuronParams] = None) -> Tuple[CheckpointState, Optional[NeuronParams], Dict[str, Any]]:
    """
    Load simulation checkpoint from disk using HDF5, YAML, and sparse NPZ files.
    
    Uses CPU-first loading strategy for maximum reliability with compressed HDF5 files.
    Since HDF5 compression makes uncompressed size unpredictable, always starts with
    CPU loading to avoid GPU memory allocation failures.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        base_name: Base name of the checkpoint
        full_neuron_params: Complete neuron parameters to merge batch data into.
                          If provided, the loaded batch neuron_params will be merged into this.
                          If None, the loaded neuron_params will be returned as-is.
    
    Returns:
        Tuple of (CheckpointState, neuron_params, metadata)
    """
    # Check memory status before loading checkpoint
    try:
        from src.memory.adaptive_memory import monitor_memory_usage
        memory_info = monitor_memory_usage()
        gpu_usage = memory_info.get('gpu_usage_percent', 0)
        if gpu_usage > 80:
            print(f"⚠️  High GPU memory usage ({gpu_usage:.1f}%) detected before checkpoint loading")
            print("🧹 Performing preventive memory cleanup...")
            import jax
            jax.clear_caches()
            import gc
            for _ in range(3):
                gc.collect()
    except Exception as e:
        print(f"Warning: Could not check memory status: {e}")
    
    # Handle both old and new checkpoint locations
    # First try the new location (inside results directory)
    checkpoint_h5_path = checkpoint_path / f"{base_name}.h5"

    if not checkpoint_h5_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_h5_path}")
    
    # Load metadata from YAML (now stored in results directory)
    metadata_yaml_path = checkpoint_path / f"{base_name}.yaml"
    metadata = {}
    
    if metadata_yaml_path.exists():
        with open(metadata_yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)

    # Load checkpoint data from HDF5 with CPU-first strategy for reliability
    # Always use CPU-first for compressed HDF5 files since size prediction is unreliable
    checkpoint_size = checkpoint_h5_path.stat().st_size
    size_mb = checkpoint_size / (1024 * 1024)
    
    print(f"📥 Loading {size_mb:.1f}MB checkpoint using CPU-first strategy for maximum reliability...")
    
    """
    MEMORY PLACEMENT STRATEGY:
    - All checkpoint data is loaded and kept on CPU devices
    - This includes completed simulation results, states, circuits, and masks
    - The async simulation initialization will handle moving necessary data to GPU as needed
    - This ensures: 1) Data preservation when old checkpoints are deleted
                   2) Minimal GPU memory usage during checkpoint loading  
                   3) Optimal placement during simulation startup
    """
    
    try:
        # Always load on CPU first for compressed checkpoints
        checkpoint_dict = ioh5.load(checkpoint_h5_path, enable_jax=False)
        
        # Convert to JAX arrays selectively and efficiently with memory management
        print("🔄 Converting essential data to JAX arrays with memory-safe GPU placement...")
        import jax  # Ensure jax is available in this scope
        
        # CRITICAL: Clear any existing compilation cache before conversion
        # This prevents conflicts when loading pre-computed checkpoint data
        jax.clear_caches()
        
        # Determine checkpoint type early to optimize array placement
        checkpoint_type = metadata.get("checkpoint_type", "batch")
        is_streaming_check = checkpoint_type == "streaming"
        
        for key, value in checkpoint_dict.items():
            if key in ['batch_index', 'completed_batches', 'total_batches', 'n_result_batches']:
                # Keep metadata as regular Python types
                continue
            elif is_streaming_check and key in ['results', 'states', 'mini_circuits', 'w_masks']:
                # CRITICAL: For streaming checkpoints, convert completed data but keep on CPU
                # This data must be preserved but doesn't need GPU memory for new simulations
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subvalue is not None and hasattr(subvalue, '__array__'):
                            try:
                                # Force CPU placement for completed simulation data
                                checkpoint_dict[key][subkey] = jax.device_put(jnp.asarray(subvalue), jax.devices("cpu")[0])
                            except Exception as convert_e:
                                print(f"⚠️  Warning: Could not convert {key}.{subkey} to JAX on CPU: {convert_e}")
                                pass
            elif isinstance(value, dict):
                # Handle nested dictionaries (like results, states, etc.)
                for subkey, subvalue in value.items():
                    if subvalue is not None and hasattr(subvalue, '__array__'):
                        try:
                            # Use device_put for explicit placement control
                            checkpoint_dict[key][subkey] = jax.device_put(jnp.asarray(subvalue), jax.devices("cpu")[0])
                        except Exception as convert_e:
                            print(f"⚠️  Warning: Could not convert {key}.{subkey} to JAX: {convert_e}")
                            # Keep as numpy array
                            pass
            elif value is not None and hasattr(value, '__array__'):
                try:
                    # Place on CPU initially for large arrays
                    checkpoint_dict[key] = jax.device_put(jnp.asarray(value), jax.devices("cpu")[0])
                except Exception as convert_e:
                    print(f"⚠️  Warning: Could not convert {key} to JAX: {convert_e}")
                    # Keep as numpy array
                    pass
        
        # Final cleanup after all conversions
        import gc
        gc.collect()
        
        print("✅ Checkpoint loaded successfully using memory-safe CPU-first strategy")
        
    except Exception as cpu_e:
        print(f"❌ CPU-first loading failed: {cpu_e}")
        raise cpu_e
    
    # Determine checkpoint type from metadata
    checkpoint_type = metadata.get("checkpoint_type", "batch")
    is_streaming = checkpoint_type == "streaming"
    
    if is_streaming:
        # Handle streaming mode - get metadata from YAML
        completed_simulations = set(metadata.get("completed_simulations", []))
        failed_simulations = set(metadata.get("failed_simulations", []))
        progress_info = metadata.get("progress_info", {})
        
        # CRITICAL: For streaming checkpoints, load completed data but keep it on CPU
        # This data must be preserved as old checkpoints get deleted, but doesn't need GPU memory
        results_dict = {}
        if "results" in checkpoint_dict:
            for key, result_array in checkpoint_dict["results"].items():
                sim_idx = int(key.replace("sim_", ""))
                # Keep results on CPU - they're not needed for new simulations but must be preserved
                results_dict[sim_idx] = jnp.asarray(result_array)  # Will stay on CPU from loading
        
        # Reconstruct states_dict - keep on CPU
        states_dict = {}
        if "states" in checkpoint_dict:
            for key, state_dict in checkpoint_dict["states"].items():
                sim_idx = int(key.replace("sim_", ""))
                # Convert dict back to Pruning_state - explicitly place on CPU
                state_converted = {}
                for field_name, field_value in state_dict.items():
                    if field_value is not None:
                        if 'mask' in field_name:
                            # Explicitly place boolean arrays on CPU
                            state_converted[field_name] = jax.device_put(
                                jnp.asarray(field_value, dtype=jnp.bool_), 
                                jax.devices("cpu")[0]
                            )
                        else:
                            # Explicitly place all other arrays on CPU
                            state_converted[field_name] = jax.device_put(
                                jnp.asarray(field_value), 
                                jax.devices("cpu")[0]
                            )
                    else:
                        state_converted[field_name] = None
                states_dict[sim_idx] = Pruning_state(**state_converted)
        
        # Reconstruct mini_circuits_dict - keep on CPU
        mini_circuits_dict = {}
        if "mini_circuits" in checkpoint_dict:
            for key, mini_circuit in checkpoint_dict["mini_circuits"].items():
                sim_idx = int(key.replace("sim_", ""))
                mini_circuits_dict[sim_idx] = jnp.asarray(mini_circuit, dtype=jnp.bool_)
        
        # Reconstruct w_masks_dict - keep on CPU
        w_masks_dict = {}
        if "w_masks" in checkpoint_dict:
            for key, w_mask in checkpoint_dict["w_masks"].items():
                sim_idx = int(key.replace("sim_", ""))
                w_masks_dict[sim_idx] = jnp.asarray(w_mask, dtype=jnp.bool_)
        
        # Log what we loaded and where it's stored
        print(f"📊 Loaded {len(results_dict)} results (kept on CPU)")
        print(f"📊 Loaded {len(states_dict)} states (kept on CPU)")
        print(f"📊 Loaded {len(mini_circuits_dict)} circuits (kept on CPU)")
        print(f"📊 Loaded {len(w_masks_dict)} masks (kept on CPU)")
        
        # Create streaming CheckpointState
        checkpoint_state = CheckpointState(
            batch_index=0,  # Not applicable for streaming
            completed_batches=0,  # Not applicable for streaming
            total_batches=0,  # Not applicable for streaming
            neuron_params=None,  # Not stored for streaming
            n_result_batches=0,  # Not applicable for streaming
            checkpoint_type="streaming",
            completed_simulations=completed_simulations,
            results_dict=results_dict,  # Loaded but on CPU
            states_dict=states_dict,    # Loaded but on CPU
            mini_circuits_dict=mini_circuits_dict,  # Loaded but on CPU
            w_masks_dict=w_masks_dict,  # Loaded but on CPU
            failed_sims=failed_simulations,
            progress_info=progress_info
        )
        
        print(f"Streaming checkpoint loaded: {checkpoint_h5_path}")
        print(f"Resuming with {len(completed_simulations)} completed simulations")
        print("✅ Memory optimized: completed data loaded on CPU (preserved but not using GPU memory)")
        
        return checkpoint_state, None, metadata
        
    else:
        # Handle batch mode (existing logic)
        
        # Convert relevant arrays to JAX arrays manually
        for key in ['batch_index', 'completed_batches', 'total_batches', 'n_result_batches']:
            if key in checkpoint_dict:
                checkpoint_dict[key] = int(checkpoint_dict[key])
        
        # Extract basic fields
        basic_fields = {
            'batch_index': int(checkpoint_dict['batch_index']),
            'completed_batches': int(checkpoint_dict['completed_batches']),
            'total_batches': int(checkpoint_dict['total_batches']),
            'n_result_batches': int(checkpoint_dict['n_result_batches']),
        }
        
        
        # Reconstruct accumulated mini circuits if available
        accumulated_mini_circuits = None
        if metadata.get("has_mini_circuits", False) and "accumulated_mini_circuits" in checkpoint_dict:
            accumulated_mini_circuits = []
            mini_circuits_dict = checkpoint_dict["accumulated_mini_circuits"]
            # Sort by batch number to maintain order
            sorted_keys = sorted(mini_circuits_dict.keys(), key=lambda x: int(x.split('_')[1]))
            for key in sorted_keys:
                accumulated_mini_circuits.append(jnp.asarray(mini_circuits_dict[key]))
        
        del full_neuron_params
        gc.collect()  # Free memory before loading neuron_params
        # Create the loaded neuron_params (might be a batch extract)
        neuron_params = NeuronParams(**checkpoint_dict['neuron_params'])

        # Reconstruct pruning_state if available
        pruning_state = None
        if metadata.get("has_pruning_state", False) and "pruning_state" in checkpoint_dict:
            pruning_state_dict = checkpoint_dict["pruning_state"]
            # Convert arrays to JAX arrays and explicitly place on CPU
            converted_pruning_state = {}
            for key, value in pruning_state_dict.items():
                if value is not None:
                    # Explicitly place all pruning state arrays on CPU
                    converted_pruning_state[key] = jax.device_put(
                        jnp.asarray(value), 
                        jax.devices("cpu")[0]
                    )
                else:
                    converted_pruning_state[key] = None
            pruning_state = Pruning_state(**converted_pruning_state)
        
        # Create checkpoint state
        checkpoint_state = CheckpointState(
            **basic_fields,
            neuron_params=neuron_params,
            accumulated_mini_circuits=accumulated_mini_circuits,
            pruning_state=pruning_state,
            checkpoint_type="batch"
        )
        
        print(f"Batch checkpoint loaded: {checkpoint_h5_path}")
        print(f"Resuming from batch {checkpoint_state.batch_index} ({checkpoint_state.completed_batches}/{checkpoint_state.total_batches} batches completed)")
        
        return checkpoint_state, neuron_params, metadata


def find_latest_checkpoint(checkpoint_dir: Path, pattern: str = "checkpoint_*") -> Optional[Tuple[Path, str]]:
    """
    Find the latest checkpoint in a directory based on checkpoint number.
    
    Unified function that works for both batch and streaming checkpoints.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        pattern: Glob pattern for checkpoint files (without extension)
        
    Returns:
        Tuple of (checkpoint_path, base_name) or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory does not exist starting from scratch: {checkpoint_dir}")
        return None
    
    # Look for results directories that contain the checkpoints
    results_dirs = natsorted(list(checkpoint_dir.glob(f"results_{pattern}")))

    # Extract checkpoint numbers and find the latest
    latest_checkpoint = None
    base_name = None
    latest_number = -1
    
    # Check new-style checkpoints (in results directories)
    for results_dir in results_dirs:
        try:
            # Extract number from directory name like "results_checkpoint_5" or "results_checkpoint_batch_5"
            base_name_candidate = results_dir.name.replace('results_', '')
            
            # Handle both "checkpoint_123" and "checkpoint_batch_123" formats
            parts = base_name_candidate.split('_')
            if len(parts) >= 2:
                number_str = parts[-1]  # Last part should be the number
                checkpoint_number = int(number_str)
                
                # Check if the HDF5 file exists in this directory
                h5_file = results_dir / f"{base_name_candidate}.h5"
                if h5_file.exists() and checkpoint_number > latest_number:
                    latest_number = checkpoint_number
                    latest_checkpoint = h5_file.parent
                    base_name = base_name_candidate
        except (ValueError, IndexError):
            # Skip directories that don't match expected naming pattern
            continue

    if latest_checkpoint:
        return latest_checkpoint, base_name
    return None

def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3, pattern: str = "checkpoint_*"):
    """
    Remove old checkpoint files, keeping only the most recent ones.
    
    Unified function that works for both batch and streaming checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of most recent checkpoints to keep
        pattern: Pattern for checkpoint files (without extension)
    """
    if not checkpoint_dir.exists():
        return
    
    # Find all checkpoint sets (new format only)
    checkpoint_bases = set()
    
    # Check for result directories (which contain both HDF5 and YAML files)
    for result_dir in checkpoint_dir.glob(f"results_{pattern}"):
        base_name = result_dir.name.replace('results_', '')
        checkpoint_bases.add(base_name)
    
    if len(checkpoint_bases) <= keep_last_n:
        return  # Nothing to clean up
    
    # Sort by checkpoint number and keep only the latest ones
    sorted_bases = []
    for base in checkpoint_bases:
        try:
            # Handle both "checkpoint_123" and "checkpoint_batch_123" formats
            parts = base.split('_')
            if len(parts) >= 2:
                number_str = parts[-1]  # Last part should be the number
                checkpoint_number = int(number_str)
                sorted_bases.append((checkpoint_number, base))
        except (ValueError, IndexError):
            continue
    
    sorted_bases.sort(key=lambda x: x[0])  # Sort by checkpoint number
    
    # Remove old checkpoints
    to_remove = sorted_bases[:-keep_last_n]  # All except the last keep_last_n
    
    for checkpoint_number, base in to_remove:
        # Remove results directory (which contains both the HDF5 and YAML files)
        results_dir = checkpoint_dir / f"results_{base}"
        if results_dir.exists():
            import shutil
            shutil.rmtree(results_dir)
            print(f"Removed old checkpoint: {base}")

