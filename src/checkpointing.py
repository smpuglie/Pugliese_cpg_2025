"""
Checkpointing functionality for VNC simulations.

This module provides comprehensive checkpointing and resume capabilities
for long-running neural network simulations, including:
- Automatic checkpoint creation at configurable intervals
- Resume functionality with complete state preservation
- Memory-efficient storage using HDF5, YAML, and sparse formats
- Integration with OmegaConf configuration management
"""

import time
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
import yaml
import jax.numpy as jnp
import sparse
from omegaconf import OmegaConf
from natsort import natsorted
from . import io_dict_to_hdf5 as ioh5
from .data_classes import NeuronParams, SimParams, SimulationConfig, Pruning_state, CheckpointState


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
                   batch_start_idx: Optional[int] = None, batch_end_idx: Optional[int] = None):
    """Save simulation checkpoint to disk using HDF5, YAML, and sparse NPZ files."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metadata as YAML for easy inspection
    metadata_dict = {
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
    

    result_path = results_dir / f"batch_{batch_start_idx}.npz"
    sparse.save_npz(result_path, sparse.COO.from_numpy(jnp.asarray(results)))

        # Create dictionary structure for checkpoint data (only large arrays go in HDF5)
    checkpoint_dict = {
        "batch_index": checkpoint_state.batch_index,
        "completed_batches": checkpoint_state.completed_batches,
        "total_batches": checkpoint_state.total_batches,
        "n_result_batches": checkpoint_state.n_result_batches,
    }
    checkpoint_dict["neuron_params"] = checkpoint_state.neuron_params._asdict()
    # Save checkpoint data using HDF5 (in the same directory as results)
    checkpoint_h5_path = results_dir / f"{checkpoint_path.stem}.h5"
    ioh5.save(checkpoint_h5_path, checkpoint_dict)

    print(f"Checkpoint saved: {checkpoint_h5_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Completed {checkpoint_state.completed_batches}/{checkpoint_state.total_batches} batches")
    


def load_checkpoint(checkpoint_path: Path, base_name: str, full_neuron_params: Optional[NeuronParams] = None) -> Tuple[CheckpointState, List[jnp.ndarray], Dict[str, Any]]:
    """
    Load simulation checkpoint from disk using HDF5, YAML, and sparse NPZ files.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        full_neuron_params: Complete neuron parameters to merge batch data into.
                          If provided, the loaded batch neuron_params will be merged into this.
                          If None, the loaded neuron_params will be returned as-is.
    
    Returns:
        Tuple of (CheckpointState, accumulated_results, metadata)
    """
    # Handle both old and new checkpoint locations
    # First try the new location (inside results directory)
    checkpoint_h5_path = checkpoint_path / f"{base_name}.h5"

    if not checkpoint_h5_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_h5_path}")
    
    # Load metadata from YAML (now stored in results directory)
    metadata_yaml_path = checkpoint_path / f"{base_name}.yaml"
    metadata = {}
    
    if metadata_yaml_path.exists():
        metadata = OmegaConf.load(metadata_yaml_path)

    # Load checkpoint data from HDF5 (without automatic JAX conversion to handle strings)
    checkpoint_dict = ioh5.load(checkpoint_h5_path, enable_jax=True)

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
        # Convert arrays to JAX arrays
        converted_pruning_state = {}
        for key, value in pruning_state_dict.items():
            if value is not None:
                converted_pruning_state[key] = jnp.asarray(value)
            else:
                converted_pruning_state[key] = None
        pruning_state = Pruning_state(**converted_pruning_state)
    
    # Create checkpoint state
    checkpoint_state = CheckpointState(
        **basic_fields,
        neuron_params=neuron_params,
        accumulated_mini_circuits=accumulated_mini_circuits,
        pruning_state=pruning_state,
    )
    
    print(f"Checkpoint loaded: {checkpoint_h5_path}")
    print(f"Resuming from batch {checkpoint_state.batch_index} ({checkpoint_state.completed_batches}/{checkpoint_state.total_batches} batches completed)")
    
    return checkpoint_state, neuron_params, metadata


def find_latest_checkpoint(checkpoint_dir: Path, pattern: str = "checkpoint_batch_*") -> Optional[Path]:
    """
    Find the latest checkpoint in a directory based on batch number.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        pattern: Glob pattern for checkpoint files (without extension)
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory does not exist starting from scratch: {checkpoint_dir}")
        return None, None
    
    # Look for results directories that contain the checkpoints
    results_dirs = natsorted(list(checkpoint_dir.glob(f"results_{pattern}")))

    # Extract batch numbers and find the latest
    latest_checkpoint = None
    base_name = None
    latest_batch = -1
    
    # Check new-style checkpoints (in results directories)
    for results_dir in results_dirs:
        try:
            # Extract batch number from directory name like "results_checkpoint_batch_5"
            base_name = results_dir.name.replace('results_', '')
            batch_str = base_name.split('_')[-1]
            batch_num = int(batch_str)
            
            # Check if the HDF5 file exists in this directory
            h5_file = results_dir / f"{base_name}.h5"
            if h5_file.exists() and batch_num > latest_batch:
                latest_batch = batch_num
                latest_checkpoint = h5_file.parent
        except (ValueError, IndexError):
            # Skip directories that don't match expected naming pattern
            continue

    return latest_checkpoint, base_name

def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3, pattern: str = "checkpoint_batch_*"):
    """
    Remove old checkpoint files, keeping only the most recent ones.
    
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
    
    # Sort by batch number and keep only the latest ones
    sorted_bases = []
    for base in checkpoint_bases:
        try:
            batch_num = int(base.split('_')[-1])
            sorted_bases.append((batch_num, base))
        except (ValueError, IndexError):
            continue
    
    sorted_bases.sort(key=lambda x: x[0])  # Sort by batch number
    
    # Remove old checkpoints
    to_remove = sorted_bases[:-keep_last_n]  # All except the last keep_last_n
    
    for batch_num, base in to_remove:
        # Remove results directory (which contains both the HDF5 and YAML files)
        results_dir = checkpoint_dir / f"results_{base}"
        if results_dir.exists():
            import shutil
            shutil.rmtree(results_dir)
            print(f"Removed old checkpoint: {base}")

