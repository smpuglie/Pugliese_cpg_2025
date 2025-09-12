"""
VNC Simulation Package

This package provides a modular framework for running VNC neural network simulations.
The package is organized into the following submodules:

- simulation: Core simulation functionality
- utils: Utility functions for simulation, plotting, and I/O
- data: Data structures and classes
- memory: Memory management and checkpointing
- run_hydra: Main Hydra-based runner for simulations
"""

# Import data classes (no external dependencies)
from .data.data_classes import NeuronParams, SimParams, SimulationConfig, Pruning_state, CheckpointState

# Import functions with dependencies conditionally
def _import_optional():
    """Import functions that require external dependencies."""
    try:
        from .simulation.vnc_sim import run_vnc_simulation, prepare_neuron_params, prepare_sim_params, parse_simulation_config
        from .memory.checkpointing import (
            save_checkpoint, load_checkpoint,
            find_latest_checkpoint, cleanup_old_checkpoints
        )
        return {
            'run_vnc_simulation': run_vnc_simulation,
            'prepare_neuron_params': prepare_neuron_params,
            'prepare_sim_params': prepare_sim_params,
            'parse_simulation_config': parse_simulation_config,
            'save_checkpoint': save_checkpoint,
            'load_checkpoint': load_checkpoint,
            'find_latest_checkpoint': find_latest_checkpoint,
            'cleanup_old_checkpoints': cleanup_old_checkpoints,
        }
    except ImportError as e:
        import warnings
        warnings.warn(f"Optional dependencies not available: {e}", ImportWarning)
        return {}

# Try to import optional functions
_optional_imports = _import_optional()
globals().update(_optional_imports)

__all__ = [
    'run_vnc_simulation',
    'prepare_neuron_params',
    'prepare_sim_params', 
    'parse_simulation_config',
    'NeuronParams', 
    'SimParams',
    'SimulationConfig',
    'Pruning_state',
    'CheckpointState',
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint',
    'cleanup_old_checkpoints',
]