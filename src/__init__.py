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

# Import main simulation functions
from .simulation.vnc_sim import run_vnc_simulation, prepare_neuron_params, prepare_sim_params, parse_simulation_config
from .memory.checkpointing import (
    CheckpointState, save_checkpoint, load_checkpoint,
    find_latest_checkpoint, cleanup_old_checkpoints
)
from .data.data_classes import NeuronParams, SimParams, SimulationConfig, Pruning_state

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