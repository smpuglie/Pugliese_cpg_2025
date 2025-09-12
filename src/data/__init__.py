"""
Data structures and classes for VNC simulations.

This package contains:
- data_classes: Core data structures and NamedTuples for simulation parameters
"""

from .data_classes import (
    NeuronParams, SimParams, SimulationConfig, 
    Pruning_state, CheckpointState
)

__all__ = [
    'NeuronParams',
    'SimParams', 
    'SimulationConfig',
    'Pruning_state',
    'CheckpointState',
]
