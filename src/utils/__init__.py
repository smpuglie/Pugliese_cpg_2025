"""
Utility functions for VNC simulations.

This package contains various utility modules:
- sim_utils: Core simulation utilities and helper functions
- plot_utils: Plotting and visualization utilities
- shuffle_utils: Network shuffling utilities
- path_utils: Path and configuration utilities
- io_dict_to_hdf5: I/O utilities for HDF5 format
"""

from .sim_utils import (
    sample_trunc_normal, set_sizes, compute_oscillation_score,
    load_W, load_wTable, make_input
)
from .shuffle_utils import (
    shuffle_W, full_shuffle, extract_shuffle_indices
)
from .plot_utils import (
    get_module_colors, plot_R_heatmap, get_active_data
)

__all__ = [
    'sample_trunc_normal',
    'set_sizes', 
    'compute_oscillation_score',
    'load_W',
    'load_wTable',
    'make_input',
    'shuffle_W',
    'full_shuffle', 
    'extract_shuffle_indices',
    'get_module_colors',
    'plot_R_heatmap',
    'get_active_data',
]
