"""
VNC Recurrent Neural Network Package - Functional Implementation

A high-performance, biologically-inspired recurrent neural network implementation
for simulating neural dynamics in the ventral nerve cord (VNC) of insects.

This package provides:
- Fast JAX-based RNN simulation with GPU/TPU acceleration support
- Functional programming approach with immutable data structures
- Multiple activation functions and input/noise generators
- Batch simulation capabilities for parallel processing
- Deterministic random number generation with JAX PRNG
- Comprehensive documentation and examples

Example usage:
    >>> import jax.random as random
    >>> import numpy as np
    >>> from dt_vnc_rnn import create_network_params, simulate_network, InputFunction
    >>>
    >>> # Create a simple weight matrix
    >>> weights = np.random.randn(50, 50) * 0.1
    >>>
    >>> # Create network parameters
    >>> params = create_network_params(
    ...     num_neurons=50,
    ...     dt=0.001,
    ...     gains=1.0,
    ...     max_firing_rates=1.0,
    ...     thresholds=0.5,
    ...     time_constants=10.0,
    ...     weights=weights
    ... )
    >>>
    >>> # Create input function
    >>> input_func = InputFunction('constant', 50, 1000, amplitude=10.0)
    >>>
    >>> # Run simulation
    >>> firing_rates = simulate_network(params, 1000, input_func)
    >>> print(f"Simulation output: {firing_rates.shape}")

For detailed examples, see example_functional_vnc.py
"""

# Core functional network components
from .vnc_network import (
    NetworkParams,
    create_network_params,
    create_random_network_params,
    update_network_params,
    simulate_network,
    process_parameter_stats,
    get_network_info,
    ensure_array_shape,
    get_activation_function,
)

# Activation functions (standalone functions)
from .activation_functions import (
    scaled_tanh_relu,
    tanh_scaled,
    sigmoid_scaled,
    relu_scaled,
    linear_clipped,
)

# Input functions (still class-based for now)
from .input_functions import InputFunction

__all__ = [
    # Core network components
    "NetworkParams",
    "create_network_params",
    "create_random_network_params",
    "update_network_params",
    "simulate_network",
    "process_parameter_stats",
    "get_network_info",
    "ensure_array_shape",
    "get_activation_function",
    # Activation functions
    "scaled_tanh_relu",
    "tanh_scaled",
    "sigmoid_scaled",
    "relu_scaled",
    "linear_clipped",
    # Input functions
    "InputFunction",
]
