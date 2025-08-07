"""
VNC Recurrent Neural Network Package

A high-performance, biologically-inspired recurrent neural network implementation
for simulating neural dynamics in the ventral nerve cord (VNC) of insects.

This package provides:
- Fast JAX-based RNN simulation with GPU/TPU acceleration support
- Multiple activation functions and input/noise generators
- Batch simulation capabilities for parallel processing
- Deterministic random number generation with JAX PRNG
- Comprehensive documentation and examples

Example usage:
    >>> import jax.random as random
    >>> from dt_vnc_rnn_jax import create_random_network, InputFunction
    >>>
    >>> # Create a simple weight matrix
    >>> key = random.PRNGKey(42)
    >>> weights = random.normal(key, (50, 50)) * 0.1
    >>>
    >>> network = create_random_network(50, dt=0.001, weights=weights, random_seed=42)
    >>> input_func = InputFunction('constant', 50, 1000, amplitude=10.0)
    >>> firing_rates = network.simulate(1000, input_func=input_func)
    >>> print(f"Simulation output: {firing_rates.shape}")

For detailed examples, see example_usage.py
For performance benchmarks, see benchmark.py
"""

from .vnc_network import VNCRecurrentNetwork, create_random_network
from .activation_functions import ActivationFunction
from .input_functions import InputFunction
from .noise_functions import NoiseFunction

__all__ = [
    "VNCRecurrentNetwork",
    "ActivationFunction",
    "InputFunction",
    "NoiseFunction",
    "create_random_network",
]
