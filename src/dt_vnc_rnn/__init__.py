"""
VNC Recurrent Neural Network Package

A high-performance, biologically-inspired recurrent neural network implementation
for simulating neural dynamics in the ventral nerve cord (VNC) of insects.

This package provides:
- Fast NumPy-based RNN simulation with sparse connectivity support
- Multiple activation functions and input/noise generators
- Batch simulation capabilities for parallel processing
- Comprehensive documentation and examples

Example usage:
    >>> from dt_vnc_rnn import create_random_network, InputFunction
    >>> network = create_random_network(50, dt=0.001)
    >>> input_func = lambda: InputFunction.constant(10.0, 50, 1000)
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
