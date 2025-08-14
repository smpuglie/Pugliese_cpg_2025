"""
Activation functions for the VNC Recurrent Neural Network.

This module provides a collection of activation functions with individual neuron gains
and maximum firing rates. All functions support vectorized operations and maintain
the specified output ranges based on neuron-specific parameters.
"""

from typing import Callable
import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def scaled_tanh_relu(x: Array, relative_gains: Array, max_rates: Array) -> Array:
    """
    Scaled tanh followed by ReLU activation function.

    Args:
        x: Input values, shape (..., N)
        relative_gains: Neuron gains (slope parameter), shape (N,)
        max_rates: Maximum firing rates, shape (N,)

    Returns:
        Activated values in range [0, max_rates], shape (..., N)
    """
    return jnp.maximum(0, max_rates * jnp.tanh(relative_gains * x / max_rates))


@jax.jit
def tanh_scaled(x: Array, relative_gains: Array, max_rates: Array) -> Array:
    """
    Scaled hyperbolic tangent activation function.

    Args:
        x: Input values, shape (..., N)
        relative_gains: Neuron gains (slope parameter), shape (N,)
        max_rates: Maximum firing rates, shape (N,)

    Returns:
        Activated values in range [0, max_rates], shape (..., N)
    """
    return max_rates * (jnp.tanh(relative_gains * x / max_rates) + 1) / 2


@jax.jit
def sigmoid_scaled(x: Array, relative_gains: Array, max_rates: Array) -> Array:
    """
    Scaled sigmoid activation function.

    Args:
        x: Input values, shape (..., N)
        relative_gains: Neuron gains (slope parameter), shape (N,)
        max_rates: Maximum firing rates, shape (N,)

    Returns:
        Activated values in range [0, max_rates], shape (..., N)
    """
    # Clip input to prevent overflow in exp
    clipped_x = jnp.clip(relative_gains * x / max_rates, -500, 500)
    return max_rates / (1 + jnp.exp(-clipped_x))


@jax.jit
def linear_clipped(x: Array, relative_gains: Array, max_rates: Array) -> Array:
    """
    Linear activation function with clipping.

    Args:
        x: Input values, shape (..., N)
        relative_gains: Neuron gains (slope parameter), shape (N,)
        max_rates: Maximum firing rates, shape (N,)

    Returns:
        Activated values in range [0, max_rates], shape (..., N)
    """
    return jnp.clip(relative_gains * x / max_rates, 0, max_rates)


@jax.jit
def relu_scaled(x: Array, relative_gains: Array, max_rates: Array) -> Array:
    """
    Scaled ReLU activation function with saturation.

    Args:
        x: Input values, shape (..., N)
        relative_gains: Neuron gains (slope parameter), shape (N,)
        max_rates: Maximum firing rates, shape (N,)

    Returns:
        Activated values in range [0, max_rates], shape (..., N)
    """
    return linear_clipped(x, relative_gains, max_rates)


@jax.jit
def get_activation_function(func_name: str) -> Callable:
    """
    Get activation function by name.

    Args:
        func_name: Name of the activation function

    Returns:
        Activation function callable

    Raises:
        ValueError: If activation function name is unknown
    """
    func_map = {
        "scaled_tanh_relu": scaled_tanh_relu,
        "tanh_scaled": tanh_scaled,
        "sigmoid_scaled": sigmoid_scaled,
        "relu_scaled": relu_scaled,
        "linear_clipped": linear_clipped,
    }

    if func_name not in func_map:
        raise ValueError(f"Unknown activation function: {func_name}")

    return func_map[func_name]
