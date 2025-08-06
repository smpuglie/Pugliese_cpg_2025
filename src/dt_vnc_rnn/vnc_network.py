"""
High-performance Ventral Nerve Cord (VNC) Recurrent Neural Network implementation.

This module implements a biologically-inspired RNN model for simulating neural dynamics
in the ventral nerve cord of insects, using efficient NumPy operations for maximum performance.

Mathematical Model:
    Continuous dynamics: τ * dr/dt = -r(t) + f(I_ext(t) + b * W @ r(t) - θ + η(t))
    Discrete update: r(t+dt) = (1-α) * r(t) + α * f(I_ext(t) + b * W @ r(t) - θ + η(t))
    where α = dt/τ
"""

import numpy as np
from scipy import sparse
from typing import Union, Callable, Optional
import warnings
from .activation_functions import ActivationFunction
from .input_functions import InputFunction
from .noise_functions import NoiseFunction


class VNCRecurrentNetwork:
    """
    High-performance Ventral Nerve Cord Recurrent Neural Network.

    This class implements a biologically-inspired RNN model with individual neuron
    parameters, sparse connectivity support, and optimized NumPy operations for
    maximum computational efficiency.

    Mathematical Model:
        Continuous: τ * dr/dt = -r(t) + f(I_ext(t) + W @ (r(t) + η_rec(t)) - θ + η_input(t))
        Discrete: r(t+dt) = (1-α) * r(t) + α * f(I_ext(t) + W @ (r(t) + η_rec(t)) - θ + η_input(t))
        where α = dt/τ

    Attributes:
        num_neurons (int): Number of neurons in the network
        dt (float): Simulation time step
        gains (np.ndarray): Individual neuron gains, shape (N,)
        max_firing_rates (np.ndarray): Maximum firing rates, shape (N,)
        thresholds (np.ndarray): Firing thresholds, shape (N,)
        time_constants (np.ndarray): Time constants, shape (N,)
        weights (np.ndarray or sparse matrix): Synaptic weights, shape (N, N)
        activation_func (Callable): Activation function
        alpha (np.ndarray): Precomputed dt/tau values, shape (N,)
        one_minus_alpha (np.ndarray): Precomputed (1-dt/tau) values, shape (N,)
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float,
        gains: Union[float, np.ndarray],
        max_firing_rates: Union[float, np.ndarray],
        thresholds: Union[float, np.ndarray],
        time_constants: Union[float, np.ndarray],
        weights: Union[np.ndarray, sparse.csr_array],
        relative_inputs_scale: float = 0.03,
        activation_function: str = "scaled_tanh_relu",
    ):
        """
        Initialize the VNC Recurrent Network.

        Args:
            num_neurons: Number of neurons in the network
            dt: Simulation time step
            gains: Individual neuron gains, scalar or shape (N,)
            max_firing_rates: Maximum firing rates, scalar or shape (N,)
            thresholds: Firing thresholds, scalar or shape (N,)
            time_constants: Time constants, scalar or shape (N,)
            weights: Synaptic weight matrix, shape (N, N)
            activation_function: Type of activation function
                                ('scaled_tanh_relu', 'tanh_scaled', 'sigmoid_scaled', 'relu_scaled', 'linear_clipped')

        Raises:
            ValueError: If parameter dimensions are incompatible
        """
        self.num_neurons = num_neurons
        self.dt = dt
        self.relative_inputs_scale = relative_inputs_scale

        # Convert scalar parameters to arrays
        self.gains = self._ensure_array_shape(gains, "gains")
        self.max_firing_rates = self._ensure_array_shape(max_firing_rates, "max_firing_rates")
        self.thresholds = self._ensure_array_shape(thresholds, "thresholds")
        self.time_constants = self._ensure_array_shape(time_constants, "time_constants")

        # Validate weight matrix
        if hasattr(weights, "shape"):
            if weights.shape != (num_neurons, num_neurons):
                raise ValueError(f"Weight matrix shape {weights.shape} != ({num_neurons}, {num_neurons})")
        self.weights = weights

        # Set activation function
        self.activation_func = self._get_activation_function(activation_function)

        # Precompute frequently used values for efficiency
        self.alpha = self.dt / self.time_constants
        self.one_minus_alpha = 1.0 - self.alpha

        # Validate alpha values
        if np.any(self.alpha > 1.0):
            warnings.warn("Some alpha values > 1.0, simulation may be unstable. Consider smaller dt.")
        if np.any(self.alpha <= 0.0):
            raise ValueError("Alpha values must be positive (dt and time_constants must be positive)")

    def _ensure_array_shape(self, param: Union[float, np.ndarray], name: str) -> np.ndarray:
        """Ensure parameter has correct shape (N,)."""
        param = np.asarray(param)
        if param.ndim == 0 or param.shape == (self.num_neurons,):
            return param
        else:
            raise ValueError(f"{name} shape {param.shape} != (1,) or ({self.num_neurons},)")

    def _get_activation_function(self, func_name: str) -> Callable:
        """Get activation function by name."""
        func_map = {
            "scaled_tanh_relu": ActivationFunction.scaled_tanh_relu,
            "tanh_scaled": ActivationFunction.tanh_scaled,
            "sigmoid_scaled": ActivationFunction.sigmoid_scaled,
            "relu_scaled": ActivationFunction.relu_scaled,
            "linear_clipped": ActivationFunction.linear_clipped,
        }

        if func_name not in func_map:
            raise ValueError(f"Unknown activation function: {func_name}")

        return func_map[func_name]

    def simulate(
        self,
        num_timesteps: int,
        input_func: Union[Callable, InputFunction] = lambda _: 0,
        noise_func_input: Union[Callable, NoiseFunction] = lambda _: 0,
        noise_func_recurrent: Union[Callable, NoiseFunction] = lambda _: 0,
        initial_state: Optional[np.ndarray] = None,
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Run simulation(s) of the network.

        Args:
            num_timesteps: Number of time steps to simulate
            input_func: Function returning external inputs
            noise_func_input: Function returning noise for input
            noise_func_recurrent: Function returning noise for recurrent activity
            initial_state: Initial firing rates. For single sim: shape (N,), for batch: shape (batch_size, N)
            batch_size: Number of parallel simulations (default 1 for single simulation)

        Returns:
            Firing rates array, shape (T+1, N) for single sim or (T+1, batch_size, N) for batch
        """
        # Handle initial state
        if initial_state is None:
            r = np.zeros((batch_size, self.num_neurons))
        else:
            r = np.asarray(initial_state).copy()
            if r.ndim == 1:
                r = r[np.newaxis, :]
            if r.shape != (batch_size, self.num_neurons):
                raise ValueError(f"Initial states shape {r.shape} != ({batch_size}, {self.num_neurons})")

        # Prepare storage
        firing_rates = np.zeros((num_timesteps + 1, batch_size, self.num_neurons))
        firing_rates[0] = r

        # Main simulation loop
        for t in range(num_timesteps):
            synaptic_input = (r + noise_func_recurrent(t)) @ self.weights.T

            # Total input
            total_input = (
                input_func(t) + self.relative_inputs_scale * synaptic_input - self.thresholds + noise_func_input(t)
            )

            # Apply activation function
            activated = self.activation_func(total_input, self.gains, self.max_firing_rates)

            # Update firing rates using Euler integration
            r = self.one_minus_alpha * r + self.alpha * activated

            # Store results
            firing_rates[t + 1] = r

        return firing_rates

    def set_parameters(self, **kwargs) -> None:
        """
        Update network parameters.

        Args:
            **kwargs: Parameter updates (gains, max_firing_rates, thresholds,
                     time_constants, weights, activation_function)
        """
        for param, value in kwargs.items():
            if param in ["gains", "max_firing_rates", "thresholds", "time_constants"]:
                setattr(self, param, self._ensure_array_shape(value, param))
            elif param == "weights":
                if hasattr(value, "shape") and value.shape != (self.num_neurons, self.num_neurons):
                    raise ValueError(
                        f"Weight matrix shape {value.shape} != " f"({self.num_neurons}, {self.num_neurons})"
                    )
                self.weights = value
            elif param == "activation_function":
                self.activation_func = self._get_activation_function(value)
            else:
                raise ValueError(f"Unknown parameter: {param}")

        # Recompute alpha values if time constants changed
        if "time_constants" in kwargs:
            self.alpha = self.dt / self.time_constants
            self.one_minus_alpha = 1.0 - self.alpha

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"VNCRecurrentNetwork(num_neurons={self.num_neurons}, dt={self.dt}, "
            f"activation={self.activation_func.__name__})"
        )


# Convenience functions for creating common network configurations
def process_parameter_stats(n_samples: int, stats: dict) -> np.ndarray:
    """
    Process parameter statistics by extracting means and standard deviations.

    Args:
        stats: Dictionary containing parameter statistics (mean, std)
    """
    if not stats:
        raise ValueError("Parameter statistics dictionary is empty")
    elif stats["distribution"] == "normal":
        mean = stats["mean"]
        std = stats["std"]
        return np.random.normal(mean, std, n_samples)
    else:
        raise ValueError(f"Unsupported distribution: {stats['distribution']}")


def create_random_network(
    num_neurons: int,
    dt: float,
    weights: Union[np.ndarray, sparse.csr_array],
    activation_function: str = "scaled_tanh_relu",
    gains_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    max_firing_rates_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    thresholds_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    time_constants_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    neuron_size_scale: Union[list, np.ndarray] = [],
    relative_inputs_scale: float = 0.03,
    random_seed: Optional[int] = None,
) -> VNCRecurrentNetwork:
    """
    Create a random VNC network with specified parameter statistics.

    Args:
        num_neurons: Number of neurons
        dt: Time step
        weights: Weight matrix (sparse or dense)
        activation_function: Activation function name
        gains_stats: Statistics for gains (mean, std)
        max_firing_rates_stats: Statistics for max firing rates (mean, std)
        thresholds_stats: Statistics for thresholds (mean, std)
        time_constants_stats: Statistics for time constants (mean, std)
        neuron_size_scale: Factors that scale the gains and thresholds of individual neurons based on their size
        random_seed: Random seed for reproducibility

    Returns:
        Configured VNCRecurrentNetwork instance
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Random parameters
    gains = process_parameter_stats(num_neurons, gains_stats)
    max_rates = process_parameter_stats(num_neurons, max_firing_rates_stats)
    thresholds = process_parameter_stats(num_neurons, thresholds_stats)
    time_constants = process_parameter_stats(num_neurons, time_constants_stats)

    # Apply size scaling if provided
    if len(neuron_size_scale) == num_neurons:
        size_scale = np.asarray(neuron_size_scale)
        gains /= size_scale
        thresholds *= size_scale
    else:
        warnings.warn("neuron_size_scale length does not match num_neurons; skipping size scaling.")

    # Create network instance
    return VNCRecurrentNetwork(
        num_neurons=num_neurons,
        dt=dt,
        gains=gains,
        max_firing_rates=max_rates,
        thresholds=thresholds,
        time_constants=time_constants,
        weights=weights,
        relative_inputs_scale=relative_inputs_scale,
        activation_function=activation_function,
    )
