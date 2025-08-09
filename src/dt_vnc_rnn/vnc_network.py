"""
High-performance Ventral Nerve Cord (VNC) Recurrent Neural Network implementation.

This module implements a biologically-inspired RNN model for simulating neural dynamics
in the ventral nerve cord of insects, using efficient JAX operations for maximum performance.

Mathematical Model:
    Continuous dynamics: τ * dr/dt = -r(t) + f(I_ext(t) + b * W @ (r(t) + η_rec(t)) - θ + η(t))
    Discrete update: r(t+dt) = (1-α) * r(t) + α * f(I_ext(t) + b * W @ (r(t) + η_rec(t)) - θ + η(t))
    where α = dt/τ
"""

import jax
import jax.numpy as jnp
from jax import Array, random, lax
from functools import partial
from typing import Union, Callable, Optional
import warnings
from .activation_functions import ActivationFunction
from .input_functions import InputFunction
from src.sim_utils import set_sizes


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
        gains (Array): Individual neuron gains, shape (N,)
        max_firing_rates (Array): Maximum firing rates, shape (N,)
        thresholds (Array): Firing thresholds, shape (N,)
        time_constants (Array): Time constants, shape (N,)
        weights (Array): Synaptic weights, shape (N, N)
        activation_func (Callable): Activation function
        alpha (Array): Precomputed dt/tau values, shape (N,)
        one_minus_alpha (Array): Precomputed (1-dt/tau) values, shape (N,)
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float,
        gains: Union[float, Array],
        max_firing_rates: Union[float, Array],
        thresholds: Union[float, Array],
        time_constants: Union[float, Array],
        weights: Array,
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
        self.weights = jnp.asarray(weights)

        # Set activation function
        self.activation_func = self._get_activation_function(activation_function)

        # Precompute frequently used values for efficiency
        self.alpha = self.dt / self.time_constants

        # Validate alpha values
        if jnp.any(self.alpha > 1.0):
            warnings.warn("Some alpha values > 1.0, simulation may be unstable. Consider smaller dt.")
        if jnp.any(self.alpha <= 0.0):
            raise ValueError("Alpha values must be positive (dt and time_constants must be positive)")

    def _ensure_array_shape(self, param: Union[float, Array], name: str) -> Array:
        """Ensure parameter has correct shape (N,)."""
        param = jnp.asarray(param)
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
        input_func: InputFunction,
        initial_state: Optional[Array] = None,
        batch_size: int = 1,
        input_noise_std: float = 0.0,
        recurrent_noise_std: float = 0.0,
        input_noise_tau: float = 1.0,
        recurrent_noise_tau: float = 1.0,
    ) -> Array:
        """
        Run simulation(s) of the network using JAX scan optimization.

        This method pre-computes all input and noise data to avoid tracing issues
        with JAX scan, then uses the optimized simulate_jit method.

        Args:
            num_timesteps: Number of time steps to simulate
            input_func: Function returning external inputs
            initial_state: Initial firing rates. For single sim: shape (N,), for batch: shape (batch_size, N)
            batch_size: Number of parallel simulations (default 1 for single simulation)
            input_noise_std: Standard deviation of input noise (default 0.0)
            recurrent_noise_std: Standard deviation of recurrent noise (default 0.0)
            input_noise_tau: Time constant for input noise (default 1.0)
            recurrent_noise_tau: Time constant for recurrent noise (default 1.0)

        Returns:
            Firing rates array, shape (T+1, N) for single sim or (T+1, batch_size, N) for batch
        """
        # Pre-compute all input data to avoid tracing issues in scan
        input_data = input_func.get_full_input_data()

        # Adjust noise time constants to ensure noise correlation time is at least dt
        if input_noise_tau < self.dt:
            warnings.warn(f"Input noise time constant too small, adjusted from {input_noise_tau} to {self.dt}")
            input_noise_tau = self.dt
        if recurrent_noise_tau < self.dt:
            warnings.warn(f"Recurrent noise time constant too small, adjusted from {recurrent_noise_tau} to {self.dt}")
            recurrent_noise_tau = self.dt

        # Use the JIT-compiled method with pre-computed data
        return self.simulate_jit(
            num_timesteps=num_timesteps,
            input_data=input_data,
            input_noise_std=input_noise_std,
            recurrent_noise_std=recurrent_noise_std,
            input_noise_tau=input_noise_tau,
            recurrent_noise_tau=recurrent_noise_tau,
            batch_size=batch_size,
            initial_state=initial_state,
        )

    @partial(jax.jit, static_argnums=(0,), static_argnames=("num_timesteps", "batch_size", "initial_state"))
    def simulate_jit(
        self,
        num_timesteps: int,
        input_data: Array,
        input_noise_std: float = 0.0,
        recurrent_noise_std: float = 0.0,
        input_noise_tau: float = 1.0,
        recurrent_noise_tau: float = 1.0,
        batch_size: int = 1,
        initial_state: Optional[Array] = None,
        random_seed: int = 11,
    ) -> Array:
        """
        Simulation with pre-computed input and noise data using JAX scan.

        This version uses pre-computed arrays instead of function calls,
        making it compatible with JAX transformations.

        Args:
            num_timesteps: Number of time steps to simulate
            input_data: Pre-computed input data, shape (num_timesteps, batch_size, num_neurons) or broadcastable
            input_noise_std: Standard deviation of input noise (default 0.0)
            recurrent_noise_std: Standard deviation of recurrent noise (default 0.0)
            input_noise_tau: Time constant for input noise (default 1.0)
            recurrent_noise_tau: Time constant for recurrent noise (default 1.0)
            batch_size: Number of parallel simulations (default 1 for single simulation)
            initial_state: Initial firing rates. For single sim: shape (N,), for batch: shape (batch_size, N)
            random_seed: Random seed for reproducibility (default 11)
        Returns:
            Firing rates array, shape (T+1, batch_size, N)
        """
        # Handle initial state
        if initial_state is None:
            r = jnp.zeros((batch_size, self.num_neurons))
        else:
            r = jnp.asarray(initial_state)
            if r.ndim == 1:
                r = r[jnp.newaxis, :]
            if r.shape != (batch_size, self.num_neurons):
                raise ValueError(f"Initial states shape {r.shape} != ({batch_size}, {self.num_neurons})")

        # Prepare storage
        firing_rates = jnp.zeros((num_timesteps + 1, batch_size, self.num_neurons))
        firing_rates = firing_rates.at[0].set(r)

        # prepare noise parameters
        alpha_input = self.dt / input_noise_tau
        alpha_recurrent = self.dt / recurrent_noise_tau
        scale_input = jnp.sqrt((2 - alpha_input) / alpha_input)
        scale_recurrent = jnp.sqrt((2 - alpha_recurrent) / alpha_recurrent)

        # Define the scan function for the simulation step
        def simulation_step(carry, inputs):
            """Single simulation step for JAX scan."""
            # Unpack carry and inputs
            (r_current, key, noise_input_t, noise_recurrent_t) = carry
            input_t = inputs

            # Split the random key for noise generation
            key, key_input, key_recurrent = random.split(key, num=3)

            # Generate noise
            noise_input_t = (1 - alpha_input) * noise_input_t + alpha_input * random.normal(
                key_input, shape=input_t.shape
            ) * input_noise_std * scale_input
            noise_recurrent_t = (1 - alpha_recurrent) * noise_recurrent_t + alpha_recurrent * random.normal(
                key_recurrent, shape=r_current.shape
            ) * recurrent_noise_std * scale_recurrent

            # Synaptic input
            synaptic_input = (r_current + noise_recurrent_t) @ self.weights.T

            # Total input
            total_input = input_t + self.relative_inputs_scale * synaptic_input - self.thresholds + noise_input_t

            # Apply activation function
            activated = self.activation_func(total_input, self.gains, self.max_firing_rates)

            # Update firing rates using Euler integration
            r_new = (1.0 - self.alpha) * r_current + self.alpha * activated

            # Update the carry state
            new_carry = (r_new, key, noise_input_t, noise_recurrent_t)
            return new_carry, r_new

        # Prepare scan variables
        key = random.key(random_seed)
        noise_input_t = jnp.zeros((batch_size, self.num_neurons))
        noise_recurrent_t = jnp.zeros((batch_size, self.num_neurons))
        scan_inputs = input_data
        scan_carry = (r, key, noise_input_t, noise_recurrent_t)

        # Run the simulation using JAX scan
        final_state, all_states = lax.scan(simulation_step, scan_carry, scan_inputs)

        # Combine initial state with all simulation states
        firing_rates = firing_rates.at[1:].set(all_states)

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
                self.weights = jnp.asarray(value)
            elif param == "activation_function":
                self.activation_func = self._get_activation_function(value)
            else:
                raise ValueError(f"Unknown parameter: {param}")

        # Recompute alpha values if time constants changed
        if "time_constants" in kwargs:
            self.alpha = self.dt / self.time_constants

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"VNCRecurrentNetwork(num_neurons={self.num_neurons}, dt={self.dt}, "
            f"activation={self.activation_func.__name__})"
        )


# Convenience functions for creating common network configurations
def process_parameter_stats(n_samples: int, stats: dict, key: Array) -> Array:
    """
    Process parameter statistics by extracting means and standard deviations.

    Args:
        n_samples: Number of samples to generate
        stats: Dictionary containing parameter statistics (mean, std)
        key: JAX random key for reproducibility
    """
    if not stats:
        raise ValueError("Parameter statistics dictionary is empty")
    elif stats["distribution"] == "normal":
        mean = stats["mean"]
        std = stats["std"]
        return random.normal(key, (n_samples,)) * std + mean
    else:
        raise ValueError(f"Unsupported distribution: {stats['distribution']}")


def create_random_network(
    num_neurons: int,
    dt: float,
    weights: Array,
    activation_function: str = "scaled_tanh_relu",
    gains_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    max_firing_rates_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    thresholds_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    time_constants_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    neuron_sizes: Union[list, Array] = [],
    relative_inputs_scale: float = 0.03,
    random_seed: Optional[int] = None,
) -> VNCRecurrentNetwork:
    """
    Create a random VNC network with specified parameter statistics.

    Args:
        num_neurons: Number of neurons
        dt: Time step
        weights: Weight matrix (dense array)
        activation_function: Activation function name
        gains_stats: Statistics for gains (mean, std)
        max_firing_rates_stats: Statistics for max firing rates (mean, std)
        thresholds_stats: Statistics for thresholds (mean, std)
        time_constants_stats: Statistics for time constants (mean, std)
        neuron_sizes: Individual neuron size that dictates the scale factor for gains and thresholds
        relative_inputs_scale: Scaling factor for relative inputs
        random_seed: Random seed for reproducibility
        numpy_random: If True, use NumPy for random number generation; if False, use JAX

    Returns:
        Configured VNCRecurrentNetwork instance
    """
    # Create base key and split for different parameter types
    base_key = random.key(random_seed if random_seed is not None else 0)
    keys = random.split(base_key, 4)

    # Random parameters
    gains = process_parameter_stats(num_neurons, gains_stats, keys[0])
    max_rates = process_parameter_stats(num_neurons, max_firing_rates_stats, keys[1])
    thresholds = process_parameter_stats(num_neurons, thresholds_stats, keys[2])
    time_constants = process_parameter_stats(num_neurons, time_constants_stats, keys[3])
    print(gains.shape)

    # Apply size scaling if provided
    if len(neuron_sizes) == num_neurons:
        gains, thresholds = set_sizes(neuron_sizes, gains, thresholds)
        gains = gains.squeeze()
        thresholds = thresholds.squeeze()
    else:
        warnings.warn("neuron_sizes length does not match num_neurons; skipping size scaling.")

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
