"""
High-performance Ventral Nerve Cord (VNC) Recurrent Neural Network.

This module implements a biologically-inspired RNN model for simulating neural dynamics
in the ventral nerve cord of insects, using efficient JAX operations for maximum performance
with a functional programming approach.

Mathematical Model:
    Continuous dynamics: τ * dr/dt = -r(t) + f(I_ext(t) + b * W @ (r(t) + η_rec(t)) - θ + η(t))
    Discrete update: r(t+dt) = (1-α) * r(t) + α * f(I_ext(t) + b * W @ (r(t) + η_rec(t)) - θ + η(t))
    where α = dt/τ
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array, random, lax
from functools import partial
from typing import Union, Optional, NamedTuple
import warnings
from .activation_functions import get_activation_function
from src.sim_utils import set_sizes


class NetworkParams(NamedTuple):
    """
    Parameters for the VNC Recurrent Network.

    Attributes:
        num_neurons: Number of neurons in the network
        dt: Simulation time step
        gains: Individual neuron gains, shape (N,)
        max_firing_rates: Maximum firing rates, shape (N,)
        thresholds: Firing thresholds, shape (N,)
        time_constants: Time constants, shape (N,)
        weights: Synaptic weights, shape (N, N)
        relative_inputs_scale: Scaling factor for relative inputs
        activation_func: Activation function
        alpha: Precomputed dt/tau values, shape (N,)
    """

    num_neurons: int
    dt: float
    gains: Array
    max_firing_rates: Array
    thresholds: Array
    time_constants: Array
    weights: Array
    relative_inputs_scale: float
    activation_function: str
    alpha: Array


def ensure_array_shape(param: Union[float, Array], num_neurons: int, name: str) -> Array:
    """
    Ensure parameter has correct shape (N,).

    Args:
        param: Parameter value (scalar or array)
        num_neurons: Number of neurons
        name: Parameter name for error messages

    Returns:
        Array with shape (N,) or scalar

    Raises:
        ValueError: If parameter dimensions are incompatible
    """
    param = jnp.asarray(param)
    if param.ndim == 0 or param.shape == (num_neurons,):
        return param
    else:
        raise ValueError(f"{name} shape {param.shape} != (1,) or ({num_neurons},)")


def create_network_params(
    num_neurons: int,
    dt: float,
    gains: Union[float, Array],
    max_firing_rates: Union[float, Array],
    thresholds: Union[float, Array],
    time_constants: Union[float, Array],
    weights: Union[np.ndarray, Array],
    relative_inputs_scale: float = 0.03,
    activation_function: str = "scaled_tanh_relu",
) -> NetworkParams:
    """
    Create network parameters for the VNC Recurrent Network.

    Args:
        num_neurons: Number of neurons in the network
        dt: Simulation time step
        gains: Individual neuron gains, scalar or shape (N,)
        max_firing_rates: Maximum firing rates, scalar or shape (N,)
        thresholds: Firing thresholds, scalar or shape (N,)
        time_constants: Time constants, scalar or shape (N,)
        weights: Synaptic weight matrix, shape (N, N)
        relative_inputs_scale: Scaling factor for relative inputs
        activation_function: Type of activation function
                            ('scaled_tanh_relu', 'tanh_scaled', 'sigmoid_scaled', 'relu_scaled', 'linear_clipped')

    Returns:
        NetworkParams object containing all network parameters

    Raises:
        ValueError: If parameter dimensions are incompatible
    """
    # Convert scalar parameters to arrays
    gains_arr = ensure_array_shape(gains, num_neurons, "gains")
    max_firing_rates_arr = ensure_array_shape(max_firing_rates, num_neurons, "max_firing_rates")
    thresholds_arr = ensure_array_shape(thresholds, num_neurons, "thresholds")
    time_constants_arr = ensure_array_shape(time_constants, num_neurons, "time_constants")

    # Validate weight matrix
    if hasattr(weights, "shape"):
        if weights.shape != (num_neurons, num_neurons):
            raise ValueError(f"Weight matrix shape {weights.shape} != ({num_neurons}, {num_neurons})")
    weights_arr = jnp.asarray(weights)

    # Precompute frequently used values for efficiency
    alpha = dt / time_constants_arr

    # # Validate alpha values
    # if jnp.any(alpha > 1.0):
    #     warnings.warn("Some alpha values > 1.0, simulation may be unstable. Consider smaller dt.")
    # if jnp.any(alpha <= 0.0):
    #     raise ValueError("Alpha values must be positive (dt and time_constants must be positive)")

    return NetworkParams(
        num_neurons=num_neurons,
        dt=dt,
        gains=gains_arr,
        max_firing_rates=max_firing_rates_arr,
        thresholds=thresholds_arr,
        time_constants=time_constants_arr,
        weights=weights_arr,
        relative_inputs_scale=relative_inputs_scale,
        activation_function=activation_function,
        alpha=alpha,
    )


def update_network_params(
    params: NetworkParams,
    gains: Optional[Union[float, Array]] = None,
    max_firing_rates: Optional[Union[float, Array]] = None,
    thresholds: Optional[Union[float, Array]] = None,
    time_constants: Optional[Union[float, Array]] = None,
    weights: Optional[Union[np.ndarray, Array]] = None,
    relative_inputs_scale: Optional[float] = None,
    activation_function: Optional[str] = None,
) -> NetworkParams:
    """
    Update network parameters.

    Args:
        params: Current network parameters
        gains: New gains (optional)
        max_firing_rates: New max firing rates (optional)
        thresholds: New thresholds (optional)
        time_constants: New time constants (optional)
        weights: New weight matrix (optional)
        relative_inputs_scale: New relative inputs scale (optional)
        activation_function: New activation function name (optional)

    Returns:
        Updated NetworkParams object
    """
    # Start with current parameters
    updated_params = {
        "num_neurons": params.num_neurons,
        "dt": params.dt,
        "gains": params.gains,
        "max_firing_rates": params.max_firing_rates,
        "thresholds": params.thresholds,
        "time_constants": params.time_constants,
        "weights": params.weights,
        "relative_inputs_scale": params.relative_inputs_scale,
        "activation_function": params.activation_function,
        "alpha": params.alpha,
    }

    # Update provided parameters
    if gains is not None:
        updated_params["gains"] = ensure_array_shape(gains, params.num_neurons, "gains")
    if max_firing_rates is not None:
        updated_params["max_firing_rates"] = ensure_array_shape(
            max_firing_rates, params.num_neurons, "max_firing_rates"
        )
    if thresholds is not None:
        updated_params["thresholds"] = ensure_array_shape(thresholds, params.num_neurons, "thresholds")
    if time_constants is not None:
        updated_params["time_constants"] = ensure_array_shape(time_constants, params.num_neurons, "time_constants")
        updated_params["alpha"] = params.dt / updated_params["time_constants"]
    if weights is not None:
        if hasattr(weights, "shape") and weights.shape != (params.num_neurons, params.num_neurons):
            raise ValueError(f"Weight matrix shape {weights.shape} != ({params.num_neurons}, {params.num_neurons})")
        updated_params["weights"] = jnp.asarray(weights)
    if relative_inputs_scale is not None:
        updated_params["relative_inputs_scale"] = relative_inputs_scale
    if activation_function is not None:
        updated_params["activation_function"] = get_activation_function(activation_function)

    return NetworkParams(**updated_params)


@jax.jit
def simulate_network_SRC_VNC_SIM_COMPATIBILITY(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    noise_stdv: jnp.ndarray,
    t_axis: jnp.ndarray,
    T: float,
    dt: float,
    pulse_start: float,
    pulse_end: float,
    r_tol: float,
    a_tol: float,
    key: jnp.ndarray,
) -> Array:
    """
    Run simulation(s) of the network using JAX scan optimization.

    This function pre-computes all input and noise data to avoid tracing issues
    with JAX scan, then uses the optimized simulate_network_jit function.

    Args:
        params: Network parameters
        num_timesteps: Number of time steps to simulate
        inputs: External inputs, shape (T, batch_size, N) or (batch_size, N) or (N,)
        initial_state: Initial firing rates. For single sim: shape (N,), for batch: shape (batch_size, N)
        batch_size: Number of parallel simulations (default 1 for single simulation)
        input_noise_std: Standard deviation of input noise (default 0.0)
        recurrent_noise_std: Standard deviation of recurrent noise (default 0.0)
        input_noise_tau: Time constant for input noise (default 1.0)
        recurrent_noise_tau: Time constant for recurrent noise (default 1.0)
        random_seed: Random seed for reproducibility (default 11)

    Returns:
        Firing rates array, shape (T+1, N) for single sim or (T+1, batch_size, N) for batch
    """
    params = create_network_params(
        num_neurons=len(W),
        dt=dt,
        gains=a,
        max_firing_rates=fr_cap,
        thresholds=threshold,
        time_constants=tau,
        weights=W,
        relative_inputs_scale=0.03,
        activation_function="scaled_tanh_relu",
    )

    # Use the JIT-compiled method with pre-computed data
    return simulate_network(
        params=params,
        num_timesteps=int(T / dt),
        input_data=inputs,
        input_noise_std=noise_stdv,
        recurrent_noise_std=noise_stdv,
        input_noise_tau=1.0,
        recurrent_noise_tau=1.0,
        batch_size=1,
        initial_state=None,
        random_key=key,
    )


@partial(jax.jit, static_argnames=("num_timesteps", "batch_size"))
def simulate_network(
    params: NetworkParams,
    num_timesteps: int,
    input_data: Array,
    input_noise_std: float = 0.0,
    recurrent_noise_std: float = 0.0,
    input_noise_tau: float = 1.0,
    recurrent_noise_tau: float = 1.0,
    batch_size: int = 1,
    initial_state: Optional[Array] = None,
    random_key: Array = random.key(11),
) -> Array:
    """
    JIT-compiled simulation with pre-computed input and noise data using JAX scan.

    This version uses pre-computed arrays instead of function calls,
    making it compatible with JAX transformations.

    Args:
        params: Network parameters
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
        r = jnp.zeros((batch_size, params.num_neurons))
    else:
        r = jnp.asarray(initial_state)
        if r.ndim == 1:
            r = r[jnp.newaxis, :]
        if r.shape != (batch_size, params.num_neurons):
            raise ValueError(f"Initial states shape {r.shape} != ({batch_size}, {params.num_neurons})")

    # Prepare storage
    firing_rates = jnp.zeros((num_timesteps + 1, batch_size, params.num_neurons))
    firing_rates = firing_rates.at[0].set(r)

    # prepare noise parameters
    alpha_input = params.dt / input_noise_tau
    alpha_recurrent = params.dt / recurrent_noise_tau
    scale_input = jnp.sqrt((2 - alpha_input) / alpha_input)
    scale_recurrent = jnp.sqrt((2 - alpha_recurrent) / alpha_recurrent)

    # parse activation function
    activation_function = get_activation_function(params.activation_function)

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
            key_input, shape=r_current.shape
        ) * input_noise_std * scale_input
        noise_recurrent_t = (1 - alpha_recurrent) * noise_recurrent_t + alpha_recurrent * random.normal(
            key_recurrent, shape=r_current.shape
        ) * recurrent_noise_std * scale_recurrent

        # Synaptic input
        synaptic_input = (r_current + noise_recurrent_t) @ params.weights.T

        # Total input
        total_input = input_t + params.relative_inputs_scale * synaptic_input - params.thresholds + noise_input_t

        # Apply activation function
        activated = activation_function(total_input, params.gains, params.max_firing_rates)

        # Update firing rates using Euler integration
        r_new = (1.0 - params.alpha) * r_current + params.alpha * activated

        # Update the carry state
        new_carry = (r_new, key, noise_input_t, noise_recurrent_t)
        return new_carry, r_new

    # Prepare scan variables
    noise_input_t = jnp.zeros((batch_size, params.num_neurons))
    noise_recurrent_t = jnp.zeros((batch_size, params.num_neurons))
    scan_inputs = input_data
    scan_carry = (r, random_key, noise_input_t, noise_recurrent_t)

    # Run the simulation using JAX scan
    _, all_states = lax.scan(simulation_step, scan_carry, scan_inputs)

    # Combine initial state with all simulation states
    firing_rates = firing_rates.at[1:].set(all_states)

    return firing_rates


def process_parameter_stats(n_samples: int, stats: dict, key: Array) -> Array:
    """
    Process parameter statistics by extracting means and standard deviations.

    Args:
        n_samples: Number of samples to generate
        stats: Dictionary containing parameter statistics (mean, std)
        key: JAX random key for reproducibility

    Returns:
        Generated parameter array

    Raises:
        ValueError: If parameter statistics are invalid
    """
    if not stats:
        raise ValueError("Parameter statistics dictionary is empty")
    elif stats["distribution"] == "normal":
        mean = stats["mean"]
        std = stats["std"]
        return random.normal(key, (n_samples,)) * std + mean
    else:
        raise ValueError(f"Unsupported distribution: {stats['distribution']}")


def create_random_network_params(
    num_neurons: int,
    dt: float,
    weights: Union[np.ndarray, Array],
    activation_function: str = "scaled_tanh_relu",
    gains_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    max_firing_rates_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    thresholds_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    time_constants_stats: dict = {"distribution": "normal", "mean": 1.0, "std": 0.3},
    neuron_sizes: Union[list, np.ndarray, Array] = [],
    relative_inputs_scale: float = 0.03,
    random_seed: Optional[int] = None,
) -> NetworkParams:
    """
    Create random VNC network parameters with specified parameter statistics.

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

    Returns:
        Configured NetworkParams object
    """
    # Create base key and split for different parameter types
    base_key = random.key(random_seed if random_seed is not None else 0)
    keys = random.split(base_key, 4)

    # Random parameters
    gains = process_parameter_stats(num_neurons, gains_stats, keys[0])
    max_rates = process_parameter_stats(num_neurons, max_firing_rates_stats, keys[1])
    thresholds = process_parameter_stats(num_neurons, thresholds_stats, keys[2])
    time_constants = process_parameter_stats(num_neurons, time_constants_stats, keys[3])

    # Apply size scaling if provided
    if len(neuron_sizes) == num_neurons:
        gains, thresholds = set_sizes(neuron_sizes, gains, thresholds)
        gains = gains.squeeze()
        thresholds = thresholds.squeeze()
    else:
        warnings.warn("neuron_sizes length does not match num_neurons; skipping size scaling.")

    # Create network parameters
    return create_network_params(
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


def get_network_info(params: NetworkParams) -> str:
    """
    Get string representation of the network parameters.

    Args:
        params: Network parameters

    Returns:
        String representation of the network
    """
    return (
        f"VNCRecurrentNetwork(num_neurons={params.num_neurons}, dt={params.dt}, "
        f"activation={params.activation_function})"
    )
