import jax
import jax.numpy as jnp
from jax import Array, random, lax
from functools import partial
from typing import Optional

from .vnc_network import create_network_params, NetworkParams
from src.vnc_sim import NeuronParams, SimParams, reweight_connectivity


@jax.jit
def simulate_network_static_input(
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
        W: Weight matrix, shape (N, N)
        tau: Time constants, shape (N,)
        a: Activation parameters, shape (N,)
        threshold: Firing thresholds, shape (N,)
        fr_cap: Firing rate caps, shape (N,)
        inputs: External inputs, shape (N,)
        noise_stdv: Standard deviation of noise, shape (1,)
        t_axis: Time axis, shape (T,)
        T: Total simulation time in seconds
        dt: Time step in seconds
        pulse_start: Start time (s) of input pulse
        pulse_end: End time (s) of input pulse
        r_tol: Relative tolerance for ODE solver
        a_tol: Absolute tolerance for ODE solver
        key: Random key for JAX

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
    return _simulate_network_static_input(
        params=params,
        num_timesteps=int(T / dt),
        input_data=inputs,
        pulse_start=pulse_start,
        pulse_end=pulse_end,
        input_noise_std=noise_stdv,
        recurrent_noise_std=0,
        input_noise_tau=dt,
        recurrent_noise_tau=dt,
        batch_size=1,
        initial_state=None,
        random_key=key,
    )


@partial(jax.jit, static_argnames=("num_timesteps", "batch_size"))
def _simulate_network_static_input(
    params: NetworkParams,
    num_timesteps: int,
    input_data: Array,
    pulse_start: float,
    pulse_end: float,
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

    # Define the scan function for the simulation step
    def simulation_step(carry, time_step):
        """Single simulation step for JAX scan."""
        # Unpack carry and inputs
        (r_current, key, noise_input_t, noise_recurrent_t) = carry
        pulse_active = (time_step >= pulse_start) & (time_step <= pulse_end)

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
        total_input = (
            input_data * pulse_active
            + params.relative_inputs_scale * synaptic_input
            - params.thresholds
            + noise_input_t
        )

        # Apply activation function
        activated = params.activation_func(total_input, params.gains, params.max_firing_rates)

        # Update firing rates using Euler integration
        r_new = (1.0 - params.alpha) * r_current + params.alpha * activated

        # Update the carry state
        new_carry = (r_new, key, noise_input_t, noise_recurrent_t)
        return new_carry, r_new

    # Prepare scan variables
    noise_input_t = jnp.zeros((batch_size, params.num_neurons))
    noise_recurrent_t = jnp.zeros((batch_size, params.num_neurons))
    scan_inputs = jnp.arange(1, num_timesteps + 1) * params.dt
    scan_carry = (r, random_key, noise_input_t, noise_recurrent_t)

    # Run the simulation using JAX scan
    _, all_states = lax.scan(simulation_step, scan_carry, scan_inputs)

    # Combine initial state with all simulation states
    firing_rates = firing_rates.at[1:].set(all_states)

    return firing_rates


@jax.jit
def process_batch_baseline_discrete_rnn(
    neuron_params: NeuronParams,
    sim_params: SimParams,
    batch_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Process a batch of baseline simulations."""

    def single_sim(idx):
        param_idx = idx % sim_params.n_param_sets
        stim_idx = idx // sim_params.n_param_sets
        return run_baseline_discrete_rnn(
            neuron_params.W,
            neuron_params.tau[param_idx],
            neuron_params.a[param_idx],
            neuron_params.threshold[param_idx],
            neuron_params.fr_cap[param_idx],
            neuron_params.input_currents[stim_idx],
            sim_params,
            neuron_params.seeds[param_idx],
        )

    return jax.vmap(single_sim)(batch_indices)


@jax.jit
def run_baseline_discrete_rnn(
    W: jnp.ndarray,
    tau: jnp.ndarray,
    a: jnp.ndarray,
    threshold: jnp.ndarray,
    fr_cap: jnp.ndarray,
    inputs: jnp.ndarray,
    sim_params: SimParams,
    seed: jnp.ndarray,
) -> jnp.ndarray:
    """Run baseline simulation without shuffle or noise."""
    W_reweighted = reweight_connectivity(W, sim_params.exc_multiplier, sim_params.inh_multiplier)

    return simulate_network_static_input(
        W_reweighted,
        tau,
        a,
        threshold,
        fr_cap,
        inputs,
        noise_stdv=sim_params.noise_stdv,
        t_axis=sim_params.t_axis,
        T=sim_params.T,
        dt=sim_params.dt,
        pulse_start=sim_params.pulse_start,
        pulse_end=sim_params.pulse_end,
        r_tol=sim_params.r_tol,
        a_tol=sim_params.a_tol,
        key=seed,
    )
