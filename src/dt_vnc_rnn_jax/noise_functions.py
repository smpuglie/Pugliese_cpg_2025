"""
Noise functions for the VNC Recurrent Neural Network.

This module provides a collection of noise functions for adding stochasticity to neural
dynamics. Supports both single simulations and batched simulations for parallel processing.
"""

import numpy as np
import jax.numpy as jnp
from jax import Array, random
from typing import Union, Optional, Any


class NoiseFunction:
    """
    Object-oriented noise function for adding stochasticity to neural dynamics.

    This class provides two modes of operation:
    1. Pre-computed mode: Generate and store all noise values (memory-intensive)
    2. On-the-fly mode: Generate noise values as needed using get(t) (memory-efficient)

    The class can be used with either method call syntax or callable syntax for convenience.

    Args:
        noise_type: Type of noise function ('gaussian', 'colored_noise')
        num_neurons: Number of neurons
        num_timesteps: Number of time steps
        batch_size: Number of parallel simulations (default: 1)
        precompute: If True, pre-compute all noise; if False, compute on-the-fly
        random_seed: Random seed for reproducibility
        dt: Time step size (required for colored noise)
        numpy_random: If True, use NumPy for random number generation; if False, use JAX
        **kwargs: Parameters specific to the noise function type

    Example:
        >>> # Pre-computed Gaussian noise
        >>> noise_func = NoiseFunction('gaussian', num_neurons=100, num_timesteps=1000,
        ...                           precompute=True, std=0.1, random_seed=42)
        >>>
        >>> # Both syntaxes are equivalent:
        >>> current_noise = noise_func.get(50)  # Method call
        >>> current_noise = noise_func(50)      # Callable interface
        >>>
        >>> # On-the-fly colored noise (memory efficient)
        >>> noise_func = NoiseFunction('colored_noise', num_neurons=100, num_timesteps=1000,
        ...                           precompute=False, dt=0.001, std=0.1, correlation_time=5.0,
        ...                           random_seed=42)
        >>> current_noise = noise_func(50)  # Generate noise at timestep 50
    """

    def __init__(
        self,
        noise_type: str,
        num_neurons: int,
        num_timesteps: int,
        batch_size: int = 1,
        precompute: bool = True,
        random_seed: Optional[int] = None,
        dt: Optional[float] = None,
        numpy_random: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        self.noise_type = noise_type
        self.num_neurons = num_neurons
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.precompute = precompute
        self.dt = dt
        self.numpy_random = numpy_random
        self.params = kwargs

        if numpy_random:
            # Set random seed for reproducibility
            self.random_seed = random_seed
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
                # Store the initial random state for on-the-fly generation
                self._initial_rng_state = np.random.get_state()
        else:
            # Set up JAX random key for reproducibility
            self.random_seed = random_seed if random_seed is not None else 0
            self._base_key = random.PRNGKey(self.random_seed)

        if self.precompute:
            # Generate and store all noise values
            self._noise_data = self._generate_all_noise()
        else:
            # Initialize state for on-the-fly generation
            self._noise_data = None
            self._initialize_onthefly_state()

    def _generate_all_noise(self) -> Array:
        """Generate all noise values at once (pre-computed mode)."""
        if self.noise_type == "gaussian":
            return self._gaussian_precomputed(**self.params)
        elif self.noise_type == "colored_noise":
            return self._colored_noise_precomputed(**self.params)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def _initialize_onthefly_state(self) -> None:
        """Initialize state variables for on-the-fly noise generation."""
        if self.noise_type == "colored_noise":
            # Initialize state for colored noise
            self._colored_noise_state = jnp.zeros((self.batch_size, self.num_neurons))
            self._last_timestep = -1
        elif self.noise_type == "gaussian":
            # Gaussian noise is stateless, no initialization needed
            pass
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def get(self, t: int) -> Array:
        """
        Get noise at timestep t.

        Args:
            t: Timestep index (0-based)

        Returns:
            Noise array with shape:
            - (batch_size, num_neurons) for batch simulation
            - (num_neurons,) for single simulation
        """
        if t < 0 or t >= self.num_timesteps:
            raise IndexError(f"Timestep {t} out of range [0, {self.num_timesteps})")

        if self.precompute:
            # Return pre-computed noise
            if self._noise_data is not None:
                result = self._noise_data[t]
            else:
                raise RuntimeError("Noise data not initialized")
        else:
            # Generate noise on-the-fly
            result = self._generate_noise_at_timestep(t)

        return result

    def __call__(self, t: int) -> Array:
        """
        Make the NoiseFunction object callable like a function.

        This allows using the object directly as a function: noise_func(t) instead of noise_func.get(t).
        Both syntaxes are equivalent and provide the same functionality.

        Args:
            t: Timestep index (0-based). Must be in range [0, num_timesteps).

        Returns:
            Noise array with shape:
            - (batch_size, num_neurons) for batch simulation
            - (num_neurons,) for single simulation

        Example:
            >>> noise_func = NoiseFunction('gaussian', num_neurons=100, num_timesteps=1000,
            ...                          batch_size=5, std=0.1, random_seed=42)
            >>> current_noise = noise_func(50)  # Equivalent to noise_func.get(50)
        """
        return self.get(t)

    def _generate_noise_at_timestep(self, t: int) -> Array:
        """Generate noise at specific timestep (on-the-fly mode)."""
        if self.noise_type == "gaussian":
            return self._gaussian_onthefly(t, **self.params)
        elif self.noise_type == "colored_noise":
            return self._colored_noise_onthefly(t, **self.params)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def _gaussian_precomputed(self, std: Union[float, Array]) -> Array:
        """Pre-computed Gaussian white noise.

        Args:
            std: Standard deviation of the Gaussian noise (scalar or array of shape (N,) or (B,) or (B,N))

        Returns:
            Noise array of shape (T, B, N)
        """
        if self.numpy_random:
            noise = jnp.asarray(np.random.normal(0, 1, (self.num_timesteps, self.batch_size, self.num_neurons)))
        else:
            key = self._base_key
            noise = random.normal(key, (self.num_timesteps, self.batch_size, self.num_neurons))
        std_array = jnp.asarray(std)
        if std_array.ndim == 1 and std_array.shape[0] == self.batch_size:
            std_array = std_array.reshape(-1, 1)

        return noise * std_array

    def _gaussian_onthefly(self, t: int, std: Union[float, Array]) -> Array:
        """On-the-fly Gaussian white noise.

        Args:
            std: Standard deviation of the Gaussian noise (scalar or array of shape (N,) or (B,) or (B,N))

        Returns:
            Noise array of shape (B, N)
        """
        if self.numpy_random:
            # Set deterministic seed based on original seed and timestep
            if self.random_seed is not None:
                np.random.seed(self.random_seed + t)

            noise = jnp.asarray(np.random.normal(0, 1, (self.batch_size, self.num_neurons)))
        else:
            # Create deterministic key based on original seed and timestep
            key = random.fold_in(self._base_key, t)
            noise = random.normal(key, (self.batch_size, self.num_neurons))

        std_array = jnp.asarray(std)

        if std_array.ndim == 1 and std_array.shape[0] == self.batch_size:
            std_array = std_array.reshape(-1, 1)

        return noise * std_array

    def _colored_noise_precomputed(self, std: Union[float, Array], correlation_time: float) -> Array:
        """Pre-computed exponentially correlated (colored) noise.

        Args:
            std: Standard deviation of the Gaussian noise (scalar or array of shape (N,) or (B,) or (B,N))
            correlation_time: Correlation time constant (tau) in seconds

        Returns:
            Noise array of shape (T, B, N)
        """
        if self.dt is None:
            raise ValueError("dt must be specified for colored noise")

        alpha = self.dt / correlation_time
        std_array = jnp.asarray(std)
        if std_array.ndim == 1 and std_array.shape[0] == self.batch_size:
            std_array = std_array.reshape(-1, 1)
        noise_scale = std_array * jnp.sqrt(2 * alpha)

        if self.numpy_random:
            white_noise = jnp.asarray(np.random.normal(0, 1, (self.num_timesteps, self.batch_size, self.num_neurons)))
        else:
            key = self._base_key
            white_noise = random.normal(key, (self.num_timesteps, self.batch_size, self.num_neurons))
        white_noise = white_noise * noise_scale

        colored_noise = jnp.zeros_like(white_noise)
        colored_noise = colored_noise.at[0].set(white_noise[0])

        for t in range(1, self.num_timesteps):
            colored_noise = colored_noise.at[t].set((1 - alpha) * colored_noise[t - 1] + white_noise[t])

        return colored_noise

    def _colored_noise_onthefly(self, t: int, std: Union[float, Array], correlation_time: float) -> Array:
        """On-the-fly exponentially correlated (colored) noise."""
        if self.dt is None:
            raise ValueError("dt must be specified for colored noise")

        alpha = self.dt / correlation_time
        std_array = jnp.asarray(std)
        noise_scale = std_array * jnp.sqrt(2 * alpha)

        # Handle non-sequential access (reset state if needed)
        if t != self._last_timestep + 1:
            if t == 0:
                # Initialize with white noise
                if self.numpy_random:
                    if self.random_seed is not None:
                        np.random.seed(self.random_seed)
                    white_noise = jnp.asarray(np.random.normal(0, 1, (self.batch_size, self.num_neurons)))
                else:
                    key = random.fold_in(self._base_key, 0)
                    white_noise = random.normal(key, (self.batch_size, self.num_neurons))
                if not jnp.isscalar(noise_scale):
                    noise_scale = noise_scale.reshape(1, -1)
                self._colored_noise_state = white_noise * jnp.asarray(noise_scale)
            else:
                # For non-sequential access, we need to compute from the beginning
                # This is less efficient but maintains correctness
                return self._compute_colored_noise_up_to_t(t, std, correlation_time)

        else:
            # Sequential access: update state
            if self.numpy_random:
                if self.random_seed is not None:
                    np.random.seed(self.random_seed + t)
                white_noise = jnp.asarray(np.random.normal(0, 1, (self.batch_size, self.num_neurons)))
            else:
                key = random.fold_in(self._base_key, t)
                white_noise = random.normal(key, (self.batch_size, self.num_neurons))
            if not jnp.isscalar(noise_scale):
                noise_scale = noise_scale.reshape(1, -1)
            white_noise = white_noise * jnp.asarray(noise_scale)

            self._colored_noise_state = (1 - alpha) * jnp.asarray(self._colored_noise_state) + white_noise

        self._last_timestep = t
        return self._colored_noise_state

    def _compute_colored_noise_up_to_t(self, t: int, std: Union[float, Array], correlation_time: float) -> Array:
        """Compute colored noise up to timestep t (for non-sequential access)."""
        if self.dt is None:
            raise ValueError("dt must be specified for colored noise")

        alpha = self.dt / correlation_time
        std_array = jnp.asarray(std)
        noise_scale = std_array * jnp.sqrt(2 * alpha)

        # Generate up to timestep t
        colored_noise = jnp.zeros((self.batch_size, self.num_neurons))

        # Initial condition
        if self.numpy_random:
            # Reset random state for reproducibility
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            white_noise = jnp.asarray(np.random.normal(0, 1, (self.batch_size, self.num_neurons)))
        else:
            key = random.fold_in(self._base_key, 0)
            white_noise = random.normal(key, (self.batch_size, self.num_neurons))
        if not jnp.isscalar(noise_scale):
            noise_scale = noise_scale.reshape(1, -1)
        colored_noise = white_noise * jnp.asarray(noise_scale)

        # Iterate to timestep t
        for i in range(1, t + 1):
            if self.numpy_random:
                if self.random_seed is not None:
                    np.random.seed(self.random_seed + i)
                white_noise = jnp.asarray(np.random.normal(0, 1, (self.batch_size, self.num_neurons)))
            else:
                key = random.fold_in(self._base_key, i)
                white_noise = random.normal(key, (self.batch_size, self.num_neurons))
            white_noise = white_noise * jnp.asarray(noise_scale)
            colored_noise = (1 - alpha) * colored_noise + white_noise

        # Update internal state
        self._colored_noise_state = colored_noise
        self._last_timestep = t

        return colored_noise

    def get_full_noise_data(self) -> Array:
        """
        Get all input data, including any stored full data.

        Returns:
            Array: The full input data array.
        """
        if self._noise_data is not None:
            if self._noise_data.ndim == 3:
                return self._noise_data
            else:
                raise ValueError("Noise data does not have the expected shape (T, B, N)")
        result = []
        for t in range(self.num_timesteps):
            result.append(self.get(t))
        return jnp.array(result)
