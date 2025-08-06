"""
Input functions for the VNC Recurrent Neural Network.

This module provides a collection of input functions for generating external stimuli
to neurons. All functions support time-varying inputs and broadcasting to multiple neurons.
"""

import jax.numpy as jnp
from jax import Array
from typing import Union, Any


class InputFunction:
    """
    Object-oriented input function for generating external stimuli to neurons.

    This class allows creating input function objects that store pre-computed input
    patterns and provide efficient access via a get(t) method or by calling the object
    directly as a function. Supports memory-efficient storage for constant or
    time-independent inputs.

    Args:
        input_type: Type of input function ('constant', 'sinusoidal', 'step', etc.)
        num_neurons: Number of neurons
        num_timesteps: Number of time steps
        batch_size: Number of parallel simulations (default: 1)
        dt: Time step size (required for time-dependent functions)
        store_full: If True, store the input in full (T, B, N) format for fast access at the cost of memory.
        **kwargs: Parameters specific to the input function type

    Example:
        >>> # Create a constant input
        >>> input_func = InputFunction('constant', num_neurons=100, num_timesteps=1000,
        ...                          amplitude=5.0)
        >>>
        >>> # Both syntaxes are equivalent:
        >>> current_input = input_func.get(50)  # Method call
        >>> current_input = input_func(50)      # Callable interface
        >>>
        >>> # Create a sinusoidal input for batch simulation
        >>> input_func = InputFunction('sinusoidal', num_neurons=100, num_timesteps=1000,
        ...                          batch_size=10, dt=0.001, frequency=10.0, amplitude=2.0)
        >>> current_input = input_func(50)  # Shape: (10, 100)
    """

    def __init__(
        self,
        input_type: str,
        num_neurons: int,
        num_timesteps: int,
        batch_size: int = 1,
        dt: float = 1e-4,  # seconds
        store_full: bool = False,
        **kwargs: Any,
    ) -> None:
        self.input_type = input_type
        self.num_neurons = num_neurons
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.dt = dt
        self.params = kwargs

        # Input validation for performance-critical parameters
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        # Generate and store the input pattern
        self._input_data = self._generate_input()
        if store_full:
            self.convert_input_to_full_storage()

    def _generate_input(self) -> Array:
        """Generate the input pattern based on the specified type."""
        if self.input_type == "constant":
            return self._constant(**self.params)
        elif self.input_type == "sinusoidal":
            return self._sinusoidal(**self.params)
        elif self.input_type == "step":
            return self._step(**self.params)
        else:
            raise ValueError(f"Unknown input type: {self.input_type}")

    def get(self, t: int) -> Array:
        """
        Get input at timestep t.

        This method provides efficient access to input values at specific timesteps,
        automatically handling different internal storage formats and broadcasting
        as needed.

        Args:
            t: Timestep index (0-based). Must be in range [0, num_timesteps).

        Returns:
            Input array with shape determined by the original input parameters:
            - Scalar: Single value for all neurons and batches
            - (num_neurons,): Values per neuron, same across batches
            - (batch_size, num_neurons): Full batch-specific values
            - (batch_size,): Values per batch, same across neurons

        Raises:
            IndexError: If timestep t is out of valid range
            RuntimeError: If input data has been cleared (call regenerate_storage())

        Example:
            >>> input_func = InputFunction('sinusoidal', num_neurons=100, num_timesteps=1000,
            ...                          batch_size=5, dt=0.001, frequency=10.0)
            >>> current_input = input_func.get(50)  # Shape: (5, 100)
        """
        if t < 0 or t >= self.num_timesteps:
            raise IndexError(f"Timestep {t} out of range [0, {self.num_timesteps})")

        if self._input_data is None:
            raise RuntimeError("Input data has been cleared. Call regenerate_storage() to restore functionality.")

        # Handle different storage formats with detailed comments for clarity
        try:
            # Shape: (T, batch_size, N) - Full precomputed storage
            if self._input_data.ndim == 3:
                result = self._input_data[t]

            # Shape: (batch_size, N) or (T, N) or (T, batch_size) - 2D storage formats
            elif self._input_data.ndim == 2:
                # Shape: (batch_size, N) - Time-independent, batch-specific
                if self._input_data.shape == (self.batch_size, self.num_neurons):
                    result = self._input_data

                # Shape: (T, N) - Time-varying, single batch
                elif self._input_data.shape == (self.num_timesteps, self.num_neurons):
                    result = self._input_data[t]

                # Shape: (T, batch_size) - Time-varying, neuron-independent
                elif self._input_data.shape == (self.num_timesteps, self.batch_size):
                    result = self._input_data[t]
                    # Broadcast across neurons: (batch_size,) -> (batch_size, num_neurons)
                    result = jnp.tile(result[:, jnp.newaxis], (1, self.num_neurons))

                else:
                    # This should never happen with properly generated data
                    raise ValueError(
                        f"Unexpected 2D input data shape: {self._input_data.shape}. "
                        f"Expected (B={self.batch_size}, N={self.num_neurons}), "
                        f"(T={self.num_timesteps}, N={self.num_neurons}), or "
                        f"(T={self.num_timesteps}, B={self.batch_size})."
                    )

            # 1D and 0D storage formats
            elif self._input_data.ndim == 1:
                # Shape: (N,) - Constant across time and batch
                if self._input_data.shape == (self.num_neurons,):
                    result = self._input_data

                # Shape: (T,) - Time-varying, constant across neurons and batches
                elif self._input_data.shape == (self.num_timesteps,):
                    result = self._input_data[t]

                else:
                    raise ValueError(
                        f"Unexpected 1D input data shape: {self._input_data.shape}. "
                        f"Expected (N={self.num_neurons},) or (T={self.num_timesteps},)."
                    )

            # Shape: () - Single constant value
            elif self._input_data.ndim == 0:
                result = self._input_data

            else:
                raise ValueError(f"Unexpected input data dimensions: {self._input_data.ndim}")

            return result

        except IndexError as e:
            # More informative error for indexing issues
            raise IndexError(
                f"Failed to access timestep {t} from input data with shape {self._input_data.shape}. "
                f"This may indicate a bug in input generation. Original error: {e}"
            )

    def __call__(self, t: int) -> Array:
        """
        Make the InputFunction object callable like a function.

        This allows using the object directly as a function: input_func(t) instead of input_func.get(t).
        Both syntaxes are equivalent and provide the same functionality.

        Args:
            t: Timestep index (0-based). Must be in range [0, num_timesteps).

        Returns:
            Input array with shape determined by the original input parameters.

        Example:
            >>> input_func = InputFunction('sinusoidal', num_neurons=100, num_timesteps=1000,
            ...                          batch_size=5, dt=0.001, frequency=10.0)
            >>> current_input = input_func(50)  # Equivalent to input_func.get(50)
        """
        return self.get(t)

    def _constant(self, amplitude: Union[float, Array]) -> Array:
        """
        Constant input across time.

        Args:
            amplitude: Input amplitude(s), scalar or shape (N,) or (batch_size, N)

        Returns:
            Input array, optimally shaped for memory efficiency, shape (), (N,), or (batch_size, N)
        """
        amplitude_arr = jnp.asarray(amplitude)
        scalar_shape = amplitude_arr.ndim == 0
        neuron_shape = amplitude_arr.shape == (self.num_neurons,)
        batch_neuron_shape = amplitude_arr.shape == (self.batch_size, self.num_neurons)

        if scalar_shape or neuron_shape or batch_neuron_shape:
            return amplitude_arr
        else:
            raise ValueError(
                f"Amplitude shape {amplitude_arr.shape} incompatible with {self.batch_size}x{self.num_neurons} neurons"
            )

    def _sinusoidal(
        self,
        frequency: Union[float, Array],
        amplitude: Union[float, Array],
        phase: Union[float, Array] = 0.0,
    ) -> Array:
        """
        Sinusoidal input function with optimized broadcasting.

        Generates sinusoidal inputs with configurable frequency, amplitude, and phase
        parameters. Supports efficient memory usage by determining optimal output
        shape based on parameter complexity.

        Args:
            frequency: Oscillation frequency(ies) in Hz. Supported shapes:
                - scalar: Same frequency for all neurons and batches
                - (N,): Different frequency per neuron, same across batches
                - (batch_size, N): Different frequency per neuron and batch
                - (batch_size, 1): Different frequency per batch, same across neurons
            amplitude: Input amplitude(s). Same shape options as frequency.
            phase: Phase offset(s) in radians. Same shape options as frequency.

        Returns:
            Sinusoidal input array with memory-efficient shape:
            - (T,): If all parameters are scalar
            - (T, N): If parameters vary by neuron only
            - (T, batch_size): If parameters vary by batch only
            - (T, batch_size, N): If parameters vary by batch and neuron

        Raises:
            ValueError: If dt is not specified or parameter shapes are incompatible

        Performance Notes:
            - Uses optimized broadcasting to minimize memory allocation
            - Output shape is automatically minimized based on parameter complexity
            - For large simulations, consider using scalar or (N,) parameters when possible
        """
        if jnp.any(jnp.asarray(frequency) < 0):
            raise ValueError("Frequency must be non-negative")

        # Pre-compute time array with optimal shape for broadcasting
        t = jnp.arange(self.num_timesteps).reshape(-1, 1, 1) * self.dt  # Shape: (T,1,1)

        # Broadcast all parameters with error checking
        try:
            freq = self._broadcast_param(frequency, "frequency")
            amp = self._broadcast_param(amplitude, "amplitude")
            ph = self._broadcast_param(phase, "phase")
        except ValueError as e:
            raise ValueError(f"Parameter broadcasting failed: {e}")

        # Compute sinusoidal input with optimized broadcasting
        # This single operation handles all shape combinations efficiently
        input_signal = amp * jnp.sin(2 * jnp.pi * freq * t + ph)

        # Return with minimal necessary dimensions (squeeze removes size-1 dimensions)
        return input_signal.squeeze()

    def _step(
        self,
        step_time: float,
        amplitude_before: Union[float, Array],
        amplitude_after: Union[float, Array],
    ) -> Array:
        """
        Step input function.

        Args:
            step_time: Time at which step occurs
            amplitude_before: Input amplitude before step, scalar or shape (N,) or (batch_size, N)
            amplitude_after: Input amplitude after step, scalar or shape (N,) or (batch_size, N)

        Returns:
            Input array, memory-efficiently shaped based on input parameters:
            - Shape (T,): if both amplitudes are scalar
            - Shape (T, N): if amplitudes are shape (N,) and batch_size=1
            - Shape (T, batch_size, N): if amplitudes are shape (batch_size, N) or batch_size>1
        """
        if self.dt is None:
            raise ValueError("dt must be specified for step input")

        # Input validation for performance-critical parameters
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        # Pre-compute time array with optimal shape for broadcasting
        t = jnp.arange(self.num_timesteps).reshape(-1, 1, 1) * self.dt  # Shape: (T,1,1)

        # Broadcast all parameters with error checking
        try:
            amplitude_before = self._broadcast_param(amplitude_before, "amplitude_before")
            amplitude_after = self._broadcast_param(amplitude_after, "amplitude_after")
        except ValueError as e:
            raise ValueError(f"Parameter broadcasting failed: {e}")

        # Compute sinusoidal input with optimized broadcasting
        # This single operation handles all shape combinations efficiently
        input_signal = amplitude_before * (t < step_time) + amplitude_after * (t >= step_time)

        # Return with minimal necessary dimensions (squeeze removes size-1 dimensions)
        return input_signal.squeeze()

    def _broadcast_param(self, param: Union[float, Array], param_name: str) -> Array:
        """
        Helper function to broadcast parameters with comprehensive error checking.

        Returns parameters in shape (1, batch_size_or_1, num_neurons_or_1) format
        for efficient broadcasting with time array.
        """
        param_array = jnp.atleast_1d(param)

        # Scalar case - most memory efficient
        if param_array.size == 1:
            return param_array.reshape(1, 1, 1)  # Shape: (1,1,1)

        # Shape (N,) case - per-neuron variation
        elif param_array.shape == (self.num_neurons,):
            return param_array.reshape(1, 1, self.num_neurons)  # Shape: (1,1,N)

        # Shape (B, 1) or (B, N) case - per-batch variation
        elif param_array.shape == (self.batch_size, 1):
            return param_array.reshape(1, self.batch_size, 1)  # Shape: (1,B,1)
        elif param_array.shape == (self.batch_size, self.num_neurons):
            return param_array.reshape(1, self.batch_size, self.num_neurons)  # Shape: (1,B,N)

        else:
            raise ValueError(
                f"{param_name} shape {param_array.shape} is not compatible. "
                f"Expected scalar, ({self.num_neurons},), ({self.batch_size}, {self.num_neurons}), "
                f"or ({self.batch_size}, 1). Got shape {param_array.shape}."
            )

    def convert_input_to_full_storage(self) -> None:
        """
        Force the input data to be stored in full (T, B, N) shape.

        This method converts the internally stored input data from its memory-efficient
        format to the full (num_timesteps, batch_size, num_neurons) format. This can
        be useful when you need repeated fast access to all timesteps but comes at
        the cost of increased memory usage.

        Note:
            This operation is irreversible within the same object. To return to
            memory-efficient storage, create a new InputFunction object.
        """
        if self._input_data is None:
            raise RuntimeError("Input data has been cleared. Call regenerate_storage() first.")

        # Handle different storage formats
        # Shape: (T, batch_size, N)
        if self._input_data.ndim == 3:
            return  # Already in full format

        # Shape: (batch_size, N) or (T, N) or (T, batch_size)
        elif self._input_data.ndim == 2:
            # Shape: (batch_size, N) -- time-independent
            if self._input_data.shape == (self.batch_size, self.num_neurons):
                self._input_data = jnp.tile(
                    self._input_data[jnp.newaxis, :, :], (self.num_timesteps, 1, 1)
                )  # Shape: (T, B, N)
                return

            # Shape:  (T, N) -- Time-varying, single batch
            elif self._input_data.shape == (self.num_timesteps, self.num_neurons):
                self._input_data = jnp.tile(
                    self._input_data[:, jnp.newaxis, :], (1, self.batch_size, 1)
                )  # Shape: (T, B, N)
                return

            # Shape: (T, batch_size) -- time-varying, neuron-independent
            elif self._input_data.shape == (self.num_timesteps, self.batch_size):
                self._input_data = jnp.tile(
                    self._input_data[:, jnp.newaxis], (1, 1, self.num_neurons)
                )  # Shape: (T, B, N)
                return

            # Unexpected shape
            else:
                raise ValueError(
                    f"Unexpected input data shape: {self._input_data.shape}. Does not match (B,N), (T,N), or (T,B)."
                )
        # Shape: (N,) -- constant across time and batch
        elif self._input_data.ndim == 1 and self._input_data.shape == (self.num_neurons,):
            self._input_data = jnp.tile(
                self._input_data[jnp.newaxis, jnp.newaxis, :], (self.num_timesteps, self.batch_size, 1)
            )  # Shape: (T, B, N)
            return
        # Shape: (T,) -- time-varying, neuron- and batch-independent
        elif self._input_data.ndim == 1 and self._input_data.shape == (self.num_timesteps,):
            self._input_data = jnp.tile(
                self._input_data[:, jnp.newaxis, jnp.newaxis], (1, self.batch_size, self.num_neurons)
            )  # Shape: (T, B, N)
            return
        # Shape () -- single constant value
        elif self._input_data.ndim == 0:
            self._input_data = jnp.full(
                (self.num_timesteps, self.batch_size, self.num_neurons), self._input_data
            )  # Shape: (T, B, N)
            return
        else:
            raise ValueError(
                f"Unexpected input data shape: {self._input_data.shape}. "
                "Does not match (T,B,N), (B,N), (T,N), (T,B), (N,), (T,) or ()."
            )

    def convert_input_to_memory_efficient_storage(self) -> None:
        self._input_data = self._generate_input()
        return

    def get_full_input_data(self) -> Array:
        """
        Get all input data, including any stored full data.

        Returns:
            Array: The full input data array.
        """
        if self._input_data.ndim == 3:
            return self._input_data
        else:
            self.convert_input_to_full_storage()
            result = self._input_data
            self.convert_input_to_memory_efficient_storage()
            return result

    def get_memory_usage(self) -> dict:
        """
        Get information about the memory usage of the stored input data.

        Returns:
            Dictionary containing memory usage information:
            - 'storage_shape': Shape of the stored array
            - 'storage_size_bytes': Size in bytes of the stored array
            - 'storage_size_mb': Size in MB of the stored array
            - 'full_shape': What the full (T, B, N) shape would be
            - 'full_size_bytes': What the full storage size would be in bytes
            - 'full_size_mb': What the full storage size would be in MB
            - 'memory_efficiency': Ratio of current storage to full storage
        """
        current_size_bytes = (
            self._input_data.nbytes if hasattr(self._input_data, "nbytes") else self._input_data.size * 4
        )
        current_size_mb = current_size_bytes / (1024**2)

        full_shape = (self.num_timesteps, self.batch_size, self.num_neurons)
        full_size_bytes = self.num_timesteps * self.batch_size * self.num_neurons * 8  # 8 bytes for float64
        full_size_mb = full_size_bytes / (1024**2)

        efficiency = 1 - current_size_bytes / full_size_bytes if full_size_bytes > 0 else 0

        return {
            "storage_shape": self._input_data.shape,
            "storage_size_bytes": current_size_bytes,
            "storage_size_mb": current_size_mb,
            "full_shape": full_shape,
            "full_size_bytes": full_size_bytes,
            "full_size_mb": full_size_mb,
            "memory_efficiency": efficiency,
        }
