"""
JAX Conversion Summary and Usage Examples

This document summarizes the conversion from NumPy to JAX and provides usage examples
demonstrating the key improvements and changes.

## Key Changes Made:

### 1. Random Number Generation
- **Before (NumPy)**: Used np.random.seed() and np.random.normal()
- **After (JAX)**: Uses jax.random.PRNGKey() and jax.random.normal()
- **Benefit**: Deterministic, reproducible, and parallelizable random number generation

### 2. Array Operations
- **Before**: numpy arrays (np.ndarray)
- **After**: JAX arrays (jax.Array)
- **Benefit**: GPU/TPU acceleration, automatic differentiation capability

### 3. In-place Operations
- **Before**: firing_rates[t + 1] = r
- **After**: firing_rates = firing_rates.at[t + 1].set(r)
- **Benefit**: Functional programming paradigm, better for JIT compilation

### 4. Import Changes
- numpy → jax.numpy
- scipy.sparse removed (JAX handles dense arrays efficiently)
- Added proper type hints with jax.Array

## Usage Examples:
"""

# Example 1: Creating a network with deterministic random parameters
import jax.numpy as jnp
from jax import random
from src.dt_vnc_rnn_jax import create_random_network, InputFunction, NoiseFunction

def example_basic_usage():
    """Basic usage example showing network creation and simulation."""
    print("=== Example 1: Basic Network Usage ===")
    
    # Create deterministic random key
    key = random.PRNGKey(42)
    
    # Create network parameters
    num_neurons = 50
    weights = random.normal(key, (num_neurons, num_neurons)) * 0.1
    
    # Create network with random parameters
    network = create_random_network(
        num_neurons=num_neurons,
        dt=0.001,
        weights=weights,
        random_seed=42
    )
    
    # Create input function
    input_func = InputFunction(
        'sinusoidal', 
        num_neurons=num_neurons, 
        num_timesteps=1000,
        dt=0.001,
        frequency=10.0,
        amplitude=2.0
    )
    
    # Run simulation
    firing_rates = network.simulate(1000, input_func=input_func)
    print(f"Simulation completed. Output shape: {firing_rates.shape}")
    return firing_rates

def example_batch_simulation():
    """Example showing batch simulation with multiple parallel runs."""
    print("\n=== Example 2: Batch Simulation ===")
    
    key = random.PRNGKey(123)
    num_neurons = 20
    batch_size = 5
    
    # Create network
    weights = random.normal(key, (num_neurons, num_neurons)) * 0.05
    network = create_random_network(
        num_neurons=num_neurons,
        dt=0.001,
        weights=weights,
        random_seed=123
    )
    
    # Create batch input
    input_func = InputFunction(
        'constant',
        num_neurons=num_neurons,
        num_timesteps=500,
        batch_size=batch_size,
        amplitude=1.5
    )
    
    # Create batch noise
    noise_func = NoiseFunction(
        'gaussian',
        num_neurons=num_neurons,
        num_timesteps=500,
        batch_size=batch_size,
        std=0.1,
        random_seed=456
    )
    
    # Run batch simulation
    firing_rates = network.simulate(
        500, 
        input_func=input_func,
        noise_func_input=noise_func,
        batch_size=batch_size
    )
    
    print(f"Batch simulation completed. Output shape: {firing_rates.shape}")
    print(f"Shape interpretation: (time_steps+1={firing_rates.shape[0]}, "
          f"batch_size={firing_rates.shape[1]}, num_neurons={firing_rates.shape[2]})")
    
    return firing_rates

def example_colored_noise():
    """Example using colored (correlated) noise."""
    print("\n=== Example 3: Colored Noise ===")
    
    key = random.PRNGKey(789)
    num_neurons = 30
    
    # Create network
    weights = random.normal(key, (num_neurons, num_neurons)) * 0.08
    network = create_random_network(
        num_neurons=num_neurons,
        dt=0.001,
        weights=weights,
        random_seed=789
    )
    
    # Create colored noise with correlation time
    colored_noise = NoiseFunction(
        'colored_noise',
        num_neurons=num_neurons,
        num_timesteps=800,
        dt=0.001,
        std=0.2,
        correlation_time=0.05,  # 50ms correlation time
        random_seed=321
    )
    
    # Simple constant input
    input_func = InputFunction(
        'constant',
        num_neurons=num_neurons,
        num_timesteps=800,
        amplitude=0.5
    )
    
    # Simulation with colored noise
    firing_rates = network.simulate(
        800,
        input_func=input_func,
        noise_func_input=colored_noise
    )
    
    print(f"Colored noise simulation completed. Output shape: {firing_rates.shape}")
    return firing_rates

def example_step_input():
    """Example with step input function."""
    print("\n=== Example 4: Step Input ===")
    
    key = random.PRNGKey(555)
    num_neurons = 25
    
    # Create network
    weights = random.normal(key, (num_neurons, num_neurons)) * 0.06
    network = create_random_network(
        num_neurons=num_neurons,
        dt=0.001,
        weights=weights,
        random_seed=555
    )
    
    # Create step input (low → high at t=0.5s)
    step_input = InputFunction(
        'step',
        num_neurons=num_neurons,
        num_timesteps=1000,
        dt=0.001,
        step_time=0.5,
        amplitude_before=0.5,
        amplitude_after=3.0
    )
    
    # Run simulation
    firing_rates = network.simulate(1000, input_func=step_input)
    
    print(f"Step input simulation completed. Output shape: {firing_rates.shape}")
    print(f"Input changes from 0.5 to 3.0 at time step {int(0.5/0.001)} (0.5s)")
    
    return firing_rates

def performance_comparison_notes():
    """Notes on performance improvements with JAX."""
    print("\n=== Performance Notes ===")
    print("""
    JAX Advantages over NumPy:
    
    1. **GPU/TPU Acceleration**: 
       - Automatic GPU utilization when available
       - Significant speedup for large networks
    
    2. **JIT Compilation**:
       - Use @jax.jit decorator on simulation functions
       - First run compiles, subsequent runs are much faster
    
    3. **Vectorization**:
       - Better support for batch operations
       - Efficient parallel simulation of multiple conditions
    
    4. **Memory Efficiency**:
       - Functional array updates prevent unnecessary copies
       - Better memory management for large simulations
    
    5. **Reproducibility**:
       - Deterministic random number generation
       - Same results across different hardware
    
    Example JIT usage:
    ```python
    import jax
    
    @jax.jit
    def fast_simulation(network, num_steps, input_func):
        return network.simulate(num_steps, input_func=input_func)
    ```
    """)

if __name__ == "__main__":
    print("JAX-based VNC RNN Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_batch_simulation()
        example_colored_noise()
        example_step_input()
        performance_comparison_notes()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
