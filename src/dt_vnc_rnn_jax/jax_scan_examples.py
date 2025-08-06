"""
Example showing how to use the optimized JAX scan-based simulation.

This demonstrates the three different simulation methods available after the JAX scan optimization:
1. simulate() - Original method, now using JAX scan internally
2. simulate_jit() - JIT-compiled method for maximum performance
3. simulate_fast() - Convenience method that combines data prep and JIT execution
"""

import jax.numpy as jnp
from jax import random

try:
    from src.dt_vnc_rnn_jax import create_random_network, InputFunction, NoiseFunction

    print("=== JAX Scan Optimization Examples ===\n")

    # Create a test network
    key = random.PRNGKey(42)
    num_neurons = 50
    weights = random.normal(key, (num_neurons, num_neurons)) * 0.08

    network = create_random_network(num_neurons=num_neurons, dt=0.001, weights=weights, random_seed=42)

    print(f"Created network with {num_neurons} neurons")

    # Create input and noise functions
    input_func = InputFunction(
        "sinusoidal", num_neurons=num_neurons, num_timesteps=500, dt=0.001, frequency=8.0, amplitude=1.5
    )

    noise_func = NoiseFunction("gaussian", num_neurons=num_neurons, num_timesteps=500, std=0.1, random_seed=123)

    print("Created input and noise functions")

    # Example 1: Using the standard simulate method (now with JAX scan)
    print("\n1. Standard simulate() method (JAX scan internally):")
    firing_rates1 = network.simulate(500, input_func=input_func, noise_func_input=noise_func)
    print(f"   Result shape: {firing_rates1.shape}")
    print(f"   Final firing rate range: [{jnp.min(firing_rates1[-1]):.3f}, {jnp.max(firing_rates1[-1]):.3f}]")

    # Example 2: Using the JIT-compiled method for maximum performance
    print("\n2. JIT-compiled simulate_jit() method:")

    # Pre-compute the data
    input_data, noise_input_data, noise_recurrent_data = network.prepare_simulation_data(
        500, input_func, noise_func, lambda _: 0
    )
    print("   Pre-computed data shapes:")
    print(f"   - Input: {input_data.shape}")
    print(f"   - Noise input: {noise_input_data.shape}")
    print(f"   - Noise recurrent: {noise_recurrent_data.shape}")

    # Run JIT simulation
    firing_rates2 = network.simulate_jit(500, input_data, noise_input_data, noise_recurrent_data)
    print(f"   Result shape: {firing_rates2.shape}")
    print(f"   Final firing rate range: [{jnp.min(firing_rates2[-1]):.3f}, {jnp.max(firing_rates2[-1]):.3f}]")

    # Example 3: Using the convenience method
    print("\n3. Convenience simulate_fast() method:")
    firing_rates3 = network.simulate_fast(500, input_func=input_func, noise_func_input=noise_func)
    print(f"   Result shape: {firing_rates3.shape}")
    print(f"   Final firing rate range: [{jnp.min(firing_rates3[-1]):.3f}, {jnp.max(firing_rates3[-1]):.3f}]")

    # Verify all methods give identical results
    print("\n4. Verification - all methods should give identical results:")
    max_diff_1_2 = jnp.max(jnp.abs(firing_rates1 - firing_rates2))
    max_diff_1_3 = jnp.max(jnp.abs(firing_rates1 - firing_rates3))
    max_diff_2_3 = jnp.max(jnp.abs(firing_rates2 - firing_rates3))

    print(f"   Max difference between methods 1 & 2: {max_diff_1_2:.2e}")
    print(f"   Max difference between methods 1 & 3: {max_diff_1_3:.2e}")
    print(f"   Max difference between methods 2 & 3: {max_diff_2_3:.2e}")

    if max_diff_1_2 < 1e-10 and max_diff_1_3 < 1e-10 and max_diff_2_3 < 1e-10:
        print("   ✅ All methods produce identical results!")
    else:
        print("   ⚠️ Methods produce different results - check implementation")

    # Example 4: Batch simulation with JIT
    print("\n5. Batch simulation example:")
    batch_size = 5

    # Create batch input
    input_func_batch = InputFunction(
        "step",
        num_neurons=num_neurons,
        num_timesteps=300,
        batch_size=batch_size,
        dt=0.001,
        step_time=0.15,  # Step at 150ms
        amplitude_before=0.5,
        amplitude_after=2.0,
    )

    # Run batch simulation
    batch_results = network.simulate_fast(300, input_func=input_func_batch, batch_size=batch_size)

    print(f"   Batch result shape: {batch_results.shape}")
    print(
        f"   Shape meaning: (time_steps+1={batch_results.shape[0]}, "
        f"batch_size={batch_results.shape[1]}, neurons={batch_results.shape[2]})"
    )

    # Show final states for each batch
    for i in range(batch_size):
        final_rates = batch_results[-1, i, :]
        print(f"   Batch {i + 1} final firing rate range: " f"[{jnp.min(final_rates):.3f}, {jnp.max(final_rates):.3f}]")

    # Performance tips
    print("\n" + "=" * 60)
    print("PERFORMANCE TIPS")
    print("=" * 60)
    print("1. Use simulate() for single runs or prototyping")
    print("2. Use simulate_fast() for best balance of convenience and performance")
    print("3. Use simulate_jit() when running many simulations with same parameters:")
    print("   - Pre-compute data once with prepare_simulation_data()")
    print("   - Reuse the pre-computed data for multiple simulate_jit() calls")
    print("4. All methods now benefit from JAX scan optimization")
    print("5. JIT compilation provides additional 2-10x speedup")
    print("6. GPU acceleration (if available) provides even greater speedup")

    print("\n✅ All examples completed successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the JAX VNC network module is available.")

except Exception as e:
    print(f"❌ Example failed with error: {e}")
    import traceback

    traceback.print_exc()
