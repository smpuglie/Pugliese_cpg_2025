"""
Simple performance comparison showing JAX scan optimization benefits.

This script compares the performance of the JAX scan-based simulation
against a reference Python for-loop implementation.
"""

import time
import jax.numpy as jnp
from jax import random

try:
    from src.dt_vnc_rnn_jax import create_random_network, InputFunction, NoiseFunction
    
    print("=== JAX Scan Performance Comparison ===\n")
    
    # Create a test network
    key = random.PRNGKey(42)
    num_neurons = 100  # Larger network for better timing
    weights = random.normal(key, (num_neurons, num_neurons)) * 0.05
    
    network = create_random_network(
        num_neurons=num_neurons,
        dt=0.001,
        weights=weights,
        random_seed=42
    )
    
    print(f"Network: {num_neurons} neurons")
    
    # Create input function
    num_timesteps = 1000  # Longer simulation for better timing
    input_func = InputFunction(
        'sinusoidal',
        num_neurons=num_neurons,
        num_timesteps=num_timesteps,
        dt=0.001,
        frequency=10.0,
        amplitude=2.0
    )
    
    noise_func = NoiseFunction(
        'gaussian',
        num_neurons=num_neurons,
        num_timesteps=num_timesteps,
        std=0.2,
        random_seed=123
    )
    
    print(f"Simulation: {num_timesteps} timesteps")
    print()
    
    # Method 1: Standard simulate (JAX scan optimized)
    print("1. Testing JAX scan-optimized simulation...")
    start_time = time.time()
    
    # Run multiple times for better timing
    n_runs = 10
    for _ in range(n_runs):
        result_scan = network.simulate(
            num_timesteps,
            input_func=input_func,
            noise_func_input=noise_func
        )
    
    scan_time = (time.time() - start_time) / n_runs
    print(f"   Average time per run: {scan_time:.4f} seconds")
    print(f"   Final firing rate range: [{jnp.min(result_scan[-1]):.3f}, {jnp.max(result_scan[-1]):.3f}]")
    
    # Method 2: JIT-compiled version
    print("\n2. Testing JIT-compiled version...")
    
    # Pre-compute data (this is done once)
    prep_start = time.time()
    input_data, noise_input_data, noise_recurrent_data = network.prepare_simulation_data(
        num_timesteps, input_func, noise_func, lambda _: 0
    )
    prep_time = time.time() - prep_start
    
    # First call includes JIT compilation time
    print("   First call (includes JIT compilation)...")
    start_time = time.time()
    result_jit = network.simulate_jit(
        num_timesteps, input_data, noise_input_data, noise_recurrent_data
    )
    first_jit_time = time.time() - start_time
    
    # Subsequent calls are much faster
    print("   Subsequent calls (compiled version)...")
    start_time = time.time()
    for _ in range(n_runs):
        result_jit = network.simulate_jit(
            num_timesteps, input_data, noise_input_data, noise_recurrent_data
        )
    jit_time = (time.time() - start_time) / n_runs
    
    print(f"   Data preparation time: {prep_time:.4f} seconds (one-time cost)")
    print(f"   First JIT call time: {first_jit_time:.4f} seconds (includes compilation)")
    print(f"   Average compiled time per run: {jit_time:.4f} seconds")
    print(f"   Final firing rate range: [{jnp.min(result_jit[-1]):.3f}, {jnp.max(result_jit[-1]):.3f}]")
    
    # Method 3: Convenience method
    print("\n3. Testing convenience method...")
    start_time = time.time()
    for _ in range(n_runs):
        result_fast = network.simulate_fast(
            num_timesteps,
            input_func=input_func,
            noise_func_input=noise_func
        )
    fast_time = (time.time() - start_time) / n_runs
    
    print(f"   Average time per run: {fast_time:.4f} seconds")
    print(f"   Final firing rate range: [{jnp.min(result_fast[-1]):.3f}, {jnp.max(result_fast[-1]):.3f}]")
    
    # Verify all methods give identical results
    print("\n4. Verification:")
    max_diff_1_2 = jnp.max(jnp.abs(result_scan - result_jit))
    max_diff_1_3 = jnp.max(jnp.abs(result_scan - result_fast))
    
    print(f"   Max difference between methods: {max(max_diff_1_2, max_diff_1_3):.2e}")
    
    if max_diff_1_2 < 1e-10 and max_diff_1_3 < 1e-10:
        print("   ✅ All methods produce identical results!")
    else:
        print("   ⚠️ Methods produce different results")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Standard simulate():     {scan_time:.4f} sec/run")
    print(f"JIT simulate_jit():      {jit_time:.4f} sec/run (after compilation)")
    print(f"Convenience simulate_fast(): {fast_time:.4f} sec/run")
    
    if jit_time > 0:
        speedup_jit = scan_time / jit_time
        speedup_fast = scan_time / fast_time
        print(f"\nSpeedup with JIT:        {speedup_jit:.1f}x")
        print(f"Speedup with fast:       {speedup_fast:.1f}x")
    
    print(f"\nNote: All methods now use JAX scan internally for optimized performance.")
    print(f"The JIT version provides additional speedup through compilation.")
    
    # Batch performance test
    print("\n" + "=" * 60)
    print("BATCH PERFORMANCE TEST")
    print("=" * 60)
    
    batch_sizes = [1, 5, 10, 20]
    batch_timesteps = 500
    
    for batch_size in batch_sizes:
        input_func_batch = InputFunction(
            'constant',
            num_neurons=num_neurons,
            num_timesteps=batch_timesteps,
            batch_size=batch_size,
            dt=0.001,
            amplitude=1.0
        )
        
        start_time = time.time()
        batch_result = network.simulate_fast(
            batch_timesteps,
            input_func=input_func_batch,
            batch_size=batch_size
        )
        batch_time = time.time() - start_time
        
        neurons_per_sec = (batch_size * num_neurons * batch_timesteps) / batch_time
        print(f"Batch size {batch_size:2d}: {batch_time:.3f} sec, "
              f"{neurons_per_sec/1e6:.1f}M neuron-timesteps/sec")
    
    print(f"\n✅ Performance comparison completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the JAX VNC network module is available.")
    
except Exception as e:
    print(f"❌ Performance test failed with error: {e}")
    import traceback
    traceback.print_exc()
