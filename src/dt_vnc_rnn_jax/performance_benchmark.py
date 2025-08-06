"""
Performance comparison script for JAX scan vs for-loop simulation.

This script demonstrates the performance improvements achieved by using JAX scan
instead of a Python for-loop in the VNC network simulation.
"""

import time
import jax
import jax.numpy as jnp
from jax import random

try:
    from src.dt_vnc_rnn_jax import create_random_network, InputFunction, NoiseFunction
    
    def benchmark_simulation_methods():
        """Compare performance of different simulation methods."""
        print("=== VNC Network Simulation Performance Comparison ===\n")
        
        # Create test network
        key = random.PRNGKey(42)
        num_neurons = 100
        num_timesteps = 1000
        batch_size = 10
        
        weights = random.normal(key, (num_neurons, num_neurons)) * 0.05
        network = create_random_network(
            num_neurons=num_neurons,
            dt=0.001,
            weights=weights,
            random_seed=42
        )
        
        # Create test inputs
        input_func = InputFunction(
            'sinusoidal',
            num_neurons=num_neurons,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            dt=0.001,
            frequency=10.0,
            amplitude=2.0
        )
        
        noise_func = NoiseFunction(
            'gaussian',
            num_neurons=num_neurons,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            std=0.1,
            random_seed=123
        )
        
        print(f"Network: {num_neurons} neurons, {num_timesteps} timesteps, {batch_size} batch size")
        print("-" * 70)
        
        # Method 1: Original simulate method with JAX scan (but function calls)
        print("1. Testing simulate() method (JAX scan + function calls)...")
        start_time = time.time()
        result1 = network.simulate(
            num_timesteps,
            input_func=input_func,
            noise_func_input=noise_func,
            batch_size=batch_size
        )
        time1 = time.time() - start_time
        print(f"   Time: {time1:.4f} seconds")
        print(f"   Output shape: {result1.shape}")
        
        # Method 2: JIT-compiled method with pre-computed data
        print("\n2. Testing simulate_jit() method (JIT + pre-computed data)...")
        
        # Pre-compute data
        start_prep = time.time()
        input_data, noise_input_data, noise_recurrent_data = network.prepare_simulation_data(
            num_timesteps, input_func, noise_func, lambda _: 0, batch_size
        )
        prep_time = time.time() - start_prep
        
        # First run (includes compilation time)
        start_time = time.time()
        result2 = network.simulate_jit(
            num_timesteps, input_data, noise_input_data, noise_recurrent_data,
            batch_size=batch_size
        )
        time2_first = time.time() - start_time
        
        # Second run (compiled, should be faster)
        start_time = time.time()
        result2_second = network.simulate_jit(
            num_timesteps, input_data, noise_input_data, noise_recurrent_data,
            batch_size=batch_size
        )
        time2_second = time.time() - start_time
        
        print(f"   Data preparation time: {prep_time:.4f} seconds")
        print(f"   First run (with compilation): {time2_first:.4f} seconds")
        print(f"   Second run (compiled): {time2_second:.4f} seconds")
        print(f"   Output shape: {result2.shape}")
        
        # Method 3: Convenience method
        print("\n3. Testing simulate_fast() method (convenience method)...")
        start_time = time.time()
        result3 = network.simulate_fast(
            num_timesteps,
            input_func=input_func,
            noise_func_input=noise_func,
            batch_size=batch_size
        )
        time3 = time.time() - start_time
        print(f"   Time: {time3:.4f} seconds")
        print(f"   Output shape: {result3.shape}")
        
        # Verify results are identical
        print("\n4. Verifying results are identical...")
        max_diff_1_2 = jnp.max(jnp.abs(result1 - result2))
        max_diff_1_3 = jnp.max(jnp.abs(result1 - result3))
        max_diff_2_3 = jnp.max(jnp.abs(result2 - result3))
        
        print(f"   Max difference between simulate() and simulate_jit(): {max_diff_1_2:.2e}")
        print(f"   Max difference between simulate() and simulate_fast(): {max_diff_1_3:.2e}")
        print(f"   Max difference between simulate_jit() and simulate_fast(): {max_diff_2_3:.2e}")
        
        # Performance summary
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Original method:      {time1:.4f} seconds")
        print(f"JIT first run:        {time2_first:.4f} seconds (includes compilation)")
        print(f"JIT subsequent runs:  {time2_second:.4f} seconds")  
        print(f"Fast method:          {time3:.4f} seconds")
        
        if time2_second > 0:
            speedup = time1 / time2_second
            print(f"\nSpeedup (JIT vs original): {speedup:.1f}x")
        
        print(f"\nRecommendation:")
        print(f"- Use simulate() for single runs or prototyping")
        print(f"- Use simulate_fast() for best performance in most cases")  
        print(f"- Use simulate_jit() when you need to run many simulations")
        print(f"  with the same parameters (pre-compute data once, reuse many times)")
        
        # Multiple runs test
        print(f"\n5. Testing multiple runs (10 simulations)...")
        
        # Original method
        start_time = time.time()
        for _ in range(10):
            _ = network.simulate(num_timesteps, input_func=input_func, batch_size=batch_size)
        time_original_multi = time.time() - start_time
        
        # JIT method (data preparation done once)
        input_data, noise_input_data, noise_recurrent_data = network.prepare_simulation_data(
            num_timesteps, input_func, lambda _: 0, lambda _: 0, batch_size
        )
        
        start_time = time.time()
        for _ in range(10):
            _ = network.simulate_jit(
                num_timesteps, input_data, noise_input_data, noise_recurrent_data,
                batch_size=batch_size
            )
        time_jit_multi = time.time() - start_time
        
        print(f"   Original method (10 runs): {time_original_multi:.4f} seconds")
        print(f"   JIT method (10 runs):      {time_jit_multi:.4f} seconds")
        
        if time_jit_multi > 0:
            speedup_multi = time_original_multi / time_jit_multi
            print(f"   Speedup for multiple runs: {speedup_multi:.1f}x")

    def test_jit_compilation():
        """Test JIT compilation benefits."""
        print("\n=== JIT Compilation Benefits ===\n")
        
        # Create a smaller network for JIT testing
        key = random.PRNGKey(123)
        num_neurons = 50
        weights = random.normal(key, (num_neurons, num_neurons)) * 0.1
        
        network = create_random_network(
            num_neurons=num_neurons,
            dt=0.001,
            weights=weights,
            random_seed=123
        )
        
        # Prepare data
        input_data = jnp.ones((100, 1, num_neurons)) * 0.5
        noise_data = jnp.zeros((100, 1, num_neurons))
        
        print("Testing JIT compilation overhead...")
        
        # First call (includes compilation)
        start_time = time.time()
        _ = network.simulate_jit(100, input_data, noise_data, noise_data)
        first_call_time = time.time() - start_time
        
        # Subsequent calls (already compiled)
        times = []
        for i in range(5):
            start_time = time.time()
            _ = network.simulate_jit(100, input_data, noise_data, noise_data)
            times.append(time.time() - start_time)
        
        avg_subsequent_time = sum(times) / len(times)
        
        print(f"First call (with compilation): {first_call_time:.4f} seconds")
        print(f"Average subsequent calls:      {avg_subsequent_time:.4f} seconds")
        print(f"Compilation overhead:          {first_call_time - avg_subsequent_time:.4f} seconds")
        
        if avg_subsequent_time > 0:
            speedup = first_call_time / avg_subsequent_time
            print(f"Speedup after compilation:     {speedup:.1f}x")

    if __name__ == "__main__":
        print("JAX VNC Network Performance Benchmarks")
        print("=" * 50)
        
        benchmark_simulation_methods()
        test_jit_compilation()
        
        print(f"\n✅ Benchmarking completed!")
        print(f"\nNotes:")
        print(f"- JAX scan eliminates Python loop overhead")
        print(f"- JIT compilation provides additional speedup")
        print(f"- Pre-computing data reduces function call overhead")
        print(f"- GPU acceleration (if available) provides even greater speedup")

        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the JAX VNC network module is properly installed.")
    
except Exception as e:
    print(f"❌ Benchmark failed with error: {e}")
    import traceback
    traceback.print_exc()
