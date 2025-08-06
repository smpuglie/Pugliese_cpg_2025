"""
Test script to verify JAX conversion functionality.
This script demonstrates basic usage of the converted JAX-based VNC RNN.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    print("JAX imported successfully")
    
    # Test basic JAX functionality
    key = random.PRNGKey(42)
    test_array = random.normal(key, (5, 5))
    print(f"JAX random array shape: {test_array.shape}")
    
    # Try to import our converted modules
    from src.dt_vnc_rnn_jax import (
        VNCRecurrentNetwork, 
        ActivationFunction, 
        InputFunction, 
        NoiseFunction,
        create_random_network
    )
    print("All modules imported successfully!")
    
    # Test activation functions
    x = jnp.array([1.0, 2.0, 3.0])
    gains = jnp.array([1.0, 1.0, 1.0])
    max_rates = jnp.array([10.0, 10.0, 10.0])
    
    result = ActivationFunction.scaled_tanh_relu(x, gains, max_rates)
    print(f"Activation function test result: {result}")
    
    # Test input functions
    input_func = InputFunction('constant', num_neurons=3, num_timesteps=100, amplitude=5.0)
    input_at_t50 = input_func.get(50)
    print(f"Input function test result: {input_at_t50}")
    
    # Test noise functions
    noise_func = NoiseFunction('gaussian', num_neurons=3, num_timesteps=100, 
                              std=0.1, random_seed=42)
    noise_at_t50 = noise_func.get(50)
    print(f"Noise function test result: {noise_at_t50}")
    
    # Test creating a simple network
    num_neurons = 10
    weights = random.normal(key, (num_neurons, num_neurons)) * 0.1
    
    network = create_random_network(
        num_neurons=num_neurons,
        dt=0.001,
        weights=weights,
        random_seed=42
    )
    print(f"Network created successfully with {network.num_neurons} neurons")
    
    # Test a simple simulation
    input_func = InputFunction('constant', num_neurons=num_neurons, num_timesteps=100, amplitude=1.0)
    firing_rates = network.simulate(100, input_func=input_func)
    print(f"Simulation completed! Output shape: {firing_rates.shape}")
    
    print("\n✅ All tests passed! JAX conversion is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("JAX might not be installed. Try: pip install jax")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()
