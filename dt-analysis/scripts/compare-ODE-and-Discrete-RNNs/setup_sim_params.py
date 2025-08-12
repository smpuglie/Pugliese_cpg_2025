import pandas as pd
from time import time
import jax.numpy as jnp
from src.dt_vnc_rnn import create_random_network

SIMULATION_TIME = 3.0
DTs = [1e-3, 1e-4, 1e-5, 1e-6]
SAVE_DT = 1e-3
PULSE_START = 0.2
RANDOM_SEEDs = [i for i in range(10)]
STIMULUS_STRENGTHs = [200, 250, 300]
R_TOLs = [1e-5, 1e-7]
A_TOLs = [1e-7, 1e-9]


def process_VNC_data(
    descending_neuron_body_ID=10093,  # stimulate neuron 10093 - DNg100 / BDN2 / DNxl058
    weights_pd_filename="../../../data/manc t1 connectome data/W_20231020_DNtoMN_unsorted.csv",
    neuron_properties_filename="../../../data/manc t1 connectome data/wTable_20231020_DNtoMN_unsorted_withModules.csv",
):
    # load data
    weights_pd = pd.read_csv(weights_pd_filename)
    neuron_properties = pd.read_csv(neuron_properties_filename, index_col=0)
    # extract weights
    weights = (
        weights_pd.drop(columns="bodyId_pre").to_numpy().astype(float).T
    )  # transpose to have shape (n_post, n_pre)
    assert weights.shape[0] == weights.shape[1], "Weights matrix must be square"
    num_neurons = weights.shape[0]
    stim_neuron = jnp.where((neuron_properties["bodyId"] == descending_neuron_body_ID).to_numpy())[0][0]
    return weights, neuron_properties["size"].to_numpy(), num_neurons, stim_neuron


def create_stimulus(stim_neuron, stimulus_strength, num_neurons):
    stimulus = jnp.zeros(num_neurons)
    stimulus = stimulus.at[stim_neuron].set(stimulus_strength)
    return stimulus


def setup_simulation_parameters(
    simulation_time,
    dt,
    random_seed,
    weights,
    neuron_sizes,
    num_neurons,
):
    t1 = time()
    num_timesteps = int(simulation_time / dt)

    # generate VNC RNN
    vncRnn = create_random_network(
        num_neurons=num_neurons,
        dt=dt,
        weights=weights,
        activation_function="scaled_tanh_relu",
        gains_stats={
            "distribution": "normal",
            "mean": 1.0,
            "std": 0.2,
        },
        max_firing_rates_stats={
            "distribution": "normal",
            "mean": 200.0,
            "std": 15.0,
        },
        thresholds_stats={
            "distribution": "normal",
            "mean": 4.0,
            "std": 0.5,
        },
        time_constants_stats={
            "distribution": "normal",
            "mean": 20e-3,
            "std": 2.5e-3,
        },
        neuron_sizes=neuron_sizes,
        random_seed=random_seed,
    )
    t2 = time()
    print(f"----- {t2 - t1:.4f} seconds. -----", end=" ")
    return (
        vncRnn,
        num_timesteps,
        dt,
    )
