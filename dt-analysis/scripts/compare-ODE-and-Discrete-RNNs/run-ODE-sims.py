from time import time
import pandas as pd
import jax.numpy as jnp
from jax import random
from src.vnc_sim import run_single_simulation
from setup_sim_params import (
    setup_simulation_parameters,
    process_VNC_data,
    create_stimulus,
    SIMULATION_TIME,
    DTs,
    SAVE_DT,
    PULSE_START,
    RANDOM_SEEDs,
    STIMULUS_STRENGTHs,
    R_TOLs,
    A_TOLs,
)


if __name__ == "__main__":

    num_timesteps_save = int(SIMULATION_TIME / SAVE_DT)

    weights, neuron_sizes, num_neurons, stim_neuron = process_VNC_data()

    list_dt = []
    list_rtol = []
    list_atol = []
    list_stimulus_strength = []
    list_random_seed = []
    list_firing_rates = []
    list_runtime = []
    list_avg_runtime = []

    for dt in DTs:
        for r_tol in R_TOLs:
            for a_tol in A_TOLs:
                print(f"Running simulations for dt={dt}, r_tol={r_tol}, a_tol={a_tol}")
                num_sims = 0
                t1 = time()
                for stimulus_strength in STIMULUS_STRENGTHs:
                    stimulus = create_stimulus(stim_neuron, stimulus_strength, num_neurons)
                    for random_seed in RANDOM_SEEDs:
                        t1_sim = time()
                        (
                            vncRnn,
                            num_timesteps,
                            dt,
                        ) = setup_simulation_parameters(
                            simulation_time=SIMULATION_TIME,
                            dt=dt,
                            random_seed=random_seed,
                            weights=weights,
                            neuron_sizes=neuron_sizes,
                            num_neurons=num_neurons,
                        )

                        # Run the ODE simulation
                        firing_rates_ODE = run_single_simulation(
                            W=vncRnn.weights * vncRnn.relative_inputs_scale,
                            tau=vncRnn.time_constants,
                            a=vncRnn.gains,
                            threshold=vncRnn.thresholds,
                            fr_cap=vncRnn.max_firing_rates,
                            inputs=stimulus,
                            t_axis=jnp.arange(1, num_timesteps_save) * SAVE_DT,
                            T=SIMULATION_TIME,
                            dt=dt,
                            pulse_start=PULSE_START,
                            pulse_end=SIMULATION_TIME,
                            r_tol=r_tol,
                            a_tol=a_tol,
                            noise_stdv=0,
                            key=random.key(random_seed),
                        )

                        # Append results
                        num_sims += 1
                        list_dt.append(dt)
                        list_rtol.append(r_tol)
                        list_atol.append(a_tol)
                        list_stimulus_strength.append(stimulus_strength)
                        list_random_seed.append(random_seed)
                        list_firing_rates.append(firing_rates_ODE)
                        t2_sim = time()
                        print(f"    Simulation {num_sims} took {t2_sim - t1_sim:.4f} seconds.")
                t2 = time()
                print(f"{num_sims} simulations took {t2 - t1:.2f} seconds, or {(t2 - t1) / num_sims:.4f} sec/sims.")
                list_avg_runtime.extend([(t2 - t1) / num_sims] * num_sims)

    sims_df = pd.DataFrame(
        {
            "dt": list_dt,
            "r_tol": list_rtol,
            "a_tol": list_atol,
            "stimulus_strength": list_stimulus_strength,
            "random_seed": list_random_seed,
            "firing_rates": list_firing_rates,
            "runtime": list_runtime,
            "avg_runtime": list_avg_runtime,
            "save_dt": [SAVE_DT] * len(list_dt),
            "simulation_time": [SIMULATION_TIME] * len(list_dt),
        }
    )

    sims_df.to_pickle("sims_df_ODE.pkl")
