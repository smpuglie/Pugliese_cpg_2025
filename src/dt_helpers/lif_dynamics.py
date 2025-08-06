import numpy as np
from typing import Callable


def run_LIF_network(params: dict, base_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_time_steps = int(params["simulation_time"] / params["dt"])
    num_refractory_steps = int(params["t_refractory"] / params["dt"])
    num_delay_steps = int(params["t_delay"] / params["dt"])
    current_V = np.ones(params["network_size"]) * params["V_init"]
    current_spikes = np.zeros(params["network_size"]).astype(bool)
    history_V = []
    history_spikes = []

    refractory_steps_left = np.zeros(params["network_size"]).astype(int)
    synaptic_input = np.zeros(params["network_size"])
    modify_synaptic_input = np.zeros(params["network_size"]).astype(int)

    for _ in range(num_time_steps):
        modify_synaptic_input -= 1

        # process refractory period
        refractory_check = refractory_steps_left > 0
        current_V[refractory_check] = params["V_reset"]
        refractory_steps_left[refractory_check] -= 1

        # check for spikes
        current_spikes = current_V >= params["V_thresh"]
        current_V[current_spikes] = params["V_reset"]
        refractory_steps_left[current_spikes] = num_refractory_steps
        modify_synaptic_input[current_spikes] = num_delay_steps

        # decay synaptic inputs
        synaptic_input = synaptic_input * (1 - params["dt"] / params["tau_synaptic"])

        # delay synaptic input after pre-spike
        synaptic_input = (
            synaptic_input + (params["network_weights"] * (modify_synaptic_input == 0)).sum(1) * params["w_synaptic"]
        )

        # 0 synaptic input after self-spike
        synaptic_input[current_spikes] = 0

        # update voltages
        d_current_V = (
            (
                synaptic_input
                - (current_V - params["V_rest"])
                + base_input * params["network_input"] / params["membrane_conductance"]
            )
            * params["dt"]
            / params["tau_membrane"]
        )
        current_V = current_V + d_current_V

        # record history
        history_V.append(current_V)
        history_spikes.append(current_spikes)

    return np.array(history_V), np.array(history_spikes)
