# RUN DISCRETE SIMULATION
inputF = InputFunction(
    input_type="step",
    num_neurons=num_neurons,
    num_timesteps=num_timesteps,
    dt=dt,
    batch_size=1,
    store_full=True,
    amplitude_before=0,
    amplitude_after=stimulus,
    step_time=pulse_start,
)

firing_rates_Discrete = vncRnn.simulate(
    num_timesteps=num_timesteps,
    input_func=inputF,
    input_noise_std=0,
    recurrent_noise_std=0,
    input_noise_tau=1.0,
    recurrent_noise_tau=1.0,
    initial_state=None,
    batch_size=inputF.batch_size,
)
