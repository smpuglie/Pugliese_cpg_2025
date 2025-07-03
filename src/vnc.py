import time

import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax import vmap
from omegaconf import DictConfig

from src.sim_utils import (
    load_W,
    load_wTable,
    rate_equation_half_tanh,
    sample_positive_truncnorm,
    set_sizes,
)


def run(
    W,
    tAxis,
    T,
    dt,
    inputs,
    tau,
    threshold,
    a,
    frCap,
    excSynapseMultiplier,
    inhSynapseMultiplier,
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Run the simulation for a single parameter set."""
    Wt = jnp.transpose(W)  # correct orientation for matrix multiplication

    # Reweight W
    Wt_exc = jnp.maximum(Wt, 0)
    Wt_inh = jnp.minimum(Wt, 0)
    Wt_reweighted = excSynapseMultiplier * Wt_exc + inhSynapseMultiplier * Wt_inh

    R0 = jnp.zeros([len(W[0])])  # initialize firing rates

    # Solve using Diffrax
    term = ODETerm(rate_equation_half_tanh)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.array(tAxis))
    odeSolution = diffeqsolve(
        term,
        solver,
        0,
        T,
        dt,
        R0,
        args=(tAxis, inputs, tau, Wt_reweighted, threshold, a, frCap),
        saveat=saveat,
        stepsize_controller=PIDController(rtol=r_tol, atol=a_tol),
        max_steps=5000000,
    )

    return jnp.transpose(jnp.array(odeSolution.ys))


class VNCNet:

    def __init__(self, cfg: DictConfig):
        """Initialize VNCNet with hydra config"""
        self.cfg = cfg
        self._validate_config()
        self._load_connectivity()
        self._setup_simulation_parameters()
        self._setup_neuron_parameters()

    def _validate_config(self) -> None:
        required_sections = {
            "experiment": ["name", "wPath", "dfPath", "stimNeurons", "stimI", "seed"],
            "sim": ["T", "dt", "pulseStart", "pulseEnd"],
            "neuron_params": [
                "tauMean",
                "tauStdv",
                "aMean",
                "aStdv",
                "thresholdMean",
                "thresholdStdv",
                "frcapMean",
                "frcapStdv",
                "excitatoryMultiplier",
                "inhibitoryMultiplier",
            ],
        }

        for section, keys in required_sections.items():
            if not hasattr(self.cfg, section):
                raise ValueError(f"Missing required config section: {section}")
            section_cfg = getattr(self.cfg, section)
            for key in keys:
                if not hasattr(section_cfg, key):
                    raise ValueError(f"Missing required config section: {section}")

    def _load_connectivity(self) -> None:
        """Load connections"""

        w_path = self.cfg.experiment.wPath
        df_path = self.cfg.experiment.dfPath

        self.W = load_W(w_path)
        self.W_table = load_wTable(df_path)

    def _setup_simulation_parameters(self) -> None:
        """Setup simulation parameters from config"""
        self.n_neurons = self.W.shape[0]

        # Handle stimulation parameters
        self.stim_neurons = np.array(self.cfg.experiment.stimNeurons, dtype=np.int64)
        self.stim_input = np.array(self.cfg.experiment.stimI)

        self.seeds = np.array(self.cfg.experiment.seed)

        self.num_sims = len(self.seeds)

        # Simulation timing
        self.T = float(self.cfg.sim.T)
        self.dt = float(self.cfg.sim.dt)
        self.pulse_start = float(self.cfg.sim.pulseStart)
        self.pulse_end = float(self.cfg.sim.pulseEnd)

        # Create time axis
        self.t_axis = np.arange(0, self.T, self.dt)

        # Optional parameters with defaults
        self.adjust_stim_I = getattr(self.cfg.sim, "adjustStimI", False)
        self.max_iters = getattr(self.cfg.sim, "maxIters", 10)
        self.n_active_upper = getattr(self.cfg.sim, "nActiveUpper", 500)
        self.n_active_lower = getattr(self.cfg.sim, "nActiveLower", 5)
        self.n_high_fr_upper = getattr(self.cfg.sim, "nHighFrUpper", 100)

        # Remove neurons if specified
        self.remove_neurons = getattr(self.cfg.experiment, "removeNeurons", [])

    def _setup_neuron_parameters(self) -> None:
        """Setup neuron parameters from config"""
        np_cfg = self.cfg.neuron_params

        # Store parameter means and standard deviations
        self.tau_mean = float(np_cfg.tauMean)
        self.tau_stdv = float(np_cfg.tauStdv)
        self.a_mean = float(np_cfg.aMean)
        self.a_stdv = float(np_cfg.aStdv)
        self.threshold_mean = float(np_cfg.thresholdMean)
        self.threshold_stdv = float(np_cfg.thresholdStdv)
        self.fr_cap_mean = float(np_cfg.frcapMean)
        self.fr_cap_stdv = float(np_cfg.frcapStdv)

        # multipliers
        self.exc_synapse_multiplier = float(np_cfg.excitatoryMultiplier)
        self.inh_synapse_multiplier = float(np_cfg.inhibitoryMultiplier)

    def _generate_pulse_input(
        self,
        start_time: float,
        stop_time: float,
        amplitudes: np.ndarray,
        stim_neurons: np.ndarray,
    ) -> np.ndarray:
        """
        Generate pulse input for neural network simulation.

        Args:
            start_time: Pulse start time (seconds)
            stop_time: Pulse stop time (seconds)
            amplitudes: Array of pulse amplitudes for each simulation [nSims,]
            stim_neurons: Array of neuron indices to stimulate [nStimNeurons,]

        Returns:
            Input array of shape [nSims, nNeurons, nTimeSteps]
        """
        # Calculate time indices
        start_idx = int(np.round(start_time / self.dt))
        stop_idx = int(np.round(stop_time / self.dt))

        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        stop_idx = min(len(self.t_axis), stop_idx)

        # Initialize input array
        input_array = np.zeros((self.num_sims, self.n_neurons, len(self.t_axis)))

        # Vectorized assignment using broadcasting
        # Shape: [nSims, 1, 1] * [1, nStimNeurons, 1] -> [nSims, nStimNeurons, nTimeSteps]
        input_array[
            np.ix_(
                range(self.num_sims),  # All simulations
                stim_neurons,  # Stimulated neurons
                range(start_idx, stop_idx),  # Time window
            )
        ] = amplitudes[:, np.newaxis, np.newaxis]

        return input_array

    def _generate_input(self) -> np.ndarray:
        """Generate input matrix for all simulations using improved pulse method."""
        return self._generate_pulse_input(
            start_time=self.pulse_start,
            stop_time=self.pulse_end,
            amplitudes=self.stim_input,
            stim_neurons=self.stim_neurons,
        )

    def simulate(self) -> dict:
        """Run the Simulation"""
        try:
            # Sample neuron parameters
            # TODO: fix this logic:
            neuron_seeds = np.zeros([4, self.num_sims], dtype=int)
            for sim in range(self.num_sims):
                neuron_seeds[:, sim] = np.random.default_rng(self.seeds[sim]).integers(
                    10000, size=4
                )

            tau = sample_positive_truncnorm(
                self.tau_mean,
                self.tau_stdv,
                self.n_neurons,
                self.num_sims,
                seeds=neuron_seeds[0],
            )
            a = sample_positive_truncnorm(
                self.a_mean,
                self.a_stdv,
                self.n_neurons,
                self.num_sims,
                seeds=neuron_seeds[1],
            )
            threshold = sample_positive_truncnorm(
                self.threshold_mean,
                self.threshold_stdv,
                self.n_neurons,
                self.num_sims,
                seeds=neuron_seeds[2],
            )
            fr_cap = sample_positive_truncnorm(
                self.fr_cap_mean,
                self.fr_cap_stdv,
                self.n_neurons,
                self.num_sims,
                seeds=neuron_seeds[3],
            )

            if "size" in self.W_table:
                a, threshold = set_sizes(self.W_table["size"], a, threshold)
            else:
                a, threshold = set_sizes(self.W_table["surf_area_um2"], a, threshold)

            # Generate input I
            input_currents = self._generate_input()

            # Convert to JAX arrays
            W_jax = jnp.array(self.W)
            t_axis_jax = jnp.array(self.t_axis)
            input_jax = jnp.array(input_currents)
            tau_jax = jnp.array(tau)
            threshold_jax = jnp.array(threshold)
            a_jax = jnp.array(a)
            fr_cap_jax = jnp.array(fr_cap)

            # Setup vmap axes based on W dimensionality
            # TODO: fix this
            if W_jax.ndim == 3:
                vmap_axes = (0, None, None, None, 0, 0, 0, 0, 0, None, None)
            else:
                vmap_axes = (None, None, None, None, 0, 0, 0, 0, 0, None, None)

            # Run parallel simulations (JAX)
            start_time = time.time()

            Rs = vmap(run, in_axes=vmap_axes)(
                W_jax,
                t_axis_jax,
                self.T,
                self.dt,
                input_jax,
                tau_jax,
                threshold_jax,
                a_jax,
                fr_cap_jax,
                self.exc_synapse_multiplier,
                self.inh_synapse_multiplier,
            )

            end_time = time.time()

            print(
                f"Time to solve system with vmap: {end_time - start_time:.3f} seconds"
            )

            # Convert results back to numpy
            result = np.array(Rs)

            results = {
                "result": result,
                "input_currents": input_currents,
                "tau": tau,
                "a": a,
                "threshold": threshold,
                "fr_cap": fr_cap,
                "t_axis": self.t_axis,
                "W": self.W,
                "W_table": self.W_table,
                "stim_neurons": self.stim_neurons,
                "stim_input": self.stim_input,
                "seeds": self.seeds,
                "n_neurons": self.n_neurons,
                "num_sims": self.num_sims,
                "config": self.cfg,
                "simulation_time": end_time - start_time,
            }

            return results

        except Exception as e:
            raise RuntimeError(f"Simulation failed: {e}")

    def __str__(self):
        return str(self.info)

    @property
    def info(self) -> dict:
        return {
            "experiment_name": self.cfg.experiment.name,
            "n_neurons": self.n_neurons,
            "num_sims": self.num_sims,
            "stim_neurons": self.stim_neurons.tolist(),
            "simulation_time": self.T,
            "time_step": self.dt,
            "pulse_duration": self.pulse_end - self.pulse_start,
            "neuron_params": {
                "tau_mean": self.tau_mean,
                "tau_stdv": self.tau_stdv,
                "a_mean": self.a_mean,
                "a_stdv": self.a_stdv,
                "threshold_mean": self.threshold_mean,
                "threshold_stdv": self.threshold_stdv,
                "fr_cap_mean": self.fr_cap_mean,
                "fr_cap_stdv": self.fr_cap_stdv,
                "exc_multiplier": self.exc_synapse_multiplier,
                "inh_multiplier": self.inh_synapse_multiplier,
            },
            "connectivity_shape": self.W.shape,
            "seeds": self.seeds.tolist(),
            "stim_input": self.stim_input.tolist(),
        }
