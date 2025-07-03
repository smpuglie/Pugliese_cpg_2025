import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax import vmap
import jax
from omegaconf import DictConfig
from scipy.stats import truncnorm

from src.cfg_sim_utils import load_W, load_wTable


def sample_positive_truncnorm(mean, stdev, nSamples, nSims, seeds=None):
    """For sampling the neuron parameters from distribution."""
    
    # TODO: fix logic
    
    a = (
        -mean / stdev
    )  # number of standard deviations to truncate on the left side -- needs to truncate at 0, so that is mean/stdev standard deviations away
    b = 100  # arbitrarily high number of standard deviations, basically we want infinity but can't pass that in I don't think
    samples = np.zeros([nSims, nSamples])
    for sim in range(nSims):

        current_seed = seeds[sim] if seeds is not None else None

        samples[sim] = truncnorm.rvs(
            a,
            b,
            loc=mean,
            scale=stdev,
            size=nSamples,
            random_state=np.random.RandomState(seed=current_seed),
        )

    return samples




def arrayInterpJax(x, xp, fp):
    return vmap(lambda f: jnp.interp(x, xp, f))(fp)

def rate_equation_half_tanh(t, R, args):

    timePoints, inputs, tau, weightedW, threshold, a, frCap = args

    I = arrayInterpJax(t, timePoints, inputs)  # interpolate to get input values
    totalInput = I + jnp.dot(weightedW, R)
    activation = jnp.maximum(
        frCap * jnp.tanh((a / frCap) * (totalInput - threshold)), 0
    )

    return (activation - R) / tau

def set_sizes(sizes, a, threshold):
    """Should correspond to surface area. This method is going to need a lot of improvement, very much testing out"""

    normSize = np.nanmedian(sizes)  # np.nanmean(sizes)
    sizes[sizes.isna()] = normSize
    sizes[sizes == 0] = normSize  # TODO this isn't great
    sizes = sizes / normSize
    sizes = np.asarray(sizes)

    a = a / sizes
    threshold = threshold * sizes
    return a, threshold

def simulate(
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
            'experiment': ['name', 'wPath', 'dfPath', 'stimNeurons', 'stimI', 'seed'],
            'sim': ['T', 'dt', 'pulseStart', 'pulseEnd'],
            'neuron_params': ['tauMean', 'tauStdv', 'aMean', 'aStdv', 'thresholdMean', 
                             'thresholdStdv', 'frcapMean', 'frcapStdv', 'excitatoryMultiplier', 
                             'inhibitoryMultiplier']
        }
        
        for section, keys in required_sections.items():
            if not hasattr(self.cfg, section):
                raise ValueError(f"Missing required config section: {section}")
            section_cfg = getattr(self.cfg, section)
            for key in keys:
                if not hasattr(section_cfg, key):
                    raise ValueError(f"Missing required {section} config: {key}")
                    
            
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

        print(self.stim_neurons)
        print(self.stim_input.ndim)

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
        self.adjust_stim_I = getattr(self.cfg.sim, 'adjustStimI', False)
        self.max_iters = getattr(self.cfg.sim, 'maxIters', 10)
        self.n_active_upper = getattr(self.cfg.sim, 'nActiveUpper', 500)
        self.n_active_lower = getattr(self.cfg.sim, 'nActiveLower', 5)
        self.n_high_fr_upper = getattr(self.cfg.sim, 'nHighFrUpper', 100)
        
        # Remove neurons if specified
        self.remove_neurons = getattr(self.cfg.experiment, 'removeNeurons', [])
        
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
        
        
    def _generate_input(self) -> np.ndarray:
        """Generate input matrix for all simulations."""
        input_matrix = np.zeros((self.num_sims, self.n_neurons, len(self.t_axis)))
        
        for sim_idx in range(self.num_sims):
            stim_current = self.stim_input[sim_idx]
            
            # Find time indices for pulse
            pulse_start_idx = np.searchsorted(self.t_axis, self.pulse_start)
            pulse_end_idx = np.searchsorted(self.t_axis, self.pulse_end)
            
            # Apply stim to specified neurons
            for neuron_idx in self.stim_neurons:
                input_matrix[sim_idx, neuron_idx, pulse_start_idx:pulse_end_idx] = stim_current
                
        return input_matrix
        
    def run(self) -> dict:
        """Run the Simulation"""
        try:
            
            # Sample neuron parameters
            # TODO: fix this logic:
            neuron_seeds = np.zeros([4, self.num_sims], dtype=int)
            for sim in range(self.num_sims):
                neuron_seeds[:, sim] = np.random.default_rng(self.seeds[sim]).integers(
                    10000, size=4
                )
            
            tau = sample_positive_truncnorm(self.tau_mean, self.tau_stdv, self.n_neurons, self.num_sims, seeds=neuron_seeds[0])
            a = sample_positive_truncnorm(self.a_mean, self.a_stdv, self.n_neurons, self.num_sims, seeds=neuron_seeds[1])
            threshold = sample_positive_truncnorm(self.threshold_mean, self.threshold_stdv, self.n_neurons, self.num_sims, seeds=neuron_seeds[2])
            fr_cap = sample_positive_truncnorm(self.fr_cap_mean, self.fr_cap_stdv, self.n_neurons, self.num_sims, seeds=neuron_seeds[3])
            
            a, threshold = set_sizes(self.W_table["size"], a, threshold)

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
            
            Rs = vmap(simulate, in_axes=vmap_axes)(
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
            
            print(f"Time to solve system with vmap: {end_time - start_time:.3f} seconds")
            
            # Convert results back to numpy
            result = np.array(Rs)
            print(f"Result shape: {result.shape}")
            
            results = {
                'result': result,
                'input_currents': input_currents,
                'tau': tau,
                'a': a,
                'threshold': threshold,
                'fr_cap': fr_cap,
                't_axis': self.t_axis,
                'W': self.W,
                'W_table': self.W_table,
                'stim_neurons': self.stim_neurons,
                'stim_input': self.stim_input,
                'seeds': self.seeds,
                'n_neurons': self.n_neurons,
                'num_sims': self.num_sims,
                'config': self.cfg,
                'simulation_time': end_time - start_time
            }
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {e}")
        
    def __str__(self):
        return str(self.info)
            
    @property
    def info(self) -> dict:
        return {
            'experiment_name': self.cfg.experiment.name,
            'n_neurons': self.n_neurons,
            'num_sims': self.num_sims,
            'stim_neurons': self.stim_neurons.tolist(),
            'simulation_time': self.T,
            'time_step': self.dt,
            'pulse_duration': self.pulse_end - self.pulse_start,
            'neuron_params': {
                'tau_mean': self.tau_mean,
                'tau_stdv': self.tau_stdv,
                'a_mean': self.a_mean,
                'a_stdv': self.a_stdv,
                'threshold_mean': self.threshold_mean,
                'threshold_stdv': self.threshold_stdv,
                'fr_cap_mean': self.fr_cap_mean,
                'fr_cap_stdv': self.fr_cap_stdv,
                'exc_multiplier': self.exc_synapse_multiplier,
                'inh_multiplier': self.inh_synapse_multiplier
            },
            'connectivity_shape': self.W.shape,
            'seeds': self.seeds.tolist(),
            'stim_input': self.stim_input.tolist()
        }