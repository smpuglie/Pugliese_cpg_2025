import time
from typing import NamedTuple, Dict, Any
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax import vmap, jit, random
from omegaconf import DictConfig
from jax.lax import cond
from src.shuffle_utils import extract_shuffle_indices, full_shuffle
from src.Archive.sim_utils_old import (
    load_W,
    load_wTable,
    sample_positive_truncnorm,
    set_sizes,
    make_input,
)


class NeuronConfig(NamedTuple):

    W: jnp.ndarray
    W_table: Any
    tau: jnp.ndarray
    a: jnp.ndarray
    threshold: jnp.ndarray
    fr_cap: jnp.ndarray
    input_currents: jnp.ndarray
    seeds: jnp.ndarray
    stim_neurons: jnp.ndarray
    exc_dn_idxs: jnp.ndarray
    inh_dn_idxs: jnp.ndarray
    exc_in_idxs: jnp.ndarray
    inh_in_idxs: jnp.ndarray
    mn_idxs: jnp.ndarray


class SimConfig(NamedTuple):

    n_neurons: int
    num_sims: int
    T: float
    dt: float
    pulse_start: float
    pulse_end: float
    t_axis: jnp.ndarray
    stim_input: float
    shuffle: bool
    noise: bool
    noise_stdv_prop: float
    exc_synapse_multiplier: float
    inh_synapse_multiplier: float
    r_tol: float
    a_tol: float


def rate_equation_half_tanh(t, R, args):
    inputs, pulseStart, pulseEnd, tau, weightedW, threshold, a, frCap = args

    pulse_active = (t >= pulseStart) & (t <= pulseEnd)
    I = inputs * pulse_active

    totalInput = I + jnp.dot(weightedW, R)
    activation = jnp.maximum(
        frCap * jnp.tanh((a / frCap) * (totalInput - threshold)), 0
    )

    return (activation - R) / tau


def run(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Pure jax function that runs the actual simulation"""
    term = ODETerm(rate_equation_half_tanh)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.array(tAxis))
    controller = PIDController(rtol=r_tol, atol=a_tol)

    odeSolution = diffeqsolve(
        term,
        solver,
        0,
        T,
        dt,
        R0,
        args=(inputs, pulseStart, pulseEnd, tau, W, threshold, a, frCap),
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=5000000,
    )

    return jnp.transpose(odeSolution.ys)


def run_shuffle(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    seed,
    extra,
    excDnIdxs,
    inhDnIdxs,
    excInIdxs,
    inhInIdxs,
    mnIdxs,
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Pure jax function that shuffles the weight matrix"""
    W_shuffled = full_shuffle(
        W, seed, excDnIdxs, inhDnIdxs, excInIdxs, inhInIdxs, mnIdxs
    )

    return run(
        W_shuffled,
        R0,
        tAxis,
        T,
        dt,
        inputs,
        pulseStart,
        pulseEnd,
        tau,
        threshold,
        a,
        frCap,
        r_tol,
        a_tol,
    )


def run_noise(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    seed,
    stdvProp,
    excDnIdxs,  # Added to match signature (ignored)
    inhDnIdxs,  # Added to match signature (ignored)
    excInIdxs,  # Added to match signature (ignored)
    inhInIdxs,  # Added to match signature (ignored)
    mnIdxs,  # Added to match signature (ignored)
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Pure jax function that adds noise to the weight matrix"""
    W_noisy = resample_W(W, seed, stdvProp)

    return run(
        W_noisy,
        R0,
        tAxis,
        T,
        dt,
        inputs,
        pulseStart,
        pulseEnd,
        tau,
        threshold,
        a,
        frCap,
        r_tol,
        a_tol,
    )


def run_no_shuffle(
    W,
    R0,
    tAxis,
    T,
    dt,
    inputs,
    pulseStart,
    pulseEnd,
    tau,
    threshold,
    a,
    frCap,
    seed,  # Added to match signature (ignored)
    stdvProp,  # Added to match signature (ignored)
    excDnIdxs,  # Added to match signature (ignored)
    inhDnIdxs,  # Added to match signature (ignored)
    excInIdxs,  # Added to match signature (ignored)
    inhInIdxs,  # Added to match signature (ignored)
    mnIdxs,  # Added to match signature (ignored)
    r_tol=1e-7,
    a_tol=1e-9,
):
    """Wrapper for run that ignores shuffle and noise parameters."""
    return run(
        W,
        R0,
        tAxis,
        T,
        dt,
        inputs,
        pulseStart,
        pulseEnd,
        tau,
        threshold,
        a,
        frCap,
        r_tol,
        a_tol,
    )


def validate_config(cfg: DictConfig) -> None:
    """Check if config has everything."""
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
        if not hasattr(cfg, section):
            raise ValueError(f"Missing required config section: {section}")
        section_cfg = getattr(cfg, section)
        for key in keys:
            if not hasattr(section_cfg, key):
                raise ValueError(
                    f"Missing required config key '{key}' in section '{section}'"
                )


def load_connectivity(cfg: DictConfig) -> tuple[jnp.ndarray, Any]:
    """Load connectivity matrix and the data."""
    w_path = cfg.experiment.wPath
    df_path = cfg.experiment.dfPath
    W = load_W(w_path)
    W_table = load_wTable(df_path)
    return jnp.array(W), W_table


def sample_neuron_parameters(
    cfg: DictConfig, n_neurons: int, num_sims: int, seeds: np.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample neuron parameters based on configuration."""
    np_cfg = cfg.neuron_params

    neuron_seeds = np.zeros([4, num_sims], dtype=int)
    for sim in range(num_sims):
        neuron_seeds[:, sim] = np.random.default_rng(int(seeds[sim])).integers(
            10000, size=4
        )

    tau = sample_positive_truncnorm(
        float(np_cfg.tauMean),
        float(np_cfg.tauStdv),
        n_neurons,
        num_sims,
        seeds=neuron_seeds[0],
    )
    a = sample_positive_truncnorm(
        float(np_cfg.aMean),
        float(np_cfg.aStdv),
        n_neurons,
        num_sims,
        seeds=neuron_seeds[1],
    )
    threshold = sample_positive_truncnorm(
        float(np_cfg.thresholdMean),
        float(np_cfg.thresholdStdv),
        n_neurons,
        num_sims,
        seeds=neuron_seeds[2],
    )
    fr_cap = sample_positive_truncnorm(
        float(np_cfg.frcapMean),
        float(np_cfg.frcapStdv),
        n_neurons,
        num_sims,
        seeds=neuron_seeds[3],
    )

    return jnp.array(tau), jnp.array(a), jnp.array(threshold), jnp.array(fr_cap)


def create_sim_config(cfg: DictConfig, W: jnp.ndarray) -> SimConfig:
    """Create simulation configuration from config and connectivity matrix."""
    n_neurons = W.shape[0]
    seeds = np.array(cfg.experiment.seed)
    num_sims = len(seeds)

    T = float(cfg.sim.T)
    dt = float(cfg.sim.dt)
    pulse_start = float(cfg.sim.pulseStart)
    pulse_end = float(cfg.sim.pulseEnd)
    t_axis = jnp.array(np.arange(0, T, dt))

    stim_input = cfg.experiment.stimI
    shuffle = getattr(cfg.sim, "shuffle", False)
    noise = getattr(cfg.sim, "noise", False)
    noise_stdv_prop = getattr(cfg.sim, "noiseStdvProp", 0.1)

    exc_synapse_multiplier = float(cfg.neuron_params.excitatoryMultiplier)
    inh_synapse_multiplier = float(cfg.neuron_params.inhibitoryMultiplier)

    r_tol = getattr(cfg.sim, "rtol", 1e-7)
    a_tol = getattr(cfg.sim, "atol", 1e-9)

    return SimConfig(
        n_neurons=n_neurons,
        num_sims=num_sims,
        T=T,
        dt=dt,
        pulse_start=pulse_start,
        pulse_end=pulse_end,
        t_axis=t_axis,
        stim_input=stim_input,
        shuffle=shuffle,
        noise=noise,
        noise_stdv_prop=noise_stdv_prop,
        exc_synapse_multiplier=exc_synapse_multiplier,
        inh_synapse_multiplier=inh_synapse_multiplier,
        r_tol=r_tol,
        a_tol=a_tol,
    )


def load_vnc_net(cfg: DictConfig) -> tuple[NeuronConfig, SimConfig]:
    """Load and configure VNC network parameters and simulation settings."""
    validate_config(cfg)

    W, W_table = load_connectivity(cfg)
    sim_config = create_sim_config(cfg, W)

    stim_neurons = jnp.array(cfg.experiment.stimNeurons, dtype=jnp.int64)
    seeds = np.array(cfg.experiment.seed)

    tau, a, threshold, fr_cap = sample_neuron_parameters(
        cfg, sim_config.n_neurons, sim_config.num_sims, seeds
    )

    if "size" in W_table:
        a, threshold = set_sizes(W_table["size"], a, threshold)
    else:
        a, threshold = set_sizes(W_table["surf_area_um2"], a, threshold)

    input_currents = jnp.array(
        make_input(sim_config.n_neurons, stim_neurons, sim_config.stim_input)
    )

    exc_dn_idxs, inh_dn_idxs, exc_in_idxs, inh_in_idxs, mn_idxs = (
        extract_shuffle_indices(W_table)
    )

    params = NeuronConfig(
        W=W,
        W_table=W_table,
        tau=tau,
        a=a,
        threshold=threshold,
        fr_cap=fr_cap,
        input_currents=input_currents,
        seeds=jnp.array(seeds),
        stim_neurons=stim_neurons,
        exc_dn_idxs=exc_dn_idxs,
        inh_dn_idxs=inh_dn_idxs,
        exc_in_idxs=exc_in_idxs,
        inh_in_idxs=inh_in_idxs,
        mn_idxs=mn_idxs,
    )

    return params, sim_config


def resample_W(W, seed, stdvProp):
    """Add noise to weight matrix."""
    stdvs = W * stdvProp
    key = random.key(seed)
    samples = random.truncated_normal(
        key, lower=-1.0 / stdvProp, upper=float("inf"), shape=W.shape
    )
    return W + stdvs * samples


@jit
def reweight_connectivity(
    W: jnp.ndarray, exc_multiplier: float, inh_multiplier: float
) -> jnp.ndarray:
    """Reweight connectivity matrix with excitatory and inhibitory multipliers."""
    Wt_exc = jnp.maximum(W, 0)
    Wt_inh = jnp.minimum(W, 0)
    W_rw = jnp.transpose(exc_multiplier * Wt_exc + inh_multiplier * Wt_inh)
    return W_rw


def simulate_vnc_net(params: NeuronConfig, config: SimConfig) -> np.ndarray:
    """Simulate the VNC network with given parameters and configuration."""
    try:
        start_time = time.time()

        W_rw = reweight_connectivity(
            params.W, config.exc_synapse_multiplier, config.inh_synapse_multiplier
        )

        R0 = jnp.zeros((W_rw.shape[1],))

        # Determine which run function to use based on configuration
        if config.shuffle:
            run_func = run_shuffle
            print("Running simulation with shuffle")
        elif config.noise:
            run_func = run_noise
            print("Running simulation with noise")
        else:
            run_func = run_no_shuffle
            print("Running simulation without shuffle or noise")

        vmap_axes = (
            None,  # W
            None,  # R0
            None,  # tAxis
            None,  # T
            None,  # dt
            None,  # inputs
            None,  # pulseStart
            None,  # pulseEnd
            0,  # tau
            0,  # threshold
            0,  # a
            0,  # frCap
            0,  # seed
            None,  # stdvProp or shuffle indices (depends on function)
            None,  # excDnIdxs
            None,  # inhDnIdxs
            None,  # excInIdxs
            None,  # inhInIdxs
            None,  # mnIdxs
        )

        # Prepare additional parameter based on simulation type
        if config.noise:
            additional_param = config.noise_stdv_prop
        else:
            additional_param = None

        Rs = jit(vmap(run_func, in_axes=vmap_axes))(
            W_rw,
            R0,
            config.t_axis,
            config.T,
            config.dt,
            params.input_currents,
            config.pulse_start,
            config.pulse_end,
            params.tau,
            params.threshold,
            params.a,
            params.fr_cap,
            params.seeds,
            additional_param,
            params.exc_dn_idxs,
            params.inh_dn_idxs,
            params.exc_in_idxs,
            params.inh_in_idxs,
            params.mn_idxs,
        )

        end_time = time.time()

        print(f"Time to solve system with vmap: {end_time - start_time:.3f} seconds")

        return np.array(Rs)

    except Exception as e:
        raise RuntimeError(f"Simulation failed: {e}")


def run_vnc_simulation(cfg: DictConfig) -> np.ndarray:
    """Run a complete VNC simulation with the given configuration."""
    params, config = load_vnc_net(cfg)
    results = simulate_vnc_net(params, config)
    return results
