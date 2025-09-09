from typing import NamedTuple, Optional, Tuple, List, Dict, Set, Any
from pathlib import Path
import jax.numpy as jnp


# #############################################################################################################################=
# IMMUTABLE CONFIGURATION STRUCTURES
# #############################################################################################################################=

class NeuronParams(NamedTuple):
    """Immutable neuron parameters for JIT compilation."""
    W: jnp.ndarray
    tau: jnp.ndarray  # Time constants for neurons
    a: jnp.ndarray  # Activation parameters
    threshold: jnp.ndarray  # Thresholds for activation
    fr_cap: jnp.ndarray  # Firing rate caps
    input_currents: jnp.ndarray  # Shape: (n_stim_configs, n_param_sets, n_neurons)
    seeds: jnp.ndarray  # Random seeds for reproducibility
    exc_dn_idxs: jnp.ndarray  # Indices of excitatory descending neurons
    inh_dn_idxs: jnp.ndarray  # Indices of inhibitory descending neurons
    exc_in_idxs: jnp.ndarray  # Indices of excitatory interneurons
    inh_in_idxs: jnp.ndarray  # Indices of inhibitory interneurons
    mn_idxs: jnp.ndarray  # Indices of motor neurons
    W_mask: Optional[jnp.ndarray] = None  # Mask for connectivity


class SimParams(NamedTuple):
    """Immutable simulation parameters for JIT compilation."""
    n_neurons: int
    n_param_sets: int
    n_stim_configs: int
    T: float
    dt: float
    pulse_start: float
    pulse_end: float
    t_axis: jnp.ndarray
    exc_multiplier: float
    inh_multiplier: float
    noise_stdv_prop: float
    r_tol: float
    a_tol: float
    noise_stdv: jnp.ndarray  # Standard deviation for noise, if applicable


class Pruning_state(NamedTuple):
    """Parameters for pruning simulation."""
    W_mask: jnp.ndarray
    interneuron_mask: jnp.ndarray
    level: jnp.ndarray
    total_removed_neurons: jnp.ndarray
    removed_stim_neurons: jnp.ndarray
    neurons_put_back: jnp.ndarray
    prev_put_back: jnp.ndarray
    last_removed: jnp.ndarray
    remove_p: jnp.ndarray
    min_circuit: jnp.ndarray
    keys: jnp.ndarray
    # New fields for tracking most recent good oscillating state
    last_good_oscillating_removed: jnp.ndarray  # Boolean mask of removed neurons when last good osc found
    last_good_oscillating_put_back: jnp.ndarray  # Boolean mask of put_back neurons when last good osc found  
    last_good_oscillation_score: jnp.ndarray     # The oscillation score value when last good osc found
    last_good_W_mask: jnp.ndarray                # The exact W_mask when last good oscillation found
    last_good_key: jnp.ndarray                   # The exact RNG key state when last good oscillation found
    last_good_stim_idx: jnp.ndarray              # The stimulation index that produced the good oscillation
    last_good_param_idx: jnp.ndarray             # The parameter index that produced the good oscillation


class SimulationConfig(NamedTuple):
    """Configuration for simulation execution."""
    sim_type: str  # "baseline", "shuffle", "noise", "prune"
    enable_pruning: bool = False
    oscillation_threshold: float = 0.5
    batch_size: Optional[int] = None
    max_pruning_iterations: int = 200
    # Auto stimulation adjustment parameters
    adjustStimI: bool = False
    max_adjustment_iters: int = 10
    n_active_upper: int = 500
    n_active_lower: int = 5
    n_high_fr_upper: int = 100
    high_fr_threshold: float = 100.0
    # Checkpointing parameters
    enable_checkpointing: bool = False
    checkpoint_interval: int = 10  # Save checkpoint every N batches or simulations
    
class CheckpointState(NamedTuple):
    """State information for simulation checkpointing - unified for batch and streaming modes."""
    batch_index: int
    completed_batches: int
    total_batches: int
    neuron_params: Optional[NeuronParams]  # Made optional for streaming mode
    n_result_batches: int  # Number of result batches (for reconstruction)
    accumulated_mini_circuits: Optional[List[jnp.ndarray]] = None  # For pruning simulations
    pruning_state: Optional[Pruning_state] = None
    # Streaming-specific fields (optional for batch mode)
    checkpoint_type: str = "batch"  # "batch" or "streaming"
    completed_simulations: Optional[Set[int]] = None  # For streaming: completed sim indices
    results_dict: Optional[Dict[int, jnp.ndarray]] = None  # For streaming: individual results
    states_dict: Optional[Dict[int, Pruning_state]] = None  # For streaming: individual states
    mini_circuits_dict: Optional[Dict[int, jnp.ndarray]] = None  # For streaming: individual mini_circuits
    w_masks_dict: Optional[Dict[int, jnp.ndarray]] = None  # For streaming: individual w_masks
    failed_sims: Optional[Set[int]] = None  # For streaming: failed simulation indices
    progress_info: Optional[Dict[str, Any]] = None  # For streaming: progress information
