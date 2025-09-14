# VNC Network Simulation

A JAX/JIT implementation for VNC (Ventral Nerve Cord) Connectome network simulation accompanying the preprint ["Connectome simulations identify a central pattern generator circuit for fly walking"](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v1) (Pugliese et al. 2025, biorXiv). Simulation data is available on [Google Drive](https://drive.google.com/drive/folders/1Dgy5W8VZsayL8iVCRC3yfBJIpfxNJkO0?usp=sharing).  We have included Jupyter notebooks that simply load this data to reproduce the analyses, as well as the codebase to run simulations for yourself (see below). This repository is currently a work in progress, please check back for updates!! 🪰

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda or Miniconda

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/smpuglie/Pugliese_2025.git
   cd Pugliese_2025
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate vnc_cpg
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

### GPU Support (Recommended)

If you need CUDA support, replace the CPU-only JAX installation:
```bash
pip uninstall jax jaxlib
pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


## Usage

The package provides simulation tools for neural network dynamics with the following key components:

- **VNC Simulation**: Main simulation engine with JAX/JIT optimization
- **Memory Management**: Adaptive memory management for large-scale simulations  
- **Checkpointing**: Save and resume simulation states
- **Utilities**: Data processing and visualization tools

## Project Structure

```
src/
├── data/           # Data classes and structures
├── memory/         # Memory management utilities
├── simulation/     # Core simulation engines
├── utils/          # Utility functions and helpers
└── run_hydra.py    # Main Hydra-based runner
configs/            # Hydra configuration files
```

## Configuration with Hydra

This project uses [Hydra](https://hydra.cc/) for configuration management, allowing flexible experiment setup and parameter sweeps.

### Configuration Structure

The configuration system is organized hierarchically in the `configs/` directory:

The file system is organized based on the path.yaml file. Edit or create your own yaml following the given structure to run simulations.

```
configs/
├── config.yaml              # Main configuration file
├── experiment/              # Experiment-specific configurations
│   ├── DNg100_Stim.yaml    # DN stimulation experiment
│   ├── DN_Screen.yaml      # DN screening experiment
│   └── ...
├── sim/                     # Simulation parameters
│   ├── default.yaml        # Default simulation settings
│   ├── Prune_Network.yaml  # Network pruning configuration
│   └── ...
├── neuron_params/          # Neuron model parameters
│   └── default.yaml        # Default neuron parameters
└── paths/                  # System-specific paths
    ├── glados.yaml        # Local machine paths
    └── hyak.yaml         # HPC cluster paths
```

### Running Simulations

Use the Hydra-based runner to execute simulations:

```bash
# Basic run with default configuration
python src/run_hydra.py

# Override specific parameters
python src/run_hydra.py experiment=DNg100_Stim  experiment.n_replicates=128 exerpiment.batch_size=16

# Run parameter sweeps
python src/run_hydra.py -m experiment=DNg100_Stim sim.noise=true,false

# Use different neuron parameters
python src/run_hydra.py neuron_params=custom_params
```

### Key Configuration Components

- **experiment**: Defines the experimental setup (stimulated neurons, connectivity data, replicates)
- **sim**: Simulation parameters (time, noise, pruning settings)
- **neuron_params**: Neuron model parameters (time constants, thresholds, firing rates)
- **paths**: File system paths for data and outputs

### Output Organization

Hydra organizes outputs hierarchically by experiment and system:
```
/data/users/username/Pugliese_2025/
├── DNg100_Stim/               # Experiment type
│   └── version/               # Version
│       └── run_id=YOUR_JOBID/ # Job/run ID
│           ├── .hydra/        # Hydra configuration files
│           ├── figures/       # Generated plots and visualizations
│           ├── ckpt/          # Checkpoint files
│           ├── logs/          # Log files
│           └── run_hydra.log  # Main execution log
├── DN_Screen/                 # Different experiment
│   └── hyak/
└── DNb08_Stim/             # Another experiment type
    └── hyak/
```

## Running on HPC Clusters with SLURM

For large-scale simulations on HPC clusters, use the `slurm_run.py` script to submit jobs to SLURM workload managers. The current configurations are set up for [Hyak](https://hyak.uw.edu/docs) at the University of Washington.

### Basic SLURM Submission

```bash
# Submit with default settings (8 L40S GPUs, 2 days)
python slurm_run.py --experiment DNg100_Stim --sim default

# Submit with custom resources
python slurm_run.py --gpus 2 --mem 256 --cpus 32 --time "12:00:00" --experiment DNb08_Stim
```

### Supported GPU Types

- `l40s` - NVIDIA L40S (default)
- `a100` - NVIDIA A100 
- `h100` - NVIDIA H100
- `a40` - NVIDIA A40
- `l40` - NVIDIA L40

### Common SLURM Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gpus` | 8 | Number of GPUs to request |
| `--gpu_type` | l40s | Type of GPU (l40s, a100, h100, a40, l40) |
| `--mem` | 512 | Memory in GB |
| `--cpus` | 64 | Number of CPU cores |
| `--time` | 2-00:00:00 | Time limit (days-hours:minutes:seconds) |
| `--partition` | gpu-l40s | SLURM partition |
| `--job_name` | vnc_cpg | Job name |
| `--experiment` | stim_neurons | Experiment configuration |
| `--sim` | default | Simulation configuration |

### Advanced Usage

```bash
# Resume from checkpoint
python slurm_run.py --load_jobid 12345678 --experiment DNg100_Stim

# Override configuration parameters
python slurm_run.py --experiment DNg100_Stim --override "experiment.n_replicates=2048 sim.T=5.0"

# Run parameter sweep
python slurm_run.py --mode MULTIRUN --experiment DNg100_Stim --override "sim.noise=true,false"
```

### Job Management

```bash
# Check job status
squeue -u $USER

# Cancel all your jobs
squeue -u $USER -h | awk '{print $1}' | xargs scancel

# Cancel specific job
scancel JOB_ID

# View job details
scontrol show job JOB_ID
```

### Output Files

SLURM jobs create output files in:
- `./OutFiles/slurm-JOBID_ARRAYID.out` - SLURM output logs
- Simulation results follow the Hydra output organization (see above)

