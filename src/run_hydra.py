import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import jax

# Configure JAX for better performance
jax.config.update("jax_enable_x64", False)  # Use float32 for better GPU performance

print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)
print('Gpu devices:', len(jax.devices('gpu')) if jax.devices('gpu') else len(jax.devices('cpu')))

import hydra
import sparse
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import src.io_dict_to_hdf5 as ioh5
from src.path_utils import convert_dict_to_path, save_config
from src.vnc_sim import run_vnc_simulation

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function to run the VNC simulation with the provided configuration."""
    # Ensure paths are set correctly
    cfg.paths = convert_dict_to_path(cfg.paths)

    ##### Save config file to the log directory #####
    config_path = cfg.paths.log_dir / "run_config.yaml"
    if not config_path.exists():
        save_config(cfg, config_path)
        print(f"Config saved to {config_path}")
    cfg.paths = convert_dict_to_path(cfg.paths)

    ##### Run the simulation #####
    print("Running VNC simulation with the following configuration:")
    results, final_mini_circuits, neuron_params = run_vnc_simulation(cfg)
    
    ##### Save results #####
    print('Saving results to:', cfg.paths.ckpt_dir)
    sparse.save_npz(cfg.paths.ckpt_dir / f"{cfg.experiment.name}_Rs.npz", sparse.COO.from_numpy(results))
    ioh5.save(cfg.paths.ckpt_dir / 'neuron_params.h5', neuron_params._asdict())
    if final_mini_circuits is not None:
        sparse.save_npz(cfg.paths.ckpt_dir / f"{cfg.experiment.name}_mini_circuits.npz", sparse.COO.from_numpy(final_mini_circuits))


if __name__ == "__main__":
    main()
