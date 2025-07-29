import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import jax
# Configure JAX for better performance
jax.config.update("jax_enable_x64", False)  # Use float32 for better GPU performance
# jax.config.update("jax_platforms", "cuda")  # Prefer GPU

# Disable XLA optimizations that might cause timing issues
# os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=false'

print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)
print('Gpu devices:', len(jax.devices('gpu')) if jax.devices('gpu') else len(jax.devices('cpu')))
import hydra
import sparse
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.path_utils import convert_dict_to_path, save_config
from src.vnc_sim import run_vnc_simulation, save_state

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
    
    ##### Run the simulation #####
    print("Running VNC simulation with the following configuration:")
    # print(OmegaConf.to_yaml(cfg))
    results, state = run_vnc_simulation(cfg)
    ##### Save results #####
    print('Saving results to:', cfg.paths.ckpt_dir)
    sparse.save_npz(cfg.paths.ckpt_dir / f"{cfg.experiment.name}_Rs.npz", sparse.COO.from_numpy(results))
    if state is not None:
        save_state(state, cfg.paths.ckpt_dir / f"{cfg.experiment.name}_state.pkl")


if __name__ == "__main__":
    main()
