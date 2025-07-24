import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import jax
print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)
print('Gpu devices:', len(jax.devices('gpu')) if jax.devices('gpu') else len(jax.devices('cpu')))
import hydra
import sparse
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
# from src.vnc import run_vnc_simulation
from src.optimized_vnc import run_vnc_simulation_optimized
from src.prune_net import run_vnc_prune_optimized, save_state
import gc

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function to run the VNC simulation with the provided configuration."""
    # Ensure paths are set correctly
    for k in cfg.paths.keys():
        if (k != 'user'):
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)

    ##### Save config file to the log directory #####
    config_path = cfg.paths.log_dir / "run_config.yaml"
    if not config_path.exists():
        OmegaConf.save(cfg, config_path)
        print(f"Config saved to {config_path}")    
    
    ##### Run the simulation #####
    print("Running VNC simulation with the following configuration:")
    # print(OmegaConf.to_yaml(cfg))
    if cfg.experiment.name == 'Prune_Test':
        results, state = run_vnc_prune_optimized(cfg)
    else:
        results = run_vnc_simulation_optimized(cfg)

    gc.collect()
    ##### Save results #####
    print('Saving results to:', cfg.paths.ckpt_dir)
    sparse.save_npz(cfg.paths.ckpt_dir / f"{cfg.experiment.name}_Rs.npz", sparse.COO.from_numpy(results))
    # save_state(state, cfg.paths.ckpt_dir / f"{cfg.experiment.name}_state.pkl")

if __name__ == "__main__":
    main()
