import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import hydra
from omegaconf import DictConfig
# from src.vnc import run_vnc_simulation
from src.optimized_vnc import run_vnc_simulation_optimized
import sparse
from pathlib import Path

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    for k in cfg.paths.keys():
        if (k != 'user'):
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)

    results = run_vnc_simulation_optimized(cfg)
    save_path = cfg.paths.ckpt_dir  / "bdn2.npz"
    print('Saving results to:', save_path)
    sparse.save_npz(save_path, sparse.COO.from_numpy(results))


if __name__ == "__main__":
    main()
