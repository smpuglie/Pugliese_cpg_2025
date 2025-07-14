import hydra
from omegaconf import DictConfig
from src.vnc import run_vnc_simulation
import sparse


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    results = run_vnc_simulation(cfg)
    sparse.save_npz(f"bdn2.npz", sparse.COO.from_numpy(results))


if __name__ == "__main__":
    main()
