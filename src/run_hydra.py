import hydra
from omegaconf import DictConfig
from src.vnc import VNCNet
import sparse


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
        
    mynet: VNCNet = VNCNet(cfg)

    result = mynet.run()

    print(result)

    sparse.save_npz(f'bdn2.npz', sparse.COO.from_numpy(result['result']))


if __name__ == "__main__":
    main()
