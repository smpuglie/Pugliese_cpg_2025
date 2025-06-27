import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import yaml


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.save(cfg, "hydra.yaml")

    old_config_path = "configs_old/20250624_mancBDN2activation.yaml"
    with open(old_config_path, "r") as f:
        old_config = yaml.safe_load(f)

    neuron_props = [
        "name",
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
    ]

    for key in neuron_props:
        if key in old_config:
            cfg.neuron_params[key] = old_config[key]

    OmegaConf.save(cfg, "old.yaml")

    print(cfg.neuron_params)


if __name__ == "__main__":
    my_app()
