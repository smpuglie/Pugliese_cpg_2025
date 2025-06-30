from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import yaml
import sys

# Property mapping
CONFIG_MAPPING = {
    "neuron_params": [
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
    ],
    "sim": [
        "name",
        "T",
        "dt",
        "pulseStart",
        "pulseEnd",
        "adjustStimI",
        "maxIters",
        "nActiveUpper",
        "nActiveLower",
        "nHighFrUpper",
    ],
    "experiment": [
        "name",
        "wPath",
        "dfPath",
        "paramsToIterate",
        "stimNeurons",
        "removeNeurons",
        "stimI",
        "seed",
        "saveFigs",
        "useTsp",
    ],
}


def migrate_config_section(
    old_config: dict, cfg: DictConfig, section: str, props: list
):
    """Copy properties from old_config to cfg.section"""

    section_cfg = getattr(cfg, section)
    for prop in props:
        if prop in old_config:
            section_cfg[prop] = old_config[prop]


def migrate_config(old_config_path: str, base_config_path: str):
    # Load base config
    with open(base_config_path, "r") as f:
        cfg = OmegaConf.load(f)

    # Load old config
    with open(old_config_path, "r") as f:
        old_config = yaml.safe_load(f)

    # Migrate sections
    for section, props in CONFIG_MAPPING.items():
        if hasattr(cfg, section):
            migrate_config_section(old_config, cfg, section, props)

    # Save
    old_file = Path(old_config_path)
    output_path = old_file.parent / (old_file.stem + "_converted.yaml")
    OmegaConf.save(cfg, output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <old_config.yaml>")
        sys.exit(1)

    migrate_config(sys.argv[1], "hydra.yaml")
