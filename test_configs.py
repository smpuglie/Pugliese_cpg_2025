import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from src.utils.path_utils import save_config, convert_dict_to_path


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Create directories specified in the config if they do not exist
    cfg.paths = convert_dict_to_path(cfg.paths)
    # Set the current working directory to the cwd_dir specified in the config
    cfg.paths.cwd_dir = Path.cwd()
    cfg.experiment.stimNeurons
    print(OmegaConf.to_yaml(cfg, resolve=True))
    save_config(cfg, cfg.paths.log_dir / 'run_config.yaml')

if __name__ == "__main__":
    my_app()