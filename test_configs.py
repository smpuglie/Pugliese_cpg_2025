import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Create directories specified in the config if they do not exist
    for k in cfg.paths.keys():
        if k != 'user':
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
            
    # Set the current working directory to the cwd_dir specified in the config
    cfg.paths.cwd_dir = Path.cwd()
    cfg.experiment.stimNeurons 
    print(OmegaConf.to_yaml(cfg,resolve=True))
    OmegaConf.save(cfg, cfg.paths.save_dir / 'run_config.yaml',resolve=True)

if __name__ == "__main__":
    my_app()