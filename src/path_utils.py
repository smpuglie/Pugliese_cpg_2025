from pathlib import Path
from omegaconf import OmegaConf

def convert_to_string(value):
    """Convert a value to a string, handling Path objects."""
    if isinstance(value, Path):
        return str(value)
    return value

def convert_to_path(value):
    """Convert a value to a Path object, handling strings."""
    if isinstance(value, str):
        return Path(value)
    return value

def convert_dict_to_string(d):
    """Convert all values in a dictionary to strings."""
    return {k: convert_to_string(v) for k, v in d.items()}

def convert_dict_to_path(d):
    """Convert all values in a dictionary to Path objects."""
    for k in d.keys():
        if k != 'user':
            d[k] = convert_to_path(d[k])
            d[k].mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg, path):
    """Save the configuration to a file."""
    cfg.paths = convert_dict_to_string(cfg.paths)
    OmegaConf.save(cfg, path)