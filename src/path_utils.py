from pathlib import Path
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)

def register_custom_resolvers():
    """Register custom OmegaConf resolvers for interpolations."""
    
    def output_dir_resolver(cfg):
        """
        Custom resolver that returns hydra.runtime.output_dir if available,
        otherwise falls back to ${paths.base_dir}/run_id=${run_id}/
        """
        try:
            # Try to get hydra runtime output dir
            if hasattr(cfg, 'hydra') and hasattr(cfg.hydra, 'runtime') and hasattr(cfg.hydra.runtime, 'output_dir'):
                return cfg.hydra.runtime.output_dir
        except Exception as e:
            logger.debug(f"Could not access hydra.runtime.output_dir: {e}")
        
        # Fallback to custom path construction
        try:
            base_dir = OmegaConf.select(cfg, "paths.base_dir")
            run_id = OmegaConf.select(cfg, "run_id")
            
            if base_dir is not None and run_id is not None:
                return f"{base_dir}/run_id={run_id}/"
            else:
                logger.warning("Could not construct fallback path - missing base_dir or run_id")
                return None
        except Exception as e:
            logger.error(f"Error constructing fallback path: {e}")
            return None
    
    # Register the resolver
    OmegaConf.register_new_resolver(
        "output_dir_or_fallback", 
        output_dir_resolver,
        use_cache=False  # Don't cache since hydra.runtime.output_dir might change
    )

# Auto-register when module is imported
register_custom_resolvers()

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