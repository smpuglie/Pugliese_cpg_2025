from pathlib import Path
from omegaconf import OmegaConf
import logging
import os
logger = logging.getLogger(__name__)


def register_custom_resolvers():
    """Register custom OmegaConf resolvers for interpolations."""
    
    def output_dir_resolver():
        """
        Custom resolver that returns hydra.runtime.output_dir if available,
        otherwise falls back to environment variables or default path.
        """
        # First try to get from Hydra's current working directory
        try:
            hydra_output_dir = os.environ.get('HYDRA_RUNTIME_OUTPUT_DIR')
            if hydra_output_dir:
                return hydra_output_dir
        except Exception as e:
            logger.debug(f"Could not get HYDRA_RUNTIME_OUTPUT_DIR: {e}")
        
        # Try to get from current working directory if it looks like a hydra output dir
        try:
            cwd = os.getcwd()
            if 'outputs' in cwd or 'multirun' in cwd:
                return cwd
        except Exception as e:
            logger.debug(f"Could not use current working directory: {e}")
        
        # If we can't determine the output dir, return None 
        # This will trigger the fallback in the YAML
        return None
    
    def fallback_path_resolver(base_dir: str, run_id: str = None):
        """
        Fallback resolver that constructs the path from base_dir and run_id.
        """
        if run_id is None:
            # Generate a simple timestamp-based run_id if not provided
            from datetime import datetime
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{base_dir}/run_id={run_id}/"
    
    # Register both resolvers
    OmegaConf.register_new_resolver(
        "hydra_output_dir", 
        output_dir_resolver,
        use_cache=False
    )
    
    OmegaConf.register_new_resolver(
        "fallback_path", 
        fallback_path_resolver,
        use_cache=False
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