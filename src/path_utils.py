from pathlib import Path
from omegaconf import OmegaConf
import logging
import os
logger = logging.getLogger(__name__)


def register_custom_resolvers():
    """Register custom OmegaConf resolvers for interpolations."""

    
    def multirun_aware_save_dir_resolver(base_dir: str, run_id: str):
        """
        Custom resolver that checks if it's a multirun.
        If multirun: use hydra.runtime.output_dir (which will be the subdir in sweep)
        If not multirun: use ${base_dir}/run_id=${run_id}/
        
        Args:
            base_dir: The base directory path
            run_id: The run ID
        """
        try:
            # Check if we're in a multirun context
            from hydra.core.hydra_config import HydraConfig
            
            # Try to get the current Hydra config
            if HydraConfig.initialized():
                hydra_cfg = HydraConfig.get()
                # Check if it's a multirun by looking at the job config
                is_multirun = hydra_cfg.mode.name == "MULTIRUN"
                
                if is_multirun:
                    # For multirun, use the current output directory (which is the subdir)
                    # This should be something like .../run_id=Testing/sim.noiseStdvProp=0.0/
                    hydra_output_dir = hydra_cfg.runtime.output_dir
                    if hydra_output_dir:
                        return str(hydra_output_dir)
                    
                    # Fallback: construct the path manually using override_dirname
                    override_dirname = hydra_cfg.job.override_dirname
                    if override_dirname:
                        return f"{base_dir}/run_id={run_id}/{override_dirname}"
                    else:
                        return f"{base_dir}/run_id={run_id}"
                else:
                    # For single run, use the standard path
                    return f"{base_dir}/run_id={run_id}"
            else:
                # If Hydra not initialized, assume single run
                return f"{base_dir}/run_id={run_id}"
                
        except Exception as e:
            logger.debug(f"Could not determine if multirun: {e}")
            # If we can't determine, use the standard path
            return f"{base_dir}/run_id={run_id}"
    
    # Register all resolvers (with replace=True to avoid conflicts if already registered)
    OmegaConf.register_new_resolver(
        "multirun_save_dir",
        multirun_aware_save_dir_resolver,
        use_cache=False,
        replace=True
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
    # Create a copy of the config to avoid modifying the original
    cfg_copy = OmegaConf.create(cfg)
    
    # Resolve all interpolations in the paths section before converting to strings
    if 'paths' in cfg_copy:
        # Resolve interpolations first
        OmegaConf.resolve(cfg_copy.paths)
        # Then convert to strings
        cfg_copy.paths = convert_dict_to_string(cfg_copy.paths)
    
    OmegaConf.save(cfg_copy, path)