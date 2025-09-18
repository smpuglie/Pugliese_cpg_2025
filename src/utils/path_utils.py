from pathlib import Path
from omegaconf import OmegaConf
import logging
import os
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig

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
    

def smart_path_replacement(original_paths, target_template_paths, preserve_relative=True, verbose=True):
    """
    Intelligently replace paths by mapping base directories while preserving relative structure.
    
    Args:
        original_paths: Dict of original paths from loaded config
        target_template_paths: Dict of target paths from template
        preserve_relative: Whether to preserve relative path structure
        verbose: Whether to print detailed information about path analysis and mapping
    
    Returns:
        Dict with updated paths
    """
    from pathlib import Path
    
    # Convert everything to Path objects for easier manipulation
    orig_paths = {k: Path(v) if isinstance(v, str) else v for k, v in original_paths.items()}
    target_paths = {k: Path(v) if isinstance(v, str) else v for k, v in target_template_paths.items()}
    
    if verbose:
        print("🔍 Analyzing path differences...")
    
    # Find common base mappings between original and target
    base_mappings = {}
    
    # Look for key paths that define the base structure
    key_paths = ['cwd_dir', 'base_dir', 'data_dir']
    
    for key in key_paths:
        if key in orig_paths and key in target_paths:
            orig_base = orig_paths[key]
            target_base = target_paths[key]
            if verbose:
                print(f"   {key}: {orig_base} -> {target_base}")
            base_mappings[str(orig_base)] = str(target_base)
    
    # If we don't have direct mappings, try to infer them
    if not base_mappings and preserve_relative:
        # Try to find common patterns
        orig_str_paths = [str(p) for p in orig_paths.values() if isinstance(p, Path)]
        target_str_paths = [str(p) for p in target_paths.values() if isinstance(p, Path)]
        
        # Find the longest common prefixes that differ
        orig_prefixes = set()
        target_prefixes = set()
        
        for path_str in orig_str_paths:
            parts = Path(path_str).parts
            for i in range(1, len(parts)):
                orig_prefixes.add(str(Path(*parts[:i])))
        
        for path_str in target_str_paths:
            parts = Path(path_str).parts
            for i in range(1, len(parts)):
                target_prefixes.add(str(Path(*parts[:i])))
        
        # Find potential mappings
        common_suffixes = {}
        for orig_path in orig_str_paths:
            for target_path in target_str_paths:
                orig_parts = Path(orig_path).parts
                target_parts = Path(target_path).parts
                
                # Find common suffix
                min_len = min(len(orig_parts), len(target_parts))
                for i in range(1, min_len + 1):
                    if orig_parts[-i:] == target_parts[-i:]:
                        orig_base = str(Path(*orig_parts[:-i])) if len(orig_parts) > i else str(Path(orig_parts[0]))
                        target_base = str(Path(*target_parts[:-i])) if len(target_parts) > i else str(Path(target_parts[0]))
                        if orig_base != target_base:
                            base_mappings[orig_base] = target_base
                            break
    
    # Apply the base mappings
    updated_paths = {}
    for key, orig_path in orig_paths.items():
        if key == 'user':
            # Keep user as-is or update from target
            updated_paths[key] = target_paths.get(key, orig_path)
            continue
            
        orig_str = str(orig_path)
        updated_str = orig_str
        
        # Apply base mappings (longest match first)
        for orig_base, target_base in sorted(base_mappings.items(), key=len, reverse=True):
            if orig_str.startswith(orig_base):
                updated_str = orig_str.replace(orig_base, target_base, 1)
                if verbose:
                    print(f"   Mapping {key}: {orig_str} -> {updated_str}")
                break
        
        # If no mapping found, use the target template if available
        if updated_str == orig_str and key in target_paths:
            updated_str = str(target_paths[key])
            if verbose:
                print(f"   Direct replacement {key}: {orig_str} -> {updated_str}")
        
        updated_paths[key] = updated_str
    
    return updated_paths

def replace_paths_with_template(cfg, paths_template="glados", config_dir="../configs", verbose=True):
    """
    Replace all paths in the config with paths from a specified template YAML file.
    
    This function loads a paths template (e.g., glados.yaml) and uses it to replace
    the paths in the current config, while preserving experiment-specific information.
    
    Args:
        cfg: The configuration object to update
        paths_template: Name of the paths template (e.g., "glados", "hyak")
        config_dir: Directory containing the configs
        verbose: Whether to print detailed information about path replacements
    
    Returns:
        Updated configuration with new paths
    """
    from pathlib import Path
    from omegaconf import OmegaConf
    
    paths_file = Path(config_dir) / "paths" / f"{paths_template}.yaml"
    
    if not paths_file.exists():
        if verbose:
            print(f"❌ Paths template file not found: {paths_file}")
        return cfg
    
    if verbose:
        print(f"🔄 Loading paths template from: {paths_file}")
    
    # Load the paths template
    paths_template_cfg = OmegaConf.load(paths_file)
    
    # Store original experiment and version info from current config
    experiment_name = cfg.get('experiment', {}).get('name', 'unknown_experiment')
    version = cfg.get('version', 'debug')
    run_id = cfg.get('run_id', 'Testing')
    
    # Create a temporary config for path resolution
    temp_cfg = OmegaConf.create({
        'paths': paths_template_cfg,
        'experiment': {'name': experiment_name},
        'version': version,
        'run_id': run_id
    })
    
    # Resolve all interpolations in the paths
    try:
        OmegaConf.resolve(temp_cfg.paths)
        resolved_template_paths = temp_cfg.paths
        
        # Get original paths (resolve them too if possible)
        original_paths = cfg.get('paths', {})
        if original_paths:
            try:
                # Try to resolve original paths as well
                temp_orig = OmegaConf.create({
                    'paths': original_paths,
                    'experiment': {'name': experiment_name},
                    'version': cfg.get('version', version),
                    'run_id': cfg.get('run_id', run_id)
                })
                OmegaConf.resolve(temp_orig.paths)
                resolved_original_paths = temp_orig.paths
            except:
                resolved_original_paths = original_paths
                
            # Use smart path replacement
            updated_paths = smart_path_replacement(
                resolved_original_paths, 
                resolved_template_paths,
                verbose=verbose
            )
            cfg.paths = OmegaConf.create(updated_paths)
        else:
            cfg.paths = resolved_template_paths
        
        if verbose:
            print(f"✅ Successfully applied template '{paths_template}':")
            print(f"   Base dir: {cfg.paths.base_dir}")
            print(f"   Save dir: {cfg.paths.save_dir}")
            print(f"   Data dir: {cfg.paths.data_dir}")
        
    except Exception as e:
        print(f"❌ Error resolving paths template: {e}")
        print("Falling back to original paths...")
    
    return cfg


def load_config_with_path_template(config_path, paths_template=None, experiment=None, version=None, run_id=None, config_dir="../configs", verbose=False):
    """
    Load a config file and replace paths using a specified template.
    
    Args:
        config_path: Path to the config file to load
        paths_template: Name of the paths template (e.g., "glados", "hyak")
        experiment: Experiment name (if not in config)
        version: Version name (if not in config) 
        run_id: Run ID for path generation
        config_dir: Directory containing the configs
    
    Returns:
        Config with updated paths from template
    """
    
    print(f"📁 Loading config from: {config_path}")
    
    # Load the config
    cfg = OmegaConf.load(config_path)
    if paths_template is None:
        return cfg
    # Override experiment/version if provided
    if experiment:
        if 'experiment' not in cfg:
            cfg.experiment = {}
        cfg.experiment.name = experiment
    
    if version:
        cfg.version = version
        
    if run_id is not None:
        cfg.run_id = run_id
    
    # Replace paths using the specified template
    cfg = replace_paths_with_template(cfg, paths_template, config_dir, verbose=verbose)
    
    return cfg

def create_fresh_config_with_paths(experiment, paths_template="glados", version="debug", run_id="Testing", config_dir="../../configs", verbose=True):
    """
    Create a fresh config using Hydra with specified paths template.
    
    Args:
        experiment: Experiment name
        paths_template: Name of the paths template (e.g., "glados", "hyak")
        version: Version name
        run_id: Run ID for path generation
        config_dir: Directory containing the configs
        verbose: Whether to print detailed information
    
    Returns:
        Fresh config with paths from template
    """
    if verbose:
        print(f"🔄 Creating fresh config for experiment '{experiment}' with paths template '{paths_template}'")
    
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(
            config_name='config.yaml',
            overrides=[
                f"experiment={experiment}", 
                f"paths={paths_template}",
                f"version={version}", 
                f'run_id={run_id}'
            ], 
            return_hydra_config=True
        )
        HydraConfig.instance().set_config(cfg)
    
    if verbose:
        print(f"✅ Created fresh config:")
        print(f"   Experiment: {cfg.experiment.name}")
        print(f"   Version: {cfg.version}")
        print(f"   Paths template: {paths_template}")
        print(f"   Save dir: {cfg.paths.save_dir}")
    
    return cfg