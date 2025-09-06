import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
import jax
import signal
import sys
import gc

# Configure JAX for better performance
jax.config.update("jax_enable_x64", False)  # Use float32 for better GPU performance
print('Gpu devices:', len(jax.devices('gpu')) if jax.devices('gpu') else len(jax.devices('cpu')))

import hydra
import sparse
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from src.utils import io_dict_to_hdf5 as ioh5
from src.utils.path_utils import convert_dict_to_path, save_config
from src.simulation.vnc_sim import run_vnc_simulation, prepare_neuron_params, prepare_sim_params, parse_simulation_config, load_wTable

# Set up logging to capture all output
def setup_logging():
    """Set up logging with file handler only (no console duplication)."""
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Don't add console handler here - we'll handle console output separately
    return logger

class CleanLoggingRedirect:
    """Context manager to redirect print statements to log file only."""
    def __init__(self, logger):
        self.logger = logger
        self.terminal = sys.stdout
        
    def write(self, message):
        # Write to terminal as normal (clean output)
        self.terminal.write(message)
        self.terminal.flush()
        # Log the message to file only (with timestamp)
        if message.strip():
            self.logger.info(message.strip())
    
    def flush(self):
        self.terminal.flush()
        
    def __enter__(self):
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal

def cleanup_jax():
    """Clean up JAX resources properly."""
    print("\nCleaning up JAX resources...")
    try:
        # Force synchronization of all pending operations
        jax.block_until_ready(jax.numpy.array(1.0))
        
        # Clear JAX caches
        jax.clear_caches()
        
        # Force garbage collection
        gc.collect()
        
        print("JAX cleanup completed.")
    except Exception as e:
        print(f"Error during JAX cleanup: {e}")

def cleanup_logging(logger):
    """Clean up logging handlers properly."""
    try:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
    except Exception as e:
        print(f"Error during logging cleanup: {e}")

def signal_handler(signum, frame):
    """Handle keyboard interrupt gracefully."""
    print(f"\nReceived signal {signum}. Cleaning up...")
    cleanup_jax()
    # Clean up logging if possible
    try:
        logger = logging.getLogger()
        cleanup_logging(logger)
    except:
        pass
    print("Cleanup completed. Exiting...")
    sys.exit(1)

# Set up signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function to run the VNC simulation with the provided configuration."""
    # Set up logging first
    logger = setup_logging()
    
    try:
        # Get the hydra output directory and set up file logging
        hydra_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        log_file = hydra_output_dir / "run_hydra.log"
        
        # Create file handler for logging to file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log initial information to file only
        logger.info("=== VNC Simulation Run Started ===")
        logger.info(f"Logging to file: {log_file}")
        logger.info(f"GPU devices: {len(jax.devices('gpu')) if jax.devices('gpu') else len(jax.devices('cpu'))}")
        
        # Show clean startup info on console
        print(f"Logging to: {log_file}")
        print("=" * 50)
        
        # Use context manager to redirect print statements to log file (clean console output)
        with CleanLoggingRedirect(logger):
            if (("load_jobid" in cfg) and (cfg["load_jobid"] is not None) and (cfg["load_jobid"] != "")):
                run_id = cfg.load_jobid
                load_cfg_path = (
                    Path(cfg.paths.base_dir) / f"run_id={run_id}/logs/run_config.yaml"
                )
                cfg = OmegaConf.load(load_cfg_path)
                
            # Ensure paths are set correctly
            cfg.paths = convert_dict_to_path(cfg.paths)

            ##### Save config file to the log directory #####
            config_path = cfg.paths.log_dir / "run_config.yaml"
            if not config_path.exists():
                save_config(cfg, config_path)
                print(f"Config saved to {config_path}")
            cfg.paths = convert_dict_to_path(cfg.paths)

            ##### Run the simulation #####
            print("Running VNC simulation with the following configuration:")
            
            # Check for async execution mode
            async_mode = getattr(cfg.sim, "async_mode", "sync")
            prune_network = getattr(cfg.sim, "prune_network", False)
            print(f"Execution mode: {async_mode}, Pruning: {prune_network}")
            
            if async_mode == "sync":
                # Use original synchronous implementation
                results, final_mini_circuits, neuron_params = run_vnc_simulation(cfg)
            else:
                # Use asynchronous implementation
                print(f"Using async mode: {async_mode}")
                
                # Import async functions
                try:
                    from src.simulation.async_vnc_sim import (
                        run_streaming_pruning_simulation,
                        run_streaming_regular_simulation
                    )
                    from src.simulation.vnc_sim import prepare_vnc_simulation_params
                except ImportError as e:
                    print(f"Warning: Could not import async modules: {e}")
                    print("Falling back to synchronous execution")
                    results, final_mini_circuits, neuron_params = run_vnc_simulation(cfg)
                else:
                    # Use the same data preparation as sync version
                    print("Loading network configuration...")
                    _, neuron_params, sim_params, sim_config, _ = prepare_vnc_simulation_params(cfg)
                    
                    # Initialize variables to handle error cases
                    results = None
                    final_mini_circuits = None
                    
                    # Run async simulation based on mode and pruning setting
                    try:
                        if prune_network:
                            # Async pruning simulations
                            if async_mode == "streaming":
                                results, final_mini_circuits, neuron_params = run_streaming_pruning_simulation(
                                    neuron_params, sim_params, sim_config, max_concurrent=cfg.experiment.batch_size
                                )
                            else:
                                raise ValueError(f"Unknown async_mode for pruning: {async_mode}. Valid options: 'sync', 'streaming'")
                        else:
                            # Async regular (non-pruning) simulations
                            if async_mode == "streaming":
                                results = run_streaming_regular_simulation(
                                    neuron_params, sim_params, sim_config, max_concurrent=cfg.experiment.batch_size
                                )
                                final_mini_circuits = None  # Regular sims don't produce mini circuits
                            else:
                                raise ValueError(f"Unknown async_mode for regular simulations: {async_mode}. Valid options: 'sync', 'streaming'")
                        
                        print(f"Async simulation completed successfully")
                    except Exception as async_error:
                        print(f"Async simulation failed: {async_error}")
                        # Fallback to sync if needed
                        print("Falling back to synchronous execution")
                        results, final_mini_circuits, neuron_params = run_vnc_simulation(cfg)
                        # Note: Could also re-raise here if you prefer to fail fast
                        # raise
            
            # Ensure all operations are completed before proceeding
            jax.block_until_ready(results)
            if final_mini_circuits is not None:
                jax.block_until_ready(final_mini_circuits)
            
            ##### Save results #####
            print('Saving results to:', cfg.paths.ckpt_dir)
            if cfg.experiment.dn_screen==False:
                sparse.save_npz(cfg.paths.ckpt_dir / f"{cfg.experiment.name}_Rs.npz", sparse.COO.from_numpy(results))
            ioh5.save(cfg.paths.ckpt_dir / 'neuron_params.h5', neuron_params._asdict())
            if final_mini_circuits is not None:
                sparse.save_npz(cfg.paths.ckpt_dir / f"{cfg.experiment.name}_mini_circuits.npz", sparse.COO.from_numpy(final_mini_circuits))


            print('Done! :)')
            
    except KeyboardInterrupt:
        logger.error("Keyboard interrupt received. Cleaning up...")
        print("\nSimulation interrupted. Checkpoints (if any) have been preserved for resuming.")
        cleanup_jax()
        cleanup_logging(logger)
        logger.info("Cleanup completed. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        print("Cleaning up JAX resources...")
        cleanup_jax()
        cleanup_logging(logger)
        raise
    finally:
        # Always try to clean up on exit
        try:
            cleanup_jax()
            cleanup_logging(logger)
            logger.info("=== VNC Simulation Run Completed ===")
        except:
            pass  # Ignore cleanup errors during normal exit
    
if __name__ == "__main__":
    main()
