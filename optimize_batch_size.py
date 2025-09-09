#!/usr/bin/env python3
"""
Batch size optimization using Hydra configuration system.

This script runs the actual simulation with different batch sizes to find optimal performance.
"""

import os
import time
import json
import psutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

class BatchSizeOptimizer:
    """Optimize batch size using actual Hydra runs."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results = []
        self.baseline_memory = psutil.virtual_memory().used / (1024**2)
        
    def run_batch_test(self, batch_size: int, n_replicates: int = 64) -> Dict:
        """Run a single batch size test."""
        print(f"\n{'='*50}")
        print(f"Testing batch_size = {batch_size}")
        print(f"{'='*50}")
        
        # Create unique run_id for this test
        run_id = f"batch_test_{batch_size}_{int(time.time())}"
        
        # Build command
        cmd = [
            sys.executable, "src/run_hydra.py",
            "experiment=DNb08_Stim",
            f"run_id={run_id}",
            f"experiment.n_replicates={n_replicates}",
            f"experiment.batch_size={batch_size}",
            "sim.async_mode=streaming",
            "experiment.save_checkpoints=False",
            "experiment.saveFigs=False",
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Monitor system resources
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)
        peak_memory = start_memory
        memory_samples = []
        
        try:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.base_dir,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor memory usage during execution
            output_lines = []
            while True:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    print(f"  {line.strip()}")
                
                # Sample memory
                current_memory = psutil.virtual_memory().used / (1024**2)
                memory_samples.append(current_memory)
                peak_memory = max(peak_memory, current_memory)
                
                # Check if process finished
                if process.poll() is not None:
                    break
            
            # Get remaining output
            remaining = process.stdout.read()
            if remaining:
                for line in remaining.strip().split('\n'):
                    if line:
                        output_lines.append(line)
                        print(f"  {line}")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Check success
            return_code = process.returncode
            success = return_code == 0
            
            # Calculate metrics
            memory_used = peak_memory - start_memory
            throughput = n_replicates / execution_time if execution_time > 0 else 0
            
            result = {
                "batch_size": batch_size,
                "success": success,
                "execution_time": execution_time,
                "throughput": throughput,
                "memory_peak_mb": memory_used,
                "return_code": return_code,
                "run_id": run_id
            }
            
            if success:
                print(f"✅ SUCCESS")
                print(f"   Time: {execution_time:.1f} seconds")
                print(f"   Throughput: {throughput:.2f} simulations/second")
                print(f"   Peak memory: {memory_used:.1f} MB")
            else:
                print(f"❌ FAILED (return code: {return_code})")
                # Include last few lines of output for debugging
                result["error_output"] = output_lines[-10:] if output_lines else []
            
            return result
            
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            return {
                "batch_size": batch_size,
                "success": False,
                "execution_time": 0,
                "throughput": 0,
                "memory_peak_mb": 0,
                "return_code": -1,
                "run_id": run_id,
                "exception": str(e)
            }
    
    def run_optimization(
        self, 
        batch_sizes: List[int], 
        n_replicates: int = 64,
        stop_on_failures: int = 2
    ) -> List[Dict]:
        """Run optimization across multiple batch sizes."""
        print(f"\n{'='*60}")
        print("BATCH SIZE OPTIMIZATION FOR DNb08_Stim")
        print(f"{'='*60}")
        print(f"Testing batch sizes: {batch_sizes}")
        print(f"Replicates per test: {n_replicates}")
        print(f"Stop after {stop_on_failures} consecutive failures")
        
        self.results = []
        consecutive_failures = 0
        
        for batch_size in sorted(batch_sizes):
            result = self.run_batch_test(batch_size, n_replicates)
            self.results.append(result)
            
            if result["success"]:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= stop_on_failures:
                    print(f"\n⚠️  Stopping after {consecutive_failures} consecutive failures")
                    print("Higher batch sizes likely to fail as well.")
                    break
            
            # Clean up output directory to save space
            try:
                run_dir = self.base_dir / f"run_id={result['run_id']}"
                if run_dir.exists():
                    import shutil
                    shutil.rmtree(run_dir)
                    print(f"  Cleaned up: {run_dir}")
            except Exception as e:
                print(f"  Warning: Could not clean up {run_dir}: {e}")
            
            # Brief pause between tests
            time.sleep(2)
        
        return self.results
    
    def analyze_results(self) -> Dict:
        """Analyze results to find optimal configurations."""
        successful = [r for r in self.results if r["success"]]
        
        if not successful:
            return {"error": "No successful tests"}
        
        # Find optimal configurations
        best_throughput = max(successful, key=lambda x: x["throughput"])
        fastest_time = min(successful, key=lambda x: x["execution_time"])
        most_memory_efficient = min(successful, key=lambda x: x["memory_peak_mb"] / x["throughput"] if x["throughput"] > 0 else float('inf'))
        
        analysis = {
            "total_tests": len(self.results),
            "successful_tests": len(successful),
            "failed_tests": len(self.results) - len(successful),
            "recommendations": {
                "best_throughput": {
                    "batch_size": best_throughput["batch_size"],
                    "throughput": best_throughput["throughput"],
                    "execution_time": best_throughput["execution_time"],
                    "memory_mb": best_throughput["memory_peak_mb"]
                },
                "fastest_execution": {
                    "batch_size": fastest_time["batch_size"],
                    "execution_time": fastest_time["execution_time"],
                    "throughput": fastest_time["throughput"],
                    "memory_mb": fastest_time["memory_peak_mb"]
                },
                "most_memory_efficient": {
                    "batch_size": most_memory_efficient["batch_size"],
                    "efficiency_mb_per_sim_s": most_memory_efficient["memory_peak_mb"] / most_memory_efficient["throughput"],
                    "throughput": most_memory_efficient["throughput"],
                    "memory_mb": most_memory_efficient["memory_peak_mb"]
                }
            },
            "all_results": self.results
        }
        
        return analysis
    
    def print_summary(self):
        """Print optimization results summary."""
        analysis = self.analyze_results()
        
        if "error" in analysis:
            print(f"\n❌ {analysis['error']}")
            return
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nTest Summary:")
        print(f"  Total tests: {analysis['total_tests']}")
        print(f"  Successful: {analysis['successful_tests']}")
        print(f"  Failed: {analysis['failed_tests']}")
        
        recs = analysis["recommendations"]
        
        print(f"\n🏆 Best Throughput Configuration:")
        best = recs["best_throughput"]
        print(f"   batch_size = {best['batch_size']}")
        print(f"   {best['throughput']:.2f} simulations/second")
        print(f"   {best['execution_time']:.1f} seconds total")
        print(f"   {best['memory_mb']:.1f} MB memory usage")
        
        print(f"\n⚡ Fastest Execution:")
        fastest = recs["fastest_execution"]
        print(f"   batch_size = {fastest['batch_size']}")
        print(f"   {fastest['execution_time']:.1f} seconds")
        print(f"   {fastest['throughput']:.2f} simulations/second")
        
        print(f"\n💾 Most Memory Efficient:")
        efficient = recs["most_memory_efficient"]
        print(f"   batch_size = {efficient['batch_size']}")
        print(f"   {efficient['efficiency_mb_per_sim_s']:.2f} MB per (simulation/second)")
        print(f"   {efficient['throughput']:.2f} simulations/second")
        print(f"   {efficient['memory_mb']:.1f} MB total memory")
        
        print(f"\n📊 Detailed Results:")
        print(f"{'Batch':<6} {'Success':<8} {'Time(s)':<8} {'Sim/s':<8} {'Memory(MB)':<12} {'Efficiency':<12}")
        print("-" * 70)
        
        for result in self.results:
            status = "✅ Pass" if result["success"] else "❌ Fail"
            batch = result["batch_size"]
            time_str = f"{result['execution_time']:.1f}" if result["success"] else "N/A"
            throughput = f"{result['throughput']:.2f}" if result["success"] else "N/A"
            memory = f"{result['memory_peak_mb']:.1f}" if result["success"] else "N/A"
            efficiency = f"{result['memory_peak_mb']/result['throughput']:.2f}" if result["success"] and result["throughput"] > 0 else "N/A"
            
            print(f"{batch:<6} {status:<8} {time_str:<8} {throughput:<8} {memory:<12} {efficiency:<12}")
        
        print(f"\n💡 Final Recommendation:")
        print(f"   For DNb08_Stim with n_replicates=64:")
        print(f"   🚀 Use experiment.batch_size={best['batch_size']} for maximum throughput")
        print(f"   💾 Use experiment.batch_size={efficient['batch_size']} for memory efficiency")
    
    def save_results(self, filename: str):
        """Save results to JSON file."""
        analysis = self.analyze_results()
        analysis["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize batch size for DNb08_Stim")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                       default=[2, 4, 6, 8, 16, 32, 64],
                       help="Batch sizes to test")
    parser.add_argument("--replicates", type=int, default=64,
                       help="Number of replicates per test")
    parser.add_argument("--output", type=str, default="dnb08_batch_optimization.json",
                       help="Output JSON file")
    parser.add_argument("--max-failures", type=int, default=2,
                       help="Stop after N consecutive failures")
    
    args = parser.parse_args()
    
    # Get repository root
    repo_root = Path(__file__).parent
    
    # Create optimizer
    optimizer = BatchSizeOptimizer(repo_root)
    
    # Run optimization
    optimizer.run_optimization(
        batch_sizes=args.batch_sizes,
        n_replicates=args.replicates,
        stop_on_failures=args.max_failures
    )
    
    # Print and save results
    optimizer.print_summary()
    optimizer.save_results(args.output)


if __name__ == "__main__":
    main()
