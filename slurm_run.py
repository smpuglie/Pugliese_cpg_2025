import argparse
import subprocess
import sys

def slurm_submit(script):
    """
    Submit the SLURM script using sbatch and return the job ID.
    """
    try:
        # Use a list for the command and pass the script via stdin
        output = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)

def submit(gpus, partition, job_name, mem, cpus, time, note, experiment, sim, mode, load_jobid, gpu_type, override):
    """
    Construct and submit the SLURM script with the specified parameters.
    """
        # Define GPU configurations
    gpu_configs = {
        'a100': 'gpu:a100:8',
        'h100': 'nvidia_h100_80gb_hbm3',
        'a40': 'gpu:a40:8',
        'l40': 'gpu:l40:8', 
        'l40s': 'gpu:l40s:8', 
        # Add more GPU types here if needed
    }

    gpu_resource = f"gpu:{gpu_configs[gpu_type]}:{gpus}"

    """Submit job to cluster."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}    
#SBATCH --partition={partition}
#SBATCH --account=portia
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gpus={gpus}
#SBATCH --mem={mem}G
#SBATCH --verbose  
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=smpuglie@uw.edu
#SBATCH --exclude=g[3001-3007,3010-3017,3020-3027,3030-3037,3114],z[3001,3002,3005,3006]
module load cuda/12.6.3
set -x
source ~/.bashrc
nvidia-smi
conda activate bdn2cpg
unset LD_LIBRARY_PATH
echo $SLURMD_NODENAME
python -u ./src/run_hydra.py hydra.mode={mode} paths=hyak-smpuglie note={note} version=hyak experiment={experiment} sim={sim} load_jobid={load_jobid} run_id=$SLURM_JOB_ID {override}
            """
    print(f"Submitting job")
    print(script)
    job_id = slurm_submit(script)
    print(job_id)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Submit a SLURM job with specified GPU type.')
    parser.add_argument('--gpus', type=int, default=8,
                        help='Number of GPUs to request (default: 8)')
    parser.add_argument('--gpu_type', type=str, default='l40s',
                        help='Number of GPUs to request (default: 8)')
    parser.add_argument('--job_name', type=str, default='bdn2cpg',
                        help='Name of the SLURM job (default: rodent)')
    parser.add_argument('--mem', type=int, default=512,
                        help='Memory in GB (default: 512)')
    parser.add_argument('--cpus', type=int, default=64,
                        help='Number of CPU cores (default: 64)')
    parser.add_argument('--time', type=str, default='2-00:00:00',
                        help='Time limit for the job day-hr-min-sec (default: 2-00:00:00)')
    parser.add_argument('--partition', type=str, default='gpu-l40s',
                        help='Partition to run job (default: gpu-l40s)')
    parser.add_argument('--note', type=str, default='hyak_ckpt',
                        help='Note for job (default: hyak_ckpt)')
    parser.add_argument('--experiment', type=str, default='stim_neurons',
                        help='Name of experiment yaml  (default: stim_neurons)')
    parser.add_argument('--sim', type=str, default='default',
                        help='Name of simulation yaml  (default: default)')
    parser.add_argument('--mode', type=str, default="RUN",
                        help='Mode for the simulation (default: RUN)')
    parser.add_argument('--load_jobid', type=str, default='',
                        help='JobID to resume training (default: '')')
    parser.add_argument('--override', type=str, default='',
                        help='JobID to resume training (default: '')')

    args = parser.parse_args()

    submit(
        gpus=args.gpus,
        job_name=args.job_name,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        partition=args.partition,
        note=args.note,
        experiment=args.experiment,
        sim=args.sim,
        mode=args.mode,
        load_jobid=args.load_jobid,
        gpu_type=args.gpu_type,
        override=args.override,
    )

if __name__ == "__main__":
    main()
    
##### Saving commands #####
#### cancel all jobs: squeue -u $USER -h | awk '{print $1}' | xargs scancel


## exclude nodes g3090,g3107,g3097,g3109,g3113,g3091,g3096
