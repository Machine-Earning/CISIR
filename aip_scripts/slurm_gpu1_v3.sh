#!/bin/bash

#SBATCH --job-name=TestJob              # Job name
#SBATCH --nodes=4                       # Number of nodes gpu1, gpu2, gpu3, gpu4 (max 4 for partition gpu2). if using two nodes then gpu[01-02]
#SBATCH --ntasks=4                      # Number of tasks needs to match number of nodes?
#SBATCH --mem=32GB                      # Memory per node
#SBATCH --time=infinite                 # Time limit
#SBATCH --partition=gpu1                # Partition, dynamically set
#SBATCH --gres=gpu:4                    # Number of GPUs per node (max 4)
#SBATCH --output=testjob.%J.out         # Output file
#SBATCH --error=testjob.%J.err          # Error file

echo "Starting at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

# Run the Python module specified as the first argument
srun python -m $1

# how to use this script:
# sbatch your_slurm_script.sh sources.mlp_pds
