#!/bin/bash

#SBATCH --job-name=inves            # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=32GB                    # Memory per node
#SBATCH --time=infinite              # Time limit
#SBATCH --partition=gpu2              # Partition, dynamically set
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --output=inves.%J.out       # Output file
#SBATCH --error=inves.%J.err        # Error file

echo "Starting at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

srun python3 $1 # Run your Python script with file name as argument
