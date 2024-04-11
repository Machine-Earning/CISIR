#!/bin/bash

#SBATCH --job-name=sep.f            # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=128GB                    # Memory per node
#SBATCH --time=infinite              # Time limit
#SBATCH --partition=gpu1              # Partition, dynamically set
#SBATCH --gres=gpu:2                # Number of GPUs per node
#SBATCH --output=sep.f.%J.out       # Output file
#SBATCH --error=sep.f.%J.err        # Error file

echo "Starting at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

# Run the Python module specified as the first argument
srun python -m $1

# how to use this script:
# sbatch your_slurm_script.sh sources.mlp_pds
