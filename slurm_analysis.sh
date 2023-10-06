#!/bin/bash

#SBATCH --job-name=TestJob            # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=16GB                    # Memory per node
#SBATCH --time=99:99:99               # Time limit (4h15 minutes)
#SBATCH --partition=gpu1              # GPU partition
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --output=testjob.%J.out       # Output file
#SBATCH --error=testjob.%J.err        # Error file

echo "Starting at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."

echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

srun python3 analysis_pds.py # Run your Python script
#srun python3 analysis_rrt.py # Run your Python script
#srun python3 analysis_regnn.py # Run your Python script