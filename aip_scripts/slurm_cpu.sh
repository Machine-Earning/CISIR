#!/bin/bash

#SBATCH --job-name=TestJob            # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=32GB                    # Memory per node
#SBATCH --time=infinite              # Time limit
#SBATCH --partition=eternity              # Partition, dynamically set
#SBATCH --output=testjob.%J.out       # Output file
#SBATCH --error=testjob.%J.err        # Error file

echo "Starting at date $(date)"

echo "Running on hosts: $SLURM_NODELIST"

echo "Running on $SLURM_NNODES nodes."
 
echo "Running on $SLURM_NPROCS processors."

echo "Current working directory is $(pwd)"

srun python3 $1 # Run your Python script with file name as argument
