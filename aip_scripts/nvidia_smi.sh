#!/bin/bash

#SBATCH --job-name=nvidia-smi-info      # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --mem=8GB                      # Memory per node (reduced, adjust if needed)
#SBATCH --time=00:10:00                # Time limit (10 minutes, adjust if needed)
#SBATCH --partition=gpu1               # Partition, dynamically set
#SBATCH --gres=gpu:1                   # Number of GPUs per node
#SBATCH --output=./logs/%x.%J.out      # Output file
#SBATCH --error=./logs/%x.%J.err       # Error file

echo "Starting at date $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is $(pwd)"

echo "Collecting NVIDIA GPU information with nvidia-smi:"
nvidia-smi

echo ""
echo "Collecting CUDA version information:"

# Try to get CUDA version from nvcc, if available
if command -v nvcc &> /dev/null; then
    echo "CUDA version from nvcc:"
    nvcc --version
else
    echo "nvcc not found."
fi

# Try to get CUDA version from /usr/local/cuda/version.txt, if available
if [ -f /usr/local/cuda/version.txt ]; then
    echo "CUDA version from /usr/local/cuda/version.txt:"
    cat /usr/local/cuda/version.txt
fi

# Try to get CUDA version from nvidia-smi output
echo "CUDA version from nvidia-smi:"
nvidia-smi | grep -i "CUDA Version"

# how to use this script:
# sbatch aip_scripts/nvidia_smi.sh