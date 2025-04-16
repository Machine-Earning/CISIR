#!/bin/bash
# kill_python_gpu_processes.sh: Kill all python (or python3) compute processes on given GPU IDs.
# Usage: ./kill_python_gpu_processes.sh 1 3 6

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <GPU_ID> [GPU_ID ...]"
    exit 1
fi

for GPU_ID in "$@"; do
    echo "Checking GPU $GPU_ID for python processes..."
    # Display current GPU status and processes
    nvidia-smi -i "$GPU_ID"
    
    # Retrieve process IDs where the process name is exactly 'python' or 'python3'
    PIDS=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name --format=csv,noheader | \
           awk -F',' '{
               # Remove leading/trailing whitespace from the process name field ($2)
               gsub(/^ +| +$/, "", $2);
               if($2 == "python" || $2 == "python3") print $1
           }')
    
    if [ -z "$PIDS" ]; then
        echo "No python processes found on GPU $GPU_ID."
    else
        echo "Found python processes on GPU $GPU_ID: $PIDS"
        for pid in $PIDS; do
            echo "Killing python process with PID $pid on GPU $GPU_ID..."
            kill -9 "$pid" 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "PID $pid killed."
            else
                echo "Failed to kill PID $pid. You might need root permissions."
            fi
        done
    fi
    echo "-------------------------------------"
done