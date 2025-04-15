#!/bin/bash
#
# Usage: ./run.sh -t=asc -s=sources.mlp.neurips.allstate_claims.cheat_ours
# This script will run "python -m <module>" using nohup and redirect output
# to a log file named <type>.<datetime_code>.log (e.g., asc.20250415_093045.log)

# Function to display usage message
usage() {
    echo "Usage: $0 -t=<type> -s=<module>"
    exit 1
}

# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        -t=*|--type=*)
            TYPE="${arg#*=}"
            ;;
        -s=*|--source=*)
            MODULE="${arg#*=}"
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            ;;
    esac
done

# Check that both variables are set
if [ -z "$TYPE" ] || [ -z "$MODULE" ]; then
    echo "Error: Both -t and -s options must be provided."
    usage
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Create a datetime code (format: YYYYMMDD_HHMMSS)
DATETIME_CODE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/${TYPE}.${DATETIME_CODE}.log"

# Inform the user
echo "Starting job: python -m ${MODULE}"
echo "Logging output to ${LOG_FILE}"

# Run the job with nohup in the background, capturing stdout and stderr
nohup python -m "${MODULE}" > "${LOG_FILE}" 2>&1 &

# Optionally, you can capture the job PID if needed
echo "Job started with PID $!"

exit 0