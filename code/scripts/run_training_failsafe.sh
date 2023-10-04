#!/bin/bash

# Set the memory limit for the Python process (e.g., 90% of available RAM)
ulimit -v $(($(awk '/MemTotal/ {print $2}' /proc/meminfo) * 99 / 100))
echo "Total available memory: $(($(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024))GB"
# Function to display usage instructions
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --exp <EXPERIMENT>          Specify the experiment name (default: HashGrid)"
    echo "  --trainable_cameras         Use trainable cameras"
    echo "  --scan_id <SCAN_ID>         Specify the scan ID (default: 114)"
    echo "  --is_continue               Continue training from the latest checkpoint"
    echo "  -h                          Display this help message"
    exit 1
}

# Default values
EXPERIMENT="HashGrid"
TRAINABLE_CAMERAS=false
SCAN_ID=114
INCLUDE_IS_CONTINUE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)
            EXPERIMENT="$2"
            shift 2
            ;;
        --trainable_cameras)
            TRAINABLE_CAMERAS=true
            shift
            ;;
        --scan_id)
            SCAN_ID="$2"
            shift 2
            ;;
        --is_continue)
            INCLUDE_IS_CONTINUE=true
            shift
            ;;
        -h)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Set the experiment name and config directory based on the provided experiment
case "$EXPERIMENT" in
    "HashGrid")
        CONFIG_DIR="./confs/embedder_conf_var/MultiResHashPointsAndViewDirs"
        ;;
    "Posenc")
        CONFIG_DIR="./confs/embedder_conf_var/PosEnc"
        ;;
    "FourierNTK")
        CONFIG_DIR="./confs/embedder_conf_var/FourierFeatures"
        ;;
    "HashGridCUDA")
        CONFIG_DIR="./confs/embedder_conf_var/CUDA_HashGrid"
        ;;
    "NFFB")
        CONFIG_DIR="./confs/embedder_conf_var/FFB"
        ;;
    "StylemodNFFB")
        CONFIG_DIR="./confs/embedder_conf_var/FFB_StyleMod"
        ;;
    "HashGridTCNN")
        CONFIG_DIR="./confs/embedder_conf_var/HashGrid_TCNN_PointsAndViewDirs"
        ;;
    "HashNerf")
        CONFIG_DIR="./confs/embedder_conf_var/MultiResHashPointsPosencViews"
        ;;
    "NFFB_TCNN")
        CONFIG_DIR="./confs/embedder_conf_var/FFB_TCNN"
        ;;
    *)
        echo "Invalid experiment name: $EXPERIMENT" >&2
        exit 1
        ;;
esac

# If trainable cameras flag is set, change the config directory
if [ "$TRAINABLE_CAMERAS" = true ]; then
    CONFIG_DIR="$CONFIG_DIR/dtu_trained_cameras.conf"
else
    CONFIG_DIR="$CONFIG_DIR/dtu_fixed_cameras.conf"
fi

# Define the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change the directory to the parent directory of the script
cd "$SCRIPT_DIR/.."
echo "Is continue: $INCLUDE_IS_CONTINUE"
while true; do
    echo "Working directory: $(pwd)"
    echo "Starting Neural Surface Reconstruction Experiment...$EXPERIMENT" 
    echo "Config directory: $CONFIG_DIR"
    echo "Scan ID: $SCAN_ID"
    if [ "$INCLUDE_IS_CONTINUE" = true ]; then
        echo "Continue training from the latest checkpoint"
        python3 -u ./training/exp_runner.py --conf "$CONFIG_DIR" --expname "$EXPERIMENT" --scan_id "$SCAN_ID" --checkpoint latest --validation_slope_print --is_continue
    else
        echo "Start training from scratch"
        python3 -u ./training/exp_runner.py --conf "$CONFIG_DIR" --expname "$EXPERIMENT" --scan_id "$SCAN_ID" --checkpoint latest --validation_slope_print
    fi
    # Exit the loop based on the success or failure of the Python command
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Python script finished successfully, exiting loop."
        break
    else
        echo "Python script failed with exit code $EXIT_CODE, restarting..."
        INCLUDE_IS_CONTINUE=true  # Set the flag to true for the next iteration
    fi
done
