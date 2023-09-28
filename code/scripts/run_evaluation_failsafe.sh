#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --exp <EXPERIMENT>          Specify the experiment name (default: HashGrid)"
    echo "  --trainable_cameras         Use trainable cameras"
    echo "  --scan_id <SCAN_ID>         Specify the scan ID (default: 114)"
    echo "  --eval_rendering            Enable rendering evaluation"
    echo "  -h                          Display this help message"
    exit 1
}

# Default values
EXPERIMENT="HashGrid"
TRAINABLE_CAMERAS=false
SCAN_ID=114
EVAL_RENDERING=false

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
        --eval_rendering)
            EVAL_RENDERING=true
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

# Define the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change the directory to the parent directory of the script
cd "$SCRIPT_DIR/.."

# Set the experiment name and config directory based on the provided experiment
case "$EXPERIMENT" in
    "HashGrid")
        CONFIG_DIR="./confs/embedder_conf_var/MultiResHashPointsAndViewDirs"
        ;;
    "Posenc")
        CONFIG_DIR="./confs/embedder_conf_var/PosEnc"
        ;;
    "FourierNTK")
        CONFIG_DIR="./code/confs/embedder_conf_var/FourierFeatures"
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

# Run the evaluation command in a loop
while true; do
    echo "Working directory: $(pwd)"
    echo "Starting Neural Surface Reconstruction Evaluation...$EXPERIMENT"
    echo "Config directory: $CONFIG_DIR"
    echo "Scan ID: $SCAN_ID"
    if [ "$EVAL_RENDERING" = true ]; then
        echo "Rendering evaluation enabled"
        python3 -u ./evaluation/eval.py --expname "$EXPERIMENT" --conf "$CONFIG_DIR" --scan_id "$SCAN_ID" --checkpoint latest --eval_rendering
    else
        echo "Rendering evaluation disabled"
        python3 -u ./evaluation/eval.py --expname "$EXPERIMENT" --conf "$CONFIG_DIR" --scan_id "$SCAN_ID" --checkpoint latest
    fi
    # Exit the loop based on the success or failure of the Python command
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Python script finished successfully, exiting loop."
        break
    else
        echo "Python script failed with exit code $EXIT_CODE, restarting..."
    fi
done
