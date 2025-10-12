#!/bin/sh

# Check if python3 is available, otherwise use python
if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Error: Neither python3 nor python found in PATH" >&2
  exit 1
fi

# Set the memory limit for the Python process (e.g., 90% of available RAM)
# Only on Linux systems with /proc/meminfo
if [ -f /proc/meminfo ] && command -v bc >/dev/null 2>&1; then
  MemTotal=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
  MemoryLimit=$((MemTotal * 90 / 100))
  ulimit -v "$MemoryLimit" 2>/dev/null || true
  echo "Total available memory: $(echo "scale=2; $MemTotal/1024/1024" | bc) GB"
elif [ -f /proc/meminfo ]; then
  echo "Warning: bc not found, skipping memory limit calculation"
  echo "Install bc with your package manager (e.g., apt install bc, yum install bc, pacman -S bc)"
fi

# Function to display usage instructions
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --exp <EXPERIMENT>          Specify the experiment name (default: HashGrid)"
  echo "  --trainable_cameras         Use trainable cameras"
  echo "  --scan_id <SCAN_ID>         Specify the scan ID (default: 114)"
  echo "  --is_continue               Continue training from the latest checkpoint"
  echo "  -h, --help                  Display this help message"
  exit 1
}

# Default values
EXPERIMENT="HashGrid"
TRAINABLE_CAMERAS="false"
SCAN_ID="114"
INCLUDE_IS_CONTINUE="false"

# Parse command line arguments
while [ $# -gt 0 ]; do
  case "$1" in
  --exp)
    if [ -z "$2" ] || [ "${2#--}" != "$2" ]; then
      echo "Error: --exp requires an argument" >&2
      exit 1
    fi
    EXPERIMENT="$2"
    shift 2
    ;;
  --trainable_cameras)
    TRAINABLE_CAMERAS="true"
    shift
    ;;
  --scan_id)
    if [ -z "$2" ] || [ "${2#--}" != "$2" ]; then
      echo "Error: --scan_id requires an argument" >&2
      exit 1
    fi
    SCAN_ID="$2"
    shift 2
    ;;
  --is_continue)
    INCLUDE_IS_CONTINUE="true"
    shift
    ;;
  -h | --help)
    usage
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage
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
"Permutohedral")
  CONFIG_DIR="./confs/embedder_conf_var/PermutohedralEncoder"
  ;;
*)
  echo "Invalid experiment name: $EXPERIMENT" >&2
  echo "Valid options: HashGrid, Posenc, FourierNTK, HashGridCUDA, NFFB, StylemodNFFB, HashGridTCNN, HashNerf, NFFB_TCNN, Permutohedral"
  exit 1
  ;;
esac

# If trainable cameras flag is set, change the config directory
if [ "$TRAINABLE_CAMERAS" = "true" ]; then
  CONFIG_DIR="$CONFIG_DIR/dtu_trained_cameras.conf"
else
  CONFIG_DIR="$CONFIG_DIR/dtu_fixed_cameras.conf"
fi

# Define the directory where the script is located (portable method)
# Get the script's directory
SCRIPT_PATH="$0"
# Handle symlinks if readlink is available
if command -v readlink >/dev/null 2>&1; then
  SCRIPT_PATH=$(readlink -f "$0" 2>/dev/null) || SCRIPT_PATH="$0"
fi
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)

# Change the directory to the parent directory of the script
cd "$SCRIPT_DIR/.." || {
  echo "Error: Cannot change to parent directory of script" >&2
  exit 1
}

# Verify the config file exists
if [ ! -f "$CONFIG_DIR" ]; then
  echo "Error: Configuration file not found: $CONFIG_DIR" >&2
  exit 1
fi

echo "Is continue: $INCLUDE_IS_CONTINUE"

while true; do
  echo "Working directory: $(pwd)"
  echo "Starting Neural Surface Reconstruction Experiment...$EXPERIMENT"
  echo "Config directory: $CONFIG_DIR"
  echo "Scan ID: $SCAN_ID"
  echo "Python command: $PYTHON_CMD"

  if [ "$INCLUDE_IS_CONTINUE" = "true" ]; then
    echo "Continue training from the latest checkpoint"
    $PYTHON_CMD -u ./training/exp_runner.py --conf "$CONFIG_DIR" --expname "$EXPERIMENT" --scan_id "$SCAN_ID" --checkpoint latest --validation_slope_print --is_continue
  else
    echo "Start training from scratch"
    $PYTHON_CMD -u ./training/exp_runner.py --conf "$CONFIG_DIR" --expname "$EXPERIMENT" --scan_id "$SCAN_ID" --checkpoint latest --validation_slope_print
  fi

  # Exit the loop based on the success or failure of the Python command
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Python script finished successfully, exiting loop."
    break
  else
    echo "Python script failed with exit code $EXIT_CODE, restarting..."
    INCLUDE_IS_CONTINUE="true" # Set the flag to true for the next iteration

    # Optional: Add a brief delay before restarting
    echo "Waiting 5 seconds before restart..."
    sleep 5
  fi
done

echo "Script execution completed."
