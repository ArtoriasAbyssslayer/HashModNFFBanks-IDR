#!/bin/bash
cd ..
# Default values for arguments
expname=""
scan_id=""
train_cameras=""
is_continue=""
use_multithreading=""
config_paths=()  # Array to store experiment names and their corresponding config paths

# Define experiment name to config path mappings
config_paths["PositionalEncoding"]="./confs/dtu_fixed_camers.conf"
config_paths["FourierFeatures"]="./confs/embedder_conf_var/FourierFeatures"
config_paths["HashGrid"]="./confs/embedder_conf_var/HashGrid_3DPoints-Posenc_ViewDirs"
config_paths["HashGridViewdirs"]="./confs/embedder_conf_var/HashGrid_3DPoints&ViewDirs"
config_paths["NFFB"]="./confs/embedder_conf_var/FFB"
config_paths["StyleModNFFB"]="./confs/embedder_conf_var/FFB_StyleMod"
# Add more mappings here

# Set default options
use_cuda_blocking="CUDA_LAUNCH_BLOCKING=0"
use_cuda_dsa="TORCH_USE_CUDA_DSA=1"
use_multithreading="OMP_NUM_THREADS=$(nproc)"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --expname)
            expname="$2"
            shift 2
            ;;
        --scan_id)
            scan_id="$2"
            shift 2
            ;;
        --train_cameras)
            train_cameras="--train_cameras"
            shift
            ;;
        --is_continue)
            is_continue="--is_continue"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$expname" ] || [ -z "$scan_id" ]; then
    echo "Usage: $0 --expname <expname> --scan_id <scan_id> [--train_cameras] [--is_continue]"
    exit 1
fi

# Get the config path based on the experiment name
conf_path="${config_paths[$expname]}"
if [ -z "$conf_path" ]; then
    echo "Unsupported experiment name: $expname"
    exit 1
fi

# Determine if cameras are fixed or trainable based on the configuration path
if [[ "$conf_path" == *"fixed_cameras"* ]]; then
    train_cameras=""
fi

# Construct the Python command
python_command="python3 -u ./training/exp_runner.py --conf $conf_path/${expname,,}_${scan_id}.conf --expname $expname --scan_id $scan_id $train_cameras --checkpoint latest $is_continue --validation_slope_print"

# Run the command
eval "$use_cuda_blocking $use_cuda_dsa $use_multithreading $python_command"
