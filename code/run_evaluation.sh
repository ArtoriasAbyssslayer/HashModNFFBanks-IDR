#!/bin/bash

# Default values for arguments
exps_folder="exps"
expname=""
scan_id=""
train_cameras=""
checkpoint=""
eval_rendering=""
use_cuda_blocking=""
use_cuda_dsa=""

# Set default options
use_cuda_blocking="CUDA_LAUNCH_BLOCKING=1"
use_cuda_dsa="TORCH_USE_CUDA_DSA=1"

# Experiment name to config path mapping
config_paths=(
    [PositionalEncoding]="./confs"
    [FourierFeatures]="./confs/embedder_conf_var/FourierFeatures"
    [HashGrid]="./confs/embedder_conf_var/HashGrid"
    [HashGridCUDA]="./confs/embedder_conf_var/HashGrid"
    [NFFB]="./confs/embedder_conf_var/FFB"
    [NFFB_TCNN]="./confs/embedder_conf_var/FFB_TCNN"
    # Add more mappings here
)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --exps_folder)
            exps_folder="$2"
            shift 2
            ;;
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
        --checkpoint)
            checkpoint="$2"
            shift 2
            ;;
        --eval_rendering)
            eval_rendering="--eval_rendering"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Set CUDA environment variables if requested
cuda_env=""
if [ -n "$use_cuda_blocking" ] || [ -n "$use_cuda_dsa" ]; then
    cuda_env="export $use_cuda_blocking $use_cuda_dsa"
fi

# Validate required arguments
if [ -z "$expname" ] || [ -z "$scan_id" ] || [ -z "$checkpoint" ]; then
    echo "Usage: $0 --exps_folder <exps_folder> --expname <expname> --scan_id <scan_id> [--train_cameras] --checkpoint <checkpoint> [--eval_rendering]"
    exit 1
fi

# Get the config path based on the experiment name
conf_path="${config_paths[$expname]}"
if [ -z "$conf_path" ]; then
    echo "Unsupported experiment name: $expname"
    exit 1
fi

# Determine config file based on the camera type
if [ "$expname" == "PositionalEncoding" ]; then
    conf_path="$conf_path/dtu_fixed_cameras.conf"
elif [[ -n "$train_cameras" ]]; then
    conf_path="$conf_path/dtu_trained_cameras.conf"
else
    conf_path="$conf_path/${expname,,}_dtu_fixed_cameras.conf"
fi

# Construct the Python command
python_command="python3 -u ./evaluation/eval.py --exps_folder $exps_folder --expname $expname --conf $conf_path --scan_id $scan_id $train_cameras --checkpoint $checkpoint $eval_rendering"

# Run the command
eval "$cuda_env $python_command"

