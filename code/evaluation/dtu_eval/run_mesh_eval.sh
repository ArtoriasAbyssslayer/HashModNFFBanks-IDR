#!/bin/bash

# Default values
expname=""
scan=""
dataset_dir="../../../data/EVAL_DIR"
vis_out_dir="../../../evals/meshDTUEvalVisDir"

# Function to display help
function display_help {
    echo "Usage: $0 -e|--expname EXPNAME -s|--scan SCAN [-d|--dataset_dir DATASET_DIR] [-v|--vis_out_dir VIS_OUT_DIR] [-h|--help]"
    echo "Options:"
    echo "  -e, --expname    Experiment name (NFFB, StylemodNFFB, etc.)"
    echo "  -s, --scan       Scan ID"
    echo "  -d, --dataset_dir Dataset directory (default: $dataset_dir)"
    echo "  -v, --vis_out_dir Visualization output directory (default: $vis_out_dir)"
    echo "  -h, --help       Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--expname)
            expname="$2"
            shift 2
            ;;
        -s|--scan)
            scan="$2"
            shift 2
            ;;
        -d|--dataset_dir)
            dataset_dir="$2"
            shift 2
            ;;
        -v|--vis_out_dir)
            vis_out_dir="$2"
            shift 2
            ;;
        -h|--help)
            display_help
            ;;
        *)
            echo "Invalid option: $1"
            display_help
            ;;
    esac
done

# Check if expname and scan are provided
if [[ -z "$expname" ]] || [[ -z "$scan" ]]; then
    echo "Error: Both --expname and --scan must be provided."
    display_help
fi

# Generate the command based on provided arguments
data_path="../../../evals/dtu_fixed_cameras${expname}_${scan}/surface_world_coordinates_2000.ply"
command="python3 dtu_eval.py --data $data_path --scan $scan --mode mesh --dataset_dir $dataset_dir --vis_out_dir $vis_out_dir"

# Print the generated command
echo "Generated command:"
echo "$command"

# Run the command
eval "$command"
