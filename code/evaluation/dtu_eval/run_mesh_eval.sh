#!/bin/bash

# Default values
exp_name="StylemodNFFB_TCNN"
scan_id=65
dataset_dir="../../../data/EVAL_DIR"
vis_out_dir="../../../evals/meshDTUEvalVisDir/"
mesh_type="stl"
onlySurfaces=false

# Function to display usage information
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --expname EXP_NAME          Specify the experiment name (default: $exp_name)"
  echo "  --scan_id SCAN_ID           Specify the scan ID (default: $scan_id)"
  echo "  --mesh_type MESH_TYPE       Specify the mesh type (default: $mesh_type)"
  echo "  --only_surfaces             Specify whether to evaluate only trimmed extracted surfaces from furu,tola,camp or all the sparse point clouds"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expname)
      exp_name="$2"
      shift 2
      ;;
    --scan_id)
      scan_id="$2"
      shift 2
      ;;
    --mesh_type)
      mesh_type="$2"
      shift 2
      ;;
    --only_surfaces)
      onlySurfaces=true
      shift
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Construct the Python command
python_command="python3 dtu_eval.py --data "../../../evals/dtu_fixed_cameras${exp_name}_${scan_id}/surface_world_coordinates_2000.ply" --scan $scan_id --mode mesh --dataset_dir $dataset_dir --vis_out_dir "${vis_out_dir}STLPointsEVAL/${exp_name}/${mesh_type}/${scan_id}/" --mesh_type $mesh_type"

# Append the --only_surfaces flag if $onlySurfaces is true
if [ "$onlySurfaces" = true ]; then
  python_command="python3 dtu_eval.py --data "../../../evals/dtu_fixed_cameras${exp_name}_${scan_id}/surface_world_coordinates_2000.ply" --scan $scan_id --mode mesh --dataset_dir $dataset_dir --vis_out_dir "${vis_out_dir}PoissonSurfacesEVAL/${exp_name}/${mesh_type}/${scan_id}/" --mesh_type $mesh_type --onlySurfaces"
fi

# Print the constructed Python command for diagnostic purposes
echo "$python_command"

# Run the Python command with the specified options
eval "$python_command"
