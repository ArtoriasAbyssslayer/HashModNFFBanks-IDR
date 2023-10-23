#!/bin/bash

scan_ids=(122 110 114 65)
expnames=("Posenc" "HashGrid" "NFFB" "StylemodNFFB" "HashGridTCNN" "StylemodNFFB_TCNN")
mesh_types=("stl" "tola" "furu" "camp")
extra_expnames=("FourierNTK")

for scan_id in "${scan_ids[@]}"; do
  for expname in "${expnames[@]}"; do
    if [[ "$expname" == "HashGridTCNN" && ($scan_id -ne 122 && $scan_id -ne 110) ]]; then
      continue  # Skip HashGridTCNN for scan_ids other than 122 and 110
    fi
    for mesh_type in "${mesh_types[@]}"; do
      ./run_mesh_eval.sh --scan_id "$scan_id" --expname "$expname" --mesh_type "$mesh_type" --only_surfaces
    done
  done
  for expname in "${extra_expnames[@]}"; do
    for mesh_type in "${mesh_types[@]}"; do
      ./run_mesh_eval.sh --scan_id "$scan_id" --expname "$expname" --mesh_type "$mesh_type" --only_surfaces
    done
  done
done
