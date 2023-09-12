#!/bin/bash
cd ..
alias python=python3
CUDA_LAUNCH_BLOCKING=0 TORCH_USE_CUDA_DSA=1 python -u  ./training/exp_runner.py --conf ./confs/embedder_conf_var/MultiResHash/dtu_fixed_cameras.conf --expname HashGrid --scan_id 0 --checkpoint latest --validation_slope_print
